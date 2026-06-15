# ============================================================
# CSI-Lock-Sim | Phase 5: Secure Lockdown State Machine
# ============================================================
# This script is the end-to-end MVP of the CSI-Lock-Sim
# project. It wires together all components from Phases 1-4
# into a single working simulation:
#
#   VirtualCSISource   — streams CSI test samples one at a
#                        time, simulating live sensor input
#
#   VirtualBLEStack    — a two-state flag (dormant /
#                        advertising) that accepts or rejects
#                        simulated "Just Works" pairing
#                        requests — mirroring the vulnerable
#                        default BLE behaviour this project
#                        is designed to prevent
#
#   VirtualESP32       — loads the primary model from Phase 4,
#                        runs preprocessing → inference →
#                        BLE mode switch in a time-multiplexed
#                        loop (never sensing AND pairing
#                        simultaneously — addressing the
#                        supervisor's concurrency concern)
#
# Output:
#   - Per-sample decision log (terminal)
#   - Summary table by class (terminal)
#   - State machine timeline plot (results/state_timeline.png)
#
# Maps to: Chapter 1 RO4, Chapter 2 Section 2.3.4,
#          Chapter 5 Section 5.4.5
# ============================================================

import numpy as np
import pandas as pd
import os
import sys
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.signal import butter, filtfilt


# ============================================================
# CONFIGURATION
# ============================================================
RANDOM_SEED        = 42
SUBCARRIERS        = 30
TIME_STEPS         = 100
FEATURE_SIZE       = SUBCARRIERS * TIME_STEPS

# BLE pairing window: how many "ticks" the BLE radio stays
# in advertising mode after a gesture is detected before
# automatically returning to dormant/sensing mode
BLE_WINDOW_TICKS   = 1

# Paths
RF_MODEL_PATH      = os.path.join("models", "rf_model.joblib")
TFLITE_MODEL_PATH  = os.path.join("models", "cnn_quantised.tflite")
DATASET_PATH       = os.path.join("data", "csi_dataset.csv")
FEATURES_PATH      = os.path.join("data", "features_rf.csv")
CNN_DATA_PATH      = os.path.join("data", "normalised_cnn.npy")
LABELS_PATH        = os.path.join("data", "labels.npy")
PLOT_PATH          = os.path.join("results", "state_timeline.png")
LOG_PATH           = os.path.join("results", "simulation_log.csv")

np.random.seed(RANDOM_SEED)


# ============================================================
# SIGNAL PREPROCESSING (mirrors Phase 2 pipeline exactly)
# ============================================================

def apply_lowpass_filter(signal_1d, cutoff=10, fs=100, order=4):
    nyquist       = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a          = butter(order, normal_cutoff,
                           btype="low", analog=False)
    return filtfilt(b, a, signal_1d)


def extract_features(sample_1d):
    return np.array([
        float(np.sqrt(np.mean(sample_1d ** 2))),
        float(np.var(sample_1d)),
        float(np.max(sample_1d) - np.min(sample_1d)),
        float(np.mean(np.abs(np.diff(sample_1d)))),
    ])


# ============================================================
# COMPONENT 1 — VIRTUAL CSI SOURCE
# ============================================================

class VirtualCSISource:
    """
    Simulates a live CSI data stream from the ESP32's WiFi radio.
    In a real deployment, this would be raw CSI frames captured
    by the ESP32 antenna. Here, it streams pre-recorded samples
    from the test set one at a time.
    Yields: (raw_sample, true_label) tuples
    """

    def __init__(self, dataset_path, n_per_class=25):
        df       = pd.read_csv(dataset_path)
        classes  = df["label"].unique()
        selected = []
        for cls in classes:
            subset = df[df["label"] == cls].sample(
                n=min(n_per_class, len(df[df["label"] == cls])),
                random_state=RANDOM_SEED
            )
            selected.append(subset)

        self.df      = pd.concat(selected).sample(
            frac=1, random_state=RANDOM_SEED
        ).reset_index(drop=True)
        self.index   = 0
        self.total   = len(self.df)
        print(f"      CSI Source ready  : {self.total} samples "
              f"({n_per_class} per class)")

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= self.total:
            raise StopIteration
        row         = self.df.iloc[self.index]
        label       = row["label"]
        raw_sample  = row.drop("label").values.astype(np.float32)
        self.index += 1
        return raw_sample, label

    def __len__(self):
        return self.total


# ============================================================
# COMPONENT 2 — VIRTUAL BLE STACK
# ============================================================

class VirtualBLEStack:
    """
    Simulates the BLE radio on the ESP32.
    States:
      DORMANT     — radio is off; no device can discover
                    or pair with this device. This is the
                    "Secure Lockdown" state.
      ADVERTISING — radio is broadcasting; a mobile device
                    can now initiate "Just Works" pairing.
                    This window closes automatically after
                    BLE_WINDOW_TICKS simulation steps.
    """

    def __init__(self):
        self.state        = "DORMANT"
        self.ticks_left   = 0
        self.total_opens  = 0
        self.total_rejects = 0
        self.total_accepts = 0

    def open_pairing_window(self):
        """Called when the ESP32 detects a valid gesture."""
        self.state       = "ADVERTISING"
        self.ticks_left  = BLE_WINDOW_TICKS
        self.total_opens += 1

    def tick(self):
        """
        Called every simulation step.
        Counts down the pairing window and returns to
        DORMANT automatically when it expires.
        """
        if self.state == "ADVERTISING":
            self.ticks_left -= 1
            if self.ticks_left <= 0:
                self.state = "DORMANT"

    def handle_pairing_request(self):
        """
        Simulates an incoming BLE pairing request.
        Returns: (outcome_string, accepted_bool)
        """
        if self.state == "ADVERTISING":
            self.total_accepts += 1
            return "ACCEPTED — Just Works pairing", True
        else:
            self.total_rejects += 1
            return "REJECTED — device not discoverable", False


# ============================================================
# COMPONENT 3 — VIRTUAL ESP32
# ============================================================

class VirtualESP32:
    """
    Simulates the ESP32 microcontroller running the
    CSI-Lock-Sim firmware in a time-multiplexed loop:

      SENSING mode  → WiFi active, CSI captured, ML inference
                       runs, BLE stack is dormant
      PAIRING mode  → BLE advertising window open, CSI
                       sensing paused

    The device is NEVER doing both simultaneously — this
    directly addresses the supervisor's concurrency concern
    about the ESP32 crashing under simultaneous WiFi+BLE+ML
    load (see Chapter 3, Supervisor Feedback).
    """

    # Hardware reference (from Espressif datasheet)
    SRAM_BUDGET_KB   = 520
    WIFI_STACK_KB    = 120
    BLE_STACK_KB     = 80
    RUNTIME_KB       = 20
    AVAILABLE_KB     = 520 - 120 - 80 - 20   # = 300 KB

    def __init__(self, model_type, model_path):
        self.mode        = "SENSING"
        self.model_type  = model_type
        self.model_path  = model_path
        self._load_model()

    def _load_model(self):
        """Load either the TFLite CNN or the RF model."""
        if self.model_type == "tflite":
            import tensorflow as tf
            self.interpreter = tf.lite.Interpreter(
                model_path=self.model_path)
            self.interpreter.allocate_tensors()
            self.input_details  = \
                self.interpreter.get_input_details()
            self.output_details = \
                self.interpreter.get_output_details()
            print(f"      Model type        : "
                  f"DS-CNN (TFLite quantised)")
        else:
            self.rf_model = joblib.load(self.model_path)
            print(f"      Model type        : "
                  f"Random Forest")

        print(f"      Model loaded from : {self.model_path}")
        print(f"      SRAM budget       : "
              f"{self.SRAM_BUDGET_KB} KB total / "
              f"{self.AVAILABLE_KB} KB available for ML")

    def _preprocess(self, raw_sample):
        """
        Mirrors the Phase 2 preprocessing pipeline exactly.
        Returns both feature vector (RF) and 2D array (CNN).
        """
        filtered     = apply_lowpass_filter(raw_sample)
        features_rf  = extract_features(filtered)
        cnn_input    = filtered.reshape(
            SUBCARRIERS, TIME_STEPS
        ).astype(np.float32)

        # Use GLOBAL min/max saved during Phase 2 training
        # to ensure inference preprocessing matches training
        norm_path    = os.path.join("data", "norm_params.npy")
        norm_params  = np.load(norm_path)
        mn, mx       = norm_params[0], norm_params[1]
        cnn_norm     = (cnn_input - mn) / (mx - mn + 1e-8)
        return features_rf, cnn_norm

    def _predict(self, features_rf, cnn_norm):
        """
        Runs inference using the loaded model.
        Returns: (prediction_int, confidence_float)
        Class 0 = GESTURE (unlock trigger)
        Class 1,2,3 = NOT_GESTURE variants (stay locked)
        """
        if self.model_type == "tflite":
            import tensorflow as tf
            inp  = cnn_norm[np.newaxis, :, :, np.newaxis]
            self.interpreter.set_tensor(
                self.input_details[0]["index"],
                inp.astype(np.float32)
            )
            self.interpreter.invoke()
            output      = self.interpreter.get_tensor(
                self.output_details[0]["index"]
            )[0]
            pred_class  = int(np.argmax(output))
            confidence  = float(np.max(output))
        else:
            pred_class  = int(
                self.rf_model.predict([features_rf])[0]
            )
            proba       = self.rf_model.predict_proba(
                [features_rf]
            )[0]
            confidence  = float(np.max(proba))

        return pred_class, confidence

    def process(self, raw_sample, ble_stack):
        """
        Main firmware loop — one iteration per CSI sample.
        Time-multiplexed: always in exactly one mode.
        Returns: (prediction, confidence, ble_state, pairing_outcome)
        """
        # --- SENSING MODE ---
        self.mode = "SENSING"
        features_rf, cnn_norm = self._preprocess(raw_sample)
        pred_class, confidence = self._predict(
            features_rf, cnn_norm)

        # Class 0 = GESTURE → open pairing window
        is_gesture = (pred_class == 0)

        if is_gesture:
            ble_stack.open_pairing_window()
            # --- SWITCH TO PAIRING MODE ---
            self.mode = "PAIRING"

        # Simulate an incoming pairing request this tick
        pairing_outcome, accepted = \
            ble_stack.handle_pairing_request()

        # Capture state BEFORE tick so the plot reflects
        # the state that was active when the decision was made
        active_state = ble_stack.state

        # Tick the BLE window countdown
        ble_stack.tick()

        return pred_class, confidence, \
               active_state, pairing_outcome, accepted


# ============================================================
# MODEL SELECTION — auto-detect primary model from Phase 4
# ============================================================

def select_model():
    """
    Tries to load the TFLite CNN first (primary model if
    Go/No-Go passed). Falls back to RF if TFLite not found.
    """
    if os.path.exists(TFLITE_MODEL_PATH):
        return "tflite", TFLITE_MODEL_PATH
    elif os.path.exists(RF_MODEL_PATH):
        return "rf", RF_MODEL_PATH
    else:
        print("[ERROR] No trained model found in models/")
        print("        Please run Phase 3 and Phase 4 first.")
        sys.exit(1)


# ============================================================
# LABEL ENCODER — must match Phase 3 encoding
# ============================================================

# Widar3.0 / synthetic class order (sorted alphabetically
# by sklearn's LabelEncoder)
CLASS_NAMES = ["GESTURE", "MIMICRY",
               "NEAR_FIELD_NEG", "NOT_GESTURE"]
GESTURE_CLASS_IDX = 0    # GESTURE is always class 0


# ============================================================
# MAIN SIMULATION RUNNER
# ============================================================

def run_simulation():

    print("\n" + "=" * 62)
    print("   CSI-Lock-Sim | Phase 5: Secure Lockdown Simulation")
    print("=" * 62)

    # --- Initialise components ---
    print("\n[1/5] Initialising simulation components...")

    model_type, model_path = select_model()
    esp32   = VirtualESP32(model_type, model_path)
    ble     = VirtualBLEStack()
    source  = VirtualCSISource(DATASET_PATH, n_per_class=25)

    print(f"      BLE window ticks  : {BLE_WINDOW_TICKS} steps")
    print(f"      Total samples     : {len(source)}")

    # --- Run the simulation loop ---
    print("\n[2/5] Running simulation loop...")
    print("-" * 62)
    print(f"  {'#':<5} {'True Class':<16} "
          f"{'Pred':<6} {'Conf':>6} "
          f"{'BLE State':<14} {'Pairing Outcome'}")
    print("-" * 62)

    log_rows       = []
    ble_states     = []
    sample_numbers = []
    sample_num     = 0

    for raw_sample, true_label in source:

        pred_class, confidence, ble_state, \
            pairing_outcome, accepted = \
            esp32.process(raw_sample, ble)

        pred_name = CLASS_NAMES[pred_class] \
            if pred_class < len(CLASS_NAMES) else "UNKNOWN"

        # Print per-sample log line
        print(f"  {sample_num:<5} {true_label:<16} "
              f"{pred_name[:5]:<6} {confidence:>5.2f} "
              f"{ble_state:<14} {pairing_outcome}")

        log_rows.append({
            "sample":          sample_num,
            "true_class":      true_label,
            "predicted_class": pred_name,
            "confidence":      round(confidence, 4),
            "ble_state":       ble_state,
            "pairing_outcome": pairing_outcome,
            "pairing_accepted": accepted,
        })

        ble_states.append(
            1 if ble_state == "ADVERTISING" else 0)
        sample_numbers.append(sample_num)
        sample_num += 1

    print("-" * 62)

    # --- Save log to CSV ---
    print("\n[3/5] Saving decision log...")
    log_df = pd.DataFrame(log_rows)
    os.makedirs("results", exist_ok=True)
    log_df.to_csv(LOG_PATH, index=False)
    print(f"      Log saved to : {LOG_PATH}")

    # --- Generate state timeline plot ---
    print("\n[4/5] Generating state machine timeline plot...")

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 7), sharex=True)
    fig.suptitle(
        "CSI-Lock-Sim | Phase 5: Secure Lockdown "
        "State Machine Timeline",
        fontsize=13, fontweight="bold"
    )

    # ── TOP PANEL: BLE unlock events ──────────────────────────
    # Grey background = DORMANT (default / secure state)
    ax1.axhspan(0, 1, color="lightgrey", alpha=0.4,
                label="DORMANT — Secure Lockdown (default)")

    # Green vertical line at every unlock event
    unlock_samples = [s for s, state in
                      zip(sample_numbers, ble_states)
                      if state == 1]

    for s in unlock_samples:
        ax1.axvline(x=s, color="green",
                    linewidth=2.5, alpha=0.85)

    # Dummy line for legend
    if unlock_samples:
        ax1.axvline(x=unlock_samples[0], color="green",
                    linewidth=2.5, alpha=0.85,
                    label="ADVERTISING — BLE pairing window open")

    ax1.set_ylim(0, 1)
    ax1.set_yticks([])
    ax1.set_ylabel("BLE State", fontsize=10)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.set_title(
        f"BLE Radio Unlock Events  "
        f"(green line = window opened by detected gesture  |  "
        f"grey = DORMANT / locked)  "
        f"[{len(unlock_samples)} unlock events]",
        fontsize=9
    )

    # Annotate unlock count
    ax1.text(
        0.01, 0.75,
        f"Unlock events : {len(unlock_samples)}\n"
        f"Total samples : {len(sample_numbers)}",
        transform=ax1.transAxes,
        fontsize=8, verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white",
                  alpha=0.7)
    )

    # ── BOTTOM PANEL: True class colour bands ─────────────────
    class_colours = {
        "GESTURE":        "green",
        "NOT_GESTURE":    "steelblue",
        "NEAR_FIELD_NEG": "orange",
        "MIMICRY":        "red",
    }

    for i, row in log_df.iterrows():
        col = class_colours.get(row["true_class"], "grey")
        ax2.axvspan(i - 0.5, i + 0.5,
                    color=col, alpha=0.45)

    # Overlay a marker on top of unlock events in bottom panel
    for s in unlock_samples:
        ax2.axvline(x=s, color="darkgreen",
                    linewidth=1.5, alpha=0.6,
                    linestyle="--")

    legend_patches = [
        mpatches.Patch(color=c, alpha=0.7, label=lbl)
        for lbl, c in class_colours.items()
    ]
    legend_patches.append(
        mpatches.Patch(color="darkgreen", alpha=0.6,
                       label="Unlock event (dashed)")
    )

    ax2.set_ylabel("True Sample Class", fontsize=10)
    ax2.set_xlabel("Sample Number", fontsize=10)
    ax2.set_xlim(-1, len(sample_numbers))
    ax2.set_yticks([])
    ax2.set_title(
        "True Sample Class per Step  "
        "(green bands = GESTURE  |  dashed = unlock triggered)",
        fontsize=9
    )
    ax2.legend(handles=legend_patches,
               loc="upper right", fontsize=8)

    plt.tight_layout()

    # Delete old file first to avoid Windows file-lock issues
    if os.path.exists(PLOT_PATH):
        os.remove(PLOT_PATH)

    plt.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"      Plot saved to : {PLOT_PATH}")

    # --- Print summary results table ---
    print("\n[5/5] Calculating summary results...")

    total   = len(log_df)
    correct = (log_df["true_class"] == log_df["predicted_class"]).sum()
    overall_acc = correct / total * 100

    # Security metrics
    # FAR: attacker sample (MIMICRY or NEAR_FIELD_NEG)
    #      incorrectly classified as GESTURE → pairing accepted
    attacker_mask  = log_df["true_class"].isin(
        ["MIMICRY", "NEAR_FIELD_NEG"])
    fa_mask        = attacker_mask & log_df["pairing_accepted"]
    far            = fa_mask.sum() / attacker_mask.sum() * 100 \
        if attacker_mask.sum() > 0 else 0.0

    # FRR: genuine GESTURE sample incorrectly blocked
    gesture_mask   = log_df["true_class"] == "GESTURE"
    fr_mask        = gesture_mask & ~log_df["pairing_accepted"]
    frr            = fr_mask.sum() / gesture_mask.sum() * 100 \
        if gesture_mask.sum() > 0 else 0.0

    print("\n" + "=" * 62)
    print("   CSI-Lock-Sim | Phase 5: Simulation Summary")
    print("=" * 62)
    print(f"\n  OVERALL RESULTS")
    print(f"  {'Total samples simulated':<32} : {total}")
    print(f"  {'Overall accuracy':<32} : {overall_acc:.2f}%")
    print(f"  {'BLE pairing windows opened':<32} : "
          f"{ble.total_opens}")
    print(f"  {'Pairing requests accepted':<32} : "
          f"{ble.total_accepts}")
    print(f"  {'Pairing requests rejected':<32} : "
          f"{ble.total_rejects}")

    print(f"\n  SECURITY METRICS")
    print(f"  {'False Acceptance Rate (FAR)':<32} : {far:.2f}%")
    print(f"  {'False Rejection Rate  (FRR)':<32} : {frr:.2f}%")
    print(f"  (FAR: attacker samples that triggered unlock)")
    print(f"  (FRR: genuine gestures that were blocked)")

    print(f"\n  PER-CLASS BREAKDOWN")
    print(f"  {'Class':<18} {'Total':>6} {'Correct':>8} "
          f"{'Accuracy':>10} {'Pairing Accepted':>18}")
    print(f"  {'-' * 62}")

    for cls in CLASS_NAMES:
        cls_mask    = log_df["true_class"] == cls
        cls_total   = cls_mask.sum()
        if cls_total == 0:
            continue
        cls_correct = (
            log_df[cls_mask]["true_class"] ==
            log_df[cls_mask]["predicted_class"]
        ).sum()
        cls_acc     = cls_correct / cls_total * 100
        cls_paired  = log_df[cls_mask]["pairing_accepted"].sum()
        print(f"  {cls:<18} {cls_total:>6} "
              f"{cls_correct:>8} "
              f"{cls_acc:>9.2f}% "
              f"{cls_paired:>18}")

    print(f"\n  {'=' * 58}")
    print(f"  STATE MACHINE BEHAVIOUR")
    print(f"  {'=' * 58}")
    print(f"  BLE radio: DORMANT by default (Secure Lockdown)")
    print(f"  Unlocked only when GESTURE class detected")
    print(f"  Auto-returns to DORMANT after {BLE_WINDOW_TICKS} "
          f"ticks (time-multiplexed)")
    print(f"  WiFi sensing and BLE pairing NEVER concurrent")
    print(f"  {'=' * 58}")

    print(f"\n  OUTPUT FILES")
    print(f"  Decision log       : {LOG_PATH}")
    print(f"  State timeline     : {PLOT_PATH}")
    print("=" * 62)
    print("\nPhase 5 complete. End-to-end MVP simulation done.\n")
    print("v0.1-midpoint-mvp is ready for tagging.\n")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_simulation()