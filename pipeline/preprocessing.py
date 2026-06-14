# ============================================================
# CSI-Lock-Sim | Phase 2: Signal Preprocessing Pipeline
# ============================================================
# This script takes the raw CSI dataset from Phase 1 and
# processes it through two parallel preparation paths:
#
#   PATH A (Random Forest): Butterworth filter → feature
#          extraction (RMS, variance, peak-to-peak, mean
#          absolute difference) → saves features_rf.csv
#
#   PATH B (CNN): Butterworth filter → min-max normalisation
#          → saves normalised_cnn.npy
#
#
# Maps to: Chapter 2 Section 2.4.1, Chapter 5 Section 5.4.2
# ============================================================

import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use("Agg")          # saves plot to file (no pop-up needed)
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt


# ============================================================
# CONFIGURATION
# ============================================================
SUBCARRIERS  = 30
TIME_STEPS   = 100
FEATURE_SIZE = SUBCARRIERS * TIME_STEPS

FILTER_CUTOFF = 10    # Hz  — low-pass cutoff frequency
FILTER_FS     = 100   # Hz  — sampling frequency
FILTER_ORDER  = 4     # Butterworth filter order

INPUT_PATH      = os.path.join("data", "csi_dataset.csv")
RF_OUTPUT_PATH  = os.path.join("data", "features_rf.csv")
CNN_OUTPUT_PATH = os.path.join("data", "normalised_cnn.npy")
LABELS_PATH     = os.path.join("data", "labels.npy")
PLOT_PATH       = os.path.join("results", "filter_comparison.png")


# ============================================================
# STEP 1 — LOAD THE DATASET
# ============================================================

print("\n" + "=" * 55)
print("   CSI-Lock-Sim | Phase 2: Signal Preprocessing")
print("=" * 55)

print("\n[1/5] Loading dataset from Phase 1...")
df     = pd.read_csv(INPUT_PATH)
labels = df["label"].values
data   = df.drop(columns=["label"]).values.astype(np.float32)

print(f"      Dataset shape  : {data.shape}")
print(f"      Classes found  : {np.unique(labels).tolist()}")


# ============================================================
# BUTTERWORTH LOW-PASS FILTER
# ============================================================

def apply_lowpass_filter(signal_1d, cutoff=FILTER_CUTOFF,
                         fs=FILTER_FS, order=FILTER_ORDER):
    """
    Applies a Butterworth low-pass filter to a 1D signal.
    Removes high-frequency background noise while preserving
    the slower signal changes caused by a hand gesture.
    Used by BOTH the RF and CNN preparation paths.
    """
    nyquist        = 0.5 * fs
    normal_cutoff  = cutoff / nyquist
    b, a           = butter(order, normal_cutoff,
                            btype="low", analog=False)
    return filtfilt(b, a, signal_1d)


# ============================================================
# STEP 2 — APPLY FILTER TO ALL SAMPLES
# ============================================================

print("\n[2/5] Applying Butterworth low-pass filter...")
filtered_data = np.zeros_like(data)

for i, sample in enumerate(data):
    filtered_data[i] = apply_lowpass_filter(sample)

print(f"      Filter cutoff  : {FILTER_CUTOFF} Hz")
print(f"      Filter order   : {FILTER_ORDER}")
print(f"      Samples filtered: {len(filtered_data)}")


# ============================================================
# STEP 3 — PRODUCE BEFORE/AFTER PLOT 
# ============================================================

print("\n[3/5] Generating before/after filter plot...")

# Pick one GESTURE sample and one NEAR_FIELD_NEG sample to show
gesture_idx  = np.where(labels == "GESTURE")[0][0]
neg_nf_idx   = np.where(labels == "NEAR_FIELD_NEG")[0][0]

# Use only the first sub-carrier's time series for clarity
raw_gesture      = data[gesture_idx][:TIME_STEPS]
filtered_gesture = filtered_data[gesture_idx][:TIME_STEPS]
raw_neg_nf       = data[neg_nf_idx][:TIME_STEPS]
filtered_neg_nf  = filtered_data[neg_nf_idx][:TIME_STEPS]

time_axis = np.arange(TIME_STEPS)

fig, axes = plt.subplots(2, 2, figsize=(12, 6))
fig.suptitle(
    "CSI-Lock-Sim | Phase 2: Butterworth Filter — Before vs After",
    fontsize=13, fontweight="bold"
)

# Row 1 — GESTURE sample
axes[0, 0].plot(time_axis, raw_gesture, color="steelblue", linewidth=1)
axes[0, 0].set_title("GESTURE — Raw Signal")
axes[0, 0].set_ylabel("CSI Amplitude")
axes[0, 0].set_xlabel("Time Step")

axes[0, 1].plot(time_axis, filtered_gesture,
                color="green", linewidth=1.5)
axes[0, 1].set_title("GESTURE — After Butterworth Filter")
axes[0, 1].set_ylabel("CSI Amplitude")
axes[0, 1].set_xlabel("Time Step")

# Row 2 — NEAR_FIELD_NEG sample ("1m walk-by")
axes[1, 0].plot(time_axis, raw_neg_nf, color="tomato", linewidth=1)
axes[1, 0].set_title("NEAR_FIELD_NEG (1m walk-by) — Raw Signal")
axes[1, 0].set_ylabel("CSI Amplitude")
axes[1, 0].set_xlabel("Time Step")

axes[1, 1].plot(time_axis, filtered_neg_nf,
                color="darkorange", linewidth=1.5)
axes[1, 1].set_title("NEAR_FIELD_NEG (1m walk-by) — After Filter")
axes[1, 1].set_ylabel("CSI Amplitude")
axes[1, 1].set_xlabel("Time Step")

plt.tight_layout()
os.makedirs("results", exist_ok=True)
plt.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
plt.close()

print(f"      Plot saved to  : {PLOT_PATH}")


# ============================================================
# PATH A — FEATURE EXTRACTION (for Random Forest)
# ============================================================

def extract_features(sample_1d):
    """
    Converts a filtered 1D CSI sample into a 4-feature vector.
    These simple statistics capture the key signal properties
    that distinguish a near-field gesture from noise/attacks.
    """
    return [
        float(np.sqrt(np.mean(sample_1d ** 2))),       # RMS
        float(np.var(sample_1d)),                       # Variance
        float(np.max(sample_1d) - np.min(sample_1d)),  # Peak-to-peak
        float(np.mean(np.abs(np.diff(sample_1d)))),    # Mean abs diff
    ]


print("\n[4/5] PATH A — Extracting features for Random Forest...")

feature_rows = []
for i, sample in enumerate(filtered_data):
    feats = extract_features(sample)
    feature_rows.append(feats)

feature_cols = ["rms", "variance", "peak_to_peak", "mean_abs_diff"]
df_features  = pd.DataFrame(feature_rows, columns=feature_cols)
df_features.insert(0, "label", labels)
df_features.to_csv(RF_OUTPUT_PATH, index=False)

print(f"      Features per sample : 4")
print(f"      Saved to            : {RF_OUTPUT_PATH}")


# ============================================================
# PATH B — NORMALISATION (for CNN)
# ============================================================

print("\n[5/5] PATH B — Normalising data for CNN...")

cnn_data = filtered_data.reshape(
    -1, SUBCARRIERS, TIME_STEPS).astype(np.float32)

# Min-max normalisation: scales all values to 0.0 – 1.0
min_val  = cnn_data.min()
max_val  = cnn_data.max()
cnn_norm = (cnn_data - min_val) / (max_val - min_val + 1e-8)

np.save(CNN_OUTPUT_PATH, cnn_norm)
np.save(LABELS_PATH,     labels)

print(f"      CNN array shape : {cnn_norm.shape}")
print(f"      Value range     : {cnn_norm.min():.3f} – "
      f"{cnn_norm.max():.3f}")
print(f"      Saved to        : {CNN_OUTPUT_PATH}")
print(f"      Labels saved to : {LABELS_PATH}")


# ============================================================
# PRINT FEATURE SAMPLE TABLE
# ============================================================

print("\n" + "=" * 55)
print("   CSI-Lock-Sim | Phase 2: Feature Summary (first 8 rows)")
print("=" * 55)
print(df_features.head(8).to_string(index=False))
print("-" * 55)
print(f"  Total samples    : {len(df_features)}")
print(f"  Features (RF)    : {len(feature_cols)}")
print(f"  CNN array shape  : {cnn_norm.shape}")
print(f"  Filter plot at   : {PLOT_PATH}")
print("=" * 55)
print("\nPhase 2 complete. Both paths ready for model training.\n")