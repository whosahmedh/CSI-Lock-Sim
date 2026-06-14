# ============================================================
# CSI-Lock-Sim | Phase 1: Data Ingestion & Relabelling
# ============================================================
# This script loads gesture data from the Widar3.0 dataset
# (if available in data/widar3/) and generates synthetic
# samples for the remaining three classes:
#
#   GESTURE         - Real near-field gesture data (Widar3.0)
#   NOT_GESTURE     - Synthetic ambient / empty room signal
#   NEAR_FIELD_NEG  - Synthetic: person walking past at ~1m
#   MIMICRY         - Synthetic: attacker faking a close signal
#
# If Widar3.0 files are not found, all four classes are
# generated synthetically as a working fallback.
#
# Maps to: Chapter 1 RO2, Chapter 3 Data Requirements,
#          Chapter 5 Section 5.4.1
# ============================================================

import numpy as np
import pandas as pd
import os
import glob

# ============================================================
# CONFIGURATION
# ============================================================
RANDOM_SEED       = 42
SAMPLES_PER_CLASS = 200
SUBCARRIERS       = 30
TIME_STEPS        = 100
WIDAR_PATH        = os.path.join("data", "widar3")

np.random.seed(RANDOM_SEED)

feature_size = SUBCARRIERS * TIME_STEPS


# ============================================================
# WIDAR3.0 LOADER
# ============================================================

def load_widar3_gestures(widar_path, n_samples, feature_size):
    """
    Attempts to load real CSI gesture data from Widar3.0 .mat files.
    Returns a numpy array of shape (n_samples, feature_size) if
    successful, or None if no files are found or loading fails.
    """
    try:
        import scipy.io as sio
    except ImportError:
        print("  [WARNING] scipy not found. Run: pip install scipy")
        return None

    mat_files = glob.glob(os.path.join(widar_path, "**", "*.mat"),
                          recursive=True)
    mat_files += glob.glob(os.path.join(widar_path, "*.mat"))

    if not mat_files:
        print(f"  [INFO] No .mat files found in {widar_path}/")
        print("  [INFO] Falling back to synthetic gesture generation.")
        return None

    print(f"  [INFO] Found {len(mat_files)} Widar3.0 .mat files.")
    samples = []

    for filepath in mat_files:
        if len(samples) >= n_samples:
            break
        try:
            mat      = sio.loadmat(filepath)
            data_key = None

            # Try common key names used in Widar3.0 .mat files
            for key in ["velocity_spectrum_density", "csi_data",
                        "csi", "data", "CSI"]:
                if key in mat:
                    data_key = key
                    break

            if data_key is None:
                # Use first non-metadata key
                keys = [k for k in mat.keys()
                        if not k.startswith("__")]
                if keys:
                    data_key = keys[0]

            if data_key:
                raw    = np.array(mat[data_key], dtype=np.float32)
                raw    = np.abs(raw)          # amplitude only
                flat   = raw.flatten()

                # Resize to consistent feature length
                if len(flat) >= feature_size:
                    flat = flat[:feature_size]
                else:
                    flat = np.pad(flat,
                                  (0, feature_size - len(flat)),
                                  mode="constant")
                samples.append(flat)

        except Exception as e:
            continue   # Skip unreadable files silently

    if len(samples) == 0:
        print("  [INFO] Could not extract data from .mat files.")
        print("  [INFO] Falling back to synthetic gesture generation.")
        return None

    print(f"  [INFO] Successfully loaded {len(samples)} real "
          f"gesture samples from Widar3.0.")
    samples = np.array(samples)

    # If we have fewer samples than needed, repeat to fill
    if len(samples) < n_samples:
        repeats  = int(np.ceil(n_samples / len(samples)))
        samples  = np.tile(samples, (repeats, 1))

    return samples[:n_samples]


# ============================================================
# SYNTHETIC SAMPLE GENERATORS (fallback + hard negatives)
# ============================================================

def generate_synthetic_gesture(n, subcarriers, time_steps):
    """
    Synthetic GESTURE fallback: sharp high-amplitude peak in the
    middle of the window — simulates a near-field hand gesture.
    Used only when Widar3.0 files are unavailable.
    """
    samples = []
    for _ in range(n):
        base = np.random.normal(0, 0.1, (subcarriers, time_steps))
        p1   = time_steps // 3
        p2   = 2 * time_steps // 3
        base[:, p1:p2] += np.random.normal(
            2.0, 0.3, (subcarriers, p2 - p1))
        samples.append(base.flatten())
    return np.array(samples)


def generate_ambient(n, subcarriers, time_steps):
    """
    NOT_GESTURE: low-amplitude flat noise — empty room baseline.
    """
    samples = []
    for _ in range(n):
        base = np.random.normal(0, 0.1, (subcarriers, time_steps))
        samples.append(base.flatten())
    return np.array(samples)


def generate_near_field_negative(ambient_samples, boost=1.5):
    """
    NEAR_FIELD_NEG: person walking past at ~1 metre.
    Moderate amplitude increase — more than background noise
    but without the sharp temporal peak of a gesture.
    Directly addresses supervisor feedback: 10 cm vs 1 m.
    """
    noise = np.random.normal(0, 0.05, ambient_samples.shape)
    return (ambient_samples * boost) + noise


def generate_mimicry(ambient_samples, boost=2.5):
    """
    MIMICRY: attacker amplifying a far-field signal to fake a
    close-range gesture. Tests the model's False Acceptance Rate
    under a simulated signal-amplification attack.
    """
    noise = np.random.normal(0, 0.08, ambient_samples.shape)
    return (ambient_samples * boost) + noise


# ============================================================
# GENERATE ALL FOUR CLASSES
# ============================================================

print("\n" + "=" * 55)
print("   CSI-Lock-Sim | Phase 1: Data Ingestion")
print("=" * 55)
print(f"\nConfiguration:")
print(f"  Samples per class : {SAMPLES_PER_CLASS}")
print(f"  Sub-carriers      : {SUBCARRIERS}")
print(f"  Time steps        : {TIME_STEPS}")
print(f"  Feature size      : {feature_size}")
print(f"\nLoading GESTURE class...")

# -- GESTURE: try Widar3.0 first, fall back to synthetic --
gesture_data = load_widar3_gestures(
    WIDAR_PATH, SAMPLES_PER_CLASS, feature_size)

if gesture_data is None:
    print("  Generating synthetic gesture samples...")
    gesture_data = generate_synthetic_gesture(
        SAMPLES_PER_CLASS, SUBCARRIERS, TIME_STEPS)
    gesture_source = "Synthetic (fallback)"
else:
    gesture_source = "Widar3.0 (real CSI data)"

print(f"\nGenerating remaining classes synthetically...")
ambient_data = generate_ambient(
    SAMPLES_PER_CLASS, SUBCARRIERS, TIME_STEPS)
neg_nf_data  = generate_near_field_negative(ambient_data)
mimicry_data = generate_mimicry(ambient_data)

# ============================================================
# COMBINE AND LABEL
# ============================================================

labels = (
    ["GESTURE"]        * SAMPLES_PER_CLASS +
    ["NOT_GESTURE"]    * SAMPLES_PER_CLASS +
    ["NEAR_FIELD_NEG"] * SAMPLES_PER_CLASS +
    ["MIMICRY"]        * SAMPLES_PER_CLASS
)

all_data = np.vstack([
    gesture_data,
    ambient_data,
    neg_nf_data,
    mimicry_data
])

df = pd.DataFrame(all_data)
df.insert(0, "label", labels)

# ============================================================
# SAVE TO /data
# ============================================================

os.makedirs("data", exist_ok=True)
output_path = os.path.join("data", "csi_dataset.csv")
df.to_csv(output_path, index=False)

# ============================================================
# PRINT CLASS SUMMARY TABLE  <-- THIS IS YOUR SCREENSHOT
# ============================================================

summary          = df["label"].value_counts().reset_index()
summary.columns  = ["Class", "Sample Count"]

print("\n" + "=" * 55)
print("   CSI-Lock-Sim | Phase 1: Dataset Class Summary")
print("=" * 55)
print(summary.to_string(index=False))
print("-" * 55)
print(f"  Gesture source   : {gesture_source}")
print(f"  Total samples    : {len(df)}")
print(f"  Feature columns  : {len(df.columns) - 1}")
print(f"  Saved to         : {output_path}")
print("=" * 55)
print("\nPhase 1 complete. Dataset ready for preprocessing.\n")