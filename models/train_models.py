# ============================================================
# CSI-Lock-Sim | Phase 3: Model Training
# ============================================================
# This script trains and evaluates two candidate models on the
# preprocessed data from Phase 2:
#
#   MODEL A — Random Forest (RF)
#             Input : features_rf.csv (4 statistical features)
#             Role  : Guaranteed-fit fallback model. Fast to
#                     train, tiny memory footprint, always
#                     passes the ESP32 SRAM fit-check.
#
#   MODEL B — Depthwise-Separable CNN
#             Input : normalised_cnn.npy (30 x 100 arrays)
#             Role  : Aspirational deep learning model.
#                     Trained and benchmarked here; whether
#                     it fits the ESP32 is determined in
#                     Phase 4's static fit-check.
#
# Both models are trained on an identical 70/15/15 split.
# Results are printed as a side-by-side comparison table.
#
# Maps to: Chapter 2 Section 2.4.2, Chapter 5 Section 5.4.3,
#          Chapter 1 RO3
# ============================================================

import numpy as np
import pandas as pd
import os
import time
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks


# ============================================================
# CONFIGURATION
# ============================================================
RANDOM_SEED   = 42
TEST_SIZE     = 0.15
VAL_SIZE      = 0.15
SUBCARRIERS   = 30
TIME_STEPS    = 100
NUM_CLASSES   = 4
CNN_EPOCHS    = 30
CNN_BATCH     = 32

RF_MODEL_PATH  = os.path.join("models", "rf_model.joblib")
CNN_MODEL_PATH = os.path.join("models", "cnn_model.keras")

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 60)
print("   CSI-Lock-Sim | Phase 3: Model Training")
print("=" * 60)

print("\n[1/6] Loading preprocessed data from Phase 2...")

# -- Random Forest path --
df_rf    = pd.read_csv(os.path.join("data", "features_rf.csv"))
X_rf     = df_rf.drop(columns=["label"]).values
y_raw    = df_rf["label"].values

# -- CNN path --
X_cnn    = np.load(os.path.join("data", "normalised_cnn.npy"))

# -- Encode labels to integers --
le       = LabelEncoder()
y_enc    = le.fit_transform(y_raw)

print(f"      RF feature shape  : {X_rf.shape}")
print(f"      CNN data shape    : {X_cnn.shape}")
print(f"      Classes           : {list(le.classes_)}")
print(f"      Label mapping     : "
      f"{dict(zip(le.classes_, le.transform(le.classes_)))}")


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

print("\n[2/6] Splitting data (70% train / 15% val / 15% test)...")

# First split off the test set
X_rf_tv, X_rf_test, X_cnn_tv, X_cnn_test, y_tv, y_test = \
    train_test_split(X_rf, X_cnn, y_enc,
                     test_size=TEST_SIZE,
                     random_state=RANDOM_SEED,
                     stratify=y_enc)

# Then split remaining into train and val
val_ratio = VAL_SIZE / (1 - TEST_SIZE)
X_rf_train, X_rf_val, X_cnn_train, X_cnn_val, y_train, y_val = \
    train_test_split(X_rf_tv, X_cnn_tv, y_tv,
                     test_size=val_ratio,
                     random_state=RANDOM_SEED,
                     stratify=y_tv)

print(f"      Train samples : {len(y_train)}")
print(f"      Val samples   : {len(y_val)}")
print(f"      Test samples  : {len(y_test)}")

# Reshape CNN arrays to include channel dimension (required by Keras)
X_cnn_train = X_cnn_train[..., np.newaxis]
X_cnn_val   = X_cnn_val[..., np.newaxis]
X_cnn_test  = X_cnn_test[..., np.newaxis]


# ============================================================
# MODEL A — RANDOM FOREST
# ============================================================

print("\n[3/6] Training MODEL A — Random Forest...")

rf_start  = time.time()

rf_model  = RandomForestClassifier(
    n_estimators  = 100,
    max_depth     = 10,
    random_state  = RANDOM_SEED,
    n_jobs        = -1       # use all CPU cores
)
rf_model.fit(X_rf_train, y_train)

rf_train_time = time.time() - rf_start

# Evaluate
rf_val_preds  = rf_model.predict(X_rf_val)
rf_test_preds = rf_model.predict(X_rf_test)
rf_val_acc    = accuracy_score(y_val,  rf_val_preds)
rf_test_acc   = accuracy_score(y_test, rf_test_preds)

# Save model
joblib.dump(rf_model, RF_MODEL_PATH)
rf_size_kb = os.path.getsize(RF_MODEL_PATH) / 1024

print(f"      Val accuracy  : {rf_val_acc * 100:.2f}%")
print(f"      Test accuracy : {rf_test_acc * 100:.2f}%")
print(f"      Training time : {rf_train_time:.2f}s")
print(f"      Model size    : {rf_size_kb:.2f} KB")
print(f"      Saved to      : {RF_MODEL_PATH}")


# ============================================================
# MODEL B — DEPTHWISE-SEPARABLE CNN
# ============================================================

print("\n[4/6] Training MODEL B — Depthwise-Separable CNN...")

def build_ds_cnn(input_shape, num_classes):
    """
    Lightweight Depthwise-Separable CNN.
    Uses SeparableConv2D layers which separate spatial and
    depth-wise filtering — reducing parameters by up to 90%
    vs a standard CNN while maintaining accuracy.
    Mirrors the CSI-DeepNet architecture discussed in
    Chapter 2 Section 2.4.2 (Kabir et al., 2022).
    """
    model = models.Sequential([
        layers.SeparableConv2D(
            8, (3, 3), activation="relu",
            padding="same",
            input_shape=input_shape
        ),
        layers.MaxPooling2D((2, 2)),
        layers.SeparableConv2D(
            16, (3, 3), activation="relu",
            padding="same"
        ),
        layers.MaxPooling2D((2, 2)),
        layers.SeparableConv2D(
            32, (3, 3), activation="relu",
            padding="same"
        ),
        layers.GlobalAveragePooling2D(),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ], name="DS_CNN")

    model.compile(
        optimizer = "adam",
        loss      = "sparse_categorical_crossentropy",
        metrics   = ["accuracy"]
    )
    return model


input_shape = (SUBCARRIERS, TIME_STEPS, 1)
cnn_model   = build_ds_cnn(input_shape, NUM_CLASSES)

# Early stopping — stops training if val accuracy stops improving
early_stop = callbacks.EarlyStopping(
    monitor              = "val_accuracy",
    patience             = 5,
    restore_best_weights = True
)

cnn_start = time.time()

history = cnn_model.fit(
    X_cnn_train, y_train,
    validation_data = (X_cnn_val, y_val),
    epochs          = CNN_EPOCHS,
    batch_size      = CNN_BATCH,
    callbacks       = [early_stop],
    verbose         = 0       # suppress per-epoch output for cleanliness
)

cnn_train_time = time.time() - cnn_start

# Evaluate
cnn_val_acc  = max(history.history["val_accuracy"])
cnn_test_loss, cnn_test_acc = cnn_model.evaluate(
    X_cnn_test, y_test, verbose=0)

# Save model
cnn_model.save(CNN_MODEL_PATH)
cnn_size_kb = os.path.getsize(CNN_MODEL_PATH) / 1024

# Count total parameters
total_params = cnn_model.count_params()

print(f"      Val accuracy  : {cnn_val_acc * 100:.2f}%")
print(f"      Test accuracy : {cnn_test_acc * 100:.2f}%")
print(f"      Training time : {cnn_train_time:.2f}s")
print(f"      Total params  : {total_params:,}")
print(f"      Model size    : {cnn_size_kb:.2f} KB")
print(f"      Saved to      : {CNN_MODEL_PATH}")


# ============================================================
# PER-CLASS BREAKDOWN (RF)
# ============================================================

print("\n[5/6] Generating per-class report...")

rf_report  = classification_report(
    y_test, rf_test_preds,
    target_names = le.classes_,
    output_dict  = True
)

cnn_report = classification_report(
    y_test,
    np.argmax(cnn_model.predict(X_cnn_test, verbose=0), axis=1),
    target_names = le.classes_,
    output_dict  = True
)


# ============================================================
# PRINT COMPARISON TABLE  <-- THIS IS YOUR SCREENSHOT
# ============================================================

print("\n[6/6] Printing comparison table...")

print("\n" + "=" * 60)
print("   CSI-Lock-Sim | Phase 3: Model Comparison Table")
print("=" * 60)
print(f"  {'Metric':<28} {'Random Forest':>14} {'DS-CNN':>14}")
print("-" * 60)
print(f"  {'Test Accuracy':<28} "
      f"{rf_test_acc*100:>13.2f}% "
      f"{cnn_test_acc*100:>13.2f}%")
print(f"  {'Validation Accuracy':<28} "
      f"{rf_val_acc*100:>13.2f}% "
      f"{cnn_val_acc*100:>13.2f}%")
print(f"  {'Training Time (s)':<28} "
      f"{rf_train_time:>14.2f} "
      f"{cnn_train_time:>14.2f}")
print(f"  {'Model Size (KB)':<28} "
      f"{rf_size_kb:>14.2f} "
      f"{cnn_size_kb:>14.2f}")
print(f"  {'Total Parameters':<28} "
      f"{'N/A (tree-based)':>14} "
      f"{total_params:>14,}")
print("-" * 60)

# Per-class accuracy
print(f"\n  Per-class Test Accuracy:")
for cls in le.classes_:
    rf_cls_acc  = rf_report[cls]["precision"]
    cnn_cls_acc = cnn_report[cls]["precision"]
    print(f"    {cls:<24} "
          f"{rf_cls_acc*100:>8.2f}%   "
          f"{cnn_cls_acc*100:>8.2f}%")

print("\n" + "-" * 60)
print(f"  Models saved to    : models/")
print(f"  Next step          : Phase 4 — Compression & Fit-Check")
print("=" * 60)
print("\nPhase 3 complete. Both models trained and saved.\n")