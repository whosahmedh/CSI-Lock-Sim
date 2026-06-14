# ============================================================
# CSI-Lock-Sim | Phase 4: Model Compression & Fit-Check
# ============================================================
# This script performs the static feasibility analysis that
# answers the core question of this research:
#
#   "Can either model realistically fit on an ESP32?"
#
# Steps performed:
#   1. Load the trained CNN from Phase 3
#   2. Convert it to TensorFlow Lite with post-training
#      quantisation (32-bit → 8-bit) — the standard TinyML
#      compression step
#   3. Save the quantised model as a .tflite file and
#      measure its size in KB
#   4. Measure the RF model's saved size in KB
#   5. Compare both against the ESP32's available SRAM
#      (after reserving space for the WiFi and BLE stacks)
#   6. Optionally compare against a Raspberry Pi-class device
#   7. Print the fit-check results table and Go/No-Go decision
#
# No physical hardware is needed — this is entirely a
# static, desk-based analysis using published spec figures.
#
# Maps to: Chapter 2 Section 2.4.3, Chapter 3 RO5,
#          Chapter 5 Section 5.4.4
# ============================================================

import numpy as np
import pandas as pd
import os
import joblib
import tensorflow as tf


# ============================================================
# HARDWARE REFERENCE SPECIFICATIONS
# ============================================================
# These values come from published manufacturer datasheets
# and are the basis of the static feasibility comparison.
# No physical device is needed to run this analysis.
#
# ESP32 (Espressif Systems):
#   Total SRAM          : 520 KB
#   WiFi stack overhead : ~120 KB  (Espressif ESP-IDF docs)
#   BLE stack overhead  : ~80 KB   (Espressif ESP-IDF docs)
#   Runtime/OS reserve  : ~20 KB
#   Available for ML    : 520 - 120 - 80 - 20 = 300 KB
#
# Raspberry Pi Zero 2 W (Raspberry Pi Foundation):
#   Total RAM           : 512,000 KB (512 MB)
#   Available for ML    : ~400,000 KB (effectively unlimited
#                         for models of this scale)
# ============================================================

ESP32_TOTAL_SRAM_KB       = 520
ESP32_WIFI_STACK_KB       = 120
ESP32_BLE_STACK_KB        = 80
ESP32_RUNTIME_KB          = 20
ESP32_AVAILABLE_KB        = (ESP32_TOTAL_SRAM_KB
                             - ESP32_WIFI_STACK_KB
                             - ESP32_BLE_STACK_KB
                             - ESP32_RUNTIME_KB)

RPI_TOTAL_RAM_KB          = 512 * 1024   # 512 MB in KB
RPI_AVAILABLE_KB          = 400 * 1024   # conservative estimate

RF_MODEL_PATH             = os.path.join("models", "rf_model.joblib")
CNN_MODEL_PATH            = os.path.join("models", "cnn_model.keras")
TFLITE_OUTPUT_PATH        = os.path.join("models", "cnn_quantised.tflite")


# ============================================================
# STEP 1 — LOAD MODELS FROM PHASE 3
# ============================================================

print("\n" + "=" * 62)
print("   CSI-Lock-Sim | Phase 4: Compression & Fit-Check")
print("=" * 62)

print("\n[1/5] Loading trained models from Phase 3...")

rf_model  = joblib.load(RF_MODEL_PATH)
cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)

print(f"      RF model loaded  : {RF_MODEL_PATH}")
print(f"      CNN model loaded : {CNN_MODEL_PATH}")
print(f"      CNN parameters   : {cnn_model.count_params():,}")


# ============================================================
# STEP 2 — CONVERT CNN TO TFLITE WITH QUANTISATION
# ============================================================

print("\n[2/5] Converting CNN to TensorFlow Lite "
      "(post-training quantisation)...")
print("      Applying DEFAULT optimisation "
      "(dynamic range quantisation)...")
print("      This reduces 32-bit float weights → 8-bit integers.")

converter = tf.lite.TFLiteConverter.from_keras_model(cnn_model)

# Apply post-training quantisation
# DEFAULT = dynamic range quantisation: reduces model size by ~4x
# with minimal accuracy loss — the standard TinyML step
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

# Save the .tflite file
with open(TFLITE_OUTPUT_PATH, "wb") as f:
    f.write(tflite_model)

tflite_size_kb = len(tflite_model) / 1024
print(f"      Quantised model saved : {TFLITE_OUTPUT_PATH}")
print(f"      Quantised model size  : {tflite_size_kb:.2f} KB")


# ============================================================
# STEP 3 — MEASURE MODEL SIZES
# ============================================================

print("\n[3/5] Measuring model sizes...")

# RF: size of the saved joblib file
rf_size_kb      = os.path.getsize(RF_MODEL_PATH) / 1024

# CNN: original Keras model size (pre-compression)
cnn_keras_size_kb = os.path.getsize(CNN_MODEL_PATH) / 1024

# CNN: quantised TFLite size (what would actually run on ESP32)
cnn_tflite_size_kb = tflite_size_kb

print(f"      RF model size          : {rf_size_kb:.2f} KB")
print(f"      CNN (Keras, original)  : {cnn_keras_size_kb:.2f} KB")
print(f"      CNN (TFLite, quantised): {cnn_tflite_size_kb:.2f} KB")
print(f"      Compression ratio      : "
      f"{cnn_keras_size_kb / cnn_tflite_size_kb:.1f}x smaller")


# ============================================================
# STEP 4 — RUN THE STATIC FIT-CHECK
# ============================================================

print("\n[4/5] Running static fit-check against hardware specs...")

def fit_check(model_size_kb, available_kb):
    """Returns PASS, MARGINAL, or FAIL based on how well
    the model fits within the available memory budget."""
    if model_size_kb <= available_kb * 0.75:
        return "PASS ✓"
    elif model_size_kb <= available_kb:
        return "MARGINAL ⚠"
    else:
        return "FAIL ✗"

rf_esp32_result       = fit_check(rf_size_kb, ESP32_AVAILABLE_KB)
cnn_esp32_result      = fit_check(cnn_tflite_size_kb, ESP32_AVAILABLE_KB)
rf_rpi_result         = fit_check(rf_size_kb, RPI_AVAILABLE_KB)
cnn_rpi_result        = fit_check(cnn_tflite_size_kb, RPI_AVAILABLE_KB)

rf_esp32_headroom     = ESP32_AVAILABLE_KB - rf_size_kb
cnn_esp32_headroom    = ESP32_AVAILABLE_KB - cnn_tflite_size_kb


# ============================================================
# STEP 5 — GO/NO-GO DECISION LOGIC
# ============================================================

print("\n[5/5] Making Go/No-Go decision...")

# Decision rule:
#   If CNN passes ESP32 fit-check → CNN is primary, RF is backup
#   If CNN fails ESP32 fit-check → RF is primary (guaranteed fit)
#   CNN results are always reported as a comparison either way

if "FAIL" not in cnn_esp32_result:
    primary_model    = "Depthwise-Separable CNN (TFLite quantised)"
    fallback_model   = "Random Forest (backup)"
    decision         = "GO — CNN proceeds to Phase 5 integration"
    decision_reason  = (f"CNN ({cnn_tflite_size_kb:.2f} KB) fits "
                        f"within ESP32's available {ESP32_AVAILABLE_KB} KB "
                        f"with {cnn_esp32_headroom:.2f} KB headroom.")
else:
    primary_model    = "Random Forest (fallback activated)"
    fallback_model   = "DS-CNN (benchmarked but not deployed)"
    decision         = "NO-GO for CNN — RF activated as primary model"
    decision_reason  = (f"CNN ({cnn_tflite_size_kb:.2f} KB) exceeds "
                        f"ESP32's available {ESP32_AVAILABLE_KB} KB. "
                        f"RF ({rf_size_kb:.2f} KB) is the primary model.")


# ============================================================
# PRINT RESULTS TABLE  <-- THIS IS YOUR SCREENSHOT
# ============================================================

print("\n" + "=" * 62)
print("   CSI-Lock-Sim | Phase 4: Static Feasibility Report")
print("=" * 62)

print(f"\n  HARDWARE REFERENCE SPECIFICATIONS")
print(f"  {'Platform':<28} {'ESP32':>12} {'RPi Zero 2W':>14}")
print(f"  {'-'*54}")
print(f"  {'Total Memory':<28} "
      f"{'520 KB':>12} {'512 MB':>14}")
print(f"  {'Stack Overhead (WiFi+BLE)':<28} "
      f"{'220 KB':>12} {'N/A':>14}")
print(f"  {'Available for ML Model':<28} "
      f"{ESP32_AVAILABLE_KB:>11} KB "
      f"{RPI_AVAILABLE_KB/1024:>10.0f} MB")

print(f"\n  MODEL SIZE COMPARISON")
print(f"  {'Model':<28} {'Size (KB)':>10} "
      f"{'ESP32':>10} {'RPi Zero 2W':>12}")
print(f"  {'-'*62}")
print(f"  {'Random Forest':<28} "
      f"{rf_size_kb:>9.2f} "
      f"{rf_esp32_result:>12} "
      f"{rf_rpi_result:>12}")
print(f"  {'CNN (Keras, pre-compress)':<28} "
      f"{cnn_keras_size_kb:>9.2f} "
      f"{'—':>12} "
      f"{'—':>12}")
print(f"  {'CNN (TFLite, quantised)':<28} "
      f"{cnn_tflite_size_kb:>9.2f} "
      f"{cnn_esp32_result:>12} "
      f"{cnn_rpi_result:>12}")

print(f"\n  COMPRESSION SUMMARY")
print(f"  {'Original CNN size':<28} : {cnn_keras_size_kb:.2f} KB")
print(f"  {'Quantised CNN size':<28} : {cnn_tflite_size_kb:.2f} KB")
print(f"  {'Compression ratio':<28} : "
      f"{cnn_keras_size_kb/cnn_tflite_size_kb:.1f}x smaller")
print(f"  {'ESP32 headroom (RF)':<28} : {rf_esp32_headroom:.2f} KB")
print(f"  {'ESP32 headroom (CNN)':<28} : {cnn_esp32_headroom:.2f} KB")

print(f"\n  {'=' * 58}")
print(f"  GO/NO-GO DECISION")
print(f"  {'=' * 58}")
print(f"  Decision       : {decision}")
print(f"  Primary model  : {primary_model}")
print(f"  Backup model   : {fallback_model}")
print(f"  Reason         : {decision_reason}")
print(f"  {'=' * 58}")

print(f"\n  {'IMPORTANT NOTE FOR REPORT':}")
print(f"  All size and headroom figures are ESTIMATES based on")
print(f"  published manufacturer specifications and static")
print(f"  analysis — not measured on physical hardware.")
print(f"  This is an accepted limitation of a simulation-based")
print(f"  feasibility study (see Chapter 3, Risk Register).")

print("=" * 62)
print("\nPhase 4 complete. Go/No-Go decision recorded.\n")
print(f"Primary model for Phase 5 : {primary_model}\n")