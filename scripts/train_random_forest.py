"""
train_random_forest.py

Train a Random Forest classifier on barley patches for weed detection.
This script serves as a baseline comparison against deep learning models.

Usage:
    python train_random_forest.py
"""

import os
import glob
import random
import numpy as np
import rasterio
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import joblib

# =========================================================
# CONFIGURATION
# =========================================================

# Base directory for data. Change this to match your setup.
BASE_DIR = os.environ.get("WEED_DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "data"))

TRAIN_IMG_DIR = os.path.join(BASE_DIR, "patches", "cebada_v1_rgb_nir_re_indices", "images")
TRAIN_MASK_DIR = os.path.join(BASE_DIR, "patches", "cebada_v1_rgb_nir_re_indices", "masks")

TEST_IMG_DIR = os.path.join(BASE_DIR, "patches", "cebada_v2_rgb_nir_re_indices", "images")
TEST_MASK_DIR = os.path.join(BASE_DIR, "patches", "cebada_v2_rgb_nir_re_indices", "masks")

OUT_DIR = os.path.join(BASE_DIR, "results", "random_forest")
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
N_TREES = 200
MAX_SAMPLES_TRAIN = 300000   # Adjust based on your RAM
MAX_SAMPLES_TEST = 300000

random.seed(SEED)
np.random.seed(SEED)

# =========================================================
# DATA LOADING FUNCTIONS
# =========================================================

def load_pixels_from_patches(img_dir, mask_dir, max_samples=None):
    """
    Load pixel-level features and labels from patches.
    Returns:
        X_all: feature matrix (n_samples, n_features)
        y_all: label vector (n_samples,)
    """
    img_paths = sorted(glob.glob(os.path.join(img_dir, "*.tif")))
    mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.tif")))

    assert len(img_paths) == len(mask_paths), "Mismatch between images and masks."
    assert len(img_paths) > 0, "No patches found."

    X_list = []
    y_list = []

    for img_path, mask_path in zip(img_paths, mask_paths):
        with rasterio.open(img_path) as src:
            img = src.read().astype(np.float32)  # (C, H, W)

        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.int64)  # (H, W)

        # Percentile-based normalisation (2nd and 98th percentiles)
        for c in range(img.shape[0]):
            band = img[c]
            valid = np.isfinite(band)
            if np.any(valid):
                p2 = np.percentile(band[valid], 2)
                p98 = np.percentile(band[valid], 98)
                if p98 > p2:
                    band = np.clip(band, p2, p98)
                    band = (band - p2) / (p98 - p2)
                else:
                    band = np.zeros_like(band)
            else:
                band = np.zeros_like(band)
            img[c] = band
        img[~np.isfinite(img)] = 0

        # Convert to pixel table
        C, H, W = img.shape
        X = img.reshape(C, -1).T          # (H*W, C)
        y = mask.reshape(-1)              # (H*W,)

        # Keep only valid classes (1, 2, 3)
        valid = np.isin(y, [1, 2, 3])
        X = X[valid]
        y = y[valid]

        # Remap: 1->crop (0), 2->weed (1), 3->soil (2)
        y_new = np.full_like(y, -1)
        y_new[y == 1] = 0
        y_new[y == 2] = 1
        y_new[y == 3] = 2
        y = y_new

        X_list.append(X)
        y_list.append(y)

    X_all = np.vstack(X_list)
    y_all = np.concatenate(y_list)

    # Optional subsampling to limit memory usage
    if max_samples is not None and len(y_all) > max_samples:
        idx = np.random.choice(len(y_all), size=max_samples, replace=False)
        X_all = X_all[idx]
        y_all = y_all[idx]

    return X_all, y_all


def compute_iou_per_class(y_true, y_pred, num_classes=3):
    ious = []
    for cls in range(num_classes):
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        iou = tp / (tp + fp + fn + 1e-8)
        ious.append(iou)
    return ious

# =========================================================
# DATA LOADING
# =========================================================

print("Loading training data...")
X_train, y_train = load_pixels_from_patches(
    TRAIN_IMG_DIR, TRAIN_MASK_DIR, max_samples=MAX_SAMPLES_TRAIN
)

print("Loading test data...")
X_test, y_test = load_pixels_from_patches(
    TEST_IMG_DIR, TEST_MASK_DIR, max_samples=MAX_SAMPLES_TEST
)

print(f"Train features shape: {X_train.shape}")
print(f"Train labels shape: {y_train.shape}")
print(f"Test features shape: {X_test.shape}")
print(f"Test labels shape: {y_test.shape}")

# =========================================================
# MODEL TRAINING
# =========================================================

rf = RandomForestClassifier(
    n_estimators=N_TREES,
    random_state=SEED,
    n_jobs=-1,
    class_weight="balanced_subsample"
)

print("Training Random Forest classifier...")
rf.fit(X_train, y_train)

print("Predicting on test set...")
y_pred = rf.predict(X_test)

# =========================================================
# METRICS
# =========================================================

f1_crop = f1_score(y_test, y_pred, labels=[0], average="macro")
f1_weed = f1_score(y_test, y_pred, labels=[1], average="macro")
f1_soil = f1_score(y_test, y_pred, labels=[2], average="macro")

iou_crop, iou_weed, iou_soil = compute_iou_per_class(y_test, y_pred, num_classes=3)
miou = (iou_crop + iou_weed + iou_soil) / 3

print("\n=== Random Forest Results ===")
print(f"Crop F1: {f1_crop:.4f}")
print(f"Weed F1:  {f1_weed:.4f}")
print(f"Soil F1:  {f1_soil:.4f}")
print(f"Crop IoU: {iou_crop:.4f}")
print(f"Weed IoU:  {iou_weed:.4f}")
print(f"Soil IoU:  {iou_soil:.4f}")
print(f"mIoU:     {miou:.4f}")

# =========================================================
# SAVE MODEL AND SUMMARY
# =========================================================

model_path = os.path.join(OUT_DIR, "random_forest_model.joblib")
joblib.dump(rf, model_path)
print(f"\nModel saved to: {model_path}")

summary_path = os.path.join(OUT_DIR, "summary.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("=== Random Forest Training Summary ===\n")
    f.write(f"Train images: {TRAIN_IMG_DIR}\n")
    f.write(f"Train masks: {TRAIN_MASK_DIR}\n")
    f.write(f"Test images: {TEST_IMG_DIR}\n")
    f.write(f"Test masks: {TEST_MASK_DIR}\n")
    f.write(f"Number of trees: {N_TREES}\n")
    f.write(f"Train samples: {len(y_train)}\n")
    f.write(f"Test samples: {len(y_test)}\n\n")
    f.write(f"Crop F1: {f1_crop:.6f}\n")
    f.write(f"Weed F1: {f1_weed:.6f}\n")
    f.write(f"Soil F1: {f1_soil:.6f}\n")
    f.write(f"Crop IoU: {iou_crop:.6f}\n")
    f.write(f"Weed IoU: {iou_weed:.6f}\n")
    f.write(f"Soil IoU: {iou_soil:.6f}\n")
    f.write(f"mIoU: {miou:.6f}\n")

print(f"Summary saved to: {summary_path}")
