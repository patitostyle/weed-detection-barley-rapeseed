"""
train_unet_spatial.py

Train U-Net on Barley 1 (V1 + V2) and evaluate on Barley 2 (spatial validation).
This script generates the base model used for few-shot fine-tuning on rapeseed.

Usage:
    python train_unet_spatial.py
"""

import os
import glob
import random
import numpy as np
import rasterio
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = os.environ.get("WEED_DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "data"))

TRAIN_DIRS = [
    os.path.join(BASE_DIR, "patches", "cebada_v1_rgb_nir_re_indices"),
    os.path.join(BASE_DIR, "patches", "cebada_v2_rgb_nir_re_indices"),
]

TEST_DIR = os.path.join(BASE_DIR, "patches", "cebada_v2_C2_rgb_nir_re_indices")

OUT_DIR = os.path.join(BASE_DIR, "results", "unet_train_v1v2_test_c2_30ep")
os.makedirs(OUT_DIR, exist_ok=True)

BATCH_SIZE = 8
EPOCHS = 30
LR = 1e-4
SEED = 42
NUM_CLASSES = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================================================
# SEED
# =========================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# =========================================================
# DATASET
# =========================================================

class PatchDatasetMultiTrain(Dataset):
    def __init__(self, train_dirs):
        self.img_paths = []
        self.mask_paths = []

        for d in train_dirs:
            img_dir = os.path.join(d, "images")
            mask_dir = os.path.join(d, "masks")

            imgs = sorted(glob.glob(os.path.join(img_dir, "*.tif")))
            masks = sorted(glob.glob(os.path.join(mask_dir, "*.tif")))

            assert len(imgs) == len(masks), f"Mismatch in {d}"
            assert len(imgs) > 0, f"No patches found in {d}"

            img_names = [os.path.basename(x) for x in imgs]
            mask_names = [os.path.basename(x) for x in masks]
            assert img_names == mask_names, f"Name mismatch in {d}"

            self.img_paths.extend(imgs)
            self.mask_paths.extend(masks)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        with rasterio.open(self.img_paths[idx]) as src:
            img = src.read().astype(np.float32)

        with rasterio.open(self.mask_paths[idx]) as src:
            mask = src.read(1).astype(np.int64)

        # Percentile-based normalisation
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

        # Remap: 0->ignore, 1->crop, 2->weed, 3->soil
        mask_new = np.full_like(mask, 255, dtype=np.uint8)
        mask_new[mask == 1] = 0
        mask_new[mask == 2] = 1
        mask_new[mask == 3] = 2
        mask = mask_new

        return torch.tensor(img, dtype=torch.float32), torch.tensor(mask, dtype=torch.long)


class PatchDatasetTest(Dataset):
    def __init__(self, test_dir):
        self.img_paths = sorted(glob.glob(os.path.join(test_dir, "images", "*.tif")))
        self.mask_paths = sorted(glob.glob(os.path.join(test_dir, "masks", "*.tif")))

        assert len(self.img_paths) == len(self.mask_paths), "Mismatch in test"
        assert len(self.img_paths) > 0, f"No test patches found in {test_dir}"

        img_names = [os.path.basename(x) for x in self.img_paths]
        mask_names = [os.path.basename(x) for x in self.mask_paths]
        assert img_names == mask_names, "Name mismatch in test"

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        with rasterio.open(self.img_paths[idx]) as src:
            img = src.read().astype(np.float32)

        with rasterio.open(self.mask_paths[idx]) as src:
            mask = src.read(1).astype(np.int64)

        # Percentile-based normalisation (same as training)
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

        mask_new = np.full_like(mask, 255, dtype=np.uint8)
        mask_new[mask == 1] = 0
        mask_new[mask == 2] = 1
        mask_new[mask == 3] = 2
        mask = mask_new

        return torch.tensor(img, dtype=torch.float32), torch.tensor(mask, dtype=torch.long)

# =========================================================
# METRICS
# =========================================================

def compute_metrics(preds, targets, num_classes=3, ignore_index=255):
    preds = preds.view(-1)
    targets = targets.view(-1)

    valid = targets != ignore_index
    preds = preds[valid]
    targets = targets[valid]

    f1_per_class = []
    iou_per_class = []

    for cls in range(num_classes):
        pred_c = preds == cls
        targ_c = targets == cls

        tp = (pred_c & targ_c).sum().item()
        fp = (pred_c & ~targ_c).sum().item()
        fn = (~pred_c & targ_c).sum().item()

        p = tp / (tp + fp + 1e-8)
        r = tp / (tp + fn + 1e-8)
        f1 = 2 * p * r / (p + r + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)

        f1_per_class.append(f1)
        iou_per_class.append(iou)

    return f1_per_class, iou_per_class

# =========================================================
# U-NET ARCHITECTURE
# =========================================================

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels=8, num_classes=3):
        super().__init__()

        self.enc1 = ConvBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(256, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(128, 64)

        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        b = self.bottleneck(self.pool3(e3))

        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.final(d1)

# =========================================================
# DATALOADERS
# =========================================================

print("Train dirs:")
for d in TRAIN_DIRS:
    print(" -", d)
print("Test dir:", TEST_DIR)
print("Output dir:", OUT_DIR)

train_ds = PatchDatasetMultiTrain(TRAIN_DIRS)
test_ds = PatchDatasetTest(TEST_DIR)

pin_memory = torch.cuda.is_available()
num_workers = 2

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=pin_memory
)

test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_memory
)

sample_x, _ = train_ds[0]
in_channels = sample_x.shape[0]

# =========================================================
# MODEL
# =========================================================

model = UNet(in_channels=in_channels, num_classes=NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss(ignore_index=255)
optimizer = optim.Adam(model.parameters(), lr=LR)

print(f"\nDevice: {DEVICE}")
print(f"Input channels: {in_channels}")
print(f"Train patches: {len(train_ds)}")
print(f"Test patches: {len(test_ds)}")

# =========================================================
# TRAINING
# =========================================================

best_test_f1_weed = -1
history = []

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    for imgs, masks in train_loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    model.eval()
    test_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for imgs, masks in test_loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)

            outputs = model(imgs)
            loss = criterion(outputs, masks)

            test_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu())
            all_targets.append(masks.cpu())

    test_loss /= len(test_loader)

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    f1s, ious = compute_metrics(all_preds, all_targets, num_classes=NUM_CLASSES, ignore_index=255)

    f1_crop, f1_weed, f1_soil = f1s
    iou_crop, iou_weed, iou_soil = ious
    miou = (iou_crop + iou_weed + iou_soil) / 3

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "test_loss": test_loss,
        "f1_crop": f1_crop,
        "f1_weed": f1_weed,
        "f1_soil": f1_soil,
        "iou_crop": iou_crop,
        "iou_weed": iou_weed,
        "iou_soil": iou_soil,
        "miou": miou
    })

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train loss: {train_loss:.4f} | "
        f"Test loss: {test_loss:.4f} | "
        f"Weed F1: {f1_weed:.4f} | "
        f"Weed IoU: {iou_weed:.4f} | "
        f"mIoU: {miou:.4f}"
    )

    if f1_weed > best_test_f1_weed:
        best_test_f1_weed = f1_weed
        torch.save(model.state_dict(), os.path.join(OUT_DIR, "best_model.pth"))

# =========================================================
# SAVE SUMMARY
# =========================================================

summary_path = os.path.join(OUT_DIR, "summary.txt")

with open(summary_path, "w", encoding="utf-8") as f:
    f.write("=== TRAIN DIRS ===\n")
    for d in TRAIN_DIRS:
        f.write(f"{d}\n")
    f.write("\n")
    f.write(f"Test dir: {TEST_DIR}\n")
    f.write(f"Input channels: {in_channels}\n")
    f.write(f"Best weed F1: {best_test_f1_weed:.6f}\n\n")

    for h in history:
        f.write(
            f"Epoch {h['epoch']} | "
            f"train_loss={h['train_loss']:.6f} | "
            f"test_loss={h['test_loss']:.6f} | "
            f"f1_crop={h['f1_crop']:.6f} | "
            f"f1_weed={h['f1_weed']:.6f} | "
            f"f1_soil={h['f1_soil']:.6f} | "
            f"iou_crop={h['iou_crop']:.6f} | "
            f"iou_weed={h['iou_weed']:.6f} | "
            f"iou_soil={h['iou_soil']:.6f} | "
            f"miou={h['miou']:.6f}\n"
        )

print("\nTraining complete.")
print(f"Best weed F1: {best_test_f1_weed:.4f}")
print(f"Model saved to: {os.path.join(OUT_DIR, 'best_model.pth')}")
print(f"Summary saved to: {summary_path}")
