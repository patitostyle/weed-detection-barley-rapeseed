"""
finetune_colza.py

Few-shot fine-tuning of U-Net pre-trained on barley for weed detection in rapeseed.

Usage:
    python finetune_colza.py --baseline
    python finetune_colza.py --experiment 5pct --freeze_encoder
    python finetune_colza.py --experiment 10pct --freeze_encoder
    python finetune_colza.py --experiment 15pct --freeze_encoder
    python finetune_colza.py --experiment 15pct
"""

import os
import sys
import glob
import random
import argparse
import time
import numpy as np
import tifffile
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

# =========================================================
# CONFIGURATION
# =========================================================

# Base directory for data. Change this to match your setup.
BASE_DIR = os.environ.get("WEED_DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "data"))

PATCHES_DIR = os.path.join(BASE_DIR, "patches", "colza_v2_rgb_nir_re_indices")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUT_DIR = os.path.join(BASE_DIR, "results", "finetune_colza")
os.makedirs(OUT_DIR, exist_ok=True)

SPLIT_DIR = os.path.join(BASE_DIR, "splits")
TRAIN_IDS = {
    "5pct": os.path.join(SPLIT_DIR, "colza_train_5pct.txt"),
    "10pct": os.path.join(SPLIT_DIR, "colza_train_10pct.txt"),
    "15pct": os.path.join(SPLIT_DIR, "colza_train_15pct.txt"),
}

MODEL_BASE = os.path.join(MODEL_DIR, "cebada_best_model.pth")

BATCH_SIZE = 8
EPOCHS = 30
PATIENCE = 7
MIN_DELTA = 1e-4
LR_PARTIAL = 1e-4
LR_FULL = 5e-5
SEED = 42
NUM_CLASSES = 3
NUM_WORKERS = 0  # 0 to avoid issues on Windows
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_WEIGHTS = torch.tensor([1.0, 3.0, 1.5]).to(DEVICE)

# =========================================================
# SEED
# =========================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

# =========================================================
# DATASET (using tifffile, no rasterio)
# =========================================================

class ColzaPatchDataset(Dataset):
    def __init__(self, patch_ids, images_dir, masks_dir):
        self.patch_ids = patch_ids
        self.images_dir = images_dir
        self.masks_dir = masks_dir

    def __len__(self):
        return len(self.patch_ids)

    def __getitem__(self, idx):
        pid = self.patch_ids[idx]
        img_path = os.path.join(self.images_dir, pid)
        mask_path = os.path.join(self.masks_dir, pid)

        # Read image with tifffile -> (H, W, C)
        img = tifffile.imread(img_path).astype(np.float32)
        # Convert to (C, H, W)
        if img.ndim == 3 and img.shape[2] == 8:
            img = img.transpose(2, 0, 1)

        # Read mask -> (H, W)
        mask = tifffile.imread(mask_path).astype(np.int64)

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

        # Remap classes: 1->crop (0), 2->weed (1), 3->soil (2)
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
    f1s, ious = [], []
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
        f1s.append(f1)
        ious.append(iou)
    return f1s, ious

# =========================================================
# U-NET ARCHITECTURE
# =========================================================

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
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
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = ConvBlock(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = ConvBlock(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = ConvBlock(128, 64)
        self.final = nn.Conv2d(64, num_classes, 1)

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


def freeze_encoder(model):
    for name, param in model.named_parameters():
        if name.startswith(("enc1", "enc2", "enc3", "pool1", "pool2", "pool3")):
            param.requires_grad = False

# =========================================================
# EXPERIMENT EXECUTION
# =========================================================

def run_experiment(exp_name, freeze):
    train_ids_path = TRAIN_IDS[exp_name]
    with open(train_ids_path) as f:
        train_raw = [line.strip() for line in f.readlines()]
    train_ids = [pid + ".tif" if not pid.endswith(".tif") else pid for pid in train_raw]

    all_files = glob.glob(os.path.join(PATCHES_DIR, "masks", "*.tif"))
    all_ids = [os.path.basename(f) for f in all_files]
    test_ids = [pid for pid in all_ids if pid not in train_ids]

    print(f"\n=== {exp_name} (encoder {'frozen' if freeze else 'trainable'}) ===")
    print(f"Train patches: {len(train_ids)} | Test patches: {len(test_ids)}")

    train_ds = ColzaPatchDataset(train_ids, os.path.join(PATCHES_DIR, "images"), os.path.join(PATCHES_DIR, "masks"))
    test_ds = ColzaPatchDataset(test_ids, os.path.join(PATCHES_DIR, "images"), os.path.join(PATCHES_DIR, "masks"))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)

    model = UNet(in_channels=8, num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_BASE, map_location=DEVICE))
    if freeze:
        freeze_encoder(model)

    lr = LR_PARTIAL if freeze else LR_FULL
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS, ignore_index=255)

    use_amp = (DEVICE == "cuda")
    scaler = torch.amp.GradScaler(enabled=use_amp)

    best_f1 = -1
    best_iou = -1
    best_miou = -1
    best_epoch = -1
    epochs_without_improvement = 0

    for epoch in range(EPOCHS):
        start_time = time.time()

        model.train()
        train_loss = 0.0
        for imgs, masks in train_loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=DEVICE, enabled=use_amp):
                outputs = model(imgs)
                loss = criterion(outputs, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for imgs, masks in test_loader:
                imgs = imgs.to(DEVICE, non_blocking=True)
                masks = masks.to(DEVICE, non_blocking=True)
                with torch.amp.autocast(device_type=DEVICE, enabled=use_amp):
                    outputs = model(imgs)
                preds = torch.argmax(outputs, dim=1)
                all_preds.append(preds.cpu())
                all_targets.append(masks.cpu())
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        f1s, ious = compute_metrics(all_preds, all_targets)
        f1_mal = f1s[1]
        iou_mal = ious[1]
        miou = (ious[0] + ious[1] + ious[2]) / 3

        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1:2d} | Time {epoch_time:.1f}s | Loss {train_loss:.4f} | Weed F1 {f1_mal:.4f} | Weed IoU {iou_mal:.4f} | mIoU {miou:.4f}")

        if f1_mal > best_f1 + MIN_DELTA:
            best_f1 = f1_mal
            best_iou = iou_mal
            best_miou = miou
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            suffix = "frozen" if freeze else "full"
            torch.save(model.state_dict(), os.path.join(OUT_DIR, f"best_{exp_name}_{suffix}.pth"))
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print(f"⚠️ Early stopping at epoch {epoch+1} (no improvement for {PATIENCE} epochs)")
            break

    suffix = "frozen" if freeze else "full"
    with open(os.path.join(OUT_DIR, f"summary_{exp_name}_{suffix}.txt"), "w") as f:
        f.write(f"Experiment: {exp_name} ({suffix})\n")
        f.write(f"Train patches: {len(train_ids)}\nTest patches: {len(test_ids)}\n")
        f.write(f"Epochs trained: {epoch+1}\n")
        f.write(f"Best weed F1: {best_f1:.6f} (epoch {best_epoch})\n")
        f.write(f"Best weed IoU: {best_iou:.6f}\n")
        f.write(f"Best mIoU: {best_miou:.6f}\n")
    print(f"✅ Summary saved to {OUT_DIR}/summary_{exp_name}_{suffix}.txt")

# =========================================================
# BASELINE (0%)
# =========================================================

def baseline():
    print("\n=== Baseline (0%) ===")
    all_files = glob.glob(os.path.join(PATCHES_DIR, "masks", "*.tif"))
    all_ids = [os.path.basename(f) for f in all_files]
    ds = ColzaPatchDataset(all_ids, os.path.join(PATCHES_DIR, "images"), os.path.join(PATCHES_DIR, "masks"))
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)
    model = UNet(in_channels=8, num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_BASE, map_location=DEVICE))
    model.eval()
    all_preds, all_targets = [], []
    use_amp = (DEVICE == "cuda")
    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)
            with torch.amp.autocast(device_type=DEVICE, enabled=use_amp):
                outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu())
            all_targets.append(masks.cpu())
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    f1s, ious = compute_metrics(all_preds, all_targets)
    miou = (ious[0] + ious[1] + ious[2]) / 3
    print(f"Weed F1: {f1s[1]:.4f} | Weed IoU: {ious[1]:.4f} | mIoU: {miou:.4f}")
    with open(os.path.join(OUT_DIR, "summary_0pct.txt"), "w") as f:
        f.write("Baseline (0%)\n")
        f.write(f"Weed F1: {f1s[1]:.6f}\n")
        f.write(f"Weed IoU: {ious[1]:.6f}\n")
        f.write(f"mIoU: {miou:.6f}\n")

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Few-shot fine-tuning of U-Net on rapeseed")
    parser.add_argument("--experiment", choices=["5pct", "10pct", "15pct"],
                        help="Experiment to run (5pct, 10pct, 15pct)")
    parser.add_argument("--freeze_encoder", action="store_true",
                        help="Freeze the encoder (partial fine-tuning)")
    parser.add_argument("--baseline", action="store_true",
                        help="Run baseline (0%) evaluation")
    args = parser.parse_args()

    set_seed(SEED)

    if args.baseline:
        baseline()
    elif args.experiment:
        run_experiment(args.experiment, args.freeze_encoder)
    else:
        print("Usage: python finetune_colza.py --baseline")
        print("   or: python finetune_colza.py --experiment 5pct/10pct/15pct [--freeze_encoder]")
