"""Dataset class extracted unchanged from original finetune_colza.py."""
import os
import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset

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
