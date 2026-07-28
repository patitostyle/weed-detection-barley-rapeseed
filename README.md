# Weed Detection in Barley and Rapeseed using Multispectral UAV Imagery

This repository contains the Python scripts used for training, evaluation, and few-shot fine-tuning of semantic segmentation models for weed detection in barley (*Hordeum vulgare*) and rapeseed (*Brassica napus*) using multispectral UAV imagery.

> 📄 Companion manuscript: *[title]*, submitted to *Computers and Electronics in Agriculture* (under review).
> 📦 Data: [Mendeley Data — DOI: 10.17632/mb4jvxk9dk.1](https://data.mendeley.com/datasets/mb4jvxk9dk/1)

---

## Overview

The study evaluates:

- Four spectral configurations (RGB, RGB+NIR+RedEdge, RGB+vegetation indices, and full multispectral)
- Three modelling approaches (U-Net, DeepLabv3, and Random Forest)
- Spatial generalisation to an independent barley field
- Cross-crop transfer to rapeseed
- Few-shot fine-tuning for domain adaptation with limited labelled data

## Repository structure

```
weed-detection-barley-rapeseed/
├── scripts/
│   ├── generate_patches.py    # Patch generation (256×256, 50% overlap) from orthomosaics and masks
│   ├── train_unet_spatial.py  # U-Net training: spectral configs + temporal/spatial validation
│   ├── train_deeplabv3.py     # DeepLabv3 training (architecture comparison)
│   ├── train_random_forest.py # Random Forest baseline (architecture comparison)
│   ├── finetune_colza.py      # Few-shot fine-tuning on rapeseed
│   └── requirements.txt
├── LICENSE
└── README.md
```

| Script | Reproduces | Manuscript section |
|---|---|---|
| `generate_patches.py` | Patch dataset construction | Section 2.3 |
| `train_unet_spatial.py` | Spectral configuration comparison, temporal & spatial validation | Section 2.4/2.5, Table 1, Figures 3, 6, 7 |
| `train_deeplabv3.py` | Architecture comparison | Section 2.5, Table 2, Figure 4 |
| `train_random_forest.py` | Architecture comparison | Section 2.5, Table 2, Figure 4 |
| `finetune_colza.py` | Few-shot domain adaptation to rapeseed | Section 2.7, Figure 8 |

> **Note:** spectral indices (NDVI, NDRE, VARI) were computed in ArcGIS Pro using standard band math (formulas detailed in Section 2.2 of the manuscript), not via scripts in this repository. Evaluation metrics (F1-score, IoU, mIoU) are computed inline within each training script rather than in a separate module.

## Data availability

The UAV image patches and reference masks used in this study are publicly available at Mendeley Data:

**DOI: [10.17632/mb4jvxk9dk.1](https://data.mendeley.com/datasets/mb4jvxk9dk/1)**

This repository does not include raw data. Scripts expect the patch files and patch-ID lists (train/test splits for the few-shot fine-tuning experiments, 21/42/63 patches) described in the linked dataset.

## Requirements

- Python 3.10+
- PyTorch 2.0+
- torchvision

Install with:

```bash
git clone https://github.com/patitostyle/weed-detection-barley-rapeseed
cd weed-detection-barley-rapeseed
pip install -r scripts/requirements.txt
```

## Usage

### 1. Generate patches from orthomosaics and masks

```bash
python scripts/generate_patches.py --data_dir <path_to_mendeley_data> --patch_size 256 --stride 128
```

### 2. Train U-Net (spectral configuration comparison + spatial validation)

```bash
python scripts/train_unet_spatial.py --config full --data_dir <path>
```

Reproduces: Table 1, Figures 3, 6, and 7.

### 3. Train DeepLabv3 / Random Forest (architecture comparison)

```bash
python scripts/train_deeplabv3.py --data_dir <path>
python scripts/train_random_forest.py --data_dir <path>
```

Reproduces: Table 2, Figure 4.

### 4. Few-shot fine-tuning on rapeseed

```bash
python scripts/finetune_colza.py --n_patches 63 --mode full --data_dir <path>
```

Reproduces: Figure 8.

## Citation

If you use this code or the associated dataset, please cite:

```
[Authors] ([Year]). From barley to rapeseed: few-shot fine-tuning of semantic
segmentation for site-specific weed management. Computers and Electronics in
Agriculture. [DOI once assigned]
```

Dataset citation:

```
[Authors] ([Year]). Weed Detection in Barley and Rapeseed using Multispectral
UAV Imagery [Data set]. Mendeley Data. https://doi.org/10.17632/mb4jvxk9dk.1
```

## License

See [LICENSE](LICENSE) for details.
