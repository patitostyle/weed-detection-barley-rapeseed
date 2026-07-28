# Weed Detection in Barley and Rapeseed using Multispectral UAV Imagery

This repository contains the Python scripts used for training, evaluation, and few-shot fine-tuning of semantic segmentation models for weed detection in barley (*Hordeum vulgare*) and rapeseed (*Brassica napus*) using multispectral UAV imagery.

## Overview

The study evaluates:

- Four spectral configurations (RGB, RGB+NIR+RedEdge, RGB+vegetation indices, and full multispectral)
- Three modelling approaches (U‑Net, DeepLabv3+, and Random Forest)
- Spatial generalisation to an independent barley field
- Cross‑crop transfer to rapeseed
- Few‑shot fine‑tuning for domain adaptation with limited labelled data

## Repository structure
📁 weed-detection-barley-rapeseed/
│
├── 📁 scripts/
│ ├── finetune_colza.py # Few-shot fine-tuning on rapeseed
│ └── train_unet_spatial.py # Train U‑Net on Barley 1 (V1+V2) and evaluate on Barley 2
│
├── README.md
├── requirements.txt
└── LICENSE

text

## Requirements

- Python 3.10+
- PyTorch 2.0+
- torchvision
- numpy
- matplotlib
- scikit-learn
- joblib
- tifffile (or rasterio)

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/weed-detection-barley-rapeseed.git
cd weed-detection-barley-rapeseed
pip install -r requirements.txt
Dataset
The processed patch datasets used in this study are available in the Mendeley Data repository:

DOI: 10.17632/mb4jvxk9dk.1

URL: https://data.mendeley.com/datasets/mb4jvxk9dk

The dataset includes:

726 patches from Barley 1‑V1 (training)

524 patches from Barley 1‑V2 (temporal validation)

Patches from Barley 2‑V2 (spatial validation)

423 patches from Rapeseed‑V2 (cross‑crop transfer and fine‑tuning)

Text files defining training splits (5%, 10%, 15%) for few‑shot fine‑tuning

Each patch is 256×256 pixels with 8 spectral bands: R, G, B, NIR, RedEdge, NDVI, NDRE, and VARI.

Usage
Train the base U‑Net model (spatial validation)
bash
python scripts/train_unet_spatial.py
This will generate the best_model.pth file used as the starting point for fine‑tuning.

Fine‑tuning on rapeseed
bash
python scripts/finetune_colza.py --baseline
python scripts/finetune_colza.py --experiment 5pct --freeze_encoder
python scripts/finetune_colza.py --experiment 10pct --freeze_encoder
python scripts/finetune_colza.py --experiment 15pct --freeze_encoder
python scripts/finetune_colza.py --experiment 15pct
License
This project is licensed under the MIT License.

Citation
If you use this code or the associated dataset, please cite:

text
Hernández Ludeña, P. A., Fernández Piñar, C., López de Herrera, J., Herrero Tejedor, T. R., Pérez Martin, E., & Calderón, J. (2026). From barley to rapeseed: few‑shot fine‑tuning of semantic segmentation models for weed detection using UAV multispectral imagery. *Computers and Electronics in Agriculture*. [DOI to be added]
Contact
Patricio Alonso Hernández Ludeña – patricio.hernandez@alumnos.upm.es

Project Link: https://github.com/yourusername/weed-detection-barley-rapeseed

text

---

## 🔧 Recuerda cambiar `yourusername` por `patitstyle`

Cuando copies este README en tu repositorio, reemplaza `yourusername` con `patitstyle` en los enlaces. Por ejemplo:
https://github.com/patitstyle/weed-detection-barley-rapeseed

text

---

## ✅ Ya lo tienes todo listo

| Elemento | Estado |
|----------|--------|
| Scripts | ✅ Subidos |
| README.md | ✅ Ya tienes el contenido completo |
| requirements.txt | ✅ Subido |
| LICENSE | ✅ Subido |
| Repositorio en GitHub | ✅ Completado |
| Enlace a Mendeley Data | ✅ Pendiente (añadir "Related links") |

---

**¿Quieres que te ayude con la Data Availability Statement final para el manuscrito?** 🚀
