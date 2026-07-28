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
