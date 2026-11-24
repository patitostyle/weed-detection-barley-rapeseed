# Reproducibility materials

This directory contains the code and archived numerical evidence for the revised study of weed segmentation and cross-crop adaptation. It supersedes the single-test-cluster, three-repetition protocol in the earlier `revision_controlada/` directory. Earlier releases remain historical records.

## Experimental scope

- Five rapeseed spatial test groups, with no shared patch footprints between adaptation, validation and testing within a fold. All groups belong to one field.
- Primary adaptation budgets of 21, 63 and the complete available pool, with five paired repetitions of scratch versus full fine-tuning: 150 fits.
- Partial fine-tuning at 63 patches: 25 fits. Source-training-fixed normalization at 63 patches: 30 fits over three paired repetitions.
- Nineteen distinct source neural fits, including the source-fixed parent model, and six Random Forest fits. There are 230 distinct fitted models in total; direct transfer adds no target training.
- Source checkpoint selection uses spatially separated Barley 1 validation. Barley 2 is a retrospective evaluation field, not a prospectively untouched test.

## Inspect the evidence without training

Run from this directory:

```powershell
python verify_revision.py
```

Only Python 3.11 or newer and the standard library are needed for this check. It verifies file hashes, all 224 neural selection histories, paired adaptation subsets and common evaluation support. It does not train a model or require the image dataset.

The principal records are:

- `analysis/experiments/checkpoint_epochs/checkpoint_epochs.csv`: selected epoch, epochs actually executed and validation weed F1 for every neural run.
- `analysis/experiments/staged_v2/partitions.json`: exact source partitions and five target folds.
- `analysis/experiments/staged_v2/subsets/`: adaptation pixel-patch subset identities.
- `analysis/experiments/staged_v2/stage1/` through `stage4/`: run metadata, full histories, test confusion matrices and summaries.
- `analysis/experiments/staged_v2/deeplab_benchmark/` and `source_inputs_benchmark/`: repeated source comparisons and their reference support.
- `analysis/experiments/class_proportions/`: class counts among unique annotated pixels.
- `analysis/experiments/figures/`: figure data and plots produced from the archived results.

Each geographic pixel is counted once within its acquisition when computing evaluation metrics. The same position on different acquisition dates remains a separate observation. Unlabelled pixels are ignored, and group 4 lacks annotated crop and soil pixels. Group-based uncertainty intervals are exploratory within-field quantities.

## Image data and environment

The processed patch dataset is available at [Mendeley Data](https://doi.org/10.17632/mb4jvxk9dk.1). Keep the four dataset subdirectories under a `patches/` directory, each with matching `images/` and `masks/` files. Set the parent directory, not the `patches` directory itself:

```powershell
$env:WEED_DATA_DIR = 'C:\datos tfm'
```

The training entry points were run on Windows with Python 3.11 and PyTorch. They retain the original Windows `msvcrt` execution locks. `environment_versions.json` records the tested package versions. Install PyTorch and torchvision for the intended hardware using the [official installation instructions](https://pytorch.org/get-started/locally/), then install the remaining packages from `requirements.txt`. CUDA affects hardware requirements and runtime; it is not required for inspection of the archived records.

The optional checkpoint-report PDF builder also uses ReportLab. GDAL 3.12.4 was used for optional whole-raster geometry and annotation diagnostics. These diagnostics require the original full multiband rasters or vector annotations, which are not included in the public patch dataset. Their archived window locations, matching outputs and quality checks are supplied; the public patch dataset alone does not reproduce every full-orthomosaic diagnostic.

## Trained models

The accompanying `v3.0.0` release assets distribute all 224 neural checkpoints and six Random Forest models separately from the Git source history. `weights_manifest.json` and `weights_manifest.csv` list each model, run identifier, archive, byte size and SHA256 checksum. No optimizer states are needed to inspect or evaluate the retained models.

For target reproduction, first download `weights_source_initializations.zip`, containing the fixed source seed-42 models for the two normalization settings. Install a downloaded archive from this directory:

```powershell
python install_weights.py 'C:\Downloads\weights_source_initializations.zip'
```

Install the other archives the same way if evaluating the released target or source models. The installer verifies archive and model checksums, constrains writes to this directory and refuses to overwrite a differing file. `python verify_revision.py --weights` checks all 230 installed models; it reports missing assets if only a subset has been downloaded.

## Explicit bounded target reproduction

Use a new output directory outside this evidence package. Each command trains only the requested stage and supports resuming that output directory. No automatic full factorial grid is started.

```powershell
python reproduce_target.py --stage 1 --output 'C:\weed_reproduction'
python reproduce_target.py --stage 1 --output 'C:\weed_reproduction' --evaluate
python reproduce_target.py --stage 1 --output 'C:\weed_reproduction' --summarize
```

Repeat the same commands with stages 2, 3 and 4 as needed. Their fit counts are 100, 30 and 25. Summaries for stages 3 and 4 compare with stage 1, so stage 1 must already have been evaluated in the same output directory. Stage 3 reuses the released source-fixed parent checkpoint rather than silently adding another source training run.

## Bounded source comparisons

Source entry points reuse the fixed source U-Net checkpoints exactly as documented in the study. Install the source comparison archive as well as the initialization archive. ImageNet-backbone fitting also requires the official ResNet50 ImageNet1K V2 weights in `analysis/experiments/torch_cache/checkpoints/resnet50-11ad3fa6.pth`; this is the standard torchvision weight file available from `https://download.pytorch.org/models/resnet50-11ad3fa6.pth`. Its expected SHA256 is recorded in the archived DeepLab protocol.

```powershell
python reproduce_source.py --benchmark deeplab --output 'C:\weed_source_deeplab'
python reproduce_source.py --benchmark inputs --output 'C:\weed_source_inputs'
```

The DeepLab comparison fits seven models and reuses source U-Net seeds 42 and 43. The input comparison fits nine spectral U-Nets and six forests, and reuses the archived full-input U-Net evaluations. target reproduction keeps the released seed-42 parent checkpoint fixed. Evaluation can be requested independently with `--evaluate`. Repeated completed fits are skipped by the saved completion records.

## Code provenance

`provenance/original_code/` preserves the analysis scripts as used locally. The packaged executable copies change data-root configuration, allow relocation of path-only protocol fields and disable superseded full-grid command-line entry points. `code_packaging_changes.json` records the hashes of both versions. Model definitions, data partitions, normalization calculations, training hyperparameters and checkpoint-selection rules are unchanged. The helper commands write fresh reproduction outputs outside the archived evidence.

Code follows the repository MIT licence. Dataset and image-derived materials follow the dataset licence. No field-level generalization, independent ground-coordinate validation or validated treatment prescription is established by these experiments.


## Current manuscript figures

The scripts and archived data in `analysis/experiments/` include the updated
location map (`make_figure1.py`), experimental workflow (`make_figure2.py`) and
qualitative predictions (`make_figure7.py` and `render_figure7.py`). Existing
scripts reproduce the grouped split and repeated numerical comparisons.
The qualitative example patch identities are unchanged; revised predictions
use the fixed source seed-42 checkpoint and average overlapping probabilities.
`figures/Figure7_arrays.npz` permits re-rendering without inference or training.

The location map can use the archived projected field geometries in
`cartography/study_fields_utm.json` when the original shapefiles are unavailable.
The regional locator uses Natural Earth Admin 0 country boundaries; their source
and checksum are recorded in `figures/figure1_data.json`. Figure scripts using
OGR require GDAL, listed in the recorded environment.

The unvalidated prediction map was removed from the manuscript. The current
paper has eight figures; its final adaptation chart is stored as `Figure9.png`
and is numbered Figure 8 by LaTeX. It is not the removed prediction map.
