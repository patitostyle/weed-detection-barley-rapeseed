# Weed Segmentation in Barley and Rapeseed Using Multispectral UAV Imagery

Code and reproducibility files for the manuscript:

**From barley to rapeseed: evaluating transfer learning for weed segmentation in UAV multispectral imagery with limited target annotations**

The study evaluates spectral inputs and segmentation models in barley, followed by transfer to rapeseed and adaptation with limited target annotations.

## Repository structure

- `scripts/`: original source-domain training and exploratory rapeseed fine-tuning scripts.
- `revision_controlada/`: additional spatially separated evaluation, paired adaptation subsets, training and evaluation code, archived histories, numerical results and figure-generation scripts.

The original exploratory experiments and the controlled evaluation use different protocols. Their scores should not be interpreted as directly interchangeable.

## Dataset

The processed multispectral UAV patch dataset is available in Mendeley Data:

**DOI:** https://doi.org/10.17632/mb4jvxk9dk.1

The data originate from barley and rapeseed fields in Valdeavero, Madrid, Spain. The full input configuration contains RGB, NIR, red edge, NDVI, NDRE and VARI.

Reference masks contain partial annotations for crop, weed and soil. Unlabelled pixels are excluded from training losses and evaluation metrics. Image patches overlap; patch counts do not represent independent annotated locations.

Refer to the dataset repository for its files and licence.

## Controlled evaluation

The revised experiment compares:

- Full fine-tuning of a barley-trained U-Net.
- Training the same architecture from random initialization.
- Direct transfer without adaptation.

The fixed rapeseed partition contains 269 adaptation-pool patches, 57 validation patches and 97 test patches, with no shared image footprints between subsets.

Nested subsets of 21, 42 and 63 adaptation patches were evaluated using three paired repetitions. Validation data controlled checkpoint selection and early stopping. Final metrics count each annotated geographic pixel once after averaging overlapping predictions.

Full reproduction instructions are provided in:

[revision_controlada/README.md](revision_controlada/README.md)

## Versioned release and source checkpoint

The version associated with the controlled experiment is:

[Release v2.0.0](https://github.com/patitostyle/weed-detection-barley-rapeseed/releases/tag/v2.0.0)

The retained barley initialization checkpoint, `cebada_best_model.pth`, is attached to that release.

Download instructions, provenance and its SHA256 checksum are documented in:

[MODEL_CHECKPOINT.md](revision_controlada/MODEL_CHECKPOINT.md)

The 18 target-domain checkpoints are not included in this repository. They can be regenerated using the documented training procedure.

## Inspecting the reported results

The repository includes:

- Fixed spatial partition lists.
- Nine nested adaptation-subset lists.
- The recorded experimental settings.
- Training histories and selected epochs for all 18 runs.
- Confusion matrices and metrics for the direct-transfer baseline and all trained runs.
- A CSV summary of the reported results.
- Annotation audits and inputs used to prepare the revised figures.

To verify the archived evidence without training or loading the image dataset:

```bash
cd revision_controlada
python code/verify_release.py
```

This verification uses the Python standard library. Training and figure generation require the dependencies described in the reproduction instructions.

## Scope of the results

The controlled experiment evaluates local adaptation within one rapeseed field. Repeated runs share the same fixed test set and do not represent independent field or seasonal replicates.

The source checkpoint was selected using Barley 2 validation. Spatial prediction maps are model-derived scores, not independently validated weed-density measurements or herbicide prescriptions.

## Licence

Source code is distributed under the MIT licence; see [LICENSE](LICENSE).

The dataset and image-derived materials are subject to the terms stated in the Mendeley Data repository.
