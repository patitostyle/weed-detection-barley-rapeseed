# Weed segmentation in barley and rapeseed

Code and reproducibility records for the manuscript **From barley to rapeseed: inconsistent benefits of transfer learning for weed segmentation in UAV multispectral imagery**.

## Current revised study

The current study files are in [`reproducibility/`](reproducibility/README.md). They contain the five-group spatial evaluation, 205 target training runs, repeated spectral and architecture comparisons, exact partitions and adaptation subsets, complete training histories, checkpoint-selection epochs, confusion matrices and figure-generation code.

The processed image patch dataset is available at [Mendeley Data DOI 10.17632/mb4jvxk9dk.1](https://doi.org/10.17632/mb4jvxk9dk.1). Annotations are partial, and unlabelled pixels are excluded from quantitative evaluation. The five target groups are within one rapeseed field, not five independent fields.

The accompanying release `v3.0.0` contains the 224 neural checkpoints and six Random Forest models. Model weights are separate release assets. Their exact filenames, checksums and installation instructions are listed in the [current revision documentation](reproducibility/README.md).

To inspect all records without training or the image dataset:

```powershell
cd reproducibility
python verify_revision.py
```

## Historical material

`scripts/` contains the initial source and exploratory adaptation code. `revision_controlada/` and release `v2.0.0` preserve the earlier three-repetition, single-test-partition experiment. That earlier protocol and its checkpoint-selection history differ from the revised study; its scores should not be combined with the current results.

See the [current revision README](reproducibility/README.md) for experimental scope, data layout, tested environment and explicit bounded reproduction commands. Source code follows the MIT licence; image data follow the Mendeley dataset licence.
