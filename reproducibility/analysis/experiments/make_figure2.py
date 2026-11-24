"""Draw the revised experimental workflow from the documented study protocol.

This schematic contains no simulated or AI-generated scientific observations.
The saved manuscript and preserved experiment settings determine its labels.
It writes replacement assets locally without changing the author's manuscript.
"""

import json
import os
import sys
from pathlib import Path

if os.name == "nt":
    native = Path(sys.prefix) / "Library" / "bin"
    if native.is_dir():
        os.environ["PATH"] = str(native) + os.pathsep + os.environ.get("PATH", "")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures"
TEXT = "#172b3a"
EDGE = "#7892a3"
ARROW = "#536b7c"


def draw_workflow():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5,
                         "svg.fonttype": "none"})
    fig = plt.figure(figsize=(7.8, 7.4), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    texts = []
    boxes = []

    def box(x, y, w, h, title, lines, fill="#f3f7fa", size=10.2):
        patch = FancyBboxPatch((x, y), w, h,
                              boxstyle="round,pad=0.005,rounding_size=0.007",
                              facecolor=fill, edgecolor=EDGE, linewidth=0.9,
                              zorder=2)
        ax.add_patch(patch)
        boxes.append((x, y, w, h, patch))
        # Allocate one equal-height row to the heading and each body line.
        rows = [title] + lines
        step = h / (len(rows) + 0.7)
        first_y = y + h - 0.8 * step
        for i, line in enumerate(rows):
            t = ax.text(x + w / 2, first_y - i * step, line,
                        ha="center", va="center", color=TEXT,
                        fontsize=size + (0.3 if i == 0 else 0),
                        fontweight="bold" if i == 0 else "normal", zorder=3)
            texts.append((t, patch))

    def arrow(start, end, color=ARROW):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                                    mutation_scale=10, linewidth=1,
                                    color=color, shrinkA=2, shrinkB=2,
                                    zorder=1))

    ax.text(0.025, 0.976, "A  Data preparation", color=TEXT,
            fontsize=12.4, fontweight="bold", va="center")
    box(0.025, 0.837, 0.296, 0.112, "Multispectral products",
        ["Five bands + three indices", "RGB from the same band products"],
        size=9.4)
    box(0.352, 0.837, 0.296, 0.112, "Partial reference masks",
        ["Crop / weed / soil", "Unlabelled pixels ignored"], size=9.6)
    box(0.679, 0.837, 0.296, 0.112, "Paired image–mask patches",
        ["256 × 256 pixels; stride 128", "Primary scaling: patch-wise"], size=9.4)
    arrow((0.326, 0.893), (0.347, 0.893))
    arrow((0.653, 0.893), (0.674, 0.893))

    # The two data branches describe separate source and target datasets.
    ax.text(0.025, 0.785, "B  Source-domain experiments", color=TEXT,
            fontsize=11.5, fontweight="bold", va="center")
    ax.text(0.535, 0.785, "C  Rapeseed experiments", color=TEXT,
            fontsize=11.5, fontweight="bold", va="center")

    box(0.025, 0.663, 0.44, 0.078, "Barley 1: V1 + V2",
        ["Spatial split: 980 train / 215 validation"], size=10.2)
    box(0.025, 0.493, 0.44, 0.136, "Source comparisons: 3 seeds",
        ["U-Net: four spectral configurations",
         "DeepLabv3: random / ImageNet backbone",
         "Random Forest: spectral / local statistics"], size=10.0)
    box(0.025, 0.365, 0.44, 0.094, "Checkpoint selection",
        ["Barley 1 validation only",
         "Transfer reference: 8-channel U-Net, seed 42"], size=9.9)
    box(0.025, 0.247, 0.44, 0.084, "Barley 2-V2",
        ["Retrospective cross-field evaluation",
         "Excluded from checkpoint selection"], size=10.1)
    for y0, y1 in [(0.658, 0.634), (0.488, 0.464), (0.360, 0.336)]:
        arrow((0.245, y0), (0.245, y1))

    box(0.535, 0.663, 0.44, 0.078, "Spatial design",
        ["9 groups; 5 held-out test groups",
         "Separate adaptation / validation / test"], size=9.8)
    box(0.535, 0.438, 0.44, 0.191, "Primary experiments",
        ["Direct transfer: no parameter updates",
         "From scratch vs full fine-tuning",
         "21 / 63 / complete-pool patches",
         "Training: 5 paired repetitions per budget"], size=10.0)
    box(0.535, 0.247, 0.44, 0.157, "Complementary experiments",
        ["63 adaptation patches per spatial fold",
         "Partial fine-tuning: 5 paired repetitions",
         "Source-fixed scaling: 3 paired repetitions",
         "Matched source retrained for fixed scaling"], size=9.6)
    arrow((0.755, 0.658), (0.755, 0.634))
    arrow((0.755, 0.433), (0.755, 0.409))

    # Source initialization branches BEFORE the Barley 2 evaluation. It never
    # passes through Barley 2, avoiding an implication of test-based selection.
    ax.plot([0.470, 0.500, 0.500], [0.405, 0.405, 0.520],
            color=ARROW, linewidth=1, zorder=1)
    arrow((0.500, 0.520), (0.530, 0.520))

    box(0.025, 0.039, 0.95, 0.151, "Evaluation and paired comparisons",
        ["Average overlapping probabilities; count each labelled pixel once per acquisition",
         "Source: identical annotated evaluation pixels for all models",
         "Target: pooled metrics by seed; group-weighted paired differences",
         "Exploratory 95% intervals within one rapeseed field"],
        fill="#e8f2f5", size=10.0)
    arrow((0.245, 0.242), (0.245, 0.195))
    arrow((0.755, 0.242), (0.755, 0.195))

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    problems = []
    for text, patch in texts:
        tb = text.get_window_extent(renderer)
        pb = patch.get_window_extent(renderer)
        if tb.x0 < pb.x0 + 2 or tb.x1 > pb.x1 - 2 or tb.y0 < pb.y0 + 2 or tb.y1 > pb.y1 - 2:
            problems.append(text.get_text())
    if problems:
        raise RuntimeError("Workflow text exceeds its box: " + repr(problems))

    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "figure2.png", dpi=400, facecolor="white")
    fig.savefig(OUT / "figure2.svg", facecolor="white")
    fig.savefig(OUT / "figure2_preview.png", dpi=150, facecolor="white")
    plt.close(fig)
    (OUT / "figure2_protocol.json").write_text(json.dumps({
        "source_train_patches": 980,
        "source_validation_patches": 215,
        "source_seeds": [42, 43, 44],
        "source_checkpoint_selection": "Barley 1 validation only",
        "barley_2_role": "retrospective evaluation, not checkpoint selection",
        "primary_transfer_reference": "eight-channel U-Net, source seed42",
        "target_test_groups": [2, 4, 6, 7, 8],
        "primary_budgets": [21, 63, "complete pool, 224--336 patches"],
        "primary_seeds": [42, 43, 44, 45, 46],
        "partial_fine_tuning_budget": 63,
        "partial_fine_tuning_seeds": [42, 43, 44, 45, 46],
        "normalization_budget": 63,
        "normalization_seeds": [42, 43, 44],
        "normalization_source": "corresponding source model retrained under fixed bounds",
        "unlabelled_pixels": "ignored",
        "overlap_evaluation": "mean class probabilities, then one decision per geographic pixel",
        "image_size_inches": [7.8, 7.4],
        "raster_dpi": 400,
        "text_box_containment_check": "passed",
        "original_manuscript_modified": False
    }, indent=2), encoding="utf-8")
    print("Saved figure2.png, figure2.svg and figure2_protocol.json; box containment passed.")


if __name__ == "__main__":
    draw_workflow()
