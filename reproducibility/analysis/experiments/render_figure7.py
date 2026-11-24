"""Render the verified qualitative prediction arrays without importing PyTorch."""
import json
import os
import sys
from pathlib import Path

if os.name == "nt":
    native = Path(sys.prefix) / "Library" / "bin"
    if native.is_dir():
        os.environ["PATH"] = str(native) + os.pathsep + os.environ.get("PATH", "")

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle

OUT = Path(__file__).resolve().parent / "figures"
COLORS = ["#009E73", "#AA3377", "#B58C42", "#E4E7EB"]


def main():
    arrays = np.load(OUT / "Figure7_arrays.npz")
    metadata = json.loads((OUT / "Figure7_data.json").read_text())
    gsd = metadata["geographic_pixel_size_m"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.fonttype": "none"})
    cmap = ListedColormap(COLORS)
    fig, axes = plt.subplots(2, 4, figsize=(8.4, 5.0))
    titles = ["RGB composite", "Partial reference", "Full prediction",
              "Prediction at\nlabelled pixels"]
    for row, name in enumerate(arrays["names"]):
        panels = [arrays[f"rgb{row}"], arrays[f"reference{row}"],
                  arrays[f"prediction{row}"], arrays[f"labelled_prediction{row}"]]
        for col, panel in enumerate(panels):
            ax = axes[row, col]
            if col == 0:
                ax.imshow(panel, interpolation="nearest")
                length = 5 / gsd
                ax.add_patch(Rectangle((7, 218), length + 12, 30,
                                       facecolor="white", alpha=0.88, linewidth=0))
                ax.plot([13, 13 + length], [241, 241], color="#172b3a", linewidth=2)
                ax.text(13 + length / 2, 227, "5 m", color="#172b3a", fontsize=8,
                        ha="center", va="center")
            else:
                ax.imshow(panel, cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#aab3bc")
                spine.set_linewidth(0.6)
            if row == 0:
                ax.set_title(titles[col], fontsize=10.2, pad=9)
        axes[row, 0].set_ylabel(str(name).removesuffix(".tif"), fontsize=10.2, labelpad=9)
    fig.subplots_adjust(left=0.065, right=0.991, bottom=0.13, top=0.875,
                        hspace=0.12, wspace=0.075)
    handles = [Patch(facecolor=COLORS[i], label=label)
               for i, label in enumerate(["Crop", "Weed", "Soil", "Unlabelled / ignored"])]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.53, 0.024),
               ncol=4, frameon=False, fontsize=10, handlelength=1.4, columnspacing=1.7)
    fig.savefig(OUT / "Figure7.png", dpi=400, facecolor="white")
    fig.savefig(OUT / "Figure7.svg", facecolor="white")
    fig.savefig(OUT / "Figure7_preview.png", dpi=150, facecolor="white")
    plt.close(fig)
    print("Rendered Figure7 from verified archived arrays.", flush=True)


if __name__ == "__main__":
    main()
