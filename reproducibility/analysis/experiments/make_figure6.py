"""Barley2 U-Net confusion, derived from the three preserved evaluation matrices."""
import json
import os
import sys
from pathlib import Path

if os.name == 'nt':
    native = Path(sys.prefix) / 'Library' / 'bin'
    if native.is_dir():
        os.environ['PATH'] = str(native) + os.pathsep + os.environ.get('PATH', '')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'figures'


def main():
    rows = json.loads((ROOT / 'staged_v2/deeplab_benchmark/test_results.json').read_text())
    rows = sorted([r for r in rows if r['model'] == 'UNet'], key=lambda r: r['seed'])
    assert [r['seed'] for r in rows] == [42, 43, 44]
    cm = np.asarray([r['test']['confusion'] for r in rows], dtype=np.int64)
    reference = cm.sum(axis=2)
    assert np.array_equal(reference, np.tile([2975858, 1023369, 808269], (3, 1)))
    normalized = cm / reference[:, :, None]
    mean = normalized.mean(axis=0) * 100
    sd = normalized.std(axis=0, ddof=1) * 100
    stored = json.loads((ROOT / 'barley_class_metrics.json').read_text())
    assert np.allclose(mean, np.asarray(stored['row_normalized_confusion_mean']) * 100)

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.titlesize': 11, 'axes.labelsize': 10,
                         'xtick.labelsize': 9, 'ytick.labelsize': 9,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none'})
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.0, 3.8),
                                     width_ratios=[1.0, 1.25], layout='constrained')
    image = left.imshow(mean, vmin=0, vmax=100, cmap='Blues')
    classes = ['Crop', 'Weed', 'Soil']
    left.set_xticks(range(3), classes)
    left.set_yticks(range(3), classes)
    left.set_xlabel('Predicted class')
    left.set_ylabel('Reference class')
    left.set_title('(a) Mean row-normalized confusion', pad=12)
    for i in range(3):
        for j in range(3):
            left.text(j, i, f'{mean[i,j]:.1f}\n± {sd[i,j]:.1f}', ha='center', va='center',
                      fontsize=10.5, color='white' if mean[i,j] >= 55 else '#111111')
    left.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    left.set_yticks(np.arange(-0.5, 3, 1), minor=True)
    left.grid(which='minor', color='white', linewidth=1.2)
    left.tick_params(which='minor', length=0)
    for spine in left.spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(image, ax=left, fraction=0.045, pad=0.035, ticks=[0, 25, 50, 75, 100])
    colorbar.ax.set_ylabel('Mean percentage (%)', fontsize=9)
    colorbar.outline.set_visible(False)

    crop_rows = normalized[:, 0, :] * 100
    colors = ['#66C2A5', '#FC8D62', '#B8A58B']
    y = np.arange(3)
    offsets = np.zeros(3)
    for j, (label, color) in enumerate(zip(classes, colors)):
        values = crop_rows[:, j]
        right.barh(y, values, left=offsets, height=0.56, color=color,
                   edgecolor='white', linewidth=0.8, label=f'Predicted {label.lower()}')
        for i, value in enumerate(values):
            right.text(offsets[i] + value/2, y[i], f'{value:.1f}%',
                       ha='center', va='center', fontsize=9.5, color='#111111')
        offsets += values
    assert np.allclose(offsets, 100)
    right.set_xlim(0, 100)
    right.set_xticks([0, 25, 50, 75, 100])
    right.set_yticks(y, ['42', '43', '44'])
    right.invert_yaxis()
    right.set_xlabel('Annotated crop pixels (%)')
    right.set_ylabel('Training seed')
    right.set_title('(b) Predictions for crop reference pixels', pad=12)
    right.grid(axis='x', alpha=0.14)
    right.set_axisbelow(True)
    right.legend(loc='upper center', bbox_to_anchor=(0.5, -0.23),
                 ncol=3, frameon=False, fontsize=8.5, columnspacing=0.9,
                 handlelength=1.1)
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / 'Figure6.png', dpi=400, facecolor='white')
    fig.savefig(OUT / 'Figure6.svg', facecolor='white')
    plt.close(fig)
    (OUT / 'Figure6_data.json').write_text(json.dumps({
        'seeds': [42, 43, 44], 'class_order': classes,
        'reference_class_counts': reference[0].tolist(),
        'confusion_matrices': cm.tolist(), 'row_normalized_percent_mean': mean.tolist(),
        'row_normalized_percent_sample_sd': sd.tolist(),
        'crop_classification_percent_by_seed': crop_rows.tolist(),
        'scope': 'Same unique annotated Barley2 pixels; sample SD across training seeds, not across fields.'
    }, indent=2), encoding='utf-8')
    print('Saved Figure6.png, Figure6.svg and Figure6_data.json; reference counts verified.')


if __name__ == '__main__':
    main()
