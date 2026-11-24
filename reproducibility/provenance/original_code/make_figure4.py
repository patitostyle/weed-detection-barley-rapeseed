"""Repeated U-Net input comparisons on shared unique Barley2 evaluation pixels."""
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
CONFIGS = [('unet_rgb', 'RGB\n(3 ch.)', '#666666'),
           ('unet_rgb_nir_re', 'RGB + NIR + RE\n(5 ch.)', '#0072B2'),
           ('unet_rgb_indices', 'RGB + VI\n(6 ch.)', '#009E73'),
           ('unet_full', 'Full\n(8 ch.)', '#AA3377')]


def main():
    raw = json.loads((ROOT / 'staged_v2/source_inputs_benchmark/test_results.json').read_text())
    data = {}
    for model, label, color in CONFIGS:
        rows = sorted([r for r in raw if r['model'] == model], key=lambda r: r['seed'])
        assert [r['seed'] for r in rows] == [42, 43, 44]
        metrics = []
        for row in rows:
            cm = np.asarray(row['test']['confusion'], dtype=np.int64)
            assert cm.sum(axis=1).tolist() == [2975858, 1023369, 808269]
            tp = np.diag(cm)
            f1 = 2 * tp / np.maximum(cm.sum(0) + cm.sum(1), 1)
            iou = tp / np.maximum(cm.sum(0) + cm.sum(1) - tp, 1)
            assert np.isclose(f1[1], row['test']['f1'][1])
            assert np.isclose(iou[1], row['test']['iou'][1])
            assert np.isclose(iou.mean(), row['test']['miou'])
            metrics.append([float(f1[1]), float(iou[1]), float(iou.mean())])
        a = np.asarray(metrics)
        data[model] = {'seeds': [42, 43, 44], 'metric_order': ['weed_f1', 'weed_iou', 'miou'],
                       'by_seed': a.tolist(), 'mean': a.mean(0).tolist(),
                       'sample_sd': a.std(0, ddof=1).tolist()}

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.titlesize': 11, 'axes.labelsize': 10,
                         'xtick.labelsize': 9, 'ytick.labelsize': 9,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), sharey=True, layout='constrained')
    for j, (ax, title) in enumerate(zip(axes, ['(a) Weed F1-score', '(b) Weed IoU'])):
        for i, (model, label, color) in enumerate(CONFIGS):
            values = np.asarray(data[model]['by_seed'])[:, j]
            ax.scatter(i + np.array([-0.065, 0, 0.065]), values, s=23,
                       color=color, alpha=0.45, zorder=2)
            ax.errorbar(i, values.mean(), yerr=values.std(ddof=1), marker='D',
                        markersize=6, color=color, capsize=4, linewidth=1.4, zorder=3)
        ax.set_xticks(range(4), [x[1] for x in CONFIGS])
        ax.set_xlim(-0.45, 3.45)
        ax.set_ylim(0, 0.75)
        ax.set_xlabel('Input configuration')
        ax.set_title(title, pad=12)
        ax.grid(axis='y', alpha=0.17)
        ax.tick_params(labelleft=True)
    axes[0].set_ylabel('Weed F1-score')
    axes[1].set_ylabel('Weed IoU')
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / 'Figure4.png', dpi=400, facecolor='white')
    fig.savefig(OUT / 'Figure4.svg', facecolor='white')
    plt.close(fig)
    (OUT / 'Figure4_data.json').write_text(json.dumps({
        'configurations': data,
        'reference_class_counts': [2975858, 1023369, 808269],
        'error_bars': 'Sample standard deviation across three source training seeds, not confidence intervals.',
        'evaluation': 'Retrospective Barley2-V2; no model selection using these evaluation scores.'
    }, indent=2), encoding='utf-8')
    print('Saved Figure4.png, Figure4.svg and Figure4_data.json; all12 confusion matrices checked.')


if __name__ == '__main__':
    main()
