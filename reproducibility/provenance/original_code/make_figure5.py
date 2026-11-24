"""Repeated source-model comparison using common unique Barley2 pixels."""
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
VARIANTS = [('UNet', 'U-Net', '#AA3377'),
            ('DeepLabv3_random', 'DeepLabv3\n(random)', '#0072B2'),
            ('DeepLabv3_imagenet', 'DeepLabv3\n(ImageNet)', '#009E73'),
            ('forest_spectral', 'RF\n(spectral)', '#D55E00'),
            ('forest_texture', 'RF\n(local statistics)', '#666666')]


def main():
    base = ROOT / 'staged_v2'
    rows = json.loads((base / 'deeplab_benchmark/test_results.json').read_text())
    rows += [r for r in json.loads((base / 'source_inputs_benchmark/test_results.json').read_text())
             if r['model'] in ['forest_spectral', 'forest_texture']]
    data = {}
    for model, label, color in VARIANTS:
        selected = sorted([r for r in rows if r['model'] == model], key=lambda r: r['seed'])
        assert [r['seed'] for r in selected] == [42, 43, 44]
        metrics = []
        for row in selected:
            cm = np.asarray(row['test']['confusion'], dtype=np.int64)
            assert cm.sum(1).tolist() == [2975858, 1023369, 808269]
            tp = np.diag(cm)
            f1 = 2*tp/np.maximum(cm.sum(0)+cm.sum(1), 1)
            iou = tp/np.maximum(cm.sum(0)+cm.sum(1)-tp, 1)
            values = [float(f1[1]), float(iou[1]), float(iou.mean())]
            assert np.allclose(values, [row['test']['f1'][1], row['test']['iou'][1], row['test']['miou']])
            metrics.append(values)
        a = np.asarray(metrics)
        data[model] = {'seeds': [42, 43, 44], 'metric_order': ['weed_f1', 'weed_iou', 'miou'],
                       'by_seed': a.tolist(), 'mean': a.mean(0).tolist(), 'sample_sd': a.std(0, ddof=1).tolist()}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.titlesize': 11, 'axes.labelsize': 10,
                         'xtick.labelsize': 9, 'ytick.labelsize': 9.5,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.8), sharey=True, layout='constrained')
    for j, (ax, title) in enumerate(zip(axes, ['(a) Weed F1-score', '(b) Weed IoU', '(c) mIoU'])):
        for i, (model, label, color) in enumerate(VARIANTS):
            values = np.asarray(data[model]['by_seed'])[:, j]
            ax.scatter(values, i + np.array([-0.075, 0, 0.075]), color=color, alpha=0.45, s=24, zorder=2)
            ax.errorbar(values.mean(), i, xerr=values.std(ddof=1), marker='D',
                        color=color, capsize=4, markersize=6, linewidth=1.4, zorder=3)
        ax.set_yticks(range(5), [v[1] for v in VARIANTS])
        ax.set_ylim(4.55, -0.55)
        ax.set_xlim(0, 0.8)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
        ax.set_xlabel('Metric value')
        ax.set_title(title, pad=12)
        ax.grid(axis='x', alpha=0.17)
        ax.tick_params(labelleft=(j == 0))
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / 'Figure5.png', dpi=400, facecolor='white')
    fig.savefig(OUT / 'Figure5.svg', facecolor='white')
    plt.close(fig)
    (OUT / 'Figure5_data.json').write_text(json.dumps({
        'models': data, 'reference_class_counts': [2975858, 1023369, 808269],
        'error_bars': 'Sample standard deviations across three source seeds, not confidence intervals.',
        'inputs': 'All variants use the full eight-channel information; RF local-statistics variant adds derived local means and SDs.',
        'evaluation': 'Same unique annotated Barley2-V2 pixels for each variant; retrospective source comparison.'
    }, indent=2), encoding='utf-8')
    print('Saved Figure5.png, Figure5.svg and Figure5_data.json; all15 confusion matrices checked.')


if __name__ == '__main__':
    main()
