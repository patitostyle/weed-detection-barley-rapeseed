"""Rebuild the rapeseed revision figure from preserved confusion matrices."""
import json
import os
import re
import sys
from pathlib import Path

if os.name == 'nt':
    # Resolve this Conda environment's native libraries without activation.
    library_bin = Path(sys.prefix) / 'Library' / 'bin'
    if library_bin.is_dir():
        os.environ['PATH'] = str(library_bin) + os.pathsep + os.environ.get('PATH', '')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'staged_v2'
OUT = ROOT / 'figures'
GROUPS = [2, 4, 6, 7, 8]
SEEDS = [42, 43, 44, 45, 46]
STYLE = {
    'scratch': ('From scratch', '#D55E00', 's'),
    'full': ('Full fine-tuning', '#0072B2', 'o'),
    'partial': ('Partial fine-tuning (63 patches)', '#009E73', 'D'),
}


def f1(cm):
    cm = np.asarray(cm, dtype=np.int64)
    tp = cm[1, 1]
    return float(2 * tp / max(cm[1].sum() + cm[:, 1].sum(), 1))


def main():
    records = {}
    direct = {}
    for stage in [1, 2, 4]:
        assert json.loads((BASE / f'stage{stage}' / 'status.json').read_text())['status'] == 'complete'
        rows = json.loads((BASE / f'stage{stage}' / 'test_results.json').read_text())
        for row in rows:
            group = row['test_group']
            cm = np.asarray(row['test']['confusion'], dtype=np.int64)
            if row['run'] == 'direct_transfer':
                if group in direct:
                    assert np.array_equal(direct[group], cm)
                direct[group] = cm
                continue
            match = re.fullmatch(r'target_patch_valid_g(\d+)_seed(\d+)_n(\d+)_(scratch|full|partial)', row['run'])
            assert match, row['run']
            g, seed, n, strategy = match.groups()
            assert int(g) == group
            budget = 'complete' if stage == 2 and int(n) > 63 else n
            key = (budget, strategy, group, int(seed))
            assert key not in records
            assert np.isclose(f1(cm), row['test']['f1'][1])
            records[key] = cm
    assert len(records) == 175
    assert set(direct) == set(GROUPS)
    baseline = f1(sum(direct.values(), np.zeros((3, 3), dtype=np.int64)))
    pooled = {}
    grouped = {}
    for strategy in STYLE:
        budgets = ['63'] if strategy == 'partial' else ['21', '63', 'complete']
        for budget in budgets:
            pooled[(budget, strategy)] = np.array([
                f1(sum((records[(budget, strategy, g, s)] for g in GROUPS), np.zeros((3, 3), dtype=np.int64)))
                for s in SEEDS
            ])
        grouped[strategy] = np.array([[f1(records[('63', strategy, g, s)]) for s in SEEDS] for g in GROUPS])
    # All strategies and repetitions use the same annotated geographic support.
    for (_, _, group, _), cm in records.items():
        assert np.array_equal(cm.sum(axis=1), direct[group].sum(axis=1))
    assert direct[4].sum(axis=1).tolist() == [0, 4524, 0]

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.titlesize': 11, 'axes.labelsize': 10,
                         'xtick.labelsize': 9, 'ytick.labelsize': 9,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none'})
    fig, (left, right) = plt.subplots(1, 2, figsize=(8.8, 4.1), layout='constrained')
    jitter = np.linspace(-0.023, 0.023, len(SEEDS))
    positions = np.arange(3)
    for strategy, offset in [('scratch', -0.045), ('full', 0.045)]:
        _, color, marker = STYLE[strategy]
        values = [pooled[(n, strategy)] for n in ['21', '63', 'complete']]
        means = np.array([v.mean() for v in values])
        sds = np.array([v.std(ddof=1) for v in values])
        left.errorbar(positions + offset, means, yerr=sds, color=color,
                      marker=marker, markersize=5, capsize=3, linewidth=1.3, zorder=3)
        for x, vals in zip(positions + offset, values):
            left.scatter(x + jitter, vals, s=13, color=color, alpha=0.38, zorder=2)
    vals = pooled[('63', 'partial')]
    _, color, marker = STYLE['partial']
    left.errorbar(1.14, vals.mean(), yerr=vals.std(ddof=1), color=color,
                  marker=marker, markersize=5, capsize=3, linewidth=1.3, zorder=3)
    left.scatter(1.14 + jitter, vals, s=13, color=color, alpha=0.38, zorder=2)
    left.axhline(baseline, color='#666666', linestyle='--', linewidth=1.1, zorder=1)
    left.set_xticks(positions, ['21', '63', 'Complete pool\n(224–336)'])
    left.set_xlim(-0.25, 2.25)
    left.set_ylim(0, max(v.max() for v in pooled.values()) * 1.18)
    left.set_xlabel('Adaptation budget (patches)')
    left.set_ylabel('Weed F1-score (pooled pixels)')
    left.set_title('(a) Learning curve')
    left.grid(axis='y', alpha=0.18)

    lows, highs = [], []
    for strategy, offset in [('scratch', -0.20), ('full', 0), ('partial', 0.20)]:
        _, color, marker = STYLE[strategy]
        values = grouped[strategy]
        means, sds = values.mean(axis=1), values.std(axis=1, ddof=1)
        lows.extend(means - sds)
        highs.extend(means + sds)
        x = np.arange(len(GROUPS)) + offset
        right.errorbar(x, means, yerr=sds, color=color, marker=marker,
                       linestyle='none', markersize=5, capsize=3, linewidth=1.2, zorder=3)
        for xx, vals in zip(x, values):
            right.scatter(xx + jitter, vals, s=12, color=color, alpha=0.35, zorder=2)
    right.axvspan(0.64, 1.36, color='#777777', alpha=0.07, zorder=0)
    right.set_xticks(np.arange(len(GROUPS)), ['2', '4\n(weed only)', '6', '7', '8'])
    right.set_xlim(-0.45, 4.45)
    right.set_ylim(min(-0.03, min(lows) - 0.025), max(1.03, max(highs) + 0.025))
    right.set_xlabel('Spatial test group')
    right.set_ylabel('Weed F1-score (within group)')
    right.set_title('(b) Spatial variation at 63 patches')
    right.grid(axis='y', alpha=0.18)

    handles = [Line2D([], [], color=c, marker=m, linestyle='none', markersize=5, label=label)
               for label, c, m in STYLE.values()]
    handles.append(Line2D([], [], color='#666666', linestyle='--', label='Direct transfer (panel a)'))
    fig.legend(handles=handles, loc='outside upper center', ncol=2,
               frameon=False, fontsize=9.5, columnspacing=1.8)
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / 'Figure9.png', dpi=400, facecolor='white')
    fig.savefig(OUT / 'Figure9.svg', facecolor='white')
    plt.close(fig)
    data = {'seeds': SEEDS, 'spatial_test_groups': GROUPS, 'direct_transfer_pooled_f1': baseline,
            'pooled_f1_by_seed': {f'{n}_{k}': v.tolist() for (n, k), v in pooled.items()},
            'group_f1_at_63_by_seed': {k: v.tolist() for k, v in grouped.items()},
            'error_bars': 'sample standard deviation across five training repetitions, not confidence intervals',
            'scope': 'one rapeseed field; group4 has only weed reference labels'}
    (OUT / 'Figure9_data.json').write_text(json.dumps(data, indent=2), encoding='utf-8')
    print('Saved Figure9.png, Figure9.svg and Figure9_data.json')


if __name__ == '__main__':
    main()
