"""Map retained rapeseed patch groups and the actual five evaluation folds."""
import json
import os
import sys
from pathlib import Path

if os.name == 'nt':
    library_bin = Path(sys.prefix) / 'Library' / 'bin'
    if library_bin.is_dir():
        os.environ['PATH'] = str(library_bin) + os.pathsep + os.environ.get('PATH', '')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
OUT = ROOT / 'figures'
COLORS = {'A': '#D9DEE3', 'V': '#56B4E9', 'T': '#E69F00'}


def main():
    records = json.loads((WORK / 'auditoria_datos_tfm.json').read_text())['colza_v2_rgb_nir_re_indices']['records']
    groups = json.loads((WORK / 'viabilidad_colza.json').read_text())['groups_14m']
    folds = json.loads((ROOT / 'staged_v2' / 'partitions.json').read_text())['target_folds']
    lookup = {r['id']: r for r in records}
    membership = {g['group']: set(g['ids']) for g in groups}
    assert len(records) == 423 and len(groups) == 9 and len(folds) == 5
    assert sum(map(len, membership.values())) == len(set.union(*membership.values())) == len(records)
    assert set.union(*membership.values()) == set(lookup)
    assert [g['group'] for g in groups if g['weed_patches']] == [2, 4, 6, 7, 8]
    x0 = min(r['x'] for r in records)
    y0 = min(r['y'] - r['h'] for r in records)

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.titlesize': 11, 'axes.labelsize': 10,
                         'xtick.labelsize': 9, 'ytick.labelsize': 9,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none'})
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.2, 4.4),
                                     width_ratios=[1.05, 1.15], layout='constrained')
    rectangles = [Rectangle((r['x'] - x0, r['y'] - r['h'] - y0), r['w'], r['h']) for r in records]
    left.add_collection(PatchCollection(rectangles, facecolor='#E5E9ED', edgecolor='#BAC3CB', linewidth=0.25))
    weed = [r for r in records if int(r['counts'].get('2', 0)) > 0]
    assert len(weed) == sum(g['weed_patches'] for g in groups) == 80
    left.scatter([r['x'] + r['w']/2 - x0 for r in weed],
                 [r['y'] - r['h']/2 - y0 for r in weed],
                 marker='o', s=9, c='#B2182B', edgecolors='white', linewidths=0.2, zorder=3)
    for g in groups:
        rr = [lookup[n] for n in g['ids']]
        assert len(rr) == g['patches']
        lo = min(r['x'] for r in rr)
        hi = max(r['x'] + r['w'] for r in rr)
        top = max(r['y'] for r in rr)
        label_y = top - y0 + 7
        label_box = None
        if g['group'] == 2:
            bottom = min(r['y'] - r['h'] for r in rr)
            label_y = (bottom + top)/2 - y0
            label_box = {'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.95, 'pad': 1}
        left.text((lo+hi)/2 - x0, label_y, str(g['group']),
                  ha='center', va='bottom', fontsize=10, fontweight='bold', bbox=label_box)
    left.set_xlim(-12, max(r['x'] + r['w'] - x0 for r in records) + 12)
    left.set_ylim(-12, max(r['y'] - y0 for r in records) + 30)
    left.set_aspect('equal', adjustable='box')
    left.set_xlabel('Relative easting (m)')
    left.set_ylabel('Relative northing (m)')
    left.set_title('(a) Retained patch groups')
    left.grid(alpha=0.12)
    left.legend(handles=[Patch(facecolor='#E5E9ED', edgecolor='#BAC3CB', label='Patch footprint'),
                         Line2D([], [], color='#B2182B', marker='o', linestyle='none', markersize=4,
                                label='Patch with weed annotations')],
                loc='lower left', frameon=False, fontsize=8.5)

    evidence = []
    for i, fold in enumerate(folds):
        test, val = fold['test_group'], fold['validation_group']
        assert set(fold['test']) == membership[test]
        assert set(fold['validation']) == membership[val]
        expected_train = set.union(*(v for g, v in membership.items() if g not in [test, val]))
        assert set(fold['training']) == expected_train
        roles = {}
        for g in range(9):
            role = 'T' if g == test else 'V' if g == val else 'A'
            roles[g] = role
            right.add_patch(Rectangle((g-0.47, i-0.39), 0.94, 0.78,
                                      facecolor=COLORS[role], edgecolor='white', linewidth=0.4))
            right.text(g, i, role, ha='center', va='center', fontsize=9.5,
                       fontweight='bold' if role in ['T', 'V'] else 'normal')
        counts = [len(fold[k]) for k in ['training', 'validation', 'test']]
        for x, n in zip([9.6, 10.9, 12.2], counts):
            right.text(x, i, str(n), ha='center', va='center', fontsize=9.5)
        evidence.append({'test_group': test, 'validation_group': val,
                         'roles': roles, 'adaptation_validation_test_patch_counts': counts})
    right.axvline(8.7, color='#BAC3CB', linewidth=0.8)
    right.set_xticks(list(range(9)) + [9.6, 10.9, 12.2],
                     [str(x) for x in range(9)] + ['A n', 'V n', 'T n'])
    right.xaxis.tick_top()
    right.tick_params(axis='both', length=0, pad=6)
    right.set_yticks(range(5), [f'Test {f["test_group"]}' for f in folds])
    right.set_xlim(-0.6, 12.9)
    right.set_ylim(4.7, -0.7)
    right.set_aspect('auto')
    for spine in right.spines.values():
        spine.set_visible(False)
    right.set_title('(b) Group roles in each evaluation fold', pad=29)
    right.set_xlabel('Spatial group ID                          Patch counts', labelpad=12, fontsize=9)
    right.legend(handles=[Patch(facecolor=COLORS['A'], label='A: Adaptation'),
                          Patch(facecolor=COLORS['V'], label='V: Validation'),
                          Patch(facecolor=COLORS['T'], label='T: Test')],
                 loc='lower center', bbox_to_anchor=(0.5, -0.27), ncol=3,
                 frameon=False, fontsize=8.5, columnspacing=0.9, handlelength=1)
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / 'Figure3.png', dpi=400, facecolor='white')
    fig.savefig(OUT / 'Figure3.svg', facecolor='white')
    plt.close(fig)
    (OUT / 'Figure3_data.json').write_text(json.dumps({
        'coordinate_origin_easting_northing_m': [x0, y0],
        'retained_patches': len(records), 'weed_annotated_patches': len(weed),
        'groups': [{'group': g['group'], 'patches': g['patches'], 'weed_patches': g['weed_patches']} for g in groups],
        'folds': evidence, 'note': 'Map displays retained patch footprints, not a complete field boundary or individual weed plants.'
    }, indent=2), encoding='utf-8')
    print('Saved Figure3.png, Figure3.svg and Figure3_data.json; all fold memberships verified.')


if __name__ == '__main__':
    main()
