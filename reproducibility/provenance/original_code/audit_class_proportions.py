"""Unique annotated-pixel composition of the four retained patch pools.

Uses saved masks, audited patch locations and the completed input-validity audit.
No training, image resampling or modification of original data is performed.
"""
import csv
import json
from pathlib import Path
import numpy as np
import tifffile

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parent
DATA=Path(r'C:\datos tfm\patches')
OUT=ROOT/'class_proportions'
FIELDS={
    'cebada_v1_rgb_nir_re_indices':'Barley 1-V1',
    'cebada_v2_rgb_nir_re_indices':'Barley 1-V2',
    'cebada_v2_C2_rgb_nir_re_indices':'Barley 2-V2',
    'colza_v2_rgb_nir_re_indices':'Rapeseed-V2',
}


def summarize(grid):
    counts=np.bincount(grid.ravel(),minlength=4)
    total=int(counts[1:4].sum())
    return {'unique_annotated_pixels':total,'class_counts':{str(c):int(counts[c]) for c in [1,2,3]},
            'class_percentages':{str(c):float(100*counts[c]/total) for c in [1,2,3]}}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    audit=json.loads((WORK/'auditoria_datos_tfm.json').read_text())
    validity=json.loads((ROOT/'input_audit.json').read_text())
    partitions=json.loads((ROOT/'staged_v2/partitions.json').read_text())
    groups=json.loads((WORK/'viabilidad_colza.json').read_text())['groups_14m']
    included={(r['dataset'],r['id']) for role in ['training','validation'] for r in partitions['source'][role]}
    results=[]
    for dataset,name in FIELDS.items():
        records=audit[dataset]['records']
        assert validity[dataset]['labelled_pixels_without_all_channels']==0
        assert {r['id'] for r in validity[dataset]['records']}=={r['id'] for r in records}
        pixel=records[0]['w']/256
        xmin=min(r['x'] for r in records)
        ymax=max(r['y'] for r in records)
        cols=[round((r['x']-xmin)/pixel) for r in records]
        rows=[round((ymax-r['y'])/pixel) for r in records]
        shape=(max(rows)+256,max(cols)+256)
        grid=np.zeros(shape,dtype=np.uint8)
        used=np.zeros(shape,dtype=np.uint8) if dataset.startswith(('cebada_v1_','cebada_v2_rgb')) else None
        occurrences=np.zeros(4,dtype=np.int64)
        conflicts=0
        used_patches=0
        for r,y,x in zip(records,rows,cols):
            assert abs((r['x']-xmin)/pixel-x)<1e-3
            assert abs((ymax-r['y'])/pixel-y)<1e-3
            m=tifffile.imread(DATA/dataset/'masks'/r['id']).astype(np.uint8)
            assert m.shape==(256,256) and np.all(m<=3)
            b=np.bincount(m.ravel(),minlength=4)
            for c in range(4):assert int(b[c])==r['counts'].get(str(c),0)
            occurrences+=b
            view=grid[y:y+256,x:x+256]
            conflicts+=int(np.count_nonzero((view>0)&(m>0)&(view!=m)))
            np.maximum(view,m,out=view)
            if used is not None and (dataset,r['id']) in included:
                view=used[y:y+256,x:x+256]
                np.maximum(view,m,out=view)
                used_patches+=1
        assert conflicts==0
        record={'dataset':dataset,'display_name':name,'patches':len(records),'pixel_size_m':pixel,
                'scope':'Retained annotated patch pool before experimental partitioning; each labelled geographic pixel counted once within acquisition.',
                **summarize(grid),'conflicting_overlap_labels':conflicts,
                'labelled_pixel_occurrences_including_overlap':int(occurrences[1:].sum()),
                'prior_input_audit_labelled_pixels_without_all_channels':0}
        if used is not None:
            record['revised_source_training_validation_union']={'patches':used_patches,**summarize(used)}
        if name=='Barley 2-V2':
            assert list(record['class_counts'].values())==[2975858,1023369,808269]
        if name=='Rapeseed-V2':
            expected={str(c):sum(g['unique_labelled_pixels'].get(str(c),0) for g in groups) for c in [1,2,3]}
            assert record['class_counts']==expected
        results.append(record)
        print(name,json.dumps(record),flush=True)
        del grid,used
    protocol={'denominator':'All unique annotated pixels (crop+weed+soil) within each retained patch pool; unlabelled pixels excluded.',
              'spatial_unit':'Unique geographic pixel within one acquisition; observations across dates are not merged.',
              'overlap':'Labels merged on the acquisition pixel grid; conflicting labels checked and absent.',
              'input_validity':'Completed full input audit reports zero labelled pixels lacking valid values in any input channel for all four datasets; patch-ID sets cross-checked.',
              'scope_note':'Dataset-level descriptive composition before source spatial buffer exclusions and target fold/budget sampling; not per-run training frequencies or field-wide cover.',
              'checks':'Per-patch class counts matched original audit; Barley2 counts matched evaluation reference; rapeseed counts matched independent saved group summaries.',
              'original_data_modified':False}
    (OUT/'class_proportions.json').write_text(json.dumps({'protocol':protocol,'datasets':results},indent=2)+'\n')
    csvrows=[]
    for r in results:
        csvrows.append({'dataset':r['display_name'],'patches':r['patches'],'unique_annotated_pixels':r['unique_annotated_pixels'],
                        **{f'{label}_pixels':r['class_counts'][str(c)] for c,label in [(1,'crop'),(2,'weed'),(3,'soil')]},
                        **{f'{label}_percent':r['class_percentages'][str(c)] for c,label in [(1,'crop'),(2,'weed'),(3,'soil')]}})
    with (OUT/'class_proportions.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(csvrows[0]));w.writeheader();w.writerows(csvrows)
    lines=[r'\begin{table}[htbp]',r'\centering',
           r'\caption{Class distribution among unique annotated pixels in the retained patch datasets before experimental partitioning. Each geographic pixel was counted once within an acquisition, excluding unlabelled pixels. Percentages describe the available annotations, not field-wide land cover.}',
           r'\label{tab:class-proportions}',r'\begin{tabular}{lrrrr}',r'\toprule',
           r'Dataset & Annotated pixels & Crop (\%) & Weed (\%) & Soil (\%) \\',r'\midrule']
    for r in results:
        cc=r['class_percentages']
        n=f"{r['unique_annotated_pixels']:,}".replace(',',r'\,')
        lines.append(f"{r['display_name']} & {n} & {cc['1']:.2f} & {cc['2']:.2f} & {cc['3']:.2f} "+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    (OUT/'class_proportions_table.tex').write_text('\n'.join(lines)+'\n')
    print('COMPLETE',flush=True)


if __name__=='__main__':main()
