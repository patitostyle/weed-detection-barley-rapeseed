"""Visual QA and a separately labelled supplemental cross-spectral diagnostic."""
import csv
import json
from pathlib import Path
import assess_interband_alignment as a
import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt

OUT=a.OUT


def qa():
    with (OUT/'all_matches.csv').open(encoding='utf-8',newline='') as f:
        rows=list(csv.DictReader(f))
    for name,path in a.FIELDS.items():
        ds=gdal.Open(str(path),gdal.GA_ReadOnly)
        fig,axes=plt.subplots(4,3,figsize=(10,12),layout='constrained')
        for i,band in enumerate(a.BANDS[1:]):
            rr=[r for r in rows if r['field']==name and r['band']==band and r.get('peak')]
            good=[r for r in rr if r['accepted']=='True']
            pool=sorted(good or rr,key=lambda r:float(r['peak']))
            r=pool[len(pool)//2]
            cx,cy=int(r['cx']),int(r['cy'])
            x=ds.ReadAsArray(cx-a.SIZE//2,cy-a.SIZE//2,a.SIZE,a.SIZE,band_list=[1,i+2])
            ref,mov=x
            ee=[a.edges(z) for z in x]
            for j,z in enumerate(x):
                axes[i,j].imshow(a.normalize(z),cmap='gray')
                axes[i,j].set_title(('Red reference' if j==0 else band)+f' | grid {r["grid_row"]},{r["grid_col"]}',fontsize=10)
                axes[i,j].axvline(a.SIZE//2,color='cyan',alpha=.6,lw=.5)
                axes[i,j].axhline(a.SIZE//2,color='cyan',alpha=.6,lw=.5)
            overlay=np.zeros((a.SIZE,a.SIZE,3))
            for j,e in enumerate(ee):
                overlay[...,j]=np.clip(e/max(np.percentile(e,99),1e-9),0,1)
            axes[i,2].imshow(overlay)
            axes[i,2].set_title(f'Edge overlay, unshifted\nNCC={float(r["peak"]):.3f}; dx={float(r["dx"]):+.2f}, dy={float(r["dy"]):+.2f}\n'+('PASSED' if good else 'NO accepted matches for this pair'),fontsize=9)
            for ax in axes[i]:ax.set_axis_off()
        fig.suptitle(name+' | diagnostic examples\nMedian-NCC accepted example per pair; when none passed, median-NCC rejected example',fontsize=11)
        fig.savefig(OUT/(name.replace(' ','_')+'_examples.png'),dpi=150)
        plt.close(fig)
        print('QA',name,flush=True)


def supplemental():
    # Added after the primary diagnostic showed low red/NIR or red/red-edge
    # similarity. It is an exploratory check, not a replacement of failed pairs.
    protocol={'reason':'Primary red-reference comparison yielded few/no acceptable NIR or red-edge matches for some acquisitions.',
              'timing':'Specified after primary results; exploratory supplemental comparison.',
              'pairs':[{'reference':'NIR','target':'Red edge'},{'reference':'Green','target':'NIR'}],
              'locations':'Same 10x10 candidate grid and fully valid 320px windows as primary analysis.',
              'qc':'Identical fixed correlation, ambiguity and forward/reverse thresholds as primary.',
              'scope':'Pairwise relative estimates only; no chained or surveyed registration accuracy claimed.'}
    (OUT/'supplemental_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    rows=[]
    for name,path in a.FIELDS.items():
        ds=gdal.Open(str(path),gdal.GA_ReadOnly)
        w,h=ds.RasterXSize,ds.RasterYSize
        for row,cy in enumerate(np.linspace(a.SIZE//2,h-a.SIZE//2-1,a.GRID).astype(int)):
            for col,cx in enumerate(np.linspace(a.SIZE//2,w-a.SIZE//2-1,a.GRID).astype(int)):
                x=ds.ReadAsArray(int(cx-a.SIZE//2),int(cy-a.SIZE//2),a.SIZE,a.SIZE,band_list=[2,4,5])
                if not np.all(np.isfinite(x)&(x>-1e30)):
                    continue
                ee=[a.edges(z) for z in x]
                for ref,mov,rn,mn in [(1,2,'NIR','Red edge'),(0,1,'Green','NIR')]:
                    m=a.match(ee[ref],ee[mov]);rev=a.match(ee[mov],ee[ref])
                    closure=float(np.hypot(m['dx']+rev['dx'],m['dy']+rev['dy'])) if m['dx'] is not None and rev['dx'] is not None else None
                    reasons=[]
                    if m['dx'] is None or rev['dx'] is None:reasons.append('low_texture')
                    if m['boundary'] or rev['boundary']:reasons.append('boundary')
                    if min(m['peak'],rev['peak'])<.60:reasons.append('low_ncc')
                    if min(m['margin'],rev['margin'])<.02:reasons.append('ambiguous_peak')
                    if closure is None or closure>1:reasons.append('closure')
                    rows.append({'field':name,'reference':rn,'target':mn,'grid_row':row,'grid_col':col,'cx':int(cx),'cy':int(cy),
                                 **m,'closure_px':closure,'accepted':not reasons,'reasons':'|'.join(reasons)})
        print('Supplemental',name,flush=True)
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/'supplemental_matches.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    summary=[]
    for name in a.FIELDS:
        for ref,mov in [('NIR','Red edge'),('Green','NIR')]:
            rr=[r for r in rows if r['field']==name and r['reference']==ref and r['target']==mov]
            passed=[r for r in rr if r['accepted']]
            r={'field':name,'reference':ref,'target':mov,'valid_windows':len(rr),'accepted':len(passed)}
            if passed:
                d=np.array([np.hypot(z['dx'],z['dy']) for z in passed])
                r.update(median_px=float(np.median(d)),p95_px=float(np.percentile(d,95)))
            summary.append(r)
    (OUT/'supplemental_results.json').write_text(json.dumps({'protocol':protocol,'summary':summary},indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    qa()
    supplemental()
