"""Read-only audit of source TIFFs. Outputs never modify historical experiments."""
import json
from pathlib import Path
import numpy as np
import tifffile

WORK = Path(__file__).resolve().parents[1]
ROOT = Path('C:/datos tfm/patches')
OUT = Path(__file__).resolve().parent
AUDIT = json.loads((WORK/'auditoria_datos_tfm.json').read_text())

def components(records, gap=14.):
    n=len(records); parent=list(range(n))
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i,a in enumerate(records):
        for j in range(i):
            b=records[j]
            dx=max(0,a['x']-b['x']-b['w'],b['x']-a['x']-a['w'])
            dy=max(0,a['y']-a['h']-b['y'],b['y']-b['h']-a['y'])
            if np.hypot(dx,dy)<=gap+1e-5:parent[find(i)]=find(j)
    groups={}
    for i in range(n):groups.setdefault(find(i),[]).append(i)
    return list(groups.values())

def main():
    report={}
    for dataset,meta in AUDIT.items():
        rows=[]
        for rec in meta['records']:
            f=ROOT/dataset/'images'/rec['id']
            with tifffile.TiffFile(f) as t:
                a=t.asarray();tag=t.pages[0].tags.get('GDAL_NODATA')
                nodata=float(tag.value) if tag else None
            if a.shape[-1]==8:a=np.moveaxis(a,-1,0)
            valid=np.isfinite(a)
            if nodata is not None:valid &= a!=nodata
            valid &= a>-1e30
            m=tifffile.imread(ROOT/dataset/'masks'/rec['id'])
            legacy=np.array([np.percentile(b[np.isfinite(b)],2) if np.isfinite(b).any() else np.nan for b in a])
            rows.append({'id':rec['id'],'nodata':nodata,'invalid_values':int((~valid).sum()),
                         'p02_is_nodata_channels':int((legacy < -1e30).sum()),
                         'labelled_pixels_without_all_channels':int(((m>0)&~valid.all(0)).sum())})
        report[dataset]={'patches':len(rows),'patches_affected_p02':sum(x['p02_is_nodata_channels']>0 for x in rows),
                         'labelled_pixels_without_all_channels':sum(x['labelled_pixels_without_all_channels'] for x in rows),'records':rows}
        print(dataset,report[dataset]['patches_affected_p02'],'affected patches',flush=True)
    source=[]
    for name in ['cebada_v1_rgb_nir_re_indices','cebada_v2_rgb_nir_re_indices']:
        source += [dict(r,dataset=name) for r in AUDIT[name]['records']]
    gg=components(source)
    report['source_geographic_groups']=[{'group':k,'patches':len(g),'class_counts_with_overlap':{str(c):sum(source[i]['counts'].get(str(c),0) for i in g) for c in [1,2,3]},'records':[source[i] for i in g]} for k,g in enumerate(gg)]
    (OUT/'input_audit.json').write_text(json.dumps(report,indent=2))
    print('source groups:',[(x['group'],x['patches'],x['class_counts_with_overlap']) for x in report['source_geographic_groups']],flush=True)

if __name__=='__main__':main()
