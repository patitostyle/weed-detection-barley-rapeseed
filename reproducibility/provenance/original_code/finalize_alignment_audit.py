"""Export a transparent alignment audit, supplementary table and numerical checks."""
import csv
import hashlib
import json
from pathlib import Path
import assess_interband_alignment as a
import numpy as np
from scipy import ndimage
from osgeo import gdal
import matplotlib.pyplot as plt

OUT=a.OUT


def main():
    primary=json.loads((OUT/'results.json').read_text())
    extra=json.loads((OUT/'supplemental_results.json').read_text())
    with (OUT/'all_matches.csv').open(newline='') as f: attempts=list(csv.DictReader(f))
    with (OUT/'supplemental_matches.csv').open(newline='') as f: supp=list(csv.DictReader(f))
    gsd={r['field']:r['transform'][1] for r in primary['metadata']}
    valid={name:len([r for r in attempts if r['field']==name and r['band']=='Blue' and r.get('peak')]) for name in a.FIELDS}
    assert valid=={'Barley 1 V1':33,'Barley 1 V2':33,'Barley 2 V2':35,'Rapeseed V2':31}
    table=[]
    for field in a.FIELDS:
        for r in primary['summary']:
            if r['field']!=field:continue
            table.append({'field':field,'analysis':'Primary','reference':'Red','target':r['band'],
                          'valid_windows':valid[field],'accepted':r['accepted'],
                          'median_px':r.get('median_displacement_px'), 'p95_px':r.get('p95_displacement_px'),
                          'rms_estimated_shift_px':r.get('rms_estimated_displacement_px'), 'gsd_cm':gsd[field]*100})
        for r in extra['summary']:
            if r['field']!=field:continue
            zz=[z for z in supp if z['field']==field and z['reference']==r['reference'] and z['target']==r['target'] and z['accepted']=='True']
            dist=np.array([np.hypot(float(z['dx']),float(z['dy'])) for z in zz])
            table.append({'field':field,'analysis':'Exploratory','reference':r['reference'],'target':r['target'],
                          'valid_windows':r['valid_windows'],'accepted':r['accepted'],
                          'median_px':r.get('median_px'),'p95_px':r.get('p95_px'),
                          'rms_estimated_shift_px':float(np.sqrt(np.mean(dist**2))) if len(dist) else None,'gsd_cm':gsd[field]*100})
    with (OUT/'summary_table.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)

    # This checks estimator recovery on known translations of the actual red
    # images. It does not validate unknown cross-band shifts against ground truth.
    checks=[]
    for field,path in a.FIELDS.items():
        r=next(r for r in attempts if r['field']==field and r.get('peak'))
        ds=gdal.Open(str(path),gdal.GA_ReadOnly)
        x=ds.GetRasterBand(1).ReadAsArray(int(r['cx'])-a.SIZE//2,int(r['cy'])-a.SIZE//2,a.SIZE,a.SIZE)
        for dx,dy,invert in [(0.,0.,False),(1.,0.,False),(.4,-.3,False),(-2.,1.5,True)]:
            y=ndimage.shift(x,(dy,dx),order=3,mode='reflect')
            y=(-2*y+5) if invert else (2*y+5)
            m=a.match(a.edges(x),a.edges(y))
            error=float(np.hypot(m['dx']-dx,m['dy']-dy))
            checks.append({'field':field,'injected_dx':dx,'injected_dy':dy,'contrast_inverted':invert,
                           'estimated_dx':m['dx'],'estimated_dy':m['dy'],'recovery_error_px':error})
            assert error<.15,checks[-1]
    (OUT/'real_image_injection_checks.json').write_text(json.dumps({'scope':'Estimator recovery only, not cross-band ground truth.','checks':checks,'max_recovery_error_px':max(x['recovery_error_px'] for x in checks)},indent=2)+'\n')

    # Recompute summaries directly from CSV and confirm they match the JSON.
    for r in primary['summary']:
        zz=[z for z in attempts if z['field']==r['field'] and z['band']==r['band'] and z['accepted']=='True']
        assert len(zz)==r['accepted']
        if zz:
            dist=np.array([np.hypot(float(z['dx']),float(z['dy'])) for z in zz])
            assert np.isclose(np.median(dist),r['median_displacement_px'])
            assert np.isclose(np.percentile(dist,95),r['p95_displacement_px'])

    # Supplemental visual review uses the median-NCC accepted example per pair.
    for field,path in a.FIELDS.items():
        ds=gdal.Open(str(path),gdal.GA_ReadOnly)
        fig,axs=plt.subplots(2,3,figsize=(10,7),layout='constrained')
        for i,(ref,tgt,ri,ti) in enumerate([('NIR','Red edge',4,5),('Green','NIR',2,4)]):
            rr=[z for z in supp if z['field']==field and z['reference']==ref and z['target']==tgt]
            good=[z for z in rr if z['accepted']=='True']
            pool=sorted(good or rr,key=lambda r:float(r['peak']))
            r=pool[len(pool)//2]
            x=ds.ReadAsArray(int(r['cx'])-a.SIZE//2,int(r['cy'])-a.SIZE//2,a.SIZE,a.SIZE,band_list=[ri,ti])
            for j,band in enumerate([ref,tgt]):
                axs[i,j].imshow(a.normalize(x[j]),cmap='gray');axs[i,j].set_title(band)
            overlay=np.zeros((a.SIZE,a.SIZE,3))
            for j,z in enumerate(x):
                e=a.edges(z);overlay[...,j]=np.clip(e/max(np.percentile(e,99),1e-9),0,1)
            axs[i,2].imshow(overlay)
            axs[i,2].set_title(f'Unshifted edge overlay\ndx={float(r["dx"]):+.2f}, dy={float(r["dy"]):+.2f}; NCC={float(r["peak"]):.3f}\n'+('Accepted' if good else 'No accepted matches'),fontsize=9)
            for ax in axs[i]:ax.set_axis_off()
        fig.suptitle(field+' | exploratory pair checks')
        fig.savefig(OUT/(field.replace(' ','_')+'_supplemental_examples.png'),dpi=150)
        plt.close(fig)

    def fmt(x):return '--' if x is None else f'{x:.2f}'
    lines=[r'% Proposed supplementary table; merge with final supplement and assign its final number.',
           r'\begin{table}[htbp]',r'\centering',r'\small',
           r'\caption{Image-based relative interband displacement diagnostic. Each dataset had 100 regularly spaced candidate centres; only fully valid windows were analysed. P denotes the primary red-reference analysis and E denotes exploratory additional band pairs. Displacements summarize accepted image matches, not independent surveyed checkpoints. A dash denotes no accepted correspondence.}',
           r'\label{tab:interband-diagnostic}',r'\begin{tabular}{lllrrrr}',r'\toprule',
           r'Dataset & Pair (reference/target) & Analysis & $n/N$ & Median (px) & P95 (px) & RMS (px) \\',r'\midrule']
    for i,r in enumerate(table):
        if i and r['field']!=table[i-1]['field']:lines.append(r'\midrule')
        lines.append(f"{r['field']} & {r['reference']}/{r['target']} & {'P' if r['analysis']=='Primary' else 'E'} & {r['accepted']}/{r['valid_windows']} & {fmt(r['median_px'])} & {fmt(r['p95_px'])} & {fmt(r['rms_estimated_shift_px'])} "+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}',
              r'\noindent\textit{Diagnostic protocol.} A $10\times10$ grid was placed across each raster bounding box. Windows of $320\times320$ pixels containing any NoData or non-finite values were excluded, leaving 33, 33, 35 and 31 windows in Barley~1-V1, Barley~1-V2, Barley~2-V2 and Rapeseed-V2, respectively. Each original band was scaled within the window using its second and ninety-eighth percentiles, clipped, and converted to Gaussian gradient magnitude with a one-pixel standard deviation. A $224\times224$-pixel template was matched within a $\pm16$-pixel translation search using normalized cross-correlation and quadratic subpixel peak refinement. Both matching directions required a peak correlation of at least 0.60, a peak margin of at least 0.02 outside the two-pixel peak neighbourhood, an interior search maximum, and forward--reverse closure of at most one pixel. All rejected comparisons were retained in the audit. RMS denotes the root mean square of estimated displacement magnitudes, not independent checkpoint RMSE. Additional NIR/red-edge and green/NIR comparisons were introduced after low cross-spectral similarity was observed in some primary comparisons; they are explicitly exploratory and do not replace the primary results. Raster pixel sizes were 5.516~cm for Barley~1-V1 and 5.477~cm for the other three datasets. Shared raster grids alone do not establish physical feature alignment. Spectral contrast, canopy geometry, matching uncertainty and exclusion of poorly matchable regions limit extrapolation of these estimates to whole-field accuracy. The synthetic shift checks validate numerical recovery, not cross-band geodetic accuracy.']
    (OUT/'supplementary_alignment_table.tex').write_text('\n'.join(lines)+'\n')
    report=['INTERBAND ALIGNMENT AUDIT - 2026-11-24','',
            'Completed: read-only inspection of four study analysis stacks; 400 grid candidates, 132 fully valid spatial windows.',
            'These are image windows, NOT surveyed control/check points. Several band pairs are evaluated at each window.',
            'Primary analysis uses red as reference. Additional pairs are exploratory and explicitly labelled.',
            'No raster, annotation or trained model was changed.',
            '', 'PAIRWISE RESULTS: accepted/valid; median displacement pixels; P95 pixels; RMS estimated displacement pixels']
    for r in table:
        report.append(f"{r['field']} | {r['reference']} / {r['target']} | {r['analysis']} | {r['accepted']}/{r['valid_windows']} | {fmt(r['median_px'])} | {fmt(r['p95_px'])} | {fmt(r['rms_estimated_shift_px'])}")
    report+=['','INTERPRETATION',
             'Accepted red/green and red/blue pair median displacements span 0.134 to 0.346 pixels. This describes accepted image matches only.',
             'Some primary red/NIR and red/red-edge comparisons returned too few or no accepted matches. These failures remain in the table.',
             'Exploratory green-reference NIR matching in Rapeseed-V2 returned 20/31 accepted windows: median 1.063923 pixels, P95 1.228805 pixels (about 5.83 and 6.73 cm).',
             'This is evidence of an apparent residual relative displacement at matchable image structures, not proof of exact absolute field accuracy.',
             'Natural cross-spectral differences can influence apparent offsets. No surveyed checkpoints are available in this audit.',
             'These measurements do not establish the causal effect of geometric mismatch on weed F1.',
             'Do not claim zero registration error, uniformly subpixel registration, independent ground accuracy or manually surveyed checkpoints.',
             '', 'VERIFICATION',
             f"Known-shift recovery on actual red images: 16 checks; maximum error {max(x['recovery_error_px'] for x in checks):.6f} pixels. This is an estimator check only.",
             'All primary table statistics were recomputed from CSV and matched JSON.',
             'Representative primary and exploratory visual panels were generated for review.',
             '', 'PENDING MANUSCRIPT WORK',
             'Insert the diagnostic method/limitations and result into Section 2.2; incorporate the complete table/protocol into the supplement.',
             'Update response3.2 with this diagnostic and explicitly retain the lack of independent ground accuracy.',
             'Reconcile May report reflectance-target n/a with operator/author panel confirmation; report calibration evidence faithfully.',
             'Assess original annotation minimum-size rule under response3.3; do not infer a manual threshold from these windows.']
    (OUT/'alignment_audit.txt').write_text('\n'.join(report)+'\n',encoding='utf-8')
    manifest={'inputs':[{'file':str(p),'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in a.FIELDS.values()],
              'script_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(a.__file__),Path(__file__),a.ROOT/'review_interband_alignment.py']},
              'source_rasters_modified':False,'numerical_checks_passed':True,'primary_results':291,'supplemental_results':193,
              'visual_review':'Primary representative panels reviewed for all four datasets; supplemental panels require review before completion.'}
    (OUT/'audit_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Report exported; known-shift maximum error',max(x['recovery_error_px'] for x in checks),flush=True)


if __name__=='__main__':main()
