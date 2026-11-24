"""Read-only image-based interband alignment diagnostic on the study stacks.

This estimates translations from image structure; it is NOT independent surveyed
checkpoint accuracy, nor a measurement of Phantom-to-Altum registration.
All candidates and rejected matches are retained. No image is corrected here.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

if os.name == 'nt':
    os.environ['PATH'] = str(Path(sys.prefix) / 'Library' / 'bin') + os.pathsep + os.environ.get('PATH', '')

import numpy as np
from scipy import ndimage, signal
from osgeo import gdal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

gdal.UseExceptions()
ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'interband_alignment'
DATA = Path(r'C:\datos tfm\multibanda')
FIELDS = {
    'Barley 1 V1': DATA / 'cebada_v1/mb_rgb_nir_re_indices_cebada_v1.tif',
    'Barley 1 V2': DATA / 'cebada_v2/mb_rgb_nir_re_indices_cebada_v2.tif',
    'Barley 2 V2': DATA / 'cebada_test/mb_rgb_nir_re_indices_cebada_v2.tif',
    'Rapeseed V2': DATA / 'colza_v2/mb_rgb_nir_re_indices_colza_v2.tif',
}
BANDS = ['Red', 'Green', 'Blue', 'NIR', 'Red edge']
SIZE = 320
TEMPLATE = 224
RADIUS = 16
GRID = 10
PROTOCOL = {
    'scope': 'Retrospective image-structure-based relative alignment diagnostic of existing five-band analysis rasters.',
    'not_measured': ['Independent ground-coordinate accuracy', 'Phantom-to-Altum alignment', 'Causal effect of registration on F1'],
    'reference_channel': 'Red, stack channel 1 (one-based).',
    'channels': BANDS,
    'candidate_locations': '10 x 10 equally spaced centres over each raster bounding box; no selection using observed shifts.',
    'read_window_px': SIZE,
    'template_px': TEMPLATE,
    'search_radius_px': RADIUS,
    'preprocessing': 'Per-window per-band 2nd/98th percentile scaling, clipping, Gaussian derivative magnitude with sigma=1 pixel. All-invalid windows rejected. No derived indices used for matching.',
    'minimum_valid_fraction': 1.0,
    'matching': 'Normalized cross-correlation of gradient magnitudes; quadratic 3x3 subpixel peak refinement.',
    'qc_fixed_before_results': {'minimum_peak_ncc': .60, 'minimum_peak_margin_outside_2px_neighbourhood': .02, 'maximum_forward_reverse_closure_px': 1.0, 'reject_search_boundary_peaks': True},
    'reporting': 'All attempts, reasons for exclusion, accepted counts, signed shifts, median/p95 displacement and RMS of estimated displacement. RMS is not independent checkpoint RMSE.',
    'data_modification': 'None. Input rasters opened read-only; no transformations applied.',
}


def normalize(a):
    a = np.asarray(a, dtype=np.float32)
    lo, hi = np.percentile(a, [2, 98])
    if not np.isfinite(hi - lo) or hi <= lo:
        return np.zeros_like(a)
    return np.clip((a - lo) / (hi - lo), 0, 1)


def edges(a):
    a = normalize(a)
    dx = ndimage.gaussian_filter(a, 1.0, order=(0, 1), mode='reflect')
    dy = ndimage.gaussian_filter(a, 1.0, order=(1, 0), mode='reflect')
    return np.hypot(dx, dy)


def box_sum(a, n):
    integ = np.pad(np.cumsum(np.cumsum(a.astype(np.float64), axis=0), axis=1), ((1, 0), (1, 0)))
    return integ[n:, n:] - integ[:-n, n:] - integ[n:, :-n] + integ[:-n, :-n]


def match(ref, mov):
    c = SIZE // 2
    h = TEMPLATE // 2
    template = ref[c-h:c+h, c-h:c+h].astype(np.float64)
    search = mov[c-h-RADIUS:c+h+RADIUS, c-h-RADIUS:c+h+RADIUS].astype(np.float64)
    t = template - template.mean()
    energy = float(np.sum(t*t))
    if energy < 1e-9:
        return {'dx': None, 'dy': None, 'peak': 0., 'margin': 0., 'boundary': False, 'texture_energy': energy}
    dots = signal.fftconvolve(search, t[::-1, ::-1], mode='valid')
    sums = box_sum(search, TEMPLATE)
    sums2 = box_sum(search*search, TEMPLATE)
    den = np.sqrt(np.maximum(sums2 - sums*sums / t.size, 0.) * energy)
    ncc = np.divide(dots, den, out=np.full_like(dots, -1.), where=den > 1e-12)
    iy, ix = np.unravel_index(np.argmax(ncc), ncc.shape)
    boundary = iy in (0, len(ncc)-1) or ix in (0, len(ncc)-1)
    sy = sx = 0.
    if not boundary:
        patch = ncc[iy-1:iy+2, ix-1:ix+2]
        yy, xx = np.mgrid[-1:2, -1:2]
        A = np.column_stack([xx.ravel()**2, yy.ravel()**2, (xx*yy).ravel(), xx.ravel(), yy.ravel(), np.ones(9)])
        q = np.linalg.lstsq(A, patch.ravel(), rcond=None)[0]
        H = np.array([[2*q[0], q[2]], [q[2], 2*q[1]]])
        if np.all(np.linalg.eigvalsh(H) < 0):
            v = np.linalg.solve(H, -q[3:5])
            if np.all(np.abs(v) <= .75):
                sx, sy = map(float, v)
    others = ncc.copy()
    others[max(0,iy-2):iy+3, max(0,ix-2):ix+3] = -np.inf
    return {'dx': float(ix-RADIUS+sx), 'dy': float(iy-RADIUS+sy), 'peak': float(ncc[iy,ix]),
            'margin': float(ncc[iy,ix]-others.max()), 'boundary': bool(boundary), 'texture_energy': energy}


def verify_matcher():
    rng = np.random.default_rng(324)
    a = ndimage.gaussian_filter(rng.normal(size=(SIZE, SIZE)), 1.3)
    checks = []
    for dx,dy in [(0,0),(2.4,-3.2),(-7.,5.),(12.,-9.)]:
        b = ndimage.shift(a, (dy, dx), order=3, mode='reflect')
        m = match(a, b)
        err = float(np.hypot(m['dx']-dx,m['dy']-dy))
        checks.append({'injected_dx':dx,'injected_dy':dy, 'estimated':m,'error_px':err})
        if err > .15:
            raise AssertionError(checks[-1])
    return checks


def summarize(rows):
    result = []
    for name in FIELDS:
        for band in BANDS[1:]:
            allr = [r for r in rows if r['field']==name and r['band']==band]
            rr = [r for r in allr if r['accepted']]
            d = {'field':name,'band':band,'attempts':len(allr),'accepted':len(rr),
                 'rejection_counts': {reason:sum(reason in r['reasons'].split('|') for r in allr) for reason in ['invalid_pixels','low_texture','boundary','low_ncc','ambiguous_peak','closure']}}
            if rr:
                delta = np.array([[r['dx'],r['dy']] for r in rr])
                dist = np.linalg.norm(delta, axis=1)
                d.update({'median_dx_px':float(np.median(delta[:,0])), 'median_dy_px':float(np.median(delta[:,1])),
                          'median_displacement_px':float(np.median(dist)), 'p95_displacement_px':float(np.percentile(dist,95)),
                          'rms_estimated_displacement_px':float(np.sqrt(np.mean(dist**2))),
                          'median_displacement_cm':float(np.median(dist)*rr[0]['gsd_m']*100),
                          'p95_displacement_cm':float(np.percentile(dist,95)*rr[0]['gsd_m']*100),
                          'median_ncc':float(np.median([r['peak'] for r in rr]))})
            result.append(d)
    return result


def plots(rows, previews):
    for name, rgb in previews.items():
        fig, ax = plt.subplots(figsize=(8,7),layout='constrained')
        ax.imshow(rgb, extent=(0,rgb.shape[1],rgb.shape[0],0))
        used = [r for r in rows if r['field']==name and r['band']=='NIR']
        for r in used:
            color = '#1b9e77' if r['accepted'] else '#d95f02'
            ax.plot(r['cx']/r['overview_factor_x'],r['cy']/r['overview_factor_y'],'.',color=color,ms=5)
        ax.set_title(name+' | NIR-to-red diagnostic locations\nGreen: passed matching checks; orange: excluded')
        ax.set_axis_off()
        fig.savefig(OUT/(name.replace(' ','_')+'_locations.png'),dpi=160)
        plt.close(fig)


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'protocol.json').write_text(json.dumps(PROTOCOL,indent=2)+'\n')
    checks=verify_matcher()
    (OUT/'synthetic_verification.json').write_text(json.dumps(checks,indent=2)+'\n')
    rows=[]
    previews={}
    metadata=[]
    for name,path in FIELDS.items():
        ds=gdal.Open(str(path),gdal.GA_ReadOnly)
        w,h=ds.RasterXSize,ds.RasterYSize
        gt=ds.GetGeoTransform()
        n=ds.GetRasterBand(1).GetNoDataValue()
        preview_w=1000
        preview_h=round(h*preview_w/w)
        rgb=ds.ReadAsArray(buf_xsize=preview_w,buf_ysize=preview_h,band_list=[1,2,3])
        valid=np.all(np.isfinite(rgb)&(rgb>-1e30),axis=0)
        disp=np.zeros((preview_h,preview_w,3),dtype=np.float32)
        for j in range(3):
            lo,hi=np.percentile(rgb[j][valid],[2,98])
            disp[...,j]=np.clip((rgb[j]-lo)/max(hi-lo,1e-9),0,1)
        disp[~valid]=1
        previews[name]=disp
        metadata.append({'field':name,'file':str(path),'width':w,'height':h,'transform':gt,'nodata':n,'projection':ds.GetProjection()})
        count=0
        for row,cy in enumerate(np.linspace(SIZE//2,h-SIZE//2-1,GRID).astype(int)):
            for col,cx in enumerate(np.linspace(SIZE//2,w-SIZE//2-1,GRID).astype(int)):
                a=ds.ReadAsArray(int(cx-SIZE//2),int(cy-SIZE//2),SIZE,SIZE,band_list=[1,2,3,4,5])
                ok=np.all(np.isfinite(a)&(a>-1e30))
                ee=[edges(x) for x in a] if ok else None
                for j in range(1,5):
                    r={'field':name,'band':BANDS[j],'grid_row':row,'grid_col':col,'cx':int(cx),'cy':int(cy),
                       'easting':gt[0]+cx*gt[1], 'northing':gt[3]+cy*gt[5], 'gsd_m':abs(gt[1]),
                       'overview_factor_x':w/preview_w, 'overview_factor_y':h/preview_h}
                    reasons=[]
                    if not ok:
                        reasons.append('invalid_pixels')
                    else:
                        m=match(ee[0],ee[j])
                        back=match(ee[j],ee[0])
                        r.update(m)
                        closure=float(np.hypot(m['dx']+back['dx'],m['dy']+back['dy'])) if m['dx'] is not None and back['dx'] is not None else None
                        r['reverse_dx']=back['dx'];r['reverse_dy']=back['dy'];r['closure_px']=closure
                        if m['dx'] is None or back['dx'] is None:reasons.append('low_texture')
                        if m['boundary'] or back['boundary']:reasons.append('boundary')
                        if min(m['peak'],back['peak'])<.60:reasons.append('low_ncc')
                        if min(m['margin'],back['margin'])<.02:reasons.append('ambiguous_peak')
                        if closure is None or closure>1.:reasons.append('closure')
                    r['accepted']=not reasons;r['reasons']='|'.join(reasons)
                    rows.append(r)
                count+=1
        summary=summarize(rows)
        print(name,json.dumps([x for x in summary if x['field']==name]),flush=True)
        ds=None
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/'all_matches.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    result={'status':'Image-based diagnostic; not independent geodetic validation. Visual QA required before manuscript use.',
            'protocol':PROTOCOL,'metadata':metadata,'summary':summarize(rows),'synthetic_verification':checks}
    (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    plots(rows,previews)


if __name__=='__main__':
    run()
