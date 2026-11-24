"""Measure saved vector annotation areas and raster-region sizes read-only.

Raster connected regions are not assumed to be original drawn polygons or plants.
Observed minima are not annotation inclusion thresholds.
"""
import csv
import json
import os
import sys
from pathlib import Path

if os.name=='nt':
    os.environ['PATH']=str(Path(sys.prefix)/'Library'/'bin')+os.pathsep+os.environ.get('PATH','')
import numpy as np
from scipy import ndimage
from osgeo import gdal,ogr
gdal.UseExceptions()

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'annotation_sizes'
DATA=Path(r'C:\datos tfm\etiquetas')
NAMES={1:'crop',2:'weed',3:'soil'}
MASKS={'Barley 1 V1':'mask_cebada_v1.tif','Barley 1 V2':'mask_cebada_v2.tif',
       'Barley 2 V2':'mask_cebada_v2_c2.tif','Rapeseed V2':'mask_colza_v2.tif'}


def stats(x):
    x=np.asarray(x,dtype=np.float64)
    if len(x)==0:return {'n':0}
    return {'n':int(len(x)),'minimum':float(x.min()),'p25':float(np.percentile(x,25)),
            'median':float(np.median(x)),'p75':float(np.percentile(x,75)),
            'maximum':float(x.max()),'sum':float(x.sum())}


def write_csv(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    shp=DATA/'cebada_v1.shp'
    vds=ogr.Open(str(shp),0)
    layer=vds.GetLayer(0)
    srs=layer.GetSpatialRef()
    assert abs(srs.GetLinearUnits()-1)<1e-9
    records=[]
    for f in layer:
        g=f.GetGeometryRef()
        records.append({'feature_id':f.GetFID(),'class_id':f.GetField('class_id'),'class_name':f.GetField('class_name'),
                        'area_m2':g.GetArea(),'stored_area_m2':f.GetField('Shape_Area'),
                        'geometry_type':g.GetGeometryName(),'valid_geometry':bool(g.IsValid())})
    assert len(records)==59
    write_csv(OUT/'vector_polygon_areas.csv',records)
    vector={'file':str(shp),'srs':srs.ExportToWkt(),'feature_count':len(records),
            'class_stats_area_m2':{NAMES[c]:stats([r['area_m2'] for r in records if r['class_id']==c]) for c in NAMES},
            'interpretation':'Measured stored polygon feature areas, not individual plant sizes or prescribed annotation thresholds.'}
    print('VECTOR',json.dumps(vector['class_stats_area_m2']),flush=True)
    raster=[]
    components=[]
    comparison=None
    for name,filename in MASKS.items():
        ds=gdal.Open(str(DATA/filename),gdal.GA_ReadOnly)
        gt=ds.GetGeoTransform()
        area=abs(gt[1]*gt[5]-gt[2]*gt[4])
        arr=ds.GetRasterBand(1).ReadAsArray()
        if name=='Barley 1 V1':
            # Verify this saved vector is spatially compatible with the raster.
            # Both products supply their CRS; GDAL uses the layer transform.
            mem=gdal.GetDriverByName('MEM').Create('',ds.RasterXSize,ds.RasterYSize,1,gdal.GDT_Byte)
            mem.SetGeoTransform(gt);mem.SetProjection(ds.GetProjection())
            layer.ResetReading()
            gdal.RasterizeLayer(mem,[1],layer,options=['ATTRIBUTE=class_id'])
            b=mem.GetRasterBand(1).ReadAsArray()
            union=(b>0)|(arr>0)
            covered=b>0
            comparison={'scope':'Diagnostic rasterization of saved vector onto existing mask grid, pixel-centre convention. Boundary rasterization conventions may differ.',
                        'vector_nonzero_pixels':int(covered.sum()),'mask_nonzero_pixels':int((arr>0).sum()),
                        'agreement_on_vector_nonzero':float(np.mean(arr[covered]==b[covered])),
                        'agreement_on_labelled_union':float(np.mean(arr[union]==b[union]))}
            mem=None;del b,union,covered
        counts={str(c):int(np.count_nonzero(arr==c)) for c in [0,1,2,3]}
        labelled=sum(counts[str(c)] for c in NAMES)
        entry={'dataset':name,'file':str(DATA/filename),'shape':list(arr.shape),'pixel_area_m2':area,'pixel_size_x_m':gt[1],
               'projection':ds.GetProjection(),'class_pixels':counts,
               'labelled_pixels':labelled,'labelled_fraction_of_raster_bounding_box':labelled/arr.size,
               'class_fraction_of_labelled':{NAMES[c]:counts[str(c)]/labelled for c in NAMES},
               'scope':'Whole saved mask, not necessarily the exact union of retained evaluation patches.'}
        for c in NAMES:
            labels,n=ndimage.label(arr==c,structure=np.ones((3,3),dtype=np.uint8))
            sizes=np.bincount(labels.ravel(),minlength=n+1)[1:]
            assert int(sizes.sum())==counts[str(c)]
            np.save(OUT/(name.replace(' ','_')+'_'+NAMES[c]+'_component_pixels.npy'),sizes,allow_pickle=False)
            pix=stats(sizes)
            metres=stats(sizes*area)
            components.append({'dataset':name,'class':NAMES[c],'regions':int(n),'min_pixels':pix.get('minimum'),
                               'median_pixels':pix.get('median'),'max_pixels':pix.get('maximum'),
                               'min_area_m2':metres.get('minimum'),'p25_area_m2':metres.get('p25'),
                               'median_area_m2':metres.get('median'),'p75_area_m2':metres.get('p75'),'max_area_m2':metres.get('maximum'),
                               'total_area_m2':metres.get('sum')})
            del labels,sizes
        raster.append(entry)
        print('RASTER',name,json.dumps(components[-3:]),flush=True)
        del arr
        ds=None
    write_csv(OUT/'raster_region_sizes.csv',components)
    result={'scope':'Read-only annotation size audit. Original saved vectors available locally for Barley1-V1 only. Other three study datasets assessed as raster connected regions.',
            'author_description':'Crop and soil polygons were intended to be medium-sized; weed polygons were drawn according to what was visually distinguishable. No numerical size threshold has been documented.',
            'vector':vector,'vector_mask_comparison':comparison,'rasters':raster,'raster_regions':components,
            'connectivity':'Eight-neighbour connectivity; diagonal contacts merge regions. Rasterization may join/split drawn polygons.',
            'limits':['An observed minimum polygon/region is not a minimum resolvable plant size or an annotation inclusion rule.',
                      'The vector sizes cannot be extrapolated to all acquisition datasets without the other original vector annotations.',
                      'Connected regions can contain multiple plants or multiple touching annotation polygons; clipped rasters can truncate them.',
                      'Class proportions here are for whole masks, not deduplicated retained patch support; do not substitute for evaluation-support counts.'],
            'originals_modified':False}
    (OUT/'annotation_size_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print('COMPLETE',json.dumps(comparison),flush=True)


if __name__=='__main__':main()
