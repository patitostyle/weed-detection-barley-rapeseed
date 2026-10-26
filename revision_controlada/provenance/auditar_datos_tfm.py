from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageDraw

root = Path('C:/datos tfm')
out = Path(__file__).parent
names = ['cebada_v1_rgb_nir_re_indices', 'cebada_v2_rgb_nir_re_indices', 'cebada_v2_C2_rgb_nir_re_indices', 'colza_v2_rgb_nir_re_indices']
canvas = Image.new('RGB', (1600, 1700), 'white')
draw = ImageDraw.Draw(canvas)
report = {}
for k, name in enumerate(names):
    records = []
    for p in sorted((root/'patches'/name/'masks').glob('*.tif')):
        with Image.open(p) as im:
            a = np.asarray(im)
            counts = {str(int(v)): int(n) for v,n in zip(*np.unique(a, return_counts=True))}
            t = im.tag_v2
            sx,sy,_ = t[33550]
            _,_,_,x,y,_ = t[33922]
            records.append(dict(id=p.name,x=x,y=y,w=im.width*sx,h=im.height*sy,counts=counts))
    xmin=min(r['x'] for r in records); xmax=max(r['x']+r['w'] for r in records)
    ymin=min(r['y']-r['h'] for r in records); ymax=max(r['y'] for r in records)
    scale=min(720/(xmax-xmin),700/(ymax-ymin))
    ox=(k%2)*800+30; oy=(k//2)*850+60
    draw.text((ox,oy-35),name,fill='black')
    for r in records:
        xx=ox+(r['x']-xmin)*scale; yy=oy+(ymax-r['y'])*scale
        draw.rectangle((xx,yy,xx+r['w']*scale,yy+r['h']*scale),outline='#bbbbbb')
        if r['counts'].get('2',0)>0:
            draw.ellipse((xx+2,yy+2,xx+6,yy+6),fill='red')
    sums={str(v):sum(r['counts'].get(str(v),0) for r in records) for v in range(4)}
    report[name]={'patches':len(records),'patches_with_weeds':sum(r['counts'].get('2',0)>0 for r in records),'pixel_counts_with_overlap':sums,'bounds':[xmin,ymin,xmax,ymax],'records':records}
    draw.text((ox,oy+730),f"{len(records)} patches; red: contains weed labels",fill='black')
canvas.save(out/'distribucion_patches_tfm.png')
(out/'auditoria_datos_tfm.json').write_text(json.dumps(report),encoding='utf-8')
for name,r in report.items(): print(name, {k:v for k,v in r.items() if k!='records'})
