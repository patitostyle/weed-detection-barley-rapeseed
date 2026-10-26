from pathlib import Path
import sys,json,os
import numpy as np
import torch,tifffile
HERE=Path(__file__).resolve().parents[1];WORK=HERE;ROOT=Path(os.environ.get('WEED_DATA_DIR',str(HERE/'data')))
OUT=HERE/'regenerated';OUT.mkdir(exist_ok=True)
from unet_architecture import UNet
torch.set_num_threads(2)
model=UNet(8,3);model.load_state_dict(torch.load(Path(os.environ.get('WEED_BASE_MODEL',str(HERE/'models'/'cebada_best_model.pth'))),map_location='cpu',weights_only=True));model.eval()
rr=json.loads((WORK/'auditoria_datos_tfm.json').read_text())['cebada_v2_C2_rgb_nir_re_indices']['records']
names=['patch_00020.tif',max(rr,key=lambda r:sum(r['counts'].get(str(c),0)>0 for c in [1,2,3])*100000+r['counts'].get('2',0))['id']]
arrays={}
for i,name in enumerate(names):
    pool=ROOT/'patches'/'cebada_v2_C2_rgb_nir_re_indices'
    a=tifffile.imread(pool/'images'/name).astype(np.float32)
    if a.shape[-1]==8:a=a.transpose(2,0,1)
    for c in range(8):
        b=a[c];v=np.isfinite(b);p,q=np.percentile(b[v],[2,98]) if v.any() else (0,0);a[c]=np.clip((b-p)/(q-p),0,1) if q>p else 0
    a[~np.isfinite(a)]=0;m=tifffile.imread(pool/'masks'/name);ref=np.where(np.isin(m,[1,2,3]),m-1,3)
    with torch.inference_mode():pred=model(torch.from_numpy(a.copy())[None]).argmax(1)[0].numpy()
    arrays.update({f'rgb{i}':a[:3].transpose(1,2,0),f'ref{i}':ref,f'pred{i}':pred})
np.savez_compressed(HERE/'qualitative_arrays.npz',names=np.array(names),**arrays)
print('Saved partial reference examples',names)
