from pathlib import Path
import sys,json
import numpy as np
import torch
from torch.utils.data import DataLoader
HERE=Path(__file__).parent;ROOT=Path('C:/datos tfm');sys.path.insert(0,str(HERE.parent/'revision_codigo'/'scripts'))
from finetune_colza import UNet,ColzaPatchDataset
torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu'
folder=ROOT/'patches'/'cebada_v2_C2_rgb_nir_re_indices'
names=sorted(p.name for p in (folder/'images').glob('*.tif'))
ds=ColzaPatchDataset(names,str(folder/'images'),str(folder/'masks'))
model=UNet(8,3).to(device);model.load_state_dict(torch.load(ROOT/'models'/'cebada_best_model.pth',map_location=device,weights_only=True));model.eval()
cm=np.zeros((3,3),np.int64)
with torch.inference_mode():
    for a,m in DataLoader(ds,batch_size=8,num_workers=0):
        pred=model(a.to(device)).argmax(1).cpu().numpy();m=m.numpy();v=m!=255
        cm+=np.bincount(m[v]*3+pred[v],minlength=9).reshape(3,3)
tp=cm.diagonal();fp=cm.sum(0)-tp;fn=cm.sum(1)-tp
result={'checkpoint':'cebada_best_model.pth','dataset':'Barley 2 V2','metric':'pooled patch pixels, original protocol, float32 inference','confusion':cm.tolist(),'f1':(2*tp/np.maximum(2*tp+fp+fn,1)).tolist(),'iou':(tp/np.maximum(tp+fp+fn,1)).tolist()}
(HERE/'source_checkpoint_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
