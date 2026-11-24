"""RF baselines with spectral and optional local texture features.

Train samples are selected without replacement from unique labelled pixels per
acquisition. All model evaluation uses identical annotated support and averages
probabilities where patch footprints overlap.
"""
import argparse,copy,json
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn.functional as F
import run_revision as r

def features(a,texture):
    if not texture:return a
    mean=F.avg_pool2d(a,3,stride=1,padding=1,count_include_pad=False)
    variance=(F.avg_pool2d(a*a,3,stride=1,padding=1,count_include_pad=False)-mean*mean).clamp(min=0)
    return torch.cat([a,mean,variance.sqrt()],1)

def sample(data,texture,seed):
    rng=np.random.default_rng(seed);positions=[]
    for part,ev in r.metric(data).parts:
        seen=set()
        # Sample a fixed number per unique pixel, not per overlapping patch.
        for index,((rows,cols),mapping) in enumerate(zip(ev.valid,ev.maps)):
            fresh=np.array([int(k) not in seen for k in mapping]);seen.update(map(int,mapping))
            if fresh.any():positions.append((part,index,rows[fresh],cols[fresh]))
    total=sum(len(x[2]) for x in positions)
    chosen=set(rng.choice(total,min(total,300000),replace=False).tolist())
    xs=[];ys=[];offset=0
    for part,i,rows,cols in positions:
        keep=np.array([offset+k in chosen for k in range(len(rows))]);offset+=len(rows)
        if not keep.any():continue
        f=features(part.images[i][None],texture)[0].numpy()
        xs.append(f[:,rows[keep],cols[keep]].T);ys.append(part.masks[i].numpy()[rows[keep],cols[keep]])
    return np.concatenate(xs),np.concatenate(ys)

class Forest(torch.nn.Module):
    def __init__(self,model,texture):super().__init__();self.model=model;self.texture=texture
    def forward(self,x):
        b,c,h,w=x.shape;f=features(x.float(),self.texture)
        p=self.model.predict_proba(f.permute(0,2,3,1).reshape(-1,f.shape[1]).cpu().numpy())
        full=np.zeros((len(p),3),np.float32);full[:,self.model.classes_.astype(int)]=p
        # Evaluator expects logits and applies softmax; log probabilities preserve them.
        return torch.from_numpy(np.log(np.maximum(full,1e-30))).reshape(b,h,w,3).permute(0,3,1,2).to(x.device)

def train():
    m=r.prepare();stats=r.source_stats(m['source']['training'])
    for norm in r.NORMS:
        ds=r.Data(m['source']['training'],norm,stats)
        for texture in [False,True]:
            for seed in r.SOURCE_SEEDS:
                p=r.OUT/f"forest_{norm}_{'texture' if texture else 'spectral'}_seed{seed}"
                if (p/'complete.json').exists():continue
                p.mkdir(parents=True,exist_ok=True)
                x,y=sample(ds,texture,seed)
                model=RandomForestClassifier(n_estimators=200,random_state=seed,n_jobs=4)
                model.fit(x,y);joblib.dump(model,p/'model.joblib',compress=3)
                r.dump(p/'complete.json',{'seed':seed,'sample_pixels':len(y),'sampling':'unique geographic pixels per acquisition, no replacement',
                                         'features':'8 spectral plus 3x3 means and standard deviations' if texture else '8 spectral channels',
                                         'training_class_counts':np.bincount(y,minlength=3).tolist()})
                print(p.name,'completed',flush=True)
        del ds

if __name__=='__main__':train()
