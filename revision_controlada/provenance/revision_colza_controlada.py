"""Paired initialization comparison with fixed spatial groups and unique-pixel metrics.
Run --train first. --evaluate requires all 18 completed runs; no test-based selection.
Original data and checkpoints are read-only. Outputs are separate and never overwritten.
"""
import argparse, json, time, random, hashlib, copy, os
from pathlib import Path
import numpy as np
import tifffile
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from unet_architecture import UNet

WORK=Path(os.environ.get('WEED_WORK_DIR',str(Path(__file__).resolve().parents[2])))
ROOT=Path(os.environ.get('WEED_DATA_DIR','C:/datos tfm'))
BASE_MODEL=Path(os.environ.get('WEED_BASE_MODEL',str(ROOT/'models'/'cebada_best_model.pth')))
OUT=WORK/'revision_agriculture'/'experimentos_colza'
PATCH=ROOT/'patches'/'colza_v2_rgb_nir_re_indices'
SEEDS=[42,43,44]; BUDGETS=[21,42,63]; EPOCHS=30; PATIENCE=7; LR=1e-4
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(4)
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True

def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2),encoding='utf8');tmp.replace(path)

def ids(kind):return (WORK/f'propuesta_colza_{kind}_ids.txt').read_text().splitlines()
RECORDS={r['id']:r for r in json.loads((WORK/'auditoria_datos_tfm.json').read_text())['colza_v2_rgb_nir_re_indices']['records']}

class Cached(Dataset):
    def __init__(self,names):
        self.names=names;self.images=[];self.masks=[]
        for name in names:
            a=tifffile.imread(PATCH/'images'/name).astype(np.float32)
            if a.shape[-1]==8:a=a.transpose(2,0,1)
            assert a.shape==(8,256,256),a.shape
            for c in range(8):
                b=a[c];valid=np.isfinite(b)
                p,q=np.percentile(b[valid],[2,98]) if valid.any() else (0,0)
                a[c]=np.clip((b-p)/(q-p),0,1) if q>p else 0
            a[~np.isfinite(a)]=0
            m=tifffile.imread(PATCH/'masks'/name)
            z=np.full(m.shape,255,np.int64)
            for c in [1,2,3]:z[m==c]=c-1
            self.images.append(torch.from_numpy(a.copy()))
            self.masks.append(torch.from_numpy(z))
    def __len__(self):return len(self.names)
    def __getitem__(self,i):return self.images[i],self.masks[i]

class UniqueMetric:
    def __init__(self,ds):
        self.valid=[];keys=[];labels=[]
        xmin=min(RECORDS[n]['x'] for n in ds.names);ymax=max(RECORDS[n]['y'] for n in ds.names)
        pixel=RECORDS[ds.names[0]]['w']/256
        width=round((max(RECORDS[n]['x']+RECORDS[n]['w'] for n in ds.names)-xmin)/pixel)+1
        for name,m in zip(ds.names,ds.masks):
            m=m.numpy();r,c=np.where(m!=255);rec=RECORDS[name]
            cols=c+round((rec['x']-xmin)/pixel);rows=r+round((ymax-rec['y'])/pixel)
            keys.append(rows.astype(np.int64)*width+cols);labels.append(m[r,c]);self.valid.append((r,c))
        unique,first,inv=np.unique(np.concatenate(keys),return_index=True,return_inverse=True)
        all_labels=np.concatenate(labels);self.labels=all_labels[first]
        assert np.all(all_labels==self.labels[inv]),'Conflicting reference labels'
        self.n=len(unique);self.maps=[];start=0
        for a in keys:self.maps.append(inv[start:start+len(a)]);start+=len(a)
        self.counts=np.bincount(inv,minlength=self.n)
    def evaluate(self,model,ds):
        totals=np.zeros((3,self.n),np.float64)
        loader=DataLoader(ds,batch_size=4,shuffle=False,num_workers=0,pin_memory=DEVICE=='cuda')
        model.eval();index=0
        with torch.inference_mode():
            for imgs,_ in loader:
                with torch.autocast(device_type=DEVICE,enabled=DEVICE=='cuda'):
                    probs=model(imgs.to(DEVICE)).float().softmax(1).cpu().numpy()
                for prob in probs:
                    r,c=self.valid[index];mapping=self.maps[index]
                    for cls in range(3):totals[cls]+=np.bincount(mapping,weights=prob[cls,r,c],minlength=self.n)
                    index+=1
        pred=(totals/self.counts).argmax(0)
        cm=np.bincount(self.labels*3+pred,minlength=9).reshape(3,3)
        tp=cm.diagonal();fp=cm.sum(0)-tp;fn=cm.sum(1)-tp
        f1=2*tp/np.maximum(2*tp+fp+fn,1);iou=tp/np.maximum(tp+fp+fn,1)
        return {'f1':f1.tolist(),'iou':iou.tolist(),'miou':float(iou.mean()),'confusion':cm.tolist(),'unique_labelled_pixels':self.n}

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    settings={'seeds':SEEDS,'budgets':BUDGETS,'epochs_max':EPOCHS,'patience':PATIENCE,'min_delta':1e-4,'lr_both_strategies':LR,'batch_size':4,'loss_weights':[1,3,1.5],'evaluation':'mean probability per unique labelled pixel','strategies':['transfer_full','scratch'],'notes':'Historical barley checkpoint previously selected using Barley 2; target test used only after all runs. Training subsets conditioned on presence of all three labelled classes.'}
    protocol=OUT/'protocol.json'
    if protocol.exists():assert json.loads(protocol.read_text())==settings,'Protocol changed'
    else:dump(protocol,settings)
    pool=ids('train')
    for seed in SEEDS:
        rng=random.Random(seed)
        # Nested prefixes; reject permutations missing any class in the smallest budget.
        for attempt in range(10000):
            order=pool.copy();rng.shuffle(order)
            if all(sum(RECORDS[n]['counts'].get(str(c),0) for n in order[:21])>0 for c in [1,2,3]):break
        else:raise RuntimeError('No admissible nested subset')
        for budget in BUDGETS:
            path=OUT/f'subset_seed{seed}_n{budget}.json'
            chosen=order[:budget]
            if path.exists():assert json.loads(path.read_text())==chosen
            else:dump(path,chosen)

def train():
    prepare();valid_ds=Cached(ids('validation'));metric=UniqueMetric(valid_ds)
    criterion=nn.CrossEntropyLoss(weight=torch.tensor([1.,3.,1.5],device=DEVICE),ignore_index=255)
    for seed in SEEDS:
        for budget in BUDGETS:
            names=json.loads((OUT/f'subset_seed{seed}_n{budget}.json').read_text());ds=Cached(names)
            for strategy in ['transfer_full','scratch']:
                run=OUT/f'{strategy}_seed{seed}_n{budget}'
                if (run/'complete.json').exists():continue
                if run.exists():raise RuntimeError(f'Partial run exists; preserve and resolve before resuming: {run}')
                run.mkdir();seed_all(seed)
                model=UNet(8,3).to(DEVICE)
                if strategy=='transfer_full':model.load_state_dict(torch.load(BASE_MODEL,map_location=DEVICE,weights_only=True))
                opt=torch.optim.Adam(model.parameters(),lr=LR)
                scaler=torch.amp.GradScaler('cuda',enabled=DEVICE=='cuda')
                generator=torch.Generator().manual_seed(seed)
                loader=DataLoader(ds,batch_size=4,shuffle=True,generator=generator,num_workers=0,pin_memory=DEVICE=='cuda')
                best=-1;stale=0;history=[];start=time.time()
                for ep in range(1,EPOCHS+1):
                    model.train();losses=[]
                    for a,m in loader:
                        opt.zero_grad(set_to_none=True)
                        with torch.autocast(device_type=DEVICE,enabled=DEVICE=='cuda'):
                            loss=criterion(model(a.to(DEVICE)),m.to(DEVICE))
                        scaler.scale(loss).backward();scaler.step(opt);scaler.update();losses.append(float(loss.detach()))
                    metrics=metric.evaluate(model,valid_ds);score=metrics['f1'][1]
                    history.append({'epoch':ep,'loss':float(np.mean(losses)),'validation':metrics})
                    if score>best+1e-4:
                        best=score;best_ep=ep;stale=0;torch.save(model.state_dict(),run/'best.pth')
                    else:stale+=1
                    dump(run/'history.json',history)
                    print(f'{strategy} seed={seed} n={budget} epoch={ep} valF1={score:.4f} best={best:.4f} elapsed={time.time()-start:.1f}s',flush=True)
                    if stale>=PATIENCE:break
                dump(run/'complete.json',{'strategy':strategy,'seed':seed,'budget':budget,'selected_epoch':best_ep,'validation_f1':best,'epochs_run':ep,'seconds':time.time()-start,'train_ids':names})
                del model,opt,scaler
                if DEVICE=='cuda':torch.cuda.empty_cache()

def evaluate():
    prepare()
    runs=[OUT/f'{s}_seed{seed}_n{n}' for seed in SEEDS for n in BUDGETS for s in ['transfer_full','scratch']]
    assert all((r/'complete.json').exists() for r in runs),'Complete all predefined training runs before opening test'
    if (OUT/'test_results.json').exists():raise RuntimeError('Test results already exist; do not overwrite')
    ds=Cached(ids('test'));metric=UniqueMetric(ds);results=[]
    model=UNet(8,3).to(DEVICE);model.load_state_dict(torch.load(BASE_MODEL,map_location=DEVICE,weights_only=True))
    results.append({'strategy':'direct_transfer','test':metric.evaluate(model,ds)})
    for run in runs:
        model.load_state_dict(torch.load(run/'best.pth',map_location=DEVICE,weights_only=True))
        record=json.loads((run/'complete.json').read_text());record['test']=metric.evaluate(model,ds);results.append(record)
    dump(OUT/'test_results.json',results)
    print(json.dumps(results,indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--train',action='store_true');parser.add_argument('--evaluate',action='store_true');args=parser.parse_args()
    if args.train:train()
    elif args.evaluate:evaluate()
    else:prepare();print('Protocol and nested sample lists saved. Use --train, then --evaluate.')
