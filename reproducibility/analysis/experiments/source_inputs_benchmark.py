"""Bounded source audit: 9 new spectral U-Nets and 6 RF fits, one normalization.

No cross-product with target budgets. Reuses the full eight-channel U-Net runs.
RF texture adds local 3x3 channel means and standard deviations. RF fitting is
saved in tree batches, so an interruption repeats at most the unfinished batch.
"""
import argparse,copy,hashlib,json,msvcrt,time
from pathlib import Path
import joblib
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
import run_revision as r
from forest_comparison import features
from spectral_comparison import CONFIGS

BASE=r.HERE/'staged_v2';OUT=BASE/'source_inputs_benchmark';SEEDS=[42,43,44]

def prepare():
    assert json.loads((BASE/'deeplab_benchmark/status.json').read_text())['status']=='complete'
    m=json.loads((BASE/'partitions.json').read_text());stats=json.loads((BASE/'source_scaling.json').read_text())['bounds']
    plan={'new_unet_runs':9,'new_forest_fits':6,'seeds':SEEDS,'normalization':'patch_valid',
          'spectral_configs':CONFIGS,'full_config':'reused from completed DeepLab benchmark, three source U-Nets',
          'partition':str(BASE/'partitions.json'),'checkpoint_selection':'Barley1 spatial validation weed F1 only',
          'neural_training':{'batch_size':4,'lr':1e-4,'max_epochs':30,'patience':7,'loss':'unweighted cross entropy','augmentation':'paired right-angle rotations and flips'},
          'forest':{'trees':200,'n_jobs':4,'max_features':'sqrt','class_weight':None,'sample_limit':300000,
                    'sampling':'without replacement from unique labelled geographic pixels within each acquisition; deterministic first patch occurrence',
                    'features':['8 spectral channels','8 spectral + 3x3 per-channel means and standard deviations'],
                    'warm_start_batch_trees':25},
          'evaluation':'identical unique annotated Barley2-V2 geographic pixels for all models; average patch probabilities before argmax',
          'scope':'retrospective source comparison, no architecture-specific optimization and no new acquisition data'}
    p=OUT/'protocol.json'
    if p.exists():assert r.protocol_matches(json.loads(p.read_text()),plan)
    else:r.dump(p,plan)
    return m,stats

def spectral_train(manifest,stats):
    # Read TIFFs once; channel subsets are in-memory views/copies.
    parent=r.Data(manifest['source']['training'],'patch_valid',stats,True)
    parent_val=r.Data(manifest['source']['validation'],'patch_valid',stats)
    for name,channels in CONFIGS.items():
        tr=copy.copy(parent);val=copy.copy(parent_val)
        tr.images=[a[channels].contiguous() for a in parent.images];val.images=[a[channels].contiguous() for a in parent_val.images]
        for seed in SEEDS:
            folder=OUT/f'unet_{name}_seed{seed}'
            if (folder/'complete.json').exists():continue
            r.dump(OUT/'status.json',{'status':'running','phase':'spectral_training','configuration':name,'seed':seed})
            r.legacy.seed_all(seed)
            print('SPECTRAL TRAINING',name,'seed',seed,flush=True)
            r.train_one(folder,tr,val,seed,'source',model=r.UNet(len(channels),3),batch_size=4)

def forest_sample(ds,seed,texture):
    cache=OUT/'samples'/f"seed{seed}_{'texture' if texture else 'spectral'}.npz"
    if cache.exists():
        with np.load(cache) as saved:return saved['x'],saved['y']
    positions=[];acquisitions=[]
    for acquisition,(part,ev) in enumerate(r.metric(ds).parts):
        acquisitions.append(part.names[0].split('/')[0]);seen=np.zeros(ev.n,dtype=bool)
        for i,((rows,cols),mapping) in enumerate(zip(ev.valid,ev.maps)):
            fresh=~seen[mapping];seen[mapping]=True
            if fresh.any():positions.append((acquisition,part,i,rows[fresh],cols[fresh],mapping[fresh]))
    total=sum(len(z[3]) for z in positions)
    chosen=np.sort(np.random.default_rng(seed).choice(total,min(total,300000),replace=False))
    xs=[];ys=[];identities=[];offset=0
    for acquisition,part,i,rows,cols,mapping in positions:
        left=np.searchsorted(chosen,offset);right=np.searchsorted(chosen,offset+len(rows))
        take=chosen[left:right]-offset;offset+=len(rows)
        if not len(take):continue
        f=features(part.images[i][None],texture)[0].numpy()
        xs.append(f[:,rows[take],cols[take]].T.copy());ys.append(part.masks[i].numpy()[rows[take],cols[take]])
        identities.append(np.column_stack([np.full(len(take),acquisition),mapping[take]]))
    x=np.concatenate(xs).astype(np.float32);y=np.concatenate(ys);identity=np.concatenate(identities)
    assert len(np.unique(identity,axis=0))==len(y),'Repeated training pixel'
    assert np.isfinite(x).all() and set(np.unique(y))=={0,1,2}
    cache.parent.mkdir(parents=True,exist_ok=True)
    temp=cache.with_suffix('.tmp')
    with temp.open('wb') as stream:np.savez_compressed(stream,x=x,y=y,identity=identity,acquisitions=np.array(acquisitions))
    temp.replace(cache)
    print('RF SAMPLE',seed,'texture',texture,'unique pixels',len(y),flush=True)
    return x,y

def forest_train(manifest,stats):
    ds=r.Data(manifest['source']['training'],'patch_valid',stats)
    for texture in [False,True]:
        name='texture' if texture else 'spectral'
        for seed in SEEDS:
            folder=OUT/f'forest_{name}_seed{seed}'
            if (folder/'complete.json').exists():continue
            r.dump(OUT/'status.json',{'status':'running','phase':'forest_training','features':name,'seed':seed})
            x,y=forest_sample(ds,seed,texture);folder.mkdir(exist_ok=True)
            checkpoint=folder/'model.joblib'
            model=joblib.load(checkpoint) if checkpoint.exists() else RandomForestClassifier(n_estimators=25,n_jobs=4,random_state=seed,warm_start=True)
            completed=len(model.estimators_) if hasattr(model,'estimators_') else 0
            for trees in range(completed+25,201,25):
                model.set_params(n_estimators=trees);model.fit(x,y)
                temp=checkpoint.with_suffix('.tmp');joblib.dump(model,temp,compress=3);temp.replace(checkpoint)
                r.dump(folder/'progress.json',{'trees_saved':trees,'trees_total':200,'seed':seed})
                print('RF',name,'seed',seed,'saved trees',trees,'/200',flush=True)
            assert len(model.estimators_)==200
            r.dump(folder/'complete.json',{'seed':seed,'trees':200,'sample_pixels':len(y),
                'training_class_counts':np.bincount(y,minlength=3).tolist(),'features':name,
                'sample_sha256':hashlib.sha256((OUT/'samples'/f'seed{seed}_{name}.npz').read_bytes()).hexdigest()})
            del model,x,y

def rf_evaluate(model,ds,texture):
    # Predict labelled occurrences only, then deduplicate exactly as neural metrics.
    combined=np.zeros((3,3),dtype=np.int64);unique_count=0
    for part,ev in r.metric(ds).parts:
        totals=np.zeros((3,ev.n),dtype=np.float64)
        for i,((rows,cols),mapping) in enumerate(zip(ev.valid,ev.maps)):
            f=features(part.images[i][None],texture)[0].numpy()[:,rows,cols].T
            probabilities=model.predict_proba(f)
            for j,c in enumerate(model.classes_.astype(int)):
                totals[c]+=np.bincount(mapping,weights=probabilities[:,j],minlength=ev.n)
        pred=(totals/ev.counts).argmax(0)
        combined+=np.bincount(ev.labels*3+pred,minlength=9).reshape(3,3);unique_count+=ev.n
    tp=combined.diagonal();fp=combined.sum(0)-tp;fn=combined.sum(1)-tp
    f1=2*tp/np.maximum(2*tp+fp+fn,1);iou=tp/np.maximum(tp+fp+fn,1)
    return {'f1':f1.tolist(),'iou':iou.tolist(),'miou':float(iou.mean()),'confusion':combined.tolist(),'unique_labelled_pixels':unique_count}

def evaluate(stats):
    expected=[OUT/f'unet_{name}_seed{s}' for name in CONFIGS for s in SEEDS]
    expected += [OUT/f'forest_{name}_seed{s}' for name in ['spectral','texture'] for s in SEEDS]
    assert all((p/'complete.json').exists() for p in expected)
    ds=r.Data([dict(x,dataset='cebada_v2_C2_rgb_nir_re_indices') for x in r.AUDIT['cebada_v2_C2_rgb_nir_re_indices']['records']],'patch_valid',stats)
    old=json.loads((BASE/'deeplab_benchmark/test_results.json').read_text())
    reference=np.array(old[0]['test']['confusion']).sum(1)
    # Reuse full-input U-Net evaluations; no retraining or re-evaluation needed.
    rows=[dict(x,model='unet_full') for x in old if x['model']=='UNet']
    for name,channels in CONFIGS.items():
        sub=copy.copy(ds);sub.images=[a[channels] for a in ds.images];ev=r.metric(sub)
        for seed in SEEDS:
            dest=OUT/'evaluations'/f'unet_{name}_seed{seed}.json'
            if dest.exists():record=json.loads(dest.read_text())
            else:
                r.dump(OUT/'status.json',{'status':'running','phase':'evaluation','model':name,'seed':seed})
                model=r.UNet(len(channels),3).to(r.DEVICE)
                model.load_state_dict(torch.load(OUT/f'unet_{name}_seed{seed}'/'best.pth',map_location=r.DEVICE,weights_only=True))
                metrics=ev.evaluate(model,sub);record={'model':'unet_'+name,'seed':seed,'test':metrics}
                r.dump(dest,record);del model
            assert np.array_equal(np.array(record['test']['confusion']).sum(1),reference)
            rows.append(record)
    for texture in [False,True]:
        name='texture' if texture else 'spectral'
        for seed in SEEDS:
            dest=OUT/'evaluations'/f'forest_{name}_seed{seed}.json'
            if dest.exists():record=json.loads(dest.read_text())
            else:
                r.dump(OUT/'status.json',{'status':'running','phase':'evaluation','model':'forest_'+name,'seed':seed})
                model=joblib.load(OUT/f'forest_{name}_seed{seed}'/'model.joblib')
                record={'model':'forest_'+name,'seed':seed,'test':rf_evaluate(model,ds,texture)}
                r.dump(dest,record);del model
            assert np.array_equal(np.array(record['test']['confusion']).sum(1),reference)
            rows.append(record)
    assert len(rows)==18
    r.dump(OUT/'test_results.json',rows)
    summary=[]
    for name in sorted({x['model'] for x in rows}):
        selected=[x for x in rows if x['model']==name];assert len(selected)==3
        record={'model':name,'runs':3}
        for label,values in [('weed_f1',[x['test']['f1'][1] for x in selected]),('weed_iou',[x['test']['iou'][1] for x in selected]),('miou',[x['test']['miou'] for x in selected])]:
            record[label+'_mean']=float(np.mean(values));record[label+'_sample_sd']=float(np.std(values,ddof=1))
        summary.append(record)
    r.dump(OUT/'summary.json',{'models':summary,'reference_class_counts':reference.tolist(),'scope':'same unique annotated Barley2 pixels; retrospective comparison, three training seeds; RF trained on a unique-pixel subsample while neural training uses patch occurrences'})
    r.dump(OUT/'status.json',{'status':'complete','training_complete':True,'test_evaluated':True,'summary_complete':True})
    print('SOURCE INPUTS BENCHMARK COMPLETE: 9 new U-Net fits, 6 RF fits, common-support evaluation and summary.',flush=True)

def main():
    manifest,stats=prepare()
    if (OUT/'summary.json').exists():print('Benchmark already complete.',flush=True);return
    spectral_train(manifest,stats);forest_train(manifest,stats);evaluate(stats)

if __name__=='__main__':
    BASE.mkdir(parents=True,exist_ok=True)
    with (BASE/'execution.lock').open('a+b') as lock:
        if lock.tell()==0:lock.write(b'0');lock.flush()
        lock.seek(0)
        try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:print('Otro experimento sigue ejecutandose.',flush=True);raise SystemExit(2)
        try:main()
        finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
