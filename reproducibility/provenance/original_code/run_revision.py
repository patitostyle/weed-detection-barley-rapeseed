"""Prospective, resumable revision experiments; historical outputs remain untouched.

Training never opens test metrics. Run --evaluate only after the frozen schedule.
Two preprocessing variants exclude TIFF NoData. Fixed scaling is fitted only on
source training data and reused unchanged in the target domain.
"""
import argparse, copy, hashlib, importlib.util, json, random, sys, time
from pathlib import Path
import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset, DataLoader

HERE=Path(__file__).resolve().parent; WORK=HERE.parent; ROOT=Path('C:/datos tfm/patches')
sys.path.insert(0,str(WORK/'revision_codigo/scripts'))
from unet_architecture import UNet
import revision_colza_controlada as legacy
from audit_inputs import components
OUT=HERE/'experiments_v1'
SEEDS=[42,43,44,45,46]; SOURCE_SEEDS=[42,43,44]
NORMS=['patch_valid','source_fixed']; STRATEGIES=['scratch','full','partial']
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(4)
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True
AUDIT=json.loads((WORK/'auditoria_datos_tfm.json').read_text())

def dump(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)

def raw(dataset,name):
    with tifffile.TiffFile(ROOT/dataset/'images'/name) as t:
        a=t.asarray().astype(np.float32);nd=t.pages[0].tags.get('GDAL_NODATA')
        nd=float(nd.value) if nd else None
    if a.shape[-1]==8:a=np.moveaxis(a,-1,0)
    valid=np.isfinite(a)&(a>-1e30)
    if nd is not None:valid &= a!=nd
    m=tifffile.imread(ROOT/dataset/'masks'/name)
    return a,valid,m

def prepare():
    protocol={'version':1,'source_seeds':SOURCE_SEEDS,'target_seeds':SEEDS,'normalizations':NORMS,
      'strategies':STRATEGIES,'budgets':[21,42,63,'full'],'epochs':30,'patience':7,'min_delta':1e-4,
      'batch_size':4,'source_lr':1e-4,'scratch_lr':1e-4,'transfer_lr':5e-5,
      'augmentation':'paired horizontal/vertical flips and 90-degree rotations; no spectral jitter',
      'weights':[1,3,1.5],'weights_note':'fixed inherited target weights, not inverse-frequency estimates',
      'source_loss':'unweighted cross entropy','selection':'unique-pixel validation weed F1 only',
      'source_fixed':'per-channel p02/p98 from deterministic source-training-only pixel sample; identical transform across domains',
      'nodata':'exclude GDAL_NODATA and nonfinite values; ignore labels missing any input channel',
      'scope':'retrospective within-field grouped evaluation; no independent rapeseed field',
      'test_access':'only separate evaluate command after all scheduled runs have completed'}
    path=OUT/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==protocol,'Frozen protocol changed'
    else:dump(path,protocol)
    records=[]
    for ds in ['cebada_v1_rgb_nir_re_indices','cebada_v2_rgb_nir_re_indices']:
        records += [dict(r,dataset=ds) for r in AUDIT[ds]['records']]
    # Source annotations form one connected component, so split geographic blocks
    # with a 14 m gap and discard footprints crossing that gap, across both dates.
    admissible=[]
    xs=np.array([r['x']+r['w']/2 for r in records])
    for quantile in [.2,.25,.3,.7,.75,.8]:
        cut=float(np.quantile(xs,quantile))
        left=[i for i,r in enumerate(records) if r['x']+r['w']<=cut-7]
        right=[i for i,r in enumerate(records) if r['x']>=cut+7]
        for tr,g in [(left,right),(right,left)]:
            if all(sum(records[i]['counts'].get(str(c),0) for i in g)>0 and
                   sum(records[i]['counts'].get(str(c),0) for i in tr)>0 for c in [1,2,3]):
                admissible.append((abs(len(g)/len(records)-.2),cut,tr,g))
    assert admissible,'No admissible geographically disjoint source split'
    _,cut,ti,vi=min(admissible,key=lambda x:(x[0],x[1]))
    src={'training':[records[i] for i in ti],
         'validation':[records[i] for i in vi],
         'split_easting':cut,'gap_m':14,
         'excluded_crossing_gap':len(records)-len(ti)-len(vi)}
    target=json.loads((WORK/'viabilidad_colza.json').read_text())['groups_14m']
    folds=[]
    for test in target:
        if not test['weed_patches']:continue
        candidates=[]
        for val in target:
            if val['group']==test['group'] or not all(val['unique_labelled_pixels'].get(str(c),0)>0 for c in [1,2,3]):continue
            train=[g for g in target if g['group'] not in [val['group'],test['group']]]
            if all(sum(g['unique_labelled_pixels'].get(str(c),0) for g in train)>0 for c in [1,2,3]):
                candidates.append((abs(val['patches']/423-.15),val['group'],train,val))
        assert candidates
        _,_,train,val=min(candidates,key=lambda x:(x[0],x[1]))
        folds.append({'test_group':test['group'],'validation_group':val['group'],
                      'training':[n for g in train for n in g['ids']],
                      'validation':val['ids'],'test':test['ids']})
    manifest={'source':src,'target_folds':folds}
    path=OUT/'partitions.json'
    if path.exists():assert json.loads(path.read_text())==manifest
    else:dump(path,manifest)
    return manifest

class Data(Dataset):
    def __init__(self,records,norm,stats,augment=False):
        self.names=[];self.images=[];self.masks=[];self.records={};self.augment=augment
        for r in records:
            a,v,m=raw(r['dataset'],r['id']);name=r['dataset']+'/'+r['id']
            z=np.full(m.shape,255,np.int64)
            for c in [1,2,3]:z[m==c]=c-1
            z[~v.all(0)]=255
            for c in range(8):
                lo,hi=np.percentile(a[c][v[c]],[2,98]) if norm=='patch_valid' and v[c].any() else stats[c]
                if hi>lo:
                    a[c]=np.clip((np.where(v[c],a[c],lo)-lo)/(hi-lo),0,1)
                else:a[c]=0
                a[c][~v[c]]=0
            assert np.isfinite(a).all()
            self.names.append(name);self.records[name]=r
            self.images.append(torch.from_numpy(a.copy()));self.masks.append(torch.from_numpy(z))
    def __len__(self):return len(self.names)
    def __getitem__(self,i):
        a,m=self.images[i],self.masks[i]
        if self.augment:
            k=random.randrange(4);a=torch.rot90(a,k,(-2,-1));m=torch.rot90(m,k,(-2,-1))
            if random.random()<.5:a=a.flip(-1);m=m.flip(-1)
            if random.random()<.5:a=a.flip(-2);m=m.flip(-2)
        return a.contiguous(),m.contiguous()

class SubsetData(Data):
    def __init__(self,parent,names,augment=False):
        lookup={n:i for i,n in enumerate(parent.names)}
        ids=[lookup[parent.names[0].split('/')[0]+'/'+n] for n in names]
        self.names=[parent.names[i] for i in ids];self.images=[parent.images[i] for i in ids]
        self.masks=[parent.masks[i] for i in ids];self.records={n:parent.records[n] for n in self.names};self.augment=augment

def metric(ds):
    legacy.RECORDS=ds.records
    class Evaluator:
        def __init__(self,data):
            self.parts=[]
            for acquisition in sorted({n.split('/')[0] for n in data.names}):
                indices=[i for i,n in enumerate(data.names) if n.split('/')[0]==acquisition]
                part=copy.copy(data);part.augment=False
                part.names=[data.names[i] for i in indices]
                part.images=[data.images[i] for i in indices];part.masks=[data.masks[i] for i in indices]
                self.parts.append((part,legacy.UniqueMetric(part)))
            self.n=sum(e.n for _,e in self.parts)
        def evaluate(self,model,data):
            results=[ev.evaluate(model,part) for part,ev in self.parts]
            cm=sum((np.array(x['confusion']) for x in results),np.zeros((3,3),dtype=np.int64))
            tp=cm.diagonal();fp=cm.sum(0)-tp;fn=cm.sum(1)-tp
            f1=2*tp/np.maximum(2*tp+fp+fn,1);iou=tp/np.maximum(tp+fp+fn,1)
            result={'f1':f1.tolist(),'iou':iou.tolist(),'miou':float(iou.mean()),'confusion':cm.tolist(),'unique_labelled_pixels':self.n,
                    'counting_unit':'unique geographic pixel within each acquisition, never merged across dates'}
            present=cm.sum(1)>0
            result['reference_classes_present']=present.tolist()
            # A weed-only test cluster does not support a three-class mIoU.
            if not present.all():result['miou']=None
            return result
    return Evaluator(ds)

def source_stats(records):
    path=OUT/'source_scaling.json'
    if path.exists():return json.loads(path.read_text())['bounds']
    values=[[] for _ in range(8)];rng=np.random.default_rng(260622)
    for r in records:
        a,v,_=raw(r['dataset'],r['id'])
        for c in range(8):
            x=a[c][v[c]]
            if len(x):values[c].append(x[rng.choice(len(x),min(256,len(x)),replace=False)])
    bounds=[np.percentile(np.concatenate(x),[2,98]).tolist() for x in values]
    dump(path,{'bounds':bounds,'support':'up to 256 valid pixels per channel per source training patch; overlap retained for scaler fitting','seed':260622})
    return bounds

def train_one(path,train,val,seed,strategy,checkpoint=None,epochs=30,model=None,batch_size=4):
    if (path/'complete.json').exists():return
    resume_file=path/'last.pth'
    if path.exists() and not resume_file.exists():raise RuntimeError('Incomplete legacy run without resumable state preserved: '+str(path))
    path.mkdir(parents=True,exist_ok=True);legacy.seed_all(seed)
    model=model if model is not None else UNet(8,3)
    if checkpoint:model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True))
    if strategy=='partial':
        for block in [model.enc1,model.enc2,model.enc3]:
            for param in block.parameters():param.requires_grad_(False)
    model=model.to(DEVICE)
    opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=1e-4 if strategy in ['source','scratch','source_deeplab'] else 5e-5)
    criterion=torch.nn.CrossEntropyLoss(ignore_index=255,weight=None if strategy.startswith('source') else torch.tensor([1.,3.,1.5],device=DEVICE))
    scaler=torch.amp.GradScaler('cuda',enabled=DEVICE=='cuda')
    generator=torch.Generator().manual_seed(seed)
    loader=DataLoader(train,batch_size=batch_size,shuffle=True,num_workers=0,
                      drop_last=strategy.startswith('source_deeplab'),
                      generator=generator)
    evaluator=metric(val);best=-1.;stale=0;history=[];start=time.time();elapsed_before=0.;first_epoch=1;selected=0
    config={'seed':seed,'strategy':strategy,'epochs_max':epochs,'batch_size':batch_size,
            'train_ids':train.names,'validation_ids':val.names,'checkpoint':str(checkpoint) if checkpoint else None}
    if resume_file.exists():
        # This is our own trusted local training state, including RNG states.
        state=torch.load(resume_file,map_location='cpu',weights_only=False)
        assert state['config']==config,'Resume configuration does not match saved state'
        model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler'])
        random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng']);torch.set_rng_state(state['torch_rng'])
        if DEVICE=='cuda':torch.cuda.set_rng_state_all(state['cuda_rng'])
        generator.set_state(state['loader_rng'])
        best=state['best'];selected=state['selected_epoch'];stale=state['stale'];history=state['history']
        first_epoch=state['epoch']+1;elapsed_before=state['seconds']
        dump(path/'history.json',history)
        print('RESUMING',path.name,'at epoch',first_epoch,flush=True)
    epoch=first_epoch-1
    for epoch in range(first_epoch,epochs+1):
        if stale>=7:break
        model.train()
        if strategy=='partial':
            for block in [model.enc1,model.enc2,model.enc3]:block.eval() # Freeze running BN statistics too.
        losses=[]
        for a,m in loader:
            if not (m!=255).any():continue
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=DEVICE,enabled=DEVICE=='cuda'):
                loss=criterion(model(a.to(DEVICE)),m.to(DEVICE))
            assert torch.isfinite(loss),'Nonfinite training loss'
            scaler.scale(loss).backward();scaler.step(opt);scaler.update();losses.append(float(loss.detach()))
        score=evaluator.evaluate(model,val)
        history.append({'epoch':epoch,'loss':float(np.mean(losses)),'validation':score})
        if score['f1'][1]>best+1e-4:
            best=score['f1'][1];selected=epoch;stale=0
            best_temp=path/'best.tmp';torch.save(model.state_dict(),best_temp);best_temp.replace(path/'best.pth')
        else:stale+=1
        dump(path/'history.json',history)
        elapsed=elapsed_before+time.time()-start
        state={'config':config,'model':model.state_dict(),'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),
               'epoch':epoch,'best':best,'selected_epoch':selected,'stale':stale,'history':history,'seconds':elapsed,
               'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),
               'cuda_rng':torch.cuda.get_rng_state_all() if DEVICE=='cuda' else [],'loader_rng':generator.get_state()}
        temp=resume_file.with_suffix('.tmp');torch.save(state,temp);temp.replace(resume_file)
        print(path.name,'epoch',epoch,'valF1',round(score['f1'][1],4),'seconds',round(elapsed),flush=True)
        if stale>=7:break
    dump(path/'complete.json',{'seed':seed,'strategy':strategy,'selected_epoch':selected,'epochs_run':len(history),'validation_f1':best,'seconds':elapsed_before+time.time()-start,'batch_size':batch_size,'train_ids':train.names,'validation_ids':val.names})
    del model,opt,scaler
    if DEVICE=='cuda':torch.cuda.empty_cache()

def target_records(names):return [dict(r,dataset='colza_v2_rgb_nir_re_indices') for r in AUDIT['colza_v2_rgb_nir_re_indices']['records'] if r['id'] in set(names)]

def train_all(manifest,stats,limit=None):
    count=0
    for norm in NORMS:
        train=Data(manifest['source']['training'],norm,stats,True);val=Data(manifest['source']['validation'],norm,stats)
        for seed in SOURCE_SEEDS:
            path=OUT/f'source_{norm}_seed{seed}'
            if not (path/'complete.json').exists():
                train_one(path,train,val,seed,'source');count+=1
                if limit and count>=limit:return
        del train,val
    for norm in NORMS:
        allnames=[r['id'] for r in AUDIT['colza_v2_rgb_nir_re_indices']['records']]
        parent=Data(target_records(allnames),norm,stats)
        checkpoint=OUT/f'source_{norm}_seed42'/'best.pth'
        for fold in manifest['target_folds']:
            val=SubsetData(parent,fold['validation']);pool=fold['training']
            for seed in SEEDS:
                rng=random.Random(seed);order=pool.copy()
                for _ in range(10000):
                    rng.shuffle(order)
                    ds=SubsetData(parent,order[:21])
                    if all(sum(int((m==c).sum()) for m in ds.masks)>0 for c in range(3)):break
                else:raise RuntimeError('No three-class adaptation subset')
                for budget in [21,42,63,len(pool)]:
                    ds=SubsetData(parent,order[:budget],True)
                    subset=OUT/'subsets'/f"g{fold['test_group']}_seed{seed}_n{budget}.json"
                    if subset.exists():assert json.loads(subset.read_text())==ds.names
                    else:dump(subset,ds.names)
                    for strategy in STRATEGIES:
                        path=OUT/f"target_{norm}_g{fold['test_group']}_seed{seed}_n{budget}_{strategy}"
                        if not (path/'complete.json').exists():
                            train_one(path,ds,val,seed,strategy,None if strategy=='scratch' else checkpoint);count+=1
                            if limit and count>=limit:return
        del parent

def evaluate(manifest,stats):
    expected=[OUT/f'source_{norm}_seed{seed}' for norm in NORMS for seed in SOURCE_SEEDS]
    for norm in NORMS:
        for f in manifest['target_folds']:
            for seed in SEEDS:
                for n in [21,42,63,len(f['training'])]:
                    expected += [OUT/f"target_{norm}_g{f['test_group']}_seed{seed}_n{n}_{s}" for s in STRATEGIES]
    assert all((p/'complete.json').exists() for p in expected),'Complete frozen schedule before test evaluation'
    architecture_paths=[OUT/f"deeplab_{norm}_{initialization}_seed{seed}" for norm in NORMS
                        for initialization in ['random','imagenet'] for seed in SOURCE_SEEDS]
    assert all((p/'complete.json').exists() for p in architecture_paths),'Complete DeepLab comparison before opening test'
    extra_paths=[OUT/f'{kind}_{norm}_{variant}_seed{seed}'
                 for kind,variants in [('spectral',['rgb','rgb_nir_re','rgb_indices']),('forest',['spectral','texture'])]
                 for norm in NORMS for variant in variants for seed in SOURCE_SEEDS]
    assert all((p/'complete.json').exists() for p in extra_paths),'Complete spectral and RF comparisons before opening test'
    result=OUT/'test_results.json'
    if result.exists():raise RuntimeError('Test results already exist')
    rows=[]
    for norm in NORMS:
        src=Data([dict(r,dataset='cebada_v2_C2_rgb_nir_re_indices') for r in AUDIT['cebada_v2_C2_rgb_nir_re_indices']['records']],norm,stats)
        for seed in SOURCE_SEEDS:
            p=OUT/f'source_{norm}_seed{seed}';model=UNet(8,3).to(DEVICE)
            model.load_state_dict(torch.load(p/'best.pth',map_location=DEVICE,weights_only=True))
            rows.append({'run':p.name,'test':metric(src).evaluate(model,src),'note':'retrospective Barley 2 assessment; excluded from revised checkpoint selection'})
        from architecture_comparison import DeepLab
        for initialization in ['random','imagenet']:
            for seed in SOURCE_SEEDS:
                p=OUT/f'deeplab_{norm}_{initialization}_seed{seed}'
                model=DeepLab(False).to(DEVICE)
                model.load_state_dict(torch.load(p/'best.pth',map_location=DEVICE,weights_only=True))
                rows.append({'run':p.name,'test':metric(src).evaluate(model,src),
                             'note':'same unique annotated geographic pixels as revised U-Net comparison'})
        from spectral_comparison import CONFIGS
        for config,channels in CONFIGS.items():
            subset=copy.copy(src);subset.images=[a[channels] for a in src.images]
            for seed in SOURCE_SEEDS:
                p=OUT/f'spectral_{norm}_{config}_seed{seed}';model=UNet(len(channels),3).to(DEVICE)
                model.load_state_dict(torch.load(p/'best.pth',map_location=DEVICE,weights_only=True))
                rows.append({'run':p.name,'test':metric(subset).evaluate(model,subset)})
        import joblib
        from forest_comparison import Forest
        for texture in [False,True]:
            for seed in SOURCE_SEEDS:
                p=OUT/f"forest_{norm}_{'texture' if texture else 'spectral'}_seed{seed}"
                model=Forest(joblib.load(p/'model.joblib'),texture)
                rows.append({'run':p.name,'test':metric(src).evaluate(model,src),
                             'note':'identical unique-pixel evaluation support to neural-network models'})
        del src
        for fold in manifest['target_folds']:
            ds=Data(target_records(fold['test']),norm,stats);ev=metric(ds)
            for p in [x for x in expected if x.name.startswith(f"target_{norm}_g{fold['test_group']}_")]+[OUT/f'source_{norm}_seed42']:
                model=UNet(8,3).to(DEVICE);model.load_state_dict(torch.load(p/'best.pth',map_location=DEVICE,weights_only=True))
                rows.append({'run':p.name,'test_group':fold['test_group'],'test':ev.evaluate(model,ds)})
    dump(result,rows)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--train',action='store_true');ap.add_argument('--evaluate',action='store_true');ap.add_argument('--limit-runs',type=int);args=ap.parse_args()
    manifest=prepare();stats=source_stats(manifest['source']['training'])
    print('Frozen schedule: 6 source runs and 600 target runs. GPU:',DEVICE,flush=True)
    if args.train:train_all(manifest,stats,args.limit_runs)
    elif args.evaluate:evaluate(manifest,stats)
    else:print('Prepared only. Test metrics remain unopened.',flush=True)
