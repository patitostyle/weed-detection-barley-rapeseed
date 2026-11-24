"""Bounded seven-run source comparison requested by reviewer 1.

Three seeds each for random and ImageNet-backbone DeepLabv3. Reuses source
U-Net seeds42/43 and trains seed44 once. No target experiments are repeated.
Every model is assessed on the same unique annotated Barley-2 pixels.
"""
import argparse,hashlib,json,msvcrt
from pathlib import Path
import numpy as np
import torch
import run_revision as r
from architecture_comparison import DeepLab

BASE=r.HERE/'staged_v2';OUT=BASE/'deeplab_benchmark';OLD=r.HERE/'experiments_v1'
SEEDS=[42,43,44]

def config():
    assert json.loads((BASE/'remaining_queue_status.json').read_text())['status']=='complete'
    m=json.loads((BASE/'partitions.json').read_text());stats=json.loads((BASE/'source_scaling.json').read_text())['bounds']
    weights=r.HERE/'torch_cache/checkpoints/resnet50-11ad3fa6.pth'
    assert weights.exists(),'Official pretrained backbone weights missing'
    plan={'seeds':SEEDS,'normalization':'patch_valid','source_split':str(BASE/'partitions.json'),
          'models':['UNet','DeepLabv3_random','DeepLabv3_ImageNet_backbone'],
          'new_training_runs':7,'reused_unet_seeds':[42,43],
          'training':{'epochs_max':30,'patience':7,'min_delta':1e-4,'batch_size':4,
                      'lr':1e-4,'optimizer':'Adam','loss':'unweighted cross entropy',
                      'augmentation':'paired flips and rotations by multiples of 90 degrees'},
          'selection':'unique-pixel weed F1 on spatially separated Barley1 validation, never Barley2',
          'pretraining':'ResNet50 ImageNet1K V2 backbone only; random decoder',
          'extra_channels':'RGB kernels copied; extra five kernels use mean RGB kernel; all kernels scaled by 3/8',
          'input_scaling':'same patch-wise [0,1] inputs as U-Net; no claim that ImageNet fine-tuning was independently optimized',
          'weights_sha256':hashlib.sha256(weights.read_bytes()).hexdigest(),
          'test':'all available unique labelled Barley2-V2 geographic pixels, identical support for every model',
          'scope':'retrospective model comparison under specified settings, not architecture-wide ranking'}
    file=OUT/'protocol.json'
    if file.exists():assert r.protocol_matches(json.loads(file.read_text()),plan)
    else:r.dump(file,plan)
    return m,stats

def candidates():
    result=[]
    for seed in SEEDS:
        folder=OLD/f'source_patch_valid_seed{seed}' if seed in [42,43] else OUT/'unet_seed44'
        result.append(('UNet',seed,folder))
    for initialization in ['random','imagenet']:
        result += [(f'DeepLabv3_{initialization}',seed,OUT/f'deeplab_{initialization}_seed{seed}') for seed in SEEDS]
    return result

def train(m,stats):
    expected_train=[x['dataset']+'/'+x['id'] for x in m['source']['training']]
    expected_val=[x['dataset']+'/'+x['id'] for x in m['source']['validation']]
    for seed in [42,43]:
        folder=OLD/f'source_patch_valid_seed{seed}'
        info=json.loads((folder/'complete.json').read_text())
        assert info['train_ids']==expected_train and info['validation_ids']==expected_val
        assert info.get('batch_size',4)==4
    tr=r.Data(m['source']['training'],'patch_valid',stats,True)
    val=r.Data(m['source']['validation'],'patch_valid',stats)
    for model_name,seed,folder in candidates():
        if (folder/'complete.json').exists():continue
        r.dump(OUT/'status.json',{'status':'running','phase':'training','model':model_name,'seed':seed})
        r.legacy.seed_all(seed)
        model=r.UNet(8,3) if model_name=='UNet' else DeepLab(model_name=='DeepLabv3_imagenet')
        strategy='source' if model_name=='UNet' else 'source_deeplab'
        print('TRAINING',model_name,'seed',seed,flush=True)
        r.train_one(folder,tr,val,seed,strategy,model=model,batch_size=4)
    r.dump(OUT/'status.json',{'status':'training_complete','test_evaluated':False})

def evaluate(stats):
    rows=[];all_runs=candidates()
    assert all((folder/'complete.json').exists() for _,_,folder in all_runs)
    result=OUT/'test_results.json'
    if result.exists():return
    ds=r.Data([dict(x,dataset='cebada_v2_C2_rgb_nir_re_indices') for x in r.AUDIT['cebada_v2_C2_rgb_nir_re_indices']['records']],
              'patch_valid',stats)
    evaluator=r.metric(ds)
    support=None
    for name,seed,folder in all_runs:
        r.dump(OUT/'status.json',{'status':'running','phase':'evaluation','model':name,'seed':seed})
        print('EVALUATING',name,'seed',seed,flush=True)
        model=r.UNet(8,3) if name=='UNet' else DeepLab(False)
        model.load_state_dict(torch.load(folder/'best.pth',map_location='cpu',weights_only=True));model=model.to(r.DEVICE)
        metrics=evaluator.evaluate(model,ds)
        counts=np.array(metrics['confusion']).sum(1).tolist()
        if support is None:support=counts
        assert counts==support,'Different reference support between model evaluations'
        info=json.loads((folder/'complete.json').read_text())
        rows.append({'model':name,'seed':seed,'selected_epoch':info['selected_epoch'],'test':metrics,
                     'checkpoint_sha256':hashlib.sha256((folder/'best.pth').read_bytes()).hexdigest()})
        del model
        if r.DEVICE=='cuda':torch.cuda.empty_cache()
    r.dump(result,rows)

def summarize():
    rows=json.loads((OUT/'test_results.json').read_text())
    assert len(rows)==9
    output=[]
    for name in ['UNet','DeepLabv3_random','DeepLabv3_imagenet']:
        selected=[x for x in rows if x['model']==name];assert sorted(x['seed'] for x in selected)==SEEDS
        rec={'model':name,'runs':3,'selected_epochs':[x['selected_epoch'] for x in selected]}
        for label,values in [('weed_f1',[x['test']['f1'][1] for x in selected]),
                             ('weed_iou',[x['test']['iou'][1] for x in selected]),
                             ('miou',[x['test']['miou'] for x in selected])]:
            rec[label+'_mean']=float(np.mean(values));rec[label+'_sample_sd']=float(np.std(values,ddof=1))
        output.append(rec)
    r.dump(OUT/'summary.json',{'models':output,'support':'identical unique annotated Barley2 pixels for all 9 models',
        'scope':'variation across three training seeds on a shared retrospective evaluation field; not independent-field replication',
        'note':'No architecture-specific hyperparameter search; ImageNet backbone pretraining compared under declared multispectral input adaptation.'})
    r.dump(OUT/'status.json',{'status':'complete','training_complete':True,'test_evaluated':True,'summary_complete':True})
    print('DEEPLAB BENCHMARK COMPLETE: seven new fits, nine matched evaluations and summary.',flush=True)

def check_gpu():
    r.legacy.seed_all(42);model=DeepLab(True).to(r.DEVICE);model.train()
    a=torch.zeros(4,8,256,256,device=r.DEVICE);y=torch.zeros(4,256,256,dtype=torch.long,device=r.DEVICE)
    scaler=torch.amp.GradScaler('cuda',enabled=r.DEVICE=='cuda')
    with torch.autocast(device_type=r.DEVICE,enabled=r.DEVICE=='cuda'):
        loss=torch.nn.functional.cross_entropy(model(a),y)
    scaler.scale(loss).backward()
    assert torch.isfinite(loss)
    print('DeepLab 8-channel GPU forward/backward check passed.',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--check',action='store_true');args=ap.parse_args()
    if args.check:check_gpu()
    else:
        BASE.mkdir(parents=True,exist_ok=True)
        with (BASE/'execution.lock').open('a+b') as lock:
            if lock.tell()==0:lock.write(b'0');lock.flush()
            lock.seek(0)
            try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
            except OSError:print('Otra etapa sigue ejecutandose.',flush=True);raise SystemExit(2)
            try:
                m,stats=config()
                if not (OUT/'summary.json').exists():train(m,stats);evaluate(stats);summarize()
                else:print('DeepLab benchmark already complete.',flush=True)
            finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
