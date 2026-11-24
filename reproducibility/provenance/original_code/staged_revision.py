"""Bounded stages, fixed in advance; never selects experiments using test scores.

Stage 1: 63-patch paired comparison, 5 spatial test groups, 5 seeds, 50 runs.
Stage 2: 21 patches and full adaptation pool, same pairs, 100 further runs.
Stage 3: alternative source-fixed scaling at 63 patches, 3 seeds, 30 runs
         plus its matching source model (seed 42, not selected by performance).
Stage 4: partial tuning at 63 patches, original scaling, 5 seeds, 25 runs.
Later source architecture comparisons are separate; no automatic all-stage queue.
"""
import argparse,hashlib,json,random,sys,time
import msvcrt
from pathlib import Path
import torch
import run_revision as r

OLD=r.HERE/'experiments_v1'
BASE=r.HERE/'staged_v2'

def prepare():
    plan={'version':2,'reason':'author requested smaller bounded stages before target testing',
          'seeds':[42,43,44,45,46],'batch_size':8,'epochs_max':30,'patience':7,
          'source_checkpoint':'source_patch_valid_seed42/best.pth; fixed seed, not best of source runs',
          'stage1':{'normalization':'patch_valid','budgets':[63],'strategies':['scratch','full'],'runs':50},
          'stage2':{'normalization':'patch_valid','budgets':[21,'full'],'strategies':['scratch','full'],'runs':100},
          'stage3':{'normalization':'source_fixed','budgets':[63],'strategies':['scratch','full'],'seeds':[42,43,44],'runs':30,'additional_source_runs':1},
          'stage4':{'normalization':'patch_valid','budgets':[63],'strategies':['partial'],'runs':25},
          'policy':'training stages launched explicitly; no automatic extension or test-driven subset choice',
          'test_policy':'63-patch stage evaluated only after all its 50 runs finish; learning-curve extension fixed before seeing test results',
          'resume':'atomic last.pth each completed epoch: model/optimizer/scaler/RNG/loader state; interrupted epoch repeats',
          'metrics':'unique geographic pixel per acquisition; no three-class mIoU for single-class test group'}
    r.OUT=BASE
    path=BASE/'staged_protocol.json'
    if path.exists():assert json.loads(path.read_text())==plan
    else:r.dump(path,plan)
    m=json.loads((OLD/'partitions.json').read_text());s=json.loads((OLD/'source_scaling.json').read_text())
    for name,value in [('partitions.json',m),('source_scaling.json',s)]:
        dest=BASE/name
        if dest.exists():assert json.loads(dest.read_text())==value
        else:r.dump(dest,value)
    checkpoint=OLD/'source_patch_valid_seed42'/'best.pth'
    assert (checkpoint.parent/'complete.json').exists()
    meta={'path':str(checkpoint),'sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
          'selection':'pre-specified seed42; no source-score comparison used'}
    provenance=BASE/'source_checkpoint_provenance.json'
    if provenance.exists():assert json.loads(provenance.read_text())==meta,'Source checkpoint changed'
    else:r.dump(provenance,meta)
    return m,s['bounds'],checkpoint

def run(stage):
    m,stats,source=prepare()
    finished=BASE/f'stage{stage}'/'summary.json'
    if finished.exists():
        print('STAGE',stage,'already complete, including evaluation and summary.',flush=True)
        return
    settings={1:('patch_valid',[63],['scratch','full'],r.SEEDS),
              2:('patch_valid',[21,'full'],['scratch','full'],r.SEEDS),
              3:('source_fixed',[63],['scratch','full'],[42,43,44]),
              4:('patch_valid',[63],['partial'],r.SEEDS)}
    norm,budgets,strategies,seeds=settings[stage]
    if stage==3:
        source=BASE/'source_source_fixed_seed42'/'best.pth'
        if not (source.parent/'complete.json').exists():
            tr=r.Data(m['source']['training'],norm,stats,True);val=r.Data(m['source']['validation'],norm,stats)
            # Match the source seed42 reference's batch size. Only the input
            # normalization changes in this source sensitivity comparison.
            source_batch=json.loads((OLD/'source_patch_valid_seed42'/'complete.json').read_text()).get('batch_size',4)
            r.dump(BASE/'source_normalization_comparison.json',{
                'source_batch_size_both':source_batch,'target_batch_size_both':8,
                'source_seed_both':42,'reason':'match source training settings across normalization variants'})
            r.train_one(source.parent,tr,val,42,'source',batch_size=source_batch)
            del tr,val
    names=[x['id'] for x in r.AUDIT['colza_v2_rgb_nir_re_indices']['records']]
    parent=r.Data(r.target_records(names),norm,stats)
    stagefolder=BASE/f'stage{stage}';stagefolder.mkdir(exist_ok=True)
    r.dump(stagefolder/'status.json',{'status':'running','stage':stage})
    for fold in m['target_folds']:
        val=r.SubsetData(parent,fold['validation']);pool=fold['training']
        for seed in seeds:
            rng=random.Random(seed);order=pool.copy()
            for _ in range(10000):
                rng.shuffle(order);ds=r.SubsetData(parent,order[:21])
                if all(sum(int((mask==c).sum()) for mask in ds.masks)>0 for c in range(3)):break
            else:raise RuntimeError('No admissible three-class subset')
            for budget in budgets:
                n=len(pool) if budget=='full' else budget
                ds=r.SubsetData(parent,order[:n],True)
                subset=BASE/'subsets'/f"g{fold['test_group']}_seed{seed}_n{n}.json"
                if subset.exists():assert json.loads(subset.read_text())==ds.names
                else:r.dump(subset,ds.names)
                for strategy in strategies:
                    path=stagefolder/f"target_{norm}_g{fold['test_group']}_seed{seed}_n{n}_{strategy}"
                    r.train_one(path,ds,val,seed,strategy,None if strategy=='scratch' else source,batch_size=8)
    r.dump(stagefolder/'status.json',{'status':'training_complete','stage':stage,'test_evaluated':False})
    print('STAGE',stage,'TRAINING COMPLETE. No next stage was launched.',flush=True)

def evaluate(stage):
    # Evaluation is explicit and never runs as a side effect of resuming training.
    m,stats,source=prepare()
    norm,budgets,strategies,seeds={1:('patch_valid',[63],['scratch','full'],r.SEEDS),
        2:('patch_valid',[21,'full'],['scratch','full'],r.SEEDS),
        3:('source_fixed',[63],['scratch','full'],[42,43,44]),
        4:('patch_valid',[63],['partial'],r.SEEDS)}[stage]
    folder=BASE/f'stage{stage}';expected=[]
    for fold in m['target_folds']:
        for seed in seeds:
            for budget in budgets:
                n=len(fold['training']) if budget=='full' else budget
                expected += [folder/f"target_{norm}_g{fold['test_group']}_seed{seed}_n{n}_{s}" for s in strategies]
    assert all((p/'complete.json').exists() for p in expected),'Stage training is incomplete; test remains closed'
    dest=folder/'test_results.json'
    if dest.exists():raise RuntimeError('Test results already exist; preserve them')
    if stage==3:source=BASE/'source_source_fixed_seed42'/'best.pth'
    results=[]
    for fold in m['target_folds']:
        ds=r.Data(r.target_records(fold['test']),norm,stats);ev=r.metric(ds)
        model=r.UNet(8,3).to(r.DEVICE)
        model.load_state_dict(torch.load(source,map_location=r.DEVICE,weights_only=True))
        results.append({'run':'direct_transfer','test_group':fold['test_group'],'test':ev.evaluate(model,ds)})
        for path in expected:
            if f"_g{fold['test_group']}_" not in path.name:continue
            model.load_state_dict(torch.load(path/'best.pth',map_location=r.DEVICE,weights_only=True))
            results.append({'run':path.name,'test_group':fold['test_group'],'training':json.loads((path/'complete.json').read_text()),'test':ev.evaluate(model,ds)})
    r.dump(dest,results)
    r.dump(folder/'status.json',{'status':'evaluation_complete','stage':stage,'test_evaluated':True})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage',type=int,choices=[1,2,3,4],required=True)
    ap.add_argument('--evaluate',action='store_true');a=ap.parse_args()
    BASE.mkdir(parents=True,exist_ok=True)
    with (BASE/'execution.lock').open('a+b') as lock:
        if lock.tell()==0:lock.write(b'0');lock.flush()
        lock.seek(0)
        try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:
            print('Ya hay una etapa ejecutandose. No se inicia una segunda copia.',flush=True);sys.exit(2)
        try:evaluate(a.stage) if a.evaluate else run(a.stage)
        finally:
            lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
