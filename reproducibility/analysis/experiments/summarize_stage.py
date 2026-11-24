"""Checked summaries for bounded stages 2-4, using every scheduled pair."""
import argparse,json,re
from pathlib import Path
import numpy as np

BASE=Path(__file__).resolve().parent/'staged_v2'
GROUPS=[2,4,6,7,8]

def parse(rows):
    found={}
    for x in rows:
        if x['run']=='direct_transfer':continue
        m=re.fullmatch(r'target_(patch_valid|source_fixed)_g(\d+)_seed(\d+)_n(\d+)_(scratch|full|partial)',x['run'])
        assert m,x['run']
        norm,g,s,n,k=m.groups();key=(norm,int(g),int(s),int(n),k)
        assert key not in found
        found[key]=x
    return found

def paired(a,b):
    d=np.asarray(a)-np.asarray(b);rng=np.random.default_rng(20261122)
    samples=[]
    for _ in range(10000):
        gs=rng.integers(0,len(d),len(d))
        samples.append(float(np.mean([d[g,rng.integers(0,d.shape[1],d.shape[1])].mean() for g in gs])))
    return {'pairs':int(d.size),'wins_first':int((d>0).sum()),'ties':int((d==0).sum()),'wins_second':int((d<0).sum()),
            'mean_paired_difference_equal_weight_per_group':float(d.mean()),
            'group_mean_differences':d.mean(1).tolist(),
            'exploratory_within_field_95_stability_interval':np.percentile(samples,[2.5,97.5]).tolist()}

def main(stage):
    folder=BASE/f'stage{stage}'
    found=parse(json.loads((folder/'test_results.json').read_text()))
    manifest=json.loads((BASE/'partitions.json').read_text())
    full_budget={f['test_group']:len(f['training']) for f in manifest['target_folds']}
    seeds=[42,43,44] if stage==3 else [42,43,44,45,46]
    norm='source_fixed' if stage==3 else 'patch_valid'
    budgets=[21,'full'] if stage==2 else [63]
    strategies=['partial'] if stage==4 else ['scratch','full']
    expected={(norm,g,s,full_budget[g] if n=='full' else n,k) for g in GROUPS for s in seeds for n in budgets for k in strategies}
    assert set(found)==expected,'Results differ from complete scheduled set'
    original=parse(json.loads((BASE/'stage1/test_results.json').read_text())) if stage in [3,4] else {}
    report={'stage':stage,'training_runs':len(found),'complete':True,'seeds':seeds,'budgets':budgets,
            'scope':'Within-field evaluation of five weed-bearing spatial groups; not independent-field replication.',
            'interval_note':'Exploratory hierarchical bootstrap over groups and paired repetitions. Small group count and overlapping training sets limit inference. No pixel-level resampling.',
            'metrics':[],'comparisons':[]}
    for budget in budgets:
        values={}
        for strategy in strategies:
            matrix=np.array([[found[(norm,g,s,full_budget[g] if budget=='full' else budget,strategy)]['test']['f1'][1] for s in seeds] for g in GROUPS])
            values[strategy]=matrix
            pooled=[]
            for s in seeds:
                cm=sum((np.array(found[(norm,g,s,full_budget[g] if budget=='full' else budget,strategy)]['test']['confusion']) for g in GROUPS),np.zeros((3,3),dtype=np.int64))
                tp=cm[1,1];fp=cm[:,1].sum()-tp;fn=cm[1].sum()-tp
                pooled.append(float(2*tp/max(2*tp+fp+fn,1)))
            report['metrics'].append({'budget':budget,'strategy':strategy,'group_by_seed_F1':matrix.tolist(),
                'pooled_group_confusions_F1_per_seed':pooled,'pooled_F1_mean':float(np.mean(pooled)),
                'pooled_F1_sample_sd':float(np.std(pooled,ddof=1))})
        if stage in [2,3]:
            report['comparisons'].append(dict(budget=budget,first='full',second='scratch',**paired(values['full'],values['scratch'])))
        if stage==3:
            for k in strategies:
                baseline=np.array([[original[('patch_valid',g,s,63,k)]['test']['f1'][1] for s in seeds] for g in GROUPS])
                report['comparisons'].append(dict(budget=63,first='source_fixed_'+k,second='patch_valid_'+k,**paired(values[k],baseline)))
        if stage==4:
            for k in ['full','scratch']:
                baseline=np.array([[original[('patch_valid',g,s,63,k)]['test']['f1'][1] for s in seeds] for g in GROUPS])
                report['comparisons'].append(dict(budget=63,first='partial',second=k,**paired(values['partial'],baseline)))
    (folder/'summary.json').write_text(json.dumps(report,indent=2))
    (folder/'status.json').write_text(json.dumps({'status':'complete','stage':stage,'training_complete':True,'test_evaluated':True,'summary_complete':True},indent=2))
    print('STAGE',stage,'COMPLETE: training, test evaluation and paired summary.',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage',type=int,choices=[2,3,4],required=True)
    main(ap.parse_args().stage)
