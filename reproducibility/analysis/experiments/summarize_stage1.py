"""Summarize all pre-specified stage-1 pairs; no test-based model selection."""
import json,re
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent/'staged_v2'
FOLDER=ROOT/'stage1'

def scores(cm):
    cm=np.asarray(cm,dtype=np.int64);tp=np.diag(cm);fp=cm.sum(0)-tp;fn=cm.sum(1)-tp
    return {'weed_f1':float(2*tp[1]/max(2*tp[1]+fp[1]+fn[1],1)),
            'weed_iou':float(tp[1]/max(tp[1]+fp[1]+fn[1],1)),
            'confusion':cm.tolist()}

def main():
    rows=json.loads((FOLDER/'test_results.json').read_text())
    expected={(g,s,k) for g in [2,4,6,7,8] for s in [42,43,44,45,46] for k in ['scratch','full']}
    found={};direct={}
    for row in rows:
        if row['run']=='direct_transfer':direct[row['test_group']]=row;continue
        m=re.fullmatch(r'target_patch_valid_g(\d+)_seed(\d+)_n63_(scratch|full)',row['run'])
        assert m,row['run']
        g,s,k=m.groups();key=(int(g),int(s),k);assert key not in found;found[key]=row
    assert set(found)==expected
    groups=[2,4,6,7,8];seeds=[42,43,44,45,46]
    d=np.array([[found[(g,s,'full')]['test']['f1'][1]-found[(g,s,'scratch')]['test']['f1'][1] for s in seeds] for g in groups])
    by_group=[]
    for i,g in enumerate(groups):
        r={'test_group':g,'paired_differences':d[i].tolist(),'mean_paired_difference':float(d[i].mean())}
        for k in ['scratch','full']:
            vals=[found[(g,s,k)]['test']['f1'][1] for s in seeds]
            r[k]={'weed_f1_mean':float(np.mean(vals)),'weed_f1_sample_sd':float(np.std(vals,ddof=1)),
                  'values':vals,'unique_test_pixels':found[(g,seeds[0],k)]['test']['unique_labelled_pixels']}
        by_group.append(r)
    pooled={}
    for k in ['scratch','full']:
        seed_scores=[]
        for s in seeds:
            cm=sum((np.asarray(found[(g,s,k)]['test']['confusion'],dtype=np.int64) for g in groups),np.zeros((3,3),dtype=np.int64))
            seed_scores.append(dict(seed=s,**scores(cm)))
        pooled[k]={'per_seed':seed_scores,
                   'weed_f1_mean':float(np.mean([x['weed_f1'] for x in seed_scores])),
                   'weed_f1_sample_sd':float(np.std([x['weed_f1'] for x in seed_scores],ddof=1)),
                   'weed_iou_mean':float(np.mean([x['weed_iou'] for x in seed_scores]))}
    direct_cm=sum((np.array(x['test']['confusion']) for x in direct.values()),np.zeros((3,3),dtype=np.int64))
    rng=np.random.default_rng(20261122);boot=[]
    for _ in range(10000):
        ix=rng.integers(0,5,5)
        boot.append(float(np.mean([d[g,rng.integers(0,5,5)].mean() for g in ix])))
    report={'stage':1,'training_runs':50,'budgets':[63],'spatial_test_groups':5,'seeds':seeds,
            'status':'training_and_test_evaluation_complete',
            'scope':'Retrospective within-field evaluation of five weed-bearing groups in one rapeseed field. Shared training data and five groups limit statistical inference; not external field validation.',
            'by_group':by_group,'pooled_confusions_across_groups_per_seed':pooled,
            'direct_transfer_pooled':scores(direct_cm),
            'paired_comparison':{'wins_full':int((d>0).sum()),'ties':int((d==0).sum()),'wins_scratch':int((d<0).sum()),
                'pairs':25,'mean_difference_equal_group_weight':float(d.mean()),
                'exploratory_hierarchical_bootstrap_95_interval':np.percentile(boot,[2.5,97.5]).tolist(),
                'interval_note':'Resamples spatial groups and paired repetitions, never individual pixels. Exploratory stability interval within this field, not a confidence interval across independent fields.'}}
    (FOLDER/'summary.json').write_text(json.dumps(report,indent=2))
    (FOLDER/'status.json').write_text(json.dumps({'status':'complete','stage':1,'training_complete':True,'test_evaluated':True,'summary_complete':True},indent=2))
    print(json.dumps({'stage_complete':True,'pooled_metrics':pooled,'paired_comparison':report['paired_comparison']},indent=2))

if __name__=='__main__':main()
