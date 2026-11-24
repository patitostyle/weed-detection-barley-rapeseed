"""Export the selected epochs of all neural fits used in the revised study.

Read metadata and histories only; no training or checkpoint mutation.
"""
import csv
import hashlib
import json
import re
import statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'staged_v2'
OUT=ROOT/'checkpoint_epochs'


def inspect(folder,extra):
    info=json.loads((folder/'complete.json').read_text())
    hist=json.loads((folder/'history.json').read_text())
    assert [r['epoch'] for r in hist]==list(range(1,len(hist)+1)),folder
    assert info['epochs_run']==len(hist)<=30,folder
    best=-1.;selected=0
    for r in hist:
        v=float(r['validation']['f1'][1])
        if v>best+1e-4:best=v;selected=r['epoch']
    assert info['selected_epoch']==selected,folder
    assert abs(best-info['validation_f1'])<1e-10,folder
    assert 1<=selected<=len(hist)
    assert len(hist)==30 or len(hist)-selected==7,(folder,info)
    return {**extra,'seed':info['seed'],'selected_epoch':selected,'epochs_run':len(hist),
            'validation_weed_f1_at_selection':best,'stopped_before_epoch30':len(hist)<30,
            'run_path':str(folder.relative_to(ROOT)),
            'complete_json_sha256':hashlib.sha256((folder/'complete.json').read_bytes()).hexdigest(),
            'history_json_sha256':hashlib.sha256((folder/'history.json').read_bytes()).hexdigest()}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rows=[]
    for stage,norm,count in [(1,'patch_valid',50),(2,'patch_valid',100),(3,'source_fixed',30),(4,'patch_valid',25)]:
        files=sorted((BASE/f'stage{stage}').glob('target_*/complete.json'))
        assert len(files)==count
        for p in files:
            m=re.fullmatch(r'target_(patch_valid|source_fixed)_g(\d+)_seed(\d+)_n(\d+)_(scratch|full|partial)',p.parent.name)
            assert m and m[1]==norm
            n=int(m[4]);strategy=m[5]
            budget='complete pool' if stage==2 and n!=21 else str(n)
            r=inspect(p.parent,{'domain':'target','stage':stage,'model':'U-Net','input':'eight channels',
                                'normalization':norm,'test_group':int(m[2]),'budget':budget,
                                'adaptation_patches':n,'strategy':strategy})
            assert r['seed']==int(m[3]);rows.append(r)
    for config,label in [('rgb','RGB'),('rgb_nir_re','RGB + NIR + RE'),('rgb_indices','RGB + indices')]:
        for seed in [42,43,44]:
            rows.append(inspect(BASE/'source_inputs_benchmark'/f'unet_{config}_seed{seed}',
                                {'domain':'source','stage':'source spectral comparison','model':'U-Net','input':label,
                                 'normalization':'patch_valid','test_group':'','budget':'','adaptation_patches':'','strategy':'source'}))
    for seed in [42,43,44]:
        p=ROOT/'experiments_v1'/f'source_patch_valid_seed{seed}' if seed in [42,43] else BASE/'deeplab_benchmark/unet_seed44'
        rows.append(inspect(p,{'domain':'source','stage':'source model comparison','model':'U-Net','input':'eight channels',
                               'normalization':'patch_valid','test_group':'','budget':'','adaptation_patches':'','strategy':'source'}))
    for initialization in ['random','imagenet']:
        for seed in [42,43,44]:
            rows.append(inspect(BASE/'deeplab_benchmark'/f'deeplab_{initialization}_seed{seed}',
                                {'domain':'source','stage':'source model comparison','model':'DeepLabv3 '+initialization,'input':'eight channels',
                                 'normalization':'patch_valid','test_group':'','budget':'','adaptation_patches':'','strategy':'source'}))
    rows.append(inspect(BASE/'source_source_fixed_seed42',{'domain':'source','stage':'normalization comparison','model':'U-Net','input':'eight channels',
                                                         'normalization':'source_fixed','test_group':'','budget':'','adaptation_patches':'','strategy':'source'}))
    assert len(rows)==224 and len({r['run_path'] for r in rows})==224
    assert sum(r['domain']=='target' for r in rows)==205
    with (OUT/'checkpoint_epochs.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary={}
    for domain in ['source','target','all']:
        z=[r for r in rows if domain=='all' or r['domain']==domain]
        summary[domain]={'runs':len(z),'selected_epoch_min':min(r['selected_epoch'] for r in z),
                         'selected_epoch_max':max(r['selected_epoch'] for r in z),'selected_epoch_median':statistics.median(r['selected_epoch'] for r in z),
                         'stopping_epoch_min':min(r['epochs_run'] for r in z),'stopping_epoch_max':max(r['epochs_run'] for r in z),
                         'stopped_early':sum(r['stopped_before_epoch30'] for r in z)}
    for stage in [1,2,3,4]:
        z=[r for r in rows if r['domain']=='target' and r['stage']==stage]
        summary[f'stage{stage}']={'runs':len(z),'selected_epoch_min':min(r['selected_epoch'] for r in z),'selected_epoch_max':max(r['selected_epoch'] for r in z)}
    protocol={'scope':'All224 distinct neural fits used in revised study:205target,19source. Six RandomForest fits have no epoch-based checkpoint selection and are excluded. Reused source42/43 fits are counted once. No pilots included.',
              'selected_epoch':'One-based epoch of last accepted validation-weed-F1 improvement strictly greater than1e-4.',
              'epochs_run':'Actual epochs executed, distinct from selected epoch; maximum30,patience7.',
              'checks':'All224 metadata selections independently replayed from full validation histories; selected score, one-based epoch sequence, epoch cap and early-stopping gap verified.',
              'direct_transfer':'No target training, so no target selected epoch. Uses fixed source U-Net seed42 checkpoint; patch-valid epoch12, source-fixed epoch verified in source rows.',
              'original_files_modified':False}
    (OUT/'checkpoint_epochs.json').write_text(json.dumps({'protocol':protocol,'summary':summary,'runs':rows},indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
