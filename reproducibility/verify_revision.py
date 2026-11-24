"""Verify the released evidence without training or loading the image dataset."""
import argparse
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
CODE=HERE/'analysis/experiments'

def run_folder(record):
    return CODE/record['run_path'].replace(chr(92),'/')


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--weights',action='store_true');args=ap.parse_args()
    manifest=json.loads((HERE/'package_manifest.json').read_text())
    for r in manifest['files']:
        p=HERE/r['file'];assert p.exists(),p
        assert p.stat().st_size==r['bytes'] and digest(p)==r['sha256'],p
    epochs=json.loads((CODE/'checkpoint_epochs/checkpoint_epochs.json').read_text())
    assert len(epochs['runs'])==224
    for r in epochs['runs']:
        folder=run_folder(r)
        meta=json.loads((folder/'complete.json').read_text());hist=json.loads((folder/'history.json').read_text())
        best=-1.;selected=0
        for h in hist:
            v=h['validation']['f1'][1]
            if v>best+1e-4:best=v;selected=h['epoch']
        assert selected==r['selected_epoch']==meta['selected_epoch']
        assert len(hist)==r['epochs_run']==meta['epochs_run']
        assert abs(best-meta['validation_f1'])<1e-10
    partitions=json.loads((CODE/'staged_v2/partitions.json').read_text())
    assert len(partitions['source']['training'])==980 and len(partitions['source']['validation'])==215
    assert [f['test_group'] for f in partitions['target_folds']]==[2,4,6,7,8]
    for f in partitions['target_folds']:
        tr,va,te=map(set,[f['training'],f['validation'],f['test']])
        assert not tr&va and not tr&te and not va&te
    selected={r['run_path']:r for r in epochs['runs']}
    target=[r for r in epochs['runs'] if r['domain']=='target'];assert len(target)==205
    for group in [2,4,6,7,8]:
        for seed in [42,43,44,45,46]:
            def ids(stage,budget,strategy):
                rr=[r for r in target if r['test_group']==group and r['seed']==seed and r['stage']==stage and r['budget']==budget and r['strategy']==strategy]
                assert len(rr)==1
                return set(json.loads((run_folder(rr[0])/'complete.json').read_text())['train_ids'])
            a=ids(2,'21','scratch');b=ids(1,'63','scratch');c=ids(2,'complete pool','scratch')
            assert a<=b<=c
            assert a==ids(2,'21','full') and b==ids(1,'63','full')==ids(4,'63','partial')
            assert c==ids(2,'complete pool','full')
    for stage,n in [(1,50),(2,100),(3,30),(4,25)]:
        result=json.loads((CODE/f'staged_v2/stage{stage}/test_results.json').read_text())
        fitted=[r for r in result if r['run']!='direct_transfer'];assert len(fitted)==n
        support={}
        for r in result:
            cm=r['test']['confusion'];counts=[sum(row) for row in cm]
            g=r['test_group']
            if g in support:assert support[g]==counts
            else:support[g]=counts
            if g==4:assert counts[0]==counts[2]==0 and counts[1]>0
    wm=json.loads((HERE/'weights_manifest.json').read_text());assert len(wm['models'])==230
    checked=0
    if args.weights:
        for r in wm['models']:
            p=HERE/r['file'];assert p.exists(),f'Missing {p}; install {r["asset"]}'
            assert p.stat().st_size==r['bytes'] and digest(p)==r['sha256'],p
            checked+=1
    print(json.dumps({'status':'passed','evidence_files_checked':len(manifest['files']),
                      'neural_histories_checked':224,'target_runs':205,'source_neural_runs':19,
                      'models_in_weight_manifest':230,'weights_checked':checked,
                      'paired_subsets_and_shared_test_support':'passed'},indent=2))


if __name__=='__main__':main()
