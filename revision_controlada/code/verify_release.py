"""Check archived evidence without PyTorch, source imagery, training or test tuning."""
from pathlib import Path
import ast,csv,json,math,statistics,hashlib

ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf8'))
results=read(ROOT/'reported_results/test_results.json')
assert len(results)==19
lookup={r['id']:r for r in read(ROOT/'auditoria_datos_tfm.json')['colza_v2_rgb_nir_re_indices']['records']}
split={k:(ROOT/f'propuesta_colza_{k}_ids.txt').read_text().splitlines() for k in ['train','validation','test']}
assert [len(split[k]) for k in split]==[269,57,97]
assert len(set(sum(split.values(),[])))==423
assert set(sum(split.values(),[]))==set(lookup)
distances={}
for a,b in [('train','validation'),('train','test'),('validation','test')]:
    distance=float('inf')
    for ai in split[a]:
        r=lookup[ai]
        for bi in split[b]:
            t=lookup[bi]
            dx=max(0,r['x']-t['x']-t['w'],t['x']-r['x']-r['w'])
            dy=max(0,r['y']-r['h']-t['y'],t['y']-t['h']-r['y'])
            distance=min(distance,math.hypot(dx,dy))
    assert distance>14-1e-5
    distances[a+'_'+b]=distance
for seed in [42,43,44]:
    subsets=[read(ROOT/f'reported_results/subset_seed{seed}_n{n}.json') for n in [21,42,63]]
    for n,ids in zip([21,42,63],subsets):
        assert len(ids)==len(set(ids))==n
        assert set(ids)<=set(split['train'])
        assert all(sum(lookup[x]['counts'].get(str(c),0) for x in ids)>0 for c in [1,2,3])
    assert subsets[0]==subsets[1][:21] and subsets[1]==subsets[2][:42]
expected={(s,n,seed) for s in ['transfer_full','scratch'] for n in [21,42,63] for seed in [42,43,44]}
assert {(r['strategy'],r['budget'],r['seed']) for r in results[1:]}==expected
for r in results:
    m=r['test'];cm=m['confusion']
    assert sum(map(sum,cm))==m['unique_labelled_pixels']==468018
    f1=[];iou=[]
    for c in range(3):
        tp=cm[c][c];fp=sum(row[c] for row in cm)-tp;fn=sum(cm[c])-tp
        f1.append(2*tp/max(2*tp+fp+fn,1));iou.append(tp/max(tp+fp+fn,1))
    assert all(abs(x-y)<1e-12 for x,y in zip(f1,m['f1']))
    assert all(abs(x-y)<1e-12 for x,y in zip(iou,m['iou']))
    assert abs(statistics.mean(iou)-m['miou'])<1e-12
    if r['strategy']=='direct_transfer':continue
    run=ROOT/f"reported_results/{r['strategy']}_seed{r['seed']}_n{r['budget']}"
    complete=read(run/'complete.json');history=read(run/'history.json')
    assert complete['train_ids']==read(ROOT/f"reported_results/subset_seed{r['seed']}_n{r['budget']}.json")
    best=-1;epoch=None
    for h in history:
        score=h['validation']['f1'][1]
        if score>best+1e-4:best=score;epoch=h['epoch']
    assert epoch==complete['selected_epoch']==r['selected_epoch']
    assert len(history)==complete['epochs_run']
for p in (ROOT/'code').glob('*.py'):ast.parse(p.read_text(encoding='utf8'))
# The portable experiment retains identical functions/classes; only paths differ.
def bodies(p):
    return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(p.read_text(encoding='utf8')).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
assert bodies(ROOT/'code/revision_colza_controlada.py')==bodies(ROOT/'provenance/revision_colza_controlada.py')
with (ROOT/'target_results.csv').open(newline='',encoding='utf8') as f:
    for row in csv.DictReader(f):
        matches=[r for r in results if r['strategy']==row['strategy'] and (r['strategy']=='direct_transfer' or (str(r['budget'])==row['adaptation_patches'] and str(r['seed'])==row['seed']))]
        assert len(matches)==1
        assert abs(float(row['test_weed_f1'])-matches[0]['test']['f1'][1])<1e-12
        assert abs(float(row['test_weed_iou'])-matches[0]['test']['iou'][1])<1e-12
manifest=ROOT/'SHA256.json'
if manifest.exists():
    for rel,digest in read(manifest).items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==digest,rel
print(json.dumps({'status':'passed','spatial_split_sizes':{k:len(v) for k,v in split.items()},'minimum_edge_distances_m':distances,'training_runs':18,'reported_evaluations':len(results),'unique_test_labelled_pixels':468018,'checks':['zero cross-subset image overlap','nested class-supported paired subsets','recomputed confusion metrics','validation checkpoint selection','matching CSV','portable experiment functions unchanged','all code parses','file hashes when manifest supplied']},indent=2))
