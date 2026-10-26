from pathlib import Path
import os
import json, itertools
import numpy as np
from PIL import Image, ImageDraw

out=Path(os.environ.get('WEED_AUDIT_OUT',str(Path(__file__).resolve().parents[1]/'audit_rebuilt')))
out.mkdir(parents=True,exist_ok=True)
data=json.loads((out/'auditoria_datos_tfm.json').read_text())
rr=data['colza_v2_rgb_nir_re_indices']['records']
n=len(rr)
x=np.array([r['x'] for r in rr]); y=np.array([r['y'] for r in rr])
w=np.array([r['w'] for r in rr]); h=np.array([r['h'] for r in rr])
dx=np.maximum(0,np.maximum(x[:,None]-(x+w)[None,:],x[None,:]-(x+w)[:,None]))
dy=np.maximum(0,np.maximum((y-h)[:,None]-y[None,:],(y-h)[None,:]-y[:,None]))
overlap=(np.minimum((x+w)[:,None],(x+w)[None,:])-np.maximum(x[:,None],x[None,:])>1e-5)&(np.minimum(y[:,None],y[None,:])-np.maximum((y-h)[:,None],(y-h)[None,:])>1e-5)
weed=np.array([r['counts'].get('2',0)>0 for r in rr])
def components(gap):
    adj=np.hypot(dx,dy)<=gap+1e-5
    unseen=set(range(n)); groups=[]
    while unseen:
        start=min(unseen); unseen.remove(start); group=[start]; todo=[start]
        while todo:
            i=todo.pop(); near=set(np.flatnonzero(adj[i]))&unseen
            unseen-=near; todo.extend(near); group.extend(near)
        groups.append(sorted(group))
    return groups

report={'note':'Feasibility only; historical split lists are absent. Simulations do not reconstruct historical leakage.','group_counts':{}}
for gap in [0,7,14,28]:
    gg=components(gap)
    report['group_counts'][str(gap)]={'groups':len(gg),'weed_groups':int(sum(weed[g].any() for g in gg)),'sizes':[len(g) for g in gg]}

groups=components(14)
sx=w[0]/256; sy=h[0]/256
stats=[]
for ci,g in enumerate(groups):
    xmin=min(x[g]); ymax=max(y[g])
    cols=np.rint((x[g]-xmin)/sx).astype(int); rows=np.rint((ymax-y[g])/sy).astype(int)
    grid=np.zeros((rows.max()+256,cols.max()+256),dtype=np.uint8)
    conflicts=0
    for idx,c,r in zip(g,cols,rows):
        with Image.open((Path(os.environ.get('WEED_DATA_DIR',str(Path(__file__).resolve().parents[1]/'data')))/'patches'/'colza_v2_rgb_nir_re_indices'/'masks')/rr[idx]['id']) as im: a=np.array(im)
        tile=grid[r:r+256,c:c+256]
        conflicts+=int(((tile>0)&(a>0)&(tile!=a)).sum())
        tile[a>0]=a[a>0]
    counts={str(k):int((grid==k).sum()) for k in [1,2,3]}
    stats.append({'group':ci,'patches':len(g),'weed_patches':int(weed[g].sum()),'unique_labelled_pixels':counts,'conflicting_pixels':conflicts,'ids':[rr[i]['id'] for i in g]})
report['groups_14m']=stats

# Simulate the published random-patch procedure, then remove overlapping evaluation patches.
rng=np.random.default_rng(42)
report['random_sampling_simulations']={}
for size in [21,42,63]:
    vals=[]
    for repeat in range(300):
        tr=rng.choice(n,size,replace=False)
        selected=np.zeros(n,dtype=bool); selected[tr]=True
        shared=overlap[:,tr].any(axis=1)&~selected
        clean=~selected&~shared
        vals.append([int(shared.sum()),int(clean.sum()),int((clean&weed).sum())])
    report['random_sampling_simulations'][str(size)]={'columns':['remaining_test_patches_sharing_training_pixels','clean_test_patches','clean_test_weed_patches'],'quantiles_5_50_95':np.percentile(vals,[5,50,95],axis=0).tolist()}

# Choose complete groups before training, balancing annotations and patch counts only.
positive=[i for i,s in enumerate(stats) if s['weed_patches']]
total_unique=sum(s['unique_labelled_pixels']['2'] for s in stats)
candidates=[]
for te,va in itertools.permutations(positive,2):
    test_groups=[te]; val_groups=[va]
    # Ensure each evaluation set contains reference labels for all three classes.
    for selected in [test_groups,val_groups]:
        for cls in ['1','2','3']:
            if sum(stats[j]['unique_labelled_pixels'][cls] for j in selected)==0:
                available=[j for j in range(len(groups)) if j not in test_groups+val_groups and stats[j]['unique_labelled_pixels'][cls]>0]
                if available:selected.append(min(available,key=lambda j:len(groups[j])))
    test=[i for j in test_groups for i in groups[j]]
    val=[i for j in val_groups for i in groups[j]]
    train=[i for j,g in enumerate(groups) if j not in test_groups+val_groups for i in g]
    if len(train)<63 or weed[train].sum()<10: continue
    wt=sum(stats[j]['unique_labelled_pixels']['2'] for j in test_groups)/total_unique
    wv=sum(stats[j]['unique_labelled_pixels']['2'] for j in val_groups)/total_unique
    score=abs(wt-.2)+abs(wv-.15)+abs(len(test)/n-.2)+abs(len(val)/n-.15)
    candidates.append((score,te,va,train,val,test))
if candidates:
    _,te,va,tr,vaids,teids=min(candidates)
    partition={'train':tr,'validation':vaids,'test':teids}
    report['candidate_split']={}
    for key,ids in partition.items():
        gs=[j for j,g in enumerate(groups) if g[0] in ids]
        report['candidate_split'][key]={'patches':len(ids),'weed_patches':int(weed[ids].sum()),'groups':gs,'unique_labelled_pixels':{str(k):sum(stats[j]['unique_labelled_pixels'][str(k)] for j in gs) for k in [1,2,3]}}
        (out/f'propuesta_colza_{key}_ids.txt').write_text('\n'.join(rr[i]['id'] for i in sorted(ids))+'\n')
    report['candidate_split']['minimum_rectangle_distance_m']={a+'_'+b:float(np.hypot(dx[np.ix_(partition[a],partition[b])],dy[np.ix_(partition[a],partition[b])]).min()) for a,b in itertools.combinations(partition,2)}
    pic=Image.new('RGB',(1100,1000),'white'); d=ImageDraw.Draw(pic)
    scale=min(950/(max(x+w)-min(x)),850/(max(y)-min(y-h)))
    for key,ids in partition.items():
        color={'train':'#277da8','validation':'#e3a018','test':'#9c4db2'}[key]
        for i in ids:
            xx=40+(x[i]-min(x))*scale; yy=70+(max(y)-y[i])*scale
            d.rectangle((xx,yy,xx+w[i]*scale,yy+h[i]*scale),outline=color,width=1)
            if weed[i]:d.ellipse((xx+2,yy+2,xx+5,yy+5),fill='red')
    d.text((30,20),'Candidate: blue=train; orange=validation; purple=test; red=weed annotation',fill='black')
    pic.save(out/'propuesta_separacion_colza.png')
(out/'viabilidad_colza.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='groups_14m'},indent=2))
print('GROUPS',json.dumps([{k:v for k,v in s.items() if k!='ids'} for s in stats]))
