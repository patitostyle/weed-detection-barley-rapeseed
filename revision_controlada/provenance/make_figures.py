from pathlib import Path
import json,re,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
from PIL import Image

HERE=Path(__file__).parent;WORK=HERE.parent;ROOT=Path('C:/datos tfm');FIG=HERE/'figures';FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
def save(fig,name):
    fig.savefig(FIG/(name+'.png'),dpi=300,bbox_inches='tight');fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight');plt.close(fig)

def split_map():
    report=json.loads((WORK/'viabilidad_colza.json').read_text());records=json.loads((WORK/'auditoria_datos_tfm.json').read_text())['colza_v2_rgb_nir_re_indices']['records'];look={r['id']:r for r in records}
    fig,ax=plt.subplots(figsize=(7,5.8));colors={'train':'#277da8','validation':'#d89000','test':'#8848a4'}
    x0=min(r['x'] for r in records);y0=min(r['y']-r['h'] for r in records)
    for key in colors:
        names=(WORK/f'propuesta_colza_{key}_ids.txt').read_text().splitlines()
        for name in names:
            r=look[name];xx=r['x']-x0;yy=r['y']-r['h']-y0
            ax.add_patch(plt.Rectangle((xx,yy),r['w'],r['h'],fill=False,ec=colors[key],lw=.45))
            if r['counts'].get('2',0):ax.plot(xx+r['w']/2,yy+r['h']/2,'.',color='#d62728',ms=2)
    ax.set_xlim(-8,max(r['x']+r['w'] for r in records)-x0+8);ax.set_ylim(-8,max(r['y'] for r in records)-y0+8);ax.set_aspect('equal');ax.set_xlabel('Easting relative to patch extent (m)');ax.set_ylabel('Northing relative to patch extent (m)')
    handles=[Patch(facecolor='none',edgecolor=colors[k],label=label) for k,label in [('train','Adaptation pool (269)'),('validation','Validation (57)'),('test','Fixed test (97)')]]
    handles.append(plt.Line2D([],[],marker='.',ls='',color='#d62728',label='Patch contains weed annotations'))
    ax.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,1.22),ncol=2,frameon=False,fontsize=8)
    save(fig,'Figure2_spatial_partition')

def source_history():
    txt=(ROOT/'resumen_modelos'/'unet_train_v1v2_test_c2_30ep.txt').read_text(errors='replace')
    ep=[];f1=[]
    for line in txt.splitlines():
        m=re.search(r'Epoch (\d+).*?f1_maleza=([\d.]+)',line)
        if m:ep.append(int(m[1]));f1.append(float(m[2]))
    fig,(ax,axcm)=plt.subplots(2,1,figsize=(6.3,7.1),gridspec_kw={'height_ratios':[1,1.3]});ax.plot(ep,f1,'o-',color='black',ms=3,lw=1)
    best=int(np.argmax(f1));ax.scatter(ep[best],f1[best],color='#bc2828',zorder=3,label=f'Selected epoch {ep[best]} (F1 = {f1[best]:.4f})')
    ax.set(xlabel='Training epoch',ylabel='Barley 2 validation weed F1',ylim=(0,1),xlim=(.5,30.5));ax.legend(frameon=False,fontsize=9);ax.grid(axis='y',alpha=.2);ax.set_title('(A) Source checkpoint selection',loc='left',fontsize=11)
    verified=HERE/'source_checkpoint_verification.json'
    if verified.exists():
        cm=np.array(json.loads(verified.read_text())['confusion'],dtype=float);cm=cm/cm.sum(1,keepdims=True)
        im=axcm.imshow(cm,cmap='Blues',vmin=0,vmax=1);axcm.set_xticks([0,1,2],['Crop','Weed','Soil']);axcm.set_yticks([0,1,2],['Crop','Weed','Soil']);axcm.set(xlabel='Predicted class',ylabel='Reference class');axcm.set_title('(B) Recomputed row-normalized confusion',loc='left',fontsize=11)
        for i in range(3):
            for j in range(3):axcm.text(j,i,f'{cm[i,j]:.2f}',ha='center',va='center',color='white' if cm[i,j]>.5 else 'black')
        fig.colorbar(im,ax=axcm,fraction=.04,pad=.03)
    fig.tight_layout();save(fig,'Figure3_source_validation')

def qualitative():
    data=np.load(HERE/'qualitative_arrays.npz');names=data['names']
    cmap=ListedColormap(['#239b45','#e52c2c','#a16a35','#dedede'])
    fig,axes=plt.subplots(2,4,figsize=(10,5.6))
    for row,name in enumerate(names):
        rgb=data[f'rgb{row}'];ref=data[f'ref{row}'];pred=data[f'pred{row}']
        for col,img in enumerate([rgb,ref,pred,np.where(ref==3,3,pred)]):
            axes[row,col].imshow(img,cmap=cmap if col else None,vmin=0 if col else None,vmax=3 if col else None);axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
            if row==0:axes[row,col].set_title(['RGB input','Partial reference','Full prediction','Prediction on labelled pixels'][col],fontsize=9)
        axes[row,0].set_ylabel(name.replace('.tif',''),fontsize=9)
    fig.legend(handles=[Patch(color=cmap(i),label=n) for i,n in enumerate(['Crop','Weed','Soil','Unlabelled / ignored'])],loc='lower center',ncol=4,frameon=False,bbox_to_anchor=(.5,-.01));fig.tight_layout(rect=(0,.04,1,1));save(fig,'Figure4_partial_reference')

def target_plot():
    path=HERE/'experimentos_colza'/'test_results.json'
    if not path.exists():return
    rows=json.loads(path.read_text());base=rows[0]['test']['f1'][1]
    fig,ax=plt.subplots(figsize=(6.4,3.8))
    for strategy,color,label,offset in [('transfer_full','#1b658b','Full fine-tuning',-.5),('scratch','#b75b22','Training from scratch',.5)]:
        means=[];sd=[]
        for n in [21,42,63]:
            vals=[r['test']['f1'][1] for r in rows if r['strategy']==strategy and r['budget']==n]
            means.append(np.mean(vals));sd.append(np.std(vals,ddof=1))
            ax.scatter(np.full(3,n+offset),vals,color=color,s=14,alpha=.65)
        ax.errorbar(np.array([21,42,63])+offset,means,yerr=sd,color=color,marker='o',capsize=3,label=label)
    ax.axhline(base,color='gray',ls='--',label=f'Direct transfer ({base:.3f})');ax.set(xlabel='Adaptation patches used for weight updates',ylabel='Fixed-test weed F1',xticks=[21,42,63],ylim=(0,1));ax.legend(frameon=False,fontsize=8);ax.grid(axis='y',alpha=.2);save(fig,'Figure5_target_comparison')

def score_map():
    path=HERE/'score_map.npy'
    if not path.exists():return
    a=np.load(path);a=np.ma.masked_where(~np.isfinite(a)|(a<0),a)
    fig,ax=plt.subplots(figsize=(4.5,6));im=ax.imshow(a,cmap='YlGn',vmin=0,vmax=100);ax.set_xticks([]);ax.set_yticks([]);ax.set_title('Barley 2 aggregated model score',fontsize=11)
    cb=fig.colorbar(im,ax=ax,fraction=.035,pad=.03);cb.set_label('Stored score index (0–100)')
    save(fig,'FigureS1_illustrative_score')

if __name__=='__main__':split_map();source_history();qualitative();target_plot();score_map()
