"""Publication-friendly static figures; every selection is named in captions."""
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import *
from finalize_tables import LABELS,PANEL
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
OUT=E/'report/figures';OUT.mkdir(parents=True,exist_ok=True)
def finish(fig,name):
    fig.tight_layout()
    for ext in ['png','svg','pdf']:fig.savefig(OUT/(name+'.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig)
def main():
    ledger=read(E/'geometry_ledger.json')['rows'];refs=read(E/'reference_manifest.json')['rows'];sets=read(E/'tables/common_and_paired_sets.json')
    fig,ax=plt.subplots(figsize=(9,4));bottom=np.zeros(6);colors=['#176b91','#70bdca','#e9a23b','#b8b8b8'];names=['Main: finite CD','Low support: finite CD','Valid empty prediction','Unavailable / unresolved']
    for idx,label in enumerate(names):
        vals=[]
        for m in METHODS:
            rows=[r for r in ledger if r['method']==m]
            vals.append(sum((r['geometry_status']=='scored' and (r['reference_panel']=='main')==(idx==0)) if idx<2 else (r['geometry_status']=='empty_pred' if idx==2 else r['geometry_status'].startswith('NA')) for r in rows))
        ax.barh(np.arange(6),vals,left=bottom,color=colors[idx],label=label)
        for j,v in enumerate(vals):
            if v>=4:ax.text(bottom[j]+v/2,j,str(v),ha='center',va='center',color='white' if idx==0 else '#202020')
        bottom+=vals
    ax.set_yticks(np.arange(6),[LABELS[m] for m in METHODS]);ax.invert_yaxis();ax.set_xlim(0,100);ax.set_xlabel('Frozen people per method (N = 100)');ax.legend(loc='upper center',bbox_to_anchor=(.5,1.2),ncol=2,frameon=False);finish(fig,'coverage')
    counts=np.array([r['query_count'] for r in refs]);dt=np.abs([r['delta_ms'] for r in refs]);fig,axs=plt.subplots(1,2,figsize=(9,3.3))
    axs[0].hist(counts,bins=np.arange(5,116,5),color='#176b91',edgecolor='white');axs[0].axvline(20,color='#d58926',linestyle='--');axs[0].set(xlabel='Retained real returns / 5 cm voxels',ylabel='People')
    axs[1].scatter(dt,counts,c=np.where(counts>=20,'#176b91','#d58926'),s=22);axs[1].axhline(20,color='#999',linestyle='--');axs[1].axvline(50,color='#999',linestyle=':');axs[1].set(xlabel='Absolute camera-scan offset (ms)',ylabel='Retained real returns');finish(fig,'reference_support')
    fig,axs=plt.subplots(1,2,figsize=(10,3.8))
    for ax,key,label in [(axs[0],'common_finite_cd','CD (cm), common finite set N=45'),(axs[1],'common_admission','F1@10cm (%), common admission N=46')]:
        ids=sets['main/'+PANEL+'/'+key];data=[]
        for m in METHODS:
            records=[read(E/'results'/m/(sid+'.json'))['panels'][PANEL]['metrics'] for sid in ids]
            data.append([r['cd_m']*100 if key=='common_finite_cd' else r['f1_10cm']*100 for r in records])
        ax.boxplot(data,showfliers=False,medianprops={'color':'black'},widths=.55)
        for i,values in enumerate(data):
            x=i+1+np.linspace(-.17,.17,len(values));ax.scatter(x,sorted(values),s=11,alpha=.45,color='#176b91');ax.plot(i+1,np.mean(values),'D',color='#d58926',markersize=5)
        ax.set_xticks(range(1,7),[LABELS[m] for m in METHODS],rotation=30,ha='right');ax.set_ylabel(label)
        if key=='common_finite_cd':ax.set_yscale('log')
        else:ax.set_ylim(-3,103)
    finish(fig,'common_scores')
    sid='HN0004';ref=next(r for r in refs if r['id']==sid);q=dict(np.load(ref['queries']));fig,axs=plt.subplots(3,2,figsize=(10,7),sharex=True)
    for ax,m in zip(axs.flat,METHODS):
        d=read(E/'results'/m/(sid+'.json'));z=dict(np.load(d['returns']));hit=z['hit'];err=(z['range_pred']-q['r'])*100;x=np.arange(len(hit))
        ax.axhspan(-10,10,color='#e5f0ed');ax.axhline(0,color='#777',linewidth=.7);ax.scatter(x[hit],err[hit],s=14,color='#176b91');ax.scatter(x[~hit],np.zeros((~hit).sum()),marker='x',color='#ce6a28',s=25);ax.set_title(f'{LABELS[m]}: {hit.sum()}/{len(hit)} returns');ax.set_ylabel('Signed range error (cm)')
    axs[-1,0].set_xlabel('Fixed real ray index');axs[-1,1].set_xlabel('Fixed real ray index');fig.suptitle('HN0004: predetermined pilot identity; orange crosses = missing returns',y=1.01);finish(fig,'pilot_ray_errors')
    save(OUT/'manifest.json',dict(script_sha256=sha(__file__),source_ledger_sha256=sha(E/'geometry_ledger.json'),selection='coverage/reference: all100; score panels: frozen common sets; illustrative rays: predetermined pilot HN0004',files={p.name:sha(p) for p in OUT.glob('*') if p.suffix in ['.png','.svg','.pdf']}))
if __name__=='__main__':main()
