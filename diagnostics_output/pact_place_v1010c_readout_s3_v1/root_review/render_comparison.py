"""Render verified endpoints without estimating significance from exposed scenes."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

D=Path(__file__).resolve().parents[1]
doc=json.loads((D/'comparison.json').read_text())
assert doc['independent_raw_trajectories']==450 and doc['all_initial_pairings_passed']
methods=('ACT','PACT-Finetune','PACT-Frozen')
blocks=('3103','3104','3105','pooled')
colors=('#64748b','#0f766e','#ca8a04')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})
fig,axes=plt.subplots(2,1,figsize=(11,7.4),sharex=True)
x=np.arange(4);width=.24
for ax,(key,title) in zip(axes,(('success','Task success'),('cfts','Collision-free task success'))):
    for index,(method,color) in enumerate(zip(methods,colors)):
        values=[doc['summaries'][block][method] for block in blocks]
        heights=[100*v[key]/v['n'] for v in values]
        bars=ax.bar(x+(index-1)*width,heights,width=.22,color=color,label=method,zorder=3)
        for bar,value in zip(bars,values):
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1.3,
                    f"{value[key]}/{value['n']}",ha='center',va='bottom',fontsize=9)
    ax.set_ylim(0,100);ax.set_yticks(np.arange(0,101,20))
    ax.set_ylabel('Episodes (%)');ax.set_title(title,loc='left',fontsize=12,fontweight='bold')
    ax.grid(axis='y',color='#e2e8f0',linewidth=.7,zorder=0)
    for side in ('top','right'):ax.spines[side].set_visible(False)
    ax.spines['left'].set_color('#cbd5e1');ax.spines['bottom'].set_color('#cbd5e1')
    ax.axvline(2.5,color='#cbd5e1',linestyle='--',linewidth=.8)
axes[-1].set_xticks(x,['Seed 3103','Seed 3104','Seed 3105','All three seeds'])
fig.suptitle('ACT vs PACT-Finetune vs PACT-Frozen',fontsize=16,fontweight='bold',y=.98)
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.941),ncol=3,frameon=False)
fig.text(.08,.025,'50 matched regression scenes per seed; 150 per method overall. Final checkpoints: 60,000 updates.',fontsize=9,color='#475569')
fig.subplots_adjust(top=.855,bottom=.10,left=.08,right=.98,hspace=.22)
directory=D/'figures';directory.mkdir(exist_ok=True)
for suffix in ('png','pdf'):fig.savefig(directory/f'three_seed_comparison.{suffix}',dpi=180,facecolor='white')
plt.close(fig)
manifest={'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'comparison_sha256':hashlib.sha256((D/'comparison.json').read_bytes()).hexdigest(),
    'artifacts':{str(p.relative_to(D)):hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.glob('three_seed_comparison.*')}}
(directory/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2))
