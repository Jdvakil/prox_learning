"""Summarize all frozen diagnostic rows and draw standalone research figures."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='1'
import collections,hashlib,json,sys
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import ConvexHull
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon,Rectangle

OUT=Path(__file__).parent;N=int(sys.argv[1]) if len(sys.argv)>1 else 234
def read(name):return json.loads((OUT/name).read_text())
def med(rr,k):
    v=[r[k] for r in rr if r.get(k) is not None]
    return float(np.median(v)) if v else None
def count(rr,k):return sum(bool(r.get(k)) for r in rr)
def safe(v):
    if isinstance(v,dict):return {str(k):safe(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [safe(x) for x in v]
    if isinstance(v,np.ndarray):return safe(v.tolist())
    if isinstance(v,np.generic):return safe(v.item())
    if isinstance(v,float) and not np.isfinite(v):return None
    return v
g=read(f'geometry_{N}.json');cm={r['rollout_id']:r for r in read(f'contact_modes_{N}.json')['rows']}
cg={r['rollout_id']:r for r in read(f'command_geometry_{N}.json')['rows']}
rows=[dict(r,**{k:v for k,v in cm[r['rollout_id']].items() if k not in r},
             **{k:v for k,v in cg[r['rollout_id']].items() if k not in r}) for r in g['rows']]
byid={r['rollout_id']:r for r in rows};paired={(r['episode_id'],r['arm']):r for r in rows}
summary={}
for seed in (3103,3104,3105):
    summary[seed]={}
    for arm in ('ACT','PACT'):
        rr=[r for r in rows if r['seed']==seed and r['arm']==arm]
        summary[seed][arm]={'n':len(rr),'counts':{k:count(rr,k) for k in
          ('success','touch','held','lift','support','collision_free_success','hazard_contact',
           'rim_blocked_signature','robot_clutter_before_first_touch','robot_clutter_near_close')},
          'stages':dict(collections.Counter(r['failure_stage'] for r in rr)),
          'centered_command_under_2cm':sum(r.get('command_at_close_xy_error_m',99)<.02 for r in rr),
          'successful_same_wall_grasp_over_1s':sum(r['success'] and r['same_wall_longest_s']>=1 for r in rr),
          'shallow_touch_no_hold':sum(r['failure_stage']=='touched_without_hold' and
            r.get('minimum_rim_relative_tcp_early_grasp_m') is not None and
            r['minimum_rim_relative_tcp_early_grasp_m']>=0 for r in rr),
          'near_close_fk_max_error_m':max(r.get('fk_verification_max_error_around_close_m',0) for r in rr),
          'medians':{k:med(rr,k) for k in ('first_close_xy_error_m','first_close_tcp_above_translated_rim_m',
              'minimum_rim_relative_tcp_early_grasp_m','command_at_close_tcp_above_rim_m',
              'initial_nearest_clutter_xy_aabb_m','longest_bilateral_pad_contact_s')},
          'by_family':{v:{'n':sum(r['family']==v for r in rr),'success':sum(r['success'] for r in rr if r['family']==v)}for v in sorted(set(r['family'] for r in rr))},
          'by_pose':{v:{'n':sum(r['pose_id']==v for r in rr),'success':sum(r['success'] for r in rr if r['pose_id']==v)}for v in sorted(set(r['pose_id'] for r in rr))},
          'by_side':{v:{'n':sum(r['side']==v for r in rr),'success':sum(r['success'] for r in rr if r['side']==v)}for v in sorted(set(r['side'] for r in rr))}}
        ss=summary[seed][arm]
        ss['touch_no_hold_medians']={k:med([r for r in rr if r['failure_stage']=='touched_without_hold'],k) for k in
          ('first_close_xy_error_m','minimum_rim_relative_tcp_early_grasp_m','command_z_change_2_steps_after_close_m',
           'actual_z_change_2_steps_after_close_m','longest_bilateral_pad_contact_s','command_at_close_xy_error_m')}
        ss['depth_sensitivity']={str(mm):{'n':len(sub), 'success':count(sub,'success'),'lift':count(sub,'lift')}
          for mm in (-10,-5,0,5) if (sub:=[r for r in rr if r.get('minimum_rim_relative_tcp_early_grasp_m') is not None
                                           and r['minimum_rim_relative_tcp_early_grasp_m']>=mm/1000]) is not None}
    pairs=[r for r in rows if r['seed']==seed and r['arm']=='PACT']
    summary[seed]['paired']={label:{'n':len(sub),'pact_stages':dict(collections.Counter(r['failure_stage'] for r in sub)),
      'median_pact_minus_act_minimum_rim_height_m':float(np.median(dd)) if (dd:=[r['minimum_rim_relative_tcp_early_grasp_m']-paired[r['episode_id'],'ACT']['minimum_rim_relative_tcp_early_grasp_m'] for r in sub if r.get('minimum_rim_relative_tcp_early_grasp_m') is not None and paired[r['episode_id'],'ACT'].get('minimum_rim_relative_tcp_early_grasp_m') is not None]) else None}
      for label in ('ACT_only','PACT_only','both_success','both_fail')
      if (sub:=[r for r in pairs if ('both_success' if r['success'] and paired[r['episode_id'],'ACT']['success'] else
        'PACT_only' if r['success'] else 'ACT_only' if paired[r['episode_id'],'ACT']['success'] else 'both_fail')==label]) is not None}

core={s:[r for r in rows if r['seed']==s and r['arm']=='PACT' and r['role_index']<2] for s in (3103,3104,3105)}
matching=[]
for cell in sorted(set(r['cell'] for r in core[3104])):
    a=[r for r in core[3104] if r['cell']==cell];b=[r for r in core[3105] if r['cell']==cell]
    assert len(a)==len(b)==2
    distances=np.array([[np.linalg.norm(np.array(x['initial_target_xyz'])[:2]-np.array(y['initial_target_xyz'])[:2]) for y in b] for x in a])
    ii,jj=linear_sum_assignment(distances)
    matching.extend({'cell':cell,'seed3104':a[i]['rollout_id'],'seed3105':b[j]['rollout_id'],
                     'initial_target_xy_distance_m':float(distances[i,j]),
                     'pact3104_success':a[i]['success'],'pact3105_success':b[j]['success'],
                     'command3104_centered':a[i].get('command_at_close_xy_error_m',99)<.02,
                     'command3105_centered':b[j].get('command_at_close_xy_error_m',99)<.02}for i,j in zip(ii,jj))
env={'balanced_core':{s:{'n':len(rr),'pact_success':count(rr,'success'),
         'centered_command_under_2cm':sum(r.get('command_at_close_xy_error_m',99)<.02 for r in rr)}for s,rr in core.items()},
     'same_cell_xy_assignment':matching,
     'same_cell_xy_assignment_median_distance_m':float(np.median([r['initial_target_xy_distance_m']for r in matching]))}

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
fig,axs=plt.subplots(2,3,figsize=(13,7.6),constrained_layout=True)
colors={'success':'#168c73','touched_without_hold':'#d34836','other':'#79838d'}
for j,seed in enumerate((3103,3104,3105)):
    rr=[r for r in rows if r['seed']==seed and r['arm']=='PACT']
    ax=axs[0,j]
    for label in ('success','touched_without_hold','other'):
        sub=[r for r in rr if ('success' if r['success'] else 'touched_without_hold' if r['failure_stage']=='touched_without_hold' else 'other')==label and r.get('first_close_tcp_above_translated_rim_m') is not None]
        ax.scatter([100*r['first_close_xy_error_m'] for r in sub],[1000*r['first_close_tcp_above_translated_rim_m']for r in sub],
                   c=colors[label],marker='o' if label=='success' else 'x',label=label.replace('_',' '),alpha=.8)
    ax.axhline(0,color='#222',lw=1,ls='--');ax.set(xlim=(0,12),ylim=(-35,35),title=f'PACT {seed}: {count(rr,"success")}/{len(rr)} success',xlabel='Hand–cup horizontal offset at closure (cm)',ylabel='TCP height above initial-orientation rim (mm)')
    omitted=sum(r.get('first_close_xy_error_m',0) is not None and (r['first_close_xy_error_m']>.12 or abs(r.get('first_close_tcp_above_translated_rim_m',0))>.035) for r in rr)
    ax.text(.98,.03,f'{omitted} outside axes; {sum(r["first_close_command_step"] is None for r in rr)} no closure',transform=ax.transAxes,ha='right',fontsize=8)
    ax=axs[1,j]
    for label in ('success','touched_without_hold','other'):
        sub=[r for r in rr if ('success' if r['success'] else 'touched_without_hold' if r['failure_stage']=='touched_without_hold' else 'other')==label]
        ax.scatter([100*r['initial_target_xyz'][0]for r in sub],[100*r['initial_target_xyz'][1]for r in sub],c=colors[label],marker='o'if label=='success'else'x',alpha=.8)
    ax.set(xlim=(74,85),ylim=(-9,9),xlabel='Initial cup world X (cm)',ylabel='Initial cup world Y (cm)',title='Initial positions: outcomes overlap spatially')
axs[0,0].legend(loc='upper left',fontsize=8)
fig.suptitle('Pickup geometry across all retained paired cases; seed 3103 sample is provisional unless n=50',fontsize=12)
fig.savefig(OUT/f'geometry_overview_{N}.png');fig.savefig(OUT/f'geometry_overview_{N}.pdf');plt.close(fig)

rep4=byid['final_3104_h100_PACT_98c64a358f22edda']
rep5=min((r for r in rows if r['seed']==3105 and r['arm']=='PACT' and r['success'] and r['cell']==rep4['cell']),
         key=lambda r:np.linalg.norm(np.array(r['initial_target_xyz'])[:2]-np.array(rep4['initial_target_xyz'])[:2]))
rep3=byid['final_3103_h100_PACT_621d8bd12acc6e72']
reps=[rep4,paired[rep4['episode_id'],'ACT'],rep5,rep3]
fig,axs=plt.subplots(3,4,figsize=(15,9),constrained_layout=True)
for j,r in enumerate(reps):
    a=np.load(OUT/'episodes'/(r['rollout_id']+'.npz'));command=np.load(OUT/'commands'/(r['rollout_id']+'.npz'))['command_tcp_world']
    c=r['first_close_command_step'];ix=np.arange(max(0,c-25),min(c+46,899));t=(ix-c)*.066
    rim=a['target_world'][ix,2]+r['target_collision_top_above_origin_m']
    ax=axs[0,j];ax.plot(t,(a['tcp_world'][ix,2]-rim)*1000,label='Actual TCP',color='#be4535');ax.plot(t,(command[ix,2]-rim)*1000,label='Commanded TCP (FK)',color='#286aa5',ls='--')
    ax.axhline(0,color='#444',ls=':',label='Cup rim reference');ax.axvline(0,color='#888',lw=.8)
    ax.set(ylim=(-45,85),ylabel='Height above rim (mm)',title=f'{r["arm"]} {r["seed"]}: {"success" if r["success"] else "pickup failure"}')
    if j==0:ax.legend(fontsize=8)
    ax=axs[1,j];delta=a['tcp_world'][ix,:2]-a['target_world'][ix,:2]
    vertices=np.array(r['target_collision_vertices_world']).reshape(5,8,3)
    for wall in (0,1,3,4):
        points=(vertices[wall,:,:2]-np.array(r['initial_target_xyz'])[:2])*100
        ax.add_patch(Polygon(points[ConvexHull(points).vertices],facecolor='#c9cfce',edgecolor='#6a7672',alpha=.65,zorder=0))
    ax.plot(delta[:,0]*100,delta[:,1]*100,color='#be4535');ax.scatter(*((a['tcp_world'][c,:2]-a['target_world'][c,:2])*100),marker='x',c='#222',s=60,label='Closure')
    ax.scatter(0,0,c='#168c73',s=30,label='Cup origin');ax.set(xlim=(-8,8),ylim=(-8,8),xlabel='Hand–cup X (cm)',ylabel='Hand–cup Y (cm)');ax.set_aspect('equal')
    ax=axs[2,j];ax.plot(t,a['lift'][ix]*1000,color='#168c73',label='Cup lift')
    ax.fill_between(t,0,5,where=a['bilateral_pad'][ix],alpha=.25,color='#286aa5',label='Both pads contact')
    ax.axvline(0,color='#888',lw=.8);ax.set(xlabel='Seconds from first close command',ylabel='Cup lift (mm)',ylim=(-2,100))
    if j==0:ax.legend(fontsize=8)
fig.suptitle('Recorded versus commanded motion; gray polygons show initial cup-wall footprints',fontsize=13)
fig.savefig(OUT/f'representative_trajectories_{N}.png');fig.savefig(OUT/f'representative_trajectories_{N}.pdf');plt.close(fig)

doc=safe({'source_sha256':{f:hashlib.sha256((OUT/f).read_bytes()).hexdigest()for f in
       (f'geometry_{N}.json',f'contact_modes_{N}.json',f'command_geometry_{N}.json')},
     'summary':summary,'environment':env,'representatives':[r['rollout_id']for r in reps],
     'representative_selection':'3104 failure with successful identical-instance ACT; closest successful 3105 initial target XY in the same nominal cell; 3103 failure with no concurrent robot/clutter contact and successful paired ACT. Illustrations are selected, but statistics include every validated pair in the source snapshot.',
     'rows':rows})
(OUT/f'analysis_{N}.json').write_text(json.dumps(doc,indent=2,allow_nan=False)+'\n')
print(json.dumps({'summary':summary,'balanced_core':env['balanced_core'],'representatives':doc['representatives']},indent=2),flush=True)
