"""Descriptive force timelines and retained new-replay wrist frames; no gate changes."""
import sys,itertools,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from pact_v1010b_contract import B,read,sha,freeze,now
import numpy as np,h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
scene='98c64a358f22eddaba4754f97d40325e9f3e9151f6968d553123681be884a33f'
out=B/'descriptive_smoke';out.mkdir(exist_ok=True)
fig,axes=plt.subplots(2,2,figsize=(11,7),sharex='col',constrained_layout=True)
montage,grid=plt.subplots(2,4,figsize=(12,5),constrained_layout=True)
report={'schema':'pact_v1010b_grasp_v1','utc':now(),'source_code_sha256':sha(__file__),'scene_id':scene,'scope':'New selected mechanism replay pair; descriptive, not a prevalence or intervention-effect estimate.','arms':{}}
for i,arm in enumerate(('ACT','PACT')):
 path=B/f'rollouts/A1/A1_{arm}_3104_frozen60000_{scene[:16]}/attempt_00/grasp_diagnostics.h5'
 with h5py.File(path) as h:
  meta=json.loads(h.attrs['metadata']);close=int(h.attrs['close_control_index']);steps=h['physics/step'][()];times=h['physics/time'][()];contacts=h['contacts/solver_samples'][()]
  forces=np.zeros((len(steps),2));counts=np.zeros((len(steps),2),int);cup=set(meta['cup_geom_ids']);bodies=np.array(meta['geom_body_ids']);padbodies=meta['body_ids'][1:]
  vertical=np.zeros(len(steps))
  for row in contacts:
   frame,g1,g2=int(row[0]),int(row[2]),int(row[3]);which=None
   if g1 in cup and bodies[g2] in padbodies:which=padbodies.index(int(bodies[g2]));fz=-row[21]
   elif g2 in cup and bodies[g1] in padbodies:which=padbodies.index(int(bodies[g1]));fz=row[21]
   if which is not None:
    forces[frame,which]+=row[13];counts[frame,which]+=1;vertical[frame]+=fz
  axes[i,0].plot(steps,forces[:,0],lw=.7,label='Left pad');axes[i,0].plot(steps,forces[:,1],lw=.7,label='Right pad')
  axes[i,0].set_ylabel(f'{arm}: pad normal force (N)');axes[i,0].axvline(close,color='k',ls='--',lw=1);axes[i,0].set_xlim(close-20,close+60);axes[i,0].legend(fontsize=8)
  body=h['control/body_pose'][()].reshape(901,3,7);lift=(body[:,0,2]-body[0,0,2])*1000
  axes[i,1].plot(np.arange(901),lift,label='Cup origin lift');axes[i,1].axvline(close,color='k',ls='--',lw=1);axes[i,1].axhline(10,color='grey',ls=':',lw=1);axes[i,1].set_xlim(close-20,close+60);axes[i,1].set_ylabel(f'{arm}: lift (mm)')
  indices=h['wrist/control_index'][()]
  for j,delta in enumerate((-10,0,15,40)):
   frame=int(np.argmin(abs(indices-(close+delta))));grid[i,j].imshow(h['wrist/rgb'][frame]);grid[i,j].set_title(f'{arm}, control {indices[frame]}');grid[i,j].axis('off')
  mask=(steps>=close-20)&(steps<=close+60);dt=np.r_[np.diff(times),0]
  report['arms'][arm]={'source_h5_sha256':sha(path),'close_control_index':close,'window_control':[close-20,close+60],
   'pad_peak_normal_force_N':forces[mask].max(0).tolist(),'bilateral_contact_seconds_in_window':float(dt[mask & np.all(counts>0,axis=1)].sum()),
   'max_cup_origin_lift_mm_in_window':float(lift[max(0,close-20):close+61].max()),
   'note':'Sum of solver contact normal forces per pad. Bilateral duration is total in the displayed window, not longest continuous hold. Vertical force uses the verified geom2 sign convention.'}
for ax in axes[-1]:ax.set_xlabel('Control index (dashed: first close command)')
fig.suptitle('Original checkpoint3104, new replay of required acquisition case')
fig.savefig(out/'contact_and_lift.png',dpi=160);montage.savefig(out/'wrist_frames.png',dpi=160)
freeze(out/'evidence.json',report);print(json.dumps(report,indent=2))
