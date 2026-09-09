"""Identify wall-pinch versus opposite-wall contacts in recorded physics samples."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='1'
import hashlib,json,re,sys
from pathlib import Path
import h5py
import numpy as np
OUT=Path(__file__).parent
ROOT=Path('/root/prox_learning_pact_remediation')
src=OUT/(sys.argv[1] if len(sys.argv)>1 else 'geometry_234.json')
rows=json.loads(src.read_text())['rows'];result=[]
def maxrun(a):
    edges=np.diff(np.r_[False,a,False].astype(int))
    return int(np.max(np.flatnonzero(edges==-1)-np.flatnonzero(edges==1),initial=0))
for i,r in enumerate(rows):
    path=ROOT/r['directory']/'telemetry.h5';before=path.stat()
    with h5py.File(path,'r') as h:
        identities=json.loads(h['contacts'].attrs['pair_identities'])
        pc=h['contacts/pair_counts'][()];ct=h['contacts/control_step'][()];t=h['contacts/sim_time_s'][()]
    assert (before.st_size,before.st_mtime_ns)==(path.stat().st_size,path.stat().st_mtime_ns)
    dt=float(np.median(np.diff(t)));pads=np.zeros((len(ct),2,5),dtype=bool)
    interactions={k:np.zeros(len(ct),dtype=bool) for k in
      ('robot_clutter','target_clutter','robot_hazard','robot_other','nonpad_gripper_target')}
    for j,ident in enumerate(identities):
        names=' '.join(ident['names'])
        isrobot='robot_0/' in names
        istarget='cavity_obj_0/Cup_10' in names
        ispad=any(f'gripper/{s}_pad' in names for s in ('left','right'))
        categories=[]
        if ident['class']=='clutter':
            if isrobot:categories.append('robot_clutter')
            if istarget:categories.append('target_clutter')
        if ident['class']=='hazard_bar' and isrobot:categories.append('robot_hazard')
        if ident['class'] in ('other_environment','mounted_fixture') and isrobot:categories.append('robot_other')
        if ident['class']=='grasp_target' and 'gripper/' in names and not ispad:categories.append('nonpad_gripper_target')
        indices=pc[pc[:,1]==j,0].astype(int)
        for key in categories:interactions[key][indices]=True
        if ident['class']!='grasp_target':continue
        m=re.search(r'Cup_10_cup_10_PrimitiveCollider_(\d)',names)
        if not m:continue
        wall=int(m[1])
        for s,side in enumerate(('left','right')):
            if f'gripper/{side}_pad' in names:
                pads[pc[pc[:,1]==j,0].astype(int),s,wall]=True
    same=(pads[:,0,[0,1,3,4]]&pads[:,1,[0,1,3,4]]).any(1)
    opposite=np.zeros(len(ct),bool)
    for left,right in ((0,3),(3,0),(1,4),(4,1)):
        opposite|=pads[:,0,left]&pads[:,1,right]
    bilateral=pads[:,0].any(1)&pads[:,1].any(1)
    adjacent=bilateral&~same&~opposite
    row={'rollout_id':r['rollout_id'],'episode_id':r['episode_id'],'seed':r['seed'],'arm':r['arm'],
         'success':r['success'],'failure_stage':r['failure_stage']}
    for name,mask in [('same_wall',same),('opposite_walls',opposite),('other_bilateral',adjacent)]:
        row[name+'_total_s']=float(mask.sum()*dt)
        row[name+'_longest_s']=float(maxrun(mask)*dt)
        row[name+'_first_control_step']=int(ct[np.flatnonzero(mask)[0]]) if mask.any() else None
    assert abs(bilateral.sum()*dt-r['bilateral_pad_contact_total_s'])<1e-8
    row['pad_wall_contact_total_s']=(pads.sum(0)*dt).tolist()
    close=r['first_close_command_step'];touch=r['first_touch_observation_step']
    for name,mask in interactions.items():
        row[name+'_first_step']=int(ct[np.flatnonzero(mask)[0]]) if mask.any() else None
        row[name+'_before_first_touch']=bool((mask&(ct<touch)).any()) if touch is not None else bool(mask.any())
        row[name+'_near_close']=bool((mask&(ct>=close-10)&(ct<=close+10)).any()) if close is not None else None
        row[name+'_total_s']=float(mask.sum()*dt)
    result.append(row)
    if i%50==0:print(i+1,len(rows),flush=True)
dest=OUT/f'contact_modes_{len(rows)}.json'
dest.write_text(json.dumps({'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),
 'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 'note':'Simultaneous recorded left/right finger-pad contacts on identical or opposing Cup_10 primitive side-wall boxes. No contact-force or pad-position inference.',
 'rows':result},indent=2)+'\n')
print('COMPLETE',len(rows),flush=True)
