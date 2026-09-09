"""Extract geometry and contact timelines without simulation or policy inference."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
import collections
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path('/root/prox_learning_pact_remediation')
WORK=ROOT/'diagnostics_output/pact_place_v1010_wrist288_s3_v1'
OUT=Path(__file__).parent
CACHE=OUT/'episodes';CACHE.mkdir(exist_ok=True)
VERSION=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def decode(x):return json.loads(bytes(x).split(b'\0')[0])
def first(x):return int(np.flatnonzero(x)[0]) if np.any(x) else None
def runs(x):
    e=np.diff(np.r_[False,x,False].astype(int))
    return [(int(a),int(b)) for a,b in zip(np.flatnonzero(e==1),np.flatnonzero(e==-1))]
def maxrun(x):return max((b-a for a,b in runs(x)),default=0)
def safe(v):
    if isinstance(v,dict):return {str(k):safe(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [safe(x) for x in v]
    if isinstance(v,np.ndarray):return safe(v.tolist())
    if isinstance(v,np.generic):return safe(v.item())
    if isinstance(v,float) and not np.isfinite(v):return None
    return v

snapshot=[]
for seed in (3103,3104,3105):
    ledger=WORK/f'evaluation/final_{seed}_full50_h100_ledger.jsonl'
    records=[json.loads(x) for x in ledger.read_text().splitlines() if x]
    paired=collections.defaultdict(set)
    for r in records:
        if r['valid_completion']:paired[r['episode_id']].add(r['arm'])
    snapshot += [r for r in records if paired[r['episode_id']]=={'ACT','PACT'}]
snapshot_hash=hashlib.sha256(json.dumps(snapshot,sort_keys=True).encode()).hexdigest()
(OUT/f'snapshot_{len(snapshot)}.json').write_text(json.dumps(snapshot,indent=2)+'\n')

all_rows=[]
for index,record in enumerate(snapshot):
    d=Path(record['directory']);name=record['rollout_id'];dest=CACHE/(name+'.json')
    if dest.exists():
        saved=read(dest)
        assert saved['extractor_sha256']==VERSION and saved['result_sha256']==record['result_sha256']
        all_rows.append(saved);continue
    assert record['valid_completion'] and record['returncode']==0
    assert sha(d/'result.json')==record['result_sha256']
    assert sha(d/'exit_receipt.json')==record['exit_receipt_sha256']
    result=read(d/'result.json');job=read(d/'job.json');row=job['schedule']
    manifest=read(Path(job['command'][job['command'].index('--manifest')+1]))
    layout=next(x for x in manifest['rows'] if x['episode_id']==record['episode_id'])
    assert result['checkpoint_seed']==record['checkpoint_seed']
    with h5py.File(d/'trajectory.h5','r') as h:
        g=h['traj_0'];extra=g['obs/extra']
        ti=[decode(x) for x in extra['task_info'][()]]
        gs=[decode(x)['gripper'] for x in extra['grasp_state_pickup_obj'][()]]
        touch=np.array([x['touching'] for x in gs]);held=np.array([x['held'] for x in gs])
        support=np.array([x['supported_by_receptacle'] for x in ti]);success=g['success'][()].astype(bool)
        assert len(touch)==901 and bool(success[-1])==result['task_success']
        base=extra['robot_base_pose'][()].astype(float);tcp=extra['tcp_pose'][()].astype(float)
        br=Rotation.from_quat(base[:,[4,5,6,3]])
        world=br.apply(tcp[:,:3])+base[:,:3]
        rotation=br*Rotation.from_quat(tcp[:,[4,5,6,3]])
        seen_counts=extra['object_image_points/pickup_obj/wrist_camera/num_points'][()].reshape(-1)
        seen=seen_counts>0
        q=[decode(x) for x in g['obs/agent/qpos'][()]]
        q_arm=np.array([x['arm'] for x in q]);q_gripper=np.array([x['gripper'] for x in q])
        commanded=[decode(x) for x in g['actions/commanded_action'][1:]]
        scene=json.loads(g['obs_scene'][()]);tray=np.array(scene['scene_params']['place_receptacle_start_pose'][:2])
    with np.load(d/'actions.npz',allow_pickle=False) as a:
        grip=a['gripper'].reshape(-1);raw=a['model_output'];arm=a['arm']
        assert np.array_equal(grip,np.where(raw[:,7]<127.5,0,255))
        assert np.array_equal(grip,np.array([c['gripper'][0] for c in commanded]))
        assert np.array_equal(arm,np.array([c['arm'] for c in commanded]))
    close=grip>=127.5;close_obs=np.r_[False,close]
    with h5py.File(d/'telemetry.h5','r') as h:
        target=h['target/control_step_and_world_xyz'][()]
        assert target.shape==(901,4) and np.array_equal(target[:,0],np.arange(901))
        target=target[:,1:]
        classes=json.loads(h['contacts'].attrs['class_names'])
        c=h['contacts/class_entries'][()];ct=h['contacts/control_step'][()].astype(int)
        times=h['contacts/sim_time_s'][()]
        identities=json.loads(h['contacts'].attrs['pair_identities']);pc=h['contacts/pair_counts'][()]
        pads={side:np.zeros(len(ct),dtype=bool) for side in ('left','right')}
        for pair_id,identity in enumerate(identities):
            if identity['class']!='grasp_target':continue
            names=' '.join(identity['names'])
            for side in pads:
                if f'gripper/{side}_pad' in names:
                    pads[side][pc[pc[:,1]==pair_id,0].astype(int)]=True
        bilateral=pads['left']&pads['right']
        bilateral_obs=np.zeros(901,dtype=bool);np.logical_or.at(bilateral_obs,ct,bilateral)
        contact_obs=np.zeros((901,len(classes)),dtype=bool)
        for k in range(len(classes)):np.logical_or.at(contact_obs[:,k],ct,c[:,k]>0)
        simdt=float(np.median(np.diff(times)))
        clutter=h['stability/measures'][()][:,:,2:5]
        slots=json.loads(h['stability'].attrs['slot_names'])
    assert np.isfinite(target).all() and np.isfinite(world).all()
    relative=rotation.inv().apply(target-world)
    delta=world-target;xy=np.linalg.norm(delta[:,:2],axis=1);dist=np.linalg.norm(delta,axis=1)
    lift=target[:,2]-target[0,2];lifted=lift>=.01
    t_close=first(close);t_touch=first(touch);t_held=first(held);t_lift=first(lifted)
    nearest=int(np.argmin(dist));before=min(t_close+1,901) if t_close is not None else 901
    approach_end=t_lift if t_lift is not None else 900
    fc=t_close if t_close is not None else nearest
    collision_before_close={k:bool(contact_obs[:before,classes.index(k)].any()) for k in ('hazard_bar','clutter','other_environment','mounted_fixture')}
    before_hold=t_held+1 if t_held is not None else 901
    stage=('success' if success[-1] else 'supported_without_final_success' if support.any() else
           'lifted_without_placement' if lifted.any() else 'held_without_lift' if held.any() else
           'touched_without_hold' if touch.any() else 'no_target_interaction')
    contacts=result['contact_audit']
    # Descriptive threshold flags are not alternative task-success criteria.
    row_out={'extractor_sha256':VERSION,'result_sha256':record['result_sha256'],'directory':str(d.relative_to(ROOT)),
        'rollout_id':name,'episode_id':record['episode_id'],'seed':record['checkpoint_seed'],'arm':record['arm'],
        'cell':layout['cell'],'family':layout['family'],'side':layout['intrusion_side'],'pose_id':layout['pose_id'],
        'success':bool(success[-1]),'touch':bool(touch.any()),'held':bool(held.any()),'lift':bool(lifted.any()),
        'support':bool(support.any()),'failure_stage':stage,'collision_free_success':result['collision_free_task_success'],
        'initial_target_xyz':target[0],'initial_tcp_world_xyz':world[0],'initial_base_pose':base[0],
        'first_close_command_step':t_close,'first_touch_observation_step':t_touch,'first_held_observation_step':t_held,
        'first_lift_observation_step':t_lift,'first_seen_observation_step':first(seen),
        'first_bilateral_pad_step':first(bilateral_obs),'bilateral_pad_contact':bool(bilateral.any()),
        'bilateral_pad_contact_total_s':float(bilateral.sum()*simdt),'longest_bilateral_pad_contact_s':float(maxrun(bilateral)*simdt),
        'bilateral_before_lift':bool(bilateral_obs[:approach_end+1].any()),
        'touch_to_held_steps':t_held-t_touch if t_held is not None and t_touch is not None else None,
        'first_touch_minus_first_close_steps':t_touch-t_close if t_touch is not None and t_close is not None else None,
        'ever_seen':bool(seen.any()),'seen_fraction':float(seen.mean()),'seen_before_close':bool(seen[:before].any()),
        'seen_at_first_close':bool(seen[fc]),'seen_before_touch':bool(seen[:t_touch+1].any()) if t_touch is not None else None,
        'first_close_xy_error_m':float(xy[fc]) if t_close is not None else None,
        'first_close_world_delta_xyz_m':delta[fc] if t_close is not None else None,
        'first_close_target_in_tcp_m':relative[fc] if t_close is not None else None,
        'first_close_distance_m':float(dist[fc]) if t_close is not None else None,
        'first_close_before_ever_within_3cm_xy':bool(not (xy[:before]<.03).any()) if t_close is not None else None,
        'min_xy_error_before_close_m':float(xy[:before].min()),
        'closest_distance_before_lift_m':float(dist[:approach_end+1].min()),
        'closest_xy_before_lift_m':float(xy[:approach_end+1].min()),
        'closest_tcp_local_target_m':relative[nearest],'max_target_lift_m':float(lift.max()),
        'collision_before_close':collision_before_close,'hazard_contact':contacts['frames_with_contact']['hazard_bar']>0,
        'clutter_contact':contacts['frames_with_contact']['clutter']>0,
        'pre_hold_hazard':bool(contact_obs[:before_hold,classes.index('hazard_bar')].any()),
        'pre_hold_clutter':bool(contact_obs[:before_hold,classes.index('clutter')].any()),
        'close_switches':int(np.count_nonzero(np.diff(close.astype(int)))),
        'tracking_l2_median_before_close_rad':float(np.median(np.linalg.norm(arm[:before]-q_arm[1:before+1],axis=1))) if before<=900 else None,
        'raw_gripper_at_first_close':float(raw[fc,7]) if t_close is not None else None,
        'target_push_xy_before_lift_m':float(np.linalg.norm(target[:approach_end+1,:2]-target[0,:2],axis=1).max()),
        'final100_tcp_extent_m':float(np.linalg.norm(np.ptp(world[-101:],axis=0))),
        'empty_hand_tray_motion':bool(not touch.any() and (np.linalg.norm(world[:,:2]-tray,axis=1)<.1).any()),
        'held_runs':runs(held),'touch_runs':runs(touch),'close_command_runs':runs(close),
        'video_retained':False,'pose_frame_note':'TCP is transformed from robot-base coordinates into world coordinates; dynamic target telemetry is already world-frame.',
        'gripper_note':'Joint-state values are retained without interpreting them as physical jaw widths.'}
    row_out=safe(row_out)
    np.savez_compressed(CACHE/(name+'.npz'),tcp_world=world,target_world=target,target_in_tcp=relative,
        touch=touch,held=held,lift=lift,support=support,success=success,seen=seen,close=close_obs,
        bilateral_pad=bilateral_obs,contact_by_class=contact_obs,contact_classes=np.array(classes),
        model_output=raw,q_arm=q_arm,q_gripper=q_gripper,clutter_positions=clutter,clutter_slots=np.array(slots))
    dest.write_text(json.dumps(row_out,indent=2,allow_nan=False)+'\n');all_rows.append(row_out)
    if index%10==0:print(f'{index+1}/{len(snapshot)} extracted; {record["checkpoint_seed"]} {record["arm"]}',flush=True)
(OUT/f'features_{len(snapshot)}.json').write_text(json.dumps({'snapshot_sha256':snapshot_hash,'extractor_sha256':VERSION,'rows':all_rows},indent=2,allow_nan=False)+'\n')
print('COMPLETE',len(all_rows),flush=True)
