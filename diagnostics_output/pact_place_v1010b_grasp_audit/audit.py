"""Read-only V10.10b audit. No environment creation, mj_step, or policy rollout.

Run with /root/act_retrain_venv/bin/python audit.py {rollouts,data,assemble}.
Only this script's directory is writable. Numbered historical snapshot is fixed.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'OMP_THREAD_LIMIT',
            'BLIS_NUM_THREADS', 'RAYON_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import argparse
import collections
import hashlib
import itertools
import json
from pathlib import Path
import pickle
import re
import sys
import time
import xml.etree.ElementTree as ET
import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R

ROOT = Path('/root/prox_learning_pact_remediation')
OUT = Path(__file__).resolve().parent
W = ROOT / 'diagnostics_output/pact_place_v1010_wrist288_s3_v1'
G = W / 'geometric_failure_20260908'
ROBOT = Path('/root/.cache/molmo-spaces-resources/robots/franka_droid/20260127/model.xml')
ASSET = Path('/root/.cache/molmo-spaces-resources/objects/thor/20251117/Kitchen Objects/Cup/Prefabs/Cup_10/Cup_10_prim.xml')
HASHES = {}

def sha(path):
    path = Path(path)
    h = hashlib.sha256()
    before = path.stat()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4*1024*1024), b''):
            h.update(chunk)
    after = path.stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    HASHES[str(path)] = h.hexdigest()
    return h.hexdigest()

def read(path):
    sha(path)
    return json.loads(Path(path).read_text())

def lines(path):
    sha(path)
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l]

def decode(x):
    return json.loads(bytes(x).split(b'\0')[0])

def safe(x):
    if isinstance(x, dict): return {str(k): safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [safe(v) for v in x]
    if isinstance(x, np.ndarray): return safe(x.tolist())
    if isinstance(x, np.generic): return safe(x.item())
    if isinstance(x, float) and not np.isfinite(x): return None
    return x

def write(name, doc):
    assert Path(name).name == name
    (OUT/name).write_text(json.dumps(safe(doc), indent=2, sort_keys=True, allow_nan=False)+'\n')

def first(mask):
    ix = np.flatnonzero(mask)
    return int(ix[0]) if len(ix) else None

def runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))

def maxrun(mask):
    return max((b-a for a,b in runs(mask)), default=0)

def med(values):
    values = [v for v in values if v is not None]
    return float(np.median(values)) if values else None

def boxes():
    sha(ASSET)
    body = ET.parse(ASSET).find('.//worldbody/body/body')
    offset = np.fromstring(body.get('pos'), sep=' ')
    signs = np.array(list(itertools.product((-1,1), repeat=3)))
    result = []
    for geom in body.findall('geom'):
        if geom.get('type') != 'box': continue
        pos = np.fromstring(geom.get('pos'), sep=' ')
        q = np.fromstring(geom.get('quat'), sep=' ')
        size = np.fromstring(geom.get('size'), sep=' ')
        rot = R.from_quat(q[[1,2,3,0]])
        result.append(dict(name=geom.get('name'), pos=pos, quat=q, size=size,
                           center=pos+offset, vertices=rot.apply(signs*size)+pos+offset,
                           normal=rot.apply([0,0,1])))
    assert len(result) == 5
    return result

def kinematics():
    import mujoco
    sha(ROBOT)
    model = mujoco.MjModel.from_xml_path(str(ROBOT))
    data = mujoco.MjData(model)
    site = model.site('gripper/grasp_site').id
    def fk(q):
        data.qpos[:7] = q
        mujoco.mj_kinematics(model, data)
        return data.site_xpos[site].copy()
    return fk, model

def raw_contacts(path):
    with h5py.File(path, 'r') as h:
        classes = json.loads(h['contacts'].attrs['class_names'])
        identities = json.loads(h['contacts'].attrs['pair_identities'])
        counts = h['contacts/class_entries'][()]
        pairs = h['contacts/pair_counts'][()]
        steps = h['contacts/control_step'][()]
        times = h['contacts/sim_time_s'][()]
        target = h['target/control_step_and_world_xyz'][()][:,1:]
    assert np.all(np.diff(times)>0)
    masks = {k: np.zeros(len(steps), bool) for k in
             ('robot_cup','robot_clutter','cup_clutter','robot_hazard','robot_other')}
    pads = np.zeros((len(steps),2,5), bool)
    rebuilt = np.zeros_like(counts)
    for j, identity in enumerate(identities):
        entries = pairs[pairs[:,1] == j]
        ix = entries[:,0].astype(int)
        np.add.at(rebuilt[:,classes.index(identity['class'])], ix, entries[:,2])
        names = ' '.join(identity['names'])
        robot = 'robot_0/' in names
        cup = 'cavity_obj_0/Cup_10' in names
        if robot and cup: masks['robot_cup'][ix] = True
        if identity['class'] == 'clutter':
            if robot: masks['robot_clutter'][ix] = True
            if cup: masks['cup_clutter'][ix] = True
        if robot and identity['class']=='hazard_bar': masks['robot_hazard'][ix] = True
        if robot and identity['class'] in ('other_environment','mounted_fixture'):
            masks['robot_other'][ix] = True
        wall = re.search(r'Cup_10_cup_10_PrimitiveCollider_(\d)', names)
        if wall and robot:
            for side, label in enumerate(('left','right')):
                if f'gripper/{label}_pad' in names: pads[ix,side,int(wall[1])] = True
    assert np.array_equal(counts, rebuilt)
    same = (pads[:,0,[0,1,3,4]] & pads[:,1,[0,1,3,4]]).any(1)
    bilateral = pads[:,0].any(1) & pads[:,1].any(1)
    return classes, counts, steps, times, target, masks, same, bilateral

def rollout_audit():
    start = time.time()
    source = read(G/'analysis_300.json')
    old = {r['rollout_id']: r for r in source['rows']}
    assert len(old)==300
    bx = boxes(); vertices = np.concatenate([b['vertices'] for b in bx])
    fk, model = kinematics()
    records=[]; audits=[]; metadata=[]
    for seed in (3103,3104,3105):
        ledger=W/f'evaluation/final_{seed}_full50_h100_ledger.jsonl'
        rr=lines(ledger)
        jobs=read(W/f'evaluation/final_{seed}_full50_h100_schedule.json')
        done=read(W/f'evaluation/final_{seed}_full50_h100_complete.json')
        assert len(rr)==len(jobs)==done['count']==100 and done['ledger_sha256']==sha(ledger)
        assert {j['schedule']['rollout_id'] for j in jobs}=={r['rollout_id'] for r in rr}
        records.extend(rr)
    for i,rec in enumerate(records):
        r=old[rec['rollout_id']]; d=Path(rec['directory'])
        assert rec['valid_completion'] and rec['returncode']==0
        result=read(d/'result.json'); receipt=read(d/'exit_receipt.json'); job=read(d/'job.json')
        assert sha(d/'result.json')==rec['result_sha256']
        assert sha(d/'exit_receipt.json')==rec['exit_receipt_sha256'] and receipt['returncode']==0
        assert receipt['job_sha256']==hashlib.sha256(json.dumps(job,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        assert result['checkpoint_sha256']==rec['checkpoint_sha256']
        assert result['stats_sha256']==rec['stats_sha256']
        for name,expected in [('trajectory.h5',result['trajectory_retention']['full_h5_sha256']),
                              ('telemetry.h5',result['policy_info']['raw_telemetry_sha256']),
                              ('initial_observation.h5',result['policy_info']['initial_observation_sha256'])]:
            assert sha(d/name)==expected
        with h5py.File(d/'trajectory.h5','r') as h:
            g=h['traj_0']; e=g['obs/extra']
            base=e['robot_base_pose'][()].astype(float); tcp=e['tcp_pose'][()].astype(float)
            br=R.from_quat(base[:,[4,5,6,3]])
            world=br.apply(tcp[:,:3])+base[:,:3]
            q=np.array([decode(x)['arm'] for x in g['obs/agent/qpos'][()]])
            commands=[decode(x) for x in g['actions/commanded_action'][1:]]
            gs=[decode(x)['gripper'] for x in e['grasp_state_pickup_obj'][()]]
            touch=np.array([x['touching'] for x in gs]); held=np.array([x['held'] for x in gs])
            support=np.array([decode(x)['supported_by_receptacle'] for x in e['task_info'][()]])
            success=bool(g['success'][-1]); assert len(g['success'])==901
        sha(d/'actions.npz')
        with np.load(d/'actions.npz') as z:
            raw=z['model_output']; arm=z['arm']; grip=z['gripper'].reshape(-1)
        assert raw.shape==(900,8) and np.isfinite(raw).all()
        assert np.array_equal(arm,raw[:,:7])
        assert np.array_equal(arm,np.array([x['arm'] for x in commands]))
        assert np.array_equal(grip,np.where(raw[:,7]<127.5,0,255))
        assert np.array_equal(grip,np.array([x['gripper'][0] for x in commands]))
        classes,counts,steps,times,target,masks,same,bilateral=raw_contacts(d/'telemetry.h5')
        lifted=target[:,2]-target[0,2]>=.01
        stage=('success' if success else 'supported_without_final_success' if support.any() else
               'lifted_without_placement' if lifted.any() else 'held_without_lift' if held.any() else
               'touched_without_hold' if touch.any() else 'no_target_interaction')
        cfts=success and not counts[:,[classes.index(k) for k in
            ('hazard_bar','clutter','other_environment','mounted_fixture')]].any()
        assert (success,stage,cfts)==(r['success'],r['failure_stage'],r['collision_free_success'])
        assert (success,cfts)==(result['task_success'],result['collision_free_task_success'])
        for j,k in enumerate(classes):
            assert int((counts[:,j]>0).sum())==result['contact_audit']['frames_with_contact'][k]
        hc=counts[:,classes.index('hazard_bar')]>0; cc=counts[:,classes.index('clutter')]>0
        actual_fk=np.array([fk(v) for v in q])
        offset=tcp[0,:3]-actual_fk[0]
        assert np.allclose(offset,[0,0,.35],atol=1e-6)
        error=np.linalg.norm(br.apply(actual_fk+offset)+base[:,:3]-world,axis=1)
        commanded=br[:-1].apply(np.array([fk(v) for v in arm])+offset)+base[:-1,:3]
        assert np.max(error)<.01
        sha(G/'commands'/(r['rollout_id']+'.npz'))
        with np.load(G/'commands'/(r['rollout_id']+'.npz')) as z:
            assert np.allclose(commanded,z['command_tcp_world'],atol=1e-10,rtol=0)
        with h5py.File(d/'initial_observation.h5','r') as h:
            for b,j in zip(bx,r['target_primitive_geom_ids']):
                assert np.allclose(h['model/geom_size'][j],b['size'],atol=1e-10,rtol=0)
                assert np.allclose(h['model/geom_pos'][j],b['pos'],atol=1e-9,rtol=0)
                assert abs(np.dot(h['model/geom_quat'][j],b['quat'])/np.linalg.norm(b['quat']))>1-1e-10
        c=first(grip>=127.5); dt=float(np.median(np.diff(times)))
        assert c==r['first_close_command_step']
        def event(mask):
            ix=first(mask)
            return None if ix is None else dict(physics_index=ix, control_step=int(steps[ix]),time_s=float(times[ix]))
        row=dict(r,raw_touched_never_held=bool(touch.any() and not held.any()),
                 physics_samples=len(times),hazard_clutter_union_frames=int((hc|cc).sum()),
                 hazard_clutter_overlap_frames=int((hc&cc).sum()),
                 contact_frames={k:int((counts[:,j]>0).sum()) for j,k in enumerate(classes)},
                 elapsed_s=rec['elapsed_s'],training_seed=rec['checkpoint_seed'],task_seed=result['seed'],
                 raw_events={k:event(mask) for k,mask in masks.items()},
                 same_wall_recomputed_longest_s=float(maxrun(same)*dt),
                 bilateral_recomputed_longest_s=float(maxrun(bilateral)*dt),
                 fk_recomputed_max_error_m=float(error.max()))
        assert abs(row['same_wall_recomputed_longest_s']-r['same_wall_longest_s'])<1e-8
        if c is not None:
            rim=target[:,2]+r['target_collision_top_above_origin_m']
            height=world[:,2]-rim; xy=np.linalg.norm(world[:,:2]-target[:,:2],axis=1)
            eligible=(np.arange(901)<=c+60)&(xy<.06)&(~lifted)
            minimum=float(height[eligible].min()) if eligible.any() else None
            assert minimum==r['minimum_rim_relative_tcp_early_grasp_m']
            row['close_local_phase_tracking_m']={label:med(np.linalg.norm(world[1:]-commanded,axis=1)[ix]) for label,ix in
                [('approach',slice(max(0,c-30),c)),('closure',slice(c,min(c+11,900))),('postclose',slice(min(c+11,900),min(c+61,900))) ]}
            command_near=(np.linalg.norm(commanded[:,:2]-target[:-1,:2],axis=1)<.06)&(~lifted[:-1])
            row['commanded_below_rim_first_step']=first(command_near & (np.arange(900)<=c+60) & (commanded[:,2]<rim[:-1]))
            row['achieved_below_rim_first_step']=first(eligible & (height<0))
            row['tracking_gap_over5mm_first_preclose_step']=first((np.arange(900)>=max(c-30,0)) & (np.arange(900)<=c) & ((world[:-1,2]-commanded[:,2])>.005))
            # Offline observable gate: next close requested, actual TCP above requested TCP by >5 mm.
            row['observable_close_gap_gate']=bool((world[c,2]-commanded[c,2])>.005)
            row['close_gap_z_m']=float(world[c,2]-commanded[c,2])
            row['close_hand_in_initial_cup_axes_m']=R.from_quat(np.array(r['initial_target_quat_wxyz'])[[1,2,3,0]]).inv().apply(world[c]-target[c])
            # Rigid rotation sensitivity of ALL box corners: max edge rise under arbitrary horizontal tilt axis.
            verts=np.array(r['target_collision_vertices_world'])-target[0]
            row['rim_tilt_sensitivity_m']={}
            for degrees in (2,5,10):
                tops=[]
                for angle in np.linspace(0,2*np.pi,72,endpoint=False):
                    rot=R.from_rotvec(np.deg2rad(degrees)*np.array([np.cos(angle),np.sin(angle),0]))
                    tops.append(float(rot.apply(verts)[:,2].max()))
                shifts=np.array(tops)-r['target_collision_top_above_origin_m']
                row['rim_tilt_sensitivity_m'][degrees]=dict(min=float(shifts.min()),max=float(shifts.max()),
                    shallow_for_all_axes=None if minimum is None else bool(minimum>=shifts.max()),
                    shallow_for_any_axis=None if minimum is None else bool(minimum>=shifts.min()))
        audits.append(row)
        if i%25==0: print('raw rollouts',i+1,'/300',round(time.time()-start,1),'s',flush=True)
    # Recheck original ACT/PACT initial state with its exact historical contract, no mutation.
    lookup={(r['episode_id'],r['arm']):r for r in audits}
    for p in [r for r in audits if r['arm']=='PACT']:
        a=lookup[p['episode_id'],'ACT']; violations=[]
        with h5py.File(ROOT/a['directory']/'initial_observation.h5') as ha,h5py.File(ROOT/p['directory']/'initial_observation.h5') as hb:
            names=[[],[]]
            for j,h in enumerate((ha,hb)):
                h.visititems(lambda n,o:names[j].append(n) if isinstance(o,h5py.Dataset) else None)
            assert names[0]==names[1]
            for key in names[0]:
                av,bv=ha[key][()],hb[key][()]
                assert ha[key].shape==hb[key].shape and ha[key].dtype==hb[key].dtype
                if key=='observation/wrist_camera':
                    delta=np.abs(av.astype(int)-bv.astype(int))
                    rgb=dict(max_abs_delta=int(delta.max()),changed_fraction=float(np.count_nonzero(delta)/delta.size))
                    assert rgb['max_abs_delta']<=2 and rgb['changed_fraction']<=.001
                else: assert np.asarray(av).tobytes()==np.asarray(bv).tobytes(),key
        metadata.append(dict(episode_id=p['episode_id'],seed=p['seed'],rgb=rgb,passed=True))
    write('rollout_audit.json',dict(rows=audits,pair_audits=metadata,source_hashes=HASHES,elapsed_s=time.time()-start))
    print('rollouts COMPLETE',flush=True)

def data_audit():
    start=time.time(); bx=boxes(); fk,model=kinematics()
    selected=lines(W/'amendments/wrist280/selected_ledger.jsonl')
    accepted={r['attempt_id']:r for r in selected if r['accepted']}
    conversion=lines(W/'conversion_ledger.jsonl'); split=read(W/'split_manifest.json')
    splitmap={r['episode_id']:r for r in split['episodes']}
    assert len(accepted)==len(conversion)==len(splitmap)==280
    assert set(accepted)==set(splitmap)=={r['episode_id'] for r in conversion}
    assert collections.Counter(r['split'] for r in splitmap.values())=={'train':240,'validation':40}
    allrows=[]; train_q=[];train_a=[]
    for i,conv in enumerate(conversion):
        ident=conv['episode_id']; raw=accepted[ident]; s=splitmap[ident]
        source=ROOT/raw['trajectory_h5']; dest=W/'converted'/conv['act_file']
        assert sha(source)==raw['trajectory_h5_sha256']==conv['source_h5_sha256']
        assert sha(dest)==conv['act_file_sha256']
        assert sha(Path(raw['directory'])/'exit_receipt.json')==raw['exit_receipt_sha256']
        with h5py.File(dest,'r') as h:
            assert bool(h.attrs['sim']) and h.attrs['supervision']=='observation[t] -> commanded_action[t+1]'
            a=h['action'][()]; q=h['observations/qpos'][()]
            assert h['observations/proximity_embeddings'].shape==(len(a),40,32)
            assert h['observations/images/wrist_camera'].shape==(len(a),240,320,3)
        with h5py.File(source,'r') as h:
            g=h['traj_0'];e=g['obs/extra']; n=len(g['success']); t=len(a)
            commands=[decode(x) for x in g['actions/commanded_action'][()]]
            assert commands[0]=={}
            aa=np.array([x['arm']+x['gripper'] for x in commands[1:t+1]],dtype=np.float32)
            qq=np.array([decode(x)['arm']+decode(x)['gripper'] for x in g['obs/agent/qpos'][:t]],dtype=np.float32)
            assert np.array_equal(a,aa) and np.array_equal(q,qq)
            assert all(bool(g['terminated'][j] or g['truncated'][j]) and
                       (commands[j].get('done',False) or not decode(g['actions/joint_pos'][j])) for j in range(t+1,n))
            assert bool(g['success'][-1])
            tcp=e['tcp_pose'][()].astype(float); base=e['robot_base_pose'][()].astype(float)
            br=R.from_quat(base[:,[4,5,6,3]]);world=br.apply(tcp[:,:3])+base[:,:3]
            tr=br*R.from_quat(tcp[:,[4,5,6,3]])
            cup=e['obj_start'][0].astype(float); grasp=e['grasp_pose'][()].astype(float)
            phase=e['policy_phase'][:t]; retries=e['policy_num_retries'][:t]
            held=np.array([decode(x)['gripper']['held'] for x in e['grasp_state_pickup_obj'][()]])
        c=first(a[:,7]>=127.5); assert c is not None
        cuprot=R.from_quat(cup[[4,5,6,3]])
        local=cuprot.inv().apply(world[c]-cup[:3])
        rim=cup[2]+cuprot.apply(np.concatenate([b['vertices'] for b in bx]))[:,2].max()
        jaw=tr[c].apply([0,1,0]);jaw=jaw/np.linalg.norm(jaw)
        wall_ids=[0,1,3,4]
        alignment=[abs(float(jaw@cuprot.apply(bx[j]['normal']))) for j in wall_ids]
        distances=[np.linalg.norm(local-bx[j]['center']) for j in wall_ids]
        nearest=wall_ids[int(np.argmin(distances))]
        offset=tcp[0,:3]-fk(q[0,:7]); assert np.allclose(offset,[0,0,.35],atol=1e-6)
        command=br[:t].apply(np.array([fk(x) for x in a[:,:7]])+offset)+base[:t,:3]
        tracking=np.linalg.norm(world[1:t+1]-command,axis=1)
        ix=np.arange(t); acquisition=(ix>=max(0,c-30)) & (ix<=min(t-1,c+30))
        close_edges=np.flatnonzero(np.diff(np.r_[False,a[:,7]>=127.5].astype(int))==1)
        row=dict(episode_id=ident,act_episode_index=conv['act_episode_index'],split=s['split'],cell=s['cell'],
                 raw_path=str(source),converted_path=str(dest),T=t,raw_T=n,alignment_exact=True,
                 first_close_command_step=c,close_edges=close_edges,max_policy_retries=int(retries.max()),
                 first_held_observation_step=first(held),phase_frames=dict(collections.Counter(map(int,phase))),
                 acquisition_frames=int(acquisition.sum()),acquisition_probability_uniform=float(acquisition.mean()),
                 full_chunk_probability=float(max(0,t-99)/t),expected_valid_chunk_fraction=float(np.minimum(100,t-np.arange(t)).mean()/100),
                 tcp_at_close_above_initial_rim_m=float(world[c,2]-rim),
                 hand_at_close_in_initial_cup_axes_m=local,nearest_initial_wall=nearest,
                 closing_axis_to_initial_wall_normal_deg=float(np.degrees(np.arccos(np.clip(max(alignment),-1,1)))),
                 command_at_close_above_initial_rim_m=float(command[c,2]-rim),
                 command_vs_achieved_close_distance_m=float(np.linalg.norm(command[c]-world[c])),
                 close_target_grasp_position_error_m=float(np.linalg.norm(world[c]-grasp[c,:3])),
                 phase_tracking_m={str(k):dict(n=int((phase==k).sum()),median_m=med(tracking[phase==k])) for k in sorted(set(phase))},
                 action_reopens_after_failed_hold_before_second_close=bool(len(close_edges)>1 and not held[:min(close_edges[1]+1,len(held))].any()) if len(close_edges)>1 else False)
        allrows.append(row)
        if s['split']=='train':train_q.append(q);train_a.append(a)
        if i%25==0: print('demos',i+1,'/280',round(time.time()-start,1),'s',flush=True)
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    q=torch.from_numpy(np.concatenate(train_q));a=torch.from_numpy(np.concatenate(train_a))
    stats=dict(action_mean=a.mean(0).numpy(),action_std=a.std(0).clamp(min=.01).numpy(),
               qpos_mean=q.mean(0).numpy(),qpos_std=q.std(0).clamp(min=.01).numpy(),example_qpos=q[0].numpy())
    sha(W/'dataset_stats.pkl'); frozen=pickle.loads((W/'dataset_stats.pkl').read_bytes())
    assert all(np.array_equal(stats[k],frozen[k]) for k in stats)
    checkpoints=[]
    for directory in sorted((W/'checkpoints').iterdir()):
        manifest=read(directory/'run_manifest.json');progress=read(directory/'progress.json')
        process=read(directory/'process_30000_60000.json')
        logs=lines(directory/'epoch_log.jsonl')
        assert len(logs)==2000 and logs[-1]['global_step']==progress['updates']==process['completed_updates']==60000
        assert sha(directory/'dataset_stats.pkl')==sha(W/'dataset_stats.pkl')
        ckpt=sha(directory/'policy_update_60000.ckpt'); best=sha(directory/'policy_best.ckpt')
        # Legacy main mentions policy_best, but Wrist288 main rewrites it to
        # policy_update_60000 before executing. Verify the actual ledger binding.
        seed=int(directory.name.rsplit('seed',1)[1]); arm=directory.name.split('_')[0].upper()
        bound=lines(W/f'evaluation/final_{seed}_full50_h100_ledger.jsonl')
        assert {r['checkpoint_sha256'] for r in bound if r['arm']==arm}=={ckpt}
        state=torch.load(directory/'policy_update_60000.ckpt',map_location='cpu',weights_only=True)
        assert all(torch.isfinite(v).all() for v in state.values())
        projection=state.get('model.input_proj_proximity.weight')
        assert (projection is not None)==directory.name.startswith('pact_')
        if projection is not None: assert tuple(projection.shape)==(512,32)
        checkpoints.append(dict(directory=str(directory),checkpoint_sha256=ckpt,unused_policy_best_sha256=best,updates=60000,
                                parameter_count=sum(v.numel() for v in state.values()),
                                finite=True,stats_sha256=sha(directory/'dataset_stats.pkl'),
                                process_elapsed_s=process['elapsed_s'],policy_config=manifest['policy_config'],
                                train_indices=manifest.get('train_act_indices'),val_indices=manifest.get('val_act_indices')))
        del state
    write('data_audit.json',dict(rows=allrows,selected_attempts=len(selected),accepted=280,rejected=len(selected)-280,
          checkpoints=checkpoints,stats_exact=True,statistics=frozen,source_hashes=HASHES,elapsed_s=time.time()-start))
    print('data COMPLETE',flush=True)

def assemble():
    ra=read(OUT/'rollout_audit.json'); da=read(OUT/'data_audit.json')
    validation=read(OUT/'validation_errors.json') if (OUT/'validation_errors.json').exists() else None
    rows=ra['rows']; groups={}
    for seed in (3103,3104,3105):
        for arm in ('ACT','PACT'):
            rr=[r for r in rows if r['seed']==seed and r['arm']==arm];pickup=[r for r in rr if r['failure_stage']=='touched_without_hold']
            groups[f'{arm}_{seed}']=dict(n=len(rr),success=sum(r['success'] for r in rr),cfts=sum(r['collision_free_success'] for r in rr),
                stages=dict(collections.Counter(r['failure_stage'] for r in rr)),raw_touched_never_held=sum(r['raw_touched_never_held'] for r in rr),
                frame_denominator=sum(r['physics_samples'] for r in rr),union_frames=sum(r['hazard_clutter_union_frames'] for r in rr),
                overlap_frames=sum(r['hazard_clutter_overlap_frames'] for r in rr),
                avoidance_percent=100*(1-sum(r['hazard_clutter_union_frames'] for r in rr)/sum(r['physics_samples'] for r in rr)),
                blocked=sum(r.get('rim_blocked_signature',False) for r in rr),
                shallow_pickup=sum(r.get('minimum_rim_relative_tcp_early_grasp_m') is not None and r['minimum_rim_relative_tcp_early_grasp_m']>=0 for r in pickup),
                pickup_depth_eligible=sum(r.get('minimum_rim_relative_tcp_early_grasp_m') is not None for r in pickup),
                pickup_depth_missing_ids=[r['episode_id'] for r in pickup if r.get('minimum_rim_relative_tcp_early_grasp_m') is None],
                pickup_command_dz2_m=med([r.get('command_z_change_2_steps_after_close_m') for r in pickup]),
                pickup_actual_dz2_m=med([r.get('actual_z_change_2_steps_after_close_m') for r in pickup]),
                successful_samewall_1s=sum(r['success'] and r['same_wall_longest_s']>=1 for r in rr),
                samewall_1s_failure_ids=[r['episode_id'] for r in rr if not r['success'] and r['same_wall_longest_s']>=1],
                centered_success_ids=[r['episode_id'] for r in rr if r['success'] and r.get('command_at_close_xy_error_m',1)<.02],
                shallow_success_ids=[r['episode_id'] for r in rr if r['success'] and r.get('minimum_rim_relative_tcp_early_grasp_m',-1) is not None and r['minimum_rim_relative_tcp_early_grasp_m']>=0],
                wide_rim_success_ids=[r['episode_id'] for r in rr if r['success'] and r.get('initial_rim_span_exceeds_nominal_open_gap')],
                observable_gate=dict(triggered=sum(r.get('observable_close_gap_gate',False) for r in rr),
                    successful_trigger_ids=[r['episode_id'] for r in rr if r['success'] and r.get('observable_close_gap_gate')],
                    pickup_trigger_ids=[r['episode_id'] for r in pickup if r.get('observable_close_gap_gate')]),
                tilt_shallow_pickup={str(deg):{name:sum(r.get('rim_tilt_sensitivity_m',{}).get(str(deg),{}).get(name,False) or False for r in pickup) for name in ('shallow_for_all_axes','shallow_for_any_axis')} for deg in (2,5,10)})
    demo_groups={}
    for split in ('train','validation'):
        rr=[r for r in da['rows'] if r['split']==split]
        demo_groups[split]=dict(n=len(rr),cells=dict(collections.Counter(r['cell'] for r in rr)),
            wall_modes=dict(collections.Counter(r['nearest_initial_wall'] for r in rr)),
            retries=sum(r['max_policy_retries']>0 for r in rr),multiple_close=sum(len(r['close_edges'])>1 for r in rr),
            failed_approach_recovery=sum(r['action_reopens_after_failed_hold_before_second_close'] for r in rr),
            mean_acquisition_probability=float(np.mean([r['acquisition_probability_uniform'] for r in rr])),
            median_T=med([r['T'] for r in rr]),
            mean_valid_chunk_fraction=float(np.mean([r['expected_valid_chunk_fraction'] for r in rr])),
            median_close_height_m=med([r['tcp_at_close_above_initial_rim_m'] for r in rr]),
            median_wall_normal_angle_deg=med([r['closing_axis_to_initial_wall_normal_deg'] for r in rr]))
    # Frozen diagnostic scene selection: one lowest digest identity per cell, independent of outcomes.
    manifests=[read(W/f'manifests/final_{s}.json') for s in (3103,3104,3105)]
    candidates=[dict(row,source_seed=s) for s,m in zip((3103,3104,3105),manifests) for row in m['rows']]
    rule='sha256(UTF8("pact_place_v1010b_grasp_v1/cross24/" + episode_id)); lexical minimum within cell'
    selected=[]
    for cell in sorted({r['cell'] for r in candidates}):
        row=min((r for r in candidates if r['cell']==cell),key=lambda r:hashlib.sha256(('pact_place_v1010b_grasp_v1/cross24/'+r['episode_id']).encode()).hexdigest())
        selected.append(row)
    write('cross24_scenes.json',dict(schema='v1010b_diagnostic_scene_selection_v1',rule=rule,rows=selected,exposed_diagnostic_only=True))
    files=[ROOT/'docs/PACT_PLACE_V1010B_GRASP_AUDIT_PLAN.md',ROOT/'docs/PACT_PLACE_V1010_WRIST288_THREE_SEED_PLAN.md',ROOT/'docs/PACT_PLACE_V1010_WRIST280_AMENDMENT.md']
    files+=list(G.glob('*_300.*'))+[G/f for f in ('extract.py','augment.py','contact_modes.py','command_geometry.py','summarize.py','final_verification.json')]
    files+=list((W/'amendments').glob('seed*full50*/*.md'))+list((W/'amendments').glob('seed*full50*/parent_closure.json'))
    files+=[W/'amendments/seed3103_full50_20260908/three_seed_full150.json',W/'amendments/wrist280/effective_config.json',W/'development_gate.json']
    files+=list((ROOT/'scripts').glob('pact_wrist*.py'))
    files+=[ROOT/'submodules/act'/f for f in ('eval_pact_place_v109_row.py','eval_pact_place_row.py','eval_pact_frontend_screen_row.py','eval_pact_collision_row.py','utils.py','fixed_split_data.py','policy.py','imitate_episodes.py','detr/models/detr_vae.py')]
    files+=[ROOT/'submodules/molmospaces/molmo_spaces'/f for f in ('env/sensors.py','tasks/pact_place_contact_audit.py','robots/robot_views/franka_droid_view.py')]
    files+=list(OUT.glob('*.py'))
    for p in files: sha(p)
    hashes={**ra['source_hashes'],**da['source_hashes'],**(validation['source_hashes'] if validation else {}),**HASHES}
    elapsed=np.array([r['elapsed_s'] for r in rows]);latest=[r['elapsed_s'] for r in rows if r['seed']==3103]
    write('evidence.json',dict(schema='v1010b_grasp_audit_v1',status='OFFLINE_AUDIT_COMPLETE',
        baseline=dict(demos=280,train=240,validation=40,pairs=150,rollouts=300,updates=60000,history=100,chunk=100),
        groups=groups,demo_groups=demo_groups,source_hashes=hashes,rollouts=rows,demos=da['rows'],checkpoints=da['checkpoints'],
        validation_phase_errors=validation,
        pair_audits=ra['pair_audits'],cross24_rule=rule,cross24_scene_ids=[r['episode_id'] for r in selected],
        cost=dict(rollout_wall_s=dict(median=float(np.median(elapsed)),p90=float(np.quantile(elapsed,.9)),latest_seed3103_median=med(latest)),
                  offline_rollout_audit_s=ra['elapsed_s'],offline_data_audit_s=da['elapsed_s']),
        definitions=dict(early_grasp='obs 0..min(close+60,900), TCP/cup XY <0.06 m, cup lift <0.01 m; null when no eligible sample',
          contact_duration='max consecutive physics samples * median physics dt; boundary-inclusive legacy descriptor',
          closing_axis='TCP local +Y; driver states are hinge radians, not jaw gap',
          frame_avoidance='100*(1-count(hazard OR clutter)/actual recorded physics samples)',
          command_alignment='command k from actions.npz equals raw commanded_action[k+1], follows observation k',
          demo_wall_mode='nearest initial wall center to achieved TCP at closure in cup coordinates; geometric proxy, not measured pad contact',
          tilt='hypothetical rigid cup tilt 2/5/10 degrees about 72 horizontal axes; no measured dynamic rotation',
          gate='first close command and measured TCP world Z - FK requested TCP world Z > 0.005 m; diagnostic trigger, not grasp proof'),
        new_rollouts=0,training_updates=0))
    print(json.dumps(safe(dict(groups=groups,demo_groups=demo_groups)),indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['rollouts','data','assemble'])
    {'rollouts':rollout_audit,'data':data_audit,'assemble':assemble}[parser.parse_args().stage]()
