"""Independent raw endpoint/contact/geometry recomputation and fixed stage gates."""
from __future__ import annotations
import collections
import importlib.util
import itertools
import json
import re
from functools import lru_cache
from pathlib import Path
import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R
from pact_v1010b_contract import *
from pact_wrist288_metrics import metrics as old_metrics


def first(mask):
    ix=np.flatnonzero(mask);return int(ix[0]) if len(ix) else None


def runs(mask):
    edges=np.diff(np.r_[False,mask,False].astype(int))
    return list(zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)))


def longest(mask,dt):return float(max((b-a for a,b in runs(mask)),default=0)*dt)


def duration(mask,times):
    # Left-closed/right-open intervals; the terminal sample has zero duration.
    return float(np.sum(np.diff(times)*np.asarray(mask[:-1],bool)))


def decode(x):return json.loads(bytes(x).split(b'\0')[0])


@lru_cache(None)
def audit_helpers():
    spec=importlib.util.spec_from_file_location('v1010b_readonly_audit',A/'audit.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@lru_cache(None)
def fk_model():return audit_helpers().kinematics()


def frame_world(local,base,mount=None):
    rot=R.from_quat(np.asarray(base)[...,[4,5,6,3]])
    point=np.asarray(local)+(np.asarray(mount) if mount is not None else 0)
    return rot.apply(point)+np.asarray(base)[...,:3]


def contact_metrics(path):
    with h5py.File(path) as h:
        classes=json.loads(h['contacts'].attrs['class_names']);counts=h['contacts/class_entries'][()]
        pairs=h['contacts/pair_counts'][()];identities=json.loads(h['contacts'].attrs['pair_identities'])
        times=h['contacts/sim_time_s'][()];steps=h['contacts/control_step'][()]
    assert np.all(np.diff(times)>0) and len(times)>29000
    pads=np.zeros((len(times),2,5),bool)
    masks={k:np.zeros(len(times),bool) for k in ('robot_cup','robot_clutter','cup_clutter','robot_hazard','robot_other')}
    for j,identity in enumerate(identities):
        ix=pairs[pairs[:,1]==j,0].astype(int);names=' '.join(identity['names'])
        robot='robot_0/' in names;cup='cavity_obj_0/Cup_10' in names
        if robot and cup:masks['robot_cup'][ix]=True
        if identity['class']=='clutter':
            if robot:masks['robot_clutter'][ix]=True
            if cup:masks['cup_clutter'][ix]=True
        if robot and identity['class']=='hazard_bar':masks['robot_hazard'][ix]=True
        if robot and identity['class'] in ('other_environment','mounted_fixture'):masks['robot_other'][ix]=True
        wall=re.search(r'Cup_10_cup_10_PrimitiveCollider_(\d)',names)
        if wall and robot:
            for side,label in enumerate(('left','right')):
                if f'gripper/{label}_pad' in names:pads[ix,side,int(wall[1])]=True
    bilateral=pads[:,0].any(1)&pads[:,1].any(1)
    fixed=pads[:,0]&pads[:,1];same=fixed[:,[0,1,3,4]].any(1)
    dt=float(np.median(np.diff(times)));hazard=counts[:,classes.index('hazard_bar')]>0;clutter=counts[:,classes.index('clutter')]>0
    union=hazard|clutter
    out={'physics_samples':len(times),'median_physics_dt_s':dt,'union_frames':int(union.sum()),
        'overlap_frames':int((hazard&clutter).sum()),'avoidance_percent':100*(1-union.mean()),
        'bilateral_longest_s':longest(bilateral,dt),'same_wall_longest_s':longest(same,dt),
        'fixed_wall_longest_s':{str(i):longest(fixed[:,i],dt) for i in (0,1,3,4)},
        'continuous_bilateral_1s':longest(bilateral,dt)>=1.,
        'bilateral_runs_1s':[{'start_control_step':int(steps[a]),'end_control_step':int(steps[b-1]),'duration_s':float((b-a)*dt)} for a,b in runs(bilateral) if (b-a)*dt>=1.],
        'contact_duration_descriptor_s':{c:float((counts[:,i]>0).sum()*dt) for i,c in enumerate(classes)},
        'contact_duration_integrated_s':{c:duration(counts[:,i]>0,times) for i,c in enumerate(classes)},
        'union_duration_descriptor_s':float(union.sum()*dt),'union_duration_integrated_s':duration(union,times),
        'integration_convention':'left-closed/right-open intervals; terminal sample contributes zero',
        'raw_contact_kinds':{k:{'frames':int(v.sum()),'duration_s':duration(v,times),'first_control_step':None if first(v) is None else int(steps[first(v)])} for k,v in masks.items()}}
    return out,steps,times,masks,bilateral


def pad_collider_gap(directory,poses,meta,axis):
    signs=np.array(list(itertools.product((-1,1),repeat=3)));intervals=[]
    with h5py.File(Path(directory)/'initial_observation.h5') as h:
        for pi,body_id in enumerate(meta['body_ids'][1:],1):
            body_rotation=R.from_quat(poses[pi,[4,5,6,3]]);vertices=[]
            for gi,gbody in enumerate(meta['geom_body_ids']):
                if gbody!=body_id or not (meta['geom_names'][gi] or '').endswith(('pad1','pad2')):continue
                geom_rotation=R.from_quat(h['model/geom_quat'][gi][[1,2,3,0]])
                local=geom_rotation.apply(signs*h['model/geom_size'][gi])+h['model/geom_pos'][gi]
                vertices.extend(body_rotation.apply(local)+poses[pi,:3])
            assert len(vertices)==16, 'exactly two frozen box colliders per pad'
            projections=np.asarray(vertices)@axis;intervals.append([float(projections.min()),float(projections.max())])
    return {'signed_gap_m':max(v[0] for v in intervals)-min(v[1] for v in intervals),'projected_intervals_m':intervals,
        'definition':'Signed interval separation of the four pad box colliders along world-transformed TCP local+Y; body origins are not pad centers.'}


def geometry(directory,contact):
    fk,_=fk_model();path=Path(directory)
    with h5py.File(path/'trajectory.h5') as h:
        g=h['traj_0'];base=g['obs/extra/robot_base_pose'][()].astype(float);tcp=g['obs/extra/tcp_pose'][()].astype(float)
        q=np.array([decode(x)['arm'] for x in g['obs/agent/qpos'][()]])
        commands=[decode(x) for x in g['actions/commanded_action'][1:]]
        grasp=[decode(x)['gripper'] for x in g['obs/extra/grasp_state_pickup_obj'][()]]
        scene=json.loads(h['traj_0/obs_scene'][()])['scene_params']
    with h5py.File(path/'telemetry.h5') as h:target=h['target/control_step_and_world_xyz'][()][:,1:]
    with np.load(path/'actions.npz') as z:raw=z['model_output'];arm=z['arm'];grip=z['gripper'].reshape(-1)
    assert raw.shape==(900,8) and np.isfinite(raw).all() and len(q)==901
    assert np.array_equal(arm,raw[:,:7]) and np.array_equal(arm,np.array([x['arm'] for x in commands]))
    assert np.array_equal(grip,np.where(raw[:,7]<127.5,0,255)) and np.array_equal(grip,[x['gripper'][0] for x in commands])
    achieved=frame_world(tcp[:,:3],base)
    actual_fk=np.array([fk(v) for v in q]);offset=tcp[0,:3]-actual_fk[0]
    assert np.allclose(offset,[0,0,.35],atol=1e-6)
    error=np.linalg.norm(frame_world(actual_fk,base,offset)-achieved,axis=1)
    assert error.max()<.01
    requested=frame_world(np.array([fk(v) for v in arm]),base[:-1],offset)
    close=first(grip>=127.5);lift=target[:,2]-target[0,2]
    touch=np.array([g['touching'] for g in grasp]);held=np.array([g['held'] for g in grasp])
    result={'close_command_index':close,'first_touch_observation':first(touch),'first_held_observation':first(held),
        'first_lift_observation':first(lift>=.01),'sustained_lift_15_observations':longest(lift>=.01,1)>=15,
        'raw_touched_never_held':bool(touch.any() and not held.any()),'fk_error_max_m':float(error.max()),
        'close_tracking_gap_z_m':None,'close_fk_error_max_m':None,'early_depth_eligible':False,
        'dynamic_rim':None,'instrumented_chronology':False,'acquisition_discrepancy_supported':False,
        'chunk_dispersion':None,'empty_hand_transport':None,'translation_only_rim':None}
    tray=np.asarray(scene['place_receptacle_start_pose'][:2])
    result['empty_hand_transport']=bool(np.any(np.linalg.norm(achieved[:,:2]-tray,axis=1)<.1) and not np.any(lift>=.01))
    boxes=audit_helpers().boxes();local_vertices=np.concatenate([b['vertices'] for b in boxes])
    with h5py.File(path/'initial_observation.h5') as h:
        initial_q=h['physics/qpos'][()]
        candidates=[i for i in range(len(initial_q)-6) if np.allclose(initial_q[i:i+3],target[0],atol=1e-7,rtol=0)]
        assert len(candidates)==1
        cup_quat=initial_q[candidates[0]+3:candidates[0]+7]
    initial_vertices=R.from_quat(cup_quat[[1,2,3,0]]).apply(local_vertices)+target[0]
    top_above_origin=float(initial_vertices[:,2].max()-target[0,2])
    reference_rim=target[:,2]+top_above_origin
    result['translation_only_rim']={'top_above_origin_m':top_above_origin,'minimum_early_achieved_depth_m':None,
        'close_achieved_depth_m':None,'close_requested_depth_m':None}
    if close is not None:
        result['close_tracking_gap_z_m']=float(achieved[close,2]-requested[close,2])
        result['close_fk_error_max_m']=float(error[max(0,close-30):min(901,close+61)].max())
        eligible=(np.arange(901)<=min(close+60,900))&(np.linalg.norm(achieved[:,:2]-target[:,:2],axis=1)<.06)&(lift<.01)
        result['early_depth_eligible']=bool(eligible.any())
        result['translation_only_rim'].update(minimum_early_achieved_depth_m=float((achieved[:,2]-reference_rim)[eligible].min()) if eligible.any() else None,
            close_achieved_depth_m=float(achieved[close,2]-reference_rim[close]),close_requested_depth_m=float(requested[close,2]-reference_rim[close]))
        result['command_close_z_m']=float(requested[close,2]);result['achieved_close_z_m']=float(achieved[close,2])
        result['next_observation_tracking_gap_z_m']=float(achieved[close+1,2]-requested[close,2])
    diag=path/'grasp_diagnostics.h5'
    if diag.exists():
        with h5py.File(diag) as h:
            if 'control/body_pose' in h:
                poses=h['control/body_pose'][()].reshape(901,3,7);positions=h['control/cup_geom_pos'][()]
                rotations=h['control/cup_geom_rotation'][()].reshape(901,5,3,3)
                meta=json.loads(h.attrs['metadata']);sizes=np.array(meta['cup_geom_sizes'])
                signs=np.array(list(itertools.product((-1,1),repeat=3)))
                vertices=np.einsum('tgij,gvj->tgvi',rotations,signs[None]*sizes[:,None])+positions[:,:,None]
                rims=vertices[:,:,:,2].max((1,2));initial_reference=target[:,2]+(rims[0]-target[0,2])
                assert np.max(np.linalg.norm(poses[:,0,:3]-target,axis=1))<.003
                result['instrumented_chronology']=close is not None and first(touch) is not None
                result['dynamic_rim']={'max_reference_shift_m':float(np.max(np.abs(rims-initial_reference))),
                    'minimum_early_achieved_depth_m':float((achieved[:,2]-rims)[eligible].min()) if close is not None and eligible.any() else None,
                    'minimum_early_initial_reference_depth_m':float((achieved[:,2]-initial_reference)[eligible].min()) if close is not None and eligible.any() else None,
                    'close_achieved_depth_m':None if close is None else float(achieved[close,2]-rims[close]),
                    'close_requested_depth_m':None if close is None else float(requested[close,2]-rims[close]),
                    'pad_body_origin_separation_at_close_m':None if close is None else float(np.linalg.norm(poses[close,1,:3]-poses[close,2,:3]))}
                if close is not None:
                    world_rotation=R.from_quat(base[close,[4,5,6,3]])*R.from_quat(tcp[close,[4,5,6,3]])
                    axis=world_rotation.apply([0,1,0])
                    result['dynamic_rim']['pad_collider_opening_at_close']=pad_collider_gap(path,poses[close],meta,axis)
                    wall_indices=[i for i,g in enumerate(meta['cup_geom_ids']) if not meta['geom_names'][g].endswith('_2')]
                    result['dynamic_rim']['closing_axis_to_wall_normal_deg']={meta['geom_names'][meta['cup_geom_ids'][i]]:float(np.rad2deg(np.arccos(np.clip(abs(rotations[close,i,:,int(np.argmin(sizes[i]))]@axis),0,1)))) for i in wall_indices}
                    gap=result['close_tracking_gap_z_m'];fkerror=result['close_fk_error_max_m']
                    near_contact=contact['raw_contact_kinds']['robot_cup']['first_control_step']
                    result['acquisition_discrepancy_supported']=bool(gap>.005 and gap>fkerror and near_contact is not None
                        and near_contact<=close+60 and not contact['continuous_bilateral_1s'])
            if 'chunks/normalized_action' in h:
                chunks=h['chunks/normalized_action'];assert chunks.shape==(900,100,8)
                assert np.array_equal(h['chunks/query_control_index'][()],np.arange(900))
                assert h['chunks'].attrs['hook_tensor_equality_checks']==900
                assert np.array_equal(h['chunks/post_aggregate'][()],raw)
                contributors=h['chunks/contributors'][()]
                # Independent reconstruction for every step, including valid zeros and exact age.
                denormalized=chunks[()]*h['chunks/action_std'][()]+h['chunks/action_mean'][()]
                for step in range(900):
                    rows=contributors[contributors[:,0]==step];queries=rows[:,1].astype(int);ages=rows[:,2].astype(int);weights=rows[:,3]
                    assert np.array_equal(queries,np.arange(max(0,step-99),step+1)) and np.array_equal(ages,step-queries)
                    w=np.exp(-.01*ages);w/=w.sum();assert np.array_equal(weights,w)
                    reconstructed=(denormalized[queries,ages]*weights[:,None]).sum(0).astype(np.float32)
                    assert np.array_equal(reconstructed,raw[step]),step
                if close is not None:
                    qs=np.arange(max(0,close-99),close+1);tips=np.array([fk(v[:7]) for v in denormalized[qs,close-qs]])
                    result['chunk_dispersion']={'close_contributors':len(qs),
                        'close_fk_spread_max_m':float(np.linalg.norm(tips-tips.mean(0),axis=1).max()),
                        'close_fk_spread_median_m':float(np.median(np.linalg.norm(tips-tips.mean(0),axis=1))),
                        'interpretation':'Dispersion alone does not prove incompatible approaches or establish aggregation causality.'}
    return result


def recompute(directory):
    directory=Path(directory);original=old_metrics(directory);result=read(directory/'result.json')
    contacts,*_=contact_metrics(directory/'telemetry.h5');geo=geometry(directory,contacts)
    return {**original,**contacts,**geo,'directory':str(directory),'scene_id':result['episode_id'],
        'training_seed':result['checkpoint_seed'],'scene_block':result.get('scene_block',result['checkpoint_seed']),
        'variant':result.get('variant','frozen60000'),'physical_row_digest':result.get('physical_row_digest'),
        'provenance':{k:result.get(k) for k in ('checkpoint_seed','checkpoint_sha256','stats_sha256','variant')},
        'pickup_failure':original['failure_stage']=='touched_without_hold','result_sha256':sha(directory/'result.json')}


def summarize(rows):
    return {'n':len(rows),'success':sum(r['task_success'] for r in rows),'cfts':sum(r['collision_free_task_success'] for r in rows),
        'pickup':sum(r['pickup_failure'] for r in rows),'failure_stages':dict(collections.Counter(r['failure_stage'] for r in rows)),
        'union_frames':sum(r['union_frames'] for r in rows),'physics_samples':sum(r['physics_samples'] for r in rows),
        'avoidance_percent':100*(1-sum(r['union_frames'] for r in rows)/sum(r['physics_samples'] for r in rows)),
        'lifted_without_placement':sum(r['failure_stage']=='lifted_without_placement' for r in rows)}


def compare_rows(candidate,control):
    assert len(candidate)==len(control)
    ca={(r['training_seed'],r['scene_id']):r for r in candidate};co={(r['training_seed'],r['scene_id']):r for r in control}
    assert set(ca)==set(co) and len(ca)==len(candidate)
    pairs={}
    for key in ('task_success','pickup_failure','collision_free_task_success'):
        pairs[key]={'candidate_only':sum(ca[k][key] and not co[k][key] for k in ca),
            'control_only':sum(co[k][key] and not ca[k][key] for k in ca)}
    repair=[k for k in ca if co[k]['pickup_failure'] and ca[k]['target_lifted_1cm'] and ca[k]['continuous_bilateral_1s']]
    return {'candidate':summarize(candidate),'control':summarize(control),'paired':pairs,
        'acquisition_repairs':len(repair),'repair_ids':[list(k) for k in repair],
        'repair_chronology':[{ 'training_seed':k[0],'scene_id':k[1],'first_lift':ca[k]['first_lift_observation'],
            'bilateral_runs_1s':ca[k]['bilateral_runs_1s']} for k in repair],
        'repair_limitation':'Counts require actual lift and a continuous bilateral episode; chronology is retained to assess whether engagement supported acquisition.'}


def compare_matched(stage):
    rows=read(B/f'metrics/{stage}.json')['rows'];output={}
    controls=['frozen60000'] if stage=='D' else ['frozen60000','uniform63000']
    for control in controls:
        output[control]={}
        for arm in ('ACT','PACT'):
            output[control][arm]={}
            for block in ['pooled',*SEEDS]:
                subset=[r for r in rows if r['arm']==arm and (block=='pooled' or r['training_seed']==block)]
                output[control][arm][str(block)]=compare_rows([r for r in subset if r['variant']=='acquisition63000'],[r for r in subset if r['variant']==control])
    return output


def evaluate_gate(stage):
    checks={};details={}
    if stage=='A':
        matrix=read(B/'metrics/A2.json')['rows'];instrumented=read(B/'metrics/A1.json')['rows']
        replay=read(B/'replay_comparison.json')
        checks['all72_matrix']=len(matrix)==72;checks['all12_replays']=len(instrumented)==12
        required=[B/f'pairings/A/{stage_name}_{r["scene_id"]}.json' for stage_name in ('A1','A2') for r in read(B/f'scene_manifests/{stage_name}.json')['scenes']]
        checks['initial_pairings']=len(required)==30 and all(p.exists() and read(p)['passed'] for p in required)
        checks['replays_explained']=replay['comparison_eligible']
        pickup=[r for r in matrix if r['pickup_failure']]
        stages=collections.Counter(r['failure_stage'] for r in matrix if not r['task_success'])
        checks['pickup_at_least12']=len(pickup)>=12;checks['pickup_spans2_checkpoints']=len({r['training_seed'] for r in pickup})>=2
        checks['pickup_largest_category']=bool(stages) and stages['touched_without_hold']==max(stages.values())
        checks['measurable_chronology4']=sum(r['instrumented_chronology'] for r in instrumented)>=4
        supported=[r for r in instrumented if r['arm']=='PACT' and not r['task_success'] and r['acquisition_discrepancy_supported']]
        checks['two_failed_acquisitions']=len(supported)>=2
        checks['no_demonstrated_defect']=replay.get('no_demonstrated_defect',False)
        details={'failure_stages':dict(stages),'supported_ids':[r['scene_id'] for r in supported],
            'by_checkpoint':{str(s):summarize([r for r in matrix if r['training_seed']==s]) for s in SEEDS},
            'by_scene':{scene:{str(r['training_seed']):{'success':r['task_success'],'pickup':r['pickup_failure']} for r in matrix if r['scene_id']==scene} for scene in {r['scene_id'] for r in matrix}}}
    elif stage in ('B','C','D'):
        rows=read(B/f'metrics/{stage}.json')['rows'];expected=600 if stage=='D' else 216
        checks['complete_denominator']=len(rows)==expected and len({(r['scene_id'],r['training_seed'],r['arm'],r['variant']) for r in rows})==expected
        details=compare_matched(stage);threshold=9 if stage=='D' else 3
        for control,arms in details.items():
            for arm,blocks in arms.items():
                pooled=blocks['pooled'];c=pooled['candidate'];b=pooled['control'];prefix=f'{control}/{arm}'
                checks[prefix+'/pooled_success']=c['success']-b['success']>=(threshold if arm=='PACT' else 0)
                checks[prefix+'/pooled_cfts']=c['cfts']>=b['cfts']
                checks[prefix+'/union_frames']=c['union_frames']<=1.10*b['union_frames']
                if arm=='PACT':
                    checks[prefix+'/pickup_reduction']=b['pickup']-c['pickup']>=threshold
                    if stage=='B':checks[prefix+'/measured_repairs']=pooled['acquisition_repairs']>=3
                for seed in SEEDS:
                    c=blocks[str(seed)]['candidate'];b=blocks[str(seed)]['control']
                    checks[prefix+f'/{seed}/success']=c['success']>=b['success']-(0 if arm=='PACT' and seed==3105 else 1)
                    checks[prefix+f'/{seed}/cfts']=c['cfts']>=b['cfts']-1
                    checks[prefix+f'/{seed}/avoidance']=c['avoidance_percent']>=b['avoidance_percent']-.5
                    if arm=='PACT' and seed==3103:checks[prefix+'/3103/postlift']=c['lifted_without_placement']<=b['lifted_without_placement']+1
        if stage=='C':
            validation=read(B/'validation_close_errors.json')
            for seed in SEEDS:
                uniform=validation[f'PACT_{seed}_uniform63000']['median_m'];candidate=validation[f'PACT_{seed}_acquisition63000']['median_m']
                checks[f'validation/{seed}']=candidate<=uniform*1.10
        if stage=='D':
            p=details['frozen60000']['PACT']['pooled']['candidate']['success'];a=details['frozen60000']['ACT']['pooled']['candidate']['success']
            checks['original_success_target']=p>=76;checks['original_advantage_target']=p-a>=15
    else:raise ValueError(stage)
    doc={'schema':SCHEMA,'stage':stage,'passed':all(checks.values()),'checks':checks,'details':details,
        'failed_checks':[k for k,v in checks.items() if not v]}
    freeze(B/f'gates/{stage}.json',doc);return doc
