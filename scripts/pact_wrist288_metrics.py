"""Reconstruct endpoints from retained physics contacts and object poses."""
import json
import re
import h5py
import numpy as np
from pact_wrist288_common import *
import pact_place_v1010_contract as collection
sha256_file=sha

DISALLOWED = ('hazard_bar','other_environment','clutter','mounted_fixture')


def metrics(directory, schedule=None):
    directory = Path(directory)
    result = read(directory / 'result.json')
    assert result['status'] == 'complete'
    assert type(result['task_success']) is bool
    info = result['policy_info']
    assert info['averaging_history'] in (10,100)
    assert result['arm'] != 'PACT' or info['proximity_projection_calls']==900
    assert info['num_queries'] == 100 and info['control_steps'] == 900
    assert info['model_output_trace_steps'] == 900
    assert info['sampler_class'] == collection.SAMPLER_CLASS
    assert info['sampler_module'] == 'molmo_spaces.tasks.enclosure_reach'
    assert info['native_thread_environment'] == THREAD_ENV
    if schedule:
        assert info['averaging_history']==schedule['averaging_history']
        for key in ('arm','episode_id','rollout_id','checkpoint_seed','checkpoint_sha256','schedule_row_sha256','row_sha256'):
            assert result[key] == schedule[key], (key,result[key],schedule[key])
    telemetry = directory / 'telemetry.h5'
    assert sha256_file(telemetry) == info['raw_telemetry_sha256']
    assert sha256_file(directory / 'initial_observation.h5') == info['initial_observation_sha256']
    audit = result['contact_audit']
    with h5py.File(telemetry,'r') as h:
        classes = json.loads(h['contacts'].attrs['class_names'])
        slots = json.loads(h['contacts'].attrs['slot_names'])
        counts = h['contacts/class_entries'][()]
        objects = h['contacts/slot_entries'][()]
        times = h['contacts/sim_time_s'][()]
        assert len(times) == audit['sample_count'] and np.all(np.diff(times)>0)
        assert counts.shape == (len(times),len(classes))
        assert objects.shape == (len(times),len(slots))
        pair_counts = h['contacts/pair_counts'][()]
        identities = json.loads(h['contacts'].attrs['pair_identities'])
        derived_counts = np.zeros_like(counts,dtype=np.int64)
        derived_objects = np.zeros_like(objects,dtype=np.int64)
        from molmo_spaces.tasks.pact_place_contact_audit import classify_contact
        for pair_id,identity in enumerate(identities):
            pair = dict(zip(('geom1','geom2','body1','body2','root1','root2'),identity['names']))
            category = classify_contact(pair)
            matched = set(re.findall(r'pact_clutter_(\d{2})(?:[/_]|$)',' '.join(identity['names']))) & set(slots)
            assert category == identity['class'] and matched == set(identity['slots'])
            entries = pair_counts[pair_counts[:,1] == pair_id]
            np.add.at(derived_counts[:,classes.index(category)],entries[:,0],entries[:,2])
            for slot in matched:
                np.add.at(derived_objects[:,slots.index(slot)],entries[:,0],entries[:,2])
        assert np.array_equal(counts,derived_counts)
        assert np.array_equal(objects,derived_objects)
        totals = {c:int(counts[:,i].sum()) for i,c in enumerate(classes)}
        frames = {c:int((counts[:,i]>0).sum()) for i,c in enumerate(classes)}
        assert totals == audit['contact_class_totals']
        assert frames == audit['frames_with_contact']
        measures = h['stability/measures'][()]
        baselines = json.loads(h['stability'].attrs['baseline'])
        stable_slots = json.loads(h['stability'].attrs['slot_names'])
        assert stable_slots == slots and measures.shape[1:] == (len(slots),14)
        assert measures.shape[0] == 901
        assert np.isfinite(measures).all()
        per_object = {}
        for i,slot in enumerate(slots):
            baseline = baselines[slot]
            displacement = np.linalg.norm(measures[:,i,2:5] - np.asarray(baseline['position_m']),axis=1)
            rotations = measures[:,i,5:].reshape(-1,3,3)
            reference = np.asarray(baseline['xmat']).reshape(3,3)
            cosine = np.clip((np.trace(np.einsum('ij,tjk->tik',reference.T,rotations),axis1=1,axis2=2)-1)/2,-1,1)
            angles = np.arccos(cosine)
            assert np.allclose(displacement,measures[:,i,0],atol=1e-12)
            assert np.allclose(angles,measures[:,i,1],atol=1e-7)
            event = bool(np.any(displacement>0.02) or np.any(angles>np.deg2rad(25)))
            per_object[slot] = {'contact_entries':int(objects[:,i].sum()),
                'contact_frames':int((objects[:,i]>0).sum()),'contact_episode':bool(np.any(objects[:,i]>0)),
                'stability_event':event,'stable':not event,'max_displacement_m':float(displacement.max()),
                'max_rotation_deg':float(np.rad2deg(angles.max()))}
            assert per_object[slot]['contact_entries'] == audit['per_object'][slot]['contact_entries']
            assert per_object[slot]['contact_frames'] == audit['per_object'][slot]['contact_frames']
        stability_events = sum(o['stability_event'] for o in per_object.values())
        assert stability_events == info['clutter_stability_event_count']
        assert stability_events == len(info['clutter_stability_events'])
    trajectory = directory / 'trajectory.h5'
    assert sha256_file(trajectory) == result['trajectory_retention']['full_h5_sha256']
    with h5py.File(trajectory,'r') as h:
        scene=json.loads(h['traj_0/obs_scene'][()])['scene_params']
        assert scene['pact_place_environment_version']==collection.ENVIRONMENT_VERSION
        assert scene['pact_v1010_active_clutter_count']==4
        assert h['traj_0/success'].shape == (901,)
        success = bool(h['traj_0/success'][-1])
        assert success == result['task_success']
        assert h['traj_0/obs/proximity/'+CANONICAL_SENSOR_NAMES[0]].shape[0] == 901
    with h5py.File(telemetry,'r') as h:
        target=h['target/control_step_and_world_xyz'][()]
        assert target.shape==(901,4) and np.array_equal(target[:,0],np.arange(901))
        assert np.isfinite(target).all()
        lift=float((target[:,3]-target[0,3]).max())
    with h5py.File(trajectory,'r') as h:
        def decode(blob): return json.loads(bytes(blob).split(b'\0',1)[0])
        task=[decode(x) for x in h['traj_0/obs/extra/task_info'][()]]
        grasp=[decode(x)['gripper'] for x in h['traj_0/obs/extra/grasp_state_pickup_obj'][()]]
        assert len(task)==len(grasp)==901
        touch=any(x['touching'] for x in grasp)
        held=any(x['held'] for x in grasp)
        support=any(x['supported_by_receptacle'] for x in task)
        assert np.array_equal(h['traj_0/success'][()],np.asarray([x['success'] for x in task]))
    stage=('success' if success else 'supported_without_final_success' if support else
        'lifted_without_placement' if lift>=0.01 else 'held_without_lift' if held else
        'touched_without_hold' if touch else 'no_target_interaction')
    collision_free = all(totals[c] == 0 for c in DISALLOWED)
    cfts = success and collision_free
    assert collision_free == audit['collision_free']
    assert cfts == result['collision_free_task_success']
    return {'arm':result['arm'],'seed':result['checkpoint_seed'],'episode_id':result['episode_id'],
        'task_seed':result['seed'],'row_sha256':result['row_sha256'],
        'task_success':success,'collision_free_task_success':cfts,'collision_free':collision_free,
        'target_touch':touch,'target_held':held,'target_lifted_1cm':lift>=0.01,'max_target_lift_m':lift,
        'placement_support':support,'failure_stage':stage,'averaging_history':info['averaging_history'],
        'hazard_contact_episode':frames['hazard_bar']>0,'hazard_frames':frames['hazard_bar'],
        'hazard_entries':totals['hazard_bar'],'clutter_contact_episode':frames['clutter']>0,
        'contact_entries':totals,'contact_frames':frames,'clutter_stability_events':stability_events,
        'per_object':per_object,'directory':str(directory.relative_to(ROOT))}
