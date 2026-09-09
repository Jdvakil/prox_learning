"""Raw endpoint reconstruction and explicitly frozen two-camera pairing."""
from pact_place_v1011c_dualcam import *
from pact_place_v1011c_metrics import metrics as original_metrics, DISALLOWED
import h5py
import numpy as np


def metrics(directory, schedule=None):
    directory = Path(directory)
    result = read(directory/'result.json')
    info = result['policy_info']
    assert info['policy_camera_names'] == CAMERAS
    assert info['camera_system'] == 'FrankaSkinHybridCameraSystem'
    assert info['action_alignment'] == ALIGNMENT
    assert info['training_global_step'] in MILESTONES
    assert info['temporal_ensemble_history'] == 100
    if schedule:
        assert info['training_global_step'] == schedule['training_global_step']
    row = original_metrics(directory,schedule)
    with h5py.File(directory/'trajectory.h5','r') as h:
        grasp_path = 'traj_0/obs/extra/grasp_state_pickup_obj'
        assert grasp_path in h
        grasp = [json.loads(bytes(x).split(b'\0',1)[0]) for x in h[grasp_path][()]]
        assert len(grasp) == 901 and all('gripper' in x and 'touching' in x['gripper'] for x in grasp)
        row['task_object_touched'] = any(x['gripper']['touching'] for x in grasp)
        row['task_object_visible'] = {}
        for camera in CAMERAS:
            path = f'traj_0/obs/extra/object_image_points/pickup_obj/{camera}/num_points'
            assert path in h, path
            n = h[path][()].reshape(-1)
            assert n.shape == (901,)
            row['task_object_visible'][camera] = bool(np.any(n>0))
    row['training_global_step'] = info['training_global_step']
    row['collision_free'] = all(row['contact_entries'][c] == 0 for c in DISALLOWED)
    return row


def check_pair(first,second):
    assert first['episode_id'] == second['episode_id']
    assert first['task_seed'] == second['task_seed']
    assert first['row_sha256'] == second['row_sha256']
    info = {'episode_id':first['episode_id'],'exact_non_rgb_datasets':0,'rgb':{}}
    with h5py.File(ROOT/first['directory']/'initial_observation.h5','r') as a, \
         h5py.File(ROOT/second['directory']/'initial_observation.h5','r') as b:
        names, other = [],[]
        a.visititems(lambda n,o:names.append(n) if isinstance(o,h5py.Dataset) else None)
        b.visititems(lambda n,o:other.append(n) if isinstance(o,h5py.Dataset) else None)
        assert names == other and names
        for name in names:
            assert a[name].dtype == b[name].dtype and a[name].shape == b[name].shape, name
            av,bv = np.asarray(a[name][()]),np.asarray(b[name][()])
            if name in ['observation/'+camera for camera in CAMERAS]:
                assert av.dtype == np.uint8 and av.ndim == 3 and av.shape[-1] == 3
                delta = np.abs(av.astype(np.int16)-bv.astype(np.int16))
                changed, maximum = int(np.count_nonzero(delta)),int(delta.max())
                info['rgb'][name] = {'shape':list(av.shape),'channel_values':int(av.size),
                    'changed_channel_values':changed,'changed_fraction':changed/av.size,'max_abs_delta_u8':maximum}
                assert maximum <= 1 and changed/av.size <= 0.001, info
            else:
                assert av.tobytes() == bv.tobytes(), f'initial non-RGB mismatch: {name}'
                info['exact_non_rgb_datasets'] += 1
        assert len(info['rgb']) == 2 and info['exact_non_rgb_datasets'] > 0
        info['total_datasets'] = len(names)
    return info


def aggregate(rows):
    n = len(rows)
    assert n
    endpoints = ('task_success','collision_free_task_success','collision_free',
                 'hazard_contact_episode','clutter_contact_episode','task_object_touched')
    return {'n':n,'counts':{key:sum(int(r[key]) for r in rows) for key in endpoints},
        'rates':{key:sum(int(r[key]) for r in rows)/n for key in endpoints},
        'hazard_contact_frames_total':sum(r['hazard_frames'] for r in rows),
        'hazard_contact_frames_mean':sum(r['hazard_frames'] for r in rows)/n,
        'per_object':{slot:{
            'contact_episodes':sum(r['per_object'][slot]['contact_episode'] for r in rows),
            'contact_frames_total':sum(r['per_object'][slot]['contact_frames'] for r in rows),
            'stability_event_episodes':sum(r['per_object'][slot]['stability_event'] for r in rows),
            'max_displacement_m':max(r['per_object'][slot]['max_displacement_m'] for r in rows),
            'max_rotation_deg':max(r['per_object'][slot]['max_rotation_deg'] for r in rows)}
            for slot in ('01','08','09')}}
