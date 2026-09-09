"""Read-only approach/arm-motion and chunk-100 decoder audit.

Distances are explicitly to the INITIAL target origin, not a live target pose.
The robot-frame TCP is transformed with the retained world-frame base pose.
"""
import json
from pathlib import Path

from pact_place_v1011c_experiment import ROOT, read, freeze, empty_authorization
import h5py
import numpy as np
from scipy.spatial.transform import Rotation

OUT = ROOT / 'diagnostics_output/pact_place_v1011c_post_eval_audit'


def decode(row):
    return json.loads(row.tobytes().split(b'\0', 1)[0])


def extract(path, actions_path=None):
    with h5py.File(path, 'r') as h:
        b = h['traj_0/obs/extra/robot_base_pose'][()].astype(float)
        tcp = h['traj_0/obs/extra/tcp_pose'][()].astype(float)
        target = h['traj_0/obs/extra/obj_start'][()].astype(float)
        assert b.shape == tcp.shape == target.shape and b.shape[1] == 7
        assert np.allclose(np.linalg.norm(b[:, 3:], axis=1), 1, atol=1e-6)
        assert np.allclose(np.linalg.norm(tcp[:, 3:], axis=1), 1, atol=1e-6)
        assert np.array_equal(b, np.broadcast_to(b[0], b.shape))
        assert np.array_equal(target, np.broadcast_to(target[0], target.shape))
        rotation = Rotation.from_quat(b[:, [4, 5, 6, 3]])
        world_tcp = rotation.apply(tcp[:, :3]) + b[:, :3]
        assert np.allclose(rotation.inv().apply(world_tcp - b[:, :3]), tcp[:, :3])
        q = np.array([decode(r)['arm'] for r in h['traj_0/obs/agent/qpos'][()]])
        touch = np.array([decode(r)['gripper']['touching'] for r in h['traj_0/obs/extra/grasp_state_pickup_obj'][()]])
        seen = h['traj_0/obs/extra/object_image_points/pickup_obj/wrist_camera/num_points'][()].reshape(-1) > 0
        scene = json.loads(h['traj_0/obs_scene'][()])
        dt = float(scene['policy_dt_ms']) / 1000
        tray = np.array(scene['scene_params']['place_receptacle_start_pose'][:2])
        distance = np.linalg.norm(world_tcp - target[:, :3], axis=1)
        increments = np.linalg.norm(np.diff(world_tcp, axis=0), axis=1)
        first_touch = int(np.flatnonzero(touch)[0]) if touch.any() else None
        first_seen = int(np.flatnonzero(seen)[0]) if seen.any() else None
        before = slice(0, first_touch + 1) if first_touch is not None else slice(None)
        result = {
            'frames': len(tcp), 'policy_dt_s': dt, 'any_target_touch': bool(touch.any()),
            'task_success': bool(h['traj_0/success'][-1]),
            'first_touch_step': first_touch, 'first_segmented_step': first_seen,
            'target_initial_xyz_m': target[0, :3].tolist(),
            'tcp_world_initial_xyz_m': world_tcp[0].tolist(),
            'tcp_world_final_xyz_m': world_tcp[-1].tolist(),
            'tcp_world_at100_xyz_m': world_tcp[min(100,len(tcp)-1)].tolist(),
            'tcp_initial_target_origin_distance_m': float(distance[0]),
            'tcp_min_initial_target_origin_distance_before_touch_m': float(distance[before].min()),
            'tcp_final_initial_target_origin_distance_m': float(distance[-1]),
            'tcp_first100_target_distance_reduction_m': float(distance[0]-distance[:101].min()),
            'tcp_path_length_m': float(increments.sum()),
            'tcp_first100_path_length_m': float(increments[:100].sum()),
            'tcp_last100_path_length_m': float(increments[-100:].sum()),
            'tcp_last100_extent_diagonal_m': float(np.linalg.norm(np.ptp(world_tcp[-101:],axis=0))),
            'tcp_net_displacement_m': float(np.linalg.norm(world_tcp[-1]-world_tcp[0])),
            'tcp_first100_delta_x_m': float(world_tcp[min(100,len(tcp)-1),0]-world_tcp[0,0]),
            'tcp_final_tray_xy_distance_m': float(np.linalg.norm(world_tcp[-1,:2]-tray)),
            'tcp_any_tray_xy_under10cm_before_touch': bool((np.linalg.norm(world_tcp[before,:2]-tray,axis=1)<.1).any()),
            'tcp_stays_within2cm_last100': bool(np.linalg.norm(np.ptp(world_tcp[-101:],axis=0))<.02),
            'tcp_gets_within10cm_initial_target_origin_before_touch': bool((distance[before]<.1).any()),
            'joint_last100_extent_l2_rad': float(np.linalg.norm(np.ptp(q[-101:],axis=0))),
        }
        if actions_path is not None:
            with np.load(actions_path) as archive:
                cmd = archive['arm'].astype(float)
                assert cmd.shape == (len(q)-1, 7) and np.isfinite(cmd).all()
                assert np.array_equal(cmd, archive['model_output'][:, :7])
            recorded = np.array([decode(r)['arm'] for r in h['traj_0/actions/commanded_action'][1:]])
            assert np.array_equal(cmd, recorded)
            errors = np.linalg.norm(cmd - q[1:], axis=1)
            result.update(
                median_arm_tracking_error_l2_rad=float(np.median(errors)),
                last100_median_arm_tracking_error_l2_rad=float(np.median(errors[-100:])),
                first100_median_arm_tracking_error_l2_rad=float(np.median(errors[:100])),
                max_arm_tracking_error_l2_rad=float(errors.max()),
                commanded_arm_last100_extent_l2_rad=float(np.linalg.norm(np.ptp(cmd[-100:],axis=0))),
            )
    return result


def aggregate(rows):
    if not rows:
        return {'n': 0}
    out = {'n': len(rows)}
    for key, value in rows[0].items():
        if isinstance(value, bool):
            out[key] = sum(r[key] for r in rows)
        elif isinstance(value, (int, float)) and key not in ('seed', 'frames'):
            values = [r[key] for r in rows if r[key] is not None]
            out['median_'+key] = float(np.median(values)) if values else None
    for key in ('first_touch_step','first_segmented_step'):
        vals = [r[key] for r in rows if r[key] is not None]
        out['median_'+key] = float(np.median(vals)) if vals else None
    return out


def main():
    old = read(OUT/'audit_v2.json')
    doc = {**empty_authorization(), 'versions': {}, 'training': {}, 'limitations': [
        'TCP is transformed from robot frame into world frame with the recorded fixed base pose.',
        'Distances reference the cached INITIAL target origin, not live target pose or surface separation.',
        'Two-centimeter last-100-step TCP extent is a post-hoc stalling descriptor, not a registered outcome.',
        'Tray distances are XY only; passing above the tray is not verified placement behavior.',
        'Only post-ensemble commands are retained; individual predictions cannot be recovered by undoing their average.',
        'Camera intrinsics were not used: stored principal-point dimensions are reversed relative to the retained RGB shape; the policy itself does not use these matrices.'
    ]}
    ages = np.arange(100)
    weights = np.exp(-.01*ages); weights /= weights.sum()
    doc['chunk100_ensemble'] = {
        'inference_every_control_step': True, 'control_step_s': .066,
        'prediction_span_s': 100*.066, 'oldest_prediction_age_s': 99*.066,
        'weighted_mean_prediction_age_steps': float(weights@ages),
        'weighted_mean_prediction_age_s': float(weights@ages*.066),
        'current_prediction_weight': float(weights[0]),
        'last10_predictions_total_weight': float(weights[:10].sum()),
        'predictions_at_least50_steps_old_weight': float(weights[50:].sum()),
        'note': 'These are ages of observations underlying current-step predictions, not a measured motor latency or an open-loop execution period.'
    }
    for version in ('V10.10', 'V10.11c'):
        rows = []
        for original in old['versions'][version]['rows']:
            directory = ROOT/original['directory']
            row = extract(directory/'trajectory.h5', directory/'actions.npz')
            row.update({k:original[k] for k in ('arm','seed','episode_id','directory','collision_episode','hazard_episode','clutter_episode')})
            assert row['task_success'] == original['task_success'] and row['any_target_touch'] == original['any_target_touch']
            rows.append(row)
        groups = {}
        for arm in ('ACT','PACT'):
            groups[arm] = {label:aggregate([r for r in rows if r['arm']==arm and predicate(r)])
                for label,predicate in (
                    ('all',lambda r:True), ('never_touch',lambda r:not r['any_target_touch']),
                    ('touch',lambda r:r['any_target_touch']),
                    ('never_touch_no_collision',lambda r:not r['any_target_touch'] and not r['collision_episode']),
                    ('never_seen',lambda r:r['first_segmented_step'] is None))}
        doc['versions'][version] = {'groups':groups,'rows':rows}
        print(version, json.dumps({a:groups[a]['never_touch'] for a in groups}), flush=True)
    for version in ('v1010','v1011c'):
        base=ROOT/f'diagnostics_output/pact_place_{version}_train_eval'
        source=read(base/'source_manifest.json');split=read(base/'split_manifest.json')
        indices={r['act_episode_index'] for r in split['episodes'] if r['split']=='train'}
        rows=[]
        for original in source['rows']:
            if original['act_episode_index'] in indices:
                row=extract(ROOT/original['trajectory_h5'])
                row['attempt_id']=original['attempt_id'];rows.append(row)
        doc['training'][version]={'aggregate':aggregate(rows),'rows':rows}
        print(version,'TRAIN',json.dumps(doc['training'][version]['aggregate']),flush=True)
    freeze(OUT/'approach_chunk100.json',doc)
    print('Verified approach_chunk100.json',flush=True)


if __name__ == '__main__':
    main()
