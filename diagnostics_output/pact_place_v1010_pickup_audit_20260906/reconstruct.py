"""Read retained artifacts only; no simulator, policy inference, or dataset writes."""
import hashlib
import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path('/root/prox_learning_pact_remediation')
OUT = Path(__file__).parent
PRIOR = ROOT / 'diagnostics_output/pact_place_v1011c_post_eval_audit'
stats_before = {}


def checked(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    assert path.resolve().is_relative_to(ROOT)
    st = path.stat()
    stats_before.setdefault(path, (st.st_size, st.st_mtime_ns))
    return path


def read(path):
    return json.loads(checked(path).read_text())


def sha(path):
    return hashlib.sha256(checked(path).read_bytes()).hexdigest()


def decode(row):
    return json.loads(row.tobytes().split(b'\0', 1)[0])


def runs(values):
    edges = np.diff(np.r_[False, values, False].astype(int))
    return [[int(a), int(b - 1)] for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]


def first(values):
    return int(np.flatnonzero(values)[0]) if np.any(values) else None


old = read(PRIOR / 'audit_v2.json')
approach = read(PRIOR / 'approach_chunk100.json')
flags = {k: False for k in old if k.startswith('authorizes_') or k in
         ('eligible_for_human_review', 'human_approval_present', 'phase0_passed')}
assert all(old[k] is False for k in flags)
doc = {**flags, 'scope': 'Reconstruction of retained completed evaluations; no new inference or rollout.',
       'versions': {}, 'labels': {}, 'source_hashes': {}}

for version, folder, n in [('V10.10', 'pact_place_v1010_eval_infra_repair_01', 80),
                           ('V10.11c', 'pact_place_v1011c_eval', 300)]:
    ledger_path = ROOT / 'diagnostics_output' / folder / 'full_run.json'
    ledger = read(ledger_path)
    assert len(ledger['results']) == n
    assert len({r['rollout_id'] for r in ledger['results']}) == n
    assert all(ledger[k] is False for k in flags if k in ledger)
    doc['source_hashes'][str(ledger_path.relative_to(ROOT))] = sha(ledger_path)
    assert sha(ledger_path) == old['versions'][version]['full_run_sha256']
    prior_rows = {r['directory']: r for r in old['versions'][version]['rows']}
    prior_motion = {r['directory']: r for r in approach['versions'][version]['rows']}
    rows = []
    for rec in ledger['results']:
        assert rec['status'] == 'complete' and rec['returncode'] == 0
        directory = checked(rec.get('row_dir', rec.get('directory')))
        rel = str(directory.relative_to(ROOT))
        result = read(directory / 'result.json')
        assert result['status'] == 'complete'
        assert all(rec[k] == result[k] for k in ('arm', 'episode_id', 'checkpoint_seed', 'checkpoint_sha256'))
        assert sha(directory / 'result.json') == prior_rows[rel]['result_sha256']
        assert result['policy_info']['num_queries'] == 100
        assert result['policy_info']['control_steps'] == 900
        with h5py.File(checked(directory / 'trajectory.h5'), 'r') as h:
            g = h['traj_0']
            ti = [decode(x) for x in g['obs/extra/task_info'][()]]
            gs = [decode(x) for x in g['obs/extra/grasp_state_pickup_obj'][()]]
            assert len(ti) == len(gs) == 901
            assert all({'success', 'supported_by_receptacle', 'robot_contact', 'position_error'} <= set(x) for x in ti)
            assert all(set(x['gripper']) == {'touching', 'held'} for x in gs)
            touch = np.array([x['gripper']['touching'] for x in gs])
            held = np.array([x['gripper']['held'] for x in gs])
            support = np.array([x['supported_by_receptacle'] for x in ti])
            success = g['success'][()]
            assert np.array_equal(success, [x['success'] for x in ti])
            assert np.all(~held | touch)
            assert bool(success[-1]) == result['task_success']
            assert np.array_equal([x['episode_step'] for x in ti], np.arange(901))
            start = g['obs/extra/obj_start'][()]
            assert np.array_equal(start, np.broadcast_to(start[0], start.shape))
            assert np.all(g['obs/extra/obj_end'][()] == 0)
            env_fields = []
            g['env_states'].visititems(lambda k, v: env_fields.append(k) if isinstance(v, h5py.Dataset) else None)
            assert env_fields == ['articulations/panda']
            base = g['obs/extra/robot_base_pose'][()].astype(float)
            tcp = g['obs/extra/tcp_pose'][()].astype(float)
            assert np.array_equal(base, np.broadcast_to(base[0], base.shape))
            assert np.allclose(np.linalg.norm(base[:, 3:], axis=1), 1, atol=1e-6)
            world = Rotation.from_quat(base[:, [4, 5, 6, 3]]).apply(tcp[:, :3]) + base[:, :3]
            scene = json.loads(g['obs_scene'][()])
            assert scene['policy_dt_ms'] == 66
            tray = np.array(scene['scene_params']['place_receptacle_start_pose'][:2])
            tray_distance = np.linalg.norm(world[:, :2] - tray, axis=1)
            target_distance = np.linalg.norm(world - start[:, :3], axis=1)
            counts = g['obs/extra/object_image_points/pickup_obj/wrist_camera/num_points'][()].reshape(-1)
            points = g['obs/extra/object_image_points/pickup_obj/wrist_camera/points'][()]
            assert np.array_equal(counts, np.isfinite(points).all(-1).sum(-1))
            assert counts.min() >= 0 and counts.max() <= 10
            seen = counts > 0
            q = np.array([decode(x)['arm'] for x in g['obs/agent/qpos'][()]])
            command = np.array([decode(x)['arm'] for x in g['actions/commanded_action'][1:]])
            with np.load(checked(directory / 'actions.npz'), allow_pickle=False) as a:
                assert command.shape == (900, 7)
                assert np.array_equal(command, a['arm'])
                assert np.array_equal(command, a['model_output'][:, :7])
            video = bytes(g['obs/sensor_data/wrist_camera'][()]).split(b'\0', 1)[0].decode()
            phase = sorted(set(g['obs/extra/policy_phase'][()].tolist()))
        contact = result['contact_audit']
        collision = any(contact['contact_class_totals'][k] > 0 for k in
                        ('hazard_bar', 'other_environment', 'clutter', 'mounted_fixture'))
        assert result['collision_free_task_success'] == (bool(success[-1]) and not collision)
        before = slice(0, first(touch) + 1) if touch.any() else slice(None)
        row = dict(directory=rel, arm=rec['arm'], seed=rec['checkpoint_seed'], episode_id=rec['episode_id'],
                   result_sha256=sha(directory / 'result.json'),
                   touch=bool(touch.any()), held_contact_only=bool(held.any()),
                   support=bool(support.any()), success=bool(success[-1]),
                   transient_success_lost=bool(success.any() and not success[-1]),
                   collision_free_success=result['collision_free_task_success'], collision=collision,
                   physics_robot_target_contact=contact['frames_with_contact']['grasp_target'] > 0,
                   physics_robot_target_contact_samples=contact['frames_with_contact']['grasp_target'],
                   first_touch=first(touch), first_held=first(held), first_support=first(support),
                   first_success=first(success), first_seen=first(seen),
                   touch_runs=runs(touch), held_runs=runs(held), support_runs=runs(support), success_runs=runs(success),
                   held_samples=int(held.sum()), max_held_run_samples=max([b-a+1 for a,b in runs(held)], default=0),
                   ever_seen=bool(seen.any()), seen_fraction=float(seen.mean()),
                   tray_before_touch=bool((tray_distance[before] < .1).any()),
                   tray_ever=bool((tray_distance < .1).any()), first_tray=first(tray_distance < .1),
                   min_initial_target_distance_before_touch_m=float(target_distance[before].min()),
                   final_tray_xy_distance_m=float(tray_distance[-1]),
                   final100_tcp_extent_m=float(np.linalg.norm(np.ptp(world[-101:], axis=0))),
                   last100_arm_tracking_l2_rad=float(np.median(np.linalg.norm(command[-100:]-q[-100:], axis=1))),
                   position_error_initial_m=ti[0]['position_error'],
                   position_error_min_m=min(x['position_error'] for x in ti), position_error_final_m=ti[-1]['position_error'],
                   final_task_info={k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                                    for k, v in ti[-1].items()}, policy_phase_values=phase,
                   video_present=(directory / video).exists(), actual_lift_countable=False,
                   stable_grasp_countable=False, carried_transport_countable=False)
        for k, prior_key in [('touch','any_target_touch'),('held_contact_only','any_target_held'),
                             ('support','any_receptacle_support'),('success','task_success')]:
            assert row[k] == prior_rows[rel][prior_key]
        assert row['tray_before_touch'] == prior_motion[rel]['tcp_any_tray_xy_under10cm_before_touch']
        row['group'] = ('success' if row['success'] else 'supported_but_final_failure' if row['support'] else
                        'contact_only_held_without_support' if row['held_contact_only'] else
                        'touch_without_held_or_support' if row['touch'] else 'never_sampled_gripper_touch')
        event_steps = sorted({0, 100, 300, 600, 900} | {x for k in ('first_touch','first_held','first_support','first_success','first_tray') if (x := row[k]) is not None}
                             | {x for a,b in row['held_runs'] for x in (a,b,min(900,b+1))})
        row['events'] = [dict(step=t, tcp_world_xyz_m=world[t].tolist(), touching=bool(touch[t]),
                              held_contact_only=bool(held[t]), support=bool(support[t]), success=bool(success[t]),
                              wrist_seen=bool(seen[t]), position_error_m=ti[t]['position_error']) for t in event_steps]
        rows.append(row)
    summaries = {}
    for arm in ('ACT', 'PACT'):
        rr = [r for r in rows if r['arm'] == arm]
        never = [r for r in rr if not r['touch']]
        summaries[arm] = {'n': len(rr), **{k: sum(r[k] for r in rr) for k in
            ('touch','held_contact_only','support','success','transient_success_lost','collision_free_success','ever_seen','physics_robot_target_contact')},
            'actual_lift': None, 'stable_grasp': None, 'carried_transport': None,
            'groups': dict(Counter(r['group'] for r in rr)),
            'never_touch_tray': sum(r['tray_ever'] for r in never),
            'never_touch_any_physics_robot_target_contact': sum(r['physics_robot_target_contact'] for r in never),
            'never_touch_tray_without_physics_robot_target_contact': sum(r['tray_ever'] and not r['physics_robot_target_contact'] for r in never),
            'never_touch_no_collision': sum(not r['collision'] for r in never),
            'never_touch_final100_extent_under2cm': sum(r['final100_tcp_extent_m'] < .02 for r in never),
            'median_seen_fraction': float(np.median([r['seen_fraction'] for r in rr])),
            'video_present': sum(r['video_present'] for r in rr)}
    doc['versions'][version] = {'ledger': str(ledger_path.relative_to(ROOT)), 'rows': rows, 'arms': summaries}
    print(version, json.dumps(summaries), flush=True)

# Independently verify the seven arm label components in both immutable corpora.
for version, count in [('v1010', 144), ('v1011c', 99)]:
    manifest = read(ROOT / f'diagnostics_output/pact_place_{version}_train_eval/source_manifest.json')
    examples = []
    for rec in manifest['rows']:
        idx = rec['act_episode_index']
        with h5py.File(checked(rec['trajectory_h5']), 'r') as h:
            g = h['traj_0']
            js = [decode(x) for x in g['actions/joint_pos'][()]]
            cs = [decode(x) for x in g['actions/commanded_action'][()]]
            qs = [decode(x) for x in g['obs/agent/qpos'][()]]
            assert js[-1] == {} and cs[0] == {}
            j = np.array([x['arm'] for x in js[:-1]], dtype=np.float32)
            c = np.array([x['arm'] for x in cs[1:-1]], dtype=np.float32)
            q = np.array([x['arm'] for x in qs[:-1]], dtype=np.float32)
            assert np.array_equal(j[1:], c)
        path = ROOT / f'assets/act_style_data/pact_place_{version}_{count}/episode_{idx}.hdf5'
        with h5py.File(checked(path), 'r') as h:
            assert np.array_equal(h['action'][:, :7], j)
            assert np.array_equal(h['observations/qpos'][:, :7], q)
            assert h.attrs['sim']
        examples.append({'episode_index': idx, 'source': rec['trajectory_h5'],
                         'converted': str(path.relative_to(ROOT)), 'timesteps': len(j),
                         'terminal_status_without_arm': 'arm' not in cs[-1]})
    assert len(examples) == count
    doc['labels'][version] = {'verified_episodes': len(examples), 'rows': examples,
                              'terminal_status_without_arm': sum(x['terminal_status_without_arm'] for x in examples)}
    print('labels', version, count, doc['labels'][version]['terminal_status_without_arm'], flush=True)

for path, before in stats_before.items():
    st = path.stat()
    assert (st.st_size, st.st_mtime_ns) == before, str(path)
doc['read_only_verification'] = {'input_files_and_directories_with_unchanged_size_mtime': len(stats_before)}
with (OUT / 'reconstruction.json').open('x') as f:
    json.dump(doc, f, indent=2, allow_nan=False)
    f.write('\n')
print('Wrote new reconstruction.json only; input size/mtime checks passed.', flush=True)
