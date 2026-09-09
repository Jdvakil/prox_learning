"""Independent raw closeout, without importing the experiment metrics module."""
import argparse
import collections
import datetime
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

C = Path(__file__).resolve().parents[1]
ROOT = C.parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def decode(value):
    return json.loads(bytes(value).split(b'\0', 1)[0])


def read_raw(saved):
    directory = Path(saved['directory'])
    with h5py.File(directory / 'trajectory.h5') as h:
        trajectory = h['traj_0']
        trace = trajectory['success'][()]
        assert trace.shape == (901,)
        tasks = [decode(x) for x in trajectory['obs/extra/task_info'][()]]
        grasp = [decode(x)['gripper'] for x in trajectory['obs/extra/grasp_state_pickup_obj'][()]]
        commands = [decode(x) for x in trajectory['actions/commanded_action'][1:]]
        assert len(tasks) == len(grasp) == 901 and len(commands) == 900
        assert np.array_equal(trace, [x['success'] for x in tasks])
        success = bool(trace[-1])
        support = any(x['supported_by_receptacle'] for x in tasks)
        held = any(x['held'] for x in grasp)
        touch = any(x['touching'] for x in grasp)
    with np.load(directory / 'actions.npz') as z:
        raw, arm, grip = z['model_output'], z['arm'], z['gripper'].reshape(-1)
        assert raw.shape == (900, 8) and np.isfinite(raw).all()
        assert np.array_equal(arm, raw[:, :7])
        assert np.array_equal(arm, [x['arm'] for x in commands])
        assert np.array_equal(grip, np.where(raw[:, 7] < 127.5, 0, 255))
        assert np.array_equal(grip, [x['gripper'][0] for x in commands])
    with h5py.File(directory / 'telemetry.h5') as h:
        classes = json.loads(h['contacts'].attrs['class_names'])
        contacts = h['contacts/class_entries'][()]
        times = h['contacts/sim_time_s'][()]
        assert contacts.shape[0] == len(times) == 29701
        assert np.all(np.diff(times) > 0)
        assert abs(float(np.median(np.diff(times))) - .002) < 1e-9
        target = h['target/control_step_and_world_xyz'][()]
        assert target.shape == (901, 4)
        lift = target[:, 3] - target[0, 3]
        union = np.any(contacts[:, [classes.index(k) for k in ('hazard_bar', 'clutter')]] > 0, axis=1)
        collision_free = not np.any(contacts[:, [classes.index(k) for k in
            ('hazard_bar', 'clutter', 'other_environment', 'mounted_fixture')]] > 0)
        contact_frames = {name: int(np.count_nonzero(contacts[:, i])) for i, name in enumerate(classes)}
    stage = ('success' if success else 'supported_without_final_success' if support else
             'lifted_without_placement' if lift.max() >= .01 else 'held_without_lift' if held else
             'touched_without_hold' if touch else 'no_target_interaction')
    edges = np.diff(np.r_[False, lift >= .01, False].astype(int))
    longest = max((b-a for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))), default=0)
    row = {key: saved[key] for key in ('training_seed', 'variant', 'scene_id')}
    row.update(task_success=success, collision_free_task_success=bool(success and collision_free),
        failure_stage=stage, pickup_failure=stage == 'touched_without_hold',
        union_frames=int(union.sum()), physics_samples=len(times), raw_touched_never_held=bool(touch and not held),
        target_lifted_1cm=bool(lift.max() >= .01), sustained_lift_15_observations=bool(longest >= 15),
        ever_held=bool(held), contact_frames=contact_frames)
    for key in ('task_success', 'collision_free_task_success', 'failure_stage', 'pickup_failure',
                'union_frames', 'physics_samples', 'raw_touched_never_held', 'target_lifted_1cm',
                'sustained_lift_15_observations'):
        assert row[key] == saved[key], (directory, key, row[key], saved[key])
    assert sha(directory / 'result.json') == saved['result_sha256']
    return row


def summary(rows):
    total = sum(row['physics_samples'] for row in rows)
    union = sum(row['union_frames'] for row in rows)
    return {'n': len(rows), 'success': sum(row['task_success'] for row in rows),
        'cfts': sum(row['collision_free_task_success'] for row in rows),
        'pickup': sum(row['pickup_failure'] for row in rows), 'union_frames': union,
        'physics_samples': total, 'avoidance_percent': 100 * (1-union/total),
        'raw_touched_never_held': sum(row['raw_touched_never_held'] for row in rows),
        'target_lifted_1cm': sum(row['target_lifted_1cm'] for row in rows),
        'sustained_lift_15_observations': sum(row['sustained_lift_15_observations'] for row in rows),
        'ever_held': sum(row['ever_held'] for row in rows),
        'failure_stages': dict(collections.Counter(row['failure_stage'] for row in rows)),
        'contact_frames': {key: sum(row['contact_frames'][key] for row in rows) for key in rows[0]['contact_frames']}}


def main(parent_session):
    closure = read(C / 'parent_closure.json')
    assert closure['status'] == 'COMPLETED_COMPARISON' and not closure['active_parent']
    assert not Path(f'/proc/{closure["parent_pid"]}').exists()
    comparison = read(C / 'comparison.json')
    pair = read(C / 'checkpoint/checkpoint_pairs.json')['policy_last.ckpt']
    completed = read(C / 'checkpoint/completed.json')
    assert pair['global_step'] == completed['global_step'] == 60000
    assert completed['strict_pair_reload_exact'] and completed['gradient_check']['stem_and_transformer_nonzero']
    assert sha(C / 'checkpoint/policy_last.ckpt') == pair['policy_sha256']
    assert sha(C / 'checkpoint/prox_encoder.pt') == pair['encoder_sha256']
    upstream = read(C / 'upstream_manifest.json')
    for row in upstream['files']:
        assert row['byte_identical'] and sha(ROOT / row['copied_path']) == row['sha256']
    contract = read(C / 'contract.json')
    for path, expected in contract['code_hashes'].items():
        assert sha(ROOT / path) == expected, path
    for path, expected in contract['input_hashes'].items():
        assert sha(path) == expected, path
    for name in ('03_thread_capacity', '04_capacity_wait', '05_eval_capacity'):
        recovery_dir = C / 'recoveries' / name
        recovery = read(recovery_dir / 'recovery.json')
        assert sha(recovery_dir / 'resume.py') == recovery['launcher_sha256']
        for path, expected in recovery.get('operational_wrapper_hashes', {}).items():
            assert sha(path) == expected, path
        if 'cache_release_code_sha256' in recovery:
            assert sha(C / 'root_review/release_training_cache.py') == recovery['cache_release_code_sha256']
    ignore = read(C / 'root_review/git_diff_recovery.json')
    assert sha(ROOT / ignore['new_ignore_file']) == ignore['ignore_sha256']
    frozen = [read_raw(row) for row in comparison['frozen_rows']]
    readout = [read_raw(row) for row in comparison['readout_rows']]
    manifest = read(C / 'evaluation_manifest.json')['scenes']
    expected_ids = {row['scene_id'] for row in manifest}
    assert len(frozen) == len(readout) == len(expected_ids) == 50
    assert {row['scene_id'] for row in frozen} == {row['scene_id'] for row in readout} == expected_ids
    assert all(row['training_seed'] == 3103 for row in frozen + readout)
    for scene in manifest:
        original = Path(scene['frozen_directory'])
        assert sha(original / 'result.json') == scene['frozen_result_sha256']
        assert sha(original / 'exit_receipt.json') == scene['frozen_exit_sha256']
        directory = C / 'rollouts' / f'readout60000_{scene["scene_id"][:20]}'
        assert read(directory / 'initial_pairing.json')['passed']
        result = read(directory / 'result.json')
        assert result['checkpoint_sha256'] == pair['policy_sha256']
        assert result['surface_encoder_sha256'] == pair['encoder_sha256']
        info = result['policy_info']
        assert info['control_steps'] == info['proximity_projection_calls'] == info['consecutive_proximity_history_frames'] == 900
        assert info['proximity_feature_dim'] == 128 and info['averaging_history'] == 100
    summaries = {'frozen': summary(frozen), 'readout': summary(readout)}
    assert summaries['frozen']['success'] == 20 and summaries['frozen']['cfts'] == 16
    for ours, side in [('frozen', 'control'), ('readout', 'candidate')]:
        for key in ('n', 'success', 'cfts', 'pickup', 'union_frames', 'physics_samples', 'avoidance_percent', 'failure_stages'):
            assert summaries[ours][key] == comparison['comparison'][side][key], (ours, key)
    old = {row['scene_id']: row for row in frozen}
    new = {row['scene_id']: row for row in readout}
    paired = {key: {'candidate_only': sum(new[s][key] and not old[s][key] for s in expected_ids),
                    'control_only': sum(old[s][key] and not new[s][key] for s in expected_ids)}
              for key in ('task_success', 'collision_free_task_success', 'pickup_failure')}
    assert paired == comparison['comparison']['paired']
    records = [json.loads(line) for line in (C / 'valid_ledger.jsonl').read_text().splitlines()]
    assert len(records) == len({row['id'] for row in records}) == 53
    assert sum(row['kind'] == 'evaluation' for row in records) == 51
    assert sum(row['kind'] == 'training' for row in records) == 2
    for record in records:
        receipt = read(record['receipt'])
        assert receipt['observed_by_parent'] and receipt['returncode'] == 0
        assert sha(record['receipt']) == record['receipt_sha256']
        assert sha(record['result_path']) == record['result_sha256']
    failures = [json.loads(line) for line in (C / 'invalid_ledger.jsonl').read_text().splitlines()]
    result = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'independently_read_raw_rows': 100, 'all_primary_and_pickup_lift_endpoints_exact': True,
        'all_raw_action_decoders_and_horizons_passed': True, 'all_initial_pairings_passed': True,
        'all_original_source_input_and_upstream_copy_hashes_match': True,
        'all_used_operational_recovery_source_hashes_match': True,
        'same_update_policy_encoder_pair_verified': 60000, 'upstream_byte_identical_files': len(upstream['files']),
        'valid_observed_exit_records': len(records), 'infrastructure_failures': failures,
        'summaries': summaries, 'paired': paired,
        'supervisor_exit_observed_by_root': {'session_id': parent_session, 'exit_code': 0},
        'source_sha256': sha(__file__), 'comparison_sha256': sha(C / 'comparison.json')}
    (C / 'root_review/final_review.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--observed-successful-parent-session', type=int, required=True)
    main(parser.parse_args().observed_successful_parent_session)
