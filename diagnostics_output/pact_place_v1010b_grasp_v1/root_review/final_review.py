"""Root closeout check: read raw Stage B endpoints without the experiment metrics module."""
import collections
import datetime
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

B = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def decode(value):
    return json.loads(bytes(value).split(b'\0', 1)[0])


def main():
    metrics = read(B / 'metrics/B.json')
    gate = read(B / 'gates/B.json')
    rows = []
    for saved in metrics['rows']:
        directory = Path(saved['directory'])
        with h5py.File(directory / 'trajectory.h5') as h:
            trajectory = h['traj_0']
            success_trace = trajectory['success'][()]
            assert success_trace.shape == (901,)
            task = [decode(x) for x in trajectory['obs/extra/task_info'][()]]
            grasp = [decode(x)['gripper'] for x in trajectory['obs/extra/grasp_state_pickup_obj'][()]]
            assert len(task) == len(grasp) == 901
            assert np.array_equal(success_trace, [x['success'] for x in task])
            success = bool(success_trace[-1])
            support = any(x['supported_by_receptacle'] for x in task)
            held = any(x['held'] for x in grasp)
            touch = any(x['touching'] for x in grasp)
        with h5py.File(directory / 'telemetry.h5') as h:
            classes = json.loads(h['contacts'].attrs['class_names'])
            contacts = h['contacts/class_entries'][()]
            times = h['contacts/sim_time_s'][()]
            assert contacts.shape[0] == len(times) == 29701
            assert np.all(np.diff(times) > 0)
            target = h['target/control_step_and_world_xyz'][()]
            assert target.shape == (901, 4)
            lift = float(np.max(target[:, 3] - target[0, 3]))
            union = np.any(contacts[:, [classes.index(c) for c in ('hazard_bar', 'clutter')]] > 0, axis=1)
            collision_free = not np.any(contacts[:, [classes.index(c) for c in
                ('hazard_bar', 'clutter', 'other_environment', 'mounted_fixture')]] > 0)
        category = ('success' if success else 'supported_without_final_success' if support else
                    'lifted_without_placement' if lift >= .01 else 'held_without_lift' if held else
                    'touched_without_hold' if touch else 'no_target_interaction')
        row = {k: saved[k] for k in ('arm', 'training_seed', 'variant', 'scene_id')}
        row.update(task_success=success, collision_free_task_success=bool(success and collision_free),
                   failure_stage=category, pickup_failure=category == 'touched_without_hold',
                   union_frames=int(union.sum()), physics_samples=len(times))
        for key in ('task_success', 'collision_free_task_success', 'failure_stage',
                    'pickup_failure', 'union_frames', 'physics_samples'):
            assert row[key] == saved[key], (directory, key)
        rows.append(row)
    assert len(rows) == 216
    scene_ids = {r['scene_id'] for r in rows}
    expected = {(arm, seed, variant, scene) for arm in ('ACT', 'PACT')
                for seed in (3103, 3104, 3105)
                for variant in ('frozen60000', 'uniform63000', 'acquisition63000')
                for scene in scene_ids}
    assert len(scene_ids) == 12 and len(expected) == len(rows)
    assert {(r['arm'], r['training_seed'], r['variant'], r['scene_id']) for r in rows} == expected
    summaries = {}
    for arm in ('ACT', 'PACT'):
        for variant in ('frozen60000', 'uniform63000', 'acquisition63000'):
            for scope in ('pooled', '3103', '3104', '3105'):
                selected = [r for r in rows if r['arm'] == arm and r['variant'] == variant
                            and (scope == 'pooled' or r['training_seed'] == int(scope))]
                summary = {'n': len(selected), 'success': sum(r['task_success'] for r in selected),
                           'cfts': sum(r['collision_free_task_success'] for r in selected),
                           'pickup': sum(r['pickup_failure'] for r in selected),
                           'union_frames': sum(r['union_frames'] for r in selected),
                           'physics_samples': sum(r['physics_samples'] for r in selected)}
                control = variant if variant != 'acquisition63000' else 'frozen60000'
                side = 'control' if variant != 'acquisition63000' else 'candidate'
                expected_summary = gate['details'][control][arm][scope][side]
                assert all(summary[k] == expected_summary[k] for k in summary)
                summaries[f'{arm}/{variant}/{scope}'] = summary
    closure = read(B / 'parent_closure.json')
    verification = read(B / 'final_verification.json')
    assert closure['status'] == 'STOPPED_AT_GATE_B' and gate['passed'] is False
    assert closure['new_rollouts_by_stage'] == {'A1': 12, 'A2': 48, 'B': 180, 'C': 0, 'D': 0}
    assert closure['completed_training_branches'] == 12 and closure['new_training_updates'] == 36000
    assert closure['all_recorded_exits_observed'] and closure['baseline_retained']
    assert not Path(f'/proc/{closure["parent_pid"]}').exists()
    assert verification['valid_rollouts'] == 240 and verification['raw_recomputation_passed']
    assert verification['historical_protected_hashes_unchanged'] and verification['user_eval_preserved']
    for model in closure['rollback_models'].values():
        assert sha(Path(model['path'])) == model['sha256']
    document = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'status': closure['status'], 'independent_raw_stage_B_rows': len(rows),
                'complete_crossed_matrix': True, 'all_saved_primary_endpoints_exact': True,
                'all_gate_aggregate_counts_exact': True, 'frozen_model_hashes_unchanged': True,
                'supervisor_exit_observed_by_root': {'session_id': 87675, 'exit_code': 0},
                'source_sha256': sha(Path(__file__)),
                'artifact_hashes': {name: sha(B / name) for name in
                    ('metrics/B.json', 'gates/B.json', 'parent_closure.json', 'final_verification.json')},
                'summaries': summaries}
    (B / 'root_review/final_review.json').write_text(json.dumps(document, indent=2) + '\n')
    print(json.dumps({k: v for k, v in document.items() if k != 'summaries'}, indent=2))


if __name__ == '__main__':
    main()
