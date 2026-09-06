import json
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'submodules/act')]
from pact_workflow import digest, grouped_split, groups_for, load_contract, validate_split
from pact_eval_protocol import rollout, summarize


def episodes():
    result = [{'id': i, 'cell': str(i % 4), 'selected_seed': i,
               'layout_sha256': str(i), 'trajectory_sha256': f't{i}'} for i in range(40)]
    result[4]['selected_seed'] = 0
    result[8]['layout_sha256'] = result[4]['layout_sha256']
    return result


def test_split_groups_transitively_covers_categories_and_is_repeatable():
    eps = episodes()
    assert [0, 4, 8] in groups_for(eps)
    split = grouped_split(eps)
    assert split == grouped_split(eps)
    for name in ('train', 'val'):
        assert {eps[i]['cell'] for i in split[name]} == {'0', '1', '2', '3'}
    assert set([0, 4, 8]).issubset(split['train']) or set([0, 4, 8]).issubset(split['val'])
    bad = dict(split, train=list(range(32)), val=list(range(32, 40)))
    bad['train'].remove(8); bad['val'].append(8)
    with pytest.raises(ValueError, match='repeated scene'):
        validate_split(eps, bad)


def test_contract_rejects_tampering(tmp_path):
    payload = {'schema_version': 'pact_experiment_v1', 'episodes': episodes(),
               'split': grouped_split(episodes())}
    payload['sha256'] = digest(payload)
    p = tmp_path / 'experiment.json'
    p.write_text(json.dumps(payload))
    assert load_contract(p) == payload
    payload['split']['seed'] += 1
    p.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='hash mismatch'):
        load_contract(p)


def test_normalization_excludes_extreme_validation_values(tmp_path):
    import h5py
    from utils import get_norm_stats
    for i, value in enumerate((1., 3., 1000.)):
        with h5py.File(tmp_path / f'episode_{i}.hdf5', 'w') as f:
            f.create_dataset('observations/qpos', data=np.full((3, 9), value, np.float32))
            f.create_dataset('action', data=np.full((3, 8), value, np.float32))
    stats = get_norm_stats(tmp_path, 3, episode_ids=[0, 1])
    np.testing.assert_allclose(stats['action_mean'], 2.)
    np.testing.assert_allclose(stats['qpos_mean'], 2.)


class Policy:
    def reset(self):
        self.resets = getattr(self, 'resets', 0) + 1
    def get_action(self, observation):
        return {'arm': np.zeros(7), 'gripper': [0.]}


class Task:
    def __init__(self, successes, early=False):
        self.values = successes
        self.early = early
    def reset(self):
        self.success_cache = []
        self.observation_cache = []
        self.count = 0
        return [{}], [{}]
    def judge_success(self):
        return False
    def step(self, action):
        self.success_cache.append([self.values[self.count]])
        self.observation_cache.append([{'image': object()}])
        self.count += 1
        return [{}], [0.], [self.early], [self.count == len(self.values)], [{}]


def test_success_ever_keeps_scoring_safety_after_success():
    task, policy = Task([False, True, False, False]), Policy()
    record = rollout(task, policy, 4)
    assert record['success'] and not record['terminal_success']
    assert record['first_success_step'] == 2 and record['steps'] == 4
    assert policy.resets == 1
    assert all(frame == [{}] for frame in task.observation_cache)


def test_false_positive_reset_and_early_termination_are_errors():
    task = Task([False] * 4, early=True)
    with pytest.raises(ValueError, match='Unexpected termination'):
        rollout(task, Policy(), 4)
    task.judge_success = lambda: True
    with pytest.raises(ValueError, match='already satisfies'):
        rollout(task, Policy(), 4)


def test_missing_actions_and_worker_failures_cannot_shrink_denominator():
    policy = Policy()
    policy.get_action = lambda obs: None
    with pytest.raises(ValueError, match='Invalid policy action'):
        rollout(Task([False] * 4), policy, 4)
    record = {'status': 'complete', 'success': True, 'collision_free': True}
    partial = summarize([record, {'status': 'error'}], 2)
    assert not partial['complete'] and partial['success_rate'] is None
    complete = summarize([record, dict(record, success=False)], 2)
    assert complete['complete'] and complete['success_rate'] == .5


def test_trace_parity_requires_inputs_actions_judges_and_contacts():
    from copy import deepcopy
    from eval_pact import compare
    record = {'status': 'complete', 'policy_input_hashes': ['abc'],
              'trace': [{'step': 0, 'arm': [0.] * 7, 'gripper': [0.], 'qpos': [0.] * 9, 'success': False}],
              'success': False, 'terminal_success': False, 'contact_audit': {'collision_free': True}}
    assert compare(record, deepcopy(record))
    for key, changed in [('policy_input_hashes', ['different']), ('contact_audit', {'collision_free': False}),
                         ('status', 'error'), ('trace', [])]:
        other = deepcopy(record); other[key] = changed
        assert not compare(record, other)
    other = deepcopy(record); other['trace'][0]['arm'][0] = .01
    assert not compare(record, other)


def test_input_diagnostics_locate_sensor_without_weakening_hash_comparison():
    from eval_pact import observation_hashes, comparison_report
    from copy import deepcopy
    obs = {'qpos': {'arm': [0.] * 7, 'gripper': [0., 0.]},
           'wrist_camera': np.zeros((2, 2, 3), np.uint8),
           'link1_sensor_0': np.zeros((8, 8), np.float32)}
    combined, components = observation_hashes(obs, list(obs))
    record = {'status': 'complete', 'policy_input_hashes': [combined],
              'policy_input_components': [{'step': 0, 'hashes': components}],
              'trace': [{'step': 0, 'arm': [0.] * 7, 'gripper': [0.], 'qpos': [0.] * 9, 'success': False}],
              'success': False, 'terminal_success': False, 'contact_audit': {}}
    other = deepcopy(record)
    obs['link1_sensor_0'][0, 0] = .01
    changed, changed_components = observation_hashes(obs, list(obs))
    other['policy_input_hashes'] = [changed]
    other['policy_input_components'][0]['hashes'] = changed_components
    report = comparison_report(record, other)
    assert not report['passed']
    assert report['first_component_mismatch']['fields'] == ['link1_sensor_0']
    assert report['success_flags_equal'] and report['contact_audit_equal']
    assert report['numeric_differences']['arm']['max_absolute_difference'] == 0
    del other['policy_input_components']
    assert not comparison_report(record, other)['component_diagnostics_available']


@pytest.mark.parametrize('mismatch', [True, False])
def test_verification_stops_at_failed_pair_and_requires_all_planned_pairs(tmp_path, monkeypatch, mismatch):
    import eval_pact
    import torch
    from types import SimpleNamespace
    args = SimpleNamespace(run_dir=tmp_path, checkpoint_dir=tmp_path,
                           checkpoint_name='policy_best.ckpt', suite='smoke',
                           reference=False, verify=True, worker=None)
    monkeypatch.setattr(eval_pact, 'parse_args', lambda: args)
    monkeypatch.setattr(eval_pact, 'load_contract', lambda path: {
        'profile': {'horizon': 1050}, 'evaluation': {'dev': [{'sha256': 'a'}, {'sha256': 'b'}]}})
    monkeypatch.setattr(eval_pact, 'identity', lambda args, contract: {'sha256': 'identity'})
    monkeypatch.setattr(torch, 'load', lambda *a, **k: {'model.query_embed.weight': np.zeros((50, 512))})
    calls = []
    def worker(command, **kwargs):
        calls.append(command)
        index = int(command[command.index('--worker') + 1])
        path = Path(command[command.index('--result') + 1])
        record = {'status': 'complete', 'identity': 'identity', 'row_sha256': 'ab'[index],
                  'policy_input_hashes': ['different' if mismatch and '--reference' not in command else 'same'],
                  'trace': [{'step': 0, 'arm': [0.] * 7, 'gripper': [0.], 'qpos': [0.] * 9, 'success': False}],
                  'success': False, 'terminal_success': False, 'contact_audit': {}}
        path.write_text(json.dumps(record))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(eval_pact.subprocess, 'run', worker)
    assert eval_pact.main() == (1 if mismatch else 0)
    report = json.loads((tmp_path / 'evaluation/identity/verification.json').read_text())
    assert report['planned_pairs'] == 2
    assert report['checked_pairs'] == (1 if mismatch else 2)
    assert report['passed'] is (not mismatch)
    assert len(calls) == (2 if mismatch else 4)


def test_wrapper_preserves_child_exit_code_without_traceback(monkeypatch, capsys):
    import runpy
    import subprocess
    monkeypatch.setattr(sys, 'argv', ['pact.py', 'setup', 'v12'])
    def failure(*args, **kwargs):
        raise subprocess.CalledProcessError(7, ['git'])
    monkeypatch.setattr(subprocess, 'check_output', failure)
    with pytest.raises(SystemExit) as result:
        runpy.run_path(str(ROOT / 'scripts/pact.py'), run_name='__main__')
    assert result.value.code == 7
    stderr = capsys.readouterr().err
    assert 'exit 7' in stderr and 'Traceback' not in stderr
