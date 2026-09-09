import collections
import copy
import sys
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import pact_wrist280_final50 as final50


def fixtures():
    manifests = {}
    for seed in final50.SEEDS:
        rows = []
        for family in range(4):
            for side in ('left', 'right'):
                for pose in ('center', 'neg5', 'pos5'):
                    for index in range(2):
                        identity = f'{seed}:{family}:{side}:{pose}:{index}'
                        rows.append({'cell': f'{family}|{side}|{pose}', 'intrusion_side': side,
                                     'pose_id': pose, 'episode_id': identity, 'row_sha256': identity})
        manifests[seed] = {'rows': rows}
    return manifests


def test_subset_balances_cells_and_seeds_without_mutating_source():
    source = fixtures()
    before = copy.deepcopy(source)
    selected = final50.select_subset(source)
    assert source == before
    assert {seed: len(rows) for seed, rows in selected.items()} == final50.COUNTS
    rows = [r for group in selected.values() for r in group]
    assert len({r['episode_id'] for r in rows}) == 50
    counts = collections.Counter(r['cell'] for r in rows)
    assert sorted(counts.values()) == [2] * 22 + [3] * 2
    extras = [r for r in rows if counts[r['cell']] == 3]
    assert {r['intrusion_side'] for r in extras} == {'left', 'right'}
    assert len({r['pose_id'] for r in extras}) == 2


def test_subset_is_independent_of_source_order_and_outcomes():
    source = fixtures()
    expected = final50.select_subset(source)
    for doc in source.values():
        doc['rows'].reverse()
        for r in doc['rows']:
            r['task_success'] = True  # The selector must never inspect this field.
    actual = final50.select_subset(source)
    assert {s: [r['episode_id'] for r in rows] for s, rows in actual.items()} == {
        s: [r['episode_id'] for r in rows] for s, rows in expected.items()}


def test_supervisor_proceeds_after_missed_gate_and_requires_all_seed_pairs(tmp_path, monkeypatch):
    monkeypatch.setattr(final50, 'WORK', tmp_path)
    monkeypatch.setattr(final50, 'remaining_eta', lambda: None)
    (tmp_path / 'supervisor.lock').touch()
    run = Mock()
    run.POOLS = []
    run.cleanup_orphans.return_value = []
    def evaluate(role, seed, history):
        return [{'seed': seed, 'episode_id': f'{seed}:{i}', 'arm': arm, 'task_success': False}
                for i in range(final50.COUNTS[seed]) for arm in ('ACT', 'PACT')]
    run.run_eval.side_effect = evaluate
    analysis = Mock()
    final50.supervise((None, None, None, analysis, run))
    assert [call.args[0] for call in run.finish_seed.call_args_list] == [3103, 3104, 3105]
    assert run.run_eval.call_count == 3
    report = analysis.report.call_args.args
    assert report[0] == 'REDUCED FINAL EVALUATION COMPLETE' and len(report[2]) == 100
    assert final50.read(tmp_path / 'SAFE_SHUTDOWN.json')['verdict'] == report[0]
