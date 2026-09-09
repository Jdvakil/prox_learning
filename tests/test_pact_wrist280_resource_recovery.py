import sys
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import pact_wrist280_resource_recovery as recovery


def test_pressure_waits_for_actual_pool_to_drain(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, 'WORK', tmp_path)
    monkeypatch.setattr(recovery.original, 'WORK', tmp_path)
    pool = recovery.SettledPool('development_h100')
    pool.limit = 14
    pool.active = {i: object() for i in range(14)}
    reason = 'three consecutive resource pressure samples'
    pool.reduce(reason)
    assert pool.limit == 12
    for _ in range(10):
        pool.reduce(reason)
    assert pool.limit == 12 and pool.pause_reason is None and len(pool.active) == 14
    pool.active = {i: object() for i in range(12)}
    pool.reduce(reason)
    assert pool.limit == 10
    pool.reduce(reason)
    assert pool.pause_reason is None
    pool.active = {i: object() for i in range(10)}
    pool.reduce(reason)
    assert pool.pause_reason == reason


def test_training_and_immediate_errors_keep_existing_behavior(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, 'WORK', tmp_path)
    monkeypatch.setattr(recovery.original, 'WORK', tmp_path)
    assert recovery.SettledPool('train_budget_60000', 1).limit == 1
    assert recovery.SettledPool('conversion', 1).limit == 1
    for name in ('smoke_h100', 'development_h10', 'final_3103_h100'):
        assert recovery.SettledPool(name).limit == 10
    pool = recovery.SettledPool('development_h100')
    pool.limit = 12
    pool.active = {i: object() for i in range(14)}
    pool.reduce('OOM/thread creation error; launches paused for diagnosis')
    assert pool.limit == 10


def test_resume_adopts_existing_benchmark_and_checks_seed_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, 'WORK', tmp_path)
    monkeypatch.setattr(recovery, 'verify_recovery', lambda: {})
    run = Mock()
    run.remaining_eta = recovery.original.remaining_eta
    analysis = Mock()
    monkeypatch.setattr(recovery.amendment, 'install', lambda: (None, None, None, analysis, run))
    monkeypatch.setattr(recovery.amendment, 'replace_function', lambda *args: None)
    pilot = run.train_pilot
    recovery.install()
    (tmp_path / 'training_benchmark.json').write_text('immutable benchmark')
    run.train_pilot()
    run.finish_seed.assert_called_once_with(3103)
    pilot.assert_not_called()
    assert (tmp_path / 'training_benchmark.json').read_text() == 'immutable benchmark'
