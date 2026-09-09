"""Operational tests for the collection-only concurrency amendment."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import pact_wrist288_capacity as capacity


def full_samples(target=14, ram=.69, pss_gib=6, at=None):
    at = capacity.HOLD_UNTIL + 3600 if at is None else at
    return [dict(epoch=at - (14 - i) * 60, target=target, active=target,
                 ram_fraction=ram, vram_fraction=.20, pid_fraction=.13,
                 worker_tree_pss_max_bytes=pss_gib * 2**30,
                 cgroup={'memory.max': str(171 * 2**30)}, disk_free_gib=100)
            for i in range(15)]


def test_hold_boundary_and_stepwise_promotion():
    at = capacity.HOLD_UNTIL - 1
    assert capacity.capacity_decision(full_samples(at=at), 14, at)['proposed'] == 14
    at = capacity.HOLD_UNTIL
    assert capacity.capacity_decision(full_samples(at=at), 14, at)['proposed'] == 16
    assert capacity.capacity_decision(full_samples(target=16, at=at), 16, at)['proposed'] == 18
    assert capacity.capacity_decision(full_samples(target=18, at=at), 18, at)['proposed'] == 18


def test_transient_memory_peak_blocks_increase():
    samples = full_samples(ram=.60)
    samples[3]['ram_fraction'] = .73
    result = capacity.capacity_decision(samples, 14, samples[-1]['epoch'])
    assert result['proposed'] == 14
    assert result['predicted']['ram_fraction'] > .79


@pytest.mark.parametrize('key,value', [('vram_fraction', .8), ('pid_fraction', .7), ('disk_free_gib', 11)])
def test_other_resource_limits_block_promotion(key, value):
    samples = full_samples()
    samples[-2][key] = value
    assert capacity.capacity_decision(samples, 14, samples[-1]['epoch'])['proposed'] == 14


def test_partial_pools_and_settling_period_do_not_justify_promotion():
    samples = full_samples()
    at = samples[-1]['epoch']
    assert capacity.capacity_decision(samples, 14, at, last_change=at - 899)['proposed'] == 14
    for sample in samples[:6]:
        sample['active'] = 13
    assert capacity.capacity_decision(samples, 14, at)['proposed'] == 14


def test_worker_tree_transient_allowance_in_capacity_estimate():
    samples = full_samples(pss_gib=9)
    result = capacity.capacity_decision(samples, 14, samples[-1]['epoch'])
    assert result['proposed'] == 14
    assert result['extra_worker_bytes'] == pytest.approx(9 * 1.1 * 2**30)


def make_pool(tmp_path, kind='collection', limit=18):
    # Bypass only immutable production setup, retaining actual Pool state/methods.
    pool = capacity.CollectionPool.__new__(capacity.CollectionPool)
    capacity.original.Pool.__init__(pool, kind, limit)
    pool.capacity_samples = []
    pool.last_capacity_change = 0
    return pool


def test_pressure_retires_capacity_without_killing_inflight(tmp_path, monkeypatch):
    monkeypatch.setattr(capacity, 'WORK', tmp_path)
    pool = make_pool(tmp_path)
    pool.active = {pid: object() for pid in range(18)}
    sample = full_samples(target=18, ram=.81)[-1]
    monkeypatch.setattr(capacity, 'snapshot', lambda *_: sample)
    for _ in range(2):
        pool.sample(force=True)
        assert pool.limit == 18
    pool.sample(force=True)
    assert pool.limit == 16
    assert len(pool.active) == 18
    assert not pool.can_launch()
    for expected in (14, 12, 10):
        pool.reduce('continued pressure')
        assert pool.limit == expected
    pool.reduce('continued pressure')
    assert pool.pause_reason == 'continued pressure'


def test_evaluation_uses_original_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(capacity.original, 'WORK', tmp_path)
    pool = make_pool(tmp_path, 'development', 14)
    pool.reduce('pressure')
    assert pool.limit == 12


def test_existing_pause_prevents_upsize(tmp_path, monkeypatch):
    monkeypatch.setattr(capacity, 'WORK', tmp_path)
    pool = make_pool(tmp_path, limit=14)
    sample = full_samples()[-1]
    pool.capacity_samples = full_samples()[:-1]
    monkeypatch.setattr(capacity, 'snapshot', lambda *_: sample)
    (tmp_path / 'PAUSE.json').write_text('{}')
    pool.sample(force=True)
    assert pool.limit == 14
    assert pool.pause_reason


def test_handoff_refuses_missing_or_failed_real_exit_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(capacity, 'WORK', tmp_path)
    monkeypatch.setattr(capacity, 'verify_amendment', lambda: {'amendment_sha256': 'a'})
    monkeypatch.setattr(capacity, 'resources', lambda: {'disk_free_gib': 100})
    monkeypatch.setattr(capacity, 'owned_processes', lambda: [])
    capacity.atomic(tmp_path / 'SAFE_SHUTDOWN.json', {'verdict': 'INCOMPLETE', 'reason': 'Paused: disk/deadline/manual guard'})
    capacity.atomic(tmp_path / 'PAUSE.json', {'reason': 'user_authorized_collection_capacity_handoff', 'amendment_sha256': 'a'})
    job = {'row': {'attempt_id': 'id'}}
    directory = tmp_path / 'raw/id'
    capacity.atomic(directory / 'job.json', job)
    with pytest.raises(FileNotFoundError):
        capacity.verify_drained()
    capacity.atomic(directory / 'exit_receipt.json', {'returncode': 1, 'job_sha256': capacity.digest(job)})
    with pytest.raises(AssertionError):
        capacity.verify_drained()
    receipt = {'returncode': 0, 'job_sha256': capacity.digest(job)}
    capacity.atomic(directory / 'exit_receipt.json', receipt)
    capacity.append(tmp_path / 'collection/ledger.jsonl', {'attempt_id': 'id', 'accepted': False,
                    'exit_receipt_sha256': capacity.sha(directory / 'exit_receipt.json')})
    (directory / 'worker.log').write_text('ordinary strict-clean rejection')
    assert capacity.verify_drained() == {'verified_attempts': 1, 'accepted': 0}
    (directory / 'worker.log').write_text('CUDA out of memory')
    with pytest.raises(AssertionError):
        capacity.verify_drained()


def test_process_identity_includes_kernel_start_and_rejects_missing_pid():
    import os
    assert capacity.process_identity(os.getpid()).isdigit()
    assert capacity.process_identity(999999999) is None


def test_handoff_archives_real_closed_supervisor_before_continuation(tmp_path, monkeypatch):
    import subprocess
    import time
    import json
    work = tmp_path / 'run'
    work.mkdir()
    (work / 'monitoring').mkdir()
    monkeypatch.setattr(capacity, 'WORK', work)
    monkeypatch.setattr(capacity, 'ROOT', tmp_path)
    monkeypatch.setattr(capacity, 'CODE', tmp_path)
    amendment = {'amendment_sha256': 'amendment', 'config_sha256': 'science'}
    monkeypatch.setattr(capacity, 'verify_amendment', lambda: amendment)
    monkeypatch.setattr(capacity, 'check_bindings', lambda: {})
    monkeypatch.setattr(capacity, 'owned_processes', lambda: [])
    monkeypatch.setattr(capacity, 'resources', lambda: {'disk_free_gib': 100,
        'vram_fraction': .1, 'pid_fraction': .1,
        'cgroup': {'memory.current': str(20 * 2**30), 'memory.max': str(171 * 2**30)}})
    real_sleep = time.sleep
    monkeypatch.setattr(capacity.time, 'sleep', lambda _: real_sleep(.01))
    old_code = '''import json, time, os
from pathlib import Path
w=Path('run')
while not (w/'PAUSE.json').exists(): time.sleep(.01)
# Model the durable closure emitted after the original parent reaps all jobs.
(w/'SAFE_SHUTDOWN.json').write_text(json.dumps({'verdict':'INCOMPLETE','reason':'Paused: disk/deadline/manual guard'}))
'''
    monitor_code = '''import time
from pathlib import Path
w=Path('run')
while not (w/'SAFE_SHUTDOWN.json').exists(): time.sleep(.01)
'''
    (tmp_path / 'pact_wrist288_monitor.py').write_text(monitor_code)
    continuation = tmp_path / 'continuation.py'
    continuation.write_text('''import json, os, time
from pathlib import Path
w=Path('run')
assert (w/'monitoring/capacity_handoff_archive/SAFE_SHUTDOWN.json').exists()
assert not (w/'PAUSE.json').exists()
(w/'state.json').write_text(json.dumps({'stage':'collection','supervisor_pid':os.getpid()}))
time.sleep(.1)
(w/'SAFE_SHUTDOWN.json').write_text(json.dumps({'verdict':'test continuation complete'}))
''')
    monkeypatch.setattr(capacity, '__file__', str(continuation))
    old = subprocess.Popen([sys.executable, '-c', old_code], cwd=tmp_path)
    monitor = subprocess.Popen([sys.executable, '-c', monitor_code], cwd=tmp_path)
    try:
        capacity.atomic(work / 'state.json', {'stage': 'collection', 'supervisor_pid': old.pid, 'started_utc': 'original-start'})
        capacity.atomic(work / 'monitoring/monitor.json', {'pid': monitor.pid})
        (work / 'supervisor.log').write_text('preserved original log')
        (work / 'EVAL.md').write_text('old report')
        decision = {'proposed': 16, 'extra_worker_bytes': 6 * 2**30}
        capacity.perform_handoff(decision, amendment)
        assert old.wait(timeout=2) == 0
        assert monitor.wait(timeout=2) == 0
        assert capacity.read(work / 'monitoring/capacity_handoff.json')['resume_target'] == 16
        assert (work / 'monitoring/capacity_handoff_archive/supervisor.log').read_text() == 'preserved original log'
        assert capacity.read(work / 'monitoring/capacity_supervisor_exit.json')['returncode'] == 0
        assert capacity.read(work / 'monitoring/capacity_status.json')['phase'] == 'experiment closed'
    finally:
        for child in (old, monitor):
            if child.poll() is None:
                child.terminate()
            child.wait(timeout=2)
