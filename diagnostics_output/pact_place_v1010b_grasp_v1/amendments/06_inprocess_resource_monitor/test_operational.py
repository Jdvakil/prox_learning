import errno
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import operational as op


def test_actual_gpu_monitor_does_not_spawn(monkeypatch):
    expected = op.run.resources()
    def forbidden(*args, **kwargs):
        raise AssertionError('resource monitor attempted to spawn a process')
    monkeypatch.setattr(subprocess, 'Popen', forbidden)
    actual = op.resources()
    assert actual['vram_total_mib'] == expected['vram_total_mib']
    assert abs(actual['vram_used_mib'] - expected['vram_used_mib']) <= 1
    assert 0 <= actual['gpu_utilization'] <= 100
    assert actual['gpu_reserved_bytes'] > 0
    assert actual['cgroup']['pids.max'] == expected['cgroup']['pids.max']


def test_egain_preserves_prelaunch_files_and_same_job_can_start(monkeypatch):
    import tempfile
    with tempfile.TemporaryDirectory(dir=op.B, prefix='monitor_test_') as directory:
        root = Path(directory)
        (root / 'amendment.json').write_text('{}')
        monkeypatch.setattr(op, 'AMENDMENT', root)
        pool = op.RecoveryPool.__new__(op.RecoveryPool)
        pool.active = {}
        job = {'job_id': 'fixture', 'job_sha256': 'fixture_hash', 'output_dir': str(root / 'job')}
        original = subprocess.Popen
        def exhausted(*args, **kwargs):
            raise BlockingIOError(errno.EAGAIN, 'test process limit')
        monkeypatch.setattr(subprocess, 'Popen', exhausted)
        with pytest.raises(op.DeferredStart):
            pool.launch(job, [sys.executable, '-c', 'pass'], 'rollout')
        assert not pool.active and not (root / 'job/job.json').exists()
        archives = list((root / 'deferred_process_starts').glob('fixture/*'))
        assert len(archives) == 1
        assert json.loads((archives[0] / 'job.json').read_text()) == job
        assert not json.loads((archives[0] / 'deferred.json').read_text())['child_started']
        monkeypatch.setattr(subprocess, 'Popen', original)
        pool.launch(job, [sys.executable, '-c', 'pass'], 'rollout')
        assert len(pool.active) == 1 and (root / 'job/launch.json').exists()
        item = next(iter(pool.active.values()))
        try:
            assert item['process'].wait(timeout=10) == 0
        finally:
            if item['process'].poll() is None:
                item['process'].kill()
                item['process'].wait()
            item['log'].close()


def test_pressure_thresholds_unchanged(monkeypatch):
    values = {'pids.current': 74, 'pids.max': 100, 'memory.current': 79, 'memory.max': 100}
    class FakePath:
        def __init__(self, key=''):
            self.key = key
        def __truediv__(self, key):
            return FakePath(key)
        def read_text(self):
            return str(values[self.key])
    monkeypatch.setattr(op, 'Path', FakePath)
    assert not op.launch_pressure()
    values['pids.current'] = 75
    assert op.launch_pressure()
    values['pids.current'] = 74
    values['memory.current'] = 80
    assert op.launch_pressure()


def test_deferred_launch_keeps_polling_and_retains_pending_identity(monkeypatch):
    clock = SimpleNamespace(value=0.)
    def sleep(seconds):
        clock.value += seconds
    monkeypatch.setattr(op, 'time', SimpleNamespace(time=lambda: clock.value,
                       monotonic=lambda: clock.value, sleep=sleep))
    monkeypatch.setattr(op.run, 'lines', lambda path: [])
    monkeypatch.setattr(op.run, 'worker_command', lambda job: ['fixture'])
    monkeypatch.setattr(op, 'launch_pressure', lambda: False)
    pool = op.RecoveryPool.__new__(op.RecoveryPool)
    pool.active, pool.durations, pool.limit, pool.paused = {}, [], 1, None
    pool.safe_stop, pool.deadline = 10000, 11000
    counts = {'poll': 0, 'launch': 0}
    def poll():
        counts['poll'] += 1
        pool.active.clear()
    identities = []
    def launch(job, command, kind):
        identities.append(job['job_id'])
        counts['launch'] += 1
        if counts['launch'] == 1:
            raise op.DeferredStart('fixture pressure')
        pool.active['fixture'] = {}
    pool.poll, pool.sample, pool.launch = poll, lambda: None, launch
    jobs = [{'job_id': 'same_job'}]
    assert pool._execute(jobs) == jobs
    assert identities == ['same_job', 'same_job'] and counts['poll'] >= 31

