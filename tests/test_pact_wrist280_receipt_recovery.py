import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import pact_wrist280_receipt_recovery as recovery


def test_resource_query_survives_fork_unavailability(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, 'WORK', tmp_path)
    monkeypatch.setattr(recovery, 'CGROUP', tmp_path)
    monkeypatch.setattr(recovery, 'GPU_HANDLE', None)
    for name, value in {'memory.current': '70', 'memory.max': '100', 'pids.current': '30',
                        'pids.max': '100', 'cpu.max': '100 100', 'cpu.stat': 'usage_usec 1'}.items():
        (tmp_path / name).write_text(value)
    monkeypatch.setattr(recovery.pynvml, 'nvmlInit', Mock())
    monkeypatch.setattr(recovery.pynvml, 'nvmlDeviceGetHandleByIndex', lambda i: 'gpu0')
    get_memory = Mock(return_value=SimpleNamespace(used=20 * 2**20, total=100 * 2**20, reserved=2 * 2**20))
    monkeypatch.setattr(recovery.pynvml, 'nvmlDeviceGetMemoryInfo', get_memory)
    monkeypatch.setattr(recovery.pynvml, 'nvmlDeviceGetUtilizationRates', lambda h: SimpleNamespace(gpu=51))
    monkeypatch.setattr(__import__('subprocess'), 'Popen', Mock(side_effect=BlockingIOError('fork unavailable')))
    result = recovery.resources()
    assert result['vram_fraction'] == 0.2 and result['ram_fraction'] == 0.7
    assert result['pid_fraction'] == 0.3 and result['gpu_util_pct'] == 51
    get_memory.assert_called_once_with('gpu0', version=recovery.pynvml.nvmlMemory_v2)


def test_query_failure_still_records_completed_child_exit(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, 'RECOVERY', tmp_path / 'recovery')
    monkeypatch.setattr(recovery.capacity.recovery.original, 'WORK', tmp_path)
    parent_sample = Mock(side_effect=BlockingIOError('query temporarily unavailable'))
    monkeypatch.setattr(recovery.capacity.CapacityPool, 'sample', parent_sample)
    pool = recovery.ReceiptSafePool.__new__(recovery.ReceiptSafePool)
    pool.kind = 'final_3105_h100'
    pool.pause_reason = None
    job = {'directory': str(tmp_path / 'worker'), 'command': ['python', 'worker']}
    directory = tmp_path / 'worker';directory.mkdir()
    (directory / 'worker.log').write_text('complete\n')
    child = Mock();child.poll.return_value = 0
    stream = Mock()
    pool.active = {42: {'job': job, 'process': child, 'stream': stream,
                       'scan_offset': 0, 'started': recovery.time.time() - 1}}
    completed = pool.poll()
    assert len(completed) == 1 and completed[0][1]['returncode'] == 0
    assert recovery.read(directory / 'exit_receipt.json')['exit_evidence'] == 'subprocess.Popen.poll / waitpid'
    assert not pool.active and 'resource query failed' in pool.pause_reason
    stream.close.assert_called_once()
