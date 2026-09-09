"""Process-free resource sampling and bounded deferral before a child exists."""
import errno
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / 'scripts'))
import pact_v1010b_run as run
from pact_v1010b_contract import B, MIB, append, freeze, inside, now, read, sha
import pynvml

AMENDMENT = Path(__file__).resolve().parent
_device = None


def resources():
    global _device
    if _device is None:
        pynvml.nvmlInit()
        _device = pynvml.nvmlDeviceGetHandleByIndex(0)
    # v1 includes driver-reserved memory. v2 used excludes it, matching nvidia-smi.
    memory = pynvml.nvmlDeviceGetMemoryInfo(_device, version=pynvml.nvmlMemory_v2)
    utilization = pynvml.nvmlDeviceGetUtilizationRates(_device)
    used, total = math.ceil(memory.used / MIB), memory.total / MIB
    cg = Path('/sys/fs/cgroup')
    values = {k: int((cg / k).read_text()) for k in
              ('memory.current', 'memory.max', 'pids.current', 'pids.max')}
    events = dict(line.split() for line in (cg / 'memory.events').read_text().splitlines())
    stat = dict(line.split() for line in (cg / 'memory.stat').read_text().splitlines())
    return {'utc': now(), 'ram_fraction': values['memory.current'] / values['memory.max'],
            'pid_fraction': values['pids.current'] / values['pids.max'],
            'vram_fraction': used / total, 'vram_used_mib': used, 'vram_total_mib': total,
            'gpu_utilization': float(utilization.gpu), 'cgroup': values,
            'oom_kill': int(events['oom_kill']), 'memory_file_bytes': int(stat['file']),
            'memory_anon_bytes': int(stat['anon']), 'disk_free_bytes': shutil.disk_usage(B).free,
            'gpu_query': 'NVML memory v2, in process; used rounded upward to MiB',
            'gpu_used_bytes': int(memory.used), 'gpu_reserved_bytes': int(memory.reserved),
            'pids_events_max': int((cg / 'pids.events').read_text().split()[1])}


def launch_pressure():
    cg = Path('/sys/fs/cgroup')
    return (int((cg / 'pids.current').read_text()) / int((cg / 'pids.max').read_text()) >= .75
            or int((cg / 'memory.current').read_text()) / int((cg / 'memory.max').read_text()) >= .80)


class DeferredStart(RuntimeError):
    pass


class RecoveryPool(run.Pool):
    def __init__(self, stage, workers=12):
        super().__init__(stage, min(workers, 10))

    def launch(self, job, command, kind):
        directory = inside(job['output_dir'])
        directory.mkdir(parents=True, exist_ok=True)
        assert not (directory / 'job.json').exists(), 'unresolved prior attempt'
        freeze(directory / 'job.json', job)
        log = (directory / 'worker.log').open('x')
        try:
            proc = subprocess.Popen(command, cwd=run.ROOT, env=run.environment(), stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
        except OSError as exc:
            log.close()
            if exc.errno != errno.EAGAIN:
                raise
            archive = AMENDMENT / 'deferred_process_starts' / job['job_id'] / str(time.time_ns())
            archive.mkdir(parents=True)
            for name in ('job.json', 'worker.log'):
                (directory / name).rename(archive / name)
            freeze(archive / 'deferred.json', {'utc': now(), 'job_id': job['job_id'],
                   'error': repr(exc), 'child_started': False, 'scientific_attempt_started': False,
                   'resources': resources()})
            raise DeferredStart(str(exc)) from exc
        start = {'pid': proc.pid, 'process_start_ticks': run.process_start(proc.pid),
                 'command': command, 'utc': now(), 'job_sha256': job['job_sha256'],
                 'parent_pid': os.getpid(), 'parent_start_ticks': run.process_start(os.getpid()),
                 'operational_amendment_sha256': sha(AMENDMENT / 'amendment.json')}
        self.active[proc.pid] = {'process': proc, 'job': job, 'log': log,
                                'started': time.monotonic(), 'launch': start, 'kind': kind}
        freeze(directory / 'launch.json', start)

    def _execute(self, jobs, kind='rollout'):
        completed = {r['job_id'] for r in run.lines(run.B / 'valid_ledger.jsonl')}
        pending = [j for j in jobs if j['job_id'] not in completed]
        retry_after = 0.
        consecutive_deferrals = 0
        while pending or self.active:
            self.poll()
            self.sample()
            estimate = (max(self.durations) * 1.20 if self.durations else
                        (1237 * 1.2 if kind == 'rollout' else 900))
            if time.time() + estimate > self.safe_stop:
                self.paused = 'projected next completion crosses47h launch boundary'
            if self.paused:
                if not self.active:
                    raise run.StopExecution(self.paused)
            elif time.monotonic() >= retry_after:
                while pending and len(self.active) < self.limit:
                    # The unchanged minute sampler still enforces the three-sample stop.
                    if launch_pressure():
                        break
                    job = pending[0]
                    command = run.worker_command(job) if kind == 'rollout' else job['command']
                    try:
                        self.launch(job, command, kind)
                    except DeferredStart:
                        consecutive_deferrals += 1
                        if consecutive_deferrals >= 3:
                            self.paused = 'three consecutive EAGAIN launch deferrals; no child started'
                        retry_after = time.monotonic() + 60
                        break
                    pending.pop(0)
                    consecutive_deferrals = 0
            if time.time() > self.deadline:
                raise run.StopExecution('hard48h deadline reached; drain owned process groups')
            if pending or self.active:
                time.sleep(2)
        return jobs


def install():
    run.resources = resources
    run.Pool = RecoveryPool

