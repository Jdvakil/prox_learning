"""Sample GPU without forking and preserve exit collection through query errors."""
from __future__ import annotations
import argparse
import math
import shutil
import time
import pynvml
from pact_wrist288_common import *
import pact_wrist280_final_cache_recovery as cache_recovery

RECOVERY = WORK / 'amendments/receipt_recovery_20260907'
CGROUP = Path('/sys/fs/cgroup')
GPU_HANDLE = None
capacity = cache_recovery.capacity


def resources():
    global GPU_HANDLE
    if GPU_HANDLE is None:
        pynvml.nvmlInit()
        GPU_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
    values = {k: (CGROUP / k).read_text().strip() for k in
              ('memory.current', 'memory.max', 'pids.current', 'pids.max', 'cpu.max', 'cpu.stat')}
    # v2 excludes driver-reserved memory from "used", matching nvidia-smi.
    memory = pynvml.nvmlDeviceGetMemoryInfo(GPU_HANDLE, version=pynvml.nvmlMemory_v2)
    utilization = pynvml.nvmlDeviceGetUtilizationRates(GPU_HANDLE)
    used, total = math.ceil(memory.used / 2**20), math.ceil(memory.total / 2**20)
    return {'utc': now(), 'gpu_used_mib': used, 'gpu_total_mib': total,
            'gpu_reserved_mib': math.ceil(memory.reserved / 2**20),
            'gpu_util_pct': int(utilization.gpu), 'vram_fraction': used / total,
            'ram_fraction': int(values['memory.current']) / int(values['memory.max']),
            'pid_fraction': int(values['pids.current']) / int(values['pids.max']),
            'disk_free_gib': shutil.disk_usage(WORK).free / 2**30, 'cgroup': values,
            'gpu_query_method': 'in-process NVML memory v2; no subprocess'}


class ReceiptSafePool(capacity.CapacityPool):
    def sample(self, force=False):
        try:
            return super().sample(force)
        except (OSError, pynvml.NVMLError) as error:
            # A resource-query failure must not prevent Popen.poll/waitpid in
            # the rest of poll(), including while draining healthy jobs.
            self.pause_reason = f'resource query failed: {type(error).__name__}: {error}'
            self.next_sample = time.monotonic() + 60
            append(RECOVERY / 'resource_errors.jsonl', {'utc': now(), 'stage': self.kind,
                   'reason': self.pause_reason, 'active_workers': len(self.active),
                   'new_launches_paused': True, 'exit_collection_continues': True})

    def launch(self, job):
        try:
            return super().launch(job)
        except BlockingIOError as error:
            self.pause_reason = f'worker launch failed before process creation: {error}'
            directory = Path(job['directory'])
            assert not (directory / 'launch.json').exists()
            log = directory / 'worker.log'
            destination = RECOVERY / 'unlaunched_logs' / f'{directory.name}_{time.time_ns()}.log'
            destination.parent.mkdir(parents=True, exist_ok=True)
            if log.exists():
                shutil.move(log, destination)
            append(RECOVERY / 'launch_errors.jsonl', {'utc': now(), 'job': job,
                   'error': self.pause_reason, 'child_process_created': False})


def install():
    contract = read(RECOVERY / 'contract.json')
    assert contract['sha256'] == digest({k: v for k, v in contract.items() if k != 'sha256'})
    assert sha(cache_recovery.RECOVERY / 'contract.json') == contract['cache_recovery_contract_file_sha256']
    assert sha(RECOVERY / 'quarantine.json') == contract['quarantine_sha256']
    assert sha(pynvml.__file__) == contract['pynvml_sha256']
    for path, expected in contract['file_hashes'].items():
        assert sha(ROOT / path) == expected, path
    installed = cache_recovery.install()
    monitor, analysis, run = installed[2:]
    monitor.resources = run.resources = resources
    run.Pool = ReceiptSafePool
    prior_report = analysis.report

    def report(verdict, reason, rows=None):
        prior_report(verdict, reason, rows)
        with (WORK / 'EVAL.md').open('a') as stream:
            stream.write('\nInfrastructure retry: system thread/memory pressure prevented the '
                         'old supervisor and monitor from spawning nvidia-smi. Twelve finished-looking '
                         'rollouts lacked parent-observed exit receipts and were excluded uniformly. '
                         'Their original files were quarantined and the same six frozen pairs rerun; '
                         'no policy outcome determined retry selection. Eight additional rollouts '
                         'were still unstarted. The intended cohort remains 50 pairs. '
                         'GPU queries now use in-process NVML, and '
                         'resource-query errors pause launches while preserving worker exit collection. '
                         'See `amendments/receipt_recovery_20260907/quarantine.json`.\n')

    analysis.report = report
    return installed


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('supervisor', 'monitor'), required=True)
    args = parser.parse_args()
    os.environ.update(environment())
    installed = install()
    if args.mode == 'monitor':
        installed[2].main()
    else:
        cache_recovery.followup.supervise(installed)
