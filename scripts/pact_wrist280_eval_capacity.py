"""Twelve-worker evaluation with scoped release of closed artifact file cache."""
from __future__ import annotations

import argparse
import time

from pact_wrist288_common import *
import pact_wrist280_resource_recovery as recovery

CAPACITY = WORK / 'amendments/eval_capacity_20260907'
STATUS = WORK / 'monitoring/eval_capacity_status.json'


def cache_candidates():
    """Only completed, experiment-owned HDF5 files; never model or asset files."""
    root = WORK.resolve()
    conversion = WORK / 'processes/conversion/exit_receipt.json'
    converted_complete = conversion.exists() and read(conversion)['returncode'] == 0
    for folder in ('raw', 'converted', 'evaluation'):
        for path in (WORK / folder).rglob('*.h5'):
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                continue
            if folder == 'converted':
                if not converted_complete:
                    continue
            elif not (path.parent / 'exit_receipt.json').exists():
                continue
            yield path


def release_closed_cache(reason):
    before = int(Path('/sys/fs/cgroup/memory.current').read_text())
    count = total = 0
    for path in cache_candidates():
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            total += os.fstat(fd).st_size
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
        count += 1
    result = {'utc': now(), 'reason': reason, 'files': count, 'file_bytes': total,
              'memory_before': before,
              'memory_after': int(Path('/sys/fs/cgroup/memory.current').read_text()),
              'operation': 'POSIX_FADV_DONTNEED; file contents unchanged'}
    append(CAPACITY / 'cache_releases.jsonl', result)
    return result


class CapacityPool(recovery.SettledPool):
    def __init__(self, kind, limit=14):
        # Use the original constructor to avoid recording a spurious 10-worker
        # fallback. Inherit the recovery's settled backoff behavior.
        self.rollout = kind.startswith(('smoke_', 'development_', 'final_'))
        recovery.BasePool.__init__(self, kind, 12 if self.rollout else limit)
        if self.rollout:
            release_closed_cache('before evaluation pool startup')
            self.record_capacity('user-requested 12-worker evaluation')

    def record_capacity(self, reason):
        atomic(STATUS, {'utc': now(), 'stage': self.kind, 'limit': self.limit,
                        'active_workers': len(self.active), 'reason': reason})
        append(WORK / 'monitoring/concurrency_changes.jsonl', {
            'utc': now(), 'stage': self.kind, 'after': self.limit,
            'reason': reason, 'paused': bool(self.pause_reason),
            'active_workers': len(self.active)})

    def sample(self, force=False):
        if self.rollout and (force or time.monotonic() >= self.next_sample):
            cg = Path('/sys/fs/cgroup')
            fraction = int((cg / 'memory.current').read_text()) / int((cg / 'memory.max').read_text())
            if fraction >= 0.73:
                release_closed_cache('minute sample at or above 73% RAM')
        return super().sample(force)

    def reduce(self, reason):
        super().reduce(reason)
        if self.rollout:
            self.record_capacity(reason)


def remaining_eta():
    bench_path = WORK / 'training_benchmark.json'
    if not bench_path.exists():
        return
    bench = read(bench_path)
    updates = sum(read(path)['updates'] for path in (WORK / 'checkpoints').glob('*/progress.json'))
    valid = [row for path in (WORK / 'evaluation').glob('*_ledger.jsonl')
             for row in lines(path) if row.get('valid_completion')]
    workers = read(STATUS)['limit'] if STATUS.exists() else 12
    assert workers in (10, 12)
    seconds_per_rollout = sum(row['elapsed_s'] for row in valid) / len(valid) if valid else 480
    estimate = (max(0, 360000 - updates) / bench['chosen_updates_per_second']
                + max(0, 404 - len(valid)) * seconds_per_rollout / workers)
    result = {'utc': now(), 'remaining_estimate_hours': estimate / 3600,
              'available_hours': (SAFE_STOP - time.time()) / 3600,
              'optimizer_rate': bench['chosen_updates_per_second'],
              'rollout_rate_measured': bool(valid), 'evaluation_workers': workers,
              'measured_rollout_mean_minutes': seconds_per_rollout / 60,
              'rollout_estimate_minutes_if_unmeasured': 8}
    append(WORK / 'monitoring/eta.jsonl', result)
    if estimate > SAFE_STOP - time.time():
        raise recovery.original.Paused(f'Measured remaining work no longer fits: {result}')
    return result


def verify_capacity():
    contract = read(CAPACITY / 'contract.json')
    assert contract['sha256'] == digest({k: v for k, v in contract.items() if k != 'sha256'})
    assert sha(recovery.RECOVERY / 'contract.json') == contract['recovery_contract_file_sha256']
    assert sha(CAPACITY / 'request.json') == contract['request_sha256']
    for path, expected in contract['file_hashes'].items():
        assert sha(ROOT / path) == expected, path
    return contract


def install():
    verify_capacity()
    data, train, monitor, analysis, run = recovery.install()
    run.Pool = CapacityPool
    run.remaining_eta = remaining_eta
    prior_audit = monitor.hourly_run_check

    def audit(final=False):
        try:
            return prior_audit(final=final)
        finally:
            state = read(WORK / 'state.json') if (WORK / 'state.json').exists() else {}
            if state.get('stage', '').startswith('evaluation_'):
                release_closed_cache('after hourly artifact validation')

    monitor.hourly_run_check = run.hourly_run_check = audit
    prior_report = analysis.report

    def report(verdict, reason, rows=None):
        prior_report(verdict, reason, rows)
        path = WORK / 'EVAL.md'
        with path.open('a') as stream:
            stream.write('\nEvaluation capacity amendment: user requested 12 or 14 workers. '
                         'Target is 12; measured VRAM excludes 14 under the original 85% guard. '
                         'Release clean file cache for completed experiment-owned HDF5 artifacts '
                         'before evaluation startup, after audits, and at minute samples above 73% RAM. '
                         'The 80% RAM guard and settled 12 -> 10 backoff remain active. '
                         'See `amendments/eval_capacity_20260907/contract.json`.\n')

    analysis.report = report
    return data, train, monitor, analysis, run


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('supervisor', 'monitor'), required=True)
    args = parser.parse_args()
    os.environ.update(environment())
    installed = install()
    installed[2 if args.mode == 'monitor' else -1].main()
