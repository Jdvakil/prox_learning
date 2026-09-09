"""Resume the untouched pending schedule after a parent monitoring fork failure."""
import fcntl
import os
import time
import traceback
from pathlib import Path
import operational

run = operational.run
am = Path(__file__).resolve().parent
B = run.B


def main():
    lock = (B / 'parent.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    amendment = run.read(am / 'amendment.json')
    for name, expected in amendment['code_hashes'].items():
        assert run.sha(am / name) == expected, name
    assert run.sha(am / 'preflight.json') == amendment['preflight_sha256']
    assert run.read(B / 'contract.json')['contract_sha256'] == amendment['contract_sha256']
    operational.install()
    run.verify_contract()
    records = run.validate_ledger()
    assert sum(r['kind'] == 'training' for r in records) == 12
    run.require_stage('B')
    run.remaining_time_gate('resume_monitor_recovery', 10)
    run.storage_gate('resume_monitor_recovery')
    run.atomic(B / 'run_status.json', {'utc': run.now(), 'status': 'RUNNING_MONITOR_RECOVERY',
               'active_parent': True, 'parent_pid': os.getpid(), 'eval_worker_cap': 10,
               'amendment_path': str(am / 'amendment.json'), 'progress_path': str(B / 'progress.json')})
    old = run.read(B / 'parent_closure.json')
    old.update(status='SUPERSEDED_BY_ACTIVE_MONITOR_RECOVERY', superseded_status=old['status'],
               superseded_closure_path=str(am / 'prior/parent_closure.json'))
    run.atomic(B / 'parent_closure.json', old)
    report = B / 'EVAL.md'
    report.write_text('> This prior closure is superseded by the active monitor recovery. '
                      'See [run_status.json](run_status.json).\n\n' + report.read_text())
    try:
        for stage in ('B', 'C', 'D'):
            run.evaluation(stage)
        run.verify()
        run.close('FINAL_ACCEPTED', 'All frozen gates passed.')
    except run.StopExecution as exc:
        run.verify()
        run.close(exc.status, str(exc))
    except BaseException as exc:
        run.atomic(B / f'errors/monitor_recovery_{time.time_ns()}.json',
                   {'utc': run.now(), 'error': repr(exc), 'traceback': traceback.format_exc()})
        run.close('INCOMPLETE_INFRASTRUCTURE_OR_BUDGET', repr(exc))
        raise
    finally:
        closure = run.read(B / 'parent_closure.json')
        closure.update(operational_amendment_sha256=run.sha(am / 'amendment.json'),
                       final_eval_worker_cap=10, infrastructure_retries_used=2,
                       parent_monitor_recoveries=1, uncommitted_optimizer_updates_replayed=240)
        run.atomic(B / 'parent_closure.json', closure)
        run.atomic(B / 'run_status.json', {'utc': run.now(), 'status': closure['status'],
                   'active_parent': False, 'parent_pid': os.getpid(),
                   'closure_path': str(B / 'parent_closure.json'),
                   'completed_new_rollouts': closure['completed_new_rollouts'],
                   'completed_training_branches': closure['completed_training_branches']})
        run.atomic(B / 'progress.json', {'utc': run.now(), 'stage': 'closed', 'active_jobs': [],
                   'paused': closure['reason'], 'valid_rollouts': closure['completed_new_rollouts']})
        print(run.json.dumps(closure), flush=True)


if __name__ == '__main__':
    main()
