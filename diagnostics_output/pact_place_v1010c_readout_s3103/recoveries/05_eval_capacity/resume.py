"""Continue the fixed50 evaluations after the first pool drains under the RAM guard."""
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / 'scripts'))
import pact_v1010c_run as run


def main():
    recovery = Path(__file__).resolve().parent
    assert not run.read(run.C / 'run_status.json')['active_parent']
    lock = (run.C / 'parent.lock').open('a')
    run.fcntl.flock(lock, run.fcntl.LOCK_EX | run.fcntl.LOCK_NB)
    contract = run.verify_contract(full=True)
    assert run.read(run.C / 'checkpoint/completed.json')['global_step'] == 60000
    records = run.valid_records()
    completed_readout = [row for row in records if row['id'].startswith('readout60000_')]
    assert len(completed_readout) == 8 and len(records) == 11
    for row in completed_readout:
        assert run.read(Path(row['directory']) / 'initial_pairing.json')['passed']
    path = run.C / 'root_review/release_training_cache.py'
    spec = importlib.util.spec_from_file_location('completed_training_cache', path)
    cache = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cache)
    released = cache.release()
    resources = run.resource_sample()
    assert resources['pid_fraction'] < .75 and resources['ram_fraction'] < .80 and resources['vram_fraction'] < .85
    run.freeze(recovery / 'recovery.json', {'utc': run.now(),
        'launcher_sha256': run.sha(__file__), 'cache_release_code_sha256': run.sha(path),
        'contract_sha256': contract['sha256'], 'completed_readout_rows_retained': 8,
        'remaining_rows': 42, 'eval_workers': 6, 'training_updates': 60000,
        'training_restarted': False, 'completed_rows_repeated': False,
        'resources_before_launch': resources, 'cache_release': released,
        'parent_pid': run.os.getpid(), 'scientific_contract_and_worker_source_unchanged': True})
    run.atomic(run.C / 'run_status.json', {'utc': run.now(), 'status': 'RUNNING',
        'active_parent': True, 'parent_pid': run.os.getpid(), 'recovery': '05_eval_capacity'})
    status, reason = 'INCOMPLETE', None
    try:
        scenes = run.read(run.C / 'evaluation_manifest.json')['scenes']
        run.run_jobs([run.eval_job(scene, 'readout60000') for scene in scenes], 6, 'evaluation_recovery01')
        run.finish()
        status = 'COMPLETED_COMPARISON'
    except BaseException as exc:
        reason = repr(exc)
        run.atomic(recovery / 'error.json', {'utc': run.now(), 'error': reason,
                   'traceback': run.traceback.format_exc()})
        raise
    finally:
        closure = {'utc': run.now(), 'status': status, 'reason': reason,
            'parent_pid': run.os.getpid(), 'active_parent': False,
            'valid_records': len(run.lines(run.C / 'valid_ledger.jsonl')),
            'recovery': '05_eval_capacity'}
        run.atomic(run.C / 'parent_closure.json', closure)
        run.atomic(run.C / 'run_status.json', closure)
        run.atomic(run.C / 'progress.json', dict(closure, stage='closed', active_jobs=[]))


if __name__ == '__main__':
    main()
