"""Resume unchanged experiment commands after the root's thread-capacity review."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / 'scripts'))
import pact_v1010c_run as run


def main():
    lock = (run.C / 'parent.lock').open('a')
    run.fcntl.flock(lock, run.fcntl.LOCK_EX | run.fcntl.LOCK_NB)
    contract = run.verify_contract(full=True)
    recovery = Path(__file__).resolve().parent
    source = run.C / 'checkpoint/resume_bundle.ckpt'
    expected = run.read(run.C / 'checkpoint/checkpoint_progress.json')
    assert run.sha(source) == expected['resume_sha256']
    assert expected['global_step'] == 15600
    bundle = run.torch.load(source, map_location='cpu', weights_only=False)
    assert bundle['contract_sha256'] == contract['sha256']
    assert bundle['global_step'] == 15600 and bundle['epoch'] == 519
    steps = {int(value['step']) for value in bundle['optimizer_state']['state'].values() if 'step' in value}
    assert steps == {15600}
    assert len(bundle['optimizer_state']['param_groups']) == 3
    assert all(run.torch.isfinite(value).all() for key in ('model_state', 'encoder_state') for value in bundle[key].values())
    del bundle
    resources = run.resource_sample()
    assert resources['pid_fraction'] < .75 and resources['ram_fraction'] < .80 and resources['vram_fraction'] < .85
    run.freeze(recovery / 'recovery.json', {
        'utc': run.now(), 'launcher_sha256': run.sha(__file__),
        'contract_sha256': contract['sha256'], 'contract_and_training_code_unchanged': True,
        'resume_sha256': expected['resume_sha256'], 'resume_step': 15600,
        'optimizer_steps_verified': True, 'policy_and_encoder_parameters_finite': True,
        'resources_before_launch': resources, 'parent_pid': run.os.getpid()})
    run.atomic(run.C / 'run_status.json', {'utc': run.now(), 'status': 'RUNNING',
        'active_parent': True, 'parent_pid': run.os.getpid(), 'recovery': '03_thread_capacity'})
    status, reason = 'INCOMPLETE', None
    try:
        valid = {record['id'] for record in run.valid_records()}
        scenes = run.read(run.C / 'evaluation_manifest.json')['scenes']
        assert 'training_to_300' in valid
        assert f'frozen60000_{scenes[0]["scene_id"][:20]}' in valid
        run.replay_check(scenes[0])
        job = run.train_job(60000)
        job['id'] = 'training_to_60000_retry01'
        run.run_jobs([job], 1, 'training_recovery01')
        run.run_jobs([run.eval_job(scene, 'readout60000') for scene in scenes], 8, 'evaluation')
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
            'recovery': '03_thread_capacity'}
        run.atomic(run.C / 'parent_closure.json', closure)
        run.atomic(run.C / 'run_status.json', closure)
        run.atomic(run.C / 'progress.json', dict(closure, stage='closed', active_jobs=[]))


if __name__ == '__main__':
    main()
