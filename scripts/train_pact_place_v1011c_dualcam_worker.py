"""Unmodified trainer math, resumable new run, exact 20k/30k snapshots.

Only checkpoint storage and progress instrumentation are wrapped. The protected
ACT submodule is neither edited nor patched on disk.
"""
from pact_place_v1011c_dualcam import *
import ast
import inspect
import time
os.environ.update(environment())
import imitate_episodes as trainer


def main():
    started = time.time()
    original_save = trainer._atomic_torch_save
    original_log = trainer._append_epoch_log
    skipped = []

    def bounded_save(obj, path):
        target = Path(path).resolve()
        assert target.is_relative_to(TRAIN.resolve()), target
        if target.name.startswith('policy_epoch_'):
            skipped.append(target.name)
            return
        original_save(obj, path)

    def progress(*args, **kwargs):
        original_log(*args, **kwargs)
        directory, epoch, step = Path(args[0]), int(args[1]), int(args[2])
        if step % 1000 == 0:
            freeze(directory/'progress'/f'step_{step:05}.json', {
                'epoch':epoch,'global_step':step,'elapsed_seconds':time.time()-started,
                'wall_time_unix':time.time(),'thread_environment':THREAD_ENV})
            print(f'PILOT_PROGRESS step={step} elapsed_seconds={time.time()-started:.3f}', flush=True)
        if step in MILESTONES:
            frame = inspect.currentframe().f_back
            assert frame.f_code.co_name == 'train_bc'
            policy = frame.f_locals['policy']
            target = directory/f'policy_step_{step}.ckpt'
            assert not target.exists()
            original_save(policy.state_dict(), target)
            freeze(directory/f'policy_step_{step}.json', {**empty_authorization(),
                'global_step':step, 'epoch':epoch, 'checkpoint_sha256':sha256_file(target),
                'camera_names':CAMERAS,'action_alignment':ALIGNMENT,
                'dataset_stats_sha256':sha256_file(directory/'dataset_stats.pkl'),
                'run_manifest_sha256':sha256_file(directory/'run_manifest.json')})
            del frame

    trainer._atomic_torch_save = bounded_save
    trainer._append_epoch_log = progress
    source = ROOT/'submodules/act/imitate_episodes.py'
    tree = ast.parse(source.read_text(), filename=str(source))
    blocks = [n for n in tree.body if isinstance(n, ast.If) and ast.unparse(n.test) == "__name__ == '__main__'"]
    assert len(blocks) == 1
    exec(compile(ast.Module(body=blocks[0].body,type_ignores=[]),str(source),'exec'), trainer.__dict__)
    directory = Path(sys.argv[sys.argv.index('--ckpt_dir')+1])
    freeze(directory/'checkpoint_storage.json', {'skipped_periodic_policy_files':skipped,
        'resume_bundle_retained':True,'exact_step_snapshots':list(MILESTONES),
        'training_math_unchanged':True, 'elapsed_seconds':time.time()-started})


if __name__ == '__main__':
    main()
