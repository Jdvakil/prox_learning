"""Run unchanged ACT training with bounded, explicitly recorded checkpoint I/O.

Periodic snapshots and optimizer resume bundles are not written. Final best and
last are written normally; scientific training and validation are unchanged.
"""
import ast
import sys
from pathlib import Path
from pact_place_v1011c_experiment import ROOT, TRAIN, freeze, environment
import os
os.environ.update(environment())
import imitate_episodes as trainer


def main():
    original = trainer._atomic_torch_save
    skipped = []

    def bounded_save(obj, path):
        target = Path(path).resolve()
        assert target.is_relative_to(TRAIN.resolve()), target
        if target.name.startswith('policy_epoch_') or target.name == 'resume_bundle.ckpt':
            skipped.append(target.name)
            return
        original(obj, path)

    trainer._atomic_torch_save = bounded_save
    source = ROOT / 'submodules/act/imitate_episodes.py'
    tree = ast.parse(source.read_text(), filename=str(source))
    entrypoints = [n for n in tree.body if isinstance(n, ast.If)
                   and ast.unparse(n.test) == "__name__ == '__main__'"]
    assert len(entrypoints) == 1
    body = ast.Module(body=entrypoints[0].body, type_ignores=[])
    exec(compile(body, str(source), 'exec'), trainer.__dict__)
    directory = Path(sys.argv[sys.argv.index('--ckpt_dir')+1])
    freeze(directory / 'checkpoint_storage.json', {
        'skipped_writes':skipped,
        'retained_at_training_exit':['policy_best.ckpt','policy_last.ckpt'],
        'optimizer_resume_available':False,
        'training_math_unchanged':True})


if __name__ == '__main__':
    main()
