"""Operational entry point; the original training and upstream encoder stay intact."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / 'scripts'))
import pact_v1010c_train as trainer
from capacity_loader import CapacityLoader

original_make_loaders = trainer.make_loaders


def make_loaders():
    train, val, stats, meta = original_make_loaders()
    return CapacityLoader(train), CapacityLoader(val), stats, meta


if __name__ == '__main__':
    trainer.os.environ.update(trainer.environment())
    recovery = Path(__file__).resolve().parent
    for path, expected in trainer.read(recovery / 'recovery.json')['operational_wrapper_hashes'].items():
        assert trainer.sha(path) == expected, path
    trainer.make_loaders = make_loaders
    trainer.train(60000)
