import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / 'scripts'))
import pact_v1010c_core as core
from fixed_split_data import seed_epoch
from capacity_loader import CapacityLoader
import capacity_loader


def test_all_real_train_and_validation_batches_and_rng_are_unchanged():
    first = core.make_loaders()
    second = core.make_loaders()
    seed_epoch(first[0], first[1], 3103, 690)
    seed_epoch(second[0], second[1], 3103, 690)
    calls = []
    rng = torch.get_rng_state().clone()
    numpy_rng = np.random.get_state()
    counts = []
    for original, underlying in zip(first[:2], second[:2]):
        wrapped = CapacityLoader(underlying, gate=lambda: calls.append(True))
        assert wrapped.generator is underlying.generator
        assert wrapped.dataset is underlying.dataset
        count = 0
        for left, right in zip(original, wrapped, strict=True):
            assert len(left) == len(right)
            assert all(torch.equal(a, b) for a, b in zip(left, right))
            count += 1
        counts.append(count)
    assert counts == [30, 5] and len(calls) == 39
    assert torch.equal(rng, torch.get_rng_state())
    after = np.random.get_state()
    assert numpy_rng[0] == after[0] and np.array_equal(numpy_rng[1], after[1]) and numpy_rng[2:] == after[2:]


def test_pressure_wait_recovers_before_iteration(monkeypatch, tmp_path):
    readings = iter([.8, .8, .8, .6, .3, .3])
    delays = []
    monkeypatch.setattr(capacity_loader, 'pid_fraction', lambda: next(readings))
    monkeypatch.setattr(capacity_loader.time, 'sleep', delays.append)
    monkeypatch.setattr(capacity_loader, 'C', tmp_path)
    (tmp_path / 'recoveries/04_capacity_wait').mkdir(parents=True)
    capacity_loader.await_capacity()
    assert delays == [.25, .25]
