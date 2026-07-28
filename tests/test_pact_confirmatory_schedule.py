from __future__ import annotations

import importlib.util
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "build_pact_confirmatory_schedule",
        ROOT / "scripts" / "build_pact_confirmatory_schedule.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


schedule = _load()


def test_arm_orders_are_balanced_and_deterministic():
    first = schedule._arm_orders(80)
    second = schedule._arm_orders(80)
    assert first == second
    assert all(set(order) == set(schedule.ARMS) for order in first)
    for position in range(3):
        counts = Counter(order[position] for order in first)
        assert max(counts.values()) - min(counts.values()) <= 1
