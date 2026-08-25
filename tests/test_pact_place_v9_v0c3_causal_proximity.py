from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_pact_place_v9_v0c3_causal_proximity import (  # noqa: E402
    _causal_metrics,
    _decision_indices,
)


def test_decision_window_is_contiguous_and_padded() -> None:
    steps = [
        {"policy_phase": phase}
        for phase in (["lift"] * 10 + ["outbound_approach"] * 3 + ["preplace"] * 10)
    ]
    indices = _decision_indices(steps)
    assert indices == list(range(2, 21))


def test_causal_metrics_preserve_sensor_and_phase_alignment() -> None:
    present = np.ones((3, 40, 4, 8, 8), dtype=np.float32)
    parked = present.copy()
    parked[1, 7, 2, 3, 4] = 2.0
    names = [f"link{i // 7 + 1}_sensor_{i}" for i in range(40)]
    metrics = _causal_metrics(
        present,
        parked,
        names,
        np.asarray([100, 101, 102]),
        ["outbound_approach", "outbound_approach", "outbound_vessel_pass"],
        1.0e-5,
    )
    assert metrics["changed_values"] == 1
    assert metrics["changed_sensors"] == 1
    assert metrics["per_sensor"][7]["changed_values"] == 1
    assert metrics["first_activation"] == {
        "window_index": 1,
        "trajectory_step": 101,
        "policy_phase": "outbound_approach",
        "steps_to_next_route_change": 1,
    }
