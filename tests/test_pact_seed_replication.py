from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import analyze_pact_seed_replication as analysis
import build_pact_seed_replication_schedule as schedule


def canonical_hash(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contrast(value: float) -> dict:
    return {"difference": value}


def decision_inputs(
    *, pact_act_3102: float, modality_3101: float, modality_3102: float
) -> dict[int, dict[str, dict]]:
    return {
        3101: {
            "PACT_minus_ACT": contrast(0.25),
            "PACT_minus_PACT_PERMUTED": contrast(modality_3101),
        },
        3102: {
            "PACT_minus_ACT": contrast(pact_act_3102),
            "PACT_minus_PACT_PERMUTED": contrast(modality_3102),
        },
    }


@pytest.mark.parametrize(
    ("pact_act", "modality_3101", "modality_3102", "expected"),
    [
        (0.10, 0.01, 0.01, "SEED_REPLICATION_CONFIRMED"),
        (0.10, 0.01, 0.00, "SEED_REPLICATION_PARTIAL"),
        (0.10, -0.01, 0.01, "SEED_REPLICATION_PARTIAL"),
        (0.099999, 0.20, 0.20, "SEED_REPLICATION_FAILED"),
    ],
)
def test_frozen_decision_boundaries(
    pact_act: float, modality_3101: float, modality_3102: float, expected: str
) -> None:
    observed = analysis.choose_decision(
        True,
        decision_inputs(
            pact_act_3102=pact_act,
            modality_3101=modality_3101,
            modality_3102=modality_3102,
        ),
    )
    assert observed == expected


def test_incomplete_preempts_outcomes() -> None:
    assert (
        analysis.choose_decision(
            False,
            decision_inputs(pact_act_3102=1.0, modality_3101=1.0, modality_3102=1.0),
        )
        == "SEED_REPLICATION_INCOMPLETE"
    )


def test_arm_order_is_balanced_and_smoke_exercises_permutation() -> None:
    orders = schedule.arm_orders(40)
    assert len(orders) == 40
    assert orders[0][0] == "PACT_PERMUTED"
    assert all(set(order) == set(schedule.ARMS) for order in orders)
    for position in range(3):
        counts = Counter(order[position] for order in orders)
        assert max(counts.values()) - min(counts.values()) <= 1


def test_preregistration_and_frozen_analyzer_hashes() -> None:
    path = ROOT / "configs/pact_seed_replication_preregistration_v1.json"
    preregistration = json.loads(path.read_text())
    payload = dict(preregistration)
    observed = payload.pop("preregistration_sha256")
    assert observed == canonical_hash(payload)
    analyzer = ROOT / "scripts/analyze_pact_seed_replication.py"
    assert preregistration["analysis"]["frozen_analysis_script_sha256"] == file_hash(analyzer)


def test_pact_zero_is_absent_from_design_and_decisions() -> None:
    preregistration = json.loads(
        (ROOT / "configs/pact_seed_replication_preregistration_v1.json").read_text()
    )
    assert preregistration["design"]["arms"] == list(analysis.ARMS)
    assert "PACT_ZERO" not in preregistration["design"]["arms"]
    assert analysis.TOKENS == {
        "SEED_REPLICATION_CONFIRMED",
        "SEED_REPLICATION_PARTIAL",
        "SEED_REPLICATION_FAILED",
        "SEED_REPLICATION_INCOMPLETE",
    }
