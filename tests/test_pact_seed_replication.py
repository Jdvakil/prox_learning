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


def result(*, task_success: bool, hazard: int = 0, other: int = 0) -> dict:
    collision_free = task_success and hazard == 0 and other == 0
    return {
        "task_success": task_success,
        "collision_free_task_success": collision_free,
        "contact_audit": {
            "contact_class_totals": {
                "grasp_target": 1 if task_success else 0,
                "hazard_bar": hazard,
                "other_environment": other,
            }
        },
        "failure_taxonomy": ("collision_free_task_success" if collision_free else "failure"),
    }


def test_primary_endpoint_recomputed_from_task_and_non_target_contacts() -> None:
    assert analysis.endpoint(result(task_success=True))
    assert not analysis.endpoint(result(task_success=False))
    assert not analysis.endpoint(result(task_success=True, hazard=1))
    assert not analysis.endpoint(result(task_success=True, other=1))


def test_pooled_bootstrap_clusters_both_seeds_by_instance() -> None:
    by_seed = {3101: [], 3102: []}
    for index in range(40):
        for policy_seed in analysis.SEEDS:
            by_seed[policy_seed].append(
                {
                    "PACT": result(task_success=index < 30),
                    "PACT_PERMUTED": result(task_success=index < 20),
                    "ACT": result(task_success=index < 10),
                }
            )
    observed = analysis.pooled_cluster_analysis(
        by_seed,
        arm_a="PACT",
        arm_b="PACT_PERMUTED",
        replicates=1000,
        seed=123,
    )
    assert observed["difference"] == pytest.approx(0.25)
    assert observed["n_unique_instances"] == 40
    assert observed["n_seed_instance_pairs"] == 80
    assert observed["cluster_unit"] == "instance_identity_with_both_policy_seeds_resampled_together"


def test_frozen_schedule_dispatch_and_storage_contracts() -> None:
    schedule_path = ROOT / "diagnostics_output/pact_seed_replication/schedule.json"
    dispatch_path = ROOT / "diagnostics_output/pact_seed_replication/dispatch.json"
    storage_path = ROOT / "configs/pact_seed_replication_storage_amendment_v1.json"
    frozen_schedule = json.loads(schedule_path.read_text())
    frozen_dispatch = json.loads(dispatch_path.read_text())
    storage = json.loads(storage_path.read_text())
    for document, key in (
        (frozen_schedule, "schedule_sha256"),
        (frozen_dispatch, "dispatch_contract_sha256"),
        (storage, "storage_amendment_sha256"),
    ):
        payload = dict(document)
        observed = payload.pop(key)
        assert observed == canonical_hash(payload)
    assert frozen_schedule["schedule_sha256"] == (
        "1490160c44f48cf885ab1dc1fc83cd01554d82a5f83c81653bf11cfd49b3c7cb"
    )
    assert Counter(row["arm"] for row in frozen_schedule["rows"]) == {
        "ACT": 40,
        "PACT": 40,
        "PACT_PERMUTED": 40,
    }
    assert [row["schedule_index"] for row in frozen_schedule["rows"]] == list(range(120))
    assert len(frozen_schedule["seed_3101_references"]) == 120
    smoke = frozen_schedule["rows"][0]
    assert smoke["arm"] == "PACT_PERMUTED"
    assert smoke["max_control_steps"] == 900
    assert smoke["token_plan_row"] == 0
    assert frozen_dispatch["execution"]["fixed_worker_count"] == 8
    assert frozen_dispatch["analysis"]["sha256"] == file_hash(
        ROOT / "scripts/analyze_pact_seed_replication.py"
    )
    assert storage["excluded_intact_schedule_indices"] == [0, 119]
    assert storage["seed_replication_compactor_wrapper_sha256"] == file_hash(
        ROOT / "scripts/compact_pact_seed_replication_storage.py"
    )
