from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import analyze_pact_blur_sweep as analysis
import build_pact_blur_manifest as manifest_builder
import build_pact_blur_schedule as schedule_builder
import pact_blur_sweep_contract as contract


CALIBRATION = ROOT / "diagnostics_output/pact_blur_sweep/calibration.json"
V3_MANIFEST = ROOT / "configs/pact_geometry_generalization_v3.json"
V3_SCHEDULE = ROOT / "diagnostics_output/pact_geometry_generalization_v3/schedule.json"
REGISTRY = ROOT / "diagnostics_output/pact_contact_endpoint/policy_training.json"
TOKEN_PLAN = ROOT / "diagnostics_output/pact_contact_endpoint/token_plan.json"
ANALYSIS_SCRIPT = ROOT / "scripts/analyze_pact_blur_sweep.py"


def build_manifest() -> dict:
    return manifest_builder.build(
        calibration=json.loads(CALIBRATION.read_text()),
        calibration_path=CALIBRATION,
        v3_manifest=manifest_builder.load_v3_manifest(V3_MANIFEST),
        v3_manifest_path=V3_MANIFEST,
        registry=json.loads(REGISTRY.read_text()),
        registry_path=REGISTRY,
        token_plan=json.loads(TOKEN_PLAN.read_text()),
        token_plan_path=TOKEN_PLAN,
    )


def build_schedule() -> dict:
    return schedule_builder.build(
        manifest=build_manifest(),
        registry=json.loads(REGISTRY.read_text()),
        token_plan=json.loads(TOKEN_PLAN.read_text()),
        token_plan_path=TOKEN_PLAN,
        v3_schedule=json.loads(V3_SCHEDULE.read_text()),
        analysis_script=ANALYSIS_SCRIPT,
    )


def test_calibrated_grid_and_shared_instances_are_frozen() -> None:
    document = build_manifest()
    contract.validate_manifest(document)
    assert document["blur_sigmas"] == [0.0, 0.5, 1.0, 2.0]
    assert document["calibration_binding"]["transition_sigmas"] == [0.5, 1.0]
    assert document["calibration_binding"]["selection_used_policy_outcomes"] is False
    assert document["source_v3"]["source_policy_outcomes_read"] is False
    assert len(document["rows"]) == len({row["episode_id"] for row in document["rows"]}) == 25
    assert Counter(row["intrusion_side"] for row in document["rows"]) == {
        "left": 12,
        "right": 13,
    }


def test_schedule_is_exact_balanced_900_row_factorial() -> None:
    schedule = build_schedule()
    assert schedule["rollouts"] == len(schedule["rows"]) == 900
    assert schedule["workers"] == 12
    assert schedule["instances_shared_across_sigmas_arms_seeds"] is True
    expected = Counter(
        (sigma, seed, arm)
        for sigma in contract.BLUR_SIGMAS
        for seed in (3101, 3102, 3103)
        for arm in ("ACT", "PACT", "PACT_PERMUTED")
        for _ in range(25)
    )
    assert Counter(
        (row["blur_sigma"], row["checkpoint_seed"], row["arm"])
        for row in schedule["rows"]
    ) == expected
    for instance in range(25):
        rows = [row for row in schedule["rows"] if row["instance_index"] == instance]
        assert len(rows) == 36
        assert len({row["instance_episode_id"] for row in rows}) == 1
        token_rows = {
            row["token_plan_row"] for row in rows if row["arm"] == "PACT_PERMUTED"
        }
        assert len(token_rows) == 1
    smoke = schedule["rows"][0]
    assert (smoke["blur_sigma"], smoke["checkpoint_seed"], smoke["arm"]) == (
        0.0,
        3101,
        "PACT_PERMUTED",
    )


def decision_fixture(gaps: list[tuple[float, float]], rates: float = 0.3) -> tuple[dict, dict]:
    schedule = {
        "blur_sigmas": [0.0, 0.5, 1.0, 2.0],
        "collapse_floor_collision_free_success": 0.1,
    }
    absolute = {
        str(sigma): {
            arm: {"pooled": {"collision_free_task_success": {"rate": rates}}}
            for arm in analysis.ARMS
        }
        for sigma in schedule["blur_sigmas"]
    }
    contrasts = {}
    for sigma, (point, lower) in zip(schedule["blur_sigmas"][1:], gaps):
        contrasts[str(sigma)] = {
            "PACT_minus_ACT": {
                "pooled": {
                    "collision_free_task_success": {
                        "difference": point,
                        "instance_cluster_bootstrap_ci_95": [lower, point + 0.1],
                    }
                }
            }
        }
    return schedule, {
        "reconciliation": {"reconciled": True},
        "absolute_performance": absolute,
        "contrasts_per_sigma": contrasts,
    }


def test_frozen_decision_precedence() -> None:
    schedule, document = decision_fixture([(0.02, -0.01), (0.05, -0.01), (0.1, 0.01)])
    assert analysis.choose_decision(schedule, document)[0] == "BLUR_ROBUSTNESS_ESTABLISHED"
    schedule, document = decision_fixture([(0.08, 0.01), (0.03, -0.02), (0.01, -0.03)])
    assert analysis.choose_decision(schedule, document)[0] == "BLUR_PARTIAL"
    schedule, document = decision_fixture([(0.02, -0.02), (0.03, -0.03), (0.04, -0.02)])
    assert analysis.choose_decision(schedule, document)[0] == "NO_BLUR_ROBUSTNESS"
    schedule, document = decision_fixture([(0.1, 0.01)] * 3, rates=0.01)
    assert analysis.choose_decision(schedule, document)[0] == "BLUR_UNINFORMATIVE_COLLAPSE"


def test_result_validation_rejects_a_silently_dropped_sigma() -> None:
    row = build_schedule()["rows"][0]
    result = {
        "status": "complete",
        "rollout_id": row["rollout_id"],
        "schedule_row_sha256": row["schedule_row_sha256"],
        "episode_id": row["instance_episode_id"],
        "arm": row["arm"],
        "checkpoint_seed": row["checkpoint_seed"],
        "checkpoint_sha256": row["checkpoint_sha256"],
        "blur_sigma": 2.0,
    }
    try:
        analysis.validate_result(result, row)
    except ValueError as error:
        assert "blur_sigma" in str(error)
    else:
        raise AssertionError("a mismatched recorded blur sigma was accepted")
