from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MOLMO = ROOT / "submodules/molmospaces"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(MOLMO))

import analyze_pact_geometry_generalization_v3 as analysis
import build_pact_geometry_v3_schedule as schedule_builder
import build_pact_geometry_v3_worker_sizing as sizing_builder
import pact_geometry_generalization_v3_contract as contract


def phase0_inputs() -> tuple[dict, dict]:
    phase0 = json.loads(
        (ROOT / "configs/pact_geometry_generalization_v2_phase0.json").read_text()
    )
    expert = json.loads(
        (
            ROOT
            / "diagnostics_output/pact_geometry_generalization_v2/expert_screen.json"
        ).read_text()
    )
    return phase0, expert


def test_v3_contract_is_fresh_balanced_and_reuses_only_passing_conditions() -> None:
    phase0, expert = phase0_inputs()
    document = contract.build_manifest(
        phase0_manifest=phase0,
        expert_screen=expert,
        source_hashes={"fixture": "0" * 64},
    )
    contract.validate_manifest(document)
    assert list(contract.CONDITIONS) == ["C0", "C2", "Z_093"]
    assert len(document["rows"]) == 120
    assert document["planned_policy_design"] == {
        "instances_per_condition": 40,
        "arms": ["PACT", "PACT_PERMUTED"],
        "checkpoint_seeds": [3101, 3102, 3103],
        "workers": 12,
        "rollouts": 720,
    }
    for condition in contract.CONDITIONS:
        rows = [row for row in document["rows"] if row["condition_id"] == condition]
        assert Counter(row["intrusion_side"] for row in rows) == {
            "left": 20,
            "right": 20,
        }
    for instance_index in range(40):
        rows = [
            row for row in document["rows"] if row["instance_index"] == instance_index
        ]
        assert len(rows) == 3
        assert len({(row["task_seed_u32"], row["task_seed_u64"]) for row in rows}) == 1
    v2 = json.loads((ROOT / "configs/pact_geometry_generalization_v2.json").read_text())
    assert not ({row["episode_id"] for row in document["rows"]} & {row["episode_id"] for row in v2["rows"]})


def test_shifted_v3_geometries_are_outside_training_support() -> None:
    for row in contract.build_rows():
        condition = contract.CONDITIONS[row["condition_id"]]
        for axis in condition["moved_axes"]:
            low, high = contract.TRAINING_SUPPORT[axis]
            assert not low <= float(row["realized_geometry"][axis]) <= high


def test_token_rows_are_frozen_unique_and_paired_across_conditions() -> None:
    token_rows = schedule_builder.token_plan_row_map()
    assert len(token_rows) == len(set(token_rows)) == 40
    assert min(token_rows) >= 0
    assert max(token_rows) < 100


def test_worker_sizing_selects_twelve_below_ceiling() -> None:
    sizing = sizing_builder.build()
    assert sizing["arithmetic_worker_limit"] == 12
    assert sizing["selected_workers"] == 12
    assert sizing["projected_peak_mib"] == 18480.0
    assert sizing["projected_peak_mib"] < sizing["ceiling_mib"]
    assert sizing["fallback_to_10_required_by_arithmetic"] is False


def test_decision_precedence_is_unchanged() -> None:
    reconciled = {"reconciled": True}
    pooled_below = {"instance_cluster_bootstrap_ci_95": [-2.0, -0.1]}
    assert analysis.choose_decision(
        reconciliation=reconciled,
        c0_reproduces=True,
        shifted_support={"C2": True, "Z_093": True},
        pooled_any=pooled_below,
        pooled_frames=pooled_below,
    )[0] == "GEOMETRY_GENERALIZES"
    assert analysis.choose_decision(
        reconciliation=reconciled,
        c0_reproduces=True,
        shifted_support={"C2": True, "Z_093": False},
        pooled_any=pooled_below,
        pooled_frames=pooled_below,
    )[0] == "GEOMETRY_PARTIAL"
    assert analysis.choose_decision(
        reconciliation=reconciled,
        c0_reproduces=False,
        shifted_support={"C2": True, "Z_093": True},
        pooled_any=pooled_below,
        pooled_frames=pooled_below,
    )[0] == "GEOMETRY_TEST_INCONCLUSIVE"


def test_v2_abandonment_is_explicitly_outcome_blind() -> None:
    path = ROOT / "diagnostics_output/pact_geometry_generalization_v2/abandonment.json"
    record = json.loads(path.read_text())
    payload = dict(record)
    observed = payload.pop("abandonment_sha256")
    assert observed == contract.sha256_payload(payload)
    assert record["decision"] == "ABANDONED_PRE_INTERPRETATION"
    assert record["result_files_opened"] is False
    assert record["endpoint_fields_read"] is False
    assert record["analysis_run"] is False


def test_frozen_contact_endpoint_is_byte_identical() -> None:
    expected = {
        "docs/PACT_CONTACT_ENDPOINT_DECISION.md": "6ea97ea789e1c5d57572c7e231b30c867881f0e21b81ad11f950cef9bf860f1b",
        "diagnostics_output/pact_contact_endpoint/analysis.json": "4bbf25c97472bbfd8f13c3352b6b1c8afae9284cd00351e0502149362ccde1f7",
        "diagnostics_output/pact_contact_endpoint/final_decision.json": "52ce515bc09771d9ecdc8a39e31111f66e7e632f5949e82ed2497908ac2458bf",
    }
    for relative, digest in expected.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest


def test_frozen_v3_artifacts_validate_if_present() -> None:
    manifest_path = ROOT / "configs/pact_geometry_generalization_v3.json"
    schedule_path = (
        ROOT / "diagnostics_output/pact_geometry_generalization_v3/schedule.json"
    )
    if not manifest_path.exists() or not schedule_path.exists():
        return
    manifest = json.loads(manifest_path.read_text())
    contract.validate_manifest(manifest)
    schedule = json.loads(schedule_path.read_text())
    payload = dict(schedule)
    assert payload.pop("schedule_sha256") == contract.sha256_payload(payload)
    assert schedule["rollouts"] == len(schedule["rows"]) == 720
    assert schedule["workers"] == 12
    assert schedule["arms"] == ["PACT", "PACT_PERMUTED"]
    expected = Counter(
        (condition, seed, arm)
        for condition in contract.CONDITIONS
        for seed in (3101, 3102, 3103)
        for arm in ("PACT", "PACT_PERMUTED")
        for _ in range(40)
    )
    assert Counter(
        (row["condition_id"], row["checkpoint_seed"], row["arm"])
        for row in schedule["rows"]
    ) == expected
    for condition in contract.CONDITIONS:
        for instance in range(40):
            token_rows = {
                row["token_plan_row"]
                for row in schedule["rows"]
                if row["condition_id"] == condition
                and row["instance_index"] == instance
                and row["arm"] == "PACT_PERMUTED"
            }
            assert token_rows == {schedule["token_plan_row_map_by_instance_index"][instance]}
