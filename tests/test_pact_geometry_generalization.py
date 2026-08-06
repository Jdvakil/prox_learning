from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MOLMO = ROOT / "submodules/molmospaces"
sys.path.insert(0, str(ROOT / "scripts"))

import analyze_pact_geometry_generalization as analysis
import pact_geometry_generalization_contract as contract


def test_manifest_is_balanced_and_every_shift_moves_two_axes() -> None:
    document = contract.build_manifest(
        source_hashes={"fixture": "0" * 64},
        sensor_names=[f"sensor_{index:02d}" for index in range(40)],
    )
    contract.validate_manifest(document)
    for condition_id, condition in document["conditions"].items():
        if condition_id != "C0":
            assert len(condition["moved_axes"]) >= 2
    expert_sides = Counter(row["intrusion_side"] for row in document["expert_screen_rows"])
    policy_sides = Counter(row["intrusion_side"] for row in document["rows"])
    assert expert_sides == {"left": 24, "right": 24}
    assert policy_sides == {"left": 50, "right": 50}


def test_shifted_realized_values_are_outside_training_support() -> None:
    document = contract.build_manifest(
        source_hashes={},
        sensor_names=[f"sensor_{index:02d}" for index in range(40)],
    )
    for row in document["expert_screen_rows"] + document["rows"]:
        condition = document["conditions"][row["condition_id"]]
        for axis in condition["moved_axes"]:
            low, high = document["training_support"][axis]
            assert row["realized_geometry"][axis] < low or row["realized_geometry"][axis] > high


def _class_source(text: str, end_marker: str) -> bytes:
    start = text.index("class PactCollisionCorridorSampler(")
    end = text.index(end_marker, start)
    return text[start:end].rstrip().encode() + b"\n"


def test_base_sampler_source_and_scene_xml_are_byte_identical_to_head() -> None:
    sampler = MOLMO / "molmo_spaces/tasks/enclosure_reach.py"
    current = sampler.read_text()
    committed = subprocess.run(
        ["git", "show", "HEAD:molmo_spaces/tasks/enclosure_reach.py"],
        cwd=MOLMO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    committed_end = (
        "\n\nclass PactCollisionCorridorControlSampler"
        if "class PactCollisionCorridorControlSampler" in committed
        else "\nclass PactCollisionCorridorPolicy"
    )
    assert _class_source(
        current, "\n\nclass PactCollisionCorridorControlSampler"
    ) == _class_source(committed, committed_end)
    xml = "molmo_spaces/data_generation/custom_scenes/pact_collision_corridor.xml"
    committed_xml = subprocess.run(
        ["git", "show", f"HEAD:{xml}"], cwd=MOLMO, check=True, capture_output=True
    ).stdout
    assert (MOLMO / xml).read_bytes() == committed_xml


def test_geometry_sampler_subclasses_are_additive_and_have_declared_values() -> None:
    sys.path.insert(0, str(MOLMO))
    from molmo_spaces.tasks.enclosure_reach import (
        PactCollisionCorridorControlSampler,
        PactCollisionCorridorDeeperHigherSampler,
        PactCollisionCorridorSampler,
        PactCollisionCorridorShallowerWiderSampler,
        PactCollisionCorridorTighterSampler,
    )

    for cls in (
        PactCollisionCorridorControlSampler,
        PactCollisionCorridorDeeperHigherSampler,
        PactCollisionCorridorTighterSampler,
        PactCollisionCorridorShallowerWiderSampler,
    ):
        assert issubclass(cls, PactCollisionCorridorSampler)
    assert PactCollisionCorridorDeeperHigherSampler.PANEL_X == pytest.approx(0.68)
    assert PactCollisionCorridorDeeperHigherSampler.PANEL_Z == pytest.approx(0.96)
    assert PactCollisionCorridorTighterSampler.PANEL_INNER_FACE_Y == pytest.approx(0.070)
    assert PactCollisionCorridorTighterSampler.APERTURE_WIDTH == pytest.approx(0.70)
    assert PactCollisionCorridorShallowerWiderSampler.PANEL_X == pytest.approx(0.55)
    assert PactCollisionCorridorShallowerWiderSampler.APERTURE_WIDTH == pytest.approx(1.00)


@pytest.mark.parametrize(
    ("reconciled", "c0", "support", "any_ci", "frame_ci", "expected"),
    [
        (False, False, {}, [-1, 1], [-1, 1], "GEOMETRY_TEST_INCONCLUSIVE"),
        (True, False, {"C1": True, "C2": True}, [-1, -0.1], [-2, -0.1], "GEOMETRY_TEST_INCONCLUSIVE"),
        (True, True, {"C1": True}, [-1, -0.1], [-2, -0.1], "GEOMETRY_TEST_INCONCLUSIVE"),
        (True, True, {"C1": True, "C2": True}, [-1, -0.1], [-2, -0.1], "GEOMETRY_GENERALIZES"),
        (True, True, {"C1": True, "C2": False}, [-1, -0.1], [-2, -0.1], "GEOMETRY_PARTIAL"),
        (True, True, {"C1": True, "C2": True}, [-1, 0.1], [-2, -0.1], "GEOMETRY_DOES_NOT_GENERALIZE"),
    ],
)
def test_frozen_decision_boundaries(reconciled, c0, support, any_ci, frame_ci, expected) -> None:
    token, _ = analysis.choose_decision(
        reconciliation={"reconciled": reconciled},
        c0_reproduces=c0,
        shifted_support=support,
        pooled_any={"instance_cluster_bootstrap_ci_95": any_ci},
        pooled_frames={"instance_cluster_bootstrap_ci_95": frame_ci},
    )
    assert token == expected


def test_frozen_contact_endpoint_artifacts_are_unchanged() -> None:
    expected = {
        "docs/PACT_CONTACT_ENDPOINT_DECISION.md": "6ea97ea789e1c5d57572c7e231b30c867881f0e21b81ad11f950cef9bf860f1b",
        "diagnostics_output/pact_contact_endpoint/analysis.json": "4bbf25c97472bbfd8f13c3352b6b1c8afae9284cd00351e0502149362ccde1f7",
        "diagnostics_output/pact_contact_endpoint/final_decision.json": "52ce515bc09771d9ecdc8a39e31111f66e7e632f5949e82ed2497908ac2458bf",
    }
    for relative, digest in expected.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest


def test_generated_manifest_regenerates_if_present() -> None:
    path = ROOT / "configs/pact_geometry_generalization_v1.json"
    if not path.exists():
        pytest.skip("manifest has not been generated yet")
    document = json.loads(path.read_text())
    contract.validate_manifest(document)


def test_phase0_stop_artifacts_reconcile_and_are_compact_if_present() -> None:
    output = ROOT / "diagnostics_output/pact_geometry_generalization"
    final_path = output / "final_decision.json"
    if not final_path.exists():
        pytest.skip("Phase-0 screen has not completed yet")
    final = json.loads(final_path.read_text())
    payload = dict(final)
    observed = payload.pop("final_decision_sha256")
    assert observed == contract.sha256_payload(payload)
    assert final["decision"] == "GEOMETRY_TEST_INCONCLUSIVE"
    schedule = json.loads((output / "schedule.json").read_text())
    assert schedule["actual_policy_rollouts"] == 0
    assert schedule["rows"] == []
    analysis_result = json.loads((output / "analysis.json").read_text())
    assert sum(item["n"] for item in analysis_result["condition_results"].values()) == 48
    assert analysis_result["surviving_shifted_condition_ids"] == ["C2"]
    row_paths = sorted((output / "expert_screen_rows").glob("*/*/result.json"))
    assert len(row_paths) == 48
    for path in row_paths:
        result = json.loads(path.read_text())
        result_hash = result.pop("result_sha256")
        assert result_hash == contract.sha256_payload(result)
        if "contact_audit" in result:
            assert result["contact_audit"]["contact_frame_payload_retained"] is False
            assert result["contact_audit"]["contact_frames"] == []
    report_lines = (ROOT / "docs/PACT_GEOMETRY_GENERALIZATION.md").read_text().splitlines()
    assert next(line for line in reversed(report_lines) if line.strip()) == "GEOMETRY_TEST_INCONCLUSIVE"
