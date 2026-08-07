from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MOLMO = ROOT / "submodules/molmospaces"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(MOLMO))

import pact_geometry_generalization_v2_contract as contract
import pact_geometry_generalization_v2_main_contract as main_contract
from build_pact_geometry_v2_phase0_manifest import class_source


def fixture_manifest() -> dict:
    return contract.build_manifest(
        source_hashes={"fixture": "0" * 64},
        sensor_names=[f"sensor_{index:02d}" for index in range(40)],
    )


def test_phase0_contract_has_all_rows_and_exact_side_balance() -> None:
    document = fixture_manifest()
    contract.validate_manifest(document)
    assert len(document["phase0a_rows"]) == 7 * 8 == 56
    assert len(document["phase0b_candidate_rows"]) == 7 * 12 == 84
    for rows, expected in (
        (document["phase0a_rows"], {"left": 4, "right": 4}),
        (document["phase0b_candidate_rows"], {"left": 6, "right": 6}),
    ):
        for condition_id in contract.CANDIDATES:
            cell = [row for row in rows if row["condition_id"] == condition_id]
            assert Counter(row["intrusion_side"] for row in cell) == expected


def test_every_candidate_moves_exactly_one_axis_outside_training_support() -> None:
    document = fixture_manifest()
    central = {
        "panel_x_m": 0.615,
        "panel_inner_face_y_m": 0.100,
        "panel_z_m": 0.89,
        "aperture_width_m": 0.85,
        "base_forward_m": 0.14,
        "panel_half_y_m": 0.240,
    }
    for candidate_id, candidate in contract.CANDIDATES.items():
        row = next(
            item for item in document["phase0a_rows"] if item["condition_id"] == candidate_id
        )
        realized = row["realized_geometry"]
        moved = []
        for axis, value in central.items():
            if realized[axis] != pytest.approx(value):
                moved.append(axis)
        assert moved == [candidate["axis"]]
        low, high = contract.TRAINING_SUPPORT[candidate["axis"]]
        assert realized[candidate["axis"]] < low or realized[candidate["axis"]] > high


@pytest.mark.parametrize(
    ("passing", "expected"),
    [
        (set(contract.CANDIDATES), ["Z_093", "HALF_Y_030"]),
        (
            set(contract.CANDIDATES) - {"Z_093"},
            ["Z_085", "HALF_Y_030"],
        ),
        (
            {"HALF_Y_018", "X_058", "X_065", "AP_W_095"},
            ["HALF_Y_018", "X_065"],
        ),
        ({"AP_W_095"}, ["AP_W_095"]),
    ],
)
def test_selection_uses_frozen_axis_priority_only(passing: set[str], expected: list[str]) -> None:
    observed = contract.select_candidates(
        {candidate_id: candidate_id in passing for candidate_id in contract.CANDIDATES}
    )
    assert observed == expected
    axes = [contract.CANDIDATES[item]["axis"] for item in observed]
    assert len(axes) == len(set(axes))


def test_new_sampler_subclasses_are_additive_and_exact() -> None:
    from molmo_spaces.tasks.enclosure_reach import (
        PactCollisionCorridorAperture095Sampler,
        PactCollisionCorridorControlSampler,
        PactCollisionCorridorPanelHalfY018Sampler,
        PactCollisionCorridorPanelHalfY030Sampler,
        PactCollisionCorridorPanelX058Sampler,
        PactCollisionCorridorPanelX065Sampler,
        PactCollisionCorridorPanelZ085Sampler,
        PactCollisionCorridorPanelZ093Sampler,
    )

    classes = {
        "X_058": PactCollisionCorridorPanelX058Sampler,
        "X_065": PactCollisionCorridorPanelX065Sampler,
        "Z_085": PactCollisionCorridorPanelZ085Sampler,
        "Z_093": PactCollisionCorridorPanelZ093Sampler,
        "HALF_Y_018": PactCollisionCorridorPanelHalfY018Sampler,
        "HALF_Y_030": PactCollisionCorridorPanelHalfY030Sampler,
        "AP_W_095": PactCollisionCorridorAperture095Sampler,
    }
    for condition_id, cls in classes.items():
        assert issubclass(cls, PactCollisionCorridorControlSampler)
        assert cls.PACT_GEOMETRY_CONDITION == condition_id
    assert PactCollisionCorridorPanelX058Sampler.PANEL_X == pytest.approx(0.58)
    assert PactCollisionCorridorPanelX065Sampler.PANEL_X == pytest.approx(0.65)
    assert PactCollisionCorridorPanelZ085Sampler.PANEL_Z == pytest.approx(0.85)
    assert PactCollisionCorridorPanelZ093Sampler.PANEL_Z == pytest.approx(0.93)
    assert list(PactCollisionCorridorPanelHalfY018Sampler.PANEL_HALF) == pytest.approx(
        [0.055, 0.180, 0.090]
    )
    assert list(PactCollisionCorridorPanelHalfY030Sampler.PANEL_HALF) == pytest.approx(
        [0.055, 0.300, 0.090]
    )
    assert PactCollisionCorridorAperture095Sampler.APERTURE_WIDTH == pytest.approx(0.95)


def test_base_sampler_and_scene_are_still_attempt1_identical() -> None:
    sampler_path = MOLMO / "molmo_spaces/tasks/enclosure_reach.py"
    observed_base = hashlib.sha256(class_source(sampler_path.read_text())).hexdigest()
    assert observed_base == "ccd5f752f5f727d76931409798a7bda7bc2401b53a842cb3ea16d02e2d1869cc"
    scene = MOLMO / "molmo_spaces/data_generation/custom_scenes/pact_collision_corridor.xml"
    assert hashlib.sha256(scene.read_bytes()).hexdigest() == (
        "f8c04b07b9416593eb60ad4797ccbae91f7d3524effd30438ef552e5a2d75540"
    )


def test_attempt1_and_contact_endpoint_are_unchanged() -> None:
    v1_manifest = json.loads((ROOT / "configs/pact_geometry_generalization_v1.json").read_text())
    v1_screen = json.loads(
        (ROOT / "diagnostics_output/pact_geometry_generalization/expert_screen.json").read_text()
    )
    assert v1_manifest["manifest_sha256"] == (
        "33e48ab83dfe398fbeb78f64565312c48a5a8b09cb1a873a2a2521e06fcbe7b2"
    )
    assert v1_screen["expert_screen_sha256"] == (
        "3bf7d5c8f86814b9c10308c10cf1576488e992d0d20564359cb911d312d78a2c"
    )
    expected = {
        "docs/PACT_CONTACT_ENDPOINT_DECISION.md": "6ea97ea789e1c5d57572c7e231b30c867881f0e21b81ad11f950cef9bf860f1b",
        "diagnostics_output/pact_contact_endpoint/analysis.json": "4bbf25c97472bbfd8f13c3352b6b1c8afae9284cd00351e0502149362ccde1f7",
        "diagnostics_output/pact_contact_endpoint/final_decision.json": "52ce515bc09771d9ecdc8a39e31111f66e7e632f5949e82ed2497908ac2458bf",
    }
    for relative, digest in expected.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest


def test_generated_phase0_manifest_regenerates_if_present() -> None:
    path = ROOT / "configs/pact_geometry_generalization_v2_phase0.json"
    if not path.exists():
        pytest.skip("attempt-2 Phase-0 manifest has not been generated")
    contract.validate_manifest(json.loads(path.read_text()))


def test_phase0_results_reconcile_and_authorize_exact_conditions() -> None:
    root = ROOT / "diagnostics_output/pact_geometry_generalization_v2"
    screen_path = root / "expert_screen.json"
    if not screen_path.exists():
        pytest.skip("attempt-2 expert screen has not completed")
    envelope = json.loads((root / "envelope_map.json").read_text())
    selection = json.loads((root / "phase0b_selection.json").read_text())
    screen = json.loads(screen_path.read_text())
    assert envelope["row_count"] == 56
    assert envelope["selected_candidate_ids_by_frozen_priority"] == [
        "Z_093",
        "HALF_Y_030",
    ]
    assert selection["selected_candidate_ids"] == ["Z_093", "HALF_Y_030"]
    assert screen["phase0b_conditions"]["Z_093"]["clean_successes"] == 11
    assert screen["phase0b_conditions"]["HALF_Y_030"]["clean_successes"] == 11
    assert screen["surviving_condition_ids"] == ["C0", "C2", "Z_093", "HALF_Y_030"]
    assert screen["main_policy_rollout_count"] == 900
    assert screen["continue_to_policy_evaluation"] is True


def test_main_policy_instances_are_fresh_paired_and_single_axis_where_declared() -> None:
    phase0 = json.loads(
        (ROOT / "configs/pact_geometry_generalization_v2_phase0.json").read_text()
    )
    expert = json.loads(
        (ROOT / "diagnostics_output/pact_geometry_generalization_v2/expert_screen.json").read_text()
    )
    document = main_contract.build_manifest(
        phase0_manifest=phase0,
        expert_screen=expert,
        source_hashes={"fixture": "0" * 64},
    )
    main_contract.validate_manifest(document)
    assert len(document["rows"]) == 100
    assert Counter(row["intrusion_side"] for row in document["rows"]) == {
        "left": 50,
        "right": 50,
    }
    for instance_index in range(25):
        cell = [row for row in document["rows"] if row["instance_index"] == instance_index]
        assert len(cell) == 4
        assert len({(row["task_seed_u32"], row["task_seed_u64"]) for row in cell}) == 1
    assert main_contract.CONDITIONS["Z_093"]["moved_axes"] == ["panel_z_m"]
    assert main_contract.CONDITIONS["HALF_Y_030"]["moved_axes"] == [
        "panel_half_y_m"
    ]
    for row in document["rows"]:
        condition = main_contract.CONDITIONS[row["condition_id"]]
        for axis in condition["moved_axes"]:
            low, high = main_contract.TRAINING_SUPPORT[axis]
            assert row["realized_geometry"][axis] < low or row["realized_geometry"][axis] > high


def test_main_manifest_and_schedule_validate_if_present() -> None:
    manifest_path = ROOT / "configs/pact_geometry_generalization_v2.json"
    schedule_path = ROOT / "diagnostics_output/pact_geometry_generalization_v2/schedule.json"
    if not manifest_path.exists():
        pytest.skip("attempt-2 main manifest has not been generated")
    main_contract.validate_manifest(json.loads(manifest_path.read_text()))
    if not schedule_path.exists():
        pytest.skip("attempt-2 main schedule has not been generated")
    schedule = json.loads(schedule_path.read_text())
    payload = dict(schedule)
    observed = payload.pop("schedule_sha256")
    assert observed == main_contract.sha256_payload(payload)
    assert schedule["rollouts"] == len(schedule["rows"]) == 900
    assert schedule["workers"] == 8
    assert Counter(
        (row["condition_id"], row["checkpoint_seed"], row["arm"])
        for row in schedule["rows"]
    ) == Counter(
        (condition, seed, arm)
        for condition in main_contract.CONDITIONS
        for seed in (3101, 3102, 3103)
        for arm in ("ACT", "PACT", "PACT_PERMUTED")
        for _ in range(25)
    )
