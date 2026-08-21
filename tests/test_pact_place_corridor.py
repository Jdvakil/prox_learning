from __future__ import annotations

import ast
import hashlib
import json
import math
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
MOLMO = ROOT / "submodules" / "molmospaces"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(MOLMO))

import pact_place_corridor_contract as contract


def _class_source(source: str, name: str) -> str:
    tree = ast.parse(source)
    node = next(
        item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == name
    )
    lines = source.splitlines(keepends=True)
    return "".join(lines[node.lineno - 1 : node.end_lineno])


def test_frozen_corridor_xml_and_classes_are_unchanged() -> None:
    xml_relative = "molmo_spaces/data_generation/custom_scenes/pact_collision_corridor.xml"
    committed_xml = subprocess.run(
        ["git", "show", f"HEAD:{xml_relative}"],
        cwd=MOLMO,
        check=True,
        capture_output=True,
    ).stdout
    current_xml = (MOLMO / xml_relative).read_bytes()
    assert current_xml == committed_xml
    assert hashlib.sha256(current_xml).hexdigest() == (
        "f8c04b07b9416593eb60ad4797ccbae91f7d3524effd30438ef552e5a2d75540"
    )

    source_path = MOLMO / "molmo_spaces/tasks/enclosure_reach.py"
    current = source_path.read_text()
    committed = subprocess.run(
        ["git", "show", "HEAD:molmo_spaces/tasks/enclosure_reach.py"],
        cwd=MOLMO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    for name in (
        "PactCollisionCorridorSampler",
        "PactCollisionCorridorPolicy",
        "PactCollisionCorridorPolicyConfig",
    ):
        assert _class_source(current, name) == _class_source(committed, name)


def test_scene_is_a_strict_fork_with_reachable_outside_tray() -> None:
    base = ET.parse(
        MOLMO
        / "molmo_spaces/data_generation/custom_scenes/pact_collision_corridor.xml"
    ).getroot()
    fork = ET.parse(
        MOLMO
        / "molmo_spaces/data_generation/custom_scenes/pact_place_corridor_v1.xml"
    ).getroot()
    base_names = {
        element.attrib["name"]
        for element in base.iter()
        if "name" in element.attrib
    }
    fork_names = {
        element.attrib["name"]
        for element in fork.iter()
        if "name" in element.attrib
    }
    assert base_names <= fork_names
    assert {
        "place_pedestal",
        "place_receptacle",
        "place_receptacle_floor_g",
        "place_receptacle_floor_visual_g",
        "place_receptacle_lips",
    } <= fork_names
    tray_center_x = 0.35
    tray_half_x = 0.10
    assert tray_center_x + tray_half_x == pytest.approx(0.45)
    assert tray_center_x - tray_half_x == pytest.approx(0.25)
    assert tray_center_x + tray_half_x < 0.58
    assert abs(tray_center_x - 0.14) < 0.30


def test_pick_and_place_success_defaults_and_class_path() -> None:
    from molmo_spaces.configs.task_configs import PickAndPlaceTaskConfig
    from molmo_spaces.tasks.enclosure_reach import PactPlaceCorridorTask
    from molmo_spaces.tasks.pick_and_place_task import PickAndPlaceTask

    assert issubclass(PactPlaceCorridorTask, PickAndPlaceTask)
    config = PickAndPlaceTaskConfig(task_cls=PactPlaceCorridorTask)
    assert config.receptacle_supported_weight_frac == pytest.approx(0.5)
    assert config.max_place_receptacle_pos_displacement == pytest.approx(0.1)
    assert config.max_place_receptacle_rot_displacement == pytest.approx(math.pi / 4)
    assert math.isinf(config.succ_pos_threshold)
    source = _class_source(
        (MOLMO / "molmo_spaces/tasks/pick_and_place_task.py").read_text(),
        "PickAndPlaceTask",
    )
    assert "supported_by_receptacle" in source
    assert "not robot_contact" in source
    assert "succ_pos_threshold" not in source


@pytest.mark.parametrize(
    ("pair", "expected"),
    [
        ({"root2": "cavity_obj_0/Cup_10"}, "grasp_target"),
        ({"root2": "pact_intrusion_left"}, "hazard_bar"),
        ({"root2": "hood_side_l"}, "other_environment"),
        ({"root2": "place_receptacle"}, "place_receptacle"),
        ({"geom2": "place_receptacle_lip_left_g"}, "place_receptacle"),
        ({"root2": "pact_clutter_l0"}, "clutter"),
        ({"root1": "cavity_obj_0/Cup_10", "root2": "pact_clutter_r1"}, "clutter"),
        ({"root2": "pact_clutter_00"}, "clutter"),
        ({"body1": "robot_0/fr3_link5", "root1": "robot_0/", "root2": "pact_clutter_15"}, "clutter"),
    ],
)
def test_contact_taxonomy_adds_only_exempt_receptacle(pair, expected) -> None:
    from molmo_spaces.tasks.pact_place_contact_audit import classify_contact

    assert classify_contact(pair) == expected


def test_contact_summary_exempts_receptacle_but_not_pedestal() -> None:
    from molmo_spaces.tasks.pact_place_contact_audit import PactPlaceContactAudit

    audit = PactPlaceContactAudit()
    audit._pair_totals.update(
        {
            "grasp_target": 3,
            "place_receptacle": 11,
            "hazard_bar": 0,
            "other_environment": 0,
        }
    )
    summary = audit.summary()
    assert summary["collision_free"] is True
    assert summary["place_receptacle_contact_exempt"] is False
    assert summary["place_receptacle_exempt_during_placement_including_preplace"] is True
    audit._pair_totals["other_environment"] = 1
    assert audit.summary()["collision_free"] is False
    audit._pair_totals["other_environment"] = 0
    audit._pair_totals["clutter"] = 2
    assert audit.summary()["collision_free"] is False


def test_place_sampler_and_expert_are_additive_subclasses() -> None:
    from molmo_spaces.policy.solvers.object_manipulation.pick_and_place_planner_policy import (
        PickAndPlacePlannerPolicy,
    )
    from molmo_spaces.tasks.enclosure_reach import (
        PactCollisionCorridorSampler,
        PactPlaceCorridorPolicy,
        PactPlaceCorridorSampler,
    )

    assert issubclass(PactPlaceCorridorSampler, PactCollisionCorridorSampler)
    from molmo_spaces.tasks.enclosure_reach import PactPlaceCorridorV2Sampler
    from molmo_spaces.tasks.enclosure_reach import PactPlaceCorridorV3Sampler
    from molmo_spaces.tasks.enclosure_reach import PactPlaceCorridorV4Sampler

    assert issubclass(PactPlaceCorridorV2Sampler, PactPlaceCorridorSampler)
    assert issubclass(PactPlaceCorridorV3Sampler, PactPlaceCorridorV2Sampler)
    assert issubclass(PactPlaceCorridorV4Sampler, PactPlaceCorridorV3Sampler)
    assert PactPlaceCorridorSampler.PACT_PLACE_ENVIRONMENT_VERSION == (
        "pact_place_corridor_v1"
    )
    assert PactPlaceCorridorSampler.PLACE_RECEPTACLE_START_POSE[:3] == [0.35, 0.0, 0.0]
    assert PactPlaceCorridorV2Sampler.PACT_PLACE_ENVIRONMENT_VERSION == (
        "pact_place_corridor_v2"
    )
    assert PactPlaceCorridorV2Sampler.PLACE_RECEPTACLE_START_POSE[:3] == [
        0.35,
        0.32,
        0.0,
    ]
    assert PactPlaceCorridorV3Sampler.PACT_PLACE_ENVIRONMENT_VERSION == (
        "pact_place_corridor_v3"
    )
    assert PactPlaceCorridorV3Sampler.CLUTTER_BODY_NAMES == (
        "pact_clutter_l0",
        "pact_clutter_l1",
        "pact_clutter_r0",
        "pact_clutter_r1",
    )
    assert PactPlaceCorridorV3Sampler.CLUTTER_SLOT_NOMINAL_XY == {
        "l0": (0.70, 0.34),
        "l1": (0.75, 0.34),
        "r0": (0.70, -0.34),
        "r1": (0.75, -0.34),
    }
    assert PactPlaceCorridorV3Sampler.CLUTTER_HEIGHT_M == pytest.approx(0.10)
    assert PactPlaceCorridorV3Sampler.CLUTTER_HALF_X_M == pytest.approx(0.025)
    assert PactPlaceCorridorV3Sampler.CLUTTER_HALF_Y_M == pytest.approx(0.05)
    assert PactPlaceCorridorV3Sampler.CLUTTER_TOP_Z_M == pytest.approx(0.82)
    assert abs(PactPlaceCorridorV3Sampler.CLUTTER_SLOT_NOMINAL_XY["l0"][1]) - (
        PactPlaceCorridorV3Sampler.CLUTTER_HALF_Y_M
    ) == pytest.approx(0.29)
    v3_slots = PactPlaceCorridorV3Sampler._clutter_slots(
        PactPlaceCorridorV3Sampler.__new__(PactPlaceCorridorV3Sampler), None
    )
    assert v3_slots["l0"]["center_m"] == [0.70, 0.34, 0.77]
    assert v3_slots["l0"]["half_m"] == [0.025, 0.05, 0.05]
    assert "l0" not in PactPlaceCorridorV4Sampler.CLUTTER_SLOT_NOMINAL
    assert tuple(PactPlaceCorridorV4Sampler.CLUTTER_SLOT_NOMINAL) == tuple(
        f"{index:02d}" for index in range(13)
    )
    assert PactPlaceCorridorV4Sampler.CLUTTER_POOL_BODY_NAMES == tuple(
        f"pact_clutter_{index:02d}" for index in range(16)
    )
    assert PactPlaceCorridorV4Sampler.PACT_PLACE_ENVIRONMENT_VERSION == (
        "pact_place_corridor_v4"
    )
    v4_slots = PactPlaceCorridorV4Sampler._clutter_slots(
        PactPlaceCorridorV4Sampler.__new__(PactPlaceCorridorV4Sampler), None
    )
    assert v4_slots["00"]["center_m"][1] == pytest.approx(-0.385)
    assert v4_slots["00"]["center_m"][2] == pytest.approx(1.12)
    assert "l0" not in v4_slots
    assert issubclass(PactPlaceCorridorPolicy, PickAndPlacePlannerPolicy)
    assert PactPlaceCorridorPolicy.OUTBOUND_SAFE_GAP > (
        PactPlaceCorridorPolicy.INBOUND_SAFE_GAP
    )
    assert PactPlaceCorridorPolicy.OUTBOUND_ENVELOPE_HALF_Y > (
        PactPlaceCorridorPolicy.INBOUND_ENVELOPE_HALF_Y
    )
    assert PactPlaceCorridorPolicy.OUTBOUND_CARRY_RAISE_M == pytest.approx(0.0)
    assert PactPlaceCorridorPolicy.OUTBOUND_PASS_SPEED == pytest.approx(0.045)
    assert PactPlaceCorridorPolicy.OUTSIDE_STAGING_X_M < 0.58
    assert PactPlaceCorridorPolicy.GRASP_WORLD_Z_OFFSET_M == pytest.approx(0.0)
    assert PactPlaceCorridorPolicy.RELEASE_CLEARANCE_M == pytest.approx(0.005)
    assert PactPlaceCorridorPolicy.OUTBOUND_APPROACH_MAX_STEP_M == pytest.approx(0.04)
    assert "_get_placement_poses" in PactPlaceCorridorPolicy.__dict__
    assert "_subdivide_tcp_segment" in PactPlaceCorridorPolicy.__dict__
    assert "BOW_SHRINK_STEP_M" not in PactPlaceCorridorPolicy.__dict__
    assert "_bow_magnitudes" not in PactPlaceCorridorPolicy.__dict__
    source = (MOLMO / "molmo_spaces/tasks/enclosure_reach.py").read_text()
    tree = ast.parse(source)
    policy = next(
        item
        for item in tree.body
        if isinstance(item, ast.ClassDef) and item.name == "PactPlaceCorridorPolicy"
    )
    bow = next(
        item
        for item in policy.body
        if isinstance(item, ast.FunctionDef) and item.name == "_bow_segment"
    )
    bow_text = ast.get_source_segment(source, bow)
    assert bow_text is not None
    assert "check_feasible_ik" not in bow_text
    assert "IK failed for required" not in bow_text
    assert "bow_fallback_taken" in bow_text
    assert "RELEASE_CLEARANCE_M" in source


def test_upstream_shared_planner_files_are_unmodified() -> None:
    pick_place = (
        MOLMO
        / "molmo_spaces/policy/solvers/object_manipulation/pick_and_place_planner_policy.py"
    )
    base = (
        MOLMO
        / "molmo_spaces/policy/solvers/object_manipulation/base_object_manipulation_planner_policy.py"
    )
    assert hashlib.sha256(pick_place.read_bytes()).hexdigest() == (
        "9ee369789397add3ae74492e4821993a981940be54cc0579e3d282328a8aa36a"
    )
    assert hashlib.sha256(base.read_bytes()).hexdigest() == (
        "a7ee35704d60b82246fc48db466aaa081a6a83717d983cc062df3988e53893d5"
    )


def test_phase0_contract_is_balanced_self_hashed_and_stops_on_failure() -> None:
    document = contract.build_contract()
    contract.validate_contract(document)
    rows = document["expert_screen_rows"]
    assert document["master_seed"] == contract.DEFAULT_MASTER_SEED
    assert document["master_seed"] == 2026082701
    assert len(rows) == 24
    assert sum(row["intrusion_side"] == "left" for row in rows) == 12
    assert sum(row["intrusion_side"] == "right" for row in rows) == 12
    assert document["phase0_gate"]["minimum_clean_successes"] == 20
    assert document["expert"]["release_clearance_m"] == 0.005
    assert document["expert"]["outbound_approach_max_step_m"] == 0.04
    assert document["expert"]["initial_observation_rejects_robot_environment_contact"] is True
    assert document["expert"]["empty_gripper_disarmed_on_placement_descent"] is True
    assert document["expert"]["empty_gripper_persist_steps"] == 3
    assert document["expert"]["gripper_empty_threshold_m"] == 0.002
    assert document["expert"]["empty_gripper_repair_is_not_a_threshold_change"] is True
    assert document["phase0_gate"]["no_fifth_attempt"] is False
    assert document["phase0_gate"]["fifth_attempt"] == (
        "receptacle_relocation_and_phase_aware_exemption"
    )
    assert document["phase0_gate"]["sixth_attempt"] == "fixed_shelf_clutter"
    assert document["phase0_gate"]["sixth_b_attempt"] == (
        "resite_clutter_laterally_after_cup_contact"
    )
    assert document["phase0_gate"]["sixth_c_attempt"] == (
        "grow_clutter_presence_hold_inner_face"
    )
    assert document["phase0_gate"]["v5_was_phase0_pass_not_a_failure"] is True
    assert document["phase0_gate"]["prior_screens_not_comparable_on_clean_success"] is True
    assert document["phase0_gate"]["v6_clean_success_stricter_than_v5"] is True
    assert document["phase0_gate"]["a0_clearance_probe_zero_clutter"] is True
    assert document["phase0_gate"]["phase0_not_run"] is False
    assert document["phase0_gate"]["do_not_freeze_until_zero_clutter_probe"] is False
    prediction = document["phase0_gate"]["attempt6b_prediction"]
    assert prediction["recorded_before_first_episode"] is True
    assert prediction["predicted_clean_successes"] == [19, 22]
    assert prediction["prior_screens_clean"] == [18, 18, 18, 15, 22]
    assert prediction["bar"] == 20
    prediction_c = document["phase0_gate"]["attempt6c_prediction"]
    assert prediction_c["recorded_before_first_episode"] is True
    assert prediction_c["predicted_clean_successes"] == [19, 22]
    assert prediction_c["prior_screens_clean"] == [18, 18, 18, 15, 22, 20]
    assert prediction_c["bar"] == 20
    assert prediction_c["v6b_clean"] == 20
    assert prediction_c["v6b_clutter_episodes"] == 0
    assert prediction_c["v6c_probe_clutter_episodes"] == 0
    assert document["scene"]["clutter_inner_face_abs_y_m"] == pytest.approx(0.29)
    assert document["scene"]["clutter_half_x_m"] == pytest.approx(0.025)
    assert document["scene"]["clutter_half_y_m"] == pytest.approx(0.05)
    assert document["scene"]["clutter_top_z_m"] == pytest.approx(0.82)
    assert document["scene"]["clutter_nominal_rear_outer_x_m"] == pytest.approx(0.775)
    assert document["scene"]["xml"].endswith("pact_place_corridor_v3.xml")
    assert document["scene"]["sampler_class"] == "PactPlaceCorridorV3Sampler"
    assert document["scene"]["place_receptacle_center_xyz_m"] == [0.35, 0.32, 0.0]
    assert document["scene"]["place_receptacle_contact_exempt"] is False
    assert document["scene"]["place_receptacle_exempt_during_placement_including_preplace"] is True
    assert document["scene"]["clutter_body_prefix"] == "pact_clutter_"
    assert document["scene"]["clutter_drawn_from_task_seed_not_intrusion_side"] is True
    assert "zero clutter entries" in document["phase0_gate"]["clean_success_definition"]
    for row in rows:
        expected_x, expected_y = contract.clutter_jitters_for_seed(row["task_seed_u64"])
        assert row["clutter_x_jitter_m"] == expected_x
        assert row["clutter_y_jitter_m"] == expected_y
    assert document["phase0_gate"]["on_fail"] == (
        "stop_without_collection_or_training"
    )
    assert document["success_criterion"][
        "lift_one_centimetre_criterion_on_success_path"
    ] is False
    source = document["source_sha256"]
    assert source[
        "submodules/molmospaces/molmo_spaces/policy/solvers/object_manipulation/"
        "pick_and_place_planner_policy.py"
    ] == "9ee369789397add3ae74492e4821993a981940be54cc0579e3d282328a8aa36a"
    assert source[
        "submodules/molmospaces/molmo_spaces/policy/solvers/object_manipulation/"
        "base_object_manipulation_planner_policy.py"
    ] == "a7ee35704d60b82246fc48db466aaa081a6a83717d983cc062df3988e53893d5"


def test_generated_contract_reproduces_if_present() -> None:
    found = False
    for relative in (
        "configs/pact_place_corridor_v1.json",
        "configs/pact_place_corridor_v2.json",
        "configs/pact_place_corridor_v3.json",
        "configs/pact_place_corridor_v4.json",
        "configs/pact_place_corridor_v5.json",
        "configs/pact_place_corridor_v6b.json",
        "configs/pact_place_corridor_v6c.json",
    ):
        path = ROOT / relative
        if not path.exists():
            continue
        found = True
        contract.validate_contract(json.loads(path.read_text()))
    if not found:
        pytest.skip("no place-corridor contracts have been generated")


def test_episode_ids_include_master_seed_and_do_not_collide() -> None:
    first = contract.build_contract(master_seed=2026081801)
    second = contract.build_contract(master_seed=2026081901)
    first_ids = {row["episode_id"] for row in first["expert_screen_rows"]}
    second_ids = {row["episode_id"] for row in second["expert_screen_rows"]}
    assert first["master_seed"] == 2026081801
    assert second["master_seed"] == 2026081901
    assert first_ids.isdisjoint(second_ids)
    assert len(first_ids) == 24
    assert len(second_ids) == 24


def test_frozen_v1_v2_episode_ids_collide_and_must_not_be_joined() -> None:
    v1 = json.loads((ROOT / "configs/pact_place_corridor_v1.json").read_text())
    v2 = json.loads((ROOT / "configs/pact_place_corridor_v2.json").read_text())
    shared = {
        row["episode_id"]
        for row in v1["expert_screen_rows"]
    } & {row["episode_id"] for row in v2["expert_screen_rows"]}
    assert len(shared) == 12
    assert v1["config_sha256"] != v2["config_sha256"]


def test_diagnostic_artifacts_contain_no_gate_tokens() -> None:
    diagnostic = ROOT / "diagnostics_output/pact_place_corridor_v2_diagnostic_original_seeds"
    assert diagnostic.is_dir()
    forbidden = (
        "PACT_PLACE_CORRIDOR_PHASE0_PASS",
        "PACT_PLACE_CORRIDOR_PHASE0_FAIL",
        "proceed_to_collection",
        "proceed_",
    )
    for path in diagnostic.rglob("*"):
        if not path.is_file():
            continue
        text = path.read_text(errors="replace")
        for token in forbidden:
            assert token not in text, f"{token} in {path}"
    role = json.loads((diagnostic / "role.json").read_text())
    summary = json.loads((diagnostic / "expert_screen.json").read_text())
    assert role["role"] == "diagnostic_not_a_gate"
    assert role["authorizes_collection"] is False
    assert summary["role"] == "diagnostic_not_a_gate"
    assert summary["authorizes_collection"] is False
    assert summary["next_action"] == "none_diagnostic_only"
    assert summary["gate_frozen_before_execution"] is False
    assert "decision" not in summary
    assert summary["diagnostic_clean_successes"] == 21


def test_failed_phase0_v1_artifacts_remain() -> None:
    config_path = ROOT / "configs/pact_place_corridor_v1.json"
    summary_path = ROOT / "diagnostics_output/pact_place_corridor/expert_screen.json"
    assert config_path.is_file()
    assert summary_path.is_file()
    saved = json.loads(config_path.read_text())
    assert saved["master_seed"] == 2026081801
    assert saved["config_sha256"] == (
        "46dff849dd16eb3b6c0baf169053829bc66203a39866f14a4667e8eaef559e40"
    )
    summary = json.loads(summary_path.read_text())
    assert summary["decision"] == "PACT_PLACE_CORRIDOR_PHASE0_FAIL"
    assert summary["expert_screen_sha256"] == (
        "143ca77a2d1df1a73447078b18100013137a030f18063bcc459bcbee876f325a"
    )
    stop = json.loads(
        (ROOT / "diagnostics_output/pact_place_corridor/stop_record.json").read_text()
    )
    assert stop["decision"] == "PACT_PLACE_CORRIDOR_PHASE0_FAIL"
    assert stop["stop_record_sha256"] == (
        "11d15daef93ef5ad96bb806cf2ef579c75b40787a0745837b394e1dd3bf1426d"
    )
    assert (ROOT / "docs/PACT_PLACE_CORRIDOR_GATE.md").read_text().endswith(
        "PACT_PLACE_CORRIDOR_PHASE0_FAIL\n"
    )


def test_failed_phase0_v2_artifacts_remain() -> None:
    config_path = ROOT / "configs/pact_place_corridor_v2.json"
    summary_path = ROOT / "diagnostics_output/pact_place_corridor_v2/expert_screen.json"
    saved = json.loads(config_path.read_text())
    assert saved["master_seed"] == 2026081901
    assert saved["config_sha256"] == (
        "a0f30725e325a73b5584895a07fa18000fe3645cb63ebd1b4e5a6746bc201c31"
    )
    summary = json.loads(summary_path.read_text())
    assert summary["decision"] == "PACT_PLACE_CORRIDOR_PHASE0_FAIL"
    assert summary["gate"]["clean_successes"] == 18
    assert summary["expert_screen_sha256"] == (
        "544d1406a3fc5ae631305864c2037390d33dd5d05e25c399538d5abdffddf1c6"
    )
    stop = json.loads(
        (ROOT / "diagnostics_output/pact_place_corridor_v2/stop_record.json").read_text()
    )
    assert stop["decision"] == "PACT_PLACE_CORRIDOR_PHASE0_FAIL"
    assert stop["stop_record_sha256"] == (
        "4879d7dd59600a979aaf69494521db67124bc942e1cafd22a3d02a05d5bc56df"
    )
    assert (ROOT / "docs/PACT_PLACE_CORRIDOR_GATE_V2.md").read_text().endswith(
        "PACT_PLACE_CORRIDOR_PHASE0_FAIL\n"
    )


def test_failed_phase0_v3_artifacts_remain() -> None:
    config_path = ROOT / "configs/pact_place_corridor_v3.json"
    summary_path = ROOT / "diagnostics_output/pact_place_corridor_v3/expert_screen.json"
    saved = json.loads(config_path.read_text())
    assert saved["master_seed"] == 2026082001
    assert saved["config_sha256"] == (
        "acd7ced5c7e5a0ea8f6a0070f98d507ea1ee0e983fe416f67273c786e596694f"
    )
    summary = json.loads(summary_path.read_text())
    assert summary["decision"] == "PACT_PLACE_CORRIDOR_PHASE0_FAIL"
    assert summary["gate"]["clean_successes"] == 18
    assert summary["hazard_contact_episodes"] == {"inbound": 0, "outbound": 0}
    assert summary["expert_screen_sha256"] == (
        "05b59f4001a37be773016e338d99216e1180c4919f8da6f62db69dc24f19f7a0"
    )
    stop = json.loads(
        (ROOT / "diagnostics_output/pact_place_corridor_v3/stop_record.json").read_text()
    )
    assert stop["decision"] == "PACT_PLACE_CORRIDOR_PHASE0_FAIL"
    assert stop["stop_record_sha256"] == (
        "e96af8dc9144e85dfef56ef01d70e467cc6e0c68f4f518fb331533ee47247a75"
    )
    assert (ROOT / "docs/PACT_PLACE_CORRIDOR_GATE_V3.md").read_text().endswith(
        "PACT_PLACE_CORRIDOR_PHASE0_FAIL\n"
    )
    assert (ROOT / "docs/PACT_PLACE_ATTEMPT3.md").read_text().endswith(
        "PACT_PLACE_CORRIDOR_PHASE0_FAIL\n"
    )


def test_v2_screen_enclosure_reach_remains_in_git() -> None:
    committed = subprocess.run(
        [
            "git",
            "show",
            "2828751ee6a1fb5ffcaa30d47fda45859f835510:"
            "molmo_spaces/tasks/enclosure_reach.py",
        ],
        cwd=MOLMO,
        check=True,
        capture_output=True,
    ).stdout
    assert hashlib.sha256(committed).hexdigest() == (
        "cb19130709d6961ac3fcf14ae18ee4d18004ea8a3273f2174d0083f53afdadbb"
    )


def test_diagnostic_summarize_cannot_emit_gate_tokens() -> None:
    from run_pact_place_expert_screen import summarize

    contract_doc = json.loads(
        (ROOT / "configs/pact_place_corridor_v1.json").read_text()
    )
    results = []
    for row in contract_doc["expert_screen_rows"]:
        results.append(
            {
                "status": "complete",
                "role_index": row["role_index"],
                "clean_success": True,
                "task_success": True,
                "grasp_phase_success": True,
                "place_phase_success": True,
                "bow_fallback_taken": False,
                "contact_audit": {
                    "inbound_hazard_contact_frames": 0,
                    "outbound_hazard_contact_frames": 0,
                    "contact_class_totals": {"other_environment": 0},
                },
            }
        )
    summary = summarize(
        contract_doc,
        results,
        1,
        ROOT / "diagnostics_output/pact_place_corridor_v2_diagnostic_original_seeds",
        role="diagnostic",
    )
    dumped = json.dumps(summary)
    assert "PACT_PLACE_CORRIDOR_PHASE0_PASS" not in dumped
    assert "PACT_PLACE_CORRIDOR_PHASE0_FAIL" not in dumped
    assert "proceed_" not in dumped
    assert summary["authorizes_collection"] is False
    assert summary["next_action"] == "none_diagnostic_only"
    assert summary["diagnostic_clean_successes"] == 24


def test_outbound_approach_subdivision_caps_cartesian_step() -> None:
    from molmo_spaces.policy.solvers.object_manipulation.base_object_manipulation_planner_policy import (
        TCPMoveSegment,
    )
    from molmo_spaces.tasks.enclosure_reach import PactPlaceCorridorPolicy

    start = np.eye(4)
    end = np.eye(4)
    end[:3, 3] = [0.12, -0.08, 0.0]
    segment = TCPMoveSegment(
        name="outbound_approach",
        start_pose=start,
        end_pose=end,
        speed=0.045,
    )
    pieces = PactPlaceCorridorPolicy._subdivide_tcp_segment(segment, 0.04)
    assert len(pieces) >= 4
    assert all(piece.name == "outbound_approach" for piece in pieces)
    for piece in pieces:
        dist = float(
            np.linalg.norm(piece.end_pose[:3, 3] - piece.start_pose[:3, 3])
        )
        assert dist <= 0.04 + 1e-9
    np.testing.assert_allclose(pieces[0].start_pose[:3, 3], start[:3, 3])
    np.testing.assert_allclose(pieces[-1].end_pose[:3, 3], end[:3, 3])

    axis = np.eye(4)
    axis[0, 3] = 0.12
    axis_pieces = PactPlaceCorridorPolicy._subdivide_tcp_segment(
        TCPMoveSegment(
            name="outbound_approach",
            start_pose=np.eye(4),
            end_pose=axis,
            speed=0.045,
        ),
        PactPlaceCorridorPolicy.OUTBOUND_APPROACH_MAX_STEP_M,
    )
    assert len(axis_pieces) == 3
    assert all(piece.name == "outbound_approach" for piece in axis_pieces)


def test_disallowed_initial_contacts_reject_panel_not_tray() -> None:
    from run_pact_place_expert_screen import disallowed_initial_contacts

    pairs = [
        {
            "geom1": "pact_intrusion_right_g",
            "geom2": "robot_0/fr3_link7_collision",
            "body1": "pact_intrusion_right",
            "body2": "robot_0/fr3_link7",
            "root1": "pact_intrusion_right",
            "root2": "robot_0/base",
            "distance_m": -0.02,
        },
        {
            "geom1": "place_receptacle_geom",
            "geom2": "robot_0/fr3_link7_collision",
            "body1": "place_receptacle",
            "body2": "robot_0/fr3_link7",
            "root1": "place_receptacle",
            "root2": "robot_0/base",
            "distance_m": -0.001,
        },
    ]
    rejected = disallowed_initial_contacts(pairs)
    assert len(rejected) == 1
    assert rejected[0]["body1"] == "pact_intrusion_right"


def test_endpoint_scalars_required_before_payload_deletion() -> None:
    from run_pact_place_expert_screen import (
        ENDPOINT_SCALAR_KEYS,
        assert_endpoint_scalars_emitted,
    )

    with pytest.raises(ValueError, match="endpoint_scalars missing"):
        assert_endpoint_scalars_emitted({"contact_frames": [1]})
    with pytest.raises(ValueError, match="missing keys"):
        assert_endpoint_scalars_emitted({"endpoint_scalars": {}})
    assert_endpoint_scalars_emitted(
        {"endpoint_scalars": {key: None for key in ENDPOINT_SCALAR_KEYS}}
    )


def test_hazard_rows_6_12_are_initial_state_overlap() -> None:
    analysis = json.loads(
        (
            ROOT
            / "diagnostics_output/pact_place_corridor_v2_hazard_rows_6_12_frames/analysis.json"
        ).read_text()
    )
    assert analysis["outcome"] == 1
    assert analysis["pause_phase0_for_inbound_scraping"] is False
    assert analysis["route_a_exists"] is True
    assert (ROOT / "docs/PACT_PLACE_HAZARD_ROWS_6_12.md").read_text().count(
        "Outcome 1."
    ) >= 1


def test_attempt3_runner_rejects_initial_panel_contact() -> None:
    source = (ROOT / "scripts/run_pact_place_expert_screen.py").read_text()
    assert "initial_robot_environment_contact" in source
    assert "assert_endpoint_scalars_emitted" in source
    assert "OUTBOUND_APPROACH_MAX_STEP_M" not in source
    policy_source = (
        MOLMO / "molmo_spaces/tasks/enclosure_reach.py"
    ).read_text()
    assert "_subdivide_tcp_segment" in policy_source
    assert "OUTBOUND_APPROACH_MAX_STEP_M" in policy_source


def test_frozen_v1_v2_contracts_omit_attempt3_expert_keys() -> None:
    for relative in (
        "configs/pact_place_corridor_v1.json",
        "configs/pact_place_corridor_v2.json",
    ):
        expert = json.loads((ROOT / relative).read_text())["expert"]
        assert "outbound_approach_max_step_m" not in expert
        assert "initial_observation_rejects_robot_environment_contact" not in expert


def test_frozen_v3_contract_if_present() -> None:
    path = ROOT / "configs/pact_place_corridor_v3.json"
    if not path.exists():
        pytest.skip("v3 contract has not been generated")
    saved = json.loads(path.read_text())
    contract.validate_contract(saved)
    assert saved["master_seed"] == 2026082001
    assert saved["config_sha256"] == (
        "acd7ced5c7e5a0ea8f6a0070f98d507ea1ee0e983fe416f67273c786e596694f"
    )
    assert saved["phase0_gate"]["minimum_clean_successes"] == 20
    assert saved["expert"]["release_clearance_m"] == 0.005
    assert saved["expert"]["outbound_approach_max_step_m"] == 0.04
    assert saved["expert"]["initial_observation_rejects_robot_environment_contact"] is True
    v1_ids = {
        row["episode_id"]
        for row in json.loads((ROOT / "configs/pact_place_corridor_v1.json").read_text())[
            "expert_screen_rows"
        ]
    }
    v2_ids = {
        row["episode_id"]
        for row in json.loads((ROOT / "configs/pact_place_corridor_v2.json").read_text())[
            "expert_screen_rows"
        ]
    }
    v3_ids = {row["episode_id"] for row in saved["expert_screen_rows"]}
    assert v3_ids.isdisjoint(v1_ids)
    assert v3_ids.isdisjoint(v2_ids)
    assert len(v3_ids) == 24


def test_failed_phase0_v3_artifacts_remain() -> None:
    config_path = ROOT / "configs/pact_place_corridor_v3.json"
    summary_path = ROOT / "diagnostics_output/pact_place_corridor_v3/expert_screen.json"
    saved = json.loads(config_path.read_text())
    assert saved["master_seed"] == 2026082001
    assert saved["config_sha256"] == (
        "acd7ced5c7e5a0ea8f6a0070f98d507ea1ee0e983fe416f67273c786e596694f"
    )
    summary = json.loads(summary_path.read_text())
    assert summary["decision"] == "PACT_PLACE_CORRIDOR_PHASE0_FAIL"
    assert summary["gate"]["clean_successes"] == 18
    assert summary["task_success"]["count"] == 18
    assert summary["hazard_contact_episodes"]["inbound"] == 0
    assert summary["hazard_contact_episodes"]["outbound"] == 0
    assert summary["bow_fallback_episodes"] == 0
    assert summary["expert_screen_sha256"] == (
        "05b59f4001a37be773016e338d99216e1180c4919f8da6f62db69dc24f19f7a0"
    )
    stop = json.loads(
        (ROOT / "diagnostics_output/pact_place_corridor_v3/stop_record.json").read_text()
    )
    assert stop["decision"] == "PACT_PLACE_CORRIDOR_PHASE0_FAIL"
    assert stop["stop_record_sha256"] == (
        "e96af8dc9144e85dfef56ef01d70e467cc6e0c68f4f518fb331533ee47247a75"
    )
    assert (ROOT / "docs/PACT_PLACE_CORRIDOR_GATE_V3.md").read_text().endswith(
        "PACT_PLACE_CORRIDOR_PHASE0_FAIL\n"
    )
    attempt3 = (ROOT / "docs/PACT_PLACE_ATTEMPT3.md").read_text()
    assert attempt3.count("PACT_PLACE_CORRIDOR_PHASE0_FAIL") >= 4
    assert "no fourth iteration" in attempt3.lower()


def test_v3_rows_emit_endpoint_scalars_and_trajectories() -> None:
    from run_pact_place_expert_screen import ENDPOINT_SCALAR_KEYS

    rows = ROOT / "diagnostics_output/pact_place_corridor_v3/expert_screen_rows"
    result_dirs = sorted(path for path in rows.iterdir() if path.is_dir())
    assert len(result_dirs) == 24
    failed = []
    for directory in result_dirs:
        result = json.loads((directory / "result.json").read_text())
        block = result["endpoint_scalars"]
        assert set(ENDPOINT_SCALAR_KEYS) <= set(block)
        assert block["endpoint_values_emitted_during_compaction"] is True
        assert all(block[key] is not None for key in ENDPOINT_SCALAR_KEYS)
        assert (directory / "trajectory.json").is_file()
        assert (directory / "initial_observation_accepted.json").is_file()
        if not result["clean_success"]:
            failed.append(result["role_index"])
    assert failed == [2, 6, 9, 12, 20, 21]


def _pose(z: float) -> np.ndarray:
    pose = np.eye(4)
    pose[2, 3] = z
    return pose


class _FakeGripper:
    def __init__(self, dist: float) -> None:
        self.inter_finger_dist = dist
        self.inter_finger_dist_range = (0.0, 0.08)
        self.leaf_frame_to_world = np.eye(4)


class _FakeRobotView:
    def __init__(self, gripper: _FakeGripper) -> None:
        self._gripper = gripper
        self.mj_data = type("MjData", (), {"time": 0.0})()

    def get_gripper_movegroup_ids(self) -> list[str]:
        return ["gripper"]

    def get_gripper(self, _move_group_id: str) -> _FakeGripper:
        return self._gripper


def _place_sequence(
    *,
    names: tuple[str, ...] = ("outbound_approach",),
    holding: bool = True,
    dist: float = 0.008,
    pos_threshold: float = 0.1,
):
    from molmo_spaces.policy.solvers.object_manipulation.base_object_manipulation_planner_policy import (
        TCPMoveSegment,
    )
    from molmo_spaces.tasks.enclosure_reach import PactPlaceTCPMoveSequence

    gripper = _FakeGripper(dist)
    view = _FakeRobotView(gripper)
    segments = [
        TCPMoveSegment(
            name=name,
            start_pose=_pose(0.0),
            end_pose=_pose(0.05),
            speed=0.05,
        )
        for name in names
    ]
    sequence = PactPlaceTCPMoveSequence(
        view,
        lambda _mg, _pose: {},
        0.0,
        segments,
        is_holding_object=holding,
        gripper_empty_threshold=0.002,
        tcp_pos_err_threshold=pos_threshold,
        tcp_rot_err_threshold=1.0,
    )
    sequence.move_seg_idx = 0
    sequence.move_seg_start_time = 0.0
    return sequence, gripper


def test_empty_gripper_disarmed_on_placement_descent() -> None:
    sequence, gripper = _place_sequence(names=("placement_descent",), dist=0.0)
    for _ in range(5):
        gripper.inter_finger_dist = 0.0
        assert sequence.check_failure() is not True
    assert sequence._empty_gripper_streak == 0


def test_empty_gripper_requires_three_consecutive_transport_samples() -> None:
    sequence, gripper = _place_sequence(names=("outbound_approach",), dist=0.0)
    gripper.inter_finger_dist = 0.0
    assert not sequence.check_failure()
    assert sequence._empty_gripper_streak == 1
    assert not sequence.check_failure()
    assert sequence._empty_gripper_streak == 2
    assert sequence.check_failure()
    assert sequence._empty_gripper_streak == 3


def test_empty_gripper_streak_resets_after_a_nonempty_sample() -> None:
    sequence, gripper = _place_sequence(names=("outbound_pass",), dist=0.0)
    gripper.inter_finger_dist = 0.0
    assert not sequence.check_failure()
    assert not sequence.check_failure()
    gripper.inter_finger_dist = 0.0085
    assert not sequence.check_failure()
    assert sequence._empty_gripper_streak == 0
    gripper.inter_finger_dist = 0.0
    assert not sequence.check_failure()
    assert sequence._empty_gripper_streak == 1


def test_tracking_failure_still_fires_on_a_single_sample() -> None:
    sequence, gripper = _place_sequence(
        names=("outbound_approach",), dist=0.0085, pos_threshold=0.01
    )
    gripper.leaf_frame_to_world = _pose(1.0)
    assert sequence.check_failure()
    assert sequence._empty_gripper_streak == 0


def test_sequence_construction_uses_the_place_subclass() -> None:
    source = (
        MOLMO / "molmo_spaces/tasks/enclosure_reach.py"
    ).read_text()
    policy = ast.parse(source)
    cls = next(
        item
        for item in policy.body
        if isinstance(item, ast.ClassDef) and item.name == "PactPlaceCorridorPolicy"
    )
    sequence = next(
        item
        for item in cls.body
        if isinstance(item, ast.FunctionDef) and item.name == "_sequence"
    )
    text = ast.get_source_segment(source, sequence)
    assert text is not None
    assert "PactPlaceTCPMoveSequence" in text
    assert "return TCPMoveSequence(" not in text


def test_empty_gripper_repair_constants_match_the_contract() -> None:
    from molmo_spaces.tasks.enclosure_reach import PactPlaceTCPMoveSequence

    document = contract.build_contract(master_seed=2026082101)
    assert (
        document["expert"]["empty_gripper_persist_steps"]
        == PactPlaceTCPMoveSequence.EMPTY_GRIPPER_PERSIST_STEPS
        == 3
    )
    assert document["expert"]["empty_gripper_disarmed_on_placement_descent"] is True
    assert PactPlaceTCPMoveSequence.EMPTY_GRIPPER_DISARMED_SEGMENTS == frozenset(
        {"placement_descent"}
    )
    assert document["expert"]["gripper_empty_threshold_m"] == 0.002


def test_frozen_v4_contract_if_present() -> None:
    path = ROOT / "configs/pact_place_corridor_v4.json"
    if not path.exists():
        pytest.skip("v4 contract has not been generated")
    saved = json.loads(path.read_text())
    contract.validate_contract(saved)
    assert saved["master_seed"] == 2026082101
    assert saved["phase0_gate"]["minimum_clean_successes"] == 20
    assert saved["expert"]["empty_gripper_persist_steps"] == 3
    assert saved["expert"]["empty_gripper_disarmed_on_placement_descent"] is True
    assert saved["expert"]["gripper_empty_threshold_m"] == 0.002
    assert saved["phase0_gate"]["attempt4_prediction"][
        "recorded_before_first_episode"
    ] is True
    assert saved["phase0_gate"]["attempt4_prediction"]["predicted_clean_successes"] == [
        19,
        22,
    ]
    v3_ids = {
        row["episode_id"]
        for row in json.loads((ROOT / "configs/pact_place_corridor_v3.json").read_text())[
            "expert_screen_rows"
        ]
    }
    v4_ids = {row["episode_id"] for row in saved["expert_screen_rows"]}
    assert v4_ids.isdisjoint(v3_ids)
    assert len(v4_ids) == 24


def test_failed_phase0_v4_artifacts_remain() -> None:
    config_path = ROOT / "configs/pact_place_corridor_v4.json"
    summary_path = ROOT / "diagnostics_output/pact_place_corridor_v4/expert_screen.json"
    saved = json.loads(config_path.read_text())
    assert saved["master_seed"] == 2026082101
    assert saved["config_sha256"] == (
        "fe45435d55cda8daee71972451ce8b460e641b71558aa8a69b3a4686319cdc65"
    )
    assert saved["phase0_gate"]["attempt4_prediction"]["predicted_clean_successes"] == [
        19,
        22,
    ]
    summary = json.loads(summary_path.read_text())
    assert summary["decision"] == "PACT_PLACE_CORRIDOR_PHASE0_FAIL"
    assert summary["gate"]["clean_successes"] == 15
    assert summary["task_success"]["count"] == 15
    assert summary["place_success_given_grasp"] == {"numerator": 15, "denominator": 15}
    assert summary["hazard_contact_episodes"] == {"inbound": 0, "outbound": 0}
    assert summary["bow_fallback_episodes"] == 0
    assert summary["next_action"] == "stop_without_collection_or_training"
    assert summary["expert_screen_sha256"] == (
        "1f19a02c945c6a96370b1cfddbd2850102c5d27983460e4dcb8c75000a244f41"
    )
    stop = json.loads(
        (ROOT / "diagnostics_output/pact_place_corridor_v4/stop_record.json").read_text()
    )
    assert stop["decision"] == "PACT_PLACE_CORRIDOR_PHASE0_FAIL"
    assert stop["stop_record_sha256"] == (
        "484f59c64f41ff316bf1ef534ad91e0ca58bcead25103044a025e77f2fd5b4da"
    )
    assert (ROOT / "docs/PACT_PLACE_CORRIDOR_GATE_V4.md").read_text().endswith(
        "PACT_PLACE_CORRIDOR_PHASE0_FAIL\n"
    )
    attempt4 = (ROOT / "docs/PACT_PLACE_ATTEMPT4.md").read_text()
    assert attempt4.endswith("PACT_PLACE_CORRIDOR_PHASE0_FAIL\n")
    assert attempt4.count("PACT_PLACE_CORRIDOR_PHASE0_FAIL") >= 4
    assert "15/24" in attempt4
    assert "no fifth" in attempt4.lower()
    for prior in (
        "docs/PACT_PLACE_CORRIDOR_GATE.md",
        "docs/PACT_PLACE_CORRIDOR_GATE_V2.md",
        "docs/PACT_PLACE_CORRIDOR_GATE_V3.md",
    ):
        text = (ROOT / prior).read_text()
        assert text.endswith("PACT_PLACE_CORRIDOR_PHASE0_FAIL\n")


def test_v4_rows_emit_endpoint_scalars_and_trajectories() -> None:
    from run_pact_place_expert_screen import ENDPOINT_SCALAR_KEYS

    rows = ROOT / "diagnostics_output/pact_place_corridor_v4/expert_screen_rows"
    result_dirs = sorted(path for path in rows.iterdir() if path.is_dir())
    assert len(result_dirs) == 24
    failed = []
    empty_gripper = 0
    for directory in result_dirs:
        result = json.loads((directory / "result.json").read_text())
        block = result["endpoint_scalars"]
        assert set(ENDPOINT_SCALAR_KEYS) <= set(block)
        assert block["endpoint_values_emitted_during_compaction"] is True
        assert all(block[key] is not None for key in ENDPOINT_SCALAR_KEYS)
        assert (directory / "trajectory.json").is_file()
        assert (directory / "initial_observation_accepted.json").is_file()
        if (result.get("terminal_tracking") or {}).get("check_failure_branch") == (
            "empty_gripper"
        ):
            empty_gripper += 1
        if not result["clean_success"]:
            failed.append(result["role_index"])
    assert failed == [5, 7, 8, 9, 10, 15, 18, 20, 22]
    assert empty_gripper == 0


def test_preplace_counts_as_placement_for_tray_exemption() -> None:
    from molmo_spaces.tasks.enclosure_reach import PactPlaceCorridorPolicy
    from molmo_spaces.tasks.pact_place_contact_audit import (
        PactPlaceContactAudit,
        disallowed_place_receptacle_contact_entries,
    )

    assert PactPlaceCorridorPolicy._traversal_phase("preplace") == "placement"
    assert PactPlaceCorridorPolicy._traversal_phase("placement_descent") == "placement"
    assert PactPlaceCorridorPolicy._traversal_phase("outbound_approach") == "outbound"
    assert PactPlaceCorridorPolicy._traversal_phase("pregrasp") == "inbound"
    audit = PactPlaceContactAudit()
    audit._phase_pair_totals["inbound"]["place_receptacle"] = 4
    audit._phase_pair_totals["outbound"]["place_receptacle"] = 0
    audit._phase_pair_totals["placement"]["place_receptacle"] = 9
    summary = audit.summary()
    assert summary["place_receptacle_outside_placement_entries"] == 4
    assert disallowed_place_receptacle_contact_entries(summary) == 4


def test_v2_xml_only_relocates_and_shrinks_the_receptacle() -> None:
    v1 = (
        MOLMO
        / "molmo_spaces/data_generation/custom_scenes/pact_place_corridor_v1.xml"
    ).read_text()
    v2 = (
        MOLMO
        / "molmo_spaces/data_generation/custom_scenes/pact_place_corridor_v2.xml"
    ).read_text()
    assert 'pos="0.35 0 0"' in v1
    assert 'size="0.10 0.16 0.02"' in v1
    restored = (
        v2.replace("pact_place_corridor_v2", "pact_place_corridor_v1")
        .replace('pos="0.35 0.32 0"', 'pos="0.35 0 0"')
        .replace('size="0.10 0.10 0.05"', 'size="0.10 0.15 0.05"')
        .replace('size="0.07 0.07 0.29"', 'size="0.07 0.11 0.29"')
        .replace('size="0.10 0.10 0.02"', 'size="0.10 0.16 0.02"')
        .replace('pos="0 0.085 0.745"', 'pos="0 0.145 0.745"')
        .replace('pos="0 -0.085 0.745"', 'pos="0 -0.145 0.745"')
    )
    assert restored == v1


def test_a0_sweep_chose_shrunk_centre_with_8cm_clearance() -> None:
    analysis = json.loads(
        (
            ROOT
            / "diagnostics_output/pact_place_reachability_sweep/analysis.json"
        ).read_text()
    )
    assert analysis["no_rollouts"] is True
    assert analysis["n_candidates"] == 26
    assert analysis["n_reachable"] == 26
    assert analysis["n_eligible"] == 3
    chosen = analysis["chosen"]
    assert chosen["center_xy_m"] == [0.35, 0.32]
    assert chosen["footprint"] == "shrunk_0.10x0.10"
    assert chosen["clearance_beyond_traversal_y_m"] == pytest.approx(0.113)
    payload = dict(analysis)
    digest = payload.pop("sweep_sha256")
    assert digest == contract.sha256_payload(payload)
    assert digest == (
        "b657da019b8638ba8b94e1bfa64a1d31ddfa7c27d7a7c6f4b6f22824602e211b"
    )


def test_a0_clutter_sweep_chose_symmetric_y022_height06() -> None:
    analysis = json.loads(
        (ROOT / "diagnostics_output/pact_place_clutter_sweep/analysis.json").read_text()
    )
    assert analysis["no_rollouts"] is True
    assert analysis["n_candidate_sets"] == 30
    assert analysis["n_footprint_ok"] == 12
    assert analysis["n_eligible"] == 12
    chosen = analysis["chosen"]
    assert chosen["height_m"] == pytest.approx(0.06)
    assert chosen["half_xy_m"] == pytest.approx(0.03)
    assert chosen["min_target_envelope_gap_m"] == pytest.approx(0.1)
    assert chosen["slots_xy_m"] == {
        "l0": [0.72, 0.22],
        "l1": [0.78, 0.22],
        "r0": [0.72, -0.22],
        "r1": [0.78, -0.22],
    }
    payload = dict(analysis)
    digest = payload.pop("sweep_sha256")
    assert digest == contract.sha256_payload(payload)
    assert digest == (
        "e34038b9e4a32e5b84729f62d5dc1a851b40c3ad2aa11b6d79bccc461c3526ae"
    )
    failed_ids = [
        item["set_id"]
        for item in analysis["candidates"]
        if not item["eligible"]
    ]
    assert failed_ids
    assert all(
        not analysis["candidates"][i]["footprint_ok"] for i in failed_ids
    )


def test_v6b_step0_clutter_contact_is_carried_cup() -> None:
    analysis = json.loads(
        (
            ROOT
            / "diagnostics_output/pact_place_corridor_v6b_contact_bodies/analysis.json"
        ).read_text()
    )
    assert analysis["role"] == "diagnostic_not_a_gate"
    assert analysis["authorizes_collection"] is False
    assert analysis["axis"] == "lateral"
    assert analysis["next_grid_abs_y_m"] == [0.32, 0.36, 0.40]
    assert len(analysis["rows"]) == 2
    for row in analysis["rows"]:
        assert row["other_body"].startswith("cavity_obj_")
        assert row["robot_link_pairs"] == 0
        assert row["traversal_phase"] == "outbound"
        assert "pact_clutter_" in row["clutter_geom"]


def test_a0b_clutter_sweep_chose_symmetric_y032_height06() -> None:
    analysis = json.loads(
        (
            ROOT / "diagnostics_output/pact_place_clutter_sweep_v6b/analysis.json"
        ).read_text()
    )
    assert analysis["no_rollouts"] is True
    assert analysis["ik_from_expert_trajectory_not_reset_qpos"] is True
    assert analysis["n_candidate_sets"] == 30
    assert analysis["n_footprint_ok"] == 30
    assert analysis["n_eligible"] == 30
    chosen = analysis["chosen"]
    assert chosen["height_m"] == pytest.approx(0.06)
    assert chosen["half_xy_m"] == pytest.approx(0.03)
    assert chosen["slots_xy_m"] == {
        "l0": [0.72, 0.32],
        "l1": [0.78, 0.32],
        "r0": [0.72, -0.32],
        "r1": [0.78, -0.32],
    }
    payload = dict(analysis)
    digest = payload.pop("sweep_sha256")
    assert digest == contract.sha256_payload(payload)
    assert digest == (
        "388c23677516a65431b51e28d4a22becaa0b2a491c52f730f37b73e00b8e66f3"
    )
    # Live sampler is v6c geometry; v6b is frozen in analysis.json, not in class constants.


def test_a0c_clutter_sweep_holds_inner_face_and_clears_back_wall() -> None:
    analysis = json.loads(
        (
            ROOT / "diagnostics_output/pact_place_clutter_sweep_v6c/analysis.json"
        ).read_text()
    )
    assert analysis["no_rollouts"] is True
    assert analysis["ik_from_expert_trajectory_not_reset_qpos"] is True
    assert analysis["inner_face_held_at_abs_y_m"] == pytest.approx(0.29)
    assert analysis["boxes_not_shrunk"] is True
    assert analysis["v6_sweep_untouched"] is True
    assert analysis["v6b_sweep_untouched"] is True
    assert analysis["n_candidate_sets"] == 2
    assert analysis["n_footprint_ok"] == 2
    assert analysis["n_enclosure_ok"] == 2
    assert analysis["n_eligible"] == 2
    assert analysis["max_outer_x_m"] == pytest.approx(0.775)
    assert analysis["shallowest_back_wall_x_m"] == pytest.approx(0.78)
    chosen = analysis["chosen"]
    assert chosen["top_z_m"] == pytest.approx(0.82)
    assert chosen["height_m"] == pytest.approx(0.10)
    assert chosen["half_x_m"] == pytest.approx(0.025)
    assert chosen["half_y_m"] == pytest.approx(0.05)
    assert chosen["min_inner_face_abs_y_m"] == pytest.approx(0.29)
    assert chosen["max_outer_face_x_m"] == pytest.approx(0.775)
    assert chosen["max_outer_face_abs_y_m"] == pytest.approx(0.39)
    assert chosen["slots_xy_m"] == {
        "l0": [0.7, 0.34],
        "l1": [0.75, 0.34],
        "r0": [0.7, -0.34],
        "r1": [0.75, -0.34],
    }
    for item in analysis["candidates"]:
        assert item["eligible"] is True
        assert item["min_inner_face_abs_y_m"] == pytest.approx(0.29)
        assert item["max_outer_face_x_m"] == pytest.approx(0.775)
        for slot, spec in item["slots"].items():
            assert spec["inner_face_ok"] is True
            assert spec["outer_x_ok"] is True
            assert spec["clear_of_shallowest_back_wall"] is True
            assert spec["outer_face_x_m"] <= 0.775 + 1e-12
            if slot.endswith("1"):
                assert spec["outer_face_x_m"] == pytest.approx(0.775)
    payload = dict(analysis)
    digest = payload.pop("sweep_sha256")
    assert digest == contract.sha256_payload(payload)
    assert digest == (
        "714e0ce8c69b71141207cad5dfe02808a7aa753603a22ec6296ca89bacfd14da"
    )
    v6 = json.loads(
        (ROOT / "diagnostics_output/pact_place_clutter_sweep/analysis.json").read_text()
    )
    v6b = json.loads(
        (
            ROOT / "diagnostics_output/pact_place_clutter_sweep_v6b/analysis.json"
        ).read_text()
    )
    assert v6["sweep_sha256"] == (
        "e34038b9e4a32e5b84729f62d5dc1a851b40c3ad2aa11b6d79bccc461c3526ae"
    )
    assert v6b["sweep_sha256"] == (
        "388c23677516a65431b51e28d4a22becaa0b2a491c52f730f37b73e00b8e66f3"
    )


def test_frozen_v5_contract_if_present() -> None:
    path = ROOT / "configs/pact_place_corridor_v5.json"
    if not path.exists():
        pytest.skip("v5 contract has not been generated")
    saved = json.loads(path.read_text())
    contract.validate_contract(saved)
    assert saved["master_seed"] == 2026082201
    assert saved["config_sha256"] == (
        "bd47f1c97d2815657211085590657f5211ca847b776f6039c9617f990da9c1f1"
    )
    assert saved["phase0_gate"]["minimum_clean_successes"] == 20
    assert saved["phase0_gate"]["attempt5_prediction"][
        "recorded_before_first_episode"
    ] is True
    assert saved["phase0_gate"]["attempt5_prediction"]["predicted_clean_successes"] == [
        20,
        23,
    ]
    assert saved["scene"]["place_receptacle_center_xyz_m"] == [0.35, 0.32, 0.0]
    assert saved["scene"]["place_receptacle_contact_exempt"] is False
    v4_ids = {
        row["episode_id"]
        for row in json.loads((ROOT / "configs/pact_place_corridor_v4.json").read_text())[
            "expert_screen_rows"
        ]
    }
    v5_ids = {row["episode_id"] for row in saved["expert_screen_rows"]}
    assert v5_ids.isdisjoint(v4_ids)
    assert len(v5_ids) == 24


def test_frozen_v6b_contract_if_present() -> None:
    path = ROOT / "configs/pact_place_corridor_v6b.json"
    if not path.exists():
        pytest.skip("v6b contract has not been generated")
    saved = json.loads(path.read_text())
    contract.validate_contract(saved)
    assert saved["master_seed"] == 2026082501
    assert saved["config_sha256"] == (
        "ebf1be0359b6ff810772a3d4bbb0adf8913710fae4eec14c88a8f403a70e3671"
    )
    assert saved["phase0_gate"]["minimum_clean_successes"] == 20
    assert saved["phase0_gate"]["attempt6b_prediction"][
        "recorded_before_first_episode"
    ] is True
    assert saved["phase0_gate"]["attempt6b_prediction"]["predicted_clean_successes"] == [
        19,
        22,
    ]
    assert saved["phase0_gate"]["a0_clearance_probe_zero_clutter"] is True
    assert saved["scene"]["place_receptacle_center_xyz_m"] == [0.35, 0.32, 0.0]
    assert saved["phase0_gate"]["clutter_sweep_v6b"]["chosen_slots_xy_m"] == {
        "l0": [0.72, 0.32],
        "l1": [0.78, 0.32],
        "r0": [0.72, -0.32],
        "r1": [0.78, -0.32],
    }
    v5_ids = {
        row["episode_id"]
        for row in json.loads((ROOT / "configs/pact_place_corridor_v5.json").read_text())[
            "expert_screen_rows"
        ]
    }
    v6b_ids = {row["episode_id"] for row in saved["expert_screen_rows"]}
    assert v6b_ids.isdisjoint(v5_ids)
    assert len(v6b_ids) == 24


def test_frozen_v6c_contract_if_present() -> None:
    path = ROOT / "configs/pact_place_corridor_v6c.json"
    if not path.exists():
        pytest.skip("v6c contract has not been generated")
    saved = json.loads(path.read_text())
    contract.validate_contract(saved)
    assert saved["master_seed"] == 2026082701
    assert saved["config_sha256"] == (
        "1dbef6cdd38c6c273d389cbe75717229f78a416831964f9907b00ce4ec58f04e"
    )
    assert saved["phase0_gate"]["minimum_clean_successes"] == 20
    assert saved["phase0_gate"]["attempt6c_prediction"][
        "recorded_before_first_episode"
    ] is True
    assert saved["phase0_gate"]["attempt6c_prediction"]["predicted_clean_successes"] == [
        19,
        22,
    ]
    assert saved["phase0_gate"]["a0_clearance_probe_zero_clutter"] is True
    assert saved["scene"]["place_receptacle_center_xyz_m"] == [0.35, 0.32, 0.0]
    assert saved["scene"]["clutter_inner_face_abs_y_m"] == pytest.approx(0.29)
    assert saved["scene"]["clutter_nominal_rear_outer_x_m"] == pytest.approx(0.775)
    assert saved["phase0_gate"]["clutter_sweep_v6c"]["chosen_top_z_m"] == pytest.approx(
        0.82
    )
    assert saved["phase0_gate"]["clearance_probe_v6c"]["clutter_contact_episodes"] == 0
    # Frozen v6c remains the source of truth. Live rebuild changed when
    # PactPlaceCorridorV4Sampler was added to enclosure_reach.py (hashed in
    # _source_hashes). Do not regenerate configs/pact_place_corridor_v6c.json.
    live = contract.build_contract(master_seed=2026082701)
    assert live["master_seed"] == 2026082701
    assert live["scene"]["xml"].endswith("pact_place_corridor_v3.xml")
    assert live["scene"]["sampler_class"] == "PactPlaceCorridorV3Sampler"
    assert live["scene"]["clutter_inner_face_abs_y_m"] == pytest.approx(0.29)
    assert live["config_sha256"] != saved["config_sha256"]
    v6b_ids = {
        row["episode_id"]
        for row in json.loads((ROOT / "configs/pact_place_corridor_v6b.json").read_text())[
            "expert_screen_rows"
        ]
    }
    v6c_ids = {row["episode_id"] for row in saved["expert_screen_rows"]}
    assert v6c_ids.isdisjoint(v6b_ids)
    assert len(v6c_ids) == 24


def test_v6_clutter_probes_did_not_meet_zero_contact() -> None:
    probes = [
        (
            ROOT / "diagnostics_output/pact_place_corridor_v6_clearance_probe",
            "0f4d80580e1aa2963906378c377e7d61e4bee65afa719e74c7ca9e799b9778d8",
            6,
            1,
        ),
        (
            ROOT / "diagnostics_output/pact_place_corridor_v6_clearance_probe_y028",
            "eea681436be34f03e3ccffa6db2cbf7a8845ad15492363fa8f8e617f7a8430be",
            3,
            4,
        ),
    ]
    for root, config_sha, clutter_episodes, clean in probes:
        summary = json.loads((root / "expert_screen.json").read_text())
        assert summary["role"] == "diagnostic_not_a_gate"
        assert summary["config_sha256"] == config_sha
        assert summary["n"] == 8
        assert summary["clutter_contact_episodes"] == clutter_episodes
        assert summary["diagnostic_clean_successes"] == clean
        assert summary["hazard_contact_episodes"] == {"inbound": 0, "outbound": 0}
        assert summary["other_environment_contact_episodes"] == 0
        assert summary["place_receptacle_outside_placement_episodes"] == 0
        rows = sorted(
            path
            for path in (root / "expert_screen_rows").iterdir()
            if path.is_dir() and (path / "result.json").is_file()
        )
        assert len(rows) == 8
        observed_clutter = 0
        inbound_clutter = 0
        for directory in rows:
            result = json.loads((directory / "result.json").read_text())
            audit = result["contact_audit"]
            clutter = audit["contact_class_totals"]["clutter"]
            if clutter:
                observed_clutter += 1
                assert audit["phase_contact_class_totals"]["outbound"]["clutter"] == clutter
            inbound_clutter += audit["phase_contact_class_totals"]["inbound"]["clutter"]
            assert audit["phase_contact_class_totals"]["placement"]["clutter"] == 0
        assert observed_clutter == clutter_episodes
        assert inbound_clutter == 0
    assert not (ROOT / "configs/pact_place_corridor_v6.json").exists()


def test_v6b_clearance_probe_has_zero_clutter_contact() -> None:
    root = ROOT / "diagnostics_output/pact_place_corridor_v6b_clearance_probe"
    summary = json.loads((root / "expert_screen.json").read_text())
    assert summary["role"] == "diagnostic_not_a_gate"
    assert summary["n"] == 8
    assert summary["clutter_contact_episodes"] == 0
    assert summary["diagnostic_clean_successes"] == 7
    assert summary["hazard_contact_episodes"] == {"inbound": 0, "outbound": 0}
    rows = sorted(
        path
        for path in (root / "expert_screen_rows").iterdir()
        if path.is_dir() and (path / "result.json").is_file()
    )
    assert len(rows) == 8
    for directory in rows:
        result = json.loads((directory / "result.json").read_text())
        audit = result["contact_audit"]
        assert audit["contact_class_totals"]["clutter"] == 0
        phases = audit["phase_contact_class_totals"]
        assert phases["inbound"]["clutter"] == 0
        assert phases["outbound"]["clutter"] == 0
        assert phases["placement"]["clutter"] == 0
    outcome = json.loads((root / "probe_outcome.json").read_text())
    assert outcome["zero_clutter_requirement_met"] is True
    assert outcome["minimum_clean_6_of_8_met"] is True


def test_v6c_clearance_probe_has_zero_clutter_contact() -> None:
    root = ROOT / "diagnostics_output/pact_place_corridor_v6c_clearance_probe"
    summary = json.loads((root / "expert_screen.json").read_text())
    assert summary["role"] == "diagnostic_not_a_gate"
    assert summary["n"] == 8
    assert summary["clutter_contact_episodes"] == 0
    assert summary["diagnostic_clean_successes"] == 6
    assert summary["hazard_contact_episodes"] == {"inbound": 0, "outbound": 0}
    rows = sorted(
        path
        for path in (root / "expert_screen_rows").iterdir()
        if path.is_dir() and (path / "result.json").is_file()
    )
    assert len(rows) == 8
    nominals = []
    for directory in rows:
        result = json.loads((directory / "result.json").read_text())
        audit = result["contact_audit"]
        assert audit["contact_class_totals"]["clutter"] == 0
        phases = audit["phase_contact_class_totals"]
        assert phases["inbound"]["clutter"] == 0
        assert phases["outbound"]["clutter"] == 0
        assert phases["placement"]["clutter"] == 0
        clutter = result["scene_params"]["pact_clutter"]
        nominal = {slot: spec["nominal_xy_m"] for slot, spec in clutter.items()}
        nominals.append((result["intrusion_side"], nominal))
        for spec in clutter.values():
            body = spec["body"]
            assert body.startswith("pact_clutter_")
            assert "cavity_obj_" not in body
            assert "pact_intrusion_" not in body
            assert "place_receptacle" not in body
            assert spec["half_m"][0] == pytest.approx(0.025)
            assert spec["half_m"][1] == pytest.approx(0.05)
    assert len({json.dumps(item[1], sort_keys=True) for item in nominals}) == 1
    outcome = json.loads((root / "probe_outcome.json").read_text())
    assert outcome["zero_clutter_requirement_met"] is True
    assert outcome["minimum_clean_6_of_8_met"] is True
    assert outcome["top_z_m"] == pytest.approx(0.82)
    assert outcome["fallback_to_top_080"] is False
    assert outcome["clutter_layout_independent_of_intrusion_side"] is True
    assert outcome["inner_face_abs_y_m"] == pytest.approx(0.29)


def test_a1_clearance_probe_has_zero_tray_contact_outside_placement() -> None:
    root = ROOT / "diagnostics_output/pact_place_corridor_v5_clearance_probe"
    summary = json.loads((root / "expert_screen.json").read_text())
    assert summary["role"] == "diagnostic_not_a_gate"
    assert summary["place_receptacle_outside_placement_episodes"] == 0
    assert summary["n"] == 8
    rows = sorted(
        path for path in (root / "expert_screen_rows").iterdir() if path.is_dir()
    )
    assert len(rows) == 8
    for directory in rows:
        result = json.loads((directory / "result.json").read_text())
        audit = result["contact_audit"]
        assert audit["place_receptacle_outside_placement_entries"] == 0
        phases = audit["phase_contact_class_totals"]
        assert phases["inbound"]["place_receptacle"] == 0
        assert phases["outbound"]["place_receptacle"] == 0
        assert phases["other"]["place_receptacle"] == 0


def test_passed_phase0_v5_artifacts_remain() -> None:
    config_path = ROOT / "configs/pact_place_corridor_v5.json"
    summary_path = ROOT / "diagnostics_output/pact_place_corridor_v5/expert_screen.json"
    saved = json.loads(config_path.read_text())
    assert saved["master_seed"] == 2026082201
    assert saved["config_sha256"] == (
        "bd47f1c97d2815657211085590657f5211ca847b776f6039c9617f990da9c1f1"
    )
    assert saved["phase0_gate"]["attempt5_prediction"]["predicted_clean_successes"] == [
        20,
        23,
    ]
    summary = json.loads(summary_path.read_text())
    assert summary["decision"] == "PACT_PLACE_CORRIDOR_PHASE0_PASS"
    assert summary["gate"]["clean_successes"] == 22
    assert summary["task_success"]["count"] == 22
    assert summary["place_success_given_grasp"] == {"numerator": 22, "denominator": 22}
    assert summary["hazard_contact_episodes"] == {"inbound": 0, "outbound": 0}
    assert summary["place_receptacle_outside_placement_episodes"] == 0
    assert summary["bow_fallback_episodes"] == 0
    assert summary["next_action"] == "proceed_to_collection_design"
    assert summary["expert_screen_sha256"] == (
        "7c4cc9ad4740c1e6bbd4be4ee0f31581854c7365634b07d0daa979224363d92f"
    )
    assert not (
        ROOT / "diagnostics_output/pact_place_corridor_v5/stop_record.json"
    ).exists()
    assert (ROOT / "docs/PACT_PLACE_CORRIDOR_GATE_V5.md").read_text().endswith(
        "PACT_PLACE_CORRIDOR_PHASE0_PASS\n"
    )
    attempt5 = (ROOT / "docs/PACT_PLACE_ATTEMPT5.md").read_text()
    assert attempt5.endswith("PACT_PLACE_CORRIDOR_PHASE0_PASS\n")
    assert attempt5.count("PACT_PLACE_CORRIDOR_PHASE0_PASS") >= 1
    assert "22/24" in attempt5
    for prior in (
        "docs/PACT_PLACE_CORRIDOR_GATE.md",
        "docs/PACT_PLACE_CORRIDOR_GATE_V2.md",
        "docs/PACT_PLACE_CORRIDOR_GATE_V3.md",
        "docs/PACT_PLACE_CORRIDOR_GATE_V4.md",
        "docs/PACT_PLACE_ATTEMPT4.md",
    ):
        text = (ROOT / prior).read_text()
        assert text.endswith("PACT_PLACE_CORRIDOR_PHASE0_FAIL\n")


def test_passed_phase0_v6b_artifacts_remain() -> None:
    config_path = ROOT / "configs/pact_place_corridor_v6b.json"
    summary_path = ROOT / "diagnostics_output/pact_place_corridor_v6b/expert_screen.json"
    saved = json.loads(config_path.read_text())
    assert saved["master_seed"] == 2026082501
    assert saved["config_sha256"] == (
        "ebf1be0359b6ff810772a3d4bbb0adf8913710fae4eec14c88a8f403a70e3671"
    )
    assert saved["phase0_gate"]["attempt6b_prediction"]["predicted_clean_successes"] == [
        19,
        22,
    ]
    summary = json.loads(summary_path.read_text())
    assert summary["decision"] == "PACT_PLACE_CORRIDOR_PHASE0_PASS"
    assert summary["gate"]["clean_successes"] == 20
    assert summary["task_success"]["count"] == 20
    assert summary["place_success_given_grasp"] == {"numerator": 20, "denominator": 20}
    assert summary["clutter_contact_episodes"] == 0
    assert summary["hazard_contact_episodes"] == {"inbound": 0, "outbound": 0}
    assert summary["place_receptacle_outside_placement_episodes"] == 0
    assert summary["bow_fallback_episodes"] == 0
    assert summary["expert_screen_sha256"] == (
        "d37f760c80a68256d76da9047d3b8706e1beec09061588d5bcf231b74c9a508a"
    )
    assert not (
        ROOT / "diagnostics_output/pact_place_corridor_v6b/stop_record.json"
    ).exists()
    assert (ROOT / "docs/PACT_PLACE_CORRIDOR_GATE_V6B.md").read_text().endswith(
        "PACT_PLACE_CORRIDOR_PHASE0_PASS\n"
    )
    attempt = (ROOT / "docs/PACT_PLACE_ATTEMPT6B.md").read_text()
    assert attempt.endswith("PACT_PLACE_CORRIDOR_PHASE0_PASS\n")
    assert "PACT_PLACE_CORRIDOR_PHASE0_NOT_RUN" in (
        ROOT / "docs/PACT_PLACE_CORRIDOR_GATE_V6.md"
    ).read_text()
    failed = []
    rows = ROOT / "diagnostics_output/pact_place_corridor_v6b/expert_screen_rows"
    result_dirs = sorted(path for path in rows.iterdir() if path.is_dir())
    assert len(result_dirs) == 24
    for directory in result_dirs:
        result = json.loads((directory / "result.json").read_text())
        assert result["config_sha256"] == saved["config_sha256"]
        assert result["contact_audit"]["contact_class_totals"]["clutter"] == 0
        assert (directory / "trajectory.json").is_file()
        if not result["clean_success"]:
            failed.append(result["role_index"])
            tracking = result.get("terminal_tracking") or {}
            if result["role_index"] in {2, 14}:
                assert tracking.get("check_failure_branch") == "empty_gripper"
            if result["role_index"] in {9, 22}:
                assert tracking.get("check_failure_branch") == "ik_cascade"
    assert failed == [2, 9, 14, 22]


def test_passed_phase0_v6c_artifacts_remain() -> None:
    config_path = ROOT / "configs/pact_place_corridor_v6c.json"
    summary_path = ROOT / "diagnostics_output/pact_place_corridor_v6c/expert_screen.json"
    saved = json.loads(config_path.read_text())
    assert saved["master_seed"] == 2026082701
    assert saved["config_sha256"] == (
        "1dbef6cdd38c6c273d389cbe75717229f78a416831964f9907b00ce4ec58f04e"
    )
    assert saved["phase0_gate"]["attempt6c_prediction"]["predicted_clean_successes"] == [
        19,
        22,
    ]
    assert saved["scene"]["clutter_inner_face_abs_y_m"] == pytest.approx(0.29)
    assert saved["scene"]["clutter_nominal_rear_outer_x_m"] == pytest.approx(0.775)
    summary = json.loads(summary_path.read_text())
    assert summary["decision"] == "PACT_PLACE_CORRIDOR_PHASE0_PASS"
    assert summary["gate"]["clean_successes"] == 23
    assert summary["task_success"]["count"] == 23
    assert summary["place_success_given_grasp"] == {"numerator": 23, "denominator": 23}
    assert summary["clutter_contact_episodes"] == 0
    assert summary["hazard_contact_episodes"] == {"inbound": 0, "outbound": 0}
    assert summary["place_receptacle_outside_placement_episodes"] == 0
    assert summary["bow_fallback_episodes"] == 0
    assert summary["expert_screen_sha256"] == (
        "fef807acfb13ce4ce400d0c0edf323da07a27283382a973486b5604f8f69fc26"
    )
    assert not (
        ROOT / "diagnostics_output/pact_place_corridor_v6c/stop_record.json"
    ).exists()
    assert (ROOT / "docs/PACT_PLACE_CORRIDOR_GATE_V6C.md").read_text().endswith(
        "PACT_PLACE_CORRIDOR_PHASE0_PASS\n"
    )
    attempt = (ROOT / "docs/PACT_PLACE_ATTEMPT6C.md").read_text()
    assert attempt.endswith("PACT_PLACE_CORRIDOR_PHASE0_PASS\n")
    assert (ROOT / "docs/PACT_PLACE_CORRIDOR_GATE_V6B.md").read_text().endswith(
        "PACT_PLACE_CORRIDOR_PHASE0_PASS\n"
    )
    assert "PACT_PLACE_CORRIDOR_PHASE0_NOT_RUN" in (
        ROOT / "docs/PACT_PLACE_CORRIDOR_GATE_V6.md"
    ).read_text()
    failed = []
    rows = ROOT / "diagnostics_output/pact_place_corridor_v6c/expert_screen_rows"
    result_dirs = sorted(path for path in rows.iterdir() if path.is_dir())
    assert len(result_dirs) == 24
    nominals = set()
    for directory in result_dirs:
        result = json.loads((directory / "result.json").read_text())
        assert result["config_sha256"] == saved["config_sha256"]
        assert result["contact_audit"]["contact_class_totals"]["clutter"] == 0
        assert (directory / "trajectory.json").is_file()
        clutter = result["scene_params"]["pact_clutter"]
        nominals.add(
            json.dumps(
                {slot: spec["nominal_xy_m"] for slot, spec in clutter.items()},
                sort_keys=True,
            )
        )
        depth = float(result["scene_params"]["depth"])
        wall = 0.58 + depth + 0.02
        for slot, spec in clutter.items():
            body = spec["body"]
            assert body.startswith("pact_clutter_")
            assert "cavity_obj_" not in body
            assert "pact_intrusion_" not in body
            assert "place_receptacle" not in body
            if slot.endswith("1"):
                outer_x = spec["center_m"][0] + spec["half_m"][0]
                # Nominal rear outer face is 0.775; jitter is unchanged, so a
                # shallow episode can still clip the wall by a few millimetres.
                assert outer_x - wall < 0.01
        if not result["clean_success"]:
            failed.append(result["role_index"])
            tracking = result.get("terminal_tracking") or {}
            assert result["role_index"] == 7
            assert tracking.get("check_failure_branch") == "ik_cascade"
            assert result.get("terminal_policy_phase") == "outbound_approach"
    assert failed == [7]
    assert len(nominals) == 1


def test_v6b_replay_renderer_guards_scene_and_clutter() -> None:
    v6b = (ROOT / "scripts/run_pact_place_v6b_replay_videos.py").read_text()
    v5 = (ROOT / "scripts/run_pact_place_v5_replay_videos.py").read_text()
    assert "pact_place_corridor_v3.xml" in v6b
    assert "_assert_static_furniture" in v6b
    assert "pact_clutter_" in v6b
    assert "running clutter frames" in v6b
    assert "ebf1be0359b6ff810772a3d4bbb0adf8913710fae4eec14c88a8f403a70e3671" in v6b
    assert "row02_FAIL_lift_cup_dropped" in v6b
    assert "row09_FAIL_outbound_approach_ik_cascade" in v6b
    assert "row14_FAIL_lift_cup_dropped" in v6b
    assert "row22_FAIL_outbound_approach_ik_cascade" in v6b
    assert "bd47f1c97d2815657211085590657f5211ca847b776f6039c9617f990da9c1f1" in v5
    assert "pact_place_corridor_v2.xml" in v5
    assert "Do not edit the v5 renderer" in v6b
    assert "pact_clutter_" not in v5


def test_v6b_replay_manifest_if_present() -> None:
    path = ROOT / "diagnostics_output/pact_place_corridor_v6b_videos/manifest.json"
    if not path.exists():
        pytest.skip("v6b replay clips have not been rendered")
    manifest = json.loads(path.read_text())
    payload = dict(manifest)
    digest = payload.pop("manifest_sha256")
    assert digest == contract.sha256_payload(payload)
    assert manifest["n"] == 24
    assert manifest["config_sha256"] == (
        "ebf1be0359b6ff810772a3d4bbb0adf8913710fae4eec14c88a8f403a70e3671"
    )
    assert manifest["required_scene_xml"] == "pact_place_corridor_v3.xml"
    assert manifest["clutter_bodies_asserted_against_config"] is True
    assert manifest["replay_only"] is True
    assert manifest["physics_stepped"] is False
    assert manifest["expert_rerun"] is False
    fail_clips = [item["clip"] for item in manifest["clips"] if not item["clean_success"]]
    assert fail_clips == [
        "row02_FAIL_lift_cup_dropped.mp4",
        "row09_FAIL_outbound_approach_ik_cascade.mp4",
        "row14_FAIL_lift_cup_dropped.mp4",
        "row22_FAIL_outbound_approach_ik_cascade.mp4",
    ]
    for item in manifest["clips"]:
        assert item["faithfulness"]["max_object_residual_m"] < 1e-3
        assert item["faithfulness"]["max_tcp_residual_m"] < 1e-3
        assert item["running_clutter_frames"] == 0
    crib = (ROOT / "diagnostics_output/pact_place_corridor_v6b_videos/CRIB.md").read_text()
    assert "Attempt-6b" in crib
    for name in fail_clips:
        assert name in crib


def test_v6c_replay_renderer_guards_scene_and_clutter() -> None:
    v6c = (ROOT / "scripts/run_pact_place_v6c_replay_videos.py").read_text()
    v6b = (ROOT / "scripts/run_pact_place_v6b_replay_videos.py").read_text()
    v5 = (ROOT / "scripts/run_pact_place_v5_replay_videos.py").read_text()
    assert "pact_place_corridor_v3.xml" in v6c
    assert "_assert_static_furniture" in v6c
    assert "pact_clutter_" in v6c
    assert "running clutter frames" in v6c
    assert "1dbef6cdd38c6c273d389cbe75717229f78a416831964f9907b00ce4ec58f04e" in v6c
    assert "row07_FAIL_outbound_approach_ik_cascade" in v6c
    assert "Do not edit the v5 renderer" in v6c
    assert "Do not edit the v6b renderer" in v6c
    assert "ebf1be0359b6ff810772a3d4bbb0adf8913710fae4eec14c88a8f403a70e3671" in v6b
    assert "bd47f1c97d2815657211085590657f5211ca847b776f6039c9617f990da9c1f1" in v5
    assert "pact_place_corridor_v2.xml" in v5
    assert "1dbef6cdd38c6c273d389cbe75717229f78a416831964f9907b00ce4ec58f04e" not in v6b
    assert "1dbef6cdd38c6c273d389cbe75717229f78a416831964f9907b00ce4ec58f04e" not in v5


def test_v6c_replay_manifest_if_present() -> None:
    path = ROOT / "diagnostics_output/pact_place_corridor_v6c_videos/manifest.json"
    if not path.exists():
        pytest.skip("v6c replay clips have not been rendered")
    manifest = json.loads(path.read_text())
    payload = dict(manifest)
    digest = payload.pop("manifest_sha256")
    assert digest == contract.sha256_payload(payload)
    assert digest == (
        "9675a05171d31fc7ab74f791827315e9eee9a703b51a9b61a062ab598c74a8ef"
    )
    assert manifest["n"] == 24
    assert manifest["config_sha256"] == (
        "1dbef6cdd38c6c273d389cbe75717229f78a416831964f9907b00ce4ec58f04e"
    )
    assert manifest["required_scene_xml"] == "pact_place_corridor_v3.xml"
    assert manifest["clutter_bodies_asserted_against_config"] is True
    assert manifest["replay_only"] is True
    assert manifest["physics_stepped"] is False
    assert manifest["expert_rerun"] is False
    fail_clips = [item["clip"] for item in manifest["clips"] if not item["clean_success"]]
    assert fail_clips == ["row07_FAIL_outbound_approach_ik_cascade.mp4"]
    for item in manifest["clips"]:
        assert item["faithfulness"]["max_object_residual_m"] < 1e-3
        assert item["faithfulness"]["max_tcp_residual_m"] < 1e-3
        assert item["running_clutter_frames"] == 0
        clip_path = path.parent / item["clip"]
        assert clip_path.is_file()
    crib = (ROOT / "diagnostics_output/pact_place_corridor_v6c_videos/CRIB.md").read_text()
    assert "Attempt-6c" in crib
    assert "28 mm" in crib
    for name in fail_clips:
        assert name in crib


def test_v5_rows_emit_endpoint_scalars_and_trajectories() -> None:
    from run_pact_place_expert_screen import ENDPOINT_SCALAR_KEYS

    rows = ROOT / "diagnostics_output/pact_place_corridor_v5/expert_screen_rows"
    result_dirs = sorted(path for path in rows.iterdir() if path.is_dir())
    assert len(result_dirs) == 24
    failed = []
    for directory in result_dirs:
        result = json.loads((directory / "result.json").read_text())
        block = result["endpoint_scalars"]
        assert set(ENDPOINT_SCALAR_KEYS) <= set(block)
        assert block["endpoint_values_emitted_during_compaction"] is True
        assert all(block[key] is not None for key in ENDPOINT_SCALAR_KEYS)
        assert (directory / "trajectory.json").is_file()
        assert (directory / "initial_observation_accepted.json").is_file()
        audit = result["contact_audit"]
        assert audit["place_receptacle_outside_placement_entries"] == 0
        if not result["clean_success"]:
            failed.append(result["role_index"])
            tracking = result.get("terminal_tracking") or {}
            if result["role_index"] == 3:
                assert tracking.get("check_failure_branch") == "ik_cascade"
            if result["role_index"] == 10:
                assert tracking.get("check_failure_branch") == "empty_gripper"
    assert failed == [3, 10]


def test_protected_artifacts_still_match_the_v5_contract() -> None:
    saved = json.loads((ROOT / "configs/pact_place_corridor_v5.json").read_text())
    for relative, digest in saved["protected_artifact_sha256_before"].items():
        assert contract.sha256_file(ROOT / relative) == digest
    assert contract.sha256_file(
        MOLMO
        / "molmo_spaces/data_generation/custom_scenes/pact_collision_corridor.xml"
    ) == "f8c04b07b9416593eb60ad4797ccbae91f7d3524effd30438ef552e5a2d75540"
    assert contract.sha256_file(
        MOLMO
        / "molmo_spaces/policy/solvers/object_manipulation/pick_and_place_planner_policy.py"
    ) == "9ee369789397add3ae74492e4821993a981940be54cc0579e3d282328a8aa36a"
    assert contract.sha256_file(
        MOLMO
        / "molmo_spaces/policy/solvers/object_manipulation/base_object_manipulation_planner_policy.py"
    ) == "a7ee35704d60b82246fc48db466aaa081a6a83717d983cc062df3988e53893d5"
    assert contract.sha256_file(
        MOLMO / "molmo_spaces/tasks/pact_contact_audit.py"
    ) == "f07aace35d856b4e7415e37b04cc6c512cdc117da1fc00648b0fa61aa01d7abd"
    assert contract.sha256_file(
        MOLMO
        / "molmo_spaces/data_generation/custom_scenes/pact_place_corridor_v1.xml"
    ) == "d853e27ca453a246a73a7fa590e3a05f5c93db0893805cd41067e73441aba942"
    assert contract.sha256_file(
        MOLMO
        / "molmo_spaces/data_generation/custom_scenes/pact_place_corridor_v2.xml"
    ) == "920860de9426fe15d607a6318fc81fb51012f4b82aa3d0e437a76f648e38be5d"


def test_v3_xml_only_adds_named_clutter_bodies() -> None:
    v2_names = {
        element.attrib["name"]
        for element in ET.parse(
            MOLMO
            / "molmo_spaces/data_generation/custom_scenes/pact_place_corridor_v2.xml"
        )
        .getroot()
        .iter()
        if "name" in element.attrib
    }
    v3_tree = ET.parse(
        MOLMO
        / "molmo_spaces/data_generation/custom_scenes/pact_place_corridor_v3.xml"
    )
    v3_names = {
        element.attrib["name"]
        for element in v3_tree.getroot().iter()
        if "name" in element.attrib
    }
    clutter = sorted(name for name in v3_names if name.startswith("pact_clutter_"))
    assert clutter == [
        "pact_clutter_l0",
        "pact_clutter_l0_g",
        "pact_clutter_l1",
        "pact_clutter_l1_g",
        "pact_clutter_r0",
        "pact_clutter_r0_g",
        "pact_clutter_r1",
        "pact_clutter_r1_g",
    ]
    for name in clutter:
        assert "cavity_obj_" not in name
        assert "pact_intrusion_" not in name
        assert "place_receptacle" not in name
    assert v2_names <= v3_names
    assert v3_names - v2_names == set(clutter)
    v3_text = (
        MOLMO
        / "molmo_spaces/data_generation/custom_scenes/pact_place_corridor_v3.xml"
    ).read_text()
    assert 'model="pact_place_corridor_v3"' in v3_text
    assert 'pos="0.35 0.32 0"' in v3_text
    assert 'size="0.10 0.10 0.02"' in v3_text
    assert 'pos="0.70 0.34 0.77"' in v3_text
    assert 'pos="0.75 -0.34 0.77"' in v3_text
    assert 'size="0.025 0.05 0.05"' in v3_text


def test_shared_contact_audit_does_not_know_clutter() -> None:
    from molmo_spaces.tasks.pact_contact_audit import classify_contact

    assert classify_contact({"root2": "pact_clutter_l0"}) == "other_environment"
    source = (MOLMO / "molmo_spaces/tasks/pact_contact_audit.py").read_text()
    assert "pact_clutter_" not in source
    assert "CLUTTER_BODY_PREFIX" not in source


def test_v3_xml_hash_is_unchanged_and_v4_pool_is_sixteen_bodies() -> None:
    v3 = (
        MOLMO
        / "molmo_spaces/data_generation/custom_scenes/pact_place_corridor_v3.xml"
    )
    v4 = (
        MOLMO
        / "molmo_spaces/data_generation/custom_scenes/pact_place_corridor_v4.xml"
    )
    assert contract.sha256_file(v3) == contract.PLACE_V3_SCENE_SHA256
    assert contract.sha256_file(v4) == contract.PLACE_V4_SCENE_SHA256
    v3_names = {
        element.attrib["name"]
        for element in ET.parse(v3).getroot().iter()
        if "name" in element.attrib
    }
    v4_names = {
        element.attrib["name"]
        for element in ET.parse(v4).getroot().iter()
        if "name" in element.attrib
    }
    v3_clutter = sorted(name for name in v3_names if name.startswith("pact_clutter_"))
    v4_clutter = sorted(name for name in v4_names if name.startswith("pact_clutter_"))
    assert v3_clutter == [
        "pact_clutter_l0",
        "pact_clutter_l0_g",
        "pact_clutter_l1",
        "pact_clutter_l1_g",
        "pact_clutter_r0",
        "pact_clutter_r0_g",
        "pact_clutter_r1",
        "pact_clutter_r1_g",
    ]
    expected_v4 = []
    for index in range(16):
        expected_v4.append(f"pact_clutter_{index:02d}")
        expected_v4.append(f"pact_clutter_{index:02d}_g")
    assert v4_clutter == expected_v4
    assert 'model="pact_place_corridor_v4"' in v4.read_text()
    assert "pact_clutter_l0" not in v4.read_text()
    v4_meta = json.loads(
        (
            MOLMO
            / "molmo_spaces/data_generation/custom_scenes/pact_place_corridor_v4_metadata.json"
        ).read_text()
    )
    assert v4_meta == {"objects": {}}


def test_v7_clutter_jitter_default_stream_stays_v6c_slots() -> None:
    seed = 123456789
    default_x, default_y = contract.clutter_jitters_for_seed(seed)
    named_x, named_y = contract.clutter_jitters_for_seed(
        seed, slot_names=contract.CLUTTER_SLOT_NAMES
    )
    assert default_x == named_x == {
        slot: default_x[slot] for slot in ("l0", "l1", "r0", "r1")
    }
    assert default_y == named_y
    pool_x, pool_y = contract.clutter_jitters_for_seed(
        seed, slot_names=contract.V7_CLUTTER_POOL_SLOT_NAMES
    )
    assert tuple(pool_x) == contract.V7_CLUTTER_POOL_SLOT_NAMES
    assert len(pool_x) == 16
    assert pool_x != default_x


def test_v7_design_review_contract_is_not_a_gate() -> None:
    document = contract.build_design_review_contract()
    contract.validate_design_review_contract(document)
    assert document["role"] == "human_design_review_not_a_gate"
    assert document["authorizes_collection"] is False
    assert document["authorizes_probe"] is False
    assert document["authorizes_gate"] is False
    assert document["not_a_clean_rate_estimate"] is True
    assert document["master_seed"] == 2026082801
    assert len(document["expert_screen_rows"]) == 12
    assert document["scene"]["xml"].endswith("pact_place_corridor_v4.xml")
    assert document["scene"]["sampler_class"] == "PactPlaceCorridorV4Sampler"
    v6c_ids = {
        row["episode_id"]
        for row in json.loads((ROOT / "configs/pact_place_corridor_v6c.json").read_text())[
            "expert_screen_rows"
        ]
    }
    review_ids = {row["episode_id"] for row in document["expert_screen_rows"]}
    assert review_ids.isdisjoint(v6c_ids)
    screen = (ROOT / "scripts/run_pact_place_expert_screen.py").read_text()
    assert "pact_place_corridor_v4.xml" in screen
    assert "PactPlaceCorridorV4Sampler" in screen
    v7 = (ROOT / "scripts/run_pact_place_v7_replay_videos.py").read_text()
    assert "pact_place_corridor_v4.xml" in v7
    assert "pact_clutter_15" in v7
    assert "Do not edit the v6c renderer" in v7
    assert "review00_clean_success" in v7


def test_v7_replay_renderer_guards_scene_and_does_not_edit_frozen_renderers() -> None:
    v7 = (ROOT / "scripts/run_pact_place_v7_replay_videos.py").read_text()
    v6c = (ROOT / "scripts/run_pact_place_v6c_replay_videos.py").read_text()
    v6b = (ROOT / "scripts/run_pact_place_v6b_replay_videos.py").read_text()
    v5 = (ROOT / "scripts/run_pact_place_v5_replay_videos.py").read_text()
    assert "pact_place_corridor_v4.xml" in v7
    assert 'REQUIRED_SCENE_XML = "pact_place_corridor_v4.xml"' in v7
    assert "Do not edit the v5 renderer" in v7
    assert "Do not edit the v6b renderer" in v7
    assert "Do not edit the v6c renderer" in v7
    assert "1dbef6cdd38c6c273d389cbe75717229f78a416831964f9907b00ce4ec58f04e" in v6c
    assert "ebf1be0359b6ff810772a3d4bbb0adf8913710fae4eec14c88a8f403a70e3671" in v6b
    assert "bd47f1c97d2815657211085590657f5211ca847b776f6039c9617f990da9c1f1" in v5
    assert "pact_place_corridor_v2.xml" in v5
    assert "pact_clutter_00" in v7
    assert "min_clearance" in v7


def test_v8_contract_freezes_real_movable_clutter_without_authorizing_gate() -> None:
    path = ROOT / "configs/pact_place_corridor_v8.json"
    document = contract.load_v8_contract(path)
    assert document["role"] == "human_design_review_not_a_gate"
    assert document["authorizes_gate"] is False
    assert document["authorizes_collection"] is False
    assert document["scene"]["sampler_class"] == "PactPlaceCorridorV5Sampler"
    assert document["scene"]["clutter_movable_free_bodies"] is True
    assert document["scene"]["clutter_added_to_obstacle_aabbs"] is False
    assert contract.sha256_file(ROOT / document["scene"]["xml"]) == (
        contract.PLACE_V5_SCENE_SHA256
    )
    assert 12 <= len(document["palette"]) <= 20
    assert len(document["selected_layouts"]) == 24
    assert {item["size_class"] for item in document["palette"]} == {
        "small",
        "medium",
        "large",
    }
    assert not any(
        token in item["uid"].lower()
        for item in document["palette"]
        for token in ("cup", "mug", "egg")
    )
    review_ids = {row["episode_id"] for row in document["family_review_rows"]}
    gate_ids = {row["episode_id"] for row in document["expert_screen_rows"]}
    assert review_ids.isdisjoint(gate_ids)
    for family in contract.V8_FAMILIES:
        layouts = [
            item for item in document["selected_layouts"] if item["family"] == family
        ]
        assert len(layouts) == 4
        assert sum(item["intrusion_side"] == "left" for item in layouts) == 2
    for relative, digest in document["source_sha256"].items():
        observed = contract.sha256_file(ROOT / relative)
        if relative.endswith("molmo_spaces/tasks/enclosure_reach.py"):
            # V8B explicitly authorized the minimal mount/prop branch after V8
            # was frozen; the historical V8 digest remains untouched.
            assert observed != digest
        else:
            assert observed == digest


def test_v8_sweep_meets_candidate_quota_and_visibility_preregistration() -> None:
    analysis = json.loads(
        (
            ROOT / "diagnostics_output/pact_place_clutter_sweep_v8/analysis.json"
        ).read_text()
    )
    assert analysis["role"] == "b1_b2_replay_sweep_not_a_gate"
    assert analysis["authorizes_gate"] is False
    assert analysis["n_candidates"] >= 400
    assert analysis["n_admitted"] > 0
    assert analysis["chosen_n"] == 24
    assert analysis["cup_is_closest_body_count"] <= 6
    assert analysis["visibility_spans_range"] is True
    assert analysis["min_pairwise_selected_layout_distance"] > 0.1
    assert len(analysis["family_side_quotas"]) == 12
    assert set(analysis["family_side_quotas"].values()) == {2}
    for layout in analysis["selected_layouts"]:
        assert len(layout["objects"]) >= 2
        for item in layout["objects"]:
            assert item["quat_wxyz"] == pytest.approx(
                [2**-0.5, 2**-0.5, 0.0, 0.0]
            )


def test_v8_scoring_check_and_family_review_stop_before_gate() -> None:
    scoring = json.loads(
        (
            ROOT
            / "diagnostics_output/pact_place_corridor_v8_scoring_check/scoring_check.json"
        ).read_text()
    )
    assert scoring["role"] == "scoring_check_not_a_gate"
    assert scoring["passed"] is True
    assert scoring["link_contact"]["passed"] is True
    assert scoring["link_contact"]["clean_success"] is False
    assert scoring["link_contact"]["n_link_clutter_contacts"] > 0
    assert scoring["topple"]["passed"] is True
    assert scoring["topple"]["events"][0]["classification"] == "other_environment"

    report_path = (
        ROOT / "diagnostics_output/pact_place_corridor_v8_family_review/family_review.json"
    )
    report = json.loads(report_path.read_text())
    payload = dict(report)
    observed = payload.pop("family_review_sha256")
    assert observed == contract.sha256_payload(payload)
    assert report["role"] == "human_design_review_not_a_gate"
    assert report["authorizes_gate"] is False
    assert report["authorizes_collection"] is False
    assert report["mandatory_stop_after_this_report"] is True
    assert report["gate_executed"] is False
    assert report["n_attempts"] == report["n_clips"] == 9
    assert report["rendered_every_attempt"] is True
    assert report["review_gate_episode_overlap"] == []
    assert report["families_without_clean_success_after_four"] == []
    assert report["attempts_needed_per_family"]["F3_front_stagger"] == 4
    realized = report["metric_table_vs_v6c"]["v8_realized_family_review"]
    assert realized["n_completed_rendered_episodes"] == 6
    assert realized["n_episodes_cup_is_closest_body"] == 5
    assert any(
        item["severity"] == "blocks_unattended_B6"
        for item in report["human_review_findings"]
    )

    video_root = report_path.parent / "videos"
    manifest = json.loads((video_root / "manifest.json").read_text())
    assert manifest["physics_stepped"] is False
    assert manifest["expert_rerun_during_render"] is False
    assert manifest["rendered_every_attempt"] is True
    assert len(manifest["clips"]) == 9
    for item in manifest["clips"]:
        clip = video_root / item["clip"]
        assert clip.is_file() and clip.stat().st_size > 0
        assert contract.sha256_file(clip) == item["video_sha256"]


def test_v8_v5_sampler_partitions_free_props_and_mocap_mounts() -> None:
    source = (MOLMO / "molmo_spaces/tasks/enclosure_reach.py").read_text()
    v5 = _class_source(source, "PactPlaceCorridorV5Sampler")
    assert "mjtJoint.mjJNT_FREE" in v5
    assert 'slot_class == "mount"' in v5
    assert "body.mocap = True" in v5
    assert "_set_mocap_pose" in v5
    assert "kinematic_mocap_overhead_fixture" in v5
    assert "active_props" in v5
    assert "active_mounts" in v5
    assert "CLUTTER_SETTLE_STEPS" in v5
    assert "CLUTTER_MAX_SETTLED_XY_DRIFT_M" in v5
    assert "_body_collision_aabb" in v5
    assert '"pact_clutter_added_to_obstacle_aabbs": False' in v5
    assert "obstacle_aabbs.append" not in v5
    scene = (
        MOLMO
        / "molmo_spaces/data_generation/custom_scenes/pact_place_corridor_v5.xml"
    )
    root = ET.parse(scene).getroot()
    assert root.attrib["model"] == "pact_place_corridor_v5"
    include = root.find("include")
    assert include is not None
    assert include.attrib["file"] == "pact_place_corridor_v3.xml"


def test_v8b_palette_stability_is_partitioned_and_thresholded() -> None:
    path = (
        ROOT
        / "diagnostics_output/pact_place_clutter_sweep_v8b/palette_stability.json"
    )
    document = json.loads(path.read_text())
    assert document["palette_size"] == 18
    assert document["slot_class_counts"] == {"mount": 5, "prop": 13}
    assert max(document["palette_category_counts"].values()) <= 3
    for record in document["records"]:
        if record["slot_class"] == "mount":
            assert record["settling_skipped"] is True
            continue
        if record["accepted"]:
            assert record["center_drift_m"] <= 0.005
            assert record["orientation_change_deg"] <= 5.0


def test_v8b_failed_admission_stops_before_family_review() -> None:
    report = json.loads(
        (
            ROOT / "diagnostics_output/pact_place_clutter_sweep_v8b/analysis.json"
        ).read_text()
    )
    payload = dict(report)
    observed = payload.pop("analysis_sha256")
    assert observed == contract.sha256_payload(payload)
    assert report["authorizes_gate"] is False
    assert report["mandatory_stop_before_B5b"] is True
    assert report["pass2"]["all_admission_gates_pass"] is False
    assert report["pass3"]["n_rescored"] >= 400
    assert report["pass3"]["selection_possible"] is False
    assert report["pass3"]["missing_quota_cell_count"] == 11
    assert report["b5b"]["executed"] is False

    scoring = json.loads(
        (
            ROOT
            / "diagnostics_output/pact_place_corridor_v8b_mount_scoring_check/scoring_check.json"
        ).read_text()
    )
    assert scoring["passed"] is True
    assert scoring["body_joint_address"] == -1
    assert scoring["body_mocap_id"] >= 0
    assert scoring["contact_audit"]["contact_class_totals"]["clutter"] > 0
    assert scoring["clean_success"] is False
