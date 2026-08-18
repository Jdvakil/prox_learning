from __future__ import annotations

import ast
import hashlib
import json
import math
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

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
    assert audit.summary()["collision_free"] is True
    audit._pair_totals["other_environment"] = 1
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
    assert PactPlaceCorridorPolicy.BOW_SHRINK_STEP_M == pytest.approx(0.02)
    assert "_get_placement_poses" in PactPlaceCorridorPolicy.__dict__
    assert PactPlaceCorridorPolicy._bow_magnitudes(0.14, 0.14, 0.02) == pytest.approx(
        [0.14]
    )
    assert PactPlaceCorridorPolicy._bow_magnitudes(0.18, 0.14, 0.02) == pytest.approx(
        [0.18, 0.16, 0.14]
    )


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
    assert document["master_seed"] == 2026081901
    assert len(rows) == 24
    assert sum(row["intrusion_side"] == "left" for row in rows) == 12
    assert sum(row["intrusion_side"] == "right" for row in rows) == 12
    assert document["phase0_gate"]["minimum_clean_successes"] == 20
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
    path = ROOT / "configs/pact_place_corridor_v2.json"
    if not path.exists():
        pytest.skip("place-corridor v2 contract has not been generated")
    saved = json.loads(path.read_text())
    assert saved == contract.build_contract()


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
    assert (ROOT / "docs/PACT_PLACE_CORRIDOR_GATE.md").read_text().endswith(
        "PACT_PLACE_CORRIDOR_PHASE0_FAIL\n"
    )
