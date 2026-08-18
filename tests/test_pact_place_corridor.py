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
    assert "_get_placement_poses" in PactPlaceCorridorPolicy.__dict__
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
    for relative in (
        "configs/pact_place_corridor_v1.json",
        "configs/pact_place_corridor_v2.json",
    ):
        path = ROOT / relative
        if not path.exists():
            pytest.skip(f"{relative} has not been generated")
        contract.validate_contract(json.loads(path.read_text()))


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
