from __future__ import annotations

import ast
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MOLMO = ROOT / "submodules" / "molmospaces"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(MOLMO))

from pact_place_abort_branch_telemetry import classify_check_failure_branch


def _class_source(source: str, name: str) -> str:
    tree = ast.parse(source)
    node = next(
        item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == name
    )
    lines = source.splitlines(keepends=True)
    return "".join(lines[node.lineno - 1 : node.end_lineno])


def test_check_failure_branch_order_matches_planner() -> None:
    common = dict(
        failed=True,
        action_index=4,
        n_primitives=8,
        sequential_ik_failures=0,
        max_sequential_ik_failures=8,
        is_holding_object=True,
        inter_finger_dist_m=0.0,
        gripper_empty_trip_m=0.002,
        pos_err_m=0.009,
        rot_err_rad=0.05,
        tcp_pos_err_threshold_m=0.1,
        tcp_rot_err_threshold_rad=math.radians(30),
    )
    assert classify_check_failure_branch(**common) == "empty_gripper"
    holding = dict(common)
    holding["inter_finger_dist_m"] = 0.04
    holding["pos_err_m"] = 0.2
    assert classify_check_failure_branch(**holding) == "pos_err"
    rotated = dict(common)
    rotated["inter_finger_dist_m"] = 0.04
    rotated["rot_err_rad"] = math.radians(31)
    assert classify_check_failure_branch(**rotated) == "rot_err"
    complete = dict(common)
    complete["inter_finger_dist_m"] = 0.04
    complete["action_index"] = 8
    assert classify_check_failure_branch(**complete) == "sequence_complete"
    ik = dict(common)
    ik["sequential_ik_failures"] = 8
    assert classify_check_failure_branch(**ik) == "ik_cascade"
    idle = dict(common)
    idle["failed"] = False
    assert classify_check_failure_branch(**idle) is None


def test_empty_gripper_is_not_classified_when_not_holding() -> None:
    assert (
        classify_check_failure_branch(
            failed=True,
            action_index=4,
            n_primitives=8,
            sequential_ik_failures=0,
            max_sequential_ik_failures=8,
            is_holding_object=False,
            inter_finger_dist_m=0.0,
            gripper_empty_trip_m=0.002,
            pos_err_m=0.009,
            rot_err_rad=0.05,
            tcp_pos_err_threshold_m=0.1,
            tcp_rot_err_threshold_rad=math.radians(30),
        )
        == "unclassified"
    )


def test_upstream_shared_planner_files_still_unmodified() -> None:
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


def test_collision_corridor_classes_untouched_by_abort_telemetry() -> None:
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
    assert "gripper_empty_threshold" not in _class_source(
        current, "PactPlaceCorridorPolicyConfig"
    )


def test_abort_branch_diagnostic_is_not_a_gate() -> None:
    diagnostic = ROOT / "diagnostics_output/pact_place_abort_branch"
    if not (diagnostic / "role.json").is_file():
        pytest.skip("abort-branch diagnostic has not been run")
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
    assert role["role"] == "diagnostic_not_a_gate"
    assert role["authorizes_collection"] is False
    assert role["next_action"] == "none_diagnostic_only"
    assert "decision" not in role
    analysis = json.loads((diagnostic / "analysis.json").read_text())
    assert analysis["role"] == "diagnostic_not_a_gate"
    assert analysis["authorizes_collection"] is False
    interpreted = [row for row in analysis["rows"] if row.get("interpreted")]
    assert interpreted, "no reproducing rows to interpret"
    for row in interpreted:
        if row["is_control"]:
            continue
        assert row["branch"] in {
            "empty_gripper",
            "pos_err",
            "rot_err",
            "sequence_complete",
            "ik_cascade",
            "unclassified",
        }
