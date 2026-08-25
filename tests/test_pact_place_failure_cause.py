from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_pact_place_expert_screen import derive_failure_cause  # noqa: E402
from run_pact_place_v9_v1b_review import clip_stem  # noqa: E402


def test_stability_collision_precedes_empty_gripper_symptom() -> None:
    cause = derive_failure_cause(
        task_success=False,
        contact_audit={"contact_class_totals": {"clutter": 2144}},
        clutter_stability_events=[
            {
                "step": 105,
                "policy_phase": "pregrasp",
                "body": "pact_clutter_01/Soap_Bottle_30",
                "displacement_m": 0.0218,
                "rotation_angle_rad": 0.209,
            }
        ],
        terminal_tracking={"check_failure_branch": "empty_gripper"},
    )
    assert cause is not None
    assert cause["code"] == "clutter_collision_stability_event"
    assert cause["terminal_symptom"] == "empty_gripper"
    assert cause["step"] == 105


def test_review_filename_uses_causal_collision_label() -> None:
    stem = clip_stem(
        9,
        "right",
        {
            "clean_success": False,
            "failure_cause": {"code": "clutter_collision_stability_event"},
            "terminal_policy_phase": "lift",
            "terminal_tracking": {"check_failure_branch": "empty_gripper"},
        },
    )
    assert stem == "attempt09_right_FAIL_clutter_collision_stability_event"
