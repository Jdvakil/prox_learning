from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "analyze_pact_environment_gate",
        ROOT / "scripts" / "analyze_pact_environment_gate.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


gate = _load()


def _expert():
    return {
        "task_success": True,
        "collision_free_task_success": True,
        "surface_activity": {
            "pregrasp_control_steps": 10,
            "steps_intrusion_inside_20cm": 4,
            "steps_intrusion_inside_12cm": 1,
            "episode_has_intrusion_sighting": True,
        },
    }


def _act(primary: bool, hazard: bool):
    return {
        "task_success": True,
        "collision_free_task_success": primary,
        "contact_audit": {
            "contact_class_totals": {
                "grasp_target": 1,
                "hazard_bar": int(hazard),
                "other_environment": 0,
            }
        },
        "failure_taxonomy": (
            "collision_free_task_success" if primary else "hazard_bar_contact"
        ),
    }


def test_gate_passes_only_in_predeclared_headroom_band():
    experts = [_expert() for _ in range(24)]
    acts = [_act(index < 12, index >= 12) for index in range(24)]
    result = gate.analyze(
        manifest={"manifest_sha256": "m"},
        expert_results=experts,
        act_results=acts,
        pilot_schedule={"schedule_sha256": "s"},
    )
    assert result["all_applicable_gates_pass"]
    assert result["decision"] == "PACT_ENVIRONMENT_ADEQUATE"

    ceiling = [_act(True, False) for _ in range(24)]
    result = gate.analyze(
        manifest={"manifest_sha256": "m"},
        expert_results=experts,
        act_results=ceiling,
        pilot_schedule={"schedule_sha256": "s"},
    )
    assert not result["all_applicable_gates_pass"]
    assert result["decision"] == "PACT_ENVIRONMENT_INADEQUATE"
