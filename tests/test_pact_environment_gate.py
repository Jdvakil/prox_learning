from __future__ import annotations

import importlib.util
import json
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


def test_terminal_expert_construction_failures_count_as_failed_rows(tmp_path):
    manifest = json.loads(
        (ROOT / "configs" / "pact_collision_candidate_manifest_v1.json").read_text()
    )
    expected = []
    for index, row in enumerate(
        sorted(
            (
                row
                for row in manifest["rows"]
                if row["role"] == "pilot_train"
            ),
            key=lambda row: row["role_index"],
        )
    ):
        status = "sampling_failure" if index % 2 == 0 else "infrastructure_failure"
        result = {
            "status": status,
            "episode_id": row["episode_id"],
            "row_sha256": row["row_sha256"],
            "task_success": False,
            "collision_free_task_success": False,
        }
        path = tmp_path / "rows" / row["episode_id"] / "result.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(result))
        expected.append(status)

    results = gate._expert_results(tmp_path, manifest)

    assert [result["status"] for result in results] == expected
    assert all(not result["task_success"] for result in results)
    assert all(not result["collision_free_task_success"] for result in results)
    assert all(
        result["surface_activity"]
        == {
            "pregrasp_control_steps": 0,
            "steps_intrusion_inside_20cm": 0,
            "steps_intrusion_inside_12cm": 0,
            "episode_has_intrusion_sighting": False,
        }
        for result in results
    )


def test_failed_expert_prerequisite_stops_before_pilot_act():
    experts = [_expert() for _ in range(24)]
    for index in range(4):
        experts[index] = {
            "status": "infrastructure_failure",
            "role_index": index,
            "episode_id": f"failed-{index}",
            "task_success": False,
            "collision_free_task_success": False,
            "surface_activity": {
                "pregrasp_control_steps": 0,
                "steps_intrusion_inside_20cm": 0,
                "steps_intrusion_inside_12cm": 0,
                "episode_has_intrusion_sighting": False,
            },
        }
    for index, expert in enumerate(experts[4:], start=4):
        expert.update(
            {
                "status": "success",
                "role_index": index,
                "episode_id": f"success-{index}",
            }
        )
    experts[4]["collision_free_task_success"] = False
    experts[4]["contact_audit"] = {
        "contact_class_totals": {
            "grasp_target": 1,
            "hazard_bar": 58,
            "other_environment": 0,
        },
        "frames_with_contact": {
            "grasp_target": 1,
            "hazard_bar": 58,
            "other_environment": 0,
        },
        "first_contact_step": {
            "grasp_target": 10,
            "hazard_bar": 20,
            "other_environment": None,
        },
        "non_target_contact_entries": 58,
    }

    result = gate.analyze_expert_prerequisite(
        manifest={"manifest_sha256": "m"},
        expert_results=experts,
    )

    assert result["expert"]["ordinary_task_success"] == 20
    assert result["expert"]["collision_free_task_success"] == 19
    assert result["expert"]["episodes_with_intrusion_sighting"] == 20
    assert not result["checks"][
        "expert_collision_free_task_success_at_least_20_of_24"
    ]
    assert result["act"]["status"] == (
        "not_run_due_to_failed_expert_prerequisite"
    )
    assert result["stop_before_policy_training"]
    assert result["decision"] == "PACT_ENVIRONMENT_INADEQUATE"


def test_environment_stop_report_ends_in_exact_token():
    spec = importlib.util.spec_from_file_location(
        "finalize_pact_environment_stop",
        ROOT / "scripts" / "finalize_pact_environment_stop.py",
    )
    finalizer = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(finalizer)
    gate_result = {
        "decision": "PACT_ENVIRONMENT_INADEQUATE",
        "all_applicable_gates_pass": False,
        "stop_before_policy_training": True,
        "manifest_sha256": "m",
        "checks": {
            "expert_task_success_at_least_20_of_24": True,
            "expert_collision_free_task_success_at_least_20_of_24": False,
            "surface_active_episodes_at_least_20_of_24": True,
            "surface_pregrasp_inside_20cm_at_least_30_percent": True,
            "surface_pregrasp_inside_12cm_at_least_5_percent": True,
        },
        "expert": {
            "n": 24,
            "ordinary_task_success": 20,
            "collision_free_task_success": 19,
            "status_counts": {
                "success": 20,
                "sampling_failure": 1,
                "infrastructure_failure": 3,
            },
            "pregrasp_control_steps": 100,
            "steps_intrusion_inside_20cm": 50,
            "fraction_pregrasp_inside_20cm": 0.5,
            "steps_intrusion_inside_12cm": 20,
            "fraction_pregrasp_inside_12cm": 0.2,
            "episodes_with_intrusion_sighting": 20,
            "collision_rows": [
                {
                    "role_index": 21,
                    "episode_id": "x",
                    "contact_class_totals": {
                        "grasp_target": 10,
                        "hazard_bar": 58,
                        "other_environment": 0,
                    },
                    "frames_with_contact": {
                        "grasp_target": 10,
                        "hazard_bar": 58,
                        "other_environment": 0,
                    },
                }
            ],
        },
        "act": {"status": "not_run", "n": 0},
        "deferred_checks_not_run": ["gate_b", "gate_c"],
    }
    schedule, stopped_analysis, decision = finalizer.build_stop_documents(
        gate=gate_result,
        preregistration={"confirmatory_design": {"instances": 80}},
        collection_summary={
            "complete": True,
            "manifest_sha256": "m",
        },
        manifest_sha256="m",
        gate_sha256="g",
        collection_summary_sha256="c",
    )
    report = finalizer.render_report(
        gate=gate_result,
        schedule=schedule,
        analysis=stopped_analysis,
        decision=decision,
    )
    assert decision["decision"] == "PACT_ENVIRONMENT_INADEQUATE"
    assert not decision["policy_training_performed"]
    assert schedule["rows"] == []
    assert report.rstrip().splitlines()[-1] == "PACT_ENVIRONMENT_INADEQUATE"
