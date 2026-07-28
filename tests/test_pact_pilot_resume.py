from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


resume = _load(
    "analyze_pact_pilot_resume",
    ROOT / "scripts" / "analyze_pact_pilot_resume.py",
)
reporter = _load(
    "render_pact_environment_report_v2_resume",
    ROOT / "scripts" / "render_pact_environment_report_v2.py",
)


def _prior_gate():
    return {
        "schema_version": "pact_environment_gate_v2",
        "manifest_sha256": "manifest",
        "expert": {
            "attempts": 64,
            "scientific_outcomes": 62,
            "ordinary_task_success": 59,
            "usable_clean_demonstrations": 58,
            "usable_clean_demo_floor": 48,
            "usable_clean_demo_floor_met": True,
            "clean_demo_fraction_wilson_95": [0.81, 0.96],
            "no_scientific_outcome": 2,
            "no_scientific_outcome_rate": 0.03125,
            "no_scientific_outcome_wilson_95": [0.009, 0.107],
            "progression_target_strictly_below_5_percent": True,
            "status_counts": {"success": 59, "task_failure": 3, "sampling_failure": 2},
        },
        "surface_observability": {
            "robust_classification": "adequate",
            "point_estimates": {
                "active_episode_fraction": 1.0,
                "inside_20cm": 0.54,
                "inside_12cm": 0.18,
            },
            "intervals_95": {
                "active_episode_fraction_wilson": [0.94, 1.0],
                "inside_20cm_episode_cluster_bootstrap": [0.46, 0.62],
                "inside_12cm_episode_cluster_bootstrap": [0.13, 0.23],
            },
            "all_leave_one_episode_out_points_pass": True,
        },
    }


def _act_results(primary: int, hazard: int):
    results = []
    for index in range(64):
        results.append(
            {
                "status": "scientific_outcome",
                "task_success": index < max(primary, 48),
                "collision_free_task_success": index < primary,
                "failure_taxonomy": "collision_free_task_success",
                "contact_audit": {
                    "contact_class_totals": {
                        "grasp_target": 1,
                        "hazard_bar": int(primary <= index < primary + hazard),
                        "other_environment": 0,
                    }
                },
            }
        )
    return results


def _recovery():
    return {
        "prior_failed_dispatch": {
            "attempts": 64,
            "initial_observations": 0,
            "actions": 0,
            "scientific_outcomes": 0,
        },
        "retry_justification": {
            "fix_committed_before_any_policy_result": True,
            "path_resolution_content_independent": True,
        },
        "launch_smoke": {
            "schedule_index": 0,
            "rollout_id": "smoke",
            "passed": True,
        },
        "repaired_dispatch": {
            "scientific_results": 64,
            "post_boundary_failures": 0,
            "pre_observation_infrastructure_failures": 0,
            "scientific_rows_rerun": 0,
        },
    }


def test_resume_carries_settled_evidence_and_adjudicates_only_b_and_c():
    prior = _prior_gate()
    result = resume.analyze(
        manifest={"manifest_sha256": "manifest"},
        prior_gate=prior,
        act_results=_act_results(primary=32, hazard=20),
        schedule={"schedule_sha256": resume.EXPECTED_SCHEDULE_SHA256},
        recovery=_recovery(),
        prior_gate_sha256="prior",
    )
    assert result["expert"] == prior["expert"]
    assert result["surface_observability"] == prior["surface_observability"]
    assert result["settled_phase1_carried_forward_without_remeasurement"] == {
        "prior_gate_sha256": "prior",
        "expert": True,
        "surface_observability": True,
    }
    assert result["science_gate_classifications"] == {
        "surface_observability": "adequate",
        "gate_b": "adequate",
        "gate_c": "adequate",
    }
    assert result["decision"] == "PACT_ENVIRONMENT_ADEQUATE"


def test_recovery_report_explicitly_justifies_zero_outcome_retry():
    gate = resume.analyze(
        manifest={"manifest_sha256": "manifest"},
        prior_gate=_prior_gate(),
        act_results=_act_results(primary=32, hazard=20),
        schedule={"schedule_sha256": resume.EXPECTED_SCHEDULE_SHA256},
        recovery=_recovery(),
        prior_gate_sha256="prior",
    )
    text = reporter.render(
        gate,
        {"manifest_sha256": "manifest", "master_seed": 9},
        {"route": "collision", "environment_version": "corridor"},
    )
    assert "zero initial observations" in text
    assert "content-independent" in text
    assert "not outcome-based replacement" in text
    assert "carried forward" in text
