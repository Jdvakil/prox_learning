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


stats = _load(
    "pact_gate_statistics",
    ROOT / "scripts" / "pact_gate_statistics.py",
)
reporter = _load(
    "render_pact_environment_report_v2",
    ROOT / "scripts" / "render_pact_environment_report_v2.py",
)


def test_gate_b_is_interval_based_and_one_outcome_robust():
    assert stats.one_outcome_robust_classification(
        32, 64, stats.gate_b_core
    )["robust_classification"] == "adequate"
    assert stats.one_outcome_robust_classification(
        20, 64, stats.gate_b_core
    )["robust_classification"] == "floor"
    assert stats.one_outcome_robust_classification(
        21, 64, stats.gate_b_core
    )["robust_classification"] == "inconclusive"
    assert stats.one_outcome_robust_classification(
        44, 64, stats.gate_b_core
    )["robust_classification"] == "ceiling"


def test_gate_c_preserves_25_percent_intent_with_stability_zone():
    assert stats.one_outcome_robust_classification(
        14, 64, stats.gate_c_core
    )["robust_classification"] == "no_collision_headroom"
    assert stats.one_outcome_robust_classification(
        15, 64, stats.gate_c_core
    )["robust_classification"] == "inconclusive"
    assert stats.one_outcome_robust_classification(
        17, 64, stats.gate_c_core
    )["robust_classification"] == "adequate"


def test_surface_gate_bootstraps_whole_episodes_and_has_robust_failure():
    strong = [
        {
            "pregrasp_control_steps": 100,
            "steps_intrusion_inside_20cm": 50,
            "steps_intrusion_inside_12cm": 20,
            "episode_has_intrusion_sighting": True,
        }
        for _ in range(12)
    ]
    strong_result = stats.classify_surface_observability(
        strong, seed=9, replicates=1_000
    )
    assert strong_result["robust_classification"] == "adequate"
    assert strong_result["all_leave_one_episode_out_points_pass"] is True

    absent = [
        {
            "pregrasp_control_steps": 100,
            "steps_intrusion_inside_20cm": 0,
            "steps_intrusion_inside_12cm": 0,
            "episode_has_intrusion_sighting": False,
        }
        for _ in range(12)
    ]
    absent_result = stats.classify_surface_observability(
        absent, seed=9, replicates=1_000
    )
    assert absent_result["robust_classification"] == "insufficient_surface_signal"


def test_demo_and_infrastructure_cannot_award_environment_inadequate():
    incomplete_demo = stats.environment_decision(
        surface_classification="adequate",
        gate_b_classification="adequate",
        gate_c_classification="adequate",
        usable_demo_floor_met=False,
        infrastructure_progression_met=True,
        minimum_scientific_rows_met=True,
    )
    incomplete_harness = stats.environment_decision(
        surface_classification="adequate",
        gate_b_classification="adequate",
        gate_c_classification="adequate",
        usable_demo_floor_met=True,
        infrastructure_progression_met=False,
        minimum_scientific_rows_met=True,
    )
    assert incomplete_demo == "PACT_EXPERIMENT_INCOMPLETE"
    assert incomplete_harness == "PACT_EXPERIMENT_INCOMPLETE"
    assert stats.environment_decision(
        surface_classification="adequate",
        gate_b_classification="floor",
        gate_c_classification="adequate",
        usable_demo_floor_met=True,
        infrastructure_progression_met=True,
        minimum_scientific_rows_met=True,
    ) == "PACT_ENVIRONMENT_INADEQUATE"


def test_corridor_expert_fix_does_not_change_scene_geometry():
    source = (
        ROOT
        / "submodules"
        / "molmospaces"
        / "molmo_spaces"
        / "tasks"
        / "enclosure_reach.py"
    ).read_text()
    policy = source[source.index("class PactCollisionCorridorPolicy(") :]
    assert "SAFE_GAP = 0.10" in policy
    assert "max_retries: int = 0" in policy
    sampler = source[
        source.index("class PactCollisionCorridorSampler(") :
        source.index("class PactCollisionCorridorPolicy(")
    ]
    assert "PANEL_X = 0.615" in sampler
    assert "PANEL_Z = 0.89" in sampler
    assert "PANEL_INNER_FACE_Y = 0.100" in sampler
    assert "SASH_APERTURE_HEIGHT = 0.70" in sampler


def test_v2_environment_report_preserves_exact_last_line():
    gate = {
        "schema_version": "pact_environment_gate_v2",
        "decision": "PACT_ENVIRONMENT_ADEQUATE",
        "manifest_sha256": "m",
        "expert": {
            "attempts": 64,
            "scientific_outcomes": 64,
            "ordinary_task_success": 60,
            "usable_clean_demonstrations": 58,
            "usable_clean_demo_floor": 48,
            "clean_demo_fraction_wilson_95": [0.81, 0.96],
            "no_scientific_outcome": 0,
            "no_scientific_outcome_wilson_95": [0.0, 0.056],
        },
        "surface_observability": {
            "point_estimates": {
                "active_episode_fraction": 0.9,
                "inside_20cm": 0.5,
                "inside_12cm": 0.2,
            },
            "intervals_95": {
                "active_episode_fraction_wilson": [0.8, 0.95],
                "inside_20cm_episode_cluster_bootstrap": [0.4, 0.6],
                "inside_12cm_episode_cluster_bootstrap": [0.1, 0.3],
            },
            "all_leave_one_episode_out_points_pass": True,
        },
        "act": {
            "scientific_outcomes": 64,
            "collision_free_task_success": 32,
            "ordinary_task_success": 48,
            "ordinary_task_success_rate": 0.75,
            "ordinary_task_success_wilson_95": [0.63, 0.84],
            "episodes_with_hazard_bar_contact": 20,
            "episodes_with_other_environment_contact": 2,
        },
        "gate_b": {
            "point_estimate": 0.5,
            "wilson_95": [0.38, 0.62],
            "one_outcome_stable": True,
        },
        "gate_c": {
            "point_estimate": 0.3125,
            "wilson_95": [0.21, 0.43],
            "one_outcome_stable": True,
        },
        "science_gate_classifications": {
            "surface_observability": "adequate",
            "gate_b": "adequate",
            "gate_c": "adequate",
        },
    }
    text = reporter.render(
        gate,
        {"manifest_sha256": "m", "master_seed": 9},
        {"route": "collision", "environment_version": "corridor"},
    )
    assert "neither rescored nor pooled" in text
    assert text.rstrip().splitlines()[-1] == "PACT_ENVIRONMENT_ADEQUATE"
