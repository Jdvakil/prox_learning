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
