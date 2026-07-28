"""Pure statistical rules frozen for the PACT remediation-v2 pilot gate."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

Z_95 = 1.959963984540054
SURFACE_ACTIVE_MIN = 5.0 / 6.0
SURFACE_20CM_MIN = 0.30
SURFACE_12CM_MIN = 0.05
GATE_B_LOW = 1.0 / 3.0
GATE_B_HIGH = 2.0 / 3.0
GATE_C_CONTACT_MIN = 0.25


def wilson_interval(successes: int, total: int, z: float = Z_95) -> tuple[float, float]:
    """Two-sided Wilson score interval for a binary proportion."""
    if total <= 0:
        raise ValueError("Wilson interval requires a positive denominator")
    if not 0 <= successes <= total:
        raise ValueError(f"successes must be in [0, total], got {successes}/{total}")
    p = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(p * (1.0 - p) / total + z2 / (4.0 * total * total))
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def gate_b_core(successes: int, total: int) -> str:
    """Classify the ACT collision-free rate before the one-row robustness rule."""
    if total <= 0:
        return "inconclusive"
    point = successes / total
    lower, upper = wilson_interval(successes, total)
    if GATE_B_LOW <= point <= GATE_B_HIGH and lower > 0.20 and upper < 0.80:
        return "adequate"
    if point < GATE_B_LOW and upper < 0.50:
        return "floor"
    if point > GATE_B_HIGH and lower > 0.50:
        return "ceiling"
    return "inconclusive"


def gate_c_core(contact_episodes: int, total: int) -> str:
    """Classify whether ACT contacts the intrusion often enough to leave headroom."""
    if total <= 0:
        return "inconclusive"
    point = contact_episodes / total
    lower, upper = wilson_interval(contact_episodes, total)
    if point >= GATE_C_CONTACT_MIN and lower > 0.10:
        return "adequate"
    if point < GATE_C_CONTACT_MIN and upper < 0.40:
        return "no_collision_headroom"
    return "inconclusive"


def one_outcome_robust_classification(
    successes: int,
    total: int,
    classifier: Callable[[int, int], str],
) -> dict[str, Any]:
    """Require the core label to survive either possible single-row outcome flip."""
    candidates = sorted({max(0, successes - 1), successes, min(total, successes + 1)})
    labels = {str(candidate): classifier(candidate, total) for candidate in candidates}
    observed = classifier(successes, total)
    stable = observed != "inconclusive" and set(labels.values()) == {observed}
    return {
        "successes": successes,
        "n": total,
        "point_estimate": successes / total if total else None,
        "wilson_95": list(wilson_interval(successes, total)) if total else None,
        "core_classification": observed,
        "one_outcome_perturbations": labels,
        "robust_classification": observed if stable else "inconclusive",
        "one_outcome_stable": stable,
    }


def cluster_ratio_interval(
    numerators: Sequence[int],
    denominators: Sequence[int],
    *,
    seed: int,
    replicates: int = 20_000,
) -> tuple[float, float]:
    """Bootstrap a ratio of sums by resampling whole episodes."""
    numerator = np.asarray(numerators, dtype=np.int64)
    denominator = np.asarray(denominators, dtype=np.int64)
    if numerator.shape != denominator.shape or numerator.ndim != 1:
        raise ValueError("numerators and denominators must be equal-length vectors")
    if len(numerator) < 2:
        raise ValueError("cluster bootstrap requires at least two episodes")
    if np.any(numerator < 0) or np.any(denominator < 0) or np.any(numerator > denominator):
        raise ValueError("invalid episode numerator/denominator")
    if denominator.sum() <= 0:
        raise ValueError("cluster bootstrap requires positive total denominator")
    rng = np.random.default_rng(seed)
    ratios = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, 1_000):
        stop = min(replicates, start + 1_000)
        indices = rng.integers(0, len(numerator), size=(stop - start, len(numerator)))
        sampled_num = numerator[indices].sum(axis=1)
        sampled_den = denominator[indices].sum(axis=1)
        ratios[start:stop] = np.divide(
            sampled_num,
            sampled_den,
            out=np.zeros(stop - start, dtype=np.float64),
            where=sampled_den > 0,
        )
    return float(np.quantile(ratios, 0.025)), float(np.quantile(ratios, 0.975))


def _surface_points(episodes: Sequence[dict[str, int]]) -> dict[str, float]:
    count = len(episodes)
    if count == 0:
        raise ValueError("surface gate requires outcome-bearing episodes")
    pregrasp = sum(int(row["pregrasp_control_steps"]) for row in episodes)
    if pregrasp <= 0:
        return {"active_episode_fraction": 0.0, "inside_20cm": 0.0, "inside_12cm": 0.0}
    return {
        "active_episode_fraction": (
            sum(bool(row["episode_has_intrusion_sighting"]) for row in episodes) / count
        ),
        "inside_20cm": (
            sum(int(row["steps_intrusion_inside_20cm"]) for row in episodes) / pregrasp
        ),
        "inside_12cm": (
            sum(int(row["steps_intrusion_inside_12cm"]) for row in episodes) / pregrasp
        ),
    }


def _surface_point_passes(points: dict[str, float]) -> bool:
    return (
        points["active_episode_fraction"] >= SURFACE_ACTIVE_MIN
        and points["inside_20cm"] >= SURFACE_20CM_MIN
        and points["inside_12cm"] >= SURFACE_12CM_MIN
    )


def classify_surface_observability(
    episodes: Sequence[dict[str, int]],
    *,
    seed: int = 2026072903,
    replicates: int = 20_000,
) -> dict[str, Any]:
    """Apply the frozen surface thresholds, intervals, and leave-one-row rule."""
    if len(episodes) < 2:
        return {
            "n": len(episodes),
            "robust_classification": "inconclusive",
            "reason": "fewer_than_two_outcome_bearing_episodes",
        }
    points = _surface_points(episodes)
    active = sum(bool(row["episode_has_intrusion_sighting"]) for row in episodes)
    active_interval = wilson_interval(active, len(episodes))
    denominators = [int(row["pregrasp_control_steps"]) for row in episodes]
    inside20 = [int(row["steps_intrusion_inside_20cm"]) for row in episodes]
    inside12 = [int(row["steps_intrusion_inside_12cm"]) for row in episodes]
    interval20 = cluster_ratio_interval(
        inside20, denominators, seed=seed, replicates=replicates
    )
    interval12 = cluster_ratio_interval(
        inside12, denominators, seed=seed + 1, replicates=replicates
    )
    leave_one_out = [
        _surface_point_passes(_surface_points([row for j, row in enumerate(episodes) if j != i]))
        for i in range(len(episodes))
    ]
    point_passes = _surface_point_passes(points)
    robust_pass = point_passes and all(leave_one_out)
    robust_fail_reasons = []
    if active_interval[1] < SURFACE_ACTIVE_MIN:
        robust_fail_reasons.append("active_episode_wilson_upper_below_threshold")
    if interval20[1] < SURFACE_20CM_MIN:
        robust_fail_reasons.append("inside_20cm_bootstrap_upper_below_threshold")
    if interval12[1] < SURFACE_12CM_MIN:
        robust_fail_reasons.append("inside_12cm_bootstrap_upper_below_threshold")
    classification = (
        "adequate"
        if robust_pass
        else ("insufficient_surface_signal" if robust_fail_reasons else "inconclusive")
    )
    return {
        "n": len(episodes),
        "active_episodes": active,
        "pregrasp_control_steps": sum(denominators),
        "steps_intrusion_inside_20cm": sum(inside20),
        "steps_intrusion_inside_12cm": sum(inside12),
        "point_estimates": points,
        "intervals_95": {
            "active_episode_fraction_wilson": list(active_interval),
            "inside_20cm_episode_cluster_bootstrap": list(interval20),
            "inside_12cm_episode_cluster_bootstrap": list(interval12),
        },
        "thresholds": {
            "active_episode_fraction_min": SURFACE_ACTIVE_MIN,
            "inside_20cm_min": SURFACE_20CM_MIN,
            "inside_12cm_min": SURFACE_12CM_MIN,
        },
        "all_leave_one_episode_out_points_pass": all(leave_one_out),
        "leave_one_episode_out_failures": sum(not value for value in leave_one_out),
        "robust_fail_reasons": robust_fail_reasons,
        "robust_classification": classification,
    }


def environment_decision(
    *,
    surface_classification: str,
    gate_b_classification: str,
    gate_c_classification: str,
    usable_demo_floor_met: bool,
    infrastructure_progression_met: bool,
    minimum_scientific_rows_met: bool,
) -> str:
    """Map independent science, demo, and infrastructure concerns to a token."""
    science = (surface_classification, gate_b_classification, gate_c_classification)
    science_failures = {
        "insufficient_surface_signal",
        "floor",
        "ceiling",
        "no_collision_headroom",
    }
    if any(label in science_failures for label in science):
        return "PACT_ENVIRONMENT_INADEQUATE"
    if (
        science == ("adequate", "adequate", "adequate")
        and usable_demo_floor_met
        and infrastructure_progression_met
        and minimum_scientific_rows_met
    ):
        return "PACT_ENVIRONMENT_ADEQUATE"
    return "PACT_EXPERIMENT_INCOMPLETE"
