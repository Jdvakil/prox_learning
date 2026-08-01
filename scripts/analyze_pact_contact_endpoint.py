#!/usr/bin/env python3
"""Frozen analysis for the four-arm PACT contact-endpoint experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np


ARMS = ("ACT", "PACT", "PACT_ZERO", "PACT_PERMUTED")
SEEDS = (3101, 3102, 3103)
INSTANCES = 100
BOOTSTRAP_REPLICATES = 20000
BOOTSTRAP_SEED = 2026080105
PRIMARY_ARM_METRICS = (
    "collision_free_task_success",
    "hazard_bar_contact_frames",
)
CONTRASTS = (
    ("PACT", "PACT_PERMUTED", True, "modality_information"),
    ("PACT_PERMUTED", "ACT", False, "architecture_training_seed"),
    ("PACT", "ACT", False, "combined_policy_difference"),
    ("PACT", "PACT_ZERO", False, "sensor_failure_robustness_ood"),
    ("PACT_ZERO", "ACT", False, "failed_sensor_cost_ood"),
)
TOKENS = {
    "CONTACT_REDUCTION_ESTABLISHED",
    "CONTACT_REDUCTION_WITH_TASK_BENEFIT",
    "CONTACT_REDUCTION_SUBSET_ONLY",
    "NO_CONTACT_REDUCTION",
    "CONTACT_INCREASE",
    "CONTACT_EXPERIMENT_INCOMPLETE",
}


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wilson(successes: int, total: int) -> list[float | None]:
    if total == 0:
        return [None, None]
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return [center - half, center + half]


def collision_free_task_success(result: dict[str, Any]) -> bool:
    contacts = result["contact_audit"]["contact_class_totals"]
    return bool(
        result["task_success"]
        and int(contacts.get("hazard_bar", 0)) == 0
        and int(contacts.get("other_environment", 0)) == 0
    )


def contact_frames(result: dict[str, Any], contact_class: str) -> float:
    return float(result["contact_audit"]["frames_with_contact"][contact_class])


def contact_entries(result: dict[str, Any], contact_class: str) -> float:
    return float(result["contact_audit"]["contact_class_totals"][contact_class])


def maximum_penetration(result: dict[str, Any], contact_class: str) -> float:
    return float(
        result["contact_audit"]["maximum_penetration_depth_m"][contact_class]
    )


METRICS: dict[str, tuple[Callable[[dict[str, Any]], float], str]] = {
    "collision_free_task_success": (
        lambda row: float(collision_free_task_success(row)),
        "higher_is_better",
    ),
    "hazard_bar_contact_frames": (
        lambda row: contact_frames(row, "hazard_bar"),
        "lower_is_better",
    ),
    "hazard_bar_contact_entries": (
        lambda row: contact_entries(row, "hazard_bar"),
        "lower_is_better",
    ),
    "hazard_bar_any_contact": (
        lambda row: float(contact_entries(row, "hazard_bar") > 0),
        "lower_is_better",
    ),
    "other_environment_contact_frames": (
        lambda row: contact_frames(row, "other_environment"),
        "lower_is_better",
    ),
    "other_environment_contact_entries": (
        lambda row: contact_entries(row, "other_environment"),
        "lower_is_better",
    ),
    "hazard_bar_maximum_penetration_depth_m": (
        lambda row: maximum_penetration(row, "hazard_bar"),
        "lower_is_better",
    ),
    "other_environment_maximum_penetration_depth_m": (
        lambda row: maximum_penetration(row, "other_environment"),
        "lower_is_better",
    ),
    "non_target_maximum_penetration_depth_m": (
        lambda row: max(
            maximum_penetration(row, "hazard_bar"),
            maximum_penetration(row, "other_environment"),
        ),
        "lower_is_better",
    ),
    "manipulation_success": (
        lambda row: float(bool(row["task_success"])),
        "higher_is_better",
    ),
    "ordinary_task_success": (
        lambda row: float(bool(row["task_success"])),
        "higher_is_better",
    ),
}


def _summary(values: np.ndarray, *, binary: bool) -> dict[str, Any]:
    output: dict[str, Any] = {
        "n": int(len(values)),
        "mean": float(np.mean(values)) if len(values) else None,
        "median": float(np.median(values)) if len(values) else None,
        "minimum": float(np.min(values)) if len(values) else None,
        "maximum": float(np.max(values)) if len(values) else None,
    }
    if binary:
        successes = int(np.sum(values))
        output.update(
            {
                "count": successes,
                "rate": successes / len(values) if len(values) else None,
                "wilson_95": wilson(successes, len(values)),
            }
        )
    return output


def arm_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    binary_metrics = {
        "collision_free_task_success",
        "hazard_bar_any_contact",
        "manipulation_success",
        "ordinary_task_success",
    }
    summary = {}
    for name, (metric, direction) in METRICS.items():
        values = np.asarray([metric(row) for row in rows], dtype=np.float64)
        summary[name] = {
            **_summary(values, binary=name in binary_metrics),
            "direction": direction,
        }
    successful = [row for row in rows if bool(row["task_success"])]
    values = np.asarray(
        [contact_frames(row, "hazard_bar") for row in successful],
        dtype=np.float64,
    )
    summary["hazard_bar_contact_frames_conditioned_on_manipulation_success"] = {
        **_summary(values, binary=False),
        "direction": "lower_is_better",
        "conditioning": "this arm's manipulation_success is true",
    }
    summary["failure_taxonomy"] = dict(
        sorted(Counter(row["failure_taxonomy"] for row in rows).items())
    )
    return summary


def arm_mean_ci(
    values_by_cluster: np.ndarray, *, replicates: int, seed: int
) -> dict[str, Any]:
    """Bootstrap a per-arm mean while moving each whole instance together."""
    values = np.asarray(values_by_cluster, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("arm bootstrap requires a non-empty instance matrix")
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, values.shape[0], size=(replicates, values.shape[0]))
    bootstrap = values[sampled].mean(axis=(1, 2))
    return {
        "mean": float(values.mean()),
        "instance_cluster_bootstrap_ci_95": [
            float(np.percentile(bootstrap, 2.5)),
            float(np.percentile(bootstrap, 97.5)),
        ],
        "n_unique_instances": int(values.shape[0]),
        "n_seed_instance_observations": int(values.size),
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "cluster_unit": "instance; all included seed outcomes move together",
    }


def paired_difference(
    instances: list[dict[str, dict[str, Any]]],
    *,
    arm_a: str,
    arm_b: str,
    metric: Callable[[dict[str, Any]], float],
    replicates: int,
    seed: int,
    require_both_manipulation_success: bool = False,
) -> dict[str, Any]:
    eligible = instances
    if require_both_manipulation_success:
        eligible = [
            instance
            for instance in instances
            if bool(instance[arm_a]["task_success"])
            and bool(instance[arm_b]["task_success"])
        ]
    values = np.asarray(
        [metric(instance[arm_a]) - metric(instance[arm_b]) for instance in eligible],
        dtype=np.float64,
    )
    if not len(values):
        return {
            "arm_a": arm_a,
            "arm_b": arm_b,
            "n_instances": 0,
            "difference": None,
            "median_paired_difference": None,
            "instance_bootstrap_ci_95": [None, None],
            "bootstrap_replicates": replicates,
            "bootstrap_seed": seed,
            "conditioning": "both manipulation_success true",
        }
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(replicates, len(values)))
    bootstrap = values[indices].mean(axis=1)
    return {
        "arm_a": arm_a,
        "arm_b": arm_b,
        "n_instances": int(len(values)),
        "difference": float(np.mean(values)),
        "median_paired_difference": float(np.median(values)),
        "instance_bootstrap_ci_95": [
            float(np.percentile(bootstrap, 2.5)),
            float(np.percentile(bootstrap, 97.5)),
        ],
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "conditioning": (
            "both manipulation_success true"
            if require_both_manipulation_success
            else "unconditional"
        ),
    }


def pooled_cluster_difference(
    by_seed: dict[int, list[dict[str, dict[str, Any]]]],
    *,
    arm_a: str,
    arm_b: str,
    metric: Callable[[dict[str, Any]], float],
    replicates: int,
    seed: int,
    require_both_manipulation_success: bool = False,
) -> dict[str, Any]:
    rows = []
    included_pairs = 0
    for instance_index in range(INSTANCES):
        values = []
        for policy_seed in SEEDS:
            instance = by_seed[policy_seed][instance_index]
            if require_both_manipulation_success and not (
                bool(instance[arm_a]["task_success"])
                and bool(instance[arm_b]["task_success"])
            ):
                continue
            values.append(metric(instance[arm_a]) - metric(instance[arm_b]))
        included_pairs += len(values)
        rows.append(values)
    eligible_rows = [values for values in rows if values]
    if not eligible_rows:
        return {
            "arm_a": arm_a,
            "arm_b": arm_b,
            "difference": None,
            "instance_cluster_bootstrap_ci_95": [None, None],
            "n_unique_instances": 0,
            "n_seed_instance_pairs": 0,
            "cluster_unit": "instance; all available seeds and arms move together",
        }
    cluster_sums = np.asarray([sum(row) for row in eligible_rows], dtype=np.float64)
    cluster_counts = np.asarray([len(row) for row in eligible_rows], dtype=np.float64)
    observed = float(cluster_sums.sum() / cluster_counts.sum())
    rng = np.random.default_rng(seed)
    sampled = rng.integers(
        0, len(eligible_rows), size=(replicates, len(eligible_rows))
    )
    bootstrap = cluster_sums[sampled].sum(axis=1) / cluster_counts[sampled].sum(
        axis=1
    )
    return {
        "arm_a": arm_a,
        "arm_b": arm_b,
        "difference": observed,
        "median_seed_instance_difference": float(
            np.median([value for row in eligible_rows for value in row])
        ),
        "instance_cluster_bootstrap_ci_95": [
            float(np.percentile(bootstrap, 2.5)),
            float(np.percentile(bootstrap, 97.5)),
        ],
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "n_unique_instances": len(eligible_rows),
        "n_seed_instance_pairs": included_pairs,
        "cluster_unit": "instance; all available seeds and arms move together",
        "conditioning": (
            "both manipulation_success true"
            if require_both_manipulation_success
            else "unconditional"
        ),
    }


def choose_decision(
    reconciled: bool,
    *,
    modality_contact: dict[str, Any] | None,
    seed_modality_contacts: dict[int, dict[str, Any]] | None,
    pact_act_task: dict[str, Any] | None,
    seed_pact_act_task: dict[int, dict[str, Any]] | None,
) -> tuple[str, str]:
    if not reconciled:
        return "CONTACT_EXPERIMENT_INCOMPLETE", "schedule_did_not_reconcile"
    assert modality_contact is not None
    assert seed_modality_contacts is not None
    assert pact_act_task is not None
    assert seed_pact_act_task is not None
    difference = float(modality_contact["difference"])
    lower, upper = modality_contact["instance_cluster_bootstrap_ci_95"]
    seed_differences = [
        float(seed_modality_contacts[seed]["difference"]) for seed in SEEDS
    ]
    reduction = upper < 0.0 and all(value < 0.0 for value in seed_differences)
    if lower > 0.0:
        return "CONTACT_INCREASE", "modality_contact_ci_strictly_above_zero"
    if reduction:
        task_seed_differences = [
            float(seed_pact_act_task[seed]["difference"]) for seed in SEEDS
        ]
        if float(pact_act_task["difference"]) > 0.0 and all(
            value > 0.0 for value in task_seed_differences
        ):
            return (
                "CONTACT_REDUCTION_WITH_TASK_BENEFIT",
                "contact_reduction_and_positive_pact_minus_act_task_success_in_every_seed",
            )
        return (
            "CONTACT_REDUCTION_ESTABLISHED",
            "contact_reduction_ci_below_zero_with_consistent_seed_sign",
        )
    return (
        "NO_CONTACT_REDUCTION",
        (
            "contact_ci_includes_zero"
            if lower <= 0.0 <= upper
            else "contact_reduction_seed_sign_inconsistent"
        ),
    )


def validate_result(result: dict[str, Any], row: dict[str, Any]) -> None:
    expected = {
        "status": "complete",
        "rollout_id": row["rollout_id"],
        "schedule_row_sha256": row["schedule_row_sha256"],
        "episode_id": row["instance_episode_id"],
        "arm": row["arm"],
        "checkpoint_seed": row["checkpoint_seed"],
        "checkpoint_sha256": row["checkpoint_sha256"],
    }
    if any(result.get(key) != value for key, value in expected.items()):
        raise ValueError(f"identity:{row['schedule_index']}")
    audit = result.get("contact_audit", {})
    for key in (
        "contact_class_totals",
        "frames_with_contact",
        "maximum_penetration_depth_m",
    ):
        if set(audit.get(key, {})) != {
            "grasp_target",
            "hazard_bar",
            "other_environment",
        }:
            raise ValueError(f"contact_audit:{key}:{row['schedule_index']}")
    if collision_free_task_success(result) != bool(
        result["collision_free_task_success"]
    ):
        raise ValueError(f"endpoint:{row['schedule_index']}")


def load_results(
    schedule: dict[str, Any], output_root: Path
) -> tuple[dict[int, list[dict[str, dict[str, Any]]]], dict[str, Any]]:
    matrix: dict[int, dict[str, dict[str, dict[str, Any]]]] = {
        seed: {} for seed in SEEDS
    }
    errors = []
    for row in schedule["rows"]:
        row_dir = output_root / row["output_relpath"]
        result_path = row_dir / "result.json"
        driver_path = row_dir / "driver_result.json"
        if not result_path.is_file() or not driver_path.is_file():
            errors.append(f"missing:{row['schedule_index']}")
            continue
        try:
            driver = json.loads(driver_path.read_text())
            result = json.loads(result_path.read_text())
            if driver.get("status") != "complete":
                raise ValueError(f"driver:{row['schedule_index']}")
            validate_result(result, row)
            seed = int(row["checkpoint_seed"])
            episode_id = row["instance_episode_id"]
            arm = row["arm"]
            if arm in matrix[seed].setdefault(episode_id, {}):
                raise ValueError(f"duplicate:{seed}:{episode_id}:{arm}")
            matrix[seed][episode_id][arm] = result
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            errors.append(str(error))
    instance_order = [row["episode_id"] for row in schedule["instances"]]
    by_seed: dict[int, list[dict[str, dict[str, Any]]]] = {}
    valid_cells = 0
    for seed in SEEDS:
        by_seed[seed] = []
        for episode_id in instance_order:
            cell = matrix[seed].get(episode_id, {})
            valid_cells += len(cell)
            by_seed[seed].append(cell)
    expected_cells = INSTANCES * len(SEEDS) * len(ARMS)
    reconciled = (
        not errors
        and valid_cells == expected_cells
        and all(
            set(instance) == set(ARMS)
            for seed in SEEDS
            for instance in by_seed[seed]
        )
    )
    return by_seed, {
        "expected_cells": expected_cells,
        "valid_cells": valid_cells,
        "errors": errors,
        "reconciled": reconciled,
    }


def analyze(
    schedule: dict[str, Any], output_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    by_seed, reconciliation = load_results(schedule, output_root)
    base = {
        "schedule_sha256": schedule["schedule_sha256"],
        "occlusion_subset_sha256": schedule["occlusion_subset_sha256"],
        "population": "full_distribution_only",
        "occlusion_subset_analysis": "dropped_pre_rollout_because_partition_was_100_percent_vision_disadvantaged",
        "reconciliation": reconciliation,
    }
    if not reconciliation["reconciled"]:
        token, reason = choose_decision(
            False,
            modality_contact=None,
            seed_modality_contacts=None,
            pact_act_task=None,
            seed_pact_act_task=None,
        )
        analysis = {
            "schema_version": "pact_contact_endpoint_analysis_v1",
            **base,
            "results_available": False,
        }
        final = {
            "schema_version": "pact_contact_endpoint_decision_v1",
            "schedule_sha256": schedule["schedule_sha256"],
            "decision": token,
            "reason": reason,
        }
        return analysis, final

    seed_summaries = {}
    seed_primary_arm_intervals = {}
    seed_contrasts: dict[int, dict[str, dict[str, Any]]] = {}
    seed_conditioned = {}
    for seed_index, policy_seed in enumerate(SEEDS):
        instances = by_seed[policy_seed]
        seed_summaries[str(policy_seed)] = {
            arm: arm_summary([instance[arm] for instance in instances])
            for arm in ARMS
        }
        seed_primary_arm_intervals[str(policy_seed)] = {}
        for arm_index, arm in enumerate(ARMS):
            seed_primary_arm_intervals[str(policy_seed)][arm] = {}
            arm_rows = [instance[arm] for instance in instances]
            for metric_index, metric_name in enumerate(PRIMARY_ARM_METRICS):
                metric = METRICS[metric_name][0]
                seed_primary_arm_intervals[str(policy_seed)][arm][metric_name] = (
                    arm_mean_ci(
                        np.asarray([metric(row) for row in arm_rows]),
                        replicates=BOOTSTRAP_REPLICATES,
                        seed=(
                            BOOTSTRAP_SEED
                            + 40000
                            + seed_index * 100
                            + arm_index * 10
                            + metric_index
                        ),
                    )
                )
        seed_contrasts[policy_seed] = {}
        seed_conditioned[str(policy_seed)] = {}
        for contrast_index, (arm_a, arm_b, decision_bearing, interpretation) in enumerate(
            CONTRASTS
        ):
            contrast_name = f"{arm_a}_minus_{arm_b}"
            seed_contrasts[policy_seed][contrast_name] = {}
            for metric_index, (metric_name, (metric, direction)) in enumerate(
                METRICS.items()
            ):
                item = paired_difference(
                    instances,
                    arm_a=arm_a,
                    arm_b=arm_b,
                    metric=metric,
                    replicates=BOOTSTRAP_REPLICATES,
                    seed=BOOTSTRAP_SEED
                    + seed_index * 1000
                    + contrast_index * 100
                    + metric_index,
                )
                item.update(
                    {
                        "direction": direction,
                        "interpretation": interpretation,
                        "decision_bearing": decision_bearing
                        and metric_name == "hazard_bar_contact_frames",
                        "pact_zero_ood_probe": "PACT_ZERO" in (arm_a, arm_b),
                    }
                )
                seed_contrasts[policy_seed][contrast_name][metric_name] = item
            seed_conditioned[str(policy_seed)][contrast_name] = paired_difference(
                instances,
                arm_a=arm_a,
                arm_b=arm_b,
                metric=METRICS["hazard_bar_contact_frames"][0],
                replicates=BOOTSTRAP_REPLICATES,
                seed=BOOTSTRAP_SEED + 9000 + seed_index * 100 + contrast_index,
                require_both_manipulation_success=True,
            )

    pooled_contrasts = {}
    pooled_conditioned = {}
    pooled_arm_summaries = {
        arm: arm_summary(
            [
                by_seed[policy_seed][instance_index][arm]
                for instance_index in range(INSTANCES)
                for policy_seed in SEEDS
            ]
        )
        for arm in ARMS
    }
    pooled_primary_arm_intervals = {}
    for arm_index, arm in enumerate(ARMS):
        pooled_primary_arm_intervals[arm] = {}
        for metric_index, metric_name in enumerate(PRIMARY_ARM_METRICS):
            metric = METRICS[metric_name][0]
            matrix = np.asarray(
                [
                    [metric(by_seed[policy_seed][instance_index][arm]) for policy_seed in SEEDS]
                    for instance_index in range(INSTANCES)
                ],
                dtype=np.float64,
            )
            pooled_primary_arm_intervals[arm][metric_name] = arm_mean_ci(
                matrix,
                replicates=BOOTSTRAP_REPLICATES,
                seed=BOOTSTRAP_SEED + 50000 + arm_index * 10 + metric_index,
            )
    for contrast_index, (arm_a, arm_b, decision_bearing, interpretation) in enumerate(
        CONTRASTS
    ):
        contrast_name = f"{arm_a}_minus_{arm_b}"
        pooled_contrasts[contrast_name] = {}
        for metric_index, (metric_name, (metric, direction)) in enumerate(METRICS.items()):
            item = pooled_cluster_difference(
                by_seed,
                arm_a=arm_a,
                arm_b=arm_b,
                metric=metric,
                replicates=BOOTSTRAP_REPLICATES,
                seed=BOOTSTRAP_SEED + 20000 + contrast_index * 100 + metric_index,
            )
            item.update(
                {
                    "direction": direction,
                    "interpretation": interpretation,
                    "decision_bearing": decision_bearing
                    and metric_name == "hazard_bar_contact_frames",
                    "pact_zero_ood_probe": "PACT_ZERO" in (arm_a, arm_b),
                }
            )
            pooled_contrasts[contrast_name][metric_name] = item
        pooled_conditioned[contrast_name] = pooled_cluster_difference(
            by_seed,
            arm_a=arm_a,
            arm_b=arm_b,
            metric=METRICS["hazard_bar_contact_frames"][0],
            replicates=BOOTSTRAP_REPLICATES,
            seed=BOOTSTRAP_SEED + 29000 + contrast_index,
            require_both_manipulation_success=True,
        )

    modality_contact = pooled_contrasts["PACT_minus_PACT_PERMUTED"][
        "hazard_bar_contact_frames"
    ]
    seed_modality = {
        seed: seed_contrasts[seed]["PACT_minus_PACT_PERMUTED"][
            "hazard_bar_contact_frames"
        ]
        for seed in SEEDS
    }
    pact_act_task = pooled_contrasts["PACT_minus_ACT"][
        "collision_free_task_success"
    ]
    seed_pact_act_task = {
        seed: seed_contrasts[seed]["PACT_minus_ACT"][
            "collision_free_task_success"
        ]
        for seed in SEEDS
    }
    token, reason = choose_decision(
        True,
        modality_contact=modality_contact,
        seed_modality_contacts=seed_modality,
        pact_act_task=pact_act_task,
        seed_pact_act_task=seed_pact_act_task,
    )
    analysis = {
        "schema_version": "pact_contact_endpoint_analysis_v1",
        **base,
        "results_available": True,
        "arm_labels": {
            "PACT_ZERO": "OOD sensor-failure probe; never modality evidence",
            "PACT_PERMUTED": "distribution-matched modality-information instrument",
        },
        "seed_summaries": seed_summaries,
        "seed_primary_arm_instance_bootstrap_intervals": seed_primary_arm_intervals,
        "seed_contrasts": {str(key): value for key, value in seed_contrasts.items()},
        "seed_contact_conditioned_on_both_manipulations_succeeding": seed_conditioned,
        "pooled_contrasts": pooled_contrasts,
        "pooled_arm_summaries": pooled_arm_summaries,
        "pooled_primary_arm_instance_cluster_bootstrap_intervals": pooled_primary_arm_intervals,
        "pooled_contact_conditioned_on_both_manipulations_succeeding": pooled_conditioned,
        "decision_rule_inputs": {
            "modality_contact": modality_contact,
            "seed_modality_contacts": {str(key): value for key, value in seed_modality.items()},
            "pact_minus_act_collision_free_task_success": pact_act_task,
            "seed_pact_minus_act_collision_free_task_success": {
                str(key): value for key, value in seed_pact_act_task.items()
            },
            "subset_token_available": False,
        },
    }
    final = {
        "schema_version": "pact_contact_endpoint_decision_v1",
        "schedule_sha256": schedule["schedule_sha256"],
        "decision": token,
        "reason": reason,
        "pact_zero_used_for_modality_attribution": False,
        "subset_analysis_performed": False,
    }
    return analysis, final


def _percent(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _difference(value: float, *, percentage: bool) -> str:
    return f"{100.0 * value:+.1f} pp" if percentage else f"{value:+.1f}"


def render_report(analysis: dict[str, Any], decision: dict[str, Any]) -> str:
    token = decision["decision"]
    lines = [
        "# PACT contact-endpoint decision",
        "",
        f"Decision: `{token}`.",
        "",
        "The decision-bearing modality contrast is PACT minus PACT_PERMUTED on hazard-bar contact frames. Negative values favor PACT. PACT_ZERO is an out-of-distribution sensor-failure probe and is never used as modality evidence.",
        "",
        "The predeclared wrist-camera partition was 285/285 vision-disadvantaged, so the subset analysis was dropped before rollout outcomes existed.",
        "",
    ]
    if not analysis.get("results_available"):
        lines.extend(
            [
                "The 1,200-row frozen schedule did not reconcile, so no endpoint comparison was made.",
                "",
                token,
            ]
        )
        return "\n".join(lines) + "\n"
    lines.extend(["## Results by policy seed", ""])
    for policy_seed in SEEDS:
        seed_key = str(policy_seed)
        lines.extend(
            [
                f"### Seed {policy_seed}",
                "",
                "| Arm | Collision-free task success | Hazard frames, mean (median) |",
                "|---|---:|---:|",
            ]
        )
        for arm in ARMS:
            summary = analysis["seed_summaries"][seed_key][arm]
            cf = summary["collision_free_task_success"]
            frames = summary["hazard_bar_contact_frames"]
            intervals = analysis["seed_primary_arm_instance_bootstrap_intervals"][
                seed_key
            ][arm]
            cf_ci = intervals["collision_free_task_success"][
                "instance_cluster_bootstrap_ci_95"
            ]
            frame_ci = intervals["hazard_bar_contact_frames"][
                "instance_cluster_bootstrap_ci_95"
            ]
            label = (
                "PACT_ZERO (OOD sensor failure)" if arm == "PACT_ZERO" else arm
            )
            lines.append(
                f"| {label} | {cf['count']}/{cf['n']} ({_percent(cf['rate'])}; "
                f"95% CI [{_percent(cf_ci[0])}, {_percent(cf_ci[1])}]) | "
                f"{frames['mean']:.1f} (95% CI [{frame_ci[0]:.1f}, {frame_ci[1]:.1f}]; "
                f"median {frames['median']:.1f}) |"
            )
        contact = analysis["seed_contrasts"][seed_key]["PACT_minus_PACT_PERMUTED"][
            "hazard_bar_contact_frames"
        ]
        contact_ci = contact["instance_bootstrap_ci_95"]
        task = analysis["seed_contrasts"][seed_key]["PACT_minus_ACT"][
            "collision_free_task_success"
        ]
        task_ci = task["instance_bootstrap_ci_95"]
        lines.extend(
            [
                "",
                f"PACT − PACT_PERMUTED hazard frames: {_difference(contact['difference'], percentage=False)} "
                f"(instance-bootstrap 95% CI [{contact_ci[0]:+.1f}, {contact_ci[1]:+.1f}]).",
                f"PACT − ACT collision-free task success: {_difference(task['difference'], percentage=True)} "
                f"(instance-bootstrap 95% CI [{100 * task_ci[0]:+.1f}, {100 * task_ci[1]:+.1f}] pp).",
                "",
            ]
        )
    modality = analysis["pooled_contrasts"]["PACT_minus_PACT_PERMUTED"][
        "hazard_bar_contact_frames"
    ]
    modality_ci = modality["instance_cluster_bootstrap_ci_95"]
    pact_act = analysis["pooled_contrasts"]["PACT_minus_ACT"][
        "collision_free_task_success"
    ]
    pact_act_ci = pact_act["instance_cluster_bootstrap_ci_95"]
    conditioned = analysis[
        "pooled_contact_conditioned_on_both_manipulations_succeeding"
    ]["PACT_minus_PACT_PERMUTED"]
    conditioned_ci = conditioned["instance_cluster_bootstrap_ci_95"]
    lines.extend(
        [
            "## Pooled after the seed-specific results",
            "",
            "Whole task instances are the bootstrap clusters; all arms and all three seed outcomes for a sampled instance move together.",
            "",
            "| Arm | Collision-free task success | Hazard frames, mean (median) | Task success | Any hazard contact |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for arm in ARMS:
        summary = analysis["pooled_arm_summaries"][arm]
        intervals = analysis[
            "pooled_primary_arm_instance_cluster_bootstrap_intervals"
        ][arm]
        cf = summary["collision_free_task_success"]
        frames = summary["hazard_bar_contact_frames"]
        task = summary["ordinary_task_success"]
        any_contact = summary["hazard_bar_any_contact"]
        cf_ci = intervals["collision_free_task_success"][
            "instance_cluster_bootstrap_ci_95"
        ]
        frame_ci = intervals["hazard_bar_contact_frames"][
            "instance_cluster_bootstrap_ci_95"
        ]
        label = "PACT_ZERO (OOD sensor failure)" if arm == "PACT_ZERO" else arm
        lines.append(
            f"| {label} | {cf['count']}/{cf['n']} ({_percent(cf['rate'])}; "
            f"95% CI [{_percent(cf_ci[0])}, {_percent(cf_ci[1])}]) | "
            f"{frames['mean']:.1f} (95% CI [{frame_ci[0]:.1f}, {frame_ci[1]:.1f}]; "
            f"median {frames['median']:.1f}) | {_percent(task['rate'])} | "
            f"{_percent(any_contact['rate'])} |"
        )
    lines.extend(
        [
            "",
            f"PACT − PACT_PERMUTED hazard frames: {_difference(modality['difference'], percentage=False)} "
            f"(95% CI [{modality_ci[0]:+.1f}, {modality_ci[1]:+.1f}]).",
            "",
            f"PACT − ACT collision-free task success: {_difference(pact_act['difference'], percentage=True)} "
            f"(95% CI [{100 * pact_act_ci[0]:+.1f}, {100 * pact_act_ci[1]:+.1f}] pp).",
            "",
            f"Diagnostic PACT − PACT_PERMUTED hazard frames when both arms succeeded at manipulation: "
            f"{_difference(conditioned['difference'], percentage=False) if conditioned['difference'] is not None else 'not estimable'}"
            + (
                f" (95% CI [{conditioned_ci[0]:+.1f}, {conditioned_ci[1]:+.1f}])."
                if conditioned_ci[0] is not None
                else "."
            ),
            "",
            "## Pooled full contrast set",
            "",
            "Negative contact/count/depth differences favor the first arm; positive success differences favor it. PACT_ZERO rows remain OOD diagnostics.",
            "",
            "| Contrast | Endpoint | Difference | Whole-instance 95% CI |",
            "|---|---|---:|---:|",
        ]
    )
    binary = {
        "collision_free_task_success",
        "hazard_bar_any_contact",
        "manipulation_success",
        "ordinary_task_success",
    }
    for arm_a, arm_b, _bearing, _interpretation in CONTRASTS:
        contrast_name = f"{arm_a}_minus_{arm_b}"
        label = contrast_name.replace("_minus_", " − ")
        if "PACT_ZERO" in (arm_a, arm_b):
            label += " (OOD)"
        for metric_name in METRICS:
            item = analysis["pooled_contrasts"][contrast_name][metric_name]
            ci = item["instance_cluster_bootstrap_ci_95"]
            percentage = metric_name in binary
            if percentage:
                difference = _difference(item["difference"], percentage=True)
                interval = f"[{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}] pp"
            elif "penetration_depth_m" in metric_name:
                difference = f"{item['difference']:+.6f} m"
                interval = f"[{ci[0]:+.6f}, {ci[1]:+.6f}] m"
            else:
                difference = _difference(item["difference"], percentage=False)
                interval = f"[{ci[0]:+.4g}, {ci[1]:+.4g}]"
            lines.append(f"| {label} | {metric_name} | {difference} | {interval} |")
    lines.extend(
        [
            "",
            "Failure taxonomies and every seed-specific full contrast are retained in the frozen `analysis.json`; the tables above show seeds first and the complete pooled contrast family.",
            "",
            "## Interpretation",
            "",
            decision["reason"].replace("_", " ").capitalize() + ".",
            "",
            token,
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--analysis-output", required=True, type=Path)
    parser.add_argument("--decision-output", required=True, type=Path)
    parser.add_argument("--report-output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    schedule = json.loads(args.schedule.read_text())
    payload = dict(schedule)
    observed = payload.pop("schedule_sha256", None)
    if observed != canonical_hash(payload):
        raise SystemExit("schedule self-hash mismatch")
    analysis, decision = analyze(schedule, args.output_root)
    analysis.pop("analysis_sha256", None)
    analysis["analysis_script_sha256"] = file_hash(Path(__file__).resolve())
    analysis["analysis_sha256"] = canonical_hash(analysis)
    decision["analysis_sha256"] = analysis["analysis_sha256"]
    decision.pop("final_decision_sha256", None)
    decision["final_decision_sha256"] = canonical_hash(decision)
    args.analysis_output.parent.mkdir(parents=True, exist_ok=True)
    args.analysis_output.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    args.decision_output.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    if args.report_output is not None:
        args.report_output.parent.mkdir(parents=True, exist_ok=True)
        args.report_output.write_text(render_report(analysis, decision))
    print(decision["decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
