#!/usr/bin/env python3
"""Frozen paired-instance analysis for the inference-time RGB blur sweep."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np


ARMS = ("ACT", "PACT", "PACT_PERMUTED")
SEEDS = (3101, 3102, 3103)
CONTRASTS = (
    ("PACT", "ACT", "PACT_minus_ACT"),
    ("PACT", "PACT_PERMUTED", "PACT_minus_PACT_PERMUTED"),
    ("PACT_PERMUTED", "ACT", "PACT_PERMUTED_minus_ACT"),
)
TOKENS = {
    "BLUR_ROBUSTNESS_ESTABLISHED",
    "BLUR_PARTIAL",
    "NO_BLUR_ROBUSTNESS",
    "BLUR_UNINFORMATIVE_COLLAPSE",
    "BLUR_EXPERIMENT_INCOMPLETE",
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


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def wilson(successes: int, total: int) -> list[float | None]:
    if total == 0:
        return [None, None]
    z = 1.959963984540054
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    half = z * math.sqrt(
        rate * (1 - rate) / total + z * z / (4 * total * total)
    ) / denominator
    return [center - half, center + half]


def collision_free_task_success(result: dict[str, Any]) -> float:
    totals = result["contact_audit"]["contact_class_totals"]
    return float(
        bool(result["task_success"])
        and int(totals["hazard_bar"]) == 0
        and int(totals["other_environment"]) == 0
    )


METRICS: dict[str, tuple[Callable[[dict[str, Any]], float], bool, str]] = {
    "collision_free_task_success": (
        collision_free_task_success,
        True,
        "higher_is_better",
    ),
    "hazard_bar_contact_frames": (
        lambda row: float(row["contact_audit"]["frames_with_contact"]["hazard_bar"]),
        False,
        "lower_is_better",
    ),
    "hazard_bar_any_contact": (
        lambda row: float(
            int(row["contact_audit"]["contact_class_totals"]["hazard_bar"]) > 0
        ),
        True,
        "lower_is_better",
    ),
    "non_target_contact_entries": (
        lambda row: float(row["contact_audit"]["non_target_contact_entries"]),
        False,
        "lower_is_better",
    ),
    "hazard_bar_maximum_penetration_depth_m": (
        lambda row: float(
            row["contact_audit"]["maximum_penetration_depth_m"]["hazard_bar"]
        ),
        False,
        "lower_is_better",
    ),
    "task_success": (
        lambda row: float(bool(row["task_success"])),
        True,
        "higher_is_better",
    ),
}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, (metric, binary, direction) in METRICS.items():
        values = np.asarray([metric(row) for row in rows], dtype=np.float64)
        item: dict[str, Any] = {
            "n": len(rows),
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "direction": direction,
        }
        if binary:
            count = int(values.sum())
            item.update(
                {
                    "count": count,
                    "rate": count / len(values),
                    "wilson_95": wilson(count, len(values)),
                }
            )
        output[name] = item
    output["failure_taxonomy"] = dict(
        sorted(Counter(row["failure_taxonomy"] for row in rows).items())
    )
    return output


def bootstrap_cluster_mean(
    cluster_values: list[list[float]], *, replicates: int, seed: int
) -> dict[str, Any]:
    sums = np.asarray([sum(values) for values in cluster_values], dtype=np.float64)
    counts = np.asarray([len(values) for values in cluster_values], dtype=np.float64)
    point = float(sums.sum() / counts.sum())
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(sums), size=(replicates, len(sums)))
    boot = sums[sampled].sum(axis=1) / counts[sampled].sum(axis=1)
    return {
        "difference": point,
        "instance_cluster_bootstrap_ci_95": [
            float(np.percentile(boot, 2.5)),
            float(np.percentile(boot, 97.5)),
        ],
        "n_unique_instances": len(cluster_values),
        "n_paired_values": int(counts.sum()),
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "cluster_unit": "instance; all sigmas, arms, and seeds move together",
    }


def validate_result(result: dict[str, Any], row: dict[str, Any]) -> None:
    expected = {
        "status": "complete",
        "rollout_id": row["rollout_id"],
        "schedule_row_sha256": row["schedule_row_sha256"],
        "episode_id": row["instance_episode_id"],
        "arm": row["arm"],
        "checkpoint_seed": row["checkpoint_seed"],
        "checkpoint_sha256": row["checkpoint_sha256"],
        "blur_sigma": row["blur_sigma"],
    }
    for key, value in expected.items():
        if result.get(key) != value:
            raise ValueError(f"row {row['schedule_index']} identity mismatch: {key}")
    if result.get("policy_info", {}).get("blur_sigma") != row["blur_sigma"]:
        raise ValueError(f"row {row['schedule_index']} policy blur sigma mismatch")
    audit = result.get("contact_audit", {})
    expected_classes = {"grasp_target", "hazard_bar", "other_environment"}
    for section in (
        "contact_class_totals",
        "frames_with_contact",
        "maximum_penetration_depth_m",
    ):
        if set(audit.get(section, {})) != expected_classes:
            raise ValueError(f"row {row['schedule_index']} contact taxonomy changed")
    if bool(collision_free_task_success(result)) != bool(
        result["collision_free_task_success"]
    ):
        raise ValueError(f"row {row['schedule_index']} collision-free endpoint mismatch")


def load_matrix(
    schedule: dict[str, Any], output_root: Path
) -> tuple[dict[int, dict[float, dict[int, dict[str, dict[str, Any]]]]], dict[str, Any]]:
    matrix: dict[int, dict[float, dict[int, dict[str, dict[str, Any]]]]] = {}
    errors = []
    for row in schedule["rows"]:
        row_dir = output_root / row["output_relpath"]
        result_path = row_dir / "result.json"
        driver_path = row_dir / "driver_result.json"
        if not result_path.is_file() or not driver_path.is_file():
            errors.append(f"missing:{row['schedule_index']}")
            continue
        try:
            result = json.loads(result_path.read_text())
            driver = json.loads(driver_path.read_text())
            if driver.get("status") != "complete":
                raise ValueError(f"row {row['schedule_index']} driver incomplete")
            validate_result(result, row)
            cell = (
                matrix.setdefault(int(row["instance_index"]), {})
                .setdefault(float(row["blur_sigma"]), {})
                .setdefault(int(row["checkpoint_seed"]), {})
            )
            if row["arm"] in cell:
                raise ValueError(f"row {row['schedule_index']} duplicate arm")
            cell[row["arm"]] = result
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            errors.append(str(error))
    expected_cells = 25 * len(schedule["blur_sigmas"]) * len(SEEDS) * len(ARMS)
    valid_cells = sum(
        len(arms)
        for instance in matrix.values()
        for sigma in instance.values()
        for arms in sigma.values()
    )
    complete = all(
        set(matrix.get(instance, {}).get(float(sigma), {}).get(seed, {}))
        == set(ARMS)
        for instance in range(25)
        for sigma in schedule["blur_sigmas"]
        for seed in SEEDS
    )
    return matrix, {
        "expected_cells": expected_cells,
        "valid_cells": valid_cells,
        "errors": errors,
        "recorded_sigma_matches_schedule_all_rows": not any(
            "sigma mismatch" in error for error in errors
        ),
        "reconciled": not errors and valid_cells == expected_cells and complete,
    }


def rows_for(
    matrix: dict[int, Any], *, sigma: float, arm: str, seeds: tuple[int, ...]
) -> list[dict[str, Any]]:
    return [
        matrix[instance][sigma][seed][arm]
        for instance in range(25)
        for seed in seeds
    ]


def contrast_clusters(
    matrix: dict[int, Any],
    *,
    sigma: float,
    arm_a: str,
    arm_b: str,
    seeds: tuple[int, ...],
    metric: Callable[[dict[str, Any]], float],
) -> list[list[float]]:
    return [
        [
            metric(matrix[instance][sigma][seed][arm_a])
            - metric(matrix[instance][sigma][seed][arm_b])
            for seed in seeds
        ]
        for instance in range(25)
    ]


def linear_slope(sigmas: np.ndarray, values: np.ndarray) -> float:
    centered = sigmas - float(sigmas.mean())
    return float(np.dot(centered, values) / np.dot(centered, centered))


def slope_clusters(
    matrix: dict[int, Any],
    *,
    sigmas: list[float],
    arm: str,
    seeds: tuple[int, ...],
    metric: Callable[[dict[str, Any]], float],
) -> list[list[float]]:
    x = np.asarray(sigmas, dtype=np.float64)
    return [
        [
            linear_slope(
                x,
                np.asarray(
                    [metric(matrix[instance][sigma][seed][arm]) for sigma in sigmas],
                    dtype=np.float64,
                ),
            )
            for seed in seeds
        ]
        for instance in range(25)
    ]


def interaction_slope_clusters(
    matrix: dict[int, Any],
    *,
    sigmas: list[float],
    arm_a: str,
    arm_b: str,
    seeds: tuple[int, ...],
    metric: Callable[[dict[str, Any]], float],
) -> list[list[float]]:
    left = slope_clusters(matrix, sigmas=sigmas, arm=arm_a, seeds=seeds, metric=metric)
    right = slope_clusters(matrix, sigmas=sigmas, arm=arm_b, seeds=seeds, metric=metric)
    return [
        [a - b for a, b in zip(left_values, right_values)]
        for left_values, right_values in zip(left, right)
    ]


def choose_decision(
    schedule: dict[str, Any], analysis: dict[str, Any]
) -> tuple[str, str]:
    if not analysis["reconciliation"]["reconciled"]:
        return "BLUR_EXPERIMENT_INCOMPLETE", "the 900-row schedule did not reconcile"
    positive_sigmas = [float(sigma) for sigma in schedule["blur_sigmas"] if sigma > 0]
    floor = float(schedule["collapse_floor_collision_free_success"])
    absolute = analysis["absolute_performance"]
    all_collapsed = all(
        absolute[str(sigma)][arm]["pooled"]["collision_free_task_success"]["rate"]
        < floor
        for sigma in positive_sigmas
        for arm in ARMS
    )
    if all_collapsed:
        return (
            "BLUR_UNINFORMATIVE_COLLAPSE",
            "every arm was below the predeclared 10% floor at every positive sigma",
        )
    headline = analysis["contrasts_per_sigma"]
    gaps = [
        headline[str(sigma)]["PACT_minus_ACT"]["pooled"]
        ["collision_free_task_success"]
        for sigma in positive_sigmas
    ]
    positive_everywhere = all(item["difference"] > 0 for item in gaps)
    strictly_growing = all(
        left["difference"] < right["difference"]
        for left, right in zip(gaps, gaps[1:])
    )
    eligible = [
        sigma
        for sigma in positive_sigmas
        if absolute[str(sigma)]["ACT"]["pooled"]["collision_free_task_success"]["rate"]
        >= floor
        and absolute[str(sigma)]["PACT"]["pooled"]["collision_free_task_success"]["rate"]
        >= floor
    ]
    largest_eligible = max(eligible) if eligible else None
    eligible_ci_positive = bool(
        largest_eligible is not None
        and headline[str(largest_eligible)]["PACT_minus_ACT"]["pooled"]
        ["collision_free_task_success"]["instance_cluster_bootstrap_ci_95"][0]
        > 0
    )
    if positive_everywhere and strictly_growing and eligible_ci_positive:
        return (
            "BLUR_ROBUSTNESS_ESTABLISHED",
            "PACT beat ACT at every positive sigma, the gap strictly grew, and the largest non-floor contrast excluded zero",
        )
    any_positive_ci = any(
        item["instance_cluster_bootstrap_ci_95"][0] > 0 for item in gaps
    )
    if any_positive_ci:
        return (
            "BLUR_PARTIAL",
            "at least one positive-sigma PACT-minus-ACT contrast excluded zero without satisfying the monotone established rule",
        )
    return (
        "NO_BLUR_ROBUSTNESS",
        "no positive-sigma PACT-minus-ACT collision-free-success contrast had a CI lower bound above zero",
    )


def analyze(
    schedule: dict[str, Any], output_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    matrix, reconciliation = load_matrix(schedule, output_root)
    base: dict[str, Any] = {
        "schema_version": "pact_blur_sweep_analysis_v1",
        "schedule_sha256": schedule["schedule_sha256"],
        "reconciliation": reconciliation,
        "analysis_contract": schedule["analysis_contract"],
        "decision_rule": schedule["decision_rule"],
        "collapse_floor_collision_free_success": schedule[
            "collapse_floor_collision_free_success"
        ],
    }
    if not reconciliation["reconciled"]:
        token, reason = choose_decision(schedule, base)
        base.update({"decision": token, "decision_reason": reason})
        final = {
            "schema_version": "pact_blur_sweep_decision_v1",
            "schedule_sha256": schedule["schedule_sha256"],
            "analysis_sha256": None,
            "decision": token,
            "reason": reason,
        }
        return base, final

    replicates = int(schedule["bootstrap_replicates"])
    bootstrap_seed = int(schedule["bootstrap_seed"])
    sigmas = [float(value) for value in schedule["blur_sigmas"]]
    absolute: dict[str, Any] = {}
    contrasts: dict[str, Any] = {}
    for sigma_index, sigma in enumerate(sigmas):
        sigma_key = str(sigma)
        absolute[sigma_key] = {}
        for arm in ARMS:
            absolute[sigma_key][arm] = {
                "seeds": {
                    str(seed): summarize(rows_for(matrix, sigma=sigma, arm=arm, seeds=(seed,)))
                    for seed in SEEDS
                },
                "pooled": summarize(rows_for(matrix, sigma=sigma, arm=arm, seeds=SEEDS)),
            }
        contrasts[sigma_key] = {}
        for contrast_index, (arm_a, arm_b, label) in enumerate(CONTRASTS):
            contrasts[sigma_key][label] = {"seeds": {}, "pooled": {}}
            for seed_index, seed in enumerate(SEEDS):
                for metric_index, (metric_name, (metric, _binary, _direction)) in enumerate(METRICS.items()):
                    contrasts[sigma_key][label]["seeds"].setdefault(str(seed), {})[
                        metric_name
                    ] = bootstrap_cluster_mean(
                        contrast_clusters(
                            matrix,
                            sigma=sigma,
                            arm_a=arm_a,
                            arm_b=arm_b,
                            seeds=(seed,),
                            metric=metric,
                        ),
                        replicates=replicates,
                        seed=bootstrap_seed
                        + sigma_index * 100000
                        + contrast_index * 10000
                        + seed_index * 1000
                        + metric_index,
                    )
            for metric_index, (metric_name, (metric, _binary, _direction)) in enumerate(METRICS.items()):
                contrasts[sigma_key][label]["pooled"][metric_name] = bootstrap_cluster_mean(
                    contrast_clusters(
                        matrix,
                        sigma=sigma,
                        arm_a=arm_a,
                        arm_b=arm_b,
                        seeds=SEEDS,
                        metric=metric,
                    ),
                    replicates=replicates,
                    seed=bootstrap_seed
                    + 500000
                    + sigma_index * 10000
                    + contrast_index * 1000
                    + metric_index,
                )

    slopes: dict[str, Any] = {}
    slope_metrics = (
        "collision_free_task_success",
        "hazard_bar_contact_frames",
        "hazard_bar_any_contact",
        "task_success",
    )
    for metric_index, metric_name in enumerate(slope_metrics):
        metric = METRICS[metric_name][0]
        slopes[metric_name] = {"arms": {}, "interactions": {}}
        for arm_index, arm in enumerate(ARMS):
            slopes[metric_name]["arms"][arm] = {
                "seeds": {
                    str(seed): bootstrap_cluster_mean(
                        slope_clusters(
                            matrix,
                            sigmas=sigmas,
                            arm=arm,
                            seeds=(seed,),
                            metric=metric,
                        ),
                        replicates=replicates,
                        seed=bootstrap_seed + 1000000 + metric_index * 10000 + arm_index * 1000 + seed,
                    )
                    for seed in SEEDS
                },
                "pooled": bootstrap_cluster_mean(
                    slope_clusters(
                        matrix,
                        sigmas=sigmas,
                        arm=arm,
                        seeds=SEEDS,
                        metric=metric,
                    ),
                    replicates=replicates,
                    seed=bootstrap_seed + 1100000 + metric_index * 1000 + arm_index,
                ),
            }
        for contrast_index, (arm_a, arm_b, label) in enumerate(CONTRASTS):
            slopes[metric_name]["interactions"][label] = {
                "seeds": {
                    str(seed): bootstrap_cluster_mean(
                        interaction_slope_clusters(
                            matrix,
                            sigmas=sigmas,
                            arm_a=arm_a,
                            arm_b=arm_b,
                            seeds=(seed,),
                            metric=metric,
                        ),
                        replicates=replicates,
                        seed=bootstrap_seed + 1200000 + metric_index * 10000 + contrast_index * 1000 + seed,
                    )
                    for seed in SEEDS
                },
                "pooled": bootstrap_cluster_mean(
                    interaction_slope_clusters(
                        matrix,
                        sigmas=sigmas,
                        arm_a=arm_a,
                        arm_b=arm_b,
                        seeds=SEEDS,
                        metric=metric,
                    ),
                    replicates=replicates,
                    seed=bootstrap_seed + 1300000 + metric_index * 1000 + contrast_index,
                ),
            }

    analysis: dict[str, Any] = {
        **base,
        "absolute_performance": absolute,
        "contrasts_per_sigma": contrasts,
        "within_instance_linear_slopes_per_sigma_unit": slopes,
        "ood_interpretation": (
            "inference-time RGB blur is out of distribution for every arm; this "
            "analysis supports robustness to vision degradation, not a claim that "
            "proximity substitutes for vision after blur-aware training"
        ),
    }
    token, reason = choose_decision(schedule, analysis)
    analysis.update({"decision": token, "decision_reason": reason})
    final = {
        "schema_version": "pact_blur_sweep_decision_v1",
        "schedule_sha256": schedule["schedule_sha256"],
        "analysis_sha256": None,
        "decision": token,
        "reason": reason,
    }
    return analysis, final


def fmt(value: float | None, *, percent: bool = False) -> str:
    if value is None:
        return "NA"
    return f"{100 * value:.1f} pp" if percent else f"{value:,.3f}"


def render_report(
    schedule: dict[str, Any], analysis: dict[str, Any], final: dict[str, Any]
) -> str:
    lines = [
        "# PACT inference-time RGB blur sweep",
        "",
        "## Decision",
        "",
        f"**{final['decision']}** — {final['reason']}.",
        "",
        "This experiment measures robustness to inference-time vision degradation. Blur is out of distribution for every arm; it does not establish that proximity substitutes for vision under blur-aware training.",
        "",
        "## Frozen design",
        "",
        f"- 25 instances shared across all {len(schedule['blur_sigmas'])} sigmas, three arms, and three checkpoint seeds.",
        f"- Sigmas: {schedule['blur_sigmas']}.",
        f"- {schedule['rollouts']} fresh-subprocess rollouts; {schedule['bootstrap_replicates']:,} deterministic instance-cluster bootstrap replicates.",
        f"- Collapse floor: {100 * schedule['collapse_floor_collision_free_success']:.0f}% collision-free task success.",
        "",
        "## Collision-free task success",
        "",
        "| sigma | ACT | PACT | PACT_PERMUTED | PACT − ACT (95% CI) | PACT − PERMUTED (95% CI) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    if analysis["reconciliation"]["reconciled"]:
        for sigma in schedule["blur_sigmas"]:
            key = str(float(sigma))
            absolute = analysis["absolute_performance"][key]
            headline = analysis["contrasts_per_sigma"][key]
            pact_act = headline["PACT_minus_ACT"]["pooled"]["collision_free_task_success"]
            pact_perm = headline["PACT_minus_PACT_PERMUTED"]["pooled"]["collision_free_task_success"]
            lines.append(
                "| "
                + f"{sigma:g} | "
                + " | ".join(
                    f"{absolute[arm]['pooled']['collision_free_task_success']['count']}/{absolute[arm]['pooled']['collision_free_task_success']['n']} ({100 * absolute[arm]['pooled']['collision_free_task_success']['rate']:.1f}%)"
                    for arm in ARMS
                )
                + f" | {fmt(pact_act['difference'], percent=True)} [{fmt(pact_act['instance_cluster_bootstrap_ci_95'][0], percent=True)}, {fmt(pact_act['instance_cluster_bootstrap_ci_95'][1], percent=True)}]"
                + f" | {fmt(pact_perm['difference'], percent=True)} [{fmt(pact_perm['instance_cluster_bootstrap_ci_95'][0], percent=True)}, {fmt(pact_perm['instance_cluster_bootstrap_ci_95'][1], percent=True)}] |"
            )
        lines.extend(
            [
                "",
                "## Contact and slope endpoints",
                "",
                "Hazard contact frames, any-contact rates, entries, penetration, task success, seed-unpooled results, all three contrasts, within-instance arm slopes, and arm-by-sigma interactions are recorded in `diagnostics_output/pact_blur_sweep/analysis.json`.",
            ]
        )
    else:
        lines.extend(["", "The fixed schedule did not reconcile; no endpoint is interpreted."])
    lines.extend(
        [
            "",
            "## Scientific boundary",
            "",
            "This study qualifies but does not replace the confirmed contact-endpoint or geometry-generalization decisions. No checkpoint, encoder, demonstration, scene, threshold, or contact taxonomy was changed.",
            "",
            final["decision"],
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--analysis-output", required=True, type=Path)
    parser.add_argument("--decision-output", required=True, type=Path)
    parser.add_argument("--report-output", required=True, type=Path)
    args = parser.parse_args()
    schedule = json.loads(args.schedule.read_text())
    payload = dict(schedule)
    observed = payload.pop("schedule_sha256", None)
    if observed != canonical_hash(payload):
        raise SystemExit("blur schedule self-hash mismatch")
    if (
        schedule.get("schema_version") != "pact_blur_sweep_schedule_v1"
        or schedule.get("rollouts") != 900
        or schedule.get("bootstrap_replicates", 0) < 20000
    ):
        raise SystemExit("blur analysis contract changed")
    analysis, final = analyze(schedule, args.output_root)
    analysis["analysis_script_sha256"] = file_hash(Path(__file__))
    analysis["analysis_sha256"] = canonical_hash(analysis)
    final["analysis_sha256"] = analysis["analysis_sha256"]
    final["final_decision_sha256"] = canonical_hash(final)
    if final["decision"] not in TOKENS:
        raise SystemExit("unknown blur decision token")
    write_json_atomic(args.analysis_output, analysis)
    write_json_atomic(args.decision_output, final)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    report = render_report(schedule, analysis, final)
    if report.rstrip().splitlines()[-1] != final["decision"]:
        raise SystemExit("blur report token is not the last nonblank line")
    args.report_output.write_text(report)
    print(final["decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
