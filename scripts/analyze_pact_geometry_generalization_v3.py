#!/usr/bin/env python3
"""Frozen analysis for held-out PACT geometry generalization attempt 3."""

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


ARMS = ("PACT", "PACT_PERMUTED")
SEEDS = (3101, 3102, 3103)
INSTANCES_PER_CONDITION = 40
CONTRASTS = (
    ("PACT", "PACT_PERMUTED", "modality"),
)
TOKENS = {
    "GEOMETRY_GENERALIZES",
    "GEOMETRY_PARTIAL",
    "GEOMETRY_DOES_NOT_GENERALIZE",
    "GEOMETRY_TEST_INCONCLUSIVE",
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
    half = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return [center - half, center + half]


def audit_value(result: dict[str, Any], section: str, contact_class: str) -> float:
    return float(result["contact_audit"][section][contact_class])


def collision_free_task_success(result: dict[str, Any]) -> float:
    totals = result["contact_audit"]["contact_class_totals"]
    return float(
        bool(result["task_success"])
        and int(totals["hazard_bar"]) == 0
        and int(totals["other_environment"]) == 0
    )


METRICS: dict[str, tuple[Callable[[dict[str, Any]], float], bool, str]] = {
    "hazard_bar_contact_frames": (
        lambda row: audit_value(row, "frames_with_contact", "hazard_bar"),
        False,
        "lower_is_better",
    ),
    "hazard_bar_any_contact": (
        lambda row: float(audit_value(row, "contact_class_totals", "hazard_bar") > 0),
        True,
        "lower_is_better",
    ),
    "collision_free_task_success": (collision_free_task_success, True, "higher_is_better"),
    "hazard_bar_contact_entries": (
        lambda row: audit_value(row, "contact_class_totals", "hazard_bar"),
        False,
        "lower_is_better",
    ),
    "hazard_bar_maximum_penetration_depth_m": (
        lambda row: audit_value(row, "maximum_penetration_depth_m", "hazard_bar"),
        False,
        "lower_is_better",
    ),
    "other_environment_contact_frames": (
        lambda row: audit_value(row, "frames_with_contact", "other_environment"),
        False,
        "lower_is_better",
    ),
    "manipulation_success": (lambda row: float(bool(row["task_success"])), True, "higher_is_better"),
}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output = {}
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
            item.update({"count": count, "rate": count / len(values), "wilson_95": wilson(count, len(values))})
        output[name] = item
    output["failure_taxonomy"] = dict(sorted(Counter(row["failure_taxonomy"] for row in rows).items()))
    return output


def bootstrap_difference(
    cluster_values: list[list[float]], *, replicates: int, seed: int
) -> dict[str, Any]:
    if not cluster_values:
        return {
            "difference": None,
            "instance_cluster_bootstrap_ci_95": [None, None],
            "n_unique_instances": 0,
            "n_pairs": 0,
        }
    sums = np.asarray([sum(values) for values in cluster_values], dtype=np.float64)
    counts = np.asarray([len(values) for values in cluster_values], dtype=np.float64)
    difference = float(sums.sum() / counts.sum())
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(sums), size=(replicates, len(sums)))
    boot = sums[sampled].sum(axis=1) / counts[sampled].sum(axis=1)
    return {
        "difference": difference,
        "median_pair_difference": float(np.median([value for values in cluster_values for value in values])),
        "instance_cluster_bootstrap_ci_95": [
            float(np.percentile(boot, 2.5)),
            float(np.percentile(boot, 97.5)),
        ],
        "n_unique_instances": len(cluster_values),
        "n_pairs": int(counts.sum()),
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "cluster_unit": "instance; all included arms, seeds, and conditions move together",
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
    }
    for key, value in expected.items():
        if result.get(key) != value:
            raise ValueError(f"row {row['schedule_index']} identity mismatch: {key}")
    audit = result.get("contact_audit", {})
    expected_classes = {"grasp_target", "hazard_bar", "other_environment"}
    for section in ("contact_class_totals", "frames_with_contact", "maximum_penetration_depth_m"):
        if set(audit.get(section, {})) != expected_classes:
            raise ValueError(f"row {row['schedule_index']} contact taxonomy changed")
    if bool(collision_free_task_success(result)) != bool(result["collision_free_task_success"]):
        raise ValueError(f"row {row['schedule_index']} collision-free endpoint mismatch")


def load_matrix(schedule: dict[str, Any], output_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    matrix: dict[str, dict[int, dict[int, dict[str, dict[str, Any]]]]] = {}
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
                matrix.setdefault(row["condition_id"], {})
                .setdefault(int(row["instance_index"]), {})
                .setdefault(int(row["checkpoint_seed"]), {})
            )
            if row["arm"] in cell:
                raise ValueError(f"row {row['schedule_index']} duplicate arm")
            cell[row["arm"]] = result
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            errors.append(str(error))
    expected_cells = (
        len(schedule["surviving_condition_ids"])
        * INSTANCES_PER_CONDITION
        * len(SEEDS)
        * len(ARMS)
    )
    valid_cells = sum(
        len(arms)
        for condition in matrix.values()
        for instance in condition.values()
        for arms in instance.values()
    )
    complete = all(
        set(matrix.get(condition, {}).get(instance, {}).get(seed, {})) == set(ARMS)
        for condition in schedule["surviving_condition_ids"]
        for instance in range(INSTANCES_PER_CONDITION)
        for seed in SEEDS
    )
    return matrix, {
        "expected_cells": expected_cells,
        "valid_cells": valid_cells,
        "errors": errors,
        "reconciled": not errors and valid_cells == expected_cells and complete,
    }


def contrast_clusters(
    matrix: dict[str, Any],
    *,
    condition_ids: list[str],
    seeds: tuple[int, ...],
    arm_a: str,
    arm_b: str,
    metric: Callable[[dict[str, Any]], float],
) -> list[list[float]]:
    clusters = []
    for instance in range(INSTANCES_PER_CONDITION):
        values = []
        for condition in condition_ids:
            for seed in seeds:
                cell = matrix[condition][instance][seed]
                values.append(metric(cell[arm_a]) - metric(cell[arm_b]))
        clusters.append(values)
    return clusters


def choose_decision(
    *,
    reconciliation: dict[str, Any],
    c0_reproduces: bool,
    shifted_support: dict[str, bool],
    pooled_any: dict[str, Any] | None,
    pooled_frames: dict[str, Any] | None,
) -> tuple[str, str]:
    if not reconciliation["reconciled"]:
        return "GEOMETRY_TEST_INCONCLUSIVE", "schedule_did_not_reconcile"
    if not c0_reproduces:
        return "GEOMETRY_TEST_INCONCLUSIVE", "C0_did_not_reproduce_the_reference_modality_gap"
    if len(shifted_support) < 2:
        return "GEOMETRY_TEST_INCONCLUSIVE", "fewer_than_two_shifted_conditions_survived"
    assert pooled_any is not None and pooled_frames is not None
    any_ci = pooled_any["instance_cluster_bootstrap_ci_95"]
    frame_ci = pooled_frames["instance_cluster_bootstrap_ci_95"]
    pooled_both_below = any_ci[1] < 0.0 and frame_ci[1] < 0.0
    if all(shifted_support.values()) and pooled_both_below:
        return "GEOMETRY_GENERALIZES", "all_shifted_conditions_favor_PACT_and_both_pooled_shifted_contact_CIs_exclude_zero"
    if any(shifted_support.values()) and not all(shifted_support.values()):
        return "GEOMETRY_PARTIAL", "both_contact_gaps_favor_PACT_in_some_but_not_all_shifted_conditions"
    return "GEOMETRY_DOES_NOT_GENERALIZE", "shifted_modality_gap_reversed_or_pooled_shifted_contact_CI_included_zero"


def analyze(schedule: dict[str, Any], output_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    matrix, reconciliation = load_matrix(schedule, output_root)
    base = {
        "schedule_sha256": schedule["schedule_sha256"],
        "analysis_script_sha256": file_hash(Path(__file__).resolve()),
        "reconciliation": reconciliation,
        "exploratory": False,
        "retraining_performed": False,
    }
    if not reconciliation["reconciled"]:
        token, reason = choose_decision(
            reconciliation=reconciliation,
            c0_reproduces=False,
            shifted_support={},
            pooled_any=None,
            pooled_frames=None,
        )
        analysis = {"schema_version": "pact_geometry_generalization_analysis_v3", **base, "results_available": False}
        final = {
            "schema_version": "pact_geometry_generalization_decision_v3",
            "schedule_sha256": schedule["schedule_sha256"],
            "decision": token,
            "reason": reason,
        }
        return analysis, final

    replicates = int(schedule["bootstrap_replicates"])
    bootstrap_seed = int(schedule["bootstrap_seed"])
    condition_results = {}
    for condition_index, condition in enumerate(schedule["surviving_condition_ids"]):
        condition_results[condition] = {
            "absolute": {},
            "seeds_unpooled_first": {},
            "pooled": {},
        }
        for arm in ARMS:
            rows = [
                matrix[condition][instance][seed][arm]
                for instance in range(INSTANCES_PER_CONDITION)
                for seed in SEEDS
            ]
            condition_results[condition]["absolute"][arm] = summarize(rows)
        for seed_index, seed in enumerate(SEEDS):
            seed_record = {"absolute": {}, "contrasts": {}}
            for arm in ARMS:
                seed_record["absolute"][arm] = summarize(
                    [
                        matrix[condition][instance][seed][arm]
                        for instance in range(INSTANCES_PER_CONDITION)
                    ]
                )
            for contrast_index, (arm_a, arm_b, interpretation) in enumerate(CONTRASTS):
                name = f"{arm_a}_minus_{arm_b}"
                seed_record["contrasts"][name] = {}
                for metric_index, (metric_name, (metric, _binary, direction)) in enumerate(METRICS.items()):
                    item = bootstrap_difference(
                        contrast_clusters(
                            matrix,
                            condition_ids=[condition],
                            seeds=(seed,),
                            arm_a=arm_a,
                            arm_b=arm_b,
                            metric=metric,
                        ),
                        replicates=replicates,
                        seed=bootstrap_seed + condition_index * 10000 + seed_index * 1000 + contrast_index * 100 + metric_index,
                    )
                    item.update({"direction": direction, "interpretation": interpretation})
                    seed_record["contrasts"][name][metric_name] = item
            condition_results[condition]["seeds_unpooled_first"][str(seed)] = seed_record
        for contrast_index, (arm_a, arm_b, interpretation) in enumerate(CONTRASTS):
            name = f"{arm_a}_minus_{arm_b}"
            condition_results[condition]["pooled"][name] = {}
            for metric_index, (metric_name, (metric, _binary, direction)) in enumerate(METRICS.items()):
                item = bootstrap_difference(
                    contrast_clusters(
                        matrix,
                        condition_ids=[condition],
                        seeds=SEEDS,
                        arm_a=arm_a,
                        arm_b=arm_b,
                        metric=metric,
                    ),
                    replicates=replicates,
                    seed=bootstrap_seed + 50000 + condition_index * 1000 + contrast_index * 100 + metric_index,
                )
                item.update({"direction": direction, "interpretation": interpretation})
                condition_results[condition]["pooled"][name][metric_name] = item

    modality = "PACT_minus_PACT_PERMUTED"
    c0_any = condition_results["C0"]["pooled"][modality]["hazard_bar_any_contact"]
    c0_frames = condition_results["C0"]["pooled"][modality]["hazard_bar_contact_frames"]
    reference = schedule["original_reference"]
    any_reference = float(reference["PACT_minus_PACT_PERMUTED_any_hazard_contact_rate"])
    frame_reference = float(reference["PACT_minus_PACT_PERMUTED_hazard_contact_frames"])
    c0_checks = {
        "any_contact_point_negative": c0_any["difference"] < 0,
        "contact_frames_point_negative": c0_frames["difference"] < 0,
        "original_any_contact_reference_inside_C0_CI": c0_any["instance_cluster_bootstrap_ci_95"][0] <= any_reference <= c0_any["instance_cluster_bootstrap_ci_95"][1],
        "original_contact_frames_reference_inside_C0_CI": c0_frames["instance_cluster_bootstrap_ci_95"][0] <= frame_reference <= c0_frames["instance_cluster_bootstrap_ci_95"][1],
    }
    c0_reproduces = all(c0_checks.values())
    shifted = schedule["shifted_condition_ids"]
    shifted_support = {
        condition: (
            condition_results[condition]["pooled"][modality]["hazard_bar_any_contact"]["difference"] < 0
            and condition_results[condition]["pooled"][modality]["hazard_bar_contact_frames"]["difference"] < 0
        )
        for condition in shifted
    }
    pooled_shifted = {}
    for metric_index, metric_name in enumerate(("hazard_bar_any_contact", "hazard_bar_contact_frames")):
        metric = METRICS[metric_name][0]
        pooled_shifted[metric_name] = bootstrap_difference(
            contrast_clusters(
                matrix,
                condition_ids=shifted,
                seeds=SEEDS,
                arm_a="PACT",
                arm_b="PACT_PERMUTED",
                metric=metric,
            ),
            replicates=replicates,
            seed=bootstrap_seed + 90000 + metric_index,
        )
    token, reason = choose_decision(
        reconciliation=reconciliation,
        c0_reproduces=c0_reproduces,
        shifted_support=shifted_support,
        pooled_any=pooled_shifted["hazard_bar_any_contact"],
        pooled_frames=pooled_shifted["hazard_bar_contact_frames"],
    )
    analysis: dict[str, Any] = {
        "schema_version": "pact_geometry_generalization_analysis_v3",
        **base,
        "results_available": True,
        "contact_taxonomy": ["grasp_target", "hazard_bar", "other_environment"],
        "seeds_unpooled_first": True,
        "condition_results": condition_results,
        "C0_reproduction": {
            "original_reference": reference,
            "checks": c0_checks,
            "reproduces": c0_reproduces,
            "any_contact": c0_any,
            "hazard_contact_frames": c0_frames,
        },
        "shifted_condition_both_contact_gaps_negative": shifted_support,
        "pooled_shifted_modality": pooled_shifted,
        "decision": token,
        "decision_reason": reason,
    }
    analysis["analysis_sha256"] = canonical_hash(analysis)
    final: dict[str, Any] = {
        "schema_version": "pact_geometry_generalization_decision_v3",
        "schedule_sha256": schedule["schedule_sha256"],
        "analysis_sha256": analysis["analysis_sha256"],
        "decision": token,
        "reason": reason,
        "cannot_award_pact_confirmatory_token": True,
    }
    final["final_decision_sha256"] = canonical_hash(final)
    return analysis, final


def fmt(value: float, *, percent: bool = False) -> str:
    return f"{value * 100:+.1f} pp" if percent else f"{value:+,.1f}"


def render_report(schedule: dict[str, Any], analysis: dict[str, Any], final: dict[str, Any]) -> str:
    lines = [
        "# PACT held-out geometry generalization",
        "",
        "This zero-shot evaluation uses the frozen PACT weights under live and preregistered permuted proximity; no policy, encoder, threshold, scene outcome, or endpoint was retuned.",
        "",
        "## Decision",
        "",
        f"**{final['decision']}** — {final['reason'].replace('_', ' ')}.",
        "",
        "## Expert solvability gate",
        "",
        "Only conditions passing at least 10/12 privileged-expert clean successes entered the policy schedule. A clean success is task success with zero hazard-bar and zero other-environment contacts.",
        "",
        "## In-distribution control",
        "",
    ]
    if analysis.get("results_available"):
        c0 = analysis["C0_reproduction"]
        lines += [
            f"C0 reproduction: **{'pass' if c0['reproduces'] else 'fail'}**.",
            "",
            "| Modality contrast | C0 estimate | 95% instance-cluster CI | Original reference |",
            "|---|---:|---:|---:|",
            f"| Any hazard contact | {fmt(c0['any_contact']['difference'], percent=True)} | [{fmt(c0['any_contact']['instance_cluster_bootstrap_ci_95'][0], percent=True)}, {fmt(c0['any_contact']['instance_cluster_bootstrap_ci_95'][1], percent=True)}] | −9.3 pp |",
            f"| Hazard contact frames | {fmt(c0['hazard_contact_frames']['difference'])} | [{fmt(c0['hazard_contact_frames']['instance_cluster_bootstrap_ci_95'][0])}, {fmt(c0['hazard_contact_frames']['instance_cluster_bootstrap_ci_95'][1])}] | −1,980 |",
            "",
            "## Absolute arm performance",
            "",
            "| Condition | Arm | n | Any hazard contact | Mean hazard frames | Collision-free task success | Task success |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for condition in schedule["surviving_condition_ids"]:
            for arm in ARMS:
                absolute = analysis["condition_results"][condition]["absolute"][arm]
                any_contact = absolute["hazard_bar_any_contact"]
                collision_free = absolute["collision_free_task_success"]
                task_success = absolute["manipulation_success"]
                lines.append(
                    f"| {condition} | {arm} | {any_contact['n']} | "
                    f"{any_contact['count']}/{any_contact['n']} ({any_contact['rate'] * 100:.1f}%) | "
                    f"{absolute['hazard_bar_contact_frames']['mean']:.1f} | "
                    f"{collision_free['count']}/{collision_free['n']} ({collision_free['rate'] * 100:.1f}%) | "
                    f"{task_success['count']}/{task_success['n']} ({task_success['rate'] * 100:.1f}%) |"
                )
        lines += [
            "",
            "## Shifted conditions",
            "",
            "| Condition | PACT − PERM any contact | PACT − PERM hazard frames | Both favor PACT |",
            "|---|---:|---:|:---:|",
        ]
        for condition in schedule["shifted_condition_ids"]:
            record = analysis["condition_results"][condition]["pooled"]["PACT_minus_PACT_PERMUTED"]
            lines.append(
                f"| {condition} | {fmt(record['hazard_bar_any_contact']['difference'], percent=True)} | {fmt(record['hazard_bar_contact_frames']['difference'])} | {'yes' if analysis['shifted_condition_both_contact_gaps_negative'][condition] else 'no'} |"
            )
        pooled = analysis["pooled_shifted_modality"]
        lines += [
            "",
            "Pooled shifted modality contrast:",
            "",
            f"- Any hazard contact: {fmt(pooled['hazard_bar_any_contact']['difference'], percent=True)}, 95% CI [{fmt(pooled['hazard_bar_any_contact']['instance_cluster_bootstrap_ci_95'][0], percent=True)}, {fmt(pooled['hazard_bar_any_contact']['instance_cluster_bootstrap_ci_95'][1], percent=True)}].",
            f"- Hazard contact frames: {fmt(pooled['hazard_bar_contact_frames']['difference'])}, 95% CI [{fmt(pooled['hazard_bar_contact_frames']['instance_cluster_bootstrap_ci_95'][0])}, {fmt(pooled['hazard_bar_contact_frames']['instance_cluster_bootstrap_ci_95'][1])}].",
            "",
            "Full absolute arm performance, seed-unpooled contrasts, collision-free task success, manipulation success, entries, penetration, failure taxonomy, counts, Wilson intervals, and all 20,000-replicate bootstrap intervals are in `diagnostics_output/pact_geometry_generalization_v3/analysis.json`.",
        ]
    else:
        lines += ["The schedule did not reconcile; no shifted-condition result is interpreted."]
    lines += [
        "",
        "This study qualifies the existing contact-endpoint result and cannot award or replace a PACT confirmatory token.",
        "",
        final["decision"],
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
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
        raise SystemExit("schedule self-hash mismatch")
    if (
        schedule.get("schema_version") != "pact_geometry_generalization_v3_schedule"
        or schedule.get("instances_per_condition") != INSTANCES_PER_CONDITION
        or schedule.get("arms") != list(ARMS)
        or schedule.get("checkpoint_seeds") != list(SEEDS)
        or schedule.get("rollouts") != 720
        or schedule.get("bootstrap_replicates", 0) < 20000
    ):
        raise SystemExit("attempt-3 analysis design changed")
    if schedule.get("analysis_script_sha256") != file_hash(Path(__file__).resolve()):
        raise SystemExit("analysis script differs from frozen schedule")
    analysis, final = analyze(schedule, args.output_root)
    if final["decision"] not in TOKENS:
        raise SystemExit("unknown decision token")
    write_json_atomic(args.analysis_output, analysis)
    write_json_atomic(args.decision_output, final)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(render_report(schedule, analysis, final))
    print(final["decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
