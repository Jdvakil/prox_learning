#!/usr/bin/env python3
"""Frozen paired analysis for the sighted-versus-blind RGB experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable

import analyze_pact_blur_sweep as common


ARMS = ("ACT", "PACT", "PACT_PERMUTED")
SEEDS = (3101, 3102, 3103)
CONDITIONS = ("sighted", "blind")
BLIND_CONTRASTS = (
    ("PACT", "ACT", "PACT_minus_ACT"),
    ("PACT", "PACT_PERMUTED", "PACT_minus_PACT_PERMUTED"),
    ("PACT_PERMUTED", "ACT", "PACT_PERMUTED_minus_ACT"),
)


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


def validate_result(result: dict[str, Any], row: dict[str, Any]) -> None:
    expected = {
        "status": "complete",
        "rollout_id": row["rollout_id"],
        "schedule_row_sha256": row["schedule_row_sha256"],
        "episode_id": row["instance_episode_id"],
        "arm": row["arm"],
        "checkpoint_seed": row["checkpoint_seed"],
        "checkpoint_sha256": row["checkpoint_sha256"],
        "blind_rgb": row["blind_rgb"],
        "blur_sigma": 0.0,
    }
    for key, value in expected.items():
        if result.get(key) != value:
            raise ValueError(f"row {row['schedule_index']} identity mismatch: {key}")
    policy = result.get("policy_info", {})
    if policy.get("blind_rgb") is not row["blind_rgb"]:
        raise ValueError(f"row {row['schedule_index']} policy blind flag mismatch")
    if policy.get("blur_sigma") != 0.0:
        raise ValueError(f"row {row['schedule_index']} policy blur flag mismatch")
    diagnostic = policy.get("blur_diagnostic", {})
    if diagnostic.get("blind_rgb") is not row["blind_rgb"]:
        raise ValueError(f"row {row['schedule_index']} diagnostic blind flag mismatch")
    if row["blind_rgb"]:
        if diagnostic.get("first_policy_visual_input_is_exact_imagenet_mean") is not True:
            raise ValueError(f"row {row['schedule_index']} is not exact ImageNet mean")
        if diagnostic.get("first_visual_input_changed") is not True:
            raise ValueError(f"row {row['schedule_index']} blind visual input did not change")
    else:
        if diagnostic.get("first_visual_input_changed") is not False:
            raise ValueError(f"row {row['schedule_index']} sighted input changed")
    audit = result.get("contact_audit", {})
    expected_classes = {"grasp_target", "hazard_bar", "other_environment"}
    for section in (
        "contact_class_totals",
        "frames_with_contact",
        "maximum_penetration_depth_m",
    ):
        if set(audit.get(section, {})) != expected_classes:
            raise ValueError(f"row {row['schedule_index']} contact taxonomy changed")
    if bool(common.collision_free_task_success(result)) != bool(
        result["collision_free_task_success"]
    ):
        raise ValueError(f"row {row['schedule_index']} collision-free endpoint mismatch")


def load_matrix(schedule: dict[str, Any], output_root: Path) -> tuple[dict, dict]:
    matrix: dict[int, dict[str, dict[int, dict[str, dict[str, Any]]]]] = {}
    errors = []
    flag_errors = []
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
                .setdefault(str(row["vision_condition"]), {})
                .setdefault(int(row["checkpoint_seed"]), {})
            )
            if row["arm"] in cell:
                raise ValueError(f"row {row['schedule_index']} duplicate arm")
            cell[row["arm"]] = result
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            message = str(error)
            errors.append(message)
            if "blind" in message or "blur" in message or "ImageNet" in message:
                flag_errors.append(message)
    expected = 25 * len(CONDITIONS) * len(SEEDS) * len(ARMS)
    valid = sum(
        len(arms)
        for instance in matrix.values()
        for condition in instance.values()
        for arms in condition.values()
    )
    complete = all(
        set(matrix.get(instance, {}).get(condition, {}).get(seed, {})) == set(ARMS)
        for instance in range(25)
        for condition in CONDITIONS
        for seed in SEEDS
    )
    return matrix, {
        "expected_cells": expected,
        "valid_cells": valid,
        "errors": errors,
        "recorded_flags_match_schedule_all_rows": not flag_errors,
        "reconciled": not errors and valid == expected and complete,
    }


def rows_for(matrix: dict, *, condition: str, arm: str, seeds: tuple[int, ...]) -> list[dict]:
    return [
        matrix[instance][condition][seed][arm]
        for instance in range(25)
        for seed in seeds
    ]


def contrast_clusters(
    matrix: dict,
    *,
    condition_a: str,
    arm_a: str,
    condition_b: str,
    arm_b: str,
    seeds: tuple[int, ...],
    metric: Callable[[dict[str, Any]], float],
) -> list[list[float]]:
    return [
        [
            metric(matrix[instance][condition_a][seed][arm_a])
            - metric(matrix[instance][condition_b][seed][arm_b])
            for seed in seeds
        ]
        for instance in range(25)
    ]


def boot(
    matrix: dict,
    *,
    condition_a: str,
    arm_a: str,
    condition_b: str,
    arm_b: str,
    seeds: tuple[int, ...],
    metric: Callable[[dict[str, Any]], float],
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    return common.bootstrap_cluster_mean(
        contrast_clusters(
            matrix,
            condition_a=condition_a,
            arm_a=arm_a,
            condition_b=condition_b,
            arm_b=arm_b,
            seeds=seeds,
            metric=metric,
        ),
        replicates=replicates,
        seed=seed,
    )


def contact_benefit(analysis: dict[str, Any]) -> bool:
    contrasts = analysis["blind_arm_contrasts"]
    return all(
        contrasts[label]["pooled"]["hazard_bar_contact_frames"]
        ["instance_cluster_bootstrap_ci_95"][1]
        < 0.0
        for label in ("PACT_minus_ACT", "PACT_minus_PACT_PERMUTED")
    )


def choose_decision(schedule: dict[str, Any], analysis: dict[str, Any]) -> tuple[str, str]:
    if not analysis["reconciliation"]["reconciled"]:
        return "BLIND_EXPERIMENT_INCOMPLETE", "the 450-row schedule did not reconcile"
    benefit = contact_benefit(analysis)
    pact_act_cfs = analysis["blind_arm_contrasts"]["PACT_minus_ACT"]["pooled"][
        "collision_free_task_success"
    ]
    pact_manipulation = analysis["absolute_performance"]["blind"]["PACT"]["pooled"][
        "task_success"
    ]["rate"]
    if (
        benefit
        and pact_act_cfs["instance_cluster_bootstrap_ci_95"][0] > 0.0
        and pact_manipulation >= schedule["collapse_floor_manipulation_success"]
    ):
        return (
            "PROXIMITY_STANDALONE_TASK_BENEFIT",
            "both blind contact contrasts exclude zero in PACT's favor and blind PACT has a collision-free task-success advantage",
        )
    if benefit:
        return (
            "PROXIMITY_STANDALONE_CONTACT_BENEFIT",
            "both decision-bearing blind hazard-frame contrasts exclude zero in PACT's favor",
        )
    absolute = analysis["absolute_performance"]["blind"]
    collapsed = all(
        absolute[arm]["pooled"]["collision_free_task_success"]["rate"]
        < schedule["collapse_floor_collision_free_success"]
        and absolute[arm]["pooled"]["task_success"]["rate"]
        < schedule["collapse_floor_manipulation_success"]
        for arm in ARMS
    )
    if collapsed:
        return (
            "BLIND_UNINFORMATIVE_COLLAPSE",
            "every blind arm is below both absolute floors and no standalone contact benefit was established",
        )
    return (
        "NO_STANDALONE_BENEFIT",
        "at least one decision-bearing blind hazard-frame confidence interval includes zero or reverses",
    )


def analyze(schedule: dict[str, Any], output_root: Path) -> tuple[dict, dict]:
    matrix, reconciliation = load_matrix(schedule, output_root)
    base = {
        "schema_version": "pact_blind_rgb_analysis_v1",
        "schedule_sha256": schedule["schedule_sha256"],
        "reconciliation": reconciliation,
        "analysis_contract": schedule["analysis_contract"],
        "decision_rule": schedule["decision_rule"],
        "predeclared_expected_outcome": schedule["predeclared_expected_outcome"],
    }
    if not reconciliation["reconciled"]:
        token, reason = choose_decision(schedule, base)
        base.update({"decision": token, "decision_reason": reason})
        return base, {
            "schema_version": "pact_blind_rgb_decision_v1",
            "schedule_sha256": schedule["schedule_sha256"],
            "analysis_sha256": None,
            "decision": token,
            "reason": reason,
        }
    replicates = int(schedule["bootstrap_replicates"])
    seed0 = int(schedule["bootstrap_seed"])
    absolute: dict[str, Any] = {}
    for condition in CONDITIONS:
        absolute[condition] = {}
        for arm in ARMS:
            absolute[condition][arm] = {
                "seeds": {
                    str(seed): common.summarize(
                        rows_for(matrix, condition=condition, arm=arm, seeds=(seed,))
                    )
                    for seed in SEEDS
                },
                "pooled": common.summarize(
                    rows_for(matrix, condition=condition, arm=arm, seeds=SEEDS)
                ),
            }
    blind_contrasts: dict[str, Any] = {}
    for contrast_index, (arm_a, arm_b, label) in enumerate(BLIND_CONTRASTS):
        blind_contrasts[label] = {"seeds": {}, "pooled": {}}
        for seed_index, checkpoint_seed in enumerate(SEEDS):
            for metric_index, (metric_name, (metric, _binary, _direction)) in enumerate(common.METRICS.items()):
                blind_contrasts[label]["seeds"].setdefault(str(checkpoint_seed), {})[
                    metric_name
                ] = boot(
                    matrix,
                    condition_a="blind",
                    arm_a=arm_a,
                    condition_b="blind",
                    arm_b=arm_b,
                    seeds=(checkpoint_seed,),
                    metric=metric,
                    replicates=replicates,
                    seed=seed0 + contrast_index * 10000 + seed_index * 1000 + metric_index,
                )
        for metric_index, (metric_name, (metric, _binary, _direction)) in enumerate(common.METRICS.items()):
            blind_contrasts[label]["pooled"][metric_name] = boot(
                matrix,
                condition_a="blind",
                arm_a=arm_a,
                condition_b="blind",
                arm_b=arm_b,
                seeds=SEEDS,
                metric=metric,
                replicates=replicates,
                seed=seed0 + 100000 + contrast_index * 1000 + metric_index,
            )
    degradation: dict[str, Any] = {}
    for arm_index, arm in enumerate(ARMS):
        degradation[arm] = {"seeds": {}, "pooled": {}}
        for seed_index, checkpoint_seed in enumerate(SEEDS):
            for metric_index, (metric_name, (metric, _binary, _direction)) in enumerate(common.METRICS.items()):
                degradation[arm]["seeds"].setdefault(str(checkpoint_seed), {})[
                    metric_name
                ] = boot(
                    matrix,
                    condition_a="blind",
                    arm_a=arm,
                    condition_b="sighted",
                    arm_b=arm,
                    seeds=(checkpoint_seed,),
                    metric=metric,
                    replicates=replicates,
                    seed=seed0 + 200000 + arm_index * 10000 + seed_index * 1000 + metric_index,
                )
        for metric_index, (metric_name, (metric, _binary, _direction)) in enumerate(common.METRICS.items()):
            degradation[arm]["pooled"][metric_name] = boot(
                matrix,
                condition_a="blind",
                arm_a=arm,
                condition_b="sighted",
                arm_b=arm,
                seeds=SEEDS,
                metric=metric,
                replicates=replicates,
                seed=seed0 + 300000 + arm_index * 1000 + metric_index,
            )
    document = {
        **base,
        "absolute_performance": absolute,
        "blind_arm_contrasts": blind_contrasts,
        "blind_minus_sighted_within_arm": degradation,
        "manipulation_success_field": "task_success",
        "interpretation_scope": (
            "inference-time constant RGB is out of distribution for every arm; this isolates retained proximity information but does not claim a blind-trained policy"
        ),
    }
    token, reason = choose_decision(schedule, document)
    document.update({"decision": token, "decision_reason": reason})
    return document, {
        "schema_version": "pact_blind_rgb_decision_v1",
        "schedule_sha256": schedule["schedule_sha256"],
        "analysis_sha256": None,
        "decision": token,
        "reason": reason,
    }


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def render_report(schedule: dict, analysis: dict, final: dict) -> str:
    lines = [
        "# PACT with the RGB camera blinded",
        "",
        "## Design",
        "",
        "The frozen ACT, PACT, and PACT_PERMUTED checkpoints were evaluated sighted and with wrist RGB replaced by the ImageNet mean. The same 25 instances and three checkpoint seeds were shared across all six cells (450 fresh subprocess rollouts, 12 fixed workers). No retraining, scene change, encoder change, or threshold change occurred.",
        "",
        "Before rollout, the expected result was declared as task success collapsing for every blind arm while PACT retained lower panel contact. The permitted safety-only interpretation was: *proximity alone keeps the arm safe but cannot do the task*.",
        "",
        "## Reconciliation",
        "",
        f"- Valid rows: {analysis['reconciliation']['valid_cells']}/{analysis['reconciliation']['expected_cells']}.",
        f"- Recorded blind/blur flags matched the schedule: {analysis['reconciliation']['recorded_flags_match_schedule_all_rows']}.",
    ]
    if analysis["reconciliation"]["reconciled"]:
        lines += ["", "## Absolute performance", ""]
        for condition in CONDITIONS:
            lines += [f"### {condition.title()}", "", "| Arm | Collision-free task success | Manipulation success | Any hazard | Mean hazard frames |", "|---|---:|---:|---:|---:|"]
            for arm in ARMS:
                summary = analysis["absolute_performance"][condition][arm]["pooled"]
                cfs = summary["collision_free_task_success"]
                task = summary["task_success"]
                any_hazard = summary["hazard_bar_any_contact"]
                frames = summary["hazard_bar_contact_frames"]
                lines.append(
                    f"| {arm} | {cfs['count']}/{cfs['n']} ({pct(cfs['rate'])}) | "
                    f"{task['count']}/{task['n']} ({pct(task['rate'])}) | "
                    f"{any_hazard['count']}/{any_hazard['n']} ({pct(any_hazard['rate'])}) | {frames['mean']:.1f} |"
                )
        lines += ["", "## Blind contrasts", ""]
        for label in ("PACT_minus_ACT", "PACT_minus_PACT_PERMUTED"):
            values = analysis["blind_arm_contrasts"][label]["pooled"]
            frames = values["hazard_bar_contact_frames"]
            cfs = values["collision_free_task_success"]
            lines.append(
                f"- {label}: hazard frames {frames['difference']:.1f}, 95% instance-cluster bootstrap CI [{frames['instance_cluster_bootstrap_ci_95'][0]:.1f}, {frames['instance_cluster_bootstrap_ci_95'][1]:.1f}]; collision-free success {pct(cfs['difference'])}, CI [{pct(cfs['instance_cluster_bootstrap_ci_95'][0])}, {pct(cfs['instance_cluster_bootstrap_ci_95'][1])}]."
            )
        lines += [
            "",
            "Seed-unpooled contrasts, any-contact rates, entries, penetration, failure taxonomy, and paired blind-minus-sighted degradation are recorded in `diagnostics_output/pact_blind_rgb/analysis.json`.",
        ]
    lines += [
        "",
        "## Decision",
        "",
        final["reason"] + ".",
        "",
        final["decision"],
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--analysis", required=True, type=Path)
    parser.add_argument("--final-decision", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    schedule = json.loads(args.schedule.read_text())
    payload = dict(schedule)
    observed = payload.pop("schedule_sha256", None)
    if observed != canonical_hash(payload):
        raise SystemExit("blind-RGB schedule self-hash mismatch")
    if (
        schedule.get("schema_version") != "pact_blind_rgb_schedule_v1"
        or schedule.get("analysis_script_sha256") != file_hash(Path(__file__))
        or schedule.get("bootstrap_replicates", 0) < 20000
    ):
        raise SystemExit("blind-RGB frozen analysis binding changed")
    analysis, final = analyze(schedule, args.output_root)
    analysis["analysis_sha256"] = canonical_hash(analysis)
    final["analysis_sha256"] = analysis["analysis_sha256"]
    final["final_decision_sha256"] = canonical_hash(final)
    write_json_atomic(args.analysis, analysis)
    write_json_atomic(args.final_decision, final)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_report(schedule, analysis, final))
    print(json.dumps({"decision": final["decision"], "analysis_sha256": analysis["analysis_sha256"]}))
    return 0 if analysis["reconciliation"]["reconciled"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
