#!/usr/bin/env python3
"""Frozen seed-wise and instance-clustered analysis for PACT replication."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest

ARMS = ("ACT", "PACT", "PACT_PERMUTED")
SEEDS = (3101, 3102)
CONTRASTS = (
    ("PACT", "PACT_PERMUTED"),
    ("PACT_PERMUTED", "ACT"),
    ("PACT", "ACT"),
)
TOKENS = {
    "SEED_REPLICATION_CONFIRMED",
    "SEED_REPLICATION_PARTIAL",
    "SEED_REPLICATION_FAILED",
    "SEED_REPLICATION_INCOMPLETE",
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


def endpoint(result: dict[str, Any]) -> bool:
    contacts = result["contact_audit"]["contact_class_totals"]
    return bool(
        result["task_success"]
        and int(contacts.get("hazard_bar", 0)) == 0
        and int(contacts.get("other_environment", 0)) == 0
    )


def wilson(successes: int, total: int) -> list[float | None]:
    if total == 0:
        return [None, None]
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [center - half, center + half]


def arm_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    primary = sum(endpoint(row) for row in rows)
    task = sum(bool(row["task_success"]) for row in rows)
    classes = ("grasp_target", "hazard_bar", "other_environment")
    return {
        "n": total,
        "collision_free_task_success": primary,
        "collision_free_task_success_rate": primary / total if total else None,
        "collision_free_task_success_wilson_95": wilson(primary, total),
        "ordinary_task_success": task,
        "ordinary_task_success_rate": task / total if total else None,
        "ordinary_task_success_wilson_95": wilson(task, total),
        "contact_pair_entry_totals": {
            key: sum(int(row["contact_audit"]["contact_class_totals"].get(key, 0)) for row in rows)
            for key in classes
        },
        "episodes_with_contact": {
            key: sum(
                int(row["contact_audit"]["contact_class_totals"].get(key, 0)) > 0 for row in rows
            )
            for key in classes
        },
        "failure_taxonomy": dict(Counter(row["failure_taxonomy"] for row in rows)),
    }


def _paired_values(
    instances: list[dict[str, dict[str, Any]]], arm_a: str, arm_b: str
) -> np.ndarray:
    return np.asarray(
        [
            float(endpoint(instance[arm_a])) - float(endpoint(instance[arm_b]))
            for instance in instances
        ],
        dtype=np.float64,
    )


def paired_analysis(
    instances: list[dict[str, dict[str, Any]]],
    *,
    arm_a: str,
    arm_b: str,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    values = _paired_values(instances, arm_a, arm_b)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(replicates, len(values)))
    bootstrap = values[indices].mean(axis=1)
    a_only = sum(endpoint(item[arm_a]) and not endpoint(item[arm_b]) for item in instances)
    b_only = sum(endpoint(item[arm_b]) and not endpoint(item[arm_a]) for item in instances)
    discordant = a_only + b_only
    p_value = (
        float(
            binomtest(
                min(a_only, b_only),
                n=discordant,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )
        if discordant
        else 1.0
    )
    return {
        "arm_a": arm_a,
        "arm_b": arm_b,
        "n_instances": len(instances),
        "difference": float(values.mean()),
        "paired_bootstrap_ci_95": [
            float(np.percentile(bootstrap, 2.5)),
            float(np.percentile(bootstrap, 97.5)),
        ],
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "discordant_pairs": {
            "arm_a_success_arm_b_failure": a_only,
            "arm_a_failure_arm_b_success": b_only,
            "total": discordant,
            "mcnemar_exact_two_sided_p": p_value,
        },
    }


def pooled_cluster_analysis(
    by_seed: dict[int, list[dict[str, dict[str, Any]]]],
    *,
    arm_a: str,
    arm_b: str,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    seed_values = {
        policy_seed: _paired_values(instances, arm_a, arm_b)
        for policy_seed, instances in by_seed.items()
    }
    matrix = np.stack([seed_values[policy_seed] for policy_seed in SEEDS], axis=1)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(matrix), size=(replicates, len(matrix)))
    bootstrap = matrix[indices].mean(axis=(1, 2))
    pooled_instances = [item for policy_seed in SEEDS for item in by_seed[policy_seed]]
    a_only = sum(endpoint(item[arm_a]) and not endpoint(item[arm_b]) for item in pooled_instances)
    b_only = sum(endpoint(item[arm_b]) and not endpoint(item[arm_a]) for item in pooled_instances)
    discordant = a_only + b_only
    p_value = (
        float(
            binomtest(
                min(a_only, b_only),
                n=discordant,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )
        if discordant
        else 1.0
    )
    return {
        "arm_a": arm_a,
        "arm_b": arm_b,
        "n_unique_instances": len(matrix),
        "n_seed_instance_pairs": int(matrix.size),
        "difference": float(matrix.mean()),
        "whole_instance_cluster_bootstrap_ci_95": [
            float(np.percentile(bootstrap, 2.5)),
            float(np.percentile(bootstrap, 97.5)),
        ],
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "cluster_unit": "instance_identity_with_both_policy_seeds_resampled_together",
        "pooled_discordant_pairs": {
            "arm_a_success_arm_b_failure": a_only,
            "arm_a_failure_arm_b_success": b_only,
            "total": discordant,
            "mcnemar_exact_two_sided_p": p_value,
        },
    }


def _validate_result(
    result: dict[str, Any],
    *,
    episode_id: str,
    arm: str,
    checkpoint_seed: int,
    checkpoint_sha256: str,
    rollout_id: str | None = None,
    schedule_row_sha256: str | None = None,
) -> None:
    expected: dict[str, Any] = {
        "episode_id": episode_id,
        "arm": arm,
        "checkpoint_seed": checkpoint_seed,
        "checkpoint_sha256": checkpoint_sha256,
    }
    if rollout_id is not None:
        expected["rollout_id"] = rollout_id
    if schedule_row_sha256 is not None:
        expected["schedule_row_sha256"] = schedule_row_sha256
    if any(result.get(key) != value for key, value in expected.items()):
        raise ValueError(f"identity:{checkpoint_seed}:{episode_id}:{arm}")
    if endpoint(result) != bool(result["collision_free_task_success"]):
        raise ValueError(f"endpoint:{checkpoint_seed}:{episode_id}:{arm}")


def load_results(
    schedule: dict[str, Any], output_root: Path
) -> tuple[dict[int, dict[str, dict[str, dict[str, Any]]]], dict[str, Any]]:
    matrix: dict[int, dict[str, dict[str, dict[str, Any]]]] = {3101: {}, 3102: {}}
    errors: list[str] = []
    for reference in schedule["seed_3101_references"]:
        result_path = Path(reference["result_path"])
        driver_path = Path(reference["driver_path"])
        try:
            if (
                file_hash(result_path) != reference["result_sha256"]
                or file_hash(driver_path) != reference["driver_sha256"]
            ):
                raise ValueError("reference_hash")
            driver = json.loads(driver_path.read_text())
            result = json.loads(result_path.read_text())
            if driver.get("status") != "complete":
                raise ValueError("reference_driver")
            _validate_result(
                result,
                episode_id=reference["instance_episode_id"],
                arm=reference["arm"],
                checkpoint_seed=3101,
                checkpoint_sha256=reference["checkpoint_sha256"],
            )
            matrix[3101].setdefault(reference["instance_episode_id"], {})[reference["arm"]] = result
        except (FileNotFoundError, json.JSONDecodeError, ValueError) as error:
            errors.append(f"seed3101:{reference['instance_episode_id']}:{reference['arm']}:{error}")
    for row in schedule["rows"]:
        row_dir = output_root / row["output_relpath"]
        result_path = row_dir / "result.json"
        driver_path = row_dir / "driver_result.json"
        try:
            driver = json.loads(driver_path.read_text())
            result = json.loads(result_path.read_text())
            if driver.get("status") != "complete":
                raise ValueError("driver")
            _validate_result(
                result,
                episode_id=row["instance_episode_id"],
                arm=row["arm"],
                checkpoint_seed=3102,
                checkpoint_sha256=row["checkpoint_sha256"],
                rollout_id=row["rollout_id"],
                schedule_row_sha256=row["schedule_row_sha256"],
            )
            matrix[3102].setdefault(row["instance_episode_id"], {})[row["arm"]] = result
        except (FileNotFoundError, json.JSONDecodeError, ValueError) as error:
            errors.append(f"seed3102:{row['rollout_id']}:{error}")
    valid_cells = sum(len(arms) for seed_matrix in matrix.values() for arms in seed_matrix.values())
    reconciled = (
        not errors
        and valid_cells == 240
        and all(
            len(seed_matrix) == 40 and all(set(arms) == set(ARMS) for arms in seed_matrix.values())
            for seed_matrix in matrix.values()
        )
    )
    return matrix, {
        "expected_cells": 240,
        "valid_cells": valid_cells,
        "errors": errors,
        "reconciled": reconciled,
    }


def choose_decision(reconciled: bool, seed_contrasts: dict[int, dict[str, dict[str, Any]]]) -> str:
    if not reconciled:
        return "SEED_REPLICATION_INCOMPLETE"
    pact_act_3102 = seed_contrasts[3102]["PACT_minus_ACT"]["difference"]
    modality_3101 = seed_contrasts[3101]["PACT_minus_PACT_PERMUTED"]["difference"]
    modality_3102 = seed_contrasts[3102]["PACT_minus_PACT_PERMUTED"]["difference"]
    if pact_act_3102 < 0.10:
        return "SEED_REPLICATION_FAILED"
    if modality_3101 > 0.0 and modality_3102 > 0.0:
        return "SEED_REPLICATION_CONFIRMED"
    return "SEED_REPLICATION_PARTIAL"


def analyze(schedule: dict[str, Any], output_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    matrix, reconciliation = load_results(schedule, output_root)
    if not reconciliation["reconciled"]:
        token = "SEED_REPLICATION_INCOMPLETE"
        return (
            {
                "schema_version": "pact_seed_replication_analysis_v1",
                "schedule_sha256": schedule["schedule_sha256"],
                "results_available": False,
                "reconciliation": reconciliation,
            },
            {
                "schema_version": "pact_seed_replication_decision_v1",
                "schedule_sha256": schedule["schedule_sha256"],
                "decision": token,
                "reason": "The frozen 240-cell two-seed matrix did not reconcile.",
            },
        )
    by_seed = {
        policy_seed: [seed_matrix[key] for key in sorted(seed_matrix)]
        for policy_seed, seed_matrix in matrix.items()
    }
    summaries: dict[int, dict[str, dict[str, Any]]] = {}
    seed_contrasts: dict[int, dict[str, dict[str, Any]]] = {}
    for policy_seed in SEEDS:
        summaries[policy_seed] = {
            arm: arm_summary([item[arm] for item in by_seed[policy_seed]]) for arm in ARMS
        }
        seed_contrasts[policy_seed] = {}
        for contrast_index, (arm_a, arm_b) in enumerate(CONTRASTS):
            seed_contrasts[policy_seed][f"{arm_a}_minus_{arm_b}"] = paired_analysis(
                by_seed[policy_seed],
                arm_a=arm_a,
                arm_b=arm_b,
                replicates=int(schedule["bootstrap_replicates"]),
                seed=int(schedule["bootstrap_seed"]) + 10 * (policy_seed - 3101) + contrast_index,
            )
    pooled_summaries = {
        arm: arm_summary([item[arm] for policy_seed in SEEDS for item in by_seed[policy_seed]])
        for arm in ARMS
    }
    pooled_contrasts = {}
    for contrast_index, (arm_a, arm_b) in enumerate(CONTRASTS):
        pooled_contrasts[f"{arm_a}_minus_{arm_b}"] = pooled_cluster_analysis(
            by_seed,
            arm_a=arm_a,
            arm_b=arm_b,
            replicates=int(schedule["bootstrap_replicates"]),
            seed=int(schedule["bootstrap_seed"]) + 100 + contrast_index,
        )
    token = choose_decision(True, seed_contrasts)
    analysis = {
        "schema_version": "pact_seed_replication_analysis_v1",
        "schedule_sha256": schedule["schedule_sha256"],
        "results_available": True,
        "reconciliation": reconciliation,
        "seed_results_unpooled_first": {
            str(policy_seed): {
                "arms": summaries[policy_seed],
                "contrasts": seed_contrasts[policy_seed],
            }
            for policy_seed in SEEDS
        },
        "pooled_after_unpooled": {
            "arms": pooled_summaries,
            "contrasts": pooled_contrasts,
        },
        "decision_rule": schedule["decision_rule"],
        "pact_zero_excluded": True,
        "confirmatory_pact_tokens_prohibited": True,
    }
    decision = {
        "schema_version": "pact_seed_replication_decision_v1",
        "schedule_sha256": schedule["schedule_sha256"],
        "decision": token,
        "seed_3102_pact_minus_act": seed_contrasts[3102]["PACT_minus_ACT"]["difference"],
        "seed_3101_pact_minus_permuted": seed_contrasts[3101]["PACT_minus_PACT_PERMUTED"][
            "difference"
        ],
        "seed_3102_pact_minus_permuted": seed_contrasts[3102]["PACT_minus_PACT_PERMUTED"][
            "difference"
        ],
        "confirmatory_run_authorized": token == "SEED_REPLICATION_CONFIRMED",
        "no_confirmatory_pact_decision_awarded": True,
    }
    return analysis, decision


def render(analysis: dict[str, Any], decision: dict[str, Any]) -> str:
    token = decision["decision"]
    lines = [
        "# PACT independent-seed replication decision",
        "",
        "This is the pre-confirmatory seed replication, not a powered confirmatory experiment.",
        "PACT_ZERO is excluded because the all-zero 32-D token is out of distribution.",
        "",
        f"Decision: `{token}`",
        "",
    ]
    if analysis.get("results_available"):
        for policy_seed in SEEDS:
            seed_data = analysis["seed_results_unpooled_first"][str(policy_seed)]
            lines.extend(
                [
                    f"## Seed {policy_seed}",
                    "",
                    "| Arm | Collision-free task success | Ordinary task success |",
                    "|---|---:|---:|",
                ]
            )
            for arm in ARMS:
                item = seed_data["arms"][arm]
                ci = item["collision_free_task_success_wilson_95"]
                lines.append(
                    f"| {arm} | {item['collision_free_task_success']}/{item['n']} "
                    f"({100 * item['collision_free_task_success_rate']:.1f}%, "
                    f"95% Wilson {100 * ci[0]:.1f}–{100 * ci[1]:.1f}%) | "
                    f"{item['ordinary_task_success']}/{item['n']} "
                    f"({100 * item['ordinary_task_success_rate']:.1f}%) |"
                )
            lines.extend(
                [
                    "",
                    "| Contrast | Difference | Paired CI | Discordance | p |",
                    "|---|---:|---:|---:|---:|",
                ]
            )
            for arm_a, arm_b in CONTRASTS:
                item = seed_data["contrasts"][f"{arm_a}_minus_{arm_b}"]
                discordant = item["discordant_pairs"]
                ci = item["paired_bootstrap_ci_95"]
                lines.append(
                    f"| {arm_a} − {arm_b} | {100 * item['difference']:+.1f} pp | "
                    f"[{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}] pp | "
                    f"{discordant['arm_a_success_arm_b_failure']} / "
                    f"{discordant['arm_a_failure_arm_b_success']} | "
                    f"{discordant['mcnemar_exact_two_sided_p']:.4g} |"
                )
            lines.append("")
        lines.extend(
            [
                "## Pooled across both seeds",
                "",
                "Whole instances are the bootstrap clusters; both seed outcomes for a sampled instance move together.",
                "",
                "| Contrast | Difference | Instance-clustered CI |",
                "|---|---:|---:|",
            ]
        )
        for arm_a, arm_b in CONTRASTS:
            item = analysis["pooled_after_unpooled"]["contrasts"][f"{arm_a}_minus_{arm_b}"]
            ci = item["whole_instance_cluster_bootstrap_ci_95"]
            lines.append(
                f"| {arm_a} − {arm_b} | {100 * item['difference']:+.1f} pp | "
                f"[{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}] pp |"
            )
        lines.append("")
    else:
        lines.extend(["The frozen schedule did not reconcile; no outcomes were analyzed.", ""])
    lines.extend(
        [
            (
                "No `PACT_BENEFIT_ESTABLISHED`, `PACT_NO_CONFIRMED_BENEFIT`, "
                "or `PACT_WORSE_THAN_ACT` token can be awarded by this "
                "replication step."
            ),
            "",
            token,
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--analysis-out", required=True, type=Path)
    parser.add_argument("--decision-out", required=True, type=Path)
    parser.add_argument("--report-out", required=True, type=Path)
    args = parser.parse_args()
    schedule = json.loads(args.schedule.read_text())
    payload = dict(schedule)
    observed = payload.pop("schedule_sha256", None)
    if observed != canonical_hash(payload):
        raise SystemExit("seed-replication schedule self-hash mismatch")
    analysis, decision = analyze(schedule, args.output_root.resolve())
    analysis["analysis_script_sha256"] = file_hash(Path(__file__).resolve())
    analysis["analysis_sha256"] = canonical_hash(analysis)
    decision["analysis_sha256"] = analysis["analysis_sha256"]
    decision["final_decision_sha256"] = canonical_hash(decision)
    for path, document in ((args.analysis_out, analysis), (args.decision_out, decision)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(render(analysis, decision))
    print(decision["decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
