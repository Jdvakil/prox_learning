#!/usr/bin/env python3
"""Frozen analysis for the PACT versus ACT confirmatory schedule."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import fisher_exact

ARMS = ("ACT", "PACT", "PACT_ZERO")
TOKENS = {
    "PACT_BENEFIT_ESTABLISHED",
    "PACT_NO_CONFIRMED_BENEFIT",
    "PACT_WORSE_THAN_ACT",
    "PACT_EXPERIMENT_INCOMPLETE",
}


def canonical_hash(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054):
    if total == 0:
        return None, None
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
    return center - half, center + half


def arm_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    primary = sum(bool(row["collision_free_task_success"]) for row in rows)
    task = sum(bool(row["task_success"]) for row in rows)
    primary_interval = wilson_interval(primary, total)
    task_interval = wilson_interval(task, total)
    contact_totals = {
        contact_class: sum(
            int(
                row["contact_audit"]["contact_class_totals"].get(contact_class, 0)
            )
            for row in rows
        )
        for contact_class in (
            "grasp_target",
            "hazard_bar",
            "other_environment",
        )
    }
    contact_episode_counts = {
        contact_class: sum(
            int(
                row["contact_audit"]["contact_class_totals"].get(contact_class, 0)
            )
            > 0
            for row in rows
        )
        for contact_class in contact_totals
    }
    return {
        "n": total,
        "collision_free_task_success": primary,
        "collision_free_task_success_rate": primary / total if total else None,
        "collision_free_task_success_wilson_95": list(primary_interval),
        "ordinary_task_success": task,
        "ordinary_task_success_rate": task / total if total else None,
        "ordinary_task_success_wilson_95": list(task_interval),
        "contact_pair_entry_totals": contact_totals,
        "episodes_with_contact": contact_episode_counts,
        "failure_taxonomy": dict(Counter(row["failure_taxonomy"] for row in rows)),
    }


def paired_bootstrap(
    instances: list[dict[str, dict[str, Any]]],
    *,
    arm_a: str,
    arm_b: str,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    differences = np.asarray(
        [
            float(instance[arm_a]["collision_free_task_success"])
            - float(instance[arm_b]["collision_free_task_success"])
            for instance in instances
        ],
        dtype=np.float64,
    )
    if not len(differences):
        return {"difference": None, "ci_95": [None, None]}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(replicates, len(differences)))
    bootstrap = differences[indices].mean(axis=1)
    return {
        "arm_a": arm_a,
        "arm_b": arm_b,
        "n_instances": len(differences),
        "difference": float(np.mean(differences)),
        "ci_95": [
            float(np.percentile(bootstrap, 2.5)),
            float(np.percentile(bootstrap, 97.5)),
        ],
        "replicates": replicates,
        "seed": seed,
    }


def fisher_comparison(
    summaries: dict[str, dict[str, Any]], arm_a: str, arm_b: str
) -> dict[str, Any]:
    first = summaries[arm_a]
    second = summaries[arm_b]
    table = [
        [
            first["collision_free_task_success"],
            first["n"] - first["collision_free_task_success"],
        ],
        [
            second["collision_free_task_success"],
            second["n"] - second["collision_free_task_success"],
        ],
    ]
    odds_ratio, p_value = fisher_exact(table, alternative="two-sided")
    return {
        "arm_a": arm_a,
        "arm_b": arm_b,
        "table_success_failure": table,
        "odds_ratio": float(odds_ratio),
        "p_value_two_sided": float(p_value),
    }


def choose_decision(
    *,
    reconciled: bool,
    pact_vs_act: dict[str, Any] | None,
    pact_vs_zero: dict[str, Any] | None,
    fisher_pact_act: dict[str, Any] | None,
    fisher_pact_zero: dict[str, Any] | None,
) -> str:
    if not reconciled:
        return "PACT_EXPERIMENT_INCOMPLETE"
    assert pact_vs_act and pact_vs_zero and fisher_pact_act and fisher_pact_zero
    if (
        pact_vs_act["difference"] < 0
        and pact_vs_act["ci_95"][1] < 0
        and fisher_pact_act["p_value_two_sided"] < 0.05
    ):
        return "PACT_WORSE_THAN_ACT"
    if (
        pact_vs_act["difference"] > 0
        and pact_vs_zero["difference"] > 0
        and pact_vs_act["ci_95"][0] > 0
        and pact_vs_zero["ci_95"][0] > 0
        and fisher_pact_act["p_value_two_sided"] < 0.05
        and fisher_pact_zero["p_value_two_sided"] < 0.05
    ):
        return "PACT_BENEFIT_ESTABLISHED"
    return "PACT_NO_CONFIRMED_BENEFIT"


def _load_results(schedule: dict, output_root: Path):
    results = []
    reconciliation = {"missing": [], "driver_noncomplete": [], "invalid": []}
    for row in schedule["rows"]:
        row_dir = output_root / row["output_relpath"]
        driver_path = row_dir / "driver_result.json"
        result_path = row_dir / "result.json"
        if not driver_path.exists() or not result_path.exists():
            reconciliation["missing"].append(row["rollout_id"])
            continue
        driver = json.loads(driver_path.read_text())
        if driver.get("status") != "complete":
            reconciliation["driver_noncomplete"].append(row["rollout_id"])
            continue
        result = json.loads(result_path.read_text())
        checks = {
            "rollout_id": row["rollout_id"],
            "schedule_row_sha256": row["schedule_row_sha256"],
            "episode_id": row["instance_episode_id"],
            "arm": row["arm"],
            "checkpoint_sha256": row["checkpoint_sha256"],
        }
        if any(result.get(key) != value for key, value in checks.items()):
            reconciliation["invalid"].append(row["rollout_id"])
            continue
        contacts = result["contact_audit"]["contact_class_totals"]
        recomputed = bool(
            result["task_success"]
            and int(contacts.get("hazard_bar", 0)) == 0
            and int(contacts.get("other_environment", 0)) == 0
        )
        if recomputed != bool(result["collision_free_task_success"]):
            reconciliation["invalid"].append(row["rollout_id"])
            continue
        results.append(result)
    reconciliation["expected"] = len(schedule["rows"])
    reconciliation["valid"] = len(results)
    reconciliation["reconciled"] = not any(
        reconciliation[key] for key in ("missing", "driver_noncomplete", "invalid")
    )
    return results, reconciliation


def analyze(schedule: dict, output_root: Path) -> tuple[dict, dict]:
    results, reconciliation = _load_results(schedule, output_root)
    if not reconciliation["reconciled"]:
        token = choose_decision(
            reconciled=False,
            pact_vs_act=None,
            pact_vs_zero=None,
            fisher_pact_act=None,
            fisher_pact_zero=None,
        )
        analysis = {
            "schema_version": "pact_confirmatory_analysis_v1",
            "schedule_sha256": schedule["schedule_sha256"],
            "reconciliation": reconciliation,
            "results_available": False,
        }
        decision = {
            "schema_version": "pact_final_decision_v1",
            "schedule_sha256": schedule["schedule_sha256"],
            "decision": token,
            "reason": "The frozen schedule did not reconcile; outcomes are not analyzed.",
        }
        return analysis, decision

    by_arm = {
        arm: [result for result in results if result["arm"] == arm] for arm in ARMS
    }
    summaries = {arm: arm_summary(rows) for arm, rows in by_arm.items()}
    by_instance: dict[str, dict[str, Any]] = {}
    for result in results:
        by_instance.setdefault(result["episode_id"], {})[result["arm"]] = result
    instances = [by_instance[key] for key in sorted(by_instance)]
    if len(instances) != 80 or any(set(instance) != set(ARMS) for instance in instances):
        reconciliation["reconciled"] = False
        reconciliation["invalid"].append("instance_arm_matrix")
        analysis = {
            "schema_version": "pact_confirmatory_analysis_v1",
            "schedule_sha256": schedule["schedule_sha256"],
            "reconciliation": reconciliation,
            "results_available": False,
        }
        decision = {
            "schema_version": "pact_final_decision_v1",
            "schedule_sha256": schedule["schedule_sha256"],
            "decision": "PACT_EXPERIMENT_INCOMPLETE",
            "reason": "The instance-by-arm matrix did not reconcile.",
        }
        return analysis, decision
    pact_vs_act = paired_bootstrap(
        instances,
        arm_a="PACT",
        arm_b="ACT",
        replicates=int(schedule["bootstrap_replicates"]),
        seed=int(schedule["bootstrap_seed"]),
    )
    pact_vs_zero = paired_bootstrap(
        instances,
        arm_a="PACT",
        arm_b="PACT_ZERO",
        replicates=int(schedule["bootstrap_replicates"]),
        seed=int(schedule["bootstrap_seed"]) + 1,
    )
    fisher_pact_act = fisher_comparison(summaries, "PACT", "ACT")
    fisher_pact_zero = fisher_comparison(summaries, "PACT", "PACT_ZERO")
    token = choose_decision(
        reconciled=True,
        pact_vs_act=pact_vs_act,
        pact_vs_zero=pact_vs_zero,
        fisher_pact_act=fisher_pact_act,
        fisher_pact_zero=fisher_pact_zero,
    )
    per_seed = {
        arm: {
            str(seed): arm_summary(
                [
                    result
                    for result in by_arm[arm]
                    if int(result["checkpoint_seed"]) == seed
                ]
            )
            for seed in (3101, 3102)
        }
        for arm in ARMS
    }
    analysis = {
        "schema_version": "pact_confirmatory_analysis_v1",
        "schedule_sha256": schedule["schedule_sha256"],
        "reconciliation": reconciliation,
        "results_available": True,
        "primary_endpoint": (
            "task_success and zero hazard_bar and other_environment contacts"
        ),
        "target_contact_exempt": True,
        "pooled": summaries,
        "per_checkpoint_seed": per_seed,
        "paired_instance_bootstrap": {
            "PACT_minus_ACT": pact_vs_act,
            "PACT_minus_PACT_ZERO": pact_vs_zero,
        },
        "fisher_exact": {
            "PACT_vs_ACT": fisher_pact_act,
            "PACT_vs_PACT_ZERO": fisher_pact_zero,
        },
    }
    decision = {
        "schema_version": "pact_final_decision_v1",
        "schedule_sha256": schedule["schedule_sha256"],
        "decision": token,
        "decision_rule": (
            "benefit: both paired CI lower bounds > 0 and both Fisher p < .05; "
            "worse: PACT-ACT paired CI upper < 0 and Fisher p < .05; "
            "otherwise no confirmed benefit"
        ),
        "PACT_minus_ACT": pact_vs_act,
        "PACT_minus_PACT_ZERO": pact_vs_zero,
        "fisher_exact": analysis["fisher_exact"],
    }
    return analysis, decision


def render_report(analysis: dict, decision: dict) -> str:
    token = decision["decision"]
    lines = [
        "# PACT versus ACT final decision",
        "",
        "## Experiment status",
        "",
        f"Decision: `{token}`",
        "",
    ]
    if not analysis["results_available"]:
        reconciliation = analysis["reconciliation"]
        lines.extend(
            [
                "The preregistered schedule did not reconcile, so no endpoint "
                "comparison was interpreted.",
                "",
                f"- Expected rows: {reconciliation['expected']}",
                f"- Valid rows: {reconciliation['valid']}",
                f"- Missing: {len(reconciliation['missing'])}",
                f"- Non-complete driver rows: "
                f"{len(reconciliation['driver_noncomplete'])}",
                f"- Invalid rows: {len(reconciliation['invalid'])}",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Primary and secondary endpoints",
                "",
                "| Arm | Collision-free task success | Wilson 95% CI | "
                "Ordinary task success | Wilson 95% CI |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for arm in ARMS:
            summary = analysis["pooled"][arm]
            pci = summary["collision_free_task_success_wilson_95"]
            tci = summary["ordinary_task_success_wilson_95"]
            lines.append(
                f"| {arm} | {summary['collision_free_task_success']}/"
                f"{summary['n']} ({summary['collision_free_task_success_rate']:.1%}) | "
                f"[{pci[0]:.1%}, {pci[1]:.1%}] | "
                f"{summary['ordinary_task_success']}/{summary['n']} "
                f"({summary['ordinary_task_success_rate']:.1%}) | "
                f"[{tci[0]:.1%}, {tci[1]:.1%}] |"
            )
        lines.extend(["", "## Preregistered comparisons", ""])
        for key, comparison in analysis["paired_instance_bootstrap"].items():
            ci = comparison["ci_95"]
            fisher_key = (
                "PACT_vs_ACT" if key.endswith("ACT") else "PACT_vs_PACT_ZERO"
            )
            p_value = analysis["fisher_exact"][fisher_key]["p_value_two_sided"]
            lines.append(
                f"- {key}: {comparison['difference']:+.1%}, paired-instance "
                f"bootstrap 95% CI [{ci[0]:+.1%}, {ci[1]:+.1%}], "
                f"two-sided Fisher p={p_value:.6g}."
            )
        lines.extend(["", "## Contact totals", ""])
        for arm in ARMS:
            summary = analysis["pooled"][arm]
            lines.append(
                f"- {arm}: pair entries "
                f"{summary['contact_pair_entry_totals']}; episodes "
                f"{summary['episodes_with_contact']}."
            )
        lines.append("")
    lines.extend(
        [
            "## Decision",
            "",
            "The final line is the exact allowed decision token.",
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
    observed = payload.pop("schedule_sha256")
    if canonical_hash(payload) != observed:
        raise SystemExit("schedule self-hash mismatch")
    analysis, decision = analyze(schedule, args.output_root)
    if decision["decision"] not in TOKENS:
        raise RuntimeError("analysis produced a disallowed decision")
    for path, value in (
        (args.analysis_out, analysis),
        (args.decision_out, decision),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(render_report(analysis, decision))
    print(decision["decision"])
    return 0 if analysis["reconciliation"]["reconciled"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
