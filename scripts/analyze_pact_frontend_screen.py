#!/usr/bin/env python3
"""Frozen paired analysis and decision rule for the 120-rollout screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest, fisher_exact

ARMS = ("ACT", "PACT", "PACT_ZERO")
TOKENS = {
    "FRONTEND_SCREEN_SIGNAL_PRESENT",
    "FRONTEND_SCREEN_WEAK_SIGNAL",
    "FRONTEND_SCREEN_NO_SIGNAL",
    "FRONTEND_SCREEN_INCONCLUSIVE",
}


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def wilson_interval(
    successes: int,
    total: int,
    z: float = 1.959963984540054,
) -> tuple[float | None, float | None]:
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
    primary = sum(
        bool(row["collision_free_task_success"]) for row in rows
    )
    task = sum(bool(row["task_success"]) for row in rows)
    contact_classes = (
        "grasp_target",
        "hazard_bar",
        "other_environment",
    )
    contact_totals = {
        contact_class: sum(
            int(
                row["contact_audit"]["contact_class_totals"].get(
                    contact_class, 0
                )
            )
            for row in rows
        )
        for contact_class in contact_classes
    }
    contact_episodes = {
        contact_class: sum(
            int(
                row["contact_audit"]["contact_class_totals"].get(
                    contact_class, 0
                )
            )
            > 0
            for row in rows
        )
        for contact_class in contact_classes
    }
    return {
        "n": total,
        "collision_free_task_success": primary,
        "collision_free_task_success_rate": (
            primary / total if total else None
        ),
        "collision_free_task_success_wilson_95": list(
            wilson_interval(primary, total)
        ),
        "ordinary_task_success": task,
        "ordinary_task_success_rate": task / total if total else None,
        "ordinary_task_success_wilson_95": list(
            wilson_interval(task, total)
        ),
        "contact_pair_entry_totals": contact_totals,
        "episodes_with_contact": contact_episodes,
        "failure_taxonomy": dict(
            Counter(row["failure_taxonomy"] for row in rows)
        ),
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
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, len(differences), size=(replicates, len(differences))
    )
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


def discordant_pairs(
    instances: list[dict[str, dict[str, Any]]],
    *,
    arm_a: str,
    arm_b: str,
) -> dict[str, Any]:
    a_only = sum(
        bool(instance[arm_a]["collision_free_task_success"])
        and not bool(instance[arm_b]["collision_free_task_success"])
        for instance in instances
    )
    b_only = sum(
        bool(instance[arm_b]["collision_free_task_success"])
        and not bool(instance[arm_a]["collision_free_task_success"])
        for instance in instances
    )
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
        "arm_a_success_arm_b_failure": a_only,
        "arm_a_failure_arm_b_success": b_only,
        "discordant_pairs": discordant,
        "p_value_exact_two_sided": p_value,
    }


def fisher_comparison(
    summaries: dict[str, dict[str, Any]],
    arm_a: str,
    arm_b: str,
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
    odds_ratio, p_value = fisher_exact(
        table, alternative="two-sided"
    )
    return {
        "arm_a": arm_a,
        "arm_b": arm_b,
        "table_success_failure": table,
        "odds_ratio": float(odds_ratio),
        "p_value_two_sided": float(p_value),
    }


def choose_decision(
    *, reconciled: bool, pact_minus_zero: dict[str, Any] | None
) -> str:
    if not reconciled:
        return "FRONTEND_SCREEN_INCONCLUSIVE"
    assert pact_minus_zero is not None
    difference = float(pact_minus_zero["difference"])
    lower = float(pact_minus_zero["ci_95"][0])
    if difference >= 0.10 and lower > 0.0:
        return "FRONTEND_SCREEN_SIGNAL_PRESENT"
    if difference >= 0.05:
        return "FRONTEND_SCREEN_WEAK_SIGNAL"
    return "FRONTEND_SCREEN_NO_SIGNAL"


def _load_results(
    schedule: dict[str, Any], output_root: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results = []
    reconciliation: dict[str, Any] = {
        "missing": [],
        "driver_noncomplete": [],
        "invalid": [],
    }
    for row in schedule["rows"]:
        row_dir = output_root / row["output_relpath"]
        driver_path = row_dir / "driver_result.json"
        result_path = row_dir / "result.json"
        if not driver_path.exists() or not result_path.exists():
            reconciliation["missing"].append(row["rollout_id"])
            continue
        driver = json.loads(driver_path.read_text())
        if driver.get("status") != "complete":
            reconciliation["driver_noncomplete"].append(
                row["rollout_id"]
            )
            continue
        result = json.loads(result_path.read_text())
        checks = {
            "rollout_id": row["rollout_id"],
            "schedule_row_sha256": row["schedule_row_sha256"],
            "episode_id": row["instance_episode_id"],
            "arm": row["arm"],
            "checkpoint_sha256": row["checkpoint_sha256"],
            "checkpoint_seed": row["checkpoint_seed"],
        }
        if any(
            result.get(key) != value
            for key, value in checks.items()
        ):
            reconciliation["invalid"].append(row["rollout_id"])
            continue
        contacts = result["contact_audit"]["contact_class_totals"]
        recomputed = bool(
            result["task_success"]
            and int(contacts.get("hazard_bar", 0)) == 0
            and int(contacts.get("other_environment", 0)) == 0
        )
        if recomputed != bool(
            result["collision_free_task_success"]
        ):
            reconciliation["invalid"].append(row["rollout_id"])
            continue
        results.append(result)
    reconciliation["expected"] = len(schedule["rows"])
    reconciliation["valid"] = len(results)
    reconciliation["reconciled"] = not any(
        reconciliation[key]
        for key in ("missing", "driver_noncomplete", "invalid")
    )
    return results, reconciliation


def analyze(
    schedule: dict[str, Any], output_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    results, reconciliation = _load_results(schedule, output_root)
    if not reconciliation["reconciled"]:
        token = choose_decision(
            reconciled=False, pact_minus_zero=None
        )
        analysis = {
            "schema_version": "pact_frontend_screen_analysis_v1",
            "schedule_sha256": schedule["schedule_sha256"],
            "reconciliation": reconciliation,
            "results_available": False,
        }
        decision = {
            "schema_version": "pact_frontend_screen_decision_v1",
            "schedule_sha256": schedule["schedule_sha256"],
            "decision": token,
            "reason": (
                "The frozen 120-row schedule did not reconcile; "
                "outcomes were not analyzed."
            ),
        }
        return analysis, decision

    by_arm = {
        arm: [result for result in results if result["arm"] == arm]
        for arm in ARMS
    }
    summaries = {
        arm: arm_summary(rows) for arm, rows in by_arm.items()
    }
    by_instance: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for result in results:
        by_instance.setdefault(result["episode_id"], {}).setdefault(
            result["arm"], []
        ).append(result)
    raw_instances = [
        by_instance[key] for key in sorted(by_instance)
    ]
    valid_matrix = (
        len(raw_instances) == int(schedule["instances"])
        and all(
            set(instance) == set(ARMS) for instance in raw_instances
        )
        and all(
            len(instance[arm]) == 1
            for instance in raw_instances
            for arm in ARMS
        )
    )
    if not valid_matrix:
        reconciliation["reconciled"] = False
        reconciliation["invalid"].append("instance_arm_matrix")
        return analyze_incomplete(schedule, reconciliation)
    instances = [
        {arm: raw[arm][0] for arm in ARMS}
        for raw in raw_instances
    ]
    replicates = int(schedule["bootstrap_replicates"])
    seed = int(schedule["bootstrap_seed"])
    pact_minus_zero = paired_bootstrap(
        instances,
        arm_a="PACT",
        arm_b="PACT_ZERO",
        replicates=replicates,
        seed=seed,
    )
    pact_minus_act = paired_bootstrap(
        instances,
        arm_a="PACT",
        arm_b="ACT",
        replicates=replicates,
        seed=seed + 1,
    )
    mcnemar = discordant_pairs(
        instances, arm_a="PACT", arm_b="PACT_ZERO"
    )
    token = choose_decision(
        reconciled=True, pact_minus_zero=pact_minus_zero
    )
    analysis = {
        "schema_version": "pact_frontend_screen_analysis_v1",
        "schedule_sha256": schedule["schedule_sha256"],
        "reconciliation": reconciliation,
        "results_available": True,
        "screen_not_confirmatory": True,
        "design": {
            "instances": schedule["instances"],
            "rollouts": schedule["rollouts"],
            "workers": schedule["workers"],
            "repeats_per_instance_per_arm": 1,
            "fresh_subprocess_per_rollout": schedule[
                "fresh_subprocess_per_rollout"
            ],
            "detectable_effect_statement": schedule[
                "detectable_effect_statement"
            ],
        },
        "primary_endpoint": (
            "task_success and zero hazard_bar and "
            "other_environment contacts"
        ),
        "target_contact_exempt": True,
        "pooled": summaries,
        "paired_instance_bootstrap": {
            "PACT_minus_PACT_ZERO": pact_minus_zero,
            "PACT_minus_ACT_secondary": pact_minus_act,
        },
        "mcnemar_exact_primary": mcnemar,
        "fisher_exact_secondary": {
            "PACT_vs_PACT_ZERO": fisher_comparison(
                summaries, "PACT", "PACT_ZERO"
            ),
            "PACT_vs_ACT": fisher_comparison(
                summaries, "PACT", "ACT"
            ),
        },
    }
    decision = {
        "schema_version": "pact_frontend_screen_decision_v1",
        "schedule_sha256": schedule["schedule_sha256"],
        "decision": token,
        "screen_not_confirmatory": True,
        "decision_rule": {
            "FRONTEND_SCREEN_SIGNAL_PRESENT": (
                "PACT-PACT_ZERO >= 0.10 and paired CI lower > 0"
            ),
            "FRONTEND_SCREEN_WEAK_SIGNAL": (
                "PACT-PACT_ZERO >= 0.05 but signal-present rule false"
            ),
            "FRONTEND_SCREEN_NO_SIGNAL": "PACT-PACT_ZERO < 0.05",
            "FRONTEND_SCREEN_INCONCLUSIVE": (
                "schedule did not reconcile"
            ),
        },
        "PACT_minus_PACT_ZERO": pact_minus_zero,
        "mcnemar_exact": mcnemar,
        "PACT_minus_ACT_secondary": pact_minus_act,
        "confirmatory_tokens_prohibited": [
            "PACT_BENEFIT_ESTABLISHED",
            "PACT_NO_CONFIRMED_BENEFIT",
            "PACT_WORSE_THAN_ACT",
        ],
    }
    if token not in TOKENS:
        raise RuntimeError(f"unexpected screen token {token}")
    return analysis, decision


def analyze_incomplete(
    schedule: dict[str, Any], reconciliation: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    analysis = {
        "schema_version": "pact_frontend_screen_analysis_v1",
        "schedule_sha256": schedule["schedule_sha256"],
        "reconciliation": reconciliation,
        "results_available": False,
    }
    decision = {
        "schema_version": "pact_frontend_screen_decision_v1",
        "schedule_sha256": schedule["schedule_sha256"],
        "decision": "FRONTEND_SCREEN_INCONCLUSIVE",
        "reason": "The instance-by-arm matrix did not reconcile.",
    }
    return analysis, decision


def render_report(
    analysis: dict[str, Any],
    decision: dict[str, Any],
    *,
    encoder: dict[str, Any] | None,
    training: dict[str, Any] | None,
) -> str:
    token = decision["decision"]
    lines = [
        "# PACT wider-front-end screen decision",
        "",
        (
            "This was a 40-instance, 120-rollout development screen. It "
            "cannot establish PACT benefit over ACT."
        ),
        "",
        f"Decision: `{token}`",
        "",
    ]
    if encoder is not None:
        metrics = encoder["heldout_metrics"]
        lines.extend(
            [
                "## Frozen 32-D front-end",
                "",
                f"- Checkpoint SHA-256: `{encoder['checkpoint_sha256']}`",
                f"- Parameters: {encoder['parameter_count']:,}",
                (
                    "- Held-out surface error, mean / median: "
                    f"{100 * metrics['mean_euclidean_error_m']:.2f} / "
                    f"{100 * metrics['median_euclidean_error_m']:.2f} cm"
                ),
                (
                    "- Within 2 cm: "
                    f"{100 * metrics['within_2cm_rate']:.1f}%"
                ),
                (
                    "- Validity precision / recall: "
                    f"{100 * metrics['validity_precision']:.1f}% / "
                    f"{100 * metrics['validity_recall']:.1f}%"
                ),
                "",
            ]
        )
    if training is not None:
        lines.extend(
            [
                "## PACT training",
                "",
                f"- Seed: {training['seed']}",
                f"- Best epoch: {training['best_epoch']}",
                (
                    "- Best validation loss: "
                    f"{training['best_validation_loss']:.6f}"
                ),
                f"- Checkpoint SHA-256: `{training['checkpoint_sha256']}`",
                "",
            ]
        )
    if analysis.get("results_available"):
        lines.extend(
            [
                "## Frozen screen analysis",
                "",
                "| Arm | Collision-free task success | Ordinary task success |",
                "|---|---:|---:|",
            ]
        )
        for arm in ARMS:
            summary = analysis["pooled"][arm]
            ci = summary[
                "collision_free_task_success_wilson_95"
            ]
            lines.append(
                f"| {arm} | "
                f"{summary['collision_free_task_success']}/{summary['n']} "
                f"({100 * summary['collision_free_task_success_rate']:.1f}%, "
                f"95% Wilson {100 * ci[0]:.1f}–{100 * ci[1]:.1f}%) | "
                f"{summary['ordinary_task_success']}/{summary['n']} "
                f"({100 * summary['ordinary_task_success_rate']:.1f}%) |"
            )
        primary = analysis["paired_instance_bootstrap"][
            "PACT_minus_PACT_ZERO"
        ]
        discordant = analysis["mcnemar_exact_primary"]
        lines.extend(
            [
                "",
                (
                    "Primary PACT − PACT_ZERO: "
                    f"{100 * primary['difference']:+.1f} pp, paired "
                    f"bootstrap 95% CI [{100 * primary['ci_95'][0]:+.1f}, "
                    f"{100 * primary['ci_95'][1]:+.1f}] pp."
                ),
                (
                    "Discordant pairs: "
                    f"PACT-only success {discordant['arm_a_success_arm_b_failure']}, "
                    f"PACT_ZERO-only success "
                    f"{discordant['arm_a_failure_arm_b_success']}; "
                    f"exact McNemar p={discordant['p_value_exact_two_sided']:.4g}."
                ),
                "",
                (
                    "The ACT comparison is a sanity reference only and is "
                    "not decision-bearing in this screen."
                ),
                "",
            ]
        )
    else:
        lines.extend(
            [
                (
                    "The schedule did not reconcile, so endpoint values "
                    "were not analyzed."
                ),
                "",
            ]
        )
    lines.extend(
        [
            (
                "This token is a screen decision only; no confirmatory "
                "PACT decision token may be inferred from it."
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
    parser.add_argument("--encoder-report", type=Path)
    parser.add_argument("--training-summary", type=Path)
    args = parser.parse_args()
    schedule = json.loads(args.schedule.read_text())
    payload = dict(schedule)
    observed = payload.pop("schedule_sha256", None)
    if observed != canonical_hash(payload):
        raise SystemExit("screen schedule self-hash mismatch")
    analysis, decision = analyze(schedule, args.output_root)
    analysis["analysis_script_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    analysis["analysis_sha256"] = canonical_hash(analysis)
    decision["analysis_sha256"] = analysis["analysis_sha256"]
    decision["final_decision_sha256"] = canonical_hash(decision)
    for path, document in (
        (args.analysis_out, analysis),
        (args.decision_out, decision),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n"
        )
    encoder = (
        json.loads(args.encoder_report.read_text())
        if args.encoder_report
        else None
    )
    training = (
        json.loads(args.training_summary.read_text())
        if args.training_summary
        else None
    )
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(
        render_report(
            analysis,
            decision,
            encoder=encoder,
            training=training,
        )
    )
    print(decision["decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
