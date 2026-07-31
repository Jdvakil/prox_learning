#!/usr/bin/env python3
"""Frozen paired analysis for PACT versus distribution-matched permutation."""

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


ARMS = ("PACT", "PACT_PERMUTED")
TOKENS = {
    "VALID_ABLATION_SIGNAL_PRESENT",
    "VALID_ABLATION_WEAK_SIGNAL",
    "VALID_ABLATION_NO_SIGNAL",
    "VALID_ABLATION_INCONCLUSIVE",
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
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = (
        z
        * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
        / denominator
    )
    return [center - half, center + half]


def endpoint(result: dict[str, Any]) -> bool:
    contacts = result["contact_audit"]["contact_class_totals"]
    return bool(
        result["task_success"]
        and int(contacts.get("hazard_bar", 0)) == 0
        and int(contacts.get("other_environment", 0)) == 0
    )


def validate_result(
    result: dict[str, Any],
    *,
    episode_id: str,
    arm: str,
    checkpoint_sha256: str,
    rollout_id: str | None = None,
    schedule_row_sha256: str | None = None,
) -> None:
    expected = {
        "episode_id": episode_id,
        "arm": arm,
        "checkpoint_sha256": checkpoint_sha256,
    }
    if rollout_id is not None:
        expected["rollout_id"] = rollout_id
    if schedule_row_sha256 is not None:
        expected["schedule_row_sha256"] = schedule_row_sha256
    if any(result.get(key) != value for key, value in expected.items()):
        raise ValueError(f"result identity mismatch: {episode_id} {arm}")
    if endpoint(result) != bool(result["collision_free_task_success"]):
        raise ValueError(f"primary endpoint mismatch: {episode_id} {arm}")


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    primary = sum(endpoint(row) for row in rows)
    task = sum(bool(row["task_success"]) for row in rows)
    classes = ("grasp_target", "hazard_bar", "other_environment")
    return {
        "n": total,
        "collision_free_task_success": primary,
        "collision_free_task_success_rate": primary / total,
        "collision_free_task_success_wilson_95": wilson(primary, total),
        "ordinary_task_success": task,
        "ordinary_task_success_rate": task / total,
        "ordinary_task_success_wilson_95": wilson(task, total),
        "contact_pair_entry_totals": {
            key: sum(
                int(row["contact_audit"]["contact_class_totals"].get(key, 0))
                for row in rows
            )
            for key in classes
        },
        "episodes_with_contact": {
            key: sum(
                int(row["contact_audit"]["contact_class_totals"].get(key, 0))
                > 0
                for row in rows
            )
            for key in classes
        },
        "failure_taxonomy": dict(
            Counter(row["failure_taxonomy"] for row in rows)
        ),
    }


def paired(
    pairs: list[dict[str, dict[str, Any]]],
    *,
    replicates: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    differences = np.asarray(
        [
            float(endpoint(pair["PACT"]))
            - float(endpoint(pair["PACT_PERMUTED"]))
            for pair in pairs
        ]
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, len(differences), size=(replicates, len(differences))
    )
    boot = differences[indices].mean(axis=1)
    pact_only = sum(
        endpoint(pair["PACT"]) and not endpoint(pair["PACT_PERMUTED"])
        for pair in pairs
    )
    permuted_only = sum(
        endpoint(pair["PACT_PERMUTED"]) and not endpoint(pair["PACT"])
        for pair in pairs
    )
    discordant = pact_only + permuted_only
    mcnemar_p = (
        float(
            binomtest(
                min(pact_only, permuted_only),
                n=discordant,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )
        if discordant
        else 1.0
    )
    return (
        {
            "arm_a": "PACT",
            "arm_b": "PACT_PERMUTED",
            "n_instances": len(pairs),
            "difference": float(differences.mean()),
            "ci_95": [
                float(np.percentile(boot, 2.5)),
                float(np.percentile(boot, 97.5)),
            ],
            "replicates": replicates,
            "seed": seed,
        },
        {
            "arm_a": "PACT",
            "arm_b": "PACT_PERMUTED",
            "arm_a_success_arm_b_failure": pact_only,
            "arm_a_failure_arm_b_success": permuted_only,
            "discordant_pairs": discordant,
            "p_value_exact_two_sided": mcnemar_p,
        },
    )


def decision(reconciled: bool, contrast: dict[str, Any] | None) -> str:
    if not reconciled:
        return "VALID_ABLATION_INCONCLUSIVE"
    assert contrast is not None
    difference = float(contrast["difference"])
    lower = float(contrast["ci_95"][0])
    if difference >= 0.10 and lower > 0.0:
        return "VALID_ABLATION_SIGNAL_PRESENT"
    if difference >= 0.05:
        return "VALID_ABLATION_WEAK_SIGNAL"
    return "VALID_ABLATION_NO_SIGNAL"


def analyze(
    schedule: dict[str, Any], output_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    by_instance: dict[str, dict[str, dict[str, Any]]] = {}
    errors: list[str] = []
    for reference in schedule["paired_pact_reference"]:
        result_path = Path(reference["result_path"])
        driver_path = Path(reference["driver_path"])
        if (
            not result_path.exists()
            or not driver_path.exists()
            or file_hash(result_path) != reference["result_sha256"]
            or file_hash(driver_path) != reference["driver_sha256"]
        ):
            errors.append(f"reference_changed:{reference['instance_episode_id']}")
            continue
        driver = json.loads(driver_path.read_text())
        result = json.loads(result_path.read_text())
        if driver.get("status") != "complete":
            errors.append(f"reference_driver:{reference['instance_episode_id']}")
            continue
        try:
            validate_result(
                result,
                episode_id=reference["instance_episode_id"],
                arm="PACT",
                checkpoint_sha256=reference["checkpoint_sha256"],
            )
        except ValueError as exc:
            errors.append(str(exc))
            continue
        by_instance.setdefault(reference["instance_episode_id"], {})[
            "PACT"
        ] = result

    for row in schedule["rows"]:
        row_dir = output_root / row["output_relpath"]
        result_path = row_dir / "result.json"
        driver_path = row_dir / "driver_result.json"
        if not result_path.exists() or not driver_path.exists():
            errors.append(f"missing:{row['rollout_id']}")
            continue
        driver = json.loads(driver_path.read_text())
        result = json.loads(result_path.read_text())
        if driver.get("status") != "complete":
            errors.append(f"driver:{row['rollout_id']}")
            continue
        try:
            validate_result(
                result,
                episode_id=row["instance_episode_id"],
                arm="PACT_PERMUTED",
                checkpoint_sha256=row["checkpoint_sha256"],
                rollout_id=row["rollout_id"],
                schedule_row_sha256=row["schedule_row_sha256"],
            )
        except ValueError as exc:
            errors.append(str(exc))
            continue
        by_instance.setdefault(row["instance_episode_id"], {})[
            "PACT_PERMUTED"
        ] = result

    pairs = [by_instance[key] for key in sorted(by_instance)]
    reconciled = (
        not errors
        and len(pairs) == 40
        and all(set(pair) == set(ARMS) for pair in pairs)
    )
    reconciliation = {
        "expected_pairs": 40,
        "valid_pairs": sum(set(pair) == set(ARMS) for pair in pairs),
        "errors": errors,
        "reconciled": reconciled,
    }
    if not reconciled:
        token = decision(False, None)
        analysis = {
            "schema_version": "pact_valid_ablation_analysis_v1",
            "schedule_sha256": schedule["schedule_sha256"],
            "results_available": False,
            "reconciliation": reconciliation,
        }
        final = {
            "schema_version": "pact_valid_ablation_decision_v1",
            "schedule_sha256": schedule["schedule_sha256"],
            "decision": token,
            "reason": "The frozen 40-pair schedule did not reconcile.",
        }
        return analysis, final
    rows_by_arm = {
        arm: [pair[arm] for pair in pairs] for arm in ARMS
    }
    contrast, mcnemar = paired(
        pairs,
        replicates=int(schedule["bootstrap_replicates"]),
        seed=int(schedule["bootstrap_seed"]),
    )
    token = decision(True, contrast)
    analysis = {
        "schema_version": "pact_valid_ablation_analysis_v1",
        "schedule_sha256": schedule["schedule_sha256"],
        "results_available": True,
        "reconciliation": reconciliation,
        "pooled": {
            arm: summary(rows_by_arm[arm]) for arm in ARMS
        },
        "paired_instance_bootstrap": {
            "PACT_minus_PACT_PERMUTED": contrast
        },
        "mcnemar_exact_primary": mcnemar,
        "decision_rule": schedule["decision_rule"],
        "zero_ablation_not_used_for_decision": True,
        "screen_not_confirmatory": True,
    }
    final = {
        "schema_version": "pact_valid_ablation_decision_v1",
        "schedule_sha256": schedule["schedule_sha256"],
        "decision": token,
        "primary_difference": contrast["difference"],
        "primary_ci_95": contrast["ci_95"],
        "mcnemar_p_value": mcnemar["p_value_exact_two_sided"],
        "screen_not_confirmatory": True,
        "zero_ablation_invalidated": True,
    }
    return analysis, final


def render(analysis: dict[str, Any], final: dict[str, Any]) -> str:
    token = final["decision"]
    lines = [
        "# PACT distribution-matched ablation decision",
        "",
        "This is a post-screen validity check, not a confirmatory experiment.",
        "The all-zero 32-D ablation is excluded because it is out of distribution.",
        "",
        f"Decision: `{token}`",
        "",
    ]
    if analysis.get("results_available"):
        lines.extend(
            [
                "| Arm | Collision-free task success | Ordinary task success |",
                "|---|---:|---:|",
            ]
        )
        for arm in ARMS:
            item = analysis["pooled"][arm]
            ci = item["collision_free_task_success_wilson_95"]
            lines.append(
                f"| {arm} | {item['collision_free_task_success']}/{item['n']} "
                f"({100 * item['collision_free_task_success_rate']:.1f}%, "
                f"95% Wilson {100 * ci[0]:.1f}–{100 * ci[1]:.1f}%) | "
                f"{item['ordinary_task_success']}/{item['n']} "
                f"({100 * item['ordinary_task_success_rate']:.1f}%) |"
            )
        contrast = analysis["paired_instance_bootstrap"][
            "PACT_minus_PACT_PERMUTED"
        ]
        discordant = analysis["mcnemar_exact_primary"]
        lines.extend(
            [
                "",
                "Primary PACT − PACT_PERMUTED: "
                f"{100 * contrast['difference']:+.1f} pp, paired bootstrap "
                f"95% CI [{100 * contrast['ci_95'][0]:+.1f}, "
                f"{100 * contrast['ci_95'][1]:+.1f}] pp.",
                "Discordant pairs: PACT-only success "
                f"{discordant['arm_a_success_arm_b_failure']}, "
                "PACT_PERMUTED-only success "
                f"{discordant['arm_a_failure_arm_b_success']}; exact "
                f"McNemar p={discordant['p_value_exact_two_sided']:.4g}.",
                "",
            ]
        )
    else:
        lines.extend(["The paired schedule did not reconcile.", ""])
    lines.extend(
        [
            "No confirmatory PACT decision token is awarded by this check.",
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
        raise SystemExit("valid-ablation schedule self-hash mismatch")
    analysis, final = analyze(schedule, args.output_root)
    analysis["analysis_script_sha256"] = file_hash(Path(__file__))
    analysis["analysis_sha256"] = canonical_hash(analysis)
    final["analysis_sha256"] = analysis["analysis_sha256"]
    final["final_decision_sha256"] = canonical_hash(final)
    for path, document in (
        (args.analysis_out, analysis),
        (args.decision_out, final),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(render(analysis, final))
    print(final["decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
