#!/usr/bin/env python3
"""Finalize a preregistered PACT experiment stopped at the Phase 1 gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

TOKEN = "PACT_ENVIRONMENT_INADEQUATE"


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def build_stop_documents(
    *,
    gate: dict[str, Any],
    preregistration: dict[str, Any],
    collection_summary: dict[str, Any],
    manifest_sha256: str,
    gate_sha256: str,
    collection_summary_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if gate.get("decision") != TOKEN:
        raise ValueError(f"environment gate must decide {TOKEN}")
    if gate.get("all_applicable_gates_pass") is not False:
        raise ValueError("environment stop requires a failed gate")
    if gate.get("stop_before_policy_training") is not True:
        raise ValueError("gate does not authorize the mandatory early stop")
    if collection_summary.get("complete") is not True:
        raise ValueError("pilot expert collection did not reconcile")
    if collection_summary.get("manifest_sha256") != manifest_sha256:
        raise ValueError("collection and candidate manifest do not match")
    if gate.get("manifest_sha256") != manifest_sha256:
        raise ValueError("gate and candidate manifest do not match")
    failed_checks = sorted(
        name for name, passed in gate["checks"].items() if not passed
    )
    if not failed_checks:
        raise ValueError("environment stop has no failed prerequisite")

    schedule: dict[str, Any] = {
        "schema_version": "pact_confirmatory_schedule_not_instantiated_v1",
        "status": "not_instantiated_due_to_phase1_environment_gate",
        "decision": TOKEN,
        "manifest_sha256": manifest_sha256,
        "environment_gate_sha256": gate_sha256,
        "planned_confirmatory_design": preregistration["confirmatory_design"],
        "rows": [],
        "rollouts_executed": 0,
        "confirmatory_outcomes_seen": False,
        "reason": (
            "The frozen expert/surface prerequisite failed before pilot ACT "
            "training; checkpoint-bound confirmatory rows therefore do not exist."
        ),
    }
    schedule["schedule_sha256"] = canonical_hash(schedule)

    expert = gate["expert"]
    analysis: dict[str, Any] = {
        "schema_version": "pact_vs_act_analysis_environment_stop_v1",
        "experiment_stage": "phase1_environment_adequacy",
        "decision": TOKEN,
        "manifest_sha256": manifest_sha256,
        "environment_gate_sha256": gate_sha256,
        "pilot_collection_summary_sha256": collection_summary_sha256,
        "schedule_sha256": schedule["schedule_sha256"],
        "primary_endpoint": (
            "task_success and zero hazard_bar and other_environment contacts"
        ),
        "target_contact_exempt": True,
        "phase1": {
            "checks": gate["checks"],
            "failed_checks": failed_checks,
            "expert": expert,
            "act": gate["act"],
            "deferred_checks_not_run": gate["deferred_checks_not_run"],
        },
        "arm_comparison": {
            "status": "not_run_due_to_failed_environment_gate",
            "ACT": "not_trained",
            "PACT": "not_trained",
            "PACT_ZERO": "not_evaluated",
            "fisher_exact": "not_applicable",
            "wilson_intervals": "not_applicable",
            "paired_instance_bootstrap": "not_applicable",
        },
        "full_dataset": "not_collected",
        "surface_encoder": "not_trained",
        "policy_checkpoint_sha256s": [],
        "valid_early_stop": True,
    }
    analysis["analysis_sha256"] = canonical_hash(analysis)

    decision: dict[str, Any] = {
        "schema_version": "pact_final_decision_v1",
        "decision": TOKEN,
        "experiment_stage": "phase1_environment_adequacy",
        "manifest_sha256": manifest_sha256,
        "environment_gate_sha256": gate_sha256,
        "analysis_sha256": analysis["analysis_sha256"],
        "schedule_sha256": schedule["schedule_sha256"],
        "failed_checks": failed_checks,
        "reason": (
            f"Expert collision-free task success was "
            f"{expert['collision_free_task_success']}/{expert['n']}; the "
            "preregistered minimum was 20/24. The protocol therefore stopped "
            "before policy training."
        ),
        "arm_comparison_available": False,
        "policy_training_performed": False,
        "confirmatory_evaluation_performed": False,
        "checkpoint_sha256s": [],
    }
    decision["final_decision_sha256"] = canonical_hash(decision)
    return schedule, analysis, decision


def render_report(
    *,
    gate: dict[str, Any],
    schedule: dict[str, Any],
    analysis: dict[str, Any],
    decision: dict[str, Any],
) -> str:
    expert = gate["expert"]
    checks = gate["checks"]
    collision = expert["collision_rows"][0]
    lines = [
        "# PACT versus ACT final decision",
        "",
        "## Outcome",
        "",
        f"Decision: `{decision['decision']}`.",
        "",
        "The frozen collision-route environment did not pass its Phase 1 "
        "solvability prerequisite. The protocol therefore stopped before "
        "dataset conversion, proximity-encoder training, ACT/PACT training, or "
        "any three-arm evaluation. This is a valid preregistered early stop, "
        "not an unreconciled confirmatory schedule.",
        "",
        "## Frozen Phase 1 result",
        "",
        "| Check | Result | Threshold | Pass |",
        "|---|---:|---:|---:|",
        f"| Expert ordinary task success | "
        f"{expert['ordinary_task_success']}/{expert['n']} | >=20/24 | "
        f"{checks['expert_task_success_at_least_20_of_24']} |",
        f"| Expert collision-free task success | "
        f"{expert['collision_free_task_success']}/{expert['n']} | >=20/24 | "
        f"{checks['expert_collision_free_task_success_at_least_20_of_24']} |",
        f"| Episodes with panel signal | "
        f"{expert['episodes_with_intrusion_sighting']}/{expert['n']} | >=20/24 | "
        f"{checks['surface_active_episodes_at_least_20_of_24']} |",
        f"| Pre-grasp frames inside 20 cm | "
        f"{expert['steps_intrusion_inside_20cm']}/"
        f"{expert['pregrasp_control_steps']} "
        f"({expert['fraction_pregrasp_inside_20cm']:.1%}) | >=30% | "
        f"{checks['surface_pregrasp_inside_20cm_at_least_30_percent']} |",
        f"| Pre-grasp frames inside 12 cm | "
        f"{expert['steps_intrusion_inside_12cm']}/"
        f"{expert['pregrasp_control_steps']} "
        f"({expert['fraction_pregrasp_inside_12cm']:.1%}) | >=5% | "
        f"{checks['surface_pregrasp_inside_12cm_at_least_5_percent']} |",
        "",
        f"The 24-row ledger reconciled as "
        f"{expert['status_counts'].get('success', 0)} successes, "
        f"{expert['status_counts'].get('sampling_failure', 0)} sampling "
        "failure, and "
        f"{expert['status_counts'].get('infrastructure_failure', 0)} "
        "infrastructure failures. All terminal rows count; none was replaced.",
        "",
        "The sole outcome-bearing collision was pilot expert row "
        f"{collision['role_index']} (`{collision['episode_id']}`): it completed "
        "the task but recorded "
        f"{collision['contact_class_totals'].get('hazard_bar', 0)} "
        "`hazard_bar` entries across "
        f"{collision['frames_with_contact'].get('hazard_bar', 0)} frames. "
        "`grasp_target` contact remained exempt and there were "
        f"{collision['contact_class_totals'].get('other_environment', 0)} "
        "`other_environment` entries.",
        "",
        "## What was not run",
        "",
        "Because one applicable prerequisite failed, Gates B/C were not run. "
        "There is no pilot ACT checkpoint, full dataset, frozen surface "
        "encoder, ACT checkpoint, PACT checkpoint, PACT_ZERO evaluation, "
        "Wilson interval, Fisher test, or paired bootstrap result. Running "
        "those steps would violate the frozen stop rule.",
        "",
        "The planned 80-instance × 3-arm schedule is retained only as the "
        "preregistered design. Its checkpoint-bound rows were never "
        "instantiated and no confirmatory outcome was seen. The stopped "
        f"schedule record has SHA-256 `{schedule['schedule_sha256']}`.",
        "",
        "## Interpretation",
        "",
        "The surface-signal guard passed, so this is not a repeat of the fridge "
        "scene's no-signal failure. The environment nevertheless missed its "
        "joint adequacy requirement: it was not sufficiently robustly solvable "
        "by the expert under the fixed seeds and contact endpoint. Consequently "
        "this run cannot establish whether PACT beats ACT or PACT_ZERO.",
        "",
        "## Machine-readable artifacts",
        "",
        "- `diagnostics_output/pact_vs_act/environment_gate.json`",
        "- `diagnostics_output/pact_vs_act/schedule.json`",
        "- `diagnostics_output/pact_vs_act/analysis.json`",
        "- `diagnostics_output/pact_vs_act/final_decision.json`",
        "- `diagnostics_output/pact_vs_act/provenance.json`",
        "",
        f"Analysis SHA-256: `{analysis['analysis_sha256']}`. Final-decision "
        f"SHA-256: `{decision['final_decision_sha256']}`.",
        "",
        "## Decision",
        "",
        "The final line is the exact allowed decision token.",
        "",
        TOKEN,
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--collection-summary", required=True, type=Path)
    parser.add_argument("--schedule-out", required=True, type=Path)
    parser.add_argument("--analysis-out", required=True, type=Path)
    parser.add_argument("--decision-out", required=True, type=Path)
    parser.add_argument("--report-out", required=True, type=Path)
    args = parser.parse_args()

    gate = json.loads(args.gate.read_text())
    preregistration = json.loads(args.preregistration.read_text())
    manifest = json.loads(args.manifest.read_text())
    collection_summary = json.loads(args.collection_summary.read_text())
    schedule, analysis, decision = build_stop_documents(
        gate=gate,
        preregistration=preregistration,
        collection_summary=collection_summary,
        manifest_sha256=manifest["manifest_sha256"],
        gate_sha256=sha256_file(args.gate),
        collection_summary_sha256=sha256_file(args.collection_summary),
    )
    for path, value in (
        (args.schedule_out, schedule),
        (args.analysis_out, analysis),
        (args.decision_out, decision),
    ):
        _write_json(path, value)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(
        render_report(
            gate=gate,
            schedule=schedule,
            analysis=analysis,
            decision=decision,
        )
    )
    print(TOKEN)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
