#!/usr/bin/env python3
"""Finalize a remediation-v2 experiment stopped by the Phase 1 infrastructure gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TOKEN = "PACT_EXPERIMENT_INCOMPLETE"
MANIFEST_ERROR = (
    "FileNotFoundError: [Errno 2] No such file or directory: "
    "'configs/pact_collision_candidate_manifest_v2.json'"
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


def git_head(path: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def tree_record(root: Path, pattern: str) -> dict[str, Any]:
    entries = [
        {"path": str(path.relative_to(root)), "sha256": file_hash(path)}
        for path in sorted(root.glob(pattern))
    ]
    return {
        "root": str(root),
        "files": len(entries),
        "tree_sha256": canonical_hash(entries),
    }


def validate_inputs(
    *,
    manifest: dict[str, Any],
    gate: dict[str, Any],
    training: dict[str, Any],
    pilot_schedule: dict[str, Any],
    execution: dict[str, Any],
    pilot_output_root: Path,
) -> tuple[dict[str, Any], dict[str, int]]:
    if gate.get("decision") != TOKEN:
        raise ValueError(f"gate must decide {TOKEN}")
    if gate.get("manifest_sha256") != manifest.get("manifest_sha256"):
        raise ValueError("gate and candidate manifest differ")
    if gate["surface_observability"]["robust_classification"] != "adequate":
        raise ValueError("this finalizer is for the infrastructure-only v2 stop")
    if gate["act"]["scientific_outcomes"] != 0:
        raise ValueError("unexpected scientific ACT outcomes")
    records = training.get("records", [])
    if len(records) != 1 or records[0]["arm"] != "ACT" or records[0]["seed"] != 1101:
        raise ValueError("expected the single preregistered pilot ACT checkpoint")
    checkpoint = records[0]
    if file_hash(Path(checkpoint["checkpoint"])) != checkpoint["checkpoint_sha256"]:
        raise ValueError("pilot checkpoint hash changed")
    if file_hash(Path(checkpoint["dataset_stats"])) != checkpoint["dataset_stats_sha256"]:
        raise ValueError("pilot dataset-statistics hash changed")

    schedule_payload = dict(pilot_schedule)
    schedule_hash = schedule_payload.pop("schedule_sha256")
    if canonical_hash(schedule_payload) != schedule_hash:
        raise ValueError("pilot schedule self-hash mismatch")
    if (
        schedule_hash != gate.get("pilot_schedule_sha256")
        or schedule_hash != execution.get("schedule_sha256")
    ):
        raise ValueError("pilot schedule identity mismatch")
    if (
        execution.get("expected") != 64
        or execution.get("complete_count") != 0
        or not execution.get("terminal_ledger_reconciled")
        or execution.get("scientific_schedule_reconciled")
    ):
        raise ValueError("unexpected pilot execution state")

    statuses: dict[str, int] = {}
    driver_paths = sorted(pilot_output_root.glob("rows/*/driver_result.json"))
    if len(driver_paths) != 64:
        raise ValueError("pilot driver ledger does not contain 64 rows")
    expected = {row["rollout_id"] for row in pilot_schedule["rows"]}
    observed: set[str] = set()
    matching_errors = 0
    for path in driver_paths:
        driver = json.loads(path.read_text())
        observed.add(driver["rollout_id"])
        status = driver["status"]
        statuses[status] = statuses.get(status, 0) + 1
        log_path = Path(driver["process_log"])
        matching_errors += MANIFEST_ERROR in log_path.read_text(errors="replace")
        if (path.parent / "result.json").exists():
            raise ValueError(f"unexpected scientific result: {path.parent}")
    if observed != expected or statuses != {"invocation_failure": 64}:
        raise ValueError("pilot terminal ledger identity/status mismatch")
    if matching_errors != 64:
        raise ValueError("pilot failures do not share the recorded root cause")
    return checkpoint, statuses


def render_report(
    *,
    gate: dict[str, Any],
    checkpoint: dict[str, Any],
    pilot_schedule: dict[str, Any],
    analysis: dict[str, Any],
    decision: dict[str, Any],
) -> str:
    expert = gate["expert"]
    surface = gate["surface_observability"]
    point = surface["point_estimates"]
    intervals = surface["intervals_95"]
    return "\n".join(
        [
            "# PACT versus ACT final decision",
            "",
            "## Outcome",
            "",
            f"Decision: `{TOKEN}`.",
            "",
            "The remediation-v2 collision environment passed the surface-signal "
            "test and produced enough clean expert demonstrations. The fixed "
            "pilot ACT evaluation nevertheless produced no scientific policy "
            "outcomes because every fresh evaluator subprocess failed before "
            "loading the manifest. The preregistered process-outcome rule makes "
            "those 64 ledger entries terminal, so none was rerun and the "
            "experiment stopped before scale.",
            "",
            "This run therefore does not establish whether PACT beats ACT or "
            "PACT_ZERO.",
            "",
            "## Environment evidence",
            "",
            "| Measure | Result | Frozen threshold |",
            "|---|---:|---:|",
            f"| Active-panel signal inside 20 cm | {point['inside_20cm']:.1%} "
            f"(95% bootstrap {intervals['inside_20cm_episode_cluster_bootstrap'][0]:.1%}"
            f"–{intervals['inside_20cm_episode_cluster_bootstrap'][1]:.1%}) | >=30% |",
            f"| Active-panel signal inside 12 cm | {point['inside_12cm']:.1%} "
            f"(95% bootstrap {intervals['inside_12cm_episode_cluster_bootstrap'][0]:.1%}"
            f"–{intervals['inside_12cm_episode_cluster_bootstrap'][1]:.1%}) | >=5% |",
            f"| Active scientific expert episodes | {surface['active_episodes']}/"
            f"{surface['n']} | >=5/6 |",
            f"| Usable clean demonstrations | "
            f"{expert['usable_clean_demonstrations']}/{expert['attempts']} | >=48 |",
            f"| Expert no-outcome rate | {expert['no_scientific_outcome']}/"
            f"{expert['attempts']} ({expert['no_scientific_outcome_rate']:.1%}) | <5% |",
            "",
            "Surface observability was robustly adequate and passed every "
            "leave-one-episode-out check. Gate A was not applicable because the "
            "predeclared route targets nearby surface geometry, not object position.",
            "",
            "## Pilot policy and terminal evaluation ledger",
            "",
            f"Vision-only ACT seed 1101 completed 2,000 epochs. Its frozen best "
            f"checkpoint was epoch {checkpoint['best_epoch']} with validation loss "
            f"{checkpoint['best_validation_loss']:.6f} and SHA-256 "
            f"`{checkpoint['checkpoint_sha256']}`.",
            "",
            f"The immutable pilot schedule contains 64 ACT rows, uses eight "
            f"workers, and has SHA-256 `{pilot_schedule['schedule_sha256']}`. "
            "All 64 driver entries are `invocation_failure`; there are zero "
            "scientific `result.json` files. Each evaluator was launched from "
            "the ACT submodule while receiving the relative manifest path "
            "`configs/pact_collision_candidate_manifest_v2.json`, which was not "
            "resolvable from that working directory.",
            "",
            "The runner now resolves manifest and output paths before changing "
            "the evaluator working directory, with a focused regression test. "
            "That repair was made only after the terminal ledger existed and "
            "was not used to rerun any row.",
            "",
            "## What was not run",
            "",
            "Gates B and C are inconclusive because 0/64 rows have scientific "
            "ACT outcomes; the frozen minimum was 61. The full train/validation "
            "collection, surface encoder, full ACT and PACT seeds, PACT_ZERO "
            "ablation, 960-rollout confirmatory schedule, Fisher tests, and "
            "instance bootstrap were not run.",
            "",
            "The pilot checkpoint is retained as provenance, but it is not one "
            "of the three final-arm checkpoints requested for a completed "
            "comparison. No claim is made from its validation loss.",
            "",
            "## Machine-readable artifacts",
            "",
            "- `diagnostics_output/pact_vs_act/provenance.json`",
            "- `diagnostics_output/pact_vs_act/schedule.json`",
            "- `diagnostics_output/pact_vs_act/analysis.json`",
            "- `diagnostics_output/pact_vs_act/final_decision.json`",
            "- `diagnostics_output/pact_vs_act/environment_gate.json`",
            "",
            f"Analysis SHA-256: `{analysis['analysis_sha256']}`. Final-decision "
            f"SHA-256: `{decision['final_decision_sha256']}`.",
            "",
            "## Decision",
            "",
            "The last line is the exact allowed decision token.",
            "",
            TOKEN,
        ]
    ) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--training-summary", required=True, type=Path)
    parser.add_argument("--pilot-schedule", required=True, type=Path)
    parser.add_argument("--pilot-output-root", required=True, type=Path)
    parser.add_argument("--environment-gate-out", required=True, type=Path)
    parser.add_argument("--schedule-out", required=True, type=Path)
    parser.add_argument("--analysis-out", required=True, type=Path)
    parser.add_argument("--decision-out", required=True, type=Path)
    parser.add_argument("--report-out", required=True, type=Path)
    parser.add_argument("--provenance-out", required=True, type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    preregistration = json.loads(args.preregistration.read_text())
    gate = json.loads(args.gate.read_text())
    training = json.loads(args.training_summary.read_text())
    pilot_schedule = json.loads(args.pilot_schedule.read_text())
    execution_path = args.pilot_output_root / "execution_summary.json"
    execution = json.loads(execution_path.read_text())
    checkpoint, statuses = validate_inputs(
        manifest=manifest,
        gate=gate,
        training=training,
        pilot_schedule=pilot_schedule,
        execution=execution,
        pilot_output_root=args.pilot_output_root,
    )

    args.environment_gate_out.parent.mkdir(parents=True, exist_ok=True)
    args.environment_gate_out.write_bytes(args.gate.read_bytes())
    gate_file_sha256 = file_hash(args.environment_gate_out)
    pilot_schedule_file_sha256 = file_hash(args.pilot_schedule)
    execution_file_sha256 = file_hash(execution_path)

    schedule: dict[str, Any] = {
        "schema_version": "pact_confirmatory_schedule_not_instantiated_v2",
        "status": "not_instantiated_due_to_phase1_infrastructure_gate",
        "decision": TOKEN,
        "manifest_sha256": manifest["manifest_sha256"],
        "environment_gate_sha256": gate_file_sha256,
        "pilot_schedule": {
            "path": str(args.pilot_schedule),
            "schedule_sha256": pilot_schedule["schedule_sha256"],
            "file_sha256": pilot_schedule_file_sha256,
            "rows": 64,
        },
        "pilot_execution": {
            "output_root": str(args.pilot_output_root),
            "execution_summary_sha256": execution_file_sha256,
            "terminal_ledger_reconciled": True,
            "scientific_schedule_reconciled": False,
            "status_counts": statuses,
        },
        "planned_confirmatory_design": preregistration["confirmatory_design"],
        "rows": [],
        "rollouts_executed": 0,
        "confirmatory_outcomes_seen": False,
        "reason": "Phase 1 had 0/64 scientific ACT rows after terminal invocation failures.",
    }
    schedule["schedule_sha256"] = canonical_hash(schedule)

    analysis: dict[str, Any] = {
        "schema_version": "pact_vs_act_analysis_phase1_incomplete_v2",
        "experiment_stage": "phase1_environment_adequacy",
        "decision": TOKEN,
        "manifest_sha256": manifest["manifest_sha256"],
        "environment_gate_sha256": gate_file_sha256,
        "confirmatory_schedule_sha256": schedule["schedule_sha256"],
        "primary_endpoint": "task success with zero hazard_bar and other_environment contact",
        "target_contact_exempt": True,
        "phase1": gate,
        "pilot_training": {
            "arm": "ACT",
            "seed": checkpoint["seed"],
            "best_epoch": checkpoint["best_epoch"],
            "best_validation_loss": checkpoint["best_validation_loss"],
            "checkpoint_sha256": checkpoint["checkpoint_sha256"],
            "dataset_stats_sha256": checkpoint["dataset_stats_sha256"],
        },
        "pilot_evaluation": {
            "attempts": 64,
            "scientific_outcomes": 0,
            "status_counts": statuses,
            "root_cause": "relative manifest path was invalid from evaluator cwd",
            "rows_rerun": 0,
        },
        "arm_comparison": {
            "status": "not_run_due_to_phase1_infrastructure_failure",
            "ACT": "pilot_only",
            "PACT": "not_trained",
            "PACT_ZERO": "not_evaluated",
            "wilson_intervals": "not_applicable",
            "fisher_exact": "not_applicable",
            "instance_cluster_bootstrap": "not_applicable",
        },
        "valid_preregistered_stop": True,
    }
    analysis["analysis_sha256"] = canonical_hash(analysis)

    decision: dict[str, Any] = {
        "schema_version": "pact_final_decision_v2",
        "decision": TOKEN,
        "experiment_stage": "phase1_environment_adequacy",
        "manifest_sha256": manifest["manifest_sha256"],
        "environment_gate_sha256": gate_file_sha256,
        "analysis_sha256": analysis["analysis_sha256"],
        "schedule_sha256": schedule["schedule_sha256"],
        "reason": "The fixed pilot evaluation has 0/64 scientific ACT outcomes and its terminal rows may not be rerun.",
        "environment_signal_result": "robustly_adequate",
        "arm_comparison_available": False,
        "full_policy_training_performed": False,
        "confirmatory_evaluation_performed": False,
        "pilot_act_checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "final_arm_checkpoint_sha256s": [],
    }
    decision["final_decision_sha256"] = canonical_hash(decision)

    write_json(args.schedule_out, schedule)
    write_json(args.analysis_out, analysis)
    write_json(args.decision_out, decision)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(
        render_report(
            gate=gate,
            checkpoint=checkpoint,
            pilot_schedule=pilot_schedule,
            analysis=analysis,
            decision=decision,
        )
    )

    small_paths = {
        "candidate_manifest": args.manifest,
        "environment_preregistration": args.preregistration,
        "environment_gate": args.environment_gate_out,
        "pilot_training_summary": args.training_summary,
        "pilot_schedule": args.pilot_schedule,
        "confirmatory_schedule": args.schedule_out,
        "analysis": args.analysis_out,
        "final_decision": args.decision_out,
        "environment_report": ROOT / "docs" / "PACT_ENVIRONMENT_ADEQUACY.md",
        "final_report": args.report_out,
    }
    provenance = {
        "schema_version": "pact_vs_act_provenance_v2",
        "experiment_stage": "stopped_at_phase1_infrastructure_gate",
        "decision": TOKEN,
        "source_commits": {
            "root": git_head(ROOT),
            "act": git_head(ROOT / "submodules" / "act"),
            "molmospaces": git_head(ROOT / "submodules" / "molmospaces"),
        },
        "small_artifacts": {
            name: {"path": str(path), "sha256": file_hash(path)}
            for name, path in small_paths.items()
        },
        "pilot_checkpoint": {
            "path": checkpoint["checkpoint"],
            "sha256": checkpoint["checkpoint_sha256"],
            "observed_sha256": file_hash(Path(checkpoint["checkpoint"])),
            "dataset_stats_path": checkpoint["dataset_stats"],
            "dataset_stats_sha256": checkpoint["dataset_stats_sha256"],
            "best_epoch": checkpoint["best_epoch"],
            "best_validation_loss": checkpoint["best_validation_loss"],
        },
        "final_arm_checkpoints": [],
        "surface_encoder": {
            "status": "not_trained_due_to_phase1_gate",
            "path": None,
            "sha256": None,
        },
        "pilot_execution": {
            "output_root": str(args.pilot_output_root),
            "execution_summary": {
                "path": str(execution_path),
                "sha256": execution_file_sha256,
            },
            "driver_ledger": tree_record(
                args.pilot_output_root, "rows/*/driver_result.json"
            ),
            "process_logs": tree_record(args.pilot_output_root, "rows/*/process.log"),
            "scientific_result_files": 0,
            "rows_rerun": 0,
        },
        "protected_chain": {
            "modified_by_pact_work": False,
            "preexisting_worktree_changes_preserved": True,
            "used_as_pact_evidence": False,
            "confirmatory41_touched_by_pact_work": False,
        },
    }
    write_json(args.provenance_out, provenance)
    print(TOKEN)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
