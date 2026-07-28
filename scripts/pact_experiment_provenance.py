#!/usr/bin/env python3
"""Assemble immutable provenance for the completed PACT experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
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


def artifact_record(path: Path) -> dict:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--environment-prereg", required=True, type=Path)
    parser.add_argument("--environment-gate", required=True, type=Path)
    parser.add_argument("--surface-report", type=Path)
    parser.add_argument("--training-summary", type=Path)
    parser.add_argument("--schedule", required=True, type=Path)
    parser.add_argument("--analysis", required=True, type=Path)
    parser.add_argument("--decision", required=True, type=Path)
    parser.add_argument("--pilot-collection", type=Path)
    parser.add_argument("--environment-report", type=Path)
    parser.add_argument("--final-report", type=Path)
    parser.add_argument("--development-screen", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    named_paths = {
        "candidate_manifest": args.manifest,
        "environment_preregistration": args.environment_prereg,
        "environment_gate": args.environment_gate,
        "confirmatory_schedule": args.schedule,
        "analysis": args.analysis,
        "final_decision": args.decision,
    }
    for name, path in (
        ("surface_encoder_report", args.surface_report),
        ("training_summary", args.training_summary),
        ("environment_report", args.environment_report),
        ("final_report", args.final_report),
        ("development_screen", args.development_screen),
    ):
        if path is not None:
            named_paths[name] = path

    checkpoints = []
    if args.training_summary is not None:
        training = json.loads(args.training_summary.read_text())
        for record in training["records"]:
            checkpoints.append(
                {
                    "arm": record["arm"],
                    "seed": record["seed"],
                    "path": record["checkpoint"],
                    "sha256": record["checkpoint_sha256"],
                    "observed_sha256": sha256_file(Path(record["checkpoint"])),
                    "best_epoch": record["best_epoch"],
                    "best_validation_loss": record["best_validation_loss"],
                    "dataset_stats_sha256": record["dataset_stats_sha256"],
                    "pact_zero_alias": record.get("pact_zero_checkpoint_alias"),
                }
            )

    surface_record: dict
    if args.surface_report is not None:
        surface = json.loads(args.surface_report.read_text())
        surface_record = {
            "status": "trained_and_frozen",
            "path": surface["checkpoint_path"],
            "sha256": surface["checkpoint_sha256"],
            "observed_sha256": sha256_file(Path(surface["checkpoint_path"])),
            "parameter_count": surface["parameter_count"],
            "heldout_metrics": surface["heldout_metrics"],
            "frozen": surface["frozen"],
        }
    else:
        surface_record = {
            "status": "not_trained_due_to_phase1_environment_gate",
            "path": None,
            "sha256": None,
            "frozen": False,
        }

    pilot_collection = None
    if args.pilot_collection is not None:
        summary_path = args.pilot_collection / "pilot_train_summary.json"
        summary = json.loads(summary_path.read_text())
        manifest = json.loads(args.manifest.read_text())
        rows_by_id = {
            row["episode_id"]: row
            for row in manifest["rows"]
            if row["role"] == "pilot_train"
        }
        expected_pilot_rows = int(manifest["role_counts"]["pilot_train"])
        if not summary.get("complete") or len(rows_by_id) != expected_pilot_rows:
            raise SystemExit("pilot expert collection is not complete")
        row_records = []
        for result_path in sorted(
            args.pilot_collection.glob("rows/*/result.json")
        ):
            result = json.loads(result_path.read_text())
            row = rows_by_id.get(result.get("episode_id"))
            if row is None or result.get("row_sha256") != row["row_sha256"]:
                raise SystemExit(f"pilot row identity mismatch: {result_path}")
            artifacts = [
                artifact_record(path)
                for path in sorted(result_path.parent.iterdir())
                if path.is_file()
            ]
            row_records.append(
                {
                    "episode_id": result["episode_id"],
                    "role_index": result["role_index"],
                    "status": result["status"],
                    "task_success": bool(result.get("task_success", False)),
                    "collision_free_task_success": bool(
                        result.get("collision_free_task_success", False)
                    ),
                    "root_source_commit": result.get("root_source_commit"),
                    "molmospaces_source_commit": result.get(
                        "molmospaces_source_commit"
                    ),
                    "artifacts": artifacts,
                }
            )
        if len(row_records) != expected_pilot_rows:
            raise SystemExit(
                f"expected {expected_pilot_rows} terminal pilot row results"
            )
        pilot_collection = {
            "path": str(args.pilot_collection),
            "summary": artifact_record(summary_path),
            "status_counts": summary["status_counts"],
            "rows": sorted(row_records, key=lambda row: row["role_index"]),
            "artifact_file_count": sum(
                len(row["artifacts"]) for row in row_records
            ),
            "artifact_size_bytes": sum(
                artifact["size_bytes"]
                for row in row_records
                for artifact in row["artifacts"]
            ),
        }

    decision = json.loads(args.decision.read_text())
    environment_gate = json.loads(args.environment_gate.read_text())
    phase1_adequate = (
        environment_gate.get("decision") == "PACT_ENVIRONMENT_ADEQUATE"
    )
    if not phase1_adequate and (
        args.surface_report is not None or args.training_summary is not None
    ):
        raise SystemExit(
            "a non-adequate Phase 1 provenance must not claim trained artifacts"
        )
    document = {
        "schema_version": "pact_vs_act_provenance_v2",
        "experiment_stage": (
            "stopped_at_phase1_environment_gate"
            if not phase1_adequate
            else (
                "confirmatory_complete"
                if decision.get("decision") != "PACT_EXPERIMENT_INCOMPLETE"
                else "confirmatory_incomplete"
            )
        ),
        "source_commits": {
            "root": git_head(ROOT),
            "act": git_head(ROOT / "submodules" / "act"),
            "molmospaces": git_head(ROOT / "submodules" / "molmospaces"),
        },
        "small_artifacts": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in named_paths.items()
        },
        "surface_encoder": surface_record,
        "policy_checkpoints": checkpoints,
        "policy_checkpoint_status": (
            "not_trained_due_to_phase1_environment_gate"
            if not phase1_adequate
            else (
                "trained"
                if args.training_summary is not None
                else "not_trained_experiment_incomplete"
            )
        ),
        "pilot_collection": pilot_collection,
        "protected_chain": {
            "modified_by_pact_work": False,
            "preexisting_worktree_changes_preserved": True,
            "used_as_pact_evidence": False,
            "confirmatory41_touched_by_pact_work": False,
        },
    }
    if any(
        item["sha256"] != item["observed_sha256"] for item in checkpoints
    ):
        raise SystemExit("one or more policy checkpoint hashes changed")
    if args.surface_report is not None and (
        surface_record["sha256"] != surface_record["observed_sha256"]
    ):
        raise SystemExit("surface encoder hash changed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
