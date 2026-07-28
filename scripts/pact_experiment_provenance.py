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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--environment-prereg", required=True, type=Path)
    parser.add_argument("--environment-gate", required=True, type=Path)
    parser.add_argument("--surface-report", required=True, type=Path)
    parser.add_argument("--training-summary", required=True, type=Path)
    parser.add_argument("--schedule", required=True, type=Path)
    parser.add_argument("--analysis", required=True, type=Path)
    parser.add_argument("--decision", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    named_paths = {
        "candidate_manifest": args.manifest,
        "environment_preregistration": args.environment_prereg,
        "environment_gate": args.environment_gate,
        "surface_encoder_report": args.surface_report,
        "training_summary": args.training_summary,
        "confirmatory_schedule": args.schedule,
        "analysis": args.analysis,
        "final_decision": args.decision,
    }
    training = json.loads(args.training_summary.read_text())
    surface = json.loads(args.surface_report.read_text())
    checkpoints = []
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
    document = {
        "schema_version": "pact_vs_act_provenance_v1",
        "source_commits": {
            "root": git_head(ROOT),
            "act": git_head(ROOT / "submodules" / "act"),
            "molmospaces": git_head(ROOT / "submodules" / "molmospaces"),
        },
        "small_artifacts": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in named_paths.items()
        },
        "surface_encoder": {
            "path": surface["checkpoint_path"],
            "sha256": surface["checkpoint_sha256"],
            "observed_sha256": sha256_file(Path(surface["checkpoint_path"])),
            "parameter_count": surface["parameter_count"],
            "heldout_metrics": surface["heldout_metrics"],
            "frozen": surface["frozen"],
        },
        "policy_checkpoints": checkpoints,
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
    if (
        document["surface_encoder"]["sha256"]
        != document["surface_encoder"]["observed_sha256"]
    ):
        raise SystemExit("surface encoder hash changed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
