#!/usr/bin/env python3
"""Finalize the frozen seed replication after detached execution reconciles."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/root/act_retrain_venv/bin/python")


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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def git_commit(path: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()


def wait_for(path: Path, *, poll_seconds: float) -> dict[str, Any]:
    while not path.exists():
        time.sleep(poll_seconds)
    return json.loads(path.read_text())


def run_throughput(output_root: Path) -> None:
    path = output_root / "throughput_first_20_minutes.json"
    if path.exists():
        return
    receipt = json.loads((output_root / "full_launcher_receipt.json").read_text())
    cutoff = parse_utc(receipt["launched_utc"]) + timedelta(minutes=20)
    remaining = (cutoff - datetime.now(timezone.utc)).total_seconds()
    if remaining > 0:
        time.sleep(remaining + 1)
    subprocess.run(
        [
            str(PYTHON),
            str(ROOT / "scripts/measure_pact_frontend_screen_throughput.py"),
            "--output-root",
            str(output_root),
            "--measurement-minutes",
            "20",
        ],
        cwd=ROOT,
        check=True,
    )


def run_analysis(schedule: Path, output_root: Path) -> None:
    subprocess.run(
        [
            str(PYTHON),
            str(ROOT / "scripts/analyze_pact_seed_replication.py"),
            "--schedule",
            str(schedule),
            "--output-root",
            str(output_root),
            "--analysis-out",
            str(ROOT / "diagnostics_output/pact_seed_replication/analysis.json"),
            "--decision-out",
            str(ROOT / "diagnostics_output/pact_seed_replication/final_decision.json"),
            "--report-out",
            str(ROOT / "docs/PACT_SEED_REPLICATION_DECISION.md"),
        ],
        cwd=ROOT,
        check=True,
    )


def write_provenance(*, schedule_path: Path, dispatch_path: Path, output_root: Path) -> None:
    schedule = json.loads(schedule_path.read_text())
    dispatch = json.loads(dispatch_path.read_text())
    training_path = ROOT / "diagnostics_output/pact_seed_replication/policy_training.json"
    analysis_path = ROOT / "diagnostics_output/pact_seed_replication/analysis.json"
    decision_path = ROOT / "diagnostics_output/pact_seed_replication/final_decision.json"
    report_path = ROOT / "docs/PACT_SEED_REPLICATION_DECISION.md"
    ledger = json.loads((output_root / "completion_ledger.json").read_text())
    execution = json.loads((output_root / "full_execution_summary.json").read_text())
    storage = json.loads((output_root / "storage_compaction_summary.json").read_text())
    decision = json.loads(decision_path.read_text())
    records = {
        label: {"path": str(path), "sha256": file_hash(path)}
        for label, path in (
            ("schedule", schedule_path),
            ("dispatch", dispatch_path),
            ("training", training_path),
            ("analysis", analysis_path),
            ("final_decision", decision_path),
            ("decision_report", report_path),
            ("launch_smoke", output_root / "launch_smoke.json"),
            ("detachment_proof", output_root / "detachment_proof.json"),
            (
                "throughput_first_20_minutes",
                output_root / "throughput_first_20_minutes.json",
            ),
            ("completion_ledger", output_root / "completion_ledger.json"),
            ("full_execution", output_root / "full_execution_summary.json"),
            (
                "storage_compaction",
                output_root / "storage_compaction_summary.json",
            ),
        )
    }
    document: dict[str, Any] = {
        "schema_version": "pact_seed_replication_provenance_v1",
        "generated_utc": utc_now(),
        "root_commit_at_finalization": git_commit(ROOT),
        "act_submodule_commit": git_commit(ROOT / "submodules/act"),
        "schedule_sha256": schedule["schedule_sha256"],
        "dispatch_contract_sha256": dispatch["dispatch_contract_sha256"],
        "output_root": str(output_root),
        "checkpoint_records": dispatch["frozen_inputs"]["checkpoints"],
        "encoder_records": dispatch["frozen_inputs"]["surface_encoders"],
        "expected_rollouts": 120,
        "completion_records": len(ledger["completions"]),
        "scientific_schedule_reconciled": execution["scientific_schedule_reconciled"],
        "losslessly_compacted_rows": storage["compacted_count"],
        "raw_intact_schedule_indices": storage["excluded_intact_schedule_indices"],
        "decision": decision["decision"],
        "frozen_artifacts": records,
        "training_preflight_failures": [
            {
                "log": "/root/pact_seed_replication_artifacts/training.log",
                "reason": "parent audit shell false-positive protected-process match",
                "model_output_created": False,
            },
            {
                "log": "/root/pact_seed_replication_artifacts/training_run.log",
                "reason": "relative split path rejected before data loading",
                "model_output_created": False,
            },
        ],
        "storage_cleanup": {
            "completed_act3102_redundant_training_state_removed_bytes": 11928321328,
            "completed_pact3102_redundant_training_state_removed_bytes": 3567607585,
            "required_policy_best_checkpoints_retained": True,
        },
        "tests": {
            "scoped_repository_tests": "763 passed, 52 warnings",
            "unrestricted_collection": (
                "13 pre-existing MolmoSpaces collection errors from duplicate "
                "module names, optional dependencies, missing legacy modules, "
                "and absent scene assets"
            ),
        },
        "rollout_payloads_committed": False,
        "weights_committed": False,
        "pushed": False,
    }
    if (
        document["completion_records"] != 120
        or document["scientific_schedule_reconciled"] is not True
        or document["losslessly_compacted_rows"] != 118
    ):
        raise RuntimeError("refusing provenance for unreconciled execution")
    document["provenance_sha256"] = canonical_hash(document)
    path = ROOT / "diagnostics_output/pact_seed_replication/provenance.json"
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", required=True, type=Path)
    parser.add_argument("--dispatch", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        raise SystemExit("poll interval must be positive")
    schedule = args.schedule.resolve()
    dispatch = args.dispatch.resolve()
    output_root = args.output_root.resolve()
    run_throughput(output_root)
    execution = wait_for(
        output_root / "full_execution_summary.json",
        poll_seconds=args.poll_seconds,
    )
    if execution.get("scientific_schedule_reconciled") is not True:
        run_analysis(schedule, output_root)
        return 2
    storage = wait_for(
        output_root / "storage_compaction_summary.json",
        poll_seconds=args.poll_seconds,
    )
    if storage.get("compacted_count") != 118:
        raise SystemExit("storage compaction did not reconcile 118 rows")
    run_analysis(schedule, output_root)
    write_provenance(
        schedule_path=schedule,
        dispatch_path=dispatch,
        output_root=output_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
