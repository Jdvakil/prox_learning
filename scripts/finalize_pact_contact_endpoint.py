#!/usr/bin/env python3
"""Finalize the frozen contact experiment after execution and storage reconcile."""

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
TOKENS = {
    "CONTACT_REDUCTION_ESTABLISHED",
    "CONTACT_REDUCTION_WITH_TASK_BENEFIT",
    "CONTACT_REDUCTION_SUBSET_ONLY",
    "NO_CONTACT_REDUCTION",
    "CONTACT_INCREASE",
    "CONTACT_EXPERIMENT_INCOMPLETE",
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


def parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def git_commit(path: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()


def wait_for(path: Path, poll_seconds: float) -> dict[str, Any]:
    while not path.exists():
        time.sleep(poll_seconds)
    return json.loads(path.read_text())


def run_throughput(schedule: Path, output_root: Path) -> None:
    output = output_root / "throughput_first_20_minutes.json"
    if output.exists():
        return
    receipt = json.loads((output_root / "full_launcher_receipt.json").read_text())
    cutoff = parse_utc(receipt["launched_utc"]) + timedelta(minutes=20)
    remaining = (cutoff - datetime.now(timezone.utc)).total_seconds()
    if remaining > 0:
        time.sleep(remaining + 1)
    subprocess.run(
        [
            str(PYTHON),
            str(ROOT / "scripts/measure_pact_contact_throughput.py"),
            "--schedule",
            str(schedule),
            "--output-root",
            str(output_root),
            "--measurement-minutes",
            "20",
        ],
        cwd=ROOT,
        check=True,
    )


def run_analysis(schedule: Path, output_root: Path) -> tuple[Path, Path, Path]:
    analysis = ROOT / "diagnostics_output/pact_contact_endpoint/analysis.json"
    decision = ROOT / "diagnostics_output/pact_contact_endpoint/final_decision.json"
    report = ROOT / "docs/PACT_CONTACT_ENDPOINT_DECISION.md"
    subprocess.run(
        [
            str(PYTHON),
            str(ROOT / "scripts/analyze_pact_contact_endpoint.py"),
            "--schedule",
            str(schedule),
            "--output-root",
            str(output_root),
            "--analysis-output",
            str(analysis),
            "--decision-output",
            str(decision),
            "--report-output",
            str(report),
        ],
        cwd=ROOT,
        check=True,
    )
    return analysis, decision, report


def validate_self_hash(document: dict[str, Any], key: str, label: str) -> None:
    payload = dict(document)
    observed = payload.pop(key, None)
    if observed != canonical_hash(payload):
        raise ValueError(f"{label} self-hash mismatch")


def write_provenance(
    *,
    schedule_path: Path,
    dispatch_path: Path,
    manifest_path: Path,
    storage_amendment_path: Path,
    output_root: Path,
    analysis_path: Path,
    decision_path: Path,
    report_path: Path,
) -> None:
    schedule = json.loads(schedule_path.read_text())
    dispatch = json.loads(dispatch_path.read_text())
    amendment = json.loads(storage_amendment_path.read_text())
    analysis = json.loads(analysis_path.read_text())
    decision = json.loads(decision_path.read_text())
    ledger = json.loads((output_root / "completion_ledger.json").read_text())
    execution = json.loads((output_root / "full_execution_summary.json").read_text())
    storage = json.loads((output_root / "storage_compaction_summary.json").read_text())
    validate_self_hash(schedule, "schedule_sha256", "schedule")
    validate_self_hash(dispatch, "dispatch_contract_sha256", "dispatch")
    validate_self_hash(amendment, "storage_amendment_sha256", "storage amendment")
    validate_self_hash(analysis, "analysis_sha256", "analysis")
    validate_self_hash(decision, "final_decision_sha256", "decision")
    validate_self_hash(storage, "storage_compaction_summary_sha256", "storage summary")
    token = decision["decision"]
    if token not in TOKENS or decision["analysis_sha256"] != analysis["analysis_sha256"]:
        raise ValueError("contact decision identity mismatch")
    report_lines = [line.strip() for line in report_path.read_text().splitlines() if line.strip()]
    if not report_lines or report_lines[-1] != token:
        raise ValueError("contact report does not end in its exact token")
    if (
        len(ledger.get("completions", [])) != 1200
        or execution.get("scientific_schedule_reconciled") is not True
        or storage.get("compacted_count") != 1198
        or storage.get("excluded_intact_schedule_indices") != [0, 1199]
        or analysis["reconciliation"].get("reconciled") is not True
    ):
        raise ValueError("contact experiment did not fully reconcile")
    completion_by_id = {item["rollout_id"]: item for item in ledger["completions"]}
    for row in schedule["rows"]:
        index = int(row["schedule_index"])
        row_dir = output_root / row["output_relpath"]
        result_path = row_dir / "result.json"
        driver_path = row_dir / "driver_result.json"
        completion = completion_by_id.get(row["rollout_id"])
        if completion is None or completion["schedule_index"] != index:
            raise ValueError(f"completion ledger identity mismatch at row {index}")
        if file_hash(driver_path) != completion["driver_result_sha256"]:
            raise ValueError(f"driver result changed at row {index}")
        archive_path = row_dir / "storage_archive.json"
        if index in (0, 1199):
            result = json.loads(result_path.read_text())
            trajectory = Path(result["trajectory_path"])
            if (
                not trajectory.is_file()
                or archive_path.exists()
                or file_hash(result_path) != completion["result_sha256"]
            ):
                raise ValueError(f"required intact row {index} is not intact")
            continue
        archive = json.loads(archive_path.read_text())
        validate_self_hash(archive, "storage_archive_sha256", f"row {index} storage")
        if (
            archive.get("status") != "complete"
            or archive.get("schedule_index") != index
            or archive.get("rollout_id") != row["rollout_id"]
            or archive.get("schedule_row_sha256") != row["schedule_row_sha256"]
            or archive["original_result"]["sha256"] != completion["result_sha256"]
            or archive["compact_result_sha256"] != file_hash(result_path)
            or any(Path(item["path"]).exists() for item in archive["deleted_payloads"])
        ):
            raise ValueError(f"compacted row {index} did not preserve its boundary")
    records = {
        label: {"path": str(path.resolve()), "sha256": file_hash(path)}
        for label, path in {
            "preregistration": ROOT / "configs/pact_contact_endpoint_preregistration_v1.json",
            "preregistration_narrative": ROOT / "docs/PACT_CONTACT_ENDPOINT_PREREGISTRATION.md",
            "worker_amendment": ROOT
            / "diagnostics_output/pact_contact_endpoint/worker_amendment_v1.json",
            "manifest": manifest_path,
            "policy_training": ROOT
            / "diagnostics_output/pact_contact_endpoint/policy_training.json",
            "occlusion_subset": ROOT
            / "diagnostics_output/pact_contact_endpoint/occlusion_subset.json",
            "power": ROOT / "diagnostics_output/pact_contact_endpoint/power.json",
            "token_plan": ROOT / "diagnostics_output/pact_contact_endpoint/token_plan.json",
            "schedule": schedule_path,
            "dispatch": dispatch_path,
            "storage_amendment": storage_amendment_path,
            "analysis": analysis_path,
            "final_decision": decision_path,
            "decision_report": report_path,
            "launch_smoke": output_root / "launch_smoke.json",
            "detachment_proof": output_root / "detachment_proof.json",
            "full_launcher_receipt": output_root / "full_launcher_receipt.json",
            "full_stack_receipt": output_root / "full_stack_receipt.json",
            "gpu_memory_first_minutes": output_root / "gpu_memory_first_minutes.json",
            "throughput": output_root / "throughput_first_20_minutes.json",
            "completion_ledger": output_root / "completion_ledger.json",
            "full_execution": output_root / "full_execution_summary.json",
            "storage_compaction": output_root / "storage_compaction_summary.json",
        }.items()
    }
    document: dict[str, Any] = {
        "schema_version": "pact_contact_endpoint_provenance_v1",
        "generated_utc": utc_now(),
        "decision": token,
        "root_commit_at_finalization": git_commit(ROOT),
        "act_submodule_commit": git_commit(ROOT / "submodules/act"),
        "molmospaces_submodule_commit": git_commit(ROOT / "submodules/molmospaces"),
        "schedule_sha256": schedule["schedule_sha256"],
        "dispatch_contract_sha256": dispatch["dispatch_contract_sha256"],
        "analysis_sha256": analysis["analysis_sha256"],
        "final_decision_sha256": decision["final_decision_sha256"],
        "expected_rollouts": 1200,
        "completion_records": 1200,
        "scientific_schedule_reconciled": True,
        "storage": {
            "summary_only_contact_instrumentation": True,
            "endpoint_complete_compacted_rows": 1198,
            "raw_intact_schedule_indices": [0, 1199],
            "deleted_payload_bytes_recoverable": False,
            "outcome_based_selection": False,
        },
        "checkpoint_records": dispatch["frozen_inputs"]["checkpoints"],
        "encoder_records": dispatch["frozen_inputs"]["surface_encoders"],
        "frozen_artifacts": records,
        "rollout_payloads_committed": False,
        "weights_committed": False,
        "pushed": False,
    }
    document["provenance_sha256"] = canonical_hash(document)
    path = ROOT / "diagnostics_output/pact_contact_endpoint/provenance.json"
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", required=True, type=Path)
    parser.add_argument("--dispatch", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--storage-amendment", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        raise SystemExit("poll interval must be positive")
    schedule = args.schedule.resolve()
    output_root = args.output_root.resolve()
    run_throughput(schedule, output_root)
    execution = wait_for(output_root / "full_execution_summary.json", args.poll_seconds)
    if execution.get("scientific_schedule_reconciled") is not True:
        run_analysis(schedule, output_root)
        return 2
    storage = wait_for(output_root / "storage_compaction_summary.json", args.poll_seconds)
    if storage.get("compacted_count") != 1198:
        raise SystemExit("contact storage compaction did not reconcile 1198 rows")
    analysis, decision, report = run_analysis(schedule, output_root)
    write_provenance(
        schedule_path=schedule,
        dispatch_path=args.dispatch.resolve(),
        manifest_path=args.manifest.resolve(),
        storage_amendment_path=args.storage_amendment.resolve(),
        output_root=output_root,
        analysis_path=analysis,
        decision_path=decision,
        report_path=report,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
