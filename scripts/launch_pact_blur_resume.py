#!/usr/bin/env python3
"""Resume the frozen blur schedule after an audited all-inflight abort."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from run_pact_confirmatory_schedule import canonical_hash, write_json_atomic
from run_pact_frontend_screen_supervisor import process_identity, utc_now


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/root/act_retrain_venv/bin/python")
SUPERVISOR = ROOT / "scripts/run_pact_blur_supervisor.py"


def validate_hash(document: dict, key: str, label: str) -> str:
    payload = dict(document)
    observed = payload.pop(key, None)
    if observed != canonical_hash(payload):
        raise ValueError(f"{label} self-hash mismatch")
    return str(observed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", required=True, type=Path)
    parser.add_argument("--dispatch", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--recovery-event", required=True, type=Path)
    parser.add_argument("--execution-amendment", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    prior_receipts = sorted(output_root.glob("resume_launcher_receipt*.json"))
    resume_attempt_index = len(prior_receipts)
    receipt_path = output_root / (
        "resume_launcher_receipt.json"
        if resume_attempt_index == 0
        else f"resume_launcher_receipt_{resume_attempt_index:03d}.json"
    )
    schedule = json.loads(args.schedule.read_text())
    schedule_sha = validate_hash(schedule, "schedule_sha256", "schedule")
    dispatch = json.loads(args.dispatch.read_text())
    dispatch_sha = validate_hash(dispatch, "dispatch_contract_sha256", "dispatch")
    recovery = json.loads(args.recovery_event.read_text())
    recovery_sha = validate_hash(recovery, "recovery_event_sha256", "recovery event")
    amendment = json.loads(args.execution_amendment.read_text())
    amendment_sha = validate_hash(amendment, "amendment_sha256", "execution amendment")
    state = json.loads((output_root / "supervisor_state.json").read_text())
    state_sha = validate_hash(state, "state_sha256", "aborted supervisor state")
    ledger = json.loads((output_root / "completion_ledger.json").read_text())
    validate_hash(ledger, "completion_ledger_sha256", "completion ledger")
    if (
        schedule_sha != dispatch["scientific_schedule"]["schedule_sha256"]
        or schedule_sha != recovery["schedule_sha256"]
        or schedule_sha != amendment["schedule_sha256"]
        or dispatch_sha != amendment["dispatch_contract_sha256"]
        or state.get("status") != "aborted"
        or state_sha != recovery["source_supervisor_state_sha256"]
        or len(ledger["completions"]) != recovery["completed_rows_preserved"]
        or len(ledger["completions"]) != 33
        or recovery.get("all_inflight_rows_rerun") is not True
        or recovery.get("active_cohort_size") != 12
        or amendment.get("fixed_worker_count") != 12
    ):
        raise ValueError("blur resume bindings changed")
    if process_identity(int(state["supervisor_pid"])) is not None:
        raise ValueError("aborted supervisor is unexpectedly still alive")
    if prior_receipts:
        previous_receipt = json.loads(prior_receipts[-1].read_text())
        validate_hash(previous_receipt, "resume_launcher_sha256", "prior resume receipt")
        old_compactor_pid = int(previous_receipt["replacement_compactor_pid"])
    else:
        original_receipt = json.loads(
            (output_root / "full_launcher_receipt.json").read_text()
        )
        old_compactor_pid = int(original_receipt["compactor_pid"])
    old_compactor = process_identity(old_compactor_pid)
    if old_compactor is not None:
        command_line = Path(f"/proc/{old_compactor_pid}/cmdline").read_bytes()
        if b"compact_pact_geometry_storage.py" not in command_line:
            raise ValueError("recorded compactor PID belongs to another process")
        os.kill(old_compactor_pid, signal.SIGTERM)
        deadline = time.monotonic() + 10.0
        while process_identity(old_compactor_pid) is not None and time.monotonic() < deadline:
            time.sleep(0.1)
        if process_identity(old_compactor_pid) is not None:
            raise RuntimeError("old compactor did not stop after SIGTERM")
    command = [
        "/usr/bin/setsid",
        "/usr/bin/nohup",
        str(PYTHON),
        str(SUPERVISOR),
        "--schedule",
        str(args.schedule.resolve()),
        "--dispatch-contract",
        str(args.dispatch.resolve()),
        "--manifest",
        str(args.manifest.resolve()),
        "--output-root",
        str(output_root),
        "--mode",
        "full",
        "--resume-recovery-event",
        str(args.recovery_event.resolve()),
    ]
    environment = dict(os.environ)
    environment.update(
        {
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
            "PYTHONUNBUFFERED": "1",
            "MLSPACES_ASSETS_DIR": str(ROOT / "assets"),
            "PYTHONPATH": f"{ROOT / 'submodules/molmospaces'}:{ROOT / 'submodules/act'}:{ROOT / 'scripts'}",
            "PACT_CONTACT_AUDIT_SUMMARY_ONLY": "1",
        }
    )
    log_path = output_root / "resume_supervisor.log"
    with log_path.open("ab") as stream:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=environment,
            start_new_session=False,
        )
    compactor_log = output_root / "resume_storage_compactor.log"
    with compactor_log.open("ab") as stream:
        compactor = subprocess.Popen(
            [
                "/usr/bin/setsid",
                "/usr/bin/nohup",
                str(PYTHON),
                str(ROOT / "scripts/compact_pact_geometry_storage.py"),
                "--schedule",
                str(args.schedule.resolve()),
                "--dispatch",
                str(args.dispatch.resolve()),
                "--output-root",
                str(output_root),
            ],
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=environment,
            start_new_session=False,
        )
    receipt = {
        "schema_version": "pact_blur_sweep_resume_launcher_v1",
        "resume_attempt_index": resume_attempt_index,
        "supervisor_pid": process.pid,
        "old_compactor_pid": old_compactor_pid,
        "old_compactor_was_alive_and_stopped": old_compactor is not None,
        "replacement_compactor_pid": compactor.pid,
        "schedule_sha256": schedule_sha,
        "dispatch_contract_sha256": dispatch_sha,
        "recovery_event_sha256": recovery_sha,
        "execution_amendment_sha256": amendment_sha,
        "completed_rows_preserved": 33,
        "recovery_rows": 12,
        "all_inflight_rows_rerun": True,
        "fixed_worker_count": 12,
        "scientific_schedule_changed": False,
        "endpoint_fields_read": False,
        "command": command,
        "log": str(log_path),
        "launched_utc": utc_now(),
    }
    receipt["resume_launcher_sha256"] = canonical_hash(receipt)
    write_json_atomic(receipt_path, receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
