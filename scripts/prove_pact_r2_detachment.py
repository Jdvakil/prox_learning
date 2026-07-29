#!/usr/bin/env python3
"""Kill the smoke launching shell and prove the detached R2 pool survives."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from run_pact_confirmatory_schedule import (
    canonical_hash,
    load_dispatch_contract,
    sha256_file,
    validate_launch_smoke,
    write_json_atomic,
)

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/root/act_retrain_venv/bin/python")
LAUNCHER = ROOT / "scripts/launch_pact_r2_detached.py"
POLL_SECONDS = 0.5
SURVIVAL_SECONDS = 8.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def pid_alive(pid: int) -> bool:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().split()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return False
    return len(fields) > 2 and fields[2] != "Z"


def process_identity(pid: int) -> dict[str, Any] | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().split()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None
    return {
        "pid": pid,
        "ppid": int(fields[3]),
        "process_group_id": int(fields[4]),
        "session_id": int(fields[5]),
        "state": fields[2],
        "start_time_ticks": int(fields[21]),
    }


def wait_for(
    predicate,
    *,
    timeout_seconds: float,
    description: str,
) -> Any:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(POLL_SECONDS)
    raise TimeoutError(f"timed out waiting for {description}")


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", required=True, type=Path)
    parser.add_argument("--dispatch-contract", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--startup-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--smoke-timeout-seconds", type=float, default=7200.0)
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    proof_path = output_root / "detachment_proof.json"
    if proof_path.exists():
        raise SystemExit("detachment proof already exists; smoke is not rerun")
    schedule = json.loads(args.schedule.read_text())
    contract = load_dispatch_contract(
        args.dispatch_contract,
        schedule,
        manifest_path=args.manifest,
        output_root=output_root,
    )
    proof_spec = contract.get("detachment_proof", {})
    if (
        proof_spec.get("required_before_full_dispatch") is not True
        or proof_spec.get("kill_launching_shell_during_smoke") is not True
    ):
        raise SystemExit("dispatch contract does not require the R2 proof")
    smoke = contract["launch_smoke"]
    smoke_dir = output_root / smoke["output_relpath"]
    if (
        (output_root / smoke["required_artifact"]).exists()
        or (smoke_dir / "result.json").exists()
    ):
        raise SystemExit("smoke already has a result; proof launch refused")

    launcher_command = [
        str(PYTHON),
        str(LAUNCHER),
        "--schedule",
        str(args.schedule.resolve()),
        "--dispatch-contract",
        str(args.dispatch_contract.resolve()),
        "--manifest",
        str(args.manifest.resolve()),
        "--output-root",
        str(output_root),
        "--mode",
        "smoke",
    ]
    shell_stdout = output_root / "smoke_launching_shell.log"
    shell_script = (
        shlex.join(launcher_command) + "\nwhile :; do sleep 60; done\n"
    )
    with shell_stdout.open("ab") as stream:
        launching_shell = subprocess.Popen(
            ["/bin/bash", "-c", shell_script],
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    shell_before = process_identity(launching_shell.pid)
    if shell_before is None:
        raise RuntimeError("launching shell disappeared before proof began")

    receipt_path = output_root / "smoke_launcher_receipt.json"
    pid_path = output_root / "supervisor_pid.json"
    heartbeat_path = output_root / "heartbeat.json"
    receipt = wait_for(
        lambda: load_json_if_exists(receipt_path),
        timeout_seconds=args.startup_timeout_seconds,
        description="detached launcher receipt",
    )
    supervisor_pid = int(receipt["supervisor_pid"])

    def active_heartbeat():
        heartbeat = load_json_if_exists(heartbeat_path)
        if heartbeat is None or heartbeat.get("mode") != "smoke":
            return None
        payload = dict(heartbeat)
        observed = payload.pop("heartbeat_sha256", None)
        if canonical_hash(payload) != observed:
            raise RuntimeError("heartbeat self-hash mismatch")
        if int(heartbeat.get("active_count", 0)) == 1:
            return heartbeat
        return None

    heartbeat_before = wait_for(
        active_heartbeat,
        timeout_seconds=args.startup_timeout_seconds,
        description="one active smoke evaluator",
    )
    if (smoke_dir / "result.json").exists():
        raise RuntimeError(
            "smoke completed before shell-kill proof could be performed"
        )
    pid_record = json.loads(pid_path.read_text())
    supervisor_before = process_identity(supervisor_pid)
    evaluator_pid = int(heartbeat_before["active"][0]["pid"])
    evaluator_before = process_identity(evaluator_pid)
    if supervisor_before is None or evaluator_before is None:
        raise RuntimeError("supervisor or evaluator was not alive before shell kill")

    killed_utc = utc_now()
    os.kill(launching_shell.pid, signal.SIGKILL)
    launching_shell.wait(timeout=10)
    if pid_alive(launching_shell.pid):
        raise RuntimeError("launching shell survived SIGKILL")
    heartbeat_hash_before = heartbeat_before["heartbeat_sha256"]
    time.sleep(SURVIVAL_SECONDS)
    supervisor_after = process_identity(supervisor_pid)
    heartbeat_after = load_json_if_exists(heartbeat_path)
    result_exists_after = (smoke_dir / "result.json").exists()
    if supervisor_after is None and not result_exists_after:
        raise RuntimeError("supervisor died with no smoke result after shell kill")
    if heartbeat_after is None:
        raise RuntimeError("heartbeat disappeared after shell kill")
    heartbeat_advanced = (
        heartbeat_after.get("heartbeat_sha256") != heartbeat_hash_before
    )
    evaluator_after = process_identity(evaluator_pid)
    evaluator_survived_or_completed = (
        evaluator_after is not None or result_exists_after
    )
    if not heartbeat_advanced or not evaluator_survived_or_completed:
        raise RuntimeError(
            "detached pool did not demonstrably survive launching-shell death"
        )

    wait_for(
        lambda: (output_root / smoke["required_artifact"]).exists(),
        timeout_seconds=args.smoke_timeout_seconds,
        description="completed launch smoke",
    )
    artifact = validate_launch_smoke(
        schedule=schedule,
        contract=contract,
        output_root=output_root,
    )
    result_paths = list(smoke_dir.glob("result.json"))
    if len(result_paths) != 1:
        raise RuntimeError("smoke did not produce exactly one scientific result")
    result_hash = sha256_file(result_paths[0])
    if result_hash != artifact["scientific_result_sha256"]:
        raise RuntimeError("smoke result hash differs from launch-smoke artifact")
    proof: dict[str, Any] = {
        "schema_version": "pact_r2_detachment_proof_v1",
        "passed": True,
        "dispatch_contract_sha256": contract["dispatch_contract_sha256"],
        "schedule_sha256": schedule["schedule_sha256"],
        "rollout_id": smoke["rollout_id"],
        "schedule_row_sha256": smoke["schedule_row_sha256"],
        "launching_shell": {
            "identity_before_kill": shell_before,
            "killed_with": "SIGKILL",
            "killed_utc": killed_utc,
            "alive_after_kill": False,
            "log": str(shell_stdout),
            "log_sha256": sha256_file(shell_stdout),
        },
        "supervisor": {
            "pid_record": pid_record,
            "identity_before_shell_kill": supervisor_before,
            "identity_after_survival_window": supervisor_after,
            "survived_or_completed": True,
        },
        "evaluator": {
            "identity_before_shell_kill": evaluator_before,
            "identity_after_survival_window": evaluator_after,
            "survived_or_completed": evaluator_survived_or_completed,
        },
        "heartbeat": {
            "before_sha256": heartbeat_hash_before,
            "after_sha256": heartbeat_after["heartbeat_sha256"],
            "advanced_after_shell_kill": heartbeat_advanced,
            "survival_window_seconds": SURVIVAL_SECONDS,
        },
        "smoke": {
            "result_count": 1,
            "result_sha256": result_hash,
            "launch_smoke_sha256": artifact["launch_smoke_sha256"],
            "full_schedule_reconciliation_required": True,
        },
        "endpoint_fields_inspected": False,
        "completed_utc": utc_now(),
    }
    proof["detachment_proof_sha256"] = canonical_hash(proof)
    write_json_atomic(proof_path, proof)
    print(proof["detachment_proof_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
