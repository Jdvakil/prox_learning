#!/usr/bin/env python3
"""Launch full contact execution, compaction, finalization, and guarded commit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from run_pact_confirmatory_schedule import (
    canonical_hash,
    load_dispatch_contract,
    validate_launch_smoke,
)


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/root/act_retrain_venv/bin/python")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def validate_proof(output_root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    path = output_root / contract["detachment_proof"]["required_artifact"]
    proof = json.loads(path.read_text())
    payload = dict(proof)
    observed = payload.pop("detachment_proof_sha256", None)
    if observed != canonical_hash(payload):
        raise ValueError("detachment proof self-hash mismatch")
    if (
        proof.get("passed") is not True
        or proof.get("dispatch_contract_sha256") != contract["dispatch_contract_sha256"]
        or proof.get("endpoint_fields_inspected") is not False
        or proof.get("endpoint_outcome_values_inspected") is not False
        or proof.get("contact_smoke_validation", {}).get("passed") is not True
    ):
        raise ValueError("detachment proof is not valid for this dispatch")
    return proof


def detached(command: list[str], log_path: Path) -> subprocess.Popen:
    with log_path.open("ab") as log:
        return subprocess.Popen(
            ["/usr/bin/setsid", "/usr/bin/nohup", *command],
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=False,
        )


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gpu_memory_used_mib() -> int:
    completed = subprocess.run(
        [
            "/usr/bin/nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return sum(int(line.strip()) for line in completed.stdout.splitlines() if line.strip())


def full_pool_memory_gate(
    *, output_root: Path, supervisor_pid: int, workers: int, seconds: int, threshold_mib: int
) -> dict[str, Any]:
    heartbeat_path = output_root / "heartbeat.json"
    samples: list[tuple[str, int, int, int]] = []
    wait_deadline = time.monotonic() + 300.0
    gate_deadline: float | None = None
    while gate_deadline is None or time.monotonic() < gate_deadline:
        try:
            os.kill(supervisor_pid, 0)
        except ProcessLookupError as error:
            raise RuntimeError("contact supervisor exited during GPU-memory gate") from error
        active_count = 0
        complete_count = 0
        if heartbeat_path.is_file():
            heartbeat = json.loads(heartbeat_path.read_text())
            active_count = int(heartbeat.get("active_count", 0))
            complete_count = int(heartbeat.get("complete_count", 0))
            if active_count >= workers and gate_deadline is None:
                gate_deadline = time.monotonic() + seconds
        memory_mib = gpu_memory_used_mib()
        samples.append((utc_now(), memory_mib, active_count, complete_count))
        if memory_mib > threshold_mib:
            os.kill(supervisor_pid, signal.SIGTERM)
            artifact = {
                "schema_version": "pact_contact_gpu_memory_gate_v1",
                "workers": workers,
                "threshold_mib": threshold_mib,
                "gate_seconds_after_full_occupancy": seconds,
                "sample_count": len(samples),
                "peak_memory_used_mib": max(item[1] for item in samples),
                "threshold_exceeded": True,
                "passed": False,
                "supervisor_terminated": True,
                "complete_count_at_abort": complete_count,
                "outcome_fields_read": False,
                "first_sample_utc": samples[0][0],
                "last_sample_utc": samples[-1][0],
            }
            artifact["gpu_memory_gate_sha256"] = canonical_hash(artifact)
            write_json_atomic(output_root / "gpu_memory_first_minutes.json", artifact)
            raise RuntimeError(
                "GPU memory exceeded the frozen 20000 MiB guard; full pool was aborted "
                "without reading outcomes and must be re-frozen once at the next lower count"
            )
        if gate_deadline is None and time.monotonic() >= wait_deadline:
            os.kill(supervisor_pid, signal.SIGTERM)
            raise RuntimeError("full worker pool did not form within five minutes")
        time.sleep(1.0)
    peak_sample = max(samples, key=lambda item: item[1])
    artifact = {
        "schema_version": "pact_contact_gpu_memory_gate_v1",
        "workers": workers,
        "threshold_mib": threshold_mib,
        "gate_seconds_after_full_occupancy": seconds,
        "sample_count": len(samples),
        "peak_memory_used_mib": peak_sample[1],
        "peak_sample_utc": peak_sample[0],
        "maximum_active_count": max(item[2] for item in samples),
        "threshold_exceeded": False,
        "passed": True,
        "supervisor_terminated": False,
        "outcome_fields_read": False,
        "first_sample_utc": samples[0][0],
        "last_sample_utc": samples[-1][0],
    }
    artifact["gpu_memory_gate_sha256"] = canonical_hash(artifact)
    write_json_atomic(output_root / "gpu_memory_first_minutes.json", artifact)
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", required=True, type=Path)
    parser.add_argument("--dispatch", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--storage-amendment", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--expected-head", required=True)
    args = parser.parse_args()
    schedule = json.loads(args.schedule.read_text())
    output_root = args.output_root.resolve()
    contract = load_dispatch_contract(
        args.dispatch.resolve(),
        schedule,
        manifest_path=args.manifest.resolve(),
        output_root=output_root,
    )
    if contract.get("schema_version") != "pact_contact_endpoint_dispatch_v1":
        raise ValueError("wrong contact dispatch")
    validate_launch_smoke(schedule=schedule, contract=contract, output_root=output_root)
    proof = validate_proof(output_root, contract)
    if (output_root / "full_launcher_receipt.json").exists():
        raise ValueError("full contact dispatch already has a launcher receipt")
    environment = dict(os.environ)
    environment.update(
        {
            "MUJOCO_GL": "egl",
            "PYTHONUNBUFFERED": "1",
            "MLSPACES_ASSETS_DIR": "/root/prox_learning_pact_remediation/assets",
            "PYTHONPATH": (
                f"{ROOT / 'submodules/molmospaces'}:{ROOT / 'submodules/act'}:{ROOT / 'scripts'}"
            ),
        }
    )
    supervisor_command = [
        str(PYTHON),
        str(ROOT / "scripts/launch_pact_contact_detached.py"),
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
    ]
    completed = subprocess.run(
        supervisor_command,
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    supervisor_pid = int(completed.stdout.strip().splitlines()[-1])
    memory_guard = contract["gpu_memory_guard"]
    memory_gate = full_pool_memory_gate(
        output_root=output_root,
        supervisor_pid=supervisor_pid,
        workers=int(contract["execution"]["fixed_worker_count"]),
        seconds=int(memory_guard["full_pool_gate_seconds"]),
        threshold_mib=int(memory_guard["abort_threshold_mib"]),
    )
    commands = {
        "compactor": [
            str(PYTHON),
            str(ROOT / "scripts/compact_pact_contact_storage.py"),
            "--schedule",
            str(args.schedule.resolve()),
            "--storage-amendment",
            str(args.storage_amendment.resolve()),
            "--output-root",
            str(output_root),
        ],
        "finalizer": [
            str(PYTHON),
            str(ROOT / "scripts/finalize_pact_contact_endpoint.py"),
            "--schedule",
            str(args.schedule.resolve()),
            "--dispatch",
            str(args.dispatch.resolve()),
            "--manifest",
            str(args.manifest.resolve()),
            "--storage-amendment",
            str(args.storage_amendment.resolve()),
            "--output-root",
            str(output_root),
        ],
        "committer": [
            str(PYTHON),
            str(ROOT / "scripts/commit_pact_contact_results.py"),
            "--expected-head",
            args.expected_head,
        ],
    }
    processes = {}
    logs = {}
    for label, command in commands.items():
        log_path = output_root / f"{label}.log"
        process = detached(command, log_path)
        processes[label] = process.pid
        logs[label] = str(log_path)
    receipt: dict[str, Any] = {
        "schema_version": "pact_contact_endpoint_full_stack_receipt_v1",
        "launched_utc": utc_now(),
        "schedule_sha256": schedule["schedule_sha256"],
        "dispatch_contract_sha256": contract["dispatch_contract_sha256"],
        "detachment_proof_sha256": proof["detachment_proof_sha256"],
        "gpu_memory_gate": {
            "path": str(output_root / "gpu_memory_first_minutes.json"),
            "file_sha256": file_hash(output_root / "gpu_memory_first_minutes.json"),
            "gpu_memory_gate_sha256": memory_gate["gpu_memory_gate_sha256"],
            "peak_memory_used_mib": memory_gate["peak_memory_used_mib"],
            "passed": True,
        },
        "expected_head": args.expected_head,
        "supervisor_pid": supervisor_pid,
        "auxiliary_pids": processes,
        "supervisor_launcher_command": supervisor_command,
        "auxiliary_commands": commands,
        "logs": logs,
        "endpoint_fields_inspected": False,
    }
    receipt["full_stack_receipt_sha256"] = canonical_hash(receipt)
    write_json_atomic(output_root / "full_stack_receipt.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
