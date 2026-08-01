#!/usr/bin/env python3
"""Launch full contact execution, compaction, finalization, and guarded commit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
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
        or proof.get("dispatch_contract_sha256")
        != contract["dispatch_contract_sha256"]
        or proof.get("endpoint_fields_inspected") is not False
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
                f"{ROOT / 'submodules/molmospaces'}:{ROOT / 'submodules/act'}:"
                f"{ROOT / 'scripts'}"
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
