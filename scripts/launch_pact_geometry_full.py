#!/usr/bin/env python3
"""Launch detached geometry evaluation, compaction, and throughput monitor."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from run_pact_confirmatory_schedule import (
    canonical_hash,
    load_dispatch_contract,
    validate_launch_smoke,
    write_json_atomic,
)
from run_pact_frontend_screen_supervisor import utc_now


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/root/act_retrain_venv/bin/python")


def detached(command: list[str], log_path: Path) -> int:
    with log_path.open("ab") as stream:
        process = subprocess.Popen(
            ["/usr/bin/setsid", "/usr/bin/nohup", *command],
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=False,
        )
    return process.pid


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", required=True, type=Path)
    parser.add_argument("--dispatch", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    schedule = json.loads(args.schedule.read_text())
    output_root = args.output_root.resolve()
    contract = load_dispatch_contract(
        args.dispatch,
        schedule,
        manifest_path=args.manifest,
        output_root=output_root,
    )
    if contract.get("schema_version") != "pact_geometry_generalization_dispatch_v1":
        raise ValueError("wrong geometry dispatch contract")
    validate_launch_smoke(schedule=schedule, contract=contract, output_root=output_root)
    proof = json.loads((output_root / contract["detachment_proof"]["required_artifact"]).read_text())
    proof_payload = dict(proof)
    proof_hash = proof_payload.pop("detachment_proof_sha256", None)
    if canonical_hash(proof_payload) != proof_hash or proof.get("passed") is not True:
        raise ValueError("detachment proof is invalid")
    if (output_root / "full_launcher_receipt.json").exists():
        raise ValueError("full geometry dispatch already launched")
    launcher_command = [
        str(PYTHON),
        str(ROOT / "scripts/launch_pact_geometry_detached.py"),
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
    environment = dict(os.environ)
    environment.update(
        {
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
            "PYTHONUNBUFFERED": "1",
            "MLSPACES_ASSETS_DIR": str(ROOT / "assets"),
            "PYTHONPATH": f"{ROOT / 'submodules/molmospaces'}:{ROOT / 'submodules/act'}:{ROOT / 'scripts'}",
        }
    )
    completed = subprocess.run(
        launcher_command,
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    supervisor_pid = int(completed.stdout.strip().splitlines()[-1])
    launched_utc = utc_now()
    receipt = {
        "schema_version": "pact_geometry_generalization_full_launcher_receipt_v1",
        "launched_utc": launched_utc,
        "schedule_sha256": schedule["schedule_sha256"],
        "dispatch_contract_sha256": contract["dispatch_contract_sha256"],
        "detachment_proof_sha256": proof_hash,
        "supervisor_pid": supervisor_pid,
        "workers": schedule["workers"],
        "rollouts": schedule["rollouts"],
    }
    receipt["full_launcher_receipt_sha256"] = canonical_hash(receipt)
    write_json_atomic(output_root / "full_launcher_receipt.json", receipt)
    compactor_pid = detached(
        [
            str(PYTHON),
            str(ROOT / "scripts/compact_pact_geometry_storage.py"),
            "--schedule",
            str(args.schedule.resolve()),
            "--dispatch",
            str(args.dispatch.resolve()),
            "--output-root",
            str(output_root),
        ],
        output_root / "storage_compactor.log",
    )
    throughput_pid = detached(
        [
            str(PYTHON),
            str(ROOT / "scripts/measure_pact_geometry_throughput.py"),
            "--schedule",
            str(args.schedule.resolve()),
            "--output-root",
            str(output_root),
            "--wait",
        ],
        output_root / "throughput_monitor.log",
    )
    receipt["compactor_pid"] = compactor_pid
    receipt["throughput_monitor_pid"] = throughput_pid
    receipt.pop("full_launcher_receipt_sha256")
    receipt["full_launcher_receipt_sha256"] = canonical_hash(receipt)
    write_json_atomic(output_root / "full_launcher_receipt.json", receipt)
    print(json.dumps({"supervisor_pid": supervisor_pid, "compactor_pid": compactor_pid, "throughput_monitor_pid": throughput_pid}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
