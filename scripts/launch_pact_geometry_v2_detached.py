#!/usr/bin/env python3
"""Launch the attempt-2 geometry supervisor with setsid/nohup."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from run_pact_confirmatory_schedule import canonical_hash, write_json_atomic
from run_pact_frontend_screen_supervisor import utc_now


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/root/act_retrain_venv/bin/python")
SUPERVISOR = ROOT / "scripts/run_pact_geometry_v2_supervisor.py"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", required=True, type=Path)
    parser.add_argument("--dispatch-contract", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("smoke", "full"))
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    command = [
        "/usr/bin/setsid",
        "/usr/bin/nohup",
        str(PYTHON),
        str(SUPERVISOR),
        "--schedule",
        str(args.schedule.resolve()),
        "--dispatch-contract",
        str(args.dispatch_contract.resolve()),
        "--manifest",
        str(args.manifest.resolve()),
        "--output-root",
        str(output_root),
        "--mode",
        args.mode,
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
    log_path = output_root / f"{args.mode}_supervisor.log"
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
    receipt = {
        "schema_version": "pact_geometry_generalization_v2_launcher_receipt",
        "launcher_pid": os.getpid(),
        "launcher_ppid": os.getppid(),
        "supervisor_pid": process.pid,
        "mode": args.mode,
        "command": command,
        "parent_log": str(log_path),
        "launched_utc": utc_now(),
        "stdin_detached": True,
        "environment": {
            key: environment[key]
            for key in (
                "MUJOCO_GL",
                "PYOPENGL_PLATFORM",
                "PYTHONUNBUFFERED",
                "MLSPACES_ASSETS_DIR",
                "PYTHONPATH",
                "PACT_CONTACT_AUDIT_SUMMARY_ONLY",
            )
        },
    }
    receipt["launcher_receipt_sha256"] = canonical_hash(receipt)
    write_json_atomic(output_root / f"{args.mode}_launcher_receipt.json", receipt)
    print(process.pid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
