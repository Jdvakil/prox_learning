#!/usr/bin/env python3
"""Launch the R2 supervisor via setsid+nohup and return immediately."""

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

PYTHON = Path("/root/act_retrain_venv/bin/python")
ROOT = Path(__file__).resolve().parents[1]
SUPERVISOR = ROOT / "scripts/run_pact_r2_supervisor.py"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
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


def build_command(args: argparse.Namespace, output_root: Path) -> list[str]:
    return [
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
    parent_log = output_root / f"{args.mode}_supervisor.log"
    command = build_command(args, output_root)
    environment = dict(os.environ)
    environment["MUJOCO_GL"] = "egl"
    environment["PYTHONUNBUFFERED"] = "1"
    environment["MLSPACES_ASSETS_DIR"] = (
        "/root/prox_learning_hybrid_safety/assets"
    )
    environment["PYTHONPATH"] = str(ROOT / "submodules/molmospaces")
    with parent_log.open("ab") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
            start_new_session=False,
        )
    receipt = {
        "schema_version": "pact_r2_launcher_receipt_v1",
        "launcher_pid": os.getpid(),
        "launcher_ppid": os.getppid(),
        "supervisor_pid": process.pid,
        "mode": args.mode,
        "command": command,
        "parent_log": str(parent_log),
        "launched_utc": utc_now(),
        "stdin_detached": True,
        "environment": {
            key: environment[key]
            for key in (
                "MUJOCO_GL",
                "PYTHONUNBUFFERED",
                "MLSPACES_ASSETS_DIR",
                "PYTHONPATH",
            )
        },
    }
    receipt["launcher_receipt_sha256"] = canonical_hash(receipt)
    final = output_root / f"{args.mode}_launcher_receipt.json"
    write_json_atomic(final, receipt)
    print(process.pid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
