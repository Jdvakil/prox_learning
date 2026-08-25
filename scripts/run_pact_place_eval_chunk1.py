#!/usr/bin/env python3
"""Run smoke or full paired chunk-1 place rollouts with bounded workers."""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/root/act_retrain_venv/bin/python3")
EVALUATOR = ROOT / "submodules" / "act" / "eval_pact_place_row.py"
ENCODER = Path(
    "/root/pact_frontend_screen_artifacts/encoder_v1/embedding_encoder_frozen.pt"
)
ENCODER_SHA256 = "6fd2dd037e3236b5b6bf7fce8cb2709ead0cf52adcbbe9cbad1061efc2fe3206"
ARMS = {
    "ACT": {
        "checkpoint_dir": Path("/root/pact_place_152_pact_vs_act_seed3101/act_seed3101"),
        "checkpoint_sha256": "cd95d805cc1caa672137ce5d58eab1671ba175e36f309cc65070eee0acee2c30",
    },
    "PACT": {
        "checkpoint_dir": Path("/root/pact_place_152_pact_vs_act_seed3101/pact_seed3101"),
        "checkpoint_sha256": "4404138b5445a168c36e0dbd463216419179b9ee6211c9bf3e27ab25f47b1e99",
    },
}
STATS_SHA256 = "1860d71a09e7c6ca5afdcb13a952c6de84f52a7bc4810517554782027322c6de"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def schedule_row(row: dict[str, Any], arm: str) -> dict[str, Any]:
    rollout_id = f"chunk1_{arm.lower()}_{int(row['role_index']):03d}_{row['episode_id'][:16]}"
    payload = {
        "arm": arm,
        "rollout_id": rollout_id,
        "episode_id": row["episode_id"],
        "row_sha256": row["row_sha256"],
        "checkpoint_sha256": ARMS[arm]["checkpoint_sha256"],
        "checkpoint_seed": 3101,
        "num_queries": 1,
    }
    payload["schedule_row_sha256"] = canonical_hash(payload)
    return payload


def command_for(
    schedule: dict[str, Any], manifest_path: Path, row_dir: Path
) -> list[str]:
    arm = schedule["arm"]
    checkpoint_dir = ARMS[arm]["checkpoint_dir"]
    command = [
        str(PYTHON),
        str(EVALUATOR),
        "--num-queries",
        "1",
        "--arm",
        arm,
        "--episode-id",
        schedule["episode_id"],
        "--manifest",
        str(manifest_path.resolve()),
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--checkpoint-sha256",
        schedule["checkpoint_sha256"],
        "--checkpoint-seed",
        "3101",
        "--schedule-row-sha256",
        schedule["schedule_row_sha256"],
        "--rollout-id",
        schedule["rollout_id"],
        "--stats-sha256",
        STATS_SHA256,
        "--output-dir",
        str(row_dir.resolve()),
    ]
    if arm == "PACT":
        command.extend(
            [
                "--surface-encoder",
                str(ENCODER),
                "--surface-encoder-sha256",
                ENCODER_SHA256,
            ]
        )
    return command


def validate_result(path: Path, schedule: dict[str, Any]) -> dict[str, Any]:
    result = json.loads(path.read_text())
    expected = {
        "status": "complete",
        "arm": schedule["arm"],
        "rollout_id": schedule["rollout_id"],
        "schedule_row_sha256": schedule["schedule_row_sha256"],
        "episode_id": schedule["episode_id"],
        "checkpoint_sha256": schedule["checkpoint_sha256"],
    }
    for key, value in expected.items():
        if result.get(key) != value:
            raise RuntimeError(f"{path}: {key}={result.get(key)!r} != {value!r}")
    if not isinstance(result.get("task_success"), bool):
        raise RuntimeError(f"{path}: task_success is not bool")
    audit = result.get("contact_audit", {})
    if "contact_class_totals" not in audit or not isinstance(audit.get("collision_free"), bool):
        raise RuntimeError(f"{path}: place contact audit summary is incomplete")
    info = result.get("policy_info", {})
    if info.get("num_queries") != 1:
        raise RuntimeError(f"{path}: checkpoint was not loaded at num_queries=1")
    if info.get("contact_audit_class") != "PactPlaceContactAudit":
        raise RuntimeError(f"{path}: place audit was not attached")
    if schedule["arm"] == "PACT":
        if info.get("proximity_feature_dim") != 32:
            raise RuntimeError(f"{path}: PACT feature width is not 32")
        if info.get("input_proj_proximity_shape") != [512, 32]:
            raise RuntimeError(f"{path}: PACT projection shape mismatch")
    return result


def run_one(
    row: dict[str, Any], arm: str, manifest_path: Path, output_root: Path
) -> dict[str, Any]:
    schedule = schedule_row(row, arm)
    row_dir = output_root / arm.lower() / f"{int(row['role_index']):03d}_{row['episode_id'][:16]}"
    row_dir.mkdir(parents=True, exist_ok=True)
    result_path = row_dir / "result.json"
    driver_path = row_dir / "driver_result.json"
    if result_path.exists() and driver_path.exists():
        result = validate_result(result_path, schedule)
        driver = json.loads(driver_path.read_text())
        if driver.get("schedule_row_sha256") != schedule["schedule_row_sha256"]:
            raise RuntimeError(f"{driver_path}: schedule identity mismatch")
        return {**driver, "reconciled": True, "task_success": result["task_success"]}
    if result_path.exists() or driver_path.exists():
        raise RuntimeError(f"{row_dir}: partial terminal state refuses automatic rerun")
    command = command_for(schedule, manifest_path, row_dir)
    started = utc_now()
    start = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=ROOT / "submodules" / "act",
        env={
            **os.environ,
            "MUJOCO_GL": "egl",
            "MLSPACES_ASSETS_DIR": "/root/prox_learning/assets",
            "PYTHONPATH": str(ROOT / "submodules" / "molmospaces"),
        },
        text=True,
        capture_output=True,
    )
    elapsed = time.monotonic() - start
    (row_dir / "stdout.log").write_text(completed.stdout)
    (row_dir / "stderr.log").write_text(completed.stderr)
    driver = {
        "schema_version": "pact_place_eval_chunk1_driver_v1",
        **schedule,
        "command": command,
        "started_utc": started,
        "finished_utc": utc_now(),
        "wall_clock_seconds": elapsed,
        "returncode": completed.returncode,
        "result_exists": result_path.exists(),
        "reconciled": False,
    }
    if completed.returncode != 0 or not result_path.exists():
        driver["status"] = "failed"
        write_json_atomic(driver_path, driver)
        raise RuntimeError(
            f"{schedule['rollout_id']} failed rc={completed.returncode}; "
            f"see {row_dir / 'stderr.log'}"
        )
    result = validate_result(result_path, schedule)
    driver.update(
        {
            "status": "complete",
            "task_success": result["task_success"],
            "collision_free_task_success": result["collision_free_task_success"],
            "failure_taxonomy": result["failure_taxonomy"],
        }
    )
    write_json_atomic(driver_path, driver)
    return driver


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--workers", required=True, type=int)
    args = parser.parse_args()
    if not 1 <= args.workers <= 10:
        raise SystemExit("workers must be between 1 and 10")
    if sha256_file(EVALUATOR) == sha256_file(ROOT / "submodules/act/eval_pact_collision_row.py"):
        raise SystemExit("place evaluator unexpectedly aliases the corridor evaluator")
    for arm, config in ARMS.items():
        checkpoint = config["checkpoint_dir"] / "policy_best.ckpt"
        stats = config["checkpoint_dir"] / "dataset_stats.pkl"
        if sha256_file(checkpoint) != config["checkpoint_sha256"]:
            raise SystemExit(f"{arm} checkpoint hash mismatch")
        if sha256_file(stats) != STATS_SHA256:
            raise SystemExit(f"{arm} statistics hash mismatch")
    if sha256_file(ENCODER) != ENCODER_SHA256:
        raise SystemExit("encoder hash mismatch")
    manifest = json.loads(args.manifest.read_text())
    rows = manifest["rows"][:2] if args.mode == "smoke" else manifest["rows"]
    jobs = [(row, arm) for row in rows for arm in ("ACT", "PACT")]
    args.output_root.mkdir(parents=True, exist_ok=True)
    started = utc_now()
    results = []
    errors = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_one, row, arm, args.manifest, args.output_root): (row, arm)
            for row, arm in jobs
        }
        for future in concurrent.futures.as_completed(futures):
            row, arm = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(
                    f"{result['rollout_id']} complete "
                    f"{result.get('wall_clock_seconds', 0.0):.1f}s",
                    flush=True,
                )
            except Exception as exc:
                errors.append(
                    {"role_index": row["role_index"], "arm": arm, "error": repr(exc)}
                )
                print(f"row={row['role_index']} arm={arm} FAILED: {exc}", flush=True)
    summary = {
        "schema_version": "pact_place_eval_chunk1_launcher_summary_v1",
        "mode": args.mode,
        "started_utc": started,
        "finished_utc": utc_now(),
        "workers": args.workers,
        "jobs_requested": len(jobs),
        "jobs_complete": len(results),
        "errors": errors,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": manifest["manifest_sha256"],
        "results": sorted(results, key=lambda value: value["rollout_id"]),
    }
    summary["summary_sha256"] = canonical_hash(summary)
    write_json_atomic(args.output_root / f"{args.mode}_launcher_summary.json", summary)
    if errors:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
