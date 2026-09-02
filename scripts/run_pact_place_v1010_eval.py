#!/usr/bin/env python3
"""V10.10: run the paired ACT/PACT held-out evaluation.

Both arms run the identical frozen instances. Each rollout is a separate
process, so one failure cannot corrupt another, and every terminal
``result.json`` is create-only: a row that has already produced a result is
never re-run.

The four-instance smoke runs first. It checks infrastructure only -- checkpoint
loading, scene identity, telemetry, memory, ETA. Its *performance* never decides
whether the full evaluation runs.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import signal
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from pact_place_v109_contract import ENCODER_PATH  # noqa: E402
from pact_place_v1010_contract import (  # noqa: E402
    CONTRACT_VERSION_V1010,
    ENCODER_SHA256,
    EVAL_ROOT,
    TRAINING_ROOT,
    WORK_ROOT,
    canonical_payload_sha256,
    empty_authorization,
    sha256_file,
    sha256_payload,
    write_immutable_create_only,
)
from pact_place_v1010_eval_contract import (  # noqa: E402
    EVAL_NUM_QUERIES,
    SMOKE_ROLE,
    load_manifest,
)

EVALUATOR = ROOT / "submodules" / "act" / "eval_pact_place_v1010_row.py"
ARMS = ("ACT", "PACT")
CHECKPOINT_SEED = 3101

# Each evaluator imports NumPy/SciPy/PyTorch/Warp. Left unconstrained, those
# libraries size independent native pools from the 128 visible CPUs. Ten
# concurrent rollouts consequently exhausted this container's 3,840 PID/thread
# cgroup limit before useful GPU work began. These settings affect execution
# resources only; they do not alter the model, manifest, seeds, environment, or
# decoder.
NATIVE_THREAD_ENV: dict[str, str] = {
    "OMP_NUM_THREADS": "1",
    "OMP_THREAD_LIMIT": "1",
    "OMP_DYNAMIC": "FALSE",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "MKL_DYNAMIC": "FALSE",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "RAYON_NUM_THREADS": "1",
}
MAX_EVAL_WORKERS = 4
DEFAULT_ROLLOUT_TIMEOUT_MINUTES = 45.0


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def arm_binding(arm: str) -> dict[str, Any]:
    directory = Path(TRAINING_ROOT) / f"{arm.lower()}_seed3101"
    checkpoint = directory / "policy_best.ckpt"
    stats = directory / "dataset_stats.pkl"
    for path in (checkpoint, stats):
        if not path.is_file():
            raise SystemExit(f"missing {path}")
    return {
        "arm": arm,
        "checkpoint_dir": str(directory),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "stats_sha256": sha256_file(stats),
        "checkpoint_seed": CHECKPOINT_SEED,
    }


def schedule_row(row: dict[str, Any], arm: str, binding: dict[str, Any],
                 role: str, save_trajectory: bool = False) -> dict[str, Any]:
    tag = "smoke" if role == SMOKE_ROLE else "eval"
    payload = {
        "arm": arm,
        "rollout_id": f"v109_{tag}_{arm.lower()}_"
                      f"{int(row['candidate_index']):03d}_{row['episode_id'][:16]}",
        "episode_id": row["episode_id"],
        "candidate_index": int(row["candidate_index"]),
        "cell": row["cell"],
        "row_sha256": row["row_sha256"],
        "checkpoint_sha256": binding["checkpoint_sha256"],
        "stats_sha256": binding["stats_sha256"],
        "checkpoint_seed": CHECKPOINT_SEED,
        "num_queries": EVAL_NUM_QUERIES,
        "task_seed_u32": int(row["task_seed_u32"]),
        "pose_id": row["pose_id"],
        "scene_sha256": row["pact_v106_scene_sha256"],
    }
    payload["schedule_row_sha256"] = sha256_payload(payload)
    # Retention is a storage decision, not part of the rollout identity: the
    # schedule hash is computed before it is recorded, so a trajectory-saving
    # re-run addresses exactly the same rows as the original.
    payload["save_trajectory"] = bool(save_trajectory)
    return payload


def command_for(schedule: dict[str, Any], manifest_path: Path, row_dir: Path,
                binding: dict[str, Any]) -> list[str]:
    command = [
        sys.executable, str(EVALUATOR),
        "--arm", schedule["arm"],
        "--episode-id", schedule["episode_id"],
        "--manifest", str(manifest_path.resolve()),
        "--checkpoint-dir", binding["checkpoint_dir"],
        "--checkpoint-sha256", binding["checkpoint_sha256"],
        "--checkpoint-seed", str(CHECKPOINT_SEED),
        "--schedule-row-sha256", schedule["schedule_row_sha256"],
        "--rollout-id", schedule["rollout_id"],
        "--stats-sha256", binding["stats_sha256"],
        "--output-dir", str(row_dir.resolve()),
    ]
    if schedule["arm"] == "PACT":
        command += ["--surface-encoder", ENCODER_PATH,
                    "--surface-encoder-sha256", ENCODER_SHA256]
    if schedule.get("save_trajectory"):
        # The inherited evaluator gates the trajectory HDF5 and the MP4s behind
        # the same flag, so retaining trajectories necessarily retains videos.
        command += ["--save-video"]
    return command


def subprocess_environment(*, h5_only: bool) -> dict[str, str]:
    """Build a deterministic, resource-bounded evaluator environment."""
    environment = dict(os.environ)
    environment.update(NATIVE_THREAD_ENV)
    environment.setdefault("MUJOCO_GL", "egl")
    environment.setdefault("PYOPENGL_PLATFORM", "egl")
    environment.setdefault("PYTHONUNBUFFERED", "1")
    environment.setdefault("MLSPACES_ASSETS_DIR", str(ROOT / "assets"))
    environment.setdefault("PACT_CONTACT_AUDIT_SUMMARY_ONLY", "1")
    # Pin every math-library thread pool, as the V10.8 collector does. Without
    # this each rollout lets OpenMP size itself to the 128 visible CPUs, and a
    # handful of concurrent rollouts exhaust the cgroup's 3840-PID budget:
    # "libgomp: Thread creation failed: Resource temporarily unavailable",
    # followed by CUDA failures on the survivors.
    for _name, _value in (("OPENBLAS_NUM_THREADS", "1"), ("OMP_NUM_THREADS", "1"),
                          ("MKL_NUM_THREADS", "1"), ("NUMEXPR_NUM_THREADS", "1"),
                          ("VECLIB_MAXIMUM_THREADS", "1")):
        environment[_name] = _value
    if h5_only:
        environment["PACT_V109_TRAJECTORY_H5_ONLY"] = "1"
    else:
        environment.pop("PACT_V109_TRAJECTORY_H5_ONLY", None)
    environment.pop("DISPLAY", None)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "submodules" / "molmospaces"), str(ROOT / "scripts"),
         environment.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    return environment


def stop_process_group(process: subprocess.Popen[Any], grace_seconds: float = 5.0) -> None:
    """Terminate only the timed-out evaluator's private process group."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait()


def run_one(payload: dict[str, Any]) -> dict[str, Any]:
    row_dir = Path(payload["row_dir"])
    result_path = row_dir / "result.json"
    if result_path.is_file():
        return {**payload["schedule"], "status": "already_present",
                "returncode": 0, "elapsed_s": 0.0}
    row_dir.mkdir(parents=True, exist_ok=True)
    environment = subprocess_environment(h5_only=bool(payload.get("h5_only")))
    started = time.monotonic()
    with (row_dir / "rollout.log").open("w") as stream:
        process = subprocess.Popen(
            payload["command"], cwd=ROOT / "submodules" / "act", env=environment,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            returncode = process.wait(timeout=float(payload["timeout_seconds"]))
            timed_out = False
        except subprocess.TimeoutExpired:
            timed_out = True
            stream.write(
                f"\nV10.10 infrastructure timeout after "
                f"{float(payload['timeout_seconds']) / 60.0:.1f} minutes\n")
            stream.flush()
            stop_process_group(process)
            returncode = -int(signal.SIGKILL)
    elapsed = time.monotonic() - started
    return {**payload["schedule"], "returncode": returncode,
            "elapsed_s": round(elapsed, 1),
            "status": ("timeout" if timed_out else
                       "complete" if result_path.is_file() else "no_result"),
            "row_dir": str(row_dir)}


def disk_free_gb() -> float:
    return shutil.disk_usage("/root").free / 1e9


def run_stage(rows: list[dict[str, Any]], manifest_path: Path, output_root: Path,
              bindings: dict[str, dict[str, Any]], role: str,
              workers: int, save_trajectory: bool = False,
              disk_floor_gb: float = 3.0,
              h5_only: bool = False,
              timeout_seconds: float = DEFAULT_ROLLOUT_TIMEOUT_MINUTES * 60.0,
              ) -> list[dict[str, Any]]:
    if not 1 <= int(workers) <= MAX_EVAL_WORKERS:
        raise ValueError(
            f"workers must be in [1,{MAX_EVAL_WORKERS}], got {workers}; "
            "larger pools exhaust the container PID/thread budget")
    payloads = []
    for arm in ARMS:
        for row in rows:
            schedule = schedule_row(row, arm, bindings[arm], role, save_trajectory)
            row_dir = (output_root / arm.lower() /
                       f"{int(row['candidate_index']):03d}_{row['episode_id'][:16]}")
            payloads.append({
                "schedule": schedule,
                "row_dir": str(row_dir),
                "command": command_for(schedule, manifest_path, row_dir, bindings[arm]),
                "h5_only": bool(h5_only),
                "timeout_seconds": float(timeout_seconds),
            })
    print(f"{role}: {len(payloads)} rollouts on {workers} workers", flush=True)
    results: list[dict[str, Any]] = []
    started = time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_one, p): p for p in payloads}
        for done, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            payload = futures[future]
            try:
                results.append(future.result())
            except BaseException as error:  # noqa: BLE001
                results.append({**payload["schedule"], "returncode": -1,
                                "status": "worker_died",
                                "error": f"{type(error).__name__}: {error}"[:300],
                                "row_dir": payload["row_dir"], "elapsed_s": 0.0})
            elapsed = time.monotonic() - started
            rate = elapsed / done
            free = disk_free_gb()
            print(f"  {done}/{len(payloads)}  {elapsed / 60:.1f} min elapsed, "
                  f"eta {(len(payloads) - done) * rate / 60:.1f} min, "
                  f"{free:.1f} GB free", flush=True)
            if save_trajectory and free < disk_floor_gb:
                print(f"HALT: {free:.1f} GB free is below the {disk_floor_gb} GB "
                      "floor; cancelling the remaining rollouts", flush=True)
                for pending in futures:
                    pending.cancel()
                break
    return sorted(results, key=lambda r: (r["arm"], r["candidate_index"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "full"), required=True)
    parser.add_argument("--workers", type=int, default=MAX_EVAL_WORKERS)
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / EVAL_ROOT / "eval_manifest.json")
    parser.add_argument("--save-trajectory", action="store_true",
                        help="retain trajectory.h5, videos and actions.npz per rollout")
    parser.add_argument("--eval-root", type=str, default=EVAL_ROOT,
                        help="output root; use a fresh one to preserve a completed run")
    parser.add_argument("--disk-floor-gb", type=float, default=3.0)
    parser.add_argument("--rollout-timeout-minutes", type=float,
                        default=DEFAULT_ROLLOUT_TIMEOUT_MINUTES)
    parser.add_argument("--h5-only", action="store_true",
                        help="publish trajectory.h5 without invoking ffmpeg")
    args = parser.parse_args()
    if not 1 <= args.workers <= MAX_EVAL_WORKERS:
        parser.error(
            f"--workers must be in [1,{MAX_EVAL_WORKERS}] for the bounded "
            "V10.10 evaluator")
    if args.rollout_timeout_minutes <= 0:
        parser.error("--rollout-timeout-minutes must be positive")
    eval_root = args.eval_root

    verification = json.loads(
        (ROOT / WORK_ROOT / "training_verification.json").read_text())
    if not verification.get("cleared_for_live_evaluation"):
        raise SystemExit("training verification did not clear live evaluation")

    manifest = load_manifest(args.manifest)
    bindings = {arm: arm_binding(arm) for arm in ARMS}
    for arm in ARMS:
        recorded = verification["arms"][arm.lower()]["hashes"]["policy_best.ckpt"]
        if bindings[arm]["checkpoint_sha256"] != recorded:
            raise SystemExit(f"{arm} checkpoint changed since verification")

    if args.stage == "smoke":
        rows = manifest["smoke"]["rows"]
        role = SMOKE_ROLE
        output_root = ROOT / eval_root / "smoke"
        out = ROOT / eval_root / "smoke_run.json"
    else:
        rows = manifest["rows"]
        role = manifest["role"]
        output_root = ROOT / eval_root / "rollouts"
        out = ROOT / eval_root / "full_run.json"
        smoke_path = ROOT / eval_root / "smoke_run.json"
        if not smoke_path.is_file():
            smoke_path = ROOT / EVAL_ROOT / "smoke_run.json"
        smoke = json.loads(smoke_path.read_text())
        if not smoke.get("infrastructure_healthy"):
            raise SystemExit("infrastructure smoke did not pass; refusing full run")

    started = utc_now()
    monotonic = time.monotonic()
    results = run_stage(rows, args.manifest, output_root, bindings, role, args.workers,
                        save_trajectory=args.save_trajectory,
                        disk_floor_gb=args.disk_floor_gb, h5_only=args.h5_only,
                        timeout_seconds=args.rollout_timeout_minutes * 60.0)
    elapsed = time.monotonic() - monotonic

    failures = [r for r in results if r["status"] != "complete" or r["returncode"] != 0]
    document: dict[str, Any] = {
        **empty_authorization(),
        "schema_version": f"pact_place_v1010_{args.stage}_run_v1",
        "contract_version": CONTRACT_VERSION_V1010,
        "role": role,
        "is_phase0_pass": False,
        "manifest_sha256": manifest["manifest_sha256"],
        "arms": bindings,
        "encoder_sha256": ENCODER_SHA256,
        "instances": len(rows),
        "rollouts_attempted": len(results),
        "rollouts_complete": sum(1 for r in results if r["status"] == "complete"),
        "failures": failures,
        "started_utc": started,
        "finished_utc": utc_now(),
        "elapsed_hours": round(elapsed / 3600, 3),
        "mean_rollout_minutes": round(
            sum(r["elapsed_s"] for r in results) / max(1, len(results)) / 60, 2),
        "workers": args.workers,
        "max_eval_workers": MAX_EVAL_WORKERS,
        "native_thread_environment": dict(NATIVE_THREAD_ENV),
        "rollout_timeout_minutes": float(args.rollout_timeout_minutes),
        "eval_root": eval_root,
        "save_trajectory": bool(args.save_trajectory),
        "h5_only": bool(args.h5_only),
        "disk_free_gb_at_end": round(disk_free_gb(), 2),
        "results": results,
    }
    if args.stage == "smoke":
        document["infrastructure_healthy"] = not failures
        document["performance_does_not_gate_full_evaluation"] = True
        document["note"] = (
            "Checkpoint loading, scene identity, telemetry, memory and ETA only. "
            "The full evaluation runs if infrastructure is healthy even if both "
            "policies perform poorly.")
    document["payload_sha256"] = canonical_payload_sha256(document)
    written = write_immutable_create_only(out, document)
    print(json.dumps({
        "stage": args.stage,
        "rollouts_complete": document["rollouts_complete"],
        "rollouts_attempted": document["rollouts_attempted"],
        "failures": len(failures),
        "elapsed_hours": document["elapsed_hours"],
        "mean_rollout_minutes": document["mean_rollout_minutes"],
        "infrastructure_healthy": document.get("infrastructure_healthy"),
        "payload_sha256": document["payload_sha256"],
        "raw_file_sha256": written.get("raw_file_sha256"),
    }, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
