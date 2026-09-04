#!/usr/bin/env python3
"""V10.8 exploratory collection: 152 strict-clean trainable demonstrations.

**Owner override for scientific curiosity. NOT a Phase-0 pass.** V10.7's gate
failed at 8/24 and stays closed and unmodified.

Uses the full datagen pipeline (``prepare_episode_for_saving`` /
``save_trajectories``), never the expert-screen harness, so every accepted
episode carries actions, qpos/qvel, wrist RGB and all 40 proximity sensors.

Attempts are scheduled by deterministic round-robin over cells with unmet
quotas, drawing from frozen per-cell seed streams. Quotas are never relaxed or
redistributed. Every attempt is recorded in an append-only ledger before its
heavy staging is pruned.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MOLMO = ROOT / "submodules" / "molmospaces"
for _p in (ROOT / "scripts", MOLMO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from pact_place_v108_contract import (  # noqa: E402
    COLLECTION_ROOT,
    CONTRACT_VERSION_V108,
    DATASET_ROOT,
    DEFAULT_WORKERS,
    ENVIRONMENT_VERSION,
    MAX_SCIENTIFIC_ATTEMPTS,
    MAX_WALL_CLOCK_HOURS,
    N_PROXIMITY_SENSORS,
    POSE_IDS,
    PROXIMITY_FRAME_SHAPE,
    PROXIMITY_SENSOR_PERIOD_MS,
    REPORT_AFTER_ATTEMPTS,
    REQUIRED_ACTION_KEYS,
    REQUIRED_AGENT_KEYS,
    SAMPLER_CLASS,
    TARGET_SUCCESSES,
    TASK_HORIZON,
    V107_CERT_JSON,
    WRIST_VIDEO_SUFFIX,
    build_attempt_row,
    build_contract,
    cell_key,
    cells,
    empty_authorization,
    next_attempts,
    quota_totals,
    quotas,
    quotas_met,
    remaining_quota,
    row_defects,
    sha256_file,
    sha256_payload,
    write_immutable_create_only,
)

THREAD_POOL_ENV = {
    "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}
MIN_FREE_GB = 12.0
PROJECTED_PEAK_GB = 7.2


def pin_threads() -> None:
    """Must run before any worker is created, not inside the worker."""
    for name, value in THREAD_POOL_ENV.items():
        os.environ[name] = value


def _git_head(repo: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
def disk_preflight(dataset_root: Path) -> dict[str, Any]:
    usage = shutil.disk_usage(dataset_root.parent if dataset_root.parent.exists()
                              else ROOT)
    free_gb = usage.free / 1e9
    return {
        "free_gb": free_gb,
        "total_gb": usage.total / 1e9,
        "used_pct": 100.0 * usage.used / usage.total,
        "projected_peak_gb": PROJECTED_PEAK_GB,
        "min_free_gb_required": MIN_FREE_GB,
        "headroom_after_peak_gb": free_gb - PROJECTED_PEAK_GB,
        "passed": bool(free_gb - PROJECTED_PEAK_GB >= MIN_FREE_GB),
    }


def cgroup_pid_preflight(workers: int) -> dict[str, Any]:
    def read(path: str) -> str | None:
        try:
            return Path(path).read_text().strip()
        except OSError:
            return None

    pids_max = read("/sys/fs/cgroup/pids.max")
    pids_cur = read("/sys/fs/cgroup/pids.current")
    cpu_max = read("/sys/fs/cgroup/cpu.max")
    mem_max = read("/sys/fs/cgroup/memory.max")
    mem_cur = read("/sys/fs/cgroup/memory.current")
    headroom = None
    if pids_max and pids_max != "max" and pids_cur:
        headroom = int(pids_max) - int(pids_cur)
    effective_cpus = None
    if cpu_max and cpu_max != "max":
        quota, period = cpu_max.split()
        if quota != "max":
            effective_cpus = int(quota) / int(period)
    mem_free_gb = None
    if mem_max and mem_max != "max" and mem_cur:
        mem_free_gb = (int(mem_max) - int(mem_cur)) / 1e9
    # Each worker is a spawned process with its own simulator; budget
    # generously so a PID ceiling never truncates the run mid-attempt.
    pids_needed = workers * 24
    return {
        "pids_max": pids_max, "pids_current": pids_cur,
        "pids_headroom": headroom, "pids_needed_estimate": pids_needed,
        "cpu_max": cpu_max, "effective_cpus": effective_cpus,
        "memory_free_gb": mem_free_gb,
        "workers_requested": workers,
        "workers_fit_cpu": bool(
            effective_cpus is None or workers <= effective_cpus),
        "passed": bool(
            (headroom is None or headroom >= pids_needed)
            and (effective_cpus is None or workers <= effective_cpus)
            and (mem_free_gb is None or mem_free_gb >= 8.0)
        ),
    }


def scene_preflight() -> dict[str, Any]:
    certification = json.loads((ROOT / V107_CERT_JSON).read_text())
    scenes: dict[str, Any] = {}
    problems: list[str] = []
    for pose in POSE_IDS:
        entry = certification["published_scenes"][pose]
        path = ROOT / entry["relative"]
        if not path.is_file():
            problems.append(f"missing scene {pose}")
            continue
        observed = sha256_file(path)
        if observed != entry["sha256"]:
            problems.append(f"scene drifted {pose}")
        scenes[pose] = {"relative": entry["relative"], "sha256": observed}
    return {
        "scenes": scenes,
        "selected": certification["selected"],
        "certification_passed": bool(certification["certification_passed"]),
        "problems": problems,
        "passed": not problems and len(scenes) == len(POSE_IDS),
    }


# ---------------------------------------------------------------------------
# Trainable-schema validation
# ---------------------------------------------------------------------------
def validate_trainable(row_dir: Path) -> dict[str, Any]:
    """Every requirement an accepted episode must satisfy, recomputed."""
    import h5py
    import numpy as np

    problems: list[str] = []
    h5_path = row_dir / "trajectory.h5"
    detail: dict[str, Any] = {"h5_present": h5_path.is_file()}
    if not h5_path.is_file():
        return {"detail": detail, "problems": ["trajectory.h5 missing"],
                "passed": False}
    try:
        with h5py.File(h5_path, "r") as handle:
            if "traj_0" not in handle:
                return {"detail": detail, "problems": ["no traj_0 group"],
                        "passed": False}
            traj = handle["traj_0"]
            n_frames = None
            for key in REQUIRED_ACTION_KEYS:
                node = traj.get(f"actions/{key}")
                if node is None:
                    problems.append(f"missing actions/{key}")
                    continue
                n_frames = n_frames or int(node.shape[0])
            for key in REQUIRED_AGENT_KEYS:
                node = traj.get(f"obs/agent/{key}")
                if node is None:
                    problems.append(f"missing obs/agent/{key}")
            proximity = traj.get("obs/proximity")
            if proximity is None:
                problems.append("missing obs/proximity")
            else:
                names = sorted(proximity)
                detail["n_proximity"] = len(names)
                if len(names) != N_PROXIMITY_SENSORS:
                    problems.append(
                        f"{len(names)} proximity sensors, need "
                        f"{N_PROXIMITY_SENSORS}")
                bad_shape, bad_dtype, nonfinite, constant = [], [], [], []
                for name in names:
                    node = proximity[name]
                    if tuple(node.shape[1:]) != PROXIMITY_FRAME_SHAPE:
                        bad_shape.append(name)
                        continue
                    if node.dtype != np.float32:
                        bad_dtype.append(name)
                        continue
                    if n_frames is None:
                        n_frames = int(node.shape[0])
                    elif int(node.shape[0]) != n_frames:
                        bad_shape.append(name)
                        continue
                    values = np.asarray(node[...], dtype=np.float64)
                    if not np.all(np.isfinite(values)):
                        nonfinite.append(name)
                    elif float(values.max() - values.min()) <= 0.0:
                        constant.append(name)
                if bad_shape:
                    problems.append(f"proximity shape wrong: {bad_shape[:3]}")
                if bad_dtype:
                    problems.append(f"proximity dtype wrong: {bad_dtype[:3]}")
                if nonfinite:
                    problems.append(f"proximity non-finite: {nonfinite[:3]}")
                if constant:
                    problems.append(f"proximity constant: {constant[:3]}")
                detail["n_nonfinite"] = len(nonfinite)
                detail["n_constant"] = len(constant)
            detail["n_frames"] = n_frames
    except Exception as error:  # noqa: BLE001
        return {"detail": detail,
                "problems": [f"h5 unreadable: {type(error).__name__}: {error}"],
                "passed": False}

    wrist = sorted(row_dir.glob(f"*{WRIST_VIDEO_SUFFIX}"))
    detail["wrist_videos"] = [p.name for p in wrist]
    if not wrist:
        problems.append("missing wrist RGB video")
    else:
        path = wrist[0]
        if path.stat().st_size <= 0:
            problems.append("wrist RGB video empty")
        else:
            try:
                import cv2

                capture = cv2.VideoCapture(str(path))
                counted = 0
                try:
                    while True:
                        ok, _ = capture.read()
                        if not ok:
                            break
                        counted += 1
                finally:
                    capture.release()
                detail["wrist_frames"] = counted
                if counted <= 0:
                    problems.append("wrist RGB video not decodable")
                elif detail.get("n_frames") and abs(
                    counted - int(detail["n_frames"])
                ) > 1:
                    problems.append(
                        f"wrist RGB frames {counted} != T {detail['n_frames']}")
            except Exception as error:  # noqa: BLE001
                problems.append(f"wrist RGB decode failed: {error}")
    return {"detail": detail, "problems": problems, "passed": not problems}


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
def run_attempt(payload: dict[str, Any]) -> dict[str, Any]:
    """One scientific attempt through the full datagen pipeline."""
    for name, value in THREAD_POOL_ENV.items():
        os.environ.setdefault(name, value)
    if sys.platform == "darwin":
        os.environ.pop("MUJOCO_GL", None)
        os.environ.pop("PYOPENGL_PLATFORM", None)
    else:
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        os.environ.pop("DISPLAY", None)
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("MLSPACES_ASSETS_DIR", str(ROOT / "assets"))
    os.environ.setdefault("PACT_CONTACT_AUDIT_SUMMARY_ONLY", "1")
    os.environ.pop("DISPLAY", None)

    row = payload["row"]
    destination = Path(payload["row_dir"])
    destination.mkdir(parents=True, exist_ok=True)
    started = time.time()

    from molmo_spaces.data_generation.pipeline import (
        ParallelRolloutRunner, cleanup_episode_resources, setup_policy,
    )
    from molmo_spaces.tasks.task_sampler_errors import HouseInvalidForTask
    from run_pact_place_expert_screen import (
        _make_config, disallowed_initial_contacts,
        initial_robot_environment_contacts,
    )

    task = policy = sampler = None
    retry_history: list[dict[str, Any]] = []
    result: dict[str, Any] = {}
    try:
        # The datagen configuration: identical physics to the screen, with the
        # observation reductions removed so the trainable suite is recorded.
        config = _make_config(destination / "datagen.json",
                              scene_xml=Path(payload["scene_xml"]),
                              sampler_class=SAMPLER_CLASS)
        config.output_dir = destination
        config.proximity_sensor_period_ms = PROXIMITY_SENSOR_PERIOD_MS
        config.task_horizon = TASK_HORIZON
        configured = [spec.name for spec in config.camera_config.cameras]
        if configured[0] != "wrist_camera":
            raise RuntimeError("runtime camera 0 is not the wrist camera")
        if len(configured) - 1 != N_PROXIMITY_SENSORS:
            raise RuntimeError(
                f"{len(configured) - 1} proximity cameras, need "
                f"{N_PROXIMITY_SENSORS}")
        if not config.proximity_sensor_period_ms > 0:
            raise RuntimeError("proximity sub-step recording is disabled")

        selected_seed = None
        initial_reset_result = None
        for retry_index in range(int(row["max_sampling_retries"]) + 1):
            if retry_index == 0:
                seed = {"seed_u32": int(row["task_seed_u32"]),
                        "seed_u64": int(row["task_seed_u64"])}
            else:
                from pact_place_v108_contract import cell_seed

                seed = cell_seed(row["family_id"], row["intrusion_side"],
                                 row["pose_id"],
                                 int(row["attempt_index"]) * 1000 + retry_index)
            sampler = config.task_sampler_config.task_sampler_class(config)
            sampler.seed_task_sampling(int(seed["seed_u32"]))
            sampler.set_pact_manifest_row(row)
            try:
                task = sampler.sample_task(house_index=1)
                if task is None:
                    raise HouseInvalidForTask("sample_task returned None")
                policy = setup_policy(config, task, None, None)
                initial_reset_result = task.reset()
                rejected = disallowed_initial_contacts(
                    initial_robot_environment_contacts(task.env))
                if rejected:
                    raise HouseInvalidForTask(
                        f"initial_robot_environment_contact n={len(rejected)}")
            except Exception as error:  # noqa: BLE001 - pre-boundary only
                retry_history.append({
                    "retry_index": retry_index,
                    "reason": f"pre_boundary:{type(error).__name__}:{error}"[:200]})
                cleanup_episode_resources(
                    task=task, policy=policy, task_sampler=sampler,
                    preloaded_policy=None, close_task_sampler=True)
                task = policy = sampler = None
                continue
            selected_seed = seed
            break

        if selected_seed is None:
            result = {"status": "sampling_failure", "task_success": False,
                      "retry_history": retry_history}
        else:
            task_success = bool(ParallelRolloutRunner.run_single_rollout(
                episode_seed=int(selected_seed["seed_u64"]), task=task,
                policy=policy, end_on_success=False,
                initial_reset_result=initial_reset_result))
            info = policy.get_info()
            trajectory = info.pop("trajectory", [])
            audit = info.get("pact_contact_audit") or {}
            telemetry = info.get("pact_v106_frame_telemetry") or {}
            result = {
                "status": "complete",
                "selected_seed": selected_seed,
                "retry_history": retry_history,
                "task_success": task_success,
                "grasp_phase_success": bool(info.get("grasp_phase_success")),
                "place_phase_success": bool(info.get("place_phase_success")),
                "cup_lifted_one_cm": bool(info.get("cup_lifted_one_cm")),
                "contact_audit": audit,
                "clutter_stability_events": list(
                    info.get("clutter_stability_events") or []),
                "pact_v106_frame_telemetry": telemetry,
                "episode_steps": int(task.episode_step_count),
                "trajectory_n": len(trajectory),
            }
            result["v108_defects"] = row_defects(result)
            result["v108_clean_success"] = not result["v108_defects"]
            if result["v108_clean_success"]:
                published = _publish_episode(
                    destination=destination, task=task, config=config, row=row)
                result.update(published)
    except Exception as error:  # noqa: BLE001 - terminal ledger entry
        result = {
            "status": "infrastructure_failure",
            "task_success": False,
            "error": f"{type(error).__name__}: {error}"[:400],
            "traceback": traceback.format_exc()[-1500:],
            "retry_history": retry_history,
        }
    finally:
        cleanup_episode_resources(
            task=task, policy=policy, task_sampler=sampler,
            preloaded_policy=None, close_task_sampler=sampler is not None)

    result.setdefault("v108_defects", row_defects(result))
    result.setdefault("v108_clean_success", not result["v108_defects"])
    result.update({
        "attempt_id": row["attempt_id"], "cell": row["cell"],
        "family_id": row["family_id"], "intrusion_side": row["intrusion_side"],
        "pose_id": row["pose_id"], "attempt_index": int(row["attempt_index"]),
        "task_seed_u32": int(row["task_seed_u32"]),
        "row_sha256": row["row_sha256"],
        "row_dir": str(destination.relative_to(ROOT)),
        "elapsed_s": time.time() - started,
    })
    result["result_sha256"] = sha256_payload(
        {k: v for k, v in result.items() if k != "result_sha256"})
    (destination / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str))
    return result


def _publish_episode(*, destination: Path, task, config, row) -> dict[str, Any]:
    from molmo_spaces.utils.save_utils import (
        prepare_episode_for_saving, save_trajectories,
    )

    def jsonable(value):
        import numpy as np

        if isinstance(value, dict):
            return {str(k): jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [jsonable(v) for v in value]
        if isinstance(value, (np.bool_,)):
            return bool(value)
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value

    history = task.get_history()
    if "obs_scene" in history:
        history["obs_scene"] = jsonable(history["obs_scene"])
    staging = Path(tempfile.mkdtemp(prefix=".staging.", dir=destination))
    try:
        prepared = prepare_episode_for_saving(
            history, task.sensor_suite, fps=config.fps, save_dir=staging,
            episode_idx=0, save_file_suffix="")
        if prepared is None:
            raise RuntimeError("episode produced no saveable observations")
        save_trajectories([prepared], save_dir=str(staging), fps=config.fps,
                          save_file_suffix="", save_mp4s=True)
        staged = staging / "trajectories.h5"
        if not staged.exists():
            raise RuntimeError("save_trajectories wrote no trajectories.h5")
        final = destination / "trajectory.h5"
        if final.exists():
            raise RuntimeError(f"refusing to overwrite {final}")
        os.replace(staged, final)
        videos = []
        for artifact in sorted(staging.iterdir()):
            target = destination / artifact.name
            if target.exists():
                raise RuntimeError(f"refusing to overwrite {target}")
            os.replace(artifact, target)
            videos.append(target.name)
        return {"trajectory_h5": str(final.relative_to(ROOT)),
                "trajectory_h5_sha256": sha256_file(final),
                "videos": videos, "published": True}
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def prune_failed(row_dir: Path) -> dict[str, Any]:
    """Drop heavy staging for a rejected attempt; keep the compact record."""
    removed, freed = [], 0
    for item in sorted(row_dir.iterdir()):
        if item.name == "result.json":
            continue
        try:
            size = item.stat().st_size if item.is_file() else 0
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
            else:
                item.unlink()
            removed.append(item.name)
            freed += size
        except OSError:
            pass
    return {"removed": removed, "freed_bytes": freed}


# ---------------------------------------------------------------------------
# Ledger (append-only, crash-safe, never rewritten)
# ---------------------------------------------------------------------------
class Ledger:
    """One fsynced JSONL line per attempt, written before staging is pruned."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: dict[str, Any]) -> str:
        line = json.dumps(record, sort_keys=True, default=str)
        digest = sha256_payload(record)
        with open(self.path, "a") as stream:
            stream.write(line + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        return digest

    def read(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        out = []
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                # A torn final line from a crash: ignore it rather than
                # letting it corrupt resume state.
                continue
        return out


def compact_record(result: dict[str, Any]) -> dict[str, Any]:
    """Everything the ledger must keep about one attempt."""
    telemetry = result.get("pact_v106_frame_telemetry") or {}
    audit = result.get("contact_audit") or {}
    totals = audit.get("contact_class_totals") or {}
    return {
        "attempt_id": result.get("attempt_id"),
        "cell": result.get("cell"),
        "family_id": result.get("family_id"),
        "intrusion_side": result.get("intrusion_side"),
        "pose_id": result.get("pose_id"),
        "attempt_index": result.get("attempt_index"),
        "task_seed_u32": result.get("task_seed_u32"),
        "row_sha256": result.get("row_sha256"),
        "status": result.get("status"),
        "accepted": bool(result.get("v108_accepted")),
        "clean_success": bool(result.get("v108_clean_success")),
        "defects": result.get("v108_defects") or [],
        "contact_class_totals": {k: int(v) for k, v in totals.items()},
        "clutter_stability_events": len(
            result.get("clutter_stability_events") or []),
        "episode_steps": result.get("episode_steps"),
        "min_pendant_clearance_m": telemetry.get("min_clearance_m"),
        "min_lobe_stem_clearance_m": telemetry.get("min_lobe_stem_clearance_m"),
        "pendant_contact_frames": telemetry.get(
            "pendant_robot_or_target_contact_frames"),
        "trajectory_h5": result.get("trajectory_h5"),
        "trajectory_h5_sha256": result.get("trajectory_h5_sha256"),
        "schema_validation": result.get("v108_schema_validation"),
        "elapsed_s": result.get("elapsed_s"),
        "error": result.get("error"),
        "result_sha256": result.get("result_sha256"),
    }


def tally(records: list[dict[str, Any]]) -> tuple[dict[str, int], dict[str, int]]:
    accepted: dict[str, int] = {}
    attempted: dict[str, int] = {}
    for record in records:
        key = record.get("cell")
        if not key:
            continue
        attempted[key] = attempted.get(key, 0) + 1
        if record.get("accepted"):
            accepted[key] = accepted.get(key, 0) + 1
    return accepted, attempted


def taxonomy(records: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for record in records:
        if record.get("accepted"):
            continue
        defects = record.get("defects") or [record.get("status") or "unknown"]
        for defect in defects:
            head = str(defect).split("=")[0]
            out[head] = out.get(head, 0) + 1
    return dict(sorted(out.items()))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection-root", type=Path,
                        default=ROOT / COLLECTION_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / DATASET_ROOT)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--smoke-only", action="store_true",
                        help="run only the first registered attempt")
    parser.add_argument("--max-attempts", type=int, default=MAX_SCIENTIFIC_ATTEMPTS)
    args = parser.parse_args()
    pin_threads()          # before any worker exists
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("MLSPACES_ASSETS_DIR", str(ROOT / "assets"))
    os.environ.pop("DISPLAY", None)
    started = time.time()

    collection_root = args.collection_root.resolve()
    dataset_root = args.dataset_root.resolve()
    rows_root = dataset_root / "rows"
    rows_root.mkdir(parents=True, exist_ok=True)
    collection_root.mkdir(parents=True, exist_ok=True)

    disk = disk_preflight(dataset_root)
    cgroup = cgroup_pid_preflight(args.workers)
    scene = scene_preflight()
    preflight = {
        "disk": disk, "cgroup_pid": cgroup, "scene": scene,
        "thread_pool_pinned_before_workers": True,
        "thread_pool_env": dict(THREAD_POOL_ENV),
        "passed": bool(disk["passed"] and cgroup["passed"] and scene["passed"]),
    }
    if args.preflight_only or not preflight["passed"]:
        print(json.dumps(preflight, indent=2, default=str))
        if not preflight["passed"]:
            print("PREFLIGHT FAILED - not starting collection", flush=True)
        return 0 if preflight["passed"] else 1

    contract = build_contract()
    contract_path = collection_root / "contract.json"
    if not contract_path.exists():
        write_immutable_create_only(contract_path, contract)
    frozen = json.loads(contract_path.read_text())

    ledger = Ledger(collection_root / "ledger.jsonl")
    records = ledger.read()
    accepted, attempted = tally(records)
    print(json.dumps({
        "resume": True, "ledger_records": len(records),
        "accepted": sum(accepted.values()), "attempted": sum(attempted.values()),
    }), flush=True)

    selected = scene["selected"]
    scene_by_pose = scene["scenes"]
    context = multiprocessing.get_context("spawn")
    budget = min(int(args.max_attempts), MAX_SCIENTIFIC_ATTEMPTS)
    deadline = started + MAX_WALL_CLOCK_HOURS * 3600.0
    stop_reason = None
    reported_first_batch = False
    infrastructure: list[dict[str, Any]] = []
    halted: dict[str, Any] | None = None

    while True:
        if quotas_met(accepted):
            stop_reason = "quotas_met"
            break
        used = sum(attempted.values())
        if used >= budget:
            stop_reason = "attempt_budget_exhausted"
            break
        if time.time() >= deadline:
            stop_reason = "wall_clock_budget_exhausted"
            break
        batch_size = 1 if args.smoke_only else min(
            args.workers, budget - used,
            max(1, sum(remaining_quota(accepted).values())))
        batch = next_attempts(accepted, attempted, batch_size)
        if not batch:
            stop_reason = "no_schedulable_cells"
            break
        payloads = []
        for attempt in batch:
            row = build_attempt_row(attempt, selected=selected,
                                    scene_by_pose=scene_by_pose)
            payloads.append({
                "row": row,
                "row_dir": str(rows_root / attempt["attempt_id"][:16]),
                "scene_xml": str(ROOT / scene_by_pose[attempt["pose_id"]]["relative"]),
            })
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max(1, len(payloads)), mp_context=context,
            max_tasks_per_child=1,
        ) as executor:
            futures = {executor.submit(run_attempt, p): p for p in payloads}
            for future in concurrent.futures.as_completed(futures):
                payload = futures[future]
                row = payload["row"]
                try:
                    result = future.result()
                except BaseException as error:  # noqa: BLE001
                    result = {
                        "status": "infrastructure_failure",
                        "error": f"worker died: {type(error).__name__}: {error}"[:300],
                        "attempt_id": row["attempt_id"], "cell": row["cell"],
                        "family_id": row["family_id"],
                        "intrusion_side": row["intrusion_side"],
                        "pose_id": row["pose_id"],
                        "attempt_index": int(row["attempt_index"]),
                        "task_seed_u32": int(row["task_seed_u32"]),
                        "row_sha256": row["row_sha256"],
                        "v108_clean_success": False,
                        "v108_defects": ["worker_died"],
                    }
                row_dir = Path(payload["row_dir"])
                if result.get("v108_clean_success") and result.get("published"):
                    validation = validate_trainable(row_dir)
                    result["v108_schema_validation"] = validation
                    result["v108_accepted"] = bool(validation["passed"])
                    if not validation["passed"]:
                        result["v108_defects"] = (
                            result.get("v108_defects") or []) + [
                            f"schema:{p}" for p in validation["problems"][:3]]
                        result["v108_infrastructure_defect"] = "schema"
                else:
                    result["v108_accepted"] = False
                    if result.get("status") == "infrastructure_failure":
                        result["v108_infrastructure_defect"] = "worker"

                # A schema or infrastructure defect is not a scientific
                # outcome. It is recorded separately, the cell's stream is NOT
                # advanced, the row is NOT replaced, and execution stops for
                # repair -- the frozen contract authorizes no retry.
                if result.get("v108_infrastructure_defect"):
                    infrastructure.append(compact_record(result))
                    halt = {
                        "reason": "infrastructure_or_schema_defect",
                        "defect": result["v108_infrastructure_defect"],
                        "attempt_id": row["attempt_id"], "cell": row["cell"],
                        "attempt_index": int(row["attempt_index"]),
                        "scientific_stream_advanced": False,
                        "row_replaced": False,
                        "detail": (result.get("v108_defects") or [])[:4],
                        "error": result.get("error"),
                    }
                    print(json.dumps({"HALT": halt}), flush=True)
                    stop_reason = "infrastructure_or_schema_defect"
                    halted = halt
                    continue

                record = compact_record(result)
                ledger.append(record)          # durable BEFORE pruning
                if not result["v108_accepted"]:
                    prune_failed(row_dir)
                key = row["cell"]
                attempted[key] = attempted.get(key, 0) + 1
                if result["v108_accepted"]:
                    accepted[key] = accepted.get(key, 0) + 1
                print(json.dumps({
                    "attempt": sum(attempted.values()),
                    "accepted_total": sum(accepted.values()),
                    "cell": key, "status": result.get("status"),
                    "acc": result["v108_accepted"],
                    "defects": (result.get("v108_defects") or [])[:2],
                }), flush=True)

        if halted is not None:
            break
        if args.smoke_only:
            stop_reason = stop_reason or "smoke_only"
            break
        if not reported_first_batch and sum(attempted.values()) >= REPORT_AFTER_ATTEMPTS:
            reported_first_batch = True
            elapsed = time.time() - started
            n = sum(attempted.values())
            yield_rate = sum(accepted.values()) / n if n else 0.0
            need = TARGET_SUCCESSES - sum(accepted.values())
            eta_h = (need / (yield_rate or 1e-9)) * (elapsed / n) / 3600.0
            print(json.dumps({
                "checkpoint": "first_24_attempts",
                "attempts": n, "accepted": sum(accepted.values()),
                "yield": round(yield_rate, 4),
                "elapsed_h": round(elapsed / 3600.0, 3),
                "attempts_per_hour": round(n / (elapsed / 3600.0), 1),
                "disk_free_gb": round(
                    shutil.disk_usage(dataset_root).free / 1e9, 1),
                "dataset_gb": round(sum(
                    f.stat().st_size for f in rows_root.rglob("*")
                    if f.is_file()) / 1e9, 2),
                "eta_remaining_h": round(eta_h, 2),
            }, indent=2), flush=True)

    records = ledger.read()
    accepted, attempted = tally(records)
    elapsed = time.time() - started
    summary = {
        "schema_version": "pact_place_v108_collection_v1",
        "contract_version": CONTRACT_VERSION_V108,
        "environment_version": ENVIRONMENT_VERSION,
        "is_phase0_pass": False,
        "is_exploratory_owner_override": True,
        "v107_phase0_result": "failed_8_of_24_permanently_closed",
        "contract_payload_sha256": frozen.get("payload_sha256"),
        "preflight": preflight,
        "stop_reason": stop_reason,
        "target_successes": TARGET_SUCCESSES,
        "accepted_total": sum(accepted.values()),
        "attempts_total": sum(attempted.values()),
        "quotas": quota_totals(),
        "accepted_by_cell": accepted,
        "attempted_by_cell": attempted,
        "remaining_quota": remaining_quota(accepted),
        "quotas_met": quotas_met(accepted),
        "failure_taxonomy": taxonomy(records),
        "infrastructure_defects": infrastructure,
        "n_infrastructure_defects": len(infrastructure),
        "halted_for_repair": halted,
        "infrastructure_defects_excluded_from_scientific_attempts": True,
        "infrastructure_retries_authorized_by_contract": False,
        "ledger_path": str((collection_root / "ledger.jsonl").relative_to(ROOT)),
        "ledger_records": len(records),
        "ledger_sha256": sha256_file(collection_root / "ledger.jsonl")
        if (collection_root / "ledger.jsonl").is_file() else None,
        "dataset_root": str(dataset_root.relative_to(ROOT)),
        "elapsed_hours": elapsed / 3600.0,
        "budget": {"max_scientific_attempts": MAX_SCIENTIFIC_ATTEMPTS,
                   "max_wall_clock_hours": MAX_WALL_CLOCK_HOURS,
                   "extended": False},
        "root_source_commit": _git_head(ROOT),
        "molmospaces_source_commit": _git_head(MOLMO),
        **empty_authorization(),
        "authorizes_conversion": False,
        "authorizes_training": False,
        "authorizes_evaluation": False,
    }
    out = collection_root / "collection.json"
    if out.exists():
        out = collection_root / f"collection_{int(time.time())}.json"
    write_immutable_create_only(out, summary)
    print(json.dumps({
        "stop_reason": stop_reason,
        "accepted": summary["accepted_total"],
        "attempts": summary["attempts_total"],
        "quotas_met": summary["quotas_met"],
        "elapsed_hours": round(summary["elapsed_hours"], 3),
        "summary_path": str(out.relative_to(ROOT)),
    }, indent=2), flush=True)
    return 0 if summary["quotas_met"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
