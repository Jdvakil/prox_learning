#!/usr/bin/env python3
"""Collect exactly ten V10.10 demonstrations with exterior-camera calibration.

This is a small schema-validation pack for a collaborator.  It does not alter
or append to the frozen 144-row V10.10 training corpus.  Physics, task sampler,
expert policy, clutter, pendant scenes, and clean-success admission are the
V10.10 path; the only observation change is replacing the wrist-only camera
system with the established hybrid camera system, which adds ``exo_camera_1``.

An accepted row must contain a decodable ``exo_camera_1`` RGB MP4 and the
standard per-frame ``extrinsic_cv``, ``cam2world_gl``, and ``intrinsic_cv``
datasets.  The ten registered cells are balanced 5/5 by side and span every
V10.10 family and pendant pose.  Failed attempts are retained only as compact
``result.json`` records.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing
import os
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "scripts", ROOT / "submodules" / "molmospaces"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from pact_place_v1010_contract import (  # noqa: E402
    ENVIRONMENT_VERSION,
    SAMPLER_CLASS,
    SCENE_BY_POSE,
    build_row,
    canonical_payload_sha256,
    cell_key,
    cell_seed,
    sha256_file,
    sha256_payload,
    write_immutable_create_only,
)

OUTPUT_ROOT = ROOT / "diagnostics_output" / "pact_place_v1010_tablecam_validation10"
DATASET_ROOT = ROOT / "assets" / "datagen" / "pact_place_corridor_v10_10_tablecam_validation10"
BASE_LEDGER = ROOT / "diagnostics_output" / "pact_place_v1010_collection" / "ledger.jsonl"
TARGET_SUCCESSES = 10
MAX_ATTEMPTS = 100
MAX_WALL_HOURS = 6.0
MIN_FREE_GIB = 6.0
CAMERA_NAME = "exo_camera_1"
CALIBRATION_KEYS = ("extrinsic_cv", "cam2world_gl", "intrinsic_cv")

# Five left, five right; neg5/center/pos5 = 4/3/3; F0/F1/F2/F3 = 3/3/2/2.
TARGET_CELLS = (
    ("F0_target_side_stagger", "left", "neg5"),
    ("F0_target_side_stagger", "right", "center"),
    ("F1_inner_panel_stagger", "left", "center"),
    ("F1_inner_panel_stagger", "right", "pos5"),
    ("F2_outer_panel_stagger", "left", "pos5"),
    ("F2_outer_panel_stagger", "right", "neg5"),
    ("F3_aperture_side_stagger", "left", "neg5"),
    ("F3_aperture_side_stagger", "right", "center"),
    ("F0_target_side_stagger", "left", "pos5"),
    ("F1_inner_panel_stagger", "right", "neg5"),
)
TARGET_KEYS = tuple(cell_key(*parts) for parts in TARGET_CELLS)

THREAD_ENV = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _next_indices(base: list[dict[str, Any]], current: list[dict[str, Any]]) -> dict[str, int]:
    out = {key: 0 for key in TARGET_KEYS}
    for record in [*base, *current]:
        key = str(record.get("cell", ""))
        if key in out and record.get("attempt_index") is not None:
            out[key] = max(out[key], int(record["attempt_index"]) + 1)
    return out


def validate_table_camera(row_dir: Path) -> dict[str, Any]:
    """Validate the exact schema requested by the collaborator."""
    import cv2
    import h5py
    import numpy as np

    problems: list[str] = []
    detail: dict[str, Any] = {}
    h5_path = row_dir / "trajectory.h5"
    if not h5_path.is_file():
        return {"passed": False, "problems": ["trajectory.h5 missing"], "detail": detail}

    expected_shapes = {
        "extrinsic_cv": (3, 4),
        "cam2world_gl": (4, 4),
        "intrinsic_cv": (3, 3),
    }
    try:
        with h5py.File(h5_path, "r") as handle:
            group_path = f"traj_0/obs/sensor_param/{CAMERA_NAME}"
            if group_path not in handle:
                return {
                    "passed": False,
                    "problems": [f"missing {group_path}"],
                    "detail": detail,
                }
            group = handle[group_path]
            frame_count = None
            for name in CALIBRATION_KEYS:
                if name not in group:
                    problems.append(f"missing {group_path}/{name}")
                    continue
                values = np.asarray(group[name][...])
                detail[f"{name}_shape"] = list(values.shape)
                detail[f"{name}_dtype"] = str(values.dtype)
                if values.ndim != 3 or tuple(values.shape[1:]) != expected_shapes[name]:
                    problems.append(f"{name} shape {values.shape} is invalid")
                    continue
                if frame_count is None:
                    frame_count = int(values.shape[0])
                elif int(values.shape[0]) != frame_count:
                    problems.append(f"{name} frame count differs")
                if not np.all(np.isfinite(values)):
                    problems.append(f"{name} contains non-finite values")
                else:
                    detail[f"{name}_within_episode_max_delta"] = float(
                        np.max(np.abs(values - values[0]))
                    )
            detail["h5_frame_count"] = frame_count
    except Exception as error:  # noqa: BLE001
        return {
            "passed": False,
            "problems": [f"HDF5 validation raised {type(error).__name__}: {error}"],
            "detail": detail,
        }

    videos = sorted(row_dir.glob(f"episode_*_{CAMERA_NAME}.mp4"))
    detail["rgb_videos"] = [item.name for item in videos]
    if len(videos) != 1:
        problems.append(f"expected one {CAMERA_NAME} RGB MP4, found {len(videos)}")
    else:
        video = videos[0]
        detail["rgb_video_bytes"] = video.stat().st_size
        detail["rgb_video_sha256"] = sha256_file(video)
        capture = cv2.VideoCapture(str(video))
        decoded = 0
        try:
            while True:
                ok, _ = capture.read()
                if not ok:
                    break
                decoded += 1
        finally:
            capture.release()
        detail["rgb_video_frames"] = decoded
        if decoded <= 0:
            problems.append(f"{CAMERA_NAME} RGB MP4 is not decodable")
        elif detail.get("h5_frame_count") and abs(
            decoded - int(detail["h5_frame_count"])
        ) > 1:
            problems.append(
                f"RGB frames {decoded} != calibration frames {detail['h5_frame_count']}"
            )

    depth = sorted(row_dir.glob(f"episode_*_{CAMERA_NAME}_depth.mp4"))
    detail["depth_videos"] = [item.name for item in depth]
    return {"passed": not problems, "problems": problems, "detail": detail}


def run_attempt(payload: dict[str, Any]) -> dict[str, Any]:
    """V10.10 attempt with only the camera system changed."""
    for name, value in THREAD_ENV.items():
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
    os.environ.setdefault("PACT_CONTACT_AUDIT_SUMMARY_ONLY", "0")
    os.environ.pop("DISPLAY", None)

    import run_pact_place_v108_collect as v108
    from molmo_spaces.configs.camera_configs import (
        FrankaSkinHybridCameraSystem,
        FrankaSkinHybridWristOnlyCameraSystem,
    )
    from molmo_spaces.data_generation.pipeline import (
        ParallelRolloutRunner,
        cleanup_episode_resources,
        setup_policy,
    )
    from molmo_spaces.tasks.task_sampler_errors import HouseInvalidForTask
    from run_pact_place_expert_screen import (
        _make_config,
        disallowed_initial_contacts,
        initial_robot_environment_contacts,
    )
    from pact_place_v108_contract import (
        PROXIMITY_SENSOR_PERIOD_MS,
        TASK_HORIZON,
        row_defects,
    )

    row = payload["row"]
    destination = Path(payload["row_dir"])
    destination.mkdir(parents=True, exist_ok=True)
    started = time.time()
    task = policy = sampler = None
    retry_history: list[dict[str, Any]] = []
    result: dict[str, Any] = {}
    try:
        config = _make_config(
            destination / "datagen.json",
            scene_xml=Path(payload["scene_xml"]),
            sampler_class=SAMPLER_CLASS,
        )
        wrist_only = FrankaSkinHybridWristOnlyCameraSystem()
        table_system = FrankaSkinHybridCameraSystem()
        old_names = [spec.name for spec in wrist_only.cameras]
        new_names = [spec.name for spec in table_system.cameras]
        if old_names[0] != "wrist_camera" or new_names[:2] != [
            "wrist_camera", CAMERA_NAME
        ]:
            raise RuntimeError("camera-system prefix changed")
        if new_names[2:] != old_names[1:]:
            raise RuntimeError("adding the table camera changed the proximity suite")
        config.camera_config = table_system
        config.output_dir = destination
        config.proximity_sensor_period_ms = PROXIMITY_SENSOR_PERIOD_MS
        config.task_horizon = TASK_HORIZON
        if not config.proximity_sensor_period_ms > 0:
            raise RuntimeError("proximity sub-step recording is disabled")

        selected_seed = None
        initial_reset_result = None
        for retry_index in range(int(row["max_sampling_retries"]) + 1):
            if retry_index == 0:
                seed = {
                    "seed_u32": int(row["task_seed_u32"]),
                    "seed_u64": int(row["task_seed_u64"]),
                }
            else:
                seed = cell_seed(
                    row["family_id"], row["intrusion_side"], row["pose_id"],
                    int(row["attempt_index"]) * 1000 + retry_index,
                )
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
                    initial_robot_environment_contacts(task.env)
                )
                if rejected:
                    raise HouseInvalidForTask(
                        f"initial_robot_environment_contact n={len(rejected)}"
                    )
            except Exception as error:  # noqa: BLE001
                retry_history.append({
                    "retry_index": retry_index,
                    "reason": f"pre_boundary:{type(error).__name__}:{error}"[:200],
                })
                cleanup_episode_resources(
                    task=task, policy=policy, task_sampler=sampler,
                    preloaded_policy=None, close_task_sampler=True,
                )
                task = policy = sampler = None
                continue
            selected_seed = seed
            break

        if selected_seed is None:
            result = {
                "status": "sampling_failure",
                "task_success": False,
                "retry_history": retry_history,
            }
        else:
            task_success = bool(ParallelRolloutRunner.run_single_rollout(
                episode_seed=int(selected_seed["seed_u64"]),
                task=task,
                policy=policy,
                end_on_success=False,
                initial_reset_result=initial_reset_result,
            ))
            info = policy.get_info()
            trajectory = info.pop("trajectory", [])
            result = {
                "status": "complete",
                "selected_seed": selected_seed,
                "retry_history": retry_history,
                "task_success": task_success,
                "grasp_phase_success": bool(info.get("grasp_phase_success")),
                "place_phase_success": bool(info.get("place_phase_success")),
                "cup_lifted_one_cm": bool(info.get("cup_lifted_one_cm")),
                "contact_audit": info.get("pact_contact_audit") or {},
                "clutter_stability_events": list(
                    info.get("clutter_stability_events") or []
                ),
                "pact_v106_frame_telemetry": (
                    info.get("pact_v106_frame_telemetry") or {}
                ),
                "episode_steps": int(task.episode_step_count),
                "trajectory_n": len(trajectory),
            }
            result["v108_defects"] = row_defects(result)
            result["v108_clean_success"] = not result["v108_defects"]
            # Keep H5/mp4 for both accepts and scientific rejects so failed
            # rollouts stay in the dataset. Schema gates still apply only to
            # clean successes.
            try:
                result.update(v108._publish_episode(  # noqa: SLF001
                    destination=destination, task=task, config=config, row=row
                ))
            except Exception as publish_error:  # noqa: BLE001
                result["publish_error"] = (
                    f"{type(publish_error).__name__}: {publish_error}"[:400]
                )
                if result["v108_clean_success"]:
                    raise
            else:
                if result["v108_clean_success"]:
                    result["base_schema_validation"] = v108.validate_trainable(
                        destination
                    )
                    result["table_camera_validation"] = validate_table_camera(
                        destination
                    )
    except Exception as error:  # noqa: BLE001
        result = {
            "status": "infrastructure_failure",
            "task_success": False,
            "error": f"{type(error).__name__}: {error}"[:400],
            "traceback": traceback.format_exc()[-1800:],
            "retry_history": retry_history,
        }
    finally:
        cleanup_episode_resources(
            task=task,
            policy=policy,
            task_sampler=sampler,
            preloaded_policy=None,
            close_task_sampler=sampler is not None,
        )

    result.setdefault("v108_defects", row_defects(result))
    result.setdefault("v108_clean_success", not result["v108_defects"])
    result.update({
        "attempt_id": row["attempt_id"],
        "cell": row["cell"],
        "family_id": row["family_id"],
        "intrusion_side": row["intrusion_side"],
        "pose_id": row["pose_id"],
        "attempt_index": int(row["attempt_index"]),
        "task_seed_u32": int(row["task_seed_u32"]),
        "row_sha256": row["row_sha256"],
        "row_dir": str(destination.relative_to(ROOT)),
        "elapsed_s": time.time() - started,
    })
    result["accepted"] = bool(
        result.get("v108_clean_success")
        and (result.get("base_schema_validation") or {}).get("passed")
        and (result.get("table_camera_validation") or {}).get("passed")
    )
    result["result_sha256"] = sha256_payload(
        {key: value for key, value in result.items() if key != "result_sha256"}
    )
    (destination / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str)
    )
    return result


def _contract() -> dict[str, Any]:
    script = Path(__file__).resolve()
    document = {
        "schema_version": "pact_place_v1010_tablecam_validation10_v1",
        "purpose": "ten V10.10 demonstrations for collaborator camera-schema validation",
        "training_corpus_modified": False,
        "environment_version": ENVIRONMENT_VERSION,
        "sampler_class": SAMPLER_CLASS,
        "target_successes": TARGET_SUCCESSES,
        "target_cells": list(TARGET_KEYS),
        "camera_system": "FrankaSkinHybridCameraSystem",
        "added_camera": CAMERA_NAME,
        "required_h5_group": f"traj_0/obs/sensor_param/{CAMERA_NAME}",
        "required_calibration_keys": list(CALIBRATION_KEYS),
        "required_rgb_mp4": f"episode_*_{CAMERA_NAME}.mp4",
        "base_ledger": str(BASE_LEDGER.relative_to(ROOT)),
        "base_ledger_sha256": sha256_file(BASE_LEDGER),
        "script": str(script.relative_to(ROOT)),
        "script_sha256": sha256_file(script),
        "created_utc": utc_now(),
    }
    document["payload_sha256"] = canonical_payload_sha256(document)
    return document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    for name, value in THREAD_ENV.items():
        os.environ[name] = value

    rows_root = DATASET_ROOT / "rows"
    rows_root.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    contract_path = OUTPUT_ROOT / "contract.json"
    if not contract_path.exists():
        write_immutable_create_only(contract_path, _contract())

    ledger_path = OUTPUT_ROOT / "ledger.jsonl"
    base_records = _read_jsonl(BASE_LEDGER)
    records = _read_jsonl(ledger_path)
    accepted = {key: 0 for key in TARGET_KEYS}
    for record in records:
        key = str(record.get("cell", ""))
        if key in accepted and record.get("accepted"):
            accepted[key] += 1
    next_index = _next_indices(base_records, records)

    if args.verify_only:
        print(json.dumps({
            "accepted": sum(accepted.values()),
            "target": TARGET_SUCCESSES,
            "accepted_by_cell": accepted,
            "ledger_records": len(records),
        }, indent=2))
        return 0 if sum(accepted.values()) == TARGET_SUCCESSES else 1

    started = time.monotonic()
    started_utc = utc_now()
    in_flight: set[str] = set()
    context = multiprocessing.get_context("spawn")
    attempts_this_run = 0
    stop_reason = "target_met"
    futures: dict[Any, dict[str, Any]] = {}

    print(
        f"V10.10 table-camera validation: {sum(accepted.values())}/"
        f"{TARGET_SUCCESSES} already accepted; {args.workers} workers",
        flush=True,
    )
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=context,
        max_tasks_per_child=1,
    ) as pool:
        while True:
            total = sum(accepted.values())
            if total >= TARGET_SUCCESSES and not futures:
                break
            if attempts_this_run >= int(args.max_attempts) and not futures:
                stop_reason = "attempt_budget_exhausted"
                break
            if (time.monotonic() - started) / 3600 >= MAX_WALL_HOURS and not futures:
                stop_reason = "wall_clock_exhausted"
                break
            if shutil.disk_usage("/root").free / 2**30 < MIN_FREE_GIB and not futures:
                stop_reason = "insufficient_disk"
                break

            remaining = TARGET_SUCCESSES - total
            while len(futures) < min(args.workers, remaining):
                open_cells = [
                    key for key in TARGET_KEYS
                    if accepted[key] == 0 and key not in in_flight
                ]
                if not open_cells or attempts_this_run >= int(args.max_attempts):
                    break
                key = min(open_cells, key=lambda item: (next_index[item], item))
                family, side, pose = key.split("|")
                index = next_index[key]
                row = build_row(family, side, pose, index)
                row_dir = rows_root / row["attempt_id"][:16]
                if row_dir.exists():
                    raise RuntimeError(f"refusing existing row directory {row_dir}")
                payload = {
                    "row": row,
                    "row_dir": str(row_dir),
                    "scene_xml": str(ROOT / SCENE_BY_POSE[pose]["relative"]),
                }
                next_index[key] += 1
                attempts_this_run += 1
                in_flight.add(key)
                futures[pool.submit(run_attempt, payload)] = payload

            if not futures:
                stop_reason = "no_schedulable_cells"
                break

            done, _ = concurrent.futures.wait(
                futures, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done:
                payload = futures.pop(future)
                row = payload["row"]
                key = row["cell"]
                in_flight.discard(key)
                try:
                    result = future.result()
                except BaseException as error:  # noqa: BLE001
                    result = {
                        "status": "worker_died",
                        "error": f"{type(error).__name__}: {error}"[:400],
                        "accepted": False,
                        "attempt_id": row["attempt_id"],
                        "cell": key,
                        "family_id": row["family_id"],
                        "intrusion_side": row["intrusion_side"],
                        "pose_id": row["pose_id"],
                        "attempt_index": int(row["attempt_index"]),
                    }
                if result.get("accepted") and accepted[key] == 0:
                    accepted[key] = 1
                else:
                    result["accepted"] = False
                    try:
                        import run_pact_place_v108_collect as v108

                        v108.prune_failed(Path(payload["row_dir"]))
                    except Exception:  # noqa: BLE001
                        pass
                record = {
                    "attempt_id": row["attempt_id"],
                    "cell": key,
                    "family_id": row["family_id"],
                    "intrusion_side": row["intrusion_side"],
                    "pose_id": row["pose_id"],
                    "attempt_index": int(row["attempt_index"]),
                    "task_seed_u32": int(row["task_seed_u32"]),
                    "status": result.get("status"),
                    "accepted": bool(result.get("accepted")),
                    "task_success": bool(result.get("task_success")),
                    "defects": result.get("v108_defects") or [],
                    "episode_steps": result.get("episode_steps"),
                    "elapsed_s": result.get("elapsed_s"),
                    "row_dir": result.get("row_dir"),
                    "trajectory_h5": result.get("trajectory_h5"),
                    "trajectory_h5_sha256": result.get("trajectory_h5_sha256"),
                    "table_camera_validation": result.get("table_camera_validation"),
                    "error": result.get("error"),
                    "result_sha256": result.get("result_sha256"),
                }
                _append_jsonl(ledger_path, record)
                records.append(record)
                print(
                    f"  accepted {sum(accepted.values())}/{TARGET_SUCCESSES} | "
                    f"attempts this run {attempts_this_run} | {key} | "
                    f"{'ACCEPT' if record['accepted'] else 'reject'}",
                    flush=True,
                )

    accepted_records = [record for record in records if record.get("accepted")]
    accepted_records.sort(key=lambda item: TARGET_KEYS.index(item["cell"]))
    unique_extrinsics: set[str] = set()
    for record in accepted_records:
        row_dir = ROOT / record["row_dir"]
        import h5py
        import numpy as np

        with h5py.File(row_dir / "trajectory.h5", "r") as handle:
            value = np.asarray(handle[
                f"traj_0/obs/sensor_param/{CAMERA_NAME}/cam2world_gl"
            ])[0]
        unique_extrinsics.add(hashlib.sha256(value.tobytes()).hexdigest())

    complete = sum(accepted.values()) == TARGET_SUCCESSES
    summary = {
        "schema_version": "pact_place_v1010_tablecam_validation10_closeout_v1",
        "passed": complete,
        "stop_reason": stop_reason,
        "target_successes": TARGET_SUCCESSES,
        "accepted_total": sum(accepted.values()),
        "attempts_total": len(records),
        "attempts_this_run": attempts_this_run,
        "accepted_by_cell": dict(sorted(accepted.items())),
        "target_cells": list(TARGET_KEYS),
        "unique_extrinsics": len(unique_extrinsics),
        "camera_name": CAMERA_NAME,
        "required_group": f"traj_0/obs/sensor_param/{CAMERA_NAME}",
        "dataset_root": str(DATASET_ROOT.relative_to(ROOT)),
        "ledger": str(ledger_path.relative_to(ROOT)),
        "ledger_sha256": sha256_file(ledger_path),
        "base_training_corpus_modified": False,
        "started_utc": started_utc,
        "finished_utc": utc_now(),
        "elapsed_hours": round((time.monotonic() - started) / 3600, 3),
        "disk_free_gib": round(shutil.disk_usage("/root").free / 2**30, 2),
    }
    summary["payload_sha256"] = canonical_payload_sha256(summary)
    closeout_path = OUTPUT_ROOT / "closeout.json"
    if complete and not closeout_path.exists():
        write_immutable_create_only(closeout_path, summary)
    print(json.dumps(summary, indent=2))
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
