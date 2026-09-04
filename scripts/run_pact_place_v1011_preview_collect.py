#!/usr/bin/env python3
"""Collect a 10-episode table-camera set of the V10.11 preview trajectory.

Schema matches Lundii/table_smoke pact_place_corridor_v10_10_tablecam_validation10:
wrist RGB/depth, exo_camera_1 RGB/depth, sensors heatmap, trajectory.h5, result.json.

This is a development preview collect. It does not lock V10.11, does not
overwrite V10.10, and does not authorize a gate.
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
for path in (ROOT / "scripts", ROOT / "submodules" / "molmospaces"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pact_place_v1010_contract import (  # noqa: E402
    MAX_SAMPLING_RETRIES,
    SAMPLER_CLASS,
    build_row,
    cell_seed,
    cells,
    empty_authorization,
    sha256_file,
    sha256_payload,
)
from pact_place_v108_contract import (  # noqa: E402
    N_PROXIMITY_SENSORS,
    PROXIMITY_SENSOR_PERIOD_MS,
    TASK_HORIZON,
)
from render_pact_place_v1011_clutter_preview import (  # noqa: E402
    SCENE,
    STANDING_KITCHEN,
    extras_overlap_motion_lane,
    _attach_standing_kitchen,
    _install_preview_contact_classes,
)
from run_pact_place_v1011_preview_one_rollout import (  # noqa: E402
    _install_preview_layout,
    _keep_glass_inside_blue_tray,
    _pin_runtime,
)
from run_pact_place_v108_collect import (  # noqa: E402
    prune_failed,
    row_defects,
    validate_trainable,
    _publish_episode,
)

COLLECTION_STREAM = "pact_place_v1011_preview_tablecam"
DATASET_REL = "assets/datagen/pact_place_corridor_v10_11_preview_tablecam"
COLLECTION_REL = "diagnostics_output/pact_place_v1011_preview_tablecam"
TARGET_SUCCESSES = 3
REQUIRED_VIDEOS = (
    "episode_00000000_exo_camera_1.mp4",
    "episode_00000000_exo_camera_1_depth.mp4",
    "episode_00000000_sensors_depth8_heatmap.mp4",
    "episode_00000000_wrist_camera.mp4",
    "episode_00000000_wrist_camera_depth.mp4",
)
TABLE_CAMERA = "exo_camera_1"
TABLE_PARAM_KEYS = ("intrinsic_cv", "extrinsic_cv", "cam2world_gl")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_preview_row(family: str, side: str, pose: str, attempt_index: int) -> dict[str, Any]:
    row = build_row(family, side, pose, attempt_index)
    row["pact_v1010_scene_relative"] = str(SCENE.relative_to(ROOT))
    row["pact_v106_scene_sha256"] = hashlib.sha256(SCENE.read_bytes()).hexdigest()
    row["environment_version"] = "pact_place_corridor_v10_11_preview_onebottle"
    row["household_layout"] = "one_inbound_bottle_toward_robot"
    row["seed_stream"] = COLLECTION_STREAM
    row["max_sampling_retries"] = MAX_SAMPLING_RETRIES
    row.pop("row_sha256", None)
    row["row_sha256"] = sha256_payload(row)
    return row


def _enable_table_camera(config) -> None:
    from molmo_spaces.configs.camera_configs import FrankaSkinHybridCameraSystem

    config.camera_config = FrankaSkinHybridCameraSystem()
    names = [spec.name for spec in config.camera_config.cameras]
    proximity = [
        spec.name
        for spec in config.camera_config.cameras
        if getattr(spec, "is_proximity_sensor", False)
    ]
    if names[0] != "wrist_camera":
        raise RuntimeError("runtime camera 0 is not the wrist camera")
    if TABLE_CAMERA not in names:
        raise RuntimeError("exo_camera_1 / table camera is missing")
    if len(proximity) != N_PROXIMITY_SENSORS:
        raise RuntimeError(
            f"{len(proximity)} proximity cameras, need {N_PROXIMITY_SENSORS}"
        )


def validate_table_camera(row_dir: Path, n_frames: int | None) -> dict[str, Any]:
    import h5py
    import numpy as np

    problems: list[str] = []
    detail: dict[str, Any] = {}
    rgb = row_dir / "episode_00000000_exo_camera_1.mp4"
    depth = row_dir / "episode_00000000_exo_camera_1_depth.mp4"
    detail["rgb_videos"] = [rgb.name] if rgb.is_file() else []
    detail["depth_videos"] = [depth.name] if depth.is_file() else []
    if not rgb.is_file() or rgb.stat().st_size <= 0:
        problems.append("missing exo_camera_1 RGB video")
    else:
        detail["rgb_video_bytes"] = int(rgb.stat().st_size)
        detail["rgb_video_sha256"] = sha256_file(rgb)
    if not depth.is_file() or depth.stat().st_size <= 0:
        problems.append("missing exo_camera_1 depth video")
    h5_path = row_dir / "trajectory.h5"
    if not h5_path.is_file():
        return {"detail": detail, "problems": problems + ["trajectory.h5 missing"],
                "passed": False}
    with h5py.File(h5_path, "r") as handle:
        traj = handle["traj_0"]
        params = traj.get("obs/sensor_param")
        if params is None or TABLE_CAMERA not in params:
            problems.append("missing obs/sensor_param/exo_camera_1")
        else:
            group = params[TABLE_CAMERA]
            for key in TABLE_PARAM_KEYS:
                if key not in group:
                    problems.append(f"missing sensor_param/{TABLE_CAMERA}/{key}")
                    continue
                node = group[key]
                detail[f"{key}_shape"] = list(node.shape)
                detail[f"{key}_dtype"] = str(node.dtype)
                if n_frames is not None and int(node.shape[0]) != int(n_frames):
                    problems.append(
                        f"{key} frames {node.shape[0]} != T {n_frames}"
                    )
                if int(node.shape[0]) >= 2:
                    first = np.asarray(node[0])
                    last = np.asarray(node[-1])
                    detail[f"{key}_within_episode_max_delta"] = float(
                        np.max(np.abs(last - first))
                    )
            detail["h5_frame_count"] = int(
                traj["obs/agent/qpos"].shape[0]
            )
    if rgb.is_file():
        import cv2

        capture = cv2.VideoCapture(str(rgb))
        counted = 0
        try:
            while True:
                ok, _ = capture.read()
                if not ok:
                    break
                counted += 1
        finally:
            capture.release()
        detail["rgb_video_frames"] = counted
        if n_frames is not None and abs(counted - int(n_frames)) > 1:
            problems.append(
                f"exo RGB frames {counted} != T {n_frames}"
            )
    return {"detail": detail, "problems": problems, "passed": not problems}


def pin_worker_gpu(gpu_id: int) -> None:
    """Pin Warp CUDA and MuJoCo EGL to one physical card before either inits."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(int(gpu_id))
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(int(gpu_id))
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ.pop("DISPLAY", None)


def pool_worker(payload: dict[str, Any]) -> dict[str, Any]:
    pin_worker_gpu(int(payload.get("gpu_id", 0)))
    print(
        json.dumps(
            {
                "worker_pid": os.getpid(),
                "gpu_id": int(payload.get("gpu_id", 0)),
                "attempt_id": payload["row"]["attempt_id"][:16],
            }
        ),
        flush=True,
    )
    return run_attempt(payload)


def run_attempt(payload: dict[str, Any]) -> dict[str, Any]:
    _pin_runtime()
    _install_preview_contact_classes()
    os.environ["PACT_CONTACT_AUDIT_SUMMARY_ONLY"] = "1"
    import logging

    logging.getLogger("molmo_spaces.env.sensors").setLevel(logging.ERROR)

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

    row = payload["row"]
    destination = Path(payload["row_dir"])
    destination.mkdir(parents=True, exist_ok=True)
    started = time.time()
    task = policy = sampler = None
    retry_history: list[dict[str, Any]] = []
    result: dict[str, Any] = {}
    extra_bodies: list[str] = []
    try:
        config = _make_config(
            destination / "datagen.json",
            scene_xml=Path(payload["scene_xml"]),
            sampler_class=SAMPLER_CLASS,
        )
        config.output_dir = destination
        config.proximity_sensor_period_ms = PROXIMITY_SENSOR_PERIOD_MS
        config.task_horizon = TASK_HORIZON
        _enable_table_camera(config)
        if not config.proximity_sensor_period_ms > 0:
            raise RuntimeError("proximity sub-step recording is disabled")

        selected_seed = None
        initial_reset_result = None
        for retry_index in range(int(row["max_sampling_retries"]) + 1):
            extra_bodies.clear()
            if retry_index == 0:
                seed = {
                    "seed_u32": int(row["task_seed_u32"]),
                    "seed_u64": int(row["task_seed_u64"]),
                }
            else:
                seed = cell_seed(
                    row["family_id"],
                    row["intrusion_side"],
                    row["pose_id"],
                    int(row["attempt_index"]) * 1000 + retry_index,
                )
            sampler = config.task_sampler_config.task_sampler_class(config)
            original_add = sampler.add_auxiliary_objects

            def add_auxiliary_objects(spec, _original=original_add):
                _original(spec)
                extra_bodies.extend(_attach_standing_kitchen(spec))

            sampler.add_auxiliary_objects = add_auxiliary_objects  # type: ignore[method-assign]
            sampler.seed_task_sampling(int(seed["seed_u32"]))
            sampler.set_pact_manifest_row(row)
            try:
                task = sampler.sample_task(house_index=1)
                if task is None:
                    raise HouseInvalidForTask("sample_task returned None")
                policy = setup_policy(config, task, None, None)
                _keep_glass_inside_blue_tray(policy)
                original_reset = policy.reset

                def reset_with_extras(reset_retries: bool = True):
                    result = original_reset(reset_retries=reset_retries)
                    placed = _install_preview_layout(task, extra_bodies)
                    hits = extras_overlap_motion_lane(
                        task.env.current_model,
                        task.env.current_data,
                        placed,
                    )
                    behind = tuple(
                        item["uid"]
                        for item in STANDING_KITCHEN
                        if item.get("behind_grasp")
                    )
                    hits = [
                        name
                        for name in hits
                        if not any(uid in name for uid in behind)
                    ]
                    if hits:
                        raise HouseInvalidForTask(
                            "extras_in_motion_lane " + ",".join(hits[:3])
                        )
                    return result

                policy.reset = reset_with_extras  # type: ignore[method-assign]
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
                    info.get("clutter_stability_events") or []
                ),
                "pact_v106_frame_telemetry": telemetry,
                "episode_steps": int(task.episode_step_count),
                "trajectory_n": len(trajectory),
            }
            result["v108_defects"] = row_defects(result)
            result["v108_clean_success"] = not result["v108_defects"]
            if result["v108_clean_success"]:
                published = _publish_episode(
                    destination=destination, task=task, config=config, row=row
                )
                result.update(published)
                trainable = validate_trainable(destination)
                result["base_schema_validation"] = trainable
                table = validate_table_camera(
                    destination, trainable.get("detail", {}).get("n_frames")
                )
                result["table_camera_validation"] = table
                missing = [
                    name for name in REQUIRED_VIDEOS
                    if not (destination / name).is_file()
                ]
                if missing:
                    table["passed"] = False
                    table["problems"] = list(table.get("problems") or []) + [
                        f"missing videos: {missing}"
                    ]
                result["accepted"] = bool(
                    trainable.get("passed") and table.get("passed")
                )
    except Exception as error:  # noqa: BLE001
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
            preloaded_policy=None, close_task_sampler=sampler is not None,
        )

    result.setdefault("v108_defects", row_defects(result))
    result.setdefault("v108_clean_success", not result["v108_defects"])
    result.setdefault("accepted", False)
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
        "role": "development_preview_not_a_gate",
        "authorizes_collection": False,
        "schema_version": "pact_place_v1011_preview_tablecam_v1",
    })
    result["result_sha256"] = sha256_payload(
        {k: v for k, v in result.items() if k != "result_sha256"}
    )
    (destination / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str)
    )
    return result


def center_cells() -> list[tuple[str, str, str]]:
    ordered = [cell for cell in cells() if cell[2] == "center"]
    preferred = ("F0_target_side_stagger", "left", "center")
    if preferred in ordered:
        ordered.remove(preferred)
        ordered.insert(0, preferred)
    return ordered


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, default=TARGET_SUCCESSES)
    parser.add_argument("--max-attempts", type=int, default=12)
    parser.add_argument(
        "--workers",
        type=int,
        default=1 if sys.platform == "darwin" else 4,
        help="Parallel attempts. On batman pin one pair to each 4090.",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1 if sys.platform == "darwin" else 2,
        help="Physical GPU count to stripe workers across.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=ROOT / DATASET_REL,
        help="Directory that will contain rows/. Isolated from V10.10.",
    )
    parser.add_argument(
        "--collection-root",
        type=Path,
        default=ROOT / COLLECTION_REL,
        help="Directory for ledger.jsonl and closeout.json.",
    )
    args = parser.parse_args()
    workers = max(1, int(args.workers))
    n_gpus = max(1, int(args.gpus))

    _pin_runtime()
    collection = args.collection_root.expanduser().resolve()
    rows_root = args.dataset_root.expanduser().resolve() / "rows"
    collection.mkdir(parents=True, exist_ok=True)
    rows_root.mkdir(parents=True, exist_ok=True)
    ledger_path = collection / "ledger.jsonl"

    quota_cells = center_cells()
    accepted = 0
    attempted_by_cell = {f"{f}|{s}|{p}": 0 for f, s, p in quota_cells}
    started = time.monotonic()
    started_utc = utc_now()
    print(
        json.dumps(
            {
                "status": "starting_preview_tablecam_collect",
                "target": args.target,
                "workers": workers,
                "gpus": n_gpus,
                "scene": str(SCENE.relative_to(ROOT)),
                "dataset_root": _rel(rows_root.parent),
                "collection_root": _rel(collection),
                "center_cells": len(quota_cells),
                "authorizes_collection": False,
            }
        ),
        flush=True,
    )

    cell_index = 0
    submit_index = 0

    def submit_next(pool, futures: dict) -> bool:
        nonlocal cell_index, submit_index
        if sum(attempted_by_cell.values()) >= args.max_attempts:
            return False
        if accepted >= args.target:
            return False
        family, side, pose = quota_cells[cell_index % len(quota_cells)]
        cell = f"{family}|{side}|{pose}"
        attempt_index = attempted_by_cell[cell]
        attempted_by_cell[cell] += 1
        cell_index += 1
        row = build_preview_row(family, side, pose, attempt_index)
        staging = rows_root / row["attempt_id"][:16]
        gpu_id = submit_index % n_gpus
        submit_index += 1
        payload = {
            "row": row,
            "row_dir": str(staging),
            "scene_xml": str(SCENE),
            "gpu_id": gpu_id,
        }
        print(
            f"attempt {sum(attempted_by_cell.values())} gpu{gpu_id} {cell} "
            f"{staging.name}",
            flush=True,
        )
        futures[pool.submit(pool_worker, payload)] = {
            "cell": cell,
            "row": row,
            "staging": staging,
        }
        return True

    context = multiprocessing.get_context("spawn")
    with ledger_path.open("a") as ledger:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers, mp_context=context, max_tasks_per_child=1,
        ) as pool:
            futures: dict[Any, dict[str, Any]] = {}
            while True:
                while (len(futures) < workers
                       and accepted < args.target
                       and sum(attempted_by_cell.values()) < args.max_attempts):
                    if not submit_next(pool, futures):
                        break
                if not futures:
                    break
                done, _ = concurrent.futures.wait(
                    futures, return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    meta = futures.pop(future)
                    cell, row, staging = meta["cell"], meta["row"], meta["staging"]
                    try:
                        result = future.result()
                    except BaseException as error:  # noqa: BLE001
                        result = {
                            "status": "worker_died",
                            "error": f"{type(error).__name__}: {error}"[:400],
                            "accepted": False,
                            "v108_clean_success": False,
                            "v108_defects": ["worker_died"],
                        }
                    ok = bool(
                        result.get("accepted") and result.get("v108_clean_success")
                    )
                    kept = False
                    final_dir = staging
                    if ok and accepted < args.target:
                        folder = f"{accepted:03d}_{row['attempt_id'][:16]}"
                        final_dir = rows_root / folder
                        if final_dir.exists():
                            raise RuntimeError(f"refusing to overwrite {final_dir}")
                        staging.rename(final_dir)
                        result["row_dir"] = _rel(final_dir)
                        (final_dir / "result.json").write_text(
                            json.dumps(
                                result, indent=2, sort_keys=True, default=str
                            )
                        )
                        accepted += 1
                        kept = True
                        print(
                            f"  ACCEPT {accepted}/{args.target} {cell} {folder}",
                            flush=True,
                        )
                    else:
                        prune_failed(staging)
                        if ok:
                            print(
                                f"  extra-accept pruned {cell} {staging.name}",
                                flush=True,
                            )
                        else:
                            print(
                                f"  reject {cell} {result.get('status')} "
                                f"{result.get('v108_defects') or result.get('error')}",
                                flush=True,
                            )
                    record = {
                        "accepted": kept,
                        "cell": cell,
                        "attempt_id": row["attempt_id"],
                        "row_dir": _rel(final_dir),
                        "status": result.get("status"),
                        "defects": result.get("v108_defects"),
                        "task_success": result.get("task_success"),
                        "elapsed_s": result.get("elapsed_s"),
                        "table_camera_validation": (
                            result.get("table_camera_validation") or {}
                        ).get("passed"),
                    }
                    ledger.write(json.dumps(record, sort_keys=True) + "\n")
                    ledger.flush()

    closeout = {
        **empty_authorization(),
        "schema_version": "pact_place_v1011_preview_tablecam_closeout_v1",
        "role": "development_preview_not_a_gate",
        "authorizes_collection": False,
        "environment_version": "pact_place_corridor_v10_11_preview_onebottle",
        "household_layout": "one_inbound_bottle_toward_robot",
        "kept_bottle": "Soap_Bottle_11",
        "parked_outbound_bottle": "Soap_Bottle_30",
        "sampler_class": SAMPLER_CLASS,
        "target_successes": args.target,
        "accepted_total": accepted,
        "attempts_total": sum(attempted_by_cell.values()),
        "attempted_by_cell": attempted_by_cell,
        "workers": workers,
        "gpus": n_gpus,
        "dataset_root": _rel(rows_root.parent),
        "collection_root": _rel(collection),
        "started_utc": started_utc,
        "finished_utc": utc_now(),
        "elapsed_hours": round((time.monotonic() - started) / 3600, 3),
        "reference_schema": (
            "https://huggingface.co/datasets/Lundii/table_smoke/tree/main/"
            "pact_place_corridor_v10_10_tablecam_validation10/rows/"
            "000_583240cae163728d"
        ),
        "hover_then_vertical_drop": True,
        "standing_kitchen_extras": True,
    }
    (collection / "closeout.json").write_text(json.dumps(closeout, indent=2) + "\n")
    print(json.dumps(closeout, indent=2), flush=True)
    return 0 if accepted >= args.target else 1


if __name__ == "__main__":
    raise SystemExit(main())
