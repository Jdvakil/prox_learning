#!/usr/bin/env python3
"""One expert rollout on the v12 environment. Not a gate."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import cv2
import mujoco
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
_MOLMO_WORKTREE = Path(
    os.environ.get("MOLMOSPACES_PACT_V1010", "/home/jaydv/code/molmospaces-pact-v1010")
)
_MOLMO = (
    _MOLMO_WORKTREE
    if (_MOLMO_WORKTREE / "molmo_spaces").is_dir()
    else ROOT / "submodules" / "molmospaces"
)
for path in (ROOT / "scripts", _MOLMO):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pact_place_v12_contract import build_row, sha256_payload  # noqa: E402
from render_pact_place_v12_clutter import (  # noqa: E402
    SCENE,
    _apply_preview_household,
    _attach_standing_kitchen,
    _hide_primitive_colliders,
    _install_preview_contact_classes,
    _place_standing_kitchen,
    _refresh_clutter_settle,
)
from run_pact_place_v12_expert import (  # noqa: E402
    _jsonable,
    _make_config,
    derive_failure_cause,
    disallowed_initial_contacts,
    initial_robot_environment_contacts,
    place_receptacle_outside_placement,
)
from molmo_spaces.utils.mj_model_and_data_utils import body_aabb  # noqa: E402
from run_pact_place_v12_cameras import (  # noqa: E402
    THIRD_PERSON_FOV,
    WRIST_FOV,
    third_person_pose,
    wrist_camera_pose,
)

OUTPUT = ROOT / "diagnostics_output/pact_place_v12/one_rollout"
FRAME_STRIDE = 5
VIDEO_FPS = 10
SAMPLER_CLASS = "PactPlaceCorridorV1010FourObjectSampler"
# Fly in at this height above the release pose, then drop straight down.
# Stock preplace is only 7 cm and shares XY change with Z, so the glass
# bottom clips the pad rim. Keep this preview-only; do not edit the expert.
HOVER_ABOVE_PLACE_M = 0.18
# Keep the predicted glass footprint this far inside the tray well.
TRAY_WELL_MARGIN_M = 0.02


def _pin_runtime() -> None:
    for name, value in {
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
    }.items():
        os.environ.setdefault(name, value)
    if sys.platform == "darwin":
        os.environ.pop("MUJOCO_GL", None)
        os.environ.pop("PYOPENGL_PLATFORM", None)
    else:
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("MLSPACES_ASSETS_DIR", str(ROOT / "assets"))
    os.environ.setdefault("PACT_CONTACT_AUDIT_SUMMARY_ONLY", "1")
    os.environ.pop("DISPLAY", None)


def _install_preview_layout(task, extra_bodies: list[str]) -> list[str]:
    env = task.env
    model, data = env.current_model, env.current_data
    _hide_primitive_colliders(model)
    _apply_preview_household(model, data)
    _refresh_clutter_settle(task, model, data)
    placed = _place_standing_kitchen(model, data, extra_bodies)
    mujoco.mj_forward(model, data)
    extra_bodies[:] = placed
    return placed


def _tray_inner_half_xy(model: mujoco.MjModel) -> np.ndarray:
    floor = int(model.geom("place_receptacle_floor_g").id)
    half = np.asarray(model.geom_size[floor, :2], dtype=float).copy()
    try:
        lip = int(model.geom("place_receptacle_lip_left_g").id)
    except Exception:
        return half
    lip_y = abs(float(model.geom_pos[lip, 1])) - float(model.geom_size[lip, 1])
    if lip_y > 0.0:
        half[1] = min(half[1], lip_y)
    return half


def _aabb_corners(center: np.ndarray, size: np.ndarray) -> np.ndarray:
    half = 0.5 * np.asarray(size, dtype=float)
    center = np.asarray(center, dtype=float)
    corners = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                corners.append(center + (sx * half[0], sy * half[1], sz * half[2]))
    return np.asarray(corners, dtype=float)


def _shift_place_xy_into_tray(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    grasp_pose_world: np.ndarray,
    pickup_obj,
    place: np.ndarray,
) -> np.ndarray:
    floor_id = int(model.geom("place_receptacle_floor_g").id)
    floor_xy = np.asarray(data.geom_xpos[floor_id, :2], dtype=float)
    inner_half = _tray_inner_half_xy(model) - TRAY_WELL_MARGIN_M
    inner_half = np.maximum(inner_half, 0.02)
    body_id = int(pickup_obj.object_id)
    aabb_c, aabb_s = body_aabb(model, data, body_id, visual_only=False)
    obj_pose = np.eye(4)
    obj_pose[:3, 3] = np.asarray(data.xpos[body_id], dtype=float)
    obj_pose[:3, :3] = np.asarray(data.xmat[body_id], dtype=float).reshape(3, 3)
    obj_at_place = place @ np.linalg.inv(grasp_pose_world) @ obj_pose
    rel = obj_at_place @ np.linalg.inv(obj_pose)
    corners = _aabb_corners(aabb_c, aabb_s)
    placed = (rel[:3, :3] @ corners.T).T + rel[:3, 3]
    min_xy = placed[:, :2].min(axis=0)
    max_xy = placed[:, :2].max(axis=0)
    shift = np.zeros(2, dtype=float)
    for axis in (0, 1):
        lo = float(floor_xy[axis] - inner_half[axis])
        hi = float(floor_xy[axis] + inner_half[axis])
        if min_xy[axis] < lo:
            shift[axis] += lo - min_xy[axis]
        if max_xy[axis] > hi:
            shift[axis] += hi - max_xy[axis]
    return shift


def _keep_glass_inside_blue_tray(policy) -> None:
    original_placement = policy._get_placement_poses
    original_traj = policy._compute_trajectory

    def _tcp_primitives(primitives):
        for prim in primitives:
            segs = getattr(prim, "_move_segments", None)
            if segs:
                yield prim, list(segs)

    def _refresh_duration(prim) -> None:
        segs = getattr(prim, "_move_segments", None) or []
        if segs:
            prim.duration = float(sum(seg.duration for seg in segs))

    def _get_placement_poses(grasp_pose_world, pickup_obj, place_receptacle):
        preplace, place, postplace = original_placement(
            grasp_pose_world, pickup_obj, place_receptacle
        )
        preplace = preplace.copy()
        place = place.copy()
        postplace = postplace.copy()
        model, data = policy.task.env.current_model, policy.task.env.current_data
        shift_xy = _shift_place_xy_into_tray(
            model, data, grasp_pose_world, pickup_obj, place
        )
        for pose in (preplace, place, postplace):
            pose[:2, 3] = pose[:2, 3] + shift_xy
        preplace[2, 3] = float(place[2, 3]) + HOVER_ABOVE_PLACE_M
        if not policy.check_feasible_ik(place):
            raise ValueError("IK failed for tray-inset place pose")
        if not policy.check_feasible_ik(preplace):
            raise ValueError("IK failed for hover preplace pose")
        print(
            json.dumps(
                {
                    "tray_place_wrap": "hover_then_vertical_drop",
                    "place_xy_m": [float(place[0, 3]), float(place[1, 3])],
                    "place_z_m": float(place[2, 3]),
                    "preplace_z_m": float(preplace[2, 3]),
                    "footprint_shift_xy_m": [float(shift_xy[0]), float(shift_xy[1])],
                }
            ),
            flush=True,
        )
        return preplace, place, postplace

    def _compute_trajectory():
        primitives = original_traj()
        place_prim = None
        preplace = descent = None
        for prim, segs in _tcp_primitives(primitives):
            for seg in segs:
                name = str(getattr(seg, "name", "") or "")
                if name == "preplace":
                    place_prim = prim
                    preplace = seg
                elif name == "placement_descent":
                    place_prim = prim
                    descent = seg
        if preplace is None or descent is None:
            return primitives
        hover = descent.end_pose.copy()
        hover[:2, 3] = descent.end_pose[:2, 3]
        hover[2, 3] = max(
            float(preplace.start_pose[2, 3]),
            float(descent.end_pose[2, 3]) + HOVER_ABOVE_PLACE_M,
        )
        if not policy.check_feasible_ik(hover):
            hover[2, 3] = float(descent.end_pose[2, 3]) + HOVER_ABOVE_PLACE_M
            if not policy.check_feasible_ik(hover):
                raise ValueError("IK failed for vertical-drop hover pose")
        inbound_z = float(hover[2, 3])
        for prim, segs in _tcp_primitives(primitives):
            for index, seg in enumerate(segs):
                name = str(getattr(seg, "name", "") or "")
                if name == "preplace":
                    start = seg.start_pose.copy()
                    start[2, 3] = inbound_z
                    if policy.check_feasible_ik(start):
                        seg.start_pose = start
                    seg.end_pose = hover.copy()
                    if index > 0:
                        prev = segs[index - 1]
                        prev_end = prev.end_pose.copy()
                        prev_end[2, 3] = inbound_z
                        if policy.check_feasible_ik(prev_end):
                            prev.end_pose = prev_end
                            seg.start_pose = prev_end.copy()
                elif name == "placement_descent":
                    start = hover.copy()
                    start[:2, 3] = seg.end_pose[:2, 3]
                    seg.start_pose = start
                    if index > 0:
                        segs[index - 1].end_pose = start.copy()
        if place_prim is not None:
            _refresh_duration(place_prim)
        print(
            json.dumps(
                {
                    "tray_drop": "vertical",
                    "hover_xyz_m": [float(v) for v in hover[:3, 3]],
                    "place_xyz_m": [float(v) for v in descent.end_pose[:3, 3]],
                    "preplace_start_xyz_m": [float(v) for v in preplace.start_pose[:3, 3]],
                    "preplace_end_xyz_m": [float(v) for v in preplace.end_pose[:3, 3]],
                }
            ),
            flush=True,
        )
        return primitives

    policy._get_placement_poses = _get_placement_poses  # type: ignore[method-assign]
    policy._compute_trajectory = _compute_trajectory  # type: ignore[method-assign]


def _render_rgb(env, position, forward, up, fov: float) -> np.ndarray:
    return np.asarray(
        env._render_frame(position, forward, up, fov, segmentation=False)
    )


def _composite_frame(env) -> np.ndarray:
    table_pos, table_fwd, table_up = third_person_pose(env)
    table = _render_rgb(env, table_pos, table_fwd, table_up, THIRD_PERSON_FOV)
    wrist_pos, wrist_fwd, wrist_up = wrist_camera_pose(env)
    wrist = _render_rgb(env, wrist_pos, wrist_fwd, wrist_up, WRIST_FOV)
    height = min(table.shape[0], wrist.shape[0])
    if table.shape[0] != height:
        table = cv2.resize(table, (int(table.shape[1] * height / table.shape[0]), height))
    if wrist.shape[0] != height:
        wrist = cv2.resize(wrist, (int(wrist.shape[1] * height / wrist.shape[0]), height))
    return np.concatenate([wrist, table], axis=1)


def _save_stills(env, stem: str) -> dict[str, str]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    table_pos, table_fwd, table_up = third_person_pose(env)
    table_path = OUTPUT / f"{stem}_table_camera.png"
    Image.fromarray(_render_rgb(env, table_pos, table_fwd, table_up, THIRD_PERSON_FOV)).save(
        table_path
    )
    wrist_pos, wrist_fwd, wrist_up = wrist_camera_pose(env)
    wrist_path = OUTPUT / f"{stem}_wrist_camera.png"
    Image.fromarray(_render_rgb(env, wrist_pos, wrist_fwd, wrist_up, WRIST_FOV)).save(
        wrist_path
    )
    return {"table": str(table_path), "wrist": str(wrist_path)}


def _run_rollout(task, policy, initial_reset_result, video_path: Path) -> bool:
    observation, _info = initial_reset_result
    try:
        task.env.current_model.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_SLEEP)
    except AttributeError:
        pass

    writer = None
    step_count = 0
    video_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        while not task.is_done():
            if step_count % FRAME_STRIDE == 0:
                frame = cv2.cvtColor(_composite_frame(task.env), cv2.COLOR_RGB2BGR)
                if writer is None:
                    height, width = frame.shape[:2]
                    writer = cv2.VideoWriter(
                        str(video_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        VIDEO_FPS,
                        (width, height),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"could not open video writer: {video_path}")
                writer.write(frame)
            action_cmd = policy.get_action(observation)
            if action_cmd is None:
                print("Policy returned None action, ending episode", flush=True)
                break
            observation, _reward, _terminal, _truncated, _infos = task.step(action_cmd)
            step_count += 1
            if step_count % 50 == 0:
                print(
                    f"rollout step={step_count} phase={policy.get_phase()}",
                    flush=True,
                )
        if writer is not None:
            writer.write(cv2.cvtColor(_composite_frame(task.env), cv2.COLOR_RGB2BGR))
    finally:
        if writer is not None:
            writer.release()
        try:
            task.env.current_model.opt.enableflags &= ~int(
                mujoco.mjtEnableBit.mjENBL_SLEEP
            )
        except AttributeError:
            pass
    return bool(task.judge_success()) if hasattr(task, "judge_success") else False


def main() -> int:
    _pin_runtime()
    _install_preview_contact_classes()
    started = time.time()
    OUTPUT.mkdir(parents=True, exist_ok=True)

    from molmo_spaces.data_generation.pipeline import (
        cleanup_episode_resources,
        setup_policy,
    )
    from molmo_spaces.env.abstract_sensors import SensorSuite
    from molmo_spaces.tasks.enclosure_reach import (
        PactPlaceCorridorV1010FourObjectSampler,
    )

    row = build_row("F0_target_side_stagger", "left", "center", 0)
    row["pact_v106_scene_sha256"] = hashlib.sha256(SCENE.read_bytes()).hexdigest()
    row["pact_v1010_scene_relative"] = str(SCENE.relative_to(ROOT))
    row["environment_version"] = "pact_place_corridor_v10_10_four_object"
    row["sampler_class"] = SAMPLER_CLASS
    row["task_sampler_class"] = SAMPLER_CLASS
    row["planner"] = SAMPLER_CLASS
    row.pop("row_sha256", None)
    row["row_sha256"] = sha256_payload(row)

    config = _make_config(
        OUTPUT / "result.json",
        scene_xml=SCENE,
        sampler_class=SAMPLER_CLASS,
    )
    sampler = PactPlaceCorridorV1010FourObjectSampler(config)
    extra_bodies: list[str] = []
    original_add = sampler.add_auxiliary_objects

    def add_auxiliary_objects(spec: mujoco.MjSpec) -> None:
        original_add(spec)
        extra_bodies.extend(_attach_standing_kitchen(spec))

    sampler.add_auxiliary_objects = add_auxiliary_objects  # type: ignore[method-assign]

    task = policy = None
    result: dict[str, Any]
    try:
        sampler.seed_task_sampling(int(row["task_seed_u32"]))
        sampler.set_pact_manifest_row(row)
        task = sampler.sample_task(house_index=1)
        if task is None:
            raise RuntimeError("preview sampler returned no task")
        task._sensor_suite = SensorSuite(
            [
                task._sensor_suite.sensors[uuid]
                for uuid in ("qpos", "tcp_pose")
            ]
        )
        policy = setup_policy(config, task, None, None)
        _keep_glass_inside_blue_tray(policy)
        original_reset = policy.reset

        extra_xy: dict[str, list[float]] = {}

        def reset_with_extras(reset_retries: bool = True):
            result = original_reset(reset_retries=reset_retries)
            _install_preview_layout(task, extra_bodies)
            model, data = task.env.current_model, task.env.current_data
            extra_xy.clear()
            for name in extra_bodies:
                mocap_id = int(model.body_mocapid[int(model.body(name).id)])
                extra_xy[name] = [
                    float(data.mocap_pos[mocap_id, 0]),
                    float(data.mocap_pos[mocap_id, 1]),
                ]
            return result

        policy.reset = reset_with_extras  # type: ignore[method-assign]
        initial_reset_result = task.reset()
        rejected = disallowed_initial_contacts(
            initial_robot_environment_contacts(task.env)
        )
        if rejected:
            first = rejected[0]
            raise RuntimeError(
                "initial_robot_environment_contact "
                f"n={len(rejected)} {first.get('body1')} vs {first.get('body2')}"
            )

        start_stills = _save_stills(task.env, "start")
        video_path = OUTPUT / "wrist_table_rollout.mp4"
        print(
            json.dumps(
                {
                    "status": "starting_rollout",
                    "expert": SAMPLER_CLASS,
                    "horizon": int(config.task_horizon),
                    "extras": list(extra_bodies),
                    "extra_xy_m": extra_xy,
                    "household": list(getattr(sampler, "_pact_active_clutter_names", [])),
                },
                indent=2,
            ),
            flush=True,
        )
        task_success = _run_rollout(task, policy, initial_reset_result, video_path)
        end_stills = _save_stills(task.env, "end")
        policy_info = _jsonable(policy.get_info())
        audit = policy_info["pact_contact_audit"]
        totals = audit["contact_class_totals"]
        clutter_stability_events = list(
            policy_info.get("clutter_stability_events") or []
        )
        clean_success = bool(
            task_success
            and int(totals["hazard_bar"]) == 0
            and int(totals["other_environment"]) == 0
            and int(totals.get("clutter", 0)) == 0
            and int(totals.get("mounted_fixture", 0)) == 0
            and not clutter_stability_events
            and place_receptacle_outside_placement(audit) == 0
        )
        result = {
            "schema_version": "pact_place_v12_one_rollout_v1",
            "role": "development_preview_not_a_gate",
            "authorizes_collection": False,
            "status": "complete",
            "planner": SAMPLER_CLASS,
            "n_trials": 1,
            "task_success": bool(task_success),
            "clean_success": clean_success,
            "success_rate_task": float(task_success),
            "success_rate_clean": float(clean_success),
            "grasp_phase_success": bool(policy_info["grasp_phase_success"]),
            "place_phase_success": bool(policy_info["place_phase_success"]),
            "cup_lifted_one_cm": bool(policy_info["cup_lifted_one_cm"]),
            "failure_cause": derive_failure_cause(
                task_success=bool(task_success),
                contact_audit=audit,
                clutter_stability_events=clutter_stability_events,
                terminal_tracking=policy_info["terminal_tracking"],
            ),
            "contact_class_totals": totals,
            "clutter_stability_events": clutter_stability_events,
            "episode_steps": int(task.episode_step_count),
            "terminal_policy_phase": str(policy.get_phase()),
            "added_standing_kitchen_objects": list(extra_bodies),
            "extra_xy_m": dict(extra_xy),
            "existing_household_objects": list(
                getattr(sampler, "_pact_active_clutter_names", [])
            ),
            "scene": str(SCENE.relative_to(ROOT)),
            "cell": row["cell"],
            "task_seed_u32": int(row["task_seed_u32"]),
            "start_stills": start_stills,
            "end_stills": end_stills,
            "video": str(video_path),
            "elapsed_s": time.time() - started,
        }
    except Exception as error:  # noqa: BLE001 - diagnostic one-shot
        result = {
            "schema_version": "pact_place_v12_one_rollout_v1",
            "role": "development_preview_not_a_gate",
            "authorizes_collection": False,
            "status": "infrastructure_failure",
            "n_trials": 1,
            "task_success": False,
            "clean_success": False,
            "success_rate_task": 0.0,
            "success_rate_clean": 0.0,
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc()[-2500:],
            "elapsed_s": time.time() - started,
        }
    finally:
        cleanup_episode_resources(
            task=task,
            policy=policy,
            task_sampler=sampler,
            preloaded_policy=None,
            close_task_sampler=True,
        )

    (OUTPUT / "result.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps(result, indent=2, default=str), flush=True)
    return 0 if result.get("status") == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
