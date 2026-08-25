#!/usr/bin/env python3
"""Site the two V9 vessels against the frozen v6c replay trajectories.

This is a forward-only measurement.  The v6c robot/target qpos is copied by
joint name into a V9 model containing the real Objaverse bodies, then only
``mj_forward`` is called.  Candidate clearances use the shared hardened
instrument; world AABBs are used only for the conservative exact-distance
prune and are never emitted as clearances.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MOLMO = ROOT / "submodules" / "molmospaces"
for path in (ROOT / "scripts", MOLMO):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pact_place_v9_contract import (  # noqa: E402
    PALETTE_PATH,
    SHELF_TOP_Z,
    TUBE_X0,
    build_layout,
    load_palette,
    sha256_payload,
)

OUTPUT_DIR = ROOT / "diagnostics_output/pact_place_v9_v0c"
V9_SCENE = MOLMO / "molmo_spaces/data_generation/custom_scenes/pact_place_corridor_v5.xml"
V9_SAMPLER_CLASS = "PactPlaceCorridorV9Sampler"
BODY_GROUPS = ("cup", "fingers", "hand", "link7", "link6", "link5")
LINK_GROUPS = ("link5", "link6")
SAFE_GROUPS = ("cup", "fingers", "hand")
INBOUND_PHASES = frozenset({"approach", "insert", "advance", "pregrasp"})
OUTBOUND_PHASES = frozenset({"outbound_approach", "outbound_pass", "outbound_exit"})
PRUNE_CUTOFF_M = 0.35
NEAR_M = 0.10
VERY_NEAR_M = 0.05
SENSOR_RANGE_M = 1.0
SENSOR_DERATE = 0.85
SENSOR_HALF_FOV_COS = float(np.cos(np.deg2rad(22.5)))
WRIST_CAMERA = "robot_0/gripper/wrist_camera"
SKIP_STATIC_NAMES = frozenset({"floor", "bench_top", "bench_body"})
SKIP_STATIC_ROOTS = frozenset({"place_receptacle", "place_pedestal"})


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _prepare_v9_task(row: dict[str, Any], selected_seed: dict[str, int]):
    from molmo_spaces.data_generation.pipeline import cleanup_episode_resources
    from molmo_spaces.data_generation.runtime_compat import assert_supported_runtime
    from run_pact_place_expert_screen import _make_config

    assert_supported_runtime(strict=True)
    scratch = Path(tempfile.mkdtemp(prefix="pact_place_v9_v0c_"))
    task = sampler = None
    try:
        config = _make_config(
            scratch / "dummy.json",
            scene_xml=V9_SCENE,
            sampler_class=V9_SAMPLER_CLASS,
        )
        sampler = config.task_sampler_config.task_sampler_class(config)
        sampler.seed_task_sampling(int(selected_seed["seed_u32"]))
        sampler.set_pact_manifest_row(row)
        task = sampler.sample_task(house_index=int(row["scene_template_house_index"]))
        if task is None:
            raise RuntimeError("V9 sample_task returned None")
        task.reset()
        return task, sampler, scratch
    except Exception:
        cleanup_episode_resources(
            task=task,
            policy=None,
            task_sampler=sampler,
            preloaded_policy=None,
            close_task_sampler=sampler is not None,
        )
        shutil.rmtree(scratch, ignore_errors=True)
        raise


def _joint_qpos_slices(model) -> dict[str, tuple[int, int]]:
    import mujoco

    widths = {
        int(mujoco.mjtJoint.mjJNT_FREE): 7,
        int(mujoco.mjtJoint.mjJNT_BALL): 4,
        int(mujoco.mjtJoint.mjJNT_HINGE): 1,
        int(mujoco.mjtJoint.mjJNT_SLIDE): 1,
    }
    result = {}
    for joint_id in range(int(model.njnt)):
        name = model.joint(joint_id).name or ""
        result[name] = (
            int(model.jnt_qposadr[joint_id]),
            widths[int(model.jnt_type[joint_id])],
        )
    return result


def _copy_named_qpos(source_model, source_qpos, target_model, target_qpos, source_slices):
    target_slices = _joint_qpos_slices(target_model)
    copied = 0
    for name, (target_address, target_width) in target_slices.items():
        source = source_slices.get(name)
        if source is None or source[1] != target_width:
            continue
        source_address, _source_width = source
        target_qpos[target_address : target_address + target_width] = source_qpos[
            source_address : source_address + target_width
        ]
        copied += target_width
    return copied


def _body_geoms(model, body_name: str) -> list[int]:
    body_id = int(model.body(body_name).id)
    root_id = int(model.body_rootid[body_id])
    return [
        gid
        for gid in range(int(model.ngeom))
        if int(model.body_rootid[int(model.geom_bodyid[gid])]) == root_id
        and (int(model.geom_contype[gid]) or int(model.geom_conaffinity[gid]))
    ]


def _body_aabb(model, data, body_name: str) -> tuple[np.ndarray, np.ndarray]:
    from pact_geom_distance import geom_world_aabb

    geoms = _body_geoms(model, body_name)
    if not geoms:
        raise ValueError(f"body has no collision geoms: {body_name}")
    lows, highs = zip(*(geom_world_aabb(model, data, gid) for gid in geoms))
    return np.min(np.stack(lows), axis=0), np.max(np.stack(highs), axis=0)


def _free_joint(model, body_name: str) -> tuple[int, int]:
    import mujoco

    body_id = int(model.body(body_name).id)
    joint_id = int(model.body_jntadr[body_id])
    if joint_id < 0 or int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
        raise ValueError(f"V9 hazard body is not free: {body_name}")
    return int(model.jnt_qposadr[joint_id]), int(model.jnt_dofadr[joint_id])


def _pose_offset(model, data, body_name: str, quat: list[float]) -> tuple[int, np.ndarray]:
    import mujoco

    qadr, _dadr = _free_joint(model, body_name)
    data.qpos[qadr : qadr + 3] = 0.0
    data.qpos[qadr + 3 : qadr + 7] = np.asarray(quat, dtype=float)
    mujoco.mj_forward(model, data)
    low, high = _body_aabb(model, data, body_name)
    return qadr, (low + high) / 2.0


def _set_body_center(data, qadr: int, local_center: np.ndarray, center: np.ndarray, quat: list[float]):
    data.qpos[qadr : qadr + 3] = np.asarray(center, dtype=float) - local_center
    data.qpos[qadr + 3 : qadr + 7] = np.asarray(quat, dtype=float)


def _group_aabb(model, data, gids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    from pact_geom_distance import geom_world_aabb

    if not gids:
        return np.full(3, np.inf), np.full(3, -np.inf)
    lows, highs = zip(*(geom_world_aabb(model, data, int(gid)) for gid in gids))
    return np.min(np.stack(lows), axis=0), np.max(np.stack(highs), axis=0)


def _aabb_lower_bound(group_low, group_high, object_low, object_high) -> float:
    gap = np.maximum(
        np.maximum(np.asarray(group_low) - np.asarray(object_high), 0.0),
        np.maximum(np.asarray(object_low) - np.asarray(group_high), 0.0),
    )
    return float(np.linalg.norm(gap))


def _skin_detects(model, data, sensor_ids: list[int], center: np.ndarray, half: np.ndarray) -> bool:
    samples = (
        (-1, 0, 0),
        (-1, 0.7, 0),
        (-1, -0.7, 0),
        (-1, 0, 0.7),
        (-1, 0, -0.7),
        (0, 0.9, 0),
        (0, -0.9, 0),
        (0, 0, -0.9),
    )
    points = [center + half * np.asarray(sample, dtype=float) for sample in samples]
    for camera_id in sensor_ids:
        position = np.asarray(data.cam_xpos[camera_id], dtype=float)
        rotation = np.asarray(data.cam_xmat[camera_id], dtype=float).reshape(3, 3)
        forward = -rotation[:, 2]
        for point in points:
            delta = point - position
            distance = float(np.linalg.norm(delta))
            if distance < 1e-9 or distance > SENSOR_RANGE_M * SENSOR_DERATE:
                continue
            if float(np.dot(delta / distance, forward)) > SENSOR_HALF_FOV_COS:
                return True
    return False


def _wrist_in_fov(model, data, wrist_id: int, center: np.ndarray, half: np.ndarray) -> bool:
    rotation = np.asarray(data.cam_xmat[wrist_id], dtype=float).reshape(3, 3)
    origin = np.asarray(data.cam_xpos[wrist_id], dtype=float)
    forward = -rotation[:, 2]
    minimum_cosine = float(np.cos(np.deg2rad(float(model.cam_fovy[wrist_id])) / 2.0))
    for sample in (
        (-1, 0, 0),
        (-1, 0.7, 0),
        (-1, -0.7, 0),
        (0, 0.9, 0),
        (0, -0.9, 0),
        (0, 0, 0.9),
        (0, 0, -0.9),
    ):
        delta = center + half * np.asarray(sample, dtype=float) - origin
        distance = float(np.linalg.norm(delta))
        if distance > 1e-9 and float(np.dot(delta / distance, forward)) >= minimum_cosine:
            return True
    return False


def _tcp_position(task) -> np.ndarray:
    robot_view = task.env.current_robot.robot_view
    move_group = robot_view.get_gripper_movegroup_ids()[0]
    return np.asarray(robot_view.get_move_group(move_group).leaf_frame_to_world[:3, 3], dtype=float)


def _phase_indices(phases: list[str], role: str) -> list[int]:
    if role == "inbound_vessel":
        return [index for index, phase in enumerate(phases) if phase in INBOUND_PHASES]
    return [index for index, phase in enumerate(phases) if phase in OUTBOUND_PHASES]


def _crossing(phases: list[str], tcp_positions: list[np.ndarray], role: str, center_x: float, half_x: float) -> dict[str, Any]:
    indices = _phase_indices(phases, role)
    if len(indices) < 2:
        return {"crossed": False, "phase_crossed": False, "t_cross": None}
    start = tcp_positions[indices[0]]
    end = tcp_positions[indices[-1]]
    delta_x = float(end[0] - start[0])
    if abs(delta_x) < 1e-9:
        return {"crossed": False, "phase_crossed": False, "t_cross": None}
    t_cross = float((center_x - start[0]) / delta_x)
    phase_crossed = any(
        abs(float(tcp_positions[index][0]) - center_x) <= half_x + 0.025
        for index in indices
    )
    return {
        "crossed": bool(0.02 < t_cross < 0.98),
        "phase_crossed": bool(phase_crossed),
        "t_cross": t_cross,
        "start_x_m": float(start[0]),
        "end_x_m": float(end[0]),
        "crossing_tcp_y_m": float(start[1] + t_cross * (end[1] - start[1])),
        "phase": "inbound" if role == "inbound_vessel" else "outbound_pass",
    }


def _lateral_clearability(crossing: dict[str, Any], center: np.ndarray, half: np.ndarray, role: str, aperture_width: float) -> dict[str, Any]:
    if not crossing.get("crossed") or not crossing.get("phase_crossed"):
        return {
            "clearable": False,
            "required_bow_m": None,
            "lateral_limit_m": None,
        }
    envelope = 0.11 if role == "inbound_vessel" else 0.15
    safe_gap = 0.10 if role == "inbound_vessel" else 0.14
    side = 1.0 if float(center[1]) >= 0.0 else -1.0
    inner_face = float(center[1] - side * half[1])
    straight = side * (inner_face - float(crossing["crossing_tcp_y_m"])) - envelope
    required = max(0.0, safe_gap - straight)
    limit = max(0.0, aperture_width / 2.0 - envelope - 0.02)
    return {
        "clearable": bool(required <= limit),
        "obstacle_side": "left" if side > 0 else "right",
        "inner_face_y_m": inner_face,
        "straight_clearance_m": straight,
        "required_bow_m": required,
        "lateral_limit_m": limit,
        "envelope_half_y_m": envelope,
        "safe_gap_m": safe_gap,
    }


def _candidate_grid(palette: list[dict[str, Any]], role: str) -> list[dict[str, Any]]:
    item = next(item for item in palette if str(item.get("role")) == role)
    half = np.asarray(item["half_m"], dtype=float)
    x_low = TUBE_X0 + half[0] + 0.004
    x_high = TUBE_X0 + 0.20 - half[0] - 0.004
    y_low = -0.425 + half[1] + 0.004
    y_high = 0.425 - half[1] - 0.004
    if x_low > x_high:
        raise ValueError(f"{role} cannot fit the minimum-depth shell")
    candidates = []
    for x in np.linspace(x_low, x_high, 5):
        for y in np.linspace(y_low, y_high,  nine := 9):
            center = [float(x), float(y), float(SHELF_TOP_Z + half[2])]
            candidates.append(
                {
                    "candidate_id": f"{role}_x{x:.4f}_y{y:+.4f}",
                    "role": role,
                    "slot": str(item["slot"]),
                    "uid": str(item["uid"]),
                    "category": str(item["category"]),
                    "center_m": center,
                    "half_m": half.tolist(),
                    "quat_wxyz": list(map(float, item["quat_wxyz"])),
                    "shell_bounds_reason": "minimum_depth_0.20m_and_aperture_0.85m",
                }
            )
    return candidates


def _candidate_static_gids(model, hazard_body_names: set[str], target_gids: list[int]) -> list[int]:
    target_roots = {
        int(model.body_rootid[int(model.geom_bodyid[gid])]) for gid in target_gids
    }
    hazard_roots = {
        int(model.body_rootid[int(model.body(name).id)]) for name in hazard_body_names
    }
    result = []
    for gid in range(int(model.ngeom)):
        if not (int(model.geom_contype[gid]) or int(model.geom_conaffinity[gid])):
            continue
        body_id = int(model.geom_bodyid[gid])
        root_id = int(model.body_rootid[body_id])
        root_name = model.body(root_id).name or ""
        body_name = model.body(body_id).name or ""
        geom_name = model.geom(gid).name or ""
        if root_name.startswith("robot_0/") or root_id in target_roots or root_id in hazard_roots:
            continue
        if root_name in SKIP_STATIC_ROOTS or body_name in SKIP_STATIC_ROOTS:
            continue
        if root_name == "" and geom_name in SKIP_STATIC_NAMES:
            continue
        if geom_name in SKIP_STATIC_NAMES:
            continue
        result.append(gid)
    return result


def _evaluate_row_batch(job: dict[str, Any]) -> list[dict[str, Any]]:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("MLSPACES_ASSETS_DIR", str(ROOT / "assets"))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.pop("DISPLAY", None)

    import mujoco
    from molmo_spaces.data_generation.pipeline import cleanup_episode_resources
    from run_pact_place_swept_volume_v7 import geom_groups
    from run_pact_place_v6c_replay_videos import _prepare_task as prepare_v6c
    import pact_geom_distance as instrument
    from run_pact_place_v8_baseline import _physical_geoms, _target_geoms, _any_visible

    source_row = job["source_row"]
    result = json.loads(Path(job["result_path"]).read_text())
    steps = list(json.loads(Path(job["trajectory_path"]).read_text())["steps"])
    palette = job["palette"]
    palette_doc = job["palette_document"]
    candidates = job["candidates"]
    layout = job["layout"]

    v9_row = dict(source_row)
    v9_row.update(
        {
            "sampler_class": V9_SAMPLER_CLASS,
            "pact_clutter_palette": palette,
            "pact_clutter_layout": layout,
        }
    )

    source_task = source_sampler = source_scratch = None
    task = sampler = scratch = None
    row_results: list[dict[str, Any]] = []
    try:
        source_task, source_sampler, source_scratch = prepare_v6c(
            source_row, result["selected_seed"]
        )
        task, sampler, scratch = _prepare_v9_task(v9_row, result["selected_seed"])

        source_model = source_task.env.current_model
        model, data = task.env.current_model, task.env.current_data
        source_slices = _joint_qpos_slices(source_model)
        template_qpos = data.qpos.copy()

        hazards = list((getattr(task, "scene_params", {}) or {}).get("pact_v9_hazards") or [])
        by_role = {str(hazard["role"]): hazard for hazard in hazards}
        if set(by_role) != {"inbound_vessel", "outbound_vessel"}:
            raise RuntimeError(f"V9 task did not expose both vessel hazards: {by_role}")
        body_by_role = {role: str(hazard["body"]) for role, hazard in by_role.items()}
        pose_info = {}
        for role, body_name in body_by_role.items():
            palette_item = next(item for item in palette if str(item["role"]) == role)
            qadr, offset = _pose_offset(model, data, body_name, palette_item["quat_wxyz"])
            pose_info[role] = {
                "body": body_name,
                "qadr": qadr,
                "local_center": offset,
                "quat_wxyz": palette_item["quat_wxyz"],
            }
        target_gids = _physical_geoms(model, _target_geoms(model))
        groups = geom_groups(model)
        by_group = {
            "cup": target_gids,
            "fingers": _physical_geoms(model, list(groups["left_finger"]) + list(groups["right_finger"])),
            "hand": _physical_geoms(model, list(groups["hand"])),
            "link7": _physical_geoms(model, list(groups["link7"])),
            "link6": _physical_geoms(model, list(groups["link6"])),
            "link5": _physical_geoms(model, list(groups["link5"])),
        }
        hazard_body_names = set(body_by_role.values())
        static_gids = _candidate_static_gids(model, hazard_body_names, target_gids)
        sensor_ids = [
            index
            for index in range(int(model.ncam))
            if "_sensor_" in (model.camera(index).name or "")
        ]
        wrist_id = int(model.camera(WRIST_CAMERA).id)
        instrument.reset_counters()

        default_centers = {}
        for role, info in pose_info.items():
            low, high = _body_aabb(model, data, info["body"])
            default_centers[role] = (low + high) / 2.0

        for candidate in candidates:
            candidate_center = np.asarray(candidate["center_m"], dtype=float)
            candidate_role = str(candidate["role"])
            other_role = "outbound_vessel" if candidate_role == "inbound_vessel" else "inbound_vessel"

            phases: list[str] = []
            tcp_positions: list[np.ndarray] = []
            min_by_body = {key: float("inf") for key in BODY_GROUPS}
            frame_values: dict[str, list[float]] = {key: [] for key in LINK_GROUPS}
            skin_flags: list[bool] = []
            n_exact_frames = 0
            min_frame = None
            min_link = float("inf")
            static_clearance = float("inf")

            for frame_index, step in enumerate(steps):
                data.qpos[:] = template_qpos
                _copy_named_qpos(
                    source_model,
                    np.asarray(step["qpos"], dtype=float),
                    model,
                    data.qpos,
                    source_slices,
                )
                _set_body_center(
                    data,
                    pose_info[candidate_role]["qadr"],
                    pose_info[candidate_role]["local_center"],
                    candidate_center,
                    pose_info[candidate_role]["quat_wxyz"],
                )
                _set_body_center(
                    data,
                    pose_info[other_role]["qadr"],
                    pose_info[other_role]["local_center"],
                    default_centers[other_role],
                    pose_info[other_role]["quat_wxyz"],
                )
                mujoco.mj_forward(model, data)
                phases.append(str(step.get("policy_phase") or "unknown"))
                tcp = _tcp_position(task)
                tcp_positions.append(tcp)
                low, high = _body_aabb(model, data, pose_info[candidate_role]["body"])
                center = (low + high) / 2.0
                half = (high - low) / 2.0
                skin_flags.append(_skin_detects(model, data, sensor_ids, center, half))
                if frame_index == 0:
                    static_clearance = instrument.true_distance(
                        model,
                        data,
                        static_gids,
                        _body_geoms(model, pose_info[candidate_role]["body"]),
                    )
                frame_min = float("inf")
                for key in BODY_GROUPS:
                    group_low, group_high = _group_aabb(model, data, by_group[key])
                    if _aabb_lower_bound(group_low, group_high, low, high) >= PRUNE_CUTOFF_M:
                        value = float("inf")
                    else:
                        value = instrument.true_distance(
                            model,
                            data,
                            by_group[key],
                            _body_geoms(model, pose_info[candidate_role]["body"]),
                        )
                        n_exact_frames += 1
                    min_by_body[key] = min(min_by_body[key], value)
                    if key in LINK_GROUPS:
                        frame_values[key].append(value)
                        frame_min = min(frame_min, value)
                if frame_min < min_link:
                    min_link = frame_min
                    min_frame = frame_index

            if min_frame is None:
                min_frame = 0

            data.qpos[:] = template_qpos
            _copy_named_qpos(
                source_model,
                np.asarray(steps[min_frame]["qpos"], dtype=float),
                model,
                data.qpos,
                source_slices,
            )
            _set_body_center(
                data,
                pose_info[candidate_role]["qadr"],
                pose_info[candidate_role]["local_center"],
                candidate_center,
                pose_info[candidate_role]["quat_wxyz"],
            )
            _set_body_center(
                data,
                pose_info[other_role]["qadr"],
                pose_info[other_role]["local_center"],
                default_centers[other_role],
                pose_info[other_role]["quat_wxyz"],
            )
            mujoco.mj_forward(model, data)
            low, high = _body_aabb(model, data, pose_info[candidate_role]["body"])
            exact_center = (low + high) / 2.0
            exact_half = (high - low) / 2.0
            wrist_fov = _wrist_in_fov(model, data, wrist_id, exact_center, exact_half)
            wrist_visible = False
            try:
                wrist_visible = bool(
                    _any_visible(model, data, wrist_id, _body_geoms(model, pose_info[candidate_role]["body"]))
                )
            except Exception:
                wrist_visible = wrist_fov

            crossing = _crossing(
                phases,
                tcp_positions,
                candidate_role,
                float(exact_center[0]),
                float(exact_half[0]),
            )
            lateral = _lateral_clearability(
                crossing,
                exact_center,
                exact_half,
                candidate_role,
                float((getattr(task, "scene_params", {}) or {}).get("ap_w", 0.85)),
            )
            finite_links = {key: value for key, value in min_by_body.items() if key in ("link5", "link6", "link7") and np.isfinite(value)}
            closest_arm_link = min(finite_links, key=finite_links.get) if finite_links else None
            finite_all = {key: value for key, value in min_by_body.items() if np.isfinite(value)}
            closest = min(finite_all, key=finite_all.get) if finite_all else None
            link_values = [min_by_body[key] for key in LINK_GROUPS if np.isfinite(min_by_body[key])]
            target_phases = INBOUND_PHASES if candidate_role == "inbound_vessel" else OUTBOUND_PHASES
            min_phase = phases[min_frame] if min_frame is not None and min_frame < len(phases) else None

            reasons = []
            if not any(skin_flags):
                reasons.append("not_detectable_by_exact_skin_gate")
            if not crossing.get("crossed") or not crossing.get("phase_crossed"):
                reasons.append("not_crossed_on_required_phase_leg")
            if min_phase not in target_phases or closest_arm_link not in LINK_GROUPS:
                reasons.append("closest_sensed_body_is_not_link5_or_link6")
            if any(min_by_body[key] <= 0.0 for key in SAFE_GROUPS):
                reasons.append("predicted_contact_with_cup_fingers_or_hand")
            if static_clearance <= 0.0:
                reasons.append("intersects_static_scene_or_active_decor")
            if not lateral.get("clearable"):
                reasons.append("required_lateral_bow_exceeds_aperture_limit")

            min_link_clearance = min(
                [value for value in (min_by_body["link5"], min_by_body["link6"]) if np.isfinite(value)],
                default=float("inf"),
            )
            all_link_frames = [
                value for values in frame_values.values() for value in values if np.isfinite(value)
            ]
            row_results.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "role": candidate_role,
                    "slot": candidate["slot"],
                    "uid": candidate["uid"],
                    "category": candidate["category"],
                    "center_m": candidate["center_m"],
                    "half_m": exact_half.tolist(),
                    "min_clearance_by_body_m": {
                        key: (None if not np.isfinite(value) else float(value))
                        for key, value in min_by_body.items()
                    },
                    "closest_body": closest,
                    "link_primary": closest in LINK_GROUPS,
                    "cup_is_closest_body": bool(
                        np.isfinite(min_by_body["cup"])
                        and link_values
                        and min_by_body["cup"] < min(link_values)
                    ),
                    "frames_link5_clearance_lt_10cm": int(sum(value < NEAR_M for value in frame_values["link5"])),
                    "frames_link6_clearance_lt_10cm": int(sum(value < NEAR_M for value in frame_values["link6"])),
                    "frames_link5_clearance_lt_5cm": int(sum(value < VERY_NEAR_M for value in frame_values["link5"])),
                    "frames_link6_clearance_lt_5cm": int(sum(value < VERY_NEAR_M for value in frame_values["link6"])),
                    "frames_link5_6_clearance_lt_10cm": int(
                        sum(min(values) < NEAR_M for values in zip(frame_values["link5"], frame_values["link6"]))
                    ),
                    "frames_link5_6_clearance_lt_5cm": int(
                        sum(min(values) < VERY_NEAR_M for values in zip(frame_values["link5"], frame_values["link6"]))
                    ),
                    "min_link5_link6_clearance_m": None if not np.isfinite(min_link_clearance) else float(min_link_clearance),
                    "phase_of_min_link_clearance": phases[min_frame] if min_frame is not None and min_frame < len(phases) else None,
                    "crossing": crossing,
                    "lateral_clearability": lateral,
                    "skin_detect_frames": int(sum(skin_flags)),
                    "skin_detect_fraction": float(sum(skin_flags) / max(1, len(skin_flags))),
                    "skin_detected_at_min": bool(skin_flags[min_frame]) if min_frame is not None and min_frame < len(skin_flags) else False,
                    "wrist_fov_at_min": bool(wrist_fov),
                    "wrist_visible_at_min": bool(wrist_visible),
                    "visibility_at_min": bool(wrist_visible),
                    "static_clearance_m": None if not np.isfinite(static_clearance) else float(static_clearance),
                    "n_exact_distance_frames": int(n_exact_frames),
                    "instrument_counters": instrument.counters(),
                    "admitted": not reasons,
                    "rejection_reasons": reasons,
                }
            )
        return row_results
    finally:
        cleanup_episode_resources(
            task=source_task,
            policy=None,
            task_sampler=source_sampler,
            preloaded_policy=None,
            close_task_sampler=source_sampler is not None,
        )
        cleanup_episode_resources(
            task=task,
            policy=None,
            task_sampler=sampler,
            preloaded_policy=None,
            close_task_sampler=sampler is not None,
        )
        if source_scratch is not None:
            shutil.rmtree(source_scratch, ignore_errors=True)
        if scratch is not None:
            shutil.rmtree(scratch, ignore_errors=True)


def _pair_is_separate(inbound: dict[str, Any], outbound: dict[str, Any]) -> bool:
    left = np.asarray(inbound["center_m"], dtype=float)
    right = np.asarray(outbound["center_m"], dtype=float)
    left_half = np.asarray(inbound["half_m"], dtype=float)
    right_half = np.asarray(outbound["half_m"], dtype=float)
    return bool(np.any(np.abs(left - right) > left_half + right_half + 1e-6))


def _coverage_pair(inbound: dict[str, Any], outbound: dict[str, Any]) -> dict[str, Any]:
    links = {inbound.get("closest_body"), outbound.get("closest_body")} - {None}
    signs = {
        np.sign(float(inbound["center_m"][1])),
        np.sign(float(outbound["center_m"][1])),
    }
    return {
        "coverage_distinct_closest_links": len(links),
        "coverage_distinct_lateral_sides": len(signs),
        "coverage_phase_legs": 2,
        "total_frames_link5_6_lt_10cm": int(
            inbound["frames_link5_6_clearance_lt_10cm"]
            + outbound["frames_link5_6_clearance_lt_10cm"]
        ),
        "minimum_link_clearance_m": min(
            inbound["min_link5_link6_clearance_m"],
            outbound["min_link5_link6_clearance_m"],
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--rows", type=int, default=24)
    parser.add_argument("--x-limit", type=int, default=0)
    args = parser.parse_args()
    if not 1 <= args.workers <= 12:
        raise SystemExit("workers must be in [1, 12]")

    from run_pact_place_v6c_replay_videos import CONFIG_PATH, row_directory

    palette_document = load_palette(PALETTE_PATH)
    palette = list(palette_document["palette"])
    layout = build_layout(palette_document)
    v6c_config = json.loads(CONFIG_PATH.read_text())
    source_rows = [
        row for row in v6c_config["expert_screen_rows"] if int(row["role_index"]) < args.rows
    ]
    candidates = [
        *_candidate_grid(palette, "inbound_vessel"),
        *_candidate_grid(palette, "outbound_vessel"),
    ]
    if args.x_limit:
        candidates = candidates[: args.x_limit]
    jobs = []
    for source_row in source_rows:
        directory = row_directory(int(source_row["role_index"]), source_row["episode_id"])
        jobs.append(
            {
                "source_row": source_row,
                "result_path": str(directory / "result.json"),
                "trajectory_path": str(directory / "trajectory.json"),
                "palette": palette,
                "palette_document": palette_document,
                "layout": layout,
                "candidates": candidates,
            }
        )

    role_candidates: dict[str, list[dict[str, Any]]] = {"inbound_vessel": [], "outbound_vessel": []}
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers, mp_context=context, max_tasks_per_child=1
    ) as pool:
        for index, row_results in enumerate(pool.map(_evaluate_row_batch, jobs), start=1):
            for record in row_results:
                role_candidates[str(record["role"])].append(record)
            print(f"measured row {index}/{len(jobs)} ({len(row_results)} candidates)", flush=True)
    for records in role_candidates.values():
        records.sort(key=lambda item: item["candidate_id"])

    inbound_admitted = [item for item in role_candidates["inbound_vessel"] if item["admitted"]]
    outbound_admitted = [item for item in role_candidates["outbound_vessel"] if item["admitted"]]
    pair_records = []
    for inbound in inbound_admitted:
        for outbound in outbound_admitted:
            separate = _pair_is_separate(inbound, outbound)
            coverage = _coverage_pair(inbound, outbound)
            pair_records.append(
                {
                    "inbound_candidate_id": inbound["candidate_id"],
                    "outbound_candidate_id": outbound["candidate_id"],
                    "separate": separate,
                    "admitted": separate,
                    "rejection_reasons": [] if separate else ["vessels_overlap_in_layout"],
                    "coverage": coverage,
                }
            )
    admissible_pairs = [item for item in pair_records if item["admitted"]]
    # Coverage is a lexicographic tuple, not one scalar rank: first preserve
    # distinct link and lateral-side exposure, then maximize measured exposure.
    admissible_pairs.sort(
        key=lambda item: (
            -item["coverage"]["coverage_distinct_closest_links"],
            -item["coverage"]["coverage_distinct_lateral_sides"],
            -item["coverage"]["coverage_phase_legs"],
            -item["coverage"]["total_frames_link5_6_lt_10cm"],
            item["coverage"]["minimum_link_clearance_m"],
        )
    )
    chosen_pair = None
    if admissible_pairs:
        best = admissible_pairs[0]
        chosen_pair = {
            "inbound": next(item for item in inbound_admitted if item["candidate_id"] == best["inbound_candidate_id"]),
            "outbound": next(item for item in outbound_admitted if item["candidate_id"] == best["outbound_candidate_id"]),
            "coverage": best["coverage"],
            "selection": "coverage_lexicographic_not_single_scalar_top_n",
        }
    totals = {"calls": 0, "fromto_span_fallback": 0, "gjk_fallback": 0, "aabb_disproof_only": 0, "accepted_zero": 0}
    for records in role_candidates.values():
        for record in records:
            for key, value in record["instrument_counters"].items():
                totals[key] = totals.get(key, 0) + int(value)
    document = {
        "schema_version": "pact_place_v9_v0c_siting_v1",
        "role": "v0c_siting_measurement_not_a_gate",
        "authorizes_gate": False,
        "authorizes_collection": False,
        "replay_only": True,
        "physics_stepped": False,
        "source_screen": "diagnostics_output/pact_place_corridor_v6c",
        "source_trajectory_operation": "joint-name qpos transfer then mj_forward; no mj_step",
        "palette_path": str(PALETTE_PATH.relative_to(ROOT)),
        "palette_sha256": sha256_payload(palette_document),
        "distance_instrument": "scripts/pact_geom_distance.true_distance",
        "aabb_use": "conservative exact-distance prune only; no AABB value is reported as clearance",
        "skin_gate": {
            "half_fov_deg": 22.5,
            "range_m": SENSOR_RANGE_M,
            "derate": SENSOR_DERATE,
            "effective_range_m": SENSOR_RANGE_M * SENSOR_DERATE,
            "sensor_poses": "real compiled *_sensor_* camera poses",
        },
        "candidate_grid": {
            "roles": ["inbound_vessel", "outbound_vessel"],
            "n_candidates_per_role": len(candidates) // 2,
            "minimum_shell_depth_m": 0.20,
            "aperture_width_m": 0.85,
            "shelf_top_z_m": SHELF_TOP_Z,
        },
        "admission_rules": {
            "detectable": True,
            "required_phase_crossing": True,
            "link_primary": ["link5", "link6"],
            "safe_groups_must_stay_positive": list(SAFE_GROUPS),
            "lateral_clearability": "required bow <= aperture_width/2 - envelope_half_y - 0.02",
            "wrist_visibility_reported_not_gated": True,
        },
        "candidate_rows": {
            role: records for role, records in role_candidates.items()
        },
        "n_admitted_by_role": {
            role: sum(bool(item["admitted"]) for item in records)
            for role, records in role_candidates.items()
        },
        "pair_records": pair_records,
        "n_admissible_pairs": len(admissible_pairs),
        "chosen_pair": chosen_pair,
        "instrument_counters_total": totals,
    }
    document["analysis_sha256"] = sha256_payload(document)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "siting.json").write_text(json.dumps(_jsonable(document), indent=2, sort_keys=True) + "\n")
    print(json.dumps({"n_jobs": len(jobs), "n_admitted": document["n_admitted_by_role"], "n_pairs": len(admissible_pairs), "chosen": chosen_pair}, indent=2, default=str)[:5000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
