#!/usr/bin/env python3
"""Replay-render attempt-7 design-review episodes from recorded qpos.

Copied from the v6c renderer. Do not edit the v5 renderer. Do not edit the v6b renderer. Do not edit the v6c renderer.

Restore `data.qpos`, call `mj_forward`, and render. Do not step physics and
do not run the expert. The scene must be `pact_place_corridor_v4.xml`. Every
`pact_clutter_00` … `pact_clutter_15` body is asserted, including parked
pool slots at z = -2.0.

Clip names: `review00_clean_success.mp4`, `review01_FAIL_<phase>_<branch>.mp4`.
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import json
import multiprocessing
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MOLMO = ROOT / "submodules" / "molmospaces"
for search_path in (ROOT / "scripts", MOLMO):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from pact_place_corridor_contract import (  # noqa: E402
    V7_CLUTTER_POOL_SLOT_NAMES,
    load_design_review_contract,
    sha256_file,
    sha256_payload,
)
from run_pact_place_expert_screen import _make_config  # noqa: E402
from run_pact_place_swept_volume_v7 import (  # noqa: E402
    LINK_KEYS,
    PASSAGE_LINKS,
    _sensor_config,
    aabb_distance,
    closest_on_aabb,
    clutter_geoms,
    clutter_surface_samples,
    geom_groups,
    point_in_camera_fov,
    resolve_camera_name,
    world_aabb_for_geom,
    wrist_visible_fraction,
)

CONFIG_PATH = ROOT / "configs/pact_place_corridor_v7_design_review.json"
SCREEN_ROOT = ROOT / "diagnostics_output/pact_place_corridor_v7_design_review"
DEFAULT_OUTPUT = (
    ROOT / "diagnostics_output/pact_place_corridor_v7_design_review/videos"
)
REQUIRED_SCENE_XML = "pact_place_corridor_v4.xml"
WRIST_CAMERA_MJCF = "robot_0/gripper/wrist_camera"
FPS = 1000.0 / 66.0
HOLD_FINAL_S = 1.0
INDEXING_TOLERANCE_M = 1e-6
DERIVED_TOLERANCE_M = 1e-3
PANE_WH = (624, 352)
THIRD_PERSON_FOV = 58.0
WRIST_FOV = 56.74
CAMERA_REFERENCE_BODY = "robot_0/fr3_link0"
CAMERA_OFFSET = np.asarray([-1.05, -0.55, 1.30], dtype=np.float64)
LOOKAT_OFFSET = np.asarray([0.55, 0.0, 0.45], dtype=np.float64)
OVERLAY_FIELDS = (
    "policy_phase + step",
    "gripper width (m)",
    "running clutter frames",
    "min_clearance link4/5/6",
)
PARK_Z_M = -1.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _slug(value: str) -> str:
    text = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(value))
    return text.strip("_") or "unknown"


def clip_stem(role_index: int, result: dict[str, Any]) -> str:
    if result.get("clean_success"):
        return f"review{role_index:02d}_clean_success"
    phase = _slug(str(result.get("terminal_policy_phase") or "unknown"))
    tracking = result.get("terminal_tracking") or {}
    branch = _slug(str(tracking.get("check_failure_branch") or "unknown"))
    return f"review{role_index:02d}_FAIL_{phase}_{branch}"


def row_directory(role_index: int, episode_id: str) -> Path:
    return SCREEN_ROOT / "expert_screen_rows" / f"{role_index:02d}_{episode_id[:16]}"


def apply_recorded_qpos(env, qpos_list: list[float]) -> np.ndarray:
    import mujoco

    qpos = np.asarray(qpos_list, dtype=np.float64)
    data = env.current_data
    if qpos.shape != (data.qpos.size,):
        raise RuntimeError(
            f"qpos length {qpos.shape[0]} != model nq {data.qpos.size}; "
            "indexing is wrong and the video is not the episode"
        )
    data.qpos[:] = qpos
    mujoco.mj_forward(env.current_model, data)
    return qpos


def third_person_pose(env) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from molmo_spaces.env.data_views import create_mlspaces_body

    body = create_mlspaces_body(env.current_data, CAMERA_REFERENCE_BODY)
    rotation = np.asarray(body.pose[:3, :3], dtype=np.float64)
    translation = np.asarray(body.pose[:3, 3], dtype=np.float64)
    position = rotation @ CAMERA_OFFSET + translation
    target = rotation @ LOOKAT_OFFSET + translation
    forward = target - position
    forward /= np.linalg.norm(forward)
    desired_up = rotation @ np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(forward, desired_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)
    return position, forward, up


def wrist_camera_pose(env) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model, data = env.current_model, env.current_data
    camera_id = int(model.camera(WRIST_CAMERA_MJCF).id)
    position = np.asarray(data.cam_xpos[camera_id], dtype=np.float64)
    rotation = np.asarray(data.cam_xmat[camera_id], dtype=np.float64).reshape(3, 3)
    forward = -rotation[:, 2]
    up = rotation[:, 1]
    forward = forward / np.linalg.norm(forward)
    up = up / np.linalg.norm(up)
    return position, forward, up


def tcp_position_m(env) -> np.ndarray:
    robot_view = env.current_robot.robot_view
    gripper_id = robot_view.get_gripper_movegroup_ids()[0]
    tcp = robot_view.get_gripper(gripper_id).leaf_frame_to_world
    return np.asarray(tcp[:3, 3], dtype=np.float64)


def gripper_width_m(env) -> float:
    robot_view = env.current_robot.robot_view
    gripper_id = robot_view.get_gripper_movegroup_ids()[0]
    return float(robot_view.get_gripper(gripper_id).inter_finger_dist)


def object_position_m(task) -> np.ndarray:
    env = task.env
    manager = env.object_managers[env.current_batch_index]
    name = task.config.task_config.pickup_obj_name
    pickup = manager.get_object_by_name(name)
    if pickup is None:
        raise RuntimeError(f"pickup object {name!r} missing after qpos restore")
    return np.asarray(pickup.position, dtype=np.float64)


def active_clutter_geoms(model, data) -> list[tuple[str, int]]:
    found = []
    for body, gid in clutter_geoms(model):
        z = float(data.body(model.body(body).id).xpos[2])
        if z > PARK_Z_M:
            found.append((body, gid))
    return found


def clutter_contact_active(env) -> bool:
    from molmo_spaces.tasks.pact_place_contact_audit import (
        classify_contact,
        place_environment_contact_pairs,
    )

    return any(
        classify_contact(pair) == "clutter"
        for pair in place_environment_contact_pairs(env)
    )


def clutter_pairs_by_link(env) -> dict[str, int]:
    from molmo_spaces.tasks.pact_place_contact_audit import (
        classify_contact,
        place_environment_contact_pairs,
    )

    counts: dict[str, int] = {}
    for pair in place_environment_contact_pairs(env):
        if classify_contact(pair) != "clutter":
            continue
        names = (pair["body1"], pair["body2"], pair["root1"], pair["root2"])
        for name in names:
            if "fr3_link" in name:
                for link in ("link7", "link6", "link5", "link4", "link3", "link2", "link1"):
                    if f"fr3_{link}" in name:
                        counts[link] = counts.get(link, 0) + 1
                        break
            if "cavity_obj_" in name:
                counts["cup"] = counts.get("cup", 0) + 1
            if "gripper/" in name:
                counts["hand_or_finger"] = counts.get("hand_or_finger", 0) + 1
    return counts


def _max_abs_diff(left, right) -> float:
    return float(np.max(np.abs(np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64))))


def _resize_pane(frame: np.ndarray) -> np.ndarray:
    width, height = PANE_WH
    if frame.shape[0] != height or frame.shape[1] != width:
        return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return frame


def overlay_composite(
    wrist_rgb: np.ndarray,
    third_rgb: np.ndarray,
    *,
    role_index: int,
    clean_success: bool,
    step: int,
    n_steps: int,
    policy_phase: str,
    gripper_width_m: float,
    clutter_frames: int,
    min_clearance_by_link: dict[str, float],
    terminal: bool,
) -> np.ndarray:
    width, height = PANE_WH
    for name, frame in (("wrist", wrist_rgb), ("third-person", third_rgb)):
        if frame.shape != (height, width, 3):
            raise ValueError(f"{name} render shape {frame.shape} != {(height, width, 3)}")
    composite = np.concatenate([wrist_rgb, third_rgb], axis=1)
    frame = cv2.cvtColor(composite, cv2.COLOR_RGB2BGR)
    shade = frame.copy()
    cv2.rectangle(shade, (0, 0), (frame.shape[1], 108), (0, 0, 0), thickness=-1)
    cv2.rectangle(
        shade, (0, height - 28), (frame.shape[1], height), (0, 0, 0), thickness=-1
    )
    frame = cv2.addWeighted(shade, 0.62, frame, 0.38, 0.0)
    status = "clean" if clean_success else "FAIL"
    terminal_tag = "  TERMINAL" if terminal else ""
    cv2.putText(
        frame,
        f"review {role_index:02d}  {status}  {policy_phase}  step {step}/{n_steps - 1}{terminal_tag}",
        (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (25, 25, 245) if terminal else (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"gripper {gripper_width_m:.3f} m   clutter {clutter_frames}",
        (12, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    clear_txt = "  ".join(
        f"{link} {min_clearance_by_link.get(link, float('nan')):.3f}m"
        for link in ("link4", "link5", "link6")
    )
    cv2.putText(
        frame,
        f"min_clearance  {clear_txt}",
        (12, 88),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "wrist",
        (12, height - 9),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "third-person",
        (width + 12, height - 9),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    return frame


def _assert_static_furniture(task, row: dict[str, Any], cfg: dict[str, Any]) -> None:
    want = np.asarray(cfg["scene"]["place_receptacle_center_xyz_m"], dtype=float)
    model = task.env.current_model
    data = task.env.current_data
    got = np.asarray(data.body(model.body("place_receptacle").id).xpos, dtype=float)
    if float(np.linalg.norm(got[:2] - want[:2])) > 1e-3:
        raise RuntimeError(
            f"rendered receptacle at {got[:2].tolist()} but config says "
            f"{want[:2].tolist()} — wrong scene loaded"
        )
    nominal = cfg["scene"]["clutter_nominal"]
    park = np.asarray(cfg["scene"]["clutter_pool_park_xyz_m"], dtype=float)
    x_jitter = row.get("clutter_x_jitter_m") or {}
    y_jitter = row.get("clutter_y_jitter_m") or {}
    seen = []
    for slot in V7_CLUTTER_POOL_SLOT_NAMES:
        body = f"pact_clutter_{slot}"
        seen.append(body)
        observed = np.asarray(data.body(model.body(body).id).xpos, dtype=float)
        if slot in nominal:
            center = nominal[slot]["center_m"]
            expected = np.asarray(
                [
                    float(center[0]) + float(x_jitter.get(slot, 0.0)),
                    float(center[1]) + float(y_jitter.get(slot, 0.0)),
                    float(center[2]),
                ],
                dtype=float,
            )
            if float(np.linalg.norm(observed - expected)) > 1e-3:
                raise RuntimeError(
                    f"rendered {body} at {observed.tolist()} but config+jitter says "
                    f"{expected.tolist()} — wrong scene or clutter not applied"
                )
        else:
            if float(np.linalg.norm(observed - park)) > 1e-3:
                raise RuntimeError(
                    f"pool body {body} at {observed.tolist()} is not parked at "
                    f"{park.tolist()}"
                )
    if seen != [f"pact_clutter_{slot}" for slot in V7_CLUTTER_POOL_SLOT_NAMES]:
        raise RuntimeError("clutter pool assertion did not visit all 16 bodies")


def _prepare_task(row: dict[str, Any], selected_seed: dict[str, int], cfg: dict[str, Any]):
    from molmo_spaces.data_generation.pipeline import cleanup_episode_resources
    from molmo_spaces.data_generation.runtime_compat import assert_supported_runtime

    assert_supported_runtime(strict=True)
    scratch = Path(tempfile.mkdtemp(prefix="pact_place_v7_replay_"))
    task = sampler = None
    try:
        scene_xml = ROOT / cfg["scene"]["xml"]
        if scene_xml.name != REQUIRED_SCENE_XML:
            raise RuntimeError(
                f"v7 must render {REQUIRED_SCENE_XML}, got {scene_xml.name}"
            )
        config = _make_config(scratch / "dummy.json", scene_xml=scene_xml)
        sampler = config.task_sampler_config.task_sampler_class(config)
        sampler.seed_task_sampling(int(selected_seed["seed_u32"]))
        sampler.set_pact_manifest_row(row)
        task = sampler.sample_task(house_index=int(row["scene_template_house_index"]))
        if task is None:
            raise RuntimeError("sample_task returned None for the recorded seed")
        task.reset()
        _assert_static_furniture(task, row, cfg)
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


def spot_check_indices(n: int) -> list[int]:
    if n <= 0:
        raise ValueError("trajectory is empty")
    if n == 1:
        return [0]
    wanted = [0, n // 4, n // 2, (3 * n) // 4, n - 1]
    return sorted(dict.fromkeys(wanted))


def iter_replay_states(env, task, steps: list[dict[str, Any]], spot_indices: set[int]):
    residuals: list[dict[str, Any]] = []
    for index, step in enumerate(steps):
        apply_recorded_qpos(env, step["qpos"])
        live_object = object_position_m(task)
        live_tcp = tcp_position_m(env)
        if index in spot_indices:
            recorded_object = step.get("object_position_m")
            recorded_tcp = step.get("tcp_position_m")
            if recorded_object is None or recorded_tcp is None:
                raise RuntimeError(f"spot-check step {index} is missing recorded poses")
            object_residual = _max_abs_diff(live_object, recorded_object)
            tcp_residual = _max_abs_diff(live_tcp, recorded_tcp)
            residuals.append(
                {
                    "step_index": index,
                    "recorded_step": int(step["step"]),
                    "object_position_residual_m": object_residual,
                    "tcp_position_residual_m": tcp_residual,
                }
            )
            limit = INDEXING_TOLERANCE_M if index == 0 else DERIVED_TOLERANCE_M
            if object_residual > limit:
                raise RuntimeError(
                    "qpos indexing is wrong and the video is not the episode: "
                    f"step {index} object residual {object_residual:.3e} m "
                    f"(limit {limit:.3e} m)"
                )
            if tcp_residual > limit:
                raise RuntimeError(
                    "derived TCP after mj_forward drifted off the recording: "
                    f"step {index} tcp residual {tcp_residual:.3e} m "
                    f"(object residual {object_residual:.3e} m, limit {limit:.3e} m)"
                )
        yield index, step, live_object, live_tcp, residuals


def assert_replay_source_is_forward_only(source: str | None = None) -> None:
    text = source if source is not None else Path(__file__).read_text()
    tree = ast.parse(text)
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"apply_recorded_qpos", "iter_replay_states", "render_row"}
    }
    for name in ("apply_recorded_qpos", "iter_replay_states", "render_row"):
        node = functions[name]
        called = [
            ast.unparse(item.func) if hasattr(ast, "unparse") else ""
            for item in ast.walk(node)
            if isinstance(item, ast.Call)
        ]
        joined = " ".join(called)
        if "mj_step" in joined:
            raise RuntimeError(f"{name} steps physics")
        if "setup_policy" in joined or "ParallelRolloutRunner" in joined:
            raise RuntimeError(f"{name} re-runs the expert")
        if "get_action" in joined:
            raise RuntimeError(f"{name} calls the controller")
    apply_src = ast.get_source_segment(text, functions["apply_recorded_qpos"]) or ""
    if "mj_forward" not in apply_src:
        raise RuntimeError("apply_recorded_qpos does not call mj_forward")


def accumulate_a0d_metrics(
    env,
    *,
    sensor_cfg: dict[str, Any],
    groups: dict[str, list[int]],
    state: dict[str, Any],
) -> dict[str, float]:
    model, data = env.current_model, env.current_data
    clutter = active_clutter_geoms(model, data)
    clutter_body_ids = {int(model.body(body).id) for body, _ in clutter}
    clutter_aabbs = []
    for body, gid in clutter:
        lo, hi = world_aabb_for_geom(model, data, gid)
        clutter_aabbs.append((body, lo, hi))
    step_clear: dict[str, float] = {}
    for key in LINK_KEYS:
        gids = groups[key]
        if not gids or not clutter_aabbs:
            continue
        lo = np.full(3, np.inf)
        hi = np.full(3, -np.inf)
        for gid in gids:
            glo, ghi = world_aabb_for_geom(model, data, gid)
            lo = np.minimum(lo, glo)
            hi = np.maximum(hi, ghi)
        dist = min(aabb_distance(lo, hi, clo, chi) for _, clo, chi in clutter_aabbs)
        step_clear[key] = float(dist)
        prev = state["min_clearance"].get(key)
        state["min_clearance"][key] = dist if prev is None else min(prev, dist)
    fov_deg = float(sensor_cfg["fov_deg"])
    far_m = float(sensor_cfg["clip_far_m"])
    encoder_far = float(sensor_cfg["encoder_max_range_m"])
    if state["samples"] is None and clutter_aabbs:
        state["samples"] = np.concatenate(
            [clutter_surface_samples(lo, hi) for _, lo, hi in clutter_aabbs],
            axis=0,
        )
        vis_n, vis_d = wrist_visible_fraction(
            model, data, state["samples"], clutter_body_ids
        )
        state["first_frame_vis"] = (vis_n, vis_d)
    vis_n, vis_d = state.get("first_frame_vis") or (0, 0)
    state["vis_num"] += vis_n
    state["vis_den"] += vis_d
    cam_names = state["cam_names"]
    cam_ids = state["cam_ids"]
    step_engaged = 0
    for cam_id, cam_name in zip(cam_ids, cam_names):
        cam_pos = np.asarray(data.cam_xpos[cam_id], dtype=np.float64)
        cam_mat = np.asarray(data.cam_xmat[cam_id], dtype=np.float64).reshape(3, 3)
        forward = -cam_mat[:, 2]
        forward = forward / np.linalg.norm(forward)
        best = None
        best_enc = None
        for _, lo, hi in clutter_aabbs:
            point = closest_on_aabb(cam_pos, lo, hi)
            in_fov, dist = point_in_camera_fov(point, cam_pos, forward, fov_deg, far_m)
            if in_fov and (best is None or dist < best):
                best = dist
            in_fov_e, dist_e = point_in_camera_fov(
                point, cam_pos, forward, fov_deg, encoder_far
            )
            if in_fov_e and (best_enc is None or dist_e < best_enc):
                best_enc = dist_e
        state["pairs"] += 1
        is_passage = any(
            token in cam_name
            for token in ("link4_sensor", "link5_front", "link5_back", "link6_sensor")
        )
        if is_passage:
            state["pairs_passage"] += 1
        if best is not None:
            state["engaged"] += 1
            step_engaged += 1
            if is_passage:
                state["engaged_passage"] += 1
        if best_enc is not None:
            state["engaged_encoder"] += 1
    state["episode_engaged_steps"].append(step_engaged)
    return step_clear


def render_row(job: dict[str, Any]) -> dict[str, Any]:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("MLSPACES_ASSETS_DIR", str(ROOT / "assets"))
    os.environ.pop("DISPLAY", None)

    from molmo_spaces.data_generation.pipeline import cleanup_episode_resources

    role_index = int(job["role_index"])
    cfg = json.loads(Path(job["config_path"]).read_text())
    result = json.loads(Path(job["result_path"]).read_text())
    trajectory = json.loads(Path(job["trajectory_path"]).read_text())
    steps = list(trajectory["steps"])
    if not steps:
        raise RuntimeError(f"row {role_index} trajectory is empty")
    if result.get("config_sha256") != cfg["config_sha256"]:
        raise RuntimeError(f"row {role_index} is not the design-review config")

    output = Path(job["video_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    spots = set(spot_check_indices(len(steps)))
    hold_frames = max(1, int(round(FPS * HOLD_FINAL_S)))
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (PANE_WH[0] * 2, PANE_WH[1]),
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer: {output}")

    task = sampler = scratch = None
    clutter_frames = 0
    video_frames = 0
    residuals: list[dict[str, Any]] = []
    clutter_by_phase: dict[str, int] = {}
    clutter_by_link: dict[str, int] = {}
    sensor_cfg = _sensor_config()
    metrics_state: dict[str, Any] = {
        "min_clearance": {},
        "samples": None,
        "vis_num": 0,
        "vis_den": 0,
        "pairs": 0,
        "pairs_passage": 0,
        "engaged": 0,
        "engaged_passage": 0,
        "engaged_encoder": 0,
        "episode_engaged_steps": [],
        "cam_names": [],
        "cam_ids": [],
    }
    try:
        task, sampler, scratch = _prepare_task(
            job["row"], result["selected_seed"], cfg
        )
        env = task.env
        model = env.current_model
        groups = geom_groups(model)
        metrics_state["cam_names"] = [
            resolve_camera_name(model, name) for name in sensor_cfg["sensor_names"]
        ]
        metrics_state["cam_ids"] = [
            int(model.camera(name).id) for name in metrics_state["cam_names"]
        ]
        for index, step, live_object, live_tcp, residuals in iter_replay_states(
            env, task, steps, spots
        ):
            del live_object, live_tcp
            position, forward, up = third_person_pose(env)
            third = _resize_pane(
                np.asarray(
                    env._render_frame(
                        position, forward, up, THIRD_PERSON_FOV, segmentation=False
                    )
                )
            )
            wrist_pos, wrist_fwd, wrist_up = wrist_camera_pose(env)
            wrist = _resize_pane(
                np.asarray(
                    env._render_frame(
                        wrist_pos, wrist_fwd, wrist_up, WRIST_FOV, segmentation=False
                    )
                )
            )
            if clutter_contact_active(env):
                clutter_frames += 1
                phase = str(step.get("policy_phase") or "unknown")
                clutter_by_phase[phase] = clutter_by_phase.get(phase, 0) + 1
                for link, count in clutter_pairs_by_link(env).items():
                    clutter_by_link[link] = clutter_by_link.get(link, 0) + count
            step_clear = accumulate_a0d_metrics(
                env, sensor_cfg=sensor_cfg, groups=groups, state=metrics_state
            )
            terminal = index == len(steps) - 1
            annotated = overlay_composite(
                wrist,
                third,
                role_index=role_index,
                clean_success=bool(result["clean_success"]),
                step=int(step["step"]),
                n_steps=len(steps),
                policy_phase=str(step["policy_phase"]),
                gripper_width_m=gripper_width_m(env),
                clutter_frames=clutter_frames,
                min_clearance_by_link=step_clear,
                terminal=terminal,
            )
            repeats = hold_frames if terminal else 1
            for _ in range(repeats):
                writer.write(annotated)
                video_frames += 1
    except Exception:
        writer.release()
        writer = None
        if output.exists():
            output.unlink()
        raise
    finally:
        if writer is not None:
            writer.release()
        cleanup_episode_resources(
            task=task,
            policy=None,
            task_sampler=sampler,
            preloaded_policy=None,
            close_task_sampler=sampler is not None,
        )
        if scratch is not None:
            shutil.rmtree(scratch, ignore_errors=True)

    if not residuals:
        if output.exists():
            output.unlink()
        raise RuntimeError(f"row {role_index} produced no faithfulness residuals")

    pairs = int(metrics_state["pairs"])
    vis_den = int(metrics_state["vis_den"])
    a0d = {
        "skin_engagement": (
            metrics_state["engaged"] / pairs if pairs else 0.0
        ),
        "skin_engagement_passage_link4_6": (
            metrics_state["engaged_passage"] / metrics_state["pairs_passage"]
            if metrics_state["pairs_passage"]
            else 0.0
        ),
        "skin_engagement_encoder_range": (
            metrics_state["engaged_encoder"] / pairs if pairs else 0.0
        ),
        "wrist_visibility": (
            metrics_state["vis_num"] / vis_den if vis_den else 0.0
        ),
        "mean_engaged_sensors_per_step": (
            float(np.mean(metrics_state["episode_engaged_steps"]))
            if metrics_state["episode_engaged_steps"]
            else 0.0
        ),
        "min_clearance_by_link_m": {
            key: None if value is None else float(value)
            for key, value in metrics_state["min_clearance"].items()
        },
        "passage_links": list(PASSAGE_LINKS),
        "n_step_sensor_pairs": pairs,
        "n_engaged_pairs": int(metrics_state["engaged"]),
    }
    return {
        "role_index": role_index,
        "episode_id": result["episode_id"],
        "config_sha256": result["config_sha256"],
        "selected_seed": result["selected_seed"],
        "clean_success": bool(result["clean_success"]),
        "task_success": bool(result["task_success"]),
        "terminal_policy_phase": result.get("terminal_policy_phase"),
        "terminal_action_index": result.get("terminal_action_index"),
        "trajectory_n": len(steps),
        "clip": output.name,
        "video_frames": video_frames,
        "fps": FPS,
        "hold_final_s": HOLD_FINAL_S,
        "faithfulness": {
            "step0_tolerance_m": INDEXING_TOLERANCE_M,
            "later_step_tolerance_m": DERIVED_TOLERANCE_M,
            "spot_check_indices": sorted(spots),
            "max_object_residual_m": max(
                item["object_position_residual_m"] for item in residuals
            ),
            "max_tcp_residual_m": max(
                item["tcp_position_residual_m"] for item in residuals
            ),
            "residuals": residuals,
        },
        "running_clutter_frames": clutter_frames,
        "clutter_contact_by_phase": clutter_by_phase,
        "clutter_contact_by_link": clutter_by_link,
        "a0d_metrics": a0d,
        "video_sha256": sha256_file(output),
    }


def write_json(path: Path, document: dict[str, Any], hash_key: str) -> None:
    payload = dict(document)
    payload.pop(hash_key, None)
    document[hash_key] = sha256_payload(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")


def build_jobs(
    contract: dict[str, Any], output_dir: Path, role_indices: list[int] | None
) -> list[dict[str, Any]]:
    wanted = None if role_indices is None else set(role_indices)
    jobs = []
    for row in contract["expert_screen_rows"]:
        index = int(row["role_index"])
        if wanted is not None and index not in wanted:
            continue
        directory = row_directory(index, row["episode_id"])
        result_path = directory / "result.json"
        if not result_path.is_file():
            continue
        result = json.loads(result_path.read_text())
        stem = clip_stem(index, result)
        jobs.append(
            {
                "role_index": index,
                "row": row,
                "config_path": str(CONFIG_PATH),
                "result_path": str(result_path),
                "trajectory_path": str(directory / "trajectory.json"),
                "video_path": str(output_dir / f"{stem}.mp4"),
            }
        )
    jobs.sort(key=lambda item: item["role_index"])
    return jobs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rows", help="Comma-separated role indices.")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise SystemExit("workers must be in [1, 8]")
    assert_replay_source_is_forward_only()
    contract = load_design_review_contract(CONFIG_PATH)
    role_indices = None
    if args.rows:
        role_indices = [int(item.strip()) for item in args.rows.split(",") if item.strip()]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(contract, output_dir, role_indices)
    if not jobs:
        raise SystemExit("no design-review trajectories to render")
    clips: list[dict[str, Any]] = []
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=context,
        max_tasks_per_child=1,
    ) as executor:
        futures = {executor.submit(render_row, job): job for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            job = futures[future]
            try:
                clip = future.result()
            except Exception as error:
                for pending in futures:
                    pending.cancel()
                raise RuntimeError(
                    f"replay failed for row {job['role_index']}: {error}"
                ) from error
            clips.append(clip)
            print(
                f"row={clip['role_index']:02d} clip={clip['clip']} "
                f"skin={clip['a0d_metrics']['skin_engagement']:.4f} "
                f"wrist={clip['a0d_metrics']['wrist_visibility']:.4f}",
                flush=True,
            )
    clips.sort(key=lambda item: item["role_index"])
    manifest = {
        "schema_version": "pact_place_v7_design_review_replay_v1",
        "status": "replay_rendered",
        "created_utc": utc_now(),
        "config_sha256": contract["config_sha256"],
        "source_screen": str(SCREEN_ROOT.relative_to(ROOT)),
        "output_dir": str(output_dir),
        "replay_only": True,
        "physics_stepped": False,
        "expert_rerun": False,
        "fps": FPS,
        "n": len(clips),
        "clips": clips,
        "required_scene_xml": REQUIRED_SCENE_XML,
        "clutter_bodies_asserted_against_config": True,
        "clutter_pool_slots_asserted": list(V7_CLUTTER_POOL_SLOT_NAMES),
        "authorizes_collection_or_training": False,
        "role": "human_design_review_not_a_gate",
        "overlay_fields": list(OVERLAY_FIELDS),
    }
    write_json(output_dir / "manifest.json", manifest, "manifest_sha256")
    print(json.dumps({"output_dir": str(output_dir), "n": len(clips)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
