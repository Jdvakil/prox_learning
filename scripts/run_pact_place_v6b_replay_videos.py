#!/usr/bin/env python3
"""Replay-render the 24 attempt-6b place-corridor episodes from recorded qpos.

Restore `data.qpos`, call `mj_forward`, and render. Do not step physics, do not
run the expert, and do not modify Phase-0 artifacts. Do not edit the v5 renderer.
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import hashlib
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
    load_contract,
    sha256_file,
    sha256_payload,
)
from run_pact_place_expert_screen import (  # noqa: E402
    _make_config,
    verify_protected_artifacts,
)

CONFIG_PATH = ROOT / "configs/pact_place_corridor_v6b.json"
SCREEN_ROOT = ROOT / "diagnostics_output/pact_place_corridor_v6b"
DEFAULT_OUTPUT = ROOT / "diagnostics_output/pact_place_corridor_v6b_videos"
V6B_CONFIG_SHA256 = (
    "ebf1be0359b6ff810772a3d4bbb0adf8913710fae4eec14c88a8f403a70e3671"
)
PASS_TOKEN = "PACT_PLACE_CORRIDOR_PHASE0_PASS"
REQUIRED_SCENE_XML = "pact_place_corridor_v3.xml"
CLUTTER_HEIGHT_M = 0.06
SHELF_TOP_Z = 0.72

# One recorded control step per frame at the 66 ms control period (true 1×).
FPS = 1000.0 / 66.0
HOLD_FINAL_S = 1.0
# Step 0 must match at 1e-6 m: that is the qpos-indexing check.
# Later frames can drift by tens of microns because xpos/TCP are derived after
# mj_forward, while the recording was taken after mj_step with contacts.
INDEXING_TOLERANCE_M = 1e-6
DERIVED_TOLERANCE_M = 1e-3
PANE_WH = (624, 352)
THIRD_PERSON_FOV = 58.0
WRIST_FOV = 56.74
WRIST_CAMERA_MJCF = "robot_0/gripper/wrist_camera"
CAMERA_REFERENCE_BODY = "robot_0/fr3_link0"
CAMERA_OFFSET = np.asarray([-1.05, -0.55, 1.30], dtype=np.float64)
LOOKAT_OFFSET = np.asarray([0.55, 0.0, 0.45], dtype=np.float64)
OVERLAY_FIELDS = (
    "policy_phase + step",
    "gripper width (m)",
    "object z minus start z",
    "TCP z minus carry-plane z",
    "running hazard_bar frames",
    "running clutter frames",
)

FAIL_CLIP_STEMS = {
    2: "row02_FAIL_lift_cup_dropped",
    9: "row09_FAIL_outbound_approach_ik_cascade",
    14: "row14_FAIL_lift_cup_dropped",
    22: "row22_FAIL_outbound_approach_ik_cascade",
}

CRIB_FAILURES = (
    {
        "row": 2,
        "steps": 139,
        "phase": "lift",
        "recorded": (
            "empty_gripper on lift; gripper_width_min_m = 0; cup max z 0.8018 vs "
            "0.7937 start; ended z 0.7527"
        ),
        "watch_for": "Grasp closes but the cup is knocked down rather than lifted.",
    },
    {
        "row": 9,
        "steps": 247,
        "phase": "outbound_approach",
        "recorded": (
            "lifted and carried, then 8 sequential IK failures; cup still held"
        ),
        "watch_for": "Arm freezes mid-bow. Clutter is not in contact on this row.",
    },
    {
        "row": 14,
        "steps": 144,
        "phase": "lift",
        "recorded": (
            "empty_gripper on lift; gripper_width_min_m = 0; cup max z 0.7995 vs "
            "0.7937 start; ended z 0.7527"
        ),
        "watch_for": "Same cup-drop as row 2. Clutter is not in contact.",
    },
    {
        "row": 22,
        "steps": 204,
        "phase": "outbound_approach",
        "recorded": (
            "lifted and carried, then 8 sequential IK failures; cup still held"
        ),
        "watch_for": "Same reach/IK class as row 9 and v5 row 3.",
    },
)

FAIL_RECORD_PATHS = (
    ROOT / "diagnostics_output/pact_place_corridor_v5/expert_screen.json",
    ROOT / "diagnostics_output/pact_place_corridor_v6b/expert_screen.json",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def clip_stem(role_index: int, *, clean_success: bool) -> str:
    if clean_success:
        return f"row{role_index:02d}_clean_success"
    if role_index not in FAIL_CLIP_STEMS:
        raise ValueError(f"uncatalogued clean-failure row {role_index}")
    return FAIL_CLIP_STEMS[role_index]


def row_directory(role_index: int, episode_id: str) -> Path:
    return SCREEN_ROOT / "expert_screen_rows" / f"{role_index:02d}_{episode_id[:16]}"


def spot_check_indices(n: int) -> list[int]:
    if n <= 0:
        raise ValueError("trajectory is empty")
    if n == 1:
        return [0]
    wanted = [0, n // 4, n // 2, (3 * n) // 4, n - 1]
    return sorted(dict.fromkeys(wanted))


def carry_plane_z_m(steps: list[dict[str, Any]]) -> float | None:
    last_lift = None
    first_outbound = None
    for step in steps:
        phase = str(step.get("policy_phase") or "")
        tcp = step.get("tcp_position_m")
        if tcp is None:
            continue
        z = float(tcp[2])
        if phase == "lift":
            last_lift = z
        if phase == "outbound_approach" and first_outbound is None:
            first_outbound = z
    if last_lift is not None:
        return last_lift
    return first_outbound


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
    object_dz_m: float,
    tcp_plane_dz_m: float | None,
    hazard_bar_frames: int,
    clutter_frames: int,
    terminal: bool,
) -> np.ndarray:
    width, height = PANE_WH
    for name, frame in (("wrist", wrist_rgb), ("third-person", third_rgb)):
        if frame.shape != (height, width, 3):
            raise ValueError(f"{name} render shape {frame.shape} != {(height, width, 3)}")
    composite = np.concatenate([wrist_rgb, third_rgb], axis=1)
    frame = cv2.cvtColor(composite, cv2.COLOR_RGB2BGR)
    shade = frame.copy()
    cv2.rectangle(shade, (0, 0), (frame.shape[1], 78), (0, 0, 0), thickness=-1)
    cv2.rectangle(
        shade, (0, height - 28), (frame.shape[1], height), (0, 0, 0), thickness=-1
    )
    frame = cv2.addWeighted(shade, 0.62, frame, 0.38, 0.0)
    status = "clean" if clean_success else "FAIL"
    plane = "n/a" if tcp_plane_dz_m is None else f"{tcp_plane_dz_m:+.4f} m"
    terminal_tag = "  TERMINAL" if terminal else ""
    cv2.putText(
        frame,
        f"row {role_index:02d}  {status}  {policy_phase}  step {step}/{n_steps - 1}{terminal_tag}",
        (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (25, 25, 245) if terminal else (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        (
            f"gripper {gripper_width_m:.3f} m   "
            f"obj dz {object_dz_m:+.4f} m   "
            f"tcp-plane {plane}   "
            f"hazard_bar {hazard_bar_frames}   "
            f"clutter {clutter_frames}"
        ),
        (12, 58),
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


def crib_markdown(*, output_dir: Path, fps: float) -> str:
    rows = [
        "| Row | Steps | Phase | Recorded | Watch for |",
        "|---:|---:|---|---|---|",
    ]
    for item in CRIB_FAILURES:
        rows.append(
            f"| {item['row']} | {item['steps']} | `{item['phase']}` | "
            f"{item['recorded']} | {item['watch_for']} |"
        )
    fail_names = "\n".join(f"- `{stem}.mp4`" for stem in FAIL_CLIP_STEMS.values())
    return "\n".join(
        [
            "# Attempt-6b place-corridor replay crib",
            "",
            "These clips restore the recorded full-model `qpos` and render. They are",
            "not a re-run: physics is not stepped, the expert is not executed, and the",
            "Phase-0 PASS record is unchanged. Step 0 after `mj_forward` matches the",
            "recording at 1e-6 m (qpos indexing). Later frames can drift by tens of",
            "microns because xpos/TCP are derived after `mj_forward`, not `mj_step`.",
            "",
            "The renderer refuses to run unless the scene is `pact_place_corridor_v3.xml`",
            "and every `pact_clutter_*` body matches the frozen config plus row jitter.",
            "",
            f"Directory: `{output_dir}`",
            f"Playback: **1×** of the 66 ms control period (`fps = {fps:.6g}`).",
            "Layout: wrist (left) and the existing render-only third-person camera (right).",
            "",
            "Named failure clips:",
            "",
            fail_names,
            "",
            "Clean successes are `rowXX_clean_success.mp4`.",
            "",
            "## Watch the four failures against a stated claim",
            "",
            *rows,
            "",
            "None of the four official failures has `pact_clutter` contact. Rows 2 and 14",
            "are the v5 row-10 cup-drop class. Rows 9 and 22 are the v5 row-3 IK-cascade",
            "class. The clutter resite did not create a new failure mode on this seed.",
            "",
        ]
    )


def apply_recorded_qpos(env, qpos_list: list[float]) -> np.ndarray:
    """Restore the recorded generalized position. Forward kinematics only."""
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


def hazard_bar_active(env) -> bool:
    from molmo_spaces.tasks.pact_place_contact_audit import (
        classify_contact,
        place_environment_contact_pairs,
    )

    return any(
        classify_contact(pair) == "hazard_bar"
        for pair in place_environment_contact_pairs(env)
    )


def clutter_contact_active(env) -> bool:
    from molmo_spaces.tasks.pact_place_contact_audit import (
        classify_contact,
        place_environment_contact_pairs,
    )

    return any(
        classify_contact(pair) == "clutter"
        for pair in place_environment_contact_pairs(env)
    )


def _max_abs_diff(left, right) -> float:
    return float(np.max(np.abs(np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64))))


def _resize_pane(frame: np.ndarray) -> np.ndarray:
    width, height = PANE_WH
    if frame.shape[0] != height or frame.shape[1] != width:
        return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return frame


def frozen_phase0_paths() -> list[Path]:
    paths = list(FAIL_RECORD_PATHS)
    rows = SCREEN_ROOT / "expert_screen_rows"
    for directory in sorted(path for path in rows.iterdir() if path.is_dir()):
        for name in (
            "result.json",
            "trajectory.json",
            "initial_observation_accepted.json",
        ):
            paths.append(directory / name)
    return paths


def snapshot_hashes(paths: list[Path]) -> dict[str, str]:
    observed: dict[str, str] = {}
    for path in paths:
        relative = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
        observed[relative] = sha256_file(path)
    return observed


def assert_hashes_unchanged(expected: dict[str, str]) -> None:
    changed = {
        relative: {"expected": digest, "actual": sha256_file(ROOT / relative)}
        for relative, digest in expected.items()
        if sha256_file(ROOT / relative) != digest
    }
    if changed:
        raise RuntimeError(f"frozen Phase-0 artifacts changed: {changed}")


def assert_replay_source_is_forward_only(source: str | None = None) -> None:
    text = source if source is not None else Path(__file__).read_text()
    tree = ast.parse(text)
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        in {"apply_recorded_qpos", "iter_replay_states", "render_row"}
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


def iter_replay_states(env, task, steps: list[dict[str, Any]], spot_indices: set[int]):
    """Yield restored states. Forward only — never mj_step."""
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



def _assert_static_furniture(task, row: dict[str, Any]) -> None:
    """Tray and clutter are not in qpos, so object/TCP residuals cannot catch a
    wrong-scene render. Check both against the frozen config and this row's jitter."""
    cfg = json.loads(CONFIG_PATH.read_text())
    want = np.asarray(cfg["scene"]["place_receptacle_center_xyz_m"], dtype=float)
    model = task.env.current_model
    data = task.env.current_data
    got = np.asarray(data.body(model.body("place_receptacle").id).xpos, dtype=float)
    if float(np.linalg.norm(got[:2] - want[:2])) > 1e-3:
        raise RuntimeError(
            f"rendered receptacle at {got[:2].tolist()} but config says "
            f"{want[:2].tolist()} — wrong scene loaded"
        )
    slots = cfg["phase0_gate"]["clutter_sweep_v6b"]["chosen_slots_xy_m"]
    height = float(cfg["phase0_gate"]["clutter_sweep_v6b"]["chosen_height_m"])
    if abs(height - CLUTTER_HEIGHT_M) > 1e-9:
        raise RuntimeError(f"unexpected clutter height {height}")
    x_jitter = row.get("clutter_x_jitter_m") or {}
    y_jitter = row.get("clutter_y_jitter_m") or {}
    want_z = SHELF_TOP_Z + height / 2.0
    for slot, xy in slots.items():
        body = f"pact_clutter_{slot}"
        expected = np.asarray(
            [
                float(xy[0]) + float(x_jitter.get(slot, 0.0)),
                float(xy[1]) + float(y_jitter.get(slot, 0.0)),
                want_z,
            ],
            dtype=float,
        )
        observed = np.asarray(data.body(model.body(body).id).xpos, dtype=float)
        if float(np.linalg.norm(observed - expected)) > 1e-3:
            raise RuntimeError(
                f"rendered {body} at {observed.tolist()} but config+jitter says "
                f"{expected.tolist()} — wrong scene or clutter not applied"
            )


def _prepare_task(row: dict[str, Any], selected_seed: dict[str, int]):
    from molmo_spaces.data_generation.pipeline import cleanup_episode_resources
    from molmo_spaces.data_generation.runtime_compat import assert_supported_runtime

    assert_supported_runtime(strict=True)
    scratch = Path(tempfile.mkdtemp(prefix="pact_place_v6b_replay_"))
    task = sampler = None
    try:
        scene_xml = ROOT / json.loads(CONFIG_PATH.read_text())["scene"]["xml"]
        if scene_xml.name != REQUIRED_SCENE_XML:
            raise RuntimeError(
                f"v6b must render {REQUIRED_SCENE_XML}, got {scene_xml.name}"
            )
        config = _make_config(scratch / "dummy.json", scene_xml=scene_xml)
        sampler = config.task_sampler_config.task_sampler_class(config)
        sampler.seed_task_sampling(int(selected_seed["seed_u32"]))
        sampler.set_pact_manifest_row(row)
        task = sampler.sample_task(house_index=int(row["scene_template_house_index"]))
        if task is None:
            raise RuntimeError("sample_task returned None for the recorded seed")
        task.reset()
        _assert_static_furniture(task, row)
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


def render_row(job: dict[str, Any]) -> dict[str, Any]:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("MLSPACES_ASSETS_DIR", str(ROOT / "assets"))
    os.environ.pop("DISPLAY", None)

    from molmo_spaces.data_generation.pipeline import cleanup_episode_resources

    role_index = int(job["role_index"])
    result = json.loads(Path(job["result_path"]).read_text())
    trajectory = json.loads(Path(job["trajectory_path"]).read_text())
    steps = list(trajectory["steps"])
    if not steps:
        raise RuntimeError(f"row {role_index} trajectory is empty")
    if result.get("config_sha256") != V6B_CONFIG_SHA256:
        raise RuntimeError(f"row {role_index} is not the frozen v6b screen")
    payload = dict(result)
    observed = payload.pop("result_sha256", None)
    if observed != sha256_payload(payload):
        raise RuntimeError(f"row {role_index} result self-hash mismatch")
    if sha256_file(Path(job["trajectory_path"])) != result["trajectory_sha256"]:
        raise RuntimeError(f"row {role_index} trajectory hash mismatch")

    output = Path(job["video_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    start_z = float(steps[0]["object_position_m"][2])
    plane_z = carry_plane_z_m(steps)
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
    first_pose = None
    hazard_frames = 0
    clutter_frames = 0
    video_frames = 0
    residuals: list[dict[str, Any]] = []
    try:
        task, sampler, scratch = _prepare_task(job["row"], result["selected_seed"])
        env = task.env
        for index, step, live_object, live_tcp, residuals in iter_replay_states(
            env, task, steps, spots
        ):
            position, forward, up = third_person_pose(env)
            if first_pose is None:
                first_pose = {
                    "position": position.tolist(),
                    "forward": forward.tolist(),
                    "up": up.tolist(),
                }
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
            if hazard_bar_active(env):
                hazard_frames += 1
            if clutter_contact_active(env):
                clutter_frames += 1
            tcp_plane = None if plane_z is None else float(live_tcp[2] - plane_z)
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
                object_dz_m=float(live_object[2] - start_z),
                tcp_plane_dz_m=tcp_plane,
                hazard_bar_frames=hazard_frames,
                clutter_frames=clutter_frames,
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

    return {
        "role_index": role_index,
        "episode_id": result["episode_id"],
        "config_sha256": result["config_sha256"],
        "selected_seed": result["selected_seed"],
        "clean_success": bool(result["clean_success"]),
        "task_success": bool(result["task_success"]),
        "terminal_policy_phase": result["terminal_policy_phase"],
        "terminal_action_index": result["terminal_action_index"],
        "trajectory_n": len(steps),
        "clip": output.name,
        "video_frames": video_frames,
        "fps": FPS,
        "hold_final_s": HOLD_FINAL_S,
        "first_frame_world_pose": first_pose,
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
        "running_hazard_bar_frames": hazard_frames,
        "running_clutter_frames": clutter_frames,
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
    wanted = set(range(24) if role_indices is None else role_indices)
    jobs = []
    for row in contract["expert_screen_rows"]:
        index = int(row["role_index"])
        if index not in wanted:
            continue
        directory = row_directory(index, row["episode_id"])
        result = json.loads((directory / "result.json").read_text())
        stem = clip_stem(index, clean_success=bool(result["clean_success"]))
        jobs.append(
            {
                "role_index": index,
                "row": row,
                "result_path": str(directory / "result.json"),
                "trajectory_path": str(directory / "trajectory.json"),
                "video_path": str(output_dir / f"{stem}.mp4"),
            }
        )
    jobs.sort(key=lambda item: item["role_index"])
    if role_indices is None and len(jobs) != 24:
        raise RuntimeError(f"expected 24 replay jobs, got {len(jobs)}")
    return jobs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--rows",
        help="Comma-separated role indices. Default: all 24.",
    )
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise SystemExit("workers must be in [1, 8]")
    usage = shutil.disk_usage(ROOT)
    if usage.free < 2 * 1024**3:
        raise SystemExit(f"disk too full: {usage.free / 1024**3:.1f} GiB free")
    assert_replay_source_is_forward_only()
    contract = load_contract(CONFIG_PATH)
    if contract["config_sha256"] != V6B_CONFIG_SHA256:
        raise SystemExit("v6b contract hash changed")
    verify_protected_artifacts(contract)
    frozen = snapshot_hashes(frozen_phase0_paths())
    role_indices = None
    if args.rows:
        role_indices = [int(item.strip()) for item in args.rows.split(",") if item.strip()]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(contract, output_dir, role_indices)
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
                f"obj_res={clip['faithfulness']['max_object_residual_m']:.3e} "
                f"tcp_res={clip['faithfulness']['max_tcp_residual_m']:.3e}",
                flush=True,
            )
    clips.sort(key=lambda item: item["role_index"])
    crib_path = output_dir / "CRIB.md"
    crib_path.write_text(crib_markdown(output_dir=output_dir, fps=FPS))
    manifest = {
        "schema_version": "pact_place_v6b_replay_videos_v1",
        "status": "replay_rendered",
        "created_utc": utc_now(),
        "config_sha256": contract["config_sha256"],
        "source_screen": str(SCREEN_ROOT.relative_to(ROOT)),
        "output_dir": str(output_dir),
        "replay_only": True,
        "physics_stepped": False,
        "expert_rerun": False,
        "determinism_gate": "not_applicable_qpos_replay",
        "fps": FPS,
        "hold_final_s": HOLD_FINAL_S,
        "resolution_width_height": [PANE_WH[0] * 2, PANE_WH[1]],
        "panes": ["wrist_camera", "render_only_third_person"],
        "camera_reference_body": CAMERA_REFERENCE_BODY,
        "camera_offset_m": CAMERA_OFFSET.tolist(),
        "lookat_offset_m": LOOKAT_OFFSET.tolist(),
        "overlay_fields": list(OVERLAY_FIELDS),
        "step0_faithfulness_tolerance_m": INDEXING_TOLERANCE_M,
        "later_step_faithfulness_tolerance_m": DERIVED_TOLERANCE_M,
        "n": len(clips),
        "clips": clips,
        "crib_path": str(crib_path),
        "required_scene_xml": REQUIRED_SCENE_XML,
        "clutter_bodies_asserted_against_config": True,
        "authorizes_collection_or_training": False,
        "phase0_reopened": False,
    }
    write_json(output_dir / "manifest.json", manifest, "manifest_sha256")
    verify_protected_artifacts(contract)
    assert_hashes_unchanged(frozen)
    print(json.dumps({"output_dir": str(output_dir), "n": len(clips)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
