#!/usr/bin/env python3
"""V10.2 real-time review renderer.

Repairs, for V10.2 only, the two V10.1 playback defects: every policy frame is
rendered (``frame_stride=1``) and the output frame rate is the policy timestep
(1000/66 fps), so one second of video is one second of simulated policy time.
The V10.1 renderer in ``run_pact_place_v9_v1b_review.py`` is left untouched.

The wrist pane is rendered from the untinted scene: it is the student's own
view and the review tint must never enter it.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MOLMO = ROOT / "submodules" / "molmospaces"
for _path in (ROOT / "scripts", MOLMO):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from pact_place_v102_raised_pendant_contract import (  # noqa: E402
    POLICY_TIMESTEP_MS,
    REVIEW_FPS,
    REVIEW_FRAME_STRIDE,
    REVIEW_TINT_LABEL,
    SAMPLER_CLASS,
)

# Pendant-focused side view: orthogonal to the front review camera so the
# arm/stem depth relationship is legible instead of inferred.
PENDANT_SIDE_CAM_POS = np.asarray([0.70, -1.15, 1.30], dtype=float)
PENDANT_SIDE_CAM_TARGET = np.asarray([0.70, 0.05, 1.24], dtype=float)
PENDANT_SIDE_CAM_FOV = 45.0
STEM_REVIEW_RGBA = (0.95, 0.75, 0.10, 1.0)
CROSSBAR_REVIEW_RGBA = (0.20, 0.85, 0.95, 1.0)
LOBE_REVIEW_RGBA = (0.95, 0.35, 0.05, 1.0)
VESSEL_REVIEW_RGBA = (0.35, 0.75, 0.35, 1.0)


def video_duration_s(n_policy_frames: int, *, fps: float = REVIEW_FPS) -> float:
    """Wall-clock duration of a stride-1 V10.2 review clip."""
    return float(int(n_policy_frames)) / float(fps)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _slug(value: str) -> str:
    return (
        "".join(
            char if char.isalnum() or char in "-_" else "_" for char in str(value)
        ).strip("_")
        or "unknown"
    )


def clip_stem(role_index: int, side: str, result: dict[str, Any]) -> str:
    prefix = f"row{int(role_index):02d}_{side}"
    if result.get("clean_success"):
        return f"{prefix}_clean_success"
    cause = (result.get("failure_cause") or {}).get("code")
    if cause:
        return f"{prefix}_FAIL_{_slug(str(cause))}"
    return f"{prefix}_FAIL_{_slug(str(result.get('status') or 'unknown'))}"


def _put(frame, text, xy, scale, color, thick=1):
    cv2.putText(
        frame, str(text), xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA
    )


def _overlay(
    wrist_rgb: np.ndarray,
    third_rgb: np.ndarray,
    side_rgb: np.ndarray,
    *,
    pane_wh: tuple[int, int],
    header_lines: Sequence[str],
    footer: Sequence[tuple[str, tuple[int, int, int]]],
    contact: bool,
    terminal: bool,
    clean_success: bool,
) -> np.ndarray:
    width, height = pane_wh
    composite = np.concatenate([wrist_rgb, third_rgb, side_rgb], axis=1)
    frame = cv2.cvtColor(composite, cv2.COLOR_RGB2BGR)
    lines = [str(line) for line in header_lines]
    header_h = 26 + 22 * len(lines)
    shade = frame.copy()
    cv2.rectangle(shade, (0, 0), (frame.shape[1], header_h), (0, 0, 0), thickness=-1)
    cv2.rectangle(
        shade, (0, height - 30), (frame.shape[1], height), (0, 0, 0), thickness=-1
    )
    frame = cv2.addWeighted(shade, 0.70, frame, 0.30, 0.0)
    colour = (
        (60, 60, 255)
        if contact
        else ((120, 255, 120) if (terminal and clean_success) else (240, 240, 240))
    )
    for index, line in enumerate(lines):
        _put(frame, line, (12, 24 + index * 22), 0.44, colour if index == 0 else (230, 230, 230), 1)
    for pane_index, (label, label_colour) in enumerate(footer):
        _put(
            frame,
            label,
            (pane_index * width + 12, height - 10),
            0.40,
            label_colour,
            1,
        )
    return frame


def _apply_review_tint(model, pendant_geoms: dict[str, str], vessel_slots: set[str]) -> dict[int, np.ndarray]:
    """Tint stems, crossbar, lobes and vessels. Returns the saved rgba rows."""
    saved: dict[int, np.ndarray] = {}

    def _set(geom_id: int, rgba) -> None:
        saved[geom_id] = np.asarray(model.geom_rgba[geom_id], dtype=float).copy()
        model.geom_rgba[geom_id] = np.asarray(rgba, dtype=float)

    for name, role in pendant_geoms.items():
        try:
            geom_id = int(model.geom(name).id)
        except KeyError:
            continue
        if int(model.geom_contype[geom_id]) == 0 and int(model.geom_conaffinity[geom_id]) == 0:
            continue
        if role == "stem":
            _set(geom_id, STEM_REVIEW_RGBA)
        elif role == "crossbar":
            _set(geom_id, CROSSBAR_REVIEW_RGBA)
        else:
            _set(geom_id, LOBE_REVIEW_RGBA)
    for geom_id in range(int(model.ngeom)):
        geom_name = model.geom(geom_id).name or ""
        if any(f"pact_clutter_{slot}" in geom_name for slot in vessel_slots):
            _set(geom_id, VESSEL_REVIEW_RGBA)
    return saved


def _restore_rgba(model, saved: dict[int, np.ndarray]) -> None:
    for geom_id, rgba in saved.items():
        model.geom_rgba[geom_id] = rgba


def _render_placeholder(job: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    from run_pact_place_v6c_replay_videos import PANE_WH

    output = Path(job["video_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(REVIEW_FPS),
        (PANE_WH[0] * 3, PANE_WH[1]),
    )
    frame = np.zeros((PANE_WH[1], PANE_WH[0] * 3, 3), dtype=np.uint8)
    _put(
        frame,
        f"row {int(job['role_index']):02d} ({job['row']['intrusion_side']}): "
        f"{result.get('status', 'failed')}",
        (35, 145),
        0.72,
        (255, 255, 255),
        2,
    )
    _put(frame, str(result.get("error") or "no trajectory recorded")[:120], (35, 190), 0.42, (180, 180, 255))
    _put(frame, f"{REVIEW_TINT_LABEL}: n/a", (35, 230), 0.42, (180, 180, 255))
    for _ in range(int(round(REVIEW_FPS * 2))):
        writer.write(frame)
    writer.release()
    return {
        "role_index": int(job["role_index"]),
        "episode_id": result.get("episode_id", job["row"]["episode_id"]),
        "clean_success": False,
        "placeholder": True,
        "fps": float(REVIEW_FPS),
        "frame_stride": int(REVIEW_FRAME_STRIDE),
        "video_path": str(output.relative_to(ROOT)),
        "video_sha256": sha256_file(output),
    }


def render_v102_review_video(job: dict[str, Any]) -> dict[str, Any]:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("MLSPACES_ASSETS_DIR", str(ROOT / "assets"))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.pop("DISPLAY", None)

    result = json.loads(Path(job["result_path"]).read_text())
    if result.get("status") != "complete" or not Path(job["trajectory_path"]).is_file():
        return _render_placeholder(job, result)

    from molmo_spaces.data_generation.pipeline import cleanup_episode_resources
    from run_pact_place_expert_screen import _make_config
    from run_pact_place_v6c_replay_videos import (
        PANE_WH,
        THIRD_PERSON_FOV,
        WRIST_FOV,
        _resize_pane,
        apply_recorded_qpos,
        third_person_pose,
        wrist_camera_pose,
    )

    row = job["row"]
    scene_xml = Path(job["scene_xml"])
    steps = list(json.loads(Path(job["trajectory_path"]).read_text())["steps"])
    n_steps = len(steps)
    output = Path(job["video_path"])
    output.parent.mkdir(parents=True, exist_ok=True)

    assembly = dict(row.get("pact_v10_pendant_assembly") or {})
    pendant_geoms = {
        str(item["geom"]): str(item["role"])
        for item in assembly.get("components") or []
        if item.get("active")
    }
    vessel_slots = {
        str(item["palette_slot"])
        for item in row["pact_clutter_layout"]["objects"]
        if str(item.get("role")) in {"inbound_vessel", "outbound_vessel"}
    }

    task = sampler = scratch = None
    writer = None
    try:
        scratch = Path(tempfile.mkdtemp(prefix="pact_place_v102_review_"))
        config = _make_config(
            scratch / "dummy.json", scene_xml=scene_xml, sampler_class=SAMPLER_CLASS
        )
        sampler = config.task_sampler_config.task_sampler_class(config)
        sampler.seed_task_sampling(int(result["selected_seed"]["seed_u32"]))
        sampler.set_pact_manifest_row(row)
        task = sampler.sample_task(house_index=int(row["scene_template_house_index"]))
        env = task.env
        model = env.current_model

        side_delta = PENDANT_SIDE_CAM_TARGET - PENDANT_SIDE_CAM_POS
        side_fwd = side_delta / np.linalg.norm(side_delta)
        side_up = np.asarray([0.0, 0.0, 1.0], dtype=float)

        writer = cv2.VideoWriter(
            str(output),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(REVIEW_FPS),
            (PANE_WH[0] * 3, PANE_WH[1]),
        )
        render_indices = list(range(0, n_steps, int(REVIEW_FRAME_STRIDE)))
        if render_indices != list(range(n_steps)):
            raise RuntimeError("V10.2 review must render every policy frame")

        contact_frames = [
            index for index, step in enumerate(steps) if step.get("pendant_contact")
        ]
        clip_label = str(job.get("clip_label") or "")
        clean = bool(result.get("clean_success"))
        for step_idx in render_indices:
            step = steps[step_idx]
            apply_recorded_qpos(env, step["qpos"])

            # Wrist pane: untinted student view.
            w_pos, w_fwd, w_up = wrist_camera_pose(env)
            wrist_rgb = _resize_pane(
                np.asarray(
                    env._render_frame(w_pos, w_fwd, w_up, WRIST_FOV, segmentation=False)
                )
            )
            saved_rgba = _apply_review_tint(model, pendant_geoms, vessel_slots)
            try:
                tp_pos, tp_fwd, tp_up = third_person_pose(env)
                third_rgb = _resize_pane(
                    np.asarray(
                        env._render_frame(
                            tp_pos, tp_fwd, tp_up, THIRD_PERSON_FOV, segmentation=False
                        )
                    )
                )
                side_rgb = _resize_pane(
                    np.asarray(
                        env._render_frame(
                            PENDANT_SIDE_CAM_POS,
                            side_fwd,
                            side_up,
                            PENDANT_SIDE_CAM_FOV,
                            segmentation=False,
                        )
                    )
                )
            finally:
                _restore_rgba(model, saved_rgba)

            clearances = step.get("component_clearance_m") or {}
            clearance_text = "  ".join(
                f"{name} {'--' if clearances.get(name) is None else format(float(clearances[name]), '.3f')}m"
                for name in sorted(clearances)
            )
            speed = step.get("commanded_speed_m_s")
            realized = step.get("realized_tcp_speed_m_s")
            contact = bool(step.get("pendant_contact"))
            wall_s = float(step_idx) / float(REVIEW_FPS)
            header = [
                f"{clip_label}  frame {step_idx}/{n_steps - 1}  "
                f"sim {float(step.get('sim_time_s') or 0.0):7.3f}s  video {wall_s:6.3f}s  "
                f"({REVIEW_FPS:.4f} fps, stride {REVIEW_FRAME_STRIDE}, real time)",
                f"phase {step.get('policy_phase')}  segment {step.get('segment_name')}  "
                f"commanded {'--' if speed is None else format(float(speed), '.3f')} m/s  "
                f"realized {'--' if realized is None else format(float(realized), '.3f')} m/s",
                f"pendant clearance: {clearance_text or 'not measured'}",
                f"pendant contact: {'YES' if contact else 'no'}  "
                f"raw pairs {step.get('n_raw_pendant_contact_pairs')}  "
                f"classified mounted_fixture {step.get('n_classified_mounted_fixture_pairs')}",
            ]
            frame = _overlay(
                wrist_rgb,
                third_rgb,
                side_rgb,
                pane_wh=PANE_WH,
                header_lines=header,
                footer=[
                    ("wrist (RGB policy view, NO review tint)", (245, 245, 245)),
                    (f"third-person  [{REVIEW_TINT_LABEL}]", (160, 220, 255)),
                    (f"pendant side view  [{REVIEW_TINT_LABEL}]", (160, 220, 255)),
                ],
                contact=contact,
                terminal=step_idx == n_steps - 1,
                clean_success=clean,
            )
            writer.write(frame)
        writer.release()
        writer = None
        return {
            "role_index": int(job["role_index"]),
            "episode_id": result["episode_id"],
            "clean_success": clean,
            "placeholder": False,
            "source_steps": n_steps,
            "rendered_frames": len(render_indices),
            "frame_stride": int(REVIEW_FRAME_STRIDE),
            "fps": float(REVIEW_FPS),
            "policy_timestep_ms": float(POLICY_TIMESTEP_MS),
            "video_duration_s": video_duration_s(n_steps),
            "real_time": True,
            "terminal_frame_rendered": True,
            "contact_frames_rendered": contact_frames,
            "review_tint_label": REVIEW_TINT_LABEL,
            "wrist_pane_untinted": True,
            "panes": ["wrist", "third_person", "pendant_side_view"],
            "video_path": str(output.relative_to(ROOT)),
            "video_sha256": sha256_file(output),
        }
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
