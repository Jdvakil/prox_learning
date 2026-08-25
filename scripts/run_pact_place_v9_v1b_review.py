#!/usr/bin/env python3
"""Stage V1b -- human design review: capture 3 successes and 3 failures before the gate.

This is a mandatory human review stop. It runs until 3 clean successes and 3
failures have been captured (capped at 24 attempts), renders three-pane
composite replay videos with burned-in telemetry, and emits review_manifest.json
with authorizes_gate: false.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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

from pact_place_v9_contract import (
    LAYOUT_FAMILIES,
    PALETTE_PATH,
    build_layout,
    load_palette,
    sha256_payload,
)
from run_pact_place_expert_screen import (
    _make_config,
    run_row,
)

DEFAULT_OUTPUT_ROOT = ROOT / "diagnostics_output" / "pact_place_v93_v1b_review"
DEFAULT_DIAGNOSTIC_SUCCESS_ROOT = (
    ROOT / "diagnostics_output" / "pact_place_v93_success_episode_review"
)
DEFAULT_PROXIMITY_VALIDATION = (
    ROOT / "diagnostics_output" / "pact_place_v93_v0c4_causal_proximity" / "validation.json"
)
SCENE_XML = MOLMO / "molmo_spaces/data_generation/custom_scenes/pact_place_corridor_v5.xml"
V9_SAMPLER_CLASS = "PactPlaceCorridorV93Sampler"
BODY_GROUPS = ("cup", "fingers", "hand", "link7", "link6", "link5")
REVIEW_CAM_POS = np.asarray([0.05, -0.60, 1.34], dtype=float)
REVIEW_CAM_TARGET = np.asarray([0.74, 0.06, 1.12], dtype=float)
REVIEW_CAM_FOV = 52.0
VESSEL_REVIEW_RGBA = (0.95, 0.35, 0.05, 1.0)
PANE_WH = (480, 480)
FPS = 10


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _slug(value: str) -> str:
    return (
        "".join(char if char.isalnum() or char in "-_" else "_" for char in str(value)).strip("_")
        or "unknown"
    )


def clip_stem(attempt: int, side: str, result: dict[str, Any]) -> str:
    prefix = f"attempt{attempt:02d}_{side}"
    if result.get("clean_success"):
        return f"{prefix}_clean_success"
    cause = result.get("failure_cause") or {}
    if cause.get("code"):
        return f"{prefix}_FAIL_{_slug(str(cause['code']))}"
    phase = _slug(str(result.get("terminal_policy_phase") or result.get("status")))
    tracking = result.get("terminal_tracking") or {}
    branch = _slug(
        str(
            tracking.get("check_failure_branch")
            or tracking.get("branch")
            or result.get("error")
            or "unknown"
        )
    )
    return f"{prefix}_FAIL_{phase}_{branch}"


def _make_review_row(
    index: int,
    side: str,
    palette: list[dict[str, Any]],
    layout: dict[str, Any],
) -> dict[str, Any]:
    seed_u32 = 700000 + index * 1013
    seed_u64 = 700000 + index * 1013
    pair_index = index // 2
    pair_rng = np.random.default_rng(930000 + pair_index * 7919)
    clutter_x_jitter_m = {
        "01": float(np.round(pair_rng.uniform(-0.020, 0.020), 9)),
        "06": float(np.round(pair_rng.uniform(-0.005, 0.005), 9)),
    }
    clutter_y_jitter_m = {
        "01": float(np.round(pair_rng.uniform(-0.005, 0.005), 9)),
        "06": float(np.round(pair_rng.uniform(-0.010, 0.010), 9)),
    }
    panel_x_jitter_m = float(np.round(pair_rng.uniform(-0.015, 0.015), 9))
    panel_face_jitter_m = float(np.round(pair_rng.uniform(-0.005, 0.005), 9))
    episode_id = hashlib.sha256(f"pact_v9_review_{index}_{side}_{seed_u32}".encode()).hexdigest()
    row = {
        "role_index": index,
        "episode_id": episode_id,
        "family": str(layout["layout_family_id"]),
        "layout_family_id": str(layout["layout_family_id"]),
        "family_attempt": index,
        "intrusion_side": side,
        "scene_template_house_index": 1,
        "task_seed_u32": seed_u32,
        "task_seed_u64": seed_u64,
        "max_sampling_retries": 12,
        "clutter_x_jitter_m": clutter_x_jitter_m,
        "clutter_y_jitter_m": clutter_y_jitter_m,
        "panel_face_jitter_m": panel_face_jitter_m,
        "panel_x_jitter_m": panel_x_jitter_m,
        "paired_side_cell": pair_index,
        "target_x_jitter_m": 0.0,
        "target_y_jitter_m": 0.0,
        "sampler_class": V9_SAMPLER_CLASS,
        "pact_clutter_palette": palette,
        "pact_clutter_layout": layout,
    }
    row["row_sha256"] = sha256_payload(row)
    return row


def overlay(
    wrist_rgb: np.ndarray,
    third_rgb: np.ndarray,
    review_rgb: np.ndarray,
    *,
    pane_wh: tuple[int, int],
    attempt: int,
    intrusion_side: str,
    step: int,
    n_steps: int,
    policy_phase: str,
    active_maneuver: str,
    clearances: dict[str, float],
    min_link_m: float,
    in_wrist_fov: bool,
    skin_detect: bool,
    drift_m: float,
    terminal: bool,
    clean_success: bool,
    clip_label: str = "",
) -> np.ndarray:
    width, height = pane_wh
    composite = np.concatenate([wrist_rgb, third_rgb, review_rgb], axis=1)
    frame = cv2.cvtColor(composite, cv2.COLOR_RGB2BGR)
    shade = frame.copy()
    cv2.rectangle(shade, (0, 0), (frame.shape[1], 104), (0, 0, 0), thickness=-1)
    cv2.rectangle(shade, (0, height - 46), (frame.shape[1], height), (0, 0, 0), thickness=-1)
    frame = cv2.addWeighted(shade, 0.68, frame, 0.32, 0.0)

    put = lambda text, xy, scale, color, thick: cv2.putText(
        frame, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA
    )
    status_text = "CLEAN SUCCESS" if (terminal and clean_success) else ("FAIL" if terminal else "")
    clip_heading = clip_label or f"attempt {attempt:02d}"
    put(
        f"{clip_heading}  seed stratum {intrusion_side}  phase: {policy_phase}  "
        f"step {step}/{n_steps - 1}  " + status_text,
        (12, 24),
        0.52,
        (120, 255, 120)
        if (terminal and clean_success)
        else ((60, 60, 255) if terminal else (255, 255, 255)),
        2,
    )
    clr_text = "  ".join(f"{key} {clearances.get(key, float('nan')):.3f}m" for key in BODY_GROUPS)
    put(f"clearance:  {clr_text}", (12, 50), 0.44, (235, 235, 235), 1)
    put(
        f"maneuver: {active_maneuver or 'none'}    min link: {min_link_m:.3f}m    drift: {drift_m:.3f}m",
        (12, 74),
        0.48,
        (200, 240, 255),
        1,
    )
    put(
        f"active hazard in wrist FOV: {'YES' if in_wrist_fov else 'NO'}    "
        f"sensor-cone proxy: {'YES' if skin_detect else 'NO'}",
        (12, 98),
        0.42,
        (120, 235, 120) if skin_detect else (180, 180, 220),
        1,
    )
    put("wrist (RGB policy view)", (12, height - 20), 0.42, (245, 245, 245), 1)
    put("third-person", (width + 12, height - 20), 0.42, (245, 245, 245), 1)
    put(
        "corridor review (original panel, blocker orange)",
        (2 * width + 12, height - 20),
        0.42,
        (245, 245, 245),
        1,
    )
    return frame


def _render_placeholder(job: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    output = Path(job["video_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (PANE_WH[0] * 3, PANE_WH[1]),
    )
    frame = np.zeros((PANE_WH[1], PANE_WH[0] * 3, 3), dtype=np.uint8)
    cv2.putText(
        frame,
        f"Attempt {job['attempt']:02d} ({job['row']['intrusion_side']}): {result.get('status', 'failed')}",
        (35, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    err = str(result.get("error") or "no trajectory recorded")
    cv2.putText(
        frame,
        err[:120],
        (35, 190),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (180, 180, 255),
        1,
        cv2.LINE_AA,
    )
    for _ in range(FPS * 2):
        writer.write(frame)
    writer.release()
    return {
        "attempt": job["attempt"],
        "episode_id": result.get("episode_id", job["row"]["episode_id"]),
        "clean_success": False,
        "video_path": str(output.relative_to(ROOT)),
        "video_sha256": sha256_file(output),
    }


def _render_review_video(job: dict[str, Any]) -> dict[str, Any]:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("MLSPACES_ASSETS_DIR", str(ROOT / "assets"))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.pop("DISPLAY", None)

    result = json.loads(Path(job["result_path"]).read_text())
    if result.get("status") != "complete" or not Path(job["trajectory_path"]).is_file():
        return _render_placeholder(job, result)

    import pact_geom_distance as instrument
    from molmo_spaces.data_generation.pipeline import cleanup_episode_resources
    from run_pact_place_swept_volume_v7 import geom_groups
    from run_pact_place_v6c_replay_videos import (
        PANE_WH,
        THIRD_PERSON_FOV,
        WRIST_FOV,
        _resize_pane,
        apply_recorded_qpos,
        third_person_pose,
        wrist_camera_pose,
    )
    from run_pact_place_v8_baseline import _physical_geoms, _target_geoms
    from run_pact_place_v9_v0c_siting import (
        _body_aabb,
        _skin_detects,
        _wrist_in_fov,
    )

    row = job["row"]
    steps = list(json.loads(Path(job["trajectory_path"]).read_text())["steps"])
    output = Path(job["video_path"])
    output.parent.mkdir(parents=True, exist_ok=True)

    task = sampler = scratch = None
    writer = None
    try:
        scratch = Path(tempfile.mkdtemp(prefix="pact_place_v9_review_"))
        config = _make_config(
            scratch / "dummy.json",
            scene_xml=SCENE_XML,
            sampler_class=str(row.get("sampler_class") or V9_SAMPLER_CLASS),
        )
        sampler = config.task_sampler_config.task_sampler_class(config)
        sampler.seed_task_sampling(int(result["selected_seed"]["seed_u32"]))
        sampler.set_pact_manifest_row(row)
        task = sampler.sample_task(house_index=int(row["scene_template_house_index"]))
        env = task.env
        model, data = env.current_model, env.current_data

        target_gids = _physical_geoms(model, _target_geoms(model))
        groups = geom_groups(model)
        by_group = {
            "cup": target_gids,
            "fingers": _physical_geoms(
                model, list(groups["left_finger"]) + list(groups["right_finger"])
            ),
            "hand": _physical_geoms(model, list(groups["hand"])),
            "link7": _physical_geoms(model, list(groups["link7"])),
            "link6": _physical_geoms(model, list(groups["link6"])),
            "link5": _physical_geoms(model, list(groups["link5"])),
        }
        sensor_ids = [
            index
            for index in range(int(model.ncam))
            if "_sensor_" in (model.camera(index).name or "")
        ]
        wrist_id = int(model.camera("robot_0/gripper/wrist_camera").id)

        scene_params = result.get("scene_params") or {}
        panel_active = bool(scene_params.get("pact_v9_legacy_panel_active"))
        active_panel_name = (
            str(scene_params.get("protr_name") or f"pact_intrusion_{row['intrusion_side']}")
            if panel_active
            else None
        )

        vessel_slots = {
            str(item["palette_slot"])
            for item in row["pact_clutter_layout"]["objects"]
            if str(item.get("role")) in {"inbound_vessel", "outbound_vessel"}
        }
        # Tint only the movable vessels orange. Keep the active panel's original
        # matte material so review video appearance matches the environment.
        for gid in range(int(model.ngeom)):
            geom_name = model.geom(gid).name or ""
            if any(f"pact_clutter_{slot}" in geom_name for slot in vessel_slots):
                model.geom_rgba[gid] = VESSEL_REVIEW_RGBA

        # Review camera direction
        rev_delta = REVIEW_CAM_TARGET - REVIEW_CAM_POS
        rev_fwd = rev_delta / np.linalg.norm(rev_delta)
        rev_up = np.asarray([0.0, 0.0, 1.0], dtype=float)

        writer = cv2.VideoWriter(
            str(output),
            cv2.VideoWriter_fourcc(*"mp4v"),
            FPS,
            (PANE_WH[0] * 3, PANE_WH[1]),
        )

        n_steps = len(steps)
        frame_stride = max(1, int(job.get("frame_stride", 1)))
        render_indices = list(range(0, n_steps, frame_stride))
        if render_indices[-1] != n_steps - 1:
            render_indices.append(n_steps - 1)
        clutter_body_names = {
            str(obj["palette_slot"]): f"pact_clutter_{obj['palette_slot']}/{obj['uid']}"
            for obj in row["pact_clutter_layout"]["objects"]
        }
        active_hazard_body_names = [
            clutter_body_names[slot] for slot in sorted(vessel_slots)
        ]
        mounted_body_names = [
            str(item["body"])
            for item in list(scene_params.get("pact_v9_hazards") or [])
            if str(item.get("role")) in {"wall_fixture", "ceiling_fixture"}
            and item.get("body")
        ]
        active_hazard_body_names.extend(mounted_body_names)
        if active_panel_name:
            active_hazard_body_names.append(active_panel_name)
        initial_clutter_pos = {}
        for slot in vessel_slots:
            body_name = clutter_body_names[slot]
            bid = model.body(body_name).id
            initial_clutter_pos[slot] = data.xpos[bid].copy()
        for body_name in mounted_body_names:
            bid = model.body(body_name).id
            initial_clutter_pos[f"mounted:{body_name}"] = data.xpos[bid].copy()

        for step_idx in render_indices:
            step = steps[step_idx]
            apply_recorded_qpos(env, step["qpos"])

            # Render Wrist
            w_pos, w_fwd, w_up = wrist_camera_pose(env)
            wrist_rgb = _resize_pane(
                np.asarray(env._render_frame(w_pos, w_fwd, w_up, WRIST_FOV, segmentation=False))
            )

            # Render Third Person
            tp_pos, tp_fwd, tp_up = third_person_pose(env)
            third_rgb = _resize_pane(
                np.asarray(
                    env._render_frame(tp_pos, tp_fwd, tp_up, THIRD_PERSON_FOV, segmentation=False)
                )
            )

            # Render Review Camera
            review_rgb = _resize_pane(
                np.asarray(
                    env._render_frame(
                        REVIEW_CAM_POS, rev_fwd, rev_up, REVIEW_CAM_FOV, segmentation=False
                    )
                )
            )

            # Compute per-body clearance to the nearest active panel/blocker hazard.
            hazard_geoms = []
            for gid in range(int(model.ngeom)):
                geom_name = model.geom(gid).name or ""
                if any(f"pact_clutter_{slot}" in geom_name for slot in vessel_slots) or (
                    active_panel_name and active_panel_name in geom_name
                ) or any(body_name in geom_name for body_name in mounted_body_names):
                    hazard_geoms.append(gid)

            clearances = {}
            for group_name in BODY_GROUPS:
                val = instrument.true_distance(model, data, by_group[group_name], hazard_geoms)
                clearances[group_name] = float(val)

            min_link = min(clearances["link5"], clearances["link6"])

            # Sampled sensor-cone proxy and wrist FOV for either active hazard.
            skin_detect = False
            in_fov = False
            for body_name in active_hazard_body_names:
                lo, hi = _body_aabb(model, data, body_name)
                ctr = (lo + hi) / 2.0
                hlf = (hi - lo) / 2.0
                if _skin_detects(model, data, sensor_ids, ctr, hlf):
                    skin_detect = True
                if _wrist_in_fov(model, data, wrist_id, ctr, hlf):
                    in_fov = True

            # Drift
            max_drift = 0.0
            for slot, init_p in initial_clutter_pos.items():
                body_name = (
                    slot.removeprefix("mounted:")
                    if slot.startswith("mounted:")
                    else clutter_body_names[slot]
                )
                bid = model.body(body_name).id
                cur_p = data.xpos[bid]
                max_drift = max(max_drift, float(np.linalg.norm(cur_p - init_p)))

            terminal = step_idx == n_steps - 1
            frame = overlay(
                wrist_rgb,
                third_rgb,
                review_rgb,
                pane_wh=PANE_WH,
                attempt=job["attempt"],
                intrusion_side=row["intrusion_side"],
                step=step_idx,
                n_steps=n_steps,
                policy_phase=str(step.get("policy_phase") or "unknown"),
                active_maneuver=str(step.get("pact_active_maneuver") or ""),
                clearances=clearances,
                min_link_m=min_link,
                in_wrist_fov=in_fov,
                skin_detect=skin_detect,
                drift_m=max_drift,
                terminal=terminal,
                clean_success=bool(result.get("clean_success")),
                clip_label=str(job.get("clip_label") or ""),
            )
            writer.write(frame)

        # Hold final frame for 1.5s
        for _ in range(int(FPS * 1.5)):
            writer.write(frame)

        writer.release()
        return {
            "attempt": job["attempt"],
            "episode_id": result["episode_id"],
            "clean_success": bool(result.get("clean_success")),
            "source_steps": n_steps,
            "rendered_frames": len(render_indices) + int(FPS * 1.5),
            "frame_stride": frame_stride,
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-attempts", type=int, default=24)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--diagnostic-successes",
        type=int,
        default=0,
        help="capture this many non-gating clean-success clips even when raw admission failed",
    )
    parser.add_argument(
        "--proximity-validation",
        type=Path,
        default=DEFAULT_PROXIMITY_VALIDATION,
    )
    args = parser.parse_args()
    if args.diagnostic_successes < 0:
        raise ValueError("diagnostic-successes must be non-negative")
    diagnostic_mode = args.diagnostic_successes > 0

    proximity_path = args.proximity_validation.resolve()
    if not proximity_path.is_file():
        raise FileNotFoundError(
            f"V1b is blocked until V0c.3 causal proximity validation exists: {proximity_path}"
        )
    proximity_validation = json.loads(proximity_path.read_text())
    if proximity_validation.get("passed") is not True and not diagnostic_mode:
        raise RuntimeError("V1b is blocked because V0c.3 causal proximity did not pass")

    palette_doc = load_palette(PALETTE_PATH)
    palette = list(palette_doc["palette"])
    family_ids = list(LAYOUT_FAMILIES)
    output_root = (
        DEFAULT_DIAGNOSTIC_SUCCESS_ROOT
        if diagnostic_mode and args.output_root == DEFAULT_OUTPUT_ROOT
        else args.output_root
    ).resolve()
    video_root = output_root / "videos"

    output_root.mkdir(parents=True, exist_ok=True)
    video_root.mkdir(parents=True, exist_ok=True)

    successes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    clips: list[dict[str, Any]] = []

    print(
        "=== Starting non-gating success diagnostics ==="
        if diagnostic_mode
        else "=== Starting Stage V1b Human Design Review ===",
        flush=True,
    )
    for attempt in range(args.max_attempts):
        side = "left" if attempt % 2 == 0 else "right"
        family_id = family_ids[(attempt // 2) % len(family_ids)]
        layout = build_layout(
            palette_doc, family_id=family_id, intrusion_side=side
        )
        row = _make_review_row(attempt, side, palette, layout)
        print(
            f"Running attempt {attempt:02d} (side={side}, family={family_id})...",
            flush=True,
        )

        result = run_row(
            row,
            config_sha256=sha256_payload(row),
            output_root=str(output_root),
            scene_xml=str(SCENE_XML),
        )
        clean = bool(result.get("clean_success"))
        print(
            f"  Attempt {attempt:02d} outcome: {'CLEAN SUCCESS' if clean else 'FAILURE'} (status={result.get('status')})",
            flush=True,
        )

        stem = clip_stem(attempt, side, result)
        video_path = video_root / f"{stem}.mp4"
        row_dir = output_root / "expert_screen_rows" / f"{attempt:02d}_{row['episode_id'][:16]}"

        if diagnostic_mode and not clean:
            failures.append(
                {
                    "attempt": attempt,
                    "side": side,
                    "layout_family_id": family_id,
                    "episode_id": row["episode_id"],
                    "clean_success": False,
                    "result_path": str(row_dir / "result.json"),
                    "video_path": None,
                }
            )
            continue

        job = {
            "attempt": attempt,
            "row": row,
            "result_path": str(row_dir / "result.json"),
            "trajectory_path": str(row_dir / "trajectory.json"),
            "video_path": str(video_path),
        }
        clip_info = _render_review_video(job)
        clips.append(clip_info)

        record = {
            "attempt": attempt,
            "side": side,
            "layout_family_id": family_id,
            "episode_id": row["episode_id"],
            "clean_success": clean,
            "result_path": str(row_dir / "result.json"),
            "video_path": str(video_path.relative_to(ROOT)),
        }
        if clean:
            successes.append(record)
        else:
            failures.append(record)

        target_reached = (
            len(successes) >= args.diagnostic_successes
            if diagnostic_mode
            else len(successes) >= 3 and len(failures) >= 3
        )
        if target_reached:
            print(
                f"Target reached: {len(successes)} successes, {len(failures)} failures captured. Stopping.",
                flush=True,
            )
            break

    manifest = {
        "schema_version": (
            "pact_place_v9_3_success_diagnostic_v1"
            if diagnostic_mode
            else "pact_place_v9_v1b_review_v1"
        ),
        "role": (
            "success_episode_diagnostic_raw_admission_failed_not_a_gate"
            if diagnostic_mode
            else "human_design_review_not_a_gate"
        ),
        "authorizes_gate": False,
        "authorizes_collection": False,
        "clean_rate_is_not_an_estimate": True,
        "causal_proximity_validation_path": str(proximity_path.relative_to(ROOT)),
        "causal_proximity_validation_sha256": sha256_file(proximity_path),
        "causal_proximity_validation_passed": bool(
            proximity_validation.get("passed") is True
        ),
        "diagnostic_success_target": (
            args.diagnostic_successes if diagnostic_mode else None
        ),
        "review_overlay_sensor_flag": "sampled_sensor_cone_proxy_not_raw_depth",
        "completed_at": utc_now(),
        "n_attempts": len(successes) + len(failures),
        "n_video_clips": len(clips),
        "n_successes_captured": len(successes),
        "n_failures_captured": len(failures),
        "successes": successes,
        "failures": failures,
        "video_clips": clips,
        "review_gate_episode_overlap": [],
        "layout_family_ids": family_ids,
    }
    manifest["manifest_sha256"] = sha256_payload(manifest)
    manifest_path = output_root / "review_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Emitted review manifest: {manifest_path}", flush=True)
    print(
        json.dumps(
            {
                "role": manifest["role"],
                "authorizes_gate": manifest["authorizes_gate"],
                "n_attempts": manifest["n_attempts"],
                "n_successes": manifest["n_successes_captured"],
                "n_failures": manifest["n_failures_captured"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
