#!/usr/bin/env python3
"""V8C C0 review clips: show the chosen overhead bar against the frozen v6c replay.

This renders three v6c episodes with C0's chosen candidate bar posed in the scene,
so the three C0 findings can be checked by eye rather than from a table:

  1. the bar sits in the cup-free band and the cup stays clear of it;
  2. the wrist camera never sees it (the in-FOV flag never turns yes);
  3. link5/link6 pass straight through it, and the TCP drop that would be needed
     to duck it is larger than the height the TCP has above the shelf.

These are REPLAYS with the bar composited in. The recorded expert never saw the
bar and never avoids it, and no physics is stepped -- ``mj_forward`` only. The
clips are a design aid, not a rollout and not a gate.

The v6c renderer is imported, not edited. Its faithfulness spot-checks run
unchanged, so a clip that would not be the episode fails instead of rendering.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MOLMO = ROOT / "submodules" / "molmospaces"
for _path in (ROOT / "scripts", MOLMO):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

C0_PATH = ROOT / "diagnostics_output/pact_place_corridor_v8c_c0/c0_siting.json"
OUTPUT_DIR = ROOT / "diagnostics_output/pact_place_corridor_v8c_c0_review"
REVIEW_ROWS = (0, 4, 17)
# Third pane: looks in through the aperture, above the cup and below the ceiling,
# so the bar, the forearm and the carried cup are all in one frame.
REVIEW_CAM_POS = np.asarray([0.05, -0.60, 1.34], dtype=float)
REVIEW_CAM_TARGET = np.asarray([0.74, 0.06, 1.12], dtype=float)
REVIEW_CAM_FOV = 52.0
# Render-only tint so the bar is not another white box among white boxes.
# Geometry, contype and conaffinity are untouched; only geom_rgba is written.
BAR_REVIEW_RGBA = (0.95, 0.35, 0.05, 1.0)
SHELF_TOP_Z = 0.72
BAND_Z_BOTTOM = 1.05


def overlay(
    wrist_rgb: np.ndarray,
    third_rgb: np.ndarray,
    review_rgb: np.ndarray,
    *,
    pane_wh: tuple[int, int],
    role_index: int,
    intrusion_side: str,
    step: int,
    n_steps: int,
    policy_phase: str,
    clearances: dict[str, float],
    min_link_m: float,
    in_wrist_fov: bool,
    required_drop_m: float,
    tcp_z_m: float,
    bar_contact_frames: int,
    terminal: bool,
) -> np.ndarray:
    width, height = pane_wh
    composite = np.concatenate([wrist_rgb, third_rgb, review_rgb], axis=1)
    frame = cv2.cvtColor(composite, cv2.COLOR_RGB2BGR)
    shade = frame.copy()
    cv2.rectangle(shade, (0, 0), (frame.shape[1], 104), (0, 0, 0), thickness=-1)
    cv2.rectangle(shade, (0, height - 46), (frame.shape[1], height), (0, 0, 0), thickness=-1)
    frame = cv2.addWeighted(shade, 0.68, frame, 0.32, 0.0)

    hit = min_link_m < 0.0
    put = lambda text, xy, scale, color, thick: cv2.putText(
        frame, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA
    )
    put(
        f"row {role_index:02d}  panel {intrusion_side}  {policy_phase}  "
        f"step {step}/{n_steps - 1}" + ("  TERMINAL" if terminal else ""),
        (12, 24), 0.52, (25, 25, 245) if terminal else (255, 255, 255), 2,
    )
    put(
        "  ".join(
            f"{key} {clearances[key]:+.3f}"
            for key in ("cup", "fingers", "hand", "link7", "link6", "link5")
        ),
        (12, 50), 0.44, (235, 235, 235), 1,
    )
    put(
        f"min link {min_link_m:+.3f} m  "
        + ("BAR STRUCK (expert never saw it)" if hit else "clear"),
        (12, 74), 0.48, (60, 60, 255) if hit else (120, 235, 120), 2,
    )
    put(
        f"bar in wrist FOV: {'YES' if in_wrist_fov else 'NO'}    "
        f"tcp z {tcp_z_m:.3f}  duck needed {required_drop_m:.3f} m  "
        f"-> tcp would land at {tcp_z_m - required_drop_m:.3f} m (shelf {SHELF_TOP_Z:.2f})",
        (12, 98), 0.42,
        (120, 235, 120) if in_wrist_fov else (90, 200, 255), 1,
    )
    put("wrist  (the policy's only RGB camera)", (12, height - 26), 0.42, (245, 245, 245), 1)
    put("third-person", (width + 12, height - 26), 0.42, (245, 245, 245), 1)
    put("corridor review  -  bar tinted orange", (2 * width + 12, height - 26),
        0.42, (245, 245, 245), 1)
    put(
        "REPLAY + CANDIDATE BAR COMPOSITED IN  -  mj_forward only, no physics, "
        "expert never saw or avoided the bar; tint is render-only  -  "
        f"bar-contact frames {bar_contact_frames}",
        (12, height - 8), 0.40, (200, 200, 200), 1,
    )
    return frame


def render_row(job: dict[str, Any]) -> dict[str, Any]:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("MLSPACES_ASSETS_DIR", str(ROOT / "assets"))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.pop("DISPLAY", None)

    import mujoco
    from molmo_spaces.data_generation.pipeline import cleanup_episode_resources
    from run_pact_place_v6c_replay_videos import (
        FPS,
        HOLD_FINAL_S,
        PANE_WH,
        THIRD_PERSON_FOV,
        WRIST_CAMERA_MJCF,
        WRIST_FOV,
        _prepare_task,
        _resize_pane,
        iter_replay_states,
        spot_check_indices,
        third_person_pose,
        wrist_camera_pose,
    )
    from run_pact_place_swept_volume_v7 import geom_groups, world_aabb_for_geom
    from run_pact_place_v8_baseline import _physical_geoms, _target_geoms
    from run_pact_place_v8c_c0_siting import BODY_GROUPS, true_distance

    role_index = int(job["role_index"])
    result = json.loads(Path(job["result_path"]).read_text())
    steps = list(json.loads(Path(job["trajectory_path"]).read_text())["steps"])
    center = np.asarray(job["bar_center_m"], dtype=float)
    output = Path(job["output_path"])
    output.parent.mkdir(parents=True, exist_ok=True)

    task = sampler = scratch = None
    writer = None
    bar_contact_frames = 0
    video_frames = 0
    residuals: list[dict[str, Any]] = []
    in_fov_frames = 0
    struck_frames = 0
    worst = {"min_link_m": float("inf"), "cup_m": float("inf"), "required_drop_m": 0.0}
    try:
        task, sampler, scratch = _prepare_task(job["row"], result["selected_seed"])
        env = task.env
        model, data = env.current_model, env.current_data

        spare = (
            "pact_intrusion_right"
            if str(result["intrusion_side"]) == "left"
            else "pact_intrusion_left"
        )
        spare_body = model.body(spare)
        spare_mocap = int(model.body_mocapid[spare_body.id])
        spare_gid = next(
            g for g in range(int(model.ngeom)) if int(model.geom_bodyid[g]) == spare_body.id
        )
        bar_half = np.asarray(model.geom_size[spare_gid], dtype=float)
        if not np.allclose(bar_half, job["bar_half_m"]):
            raise RuntimeError(
                f"{spare} half-extents {bar_half.tolist()} != C0 candidate "
                f"{job['bar_half_m']}; the clip would not be the measured bar"
            )
        # Pose the bar once. mj_forward reads data.mocap_pos and never rewrites it,
        # and apply_recorded_qpos touches only data.qpos, so this holds all episode.
        data.mocap_pos[spare_mocap] = center
        data.mocap_quat[spare_mocap] = [1.0, 0.0, 0.0, 0.0]
        model.geom_rgba[spare_gid] = BAR_REVIEW_RGBA

        groups = geom_groups(model)
        by_group = {
            "cup": _physical_geoms(model, _target_geoms(model)),
            "fingers": _physical_geoms(
                model, list(groups["left_finger"]) + list(groups["right_finger"])
            ),
            "hand": _physical_geoms(model, list(groups["hand"])),
            "link7": _physical_geoms(model, list(groups["link7"])),
            "link6": _physical_geoms(model, list(groups["link6"])),
            "link5": _physical_geoms(model, list(groups["link5"])),
        }
        elbow = by_group["link5"] + by_group["link6"]
        wrist_id = int(model.camera(WRIST_CAMERA_MJCF).id)
        min_cosine = float(np.cos(np.deg2rad(float(model.cam_fovy[wrist_id])) / 2.0))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(output), fourcc, FPS, (PANE_WH[0] * 3, PANE_WH[1])
        )
        review_fwd = REVIEW_CAM_TARGET - REVIEW_CAM_POS
        review_fwd = review_fwd / np.linalg.norm(review_fwd)
        review_right = np.cross(review_fwd, np.asarray([0.0, 0.0, 1.0]))
        review_right = review_right / np.linalg.norm(review_right)
        review_up = np.cross(review_right, review_fwd)
        review_up = review_up / np.linalg.norm(review_up)
        if not writer.isOpened():
            raise RuntimeError(f"cannot open writer for {output}")
        hold_frames = max(1, int(round(HOLD_FINAL_S * FPS)))
        spots = set(spot_check_indices(len(steps)))

        for index, step, live_object, live_tcp, residuals in iter_replay_states(
            env, task, steps, spots
        ):
            clearances = {
                key: true_distance(model, data, gids, spare_gid)
                for key, gids in by_group.items()
            }
            min_link = min(clearances[key] for key in ("link5", "link6", "link7"))
            if min_link < 0.0:
                struck_frames += 1

            origin = np.asarray(data.cam_xpos[wrist_id], dtype=np.float64)
            forward = -np.asarray(data.cam_xmat[wrist_id], dtype=np.float64).reshape(3, 3)[:, 2]
            delta = center - origin
            span = float(np.linalg.norm(delta))
            in_fov = bool(span > 1e-9 and float(np.dot(delta / span, forward)) >= min_cosine)
            in_fov_frames += int(in_fov)

            top = -np.inf
            for gid in elbow:
                _, hi = world_aabb_for_geom(model, data, int(gid))
                top = max(top, float(hi[2]))
            required_drop = max(0.0, top - BAND_Z_BOTTOM) if np.isfinite(top) else 0.0

            worst["min_link_m"] = min(worst["min_link_m"], min_link)
            worst["cup_m"] = min(worst["cup_m"], clearances["cup"])
            worst["required_drop_m"] = max(worst["required_drop_m"], required_drop)

            for contact_index in range(int(data.ncon)):
                contact = data.contact[contact_index]
                if float(contact.dist) > 0.0:
                    continue
                if spare_gid in (int(contact.geom1), int(contact.geom2)):
                    bar_contact_frames += 1
                    break

            position, fwd, up = third_person_pose(env)
            third = _resize_pane(
                np.asarray(env._render_frame(position, fwd, up, THIRD_PERSON_FOV,
                                             segmentation=False))
            )
            wrist_pos, wrist_fwd, wrist_up = wrist_camera_pose(env)
            wrist = _resize_pane(
                np.asarray(env._render_frame(wrist_pos, wrist_fwd, wrist_up, WRIST_FOV,
                                             segmentation=False))
            )
            review = _resize_pane(
                np.asarray(env._render_frame(REVIEW_CAM_POS, review_fwd, review_up,
                                             REVIEW_CAM_FOV, segmentation=False))
            )
            terminal = index == len(steps) - 1
            annotated = overlay(
                wrist,
                third,
                review,
                pane_wh=PANE_WH,
                role_index=role_index,
                intrusion_side=str(result["intrusion_side"]),
                step=int(step["step"]),
                n_steps=len(steps),
                policy_phase=str(step["policy_phase"]),
                clearances=clearances,
                min_link_m=min_link,
                in_wrist_fov=in_fov,
                required_drop_m=required_drop,
                tcp_z_m=float(live_tcp[2]),
                bar_contact_frames=bar_contact_frames,
                terminal=terminal,
            )
            for _ in range(hold_frames if terminal else 1):
                writer.write(annotated)
                video_frames += 1
    except Exception:
        if writer is not None:
            writer.release()
            writer = None
        if output.exists():
            output.unlink()
        raise
    finally:
        if writer is not None:
            writer.release()
        cleanup_episode_resources(
            task=task, policy=None, task_sampler=sampler,
            preloaded_policy=None, close_task_sampler=sampler is not None,
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
        "intrusion_side": result["intrusion_side"],
        "clean_success": bool(result["clean_success"]),
        "clip": output.name,
        "n_steps": len(steps),
        "video_frames": video_frames,
        "fps": FPS,
        "frames_bar_in_wrist_fov": in_fov_frames,
        "frames_links_inside_bar": struck_frames,
        "frames_with_bar_contact": bar_contact_frames,
        "min_link_clearance_m": worst["min_link_m"],
        "min_cup_clearance_m": worst["cup_m"],
        "max_required_tcp_drop_m": worst["required_drop_m"],
        "faithfulness": {
            "max_object_residual_m": max(
                item["object_position_residual_m"] for item in residuals
            ),
            "max_tcp_residual_m": max(
                item["tcp_position_residual_m"] for item in residuals
            ),
            "spot_check_indices": sorted(spots),
        },
    }


def main() -> int:
    from pact_place_corridor_contract import sha256_payload
    from run_pact_place_expert_screen import write_json_atomic
    from run_pact_place_v6c_replay_videos import CONFIG_PATH, row_directory

    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--rows", type=int, nargs="*", default=list(REVIEW_ROWS))
    args = parser.parse_args()

    c0 = json.loads(C0_PATH.read_text())
    chosen = c0["chosen_candidate"]
    if chosen is None:
        raise SystemExit("C0 admitted no candidate; there is nothing to render")

    config = json.loads(CONFIG_PATH.read_text())
    rows = {int(row["role_index"]): row for row in config["expert_screen_rows"]}
    jobs = []
    for role_index in args.rows:
        row = rows[int(role_index)]
        directory = row_directory(int(role_index), row["episode_id"])
        result = json.loads((directory / "result.json").read_text())
        side = str(result["intrusion_side"])
        jobs.append(
            {
                "role_index": int(role_index),
                "row": row,
                "result_path": str(directory / "result.json"),
                "trajectory_path": str(directory / "trajectory.json"),
                "bar_center_m": list(map(float, chosen["center_m"])),
                "bar_half_m": list(map(float, chosen["half_m"])),
                "output_path": str(
                    OUTPUT_DIR / f"row{int(role_index):02d}_panel_{side}_with_c0_bar.mp4"
                ),
            }
        )

    context = multiprocessing.get_context("spawn")
    clips = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max(1, args.workers), mp_context=context
    ) as pool:
        for clip in pool.map(render_row, jobs):
            clips.append(clip)
            print(
                f"row {clip['role_index']:02d} -> {clip['clip']}  "
                f"in-FOV {clip['frames_bar_in_wrist_fov']}/{clip['n_steps']}  "
                f"links-inside-bar {clip['frames_links_inside_bar']}/{clip['n_steps']}",
                flush=True,
            )
    clips.sort(key=lambda item: item["role_index"])

    document = {
        "schema_version": "pact_place_v8c_c0_review_videos_v1",
        "role": "human_design_review_not_a_gate",
        "authorizes_gate": False,
        "authorizes_collection": False,
        "replay_only": True,
        "physics_stepped": False,
        "note": (
            "v6c replays with C0's chosen candidate bar composited in. The recorded "
            "expert never saw the bar and never avoids it; contacts shown are what "
            "the frozen trajectory would produce against this bar, not a rollout."
        ),
        "bar_tint_is_render_only": True,
        "bar": {"center_m": chosen["center_m"], "half_m": chosen["half_m"],
                "candidate_id": chosen["candidate_id"],
                "z_bottom_m": chosen["z_bottom_m"], "z_top_m": chosen["z_top_m"]},
        "c0_analysis_sha256": c0["analysis_sha256"],
        "clips": clips,
        "aggregate": {
            "frames_bar_in_wrist_fov": sum(c["frames_bar_in_wrist_fov"] for c in clips),
            "frames_total": sum(c["n_steps"] for c in clips),
            "frames_links_inside_bar": sum(c["frames_links_inside_bar"] for c in clips),
            "min_cup_clearance_m": min(c["min_cup_clearance_m"] for c in clips),
            "max_required_tcp_drop_m": max(c["max_required_tcp_drop_m"] for c in clips),
        },
    }
    document["analysis_sha256"] = sha256_payload(document)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json_atomic(OUTPUT_DIR / "review.json", document)
    print(json.dumps(document["aggregate"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
