#!/usr/bin/env python3
"""V8C C0b: how high inside the enclosure does the wrist camera ever see?

The plan retains v8b's ``non-zero visibility_at_min >= 1/3`` gate on the ground
that "vision must stay useful but imperfect".  C0 found that every one of the 567
overhead-bar candidates scores 0/24 on it.  This measures the reason as a number
rather than an argument.

For each replayed v6c frame the wrist camera's view cone is intersected with the
enclosure interior on the same 2 cm lattice the v7 swept-volume analysis used.
Occlusion is ignored on purpose: an unoccluded cone is an upper bound on what the
camera can see, so a ceiling below the band bottom proves nothing in the band is
ever visible, whatever the occluders do.  A second, occlusion-aware pass
(``mj_ray``) reports the ceiling that is actually reached.
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

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MOLMO = ROOT / "submodules" / "molmospaces"
for _path in (ROOT / "scripts", MOLMO):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

OUTPUT_DIR = ROOT / "diagnostics_output/pact_place_corridor_v8c_c0"
INTERIOR_X = (0.58, 0.86)
INTERIOR_Y = (-0.445, 0.445)
INTERIOR_Z = (0.72, 1.42)
VOXEL_M = 0.02
CUP_SWEPT_Z_MAX = 1.0114741856483134
BAND_Z_BOTTOM = 1.05


def lattice() -> np.ndarray:
    axes = [
        np.arange(lo + VOXEL_M / 2, hi, VOXEL_M)
        for lo, hi in (INTERIOR_X, INTERIOR_Y, INTERIOR_Z)
    ]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    return grid.reshape(-1, 3)


def measure_episode(job: dict[str, Any]) -> dict[str, Any]:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("MLSPACES_ASSETS_DIR", str(ROOT / "assets"))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.pop("DISPLAY", None)

    import mujoco
    from molmo_spaces.data_generation.pipeline import cleanup_episode_resources
    from run_pact_place_v6c_replay_videos import _prepare_task, apply_recorded_qpos, row_directory
    from run_pact_place_v7_replay_videos import WRIST_CAMERA_MJCF

    row = job["row"]
    result = json.loads(Path(job["result_path"]).read_text())
    steps = list(json.loads(Path(job["trajectory_path"]).read_text())["steps"])
    points = lattice()
    far_m = float(job["clip_far_m"])

    task = sampler = scratch = None
    try:
        task, sampler, scratch = _prepare_task(row, result["selected_seed"])
        model, data = task.env.current_model, task.env.current_data
        wrist_id = int(model.camera(WRIST_CAMERA_MJCF).id)
        min_cosine = float(np.cos(np.deg2rad(float(model.cam_fovy[wrist_id])) / 2.0))
        in_cone_ever = np.zeros(len(points), dtype=bool)
        cam_z = []
        cam_pitch_deg = []
        for step in steps:
            apply_recorded_qpos(task.env, step["qpos"])
            origin = np.asarray(data.cam_xpos[wrist_id], dtype=np.float64)
            forward = -np.asarray(data.cam_xmat[wrist_id], dtype=np.float64).reshape(3, 3)[:, 2]
            cam_z.append(float(origin[2]))
            cam_pitch_deg.append(float(np.degrees(np.arcsin(np.clip(forward[2], -1.0, 1.0)))))
            delta = points - origin
            span = np.linalg.norm(delta, axis=1)
            good = span > 1e-9
            cosine = np.zeros(len(points))
            cosine[good] = (delta[good] @ forward) / span[good]
            in_cone_ever |= (cosine >= min_cosine) & (span <= far_m)

        cone_z = points[in_cone_ever][:, 2] if in_cone_ever.any() else np.array([])
        cone_ceiling = float(cone_z.max()) if cone_z.size else None

        # Occlusion-aware ceiling: only the points the cone ever reached need rays.
        visible_ever = np.zeros(len(points), dtype=bool)
        candidate_idx = np.flatnonzero(in_cone_ever)
        if candidate_idx.size:
            # Rays are expensive; walk the frames again and test only the
            # still-unconfirmed cone points, highest z first.
            order = candidate_idx[np.argsort(-points[candidate_idx][:, 2])]
            geomid = np.zeros(1, dtype=np.int32)
            for step in steps[:: int(job["ray_stride"])]:
                apply_recorded_qpos(task.env, step["qpos"])
                origin = np.asarray(data.cam_xpos[wrist_id], dtype=np.float64)
                forward = -np.asarray(data.cam_xmat[wrist_id], dtype=np.float64).reshape(3, 3)[:, 2]
                pending = order[~visible_ever[order]]
                if pending.size == 0:
                    break
                delta = points[pending] - origin
                span = np.linalg.norm(delta, axis=1)
                cosine = np.where(span > 1e-9, (delta @ forward) / np.maximum(span, 1e-12), -1.0)
                live = pending[(cosine >= min_cosine) & (span <= far_m)]
                for index in live:
                    direction = points[index] - origin
                    distance = float(np.linalg.norm(direction))
                    hit = mujoco.mj_ray(
                        model, data, origin, (direction / distance).astype(np.float64),
                        None, 1, -1, geomid,
                    )
                    # Nothing between the camera and the point: the point is seen.
                    if hit < 0 or float(hit) >= distance - VOXEL_M:
                        visible_ever[index] = True

        seen_z = points[visible_ever][:, 2] if visible_ever.any() else np.array([])
        return {
            "role_index": int(row["role_index"]),
            "episode_id": row["episode_id"],
            "n_steps": len(steps),
            "wrist_camera_z_m": {
                "min": float(np.min(cam_z)),
                "mean": float(np.mean(cam_z)),
                "max": float(np.max(cam_z)),
            },
            "wrist_camera_pitch_deg": {
                "min": float(np.min(cam_pitch_deg)),
                "mean": float(np.mean(cam_pitch_deg)),
                "max": float(np.max(cam_pitch_deg)),
            },
            "cone_ceiling_z_m": cone_ceiling,
            "occluded_aware_ceiling_z_m": (float(seen_z.max()) if seen_z.size else None),
            "n_lattice_points": int(len(points)),
            "n_points_in_cone_ever": int(in_cone_ever.sum()),
            "n_points_visible_ever": int(visible_ever.sum()),
            "n_cone_points_above_band_bottom": int(
                (points[in_cone_ever][:, 2] >= BAND_Z_BOTTOM).sum() if in_cone_ever.any() else 0
            ),
        }
    finally:
        cleanup_episode_resources(
            task=task, policy=None, task_sampler=sampler,
            preloaded_policy=None, close_task_sampler=sampler is not None,
        )
        if scratch is not None:
            shutil.rmtree(scratch, ignore_errors=True)


def main() -> int:
    from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
    from pact_place_corridor_contract import sha256_payload
    from run_pact_place_expert_screen import write_json_atomic
    from run_pact_place_v6c_replay_videos import CONFIG_PATH, row_directory

    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--ray-stride", type=int, default=5)
    args = parser.parse_args()

    near_m, far_m = tuple(MlSpacesExpConfig.model_fields["viz_depth_range"].default)
    config = json.loads(CONFIG_PATH.read_text())
    jobs = []
    for row in config["expert_screen_rows"]:
        directory = row_directory(int(row["role_index"]), row["episode_id"])
        jobs.append(
            {
                "row": row,
                "result_path": str(directory / "result.json"),
                "trajectory_path": str(directory / "trajectory.json"),
                "clip_far_m": float(far_m),
                "ray_stride": int(args.ray_stride),
            }
        )

    context = multiprocessing.get_context("spawn")
    episodes = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max(1, args.workers), mp_context=context
    ) as pool:
        for episode in pool.map(measure_episode, jobs):
            episodes.append(episode)
            print(
                f"row {episode['role_index']:02d} cone ceiling "
                f"{episode['cone_ceiling_z_m']} occl {episode['occluded_aware_ceiling_z_m']}",
                flush=True,
            )
    episodes.sort(key=lambda item: item["role_index"])

    cone = [e["cone_ceiling_z_m"] for e in episodes if e["cone_ceiling_z_m"] is not None]
    occl = [e["occluded_aware_ceiling_z_m"] for e in episodes
            if e["occluded_aware_ceiling_z_m"] is not None]
    document = {
        "schema_version": "pact_place_v8c_c0_visibility_ceiling_v1",
        "role": "c0_visibility_ceiling_measurement_not_a_gate",
        "authorizes_gate": False,
        "replay_only": True,
        "physics_stepped": False,
        "camera": "robot_0/gripper/wrist_camera (the only RGB camera in the "
                  "collection observation; run_pact_place_recovery_datagen.py:386)",
        "lattice_m": VOXEL_M,
        "enclosure_interior_m": {"x_m": list(INTERIOR_X), "y_m": list(INTERIOR_Y),
                                 "z_m": list(INTERIOR_Z)},
        "clip_far_m": float(far_m),
        "clip_near_m": float(near_m),
        "cup_swept_z_max_m": CUP_SWEPT_Z_MAX,
        "band_z_bottom_m": BAND_Z_BOTTOM,
        "per_episode": episodes,
        "aggregate": {
            "n_episodes": len(episodes),
            "max_cone_ceiling_z_m": (max(cone) if cone else None),
            "max_occlusion_aware_ceiling_z_m": (max(occl) if occl else None),
            "episodes_with_any_cone_point_at_or_above_band_bottom": sum(
                e["n_cone_points_above_band_bottom"] > 0 for e in episodes
            ),
            "total_cone_points_at_or_above_band_bottom": sum(
                e["n_cone_points_above_band_bottom"] for e in episodes
            ),
        },
    }
    document["analysis_sha256"] = sha256_payload(document)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json_atomic(OUTPUT_DIR / "c0_visibility_ceiling.json", document)
    print(json.dumps(document["aggregate"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
