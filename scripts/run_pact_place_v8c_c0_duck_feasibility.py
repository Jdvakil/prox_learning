#!/usr/bin/env python3
"""V8C C0c: how far above the TCP is the body that would hit an overhead bar?

C1 would clear the bar by ducking, and both existing maneuvers are written in TCP
space: the inbound branch shifts ``z_travel`` by ``dz`` and ``_bow_segment``
displaces TCP waypoints. That only works if lowering the TCP lowers the colliding
body by a comparable amount. The lateral panel case is safe because the panel sits
at the wrist's own height; an overhead bar is hit by link5/link6, which ride well
above the TCP.

This measures, over the frozen v6c trajectories, the vertical offset between the
TCP and the top of the link5/link6 collision geoms while the arm is inside the
corridor, and how far the TCP would have to drop for those links to clear a bar
whose bottom face is at ``BAND_Z_BOTTOM``. Replay only.
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
BAND_Z_BOTTOM = 1.05
SHELF_TOP_Z = 0.72
CORRIDOR_X = (0.58, 0.86)


def measure_episode(job: dict[str, Any]) -> dict[str, Any]:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("MLSPACES_ASSETS_DIR", str(ROOT / "assets"))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.pop("DISPLAY", None)

    from molmo_spaces.data_generation.pipeline import cleanup_episode_resources
    from run_pact_place_v6c_replay_videos import _prepare_task, apply_recorded_qpos
    from run_pact_place_swept_volume_v7 import geom_groups, world_aabb_for_geom
    from run_pact_place_v8_baseline import _physical_geoms

    row = job["row"]
    result = json.loads(Path(job["result_path"]).read_text())
    steps = list(json.loads(Path(job["trajectory_path"]).read_text())["steps"])

    task = sampler = scratch = None
    try:
        task, sampler, scratch = _prepare_task(row, result["selected_seed"])
        model, data = task.env.current_model, task.env.current_data
        groups = geom_groups(model)
        elbow = _physical_geoms(
            model, list(groups["link5"]) + list(groups["link6"])
        )
        records = []
        for step in steps:
            apply_recorded_qpos(task.env, step["qpos"])
            tcp = np.asarray(step["tcp_position_m"], dtype=float)
            top = -np.inf
            top_x = None
            for gid in elbow:
                lo, hi = world_aabb_for_geom(model, data, int(gid))
                # Only the part of the link that is inside the corridor can meet
                # a bar that lives inside the corridor.
                if hi[0] < CORRIDOR_X[0] or lo[0] > CORRIDOR_X[1]:
                    continue
                if hi[2] > top:
                    top = float(hi[2])
                    top_x = float(np.clip(0.5 * (lo[0] + hi[0]), *CORRIDOR_X))
            if not np.isfinite(top):
                continue
            records.append(
                {
                    "phase": str(step.get("policy_phase") or "unknown"),
                    "tcp_z_m": float(tcp[2]),
                    "tcp_x_m": float(tcp[0]),
                    "elbow_top_z_m": top,
                    "elbow_x_m": top_x,
                    "offset_m": top - float(tcp[2]),
                    "required_drop_m": max(0.0, top - BAND_Z_BOTTOM),
                }
            )
        offending = [r for r in records if r["required_drop_m"] > 0.0]
        offsets = np.asarray([r["offset_m"] for r in offending], dtype=float)
        drops = np.asarray([r["required_drop_m"] for r in offending], dtype=float)
        tcp_z = np.asarray([r["tcp_z_m"] for r in offending], dtype=float)
        return {
            "role_index": int(row["role_index"]),
            "episode_id": row["episode_id"],
            "n_steps": len(steps),
            "n_frames_link5_link6_in_corridor": len(records),
            "n_frames_above_band_bottom": len(offending),
            "elbow_above_tcp_offset_m": {
                "min": (float(offsets.min()) if offsets.size else None),
                "median": (float(np.median(offsets)) if offsets.size else None),
                "max": (float(offsets.max()) if offsets.size else None),
            },
            "required_tcp_drop_m": {
                "min": (float(drops.min()) if drops.size else None),
                "median": (float(np.median(drops)) if drops.size else None),
                "max": (float(drops.max()) if drops.size else None),
            },
            "tcp_z_after_max_drop_m": (
                float((tcp_z - drops).min()) if drops.size else None
            ),
            "frames_where_drop_puts_tcp_below_shelf": int(
                np.sum((tcp_z - drops) < SHELF_TOP_Z) if drops.size else 0
            ),
            "phases_above_band_bottom": sorted({r["phase"] for r in offending}),
        }
    finally:
        cleanup_episode_resources(
            task=task, policy=None, task_sampler=sampler,
            preloaded_policy=None, close_task_sampler=sampler is not None,
        )
        if scratch is not None:
            shutil.rmtree(scratch, ignore_errors=True)


def main() -> int:
    from pact_place_corridor_contract import sha256_payload
    from run_pact_place_expert_screen import write_json_atomic
    from run_pact_place_v6c_replay_videos import CONFIG_PATH, row_directory

    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    config = json.loads(CONFIG_PATH.read_text())
    jobs = []
    for row in config["expert_screen_rows"]:
        directory = row_directory(int(row["role_index"]), row["episode_id"])
        jobs.append(
            {
                "row": row,
                "result_path": str(directory / "result.json"),
                "trajectory_path": str(directory / "trajectory.json"),
            }
        )

    context = multiprocessing.get_context("spawn")
    episodes = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max(1, args.workers), mp_context=context
    ) as pool:
        for episode in pool.map(measure_episode, jobs):
            episodes.append(episode)
            print(f"row {episode['role_index']:02d} "
                  f"drop {episode['required_tcp_drop_m']}", flush=True)
    episodes.sort(key=lambda item: item["role_index"])

    drops = [e["required_tcp_drop_m"]["max"] for e in episodes
             if e["required_tcp_drop_m"]["max"] is not None]
    offsets = [e["elbow_above_tcp_offset_m"]["median"] for e in episodes
               if e["elbow_above_tcp_offset_m"]["median"] is not None]
    document = {
        "schema_version": "pact_place_v8c_c0_duck_feasibility_v1",
        "role": "c0_duck_feasibility_measurement_not_a_gate",
        "authorizes_gate": False,
        "replay_only": True,
        "physics_stepped": False,
        "band_z_bottom_m": BAND_Z_BOTTOM,
        "shelf_top_z_m": SHELF_TOP_Z,
        "corridor_x_m": list(CORRIDOR_X),
        "per_episode": episodes,
        "aggregate": {
            "n_episodes": len(episodes),
            "max_required_tcp_drop_m": (max(drops) if drops else None),
            "median_required_tcp_drop_m": (float(np.median(drops)) if drops else None),
            "median_elbow_above_tcp_offset_m": (
                float(np.median(offsets)) if offsets else None
            ),
            "episodes_where_drop_puts_tcp_below_shelf": sum(
                e["frames_where_drop_puts_tcp_below_shelf"] > 0 for e in episodes
            ),
            "episodes_with_any_frame_above_band_bottom": sum(
                e["n_frames_above_band_bottom"] > 0 for e in episodes
            ),
        },
    }
    document["analysis_sha256"] = sha256_payload(document)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json_atomic(OUTPUT_DIR / "c0_duck_feasibility.json", document)
    print(json.dumps(document["aggregate"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
