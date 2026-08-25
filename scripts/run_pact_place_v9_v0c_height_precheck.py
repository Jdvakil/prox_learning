#!/usr/bin/env python3
"""V0c pre-check: does raising the clutter actually bring it into link5/link6 range?

The v9 plan's premise 3 is a z-band argument: v6c clutter tops out at 0.820 and
link5's swept floor is 0.819, so a taller vessel should sit inside link5's band.
That argument is about one axis. V0a's hardened re-measurement says link5 never
comes within 0.258 m of the v6c clutter, which a 1 mm vertical miss cannot
explain -- so the binding gap may be horizontal, in which case height buys
nothing and the palette work is on the wrong axis.

This measures it directly. The four v6c clutter boxes are grown upward in place
(base pinned to the shelf at SHELF_TOP_Z, half-z and centre raised) to each
candidate vessel height, and for every height the exact hardened clearance and
the spec-true skin detectability are recomputed over the frozen v6c replays.

Also decomposes the closest approach into axis components using the witness
points MuJoCo reports, so "how much of the gap is vertical" is a number.

Replay only. Nothing is stepped, no scene file or v6c artifact is touched; the
clutter geoms are resized in the in-memory model exactly as C0 posed its bar.
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

OUTPUT_DIR = ROOT / "diagnostics_output/pact_place_v9_v0a"
SHELF_TOP_Z = 0.72
HEIGHTS_M = (0.10, 0.15, 0.20, 0.25)      # 0.10 is the v6c baseline
BODY_KEYS = ("link5", "link6", "link7", "hand", "fingers", "cup")
NEAR_M = 0.10
VERY_NEAR_M = 0.05
# enclosure_reach.py:50-52
SENSOR_RANGE = 1.0
SENSOR_RANGE_DERATE = 0.85
SENSOR_HALF_FOV_COS = float(np.cos(np.deg2rad(22.5)))
# enclosure_reach.py:533-534, _AABB_SAMPLES
AABB_SAMPLES = (
    (-1, 0, 0), (-1, .7, 0), (-1, -.7, 0), (-1, 0, .7), (-1, 0, -.7),
    (0, .9, 0), (0, -.9, 0), (0, 0, -.9),
)


def skin_detects(model, data, sensor_cam_ids, boxes) -> bool:
    """_protrusion_detected's exact geometry, applied to world-space boxes."""
    rng_eff = SENSOR_RANGE * SENSOR_RANGE_DERATE
    points = []
    for center, half in boxes:
        for sample in AABB_SAMPLES:
            points.append(center + half * np.asarray(sample, dtype=float))
    points = np.asarray(points)
    for cid in sensor_cam_ids:
        pos = np.asarray(data.cam_xpos[cid], dtype=np.float64)
        fwd = -np.asarray(data.cam_xmat[cid], dtype=np.float64).reshape(3, 3)[:, 2]
        delta = points - pos
        dist = np.linalg.norm(delta, axis=1)
        good = (dist > 1e-9) & (dist <= rng_eff)
        if not good.any():
            continue
        cosine = (delta[good] @ fwd) / dist[good]
        if float(cosine.max()) > SENSOR_HALF_FOV_COS:
            return True
    return False


def measure_episode(job: dict[str, Any]) -> dict[str, Any]:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("MLSPACES_ASSETS_DIR", str(ROOT / "assets"))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.pop("DISPLAY", None)

    import mujoco
    from molmo_spaces.data_generation.pipeline import cleanup_episode_resources
    from run_pact_place_v6c_replay_videos import _prepare_task, apply_recorded_qpos
    from run_pact_place_swept_volume_v7 import geom_groups
    from run_pact_place_v8_baseline import (
        _active_clutter_geoms, _physical_geoms, _target_geoms,
    )
    import pact_geom_distance as inst

    inst.reset_counters()
    row = job["row"]
    result = json.loads(Path(job["result_path"]).read_text())
    steps = list(json.loads(Path(job["trajectory_path"]).read_text())["steps"])

    task = sampler = scratch = None
    try:
        task, sampler, scratch = _prepare_task(row, result["selected_seed"])
        model, data = task.env.current_model, task.env.current_data
        groups = geom_groups(model)
        phys = {
            **{k: _physical_geoms(model, list(groups[k]))
               for k in ("link5", "link6", "link7", "hand")},
            "fingers": _physical_geoms(
                model, list(groups["left_finger"]) + list(groups["right_finger"])),
            "cup": _physical_geoms(model, _target_geoms(model)),
        }
        sensor_cam_ids = [i for i in range(int(model.ncam))
                          if "_sensor_" in (model.camera(i).name or "")]

        apply_recorded_qpos(task.env, steps[0]["qpos"])
        clutter = _physical_geoms(model, _active_clutter_geoms(model, data))
        base_size = {gid: np.asarray(model.geom_size[gid], dtype=float).copy()
                     for gid in clutter}
        base_mocap = {}
        for gid in clutter:
            body_id = int(model.geom_bodyid[gid])
            mocap_id = int(model.body_mocapid[body_id])
            base_mocap[gid] = (mocap_id,
                               np.asarray(data.mocap_pos[mocap_id], dtype=float).copy())

        results = {}
        for height in job["heights"]:
            for gid in clutter:
                size = base_size[gid].copy()
                size[2] = height / 2.0
                model.geom_size[gid] = size
                mocap_id, pos = base_mocap[gid]
                data.mocap_pos[mocap_id] = [pos[0], pos[1], SHELF_TOP_Z + height / 2.0]

            best = {k: np.inf for k in BODY_KEYS}
            frames_near = {k: 0 for k in ("link5", "link6")}
            frames_very_near = {k: 0 for k in ("link5", "link6")}
            detect_frames = 0
            n_frames = 0
            witness = None
            for step in steps:
                apply_recorded_qpos(task.env, step["qpos"])
                n_frames += 1
                boxes = [
                    (np.asarray(data.geom_xpos[gid], dtype=float),
                     np.asarray(model.geom_size[gid], dtype=float))
                    for gid in clutter
                ]
                if skin_detects(model, data, sensor_cam_ids, boxes):
                    detect_frames += 1
                for key in BODY_KEYS:
                    if not phys[key]:
                        continue
                    value = inst.true_distance(model, data, phys[key], clutter)
                    if value < best[key]:
                        best[key] = value
                        if key in ("link5", "link6"):
                            # witness points for the axis decomposition
                            segment = np.zeros(6)
                            pair_best = np.inf
                            pair_seg = None
                            for lg in phys[key]:
                                for cg in clutter:
                                    segment[:] = 0.0
                                    v = float(mujoco.mj_geomDistance(
                                        model, data, int(lg), int(cg), 10.0, segment))
                                    if 0.0 < v < pair_best:
                                        pair_best = v
                                        pair_seg = segment.copy()
                            if pair_seg is not None:
                                witness = {
                                    "body": key,
                                    "distance_m": float(pair_best),
                                    "delta_xyz_m": np.abs(
                                        pair_seg[3:] - pair_seg[:3]).tolist(),
                                }
                    if key in ("link5", "link6"):
                        if value < NEAR_M:
                            frames_near[key] += 1
                        if value < VERY_NEAR_M:
                            frames_very_near[key] += 1
            results[f"{height:.2f}"] = {
                "height_m": height,
                "clutter_top_z_m": SHELF_TOP_Z + height,
                "min_clearance_m": {k: (None if not np.isfinite(v) else float(v))
                                    for k, v in best.items()},
                "frames_lt_10cm": dict(frames_near),
                "frames_lt_5cm": dict(frames_very_near),
                "skin_detect_frames": detect_frames,
                "skin_detect_fraction": float(detect_frames / max(1, n_frames)),
                "closest_approach_axis_decomposition": witness,
            }

        for gid in clutter:                                    # restore
            model.geom_size[gid] = base_size[gid]
            mocap_id, pos = base_mocap[gid]
            data.mocap_pos[mocap_id] = pos
        return {
            "role_index": int(row["role_index"]),
            "episode_id": row["episode_id"],
            "n_steps": len(steps),
            "n_clutter_geoms": len(clutter),
            "by_height": results,
            "instrument_counters": inst.counters(),
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
    parser.add_argument("--rows", type=int, default=24)
    args = parser.parse_args()

    config = json.loads(CONFIG_PATH.read_text())
    jobs = []
    for row in config["expert_screen_rows"]:
        if int(row["role_index"]) >= args.rows:
            continue
        directory = row_directory(int(row["role_index"]), row["episode_id"])
        jobs.append({
            "row": row,
            "result_path": str(directory / "result.json"),
            "trajectory_path": str(directory / "trajectory.json"),
            "heights": list(HEIGHTS_M),
        })

    context = multiprocessing.get_context("spawn")
    episodes = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max(1, args.workers), mp_context=context
    ) as pool:
        for episode in pool.map(measure_episode, jobs):
            episodes.append(episode)
            print(f"row {episode['role_index']:02d} done", flush=True)
    episodes.sort(key=lambda item: item["role_index"])

    summary = {}
    for height in HEIGHTS_M:
        key = f"{height:.2f}"
        rows = [e["by_height"][key] for e in episodes]
        per_body = {}
        for body in BODY_KEYS:
            values = [r["min_clearance_m"][body] for r in rows
                      if r["min_clearance_m"][body] is not None]
            per_body[body] = float(min(values)) if values else None
        summary[key] = {
            "height_m": height,
            "clutter_top_z_m": SHELF_TOP_Z + height,
            "min_clearance_by_body_m": per_body,
            "cup_is_closest_body_episodes": sum(
                1 for r in rows
                if r["min_clearance_m"]["cup"] is not None
                and r["min_clearance_m"]["cup"] < min(
                    r["min_clearance_m"]["link5"], r["min_clearance_m"]["link6"])
            ),
            "frames_lt_10cm_link5": sum(r["frames_lt_10cm"]["link5"] for r in rows),
            "frames_lt_10cm_link6": sum(r["frames_lt_10cm"]["link6"] for r in rows),
            "frames_lt_5cm_link5": sum(r["frames_lt_5cm"]["link5"] for r in rows),
            "frames_lt_5cm_link6": sum(r["frames_lt_5cm"]["link6"] for r in rows),
            "skin_detect_frames": sum(r["skin_detect_frames"] for r in rows),
            "total_frames": sum(e["n_steps"] for e in episodes),
            "episodes_with_any_skin_detection": sum(
                1 for r in rows if r["skin_detect_frames"] > 0),
        }

    document = {
        "schema_version": "pact_place_v9_v0c_height_precheck_v1",
        "role": "v0c_precheck_measurement_not_a_gate",
        "authorizes_gate": False,
        "replay_only": True,
        "physics_stepped": False,
        "shelf_top_z_m": SHELF_TOP_Z,
        "skin_gate": {
            "source": "enclosure_reach.py:536-557 _protrusion_detected",
            "half_fov_deg": 22.5,
            "range_m": SENSOR_RANGE,
            "derate": SENSOR_RANGE_DERATE,
            "effective_range_m": SENSOR_RANGE * SENSOR_RANGE_DERATE,
        },
        "summary": summary,
        "episodes": episodes,
    }
    document["analysis_sha256"] = sha256_payload(document)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json_atomic(OUTPUT_DIR / "v0c_height_precheck.json", document)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
