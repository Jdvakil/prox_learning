#!/usr/bin/env python3
"""V0a: re-measure the v6c clutter clearances with the hardened instrument.

Every "the cup is closest" claim in this programme traces to
``run_pact_place_swept_volume_v7.py:443-448``, which takes an ``aabb_distance``
between a link's group AABB and the clutter AABBs. That instrument has two
independent defects:

  (a) an AABB gap is a *lower bound* -- once the boxes overlap it returns exactly
      0.0 whether the true separation is 0 or 0.2 m; and
  (b) it iterates ``groups[key]``, i.e. every geom on the body including visual
      and skin-sensor meshes, and ``world_aabb_for_geom`` falls back to
      ``pos +/- geom_rbound`` for a mesh -- a sphere far larger than the part.

This replays the 24 frozen v6c trajectories and reports three instruments side by
side so the two defects are attributed separately:

  ``aabb_v7``        verbatim reproduction of the published instrument
  ``aabb_physical``  same AABB gap, collision geoms only  -> isolates (b)
  ``hardened``       exact per-geom distance, collision geoms only -> the truth

Replay only; no physics is stepped and no v6c artifact is modified.
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
BODY_KEYS = (
    "link1", "link2", "link3", "link4", "link5", "link6", "link7",
    "hand", "fingers", "cup",
)
LINK56 = ("link5", "link6")
NEAR_M = 0.10
VERY_NEAR_M = 0.05
# v7's published figure, for the side-by-side
V7_PUBLISHED_MIN_CLEARANCE_M = {
    "cup": 0.0, "hand": 0.08149510062545895, "left_finger": 0.04506947433788346,
    "link1": 0.27165919022237006, "link2": 0.26817173300962305,
    "link3": 0.16143183021730165, "link4": 0.10435013938227027,
    "link5": 0.05681711666498512, "link6": 0.08843616478916053,
    "link7": 0.09867444414145723, "right_finger": 0.04496911511634841,
}


def measure_episode(job: dict[str, Any]) -> dict[str, Any]:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("MLSPACES_ASSETS_DIR", str(ROOT / "assets"))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.pop("DISPLAY", None)

    from molmo_spaces.data_generation.pipeline import cleanup_episode_resources
    from run_pact_place_v6c_replay_videos import _prepare_task, apply_recorded_qpos
    from run_pact_place_swept_volume_v7 import (
        aabb_distance, clutter_geoms, geom_groups, world_aabb_for_geom,
    )
    from run_pact_place_v8_baseline import (
        _active_clutter_geoms, _any_visible, _physical_geoms, _target_geoms,
    )
    from run_pact_place_v7_replay_videos import WRIST_CAMERA_MJCF
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
        target = _target_geoms(model)

        all_geoms = {
            **{k: list(groups[k]) for k in BODY_KEYS if k not in ("fingers", "cup")},
            "fingers": list(groups["left_finger"]) + list(groups["right_finger"]),
            "cup": list(groups["cup"]),
        }
        phys_geoms = {
            **{k: _physical_geoms(model, list(groups[k]))
               for k in BODY_KEYS if k not in ("fingers", "cup")},
            "fingers": _physical_geoms(
                model, list(groups["left_finger"]) + list(groups["right_finger"])),
            "cup": _physical_geoms(model, target),
        }
        v7_clutter = clutter_geoms(model)              # verbatim v7 selection
        wrist_id = int(model.camera(WRIST_CAMERA_MJCF).id)

        best = {name: {k: np.inf for k in BODY_KEYS}
                for name in ("aabb_v7", "aabb_physical", "hardened")}
        frames_near = {k: 0 for k in LINK56}
        frames_very_near = {k: 0 for k in LINK56}
        argmin_frame = 0
        running_min = np.inf
        n_frames = 0

        for frame_index, step in enumerate(steps):
            apply_recorded_qpos(task.env, step["qpos"])
            n_frames += 1
            clutter_phys = _physical_geoms(model, _active_clutter_geoms(model, data))
            v7_boxes = [world_aabb_for_geom(model, data, gid) for _, gid in v7_clutter]
            phys_boxes = [world_aabb_for_geom(model, data, gid) for gid in clutter_phys]

            frame_hardened = {}
            for key in BODY_KEYS:
                # (1) verbatim v7: every geom on the body, group AABB
                gids = all_geoms[key]
                if gids and v7_boxes:
                    lo = np.full(3, np.inf)
                    hi = np.full(3, -np.inf)
                    for gid in gids:
                        glo, ghi = world_aabb_for_geom(model, data, gid)
                        lo = np.minimum(lo, glo)
                        hi = np.maximum(hi, ghi)
                    value = min(aabb_distance(lo, hi, clo, chi) for clo, chi in v7_boxes)
                    best["aabb_v7"][key] = min(best["aabb_v7"][key], value)

                # (2) same AABB gap, collision geoms only
                pgids = phys_geoms[key]
                if pgids and phys_boxes:
                    lo = np.full(3, np.inf)
                    hi = np.full(3, -np.inf)
                    for gid in pgids:
                        glo, ghi = world_aabb_for_geom(model, data, gid)
                        lo = np.minimum(lo, glo)
                        hi = np.maximum(hi, ghi)
                    value = min(aabb_distance(lo, hi, clo, chi) for clo, chi in phys_boxes)
                    best["aabb_physical"][key] = min(best["aabb_physical"][key], value)

                # (3) hardened exact
                if pgids and clutter_phys:
                    value = inst.true_distance(model, data, pgids, clutter_phys)
                    frame_hardened[key] = value
                    best["hardened"][key] = min(best["hardened"][key], value)

            if frame_hardened:
                link_min = min(frame_hardened[k] for k in LINK56 if k in frame_hardened)
                for key in LINK56:
                    if key in frame_hardened:
                        if frame_hardened[key] < NEAR_M:
                            frames_near[key] += 1
                        if frame_hardened[key] < VERY_NEAR_M:
                            frames_very_near[key] += 1
                if link_min < running_min:
                    running_min = link_min
                    argmin_frame = frame_index

        # visibility of the clutter at the frame of closest link5/link6 approach
        apply_recorded_qpos(task.env, steps[argmin_frame]["qpos"])
        clutter_phys = _physical_geoms(model, _active_clutter_geoms(model, data))
        visible_at_min = bool(_any_visible(model, data, wrist_id, clutter_phys))

        out: dict[str, Any] = {
            "role_index": int(row["role_index"]),
            "episode_id": row["episode_id"],
            "n_steps": n_frames,
            "clean_success": bool(result.get("clean_success")),
            "min_clearance_m": {
                name: {k: (None if not np.isfinite(v) else float(v))
                       for k, v in table.items()}
                for name, table in best.items()
            },
            "frames_lt_10cm": dict(frames_near),
            "frames_lt_5cm": dict(frames_very_near),
            "min_link5_link6_m": (None if not np.isfinite(running_min)
                                  else float(running_min)),
            "phase_of_min": str(steps[argmin_frame].get("policy_phase") or "unknown"),
            "clutter_visible_at_min_link_clearance": visible_at_min,
            "instrument_counters": inst.counters(),
        }
        for name, table in best.items():
            finite = {k: v for k, v in table.items() if np.isfinite(v)}
            if not finite:
                continue
            closest = min(finite, key=finite.get)
            link_values = [finite[k] for k in LINK56 if k in finite]
            out.setdefault("closest_body", {})[name] = closest
            out.setdefault("cup_is_closest_body", {})[name] = bool(
                "cup" in finite and link_values and finite["cup"] < min(link_values)
            )
        return out
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
        jobs.append({
            "row": row,
            "result_path": str(directory / "result.json"),
            "trajectory_path": str(directory / "trajectory.json"),
        })

    context = multiprocessing.get_context("spawn")
    episodes = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max(1, args.workers), mp_context=context
    ) as pool:
        for episode in pool.map(measure_episode, jobs):
            episodes.append(episode)
            print(f"row {episode['role_index']:02d} "
                  f"cup_closest={episode['cup_is_closest_body']}", flush=True)
    episodes.sort(key=lambda item: item["role_index"])

    instruments = ("aabb_v7", "aabb_physical", "hardened")
    summary = {}
    for name in instruments:
        per_body = {}
        for key in BODY_KEYS:
            values = [e["min_clearance_m"][name][key] for e in episodes
                      if e["min_clearance_m"][name][key] is not None]
            per_body[key] = float(min(values)) if values else None
        summary[name] = {
            "min_clearance_by_body_m": per_body,
            "episodes_cup_is_closest_body": sum(
                e["cup_is_closest_body"][name] for e in episodes),
            "closest_body_counts": {
                body: sum(e["closest_body"][name] == body for e in episodes)
                for body in sorted({e["closest_body"][name] for e in episodes})
            },
        }

    totals = {"calls": 0, "fromto_span_fallback": 0, "gjk_fallback": 0,
              "aabb_disproof_only": 0, "accepted_zero": 0}
    for episode in episodes:
        for key, value in episode["instrument_counters"].items():
            totals[key] = totals.get(key, 0) + int(value)

    document = {
        "schema_version": "pact_place_v9_v0a_v6c_remeasure_v1",
        "role": "v0a_record_correction_not_a_gate",
        "authorizes_gate": False,
        "authorizes_collection": False,
        "replay_only": True,
        "physics_stepped": False,
        "source_screen": "diagnostics_output/pact_place_corridor_v6c",
        "instruments": {
            "aabb_v7": ("verbatim run_pact_place_swept_volume_v7.py:443-448 -- "
                        "aabb_distance over every geom on the body, including "
                        "visual and skin meshes whose AABB is pos +/- geom_rbound"),
            "aabb_physical": "same AABB gap, collision geoms only",
            "hardened": "scripts/pact_geom_distance.true_distance, collision geoms only",
        },
        "v7_published_min_clearance_by_link_m": V7_PUBLISHED_MIN_CLEARANCE_M,
        "summary": summary,
        "instrument_counters_total": totals,
        "aggregate": {
            "n_episodes": len(episodes),
            "frames_lt_10cm_link5": sum(e["frames_lt_10cm"]["link5"] for e in episodes),
            "frames_lt_10cm_link6": sum(e["frames_lt_10cm"]["link6"] for e in episodes),
            "frames_lt_5cm_link5": sum(e["frames_lt_5cm"]["link5"] for e in episodes),
            "frames_lt_5cm_link6": sum(e["frames_lt_5cm"]["link6"] for e in episodes),
            "episodes_clutter_visible_at_min": sum(
                e["clutter_visible_at_min_link_clearance"] for e in episodes),
        },
        "episodes": episodes,
    }
    document["analysis_sha256"] = sha256_payload(document)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json_atomic(OUTPUT_DIR / "v6c_remeasure.json", document)
    print(json.dumps({"summary": summary, "aggregate": document["aggregate"],
                      "counters": totals}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
