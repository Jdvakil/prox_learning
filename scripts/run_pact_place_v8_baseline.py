#!/usr/bin/env python3
"""B0: replay v6c and v7 and measure link-level proximity and visibility.

This is a forward-only replay.  It restores the complete recorded model qpos,
calls ``mj_forward``, and never steps physics or runs the expert.  Distances are
computed between MuJoCo geoms; TCP-to-box proxies are intentionally not used.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MOLMO = ROOT / "submodules" / "molmospaces"
for search_path in (ROOT / "scripts", MOLMO):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from pact_place_corridor_contract import (  # noqa: E402
    load_contract,
    load_design_review_contract,
    sha256_payload,
)
from run_pact_place_swept_volume_v7 import (  # noqa: E402
    INTERIOR_X,
    INTERIOR_Y,
    INTERIOR_Z,
    VOXEL_M,
    aabb_distance,
    geom_groups,
    mark_aabb_voxels,
    pack_voxel,
    world_aabb_for_geom,
)
from run_pact_place_v6c_replay_videos import (  # noqa: E402
    CONFIG_PATH as V6C_CONFIG,
    V6C_CONFIG_SHA256,
    _prepare_task as prepare_v6c,
    apply_recorded_qpos,
    row_directory as v6c_row_directory,
)
from run_pact_place_v7_replay_videos import (  # noqa: E402
    CONFIG_PATH as V7_CONFIG,
    SCREEN_ROOT as V7_SCREEN_ROOT,
    WRIST_CAMERA_MJCF,
    _prepare_task as prepare_v7,
)

OUTPUT_DIR = ROOT / "diagnostics_output" / "pact_place_v8_baseline"
LINKS = ("link1", "link2", "link3", "link4", "link5", "link6")
BODIES = LINKS + ("cup",)
TRACK_BODIES = BODIES + ("hand_assembly",)
NEAR_M = 0.05
MEDIUM_M = 0.10
FAR_M = 0.15
RAY_SAMPLES_PER_GEOM = 7


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _root_name(model, body_id: int) -> str:
    root_id = int(model.body_rootid[int(body_id)])
    return model.body(root_id).name or ""


def _active_clutter_geoms(model, data) -> list[int]:
    result: list[int] = []
    for gid in range(int(model.ngeom)):
        body_id = int(model.geom_bodyid[gid])
        names = (
            model.geom(gid).name or "",
            model.body(body_id).name or "",
            _root_name(model, body_id),
        )
        if not any("pact_clutter_" in name for name in names):
            continue
        # Both frozen baselines park unused mocap slots at z=-2.
        if float(data.geom_xpos[gid][2]) > -1.0:
            result.append(gid)
    return result


def _target_geoms(model) -> list[int]:
    result = []
    for gid in range(int(model.ngeom)):
        body_id = int(model.geom_bodyid[gid])
        names = (
            model.geom(gid).name or "",
            model.body(body_id).name or "",
            _root_name(model, body_id),
        )
        if any("cavity_obj_" in name for name in names) and int(
            model.geom_contype[gid]
        ):
            result.append(gid)
    return result


def _physical_geoms(model, gids: list[int]) -> list[int]:
    """Collision bodies define physical clearance; visual/skin meshes overlap them."""
    return [
        int(gid)
        for gid in gids
        if int(model.geom_contype[int(gid)]) or int(model.geom_conaffinity[int(gid)])
    ]


def _geom_distance(model, data, left: list[int], right: list[int]) -> float:
    if not left or not right:
        return float("inf")
    best = float("inf")
    for gid1 in left:
        left_lo, left_hi = world_aabb_for_geom(model, data, int(gid1))
        for gid2 in right:
            right_lo, right_hi = world_aabb_for_geom(model, data, int(gid2))
            value = aabb_distance(
                left_lo, left_hi, right_lo, right_hi
            )
            if value < best:
                best = value
                if best <= 0.0:
                    return best
    return best


def _surface_samples(model, data, gids: list[int]) -> np.ndarray:
    points = []
    for gid in gids:
        lo, hi = world_aabb_for_geom(model, data, int(gid))
        center = (lo + hi) / 2.0
        points.append(center)
        for axis in range(3):
            low = center.copy()
            high = center.copy()
            low[axis] = lo[axis]
            high[axis] = hi[axis]
            points.extend((low, high))
    if not points:
        return np.empty((0, 3), dtype=np.float64)
    return np.asarray(points, dtype=np.float64)


def _any_visible(model, data, camera_id: int, gids: list[int]) -> bool:
    """Boolean ray visibility; recorded replay has no segmentation masks."""
    import mujoco

    if not gids:
        return False
    wanted_roots = {
        int(model.body_rootid[int(model.geom_bodyid[int(gid)])]) for gid in gids
    }
    origin = np.asarray(data.cam_xpos[int(camera_id)], dtype=np.float64)
    rotation = np.asarray(data.cam_xmat[int(camera_id)], dtype=np.float64).reshape(3, 3)
    forward = -rotation[:, 2]
    half_fov = np.deg2rad(float(model.cam_fovy[int(camera_id)])) / 2.0
    min_cosine = float(np.cos(half_fov))
    geomid = np.zeros(1, dtype=np.int32)
    for point in _surface_samples(model, data, gids):
        delta = point - origin
        distance = float(np.linalg.norm(delta))
        if distance < 1e-8:
            continue
        direction = delta / distance
        if float(np.dot(direction, forward)) < min_cosine:
            continue
        hit = mujoco.mj_ray(
            model,
            data,
            origin,
            direction.astype(np.float64),
            None,
            1,
            -1,
            geomid,
        )
        if hit < 0.0 or int(geomid[0]) < 0:
            continue
        hit_root = int(
            model.body_rootid[int(model.geom_bodyid[int(geomid[0])])]
        )
        if hit_root in wanted_roots:
            return True
    return False


def _union_aabb(current: dict[str, list[float]] | None, lo, hi):
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    if current is None:
        return {"min_m": lo.tolist(), "max_m": hi.tolist()}
    return {
        "min_m": np.minimum(current["min_m"], lo).tolist(),
        "max_m": np.maximum(current["max_m"], hi).tolist(),
    }


def _longest_true_run(values: list[bool]) -> int:
    best = current = 0
    for value in values:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def _row_paths(source: str, row: dict[str, Any]) -> tuple[Path, Path]:
    index = int(row["role_index"])
    if source == "v6c":
        directory = v6c_row_directory(index, row["episode_id"])
    else:
        directory = (
            V7_SCREEN_ROOT
            / "expert_screen_rows"
            / f"{index:02d}_{row['episode_id'][:16]}"
        )
    return directory / "result.json", directory / "trajectory.json"


def _prepare(source: str, row: dict[str, Any], result: dict[str, Any], cfg):
    if source == "v6c":
        return prepare_v6c(row, result["selected_seed"])
    return prepare_v7(row, result["selected_seed"], cfg)


def measure_row(job: dict[str, Any]) -> dict[str, Any]:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("MLSPACES_ASSETS_DIR", "/root/prox_learning/assets")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.pop("DISPLAY", None)

    from molmo_spaces.data_generation.pipeline import cleanup_episode_resources

    source = str(job["source"])
    row = job["row"]
    cfg = job["config"]
    result = json.loads(Path(job["result_path"]).read_text())
    trajectory = json.loads(Path(job["trajectory_path"]).read_text())
    steps = list(trajectory["steps"])
    task = sampler = scratch = None
    try:
        task, sampler, scratch = _prepare(source, row, result, cfg)
        env = task.env
        model, data = env.current_model, env.current_data
        groups_all = geom_groups(model)
        groups = {
            key: _physical_geoms(model, list(groups_all.get(key) or []))
            for key in BODIES
        }
        groups["hand_assembly"] = _physical_geoms(
            model,
            [
                gid
                for key in ("link7", "hand", "left_finger", "right_finger")
                for gid in list(groups_all.get(key) or [])
            ],
        )
        target = _target_geoms(model)
        wrist_id = int(model.camera(WRIST_CAMERA_MJCF).id)

        origin = np.asarray(
            [INTERIOR_X[0], INTERIOR_Y[0], INTERIOR_Z[0]], dtype=float
        )
        shape = tuple(
            int(np.ceil((hi - lo) / VOXEL_M))
            for lo, hi in (INTERIOR_X, INTERIOR_Y, INTERIOR_Z)
        )
        occupancy = {link: set() for link in LINKS}
        swept = {link: None for link in LINKS}
        min_clear = {body: float("inf") for body in BODIES}
        min_frame = {body: None for body in BODIES}
        per_link_clearances = {link: [] for link in LINKS}
        any_link_clearances: list[float] = []
        cup_clearances: list[float] = []
        phases: list[str] = []
        clutter_visible: list[bool] = []
        target_visible: list[bool] = []
        track: dict[str, list[np.ndarray]] = {
            f"geom_aabb_{body}": [] for body in TRACK_BODIES
        }
        track.update({"wrist_pos": [], "wrist_fwd": [], "target_center": []})

        qpos_width = int(model.nq)
        for frame_index, step in enumerate(steps):
            qpos = step.get("qpos")
            if len(qpos) != qpos_width:
                raise RuntimeError(
                    f"{source} row {row['role_index']} qpos width {len(qpos)} != nq {qpos_width}"
                )
            apply_recorded_qpos(env, qpos)
            clutter = _active_clutter_geoms(model, data)
            phase = str(step.get("policy_phase") or "unknown")
            phases.append(phase)
            clutter_visible.append(_any_visible(model, data, wrist_id, clutter))
            target_visible.append(_any_visible(model, data, wrist_id, target))

            frame_link_min = float("inf")
            for body in BODIES:
                distance = _geom_distance(model, data, groups[body], clutter)
                if body in LINKS:
                    per_link_clearances[body].append(distance)
                    frame_link_min = min(frame_link_min, distance)
                else:
                    cup_clearances.append(distance)
                if distance < min_clear[body]:
                    min_clear[body] = distance
                    min_frame[body] = frame_index
            any_link_clearances.append(frame_link_min)

            if source == "v6c":
                for body in TRACK_BODIES:
                    aabbs = []
                    for gid in groups[body]:
                        lo, hi = world_aabb_for_geom(model, data, gid)
                        aabbs.append(np.concatenate((lo, hi)))
                        if body in LINKS:
                            swept[body] = _union_aabb(swept[body], lo, hi)
                            mark_aabb_voxels(lo, hi, origin, shape, occupancy[body])
                    track[f"geom_aabb_{body}"].append(
                        np.asarray(aabbs, dtype=np.float32)
                    )
                wpos = np.asarray(data.cam_xpos[wrist_id], dtype=np.float64)
                wmat = np.asarray(data.cam_xmat[wrist_id], dtype=np.float64).reshape(3, 3)
                wfwd = -wmat[:, 2]
                wfwd /= np.linalg.norm(wfwd)
                track["wrist_pos"].append(wpos)
                track["wrist_fwd"].append(wfwd)
                target_points = _surface_samples(model, data, target)
                track["target_center"].append(
                    target_points.mean(axis=0)
                    if len(target_points)
                    else np.full(3, np.nan)
                )

        overall_index = int(np.argmin(any_link_clearances))
        overall_min = float(any_link_clearances[overall_index])
        cup_min = float(min_clear["cup"])
        by_phase: dict[str, dict[str, int]] = defaultdict(
            lambda: {"frames": 0, "visible_frames": 0}
        )
        for phase, visible in zip(phases, target_visible):
            by_phase[phase]["frames"] += 1
            by_phase[phase]["visible_frames"] += int(visible)
        # In the place expert the target-acquisition approach is named
        # ``pregrasp``.  The inbound/outbound labels describe the later carried
        # traversal, after the target has already been acquired.
        approach_mask = [phase == "pregrasp" for phase in phases]
        approach_visible = [
            bool(visible and mask)
            for visible, mask in zip(target_visible, approach_mask)
        ]
        first_clutter = next(
            (i for i, visible in enumerate(clutter_visible) if visible), None
        )
        row_payload = {
            "source": source,
            "role_index": int(row["role_index"]),
            "episode_id": row["episode_id"],
            "intrusion_side": row.get("intrusion_side"),
            "n_steps": len(steps),
            "qpos_width": qpos_width,
            "episode_depth_m": float(
                (getattr(task, "scene_params", {}) or {}).get("depth", 0.0)
            ),
            "episode_back_wall_x_m": float(
                0.58
                + (getattr(task, "scene_params", {}) or {}).get("depth", 0.0)
                + 0.02
            ),
            "min_clearance_by_link_m": {
                link: float(min_clear[link]) for link in LINKS
            },
            "min_cup_clearance_m": cup_min,
            "frames_link_clearance_lt_5cm": int(
                np.sum(np.asarray(any_link_clearances) < NEAR_M)
            ),
            "frames_link_clearance_lt_10cm": int(
                np.sum(np.asarray(any_link_clearances) < MEDIUM_M)
            ),
            "frames_link_clearance_lt_15cm": int(
                np.sum(np.asarray(any_link_clearances) < FAR_M)
            ),
            "frames_by_link_clearance_band": {
                link: {
                    "lt_5cm": int(np.sum(np.asarray(values) < NEAR_M)),
                    "lt_10cm": int(np.sum(np.asarray(values) < MEDIUM_M)),
                    "lt_15cm": int(np.sum(np.asarray(values) < FAR_M)),
                }
                for link, values in per_link_clearances.items()
            },
            "n_distinct_links_exposed": int(
                sum(min_clear[link] < MEDIUM_M for link in LINKS)
            ),
            "phase_of_min_clearance": phases[overall_index],
            "frame_of_min_clearance": overall_index,
            "min_link_clearance_m": overall_min,
            "closest_link": min(LINKS, key=lambda link: min_clear[link]),
            "cup_is_closest_body": bool(cup_min < overall_min),
            "clutter_visible_frames": int(sum(clutter_visible)),
            "first_visible_frame": first_clutter,
            "visibility_at_min_link_clearance": bool(
                clutter_visible[overall_index]
            ),
            "target_visible_frames": int(sum(target_visible)),
            "target_visibility_by_phase": dict(by_phase),
            "target_longest_consecutive_visible_approach_frames": _longest_true_run(
                approach_visible
            ),
            "visibility_instrument": {
                "method": "wrist_camera_boolean_mj_ray_to_geom_aabb_surface_samples",
                "segmentation_masks_available_in_recorded_replay": False,
                "pixel_fraction_substituted": False,
                "samples_per_geom": RAY_SAMPLES_PER_GEOM,
            },
        }
        if source == "v6c":
            track_dir = OUTPUT_DIR / "tracks"
            track_dir.mkdir(parents=True, exist_ok=True)
            track_path = track_dir / f"row{int(row['role_index']):02d}.npz"
            np.savez_compressed(
                track_path,
                **{key: np.asarray(value) for key, value in track.items()},
                phases=np.asarray(phases),
                target_visible=np.asarray(target_visible, dtype=bool),
                clutter_visible=np.asarray(clutter_visible, dtype=bool),
                step_index=np.arange(len(steps), dtype=np.int32),
            )
            row_payload["track_path"] = str(track_path.relative_to(ROOT))
            row_payload["swept_volume_by_link"] = swept
            row_payload["link_occupancy_voxels"] = {
                link: sorted(values) for link, values in occupancy.items()
            }
        return row_payload
    finally:
        cleanup_episode_resources(
            task=task,
            policy=None,
            task_sampler=sampler,
            preloaded_policy=None,
            close_task_sampler=sampler is not None,
        )
        if scratch is not None:
            shutil.rmtree(scratch, ignore_errors=True)


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_steps = sum(int(row["n_steps"]) for row in rows)
    return {
        "n_episodes": len(rows),
        "n_frames": total_steps,
        "min_clearance_by_link_m": {
            link: min(row["min_clearance_by_link_m"][link] for row in rows)
            for link in LINKS
        },
        "frames_link_clearance_lt_5cm": sum(
            row["frames_link_clearance_lt_5cm"] for row in rows
        ),
        "frames_link_clearance_lt_10cm": sum(
            row["frames_link_clearance_lt_10cm"] for row in rows
        ),
        "frames_link_clearance_lt_15cm": sum(
            row["frames_link_clearance_lt_15cm"] for row in rows
        ),
        "mean_distinct_links_exposed": float(
            np.mean([row["n_distinct_links_exposed"] for row in rows])
        ),
        "n_episodes_cup_is_closest_body": sum(
            bool(row["cup_is_closest_body"]) for row in rows
        ),
        "cup_is_closest_body_fraction": float(
            np.mean([row["cup_is_closest_body"] for row in rows])
        ),
        "visibility_at_min_link_clearance_fraction": float(
            np.mean([row["visibility_at_min_link_clearance"] for row in rows])
        ),
        "clutter_visible_frame_fraction": (
            sum(row["clutter_visible_frames"] for row in rows) / total_steps
            if total_steps
            else 0.0
        ),
        "target_visible_frame_fraction": (
            sum(row["target_visible_frames"] for row in rows) / total_steps
            if total_steps
            else 0.0
        ),
        "phase_of_min_clearance_counts": {
            phase: sum(row["phase_of_min_clearance"] == phase for row in rows)
            for phase in sorted({row["phase_of_min_clearance"] for row in rows})
        },
    }


def _merge_v6c_spatial(rows: list[dict[str, Any]]):
    swept = {link: None for link in LINKS}
    occupied = {link: set() for link in LINKS}
    for row in rows:
        for link in LINKS:
            block = row["swept_volume_by_link"][link]
            if block is not None:
                swept[link] = _union_aabb(
                    swept[link], block["min_m"], block["max_m"]
                )
            occupied[link].update(row["link_occupancy_voxels"][link])
    return swept, {link: sorted(values) for link, values in occupied.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--source", choices=("all", "v6c", "v7"), default="all")
    parser.add_argument("--role-index", type=int, nargs="*")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise SystemExit("workers must be in [1,8]")

    v6c = load_contract(V6C_CONFIG)
    if v6c["config_sha256"] != V6C_CONFIG_SHA256:
        raise SystemExit("frozen v6c config hash changed")
    v7 = load_design_review_contract(V7_CONFIG)
    sources = ("v6c", "v7") if args.source == "all" else (args.source,)
    jobs = []
    for source, cfg in (("v6c", v6c), ("v7", v7)):
        if source not in sources:
            continue
        rows = cfg["expert_screen_rows"]
        if source == "v7":
            # The v7 human review stopped after three successful episodes.
            rows = rows[:3]
        for row in rows:
            if args.role_index is not None and int(row["role_index"]) not in args.role_index:
                continue
            result_path, trajectory_path = _row_paths(source, row)
            if not result_path.is_file() or not trajectory_path.is_file():
                raise SystemExit(f"missing recorded {source} row: {result_path.parent}")
            jobs.append(
                {
                    "source": source,
                    "row": row,
                    "config": cfg,
                    "result_path": str(result_path),
                    "trajectory_path": str(trajectory_path),
                }
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    measured: list[dict[str, Any]] = []
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers, mp_context=context, max_tasks_per_child=1
    ) as executor:
        futures = {executor.submit(measure_row, job): job for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            measured.append(row)
            print(
                f"{row['source']} row={row['role_index']:02d} "
                f"min_link={row['min_link_clearance_m']:.4f}m "
                f"cup_closest={row['cup_is_closest_body']} "
                f"lt10={row['frames_link_clearance_lt_10cm']}",
                flush=True,
            )
    measured.sort(key=lambda row: (row["source"], row["role_index"]))
    by_source = {
        source: [row for row in measured if row["source"] == source]
        for source in sources
    }
    aggregates = {
        source: _aggregate(rows) for source, rows in by_source.items() if rows
    }
    swept = occupancy = None
    if by_source.get("v6c"):
        swept, occupancy = _merge_v6c_spatial(by_source["v6c"])
        occupancy_path = OUTPUT_DIR / "link_occupancy_voxels.json"
        occupancy_path.write_text(json.dumps(occupancy, sort_keys=True) + "\n")
    analysis = {
        "schema_version": "pact_place_v8_baseline_v1",
        "role": "b0_replay_measurement_not_a_gate",
        "authorizes_gate": False,
        "authorizes_collection": False,
        "no_new_rollouts": True,
        "physics_stepped": False,
        "replay_operation": "restore_complete_qpos_then_mj_forward",
        "distance_instrument": (
            "pairwise_world_AABB_distance_over_each_physical_link_geom_and_clutter_geom"
        ),
        "distance_instrument_note": (
            "MuJoCo mj_geomDistance returned false zeroes for distant mesh pairs in "
            "clean frozen v6c frames; per-geom world AABBs are used instead. This is "
            "not a per-link hull and never uses TCP-to-AABB distance."
        ),
        "visibility_instrument": {
            "method": "boolean_mj_ray_from_wrist_to_geom_surface_samples",
            "segmentation_masks_available": False,
            "pixel_fraction_substituted": False,
        },
        "clearance_bands_m": {
            "near": NEAR_M,
            "medium": MEDIUM_M,
            "far": FAR_M,
            "diagnostic_not_frozen_constants": True,
        },
        "sources": {
            "v6c": "diagnostics_output/pact_place_corridor_v6c",
            "v7": "diagnostics_output/pact_place_corridor_v7_design_review",
        },
        "aggregates": aggregates,
        "swept_volume_by_link": swept,
        "link_occupancy_voxels": (
            None
            if occupancy is None
            else "diagnostics_output/pact_place_v8_baseline/link_occupancy_voxels.json"
        ),
        "voxel_m": VOXEL_M,
        "rows": [
            {
                key: value
                for key, value in row.items()
                if key not in {"link_occupancy_voxels", "swept_volume_by_link"}
            }
            for row in measured
        ],
    }
    analysis["analysis_sha256"] = sha256_payload(analysis)
    output = OUTPUT_DIR / "analysis.json"
    output.write_text(json.dumps(_jsonable(analysis), indent=2, sort_keys=True) + "\n")
    print(output)
    print(json.dumps(aggregates, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
