#!/usr/bin/env python3
"""V8C C0: site the overhead bar by measurement over the frozen v6c trajectories.

No rollouts, no physics steps, no scene edits. The 24 v6c trajectories are
replayed with ``mj_forward`` only; for each candidate overhead bar the exact
``mujoco.mj_geomDistance`` (with the v8b ``fromto`` mesh fallback) is taken from
the bar's collision geom to each robot body group separately.

The candidate bar is posed by writing ``data.geom_xpos``/``data.geom_xmat`` for a
parked mocap collision geom after ``mj_forward``.  ``verify_pose_equivalence``
proves this is bit-identical to posing the mocap body and re-forwarding, which is
what makes one ``mj_forward`` per frame sufficient for the whole candidate grid.

An axis-aligned bounding-box distance is used ONLY as a conservative skip test
(AABB distance is a lower bound on true distance, so a bound above the cutoff
proves the exact distance is above it too). Every number reported is exact
``mj_geomDistance``; no AABB value is ever recorded as a clearance.
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

# --- measured bounds (see the v7 swept-volume analysis) -----------------------
CUP_SWEPT_Z_MAX = 1.0114741856483134     # swept_aabb_by_link.cup.max_m[2]
ENCLOSURE_CEILING_Z = 1.42               # enclosure_interior_m.z_m[1]
BAND_Z_BOTTOM = 1.05                     # plan C0: >= cup z_max + 39 mm
BAND_Z_TOP = 1.40                        # plan C0: <= ceiling - 20 mm
SHALLOWEST_BACK_WALL_X = 0.7802336048549172
BAND_X_MIN = 0.60
BAND_X_MAX = SHALLOWEST_BACK_WALL_X

# Bodies that can carry the bar. Only ``pact_intrusion_*`` is routed to the
# ``hazard_bar`` contact class (pact_contact_audit.HAZARD_BODY_PREFIX); the
# legacy ``protr_*`` bars fall through to ``other_environment``.
PANEL_HALF = (0.055, 0.240, 0.090)
PROTR_L_HALF_ROTATED = (0.035, 0.120, 0.035)   # protr_l turned to span y

BODY_GROUPS = ("cup", "fingers", "hand", "link7", "link6", "link5")
CUP_SAFE_GROUPS = ("cup", "fingers", "hand")
LINK_GROUPS = ("link5", "link6", "link7")
# measure_pact_place_v8b_realized.py counts distinct exposed links over link1-link6.
# link1-link4 have zero voxels inside the enclosure (v7 swept volume), so that
# instrument's ceiling is 2. Reported separately so the recalibrated gate stays
# comparable with the v8b number it replaces.
V8B_LINK_GROUPS = ("link5", "link6")
PRUNE_CUTOFF_M = 0.20    # exact distance is only needed below this
NEAR_M = 0.10
VERY_NEAR_M = 0.05


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


def candidate_grid() -> list[dict[str, Any]]:
    """Every candidate the plan's bounds allow, on a fixed lattice."""
    candidates: list[dict[str, Any]] = []
    specs = (
        ("pact_intrusion_other", PANEL_HALF, "hazard_bar", 7, 9, 9),
        ("protr_l_rotated", PROTR_L_HALF_ROTATED, "other_environment", 4, 5, 5),
    )
    for body_kind, half, contact_class, nx, ny, nz in specs:
        hx, hy, hz = half
        xs = np.linspace(BAND_X_MIN + hx, BAND_X_MAX - hx, nx)
        ys = np.linspace(-0.20, 0.20, ny)
        zs = np.linspace(BAND_Z_BOTTOM + hz, BAND_Z_TOP - hz, nz)
        for x in xs:
            for y in ys:
                for z in zs:
                    candidates.append(
                        {
                            "candidate_id": f"{body_kind}_x{x:.3f}_y{y:+.3f}_z{z:.3f}",
                            "body_kind": body_kind,
                            "contact_class_if_touched": contact_class,
                            "center_m": [float(x), float(y), float(z)],
                            "half_m": [float(hx), float(hy), float(hz)],
                            "z_bottom_m": float(z - hz),
                            "z_top_m": float(z + hz),
                            "x_front_m": float(x - hx),
                            "x_back_m": float(x + hx),
                        }
                    )
    return candidates


# --- exact distance ----------------------------------------------------------
SPURIOUS_ZEROS = {"n": 0}


def true_distance(model, data, left: list[int], right_gid: int) -> float:
    """Exact geom-to-geom distance, hardened against mj_geomDistance false zeros.

    Two defects are handled. The first is the one v8b found: MuJoCo 3.5 returns a
    scalar zero for some separated pairs while still writing distinct ``fromto``
    endpoints, so that oriented closest-point segment is used instead. The second
    is not covered by v8b's fallback and is measured here: for a small number of
    pairs the scalar is zero AND ``fromto`` is left untouched, at every
    ``distmax``. Left alone that reports a phantom contact between geoms 25 cm
    apart.

    The buffer is cleared before every call so a stale segment can never be read
    as this call's answer. When both the scalar and the segment are empty, the
    two geoms' world AABBs are used only to DISPROVE the contact -- an AABB gap
    is a strict lower bound on true separation, so a positive gap proves the zero
    is spurious. Such a pair is counted and skipped; no AABB value is ever
    returned as a clearance. A pair whose AABBs do overlap may genuinely touch,
    so its zero is kept.
    """
    import mujoco

    best = float("inf")
    segment = np.zeros(6, dtype=np.float64)
    for gid in left:
        segment[:] = 0.0
        value = float(
            mujoco.mj_geomDistance(model, data, int(gid), int(right_gid), 10.0, segment)
        )
        if value == 0.0:
            span = float(np.linalg.norm(segment[3:] - segment[:3]))
            if span > 1e-9:
                value = span
            else:
                left_lo, left_hi = geom_world_aabb(model, data, int(gid))
                right_lo, right_hi = geom_world_aabb(model, data, int(right_gid))
                gap = np.maximum(
                    np.maximum(left_lo - right_hi, right_lo - left_hi), 0.0
                )
                if float(np.linalg.norm(gap)) > 0.0:
                    SPURIOUS_ZEROS["n"] += 1
                    continue
        if value < best:
            best = value
    return best


def geom_world_aabb(model, data, gid: int) -> tuple[np.ndarray, np.ndarray]:
    from run_pact_place_swept_volume_v7 import world_aabb_for_geom

    return world_aabb_for_geom(model, data, int(gid))


def group_aabb(model, data, gids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for gid in gids:
        g_lo, g_hi = geom_world_aabb(model, data, gid)
        lo = np.minimum(lo, g_lo)
        hi = np.maximum(hi, g_hi)
    return lo, hi


def aabb_lower_bound(lo: np.ndarray, hi: np.ndarray, centers: np.ndarray, half: np.ndarray) -> np.ndarray:
    """Lower bound on true distance from each candidate box to the group AABB."""
    c_lo = centers - half
    c_hi = centers + half
    gap = np.maximum(np.maximum(lo - c_hi, c_lo - hi), 0.0)
    return np.linalg.norm(gap, axis=1)


def verify_pose_equivalence(model, data, bar_gid: int, mocap_id: int,
                            probe_gids: list[int]) -> dict[str, Any]:
    """Prove the direct geom-pose write equals posing the mocap body."""
    import mujoco

    checks = []
    for center in ([0.69, 0.0, 1.20], [0.70, 0.10, 1.25], [0.66, -0.15, 1.15]):
        data.mocap_pos[mocap_id] = center
        data.mocap_quat[mocap_id] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(model, data)
        truth = true_distance(model, data, probe_gids, bar_gid)
        data.mocap_pos[mocap_id] = [0.0, 5.0, -2.0]
        mujoco.mj_forward(model, data)
        data.geom_xpos[bar_gid] = center
        data.geom_xmat[bar_gid] = np.eye(3).ravel()
        direct = true_distance(model, data, probe_gids, bar_gid)
        checks.append(
            {
                "center_m": list(map(float, center)),
                "mocap_forward_m": float(truth),
                "direct_pose_write_m": float(direct),
                "abs_difference_m": float(abs(truth - direct)),
            }
        )
    return {
        "checks": checks,
        "max_abs_difference_m": float(max(c["abs_difference_m"] for c in checks)),
        "bit_identical": all(c["abs_difference_m"] == 0.0 for c in checks),
    }


def measure_episode(job: dict[str, Any]) -> dict[str, Any]:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("MLSPACES_ASSETS_DIR", str(ROOT / "assets"))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.pop("DISPLAY", None)

    import mujoco
    from molmo_spaces.data_generation.pipeline import cleanup_episode_resources
    from run_pact_place_v6c_replay_videos import (
        _prepare_task,
        apply_recorded_qpos,
        row_directory,
    )
    from run_pact_place_swept_volume_v7 import geom_groups
    from run_pact_place_v8_baseline import (
        _active_clutter_geoms,
        _any_visible,
        _physical_geoms,
        _target_geoms,
    )
    from run_pact_place_v7_replay_videos import WRIST_CAMERA_MJCF

    role_index = int(job["role_index"])
    row = job["row"]
    result = json.loads(Path(job["result_path"]).read_text())
    trajectory = json.loads(Path(job["trajectory_path"]).read_text())
    steps = list(trajectory["steps"])
    candidates = job["candidates"]

    task = sampler = scratch = None
    try:
        task, sampler, scratch = _prepare_task(row, result["selected_seed"])
        model, data = task.env.current_model, task.env.current_data
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
        wrist_id = int(model.camera(WRIST_CAMERA_MJCF).id)

        active_side = str(result["intrusion_side"])
        spare = "pact_intrusion_right" if active_side == "left" else "pact_intrusion_left"
        spare_body = model.body(spare)
        spare_gid = next(
            g for g in range(int(model.ngeom)) if int(model.geom_bodyid[g]) == spare_body.id
        )
        protr_body = model.body("protr_l")
        protr_gid = next(
            g for g in range(int(model.ngeom)) if int(model.geom_bodyid[g]) == protr_body.id
        )

        verification = None
        if job.get("verify"):
            apply_recorded_qpos(task.env, steps[len(steps) // 2]["qpos"])
            verification = verify_pose_equivalence(
                model,
                data,
                spare_gid,
                int(model.body_mocapid[spare_body.id]),
                by_group["link6"],
            )

        # Static-scene interference: everything that is neither robot nor the two
        # bar carriers. Measured once, on the reset pose, which is when the scene
        # furniture is placed and never moves afterwards.
        apply_recorded_qpos(task.env, steps[0]["qpos"])
        static_gids = []
        for gid in range(int(model.ngeom)):
            body_id = int(model.geom_bodyid[gid])
            root = model.body(int(model.body_rootid[body_id])).name or ""
            name = model.body(body_id).name or ""
            if root.startswith("robot_0/") or name in {spare, "protr_l"}:
                continue
            if not (int(model.geom_contype[gid]) or int(model.geom_conaffinity[gid])):
                continue
            if float(data.geom_xpos[gid][2]) < -1.0:      # parked mocap slot
                continue
            static_gids.append(int(gid))

        n_cand = len(candidates)
        centers = np.asarray([c["center_m"] for c in candidates], dtype=float)
        halves = np.asarray([c["half_m"] for c in candidates], dtype=float)
        gid_for = np.asarray(
            [spare_gid if c["body_kind"] == "pact_intrusion_other" else protr_gid
             for c in candidates],
            dtype=int,
        )
        rot_for = [
            np.eye(3).ravel()
            if c["body_kind"] == "pact_intrusion_other"
            # protr_l is 0.035 x 0.035 x 0.120; rotate -90 deg about x so the
            # long axis spans y and the half-extents become (0.035, 0.120, 0.035)
            else np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=float).ravel()
            for c in candidates
        ]

        # Static interference and clutter interference, once per candidate.
        static_clearance = np.full(n_cand, np.inf)
        clutter_clearance = np.full(n_cand, np.inf)
        clutter_gids = _physical_geoms(model, _active_clutter_geoms(model, data))
        for index in range(n_cand):
            data.geom_xpos[int(gid_for[index])] = centers[index]
            data.geom_xmat[int(gid_for[index])] = rot_for[index]
            static_clearance[index] = true_distance(
                model, data, static_gids, int(gid_for[index])
            )
            if clutter_gids:
                clutter_clearance[index] = true_distance(
                    model, data, clutter_gids, int(gid_for[index])
                )

        min_by_group = np.full((n_cand, len(BODY_GROUPS)), np.inf)
        frames_lt_10 = np.zeros(n_cand, dtype=np.int64)
        frames_lt_5 = np.zeros(n_cand, dtype=np.int64)
        link_min = np.full(n_cand, np.inf)
        argmin_frame = np.zeros(n_cand, dtype=np.int64)

        for frame_index, step in enumerate(steps):
            apply_recorded_qpos(task.env, step["qpos"])
            bounds = {key: group_aabb(model, data, gids) for key, gids in by_group.items()}
            lower = np.stack(
                [aabb_lower_bound(*bounds[key], centers, halves) for key in BODY_GROUPS],
                axis=1,
            )
            active = np.flatnonzero(lower.min(axis=1) < PRUNE_CUTOFF_M)
            if active.size == 0:
                continue
            for index in active:
                gid = int(gid_for[index])
                data.geom_xpos[gid] = centers[index]
                data.geom_xmat[gid] = rot_for[index]
                frame_link_min = np.inf
                for slot, key in enumerate(BODY_GROUPS):
                    if lower[index, slot] >= PRUNE_CUTOFF_M:
                        continue
                    value = true_distance(model, data, by_group[key], gid)
                    if value < min_by_group[index, slot]:
                        min_by_group[index, slot] = value
                    if key in LINK_GROUPS and value < frame_link_min:
                        frame_link_min = value
                if frame_link_min < NEAR_M:
                    frames_lt_10[index] += 1
                if frame_link_min < VERY_NEAR_M:
                    frames_lt_5[index] += 1
                if frame_link_min < link_min[index]:
                    link_min[index] = frame_link_min
                    argmin_frame[index] = frame_index

        # Visibility of the bar from the wrist camera at each candidate's own
        # frame of minimum link clearance.
        visibility = np.zeros(n_cand, dtype=bool)
        in_cone = np.zeros(n_cand, dtype=bool)
        order = np.argsort(argmin_frame)
        current = -1
        for index in order:
            if not np.isfinite(link_min[index]):
                continue
            frame_index = int(argmin_frame[index])
            if frame_index != current:
                apply_recorded_qpos(task.env, steps[frame_index]["qpos"])
                current = frame_index
            gid = int(gid_for[index])
            # Park the carrier this candidate does not use: a stale pose left by
            # the previous candidate would occlude the ray and undercount.
            for other_gid in (spare_gid, protr_gid):
                if int(other_gid) != gid:
                    data.geom_xpos[int(other_gid)] = [0.0, 9.0, -9.0]
            data.geom_xpos[gid] = centers[index]
            data.geom_xmat[gid] = rot_for[index]
            visibility[index] = bool(_any_visible(model, data, wrist_id, [gid]))
            # Occlusion-independent: was the bar centre inside the wrist cone at all?
            origin = np.asarray(data.cam_xpos[wrist_id], dtype=float)
            forward = -np.asarray(data.cam_xmat[wrist_id], dtype=float).reshape(3, 3)[:, 2]
            delta = centers[index] - origin
            span = float(np.linalg.norm(delta))
            in_cone[index] = bool(
                span > 1e-9
                and float(np.dot(delta / span, forward))
                >= float(np.cos(np.deg2rad(float(model.cam_fovy[wrist_id])) / 2.0))
            )

        rows = []
        for index, candidate in enumerate(candidates):
            group_min = {
                key: (None if not np.isfinite(min_by_group[index, slot])
                      else float(min_by_group[index, slot]))
                for slot, key in enumerate(BODY_GROUPS)
            }
            finite = {k: v for k, v in group_min.items() if v is not None}
            closest = min(finite, key=finite.get) if finite else None
            cup_value = group_min["cup"]
            link_values = [group_min[k] for k in LINK_GROUPS if group_min[k] is not None]
            rows.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "min_clearance_by_group_m": group_min,
                    "closest_body": closest,
                    "cup_is_closest_body": bool(
                        cup_value is not None
                        and link_values
                        and cup_value < min(link_values)
                    ),
                    "min_link_clearance_m": (
                        None if not np.isfinite(link_min[index]) else float(link_min[index])
                    ),
                    "frames_link_clearance_lt_10cm": int(frames_lt_10[index]),
                    "frames_link_clearance_lt_5cm": int(frames_lt_5[index]),
                    "n_distinct_links_lt_10cm": int(
                        sum(v is not None and v < NEAR_M for k, v in group_min.items()
                            if k in LINK_GROUPS)
                    ),
                    "n_distinct_v8b_links_lt_10cm": int(
                        sum(v is not None and v < NEAR_M for k, v in group_min.items()
                            if k in V8B_LINK_GROUPS)
                    ),
                    "visibility_at_min_link_clearance": bool(visibility[index]),
                    "bar_center_in_wrist_cone_at_min": bool(in_cone[index]),
                    "phase_of_min_link_clearance": str(
                        steps[int(argmin_frame[index])].get("policy_phase") or "unknown"
                    ) if np.isfinite(link_min[index]) else None,
                    "static_scene_clearance_m": float(static_clearance[index]),
                    "clutter_clearance_m": float(clutter_clearance[index]),
                }
            )
        return {
            "role_index": role_index,
            "episode_id": row["episode_id"],
            "intrusion_side": active_side,
            "spare_body": spare,
            "n_steps": len(steps),
            "clean_success": bool(result.get("clean_success")),
            "mj_geom_distance_spurious_zeros_rejected": int(SPURIOUS_ZEROS["n"]),
            "pose_equivalence_verification": verification,
            "candidate_rows": rows,
        }
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


def main() -> int:
    from pact_place_corridor_contract import sha256_payload
    from run_pact_place_expert_screen import write_json_atomic
    from run_pact_place_v6c_replay_videos import CONFIG_PATH, row_directory

    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit-rows", type=int, default=0)
    args = parser.parse_args()

    config = json.loads(CONFIG_PATH.read_text())
    candidates = candidate_grid()
    jobs = []
    for row in config["expert_screen_rows"]:
        role_index = int(row["role_index"])
        if args.limit_rows and role_index >= args.limit_rows:
            continue
        directory = row_directory(role_index, row["episode_id"])
        jobs.append(
            {
                "role_index": role_index,
                "row": row,
                "result_path": str(directory / "result.json"),
                "trajectory_path": str(directory / "trajectory.json"),
                "candidates": candidates,
                "verify": role_index == 0,
            }
        )

    context = multiprocessing.get_context("spawn")
    episodes: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max(1, args.workers), mp_context=context
    ) as pool:
        for episode in pool.map(measure_episode, jobs):
            episodes.append(episode)
            print(f"row {episode['role_index']:02d} done ({episode['n_steps']} steps)", flush=True)
    episodes.sort(key=lambda item: item["role_index"])

    n_episodes = len(episodes)
    aggregate = []
    for slot, candidate in enumerate(candidates):
        rows = [episode["candidate_rows"][slot] for episode in episodes]
        group_mins = {
            key: [r["min_clearance_by_group_m"][key] for r in rows
                  if r["min_clearance_by_group_m"][key] is not None]
            for key in BODY_GROUPS
        }
        overall = {key: (min(values) if values else None) for key, values in group_mins.items()}
        safe_values = [overall[key] for key in CUP_SAFE_GROUPS if overall[key] is not None]
        link_episodes = [r for r in rows if r["min_link_clearance_m"] is not None]
        aggregate.append(
            {
                **candidate,
                "n_episodes": n_episodes,
                "min_clearance_by_group_m": overall,
                "min_cup_fingers_hand_clearance_m": (min(safe_values) if safe_values else None),
                "zero_predicted_contact_with_cup_fingers_hand": bool(
                    safe_values and min(safe_values) > 0.0
                ),
                "min_static_scene_clearance_m": min(r["static_scene_clearance_m"] for r in rows),
                "min_clutter_clearance_m": min(r["clutter_clearance_m"] for r in rows),
                "frames_link_clearance_lt_10cm": sum(r["frames_link_clearance_lt_10cm"] for r in rows),
                "frames_link_clearance_lt_5cm": sum(r["frames_link_clearance_lt_5cm"] for r in rows),
                "episodes_with_frames_lt_10cm": sum(
                    r["frames_link_clearance_lt_10cm"] > 0 for r in rows
                ),
                "episodes_with_frames_lt_5cm": sum(
                    r["frames_link_clearance_lt_5cm"] > 0 for r in rows
                ),
                "episodes_cup_is_closest_body": sum(r["cup_is_closest_body"] for r in rows),
                "cup_is_closest_body_fraction": float(
                    np.mean([r["cup_is_closest_body"] for r in rows])
                ),
                "episodes_nonzero_visibility_at_min": sum(
                    r["visibility_at_min_link_clearance"] for r in rows
                ),
                "nonzero_visibility_at_min_fraction": float(
                    np.mean([r["visibility_at_min_link_clearance"] for r in rows])
                ),
                "episodes_bar_center_in_wrist_cone_at_min": sum(
                    r["bar_center_in_wrist_cone_at_min"] for r in rows
                ),
                "mean_distinct_links_lt_10cm": float(
                    np.mean([r["n_distinct_links_lt_10cm"] for r in rows])
                ),
                "mean_distinct_v8b_links_lt_10cm": float(
                    np.mean([r["n_distinct_v8b_links_lt_10cm"] for r in rows])
                ),
                "min_link_clearance_m": (
                    min(r["min_link_clearance_m"] for r in link_episodes)
                    if link_episodes else None
                ),
                "max_link_penetration_m": (
                    float(max(0.0, -min(r["min_link_clearance_m"] for r in link_episodes)))
                    if link_episodes else 0.0
                ),
                "episodes_with_link_penetration": sum(
                    r["min_link_clearance_m"] is not None and r["min_link_clearance_m"] < 0.0
                    for r in rows
                ),
            }
        )

    for row in aggregate:
        reasons = []
        if row["contact_class_if_touched"] != "hazard_bar":
            reasons.append(
                "body is not pact_intrusion_*; contact would score as "
                "other_environment, not hazard_bar"
            )
        if not row["zero_predicted_contact_with_cup_fingers_hand"]:
            reasons.append(
                "predicted contact with cup, fingers or hand in the frozen replay"
            )
        if row["min_static_scene_clearance_m"] <= 0.0:
            reasons.append("intersects static scene furniture")
        if row["min_clutter_clearance_m"] <= 0.0:
            reasons.append("intersects v6c clutter")
        row["admitted"] = not reasons
        row["rejection_reasons"] = reasons

    admissible = [row for row in aggregate if row["admitted"]]
    admissible.sort(
        key=lambda row: (-row["frames_link_clearance_lt_10cm"], row["max_link_penetration_m"])
    )
    chosen = admissible[0] if admissible else None

    document = {
        "schema_version": "pact_place_v8c_c0_siting_v1",
        "role": "c0_siting_measurement_not_a_gate",
        "authorizes_gate": False,
        "authorizes_collection": False,
        "replay_only": True,
        "physics_stepped": False,
        "source_screen": "diagnostics_output/pact_place_corridor_v6c",
        "config_sha256": config["config_sha256"],
        "distance_instrument": (
            "mujoco.mj_geomDistance on collision geoms, with the v8b fromto mesh "
            "fallback; AABB used only as a conservative skip test above "
            f"{PRUNE_CUTOFF_M} m and never recorded as a clearance"
        ),
        "bar_pose_method": (
            "direct data.geom_xpos/geom_xmat write on a parked mocap collision geom "
            "after mj_forward; verified bit-identical to mocap posing + mj_forward"
        ),
        "pose_equivalence_verification": episodes[0]["pose_equivalence_verification"],
        "search_bounds": {
            "z_bottom_min_m": BAND_Z_BOTTOM,
            "z_top_max_m": BAND_Z_TOP,
            "x_min_m": BAND_X_MIN,
            "x_max_m": BAND_X_MAX,
            "cup_swept_z_max_m": CUP_SWEPT_Z_MAX,
            "enclosure_ceiling_z_m": ENCLOSURE_CEILING_Z,
            "shallowest_back_wall_x_m": SHALLOWEST_BACK_WALL_X,
        },
        "contact_attribution_note": (
            "pact_contact_audit.HAZARD_BODY_PREFIX is 'pact_intrusion_', so only a "
            "pact_intrusion_* body is scored as hazard_bar. protr_l candidates are "
            "measured for the record but route to other_environment and are excluded "
            "from admission."
        ),
        "mj_geom_distance_defect": {
            "note": (
                "mj_geomDistance can return a scalar 0.0 with an unwritten fromto "
                "for demonstrably separated geoms, at every distmax tried "
                "(10.0/1.0/0.5/0.2). v8b's fromto fallback does not cover that case "
                "and would record it as a contact. Such pairs are disproved with a "
                "strict AABB lower bound, counted, and skipped."
            ),
            "spurious_zeros_rejected_total": sum(
                int(e.get("mj_geom_distance_spurious_zeros_rejected") or 0)
                for e in episodes
            ),
        },
        "n_candidates": len(aggregate),
        "n_admissible": len(admissible),
        "candidates": _jsonable(aggregate),
        "chosen_candidate": _jsonable(chosen),
        "episodes": _jsonable(
            [{k: v for k, v in e.items() if k != "candidate_rows"} for e in episodes]
        ),
    }
    document["analysis_sha256"] = sha256_payload(document)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json_atomic(OUTPUT_DIR / "c0_siting.json", document)
    write_json_atomic(
        OUTPUT_DIR / "c0_per_episode_rows.json",
        _jsonable({"schema_version": "pact_place_v8c_c0_rows_v1", "episodes": episodes}),
    )
    print(json.dumps({"n_candidates": len(aggregate), "n_admissible": len(admissible),
                      "chosen": chosen}, indent=2, default=str)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
