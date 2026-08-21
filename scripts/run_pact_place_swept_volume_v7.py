#!/usr/bin/env python3
"""A0d: measure the v6c swept volume, skin engagement, and wrist visibility.

Replay the 24 frozen v6c trajectories with mj_forward only. Copy the v6c
renderer setup, including the scene-name guard. Do not edit the v6c renderer
or any v6c artifact.
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
ACT = ROOT / "submodules" / "act"
for search_path in (ROOT / "scripts", MOLMO, ACT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig  # noqa: E402
from molmo_spaces.configs.camera_configs import (  # noqa: E402
    _HYBRID_SKIN_SENSOR_NAMES,
    _hybrid_skin_sensor_camera_specs,
)
from pact_place_corridor_contract import (  # noqa: E402
    load_contract,
    sha256_payload,
)
from run_pact_place_v6c_replay_videos import (  # noqa: E402
    CONFIG_PATH,
    REQUIRED_SCENE_XML,
    V6C_CONFIG_SHA256,
    WRIST_CAMERA_MJCF,
    _prepare_task,
    apply_recorded_qpos,
    row_directory,
)

OUTPUT_DIR = ROOT / "diagnostics_output/pact_place_swept_volume_v7"
SHELF_TOP_Z = 0.72
TUBE_X0 = 0.58
ENCLOSURE_Y_ABS = 0.445
VOXEL_M = 0.02
INTERIOR_X = (0.58, 0.86)
INTERIOR_Y = (-0.445, 0.445)
INTERIOR_Z = (0.72, 1.42)
LINK_KEYS = (
    "link1",
    "link2",
    "link3",
    "link4",
    "link5",
    "link6",
    "link7",
    "hand",
    "left_finger",
    "right_finger",
    "cup",
)
PASSAGE_LINKS = ("link4", "link5", "link6")
TRACK_STRIDE = 5


def _sensor_config() -> dict[str, Any]:
    near_m, far_m = tuple(MlSpacesExpConfig.model_fields["viz_depth_range"].default)
    specs = _hybrid_skin_sensor_camera_specs()
    fov_deg = float(specs[0].fov) if specs else 45.0
    from surface_proximity_encoder import MAX_SURFACE_RANGE_M

    return {
        "sensor_names": list(_HYBRID_SKIN_SENSOR_NAMES),
        "n_sensors": len(_HYBRID_SKIN_SENSOR_NAMES),
        "fov_deg": fov_deg,
        "clip_near_m": float(near_m),
        "clip_far_m": float(far_m),
        "clip_range_source": "MlSpacesExpConfig.viz_depth_range",
        "engagement_model": (
            "nearest clutter AABB point in the sensor frustum "
            "(half-angle = fov/2) and Euclidean distance <= clip_far_m"
        ),
        "encoder_max_range_m": float(MAX_SURFACE_RANGE_M),
        "encoder_max_range_source": "act.surface_proximity_encoder.MAX_SURFACE_RANGE_M",
    }


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


def world_aabb_for_geom(model, data, gid: int) -> tuple[np.ndarray, np.ndarray]:
    import mujoco

    pos = np.asarray(data.geom_xpos[gid], dtype=np.float64)
    mat = np.asarray(data.geom_xmat[gid], dtype=np.float64).reshape(3, 3)
    size = np.asarray(model.geom_size[gid], dtype=np.float64)
    gtype = int(model.geom_type[gid])
    if gtype == int(mujoco.mjtGeom.mjGEOM_BOX):
        corners = np.array(
            [
                [ix * size[0], iy * size[1], iz * size[2]]
                for ix in (-1.0, 1.0)
                for iy in (-1.0, 1.0)
                for iz in (-1.0, 1.0)
            ],
            dtype=np.float64,
        )
        world = corners @ mat.T + pos
        return world.min(axis=0), world.max(axis=0)
    if gtype == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        r = float(size[0])
        return pos - r, pos + r
    if gtype in (
        int(mujoco.mjtGeom.mjGEOM_CAPSULE),
        int(mujoco.mjtGeom.mjGEOM_CYLINDER),
    ):
        radius, half_len = float(size[0]), float(size[1])
        axis = mat[:, 2]
        p1 = pos - axis * half_len
        p2 = pos + axis * half_len
        return np.minimum(p1, p2) - radius, np.maximum(p1, p2) + radius
    bound = float(model.geom_rbound[gid])
    return pos - bound, pos + bound


def occupancy_aabb_for_geom(model, data, gid: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Tight keep-out AABB. Skip inflated mesh rbounds that fill the enclosure."""
    import mujoco

    gtype = int(model.geom_type[gid])
    primitive = {
        int(mujoco.mjtGeom.mjGEOM_BOX),
        int(mujoco.mjtGeom.mjGEOM_SPHERE),
        int(mujoco.mjtGeom.mjGEOM_CAPSULE),
        int(mujoco.mjtGeom.mjGEOM_CYLINDER),
        int(mujoco.mjtGeom.mjGEOM_ELLIPSOID),
    }
    if gtype in primitive:
        return world_aabb_for_geom(model, data, gid)
    bound = float(model.geom_rbound[gid])
    if bound > 0.06:
        return None
    pos = np.asarray(data.geom_xpos[gid], dtype=np.float64)
    return pos - bound, pos + bound


def aabb_distance(a_min, a_max, b_min, b_max) -> float:
    gap = np.maximum(0.0, np.maximum(a_min - b_max, b_min - a_max))
    return float(np.linalg.norm(gap))


def closest_on_aabb(point, lo, hi) -> np.ndarray:
    return np.clip(point, lo, hi)


def union_aabb(current: dict[str, list[float]] | None, lo, hi) -> dict[str, list[float]]:
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    if current is None:
        return {"min_m": lo.tolist(), "max_m": hi.tolist()}
    return {
        "min_m": np.minimum(current["min_m"], lo).tolist(),
        "max_m": np.maximum(current["max_m"], hi).tolist(),
    }


def aabb_extent(block: dict[str, list[float]]) -> dict[str, float]:
    lo = np.asarray(block["min_m"], dtype=float)
    hi = np.asarray(block["max_m"], dtype=float)
    span = hi - lo
    return {
        "x_m": float(span[0]),
        "y_m": float(span[1]),
        "z_m": float(span[2]),
        "volume_m3": float(span[0] * span[1] * span[2]),
    }


def classify_body(name: str) -> str | None:
    if "cavity_obj_" in name:
        return "cup"
    if "gripper/left_" in name:
        return "left_finger"
    if "gripper/right_" in name:
        return "right_finger"
    if "gripper/base" in name or name.endswith("wrist_cam_body"):
        return "hand"
    for link in ("link7", "link6", "link5", "link4", "link3", "link2", "link1"):
        if f"fr3_{link}" in name or f"{link}_skin" in name or f"{link}_front_skin" in name or f"{link}_back_skin" in name:
            return link
    return None


def geom_groups(model) -> dict[str, list[int]]:
    groups = {key: [] for key in LINK_KEYS}
    for gid in range(int(model.ngeom)):
        body = model.body(int(model.geom_bodyid[gid])).name or ""
        key = classify_body(body)
        if key is not None:
            groups[key].append(int(gid))
    return groups


def clutter_geoms(model) -> list[tuple[str, int]]:
    found = []
    for gid in range(int(model.ngeom)):
        name = model.geom(gid).name or ""
        body = model.body(int(model.geom_bodyid[gid])).name or ""
        if name.startswith("pact_clutter_") or body.startswith("pact_clutter_"):
            found.append((body or name, int(gid)))
    return found


def resolve_camera_name(model, short_name: str) -> str:
    for candidate in (short_name, f"robot_0/{short_name}"):
        try:
            model.camera(candidate)
            return candidate
        except Exception:
            continue
    raise RuntimeError(f"camera {short_name!r} not in the model")


def pack_voxel(ix: int, iy: int, iz: int) -> int:
    return (ix & 0x3FF) | ((iy & 0x3FF) << 10) | ((iz & 0x3FF) << 20)


def mark_aabb_voxels(
    lo: np.ndarray,
    hi: np.ndarray,
    origin: np.ndarray,
    shape: tuple[int, int, int],
    bucket: set[int],
) -> None:
    i0 = np.floor((lo - origin) / VOXEL_M).astype(int)
    i1 = np.ceil((hi - origin) / VOXEL_M).astype(int)
    nx, ny, nz = shape
    x0, x1 = max(0, int(i0[0])), min(nx, int(i1[0]))
    y0, y1 = max(0, int(i0[1])), min(ny, int(i1[1]))
    z0, z1 = max(0, int(i0[2])), min(nz, int(i1[2]))
    for ix in range(x0, x1):
        for iy in range(y0, y1):
            for iz in range(z0, z1):
                bucket.add(pack_voxel(ix, iy, iz))


def clutter_surface_samples(lo, hi, pitch: float = 0.02) -> np.ndarray:
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    points = []
    xs = np.linspace(lo[0], hi[0], max(2, int(round((hi[0] - lo[0]) / pitch)) + 1))
    ys = np.linspace(lo[1], hi[1], max(2, int(round((hi[1] - lo[1]) / pitch)) + 1))
    zs = np.linspace(lo[2], hi[2], max(2, int(round((hi[2] - lo[2]) / pitch)) + 1))
    for x in xs:
        for y in ys:
            points.append((x, y, lo[2]))
            points.append((x, y, hi[2]))
    for x in xs:
        for z in zs:
            points.append((x, lo[1], z))
            points.append((x, hi[1], z))
    for y in ys:
        for z in zs:
            points.append((lo[0], y, z))
            points.append((hi[0], y, z))
    return np.unique(np.asarray(points, dtype=float), axis=0)


def point_in_camera_fov(
    point: np.ndarray,
    cam_pos: np.ndarray,
    cam_forward: np.ndarray,
    fov_deg: float,
    max_range: float,
) -> tuple[bool, float]:
    delta = point - cam_pos
    dist = float(np.linalg.norm(delta))
    if dist < 1e-9 or dist > max_range:
        return False, dist
    cosine = float(np.dot(delta / dist, cam_forward))
    half = np.deg2rad(fov_deg) / 2.0
    return cosine >= float(np.cos(half)), dist


def wrist_visible_fraction(model, data, samples: np.ndarray, clutter_body_ids: set[int]) -> tuple[int, int]:
    """Same mj_ray first-hit test as PactCollisionCorridorSampler._cam_visible_label."""
    import mujoco

    cam_id = int(model.camera(WRIST_CAMERA_MJCF).id)
    origin = np.asarray(data.cam_xpos[cam_id], dtype=np.float64)
    visible = 0
    geomid = np.zeros(1, dtype=np.int32)
    for point in samples:
        vec = point - origin
        dist = float(np.linalg.norm(vec))
        if dist < 1e-6:
            continue
        hit = mujoco.mj_ray(
            model,
            data,
            origin.astype(np.float64),
            (vec / dist).astype(np.float64),
            None,
            1,
            -1,
            geomid,
        )
        if hit >= 0 and geomid[0] >= 0:
            body_id = int(model.geom_bodyid[int(geomid[0])])
            if body_id in clutter_body_ids:
                visible += 1
    return visible, int(len(samples))


def measure_row(job: dict[str, Any]) -> dict[str, Any]:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("MLSPACES_ASSETS_DIR", str(ROOT / "assets"))
    os.environ.pop("DISPLAY", None)

    from molmo_spaces.data_generation.pipeline import cleanup_episode_resources

    sensor_cfg = job["sensor_config"]
    fov_deg = float(sensor_cfg["fov_deg"])
    far_m = float(sensor_cfg["clip_far_m"])
    encoder_far = float(sensor_cfg["encoder_max_range_m"])
    row = job["row"]
    result = json.loads(Path(job["result_path"]).read_text())
    trajectory = json.loads(Path(job["trajectory_path"]).read_text())
    steps = list(trajectory["steps"])
    if result.get("config_sha256") != V6C_CONFIG_SHA256:
        raise RuntimeError(f"row {job['role_index']} is not the frozen v6c screen")
    selected = result["selected_seed"]
    task = sampler = scratch = None
    try:
        task, sampler, scratch = _prepare_task(row, selected)
        env = task.env
        model, data = env.current_model, env.current_data
        groups = geom_groups(model)
        clutter = clutter_geoms(model)
        clutter_body_ids = {int(model.body(body).id) for body, _ in clutter}
        cam_names = [
            resolve_camera_name(model, name) for name in sensor_cfg["sensor_names"]
        ]
        cam_ids = [int(model.camera(name).id) for name in cam_names]
        origin = np.asarray([INTERIOR_X[0], INTERIOR_Y[0], INTERIOR_Z[0]], dtype=float)
        nx = int(np.ceil((INTERIOR_X[1] - INTERIOR_X[0]) / VOXEL_M))
        ny = int(np.ceil((INTERIOR_Y[1] - INTERIOR_Y[0]) / VOXEL_M))
        nz = int(np.ceil((INTERIOR_Z[1] - INTERIOR_Z[0]) / VOXEL_M))
        occupancy = {key: set() for key in LINK_KEYS}
        occupancy_per_geom = {key: set() for key in LINK_KEYS}
        occupancy_only = bool(job.get("occupancy_only"))
        swept = {key: None for key in LINK_KEYS}
        min_clear = {key: None for key in LINK_KEYS}
        engaged = 0
        engaged_encoder = 0
        engaged_passage = 0
        pairs = 0
        pairs_passage = 0
        vis_num = 0
        vis_den = 0
        episode_engaged_steps = []
        depth = float((getattr(task, "scene_params", {}) or {}).get("depth", 0.0))
        back_wall = TUBE_X0 + depth + 0.02
        track_cam_pos = []
        track_cam_fwd = []
        track_wrist_pos = []
        track_wrist_fwd = []
        track_link_aabb = []
        track_steps = []
        samples = None
        clutter_aabb_dump = []
        for step_i, step in enumerate(steps):
            apply_recorded_qpos(env, step["qpos"])
            clutter_aabbs = []
            for body, gid in clutter:
                lo, hi = world_aabb_for_geom(model, data, gid)
                clutter_aabbs.append((body, lo, hi))
            if samples is None:
                samples = np.concatenate(
                    [clutter_surface_samples(lo, hi) for _, lo, hi in clutter_aabbs],
                    axis=0,
                )
                clutter_aabb_dump = [
                    {
                        "body": body,
                        "min_m": lo.tolist(),
                        "max_m": hi.tolist(),
                    }
                    for body, lo, hi in clutter_aabbs
                ]
                if occupancy_only:
                    vis_n, vis_d = 0, 0
                else:
                    vis_n, vis_d = wrist_visible_fraction(
                        model, data, samples, clutter_body_ids
                    )
            vis_num += vis_n
            vis_den += vis_d
            step_engaged = 0
            step_link_aabb = np.full((len(LINK_KEYS), 6), np.nan, dtype=np.float64)
            for key in LINK_KEYS:
                gids = groups[key]
                if not gids:
                    continue
                lo = np.full(3, np.inf)
                hi = np.full(3, -np.inf)
                for gid in gids:
                    glo, ghi = world_aabb_for_geom(model, data, gid)
                    lo = np.minimum(lo, glo)
                    hi = np.maximum(hi, ghi)
                swept[key] = union_aabb(swept[key], lo, hi)
                mark_aabb_voxels(lo, hi, origin, (nx, ny, nz), occupancy[key])
                for gid in gids:
                    tight = occupancy_aabb_for_geom(model, data, gid)
                    if tight is None:
                        continue
                    glo, ghi = tight
                    mark_aabb_voxels(
                        glo, ghi, origin, (nx, ny, nz), occupancy_per_geom[key]
                    )
                step_link_aabb[LINK_KEYS.index(key)] = np.concatenate([lo, hi])
                if clutter_aabbs:
                    dist = min(
                        aabb_distance(lo, hi, clo, chi) for _, clo, chi in clutter_aabbs
                    )
                    prev = min_clear[key]
                    min_clear[key] = dist if prev is None else min(prev, dist)
            if occupancy_only:
                continue
            for cam_id, cam_name in zip(cam_ids, cam_names):
                cam_pos = np.asarray(data.cam_xpos[cam_id], dtype=np.float64)
                cam_mat = np.asarray(data.cam_xmat[cam_id], dtype=np.float64).reshape(3, 3)
                forward = -cam_mat[:, 2]
                forward = forward / np.linalg.norm(forward)
                best = None
                best_enc = None
                for _, lo, hi in clutter_aabbs:
                    point = closest_on_aabb(cam_pos, lo, hi)
                    in_fov, dist = point_in_camera_fov(
                        point, cam_pos, forward, fov_deg, far_m
                    )
                    if in_fov and (best is None or dist < best):
                        best = dist
                    in_fov_e, dist_e = point_in_camera_fov(
                        point, cam_pos, forward, fov_deg, encoder_far
                    )
                    if in_fov_e and (best_enc is None or dist_e < best_enc):
                        best_enc = dist_e
                pairs += 1
                link_of = cam_name
                is_passage = any(
                    token in cam_name
                    for token in ("link4_sensor", "link5_front", "link5_back", "link6_sensor")
                )
                if is_passage:
                    pairs_passage += 1
                if best is not None:
                    engaged += 1
                    step_engaged += 1
                    if is_passage:
                        engaged_passage += 1
                if best_enc is not None:
                    engaged_encoder += 1
            episode_engaged_steps.append(step_engaged)
            if step_i % TRACK_STRIDE == 0 or step_i == len(steps) - 1:
                cam_pos_row = []
                cam_fwd_row = []
                for cam_id in cam_ids:
                    pos = np.asarray(data.cam_xpos[cam_id], dtype=np.float64)
                    mat = np.asarray(data.cam_xmat[cam_id], dtype=np.float64).reshape(3, 3)
                    fwd = -mat[:, 2]
                    fwd = fwd / np.linalg.norm(fwd)
                    cam_pos_row.append(pos)
                    cam_fwd_row.append(fwd)
                wrist_id = int(model.camera(WRIST_CAMERA_MJCF).id)
                wpos = np.asarray(data.cam_xpos[wrist_id], dtype=np.float64)
                wmat = np.asarray(data.cam_xmat[wrist_id], dtype=np.float64).reshape(3, 3)
                wfwd = -wmat[:, 2]
                wfwd = wfwd / np.linalg.norm(wfwd)
                track_cam_pos.append(np.stack(cam_pos_row))
                track_cam_fwd.append(np.stack(cam_fwd_row))
                track_wrist_pos.append(wpos)
                track_wrist_fwd.append(wfwd)
                track_link_aabb.append(step_link_aabb.copy())
                track_steps.append(step_i)
        occupancy_counts = {key: len(vals) for key, vals in occupancy.items()}
        occupancy_per_geom_counts = {
            key: len(vals) for key, vals in occupancy_per_geom.items()
        }
        track_path = OUTPUT_DIR / "tracks" / f"row{job['role_index']:02d}.npz"
        if not occupancy_only:
            track_dir = OUTPUT_DIR / "tracks"
            track_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                track_path,
                step_index=np.asarray(track_steps, dtype=np.int32),
                cam_pos=np.stack(track_cam_pos),
                cam_fwd=np.stack(track_cam_fwd),
                wrist_pos=np.stack(track_wrist_pos),
                wrist_fwd=np.stack(track_wrist_fwd),
                link_aabb=np.stack(track_link_aabb),
                link_keys=np.asarray(LINK_KEYS),
                camera_names=np.asarray(cam_names),
            )
        payload = {
            "role_index": job["role_index"],
            "n_steps": len(steps),
            "episode_depth_m": depth,
            "episode_back_wall_x_m": back_wall,
            "swept_aabb_by_link": swept,
            "min_clearance_by_link_m": min_clear,
            "occupancy_voxel_counts": occupancy_counts,
            "occupancy_voxels": {key: sorted(vals) for key, vals in occupancy.items()},
            "occupancy_per_geom_voxel_counts": occupancy_per_geom_counts,
            "occupancy_per_geom_voxels": {
                key: sorted(vals) for key, vals in occupancy_per_geom.items()
            },
            "skin_pairs": pairs,
            "skin_engaged": engaged,
            "skin_engaged_encoder_range": engaged_encoder,
            "skin_pairs_passage": pairs_passage,
            "skin_engaged_passage": engaged_passage,
            "episode_median_engaged_sensors": (
                float(np.median(episode_engaged_steps)) if episode_engaged_steps else 0.0
            ),
            "wrist_visible_samples": vis_num,
            "wrist_total_samples": vis_den,
            "groups_n_geoms": {key: len(groups[key]) for key in LINK_KEYS},
            "clutter_n_geoms": len(clutter),
            "camera_names": cam_names,
            "clutter_aabbs": clutter_aabb_dump,
            "track_path": str(track_path.relative_to(ROOT)),
            "track_stride": TRACK_STRIDE,
            "occupancy_only": occupancy_only,
        }
        return payload
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


def merge_occupancy(rows: list[dict[str, Any]]) -> dict[str, list[int]]:
    merged = {key: set() for key in LINK_KEYS}
    for row in rows:
        for key, voxels in row["occupancy_voxels"].items():
            merged[key].update(voxels)
    return {key: sorted(vals) for key, vals in merged.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--occupancy-only",
        action="store_true",
        help="recompute per-geom occupancy without wrist rays; keep v6c baseline",
    )
    parser.add_argument(
        "--role-index",
        type=int,
        nargs="*",
        default=None,
        help="optional subset of official rows for a smoke run",
    )
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise SystemExit("workers must be in [1, 8]")
    contract = load_contract(CONFIG_PATH)
    if contract["config_sha256"] != V6C_CONFIG_SHA256:
        raise SystemExit("v6c contract hash changed")
    sensor_cfg = _sensor_config()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "tracks").mkdir(parents=True, exist_ok=True)
    jobs = []
    for row in contract["expert_screen_rows"]:
        index = int(row["role_index"])
        if args.role_index is not None and index not in args.role_index:
            continue
        directory = row_directory(index, row["episode_id"])
        jobs.append(
            {
                "role_index": index,
                "row": row,
                "result_path": str(directory / "result.json"),
                "trajectory_path": str(directory / "trajectory.json"),
                "sensor_config": sensor_cfg,
                "occupancy_only": bool(args.occupancy_only),
            }
        )
    if not jobs:
        raise SystemExit("no rows selected")
    rows: list[dict[str, Any]] = []
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=context,
        max_tasks_per_child=1,
    ) as executor:
        futures = {executor.submit(measure_row, job): job for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            job = futures[future]
            result = future.result()
            rows.append(result)
            print(
                f"row={result['role_index']:02d} steps={result['n_steps']} "
                f"engaged={result['skin_engaged']}/{result['skin_pairs']}",
                flush=True,
            )
    rows.sort(key=lambda item: item["role_index"])
    swept = {key: None for key in LINK_KEYS}
    min_clear = {key: None for key in LINK_KEYS}
    for row in rows:
        for key in LINK_KEYS:
            block = row["swept_aabb_by_link"].get(key)
            if block:
                swept[key] = union_aabb(
                    swept[key], block["min_m"], block["max_m"]
                )
            value = row["min_clearance_by_link_m"].get(key)
            if value is not None:
                prev = min_clear[key]
                min_clear[key] = value if prev is None else min(prev, value)
    union = None
    for key in LINK_KEYS:
        if swept[key] is not None:
            union = union_aabb(union, swept[key]["min_m"], swept[key]["max_m"])
    occupancy = merge_occupancy(rows)
    occupancy_per_geom = {key: set() for key in LINK_KEYS}
    for row in rows:
        for key, voxels in row.get("occupancy_per_geom_voxels", {}).items():
            occupancy_per_geom[key].update(voxels)
    occupancy_per_geom = {key: sorted(vals) for key, vals in occupancy_per_geom.items()}
    interior_volume = (
        (INTERIOR_X[1] - INTERIOR_X[0])
        * (INTERIOR_Y[1] - INTERIOR_Y[0])
        * (INTERIOR_Z[1] - INTERIOR_Z[0])
    )
    voxel_volume = VOXEL_M ** 3
    occupancy_report = {}
    occupied_any = set()
    for key in LINK_KEYS:
        n = len(occupancy[key])
        occupied_any.update(occupancy[key])
        occupancy_report[key] = {
            "n_voxels": n,
            "volume_m3": n * voxel_volume,
            "fraction_of_enclosure_interior": n * voxel_volume / interior_volume,
        }
    occupancy_union_n = len(occupied_any)
    occupied_geom = set()
    occupancy_per_geom_report = {}
    for key in LINK_KEYS:
        n = len(occupancy_per_geom[key])
        occupied_geom.update(occupancy_per_geom[key])
        occupancy_per_geom_report[key] = {
            "n_voxels": n,
            "volume_m3": n * voxel_volume,
            "fraction_of_enclosure_interior": n * voxel_volume / interior_volume,
        }
    occupancy_per_geom_union_n = len(occupied_geom)
    if union is not None:
        clipped = {
            "min_m": np.maximum(union["min_m"], [INTERIOR_X[0], INTERIOR_Y[0], INTERIOR_Z[0]]).tolist(),
            "max_m": np.minimum(union["max_m"], [INTERIOR_X[1], INTERIOR_Y[1], INTERIOR_Z[1]]).tolist(),
        }
        clipped_extent = aabb_extent(clipped)
    else:
        clipped = None
        clipped_extent = None
    total_pairs = sum(row["skin_pairs"] for row in rows)
    total_engaged = sum(row["skin_engaged"] for row in rows)
    total_enc = sum(row["skin_engaged_encoder_range"] for row in rows)
    total_pass = sum(row["skin_pairs_passage"] for row in rows)
    total_pass_e = sum(row["skin_engaged_passage"] for row in rows)
    vis_n = sum(row["wrist_visible_samples"] for row in rows)
    vis_d = sum(row["wrist_total_samples"] for row in rows)
    analysis = {
        "schema_version": "pact_place_swept_volume_v7",
        "role": "a0d_measurement_not_a_gate",
        "authorizes_collection": False,
        "replay_only": True,
        "physics_stepped": False,
        "source_screen": "diagnostics_output/pact_place_corridor_v6c",
        "config_sha256": V6C_CONFIG_SHA256,
        "required_scene_xml": REQUIRED_SCENE_XML,
        "n_episodes": len(rows),
        "sensor_config": sensor_cfg,
        "enclosure_interior_m": {
            "x_m": list(INTERIOR_X),
            "y_m": list(INTERIOR_Y),
            "z_m": list(INTERIOR_Z),
            "volume_m3": interior_volume,
        },
        "voxel_m": VOXEL_M,
        "swept_aabb_by_link": swept,
        "swept_aabb_union": union,
        "swept_aabb_union_extent": aabb_extent(union) if union else None,
        "link_occupancy_by_zone": occupancy_report,
        "link_occupancy_union": {
            "n_voxels": occupancy_union_n,
            "volume_m3": occupancy_union_n * voxel_volume,
            "fraction_of_enclosure_interior": occupancy_union_n * voxel_volume / interior_volume,
            "note": (
                "Per-link hull occupancy. Too coarse for keep-out; see "
                "link_occupancy_per_geom_union."
            ),
        },
        "link_occupancy_by_zone_per_geom": occupancy_per_geom_report,
        "link_occupancy_per_geom_union": {
            "n_voxels": occupancy_per_geom_union_n,
            "volume_m3": occupancy_per_geom_union_n * voxel_volume,
            "fraction_of_enclosure_interior": occupancy_per_geom_union_n
            * voxel_volume
            / interior_volume,
            "note": (
                "Keep-out for A0e: voxels overlapping any per-geom AABB."
            ),
        },
        "min_clearance_by_link_m": min_clear,
        "v6c_baseline": {
            "skin_engagement": total_engaged / total_pairs if total_pairs else 0.0,
            "skin_engagement_passage_link4_6": (
                total_pass_e / total_pass if total_pass else 0.0
            ),
            "skin_engagement_encoder_range": (
                total_enc / total_pairs if total_pairs else 0.0
            ),
            "per_episode_median_engaged_sensors": float(
                np.median([row["episode_median_engaged_sensors"] for row in rows])
            ),
            "per_episode_mean_engaged_sensors": float(
                np.mean([row["episode_median_engaged_sensors"] for row in rows])
            ),
            "mean_engaged_sensors_per_step": (
                total_engaged / (total_pairs / sensor_cfg["n_sensors"])
                if total_pairs
                else 0.0
            ),
            "wrist_visibility": vis_n / vis_d if vis_d else 0.0,
            "min_clearance_by_link_m": min_clear,
            "n_step_sensor_pairs": total_pairs,
            "n_engaged_pairs": total_engaged,
        },
        "free_enclosure_fraction_outside_occupancy": (
            1.0 - occupancy_union_n * voxel_volume / interior_volume
        ),
        "swept_aabb_union_clipped_to_enclosure": clipped,
        "swept_aabb_union_clipped_extent": clipped_extent,
        "aabb_union_note": (
            "swept_aabb_union spans the robot base through the enclosure; "
            "its volume exceeds the enclosure so it is not a keep-out shape. "
            "A0e uses per-geom occupancy voxels of links that enter the enclosure."
        ),
        "episode_back_wall_x_m": [row["episode_back_wall_x_m"] for row in rows],
        "shallowest_episode_back_wall_x_m": min(
            row["episode_back_wall_x_m"] for row in rows
        ),
        "rows": [
            {
                k: v
                for k, v in row.items()
                if k
                not in {"occupancy_voxels", "occupancy_per_geom_voxels"}
            }
            for row in rows
        ],
    }
    analysis["analysis_sha256"] = sha256_payload(
        {k: v for k, v in analysis.items() if k != "analysis_sha256"}
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    occupancy_path = OUTPUT_DIR / "occupancy_voxels.json"
    occupancy_path.write_text(json.dumps(_jsonable(occupancy), sort_keys=True) + "\n")
    occupancy_geom_path = OUTPUT_DIR / "occupancy_voxels_per_geom.json"
    occupancy_geom_path.write_text(
        json.dumps(_jsonable(occupancy_per_geom), sort_keys=True) + "\n"
    )
    if args.occupancy_only and (OUTPUT_DIR / "analysis.json").exists():
        existing = json.loads((OUTPUT_DIR / "analysis.json").read_text())
        existing["link_occupancy_by_zone_per_geom"] = occupancy_per_geom_report
        existing["link_occupancy_per_geom_union"] = {
            "n_voxels": occupancy_per_geom_union_n,
            "volume_m3": occupancy_per_geom_union_n * voxel_volume,
            "fraction_of_enclosure_interior": occupancy_per_geom_union_n
            * voxel_volume
            / interior_volume,
            "note": (
                "Keep-out for A0e: voxels overlapping any per-geom AABB, not the "
                "per-link hull."
            ),
        }
        existing["occupancy_voxels_per_geom_path"] = str(
            occupancy_geom_path.relative_to(ROOT)
        )
        existing["free_enclosure_fraction_outside_per_geom_occupancy"] = (
            1.0 - occupancy_per_geom_union_n * voxel_volume / interior_volume
        )
        existing["analysis_sha256"] = sha256_payload(
            {k: v for k, v in existing.items() if k != "analysis_sha256"}
        )
        (OUTPUT_DIR / "analysis.json").write_text(
            json.dumps(_jsonable(existing), indent=2, sort_keys=True) + "\n"
        )
        print(OUTPUT_DIR / "analysis.json")
        print(
            json.dumps(
                {
                    "occupancy_only": True,
                    "v6c_baseline_preserved": True,
                    "link_occupancy_per_geom_union": existing[
                        "link_occupancy_per_geom_union"
                    ],
                    "analysis_sha256": existing["analysis_sha256"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    analysis["occupancy_voxels_path"] = str(occupancy_path.relative_to(ROOT))
    analysis["occupancy_voxels_per_geom_path"] = str(
        occupancy_geom_path.relative_to(ROOT)
    )
    analysis["analysis_sha256"] = sha256_payload(
        {k: v for k, v in analysis.items() if k != "analysis_sha256"}
    )
    out = OUTPUT_DIR / "analysis.json"
    out.write_text(json.dumps(_jsonable(analysis), indent=2, sort_keys=True) + "\n")
    print(out)
    print(
        json.dumps(
            {
                "n_episodes": analysis["n_episodes"],
                "v6c_baseline": analysis["v6c_baseline"],
                "swept_aabb_union_extent": analysis["swept_aabb_union_extent"],
                "shallowest_episode_back_wall_x_m": analysis[
                    "shallowest_episode_back_wall_x_m"
                ],
                "analysis_sha256": analysis["analysis_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
