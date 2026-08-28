#!/usr/bin/env python3
"""Offline V10 union-clustered sequential-IK route search.

Does not overwrite V9.9 or V10 siting v1/v2 artifacts. Does not step the
environment, run episodes, paired screens, the 24-row gate, collection,
training, evaluation, or three-lobe search. Signal screening runs only against
a preregistered complete screen limit; none is registered.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for search_path in (ROOT / "scripts", ROOT / "submodules" / "molmospaces"):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from pact_place_corridor_contract import PLACE_V5_SCENE_SHA256, sha256_file  # noqa: E402
from pact_place_v10_catalog import (  # noqa: E402
    SurvivorCatalogV2,
    crossbar_keys_from_lobe_keys,
    stem_keys_from_lobe_keys,
    unique_union_count,
)
from pact_place_v10_compound_pendant_contract import (  # noqa: E402
    EXPECTED_FULL_ENVIRONMENT_SURVIVOR_COUNT,
    EXPECTED_UNIQUE_UNION_COUNT,
    MIN_NOMINAL_CLEARANCE_M,
    MIN_ROBUST_CLEARANCE_M,
    NOMINAL_PERTURBATION_INDEX,
    PLACE_V10_SCENE_SHA256,
    ROUTE_RELATIVE,
    ROUTE_SCHEMA,
    SCENE_XML_RELATIVE,
    SLAB_PADDINGS_M,
    V2_ATOMIC_SCORES_RELATIVE,
    V2_ATOMIC_SCORES_SHA256,
    V2_CATALOG_RELATIVE,
    V2_CATALOG_SHA256,
    V2_PREFILTER_INDICES_RELATIVE,
    V2_PREFILTER_INDICES_SHA256,
    V1_PREFILTER_CATALOG_SHA256,
    V2_SITING_PAYLOAD_SHA256,
    V2_SITING_RELATIVE,
    V5_SCENE_XML_RELATIVE,
    V99_RECONSTRUCTION_SHA256,
    V99_SCOPED_CONCLUSION,
    V99_SITING_SHA256,
    V99_SNAPSHOT_SHA256,
    empty_authorization,
    v10_implementation_hashes,
)
from pact_place_v10_environment import load_environment_geoms  # noqa: E402
from pact_place_v10_exact import verify_v99_inputs  # noqa: E402
from pact_place_v10_route import (  # noqa: E402
    RouteIkCache,
    cluster_two_lobe_unions,
    copy_qpos_dict,
    dump_probe_geoms,
    evaluate_environment_no_intersection,
    min_component_distance_to_probes,
    plan_lane_at_parameters,
    probes_min_environment_distance,
    route_ik_cache_key,
    sequential_ik_split_clearance,
    signal_screen_admission,
    stock_tcp_from_cell,
)
from pact_place_v10_runtime import establish_v10_runtime_env, write_immutable  # noqa: E402
from pact_place_v99_exact import snapshot_jobs_from_reconstruction  # noqa: E402
from pact_place_v99_route import (  # noqa: E402
    MAX_SEGMENT_ROTATION_RAD,
    apply_constant_lane,
    densify_path,
    entry_exit_x,
    lane_inside_aperture,
    lane_y_grid,
    panel_lane_sign,
    perturbation_corners,
    rotation_angle_rad,
    slab_x_bounds,
    travel_sign_through_slab,
)

DEFAULT_OUTPUT = ROOT / Path(ROUTE_RELATIVE).parent
FAIL_LANE = "lane_construction"
FAIL_DETOUR = "detour_clip_wrong_way"
FAIL_NOMINAL_IK = "nominal_ik"
FAIL_ENV = "strict_environment"
FAIL_ROBUST = "robust_ik_or_corner"
REASON_CODES = {
    "success": 0,
    FAIL_LANE: 1,
    FAIL_DETOUR: 2,
    FAIL_NOMINAL_IK: 3,
    FAIL_ENV: 4,
    FAIL_ROBUST: 5,
}


class ProvenanceError(RuntimeError):
    pass


def _fixture_from_union_key(key: np.ndarray) -> dict[str, Any]:
    packed = np.asarray(key, dtype=np.float64).reshape(6)
    lo, hi = packed[:3], packed[3:]
    return {
        "center_m": (0.5 * (lo + hi)).tolist(),
        "half_m": (0.5 * (hi - lo)).tolist(),
    }


def verify_siting_v2_inputs() -> dict[str, Any]:
    siting_path = ROOT / V2_SITING_RELATIVE
    catalog_path = ROOT / V2_CATALOG_RELATIVE
    siting = json.loads(siting_path.read_text())
    if siting.get("artifact_sha256") != V2_SITING_PAYLOAD_SHA256:
        raise ProvenanceError("siting-v2 payload hash mismatch")
    if sha256_file(catalog_path) != V2_CATALOG_SHA256:
        raise ProvenanceError("siting-v2 catalog hash mismatch")
    if sha256_file(ROOT / V2_ATOMIC_SCORES_RELATIVE) != V2_ATOMIC_SCORES_SHA256:
        raise ProvenanceError("atomic environment score hash mismatch")
    if sha256_file(ROOT / V2_PREFILTER_INDICES_RELATIVE) != V2_PREFILTER_INDICES_SHA256:
        raise ProvenanceError("prefilter index hash mismatch")
    if sha256_file(ROOT / V5_SCENE_XML_RELATIVE) != PLACE_V5_SCENE_SHA256:
        raise ProvenanceError("V5 scene hash mismatch")
    if sha256_file(ROOT / SCENE_XML_RELATIVE) != PLACE_V10_SCENE_SHA256:
        raise ProvenanceError("V10 scene hash mismatch")
    env_path = ROOT / "diagnostics_output/pact_place_v10_siting_v2/environment_geoms.pkl.gz"
    expected_env = (siting.get("environment_dump") or {}).get("geoms_sha256")
    env_sha = sha256_file(env_path)
    if expected_env and env_sha != expected_env:
        raise ProvenanceError("environment dump hash mismatch")
    catalog = SurvivorCatalogV2(catalog_path)
    n_rows = len(catalog)
    n_unions = unique_union_count(np.asarray(catalog.lobe_keys))
    if n_rows != EXPECTED_FULL_ENVIRONMENT_SURVIVOR_COUNT:
        raise ProvenanceError(f"catalog rows {n_rows} != {EXPECTED_FULL_ENVIRONMENT_SURVIVOR_COUNT}")
    if n_unions != EXPECTED_UNIQUE_UNION_COUNT:
        raise ProvenanceError(f"unique unions {n_unions} != {EXPECTED_UNIQUE_UNION_COUNT}")
    unique, inverse = cluster_two_lobe_unions(np.asarray(catalog.lobe_keys))
    if int(unique.shape[0]) != n_unions:
        raise ProvenanceError("union clustering disagrees with unique_union_count")
    return {
        "siting": siting,
        "catalog": catalog,
        "union_keys": unique,
        "union_inverse": inverse,
        "n_rows": n_rows,
        "n_unions": n_unions,
        "environment_geoms_sha256": env_sha,
    }


def _place_pose(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = position
    return pose


def _seed_qpos(robot_view, qpos_list, env) -> dict[str, np.ndarray]:
    from run_pact_place_v7_replay_videos import apply_recorded_qpos

    apply_recorded_qpos(env, qpos_list)
    return copy_qpos_dict(robot_view.get_qpos_dict())


def _min_abs_detour_vectorized(
    planned_xy: np.ndarray,
    stock_xy: np.ndarray,
    *,
    x_lo: float,
    x_hi: float,
) -> dict[str, Any]:
    """Same predicate as min_abs_detour_in_slab_m, without Python segment loops."""
    from pact_place_v99_pendant_contract import MIN_DETOUR_M

    planned = np.asarray(planned_xy, dtype=np.float64)
    stock = np.asarray(stock_xy, dtype=np.float64)
    in_slab = (planned[:, 0] >= x_lo - 1e-12) & (planned[:, 0] <= x_hi + 1e-12)
    xs = planned[in_slab, 0]
    ys = planned[in_slab, 1]
    if xs.size == 0 or len(stock) < 2:
        return {
            "n_samples": 0,
            "missing_stock_x": 0 if xs.size == 0 else int(xs.size),
            "min_abs_detour_m": 0.0,
            "meets_minimum": False,
        }
    x0 = stock[:-1, 0]
    x1 = stock[1:, 0]
    y0 = stock[:-1, 1]
    y1 = stock[1:, 1]
    dx = x1 - x0
    lo = np.minimum(x0, x1)
    hi = np.maximum(x0, x1)
    hit = (lo[None, :] - 1e-12 <= xs[:, None]) & (xs[:, None] <= hi[None, :] + 1e-12)
    any_hit = np.any(hit, axis=1)
    missing = int(np.sum(~any_hit))
    valid_dx = np.abs(dx) >= 1e-12
    t = np.where(
        valid_dx[None, :],
        (xs[:, None] - x0[None, :]) / np.where(valid_dx, dx, 1.0)[None, :],
        0.0,
    )
    stock_y = y0[None, :] + t * (y1 - y0)[None, :]
    delta = np.where(hit, np.abs(ys[:, None] - stock_y), np.inf)
    per_point = np.min(delta, axis=1)
    samples = per_point[any_hit]
    if samples.size == 0:
        return {
            "n_samples": 0,
            "missing_stock_x": missing,
            "min_abs_detour_m": 0.0,
            "meets_minimum": False,
        }
    minimum = float(np.min(samples))
    return {
        "n_samples": int(samples.size),
        "missing_stock_x": missing,
        "min_abs_detour_m": minimum,
        "meets_minimum": bool(minimum + 1e-12 >= MIN_DETOUR_M and missing == 0),
    }


def _densify_positions_xy(
    positions: np.ndarray,
    rotations: np.ndarray,
    *,
    max_translation_m: float = 0.005,
    max_rotation_rad: float = MAX_SEGMENT_ROTATION_RAD,
) -> np.ndarray:
    """Match densify_path sample locations without Slerp. Detour uses XY only."""
    import math

    from pact_place_v99_pendant_contract import MAX_SEGMENT_TRANSLATION_M

    del max_translation_m
    positions = np.asarray(positions, dtype=np.float64)
    rotations = np.asarray(rotations, dtype=np.float64)
    if len(positions) == 0:
        return positions
    out = [positions[0].copy()]
    for index in range(1, len(positions)):
        start_p, end_p = positions[index - 1], positions[index]
        dist = float(np.linalg.norm(end_p - start_p))
        angle = rotation_angle_rad(rotations[index - 1], rotations[index])
        n_pieces = max(
            1,
            int(math.ceil(dist / MAX_SEGMENT_TRANSLATION_M - 1e-12)),
            int(math.ceil(angle / max_rotation_rad - 1e-12)),
        )
        for piece in range(1, n_pieces + 1):
            t = piece / n_pieces
            out.append(start_p + t * (end_p - start_p))
    return np.asarray(out, dtype=np.float64)


def _geometry_plan(
    dense_stock_p: np.ndarray,
    dense_stock_r: np.ndarray,
    *,
    fixture: dict[str, Any],
    panel_side: str,
    lane_y_m: float,
    padding_m: float,
    freeze_start: bool,
    freeze_final: bool,
) -> tuple[dict[str, Any] | None, str]:
    clipped = not lane_inside_aperture(lane_y_m)
    wrong_way = float(lane_y_m) * panel_lane_sign(panel_side) <= 0.0
    if clipped or wrong_way:
        return None, FAIL_DETOUR
    physical_lo, physical_hi, padded_lo, padded_hi = slab_x_bounds(fixture, padding_m)
    travel = travel_sign_through_slab(dense_stock_p, padded_lo, padded_hi)
    entry_x, exit_x = entry_exit_x(fixture, padding_m, travel)
    planned_p, planned_r, rewritten = apply_constant_lane(
        dense_stock_p,
        dense_stock_r,
        lane_y=float(lane_y_m),
        entry_x=float(entry_x),
        exit_x=float(exit_x),
        freeze_start=freeze_start,
        freeze_final=freeze_final,
    )
    detour_coarse = _min_abs_detour_vectorized(
        planned_p[:, :2],
        dense_stock_p[:, :2],
        x_lo=physical_lo,
        x_hi=physical_hi,
    )
    # Stock is already translation-densified. Extra densify cannot introduce new
    # X samples, and Y ramps only reduce the min detour.
    if detour_coarse["n_samples"] == 0 or not detour_coarse["meets_minimum"]:
        return None, FAIL_DETOUR
    dense_xy = _densify_positions_xy(planned_p, planned_r)
    detour = _min_abs_detour_vectorized(
        dense_xy[:, :2],
        dense_stock_p[:, :2],
        x_lo=physical_lo,
        x_hi=physical_hi,
    )
    if not detour["meets_minimum"]:
        return None, FAIL_DETOUR
    planned_p, planned_r = densify_path(planned_p, planned_r)
    return {
        "lane_y_m": float(lane_y_m),
        "padding_m": float(padding_m),
        "panel_side": str(panel_side),
        "travel_sign": float(travel),
        "entry_x_m": float(entry_x),
        "exit_x_m": float(exit_x),
        "physical_x_lo_m": float(physical_lo),
        "physical_x_hi_m": float(physical_hi),
        "padded_x_lo_m": float(padded_lo),
        "padded_x_hi_m": float(padded_hi),
        "clipped": False,
        "wrong_way": False,
        "detour": detour,
        "planned_positions_m": planned_p,
        "planned_rotations": planned_r,
        "stock_positions_m": dense_stock_p,
        "stock_rotations": dense_stock_r,
        "union_fixture": fixture,
        "perturbation_corners": perturbation_corners(lane_y_m, entry_x, exit_x),
        "rewritten_samples": int(np.sum(rewritten)),
    }, ""


def _solve_path(
    *,
    positions: np.ndarray,
    rotations: np.ndarray,
    saved_qpos: dict[str, Any],
    robot_view,
    kinematics,
    model,
    data,
    probe_ids: list[int],
    env_geoms: list[dict[str, Any]],
    min_pendant_m: float,
) -> dict[str, Any]:
    import mujoco

    gripper_mg_id = robot_view.get_gripper_movegroup_ids()[0]

    def solve_ik(pose, seed):
        return kinematics.ik(
            gripper_mg_id,
            pose,
            robot_view.move_group_ids(),
            seed,
            base_pose=robot_view.base.pose,
        )

    def measure_pendant():
        return 1.0

    def measure_environment():
        dumped = dump_probe_geoms(model, data, probe_ids)
        return probes_min_environment_distance(dumped, env_geoms)

    return sequential_ik_split_clearance(
        positions,
        rotations,
        saved_qpos=saved_qpos,
        set_qpos=robot_view.set_qpos_dict,
        get_qpos=robot_view.get_qpos_dict,
        solve_ik=solve_ik,
        forward=lambda: mujoco.mj_forward(model, data),
        place_pose=_place_pose,
        measure_pendant=measure_pendant,
        measure_environment=measure_environment,
        min_pendant_m=min_pendant_m,
        abort_on_ik_failure=True,
        abort_on_environment_failure=True,
    )


def _score_members_from_qpos(
    member_rows: np.ndarray,
    lobe_keys: np.ndarray,
    qpos_paths: list[list[dict[str, Any]]],
    *,
    set_qpos,
    forward,
    model,
    data,
    probe_ids: list[int],
    saved_qpos: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    members = np.asarray(member_rows, dtype=np.int32)
    keys = np.asarray(lobe_keys[members], dtype=np.float64)
    stems = stem_keys_from_lobe_keys(keys)
    bars = crossbar_keys_from_lobe_keys(keys)
    component_keys = np.concatenate(
        [keys.reshape(len(members), -1, 6), stems, bars[:, None, :]], axis=1
    )
    unique, inverse = np.unique(
        np.round(component_keys.reshape(-1, 6), 9), axis=0, return_inverse=True
    )
    inverse = inverse.reshape(len(members), -1)
    unique_scores = np.full((len(qpos_paths), len(unique)), np.inf, dtype=np.float64)
    unique_ok = np.ones((len(qpos_paths), len(unique)), dtype=bool)
    try:
        for path_i, qpos_list in enumerate(qpos_paths):
            if not qpos_list:
                unique_ok[path_i] = False
                unique_scores[path_i] = np.nan
                continue
            for qpos in qpos_list:
                set_qpos(qpos)
                forward()
                probes = dump_probe_geoms(model, data, probe_ids)
                for comp_i, key in enumerate(unique):
                    if not unique_ok[path_i, comp_i]:
                        continue
                    distance = min_component_distance_to_probes(key[:3], key[3:], probes)
                    if distance is None:
                        unique_ok[path_i, comp_i] = False
                        unique_scores[path_i, comp_i] = np.nan
                    else:
                        unique_scores[path_i, comp_i] = min(
                            unique_scores[path_i, comp_i], float(distance)
                        )
    finally:
        set_qpos(copy_qpos_dict(saved_qpos))
        forward()
    member_nominal = np.full(len(members), np.nan, dtype=np.float64)
    member_robust = np.full(len(members), np.nan, dtype=np.float64)
    for row_i in range(len(members)):
        comps = inverse[row_i]
        if not np.all(unique_ok[0, comps]):
            continue
        member_nominal[row_i] = float(np.min(unique_scores[0, comps]))
        if unique_ok.shape[0] < 9 or not np.all(unique_ok[1:, comps]):
            continue
        member_robust[row_i] = float(np.min(unique_scores[1:, comps]))
    return member_nominal, member_robust


def _search_one_cell(payload: dict[str, Any]) -> dict[str, Any]:
    establish_v10_runtime_env()
    import mujoco

    from pact_place_v10_environment import prepare_v10_parked_task
    from reconstruct_pact_place_v99_baseline import (
        cleanup_task,
        collision_enabled_robot_geom_ids,
        pickup_collision_geom_ids,
    )

    role_index = int(payload["role_index"])
    catalog = SurvivorCatalogV2(ROOT / V2_CATALOG_RELATIVE)
    lobe_keys = np.asarray(catalog.lobe_keys, dtype=np.float64)
    union_keys = np.asarray(payload["union_keys"], dtype=np.float64)
    inverse = np.asarray(payload["union_inverse"], dtype=np.int32)
    dumped_env = load_environment_geoms(
        ROOT / "diagnostics_output/pact_place_v10_siting_v2/environment_geoms.pkl.gz"
    )
    env_cell = next(item for item in dumped_env if int(item["role_index"]) == role_index)
    env_geoms = list(env_cell["geoms"])
    reconstruction, _snapshot, cells = verify_v99_inputs()
    cell = next(item for item in cells if int(item["role_index"]) == role_index)
    jobs = snapshot_jobs_from_reconstruction(reconstruction)
    job = next(
        item
        for item in jobs
        if int(json.loads((Path(item["row_dir"]) / "result.json").read_text())["role_index"])
        == role_index
    )
    trajectory = json.loads((Path(job["row_dir"]) / "trajectory.json").read_text())
    task, sampler, scratch = prepare_v10_parked_task(
        job["manifest_row"],
        seed_u32=(job.get("selected_seed") or {}).get("seed_u32"),
    )
    cache = RouteIkCache()
    n_unions = int(union_keys.shape[0])
    n_rows = int(lobe_keys.shape[0])
    counts = {
        direction: {
            "lane_construction": 0,
            "detour_clip_wrong_way": 0,
            "nominal_ik": 0,
            "strict_environment": 0,
            "robust_ik_or_corner": 0,
            "unions_attempted": n_unions,
            "unions_with_path": 0,
            "n_corners_evaluated_on_success": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }
        for direction in ("inbound", "outbound")
    }
    morphology_nominal = {
        "inbound": np.full(n_rows, np.nan, dtype=np.float64),
        "outbound": np.full(n_rows, np.nan, dtype=np.float64),
    }
    morphology_robust = {
        "inbound": np.full(n_rows, np.nan, dtype=np.float64),
        "outbound": np.full(n_rows, np.nan, dtype=np.float64),
    }
    chosen = {
        "inbound": np.full((n_unions, 2), np.nan, dtype=np.float64),
        "outbound": np.full((n_unions, 2), np.nan, dtype=np.float64),
    }
    reasons = {
        "inbound": np.full(n_unions, REASON_CODES[FAIL_LANE], dtype=np.int8),
        "outbound": np.full(n_unions, REASON_CODES[FAIL_LANE], dtype=np.int8),
    }
    try:
        env = task.env
        robot_view = env.current_robot.robot_view
        kinematics = env.current_robot.kinematics
        model, data = env.current_model, env.current_data
        robot_ids = collision_enabled_robot_geom_ids(model)
        target_ids = pickup_collision_geom_ids(task)
        panel_side = str(cell["intrusion_side"])
        lanes = lane_y_grid(panel_side)
        dense_stock = {}
        saved_by_dir = {}
        for direction, include_target in (("inbound", False), ("outbound", True)):
            stock_p, stock_r = stock_tcp_from_cell(cell, direction)
            dense_p, dense_r = densify_path(stock_p, stock_r)
            dense_stock[direction] = (dense_p, dense_r)
            mask = np.asarray(
                cell["outbound_mask"] if include_target else cell["inbound_mask"]
            )
            first = int(np.flatnonzero(mask)[0])
            saved_by_dir[direction] = _seed_qpos(
                robot_view, trajectory["steps"][first]["qpos"], env
            )
        for union_i, union_key in enumerate(union_keys):
            fixture = _fixture_from_union_key(union_key)
            members = np.flatnonzero(inverse == union_i)
            for direction, include_target in (("inbound", False), ("outbound", True)):
                probe_ids = list(robot_ids) + (list(target_ids) if include_target else [])
                dense_p, dense_r = dense_stock[direction]
                saved = saved_by_dir[direction]
                freeze_start = include_target
                freeze_final = not include_target
                found = False
                last_reason = FAIL_LANE
                saw_geometry = False
                for padding in SLAB_PADDINGS_M:
                    if found:
                        break
                    physical_lo, physical_hi, padded_lo, padded_hi = slab_x_bounds(
                        fixture, float(padding)
                    )
                    travel = travel_sign_through_slab(dense_p, padded_lo, padded_hi)
                    entry_x, exit_x = entry_exit_x(fixture, float(padding), travel)
                    _, _, rewritten = apply_constant_lane(
                        dense_p,
                        dense_r,
                        lane_y=float(lanes[0]),
                        entry_x=float(entry_x),
                        exit_x=float(exit_x),
                        freeze_start=freeze_start,
                        freeze_final=freeze_final,
                    )
                    if not np.any(rewritten):
                        last_reason = FAIL_DETOUR
                        continue
                    for lane_y in lanes:
                        planned, geom_reason = _geometry_plan(
                            dense_p,
                            dense_r,
                            fixture=fixture,
                            panel_side=panel_side,
                            lane_y_m=float(lane_y),
                            padding_m=float(padding),
                            freeze_start=freeze_start,
                            freeze_final=freeze_final,
                        )
                        if planned is None:
                            last_reason = geom_reason
                            continue
                        saw_geometry = True
                        cache_key = route_ik_cache_key(
                            cell_role_index=role_index,
                            direction=direction,
                            union_key=tuple(union_key.tolist()),
                            padding_m=float(padding),
                            lane_y_m=float(lane_y),
                            perturbation_index=NOMINAL_PERTURBATION_INDEX,
                        )
                        cached = cache.get(cache_key)
                        if cached is not None and (
                            not cached["ik_ok"] or not cached["environment_clear"]
                        ):
                            last_reason = (
                                FAIL_NOMINAL_IK if not cached["ik_ok"] else FAIL_ENV
                            )
                            continue
                        nominal = _solve_path(
                            positions=planned["planned_positions_m"],
                            rotations=planned["planned_rotations"],
                            saved_qpos=saved,
                            robot_view=robot_view,
                            kinematics=kinematics,
                            model=model,
                            data=data,
                            probe_ids=probe_ids,
                            env_geoms=env_geoms,
                            min_pendant_m=MIN_NOMINAL_CLEARANCE_M,
                        )
                        cache.put(
                            cache_key,
                            {
                                "ik_ok": nominal["ik_ok"],
                                "environment_clear": nominal["environment_clear"],
                            },
                        )
                        if not nominal["ik_ok"]:
                            last_reason = FAIL_NOMINAL_IK
                            continue
                        env_ok = evaluate_environment_no_intersection(
                            nominal["environment_distances_m"]
                        )["environment_clear"]
                        if not env_ok:
                            last_reason = FAIL_ENV
                            continue
                        qpos_paths = [nominal["qpos_sequence"]]
                        corner_fail = False
                        for corner_i, corner in enumerate(planned["perturbation_corners"]):
                            corner_plan = plan_lane_at_parameters(
                                planned["stock_positions_m"],
                                planned["stock_rotations"],
                                fixture=planned["union_fixture"],
                                panel_side=panel_side,
                                lane_y_m=float(corner["lane_y_m"]),
                                padding_m=float(padding),
                                entry_x_m=float(corner["entry_x_m"]),
                                exit_x_m=float(corner["exit_x_m"]),
                                freeze_start=freeze_start,
                                freeze_final=freeze_final,
                            )
                            if corner_plan["clipped"] or corner_plan["wrong_way"]:
                                corner_fail = True
                                break
                            corner_key = route_ik_cache_key(
                                cell_role_index=role_index,
                                direction=direction,
                                union_key=tuple(union_key.tolist()),
                                padding_m=float(padding),
                                lane_y_m=float(corner["lane_y_m"]),
                                perturbation_index=int(corner_i),
                            )
                            corner_rep = _solve_path(
                                positions=corner_plan["planned_positions_m"],
                                rotations=corner_plan["planned_rotations"],
                                saved_qpos=saved,
                                robot_view=robot_view,
                                kinematics=kinematics,
                                model=model,
                                data=data,
                                probe_ids=probe_ids,
                                env_geoms=env_geoms,
                                min_pendant_m=MIN_ROBUST_CLEARANCE_M,
                            )
                            cache.put(
                                corner_key,
                                {
                                    "ik_ok": corner_rep["ik_ok"],
                                    "environment_clear": corner_rep["environment_clear"],
                                },
                            )
                            if not corner_rep["ik_ok"] or not corner_rep["environment_clear"]:
                                corner_fail = True
                                break
                            qpos_paths.append(corner_rep["qpos_sequence"])
                        if corner_fail or len(qpos_paths) != 9:
                            last_reason = FAIL_ROBUST
                            continue
                        found = True
                        counts[direction]["unions_with_path"] += 1
                        counts[direction]["n_corners_evaluated_on_success"] += 8
                        chosen[direction][union_i] = (float(padding), float(lane_y))
                        reasons[direction][union_i] = REASON_CODES["success"]
                        nom, rob = _score_members_from_qpos(
                            members,
                            lobe_keys,
                            qpos_paths,
                            set_qpos=robot_view.set_qpos_dict,
                            forward=lambda: mujoco.mj_forward(model, data),
                            model=model,
                            data=data,
                            probe_ids=probe_ids,
                            saved_qpos=saved,
                        )
                        morphology_nominal[direction][members] = nom
                        morphology_robust[direction][members] = rob
                        break
                if not found:
                    if last_reason == FAIL_LANE and saw_geometry:
                        last_reason = FAIL_DETOUR
                    counts[direction][last_reason] += 1
                    reasons[direction][union_i] = REASON_CODES[last_reason]
            if union_i == 0 or (union_i + 1) % 25 == 0 or union_i + 1 == n_unions:
                print(
                    f"[v10-route role {role_index}] {union_i + 1}/{n_unions} "
                    f"in={counts['inbound']['unions_with_path']} "
                    f"out={counts['outbound']['unions_with_path']}",
                    flush=True,
                )
        provenance = cache.provenance()
        counts["inbound"]["cache_hits"] = int(provenance["hits"])
        counts["inbound"]["cache_misses"] = int(provenance["misses"])
        counts["outbound"]["cache_hits"] = int(provenance["hits"])
        counts["outbound"]["cache_misses"] = int(provenance["misses"])
        counts["inbound"]["cache_entries"] = int(provenance["entries"])
        counts["outbound"]["cache_entries"] = int(provenance["entries"])
    finally:
        cleanup_task(task, sampler, scratch)
    scratch_dir = Path(payload["scratch"]) / f"cell_{role_index}"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        scratch_dir / "morphology.npz",
        inbound_nominal=morphology_nominal["inbound"],
        inbound_robust=morphology_robust["inbound"],
        outbound_nominal=morphology_nominal["outbound"],
        outbound_robust=morphology_robust["outbound"],
        inbound_chosen=chosen["inbound"],
        outbound_chosen=chosen["outbound"],
        inbound_reason=reasons["inbound"],
        outbound_reason=reasons["outbound"],
    )
    (scratch_dir / "counts.json").write_text(json.dumps(counts, sort_keys=True) + "\n")
    return {"role_index": role_index, "path": str(scratch_dir)}


def _pendant_ok(nominal: np.ndarray, robust: np.ndarray) -> np.ndarray:
    return (
        np.isfinite(nominal)
        & np.isfinite(robust)
        & (nominal + 1e-12 >= MIN_NOMINAL_CLEARANCE_M)
        & (robust + 1e-12 >= MIN_ROBUST_CLEARANCE_M)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    establish_v10_runtime_env()
    output_root = args.output_root.resolve()
    if output_root in {
        (ROOT / "diagnostics_output/pact_place_v10_siting").resolve(),
        (ROOT / "diagnostics_output/pact_place_v10_siting_v2").resolve(),
        (ROOT / "diagnostics_output/pact_place_v99_siting").resolve(),
    }:
        raise RuntimeError("refusing to write into an immutable siting directory")
    output_root.mkdir(parents=True, exist_ok=True)
    verified = verify_siting_v2_inputs()
    reconstruction, snapshot, cells = verify_v99_inputs()
    if reconstruction.get("artifact_sha256") != V99_RECONSTRUCTION_SHA256:
        raise ProvenanceError("V9.9 reconstruction hash mismatch")
    if snapshot.get("artifact_sha256") != V99_SNAPSHOT_SHA256:
        raise ProvenanceError("V9.9 snapshot hash mismatch")
    siting99 = json.loads(
        (ROOT / "diagnostics_output/pact_place_v99_siting/siting.json").read_text()
    )
    if siting99.get("artifact_sha256") != V99_SITING_SHA256:
        raise ProvenanceError("V9.9 siting hash mismatch")
    union_keys = verified["union_keys"]
    inverse = verified["union_inverse"]
    n_rows = int(verified["n_rows"])
    n_unions = int(verified["n_unions"])
    scratch = output_root / "_cell_scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    shared = scratch / "unions.npz"
    np.savez(shared, union_keys=union_keys, union_inverse=inverse)
    payloads = [
        {
            "role_index": int(cell["role_index"]),
            "union_keys": union_keys,
            "union_inverse": inverse,
            "scratch": str(scratch),
        }
        for cell in cells
    ]
    import concurrent.futures
    import multiprocessing

    print(f"[v10-route] unions={n_unions} rows={n_rows} cells={len(payloads)}", flush=True)
    workers = max(1, min(int(args.workers), len(payloads)))
    if workers == 1:
        cell_results = [_search_one_cell(item) for item in payloads]
    else:
        ctx = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers, mp_context=ctx
        ) as pool:
            cell_results = list(pool.map(_search_one_cell, payloads))
    cell_results.sort(key=lambda item: int(item["role_index"]))
    inbound_ok = np.ones(n_rows, dtype=bool)
    outbound_ok = np.ones(n_rows, dtype=bool)
    min_nominal = np.full(n_rows, np.inf, dtype=np.float64)
    min_robust = np.full(n_rows, np.inf, dtype=np.float64)
    per_cell = []
    cache_hits = 0
    cache_misses = 0
    cache_entries = 0
    unions_with_path_all = np.ones(n_unions, dtype=bool)
    eight_ok = True
    for item in cell_results:
        packed = np.load(Path(item["path"]) / "morphology.npz")
        counts = json.loads((Path(item["path"]) / "counts.json").read_text())
        per_cell.append({"role_index": int(item["role_index"]), "counts": counts})
        for direction, bucket in (("inbound", inbound_ok), ("outbound", outbound_ok)):
            nom = np.asarray(packed[f"{direction}_nominal"])
            rob = np.asarray(packed[f"{direction}_robust"])
            bucket &= _pendant_ok(nom, rob)
            finite_n = np.isfinite(nom)
            min_nominal[finite_n] = np.minimum(min_nominal[finite_n], nom[finite_n])
            finite_r = np.isfinite(rob)
            min_robust[finite_r] = np.minimum(min_robust[finite_r], rob[finite_r])
            n_success = int(counts[direction]["unions_with_path"])
            n_corners = int(counts[direction]["n_corners_evaluated_on_success"])
            eight_ok = eight_ok and n_corners == 8 * n_success
        in_path = np.isfinite(packed["inbound_chosen"][:, 0])
        out_path = np.isfinite(packed["outbound_chosen"][:, 0])
        unions_with_path_all &= in_path & out_path
        cache_hits += int(counts["inbound"]["cache_hits"])
        cache_misses += int(counts["inbound"]["cache_misses"])
        cache_entries += int(counts["inbound"].get("cache_entries", 0))
    morphology_ok = inbound_ok & outbound_ok
    n_morphology = int(np.sum(morphology_ok))
    surviving_unions = sorted(
        {int(index) for index in inverse[np.flatnonzero(morphology_ok)].tolist()}
    )
    n_union_surv = len(surviving_unions)
    rejected_panel = {
        "lane_construction": 0,
        "detour_clip_wrong_way": 0,
        "nominal_ik": 0,
        "strict_environment": 0,
        "robust_ik_or_corner": 0,
    }
    for item in per_cell:
        for direction in ("inbound", "outbound"):
            for key in rejected_panel:
                rejected_panel[key] += int(item["counts"][direction][key])
    n_union_path = int(np.sum(unions_with_path_all))
    path_rows = unions_with_path_all[inverse]
    nom_arr = np.where(np.isfinite(min_nominal), min_nominal, np.nan)
    rob_arr = np.where(np.isfinite(min_robust), min_robust, np.nan)
    rejected_nominal_pendant = int(
        np.sum(
            path_rows
            & (
                ~np.isfinite(nom_arr)
                | (nom_arr + 1e-12 < MIN_NOMINAL_CLEARANCE_M)
            )
        )
    )
    rejected_robust_pendant = int(
        np.sum(
            path_rows
            & np.isfinite(nom_arr)
            & (nom_arr + 1e-12 >= MIN_NOMINAL_CLEARANCE_M)
            & (
                ~np.isfinite(rob_arr)
                | (rob_arr + 1e-12 < MIN_ROBUST_CLEARANCE_M)
            )
        )
    )
    admission = signal_screen_admission(n_morphology)
    if n_morphology == 0:
        stop_reason = "no_two_lobe_route_survivor"
        v10_closed = True
        signal_run = False
    else:
        stop_reason = admission["stop_reason"] or "route_survivors_signal_not_run"
        v10_closed = False
        signal_run = bool(admission["signal_screen_run"])
    catalog_path = output_root / "route_morphology_mask.npz"
    np.savez(
        catalog_path,
        schema_version=np.asarray(ROUTE_SCHEMA),
        morphology_ok=morphology_ok,
        min_nominal_clearance_m=nom_arr,
        min_robust_clearance_m=rob_arr,
        union_keys=union_keys,
        union_inverse=inverse,
        surviving_union_indices=np.asarray(surviving_unions, dtype=np.int32),
        unions_with_ik_env_path=unions_with_path_all,
    )
    document = {
        "schema_version": ROUTE_SCHEMA,
        **empty_authorization(),
        "routing_run": True,
        "physics_stepped": False,
        "episodes_run": False,
        "signal_screen_run": bool(signal_run),
        "three_lobe_searched": False,
        "v10_closed": bool(v10_closed),
        "v99_closed_untouched": True,
        "v99_scoped_conclusion": V99_SCOPED_CONCLUSION,
        "reconstruction_sha256": V99_RECONSTRUCTION_SHA256,
        "snapshot_sha256": V99_SNAPSHOT_SHA256,
        "v99_siting_sha256": V99_SITING_SHA256,
        "v5_scene_xml_sha256": PLACE_V5_SCENE_SHA256,
        "v10_scene_xml_sha256": PLACE_V10_SCENE_SHA256,
        "siting_v2_sha256": V2_SITING_PAYLOAD_SHA256,
        "catalog_sha256": V2_CATALOG_SHA256,
        "atomic_scores_sha256": V2_ATOMIC_SCORES_SHA256,
        "prefilter_indices_sha256": V2_PREFILTER_INDICES_SHA256,
        "environment_geoms_sha256": verified["environment_geoms_sha256"],
        "superseded_v1_catalog_sha256": V1_PREFILTER_CATALOG_SHA256,
        "implementation_sha256": v10_implementation_hashes(),
        "verified_counts": {
            "robot_target_prefilter_count": int(
                verified["siting"]["robot_target_prefilter_count"]
            ),
            "full_environment_exact_survivor_count": n_rows,
            "unique_union_aabb_count": n_unions,
        },
        "unions_attempted": n_unions,
        "union_direction_evaluations": n_unions * len(cells) * 2,
        "unions_with_ik_env_path": n_union_path,
        "rejected_by_lane_construction": rejected_panel["lane_construction"],
        "rejected_by_detour_clipping_wrong_way": rejected_panel["detour_clip_wrong_way"],
        "rejected_by_nominal_ik": rejected_panel["nominal_ik"],
        "rejected_by_strict_environment_contact": rejected_panel["strict_environment"],
        "rejected_by_nominal_pendant_clearance": rejected_nominal_pendant,
        "rejected_by_robust_pendant_clearance": rejected_robust_pendant,
        "rejected_by_robust_ik_or_corner": rejected_panel["robust_ik_or_corner"],
        "route_surviving_union_count": n_union_surv,
        "route_surviving_morphology_count": n_morphology,
        "all_eight_corners_evaluated_for_admitted_routes": bool(eight_ok),
        "per_cell": per_cell,
        "cache_provenance": {
            "hits": cache_hits,
            "misses": cache_misses,
            "entries": cache_entries,
        },
        "minimum_nominal_witness_m": (
            None
            if not np.any(np.isfinite(min_nominal[morphology_ok]))
            else float(np.min(min_nominal[morphology_ok]))
        ),
        "minimum_robust_witness_m": (
            None
            if not np.any(np.isfinite(min_robust[morphology_ok]))
            else float(np.min(min_robust[morphology_ok]))
        ),
        "signal_screen": admission,
        "stop_reason": stop_reason,
        "selected": [],
        "note": (
            "Union-clustered sequential IK on the frozen six-cell grid. "
            "Pendant clearance used actual component boxes, not union representatives. "
            "Signal screening was not invented post-hoc. "
            "REGISTERED_COMPLETE_SIGNAL_SCREEN_LIMIT is None."
        ),
        "survivor_mask": {
            "path": "route_morphology_mask.npz",
            "sha256": sha256_file(catalog_path),
            "n": n_morphology,
        },
    }
    digest = write_immutable(output_root / "route.json", document)
    print(
        json.dumps(
            {
                "path": str(output_root / "route.json"),
                "artifact_sha256": digest,
                "route_surviving_union_count": n_union_surv,
                "route_surviving_morphology_count": n_morphology,
                "stop_reason": stop_reason,
                "signal_screen_run": bool(signal_run),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
