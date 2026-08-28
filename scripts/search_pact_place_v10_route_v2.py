#!/usr/bin/env python3
"""V10 route-v2: endpoint-only geometry, then exhaustive IK/clearance.

Preserves route-v1 as a valid scoped historical result. Does not overwrite
V9.9, V10 siting, or route-v1 artifacts. Does not step the environment, run
signal screening, episodes, paired screens, collection, training, evaluation,
or three-lobe search.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for search_path in (ROOT / "scripts", ROOT / "submodules" / "molmospaces"):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from pact_place_corridor_contract import PLACE_V5_SCENE_SHA256, sha256_file  # noqa: E402
from pact_place_v10_catalog import SurvivorCatalogV2  # noqa: E402
from pact_place_v10_compound_pendant_contract import (  # noqa: E402
    AUDIT_ROUTE_V2_ENDPOINT_ONLY_GEOMETRY_EVALUATIONS,
    AUDIT_ROUTE_V2_ENDPOINT_ONLY_GEOMETRY_UNIONS,
    AUDIT_ROUTE_V2_UNION_DIRECTION_EVALUATIONS,
    ENDPOINT_ONLY_PRIMITIVE,
    EXPECTED_FULL_ENVIRONMENT_SURVIVOR_COUNT,
    EXPECTED_UNIQUE_UNION_COUNT,
    MIN_DETOUR_M,
    MIN_NOMINAL_CLEARANCE_M,
    MIN_ROBUST_CLEARANCE_M,
    NOMINAL_PERTURBATION_INDEX,
    PLACE_V10_SCENE_SHA256,
    ROUTE_SCHEMA_V2,
    ROUTE_V1_PAYLOAD_SHA256,
    ROUTE_V1_SCOPED_CONCLUSION,
    ROUTE_V2_GEOMETRY_RELATIVE,
    ROUTE_V2_RELATIVE,
    SIGNAL_SCREEN_LIMIT_UNREGISTERED,
    SLAB_PADDINGS_M,
    V1_PREFILTER_CATALOG_SHA256,
    V2_CATALOG_SHA256,
    V2_SITING_PAYLOAD_SHA256,
    V99_RECONSTRUCTION_SHA256,
    V99_SCOPED_CONCLUSION,
    V99_SITING_SHA256,
    V99_SNAPSHOT_SHA256,
    empty_authorization,
    v10_implementation_hashes,
)
from pact_place_v10_exact import verify_v99_inputs  # noqa: E402
from pact_place_v10_route import (  # noqa: E402
    RouteIkCache,
    apply_constant_lane_endpoint_only,
    assign_members_across_routes,
    copy_qpos_dict,
    evaluate_environment_no_intersection,
    min_component_distance_to_probes,
    plan_lane_at_parameters_endpoint_only,
    route_ik_cache_key,
    sequential_ik_split_clearance,
    signal_screen_admission,
    stage_counts,
    stock_tcp_from_cell,
)
from pact_place_v10_runtime import establish_v10_runtime_env, write_immutable  # noqa: E402
from pact_place_v99_route import (  # noqa: E402
    densify_path,
    entry_exit_x,
    lane_inside_aperture,
    lane_y_grid,
    panel_lane_sign,
    slab_x_bounds,
    travel_sign_through_slab,
)
from search_pact_place_v10_route import (  # noqa: E402
    ProvenanceError,
    _fixture_from_union_key,
    _min_abs_detour_vectorized,
    _place_pose,
    _score_members_from_qpos,
    _seed_qpos,
    _solve_path,
    verify_siting_v2_inputs,
)

DEFAULT_OUTPUT = ROOT / Path(ROUTE_V2_RELATIVE).parent
FORBIDDEN_OUTPUTS = {
    (ROOT / "diagnostics_output/pact_place_v10_siting").resolve(),
    (ROOT / "diagnostics_output/pact_place_v10_siting_v2").resolve(),
    (ROOT / "diagnostics_output/pact_place_v99_siting").resolve(),
    (ROOT / "diagnostics_output/pact_place_v10_route").resolve(),
}


def _translation_densify(positions: np.ndarray, max_t: float = 0.005) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.float64)
    if len(positions) == 0:
        return positions
    out = [positions[0].copy()]
    for index in range(1, len(positions)):
        start, end = positions[index - 1], positions[index]
        dist = float(np.linalg.norm(end - start))
        n_pieces = max(1, int(np.ceil(dist / max_t - 1e-12)))
        for piece in range(1, n_pieces + 1):
            t = piece / n_pieces
            out.append(start + t * (end - start))
    return np.asarray(out, dtype=np.float64)


def endpoint_only_identities(
    dense_p: np.ndarray,
    dense_r: np.ndarray,
    *,
    fixture: dict[str, Any],
    panel_side: str,
    freeze_start: bool,
    freeze_final: bool,
) -> list[dict[str, float]]:
    """Return every registered geometry-feasible (padding, lane) identity."""
    lanes = lane_y_grid(panel_side)
    sign = panel_lane_sign(panel_side)
    identities: list[dict[str, float]] = []
    for padding in SLAB_PADDINGS_M:
        physical_lo, physical_hi, padded_lo, padded_hi = slab_x_bounds(fixture, padding)
        travel = travel_sign_through_slab(dense_p, padded_lo, padded_hi)
        entry_x, exit_x = entry_exit_x(fixture, padding, travel)
        planned_p, _planned_r, rewritten = apply_constant_lane_endpoint_only(
            dense_p,
            dense_r,
            lane_y=float(lanes[0]),
            entry_x=float(entry_x),
            exit_x=float(exit_x),
            freeze_start=freeze_start,
            freeze_final=freeze_final,
        )
        in_phys = (dense_p[:, 0] >= physical_lo - 1e-12) & (
            dense_p[:, 0] <= physical_hi + 1e-12
        )
        frozen_in_phys = bool(
            (freeze_final and len(in_phys) and bool(in_phys[-1]))
            or (freeze_start and len(in_phys) and bool(in_phys[0]))
        )
        if frozen_in_phys or not np.any(in_phys) or not np.any(rewritten & in_phys):
            continue
        stock_y_phys = dense_p[in_phys, 1]
        rewritten_mask = rewritten
        for lane_y in lanes:
            if (not lane_inside_aperture(lane_y)) or float(lane_y) * sign <= 0.0:
                continue
            if float(np.min(np.abs(float(lane_y) - stock_y_phys))) + 1e-12 < MIN_DETOUR_M:
                continue
            planned = dense_p.copy()
            planned[rewritten_mask, 1] = float(lane_y)
            xy = _translation_densify(planned)
            detour = _min_abs_detour_vectorized(
                xy[:, :2],
                dense_p[:, :2],
                x_lo=physical_lo,
                x_hi=physical_hi,
            )
            if not detour["meets_minimum"]:
                continue
            identities.append(
                {
                    "padding_m": float(padding),
                    "lane_y_m": float(lane_y),
                    "entry_x_m": float(entry_x),
                    "exit_x_m": float(exit_x),
                }
            )
    return identities


def run_phase_a_geometry(
    verified: dict[str, Any], cells: list[dict[str, Any]]
) -> dict[str, Any]:
    union_keys = verified["union_keys"]
    n_unions = int(verified["n_unions"])
    n_cells = len(cells)
    has = np.zeros((n_unions, n_cells, 2), dtype=bool)
    n_ids = np.zeros((n_unions, n_cells, 2), dtype=np.int32)
    print("[v10-route-v2] phase A geometry", flush=True)
    for cell_i, cell in enumerate(cells):
        panel_side = str(cell["intrusion_side"])
        for dir_i, direction in enumerate(("inbound", "outbound")):
            freeze_start = direction == "outbound"
            freeze_final = direction == "inbound"
            dense_p, dense_r = densify_path(*stock_tcp_from_cell(cell, direction))
            n_ok = 0
            for union_i, key in enumerate(union_keys):
                identities = endpoint_only_identities(
                    dense_p,
                    dense_r,
                    fixture=_fixture_from_union_key(key),
                    panel_side=panel_side,
                    freeze_start=freeze_start,
                    freeze_final=freeze_final,
                )
                n_ids[union_i, cell_i, dir_i] = len(identities)
                if identities:
                    has[union_i, cell_i, dir_i] = True
                    n_ok += 1
            print(
                f"[v10-route-v2] role {cell['role_index']} {direction} "
                f"{n_ok}/{n_unions} identities={int(n_ids[:, cell_i, dir_i].sum())}",
                flush=True,
            )
    n_eval = int(np.sum(has))
    n_union_all = int(np.sum(np.all(has, axis=(1, 2))))
    n_identities = int(np.sum(n_ids))
    n_expected_eval = n_unions * n_cells * 2
    mismatch = (
        n_eval != AUDIT_ROUTE_V2_ENDPOINT_ONLY_GEOMETRY_EVALUATIONS
        or n_union_all != AUDIT_ROUTE_V2_ENDPOINT_ONLY_GEOMETRY_UNIONS
        or n_expected_eval != AUDIT_ROUTE_V2_UNION_DIRECTION_EVALUATIONS
    )
    stop_reason = (
        "geometry_reproduction_mismatch"
        if mismatch
        else "phase_a_geometry_reproduced"
    )
    return {
        "has_geometry": has,
        "n_identities": n_ids,
        "n_evaluations_with_geometry": n_eval,
        "n_union_direction_evaluations": n_expected_eval,
        "n_unions_with_geometry_all_cells_dirs": n_union_all,
        "n_route_identities_generated": n_identities,
        "mismatch": mismatch,
        "stop_reason": stop_reason,
        "union_all_mask": np.all(has, axis=(1, 2)),
    }


def write_phase_a_artifact(
    output_root: Path,
    verified: dict[str, Any],
    geometry: dict[str, Any],
    cells: list[dict[str, Any]],
) -> str:
    packed_path = output_root / "geometry_mask.npz"
    np.savez(
        packed_path,
        schema_version=np.asarray(ROUTE_SCHEMA_V2),
        has_geometry=geometry["has_geometry"],
        n_identities=geometry["n_identities"],
        union_all_mask=geometry["union_all_mask"],
        union_keys=verified["union_keys"],
        union_inverse=verified["union_inverse"],
    )
    per_cell = []
    for cell_i, cell in enumerate(cells):
        per_cell.append(
            {
                "role_index": int(cell["role_index"]),
                "inbound_evaluations_with_geometry": int(
                    np.sum(geometry["has_geometry"][:, cell_i, 0])
                ),
                "outbound_evaluations_with_geometry": int(
                    np.sum(geometry["has_geometry"][:, cell_i, 1])
                ),
                "inbound_identities": int(np.sum(geometry["n_identities"][:, cell_i, 0])),
                "outbound_identities": int(np.sum(geometry["n_identities"][:, cell_i, 1])),
            }
        )
    document = {
        "schema_version": ROUTE_SCHEMA_V2,
        "stage": "geometry",
        **empty_authorization(),
        "rewrite_primitive": ENDPOINT_ONLY_PRIMITIVE,
        "route_v1_payload_sha256": ROUTE_V1_PAYLOAD_SHA256,
        "route_v1_scoped_conclusion": ROUTE_V1_SCOPED_CONCLUSION,
        "routing_run": False,
        "geometry_run": True,
        "physics_stepped": False,
        "episodes_run": False,
        "signal_screen_run": False,
        "three_lobe_searched": False,
        "v10_closed": False,
        "v99_closed_untouched": True,
        "v99_scoped_conclusion": V99_SCOPED_CONCLUSION,
        "reconstruction_sha256": V99_RECONSTRUCTION_SHA256,
        "snapshot_sha256": V99_SNAPSHOT_SHA256,
        "v99_siting_sha256": V99_SITING_SHA256,
        "v5_scene_xml_sha256": PLACE_V5_SCENE_SHA256,
        "v10_scene_xml_sha256": PLACE_V10_SCENE_SHA256,
        "siting_v2_sha256": V2_SITING_PAYLOAD_SHA256,
        "catalog_sha256": V2_CATALOG_SHA256,
        "environment_geoms_sha256": verified["environment_geoms_sha256"],
        "superseded_v1_catalog_sha256": V1_PREFILTER_CATALOG_SHA256,
        "implementation_sha256": v10_implementation_hashes(),
        "verified_counts": {
            "full_environment_exact_survivor_count": int(verified["n_rows"]),
            "unique_union_aabb_count": int(verified["n_unions"]),
        },
        "audit_expectations": {
            "union_direction_evaluations": AUDIT_ROUTE_V2_UNION_DIRECTION_EVALUATIONS,
            "evaluations_with_geometry": AUDIT_ROUTE_V2_ENDPOINT_ONLY_GEOMETRY_EVALUATIONS,
            "unions_with_geometry_all_cells_dirs": AUDIT_ROUTE_V2_ENDPOINT_ONLY_GEOMETRY_UNIONS,
        },
        "reproduced": {
            "union_direction_evaluations": geometry["n_union_direction_evaluations"],
            "evaluations_with_geometry": geometry["n_evaluations_with_geometry"],
            "unions_with_geometry_all_cells_dirs": geometry[
                "n_unions_with_geometry_all_cells_dirs"
            ],
            "route_identities_generated": geometry["n_route_identities_generated"],
        },
        "nominal_ik": stage_counts(not_evaluated=geometry["n_route_identities_generated"]),
        "strict_environment": stage_counts(
            not_evaluated=geometry["n_route_identities_generated"]
        ),
        "robust_routes": stage_counts(
            not_evaluated=geometry["n_route_identities_generated"]
        ),
        "corners_evaluated": 0,
        "all_eight_corners_evaluated": "not_applicable",
        "per_cell": per_cell,
        "geometry_mask": {
            "path": "geometry_mask.npz",
            "sha256": sha256_file(packed_path),
        },
        "stop_reason": geometry["stop_reason"],
        "note": (
            "Endpoint-only freeze primitive. Route-v1 remains a valid scoped "
            "historical result under contiguous-group-freeze."
        ),
    }
    return write_immutable(output_root / "geometry.json", document)


def load_phase_a_geometry(output_root: Path, verified: dict[str, Any]) -> dict[str, Any]:
    """Reuse a frozen Phase-A artifact. Do not overwrite geometry.json."""
    geometry_path = output_root / "geometry.json"
    packed_path = output_root / "geometry_mask.npz"
    if not geometry_path.is_file() or not packed_path.is_file():
        raise FileNotFoundError("Phase-A geometry artifact is missing")
    document = json.loads(geometry_path.read_text())
    packed = np.load(packed_path)
    reproduced = document.get("reproduced") or {}
    n_eval = int(reproduced.get("evaluations_with_geometry", -1))
    n_union_all = int(reproduced.get("unions_with_geometry_all_cells_dirs", -1))
    n_expected = int(reproduced.get("union_direction_evaluations", -1))
    mismatch = (
        n_eval != AUDIT_ROUTE_V2_ENDPOINT_ONLY_GEOMETRY_EVALUATIONS
        or n_union_all != AUDIT_ROUTE_V2_ENDPOINT_ONLY_GEOMETRY_UNIONS
        or n_expected != AUDIT_ROUTE_V2_UNION_DIRECTION_EVALUATIONS
        or document.get("stop_reason") != "phase_a_geometry_reproduced"
        or document.get("schema_version") != ROUTE_SCHEMA_V2
    )
    union_all_mask = np.asarray(packed["union_all_mask"], dtype=bool)
    if int(union_all_mask.shape[0]) != int(verified["n_unions"]):
        raise ProvenanceError("Phase-A union mask does not match siting-v2 unions")
    return {
        "has_geometry": np.asarray(packed["has_geometry"]),
        "n_identities": np.asarray(packed["n_identities"]),
        "n_evaluations_with_geometry": n_eval,
        "n_union_direction_evaluations": n_expected,
        "n_unions_with_geometry_all_cells_dirs": n_union_all,
        "n_route_identities_generated": int(
            reproduced.get("route_identities_generated", np.sum(packed["n_identities"]))
        ),
        "mismatch": bool(mismatch),
        "stop_reason": str(document.get("stop_reason") or "geometry_reproduction_mismatch"),
        "union_all_mask": union_all_mask,
        "artifact_sha256": str(document.get("artifact_sha256") or ""),
    }


def _search_one_cell(payload: dict[str, Any]) -> dict[str, Any]:
    establish_v10_runtime_env()
    import mujoco

    from pact_place_v10_environment import load_environment_geoms, prepare_v10_parked_task
    from pact_place_v99_exact import snapshot_jobs_from_reconstruction
    from reconstruct_pact_place_v99_baseline import (
        cleanup_task,
        collision_enabled_robot_geom_ids,
        pickup_collision_geom_ids,
    )

    role_index = int(payload["role_index"])
    shard = int(payload.get("shard", 0))
    n_shards = max(1, int(payload.get("n_shards", 1)))
    catalog = SurvivorCatalogV2(
        ROOT / "diagnostics_output/pact_place_v10_siting_v2/exact_survivors_v2.npz"
    )
    lobe_keys = np.asarray(catalog.lobe_keys, dtype=np.float64)
    union_keys = np.asarray(payload["union_keys"], dtype=np.float64)
    inverse = np.asarray(payload["union_inverse"], dtype=np.int32)
    union_all = np.asarray(payload["union_all_mask"], dtype=bool)
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
    morphology_ok = {
        "inbound": np.zeros(n_rows, dtype=bool),
        "outbound": np.zeros(n_rows, dtype=bool),
    }
    selected_padding = {
        "inbound": np.full(n_rows, np.nan, dtype=np.float64),
        "outbound": np.full(n_rows, np.nan, dtype=np.float64),
    }
    selected_lane = {
        "inbound": np.full(n_rows, np.nan, dtype=np.float64),
        "outbound": np.full(n_rows, np.nan, dtype=np.float64),
    }
    min_nominal = {
        "inbound": np.full(n_rows, np.nan, dtype=np.float64),
        "outbound": np.full(n_rows, np.nan, dtype=np.float64),
    }
    min_robust = {
        "inbound": np.full(n_rows, np.nan, dtype=np.float64),
        "outbound": np.full(n_rows, np.nan, dtype=np.float64),
    }
    stats = {
        direction: {
            "identities_generated": 0,
            "alternative_route_recoveries": 0,
            "exhausted_morphologies": 0,
            "nominal_ik": stage_counts(),
            "strict_environment": stage_counts(),
            "robust_routes": stage_counts(),
            "morphology_clearance": stage_counts(),
            "corners_evaluated": 0,
        }
        for direction in ("inbound", "outbound")
    }
    try:
        env = task.env
        robot_view = env.current_robot.robot_view
        kinematics = env.current_robot.kinematics
        model, data = env.current_model, env.current_data
        robot_ids = collision_enabled_robot_geom_ids(model)
        target_ids = pickup_collision_geom_ids(task)
        panel_side = str(cell["intrusion_side"])
        dense_stock = {}
        saved_by_dir = {}
        for direction, include_target in (("inbound", False), ("outbound", True)):
            stock_p, stock_r = stock_tcp_from_cell(cell, direction)
            dense_stock[direction] = densify_path(stock_p, stock_r)
            mask = np.asarray(
                cell["outbound_mask"] if include_target else cell["inbound_mask"]
            )
            first = int(np.flatnonzero(mask)[0])
            saved_by_dir[direction] = _seed_qpos(
                robot_view, trajectory["steps"][first]["qpos"], env
            )
        union_indices = np.flatnonzero(union_all)[shard::n_shards]
        for progress_i, union_i in enumerate(union_indices):
            fixture = _fixture_from_union_key(union_keys[union_i])
            members = np.flatnonzero(inverse == union_i)
            inbound_selected: dict[int, dict[str, Any]] | None = None
            for direction, include_target in (("inbound", False), ("outbound", True)):
                freeze_start = include_target
                freeze_final = not include_target
                dense_p, dense_r = dense_stock[direction]
                saved = saved_by_dir[direction]
                probe_ids = list(robot_ids) + (list(target_ids) if include_target else [])
                identities = endpoint_only_identities(
                    dense_p,
                    dense_r,
                    fixture=fixture,
                    panel_side=panel_side,
                    freeze_start=freeze_start,
                    freeze_final=freeze_final,
                )
                stats[direction]["identities_generated"] += len(identities)
                if direction == "outbound" and not inbound_selected:
                    stats[direction]["nominal_ik"]["not_evaluated"] += len(identities)
                    stats[direction]["strict_environment"]["not_evaluated"] += len(identities)
                    stats[direction]["robust_routes"]["not_evaluated"] += len(identities)
                    stats[direction]["exhausted_morphologies"] += len(members)
                    continue

                def evaluate_route(
                    identity,
                    remaining=None,
                    *,
                    _dir=direction,
                    _probe=probe_ids,
                    _saved=saved,
                    _union_i=int(union_i),
                    _fixture=fixture,
                    _freeze_start=freeze_start,
                    _freeze_final=freeze_final,
                    _dense_p=dense_p,
                    _dense_r=dense_r,
                    _members=members,
                ):
                    padding = float(identity["padding_m"])
                    lane_y = float(identity["lane_y_m"])
                    planned = plan_lane_at_parameters_endpoint_only(
                        _dense_p,
                        _dense_r,
                        fixture=_fixture,
                        panel_side=panel_side,
                        lane_y_m=lane_y,
                        padding_m=padding,
                        entry_x_m=float(identity["entry_x_m"]),
                        exit_x_m=float(identity["exit_x_m"]),
                        freeze_start=_freeze_start,
                        freeze_final=_freeze_final,
                    )
                    if not planned.get("accepted_geometry"):
                        return {
                            "ik_ok": False,
                            "environment_clear": False,
                            "robust_ok": False,
                            "n_corners_evaluated": 0,
                            "qpos_paths": None,
                            "qpos_sequence": None,
                        }
                    cache_key = route_ik_cache_key(
                        cell_role_index=role_index,
                        direction=_dir,
                        union_key=tuple(union_keys[_union_i].tolist()),
                        padding_m=padding,
                        lane_y_m=lane_y,
                        perturbation_index=NOMINAL_PERTURBATION_INDEX,
                    )
                    cached = cache.get(cache_key)
                    if cached is not None:
                        return cached
                    nominal = _solve_path(
                        positions=planned["planned_positions_m"],
                        rotations=planned["planned_rotations"],
                        saved_qpos=_saved,
                        robot_view=robot_view,
                        kinematics=kinematics,
                        model=model,
                        data=data,
                        probe_ids=_probe,
                        env_geoms=env_geoms,
                        min_pendant_m=MIN_NOMINAL_CLEARANCE_M,
                    )
                    report = {
                        "ik_ok": bool(nominal["ik_ok"]),
                        "environment_clear": False,
                        "robust_ok": False,
                        "n_corners_evaluated": 0,
                        "qpos_paths": None,
                    }
                    if not nominal["ik_ok"]:
                        cache.put(cache_key, report)
                        return report
                    env_ok = evaluate_environment_no_intersection(
                        nominal["environment_distances_m"]
                    )["environment_clear"]
                    report["environment_clear"] = bool(env_ok)
                    if not env_ok:
                        cache.put(cache_key, report)
                        return report
                    gate_rows = np.asarray(
                        remaining if remaining is not None else _members.tolist(),
                        dtype=np.int32,
                    )
                    nom_gate, _rob_gate = _score_members_from_qpos(
                        gate_rows,
                        lobe_keys,
                        [nominal["qpos_sequence"]],
                        set_qpos=robot_view.set_qpos_dict,
                        forward=lambda: mujoco.mj_forward(model, data),
                        model=model,
                        data=data,
                        probe_ids=_probe,
                        saved_qpos=_saved,
                    )
                    if not any(
                        bool(
                            np.isfinite(nom_gate[local_i])
                            and float(nom_gate[local_i]) + 1e-12 >= MIN_NOMINAL_CLEARANCE_M
                        )
                        for local_i in range(len(gate_rows))
                    ):
                        report["nominal_clearance_empty"] = True
                        report["qpos_sequence"] = nominal["qpos_sequence"]
                        cache.put(cache_key, report)
                        return report
                    qpos_paths = [nominal["qpos_sequence"]]
                    corners_ok = True
                    n_corners = 0
                    for corner_i, corner in enumerate(planned["perturbation_corners"]):
                        corner_plan = plan_lane_at_parameters_endpoint_only(
                            planned["stock_positions_m"],
                            planned["stock_rotations"],
                            fixture=_fixture,
                            panel_side=panel_side,
                            lane_y_m=float(corner["lane_y_m"]),
                            padding_m=padding,
                            entry_x_m=float(corner["entry_x_m"]),
                            exit_x_m=float(corner["exit_x_m"]),
                            freeze_start=_freeze_start,
                            freeze_final=_freeze_final,
                        )
                        if corner_plan["clipped"] or corner_plan["wrong_way"]:
                            corners_ok = False
                            break
                        corner_key = route_ik_cache_key(
                            cell_role_index=role_index,
                            direction=_dir,
                            union_key=tuple(union_keys[_union_i].tolist()),
                            padding_m=padding,
                            lane_y_m=float(corner["lane_y_m"]),
                            perturbation_index=int(corner_i),
                        )
                        corner_cached = cache.get(corner_key)
                        if corner_cached is not None and corner_cached.get("qpos_sequence") is not None:
                            corner_rep = corner_cached
                        else:
                            corner_rep = _solve_path(
                                positions=corner_plan["planned_positions_m"],
                                rotations=corner_plan["planned_rotations"],
                                saved_qpos=_saved,
                                robot_view=robot_view,
                                kinematics=kinematics,
                                model=model,
                                data=data,
                                probe_ids=_probe,
                                env_geoms=env_geoms,
                                min_pendant_m=MIN_ROBUST_CLEARANCE_M,
                            )
                            cache.put(
                                corner_key,
                                {
                                    "ik_ok": corner_rep["ik_ok"],
                                    "environment_clear": corner_rep["environment_clear"],
                                    "qpos_sequence": corner_rep.get("qpos_sequence"),
                                },
                            )
                        n_corners += 1
                        if not corner_rep["ik_ok"] or not corner_rep["environment_clear"]:
                            corners_ok = False
                            break
                        qpos_paths.append(corner_rep["qpos_sequence"])
                    report["n_corners_evaluated"] = n_corners
                    report["robust_ok"] = bool(corners_ok and n_corners == 8)
                    report["qpos_sequence"] = qpos_paths[0] if qpos_paths else None
                    if report["robust_ok"]:
                        report["qpos_paths"] = qpos_paths
                    cache.put(cache_key, report)
                    return report

                def score_members(remaining, report, *, _members=members, _saved=saved, _probe=probe_ids):
                    member_rows = np.asarray(remaining, dtype=np.int32)
                    nom, rob = _score_members_from_qpos(
                        member_rows,
                        lobe_keys,
                        report["qpos_paths"],
                        set_qpos=robot_view.set_qpos_dict,
                        forward=lambda: mujoco.mj_forward(model, data),
                        model=model,
                        data=data,
                        probe_ids=_probe,
                        saved_qpos=_saved,
                    )
                    out = {}
                    for local_i, member in enumerate(member_rows.tolist()):
                        ok = bool(
                            np.isfinite(nom[local_i])
                            and np.isfinite(rob[local_i])
                            and float(nom[local_i]) + 1e-12 >= MIN_NOMINAL_CLEARANCE_M
                            and float(rob[local_i]) + 1e-12 >= MIN_ROBUST_CLEARANCE_M
                        )
                        out[int(member)] = ok
                        if ok:
                            min_nominal[direction][member] = float(nom[local_i])
                            min_robust[direction][member] = float(rob[local_i])
                    return out

                assigned = assign_members_across_routes(
                    members.tolist(),
                    identities,
                    evaluate_route=evaluate_route,
                    score_members=score_members,
                )
                if direction == "inbound":
                    inbound_selected = assigned["selected"]
                for member, identity in assigned["selected"].items():
                    morphology_ok[direction][int(member)] = True
                    selected_padding[direction][int(member)] = float(identity["padding_m"])
                    selected_lane[direction][int(member)] = float(identity["lane_y_m"])
                stats[direction]["alternative_route_recoveries"] += int(
                    assigned["alternative_route_recoveries"]
                )
                stats[direction]["exhausted_morphologies"] += len(assigned["exhausted"])
                for key in ("nominal_ik", "strict_environment", "robust_routes", "morphology_clearance"):
                    for field in ("attempted", "passed", "failed", "not_evaluated", "evaluated_count"):
                        stats[direction][key][field] += int(assigned[key][field])
                stats[direction]["corners_evaluated"] += int(assigned["corners_evaluated"])
            if progress_i == 0 or (progress_i + 1) % 5 == 0 or progress_i + 1 == len(union_indices):
                print(
                    f"[v10-route-v2 role {role_index} shard {shard}] "
                    f"{progress_i + 1}/{len(union_indices)} "
                    f"in={int(np.sum(morphology_ok['inbound']))} "
                    f"out={int(np.sum(morphology_ok['outbound']))} "
                    f"ik={stats['inbound']['nominal_ik']['attempted'] + stats['outbound']['nominal_ik']['attempted']} "
                    f"ik_pass={stats['inbound']['nominal_ik']['passed'] + stats['outbound']['nominal_ik']['passed']} "
                    f"env_pass={stats['inbound']['strict_environment']['passed'] + stats['outbound']['strict_environment']['passed']} "
                    f"robust_pass={stats['inbound']['robust_routes']['passed'] + stats['outbound']['robust_routes']['passed']} "
                    f"clear_pass={stats['inbound']['morphology_clearance']['passed'] + stats['outbound']['morphology_clearance']['passed']} "
                    f"recoveries={stats['inbound']['alternative_route_recoveries'] + stats['outbound']['alternative_route_recoveries']}",
                    flush=True,
                )
        provenance = cache.provenance()
        stats["cache"] = provenance
    finally:
        cleanup_task(task, sampler, scratch)
    scratch_dir = Path(payload["scratch"]) / f"cell_{role_index}_shard_{shard}"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        scratch_dir / "morphology.npz",
        inbound_ok=morphology_ok["inbound"],
        outbound_ok=morphology_ok["outbound"],
        inbound_padding=selected_padding["inbound"],
        outbound_padding=selected_padding["outbound"],
        inbound_lane=selected_lane["inbound"],
        outbound_lane=selected_lane["outbound"],
        inbound_nominal=min_nominal["inbound"],
        outbound_nominal=min_nominal["outbound"],
        inbound_robust=min_robust["inbound"],
        outbound_robust=min_robust["outbound"],
    )
    (scratch_dir / "stats.json").write_text(json.dumps(stats, sort_keys=True) + "\n")
    return {"role_index": role_index, "shard": shard, "path": str(scratch_dir)}


def _add_stage(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    out = stage_counts()
    for key in ("attempted", "passed", "failed", "not_evaluated"):
        out[key] = int(left.get(key, 0)) + int(right.get(key, 0))
    out["evaluated_count"] = out["attempted"]
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--stage", choices=("geometry", "ik", "all"), default="all")
    args = parser.parse_args()
    establish_v10_runtime_env()
    output_root = args.output_root.resolve()
    if output_root in FORBIDDEN_OUTPUTS:
        raise RuntimeError("refusing to write into an immutable historical directory")
    output_root.mkdir(parents=True, exist_ok=True)
    verified = verify_siting_v2_inputs()
    reconstruction, snapshot, cells = verify_v99_inputs()
    if reconstruction.get("artifact_sha256") != V99_RECONSTRUCTION_SHA256:
        raise ProvenanceError("V9.9 reconstruction hash mismatch")
    if snapshot.get("artifact_sha256") != V99_SNAPSHOT_SHA256:
        raise ProvenanceError("V9.9 snapshot hash mismatch")
    route_v1 = json.loads(
        (ROOT / "diagnostics_output/pact_place_v10_route/route.json").read_text()
    )
    if route_v1.get("artifact_sha256") != ROUTE_V1_PAYLOAD_SHA256:
        raise ProvenanceError("route-v1 payload hash mismatch")
    if verified["n_rows"] != EXPECTED_FULL_ENVIRONMENT_SURVIVOR_COUNT:
        raise ProvenanceError("row count mismatch")
    if verified["n_unions"] != EXPECTED_UNIQUE_UNION_COUNT:
        raise ProvenanceError("union count mismatch")
    geometry_path = output_root / "geometry.json"
    if args.stage == "ik" or (args.stage == "all" and geometry_path.is_file()):
        geometry = load_phase_a_geometry(output_root, verified)
        geometry_sha = str(geometry["artifact_sha256"])
        print(
            json.dumps(
                {
                    "reused_phase_a": True,
                    "geometry_sha256": geometry_sha,
                    "evaluations_with_geometry": geometry["n_evaluations_with_geometry"],
                    "unions_all_cells_dirs": geometry["n_unions_with_geometry_all_cells_dirs"],
                    "identities": geometry["n_route_identities_generated"],
                    "stop_reason": geometry["stop_reason"],
                },
                indent=2,
            ),
            flush=True,
        )
    else:
        geometry = run_phase_a_geometry(verified, cells)
        if geometry_path.is_file():
            raise RuntimeError("refusing to overwrite frozen Phase-A geometry.json")
        geometry_sha = write_phase_a_artifact(output_root, verified, geometry, cells)
        print(
            json.dumps(
                {
                    "reused_phase_a": False,
                    "geometry_sha256": geometry_sha,
                    "evaluations_with_geometry": geometry["n_evaluations_with_geometry"],
                    "unions_all_cells_dirs": geometry["n_unions_with_geometry_all_cells_dirs"],
                    "identities": geometry["n_route_identities_generated"],
                    "stop_reason": geometry["stop_reason"],
                },
                indent=2,
            ),
            flush=True,
        )
    if geometry["mismatch"]:
        print(
            "Phase-A geometry did not reproduce the registered audit counts; "
            "stopping before IK.",
            flush=True,
        )
        return 1
    if args.stage == "geometry":
        return 0
    import concurrent.futures
    import multiprocessing

    scratch = output_root / "_cell_scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    n_cells = len(cells)
    requested = max(1, int(args.workers))
    n_shards = max(1, requested // max(n_cells, 1))
    payloads = [
        {
            "role_index": int(cell["role_index"]),
            "shard": shard,
            "n_shards": n_shards,
            "union_keys": verified["union_keys"],
            "union_inverse": verified["union_inverse"],
            "union_all_mask": geometry["union_all_mask"],
            "scratch": str(scratch),
        }
        for cell in cells
        for shard in range(n_shards)
    ]
    workers = max(1, min(requested, len(payloads)))
    print(
        f"[v10-route-v2] phase B IK unions={int(np.sum(geometry['union_all_mask']))} "
        f"workers={workers} shards_per_cell={n_shards}",
        flush=True,
    )
    if workers == 1:
        cell_results = [_search_one_cell(item) for item in payloads]
    else:
        ctx = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers, mp_context=ctx
        ) as pool:
            cell_results = list(pool.map(_search_one_cell, payloads))
    cell_results.sort(key=lambda item: (int(item["role_index"]), int(item.get("shard", 0))))
    n_rows = int(verified["n_rows"])
    inbound_ok = np.ones(n_rows, dtype=bool)
    outbound_ok = np.ones(n_rows, dtype=bool)
    min_nominal = np.full(n_rows, np.inf)
    min_robust = np.full(n_rows, np.inf)
    selected_padding = np.full((n_rows, len(cells), 2), np.nan)
    selected_lane = np.full((n_rows, len(cells), 2), np.nan)
    recoveries = 0
    exhausted = 0
    identities = 0
    ik = stage_counts()
    env = stage_counts()
    robust = stage_counts()
    clearance = stage_counts()
    corners = 0
    cache_hits = cache_misses = cache_entries = qpos_reuses = 0
    per_cell = []
    per_direction = {
        "inbound": {
            "nominal_ik": stage_counts(),
            "strict_environment": stage_counts(),
            "robust_routes": stage_counts(),
            "morphology_clearance": stage_counts(),
            "alternative_route_recoveries": 0,
            "exhausted_morphologies": 0,
            "identities_generated": 0,
            "corners_evaluated": 0,
        },
        "outbound": {
            "nominal_ik": stage_counts(),
            "strict_environment": stage_counts(),
            "robust_routes": stage_counts(),
            "morphology_clearance": stage_counts(),
            "alternative_route_recoveries": 0,
            "exhausted_morphologies": 0,
            "identities_generated": 0,
            "corners_evaluated": 0,
        },
    }
    by_role: dict[int, list[dict[str, Any]]] = {}
    for item in cell_results:
        by_role.setdefault(int(item["role_index"]), []).append(item)
    for cell_i, cell in enumerate(cells):
        role = int(cell["role_index"])
        cell_in = np.zeros(n_rows, dtype=bool)
        cell_out = np.zeros(n_rows, dtype=bool)
        cell_stats_acc = []
        for item in by_role.get(role, []):
            packed = np.load(Path(item["path"]) / "morphology.npz")
            stats = json.loads((Path(item["path"]) / "stats.json").read_text())
            cell_in |= packed["inbound_ok"]
            cell_out |= packed["outbound_ok"]
            cell_stats_acc.append(stats)
            for direction, axis in (("inbound", 0), ("outbound", 1)):
                nom = packed[f"{direction}_nominal"]
                rob = packed[f"{direction}_robust"]
                finite_n = np.isfinite(nom)
                min_nominal[finite_n] = np.minimum(min_nominal[finite_n], nom[finite_n])
                finite_r = np.isfinite(rob)
                min_robust[finite_r] = np.minimum(min_robust[finite_r], rob[finite_r])
                pad = packed[f"{direction}_padding"]
                lane = packed[f"{direction}_lane"]
                finite_p = np.isfinite(pad)
                selected_padding[finite_p, cell_i, axis] = pad[finite_p]
                selected_lane[np.isfinite(lane), cell_i, axis] = lane[np.isfinite(lane)]
                identities += int(stats[direction]["identities_generated"])
                recoveries += int(stats[direction]["alternative_route_recoveries"])
                exhausted += int(stats[direction]["exhausted_morphologies"])
                ik = _add_stage(ik, stats[direction]["nominal_ik"])
                env = _add_stage(env, stats[direction]["strict_environment"])
                robust = _add_stage(robust, stats[direction]["robust_routes"])
                clearance = _add_stage(clearance, stats[direction]["morphology_clearance"])
                corners += int(stats[direction]["corners_evaluated"])
                per_direction[direction]["identities_generated"] += int(
                    stats[direction]["identities_generated"]
                )
                per_direction[direction]["alternative_route_recoveries"] += int(
                    stats[direction]["alternative_route_recoveries"]
                )
                per_direction[direction]["exhausted_morphologies"] += int(
                    stats[direction]["exhausted_morphologies"]
                )
                per_direction[direction]["corners_evaluated"] += int(
                    stats[direction]["corners_evaluated"]
                )
                per_direction[direction]["nominal_ik"] = _add_stage(
                    per_direction[direction]["nominal_ik"], stats[direction]["nominal_ik"]
                )
                per_direction[direction]["strict_environment"] = _add_stage(
                    per_direction[direction]["strict_environment"],
                    stats[direction]["strict_environment"],
                )
                per_direction[direction]["robust_routes"] = _add_stage(
                    per_direction[direction]["robust_routes"],
                    stats[direction]["robust_routes"],
                )
                per_direction[direction]["morphology_clearance"] = _add_stage(
                    per_direction[direction]["morphology_clearance"],
                    stats[direction]["morphology_clearance"],
                )
            cache = stats.get("cache") or {}
            cache_hits += int(cache.get("hits", 0))
            cache_misses += int(cache.get("misses", 0))
            cache_entries += int(cache.get("entries", 0))
            qpos_reuses += int(cache.get("qpos_reuses", 0))
        inbound_ok &= cell_in
        outbound_ok &= cell_out
        per_cell.append({"role_index": role, "stats": cell_stats_acc})
    morphology_ok = inbound_ok & outbound_ok
    n_morphology = int(np.sum(morphology_ok))
    inverse = verified["union_inverse"]
    surviving_unions = sorted(
        {int(index) for index in inverse[np.flatnonzero(morphology_ok)].tolist()}
    )
    if n_morphology == 0:
        if int(np.sum(geometry["union_all_mask"])) == 0:
            stop_reason = "no_route_v2_geometry_survivor"
        else:
            stop_reason = "no_route_v2_ik_clearance_survivor"
        v10_closed = True
    else:
        stop_reason = SIGNAL_SCREEN_LIMIT_UNREGISTERED
        v10_closed = False
    mask_path = output_root / "route_morphology_mask.npz"
    np.savez(
        mask_path,
        schema_version=np.asarray(ROUTE_SCHEMA_V2),
        morphology_ok=morphology_ok,
        min_nominal_clearance_m=np.where(np.isfinite(min_nominal), min_nominal, np.nan),
        min_robust_clearance_m=np.where(np.isfinite(min_robust), min_robust, np.nan),
        selected_padding_m=selected_padding,
        selected_lane_y_m=selected_lane,
        union_keys=verified["union_keys"],
        union_inverse=inverse,
        surviving_union_indices=np.asarray(surviving_unions, dtype=np.int32),
    )
    document = {
        "schema_version": ROUTE_SCHEMA_V2,
        "stage": "route",
        **empty_authorization(),
        "rewrite_primitive": ENDPOINT_ONLY_PRIMITIVE,
        "route_v1_payload_sha256": ROUTE_V1_PAYLOAD_SHA256,
        "route_v1_scoped_conclusion": ROUTE_V1_SCOPED_CONCLUSION,
        "phase_a_geometry_sha256": geometry_sha,
        "routing_run": True,
        "geometry_run": True,
        "physics_stepped": False,
        "episodes_run": False,
        "signal_screen_run": False,
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
        "environment_geoms_sha256": verified["environment_geoms_sha256"],
        "superseded_v1_catalog_sha256": V1_PREFILTER_CATALOG_SHA256,
        "implementation_sha256": v10_implementation_hashes(),
        "phase_a": {
            "evaluations_with_geometry": geometry["n_evaluations_with_geometry"],
            "unions_with_geometry_all_cells_dirs": geometry[
                "n_unions_with_geometry_all_cells_dirs"
            ],
            "route_identities_generated": geometry["n_route_identities_generated"],
        },
        "identities_generated_phase_b": identities,
        "nominal_ik": ik,
        "strict_environment": env,
        "robust_routes": robust,
        "morphology_clearance": clearance,
        "corners_evaluated": corners,
        "all_eight_corners_evaluated": (
            "not_applicable"
            if robust["attempted"] == 0
            else bool(corners == 8 * robust["attempted"])
        ),
        "alternative_route_recoveries": recoveries,
        "exhausted_morphologies": exhausted,
        "route_surviving_union_count": len(surviving_unions),
        "route_surviving_morphology_count": n_morphology,
        "geometry_surviving_union_count": int(
            geometry["n_unions_with_geometry_all_cells_dirs"]
        ),
        "per_cell": per_cell,
        "per_direction": per_direction,
        "selected_route_identities": {
            "path": "route_morphology_mask.npz",
            "fields": [
                "selected_padding_m",
                "selected_lane_y_m",
            ],
            "shape": "n_morphologies x n_cells x {inbound, outbound}",
            "note": (
                "Each surviving morphology may keep a different padding/lane "
                "identity per cell and direction."
            ),
        },
        "cache_provenance": {
            "hits": cache_hits,
            "misses": cache_misses,
            "entries": cache_entries,
            "qpos_reuses": qpos_reuses,
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
        "signal_screen": signal_screen_admission(n_morphology),
        "stop_reason": stop_reason,
        "selected": [],
        "survivor_mask": {
            "path": "route_morphology_mask.npz",
            "sha256": sha256_file(mask_path),
            "n": n_morphology,
        },
        "note": (
            "Endpoint-only freeze amendment. Route-v1 remains valid under "
            "contiguous-group-freeze. Signal screening was not registered."
        ),
    }
    digest = write_immutable(output_root / "route.json", document)
    print(
        json.dumps(
            {
                "path": str(output_root / "route.json"),
                "artifact_sha256": digest,
                "route_surviving_union_count": len(surviving_unions),
                "route_surviving_morphology_count": n_morphology,
                "stop_reason": stop_reason,
                "signal_screen_run": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
