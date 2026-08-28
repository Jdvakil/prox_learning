#!/usr/bin/env python3
"""V10.2 Step-0 non-episode admission.

Runs before any V10.2 review episode. Writes
``diagnostics_output/pact_place_v102_preflight/preflight.json`` with named
per-cell, per-direction, per-component witnesses and keeps every downstream
authorization false. Stops V10.2 if any item fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "scripts", ROOT / "submodules" / "molmospaces"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from pact_geom_distance import (  # noqa: E402
    CONTACT_DISTANCE_M,
    geom_world_aabb,
    true_distance,
)
from pact_place_corridor_contract import sha256_file, sha256_payload  # noqa: E402
from pact_place_v10_compound_pendant_contract import (  # noqa: E402
    ALL_GEOMS,
    PLACE_V10_SCENE_SHA256,
    SCENE_XML_RELATIVE,
)
from pact_place_v10_environment import (  # noqa: E402
    load_environment_geoms,
    score_assembly_environment,
)
from pact_place_v10_exact import score_component_on_cell, verify_v99_inputs  # noqa: E402
from pact_place_v10_geometry import active_components  # noqa: E402
from pact_place_v10_route import plan_lane_endpoint_only, stock_tcp_from_cell  # noqa: E402
from pact_place_v10_runtime import establish_v10_runtime_env, write_immutable  # noqa: E402
from pact_place_v101_empirical_qualification_contract import (  # noqa: E402
    ENVIRONMENT_DUMP_RELATIVE,
    verify_protected_artifacts,
)
from pact_place_v102_raised_pendant_contract import (  # noqa: E402
    CONTRACT_VERSION,
    ENVIRONMENT_VERSION,
    MIN_DETOUR_M,
    MIN_PENDANT_CLEARANCE_M,
    PHYSICS_CLEAN_FAMILIES,
    SAMPLER_CLASS,
    SLAB_PADDING_M,
    empty_authorization,
    frozen_assembly,
    frozen_route_for_side,
    implementation_hashes,
    implementation_sha256,
    registered_assembly_expectations,
)
from pact_place_v102_geometry import raised_assembly_expectations  # noqa: E402
from pact_place_v102_route import (  # noqa: E402
    sequential_ik_component_clearance,
    speed_schedule,
    speed_schedule_sha256,
)
from pact_place_v99_exact import snapshot_jobs_from_reconstruction  # noqa: E402
from pact_place_v99_route import panel_lane_sign  # noqa: E402
from reconstruct_pact_place_v99_baseline import cleanup_task  # noqa: E402

DEFAULT_OUTPUT_ROOT = ROOT / "diagnostics_output" / "pact_place_v102_preflight"
SCENE_XML = ROOT / SCENE_XML_RELATIVE
SCHEMA_VERSION = "pact_place_v102_preflight_v1"


def _place_pose(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = position
    return pose


# --------------------------------------------------------------------------
# Item 1: protected artifacts and scene hashes
# --------------------------------------------------------------------------
def item_protected_artifacts() -> dict[str, Any]:
    observed = verify_protected_artifacts()
    scene_sha = sha256_file(SCENE_XML)
    if scene_sha != PLACE_V10_SCENE_SHA256:
        raise ValueError(f"V10 scene hash drifted: {scene_sha}")
    return {
        "passed": True,
        "n_protected_artifacts": len(observed),
        "protected_artifacts": observed,
        "scene_xml": SCENE_XML_RELATIVE,
        "scene_sha256": scene_sha,
        "v102_implementation_sha256": implementation_sha256(),
        "v102_implementation_files": implementation_hashes(),
        "note": (
            "V10.1 artifacts are verified byte-for-byte and are never rewritten "
            "with the new V10.2 implementation hash."
        ),
    }


# --------------------------------------------------------------------------
# Item 2: raised assembly against enclosure, panel, clutter, initial state
# --------------------------------------------------------------------------
def item_assembly_validation(
    assembly: dict[str, Any], cells: list[dict[str, Any]], cells_env: list[dict[str, Any]]
) -> dict[str, Any]:
    observed = raised_assembly_expectations(assembly)
    expected = registered_assembly_expectations()
    environment = score_assembly_environment(assembly, cells_env)
    initial: list[dict[str, Any]] = []
    initial_ok = True
    for component in active_components(assembly):
        for cell in cells:
            report = score_component_on_cell(component, cell)
            initial_ok = initial_ok and bool(report["initial_clear"])
            initial.append(
                {
                    "component": component["name"],
                    "geom": component["geom"],
                    "role_index": int(cell["role_index"]),
                    "family": cell.get("family"),
                    "intrusion_side": cell.get("intrusion_side"),
                    "initial_robot_and_target_clear": bool(report["initial_clear"]),
                    "initial_witness_distance_m": report["initial_witness"].get("distance_m"),
                }
            )
    per_component_env = [
        {
            "component": item["name"],
            "role": item["role"],
            "panel_clear_all_cells": bool(item["panel_clear_all"]),
            "environment_clear_all_cells": bool(item["environment_clear_all"]),
            "per_cell": [
                {
                    "role_index": cell["role_index"],
                    "intrusion_side": cell["intrusion_side"],
                    "panel_clear": bool(cell["panel_clear"]),
                    "clutter_clear": bool(cell["clutter_clear"]),
                    "static_clear": bool(cell["static_clear"]),
                    "environment_clear": bool(cell["environment_clear"]),
                    "min_distance_m": cell["min_distance_m"],
                }
                for cell in item["per_cell"]
            ],
        }
        for item in environment["per_component"]
    ]
    passed = bool(
        environment["panel_clear"] and environment["environment_clear"] and initial_ok
    )
    return {
        "passed": passed,
        "assembly_expectations_observed": observed,
        "assembly_expectations_registered": expected,
        "aperture_and_hood_enforced_by_builder": True,
        "panel_clear_all_cells": bool(environment["panel_clear"]),
        "environment_clear_all_cells": bool(environment["environment_clear"]),
        "initial_target_and_robot_clear_all_cells": bool(initial_ok),
        "per_component_environment": per_component_env,
        "per_component_initial_state": initial,
        "n_cells": len(cells),
    }


# --------------------------------------------------------------------------
# Item 3: exact stock-route necessity, all 12 cell x direction cases
# --------------------------------------------------------------------------
def item_necessity(assembly: dict[str, Any], cells: list[dict[str, Any]]) -> dict[str, Any]:
    components = active_components(assembly)
    per_case: list[dict[str, Any]] = []
    obstructed = 0
    for cell in cells:
        scored = {
            component["name"]: score_component_on_cell(component, cell)
            for component in components
        }
        for direction in ("inbound", "outbound"):
            key = f"{direction}_contact"
            blockers = sorted(
                name for name, report in scored.items() if bool(report[key])
            )
            case = {
                "role_index": int(cell["role_index"]),
                "family": cell.get("family"),
                "intrusion_side": cell.get("intrusion_side"),
                "direction": direction,
                "obstructed": bool(blockers),
                "blocking_components": blockers,
                "per_component_witness_distance_m": {
                    name: report[f"{direction}_witness"].get("distance_m")
                    for name, report in scored.items()
                },
            }
            per_case.append(case)
            if blockers:
                obstructed += 1
    return {
        "passed": bool(obstructed == 12),
        "obstructed_cases": obstructed,
        "required_cases": 12,
        "counts_lobe_stem_and_crossbar": True,
        "per_case": per_case,
    }


# --------------------------------------------------------------------------
# Item 4: fixed endpoint-only route geometry on all twelve cases
# --------------------------------------------------------------------------
def _plan_case(cell: dict[str, Any], direction: str, assembly: dict[str, Any]) -> dict[str, Any]:
    side = str(cell["intrusion_side"])
    route = frozen_route_for_side(side)
    positions, rotations = stock_tcp_from_cell(cell, direction)
    planned = plan_lane_endpoint_only(
        positions,
        rotations,
        assembly=assembly,
        panel_side=side,
        lane_y_m=float(route["inbound_lane_y_m"]),
        padding_m=float(SLAB_PADDING_M),
        freeze_start=direction == "outbound",
        freeze_final=direction == "inbound",
    )
    return planned


def item_route_geometry(
    assembly: dict[str, Any], cells: list[dict[str, Any]]
) -> tuple[dict[str, Any], dict[tuple[int, str], dict[str, Any]]]:
    per_case: list[dict[str, Any]] = []
    plans: dict[tuple[int, str], dict[str, Any]] = {}
    admitted = 0
    for cell in cells:
        side = str(cell["intrusion_side"])
        lane = float(frozen_route_for_side(side)["inbound_lane_y_m"])
        for direction in ("inbound", "outbound"):
            planned = _plan_case(cell, direction, assembly)
            plans[(int(cell["role_index"]), direction)] = planned
            steps = planned.get("path_steps") or {}
            side_ok = bool(np.sign(lane) == panel_lane_sign(side))
            case = {
                "role_index": int(cell["role_index"]),
                "family": cell.get("family"),
                "intrusion_side": side,
                "direction": direction,
                "lane_y_m": lane,
                "padding_m": float(SLAB_PADDING_M),
                "correct_side": side_ok,
                "clipped": bool(planned["clipped"]),
                "wrong_way": bool(planned["wrong_way"]),
                "min_abs_detour_m": float(planned["detour"]["min_abs_detour_m"]),
                "detour_meets_minimum": bool(planned["detour"]["meets_minimum"]),
                "min_detour_required_m": float(MIN_DETOUR_M),
                "frozen_endpoint_preserved": bool(
                    planned["frozen_endpoints"]["preserved"]
                ),
                "continuous_after_densify": bool(planned["continuous_after_densify"]),
                "max_segment_translation_m": steps.get("max_translation_m"),
                "max_segment_rotation_deg": steps.get("max_rotation_deg"),
                "n_waypoints": int(len(planned["planned_positions_m"])),
            }
            case["admitted"] = bool(
                side_ok
                and not case["clipped"]
                and not case["wrong_way"]
                and case["detour_meets_minimum"]
                and case["frozen_endpoint_preserved"]
                and case["continuous_after_densify"]
            )
            per_case.append(case)
            admitted += int(case["admitted"])
    return (
        {
            "passed": bool(admitted == 12),
            "admitted_cases": admitted,
            "required_cases": 12,
            "rewrite_primitive": "endpoint_only",
            "per_case": per_case,
        },
        plans,
    )


# --------------------------------------------------------------------------
# Items 5-7: live sequential IK, per-component clearance, contact parity
# --------------------------------------------------------------------------
def _scene_state_digest(model, data) -> str:
    payload = {
        "geom_pos": [
            [float(value) for value in model.geom_pos[int(model.geom(name).id)]]
            for name in ALL_GEOMS
        ],
        "geom_size": [
            [float(value) for value in model.geom_size[int(model.geom(name).id)]]
            for name in ALL_GEOMS
        ],
        "geom_contype": [
            int(model.geom_contype[int(model.geom(name).id)]) for name in ALL_GEOMS
        ],
        "geom_conaffinity": [
            int(model.geom_conaffinity[int(model.geom(name).id)]) for name in ALL_GEOMS
        ],
        "mocap_pos": [[float(v) for v in row] for row in np.asarray(data.mocap_pos)],
        "qpos": [float(value) for value in np.asarray(data.qpos)],
    }
    return sha256_payload(payload)


def _robot_collision_geom_ids(model) -> list[int]:
    ids: list[int] = []
    for geom_id in range(int(model.ngeom)):
        body = str(model.body(int(model.geom_bodyid[geom_id])).name or "")
        if not body.startswith("robot_0/"):
            continue
        if int(model.geom_contype[geom_id]) == 0 and int(
            model.geom_conaffinity[geom_id]
        ) == 0:
            continue
        ids.append(int(geom_id))
    return ids


def _target_collision_geom_ids(task) -> list[int]:
    from reconstruct_pact_place_v99_baseline import pickup_collision_geom_ids

    return [int(value) for value in pickup_collision_geom_ids(task)]


def _pick_parity_probe_geom(model, robot_ids: list[int]) -> int:
    for token in ("link5", "link6", "link4"):
        for geom_id in robot_ids:
            name = str(model.geom(int(geom_id)).name or "")
            body = str(model.body(int(model.geom_bodyid[int(geom_id)])).name or "")
            if token in name or token in body:
                return int(geom_id)
    return int(robot_ids[0])


def _contact_parity_fixture(task, assembly: dict[str, Any]) -> dict[str, Any]:
    """Pose each collision stem across a retained robot geom and require contact."""
    import mujoco
    from molmo_spaces.tasks.pact_place_contact_audit import classify_contact

    model, data = task.env.current_model, task.env.current_data
    mujoco.mj_forward(model, data)
    before = _scene_state_digest(model, data)
    robot_ids = _robot_collision_geom_ids(model)
    probe_gid = _pick_parity_probe_geom(model, robot_ids)
    probe_name = str(model.geom(probe_gid).name or f"geom_{probe_gid}")
    probe_body = str(model.body(int(model.geom_bodyid[probe_gid])).name or "")
    cases: list[dict[str, Any]] = []
    stems = [
        item
        for item in active_components(assembly)
        if item["role"] == "stem"
    ]
    for stem in stems:
        stem_gid = int(model.geom(str(stem["geom"])).id)
        saved_pos = np.asarray(model.geom_pos[stem_gid], dtype=float).copy()
        try:
            # Cross the probe geom rather than sitting exactly on its centre:
            # concentric box/box is degenerate in MuJoCo and yields zero
            # contacts. The offset is a quarter of the probe's own world
            # half-extent along its longest axis, so the stem centre stays
            # inside the probe for boxes, capsules, and meshes alike.
            probe_center = np.asarray(data.geom_xpos[probe_gid], dtype=float).copy()
            probe_lo, probe_hi = geom_world_aabb(model, data, probe_gid)
            probe_half_world = 0.5 * (np.asarray(probe_hi) - np.asarray(probe_lo))
            axis = int(np.argmax(probe_half_world))
            offset = np.zeros(3, dtype=float)
            offset[axis] = 0.25 * float(max(probe_half_world[axis], 0.004))
            target = probe_center + offset
            model.geom_pos[stem_gid] = target
            mujoco.mj_forward(model, data)
            overlap_distance = float(
                true_distance(model, data, [probe_gid], [stem_gid])
            )
            raw_pairs = []
            classified = []
            for index in range(int(data.ncon)):
                contact = data.contact[index]
                geom1, geom2 = int(contact.geom1), int(contact.geom2)
                if stem_gid not in (geom1, geom2):
                    continue
                record = {
                    "geom1": model.geom(geom1).name or f"geom_{geom1}",
                    "geom2": model.geom(geom2).name or f"geom_{geom2}",
                    "body1": model.body(int(model.geom_bodyid[geom1])).name or "",
                    "body2": model.body(int(model.geom_bodyid[geom2])).name or "",
                    "root1": model.body(
                        int(model.body_rootid[int(model.geom_bodyid[geom1])])
                    ).name
                    or "",
                    "root2": model.body(
                        int(model.body_rootid[int(model.geom_bodyid[geom2])])
                    ).name
                    or "",
                    "distance_m": float(contact.dist),
                }
                raw_pairs.append(record)
                classified.append(classify_contact(record))
            contype = int(model.geom_contype[stem_gid])
            conaffinity = int(model.geom_conaffinity[stem_gid])
            probe_contype = int(model.geom_contype[probe_gid])
            probe_conaffinity = int(model.geom_conaffinity[probe_gid])
            compatible = bool(
                (contype & probe_conaffinity) or (probe_contype & conaffinity)
            )
            cases.append(
                {
                    "stem": stem["name"],
                    "stem_geom": stem["geom"],
                    "probe_geom": probe_name,
                    "probe_body": probe_body,
                    "posed_at_m": [float(value) for value in target],
                    "probe_center_m": [float(value) for value in probe_center],
                    "cross_offset_m": [float(value) for value in offset],
                    "posed_overlap_distance_m": overlap_distance,
                    "geometrically_overlapping": bool(
                        overlap_distance <= CONTACT_DISTANCE_M
                    ),
                    "n_raw_contacts": len(raw_pairs),
                    "observed_by_data_contact": bool(raw_pairs),
                    "classified_mounted_fixture": bool(
                        "mounted_fixture" in classified
                    ),
                    "contact_classes": sorted(set(classified)),
                    "witness_pairs": raw_pairs[:4],
                    "stem_contype": contype,
                    "stem_conaffinity": conaffinity,
                    "probe_contype": probe_contype,
                    "probe_conaffinity": probe_conaffinity,
                    "collision_compatible_robot_pair": compatible,
                }
            )
        finally:
            model.geom_pos[stem_gid] = saved_pos
            mujoco.mj_forward(model, data)
    after = _scene_state_digest(model, data)
    component_flags = []
    for item in active_components(assembly):
        geom_id = int(model.geom(str(item["geom"])).id)
        contype = int(model.geom_contype[geom_id])
        conaffinity = int(model.geom_conaffinity[geom_id])
        compatible = any(
            (contype & int(model.geom_conaffinity[gid]))
            or (int(model.geom_contype[gid]) & conaffinity)
            for gid in robot_ids
        )
        component_flags.append(
            {
                "component": item["name"],
                "geom": item["geom"],
                "contype": contype,
                "conaffinity": conaffinity,
                "contype_is_8": contype == 8,
                "conaffinity_is_15": conaffinity == 15,
                "collision_compatible_robot_pair": bool(compatible),
            }
        )
    passed = bool(
        cases
        and all(
            case["geometrically_overlapping"]
            and case["observed_by_data_contact"]
            and case["classified_mounted_fixture"]
            and case["collision_compatible_robot_pair"]
            for case in cases
        )
        and all(
            flag["contype_is_8"]
            and flag["conaffinity_is_15"]
            and flag["collision_compatible_robot_pair"]
            for flag in component_flags
        )
        and before == after
    )
    return {
        "passed": passed,
        "scene_state_sha256_before": before,
        "scene_state_sha256_after": after,
        "scene_state_restored": bool(before == after),
        "stem_cases": cases,
        "component_collision_flags": component_flags,
    }


def _live_cell(
    job: dict[str, Any],
    cell: dict[str, Any],
    assembly: dict[str, Any],
    plans: dict[tuple[int, str], dict[str, Any]],
) -> dict[str, Any]:
    import mujoco
    from molmo_spaces.env.abstract_sensors import SensorSuite  # noqa: F401
    from pact_place_v10_environment import _prepare_with_scene
    from run_pact_place_v7_replay_videos import apply_recorded_qpos

    side = str(cell["intrusion_side"])
    row = dict(job["manifest_row"])
    row["sampler_class"] = SAMPLER_CLASS
    row["pact_v10_pendant_parked"] = False
    row["pact_v10_pendant_assembly"] = assembly
    row["pact_v10_route"] = frozen_route_for_side(side)
    trajectory = json.loads((Path(job["row_dir"]) / "trajectory.json").read_text())
    first_qpos = trajectory["steps"][0]["qpos"]
    task = sampler = scratch = None
    try:
        task, sampler, scratch = _prepare_with_scene(
            row,
            seed_u32=(job.get("selected_seed") or {}).get("seed_u32"),
            scene_xml=SCENE_XML,
            sampler_class=SAMPLER_CLASS,
        )
        apply_recorded_qpos(task.env, first_qpos)
        model, data = task.env.current_model, task.env.current_data
        mujoco.mj_forward(model, data)
        robot_view = task.env.current_robot.robot_view
        kinematics = task.env.current_robot.kinematics
        gripper_mg_id = robot_view.get_gripper_movegroup_ids()[0]
        robot_ids = _robot_collision_geom_ids(model)
        target_ids = _target_collision_geom_ids(task)
        components = [
            (str(item["name"]), int(model.geom(str(item["geom"])).id))
            for item in active_components(assembly)
        ]
        component_names = [name for name, _gid in components]

        def solve_ik(pose, seed):
            return kinematics.ik(
                gripper_mg_id,
                pose,
                robot_view.move_group_ids(),
                seed,
                base_pose=robot_view.base.pose,
            )

        directions: dict[str, Any] = {}
        for direction in ("inbound", "outbound"):
            planned = plans[(int(cell["role_index"]), direction)]
            probe_ids = list(robot_ids) + (
                list(target_ids) if direction == "outbound" else []
            )

            def measure() -> dict[str, float | None]:
                out: dict[str, float | None] = {}
                for name, gid in components:
                    value = float(true_distance(model, data, probe_ids, [gid]))
                    out[name] = None if not np.isfinite(value) else value
                return out

            saved = {
                key: np.asarray(value, dtype=float).copy()
                for key, value in robot_view.get_qpos_dict().items()
            }
            report = sequential_ik_component_clearance(
                planned["planned_positions_m"],
                planned["planned_rotations"],
                saved_qpos=saved,
                set_qpos=robot_view.set_qpos_dict,
                get_qpos=robot_view.get_qpos_dict,
                solve_ik=solve_ik,
                forward=lambda: mujoco.mj_forward(model, data),
                place_pose=_place_pose,
                component_names=component_names,
                measure_components=measure,
            )
            per_component = report["per_component_min_clearance_m"]
            below = sorted(
                name
                for name, value in per_component.items()
                if value is None or float(value) < MIN_PENDANT_CLEARANCE_M - 1e-12
            )
            worst_index = None
            worst_value = None
            for item in report["per_waypoint"]:
                values = [
                    float(value)
                    for value in item["clearance_m"].values()
                    if value is not None
                ]
                if not values:
                    continue
                local = min(values)
                if worst_value is None or local < worst_value:
                    worst_value = local
                    worst_index = int(item["waypoint_index"])
            directions[direction] = {
                "waypoints_attempted": int(report["waypoints_attempted"]),
                "waypoints_solved": int(report["waypoints_solved"]),
                "complete_sequential_ik": bool(report["complete_sequential_ik"]),
                "ik_failure_indices": report["ik_failure_indices"][:16],
                "includes_target": bool(direction == "outbound"),
                "per_component_min_clearance_m": per_component,
                "min_clearance_m": report["min_clearance_m"],
                "min_clearance_floor_m": float(MIN_PENDANT_CLEARANCE_M),
                "components_below_floor": below,
                "meets_clearance_floor": bool(not below),
                "worst_waypoint_index": worst_index,
                "worst_waypoint_clearance_m": worst_value,
                "qpos_restored": bool(report["qpos_restored"]),
                "offline_strict_environment_preclearance_used": False,
                "passed": bool(report["complete_sequential_ik"] and not below),
            }
        parity = _contact_parity_fixture(task, assembly)
        return {
            "role_index": int(cell["role_index"]),
            "family": cell.get("family"),
            "intrusion_side": side,
            "directions": directions,
            "contact_parity": parity,
            "passed": bool(
                all(item["passed"] for item in directions.values()) and parity["passed"]
            ),
        }
    finally:
        cleanup_task(task, sampler, scratch)


def preflight() -> dict[str, Any]:
    assembly = frozen_assembly()
    protected = item_protected_artifacts()
    reconstruction, _snapshot, cells = verify_v99_inputs()
    cells = sorted(cells, key=lambda item: int(item["role_index"]))
    cells_env = load_environment_geoms(ROOT / ENVIRONMENT_DUMP_RELATIVE)
    validation = item_assembly_validation(assembly, cells, cells_env)
    necessity = item_necessity(assembly, cells)
    route, plans = item_route_geometry(assembly, cells)
    jobs = {
        int(json.loads((Path(job["row_dir"]) / "result.json").read_text())["role_index"]): job
        for job in snapshot_jobs_from_reconstruction(reconstruction)
    }
    live_cells = []
    for cell in cells:
        live_cells.append(
            _live_cell(jobs[int(cell["role_index"])], cell, assembly, plans)
        )
        print(
            f"live cell role={cell['role_index']} {cell.get('family')} "
            f"{cell.get('intrusion_side')} passed={live_cells[-1]['passed']}",
            flush=True,
        )
    sequential_ik = {
        "passed": all(
            item["directions"][direction]["complete_sequential_ik"]
            for item in live_cells
            for direction in ("inbound", "outbound")
        ),
        "cases": [
            {
                "role_index": item["role_index"],
                "family": item["family"],
                "intrusion_side": item["intrusion_side"],
                "direction": direction,
                "waypoints_attempted": item["directions"][direction]["waypoints_attempted"],
                "waypoints_solved": item["directions"][direction]["waypoints_solved"],
                "complete_sequential_ik": item["directions"][direction][
                    "complete_sequential_ik"
                ],
            }
            for item in live_cells
            for direction in ("inbound", "outbound")
        ],
    }
    clearance = {
        "passed": all(
            item["directions"][direction]["meets_clearance_floor"]
            for item in live_cells
            for direction in ("inbound", "outbound")
        ),
        "floor_m": float(MIN_PENDANT_CLEARANCE_M),
        "reported_per_component": True,
        "cases": [
            {
                "role_index": item["role_index"],
                "family": item["family"],
                "intrusion_side": item["intrusion_side"],
                "direction": direction,
                "per_component_min_clearance_m": item["directions"][direction][
                    "per_component_min_clearance_m"
                ],
                "min_clearance_m": item["directions"][direction]["min_clearance_m"],
                "components_below_floor": item["directions"][direction][
                    "components_below_floor"
                ],
                "worst_waypoint_index": item["directions"][direction][
                    "worst_waypoint_index"
                ],
            }
            for item in live_cells
            for direction in ("inbound", "outbound")
        ],
    }
    parity = {
        "passed": all(item["contact_parity"]["passed"] for item in live_cells),
        "cases": [
            {
                "role_index": item["role_index"],
                "intrusion_side": item["intrusion_side"],
                **item["contact_parity"],
            }
            for item in live_cells
        ],
    }
    items = {
        "1_protected_artifacts_and_scene_hashes": protected,
        "2_raised_assembly_validation": validation,
        "3_stock_route_necessity": necessity,
        "4_fixed_endpoint_only_route_geometry": route,
        "5_complete_sequential_ik": sequential_ik,
        "6_per_component_pendant_clearance": clearance,
        "7_contact_parity_fixture": parity,
    }
    failures = [
        {"item": key, "code": "preflight_item_failed"}
        for key, value in items.items()
        if not value["passed"]
    ]
    document = {
        "schema_version": SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "environment_version": ENVIRONMENT_VERSION,
        "sampler_class": SAMPLER_CLASS,
        "scene_xml": SCENE_XML_RELATIVE,
        "scene_sha256": PLACE_V10_SCENE_SHA256,
        "assembly_id": assembly.get("assembly_id"),
        "assembly": assembly,
        "assembly_self_sha256": sha256_payload(assembly),
        "speed_schedule": speed_schedule(),
        "speed_schedule_sha256": speed_schedule_sha256(),
        "implementation_sha256": implementation_sha256(),
        "families": list(PHYSICS_CLEAN_FAMILIES),
        "min_pendant_clearance_m": float(MIN_PENDANT_CLEARANCE_M),
        "contact_distance_m": float(CONTACT_DISTANCE_M),
        "items": items,
        "live_cells": live_cells,
        "failures": failures,
        "preflight_passed": not failures,
        "stop": bool(failures),
        **empty_authorization(),
    }
    return document


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    establish_v10_runtime_env()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    document = preflight()
    digest = write_immutable(output_root / "preflight.json", document)
    print(
        json.dumps(
            {
                "preflight_passed": document["preflight_passed"],
                "failures": document["failures"],
                "artifact_sha256": digest,
                "output": str(output_root / "preflight.json"),
            },
            indent=2,
        )
    )
    return 0 if document["preflight_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
