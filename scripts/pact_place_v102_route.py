#!/usr/bin/env python3
"""V10.2 route dispatch, per-piece speed schedule, and component clearance.

This module is additive. It does not import from, patch, or reinterpret the
frozen V10/V10.1 dispatch in ``pact_place_v10_route``; V10 and V10.1 rows never
reach any code path here because every entry point requires the exact V10.2
environment marker and speed-schedule hash.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

from pact_place_v10_compound_pendant_contract import ENDPOINT_ONLY_PRIMITIVE
from pact_place_v102_geometry import ENVIRONMENT_VERSION_V102

EMPIRICAL_LIVE_CONTACT_V2 = "empirical_live_contact_v2"
SPEED_SCHEDULE_SCHEMA = "pact_place_v102_route_piece_speed_v1"

# Commanded TCP speeds by route piece (m/s). ``None`` means "leave the
# inherited historical speed of the rebuilt lane alone".
EMPTY_ARM_APPROACH_SPEED_M_S = 0.15
PENDANT_PASS_SPEED_M_S = 0.045
PREGRASP_APPROACH_SPEED_M_S = 0.08  # inherited PactPlace speed_slow
HISTORICAL_TRANSPORT_SPEED_M_S = None

INHERITED_SPEED_FAST_M_S = 0.20
INHERITED_SPEED_SLOW_M_S = 0.08

EMPTY_ARM_APPROACH = "empty_arm_approach"
PENDANT_PASS = "pendant_pass"
PREGRASP_APPROACH = "pregrasp_approach"
HISTORICAL_TRANSPORT = "historical_transport"

SPEED_EPSILON_M_S = 1e-9


def speed_schedule() -> dict[str, Any]:
    """The single registered V10.2 route-piece speed schedule."""
    return {
        "schema": SPEED_SCHEDULE_SCHEMA,
        "environment_version": ENVIRONMENT_VERSION_V102,
        "qualification_mode": EMPIRICAL_LIVE_CONTACT_V2,
        "rewrite_primitive": ENDPOINT_ONLY_PRIMITIVE,
        "empty_arm_approach_m_s": float(EMPTY_ARM_APPROACH_SPEED_M_S),
        "pendant_pass_m_s": float(PENDANT_PASS_SPEED_M_S),
        "pregrasp_approach_m_s": float(PREGRASP_APPROACH_SPEED_M_S),
        "post_pick_transport_m_s": HISTORICAL_TRANSPORT_SPEED_M_S,
        "inherited_speed_fast_m_s": float(INHERITED_SPEED_FAST_M_S),
        "inherited_speed_slow_m_s": float(INHERITED_SPEED_SLOW_M_S),
    }


def speed_schedule_sha256() -> str:
    from pact_place_v9_contract import sha256_payload

    return sha256_payload(speed_schedule())


def classify_route_piece(name: str) -> str:
    """Map a ``named_lane_segments`` piece name onto its speed class."""
    label = str(name)
    inbound = label.startswith("inbound_")
    if label.endswith("_pass") or label.endswith("_exit"):
        return PENDANT_PASS
    if label.endswith("_approach"):
        return EMPTY_ARM_APPROACH if inbound else HISTORICAL_TRANSPORT
    if label.endswith("_rejoin"):
        return PREGRASP_APPROACH if inbound else HISTORICAL_TRANSPORT
    raise ValueError(f"unregistered V10.2 route piece: {name!r}")


def route_piece_speed(name: str, *, inherited_speed_m_s: float) -> float:
    """Commanded speed for one rebuilt lane piece."""
    schedule = speed_schedule()
    piece_class = classify_route_piece(name)
    if piece_class == EMPTY_ARM_APPROACH:
        return float(schedule["empty_arm_approach_m_s"])
    if piece_class == PENDANT_PASS:
        return float(schedule["pendant_pass_m_s"])
    if piece_class == PREGRASP_APPROACH:
        return float(schedule["pregrasp_approach_m_s"])
    return float(inherited_speed_m_s)


def speed_cap_violation(name: str, speed_m_s: float) -> str | None:
    """Named cap violation, or None. Rejects a V10.2 row when it fires."""
    piece_class = classify_route_piece(name)
    value = float(speed_m_s)
    if piece_class == PENDANT_PASS and value > PENDANT_PASS_SPEED_M_S + SPEED_EPSILON_M_S:
        return "pendant_pass_speed_above_cap"
    if (
        piece_class == EMPTY_ARM_APPROACH
        and value > EMPTY_ARM_APPROACH_SPEED_M_S + SPEED_EPSILON_M_S
    ):
        return "initial_approach_speed_above_cap"
    return None


def route_is_v102(scene_params: dict[str, Any] | None, route: dict[str, Any] | None) -> bool:
    """True only for the exact registered V10.2 contract marker and hash."""
    scene = dict(scene_params or {})
    payload = dict(route or {})
    if str(scene.get("pact_place_environment_version") or "") != ENVIRONMENT_VERSION_V102:
        return False
    if str(payload.get("environment_version") or "") != ENVIRONMENT_VERSION_V102:
        return False
    if str(payload.get("qualification_mode") or "") != EMPIRICAL_LIVE_CONTACT_V2:
        return False
    if str(payload.get("rewrite_primitive") or "") != ENDPOINT_ONLY_PRIMITIVE:
        return False
    schedule = payload.get("speed_schedule")
    if not isinstance(schedule, dict) or dict(schedule) != speed_schedule():
        return False
    if str(payload.get("speed_schedule_sha256") or "") != speed_schedule_sha256():
        return False
    return True


def resolve_v102_runtime_route(
    scene_params: dict[str, Any] | None, route: dict[str, Any] | None
) -> dict[str, Any] | None:
    """Dispatch record for a V10.2 row, or None when the markers are absent.

    A row that claims V10.2 in one place and not another is a hard error: it
    must never fall through to the historical group-freeze primitive.
    """
    scene = dict(scene_params or {})
    payload = dict(route or {})
    scene_marker = (
        str(scene.get("pact_place_environment_version") or "") == ENVIRONMENT_VERSION_V102
    )
    route_marker = (
        str(payload.get("environment_version") or "") == ENVIRONMENT_VERSION_V102
        or str(payload.get("qualification_mode") or "") == EMPIRICAL_LIVE_CONTACT_V2
    )
    if not scene_marker and not route_marker:
        return None
    if not route_is_v102(scene, payload):
        raise ValueError(
            "V10.2 route markers are incomplete or the speed schedule hash "
            "does not match the registered schedule"
        )
    return {
        "rewrite_primitive": ENDPOINT_ONLY_PRIMITIVE,
        "qualification_mode": EMPIRICAL_LIVE_CONTACT_V2,
        "use_endpoint_only": True,
        "skip_offline_strict_environment": True,
        "speed_schedule": speed_schedule(),
        "speed_schedule_sha256": speed_schedule_sha256(),
        "environment_version": ENVIRONMENT_VERSION_V102,
    }


def copy_qpos_dict(qpos: dict[str, Any]) -> dict[str, np.ndarray]:
    return {key: np.asarray(value, dtype=float).copy() for key, value in qpos.items()}


def sequential_ik_component_clearance(
    positions: np.ndarray,
    rotations: np.ndarray,
    *,
    saved_qpos: dict[str, Any],
    set_qpos: Callable[[dict[str, Any]], Any],
    get_qpos: Callable[[], dict[str, Any]],
    solve_ik: Callable[[np.ndarray, dict[str, Any]], dict[str, Any] | None],
    forward: Callable[[], Any],
    place_pose: Callable[[np.ndarray, np.ndarray], np.ndarray],
    component_names: Sequence[str],
    measure_components: Callable[[], dict[str, float | None]],
) -> dict[str, Any]:
    """Solve every waypoint and record per-component pendant clearance.

    Never aborts early: an environment abort after one waypoint must not be
    reported as an IK pass. ``saved_qpos`` is restored on success, on IK
    failure, and on exception.
    """
    positions = np.asarray(positions, dtype=np.float64)
    rotations = np.asarray(rotations, dtype=np.float64)
    n_waypoints = int(len(positions))
    seed = copy_qpos_dict(saved_qpos)
    solved = 0
    failed_indices: list[int] = []
    per_waypoint: list[dict[str, Any]] = []
    restored = False

    def restore() -> None:
        nonlocal restored
        if restored:
            return
        set_qpos(copy_qpos_dict(saved_qpos))
        forward()
        restored = True

    try:
        for index in range(n_waypoints):
            pose = place_pose(positions[index], rotations[index])
            solution = solve_ik(pose, seed)
            if solution is None:
                failed_indices.append(index)
                per_waypoint.append(
                    {
                        "waypoint_index": index,
                        "ik_ok": False,
                        "clearance_m": {name: None for name in component_names},
                    }
                )
                continue
            set_qpos(solution)
            forward()
            seed = copy_qpos_dict(get_qpos())
            solved += 1
            measured = measure_components()
            per_waypoint.append(
                {
                    "waypoint_index": index,
                    "ik_ok": True,
                    "clearance_m": {
                        name: (
                            None
                            if measured.get(name) is None
                            or not np.isfinite(float(measured[name]))
                            else float(measured[name])
                        )
                        for name in component_names
                    },
                }
            )
    except Exception:
        restore()
        raise
    finally:
        restore()

    per_component_min: dict[str, float | None] = {}
    for name in component_names:
        values = [
            float(item["clearance_m"][name])
            for item in per_waypoint
            if item["clearance_m"].get(name) is not None
        ]
        per_component_min[name] = float(min(values)) if values else None
    finite = [value for value in per_component_min.values() if value is not None]
    return {
        "waypoints_attempted": n_waypoints,
        "waypoints_solved": int(solved),
        "complete_sequential_ik": bool(solved == n_waypoints and n_waypoints > 0),
        "ik_failure_indices": failed_indices,
        "per_waypoint": per_waypoint,
        "per_component_min_clearance_m": per_component_min,
        "min_clearance_m": float(min(finite)) if finite else None,
        "qpos_restored": bool(restored),
    }
