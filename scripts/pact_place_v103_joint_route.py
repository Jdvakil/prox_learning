#!/usr/bin/env python3
"""V10.3 deterministic multi-branch joint-space route planner.

Additive. Nothing here alters the historical V9.9/V10/V10.1/V10.2 route
functions, and no V10.3 code path is reachable from a historical row.

The planner builds a small layered graph of IK branches over named control
poses, validates every adjacent-layer edge by joint-space interpolation with
exact distance at every sample, and selects one joint trajectory per
cell/direction. The selected joint trajectory is what the runtime executes:
there is no re-solve of TCP IK at episode time.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from pact_place_v102_route import (
    EMPTY_ARM_APPROACH_SPEED_M_S,
    PENDANT_PASS_SPEED_M_S,
    PREGRASP_APPROACH_SPEED_M_S,
    INHERITED_SPEED_FAST_M_S,
)
from pact_place_v103_geometry import (
    PENDANT_FAR_FACE_X_M,
    PENDANT_NEAR_FACE_X_M,
)

ARM_GROUP = "arm"
N_ARM_JOINTS = 7

# ---------------------------------------------------------------------------
# Registered route-template lattice. Enumerated exactly once.
# ---------------------------------------------------------------------------
LANE_MAGNITUDES_M = (0.28, 0.30, 0.32)
STAGING_BUFFERS_M = (0.10, 0.12)
PASS_Z_OFFSETS_M = (-0.06, -0.04, -0.02, 0.00)
LEFT_PASS_ROTATIONS = (
    ("identity", None, 0.0),
    ("Rx-5", "x", -5.0),
    ("Rx-10", "x", -10.0),
    ("Ry+5", "y", 5.0),
    ("Ry+10", "y", 10.0),
)
RIGHT_PASS_ROTATIONS = (
    ("identity", None, 0.0),
    ("Rx+5", "x", 5.0),
    ("Rx+10", "x", 10.0),
    ("Ry-5", "y", -5.0),
    ("Ry-10", "y", -10.0),
)

CONTROL_POSES_INBOUND = (
    "actual_initial",
    "near_stock_staging",
    "near_lane_staging",
    "far_lane_staging",
    "far_pregrasp_staging",
    "actual_pregrasp_endpoint",
)
CONTROL_POSES_OUTBOUND = (
    "actual_loaded_lift",
    "far_stock_staging",
    "far_lane_staging",
    "near_lane_staging",
    "near_exit_staging",
    "actual_outbound_endpoint",
)

# ---------------------------------------------------------------------------
# Registered planner thresholds.
# ---------------------------------------------------------------------------
N_HALTON_SEEDS = 24
HALTON_SCRAMBLE_SEED = 20261030
HALTON_INNER_FRACTION = 0.80
DEDUP_LINF_RAD = 0.02
MAX_JOINT_STEP_RAD = 0.01
NODE_MIN_CLEARANCE_M = 0.020
EDGE_MIN_CLEARANCE_M = 0.020
CORNER_MIN_CLEARANCE_M = 0.015
CORNER_PERTURBATION_M = 0.005
MAX_POSITION_RESIDUAL_M = 0.001
MAX_ORIENTATION_RESIDUAL_DEG = 1.0
MIN_DETOUR_M = 0.050
VELOCITY_LIMIT_FRACTION = 0.50
# Franka FR3 datasheet joint velocity limits (rad/s), used only when the live
# move group exposes none. The source actually used is recorded per run.
FR3_JOINT_VELOCITY_LIMITS_RAD_S = (2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26)

DISALLOWED_CONTACT_CLASSES = frozenset(
    {"hazard_bar", "other_environment", "clutter", "place_receptacle", "mounted_fixture"}
)

STOP_REASONS = (
    "no_static_geometry_with_twelve_joint_routes",
    "no_route_with_nominal_clearance",
    "no_route_with_robust_clearance",
    "contact_parity_failed",
    "search_input_hash_mismatch",
)


def lane_sign(intrusion_side: str) -> float:
    side = str(intrusion_side).strip().lower()
    if side == "left":
        return -1.0
    if side == "right":
        return 1.0
    raise ValueError(f"intrusion_side must be left or right, got {intrusion_side!r}")


def pass_rotations_for_side(intrusion_side: str):
    return LEFT_PASS_ROTATIONS if lane_sign(intrusion_side) < 0 else RIGHT_PASS_ROTATIONS


def rotation_offset(axis: str | None, degrees: float) -> np.ndarray:
    if axis is None:
        return np.eye(3)
    theta = math.radians(float(degrees))
    c, s = math.cos(theta), math.sin(theta)
    if axis == "x":
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    if axis == "y":
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    raise ValueError(f"unregistered rotation axis {axis!r}")


def enumerate_templates(intrusion_side: str) -> tuple[dict[str, Any], ...]:
    """The registered lattice for one side. 3 x 2 x 4 x 5 = 120 templates."""
    out: list[dict[str, Any]] = []
    sign = lane_sign(intrusion_side)
    for lane in LANE_MAGNITUDES_M:
        for buffer_m in STAGING_BUFFERS_M:
            for pass_dz in PASS_Z_OFFSETS_M:
                for rot_key, axis, degrees in pass_rotations_for_side(intrusion_side):
                    out.append(
                        {
                            "lane_magnitude_m": float(lane),
                            "lane_y_m": float(sign * lane),
                            "staging_buffer_m": float(buffer_m),
                            "pass_z_offset_m": float(pass_dz),
                            "pass_rotation_key": rot_key,
                            "pass_rotation_axis": axis,
                            "pass_rotation_deg": float(degrees),
                            "near_staging_x_m": float(PENDANT_NEAR_FACE_X_M - buffer_m),
                            "far_staging_x_m": float(PENDANT_FAR_FACE_X_M + buffer_m),
                            "template_key": (
                                f"lane{lane:.2f}_buf{buffer_m:.2f}_"
                                f"dz{pass_dz:+.2f}_{rot_key}"
                            ),
                        }
                    )
    return tuple(out)


def template_corners(template: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    """Eight deterministic +/-5 mm corners of lane / staging / pass-z."""
    out: list[dict[str, Any]] = []
    for lane_delta in (-CORNER_PERTURBATION_M, CORNER_PERTURBATION_M):
        for buffer_delta in (-CORNER_PERTURBATION_M, CORNER_PERTURBATION_M):
            for z_delta in (-CORNER_PERTURBATION_M, CORNER_PERTURBATION_M):
                corner = dict(template)
                sign = 1.0 if template["lane_y_m"] >= 0.0 else -1.0
                corner["lane_magnitude_m"] = float(
                    template["lane_magnitude_m"] + lane_delta
                )
                corner["lane_y_m"] = float(sign * corner["lane_magnitude_m"])
                corner["staging_buffer_m"] = float(
                    template["staging_buffer_m"] + buffer_delta
                )
                corner["near_staging_x_m"] = float(
                    PENDANT_NEAR_FACE_X_M - corner["staging_buffer_m"]
                )
                corner["far_staging_x_m"] = float(
                    PENDANT_FAR_FACE_X_M + corner["staging_buffer_m"]
                )
                corner["pass_z_offset_m"] = float(template["pass_z_offset_m"] + z_delta)
                corner["corner_key"] = (
                    f"{template['template_key']}"
                    f"|lane{lane_delta:+.3f}|buf{buffer_delta:+.3f}|dz{z_delta:+.3f}"
                )
                out.append(corner)
    return tuple(out)


# ---------------------------------------------------------------------------
# Deterministic scrambled Halton joint seeds
# ---------------------------------------------------------------------------
_HALTON_BASES = (2, 3, 5, 7, 11, 13, 17)


def _halton_scrambled(index: int, base: int, permutation: np.ndarray) -> float:
    value = 0.0
    factor = 1.0 / base
    i = int(index)
    while i > 0:
        digit = i % base
        value += float(permutation[digit]) * factor
        i //= base
        factor /= base
    return value


def halton_joint_seeds(
    joint_low: np.ndarray,
    joint_high: np.ndarray,
    *,
    n: int = N_HALTON_SEEDS,
    scramble_seed: int = HALTON_SCRAMBLE_SEED,
    inner_fraction: float = HALTON_INNER_FRACTION,
) -> np.ndarray:
    """``n`` fixed joint-space seeds in the inner fraction of each finite range."""
    low = np.asarray(joint_low, dtype=float).reshape(N_ARM_JOINTS)
    high = np.asarray(joint_high, dtype=float).reshape(N_ARM_JOINTS)
    finite = np.isfinite(low) & np.isfinite(high)
    center = np.where(finite, 0.5 * (low + high), 0.0)
    half = np.where(finite, 0.5 * (high - low) * float(inner_fraction), 0.0)
    rng = np.random.default_rng(int(scramble_seed))
    permutations = [rng.permutation(base) for base in _HALTON_BASES]
    seeds = np.zeros((int(n), N_ARM_JOINTS), dtype=float)
    for row in range(int(n)):
        for joint in range(N_ARM_JOINTS):
            unit = _halton_scrambled(row + 1, _HALTON_BASES[joint], permutations[joint])
            seeds[row, joint] = center[joint] + (2.0 * unit - 1.0) * half[joint]
    return seeds


def dedup_joint_solutions(
    solutions: Sequence[np.ndarray], *, tolerance_rad: float = DEDUP_LINF_RAD
) -> list[int]:
    """Indices of solutions kept after L-infinity deduplication."""
    kept: list[int] = []
    kept_values: list[np.ndarray] = []
    for index, candidate in enumerate(solutions):
        value = np.asarray(candidate, dtype=float).reshape(N_ARM_JOINTS)
        if any(
            float(np.max(np.abs(value - other))) <= float(tolerance_rad)
            for other in kept_values
        ):
            continue
        kept.append(index)
        kept_values.append(value)
    return kept


# ---------------------------------------------------------------------------
# Stock reference and control poses
# ---------------------------------------------------------------------------
def stock_pose_at_x(
    positions: np.ndarray, rotations: np.ndarray, x_query: float
) -> tuple[np.ndarray, np.ndarray]:
    """Linear/SLERP interpolation of the retained stock path at one x.

    Clamped to the path's own x range: staging poses registered beyond the
    stock endpoint inherit the endpoint's y/z/orientation.
    """
    from pact_place_v99_route import interpolate_rotation

    pos = np.asarray(positions, dtype=float)
    rot = np.asarray(rotations, dtype=float)
    xs = pos[:, 0]
    query = float(x_query)
    order = 1.0 if xs[-1] >= xs[0] else -1.0
    if (query - xs[0]) * order <= 0.0:
        return pos[0].copy(), rot[0].copy()
    if (query - xs[-1]) * order >= 0.0:
        return pos[-1].copy(), rot[-1].copy()
    for index in range(len(pos) - 1):
        x0, x1 = float(xs[index]), float(xs[index + 1])
        lo, hi = min(x0, x1), max(x0, x1)
        if lo - 1e-12 <= query <= hi + 1e-12:
            span = x1 - x0
            t = 0.0 if abs(span) < 1e-12 else (query - x0) / span
            t = float(np.clip(t, 0.0, 1.0))
            position = pos[index] + t * (pos[index + 1] - pos[index])
            position[0] = query
            rotation = interpolate_rotation(rot[index], rot[index + 1], t)
            return position, rotation
    return pos[-1].copy(), rot[-1].copy()


def build_control_poses(
    stock: dict[str, Any], template: dict[str, Any], *, direction: str
) -> list[dict[str, Any]]:
    """The four intermediate control poses. Endpoints are pinned separately."""
    positions = np.asarray(stock["positions_m"], dtype=float)
    rotations = np.asarray(stock["rotations"], dtype=float)
    near_x = float(template["near_staging_x_m"])
    far_x = float(template["far_staging_x_m"])
    lane_y = float(template["lane_y_m"])
    dz = float(template["pass_z_offset_m"])
    offset = rotation_offset(
        template["pass_rotation_axis"], template["pass_rotation_deg"]
    )
    near_pos, near_rot = stock_pose_at_x(positions, rotations, near_x)
    far_pos, far_rot = stock_pose_at_x(positions, rotations, far_x)

    def lane_pose(pos: np.ndarray, rot: np.ndarray, x_value: float) -> dict[str, Any]:
        point = np.array([x_value, lane_y, float(pos[2]) + dz], dtype=float)
        return {"position_m": point, "rotation": rot @ offset}

    if str(direction) == "inbound":
        end_pos = np.asarray(stock["end_position_m"], dtype=float)
        end_rot = np.asarray(stock["end_rotation"], dtype=float)
        return [
            {
                "name": "near_stock_staging",
                "position_m": np.array([near_x, float(near_pos[1]), float(near_pos[2])]),
                "rotation": near_rot.copy(),
            },
            {"name": "near_lane_staging", **lane_pose(near_pos, near_rot, near_x)},
            {"name": "far_lane_staging", **lane_pose(far_pos, far_rot, far_x)},
            {
                "name": "far_pregrasp_staging",
                "position_m": np.array([far_x, float(end_pos[1]), float(end_pos[2])]),
                "rotation": end_rot.copy(),
            },
        ]
    start_pos = np.asarray(stock["start_position_m"], dtype=float)
    start_rot = np.asarray(stock["start_rotation"], dtype=float)
    return [
        {
            "name": "far_stock_staging",
            "position_m": np.array([far_x, float(start_pos[1]), float(start_pos[2])]),
            "rotation": start_rot.copy(),
        },
        {"name": "far_lane_staging", **lane_pose(far_pos, far_rot, far_x)},
        {"name": "near_lane_staging", **lane_pose(near_pos, near_rot, near_x)},
        {
            "name": "near_exit_staging",
            "position_m": np.array([near_x, float(near_pos[1]), float(near_pos[2])]),
            "rotation": near_rot.copy(),
        },
    ]


def control_pose_key(pose: dict[str, Any]) -> tuple:
    position = np.asarray(pose["position_m"], dtype=float)
    rotation = np.asarray(pose["rotation"], dtype=float)
    return (
        str(pose["name"]),
        tuple(np.round(position, 9).tolist()),
        tuple(np.round(rotation.reshape(9), 9).tolist()),
    )


# ---------------------------------------------------------------------------
# Route diagnostics on the realized FK path
# ---------------------------------------------------------------------------
def route_geometry_report(
    path_positions: np.ndarray,
    stock_positions: np.ndarray,
    *,
    lane_y_m: float,
    direction: str,
    control_x_sequence: Sequence[float],
) -> dict[str, Any]:
    """Signed slab detour, lane side, aperture containment, and x ordering.

    ``control_x_sequence`` is the registered topology's own x ordering. A
    reversal that the registered topology itself contains (the outbound
    retreat in +x from the loaded lift, and the inbound approach back from
    ``far_pregrasp_staging`` to the pregrasp endpoint) is registered and is not
    a wrong-way route; any *additional* reversal inside a segment is.
    """
    from pact_place_v10_compound_pendant_contract import lane_y_limit_m
    from pact_place_v99_route import min_abs_detour_in_slab_m

    path = np.asarray(path_positions, dtype=float)
    stock = np.asarray(stock_positions, dtype=float)
    detour = min_abs_detour_in_slab_m(
        path[:, :2],
        stock[:, :2],
        x_lo=PENDANT_NEAR_FACE_X_M,
        x_hi=PENDANT_FAR_FACE_X_M,
    )
    inside = (path[:, 0] >= PENDANT_NEAR_FACE_X_M - 1e-12) & (
        path[:, 0] <= PENDANT_FAR_FACE_X_M + 1e-12
    )
    sign = 1.0 if float(lane_y_m) >= 0.0 else -1.0
    slab_ys = path[inside, 1] if np.any(inside) else np.zeros(0)
    correct_side = bool(
        slab_ys.size > 0 and np.all(sign * slab_ys > 0.0)
    )
    limit = lane_y_limit_m()
    return {
        "n_slab_samples": int(np.count_nonzero(inside)),
        "min_abs_detour_m": float(detour["min_abs_detour_m"]),
        "detour_samples": int(detour["n_samples"]),
        "missing_stock_x": int(detour["missing_stock_x"]),
        "detour_meets_minimum": bool(
            detour["min_abs_detour_m"] + 1e-12 >= MIN_DETOUR_M
            and detour["missing_stock_x"] == 0
            and detour["n_samples"] > 0
        ),
        "correct_lane_side": correct_side,
        "min_slab_abs_y_m": float(np.min(np.abs(slab_ys))) if slab_ys.size else None,
        "aperture_contained": bool(np.all(np.abs(path[:, 1]) <= limit + 1e-12)),
        "aperture_limit_m": float(limit),
        "registered_control_x_sequence": [float(v) for v in control_x_sequence],
        "direction": str(direction),
    }


def segment_x_monotonic(path_positions: np.ndarray, *, atol: float = 2e-3) -> bool:
    """No additional x reversal inside one registered segment."""
    xs = np.asarray(path_positions, dtype=float)[:, 0]
    if xs.size < 2:
        return True
    deltas = np.diff(xs)
    forward = float(np.sum(deltas[deltas > 0]))
    backward = float(-np.sum(deltas[deltas < 0]))
    return bool(min(forward, backward) <= atol)


# ---------------------------------------------------------------------------
# Segment durations from the inherited V10.2 speed schedule
# ---------------------------------------------------------------------------
def segment_speed_class(name: str, *, direction: str) -> str:
    label = str(name)
    if "lane_staging->far_lane_staging" in label or "lane_staging->near_lane_staging" in label:
        return "pendant_pass"
    if label.endswith("->near_lane_staging") or label.endswith("->far_lane_staging"):
        return "pendant_pass"
    if str(direction) == "inbound":
        if label.startswith("actual_initial") or label.startswith("near_stock_staging"):
            return "empty_arm_approach"
        return "pregrasp_approach"
    return "historical_transport"


def segment_commanded_speed(speed_class: str) -> float:
    if speed_class == "empty_arm_approach":
        return float(EMPTY_ARM_APPROACH_SPEED_M_S)
    if speed_class == "pendant_pass":
        return float(PENDANT_PASS_SPEED_M_S)
    if speed_class == "pregrasp_approach":
        return float(PREGRASP_APPROACH_SPEED_M_S)
    return float(INHERITED_SPEED_FAST_M_S)


def segment_duration_s(
    *,
    tcp_arc_length_m: float,
    joint_displacements_rad: Sequence[float],
    commanded_speed_m_s: float,
    velocity_limits_rad_s: Sequence[float],
    velocity_fraction: float = VELOCITY_LIMIT_FRACTION,
) -> dict[str, Any]:
    """max(FK arc / commanded speed, per-joint travel / 50% velocity limit)."""
    speed = max(float(commanded_speed_m_s), 1e-9)
    tcp_duration = float(tcp_arc_length_m) / speed
    limits = np.asarray(velocity_limits_rad_s, dtype=float).reshape(N_ARM_JOINTS)
    travel = np.abs(np.asarray(joint_displacements_rad, dtype=float)).reshape(
        N_ARM_JOINTS
    )
    joint_durations = travel / np.maximum(limits * float(velocity_fraction), 1e-9)
    joint_duration = float(np.max(joint_durations))
    return {
        "tcp_duration_s": tcp_duration,
        "joint_duration_s": joint_duration,
        "duration_s": float(max(tcp_duration, joint_duration)),
        "binding": "joint" if joint_duration > tcp_duration else "tcp",
        "commanded_speed_m_s": float(commanded_speed_m_s),
        "velocity_fraction": float(velocity_fraction),
    }


# ---------------------------------------------------------------------------
# Graph selection
# ---------------------------------------------------------------------------
def select_path(
    layers: Sequence[Sequence[dict[str, Any]]],
    edges: dict[tuple[int, int, int], dict[str, Any]],
) -> dict[str, Any] | None:
    """Fixed lexicographic ranking over complete layer paths.

    1. maximize minimum robust pendant clearance
    2. maximize minimum joint-limit margin
    3. minimize total seven-arm-joint travel
    4. minimize total pass orientation deviation
    5. lexicographic route key
    """

    def better(candidate, incumbent) -> bool:
        if incumbent is None:
            return True
        for index, sense in enumerate((1, 1, -1, -1)):
            a, b = candidate[index], incumbent[index]
            if abs(a - b) > 1e-12:
                return (a > b) if sense > 0 else (a < b)
        return candidate[4] < incumbent[4]

    n_layers = len(layers)
    if n_layers < 2 or any(len(layer) == 0 for layer in layers):
        return None
    # (min_clearance, min_margin, -travel, -orientation, key, path)
    frontier: dict[int, tuple] = {}
    for index in range(len(layers[0])):
        node = layers[0][index]
        frontier[index] = (
            float("inf"),
            float(node.get("joint_limit_margin_rad", float("inf"))),
            0.0,
            0.0,
            (str(node.get("node_key", index)),),
            [index],
        )
    for layer_index in range(n_layers - 1):
        nxt: dict[int, tuple] = {}
        for src, state in frontier.items():
            for dst in range(len(layers[layer_index + 1])):
                edge = edges.get((layer_index, src, dst))
                if edge is None or not edge.get("passed"):
                    continue
                node = layers[layer_index + 1][dst]
                candidate = (
                    min(state[0], float(edge["min_clearance_m"])),
                    min(
                        state[1],
                        float(edge.get("min_joint_limit_margin_rad", float("inf"))),
                        float(node.get("joint_limit_margin_rad", float("inf"))),
                    ),
                    state[2] - float(edge["joint_travel_rad"]),
                    state[3] - float(node.get("orientation_deviation_deg", 0.0)),
                    state[4] + (str(node.get("node_key", dst)),),
                    state[5] + [dst],
                )
                if better(candidate, nxt.get(dst)):
                    nxt[dst] = candidate
        frontier = nxt
        if not frontier:
            return None
    best = None
    for state in frontier.values():
        if better(state, best):
            best = state
    if best is None:
        return None
    return {
        "node_indices": list(best[5]),
        "min_clearance_m": float(best[0]),
        "min_joint_limit_margin_rad": float(best[1]),
        "total_joint_travel_rad": float(-best[2]),
        "total_orientation_deviation_deg": float(-best[3]),
        "route_key": "|".join(best[4]),
    }


def qpos_sequence_sha256(waypoints: Iterable[Sequence[float]]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for item in waypoints:
        values = np.asarray(item, dtype=float).reshape(-1)
        digest.update(np.round(values, 9).tobytes())
    return digest.hexdigest()
