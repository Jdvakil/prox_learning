#!/usr/bin/env python3
"""V9.9 full-route lane primitive.

Pure geometry: densify, lane rewrite, 0.05 m detour vs stock x, and the eight
±5 mm perturbation corners. Sequential IK and live clearance live in the
policy/search layer.
"""

from __future__ import annotations

import itertools
import math
from typing import Any, Sequence

import numpy as np

from pact_place_v99_pendant_contract import (
    APERTURE_EDGE_RESERVE_M,
    DEFAULT_APERTURE_WIDTH_M,
    LANE_Y_STEP_M,
    MAX_SEGMENT_ROTATION_DEG,
    MAX_SEGMENT_TRANSLATION_M,
    MIN_DETOUR_M,
    PERTURBATION_M,
    SLAB_PADDINGS_M,
    pendant_aabb,
)

MAX_SEGMENT_ROTATION_RAD = math.radians(MAX_SEGMENT_ROTATION_DEG)
PERTURBATION_CORNERS = tuple(
    itertools.product((-PERTURBATION_M, PERTURBATION_M), repeat=3)
)


def panel_lane_sign(panel_side: str) -> float:
    side = str(panel_side).strip().lower()
    if side in {"left", "-y", "minus_y"}:
        return -1.0
    if side in {"right", "+y", "plus_y"}:
        return 1.0
    raise ValueError(f"V9.9 panel side must be left or right, got {panel_side!r}")


def lane_y_grid(
    panel_side: str,
    *,
    aperture_width_m: float = DEFAULT_APERTURE_WIDTH_M,
) -> np.ndarray:
    limit = float(aperture_width_m) / 2.0 - APERTURE_EDGE_RESERVE_M
    count = int(math.floor(limit / LANE_Y_STEP_M + 1e-12))
    magnitudes = LANE_Y_STEP_M * np.arange(1, count + 1, dtype=np.float64)
    magnitudes = magnitudes[magnitudes <= limit + 1e-12]
    sign = panel_lane_sign(panel_side)
    return np.round(sign * magnitudes, 9)


def rotation_angle_rad(start_R: np.ndarray, end_R: np.ndarray) -> float:
    delta = np.asarray(start_R, dtype=np.float64).T @ np.asarray(end_R, dtype=np.float64)
    trace = float(np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0))
    return float(math.acos(trace))


def interpolate_rotation(start_R: np.ndarray, end_R: np.ndarray, t: float) -> np.ndarray:
    from scipy.spatial.transform import Rotation as R
    from scipy.spatial.transform import Slerp

    times = np.asarray([0.0, 1.0], dtype=float)
    rots = R.from_matrix(
        [np.asarray(start_R, dtype=float), np.asarray(end_R, dtype=float)]
    )
    slerp = Slerp(times, rots)
    return np.asarray(slerp([float(t)]).as_matrix()[0], dtype=np.float64)


def densify_path(
    positions: np.ndarray,
    rotations: np.ndarray,
    *,
    max_translation_m: float = MAX_SEGMENT_TRANSLATION_M,
    max_rotation_rad: float = MAX_SEGMENT_ROTATION_RAD,
) -> tuple[np.ndarray, np.ndarray]:
    positions = np.asarray(positions, dtype=np.float64)
    rotations = np.asarray(rotations, dtype=np.float64)
    if len(positions) == 0:
        return positions, rotations
    out_p = [positions[0].copy()]
    out_r = [rotations[0].copy()]
    for index in range(1, len(positions)):
        start_p, end_p = positions[index - 1], positions[index]
        start_r, end_r = rotations[index - 1], rotations[index]
        dist = float(np.linalg.norm(end_p - start_p))
        angle = rotation_angle_rad(start_r, end_r)
        n_pieces = max(
            1,
            int(math.ceil(dist / max_translation_m - 1e-12)),
            int(math.ceil(angle / max_rotation_rad - 1e-12)),
        )
        for piece in range(1, n_pieces + 1):
            t = piece / n_pieces
            out_p.append(start_p + t * (end_p - start_p))
            out_r.append(interpolate_rotation(start_r, end_r, t))
    return np.asarray(out_p, dtype=np.float64), np.asarray(out_r, dtype=np.float64)


def slab_x_bounds(
    fixture: dict[str, Any], padding_m: float
) -> tuple[float, float, float, float]:
    low, high = pendant_aabb(fixture)
    physical_lo, physical_hi = float(low[0]), float(high[0])
    return (
        physical_lo,
        physical_hi,
        physical_lo - float(padding_m),
        physical_hi + float(padding_m),
    )


def travel_sign_through_slab(positions: np.ndarray, x_lo: float, x_hi: float) -> float:
    xs = np.asarray(positions[:, 0], dtype=np.float64)
    inside = np.flatnonzero((xs >= x_lo - 1e-12) & (xs <= x_hi + 1e-12))
    if inside.size < 2:
        dx = float(xs[-1] - xs[0]) if len(xs) else 0.0
        return 1.0 if dx >= 0.0 else -1.0
    dx = float(xs[inside[-1]] - xs[inside[0]])
    if abs(dx) < 1e-9:
        dx = float(xs[-1] - xs[0])
    return 1.0 if dx >= 0.0 else -1.0


def entry_exit_x(
    fixture: dict[str, Any], padding_m: float, travel_sign: float
) -> tuple[float, float]:
    physical_lo, physical_hi, padded_lo, padded_hi = slab_x_bounds(fixture, padding_m)
    del physical_lo, physical_hi
    if float(travel_sign) >= 0.0:
        return float(padded_lo), float(padded_hi)
    return float(padded_hi), float(padded_lo)


def apply_constant_lane(
    positions: np.ndarray,
    rotations: np.ndarray,
    *,
    lane_y: float,
    entry_x: float,
    exit_x: float,
    freeze_start: bool = False,
    freeze_final: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rewrite y to lane_y between entry and exit x. Keep stock orientation.

    Frozen endpoints are not rewritten so inbound can rejoin the stock
    pregrasp and outbound can leave the stock lift endpoint unchanged. A
    contiguous in-slab group that contains a frozen endpoint is treated as a
    rejoin/lead-in and is not rewritten.
    """
    source = np.asarray(positions, dtype=np.float64)
    planned = source.copy()
    rotations = np.asarray(rotations, dtype=np.float64).copy()
    x_lo, x_hi = min(entry_x, exit_x), max(entry_x, exit_x)
    inside = (planned[:, 0] >= x_lo - 1e-12) & (planned[:, 0] <= x_hi + 1e-12)
    groups: list[np.ndarray] = []
    current: list[int] = []
    for index, flag in enumerate(inside.tolist()):
        if flag:
            current.append(index)
        elif current:
            groups.append(np.asarray(current, dtype=int))
            current = []
    if current:
        groups.append(np.asarray(current, dtype=int))
    frozen: set[int] = set()
    if freeze_final:
        frozen.add(len(planned) - 1)
    if freeze_start:
        frozen.add(0)
    kept: list[np.ndarray] = []
    for group in groups:
        if frozen.intersection(int(index) for index in group.tolist()):
            continue
        kept.append(group)
    rewritten = np.zeros(len(planned), dtype=bool)
    for group in kept:
        planned[group, 1] = float(lane_y)
        rewritten[group] = True
    for index in frozen:
        planned[index] = source[index]
    return planned, rotations, rewritten


def lane_inside_aperture(
    lane_y: float, *, aperture_width_m: float = DEFAULT_APERTURE_WIDTH_M
) -> bool:
    limit = float(aperture_width_m) / 2.0 - APERTURE_EDGE_RESERVE_M
    return bool(abs(float(lane_y)) <= limit + 1e-12)


def interpolate_y_at_x(path_xy: np.ndarray, x_query: float) -> list[float]:
    ys: list[float] = []
    path = np.asarray(path_xy, dtype=np.float64)
    for index in range(len(path) - 1):
        x0, x1 = float(path[index, 0]), float(path[index + 1, 0])
        y0, y1 = float(path[index, 1]), float(path[index + 1, 1])
        lo, hi = min(x0, x1), max(x0, x1)
        if hi - lo < 1e-12:
            if abs(x_query - x0) <= 1e-9:
                ys.append(y0)
            continue
        if lo - 1e-12 <= float(x_query) <= hi + 1e-12:
            t = (float(x_query) - x0) / (x1 - x0)
            ys.append(y0 + t * (y1 - y0))
    return ys


def min_abs_detour_in_slab_m(
    planned_xy: np.ndarray,
    stock_xy: np.ndarray,
    *,
    x_lo: float,
    x_hi: float,
) -> dict[str, Any]:
    """Compare planned y vs stock y at the same monotonic x inside the slab."""
    planned = np.asarray(planned_xy, dtype=np.float64)
    stock = np.asarray(stock_xy, dtype=np.float64)
    samples: list[float] = []
    missing = 0
    for point in planned:
        x_val = float(point[0])
        if x_val < x_lo - 1e-12 or x_val > x_hi + 1e-12:
            continue
        stock_ys = interpolate_y_at_x(stock, x_val)
        if not stock_ys:
            missing += 1
            continue
        samples.append(min(abs(float(point[1]) - y_val) for y_val in stock_ys))
    if not samples:
        return {
            "n_samples": 0,
            "missing_stock_x": missing,
            "min_abs_detour_m": 0.0,
            "meets_minimum": False,
        }
    minimum = float(np.min(samples))
    return {
        "n_samples": len(samples),
        "missing_stock_x": missing,
        "min_abs_detour_m": minimum,
        "meets_minimum": bool(minimum + 1e-12 >= MIN_DETOUR_M and missing == 0),
        "all_abs_detours_m": samples,
    }


def perturbation_corners(
    lane_y: float, entry_x: float, exit_x: float
) -> tuple[dict[str, float], ...]:
    corners = []
    for dy, dx_entry, dx_exit in PERTURBATION_CORNERS:
        corners.append(
            {
                "lane_y_m": float(lane_y + dy),
                "entry_x_m": float(entry_x + dx_entry),
                "exit_x_m": float(exit_x + dx_exit),
                "delta_lane_y_m": float(dy),
                "delta_entry_x_m": float(dx_entry),
                "delta_exit_x_m": float(dx_exit),
            }
        )
    return tuple(corners)


def plan_lane(
    stock_positions: np.ndarray,
    stock_rotations: np.ndarray,
    *,
    fixture: dict[str, Any],
    panel_side: str,
    lane_y_m: float,
    padding_m: float,
    aperture_width_m: float = DEFAULT_APERTURE_WIDTH_M,
    freeze_start: bool = False,
    freeze_final: bool = True,
) -> dict[str, Any]:
    stock_p = np.asarray(stock_positions, dtype=np.float64)
    stock_r = np.asarray(stock_rotations, dtype=np.float64)
    sign = panel_lane_sign(panel_side)
    clipped = not lane_inside_aperture(lane_y_m, aperture_width_m=aperture_width_m)
    wrong_way = float(lane_y_m) * sign <= 0.0
    physical_lo, physical_hi, padded_lo, padded_hi = slab_x_bounds(fixture, padding_m)
    travel = travel_sign_through_slab(stock_p, padded_lo, padded_hi)
    entry_x, exit_x = entry_exit_x(fixture, padding_m, travel)
    stock_dense_p, stock_dense_r = densify_path(stock_p, stock_r)
    planned_p, planned_r, rewritten = apply_constant_lane(
        stock_dense_p,
        stock_dense_r,
        lane_y=lane_y_m,
        entry_x=entry_x,
        exit_x=exit_x,
        freeze_start=freeze_start,
        freeze_final=freeze_final,
    )
    planned_p, planned_r = densify_path(planned_p, planned_r)
    detour = min_abs_detour_in_slab_m(
        planned_p[:, :2],
        stock_dense_p[:, :2],
        x_lo=physical_lo,
        x_hi=physical_hi,
    )
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
        "clipped": bool(clipped),
        "wrong_way": bool(wrong_way),
        "detour": detour,
        "planned_positions_m": planned_p,
        "planned_rotations": planned_r,
        "stock_positions_m": stock_dense_p,
        "orientation_source": "stock_interpolated",
        "extra_orientation_change": False,
        "perturbation_corners": perturbation_corners(lane_y_m, entry_x, exit_x),
        "rewritten_samples": int(np.sum(rewritten)),
        "accepted_geometry": bool(
            (not clipped)
            and (not wrong_way)
            and detour["meets_minimum"]
            and any(abs(float(padding_m) - value) <= 1e-9 for value in SLAB_PADDINGS_M)
        ),
    }


def named_lane_segments(
    planned_positions: np.ndarray,
    planned_rotations: np.ndarray,
    *,
    prefix: str,
    entry_x: float,
    exit_x: float,
    stock_end: np.ndarray,
) -> list[dict[str, Any]]:
    """Split a planned polyline into approach / pass / exit named pieces."""
    positions = np.asarray(planned_positions, dtype=np.float64)
    rotations = np.asarray(planned_rotations, dtype=np.float64)
    x_lo, x_hi = min(entry_x, exit_x), max(entry_x, exit_x)
    inside = (positions[:, 0] >= x_lo - 1e-12) & (positions[:, 0] <= x_hi + 1e-12)
    if not np.any(inside):
        return [
            {
                "name": f"{prefix}_pass",
                "positions_m": positions,
                "rotations": rotations,
            }
        ]
    first = int(np.flatnonzero(inside)[0])
    last = int(np.flatnonzero(inside)[-1])
    pieces: list[dict[str, Any]] = []
    if first > 0:
        pieces.append(
            {
                "name": f"{prefix}_approach",
                "positions_m": positions[: first + 1],
                "rotations": rotations[: first + 1],
            }
        )
    pieces.append(
        {
            "name": f"{prefix}_pass",
            "positions_m": positions[first : last + 1],
            "rotations": rotations[first : last + 1],
        }
    )
    tail_p = positions[last:]
    tail_r = rotations[last:]
    if len(tail_p) > 1:
        pieces.append(
            {
                "name": f"{prefix}_exit",
                "positions_m": tail_p,
                "rotations": tail_r,
            }
        )
    end_p = np.asarray(stock_end, dtype=np.float64).reshape(3)
    if float(np.linalg.norm(positions[-1] - end_p)) > 1e-9:
        pieces.append(
            {
                "name": f"{prefix}_rejoin",
                "positions_m": np.vstack([positions[-1], end_p]),
                "rotations": np.stack([rotations[-1], rotations[-1]]),
            }
        )
    return pieces


def select_at_most_two_candidates(
    scored: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Signal then clearance ranking. At most two distinct geometries."""
    if not scored:
        return []

    def _signal_key(item: dict[str, Any]) -> tuple[float, int, float, float]:
        return (
            float(item["worst_cell_changed_value_fraction"]),
            int(item["worst_cell_changed_sensors"]),
            float(item["min_robust_clearance_m"]),
            -float(item["volume_m3"]),
        )

    def _clearance_key(item: dict[str, Any]) -> tuple[float, float, int]:
        return (
            float(item["min_robust_clearance_m"]),
            float(item["worst_cell_changed_value_fraction"]),
            int(item["worst_cell_changed_sensors"]),
        )

    def _key_of(item: dict[str, Any]) -> Any:
        return item.get("key") or tuple(item["fixture"]["center_m"] + item["fixture"]["half_m"])

    by_signal = sorted(scored, key=_signal_key, reverse=True)
    by_clearance = sorted(scored, key=_clearance_key, reverse=True)
    signal = dict(by_signal[0])
    signal["rank_role"] = "signal"
    clearance = dict(by_clearance[0])
    if _key_of(signal) == _key_of(clearance):
        nxt = next(
            (item for item in by_clearance if _key_of(item) != _key_of(signal)),
            None,
        )
        if nxt is None:
            return [signal]
        distinct = dict(nxt)
        distinct["rank_role"] = "clearance"
        return [signal, distinct]
    clearance["rank_role"] = "clearance"
    return [signal, clearance]


def path_crosses_padded_slab(positions: np.ndarray, fixture: dict[str, Any], padding_m: float) -> bool:
    _, _, padded_lo, padded_hi = slab_x_bounds(fixture, padding_m)
    xs = np.asarray(positions, dtype=np.float64)[:, 0]
    return bool(np.any((xs >= padded_lo - 1e-12) & (xs <= padded_hi + 1e-12)))
