#!/usr/bin/env python3
"""Deterministic V9.9 pendant lattice and AABB predicates.

Pure numpy. No MuJoCo, no V9.8 lag/window imports.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

from pact_place_v99_pendant_contract import (
    BOTTOM_Z_BOUNDS_M,
    BOTTOM_Z_STEP_M,
    CENTER_X_BOUNDS_M,
    CENTER_X_STEP_M,
    CENTER_Y_BOUNDS_M,
    CENTER_Y_STEP_M,
    DEFAULT_APERTURE_WIDTH_M,
    HALF_X_CHOICES_M,
    HALF_Y_BOUNDS_M,
    HALF_Y_STEP_M,
    build_pendant_fixture,
    pendant_aabb,
    pendant_volume_m3,
)

GRASP_WINDOW_PHASES = frozenset(
    {"pregrasp", "grasp", "grasp_settle", "gripper-close", "lift"}
)



def _arange_inclusive(start: float, stop: float, step: float) -> np.ndarray:
    count = int(round((stop - start) / step)) + 1
    values = start + step * np.arange(count, dtype=np.float64)
    values = np.round(values, 9)
    return values[(values >= start - 1e-12) & (values <= stop + 1e-12)]


def aabb_from_center_half(
    center: Sequence[float], half: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    center_v = np.asarray(center, dtype=np.float64).reshape(3)
    half_v = np.asarray(half, dtype=np.float64).reshape(3)
    return center_v - half_v, center_v + half_v


def aabb_overlap(
    lo_a: Sequence[float],
    hi_a: Sequence[float],
    lo_b: Sequence[float],
    hi_b: Sequence[float],
) -> bool:
    lo_a_v = np.asarray(lo_a, dtype=np.float64)
    hi_a_v = np.asarray(hi_a, dtype=np.float64)
    lo_b_v = np.asarray(lo_b, dtype=np.float64)
    hi_b_v = np.asarray(hi_b, dtype=np.float64)
    return bool(np.all(lo_a_v <= hi_b_v) and np.all(lo_b_v <= hi_a_v))


def aabb_separation_m(
    lo_a: Sequence[float],
    hi_a: Sequence[float],
    lo_b: Sequence[float],
    hi_b: Sequence[float],
) -> float:
    lo_a_v = np.asarray(lo_a, dtype=np.float64)
    hi_a_v = np.asarray(hi_a, dtype=np.float64)
    lo_b_v = np.asarray(lo_b, dtype=np.float64)
    hi_b_v = np.asarray(hi_b, dtype=np.float64)
    delta = np.maximum(0.0, np.maximum(lo_b_v - hi_a_v, lo_a_v - hi_b_v))
    return float(np.linalg.norm(delta))


def geoms_intersect_box(
    geom_lo: np.ndarray,
    geom_hi: np.ndarray,
    box_lo: Sequence[float],
    box_hi: Sequence[float],
) -> np.ndarray:
    """Return a boolean mask over frames: any geom AABB overlaps the box."""
    lo = np.asarray(geom_lo, dtype=np.float64)
    hi = np.asarray(geom_hi, dtype=np.float64)
    box_lo_v = np.asarray(box_lo, dtype=np.float64).reshape(3)
    box_hi_v = np.asarray(box_hi, dtype=np.float64).reshape(3)
    overlap = np.all(lo <= box_hi_v, axis=-1) & np.all(box_lo_v <= hi, axis=-1)
    return np.any(overlap, axis=-1)


def geoms_min_separation_m(
    geom_lo: np.ndarray,
    geom_hi: np.ndarray,
    box_lo: Sequence[float],
    box_hi: Sequence[float],
) -> np.ndarray:
    """Per-frame minimum AABB separation to the box (0 if overlapping)."""
    lo = np.asarray(geom_lo, dtype=np.float64)
    hi = np.asarray(geom_hi, dtype=np.float64)
    box_lo_v = np.asarray(box_lo, dtype=np.float64).reshape(3)
    box_hi_v = np.asarray(box_hi, dtype=np.float64).reshape(3)
    delta = np.maximum(0.0, np.maximum(box_lo_v - hi, lo - box_hi_v))
    distances = np.linalg.norm(delta, axis=-1)
    overlap = np.all(lo <= box_hi_v, axis=-1) & np.all(box_lo_v <= hi, axis=-1)
    distances = np.where(overlap, 0.0, distances)
    return np.min(distances, axis=-1)


def enumerate_lattice(
    *,
    aperture_width_m: float = DEFAULT_APERTURE_WIDTH_M,
) -> tuple[dict[str, Any], ...]:
    """Return every lattice box that survives enclosure/aperture rejection."""
    fixtures: list[dict[str, Any]] = []
    for center_x in _arange_inclusive(*CENTER_X_BOUNDS_M, CENTER_X_STEP_M):
        for half_x in HALF_X_CHOICES_M:
            for center_y in _arange_inclusive(*CENTER_Y_BOUNDS_M, CENTER_Y_STEP_M):
                for half_y in _arange_inclusive(*HALF_Y_BOUNDS_M, HALF_Y_STEP_M):
                    for bottom_z in _arange_inclusive(
                        *BOTTOM_Z_BOUNDS_M, BOTTOM_Z_STEP_M
                    ):
                        try:
                            fixture = build_pendant_fixture(
                                center_x_m=float(center_x),
                                center_y_m=float(center_y),
                                half_x_m=float(half_x),
                                half_y_m=float(half_y),
                                bottom_z_m=float(bottom_z),
                                aperture_width_m=aperture_width_m,
                            )
                        except ValueError:
                            continue
                        fixtures.append(fixture)
    return tuple(fixtures)


def lattice_raw_count() -> int:
    n_x = len(_arange_inclusive(*CENTER_X_BOUNDS_M, CENTER_X_STEP_M))
    n_y = len(_arange_inclusive(*CENTER_Y_BOUNDS_M, CENTER_Y_STEP_M))
    n_hy = len(_arange_inclusive(*HALF_Y_BOUNDS_M, HALF_Y_STEP_M))
    n_z = len(_arange_inclusive(*BOTTOM_Z_BOUNDS_M, BOTTOM_Z_STEP_M))
    return int(n_x * len(HALF_X_CHOICES_M) * n_y * n_hy * n_z)


def fixture_key(fixture: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
    center = fixture["center_m"]
    half = fixture["half_m"]
    return (
        float(center[0]),
        float(center[1]),
        float(center[2]),
        float(half[0]),
        float(half[1]),
        float(half[2]),
    )


def evaluate_stock_necessity(
    *,
    fixture: dict[str, Any],
    robot_lo: np.ndarray,
    robot_hi: np.ndarray,
    target_lo: np.ndarray | None,
    target_hi: np.ndarray | None,
    inbound_mask: np.ndarray,
    outbound_mask: np.ndarray,
    grasp_mask: np.ndarray,
    initial_mask: np.ndarray,
    min_grasp_clearance_m: float,
) -> dict[str, Any]:
    """AABB necessity/clearance filter against one reconstructed cell.

    Intersection and clearance here are bounding-box predicates. They do not
    prove mesh contact and are not exact clearance.
    """
    box_lo, box_hi = pendant_aabb(fixture)
    inbound_hit = bool(
        np.any(geoms_intersect_box(robot_lo, robot_hi, box_lo, box_hi)[inbound_mask])
    )
    outbound_robot = geoms_intersect_box(robot_lo, robot_hi, box_lo, box_hi)[
        outbound_mask
    ]
    outbound_hit = bool(np.any(outbound_robot))
    if target_lo is not None and target_hi is not None and target_lo.size:
        outbound_target = geoms_intersect_box(target_lo, target_hi, box_lo, box_hi)[
            outbound_mask
        ]
        outbound_hit = outbound_hit or bool(np.any(outbound_target))
    initial_hit = bool(
        np.any(geoms_intersect_box(robot_lo, robot_hi, box_lo, box_hi)[initial_mask])
    )
    grasp_sep = geoms_min_separation_m(robot_lo, robot_hi, box_lo, box_hi)[grasp_mask]
    if target_lo is not None and target_hi is not None and target_lo.size:
        grasp_target = geoms_min_separation_m(target_lo, target_hi, box_lo, box_hi)[
            grasp_mask
        ]
        grasp_sep = np.minimum(grasp_sep, grasp_target)
    min_grasp = float(np.min(grasp_sep)) if grasp_sep.size else float("inf")
    return {
        "inbound_stock_intersects": inbound_hit,
        "outbound_stock_intersects": outbound_hit,
        "initial_state_collision": initial_hit,
        "min_grasp_window_aabb_clearance_m": min_grasp,
        "grasp_window_clear": bool(min_grasp + 1e-12 >= float(min_grasp_clearance_m)),
        "necessary_both_traversals": bool(inbound_hit and outbound_hit),
        "accepted": bool(
            inbound_hit
            and outbound_hit
            and not initial_hit
            and min_grasp + 1e-12 >= float(min_grasp_clearance_m)
        ),
        "volume_m3": pendant_volume_m3(fixture),
    }


def filter_lattice_for_cells(
    fixtures: Iterable[dict[str, Any]],
    cells: Sequence[dict[str, Any]],
    *,
    min_grasp_clearance_m: float,
    batch_size: int = 256,
    require_grasp_aabb_clearance: bool = True,
) -> list[dict[str, Any]]:
    fixture_list = list(fixtures)
    if not fixture_list or not cells:
        return []
    box_lo = np.asarray(
        [pendant_aabb(item)[0] for item in fixture_list], dtype=np.float64
    )
    box_hi = np.asarray(
        [pendant_aabb(item)[1] for item in fixture_list], dtype=np.float64
    )
    accepted = np.ones(len(fixture_list), dtype=bool)
    for start in range(0, len(fixture_list), batch_size):
        stop = min(start + batch_size, len(fixture_list))
        batch_lo = box_lo[start:stop]
        batch_hi = box_hi[start:stop]
        batch_ok = np.ones(stop - start, dtype=bool)
        for cell in cells:
            inbound = _batch_any_overlap(
                cell["robot_lo"], cell["robot_hi"], batch_lo, batch_hi, cell["inbound_mask"]
            )
            outbound = _batch_any_overlap(
                cell["robot_lo"], cell["robot_hi"], batch_lo, batch_hi, cell["outbound_mask"]
            )
            target_lo = cell.get("target_lo")
            target_hi = cell.get("target_hi")
            if target_lo is not None and target_hi is not None and np.asarray(target_lo).size:
                outbound = outbound | _batch_any_overlap(
                    target_lo, target_hi, batch_lo, batch_hi, cell["outbound_mask"]
                )
            initial = _batch_any_overlap(
                cell["robot_lo"], cell["robot_hi"], batch_lo, batch_hi, cell["initial_mask"]
            )
            batch_ok &= inbound & outbound & (~initial)
            if require_grasp_aabb_clearance:
                grasp = _batch_min_separation(
                    cell["robot_lo"], cell["robot_hi"], batch_lo, batch_hi, cell["grasp_mask"]
                )
                if target_lo is not None and target_hi is not None and np.asarray(target_lo).size:
                    grasp = np.minimum(
                        grasp,
                        _batch_min_separation(
                            target_lo, target_hi, batch_lo, batch_hi, cell["grasp_mask"]
                        ),
                    )
                batch_ok &= grasp + 1e-12 >= float(min_grasp_clearance_m)
            if not np.any(batch_ok):
                break
        accepted[start:stop] = batch_ok
    survivors = []
    for fixture, flag in zip(fixture_list, accepted.tolist()):
        if not flag:
            continue
        survivors.append(
            {
                "fixture": fixture,
                "key": fixture_key(fixture),
                "volume_m3": pendant_volume_m3(fixture),
            }
        )
    return survivors


def filter_lattice_dual_transit(
    fixtures: Iterable[dict[str, Any]],
    cells: Sequence[dict[str, Any]],
    *,
    batch_size: int = 256,
) -> list[dict[str, Any]]:
    """AABB broad-phase: stock inbound and outbound overlap on every cell.

    This is not an exact contact certificate and does not measure grasp
    clearance. Surviving boxes are the pool for a later true_distance pass.
    """
    return filter_lattice_for_cells(
        fixtures,
        cells,
        min_grasp_clearance_m=0.0,
        batch_size=batch_size,
        require_grasp_aabb_clearance=False,
    )


def _batch_any_overlap(
    geom_lo: np.ndarray,
    geom_hi: np.ndarray,
    box_lo: np.ndarray,
    box_hi: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    lo = np.asarray(geom_lo, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    hi = np.asarray(geom_hi, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    if lo.size == 0:
        return np.zeros(len(box_lo), dtype=bool)
    overlap = np.all(lo[:, :, None, :] <= box_hi[None, None, :, :], axis=-1) & np.all(
        box_lo[None, None, :, :] <= hi[:, :, None, :], axis=-1
    )
    return np.any(overlap, axis=(0, 1))


def _batch_min_separation(
    geom_lo: np.ndarray,
    geom_hi: np.ndarray,
    box_lo: np.ndarray,
    box_hi: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    lo = np.asarray(geom_lo, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    hi = np.asarray(geom_hi, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    if lo.size == 0:
        return np.full(len(box_lo), np.inf, dtype=np.float64)
    delta = np.maximum(
        0.0,
        np.maximum(
            box_lo[None, None, :, :] - hi[:, :, None, :],
            lo[:, :, None, :] - box_hi[None, None, :, :],
        ),
    )
    distance = np.linalg.norm(delta, axis=-1)
    overlap = np.all(lo[:, :, None, :] <= box_hi[None, None, :, :], axis=-1) & np.all(
        box_lo[None, None, :, :] <= hi[:, :, None, :], axis=-1
    )
    distance = np.where(overlap, 0.0, distance)
    return np.min(distance, axis=(0, 1))


def window_masks(phases: Sequence[str]) -> dict[str, Any]:
    """Split a retained trajectory into inbound, grasp-window, and outbound."""
    labels = [str(phase) for phase in phases]
    n_steps = len(labels)
    inbound = np.array([phase.startswith("inbound") for phase in labels])
    outbound = np.array([phase.startswith("outbound") for phase in labels])
    grasp_i = next((index for index, phase in enumerate(labels) if phase == "grasp"), None)
    if grasp_i is None:
        raise ValueError("trajectory has no grasp phase")
    pre_i = grasp_i
    while pre_i > 0 and labels[pre_i - 1] == "pregrasp":
        pre_i -= 1
    grasp_window = np.zeros(n_steps, dtype=bool)
    grasp_window[pre_i:grasp_i] = True
    for index, phase in enumerate(labels):
        if phase in {"grasp", "grasp_settle", "gripper-close"}:
            grasp_window[index] = True
        if phase == "lift" and index > grasp_i:
            grasp_window[index] = True
    initial = np.zeros(n_steps, dtype=bool)
    if n_steps:
        initial[0] = True
    return {
        "inbound_mask": inbound,
        "outbound_mask": outbound,
        "grasp_mask": grasp_window,
        "initial_mask": initial,
        "pregrasp_index": int(pre_i),
        "grasp_index": int(grasp_i),
        "final_grasp_index": int(
            max(index for index, phase in enumerate(labels) if phase == "grasp")
        ),
    }
