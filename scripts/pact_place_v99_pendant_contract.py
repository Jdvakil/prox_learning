#!/usr/bin/env python3
"""Frozen geometry and admission contract for the V9.9 fixed ceiling pendant.

Independent of V9.8. This module must not import V9.8 lag, face-window, or
ceiling-envelope helpers. It does not render, step physics, or authorize
collection.
"""

from __future__ import annotations

from typing import Any, Sequence

from run_pact_place_v96_cluster_causal_proximity import ADMISSION_FLOOR, CORRIDOR_LINKS

ENVIRONMENT_VERSION = "pact_place_corridor_v9_9_pendant"
CONTRACT_VERSION = "pact_place_v9_9_pendant_v1"
SAMPLER_CLASS = "PactPlaceCorridorV99PendantSampler"

PENDANT_SUPPORT = "ceiling"
PENDANT_BODY = "pact_clutter_mount_ceiling"
PENDANT_GEOM = "pact_clutter_mount_ceiling_g"
CEILING_TOP_Z_M = 1.515
PENDANT_DEPTH_BOUNDS_M = (0.58, 1.36)
CENTER_X_BOUNDS_M = (0.58, 0.90)
CENTER_X_STEP_M = 0.02
HALF_X_CHOICES_M = (0.03, 0.05, 0.07, 0.09)
CENTER_Y_BOUNDS_M = (-0.12, 0.12)
CENTER_Y_STEP_M = 0.02
HALF_Y_BOUNDS_M = (0.04, 0.18)
HALF_Y_STEP_M = 0.02
BOTTOM_Z_BOUNDS_M = (1.10, 1.25)
BOTTOM_Z_STEP_M = 0.025
DEFAULT_APERTURE_WIDTH_M = 0.85
APERTURE_EDGE_RESERVE_M = 0.02
LANE_Y_STEP_M = 0.010
SLAB_PADDINGS_M = (0.080, 0.100, 0.120, 0.140)
MIN_NOMINAL_CLEARANCE_M = 0.025
MIN_ROBUST_CLEARANCE_M = 0.020
MIN_LIVE_CLEARANCE_M = 0.020
MIN_DETOUR_M = 0.050
PERTURBATION_M = 0.005
MAX_SEGMENT_TRANSLATION_M = 0.005
MAX_SEGMENT_ROTATION_DEG = 2.0
MAX_TCP_RESIDUAL_M = 0.001
MAX_GRASP_JOINT_ERROR_RAD = 0.001
MAX_SIDE_IMBALANCE = 4.0
DEFAULT_SEED = 955339
N_BASELINE_ROWS = 8
N_CLEAN_CELLS = 6
N_GATE_REPEATS = 4
N_GATE_ROWS = N_CLEAN_CELLS * N_GATE_REPEATS
MIN_GATE_CLEAN_SUCCESSES = 20
PHYSICS_CLEAN_FAMILIES = (
    "F0_target_side_stagger",
    "F1_inner_panel_stagger",
    "F2_outer_panel_stagger",
)
REGRESSION_ONLY_FAMILIES = ("F3_aperture_side_stagger",)
V9_8_FORBIDDEN_KEYS = (
    "wrist_lag",
    "wrist_lag_m",
    "wrist_lag_neg",
    "wrist_lag_pos",
    "wrist_lag_neg_m",
    "wrist_lag_pos_m",
    "face_window",
    "face_window_m",
    "envelope_half_y",
    "envelope_half_y_m",
    "envelope_half_y_neg_m",
    "envelope_half_y_pos_m",
    "ceiling_envelope",
    "lag",
)

SOURCE_SUMMARY_RELATIVE = "diagnostics_output/pact_place_v95_raw_smoke/summary.json"
SCENE_XML_RELATIVE = (
    "submodules/molmospaces/molmo_spaces/data_generation/custom_scenes/"
    "pact_place_corridor_v5.xml"
)


def _vec3(value: Sequence[float], name: str) -> tuple[float, float, float]:
    values = tuple(float(item) for item in value)
    if len(values) != 3:
        raise ValueError(f"{name} must have three values: {value!r}")
    return values  # type: ignore[return-value]


def _round_m(value: float) -> float:
    return float(round(float(value), 9))


def pendant_aabb(fixture: dict[str, Any]) -> tuple[list[float], list[float]]:
    center = _vec3(fixture["center_m"], "center_m")
    half = _vec3(fixture["half_m"], "half_m")
    low = [center[i] - half[i] for i in range(3)]
    high = [center[i] + half[i] for i in range(3)]
    return low, high


def pendant_volume_m3(fixture: dict[str, Any]) -> float:
    half = _vec3(fixture["half_m"], "half_m")
    return float(8.0 * half[0] * half[1] * half[2])


def lane_y_limit_m(aperture_width_m: float = DEFAULT_APERTURE_WIDTH_M) -> float:
    return float(aperture_width_m) / 2.0 - APERTURE_EDGE_RESERVE_M


def validate_pendant_geometry(
    center_m: Sequence[float],
    half_m: Sequence[float],
    *,
    aperture_width_m: float = DEFAULT_APERTURE_WIDTH_M,
    **kwargs: Any,
) -> dict[str, Any]:
    """Validate one V9.9 ceiling-flush box. Rejects V9.8 lag/window kwargs."""
    forbidden = sorted(key for key in kwargs if key in V9_8_FORBIDDEN_KEYS)
    if forbidden:
        raise ValueError(
            "V9.9 geometry validation does not accept V9.8 lag/window inputs: "
            + ", ".join(forbidden)
        )
    if kwargs:
        raise ValueError(
            "V9.9 geometry validation received unsupported inputs: "
            + ", ".join(sorted(kwargs))
        )
    center = _vec3(center_m, "center_m")
    half = _vec3(half_m, "half_m")
    if any(value <= 0.0 for value in half):
        raise ValueError(f"pendant half extents must be positive: {half}")
    if abs(center[2] + half[2] - CEILING_TOP_Z_M) > 1e-9:
        raise ValueError("V9.9 pendant must be flush to the ceiling at z=1.515")
    bottom = center[2] - half[2]
    if not (
        BOTTOM_Z_BOUNDS_M[0] - 1e-9 <= bottom <= BOTTOM_Z_BOUNDS_M[1] + 1e-9
    ):
        raise ValueError(
            f"V9.9 pendant bottom {bottom:.9f} is outside {BOTTOM_Z_BOUNDS_M}"
        )
    if not (
        CENTER_X_BOUNDS_M[0] - 1e-9 <= center[0] <= CENTER_X_BOUNDS_M[1] + 1e-9
    ):
        raise ValueError(
            f"V9.9 pendant center x {center[0]:.9f} is outside {CENTER_X_BOUNDS_M}"
        )
    if not (
        CENTER_Y_BOUNDS_M[0] - 1e-9 <= center[1] <= CENTER_Y_BOUNDS_M[1] + 1e-9
    ):
        raise ValueError(
            f"V9.9 pendant center y {center[1]:.9f} is outside {CENTER_Y_BOUNDS_M}"
        )
    if half[0] not in HALF_X_CHOICES_M and not any(
        abs(half[0] - choice) <= 1e-9 for choice in HALF_X_CHOICES_M
    ):
        raise ValueError(f"V9.9 pendant half x {half[0]:.9f} is not in {HALF_X_CHOICES_M}")
    if not (
        HALF_Y_BOUNDS_M[0] - 1e-9 <= half[1] <= HALF_Y_BOUNDS_M[1] + 1e-9
    ):
        raise ValueError(
            f"V9.9 pendant half y {half[1]:.9f} is outside {HALF_Y_BOUNDS_M}"
        )
    x_low, x_high = center[0] - half[0], center[0] + half[0]
    if not (
        PENDANT_DEPTH_BOUNDS_M[0] - 1e-9 <= x_low
        and x_high <= PENDANT_DEPTH_BOUNDS_M[1] + 1e-9
    ):
        raise ValueError(
            f"V9.9 pendant x [{x_low:.9f}, {x_high:.9f}] is outside "
            f"{PENDANT_DEPTH_BOUNDS_M}"
        )
    y_limit = float(aperture_width_m) / 2.0
    y_low, y_high = center[1] - half[1], center[1] + half[1]
    if y_low < -y_limit - 1e-9 or y_high > y_limit + 1e-9:
        raise ValueError(
            f"V9.9 pendant y [{y_low:.9f}, {y_high:.9f}] exceeds aperture "
            f"±{y_limit:.9f}"
        )
    return {
        "center_m": [_round_m(value) for value in center],
        "half_m": [_round_m(value) for value in half],
        "bottom_z_m": _round_m(bottom),
        "top_z_m": _round_m(CEILING_TOP_Z_M),
        "aperture_width_m": float(aperture_width_m),
    }


def build_pendant_fixture(
    *,
    center_x_m: float,
    center_y_m: float,
    half_x_m: float,
    half_y_m: float,
    bottom_z_m: float,
    aperture_width_m: float = DEFAULT_APERTURE_WIDTH_M,
) -> dict[str, Any]:
    half_z = (CEILING_TOP_Z_M - float(bottom_z_m)) / 2.0
    center_z = float(bottom_z_m) + half_z
    center = (float(center_x_m), float(center_y_m), float(center_z))
    half = (float(half_x_m), float(half_y_m), float(half_z))
    validated = validate_pendant_geometry(
        center, half, aperture_width_m=aperture_width_m
    )
    return {
        "support": PENDANT_SUPPORT,
        "body": PENDANT_BODY,
        "geom": PENDANT_GEOM,
        "role": "ceiling_fixture",
        "center_m": validated["center_m"],
        "half_m": validated["half_m"],
        "bottom_z_m": validated["bottom_z_m"],
        "top_z_m": validated["top_z_m"],
        "lateral_lane_cost_m": 0.0,
        "identical_on_both_panel_sides": True,
        "active_on": ["inbound_empty", "outbound_loaded"],
    }


def empty_authorization() -> dict[str, bool]:
    return {
        "authorizes_collection": False,
        "authorizes_paired_screen": False,
        "authorizes_24_row_gate": False,
        "authorizes_s2b": False,
        "authorizes_human_review": False,
        "authorizes_training": False,
        "authorizes_eval": False,
    }


def six_cell_gate_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Score the pinned 24-row / six-cell engineering gate."""
    by_cell: dict[str, list[dict[str, Any]]] = {}
    infrastructure = 0
    for row in rows:
        if str(row.get("status") or "") == "infrastructure_failure":
            infrastructure += 1
        family = str(row.get("family") or row.get("layout_family_id") or "")
        side = str(row.get("intrusion_side") or "")
        by_cell.setdefault(f"{family}:{side}", []).append(row)
    clean_total = sum(1 for row in rows if bool(row.get("clean_success")))
    cells_with_clean = {
        key: sum(1 for row in items if bool(row.get("clean_success")))
        for key, items in by_cell.items()
    }
    required = {
        f"{family}:{side}"
        for family in PHYSICS_CLEAN_FAMILIES
        for side in ("left", "right")
    }
    missing = sorted(required - set(cells_with_clean))
    uncovered = sorted(
        key for key in required if cells_with_clean.get(key, 0) < 1
    )
    passed = (
        len(rows) == N_GATE_ROWS
        and infrastructure == 0
        and clean_total >= MIN_GATE_CLEAN_SUCCESSES
        and not missing
        and not uncovered
    )
    return {
        "n_rows": len(rows),
        "clean_successes": clean_total,
        "infrastructure_failures": infrastructure,
        "cells_with_clean": cells_with_clean,
        "missing_cells": missing,
        "uncovered_cells": uncovered,
        "passed": passed,
        "authorizes_collection": False,
        "note": (
            "six-cell engineering gate, not a multi-seed robustness estimate"
        ),
    }


def grasp_posture_preserved(
    baseline_q: Sequence[float],
    live_q: Sequence[float],
    *,
    max_error_rad: float = MAX_GRASP_JOINT_ERROR_RAD,
) -> dict[str, Any]:
    import numpy as np

    baseline = np.asarray(baseline_q, dtype=float).reshape(-1)
    live = np.asarray(live_q, dtype=float).reshape(-1)
    if baseline.shape != (7,) or live.shape != (7,):
        raise ValueError("canonical grasp posture must be seven arm joints")
    delta = np.abs(live - baseline)
    return {
        "max_abs_error_rad": float(np.max(delta)),
        "per_joint_abs_error_rad": [float(value) for value in delta],
        "preserved": bool(np.all(delta <= float(max_error_rad) + 1e-12)),
    }
