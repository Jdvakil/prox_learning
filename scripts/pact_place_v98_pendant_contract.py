#!/usr/bin/env python3
"""Frozen geometry and admission contract for the V9.8 ceiling pendant.

The pendant is a kinematic, symmetric ceiling fixture.  This module contains
only validation and construction helpers; it does not render, step physics,
or authorize collection.
"""

from __future__ import annotations

from typing import Any, Sequence

from run_pact_place_v96_cluster_causal_proximity import ADMISSION_FLOOR

ENVIRONMENT_VERSION = "pact_place_corridor_v9_8_pendant"
CONTRACT_VERSION = "pact_place_v9_8_pendant_v1"
SAMPLER_CLASS = "PactPlaceCorridorV98PendantSampler"

PENDANT_SUPPORT = "ceiling"
PENDANT_BODY = "pact_clutter_mount_ceiling"
PENDANT_GEOM = "pact_clutter_mount_ceiling_g"
CEILING_TOP_Z_M = 1.515
PENDANT_CENTER_X_M = 0.72
PENDANT_HALF_X_M = 0.10
PENDANT_CENTER_Y_M = 0.0
PENDANT_HALF_Z_NOMINAL_M = 0.1825
PENDANT_BOTTOM_Z_NOMINAL_M = 1.15
PENDANT_BOTTOM_Z_BOUNDS_M = (1.10, 1.20)
PENDANT_HALF_Y_BOUNDS_M = (0.12, 0.18)
PENDANT_DEPTH_BOUNDS_M = (0.58, 1.36)
PENDANT_LATERAL_LANE_COST_M = 0.0
MIN_ROUTE_INTRUSION_FRAMES = 100
MAX_SIDE_IMBALANCE = 4.0
N_EXPERT_ROWS = 24
MIN_CLEAN_SUCCESSES = 20
PHYSICS_CLEAN_FAMILIES = (
    "F0_target_side_stagger",
    "F1_inner_panel_stagger",
    "F2_outer_panel_stagger",
)


def _vec3(value: Sequence[float], name: str) -> tuple[float, float, float]:
    values = tuple(float(item) for item in value)
    if len(values) != 3:
        raise ValueError(f"{name} must have three values: {value!r}")
    return values  # type: ignore[return-value]


def validate_pendant_geometry(
    center_m: Sequence[float], half_m: Sequence[float]
) -> dict[str, Any]:
    """Validate one fixture against the bounds frozen before measurement."""
    center = _vec3(center_m, "center_m")
    half = _vec3(half_m, "half_m")
    if any(value <= 0.0 for value in half):
        raise ValueError(f"pendant half extents must be positive: {half}")
    if abs(center[1] - PENDANT_CENTER_Y_M) > 1e-9:
        raise ValueError("V9.8 pendant center y must be exactly 0.0")
    if abs(center[2] + half[2] - CEILING_TOP_Z_M) > 1e-9:
        raise ValueError("V9.8 pendant must be flush to the ceiling at z=1.515")
    bottom = center[2] - half[2]
    if not PENDANT_BOTTOM_Z_BOUNDS_M[0] - 1e-9 <= bottom <= PENDANT_BOTTOM_Z_BOUNDS_M[1] + 1e-9:
        raise ValueError(
            f"V9.8 pendant bottom {bottom:.9f} is outside "
            f"{PENDANT_BOTTOM_Z_BOUNDS_M}"
        )
    if not PENDANT_HALF_Y_BOUNDS_M[0] - 1e-9 <= half[1] <= PENDANT_HALF_Y_BOUNDS_M[1] + 1e-9:
        raise ValueError(
            f"V9.8 pendant half-width {half[1]:.9f} is outside "
            f"{PENDANT_HALF_Y_BOUNDS_M}"
        )
    if not (
        PENDANT_DEPTH_BOUNDS_M[0] - 1e-9
        <= center[0] - half[0]
        and center[0] + half[0]
        <= PENDANT_DEPTH_BOUNDS_M[1] + 1e-9
    ):
        raise ValueError("V9.8 pendant escapes the enclosure depth bounds")
    return {
        "center_m": list(center),
        "half_m": list(half),
        "bottom_z_m": float(bottom),
        "top_z_m": float(center[2] + half[2]),
        "lateral_lane_cost_m": PENDANT_LATERAL_LANE_COST_M,
    }


def build_pendant_fixture(
    *,
    bottom_z_m: float = PENDANT_BOTTOM_Z_NOMINAL_M,
    half_y_m: float = PENDANT_HALF_Y_BOUNDS_M[1],
    center_x_m: float = PENDANT_CENTER_X_M,
    half_x_m: float = PENDANT_HALF_X_M,
) -> dict[str, Any]:
    """Build an attached fixture while enforcing the exact ceiling relation."""
    bottom = float(bottom_z_m)
    half_y = float(half_y_m)
    center_x = float(center_x_m)
    half_x = float(half_x_m)
    half_z = (CEILING_TOP_Z_M - bottom) / 2.0
    center_z = (CEILING_TOP_Z_M + bottom) / 2.0
    geometry = validate_pendant_geometry(
        [center_x, PENDANT_CENTER_Y_M, center_z], [half_x, half_y, half_z]
    )
    return {
        "fixture_id": "ceiling_pendant",
        "support": PENDANT_SUPPORT,
        "center_m": geometry["center_m"],
        "half_m": geometry["half_m"],
        "bottom_z_m": geometry["bottom_z_m"],
        "top_z_m": geometry["top_z_m"],
        "appearance": "neutral_matte_structure",
        "siting_contract": CONTRACT_VERSION,
        "lateral_lane_cost_m": PENDANT_LATERAL_LANE_COST_M,
    }


def pendant_aabb(fixture: dict[str, Any]) -> tuple[list[float], list[float]]:
    """Return the world AABB pair used by the expert obstacle-speed law."""
    validated = validate_pendant_geometry(fixture["center_m"], fixture["half_m"])
    center = validated["center_m"]
    half = validated["half_m"]
    return (
        [float(c - h) for c, h in zip(center, half)],
        [float(c + h) for c, h in zip(center, half)],
    )


__all__ = [
    "ADMISSION_FLOOR",
    "CEILING_TOP_Z_M",
    "CONTRACT_VERSION",
    "ENVIRONMENT_VERSION",
    "MAX_SIDE_IMBALANCE",
    "MIN_CLEAN_SUCCESSES",
    "MIN_ROUTE_INTRUSION_FRAMES",
    "N_EXPERT_ROWS",
    "PENDANT_BODY",
    "PENDANT_BOTTOM_Z_BOUNDS_M",
    "PENDANT_CENTER_X_M",
    "PENDANT_CENTER_Y_M",
    "PENDANT_DEPTH_BOUNDS_M",
    "PENDANT_GEOM",
    "PENDANT_HALF_X_M",
    "PENDANT_HALF_Y_BOUNDS_M",
    "PENDANT_HALF_Z_NOMINAL_M",
    "PENDANT_LATERAL_LANE_COST_M",
    "PENDANT_SUPPORT",
    "PHYSICS_CLEAN_FAMILIES",
    "SAMPLER_CLASS",
    "build_pendant_fixture",
    "pendant_aabb",
    "validate_pendant_geometry",
]
