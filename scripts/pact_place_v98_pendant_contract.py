#!/usr/bin/env python3
"""Frozen geometry and admission contract for the V9.8 ceiling pendant.

The pendant is a kinematic ceiling fixture.  This module contains only
validation and construction helpers; it does not render, step physics, or
authorize collection.

v2 supersedes the centred y=0 / half_y in (0.12, 0.18) bounds after the
wrist-lag measurement showed a centred pendant is unavoidable: the wrist
trails the TCP toward the centreline by 0.208 m on −y bows and 0.108 m on
+y bows.
"""

from __future__ import annotations

from typing import Any, Sequence

from run_pact_place_v96_cluster_causal_proximity import ADMISSION_FLOOR

ENVIRONMENT_VERSION = "pact_place_corridor_v9_8_pendant"
CONTRACT_VERSION = "pact_place_v9_8_pendant_v2"
SAMPLER_CLASS = "PactPlaceCorridorV98PendantSampler"

PENDANT_SUPPORT = "ceiling"
PENDANT_BODY = "pact_clutter_mount_ceiling"
PENDANT_GEOM = "pact_clutter_mount_ceiling_g"
CEILING_TOP_Z_M = 1.515
PENDANT_CENTER_X_M = 0.72
PENDANT_HALF_X_M = 0.10
# v1 required exactly 0.0. v2 offsets toward +y so the wrist lags *away*
# from the pendant on both panel sides.
PENDANT_CENTER_Y_M = 0.100
PENDANT_CENTER_Y_BOUNDS_M = (0.080, 0.120)
PENDANT_HALF_Z_NOMINAL_M = 0.1825
PENDANT_BOTTOM_Z_NOMINAL_M = 1.15
PENDANT_BOTTOM_Z_BOUNDS_M = (1.10, 1.20)
PENDANT_HALF_Y_BOUNDS_M = (0.040, 0.080)
PENDANT_HALF_Y_WIDE_M = 0.056
PENDANT_HALF_Y_CONS_M = 0.045
PENDANT_DEPTH_BOUNDS_M = (0.58, 1.36)
PENDANT_LATERAL_LANE_COST_M = 0.0
MIN_ROUTE_INTRUSION_FRAMES = 100
MAX_SIDE_IMBALANCE = 4.0
DEFAULT_APERTURE_WIDTH_M = 0.85
MIN_DETOUR_SLACK_M = 0.020
N_EXPERT_ROWS = 24
MIN_CLEAN_SUCCESSES = 20
PHYSICS_CLEAN_FAMILIES = (
    "F0_target_side_stagger",
    "F1_inner_panel_stagger",
    "F2_outer_panel_stagger",
)

# Wrist lag toward the centreline, measured by mj_forward on retained qpos
# from the centred-pendant bowed paired runs. Step 5 re-measures on offset
# trajectories; these numbers are the design assumption, not a fixed robot
# property.
WRIST_LAG_NEG_M = 0.208
WRIST_LAG_POS_M = 0.108
WRIST_LAG_PROVENANCE = {
    "method": "mj_forward on retained qpos; no physics step; no render",
    "instrument": "scripts/measure_pact_place_v98_wrist_lag.py",
    "source_runs": [
        "diagnostics_output/pact_place_v98_paired_halfy016/",
        "diagnostics_output/pact_place_v98_paired_halfy014/",
        "diagnostics_output/pact_place_v98_paired_halfy012/",
    ],
    "left_panel_minus_y_bow_m": "0.198-0.208",
    "right_panel_plus_y_bow_m": "0.107-0.108",
    "design_values_m": {"lag_neg": WRIST_LAG_NEG_M, "lag_pos": WRIST_LAG_POS_M},
    "note": (
        "Wrist trails TCP toward the centreline. At TCP y=-0.268 the wrist "
        "was at y=-0.061. Configuration-dependent: a different bow is a "
        "different arm pose."
    ),
}

OFFSET_CANDIDATES = {
    "wide": {
        "center_y_m": 0.100,
        "half_y_m": PENDANT_HALF_Y_WIDE_M,
        "bottom_z_m": PENDANT_BOTTOM_Z_BOUNDS_M[0],
    },
    "cons": {
        "center_y_m": 0.100,
        "half_y_m": PENDANT_HALF_Y_CONS_M,
        "bottom_z_m": PENDANT_BOTTOM_Z_BOUNDS_M[0],
    },
}

CONTRACT_SUPERSESSION = {
    "from_version": "pact_place_v9_8_pendant_v1",
    "to_version": CONTRACT_VERSION,
    "cause": "wrist_lag_measurement",
    "wrist_lag_provenance": WRIST_LAG_PROVENANCE,
    "changes": [
        "PENDANT_CENTER_Y_M was exactly 0.0; v2 allows a bounded +y offset.",
        "PENDANT_HALF_Y_BOUNDS_M was (0.12, 0.18); v2 is (0.040, 0.080).",
        "Ceiling-fixture bow envelope is side-dependent (lag + 4 mm); wall fixtures stay at 0.10.",
    ],
}


def _vec3(value: Sequence[float], name: str) -> tuple[float, float, float]:
    values = tuple(float(item) for item in value)
    if len(values) != 3:
        raise ValueError(f"{name} must have three values: {value!r}")
    return values  # type: ignore[return-value]


def fixture_bow_policy_constants() -> dict[str, float]:
    """Read the wall-fixture bow constants from the expert policy, not copies."""
    from molmo_spaces.tasks.enclosure_reach import PactPlaceCorridorPolicy

    return {
        "aperture_width_m": DEFAULT_APERTURE_WIDTH_M,
        "envelope_half_y_m": float(
            PactPlaceCorridorPolicy.MOUNTED_FIXTURE_ENVELOPE_HALF_Y
        ),
        "safe_gap_m": float(PactPlaceCorridorPolicy.MOUNTED_FIXTURE_SAFE_GAP),
        "aperture_edge_reserve_m": float(
            PactPlaceCorridorPolicy.APERTURE_EDGE_RESERVE
        ),
    }


def ceiling_fixture_bow_policy_constants() -> dict[str, float]:
    """Live V9.8 ceiling-fixture envelopes, gap, and aperture constants."""
    from molmo_spaces.tasks.enclosure_reach import PactPlaceCorridorPolicy

    return {
        "aperture_width_m": DEFAULT_APERTURE_WIDTH_M,
        "envelope_half_y_neg_m": float(
            PactPlaceCorridorPolicy.CEILING_FIXTURE_ENVELOPE_HALF_Y_NEG
        ),
        "envelope_half_y_pos_m": float(
            PactPlaceCorridorPolicy.CEILING_FIXTURE_ENVELOPE_HALF_Y_POS
        ),
        "safe_gap_m": float(PactPlaceCorridorPolicy.MOUNTED_FIXTURE_SAFE_GAP),
        "aperture_edge_reserve_m": float(
            PactPlaceCorridorPolicy.APERTURE_EDGE_RESERVE
        ),
        "wrist_lag_neg_m": float(WRIST_LAG_NEG_M),
        "wrist_lag_pos_m": float(WRIST_LAG_POS_M),
    }


def pendant_admissible_faces_m(
    constants: dict[str, float] | None = None,
) -> tuple[float, float]:
    """(a_min, b_max) from live envelopes: 2·ENV − (ap/2 − reserve − gap)."""
    values = constants or ceiling_fixture_bow_policy_constants()
    gap = float(values["safe_gap_m"])
    reserve = float(values["aperture_edge_reserve_m"])
    aperture = float(values["aperture_width_m"])
    room = aperture / 2.0 - reserve - gap
    a_min = 2.0 * float(values["envelope_half_y_neg_m"]) - room
    b_max = room - 2.0 * float(values["envelope_half_y_pos_m"])
    return float(a_min), float(b_max)


def pendant_faces_m(center_y_m: float, half_y_m: float) -> tuple[float, float]:
    return float(center_y_m) - float(half_y_m), float(center_y_m) + float(half_y_m)


def validate_pendant_geometry(
    center_m: Sequence[float], half_m: Sequence[float]
) -> dict[str, Any]:
    """Validate one fixture against the v2 offset bounds and face window."""
    center = _vec3(center_m, "center_m")
    half = _vec3(half_m, "half_m")
    if any(value <= 0.0 for value in half):
        raise ValueError(f"pendant half extents must be positive: {half}")
    if not (
        PENDANT_CENTER_Y_BOUNDS_M[0] - 1e-9
        <= center[1]
        <= PENDANT_CENTER_Y_BOUNDS_M[1] + 1e-9
    ):
        raise ValueError(
            f"V9.8 pendant center y {center[1]:.9f} is outside "
            f"{PENDANT_CENTER_Y_BOUNDS_M}"
        )
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
    a_face, b_face = pendant_faces_m(center[1], half[1])
    a_min, b_max = pendant_admissible_faces_m()
    if a_face + 1e-9 < a_min or b_face - 1e-9 > b_max:
        raise ValueError(
            f"V9.8 pendant faces [{a_face:.9f}, {b_face:.9f}] escape the "
            f"admissible window [{a_min:.9f}, {b_max:.9f}] from the live "
            "ceiling-fixture envelopes"
        )
    return {
        "center_m": list(center),
        "half_m": list(half),
        "bottom_z_m": float(bottom),
        "top_z_m": float(center[2] + half[2]),
        "lateral_lane_cost_m": PENDANT_LATERAL_LANE_COST_M,
        "neg_face_y_m": float(a_face),
        "pos_face_y_m": float(b_face),
        "admissible_neg_face_y_m": float(a_min),
        "admissible_pos_face_y_m": float(b_max),
        "siting_contract": CONTRACT_VERSION,
    }


def build_pendant_fixture(
    *,
    bottom_z_m: float = PENDANT_BOTTOM_Z_NOMINAL_M,
    half_y_m: float = PENDANT_HALF_Y_WIDE_M,
    center_x_m: float = PENDANT_CENTER_X_M,
    center_y_m: float = PENDANT_CENTER_Y_M,
    half_x_m: float = PENDANT_HALF_X_M,
) -> dict[str, Any]:
    """Build an attached fixture while enforcing the exact ceiling relation."""
    bottom = float(bottom_z_m)
    half_y = float(half_y_m)
    center_x = float(center_x_m)
    center_y = float(center_y_m)
    half_x = float(half_x_m)
    half_z = (CEILING_TOP_Z_M - bottom) / 2.0
    center_z = (CEILING_TOP_Z_M + bottom) / 2.0
    geometry = validate_pendant_geometry(
        [center_x, center_y, center_z], [half_x, half_y, half_z]
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


def fixture_bow_lateral_limit_m(
    constants: dict[str, float] | None = None,
) -> float:
    values = constants or fixture_bow_policy_constants()
    envelope = float(
        values.get("envelope_half_y_m", values.get("envelope_half_y_neg_m"))
    )
    return (
        float(values["aperture_width_m"]) / 2.0
        - envelope
        - float(values["aperture_edge_reserve_m"])
    )


def fixture_bow_waypoint_abs_y_m(
    half_y_m: float, constants: dict[str, float] | None = None
) -> float:
    """|waypoint_y| for a y=0 pendant: safe_gap + envelope + half_y."""
    values = constants or fixture_bow_policy_constants()
    return (
        float(values["safe_gap_m"])
        + float(values["envelope_half_y_m"])
        + float(half_y_m)
    )


def fixture_bow_detour_slack_m(
    half_y_m: float, constants: dict[str, float] | None = None
) -> float:
    """lateral_limit − |waypoint_y|. Zero at half_y=0.18; 20 mm at 0.16."""
    return fixture_bow_lateral_limit_m(constants) - fixture_bow_waypoint_abs_y_m(
        half_y_m, constants
    )


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
    "CONTRACT_SUPERSESSION",
    "CONTRACT_VERSION",
    "DEFAULT_APERTURE_WIDTH_M",
    "ENVIRONMENT_VERSION",
    "MAX_SIDE_IMBALANCE",
    "MIN_CLEAN_SUCCESSES",
    "MIN_DETOUR_SLACK_M",
    "MIN_ROUTE_INTRUSION_FRAMES",
    "N_EXPERT_ROWS",
    "OFFSET_CANDIDATES",
    "PENDANT_BODY",
    "PENDANT_BOTTOM_Z_BOUNDS_M",
    "PENDANT_CENTER_X_M",
    "PENDANT_CENTER_Y_BOUNDS_M",
    "PENDANT_CENTER_Y_M",
    "PENDANT_DEPTH_BOUNDS_M",
    "PENDANT_GEOM",
    "PENDANT_HALF_X_M",
    "PENDANT_HALF_Y_BOUNDS_M",
    "PENDANT_HALF_Y_CONS_M",
    "PENDANT_HALF_Y_WIDE_M",
    "PENDANT_HALF_Z_NOMINAL_M",
    "PENDANT_LATERAL_LANE_COST_M",
    "PENDANT_SUPPORT",
    "PHYSICS_CLEAN_FAMILIES",
    "SAMPLER_CLASS",
    "WRIST_LAG_NEG_M",
    "WRIST_LAG_POS_M",
    "WRIST_LAG_PROVENANCE",
    "build_pendant_fixture",
    "ceiling_fixture_bow_policy_constants",
    "fixture_bow_detour_slack_m",
    "fixture_bow_lateral_limit_m",
    "fixture_bow_policy_constants",
    "fixture_bow_waypoint_abs_y_m",
    "pendant_aabb",
    "pendant_admissible_faces_m",
    "pendant_faces_m",
    "validate_pendant_geometry",
]
