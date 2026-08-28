#!/usr/bin/env python3
"""Frozen geometry and admission contract for the V10 compound pendant.

Independent of V9.8. Must not import V9.8 lag, face-window, or ceiling-envelope
helpers. Does not reopen V9.9 or authorize collection.
"""

from __future__ import annotations

from typing import Any, Sequence

from run_pact_place_v96_cluster_causal_proximity import ADMISSION_FLOOR, CORRIDOR_LINKS

ENVIRONMENT_VERSION = "pact_place_corridor_v10_compound_pendant"
CONTRACT_VERSION = "pact_place_v10_compound_pendant_v1"
SAMPLER_CLASS = "PactPlaceCorridorV10CompoundPendantSampler"

PENDANT_SUPPORT = "ceiling"
PENDANT_BODY = "pact_clutter_mount_v10"
LOBE_GEOMS = (
    "pact_clutter_mount_v10_lobe_0_g",
    "pact_clutter_mount_v10_lobe_1_g",
    "pact_clutter_mount_v10_lobe_2_g",
)
STEM_GEOMS = (
    "pact_clutter_mount_v10_stem_0_g",
    "pact_clutter_mount_v10_stem_1_g",
    "pact_clutter_mount_v10_stem_2_g",
)
CROSSBAR_GEOM = "pact_clutter_mount_v10_crossbar_g"
ALL_GEOMS = LOBE_GEOMS + STEM_GEOMS + (CROSSBAR_GEOM,)
LEGACY_MOUNT_BODIES = (
    "pact_clutter_mount_wall_left",
    "pact_clutter_mount_wall_right",
    "pact_clutter_mount_ceiling",
)

CEILING_TOP_Z_M = 1.515
HOOD_TOP_BOTTOM_Z_M = 1.515
STEM_TOP_Z_M = 1.505
CROSSBAR_HEIGHT_M = 0.010
STEM_SQUARE_M = 0.006
STEM_HALF_M = STEM_SQUARE_M / 2.0
ASSEMBLY_PARK_XYZ_M = (4.0, 4.0, -2.0)

PENDANT_DEPTH_BOUNDS_M = (0.58, 1.36)
CENTER_X_BOUNDS_M = (0.60, 0.90)
CENTER_X_STEP_M = 0.02
HALF_X_CHOICES_M = (0.01, 0.02, 0.03)
CENTER_Y_ABS_BOUNDS_M = (0.12, 0.30)
CENTER_Y_STEP_M = 0.02
HALF_Y_CHOICES_M = (0.02, 0.04, 0.06, 0.08)
CENTER_Z_BOUNDS_M = (0.86, 0.98)
CENTER_Z_STEP_M = 0.02
HALF_Z_CHOICES_M = (0.02, 0.04, 0.06, 0.08)
LOBE_BOTTOM_MIN_M = 0.82
LOBE_TOP_MAX_M = 1.10
NEGATIVE_LOBE_MAX_Y_M = -0.08
POSITIVE_LOBE_MIN_Y_M = 0.08

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
N_NECESSITY_BITS = N_CLEAN_CELLS * 2
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
V99_RECONSTRUCTION_RELATIVE = (
    "diagnostics_output/pact_place_v99_baseline_reconstruction/reconstruction.json"
)
V99_SNAPSHOT_RELATIVE = "diagnostics_output/pact_place_v99_siting/snapshots/snapshots.json"
SCENE_XML_RELATIVE = (
    "submodules/molmospaces/molmo_spaces/data_generation/custom_scenes/"
    "pact_place_corridor_v10.xml"
)
V5_SCENE_XML_RELATIVE = (
    "submodules/molmospaces/molmo_spaces/data_generation/custom_scenes/"
    "pact_place_corridor_v5.xml"
)

HOOD_TOP_GEOM_NAME = "hood_top"
# hood_top: pos 0.95 0 1.53, size 0.42 0.46 0.015
HOOD_TOP_CENTER_M = (0.95, 0.0, 1.53)
HOOD_TOP_HALF_M = (0.42, 0.46, 0.015)
PLACE_V10_SCENE_SHA256 = (
    "360b1407a01d1447d8b440ade3115866399a1db09efc76321016aa3c04eaddf7"
)
V99_RECONSTRUCTION_SHA256 = (
    "ae2964c41ebd85ce61ac4d703d809a4198759a4c116728daa459d55d796eff1c"
)
V99_SNAPSHOT_SHA256 = (
    "0d6e61baeab68e645d6e04ce54a2406bc588f05e540b7441463f7c1e06af8465"
)
V99_SITING_SHA256 = (
    "71389801e8ba0663af68629234e5d767478fb7bbf452a1b333ac7963064a774f"
)
V99_SCOPED_CONCLUSION = (
    "no survivor in the registered fixed rectangular-box lattice"
)

# Independent-audit expectations. Recorded siting counts must be computed,
# then compared to these; a mismatch is a stop, not a rewrite of geometry.
AUDIT_ROBOT_TARGET_PREFILTER_COUNT = 8_554_036
AUDIT_PANEL_CLEAR_COUNT = 150_288
AUDIT_PANEL_CLEAR_UNION_COUNT = 1_779

IMPLEMENTATION_PATHS_V10 = (
    "scripts/pact_place_v10_compound_pendant_contract.py",
    "scripts/pact_place_v10_geometry.py",
    "scripts/pact_place_v10_exact.py",
    "scripts/pact_place_v10_catalog.py",
    "scripts/pact_place_v10_environment.py",
    "scripts/pact_place_v10_route.py",
    "scripts/pact_place_v10_scene.py",
    "scripts/pact_place_v10_runtime.py",
    "scripts/search_pact_place_v10_compound_pendant.py",
    "scripts/search_pact_place_v10_siting_v2.py",
    "scripts/search_pact_place_v10_route.py",
    "scripts/search_pact_place_v10_route_v2.py",
    "scripts/pact_geom_distance.py",
    "submodules/molmospaces/molmo_spaces/tasks/enclosure_reach.py",
    SCENE_XML_RELATIVE,
    "submodules/molmospaces/molmo_spaces/data_generation/custom_scenes/"
    "pact_place_corridor_v10_metadata.json",
)

# Invalid enumerator-era fixture. Intersects the opposite-side intrusion panel.
PROBE_V1_NEGATIVE_LOBE = {
    "center_m": (0.60, -0.26, 0.90),
    "half_m": (0.02, 0.02, 0.04),
}
PROBE_V1_POSITIVE_LOBE = {
    "center_m": (0.60, 0.20, 0.90),
    "half_m": (0.02, 0.08, 0.04),
}
PROBE_V1_STEM_Y_M = (-0.28, 0.28)

# Trust-anchor planning-probe-v2. Must pass robot/target exact plus live-scene
# panel/clutter/environment checks on all six cells.
PROBE_NEGATIVE_LOBE = {
    "center_m": (0.70, -0.18, 0.86),
    "half_m": (0.01, 0.04, 0.04),
}
PROBE_POSITIVE_LOBE = {
    "center_m": (0.70, 0.22, 0.86),
    "half_m": (0.01, 0.02, 0.02),
}
PROBE_STEM_Y_M = (-0.22, 0.24)

V1_SITING_RELATIVE = "diagnostics_output/pact_place_v10_siting/siting.json"
V1_PREFILTER_CATALOG_RELATIVE = (
    "diagnostics_output/pact_place_v10_siting/exact_survivors.npz"
)
V1_PREFILTER_CATALOG_SHA256 = (
    "63369af3552bbb806a61fea97d281011374ee25bb375004876704b920b6f3443"
)
V1_SITING_PAYLOAD_SHA256 = (
    "923c9380319b343e43e55f018080995db8a4b59e5a5f3cbd7f5d1a3be79d0eb6"
)
V2_SITING_RELATIVE = "diagnostics_output/pact_place_v10_siting_v2/siting.json"
V2_SITING_PAYLOAD_SHA256 = (
    "2e0b2a56bd4c22ecc920927dc149adf9c1bbc0d1d3ccbd3ee433ea450b187c1c"
)
V2_CATALOG_RELATIVE = "diagnostics_output/pact_place_v10_siting_v2/exact_survivors_v2.npz"
V2_CATALOG_SHA256 = (
    "b84e19bf269c39cd052551639c22d4cbb5b4348eaf6188663fae0659af824d6e"
)
V2_ATOMIC_SCORES_RELATIVE = (
    "diagnostics_output/pact_place_v10_siting_v2/atomic_component_env_scores.npz"
)
V2_ATOMIC_SCORES_SHA256 = (
    "fae3489eac562a830c46e3f4ceb0e58b370fe21eae0429ec4729a3f8e0ee2a80"
)
V2_PREFILTER_INDICES_RELATIVE = (
    "diagnostics_output/pact_place_v10_siting_v2/survivor_prefilter_indices.npy"
)
V2_PREFILTER_INDICES_SHA256 = (
    "9a5f5c6ffe0d730ac3b0667f48f9b8bf34cb28a2d21aaabdbcf78679f24b3ff4"
)
EXPECTED_FULL_ENVIRONMENT_SURVIVOR_COUNT = 150_288
EXPECTED_UNIQUE_UNION_COUNT = 1_779
SITING_SCHEMA_V2 = "pact_place_v10_siting_v2"
CATALOG_SCHEMA_V2 = "pact_place_v10_survivor_catalog_v2"
ROUTE_SCHEMA = "pact_place_v10_route_v1"
ROUTE_RELATIVE = "diagnostics_output/pact_place_v10_route/route.json"
ROUTE_V1_PAYLOAD_SHA256 = (
    "c0f1b35084d6950a88531c45e6805b06437add31c82ca5fd68bb5da4f5de3ff7"
)
ROUTE_V1_SCOPED_CONCLUSION = (
    "No route survivor under the registered V9.9 contiguous-group-freeze "
    "route primitive."
)
ROUTE_SCHEMA_V2 = "pact_place_v10_route_v2_endpoint_only"
ROUTE_V2_RELATIVE = "diagnostics_output/pact_place_v10_route_v2/route.json"
ROUTE_V2_GEOMETRY_RELATIVE = (
    "diagnostics_output/pact_place_v10_route_v2/geometry.json"
)
# Independently reproduced; a mismatch is a stop, not a rewrite of geometry.
AUDIT_ROUTE_V2_UNION_DIRECTION_EVALUATIONS = 21_348
AUDIT_ROUTE_V2_ENDPOINT_ONLY_GEOMETRY_EVALUATIONS = 17_826
AUDIT_ROUTE_V2_ENDPOINT_ONLY_GEOMETRY_UNIONS = 1_032
ENDPOINT_ONLY_PRIMITIVE = "endpoint_only"
GROUP_FREEZE_PRIMITIVE = "contiguous_group_freeze"
EMPIRICAL_LIVE_CONTACT_V1 = "empirical_live_contact_v1"
FROZEN_ENDPOINT_ATOL_M = 1e-12
ROUTE_SHORTLIST_AMENDMENT = "too_many_morphologies_for_signal_screen"
SIGNAL_SCREEN_LIMIT_UNREGISTERED = "signal_screen_limit_unregistered"
# No complete-signal-screen cap is registered for V10. Do not invent one
# after seeing route results.
REGISTERED_COMPLETE_SIGNAL_SCREEN_LIMIT = None
NOMINAL_PERTURBATION_INDEX = -1


def _vec3(value: Sequence[float], name: str) -> tuple[float, float, float]:
    values = tuple(float(item) for item in value)
    if len(values) != 3:
        raise ValueError(f"{name} must have three values: {value!r}")
    return values  # type: ignore[return-value]


def round_m(value: float) -> float:
    return float(round(float(value), 9))


def round_vec(value: Sequence[float]) -> list[float]:
    return [round_m(item) for item in _vec3(value, "vec")]


def lane_y_limit_m(aperture_width_m: float = DEFAULT_APERTURE_WIDTH_M) -> float:
    return float(aperture_width_m) / 2.0 - APERTURE_EDGE_RESERVE_M


def component_aabb(center_m: Sequence[float], half_m: Sequence[float]) -> tuple[list[float], list[float]]:
    center = _vec3(center_m, "center_m")
    half = _vec3(half_m, "half_m")
    return (
        [round_m(center[i] - half[i]) for i in range(3)],
        [round_m(center[i] + half[i]) for i in range(3)],
    )


def component_volume_m3(half_m: Sequence[float]) -> float:
    half = _vec3(half_m, "half_m")
    return float(8.0 * half[0] * half[1] * half[2])


def v10_implementation_hashes() -> dict[str, str]:
    from pathlib import Path

    from pact_place_corridor_contract import sha256_file

    root = Path(__file__).resolve().parents[1]
    return {path: sha256_file(root / path) for path in IMPLEMENTATION_PATHS_V10}


def empty_authorization() -> dict[str, bool]:
    return {
        "authorizes_collection": False,
        "authorizes_paired_screen": False,
        "authorizes_24_row_gate": False,
        "authorizes_s2b": False,
        "authorizes_human_review": False,
        "authorizes_training": False,
        "authorizes_eval": False,
        "environment_qualified": False,
    }


def apply_human_environment_qualification(
    document: dict[str, Any],
    *,
    approved: bool,
) -> dict[str, Any]:
    """Human review may set only environment_qualified. Collection stays false."""
    updated = dict(document)
    updated.update(empty_authorization())
    updated["environment_qualified"] = bool(approved)
    updated["authorizes_collection"] = False
    updated["authorizes_training"] = False
    updated["authorizes_eval"] = False
    return updated


def reject_v98_kwargs(kwargs: dict[str, Any]) -> None:
    forbidden = sorted(key for key in kwargs if key in V9_8_FORBIDDEN_KEYS)
    if forbidden:
        raise ValueError(
            "V10 geometry validation does not accept V9.8 lag/window inputs: "
            + ", ".join(forbidden)
        )
    if kwargs:
        raise ValueError(
            "V10 geometry validation received unsupported inputs: "
            + ", ".join(sorted(kwargs))
        )


def six_cell_gate_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    from pact_place_v99_pendant_contract import six_cell_gate_summary as _v99

    summary = _v99(rows)
    summary["authorizes_collection"] = False
    return summary


def grasp_posture_preserved(
    baseline_q: Sequence[float],
    live_q: Sequence[float],
    *,
    max_error_rad: float = MAX_GRASP_JOINT_ERROR_RAD,
) -> dict[str, Any]:
    from pact_place_v99_pendant_contract import grasp_posture_preserved as _v99

    return _v99(baseline_q, live_q, max_error_rad=max_error_rad)
