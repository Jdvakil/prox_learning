#!/usr/bin/env python3
"""V10 full-route lane primitive over the union of active assembly components."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from pact_place_v10_compound_pendant_contract import (
    DEFAULT_APERTURE_WIDTH_M,
    EMPIRICAL_LIVE_CONTACT_V1,
    ENDPOINT_ONLY_PRIMITIVE,
    FROZEN_ENDPOINT_ATOL_M,
    GROUP_FREEZE_PRIMITIVE,
    SLAB_PADDINGS_M,
)
from pact_place_v10_geometry import union_fixture
from pact_place_v99_route import (
    PERTURBATION_CORNERS,
    apply_constant_lane,
    densify_path,
    interpolate_y_at_x,
    lane_y_grid,
    min_abs_detour_in_slab_m,
    named_lane_segments,
    panel_lane_sign,
    perturbation_corners,
    plan_lane as plan_lane_single,
    rotation_angle_rad,
    travel_sign_through_slab,
)


def assembly_slab_x_bounds(
    assembly: dict[str, Any], padding_m: float
) -> tuple[float, float, float, float]:
    fixture = union_fixture(assembly)
    center = fixture["center_m"]
    half = fixture["half_m"]
    physical_lo = float(center[0] - half[0])
    physical_hi = float(center[0] + half[0])
    return (
        physical_lo,
        physical_hi,
        physical_lo - float(padding_m),
        physical_hi + float(padding_m),
    )


def plan_lane(
    stock_positions: np.ndarray,
    stock_rotations: np.ndarray,
    *,
    assembly: dict[str, Any],
    panel_side: str,
    lane_y_m: float,
    padding_m: float,
    aperture_width_m: float = DEFAULT_APERTURE_WIDTH_M,
    freeze_start: bool = False,
    freeze_final: bool = True,
) -> dict[str, Any]:
    planned = plan_lane_single(
        stock_positions,
        stock_rotations,
        fixture=union_fixture(assembly),
        panel_side=panel_side,
        lane_y_m=lane_y_m,
        padding_m=padding_m,
        aperture_width_m=aperture_width_m,
        freeze_start=freeze_start,
        freeze_final=freeze_final,
    )
    dense_stock_p, dense_stock_r = densify_path(
        np.asarray(stock_positions, dtype=np.float64),
        np.asarray(stock_rotations, dtype=np.float64),
    )
    planned["assembly_id"] = assembly.get("assembly_id")
    planned["union_fixture"] = union_fixture(assembly)
    planned["stock_positions_m"] = dense_stock_p
    planned["stock_rotations"] = dense_stock_r
    planned["rewrite_primitive"] = GROUP_FREEZE_PRIMITIVE
    return planned


def plan_lane_at_parameters(
    stock_positions: np.ndarray,
    stock_rotations: np.ndarray,
    *,
    fixture: dict[str, Any],
    panel_side: str,
    lane_y_m: float,
    padding_m: float,
    entry_x_m: float,
    exit_x_m: float,
    aperture_width_m: float = DEFAULT_APERTURE_WIDTH_M,
    freeze_start: bool = False,
    freeze_final: bool = True,
) -> dict[str, Any]:
    """Rewrite a stock path onto an explicit lane/entry/exit.

    Perturbed corners must pass the already-offset lane-y, entry-x, and exit-x
    rather than re-deriving them from the union box.
    """
    from pact_place_v99_route import (
        apply_constant_lane,
        densify_path,
        lane_inside_aperture,
        min_abs_detour_in_slab_m,
        panel_lane_sign,
        slab_x_bounds,
        travel_sign_through_slab,
    )

    stock_p = np.asarray(stock_positions, dtype=np.float64)
    stock_r = np.asarray(stock_rotations, dtype=np.float64)
    sign = panel_lane_sign(panel_side)
    clipped = not lane_inside_aperture(lane_y_m, aperture_width_m=aperture_width_m)
    wrong_way = float(lane_y_m) * sign <= 0.0
    physical_lo, physical_hi, padded_lo, padded_hi = slab_x_bounds(fixture, padding_m)
    travel = travel_sign_through_slab(stock_p, padded_lo, padded_hi)
    stock_dense_p, stock_dense_r = densify_path(stock_p, stock_r)
    planned_p, planned_r, rewritten = apply_constant_lane(
        stock_dense_p,
        stock_dense_r,
        lane_y=float(lane_y_m),
        entry_x=float(entry_x_m),
        exit_x=float(exit_x_m),
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
        "entry_x_m": float(entry_x_m),
        "exit_x_m": float(exit_x_m),
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
        "stock_rotations": stock_dense_r,
        "orientation_source": "stock_interpolated",
        "extra_orientation_change": False,
        "rewritten_samples": int(np.sum(rewritten)),
        "accepted_geometry": bool(
            (not clipped)
            and (not wrong_way)
            and detour["meets_minimum"]
            and any(abs(float(padding_m) - value) <= 1e-9 for value in SLAB_PADDINGS_M)
        ),
        "rewrite_primitive": GROUP_FREEZE_PRIMITIVE,
    }


def apply_constant_lane_endpoint_only(
    positions: np.ndarray,
    rotations: np.ndarray,
    *,
    lane_y: float,
    entry_x: float,
    exit_x: float,
    freeze_start: bool = False,
    freeze_final: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rewrite in-slab samples to ``lane_y``, freezing only requested endpoints.

    This is the registered V10 route-v2 primitive. It must not be used as a
    silent replacement for V9.9 ``apply_constant_lane``, which still suppresses
    an entire contiguous in-slab group that contains a frozen endpoint.
    """
    source = np.asarray(positions, dtype=np.float64)
    planned = source.copy()
    rotations = np.asarray(rotations, dtype=np.float64).copy()
    x_lo, x_hi = min(entry_x, exit_x), max(entry_x, exit_x)
    inside = (planned[:, 0] >= x_lo - 1e-12) & (planned[:, 0] <= x_hi + 1e-12)
    frozen: set[int] = set()
    if freeze_final and len(planned):
        frozen.add(len(planned) - 1)
    if freeze_start and len(planned):
        frozen.add(0)
    rewritten = np.zeros(len(planned), dtype=bool)
    for index, flag in enumerate(inside.tolist()):
        if not flag or index in frozen:
            continue
        planned[index, 1] = float(lane_y)
        rewritten[index] = True
    for index in frozen:
        planned[index] = source[index]
        rotations[index] = np.asarray(rotations[index], dtype=np.float64)
    return planned, rotations, rewritten


def path_step_limits(
    positions: np.ndarray,
    rotations: np.ndarray,
    *,
    max_translation_m: float = 0.005,
    max_rotation_deg: float = 2.0,
) -> dict[str, Any]:
    """Return per-step translation/rotation maxima after densification."""
    import math

    positions = np.asarray(positions, dtype=np.float64)
    rotations = np.asarray(rotations, dtype=np.float64)
    if len(positions) < 2:
        return {
            "max_translation_m": 0.0,
            "max_rotation_deg": 0.0,
            "within_limits": True,
            "n_steps": 0,
        }
    translations = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    angles = [
        math.degrees(rotation_angle_rad(rotations[index], rotations[index + 1]))
        for index in range(len(rotations) - 1)
    ]
    max_t = float(np.max(translations)) if len(translations) else 0.0
    max_a = float(max(angles)) if angles else 0.0
    return {
        "max_translation_m": max_t,
        "max_rotation_deg": max_a,
        "within_limits": bool(
            max_t <= float(max_translation_m) + 1e-12
            and max_a <= float(max_rotation_deg) + 1e-12
        ),
        "n_steps": int(len(translations)),
    }


def frozen_endpoint_preserved(
    planned_positions: np.ndarray,
    planned_rotations: np.ndarray,
    stock_positions: np.ndarray,
    stock_rotations: np.ndarray,
    *,
    freeze_start: bool,
    freeze_final: bool,
    atol: float = FROZEN_ENDPOINT_ATOL_M,
) -> dict[str, Any]:
    planned_p = np.asarray(planned_positions, dtype=np.float64)
    planned_r = np.asarray(planned_rotations, dtype=np.float64)
    stock_p = np.asarray(stock_positions, dtype=np.float64)
    stock_r = np.asarray(stock_rotations, dtype=np.float64)
    start_ok = True
    final_ok = True
    if freeze_start:
        start_ok = bool(
            np.allclose(planned_p[0], stock_p[0], atol=atol)
            and np.allclose(planned_r[0], stock_r[0], atol=atol)
        )
    if freeze_final:
        final_ok = bool(
            np.allclose(planned_p[-1], stock_p[-1], atol=atol)
            and np.allclose(planned_r[-1], stock_r[-1], atol=atol)
        )
    return {
        "start_preserved": start_ok,
        "final_preserved": final_ok,
        "preserved": bool(start_ok and final_ok),
    }


REGISTERED_REWRITE_PRIMITIVES = frozenset(
    {ENDPOINT_ONLY_PRIMITIVE, GROUP_FREEZE_PRIMITIVE}
)
REGISTERED_QUALIFICATION_MODES = frozenset({EMPIRICAL_LIVE_CONTACT_V1})


def resolve_v10_runtime_route(route: dict[str, Any] | None) -> dict[str, Any]:
    """Dispatch historical group-freeze vs endpoint-only empirical rows.

    Rows without the new markers keep contiguous-group-freeze and the offline
    strict-environment preclearance. Unregistered bypass markers are refused.
    """
    payload = dict(route or {})
    primitive = payload.get("rewrite_primitive")
    mode = payload.get("qualification_mode")
    if primitive is None or primitive == "":
        primitive = GROUP_FREEZE_PRIMITIVE
    if primitive not in REGISTERED_REWRITE_PRIMITIVES:
        raise ValueError(f"unregistered V10 rewrite primitive: {primitive!r}")
    if mode is None or mode == "":
        mode = None
    elif mode not in REGISTERED_QUALIFICATION_MODES:
        raise ValueError(f"unregistered V10 qualification mode: {mode!r}")
    if mode == EMPIRICAL_LIVE_CONTACT_V1 and primitive != ENDPOINT_ONLY_PRIMITIVE:
        raise ValueError(
            "empirical_live_contact_v1 requires rewrite_primitive=endpoint_only"
        )
    return {
        "rewrite_primitive": str(primitive),
        "qualification_mode": mode,
        "use_endpoint_only": primitive == ENDPOINT_ONLY_PRIMITIVE,
        "skip_offline_strict_environment": mode == EMPIRICAL_LIVE_CONTACT_V1,
    }


def _plan_lane_with_rewrite(
    stock_positions: np.ndarray,
    stock_rotations: np.ndarray,
    *,
    fixture: dict[str, Any],
    panel_side: str,
    lane_y_m: float,
    padding_m: float,
    entry_x_m: float | None,
    exit_x_m: float | None,
    aperture_width_m: float,
    freeze_start: bool,
    freeze_final: bool,
    rewrite,
    primitive: str,
) -> dict[str, Any]:
    from pact_place_v99_route import (
        entry_exit_x,
        lane_inside_aperture,
        slab_x_bounds,
    )

    stock_p = np.asarray(stock_positions, dtype=np.float64)
    stock_r = np.asarray(stock_rotations, dtype=np.float64)
    sign = panel_lane_sign(panel_side)
    clipped = not lane_inside_aperture(lane_y_m, aperture_width_m=aperture_width_m)
    wrong_way = float(lane_y_m) * sign <= 0.0
    physical_lo, physical_hi, padded_lo, padded_hi = slab_x_bounds(fixture, padding_m)
    travel = travel_sign_through_slab(stock_p, padded_lo, padded_hi)
    if entry_x_m is None or exit_x_m is None:
        entry_x, exit_x = entry_exit_x(fixture, padding_m, travel)
    else:
        entry_x, exit_x = float(entry_x_m), float(exit_x_m)
    stock_dense_p, stock_dense_r = densify_path(stock_p, stock_r)
    planned_p, planned_r, rewritten = rewrite(
        stock_dense_p,
        stock_dense_r,
        lane_y=float(lane_y_m),
        entry_x=float(entry_x),
        exit_x=float(exit_x),
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
    steps = path_step_limits(planned_p, planned_r)
    endpoints = frozen_endpoint_preserved(
        planned_p,
        planned_r,
        stock_dense_p,
        stock_dense_r,
        freeze_start=freeze_start,
        freeze_final=freeze_final,
    )
    continuous = bool(steps["within_limits"])
    accepted = bool(
        (not clipped)
        and (not wrong_way)
        and detour["meets_minimum"]
        and continuous
        and endpoints["preserved"]
        and any(abs(float(padding_m) - value) <= 1e-9 for value in SLAB_PADDINGS_M)
        and int(detour.get("missing_stock_x", 0)) == 0
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
        "stock_rotations": stock_dense_r,
        "orientation_source": "stock_interpolated",
        "extra_orientation_change": False,
        "rewritten_samples": int(np.sum(rewritten)),
        "rewritten_mask": rewritten,
        "accepted_geometry": accepted,
        "rewrite_primitive": primitive,
        "path_steps": steps,
        "frozen_endpoints": endpoints,
        "continuous_after_densify": continuous,
        "perturbation_corners": perturbation_corners(lane_y_m, entry_x, exit_x),
    }


def plan_lane_endpoint_only(
    stock_positions: np.ndarray,
    stock_rotations: np.ndarray,
    *,
    assembly: dict[str, Any] | None = None,
    fixture: dict[str, Any] | None = None,
    panel_side: str,
    lane_y_m: float,
    padding_m: float,
    aperture_width_m: float = DEFAULT_APERTURE_WIDTH_M,
    freeze_start: bool = False,
    freeze_final: bool = True,
) -> dict[str, Any]:
    if fixture is None:
        if assembly is None:
            raise ValueError("plan_lane_endpoint_only requires assembly or fixture")
        fixture = union_fixture(assembly)
    planned = _plan_lane_with_rewrite(
        stock_positions,
        stock_rotations,
        fixture=fixture,
        panel_side=panel_side,
        lane_y_m=lane_y_m,
        padding_m=padding_m,
        entry_x_m=None,
        exit_x_m=None,
        aperture_width_m=aperture_width_m,
        freeze_start=freeze_start,
        freeze_final=freeze_final,
        rewrite=apply_constant_lane_endpoint_only,
        primitive=ENDPOINT_ONLY_PRIMITIVE,
    )
    if assembly is not None:
        planned["assembly_id"] = assembly.get("assembly_id")
        planned["union_fixture"] = fixture
    else:
        planned["union_fixture"] = fixture
    return planned


def plan_lane_at_parameters_endpoint_only(
    stock_positions: np.ndarray,
    stock_rotations: np.ndarray,
    *,
    fixture: dict[str, Any],
    panel_side: str,
    lane_y_m: float,
    padding_m: float,
    entry_x_m: float,
    exit_x_m: float,
    aperture_width_m: float = DEFAULT_APERTURE_WIDTH_M,
    freeze_start: bool = False,
    freeze_final: bool = True,
) -> dict[str, Any]:
    return _plan_lane_with_rewrite(
        stock_positions,
        stock_rotations,
        fixture=fixture,
        panel_side=panel_side,
        lane_y_m=lane_y_m,
        padding_m=padding_m,
        entry_x_m=float(entry_x_m),
        exit_x_m=float(exit_x_m),
        aperture_width_m=aperture_width_m,
        freeze_start=freeze_start,
        freeze_final=freeze_final,
        rewrite=apply_constant_lane_endpoint_only,
        primitive=ENDPOINT_ONLY_PRIMITIVE,
    )


def stage_counts(
    *,
    attempted: int = 0,
    passed: int = 0,
    failed: int = 0,
    not_evaluated: int = 0,
) -> dict[str, Any]:
    return {
        "attempted": int(attempted),
        "passed": int(passed),
        "failed": int(failed),
        "not_evaluated": int(not_evaluated),
        "evaluated_count": int(attempted),
    }


def assign_members_across_routes(
    member_indices: Sequence[int],
    route_identities: Sequence[dict[str, Any]],
    *,
    evaluate_route,
    score_members,
) -> dict[str, Any]:
    """Exhaust registered routes per member. IK/env success is not a pass.

    Different members may keep different route identities. A member is
    exhausted only after every identity has been tried or remaining members
    are empty. Identities after that are ``not_evaluated`` for IK.
    """
    remaining = {int(index) for index in member_indices}
    selected: dict[int, dict[str, Any]] = {}
    failed_clearance_attempts: dict[int, int] = {int(index): 0 for index in remaining}
    recoveries = 0
    ik = stage_counts(not_evaluated=0)
    env = stage_counts(not_evaluated=0)
    robust = stage_counts(not_evaluated=0)
    clearance = stage_counts(not_evaluated=0)
    corners_evaluated = 0
    identities_generated = len(list(route_identities))
    for identity in route_identities:
        if not remaining:
            ik["not_evaluated"] += 1
            env["not_evaluated"] += 1
            robust["not_evaluated"] += 1
            continue
        try:
            report = evaluate_route(identity, remaining=sorted(remaining))
        except TypeError:
            report = evaluate_route(identity)
        ik["attempted"] += 1
        if not report.get("ik_ok"):
            ik["failed"] += 1
            env["not_evaluated"] += 1
            robust["not_evaluated"] += 1
            continue
        ik["passed"] += 1
        env["attempted"] += 1
        if not report.get("environment_clear"):
            env["failed"] += 1
            robust["not_evaluated"] += 1
            continue
        env["passed"] += 1
        if report.get("nominal_clearance_empty"):
            robust["not_evaluated"] += 1
            for member in remaining:
                clearance["attempted"] += 1
                clearance["failed"] += 1
                failed_clearance_attempts[int(member)] += 1
            continue
        n_corners = int(report.get("n_corners_evaluated") or 0)
        robust["attempted"] += 1
        corners_evaluated += n_corners
        if n_corners != 8 or not report.get("robust_ok", True):
            robust["failed"] += 1
            continue
        robust["passed"] += 1
        scores = score_members(sorted(remaining), report)
        for member in list(remaining):
            clearance["attempted"] += 1
            if scores.get(int(member)):
                clearance["passed"] += 1
                if failed_clearance_attempts[int(member)] > 0:
                    recoveries += 1
                selected[int(member)] = dict(identity)
                remaining.discard(int(member))
            else:
                clearance["failed"] += 1
                failed_clearance_attempts[int(member)] += 1
    ik["evaluated_count"] = ik["attempted"]
    env["evaluated_count"] = env["attempted"]
    robust["evaluated_count"] = robust["attempted"]
    clearance["evaluated_count"] = clearance["attempted"]
    return {
        "selected": selected,
        "exhausted": sorted(remaining),
        "alternative_route_recoveries": int(recoveries),
        "identities_generated": int(identities_generated),
        "nominal_ik": ik,
        "strict_environment": env,
        "robust_routes": robust,
        "morphology_clearance": clearance,
        "corners_evaluated": int(corners_evaluated),
        "all_eight_corners_evaluated": (
            "not_applicable"
            if robust["attempted"] == 0
            else bool(corners_evaluated == 8 * robust["attempted"])
        ),
    }


def select_at_most_two_candidates(
    scored: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Signal then clearance ranking. At most two distinct assemblies."""
    if not scored:
        return []

    def _volume(item: dict[str, Any]) -> float:
        return float(
            item.get("volume_m3")
            if item.get("volume_m3") is not None
            else item.get("assembly", {}).get("volume_m3", 0.0)
        )

    def _signal_key(item: dict[str, Any]) -> tuple[float, int, float, float]:
        return (
            float(item["worst_cell_changed_value_fraction"]),
            int(item["worst_cell_changed_sensors"]),
            float(item["min_robust_clearance_m"]),
            -_volume(item),
        )

    def _clearance_key(item: dict[str, Any]) -> tuple[float, float, int, float]:
        return (
            float(item["min_robust_clearance_m"]),
            float(item["worst_cell_changed_value_fraction"]),
            int(item["worst_cell_changed_sensors"]),
            -_volume(item),
        )

    def _key_of(item: dict[str, Any]) -> Any:
        return item.get("assembly_id") or item.get("key")

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


ROUTE_CACHE_FIELDS = (
    "cell_role_index",
    "direction",
    "union_key",
    "padding_m",
    "lane_y_m",
    "perturbation_index",
)


def route_ik_cache_key(
    *,
    cell_role_index: int,
    direction: str,
    union_key: Sequence[float],
    padding_m: float,
    lane_y_m: float,
    perturbation_index: int,
) -> tuple[Any, ...]:
    """IK qpos may be reused across morphologies that share a union path.

    Actual component clearance, sensing, volume, and ranking stay per-assembly.
    """
    return (
        int(cell_role_index),
        str(direction),
        tuple(round(float(value), 9) for value in union_key),
        round(float(padding_m), 9),
        round(float(lane_y_m), 9),
        int(perturbation_index),
    )


def evaluate_pendant_nominal_and_robust(
    *,
    min_nominal_m: float,
    min_robust_m: float,
    nominal_clearance_m: float | None,
    corner_clearances_m: Sequence[float | None],
) -> dict[str, Any]:
    """Pendant-only thresholds. Environment geoms do not use the 25 mm floor."""
    if len(list(corner_clearances_m)) != 8:
        raise ValueError("robust clearance requires all eight perturbation corners")
    nominal_ok = bool(
        nominal_clearance_m is not None
        and float(nominal_clearance_m) + 1e-12 >= float(min_nominal_m)
    )
    robust_ok = True
    scored = []
    for index, value in enumerate(corner_clearances_m):
        ok = bool(value is not None and float(value) + 1e-12 >= float(min_robust_m))
        robust_ok = robust_ok and ok
        scored.append(
            {
                "perturbation_index": int(index),
                "clearance_m": None if value is None else float(value),
                "meets_robust": ok,
            }
        )
    return {
        "nominal_clearance_m": None if nominal_clearance_m is None else float(nominal_clearance_m),
        "meets_nominal": nominal_ok,
        "robust_corners": scored,
        "meets_robust": bool(robust_ok),
        "n_corners_evaluated": 8,
    }


def evaluate_environment_no_intersection(
    distances_m: Sequence[float | None],
) -> dict[str, Any]:
    """Strict environment: contact or a missing distance is a failure.

    Do not impose the 25 mm pendant floor.
    """
    from pact_geom_distance import CONTACT_DISTANCE_M

    values = list(distances_m)
    contacts = []
    clear = bool(values)
    for index, value in enumerate(values):
        missing = value is None
        hit = bool(missing or float(value) <= CONTACT_DISTANCE_M)
        clear = clear and not hit
        contacts.append(
            {
                "index": int(index),
                "distance_m": None if missing else float(value),
                "missing": missing,
                "contact": hit,
            }
        )
    return {"environment_clear": bool(clear), "geoms": contacts}


def evaluate_all_perturbation_corners(
    plan: dict[str, Any],
    evaluator,
) -> list[dict[str, Any]]:
    """Call evaluator on every corner. Metadata-only lists are not a predicate."""
    corners = list(plan.get("perturbation_corners") or perturbation_corners(
        float(plan["lane_y_m"]),
        float(plan["entry_x_m"]),
        float(plan["exit_x_m"]),
    ))
    if len(corners) != 8:
        raise ValueError(f"expected 8 perturbation corners, got {len(corners)}")
    reports = []
    for index, corner in enumerate(corners):
        payload = dict(corner)
        payload["perturbation_index"] = int(index)
        result = evaluator(payload)
        if not isinstance(result, dict):
            raise TypeError("perturbation evaluator must return a dict report")
        reports.append({**payload, **result, "evaluated": True})
    if len(reports) != 8 or not all(item.get("evaluated") for item in reports):
        raise RuntimeError("perturbation evaluation did not score all eight corners")
    return reports


def union_cluster_row_indices(
    union_keys: Sequence[tuple[float, ...]],
) -> dict[tuple[float, ...], list[int]]:
    """Map union AABB keys to morphology row indices. Do not collapse the lists."""
    clusters: dict[tuple[float, ...], list[int]] = {}
    for index, key in enumerate(union_keys):
        clusters.setdefault(tuple(key), []).append(int(index))
    return clusters


def signal_screen_requires_shortlist(
    n_morphologies: int,
    *,
    max_complete_screens: int,
) -> dict[str, Any]:
    from pact_place_v10_compound_pendant_contract import ROUTE_SHORTLIST_AMENDMENT

    too_many = int(n_morphologies) > int(max_complete_screens)
    return {
        "n_morphologies": int(n_morphologies),
        "max_complete_screens": int(max_complete_screens),
        "requires_shortlist_amendment": too_many,
        "stop_reason": ROUTE_SHORTLIST_AMENDMENT if too_many else None,
        "collapsed_by_union_aabb": False,
    }


def signal_screen_admission(n_morphologies: int) -> dict[str, Any]:
    """Admit a complete signal screen only against a preregistered limit."""
    from pact_place_v10_compound_pendant_contract import (
        REGISTERED_COMPLETE_SIGNAL_SCREEN_LIMIT,
        ROUTE_SHORTLIST_AMENDMENT,
        SIGNAL_SCREEN_LIMIT_UNREGISTERED,
    )

    limit = REGISTERED_COMPLETE_SIGNAL_SCREEN_LIMIT
    n_rows = int(n_morphologies)
    if limit is None:
        return {
            "n_morphologies": n_rows,
            "registered_limit": None,
            "signal_screen_run": False,
            "stop_reason": SIGNAL_SCREEN_LIMIT_UNREGISTERED,
            "collapsed_by_union_aabb": False,
            "post_hoc_shortlist": False,
        }
    limit_i = int(limit)
    too_many = n_rows > limit_i
    return {
        "n_morphologies": n_rows,
        "registered_limit": limit_i,
        "signal_screen_run": (not too_many) and n_rows > 0,
        "stop_reason": ROUTE_SHORTLIST_AMENDMENT if too_many else None,
        "collapsed_by_union_aabb": False,
        "post_hoc_shortlist": False,
    }


def copy_qpos_dict(qpos: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        str(key): np.asarray(value, dtype=float).copy() for key, value in qpos.items()
    }


def qpos_dicts_equal(left: dict[str, Any], right: dict[str, Any], *, atol: float = 1e-12) -> bool:
    if set(left) != set(right):
        return False
    return all(
        np.allclose(
            np.asarray(left[key], dtype=float),
            np.asarray(right[key], dtype=float),
            atol=atol,
        )
        for key in left
    )


def finite_distance_or_none(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not np.isfinite(number):
        return None
    return number


def score_split_clearance(
    *,
    pendant_distances_m: Sequence[float | None],
    environment_distances_m: Sequence[float | None],
    ik_failures: int,
    n_waypoints: int,
    min_pendant_m: float,
) -> dict[str, Any]:
    """Split pendant vs strict-environment decisions. None fails closed."""
    from pact_place_v10_compound_pendant_contract import MIN_NOMINAL_CLEARANCE_M

    pendant_values = list(pendant_distances_m)
    env_values = list(environment_distances_m)
    if len(pendant_values) != int(n_waypoints) or len(env_values) != int(n_waypoints):
        raise ValueError("split clearance requires one distance per waypoint")
    pendant_ok = bool(
        pendant_values
        and all(value is not None for value in pendant_values)
        and min(float(value) for value in pendant_values) + 1e-12 >= float(min_pendant_m)
    )
    env_report = evaluate_environment_no_intersection(env_values)
    pendant_min = (
        None
        if (not pendant_values or any(value is None for value in pendant_values))
        else float(min(float(value) for value in pendant_values))
    )
    env_min = (
        None
        if (not env_values or any(value is None for value in env_values))
        else float(min(float(value) for value in env_values))
    )
    ik_ok = bool(int(ik_failures) == 0 and n_waypoints > 0)
    imposed_pendant_floor_on_env = bool(
        env_min is not None
        and float(env_min) + 1e-12 < float(MIN_NOMINAL_CLEARANCE_M)
        and env_report["environment_clear"]
        and pendant_ok
    )
    return {
        "ik_ok": ik_ok,
        "ik_failures": int(ik_failures),
        "n_waypoints": int(n_waypoints),
        "pendant_clearance_m": pendant_min,
        "environment_clearance_m": env_min,
        "meets_pendant": pendant_ok,
        "environment_clear": bool(env_report["environment_clear"]),
        "missing_pendant_distance": any(value is None for value in pendant_values) or not pendant_values,
        "missing_environment_distance": any(value is None for value in env_values) or not env_values,
        "imposed_pendant_floor_on_environment": imposed_pendant_floor_on_env,
        "accepted": bool(ik_ok and pendant_ok and env_report["environment_clear"]),
    }


def sequential_ik_split_clearance(
    positions: np.ndarray,
    rotations: np.ndarray,
    *,
    saved_qpos: dict[str, Any],
    set_qpos,
    get_qpos,
    solve_ik,
    forward,
    place_pose,
    measure_pendant,
    measure_environment,
    min_pendant_m: float,
    abort_on_ik_failure: bool = False,
    abort_on_environment_failure: bool = False,
) -> dict[str, Any]:
    """Sequential IK with split pendant / environment measurements.

    Restores ``saved_qpos`` on success, IK failure, and exceptions. Seeds each
    waypoint from the preceding successful solution.
    """
    from pact_geom_distance import CONTACT_DISTANCE_M

    positions = np.asarray(positions, dtype=np.float64)
    rotations = np.asarray(rotations, dtype=np.float64)
    n_waypoints = int(len(positions))
    seed = copy_qpos_dict(saved_qpos)
    failures = 0
    pendant_distances: list[float | None] = []
    environment_distances: list[float | None] = []
    qpos_sequence: list[dict[str, np.ndarray]] = []
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
                failures += 1
                qpos_sequence.append(copy_qpos_dict(seed))
                if abort_on_ik_failure:
                    remaining = n_waypoints - index
                    pendant_distances.extend([None] * remaining)
                    environment_distances.extend([None] * remaining)
                    break
                pendant_distances.append(None)
                environment_distances.append(None)
                continue
            set_qpos(solution)
            forward()
            seed = copy_qpos_dict(get_qpos())
            qpos_sequence.append(copy_qpos_dict(seed))
            pendant_distances.append(finite_distance_or_none(measure_pendant()))
            env_distance = finite_distance_or_none(measure_environment())
            environment_distances.append(env_distance)
            if abort_on_environment_failure:
                hit = env_distance is None or float(env_distance) <= CONTACT_DISTANCE_M
                if hit:
                    remaining = n_waypoints - index - 1
                    if remaining:
                        pendant_distances.extend([None] * remaining)
                        environment_distances.extend([None] * remaining)
                    break
    except Exception:
        restore()
        raise
    finally:
        restore()

    scored = score_split_clearance(
        pendant_distances_m=pendant_distances,
        environment_distances_m=environment_distances,
        ik_failures=failures,
        n_waypoints=n_waypoints,
        min_pendant_m=min_pendant_m,
    )
    scored["qpos_sequence"] = qpos_sequence
    scored["pendant_distances_m"] = pendant_distances
    scored["environment_distances_m"] = environment_distances
    scored["restored_qpos"] = copy_qpos_dict(saved_qpos)
    return scored


class RouteIkCache:
    """Cache lane / IK qpos / strict-environment results by union path identity."""

    def __init__(self) -> None:
        self.store: dict[tuple[Any, ...], dict[str, Any]] = {}
        self.hits = 0
        self.misses = 0
        self.qpos_reuses = 0

    def key(self, **fields: Any) -> tuple[Any, ...]:
        return route_ik_cache_key(**fields)

    def get(self, key: tuple[Any, ...]) -> dict[str, Any] | None:
        item = self.store.get(key)
        if item is None:
            self.misses += 1
            return None
        self.hits += 1
        if item.get("qpos_sequence") is not None or item.get("qpos_paths") is not None:
            self.qpos_reuses += 1
        return item

    def put(self, key: tuple[Any, ...], value: dict[str, Any]) -> None:
        self.store[key] = value

    def provenance(self) -> dict[str, int]:
        return {
            "hits": int(self.hits),
            "misses": int(self.misses),
            "entries": int(len(self.store)),
            "qpos_reuses": int(self.qpos_reuses),
        }


def aabb_gap_m(
    lo_a: Sequence[float],
    hi_a: Sequence[float],
    lo_b: Sequence[float],
    hi_b: Sequence[float],
) -> float:
    lo_a_v = np.asarray(lo_a, dtype=np.float64)
    hi_a_v = np.asarray(hi_a, dtype=np.float64)
    lo_b_v = np.asarray(lo_b, dtype=np.float64)
    hi_b_v = np.asarray(hi_b, dtype=np.float64)
    delta = np.maximum(lo_a_v - hi_b_v, lo_b_v - hi_a_v)
    return float(np.max(delta))


def min_component_distance_to_probes(
    center_m: Sequence[float],
    half_m: Sequence[float],
    probes: Sequence[dict[str, Any]],
    *,
    certify_gap_m: float = 0.050,
) -> float | None:
    """Exact GJK unless AABBs certify a gap at least ``certify_gap_m``."""
    from pact_geom_distance import GeomShape, gjk_distance
    from pact_place_v10_compound_pendant_contract import component_aabb

    if not probes:
        return None
    box_lo, box_hi = component_aabb(center_m, half_m)
    box = GeomShape.posed_axis_aligned_box(
        np.asarray(center_m, dtype=np.float64),
        np.asarray(half_m, dtype=np.float64),
    )
    best = np.inf
    measured = False
    for probe in probes:
        gap = aabb_gap_m(box_lo, box_hi, probe["lo"], probe["hi"])
        if gap >= float(certify_gap_m):
            best = min(best, gap)
            measured = True
            continue
        shape = probe.get("shape")
        if shape is None:
            shape = GeomShape(
                int(probe["gtype"]),
                np.asarray(probe["pos"], dtype=np.float64),
                np.asarray(probe["mat"], dtype=np.float64).reshape(3, 3),
                np.asarray(probe["size"], dtype=np.float64),
                None if probe.get("verts") is None else np.asarray(probe["verts"]),
            )
        distance = gjk_distance(box, shape)
        if distance is None or not np.isfinite(distance):
            return None
        best = min(best, float(distance))
        measured = True
    if not measured:
        return None
    return float(best)


def min_assembly_pendant_clearance(
    components: Sequence[dict[str, Any]],
    probes: Sequence[dict[str, Any]],
) -> float | None:
    distances: list[float] = []
    for item in components:
        if not item.get("active", True):
            continue
        if item.get("role") not in {"lobe", "stem", "crossbar"}:
            continue
        distance = min_component_distance_to_probes(
            item["center_m"], item["half_m"], probes
        )
        if distance is None:
            return None
        distances.append(float(distance))
    if not distances:
        return None
    return float(min(distances))


def probes_min_environment_distance(
    probes: Sequence[dict[str, Any]],
    env_geoms: Sequence[dict[str, Any]],
) -> float | None:
    """Strict environment: AABB certifies separation; GJK otherwise. None fails closed."""
    from pact_geom_distance import CONTACT_DISTANCE_M, GeomShape, gjk_distance

    if not probes:
        return None
    best = np.inf
    measured = False
    for probe in probes:
        probe_shape = probe.get("shape")
        for env in env_geoms:
            gap = aabb_gap_m(probe["lo"], probe["hi"], env["lo"], env["hi"])
            if gap > CONTACT_DISTANCE_M:
                best = min(best, gap)
                measured = True
                continue
            env_shape = env.get("shape")
            if env_shape is None:
                verts = env.get("verts")
                env_shape = GeomShape(
                    int(env["gtype"]),
                    np.asarray(env["pos"], dtype=np.float64),
                    np.asarray(env["mat"], dtype=np.float64).reshape(3, 3),
                    np.asarray(env["size"], dtype=np.float64),
                    None if verts is None else np.asarray(verts),
                )
            if probe_shape is None:
                probe_shape = GeomShape(
                    int(probe["gtype"]),
                    np.asarray(probe["pos"], dtype=np.float64),
                    np.asarray(probe["mat"], dtype=np.float64).reshape(3, 3),
                    np.asarray(probe["size"], dtype=np.float64),
                    None if probe.get("verts") is None else np.asarray(probe["verts"]),
                )
            if not getattr(probe_shape, "supported", True) or not getattr(env_shape, "supported", True):
                return 0.0
            distance = gjk_distance(probe_shape, env_shape)
            if distance is None or not np.isfinite(distance):
                return None
            best = min(best, float(distance))
            measured = True
            if best <= CONTACT_DISTANCE_M:
                return float(best)
    if not measured:
        return None
    return float(best)


def dump_probe_geoms(model, data, geom_ids: Sequence[int]) -> list[dict[str, Any]]:
    from pact_geom_distance import GeomShape, geom_world_aabb

    records: list[dict[str, Any]] = []
    for geom_id in geom_ids:
        gid = int(geom_id)
        shape = None
        try:
            shape = GeomShape.from_mujoco(model, data, gid)
            lo, hi = shape.world_aabb() if shape.supported else geom_world_aabb(model, data, gid)
        except Exception:
            try:
                lo, hi = geom_world_aabb(model, data, gid)
            except ValueError:
                continue
        records.append(
            {
                "geom_id": gid,
                "lo": np.asarray(lo, dtype=np.float64),
                "hi": np.asarray(hi, dtype=np.float64),
                "gtype": int(model.geom_type[gid]),
                "pos": np.asarray(data.geom_xpos[gid], dtype=np.float64),
                "mat": np.asarray(data.geom_xmat[gid], dtype=np.float64).reshape(3, 3),
                "size": np.asarray(model.geom_size[gid], dtype=np.float64),
                "verts": None if shape is None else shape.verts,
                "shape": shape,
            }
        )
    return records


def stock_tcp_from_cell(cell: dict[str, Any], direction: str) -> tuple[np.ndarray, np.ndarray]:
    mask_key = "inbound_mask" if direction == "inbound" else "outbound_mask"
    mask = np.asarray(cell[mask_key], dtype=bool)
    tcp = np.asarray(cell["tcp_m"], dtype=np.float64)
    mat = np.asarray(cell["tcp_mat"], dtype=np.float64).reshape(len(tcp), 3, 3)
    if not np.any(mask):
        raise ValueError(f"{direction} mask is empty for role {cell.get('role_index')}")
    return tcp[mask], mat[mask]


def cluster_two_lobe_unions(lobe_keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return unique union keys (N_u, 6) and inverse row map (N,)."""
    from pact_place_v10_catalog import union_aabb_from_two_lobe_keys

    lo, hi = union_aabb_from_two_lobe_keys(np.asarray(lobe_keys, dtype=np.float64))
    packed = np.round(np.concatenate([lo, hi], axis=1), 9)
    unique, inverse = np.unique(packed, axis=0, return_inverse=True)
    return unique.astype(np.float64), inverse.astype(np.int32)

