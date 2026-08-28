#!/usr/bin/env python3
"""Deterministic V10 lobe lattice, stems, crossbar, and set-cover helpers.

Pure numpy. No MuJoCo, no V9.8 lag/window imports.
"""

from __future__ import annotations

from typing import Any, Iterable, Iterator, Sequence

import numpy as np

from pact_place_v10_compound_pendant_contract import (
    CENTER_X_BOUNDS_M,
    CENTER_X_STEP_M,
    CENTER_Y_ABS_BOUNDS_M,
    CENTER_Y_STEP_M,
    CENTER_Z_BOUNDS_M,
    CENTER_Z_STEP_M,
    CEILING_TOP_Z_M,
    CROSSBAR_GEOM,
    CROSSBAR_HEIGHT_M,
    DEFAULT_APERTURE_WIDTH_M,
    HALF_X_CHOICES_M,
    HALF_Y_CHOICES_M,
    HALF_Z_CHOICES_M,
    HOOD_TOP_BOTTOM_Z_M,
    HOOD_TOP_CENTER_M,
    HOOD_TOP_HALF_M,
    LOBE_BOTTOM_MIN_M,
    LOBE_GEOMS,
    LOBE_TOP_MAX_M,
    NEGATIVE_LOBE_MAX_Y_M,
    N_NECESSITY_BITS,
    PENDANT_BODY,
    PENDANT_DEPTH_BOUNDS_M,
    PENDANT_SUPPORT,
    POSITIVE_LOBE_MIN_Y_M,
    PROBE_NEGATIVE_LOBE,
    PROBE_POSITIVE_LOBE,
    PROBE_STEM_Y_M,
    PROBE_V1_NEGATIVE_LOBE,
    PROBE_V1_POSITIVE_LOBE,
    PROBE_V1_STEM_Y_M,
    STEM_GEOMS,
    STEM_HALF_M,
    STEM_TOP_Z_M,
    component_aabb,
    component_volume_m3,
    reject_v98_kwargs,
    round_m,
    round_vec,
)

NECESSITY_ALL_BITS = (1 << N_NECESSITY_BITS) - 1


def arange_inclusive(start: float, stop: float, step: float) -> np.ndarray:
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


def signed_center_y_values() -> np.ndarray:
    magnitudes = arange_inclusive(*CENTER_Y_ABS_BOUNDS_M, CENTER_Y_STEP_M)
    return np.round(
        np.concatenate([-magnitudes[::-1], magnitudes]),
        9,
    )


def _inside_aperture_enclosure(
    center: Sequence[float],
    half: Sequence[float],
    *,
    aperture_width_m: float,
) -> None:
    lo, hi = component_aabb(center, half)
    if lo[0] < PENDANT_DEPTH_BOUNDS_M[0] - 1e-9 or hi[0] > PENDANT_DEPTH_BOUNDS_M[1] + 1e-9:
        raise ValueError(
            f"component x [{lo[0]:.9f}, {hi[0]:.9f}] is outside {PENDANT_DEPTH_BOUNDS_M}"
        )
    y_limit = float(aperture_width_m) / 2.0
    if lo[1] < -y_limit - 1e-9 or hi[1] > y_limit + 1e-9:
        raise ValueError(
            f"component y [{lo[1]:.9f}, {hi[1]:.9f}] exceeds aperture ±{y_limit:.9f}"
        )


def validate_lobe_geometry(
    center_m: Sequence[float],
    half_m: Sequence[float],
    *,
    aperture_width_m: float = DEFAULT_APERTURE_WIDTH_M,
    **kwargs: Any,
) -> dict[str, Any]:
    reject_v98_kwargs(kwargs)
    center = tuple(float(item) for item in center_m)
    half = tuple(float(item) for item in half_m)
    if any(value <= 0.0 for value in half):
        raise ValueError(f"lobe half extents must be positive: {half}")
    if not (
        CENTER_X_BOUNDS_M[0] - 1e-9 <= center[0] <= CENTER_X_BOUNDS_M[1] + 1e-9
    ):
        raise ValueError(f"lobe center x {center[0]:.9f} is outside {CENTER_X_BOUNDS_M}")
    abs_y = abs(center[1])
    if not (
        CENTER_Y_ABS_BOUNDS_M[0] - 1e-9 <= abs_y <= CENTER_Y_ABS_BOUNDS_M[1] + 1e-9
    ):
        raise ValueError(
            f"lobe |center y| {abs_y:.9f} is outside {CENTER_Y_ABS_BOUNDS_M}"
        )
    if not any(abs(half[0] - choice) <= 1e-9 for choice in HALF_X_CHOICES_M):
        raise ValueError(f"lobe half x {half[0]:.9f} is not in {HALF_X_CHOICES_M}")
    if not any(abs(half[1] - choice) <= 1e-9 for choice in HALF_Y_CHOICES_M):
        raise ValueError(f"lobe half y {half[1]:.9f} is not in {HALF_Y_CHOICES_M}")
    if not (
        CENTER_Z_BOUNDS_M[0] - 1e-9 <= center[2] <= CENTER_Z_BOUNDS_M[1] + 1e-9
    ):
        raise ValueError(f"lobe center z {center[2]:.9f} is outside {CENTER_Z_BOUNDS_M}")
    if not any(abs(half[2] - choice) <= 1e-9 for choice in HALF_Z_CHOICES_M):
        raise ValueError(f"lobe half z {half[2]:.9f} is not in {HALF_Z_CHOICES_M}")
    bottom = center[2] - half[2]
    top = center[2] + half[2]
    if bottom < LOBE_BOTTOM_MIN_M - 1e-9:
        raise ValueError(f"lobe bottom {bottom:.9f} is below {LOBE_BOTTOM_MIN_M}")
    if top > LOBE_TOP_MAX_M + 1e-9:
        raise ValueError(f"lobe top {top:.9f} is above {LOBE_TOP_MAX_M}")
    if center[1] < 0.0:
        if center[1] + half[1] > NEGATIVE_LOBE_MAX_Y_M + 1e-9:
            raise ValueError("negative lobe must lie wholly below y=-0.08")
        side = "negative"
    else:
        if center[1] - half[1] < POSITIVE_LOBE_MIN_Y_M - 1e-9:
            raise ValueError("positive lobe must lie wholly above y=+0.08")
        side = "positive"
    _inside_aperture_enclosure(center, half, aperture_width_m=aperture_width_m)
    return {
        "center_m": round_vec(center),
        "half_m": round_vec(half),
        "side": side,
        "bottom_z_m": round_m(bottom),
        "top_z_m": round_m(top),
    }


def lobe_key(center_m: Sequence[float], half_m: Sequence[float]) -> tuple[float, ...]:
    center = round_vec(center_m)
    half = round_vec(half_m)
    return tuple(center + half)


def lobe_from_key(
    key: Sequence[float],
    *,
    aperture_width_m: float = DEFAULT_APERTURE_WIDTH_M,
) -> dict[str, Any]:
    values = [float(item) for item in key]
    if len(values) != 6:
        raise ValueError(f"lobe key must have six values, got {key!r}")
    return build_lobe(
        center_x_m=values[0],
        center_y_m=values[1],
        center_z_m=values[2],
        half_x_m=values[3],
        half_y_m=values[4],
        half_z_m=values[5],
        aperture_width_m=aperture_width_m,
    )


def build_lobe(
    *,
    center_x_m: float,
    center_y_m: float,
    center_z_m: float,
    half_x_m: float,
    half_y_m: float,
    half_z_m: float,
    aperture_width_m: float = DEFAULT_APERTURE_WIDTH_M,
) -> dict[str, Any]:
    validated = validate_lobe_geometry(
        (center_x_m, center_y_m, center_z_m),
        (half_x_m, half_y_m, half_z_m),
        aperture_width_m=aperture_width_m,
    )
    return {
        "role": "lobe",
        "side": validated["side"],
        "center_m": validated["center_m"],
        "half_m": validated["half_m"],
        "bottom_z_m": validated["bottom_z_m"],
        "top_z_m": validated["top_z_m"],
        "key": lobe_key(validated["center_m"], validated["half_m"]),
        "volume_m3": component_volume_m3(validated["half_m"]),
    }


def stem_for_lobe(
    lobe: dict[str, Any],
    *,
    aperture_width_m: float = DEFAULT_APERTURE_WIDTH_M,
) -> dict[str, Any]:
    center = lobe["center_m"]
    half = lobe["half_m"]
    outward = -1.0 if lobe["side"] == "negative" else 1.0
    stem_y = float(center[1]) + outward * float(half[1])
    lobe_top = float(lobe["top_z_m"])
    if STEM_TOP_Z_M <= lobe_top + 1e-12:
        raise ValueError("stem has non-positive height")
    half_z = (STEM_TOP_Z_M - lobe_top) / 2.0
    stem_center = (float(center[0]), stem_y, lobe_top + half_z)
    stem_half = (STEM_HALF_M, STEM_HALF_M, half_z)
    _inside_aperture_enclosure(stem_center, stem_half, aperture_width_m=aperture_width_m)
    return {
        "role": "stem",
        "side": lobe["side"],
        "center_m": round_vec(stem_center),
        "half_m": round_vec(stem_half),
        "volume_m3": component_volume_m3(stem_half),
    }


def crossbar_for_stems(
    stems: Sequence[dict[str, Any]],
    *,
    aperture_width_m: float = DEFAULT_APERTURE_WIDTH_M,
) -> dict[str, Any]:
    if len(stems) < 2:
        raise ValueError("crossbar requires at least two stems")
    xs = [float(item["center_m"][0]) for item in stems]
    if max(xs) - min(xs) > 1e-9:
        raise ValueError("stems must share center x")
    ys = [float(item["center_m"][1]) for item in stems]
    y_lo = min(ys) - STEM_HALF_M
    y_hi = max(ys) + STEM_HALF_M
    half_z = CROSSBAR_HEIGHT_M / 2.0
    center = (xs[0], 0.5 * (y_lo + y_hi), CEILING_TOP_Z_M - half_z)
    half = (STEM_HALF_M, 0.5 * (y_hi - y_lo), half_z)
    _inside_aperture_enclosure(center, half, aperture_width_m=aperture_width_m)
    top = center[2] + half[2]
    if abs(top - HOOD_TOP_BOTTOM_Z_M) > 1e-9:
        raise ValueError(f"crossbar top {top:.9f} is not flush to hood_top at 1.515")
    return {
        "role": "crossbar",
        "center_m": round_vec(center),
        "half_m": round_vec(half),
        "volume_m3": component_volume_m3(half),
    }


def assembly_aabb(assembly: dict[str, Any]) -> tuple[list[float], list[float]]:
    lows = []
    highs = []
    for item in active_components(assembly):
        lo, hi = component_aabb(item["center_m"], item["half_m"])
        lows.append(lo)
        highs.append(hi)
    return (
        [min(values) for values in zip(*lows)],
        [max(values) for values in zip(*highs)],
    )


def assembly_volume_m3(assembly: dict[str, Any]) -> float:
    return float(sum(float(item["volume_m3"]) for item in active_components(assembly)))


def active_components(assembly: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in assembly["components"] if item.get("active", True)]


def union_fixture(assembly: dict[str, Any]) -> dict[str, Any]:
    """Axis-aligned union box used by the lane slab primitive."""
    lo, hi = assembly_aabb(assembly)
    center = [(lo[i] + hi[i]) / 2.0 for i in range(3)]
    half = [(hi[i] - lo[i]) / 2.0 for i in range(3)]
    return {
        "center_m": round_vec(center),
        "half_m": round_vec(half),
        "support": PENDANT_SUPPORT,
        "body": PENDANT_BODY,
    }


def union_aabb_key(
    lo: Sequence[float], hi: Sequence[float], *, ndigits: int = 9
) -> tuple[float, ...]:
    return tuple(
        round(float(value), ndigits) for value in list(lo) + list(hi)
    )


def connected_stems_and_crossbar(stems: Sequence[dict[str, Any]], crossbar: dict[str, Any]) -> bool:
    """Stems meet the crossbar bottom and share the crossbar x slab."""
    bar_lo, bar_hi = component_aabb(crossbar["center_m"], crossbar["half_m"])
    for stem in stems:
        stem_lo, stem_hi = component_aabb(stem["center_m"], stem["half_m"])
        if abs(stem_hi[2] - bar_lo[2]) > 1e-8:
            return False
        if not aabb_overlap(stem_lo, stem_hi, bar_lo, bar_hi):
            return False
    return True


def hood_top_attachment_ok(crossbar: dict[str, Any]) -> bool:
    bar_lo, bar_hi = component_aabb(crossbar["center_m"], crossbar["half_m"])
    hood_lo, hood_hi = component_aabb(HOOD_TOP_CENTER_M, HOOD_TOP_HALF_M)
    if abs(bar_hi[2] - HOOD_TOP_BOTTOM_Z_M) > 1e-9:
        return False
    return aabb_overlap(bar_lo, bar_hi, hood_lo, hood_hi)


def forbidden_static_overlap(
    component: dict[str, Any],
    *,
    allow_hood_top: bool,
) -> bool:
    """True if this component overlaps a static enclosure box it must not."""
    lo, hi = component_aabb(component["center_m"], component["half_m"])
    hood_lo, hood_hi = component_aabb(HOOD_TOP_CENTER_M, HOOD_TOP_HALF_M)
    overlaps_hood = aabb_overlap(lo, hi, hood_lo, hood_hi)
    if overlaps_hood and not allow_hood_top:
        return True
    if overlaps_hood and allow_hood_top:
        # Only the designed flush face is allowed; reject volume intrusion.
        if hi[2] > HOOD_TOP_BOTTOM_Z_M + 1e-8:
            return True
    static = (
        ((0.95, 0.45, 1.12), (0.40, 0.012, 0.40)),  # hood_side_l
        ((0.95, -0.45, 1.12), (0.40, 0.012, 0.40)),  # hood_side_r
        ((1.36, 0.0, 1.12), (0.012, 0.46, 0.40)),  # hood_back
        ((0.58, 0.45, 1.12), (0.02, 0.02, 0.40)),  # hood_frame_l
        ((0.58, -0.45, 1.12), (0.02, 0.02, 0.40)),  # hood_frame_r
        ((0.95, 0.0, 0.70), (0.45, 0.60, 0.02)),  # bench_top
    )
    for center, half in static:
        env_lo, env_hi = component_aabb(center, half)
        if aabb_overlap(lo, hi, env_lo, env_hi):
            return True
    return False


def _component_record(
    *,
    name: str,
    role: str,
    geom: str,
    slot: int,
    payload: dict[str, Any],
    active: bool,
) -> dict[str, Any]:
    record = {
        "name": name,
        "role": role,
        "geom": geom,
        "slot": int(slot),
        "active": bool(active),
        "center_m": list(payload["center_m"]) if active else [0.0, 0.0, 0.0],
        "half_m": list(payload["half_m"]) if active else [0.001, 0.001, 0.001],
        "volume_m3": float(payload["volume_m3"]) if active else 0.0,
    }
    if "side" in payload:
        record["side"] = payload["side"]
    if "key" in payload:
        record["key"] = list(payload["key"])
    return record


def assembly_id_for(topology: str, lobes: Sequence[dict[str, Any]]) -> str:
    keys = ["-".join(f"{value:.9f}" for value in lobe["key"]) for lobe in lobes]
    return "v10_" + topology + "_" + "_".join(keys)


def build_assembly(
    lobes: Sequence[dict[str, Any]],
    *,
    aperture_width_m: float = DEFAULT_APERTURE_WIDTH_M,
) -> dict[str, Any]:
    if len(lobes) not in {2, 3}:
        raise ValueError("V10 assemblies must have two or three lobes")
    sides = {item["side"] for item in lobes}
    if "negative" not in sides or "positive" not in sides:
        raise ValueError("assembly must contain at least one lobe on each side")
    xs = [float(item["center_m"][0]) for item in lobes]
    if max(xs) - min(xs) > 1e-9:
        raise ValueError("assembly lobes must share center x")
    ordered = sorted(
        lobes,
        key=lambda item: (item["side"], item["center_m"][1], item["key"]),
    )
    topology = "two_lobe" if len(ordered) == 2 else "three_lobe"
    stems = [
        stem_for_lobe(lobe, aperture_width_m=aperture_width_m) for lobe in ordered
    ]
    bar = crossbar_for_stems(stems, aperture_width_m=aperture_width_m)
    if not connected_stems_and_crossbar(stems, bar):
        raise ValueError("stems and crossbar are not a connected assembly")
    if not hood_top_attachment_ok(bar):
        raise ValueError("crossbar is not attached to hood_top at z=1.515")
    components: list[dict[str, Any]] = []
    for slot in range(3):
        active = slot < len(ordered)
        lobe_payload = ordered[slot] if active else {
            "center_m": [0.0, 0.0, 0.0],
            "half_m": [0.001, 0.001, 0.001],
            "volume_m3": 0.0,
        }
        stem_payload = stems[slot] if active else {
            "center_m": [0.0, 0.0, 0.0],
            "half_m": [0.001, 0.001, 0.001],
            "volume_m3": 0.0,
        }
        components.append(
            _component_record(
                name=f"lobe_{slot}",
                role="lobe",
                geom=LOBE_GEOMS[slot],
                slot=slot,
                payload=lobe_payload,
                active=active,
            )
        )
        components.append(
            _component_record(
                name=f"stem_{slot}",
                role="stem",
                geom=STEM_GEOMS[slot],
                slot=slot,
                payload=stem_payload,
                active=active,
            )
        )
    components.append(
        _component_record(
            name="crossbar",
            role="crossbar",
            geom=CROSSBAR_GEOM,
            slot=0,
            payload=bar,
            active=True,
        )
    )
    for item in components:
        if not item["active"]:
            continue
        if forbidden_static_overlap(item, allow_hood_top=item["role"] == "crossbar"):
            raise ValueError(f"{item['name']} overlaps forbidden static geometry")
    assembly = {
        "topology": topology,
        "assembly_id": assembly_id_for(topology, ordered),
        "support": PENDANT_SUPPORT,
        "body": PENDANT_BODY,
        "identical_on_both_panel_sides": True,
        "active_on": ["inbound_empty", "outbound_loaded"],
        "center_x_m": round_m(xs[0]),
        "components": components,
        "volume_m3": 0.0,
        "union_fixture": {},
    }
    assembly["volume_m3"] = assembly_volume_m3(assembly)
    assembly["union_fixture"] = union_fixture(assembly)
    return assembly


def planning_probe_v1_invalid_assembly(
    *,
    aperture_width_m: float = DEFAULT_APERTURE_WIDTH_M,
) -> dict[str, Any]:
    """Old probe fixture. Intersects posed intrusion panels. Not a success fixture."""
    negative = build_lobe(
        center_x_m=PROBE_V1_NEGATIVE_LOBE["center_m"][0],
        center_y_m=PROBE_V1_NEGATIVE_LOBE["center_m"][1],
        center_z_m=PROBE_V1_NEGATIVE_LOBE["center_m"][2],
        half_x_m=PROBE_V1_NEGATIVE_LOBE["half_m"][0],
        half_y_m=PROBE_V1_NEGATIVE_LOBE["half_m"][1],
        half_z_m=PROBE_V1_NEGATIVE_LOBE["half_m"][2],
        aperture_width_m=aperture_width_m,
    )
    positive = build_lobe(
        center_x_m=PROBE_V1_POSITIVE_LOBE["center_m"][0],
        center_y_m=PROBE_V1_POSITIVE_LOBE["center_m"][1],
        center_z_m=PROBE_V1_POSITIVE_LOBE["center_m"][2],
        half_x_m=PROBE_V1_POSITIVE_LOBE["half_m"][0],
        half_y_m=PROBE_V1_POSITIVE_LOBE["half_m"][1],
        half_z_m=PROBE_V1_POSITIVE_LOBE["half_m"][2],
        aperture_width_m=aperture_width_m,
    )
    assembly = build_assembly(
        [negative, positive], aperture_width_m=aperture_width_m
    )
    stems = [item for item in assembly["components"] if item["role"] == "stem" and item["active"]]
    stem_ys = tuple(sorted(float(item["center_m"][1]) for item in stems))
    if abs(stem_ys[0] - PROBE_V1_STEM_Y_M[0]) > 1e-9 or abs(stem_ys[1] - PROBE_V1_STEM_Y_M[1]) > 1e-9:
        raise ValueError(f"probe-v1 stems are {stem_ys}, expected {PROBE_V1_STEM_Y_M}")
    assembly["probe_label"] = "probe_v1_invalid_panel_overlap"
    return assembly


def planning_probe_assembly(
    *,
    aperture_width_m: float = DEFAULT_APERTURE_WIDTH_M,
) -> dict[str, Any]:
    negative = build_lobe(
        center_x_m=PROBE_NEGATIVE_LOBE["center_m"][0],
        center_y_m=PROBE_NEGATIVE_LOBE["center_m"][1],
        center_z_m=PROBE_NEGATIVE_LOBE["center_m"][2],
        half_x_m=PROBE_NEGATIVE_LOBE["half_m"][0],
        half_y_m=PROBE_NEGATIVE_LOBE["half_m"][1],
        half_z_m=PROBE_NEGATIVE_LOBE["half_m"][2],
        aperture_width_m=aperture_width_m,
    )
    positive = build_lobe(
        center_x_m=PROBE_POSITIVE_LOBE["center_m"][0],
        center_y_m=PROBE_POSITIVE_LOBE["center_m"][1],
        center_z_m=PROBE_POSITIVE_LOBE["center_m"][2],
        half_x_m=PROBE_POSITIVE_LOBE["half_m"][0],
        half_y_m=PROBE_POSITIVE_LOBE["half_m"][1],
        half_z_m=PROBE_POSITIVE_LOBE["half_m"][2],
        aperture_width_m=aperture_width_m,
    )
    assembly = build_assembly(
        [negative, positive], aperture_width_m=aperture_width_m
    )
    stems = [item for item in assembly["components"] if item["role"] == "stem" and item["active"]]
    stem_ys = tuple(sorted(float(item["center_m"][1]) for item in stems))
    if abs(stem_ys[0] - PROBE_STEM_Y_M[0]) > 1e-9 or abs(stem_ys[1] - PROBE_STEM_Y_M[1]) > 1e-9:
        raise ValueError(f"probe stems are {stem_ys}, expected {PROBE_STEM_Y_M}")
    assembly["probe_label"] = "probe_v2"
    return assembly


def lattice_raw_count() -> int:
    n_x = len(arange_inclusive(*CENTER_X_BOUNDS_M, CENTER_X_STEP_M))
    n_y = len(signed_center_y_values())
    n_z = len(arange_inclusive(*CENTER_Z_BOUNDS_M, CENTER_Z_STEP_M))
    return int(
        n_x
        * len(HALF_X_CHOICES_M)
        * n_y
        * len(HALF_Y_CHOICES_M)
        * n_z
        * len(HALF_Z_CHOICES_M)
    )


def enumerate_lobes(
    *,
    aperture_width_m: float = DEFAULT_APERTURE_WIDTH_M,
) -> tuple[dict[str, Any], ...]:
    lobes: list[dict[str, Any]] = []
    for center_x in arange_inclusive(*CENTER_X_BOUNDS_M, CENTER_X_STEP_M):
        for half_x in HALF_X_CHOICES_M:
            for center_y in signed_center_y_values():
                for half_y in HALF_Y_CHOICES_M:
                    for center_z in arange_inclusive(*CENTER_Z_BOUNDS_M, CENTER_Z_STEP_M):
                        for half_z in HALF_Z_CHOICES_M:
                            try:
                                lobe = build_lobe(
                                    center_x_m=float(center_x),
                                    center_y_m=float(center_y),
                                    center_z_m=float(center_z),
                                    half_x_m=float(half_x),
                                    half_y_m=float(half_y),
                                    half_z_m=float(half_z),
                                    aperture_width_m=aperture_width_m,
                                )
                            except ValueError:
                                continue
                            lobes.append(lobe)
    probe = planning_probe_assembly(aperture_width_m=aperture_width_m)
    probe_keys = {
        tuple(item["key"])
        for item in probe["components"]
        if item["role"] == "lobe" and item["active"]
    }
    have = {tuple(item["key"]) for item in lobes}
    if not probe_keys <= have:
        raise RuntimeError("planning-probe lobes are missing from the enumerator")
    lobes.sort(key=lambda item: item["key"])
    return tuple(lobes)


def necessity_bit(cell_index: int, outbound: bool) -> int:
    return 1 << (int(cell_index) * 2 + int(outbound))


def covers_all_necessity(bits: Iterable[int]) -> bool:
    acc = 0
    for value in bits:
        acc |= int(value)
    return acc == NECESSITY_ALL_BITS


def next_search_family(
    *,
    two_lobe_exact_survivors: Sequence[Any],
    two_lobe_failed_later: bool,
) -> str | None:
    """Escalate two→three only when exact two-lobe is empty and later stages did not fail."""
    if two_lobe_failed_later:
        return None
    if two_lobe_exact_survivors:
        return None
    return "three_lobe"


def stream_covering_two_lobe_pairs(
    lobes: Sequence[dict[str, Any]],
    bits_by_key: dict[tuple[float, ...], int],
) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    """Opposite-side pairs that AABB-or-exact-OR cover all twelve necessity bits."""
    from collections import defaultdict

    by_x: dict[float, dict[str, list[dict[str, Any]]]] = {}
    for lobe in lobes:
        center_x = round_m(lobe["center_m"][0])
        bucket = by_x.setdefault(center_x, {"negative": [], "positive": []})
        bucket[lobe["side"]].append(lobe)
    for center_x in sorted(by_x):
        negatives = by_x[center_x]["negative"]
        positives = by_x[center_x]["positive"]
        pos_by_mask: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for positive in positives:
            pos_by_mask[int(bits_by_key[tuple(positive["key"])])].append(positive)
        for negative in negatives:
            negative_bits = int(bits_by_key[tuple(negative["key"])])
            for positive_bits, group in pos_by_mask.items():
                if (negative_bits | positive_bits) != NECESSITY_ALL_BITS:
                    continue
                for positive in group:
                    yield negative, positive


def stream_covering_three_lobe_sets(
    lobes: Sequence[dict[str, Any]],
    bits_by_key: dict[tuple[float, ...], int],
) -> Iterator[tuple[dict[str, Any], ...]]:
    """Mixed-side triples whose bit-OR covers all twelve necessity bits."""
    from collections import defaultdict

    by_x: dict[float, list[dict[str, Any]]] = {}
    for lobe in lobes:
        by_x.setdefault(round_m(lobe["center_m"][0]), []).append(lobe)
    for center_x in sorted(by_x):
        group = by_x[center_x]
        buckets: dict[int, list[int]] = defaultdict(list)
        for index, lobe in enumerate(group):
            buckets[int(bits_by_key[tuple(lobe["key"])])].append(index)
        mask_values = sorted(buckets)
        seen: set[tuple[int, int, int]] = set()
        for a, mask_a in enumerate(mask_values):
            for b in range(a, len(mask_values)):
                mask_b = mask_values[b]
                for c in range(b, len(mask_values)):
                    mask_c = mask_values[c]
                    if (mask_a | mask_b | mask_c) != NECESSITY_ALL_BITS:
                        continue
                    for i in buckets[mask_a]:
                        for j in buckets[mask_b]:
                            for k in buckets[mask_c]:
                                ids = tuple(sorted({i, j, k}))
                                if len(ids) != 3 or ids in seen:
                                    continue
                                seen.add(ids)
                                triple = tuple(group[idx] for idx in ids)
                                sides = {item["side"] for item in triple}
                                if "negative" in sides and "positive" in sides:
                                    yield triple


def stream_two_lobe_pairs(
    lobes: Sequence[dict[str, Any]],
) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    by_x: dict[float, dict[str, list[dict[str, Any]]]] = {}
    for lobe in lobes:
        center_x = round_m(lobe["center_m"][0])
        bucket = by_x.setdefault(center_x, {"negative": [], "positive": []})
        bucket[lobe["side"]].append(lobe)
    for center_x in sorted(by_x):
        negatives = by_x[center_x]["negative"]
        positives = by_x[center_x]["positive"]
        for negative in negatives:
            for positive in positives:
                yield negative, positive


def stream_three_lobe_sets(
    lobes: Sequence[dict[str, Any]],
) -> Iterator[tuple[dict[str, Any], ...]]:
    by_x: dict[float, list[dict[str, Any]]] = {}
    for lobe in lobes:
        by_x.setdefault(round_m(lobe["center_m"][0]), []).append(lobe)
    for center_x in sorted(by_x):
        group = by_x[center_x]
        n = len(group)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    triple = (group[i], group[j], group[k])
                    sides = {item["side"] for item in triple}
                    if "negative" in sides and "positive" in sides:
                        yield triple
