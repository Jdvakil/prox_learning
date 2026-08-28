#!/usr/bin/env python3
"""Occlusion-free V9.8 pendant siting sweep over the frozen W1 trajectories."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "scripts", ROOT / "submodules" / "molmospaces"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import pact_skin_resolvability as psr  # noqa: E402
from pact_place_v98_pendant_contract import (  # noqa: E402
    ADMISSION_FLOOR,
    MAX_SIDE_IMBALANCE,
    MIN_DETOUR_SLACK_M,
    MIN_ROUTE_INTRUSION_FRAMES,
    PHYSICS_CLEAN_FAMILIES,
    PENDANT_BOTTOM_Z_BOUNDS_M,
    PENDANT_CENTER_X_M,
    PENDANT_DEPTH_BOUNDS_M,
    PENDANT_HALF_X_M,
    PENDANT_HALF_Y_BOUNDS_M,
    build_pendant_fixture,
    fixture_bow_detour_slack_m,
    fixture_bow_lateral_limit_m,
    fixture_bow_waypoint_abs_y_m,
    pendant_aabb,
)
from pact_place_corridor_contract import sha256_file, sha256_payload  # noqa: E402
from run_pact_place_v9_w2_cluster_siting import _load_cache  # noqa: E402

DEFAULT_CACHE_ROOT = ROOT / "diagnostics_output/pact_place_v9_w1_resolvability_full"
DEFAULT_OUTPUT_ROOT = ROOT / "diagnostics_output/pact_place_v98_pendant_siting"
MAX_RANGE_M = 4.0


def _clean_cache(cache_root: Path) -> dict[str, Any]:
    cache = _load_cache(cache_root)
    allowed = {
        (family, side)
        for family in PHYSICS_CLEAN_FAMILIES
        for side in ("left", "right")
    }
    variants = {
        key: value for key, value in cache["variants"].items() if key in allowed
    }
    if set(variants) != allowed:
        raise RuntimeError(
            f"W1 cache is missing one of the six clean variants: {sorted(set(allowed) - set(variants))}"
        )
    return {"variants": variants, "sensor_names": cache["sensor_names"]}


def _route_intrusion_frames(
    cam_pos: np.ndarray, low: np.ndarray, high: np.ndarray
) -> int:
    inside = np.all(
        (cam_pos >= low[None, None, :]) & (cam_pos <= high[None, None, :]),
        axis=-1,
    )
    return int(np.any(inside, axis=1).sum())


def score_candidate(
    cache: dict[str, Any], fixture: dict[str, Any], max_range_m: float = MAX_RANGE_M
) -> dict[str, Any]:
    low, high = (np.asarray(value, dtype=float) for value in pendant_aabb(fixture))
    boxes = [(low, high)]
    sensor_names = np.asarray(cache["sensor_names"])
    per_variant: dict[str, dict[str, Any]] = {}
    for (family_id, side), variant in sorted(cache["variants"].items()):
        scored = psr.screen_candidate(
            variant["cam_pos"], variant["cam_xmat"], boxes, max_range_m
        )
        clears = scored["image_span_px"] >= 2.0
        hits = scored["pixel_hits"]
        responders = hits.sum(axis=0) > 0
        route_frames = _route_intrusion_frames(variant["cam_pos"], low, high)
        per_variant[f"{family_id}|{side}"] = {
            "family_id": family_id,
            "side": side,
            "n_sensors_ge_2px": int(clears.any(axis=0).sum()),
            "sensors_ge_2px": sorted(sensor_names[clears.any(axis=0)].tolist()),
            "n_sensor_frames_ge_2px": int(clears.sum()),
            "total_pixel_hits": int(hits.sum()),
            "n_responding_sensors": int(responders.sum()),
            "responding_sensors": sorted(sensor_names[responders].tolist()),
            "corridor_link_responders": sorted(
                {
                    str(name).split("_sensor_", 1)[0]
                    for name in sensor_names[responders]
                    if str(name).split("_sensor_", 1)[0]
                    in {"link5_front", "link5_back", "link6"}
                }
            ),
            "route_intrusion_frames": route_frames,
            "n_trajectory_frames": int(variant["cam_pos"].shape[0]),
            "max_image_span_px": float(scored["image_span_px"].max()),
            "max_subtense_px": float(scored["subtense_px"].max()),
        }
    entries = list(per_variant.values())
    by_side = {}
    for side in ("left", "right"):
        side_entries = [entry for entry in entries if entry["side"] == side]
        by_side[side] = {
            "sum_total_pixel_hits": int(sum(e["total_pixel_hits"] for e in side_entries)),
            "min_n_sensors_ge_2px": int(min(e["n_sensors_ge_2px"] for e in side_entries)),
            "min_n_sensor_frames_ge_2px": int(min(e["n_sensor_frames_ge_2px"] for e in side_entries)),
            "min_route_intrusion_frames": int(min(e["route_intrusion_frames"] for e in side_entries)),
        }
    side_totals = [by_side[side]["sum_total_pixel_hits"] for side in ("left", "right")]
    low_hits, high_hits = min(side_totals), max(side_totals)
    route_values = [entry["route_intrusion_frames"] for entry in entries]
    half_y = float(fixture["half_m"][1])
    detour_slack_m = float(fixture_bow_detour_slack_m(half_y))
    return {
        "fixture": fixture,
        "aabb_low_m": low.tolist(),
        "aabb_high_m": high.tolist(),
        "per_variant": per_variant,
        "by_side": by_side,
        "worst_variant_n_sensors_ge_2px": int(min(e["n_sensors_ge_2px"] for e in entries)),
        "worst_variant_sensor_frames_ge_2px": int(min(e["n_sensor_frames_ge_2px"] for e in entries)),
        "worst_variant_total_pixel_hits": int(min(e["total_pixel_hits"] for e in entries)),
        "worst_variant_route_intrusion_frames": int(min(route_values)),
        "side_imbalance_ratio": float(high_hits / low_hits) if low_hits else None,
        "detour_slack_m": detour_slack_m,
        "fixture_bow_waypoint_abs_y_m": float(fixture_bow_waypoint_abs_y_m(half_y)),
        "fixture_bow_lateral_limit_m": float(fixture_bow_lateral_limit_m()),
        "corridor_link_responds_every_variant": bool(
            all(entry["corridor_link_responders"] for entry in entries)
        ),
        "meets_selection_constraints": bool(
            min(route_values) >= MIN_ROUTE_INTRUSION_FRAMES
            and detour_slack_m + 1e-9 >= MIN_DETOUR_SLACK_M
            and min(e["n_sensors_ge_2px"] for e in entries)
            >= ADMISSION_FLOOR["min_distinct_changed_sensors_per_role_side"]
            and low_hits > 0
            and high_hits / low_hits <= MAX_SIDE_IMBALANCE
        ),
    }


def _candidate_records(cache: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    for bottom in np.round(np.arange(PENDANT_BOTTOM_Z_BOUNDS_M[0], PENDANT_BOTTOM_Z_BOUNDS_M[1] + 0.0001, 0.01), 2):
        for half_y in np.round(np.arange(PENDANT_HALF_Y_BOUNDS_M[0], PENDANT_HALF_Y_BOUNDS_M[1] + 0.0001, 0.01), 2):
            fixture = build_pendant_fixture(
                bottom_z_m=float(bottom), half_y_m=float(half_y),
                center_x_m=PENDANT_CENTER_X_M, half_x_m=PENDANT_HALF_X_M,
            )
            record = score_candidate(cache, fixture)
            record["candidate_id"] = f"bottom{bottom:.2f}_half_y{half_y:.2f}"
            record["selection_rank"] = None
            records.append(record)
    return records


def select_candidate(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [record for record in records if record["meets_selection_constraints"]]
    eligible.sort(
        key=lambda record: (
            -record["worst_variant_n_sensors_ge_2px"],
            -record["worst_variant_sensor_frames_ge_2px"],
            -record["worst_variant_total_pixel_hits"],
            record["side_imbalance_ratio"] or float("inf"),
        )
    )
    for rank, record in enumerate(eligible, start=1):
        record["selection_rank"] = rank
    return eligible[0] if eligible else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    cache_root = args.cache_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    cache = _clean_cache(cache_root)
    records = _candidate_records(cache)
    selected = select_candidate(records)
    document = {
        "schema_version": "pact_place_v9_8_pendant_siting_v2",
        "role": "occlusion_free_siting_screen_not_admission",
        "authorizes_gate": False,
        "authorizes_collection": False,
        "cache_root": str(cache_root.relative_to(ROOT)),
        "cache_sha256": sha256_file(cache_root / "cache" / "F0_target_side_stagger_left.npz"),
        "variants": sorted(f"{family}|{side}" for family, side in cache["variants"]),
        "candidate_count": len(records),
        "selection_rule": {
            "maximize": [
                "worst_variant_n_sensors_ge_2px",
                "worst_variant_sensor_frames_ge_2px",
                "worst_variant_total_pixel_hits",
            ],
            "min_route_intrusion_frames_every_variant": MIN_ROUTE_INTRUSION_FRAMES,
            "min_detour_slack_m": MIN_DETOUR_SLACK_M,
            "min_worst_variant_sensors_ge_2px": ADMISSION_FLOOR[
                "min_distinct_changed_sensors_per_role_side"
            ],
            "max_side_imbalance": MAX_SIDE_IMBALANCE,
        },
        "previous_selection_admitted_zero_slack": {
            "half_y_m": 0.18,
            "detour_slack_m": 0.0,
            "artifact": "diagnostics_output/pact_place_v98_pendant_siting/siting.json",
            "note": (
                "The first S1 rule maximised sensors subject to intrusion "
                ">= 100 and admitted half_y=0.18, which lands the fixture-bow "
                "waypoint exactly on the lateral limit with zero slack."
            ),
        },
        "records": records,
        "selected": selected,
        "selected_candidate_id": selected["candidate_id"] if selected else None,
    }
    document["document_sha256"] = sha256_payload(document)
    path = output_root / "siting.json"
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(path)
    print(json.dumps({
        "candidate_count": len(records),
        "selected_candidate_id": document["selected_candidate_id"],
        "selected_worst_sensors": selected["worst_variant_n_sensors_ge_2px"] if selected else None,
        "selected_worst_intrusion_frames": selected["worst_variant_route_intrusion_frames"] if selected else None,
    }, sort_keys=True))
    return 0 if selected else 2


if __name__ == "__main__":
    raise SystemExit(main())
