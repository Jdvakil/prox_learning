#!/usr/bin/env python3
"""W2b: why no clustered hazard is admissible, and by how much it misses.

The W2 sweep admits 100+ outbound placements and zero inbound ones.  The cause
is not the cluster recipe.  It is that the band where the skin resolves an
inbound hazard is the same depth band the intrusion panel occupies, and the
paired-side design requires the clutter layout to be identical under both panel
sides -- so no bench-standing object taller than the panel's underside may sit
there at all.

This script measures that claim three ways, all replay-only against the W1
cache:

``occupancy``   how much of the sensed inbound band the panel envelope removes;
``width_ladder``  at the geometry-feasible deep band, how wide a hazard must be
                  before three sensors clear 2 px and the sides balance;
``height_ladder`` at the panel's own depth band, how tall a hazard may be before
                  it fouls the panel, and what it is worth at that height.

Nothing here authorizes a gate, collection, or V1b.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "scripts", ROOT / "submodules" / "molmospaces"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import pact_skin_resolvability as psr  # noqa: E402
import pact_place_v96_cluster_contract as v96  # noqa: E402
from pact_place_corridor_contract import sha256_file  # noqa: E402
from pact_place_v9_contract import sha256_payload  # noqa: E402
from run_pact_place_v9_w2_cluster_siting import (  # noqa: E402
    MAX_SIDE_IMBALANCE,
    MIN_DISTINCT_SENSORS,
    SUBTENSE_THRESHOLD_PX,
    leg_masks,
)

DEFAULT_CACHE_ROOT = ROOT / "diagnostics_output" / "pact_place_v9_w1_resolvability_full"
DEFAULT_OUTPUT_ROOT = ROOT / "diagnostics_output" / "pact_place_v9_w2b_inbound_diagnostic"
BENCH_TOP_Z = 0.72
PANEL_UNDERSIDE_Z = 0.80


def _load(cache_root: Path) -> dict[str, Any]:
    variants = {}
    for path in sorted((cache_root / "cache").glob("*.npz")):
        family_id, side = path.stem.rsplit("_", 1)
        data = np.load(path)
        variants[(family_id, side)] = {
            "cam_pos": data["cam_pos"].astype(np.float64),
            "cam_xmat": data["cam_xmat"].astype(np.float64),
            "masks": leg_masks([str(value) for value in data["policy_phases"]]),
        }
    names = [str(value) for value in np.load(next((cache_root / "cache").glob("*.npz")))["sensor_names"]]
    if len(variants) != 8:
        raise RuntimeError(f"expected 8 cached variants, found {len(variants)}")
    return {"variants": variants, "sensor_names": np.asarray(names)}


def _score(cache: dict[str, Any], role: str, boxes, max_range_m: float) -> dict[str, Any]:
    names = cache["sensor_names"]
    per = {}
    for (family_id, side), variant in cache["variants"].items():
        mask = variant["masks"][role]
        scored = psr.screen_candidate(
            variant["cam_pos"][mask], variant["cam_xmat"][mask], boxes, max_range_m
        )
        clears = scored["image_span_px"] >= SUBTENSE_THRESHOLD_PX
        per[f"{family_id}|{side}"] = {
            "side": side,
            "n_sensors_ge_2px": int(clears.any(axis=0).sum()),
            "n_sensor_frames_ge_2px": int(clears.sum()),
            "total_pixel_hits": int(scored["pixel_hits"].sum()),
            "sensors_ge_2px": sorted(names[clears.any(axis=0)].tolist()),
        }
    left = sum(v["total_pixel_hits"] for v in per.values() if v["side"] == "left")
    right = sum(v["total_pixel_hits"] for v in per.values() if v["side"] == "right")
    low, high = min(left, right), max(left, right)
    return {
        "per_variant": per,
        "min_n_sensors_ge_2px": min(v["n_sensors_ge_2px"] for v in per.values()),
        "min_n_sensor_frames_ge_2px": min(v["n_sensor_frames_ge_2px"] for v in per.values()),
        "min_total_pixel_hits": min(v["total_pixel_hits"] for v in per.values()),
        "left_pixel_hits": left,
        "right_pixel_hits": right,
        "side_imbalance_ratio": float(high / low) if low > 0 else None,
        "clears_floor": bool(
            min(v["n_sensors_ge_2px"] for v in per.values()) >= MIN_DISTINCT_SENSORS
            and low > 0
            and high / low <= MAX_SIDE_IMBALANCE
        ),
    }


def _slab(center_xy, half_xy, z_low, z_high):
    x, y = float(center_xy[0]), float(center_xy[1])
    hx, hy = float(half_xy[0]), float(half_xy[1])
    return [(np.array([x - hx, y - hy, z_low]), np.array([x + hx, y + hy, z_high]))]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-range-m", type=float, default=4.0)
    args = parser.parse_args()
    cache = _load(args.cache_root.resolve())
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    # --- 1. how much of the corridor the panel envelope removes ---------------
    occupancy = []
    for side in ("left", "right"):
        low, high = v96.panel_envelope(side)
        occupancy.append(
            {
                "side": side,
                "panel_low_m": [float(v) for v in low],
                "panel_high_m": [float(v) for v in high],
            }
        )
    panel_left = v96.panel_envelope("left")
    panel_x_band = [float(panel_left[0][0]), float(panel_left[1][0])]
    forbidden_lateral = float(panel_left[0][1])

    # --- 2. width ladder at the geometry-feasible deep band -------------------
    width_ladder = []
    for x in (0.750, 0.800):
        for y in (0.24, 0.28, -0.24, -0.28):
            for half_y in (0.05, 0.10, 0.15, 0.20, 0.30, 0.40):
                boxes = _slab((x, y), (0.05, half_y), BENCH_TOP_Z + 0.001, BENCH_TOP_Z + 0.247)
                score = _score(cache, "inbound_cluster", boxes, args.max_range_m)
                width_ladder.append(
                    {
                        "center_xy_m": [x, y],
                        "frontal_width_m": round(2 * half_y, 3),
                        "height_m": 0.247,
                        **{k: v for k, v in score.items() if k != "per_variant"},
                    }
                )

    # --- 3. height ladder inside the panel's own depth band -------------------
    height_ladder = []
    for x in (0.560, 0.600, 0.640, 0.680):
        for y in (0.26, 0.30, 0.34, -0.26, -0.30, -0.34):
            for top in (0.760, 0.790, 0.850, 0.920, 0.967):
                fouls_panel = bool(
                    top > PANEL_UNDERSIDE_Z - 0.010
                    and panel_x_band[0] - 0.010 < x + 0.06
                    and x - 0.06 < panel_x_band[1] + 0.010
                    and abs(y) + 0.16 > forbidden_lateral - 0.010
                )
                boxes = _slab((x, y), (0.06, 0.16), BENCH_TOP_Z + 0.001, top)
                score = _score(cache, "inbound_cluster", boxes, args.max_range_m)
                height_ladder.append(
                    {
                        "center_xy_m": [x, y],
                        "top_z_m": top,
                        "height_above_bench_m": round(top - BENCH_TOP_Z, 3),
                        "frontal_width_m": 0.32,
                        "fouls_panel_envelope_on_one_side": fouls_panel,
                        **{k: v for k, v in score.items() if k != "per_variant"},
                    }
                )

    # --- 4. the corridor's width budget, per panel side and per depth --------
    budget = []
    for side in ("left", "right"):
        panel_low, panel_high = v96.panel_envelope(side)
        for x in np.round(np.arange(0.50, 0.8251, 0.025), 4):
            blocked = []
            if panel_low[0] - v96.ARM_OBSTACLE_CLEARANCE_M <= x <= panel_high[0] + v96.ARM_OBSTACLE_CLEARANCE_M:
                blocked.append(
                    (
                        float(panel_low[1]) - v96.ARM_OBSTACLE_CLEARANCE_M,
                        float(panel_high[1]) + v96.ARM_OBSTACLE_CLEARANCE_M,
                    )
                )
            free = v96._free_intervals(
                blocked,
                -v96.APERTURE_HALF_WIDTH_M + v96.APERTURE_EDGE_RESERVE_M,
                v96.APERTURE_HALF_WIDTH_M - v96.APERTURE_EDGE_RESERVE_M,
            )
            widest = max((high - low for low, high in free), default=0.0)
            spare = widest - 2.0 * v96.ARM_ENVELOPE_HALF_Y_M - v96.ARM_OBSTACLE_CLEARANCE_M
            budget.append(
                {
                    "panel_side": side,
                    "x_m": float(x),
                    "panel_spans_this_depth": bool(blocked),
                    "widest_free_band_m": round(float(widest), 4),
                    "max_hazard_width_leaving_a_lane_m": round(float(max(0.0, spare)), 4),
                    "resolving_floor_m": v96.MIN_CLUSTER_SPAN_M,
                    "shortfall_m": round(float(v96.MIN_CLUSTER_SPAN_M - max(0.0, spare)), 4),
                }
            )
    at_panel_depth = [
        item for item in budget if item["panel_spans_this_depth"]
    ]
    worst_budget = min(
        item["max_hazard_width_leaving_a_lane_m"] for item in at_panel_depth
    )
    best_budget = max(item["max_hazard_width_leaving_a_lane_m"] for item in budget)

    admissible_deep = [item for item in width_ladder if item["clears_floor"]]
    admissible_low = [
        item
        for item in height_ladder
        if item["clears_floor"] and not item["fouls_panel_envelope_on_one_side"]
    ]
    document = {
        "schema_version": "pact_place_v9_w2b_inbound_diagnostic_v1",
        "role": "non_authorizing_inbound_siting_diagnostic",
        "authorizes_gate": False,
        "authorizes_collection": False,
        "authorizes_v1b": False,
        "instrument_sha256": sha256_file(ROOT / "scripts/pact_skin_resolvability.py"),
        "floor_under_test": {
            "min_distinct_sensors_ge_2px_per_variant": MIN_DISTINCT_SENSORS,
            "max_side_imbalance_ratio": MAX_SIDE_IMBALANCE,
            "subtense_threshold_px": SUBTENSE_THRESHOLD_PX,
        },
        "panel_envelope": occupancy,
        "panel_depth_band_m": panel_x_band,
        "panel_underside_z_m": PANEL_UNDERSIDE_Z,
        "bench_top_z_m": BENCH_TOP_Z,
        "max_bench_standing_height_clear_of_panel_m": round(
            PANEL_UNDERSIDE_Z - 0.010 - BENCH_TOP_Z, 3
        ),
        "width_ladder": width_ladder,
        "height_ladder": height_ladder,
        "n_deep_band_widths_clearing_floor": len(admissible_deep),
        "min_width_clearing_floor_in_deep_band_m": (
            min(item["frontal_width_m"] for item in admissible_deep) if admissible_deep else None
        ),
        "n_low_placements_clearing_floor": len(admissible_low),
        "low_placements_clearing_floor": admissible_low[:12],
        "corridor_width_budget": budget,
        "arm_envelope_width_m": 2.0 * v96.ARM_ENVELOPE_HALF_Y_M,
        "arm_obstacle_clearance_m": v96.ARM_OBSTACLE_CLEARANCE_M,
        "usable_aperture_width_m": 2.0
        * (v96.APERTURE_HALF_WIDTH_M - v96.APERTURE_EDGE_RESERVE_M),
        "max_hazard_width_at_panel_depth_m": worst_budget,
        "max_hazard_width_anywhere_in_corridor_m": best_budget,
        "resolving_floor_m": v96.MIN_CLUSTER_SPAN_M,
        "corridor_shortfall_at_panel_depth_m": round(
            v96.MIN_CLUSTER_SPAN_M - worst_budget, 4
        ),
    }
    document["document_sha256"] = sha256_payload(psr.jsonable(document))
    path = output_root / "inbound_diagnostic.json"
    path.write_text(json.dumps(psr.jsonable(document), indent=2, sort_keys=True) + "\n")
    print(path, flush=True)
    print(
        json.dumps(
            {
                "min_width_clearing_floor_in_deep_band_m": document[
                    "min_width_clearing_floor_in_deep_band_m"
                ],
                "max_bench_standing_height_clear_of_panel_m": document[
                    "max_bench_standing_height_clear_of_panel_m"
                ],
                "n_low_placements_clearing_floor": document["n_low_placements_clearing_floor"],
                "max_hazard_width_at_panel_depth_m": worst_budget,
                "corridor_shortfall_at_panel_depth_m": document[
                    "corridor_shortfall_at_panel_depth_m"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
