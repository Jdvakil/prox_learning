#!/usr/bin/env python3
"""E1: subtense-first siting -- search the region the span contract excluded.

V9.6's sweep required every candidate to span >= 0.25 m and to clear 2 px on at
least three distinct sensors.  Neither number is a property of the sensor.  The
first made every candidate too wide to fit the corridor's 0.120 m budget by
construction; the second was never validated against what the policy needs.

E1 changes exactly those two contracts and nothing else:

* **span floor dropped.**  Any composition of one to four accepted vessels is a
  candidate, widths 0.051-0.296 m.  A 0.120 m hazard clears 2 px out to
  R = 0.58 m and the ranges W1 measured are 0.11-0.14 m, so a hazard that fits
  the corridor is resolvable at the ranges that actually occur.
* **sensor-count floor dropped.**  Every candidate's distinct-sensor count and
  changed-value total is recorded and the distribution published, so the floor
  can be chosen against evidence rather than invented again.

Retained from W2 unchanged: the scoring path (`_score_placement` and
`pact_skin_resolvability` are imported, not copied), the corridor lane test,
paired-side balance, the <= 4x imbalance criterion, per-candidate rejection
reasons, and the rule that no geometry-only score is ever an admission.

Nothing here authorizes a gate, collection, or V1b.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "scripts", ROOT / "submodules" / "molmospaces"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import pact_place_v96_cluster_contract as v96  # noqa: E402
import pact_place_v97_hazard_contract as v97  # noqa: E402
import pact_skin_resolvability as psr  # noqa: E402
from pact_place_corridor_contract import sha256_file  # noqa: E402
from pact_place_v9_contract import sha256_payload  # noqa: E402

# The scoring path is imported from the W2 sweep, not reimplemented.
from run_pact_place_v9_w2_cluster_siting import (  # noqa: E402
    CORRIDOR_LINKS,
    MAX_SIDE_IMBALANCE,
    SUBTENSE_THRESHOLD_PX,
    _init_worker,
    _score_placement,
)

DEFAULT_CACHE_ROOT = ROOT / "diagnostics_output" / "pact_place_v9_w1_resolvability_full"
DEFAULT_OUTPUT_ROOT = ROOT / "diagnostics_output" / "pact_place_v9_e1_subtense_siting"


def _placement_grid() -> tuple[np.ndarray, np.ndarray, tuple[float, ...]]:
    x_values = np.round(
        np.unique(
            np.concatenate(
                [
                    np.arange(0.550, 0.8251, 0.025),
                    np.array([0.7300, 0.7350, 0.7400, 0.7420, 0.7435]),
                ]
            )
        ),
        4,
    )
    # A narrow hazard can now sit near the centreline, which the V9.6 lateral
    # band excluded outright, so resolve that region an order finer.
    y_values = np.round(
        np.unique(
            np.concatenate(
                [np.arange(-0.340, 0.3401, 0.020), np.arange(-0.100, 0.1001, 0.010)]
            )
        ),
        4,
    )
    return x_values, y_values, (0.0, 30.0, 60.0, 90.0, 120.0, 150.0)


def _candidate_job(job: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for spec in job["specs"]:
        geometry = v97.hazard_geometry(
            spec["uids"], spec["role"], spec["center_xy_m"], spec["theta_deg"], spec["gap_m"]
        )
        reasons = v97.unary_geometry_reasons(geometry, spec["role"])
        boxes = [(np.asarray(a), np.asarray(b)) for a, b in geometry["boxes"]]
        sensing = _score_placement(spec["role"], boxes, job["max_range_m"])
        ratio = sensing["side_imbalance_ratio"]
        sensing_reasons: list[str] = []
        if sensing["min_total_pixel_hits"] <= 0:
            sensing_reasons.append("silent_in_some_variant")
        if ratio is None or ratio > MAX_SIDE_IMBALANCE:
            sensing_reasons.append("left_right_imbalance_above_4x")
        out.append(
            {
                "candidate_id": spec["candidate_id"],
                "role": spec["role"],
                "composition_id": spec["composition_id"],
                "uids": list(spec["uids"]),
                "n_members": len(spec["uids"]),
                "center_xy_m": [float(v) for v in spec["center_xy_m"]],
                "theta_deg": float(spec["theta_deg"]),
                "gap_m": float(spec["gap_m"]),
                "span_along_line_m": float(geometry["span_along_line_m"]),
                "union_extent_m": [float(v) for v in geometry["union_extent_m"]],
                "union_low_m": [float(v) for v in geometry["union_low_m"]],
                "union_high_m": [float(v) for v in geometry["union_high_m"]],
                "lateral_offset_m": abs(float(geometry["union_center_m"][1])),
                "geometry_feasible": not reasons,
                "geometry_rejection_reasons": reasons,
                "sensing_rejection_reasons": sensing_reasons,
                # E1's admission: geometry must be possible and the response must
                # exist and be side-balanced.  No span floor, no sensor-count floor.
                "reachable": bool(not reasons and not sensing_reasons),
                "min_n_sensors_ge_2px": sensing["min_n_sensors_ge_2px"],
                "min_n_sensor_frames_ge_2px": sensing["min_n_sensor_frames_ge_2px"],
                "min_total_pixel_hits": sensing["min_total_pixel_hits"],
                "min_predicted_changed_values": sensing["min_total_pixel_hits"]
                * psr.PRODUCTION_SUBSTEPS,
                "side_imbalance_ratio": ratio,
                "corridor_link_responds_in_every_variant": sensing[
                    "corridor_link_responds_in_every_variant"
                ],
                "sensing": sensing,
            }
        )
    return out


def _chunks(items: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _range_stats(cache_root: Path, role: str, boxes, max_range_m: float) -> dict[str, Any]:
    """Ranges at which the hazard is actually in frame, for the shortlist only."""
    from run_pact_place_v9_w2_cluster_siting import _load_cache

    cache = _load_cache(cache_root)
    per_variant = {}
    for (family_id, side), variant in cache["variants"].items():
        mask = variant["masks"][role]
        scored = psr.screen_candidate(
            variant["cam_pos"][mask], variant["cam_xmat"][mask], boxes, max_range_m
        )
        visible = scored["image_span_px"] >= SUBTENSE_THRESHOLD_PX
        ranges = scored["range_m"][visible]
        ranges = ranges[np.isfinite(ranges)]
        per_variant[f"{family_id}|{side}"] = {
            "n_sensor_frames_ge_2px": int(visible.sum()),
            "min_range_m": float(ranges.min()) if ranges.size else None,
            "median_range_m": float(np.median(ranges)) if ranges.size else None,
            "max_range_m": float(ranges.max()) if ranges.size else None,
        }
    allr = [v["median_range_m"] for v in per_variant.values() if v["median_range_m"] is not None]
    return {
        "per_variant": per_variant,
        "median_of_median_range_m": float(np.median(allr)) if allr else None,
        "width_needed_for_2px_at_that_range_m": (
            float(2.0 * psr.PIXEL_PITCH_COEFF * np.median(allr)) if allr else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-range-m", type=float, default=4.0)
    parser.add_argument("--top-k", type=int, default=25)
    args = parser.parse_args()

    cache_root = args.cache_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    x_values, y_values, theta_values = _placement_grid()

    specs: list[dict[str, Any]] = []
    for role, table in (
        ("inbound_cluster", v97.INBOUND_COMPOSITIONS),
        ("outbound_cluster", v97.OUTBOUND_COMPOSITIONS),
    ):
        for composition_id, uids in table.items():
            # A single member sits at the placement centre whatever the line
            # angle, so sweeping theta for it would only duplicate candidates.
            angles = (0.0,) if len(uids) == 1 else theta_values
            for x in x_values:
                for y in y_values:
                    for theta in angles:
                        specs.append(
                            {
                                "candidate_id": (
                                    f"{composition_id}:{role}:x{x:.4f}:y{y:+.3f}:t{theta:03.0f}"
                                ),
                                "composition_id": composition_id,
                                "role": role,
                                "uids": uids,
                                "center_xy_m": (float(x), float(y)),
                                "theta_deg": float(theta),
                                "gap_m": v97.DEFAULT_GAP_M,
                            }
                        )
    print(f"sweeping {len(specs)} hazard placements", flush=True)

    jobs = [
        {"specs": chunk, "max_range_m": float(args.max_range_m)}
        for chunk in _chunks(specs, 64)
    ]
    records: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max(1, args.workers),
        initializer=_init_worker,
        initargs=(str(cache_root),),
    ) as executor:
        done = 0
        for chunk in executor.map(_candidate_job, jobs):
            records.extend(chunk)
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(jobs)} chunks", flush=True)

    by_role: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for record in records:
        by_role[record["role"]].append(record)

    # --- the deliverable: what is reachable, as a distribution -----------------
    def _distribution(items: list[dict[str, Any]]) -> dict[str, Any]:
        reachable = [item for item in items if item["reachable"]]
        sensor_hist = collections.Counter(item["min_n_sensors_ge_2px"] for item in reachable)
        width_bins = collections.Counter()
        for item in reachable:
            width_bins[round(item["span_along_line_m"], 2)] += 1
        changed = sorted(item["min_predicted_changed_values"] for item in reachable)
        return {
            "candidates": len(items),
            "geometry_feasible": sum(item["geometry_feasible"] for item in items),
            "reachable": len(reachable),
            "min_distinct_sensors_ge_2px_histogram": {
                str(key): int(value) for key, value in sorted(sensor_hist.items())
            },
            "reachable_by_member_count": {
                str(key): int(value)
                for key, value in sorted(
                    collections.Counter(item["n_members"] for item in reachable).items()
                )
            },
            "reachable_by_span_m": {
                f"{key:.2f}": int(value) for key, value in sorted(width_bins.items())
            },
            "corridor_link_responds_count": sum(
                item["corridor_link_responds_in_every_variant"] for item in reachable
            ),
            "min_predicted_changed_values_quantiles": (
                {
                    "min": changed[0],
                    "p25": changed[len(changed) // 4],
                    "median": changed[len(changed) // 2],
                    "p75": changed[(3 * len(changed)) // 4],
                    "max": changed[-1],
                }
                if changed
                else None
            ),
            "rejection_reason_counts": {
                key: int(value)
                for key, value in sorted(
                    collections.Counter(
                        reason
                        for item in items
                        for reason in item["geometry_rejection_reasons"]
                        + item["sensing_rejection_reasons"]
                    ).items()
                )
            },
        }

    distribution = {role: _distribution(items) for role, items in by_role.items()}

    shortlist: dict[str, list[dict[str, Any]]] = {}
    for role, items in by_role.items():
        reachable = [item for item in items if item["reachable"]]
        reachable.sort(
            key=lambda item: (
                -item["min_n_sensors_ge_2px"],
                -item["min_n_sensor_frames_ge_2px"],
                -item["min_total_pixel_hits"],
            )
        )
        shortlist[role] = reachable[: args.top_k]

    # --- pairing, with the corridor lane test retained -------------------------
    pairs: list[dict[str, Any]] = []
    for inbound in shortlist.get("inbound_cluster", []):
        for outbound in shortlist.get("outbound_cluster", []):
            if set(inbound["uids"]) & set(outbound["uids"]):
                continue
            palette = v97.build_hazard_palette(inbound["uids"], outbound["uids"])
            try:
                layout = v97.build_hazard_layout(
                    palette, family_id="E1_candidate", intrusion_side="left",
                    inbound={
                        "center_xy_m": inbound["center_xy_m"],
                        "theta_deg": inbound["theta_deg"],
                        "gap_m": inbound["gap_m"],
                    },
                    outbound={
                        "center_xy_m": outbound["center_xy_m"],
                        "theta_deg": outbound["theta_deg"],
                        "gap_m": outbound["gap_m"],
                    },
                )
                feasibility = v97.hazard_feasibility(layout)
            except ValueError as error:
                feasibility = {"feasible": False, "reasons": [f"layout_rejected:{error}"]}
            pairs.append(
                {
                    "inbound_candidate_id": inbound["candidate_id"],
                    "outbound_candidate_id": outbound["candidate_id"],
                    "inbound_uids": inbound["uids"],
                    "outbound_uids": outbound["uids"],
                    "feasible": bool(feasibility["feasible"]),
                    "rejection_reasons": feasibility.get("reasons", []),
                    "corridor_lanes": feasibility.get("corridor_lanes", {}),
                    "worst_min_n_sensors_ge_2px": min(
                        inbound["min_n_sensors_ge_2px"], outbound["min_n_sensors_ge_2px"]
                    ),
                    "worst_min_predicted_changed_values": min(
                        inbound["min_predicted_changed_values"],
                        outbound["min_predicted_changed_values"],
                    ),
                    "worst_side_imbalance_ratio": max(
                        inbound["side_imbalance_ratio"], outbound["side_imbalance_ratio"]
                    ),
                }
            )
    pairs.sort(
        key=lambda item: (
            not item["feasible"],
            -item["worst_min_n_sensors_ge_2px"],
            -item["worst_min_predicted_changed_values"],
        )
    )
    feasible_pairs = [item for item in pairs if item["feasible"]]

    # Ranges for the best feasible pair, so the subtense argument is checkable.
    best_ranges = None
    if feasible_pairs:
        best = feasible_pairs[0]
        best_ranges = {}
        for role, cid in (
            ("inbound_cluster", best["inbound_candidate_id"]),
            ("outbound_cluster", best["outbound_candidate_id"]),
        ):
            record = next(item for item in by_role[role] if item["candidate_id"] == cid)
            boxes = [
                (np.asarray(record["union_low_m"]), np.asarray(record["union_high_m"]))
            ]
            geometry = v97.hazard_geometry(
                record["uids"], role, record["center_xy_m"], record["theta_deg"],
                record["gap_m"],
            )
            boxes = [(np.asarray(a), np.asarray(b)) for a, b in geometry["boxes"]]
            best_ranges[role] = _range_stats(cache_root, role, boxes, float(args.max_range_m))

    document = {
        "schema_version": "pact_place_v9_e1_subtense_siting_v1",
        "role": "non_authorizing_subtense_first_siting_sweep",
        "authorizes_gate": False,
        "authorizes_collection": False,
        "authorizes_v1b": False,
        "contract_changes_from_w2": [
            "min_cluster_span_m removed: hazard width is no longer contracted",
            "min_distinct_sensors removed: the sensor-count floor is reported, not applied",
            "lateral offset band removed: the corridor lane test is the physical constraint",
        ],
        "retained_from_w2": [
            "scoring path imported unchanged (_score_placement, pact_skin_resolvability)",
            "corridor lane test",
            "paired-side balance and the <= 4x imbalance criterion",
            "per-candidate rejection reasons",
            "no geometry-only score is ever an admission",
        ],
        "instrument_path": "scripts/pact_skin_resolvability.py",
        "instrument_sha256": sha256_file(ROOT / "scripts/pact_skin_resolvability.py"),
        "w2_sweep_path": "scripts/run_pact_place_v9_w2_cluster_siting.py",
        "w2_sweep_sha256": sha256_file(ROOT / "scripts/run_pact_place_v9_w2_cluster_siting.py"),
        "v96_contract_sha256": sha256_file(ROOT / "scripts/pact_place_v96_cluster_contract.py"),
        "v97_contract_sha256": sha256_file(ROOT / "scripts/pact_place_v97_hazard_contract.py"),
        "subtense_threshold_px": SUBTENSE_THRESHOLD_PX,
        "max_side_imbalance_ratio": MAX_SIDE_IMBALANCE,
        "corridor_links": list(CORRIDOR_LINKS),
        "width_scope_m": list(v97.WIDTH_SCOPE_M),
        "pixel_pitch_coefficient_per_m": psr.PIXEL_PITCH_COEFF,
        "width_needed_for_2px": {
            f"R={r:.2f}": round(2.0 * psr.PIXEL_PITCH_COEFF * r, 4)
            for r in (0.15, 0.30, 0.40, 0.58)
        },
        "grid": {
            "x_values_m": [float(v) for v in x_values],
            "y_values_m": [float(v) for v in y_values],
            "theta_values_deg": list(theta_values),
            "inbound_compositions": {k: list(v) for k, v in v97.INBOUND_COMPOSITIONS.items()},
            "outbound_compositions": {k: list(v) for k, v in v97.OUTBOUND_COMPOSITIONS.items()},
        },
        "candidate_count": len(records),
        "reachability_distribution": distribution,
        "pair_count": len(pairs),
        "feasible_pair_count": len(feasible_pairs),
        "best_pair_ranges": best_ranges,
        "shortlist": shortlist,
        "pairs": pairs[:80],
        "candidates": records,
    }
    document["document_sha256"] = sha256_payload(psr.jsonable(document))
    path = output_root / "siting.json"
    path.write_text(json.dumps(psr.jsonable(document), indent=2, sort_keys=True) + "\n")
    print(path, flush=True)
    print(
        json.dumps(
            {
                "candidates": len(records),
                "reachable": {r: distribution[r]["reachable"] for r in distribution},
                "feasible_pairs": len(feasible_pairs),
                "best_pair": (
                    feasible_pairs[0]["inbound_candidate_id"]
                    + " + "
                    + feasible_pairs[0]["outbound_candidate_id"]
                )
                if feasible_pairs
                else None,
            },
            sort_keys=True,
        )
    )
    return 0 if feasible_pairs else 2


if __name__ == "__main__":
    raise SystemExit(main())
