#!/usr/bin/env python3
"""W2: site clustered hazards by angular subtense, not by clearance.

Every siting sweep before this one scored TCP clearance or link collision
clearance.  Neither decides whether the skin sees anything.  This sweep scores
**angular subtense at the 40 sensors**, replayed against the frozen V9.5
trajectories cached by W1 -- no rollouts, no physics, no rendering.

Stage 1 sweeps single-cluster placements for each leg and records every
candidate with its sensing scores and, when rejected, the reason.  Stage 2
pairs the survivors and applies the joint geometric feasibility contract
(workspace, mutual overlap, panel envelope, route blocking, corridor lane),
which is kept strictly separate from the sensing score.

Nothing here authorizes a gate, collection, or V1b.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "scripts", ROOT / "submodules" / "molmospaces"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import pact_skin_resolvability as psr  # noqa: E402
import pact_place_v96_cluster_contract as v96  # noqa: E402
from pact_place_corridor_contract import sha256_file  # noqa: E402
from pact_place_v9_contract import sha256_payload  # noqa: E402
from pact_place_v9_contract import (  # noqa: E402
    NOMINAL_OUTBOUND_END_XY_M,
    NOMINAL_OUTBOUND_START_XY_M,
)
from run_pact_place_v9_v0c3_causal_proximity import (  # noqa: E402
    INBOUND_DECISION_PHASES,
    OUTBOUND_DECISION_PHASES,
)

DEFAULT_CACHE_ROOT = ROOT / "diagnostics_output" / "pact_place_v9_w1_resolvability_full"
DEFAULT_OUTPUT_ROOT = ROOT / "diagnostics_output" / "pact_place_v9_w2_cluster_siting"
ROLE_WINDOW = {
    "inbound_cluster": INBOUND_DECISION_PHASES,
    "outbound_cluster": OUTBOUND_DECISION_PHASES,
}
#: Phases that end the inbound leg and begin the outbound leg.
GRASP_PHASES = frozenset({"grasp", "gripper-close"})
LIFT_PHASE = "lift"
RELEASE_PHASES = frozenset({"gripper-open", "retreat"})


def leg_masks(phases: Sequence[str]) -> dict[str, np.ndarray]:
    """Split a replay into the physical inbound and outbound legs.

    The V9.5 phase labels name the segment where the arm passed *that layout's*
    vessel.  Scoring a differently sited cluster inside those labels would ask
    whether the new hazard happens to sit where the old one did.  The legs are
    the honest window: everything before the grasp, and everything from the lift
    to the release.
    """
    phases = [str(value) for value in phases]
    n = len(phases)
    grasp = next((i for i, phase in enumerate(phases) if phase in GRASP_PHASES), None)
    lift = next((i for i, phase in enumerate(phases) if phase == LIFT_PHASE), None)
    if grasp is None or lift is None:
        # A decision-window replay drops the grasp itself, which would silently
        # make the whole replay count as the inbound leg.  Refuse instead.
        raise ValueError(
            "replay has no grasp or lift phase; leg windows need a full-window replay"
        )
    release = next(
        (i for i, phase in enumerate(phases) if i > lift and phase in RELEASE_PHASES), n
    )
    inbound = np.zeros(n, dtype=bool)
    inbound[:grasp] = True
    outbound = np.zeros(n, dtype=bool)
    outbound[lift:release] = True
    if not inbound.any() or not outbound.any():
        raise ValueError("replay does not contain both legs")
    return {"inbound_cluster": inbound, "outbound_cluster": outbound}
CORRIDOR_LINKS = ("link5_front", "link5_back", "link6")
SUBTENSE_THRESHOLD_PX = 2.0
MIN_DISTINCT_SENSORS = 3
MAX_SIDE_IMBALANCE = 4.0
#: Lateral band the panel's own sensed response occupies; siting head-on at
#: ``y = 0`` in front of the unsensed gripper is what failed in V9.5.
MIN_LATERAL_OFFSET_M = 0.10
MAX_LATERAL_OFFSET_M = 0.34

_CACHE: dict[str, Any] | None = None


def _load_cache(cache_root: Path) -> dict[str, Any]:
    variants: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted((cache_root / "cache").glob("*.npz")):
        family_id, side = path.stem.rsplit("_", 1)
        data = np.load(path)
        phases = [str(value) for value in data["policy_phases"]]
        variants[(family_id, side)] = {
            "cam_pos": data["cam_pos"].astype(np.float64),
            "cam_xmat": data["cam_xmat"].astype(np.float64),
            "sensor_names": [str(value) for value in data["sensor_names"]],
            "masks": leg_masks(phases),
            "decision_masks": {
                role: np.asarray([phase in window for phase in phases], dtype=bool)
                for role, window in ROLE_WINDOW.items()
            },
        }
    if len(variants) != 8:
        raise RuntimeError(f"expected 8 cached W1 variants, found {len(variants)}")
    return {"variants": variants, "sensor_names": next(iter(variants.values()))["sensor_names"]}


def _init_worker(cache_root: str) -> None:
    global _CACHE
    _CACHE = _load_cache(Path(cache_root))


def _score_placement(role: str, boxes, max_range_m: float) -> dict[str, Any]:
    """Sensing score of one posed cluster across all eight cached variants."""
    assert _CACHE is not None
    sensor_names = np.asarray(_CACHE["sensor_names"])
    per_variant: dict[str, dict[str, Any]] = {}
    for (family_id, side), variant in _CACHE["variants"].items():
        mask = variant["masks"][role]
        scored = psr.screen_candidate(
            variant["cam_pos"][mask], variant["cam_xmat"][mask], boxes, max_range_m
        )
        clears = scored["image_span_px"] >= SUBTENSE_THRESHOLD_PX
        responding = scored["pixel_hits"].sum(axis=0) > 0
        decision_mask = variant["decision_masks"][role]
        decision = psr.screen_candidate(
            variant["cam_pos"][decision_mask], variant["cam_xmat"][decision_mask],
            boxes, max_range_m,
        )
        per_variant[f"{family_id}|{side}"] = {
            "family_id": family_id,
            "side": side,
            "n_sensor_frames_ge_2px": int(clears.sum()),
            "n_sensors_ge_2px": int(clears.any(axis=0).sum()),
            "sensors_ge_2px": sorted(sensor_names[clears.any(axis=0)].tolist()),
            "max_image_span_px": float(scored["image_span_px"].max()),
            "max_subtense_px": float(scored["subtense_px"].max()),
            "total_pixel_hits": int(scored["pixel_hits"].sum()),
            "predicted_changed_values_upper_bound": int(scored["pixel_hits"].sum())
            * psr.PRODUCTION_SUBSTEPS,
            "n_responding_sensors": int(responding.sum()),
            "responding_sensors": sorted(sensor_names[responding].tolist()),
            "n_leg_frames": int(mask.sum()),
            "decision_window": {
                "n_frames": int(decision_mask.sum()),
                "n_sensor_frames_ge_2px": int(
                    (decision["image_span_px"] >= SUBTENSE_THRESHOLD_PX).sum()
                ),
                "n_sensors_ge_2px": int(
                    (decision["image_span_px"] >= SUBTENSE_THRESHOLD_PX).any(axis=0).sum()
                ),
                "total_pixel_hits": int(decision["pixel_hits"].sum()),
            },
        }
    sides: dict[str, list[dict[str, Any]]] = {"left": [], "right": []}
    for entry in per_variant.values():
        sides[entry["side"]].append(entry)
    by_side = {}
    for side, entries in sides.items():
        by_side[side] = {
            "min_n_sensor_frames_ge_2px": min(e["n_sensor_frames_ge_2px"] for e in entries),
            "min_n_sensors_ge_2px": min(e["n_sensors_ge_2px"] for e in entries),
            "min_total_pixel_hits": min(e["total_pixel_hits"] for e in entries),
            "sum_total_pixel_hits": sum(e["total_pixel_hits"] for e in entries),
            "sensors_ge_2px_in_every_family": sorted(
                set.intersection(*[set(e["sensors_ge_2px"]) for e in entries])
            ),
        }
    totals = [by_side[side]["sum_total_pixel_hits"] for side in ("left", "right")]
    low, high = min(totals), max(totals)
    corridor_ok = all(
        any(
            str(name).split("_sensor_", 1)[0] in CORRIDOR_LINKS
            for name in entry["sensors_ge_2px"]
        )
        for entry in per_variant.values()
    )
    return {
        "per_variant": per_variant,
        "by_side": by_side,
        "min_n_sensor_frames_ge_2px": min(
            entry["n_sensor_frames_ge_2px"] for entry in per_variant.values()
        ),
        "min_n_sensors_ge_2px": min(
            entry["n_sensors_ge_2px"] for entry in per_variant.values()
        ),
        "min_total_pixel_hits": min(
            entry["total_pixel_hits"] for entry in per_variant.values()
        ),
        "side_imbalance_ratio": float(high / low) if low > 0 else None,
        "corridor_link_responds_in_every_variant": bool(corridor_ok),
    }


def _candidate_job(job: dict[str, Any]) -> list[dict[str, Any]]:
    assert _CACHE is not None
    out: list[dict[str, Any]] = []
    for spec in job["specs"]:
        palette = v96.build_cluster_palette(spec["recipe_id"])
        members = [
            item for item in palette["palette"] if str(item["role"]) == spec["role"]
        ]
        geometry = v96.cluster_geometry(
            members, spec["center_xy_m"], spec["theta_deg"], spec["gap_m"]
        )
        low = np.asarray(geometry["union_low_m"])
        high = np.asarray(geometry["union_high_m"])
        reasons: list[str] = []
        if not v96.within_workspace(low, high):
            reasons.append("workspace_escape")
        if float(geometry["span_along_line_m"]) < v96.MIN_CLUSTER_SPAN_M - 1e-9:
            reasons.append("span_below_resolving_floor")
        lateral = abs(float(geometry["union_center_m"][1]))
        if lateral < MIN_LATERAL_OFFSET_M:
            reasons.append("head_on_lateral_offset_below_0p10m")
        if lateral > MAX_LATERAL_OFFSET_M:
            reasons.append("lateral_offset_beyond_0p34m")
        for side in ("left", "right"):
            panel_low, panel_high = v96.panel_envelope(side)
            if v96.boxes_overlap(low, high, panel_low, panel_high, 0.010):
                reasons.append(f"panel_envelope_overlap:{side}")
        if spec["role"] == "outbound_cluster":
            # A route blocker that the loaded leg never crosses is decor, not a
            # hazard.  This is the settled V9 criterion, applied per candidate so
            # the shortlist cannot drift behind the target.
            start, end = NOMINAL_OUTBOUND_START_XY_M, NOMINAL_OUTBOUND_END_XY_M
            t_cross = (float(geometry["union_center_m"][0]) - start[0]) / (end[0] - start[0])
            if not 0.02 < t_cross < 0.98:
                reasons.append("not_crossed_on_the_nominal_loaded_leg")
        record = {
            "candidate_id": spec["candidate_id"],
            "role": spec["role"],
            "recipe_id": spec["recipe_id"],
            "center_xy_m": [float(value) for value in spec["center_xy_m"]],
            "theta_deg": float(spec["theta_deg"]),
            "gap_m": float(spec["gap_m"]),
            "span_along_line_m": float(geometry["span_along_line_m"]),
            "union_low_m": [float(v) for v in low],
            "union_high_m": [float(v) for v in high],
            "union_extent_m": [float(v) for v in geometry["union_extent_m"]],
            "geometry_feasible": not reasons,
            "rejection_reasons": sorted(set(reasons)),
        }
        # Score sensing for every candidate, admitted or not, so the rejected
        # region is documented rather than merely asserted.
        record["sensing"] = _score_placement(
            spec["role"], [(np.asarray(a), np.asarray(b)) for a, b in geometry["boxes"]],
            job["max_range_m"],
        )
        record["boxes"] = geometry["boxes"]
        out.append(record)
    return out


def _chunks(items: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _sensing_admitted(record: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    sensing = record["sensing"]
    if sensing["min_n_sensors_ge_2px"] < MIN_DISTINCT_SENSORS:
        reasons.append("fewer_than_3_sensors_clear_2px_in_some_variant")
    if not sensing["corridor_link_responds_in_every_variant"]:
        reasons.append("no_link5_or_link6_response_in_some_variant")
    ratio = sensing["side_imbalance_ratio"]
    if ratio is None or ratio > MAX_SIDE_IMBALANCE:
        reasons.append("left_right_imbalance_above_4x")
    if sensing["min_total_pixel_hits"] <= 0:
        reasons.append("silent_in_some_variant")
    return (not reasons), reasons


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-range-m", type=float, default=4.0)
    parser.add_argument("--top-k", type=int, default=12)
    args = parser.parse_args()

    cache_root = args.cache_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    # The coarse grid plus the exact depth window left open between the panel
    # envelope (x_high = 0.685) and the far end of the nominal loaded leg
    # (t_cross = 0.02 at x = 0.7438).  That window is a few millimetres wide, so
    # a uniform grid steps straight over it.
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
    y_values = np.round(np.arange(-0.340, 0.3401, 0.020), 4)
    theta_values = (0.0, 30.0, 60.0, 90.0, 120.0, 150.0)
    specs: list[dict[str, Any]] = []
    for recipe_id in v96.CLUSTER_RECIPES:
        for role in v96.CLUSTER_ROLES:
            for x in x_values:
                for y in y_values:
                    for theta in theta_values:
                        specs.append(
                            {
                                "candidate_id": (
                                    f"{recipe_id}:{role}:x{x:.3f}:y{y:+.3f}:t{theta:03.0f}"
                                ),
                                "recipe_id": recipe_id,
                                "role": role,
                                "center_xy_m": (float(x), float(y)),
                                "theta_deg": float(theta),
                                "gap_m": v96.DEFAULT_CLUSTER_GAP_M,
                            }
                        )
    print(f"sweeping {len(specs)} cluster placements", flush=True)

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
            if done % 10 == 0:
                print(f"  {done}/{len(jobs)} chunks", flush=True)

    for record in records:
        admitted, reasons = _sensing_admitted(record)
        record["sensing_admitted"] = bool(admitted and record["geometry_feasible"])
        record["rejection_reasons"] = sorted(set(record["rejection_reasons"]) | set(reasons))

    by_role: dict[str, list[dict[str, Any]]] = {role: [] for role in v96.CLUSTER_ROLES}
    for record in records:
        by_role[record["role"]].append(record)
    shortlist: dict[str, list[dict[str, Any]]] = {}
    for role, items in by_role.items():
        admitted = [item for item in items if item["sensing_admitted"]]
        # Rank on the two quantities the plan names, worst-case over the eight
        # variants, never on a single scalar.
        admitted.sort(
            key=lambda item: (
                -item["sensing"]["min_n_sensor_frames_ge_2px"],
                -item["sensing"]["min_n_sensors_ge_2px"],
                -item["sensing"]["min_total_pixel_hits"],
            )
        )
        shortlist[role] = admitted[: args.top_k]

    pairs: list[dict[str, Any]] = []
    palette_cache = {
        recipe_id: v96.build_cluster_palette(recipe_id) for recipe_id in v96.CLUSTER_RECIPES
    }
    for inbound in shortlist.get("inbound_cluster", []):
        for outbound in shortlist.get("outbound_cluster", []):
            if inbound["recipe_id"] != outbound["recipe_id"]:
                continue
            layout = v96.build_cluster_layout(
                palette_cache[inbound["recipe_id"]],
                family_id="W2_candidate",
                intrusion_side="left",
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
            try:
                feasibility = v96.layout_feasibility(layout)
            except ValueError as error:
                feasibility = {
                    "feasible": False,
                    "reasons": [f"layout_contract_rejected:{error}"],
                    "route_metrics": {},
                    "corridor_metrics": {},
                }
            pairs.append(
                {
                    "recipe_id": inbound["recipe_id"],
                    "inbound_candidate_id": inbound["candidate_id"],
                    "outbound_candidate_id": outbound["candidate_id"],
                    "inbound": {
                        key: inbound[key]
                        for key in ("center_xy_m", "theta_deg", "gap_m", "span_along_line_m")
                    },
                    "outbound": {
                        key: outbound[key]
                        for key in ("center_xy_m", "theta_deg", "gap_m", "span_along_line_m")
                    },
                    "feasible": bool(feasibility["feasible"]),
                    "rejection_reasons": feasibility["reasons"],
                    "route_metrics": feasibility["route_metrics"],
                    "corridor_metrics": feasibility["corridor_metrics"],
                    "worst_case_min_n_sensor_frames_ge_2px": min(
                        inbound["sensing"]["min_n_sensor_frames_ge_2px"],
                        outbound["sensing"]["min_n_sensor_frames_ge_2px"],
                    ),
                    "worst_case_min_n_sensors_ge_2px": min(
                        inbound["sensing"]["min_n_sensors_ge_2px"],
                        outbound["sensing"]["min_n_sensors_ge_2px"],
                    ),
                    "worst_case_min_total_pixel_hits": min(
                        inbound["sensing"]["min_total_pixel_hits"],
                        outbound["sensing"]["min_total_pixel_hits"],
                    ),
                    "predicted_changed_values_upper_bound": {
                        "inbound_cluster": {
                            key: entry["predicted_changed_values_upper_bound"]
                            for key, entry in inbound["sensing"]["per_variant"].items()
                        },
                        "outbound_cluster": {
                            key: entry["predicted_changed_values_upper_bound"]
                            for key, entry in outbound["sensing"]["per_variant"].items()
                        },
                    },
                }
            )
    pairs.sort(
        key=lambda item: (
            not item["feasible"],
            -item["worst_case_min_n_sensor_frames_ge_2px"],
            -item["worst_case_min_n_sensors_ge_2px"],
        )
    )

    admitted_counts = {
        role: sum(1 for item in by_role[role] if item["sensing_admitted"])
        for role in by_role
    }
    reason_counts: dict[str, int] = {}
    for record in records:
        for reason in record["rejection_reasons"]:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    document = {
        "schema_version": "pact_place_v9_w2_cluster_siting_v1",
        "role": "non_authorizing_subtense_siting_sweep",
        "authorizes_gate": False,
        "authorizes_collection": False,
        "authorizes_v1b": False,
        "instrument_path": "scripts/pact_skin_resolvability.py",
        "instrument_sha256": sha256_file(ROOT / "scripts/pact_skin_resolvability.py"),
        "contract_path": "scripts/pact_place_v96_cluster_contract.py",
        "contract_sha256": sha256_file(ROOT / "scripts/pact_place_v96_cluster_contract.py"),
        "cache_root": str(cache_root.relative_to(ROOT)) if cache_root.is_relative_to(ROOT) else str(cache_root),
        "scored_on": "frozen_v9_5_trajectories_replayed_by_w1",
        "scoring_window": "physical_leg_before_grasp_and_lift_to_release",
        "decision_window_reported_for_comparability": True,
        "selection_criteria": {
            "subtense_threshold_px": SUBTENSE_THRESHOLD_PX,
            "min_distinct_sensors_per_variant": MIN_DISTINCT_SENSORS,
            "requires_link5_or_link6_response": True,
            "max_side_imbalance_ratio": MAX_SIDE_IMBALANCE,
            "min_cluster_span_m": v96.MIN_CLUSTER_SPAN_M,
            "max_cluster_gap_m": v96.MAX_CLUSTER_GAP_M,
            "lateral_offset_band_m": [MIN_LATERAL_OFFSET_M, MAX_LATERAL_OFFSET_M],
            "clearance_is_not_used_for_admission": True,
        },
        "grid": {
            "x_values_m": [float(value) for value in x_values],
            "y_values_m": [float(value) for value in y_values],
            "theta_values_deg": list(theta_values),
            "recipes": {k: v for k, v in v96.CLUSTER_RECIPES.items()},
        },
        "candidate_count": len(records),
        "admitted_counts": admitted_counts,
        "rejection_reason_counts": reason_counts,
        "shortlist": {role: items for role, items in shortlist.items()},
        "pairs": pairs[:60],
        "candidates": records,
    }
    document["document_sha256"] = sha256_payload(psr.jsonable(document))
    path = output_root / "siting.json"
    path.write_text(json.dumps(psr.jsonable(document), indent=2, sort_keys=True) + "\n")
    print(path, flush=True)
    feasible = [item for item in pairs if item["feasible"]]
    print(
        json.dumps(
            {
                "candidates": len(records),
                "admitted": admitted_counts,
                "feasible_pairs": len(feasible),
                "best_pair": feasible[0]["inbound_candidate_id"] + " + " + feasible[0]["outbound_candidate_id"]
                if feasible else None,
            },
            sort_keys=True,
        )
    )
    return 0 if feasible else 2


if __name__ == "__main__":
    raise SystemExit(main())
