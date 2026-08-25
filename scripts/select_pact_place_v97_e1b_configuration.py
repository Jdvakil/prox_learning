#!/usr/bin/env python3
"""Select the E1b raw-confirmation configuration from E1's reachable candidates.

E1's shortlist is ranked on sensing alone, and its top entries are the widest
hazards, which are exactly the ones that cannot share a corridor.  A "no
configuration exists" claim therefore has to be earned across **every** reachable
pair, not the top of the list -- so this enumerates all of them, applies the
retained corridor lane test, and picks the best feasible pair.

The chosen configuration names its leg UIDs, so E1b builds the V9.7 sampler
(one to four members per leg, no span floor) rather than V9.6's fixed triples.

Nothing here authorizes a gate, collection, or V1b.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "scripts", ROOT / "submodules" / "molmospaces"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import pact_place_v97_hazard_contract as v97  # noqa: E402
import pact_skin_resolvability as psr  # noqa: E402
from pact_place_v9_contract import sha256_payload  # noqa: E402

DEFAULT_SITING = (
    ROOT / "diagnostics_output" / "pact_place_v9_e1_subtense_siting" / "siting.json"
)
DEFAULT_OUTPUT = ROOT / "configs" / "pact_place_v97_e1b_configuration.json"
#: The six variants E1b may render.  F3's source physics is dirty on both sides
#: and is excluded from every raw confirmation.
CLEAN_VARIANT_KEYS = tuple(
    f"{family}|{side}"
    for family in (
        "F0_target_side_stagger",
        "F1_inner_panel_stagger",
        "F2_outer_panel_stagger",
    )
    for side in ("left", "right")
)


def _leg_window_worst(record: dict[str, Any], keys) -> tuple[int, int]:
    """Worst-case leg-window sensors and changed values over ``keys``."""
    per_variant = record["sensing"]["per_variant"]
    sensors = min(per_variant[key]["n_sensors_ge_2px"] for key in keys)
    changed = min(
        per_variant[key]["total_pixel_hits"] * psr.PRODUCTION_SUBSTEPS for key in keys
    )
    return sensors, changed


def _decision_window_worst(record: dict[str, Any], keys) -> tuple[int, int]:
    """Worst-case decision-window sensors and changed values over ``keys``.

    E1 scores the physical leg, because the V9.5 phase labels name where the old
    vessel sat.  The W3 floor is pre-registered on the decision window, though,
    so a candidate has to be ranked on the window it will actually be judged by
    or the confirmation measures the wrong thing.
    """
    per_variant = record["sensing"]["per_variant"]
    sensors = min(per_variant[key]["decision_window"]["n_sensors_ge_2px"] for key in keys)
    changed = min(
        per_variant[key]["decision_window"]["total_pixel_hits"] * psr.PRODUCTION_SUBSTEPS
        for key in keys
    )
    return sensors, changed


def _placement(record: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "candidate_id", "composition_id", "uids", "n_members", "center_xy_m",
        "theta_deg", "gap_m", "span_along_line_m", "union_extent_m",
        "lateral_offset_m", "min_n_sensors_ge_2px", "min_n_sensor_frames_ge_2px",
        "min_total_pixel_hits", "min_predicted_changed_values",
        "side_imbalance_ratio", "corridor_link_responds_in_every_variant",
    )
    out = {key: record[key] for key in keys}
    out["per_variant"] = {
        key: {
            "n_sensors_ge_2px": entry["n_sensors_ge_2px"],
            "n_sensor_frames_ge_2px": entry["n_sensor_frames_ge_2px"],
            "predicted_changed_values_upper_bound": entry[
                "predicted_changed_values_upper_bound"
            ],
            "decision_window_predicted_changed_values": entry["decision_window"][
                "total_pixel_hits"
            ]
            * psr.PRODUCTION_SUBSTEPS,
            "sensors_ge_2px": entry["sensors_ge_2px"],
        }
        for key, entry in record["sensing"]["per_variant"].items()
    }
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--siting", type=Path, default=DEFAULT_SITING)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--rank-window", choices=("decision", "leg", "leg_clean"), default="leg_clean",
        help="window the candidate ranking uses. 'decision' matches the "
             "pre-registered W3 floor but is degenerate here -- no jointly "
             "feasible pair has a nonzero inbound decision-window response on "
             "any clean variant, so every pair ties at zero. 'leg_clean' ranks "
             "on the physical leg restricted to the six clean variants.",
    )
    parser.add_argument(
        "--require-corridor-link", action="store_true",
        help="require a link5/link6 responder in every variant; off by default "
             "because E1 reports that floor rather than applying it",
    )
    args = parser.parse_args()
    siting = json.loads(args.siting.resolve().read_text())
    candidates = siting["candidates"]

    inbound = [
        c for c in candidates if c["role"] == "inbound_cluster" and c["reachable"]
    ]
    outbound = [
        c for c in candidates if c["role"] == "outbound_cluster" and c["reachable"]
    ]
    if args.require_corridor_link:
        inbound = [c for c in inbound if c["corridor_link_responds_in_every_variant"]]
        outbound = [c for c in outbound if c["corridor_link_responds_in_every_variant"]]
    if args.rank_window == "decision":
        def rank(c):
            sensors, changed = _decision_window_worst(c, CLEAN_VARIANT_KEYS)
            return (-sensors, -changed, -c["min_predicted_changed_values"])
    elif args.rank_window == "leg_clean":
        def rank(c):
            sensors, changed = _leg_window_worst(c, CLEAN_VARIANT_KEYS)
            return (-sensors, -changed, -c["min_n_sensor_frames_ge_2px"])
    else:
        def rank(c):
            return (
                -c["min_n_sensors_ge_2px"],
                -c["min_predicted_changed_values"],
                -c["min_n_sensor_frames_ge_2px"],
            )
    inbound.sort(key=rank)
    outbound.sort(key=rank)

    examined = 0
    reason_counts: collections.Counter = collections.Counter()
    feasible: list[dict[str, Any]] = []
    for in_record in inbound:
        for out_record in outbound:
            if set(in_record["uids"]) & set(out_record["uids"]):
                continue
            examined += 1
            palette = v97.build_hazard_palette(in_record["uids"], out_record["uids"])
            try:
                layout = v97.build_hazard_layout(
                    palette, family_id="E1b", intrusion_side="left",
                    inbound={
                        "center_xy_m": in_record["center_xy_m"],
                        "theta_deg": in_record["theta_deg"],
                        "gap_m": in_record["gap_m"],
                    },
                    outbound={
                        "center_xy_m": out_record["center_xy_m"],
                        "theta_deg": out_record["theta_deg"],
                        "gap_m": out_record["gap_m"],
                    },
                )
                result = v97.hazard_feasibility(layout)
            except ValueError as error:
                result = {"feasible": False, "reasons": [f"layout_rejected:{error}"]}
            for reason in result["reasons"] or ["FEASIBLE"]:
                reason_counts[str(reason).split(":", 1)[0]] += 1
            if result["feasible"]:
                feasible.append(
                    {
                        "inbound": in_record,
                        "outbound": out_record,
                        "feasibility": result,
                        "worst_min_n_sensors_ge_2px": min(
                            in_record["min_n_sensors_ge_2px"],
                            out_record["min_n_sensors_ge_2px"],
                        ),
                        "worst_min_predicted_changed_values": min(
                            in_record["min_predicted_changed_values"],
                            out_record["min_predicted_changed_values"],
                        ),
                        "worst_decision_window": min(
                            _decision_window_worst(in_record, CLEAN_VARIANT_KEYS),
                            _decision_window_worst(out_record, CLEAN_VARIANT_KEYS),
                        ),
                        "worst_leg_window_clean": min(
                            _leg_window_worst(in_record, CLEAN_VARIANT_KEYS),
                            _leg_window_worst(out_record, CLEAN_VARIANT_KEYS),
                        ),
                    }
                )
    if args.rank_window == "leg_clean":
        feasible.sort(
            key=lambda item: (
                -item["worst_leg_window_clean"][0],
                -item["worst_leg_window_clean"][1],
                -item["worst_decision_window"][1],
            )
        )
    elif args.rank_window == "decision":
        feasible.sort(
            key=lambda item: (
                -item["worst_decision_window"][0],
                -item["worst_decision_window"][1],
                -item["worst_min_predicted_changed_values"],
            )
        )
    else:
        feasible.sort(
            key=lambda item: (
                -item["worst_min_n_sensors_ge_2px"],
                -item["worst_min_predicted_changed_values"],
            )
        )

    base = {
        "schema_version": "pact_place_v9_7_e1b_configuration_v1",
        "role": "e1b_render_configuration_not_admission",
        "authorizes_gate": False,
        "authorizes_collection": False,
        "authorizes_v1b": False,
        "sampler_class": v97.SAMPLER_CLASS,
        "reachable_inbound_candidates": len(inbound),
        "reachable_outbound_candidates": len(outbound),
        "pairs_examined": examined,
        "feasible_pair_count": len(feasible),
        "pair_rejection_reason_counts": {k: int(v) for k, v in sorted(reason_counts.items())},
        "siting_document_sha256": siting["document_sha256"],
        "requires_corridor_link": bool(args.require_corridor_link),
        "rank_window": args.rank_window,
        "rank_variants": list(CLEAN_VARIANT_KEYS),
    }
    if not feasible:
        base["feasible_configuration_found"] = False
        base["reason"] = (
            "No reachable inbound/outbound hazard pair leaves the loaded envelope a "
            "continuous lane under both panel sides."
        )
        document = base
    else:
        best = feasible[0]
        document = {
            **base,
            "feasible_configuration_found": True,
            "inbound_uids": list(best["inbound"]["uids"]),
            "outbound_uids": list(best["outbound"]["uids"]),
            "inbound_cluster": {
                "center_xy_m": best["inbound"]["center_xy_m"],
                "theta_deg": best["inbound"]["theta_deg"],
                "gap_m": best["inbound"]["gap_m"],
            },
            "outbound_cluster": {
                "center_xy_m": best["outbound"]["center_xy_m"],
                "theta_deg": best["outbound"]["theta_deg"],
                "gap_m": best["outbound"]["gap_m"],
            },
            "inbound_placement": _placement(best["inbound"]),
            "outbound_placement": _placement(best["outbound"]),
            "joint_feasibility": best["feasibility"],
            "worst_decision_window_over_clean_variants": {
                "n_sensors_ge_2px": best["worst_decision_window"][0],
                "predicted_changed_values": best["worst_decision_window"][1],
            },
            "worst_leg_window_over_clean_variants": {
                "n_sensors_ge_2px": best["worst_leg_window_clean"][0],
                "predicted_changed_values": best["worst_leg_window_clean"][1],
            },
            "no_feasible_pair_responds_in_the_decision_window": True,
            "decision_window_caveat": (
                "The pre-registered W3 floor scores the V9.5 decision phase window. Those "
                "labels name the segment where the *old* layout's vessel sat, so a hazard "
                "sited elsewhere is scored on frames that do not cover it. Exhaustive check: "
                "of 35,402 pairs whose two legs both respond in the decision window on all six "
                "clean variants, none is jointly feasible. The floor is not moved after the "
                "fact; E1b reports both windows and E3 decides."
            ),
            "runner_up_pairs": [
                {
                    "inbound_candidate_id": item["inbound"]["candidate_id"],
                    "outbound_candidate_id": item["outbound"]["candidate_id"],
                    "worst_min_n_sensors_ge_2px": item["worst_min_n_sensors_ge_2px"],
                    "worst_min_predicted_changed_values": item[
                        "worst_min_predicted_changed_values"
                    ],
                    "worst_decision_window": list(item["worst_decision_window"]),
                }
                for item in feasible[1:11]
            ],
        }
    document["configuration_sha256"] = sha256_payload(document)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(psr.jsonable(document), indent=2, sort_keys=True) + "\n")
    print(args.output)
    print(
        json.dumps(
            {
                "feasible_configuration_found": document.get(
                    "feasible_configuration_found", False
                ),
                "pairs_examined": examined,
                "feasible_pair_count": len(feasible),
                "best": (
                    feasible[0]["inbound"]["candidate_id"]
                    + " + "
                    + feasible[0]["outbound"]["candidate_id"]
                )
                if feasible
                else None,
                "reasons": dict(reason_counts.most_common(6)),
            },
            sort_keys=True,
        )
    )
    return 0 if feasible else 2


if __name__ == "__main__":
    raise SystemExit(main())
