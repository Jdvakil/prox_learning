#!/usr/bin/env python3
"""Select the configuration W3 renders, from W2's recorded candidates.

W2 admits outbound placements and admits no inbound placement at all.  W3 is
still worth rendering: the W2 score is occlusion-free and W1 measured it to
over-predict a compact hazard by up to 32x, so only the raw 40-camera tensor
settles what a cluster is actually worth.  This picks the best admitted outbound
cluster and the best *jointly feasible* inbound cluster -- explicitly recording
that the inbound leg enters W3 below the floor -- and writes the configuration
the W3 driver pre-registers.

Nothing here authorizes a gate, collection, or V1b.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "scripts", ROOT / "submodules" / "molmospaces"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import pact_place_v96_cluster_contract as v96  # noqa: E402
import pact_skin_resolvability as psr  # noqa: E402
from pact_place_v9_contract import sha256_payload  # noqa: E402

DEFAULT_SITING = (
    ROOT / "diagnostics_output" / "pact_place_v9_w2_cluster_siting" / "siting.json"
)
DEFAULT_OUTPUT = (
    ROOT / "configs" / "pact_place_v96_w3_configuration.json"
)


def _placement(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": record["candidate_id"],
        "center_xy_m": record["center_xy_m"],
        "theta_deg": record["theta_deg"],
        "gap_m": record["gap_m"],
        "span_along_line_m": record["span_along_line_m"],
        "sensing_admitted": bool(record["sensing_admitted"]),
        "rejection_reasons": record["rejection_reasons"],
        "min_n_sensors_ge_2px": record["sensing"]["min_n_sensors_ge_2px"],
        "min_n_sensor_frames_ge_2px": record["sensing"]["min_n_sensor_frames_ge_2px"],
        "min_total_pixel_hits": record["sensing"]["min_total_pixel_hits"],
        "side_imbalance_ratio": record["sensing"]["side_imbalance_ratio"],
        "predicted_changed_values_upper_bound_decision_window": {
            key: entry["decision_window"]["total_pixel_hits"] * psr.PRODUCTION_SUBSTEPS
            for key, entry in record["sensing"]["per_variant"].items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--siting", type=Path, default=DEFAULT_SITING)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    siting = json.loads(args.siting.resolve().read_text())
    candidates = siting["candidates"]

    outbound = sorted(
        (c for c in candidates if c["role"] == "outbound_cluster" and c["sensing_admitted"]),
        key=lambda c: (
            -c["sensing"]["min_n_sensor_frames_ge_2px"],
            -c["sensing"]["min_n_sensors_ge_2px"],
            -c["sensing"]["min_total_pixel_hits"],
        ),
    )
    if not outbound:
        raise SystemExit("W2 admitted no outbound cluster; nothing to confirm")
    inbound_pool = sorted(
        (c for c in candidates if c["role"] == "inbound_cluster" and c["geometry_feasible"]),
        key=lambda c: (
            -c["sensing"]["min_n_sensors_ge_2px"],
            -c["sensing"]["min_n_sensor_frames_ge_2px"],
            -c["sensing"]["min_total_pixel_hits"],
        ),
    )

    # Exhaustive over every geometry-feasible pair, not just the shortlists:
    # a "no configuration exists" claim has to be earned across the whole space.
    outbound_pool = sorted(
        (c for c in candidates if c["role"] == "outbound_cluster" and c["geometry_feasible"]),
        key=lambda c: (
            -c["sensing"]["min_n_sensor_frames_ge_2px"],
            -c["sensing"]["min_n_sensors_ge_2px"],
        ),
    )
    chosen = None
    considered = []
    reason_counts: dict[str, int] = {}
    for out_record in outbound_pool:
        for in_record in inbound_pool:
            if in_record["recipe_id"] != out_record["recipe_id"]:
                continue
            palette = v96.build_cluster_palette(out_record["recipe_id"])
            try:
                layout = v96.build_cluster_layout(
                    palette,
                    family_id="W3_configuration",
                    intrusion_side="left",
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
                feasibility = v96.layout_feasibility(layout)
            except ValueError as error:
                feasibility = {"feasible": False, "reasons": [str(error)]}
            considered.append(1)
            for reason in feasibility.get("reasons", []) or ["feasible"]:
                key = str(reason).split(":", 1)[0]
                reason_counts[key] = reason_counts.get(key, 0) + 1
            if feasibility["feasible"]:
                chosen = (out_record, in_record, feasibility)
                break
        if chosen:
            break
    if chosen is None:
        document = {
            "schema_version": "pact_place_v9_6_w3_configuration_v1",
            "role": "w3_render_configuration_not_admission",
            "authorizes_gate": False,
            "authorizes_collection": False,
            "authorizes_v1b": False,
            "feasible_configuration_found": False,
            "reason": (
                "No pair of >=0.25 m clusters, one per leg, both leaves the loaded "
                "envelope a continuous lane under BOTH panel sides. The active panel "
                "removes one half of the aperture and which half flips between paired "
                "rows; a laterally sited cluster does not flip, so it closes the lane "
                "in exactly the rows whose panel forces the arm to its side."
            ),
            "geometry_feasible_outbound_candidates": len(outbound_pool),
            "geometry_feasible_inbound_candidates": len(inbound_pool),
            "sensing_admitted_outbound_candidates": len(outbound),
            "sensing_admitted_inbound_candidates": 0,
            "combinations_examined": len(considered),
            "rejection_reason_counts": reason_counts,
            "siting_document_sha256": siting["document_sha256"],
        }
        document["configuration_sha256"] = sha256_payload(document)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(psr.jsonable(document), indent=2, sort_keys=True) + "\n"
        )
        print(args.output)
        print(
            json.dumps(
                {
                    "feasible_configuration_found": False,
                    "combinations_examined": len(considered),
                    "rejection_reason_counts": reason_counts,
                },
                sort_keys=True,
            )
        )
        return 2
    out_record, in_record, feasibility = chosen

    document = {
        "schema_version": "pact_place_v9_6_w3_configuration_v1",
        "role": "w3_render_configuration_not_admission",
        "authorizes_gate": False,
        "authorizes_collection": False,
        "authorizes_v1b": False,
        "feasible_configuration_found": True,
        "combinations_examined": len(considered),
        "rejection_reason_counts": reason_counts,
        "recipe_id": out_record["recipe_id"],
        "recipe": v96.CLUSTER_RECIPES[out_record["recipe_id"]],
        "inbound_cluster": {
            "center_xy_m": in_record["center_xy_m"],
            "theta_deg": in_record["theta_deg"],
            "gap_m": in_record["gap_m"],
        },
        "outbound_cluster": {
            "center_xy_m": out_record["center_xy_m"],
            "theta_deg": out_record["theta_deg"],
            "gap_m": out_record["gap_m"],
        },
        "inbound_placement": _placement(in_record),
        "outbound_placement": _placement(out_record),
        "inbound_enters_w3_below_the_floor": not in_record["sensing_admitted"],
        "outbound_enters_w3_above_the_floor": bool(out_record["sensing_admitted"]),
        "joint_feasibility": feasibility,
        "siting_document_sha256": siting["document_sha256"],
    }
    document["configuration_sha256"] = sha256_payload(document)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(psr.jsonable(document), indent=2, sort_keys=True) + "\n")
    print(args.output)
    print(
        json.dumps(
            {
                "inbound": in_record["candidate_id"],
                "outbound": out_record["candidate_id"],
                "inbound_below_floor": document["inbound_enters_w3_below_the_floor"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
