#!/usr/bin/env python3
"""Freeze the V9 real-object vessel and decor palette.

This deliberately keeps the V8B ``settle_prop`` implementation as the
stability oracle.  V9 changes the selection policy, not the 300-step settling
test: two stable 0.15-0.25 m vessels are selected first, then six short RGB
decor objects with a category cap of two.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
MOLMO = ROOT / "submodules" / "molmospaces"
for path in (SCRIPTS, MOLMO):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_pact_place_v8b_palette import (
    MAX_CENTER_DRIFT_M,
    MAX_ORIENTATION_CHANGE_DEG,
    POSE_QUAT_WXYZ,
    SETTLE_STEPS,
    settle_prop,
)

OUTPUT = ROOT / "diagnostics_output/pact_place_v9_v0b/palette_v9_1.json"
VESSEL_CATEGORIES = {
    "vase",
    "soapbottle",
    "pot",
    "candle",
    "spray can",
    "aerosol can",
    "travel mug",
}
DECOR_CATEGORIES = {"mug", "apple", "bowl", "plate", "potato", "can", "candle"}
EXCLUDED_CATEGORIES = {"cup", "teacup", "plastic cup", "ceramic cup", "clay cup"}

# Alternatives are measured as well as the eventual selections so the report
# contains an acceptance rate rather than only a hand-picked success list.
VESSEL_CANDIDATE_UIDS = (
    # Tall, stable candidates measured during the V9.1 redesign.  The original
    # pair topped out at 0.180/0.158 m and was rarely visible to the link-5/6
    # skin during the phase in which avoidance was allowed.
    "Soap_Bottle_30",
    "857d3f1a93b54b25bcc14aab9203346e",
    "Soap_Bottle_3",
    "e3227ecd37d44cd6be1331941d9cfa2f",
    "Soap_Bottle_1",
    "Candle_4",
    "Candle_2",
    "Soap_Bottle_11",
    "0f005c5210c241f0b7b03933cc78bd1c",
)
DECOR_CANDIDATE_UIDS = (
    # Flat/non-rolling decor for the rear scatter.  Apples and potatoes passed
    # the 300-step settle probe but later rolled without robot contact in the
    # live V9.1 smoke, contaminating clean_success with environment drift.
    "Mug_2",
    "Plate_10",
    "Plate_22",
    "663b5edc92a543668c1b602981e724a4",
    "5d13903e21044558bfb2bb7b72e76b4d",
    "Candle_1",
    "1c2ac5c48e0144e6b279501981ac7771",
    "Apple_29",
    "db4defc696c349f5abe5a99d00abc9d4",
    "Potato_1",
    "Potato_26",
    "Apple_19",
    "Apple_1",
    "Apple_2",
)


def _measure(uid: str) -> dict[str, Any]:
    try:
        record = settle_prop(uid)
    except Exception as error:  # noqa: BLE001 - rejected palette candidate
        return {
            "uid": uid,
            "attempted": True,
            "accepted": False,
            "reject_reason": f"{type(error).__name__}: {error}",
        }
    record["attempted"] = True
    return record


def _category(record: dict[str, Any]) -> str:
    return str(record.get("category") or "object").strip().lower()


def _vessel_score(record: dict[str, Any]) -> float:
    dimensions = [float(value) for value in record["collision_dimensions_m"]]
    # Height is the sensing requirement; base width is only a stability
    # tiebreaker after the 300-step settle gate has passed.
    return dimensions[2] + 0.05 * min(dimensions[0], dimensions[1])


def _select_vessels(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    eligible = [
        record
        for record in records
        if record.get("accepted")
        and _category(record) in VESSEL_CATEGORIES
        and 0.15 <= float(record["collision_dimensions_m"][2]) <= 0.25
        and max(
            float(record["collision_dimensions_m"][0]), float(record["collision_dimensions_m"][1])
        )
        <= 0.10
        and float(record["center_drift_m"]) <= 0.003
        and float(record["orientation_change_deg"]) <= 1.0
    ]
    selected: list[dict[str, Any]] = []
    categories: set[str] = set()
    for record in sorted(eligible, key=lambda item: (-_vessel_score(item), item["uid"])):
        category = _category(record)
        if category in categories:
            continue
        selected.append(record)
        categories.add(category)
        if len(selected) == 2:
            break
    if len(selected) != 2:
        raise RuntimeError("fewer than two stable, category-distinct V9 vessels")
    return selected


def _select_decor(
    records: list[dict[str, Any]], vessels: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    vessel_uids = {str(item["uid"]) for item in vessels}
    selected: list[dict[str, Any]] = []
    counts: Counter[str] = Counter(_category(v) for v in vessels)
    for record in records:
        category = _category(record)
        if (
            not record.get("accepted")
            or record["uid"] in vessel_uids
            or record["uid"] in VESSEL_CANDIDATE_UIDS
            or category not in DECOR_CATEGORIES
            or category in EXCLUDED_CATEGORIES
            or counts[category] >= 2
            or max(
                float(record["collision_dimensions_m"][0]),
                float(record["collision_dimensions_m"][1]),
            )
            > 0.20
        ):
            continue
        selected.append(record)
        counts[category] += 1
        if len(selected) == 6:
            break
    if len(selected) < 6:
        raise RuntimeError(f"only {len(selected)} stable category-capped decor objects")
    return selected


def _palette_item(record: dict[str, Any], *, slot: str, role: str) -> dict[str, Any]:
    dimensions = [float(value) for value in record["collision_dimensions_m"]]
    return {
        "slot": slot,
        "slot_class": "prop",
        "role": role,
        "uid": str(record["uid"]),
        "category": str(record["category"]),
        "dimensions_m": dimensions,
        "annotation_dimensions_m": [float(value) for value in record["dimensions_m"]],
        "half_m": [value / 2.0 for value in dimensions],
        "max_dimension_m": max(dimensions),
        "support": "shelf_standing",
        "quat_wxyz": [float(value) for value in POSE_QUAT_WXYZ],
        "body_prefix": f"pact_clutter_{slot}/",
    }


def build_palette(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    vessels = _select_vessels(records)
    decor = _select_decor(records, vessels)
    route_blocker = max(vessels, key=lambda item: float(item["collision_dimensions_m"][2]))
    perimeter_vessel = next(item for item in vessels if item is not route_blocker)
    palette = [
        _palette_item(perimeter_vessel, slot="00", role="inbound_vessel"),
        _palette_item(route_blocker, slot="01", role="outbound_vessel"),
    ]
    palette.extend(
        _palette_item(record, slot=f"{index:02d}", role="decor")
        for index, record in enumerate(decor, start=2)
    )
    categories = Counter(str(item["category"]) for item in palette)
    if len(palette) not in range(8, 13) or max(categories.values()) > 2:
        raise RuntimeError("V9 palette cardinality or category cap is invalid")
    if any(str(item["category"]).lower() in EXCLUDED_CATEGORIES for item in palette):
        raise RuntimeError("V9 palette contains a cup-like lookalike")
    return palette, vessels + decor


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    records = [_measure(uid) for uid in (*VESSEL_CANDIDATE_UIDS, *DECOR_CANDIDATE_UIDS)]
    palette, selected_records = build_palette(records)
    accepted = sum(bool(record.get("accepted")) for record in records)
    document = {
        "schema_version": "pact_place_v9_v0b_palette_v1",
        "role": "v0b_palette_selection_not_a_gate",
        "authorizes_gate": False,
        "authorizes_collection": False,
        "settle_steps": SETTLE_STEPS,
        "max_center_drift_m": MAX_CENTER_DRIFT_M,
        "max_orientation_change_deg": MAX_ORIENTATION_CHANGE_DEG,
        "selection_policy": {
            "vessel_height_m": [0.15, 0.25],
            "vessel_categories": sorted(VESSEL_CATEGORIES),
            "decor_categories": sorted(DECOR_CATEGORIES),
            "category_cap": 2,
            "n_vessels": 2,
            "n_decor": len([item for item in palette if item["role"] == "decor"]),
            "vessels_require_distinct_categories": True,
            "vessel_live_stability_center_drift_max_m": 0.003,
            "vessel_live_stability_orientation_change_max_deg": 1.0,
            "route_blocker_selection": "tallest eligible vessel",
            "second_vessel_use": "rear_perimeter_control_not_a_second_chicane_wall",
            "score": "height_m + 0.05 * min(base_x_m, base_y_m)",
            "excluded_categories": sorted(EXCLUDED_CATEGORIES),
        },
        "records": records,
        "palette": palette,
        "palette_size": len(palette),
        "selected_uids": [str(record["uid"]) for record in selected_records],
        "acceptance": {
            "n_attempted": len(records),
            "n_accepted": accepted,
            "rate": accepted / max(1, len(records)),
        },
        "category_counts": dict(Counter(str(item["category"]) for item in palette)),
        "movable_free_bodies": True,
        "settle_orientation_quat_wxyz": [float(value) for value in POSE_QUAT_WXYZ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "palette_size": len(palette),
                "acceptance": document["acceptance"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
