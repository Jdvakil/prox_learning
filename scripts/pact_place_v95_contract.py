"""V9.5 raw-remediation palette and paired-side layout helpers."""

from __future__ import annotations

from typing import Any

from pact_place_v9_contract import build_layout, load_palette

V95_LAYOUT_FAMILIES = {
    "F0_target_side_stagger": ((0.570, -0.005), (0.680, 0.010)),
    "F1_inner_panel_stagger": ((0.575, 0.005), (0.690, -0.005)),
    "F2_outer_panel_stagger": ((0.565, -0.010), (0.680, 0.005)),
    "F3_aperture_side_stagger": ((0.580, 0.010), (0.690, -0.010)),
}


def load_v95_palette() -> dict[str, Any]:
    """Use a distinct, settled 88 mm vessel without mutating V9.3 artifacts."""
    document = load_palette()
    records = {str(item.get("uid")): item for item in document.get("records") or []}
    source = records.get("Soap_Bottle_11")
    if not source or not source.get("accepted"):
        raise ValueError("V9.5 requires accepted Soap_Bottle_11 siting evidence")
    dimensions = [float(value) for value in source["collision_dimensions_m"]]
    palette = [dict(item) for item in document["palette"]]
    by_slot = {str(item["slot"]): item for item in palette}
    by_slot["06"] = {
        "slot": "06",
        "slot_class": "prop",
        "role": "inbound_vessel",
        "uid": "Soap_Bottle_11",
        "category": str(source["category"]),
        "dimensions_m": dimensions,
        "annotation_dimensions_m": [float(value) for value in source["dimensions_m"]],
        "half_m": [value / 2.0 for value in dimensions],
        "max_dimension_m": max(dimensions),
        "support": "shelf_standing",
        "quat_wxyz": [2**-0.5, 2**-0.5, 0.0, 0.0],
        "body_prefix": "pact_clutter_06/",
    }
    derived = dict(document)
    derived["palette"] = [by_slot[str(item["slot"])] for item in palette]
    derived["derived_for_environment_version"] = "pact_place_corridor_v9_5_raw_remediation"
    derived["v95_inbound_vessel_change"] = {
        "from_uid": "Soap_Bottle_1",
        "to_uid": "Soap_Bottle_11",
        "reason": "paired_side_raw_proximity_remediation",
    }
    return derived


def build_v95_layout(
    palette_document: dict[str, Any], *, family_id: str, intrusion_side: str
) -> dict[str, Any]:
    inbound_xy, outbound_xy = V95_LAYOUT_FAMILIES[family_id]
    layout = build_layout(
        palette_document,
        family_id=family_id,
        intrusion_side=intrusion_side,
        inbound_center_xy=inbound_xy,
        outbound_center_xy=outbound_xy,
    )
    layout["layout_id"] = f"v9_5_raw_{intrusion_side}_{family_id}"
    layout["layout_contract_version"] = "pact_place_v9_5_raw_remediation_v1"
    return layout
