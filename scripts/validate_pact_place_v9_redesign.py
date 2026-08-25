#!/usr/bin/env python3
"""Validate the V9.2 active-panel layout and summarize non-gating smoke evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from pact_place_v9_contract import (
    LAYOUT_FAMILIES,
    PALETTE_PATH,
    SITING_PATH,
    build_layout,
    load_palette,
    panel_corridor_metrics,
    route_blocker_metrics,
    sha256_payload,
    validate_layout,
)

DEFAULT_SMOKE_ROOT = ROOT / "diagnostics_output/pact_place_v9_panel_smoke"
DEFAULT_OUTPUT = ROOT / "diagnostics_output/pact_place_v9_panel_redesign/validation.json"
OLD_REVIEW = ROOT / "diagnostics_output/pact_place_v9_v1b/review_manifest.json"


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def _variant(result: dict[str, Any]) -> str | None:
    layout = (result.get("scene_params") or {}).get("pact_clutter_layout") or {}
    family = layout.get("layout_family_id")
    side = layout.get("intrusion_side")
    if family is None or side not in {"left", "right"}:
        return None
    return f"{family}__{side}"


def _valid_smoke(result: dict[str, Any]) -> bool:
    audit = result.get("contact_audit") or {}
    bows = result.get("bow_diagnostics") or {}
    panel_bow = bows.get("outbound") or {}
    vessel_bow = bows.get("outbound_vessel") or {}
    scene = result.get("scene_params") or {}
    return bool(
        result.get("status") == "complete"
        and result.get("task_success")
        and result.get("clean_success")
        and result.get("outbound_deflected")
        and float(panel_bow.get("accepted_bow_m") or 0.0) > 0.0
        and float(vessel_bow.get("accepted_bow_m") or 0.0) > 0.0
        and scene.get("pact_v9_legacy_panel_active") is True
        and scene.get("protrusion_present") is True
        and audit.get("collision_free") is True
        and not list(result.get("clutter_stability_events") or [])
    )


def _smoke_summary(path: Path, result: dict[str, Any]) -> dict[str, Any]:
    audit = result.get("contact_audit") or {}
    bow = result.get("bow_diagnostics") or {}
    return {
        "result_path": _relative(path),
        "episode_id": result.get("episode_id"),
        "role_index": result.get("role_index"),
        "task_success": bool(result.get("task_success")),
        "clean_success": bool(result.get("clean_success")),
        "collision_free": bool(audit.get("collision_free")),
        "contact_class_totals": audit.get("contact_class_totals") or {},
        "clutter_stability_events": list(result.get("clutter_stability_events") or []),
        "inbound_deflected": bool(result.get("inbound_deflected")),
        "outbound_deflected": bool(result.get("outbound_deflected")),
        "bow_diagnostics": bow,
        "phase_gated_detected_hazards": list(result.get("detected_hazards") or []),
        "episode_steps": int(result.get("episode_steps") or 0),
    }


def _select_smokes(smoke_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    candidates: dict[str, list[tuple[Path, dict[str, Any]]]] = {
        f"{family_id}__{side}": []
        for family_id in LAYOUT_FAMILIES
        for side in ("left", "right")
    }
    unassigned: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(smoke_root.glob("expert_screen_rows/*/result.json")):
        result = json.loads(path.read_text())
        variant_id = _variant(result)
        if variant_id in candidates:
            candidates[variant_id].append((path, result))
        else:
            unassigned.append((path, result))
    selected: dict[str, Any] = {}
    outcome_counts: dict[str, Any] = {}
    for variant_id, records in candidates.items():
        passing = [(path, result) for path, result in records if _valid_smoke(result)]
        outcome_counts[variant_id] = {
            "total_development_rows": len(records),
            "clean_route_smokes": len(passing),
            "infrastructure_failures": sum(
                result.get("status") == "infrastructure_failure" for _path, result in records
            ),
            "sampling_failures": sum(
                result.get("status") == "sampling_failure" for _path, result in records
            ),
            "completed_nonclean_rows": sum(
                result.get("status") == "complete" and not _valid_smoke(result)
                for _path, result in records
            ),
        }
        if not passing:
            raise ValueError(f"no clean route-forcing smoke for {variant_id}")
        path, result = passing[-1]
        selected[variant_id] = _smoke_summary(path, result)
    outcome_counts["unassigned_before_scene_metadata"] = {
        "total_development_rows": len(unassigned),
        "infrastructure_failures": sum(
            result.get("status") == "infrastructure_failure" for _path, result in unassigned
        ),
        "sampling_failures": sum(
            result.get("status") == "sampling_failure" for _path, result in unassigned
        ),
    }
    return selected, outcome_counts


def _old_result_summary() -> dict[str, Any]:
    siting = json.loads(SITING_PATH.read_text())
    review = json.loads(OLD_REVIEW.read_text())
    return {
        "siting_path": _relative(SITING_PATH),
        "chosen_pair": siting.get("chosen_pair"),
        "n_admitted_by_role": siting.get("n_admitted_by_role"),
        "n_admissible_pairs": len(siting.get("admissible_pairs") or []),
        "review_manifest_path": _relative(OLD_REVIEW),
        "review_attempts": int(review.get("n_attempts") or 0),
        "review_clean_successes": int(review.get("n_successes_captured") or 0),
        "review_failures": int(review.get("n_failures_captured") or 0),
        "invalidated_by_redesign": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-root", type=Path, default=DEFAULT_SMOKE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    palette_document = load_palette(PALETTE_PATH)
    layouts: dict[str, Any] = {}
    for family_id in LAYOUT_FAMILIES:
        layouts[family_id] = {}
        left_objects = None
        for side in ("left", "right"):
            layout = build_layout(
                palette_document, family_id=family_id, intrusion_side=side
            )
            validate_layout(layout)
            centers = [item["center_m"] for item in layout["objects"]]
            metrics = route_blocker_metrics(layout)
            corridor = panel_corridor_metrics(layout)
            if left_objects is None:
                left_objects = layout["objects"]
            elif layout["objects"] != left_objects:
                raise ValueError(f"{family_id} visible clutter leaks panel side")
            layouts[family_id][side] = {
                "layout_id": layout["layout_id"],
                "route_blocker_center_xy_m": layout["route_blocker_center_xy_m"],
                "expected_bow_direction": layout["expected_bow_direction"],
                "object_count": len(layout["objects"]),
                "x_span_m": max(point[0] for point in centers)
                - min(point[0] for point in centers),
                "y_span_m": max(point[1] for point in centers)
                - min(point[1] for point in centers),
                "distinct_x_count": len({round(point[0], 3) for point in centers}),
                "nominal_route_metrics": metrics,
                "panel_corridor_metrics": corridor,
            }

    smokes, smoke_outcome_counts = _select_smokes(args.smoke_root)
    phase_gate_triggered = any(record["phase_gated_detected_hazards"] for record in smokes.values())
    phase_gate_families = [
        family_id for family_id, record in smokes.items() if record["phase_gated_detected_hazards"]
    ]
    document: dict[str, Any] = {
        "schema_version": "pact_place_v9_2_panel_redesign_validation_v1",
        "status": "panel_geometry_and_expert_smoke_pass_proximity_excitation_pending",
        "role": "redesign_validation_not_a_gate",
        "authorizes_gate": False,
        "authorizes_collection": False,
        "palette_path": _relative(PALETTE_PATH),
        "palette_sha256": sha256_payload(palette_document),
        "layout_requirements": {
            "two_dimensional_scatter": True,
            "minimum_x_span_m": 0.40,
            "minimum_distinct_x_count": 6,
            "direct_loaded_route_must_be_blocked": True,
            "minimum_forced_bow_m": 0.04,
            "detour_must_fit_inside_aperture": True,
            "legacy_panel_active": True,
            "exactly_one_active_panel": True,
            "one_movable_route_blocker": True,
            "visible_clutter_panel_side_invariant": True,
            "minimum_combined_corridor_margin_m": 0.019,
        },
        "layout_families": layouts,
        "smoke_evidence": {
            "not_a_clean_rate_estimate": True,
            "fixed_layout_side_variant_coverage": list(smokes),
            "selected_clean_result_by_variant": smokes,
            "development_outcome_counts_by_variant": smoke_outcome_counts,
        },
        "proximity_excitation": {
            "phase_gated_detector_triggered_in_selected_smokes": phase_gate_triggered,
            "phase_gated_detector_triggered_families": phase_gate_families,
            "phase_gated_detector_is_only_a_sampled_surface_proxy": True,
            "raw_pact_40x8x8_collected_by_screen_harness": False,
            "passed": False,
            "required_next_measurement": (
                "fresh replay/collection must show route-blocker-caused, non-degenerate "
                "proximity during traversal before V1b or Phase 0"
            ),
        },
        "invalidated_previous_result": _old_result_summary(),
        "conclusion": (
            "Every family/side variant keeps one physical panel active, preserves panel-side "
            "invariance in visible clutter, blocks the nominal loaded route, admits one "
            "panel-selected detour, and has a collision-free clean expert smoke. This proves "
            "geometric challenge and feasibility, not a PACT advantage."
        ),
    }
    document["validation_sha256"] = sha256_payload(document)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": document["status"],
                "families": list(layouts),
                "phase_gate_triggered": phase_gate_triggered,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
