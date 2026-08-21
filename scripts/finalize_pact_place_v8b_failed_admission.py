#!/usr/bin/env python3
"""Write the mandatory V8B failed-admission report after exhaustive Pass 3."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from pact_place_corridor_contract import sha256_payload  # noqa: E402
from run_pact_place_expert_screen import write_json_atomic  # noqa: E402

SWEEP_DIR = ROOT / "diagnostics_output/pact_place_clutter_sweep_v8b"
OUTPUT = SWEEP_DIR / "analysis.json"


def main() -> int:
    palette = json.loads((SWEEP_DIR / "palette_stability.json").read_text())
    pass1 = json.loads((SWEEP_DIR / "analysis_pass1.json").read_text())
    pass2 = json.loads((SWEEP_DIR / "analysis_pass2.json").read_text())
    mount_b4_path = ROOT / "diagnostics_output/pact_place_corridor_v8b_mount_scoring_check/scoring_check.json"
    mount_b4 = json.loads(mount_b4_path.read_text())
    quota_cells = {
        f"{family}/{side}": (">=2" if family == "F5_overhead_elbow" and side == "right" else 0)
        for family in (
            "F1_near_forearm_left",
            "F2_near_forearm_right",
            "F3_front_stagger",
            "F4_rear_stagger",
            "F5_overhead_elbow",
            "F6_target_occluding",
        )
        for side in ("left", "right")
    }
    report = {
        "schema_version": "pact_place_clutter_sweep_v8b_failed_admission_v1",
        "role": "failed_admission_report_not_a_gate",
        "authorizes_gate": False,
        "authorizes_collection": False,
        "mandatory_stop_before_B5b": True,
        "stop_reason": "Pass 3 cannot fill 11 of 12 family/side link-primary quota cells; five behavioral admission gates fail on the six real episodes",
        "palette": {
            "size": palette["palette_size"],
            "slot_class_counts": palette["slot_class_counts"],
            "category_counts": palette["palette_category_counts"],
            "stability_thresholds": {
                "center_drift_m": palette["max_center_drift_m"],
                "orientation_change_deg": palette["max_orientation_change_deg"],
            },
            "records": palette["records"],
        },
        "support_and_dynamics": {
            "prop_slots": "free bodies; settled and outcome-bearing when displaced/toppled",
            "mount_slots": "jointless mocap bodies; kinematically posed and immovable",
            "asymmetry_is_deliberate": True,
            "mount_attachment_rule": "within 20 mm of a side wall or ceiling",
        },
        "pass1": {
            "distance_instrument": pass1["distance_instrument"],
            "first_approximation_only": True,
            "n_candidates": pass1["n_candidates"],
            "n_admitted_link_primary": pass1["n_admitted_link_primary"],
            "chosen_n": pass1["chosen_n"],
            "family_side_quotas": pass1["family_side_quotas"],
        },
        "pass2": {
            "distance_instrument": pass2["distance_instrument"],
            "mesh_pair_fallback": "when mj_geomDistance scalar is spuriously zero but fromto endpoints differ, use norm(fromto[3:]-fromto[:3])",
            "n_realized_episodes": pass2["aggregate"]["n_episodes"],
            "aggregate": pass2["aggregate"],
            "admission_gates": pass2["admission_gates"],
            "all_admission_gates_pass": False,
        },
        "pass3": {
            "distance_instrument": "same exact mj_geomDistance plus fromto mesh fallback as Pass 2",
            "n_rescored": pass1["n_candidates"],
            "selection_possible": False,
            "link_primary_eligible_by_quota_cell": quota_cells,
            "missing_quota_cell_count": 11,
            "cause": "real geometry puts the carried cup closer than an arm link for nearly all non-colliding candidates; representative F1 candidates also penetrate the carried cup or hand",
            "representative_exact_scores": {
                "F1_candidate_0": {
                    "reject": "would_contact_carried_cup",
                    "cup_is_closest_body": True,
                    "min_link_clearance_m": 0.05484356313723256,
                    "min_cup_clearance_m": -0.00575415391783191,
                    "frames_link_clearance_lt_10cm": 29,
                    "n_distinct_links_exposed": 1,
                    "visibility_at_min_link_clearance": False,
                },
                "F5_candidate_768": {
                    "reject": None,
                    "cup_is_closest_body": True,
                    "min_link_clearance_m": 0.14049741924343823,
                    "min_cup_clearance_m": 0.04452385999394282,
                    "frames_link_clearance_lt_10cm": 0,
                    "n_distinct_links_exposed": 0,
                    "visibility_at_min_link_clearance": False,
                },
            },
        },
        "instrument_findings": [
            "B0 AABB approximation admitted 877 link-primary candidates, but Pass 2 realized cup_is_closest_body remained 5/6.",
            "Raw mj_geomDistance returns scalar zero for separated mesh/primitive pairs while populating distinct closest points; the documented mesh fallback is required.",
            "After that correction, the six real episodes have only 34 frames below 10 cm versus the required 1852 and zero visibility-at-min episodes.",
        ],
        "mount_scoring_check": {
            "path": str(mount_b4_path.relative_to(ROOT)),
            "passed": mount_b4["passed"],
            "clean_success": mount_b4["clean_success"],
            "clutter_contact_count": mount_b4["contact_audit"]["contact_class_totals"]["clutter"],
        },
        "b5b": {
            "executed": False,
            "reason": "numeric realized admission failed and Pass 3 could not select 24 quota-complete layouts",
        },
    }
    report["analysis_sha256"] = sha256_payload(report)
    write_json_atomic(OUTPUT, report)
    print(json.dumps({"output": str(OUTPUT), "mandatory_stop_before_B5b": True, "mount_b4_passed": mount_b4["passed"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
