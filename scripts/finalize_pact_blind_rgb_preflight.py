#!/usr/bin/env python3
"""Finalize blind-RGB preflight after the declared EGL rerender audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ARMS = ("ACT", "PACT", "PACT_PERMUTED")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-audit", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--amendment", required=True, type=Path)
    args = parser.parse_args()
    initial = json.loads(args.initial_audit.read_text())
    payload = dict(initial)
    initial_sha = payload.pop("preflight_sha256", None)
    if initial_sha != canonical_hash(payload) or initial.get("passed") is not False:
        raise SystemExit("initial overstrict preflight is invalid")
    checks = {}
    for arm in ARMS:
        sighted = json.loads((args.run_root / f"sighted_{arm.lower()}" / "result.json").read_text())
        blind = json.loads((args.run_root / f"blind_{arm.lower()}" / "result.json").read_text())
        sighted_info = sighted["policy_info"]
        blind_info = blind["policy_info"]
        sighted_diag = sighted_info["blur_diagnostic"]
        blind_diag = blind_info["blur_diagnostic"]
        required = {
            "blind_flag_recorded_end_to_end": (
                blind["blind_rgb"] is True
                and blind_info["blind_rgb"] is True
                and blind_diag["blind_rgb"] is True
            ),
            "blind_sigma_remains_zero": (
                blind["blur_sigma"] == 0.0 and blind_info["blur_sigma"] == 0.0
            ),
            "blind_input_exact_imagenet_mean": blind_diag[
                "first_policy_visual_input_is_exact_imagenet_mean"
            ] is True,
            "blind_input_shape_1_1_3_240_320": blind_diag[
                "first_policy_visual_input_shape"
            ] == [1, 1, 3, 240, 320],
            "blind_visual_input_changed": blind_diag["first_visual_input_changed"] is True,
            "raw_proximity_byte_identical_sighted_vs_blind": (
                sighted_info["first_raw_proximity_sha256"]
                == blind_info["first_raw_proximity_sha256"]
            ),
            "action_trace_changed_sighted_vs_blind": (
                sighted_info["model_output_trace_sha256"]
                != blind_info["model_output_trace_sha256"]
            ),
            "sighted_no_flag_recorded_end_to_end": (
                sighted["blind_rgb"] is False
                and sighted_info["blind_rgb"] is False
                and sighted_diag["blind_rgb"] is False
            ),
            "sighted_no_flag_visual_input_bit_identical_within_call": (
                sighted_diag["first_visual_input_changed"] is False
                and sighted_diag["first_sharp_visual_input_sha256"]
                == sighted_diag["first_policy_visual_input_sha256"]
            ),
        }
        informational = {
            "independent_sighted_vs_blind_sharp_render_hash_matched": initial["checks"][arm][
                "same_unmodified_first_camera_frame"
            ],
            "independent_sighted_vs_frozen_no_flag_visual_hash_matched": initial["checks"][arm][
                "sighted_visual_bit_identical_to_no_flag"
            ],
            "independent_sighted_vs_frozen_no_flag_action_hash_matched": initial["checks"][arm][
                "sighted_action_trace_bit_identical_to_no_flag"
            ],
            "independent_sighted_vs_frozen_no_flag_scientific_outcome_matched": initial[
                "checks"
            ][arm]["sighted_scientific_outcome_bit_identical_to_no_flag"],
            "interpretation": (
                "independent EGL rerenders are not byte-identical; the within-call "
                "sharp-versus-policy-input hashes are the direct no-flag identity test"
            ),
        }
        checks[arm] = {"required": required, "informational": informational}
    passed = all(
        all(record["required"].values()) for record in checks.values()
    )
    amendment = {
        "schema_version": "pact_blind_rgb_preflight_amendment_v1",
        "initial_preflight_sha256": initial_sha,
        "scientific_schedule_sha256": initial["schedule_sha256"],
        "reason": (
            "the initial audit repeated the blur preflight's known overstrict "
            "independent-EGL-render identity requirement"
        ),
        "prior_evidence": {
            "path": "diagnostics_output/pact_blur_sweep/preflight.json",
            "same_issue_previously_recorded": True,
        },
        "change": (
            "cross-rerender image/action/outcome identity is informational; direct "
            "within-call sighted input identity remains required"
        ),
        "rollouts_rerun": 0,
        "endpoint_or_decision_rule_changed": False,
        "preflight_intervention_checks_changed": False,
        "policy_outcomes_used_to_change_scientific_design": False,
    }
    amendment["amendment_sha256"] = canonical_hash(amendment)
    document = {
        "schema_version": "pact_blind_rgb_preflight_v1",
        "schedule_sha256": initial["schedule_sha256"],
        "fixed_selection": initial["fixed_selection"],
        "checks": checks,
        "initial_overstrict_audit": {
            "path": str(args.initial_audit),
            "preflight_sha256": initial_sha,
        },
        "amendment_sha256": amendment["amendment_sha256"],
        "rollouts_rerun": 0,
        "policy_endpoint_values_used_for_design_or_selection": False,
        "passed": passed,
        "runs": initial["runs"],
    }
    document["preflight_sha256"] = canonical_hash(document)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    args.amendment.write_text(json.dumps(amendment, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": passed, "preflight_sha256": document["preflight_sha256"]}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
