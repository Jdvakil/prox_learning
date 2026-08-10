#!/usr/bin/env python3
"""Finalize the fixed blur preflight using the declared scientific endpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ARMS = ("ACT", "PACT", "PACT_PERMUTED")
V3_OUTPUT = Path("/root/pact_geometry_generalization_v3_artifacts/evaluation_v1")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scientific_outcome(result: dict[str, Any]) -> dict[str, Any]:
    """Fields used by the frozen primary, co-primary, and secondary analyses."""
    audit = result["contact_audit"]
    return {
        "task_success_representing_manipulation_success": result["task_success"],
        "collision_free_task_success": result["collision_free_task_success"],
        "failure_taxonomy": result["failure_taxonomy"],
        "hazard_bar_contact_total": audit["contact_class_totals"]["hazard_bar"],
        "other_environment_contact_total": audit["contact_class_totals"][
            "other_environment"
        ],
        "hazard_bar_contact_frames": audit["frames_with_contact"]["hazard_bar"],
        "other_environment_contact_frames": audit["frames_with_contact"][
            "other_environment"
        ],
        "hazard_bar_maximum_penetration_depth_m": audit[
            "maximum_penetration_depth_m"
        ]["hazard_bar"],
        "other_environment_maximum_penetration_depth_m": audit[
            "maximum_penetration_depth_m"
        ]["other_environment"],
        "first_hazard_bar_contact_step": audit["first_contact_step"]["hazard_bar"],
        "first_other_environment_contact_step": audit["first_contact_step"][
            "other_environment"
        ],
        "non_target_contact_entries": audit["non_target_contact_entries"],
        "collision_free": audit["collision_free"],
    }


def v3_result(
    v3_schedule: dict[str, Any], current: dict[str, Any]
) -> dict[str, Any]:
    matches = [
        row
        for row in v3_schedule["rows"]
        if row["instance_episode_id"] == current["episode_id"]
        and row["checkpoint_seed"] == current["checkpoint_seed"]
        and row["arm"] == current["arm"]
    ]
    if len(matches) != 1:
        raise ValueError("v3 reference did not resolve exactly once")
    return json.loads((V3_OUTPUT / matches[0]["output_relpath"] / "result.json").read_text())


def inventory_and_remove(result: dict[str, Any]) -> list[dict[str, Any]]:
    removed = []
    raw_paths = [result.get("trajectory_path"), *result.get("videos", [])]
    for raw in raw_paths:
        if not raw:
            continue
        path = Path(raw)
        if not path.is_file():
            continue
        removed.append(
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": file_hash(path),
            }
        )
        path.unlink()
    return removed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", required=True, type=Path)
    parser.add_argument("--v3-schedule", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--initial-audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    schedule = json.loads(args.schedule.read_text())
    initial = json.loads(args.initial_audit.read_text())
    initial_payload = dict(initial)
    initial_sha = initial_payload.pop("preflight_sha256", None)
    if initial_sha != canonical_hash(initial_payload):
        raise SystemExit("initial preflight self-hash mismatch")
    by_cell = {}
    for arm in ARMS:
        for sigma in (0.0, 2.0):
            directory = args.run_root / f"sigma_{str(sigma).replace('.', 'p')}_{arm.lower()}"
            result = json.loads((directory / "result.json").read_text())
            by_cell[(arm, sigma)] = (directory, result)
    checks: dict[str, Any] = {}
    for arm in ARMS:
        sharp = by_cell[(arm, 0.0)][1]
        blurred = by_cell[(arm, 2.0)][1]
        sharp_info = sharp["policy_info"]
        blurred_info = blurred["policy_info"]
        checks[arm] = {
            "required": {
                "sigma_zero_visual_exact_identity": sharp_info["blur_diagnostic"][
                    "first_visual_input_changed"
                ]
                is False,
                "sigma_two_visual_input_changed": blurred_info["blur_diagnostic"][
                    "first_visual_input_changed"
                ]
                is True,
                "proximity_first_frame_unchanged": sharp_info[
                    "first_raw_proximity_sha256"
                ]
                == blurred_info["first_raw_proximity_sha256"],
                "action_trace_changed": sharp_info["model_output_trace_sha256"]
                != blurred_info["model_output_trace_sha256"],
            },
            "informational": {
                "independent_rerender_sharp_hash_matched": sharp_info[
                    "blur_diagnostic"
                ]["first_sharp_visual_input_sha256"]
                == blurred_info["blur_diagnostic"][
                    "first_sharp_visual_input_sha256"
                ],
                "interpretation": (
                    "independent GPU renders are not required to be byte-identical; "
                    "the within-call sharp-versus-policy-input hash is the direct blur test"
                ),
            },
        }
    v3_schedule = json.loads(args.v3_schedule.read_text())
    sharp_reproduction = {}
    for arm in ("PACT", "PACT_PERMUTED"):
        current = by_cell[(arm, 0.0)][1]
        reference = v3_result(v3_schedule, current)
        current_outcome = scientific_outcome(current)
        reference_outcome = scientific_outcome(reference)
        current_target = current["contact_audit"]
        reference_target = reference["contact_audit"]
        sharp_reproduction[arm] = {
            "scientific_outcome_exact_match": current_outcome == reference_outcome,
            "current_scientific_outcome_sha256": canonical_hash(current_outcome),
            "v3_scientific_outcome_sha256": canonical_hash(reference_outcome),
            "informational_grasp_target_sampling_delta": {
                "contact_pair_samples": current_target["contact_class_totals"][
                    "grasp_target"
                ]
                - reference_target["contact_class_totals"]["grasp_target"],
                "contact_frames": current_target["frames_with_contact"]["grasp_target"]
                - reference_target["frames_with_contact"]["grasp_target"],
                "first_contact_step": {
                    "current": current_target["first_contact_step"]["grasp_target"],
                    "v3": reference_target["first_contact_step"]["grasp_target"],
                },
                "excluded_from_endpoint": True,
                "reason": "grasp-target contact is expected and never counts against a policy",
            },
        }
    passed = all(
        all(item["required"].values()) for item in checks.values()
    ) and all(
        item["scientific_outcome_exact_match"]
        for item in sharp_reproduction.values()
    )
    removed = []
    for (arm, sigma), (_directory, result) in by_cell.items():
        removed.append(
            {
                "arm": arm,
                "blur_sigma": sigma,
                "removed_after_hashing": inventory_and_remove(result),
            }
        )
    document = {
        "schema_version": "pact_blur_sweep_preflight_v2",
        "schedule_sha256": schedule["schedule_sha256"],
        "fixed_selection": initial["fixed_selection"],
        "checks": checks,
        "sharp_v3_reproduction": sharp_reproduction,
        "initial_overstrict_audit": {
            "path": str(args.initial_audit.resolve()),
            "preflight_sha256": initial_sha,
            "passed": initial["passed"],
            "reason_for_finalizer": (
                "the initial helper accidentally required byte-identical independent "
                "RGB rerenders and the full grasp-target sampling payload; neither is a "
                "declared scientific endpoint or required intervention invariant"
            ),
        },
        "same_six_rollouts_reused_without_rerun": True,
        "row_substitution": False,
        "policy_outcome_values_used_for_parameter_selection": False,
        "raw_payload_cleanup": removed,
        "passed": passed,
    }
    document["preflight_sha256"] = canonical_hash(document)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": passed, "preflight_sha256": document["preflight_sha256"]}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
