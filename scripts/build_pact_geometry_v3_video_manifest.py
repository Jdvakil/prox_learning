#!/usr/bin/env python3
"""Freeze the geometry-v3 qualitative selections and determinism gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEDULE_PATH = ROOT / "diagnostics_output/pact_geometry_generalization_v3/schedule.json"
SCIENTIFIC_MANIFEST_PATH = ROOT / "configs/pact_geometry_generalization_v3.json"
ANALYSIS_PATH = ROOT / "diagnostics_output/pact_geometry_generalization_v3/analysis.json"
FINAL_DECISION_PATH = ROOT / "diagnostics_output/pact_geometry_generalization_v3/final_decision.json"
REPORT_PATH = ROOT / "docs/PACT_GEOMETRY_GENERALIZATION_V3.md"
RESULT_ROOT = Path("/root/pact_geometry_generalization_v3_artifacts/evaluation_v1")
OUTPUT_PATH = ROOT / "diagnostics_output/pact_geometry_generalization_v3/qualitative_video_manifest.json"
BUILDER_PATH = ROOT / "scripts/build_pact_geometry_v3_video_manifest.py"
RUNNER_PATH = ROOT / "scripts/run_pact_geometry_v3_videos.py"
QUALITATIVE_EVALUATOR_PATH = ROOT / "submodules/act/eval_pact_qualitative_row.py"
GEOMETRY_EVALUATOR_PATH = ROOT / "submodules/act/eval_pact_geometry_v3_qualitative_row.py"

CONDITION_SPECS = {
    "C2": {
        "pair_id": "pairA_c2",
        "label": "tighter: aperture 0.70 m, inner face 0.070 m",
        "qualifying_count": 14,
        "fixed_rank1": {
            "episode_id": "41c6e3f2f5b575e2c7ea27587fa2c8441201f39be51a199882036636937171d0",
            "checkpoint_seed": 3101,
            "PACT": 269,
            "PACT_PERMUTED": 268,
        },
    },
    "Z_093": {
        "pair_id": "pairB_z093",
        "label": "panel 4 cm higher: z 0.89 to 0.93 m",
        "qualifying_count": 18,
        "fixed_rank1": {
            "episode_id": "f024c124bc3c0f5f51711d1a7b4afb1b4873feab76a9534eee5f83463ec1f218",
            "checkpoint_seed": 3101,
            "PACT": 607,
            "PACT_PERMUTED": 606,
        },
    },
}

GATE = {
    "declared_before_render": True,
    "task_success": {"comparison": "exact"},
    "manipulation_success": {
        "comparison": "exact",
        "represented_by": "task_success",
    },
    "first_hazard_bar_contact_step": {"comparison": "exact"},
    "first_grasp_target_contact_step": {
        "comparison": "absolute_step_delta_lte",
        "tolerance_steps": 2,
    },
    "contact_pair_sample_counts": {
        "comparison": "informational_only",
        "record_delta": True,
    },
    "on_pair_breach": (
        "drop both clips without retry and advance mechanically to the next "
        "predeclared rank in the same condition"
    ),
    "fallback_ranks": [2, 3],
    "if_all_three_fail": "ship the other condition alone and disclose the drop",
}


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


def validate_self_hash(document: dict[str, Any], key: str, label: str) -> str:
    payload = dict(document)
    observed = payload.pop(key, None)
    expected = canonical_hash(payload)
    if observed != expected:
        raise RuntimeError(f"{label} self-hash mismatch: {observed} != {expected}")
    return str(observed)


def load_result(row: dict[str, Any], result_root: Path) -> tuple[dict[str, Any], Path]:
    path = result_root / row["output_relpath"] / "result.json"
    result = json.loads(path.read_text())
    expected = {
        "status": "complete",
        "arm": row["arm"],
        "episode_id": row["instance_episode_id"],
        "checkpoint_seed": row["checkpoint_seed"],
        "rollout_id": row["rollout_id"],
        "schedule_row_sha256": row["schedule_row_sha256"],
        "checkpoint_sha256": row["checkpoint_sha256"],
    }
    observed = {key: result.get(key) for key in expected}
    if observed != expected:
        raise RuntimeError(
            f"row {row['schedule_index']} result identity mismatch: {observed} != {expected}"
        )
    return result, path


def frozen_arm_record(
    row: dict[str, Any], result: dict[str, Any], result_path: Path
) -> dict[str, Any]:
    audit = result["contact_audit"]
    return {
        "arm": row["arm"],
        "schedule_index": int(row["schedule_index"]),
        "rollout_id": row["rollout_id"],
        "schedule_row_sha256": row["schedule_row_sha256"],
        "source_directory": Path(row["output_relpath"]).name,
        "original_result_path": str(result_path.resolve()),
        "original_result_sha256": file_hash(result_path),
        "checkpoint_path": row["checkpoint_path"],
        "checkpoint_sha256": row["checkpoint_sha256"],
        "dataset_stats_path": row["dataset_stats_path"],
        "dataset_stats_sha256": row["dataset_stats_sha256"],
        "surface_encoder_path": row["surface_encoder_path"],
        "surface_encoder_sha256": row["surface_encoder_sha256"],
        "token_plan_manifest_path": row.get("token_plan_manifest_path"),
        "token_plan_sha256": row.get("token_plan_sha256"),
        "token_plan_row": row.get("token_plan_row"),
        "outcome": {
            "task_success": bool(result["task_success"]),
            "manipulation_success": bool(result["task_success"]),
            "hazard_contact": bool(audit["frames_with_contact"]["hazard_bar"]),
            "hazard_contact_frames": int(audit["frames_with_contact"]["hazard_bar"]),
            "grasp_target_contact_frames": int(
                audit["frames_with_contact"]["grasp_target"]
            ),
            "first_hazard_bar_contact_step": audit["first_contact_step"]["hazard_bar"],
            "first_grasp_target_contact_step": audit["first_contact_step"]["grasp_target"],
            "hazard_contact_pair_samples": int(
                audit["contact_class_totals"]["hazard_bar"]
            ),
            "maximum_hazard_penetration_depth_m": float(
                audit["maximum_penetration_depth_m"]["hazard_bar"]
            ),
        },
    }


def ranked_candidates(
    schedule: dict[str, Any], result_root: Path
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, int], dict[str, tuple[dict[str, Any], dict[str, Any], Path]]] = {}
    for row in schedule["rows"]:
        if row["condition_id"] not in CONDITION_SPECS:
            continue
        result, result_path = load_result(row, result_root)
        key = (
            row["condition_id"],
            row["instance_episode_id"],
            int(row["checkpoint_seed"]),
        )
        grouped.setdefault(key, {})[row["arm"]] = (row, result, result_path)

    ranked: dict[str, list[dict[str, Any]]] = {}
    for condition_id, spec in CONDITION_SPECS.items():
        candidates = []
        for (condition, episode_id, seed), arms in grouped.items():
            if condition != condition_id or set(arms) != {"PACT", "PACT_PERMUTED"}:
                continue
            pact = arms["PACT"][1]["contact_audit"]["frames_with_contact"]["hazard_bar"]
            perm = arms["PACT_PERMUTED"][1]["contact_audit"]["frames_with_contact"]["hazard_bar"]
            if int(pact) != 0 or int(perm) <= 500:
                continue
            pact_row = arms["PACT"][0]
            perm_row = arms["PACT_PERMUTED"][0]
            if (
                pact_row["intrusion_side"] != perm_row["intrusion_side"]
                or pact_row["instance_row_sha256"] != perm_row["instance_row_sha256"]
            ):
                raise RuntimeError("matched candidate geometry/side mismatch")
            candidates.append(
                {
                    "pair_id": spec["pair_id"],
                    "condition_id": condition_id,
                    "condition_label": spec["label"],
                    "episode_id": episode_id,
                    "checkpoint_seed": seed,
                    "intrusion_side": pact_row["intrusion_side"],
                    "realized_geometry": pact_row["realized_geometry"],
                    "permuted_hazard_frames_sort_key": int(perm),
                    "arms": {
                        arm: frozen_arm_record(*arms[arm])
                        for arm in ("PACT", "PACT_PERMUTED")
                    },
                }
            )
        candidates.sort(
            key=lambda item: (
                -item["permuted_hazard_frames_sort_key"],
                item["episode_id"],
                item["checkpoint_seed"],
            )
        )
        if len(candidates) != int(spec["qualifying_count"]):
            raise RuntimeError(
                f"{condition_id} qualifying count {len(candidates)} != {spec['qualifying_count']}"
            )
        for rank, candidate in enumerate(candidates, 1):
            candidate["selection_rank"] = rank
        expected = spec["fixed_rank1"]
        first = candidates[0]
        observed = {
            "episode_id": first["episode_id"],
            "checkpoint_seed": first["checkpoint_seed"],
            "PACT": first["arms"]["PACT"]["schedule_index"],
            "PACT_PERMUTED": first["arms"]["PACT_PERMUTED"]["schedule_index"],
        }
        if observed != expected:
            raise RuntimeError(f"{condition_id} fixed rank-1 mismatch: {observed} != {expected}")
        ranked[condition_id] = candidates
    return ranked


def build_document(result_root: Path = RESULT_ROOT) -> dict[str, Any]:
    schedule = json.loads(SCHEDULE_PATH.read_text())
    schedule_hash = validate_self_hash(schedule, "schedule_sha256", "v3 schedule")
    scientific_manifest = json.loads(SCIENTIFIC_MANIFEST_PATH.read_text())
    manifest_hash = validate_self_hash(
        scientific_manifest, "manifest_sha256", "v3 scientific manifest"
    )
    final = json.loads(FINAL_DECISION_PATH.read_text())
    if final.get("decision") != "GEOMETRY_GENERALIZES":
        raise RuntimeError("v3 final decision is not GEOMETRY_GENERALIZES")
    ranked = ranked_candidates(schedule, result_root)
    selections = []
    for condition_id in ("C2", "Z_093"):
        for candidate in ranked[condition_id][:3]:
            candidate = dict(candidate)
            candidate["selection_rule"] = (
                "within condition, among matched instance-seeds with PACT hazard frames "
                "equal to 0 and PACT_PERMUTED hazard frames greater than 500, order by "
                "descending PACT_PERMUTED hazard frames; break ties by episode ID then seed"
            )
            selections.append(candidate)

    document: dict[str, Any] = {
        "schema_version": "pact_geometry_v3_qualitative_video_manifest_v1",
        "status": "selection_and_gate_frozen_pre_render",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "presentation_release_only": True,
        "scientific_record_reopened": False,
        "sources": {
            "schedule": {"path": str(SCHEDULE_PATH), "schedule_sha256": schedule_hash},
            "scientific_manifest": {
                "path": str(SCIENTIFIC_MANIFEST_PATH),
                "manifest_sha256": manifest_hash,
            },
            "analysis": {"path": str(ANALYSIS_PATH), "sha256": file_hash(ANALYSIS_PATH)},
            "final_decision": {
                "path": str(FINAL_DECISION_PATH),
                "sha256": file_hash(FINAL_DECISION_PATH),
                "decision": "GEOMETRY_GENERALIZES",
            },
            "report": {"path": str(REPORT_PATH), "sha256": file_hash(REPORT_PATH)},
            "result_root": str(result_root.resolve()),
            "runtime_code": {
                "builder": {"path": str(BUILDER_PATH), "sha256": file_hash(BUILDER_PATH)},
                "runner": {"path": str(RUNNER_PATH), "sha256": file_hash(RUNNER_PATH)},
                "qualitative_evaluator": {
                    "path": str(QUALITATIVE_EVALUATOR_PATH),
                    "sha256": file_hash(QUALITATIVE_EVALUATOR_PATH),
                },
                "geometry_evaluator": {
                    "path": str(GEOMETRY_EVALUATOR_PATH),
                    "sha256": file_hash(GEOMETRY_EVALUATOR_PATH),
                },
            },
        },
        "selection_contract": {
            "conditions": ["C2", "Z_093"],
            "arms": ["PACT", "PACT_PERMUTED"],
            "qualifying_counts": {
                condition: len(ranked[condition]) for condition in ("C2", "Z_093")
            },
            "frozen_ranks_per_condition": 3,
            "initial_rank": 1,
            "fallback_order": [2, 3],
            "no_substitution_after_footage_review": True,
        },
        "determinism_gate": GATE,
        "render_contract": {
            "render_only_third_person_camera": True,
            "camera_source": "submodules/act/eval_pact_qualitative_row.py",
            "camera_reference_body": "robot_0/fr3_link0",
            "camera_offset_m": [-1.05, -0.55, 1.3],
            "lookat_offset_m": [0.55, 0.0, 0.45],
            "fov_degrees": 58.0,
            "resolution_width_height": [624, 352],
            "raw_fps": 1000.0 / 66.0,
            "playback_speed_factor": 3.0,
            "max_control_steps": 900,
            "side_by_side_order": ["PACT", "PACT_PERMUTED"],
            "synchronized_on": "control_step",
            "overlay_fields": [
                "policy arm and checkpoint seed",
                "condition ID",
                "episode ID first 12 characters",
                "task success yes/no",
                "any hazard contact yes/no",
                "hazard-contact frames running cumulative",
                "maximum hazard penetration",
                "constant playback speed factor",
            ],
        },
        "geometry_figure_contract": {
            "conditions": ["C0", "C2", "Z_093"],
            "camera": "same render-only third-person camera",
            "fixed_arm_pose": "initial control step 0",
            "annotations": {
                "C0": "baseline: aperture 0.85 m; inner face 0.100 m; panel z 0.89 m",
                "C2": "aperture 0.85 to 0.70 m; inner face 0.100 to 0.070 m",
                "Z_093": "panel z 0.89 to 0.93 m",
            },
        },
        "ranked_selections": selections,
    }
    document["qualitative_video_manifest_sha256"] = canonical_hash(document)
    return document


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to replace frozen manifest: {args.output}")
    document = build_document(args.result_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "sha256": document["qualitative_video_manifest_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
