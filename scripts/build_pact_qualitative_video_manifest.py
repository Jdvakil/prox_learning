#!/usr/bin/env python3
"""Mechanically freeze the five paired PACT qualitative-video selections."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEDULE = ROOT / "diagnostics_output/pact_contact_endpoint/schedule.json"
DEFAULT_TAIL = ROOT / "diagnostics_output/pact_contact_endpoint/tail_characterization.json"
DEFAULT_OUTPUT = (
    ROOT
    / "diagnostics_output/pact_contact_endpoint/qualitative_video_manifest.json"
)
V2_OUTPUT = (
    ROOT
    / "diagnostics_output/pact_contact_endpoint/qualitative_clips_v2_manifest.json"
)
V3_OUTPUT = (
    ROOT
    / "diagnostics_output/pact_contact_endpoint/qualitative_clips_v3_manifest.json"
)
V3_FALLBACK_OUTPUT = (
    ROOT
    / "diagnostics_output/pact_contact_endpoint/qualitative_clip3_fallback_manifest.json"
)
V3_FALLBACK2_OUTPUT = (
    ROOT
    / "diagnostics_output/pact_contact_endpoint/qualitative_clip3_fallback_rank2_manifest.json"
)
ACT_SUCCESS_SUPPLEMENT_OUTPUT = (
    ROOT
    / "diagnostics_output/pact_contact_endpoint/qualitative_act_success_supplement.json"
)
RECORDED_ACT_SUCCESS_OUTPUT = (
    ROOT
    / "diagnostics_output/pact_contact_endpoint/qualitative_recorded_act_success_supplement.json"
)
FRONTEND_SCREEN_SCHEDULE = ROOT / "diagnostics_output/pact_frontend_screen/schedule.json"
FRONTEND_SCREEN_RESULT_ROOT = Path(
    "/root/pact_frontend_screen_artifacts/evaluation_621764f8/rows"
)
DEFAULT_RESULT_ROOT = Path("/root/pact_contact_endpoint_artifacts/evaluation_v1")
DEFAULT_VIDEO_ROOT = Path("/root/pact_contact_endpoint_artifacts/qualitative_videos")
ARMS = ("ACT", "PACT")
THRESHOLD = 500

V2_FIXED_CLIPS = (
    {
        "clip_id": "clip1_54a6272f66ca_pact_success",
        "pair_id": "instance_a",
        "arm": "PACT",
        "episode_id": "54a6272f66ca3c7bb57dc603550a4c29d35605e3fe65aaef627a9a06bad00b6f",
        "checkpoint_seed": 3101,
        "intrusion_side": "right",
        "source_directory": "0310_6b7f11de957a9396_pact_s3101",
        "hazard_frames": 0,
        "task_success": True,
        "selection_role": "discordant_safety_pact_success",
    },
    {
        "clip_id": "clip2_54a6272f66ca_act_failure",
        "pair_id": "instance_a",
        "arm": "ACT",
        "episode_id": "54a6272f66ca3c7bb57dc603550a4c29d35605e3fe65aaef627a9a06bad00b6f",
        "checkpoint_seed": 3101,
        "intrusion_side": "right",
        "source_directory": "0308_66a8fc62c0f3ec7b_act_s3101",
        "hazard_frames": 29022,
        "task_success": False,
        "selection_role": "discordant_safety_act_failure",
    },
    {
        "clip_id": "clip3_e99dc657bfa7_act_success",
        "pair_id": "instance_b",
        "arm": "ACT",
        "episode_id": "e99dc657bfa703eac0d75566c733613ca0ffede3a4bbc394a35c350c753a4391",
        "checkpoint_seed": 3103,
        "intrusion_side": "left",
        "source_directory": "0989_2a7bc05291d96619_act_s3103",
        "hazard_frames": 19757,
        "task_success": True,
        "selection_role": "task_success_with_contact_act",
    },
    {
        "clip_id": "clip4_e99dc657bfa7_pact_failure",
        "pair_id": "instance_b",
        "arm": "PACT",
        "episode_id": "e99dc657bfa703eac0d75566c733613ca0ffede3a4bbc394a35c350c753a4391",
        "checkpoint_seed": 3103,
        "intrusion_side": "left",
        "source_directory": "0991_a0ffe52e049f6a3b_pact_s3103",
        "hazard_frames": 17609,
        "task_success": False,
        "selection_role": "honest_pact_failure",
    },
)

V2_SELECTION_RULES = {
    "instance_a": (
        "among the 48 instance-seeds where PACT succeeds and ACT fails, select "
        "the pair with the largest ACT hazard-frame total"
    ),
    "instance_b": (
        "among the 34 instance-seeds where ACT succeeds and PACT fails, select "
        "the pair with the largest PACT hazard-frame total"
    ),
}

V3_FIXED_CLIPS = (
    {
        "clip_id": "clip3_e99dc657bfa7_act_success_s3102",
        "pair_id": "instance_b_seed3102",
        "arm": "ACT",
        "episode_id": "e99dc657bfa703eac0d75566c733613ca0ffede3a4bbc394a35c350c753a4391",
        "checkpoint_seed": 3102,
        "intrusion_side": "left",
        "source_directory": "0986_35da71238ef0ed1d_act_s3102",
        "schedule_index": 986,
        "hazard_frames": 18447,
        "grasp_target_frames": 24923,
        "first_hazard_contact_step": 340,
        "first_grasp_target_contact_step": 145,
        "task_success": True,
        "selection_role": "task_success_while_scraping",
    },
    {
        "clip_id": "clip4_e99dc657bfa7_pact_failure_s3102",
        "pair_id": "instance_b_seed3102",
        "arm": "PACT",
        "episode_id": "e99dc657bfa703eac0d75566c733613ca0ffede3a4bbc394a35c350c753a4391",
        "checkpoint_seed": 3102,
        "intrusion_side": "left",
        "source_directory": "0984_ee62f8041692665e_pact_s3102",
        "schedule_index": 984,
        "hazard_frames": 14675,
        "grasp_target_frames": 809,
        "first_hazard_contact_step": 456,
        "first_grasp_target_contact_step": 147,
        "task_success": False,
        "selection_role": "matched_pact_failure_with_delayed_contact",
    },
)

V3_FALLBACK_FIXED = {
    "clip_id": "clip3_3fe3a173f2bf_act_success_s3103",
    "arm": "ACT",
    "episode_id": "3fe3a173f2bf117388f4bfd6f7035c3e4665d6ac9da6a0fef86b0cc2eb4aa236",
    "checkpoint_seed": 3103,
    "intrusion_side": "right",
    "source_directory": "1133_c39a3a8fdd8c6ae4_act_s3103",
    "schedule_index": 1133,
    "hazard_frames": 16739,
    "grasp_target_frames": 25126,
    "first_hazard_contact_step": 393,
    "first_grasp_target_contact_step": 135,
    "task_success": True,
}

V3_FALLBACK2_FIXED = {
    "clip_id": "clip3_178a8383cda2_act_success_s3101",
    "arm": "ACT",
    "episode_id": "178a8383cda28da99d539d84364aa6cb758943bb51b8230b49627cf5cb6f63b4",
    "checkpoint_seed": 3101,
    "intrusion_side": "right",
    "source_directory": "0282_dfbef2c46b0cf55b_act_s3101",
    "schedule_index": 282,
    "hazard_frames": 12087,
    "grasp_target_frames": 24686,
    "first_hazard_contact_step": 137,
    "first_grasp_target_contact_step": 119,
    "task_success": True,
}

ACT_SUCCESS_SUPPLEMENT_FIXED = {
    "clip_id": "clip3_25d96dd30260_act_success",
    "arm": "ACT",
    "episode_id": "25d96dd30260534413867e3d233ac259731dba68b82ecdcddd9dd10d000348be",
    "checkpoint_seed": 3103,
    "intrusion_side": "right",
    "source_directory": "0011_429ae21412b1a318_act_s3103",
    "schedule_index": 11,
    "hazard_frames": 0,
    "grasp_target_frames": 24576,
    "task_success": True,
}

RECORDED_ACT_SUCCESS_FIXED = {
    "clip_id": "clip3_5b4288fea187_act_success_wrist",
    "arm": "ACT",
    "episode_id": "5b4288fea1870475134ce93e8613654ffa3a535466789e50525f36b254722546",
    "checkpoint_seed": 3101,
    "intrusion_side": "right",
    "source_directory": "000_00683ce9e651981d_act",
    "schedule_index": 0,
    "hazard_frames": 0,
    "grasp_target_frames": 24769,
    "task_success": True,
}

SELECTION_RULES = {
    "mechanism": (
        "among instance-seed pairs where ACT hazard frames > 500 and PACT = 0, "
        "take the three with the largest ACT hazard-frame total"
    ),
    "routine": (
        "among pairs where both arms have 0 hazard frames and both succeed at the "
        "task, take the lowest instance ID"
    ),
    "counterexample": (
        "among pairs where PACT hazard frames ≥ ACT, take the one with the largest "
        "PACT total"
    ),
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
        raise ValueError(f"{label} self-hash mismatch: {observed} != {expected}")
    return str(observed)


def load_result(
    row: dict[str, Any], result_root: Path
) -> tuple[dict[str, Any], Path]:
    path = result_root / row["output_relpath"] / "result.json"
    result = json.loads(path.read_text())
    expected = {
        "status": "complete",
        "rollout_id": row["rollout_id"],
        "schedule_row_sha256": row["schedule_row_sha256"],
        "episode_id": row["instance_episode_id"],
        "arm": row["arm"],
        "checkpoint_seed": row["checkpoint_seed"],
    }
    observed = {key: result.get(key) for key in expected}
    if observed != expected:
        raise ValueError(
            f"result identity mismatch at schedule row {row['schedule_index']}: "
            f"{observed} != {expected}"
        )
    return result, path


def arm_record(
    row: dict[str, Any], result: dict[str, Any], result_path: Path
) -> dict[str, Any]:
    audit = result["contact_audit"]
    return {
        "schedule_index": int(row["schedule_index"]),
        "rollout_id": row["rollout_id"],
        "schedule_row_sha256": row["schedule_row_sha256"],
        "checkpoint_path": row["checkpoint_path"],
        "checkpoint_sha256": row["checkpoint_sha256"],
        "dataset_stats_path": row["dataset_stats_path"],
        "dataset_stats_sha256": row["dataset_stats_sha256"],
        "surface_encoder_path": row.get("surface_encoder_path"),
        "surface_encoder_sha256": row.get("surface_encoder_sha256"),
        "original_result_path": str(result_path.resolve()),
        "original_result_sha256": file_hash(result_path),
        "original_outcome": {
            "task_success": bool(result["task_success"]),
            "collision_free_task_success": bool(
                result["collision_free_task_success"]
            ),
            "contact_class_totals": dict(audit["contact_class_totals"]),
            "frames_with_contact": dict(audit["frames_with_contact"]),
            "first_contact_step": dict(audit["first_contact_step"]),
            "failure_taxonomy": result["failure_taxonomy"],
        },
    }


def pair_records(
    schedule: dict[str, Any], result_root: Path
) -> dict[tuple[str, int], dict[str, Any]]:
    records: dict[tuple[str, int], dict[str, Any]] = {}
    for row in schedule["rows"]:
        if row["arm"] not in ARMS:
            continue
        key = (row["instance_episode_id"], int(row["checkpoint_seed"]))
        pair = records.setdefault(
            key,
            {
                "episode_id": key[0],
                "policy_seed": key[1],
                "instance_role_index": int(row["instance_role_index"]),
                "arms": {},
            },
        )
        if row["arm"] in pair["arms"]:
            raise ValueError(f"duplicate {row['arm']} result for {key}")
        result, result_path = load_result(row, result_root)
        pair["arms"][row["arm"]] = arm_record(row, result, result_path)
    expected = int(schedule["instance_count"]) * len(schedule["checkpoint_seeds"])
    if len(records) != expected:
        raise ValueError(f"paired-record count {len(records)} != {expected}")
    incomplete = [key for key, pair in records.items() if set(pair["arms"]) != set(ARMS)]
    if incomplete:
        raise ValueError(f"incomplete paired records: {incomplete[:3]}")
    return records


def hazard_frames(pair: dict[str, Any], arm: str) -> int:
    return int(pair["arms"][arm]["original_outcome"]["frames_with_contact"]["hazard_bar"])


def task_success(pair: dict[str, Any], arm: str) -> bool:
    return bool(pair["arms"][arm]["original_outcome"]["task_success"])


def select(records: dict[tuple[str, int], dict[str, Any]]) -> list[dict[str, Any]]:
    pairs = list(records.values())
    mechanism = sorted(
        (
            pair
            for pair in pairs
            if hazard_frames(pair, "ACT") > THRESHOLD
            and hazard_frames(pair, "PACT") == 0
        ),
        key=lambda pair: (
            -hazard_frames(pair, "ACT"),
            pair["episode_id"],
            pair["policy_seed"],
        ),
    )
    if len(mechanism) < 3:
        raise ValueError(f"only {len(mechanism)} mechanism candidates")
    selected: list[tuple[str, dict[str, Any]]] = [
        ("mechanism", pair) for pair in mechanism[:3]
    ]

    routine = sorted(
        (
            pair
            for pair in pairs
            if hazard_frames(pair, "ACT") == 0
            and hazard_frames(pair, "PACT") == 0
            and task_success(pair, "ACT")
            and task_success(pair, "PACT")
        ),
        key=lambda pair: (pair["episode_id"], pair["policy_seed"]),
    )
    if not routine:
        raise ValueError("no routine candidate")
    selected.append(("routine", routine[0]))

    counterexample = sorted(
        (
            pair
            for pair in pairs
            if hazard_frames(pair, "PACT") >= hazard_frames(pair, "ACT")
        ),
        key=lambda pair: (
            -hazard_frames(pair, "PACT"),
            pair["episode_id"],
            pair["policy_seed"],
        ),
    )
    if not counterexample:
        raise ValueError("no counterexample candidate")
    selected.append(("counterexample", counterexample[0]))

    output = []
    for index, (category, pair) in enumerate(selected, start=1):
        record = dict(pair)
        record["video_index"] = index
        record["video_id"] = f"video_{index:02d}"
        record["category"] = category
        record["selection_rule"] = SELECTION_RULES[category]
        output.append(record)
    return output


def build_v2_manifest(
    schedule: dict[str, Any], result_root: Path, output: Path
) -> dict[str, Any]:
    records = pair_records(schedule, result_root)
    pairs = list(records.values())
    instance_a_candidates = sorted(
        (
            pair
            for pair in pairs
            if task_success(pair, "PACT") and not task_success(pair, "ACT")
        ),
        key=lambda pair: (
            -hazard_frames(pair, "ACT"),
            pair["episode_id"],
            pair["policy_seed"],
        ),
    )
    instance_b_candidates = sorted(
        (
            pair
            for pair in pairs
            if task_success(pair, "ACT") and not task_success(pair, "PACT")
        ),
        key=lambda pair: (
            -hazard_frames(pair, "PACT"),
            pair["episode_id"],
            pair["policy_seed"],
        ),
    )
    if len(instance_a_candidates) != 48:
        raise ValueError(
            f"instance A candidate count {len(instance_a_candidates)} != 48"
        )
    if len(instance_b_candidates) != 34:
        raise ValueError(
            f"instance B candidate count {len(instance_b_candidates)} != 34"
        )
    expected_a = (V2_FIXED_CLIPS[0]["episode_id"], 3101)
    expected_b = (V2_FIXED_CLIPS[2]["episode_id"], 3103)
    observed_a = (
        instance_a_candidates[0]["episode_id"],
        instance_a_candidates[0]["policy_seed"],
    )
    observed_b = (
        instance_b_candidates[0]["episode_id"],
        instance_b_candidates[0]["policy_seed"],
    )
    if observed_a != expected_a or observed_b != expected_b:
        raise ValueError(
            "fixed v2 selections are no longer the mechanically ranked maxima: "
            f"A={observed_a}, B={observed_b}"
        )

    clips = []
    for clip_index, fixed in enumerate(V2_FIXED_CLIPS, start=1):
        key = (fixed["episode_id"], int(fixed["checkpoint_seed"]))
        pair = records[key]
        arm = str(fixed["arm"])
        frozen = pair["arms"][arm]
        result_path = Path(frozen["original_result_path"])
        result = json.loads(result_path.read_text())
        source_directory = result_path.parent.name
        audit = result["contact_audit"]
        observed = {
            "arm": result["arm"],
            "episode_id": result["episode_id"],
            "checkpoint_seed": int(result["checkpoint_seed"]),
            "intrusion_side": result["intrusion_side"],
            "source_directory": source_directory,
            "hazard_frames": int(audit["frames_with_contact"]["hazard_bar"]),
            "task_success": bool(result["task_success"]),
        }
        expected = {
            key: fixed[key]
            for key in (
                "arm",
                "episode_id",
                "checkpoint_seed",
                "intrusion_side",
                "source_directory",
                "hazard_frames",
                "task_success",
            )
        }
        if observed != expected:
            raise ValueError(
                f"fixed v2 clip {fixed['clip_id']} source mismatch: "
                f"{observed} != {expected}"
            )
        clips.append(
            {
                "clip_index": clip_index,
                "clip_id": fixed["clip_id"],
                "pair_id": fixed["pair_id"],
                "selection_role": fixed["selection_role"],
                "selection_rule": V2_SELECTION_RULES[fixed["pair_id"]],
                "episode_id": fixed["episode_id"],
                "checkpoint_seed": int(fixed["checkpoint_seed"]),
                "intrusion_side": fixed["intrusion_side"],
                "arm": arm,
                "source_directory": source_directory,
                "schedule_index": int(frozen["schedule_index"]),
                "rollout_id": frozen["rollout_id"],
                "schedule_row_sha256": frozen["schedule_row_sha256"],
                "checkpoint_path": frozen["checkpoint_path"],
                "checkpoint_sha256": frozen["checkpoint_sha256"],
                "dataset_stats_path": frozen["dataset_stats_path"],
                "dataset_stats_sha256": frozen["dataset_stats_sha256"],
                "surface_encoder_path": frozen["surface_encoder_path"],
                "surface_encoder_sha256": frozen["surface_encoder_sha256"],
                "original_result_path": str(result_path.resolve()),
                "original_result_sha256": file_hash(result_path),
                "original_outcome": {
                    "task_success": bool(result["task_success"]),
                    "collision_free_task_success": bool(
                        result["collision_free_task_success"]
                    ),
                    "hazard_contact": bool(
                        audit["frames_with_contact"]["hazard_bar"]
                    ),
                    "hazard_frames": int(
                        audit["frames_with_contact"]["hazard_bar"]
                    ),
                    "hazard_contact_pair_samples": int(
                        audit["contact_class_totals"]["hazard_bar"]
                    ),
                    "first_hazard_contact_step": audit["first_contact_step"][
                        "hazard_bar"
                    ],
                    "maximum_hazard_penetration_depth_m": audit[
                        "maximum_penetration_depth_m"
                    ]["hazard_bar"],
                },
            }
        )

    schedule_payload = dict(schedule)
    schedule_hash = schedule_payload.pop("schedule_sha256")
    if schedule_hash != canonical_hash(schedule_payload):
        raise ValueError("schedule self-hash mismatch")
    frozen_manifest = DEFAULT_OUTPUT
    frozen_report = ROOT / "docs/PACT_QUALITATIVE_VIDEOS.md"
    determinism_check = DEFAULT_VIDEO_ROOT / "determinism_check.json"
    document: dict[str, Any] = {
        "schema_version": "pact_qualitative_clips_v2",
        "status": "selection_frozen_pre_render",
        "decision_bearing": False,
        "presentation_release": True,
        "selection_frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_viewing_before_freeze": False,
        "selection_rules": dict(V2_SELECTION_RULES),
        "selection_candidate_counts": {"instance_a": 48, "instance_b": 34},
        "clips": clips,
        "caption_text": (
            "Re-rendered from the analyzed rollout. Task success, manipulation "
            "success, and first-contact step reproduce exactly; contact-pair "
            "samples differ by 0.017%."
        ),
        "render_contract": {
            "camera_type": "MuJoCo free camera, offscreen render only",
            "camera_pose_identical_across_clips": True,
            "registered_sensor_or_observation_camera": False,
            "policy_camera_names": ["wrist_camera"],
            "resolution_width_height": [624, 352],
            "raw_fps": 1000.0 / 66.0,
            "raw_frames": 901,
            "playback_speed_factor": 3.0,
            "expected_release_duration_seconds": 19.821366,
            "overlay_fields": [
                "policy arm and checkpoint seed",
                "episode ID first 12 characters",
                "task success yes/no",
                "any hazard contact yes/no",
                "hazard-contact frames running cumulative",
                "maximum hazard penetration",
                "constant playback speed factor",
            ],
        },
        "determinism_contract": {
            "required_exact_fields": [
                "task_success",
                "manipulation_success (represented by task_success)",
                "contact_audit.first_contact_step",
            ],
            "contact_pair_samples_are_descriptive": True,
            "drop_clip_on_required_field_mismatch": True,
        },
        "sources": {
            "schedule": {
                "path": str(DEFAULT_SCHEDULE.resolve()),
                "file_sha256": file_hash(DEFAULT_SCHEDULE),
                "schedule_sha256": schedule_hash,
            },
            "result_root": str(result_root.resolve()),
            "frozen_qualitative_manifest": {
                "path": str(frozen_manifest.resolve()),
                "sha256": file_hash(frozen_manifest),
            },
            "frozen_qualitative_report": {
                "path": str(frozen_report.resolve()),
                "sha256": file_hash(frozen_report),
            },
            "prior_determinism_check": {
                "path": str(determinism_check.resolve()),
                "sha256": file_hash(determinism_check),
            },
        },
        "render_outputs": [],
    }
    document["qualitative_clips_v2_manifest_sha256"] = canonical_hash(document)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return document


def build_v3_manifest(
    schedule: dict[str, Any], result_root: Path, output: Path
) -> dict[str, Any]:
    rows = {int(row["schedule_index"]): row for row in schedule["rows"]}
    clips = []
    for fixed in V3_FIXED_CLIPS:
        row = rows[int(fixed["schedule_index"])]
        result, result_path = load_result(row, result_root)
        audit = result["contact_audit"]
        observed = {
            "clip_id": fixed["clip_id"],
            "pair_id": fixed["pair_id"],
            "arm": result["arm"],
            "episode_id": result["episode_id"],
            "checkpoint_seed": int(result["checkpoint_seed"]),
            "intrusion_side": result["intrusion_side"],
            "source_directory": result_path.parent.name,
            "schedule_index": int(row["schedule_index"]),
            "hazard_frames": int(audit["frames_with_contact"]["hazard_bar"]),
            "grasp_target_frames": int(
                audit["frames_with_contact"]["grasp_target"]
            ),
            "first_hazard_contact_step": audit["first_contact_step"][
                "hazard_bar"
            ],
            "first_grasp_target_contact_step": audit["first_contact_step"][
                "grasp_target"
            ],
            "task_success": bool(result["task_success"]),
            "selection_role": fixed["selection_role"],
        }
        if observed != fixed:
            raise ValueError(
                f"fixed v3 clip {fixed['clip_id']} source mismatch: "
                f"{observed} != {fixed}"
            )
        clips.append(
            {
                **observed,
                "rollout_id": row["rollout_id"],
                "schedule_row_sha256": row["schedule_row_sha256"],
                "checkpoint_path": row["checkpoint_path"],
                "checkpoint_sha256": row["checkpoint_sha256"],
                "dataset_stats_path": row["dataset_stats_path"],
                "dataset_stats_sha256": row["dataset_stats_sha256"],
                "surface_encoder_path": row.get("surface_encoder_path"),
                "surface_encoder_sha256": row.get("surface_encoder_sha256"),
                "original_result_path": str(result_path.resolve()),
                "original_result_sha256": file_hash(result_path),
                "original_outcome": {
                    "task_success": bool(result["task_success"]),
                    "collision_free_task_success": bool(
                        result["collision_free_task_success"]
                    ),
                    "hazard_contact": bool(
                        audit["frames_with_contact"]["hazard_bar"]
                    ),
                    "hazard_frames": int(
                        audit["frames_with_contact"]["hazard_bar"]
                    ),
                    "hazard_contact_pair_samples": int(
                        audit["contact_class_totals"]["hazard_bar"]
                    ),
                    "grasp_target_frames": int(
                        audit["frames_with_contact"]["grasp_target"]
                    ),
                    "first_contact_step": dict(audit["first_contact_step"]),
                    "maximum_hazard_penetration_depth_m": float(
                        audit["maximum_penetration_depth_m"]["hazard_bar"]
                    ),
                },
            }
        )
    if len({(clip["episode_id"], clip["checkpoint_seed"]) for clip in clips}) != 1:
        raise ValueError("v3 clips do not form one matched episode-seed pair")

    schedule_payload = dict(schedule)
    schedule_hash = schedule_payload.pop("schedule_sha256")
    if schedule_hash != canonical_hash(schedule_payload):
        raise ValueError("schedule self-hash mismatch")
    document: dict[str, Any] = {
        "schema_version": "pact_qualitative_clips_v3_manifest",
        "status": "selection_and_gate_frozen_pre_render",
        "decision_bearing": False,
        "presentation_release": True,
        "selection_frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_viewing_before_freeze": False,
        "selection_rule": (
            "replace Instance B with the same frozen episode at adjacent seed 3102, "
            "where ACT succeeds while scraping and PACT fails after later, lower-total contact"
        ),
        "clips": clips,
        "determinism_gate": {
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
                "rationale": (
                    "the published caption claims exact hazard first-contact, not exact "
                    "target first-contact"
                ),
            },
            "contact_pair_sample_counts": {
                "comparison": "informational_only",
                "record_delta": True,
            },
            "on_breach": "drop_clip_without_retry",
            "relaxation_scope": "grasp_target first-contact step only",
        },
        "fallback_order_if_seed3102_gate_fails": [
            {
                "rank": 1,
                "episode_id": "3fe3a173f2bf117388f4bfd6f7035c3e4665d6ac9da6a0fef86b0cc2eb4aa236",
                "checkpoint_seed": 3103,
                "arm": "ACT",
                "source_directory": "1133_c39a3a8fdd8c6ae4_act_s3103",
                "hazard_frames": 16739,
            },
            {
                "rank": 2,
                "episode_id": "178a8383cda28da99d539d84364aa6cb758943bb51b8230b49627cf5cb6f63b4",
                "checkpoint_seed": 3101,
                "arm": "ACT",
                "source_directory": "0282_dfbef2c46b0cf55b_act_s3101",
                "hazard_frames": 12087,
            },
            {
                "rank": 3,
                "rule": (
                    "among the 159 clean contact-endpoint ACT successes, choose the "
                    "lowest schedule_index"
                ),
            },
        ],
        "render_contract": {
            "camera_type": "MuJoCo free camera, offscreen render only",
            "camera_pose_identical_to_clips_1_2_4_v2": True,
            "registered_sensor_or_observation_camera": False,
            "policy_camera_names": ["wrist_camera"],
            "resolution_width_height": [624, 352],
            "raw_fps": 1000.0 / 66.0,
            "raw_frames": 901,
            "playback_speed_factor": 3.0,
            "expected_release_duration_seconds": 19.821366,
            "overlay_fields": [
                "policy arm and checkpoint seed",
                "episode ID first 12 characters",
                "task success yes/no",
                "any hazard contact yes/no",
                "hazard-contact frames running cumulative",
                "maximum hazard penetration",
                "constant playback speed factor",
            ],
        },
        "sources": {
            "schedule": {
                "path": str(DEFAULT_SCHEDULE.resolve()),
                "file_sha256": file_hash(DEFAULT_SCHEDULE),
                "schedule_sha256": schedule_hash,
            },
            "result_root": str(result_root.resolve()),
            "qualitative_clips_v2_manifest": {
                "path": str(V2_OUTPUT.resolve()),
                "sha256": file_hash(V2_OUTPUT),
            },
            "frozen_qualitative_report": {
                "path": str((ROOT / "docs/PACT_QUALITATIVE_VIDEOS.md").resolve()),
                "sha256": file_hash(ROOT / "docs/PACT_QUALITATIVE_VIDEOS.md"),
            },
        },
        "render_outputs": [],
    }
    document["qualitative_clips_v3_manifest_sha256"] = canonical_hash(document)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return document


def build_v3_fallback_manifest(
    schedule: dict[str, Any], result_root: Path, output: Path
) -> dict[str, Any]:
    v3 = json.loads(V3_OUTPUT.read_text())
    v3_payload = dict(v3)
    v3_hash = v3_payload.pop("qualitative_clips_v3_manifest_sha256")
    if v3_hash != canonical_hash(v3_payload):
        raise ValueError("v3 manifest self-hash mismatch")
    if (
        v3["status"] != "presentation_release_incomplete_gate_drop"
        or v3["determinism_summary"]["dropped_clip_ids"]
        != [clip["clip_id"] for clip in V3_FIXED_CLIPS]
    ):
        raise ValueError("v3 primary pair did not fail exactly as recorded")
    fallback_order = v3["fallback_order_if_seed3102_gate_fails"]
    if fallback_order[0]["source_directory"] != V3_FALLBACK_FIXED[
        "source_directory"
    ]:
        raise ValueError("v3 fallback rank 1 changed")

    row = next(
        row
        for row in schedule["rows"]
        if int(row["schedule_index"]) == V3_FALLBACK_FIXED["schedule_index"]
    )
    result, result_path = load_result(row, result_root)
    audit = result["contact_audit"]
    observed = {
        "clip_id": V3_FALLBACK_FIXED["clip_id"],
        "arm": result["arm"],
        "episode_id": result["episode_id"],
        "checkpoint_seed": int(result["checkpoint_seed"]),
        "intrusion_side": result["intrusion_side"],
        "source_directory": result_path.parent.name,
        "schedule_index": int(row["schedule_index"]),
        "hazard_frames": int(audit["frames_with_contact"]["hazard_bar"]),
        "grasp_target_frames": int(
            audit["frames_with_contact"]["grasp_target"]
        ),
        "first_hazard_contact_step": audit["first_contact_step"]["hazard_bar"],
        "first_grasp_target_contact_step": audit["first_contact_step"][
            "grasp_target"
        ],
        "task_success": bool(result["task_success"]),
    }
    if observed != V3_FALLBACK_FIXED:
        raise ValueError(
            f"v3 fallback rank-1 source mismatch: {observed} != "
            f"{V3_FALLBACK_FIXED}"
        )
    schedule_payload = dict(schedule)
    schedule_hash = schedule_payload.pop("schedule_sha256")
    if schedule_hash != canonical_hash(schedule_payload):
        raise ValueError("schedule self-hash mismatch")
    document: dict[str, Any] = {
        "schema_version": "pact_qualitative_clip3_fallback_manifest_v1",
        "status": "selection_and_gate_frozen_pre_render",
        "decision_bearing": False,
        "presentation_release": True,
        "selection_frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_viewing_before_freeze": False,
        "fallback_rank": 1,
        "selection_rule": (
            "use the first predeclared fallback after the matched seed-3102 ACT and "
            "PACT rerenders both failed the frozen v3 gate"
        ),
        "pairing_claim_allowed": False,
        "clip": {
            **observed,
            "rollout_id": row["rollout_id"],
            "schedule_row_sha256": row["schedule_row_sha256"],
            "checkpoint_path": row["checkpoint_path"],
            "checkpoint_sha256": row["checkpoint_sha256"],
            "dataset_stats_path": row["dataset_stats_path"],
            "dataset_stats_sha256": row["dataset_stats_sha256"],
            "surface_encoder_path": row.get("surface_encoder_path"),
            "surface_encoder_sha256": row.get("surface_encoder_sha256"),
            "original_result_path": str(result_path.resolve()),
            "original_result_sha256": file_hash(result_path),
            "original_outcome": {
                "task_success": True,
                "collision_free_task_success": bool(
                    result["collision_free_task_success"]
                ),
                "hazard_contact": True,
                "hazard_frames": int(
                    audit["frames_with_contact"]["hazard_bar"]
                ),
                "hazard_contact_pair_samples": int(
                    audit["contact_class_totals"]["hazard_bar"]
                ),
                "grasp_target_frames": int(
                    audit["frames_with_contact"]["grasp_target"]
                ),
                "first_contact_step": dict(audit["first_contact_step"]),
                "maximum_hazard_penetration_depth_m": float(
                    audit["maximum_penetration_depth_m"]["hazard_bar"]
                ),
            },
        },
        "determinism_gate": dict(v3["determinism_gate"]),
        "remaining_fallback_order_on_gate_failure": fallback_order[1:],
        "render_contract": dict(v3["render_contract"]),
        "sources": {
            "schedule": {
                "path": str(DEFAULT_SCHEDULE.resolve()),
                "file_sha256": file_hash(DEFAULT_SCHEDULE),
                "schedule_sha256": schedule_hash,
            },
            "result_root": str(result_root.resolve()),
            "failed_v3_primary_manifest": {
                "path": str(V3_OUTPUT.resolve()),
                "file_sha256": file_hash(V3_OUTPUT),
                "manifest_sha256": v3_hash,
            },
        },
        "render_output": None,
    }
    document["qualitative_clip3_fallback_manifest_sha256"] = canonical_hash(
        document
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return document


def build_v3_fallback_rank2_manifest(
    schedule: dict[str, Any], result_root: Path, output: Path
) -> dict[str, Any]:
    rank1 = json.loads(V3_FALLBACK_OUTPUT.read_text())
    rank1_payload = dict(rank1)
    rank1_hash = rank1_payload.pop("qualitative_clip3_fallback_manifest_sha256")
    if rank1_hash != canonical_hash(rank1_payload):
        raise ValueError("fallback-rank-1 manifest self-hash mismatch")
    if rank1["status"] != "fallback_rank1_dropped_gate_failure":
        raise ValueError("fallback rank 1 did not fail its frozen gate")
    rank2_plan = rank1["remaining_fallback_order_on_gate_failure"][0]
    if rank2_plan["source_directory"] != V3_FALLBACK2_FIXED[
        "source_directory"
    ]:
        raise ValueError("fallback rank 2 changed")

    row = next(
        row
        for row in schedule["rows"]
        if int(row["schedule_index"]) == V3_FALLBACK2_FIXED["schedule_index"]
    )
    result, result_path = load_result(row, result_root)
    audit = result["contact_audit"]
    observed = {
        "clip_id": V3_FALLBACK2_FIXED["clip_id"],
        "arm": result["arm"],
        "episode_id": result["episode_id"],
        "checkpoint_seed": int(result["checkpoint_seed"]),
        "intrusion_side": result["intrusion_side"],
        "source_directory": result_path.parent.name,
        "schedule_index": int(row["schedule_index"]),
        "hazard_frames": int(audit["frames_with_contact"]["hazard_bar"]),
        "grasp_target_frames": int(
            audit["frames_with_contact"]["grasp_target"]
        ),
        "first_hazard_contact_step": audit["first_contact_step"]["hazard_bar"],
        "first_grasp_target_contact_step": audit["first_contact_step"][
            "grasp_target"
        ],
        "task_success": bool(result["task_success"]),
    }
    if observed != V3_FALLBACK2_FIXED:
        raise ValueError(
            f"fallback-rank-2 source mismatch: {observed} != "
            f"{V3_FALLBACK2_FIXED}"
        )
    schedule_payload = dict(schedule)
    schedule_hash = schedule_payload.pop("schedule_sha256")
    if schedule_hash != canonical_hash(schedule_payload):
        raise ValueError("schedule self-hash mismatch")
    document: dict[str, Any] = {
        "schema_version": "pact_qualitative_clip3_fallback_rank2_manifest_v1",
        "status": "selection_and_gate_frozen_pre_render",
        "decision_bearing": False,
        "presentation_release": True,
        "selection_frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_viewing_before_freeze": False,
        "fallback_rank": 2,
        "selection_rule": (
            "use the second predeclared fallback after the matched seed-3102 pair "
            "and fallback rank 1 failed the frozen gate"
        ),
        "pairing_claim_allowed": False,
        "clip": {
            **observed,
            "rollout_id": row["rollout_id"],
            "schedule_row_sha256": row["schedule_row_sha256"],
            "checkpoint_path": row["checkpoint_path"],
            "checkpoint_sha256": row["checkpoint_sha256"],
            "dataset_stats_path": row["dataset_stats_path"],
            "dataset_stats_sha256": row["dataset_stats_sha256"],
            "surface_encoder_path": row.get("surface_encoder_path"),
            "surface_encoder_sha256": row.get("surface_encoder_sha256"),
            "original_result_path": str(result_path.resolve()),
            "original_result_sha256": file_hash(result_path),
            "original_outcome": {
                "task_success": True,
                "collision_free_task_success": bool(
                    result["collision_free_task_success"]
                ),
                "hazard_contact": True,
                "hazard_frames": int(
                    audit["frames_with_contact"]["hazard_bar"]
                ),
                "hazard_contact_pair_samples": int(
                    audit["contact_class_totals"]["hazard_bar"]
                ),
                "grasp_target_frames": int(
                    audit["frames_with_contact"]["grasp_target"]
                ),
                "first_contact_step": dict(audit["first_contact_step"]),
                "maximum_hazard_penetration_depth_m": float(
                    audit["maximum_penetration_depth_m"]["hazard_bar"]
                ),
            },
        },
        "determinism_gate": dict(rank1["determinism_gate"]),
        "remaining_fallback_order_on_gate_failure": rank1[
            "remaining_fallback_order_on_gate_failure"
        ][1:],
        "render_contract": dict(rank1["render_contract"]),
        "sources": {
            "schedule": {
                "path": str(DEFAULT_SCHEDULE.resolve()),
                "file_sha256": file_hash(DEFAULT_SCHEDULE),
                "schedule_sha256": schedule_hash,
            },
            "result_root": str(result_root.resolve()),
            "failed_fallback_rank1_manifest": {
                "path": str(V3_FALLBACK_OUTPUT.resolve()),
                "file_sha256": file_hash(V3_FALLBACK_OUTPUT),
                "manifest_sha256": rank1_hash,
            },
        },
        "render_output": None,
    }
    document["qualitative_clip3_fallback_rank2_manifest_sha256"] = canonical_hash(
        document
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return document


def build_act_success_supplement(
    schedule: dict[str, Any], result_root: Path, output: Path
) -> dict[str, Any]:
    candidates = []
    for row in schedule["rows"]:
        if row["arm"] != "ACT":
            continue
        result, result_path = load_result(row, result_root)
        if result["task_success"]:
            candidates.append((int(row["schedule_index"]), row, result, result_path))
    candidates.sort(key=lambda item: item[0])
    if len(candidates) != 169:
        raise ValueError(f"ACT-success candidate count {len(candidates)} != 169")
    schedule_index, row, result, result_path = candidates[0]
    audit = result["contact_audit"]
    observed = {
        "clip_id": ACT_SUCCESS_SUPPLEMENT_FIXED["clip_id"],
        "arm": result["arm"],
        "episode_id": result["episode_id"],
        "checkpoint_seed": int(result["checkpoint_seed"]),
        "intrusion_side": result["intrusion_side"],
        "source_directory": result_path.parent.name,
        "schedule_index": schedule_index,
        "hazard_frames": int(audit["frames_with_contact"]["hazard_bar"]),
        "grasp_target_frames": int(
            audit["frames_with_contact"]["grasp_target"]
        ),
        "task_success": bool(result["task_success"]),
    }
    if observed != ACT_SUCCESS_SUPPLEMENT_FIXED:
        raise ValueError(
            f"mechanically selected ACT success changed: {observed} != "
            f"{ACT_SUCCESS_SUPPLEMENT_FIXED}"
        )
    schedule_payload = dict(schedule)
    schedule_hash = schedule_payload.pop("schedule_sha256")
    if schedule_hash != canonical_hash(schedule_payload):
        raise ValueError("schedule self-hash mismatch")
    document: dict[str, Any] = {
        "schema_version": "pact_qualitative_act_success_supplement_v1",
        "status": "selection_frozen_pre_render",
        "decision_bearing": False,
        "presentation_release": True,
        "selection_frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_viewing_before_freeze": False,
        "selection_rule": (
            "among all registered ACT rollouts with task_success=true, select the "
            "lowest schedule_index"
        ),
        "selection_candidate_count": len(candidates),
        "clip": {
            **observed,
            "rollout_id": row["rollout_id"],
            "schedule_row_sha256": row["schedule_row_sha256"],
            "checkpoint_path": row["checkpoint_path"],
            "checkpoint_sha256": row["checkpoint_sha256"],
            "dataset_stats_path": row["dataset_stats_path"],
            "dataset_stats_sha256": row["dataset_stats_sha256"],
            "original_result_path": str(result_path.resolve()),
            "original_result_sha256": file_hash(result_path),
            "original_outcome": {
                "task_success": True,
                "collision_free_task_success": bool(
                    result["collision_free_task_success"]
                ),
                "hazard_contact": False,
                "hazard_frames": 0,
                "hazard_contact_pair_samples": int(
                    audit["contact_class_totals"]["hazard_bar"]
                ),
                "grasp_target_frames": int(
                    audit["frames_with_contact"]["grasp_target"]
                ),
                "first_contact_step": dict(audit["first_contact_step"]),
                "maximum_hazard_penetration_depth_m": float(
                    audit["maximum_penetration_depth_m"]["hazard_bar"]
                ),
            },
        },
        "render_contract": {
            "camera_type": "MuJoCo free camera, offscreen render only",
            "camera_pose_identical_to_existing_release": True,
            "registered_sensor_or_observation_camera": False,
            "policy_camera_names": ["wrist_camera"],
            "resolution_width_height": [624, 352],
            "raw_fps": 1000.0 / 66.0,
            "raw_frames": 901,
            "playback_speed_factor": 3.0,
            "expected_release_duration_seconds": 19.821366,
        },
        "determinism_contract": {
            "required_exact_fields": [
                "task_success",
                "manipulation_success (represented by task_success)",
                "contact_audit.first_contact_step",
            ],
            "drop_clip_on_required_field_mismatch": True,
            "rerun_or_substitute_on_mismatch": False,
        },
        "sources": {
            "schedule": {
                "path": str(DEFAULT_SCHEDULE.resolve()),
                "file_sha256": file_hash(DEFAULT_SCHEDULE),
                "schedule_sha256": schedule_hash,
            },
            "result_root": str(result_root.resolve()),
            "qualitative_clips_v2_manifest": {
                "path": str(V2_OUTPUT.resolve()),
                "sha256": file_hash(V2_OUTPUT),
            },
        },
        "render_output": None,
    }
    document["act_success_supplement_manifest_sha256"] = canonical_hash(document)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return document


def build_recorded_act_success_supplement(output: Path) -> dict[str, Any]:
    schedule = json.loads(FRONTEND_SCREEN_SCHEDULE.read_text())
    schedule_payload = dict(schedule)
    schedule_hash = schedule_payload.pop("schedule_sha256")
    if schedule_hash != canonical_hash(schedule_payload):
        raise ValueError("front-end screen schedule self-hash mismatch")

    candidates = []
    for row in schedule["rows"]:
        if row["arm"] != "ACT":
            continue
        result_path = FRONTEND_SCREEN_RESULT_ROOT / row["output_relpath"].split("/", 1)[1] / "result.json"
        source_video = result_path.parent / "episode_00000000_wrist_camera.mp4"
        if not result_path.is_file() or not source_video.is_file():
            continue
        result = json.loads(result_path.read_text())
        if result["task_success"]:
            candidates.append(
                (int(row["schedule_index"]), row, result, result_path, source_video)
            )
    candidates.sort(key=lambda item: item[0])
    if len(candidates) != 19:
        raise ValueError(
            f"recorded ACT-success candidate count {len(candidates)} != 19"
        )
    schedule_index, row, result, result_path, source_video = candidates[0]
    audit = result["contact_audit"]
    observed = {
        "clip_id": RECORDED_ACT_SUCCESS_FIXED["clip_id"],
        "arm": result["arm"],
        "episode_id": result["episode_id"],
        "checkpoint_seed": int(result["checkpoint_seed"]),
        "intrusion_side": result["intrusion_side"],
        "source_directory": result_path.parent.name,
        "schedule_index": schedule_index,
        "hazard_frames": int(audit["frames_with_contact"]["hazard_bar"]),
        "grasp_target_frames": int(audit["frames_with_contact"]["grasp_target"]),
        "task_success": bool(result["task_success"]),
    }
    if observed != RECORDED_ACT_SUCCESS_FIXED:
        raise ValueError(
            f"mechanically selected recorded ACT success changed: {observed} != "
            f"{RECORDED_ACT_SUCCESS_FIXED}"
        )

    failed_attempt = json.loads(ACT_SUCCESS_SUPPLEMENT_OUTPUT.read_text())
    failed_payload = dict(failed_attempt)
    failed_hash = failed_payload.pop("act_success_supplement_manifest_sha256")
    if failed_hash != canonical_hash(failed_payload):
        raise ValueError("contact-endpoint ACT-success attempt self-hash mismatch")
    if failed_attempt["status"] != "dropped_determinism_mismatch":
        raise ValueError("contact-endpoint ACT-success attempt was not frozen as dropped")

    document: dict[str, Any] = {
        "schema_version": "pact_qualitative_recorded_act_success_supplement_v1",
        "status": "selection_frozen_pre_overlay",
        "decision_bearing": False,
        "presentation_release": True,
        "selection_frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_viewing_before_freeze": False,
        "selection_rule": (
            "among ACT task successes in the frozen 120-row front-end screen with "
            "an intact original wrist-camera recording, select the lowest schedule_index"
        ),
        "selection_candidate_count": len(candidates),
        "clip": {
            **observed,
            "rollout_id": row["rollout_id"],
            "schedule_row_sha256": row["schedule_row_sha256"],
            "checkpoint_path": row["checkpoint_path"],
            "checkpoint_sha256": row["checkpoint_sha256"],
            "dataset_stats_path": row["dataset_stats_path"],
            "dataset_stats_sha256": row["dataset_stats_sha256"],
            "original_result_path": str(result_path.resolve()),
            "original_result_sha256": file_hash(result_path),
            "original_video_path": str(source_video.resolve()),
            "original_video_sha256": file_hash(source_video),
            "original_outcome": {
                "task_success": True,
                "collision_free_task_success": bool(
                    result["collision_free_task_success"]
                ),
                "hazard_contact": False,
                "hazard_frames": 0,
                "hazard_contact_pair_samples": int(
                    audit["contact_class_totals"]["hazard_bar"]
                ),
                "grasp_target_frames": int(
                    audit["frames_with_contact"]["grasp_target"]
                ),
                "first_contact_step": dict(audit["first_contact_step"]),
                "maximum_hazard_penetration_depth_m": None,
            },
        },
        "render_contract": {
            "source": "original wrist_camera recording from the analyzed rollout",
            "policy_rerun": False,
            "overlay_only_transcode": True,
            "camera_identical_to_three_third_person_clips": False,
            "resolution_width_height": [624, 352],
            "raw_fps": 1000.0 / 66.0,
            "raw_frames": 901,
            "playback_speed_factor": 3.0,
            "expected_release_duration_seconds": 19.823982,
            "maximum_hazard_penetration_overlay": "n/a (no hazard contact)",
        },
        "sources": {
            "schedule": {
                "path": str(FRONTEND_SCREEN_SCHEDULE.resolve()),
                "file_sha256": file_hash(FRONTEND_SCREEN_SCHEDULE),
                "schedule_sha256": schedule_hash,
            },
            "result_root": str(FRONTEND_SCREEN_RESULT_ROOT.resolve()),
            "dropped_contact_endpoint_attempt": {
                "path": str(ACT_SUCCESS_SUPPLEMENT_OUTPUT.resolve()),
                "file_sha256": file_hash(ACT_SUCCESS_SUPPLEMENT_OUTPUT),
                "manifest_sha256": failed_hash,
                "reason": "first target contact differed by one step (154 to 153)",
            },
        },
        "render_output": None,
    }
    document["recorded_act_success_supplement_manifest_sha256"] = canonical_hash(
        document
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=(
            "legacy-pairs-v1",
            "clips-v2",
            "clips-v3",
            "clips-v3-fallback",
            "clips-v3-fallback-rank2",
            "act-success-supplement",
            "recorded-act-success",
        ),
        default="legacy-pairs-v1",
    )
    parser.add_argument("--schedule", type=Path, default=DEFAULT_SCHEDULE)
    parser.add_argument("--tail", type=Path, default=DEFAULT_TAIL)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is None:
        if args.mode == "clips-v2":
            args.output = V2_OUTPUT
        elif args.mode == "clips-v3":
            args.output = V3_OUTPUT
        elif args.mode == "clips-v3-fallback":
            args.output = V3_FALLBACK_OUTPUT
        elif args.mode == "clips-v3-fallback-rank2":
            args.output = V3_FALLBACK2_OUTPUT
        elif args.mode == "act-success-supplement":
            args.output = ACT_SUCCESS_SUPPLEMENT_OUTPUT
        elif args.mode == "recorded-act-success":
            args.output = RECORDED_ACT_SUCCESS_OUTPUT
        else:
            args.output = DEFAULT_OUTPUT
    if args.output.exists():
        raise SystemExit(f"refusing to replace frozen manifest: {args.output}")

    if args.mode == "clips-v2":
        schedule = json.loads(args.schedule.read_text())
        document = build_v2_manifest(schedule, args.result_root, args.output)
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "sha256": document["qualitative_clips_v2_manifest_sha256"],
                    "clips": [
                        {
                            "clip_id": clip["clip_id"],
                            "arm": clip["arm"],
                            "episode_id": clip["episode_id"],
                            "checkpoint_seed": clip["checkpoint_seed"],
                            "hazard_frames": clip["original_outcome"][
                                "hazard_frames"
                            ],
                            "task_success": clip["original_outcome"][
                                "task_success"
                            ],
                        }
                        for clip in document["clips"]
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.mode == "clips-v3":
        schedule = json.loads(args.schedule.read_text())
        document = build_v3_manifest(schedule, args.result_root, args.output)
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "sha256": document["qualitative_clips_v3_manifest_sha256"],
                    "gate": document["determinism_gate"],
                    "clips": [
                        {
                            "clip_id": clip["clip_id"],
                            "arm": clip["arm"],
                            "episode_id": clip["episode_id"],
                            "checkpoint_seed": clip["checkpoint_seed"],
                            "hazard_frames": clip["hazard_frames"],
                            "grasp_target_frames": clip["grasp_target_frames"],
                            "task_success": clip["task_success"],
                        }
                        for clip in document["clips"]
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.mode == "clips-v3-fallback":
        schedule = json.loads(args.schedule.read_text())
        document = build_v3_fallback_manifest(
            schedule, args.result_root, args.output
        )
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "sha256": document[
                        "qualitative_clip3_fallback_manifest_sha256"
                    ],
                    "fallback_rank": document["fallback_rank"],
                    "clip": document["clip"],
                    "gate": document["determinism_gate"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.mode == "clips-v3-fallback-rank2":
        schedule = json.loads(args.schedule.read_text())
        document = build_v3_fallback_rank2_manifest(
            schedule, args.result_root, args.output
        )
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "sha256": document[
                        "qualitative_clip3_fallback_rank2_manifest_sha256"
                    ],
                    "fallback_rank": document["fallback_rank"],
                    "clip": document["clip"],
                    "gate": document["determinism_gate"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.mode == "act-success-supplement":
        schedule = json.loads(args.schedule.read_text())
        document = build_act_success_supplement(
            schedule, args.result_root, args.output
        )
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "sha256": document[
                        "act_success_supplement_manifest_sha256"
                    ],
                    "clip": document["clip"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.mode == "recorded-act-success":
        document = build_recorded_act_success_supplement(args.output)
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "sha256": document[
                        "recorded_act_success_supplement_manifest_sha256"
                    ],
                    "clip": document["clip"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    tail = json.loads(args.tail.read_text())
    tail_hash = validate_self_hash(
        tail, "tail_characterization_sha256", "tail characterization"
    )
    if (
        tail.get("status") != "post_hoc_exploratory_descriptive"
        or tail["threshold"].get("metric") != "hazard_bar frames_with_contact"
        or tail["threshold"].get("operator") != ">"
        or int(tail["threshold"].get("value", -1)) != THRESHOLD
    ):
        raise ValueError("tail characterization threshold contract changed")
    schedule = json.loads(args.schedule.read_text())
    schedule_payload = dict(schedule)
    schedule_observed = schedule_payload.pop("schedule_sha256", None)
    if schedule_observed != canonical_hash(schedule_payload):
        raise ValueError("schedule self-hash mismatch")

    records = pair_records(schedule, args.result_root)
    selections = select(records)
    all_selected_rows = [
        arm["schedule_index"]
        for selection in selections
        for arm in selection["arms"].values()
    ]
    if len(all_selected_rows) != 10 or len(set(all_selected_rows)) != 10:
        raise ValueError("selection did not resolve to ten unique schedule rows")

    now = datetime.now(timezone.utc).isoformat()
    document: dict[str, Any] = {
        "schema_version": "pact_qualitative_video_manifest_v1",
        "status": "selection_frozen_pre_render",
        "decision_bearing": False,
        "illustrative_only": True,
        "selection_frozen_at_utc": now,
        "selection_rule_verbatim": dict(SELECTION_RULES),
        "selection_tie_breaks": {
            "mechanism": "episode ID ascending, then policy seed ascending",
            "routine": "policy seed ascending within the lowest episode ID",
            "counterexample": "episode ID ascending, then policy seed ascending",
        },
        "selection_viewing_before_freeze": False,
        "selections": selections,
        "determinism_check": {
            "status": "pending",
            "required_exact_fields": [
                "contact_audit.contact_class_totals",
                "task_success",
                "manipulation_success (represented by task_success)",
                "contact_audit.first_contact_step",
            ],
            "probe_video_id": selections[0]["video_id"],
            "probe_arm": "ACT",
        },
        "render_contract": {
            "camera_type": "MuJoCo free camera, offscreen render only",
            "registered_sensor_or_observation_camera": False,
            "observation_key_added": False,
            "policy_camera_names": ["wrist_camera"],
            "policy_inputs": {
                "ACT": ["wrist_camera", "qpos"],
                "PACT": ["wrist_camera", "qpos", "40 skin streams"],
            },
            "camera_pose": {
                "reference_body": "robot_0/fr3_link0",
                "camera_offset_m": [-1.05, -0.55, 1.3],
                "lookat_offset_m": [0.55, 0.0, 0.45],
                "up_axis": "world_z",
                "vertical_fov_degrees": 58.0,
            },
            "resolution_width_height": [624, 352],
            "fps": 1000.0 / 66.0,
            "max_control_steps": 900,
            "rng_calls_added": 0,
        },
        "video_root": str(args.video_root.resolve()),
        "sources": {
            "schedule": {
                "path": str(args.schedule.resolve()),
                "file_sha256": file_hash(args.schedule),
                "schedule_sha256": schedule_observed,
            },
            "tail_characterization": {
                "path": str(args.tail.resolve()),
                "file_sha256": file_hash(args.tail),
                "tail_characterization_sha256": tail_hash,
            },
            "result_root": str(args.result_root.resolve()),
        },
        "render_outputs": [],
        "composition_outputs": [],
    }
    document["qualitative_video_manifest_sha256"] = canonical_hash(document)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": document["qualitative_video_manifest_sha256"],
                "selections": [
                    {
                        "video_id": item["video_id"],
                        "category": item["category"],
                        "episode_id": item["episode_id"],
                        "policy_seed": item["policy_seed"],
                        "act_hazard_frames": hazard_frames(item, "ACT"),
                        "pact_hazard_frames": hazard_frames(item, "PACT"),
                    }
                    for item in selections
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
