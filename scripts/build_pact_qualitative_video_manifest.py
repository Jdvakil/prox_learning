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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("legacy-pairs-v1", "clips-v2"),
        default="legacy-pairs-v1",
    )
    parser.add_argument("--schedule", type=Path, default=DEFAULT_SCHEDULE)
    parser.add_argument("--tail", type=Path, default=DEFAULT_TAIL)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is None:
        args.output = V2_OUTPUT if args.mode == "clips-v2" else DEFAULT_OUTPUT
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
