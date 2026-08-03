#!/usr/bin/env python3
"""Record the mandatory qualitative determinism-gate stop in the manifest."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "diagnostics_output/pact_contact_endpoint/qualitative_video_manifest.json"
VIDEO_ROOT = Path("/root/pact_contact_endpoint_artifacts/qualitative_videos")
CHECK = VIDEO_ROOT / "determinism_check.json"
RERUN_RESULT = VIDEO_ROOT / "reruns/video_01_act/result.json"
PROBE_VIDEO = VIDEO_ROOT / "raw/video_01_act.mp4"


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


def main() -> int:
    manifest = json.loads(MANIFEST.read_text())
    validate_self_hash(
        manifest, "qualitative_video_manifest_sha256", "qualitative manifest"
    )
    if manifest.get("status") != "selection_frozen_pre_render":
        raise SystemExit("qualitative manifest is not at the pre-render freeze")
    selections_hash = canonical_hash(manifest["selections"])

    check = json.loads(CHECK.read_text())
    check_self_hash = validate_self_hash(
        check, "determinism_check_sha256", "determinism check"
    )
    if (
        check.get("status") != "failed_mismatch_stop"
        or check.get("exact_match") is not False
    ):
        raise ValueError("this finalizer only records the mandatory mismatch stop")
    rerun = json.loads(RERUN_RESULT.read_text())
    if (
        rerun.get("status") != "complete"
        or rerun.get("episode_id") != check["episode_id"]
        or rerun.get("arm") != check["arm"]
        or int(rerun.get("checkpoint_seed", -1)) != int(check["policy_seed"])
    ):
        raise ValueError("determinism rerun identity mismatch")
    original_path = Path(
        manifest["selections"][0]["arms"]["ACT"]["original_result_path"]
    )
    original = json.loads(original_path.read_text())
    if file_hash(original_path) != manifest["selections"][0]["arms"]["ACT"][
        "original_result_sha256"
    ]:
        raise ValueError("original result hash changed")
    render = rerun["policy_info"]["qualitative_render"]
    if (
        render.get("render_only") is not True
        or render.get("camera_registered_in_observation") is not False
        or render.get("policy_camera_names") != ["wrist_camera"]
        or int(render.get("video_frames", -1)) != 901
    ):
        raise ValueError("probe render contract did not hold")
    if file_hash(PROBE_VIDEO) != check["rendered_video"]["sha256"]:
        raise ValueError("probe-video hash differs from determinism record")

    manifest["status"] = "aborted_determinism_mismatch"
    manifest["determinism_check"] = {
        "status": check["status"],
        "exact_match": False,
        "probe_video_id": "video_01",
        "probe_arm": check["arm"],
        "episode_id": check["episode_id"],
        "policy_seed": check["policy_seed"],
        "schedule_index": check["schedule_index"],
        "required_exact_fields": [
            "contact_audit.contact_class_totals",
            "task_success",
            "manipulation_success (represented by task_success)",
            "contact_audit.first_contact_step",
        ],
        "comparisons": check["comparisons"],
        "additional_descriptive_differences": {
            "hazard_frames_with_contact": {
                "original": manifest["selections"][0]["arms"]["ACT"][
                    "original_outcome"
                ]["frames_with_contact"]["hazard_bar"],
                "rerun": rerun["contact_audit"]["frames_with_contact"][
                    "hazard_bar"
                ],
            },
            "hazard_maximum_penetration_depth_m": {
                "original": original["contact_audit"][
                    "maximum_penetration_depth_m"
                ]["hazard_bar"],
                "rerun": rerun["contact_audit"]["maximum_penetration_depth_m"][
                    "hazard_bar"
                ],
            },
            "contact_audit_sample_count": {
                "original": original["contact_audit"]["sample_count"],
                "rerun": rerun["contact_audit"]["sample_count"],
            },
            "initial_observation_boundary_sha256": {
                "original": original["initial_observation_boundary_sha256"],
                "rerun": rerun["initial_observation_boundary_sha256"],
            },
        },
        "cause": (
            "undetermined: the exact check cannot separate render perturbation from "
            "rollout non-determinism"
        ),
        "action": "stopped before the remaining nine reruns",
        "external_record": {
            "path": str(CHECK.resolve()),
            "file_sha256": file_hash(CHECK),
            "determinism_check_sha256": check_self_hash,
        },
    }
    manifest["render_outputs"] = [
        {
            "role": "unpublished_determinism_probe_only",
            "deliverable_video": False,
            "labelled_as_analyzed_rollout": False,
            "independent_draw_warning_required_if_ever_published": True,
            "video_id": "video_01",
            "arm": "ACT",
            "episode_id": check["episode_id"],
            "policy_seed": check["policy_seed"],
            "path": str(PROBE_VIDEO.resolve()),
            "file_sha256": file_hash(PROBE_VIDEO),
            "size_bytes": PROBE_VIDEO.stat().st_size,
            "codec": "mpeg4",
            "resolution_width_height": [624, 352],
            "frames": 901,
            "fps": 1000.0 / 66.0,
            "duration_seconds": 59.464097,
            "rerun_result": {
                "path": str(RERUN_RESULT.resolve()),
                "file_sha256": file_hash(RERUN_RESULT),
            },
        }
    ]
    manifest["composition_outputs"] = []
    manifest["completion"] = {
        "requested_paired_videos": 5,
        "completed_paired_videos": 0,
        "remaining_selected_reruns_launched": 0,
        "stopped_by_predeclared_gate": True,
        "scientific_results_or_token_changed": False,
    }
    if canonical_hash(manifest["selections"]) != selections_hash:
        raise AssertionError("frozen qualitative selections changed during finalization")
    manifest.pop("qualitative_video_manifest_sha256", None)
    manifest["qualitative_video_manifest_sha256"] = canonical_hash(manifest)
    temporary = MANIFEST.with_name(f".{MANIFEST.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, MANIFEST)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "output": str(MANIFEST),
                "qualitative_video_manifest_sha256": manifest[
                    "qualitative_video_manifest_sha256"
                ],
                "selections_sha256": selections_hash,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
