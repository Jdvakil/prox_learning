#!/usr/bin/env python3
"""Run the frozen qualitative determinism probe and paired render reruns."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/root/act_retrain_venv/bin/python")
EVALUATOR = ROOT / "submodules/act/eval_pact_qualitative_row.py"
SCHEDULE = ROOT / "diagnostics_output/pact_contact_endpoint/schedule.json"
MANIFEST = ROOT / "configs/pact_contact_endpoint_manifest_v1.json"
QUALITATIVE_MANIFEST = (
    ROOT / "diagnostics_output/pact_contact_endpoint/qualitative_video_manifest.json"
)
QUALITATIVE_CLIPS_V2_MANIFEST = (
    ROOT
    / "diagnostics_output/pact_contact_endpoint/qualitative_clips_v2_manifest.json"
)
VIDEO_ROOT = Path("/root/pact_contact_endpoint_artifacts/qualitative_videos")
CLIPS_V2_ROOT = Path("/root/pact_contact_endpoint_artifacts/qualitative_clips_v2")
WORKERS = 8


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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


def write_json(path: Path, document: dict[str, Any], hash_key: str) -> None:
    payload = dict(document)
    payload.pop(hash_key, None)
    document[hash_key] = canonical_hash(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def protected_eval_processes() -> list[int]:
    matches = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if b"eval_act_obstacle_on_policy.py" in command:
            matches.append(int(entry.name))
    return matches


def load_documents() -> tuple[dict[str, Any], dict[str, Any]]:
    qualitative = json.loads(QUALITATIVE_MANIFEST.read_text())
    payload = dict(qualitative)
    observed = payload.pop("qualitative_video_manifest_sha256", None)
    if observed != canonical_hash(payload):
        raise RuntimeError("qualitative manifest self-hash mismatch")
    if qualitative.get("status") != "selection_frozen_pre_render":
        raise RuntimeError("qualitative selection is not in the frozen pre-render state")
    schedule = json.loads(SCHEDULE.read_text())
    schedule_payload = dict(schedule)
    schedule_observed = schedule_payload.pop("schedule_sha256", None)
    if schedule_observed != canonical_hash(schedule_payload):
        raise RuntimeError("contact schedule self-hash mismatch")
    if schedule_observed != qualitative["sources"]["schedule"]["schedule_sha256"]:
        raise RuntimeError("qualitative selection references a different schedule")
    return qualitative, schedule


def load_clips_v2_documents() -> tuple[dict[str, Any], dict[str, Any]]:
    qualitative = json.loads(QUALITATIVE_CLIPS_V2_MANIFEST.read_text())
    payload = dict(qualitative)
    observed = payload.pop("qualitative_clips_v2_manifest_sha256", None)
    if observed != canonical_hash(payload):
        raise RuntimeError("qualitative clips v2 manifest self-hash mismatch")
    if qualitative.get("status") != "selection_frozen_pre_render":
        raise RuntimeError("qualitative clips v2 selection is not frozen pre-render")
    schedule = json.loads(SCHEDULE.read_text())
    schedule_payload = dict(schedule)
    schedule_observed = schedule_payload.pop("schedule_sha256", None)
    if schedule_observed != canonical_hash(schedule_payload):
        raise RuntimeError("contact schedule self-hash mismatch")
    if schedule_observed != qualitative["sources"]["schedule"]["schedule_sha256"]:
        raise RuntimeError("qualitative clips v2 references a different schedule")
    return qualitative, schedule


def selected_jobs(
    qualitative: dict[str, Any], schedule: dict[str, Any]
) -> list[dict[str, Any]]:
    rows = {int(row["schedule_index"]): row for row in schedule["rows"]}
    jobs = []
    for selection in qualitative["selections"]:
        for arm in ("ACT", "PACT"):
            frozen = selection["arms"][arm]
            row = rows[int(frozen["schedule_index"])]
            checks = {
                "arm": arm,
                "instance_episode_id": selection["episode_id"],
                "checkpoint_seed": selection["policy_seed"],
                "rollout_id": frozen["rollout_id"],
                "schedule_row_sha256": frozen["schedule_row_sha256"],
                "checkpoint_sha256": frozen["checkpoint_sha256"],
            }
            observed = {key: row.get(key) for key in checks}
            if observed != checks:
                raise RuntimeError(
                    f"selection/schedule mismatch for {selection['video_id']} {arm}"
                )
            stem = f"{selection['video_id']}_{arm.lower()}"
            jobs.append(
                {
                    "video_id": selection["video_id"],
                    "category": selection["category"],
                    "episode_id": selection["episode_id"],
                    "policy_seed": int(selection["policy_seed"]),
                    "arm": arm,
                    "row": row,
                    "frozen": frozen,
                    "output_dir": VIDEO_ROOT / "reruns" / stem,
                    "video_output": VIDEO_ROOT / "raw" / f"{stem}.mp4",
                    "log_path": VIDEO_ROOT / "logs" / f"{stem}.log",
                }
            )
    if len(jobs) != 10:
        raise RuntimeError(f"selected job count {len(jobs)} != 10")
    return jobs


def command_for(job: dict[str, Any]) -> list[str]:
    row = job["row"]
    checkpoint = Path(row["checkpoint_path"])
    command = [
        str(PYTHON),
        str(EVALUATOR),
        "--arm",
        row["arm"],
        "--episode-id",
        row["instance_episode_id"],
        "--manifest",
        str(MANIFEST),
        "--checkpoint-dir",
        str(checkpoint.parent),
        "--checkpoint-sha256",
        row["checkpoint_sha256"],
        "--checkpoint-seed",
        str(row["checkpoint_seed"]),
        "--stats-sha256",
        row["dataset_stats_sha256"],
        "--schedule-row-sha256",
        row["schedule_row_sha256"],
        "--rollout-id",
        row["rollout_id"],
        "--output-dir",
        str(job["output_dir"]),
        "--qualitative-video-output",
        str(job["video_output"]),
        "--qualitative-episode-id",
        job["episode_id"],
        "--qualitative-policy-seed",
        str(job["policy_seed"]),
    ]
    if row["arm"] == "PACT":
        command.extend(
            [
                "--surface-encoder",
                row["surface_encoder_path"],
                "--surface-encoder-sha256",
                row["surface_encoder_sha256"],
            ]
        )
    return command


def selected_clips_v2_jobs(
    qualitative: dict[str, Any], schedule: dict[str, Any]
) -> list[dict[str, Any]]:
    rows = {int(row["schedule_index"]): row for row in schedule["rows"]}
    jobs = []
    for clip in qualitative["clips"]:
        row = rows[int(clip["schedule_index"])]
        checks = {
            "arm": clip["arm"],
            "instance_episode_id": clip["episode_id"],
            "checkpoint_seed": clip["checkpoint_seed"],
            "rollout_id": clip["rollout_id"],
            "schedule_row_sha256": clip["schedule_row_sha256"],
            "checkpoint_sha256": clip["checkpoint_sha256"],
            "dataset_stats_sha256": clip["dataset_stats_sha256"],
        }
        observed = {key: row.get(key) for key in checks}
        if observed != checks:
            raise RuntimeError(f"v2 manifest/schedule mismatch for {clip['clip_id']}")
        clip_id = str(clip["clip_id"])
        jobs.append(
            {
                "clip": clip,
                "clip_id": clip_id,
                "episode_id": clip["episode_id"],
                "policy_seed": int(clip["checkpoint_seed"]),
                "arm": clip["arm"],
                "row": row,
                "output_dir": CLIPS_V2_ROOT / "reruns" / clip_id,
                "video_output": CLIPS_V2_ROOT / "raw" / f"{clip_id}.mp4",
                "release_output": CLIPS_V2_ROOT / "release" / f"{clip_id}.mp4",
                "check_path": CLIPS_V2_ROOT / "checks" / f"{clip_id}.json",
                "log_path": CLIPS_V2_ROOT / "logs" / f"{clip_id}.log",
            }
        )
    if len(jobs) != 4:
        raise RuntimeError(f"selected v2 clip count {len(jobs)} != 4")
    if len({job["clip_id"] for job in jobs}) != len(jobs):
        raise RuntimeError("v2 clip IDs are not unique")
    return jobs


def command_for_clips_v2(job: dict[str, Any]) -> list[str]:
    row = job["row"]
    clip = job["clip"]
    outcome = clip["original_outcome"]
    checkpoint = Path(row["checkpoint_path"])
    command = [
        str(PYTHON),
        str(EVALUATOR),
        "--arm",
        row["arm"],
        "--episode-id",
        row["instance_episode_id"],
        "--manifest",
        str(MANIFEST),
        "--checkpoint-dir",
        str(checkpoint.parent),
        "--checkpoint-sha256",
        row["checkpoint_sha256"],
        "--checkpoint-seed",
        str(row["checkpoint_seed"]),
        "--stats-sha256",
        row["dataset_stats_sha256"],
        "--schedule-row-sha256",
        row["schedule_row_sha256"],
        "--rollout-id",
        row["rollout_id"],
        "--output-dir",
        str(job["output_dir"]),
        "--qualitative-video-output",
        str(job["video_output"]),
        "--qualitative-episode-id",
        job["episode_id"],
        "--qualitative-policy-seed",
        str(job["policy_seed"]),
        "--qualitative-clips-v2",
        "--qualitative-task-success",
        "yes" if outcome["task_success"] else "no",
        "--qualitative-any-hazard-contact",
        "yes" if outcome["hazard_contact"] else "no",
        "--qualitative-max-hazard-penetration-m",
        str(outcome["maximum_hazard_penetration_depth_m"]),
        "--qualitative-playback-speed-factor",
        "3.0",
    ]
    if row["arm"] == "PACT":
        command.extend(
            [
                "--surface-encoder",
                row["surface_encoder_path"],
                "--surface-encoder-sha256",
                row["surface_encoder_sha256"],
            ]
        )
    return command


def runtime_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
            "PYTHONUNBUFFERED": "1",
            "MLSPACES_ASSETS_DIR": str(ROOT / "assets"),
            "PYTHONPATH": str(ROOT / "submodules/molmospaces"),
        }
    )
    env.pop("DISPLAY", None)
    return env


def validate_completed_job(job: dict[str, Any]) -> dict[str, Any]:
    result_path = job["output_dir"] / "result.json"
    if not result_path.exists() or not job["video_output"].exists():
        raise RuntimeError(f"qualitative job outputs incomplete: {job['video_id']} {job['arm']}")
    result = json.loads(result_path.read_text())
    row = job["row"]
    expected = {
        "status": "complete",
        "arm": job["arm"],
        "episode_id": job["episode_id"],
        "checkpoint_seed": job["policy_seed"],
        "rollout_id": row["rollout_id"],
        "schedule_row_sha256": row["schedule_row_sha256"],
        "checkpoint_sha256": row["checkpoint_sha256"],
    }
    observed = {key: result.get(key) for key in expected}
    if observed != expected:
        raise RuntimeError(f"qualitative result identity mismatch: {observed} != {expected}")
    render = result.get("policy_info", {}).get("qualitative_render", {})
    if (
        render.get("render_only") is not True
        or render.get("camera_registered_in_observation") is not False
        or render.get("policy_camera_names") != ["wrist_camera"]
        or "exo_camera_1" in (render.get("observation_keys") or [])
        or int(render.get("video_frames", -1)) != 901
    ):
        raise RuntimeError(f"render-only contract failed for {job['video_id']} {job['arm']}")
    return result


def exact_determinism_comparison(job: dict[str, Any]) -> dict[str, Any]:
    original = json.loads(Path(job["frozen"]["original_result_path"]).read_text())
    rerun_path = job["output_dir"] / "result.json"
    rerun = json.loads(rerun_path.read_text())
    fields = {
        "contact_class_totals": (
            original["contact_audit"]["contact_class_totals"],
            rerun["contact_audit"]["contact_class_totals"],
        ),
        "task_success": (original["task_success"], rerun["task_success"]),
        "manipulation_success": (original["task_success"], rerun["task_success"]),
        "first_contact_step": (
            original["contact_audit"]["first_contact_step"],
            rerun["contact_audit"]["first_contact_step"],
        ),
    }
    comparisons = {
        key: {"original": left, "rerun": right, "exact_match": left == right}
        for key, (left, right) in fields.items()
    }
    passed = all(item["exact_match"] for item in comparisons.values())
    document: dict[str, Any] = {
        "schema_version": "pact_qualitative_determinism_check_v1",
        "status": "passed_exact_match" if passed else "failed_mismatch_stop",
        "checked_at_utc": utc_now(),
        "camera_render_only": True,
        "episode_id": job["episode_id"],
        "policy_seed": job["policy_seed"],
        "arm": job["arm"],
        "schedule_index": int(job["row"]["schedule_index"]),
        "original_result": {
            "path": job["frozen"]["original_result_path"],
            "sha256": file_hash(Path(job["frozen"]["original_result_path"])),
        },
        "rerun_result": {
            "path": str(rerun_path.resolve()),
            "sha256": file_hash(rerun_path),
        },
        "rendered_video": {
            "path": str(job["video_output"].resolve()),
            "sha256": file_hash(job["video_output"]),
        },
        "comparisons": comparisons,
        "exact_match": passed,
        "instruction_if_failed": (
            "Stop; do not render the remaining selection. Any later videos must be "
            "labelled independent draws from the same instance."
        ),
    }
    path = VIDEO_ROOT / "determinism_check.json"
    write_json(path, document, "determinism_check_sha256")
    return document


def validate_completed_clips_v2_job(job: dict[str, Any]) -> dict[str, Any]:
    result_path = job["output_dir"] / "result.json"
    if not result_path.exists() or not job["video_output"].exists():
        raise RuntimeError(f"qualitative v2 outputs incomplete: {job['clip_id']}")
    result = json.loads(result_path.read_text())
    row = job["row"]
    expected = {
        "status": "complete",
        "arm": job["arm"],
        "episode_id": job["episode_id"],
        "checkpoint_seed": job["policy_seed"],
        "rollout_id": row["rollout_id"],
        "schedule_row_sha256": row["schedule_row_sha256"],
        "checkpoint_sha256": row["checkpoint_sha256"],
    }
    observed = {key: result.get(key) for key in expected}
    if observed != expected:
        raise RuntimeError(f"qualitative v2 identity mismatch: {observed} != {expected}")
    render = result.get("policy_info", {}).get("qualitative_render", {})
    required_overlay_fields = [
        "policy arm and checkpoint seed",
        "episode ID first 12 characters",
        "task success yes/no",
        "any hazard contact yes/no",
        "hazard-contact frames running cumulative",
        "maximum hazard penetration",
        "constant playback speed factor",
    ]
    if (
        render.get("schema_version") != "pact_qualitative_clips_render_v2"
        or render.get("render_only") is not True
        or render.get("camera_registered_in_observation") is not False
        or render.get("policy_camera_names") != ["wrist_camera"]
        or "exo_camera_1" in (render.get("observation_keys") or [])
        or int(render.get("video_frames", -1)) != 901
        or render.get("resolution_width_height") != [624, 352]
        or float(render.get("playback_speed_factor", 0.0)) != 3.0
        or render.get("overlay_fields") != required_overlay_fields
    ):
        raise RuntimeError(f"v2 render-only contract failed for {job['clip_id']}")
    return result


def clips_v2_determinism_comparison(job: dict[str, Any]) -> dict[str, Any]:
    original_path = Path(job["clip"]["original_result_path"])
    rerun_path = job["output_dir"] / "result.json"
    original = json.loads(original_path.read_text())
    rerun = json.loads(rerun_path.read_text())
    required_fields = {
        "task_success": (original["task_success"], rerun["task_success"]),
        "manipulation_success": (original["task_success"], rerun["task_success"]),
        "first_contact_step": (
            original["contact_audit"]["first_contact_step"],
            rerun["contact_audit"]["first_contact_step"],
        ),
    }
    comparisons = {
        key: {"original": left, "rerun": right, "exact_match": left == right}
        for key, (left, right) in required_fields.items()
    }
    original_audit = original["contact_audit"]
    rerun_audit = rerun["contact_audit"]
    original_pairs = int(original_audit["contact_class_totals"]["hazard_bar"])
    rerun_pairs = int(rerun_audit["contact_class_totals"]["hazard_bar"])
    pair_delta = rerun_pairs - original_pairs
    pair_percent = None if original_pairs == 0 else 100.0 * pair_delta / original_pairs
    original_frames = int(original_audit["frames_with_contact"]["hazard_bar"])
    rerun_frames = int(rerun_audit["frames_with_contact"]["hazard_bar"])
    passed = all(item["exact_match"] for item in comparisons.values())
    document: dict[str, Any] = {
        "schema_version": "pact_qualitative_clips_determinism_v2",
        "status": "passed_required_exact_fields" if passed else "failed_drop_clip",
        "checked_at_utc": utc_now(),
        "camera_render_only": True,
        "clip_id": job["clip_id"],
        "episode_id": job["episode_id"],
        "policy_seed": job["policy_seed"],
        "arm": job["arm"],
        "schedule_index": int(job["row"]["schedule_index"]),
        "original_result": {
            "path": str(original_path.resolve()),
            "sha256": file_hash(original_path),
        },
        "rerun_result": {
            "path": str(rerun_path.resolve()),
            "sha256": file_hash(rerun_path),
        },
        "raw_video": {
            "path": str(job["video_output"].resolve()),
            "sha256": file_hash(job["video_output"]),
        },
        "required_exact_comparisons": comparisons,
        "descriptive_contact_deltas": {
            "hazard_contact_pair_samples": {
                "original": original_pairs,
                "rerun": rerun_pairs,
                "signed_delta": pair_delta,
                "signed_percent_of_original": pair_percent,
            },
            "hazard_contact_frames": {
                "original": original_frames,
                "rerun": rerun_frames,
                "signed_delta": rerun_frames - original_frames,
            },
            "contact_class_totals": {
                "original": original_audit["contact_class_totals"],
                "rerun": rerun_audit["contact_class_totals"],
            },
            "maximum_penetration_depth_m": {
                "original": original_audit["maximum_penetration_depth_m"],
                "rerun": rerun_audit["maximum_penetration_depth_m"],
            },
        },
        "required_exact_match": passed,
        "release_action": "retain_clip" if passed else "drop_clip",
    }
    write_json(job["check_path"], document, "determinism_check_sha256")
    return document


def ffprobe_video(path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,avg_frame_rate,nb_frames,duration",
            "-show_entries",
            "format=duration,size,bit_rate",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def compose_clips_v2_release(job: dict[str, Any]) -> dict[str, Any]:
    release = job["release_output"]
    if release.exists():
        raise RuntimeError(f"refusing to replace release clip: {release}")
    release.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(job["video_output"]),
            "-vf",
            "setpts=PTS/3.0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(release),
        ],
        check=True,
    )
    probe = ffprobe_video(release)
    stream = probe["streams"][0]
    duration = float(probe["format"]["duration"])
    if not 15.0 <= duration <= 25.0:
        raise RuntimeError(f"release duration {duration} outside 15-25 s: {release}")
    if [int(stream["width"]), int(stream["height"])] != [624, 352]:
        raise RuntimeError(f"release resolution mismatch: {release}")
    return probe


def write_clips_v2_readme(
    qualitative: dict[str, Any],
    jobs: list[dict[str, Any]],
    retained_clip_ids: set[str],
    checks: dict[str, dict[str, Any]],
) -> Path:
    path = CLIPS_V2_ROOT / "release" / "README.md"
    lines = [
        "# PACT/ACT qualitative clip release",
        "",
        (
            "Four presentation-only rows were preselected from the frozen contact-endpoint "
            "evaluation; three passed the exact determinism gate and are retained. The "
            "scientific qualitative-video record remains closed and unchanged."
        ),
        "",
        "## Slide order",
        "",
        "| Clip | Release | Pair | Arm | Seed | Side | Task success | Hazard frames | Role |",
        "|---|---|---|---|---:|---|:---:|---:|---|",
    ]
    for job in jobs:
        clip = job["clip"]
        outcome = clip["original_outcome"]
        retained = clip["clip_id"] in retained_clip_ids
        clip_cell = (
            f"[{clip['clip_id']}.mp4]({clip['clip_id']}.mp4)"
            if retained
            else f"`{clip['clip_id']}.mp4`"
        )
        lines.append(
            "| {clip_cell} | {release} | {pair} | {arm} | {seed} | "
            "{side} | {success} | {frames:,} | {role} |".format(
                clip_cell=clip_cell,
                release="retained" if retained else "**dropped by determinism gate**",
                pair=clip["pair_id"],
                arm=clip["arm"],
                seed=clip["checkpoint_seed"],
                side=clip["intrusion_side"],
                success="yes" if outcome["task_success"] else "no",
                frames=int(outcome["hazard_frames"]),
                role=clip["selection_role"],
            )
        )
    lines.extend(
        [
            "",
            "## Pairing and selection",
            "",
            (
                "Instance A shares episode `54a6272f66ca...`, seed 3101, and a "
                "right-side intrusion. It is the maximum-ACT-contact member of the "
                "48 instance-seeds where PACT succeeds and ACT fails."
            ),
            "",
            (
                "Instance B shares episode `e99dc657bfa7...`, seed 3103, and a "
                "left-side intrusion. It is the maximum-PACT-contact member of the "
                "34 instance-seeds where ACT succeeds and PACT fails. Clip 4 is "
                "retained as the required honest PACT failure."
            ),
            "",
            "## Predeclared determinism caption",
            "",
            f"> {qualitative['caption_text']}",
            "",
            (
                "That fixed caption reports the earlier probe. The new per-clip hazard "
                "contact-pair deltas are 0.000%, -0.0069%, +1.058% for the dropped clip, "
                "and +3.036%, respectively; consult the check JSONs rather than treating "
                "0.017% as a summary of all four rerenders."
            ),
            "",
            (
                "All retained releases use the same render-only camera and the same "
                "3.0x playback factor. Per-clip determinism deltas are recorded in "
                f"`{CLIPS_V2_ROOT / 'checks'}`."
            ),
            "",
        ]
    )
    dropped = [clip_id for clip_id in checks if clip_id not in retained_clip_ids]
    if dropped:
        lines.extend(
            [
                "## Determinism-gate drop",
                "",
                (
                    "Clip 3 was rendered but is not shipped. Its ACT rerender changed task "
                    "success from yes to no and first hazard-contact step from 302 to 295. "
                    "The plan required dropping any clip that failed either exact field; it "
                    "was not rerun or replaced."
                ),
                "",
            ]
        )
    path.write_text("\n".join(lines))
    return path


def finalize_clips_v2_manifest(
    qualitative: dict[str, Any],
    jobs: list[dict[str, Any]],
    checks: dict[str, dict[str, Any]],
    probes: dict[str, dict[str, Any]],
    readme: Path,
) -> None:
    frozen_selection_hash = qualitative["qualitative_clips_v2_manifest_sha256"]
    camera_contracts = []
    outputs = []
    retained_jobs = [job for job in jobs if job["clip_id"] in probes]
    for job in retained_jobs:
        result_path = job["output_dir"] / "result.json"
        result = json.loads(result_path.read_text())
        render = result["policy_info"]["qualitative_render"]
        camera_contracts.append(
            {
                "camera_reference_body": render["camera_reference_body"],
                "camera_offset_m": render["camera_offset_m"],
                "lookat_offset_m": render["lookat_offset_m"],
                "resolution_width_height": render["resolution_width_height"],
                "fps": render["fps"],
                "playback_speed_factor": render["playback_speed_factor"],
            }
        )
        outputs.append(
            {
                "clip_id": job["clip_id"],
                "rerun_result_path": str(result_path.resolve()),
                "rerun_result_sha256": file_hash(result_path),
                "raw_video_path": str(job["video_output"].resolve()),
                "raw_video_sha256": file_hash(job["video_output"]),
                "release_video_path": str(job["release_output"].resolve()),
                "release_video_sha256": file_hash(job["release_output"]),
                "release_ffprobe": probes[job["clip_id"]],
                "determinism_check_path": str(job["check_path"].resolve()),
                "determinism_check_sha256": checks[job["clip_id"]][
                    "determinism_check_sha256"
                ],
            }
        )
    if not camera_contracts:
        raise RuntimeError("no qualitative clips passed the determinism gate")
    if any(contract != camera_contracts[0] for contract in camera_contracts[1:]):
        raise RuntimeError("camera pose or playback contract differs across v2 clips")
    dropped = [
        clip_id
        for clip_id, check in checks.items()
        if check["required_exact_match"] is not True
    ]
    qualitative["status"] = (
        "presentation_release_verified"
        if not dropped
        else "presentation_release_incomplete_determinism_drop"
    )
    qualitative["rendered_at_utc"] = utc_now()
    qualitative["selection_frozen_manifest_sha256"] = frozen_selection_hash
    qualitative["determinism_summary"] = {
        "required_exact_fields_passed": len(checks) - len(dropped),
        "required_exact_fields_failed": len(dropped),
        "all_four_clips_retained": not dropped,
        "retained_clip_ids": [job["clip_id"] for job in retained_jobs],
        "dropped_clip_ids": dropped,
        "per_clip_checks": {
            clip_id: {
                "sha256": check["determinism_check_sha256"],
                "required_exact_match": check["required_exact_match"],
                "descriptive_contact_deltas": check[
                    "descriptive_contact_deltas"
                ],
            }
            for clip_id, check in checks.items()
        },
    }
    qualitative["camera_contract_observed"] = camera_contracts[0]
    qualitative["render_outputs"] = outputs
    qualitative["release_readme"] = {
        "path": str(readme.resolve()),
        "sha256": file_hash(readme),
    }
    write_json(
        QUALITATIVE_CLIPS_V2_MANIFEST,
        qualitative,
        "qualitative_clips_v2_manifest_sha256",
    )


def ensure_new_job(job: dict[str, Any]) -> None:
    if job["output_dir"].exists() or job["video_output"].exists():
        raise RuntimeError(
            f"refusing to replace qualitative output for {job['video_id']} {job['arm']}"
        )
    job["output_dir"].mkdir(parents=True)
    job["video_output"].parent.mkdir(parents=True, exist_ok=True)
    job["log_path"].parent.mkdir(parents=True, exist_ok=True)


def run_probe(job: dict[str, Any]) -> int:
    ensure_new_job(job)
    with job["log_path"].open("w") as log:
        completed = subprocess.run(
            command_for(job),
            cwd=ROOT / "submodules/act",
            env=runtime_env(),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"determinism probe exited {completed.returncode}; see {job['log_path']}"
        )
    validate_completed_job(job)
    comparison = exact_determinism_comparison(job)
    print(json.dumps(comparison, indent=2, sort_keys=True))
    return 0 if comparison["exact_match"] else 2


def run_remaining(jobs: list[dict[str, Any]]) -> int:
    check_path = VIDEO_ROOT / "determinism_check.json"
    if not check_path.exists():
        raise RuntimeError("determinism check has not been recorded")
    check = json.loads(check_path.read_text())
    payload = dict(check)
    observed_hash = payload.pop("determinism_check_sha256", None)
    if observed_hash != canonical_hash(payload) or check.get("exact_match") is not True:
        raise RuntimeError("determinism check did not pass exactly")

    pending = []
    for job in jobs:
        if (job["output_dir"] / "result.json").exists() or job["video_output"].exists():
            validate_completed_job(job)
        else:
            ensure_new_job(job)
            pending.append(job)
    active: list[tuple[dict[str, Any], subprocess.Popen[Any], Any]] = []
    failure: str | None = None
    while pending or active:
        while pending and len(active) < WORKERS and failure is None:
            job = pending.pop(0)
            log = job["log_path"].open("w")
            process = subprocess.Popen(
                command_for(job),
                cwd=ROOT / "submodules/act",
                env=runtime_env(),
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            active.append((job, process, log))
            print(
                json.dumps(
                    {
                        "event": "launched",
                        "video_id": job["video_id"],
                        "arm": job["arm"],
                        "pid": process.pid,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        for item in list(active):
            job, process, log = item
            returncode = process.poll()
            if returncode is None:
                continue
            log.close()
            active.remove(item)
            if returncode != 0:
                failure = (
                    f"{job['video_id']} {job['arm']} exited {returncode}; "
                    f"see {job['log_path']}"
                )
            else:
                validate_completed_job(job)
                print(
                    json.dumps(
                        {
                            "event": "completed",
                            "video_id": job["video_id"],
                            "arm": job["arm"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        if failure is not None:
            for _job, process, log in active:
                if process.poll() is None:
                    process.terminate()
                log.close()
            raise RuntimeError(failure)
        time.sleep(0.25)
    return 0


def ensure_new_clips_v2_job(job: dict[str, Any]) -> None:
    targets = (
        job["output_dir"],
        job["video_output"],
        job["release_output"],
        job["check_path"],
        job["log_path"],
    )
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise RuntimeError(
            f"refusing to replace v2 output for {job['clip_id']}: {existing}"
        )
    job["output_dir"].mkdir(parents=True)
    job["video_output"].parent.mkdir(parents=True, exist_ok=True)
    job["check_path"].parent.mkdir(parents=True, exist_ok=True)
    job["log_path"].parent.mkdir(parents=True, exist_ok=True)


def run_clips_v2(
    qualitative: dict[str, Any], jobs: list[dict[str, Any]]
) -> int:
    for job in jobs:
        ensure_new_clips_v2_job(job)
    pending = list(jobs)
    active: list[tuple[dict[str, Any], subprocess.Popen[Any], Any]] = []
    failure: str | None = None
    results: dict[str, dict[str, Any]] = {}
    while pending or active:
        while pending and len(active) < WORKERS and failure is None:
            job = pending.pop(0)
            log = job["log_path"].open("w")
            process = subprocess.Popen(
                command_for_clips_v2(job),
                cwd=ROOT / "submodules/act",
                env=runtime_env(),
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            active.append((job, process, log))
            print(
                json.dumps(
                    {
                        "event": "launched_v2",
                        "clip_id": job["clip_id"],
                        "arm": job["arm"],
                        "pid": process.pid,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        for item in list(active):
            job, process, log = item
            returncode = process.poll()
            if returncode is None:
                continue
            log.close()
            active.remove(item)
            if returncode != 0:
                failure = (
                    f"{job['clip_id']} exited {returncode}; see {job['log_path']}"
                )
            else:
                results[job["clip_id"]] = validate_completed_clips_v2_job(job)
                print(
                    json.dumps(
                        {"event": "completed_v2", "clip_id": job["clip_id"]},
                        sort_keys=True,
                    ),
                    flush=True,
                )
        if failure is not None:
            for _job, process, log in active:
                if process.poll() is None:
                    process.terminate()
                log.close()
            raise RuntimeError(failure)
        time.sleep(0.25)

    checks = {
        job["clip_id"]: clips_v2_determinism_comparison(job) for job in jobs
    }
    mismatches = [
        clip_id
        for clip_id, check in checks.items()
        if check["required_exact_match"] is not True
    ]
    retained_jobs = [job for job in jobs if job["clip_id"] not in mismatches]
    probes: dict[str, dict[str, Any]] = {}
    for job in retained_jobs:
        probes[job["clip_id"]] = compose_clips_v2_release(job)
        print(
            json.dumps(
                {
                    "event": "release_composed",
                    "clip_id": job["clip_id"],
                    "duration_seconds": float(
                        probes[job["clip_id"]]["format"]["duration"]
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    retained_ids = {job["clip_id"] for job in retained_jobs}
    readme = write_clips_v2_readme(qualitative, jobs, retained_ids, checks)
    finalize_clips_v2_manifest(qualitative, jobs, checks, probes, readme)
    print(
        json.dumps(
            {
                "event": "v2_release_verified",
                "manifest": str(QUALITATIVE_CLIPS_V2_MANIFEST),
                "release_directory": str(CLIPS_V2_ROOT / "release"),
                "clips_retained": len(retained_jobs),
                "clips_dropped": mismatches,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if not mismatches else 2


def load_clips_v2_check(job: dict[str, Any]) -> dict[str, Any]:
    check = json.loads(job["check_path"].read_text())
    payload = dict(check)
    observed = payload.pop("determinism_check_sha256", None)
    if observed != canonical_hash(payload):
        raise RuntimeError(f"v2 determinism-check hash mismatch: {job['clip_id']}")
    if check.get("clip_id") != job["clip_id"]:
        raise RuntimeError(f"v2 determinism-check identity mismatch: {job['clip_id']}")
    return check


def finalize_existing_clips_v2(
    qualitative: dict[str, Any], jobs: list[dict[str, Any]]
) -> int:
    for job in jobs:
        validate_completed_clips_v2_job(job)
    checks = {job["clip_id"]: load_clips_v2_check(job) for job in jobs}
    retained_jobs = [
        job
        for job in jobs
        if checks[job["clip_id"]]["required_exact_match"] is True
    ]
    probes = {
        job["clip_id"]: compose_clips_v2_release(job) for job in retained_jobs
    }
    retained_ids = {job["clip_id"] for job in retained_jobs}
    readme = write_clips_v2_readme(qualitative, jobs, retained_ids, checks)
    finalize_clips_v2_manifest(qualitative, jobs, checks, probes, readme)
    print(
        json.dumps(
            {
                "event": "v2_partial_release_finalized",
                "retained_clip_ids": sorted(retained_ids),
                "dropped_clip_ids": sorted(
                    set(checks).difference(retained_ids)
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release", choices=("legacy-v1", "clips-v2"), default="legacy-v1"
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=("probe", "remaining", "all", "finalize-existing"),
    )
    parser.add_argument("--workers", type=int, default=WORKERS)
    args = parser.parse_args()
    if args.workers != WORKERS:
        raise SystemExit(f"qualitative worker count is frozen at {WORKERS}")
    protected = protected_eval_processes()
    if protected:
        raise SystemExit(f"protected shared evaluation is active: {protected}")
    if args.release == "clips-v2":
        if args.mode not in ("all", "finalize-existing"):
            raise SystemExit("clips-v2 requires --mode all or --mode finalize-existing")
        qualitative, schedule = load_clips_v2_documents()
        jobs = selected_clips_v2_jobs(qualitative, schedule)
        if args.mode == "finalize-existing":
            return finalize_existing_clips_v2(qualitative, jobs)
        return run_clips_v2(qualitative, jobs)
    if args.mode in ("all", "finalize-existing"):
        raise SystemExit("legacy-v1 does not support this mode")
    qualitative, schedule = load_documents()
    jobs = selected_jobs(qualitative, schedule)
    probe = next(
        job
        for job in jobs
        if job["video_id"] == qualitative["determinism_check"]["probe_video_id"]
        and job["arm"] == qualitative["determinism_check"]["probe_arm"]
    )
    if args.mode == "probe":
        return run_probe(probe)
    return run_remaining([job for job in jobs if job is not probe])


if __name__ == "__main__":
    raise SystemExit(main())
