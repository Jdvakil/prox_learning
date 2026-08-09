#!/usr/bin/env python3
"""Render and gate the frozen geometry-v3 qualitative presentation release."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/root/act_retrain_venv/bin/python")
EVALUATOR = ROOT / "submodules/act/eval_pact_geometry_v3_qualitative_row.py"
MANIFEST_PATH = ROOT / "diagnostics_output/pact_geometry_generalization_v3/qualitative_video_manifest.json"
SCHEDULE_PATH = ROOT / "diagnostics_output/pact_geometry_generalization_v3/schedule.json"
SCIENTIFIC_MANIFEST_PATH = ROOT / "configs/pact_geometry_generalization_v3.json"
ARTIFACT_ROOT = Path("/root/pact_geometry_generalization_v3_artifacts/qualitative_videos")
BUNDLE_ROOT = Path("/root/pact_slideshow_bundle")
C0_RAW = Path(
    "/root/pact_contact_endpoint_artifacts/qualitative_clips_v2/raw/"
    "clip1_54a6272f66ca_pact_success.mp4"
)
WORKERS = 4
HASH_KEY = "qualitative_video_manifest_sha256"


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


def validate_self_hash(document: dict[str, Any], key: str, label: str) -> str:
    payload = dict(document)
    observed = payload.pop(key, None)
    expected = canonical_hash(payload)
    if observed != expected:
        raise RuntimeError(f"{label} self-hash mismatch: {observed} != {expected}")
    return str(observed)


def load_frozen() -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = json.loads(MANIFEST_PATH.read_text())
    validate_self_hash(manifest, HASH_KEY, "qualitative manifest")
    if manifest.get("status") != "selection_and_gate_frozen_pre_render":
        raise RuntimeError("qualitative manifest is not frozen pre-render")
    if manifest["determinism_gate"] != {
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
    }:
        raise RuntimeError("declared determinism gate changed")
    for source_name in ("analysis", "final_decision", "report"):
        source = manifest["sources"][source_name]
        if file_hash(Path(source["path"])) != source["sha256"]:
            raise RuntimeError(f"protected scientific {source_name} changed")
    for source_name, source in manifest["sources"]["runtime_code"].items():
        if file_hash(Path(source["path"])) != source["sha256"]:
            raise RuntimeError(f"frozen runtime code {source_name} changed")
    schedule = json.loads(SCHEDULE_PATH.read_text())
    schedule_hash = validate_self_hash(schedule, "schedule_sha256", "v3 schedule")
    if schedule_hash != manifest["sources"]["schedule"]["schedule_sha256"]:
        raise RuntimeError("qualitative manifest references a different schedule")
    return manifest, schedule


def protected_eval_processes() -> list[int]:
    matches = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            command = (entry / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if b"eval_" in command and b"pgrep" not in command:
            matches.append(int(entry.name))
    return matches


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


def selections_by_rank(manifest: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    result = {}
    for selection in manifest["ranked_selections"]:
        key = (selection["condition_id"], int(selection["selection_rank"]))
        if key in result:
            raise RuntimeError(f"duplicate frozen selection {key}")
        result[key] = selection
    if set(result) != {(condition, rank) for condition in ("C2", "Z_093") for rank in (1, 2, 3)}:
        raise RuntimeError("frozen candidate ranks are incomplete")
    return result


def jobs_for_selection(
    selection: dict[str, Any], schedule: dict[str, Any], frozen_hash: str
) -> list[dict[str, Any]]:
    rows = {int(row["schedule_index"]): row for row in schedule["rows"]}
    jobs = []
    for arm in ("PACT", "PACT_PERMUTED"):
        frozen = selection["arms"][arm]
        row = rows[int(frozen["schedule_index"])]
        expected = {
            "arm": arm,
            "instance_episode_id": selection["episode_id"],
            "checkpoint_seed": selection["checkpoint_seed"],
            "rollout_id": frozen["rollout_id"],
            "schedule_row_sha256": frozen["schedule_row_sha256"],
            "checkpoint_sha256": frozen["checkpoint_sha256"],
            "condition_id": selection["condition_id"],
        }
        if {key: row.get(key) for key in expected} != expected:
            raise RuntimeError(f"selection/schedule mismatch for {selection['pair_id']} {arm}")
        rank = int(selection["selection_rank"])
        stem = f"{selection['pair_id']}_rank{rank}_{arm.lower()}"
        jobs.append(
            {
                "selection": selection,
                "arm_record": frozen,
                "row": row,
                "arm": arm,
                "condition_id": selection["condition_id"],
                "pair_id": selection["pair_id"],
                "selection_rank": rank,
                "clip_id": stem,
                "gate_manifest_sha256": frozen_hash,
                "output_dir": ARTIFACT_ROOT / "reruns" / stem,
                "raw_video": ARTIFACT_ROOT / "raw" / f"{stem}.mp4",
                "release_video": ARTIFACT_ROOT / "release" / f"{stem}.mp4",
                "check_path": ARTIFACT_ROOT / "checks" / f"{stem}.json",
                "log_path": ARTIFACT_ROOT / "logs" / f"{stem}.log",
            }
        )
    return jobs


def command_for(job: dict[str, Any]) -> list[str]:
    row = job["row"]
    frozen = job["arm_record"]
    outcome = frozen["outcome"]
    checkpoint = Path(row["checkpoint_path"])
    command = [
        str(PYTHON),
        str(EVALUATOR),
        "--arm",
        row["arm"],
        "--episode-id",
        row["instance_episode_id"],
        "--manifest",
        str(SCIENTIFIC_MANIFEST_PATH),
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
        "--surface-encoder",
        row["surface_encoder_path"],
        "--surface-encoder-sha256",
        row["surface_encoder_sha256"],
        "--qualitative-video-output",
        str(job["raw_video"]),
        "--qualitative-episode-id",
        row["instance_episode_id"],
        "--qualitative-policy-seed",
        str(row["checkpoint_seed"]),
        "--qualitative-clips-v2",
        "--qualitative-condition-id",
        row["condition_id"],
        "--qualitative-task-success",
        "yes" if outcome["task_success"] else "no",
        "--qualitative-any-hazard-contact",
        "yes" if outcome["hazard_contact"] else "no",
        "--qualitative-max-hazard-penetration-m",
        str(outcome["maximum_hazard_penetration_depth_m"]),
        "--qualitative-playback-speed-factor",
        "3.0",
    ]
    if row["arm"] == "PACT_PERMUTED":
        command.extend(
            [
                "--token-plan-manifest",
                row["token_plan_manifest_path"],
                "--token-plan-row",
                str(row["token_plan_row"]),
            ]
        )
    return command


def ensure_new_job(job: dict[str, Any]) -> None:
    targets = (
        job["output_dir"],
        job["raw_video"],
        job["release_video"],
        job["check_path"],
        job["log_path"],
    )
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise RuntimeError(f"refusing to replace qualitative outputs: {existing}")
    job["output_dir"].mkdir(parents=True)
    job["raw_video"].parent.mkdir(parents=True, exist_ok=True)
    job["check_path"].parent.mkdir(parents=True, exist_ok=True)
    job["log_path"].parent.mkdir(parents=True, exist_ok=True)


def run_jobs(jobs: list[dict[str, Any]]) -> None:
    for job in jobs:
        ensure_new_job(job)
    pending = list(jobs)
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
                        "clip_id": job["clip_id"],
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
                failure = f"{job['clip_id']} exited {returncode}; see {job['log_path']}"
            else:
                validate_completed_job(job)
                print(json.dumps({"event": "completed", "clip_id": job["clip_id"]}, sort_keys=True), flush=True)
        if failure:
            for _job, process, log in active:
                if process.poll() is None:
                    process.terminate()
                log.close()
            raise RuntimeError(failure)
        time.sleep(0.25)


def validate_completed_job(job: dict[str, Any]) -> dict[str, Any]:
    result_path = job["output_dir"] / "result.json"
    if not result_path.exists() or not job["raw_video"].exists():
        raise RuntimeError(f"incomplete render output for {job['clip_id']}")
    result = json.loads(result_path.read_text())
    row = job["row"]
    expected = {
        "status": "complete",
        "arm": row["arm"],
        "episode_id": row["instance_episode_id"],
        "checkpoint_seed": row["checkpoint_seed"],
        "rollout_id": row["rollout_id"],
        "schedule_row_sha256": row["schedule_row_sha256"],
        "checkpoint_sha256": row["checkpoint_sha256"],
    }
    if {key: result.get(key) for key in expected} != expected:
        raise RuntimeError(f"render result identity mismatch for {job['clip_id']}")
    render = result.get("policy_info", {}).get("qualitative_render", {})
    required_overlay = [
        "policy arm and checkpoint seed",
        "condition ID",
        "episode ID first 12 characters",
        "task success yes/no",
        "any hazard contact yes/no",
        "hazard-contact frames running cumulative",
        "maximum hazard penetration",
        "constant playback speed factor",
    ]
    if (
        render.get("schema_version") != "pact_geometry_v3_qualitative_render_v1"
        or render.get("render_only") is not True
        or render.get("camera_registered_in_observation") is not False
        or render.get("policy_camera_names") != ["wrist_camera"]
        or "exo_camera_1" in (render.get("observation_keys") or [])
        or int(render.get("video_frames", -1)) != 901
        or render.get("resolution_width_height") != [624, 352]
        or float(render.get("playback_speed_factor", 0.0)) != 3.0
        or render.get("overlay_fields") != required_overlay
    ):
        raise RuntimeError(f"render-only contract failed for {job['clip_id']}: {render}")
    return result


def first_step_delta(original: Any, rerun: Any) -> tuple[int | None, bool]:
    if original is None and rerun is None:
        return 0, True
    if original is None or rerun is None:
        return None, False
    delta = int(rerun) - int(original)
    return delta, abs(delta) <= 2


def determinism_check(job: dict[str, Any]) -> dict[str, Any]:
    original_path = Path(job["arm_record"]["original_result_path"])
    rerun_path = job["output_dir"] / "result.json"
    original = json.loads(original_path.read_text())
    rerun = json.loads(rerun_path.read_text())
    original_first = original["contact_audit"]["first_contact_step"]
    rerun_first = rerun["contact_audit"]["first_contact_step"]
    target_delta, target_passed = first_step_delta(
        original_first["grasp_target"], rerun_first["grasp_target"]
    )
    comparisons = {
        "task_success": {
            "requirement": "exact",
            "original": original["task_success"],
            "rerun": rerun["task_success"],
            "passed": original["task_success"] == rerun["task_success"],
        },
        "manipulation_success": {
            "requirement": "exact (represented by task_success)",
            "original": original["task_success"],
            "rerun": rerun["task_success"],
            "passed": original["task_success"] == rerun["task_success"],
        },
        "first_hazard_bar_contact_step": {
            "requirement": "exact",
            "original": original_first["hazard_bar"],
            "rerun": rerun_first["hazard_bar"],
            "passed": original_first["hazard_bar"] == rerun_first["hazard_bar"],
        },
        "first_grasp_target_contact_step": {
            "requirement": "absolute step delta <= 2",
            "tolerance_steps": 2,
            "original": original_first["grasp_target"],
            "rerun": rerun_first["grasp_target"],
            "signed_delta_steps": target_delta,
            "absolute_delta_steps": None if target_delta is None else abs(target_delta),
            "passed": target_passed,
        },
    }
    passed = all(value["passed"] for value in comparisons.values())
    original_audit = original["contact_audit"]
    rerun_audit = rerun["contact_audit"]
    pair_deltas = {
        contact_class: {
            "original": int(original_audit["contact_class_totals"][contact_class]),
            "rerun": int(rerun_audit["contact_class_totals"][contact_class]),
            "signed_delta": int(rerun_audit["contact_class_totals"][contact_class])
            - int(original_audit["contact_class_totals"][contact_class]),
        }
        for contact_class in ("grasp_target", "hazard_bar", "other_environment")
    }
    document: dict[str, Any] = {
        "schema_version": "pact_geometry_v3_qualitative_determinism_v1",
        "status": "passed_declared_gate" if passed else "failed_drop_pair",
        "checked_at_utc": utc_now(),
        "gate_declared_manifest_sha256": job["gate_manifest_sha256"],
        "clip_id": job["clip_id"],
        "pair_id": job["pair_id"],
        "condition_id": job["condition_id"],
        "selection_rank": job["selection_rank"],
        "arm": job["arm"],
        "episode_id": job["row"]["instance_episode_id"],
        "checkpoint_seed": job["row"]["checkpoint_seed"],
        "camera_render_only": True,
        "original_result": {"path": str(original_path), "sha256": file_hash(original_path)},
        "rerun_result": {"path": str(rerun_path), "sha256": file_hash(rerun_path)},
        "raw_video": {"path": str(job["raw_video"]), "sha256": file_hash(job["raw_video"])},
        "required_gate_comparisons": comparisons,
        "informational_contact_pair_sample_deltas": pair_deltas,
        "declared_gate_passed": passed,
        "pair_action": "retain_if_mate_passes" if passed else "drop_pair_without_retry",
    }
    write_json(job["check_path"], document, "determinism_check_sha256")
    return document


def ffprobe_video(path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,width,height,avg_frame_rate,nb_frames,duration",
            "-show_entries", "format=duration,size,bit_rate", "-of", "json", str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def compose_release(job: dict[str, Any]) -> dict[str, Any]:
    output = job["release_video"]
    if output.exists():
        raise RuntimeError(f"refusing to replace release clip: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(job["raw_video"]),
            "-vf", "setpts=PTS/3.0", "-an", "-c:v", "libx264", "-preset", "medium",
            "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
        ],
        check=True,
    )
    probe = ffprobe_video(output)
    duration = float(probe["format"]["duration"])
    stream = probe["streams"][0]
    if not 15.0 <= duration <= 25.0:
        raise RuntimeError(f"release duration {duration} outside 15-25 seconds")
    if [int(stream["width"]), int(stream["height"])] != [624, 352]:
        raise RuntimeError("individual release resolution mismatch")
    return probe


def compose_pair(selection: dict[str, Any], jobs: list[dict[str, Any]]) -> tuple[Path, dict[str, Any]]:
    by_arm = {job["arm"]: job for job in jobs}
    output = ARTIFACT_ROOT / "release" / f"{selection['pair_id']}_side_by_side.mp4"
    if output.exists():
        raise RuntimeError(f"refusing to replace paired release: {output}")
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", str(by_arm["PACT"]["release_video"]),
            "-i", str(by_arm["PACT_PERMUTED"]["release_video"]),
            "-filter_complex", "[0:v][1:v]hstack=inputs=2[v]",
            "-map", "[v]", "-an", "-c:v", "libx264", "-preset", "medium",
            "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
        ],
        check=True,
    )
    probe = ffprobe_video(output)
    stream = probe["streams"][0]
    duration = float(probe["format"]["duration"])
    if [int(stream["width"]), int(stream["height"])] != [1248, 352]:
        raise RuntimeError("paired release resolution mismatch")
    if not 15.0 <= duration <= 25.0:
        raise RuntimeError("paired release duration outside 15-25 seconds")
    return output, probe


def first_frame(path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None or frame.shape[:2] != (352, 624):
        raise RuntimeError(f"could not read fixed-pose frame from {path}")
    return frame


def put_wrapped(frame: np.ndarray, lines: list[str], x: int, y: int) -> None:
    for index, line in enumerate(lines):
        cv2.putText(
            frame, line, (x, y + 27 * index), cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (228, 232, 239), 1, cv2.LINE_AA,
        )


def build_geometry_figure(retained: dict[str, list[dict[str, Any]]]) -> tuple[Path, Path]:
    sources = [C0_RAW]
    for condition in ("C2", "Z_093"):
        pact_job = next(job for job in retained[condition] if job["arm"] == "PACT")
        sources.append(pact_job["raw_video"])
    frames = [first_frame(path)[94:, :, :] for path in sources]
    panel_width, image_height = 624, 258
    canvas = np.full((420, panel_width * 3, 3), (22, 26, 34), dtype=np.uint8)
    title = "Held-out obstacle geometry (same camera, arm pose, and scene)"
    cv2.putText(canvas, title, (28, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (255, 255, 255), 2, cv2.LINE_AA)
    labels = [
        ("C0 — baseline", ["aperture 0.85 m | inner face 0.100 m", "panel z 0.89 m"]),
        ("C2 — tighter corridor", ["aperture 0.85 -> 0.70 m", "inner face 0.100 -> 0.070 m"]),
        ("Z_093 — higher panel", ["panel z 0.89 -> 0.93 m", "aperture and inner face unchanged"]),
    ]
    for index, (frame, (heading, detail)) in enumerate(zip(frames, labels)):
        x = index * panel_width
        canvas[58 : 58 + image_height, x : x + panel_width] = frame
        if index:
            cv2.line(canvas, (x, 58), (x, 414), (92, 99, 112), 2)
        cv2.putText(canvas, heading, (x + 16, 346), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        put_wrapped(canvas, detail, x + 16, 375)
    figure_dir = BUNDLE_ROOT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    png = figure_dir / "fig11_geometry_conditions.png"
    svg = figure_dir / "fig11_geometry_conditions.svg"
    if png.exists() or svg.exists():
        raise RuntimeError("refusing to replace geometry figure")
    if not cv2.imwrite(str(png), canvas):
        raise RuntimeError("failed to write geometry PNG")
    encoded = base64.b64encode(png.read_bytes()).decode("ascii")
    svg.write_text(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"1872\" height=\"420\" "
        "viewBox=\"0 0 1872 420\"><title>Held-out geometry conditions C0, C2, and Z_093</title>"
        f"<image width=\"1872\" height=\"420\" href=\"data:image/png;base64,{encoded}\"/></svg>\n"
    )
    return png, svg


def deliver_bundle(
    retained: dict[str, list[dict[str, Any]]],
    checks: dict[str, dict[str, Any]],
    pair_outputs: dict[str, tuple[Path, dict[str, Any]]],
) -> tuple[list[dict[str, Any]], Path, Path, Path]:
    video_dir = BUNDLE_ROOT / "videos/geometry_v3"
    if video_dir.exists():
        raise RuntimeError(f"refusing to replace bundle video directory: {video_dir}")
    video_dir.mkdir(parents=True)
    delivered = []
    names = {
        ("C2", "PACT"): "pairA_c2_pact.mp4",
        ("C2", "PACT_PERMUTED"): "pairA_c2_permuted.mp4",
        ("Z_093", "PACT"): "pairB_z093_pact.mp4",
        ("Z_093", "PACT_PERMUTED"): "pairB_z093_permuted.mp4",
    }
    pair_names = {
        "C2": "pairA_c2_side_by_side.mp4",
        "Z_093": "pairB_z093_side_by_side.mp4",
    }
    for condition, jobs in retained.items():
        for job in jobs:
            target = video_dir / names[(condition, job["arm"])]
            shutil.copy2(job["release_video"], target)
            delivered.append(
                {
                    "condition_id": condition,
                    "arm": job["arm"],
                    "path": str(target),
                    "sha256": file_hash(target),
                    "ffprobe": ffprobe_video(target),
                }
            )
        pair_target = video_dir / pair_names[condition]
        shutil.copy2(pair_outputs[condition][0], pair_target)
        delivered.append(
            {
                "condition_id": condition,
                "arm": "SIDE_BY_SIDE",
                "path": str(pair_target),
                "sha256": file_hash(pair_target),
                "ffprobe": ffprobe_video(pair_target),
            }
        )
    png, svg = build_geometry_figure(retained)
    readme = video_dir / "README.md"
    lines = [
        "# Geometry-v3 matched qualitative clips",
        "",
        "These are presentation examples selected by a frozen rule from the 720-rollout geometry-v3 evaluation. The aggregate evidence remains `PACT_GEOMETRY_GENERALIZATION_V3.md`: pooled −11.7 pp any-contact reduction on held-out geometry, CI [−18.3, −5.4].",
        "",
        "Both sides use identical PACT weights. PACT_PERMUTED replaces aligned proximity with the registered in-distribution temporal permutation. PACT is left and PACT_PERMUTED is right in each synchronized side-by-side file. All clips use the same render-only camera and 3.0× playback.",
        "",
    ]
    for condition in ("C2", "Z_093"):
        jobs = retained[condition]
        selection = jobs[0]["selection"]
        lines.extend(
            [
                f"## {selection['pair_id']} — {condition}",
                "",
                f"Episode `{selection['episode_id']}`, seed {selection['checkpoint_seed']}, intrusion side {selection['intrusion_side']}, selection rank {selection['selection_rank']}.",
                "",
                "Selected among matched instance-seeds where PACT had zero hazard frames and PACT_PERMUTED exceeded 500, ranked by descending PACT_PERMUTED hazard frames.",
                "",
            ]
        )
        for job in jobs:
            outcome = job["arm_record"]["outcome"]
            check = checks[job["clip_id"]]
            target = check["required_gate_comparisons"]["first_grasp_target_contact_step"]
            pairs = check["informational_contact_pair_sample_deltas"]["hazard_bar"]
            lines.append(
                f"- **{job['arm']}**: task success {'yes' if outcome['task_success'] else 'no'}; "
                f"hazard frames {outcome['hazard_contact_frames']:,}; first-target delta "
                f"{target['signed_delta_steps']}; hazard contact-pair delta {pairs['signed_delta']}."
            )
        lines.extend(
            [
                "",
                "> Re-rendered from the analyzed rollout. Task success, manipulation success, and first hazard-contact step reproduce exactly; first target-contact is within the declared ≤2-step tolerance; contact-pair sample deltas are informational.",
                "",
            ]
        )
    lines.extend(
        [
            "## Scope",
            "",
            "These clips illustrate held-out obstacle geometry within one scene, not multiple environments. They are examples, not statistical evidence.",
            "",
        ]
    )
    readme.write_text("\n".join(lines))
    return delivered, png, svg, readme


def verify_protected(manifest: dict[str, Any]) -> None:
    for source_name in ("analysis", "final_decision", "report"):
        source = manifest["sources"][source_name]
        observed = file_hash(Path(source["path"]))
        if observed != source["sha256"]:
            raise RuntimeError(f"protected scientific {source_name} changed after rendering")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="required acknowledgement")
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("pass --execute after reviewing the frozen manifest")
    running = protected_eval_processes()
    if running:
        raise SystemExit(f"other eval processes are active: {running}")
    manifest, schedule = load_frozen()
    frozen_hash = manifest[HASH_KEY]
    ranked = selections_by_rank(manifest)
    attempts: list[dict[str, Any]] = []
    checks: dict[str, dict[str, Any]] = {}
    retained: dict[str, list[dict[str, Any]]] = {}

    rank1_jobs = []
    for condition in ("C2", "Z_093"):
        rank1_jobs.extend(jobs_for_selection(ranked[(condition, 1)], schedule, frozen_hash))
    run_jobs(rank1_jobs)
    for job in rank1_jobs:
        checks[job["clip_id"]] = determinism_check(job)
    for condition in ("C2", "Z_093"):
        condition_jobs = [job for job in rank1_jobs if job["condition_id"] == condition]
        pair_passed = all(checks[job["clip_id"]]["declared_gate_passed"] for job in condition_jobs)
        attempts.append({"condition_id": condition, "rank": 1, "pair_passed": pair_passed, "clip_ids": [job["clip_id"] for job in condition_jobs]})
        if pair_passed:
            retained[condition] = condition_jobs

    for rank in (2, 3):
        pending_conditions = [condition for condition in ("C2", "Z_093") if condition not in retained]
        if not pending_conditions:
            break
        fallback_jobs = []
        for condition in pending_conditions:
            fallback_jobs.extend(jobs_for_selection(ranked[(condition, rank)], schedule, frozen_hash))
        run_jobs(fallback_jobs)
        for job in fallback_jobs:
            checks[job["clip_id"]] = determinism_check(job)
        for condition in pending_conditions:
            condition_jobs = [job for job in fallback_jobs if job["condition_id"] == condition]
            pair_passed = all(checks[job["clip_id"]]["declared_gate_passed"] for job in condition_jobs)
            attempts.append({"condition_id": condition, "rank": rank, "pair_passed": pair_passed, "clip_ids": [job["clip_id"] for job in condition_jobs]})
            if pair_passed:
                retained[condition] = condition_jobs

    if not retained:
        verify_protected(manifest)
        manifest["status"] = "presentation_release_empty_all_pairs_failed_gate"
        manifest["selection_and_gate_frozen_manifest_sha256"] = frozen_hash
        manifest["render_attempts"] = attempts
        manifest["determinism_checks"] = checks
        write_json(MANIFEST_PATH, manifest, HASH_KEY)
        return 2

    releases: dict[str, dict[str, Any]] = {}
    pair_outputs: dict[str, tuple[Path, dict[str, Any]]] = {}
    for condition, jobs in retained.items():
        for job in jobs:
            releases[job["clip_id"]] = compose_release(job)
        pair_outputs[condition] = compose_pair(jobs[0]["selection"], jobs)
    delivered, png, svg, readme = deliver_bundle(retained, checks, pair_outputs)
    verify_protected(manifest)

    manifest["status"] = (
        "presentation_release_verified"
        if set(retained) == {"C2", "Z_093"}
        else "presentation_release_partial_condition_drop"
    )
    manifest["rendered_at_utc"] = utc_now()
    manifest["selection_and_gate_frozen_manifest_sha256"] = frozen_hash
    manifest["render_attempts"] = attempts
    manifest["retained_conditions"] = sorted(retained)
    manifest["determinism_checks"] = {
        clip_id: {
            "path": str(next(job["check_path"] for jobs in [*retained.values()] for job in jobs if job["clip_id"] == clip_id))
            if any(clip_id == job["clip_id"] for jobs in retained.values() for job in jobs)
            else check["raw_video"]["path"].replace("/raw/", "/checks/").replace(".mp4", ".json"),
            "sha256": check["determinism_check_sha256"],
            "declared_gate_passed": check["declared_gate_passed"],
            "required_gate_comparisons": check["required_gate_comparisons"],
            "informational_contact_pair_sample_deltas": check["informational_contact_pair_sample_deltas"],
        }
        for clip_id, check in checks.items()
    }
    manifest["release_outputs"] = {
        "individual_ffprobes": releases,
        "side_by_side": {
            condition: {
                "path": str(output),
                "sha256": file_hash(output),
                "ffprobe": probe,
            }
            for condition, (output, probe) in pair_outputs.items()
        },
        "bundle_files": delivered,
        "geometry_figure": {
            "png": {"path": str(png), "sha256": file_hash(png)},
            "svg": {"path": str(svg), "sha256": file_hash(svg)},
        },
        "readme": {"path": str(readme), "sha256": file_hash(readme)},
    }
    write_json(MANIFEST_PATH, manifest, HASH_KEY)
    print(
        json.dumps(
            {
                "event": "geometry_v3_presentation_release_complete",
                "status": manifest["status"],
                "retained_conditions": sorted(retained),
                "bundle_video_dir": str(BUNDLE_ROOT / "videos/geometry_v3"),
                "figure": str(png),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if set(retained) == {"C2", "Z_093"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
