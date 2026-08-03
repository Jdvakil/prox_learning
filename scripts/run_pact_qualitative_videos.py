#!/usr/bin/env python3
"""Run the frozen qualitative determinism probe and paired render reruns."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
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
VIDEO_ROOT = Path("/root/pact_contact_endpoint_artifacts/qualitative_videos")
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("probe", "remaining"))
    parser.add_argument("--workers", type=int, default=WORKERS)
    args = parser.parse_args()
    if args.workers != WORKERS:
        raise SystemExit(f"qualitative worker count is frozen at {WORKERS}")
    protected = protected_eval_processes()
    if protected:
        raise SystemExit(f"protected shared evaluation is active: {protected}")
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
