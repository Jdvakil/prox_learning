#!/usr/bin/env python3
"""Render, gate, and deliver the frozen PACT blur-media release."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/root/act_retrain_venv/bin/python")
EVALUATOR = ROOT / "submodules/act/eval_pact_blur_media_row.py"
MEDIA_MANIFEST = ROOT / "diagnostics_output/pact_blur_sweep/media_manifest.json"
SCIENTIFIC_MANIFEST = ROOT / "configs/pact_blur_sweep_v1.json"
SCHEDULE = ROOT / "diagnostics_output/pact_blur_sweep/schedule.json"
ARTIFACT_ROOT = Path("/root/pact_blur_sweep_artifacts/media_v1")
BUNDLE = Path("/root/pact_slideshow_bundle")
BUNDLE_MANIFEST = BUNDLE / "BUNDLE_MANIFEST.json"
REPO_BUNDLE_MANIFEST = ROOT / "diagnostics_output/pact_slideshow_bundle_manifest.json"
HASH_KEY = "media_manifest_sha256"
WORKERS = 4


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


def validate_self_hash(document: dict[str, Any], key: str, label: str) -> str:
    payload = dict(document)
    observed = payload.pop(key, None)
    expected = canonical_hash(payload)
    if observed != expected:
        raise RuntimeError(f"{label} self-hash mismatch: {observed} != {expected}")
    return str(observed)


def atomic_json(path: Path, document: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


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
            "OMP_NUM_THREADS": "8",
        }
    )
    env.pop("DISPLAY", None)
    return env


def sigma_stem(sigma: float) -> str:
    return f"{sigma:.1f}".replace(".", "p")


def load_frozen() -> tuple[dict[str, Any], dict[str, Any], str]:
    manifest = json.loads(MEDIA_MANIFEST.read_text())
    frozen_hash = validate_self_hash(manifest, HASH_KEY, "blur media manifest")
    if manifest.get("status") != "selection_and_gate_frozen_pre_render":
        raise RuntimeError("blur media manifest is not frozen pre-render")
    schedule = json.loads(SCHEDULE.read_text())
    validate_self_hash(schedule, "schedule_sha256", "blur schedule")
    if manifest["sources"]["schedule"]["sha256"] != file_hash(SCHEDULE):
        raise RuntimeError("frozen media selection references a different schedule")
    for path, expected in manifest["sources"]["protected_scientific_artifacts"].items():
        if file_hash(Path(path)) != expected:
            raise RuntimeError(f"protected scientific artifact changed: {path}")
    return manifest, schedule, frozen_hash


def jobs_from_manifest(
    manifest: dict[str, Any], schedule: dict[str, Any], frozen_hash: str
) -> list[dict[str, Any]]:
    schedule_rows = {int(row["schedule_index"]): row for row in schedule["rows"]}
    retained = manifest["visual_contract"]["retained_detail_percent_by_sigma"]
    jobs = []
    for record in manifest["selection"]["rows"]:
        sigma = float(record["sigma"])
        row = schedule_rows[int(record["schedule_index"])]
        expected = {
            "arm": "PACT",
            "instance_episode_id": manifest["selection"]["episode_id"],
            "checkpoint_seed": manifest["selection"]["checkpoint_seed"],
            "rollout_id": record["rollout_id"],
            "schedule_row_sha256": record["schedule_row_sha256"],
            "checkpoint_sha256": record["checkpoint_sha256"],
            "blur_sigma": sigma,
        }
        if {key: row.get(key) for key in expected} != expected:
            raise RuntimeError(f"media manifest/schedule mismatch at sigma {sigma}")
        stem = f"sigma_{sigma_stem(sigma)}_pact"
        jobs.append(
            {
                "sigma": sigma,
                "row": row,
                "record": record,
                "gate_manifest_sha256": frozen_hash,
                "output_dir": ARTIFACT_ROOT / "reruns" / stem,
                "raw_video": ARTIFACT_ROOT / "raw" / f"{stem}.mp4",
                "release_video": ARTIFACT_ROOT / "release" / f"{stem}.mp4",
                "check_path": ARTIFACT_ROOT / "checks" / f"{stem}.json",
                "log_path": ARTIFACT_ROOT / "logs" / f"{stem}.log",
                "retained_detail_percent": float(retained[str(sigma)]),
            }
        )
    if [job["sigma"] for job in jobs] != [0.0, 0.5, 1.0, 2.0]:
        raise RuntimeError("blur-media sigma order changed")
    return jobs


def command_for(job: dict[str, Any]) -> list[str]:
    row = job["row"]
    record = job["record"]
    return [
        str(PYTHON),
        str(EVALUATOR),
        "--arm",
        "PACT",
        "--episode-id",
        row["instance_episode_id"],
        "--manifest",
        str(SCIENTIFIC_MANIFEST),
        "--checkpoint-dir",
        str(Path(row["checkpoint_path"]).parent),
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
        "--blur-sigma",
        str(job["sigma"]),
        "--media-video-output",
        str(job["raw_video"]),
        "--media-episode-id",
        row["instance_episode_id"],
        "--media-policy-seed",
        str(row["checkpoint_seed"]),
        "--media-task-success",
        "yes" if record["outcome"]["task_success"] else "no",
        "--media-retained-detail-percent",
        str(job["retained_detail_percent"]),
        "--media-playback-speed-factor",
        "3.0",
    ]


def prepare_jobs(jobs: list[dict[str, Any]]) -> None:
    for job in jobs:
        targets = (
            job["output_dir"],
            job["raw_video"],
            job["release_video"],
            job["check_path"],
            job["log_path"],
        )
        existing = [str(path) for path in targets if path.exists()]
        if existing:
            raise RuntimeError(f"refusing to replace blur-media outputs: {existing}")
        job["output_dir"].mkdir(parents=True)
        job["raw_video"].parent.mkdir(parents=True, exist_ok=True)
        job["log_path"].parent.mkdir(parents=True, exist_ok=True)


def run_jobs(jobs: list[dict[str, Any]]) -> None:
    prepare_jobs(jobs)
    active: list[tuple[dict[str, Any], subprocess.Popen[Any], Any]] = []
    for job in jobs:
        log = job["log_path"].open("w")
        process = subprocess.Popen(
            command_for(job),
            cwd=ROOT / "submodules/act",
            env=runtime_env(),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        active.append((job, process, log))
        print(json.dumps({"event": "launched", "sigma": job["sigma"], "pid": process.pid}), flush=True)
    failure = None
    while active:
        for item in list(active):
            job, process, log = item
            returncode = process.poll()
            if returncode is None:
                continue
            log.close()
            active.remove(item)
            if returncode != 0:
                failure = f"sigma {job['sigma']} exited {returncode}; see {job['log_path']}"
            else:
                print(json.dumps({"event": "completed", "sigma": job["sigma"]}), flush=True)
        if failure:
            for _job, process, log in active:
                if process.poll() is None:
                    process.terminate()
                log.close()
            raise RuntimeError(failure)
        time.sleep(0.5)


def validate_completed(job: dict[str, Any]) -> dict[str, Any]:
    result_path = job["output_dir"] / "result.json"
    if not result_path.exists() or not job["raw_video"].exists():
        raise RuntimeError(f"incomplete blur-media output at sigma {job['sigma']}")
    result = json.loads(result_path.read_text())
    row = job["row"]
    expected = {
        "status": "complete",
        "arm": "PACT",
        "episode_id": row["instance_episode_id"],
        "checkpoint_seed": row["checkpoint_seed"],
        "rollout_id": row["rollout_id"],
        "schedule_row_sha256": row["schedule_row_sha256"],
        "checkpoint_sha256": row["checkpoint_sha256"],
        "blur_sigma": job["sigma"],
    }
    if {key: result.get(key) for key in expected} != expected:
        raise RuntimeError(f"blur-media result identity mismatch at sigma {job['sigma']}")
    render = result.get("policy_info", {}).get("blur_media_render", {})
    if (
        render.get("schema_version") != "pact_blur_media_render_v1"
        or render.get("render_only_third_person_camera") is not True
        or render.get("third_person_camera_registered_in_observation") is not False
        or render.get("policy_camera_names") != ["wrist_camera"]
        or "exo_camera_1" in (render.get("observation_keys") or [])
        or int(render.get("video_frames", -1)) != 900
        or render.get("resolution_width_height") != [1248, 352]
        or float(render.get("playback_speed_factor", 0.0)) != 3.0
        or float(render.get("sigma", -1.0)) != job["sigma"]
        or abs(float(render.get("retained_detail_percent", -1.0)) - job["retained_detail_percent"]) > 1e-9
        or render.get("rng_calls_added") != 0
    ):
        raise RuntimeError(f"blur-media render contract failed at sigma {job['sigma']}: {render}")
    return result


def first_target_comparison(original: Any, rerun: Any) -> dict[str, Any]:
    if original is None and rerun is None:
        delta, passed = 0, True
    elif original is None or rerun is None:
        delta, passed = None, False
    else:
        delta = int(rerun) - int(original)
        passed = abs(delta) <= 2
    return {
        "requirement": "absolute step delta <= 2",
        "tolerance_steps": 2,
        "original": original,
        "rerun": rerun,
        "signed_delta_steps": delta,
        "absolute_delta_steps": None if delta is None else abs(delta),
        "passed": passed,
    }


def determinism_check(job: dict[str, Any]) -> dict[str, Any]:
    original_path = Path(job["record"]["original_result_path"])
    rerun_path = job["output_dir"] / "result.json"
    original = json.loads(original_path.read_text())
    rerun = json.loads(rerun_path.read_text())
    original_first = original["contact_audit"]["first_contact_step"]
    rerun_first = rerun["contact_audit"]["first_contact_step"]
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
        "first_grasp_target_contact_step": first_target_comparison(
            original_first["grasp_target"], rerun_first["grasp_target"]
        ),
    }
    passed = all(record["passed"] for record in comparisons.values())
    original_audit = original["contact_audit"]
    rerun_audit = rerun["contact_audit"]
    document: dict[str, Any] = {
        "schema_version": "pact_blur_media_determinism_v1",
        "status": "passed_declared_gate" if passed else "failed_drop_clip",
        "checked_at_utc": utc_now(),
        "gate_declared_manifest_sha256": job["gate_manifest_sha256"],
        "sigma": job["sigma"],
        "episode_id": job["row"]["instance_episode_id"],
        "checkpoint_seed": job["row"]["checkpoint_seed"],
        "required_gate_comparisons": comparisons,
        "informational_contact_pair_counts": {
            "sample_count": {
                "original": int(original_audit["sample_count"]),
                "rerun": int(rerun_audit["sample_count"]),
                "signed_delta": int(rerun_audit["sample_count"]) - int(original_audit["sample_count"]),
            },
            "contact_class_totals": {
                contact_class: {
                    "original": int(original_audit["contact_class_totals"][contact_class]),
                    "rerun": int(rerun_audit["contact_class_totals"][contact_class]),
                    "signed_delta": int(rerun_audit["contact_class_totals"][contact_class])
                    - int(original_audit["contact_class_totals"][contact_class]),
                }
                for contact_class in ("grasp_target", "hazard_bar", "other_environment")
            },
        },
        "original_result": {"path": str(original_path), "sha256": file_hash(original_path)},
        "rerun_result": {"path": str(rerun_path), "sha256": file_hash(rerun_path)},
        "raw_video": {"path": str(job["raw_video"]), "sha256": file_hash(job["raw_video"])},
        "declared_gate_passed": passed,
        "release_action": "retain_clip" if passed else "drop_clip_without_retry",
    }
    document["determinism_check_sha256"] = canonical_hash(document)
    job["check_path"].parent.mkdir(parents=True, exist_ok=True)
    atomic_json(job["check_path"], document)
    return document


def ffprobe(path: Path) -> dict[str, Any]:
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
    probe = ffprobe(output)
    stream = probe["streams"][0]
    duration = float(probe["format"]["duration"])
    if [int(stream["width"]), int(stream["height"])] != [1248, 352]:
        raise RuntimeError("blur release resolution mismatch")
    if not 15.0 <= duration <= 25.0:
        raise RuntimeError(f"blur release duration {duration} outside 15-25 seconds")
    return probe


def update_bundle_docs() -> None:
    index = BUNDLE / "INDEX.md"
    text = index.read_text()
    if "fig_blur_sigma_panel" not in text:
        anchor = "Optional Figure 10 (projection magnitudes)"
        addition = (
            "| `figures/fig_blur_sigma_panel.{png,svg}` | Exact wrist input at sigma 0, 0.5, 1, and 2 | Blur progressively removes RGB detail; it did not widen PACT task advantage | `data/blur_calibration.json`; `data/blur_analysis.json` | Blur is OOD; sigma 2 is near collapse |\n"
            "| `figures/fig_blur_sigma_0p5.{png,svg}` | Policy wrist input at mild blur | Sigma 0.5 retains 46.2% of sharp detail | `data/blur_calibration.json` | Input illustration, not outcome evidence |\n\n"
        )
        text = text.replace(anchor, addition + anchor)
    if "videos/blur_sweep/sigma_0p0_pact.mp4" not in text:
        anchor = "## One-page text summary"
        rows = "\n".join(
            [
                f"| `videos/blur_sweep/sigma_{stem}_pact.mp4` | Same PACT instance at sigma {sigma} | Two-pane policy input and resulting behavior | `data/blur_media_manifest.json` | Selected illustration; videos are not evidence |"
                for stem, sigma in (("0p0", "0"), ("0p5", "0.5"), ("1p0", "1"), ("2p0", "2"))
            ]
        )
        caveat = (
            "\n\nBlur result: `NO_BLUR_ROBUSTNESS`. The PACT-minus-ACT task-success advantage did not widen with blur (within-instance slope -5.3 pp per sigma, 95% CI [-12.6, +1.4]), while PACT-minus-ACT hazard frames remained lower at every sigma (-2,654 / -2,661 / -3,351 / -1,759; all intervals exclude zero). Sigma 2 is near collapse and seed behavior diverges; blur was OOD for every arm.\n\n"
        )
        text = text.replace(anchor, rows + caveat + anchor)
    index.write_text(text)

    key_numbers = BUNDLE / "KEY_NUMBERS.md"
    text = key_numbers.read_text()
    if "## Inference-time RGB blur" not in text:
        text += (
            "\n## Inference-time RGB blur\n\n"
            "- Frozen decision: **NO_BLUR_ROBUSTNESS**. Source: `data/blur_final_decision.json`.\n"
            "- PACT-minus-ACT collision-free success by sigma 0 / 0.5 / 1 / 2: **+10.7 / +12.0 / +5.3 / +1.3 pp**; every positive-sigma 95% interval includes zero. Source: `data/blur_analysis.json -> contrasts_per_sigma`.\n"
            "- Within-instance PACT-minus-ACT task-success slope: **-5.3 pp per sigma**, 95% CI **[-12.6, +1.4]**. Source: `data/blur_analysis.json -> within_instance_linear_slopes_per_sigma_unit.collision_free_task_success.interactions.PACT_minus_ACT.pooled`.\n"
            "- PACT-minus-ACT hazard frames at sigma 0 / 0.5 / 1 / 2: **-2,654 / -2,661 / -3,351 / -1,759**; all 95% intervals exclude zero. Same analysis source.\n"
            "- Calibration retained sharp detail at sigma 0 / 0.5 / 1 / 2: **100.0% / 46.2% / 9.3% / 1.7%**. Source: `data/blur_calibration.json -> measurements`.\n"
            "- **Required caveats:** sigma 2 is near the collapse floor and seeds diverge (PACT seed 3102 0/25, seed 3103 10/25); blur was out of distribution for every arm. This is unexpected-degradation robustness, not blur-trained performance.\n"
        )
    key_numbers.write_text(text)


def copy_bundle_sources(manifest: dict[str, Any]) -> None:
    data = BUNDLE / "data"
    reports = BUNDLE / "reports"
    data.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    copies = {
        data / "blur_calibration.json": ROOT / "diagnostics_output/pact_blur_sweep/calibration.json",
        data / "blur_analysis.json": ROOT / "diagnostics_output/pact_blur_sweep/analysis.json",
        data / "blur_final_decision.json": ROOT / "diagnostics_output/pact_blur_sweep/final_decision.json",
        data / "blur_media_manifest.json": MEDIA_MANIFEST,
        reports / "PACT_BLUR_SWEEP.md": ROOT / "docs/PACT_BLUR_SWEEP.md",
    }
    for target, source in copies.items():
        shutil.copy2(source, target)
        if file_hash(target) != file_hash(source):
            raise RuntimeError(f"bundle source copy mismatch: {target}")


def bundle_video_records() -> dict[str, Any]:
    records = {}
    for path in sorted(BUNDLE.glob("videos/**/*.mp4")):
        key = str(path.relative_to(BUNDLE).with_suffix("")).replace("/", "__")
        records[key] = {
            "path": str(path.relative_to(BUNDLE)),
            "sha256": file_hash(path),
            "size_bytes": path.stat().st_size,
            "video": ffprobe(path)["streams"][0],
        }
    return records


def refresh_bundle_manifest() -> dict[str, Any]:
    existing = json.loads(BUNDLE_MANIFEST.read_text())
    entries = []
    for path in sorted(
        item for item in BUNDLE.rglob("*")
        if item.is_file() and item.name != "BUNDLE_MANIFEST.json"
    ):
        entries.append(
            {
                "path": str(path.relative_to(BUNDLE)),
                "sha256": file_hash(path),
                "size_bytes": path.stat().st_size,
            }
        )
    extension_counts: dict[str, int] = defaultdict(int)
    for entry in entries:
        extension_counts[Path(entry["path"]).suffix.lower() or "no_extension"] += 1
    figure_stems = {
        str(Path(entry["path"]).with_suffix(""))
        for entry in entries if entry["path"].startswith("figures/")
    }
    document = dict(existing)
    document.pop("bundle_manifest_sha256", None)
    document.update(
        {
            "scientific_artifacts_modified": False,
            "gpu_work_performed": True,
            "rollouts_or_training_performed": True,
            "presentation_only_rollout_rerenders": 17,
            "training_performed": False,
            "figure_concepts": len(figure_stems),
            "figure_files": sum(1 for entry in entries if entry["path"].startswith("figures/")),
            "video_files": sum(1 for entry in entries if entry["path"].endswith(".mp4")),
            "blur_media_individual_clip_files": 4,
            "blur_media_figure_concepts": 2,
            "total_payload_size_bytes_excluding_manifest": sum(entry["size_bytes"] for entry in entries),
            "extension_counts": dict(sorted(extension_counts.items())),
            "video_records": bundle_video_records(),
            "entries": entries,
        }
    )
    document["bundle_manifest_sha256"] = canonical_hash(document)
    atomic_json(BUNDLE_MANIFEST, document)
    atomic_json(REPO_BUNDLE_MANIFEST, document)
    return document


def verify_protected(manifest: dict[str, Any]) -> None:
    for path, expected in manifest["sources"]["protected_scientific_artifacts"].items():
        if file_hash(Path(path)) != expected:
            raise RuntimeError(f"protected scientific artifact changed after render: {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("pass --execute after reviewing the frozen blur-media manifest")
    active = protected_eval_processes()
    if active:
        raise SystemExit(f"other eval processes are active: {active}")
    manifest, schedule, frozen_hash = load_frozen()
    jobs = jobs_from_manifest(manifest, schedule, frozen_hash)
    run_jobs(jobs)
    checks = {}
    releases = {}
    video_dir = BUNDLE / "videos/blur_sweep"
    if video_dir.exists():
        raise RuntimeError(f"refusing to replace bundle blur video directory: {video_dir}")
    video_dir.mkdir(parents=True)
    for job in jobs:
        validate_completed(job)
        check = determinism_check(job)
        checks[str(job["sigma"])] = check
        if not check["declared_gate_passed"]:
            continue
        probe = compose_release(job)
        target = video_dir / f"sigma_{sigma_stem(job['sigma'])}_pact.mp4"
        shutil.copy2(job["release_video"], target)
        releases[str(job["sigma"])] = {
            "path": str(target),
            "sha256": file_hash(target),
            "ffprobe": probe,
        }
    verify_protected(manifest)
    manifest["status"] = (
        "presentation_release_verified" if len(releases) == 4 else "partial_gate_drop"
    )
    manifest["rendered_at_utc"] = utc_now()
    manifest["determinism_checks"] = {
        sigma: {
            "path": str(next(job["check_path"] for job in jobs if str(job["sigma"]) == sigma)),
            "sha256": check["determinism_check_sha256"],
            "declared_gate_passed": check["declared_gate_passed"],
            "required_gate_comparisons": check["required_gate_comparisons"],
            "informational_contact_pair_counts": check["informational_contact_pair_counts"],
        }
        for sigma, check in checks.items()
    }
    manifest["release_videos"] = releases
    manifest["scientific_record_changed"] = False
    manifest.pop(HASH_KEY, None)
    manifest[HASH_KEY] = canonical_hash(manifest)
    atomic_json(MEDIA_MANIFEST, manifest)
    update_bundle_docs()
    copy_bundle_sources(manifest)
    bundle = refresh_bundle_manifest()
    verify_protected(manifest)
    print(
        json.dumps(
            {
                "event": "blur_media_release_complete",
                "status": manifest["status"],
                "released_sigmas": sorted(releases),
                "media_manifest_sha256": manifest[HASH_KEY],
                "bundle_manifest_sha256": bundle["bundle_manifest_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if len(releases) == 4 else 2


if __name__ == "__main__":
    raise SystemExit(main())
