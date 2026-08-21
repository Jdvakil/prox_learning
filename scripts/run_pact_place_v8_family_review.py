#!/usr/bin/env python3
"""B5 bounded six-family review and forward-only replay rendering.

Run each preregistered family until its first clean success, with at most four
attempts.  This is a human design review, never a gate or a clean-rate estimate.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MOLMO = ROOT / "submodules" / "molmospaces"
for search_path in (ROOT / "scripts", MOLMO):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from pact_place_corridor_contract import (  # noqa: E402
    V8_FAMILIES,
    load_v8_contract,
    sha256_file,
    sha256_payload,
)
from run_pact_place_expert_screen import (  # noqa: E402
    _make_config,
    _protected_eval_processes,
    run_row,
    write_json_atomic,
)

CONFIG_PATH = ROOT / "configs" / "pact_place_corridor_v8.json"
OUTPUT_ROOT = ROOT / "diagnostics_output" / "pact_place_corridor_v8_family_review"
VIDEO_ROOT = OUTPUT_ROOT / "videos"
BASELINE_PATH = ROOT / "diagnostics_output" / "pact_place_v8_baseline" / "analysis.json"
SWEEP_PATH = ROOT / "diagnostics_output" / "pact_place_clutter_sweep_v8" / "analysis.json"
SCENE_XML = (
    MOLMO / "molmo_spaces/data_generation/custom_scenes/pact_place_corridor_v5.xml"
)
LINKS = tuple(f"link{index}" for index in range(1, 7))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _slug(value: str) -> str:
    return "".join(
        char if char.isalnum() or char in "-_" else "_" for char in str(value)
    ).strip("_") or "unknown"


def clip_stem(row: dict[str, Any], result: dict[str, Any]) -> str:
    prefix = f"{row['family']}__attempt{int(row['family_attempt']):02d}"
    if result.get("clean_success"):
        return f"{prefix}_clean_success"
    phase = _slug(str(result.get("terminal_policy_phase") or result.get("status")))
    tracking = result.get("terminal_tracking") or {}
    branch = _slug(
        str(
            tracking.get("check_failure_branch")
            or tracking.get("branch")
            or result.get("error")
            or "unknown"
        )
    )
    return f"{prefix}_FAIL_{phase}_{branch}"


def _row_result_path(row: dict[str, Any]) -> Path:
    return (
        OUTPUT_ROOT
        / "expert_screen_rows"
        / f"{int(row['role_index']):02d}_{row['episode_id'][:16]}"
        / "result.json"
    )


def _run_job(job: dict[str, Any]) -> dict[str, Any]:
    return run_row(
        job["row"],
        config_sha256=job["config_sha256"],
        output_root=str(OUTPUT_ROOT),
        scene_xml=str(SCENE_XML),
    )


def _prepare_v8_task(
    row: dict[str, Any], selected_seed: dict[str, int], cfg: dict[str, Any]
):
    from molmo_spaces.data_generation.pipeline import cleanup_episode_resources
    from molmo_spaces.data_generation.runtime_compat import assert_supported_runtime

    assert_supported_runtime(strict=True)
    scratch = Path(tempfile.mkdtemp(prefix="pact_place_v8_replay_"))
    task = sampler = None
    try:
        scene_xml = ROOT / cfg["scene"]["xml"]
        if scene_xml.name != "pact_place_corridor_v5.xml":
            raise RuntimeError(f"B5 replay got the wrong scene: {scene_xml}")
        config = _make_config(scratch / "dummy.json", scene_xml=scene_xml)
        sampler = config.task_sampler_config.task_sampler_class(config)
        sampler.seed_task_sampling(int(selected_seed["seed_u32"]))
        sampler.set_pact_manifest_row(row)
        task = sampler.sample_task(house_index=int(row["scene_template_house_index"]))
        if task is None:
            raise RuntimeError("sample_task returned None for the recorded seed")
        task.reset()
        scene = task.scene_params
        if scene.get("pact_clutter_added_to_obstacle_aabbs") is not False:
            raise RuntimeError("V8 replay says clutter changed obstacle_aabbs")
        expected = len(row["pact_clutter_layout"]["objects"])
        observed = len(scene.get("pact_clutter_active_body_names") or [])
        if observed != expected:
            raise RuntimeError(f"active clutter count {observed} != manifest {expected}")
        return task, sampler, scratch
    except Exception:
        cleanup_episode_resources(
            task=task,
            policy=None,
            task_sampler=sampler,
            preloaded_policy=None,
            close_task_sampler=sampler is not None,
        )
        shutil.rmtree(scratch, ignore_errors=True)
        raise


def _v8_overlay(base_overlay):
    def overlay(*args, **kwargs):
        frame = base_overlay(*args, **kwargs)
        clearances = kwargs["min_clearance_by_link"]
        cv2.rectangle(frame, (0, 66), (frame.shape[1], 111), (0, 0, 0), -1)
        for row_index, keys in enumerate((LINKS[:3], LINKS[3:])):
            text = "  ".join(
                f"{key} {clearances.get(key, float('nan')):.3f}m" for key in keys
            )
            cv2.putText(
                frame,
                f"clearance  {text}",
                (12, 84 + 21 * row_index),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (235, 235, 235),
                1,
                cv2.LINE_AA,
            )
        return frame

    return overlay


def _render_placeholder(job: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    """Keep the one-clip-per-attempt invariant for a pre-boundary failure."""
    import run_pact_place_v7_replay_videos as renderer

    output = Path(job["video_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        renderer.FPS,
        (renderer.PANE_WH[0] * 2, renderer.PANE_WH[1]),
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not create placeholder clip {output}")
    frame = np.zeros(
        (renderer.PANE_WH[1], renderer.PANE_WH[0] * 2, 3), dtype=np.uint8
    )
    cv2.putText(
        frame,
        f"{job['row']['family']} attempt {job['row']['family_attempt']}: {result['status']}",
        (35, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    reason = str((result.get("retry_history") or [{}])[-1].get("reason", "no trajectory"))
    cv2.putText(
        frame,
        reason[:120],
        (35, 190),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (180, 180, 255),
        1,
        cv2.LINE_AA,
    )
    n_frames = max(1, int(round(renderer.FPS * 2.0)))
    for _ in range(n_frames):
        writer.write(frame)
    writer.release()
    return {
        "role_index": int(job["row"]["role_index"]),
        "episode_id": result["episode_id"],
        "family": job["row"]["family"],
        "family_attempt": int(job["row"]["family_attempt"]),
        "clean_success": False,
        "task_success": False,
        "clip": output.name,
        "placeholder_for_preboundary_failure": True,
        "video_frames": n_frames,
        "video_sha256": sha256_file(output),
        "proximity_metrics": None,
    }


def _render_job(job: dict[str, Any]) -> dict[str, Any]:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("MLSPACES_ASSETS_DIR", str(ROOT / "assets"))
    os.environ.pop("DISPLAY", None)

    import run_pact_place_v7_replay_videos as renderer

    result = json.loads(Path(job["result_path"]).read_text())
    if result.get("status") != "complete" or not Path(job["trajectory_path"]).is_file():
        return _render_placeholder(job, result)

    captured: list[dict[str, float]] = []

    def accumulate(env, *, sensor_cfg, groups, state):
        del sensor_cfg
        model, data = env.current_model, env.current_data
        clutter = [
            (body, gid)
            for body, gid in renderer.active_clutter_geoms(model, data)
            if int(model.geom_contype[gid]) != 0
            or int(model.geom_conaffinity[gid]) != 0
        ]
        clutter_boxes = [renderer.world_aabb_for_geom(model, data, gid) for _, gid in clutter]
        clearances: dict[str, float] = {}
        for key in (*LINKS, "cup"):
            distances = []
            for gid in groups.get(key, []):
                lo, hi = renderer.world_aabb_for_geom(model, data, gid)
                distances.extend(
                    renderer.aabb_distance(lo, hi, clutter_lo, clutter_hi)
                    for clutter_lo, clutter_hi in clutter_boxes
                )
            if distances:
                value = float(min(distances))
                clearances[key] = value
                previous = state["min_clearance"].get(key)
                state["min_clearance"][key] = (
                    value if previous is None else min(previous, value)
                )
        captured.append(clearances)
        return clearances

    renderer._prepare_task = _prepare_v8_task
    renderer.overlay_composite = _v8_overlay(renderer.overlay_composite)
    renderer.accumulate_a0d_metrics = accumulate
    clip = renderer.render_row(job)
    trajectory = json.loads(Path(job["trajectory_path"]).read_text())["steps"]
    if len(captured) != len(trajectory):
        raise RuntimeError("clearance metric count does not match replay trajectory")
    minima = {
        key: min(frame[key] for frame in captured if key in frame)
        for key in (*LINKS, "cup")
        if any(key in frame for frame in captured)
    }
    link_minimum = min(minima[key] for key in LINKS if key in minima)
    minimum_frame = min(
        range(len(captured)),
        key=lambda index: min(
            captured[index].get(key, float("inf")) for key in LINKS
        ),
    )
    proximity = {
        "distance_instrument": "per_geom_world_aabb_to_active_collision_geom_world_aabb",
        "min_clearance_by_link_m": {key: minima.get(key) for key in LINKS},
        "min_link_clearance_m": link_minimum,
        "min_cup_clearance_m": minima.get("cup"),
        "cup_is_closest_body": bool(
            minima.get("cup") is not None and minima["cup"] < link_minimum
        ),
        "frames_link_clearance_lt_5cm": sum(
            min(frame.get(key, float("inf")) for key in LINKS) < 0.05
            for frame in captured
        ),
        "frames_link_clearance_lt_10cm": sum(
            min(frame.get(key, float("inf")) for key in LINKS) < 0.10
            for frame in captured
        ),
        "frames_link_clearance_lt_15cm": sum(
            min(frame.get(key, float("inf")) for key in LINKS) < 0.15
            for frame in captured
        ),
        "n_distinct_links_exposed": sum(minima.get(key, float("inf")) < 0.10 for key in LINKS),
        "phase_of_min_clearance": trajectory[minimum_frame]["policy_phase"],
    }
    clip.pop("a0d_metrics", None)
    clip.update(
        {
            "family": job["row"]["family"],
            "family_attempt": int(job["row"]["family_attempt"]),
            "layout_id": job["row"]["layout_id"],
            "placeholder_for_preboundary_failure": False,
            "proximity_metrics": proximity,
            "admission_reference_metrics": job["row"]["pact_clutter_layout"]["score"],
        }
    )
    return clip


def _selected_aggregate(sweep: dict[str, Any]) -> dict[str, Any]:
    selected = list(sweep["selected_layouts"])
    scores = [item["score"] for item in selected]
    return {
        "n_episodes": len(scores),
        "frames_link_clearance_lt_5cm": sum(
            int(score["frames_link_clearance_lt_5cm"]) for score in scores
        ),
        "frames_link_clearance_lt_10cm": sum(
            int(score["frames_link_clearance_lt_10cm"]) for score in scores
        ),
        "frames_link_clearance_lt_15cm": sum(
            int(score["frames_link_clearance_lt_15cm"]) for score in scores
        ),
        "mean_distinct_links_exposed": float(
            np.mean([score["n_distinct_links_exposed"] for score in scores])
        ),
        "n_episodes_cup_is_closest_body": sum(
            bool(score["cup_is_closest_body"]) for score in scores
        ),
        "cup_is_closest_body_fraction": float(
            np.mean([bool(score["cup_is_closest_body"]) for score in scores])
        ),
        "visibility_at_min_link_clearance_fraction": float(
            np.mean([bool(score["visibility_at_min_link_clearance"]) for score in scores])
        ),
        "clutter_visible_frame_fraction": float(
            np.mean([score["clutter_visible_frame_fraction"] for score in scores])
        ),
        "target_visible_frame_fraction": float(
            np.mean([score["target_visible_frame_fraction"] for score in scores])
        ),
    }


def _write_hashed(path: Path, document: dict[str, Any], hash_key: str) -> None:
    payload = dict(document)
    payload.pop(hash_key, None)
    document[hash_key] = sha256_payload(payload)
    write_json_atomic(path, document)


def _write_progress(
    contract: dict[str, Any], attempts: list[dict[str, Any]], solved: set[str]
) -> None:
    document = {
        "schema_version": "pact_place_v8_family_review_progress_v1",
        "role": "human_design_review_not_a_gate",
        "authorizes_gate": False,
        "authorizes_collection": False,
        "config_sha256": contract["config_sha256"],
        "solved_families": sorted(solved),
        "n_attempts": len(attempts),
        "attempts": attempts,
    }
    _write_hashed(OUTPUT_ROOT / "review_progress.json", document, "progress_sha256")


def run_review(workers: int, render_workers: int) -> dict[str, Any]:
    contract = load_v8_contract(CONFIG_PATH)
    if contract.get("authorizes_gate") is not False:
        raise RuntimeError("B5 config unexpectedly authorizes the gate")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows_by_family_attempt = {
        (row["family"], int(row["family_attempt"])): row
        for row in contract["family_review_rows"]
    }
    attempts: list[dict[str, Any]] = []
    solved: set[str] = set()
    context = multiprocessing.get_context("spawn")
    for attempt_number in range(1, 5):
        jobs = [
            {
                "row": rows_by_family_attempt[(family, attempt_number)],
                "config_sha256": contract["config_sha256"],
            }
            for family in V8_FAMILIES
            if family not in solved
        ]
        if not jobs:
            break
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=min(workers, len(jobs)),
            mp_context=context,
            max_tasks_per_child=1,
        ) as executor:
            future_jobs = {executor.submit(_run_job, job): job for job in jobs}
            round_results = []
            for future in concurrent.futures.as_completed(future_jobs):
                job = future_jobs[future]
                result = future.result()
                round_results.append((job["row"], result))
        round_results.sort(key=lambda item: int(item[0]["role_index"]))
        for row, result in round_results:
            record = {
                "family": row["family"],
                "family_attempt": int(row["family_attempt"]),
                "layout_id": row["layout_id"],
                "role_index": int(row["role_index"]),
                "episode_id": row["episode_id"],
                "result_path": str(_row_result_path(row).relative_to(ROOT)),
                "status": result["status"],
                "task_success": bool(result.get("task_success")),
                "clean_success": bool(result.get("clean_success")),
                "terminal_policy_phase": result.get("terminal_policy_phase"),
                "failure_branch": (result.get("terminal_tracking") or {}).get(
                    "check_failure_branch"
                ),
            }
            attempts.append(record)
            if record["clean_success"]:
                solved.add(str(row["family"]))
            print(
                f"family={row['family']} attempt={row['family_attempt']} "
                f"status={result['status']} clean={bool(result.get('clean_success'))}",
                flush=True,
            )
        _write_progress(contract, attempts, solved)

    VIDEO_ROOT.mkdir(parents=True, exist_ok=True)
    render_jobs = []
    rows_by_index = {int(row["role_index"]): row for row in contract["family_review_rows"]}
    for attempt in attempts:
        row = rows_by_index[int(attempt["role_index"])]
        result_path = _row_result_path(row)
        result = json.loads(result_path.read_text())
        render_jobs.append(
            {
                "role_index": int(row["role_index"]),
                "row": row,
                "config_path": str(CONFIG_PATH),
                "result_path": str(result_path),
                "trajectory_path": str(result_path.parent / "trajectory.json"),
                "video_path": str(VIDEO_ROOT / f"{clip_stem(row, result)}.mp4"),
            }
        )
    clips: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=min(render_workers, max(1, len(render_jobs))),
        mp_context=context,
        max_tasks_per_child=1,
    ) as executor:
        future_jobs = {executor.submit(_render_job, job): job for job in render_jobs}
        for future in concurrent.futures.as_completed(future_jobs):
            job = future_jobs[future]
            clip = future.result()
            clips.append(clip)
            print(f"rendered {clip['clip']}", flush=True)
    clips.sort(key=lambda item: int(item["role_index"]))

    attempts_needed = {
        family: sum(item["family"] == family for item in attempts)
        for family in V8_FAMILIES
    }
    failure_branches = [
        {
            "family": item["family"],
            "family_attempt": item["family_attempt"],
            "phase": item["terminal_policy_phase"],
            "branch": item["failure_branch"],
            "status": item["status"],
        }
        for item in attempts
        if not item["clean_success"]
    ]
    baseline = json.loads(BASELINE_PATH.read_text())["aggregates"]["v6c"]
    selected = _selected_aggregate(json.loads(SWEEP_PATH.read_text()))
    manifest = {
        "schema_version": "pact_place_v8_family_review_replay_v1",
        "created_utc": utc_now(),
        "role": "human_design_review_not_a_gate",
        "authorizes_gate": False,
        "authorizes_collection": False,
        "config_sha256": contract["config_sha256"],
        "replay_only": True,
        "physics_stepped": False,
        "expert_rerun_during_render": False,
        "rendered_every_attempt": len(clips) == len(attempts),
        "n": len(clips),
        "clips": clips,
        "overlay_fields": [
            "current policy phase",
            "running clutter contact frames",
            "per-link current minimum clutter distance link1-link6",
        ],
    }
    _write_hashed(VIDEO_ROOT / "manifest.json", manifest, "manifest_sha256")
    report = {
        "schema_version": "pact_place_v8_family_review_v1",
        "created_utc": utc_now(),
        "role": "human_design_review_not_a_gate",
        "authorizes_gate": False,
        "authorizes_collection": False,
        "mandatory_stop_after_this_report": True,
        "gate_executed": False,
        "collection_executed": False,
        "config_sha256": contract["config_sha256"],
        "family_review_seed_stream": contract["family_review"]["stream"],
        "gate_seed_stream": contract["phase0_gate"]["stream"],
        "review_gate_episode_overlap": sorted(
            {row["episode_id"] for row in contract["family_review_rows"]}
            & {row["episode_id"] for row in contract["expert_screen_rows"]}
        ),
        "stopping_rule": "first clean success per family, maximum four attempts",
        "clean_rate_is_not_an_estimate": True,
        "n_attempts": len(attempts),
        "n_clips": len(clips),
        "rendered_every_attempt": len(clips) == len(attempts),
        "families_with_clean_success": sorted(solved),
        "families_without_clean_success_after_four": [
            family for family in V8_FAMILIES if family not in solved
        ],
        "attempts_needed_per_family": attempts_needed,
        "failure_branches": failure_branches,
        "attempts": attempts,
        "metric_table_vs_v6c": {
            "instrument": "B0 replay for v6c and B2 frozen-layout replay for v8",
            "v6c": baseline,
            "v8_selected_layouts": selected,
            "min_pairwise_selected_layout_distance": json.loads(
                SWEEP_PATH.read_text()
            )["min_pairwise_selected_layout_distance"],
        },
        "scoring_check_path": str(
            (
                ROOT
                / "diagnostics_output/pact_place_corridor_v8_scoring_check/scoring_check.json"
            ).relative_to(ROOT)
        ),
        "videos_manifest_path": str((VIDEO_ROOT / "manifest.json").relative_to(ROOT)),
    }
    _write_hashed(OUTPUT_ROOT / "family_review.json", report, "family_review_sha256")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--render-workers", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4 or not 1 <= args.render_workers <= 4:
        raise SystemExit("worker counts must be in [1, 4]")
    protected = _protected_eval_processes()
    if protected:
        raise SystemExit(f"protected confirmatory evaluation is active: {protected}")
    report = run_review(args.workers, args.render_workers)
    print(
        json.dumps(
            {
                "n_attempts": report["n_attempts"],
                "attempts_needed_per_family": report["attempts_needed_per_family"],
                "families_without_clean_success_after_four": report[
                    "families_without_clean_success_after_four"
                ],
                "stopped_before_B6": True,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
