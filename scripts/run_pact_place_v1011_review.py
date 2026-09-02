#!/usr/bin/env python3
"""Generate and publish the V10.11 six-video owner review packet."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _path in (
    ROOT / "scripts",
    ROOT / "submodules" / "molmospaces",
    ROOT / "submodules" / "act",
):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

CONTRACT_MODULE_NAME = os.environ.get(
    "PACT_PLACE_V1011_CONTRACT_MODULE", "pact_place_v1011_contract"
)
_contract = importlib.import_module(CONTRACT_MODULE_NAME)
CONTRACT_VERSION = _contract.CONTRACT_VERSION
DISPLAY_VERSION = getattr(_contract, "DISPLAY_VERSION", "V10.11")
ENVIRONMENT_VERSION = _contract.ENVIRONMENT_VERSION
PREFLIGHT_ROOT = _contract.PREFLIGHT_ROOT
REVIEW_FAILURES = _contract.REVIEW_FAILURES
REVIEW_MAX_ATTEMPTS_PER_CELL = _contract.REVIEW_MAX_ATTEMPTS_PER_CELL
REVIEW_ROOT = _contract.REVIEW_ROOT
REVIEW_SUCCESSES = _contract.REVIEW_SUCCESSES
VISIBILITY_ROOT = getattr(_contract, "VISIBILITY_ROOT", None)
build_contract = _contract.build_contract
build_row = _contract.build_row
canonical_payload_sha256 = _contract.canonical_payload_sha256
cells = _contract.cells
empty_authorization = _contract.empty_authorization
sha256_payload = _contract.sha256_payload
write_immutable_create_only = _contract.write_immutable_create_only
from pact_place_v105_contract import recompute_payload_sha256, sha256_file  # noqa: E402

FPS = 1000.0 / 66.0
PANE_WH = (720, 405)
AIM = np.asarray([0.664, 0.006, 0.82], dtype=float)
VIEWS = {
    "table_left_high": (210.0, 45.0, 1.35, 46.0),
    "table_left_mid": (225.0, 35.0, 1.35, 46.0),
}


def camera_pose(aim, azimuth_deg, elevation_deg, distance_m):
    azimuth, elevation = np.radians([azimuth_deg, elevation_deg])
    position = np.asarray(aim, dtype=float) + distance_m * np.asarray(
        [
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        ]
    )
    forward = np.asarray(aim, dtype=float) - position
    forward /= np.linalg.norm(forward)
    up_hint = np.asarray([0.0, 0.0, 1.0])
    right = np.cross(forward, up_hint)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)
    return position, forward, up


def _implementation_binding() -> dict[str, str]:
    paths = [
        Path(_contract.__file__).resolve(),
        ROOT / "scripts/pact_place_v1011_contract.py",
        ROOT / "scripts/run_pact_place_v1011_preflight.py",
        ROOT / "scripts/run_pact_place_v1011_review.py",
        ROOT / "scripts/run_pact_place_expert_screen.py",
        ROOT / "submodules/molmospaces/molmo_spaces/tasks/enclosure_reach.py",
    ]
    entrypoint = os.environ.get("PACT_PLACE_V1011_REVIEW_ENTRYPOINT")
    if entrypoint:
        paths.append(ROOT / entrypoint)
    visibility_runner = ROOT / getattr(
        _contract,
        "VISIBILITY_RUNNER_RELATIVE",
        "scripts/run_pact_place_v1011b_visibility.py",
    )
    if VISIBILITY_ROOT and visibility_runner.is_file():
        paths.append(visibility_runner)
        paths.append(
            ROOT
            / getattr(
                _contract,
                "PIPELINE_RUNNER_RELATIVE",
                "scripts/run_pact_place_v1011b_pipeline.py",
            )
        )
    paths = list(dict.fromkeys(path.resolve() for path in paths))
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in paths}


def frozen_config() -> dict[str, Any]:
    preflight_path = ROOT / PREFLIGHT_ROOT / "preflight.json"
    if not preflight_path.is_file():
        raise SystemExit(f"{DISPLAY_VERSION} preflight artifact is missing")
    preflight = json.loads(preflight_path.read_text())
    if not preflight.get("passed") or preflight.get("rows_passed") != 96:
        raise SystemExit(f"{DISPLAY_VERSION} preflight is not a 96/96 pass")
    contract = build_contract()
    frozen = {
        "contract_payload_sha256": contract["payload_sha256"],
        "preflight_payload_sha256": recompute_payload_sha256(preflight_path),
        "preflight_raw_sha256": sha256_file(preflight_path),
        "implementation_files": _implementation_binding(),
    }
    if VISIBILITY_ROOT:
        visibility_path = ROOT / VISIBILITY_ROOT / "visibility.json"
        if not visibility_path.is_file():
            raise SystemExit(f"{DISPLAY_VERSION} paired visibility artifact is missing")
        visibility = json.loads(visibility_path.read_text())
        if not visibility.get("passed"):
            raise SystemExit(f"{DISPLAY_VERSION} paired visibility audit did not pass")
        frozen.update(
            {
                "visibility_payload_sha256": recompute_payload_sha256(
                    visibility_path
                ),
                "visibility_raw_sha256": sha256_file(visibility_path),
                "visibility_npz_sha256": str(visibility["raw_npz_sha256"]),
            }
        )
    return frozen


def _run_job(job: dict[str, Any]) -> dict[str, Any]:
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    from run_pact_place_expert_screen import run_row

    row = job["row"]
    return run_row(
        row,
        config_sha256=job["config_sha256"],
        output_root=job["output_root"],
        scene_xml=str(ROOT / row["pact_v1011_scene_relative"]),
    )


def rows_in_frozen_order() -> list[dict[str, Any]]:
    out = []
    role = 0
    for attempt_index in range(REVIEW_MAX_ATTEMPTS_PER_CELL):
        for family, side, pose in cells():
            out.append(
                build_row(
                    family,
                    side,
                    pose,
                    attempt_index,
                    role_index=role,
                )
            )
            role += 1
    return out


def choose_first(results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ordered = sorted(results, key=lambda item: int(item["role_index"]))
    complete = [item for item in ordered if item.get("status") == "complete"]
    successes = [item for item in complete if item.get("clean_success") is True][
        :REVIEW_SUCCESSES
    ]
    failures = [item for item in complete if item.get("clean_success") is not True][
        :REVIEW_FAILURES
    ]
    return successes, failures


def _result_row_map(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(row["role_index"]): row for row in rows}


def _failure_label(result: dict[str, Any]) -> str:
    cause = result.get("failure_cause") or {}
    if cause.get("code"):
        return str(cause["code"])
    audit = result.get("contact_audit") or {}
    totals = audit.get("contact_class_totals") or {}
    contacts = [name for name, count in totals.items() if int(count or 0) > 0]
    return "contact_" + "_".join(sorted(contacts)) if contacts else "task_failure"


def render_clip(job: dict[str, Any]) -> dict[str, Any]:
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("MLSPACES_ASSETS_DIR", str(ROOT / "assets"))
    os.environ.pop("DISPLAY", None)

    import cv2
    import mujoco
    from molmo_spaces.data_generation.pipeline import cleanup_episode_resources
    from run_pact_place_expert_screen import _make_config

    row = job["row"]
    result = job["result"]
    row_directory = Path(job["row_directory"])
    trajectory_path = row_directory / "trajectory.json"
    trajectory = json.loads(trajectory_path.read_text())
    steps = list(trajectory.get("steps") or [])
    if int(trajectory.get("n") or -1) != len(steps) or not steps:
        raise RuntimeError("retained trajectory is empty or has a bad frame count")
    selected_seed = dict(result.get("selected_seed") or {})
    if "seed_u32" not in selected_seed:
        raise RuntimeError("result lacks the accepted sampling seed")

    output = Path(job["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix=f"v1011_review_{row['role_index']}_"))
    sampler = task = writer = None
    try:
        config = _make_config(
            scratch / "result.json",
            scene_xml=ROOT / row["pact_v1011_scene_relative"],
            sampler_class=row["sampler_class"],
        )
        sampler = config.task_sampler_config.task_sampler_class(config)
        sampler.seed_task_sampling(int(selected_seed["seed_u32"]))
        sampler.set_pact_manifest_row(row)
        task = sampler.sample_task(house_index=int(row["scene_template_house_index"]))
        if task is None:
            raise RuntimeError("review reconstruction returned no task")
        recorded_layout_hash = (
            (result.get("scene_params") or {}).get("pact_v1011_layout_sha256")
        )
        rebuilt_layout_hash = task.scene_params.get("pact_v1011_layout_sha256")
        if rebuilt_layout_hash != recorded_layout_hash:
            raise RuntimeError(
                f"layout reconstruction mismatch: {rebuilt_layout_hash} != "
                f"{recorded_layout_hash}"
            )

        env = task.env
        model, data = env.current_model, env.current_data
        pane_width, pane_height = PANE_WH
        frame_size = (pane_width * len(VIEWS), pane_height)
        writer = cv2.VideoWriter(
            str(output), cv2.VideoWriter_fourcc(*"mp4v"), float(FPS), frame_size
        )
        if not writer.isOpened():
            raise RuntimeError("OpenCV could not open the MP4 writer")
        poses = {
            name: camera_pose(AIM, *values[:3]) for name, values in VIEWS.items()
        }
        clean = bool(result.get("clean_success"))
        outcome = "CLEAN SUCCESS" if clean else f"FAILURE: {_failure_label(result)}"
        totals = (result.get("contact_audit") or {}).get("contact_class_totals") or {}
        clutter_contacts = int(totals.get("clutter", 0) or 0)
        stability = len(result.get("clutter_stability_events") or [])
        for frame_index, step in enumerate(steps):
            qpos = np.asarray(step["qpos"], dtype=np.float64)
            if qpos.shape != data.qpos.shape:
                raise RuntimeError(
                    f"frame {frame_index} qpos {qpos.shape} != model {data.qpos.shape}"
                )
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            panes = []
            for view_name, values in VIEWS.items():
                position, forward, up = poses[view_name]
                image = np.asarray(
                    env._render_frame(
                        position,
                        forward,
                        up,
                        values[3],
                        segmentation=False,
                    )
                )
                if image.shape[:2] != (pane_height, pane_width):
                    image = cv2.resize(
                        image, (pane_width, pane_height), interpolation=cv2.INTER_AREA
                    )
                panes.append(image)
            frame = cv2.cvtColor(np.concatenate(panes, axis=1), cv2.COLOR_RGB2BGR)
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], 82), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.70, frame, 0.30, 0.0)
            color = (110, 255, 110) if clean else (70, 170, 255)
            lines = (
                f"{DISPLAY_VERSION} {outcome} | role {row['role_index']} | "
                f"{row['family_id']} | {row['intrusion_side']} | {row['pose_id']}",
                f"frame {frame_index + 1}/{len(steps)} | phase "
                f"{step.get('policy_phase')} | task_success={bool(result.get('task_success'))} | "
                f"clutter contacts={clutter_contacts} | stability events={stability}",
                "table_left_high                                      table_left_mid",
            )
            for index, line in enumerate(lines):
                cv2.putText(
                    frame,
                    line,
                    (12, 24 + index * 23),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    color if index == 0 else (235, 235, 235),
                    1,
                    cv2.LINE_AA,
                )
            writer.write(frame)
        writer.release()
        writer = None
        if not output.is_file() or output.stat().st_size <= 0:
            raise RuntimeError("review MP4 was not written")
        capture = cv2.VideoCapture(str(output))
        decoded_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        decoded_fps = float(capture.get(cv2.CAP_PROP_FPS))
        capture.release()
        if decoded_frames != len(steps):
            raise RuntimeError(f"decoded {decoded_frames} frames, expected {len(steps)}")
        return {
            "ok": True,
            "role_index": row["role_index"],
            "kind": "success" if clean else "failure",
            "outcome": outcome,
            "path": str(output.relative_to(ROOT)),
            "raw_sha256": sha256_file(output),
            "bytes": output.stat().st_size,
            "frames": decoded_frames,
            "fps": decoded_fps,
            "duration_s": decoded_frames / decoded_fps,
            "layout_sha256": rebuilt_layout_hash,
            "trajectory_path": str(trajectory_path.relative_to(ROOT)),
            "trajectory_sha256": sha256_file(trajectory_path),
            "result_path": str((row_directory / "result.json").relative_to(ROOT)),
            "result_sha256": sha256_file(row_directory / "result.json"),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "role_index": row["role_index"],
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        if writer is not None:
            writer.release()
        try:
            cleanup_episode_resources(
                task=task,
                policy=None,
                task_sampler=sampler,
                preloaded_policy=None,
                close_task_sampler=sampler is not None,
            )
        except Exception:
            pass


def review_readme(manifest: dict[str, Any]) -> str:
    lines = [
        f"# {DISPLAY_VERSION} owner review",
        "",
        "Review the six videos in `videos/`: three strict-clean successes and three",
        "natural production failures, selected as the first three of each class in",
        "the preregistered cell/attempt order. Both panes are fixed measured table",
        "views; every retained qpos frame is rendered at true control-step timing.",
        "",
        "This packet validates only the environment appearance and expert behavior.",
        "It does not authorize collection, conversion, training, or evaluation.",
        "",
        "## Clips",
        "",
    ]
    for clip in manifest["videos"]:
        lines.append(
            f"- `{Path(clip['path']).name}` — {clip['outcome']}; "
            f"{clip['frames']} frames, {clip['duration_s']:.1f} s"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--render-workers", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=12)
    args = parser.parse_args()

    root = ROOT / REVIEW_ROOT
    manifest_path = root / "review_manifest.json"
    if manifest_path.exists():
        raise SystemExit(f"review packet already exists: {manifest_path}")
    config = frozen_config()
    config_sha256 = sha256_payload(config)
    rows = rows_in_frozen_order()
    row_by_role = _result_row_map(rows)
    results: list[dict[str, Any]] = []
    start = 0
    while start < len(rows):
        batch = rows[start : start + int(args.batch_size)]
        jobs = [
            {
                "row": row,
                "config_sha256": config_sha256,
                "output_root": str(root),
            }
            for row in batch
        ]
        print(f"generating roles {batch[0]['role_index']}..{batch[-1]['role_index']}", flush=True)
        with ProcessPoolExecutor(max_workers=min(args.workers, len(batch))) as pool:
            futures = {pool.submit(_run_job, job): job for job in jobs}
            for index, future in enumerate(as_completed(futures), 1):
                result = future.result()
                results.append(result)
                print(
                    f"  {index}/{len(batch)} role={result['role_index']} "
                    f"status={result['status']} clean={result.get('clean_success')}",
                    flush=True,
                )
        successes, failures = choose_first(results)
        state = {
            "schema_version": "pact_place_v1011_review_generation_state_v1",
            "config_sha256": config_sha256,
            "roles_attempted": sorted(int(item["role_index"]) for item in results),
            "success_roles": [int(item["role_index"]) for item in successes],
            "failure_roles": [int(item["role_index"]) for item in failures],
            "status_counts": {
                status: sum(item.get("status") == status for item in results)
                for status in ("complete", "sampling_failure", "infrastructure_failure")
            },
        }
        root.mkdir(parents=True, exist_ok=True)
        (root / "generation_state.json").write_text(
            json.dumps(state, indent=2, sort_keys=True) + "\n"
        )
        if len(successes) == REVIEW_SUCCESSES and len(failures) == REVIEW_FAILURES:
            break
        start += len(batch)
    successes, failures = choose_first(results)
    if len(successes) != REVIEW_SUCCESSES or len(failures) != REVIEW_FAILURES:
        raise SystemExit(
            f"review population incomplete: {len(successes)} success, "
            f"{len(failures)} failure"
        )
    selected = successes + failures
    render_jobs = []
    for result in selected:
        row = row_by_role[int(result["role_index"])]
        row_directory = (
            root
            / "expert_screen_rows"
            / f"{row['role_index']:02d}_{row['episode_id'][:16]}"
        )
        kind = "success" if result.get("clean_success") else "failure"
        label = "clean" if kind == "success" else _failure_label(result)
        filename = (
            f"{kind}_{int(row['role_index']):03d}_{row['family_id']}_"
            f"{row['intrusion_side']}_{row['pose_id']}_{label}.mp4"
        )
        render_jobs.append(
            {
                "row": row,
                "result": result,
                "row_directory": str(row_directory),
                "output": str(root / "videos" / filename),
            }
        )
    videos = []
    with ProcessPoolExecutor(max_workers=args.render_workers) as pool:
        futures = {pool.submit(render_clip, job): job for job in render_jobs}
        for index, future in enumerate(as_completed(futures), 1):
            video = future.result()
            videos.append(video)
            print(f"  rendered {index}/6 role={video['role_index']} ok={video['ok']}", flush=True)
    videos.sort(key=lambda item: (item.get("kind") != "success", int(item["role_index"])))
    failures_to_render = [item for item in videos if not item.get("ok")]
    selected_records = []
    for result in selected:
        row = row_by_role[int(result["role_index"])]
        selected_records.append(
            {
                "role_index": row["role_index"],
                "selection_basis": (
                    "first_strict_clean_in_frozen_order"
                    if result.get("clean_success")
                    else "first_complete_non_clean_in_frozen_order"
                ),
                "family_id": row["family_id"],
                "intrusion_side": row["intrusion_side"],
                "pose_id": row["pose_id"],
                "attempt_index": row["attempt_index"],
                "episode_id": row["episode_id"],
                "row_sha256": row["row_sha256"],
                "status": result["status"],
                "clean_success": bool(result.get("clean_success")),
                "task_success": bool(result.get("task_success")),
                "failure_cause": result.get("failure_cause"),
                "contact_class_totals": (
                    (result.get("contact_audit") or {}).get("contact_class_totals") or {}
                ),
                "clutter_stability_events": len(
                    result.get("clutter_stability_events") or []
                ),
                "episode_steps": result.get("episode_steps"),
            }
        )
    manifest = {
        **empty_authorization(),
        "schema_version": f"{CONTRACT_VERSION}_owner_review",
        "contract_version": CONTRACT_VERSION,
        "environment_version": ENVIRONMENT_VERSION,
        "scope": "owner_review_only",
        "config": config,
        "config_sha256": config_sha256,
        "generation": {
            "rows_executed": len(results),
            "complete": sum(item.get("status") == "complete" for item in results),
            "sampling_failures": sum(
                item.get("status") == "sampling_failure" for item in results
            ),
            "infrastructure_failures": sum(
                item.get("status") == "infrastructure_failure" for item in results
            ),
            "batch_size": args.batch_size,
            "selection_order": "role_index ascending after each completed batch",
        },
        "selected": selected_records,
        "videos": videos,
        "selection_checks": {
            "three_clean_successes": len(successes) == 3,
            "three_natural_failures": len(failures) == 3,
            "successes_are_first_in_order": [r["role_index"] for r in successes]
            == [r["role_index"] for r in choose_first(results)[0]],
            "failures_are_first_in_order": [r["role_index"] for r in failures]
            == [r["role_index"] for r in choose_first(results)[1]],
            "six_decodable_videos": len(videos) == 6 and not failures_to_render,
            "all_layout_reconstructions_match": all(
                item.get("layout_sha256") for item in videos
            ),
        },
        "eligible_for_owner_review": bool(
            len(successes) == 3
            and len(failures) == 3
            and len(videos) == 6
            and not failures_to_render
        ),
        "owner_accepted": None,
        "next_action": "owner_reviews_six_videos",
    }
    manifest["payload_sha256"] = canonical_payload_sha256(manifest)
    write_immutable_create_only(manifest_path, manifest)
    (root / "REVIEW.md").write_text(review_readme(manifest))
    print(
        json.dumps(
            {
                "eligible_for_owner_review": manifest["eligible_for_owner_review"],
                "success_roles": [item["role_index"] for item in selected_records[:3]],
                "failure_roles": [item["role_index"] for item in selected_records[3:]],
                "videos": [item.get("path") for item in videos],
                "payload_sha256": manifest["payload_sha256"],
            },
            indent=2,
        )
    )
    return 0 if manifest["eligible_for_owner_review"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
