#!/usr/bin/env python3
"""V10.7 Step 7: the frozen 48-row production pool, then six review videos.

The pool manifest is frozen and written before row 0. The scaled yield floors
are checked BEFORE any video is rendered, so a curated packet cannot be
published for an environment already unlikely to pass 16/24.

All six clips are complete production episodes from this pool. No diagnostic
assembly, induced contact, or retained-qpos control is used as one of the six.
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
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "scripts", ROOT / "submodules" / "molmospaces"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from pact_place_v107_contract import (  # noqa: E402
    CAUSAL_ROOT,
    CERT_ROOT,
    CONTRACT_VERSION_V107,
    ENVIRONMENT_VERSION,
    INTRUSION_SIDES,
    N_POOL_ROWS,
    N_REVIEW_VIDEOS,
    POOL_ROOT,
    POSE_IDS,
    REVIEW_FPS,
    REVIEW_FRAME_STRIDE,
    REVIEW_ROOT,
    SPEC_ROOT,
    V95_LAYOUT_FAMILY_IDS,
    assert_no_drift,
    empty_authorization,
    is_clean,
    pool_eligibility,
    pool_rows,
    recompute_payload_sha256,
    row_defects,
    sha256_file,
    sha256_payload,
    write_immutable_create_only,
    write_immutable_text_create_only,
)


def _pin() -> None:
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"


def run_pool_row(payload: dict[str, Any]) -> dict[str, Any]:
    _pin()
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("MLSPACES_ASSETS_DIR", str(ROOT / "assets"))
    os.environ.pop("DISPLAY", None)
    from run_pact_place_expert_screen import run_row

    return run_row(
        payload["row"],
        config_sha256=payload["config_sha256"],
        output_root=payload["output_root"],
        scene_xml=payload["scene_xml"],
    )


def select_six(rows, results) -> dict[str, Any]:
    """Three natural clean successes and three natural failures, balanced.

    One per pendant pose in each outcome class, three left and three right
    across all six, and at least two V9.5 layout families per class. Ties break
    lexicographically by manifest role index; nothing is relabelled.
    """
    by_role = {int(r["role_index"]): r for r in results}
    records = []
    for row in rows:
        result = by_role.get(int(row["role_index"]))
        if result is None:
            continue
        records.append({
            "role_index": int(row["role_index"]),
            "pose_id": row["pose_id"], "intrusion_side": row["intrusion_side"],
            "family_id": row["family_id"], "episode_id": row["episode_id"],
            "clean": bool(result.get("v107_clean_success")),
            "defects": result.get("v107_defects") or [],
            "min_clearance_m": (
                (result.get("pact_v106_frame_telemetry") or {}).get("min_clearance_m")
            ),
        })
    clean = sorted([r for r in records if r["clean"]],
                   key=lambda r: r["role_index"])
    failed = sorted([r for r in records if not r["clean"]],
                    key=lambda r: r["role_index"])

    def pick(pool, want_sides):
        """One row per pose, honouring a side multiset, families >= 2."""
        import itertools

        by_pose = {p: [r for r in pool if r["pose_id"] == p] for p in POSE_IDS}
        if any(not by_pose[p] for p in POSE_IDS):
            return None
        best = None
        for combo in itertools.product(*(by_pose[p] for p in POSE_IDS)):
            sides = sorted(r["intrusion_side"] for r in combo)
            if sides != sorted(want_sides):
                continue
            if len({r["family_id"] for r in combo}) < 2:
                continue
            key = tuple(sorted(r["role_index"] for r in combo))
            if best is None or key < best[0]:
                best = (key, list(combo))
        return None if best is None else best[1]

    for success_sides in (["left", "left", "right"], ["left", "right", "right"]):
        failure_sides = ["left", "right", "right"] if success_sides.count(
            "left") == 2 else ["left", "left", "right"]
        successes = pick(clean, success_sides)
        failures = pick(failed, failure_sides)
        if successes and failures:
            return {
                "found": True, "successes": successes, "failures": failures,
                "n_clean": len(clean), "n_failed": len(failed),
                "success_sides": success_sides, "failure_sides": failure_sides,
            }
    return {"found": False, "n_clean": len(clean), "n_failed": len(failed),
            "reason": "no balanced six-clip subset exists in the pool"}


def render_clip(job: dict[str, Any]) -> dict[str, Any]:
    """Complete production episode, stride 1, true time, four panes."""
    import cv2
    import mujoco
    from molmo_spaces.data_generation.pipeline import cleanup_episode_resources
    from pact_place_v105_clearance import (
        assembly_boxes, frame_clearances, geom_shape_cache,
        pendant_contact_state, pendant_geom_ids, robot_collision_geom_ids,
        target_collision_geom_ids,
    )
    from pact_place_v106_geometry import ALL_GEOMS_V106, build_assembly
    from pact_place_v107_contract import POSE_OFFSETS_M
    from run_pact_place_expert_screen import _make_config
    from run_pact_place_v6c_replay_videos import (
        PANE_WH, THIRD_PERSON_FOV, WRIST_FOV, _resize_pane,
        apply_recorded_qpos, third_person_pose, wrist_camera_pose,
    )

    row = job["row"]
    steps = json.loads(Path(job["trajectory_path"]).read_text())["steps"]
    output = Path(job["video_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    assembly = build_assembly(
        float(row["pact_v106_x_m"]), float(row["pact_v106_r_neg_m"]),
        float(row["pact_v106_r_pos_m"]), POSE_OFFSETS_M[row["pose_id"]],
        pose_id=row["pose_id"],
    )
    pendant_cam_pos = np.asarray([0.16, -0.02, 1.30], dtype=float)
    pendant_cam_target = np.asarray([0.78, 0.00, 1.02], dtype=float)

    task = sampler = scratch = writer = None
    try:
        scratch = Path(tempfile.mkdtemp(prefix="v107_render_"))
        config = _make_config(scratch / "d.json",
                              scene_xml=Path(job["scene_xml"]),
                              sampler_class=row["sampler_class"])
        sampler = config.task_sampler_config.task_sampler_class(config)
        sampler.seed_task_sampling(int(job["seed_u32"]))
        sampler.set_pact_manifest_row(row)
        task = sampler.sample_task(house_index=int(row["scene_template_house_index"]))
        env = task.env
        model, data = env.current_model, env.current_data
        probe = robot_collision_geom_ids(model) + target_collision_geom_ids(task)
        cache = geom_shape_cache(model, probe)
        boxes = assembly_boxes(assembly)
        pendant_ids = pendant_geom_ids(model, list(ALL_GEOMS_V106))
        side_delta = pendant_cam_target - pendant_cam_pos
        side_fwd = side_delta / np.linalg.norm(side_delta)
        side_up = np.asarray([0.0, 0.0, 1.0], dtype=float)
        width, height = PANE_WH
        writer = cv2.VideoWriter(
            str(output), cv2.VideoWriter_fourcc(*"mp4v"), float(REVIEW_FPS),
            (width * 3, height))
        contact_frames = 0
        for index in range(0, len(steps), int(REVIEW_FRAME_STRIDE)):
            step = steps[index]
            apply_recorded_qpos(env, step["qpos"])
            mujoco.mj_forward(model, data)
            report = frame_clearances(model, data, boxes, probe, cache)
            contact = pendant_contact_state(model, data, pendant_ids)
            if contact["contact"]:
                contact_frames += 1
            w_pos, w_fwd, w_up = wrist_camera_pose(env)
            wrist = _resize_pane(np.asarray(
                env._render_frame(w_pos, w_fwd, w_up, WRIST_FOV, segmentation=False)))
            t_pos, t_fwd, t_up = third_person_pose(env)
            third = _resize_pane(np.asarray(
                env._render_frame(t_pos, t_fwd, t_up, THIRD_PERSON_FOV,
                                  segmentation=False)))
            pendant = _resize_pane(np.asarray(
                env._render_frame(pendant_cam_pos, side_fwd, side_up, 52.0,
                                  segmentation=False)))
            frame = cv2.cvtColor(np.concatenate([wrist, third, pendant], axis=1),
                                 cv2.COLOR_RGB2BGR)
            shade = frame.copy()
            header = 128
            cv2.rectangle(shade, (0, 0), (frame.shape[1], header), (0, 0, 0), -1)
            cv2.rectangle(shade, (0, height - 30), (frame.shape[1], height),
                          (0, 0, 0), -1)
            frame = cv2.addWeighted(shade, 0.70, frame, 0.30, 0.0)
            minimum = report["min_m"]
            limiting = report["limiting"] or {}
            colour = (120, 255, 120) if job["clean"] else (60, 160, 255)
            lines = [
                f"{'PRODUCTION CLEAN SUCCESS' if job['clean'] else 'PRODUCTION FAILURE'}"
                f"  row{row['role_index']:02d}  {row['family_id']}  "
                f"side {row['intrusion_side']}  pose {row['pose_id']}  "
                f"frame {index}/{len(steps) - 1}  ({REVIEW_FPS:.4f} fps, stride 1)",
                f"phase {step.get('policy_phase')}  "
                f"commanded {step.get('commanded_speed_m_s')}  "
                f"realized {step.get('realized_tcp_speed_m_s')}",
                f"pendant clearance "
                f"{'--' if minimum is None else format(float(minimum), '.4f')} m"
                f"   limiting {limiting.get('component')} <-> "
                f"{limiting.get('probe_body')}",
                f"pendant contact: {'YES' if contact['contact'] else 'no'}   "
                f"cumulative {contact_frames}   outcome {job['outcome']}",
            ]
            for i, line in enumerate(lines):
                cv2.putText(frame, line, (12, 24 + i * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                            colour if i == 0 else (230, 230, 230), 1, cv2.LINE_AA)
            for pane, label in enumerate((
                "wrist (RGB policy view, untinted)",
                "third-person: household clutter, robot, target, panel, pendant",
                "pendant view / clearance",
            )):
                cv2.putText(frame, label, (pane * width + 12, height - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (245, 245, 245), 1,
                            cv2.LINE_AA)
            writer.write(frame)
        writer.release()
        writer = None
        return {
            "label": job["label"], "kind": job["kind"],
            "role_index": int(row["role_index"]),
            "pose_id": row["pose_id"], "intrusion_side": row["intrusion_side"],
            "family_id": row["family_id"],
            "video_path": str(output), "video_name": output.name,
            "video_sha256": sha256_file(output),
            "size_bytes": int(output.stat().st_size),
            "n_frames": len(steps), "fps": float(REVIEW_FPS),
            "duration_s": len(steps) / float(REVIEW_FPS),
            "complete_production_episode": True,
            "outcome": job["outcome"],
            "n_pendant_contact_frames": contact_frames,
        }
    finally:
        if writer is not None:
            writer.release()
        cleanup_episode_resources(
            task=task, policy=None, task_sampler=sampler,
            preloaded_policy=None, close_task_sampler=sampler is not None)
        if scratch is not None:
            shutil.rmtree(scratch, ignore_errors=True)


def decode_video(path: Path) -> dict[str, Any]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return {"decoded": False}
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        counted = 0
        while True:
            ok, _ = capture.read()
            if not ok:
                break
            counted += 1
    finally:
        capture.release()
    return {"decoded": True, "decoded_frames": counted, "decoded_fps": fps,
            "size_bytes": int(path.stat().st_size), "sha256": sha256_file(path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-root", type=Path, default=ROOT / POOL_ROOT)
    parser.add_argument("--review-root", type=Path, default=ROOT / REVIEW_ROOT)
    parser.add_argument("--certification", type=Path,
                        default=ROOT / CERT_ROOT / "certification.json")
    parser.add_argument("--causal", type=Path,
                        default=ROOT / CAUSAL_ROOT / "causal.json")
    parser.add_argument("--specification", type=Path,
                        default=ROOT / SPEC_ROOT / "specification.json")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    _pin()
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("MLSPACES_ASSETS_DIR", str(ROOT / "assets"))
    os.environ.pop("DISPLAY", None)
    started = time.time()

    spec = json.loads(args.specification.resolve().read_text())
    drift = assert_no_drift(spec)
    cert_path = args.certification.resolve()
    cert = json.loads(cert_path.read_text())
    causal_path = args.causal.resolve()
    causal = json.loads(causal_path.read_text())
    if not cert.get("certification_passed"):
        raise SystemExit("certification did not pass; the pool does not run")
    if not causal.get("causal_passed"):
        raise SystemExit("six-group causality did not pass; the pool does not run")

    pool_root = args.pool_root.resolve()
    if pool_root.exists():
        raise SystemExit(f"refusing to overwrite {pool_root}")
    pool_root.mkdir(parents=True)

    scene_by_pose = {
        p: {"relative": cert["published_scenes"][p]["relative"],
            "sha256": cert["published_scenes"][p]["sha256"]}
        for p in POSE_IDS
    }
    assembly_by_pose = {
        p: sha256_payload(
            next(c for c in cert["compiled_checks"] if c["pose_id"] == p)
        ) for p in POSE_IDS
    }
    selected = cert["selected"]
    rows = pool_rows(selected=selected, scene_by_pose=scene_by_pose,
                     assembly_by_pose=assembly_by_pose)
    if len(rows) != N_POOL_ROWS:
        raise RuntimeError(f"pool must have {N_POOL_ROWS} rows, got {len(rows)}")

    config = {
        "schema_version": "pact_place_v107_pool_config_v1",
        "contract_version": CONTRACT_VERSION_V107,
        "environment_version": ENVIRONMENT_VERSION,
        "selected": selected,
        "scene_by_pose": scene_by_pose,
        "frozen_before_row_0": True,
        "expert_screen_rows": rows,
        **empty_authorization(),
    }
    config["config_sha256"] = sha256_payload(config)
    write_immutable_create_only(pool_root / "pool_manifest.json", config)

    print(f"executing frozen pool: {len(rows)} rows", flush=True)
    context = multiprocessing.get_context("spawn")
    results: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max(1, min(args.workers, len(rows))),
        mp_context=context, max_tasks_per_child=1,
    ) as executor:
        futures = [
            executor.submit(run_pool_row, {
                "row": row, "config_sha256": config["config_sha256"],
                "output_root": str(pool_root),
                "scene_xml": str(ROOT / scene_by_pose[row["pose_id"]]["relative"]),
            })
            for row in rows
        ]
        for done, future in enumerate(
            concurrent.futures.as_completed(futures), start=1
        ):
            result = future.result()
            result["v107_defects"] = row_defects(result)
            result["v107_clean_success"] = not result["v107_defects"]
            results.append(result)
            print(json.dumps({
                "done": done, "of": len(rows),
                "role": int(result["role_index"]),
                "clean": result["v107_clean_success"],
                "defects": result["v107_defects"][:2],
            }), flush=True)
    results.sort(key=lambda item: int(item["role_index"]))
    eligibility = pool_eligibility(rows, results)

    pool_doc = {
        "schema_version": "pact_place_v107_pool_v1",
        "contract_version": CONTRACT_VERSION_V107,
        "environment_version": ENVIRONMENT_VERSION,
        "specification_payload_sha256": recompute_payload_sha256(
            args.specification.resolve()),
        "drift_check": drift,
        "certification_payload_sha256": recompute_payload_sha256(cert_path),
        "causal_payload_sha256": recompute_payload_sha256(causal_path),
        "config_sha256": config["config_sha256"],
        "selected": selected,
        "n_rows": len(rows),
        "rows": [
            {k: row[k] for k in ("role_index", "replicate", "family_id",
                                 "intrusion_side", "pose_id", "episode_id",
                                 "task_seed_u32", "row_sha256")}
            for row in rows
        ],
        "results": [
            {
                "role_index": int(r["role_index"]),
                "status": r.get("status"),
                "v107_clean_success": r["v107_clean_success"],
                "v107_defects": r["v107_defects"],
                "episode_steps": r.get("episode_steps"),
                "failure_cause": r.get("failure_cause"),
                "pact_v106_frame_telemetry": r.get("pact_v106_frame_telemetry"),
                "result_sha256": r.get("result_sha256"),
            }
            for r in results
        ],
        "eligibility": eligibility,
        "elapsed_s": time.time() - started,
        **empty_authorization(),
        "pool_passed": bool(eligibility["pool_passed"]),
    }
    pool_hashes = write_immutable_create_only(pool_root / "pool.json", pool_doc)
    print(json.dumps({
        "pool_passed": pool_doc["pool_passed"],
        "clean": eligibility["clean_successes"],
        "by_side": eligibility["clean_by_side"],
        "by_pose": eligibility["clean_by_pose"],
        "by_cell": eligibility["clean_by_side_pose"],
        "limiting": eligibility["limiting_predicates"],
        **pool_hashes,
    }, indent=2))
    if not pool_doc["pool_passed"]:
        print("pool floors not met; no packet published", flush=True)
        return 1

    # --- videos, only after the floors pass -------------------------------
    from run_pact_place_expert_screen import _result_path

    plan = select_six(rows, results)
    if not plan["found"]:
        stop = {
            "schema_version": "pact_place_v107_review_stop_v1",
            "contract_version": CONTRACT_VERSION_V107,
            "stop_reason": plan["reason"],
            "n_clean": plan["n_clean"], "n_failed": plan["n_failed"],
            "videos_published": 0,
            **empty_authorization(),
        }
        review_root = args.review_root.resolve()
        review_root.mkdir(parents=True, exist_ok=True)
        write_immutable_create_only(review_root / "review_stop.json", stop)
        print(json.dumps(stop, indent=2))
        return 1

    by_role = {int(r["role_index"]): r for r in rows}
    staging = Path(tempfile.mkdtemp(prefix="v107_review_", dir=ROOT / "diagnostics_output"))
    try:
        jobs = []
        for record in plan["successes"]:
            row = by_role[record["role_index"]]
            path = _result_path(pool_root, row)
            jobs.append({
                "kind": "production_clean_success", "clean": True,
                "label": f"row{record['role_index']:02d}-{row['pose_id']}-"
                         f"{row['intrusion_side']}-clean",
                "row": row, "seed_u32": int(row["task_seed_u32"]),
                "trajectory_path": str(path.parent / "trajectory.json"),
                "scene_xml": str(ROOT / scene_by_pose[row["pose_id"]]["relative"]),
                "outcome": "clean success",
                "video_path": str(staging / "videos" /
                                  f"success_{record['role_index']:02d}_"
                                  f"{row['pose_id']}_{row['intrusion_side']}.mp4"),
            })
        for record in plan["failures"]:
            row = by_role[record["role_index"]]
            path = _result_path(pool_root, row)
            jobs.append({
                "kind": "production_failure", "clean": False,
                "label": f"row{record['role_index']:02d}-{row['pose_id']}-"
                         f"{row['intrusion_side']}-failure",
                "row": row, "seed_u32": int(row["task_seed_u32"]),
                "trajectory_path": str(path.parent / "trajectory.json"),
                "scene_xml": str(ROOT / scene_by_pose[row["pose_id"]]["relative"]),
                "outcome": ", ".join(record["defects"][:3]) or "failure",
                "video_path": str(staging / "videos" /
                                  f"failure_{record['role_index']:02d}_"
                                  f"{row['pose_id']}_{row['intrusion_side']}.mp4"),
            })
        rendered = [render_clip(job) for job in jobs]
        verifications = []
        for record in rendered:
            decoded = decode_video(Path(record["video_path"]))
            verifications.append({
                **decoded,
                "expected_frames": record["n_frames"],
                "frame_count_matches": decoded.get("decoded_frames")
                == record["n_frames"],
                "sha256_stable": decoded.get("sha256") == record["video_sha256"],
                "nonzero_size": int(decoded.get("size_bytes", 0)) > 0,
            })
        if not all(v["frame_count_matches"] and v["sha256_stable"]
                   and v["nonzero_size"] for v in verifications):
            raise RuntimeError(f"video decode verification failed: {verifications}")

        for record in rendered:
            record["video_path"] = (
                f"{REVIEW_ROOT}/videos/{record['video_name']}"
            )
        manifest = {
            "schema_version": "pact_place_v107_review_manifest_v1",
            "contract_version": CONTRACT_VERSION_V107,
            "environment_version": ENVIRONMENT_VERSION,
            "pool_payload_sha256": pool_hashes["payload_sha256"],
            "pool_raw_file_sha256": pool_hashes["raw_file_sha256"],
            "certification_payload_sha256": recompute_payload_sha256(cert_path),
            "causal_payload_sha256": recompute_payload_sha256(causal_path),
            "selected": selected,
            "n_videos": len(rendered),
            "n_production_successes": sum(1 for r in rendered if r["kind"]
                                          == "production_clean_success"),
            "n_production_failures": sum(1 for r in rendered if r["kind"]
                                         == "production_failure"),
            "all_clips_are_complete_production_episodes": True,
            "diagnostic_assemblies_used": False,
            "videos": rendered,
            "video_sha256": {r["video_name"]: r["video_sha256"] for r in rendered},
            "decode_verifications": verifications,
            "pool_eligibility": eligibility,
            "selection_plan": {k: v for k, v in plan.items()
                               if k in ("success_sides", "failure_sides",
                                        "n_clean", "n_failed")},
            **empty_authorization(),
            "eligible_for_human_review": True,
        }
        write_immutable_create_only(staging / "review_manifest.json", manifest)
        lines = [
            "# V10.7 review packet", "",
            "Six complete production episodes from the frozen 48-row pool:",
            "three natural strict-clean successes and three natural failures.",
            "No diagnostic assembly, induced contact, or retained-qpos control",
            "is used as one of the six.", "",
            f"Pool yield: **{eligibility['clean_successes']}/48** clean "
            f"(floor {eligibility['min_clean_required']}), "
            f"by side {eligibility['clean_by_side']}, "
            f"by pose {eligibility['clean_by_pose']}.", "",
            "| # | kind | pose | side | family | frames | duration |",
            "|---|---|---|---|---|---|---|",
        ]
        for i, video in enumerate(rendered, start=1):
            lines.append(
                f"| {i} | {video['kind']} | {video['pose_id']} | "
                f"{video['intrusion_side']} | {video['family_id']} | "
                f"{video['n_frames']} | {video['duration_s']:.2f} s |"
            )
        lines += [
            "", "`human_approval.json` is absent and was not created.",
            "Phase 0 has not run. Every authorization field is false.", "",
        ]
        write_immutable_text_create_only(staging / "REVIEW.md", "\n".join(lines))

        review_root = args.review_root.resolve()
        if review_root.exists():
            raise RuntimeError(f"refusing to publish over {review_root}")
        os.rename(staging, review_root)
        published = sorted((review_root / "videos").glob("*.mp4"))
        print(json.dumps({
            "review_published": True,
            "n_videos": len(published),
            "videos": [str(p.relative_to(ROOT)) for p in published],
            "human_approval_present": (
                review_root / "human_approval.json").exists(),
            **empty_authorization(),
            "eligible_for_human_review": True,
        }, indent=2))
        return 0
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
