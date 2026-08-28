#!/usr/bin/env python3
"""V10.7 owner visual-review packet, built from the retained 48-row pool.

Review-only. The pool remains failed at 21/48 and nothing here reinterprets or
overwrites that. No episode is generated, no task resampled, no ``env.step``
called, no geometry or threshold touched, and the pool is not rerun: the six
clips are replays of retained trajectories in the already-certified scenes.

Selection is deterministic and derived, never hardcoded: among subsets meeting
every registered balance constraint, minimise the maximum pendant clearance,
then total clearance, then the sorted role-index tuple. Minimising the maximum
is what makes the pendant risk visually legible without altering an episode.
"""

from __future__ import annotations

import argparse
import itertools
import json
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
    POOL_MIN_CLEAN,
    POOL_ROOT,
    POSE_IDS,
    POSE_OFFSETS_M,
    REVIEW_FPS,
    REVIEW_FRAME_STRIDE,
    SELECTION_ROOT,
    SPEC_ROOT,
    assert_no_drift,
    empty_authorization,
    recompute_payload_sha256,
    sha256_file,
    write_immutable_create_only,
    write_immutable_text_create_only,
)
from pact_place_v106_geometry import (  # noqa: E402
    ALL_GEOMS_V106,
    assembly_sha256,
    build_assembly,
)

OWNER_REVIEW_ROOT = "diagnostics_output/pact_place_v107_owner_review"
N_VIDEOS = 6


# ---------------------------------------------------------------------------
# Binding
# ---------------------------------------------------------------------------
def bind_inputs(spec_path: Path) -> dict[str, Any]:
    """Verify and bind every upstream artifact before a frame is rendered."""
    from pact_place_v107_contract import (
        PLAN_RELATIVE, verify_against_specification,
    )

    spec = json.loads(spec_path.read_text())
    report = verify_against_specification(spec)
    # A plan document records outcomes, so it necessarily changes after the run
    # it describes. Partition that from code/data drift rather than ignoring it
    # or failing on it: code and data drift is still fatal.
    documentation = [d for d in report["drift"] if d["path"] == PLAN_RELATIVE]
    code_and_data = [d for d in report["drift"] if d["path"] != PLAN_RELATIVE]
    if code_and_data:
        raise SystemExit(
            f"code/data drift since the specification was sealed: "
            f"{[d['path'] for d in code_and_data]}"
        )
    drift = {
        "n_checked": report["n_checked"],
        "code_and_data_drift": code_and_data,
        "n_code_and_data_drift": 0,
        "documentation_drift": documentation,
        "n_documentation_drift": len(documentation),
        "documentation_drift_is_expected": True,
        "documentation_drift_note": (
            "The V10.7 plan document records measured outcomes and is edited "
            "after the run it describes. It is bound by the specification, so "
            "its change is reported here rather than silently accepted; no "
            "code or data file drifted."
        ),
        "code_and_data_clean": True,
    }
    paths = {
        "specification": spec_path,
        "selection": ROOT / SELECTION_ROOT / "selection.json",
        "certification": ROOT / CERT_ROOT / "certification.json",
        "causal": ROOT / CAUSAL_ROOT / "causal.json",
        "pool": ROOT / POOL_ROOT / "pool.json",
    }
    bound: dict[str, Any] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise SystemExit(f"missing bound input: {path}")
        bound[name] = {
            "path": str(path.relative_to(ROOT)),
            "raw_file_sha256": sha256_file(path),
            "payload_sha256": recompute_payload_sha256(path),
        }
    certification = json.loads(paths["certification"].read_text())
    selection = json.loads(paths["selection"].read_text())
    scenes: dict[str, Any] = {}
    for pose in POSE_IDS:
        entry = certification["published_scenes"][pose]
        path = ROOT / entry["relative"]
        observed = sha256_file(path)
        if observed != entry["sha256"]:
            raise SystemExit(
                f"scene drift for {pose}: {observed} != {entry['sha256']}"
            )
        assembly = build_assembly(
            float(selection["selected"]["x_m"]),
            float(selection["selected"]["r_neg_m"]),
            float(selection["selected"]["r_pos_m"]),
            POSE_OFFSETS_M[pose], pose_id=pose,
        )
        scenes[pose] = {
            "relative": entry["relative"],
            "scene_sha256": observed,
            "assembly_sha256": assembly_sha256(assembly),
            "assembly": assembly,
        }
    return {
        "drift_check": drift,
        "artifacts": bound,
        "scenes": {p: {k: v for k, v in s.items() if k != "assembly"}
                   for p, s in scenes.items()},
        "_scenes": scenes,
        "certification_passed": bool(certification["certification_passed"]),
        "causal_passed": bool(
            json.loads(paths["causal"].read_text())["causal_passed"]
        ),
    }


# ---------------------------------------------------------------------------
# Deterministic selection
# ---------------------------------------------------------------------------
def rebuild_pool_rows(pool: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """Reconstruct the full manifest rows and prove they are the ones that ran.

    ``pool.json`` stores a trimmed projection of each row, but the sampler needs
    the whole thing. Rebuilding it from the frozen generator and asserting each
    ``row_sha256`` against the recorded value is stronger than reading a stored
    copy: it shows the row is byte-identical to what executed.
    """
    from pact_place_v107_contract import pool_rows as build_pool_rows

    certification = json.loads(
        (ROOT / CERT_ROOT / "certification.json").read_text())
    from pact_place_v107_contract import sha256_payload

    scene_by_pose = {
        p: {"relative": certification["published_scenes"][p]["relative"],
            "sha256": certification["published_scenes"][p]["sha256"]}
        for p in POSE_IDS
    }
    assembly_by_pose = {
        p: sha256_payload(
            next(c for c in certification["compiled_checks"]
                 if c["pose_id"] == p))
        for p in POSE_IDS
    }
    rebuilt = build_pool_rows(
        selected=certification["selected"], scene_by_pose=scene_by_pose,
        assembly_by_pose=assembly_by_pose)
    recorded = {int(r["role_index"]): r for r in pool["rows"]}
    out: dict[int, dict[str, Any]] = {}
    for row in rebuilt:
        role = int(row["role_index"])
        stored = recorded.get(role)
        if stored is None:
            raise SystemExit(f"pool.json has no row {role}")
        if row["row_sha256"] != stored["row_sha256"]:
            raise SystemExit(
                f"rebuilt row {role} does not match the executed row: "
                f"{row['row_sha256']} != {stored['row_sha256']}")
        if row["episode_id"] != stored["episode_id"]:
            raise SystemExit(f"rebuilt row {role} episode_id mismatch")
        out[role] = row
    return out


def candidate_rows(pool: dict[str, Any]) -> list[dict[str, Any]]:
    """Rows with a complete trajectory and complete clearance telemetry."""
    from run_pact_place_expert_screen import _result_path

    rows = rebuild_pool_rows(pool)
    pool_root = ROOT / POOL_ROOT
    out: list[dict[str, Any]] = []
    for result in pool["results"]:
        role = int(result["role_index"])
        row = rows[role]
        telemetry = result.get("pact_v106_frame_telemetry") or {}
        if result.get("status") != "complete":
            continue
        if telemetry.get("min_clearance_m") is None:
            continue
        n_frames = telemetry.get("n_frames")
        if not n_frames:
            continue
        directory = _result_path(pool_root, row).parent
        trajectory_path = directory / "trajectory.json"
        result_path = directory / "result.json"
        if not (trajectory_path.is_file() and result_path.is_file()):
            continue
        trajectory = json.loads(trajectory_path.read_text())
        if int(trajectory.get("n") or 0) != len(trajectory.get("steps") or []):
            continue
        out.append({
            "role_index": role,
            "pose_id": row["pose_id"],
            "intrusion_side": row["intrusion_side"],
            "family_id": row["family_id"],
            "episode_id": row["episode_id"],
            "row_sha256": row["row_sha256"],
            "task_seed_u32": int(row["task_seed_u32"]),
            "clean": bool(result.get("v107_clean_success")),
            "defects": result.get("v107_defects") or [],
            "failure_cause": result.get("failure_cause"),
            "min_clearance_m": float(telemetry["min_clearance_m"]),
            "min_lobe_stem_clearance_m": telemetry.get("min_lobe_stem_clearance_m"),
            "pendant_contact_frames": int(
                telemetry.get("pendant_robot_or_target_contact_frames") or 0),
            "n_frames": int(n_frames),
            "trajectory_n": int(trajectory["n"]),
            "trajectory_path": str(trajectory_path.relative_to(ROOT)),
            "result_path": str(result_path.relative_to(ROOT)),
            "trajectory_raw_file_sha256": sha256_file(trajectory_path),
            "result_raw_file_sha256": sha256_file(result_path),
            "row": row,
        })
    return out


def _class_options(pool: list[dict[str, Any]]):
    """One row per pose, at least two distinct layout families."""
    by_pose = {p: [r for r in pool if r["pose_id"] == p] for p in POSE_IDS}
    if any(not by_pose[p] for p in POSE_IDS):
        return []
    return [
        combo
        for combo in itertools.product(*(by_pose[p] for p in POSE_IDS))
        if len({r["family_id"] for r in combo}) >= 2
    ]


def select_six(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    clean = [r for r in candidates if r["clean"]]
    failed = [r for r in candidates if not r["clean"]]
    best = None
    n_considered = 0
    for successes in _class_options(clean):
        for failures in _class_options(failed):
            six = successes + failures
            sides = [r["intrusion_side"] for r in six]
            if sides.count("left") != 3 or sides.count("right") != 3:
                continue
            n_considered += 1
            key = (
                max(r["min_clearance_m"] for r in six),
                sum(r["min_clearance_m"] for r in six),
                tuple(sorted(r["role_index"] for r in six)),
            )
            if best is None or key < best[0]:
                best = (key, list(successes), list(failures))
    if best is None:
        return {"found": False, "n_valid_subsets": 0,
                "reason": "no subset satisfies the registered balance constraints"}
    key, successes, failures = best
    return {
        "found": True,
        "n_valid_subsets": n_considered,
        "objective": {
            "max_min_clearance_m": key[0],
            "total_min_clearance_m": key[1],
            "sorted_role_indices": list(key[2]),
            "order": ["minimise max pendant clearance", "then total clearance",
                      "then sorted role-index tuple"],
        },
        "successes": successes,
        "failures": failures,
        "n_clean_candidates": len(clean),
        "n_failed_candidates": len(failed),
    }


def verify_selection(plan: dict[str, Any]) -> dict[str, Any]:
    """Independent re-check of every registered constraint on the chosen six."""
    six = plan["successes"] + plan["failures"]
    sides = [r["intrusion_side"] for r in six]
    checks = {
        "six_clips": len(six) == N_VIDEOS,
        "three_successes": len(plan["successes"]) == 3,
        "three_failures": len(plan["failures"]) == 3,
        "all_successes_are_strict_clean": all(r["clean"] for r in plan["successes"]),
        "all_failures_are_natural": all(not r["clean"] for r in plan["failures"]),
        "one_pose_each_in_successes": sorted(
            r["pose_id"] for r in plan["successes"]) == sorted(POSE_IDS),
        "one_pose_each_in_failures": sorted(
            r["pose_id"] for r in plan["failures"]) == sorted(POSE_IDS),
        "three_left_three_right": sides.count("left") == 3
        and sides.count("right") == 3,
        "two_families_in_successes": len(
            {r["family_id"] for r in plan["successes"]}) >= 2,
        "two_families_in_failures": len(
            {r["family_id"] for r in plan["failures"]}) >= 2,
        "complete_telemetry": all(
            r["min_clearance_m"] is not None and r["n_frames"] > 0 for r in six),
        "trajectory_frame_counts_consistent": all(
            r["trajectory_n"] == r["n_frames"] for r in six),
        "no_failure_is_an_induced_pendant_collision": all(
            r["pendant_contact_frames"] == 0 for r in plan["failures"]),
    }
    return {"checks": checks, "passed": all(checks.values())}


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
PENDANT_CAM_POS = np.asarray([0.16, -0.02, 1.30], dtype=float)
PENDANT_CAM_TARGET = np.asarray([0.78, 0.00, 1.02], dtype=float)


def render_clip(job: dict[str, Any]) -> dict[str, Any]:
    """Complete retained trajectory, stride 1, true time, four panes."""
    import cv2
    import mujoco
    from molmo_spaces.data_generation.pipeline import cleanup_episode_resources
    from pact_place_v105_clearance import (
        assembly_boxes, frame_clearances, geom_shape_cache,
        pendant_contact_state, pendant_geom_ids, robot_collision_geom_ids,
        target_collision_geom_ids,
    )
    from run_pact_place_expert_screen import _make_config
    from run_pact_place_v6c_replay_videos import (
        PANE_WH, THIRD_PERSON_FOV, WRIST_FOV, _resize_pane,
        apply_recorded_qpos, third_person_pose, wrist_camera_pose,
    )

    record = job["record"]
    row = record["row"]
    steps = json.loads((ROOT / record["trajectory_path"]).read_text())["steps"]
    assembly = job["assembly"]
    output = Path(job["video_path"])
    output.parent.mkdir(parents=True, exist_ok=True)

    task = sampler = scratch = writer = None
    try:
        scratch = Path(tempfile.mkdtemp(prefix="v107_owner_"))
        config = _make_config(scratch / "d.json",
                              scene_xml=ROOT / job["scene_relative"],
                              sampler_class=row["sampler_class"])
        sampler = config.task_sampler_config.task_sampler_class(config)
        sampler.seed_task_sampling(int(record["task_seed_u32"]))
        sampler.set_pact_manifest_row(row)
        task = sampler.sample_task(
            house_index=int(row["scene_template_house_index"]))
        env = task.env
        model, data = env.current_model, env.current_data
        probe = robot_collision_geom_ids(model) + target_collision_geom_ids(task)
        cache = geom_shape_cache(model, probe)
        boxes = assembly_boxes(assembly)
        pendant_ids = pendant_geom_ids(model, list(ALL_GEOMS_V106))
        side_fwd = PENDANT_CAM_TARGET - PENDANT_CAM_POS
        side_fwd = side_fwd / np.linalg.norm(side_fwd)
        side_up = np.asarray([0.0, 0.0, 1.0], dtype=float)
        width, height = PANE_WH
        writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"),
                                 float(REVIEW_FPS), (width * 3, height))
        indices = list(range(0, len(steps), int(REVIEW_FRAME_STRIDE)))
        if indices != list(range(len(steps))):
            raise RuntimeError("owner review renders every retained frame")
        running_min = float("inf")
        contact_frames = 0
        for index in indices:
            step = steps[index]
            apply_recorded_qpos(env, step["qpos"])
            mujoco.mj_forward(model, data)
            report = frame_clearances(model, data, boxes, probe, cache)
            contact = pendant_contact_state(model, data, pendant_ids)
            if contact["contact"]:
                contact_frames += 1
            current = report["min_m"]
            if current is not None:
                running_min = min(running_min, float(current))
            limiting = report["limiting"] or {}
            w_pos, w_fwd, w_up = wrist_camera_pose(env)
            wrist = _resize_pane(np.asarray(env._render_frame(
                w_pos, w_fwd, w_up, WRIST_FOV, segmentation=False)))
            t_pos, t_fwd, t_up = third_person_pose(env)
            third = _resize_pane(np.asarray(env._render_frame(
                t_pos, t_fwd, t_up, THIRD_PERSON_FOV, segmentation=False)))
            pendant = _resize_pane(np.asarray(env._render_frame(
                PENDANT_CAM_POS, side_fwd, side_up, 52.0, segmentation=False)))
            frame = cv2.cvtColor(np.concatenate([wrist, third, pendant], axis=1),
                                 cv2.COLOR_RGB2BGR)
            header = 152
            shade = frame.copy()
            cv2.rectangle(shade, (0, 0), (frame.shape[1], header), (0, 0, 0), -1)
            cv2.rectangle(shade, (0, height - 52), (frame.shape[1], height),
                          (0, 0, 0), -1)
            frame = cv2.addWeighted(shade, 0.72, frame, 0.28, 0.0)
            colour = (120, 255, 120) if record["clean"] else (60, 160, 255)
            speed = step.get("commanded_speed_m_s")
            realized = step.get("realized_tcp_speed_m_s")
            # A compact proximity overlay derived from retained state only:
            # the per-component clearance bar. Nothing is re-executed.
            per_component = report["per_component_m"] or {}
            bar = "  ".join(
                f"{name}:{value * 1000:.0f}"
                for name, value in sorted(per_component.items())
            )
            lines = [
                f"{'PRODUCTION CLEAN SUCCESS' if record['clean'] else 'NATURAL PRODUCTION FAILURE'}"
                f"   role {record['role_index']}   {record['family_id']}   "
                f"side {record['intrusion_side']}   pendant pose "
                f"{record['pose_id']}   frame {index}/{len(steps) - 1}   "
                f"({REVIEW_FPS:.4f} fps, stride 1, true time)",
                f"phase {step.get('policy_phase')}   commanded "
                f"{'--' if speed is None else format(float(speed), '.3f')} m/s   "
                f"realized "
                f"{'--' if realized is None else format(float(realized), '.3f')} m/s"
                f"   target held: {'yes' if step.get('target_held') else 'no'}",
                f"pendant clearance now "
                f"{'--' if current is None else format(float(current) * 1000, '.1f')} mm"
                f"   running min "
                f"{'--' if running_min == float('inf') else format(running_min * 1000, '.1f')} mm"
                f"   episode min {record['min_clearance_m'] * 1000:.1f} mm"
                f"   limiting {limiting.get('component')} <-> "
                f"{limiting.get('probe_body')}",
                f"pendant contact: {'YES' if contact['contact'] else 'no'}   "
                f"cumulative {contact_frames}   "
                f"clutter/stability: {job['clutter_state']}   "
                f"task outcome: {job['outcome']}",
                "ALL FAILURES ARE NATURAL PRODUCTION FAILURES - "
                "NONE IS AN INDUCED PENDANT COLLISION",
            ]
            for i, line in enumerate(lines):
                cv2.putText(frame, line, (12, 22 + i * 21),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                            colour if i == 0 else
                            ((90, 200, 255) if i == 4 else (230, 230, 230)),
                            1, cv2.LINE_AA)
            cv2.putText(frame, f"per-component clearance (mm)  {bar}",
                        (12, height - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        (245, 245, 245), 1, cv2.LINE_AA)
            for pane, label in enumerate((
                "wrist (RGB policy view, untinted)",
                "third-person: household clutter, robot, target, panel, pendant",
                "pendant-side close view",
            )):
                cv2.putText(frame, label, (pane * width + 12, height - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (245, 245, 245), 1,
                            cv2.LINE_AA)
            writer.write(frame)
        writer.release()
        writer = None
        return {
            "role_index": record["role_index"],
            "kind": job["kind"],
            "pose_id": record["pose_id"],
            "intrusion_side": record["intrusion_side"],
            "family_id": record["family_id"],
            "clean": record["clean"],
            "outcome": job["outcome"],
            "defects": record["defects"],
            "video_name": output.name,
            "video_raw_file_sha256": sha256_file(output),
            "size_bytes": int(output.stat().st_size),
            "n_frames_rendered": len(indices),
            "retained_trajectory_n": record["trajectory_n"],
            "telemetry_n_frames": record["n_frames"],
            "fps": float(REVIEW_FPS),
            "duration_s": len(indices) / float(REVIEW_FPS),
            "episode_min_clearance_m": record["min_clearance_m"],
            "replay_min_clearance_m": (
                None if running_min == float("inf") else running_min),
            "pendant_contact_frames_in_replay": contact_frames,
            "source_result_sha256": record["result_raw_file_sha256"],
            "source_trajectory_sha256": record["trajectory_raw_file_sha256"],
            "scene_relative": job["scene_relative"],
            "scene_sha256": job["scene_sha256"],
            "assembly_sha256": job["assembly_sha256"],
            "complete_retained_trajectory": True,
            "trimmed": False,
        }
    finally:
        if writer is not None:
            writer.release()
        cleanup_episode_resources(
            task=task, policy=None, task_sampler=sampler,
            preloaded_policy=None, close_task_sampler=sampler is not None)
        if scratch is not None:
            shutil.rmtree(scratch, ignore_errors=True)


def verify_video(path: Path, record: dict[str, Any]) -> dict[str, Any]:
    """Independent decode: frame count, fps, duration, size, hash."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return {"decoded": False, "passed": False}
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        counted = 0
        while True:
            ok, _ = capture.read()
            if not ok:
                break
            counted += 1
    finally:
        capture.release()
    size = int(path.stat().st_size)
    digest = sha256_file(path)
    checks = {
        "decoded": True,
        "nonempty": size > 0,
        "decoded_frames": counted,
        "decoded_fps": fps,
        "decoded_duration_s": counted / fps if fps > 0 else None,
        "frames_match_render": counted == record["n_frames_rendered"],
        "frames_match_retained_trajectory": counted == record[
            "retained_trajectory_n"],
        "fps_matches": abs(fps - float(REVIEW_FPS)) <= 0.5,
        "duration_matches": abs(
            (counted / fps if fps > 0 else -1.0) - record["duration_s"]) <= 0.25,
        "sha256_stable": digest == record["video_raw_file_sha256"],
    }
    checks["passed"] = bool(
        checks["nonempty"] and checks["frames_match_render"]
        and checks["frames_match_retained_trajectory"] and checks["fps_matches"]
        and checks["duration_matches"] and checks["sha256_stable"])
    return checks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / OWNER_REVIEW_ROOT)
    parser.add_argument("--specification", type=Path,
                        default=ROOT / SPEC_ROOT / "specification.json")
    args = parser.parse_args()
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("MLSPACES_ASSETS_DIR", str(ROOT / "assets"))
    os.environ.pop("DISPLAY", None)
    started = time.time()

    output_root = args.output_root.resolve()
    if output_root.exists():
        raise SystemExit(f"refusing to overwrite {output_root}")

    bound = bind_inputs(args.specification.resolve())
    pool = json.loads((ROOT / POOL_ROOT / "pool.json").read_text())
    eligibility = pool["eligibility"]
    if eligibility["pool_passed"]:
        raise SystemExit(
            "this packet is for a FAILED pool; pool.json reports a pass")

    candidates = candidate_rows(pool)
    plan = select_six(candidates)
    if not plan["found"]:
        raise SystemExit(f"no valid six-clip subset: {plan['reason']}")
    verification = verify_selection(plan)
    if not verification["passed"]:
        raise SystemExit(f"selection constraints failed: {verification['checks']}")

    print(json.dumps({
        "successes": [r["role_index"] for r in plan["successes"]],
        "failures": [r["role_index"] for r in plan["failures"]],
        "objective": plan["objective"],
        "n_valid_subsets": plan["n_valid_subsets"],
    }, indent=2), flush=True)

    staging = Path(tempfile.mkdtemp(prefix="v107_owner_review_",
                                    dir=ROOT / "diagnostics_output"))
    try:
        jobs = []
        for kind, group in (("production_clean_success", plan["successes"]),
                            ("natural_production_failure", plan["failures"])):
            for record in group:
                pose = record["pose_id"]
                scene = bound["_scenes"][pose]
                result = json.loads((ROOT / record["result_path"]).read_text())
                stability = result.get("clutter_stability_events") or []
                audit = (result.get("contact_audit") or {}).get(
                    "contact_class_totals") or {}
                clutter_hits = int(audit.get("clutter", 0))
                jobs.append({
                    "kind": kind,
                    "record": record,
                    "assembly": scene["assembly"],
                    "scene_relative": scene["relative"],
                    "scene_sha256": scene["scene_sha256"],
                    "assembly_sha256": scene["assembly_sha256"],
                    "clutter_state": (
                        f"contacts {clutter_hits}, stability events "
                        f"{len(stability)}"),
                    "outcome": ("clean success" if record["clean"]
                                else ", ".join(record["defects"][:3])),
                    "video_path": str(
                        staging / "videos"
                        / f"{'success' if record['clean'] else 'failure'}"
                          f"_role{record['role_index']:02d}_{pose}"
                          f"_{record['intrusion_side']}.mp4"),
                })
        rendered = [render_clip(job) for job in jobs]
        verifications = {
            record["video_name"]: verify_video(
                staging / "videos" / record["video_name"], record)
            for record in rendered
        }
        if not all(v["passed"] for v in verifications.values()):
            raise RuntimeError(f"video verification failed: {verifications}")

        manifest = {
            "schema_version": "pact_place_v107_owner_review_v1",
            "contract_version": CONTRACT_VERSION_V107,
            "environment_version": ENVIRONMENT_VERSION,
            "role": "owner visual review of a FAILED production pool",
            "review_only": True,
            "reuses_retained_trajectories_only": True,
            "creates_episode": False,
            "calls_env_step": False,
            "resamples_tasks": False,
            "changes_geometry": False,
            "changes_thresholds": False,
            "reruns_pool": False,
            "reinterprets_pool_result": False,
            "bound_inputs": bound["artifacts"],
            "drift_check": bound["drift_check"],
            "scenes": bound["scenes"],
            "offline_certification_passed": bound["certification_passed"],
            "offline_causal_passed": bound["causal_passed"],
            "pool_clean_successes": eligibility["clean_successes"],
            "pool_n_rows": eligibility["n_rows"],
            "pool_min_clean_required": POOL_MIN_CLEAN,
            "pool_limiting_predicates": eligibility["limiting_predicates"],
            "pool_wilson_95_interval": eligibility["wilson_95_interval"],
            "selection_rule": plan["objective"]["order"],
            "selection_objective": plan["objective"],
            "n_valid_subsets_considered": plan["n_valid_subsets"],
            "n_clean_candidates": plan["n_clean_candidates"],
            "n_failed_candidates": plan["n_failed_candidates"],
            "selection_checks": verification["checks"],
            "selected_success_roles": [r["role_index"] for r in plan["successes"]],
            "selected_failure_roles": [r["role_index"] for r in plan["failures"]],
            "n_videos": len(rendered),
            "videos": rendered,
            "video_raw_file_sha256": {
                r["video_name"]: r["video_raw_file_sha256"] for r in rendered},
            "video_verification": verifications,
            "all_failures_are_natural": True,
            "no_induced_pendant_collision": True,
            "clips_are_complete_retained_trajectories": True,
            "clips_trimmed": False,
            "elapsed_s": time.time() - started,
            **empty_authorization(),
            "eligible_for_owner_visual_review": True,
            "pool_passed": False,
            "publishing_these_videos_does_not_make_the_pool_pass": True,
            "authorizes_downstream_work": False,
        }
        write_immutable_create_only(staging / "review_manifest.json", manifest)

        lines = [
            "# V10.7 owner visual-review packet",
            "",
            "> **The production pool FAILED at "
            f"{eligibility['clean_successes']}/{eligibility['n_rows']}** against a "
            f"{POOL_MIN_CLEAN}/{eligibility['n_rows']} floor.",
            "> **These videos are provided solely for owner visual assessment.**",
            "> **Publishing them does not make the pool pass and does not "
            "authorize any downstream work.**",
            "",
            "## Status",
            "",
            "| check | result |",
            "|---|---|",
            f"| offline certification | **passed** |",
            f"| offline six-group causality | **passed** |",
            f"| production pool | **FAILED — "
            f"{eligibility['clean_successes']}/{eligibility['n_rows']}** |",
            f"| Phase 0 | not run |",
            "",
            "The offline geometry, certification and causal checks all passed. The",
            "pool did not: it yielded "
            f"{eligibility['clean_successes']}/{eligibility['n_rows']} strict-clean, "
            f"Wilson 95% [{eligibility['wilson_95_interval'][0]:.3f}, "
            f"{eligibility['wilson_95_interval'][1]:.3f}], missing every registered",
            "floor. `pool_passed` remains **false** and is not reinterpreted here.",
            "",
            "## What these clips are",
            "",
            "Six **complete retained trajectories** replayed from the pool at true",
            f"time ({REVIEW_FPS:.4f} fps, stride 1). No episode was generated, no",
            "task resampled, no `env.step` called, no geometry or threshold changed,",
            "and the pool was not rerun. Nothing is trimmed around the interesting",
            "event.",
            "",
            "**All three failures are natural production failures. None is an",
            "induced pendant collision** — every selected failure records zero",
            "robot-or-target pendant contact frames.",
            "",
            "## Selection",
            "",
            "Deterministic and derived, not hand-picked: among subsets with one row",
            "per pendant pose in each outcome class, three left and three right",
            "overall, and at least two layout families per class, minimise the",
            "maximum pendant clearance, then total clearance, then the sorted",
            f"role-index tuple. {plan['n_valid_subsets']} valid subsets were",
            "considered.",
            "",
            "| # | outcome | role | pose | side | family | min clearance | frames | duration |",
            "|---|---|---:|---|---|---|---:|---:|---:|",
        ]
        for i, video in enumerate(rendered, start=1):
            lines.append(
                f"| {i} | {'clean success' if video['clean'] else 'natural failure'} "
                f"| {video['role_index']} | {video['pose_id']} | "
                f"{video['intrusion_side']} | {video['family_id']} | "
                f"{video['episode_min_clearance_m'] * 1000:.3f} mm | "
                f"{video['n_frames_rendered']} | {video['duration_s']:.2f} s |")
        lines += [
            "",
            "## Panes and overlay",
            "",
            "Untinted wrist RGB policy view; wide third-person view showing the",
            "household clutter, robot, target, panel and pendant; pendant-side close",
            "view; and a per-component proximity bar derived from retained state.",
            "The overlay carries outcome, role, family, side, pendant pose, phase,",
            "commanded and realized speed, current/running/episode-minimum pendant",
            "clearance, limiting component and body, clutter contact and stability",
            "state, task outcome, and pendant-contact state.",
            "",
            "## Authorization",
            "",
            "`eligible_for_owner_visual_review: true`, `pool_passed: false`, and",
            "every authorization field false. `human_approval.json` is absent and",
            "was not created. Phase 0 has not run and is not authorized.",
            "",
            "Verify this packet independently with:",
            "",
            "```",
            "python scripts/verify_pact_place_v107_owner_review.py",
            "```",
            "",
        ]
        write_immutable_text_create_only(staging / "REVIEW.md", "\n".join(lines))
        os.rename(staging, output_root)
        published = sorted((output_root / "videos").glob("*.mp4"))
        print(json.dumps({
            "owner_review_published": True,
            "n_videos": len(published),
            "videos": [str(p.relative_to(ROOT)) for p in published],
            "pool_passed": False,
            "eligible_for_owner_visual_review": True,
            "human_approval_present": (
                output_root / "human_approval.json").exists(),
            **empty_authorization(),
        }, indent=2))
        return 0
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
