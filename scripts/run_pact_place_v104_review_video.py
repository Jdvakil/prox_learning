#!/usr/bin/env python3
"""V10.4 Step-3: exactly six owner-review videos, then stop.

Three production successes and three failures, chosen deterministically. Real
production failures are used first; diagnostic negative controls fill any
shortfall and are labelled as such in the frame, the filename, and the manifest.

True time at the 66 ms control period with frame stride 1. The wrist pane is
never tinted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "scripts", ROOT / "submodules" / "molmospaces"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from pact_place_corridor_contract import sha256_file  # noqa: E402
from pact_place_v104_contract import (  # noqa: E402
    CAUSAL_ROOT,
    CONTRACT_VERSION,
    N_REVIEW_VIDEOS,
    PRODUCTION_ROOT,
    REVIEW_FPS,
    REVIEW_FRAME_STRIDE,
    REVIEW_MIN_CLEARANCE_M,
    REVIEW_ROOT,
    SAMPLER_CLASS,
    empty_authorization,
    implementation_sha256,
    is_clean_success,
    row_defects,
    write_immutable_create_only,
)
from pact_place_v104_geometry import (  # noqa: E402
    ALL_GEOMS_V104,
    PENDANT_BODY_V104,
    SCENE_XML_RELATIVE_V104,
    production_assembly,
)
from run_pact_place_expert_screen import _make_config, _result_path  # noqa: E402

SCENE_XML = ROOT / SCENE_XML_RELATIVE_V104
CONTROL_BANNER = (
    "DIAGNOSTIC NEGATIVE CONTROL - NOT PRODUCTION GEOMETRY - NOT AN EPISODE"
)
# Frozen inward-shift grid for diagnostic controls only. Never a production search.
CONTROL_SHIFT_GRID_M = tuple(round(0.001 * i, 3) for i in range(0, 161))
CONTROL_PENETRATION_BAND_M = (0.005, 0.030)
CONTROL_ORDER = ("left_lobe_contact", "right_lobe_contact", "stem_contact")
CONTROL_COMPONENT = {
    "left_lobe_contact": "lobe_0",
    "right_lobe_contact": "lobe_1",
    "stem_contact": "stem_0",
}
PENDANT_CAM_POS = np.asarray([0.16, -0.02, 1.30], dtype=float)
PENDANT_CAM_TARGET = np.asarray([0.78, 0.00, 1.02], dtype=float)
PENDANT_CAM_FOV = 52.0


def _put(frame, text, xy, scale, color, thick=1):
    cv2.putText(frame, str(text), xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def video_duration_s(n_frames: int, *, fps: float = REVIEW_FPS) -> float:
    return float(int(n_frames)) / float(fps)


def select_clips(manifest: dict[str, Any], production_root: Path) -> dict[str, Any]:
    """Deterministic three-success / three-failure selection."""
    rows = {int(row["role_index"]): row for row in manifest["rows"]}
    clean, failures = [], []
    for entry in sorted(manifest["results"], key=lambda item: int(item["role_index"])):
        role = int(entry["role_index"])
        result = json.loads(_result_path(production_root, rows[role]).read_text())
        frames = result.get("pact_v104_frame_telemetry") or {}
        record = {
            "role_index": role,
            "intrusion_side": str(rows[role]["intrusion_side"]),
            "min_clearance_m": frames.get("min_clearance_m"),
            "status": result.get("status"),
            "defects": row_defects(result, min_clearance_m=REVIEW_MIN_CLEARANCE_M),
        }
        if is_clean_success(result, min_clearance_m=REVIEW_MIN_CLEARANCE_M):
            clean.append(record)
        elif result.get("status") == "complete":
            failures.append(record)
    def smallest(pool):
        return sorted(
            pool,
            key=lambda item: (
                float("inf") if item["min_clearance_m"] is None else float(item["min_clearance_m"]),
                item["role_index"],
            ),
        )
    successes = []
    for side in ("left", "right"):
        pool = smallest([item for item in clean if item["intrusion_side"] == side])
        if pool:
            successes.append(pool[0])
    remaining = [item for item in smallest(clean) if item not in successes]
    if remaining:
        successes.append(remaining[0])
    natural = sorted(failures, key=lambda item: item["role_index"])[:3]
    controls = list(CONTROL_ORDER)[: max(0, 3 - len(natural))]
    return {
        "successes": successes[:3],
        "natural_failures": natural,
        "controls": controls,
        "n_clean": len(clean),
        "n_natural_failures": len(failures),
    }


def find_control_shift(row: dict[str, Any], seed_u32: int, component_name: str,
                       steps: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Smallest frozen-grid inward shift giving 5-30 mm exact penetration.

    Diagnostic-only. The whole frozen grid is evaluated in order and every
    tested shift is recorded. The production XML is never modified: the grid is
    scored by translating the assembly's boxes inside one throwaway task, and
    only the chosen shift is later compiled into a separate diagnostic scene.

    Cheap by construction: a rigid inward shift of ``s`` can improve any frame's
    clearance by at most ``s``, so a frame whose unshifted clearance already
    exceeds ``min_clearance + s`` can never be the worst frame at that shift.
    Only the frames that survive that bound are measured exactly.
    """
    import mujoco
    from molmo_spaces.data_generation.pipeline import cleanup_episode_resources
    from pact_geom_distance import true_distance
    from pact_place_v104_clearance import (
        assembly_boxes,
        frame_clearances,
        geom_shape_cache,
        robot_collision_geom_ids,
        target_collision_geom_ids,
    )
    from run_pact_place_v7_replay_videos import apply_recorded_qpos

    assembly = production_assembly()
    component = next(i for i in assembly["components"] if i["name"] == component_name)
    sign = -1.0 if float(component["center_m"][1]) < 0 else 1.0

    def shifted_assembly(shift: float) -> dict[str, Any]:
        moved = dict(assembly)
        moved["components"] = [
            {
                **item,
                "center_m": [
                    item["center_m"][0],
                    round(item["center_m"][1] - sign * float(shift), 9),
                    item["center_m"][2],
                ],
            }
            for item in assembly["components"]
        ]
        return moved

    scratch = Path(tempfile.mkdtemp(prefix="v104_control_"))
    task = sampler = None
    tested: list[dict[str, Any]] = []
    try:
        config = _make_config(scratch / "d.json", scene_xml=SCENE_XML,
                              sampler_class=SAMPLER_CLASS)
        sampler = config.task_sampler_config.task_sampler_class(config)
        sampler.seed_task_sampling(int(seed_u32))
        sampler.set_pact_manifest_row(row)
        task = sampler.sample_task(house_index=int(row["scene_template_house_index"]))
        env = task.env
        model, data = env.current_model, env.current_data
        probe = robot_collision_geom_ids(model) + target_collision_geom_ids(task)
        cache = geom_shape_cache(model, probe)
        gid = int(model.geom(component["geom"]).id)
        saved_pos = np.asarray(model.geom_pos[gid], dtype=float).copy()

        # Pass 1: unshifted per-frame clearance to this component.
        base_boxes = [b for b in assembly_boxes(assembly) if b["name"] == component_name]
        per_frame = []
        for index, step in enumerate(steps):
            apply_recorded_qpos(env, step["qpos"])
            mujoco.mj_forward(model, data)
            value = frame_clearances(model, data, base_boxes, probe, cache)["min_m"]
            per_frame.append(float(value) if value is not None else float("inf"))
        base_min = float(min(per_frame))
        order = sorted(range(len(steps)), key=lambda i: per_frame[i])

        try:
            for shift in CONTROL_SHIFT_GRID_M:
                bound = base_min + float(shift) + 1e-6
                # Sound bound, no cap: a rigid inward shift of s changes any
                # frame's clearance by at most s, so a frame above this bound
                # cannot be the worst frame at this shift.
                candidates = [i for i in order if per_frame[i] <= bound]
                worst = 0.0
                worst_step = None
                if candidates:
                    model.geom_pos[gid] = saved_pos - np.array([0.0, sign * float(shift), 0.0])
                    for index in candidates:
                        apply_recorded_qpos(env, steps[index]["qpos"])
                        mujoco.mj_forward(model, data)
                        distance = float(true_distance(model, data, probe, [gid]))
                        if distance < worst:
                            worst, worst_step = distance, index
                    model.geom_pos[gid] = saved_pos
                penetration = float(-worst)
                tested.append({
                    "shift_m": float(shift),
                    "max_penetration_m": penetration,
                    "n_frames_measured": len(candidates),
                })
                if CONTROL_PENETRATION_BAND_M[0] <= penetration <= CONTROL_PENETRATION_BAND_M[1]:
                    return {
                        "found": True,
                        "shift_m": float(shift),
                        "penetration_m": penetration,
                        "worst_step": worst_step,
                        "component": component_name,
                        "unshifted_min_clearance_m": base_min,
                        "tested": tested,
                        "assembly": shifted_assembly(shift),
                    }
        finally:
            model.geom_pos[gid] = saved_pos
            mujoco.mj_forward(model, data)
        return {"found": False, "component": component_name,
                "unshifted_min_clearance_m": base_min, "tested": tested}
    finally:
        cleanup_episode_resources(task=task, policy=None, task_sampler=sampler,
                                  preloaded_policy=None, close_task_sampler=sampler is not None)
        shutil.rmtree(scratch, ignore_errors=True)


def render_clip(job: dict[str, Any]) -> dict[str, Any]:
    """Three-pane, stride-1, true-time clip with burned-in telemetry."""
    import mujoco
    from molmo_spaces.data_generation.pipeline import cleanup_episode_resources
    from pact_place_v104_clearance import (
        assembly_boxes,
        frame_clearances,
        geom_shape_cache,
        robot_collision_geom_ids,
        target_collision_geom_ids,
    )
    from pact_place_v104_runtime import pendant_contact_state
    from run_pact_place_v6c_replay_videos import (
        PANE_WH,
        THIRD_PERSON_FOV,
        WRIST_FOV,
        _resize_pane,
        apply_recorded_qpos,
        third_person_pose,
        wrist_camera_pose,
    )

    row = job["row"]
    steps = json.loads(Path(job["trajectory_path"]).read_text())["steps"]
    result = json.loads(Path(job["result_path"]).read_text())
    is_control = bool(job.get("is_control"))
    assembly = job.get("assembly") or production_assembly()
    scene_path = Path(job["scene_xml"])
    output = Path(job["video_path"])
    output.parent.mkdir(parents=True, exist_ok=True)

    task = sampler = scratch = None
    writer = None
    try:
        scratch = Path(tempfile.mkdtemp(prefix="v104_render_"))
        config = _make_config(scratch / "d.json", scene_xml=scene_path,
                              sampler_class=SAMPLER_CLASS)
        sampler = config.task_sampler_config.task_sampler_class(config)
        sampler.seed_task_sampling(int(result["selected_seed"]["seed_u32"]))
        sampler.set_pact_manifest_row(row)
        task = sampler.sample_task(house_index=int(row["scene_template_house_index"]))
        env = task.env
        model, data = env.current_model, env.current_data
        probe = robot_collision_geom_ids(model) + target_collision_geom_ids(task)
        cache = geom_shape_cache(model, probe)
        boxes = assembly_boxes(assembly)
        pendant_ids = [int(model.geom(name).id) for name in ALL_GEOMS_V104]
        side_delta = PENDANT_CAM_TARGET - PENDANT_CAM_POS
        side_fwd = side_delta / np.linalg.norm(side_delta)
        side_up = np.asarray([0.0, 0.0, 1.0], dtype=float)
        writer = cv2.VideoWriter(
            str(output), cv2.VideoWriter_fourcc(*"mp4v"), float(REVIEW_FPS),
            (PANE_WH[0] * 3, PANE_WH[1]),
        )
        indices = list(range(0, len(steps), int(REVIEW_FRAME_STRIDE)))
        if indices != list(range(len(steps))):
            raise RuntimeError("V10.4 review must render every control frame")
        contact_frames: list[int] = []
        for index in indices:
            step = steps[index]
            apply_recorded_qpos(env, step["qpos"])
            mujoco.mj_forward(model, data)
            report = frame_clearances(model, data, boxes, probe, cache)
            contact = pendant_contact_state(model, data, pendant_ids)
            if contact["contact"]:
                contact_frames.append(index)
            w_pos, w_fwd, w_up = wrist_camera_pose(env)
            wrist = _resize_pane(np.asarray(
                env._render_frame(w_pos, w_fwd, w_up, WRIST_FOV, segmentation=False)))
            tp_pos, tp_fwd, tp_up = third_person_pose(env)
            third = _resize_pane(np.asarray(
                env._render_frame(tp_pos, tp_fwd, tp_up, THIRD_PERSON_FOV, segmentation=False)))
            pendant = _resize_pane(np.asarray(
                env._render_frame(PENDANT_CAM_POS, side_fwd, side_up, PENDANT_CAM_FOV,
                                  segmentation=False)))
            frame = cv2.cvtColor(np.concatenate([wrist, third, pendant], axis=1),
                                 cv2.COLOR_RGB2BGR)
            width, height = PANE_WH
            shade = frame.copy()
            header_h = 128 if is_control else 106
            cv2.rectangle(shade, (0, 0), (frame.shape[1], header_h), (0, 0, 0), -1)
            cv2.rectangle(shade, (0, height - 30), (frame.shape[1], height), (0, 0, 0), -1)
            frame = cv2.addWeighted(shade, 0.70, frame, 0.30, 0.0)
            minimum = report["min_m"]
            limiting = report["limiting"] or {}
            speed = step.get("commanded_speed_m_s")
            realized = step.get("realized_tcp_speed_m_s")
            status = "DIAGNOSTIC NEGATIVE CONTROL" if is_control else (
                "PRODUCTION CLEAN" if job.get("clean") else "PRODUCTION FAILURE")
            colour = (60, 60, 255) if (is_control or contact["contact"]) else (
                (120, 255, 120) if job.get("clean") else (60, 160, 255))
            wall = float(index) / float(REVIEW_FPS)
            lines = [
                f"{status}  {job['label']}  side {row['intrusion_side']}  "
                f"frame {index}/{len(steps) - 1}  sim {float(step.get('sim_time_s') or 0.0):7.3f}s  "
                f"video {wall:6.3f}s  ({REVIEW_FPS:.4f} fps, stride 1, true time)",
                f"phase {step.get('policy_phase')}  primitive {step.get('primitive_index')} "
                f"segment {step.get('segment_index')} {step.get('segment_name')}  "
                f"commanded {'--' if speed is None else format(float(speed), '.3f')} m/s  "
                f"realized {'--' if realized is None else format(float(realized), '.3f')} m/s",
                f"pendant clearance {'--' if minimum is None else format(float(minimum), '.4f')} m"
                f"   limiting {limiting.get('component')} <-> {limiting.get('probe_body')}",
                f"pendant contact: {'YES' if contact['contact'] else 'no'}   "
                f"cumulative contact frames {len(contact_frames)}   "
                f"target held: {'yes' if step.get('target_held') else 'no'}   "
                f"outcome {job.get('outcome')}",
            ]
            for line_index, line in enumerate(lines):
                cv2.putText(frame, line, (12, 24 + line_index * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                            colour if line_index == 0 else (230, 230, 230), 1, cv2.LINE_AA)
            if is_control:
                cv2.rectangle(frame, (0, header_h - 24), (frame.shape[1], header_h), (0, 0, 200), -1)
                cv2.putText(frame, CONTROL_BANNER, (12, header_h - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)
            for pane_index, label in enumerate((
                "wrist (RGB policy view, untinted)", "third-person task view",
                "pendant view: both lobes, stems, crossbar",
            )):
                cv2.putText(frame, label, (pane_index * width + 12, height - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (245, 245, 245), 1, cv2.LINE_AA)
            writer.write(frame)
        writer.release()
        writer = None
        return {
            "label": job["label"],
            "kind": job["kind"],
            "role_index": int(row["role_index"]),
            "intrusion_side": str(row["intrusion_side"]),
            "video_path": str(output.relative_to(ROOT)),
            "video_sha256": sha256_file(output),
            "n_frames": len(indices),
            "fps": float(REVIEW_FPS),
            "frame_stride": int(REVIEW_FRAME_STRIDE),
            "duration_s": video_duration_s(len(indices)),
            "true_time": True,
            "wrist_pane_untinted": True,
            "contact_frames": contact_frames[:16],
            "n_contact_frames": len(contact_frames),
            "is_diagnostic_negative_control": is_control,
            "control_banner": CONTROL_BANNER if is_control else None,
            "scene_xml": str(scene_path),
        }
    finally:
        if writer is not None:
            writer.release()
        cleanup_episode_resources(task=task, policy=None, task_sampler=sampler,
                                  preloaded_policy=None, close_task_sampler=sampler is not None)
        if scratch is not None:
            shutil.rmtree(scratch, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--production-root", type=Path, default=ROOT / PRODUCTION_ROOT)
    parser.add_argument("--causal", type=Path, default=ROOT / CAUSAL_ROOT / "causal.json")
    parser.add_argument("--output-root", type=Path, default=ROOT / REVIEW_ROOT)
    args = parser.parse_args()
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("MLSPACES_ASSETS_DIR", str(ROOT / "assets"))
    os.environ.pop("DISPLAY", None)

    production_root = args.production_root.resolve()
    manifest = json.loads((production_root / "production_manifest.json").read_text())
    if not manifest["eligibility"]["production_pack_passed"]:
        raise SystemExit("the six-row production pack did not pass; no review packet")
    causal = json.loads(args.causal.resolve().read_text())
    if not causal.get("causal_passed"):
        raise SystemExit("the panel causal check did not pass; no review packet")

    output_root = args.output_root.resolve()
    (output_root / "videos").mkdir(parents=True, exist_ok=True)
    rows = {int(row["role_index"]): row for row in manifest["rows"]}
    plan = select_clips(manifest, production_root)

    jobs: list[dict[str, Any]] = []
    for record in plan["successes"]:
        row = rows[record["role_index"]]
        path = _result_path(production_root, row)
        jobs.append({
            "kind": "production_clean", "clean": True,
            "label": f"row{record['role_index']:02d}-{record['intrusion_side']}-clean",
            "row": row, "result_path": str(path),
            "trajectory_path": str(path.parent / "trajectory.json"),
            "scene_xml": str(SCENE_XML), "is_control": False,
            "outcome": "clean success",
            "video_path": str(output_root / "videos"
                              / f"success_{record['role_index']:02d}_{record['intrusion_side']}.mp4"),
            "selection_reason": "smallest exact pendant clearance among strict-clean rows",
            "min_clearance_m": record["min_clearance_m"],
        })
    for record in plan["natural_failures"]:
        row = rows[record["role_index"]]
        path = _result_path(production_root, row)
        jobs.append({
            "kind": "production_failure", "clean": False,
            "label": f"row{record['role_index']:02d}-{record['intrusion_side']}-failure",
            "row": row, "result_path": str(path),
            "trajectory_path": str(path.parent / "trajectory.json"),
            "scene_xml": str(SCENE_XML), "is_control": False,
            "outcome": ", ".join(record["defects"][:3]) or "failure",
            "video_path": str(output_root / "videos"
                              / f"failure_{record['role_index']:02d}_{record['intrusion_side']}.mp4"),
            "selection_reason": "natural production failure, lowest role index first",
            "min_clearance_m": record["min_clearance_m"],
        })
    control_records: list[dict[str, Any]] = []
    clean_by_side: dict[str, list[dict[str, Any]]] = {"left": [], "right": []}
    for record in plan["successes"]:
        clean_by_side[record["intrusion_side"]].append(record)
    for control in plan["controls"]:
        component = CONTROL_COMPONENT[control]
        # Deterministic, side-matched source: the smallest-clearance strict-clean
        # row on the side the control targets. Using an opposite-side row would
        # put the control component far from the arm for no reason.
        side = "right" if control == "right_lobe_contact" else "left"
        pool = clean_by_side.get(side) or plan["successes"]
        source = sorted(
            pool,
            key=lambda item: (
                float("inf") if item["min_clearance_m"] is None
                else float(item["min_clearance_m"]),
                item["role_index"],
            ),
        )[0]
        row = rows[source["role_index"]]
        path = _result_path(production_root, row)
        result = json.loads(path.read_text())
        steps = json.loads((path.parent / "trajectory.json").read_text())["steps"]
        found = find_control_shift(
            row, int(result["selected_seed"]["seed_u32"]), component, steps
        )
        control_records.append({"control": control, **{k: v for k, v in found.items()
                                                       if k not in ("assembly", "tested")},
                                "n_shifts_tested": len(found["tested"]),
                                "tested_shifts_m": [t["shift_m"] for t in found["tested"]]})
        if not found["found"]:
            stop = {
                "schema_version": "pact_place_v104_review_control_stop_v1",
                "contract_version": CONTRACT_VERSION,
                "stop_reason": "diagnostic_control_cannot_reach_registered_penetration_band",
                "control": control,
                "component": component,
                "source_role_index": int(row["role_index"]),
                "source_intrusion_side": str(row["intrusion_side"]),
                "registered_band_m": list(CONTROL_PENETRATION_BAND_M),
                "grid_m": [CONTROL_SHIFT_GRID_M[0], CONTROL_SHIFT_GRID_M[-1],
                           len(CONTROL_SHIFT_GRID_M)],
                "unshifted_min_clearance_m": found.get("unshifted_min_clearance_m"),
                "max_penetration_reached_m": max(
                    (t["max_penetration_m"] for t in found["tested"]), default=0.0
                ),
                "n_shifts_tested": len(found["tested"]),
                "tested": found["tested"],
                "grid_extended": False,
                "production_scene_modified": False,
                "production_geometry_changed": False,
                "note": (
                    "The plan requires stopping rather than extending the grid. "
                    "Steps 0, 1 and 2 passed; only the diagnostic negative "
                    "control could not be made to touch."
                ),
                "n_natural_production_failures": plan["n_natural_failures"],
                "n_production_clean": plan["n_clean"],
                **empty_authorization(),
            }
            write_immutable_create_only(
                output_root / "control_shortfall_stop.json", stop
            )
            print(json.dumps({
                "stopped": True,
                "stop_reason": stop["stop_reason"],
                "control": control,
                "max_penetration_reached_m": stop["max_penetration_reached_m"],
                "registered_band_m": stop["registered_band_m"],
                "videos_written": 0,
            }, indent=2))
            return 1
        scratch = Path(tempfile.mkdtemp(prefix="v104_control_scene_"))
        base = ROOT / "submodules/molmospaces/molmo_spaces/data_generation/custom_scenes"
        for name in ("pact_place_corridor_v3.xml", "pact_place_corridor_v5.xml"):
            shutil.copy(base / name, scratch / name)
        from pact_place_v104_geometry import scene_xml_text

        control_scene = scratch / f"control_{control}.xml"
        control_scene.write_text(scene_xml_text(found["assembly"]))
        jobs.append({
            "kind": "diagnostic_negative_control", "clean": False,
            "label": f"control-{control}",
            "row": row, "result_path": str(path),
            "trajectory_path": str(path.parent / "trajectory.json"),
            "scene_xml": str(control_scene), "is_control": True,
            "assembly": found["assembly"],
            "outcome": f"{control} at {found['penetration_m']*1000:.1f} mm penetration",
            "video_path": str(output_root / "videos" / f"control_{control}.mp4"),
            "selection_reason": (
                f"deterministic control: smallest frozen-grid inward shift "
                f"({found['shift_m']:.3f} m) giving 5-30 mm penetration"
            ),
            "min_clearance_m": -float(found["penetration_m"]),
        })

    if len(jobs) != N_REVIEW_VIDEOS:
        raise SystemExit(f"expected exactly {N_REVIEW_VIDEOS} clips, built {len(jobs)}")
    videos = []
    for job in jobs:
        record = render_clip(job)
        record["selection_reason"] = job["selection_reason"]
        record["min_clearance_m"] = job["min_clearance_m"]
        videos.append(record)
        print(f"rendered {record['video_path']} ({record['n_frames']} frames, "
              f"{record['duration_s']:.2f}s)", flush=True)

    review_manifest = {
        "schema_version": "pact_place_v104_review_manifest_v1",
        "contract_version": CONTRACT_VERSION,
        "production_manifest_sha256": manifest.get("artifact_sha256"),
        "causal_sha256": causal.get("artifact_sha256"),
        "implementation_sha256": implementation_sha256(),
        "n_videos": len(videos),
        "n_production_clean": sum(1 for v in videos if v["kind"] == "production_clean"),
        "n_production_failure": sum(1 for v in videos if v["kind"] == "production_failure"),
        "n_diagnostic_negative_control": sum(
            1 for v in videos if v["kind"] == "diagnostic_negative_control"),
        "videos": videos,
        "selection": plan,
        "diagnostic_controls": control_records,
        "control_banner": CONTROL_BANNER,
        "control_shift_grid_m": [CONTROL_SHIFT_GRID_M[0], CONTROL_SHIFT_GRID_M[-1],
                                 len(CONTROL_SHIFT_GRID_M)],
        "control_penetration_band_m": list(CONTROL_PENETRATION_BAND_M),
        "production_scene_modified": False,
        "clean_rate_is_not_an_estimate": True,
        **empty_authorization(),
        "eligible_for_human_review": True,
    }
    digest = write_immutable_create_only(output_root / "review_manifest.json", review_manifest)
    lines = [
        "# PACT place V10.4 — owner review packet",
        "",
        "Six clips: three production successes and three failures/controls.",
        "Every clip is stride-1 at the 66 ms control period, so playback is true time.",
        "The wrist pane is the untinted policy view.",
        "",
        f"Review manifest SHA-256 `{digest}`.",
        "",
        "`clean_rate_is_not_an_estimate: true` — six rows are a qualification check.",
        "",
        "| # | kind | clip | side | frames | duration | min pendant clearance |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for index, video in enumerate(videos, start=1):
        clearance = video["min_clearance_m"]
        lines.append(
            f"| {index} | {video['kind']} | `{video['video_path']}` | "
            f"{video['intrusion_side']} | {video['n_frames']} | "
            f"{video['duration_s']:.2f} s | "
            f"{'n/a' if clearance is None else format(float(clearance), '+.4f') + ' m'} |"
        )
    lines += [
        "",
        "## What to confirm",
        "",
        "1. the pendant is visibly above and separated from the table clutter;",
        "2. lobes, stems, and crossbar remain static;",
        "3. production successes contain no pass-through or touch;",
        "4. negative-control contacts are visibly and numerically detected;",
        "5. the route is smooth and does not jump IK branches;",
        "6. the initial free-space motion is acceptably slower;",
        "7. overlays agree with visible motion and outcome.",
        "",
        "Clips labelled `diagnostic_negative_control` are **not** production episodes",
        "and carry a red in-frame banner. They validate visibility and scoring only.",
        "",
        "`human_approval.json` is absent and was not created. Phase 0 is unauthorized.",
        "",
    ]
    (output_root / "REVIEW.md").write_text("\n".join(lines))
    print(json.dumps({
        "n_videos": len(videos),
        "review_manifest_sha256": digest,
        "eligible_for_human_review": True,
        "human_approval_present": False,
        "authorizes_phase0": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
