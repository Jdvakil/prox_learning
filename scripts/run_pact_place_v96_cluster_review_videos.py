#!/usr/bin/env python3
"""Review clips for the V9.6 clustered hazards, one per layout family.

These are **replays with the clusters composited in**, on the frozen V9.5
trajectories.  The recorded expert never saw a cluster and never avoids one; no
physics is stepped, `mj_forward` only.  They are a design aid for judging by eye
whether a cluster of three tall vessels still reads as household clutter in the
wrist camera while being a slab to the skin.  They are not a rollout and not a
gate.

Nothing is tinted.  The wrist pane shows exactly what the RGB policy would see,
because "does it read as clutter" is the question these clips exist to answer.

The burned-in telemetry is measured live, per frame:

* whether either cluster falls inside the wrist camera's frustum;
* the cluster's largest image-plane span across the 40 skin sensors, in pixels
  of the 8x8 grid, and how many sensors clear 2 px -- the same instrument W1
  validated against the raw counterfactual to r = 0.99997.

The configuration rendered is the one W2 **rejected**: it closes the arm's
corridor lane under one panel side, so on that side the frozen path runs through
the cluster.  The banner says so on every frame.

Nothing here authorizes a gate, collection, or V1b.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MOLMO = ROOT / "submodules" / "molmospaces"
for _path in (ROOT / "scripts", MOLMO):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import pact_place_v96_cluster_contract as v96  # noqa: E402
import pact_skin_resolvability as psr  # noqa: E402
from pact_place_corridor_contract import sha256_file  # noqa: E402
from pact_place_v9_contract import sha256_payload  # noqa: E402
from run_pact_place_v9_v0c3_causal_proximity import SCENE_XML, _find_episode_dir  # noqa: E402
from run_pact_place_v96_cluster_causal_proximity import (  # noqa: E402
    _build_row,
    _clutter_qpos_start,
)

DEFAULT_SMOKE_ROOT = ROOT / "diagnostics_output" / "pact_place_v95_raw_smoke"
DEFAULT_CONFIGURATION = ROOT / "configs" / "pact_place_v96_w3_pipeline_validation.json"
DEFAULT_OUTPUT_ROOT = ROOT / "diagnostics_output" / "pact_place_v96_cluster_review"
PANE_WH = (480, 480)
FPS = 12
WRIST_FOV = 56.74
THIRD_PERSON_FOV = 60.0
REVIEW_CAM_POS = np.asarray([0.05, -0.60, 1.34], dtype=float)
REVIEW_CAM_TARGET = np.asarray([0.74, 0.06, 1.12], dtype=float)
REVIEW_CAM_FOV = 52.0
SUBTENSE_THRESHOLD_PX = 2.0


def _resize(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame)
    if frame.shape[:2] != (PANE_WH[1], PANE_WH[0]):
        frame = cv2.resize(frame, PANE_WH, interpolation=cv2.INTER_AREA)
    return frame.astype(np.uint8)


def _overlay(panes: list[np.ndarray], telemetry: dict[str, Any]) -> np.ndarray:
    width, height = PANE_WH
    frame = cv2.cvtColor(np.concatenate(panes, axis=1), cv2.COLOR_RGB2BGR)
    shade = frame.copy()
    cv2.rectangle(shade, (0, 0), (frame.shape[1], 118), (0, 0, 0), thickness=-1)
    cv2.rectangle(shade, (0, height - 64), (frame.shape[1], height), (0, 0, 0), thickness=-1)
    frame = cv2.addWeighted(shade, 0.70, frame, 0.30, 0.0)

    def put(text, xy, scale, color, thick=1):
        cv2.putText(frame, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

    put(
        f"V9.6 clustered hazards   {telemetry['family_id']}   panel {telemetry['side']}   "
        f"phase: {telemetry['phase']}   step {telemetry['step']}/{telemetry['n_steps'] - 1}",
        (12, 26), 0.52, (255, 255, 255), 2,
    )
    for row_index, role in enumerate(("inbound_cluster", "outbound_cluster")):
        block = telemetry[role]
        seen = block["max_span_px"] >= SUBTENSE_THRESHOLD_PX
        put(
            f"{role.replace('_', ' '):17s} span {block['realized_span_m']:.3f} m   "
            f"skin: max {block['max_span_px']:5.1f} px, {block['n_sensors_ge_2px']} sensors >= 2 px   "
            f"wrist camera: {'IN FOV' if block['in_wrist_fov'] else 'not visible'}",
            (12, 52 + row_index * 24), 0.44,
            (120, 235, 120) if seen else (185, 185, 210), 1,
        )
    put(
        "REPLAY with clusters composited in - the recorded expert never saw them, no physics stepped.  "
        "W2 REJECTED this pair: it closes the corridor lane under one panel side.",
        (12, 108), 0.40, (120, 200, 255), 1,
    )
    put("wrist (RGB policy view)", (12, height - 22), 0.44, (245, 245, 245), 1)
    put("third-person", (width + 12, height - 22), 0.44, (245, 245, 245), 1)
    put("corridor review", (2 * width + 12, height - 22), 0.44, (245, 245, 245), 1)
    put(
        f"resolving floor: a hazard needs ~0.25 m of contiguous width to clear 2 px at working range "
        f"(pixel pitch {psr.PIXEL_PITCH_COEFF:.4f} x R)",
        (12, height - 44), 0.40, (200, 200, 200), 1,
    )
    return frame


def _cluster_boxes(model, data, bodies: list[str]) -> list[tuple[np.ndarray, np.ndarray]]:
    from run_pact_place_v9_w1_resolvability import HazardSource

    boxes = []
    for body in bodies:
        low, high = HazardSource(model, "cluster", body).pose(data).aabb
        boxes.append((np.asarray(low), np.asarray(high)))
    return boxes


def _render(job: dict[str, Any]) -> dict[str, Any]:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("MLSPACES_ASSETS_DIR", str(ROOT / "assets"))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.pop("DISPLAY", None)

    import mujoco
    from molmo_spaces.data_generation.pipeline import cleanup_episode_resources
    from run_pact_place_expert_screen import _make_config
    from run_pact_place_v6c_replay_videos import third_person_pose, wrist_camera_pose
    from run_pact_place_v9_v0c_siting import _wrist_in_fov

    row = job["row"]
    result = json.loads(Path(job["result_path"]).read_text())
    steps = list(json.loads(Path(job["trajectory_path"]).read_text())["steps"])
    output = Path(job["video_path"])
    output.parent.mkdir(parents=True, exist_ok=True)

    task = sampler = writer = None
    scratch = Path(tempfile.mkdtemp(prefix="pact_v96_review_"))
    try:
        config = _make_config(
            scratch / "dummy.json", scene_xml=SCENE_XML, sampler_class=v96.SAMPLER_CLASS
        )
        # The proximity suite is only registered when the sub-step recorder is
        # configured; without it the skin telemetry would silently read zero.
        config.proximity_sensor_period_ms = 16.6667
        sampler = config.task_sampler_config.task_sampler_class(config)
        sampler.seed_task_sampling(int(result["selected_seed"]["seed_u32"]))
        sampler.set_pact_manifest_row(row)
        task = sampler.sample_task(house_index=int(row["scene_template_house_index"]))
        if task is None:
            raise RuntimeError("V9.6 sample_task returned None for the review clip")
        task.reset()
        env = task.env
        model, data = env.current_model, env.current_data

        clutter_start = _clutter_qpos_start(model)
        source_nq = len(steps[0]["qpos"])
        if clutter_start != source_nq - 8 * 7:
            raise RuntimeError(
                "V9.5 and V9.6 disagree on the shared arm/target qpos block: "
                f"{clutter_start} vs {source_nq}"
            )
        clutter_reset = np.asarray(data.qpos[clutter_start:], dtype=float).copy()
        panel_mocap_id = int(
            np.asarray(model.body(f"pact_intrusion_{job['side']}").mocapid).reshape(-1)[0]
        )
        panel_position = np.asarray(data.mocap_pos[panel_mocap_id], dtype=float).copy()

        palette_by_slot = {str(item["slot"]): item for item in row["pact_clutter_palette"]}
        cluster_bodies = {
            role: [
                f"pact_clutter_{item['slot']}/{palette_by_slot[item['slot']]['uid']}"
                for item in row["pact_clutter_palette"]
                if str(item["role"]) == role
            ]
            for role in v96.CLUSTER_ROLES
        }
        sensor_names = list(task._proximity_camera_names)
        if len(sensor_names) != 40 or len(set(sensor_names)) != 40:
            raise RuntimeError(f"expected 40 unique proximity cameras: {sensor_names}")
        sensor_cam_ids = [int(model.camera(f"robot_0/{name}").id) for name in sensor_names]
        wrist_id = int(model.camera("robot_0/gripper/wrist_camera").id)

        review_delta = REVIEW_CAM_TARGET - REVIEW_CAM_POS
        review_forward = review_delta / np.linalg.norm(review_delta)
        review_up = np.asarray([0.0, 0.0, 1.0], dtype=float)

        writer = cv2.VideoWriter(
            str(output), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (PANE_WH[0] * 3, PANE_WH[1])
        )
        stride = max(1, int(job["frame_stride"]))
        indices = list(range(0, len(steps), stride))
        if indices[-1] != len(steps) - 1:
            indices.append(len(steps) - 1)

        peak = {role: {"max_span_px": 0.0, "n_sensors_ge_2px": 0, "wrist_frames": 0}
                for role in v96.CLUSTER_ROLES}
        for step_index in indices:
            qpos = np.asarray(steps[step_index]["qpos"], dtype=float)
            data.qpos[:clutter_start] = qpos[:clutter_start]
            data.qpos[clutter_start:] = clutter_reset
            data.mocap_pos[panel_mocap_id] = panel_position
            mujoco.mj_forward(model, data)

            w_pos, w_fwd, w_up = wrist_camera_pose(env)
            tp_pos, tp_fwd, tp_up = third_person_pose(env)
            panes = [
                _resize(env._render_frame(w_pos, w_fwd, w_up, WRIST_FOV, segmentation=False)),
                _resize(env._render_frame(tp_pos, tp_fwd, tp_up, THIRD_PERSON_FOV, segmentation=False)),
                _resize(
                    env._render_frame(
                        REVIEW_CAM_POS, review_forward, review_up, REVIEW_CAM_FOV,
                        segmentation=False,
                    )
                ),
            ]

            cam_pos = np.asarray(data.cam_xpos[sensor_cam_ids], dtype=np.float64)[None]
            cam_xmat = np.asarray(data.cam_xmat[sensor_cam_ids], dtype=np.float64).reshape(
                1, -1, 3, 3
            )
            telemetry: dict[str, Any] = {
                "family_id": job["family_id"],
                "side": job["side"],
                "phase": str(steps[step_index].get("policy_phase") or "unknown"),
                "step": step_index,
                "n_steps": len(steps),
            }
            for role, bodies in cluster_bodies.items():
                boxes = _cluster_boxes(model, data, bodies)
                scored = psr.screen_candidate(cam_pos, cam_xmat, boxes, 4.0)
                spans = scored["image_span_px"][0]
                lows = np.min([box[0] for box in boxes], axis=0)
                highs = np.max([box[1] for box in boxes], axis=0)
                center, half = (highs + lows) / 2.0, (highs - lows) / 2.0
                in_fov = bool(_wrist_in_fov(model, data, wrist_id, center, half))
                block = {
                    "realized_span_m": float(
                        row["pact_clutter_layout"][role]["span_along_line_m"]
                    ),
                    "max_span_px": float(spans.max()),
                    "n_sensors_ge_2px": int((spans >= SUBTENSE_THRESHOLD_PX).sum()),
                    "in_wrist_fov": in_fov,
                }
                telemetry[role] = block
                peak[role]["max_span_px"] = max(peak[role]["max_span_px"], block["max_span_px"])
                peak[role]["n_sensors_ge_2px"] = max(
                    peak[role]["n_sensors_ge_2px"], block["n_sensors_ge_2px"]
                )
                peak[role]["wrist_frames"] += int(in_fov)

            writer.write(_overlay(panes, telemetry))

        writer.release()
        writer = None
        return {
            "family_id": job["family_id"],
            "side": job["side"],
            "source_episode_id": job["source_episode_id"],
            "video_path": str(output.relative_to(ROOT)),
            "video_sha256": sha256_file(output),
            "frames_rendered": len(indices),
            "frame_stride": stride,
            "fps": FPS,
            "n_trajectory_steps": len(steps),
            "peak_telemetry": peak,
            "wrist_visible_fraction": {
                role: round(peak[role]["wrist_frames"] / len(indices), 4)
                for role in v96.CLUSTER_ROLES
            },
        }
    finally:
        if writer is not None:
            writer.release()
        cleanup_episode_resources(
            task=task, policy=None, task_sampler=sampler,
            preloaded_policy=None, close_task_sampler=sampler is not None,
        )
        shutil.rmtree(scratch, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-root", type=Path, default=DEFAULT_SMOKE_ROOT)
    parser.add_argument("--configuration", type=Path, default=DEFAULT_CONFIGURATION)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--side", choices=("left", "right"), default="left")
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    configuration = json.loads(args.configuration.resolve().read_text())
    smoke_root = args.smoke_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    smoke_summary = json.loads((smoke_root / "summary.json").read_text())
    retained_rows = {
        (str(item["layout_family_id"]), str(item["intrusion_side"])): item
        for item in list(smoke_summary.get("manifest_rows") or [])
    }

    jobs = []
    for item in smoke_summary["results"]:
        family_id, side = str(item["family_id"]), str(item["intrusion_side"])
        if side != args.side:
            continue
        episode_dir = _find_episode_dir(smoke_root, str(item["episode_id"]))
        jobs.append(
            {
                "family_id": family_id,
                "side": side,
                "source_episode_id": str(item["episode_id"]),
                "row": _build_row(retained_rows[(family_id, side)], configuration),
                "result_path": str(episode_dir / "result.json"),
                "trajectory_path": str(episode_dir / "trajectory.json"),
                "video_path": str(output_root / "videos" / f"{family_id}_{side}_v96_cluster.mp4"),
                "frame_stride": int(args.frame_stride),
            }
        )
    jobs.sort(key=lambda job: job["family_id"])
    print(f"rendering {len(jobs)} review clips", flush=True)

    results = []
    if args.workers == 1:
        for job in jobs:
            print(f"  {job['family_id']} / {job['side']}", flush=True)
            results.append(_render(job))
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_render, job): job for job in jobs}
            for future in concurrent.futures.as_completed(futures):
                item = future.result()
                print(json.dumps({k: item[k] for k in ("family_id", "side", "video_path")}), flush=True)
                results.append(item)
    results.sort(key=lambda item: item["family_id"])

    document = {
        "schema_version": "pact_place_v9_6_cluster_review_videos_v1",
        "role": "non_authorizing_design_review_clips",
        "authorizes_gate": False,
        "authorizes_collection": False,
        "authorizes_v1b": False,
        "clips_are_replays_with_clusters_composited_in": True,
        "recorded_expert_never_saw_the_clusters": True,
        "physics_stepped": False,
        "geometry_tinted_for_review": False,
        "configuration_rejected_by_w2": True,
        "configuration_path": str(args.configuration.resolve().relative_to(ROOT)),
        "configuration_sha256": configuration["configuration_sha256"],
        "renderer_path": str(Path(__file__).resolve().relative_to(ROOT)),
        "renderer_sha256": sha256_file(Path(__file__).resolve()),
        "clips": results,
    }
    document["document_sha256"] = sha256_payload(document)
    path = output_root / "review_manifest.json"
    path.write_text(json.dumps(psr.jsonable(document), indent=2, sort_keys=True) + "\n")
    print(path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
