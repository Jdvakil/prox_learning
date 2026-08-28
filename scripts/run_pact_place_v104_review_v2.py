#!/usr/bin/env python3
"""V10.4 review-v2: bridge v1, certify three controls, publish six videos, stop.

Order matters and is enforced: the provenance bridge runs first, then all
three diagnostic controls are certified, and only then is a single frame
rendered. A control that cannot be certified writes a stop artifact and
publishes nothing.

Publication is atomic. Everything is rendered and decode-verified inside a
temporary directory and moved into place in one step, so a partial packet can
never appear under the reviewed name.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
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
    PRODUCTION_ROOT,
    REVIEW_FPS,
    REVIEW_FRAME_STRIDE,
    REVIEW_MIN_CLEARANCE_M,
)
from pact_place_v104_control_certify import certify_all  # noqa: E402
from pact_place_v104_control_certify import (  # noqa: E402
    build_scene_bundle,
    production_scene_unchanged,
)
from pact_place_v104_geometry import (  # noqa: E402
    ALL_GEOMS_V104,
    SCENE_XML_RELATIVE_V104,
    production_assembly,
)
from pact_place_v104_review_v2_contract import (  # noqa: E402
    CONTRACT_VERSION_V2,
    CONTROL_BANNER_LINES,
    CONTROL_ORDER_V2,
    CONTROL_SHIFT_GRID_V2_M,
    CONTROL_SPEC,
    EXECUTED_V1_CONTRACT_SHA256,
    EXECUTED_V1_IMPLEMENTATION_SHA256,
    LEFT_LOBE_SECONDARY_STEM_FIRST_FRAME,
    LEFT_LOBE_SECONDARY_STEM_MAX_FRAME,
    LEFT_LOBE_SECONDARY_STEM_MAX_PENETRATION_M,
    N_REVIEW_V2_CONTROLS,
    N_REVIEW_V2_SUCCESSES,
    N_REVIEW_V2_VIDEOS,
    PHASE0_V2_ROOT,
    REVIEW_V2_ROOT,
    SCENE_METADATA_SHA256,
    SCHEMA_PREFIX,
    SUCCESS_ROLES,
    build_provenance_bridge,
    empty_authorization,
    gate_v2_implementation_sha256,
    review_v2_implementation_sha256,
    scoped_production_sha256,
    sha256_bytes_of,
    write_immutable_create_only,
    write_immutable_text_create_only,
)
from run_pact_place_expert_screen import _make_config, _result_path  # noqa: E402

SAMPLER_CLASS = "PactPlaceCorridorV104Sampler"
PENDANT_CAM_POS = np.asarray([0.16, -0.02, 1.30], dtype=float)
PENDANT_CAM_TARGET = np.asarray([0.78, 0.00, 1.02], dtype=float)
PENDANT_CAM_FOV = 52.0


def video_duration_s(n_frames: int, *, fps: float = REVIEW_FPS) -> float:
    return float(int(n_frames)) / float(fps)


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------
def select_successes(manifest: dict[str, Any], production_root: Path) -> list[dict[str, Any]]:
    """The three registered production successes: roles 0 left, 3 right, 4 left."""
    rows = {int(row["role_index"]): row for row in manifest["rows"]}
    out: list[dict[str, Any]] = []
    for role, side in SUCCESS_ROLES:
        row = rows[int(role)]
        if str(row["intrusion_side"]) != side:
            raise RuntimeError(
                f"registered success role {role} is {row['intrusion_side']}, not {side}"
            )
        result_path = _result_path(production_root, row)
        result = json.loads(result_path.read_text())
        telemetry = result.get("pact_v104_frame_telemetry") or {}
        if not result.get("clean_success"):
            raise RuntimeError(f"registered success role {role} is not strict-clean")
        out.append(
            {
                "role_index": int(role),
                "intrusion_side": side,
                "row": row,
                "result_path": str(result_path),
                "trajectory_path": str(result_path.parent / "trajectory.json"),
                "min_clearance_m": telemetry.get("min_clearance_m"),
                "episode_steps": result.get("episode_steps"),
                "seed_u32": int(result["selected_seed"]["seed_u32"]),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def render_clip(job: dict[str, Any]) -> dict[str, Any]:
    """Three-pane, stride-1, true-time clip with burned-in telemetry."""
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
    is_control = bool(job.get("is_control"))
    if is_control:
        # A control must render the shifted assembly it was certified against.
        # Falling back to production geometry here would draw a "control" that
        # never touches anything, which is worse than failing.
        assembly = job["assembly"]
    else:
        assembly = production_assembly()
    output = Path(job["video_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    frames = list(job.get("frames") or range(len(steps)))

    task = sampler = scratch = None
    writer = None
    try:
        scratch = Path(tempfile.mkdtemp(prefix="v104_render_v2_"))
        if is_control:
            scene_path = build_scene_bundle(assembly, scratch / "scene")
        else:
            scene_path = ROOT / SCENE_XML_RELATIVE_V104
        config = _make_config(
            scratch / "d.json", scene_xml=scene_path, sampler_class=SAMPLER_CLASS
        )
        sampler = config.task_sampler_config.task_sampler_class(config)
        sampler.seed_task_sampling(int(job["seed_u32"]))
        sampler.set_pact_manifest_row(row)
        task = sampler.sample_task(house_index=int(row["scene_template_house_index"]))
        env = task.env
        model, data = env.current_model, env.current_data
        probe = robot_collision_geom_ids(model) + target_collision_geom_ids(task)
        cache = geom_shape_cache(model, probe)
        boxes = assembly_boxes(assembly)
        pendant_ids = [int(model.geom(name).id) for name in ALL_GEOMS_V104]
        target_gid = (
            int(model.geom(job["component_geom"]).id) if is_control else None
        )
        side_delta = PENDANT_CAM_TARGET - PENDANT_CAM_POS
        side_fwd = side_delta / np.linalg.norm(side_delta)
        side_up = np.asarray([0.0, 0.0, 1.0], dtype=float)
        writer = cv2.VideoWriter(
            str(output),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(REVIEW_FPS),
            (PANE_WH[0] * 3, PANE_WH[1]),
        )
        contact_frames: list[int] = []
        for index in frames:
            step = steps[index]
            apply_recorded_qpos(env, step["qpos"])
            mujoco.mj_forward(model, data)
            report = frame_clearances(model, data, boxes, probe, cache)
            contact = pendant_contact_state(model, data, pendant_ids)
            if contact["contact"]:
                contact_frames.append(index)
            penetration = None
            if target_gid is not None:
                penetration = -float(true_distance(model, data, probe, [target_gid]))
            w_pos, w_fwd, w_up = wrist_camera_pose(env)
            wrist = _resize_pane(
                np.asarray(
                    env._render_frame(w_pos, w_fwd, w_up, WRIST_FOV, segmentation=False)
                )
            )
            tp_pos, tp_fwd, tp_up = third_person_pose(env)
            third = _resize_pane(
                np.asarray(
                    env._render_frame(
                        tp_pos, tp_fwd, tp_up, THIRD_PERSON_FOV, segmentation=False
                    )
                )
            )
            pendant = _resize_pane(
                np.asarray(
                    env._render_frame(
                        PENDANT_CAM_POS,
                        side_fwd,
                        side_up,
                        PENDANT_CAM_FOV,
                        segmentation=False,
                    )
                )
            )
            frame = cv2.cvtColor(
                np.concatenate([wrist, third, pendant], axis=1), cv2.COLOR_RGB2BGR
            )
            width, height = PANE_WH
            header_h = 176 if is_control else 106
            shade = frame.copy()
            cv2.rectangle(shade, (0, 0), (frame.shape[1], header_h), (0, 0, 0), -1)
            cv2.rectangle(
                shade, (0, height - 30), (frame.shape[1], height), (0, 0, 0), -1
            )
            frame = cv2.addWeighted(shade, 0.70, frame, 0.30, 0.0)
            minimum = report["min_m"]
            limiting = report["limiting"] or {}
            speed = step.get("commanded_speed_m_s")
            realized = step.get("realized_tcp_speed_m_s")
            status = (
                "DIAGNOSTIC NEGATIVE CONTROL"
                if is_control
                else "PRODUCTION CLEAN SUCCESS"
            )
            colour = (
                (60, 60, 255) if (is_control or contact["contact"]) else (120, 255, 120)
            )
            wall = float(frames.index(index)) / float(REVIEW_FPS)
            lines = [
                f"{status}  {job['label']}  side {row['intrusion_side']}  "
                f"retained frame {index}/{len(steps) - 1}  "
                f"sim {float(step.get('sim_time_s') or 0.0):7.3f}s  "
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
            if is_control:
                lines += [
                    f"source role {int(row['role_index'])} ({row['intrusion_side']})   "
                    f"target component {job['component']}   "
                    f"assembly shift {float(job['shift_m']):.3f} m inward along y",
                    f"signed penetration "
                    f"{'--' if penetration is None else format(penetration * 1000.0, '.3f')} mm"
                    f"   limiting pair {job['limiting_pair']}"
                    f"   classification {job['contact_classification']}",
                ]
            for line_index, line in enumerate(lines):
                cv2.putText(
                    frame,
                    line,
                    (12, 24 + line_index * 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    colour if line_index == 0 else (230, 230, 230),
                    1,
                    cv2.LINE_AA,
                )
            if is_control:
                banner = "  -  ".join(CONTROL_BANNER_LINES)
                cv2.rectangle(
                    frame, (0, header_h - 24), (frame.shape[1], header_h), (0, 0, 200), -1
                )
                cv2.putText(
                    frame,
                    banner,
                    (12, header_h - 7),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            for pane_index, label in enumerate(
                (
                    "wrist (RGB policy view, untinted)",
                    "third-person task view",
                    "pendant view: both lobes, stems, crossbar",
                )
            ):
                cv2.putText(
                    frame,
                    label,
                    (pane_index * width + 12, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.40,
                    (245, 245, 245),
                    1,
                    cv2.LINE_AA,
                )
            writer.write(frame)
        writer.release()
        writer = None
        return {
            "label": job["label"],
            "kind": job["kind"],
            "role_index": int(row["role_index"]),
            "intrusion_side": str(row["intrusion_side"]),
            "video_path": str(output),
            "video_name": output.name,
            "video_sha256": sha256_file(output),
            "size_bytes": int(output.stat().st_size),
            "n_frames": len(frames),
            "first_frame": int(frames[0]),
            "last_frame": int(frames[-1]),
            "fps": float(REVIEW_FPS),
            "frame_stride": int(REVIEW_FRAME_STRIDE),
            "duration_s": video_duration_s(len(frames)),
            "true_time": True,
            "wrist_pane_untinted": True,
            "n_contact_frames": len(contact_frames),
            "contact_frames": contact_frames[:16],
            "is_diagnostic_negative_control": is_control,
            "trimmed_contact_window": bool(is_control),
            "control_banner_lines": list(CONTROL_BANNER_LINES) if is_control else None,
            "scene_xml": str(scene_path),
        }
    finally:
        if writer is not None:
            writer.release()
        cleanup_episode_resources(
            task=task,
            policy=None,
            task_sampler=sampler,
            preloaded_policy=None,
            close_task_sampler=sampler is not None,
        )
        if scratch is not None:
            shutil.rmtree(scratch, ignore_errors=True)


# ---------------------------------------------------------------------------
# Decode verification
# ---------------------------------------------------------------------------
def decode_video(path: Path) -> dict[str, Any]:
    """Decode the written file back. Never trusts the writer's own count."""
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return {"decoded": False, "reason": "cv2 could not open the file"}
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        counted = 0
        while True:
            ok, _ = capture.read()
            if not ok:
                break
            counted += 1
    finally:
        capture.release()
    return {
        "decoded": True,
        "decoded_frames": counted,
        "decoded_fps": fps,
        "decoded_duration_s": counted / fps if fps > 0 else None,
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def verify_rendered(record: dict[str, Any]) -> dict[str, Any]:
    path = Path(record["video_path"])
    decoded = decode_video(path)
    checks = {
        **decoded,
        "expected_frames": int(record["n_frames"]),
        "frame_count_matches": decoded.get("decoded_frames") == int(record["n_frames"]),
        "fps_matches": (
            decoded.get("decoded_fps") is not None
            and abs(float(decoded.get("decoded_fps", 0.0)) - float(REVIEW_FPS)) <= 0.5
        ),
        "duration_matches": (
            decoded.get("decoded_duration_s") is not None
            and abs(
                float(decoded.get("decoded_duration_s", 0.0))
                - float(record["duration_s"])
            )
            <= 0.25
        ),
        "nonzero_size": int(decoded.get("size_bytes", 0)) > 0,
        "sha256_stable": decoded.get("sha256") == record["video_sha256"],
    }
    checks["passed"] = bool(
        checks["decoded"]
        and checks["frame_count_matches"]
        and checks["fps_matches"]
        and checks["duration_matches"]
        and checks["nonzero_size"]
        and checks["sha256_stable"]
    )
    return checks


# ---------------------------------------------------------------------------
# REVIEW.md
# ---------------------------------------------------------------------------
def review_markdown(manifest: dict[str, Any], bindings: dict[str, str]) -> str:
    lines = [
        "# V10.4 review-v2 packet",
        "",
        "Three production clean successes and three diagnostic negative controls.",
        "Nothing in this packet is a new episode. The six production episodes were",
        "run under V10.4-v1 and are reused byte-for-byte through the provenance",
        "bridge; the controls are separately compiled diagnostic scenes and are",
        "**not** production geometry.",
        "",
        "## Videos",
        "",
        "| # | kind | label | frames | duration | window |",
        "|---|---|---|---|---|---|",
    ]
    for index, video in enumerate(manifest["videos"], start=1):
        window = (
            f"frames {video['first_frame']}-{video['last_frame']} (trimmed)"
            if video["is_diagnostic_negative_control"]
            else "complete retained trajectory"
        )
        lines.append(
            f"| {index} | {video['kind']} | `{video['video_name']}` | "
            f"{video['n_frames']} | {video['duration_s']:.2f} s | {window} |"
        )
    lines += [
        "",
        "All six are stride 1 at the 66 ms control period "
        f"({REVIEW_FPS:.4f} fps), true time, with an untinted wrist pane, a",
        "third-person pane, and a pendant pane.",
        "",
        "## Diagnostic controls",
        "",
        "| control | component | source role | shift | penetration | max frame | limiting body |",
        "|---|---|---|---|---|---|---|",
    ]
    for certificate in manifest["control_certificates"]:
        lines.append(
            f"| {certificate['control']} | `{certificate['component']}` | "
            f"{certificate['source_role_index']} "
            f"({certificate['source_intrusion_side']}) | "
            f"{certificate['shift_m']:.3f} m | "
            f"{certificate['penetration_m'] * 1000.0:.3f} mm | "
            f"{certificate['max_frame']} | "
            f"`{certificate['parity']['limiting_robot_body']}` |"
        )
    lines += [
        "",
        "Each control rigidly translates the complete assembly inward along y on",
        "a separately compiled scene, reloaded through the real task sampler and",
        "confirmed compiled-static with enclosing bounds. At the certified frame",
        "signed distance, analytic GJK, live `data.contact`, and the place audit's",
        "`mounted_fixture` classification all agree.",
        "",
        "The production scene XML is byte-identical before and after every control",
        f"(`{bindings['production_scene_sha256']}`).",
        "",
        "## What this packet does not authorize",
        "",
        "Every authorization field is false and `human_approval.json` is absent.",
        "Phase 0 has not run. No episode, `env.step`, collection, training, or",
        "evaluation occurred while this packet was built.",
        "",
        "## Owner approval schema",
        "",
        "To authorize Phase0-v2, write `human_approval.json` into",
        f"`{REVIEW_V2_ROOT}/` yourself, containing exactly:",
        "",
        "```json",
        "{",
        '  "decision": "approve_phase0",',
        '  "created_by_agent": false,',
        '  "reviewed_videos": [',
    ]
    for video in manifest["videos"]:
        lines.append(f'    "{video["video_name"]}",')
    lines[-1] = lines[-1].rstrip(",")
    lines += ["  ],"]
    for key, digest in sorted(bindings.items()):
        lines.append(f'  "{key}": "{digest}",')
    lines[-1] = lines[-1].rstrip(",")
    lines += [
        "}",
        "```",
        "",
        "The verifier recomputes every one of these from file bytes rather than",
        "trusting an embedded value, requires the video list to be exactly the six",
        "names above with no extras, and refuses a record with",
        "`created_by_agent: true`. A passing gate sets only `phase0_passed: true`;",
        "collection, training, and evaluation stay unauthorized.",
        "",
        "Then run:",
        "",
        "```",
        "python scripts/run_pact_place_v104_phase0_v2.py",
        "```",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--production-root", type=Path, default=ROOT / PRODUCTION_ROOT)
    parser.add_argument("--output-root", type=Path, default=ROOT / REVIEW_V2_ROOT)
    args = parser.parse_args()
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("MLSPACES_ASSETS_DIR", str(ROOT / "assets"))
    os.environ.pop("DISPLAY", None)

    output_root = args.output_root.resolve()
    if output_root.exists():
        raise SystemExit(
            f"refusing to publish over an existing review packet: {output_root}"
        )
    production_root = args.production_root.resolve()
    manifest_document = json.loads(
        (production_root / "production_manifest.json").read_text()
    )

    staging = Path(tempfile.mkdtemp(prefix="v104_review_v2_", dir=output_root.parent))
    try:
        # --- 1. provenance bridge, before anything else --------------------
        bridge = build_provenance_bridge()
        write_immutable_create_only(staging / "provenance_bridge.json", bridge)
        if not bridge["bridge_passed"]:
            stop = {
                "schema_version": f"{SCHEMA_PREFIX}_stop_v1",
                "contract_version": CONTRACT_VERSION_V2,
                "stop_reason": "provenance_bridge_failed",
                "failed_sections": bridge["failed_sections"],
                "videos_published": 0,
                **empty_authorization(),
            }
            write_immutable_create_only(staging / "review_v2_stop.json", stop)
            _publish(staging, output_root)
            print(json.dumps(stop, indent=2))
            return 1

        # --- 2. certify every control before rendering a single frame ------
        certification = certify_all(production_root, manifest_document)
        certificates_document = {
            "schema_version": f"{SCHEMA_PREFIX}_control_certificates_v1",
            "contract_version": CONTRACT_VERSION_V2,
            "grid_m": {
                "first": float(CONTROL_SHIFT_GRID_V2_M[0]),
                "last": float(CONTROL_SHIFT_GRID_V2_M[-1]),
                "n_points": len(CONTROL_SHIFT_GRID_V2_M),
                "increment_m": 0.001,
            },
            "diagnostic_only_not_a_production_search": True,
            "control_order": list(CONTROL_ORDER_V2),
            "control_spec": CONTROL_SPEC,
            "certificates": certification["certificates"],
            "shortfalls": certification["shortfalls"],
            "n_certified": certification["n_certified"],
            "all_certified": certification["all_certified"],
            **empty_authorization(),
        }
        write_immutable_create_only(
            staging / "control_certificates.json", certificates_document
        )
        if not certification["all_certified"]:
            stop = {
                "schema_version": f"{SCHEMA_PREFIX}_stop_v1",
                "contract_version": CONTRACT_VERSION_V2,
                "stop_reason": "diagnostic_control_certification_failed",
                "shortfalls": certification["shortfalls"],
                "n_certified": certification["n_certified"],
                "videos_published": 0,
                "production_geometry_changed": False,
                "production_scene": production_scene_unchanged(),
                **empty_authorization(),
            }
            write_immutable_create_only(staging / "review_v2_stop.json", stop)
            _publish(staging, output_root)
            print(json.dumps(stop, indent=2))
            return 1

        # --- 3. render ------------------------------------------------------
        successes = select_successes(manifest_document, production_root)
        videos_dir = staging / "videos"
        videos_dir.mkdir(parents=True, exist_ok=True)
        jobs: list[dict[str, Any]] = []
        for record in successes:
            jobs.append(
                {
                    "kind": "production_clean_success",
                    "is_control": False,
                    "label": f"row{record['role_index']:02d}-"
                    f"{record['intrusion_side']}-clean",
                    "row": record["row"],
                    "seed_u32": record["seed_u32"],
                    "trajectory_path": record["trajectory_path"],
                    "outcome": "clean success",
                    "frames": None,
                    "video_path": str(
                        videos_dir
                        / f"success_{record['role_index']:02d}_"
                        f"{record['intrusion_side']}.mp4"
                    ),
                }
            )
        rows = {int(row["role_index"]): row for row in manifest_document["rows"]}
        by_control = {item["control"]: item for item in certification["certificates"]}
        for control in CONTROL_ORDER_V2:
            certificate = by_control[control]
            row = rows[int(certificate["source_role_index"])]
            result_path = _result_path(production_root, row)
            result = json.loads(result_path.read_text())
            found_assembly = certificate["assembly"]
            window = certificate["window"]
            pair = certificate["parity"]["live_contact_pairs"]
            limiting = (
                f"{certificate['component_geom']} <-> "
                f"{certificate['parity']['limiting_robot_body']}"
            )
            jobs.append(
                {
                    "kind": "diagnostic_negative_control",
                    "is_control": True,
                    "control": control,
                    "label": f"{control}-role{certificate['source_role_index']:02d}",
                    "row": row,
                    "seed_u32": int(result["selected_seed"]["seed_u32"]),
                    "trajectory_path": str(result_path.parent / "trajectory.json"),
                    "outcome": (
                        f"{certificate['penetration_m'] * 1000.0:.3f} mm penetration"
                    ),
                    "assembly": found_assembly,
                    "component": certificate["component"],
                    "component_geom": certificate["component_geom"],
                    "shift_m": certificate["shift_m"],
                    "limiting_pair": limiting,
                    "contact_classification": ", ".join(
                        certificate["parity"]["contact_classes"]
                    ),
                    "frames": list(
                        range(window["first_frame"], window["last_frame"] + 1)
                    ),
                    "video_path": str(videos_dir / f"control_{control}.mp4"),
                }
            )
        rendered = [render_clip(job) for job in jobs]
        verifications = [verify_rendered(record) for record in rendered]
        if not all(item["passed"] for item in verifications):
            stop = {
                "schema_version": f"{SCHEMA_PREFIX}_stop_v1",
                "contract_version": CONTRACT_VERSION_V2,
                "stop_reason": "rendered_video_failed_decode_verification",
                "verifications": verifications,
                "videos_published": 0,
                **empty_authorization(),
            }
            write_immutable_create_only(staging / "review_v2_stop.json", stop)
            _publish(staging, output_root)
            print(json.dumps(stop, indent=2))
            return 1

        # --- 4. manifests ---------------------------------------------------
        for record in rendered:
            record["video_path"] = f"{REVIEW_V2_ROOT}/videos/{record['video_name']}"
        bindings = {
            "contract_version_v2": CONTRACT_VERSION_V2,
            "scoped_production_sha256": scoped_production_sha256(),
            "review_v2_implementation_sha256": review_v2_implementation_sha256(),
            "gate_v2_implementation_sha256": gate_v2_implementation_sha256(),
            "production_scene_sha256": sha256_bytes_of(ROOT / SCENE_XML_RELATIVE_V104),
            "scene_metadata_sha256": SCENE_METADATA_SHA256,
            "executed_v1_contract_sha256": EXECUTED_V1_CONTRACT_SHA256,
            "executed_v1_implementation_sha256": EXECUTED_V1_IMPLEMENTATION_SHA256,
            "provenance_bridge_sha256": _payload_of(staging / "provenance_bridge.json"),
            "control_certificates_sha256": _payload_of(
                staging / "control_certificates.json"
            ),
        }
        preflight = {
            "schema_version": f"{SCHEMA_PREFIX}_review_preflight_v1",
            "contract_version": CONTRACT_VERSION_V2,
            "bridge_passed": True,
            "all_controls_certified": True,
            "n_successes": N_REVIEW_V2_SUCCESSES,
            "n_controls": N_REVIEW_V2_CONTROLS,
            "n_videos": N_REVIEW_V2_VIDEOS,
            "video_verifications": verifications,
            "bindings": bindings,
            "creates_episode": False,
            "calls_env_step": False,
            "replacement_episodes_generated": False,
            "production_scene": production_scene_unchanged(),
            **empty_authorization(),
            "review_preflight_passed": True,
        }
        write_immutable_create_only(staging / "review_preflight.json", preflight)
        bindings["review_preflight_sha256"] = _payload_of(
            staging / "review_preflight.json"
        )
        manifest = {
            "schema_version": f"{SCHEMA_PREFIX}_review_manifest_v1",
            "contract_version": CONTRACT_VERSION_V2,
            "n_videos": len(rendered),
            "n_production_successes": sum(
                1 for item in rendered if not item["is_diagnostic_negative_control"]
            ),
            "n_diagnostic_controls": sum(
                1 for item in rendered if item["is_diagnostic_negative_control"]
            ),
            "videos": rendered,
            "video_sha256": {
                item["video_name"]: item["video_sha256"] for item in rendered
            },
            "control_certificates": certification["certificates"],
            "left_lobe_secondary_stem": {
                "first_contact_frame": LEFT_LOBE_SECONDARY_STEM_FIRST_FRAME,
                "max_penetration_m": LEFT_LOBE_SECONDARY_STEM_MAX_PENETRATION_M,
                "max_penetration_frame": LEFT_LOBE_SECONDARY_STEM_MAX_FRAME,
                "excluded_from_left_lobe_clip": True,
                "note": (
                    "The stem first touches at frame 90 and only reaches "
                    "38.15 mm at frame 193. The clip is trimmed at 89, so "
                    "neither is shown."
                ),
            },
            "bindings": bindings,
            "successes": [
                {k: v for k, v in item.items() if k not in ("row",)}
                for item in successes
            ],
            "production_min_clearance_m": manifest_document["eligibility"][
                "min_observed_clearance_m"
            ],
            "production_pendant_contact_rows": manifest_document["eligibility"][
                "pendant_contact_rows"
            ],
            "review_min_clearance_m": REVIEW_MIN_CLEARANCE_M,
            "phase0_v2_root": PHASE0_V2_ROOT,
            **empty_authorization(),
            "eligible_for_human_review": True,
        }
        write_immutable_create_only(staging / "review_manifest.json", manifest)
        # These two can only be computed after the manifest exists: a document
        # cannot contain its own hash, and the video hashes are what the
        # manifest binds. The owner schema must still list them, because the
        # verifier demands them.
        approval_bindings = dict(bindings)
        approval_bindings["review_manifest_sha256"] = _payload_of(
            staging / "review_manifest.json"
        )
        for record in rendered:
            approval_bindings[f"video_sha256:{record['video_name']}"] = record[
                "video_sha256"
            ]
        write_immutable_text_create_only(
            staging / "REVIEW.md", review_markdown(manifest, approval_bindings)
        )
        _publish(staging, output_root)
        published = sorted((output_root / "videos").glob("*.mp4"))
        print(
            json.dumps(
                {
                    "review_v2_published": True,
                    "root": str(output_root.relative_to(ROOT)),
                    "n_videos": len(published),
                    "videos": [str(item.relative_to(ROOT)) for item in published],
                    "human_approval_present": (
                        output_root / "human_approval.json"
                    ).exists(),
                    **empty_authorization(),
                    "eligible_for_human_review": True,
                },
                indent=2,
            )
        )
        return 0
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _payload_of(path: Path) -> str:
    return json.loads(Path(path).read_text())["artifact_sha256"]


def _publish(staging: Path, output_root: Path) -> None:
    """Atomic: one rename, or nothing."""
    try:
        os.rename(staging, output_root)
    except OSError as error:
        raise RuntimeError(
            f"refusing to publish over an existing review packet: {output_root}"
        ) from error


if __name__ == "__main__":
    raise SystemExit(main())
