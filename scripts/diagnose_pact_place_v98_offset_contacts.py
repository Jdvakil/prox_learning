#!/usr/bin/env python3
"""Retained-qpos contact and lag diagnostic for the V9.8 offset pendant.

Restores recorded qpos and calls mj_forward only. Does not step physics, run
the expert, or overwrite existing V9.8 artifacts.
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
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for search_path in (ROOT / "scripts", ROOT / "submodules" / "molmospaces"):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from pact_place_corridor_contract import sha256_file, sha256_payload  # noqa: E402
from pact_place_v98_offset_contact_lib import (  # noqa: E402
    CEILING_FIXTURE_PHASES,
    DESIGN_LAG_NEG_M,
    DESIGN_LAG_POS_M,
    FACE_WINDOW_M,
    IMMUTABLE_JSON_RELATIVE,
    LINK6_BODY_LABEL,
    LINK6_BODY_NAME,
    NAMED_LAG_QUANTITIES,
    RUN_SPECS,
    SCENE_XML_NAME,
    SCHEMA_VERSION,
    WORKING_CONCLUSION,
    classify_onset_category,
    count_categories,
    definition_reproduces_provenance,
    empty_authorization,
    fixture_from_result,
    lag_rows_for_aggregate,
    lag_toward_centreline_m,
    lookup_manifest,
    patch_manifest_for_row,
    pendant_aabb_faces,
    reconstruction_is_valid,
    select_causal_category,
    split_robot_fixture_sides,
    tcp_x_relation,
    trajectory_phase_at,
    trajectory_phase_sequence,
)
from pact_place_v98_pendant_contract import (  # noqa: E402
    CONTRACT_VERSION,
    PENDANT_GEOM,
    SAMPLER_CLASS,
    validate_pendant_geometry as LIVE_VALIDATE_PENDANT,
)

SOURCE_SUMMARY = ROOT / "diagnostics_output/pact_place_v95_raw_smoke/summary.json"
DEFAULT_OUTPUT = (
    ROOT / "diagnostics_output/pact_place_v98_offset_contact_diagnosis"
)
SCENE_XML = ROOT / (
    "submodules/molmospaces/molmo_spaces/data_generation/custom_scenes/"
    + SCENE_XML_NAME
)
ONSET_NEIGHBOR_OFFSETS = (-2, -1, 0, 1, 2)


def _json_float(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not np.isfinite(number):
        return None
    return number


def _json_vec(values: Any) -> list[float] | None:
    if values is None:
        return None
    return [_json_float(item) for item in np.asarray(values, dtype=float).reshape(-1)]


def _body_origin(model, data, name: str) -> np.ndarray | None:
    try:
        return np.asarray(data.body(name).xpos, dtype=np.float64)
    except (KeyError, ValueError, TypeError):
        return None


def _geom_origin(model, data, geom_id: int) -> np.ndarray:
    return np.asarray(data.geom_xpos[int(geom_id)], dtype=np.float64)


def _geom_type_name(model, geom_id: int) -> str:
    import mujoco

    return str(mujoco.mjtGeom(int(model.geom_type[int(geom_id)])).name)


def _aabb_xz_overlap(a_lo, a_hi, b_lo, b_hi) -> bool:
    return (
        float(a_lo[0]) <= float(b_hi[0])
        and float(b_lo[0]) <= float(a_hi[0])
        and float(a_lo[2]) <= float(b_hi[2])
        and float(b_lo[2]) <= float(a_hi[2])
    )


def live_mounted_fixture_pairs(env) -> list[dict[str, Any]]:
    from molmo_spaces.tasks.pact_place_contact_audit import (
        _contact_pair_record,
        classify_contact,
        place_environment_contact_pairs,
    )

    model, data = env.current_model, env.current_data
    production = [
        pair
        for pair in place_environment_contact_pairs(env)
        if classify_contact(pair) == "mounted_fixture"
    ]
    enriched: list[dict[str, Any]] = []
    for pair in production:
        match = None
        for index in range(int(data.ncon)):
            contact = data.contact[index]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            record = _contact_pair_record(model, contact, geom1, geom2)
            if (
                record["geom1"] == pair.get("geom1")
                and record["geom2"] == pair.get("geom2")
                and record["body1"] == pair.get("body1")
                and record["body2"] == pair.get("body2")
                and abs(float(record["distance_m"]) - float(pair.get("distance_m", 0.0)))
                < 1e-9
            ):
                match = (contact, record, geom1, geom2)
                break
        item = dict(pair)
        item["contact_class"] = "mounted_fixture"
        if match is None:
            item["contact_position_m"] = None
            item["signed_distance_m"] = _json_float(pair.get("distance_m"))
            item["penetration_m"] = max(0.0, -float(pair.get("distance_m") or 0.0))
            item["geom_match_limitation"] = "ncon_record_not_matched"
        else:
            contact, record, geom1, geom2 = match
            item["geom1_id"] = geom1
            item["geom2_id"] = geom2
            item["contact_position_m"] = _json_vec(contact.pos[:3])
            item["signed_distance_m"] = float(contact.dist)
            item["penetration_m"] = max(0.0, -float(contact.dist))
            sides = split_robot_fixture_sides(item)
            item.update(sides)
            robot_gid = sides.get("robot_geom_id")
            robot_body = sides.get("robot_body")
            if robot_gid is not None:
                item["robot_geom_origin_m"] = _json_vec(_geom_origin(model, data, robot_gid))
                item["robot_geom_type"] = _geom_type_name(model, robot_gid)
                rotation = np.asarray(data.geom_xmat[int(robot_gid)], dtype=float).reshape(3, 3)
                item["robot_geom_xmat_row_major"] = _json_vec(rotation.reshape(-1))
            if robot_body:
                origin = _body_origin(model, data, str(robot_body))
                item["robot_parent_body"] = robot_body
                item["robot_parent_body_origin_m"] = _json_vec(origin)
                item["robot_parent_body_origin_label"] = "parent_body_origin"
        enriched.append(item)
    return enriched


def geom_signed_distance_m(model, data, geom_a: int, geom_b: int) -> dict[str, Any]:
    import mujoco

    fromto = np.zeros(6, dtype=np.float64)
    try:
        distance = float(
            mujoco.mj_geomDistance(model, data, int(geom_a), int(geom_b), 2.0, fromto)
        )
    except Exception as error:  # noqa: BLE001 - API availability is diagnostic evidence
        return {
            "signed_distance_m": None,
            "fromto_m": None,
            "api": "mujoco.mj_geomDistance",
            "error": f"{type(error).__name__}: {error}",
        }
    return {
        "signed_distance_m": distance,
        "fromto_m": _json_vec(fromto),
        "api": "mujoco.mj_geomDistance",
        "error": None,
    }


def collision_facing_y_m(model, data, geom_id: int, route_sign: float) -> dict[str, Any]:
    from run_pact_place_swept_volume_v7 import world_aabb_for_geom

    lo, hi = world_aabb_for_geom(model, data, int(geom_id))
    import mujoco

    gtype = int(model.geom_type[int(geom_id)])
    used_rbound = gtype not in {
        int(mujoco.mjtGeom.mjGEOM_BOX),
        int(mujoco.mjtGeom.mjGEOM_SPHERE),
        int(mujoco.mjtGeom.mjGEOM_CAPSULE),
        int(mujoco.mjtGeom.mjGEOM_CYLINDER),
    }
    facing = float(hi[1]) if float(route_sign) < 0.0 else float(lo[1])
    return {
        "aabb_min_m": _json_vec(lo),
        "aabb_max_m": _json_vec(hi),
        "collision_facing_y_m": facing,
        "collision_facing_side": "+y" if float(route_sign) < 0.0 else "-y",
        "aabb_used_rbound": used_rbound,
        "geom_type": _geom_type_name(model, geom_id),
        "sign_convention": (
            "negative route uses the robot geom AABB +y face; "
            "positive route uses the AABB -y face"
        ),
    }


def pendant_geom_id(model) -> int | None:
    try:
        return int(model.geom(PENDANT_GEOM).id)
    except (KeyError, ValueError, TypeError):
        return None


def prepare_task(row: dict[str, Any]):
    from molmo_spaces.data_generation.runtime_compat import assert_supported_runtime
    from molmo_spaces.env.abstract_sensors import SensorSuite
    from run_pact_place_expert_screen import _make_config

    assert_supported_runtime(strict=True)
    scratch = Path(tempfile.mkdtemp(prefix="pact_place_v98_offset_diag_"))
    config = _make_config(
        scratch / "dummy.json",
        scene_xml=SCENE_XML,
        sampler_class=row.get("sampler_class") or SAMPLER_CLASS,
    )
    sampler = config.task_sampler_config.task_sampler_class(config)
    sampler.seed_task_sampling(int(row["task_seed_u32"]))
    sampler.set_pact_manifest_row(row)

    def _validate_recorded_or_live(center, half):
        try:
            return LIVE_VALIDATE_PENDANT(center, half)
        except ValueError as error:
            # Historical centred y=0 / half_y in (0.12, 0.18) fixtures are
            # outside the v2 contract. Reconstruct them as recorded; do not
            # change the live contract bounds.
            return {
                "center_m": [float(center[0]), float(center[1]), float(center[2])],
                "half_m": [float(half[0]), float(half[1]), float(half[2])],
                "historical_reconstruction": True,
                "live_validation_error": str(error),
            }

    from unittest.mock import patch

    with patch(
        "pact_place_v98_pendant_contract.validate_pendant_geometry",
        _validate_recorded_or_live,
    ):
        task = sampler.sample_task(house_index=int(row["scene_template_house_index"]))
        if task is None:
            raise RuntimeError("sample_task returned None")
        task._sensor_suite = SensorSuite(
            [task._sensor_suite.sensors[uuid] for uuid in ("qpos", "tcp_pose")]
        )
        task.reset()
    return task, sampler, scratch


def cleanup_task(task, sampler, scratch: Path) -> None:
    from molmo_spaces.data_generation.pipeline import cleanup_episode_resources

    cleanup_episode_resources(
        task=task,
        policy=None,
        task_sampler=sampler,
        preloaded_policy=None,
        close_task_sampler=True,
    )
    shutil.rmtree(scratch, ignore_errors=True)


def json_onset_fields(row_dir: Path, comparison: Mapping[str, Any]) -> dict[str, Any]:
    result = json.loads((row_dir / "result.json").read_text())
    traj_path = row_dir / "trajectory.json"
    audit = result.get("contact_audit") or {}
    first_contact_step = (audit.get("first_contact_step") or {}).get("mounted_fixture")
    onset_phase = None
    category = "unreconstructed"
    if traj_path.is_file():
        trajectory = json.loads(traj_path.read_text())
        steps = trajectory.get("steps") or []
        onset_phase = trajectory_phase_at(steps, first_contact_step)
        category = classify_onset_category(
            onset_phase,
            step=first_contact_step,
            trajectory_phases=trajectory_phase_sequence(steps),
        )
    return {
        "role_index": result.get("role_index"),
        "episode_id": result.get("episode_id"),
        "status": result.get("status"),
        "intrusion_side": result.get("intrusion_side")
        or comparison.get("intrusion_side"),
        "family": comparison.get("family") or result.get("layout_family_id"),
        "terminal_policy_phase": result.get("terminal_policy_phase"),
        "authoritative_first_contact_step": first_contact_step,
        "retained_policy_phase_at_first_contact": onset_phase,
        "onset_category": category,
        "result_sha256": sha256_file(row_dir / "result.json"),
        "trajectory_sha256": sha256_file(traj_path) if traj_path.is_file() else None,
        "baseline_clean_success": bool(comparison.get("baseline_clean_success")),
    }


def diagnose_one_row(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        return _diagnose_one_row_impl(payload)
    except Exception as error:  # noqa: BLE001 - row failure must not kill the audit
        import traceback

        fallback = json_onset_fields(
            Path(payload["row_dir"]), payload.get("comparison") or {}
        )
        fallback.update(
            {
                "run_id": payload.get("run_id"),
                "candidate": payload.get("candidate"),
                "kind": payload.get("kind"),
                "reconstruction_valid": False,
                "reconstruction_exclusion_reason": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
                "physics_stepped": False,
                "episodes_ran": False,
            }
        )
        return fallback


def _diagnose_one_row_impl(payload: dict[str, Any]) -> dict[str, Any]:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("MLSPACES_ASSETS_DIR", str(ROOT / "assets"))
    os.environ.pop("DISPLAY", None)

    from run_pact_place_swept_volume_v7 import world_aabb_for_geom
    from run_pact_place_v7_replay_videos import apply_recorded_qpos, tcp_position_m

    row_dir = Path(payload["row_dir"])
    result = json.loads((row_dir / "result.json").read_text())
    traj_path = row_dir / "trajectory.json"
    comparison = payload.get("comparison") or {}
    episode_id = str(result.get("episode_id") or "")
    role_index = result.get("role_index")
    side = str(result.get("intrusion_side") or comparison.get("intrusion_side") or "")
    family = (
        result.get("layout_family_id")
        or comparison.get("family")
        or result.get("family")
    )
    audit = result.get("contact_audit") or {}
    first_contact_step = (audit.get("first_contact_step") or {}).get("mounted_fixture")
    totals = audit.get("contact_class_totals") or {}
    penetrations = audit.get("maximum_penetration_depth_m") or {}
    base = {
        "run_id": payload["run_id"],
        "candidate": payload["candidate"],
        "kind": payload["kind"],
        "role_index": role_index,
        "family": family,
        "intrusion_side": side,
        "episode_id": episode_id,
        "status": result.get("status"),
        "baseline_clean_success": bool(comparison.get("baseline_clean_success")),
        "terminal_policy_phase": result.get("terminal_policy_phase"),
        "authoritative_first_contact_step": first_contact_step,
        "mounted_fixture_contact_entries": int(totals.get("mounted_fixture") or 0),
        "mounted_fixture_max_penetration_m": _json_float(
            penetrations.get("mounted_fixture")
        ),
        "result_sha256": sha256_file(row_dir / "result.json"),
        "trajectory_sha256": None,
        "reconstruction_valid": False,
        "reconstruction_exclusion_reason": None,
        "max_tcp_residual_m": None,
        "onset_category": "unreconstructed",
        "retained_policy_phase_at_first_contact": None,
        "first_reconstructed_contact_step": None,
        "onset_step_delta": None,
        "live_pair_at_authoritative_onset": False,
        "live_pair_absent_at_control_resolution": False,
        "contacting_geoms": [],
        "onset_window": [],
        "frames": [],
        "lag_samples": [],
        "patched_episode_id": payload["patched_row"].get("episode_id"),
        "patched_layout_id": payload["patched_row"].get("layout_id"),
        "physics_stepped": False,
        "episodes_ran": False,
    }
    if str(base["patched_episode_id"]) != episode_id:
        base["reconstruction_exclusion_reason"] = (
            "patched manifest episode_id does not match this row"
        )
        return base
    if result.get("status") != "complete" or not traj_path.is_file():
        base["reconstruction_exclusion_reason"] = "no_complete_trajectory"
        return base

    trajectory = json.loads(traj_path.read_text())
    base["trajectory_sha256"] = sha256_file(traj_path)
    steps = trajectory.get("steps") or []
    phases = trajectory_phase_sequence(steps)
    onset_phase = trajectory_phase_at(steps, first_contact_step)
    base["retained_policy_phase_at_first_contact"] = onset_phase
    base["onset_category"] = classify_onset_category(
        onset_phase, step=first_contact_step, trajectory_phases=phases
    )

    fixture = fixture_from_result(result)
    if fixture is None:
        base["reconstruction_exclusion_reason"] = "missing_recorded_fixture"
        return base
    center = fixture["center_m"]
    half = fixture["half_m"]
    faces = pendant_aabb_faces(center, half)
    route_sign = -1.0 if side == "left" else 1.0
    pendant_lo = np.asarray(
        [faces["x_min_m"], faces["y_min_m"], faces["z_min_m"]], dtype=float
    )
    pendant_hi = np.asarray(
        [faces["x_max_m"], faces["y_max_m"], faces["z_max_m"]], dtype=float
    )

    task = sampler = scratch = None
    try:
        task, sampler, scratch = prepare_task(payload["patched_row"])
        env = task.env
        model, data = env.current_model, env.current_data
        mount_gid = pendant_geom_id(model)
        residuals: list[float] = []
        frames: list[dict[str, Any]] = []
        lag_samples: list[dict[str, Any]] = []
        contacting_counter: Counter[str] = Counter()
        first_live_step = None
        live_at_onset = False
        neighbor_steps = (
            set()
            if first_contact_step is None
            else {int(first_contact_step) + offset for offset in ONSET_NEIGHBOR_OFFSETS}
        )
        onset_window: list[dict[str, Any]] = []

        for step in steps:
            control_step = int(step["step"])
            apply_recorded_qpos(env, step["qpos"])
            live_tcp = tcp_position_m(env)
            recorded = step.get("tcp_position_m")
            residual = None
            if recorded is not None:
                residual = float(
                    np.linalg.norm(
                        live_tcp - np.asarray(recorded, dtype=np.float64)
                    )
                )
                residuals.append(residual)
            phase = str(step.get("policy_phase") or "")
            link6 = _body_origin(model, data, LINK6_BODY_NAME)
            pairs = live_mounted_fixture_pairs(env)
            if pairs and first_live_step is None:
                first_live_step = control_step
            if first_contact_step is not None and control_step == int(first_contact_step):
                live_at_onset = bool(pairs)
            compact_pairs = []
            for pair in pairs:
                robot_gid = pair.get("robot_geom_id")
                identity = (
                    f"{pair.get('robot_geom')}|{pair.get('robot_body')}|"
                    f"{pair.get('pendant_geom')}"
                )
                contacting_counter[identity] += 1
                compact_pairs.append(
                    {
                        "geom1": pair.get("geom1"),
                        "geom2": pair.get("geom2"),
                        "body1": pair.get("body1"),
                        "body2": pair.get("body2"),
                        "root1": pair.get("root1"),
                        "root2": pair.get("root2"),
                        "robot_geom": pair.get("robot_geom"),
                        "robot_body": pair.get("robot_body"),
                        "robot_geom_id": pair.get("robot_geom_id"),
                        "robot_geom_type": pair.get("robot_geom_type"),
                        "robot_geom_origin_m": pair.get("robot_geom_origin_m"),
                        "robot_parent_body": pair.get("robot_parent_body"),
                        "robot_parent_body_origin_m": pair.get(
                            "robot_parent_body_origin_m"
                        ),
                        "pendant_geom": pair.get("pendant_geom"),
                        "pendant_body": pair.get("pendant_body"),
                        "contact_position_m": pair.get("contact_position_m"),
                        "signed_distance_m": pair.get("signed_distance_m"),
                        "penetration_m": pair.get("penetration_m"),
                    }
                )
            frame = {
                "role_index": role_index,
                "family": family,
                "intrusion_side": side,
                "candidate": payload["candidate"],
                "episode_id": episode_id,
                "step": control_step,
                "policy_phase": phase,
                "tcp_m": _json_vec(live_tcp),
                "tcp_x_vs_pendant": tcp_x_relation(float(live_tcp[0]), center[0], half[0]),
                "pendant_aabb": faces,
                "mounted_fixture_pairs": compact_pairs,
                "fr3_link6_body_origin_m": _json_vec(link6),
                "fr3_link6_label": LINK6_BODY_LABEL,
                "ceiling_fixture_phase": phase in CEILING_FIXTURE_PHASES,
                "tcp_residual_m": residual,
            }
            frames.append(frame)
            if control_step in neighbor_steps:
                onset_window.append(frame)

            primary = pairs[0] if pairs else None
            robot_gid = None if primary is None else primary.get("robot_geom_id")
            xz_overlap = False
            facing = None
            geom_distance = None
            if robot_gid is not None:
                geom_lo, geom_hi = world_aabb_for_geom(model, data, int(robot_gid))
                xz_overlap = _aabb_xz_overlap(geom_lo, geom_hi, pendant_lo, pendant_hi)
                facing = collision_facing_y_m(model, data, int(robot_gid), route_sign)
                if mount_gid is not None:
                    geom_distance = geom_signed_distance_m(
                        model, data, int(robot_gid), int(mount_gid)
                    )
            contacting_body_origin = None
            if primary and primary.get("robot_parent_body_origin_m") is not None:
                contacting_body_origin = primary["robot_parent_body_origin_m"]
            lag_samples.append(
                {
                    "step": control_step,
                    "policy_phase": phase,
                    "ceiling_fixture_phase": phase in CEILING_FIXTURE_PHASES,
                    "xz_overlap_robot_geom_and_pendant": xz_overlap,
                    "has_mounted_fixture_contact": bool(pairs),
                    "tcp_y_m": float(live_tcp[1]),
                    "fr3_link6_body_origin_y_m": None
                    if link6 is None
                    else float(link6[1]),
                    "tcp_to_fr3_link6_body_origin_lateral_m": None
                    if link6 is None
                    else lag_toward_centreline_m(live_tcp[1], link6[1]),
                    "contacting_robot_geom": None
                    if primary is None
                    else primary.get("robot_geom"),
                    "contacting_robot_body": None
                    if primary is None
                    else primary.get("robot_body"),
                    "tcp_to_contacting_robot_geom_body_origin_lateral_m": None
                    if contacting_body_origin is None
                    else lag_toward_centreline_m(
                        live_tcp[1], contacting_body_origin[1]
                    ),
                    "collision_facing_y_m": None
                    if facing is None
                    else facing["collision_facing_y_m"],
                    "tcp_to_collision_facing_extent_lateral_m": None
                    if facing is None
                    else lag_toward_centreline_m(
                        live_tcp[1], facing["collision_facing_y_m"]
                    ),
                    "collision_facing_meta": facing,
                    "signed_robot_geom_to_pendant_geom_distance_m": None
                    if geom_distance is None
                    else geom_distance.get("signed_distance_m"),
                    "geom_distance_meta": geom_distance,
                    "contact_point_y_m": None
                    if primary is None or primary.get("contact_position_m") is None
                    else float(primary["contact_position_m"][1]),
                    "tcp_to_contact_point_lateral_m": None
                    if primary is None or primary.get("contact_position_m") is None
                    else lag_toward_centreline_m(
                        live_tcp[1], primary["contact_position_m"][1]
                    ),
                }
            )
    finally:
        if task is not None or sampler is not None:
            cleanup_task(task, sampler, scratch)

    max_residual = max(residuals) if residuals else None
    valid = reconstruction_is_valid(max_residual)
    base["max_tcp_residual_m"] = max_residual
    base["reconstruction_valid"] = valid
    if not valid:
        base["reconstruction_exclusion_reason"] = (
            None
            if max_residual is None
            else f"tcp_residual_{max_residual:.6f}_m_exceeds_1mm"
        )
        base["frames"] = []
        base["lag_samples"] = []
        base["onset_window"] = [
            {
                "step": item["step"],
                "policy_phase": item["policy_phase"],
                "tcp_residual_m": item.get("tcp_residual_m"),
            }
            for item in onset_window
        ]
        return base

    if first_contact_step is not None and not live_at_onset:
        neighbor_live = any(
            frame["mounted_fixture_pairs"] for frame in onset_window
        )
        base["live_pair_absent_at_control_resolution"] = not neighbor_live
        base["live_pair_limitation"] = (
            "production audit samples 2 ms physics; retained trajectory stores "
            "control-step qpos. No mounted_fixture pair at the authoritative "
            "onset control step"
            + (" or its neighbors." if not neighbor_live else " (present on a neighbor).")
        )
    base["live_pair_at_authoritative_onset"] = live_at_onset
    base["first_reconstructed_contact_step"] = first_live_step
    if first_contact_step is not None and first_live_step is not None:
        base["onset_step_delta"] = int(first_live_step) - int(first_contact_step)
    base["contacting_geoms"] = [
        {"identity": key, "control_frames": int(count)}
        for key, count in contacting_counter.most_common()
    ]
    if first_live_step is not None:
        first_pairs = next(
            (
                frame["mounted_fixture_pairs"]
                for frame in frames
                if frame["step"] == first_live_step
            ),
            [],
        )
        base["first_reconstructed_contacting_pair"] = (
            None if not first_pairs else first_pairs[0]
        )
    else:
        base["first_reconstructed_contacting_pair"] = None
        base["geom_identity"] = "unreconstructed_at_control_resolution"
    onset_pair = None
    onset_pair_step = None
    if first_contact_step is not None:
        exact = next(
            (
                frame
                for frame in onset_window
                if int(frame["step"]) == int(first_contact_step)
                and frame["mounted_fixture_pairs"]
            ),
            None,
        )
        neighbor = next(
            (frame for frame in onset_window if frame["mounted_fixture_pairs"]),
            None,
        )
        chosen = exact or neighbor
        if chosen:
            onset_pair = chosen["mounted_fixture_pairs"][0]
            onset_pair_step = chosen["step"]
    base["onset_contacting_pair"] = onset_pair
    base["onset_contacting_pair_step"] = onset_pair_step
    if onset_pair is None and base.get("first_reconstructed_contacting_pair") is None:
        base["geom_identity"] = "unreconstructed_at_control_resolution"
    base["onset_window"] = onset_window
    # Keep every retained frame, but drop bulky xmat copies already omitted.
    base["frames"] = frames
    base["lag_samples"] = lag_samples
    base["n_frames"] = len(frames)
    base["n_contact_frames"] = sum(
        1 for frame in frames if frame["mounted_fixture_pairs"]
    )
    return base


def _peak_quantity(
    samples: Sequence[Mapping[str, Any]],
    key: str,
    *,
    require_xz_overlap: bool,
    require_contact: bool,
    ceiling_only: bool,
) -> dict[str, Any] | None:
    eligible = []
    for sample in samples:
        value = sample.get(key)
        if value is None:
            continue
        if require_contact and not sample.get("has_mounted_fixture_contact"):
            continue
        if require_xz_overlap and not sample.get(
            "xz_overlap_robot_geom_and_pendant"
        ):
            continue
        if ceiling_only and not sample.get("ceiling_fixture_phase"):
            continue
        eligible.append(sample)
    if not eligible:
        return None
    peak = max(eligible, key=lambda item: abs(float(item["tcp_y_m"])))
    return {
        "step": peak["step"],
        "policy_phase": peak["policy_phase"],
        "tcp_y_m": peak["tcp_y_m"],
        "value_m": float(peak[key]),
        "n_eligible_frames": len(eligible),
        "eligibility": {
            "require_xz_overlap": require_xz_overlap,
            "require_contact": require_contact,
            "ceiling_only": ceiling_only,
        },
    }


def summarize_lag(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid = lag_rows_for_aggregate(rows)
    definitions: dict[str, Any] = {}
    for key in NAMED_LAG_QUANTITIES + ("tcp_to_contact_point_lateral_m",):
        collision_relevant = key in {
            "tcp_to_collision_facing_extent_lateral_m",
            "signed_robot_geom_to_pendant_geom_distance_m",
            "tcp_to_contact_point_lateral_m",
        }
        rules = (
            {
                "name": "collision_relevant_xz_overlap_and_contact",
                "require_xz_overlap": True,
                "require_contact": True,
                "ceiling_only": False,
            },
            {
                "name": "ceiling_phases_only_not_collision_relevant",
                "require_xz_overlap": False,
                "require_contact": False,
                "ceiling_only": True,
            },
        )
        rule_blocks = []
        for rule in rules:
            peaks: dict[str, list[float]] = {"neg": [], "pos": []}
            row_peaks = []
            for row in valid:
                peak = _peak_quantity(
                    row.get("lag_samples") or [],
                    key,
                    require_xz_overlap=rule["require_xz_overlap"],
                    require_contact=rule["require_contact"],
                    ceiling_only=rule["ceiling_only"],
                )
                if peak is None:
                    continue
                sign = "neg" if float(peak["tcp_y_m"]) < 0.0 else "pos"
                peaks[sign].append(float(peak["value_m"]))
                row_peaks.append(
                    {
                        "run_id": row.get("run_id"),
                        "role_index": row.get("role_index"),
                        "intrusion_side": row.get("intrusion_side"),
                        "sign": sign,
                        **peak,
                    }
                )
            reproduced = definition_reproduces_provenance(peaks)
            rule_blocks.append(
                {
                    "rule": rule["name"],
                    "collision_relevant": collision_relevant
                    and rule["require_xz_overlap"],
                    "reproduces_design_provenance": reproduced,
                    "n_neg": len(peaks["neg"]),
                    "n_pos": len(peaks["pos"]),
                    "neg_min_m": min(peaks["neg"]) if peaks["neg"] else None,
                    "neg_max_m": max(peaks["neg"]) if peaks["neg"] else None,
                    "pos_min_m": min(peaks["pos"]) if peaks["pos"] else None,
                    "pos_max_m": max(peaks["pos"]) if peaks["pos"] else None,
                    "design_neg_m": DESIGN_LAG_NEG_M,
                    "design_pos_m": DESIGN_LAG_POS_M,
                    "rows": row_peaks,
                }
            )
        definitions[key] = {
            "quantity": key,
            "body_origin": key.endswith("body_origin_lateral_m"),
            "collision_relevant_candidate": collision_relevant,
            "rules": rule_blocks,
        }
    reproduced_any = False
    reproduced_name = None
    contributing = None
    for quantity, block in definitions.items():
        for rule in block["rules"]:
            if rule["reproduces_design_provenance"] and rule["collision_relevant"]:
                reproduced_any = True
                reproduced_name = {
                    "quantity": quantity,
                    "rule": rule["rule"],
                }
                contributing = rule["rows"]
                break
        if reproduced_any:
            break
    return {
        "n_rows_considered": len(rows),
        "n_rows_valid_reconstruction": len(valid),
        "n_rows_excluded_for_residual": len(rows) - len(valid),
        "definitions": definitions,
        "lag_reproduced": reproduced_any,
        "reproduced_definition": reproduced_name,
        "contributing_frames": contributing,
        "excluded_rows": [
            {
                "run_id": row.get("run_id"),
                "role_index": row.get("role_index"),
                "max_tcp_residual_m": row.get("max_tcp_residual_m"),
                "reason": row.get("reconstruction_exclusion_reason"),
            }
            for row in rows
            if not row.get("reconstruction_valid")
        ],
    }


def compact_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "run_id": row.get("run_id"),
        "candidate": row.get("candidate"),
        "kind": row.get("kind"),
        "role_index": row.get("role_index"),
        "family": row.get("family"),
        "intrusion_side": row.get("intrusion_side"),
        "episode_id": row.get("episode_id"),
        "status": row.get("status"),
        "baseline_clean_success": row.get("baseline_clean_success"),
        "authoritative_first_contact_step": row.get(
            "authoritative_first_contact_step"
        ),
        "retained_policy_phase_at_first_contact": row.get(
            "retained_policy_phase_at_first_contact"
        ),
        "terminal_policy_phase": row.get("terminal_policy_phase"),
        "onset_category": row.get("onset_category"),
        "first_reconstructed_contact_step": row.get(
            "first_reconstructed_contact_step"
        ),
        "onset_step_delta": row.get("onset_step_delta"),
        "live_pair_at_authoritative_onset": row.get(
            "live_pair_at_authoritative_onset"
        ),
        "live_pair_absent_at_control_resolution": row.get(
            "live_pair_absent_at_control_resolution"
        ),
        "live_pair_limitation": row.get("live_pair_limitation"),
        "first_reconstructed_contacting_pair": row.get(
            "first_reconstructed_contacting_pair"
        ),
        "onset_contacting_pair": row.get("onset_contacting_pair"),
        "onset_contacting_pair_step": row.get("onset_contacting_pair_step"),
        "contacting_geoms": row.get("contacting_geoms"),
        "geom_identity": row.get("geom_identity"),
        "mounted_fixture_contact_entries": row.get(
            "mounted_fixture_contact_entries"
        ),
        "mounted_fixture_max_penetration_m": row.get(
            "mounted_fixture_max_penetration_m"
        ),
        "reconstruction_valid": row.get("reconstruction_valid"),
        "max_tcp_residual_m": row.get("max_tcp_residual_m"),
        "reconstruction_exclusion_reason": row.get(
            "reconstruction_exclusion_reason"
        ),
        "n_frames": row.get("n_frames"),
        "n_contact_frames": row.get("n_contact_frames"),
        "patched_episode_id": row.get("patched_episode_id"),
        "patched_layout_id": row.get("patched_layout_id"),
        "result_sha256": row.get("result_sha256"),
        "trajectory_sha256": row.get("trajectory_sha256"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--assemble-only",
        action="store_true",
        help="Rebuild diagnosis.json from existing row JSON files; no mj_forward.",
    )
    args = parser.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("MLSPACES_ASSETS_DIR", str(ROOT / "assets"))

    output_root = args.output_root.resolve()
    if output_root.exists() and output_root.samefile(
        ROOT / "diagnostics_output/pact_place_v98_paired_offset_wide"
    ):
        raise SystemExit("refusing to write into an immutable V9.8 result directory")
    output_root.mkdir(parents=True, exist_ok=True)
    rows_out = output_root / "rows"
    rows_out.mkdir(parents=True, exist_ok=True)

    source = json.loads(SOURCE_SUMMARY.read_text())
    input_hashes = {
        relative: sha256_file(ROOT / relative)
        for relative in IMMUTABLE_JSON_RELATIVE
    }

    jobs: list[dict[str, Any]] = []
    reconstructed: list[dict[str, Any]] = []
    if args.assemble_only:
        reconstructed = [
            json.loads(path.read_text())
            for path in sorted(rows_out.glob("*.json"))
        ]
        if not reconstructed:
            raise SystemExit(f"no row JSON files in {rows_out}")
    else:
        for spec in RUN_SPECS:
            paired = json.loads((ROOT / spec["paired"]).read_text())
            comparisons = {
                int(item["role_index"]): item for item in paired.get("comparisons") or []
            }
            rows_root = ROOT / spec["rows_root"]
            for row_dir in sorted(path for path in rows_root.iterdir() if path.is_dir()):
                result_path = row_dir / "result.json"
                if not result_path.is_file():
                    continue
                result = json.loads(result_path.read_text())
                episode_id = str(result["episode_id"])
                comparison = comparisons.get(int(result["role_index"]), {})
                if result.get("status") != "complete" or not (row_dir / "trajectory.json").is_file():
                    jobs.append(
                        {
                            "skip_reconstruct": True,
                            "run_id": spec["run_id"],
                            "candidate": spec["candidate"],
                            "kind": spec["kind"],
                            "row_dir": str(row_dir),
                            "comparison": comparison,
                            "result": result,
                        }
                    )
                    continue
                manifest = lookup_manifest(source, episode_id)
                patched = patch_manifest_for_row(
                    manifest,
                    result,
                    sampler_class=SAMPLER_CLASS,
                    contract_version=CONTRACT_VERSION,
                )
                jobs.append(
                    {
                        "skip_reconstruct": False,
                        "run_id": spec["run_id"],
                        "candidate": spec["candidate"],
                        "kind": spec["kind"],
                        "row_dir": str(row_dir),
                        "comparison": comparison,
                        "patched_row": patched,
                    }
                )
        reconstruct_jobs = [job for job in jobs if not job.get("skip_reconstruct")]
        skipped = [job for job in jobs if job.get("skip_reconstruct")]
        context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max(1, int(args.workers)),
            mp_context=context,
            max_tasks_per_child=1,
        ) as executor:
            futures = [
                executor.submit(
                    diagnose_one_row,
                    {
                        "run_id": job["run_id"],
                        "candidate": job["candidate"],
                        "kind": job["kind"],
                        "row_dir": job["row_dir"],
                        "comparison": job["comparison"],
                        "patched_row": job["patched_row"],
                    },
                )
                for job in reconstruct_jobs
            ]
            for future in concurrent.futures.as_completed(futures):
                row = future.result()
                reconstructed.append(row)
                print(
                    f"{row.get('run_id')} role={row.get('role_index')} "
                    f"valid={row.get('reconstruction_valid')} "
                    f"onset={row.get('onset_category')} "
                    f"residual={row.get('max_tcp_residual_m')}",
                    flush=True,
                )

        for job in skipped:
            result = job["result"]
            comparison = job["comparison"]
            reconstructed.append(
                {
                    "run_id": job["run_id"],
                    "candidate": job["candidate"],
                    "kind": job["kind"],
                    "role_index": result.get("role_index"),
                    "family": comparison.get("family"),
                    "intrusion_side": result.get("intrusion_side")
                    or comparison.get("intrusion_side"),
                    "episode_id": result.get("episode_id"),
                    "status": result.get("status"),
                    "baseline_clean_success": bool(comparison.get("baseline_clean_success")),
                    "terminal_policy_phase": result.get("terminal_policy_phase"),
                    "authoritative_first_contact_step": (
                        (result.get("contact_audit") or {}).get("first_contact_step") or {}
                    ).get("mounted_fixture"),
                    "onset_category": "unreconstructed",
                    "reconstruction_valid": False,
                    "reconstruction_exclusion_reason": "no_complete_trajectory",
                    "result_sha256": sha256_file(Path(job["row_dir"]) / "result.json"),
                    "physics_stepped": False,
                    "episodes_ran": False,
                }
            )

        reconstructed.sort(
            key=lambda item: (str(item.get("run_id")), int(item.get("role_index") or -1))
        )
        for row in reconstructed:
            name = f"{row.get('run_id')}_{int(row.get('role_index') or 0):03d}.json"
            (rows_out / name).write_text(
                json.dumps(row, indent=2, sort_keys=True) + "\n"
            )

    offset_rows = [row for row in reconstructed if row.get("kind") == "offset"]
    centred_rows = [row for row in reconstructed if row.get("kind") == "centred_source"]
    lag_centred = summarize_lag(centred_rows)
    lag_offset = summarize_lag(offset_rows)
    lag_reproduced = bool(lag_centred.get("lag_reproduced"))

    joins = {}
    for candidate in ("wide", "cons"):
        rows = [row for row in offset_rows if row.get("candidate") == candidate]
        complete = [row for row in rows if row.get("status") == "complete"]
        baseline = [row for row in complete if row.get("baseline_clean_success")]
        reconstruction_ok = all(row.get("reconstruction_valid") for row in baseline)
        joins[candidate] = {
            "n_complete": len(complete),
            "n_baseline_clean_complete": len(baseline),
            "onset_all_complete": count_categories(complete, baseline_clean_only=False),
            "onset_baseline_clean": count_categories(
                complete, baseline_clean_only=True
            ),
            "by_side": {
                side: {
                    "all_complete": count_categories(
                        [row for row in complete if row.get("intrusion_side") == side],
                        baseline_clean_only=False,
                    ),
                    "baseline_clean": count_categories(
                        [row for row in complete if row.get("intrusion_side") == side],
                        baseline_clean_only=True,
                    ),
                    "rows": [
                        {
                            "role_index": row.get("role_index"),
                            "onset_category": row.get("onset_category"),
                            "first_contact_phase": row.get(
                                "retained_policy_phase_at_first_contact"
                            ),
                            "terminal_policy_phase": row.get("terminal_policy_phase"),
                            "authoritative_first_contact_step": row.get(
                                "authoritative_first_contact_step"
                            ),
                            "live_pair_at_authoritative_onset": row.get(
                                "live_pair_at_authoritative_onset"
                            ),
                            "live_pair_absent_at_control_resolution": row.get(
                                "live_pair_absent_at_control_resolution"
                            ),
                            "onset_contacting_pair": row.get("onset_contacting_pair"),
                            "onset_contacting_pair_step": row.get(
                                "onset_contacting_pair_step"
                            ),
                            "first_later_reconstructed_pair": row.get(
                                "first_reconstructed_contacting_pair"
                            )
                            if not row.get("onset_contacting_pair")
                            else None,
                            "first_reconstructed_contact_step": row.get(
                                "first_reconstructed_contact_step"
                            ),
                            "geom_identity": (
                                None
                                if row.get("onset_contacting_pair")
                                else "unreconstructed_at_control_resolution"
                                if row.get("live_pair_absent_at_control_resolution")
                                else row.get("geom_identity")
                            ),
                            "reconstruction_valid": row.get("reconstruction_valid"),
                            "max_tcp_residual_m": row.get("max_tcp_residual_m"),
                        }
                        for row in complete
                        if row.get("intrusion_side") == side
                    ],
                }
                for side in ("left", "right")
            },
            "reconstruction_ok_for_baseline_clean": reconstruction_ok,
        }

    baseline_onsets = []
    reconstruction_ok = True
    for candidate in ("wide", "cons"):
        rows = [
            row
            for row in offset_rows
            if row.get("candidate") == candidate
            and row.get("status") == "complete"
            and row.get("baseline_clean_success")
        ]
        baseline_onsets.extend(str(row.get("onset_category")) for row in rows)
        reconstruction_ok = reconstruction_ok and all(
            row.get("reconstruction_valid") for row in rows
        )

    causal = select_causal_category(
        baseline_clean_onset_categories=baseline_onsets,
        lag_reproduced=lag_reproduced,
        reconstruction_ok_for_baseline_clean=reconstruction_ok,
        protected_clearance_violated=False,
    )
    if lag_reproduced:
        lag_status = "reproduced"
        window_status = "physical_input_verified"
        predictor_role = (
            "algebra_dispatch_and_named_lag_definition; not a swept-volume proof"
        )
    else:
        lag_status = "unverified_provenance"
        window_status = "physical_input_invalid"
        predictor_role = (
            "live _bow_segment algebra, dispatch, and no-clip only; "
            "not swept-arm clearance"
        )

    diagnosis = {
        "schema_version": SCHEMA_VERSION,
        "working_conclusion": WORKING_CONCLUSION,
        "causal_category": causal,
        **empty_authorization(),
        "scene_xml": str(SCENE_XML.relative_to(ROOT)),
        "input_sha256": input_hashes,
        "paired_selection_stop": {
            "wide_baseline_clean_rows_preserved": False,
            "cons_baseline_clean_rows_preserved": False,
            "gate_ran": False,
            "s2b_ran": False,
            "s4_ran": False,
        },
        "lag_provenance": {
            "design_neg_m": DESIGN_LAG_NEG_M,
            "design_pos_m": DESIGN_LAG_POS_M,
            "status": lag_status,
            "face_window_m": list(FACE_WINDOW_M),
            "face_window_status": window_status,
            "predictor_role": predictor_role,
            "centred_source": lag_centred,
            "offset_diagnostic_only": lag_offset,
            "did_not_substitute_body_origin_into_design_formula": True,
            "did_not_derive_new_envelopes_or_candidates": True,
        },
        "offset_joins": joins,
        "rows": [compact_row(row) for row in reconstructed],
        "output_row_dir": str((output_root / "rows").relative_to(ROOT)),
        "document_sha256": None,
    }
    diagnosis["document_sha256"] = sha256_payload(diagnosis)
    out_path = output_root / "diagnosis.json"
    out_path.write_text(json.dumps(diagnosis, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "path": str(out_path),
                "sha256": diagnosis["document_sha256"],
                "causal_category": causal,
                "lag_status": lag_status,
                "face_window_status": window_status,
                "authorizes_new_episodes": False,
                "authorizes_gate": False,
                "authorizes_collection": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
