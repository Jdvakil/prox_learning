#!/usr/bin/env python3
"""V10.11 96-layout settle, contact, geometry and full-IK preflight."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import tempfile
import traceback
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
ACTIVE_CLUTTER_COUNT = _contract.ACTIVE_CLUTTER_COUNT
ACTIVE_CLUTTER_SLOTS = _contract.ACTIVE_CLUTTER_SLOTS
CONTRACT_VERSION = _contract.CONTRACT_VERSION
DISPLAY_VERSION = getattr(_contract, "DISPLAY_VERSION", "V10.11")
ENVIRONMENT_VERSION = _contract.ENVIRONMENT_VERSION
INACTIVE_CLUTTER_SLOTS = _contract.INACTIVE_CLUTTER_SLOTS
PREFLIGHT_ROOT = _contract.PREFLIGHT_ROOT
PRIMITIVE_SLOTS = _contract.PRIMITIVE_SLOTS
canonical_payload_sha256 = _contract.canonical_payload_sha256
empty_authorization = _contract.empty_authorization
preflight_rows = _contract.preflight_rows
write_immutable_create_only = _contract.write_immutable_create_only


def _retry_seed(row: dict[str, Any], retry: int) -> tuple[int, int]:
    import hashlib

    digest = hashlib.sha256(
        f"{row['row_sha256']}:preflight-retry:{int(retry)}".encode()
    ).digest()
    value = int.from_bytes(digest[:8], "big")
    return value % (2**32), value


def check_row(row: dict[str, Any]) -> dict[str, Any]:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("MLSPACES_ASSETS_DIR", str(ROOT / "assets"))
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.pop("DISPLAY", None)

    import hashlib
    import mujoco

    from molmo_spaces.data_generation.pipeline import (
        cleanup_episode_resources,
        setup_policy,
    )
    from molmo_spaces.env.abstract_sensors import SensorSuite
    from molmo_spaces.tasks.pact_place_contact_audit import classify_contact
    from molmo_spaces.tasks.pact_contact_audit import robot_environment_contact_pairs
    from run_pact_place_expert_screen import _make_config

    problems: list[str] = []
    detail: dict[str, Any] = {"sampling_rejections": []}
    sampler = task = policy = None
    scene = ROOT / row["pact_v1011_scene_relative"]
    try:
        observed_scene = hashlib.sha256(scene.read_bytes()).hexdigest()
        detail["scene_sha256"] = observed_scene
        if observed_scene != row["pact_v106_scene_sha256"]:
            problems.append("certified scene hash mismatch")
        for retry in range(int(row["max_sampling_retries"]) + 1):
            seed_u32, seed_u64 = (
                (int(row["task_seed_u32"]), int(row["task_seed_u64"]))
                if retry == 0
                else _retry_seed(row, retry)
            )
            scratch = Path(tempfile.gettempdir()) / "pact_v1011_preflight" / row["episode_id"]
            config = _make_config(
                scratch / "result.json",
                scene_xml=scene,
                sampler_class=row["sampler_class"],
            )
            try:
                sampler = config.task_sampler_config.task_sampler_class(config)
                sampler.seed_task_sampling(seed_u32)
                sampler.set_pact_manifest_row(row)
                task = sampler.sample_task(house_index=int(row["scene_template_house_index"]))
                if task is None:
                    raise RuntimeError("sample_task returned None")
                initial_pairs = robot_environment_contact_pairs(task.env)
                forbidden = [
                    pair
                    for pair in initial_pairs
                    if classify_contact(pair)
                    in {"hazard_bar", "other_environment", "clutter", "mounted_fixture"}
                ]
                if forbidden:
                    raise RuntimeError(
                        f"forbidden initial contact: {classify_contact(forbidden[0])}"
                    )
                policy = setup_policy(config, task, None, None)
                task._sensor_suite = SensorSuite(
                    [task._sensor_suite.sensors[name] for name in ("qpos", "tcp_pose")]
                )
                task.reset()  # resets the expert and therefore builds its complete IK route
                detail["accepted_retry"] = retry
                detail["selected_seed_u32"] = seed_u32
                detail["selected_seed_u64"] = seed_u64
                break
            except Exception as exc:  # noqa: BLE001 - bounded pre-boundary retry
                detail["sampling_rejections"].append(
                    {
                        "retry": retry,
                        "seed_u32": seed_u32,
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
                cleanup_episode_resources(
                    task=task,
                    policy=policy,
                    task_sampler=sampler,
                    preloaded_policy=None,
                    close_task_sampler=sampler is not None,
                )
                sampler = task = policy = None
        if task is None or sampler is None or policy is None:
            problems.append("sampling/full-IK retries exhausted")
            return {
                "role_index": row["role_index"],
                "cell": row["cell"],
                "replicate": row["replicate"],
                "passed": False,
                "problems": problems,
                "detail": detail,
            }

        env = task.env
        model, data = env.current_model, env.current_data
        mujoco.mj_forward(model, data)
        params = task.scene_params
        detail["environment_version"] = params.get("pact_place_environment_version")
        if detail["environment_version"] != ENVIRONMENT_VERSION:
            problems.append(f"wrong environment marker {detail['environment_version']!r}")
        if params.get("pact_v1011_identity_sha256") != row["pact_v1011_identity_sha256"]:
            problems.append("sampler identity differs from row binding")
        if int(params.get("pact_v1011_target_draw_count") or 0) != 1:
            problems.append("target was not drawn exactly once")

        active = list(sampler._pact_active_clutter_names)
        detail["active_bodies"] = active
        if len(active) != ACTIVE_CLUTTER_COUNT:
            problems.append(f"active clutter count {len(active)} != {ACTIVE_CLUTTER_COUNT}")
        active_layout = dict(sampler._pact_active_clutter_layout)
        active_slots = sorted(str(item["palette_slot"]) for item in active_layout.values())
        detail["active_slots"] = active_slots
        if active_slots != sorted(ACTIVE_CLUTTER_SLOTS):
            problems.append(f"active slots {active_slots}")
        all_bodies = {str(item["slot"]): str(item["body"])
                      for item in sampler._pact_clutter_objects}
        parked_slots = sorted(set(all_bodies) - set(active_slots))
        detail["parked_slots"] = parked_slots
        if parked_slots != sorted(INACTIVE_CLUTTER_SLOTS):
            problems.append(f"parked slots {parked_slots}")
        for slot in parked_slots:
            z = float(data.xpos[int(model.body(all_bodies[slot]).id), 2])
            if z > 0.0:
                problems.append(f"parked slot {slot} has z={z:.3f}")

        settle = dict(params.get("pact_clutter_settle") or {})
        detail["settle_stable"] = settle.get("stable_at_step0")
        if not settle.get("stable_at_step0"):
            problems.append("settle stability marker missing")
        settled = list(settle.get("objects") or [])
        if len(settled) != ACTIVE_CLUTTER_COUNT:
            problems.append(f"settled object count {len(settled)}")
        detail["max_settled_xy_drift_m"] = max(
            (float(item["xy_drift_m"]) for item in settled), default=None
        )
        detail["max_settled_linear_speed_m_s"] = max(
            (float(item["linear_speed_m_s"]) for item in settled), default=None
        )
        detail["max_settled_angular_speed_rad_s"] = max(
            (float(item["angular_speed_rad_s"]) for item in settled), default=None
        )

        workspace_low = np.asarray(sampler.CLUTTER_WORKSPACE_LOW, dtype=float)
        workspace_high = np.asarray(sampler.CLUTTER_WORKSPACE_HIGH, dtype=float)
        body_by_slot = {str(item["slot"]): str(item["body"])
                        for item in sampler._pact_clutter_objects}
        measured: dict[str, Any] = {}
        for slot in ACTIVE_CLUTTER_SLOTS:
            body = body_by_slot[slot]
            low, high = sampler._body_collision_aabb(model, data, body)
            measured[slot] = {
                "body": body,
                "low_m": low.tolist(),
                "high_m": high.tolist(),
                "dimensions_m": (high - low).tolist(),
            }
            if np.any(low[:2] < workspace_low[:2] - 1e-4) or np.any(
                high > workspace_high + 1e-4
            ):
                problems.append(f"active slot {slot} escapes workspace")
        detail["measured_active_bounds"] = measured

        primitive_checks = {}
        palette = {str(item["slot"]): item for item in row["pact_clutter_palette"]}
        for slot in PRIMITIVE_SLOTS:
            body = body_by_slot[slot]
            body_id = int(model.body(body).id)
            geom_ids = [gid for gid in range(model.ngeom)
                        if int(model.geom_bodyid[gid]) == body_id]
            expected_type = (
                int(mujoco.mjtGeom.mjGEOM_BOX)
                if palette[slot]["primitive"]["shape"] == "box"
                else int(mujoco.mjtGeom.mjGEOM_CYLINDER)
            )
            types = [int(model.geom_type[gid]) for gid in geom_ids]
            primitive_checks[slot] = {
                "body": body,
                "geom_names": [model.geom(gid).name for gid in geom_ids],
                "geom_types": types,
                "expected_type": expected_type,
                "expected_dimensions_m": [
                    float(value) for value in palette[slot]["dimensions_m"]
                ],
            }
            if not geom_ids or any(value != expected_type for value in types):
                problems.append(f"slot {slot} primitive geom type mismatch")
            joint = int(model.body_jntadr[body_id])
            if joint < 0 or int(model.jnt_type[joint]) != int(mujoco.mjtJoint.mjJNT_FREE):
                problems.append(f"slot {slot} is not a free body")
            if "pact_clutter_mount_" in body or not body.startswith(f"pact_clutter_{slot}/"):
                problems.append(f"slot {slot} has unsafe contact namespace {body!r}")
            if len(geom_ids) == 1:
                size = np.asarray(model.geom_size[geom_ids[0]], dtype=float)
                compiled_dimensions = (
                    2.0 * size[:3]
                    if expected_type == int(mujoco.mjtGeom.mjGEOM_BOX)
                    else np.asarray([2.0 * size[0], 2.0 * size[0], 2.0 * size[1]])
                )
                primitive_checks[slot]["compiled_dimensions_m"] = (
                    compiled_dimensions.tolist()
                )
                if not np.allclose(
                    compiled_dimensions,
                    np.asarray(palette[slot]["dimensions_m"], dtype=float),
                    atol=1e-12,
                    rtol=0.0,
                ):
                    problems.append(f"slot {slot} compiled dimensions mismatch")
            else:
                problems.append(f"slot {slot} has {len(geom_ids)} collision geoms")
        detail["primitive_checks"] = primitive_checks

        target_body = str(sampler._injected_obj_name)
        target_low, target_high = sampler._body_collision_aabb(model, data, target_body)
        detail["measured_target_aabb_m"] = {
            "body": target_body,
            "low": target_low.tolist(),
            "high": target_high.tolist(),
            "dimensions": (target_high - target_low).tolist(),
            "xy_radius_max_half_extent_m": float(max((target_high - target_low)[:2]) / 2.0),
        }
        detail["primitive_heights_m"] = {
            slot: measured[slot]["dimensions_m"][2] for slot in PRIMITIVE_SLOTS
        }

        layout = dict(params["pact_clutter_layout"])
        near_target = dict(layout.get("near_target_placements") or {})
        detail["near_target_placements"] = near_target
        for slot in ("08", "09"):
            placement = dict(near_target.get(slot) or {})
            expected_object_radius = (
                0.035 if slot == "08" else float(np.hypot(0.035, 0.035))
            )
            expected_radius_min = (
                float(placement.get("target_planar_bounding_radius_m", -1.0))
                + expected_object_radius
                + float(sampler.NEAR_TARGET_GAP_M)
            )
            if not np.isclose(
                float(placement.get("object_planar_bounding_radius_m", -1.0)),
                expected_object_radius,
                atol=1e-12,
                rtol=0.0,
            ):
                problems.append(f"near-target slot {slot} has wrong planar radius")
            if not np.isclose(
                float(placement.get("radius_min_m", -1.0)),
                expected_radius_min,
                atol=1e-12,
                rtol=0.0,
            ):
                problems.append(f"near-target slot {slot} has wrong annulus floor")
            if float(placement.get("radius_m", -1.0)) < expected_radius_min - 1e-12:
                problems.append(f"near-target slot {slot} violates annulus floor")
        route = dict(layout.get("nominal_route_metrics") or {})
        corridor = dict(layout.get("panel_corridor_metrics") or {})
        detail["route_metrics"] = route
        detail["panel_corridor_metrics"] = corridor
        if not route.get("direct_route_blocked") or not route.get("detour_admitted"):
            problems.append("new vessel route predicate failed")
        if not corridor.get("detour_admitted"):
            problems.append("new vessel panel corridor failed")

        pairs = robot_environment_contact_pairs(env)
        classes: dict[str, int] = {}
        for pair in pairs:
            kind = classify_contact(pair)
            classes[kind] = classes.get(kind, 0) + 1
        detail["post_ik_initial_contact_classes"] = classes
        for kind in ("hazard_bar", "other_environment", "clutter", "mounted_fixture"):
            if classes.get(kind):
                problems.append(f"post-IK initial contact {kind}={classes[kind]}")

        detail["full_ik"] = {
            "action_primitives": len(policy.action_primitives),
            "target_poses": len(policy.target_poses),
            "sequential_ik_failures": int(policy.sequential_ik_failures),
        }
        if not policy.action_primitives or not policy.target_poses:
            problems.append("full expert route did not materialize")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"preflight raised: {type(exc).__name__}: {exc}")
        detail["traceback"] = traceback.format_exc()
    finally:
        if task is not None or policy is not None or sampler is not None:
            try:
                cleanup_episode_resources(
                    task=task,
                    policy=policy,
                    task_sampler=sampler,
                    preloaded_policy=None,
                    close_task_sampler=sampler is not None,
                )
            except Exception as exc:  # noqa: BLE001
                detail["cleanup_warning"] = f"{type(exc).__name__}: {exc}"
    return {
        "role_index": row["role_index"],
        "cell": row["cell"],
        "replicate": row["replicate"],
        "row_sha256": row["row_sha256"],
        "passed": not problems,
        "problems": problems,
        "detail": detail,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=ROOT / PREFLIGHT_ROOT / "preflight.json")
    args = parser.parse_args()
    rows = preflight_rows()
    if args.limit is not None:
        rows = rows[: max(0, int(args.limit))]
    print(
        f"{DISPLAY_VERSION} preflight: {len(rows)} rows on {args.workers} workers",
        flush=True,
    )
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(check_row, row): row for row in rows}
        for index, future in enumerate(as_completed(futures), 1):
            row = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001
                result = {
                    "role_index": row["role_index"],
                    "cell": row["cell"],
                    "replicate": row["replicate"],
                    "row_sha256": row["row_sha256"],
                    "passed": False,
                    "problems": [f"worker died: {type(exc).__name__}: {exc}"],
                    "detail": {},
                }
            results.append(result)
            print(
                f"  {index}/{len(rows)} role={result['role_index']} "
                f"{'PASS' if result['passed'] else 'FAIL'}",
                flush=True,
            )
    results.sort(key=lambda item: int(item["role_index"]))
    failed = [item for item in results if not item["passed"]]
    target_dims = [
        item["detail"].get("measured_target_aabb_m", {}).get("dimensions")
        for item in results
        if item["detail"].get("measured_target_aabb_m")
    ]
    document = {
        **empty_authorization(),
        "schema_version": f"{CONTRACT_VERSION}_preflight",
        "contract_version": CONTRACT_VERSION,
        "environment_version": ENVIRONMENT_VERSION,
        "expected_rows": len(preflight_rows()),
        "rows_checked": len(results),
        "rows_passed": len(results) - len(failed),
        "rows_failed": len(failed),
        "complete_registered_preflight": len(results) == len(preflight_rows()),
        "passed": len(results) == len(preflight_rows()) and not failed,
        "measured_target_aabb_dimensions_m": target_dims,
        "results": results,
    }
    document["payload_sha256"] = canonical_payload_sha256(document)
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    write_immutable_create_only(args.output, document)
    print(
        json.dumps(
            {
                "passed": document["passed"],
                "rows_passed": document["rows_passed"],
                "rows_failed": document["rows_failed"],
                "first_failures": [
                    {"role": item["role_index"], "problems": item["problems"][:2]}
                    for item in failed[:5]
                ],
                "payload_sha256": document["payload_sha256"],
            },
            indent=2,
        )
    )
    return 0 if document["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
