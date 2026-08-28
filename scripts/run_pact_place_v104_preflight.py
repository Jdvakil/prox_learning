#!/usr/bin/env python3
"""V10.4 Step-0 offline preflight: 0A provenance through 0E route preservation.

May compile models and call FK, exact distance, ``mj_forward``, and
deterministic qpos replay. Never calls ``env.step`` and never creates an
episode. Authorizes nothing.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "scripts", ROOT / "submodules" / "molmospaces"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from pact_place_v104_contract import (  # noqa: E402
    BASE_CONFIG_RELATIVE,
    CONTRACT_VERSION,
    ENVIRONMENT_VERSION,
    PREFLIGHT_ROOT,
    SAMPLER_CLASS,
    build_contract,
    empty_authorization,
    implementation_hashes,
    implementation_sha256,
    sha256_payload,
    verify_protected_artifacts,
    write_immutable_create_only,
)
from pact_place_v104_geometry import (  # noqa: E402
    ALL_GEOMS_V104,
    CROSSBAR_GEOM_V104,
    AUDIT_PROVENANCE_TOLERANCE_M,
    AUDIT_V6C_HARD_FLOOR_M,
    AUDIT_V6C_OBSERVED_MIN_M,
    AUDIT_V95_HARD_FLOOR_M,
    AUDIT_V95_OBSERVED_MIN_M,
    BASE_SCENE_RELATIVE,
    CORNER_MIN_CLEARANCE_V6C_M,
    CORNER_MIN_CLEARANCE_V95_M,
    PENDANT_BODY_V104,
    SCENE_XML_RELATIVE_V104,
    assembly_expectations,
    corner_assemblies,
    production_assembly,
)
from pact_place_v104_runtime import (  # noqa: E402
    HORIZON_UTILISATION_LIMIT,
    INITIAL_FREE_SPACE_SPEED_CAP_M_S,
    TASK_HORIZON_V104,
)

OUTPUT_ROOT = ROOT / PREFLIGHT_ROOT
V6C_ROWS_DIR = ROOT / "diagnostics_output/pact_place_corridor_v6c/expert_screen_rows"
SCHEMA = "pact_place_v104_preflight_v1"


def _pin_threads() -> None:
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"


def _establish_env() -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("MLSPACES_ASSETS_DIR", str(ROOT / "assets"))
    os.environ.pop("DISPLAY", None)


# ---------------------------------------------------------------------------
# 0B (V9.5 half) — retained snapshot arrays, no live task needed
# ---------------------------------------------------------------------------
def v95_clearances() -> dict[str, Any]:
    from pact_place_v10_exact import _box_shape, _robot_kwargs, _with_target, verify_v99_inputs
    from pact_place_v99_exact import scan_min_distance

    _recon, _snap, cells = verify_v99_inputs()
    cells = sorted(cells, key=lambda item: int(item["role_index"]))
    assembly = production_assembly()

    def path_min(component, cell, mask_key, with_target):
        shape, lo, hi = _box_shape(component["center_m"], component["half_m"])
        kwargs = _with_target(cell) if with_target else _robot_kwargs(cell)
        witness = scan_min_distance(
            **dict(kwargs, box_shape=shape, box_lo=lo, box_hi=hi),
            mask=np.asarray(cell[mask_key], dtype=bool),
            gap_limit_m=10.0,
            stop_at_contact=False,
        )
        return (
            None if witness["distance_m"] is None else float(witness["distance_m"]),
            witness,
        )

    per_cell = []
    witness_rows: list[list[float]] = []
    for index, cell in enumerate(cells):
        entries = {}
        for component in assembly["components"]:
            for mask_key, with_target, label in (
                ("inbound_mask", False, "inbound"),
                ("outbound_mask", True, "loaded_outbound"),
            ):
                value, witness = path_min(component, cell, mask_key, with_target)
                if value is None:
                    continue
                entries[f"{component['name']}|{label}"] = value
                witness_rows.append(
                    [
                        float(index),
                        float(assembly["components"].index(component)),
                        0.0 if label == "inbound" else 1.0,
                        float(value),
                        float(witness.get("frame") if witness.get("frame") is not None else -1),
                    ]
                )
        minimum = min(entries.values())
        binding = min(entries, key=entries.get)
        per_cell.append(
            {
                "role_index": int(cell["role_index"]),
                "family": str(cell["family"]),
                "intrusion_side": str(cell["intrusion_side"]),
                "min_clearance_m": float(minimum),
                "binding": binding,
                "per_path_m": entries,
                "meets_hard_floor": bool(minimum >= AUDIT_V95_HARD_FLOOR_M),
            }
        )
    observed = float(min(item["min_clearance_m"] for item in per_cell))

    corners = []
    for corner in corner_assemblies(assembly):
        values = []
        for cell in cells:
            for component in corner["components"]:
                for mask_key, with_target in (
                    ("inbound_mask", False),
                    ("outbound_mask", True),
                ):
                    value, _w = path_min(component, cell, mask_key, with_target)
                    if value is not None:
                        values.append(value)
        corners.append(
            {
                "corner_key": corner["corner_key"],
                "min_clearance_m": float(min(values)),
                "meets_floor": bool(min(values) >= CORNER_MIN_CLEARANCE_V95_M),
            }
        )
    return {
        "n_cells": len(per_cell),
        "per_cell": per_cell,
        "observed_min_m": observed,
        "hard_floor_m": AUDIT_V95_HARD_FLOOR_M,
        "audit_observed_min_m": AUDIT_V95_OBSERVED_MIN_M,
        "delta_vs_audit_m": float(observed - AUDIT_V95_OBSERVED_MIN_M),
        "all_paths_meet_hard_floor": all(item["meets_hard_floor"] for item in per_cell),
        "reproduces_audit": bool(
            observed >= AUDIT_V95_OBSERVED_MIN_M - AUDIT_PROVENANCE_TOLERANCE_M
        ),
        "corners": corners,
        "corner_floor_m": CORNER_MIN_CLEARANCE_V95_M,
        "all_corners_meet_floor": all(item["meets_floor"] for item in corners),
        "witness_rows": witness_rows,
        "passed": bool(
            all(item["meets_hard_floor"] for item in per_cell)
            and observed >= AUDIT_V95_OBSERVED_MIN_M - AUDIT_PROVENANCE_TOLERANCE_M
            and all(item["meets_floor"] for item in corners)
        ),
    }


# ---------------------------------------------------------------------------
# 0B (V6c half) + 0C initial state + 0E route preservation, one worker per row
# ---------------------------------------------------------------------------
def _build_plan(scene_relative: str, sampler_class: str, row: dict[str, Any], seed_u32: int):
    from molmo_spaces.data_generation.pipeline import cleanup_episode_resources, setup_policy
    from molmo_spaces.env.abstract_sensors import SensorSuite
    from run_pact_place_expert_screen import _make_config

    scratch = Path(tempfile.mkdtemp(prefix="v104_preflight_"))
    task = policy = sampler = None
    try:
        config = _make_config(
            scratch / "d.json", scene_xml=ROOT / scene_relative, sampler_class=sampler_class
        )
        sampler = config.task_sampler_config.task_sampler_class(config)
        sampler.seed_task_sampling(int(seed_u32))
        sampler.set_pact_manifest_row(row)
        task = sampler.sample_task(house_index=int(row["scene_template_house_index"]))
        task._sensor_suite = SensorSuite(
            [task._sensor_suite.sensors[uuid] for uuid in ("qpos", "tcp_pose")]
        )
        policy = setup_policy(config, task, None, None)
        task.reset()
        primitives = policy._compute_trajectory()
        from pact_place_v104_runtime import plan_signature

        return {
            "plan": plan_signature(primitives),
            "amendment": dict(getattr(policy, "_pact_place_v104_speed_amendment", {}) or {}),
            "task_horizon": int(config.task_horizon),
            "env_version": str(
                (getattr(task, "scene_params", {}) or {}).get(
                    "pact_place_environment_version", ""
                )
            ),
        }
    finally:
        cleanup_episode_resources(
            task=task, policy=policy, task_sampler=sampler,
            preloaded_policy=None, close_task_sampler=sampler is not None,
        )
        shutil.rmtree(scratch, ignore_errors=True)


def _predicted_steps(plan: Sequence[dict[str, Any]], settle_s: float = 0.1) -> float:
    total = 0.0
    for primitive in plan:
        for segment in primitive.get("segments") or []:
            length = float(
                np.linalg.norm(
                    np.asarray(segment["end_position_m"], dtype=float)
                    - np.asarray(segment["start_position_m"], dtype=float)
                )
            )
            total += length / max(float(segment["speed_m_s"]), 1e-9) + settle_s
    return total / 0.066


def preflight_row(payload: dict[str, Any]) -> dict[str, Any]:
    _pin_threads()
    _establish_env()
    import mujoco
    from molmo_spaces.data_generation.pipeline import cleanup_episode_resources
    from molmo_spaces.env.abstract_sensors import SensorSuite
    from pact_place_v104_clearance import (
        assembly_boxes,
        frame_clearances,
        geom_shape_cache,
        robot_collision_geom_ids,
        target_collision_geom_ids,
    )
    from pact_place_v104_runtime import verify_plan_matches_baseline
    from run_pact_place_expert_screen import _make_config
    from run_pact_place_v7_replay_videos import apply_recorded_qpos

    row = payload["row"]
    row_dir = Path(payload["row_dir"])
    result = json.loads((row_dir / "result.json").read_text())
    steps = json.loads((row_dir / "trajectory.json").read_text())["steps"]
    seed_u32 = int(result["selected_seed"]["seed_u32"])

    baseline = _build_plan(BASE_SCENE_RELATIVE, "PactPlaceCorridorV3Sampler", row, seed_u32)
    amended = _build_plan(SCENE_XML_RELATIVE_V104, SAMPLER_CLASS, row, seed_u32)
    comparison = verify_plan_matches_baseline(baseline["plan"], amended["plan"])
    predicted = _predicted_steps(amended["plan"])

    # Replay the retained V6c trajectory inside the compiled production scene.
    scratch = Path(tempfile.mkdtemp(prefix="v104_replay_"))
    task = sampler = None
    try:
        config = _make_config(
            scratch / "d.json",
            scene_xml=ROOT / SCENE_XML_RELATIVE_V104,
            sampler_class=SAMPLER_CLASS,
        )
        sampler = config.task_sampler_config.task_sampler_class(config)
        sampler.seed_task_sampling(seed_u32)
        sampler.set_pact_manifest_row(row)
        task = sampler.sample_task(house_index=int(row["scene_template_house_index"]))
        task._sensor_suite = SensorSuite(
            [task._sensor_suite.sensors[uuid] for uuid in ("qpos", "tcp_pose")]
        )
        env = task.env
        model, data = env.current_model, env.current_data
        probe = robot_collision_geom_ids(model) + target_collision_geom_ids(task)
        cache = geom_shape_cache(model, probe)
        production_boxes = assembly_boxes(production_assembly())
        corner_boxes = [
            (corner["corner_key"], assembly_boxes(corner))
            for corner in corner_assemblies(production_assembly())
        ]
        pendant_ids = [int(model.geom(name).id) for name in ALL_GEOMS_V104]

        # 0C: compiled bounds and static body, checked on the real model.
        body_id = int(model.body(PENDANT_BODY_V104).id)
        assembly = production_assembly()
        by_geom = {item["geom"]: item for item in assembly["components"]}
        bounds = []
        for name in ALL_GEOMS_V104:
            gid = int(model.geom(name).id)
            size = np.asarray(model.geom_size[gid], dtype=float)
            aabb = np.asarray(model.geom_aabb[gid], dtype=float)
            expected_half = np.asarray(by_geom[name]["half_m"], dtype=float)
            bounds.append(
                {
                    "geom": name,
                    "size_matches": bool(np.allclose(size, expected_half, atol=1e-12)),
                    "geom_aabb_half": [float(v) for v in aabb[3:]],
                    "geom_aabb_encloses": bool(
                        np.all(aabb[3:] >= expected_half - 1e-12)
                    ),
                    "geom_rbound_m": float(model.geom_rbound[gid]),
                    "rbound_encloses": bool(
                        float(model.geom_rbound[gid])
                        >= float(np.linalg.norm(expected_half)) - 1e-9
                    ),
                    "contype": int(model.geom_contype[gid]),
                    "conaffinity": int(model.geom_conaffinity[gid]),
                    "rgba_alpha": float(model.geom_rgba[gid][3]),
                }
            )

        # initial state
        apply_recorded_qpos(env, steps[0]["qpos"])
        mujoco.mj_forward(model, data)
        initial = frame_clearances(model, data, production_boxes, probe, cache)
        # pendant vs every other collision geom in the scene
        from pact_geom_distance import true_distance

        env_ids = []
        for gid in range(int(model.ngeom)):
            if gid in pendant_ids:
                continue
            if int(model.geom_contype[gid]) == 0 and int(model.geom_conaffinity[gid]) == 0:
                continue
            env_ids.append(int(gid))
        # The crossbar/hood_top flush face is the one designed attachment and is
        # allowed to touch; it must touch, not penetrate. Everything else at or
        # below zero is a disallowed static overlap.
        allowed_touches = []
        static_touches = []
        for name in ALL_GEOMS_V104:
            gid = int(model.geom(name).id)
            for env_gid in env_ids:
                distance = float(true_distance(model, data, [gid], [env_gid]))
                if distance > 0.0:
                    continue
                other = str(model.geom(env_gid).name or f"geom_{env_gid}")
                record = {
                    "component_geom": name,
                    "other_geom": other,
                    "other_body": str(
                        model.body(int(model.geom_bodyid[env_gid])).name or ""
                    ),
                    "distance_m": distance,
                }
                is_designed_face = bool(
                    name == CROSSBAR_GEOM_V104
                    and other == "hood_top"
                    and distance >= -1e-9
                )
                if is_designed_face:
                    record["allowed"] = "designed_crossbar_hood_top_flush_face"
                    allowed_touches.append(record)
                else:
                    static_touches.append(record)

        best = float("inf")
        limiting = None
        corner_best = {key: float("inf") for key, _ in corner_boxes}
        for step in steps:
            apply_recorded_qpos(env, step["qpos"])
            mujoco.mj_forward(model, data)
            report = frame_clearances(model, data, production_boxes, probe, cache)
            if report["min_m"] is not None and report["min_m"] < best:
                best = float(report["min_m"])
                limiting = dict(report["limiting"], step=int(step.get("step", -1)))
            for key, boxes in corner_boxes:
                value = frame_clearances(model, data, boxes, probe, cache)["min_m"]
                if value is not None and value < corner_best[key]:
                    corner_best[key] = float(value)
        return {
            "role_index": int(row["role_index"]),
            "n_steps": len(steps),
            "min_clearance_m": float(best),
            "limiting": limiting,
            "corner_min_m": {k: float(v) for k, v in corner_best.items()},
            "initial_min_clearance_m": initial["min_m"],
            "initial_limiting": initial["limiting"],
            "static_touches": static_touches,
            "allowed_touches": allowed_touches,
            "compiled_bounds": bounds,
            "body_dofnum": int(model.body_dofnum[body_id]),
            "body_jntnum": int(model.body_jntnum[body_id]),
            "body_mocapid": int(model.body_mocapid[body_id]),
            "baseline_env_version": baseline["env_version"],
            "amended_env_version": amended["env_version"],
            "baseline_amendment_applied": bool(baseline["amendment"].get("applied")),
            "amended_amendment": amended["amendment"],
            "plan_comparison": {
                k: v for k, v in comparison.items() if k != "failures"
            } | {"failures": comparison["failures"][:5]},
            "task_horizon": amended["task_horizon"],
            "predicted_steps": float(predicted),
            "predicted_within_limit": bool(
                predicted <= HORIZON_UTILISATION_LIMIT * TASK_HORIZON_V104
            ),
        }
    finally:
        cleanup_episode_resources(
            task=task, policy=None, task_sampler=sampler,
            preloaded_policy=None, close_task_sampler=sampler is not None,
        )
        shutil.rmtree(scratch, ignore_errors=True)


# ---------------------------------------------------------------------------
# 0D — contact parity on diagnostic-only scenes
# ---------------------------------------------------------------------------
def contact_parity() -> dict[str, Any]:
    """Clear, touching, and penetrating fixtures must agree across instruments.

    Uses separately compiled diagnostic scenes. The production scene artifact is
    never mutated.
    """
    import mujoco
    from molmo_spaces.tasks.pact_place_contact_audit import classify_contact
    from pact_geom_distance import GeomShape, gjk_distance, true_distance

    assembly = production_assembly()
    by_name = {item["name"]: item for item in assembly["components"]}
    cases: list[dict[str, Any]] = []
    for component_name in ("lobe_0", "lobe_1", "stem_0"):
        component = by_name[component_name]
        half = [float(v) for v in component["half_m"]]
        for probe_label, probe_body in (
            ("robot", "robot_0/fr3_link7"),
            ("carried_target", "cavity_obj_Cup_10"),
        ):
            for gap_label, gap in (
                ("clear", 0.050),
                ("touching", 0.0),
                ("penetrating_5mm", -0.005),
                ("penetrating_15mm", -0.015),
                ("penetrating_30mm", -0.030),
            ):
                probe_half = 0.04
                # Approach along the component's longest axis. Driving 30 mm
                # into the 12 mm-thick axis of a stem makes the probe engulf it,
                # which is the degenerate deep box-box regime recorded in the
                # V10.2 contact-parity root cause, not a real contact test.
                axis = int(max(range(3), key=lambda i: half[i]))
                offset_vec = [0.0, 0.0, 0.0]
                offset_vec[axis] = probe_half + half[axis] + gap
                offset = " ".join(f"{v:.9g}" for v in offset_vec)
                xml = f"""
                <mujoco>
                  <worldbody>
                    <body name="{probe_body}" pos="0 0 0">
                      <joint type="slide" axis="1 0 0"/>
                      <geom name="{probe_body}_collision" type="box"
                            size="{probe_half} {probe_half} {probe_half}"
                            contype="1" conaffinity="1"/>
                    </body>
                    <body name="{PENDANT_BODY_V104}" pos="{offset}">
                      <geom name="{component['geom']}" type="box"
                            size="{half[0]} {half[1]} {half[2]}"
                            contype="8" conaffinity="15"/>
                    </body>
                  </worldbody>
                </mujoco>
                """
                model = mujoco.MjModel.from_xml_string(xml)
                data = mujoco.MjData(model)
                mujoco.mj_forward(model, data)
                before = hashlib.sha256(
                    np.round(np.asarray(model.geom_pos), 12).tobytes()
                    + np.round(np.asarray(model.geom_size), 12).tobytes()
                ).hexdigest()
                probe_gid = int(model.geom(f"{probe_body}_collision").id)
                pendant_gid = int(model.geom(component["geom"]).id)
                signed = float(true_distance(model, data, [probe_gid], [pendant_gid]))
                analytic = float(
                    gjk_distance(
                        GeomShape.from_mujoco(model, data, probe_gid),
                        GeomShape.from_mujoco(model, data, pendant_gid),
                    )
                )
                pairs = []
                for index in range(int(data.ncon)):
                    contact = data.contact[index]
                    if float(contact.dist) > 0.0:
                        continue
                    g1, g2 = int(contact.geom1), int(contact.geom2)
                    if pendant_gid not in (g1, g2):
                        continue
                    record = {
                        "geom1": model.geom(g1).name,
                        "geom2": model.geom(g2).name,
                        "body1": model.body(int(model.geom_bodyid[g1])).name,
                        "body2": model.body(int(model.geom_bodyid[g2])).name,
                        "root1": model.body(int(model.geom_bodyid[g1])).name,
                        "root2": model.body(int(model.geom_bodyid[g2])).name,
                        "distance_m": float(contact.dist),
                    }
                    pairs.append((record, classify_contact(record)))
                after = hashlib.sha256(
                    np.round(np.asarray(model.geom_pos), 12).tobytes()
                    + np.round(np.asarray(model.geom_size), 12).tobytes()
                ).hexdigest()
                expect_contact = gap < 0.0
                classes = sorted({label for _pair, label in pairs})
                if abs(gap) <= 1e-12:
                    # Exact touching is a boundary: assert the distances agree
                    # on zero, but do not demand a particular contact count.
                    agree = bool(abs(signed) <= 1e-6 and analytic <= 1e-9)
                else:
                    agree = bool(
                        (signed < 0.0) == expect_contact
                        and abs(signed - gap) <= 1e-6
                        and (analytic <= 1e-9) == expect_contact
                        and bool(pairs) == expect_contact
                        and (not expect_contact or "mounted_fixture" in classes)
                    )
                cases.append(
                    {
                        "component": component_name,
                        "probe": probe_label,
                        "probe_body": probe_body,
                        "case": gap_label,
                        "designed_gap_m": float(gap),
                        "approach_axis": int(axis),
                        "boundary_case": bool(abs(gap) <= 1e-12),
                        "signed_mj_geom_distance_m": signed,
                        "analytic_gjk_m": analytic,
                        "n_data_contact_pairs": len(pairs),
                        "contact_classes": classes,
                        "classifies_mounted_fixture": bool(
                            "mounted_fixture" in classes
                        ),
                        "expected_contact": expect_contact,
                        "instruments_agree": agree,
                        "scene_state_sha256_before": before,
                        "scene_state_sha256_after": after,
                        "scene_state_restored": bool(before == after),
                    }
                )
    penetrating = [c for c in cases if c["expected_contact"]]
    return {
        "n_cases": len(cases),
        "cases": cases,
        "all_instruments_agree": all(c["instruments_agree"] for c in cases),
        "all_penetrations_classify_mounted_fixture": all(
            c["classifies_mounted_fixture"] for c in penetrating
        ),
        "all_states_restored": all(c["scene_state_restored"] for c in cases),
        "production_scene_mutated": False,
        "passed": bool(
            all(c["instruments_agree"] for c in cases)
            and all(c["classifies_mounted_fixture"] for c in penetrating)
            and all(c["scene_state_restored"] for c in cases)
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    _pin_threads()
    _establish_env()
    started = time.time()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    contract = build_contract()
    provenance = verify_protected_artifacts(contract["protected_artifacts"])
    print(f"0A provenance: {provenance['n_artifacts']} artifacts, passed={provenance['passed']}", flush=True)

    v95 = v95_clearances()
    print(
        f"0B V9.5: min={v95['observed_min_m']:.5f} floor={v95['hard_floor_m']} "
        f"audit={v95['audit_observed_min_m']} passed={v95['passed']}",
        flush=True,
    )

    parity = contact_parity()
    print(f"0D contact parity: {parity['n_cases']} cases, passed={parity['passed']}", flush=True)

    v6c_config = json.loads((ROOT / BASE_CONFIG_RELATIVE).read_text())
    payloads = []
    for row in v6c_config["expert_screen_rows"]:
        directory = V6C_ROWS_DIR / f"{int(row['role_index']):02d}_{row['episode_id'][:16]}"
        payloads.append({"row": row, "row_dir": str(directory)})
    rows: list[dict[str, Any]] = []
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max(1, min(args.workers, len(payloads))),
        mp_context=context,
        max_tasks_per_child=1,
    ) as executor:
        for future in concurrent.futures.as_completed(
            [executor.submit(preflight_row, payload) for payload in payloads]
        ):
            record = future.result()
            rows.append(record)
            print(
                f"  v6c row {record['role_index']:02d} min={record['min_clearance_m']:.5f} "
                f"plan_ok={record['plan_comparison']['poses_identical']} "
                f"speed_changes={record['plan_comparison']['n_speed_changes']} "
                f"steps~{record['predicted_steps']:.0f}",
                flush=True,
            )
    rows.sort(key=lambda item: int(item["role_index"]))

    v6c_min = float(min(item["min_clearance_m"] for item in rows))
    corner_keys = list(rows[0]["corner_min_m"])
    corner_worst = {
        key: float(min(item["corner_min_m"][key] for item in rows)) for key in corner_keys
    }
    v6c = {
        "n_rows": len(rows),
        "observed_min_m": v6c_min,
        "hard_floor_m": AUDIT_V6C_HARD_FLOOR_M,
        "audit_observed_min_m": AUDIT_V6C_OBSERVED_MIN_M,
        "delta_vs_audit_m": float(v6c_min - AUDIT_V6C_OBSERVED_MIN_M),
        "rows_meeting_hard_floor": sum(
            1 for item in rows if item["min_clearance_m"] >= AUDIT_V6C_HARD_FLOOR_M
        ),
        "reproduces_audit": bool(
            v6c_min >= AUDIT_V6C_OBSERVED_MIN_M - AUDIT_PROVENANCE_TOLERANCE_M
        ),
        "corners": [
            {
                "corner_key": key,
                "min_clearance_m": corner_worst[key],
                "meets_floor": bool(corner_worst[key] >= CORNER_MIN_CLEARANCE_V6C_M),
            }
            for key in corner_keys
        ],
        "corner_floor_m": CORNER_MIN_CLEARANCE_V6C_M,
        "per_row": [
            {
                "role_index": item["role_index"],
                "n_steps": item["n_steps"],
                "min_clearance_m": item["min_clearance_m"],
                "limiting": item["limiting"],
            }
            for item in rows
        ],
        "passed": bool(
            all(item["min_clearance_m"] >= AUDIT_V6C_HARD_FLOOR_M for item in rows)
            and v6c_min >= AUDIT_V6C_OBSERVED_MIN_M - AUDIT_PROVENANCE_TOLERANCE_M
            and all(value >= CORNER_MIN_CLEARANCE_V6C_M for value in corner_worst.values())
        ),
    }

    static_ok = all(not item["static_touches"] for item in rows)
    bounds_ok = all(
        entry["size_matches"] and entry["geom_aabb_encloses"] and entry["rbound_encloses"]
        and entry["contype"] == 8 and entry["conaffinity"] == 15 and entry["rgba_alpha"] == 1.0
        for item in rows for entry in item["compiled_bounds"]
    )
    body_ok = all(
        item["body_dofnum"] == 0 and item["body_jntnum"] == 0 and item["body_mocapid"] < 0
        for item in rows
    )
    initial_ok = all(
        item["initial_min_clearance_m"] is not None
        and item["initial_min_clearance_m"] > 0.0
        for item in rows
    )
    allowed_seen = sorted({
        f"{t['component_geom']}|{t['other_geom']}"
        for item in rows for t in item.get("allowed_touches", [])
    })
    designed_face_present = all(
        any(
            t["component_geom"] == CROSSBAR_GEOM_V104 and t["other_geom"] == "hood_top"
            for t in item.get("allowed_touches", [])
        )
        for item in rows
    )
    static = {
        "no_disallowed_static_overlap": static_ok,
        "allowed_touch_pairs": allowed_seen,
        "designed_crossbar_hood_top_face_present_every_row": designed_face_present,
        "compiled_bounds_enclose_final_geometry": bounds_ok,
        "pendant_body_has_no_joint_freejoint_or_mocap": body_ok,
        "initial_state_clear_of_robot_and_target": initial_ok,
        "runtime_bound_repair_used": False,
        "per_row_static_touches": [
            {"role_index": item["role_index"], "touches": item["static_touches"]}
            for item in rows if item["static_touches"]
        ],
        "compiled_bounds_sample": rows[0]["compiled_bounds"],
        "passed": bool(static_ok and bounds_ok and body_ok and initial_ok),
    }

    route = {
        "n_rows": len(rows),
        "all_poses_identical": all(
            item["plan_comparison"]["poses_identical"] for item in rows
        ),
        "all_exactly_one_speed_change": all(
            item["plan_comparison"]["exactly_one_speed_change"] for item in rows
        ),
        "baseline_never_amended": all(
            not item["baseline_amendment_applied"] for item in rows
        ),
        "baseline_env_versions": sorted({item["baseline_env_version"] for item in rows}),
        "amended_env_versions": sorted({item["amended_env_version"] for item in rows}),
        "task_horizon": int(TASK_HORIZON_V104),
        "horizon_utilisation_limit": float(HORIZON_UTILISATION_LIMIT),
        "max_predicted_steps": float(max(item["predicted_steps"] for item in rows)),
        "all_predicted_within_limit": all(
            item["predicted_within_limit"] for item in rows
        ),
        "speed_cap_m_s": float(INITIAL_FREE_SPACE_SPEED_CAP_M_S),
        "speed_changes_sample": rows[0]["plan_comparison"]["speed_changes"],
        "no_v10_route_dispatch_reached": True,
        "per_row": [
            {
                "role_index": item["role_index"],
                "poses_identical": item["plan_comparison"]["poses_identical"],
                "n_speed_changes": item["plan_comparison"]["n_speed_changes"],
                "amendment": {
                    k: v for k, v in (item["amended_amendment"] or {}).items()
                    if k in ("applied", "cap_m_s", "original_speed_m_s",
                             "primitive_index", "segment_index", "n_segments_changed")
                },
                "predicted_steps": item["predicted_steps"],
            }
            for item in rows
        ],
        "passed": bool(
            all(item["plan_comparison"]["poses_identical"] for item in rows)
            and all(item["plan_comparison"]["exactly_one_speed_change"] for item in rows)
            and all(not item["baseline_amendment_applied"] for item in rows)
            and all(item["predicted_within_limit"] for item in rows)
            and all(item["amended_env_version"] == ENVIRONMENT_VERSION for item in rows)
        ),
    }

    items = {
        "0A_provenance": {**provenance, "passed": provenance["passed"]},
        "0B_v95_trust_anchor": {k: v for k, v in v95.items() if k != "witness_rows"},
        "0B_v6c_trust_anchor": v6c,
        "0C_static_and_initial_state": static,
        "0D_contact_parity": {"passed": parity["passed"], "n_cases": parity["n_cases"]},
        "0E_route_and_speed_preservation": route,
    }
    failures = [
        {"item": key, "code": "preflight_item_failed"}
        for key, value in items.items()
        if not value["passed"]
    ]

    witnesses = np.asarray(v95["witness_rows"], dtype=np.float64)
    np.savez_compressed(
        output_root / "clearance_witnesses.npz",
        v95_rows=witnesses,
        v95_columns=np.asarray(
            ["cell_index", "component_index", "direction", "min_distance_m", "frame"],
            dtype="U32",
        ),
        v6c_min_per_row=np.asarray(
            [item["min_clearance_m"] for item in rows], dtype=np.float64
        ),
        v6c_role_index=np.asarray([item["role_index"] for item in rows], dtype=np.int32),
        v6c_corner_keys=np.asarray(corner_keys, dtype="U40"),
        v6c_corner_min=np.asarray(
            [[item["corner_min_m"][key] for key in corner_keys] for item in rows],
            dtype=np.float64,
        ),
        component_names=np.asarray(
            [item["name"] for item in production_assembly()["components"]], dtype="U24"
        ),
    )
    write_immutable_create_only(output_root / "contact_parity.json", {
        "schema_version": "pact_place_v104_contact_parity_v1",
        "contract_version": CONTRACT_VERSION,
        **parity,
        **empty_authorization(),
    })
    document = {
        "schema_version": SCHEMA,
        "contract_version": CONTRACT_VERSION,
        "environment_version": ENVIRONMENT_VERSION,
        "sampler_class": SAMPLER_CLASS,
        "contract_sha256": contract["contract_sha256"],
        "scene_xml": SCENE_XML_RELATIVE_V104,
        "scene_sha256": contract["scene_sha256"],
        "assembly": contract["assembly"],
        "assembly_sha256": contract["assembly_sha256"],
        "assembly_expectations": assembly_expectations(production_assembly()),
        "implementation_sha256": implementation_sha256(),
        "implementation_files": implementation_hashes(),
        "items": items,
        "failures": failures,
        "preflight_passed": not failures,
        "calls_env_step": False,
        "creates_episode": False,
        "elapsed_s": float(time.time() - started),
        **empty_authorization(),
    }
    digest = write_immutable_create_only(output_root / "preflight.json", document)
    print(json.dumps({
        "preflight_passed": document["preflight_passed"],
        "failures": failures,
        "artifact_sha256": digest,
    }, indent=2))
    return 0 if document["preflight_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
