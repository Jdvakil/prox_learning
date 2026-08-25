#!/usr/bin/env python3
"""Run the blocking V9.3 causal-proximity validation on retained expert paths.

For every family/side variant, this script rebuilds the exact retained smoke
episode and replays its full MuJoCo qpos through the normal 40-camera datagen
observation path. At each state in the inbound and outbound decision windows it
renders four controlled worlds: all hazards present, the active panel parked,
the inbound can parked, and the outbound bottle parked. No simulation step or
planner is run between those worlds.

The raw production tensors are retained in compressed NPZ files.  Geometry
proxies are deliberately not used for admission.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MOLMO = ROOT / "submodules" / "molmospaces"
for path in (ROOT / "scripts", MOLMO):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pact_place_corridor_contract import sha256_file  # noqa: E402
from pact_place_v9_contract import (  # noqa: E402
    PALETTE_PATH,
    build_layout,
    load_palette,
    sha256_payload,
)
from run_pact_place_expert_screen import _make_config  # noqa: E402
from run_pact_place_v9_panel_smoke import _row  # noqa: E402

DEFAULT_SMOKE_ROOT = ROOT / "diagnostics_output" / "pact_place_v93_panel_smoke"
DEFAULT_OUTPUT_ROOT = (
    ROOT / "diagnostics_output" / "pact_place_v93_v0c4_causal_proximity"
)
SCENE_XML = (
    MOLMO
    / "molmo_spaces"
    / "data_generation"
    / "custom_scenes"
    / "pact_place_corridor_v5.xml"
)
INBOUND_DECISION_PHASES = frozenset(
    {
        "inbound_vessel_approach",
        "inbound_vessel_pass",
        "inbound_vessel_exit",
        "inbound_cross_vessel_approach",
        "inbound_cross_vessel_pass",
        "inbound_cross_vessel_exit",
    }
)
OUTBOUND_DECISION_PHASES = frozenset(
    {
        "outbound_approach",
        "outbound_vessel_approach",
        "outbound_vessel_pass",
        "outbound_vessel_exit",
        "outbound_exit",
    }
)
DECISION_PHASES = INBOUND_DECISION_PHASES | OUTBOUND_DECISION_PHASES
WINDOW_PADDING_STEPS = 8
ABS_DELTA_FLOOR_M = 1.0e-5
MAX_PAIRED_CHANGED_VALUE_RATIO = 4.0
VESSEL_BODIES = {
    "inbound_vessel": "pact_clutter_06/Soap_Bottle_1",
    "outbound_vessel": "pact_clutter_01/Soap_Bottle_30",
}
PARK_Z_M = -2.0


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _decision_indices(steps: list[dict[str, Any]]) -> list[int]:
    hits = [
        index
        for index, step in enumerate(steps)
        if str(step.get("policy_phase")) in DECISION_PHASES
    ]
    if not hits:
        raise ValueError("trajectory has no compound-hazard decision phase")
    # Expand around each decision state, then take the union. This retains the
    # local lead-in/exit context without rendering the grasp/lift interval that
    # separates inbound from outbound motion.
    expanded: set[int] = set()
    for hit in hits:
        expanded.update(
            range(
                max(0, hit - WINDOW_PADDING_STEPS),
                min(len(steps), hit + WINDOW_PADDING_STEPS + 1),
            )
        )
    return sorted(expanded)


def _phase_mask(phases: list[str], admitted: frozenset[str]) -> np.ndarray:
    mask = np.asarray([phase in admitted for phase in phases], dtype=bool)
    if not np.any(mask):
        raise ValueError(f"no frames for decision phases: {sorted(admitted)}")
    return mask


def _first_activation(
    delta: np.ndarray,
    trajectory_indices: np.ndarray,
    phases: list[str],
    threshold_m: float,
) -> dict[str, Any] | None:
    # delta is (time, sensor, substep, height, width).
    active = np.any(delta > threshold_m, axis=(1, 2, 3, 4))
    hits = np.flatnonzero(active)
    if not hits.size:
        return None
    local = int(hits[0])
    phase = phases[local]
    distance = None
    for future in range(local + 1, len(phases)):
        if phases[future] != phase:
            distance = int(trajectory_indices[future] - trajectory_indices[local])
            break
    return {
        "window_index": local,
        "trajectory_step": int(trajectory_indices[local]),
        "policy_phase": phase,
        "steps_to_next_route_change": distance,
    }


def _link_name(sensor_name: str) -> str:
    return sensor_name.split("_sensor_", 1)[0]


def _causal_metrics(
    present: np.ndarray,
    counterfactual: np.ndarray,
    sensor_names: list[str],
    trajectory_indices: np.ndarray,
    phases: list[str],
    threshold_m: float,
) -> dict[str, Any]:
    delta = np.abs(present.astype(np.float64) - counterfactual.astype(np.float64))
    per_sensor = []
    for sensor_index, name in enumerate(sensor_names):
        values = delta[:, sensor_index]
        per_sensor.append(
            {
                "sensor_index": sensor_index,
                "sensor_name": name,
                "link": _link_name(name),
                "max_abs_delta_m": float(np.max(values)),
                "mean_abs_delta_m": float(np.mean(values)),
                "changed_values": int(np.count_nonzero(values > threshold_m)),
            }
        )
    per_link = []
    for link in sorted({_link_name(name) for name in sensor_names}):
        ids = [i for i, name in enumerate(sensor_names) if _link_name(name) == link]
        values = delta[:, ids]
        per_link.append(
            {
                "link": link,
                "sensor_count": len(ids),
                "max_abs_delta_m": float(np.max(values)),
                "mean_abs_delta_m": float(np.mean(values)),
                "changed_values": int(np.count_nonzero(values > threshold_m)),
            }
        )
    return {
        "max_abs_delta_m": float(np.max(delta)),
        "mean_abs_delta_m": float(np.mean(delta)),
        "changed_values": int(np.count_nonzero(delta > threshold_m)),
        "changed_sensors": int(
            sum(item["changed_values"] > 0 for item in per_sensor)
        ),
        "first_activation": _first_activation(
            delta, trajectory_indices, phases, threshold_m
        ),
        "per_sensor": per_sensor,
        "per_link": per_link,
    }


def _free_joint_qpos_address(model, body_name: str) -> int:
    import mujoco

    body_id = int(model.body(body_name).id)
    joint_id = int(model.body_jntadr[body_id])
    if joint_id < 0 or int(model.jnt_type[joint_id]) != int(
        mujoco.mjtJoint.mjJNT_FREE
    ):
        raise ValueError(f"expected a free body: {body_name}")
    return int(model.jnt_qposadr[joint_id])


def _render_observation(task, sensor_names: list[str]) -> np.ndarray:
    """Use the real buffer sensor path, including its production substep padding."""
    task.env.reset_proximity_depth_buffer(sensor_names)
    task.env.record_proximity_depths(sensor_names)
    tensor = np.stack(
        [
            np.asarray(
                task._sensor_suite.sensors[name].get_observation(
                    env=task.env, task=task, batch_index=0
                ),
                dtype=np.float32,
            )
            for name in sensor_names
        ]
    )
    if tensor.shape != (40, 4, 8, 8):
        raise RuntimeError(f"unexpected raw proximity shape: {tensor.shape}")
    if not np.all(np.isfinite(tensor)) or np.any(tensor <= 0.0):
        raise RuntimeError("raw proximity contains non-finite or non-positive depth")
    return tensor


def _find_episode_dir(smoke_root: Path, episode_id: str) -> Path:
    matches = []
    for directory in (smoke_root / "expert_screen_rows").glob("*"):
        result_path = directory / "result.json"
        if not result_path.is_file():
            continue
        result = json.loads(result_path.read_text())
        if result.get("episode_id") == episode_id:
            matches.append(directory)
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one retained directory for {episode_id}, found {len(matches)}"
        )
    return matches[0]


def _run_variant(job: dict[str, Any]) -> dict[str, Any]:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("MLSPACES_ASSETS_DIR", str(ROOT / "assets"))
    os.environ.pop("DISPLAY", None)

    import mujoco
    from molmo_spaces.data_generation.pipeline import cleanup_episode_resources

    row = job["row"]
    result_path = Path(job["result_path"])
    trajectory_path = Path(job["trajectory_path"])
    output_path = Path(job["output_path"])
    result = json.loads(result_path.read_text())
    trajectory_document = json.loads(trajectory_path.read_text())
    steps = list(trajectory_document["steps"])
    indices = _decision_indices(steps)
    phases = [str(steps[index].get("policy_phase")) for index in indices]

    scratch = Path(tempfile.mkdtemp(prefix="pact_place_v93_v0c4_"))
    task = sampler = None
    try:
        config = _make_config(
            scratch / "dummy.json",
            scene_xml=SCENE_XML,
            sampler_class="PactPlaceCorridorV93Sampler",
        )
        # Four-substep production contract. At a frozen replay qpos the buffer's
        # official sensor implementation repeats the real rendered frame.
        config.proximity_sensor_period_ms = 16.6667
        sampler = config.task_sampler_config.task_sampler_class(config)
        sampler.seed_task_sampling(int(result["selected_seed"]["seed_u32"]))
        sampler.set_pact_manifest_row(row)
        task = sampler.sample_task(house_index=int(row["scene_template_house_index"]))
        if task is None:
            raise RuntimeError("V9.3 sample_task returned None")
        task.reset()

        model = task.env.mj_model
        data = task.env.current_data
        sensor_names = list(task._proximity_camera_names)
        if len(sensor_names) != 40 or len(set(sensor_names)) != 40:
            raise RuntimeError(f"expected 40 unique proximity cameras: {sensor_names}")
        active_panel = f"pact_intrusion_{job['intrusion_side']}"
        panel_mocap_id = int(np.asarray(model.body(active_panel).mocapid).reshape(-1)[0])
        if panel_mocap_id < 0:
            raise RuntimeError(f"active panel is not mocap-controlled: {active_panel}")
        panel_position = np.asarray(data.mocap_pos[panel_mocap_id], dtype=float).copy()
        vessel_bodies = {
            str(obj["role"]): f"pact_clutter_{obj['palette_slot']}/{obj['uid']}"
            for obj in row["pact_clutter_layout"]["objects"]
            if str(obj.get("role")) in {"inbound_vessel", "outbound_vessel"}
        }
        if set(vessel_bodies) != {"inbound_vessel", "outbound_vessel"}:
            raise RuntimeError(f"invalid vessel body contract: {vessel_bodies}")
        vessel_qadrs = {
            role: _free_joint_qpos_address(model, body)
            for role, body in vessel_bodies.items()
        }
        vessel_overrides_xy = {
            str(role): [float(value) for value in xy]
            for role, xy in dict(job.get("vessel_overrides_xy") or {}).items()
        }

        def set_world(qpos: np.ndarray) -> None:
            data.qpos[:] = qpos
            for role, xy in vessel_overrides_xy.items():
                qadr = vessel_qadrs[role]
                data.qpos[qadr : qadr + 2] = xy
            data.mocap_pos[panel_mocap_id] = panel_position

        present_frames = []
        panel_parked_frames = []
        inbound_parked_frames = []
        outbound_parked_frames = []
        for trajectory_index in indices:
            qpos = np.asarray(steps[trajectory_index]["qpos"], dtype=float)
            if qpos.shape != (int(model.nq),):
                raise RuntimeError(
                    f"qpos shape mismatch at {trajectory_index}: {qpos.shape} vs {model.nq}"
                )

            set_world(qpos)
            mujoco.mj_forward(model, data)
            present_frames.append(_render_observation(task, sensor_names))

            set_world(qpos)
            data.mocap_pos[panel_mocap_id, 2] = PARK_Z_M
            mujoco.mj_forward(model, data)
            panel_parked_frames.append(_render_observation(task, sensor_names))

            set_world(qpos)
            data.qpos[vessel_qadrs["inbound_vessel"] + 2] = PARK_Z_M
            mujoco.mj_forward(model, data)
            inbound_parked_frames.append(_render_observation(task, sensor_names))

            set_world(qpos)
            data.qpos[vessel_qadrs["outbound_vessel"] + 2] = PARK_Z_M
            mujoco.mj_forward(model, data)
            outbound_parked_frames.append(_render_observation(task, sensor_names))

        present = np.stack(present_frames).astype(np.float32)
        panel_parked = np.stack(panel_parked_frames).astype(np.float32)
        inbound_parked = np.stack(inbound_parked_frames).astype(np.float32)
        outbound_parked = np.stack(outbound_parked_frames).astype(np.float32)
        trajectory_indices = np.asarray(indices, dtype=np.int32)

        # A repeated baseline at the first, middle, and last state establishes
        # the renderer's empirical numerical floor without doubling the run.
        repeat_deltas = []
        for local_index in sorted({0, len(indices) // 2, len(indices) - 1}):
            trajectory_index = indices[local_index]
            set_world(np.asarray(steps[trajectory_index]["qpos"], dtype=float))
            mujoco.mj_forward(model, data)
            repeated = _render_observation(task, sensor_names)
            repeat_deltas.append(float(np.max(np.abs(repeated - present[local_index]))))
        noise_floor_m = max(repeat_deltas)
        threshold_m = max(ABS_DELTA_FLOOR_M, noise_floor_m * 10.0)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            present=present,
            panel_parked=panel_parked,
            inbound_vessel_parked=inbound_parked,
            outbound_vessel_parked=outbound_parked,
            trajectory_indices=trajectory_indices,
            policy_phases=np.asarray(phases, dtype="U40"),
            sensor_names=np.asarray(sensor_names, dtype="U40"),
        )
        inbound_mask = _phase_mask(phases, INBOUND_DECISION_PHASES)
        outbound_mask = _phase_mask(phases, OUTBOUND_DECISION_PHASES)
        panel_metrics = _causal_metrics(
            present[outbound_mask],
            panel_parked[outbound_mask],
            sensor_names,
            trajectory_indices[outbound_mask],
            [phase for phase, keep in zip(phases, outbound_mask) if keep],
            threshold_m,
        )
        inbound_metrics = _causal_metrics(
            present[inbound_mask],
            inbound_parked[inbound_mask],
            sensor_names,
            trajectory_indices[inbound_mask],
            [phase for phase, keep in zip(phases, inbound_mask) if keep],
            threshold_m,
        )
        outbound_metrics = _causal_metrics(
            present[outbound_mask],
            outbound_parked[outbound_mask],
            sensor_names,
            trajectory_indices[outbound_mask],
            [phase for phase, keep in zip(phases, outbound_mask) if keep],
            threshold_m,
        )
        # A variant whose source episode was not collision-free cannot admit.
        # V9.5's single passing variant was exactly such an episode: the arm was
        # already contacting clutter, so its 40-value inbound reading is a sensor
        # nearly touching an object, not a detection at range.
        source_physics_clean = bool(job["source_physics_clean"])
        signal_passed = bool(
            panel_metrics["changed_values"] > 0
            and inbound_metrics["changed_values"] > 0
            and outbound_metrics["changed_values"] > 0
            and panel_metrics["first_activation"] is not None
            and inbound_metrics["first_activation"] is not None
            and outbound_metrics["first_activation"] is not None
        )
        passed = bool(signal_passed and source_physics_clean)
        return {
            "family_id": job["family_id"],
            "intrusion_side": job["intrusion_side"],
            "episode_id": row["episode_id"],
            "row_sha256": row["row_sha256"],
            "source_result_path": str(result_path.relative_to(ROOT)),
            "source_result_sha256": sha256_file(result_path),
            "source_trajectory_path": str(trajectory_path.relative_to(ROOT)),
            "source_trajectory_sha256": sha256_file(trajectory_path),
            "raw_tensor_path": str(output_path.relative_to(ROOT)),
            "raw_tensor_sha256": sha256_file(output_path),
            "tensor_shape_per_world": list(present.shape),
            "sensor_count": len(sensor_names),
            "sensor_names": sensor_names,
            "substeps": int(present.shape[2]),
            "decision_window": {
                "trajectory_start_step": indices[0],
                "trajectory_stop_step_inclusive": indices[-1],
                "n_policy_steps": len(indices),
                "phases": list(dict.fromkeys(phases)),
                "padding_steps": WINDOW_PADDING_STEPS,
            },
            "baseline_repeat_max_abs_delta_m": noise_floor_m,
            "causal_threshold_m": threshold_m,
            "active_panel": active_panel,
            "vessel_bodies": dict(vessel_bodies),
            "counterfactual_vessel_overrides_xy_m": vessel_overrides_xy,
            "panel_causal_effect": panel_metrics,
            "inbound_vessel_causal_effect": inbound_metrics,
            "outbound_vessel_causal_effect": outbound_metrics,
            "source_physics_clean": source_physics_clean,
            "source_clean_success": bool(job["source_physics_clean"]),
            "signal_passed": signal_passed,
            "passed": passed,
        }
    finally:
        cleanup_episode_resources(
            task=task,
            policy=None,
            task_sampler=sampler,
            preloaded_policy=None,
            close_task_sampler=sampler is not None,
        )
        shutil.rmtree(scratch, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-root", type=Path, default=DEFAULT_SMOKE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--vessel-overrides",
        type=Path,
        help="counterfactual family->role->xy siting candidates; never authorizes admission",
    )
    parser.add_argument("--family", action="append")
    parser.add_argument("--side", choices=("left", "right"), action="append")
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise SystemExit("workers must be in [1, 4]")

    smoke_root = args.smoke_root.resolve()
    output_root = args.output_root.resolve()
    smoke_summary_path = smoke_root / "summary.json"
    smoke_summary = json.loads(smoke_summary_path.read_text())
    palette_document = load_palette(PALETTE_PATH)
    vessel_overrides = (
        json.loads(args.vessel_overrides.resolve().read_text())
        if args.vessel_overrides
        else {}
    )
    output_root.mkdir(parents=True, exist_ok=True)

    jobs = []
    clean_by_episode = {
        str(item["episode_id"]): bool(item.get("clean_success"))
        for item in list(smoke_summary.get("results") or [])
    }
    retained_rows = {
        (str(row["layout_family_id"]), str(row["intrusion_side"])): row
        for row in list(smoke_summary.get("manifest_rows") or [])
    }
    for item in smoke_summary["results"]:
        if args.family and str(item["family_id"]) not in set(args.family):
            continue
        if args.side and str(item["intrusion_side"]) not in set(args.side):
            continue
        episode_dir = _find_episode_dir(smoke_root, str(item["episode_id"]))
        row = retained_rows.get(
            (str(item["family_id"]), str(item["intrusion_side"]))
        ) or _row(
            index=int(item["role_index"]),
            family_id=str(item["family_id"]),
            side=str(item["intrusion_side"]),
            palette_document=palette_document,
            implementation_sha256=str(smoke_summary["implementation_sha256"]),
            seed=int(smoke_summary["seed"]),
        )
        if row["episode_id"] != item["episode_id"]:
            raise RuntimeError("retained smoke row reconstruction mismatch")
        jobs.append(
            {
                "family_id": item["family_id"],
                "intrusion_side": item["intrusion_side"],
                "row": row,
                "result_path": str(episode_dir / "result.json"),
                "trajectory_path": str(episode_dir / "trajectory.json"),
                "output_path": str(
                    output_root
                    / "raw"
                    / f"{item['family_id']}_{item['intrusion_side']}.npz"
                ),
                "vessel_overrides_xy": dict(vessel_overrides.get(str(item["family_id"])) or {}),
                # Joined on episode_id against the smoke summary's clean_success.
                "source_physics_clean": bool(clean_by_episode.get(str(item["episode_id"]))),
            }
        )

    results = []
    if args.workers == 1:
        for job in jobs:
            print(f"Validating {job['family_id']} / {job['intrusion_side']}", flush=True)
            result = _run_variant(job)
            print(
                json.dumps(
                    {
                        "family_id": result["family_id"],
                        "side": result["intrusion_side"],
                        "panel_changed": result["panel_causal_effect"]["changed_values"],
                        "inbound_changed": result["inbound_vessel_causal_effect"]["changed_values"],
                        "outbound_changed": result["outbound_vessel_causal_effect"]["changed_values"],
                        "passed": result["passed"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            results.append(result)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_to_job = {executor.submit(_run_variant, job): job for job in jobs}
            for future in concurrent.futures.as_completed(future_to_job):
                result = future.result()
                print(
                    json.dumps(
                        {
                            "family_id": result["family_id"],
                            "side": result["intrusion_side"],
                            "panel_changed": result["panel_causal_effect"]["changed_values"],
                            "inbound_changed": result["inbound_vessel_causal_effect"]["changed_values"],
                            "outbound_changed": result["outbound_vessel_causal_effect"]["changed_values"],
                            "passed": result["passed"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                results.append(result)

    results.sort(key=lambda item: (item["family_id"], item["intrusion_side"]))
    side_role_counts = {
        side: {
            role: sum(
                int(item[f"{role}_causal_effect"]["changed_values"] > 0)
                for item in results
                if item["intrusion_side"] == side
            )
            for role in ("inbound_vessel", "outbound_vessel")
        }
        for side in ("left", "right")
    }
    paired_family_balance = []
    for family_id in sorted({str(item["family_id"]) for item in results}):
        pair = {
            str(item["intrusion_side"]): item
            for item in results
            if str(item["family_id"]) == family_id
        }
        for role in ("inbound_vessel", "outbound_vessel"):
            values = {
                side: int(item[f"{role}_causal_effect"]["changed_values"])
                for side, item in pair.items()
            }
            low = min(values.values()) if len(values) == 2 else 0
            high = max(values.values()) if values else 0
            ratio = float(high / low) if low > 0 else None
            paired_family_balance.append(
                {
                    "family_id": family_id,
                    "role": role,
                    "changed_values_by_side": values,
                    "max_to_min_ratio": ratio,
                    "passed": bool(
                        set(values) == {"left", "right"}
                        and low > 0
                        and ratio is not None
                        and ratio <= MAX_PAIRED_CHANGED_VALUE_RATIO
                    ),
                }
            )
    full_matrix = len(results) == 8
    paired_side_raw_ok = all(
        counts[role] == 4
        for counts in side_role_counts.values()
        for role in ("inbound_vessel", "outbound_vessel")
    ) and len(paired_family_balance) == 8 and all(
        item["passed"] for item in paired_family_balance
    )
    requires_clean_source = str(smoke_summary.get("schema_version", "")).startswith(
        "pact_place_v9_5"
    )
    source_physics_clean = bool(
        len(smoke_summary.get("results") or []) == 8
        and all(item.get("clean_success") is True for item in smoke_summary["results"])
    )
    passed = bool(
        full_matrix
        and all(item["passed"] for item in results)
        and paired_side_raw_ok
        and (source_physics_clean or not requires_clean_source)
    )
    counterfactual_siting = bool(args.vessel_overrides)
    validation = {
        "schema_version": (
            "pact_place_v9_5_v0c5_counterfactual_siting_v1"
            if counterfactual_siting
            else "pact_place_v9_3_v0c4_causal_proximity_v1"
        ),
        "role": (
            "counterfactual_raw_siting_not_admission"
            if counterfactual_siting
            else "blocking_causal_proximity_validation"
        ),
        "passed": passed,
        "authorizes_v1b": bool(passed and not counterfactual_siting),
        "authorizes_gate": False,
        "authorizes_collection": False,
        "uses_real_40_sensor_observation": True,
        "uses_geometry_proxy_for_admission": False,
        "counterfactual_vessel_overrides_xy_m": vessel_overrides,
        "requires_physics_regeneration_for_admission": counterfactual_siting,
        "counterfactual_worlds": [
            "panel_and_both_vessels_present",
            "active_panel_parked_both_vessels_unchanged",
            "inbound_vessel_parked_panel_and_outbound_vessel_unchanged",
            "outbound_vessel_parked_panel_and_inbound_vessel_unchanged",
        ],
        "frozen_qpos_no_simulation_between_worlds": True,
        "production_tensor_contract": [40, 4, 8, 8],
        "smoke_summary_path": str(smoke_summary_path.relative_to(ROOT)),
        "smoke_summary_sha256": sha256_file(smoke_summary_path),
        "palette_path": str(PALETTE_PATH.relative_to(ROOT)),
        "palette_sha256": sha256_file(PALETTE_PATH),
        "validator_path": str(Path(__file__).resolve().relative_to(ROOT)),
        "validator_sha256": sha256_file(Path(__file__).resolve()),
        "variant_count": len(results),
        "full_eight_variant_matrix": full_matrix,
        "requires_clean_source_physics": requires_clean_source,
        "source_physics_clean": source_physics_clean,
        "passing_variant_count": sum(item["passed"] for item in results),
        "physics_clean_variant_count": sum(
            bool(item.get("source_physics_clean")) for item in results
        ),
        "passing_physics_clean_variant_count": sum(
            bool(item["passed"]) and bool(item.get("source_physics_clean"))
            for item in results
        ),
        "signal_passing_variant_count": sum(
            bool(item.get("signal_passed")) for item in results
        ),
        "dirty_source_variants": [
            {
                "family_id": item["family_id"],
                "intrusion_side": item["intrusion_side"],
                "episode_id": item["episode_id"],
            }
            for item in results
            if not item.get("source_physics_clean")
        ],
        "paired_side_raw_vessel_admission": {
            "passed": paired_side_raw_ok,
            "required_detected_variants_per_side_and_role": 4,
            "maximum_paired_changed_value_ratio": MAX_PAIRED_CHANGED_VALUE_RATIO,
            "counts": side_role_counts,
            "family_role_balance": paired_family_balance,
        },
        "variants": results,
    }
    validation["validation_sha256"] = sha256_payload(validation)
    validation_path = output_root / "validation.json"
    validation_path.write_text(json.dumps(_jsonable(validation), indent=2, sort_keys=True) + "\n")
    print(validation_path, flush=True)
    print(json.dumps({"passed": passed, "variants": len(results)}, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
