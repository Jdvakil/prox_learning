#!/usr/bin/env python3
"""Paired raw-skin visibility audit for V10.11 versus taller V10.11b."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "scripts", ROOT / "submodules" / "molmospaces"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

PARENT_CONTRACT_MODULE = os.environ.get(
    "PACT_PLACE_VISIBILITY_PARENT_CONTRACT", "pact_place_v1011_contract"
)
TALL_CONTRACT_MODULE = os.environ.get(
    "PACT_PLACE_VISIBILITY_TALL_CONTRACT", "pact_place_v1011b_contract"
)
parent_contract = importlib.import_module(PARENT_CONTRACT_MODULE)
tall_contract = importlib.import_module(TALL_CONTRACT_MODULE)
from pact_place_v105_contract import sha256_file  # noqa: E402
from run_pact_place_expert_screen import _make_config  # noqa: E402
from run_pact_place_v9_v0c3_causal_proximity import _render_observation  # noqa: E402

PARENT_REVIEW = ROOT / parent_contract.REVIEW_ROOT
OUTPUT = ROOT / tall_contract.VISIBILITY_ROOT / "visibility.json"
RAW_OUTPUT = ROOT / tall_contract.VISIBILITY_ROOT / "paired_depths.npz"
ABS_DELTA_FLOOR_M = 1.0e-5
MAX_FRAMES_PER_ROW = 48


def _episode_dir(role_index: int) -> Path:
    matches = sorted(
        path
        for path in (PARENT_REVIEW / "expert_screen_rows").glob(f"{role_index:02d}_*")
        if (path / "result.json").is_file() and (path / "trajectory.json").is_file()
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one parent retained row for role {role_index}, got {matches}"
        )
    return matches[0]


def _qpos_address(model, body_name: str) -> int:
    import mujoco

    body_id = int(model.body(body_name).id)
    joint_id = int(model.body_jntadr[body_id])
    if joint_id < 0 or int(model.jnt_type[joint_id]) != int(
        mujoco.mjtJoint.mjJNT_FREE
    ):
        raise RuntimeError(f"expected primitive free body: {body_name}")
    return int(model.jnt_qposadr[joint_id])


def _sample_indices(steps: list[dict[str, Any]]) -> list[int]:
    if not steps:
        raise RuntimeError("retained trajectory is empty")
    indices = set(
        int(value)
        for value in np.linspace(
            0, len(steps) - 1, min(MAX_FRAMES_PER_ROW, len(steps)), dtype=int
        )
    )
    by_phase: dict[str, list[int]] = {}
    for index, step in enumerate(steps):
        by_phase.setdefault(str(step.get("policy_phase")), []).append(index)
    # Preserve phase transitions even when an evenly spaced sample skips a
    # short segment. This is still bounded by three frames per phase.
    for phase_indices in by_phase.values():
        indices.update(
            (phase_indices[0], phase_indices[len(phase_indices) // 2], phase_indices[-1])
        )
    return sorted(indices)


def _make_task(row: dict[str, Any], seed_u32: int, scratch: Path):
    config = _make_config(
        scratch / "dummy.json",
        scene_xml=ROOT / row["pact_v1011_scene_relative"],
        sampler_class=row["sampler_class"],
    )
    config.proximity_sensor_period_ms = 16.6667
    sampler = config.task_sampler_config.task_sampler_class(config)
    sampler.seed_task_sampling(int(seed_u32))
    sampler.set_pact_manifest_row(row)
    task = sampler.sample_task(house_index=int(row["scene_template_house_index"]))
    if task is None:
        raise RuntimeError(f"{row['environment_version']} reconstruction returned None")
    task.reset()
    return sampler, task


def _selected_rows() -> list[dict[str, Any]]:
    manifest_path = PARENT_REVIEW / "review_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if not manifest.get("eligible_for_owner_review") or len(manifest.get("selected") or []) != 6:
        raise RuntimeError("parent packet is not the frozen six-row review")
    return list(manifest["selected"])


def run_row(selected: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    import mujoco
    from molmo_spaces.data_generation.pipeline import cleanup_episode_resources

    role = int(selected["role_index"])
    directory = _episode_dir(role)
    result_path = directory / "result.json"
    trajectory_path = directory / "trajectory.json"
    result = json.loads(result_path.read_text())
    trajectory = json.loads(trajectory_path.read_text())
    steps = list(trajectory.get("steps") or [])
    indices = _sample_indices(steps)
    seed_u32 = int(result["selected_seed"]["seed_u32"])
    args = (
        str(selected["family_id"]),
        str(selected["intrusion_side"]),
        str(selected["pose_id"]),
        int(selected["attempt_index"]),
    )
    parent_row = parent_contract.build_row(*args, role_index=role)
    tall_row = tall_contract.build_row(*args, role_index=role)
    scratch = Path(tempfile.mkdtemp(prefix=f"v1011b_visibility_{role}_"))
    parent_sampler = parent_task = tall_sampler = tall_task = None
    try:
        parent_sampler, parent_task = _make_task(parent_row, seed_u32, scratch / "parent")
        tall_sampler, tall_task = _make_task(tall_row, seed_u32, scratch / "tall")
        parent_model, parent_data = (
            parent_task.env.current_model,
            parent_task.env.current_data,
        )
        tall_model, tall_data = tall_task.env.current_model, tall_task.env.current_data
        if int(parent_model.nq) != int(tall_model.nq):
            raise RuntimeError("parent/tall model qpos dimensions differ")
        parent_names = list(parent_task._proximity_camera_names)
        tall_names = list(tall_task._proximity_camera_names)
        if parent_names != tall_names or len(parent_names) != 40:
            raise RuntimeError("parent/tall production proximity suites differ")

        parent_layout = {
            str(item["palette_slot"]): item
            for item in parent_task.scene_params["pact_clutter_layout"]["objects"]
        }
        tall_layout = {
            str(item["palette_slot"]): item
            for item in tall_task.scene_params["pact_clutter_layout"]["objects"]
        }
        xy_equal = all(
            np.allclose(
                parent_layout[slot]["center_m"][:2],
                tall_layout[slot]["center_m"][:2],
                atol=1e-12,
                rtol=0.0,
            )
            for slot in tall_contract.ACTIVE_CLUTTER_SLOTS
        )
        if not xy_equal:
            raise RuntimeError("parent/tall sampled XY layouts differ")

        qaddrs = {}
        for slot in tall_contract.PRIMITIVE_SLOTS:
            body = (
                f"pact_clutter_{slot}/"
                f"{tall_layout[slot]['uid']}"
            )
            qaddrs[slot] = _qpos_address(tall_model, body)

        parent_frames: list[np.ndarray] = []
        tall_frames: list[np.ndarray] = []
        for trajectory_index in indices:
            qpos = np.asarray(steps[trajectory_index]["qpos"], dtype=np.float64)
            if qpos.shape != parent_data.qpos.shape or qpos.shape != tall_data.qpos.shape:
                raise RuntimeError(f"retained qpos shape mismatch at {trajectory_index}")
            parent_data.qpos[:] = qpos
            mujoco.mj_forward(parent_model, parent_data)
            parent_frames.append(_render_observation(parent_task, parent_names))

            tall_data.qpos[:] = qpos
            for slot, qadr in qaddrs.items():
                height_delta = (
                    tall_contract.PRIMITIVE_HEIGHTS_M[slot]
                    - tall_contract.PARENT_PRIMITIVE_HEIGHTS_M[slot]
                )
                # Retain the same robot, target and planar object state while
                # keeping the taller primitive's bottom at the parent's base.
                tall_data.qpos[qadr + 2] += height_delta / 2.0
            mujoco.mj_forward(tall_model, tall_data)
            tall_frames.append(_render_observation(tall_task, tall_names))

        parent_array = np.stack(parent_frames).astype(np.float32)
        tall_array = np.stack(tall_frames).astype(np.float32)
        # Re-render one unchanged world to establish the empirical renderer
        # floor; admission is based only on deltas above ten times this noise.
        repeat_index = indices[len(indices) // 2]
        parent_data.qpos[:] = np.asarray(steps[repeat_index]["qpos"], dtype=np.float64)
        mujoco.mj_forward(parent_model, parent_data)
        repeated = _render_observation(parent_task, parent_names)
        noise_floor = float(
            np.max(np.abs(repeated - parent_array[len(indices) // 2]))
        )
        threshold = max(ABS_DELTA_FLOOR_M, noise_floor * 10.0)
        signed = parent_array.astype(np.float64) - tall_array.astype(np.float64)
        absolute = np.abs(signed)
        sensor_changed = np.any(absolute > threshold, axis=(0, 2, 3, 4))
        per_sensor = []
        for sensor_index, name in enumerate(parent_names):
            values = signed[:, sensor_index]
            per_sensor.append(
                {
                    "sensor_index": sensor_index,
                    "sensor_name": name,
                    "link": name.split("_sensor_", 1)[0],
                    "changed_values": int(np.count_nonzero(np.abs(values) > threshold)),
                    "nearer_values": int(np.count_nonzero(values > threshold)),
                    "farther_values": int(np.count_nonzero(values < -threshold)),
                    "max_abs_delta_m": float(np.max(np.abs(values))),
                }
            )
        record = {
            "role_index": role,
            "family_id": selected["family_id"],
            "intrusion_side": selected["intrusion_side"],
            "pose_id": selected["pose_id"],
            "parent_clean_success": bool(selected["clean_success"]),
            "selected_seed_u32": seed_u32,
            "trajectory_sha256": sha256_file(trajectory_path),
            "frames_sampled": len(indices),
            "trajectory_indices": indices,
            "xy_layout_equal": xy_equal,
            "noise_floor_m": noise_floor,
            "threshold_m": threshold,
            "changed_values": int(np.count_nonzero(absolute > threshold)),
            "nearer_values": int(np.count_nonzero(signed > threshold)),
            "farther_values": int(np.count_nonzero(signed < -threshold)),
            "changed_sensors": int(np.count_nonzero(sensor_changed)),
            "max_abs_delta_m": float(np.max(absolute)),
            "per_sensor": per_sensor,
        }
        arrays = {
            f"role_{role:03d}_parent": parent_array,
            f"role_{role:03d}_tall": tall_array,
            f"role_{role:03d}_indices": np.asarray(indices, dtype=np.int32),
        }
        return record, arrays
    finally:
        for sampler, task in (
            (parent_sampler, parent_task),
            (tall_sampler, tall_task),
        ):
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
        shutil.rmtree(scratch, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--raw-output", type=Path, default=RAW_OUTPUT)
    args = parser.parse_args()
    if args.output.exists() or args.raw_output.exists():
        raise SystemExit("refusing to overwrite paired visibility artifacts")
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("MLSPACES_ASSETS_DIR", str(ROOT / "assets"))
    os.environ.pop("DISPLAY", None)

    records = []
    arrays: dict[str, np.ndarray] = {}
    for selected in _selected_rows():
        record, row_arrays = run_row(selected)
        records.append(record)
        arrays.update(row_arrays)
        print(
            f"role={record['role_index']} changed={record['changed_values']} "
            f"sensors={record['changed_sensors']} max={record['max_abs_delta_m']:.6f}",
            flush=True,
        )
    args.raw_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.raw_output, **arrays)
    side_changed = {
        side: sum(
            int(record["nearer_values"])
            for record in records
            if record["intrusion_side"] == side
        )
        for side in ("left", "right")
    }
    passed = (
        len(records) == 6
        and all(record["xy_layout_equal"] for record in records)
        and all(value > 0 for value in side_changed.values())
    )
    document = {
        **tall_contract.empty_authorization(),
        "schema_version": f"{tall_contract.CONTRACT_VERSION}_paired_raw_visibility",
        "contract_version": tall_contract.CONTRACT_VERSION,
        "parent_environment_version": parent_contract.ENVIRONMENT_VERSION,
        "environment_version": tall_contract.ENVIRONMENT_VERSION,
        "method": {
            "production_raw_proximity_path": True,
            "identical_retained_robot_qpos": True,
            "identical_sampled_xy_layout": True,
            "primitive_bases_aligned_by_half_height_translation": True,
            "visibility_hardcoded": False,
            "absolute_delta_floor_m": ABS_DELTA_FLOOR_M,
        },
        "rows": records,
        "nearer_values_by_side": side_changed,
        "raw_npz": str(args.raw_output.relative_to(ROOT)),
        "raw_npz_sha256": sha256_file(args.raw_output),
        "passed": passed,
    }
    document["payload_sha256"] = tall_contract.canonical_payload_sha256(document)
    tall_contract.write_immutable_create_only(args.output, document)
    print(json.dumps({"passed": passed, "nearer_values_by_side": side_changed,
                      "payload_sha256": document["payload_sha256"]}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
