#!/usr/bin/env python3
"""V10.1 frozen-qpos causal proximity on the 12-row empirical review pack.

Replays retained qpos without env.step. Does not infer Phase-0 approval.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "scripts", ROOT / "submodules" / "molmospaces"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pact_place_corridor_contract import sha256_file  # noqa: E402
from pact_place_v10_runtime import establish_v10_runtime_env, write_immutable  # noqa: E402
from pact_place_v10_scene import pose_assembly_on_data  # noqa: E402
from pact_place_v101_empirical_qualification_contract import (  # noqa: E402
    CONTRACT_VERSION,
    PHYSICS_CLEAN_FAMILIES,
    SAMPLER_CLASS,
    SCENE_XML_RELATIVE,
    cell_key,
    empty_authorization,
    frozen_assembly,
    is_v101_clean_success,
    lowest_clean_row_per_cell,
    sha256_payload,
)
from run_pact_place_expert_screen import _make_config  # noqa: E402
from run_pact_place_v96_cluster_causal_proximity import ADMISSION_FLOOR, CORRIDOR_LINKS  # noqa: E402
from run_pact_place_v9_v0c3_causal_proximity import (  # noqa: E402
    ABS_DELTA_FLOOR_M,
    INBOUND_DECISION_PHASES,
    MAX_PAIRED_CHANGED_VALUE_RATIO,
    OUTBOUND_DECISION_PHASES,
    _causal_metrics,
    _render_observation,
)

DEFAULT_REVIEW_ROOT = ROOT / "diagnostics_output" / "pact_place_v101_empirical_review"
DEFAULT_OUTPUT_ROOT = ROOT / "diagnostics_output" / "pact_place_v101_empirical_causal"
SCENE_XML = ROOT / SCENE_XML_RELATIVE
WINDOWS = {
    "inbound": INBOUND_DECISION_PHASES,
    "outbound": OUTBOUND_DECISION_PHASES,
}


def _link_set(metrics: dict[str, Any]) -> set[str]:
    return {
        str(item["link"])
        for item in metrics.get("per_sensor") or []
        if int(item.get("changed_values") or 0) > 0
    }


def _verdict(metrics: dict[str, Any]) -> dict[str, Any]:
    links = _link_set(metrics)
    corridor = set(ADMISSION_FLOOR["required_responding_links_any_of"])
    return {
        "changed_values": int(metrics["changed_values"]),
        "changed_sensors": int(metrics["changed_sensors"]),
        "responding_links": sorted(links),
        "meets_min_sensors": int(metrics["changed_sensors"])
        >= int(ADMISSION_FLOOR["min_distinct_changed_sensors_per_role_side"]),
        "meets_min_changed_values": int(metrics["changed_values"])
        >= int(ADMISSION_FLOOR["min_changed_values_per_role_side"]),
        "meets_corridor_link": bool(links & corridor),
        "passed": bool(
            int(metrics["changed_sensors"])
            >= int(ADMISSION_FLOOR["min_distinct_changed_sensors_per_role_side"])
            and int(metrics["changed_values"])
            >= int(ADMISSION_FLOOR["min_changed_values_per_role_side"])
            and bool(links & corridor)
        ),
    }


def _side_balance(cells: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    reports = []
    failures = []
    for family in PHYSICS_CLEAN_FAMILIES:
        for direction in ("inbound", "outbound"):
            left = cells.get((family, "left"), {}).get("windows", {}).get(direction)
            right = cells.get((family, "right"), {}).get("windows", {}).get(direction)
            if not left or not right:
                failures.append(
                    {
                        "code": "missing_side_window",
                        "family": family,
                        "direction": direction,
                    }
                )
                continue
            left_n = max(int(left["changed_values"]), 1)
            right_n = max(int(right["changed_values"]), 1)
            ratio = max(left_n, right_n) / min(left_n, right_n)
            item = {
                "family": family,
                "direction": direction,
                "left_changed_values": int(left["changed_values"]),
                "right_changed_values": int(right["changed_values"]),
                "ratio": float(ratio),
                "max_ratio": float(MAX_PAIRED_CHANGED_VALUE_RATIO),
                "passed": bool(ratio <= MAX_PAIRED_CHANGED_VALUE_RATIO + 1e-12),
            }
            reports.append(item)
            if not item["passed"]:
                failures.append({"code": "side_imbalance", **item})
    return reports, failures


def _run_cell(job: dict[str, Any], output_root: Path) -> dict[str, Any]:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("MLSPACES_ASSETS_DIR", str(ROOT / "assets"))
    os.environ.pop("DISPLAY", None)
    import mujoco
    from molmo_spaces.data_generation.pipeline import cleanup_episode_resources

    row = job["row"]
    result = json.loads(Path(job["result_path"]).read_text())
    steps = list(json.loads(Path(job["trajectory_path"]).read_text())["steps"])
    phases = [str(step.get("policy_phase") or "") for step in steps]
    scratch = Path(tempfile.mkdtemp(prefix="pact_v101_causal_"))
    task = sampler = None
    try:
        config = _make_config(
            scratch / "dummy.json",
            scene_xml=SCENE_XML,
            sampler_class=SAMPLER_CLASS,
        )
        config.proximity_sensor_period_ms = 16.6667
        sampler = config.task_sampler_config.task_sampler_class(config)
        sampler.seed_task_sampling(int(result["selected_seed"]["seed_u32"]))
        sampler.set_pact_manifest_row(row)
        task = sampler.sample_task(house_index=int(row["scene_template_house_index"]))
        if task is None:
            raise RuntimeError("V10.1 sample_task returned None")
        task.reset()
        model, data = task.env.mj_model, task.env.current_data
        sensor_names = list(task._proximity_camera_names)
        if len(sensor_names) != 40 or len(set(sensor_names)) != 40:
            raise RuntimeError("V10.1 requires the frozen 40-sensor suite")
        assembly = frozen_assembly()

        def set_qpos(step_index: int) -> None:
            qpos = np.asarray(steps[step_index]["qpos"], dtype=float)
            if qpos.shape != (int(model.nq),):
                raise RuntimeError(f"qpos shape mismatch: {qpos.shape} vs {model.nq}")
            data.qpos[:] = qpos

        present = []
        present_repeat = []
        parked = []
        for index in range(len(steps)):
            set_qpos(index)
            pose_assembly_on_data(model, data, assembly, parked=False)
            mujoco.mj_forward(model, data)
            present.append(_render_observation(task, sensor_names))
            pose_assembly_on_data(model, data, assembly, parked=False)
            mujoco.mj_forward(model, data)
            present_repeat.append(_render_observation(task, sensor_names))
            pose_assembly_on_data(model, data, assembly, parked=True)
            mujoco.mj_forward(model, data)
            parked.append(_render_observation(task, sensor_names))
        stacked = {
            "present": np.stack(present).astype(np.float32),
            "present_repeat": np.stack(present_repeat).astype(np.float32),
            "assembly_parked": np.stack(parked).astype(np.float32),
        }
        if stacked["present"].shape[1:] != (40, 4, 8, 8):
            raise RuntimeError(f"unexpected tensor shape {stacked['present'].shape}")
        noise = float(
            np.max(np.abs(stacked["present"] - stacked["present_repeat"]))
        )
        threshold = max(float(ABS_DELTA_FLOOR_M), 10.0 * noise)
        raw_path = (
            output_root
            / "raw"
            / f"{job['family']}_{job['intrusion_side']}.npz"
        )
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            raw_path,
            present=stacked["present"],
            present_repeat=stacked["present_repeat"],
            assembly_parked=stacked["assembly_parked"],
            trajectory_indices=np.arange(len(steps), dtype=np.int32),
            policy_phases=np.asarray(phases, dtype="U40"),
            sensor_names=np.asarray(sensor_names, dtype="U40"),
        )
        windows = {}
        for name, phase_set in WINDOWS.items():
            mask = np.asarray([phase in phase_set for phase in phases], dtype=bool)
            if not np.any(mask):
                raise RuntimeError(
                    f"missing {name} decision window for {job['family']} {job['intrusion_side']}"
                )
            metrics = _causal_metrics(
                stacked["present"][mask],
                stacked["assembly_parked"][mask],
                sensor_names,
                np.arange(len(steps), dtype=np.int32)[mask],
                [phase for phase, keep in zip(phases, mask) if keep],
                threshold,
            )
            windows[name] = {
                **_verdict(metrics),
                "n_window_steps": int(np.count_nonzero(mask)),
                "metrics": {
                    "max_abs_delta_m": metrics["max_abs_delta_m"],
                    "mean_abs_delta_m": metrics["mean_abs_delta_m"],
                    "changed_values": metrics["changed_values"],
                    "changed_sensors": metrics["changed_sensors"],
                    "first_activation": metrics["first_activation"],
                    "per_link": metrics["per_link"],
                },
            }
        return {
            "family": job["family"],
            "intrusion_side": job["intrusion_side"],
            "role_index": int(row["role_index"]),
            "episode_id": str(result["episode_id"]),
            "row_sha256": row["row_sha256"],
            "raw_tensor_path": str(raw_path.relative_to(ROOT)),
            "raw_tensor_sha256": sha256_file(raw_path),
            "tensor_shape": list(stacked["present"].shape),
            "sensor_count": 40,
            "baseline_repeat_max_abs_delta_m": noise,
            "causal_threshold_m": threshold,
            "abs_delta_floor_m": ABS_DELTA_FLOOR_M,
            "windows": windows,
            "cell_passed": bool(
                windows["inbound"]["passed"] and windows["outbound"]["passed"]
            ),
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


def evaluate_causal(review_root: Path, output_root: Path) -> dict[str, Any]:
    manifest = json.loads((review_root / "review_manifest.json").read_text())
    rows = list(manifest["rows"])
    results = []
    for item in manifest.get("results") or []:
        result_path = (
            review_root
            / "expert_screen_rows"
            / f"{int(item['role_index']):02d}_{str(item['episode_id'])[:16]}"
            / "result.json"
        )
        results.append(json.loads(result_path.read_text()))
    selected = lowest_clean_row_per_cell(rows, results)
    failures: list[dict[str, Any]] = []
    required = [
        (family, side)
        for family in PHYSICS_CLEAN_FAMILIES
        for side in ("left", "right")
    ]
    for key in required:
        if key not in selected:
            failures.append(
                {
                    "code": "missing_clean_cell",
                    "family": key[0],
                    "intrusion_side": key[1],
                }
            )
    cells: dict[tuple[str, str], dict[str, Any]] = {}
    for key in required:
        if key not in selected:
            continue
        row = selected[key]["row"]
        result = selected[key]["result"]
        if not is_v101_clean_success(result):
            failures.append(
                {
                    "code": "selected_row_not_clean",
                    "family": key[0],
                    "intrusion_side": key[1],
                }
            )
            continue
        result_path = (
            review_root
            / "expert_screen_rows"
            / f"{int(row['role_index']):02d}_{str(row['episode_id'])[:16]}"
            / "result.json"
        )
        job = {
            "family": key[0],
            "intrusion_side": key[1],
            "row": row,
            "result_path": str(result_path),
            "trajectory_path": str(result_path.parent / "trajectory.json"),
        }
        cell = _run_cell(job, output_root)
        cells[key] = cell
        for direction, window in cell["windows"].items():
            if not window["passed"]:
                failures.append(
                    {
                        "code": "window_failed",
                        "family": key[0],
                        "intrusion_side": key[1],
                        "direction": direction,
                        **{k: window[k] for k in ("changed_sensors", "changed_values", "responding_links")},
                    }
                )
    balance, balance_failures = _side_balance(cells)
    failures.extend(balance_failures)
    passed = not failures and len(cells) == 6
    artifact = {
        "schema_version": "pact_place_v101_empirical_causal_v1",
        "contract_version": CONTRACT_VERSION,
        "review_manifest_sha256": manifest.get("artifact_sha256"),
        "admission_floor": ADMISSION_FLOOR,
        "corridor_links": list(CORRIDOR_LINKS),
        "max_paired_changed_value_ratio": MAX_PAIRED_CHANGED_VALUE_RATIO,
        "cells": [
            cells[key]
            for key in required
            if key in cells
        ],
        "side_balance": balance,
        "failures": failures,
        "causal_passed": passed,
        "blocks_phase0": not passed,
        **empty_authorization(),
    }
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    establish_v10_runtime_env()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    artifact = evaluate_causal(args.review_root.resolve(), output_root)
    write_immutable(output_root / "causal.json", artifact)
    print(
        json.dumps(
            {
                "causal_passed": artifact["causal_passed"],
                "blocks_phase0": artifact["blocks_phase0"],
                "failures": artifact["failures"],
                "output": str(output_root / "causal.json"),
            },
            indent=2,
        )
    )
    return 0 if artifact["causal_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
