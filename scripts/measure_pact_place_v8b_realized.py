#!/usr/bin/env python3
"""Measure V8B Pass-2 episodes with true MuJoCo geom-to-geom distance."""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MOLMO = ROOT / "submodules" / "molmospaces"
for path in (ROOT / "scripts", MOLMO):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pact_place_corridor_contract import sha256_payload  # noqa: E402
from run_pact_place_expert_screen import _make_config, write_json_atomic  # noqa: E402
from run_pact_place_swept_volume_v7 import geom_groups  # noqa: E402
from run_pact_place_v7_replay_videos import WRIST_CAMERA_MJCF  # noqa: E402
from run_pact_place_v8_baseline import (  # noqa: E402
    _active_clutter_geoms,
    _any_visible,
    _physical_geoms,
    _target_geoms,
)

CONFIG_PATH = ROOT / "configs/pact_place_corridor_v8b_pass1.json"
RUNS_PATH = ROOT / "diagnostics_output/pact_place_corridor_v8b_pass2c/realized_runs.json"
OUTPUT_PATH = ROOT / "diagnostics_output/pact_place_clutter_sweep_v8b/analysis_pass2.json"
SCENE_XML = MOLMO / "molmo_spaces/data_generation/custom_scenes/pact_place_corridor_v5.xml"
LINKS = tuple(f"link{index}" for index in range(1, 7))


def _true_distance(model, data, left: list[int], right: list[int]) -> float:
    best = float("inf")
    segment = np.empty(6, dtype=np.float64)
    for left_geom in left:
        for right_geom in right:
            value = float(
                mujoco.mj_geomDistance(
                    model, data, int(left_geom), int(right_geom), 10.0, segment
                )
            )
            # MuJoCo 3.5 returns a scalar zero for some separated mesh/primitive
            # pairs while still populating ``fromto`` with distinct closest
            # points. Use that oriented closest-point segment as the plan's
            # mesh-distance fallback; preserve negative penetration values.
            if value == 0.0:
                segment_distance = float(np.linalg.norm(segment[3:] - segment[:3]))
                if segment_distance > 1e-9:
                    value = segment_distance
            if value < best:
                best = value
    return best


def _prepare(row: dict[str, Any], selected_seed: dict[str, int]):
    scratch = Path(tempfile.mkdtemp(prefix="pact_place_v8b_measure_"))
    config = _make_config(scratch / "result.json", scene_xml=SCENE_XML)
    sampler = config.task_sampler_config.task_sampler_class(config)
    sampler.seed_task_sampling(int(selected_seed["seed_u32"]))
    sampler.set_pact_manifest_row(row)
    task = sampler.sample_task(house_index=int(row["scene_template_house_index"]))
    if task is None:
        raise RuntimeError("recorded realized seed no longer samples")
    task.reset()
    return task, sampler, scratch


def measure(row: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    from molmo_spaces.data_generation.pipeline import cleanup_episode_resources

    result_path = ROOT / record["result_path"]
    result = json.loads(result_path.read_text())
    trajectory = json.loads((result_path.parent / "trajectory.json").read_text())
    task = sampler = scratch = None
    try:
        task, sampler, scratch = _prepare(row, result["selected_seed"])
        model, data = task.env.current_model, task.env.current_data
        all_groups = geom_groups(model)
        robot = {
            link: _physical_geoms(model, list(all_groups.get(link) or []))
            for link in LINKS
        }
        hand = _physical_geoms(
            model,
            [
                geom
                for name in ("link7", "hand", "left_finger", "right_finger")
                for geom in list(all_groups.get(name) or [])
            ],
        )
        cup = _physical_geoms(model, _target_geoms(model))
        wrist_id = int(model.camera(WRIST_CAMERA_MJCF).id)
        by_link = {link: [] for link in LINKS}
        cup_distance, hand_distance = [], []
        clutter_visible, target_visible, phases = [], [], []
        for step in trajectory["steps"]:
            qpos = np.asarray(step["qpos"], dtype=float)
            if qpos.shape != data.qpos.shape:
                raise RuntimeError(f"qpos width {len(qpos)} != model nq {model.nq}")
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            clutter = _physical_geoms(model, _active_clutter_geoms(model, data))
            for link in LINKS:
                by_link[link].append(_true_distance(model, data, robot[link], clutter))
            cup_distance.append(_true_distance(model, data, cup, clutter))
            hand_distance.append(_true_distance(model, data, hand, clutter))
            clutter_visible.append(_any_visible(model, data, wrist_id, clutter))
            target_visible.append(_any_visible(model, data, wrist_id, cup))
            phases.append(str(step.get("policy_phase") or "unknown"))
        arrays = {link: np.asarray(values, dtype=float) for link, values in by_link.items()}
        matrix = np.stack([arrays[link] for link in LINKS], axis=1)
        frame_min = matrix.min(axis=1)
        frame_index = int(np.argmin(frame_min))
        min_by_link = {link: float(arrays[link].min()) for link in LINKS}
        cup_min = float(np.min(cup_distance))
        visibility = np.asarray(clutter_visible, dtype=bool)
        layout = row["pact_clutter_layout"]
        category_counts = {
            key: value for key, value in __import__("collections").Counter(
                item["category"] for item in layout["objects"]
            ).items()
        }
        return {
            "family": row["family"],
            "role_index": row["role_index"],
            "layout_id": row["layout_id"],
            "status": result["status"],
            "clean_success": bool(result.get("clean_success")),
            "n_steps": len(frame_min),
            "min_clearance_by_link_m": min_by_link,
            "min_link_clearance_m": float(frame_min[frame_index]),
            "min_cup_clearance_m": cup_min,
            "min_hand_clearance_m": float(np.min(hand_distance)),
            "closest_robot_link": min(min_by_link, key=min_by_link.get),
            "phase_of_min_clearance": phases[frame_index],
            "cup_is_closest_body": bool(cup_min < float(frame_min[frame_index])),
            "frames_link_clearance_lt_5cm": int(np.sum(frame_min < 0.05)),
            "frames_link_clearance_lt_10cm": int(np.sum(frame_min < 0.10)),
            "frames_link_clearance_lt_15cm": int(np.sum(frame_min < 0.15)),
            "n_distinct_links_exposed": int(sum(value < 0.10 for value in min_by_link.values())),
            "visibility_at_min_link_clearance": bool(visibility[frame_index]),
            "clutter_visible_frame_fraction": float(visibility.mean()),
            "target_visible_frame_fraction": float(np.mean(target_visible)),
            "object_count": len(layout["objects"]),
            "overhead_count": sum(float(item["center_m"][2]) - float(item["half_m"][2]) >= 0.95 for item in layout["objects"]),
            "max_category_count": max(category_counts.values()),
            "category_counts": category_counts,
        }
    finally:
        cleanup_episode_resources(
            task=task,
            policy=None,
            task_sampler=sampler,
            preloaded_policy=None,
            close_task_sampler=sampler is not None,
        )
        if scratch is not None:
            shutil.rmtree(scratch, ignore_errors=True)


def main() -> int:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("MLSPACES_ASSETS_DIR", "/root/prox_learning/assets")
    os.environ.pop("DISPLAY", None)
    config = json.loads(CONFIG_PATH.read_text())
    runs = json.loads(RUNS_PATH.read_text())
    if not runs["one_real_episode_per_family"]:
        raise SystemExit("Pass 2 does not contain one realized episode per family")
    rows = {int(row["role_index"]): row for row in config["family_review_rows"]}
    measured = [measure(rows[int(record["role_index"])], record) for record in runs["realized_results"]]
    aggregate = {
        "n_episodes": len(measured),
        "cup_is_closest_body_count": sum(row["cup_is_closest_body"] for row in measured),
        "cup_is_closest_body_fraction": float(np.mean([row["cup_is_closest_body"] for row in measured])),
        "nonzero_visibility_at_min_count": sum(row["visibility_at_min_link_clearance"] for row in measured),
        "nonzero_visibility_at_min_fraction": float(np.mean([row["visibility_at_min_link_clearance"] for row in measured])),
        "mean_distinct_links_exposed": float(np.mean([row["n_distinct_links_exposed"] for row in measured])),
        "frames_link_clearance_lt_10cm": sum(row["frames_link_clearance_lt_10cm"] for row in measured),
        "episodes_with_frames_link_clearance_lt_5cm": sum(row["frames_link_clearance_lt_5cm"] > 0 for row in measured),
        "objects_per_layout": [row["object_count"] for row in measured],
        "overhead_per_layout": [row["overhead_count"] for row in measured],
        "max_category_per_layout": [row["max_category_count"] for row in measured],
    }
    gates = {
        "cup_is_closest_body_fraction_le_0_25": aggregate["cup_is_closest_body_fraction"] <= 0.25,
        "nonzero_visibility_at_min_fraction_ge_1_3": aggregate["nonzero_visibility_at_min_fraction"] >= 1 / 3,
        "mean_distinct_links_exposed_ge_3": aggregate["mean_distinct_links_exposed"] >= 3.0,
        "frames_link_clearance_lt_10cm_ge_1852": aggregate["frames_link_clearance_lt_10cm"] >= 1852,
        "lt_5cm_in_at_least_half": aggregate["episodes_with_frames_link_clearance_lt_5cm"] >= 3,
        "objects_8_to_12": all(8 <= value <= 12 for value in aggregate["objects_per_layout"]),
        "overhead_at_least_2": all(value >= 2 for value in aggregate["overhead_per_layout"]),
        "category_max_2": all(value <= 2 for value in aggregate["max_category_per_layout"]),
    }
    document = {
        "schema_version": "pact_place_v8b_realized_measurement_v1",
        "role": "pass2_realized_measurement_not_a_gate",
        "authorizes_gate": False,
        "distance_instrument": "mujoco.mj_geomDistance_real_collision_geoms_to_robot_collision_geoms",
        "rows": measured,
        "aggregate": aggregate,
        "admission_gates": gates,
        "all_admission_gates_pass": all(gates.values()),
    }
    document["analysis_sha256"] = sha256_payload(document)
    write_json_atomic(OUTPUT_PATH, document)
    print(json.dumps({"aggregate": aggregate, "admission_gates": gates, "all_pass": all(gates.values())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
