#!/usr/bin/env python3
"""Exact link-clearance and raw-PACT admission for V9.5 preview candidates."""

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
MOLMO = ROOT / "submodules/molmospaces"
for path in (ROOT / "scripts", MOLMO):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pact_geom_distance import counters, reset_counters, true_distance
from pact_place_corridor_contract import sha256_file
from pact_place_v9_contract import sha256_payload
from run_pact_place_expert_screen import _make_config
from run_pact_place_swept_volume_v7 import geom_groups
from run_pact_place_v8_baseline import _physical_geoms
from run_pact_place_v9_v0c3_causal_proximity import (
    ABS_DELTA_FLOOR_M,
    _causal_metrics,
    _render_observation,
)

DEFAULT_PREVIEW_ROOT = ROOT / "diagnostics_output/pact_place_v95_low_wall_preview"
SCENE_XML = ROOT / "submodules/molmospaces/molmo_spaces/data_generation/custom_scenes/pact_place_corridor_v5.xml"
MIN_CLEARANCE_M = 0.020
MAX_LINK_ENGAGEMENT_CLEARANCE_M = 0.100
MIN_BOW_M = 0.040
PARK_Z_M = -2.0


def _fixture_indices(steps: list[dict[str, Any]]) -> list[int]:
    hits = [
        i
        for i, step in enumerate(steps)
        if "wall_fixture" in str(step.get("policy_phase") or "")
    ]
    if not hits:
        raise ValueError("trajectory has no wall-fixture maneuver")
    expanded = set()
    for hit in hits:
        expanded.update(range(max(0, hit - 8), min(len(steps), hit + 9)))
    return sorted(expanded)


def _validate_episode(item: dict[str, Any], output_root: Path) -> dict[str, Any]:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("MLSPACES_ASSETS_DIR", str(ROOT / "assets"))
    os.environ.pop("DISPLAY", None)

    import mujoco
    from molmo_spaces.data_generation.pipeline import cleanup_episode_resources

    row = dict(item["row"])
    result_path = Path(item["result_path"])
    trajectory_path = Path(str(item["trajectory_path"]))
    result = json.loads(result_path.read_text())
    steps = list(json.loads(trajectory_path.read_text())["steps"])
    indices = _fixture_indices(steps)
    scratch = Path(tempfile.mkdtemp(prefix="pact_place_v95_admission_"))
    task = sampler = None
    try:
        config = _make_config(
            scratch / "dummy.json", scene_xml=SCENE_XML, sampler_class=row["sampler_class"]
        )
        config.proximity_sensor_period_ms = 16.6667
        sampler = config.task_sampler_config.task_sampler_class(config)
        sampler.seed_task_sampling(int(result["selected_seed"]["seed_u32"]))
        sampler.set_pact_manifest_row(row)
        task = sampler.sample_task(house_index=int(row["scene_template_house_index"]))
        if task is None:
            raise RuntimeError("V9.5 task reconstruction failed")
        task.reset()
        model, data = task.env.current_model, task.env.current_data
        sensor_names = list(task._proximity_camera_names)
        fixture_body = str((result.get("scene_params") or {}).get("pact_v95_active_mount_body"))
        fixture_gid = int(model.geom(f"{fixture_body}_g").id)
        mocap_id = int(np.asarray(model.body(fixture_body).mocapid).reshape(-1)[0])
        fixture_position = np.asarray(data.mocap_pos[mocap_id], dtype=float).copy()
        groups = geom_groups(model)
        measured_groups = ("cup", "left_finger", "right_finger", "hand", "link7", "link6", "link5")
        group_geoms = {
            name: _physical_geoms(model, list(groups[name])) for name in measured_groups
        }
        minima = {name: float("inf") for name in measured_groups}
        present_frames, parked_frames, phases = [], [], []
        reset_counters()
        for index in indices:
            qpos = np.asarray(steps[index]["qpos"], dtype=float)
            data.qpos[:] = qpos
            data.mocap_pos[mocap_id] = fixture_position
            mujoco.mj_forward(model, data)
            for name in measured_groups:
                minima[name] = min(
                    minima[name],
                    float(true_distance(model, data, group_geoms[name], [fixture_gid])),
                )
            present_frames.append(_render_observation(task, sensor_names))
            phases.append(str(steps[index].get("policy_phase") or "unknown"))

            data.qpos[:] = qpos
            data.mocap_pos[mocap_id] = fixture_position
            data.mocap_pos[mocap_id, 2] = PARK_Z_M
            mujoco.mj_forward(model, data)
            parked_frames.append(_render_observation(task, sensor_names))

        data.qpos[:] = np.asarray(steps[indices[len(indices) // 2]]["qpos"], dtype=float)
        data.mocap_pos[mocap_id] = fixture_position
        mujoco.mj_forward(model, data)
        repeat = _render_observation(task, sensor_names)
        noise = float(np.max(np.abs(repeat - present_frames[len(indices) // 2])))
        threshold = max(ABS_DELTA_FLOOR_M, 10.0 * noise)
        raw = _causal_metrics(
            np.stack(present_frames),
            np.stack(parked_frames),
            sensor_names,
            np.asarray(indices, dtype=np.int32),
            phases,
            threshold,
        )
        link56_changed = sum(
            int(record["changed_values"])
            for record in raw["per_link"]
            if str(record["link"]).startswith(("link5", "link6"))
        )
        wall_bow = float(item["wall_fixture_bow_m"])
        safe_clearance = min(minima.values()) >= MIN_CLEARANCE_M
        link_primary = min(minima["link5"], minima["link6"]) <= MAX_LINK_ENGAGEMENT_CLEARANCE_M
        passed = bool(
            item.get("clean_success") is True
            and wall_bow >= MIN_BOW_M
            and safe_clearance
            and link_primary
            and raw["changed_values"] > 0
            and link56_changed > 0
            and int(item.get("clutter_contacts", 0)) == 0
            and int(item.get("hazard_bar_contacts", 0)) == 0
            and int(item.get("other_environment_contacts", 0)) == 0
        )
        return {
            "episode_id": row["episode_id"],
            "panel_side": row["intrusion_side"],
            "wall_support": row["mounted_wall_support"],
            "passed": passed,
            "wall_fixture_bow_m": wall_bow,
            "minimum_clearance_by_group_m": minima,
            "minimum_clearance_m": min(minima.values()),
            "link_primary_engagement": link_primary,
            "raw_fixture_causal_effect": raw,
            "raw_link5_link6_changed_values": link56_changed,
            "causal_threshold_m": threshold,
            "baseline_repeat_max_abs_delta_m": noise,
            "distance_instrument_counters": counters(),
            "source_result_path": str(result_path.relative_to(ROOT)),
            "source_result_sha256": sha256_file(result_path),
            "source_trajectory_path": str(trajectory_path.relative_to(ROOT)),
            "source_trajectory_sha256": sha256_file(trajectory_path),
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
    parser.add_argument("--preview-root", type=Path, default=DEFAULT_PREVIEW_ROOT)
    args = parser.parse_args()
    preview_root = args.preview_root.resolve()
    manifest_path = preview_root / "preview_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    candidates = [
        item
        for item in manifest["attempts"]
        if item.get("clean_success") is True
        and float(item.get("wall_fixture_bow_m", 0.0)) >= MIN_BOW_M
    ]
    by_cell = {}
    for item in candidates:
        by_cell.setdefault(int(item["cell_index"]), item)
    rows = [_validate_episode(by_cell[index], preview_root) for index in range(4)] if len(by_cell) == 4 else []
    passed = len(rows) == 4 and all(item["passed"] for item in rows)
    document = {
        "schema_version": "pact_place_v9_5_fixture_admission_v1",
        "role": "blocking_fixture_admission_not_collection_gate",
        "passed": passed,
        "authorizes_gate": False,
        "authorizes_collection": False,
        "minimum_clearance_m": MIN_CLEARANCE_M,
        "maximum_link_engagement_clearance_m": MAX_LINK_ENGAGEMENT_CLEARANCE_M,
        "minimum_fixture_bow_m": MIN_BOW_M,
        "uses_real_40_sensor_observation": True,
        "uses_exact_geom_distance": True,
        "admitted_episode_ids": [item["episode_id"] for item in rows if item["passed"]],
        "episodes": rows,
    }
    document["validation_sha256"] = sha256_payload(document)
    output = preview_root / "fixture_admission.json"
    output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(output)
    print(json.dumps({"passed": passed, "episodes": len(rows)}, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
