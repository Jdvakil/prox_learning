#!/usr/bin/env python3
"""A0c: larger clutter, inner face held at |y|=0.29, rear box out of the wall.

No rollouts. Do not overwrite the v6 or v6b sweeps. Do not move the inner
face. Do not shrink the boxes.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MOLMO = ROOT / "submodules" / "molmospaces"
for search_path in (ROOT / "scripts", MOLMO):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from pact_place_corridor_contract import CLUTTER_SLOT_NAMES, sha256_payload  # noqa: E402
from run_pact_place_clutter_sweep import (  # noqa: E402
    ENCLOSURE_Y_ABS,
    PANEL_X_SPAN,
    SCENE_XML,
    SHELF_TOP_Z,
    TARGET_CLEARANCE_M,
    TARGET_ENVELOPE_X,
    TARGET_ENVELOPE_Y,
    V5_ROW0,
)

TUBE_X0 = 0.58
DEPTH_MIN_M = 0.18
BACK_WALL_OFFSET_M = 0.02
SHALLOWEST_BACK_WALL_X_M = TUBE_X0 + DEPTH_MIN_M + BACK_WALL_OFFSET_M  # 0.78
MAX_OUTER_X_M = 0.775
MAX_OUTER_Y_ABS_M = 0.39
MIN_INNER_Y_ABS_M = 0.29
HALF_X_M = 0.025
HALF_Y_M = 0.05
CENTER_ABS_Y_M = 0.34
X_FRONT_M = 0.70
X_REAR_M = 0.75
CANDIDATE_TOP_Z_M = (0.80, 0.82)
MAX_TOP_Z_M = 0.82
OUTPUT = ROOT / "diagnostics_output/pact_place_clutter_sweep_v6c/analysis.json"


def aabb_gap_xy(
    center_xy: tuple[float, float],
    half_x: float,
    half_y: float,
) -> float:
    cx, cy = center_xy
    cmin = (cx - half_x, cy - half_y)
    cmax = (cx + half_x, cy + half_y)
    tmin = (TARGET_ENVELOPE_X[0], TARGET_ENVELOPE_Y[0])
    tmax = (TARGET_ENVELOPE_X[1], TARGET_ENVELOPE_Y[1])
    dx = max(tmin[0] - cmax[0], cmin[0] - tmax[0], 0.0)
    dy = max(tmin[1] - cmax[1], cmin[1] - tmax[1], 0.0)
    if dx == 0.0 and dy == 0.0:
        overlap_x = min(cmax[0], tmax[0]) - max(cmin[0], tmin[0])
        overlap_y = min(cmax[1], tmax[1]) - max(cmin[1], tmin[1])
        return -min(overlap_x, overlap_y)
    return float((dx**2 + dy**2) ** 0.5)


def slot_geometry(
    center_xy: tuple[float, float],
    half_x: float,
    half_y: float,
    height: float,
    episode_back_wall_x: float,
) -> dict[str, Any]:
    cx, cy = center_xy
    inner_face_abs_y = abs(cy) - half_y
    outer_face_abs_y = abs(cy) + half_y
    inner_face_x = cx - half_x
    outer_face_x = cx + half_x
    top_z = SHELF_TOP_Z + height
    return {
        "inner_face_abs_y_m": float(inner_face_abs_y),
        "outer_face_abs_y_m": float(outer_face_abs_y),
        "inner_face_x_m": float(inner_face_x),
        "outer_face_x_m": float(outer_face_x),
        "top_z_m": float(top_z),
        "inner_face_ok": bool(inner_face_abs_y + 1e-12 >= MIN_INNER_Y_ABS_M),
        "outer_y_ok": bool(outer_face_abs_y - 1e-12 <= MAX_OUTER_Y_ABS_M),
        "outer_x_ok": bool(outer_face_x - 1e-12 <= MAX_OUTER_X_M),
        "top_z_ok": bool(top_z - 1e-12 <= MAX_TOP_Z_M),
        "clear_of_panel_x_span": bool(inner_face_x + 1e-12 >= PANEL_X_SPAN[1]),
        "inside_enclosure_y": bool(outer_face_abs_y - 1e-12 <= ENCLOSURE_Y_ABS),
        "on_shelf": bool(height > 0.0),
        "back_wall_margin_vs_shallowest_m": float(SHALLOWEST_BACK_WALL_X_M - outer_face_x),
        "back_wall_margin_vs_episode_m": float(episode_back_wall_x - outer_face_x),
        "clear_of_shallowest_back_wall": bool(
            outer_face_x - 1e-12 <= SHALLOWEST_BACK_WALL_X_M
        ),
        "clear_of_episode_back_wall": bool(outer_face_x - 1e-12 <= episode_back_wall_x),
        "nominal_outer_x_limit_ok": bool(outer_face_x - 1e-12 <= MAX_OUTER_X_M),
    }


def enclosure_ok(geom: dict[str, Any]) -> dict[str, Any]:
    ok = bool(
        geom["inner_face_ok"]
        and geom["outer_y_ok"]
        and geom["outer_x_ok"]
        and geom["top_z_ok"]
        and geom["clear_of_panel_x_span"]
        and geom["inside_enclosure_y"]
        and geom["on_shelf"]
        and geom["clear_of_shallowest_back_wall"]
        and geom["nominal_outer_x_limit_ok"]
    )
    return {**geom, "ok": ok}


def candidate_sets() -> list[dict[str, Any]]:
    slots = {
        "l0": (float(X_FRONT_M), float(CENTER_ABS_Y_M)),
        "l1": (float(X_REAR_M), float(CENTER_ABS_Y_M)),
        "r0": (float(X_FRONT_M), float(-CENTER_ABS_Y_M)),
        "r1": (float(X_REAR_M), float(-CENTER_ABS_Y_M)),
    }
    sets: list[dict[str, Any]] = []
    for set_id, top_z in enumerate(CANDIDATE_TOP_Z_M):
        height = float(top_z) - SHELF_TOP_Z
        sets.append(
            {
                "set_id": set_id,
                "top_z_m": float(top_z),
                "height_m": height,
                "half_x_m": float(HALF_X_M),
                "half_y_m": float(HALF_Y_M),
                "slots_xy_m": slots,
            }
        )
    return sets


def pose_clutter(
    env,
    slots_xy: dict[str, tuple[float, float]],
    height: float,
    half_x: float,
    half_y: float,
) -> None:
    import mujoco
    import numpy as np

    model, data = env.current_model, env.current_data
    hz = height / 2.0
    for slot in CLUTTER_SLOT_NAMES:
        x, y = slots_xy[slot]
        body = f"pact_clutter_{slot}"
        mid = int(model.body_mocapid[model.body(body).id])
        data.mocap_pos[mid] = np.asarray([x, y, SHELF_TOP_Z + hz], dtype=float)
        gid = int(model.geom(f"pact_clutter_{slot}_g").id)
        model.geom_size[gid] = np.asarray([half_x, half_y, hz], dtype=float)
    mujoco.mj_forward(model, data)


def mean_distance_to_target(slots_xy: dict[str, tuple[float, float]]) -> float:
    tx, ty = 0.76, 0.0
    dists = [
        float(((x - tx) ** 2 + (y - ty) ** 2) ** 0.5) for x, y in slots_xy.values()
    ]
    return float(sum(dists) / len(dists))


def main() -> int:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("MLSPACES_ASSETS_DIR", str(ROOT / "assets"))
    os.environ.pop("DISPLAY", None)

    import mujoco
    import numpy as np
    from molmo_spaces.data_generation.pipeline import (
        cleanup_episode_resources,
        setup_policy,
    )
    from molmo_spaces.data_generation.runtime_compat import assert_supported_runtime
    from molmo_spaces.tasks.enclosure_reach import TUBE_X0 as LIVE_TUBE_X0
    from run_pact_place_expert_screen import _make_config

    assert_supported_runtime(strict=True)
    if abs(float(LIVE_TUBE_X0) - TUBE_X0) > 1e-12:
        raise RuntimeError(f"TUBE_X0 drifted: {LIVE_TUBE_X0}")
    result = json.loads((V5_ROW0 / "result.json").read_text())
    row = json.loads((ROOT / "configs/pact_place_corridor_v5.json").read_text())[
        "expert_screen_rows"
    ][0]
    scratch = Path(tempfile.mkdtemp(prefix="pact_place_clutter_a0c_"))
    task = policy = sampler = None
    records: list[dict[str, Any]] = []
    episode_depth = None
    episode_back_wall_x = None
    try:
        config = _make_config(scratch / "dummy.json", scene_xml=SCENE_XML)
        sampler = config.task_sampler_config.task_sampler_class(config)
        sampler.seed_task_sampling(int(result["selected_seed"]["seed_u32"]))
        sampler.set_pact_manifest_row(row)
        task = sampler.sample_task(house_index=int(row["scene_template_house_index"]))
        if task is None:
            raise RuntimeError("sample_task returned None")
        policy = setup_policy(config, task, None, None)
        task.reset()
        scene = getattr(task, "scene_params", {}) or {}
        episode_depth = float(scene["depth"])
        episode_back_wall_x = float(LIVE_TUBE_X0 + episode_depth + BACK_WALL_OFFSET_M)
        robot_view = task.env.current_robot.robot_view
        reset_qpos = {
            key: np.asarray(value, dtype=float).copy()
            for key, value in robot_view.get_qpos_dict().items()
        }

        def expert_trajectory_ok() -> tuple[bool, str | None]:
            robot_view.set_qpos_dict(reset_qpos)
            mujoco.mj_forward(task.env.current_model, task.env.current_data)
            try:
                policy._compute_trajectory()
            except ValueError as error:
                return False, str(error)
            return True, None

        for spec in candidate_sets():
            slots = spec["slots_xy_m"]
            height = spec["height_m"]
            half_x = spec["half_x_m"]
            half_y = spec["half_y_m"]
            slot_records = {}
            footprint_ok = True
            enclosure_all_ok = True
            min_gap = None
            min_inner = None
            max_outer_x = None
            max_outer_y = None
            for slot, xy in slots.items():
                gap = aabb_gap_xy(xy, half_x, half_y)
                geom = enclosure_ok(
                    slot_geometry(
                        xy, half_x, half_y, height, episode_back_wall_x
                    )
                )
                slot_records[slot] = {
                    "xy_m": [xy[0], xy[1]],
                    "target_envelope_gap_m": gap,
                    "clears_4cm": bool(gap + 1e-12 >= TARGET_CLEARANCE_M),
                    **geom,
                }
                footprint_ok = footprint_ok and slot_records[slot]["clears_4cm"]
                enclosure_all_ok = enclosure_all_ok and geom["ok"]
                min_gap = gap if min_gap is None else min(min_gap, gap)
                min_inner = (
                    geom["inner_face_abs_y_m"]
                    if min_inner is None
                    else min(min_inner, geom["inner_face_abs_y_m"])
                )
                max_outer_x = (
                    geom["outer_face_x_m"]
                    if max_outer_x is None
                    else max(max_outer_x, geom["outer_face_x_m"])
                )
                max_outer_y = (
                    geom["outer_face_abs_y_m"]
                    if max_outer_y is None
                    else max(max_outer_y, geom["outer_face_abs_y_m"])
                )
            record: dict[str, Any] = {
                **spec,
                "slots": slot_records,
                "min_target_envelope_gap_m": min_gap,
                "min_inner_face_abs_y_m": min_inner,
                "max_outer_face_x_m": max_outer_x,
                "max_outer_face_abs_y_m": max_outer_y,
                "footprint_ok": bool(footprint_ok),
                "enclosure_ok": bool(enclosure_all_ok),
                "mean_distance_to_target_xy_m": mean_distance_to_target(slots),
            }
            if not (footprint_ok and enclosure_all_ok):
                record["ik_ok"] = False
                record["ik_error"] = None
                record["eligible"] = False
                records.append(record)
                continue
            pose_clutter(task.env, slots, height, half_x, half_y)
            ik_ok, ik_error = expert_trajectory_ok()
            record["ik_ok"] = bool(ik_ok)
            record["ik_error"] = ik_error
            record["eligible"] = bool(
                record["footprint_ok"] and record["enclosure_ok"] and record["ik_ok"]
            )
            records.append(record)
    finally:
        cleanup_episode_resources(
            task=task,
            policy=policy,
            task_sampler=sampler,
            preloaded_policy=None,
            close_task_sampler=True,
        )

    eligible = [item for item in records if item.get("eligible")]
    chosen = None
    if eligible:
        # Presence first: taller top, then set_id. Inner face is identical.
        eligible.sort(key=lambda item: (-item["top_z_m"], item["set_id"]))
        chosen = {
            "set_id": eligible[0]["set_id"],
            "slots_xy_m": eligible[0]["slots_xy_m"],
            "height_m": eligible[0]["height_m"],
            "top_z_m": eligible[0]["top_z_m"],
            "half_x_m": eligible[0]["half_x_m"],
            "half_y_m": eligible[0]["half_y_m"],
            "min_inner_face_abs_y_m": eligible[0]["min_inner_face_abs_y_m"],
            "max_outer_face_x_m": eligible[0]["max_outer_face_x_m"],
            "max_outer_face_abs_y_m": eligible[0]["max_outer_face_abs_y_m"],
            "min_target_envelope_gap_m": eligible[0]["min_target_envelope_gap_m"],
            "mean_distance_to_target_xy_m": eligible[0]["mean_distance_to_target_xy_m"],
            "tie_break": "taller_top_then_set_id",
        }
    analysis = {
        "schema_version": "pact_place_clutter_sweep_v6c",
        "no_rollouts": True,
        "ik_from_expert_trajectory_not_reset_qpos": True,
        "inner_face_held_at_abs_y_m": MIN_INNER_Y_ABS_M,
        "boxes_not_shrunk": True,
        "v6_sweep_untouched": True,
        "v6b_sweep_untouched": True,
        "n_candidate_sets": len(records),
        "n_footprint_ok": sum(item["footprint_ok"] for item in records),
        "n_enclosure_ok": sum(item["enclosure_ok"] for item in records),
        "n_ik_ok": sum(bool(item.get("ik_ok")) for item in records),
        "n_eligible": len(eligible),
        "half_x_m": HALF_X_M,
        "half_y_m": HALF_Y_M,
        "center_abs_y_m": CENTER_ABS_Y_M,
        "x_front_m": X_FRONT_M,
        "x_rear_m": X_REAR_M,
        "candidate_top_z_m": list(CANDIDATE_TOP_Z_M),
        "max_outer_x_m": MAX_OUTER_X_M,
        "max_outer_y_abs_m": MAX_OUTER_Y_ABS_M,
        "shallowest_back_wall_x_m": SHALLOWEST_BACK_WALL_X_M,
        "episode_depth_m": episode_depth,
        "episode_back_wall_x_m": episode_back_wall_x,
        "episode_depth_source": "v5_row0_sampled_task",
        "target_envelope_xy": {
            "x_m": list(TARGET_ENVELOPE_X),
            "y_m": list(TARGET_ENVELOPE_Y),
        },
        "target_clearance_m": TARGET_CLEARANCE_M,
        "enclosure_y_abs_m": ENCLOSURE_Y_ABS,
        "chosen": chosen,
        "stop_if_none_eligible": chosen is None,
        "candidates": records,
    }
    analysis["sweep_sha256"] = sha256_payload(analysis)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    print(OUTPUT)
    print(
        json.dumps(
            {
                k: analysis[k]
                for k in (
                    "n_candidate_sets",
                    "n_footprint_ok",
                    "n_enclosure_ok",
                    "n_ik_ok",
                    "n_eligible",
                    "episode_depth_m",
                    "episode_back_wall_x_m",
                    "chosen",
                    "sweep_sha256",
                )
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if chosen is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
