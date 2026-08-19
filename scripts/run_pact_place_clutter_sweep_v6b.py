#!/usr/bin/env python3
"""A0b: extended lateral clutter grid. No rollouts.

Step 0 showed carried-cup vs opposite-side inner box, not a robot link.
This grid is |y| in {0.32, 0.36, 0.40} — the range the v6 grid never reached.
Do not overwrite the v6 sweep. Do not shrink boxes.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from itertools import combinations, product
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MOLMO = ROOT / "submodules" / "molmospaces"
for search_path in (ROOT / "scripts", MOLMO):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from pact_place_corridor_contract import CLUTTER_SLOT_NAMES, sha256_payload  # noqa: E402
from run_pact_place_clutter_sweep import (  # noqa: E402
    BACK_WALL_X_M,
    CANDIDATE_HEIGHT_M,
    CANDIDATE_X,
    ENCLOSURE_Y_ABS,
    HALF_XY_M,
    PANEL_X_SPAN,
    SCENE_XML,
    SHELF_TOP_Z,
    TARGET_CLEARANCE_M,
    TARGET_ENVELOPE_X,
    TARGET_ENVELOPE_Y,
    V5_ROW0,
    aabb_gap_xy,
    enclosure_ok,
    mean_distance_to_target,
    pose_clutter,
)

CANDIDATE_ABS_Y = (0.32, 0.36, 0.40)
OUTPUT = ROOT / "diagnostics_output/pact_place_clutter_sweep_v6b/analysis.json"


def candidate_sets() -> list[dict[str, Any]]:
    sets: list[dict[str, Any]] = []
    set_id = 0
    for height in CANDIDATE_HEIGHT_M:
        for y0, y1 in combinations(CANDIDATE_ABS_Y, 2):
            for x0, x1 in product(CANDIDATE_X, CANDIDATE_X):
                slots = {
                    "l0": (float(x0), float(y0)),
                    "r0": (float(x0), float(-y0)),
                    "l1": (float(x1), float(y1)),
                    "r1": (float(x1), float(-y1)),
                }
                sets.append(
                    {
                        "set_id": set_id,
                        "height_m": float(height),
                        "half_xy_m": float(HALF_XY_M),
                        "slots_xy_m": slots,
                    }
                )
                set_id += 1
        for y in CANDIDATE_ABS_Y:
            x0, x1 = CANDIDATE_X
            slots = {
                "l0": (float(x0), float(y)),
                "r0": (float(x0), float(-y)),
                "l1": (float(x1), float(y)),
                "r1": (float(x1), float(-y)),
            }
            sets.append(
                {
                    "set_id": set_id,
                    "height_m": float(height),
                    "half_xy_m": float(HALF_XY_M),
                    "slots_xy_m": slots,
                }
            )
            set_id += 1
    return sets


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
    from run_pact_place_expert_screen import _make_config

    assert_supported_runtime(strict=True)
    result = json.loads((V5_ROW0 / "result.json").read_text())
    row = json.loads((ROOT / "configs/pact_place_corridor_v5.json").read_text())[
        "expert_screen_rows"
    ][0]
    scratch = Path(tempfile.mkdtemp(prefix="pact_place_clutter_a0b_"))
    task = policy = sampler = None
    records: list[dict[str, Any]] = []
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
            half_xy = spec["half_xy_m"]
            slot_records = {}
            footprint_ok = True
            enclosure_all_ok = True
            min_gap = None
            for slot, xy in slots.items():
                gap = aabb_gap_xy(xy, half_xy)
                enc = enclosure_ok(xy, half_xy, height)
                slot_records[slot] = {
                    "xy_m": [xy[0], xy[1]],
                    "target_envelope_gap_m": gap,
                    "clears_4cm": bool(gap + 1e-12 >= TARGET_CLEARANCE_M),
                    **enc,
                }
                footprint_ok = footprint_ok and slot_records[slot]["clears_4cm"]
                enclosure_all_ok = enclosure_all_ok and enc["ok"]
                min_gap = gap if min_gap is None else min(min_gap, gap)
            record: dict[str, Any] = {
                **spec,
                "slots": slot_records,
                "min_target_envelope_gap_m": min_gap,
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
            pose_clutter(task.env, slots, height, half_xy)
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
        eligible.sort(
            key=lambda item: (
                item["mean_distance_to_target_xy_m"],
                item["height_m"],
                item["set_id"],
            )
        )
        chosen = {
            "set_id": eligible[0]["set_id"],
            "slots_xy_m": eligible[0]["slots_xy_m"],
            "height_m": eligible[0]["height_m"],
            "half_xy_m": eligible[0]["half_xy_m"],
            "min_target_envelope_gap_m": eligible[0]["min_target_envelope_gap_m"],
            "mean_distance_to_target_xy_m": eligible[0]["mean_distance_to_target_xy_m"],
            "tie_break": "closest_mean_xy_to_target_then_shorter_then_set_id",
        }
    analysis = {
        "schema_version": "pact_place_clutter_sweep_v6b",
        "no_rollouts": True,
        "ik_from_expert_trajectory_not_reset_qpos": True,
        "step0_axis": "lateral",
        "step0_analysis": (
            "diagnostics_output/pact_place_corridor_v6b_contact_bodies/analysis.json"
        ),
        "n_candidate_sets": len(records),
        "n_footprint_ok": sum(item["footprint_ok"] for item in records),
        "n_enclosure_ok": sum(item["enclosure_ok"] for item in records),
        "n_ik_ok": sum(bool(item.get("ik_ok")) for item in records),
        "n_eligible": len(eligible),
        "half_xy_m": HALF_XY_M,
        "target_envelope_xy": {
            "x_m": list(TARGET_ENVELOPE_X),
            "y_m": list(TARGET_ENVELOPE_Y),
        },
        "target_clearance_m": TARGET_CLEARANCE_M,
        "enclosure_y_abs_m": ENCLOSURE_Y_ABS,
        "candidate_x_m": list(CANDIDATE_X),
        "candidate_abs_y_m": list(CANDIDATE_ABS_Y),
        "candidate_height_m": list(CANDIDATE_HEIGHT_M),
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
                    "n_ik_ok",
                    "n_eligible",
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
