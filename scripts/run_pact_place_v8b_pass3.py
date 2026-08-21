#!/usr/bin/env python3
"""B2b Pass 3: rescore every V8B candidate on realized tracks and reselect."""

from __future__ import annotations

import concurrent.futures
import json
import multiprocessing
import os
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MOLMO = ROOT / "submodules" / "molmospaces"
for path in (ROOT / "scripts", MOLMO):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_pact_place_clutter_sweep_v8 as v8  # noqa: E402
from measure_pact_place_v8b_realized import _true_distance  # noqa: E402
from pact_place_corridor_contract import sha256_payload  # noqa: E402
from run_pact_place_clutter_sweep_v8b import (  # noqa: E402
    FAMILIES,
    build_config,
    support_reject,
)
from run_pact_place_expert_screen import _make_config, write_json_atomic  # noqa: E402
from run_pact_place_swept_volume_v7 import geom_groups  # noqa: E402
from run_pact_place_v7_replay_videos import WRIST_CAMERA_MJCF  # noqa: E402
from run_pact_place_v8_baseline import _any_visible, _physical_geoms, _target_geoms  # noqa: E402

PASS1 = ROOT / "diagnostics_output/pact_place_clutter_sweep_v8b/analysis_pass1.json"
PASS2 = ROOT / "diagnostics_output/pact_place_clutter_sweep_v8b/analysis_pass2.json"
PASS2_RUNS = ROOT / "diagnostics_output/pact_place_corridor_v8b_pass2c/realized_runs.json"
PASS1_CONFIG = ROOT / "configs/pact_place_corridor_v8b_pass1.json"
OUTPUT = ROOT / "diagnostics_output/pact_place_clutter_sweep_v8b/analysis.json"
SELECTED_OUTPUT = ROOT / "diagnostics_output/pact_place_clutter_sweep_v8b/selected_layouts.json"
CONFIG_OUTPUT = ROOT / "configs/pact_place_corridor_v8b.json"
SCENE_XML = MOLMO / "molmo_spaces/data_generation/custom_scenes/pact_place_corridor_v5.xml"
LINKS = tuple(f"link{index}" for index in range(1, 7))


def _longest(values: list[bool]) -> int:
    best = current = 0
    for value in values:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


class Context:
    def __init__(self, row: dict[str, Any], record: dict[str, Any]):
        from molmo_spaces.data_generation.pipeline import cleanup_episode_resources

        self._cleanup = cleanup_episode_resources
        self.row = row
        result_path = ROOT / record["result_path"]
        result = json.loads(result_path.read_text())
        self.steps = json.loads((result_path.parent / "trajectory.json").read_text())["steps"]
        self.scratch = Path(tempfile.mkdtemp(prefix="pact_place_v8b_pass3_"))
        config = _make_config(self.scratch / "result.json", scene_xml=SCENE_XML)
        self.sampler = config.task_sampler_config.task_sampler_class(config)
        self.sampler.seed_task_sampling(int(result["selected_seed"]["seed_u32"]))
        self.sampler.set_pact_manifest_row(row)
        self.task = self.sampler.sample_task(house_index=int(row["scene_template_house_index"]))
        if self.task is None:
            raise RuntimeError("realized Pass-2 seed no longer samples")
        self.task.reset()
        self.model = self.task.env.current_model
        self.data = self.task.env.current_data
        groups = geom_groups(self.model)
        self.robot = {
            link: _physical_geoms(self.model, list(groups.get(link) or []))
            for link in LINKS
        }
        self.hand = _physical_geoms(
            self.model,
            [geom for name in ("link7", "hand", "left_finger", "right_finger") for geom in list(groups.get(name) or [])],
        )
        self.cup = _physical_geoms(self.model, _target_geoms(self.model))
        self.wrist_id = int(self.model.camera(WRIST_CAMERA_MJCF).id)
        self.by_slot = {str(item["slot"]): item for item in self.sampler._pact_clutter_objects}
        self.prop_addresses = {
            slot: self.sampler._free_joint_addresses(self.model, item["body"])
            for slot, item in self.by_slot.items()
            if item["slot_class"] == "prop"
        }

    def close(self) -> None:
        self._cleanup(
            task=self.task,
            policy=None,
            task_sampler=self.sampler,
            preloaded_policy=None,
            close_task_sampler=True,
        )
        shutil.rmtree(self.scratch, ignore_errors=True)

    def _pose(self, layout: dict[str, Any]) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, list[int]]:
        model, data = self.model, self.data
        first_qpos = np.asarray(self.steps[0]["qpos"], dtype=float)
        data.qpos[:] = first_qpos
        for item in self.by_slot.values():
            if item["slot_class"] == "mount":
                self.sampler._set_mocap_pose(self.task.env, item["body"], item["park_m"], [1, 0, 0, 0])
            else:
                self.sampler._set_free_pose(self.task.env, item["body"], item["park_m"], [1, 0, 0, 0])
        active_geoms: list[int] = []
        for placed in layout["objects"]:
            slot = str(placed["palette_slot"])
            item = self.by_slot[slot]
            quat = list(map(float, placed["quat_wxyz"]))
            setter = self.sampler._set_mocap_pose if item["slot_class"] == "mount" else self.sampler._set_free_pose
            setter(self.task.env, item["body"], [0.0, 0.0, 0.0], quat)
            mujoco.mj_forward(model, data)
            low, high = self.sampler._body_collision_aabb(model, data, item["body"])
            position = np.asarray(placed["center_m"], dtype=float) - (low + high) / 2.0
            setter(self.task.env, item["body"], position.tolist(), quat)
            root_id = int(model.body_rootid[int(model.body(item["body"]).id)])
            active_geoms.extend(
                geom_id
                for geom_id in range(int(model.ngeom))
                if int(model.body_rootid[int(model.geom_bodyid[geom_id])]) == root_id
                and (int(model.geom_contype[geom_id]) or int(model.geom_conaffinity[geom_id]))
            )
        mujoco.mj_forward(model, data)
        prop_qpos = {
            slot: data.qpos[qadr : qadr + 7].copy()
            for slot, (qadr, _dadr) in self.prop_addresses.items()
        }
        return prop_qpos, data.mocap_pos.copy(), data.mocap_quat.copy(), active_geoms

    def score(self, candidate: dict[str, Any]) -> dict[str, Any]:
        model, data = self.model, self.data
        prop_qpos, mocap_pos, mocap_quat, clutter = self._pose(candidate)
        by_link = {link: [] for link in LINKS}
        cups, hands, phases = [], [], []

        def restore(step: dict[str, Any]) -> None:
            data.qpos[:] = np.asarray(step["qpos"], dtype=float)
            for slot, values in prop_qpos.items():
                qadr, _dadr = self.prop_addresses[slot]
                data.qpos[qadr : qadr + 7] = values
            data.mocap_pos[:] = mocap_pos
            data.mocap_quat[:] = mocap_quat
            mujoco.mj_forward(model, data)

        for step in self.steps:
            restore(step)
            for link in LINKS:
                by_link[link].append(_true_distance(model, data, self.robot[link], clutter))
            cups.append(_true_distance(model, data, self.cup, clutter))
            hands.append(_true_distance(model, data, self.hand, clutter))
            phases.append(str(step.get("policy_phase") or "unknown"))
        arrays = {link: np.asarray(values) for link, values in by_link.items()}
        matrix = np.stack([arrays[link] for link in LINKS], axis=1)
        frame_min = matrix.min(axis=1)
        index = int(np.argmin(frame_min))
        minimums = {link: float(values.min()) for link, values in arrays.items()}
        cup_min = float(np.min(cups))
        restore(self.steps[index])
        visible_at_min = _any_visible(model, data, self.wrist_id, clutter)
        target_visible_run = target_visible_best = 0
        for step, phase in zip(self.steps, phases):
            if phase != "pregrasp":
                continue
            restore(step)
            target_visible_run = target_visible_run + 1 if _any_visible(
                model, data, self.wrist_id, self.cup
            ) else 0
            target_visible_best = max(target_visible_best, target_visible_run)
            if target_visible_best >= 5:
                break
        return {
            "min_clearance_by_link_m": minimums,
            "min_link_clearance_m": float(frame_min[index]),
            "min_cup_clearance_m": cup_min,
            "min_hand_clearance_m": float(np.min(hands)),
            "closest_robot_link": min(minimums, key=minimums.get),
            "phase_of_min_clearance": phases[index],
            "frames_link_clearance_lt_5cm": int(np.sum(frame_min < 0.05)),
            "frames_link_clearance_lt_10cm": int(np.sum(frame_min < 0.10)),
            "frames_link_clearance_lt_15cm": int(np.sum(frame_min < 0.15)),
            "n_distinct_links_exposed": int(sum(value < 0.10 for value in minimums.values())),
            "cup_is_closest_body": bool(cup_min < float(frame_min[index])),
            "visibility_at_min_link_clearance": bool(visible_at_min),
            "target_visibility_floor_min_consecutive_frames": target_visible_best,
        }


def _reject(candidate: dict[str, Any], score: dict[str, Any]) -> str | None:
    if score["min_link_clearance_m"] < 0.030:
        return "intersects_realized_swept_volume_plus_C"
    if score["min_hand_clearance_m"] <= 0.0:
        return "would_contact_hand"
    if score["min_cup_clearance_m"] <= 0.0:
        return "would_contact_carried_cup"
    if score["target_visibility_floor_min_consecutive_frames"] < 5:
        return "target_visibility_floor_violated"
    return support_reject(candidate)


def _score_family(job: dict[str, Any]) -> list[dict[str, Any]]:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("MLSPACES_ASSETS_DIR", "/root/prox_learning/assets")
    os.environ.pop("DISPLAY", None)
    context = Context(job["row"], job["record"])
    records = []
    try:
        for index, source in enumerate(job["candidates"]):
            candidate = {
                key: value
                for key, value in source.items()
                if key not in {
                    "score",
                    "quality",
                    "admitted",
                    "reject_reason",
                    "reference_track_role_index",
                }
            }
            score = context.score(candidate)
            reason = _reject(candidate, score)
            candidate["score"] = score
            candidate["admitted"] = reason is None
            candidate["reject_reason"] = reason
            candidate["quality"] = float(
                score["frames_link_clearance_lt_10cm"]
                + 20 * score["n_distinct_links_exposed"]
                + 20 * int(score["visibility_at_min_link_clearance"])
                - 500 * int(score["cup_is_closest_body"])
            )
            records.append(candidate)
            if (index + 1) % 48 == 0:
                print(
                    f"{job['family']}: rescored {index + 1}/{len(job['candidates'])}",
                    flush=True,
                )
    finally:
        context.close()
    return records


def main() -> int:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("MLSPACES_ASSETS_DIR", "/root/prox_learning/assets")
    os.environ.pop("DISPLAY", None)
    pass1 = json.loads(PASS1.read_text())
    pass2 = json.loads(PASS2.read_text())
    config = json.loads(PASS1_CONFIG.read_text())
    runs = json.loads(PASS2_RUNS.read_text())
    rows = {int(row["role_index"]): row for row in config["family_review_rows"]}
    realized_by_family = {record["family"]: record for record in runs["realized_results"]}
    jobs = [
        {
            "family": family,
            "row": rows[int(realized_by_family[family]["role_index"])],
            "record": realized_by_family[family],
            "candidates": [item for item in pass1["candidates"] if item["family"] == family],
        }
        for family in FAMILIES
    ]
    records = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=3,
        mp_context=multiprocessing.get_context("spawn"),
        max_tasks_per_child=1,
    ) as executor:
        futures = {executor.submit(_score_family, job): job["family"] for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            family = futures[future]
            family_records = future.result()
            records.extend(family_records)
            print(f"{family}: complete ({len(family_records)} candidates)", flush=True)
    records.sort(key=lambda item: int(item["candidate_id"]))
    admitted = [item for item in records if item["admitted"] and not item["score"]["cup_is_closest_body"]]
    availability = Counter((item["family"], item["intrusion_side"]) for item in admitted)
    missing = {(family, side): availability[(family, side)] for family in FAMILIES for side in ("left", "right") if availability[(family, side)] < 2}
    if missing:
        raise SystemExit(f"Pass 3 lacks quota candidates: {missing}")
    chosen, features = v8.farthest_point_select(admitted)
    for index, item in enumerate(chosen):
        item["layout_id"] = f"v8b_layout_{index:02d}"
    aggregate = {
        "cup_is_closest_body_fraction": float(np.mean([item["score"]["cup_is_closest_body"] for item in chosen])),
        "nonzero_visibility_at_min_fraction": float(np.mean([item["score"]["visibility_at_min_link_clearance"] for item in chosen])),
        "mean_distinct_links_exposed": float(np.mean([item["score"]["n_distinct_links_exposed"] for item in chosen])),
        "frames_link_clearance_lt_10cm": sum(item["score"]["frames_link_clearance_lt_10cm"] for item in chosen),
        "episodes_with_frames_link_clearance_lt_5cm": sum(item["score"]["frames_link_clearance_lt_5cm"] > 0 for item in chosen),
    }
    gates = {
        "cup_is_closest_body_fraction_le_0_25": aggregate["cup_is_closest_body_fraction"] <= 0.25,
        "nonzero_visibility_at_min_fraction_ge_1_3": aggregate["nonzero_visibility_at_min_fraction"] >= 1 / 3,
        "mean_distinct_links_exposed_ge_3": aggregate["mean_distinct_links_exposed"] >= 3.0,
        "frames_link_clearance_lt_10cm_ge_1852": aggregate["frames_link_clearance_lt_10cm"] >= 1852,
        "lt_5cm_in_at_least_half": aggregate["episodes_with_frames_link_clearance_lt_5cm"] >= 12,
        "objects_8_to_12": all(8 <= len(item["objects"]) <= 12 for item in chosen),
        "overhead_at_least_2": all(sum(float(obj["center_m"][2]) - float(obj["half_m"][2]) >= 0.95 for obj in item["objects"]) >= 2 for item in chosen),
        "category_max_2": all(max(Counter(obj["category"] for obj in item["objects"]).values()) <= 2 for item in chosen),
    }
    pass1_ids = {item["candidate_id"] for item in pass1["selected_layouts"]}
    selected_document = {"schema_version": "pact_place_v8b_selected_layouts_v1", "palette": pass1["selected_layouts"][0] and config["palette"], "layouts": chosen}
    selected_document["selected_layouts_sha256"] = sha256_payload(selected_document)
    analysis = {
        "schema_version": "pact_place_clutter_sweep_v8b_v1",
        "role": "b1b_b2b_realized_sweep_not_a_gate",
        "authorizes_gate": False,
        "authorizes_collection": False,
        "distance_instrument": "mujoco.mj_geomDistance_real_collision_geoms_to_robot_collision_geoms",
        "pass1": {"n_candidates": pass1["n_candidates"], "chosen_n": 24, "cup_is_closest_body_count": sum(item["score"]["cup_is_closest_body"] for item in pass1["selected_layouts"])},
        "pass2": pass2["aggregate"],
        "pass3": {"n_rescored": len(records), "n_admitted_link_primary": len(admitted), "chosen_n": 24, "original_24_survived": sum(item["candidate_id"] in pass1_ids for item in chosen), "aggregate": aggregate},
        "pass1_vs_pass3_cup_fraction_disagreement_gt_20pp": abs(
            float(np.mean([item["score"]["cup_is_closest_body"] for item in pass1["selected_layouts"]]))
            - aggregate["cup_is_closest_body_fraction"]
        ) > 0.20,
        "selection_rule": "quota_constrained_farthest_point_after_realized_true_geom_rescore",
        "min_pairwise_selected_layout_distance": v8.min_pairwise(features),
        "admission_gates": gates,
        "all_admission_gates_pass": all(gates.values()),
        "selected_layouts": chosen,
        "reject_counts": dict(Counter(item["reject_reason"] or "admitted" for item in records)),
        "candidates": records,
    }
    analysis["analysis_sha256"] = sha256_payload(analysis)
    write_json_atomic(OUTPUT, analysis)
    write_json_atomic(SELECTED_OUTPUT, selected_document)
    final_config = build_config(chosen, config["palette"], selected_document["selected_layouts_sha256"])
    final_config["schema_version"] = "pact_place_corridor_v8b"
    final_config["status"] = "pass3_selected_family_review_not_yet_run"
    final_config["selected_layouts_path"] = str(SELECTED_OUTPUT.relative_to(ROOT))
    final_config["config_sha256"] = sha256_payload({key: value for key, value in final_config.items() if key != "config_sha256"})
    write_json_atomic(CONFIG_OUTPUT, final_config)
    print(json.dumps({"pass3_aggregate": aggregate, "admission_gates": gates, "all_pass": all(gates.values())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
