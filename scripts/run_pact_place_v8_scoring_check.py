#!/usr/bin/env python3
"""B4 diagnostic: prove V8 link-contact and topple outcomes are unclean.

Both constructions are process-local.  The deliberate collision layout is not
part of the frozen 24-layout selection, and no object is added to the V5 scene.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MOLMO = ROOT / "submodules" / "molmospaces"
for search_path in (ROOT / "scripts", MOLMO):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from pact_place_corridor_contract import (  # noqa: E402
    load_v8_contract,
    retry_seed,
    sha256_payload,
)
from run_pact_place_expert_screen import (  # noqa: E402
    _make_config,
    _protected_eval_processes,
    _jsonable,
    disallowed_initial_contacts,
    initial_robot_environment_contacts,
    write_json_atomic,
)

CONFIG_PATH = ROOT / "configs" / "pact_place_corridor_v8.json"
OUTPUT_DIR = ROOT / "diagnostics_output" / "pact_place_corridor_v8_scoring_check"
SCENE_XML = (
    MOLMO
    / "molmo_spaces/data_generation/custom_scenes/pact_place_corridor_v5.xml"
)


def _link_name(pair: dict[str, Any]) -> str | None:
    blob = " ".join(
        str(pair.get(key, ""))
        for key in ("geom1", "geom2", "body1", "body2", "root1", "root2")
    )
    for link in ("link7", "link6", "link5", "link4", "link3", "link2", "link1"):
        if f"fr3_{link}" in blob:
            return link
    return None


def _has_clutter(pair: dict[str, Any]) -> bool:
    return "pact_clutter_" in " ".join(str(value) for value in pair.values())


def _find_link_contacts(result: dict[str, Any]) -> list[dict[str, Any]]:
    contacts: list[dict[str, Any]] = []
    for frame in result.get("contact_audit", {}).get("contact_frames", []):
        for pair in frame.get("pairs", []):
            link = _link_name(pair)
            if link is None or not _has_clutter(pair):
                continue
            contacts.append(
                {
                    "step": frame["step"],
                    "policy_phase": frame["policy_phase"],
                    "link": link,
                    "pair": pair,
                }
            )
    return contacts


def _construct_scoring_events(contract: dict[str, Any]) -> dict[str, Any]:
    """Construct a topple, restore it, then inject a real object into link6."""
    import mujoco
    from molmo_spaces.data_generation.pipeline import (
        ParallelRolloutRunner,
        cleanup_episode_resources,
        setup_policy,
    )
    from molmo_spaces.env.abstract_sensors import SensorSuite
    from molmo_spaces.tasks.pact_place_contact_audit import classify_contact
    from molmo_spaces.tasks.task_sampler_errors import HouseInvalidForTask

    row = dict(contract["family_review_rows"][0])
    retry_history: list[dict[str, Any]] = []
    task = policy = sampler = None
    selected_seed = None
    initial_reset_result = None
    try:
        config = _make_config(OUTPUT_DIR / "topple.json", scene_xml=SCENE_XML)
        for retry_index in range(int(row["max_sampling_retries"]) + 1):
            if retry_index == 0:
                seed_u32, seed_u64 = int(row["task_seed_u32"]), int(row["task_seed_u64"])
            else:
                seed_u32, seed_u64 = retry_seed(row, retry_index)
            seed = {"seed_u32": seed_u32, "seed_u64": seed_u64}
            sampler = config.task_sampler_config.task_sampler_class(config)
            sampler.seed_task_sampling(seed_u32)
            sampler.set_pact_manifest_row(row)
            try:
                task = sampler.sample_task(
                    house_index=int(row["scene_template_house_index"])
                )
                if task is None:
                    raise HouseInvalidForTask("sample_task returned None")
                task._sensor_suite = SensorSuite(
                    [task._sensor_suite.sensors[key] for key in ("qpos", "tcp_pose")]
                )
                policy = setup_policy(config, task, None, None)
                initial_reset_result = task.reset()
                rejected = disallowed_initial_contacts(
                    initial_robot_environment_contacts(task.env)
                )
                if rejected:
                    raise HouseInvalidForTask(f"initial disallowed contacts: {rejected[0]}")
            except Exception as error:  # noqa: BLE001 - pre-boundary retry diagnostic
                retry_history.append(
                    {
                        "retry_index": retry_index,
                        "seed": seed,
                        "reason": f"pre_boundary:{type(error).__name__}:{error}",
                    }
                )
                cleanup_episode_resources(
                    task=task,
                    policy=policy,
                    task_sampler=sampler,
                    preloaded_policy=None,
                    close_task_sampler=True,
                )
                task = policy = sampler = None
                continue
            selected_seed = seed
            break
        if (
            selected_seed is None
            or task is None
            or policy is None
            or initial_reset_result is None
        ):
            raise RuntimeError(f"B4 check failed to sample: {retry_history}")

        settle = task.scene_params["pact_clutter_settle"]
        if len(settle["objects"]) < 2:
            raise RuntimeError("B4 selected layout must contain at least two objects")
        topple_baseline = settle["objects"][1]
        topple_body = str(topple_baseline["body"])
        model, data = task.env.current_model, task.env.current_data
        qpos_before_constructions = np.asarray(data.qpos, dtype=float).copy()
        qvel_before_constructions = np.asarray(data.qvel, dtype=float).copy()
        qadr, dadr = sampler._free_joint_addresses(model, topple_body)
        before_qpos = np.asarray(data.qpos[qadr : qadr + 7], dtype=float).copy()
        quarter_turn = np.asarray([np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0], dtype=float)
        toppled_quat = np.zeros(4, dtype=float)
        mujoco.mju_mulQuat(toppled_quat, quarter_turn, before_qpos[3:7])
        data.qpos[qadr + 3 : qadr + 7] = toppled_quat
        data.qvel[dadr : dadr + 6] = 0.0
        mujoco.mj_forward(model, data)
        policy._update_clutter_stability()
        events = _jsonable(policy._pact_clutter_stability_events)
        topple_passed = bool(
            events
            and events[0].get("classification") == "other_environment"
            and float(events[0].get("rotation_angle_rad", 0.0)) > np.deg2rad(25.0)
        )
        topple = {
            "construction": "90_degree_free_body_rotation_after_clean_reset",
            "body": topple_body,
            "selected_seed": selected_seed,
            "retry_history": retry_history,
            "before_qpos": before_qpos.tolist(),
            "after_qpos": np.asarray(data.qpos[qadr : qadr + 7], dtype=float).tolist(),
            "events": events,
            "would_force_clean_success_false": bool(events),
            "passed": topple_passed,
        }

        # Restore the accepted step-0 state exactly.  The topple was a scoring
        # construction, not an episode.  Clear its policy ledger before the
        # independent deliberate-contact episode below.
        data.qpos[:] = qpos_before_constructions
        data.qvel[:] = qvel_before_constructions
        mujoco.mj_forward(model, data)
        policy._pact_clutter_stability_events.clear()
        policy._pact_clutter_stability_bodies.clear()

        link_baseline = settle["objects"][0]
        link_body = str(link_baseline["body"])
        link_qadr, link_dadr = sampler._free_joint_addresses(model, link_body)
        link6_id = int(model.body("robot_0/fr3_link6").id)
        desired_mesh_center = np.asarray(data.xpos[link6_id], dtype=float).copy()
        original_quat = np.asarray(data.qpos[link_qadr + 3 : link_qadr + 7], dtype=float)
        sampler._set_free_pose(task.env, link_body, [0.0, 0.0, 0.0], original_quat.tolist())
        mujoco.mj_forward(model, data)
        local_low, local_high = sampler._body_collision_aabb(model, data, link_body)
        root_position = desired_mesh_center - (local_low + local_high) / 2.0
        sampler._set_free_pose(
            task.env, link_body, root_position.tolist(), original_quat.tolist()
        )
        data.qvel[link_dadr : link_dadr + 6] = 0.0
        mujoco.mj_forward(model, data)
        injected_pairs = [
            pair
            for pair in initial_robot_environment_contacts(task.env)
            if classify_contact(pair) == "clutter" and _link_name(pair) is not None
        ]
        task_success = bool(
            ParallelRolloutRunner.run_single_rollout(
                episode_seed=int(selected_seed["seed_u64"]),
                task=task,
                policy=policy,
                end_on_success=False,
                initial_reset_result=initial_reset_result,
            )
        )
        info = _jsonable(policy.get_info())
        audit = info["pact_contact_audit"]
        totals = audit["contact_class_totals"]
        link_contacts = _find_link_contacts({"contact_audit": audit})
        stability_events = list(info.get("clutter_stability_events") or [])
        clean_success = bool(
            task_success
            and int(totals.get("hazard_bar", 0)) == 0
            and int(totals.get("other_environment", 0)) == 0
            and int(totals.get("clutter", 0)) == 0
            and not stability_events
        )
        link_contact = {
            "construction": "real_pickupable_free_body_centered_on_link6_after_accepted_reset",
            "body": link_body,
            "desired_mesh_center_m": desired_mesh_center.tolist(),
            "root_position_m": root_position.tolist(),
            "clutter_link_pairs_immediately_after_injection": _jsonable(injected_pairs),
            "task_success": task_success,
            "clean_success": clean_success,
            "contact_class_totals": totals,
            "n_link_clutter_contacts": len(link_contacts),
            "first_link_clutter_contact": _jsonable(
                link_contacts[0] if link_contacts else None
            ),
            "clutter_stability_events": stability_events,
            "passed": bool(
                not clean_success
                and int(totals.get("clutter", 0)) > 0
                and link_contacts
                and injected_pairs
            ),
        }
        return {"topple": topple, "link_contact": link_contact}
    finally:
        if task is not None or policy is not None or sampler is not None:
            cleanup_episode_resources(
                task=task,
                policy=policy,
                task_sampler=sampler,
                preloaded_policy=None,
                close_task_sampler=sampler is not None,
            )


def run_scoring_check() -> dict[str, Any]:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MLSPACES_ASSETS_DIR", str(ROOT / "assets"))
    os.environ["PACT_CONTACT_AUDIT_SUMMARY_ONLY"] = "0"
    os.environ.pop("DISPLAY", None)

    contract = load_v8_contract(CONFIG_PATH)
    constructions = _construct_scoring_events(contract)
    link_contact = constructions["link_contact"]
    topple = constructions["topple"]
    payload = {
        "schema_version": "pact_place_v8_scoring_check_v1",
        "role": "scoring_check_not_a_gate",
        "production_layout_selection_unchanged": True,
        "deliberate_objects_deleted_after_process_exit": True,
        "link_contact": link_contact,
        "topple": topple,
        "passed": bool(link_contact["passed"] and topple["passed"]),
    }
    payload["scoring_check_sha256"] = sha256_payload(payload)
    write_json_atomic(OUTPUT_DIR / "scoring_check.json", payload)
    return payload


def main() -> int:
    protected = _protected_eval_processes()
    if protected:
        raise SystemExit(f"protected confirmatory evaluation is active: {protected}")
    payload = run_scoring_check()
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit("B4 scoring check failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
