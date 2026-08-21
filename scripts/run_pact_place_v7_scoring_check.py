#!/usr/bin/env python3
"""A0f scoring check: a link-vs-clutter contact must classify as clutter.

Place one parked pool box inside the measured swept volume after reset, run
one expert episode, and require an unclean row whose contact is attributed to
`clutter` for a robot link (not only the carried cup). The injected box is
not part of the lattice and is discarded with the process.
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
    V7_CLUTTER_POOL_SLOT_NAMES,
    V7_DESIGN_REVIEW_MASTER_SEED,
    clutter_jitters_for_seed,
    review_episode_id_for,
    review_row_seed,
    sha256_payload,
)
from run_pact_place_expert_screen import (  # noqa: E402
    _jsonable,
    _make_config,
    _protected_eval_processes,
    disallowed_initial_contacts,
    initial_robot_environment_contacts,
    retry_seed,
    write_json_atomic,
)

OUTPUT_DIR = (
    ROOT / "diagnostics_output/pact_place_corridor_v7_design_review/scoring_check"
)
SCENE_XML = (
    ROOT
    / "submodules/molmospaces/molmo_spaces/data_generation/custom_scenes"
    / "pact_place_corridor_v4.xml"
)
INJECT_BODY = "pact_clutter_15"
INJECT_CENTER_M = [0.68, 0.0, 1.15]
INJECT_HALF_M = [0.04, 0.04, 0.12]


def _scoring_row() -> dict[str, Any]:
    index = 99
    side = "left"
    seed_u32, seed_u64 = review_row_seed(index, V7_DESIGN_REVIEW_MASTER_SEED)
    clutter_x, clutter_y = clutter_jitters_for_seed(
        seed_u64, slot_names=V7_CLUTTER_POOL_SLOT_NAMES
    )
    row = {
        "role_index": index,
        "episode_id": review_episode_id_for(index, side, V7_DESIGN_REVIEW_MASTER_SEED),
        "intrusion_side": side,
        "panel_x_jitter_m": 0.0,
        "panel_face_jitter_m": 0.0,
        "clutter_x_jitter_m": clutter_x,
        "clutter_y_jitter_m": clutter_y,
        "scene_template_house_index": 1,
        "task_seed_u32": seed_u32,
        "task_seed_u64": seed_u64,
        "max_sampling_retries": 4,
    }
    row["row_sha256"] = sha256_payload(row)
    return row


def _link_name(pair: dict[str, Any]) -> str | None:
    names = (pair.get("body1"), pair.get("body2"), pair.get("root1"), pair.get("root2"))
    for name in names:
        text = str(name or "")
        for link in ("link7", "link6", "link5", "link4", "link3", "link2", "link1"):
            if f"fr3_{link}" in text:
                return link
    return None


def _has_cup(pair: dict[str, Any]) -> bool:
    blob = " ".join(
        str(pair.get(key, ""))
        for key in ("geom1", "geom2", "body1", "body2", "root1", "root2")
    )
    return "cavity_obj_" in blob


def _has_robot(pair: dict[str, Any]) -> bool:
    return str(pair.get("root1") or "").startswith("robot_0/") or str(
        pair.get("root2") or ""
    ).startswith("robot_0/")


def run_scoring_check() -> dict[str, Any]:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MLSPACES_ASSETS_DIR", str(ROOT / "assets"))
    os.environ["PACT_CONTACT_AUDIT_SUMMARY_ONLY"] = "0"
    os.environ.pop("DISPLAY", None)

    import mujoco
    from molmo_spaces.data_generation.pipeline import (
        ParallelRolloutRunner,
        cleanup_episode_resources,
        setup_policy,
    )
    from molmo_spaces.data_generation.runtime_compat import assert_supported_runtime
    from molmo_spaces.tasks.pact_place_contact_audit import classify_contact
    from molmo_spaces.tasks.task_sampler_errors import HouseInvalidForTask

    assert_supported_runtime(strict=True)
    row = _scoring_row()
    destination = OUTPUT_DIR / "result.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    retry_history: list[dict[str, Any]] = []
    task = policy = sampler = None
    try:
        config = _make_config(destination, scene_xml=SCENE_XML)
        selected_seed: dict[str, int] | None = None
        initial_reset_result = None
        for retry_index in range(int(row["max_sampling_retries"]) + 1):
            if retry_index == 0:
                seed_u32 = int(row["task_seed_u32"])
                seed_u64 = int(row["task_seed_u64"])
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
                from molmo_spaces.env.abstract_sensors import SensorSuite

                task._sensor_suite = SensorSuite(
                    [
                        task._sensor_suite.sensors[uuid]
                        for uuid in ("qpos", "tcp_pose")
                    ]
                )
                policy = setup_policy(config, task, None, None)
                initial_reset_result = task.reset()
                rejected = disallowed_initial_contacts(
                    initial_robot_environment_contacts(task.env)
                )
                if rejected:
                    first = rejected[0]
                    raise HouseInvalidForTask(
                        "initial_robot_environment_contact "
                        f"n={len(rejected)} "
                        f"{first.get('body1')} vs {first.get('body2')}"
                    )
            except Exception as error:  # noqa: BLE001
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
        if selected_seed is None or initial_reset_result is None or task is None:
            raise RuntimeError(f"scoring check failed to sample: {retry_history}")

        env = task.env
        parked = np.array(
            env.current_data.body(INJECT_BODY).xpos, dtype=float, copy=True
        )
        sampler._mocap_set(env, INJECT_BODY, INJECT_CENTER_M)
        geom = env.current_model.geom(f"{INJECT_BODY}_g")
        env.current_model.geom_size[int(geom.id)] = np.asarray(
            INJECT_HALF_M, dtype=float
        )
        mujoco.mj_forward(env.current_model, env.current_data)
        after = np.array(
            env.current_data.body(INJECT_BODY).xpos, dtype=float, copy=True
        )
        contacts_after_inject = initial_robot_environment_contacts(env)
        clutter_at_t0 = [
            pair
            for pair in contacts_after_inject
            if classify_contact(pair) == "clutter"
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
        info = policy.get_info()
        audit_summary = info["pact_contact_audit"]
        totals = audit_summary["contact_class_totals"]
        audit = policy._pact_place_contact_audit
        link_pairs = []
        cup_pairs = []
        first_link_pair = None
        for frame in audit._pairs_by_step:
            for pair in frame["pairs"]:
                if classify_contact(pair) != "clutter":
                    continue
                if _has_cup(pair):
                    cup_pairs.append(
                        {
                            "step": frame["step"],
                            "policy_phase": frame["policy_phase"],
                            "pair": pair,
                        }
                    )
                if _has_robot(pair) and _link_name(pair) is not None:
                    record = {
                        "step": frame["step"],
                        "policy_phase": frame["policy_phase"],
                        "link": _link_name(pair),
                        "pair": pair,
                    }
                    link_pairs.append(record)
                    if first_link_pair is None:
                        first_link_pair = record
        clean_success = bool(
            task_success
            and int(totals["hazard_bar"]) == 0
            and int(totals["other_environment"]) == 0
            and int(totals.get("clutter", 0)) == 0
        )
        passed = (
            not clean_success
            and int(totals.get("clutter", 0)) > 0
            and first_link_pair is not None
        )
        payload = {
            "schema_version": "pact_place_v7_scoring_check_v1",
            "role": "scoring_check_not_a_gate",
            "injected_body": INJECT_BODY,
            "injected_center_m": INJECT_CENTER_M,
            "injected_half_m": INJECT_HALF_M,
            "parked_before_inject_m": parked.tolist(),
            "posed_after_inject_m": after.tolist(),
            "deleted_after_process_exit": True,
            "not_part_of_lattice": True,
            "selected_seed": selected_seed,
            "retry_history": retry_history,
            "task_success": task_success,
            "clean_success": clean_success,
            "contact_class_totals": totals,
            "clutter_pairs_at_t0_after_inject": len(clutter_at_t0),
            "n_link_clutter_pairs": len(link_pairs),
            "n_cup_clutter_pairs": len(cup_pairs),
            "first_link_clutter_pair": _jsonable(first_link_pair),
            "passed": passed,
            "requirement": (
                "unclean episode with classify_contact == clutter for a "
                "robot_0/fr3_link* pair, not only cavity_obj_ vs clutter"
            ),
        }
        payload["scoring_check_sha256"] = sha256_payload(payload)
        write_json_atomic(destination, payload)
        write_json_atomic(OUTPUT_DIR / "scoring_check.json", payload)
        return payload
    finally:
        os.environ["PACT_CONTACT_AUDIT_SUMMARY_ONLY"] = "1"
        if task is not None or policy is not None or sampler is not None:
            cleanup_episode_resources(
                task=task,
                policy=policy,
                task_sampler=sampler,
                preloaded_policy=None,
                close_task_sampler=sampler is not None,
            )


def main() -> int:
    protected = _protected_eval_processes()
    if protected:
        raise SystemExit(f"protected confirmatory evaluation is active: {protected}")
    payload = run_scoring_check()
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    if not payload.get("passed"):
        raise SystemExit("scoring check failed: link-vs-clutter was not classified as clutter")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
