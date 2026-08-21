#!/usr/bin/env python3
"""B4b: prove a robot contact with a mocap mount scores as unclean clutter."""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MOLMO = ROOT / "submodules" / "molmospaces"
for path in (ROOT / "scripts", MOLMO):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pact_place_corridor_contract import sha256_payload  # noqa: E402
from run_pact_place_expert_screen import (  # noqa: E402
    _make_config,
    initial_robot_environment_contacts,
    write_json_atomic,
)

CONFIG = ROOT / "configs/pact_place_corridor_v8b_pass1.json"
RUNS = ROOT / "diagnostics_output/pact_place_corridor_v8b_pass2c/realized_runs.json"
OUTPUT = ROOT / "diagnostics_output/pact_place_corridor_v8b_mount_scoring_check/scoring_check.json"
SCENE = MOLMO / "molmo_spaces/data_generation/custom_scenes/pact_place_corridor_v5.xml"


def main() -> int:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("MLSPACES_ASSETS_DIR", "/root/prox_learning/assets")
    os.environ["PACT_CONTACT_AUDIT_SUMMARY_ONLY"] = "0"
    from molmo_spaces.data_generation.pipeline import cleanup_episode_resources
    from molmo_spaces.tasks.pact_place_contact_audit import (
        PactPlaceContactAudit,
        classify_contact,
    )

    config = json.loads(CONFIG.read_text())
    runs = json.loads(RUNS.read_text())
    realized = next(item for item in runs["realized_results"] if item["family"] == "F5_overhead_elbow")
    row = next(item for item in config["family_review_rows"] if int(item["role_index"]) == int(realized["role_index"]))
    result = json.loads((ROOT / realized["result_path"]).read_text())
    scratch = Path(tempfile.mkdtemp(prefix="pact_place_v8b_mount_b4_"))
    task = sampler = None
    try:
        runtime = _make_config(scratch / "result.json", scene_xml=SCENE)
        sampler = runtime.task_sampler_config.task_sampler_class(runtime)
        sampler.seed_task_sampling(int(result["selected_seed"]["seed_u32"]))
        sampler.set_pact_manifest_row(row)
        task = sampler.sample_task(house_index=int(row["scene_template_house_index"]))
        if task is None:
            raise RuntimeError("F5 realized seed no longer samples")
        task.reset()
        mount = task.scene_params["pact_clutter_settle"]["mounts"][0]
        body = str(mount["body"])
        model, data = task.env.current_model, task.env.current_data
        body_id = int(model.body(body).id)
        if int(model.body_mocapid[body_id]) < 0 or int(model.body_jntadr[body_id]) >= 0:
            raise RuntimeError("B4b body is not a jointless mocap mount")
        layout = mount["layout"]
        quat = list(map(float, layout["quat_wxyz"]))
        desired_center = np.asarray(data.xpos[int(model.body("robot_0/fr3_link6").id)], dtype=float)
        sampler._set_mocap_pose(task.env, body, [0.0, 0.0, 0.0], quat)
        mujoco.mj_forward(model, data)
        low, high = sampler._body_collision_aabb(model, data, body)
        root_position = desired_center - (low + high) / 2.0
        sampler._set_mocap_pose(task.env, body, root_position.tolist(), quat)
        mujoco.mj_forward(model, data)
        immediate = [
            pair
            for pair in initial_robot_environment_contacts(task.env)
            if classify_contact(pair) == "clutter"
        ]
        audit = PactPlaceContactAudit()
        audit.set_phase("inbound", "b4b_deliberate_mount_contact")
        audit.observe(task.env, 0)
        summary = audit.summary()
        clean_success = bool(
            True
            and int(summary["contact_class_totals"].get("hazard_bar", 0)) == 0
            and int(summary["contact_class_totals"].get("other_environment", 0)) == 0
            and int(summary["contact_class_totals"].get("clutter", 0)) == 0
        )
        passed = bool(immediate and summary["contact_class_totals"]["clutter"] > 0 and not clean_success)
        report = {
            "schema_version": "pact_place_v8b_mount_scoring_check_v1",
            "role": "mount_scoring_check_not_a_gate",
            "authorizes_gate": False,
            "construction": "jointless_mocap_mount_centered_on_link6_after_accepted_reset",
            "body": body,
            "body_mocap_id": int(model.body_mocapid[body_id]),
            "body_joint_address": int(model.body_jntadr[body_id]),
            "desired_collision_center_m": desired_center.tolist(),
            "root_position_m": root_position.tolist(),
            "immediate_clutter_pairs": immediate,
            "contact_audit": summary,
            "clean_success": clean_success,
            "passed": passed,
        }
        report["scoring_check_sha256"] = sha256_payload(report)
        write_json_atomic(OUTPUT, report)
        print(json.dumps({"passed": passed, "clutter_contacts": summary["contact_class_totals"]["clutter"], "clean_success": clean_success}, indent=2))
        if not passed:
            raise SystemExit("mocap mount scoring check failed")
    finally:
        cleanup_episode_resources(task=task, policy=None, task_sampler=sampler, preloaded_policy=None, close_task_sampler=sampler is not None)
        shutil.rmtree(scratch, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
