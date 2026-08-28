#!/usr/bin/env python3
"""Root-cause diagnostic for the failed V10.2 Step-0 contact-parity fixture.

Step 0 item 7 requires that a deliberate stem/robot overlap be observed by both
``data.contact`` and the contact classifier. It failed. This script isolates
why, on one frozen cell, and records the witness.

It is a diagnostic only. It authorizes nothing, writes no approval, and does
not modify ``pose_assembly_geoms`` or any V10/V10.1 artifact. It is not part of
the V10.2 implementation hash.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "scripts", ROOT / "submodules" / "molmospaces"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from pact_geom_distance import true_distance  # noqa: E402
from pact_place_v10_compound_pendant_contract import (  # noqa: E402
    PENDANT_BODY,
    SCENE_XML_RELATIVE,
)
from pact_place_v10_environment import _prepare_with_scene  # noqa: E402
from pact_place_v10_exact import verify_v99_inputs  # noqa: E402
from pact_place_v10_geometry import active_components  # noqa: E402
from pact_place_v10_runtime import establish_v10_runtime_env, write_immutable  # noqa: E402
from pact_place_v102_raised_pendant_contract import (  # noqa: E402
    CONTRACT_VERSION,
    SAMPLER_CLASS,
    empty_authorization,
    frozen_assembly,
    frozen_route_for_side,
)
from pact_place_v99_exact import snapshot_jobs_from_reconstruction  # noqa: E402
from reconstruct_pact_place_v99_baseline import cleanup_task  # noqa: E402
from run_pact_place_v7_replay_videos import apply_recorded_qpos  # noqa: E402

DEFAULT_OUTPUT = (
    ROOT / "diagnostics_output" / "pact_place_v102_preflight" / "contact_parity_root_cause.json"
)
SCENE_XML = ROOT / SCENE_XML_RELATIVE
PROBE_GEOM = "robot_0/fr3_link5_collision"


def _stem_pairs(model, data, stem_gid: int) -> list[dict[str, Any]]:
    out = []
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        if stem_gid not in (geom1, geom2):
            continue
        out.append(
            {
                "geom1": model.geom(geom1).name,
                "geom2": model.geom(geom2).name,
                "distance_m": float(contact.dist),
            }
        )
    return out


def diagnose() -> dict[str, Any]:
    import mujoco

    reconstruction, _snapshot, cells = verify_v99_inputs()
    cells = sorted(cells, key=lambda item: int(item["role_index"]))
    jobs = {
        int(json.loads((Path(job["row_dir"]) / "result.json").read_text())["role_index"]): job
        for job in snapshot_jobs_from_reconstruction(reconstruction)
    }
    cell = cells[0]
    job = jobs[int(cell["role_index"])]
    assembly = frozen_assembly()
    row = dict(job["manifest_row"])
    row["sampler_class"] = SAMPLER_CLASS
    row["pact_v10_pendant_parked"] = False
    row["pact_v10_pendant_assembly"] = assembly
    row["pact_v10_route"] = frozen_route_for_side(str(cell["intrusion_side"]))
    trajectory = json.loads((Path(job["row_dir"]) / "trajectory.json").read_text())
    task = sampler = scratch = None
    try:
        task, sampler, scratch = _prepare_with_scene(
            row,
            seed_u32=(job.get("selected_seed") or {}).get("seed_u32"),
            scene_xml=SCENE_XML,
            sampler_class=SAMPLER_CLASS,
        )
        apply_recorded_qpos(task.env, trajectory["steps"][0]["qpos"])
        model, data = task.env.current_model, task.env.current_data
        mujoco.mj_forward(model, data)
        baseline_ncon = int(data.ncon)
        probe = int(model.geom(PROBE_GEOM).id)
        stem = next(
            item for item in active_components(assembly) if item["name"] == "stem_0"
        )
        stem_gid = int(model.geom(str(stem["geom"])).id)
        body_id = int(model.body(PENDANT_BODY).id)
        bvh_adr = int(model.body_bvhadr[body_id])
        bvh_num = int(model.body_bvhnum[body_id])
        compiled = {
            "pendant_body": PENDANT_BODY,
            "body_bvhadr": bvh_adr,
            "body_bvhnum": bvh_num,
            "stale_bvh_aabb_row_0": [
                float(value) for value in model.bvh_aabb[bvh_adr]
            ]
            if bvh_adr >= 0
            else None,
            "stale_geom_aabb": [float(value) for value in model.geom_aabb[stem_gid]],
            "stale_geom_rbound_m": float(model.geom_rbound[stem_gid]),
            "posed_geom_size_m": [float(value) for value in model.geom_size[stem_gid]],
            "stem_contype": int(model.geom_contype[stem_gid]),
            "stem_conaffinity": int(model.geom_conaffinity[stem_gid]),
            "probe_geom": PROBE_GEOM,
            "probe_geom_type": int(model.geom_type[probe]),
            "probe_contype": int(model.geom_contype[probe]),
            "probe_conaffinity": int(model.geom_conaffinity[probe]),
            "collision_compatible_robot_pair": bool(
                (int(model.geom_contype[stem_gid]) & int(model.geom_conaffinity[probe]))
                or (int(model.geom_contype[probe]) & int(model.geom_conaffinity[stem_gid]))
            ),
        }
        saved_pos = np.asarray(model.geom_pos[stem_gid], dtype=float).copy()
        saved_aabb = np.asarray(model.geom_aabb[stem_gid], dtype=float).copy()
        saved_rbound = float(model.geom_rbound[stem_gid])
        saved_bvh = (
            np.asarray(model.bvh_aabb[bvh_adr : bvh_adr + bvh_num], dtype=float).copy()
            if bvh_adr >= 0
            else None
        )
        origin = np.asarray(data.geom_xpos[probe], dtype=float).copy()
        sweep: list[dict[str, Any]] = []
        try:
            for axis in (0, 1, 2):
                for delta in (0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10):
                    offset = np.zeros(3)
                    offset[axis] = delta
                    model.geom_pos[stem_gid] = origin + offset
                    mujoco.mj_forward(model, data)
                    distance = float(true_distance(model, data, [probe], [stem_gid]))
                    pairs = _stem_pairs(model, data, stem_gid)
                    sweep.append(
                        {
                            "axis": axis,
                            "offset_m": float(delta),
                            "gjk_distance_m": distance,
                            "penetrating": bool(distance < 0.0),
                            "n_data_contact_pairs": len(pairs),
                            "pairs": pairs[:2],
                        }
                    )
            penetrating = [item for item in sweep if item["penetrating"]]
            observed = [item for item in penetrating if item["n_data_contact_pairs"]]
            # Control: refresh the stale broadphase bounds only, same pose.
            control_offset = np.zeros(3)
            control_offset[1] = 0.06
            model.geom_pos[stem_gid] = origin + control_offset
            mujoco.mj_forward(model, data)
            before_distance = float(true_distance(model, data, [probe], [stem_gid]))
            before_pairs = _stem_pairs(model, data, stem_gid)
            model.geom_aabb[stem_gid] = np.concatenate(
                [np.zeros(3), np.asarray(model.geom_size[stem_gid], dtype=float)]
            )
            model.geom_rbound[stem_gid] = float(
                np.linalg.norm(np.asarray(model.geom_size[stem_gid], dtype=float))
            )
            if bvh_adr >= 0:
                for index in range(bvh_num):
                    model.bvh_aabb[bvh_adr + index] = np.concatenate(
                        [
                            origin + control_offset,
                            3.0 * np.asarray(model.geom_size[stem_gid], dtype=float),
                        ]
                    )
            mujoco.mj_forward(model, data)
            after_distance = float(true_distance(model, data, [probe], [stem_gid]))
            after_pairs = _stem_pairs(model, data, stem_gid)
        finally:
            model.geom_pos[stem_gid] = saved_pos
            model.geom_aabb[stem_gid] = saved_aabb
            model.geom_rbound[stem_gid] = saved_rbound
            if saved_bvh is not None:
                model.bvh_aabb[bvh_adr : bvh_adr + bvh_num] = saved_bvh
            mujoco.mj_forward(model, data)
        restored_ncon = int(data.ncon)
        return {
            "schema_version": "pact_place_v102_contact_parity_root_cause_v1",
            "contract_version": CONTRACT_VERSION,
            "role_index": int(cell["role_index"]),
            "family": cell.get("family"),
            "intrusion_side": cell.get("intrusion_side"),
            "baseline_scene_ncon": baseline_ncon,
            "restored_scene_ncon": restored_ncon,
            "scene_contacts_work_generally": bool(baseline_ncon > 0),
            "compiled_state": compiled,
            "penetration_sweep": sweep,
            "n_penetrating_poses": len(penetrating),
            "n_penetrating_poses_seen_by_data_contact": len(observed),
            "control_broadphase_refresh": {
                "offset_m": [float(value) for value in control_offset],
                "before_refresh_gjk_distance_m": before_distance,
                "before_refresh_pairs": before_pairs[:2],
                "after_refresh_gjk_distance_m": after_distance,
                "after_refresh_pairs": after_pairs[:2],
                "refresh_makes_contact_visible": bool(
                    not before_pairs and after_pairs
                ),
            },
            "root_cause": (
                "pact_place_v10_scene.pose_assembly_geoms writes model.geom_pos and "
                "model.geom_size at runtime but leaves model.geom_aabb, "
                "model.geom_rbound and the pendant body's model.bvh_aabb rows at their "
                "compile-time 1 mm placeholder values, so MuJoCo's broadphase never "
                "proposes any pendant/robot pair and no pendant contact can be "
                "generated at any penetration depth."
            ),
            "consequence": (
                "Recorded zero mounted_fixture contact for the runtime-posed V10 "
                "pendant is not evidence of clearance in any pack that used this "
                "posing path. Repairing it would change V10/V10.1 behaviour and is "
                "not authorized by the V10.2 plan."
            ),
            "authorizes_repair": False,
            **empty_authorization(),
        }
    finally:
        cleanup_task(task, sampler, scratch)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    establish_v10_runtime_env()
    document = diagnose()
    digest = write_immutable(args.output.resolve(), document)
    print(
        json.dumps(
            {
                "n_penetrating_poses": document["n_penetrating_poses"],
                "n_penetrating_poses_seen_by_data_contact": document[
                    "n_penetrating_poses_seen_by_data_contact"
                ],
                "refresh_makes_contact_visible": document[
                    "control_broadphase_refresh"
                ]["refresh_makes_contact_visible"],
                "artifact_sha256": digest,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
