#!/usr/bin/env python3
"""Characterise the clearance-metric defect inherited from the audited adapter.

``eval_act_obstacle_safety._minimum_environment_distance`` takes a minimum of
``mujoco.mj_geomDistance`` over every robot x environment geom pair. This probe measures
that call on the compiled development scene, at the reset state, and shows that one robot
geom returns exactly 0.0 while every geometrically comparable neighbour returns a
sensible distance. Because the adapter takes a minimum, that single geom pins the
reported clearance at <= 0 on every frame of every rollout, in every condition.

Consequence: ``minimum_environment_distance_m`` and ``minimum_clearance_m`` are usable as
"deepest penetration observed" but NOT as a clearance margin, and any
hazard-only distance built the same way is unusable. Safety claims in this task therefore
rest on contact classification from ``data.contact``, which uses MuJoCo's real collision
pipeline and is unaffected.

Reset-state only: no policy, no rollout, no stepping.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.pop("DISPLAY", None)

ROOT = Path(__file__).resolve().parents[1]
for extra in (str(ROOT / "scripts"), str(ROOT / "submodules" / "act")):
    if extra not in sys.path:
        sys.path.insert(0, extra)

import mujoco

GEOM_TYPES = {0: "PLANE", 1: "HFIELD", 2: "SPHERE", 3: "CAPSULE", 4: "ELLIPSOID",
              5: "CYLINDER", 6: "BOX", 7: "MESH", 8: "SDF"}


def canonical_hash(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--development-manifest", required=True, type=Path)
    ap.add_argument("--collection-manifest", required=True, type=Path)
    ap.add_argument("--candidate", type=int, default=106)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    from molmo_spaces.data_generation.config.object_manipulation_datagen_configs import (
        FrankaSkinHybridObstacleManifestV2Config,
    )
    from molmo_spaces.data_generation.episode_manifest import install_row_seed_contract
    from molmo_spaces.data_generation.manifest_runner import (
        reset_episode_scoped_sampler_state,
    )

    dev = json.loads(args.development_manifest.read_text())
    coll = json.loads(args.collection_manifest.read_text())
    row = next(r for r in dev["rows"] if r["candidate_index"] == args.candidate)
    coll_row = next(r for r in coll["rows"] if r["episode_id"] == row["episode_id"])

    config = FrankaSkinHybridObstacleManifestV2Config()
    config.task_horizon = 5
    sampler = config.task_sampler_config.task_sampler_class(config)
    reset_episode_scoped_sampler_state(sampler)
    install_row_seed_contract(coll_row, int(row["accepted_retry_index"]), task_sampler=sampler)
    sampler.set_manifest_row(coll_row, int(row["accepted_retry_index"]))
    task = sampler.sample_task(house_index=coll_row["scene_template_house_index"])
    model, data = task.env.current_model, task.env.current_data

    hazard_body = int(model.body("protr_s").id)
    hazard_geoms = [g for g in range(model.ngeom)
                    if int(model.geom_bodyid[g]) == hazard_body]
    robot_geoms = []
    for geom in range(model.ngeom):
        if not (model.geom_contype[geom] or model.geom_conaffinity[geom]):
            continue
        root = model.body(int(model.body_rootid[int(model.geom_bodyid[geom])])).name or ""
        if root.startswith("robot_0/"):
            robot_geoms.append(geom)

    from_to = np.zeros(6, dtype=np.float64)
    measurements = []
    for geom in robot_geoms:
        distance = float(mujoco.mj_geomDistance(model, data, geom, hazard_geoms[0],
                                                1.0, from_to))
        measurements.append({
            "geom_id": geom,
            "name": model.geom(geom).name or "",
            "type": GEOM_TYPES.get(int(model.geom_type[geom]), "?"),
            "mj_geomDistance_m": distance,
            "centre_to_centre_m": float(np.linalg.norm(
                data.geom_xpos[geom] - data.geom_xpos[hazard_geoms[0]])),
        })

    zeros = [m for m in measurements if m["mj_geomDistance_m"] == 0.0]
    nonzero = [m for m in measurements if m["mj_geomDistance_m"] != 0.0]

    # control: the same call on a synthetic box pair, to show distmax semantics are fine
    control_model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody>'
        '<body pos="0 0 0"><geom type="box" size=".05 .05 .05"/></body>'
        '<body pos="5 0 0"><geom type="box" size=".05 .05 .05"/></body>'
        '<body pos="0.3 0 0"><geom type="box" size=".05 .05 .05"/></body>'
        '</worldbody></mujoco>')
    control_data = mujoco.MjData(control_model)
    mujoco.mj_forward(control_model, control_data)
    control = {
        "far_pair_5m_distmax_1m": float(mujoco.mj_geomDistance(
            control_model, control_data, 0, 1, 1.0, from_to)),
        "near_pair_0p3m_distmax_1m": float(mujoco.mj_geomDistance(
            control_model, control_data, 0, 2, 1.0, from_to)),
        "interpretation": ("mj_geomDistance returns distmax when the pair is farther than "
                           "distmax, and the true surface distance otherwise; it does not "
                           "return 0 for distant pairs"),
    }

    report = {
        "schema": "hybrid_obstacle_geom_distance_defect_v1",
        "mujoco_version": mujoco.__version__,
        "candidate_index": args.candidate,
        "episode_id": row["episode_id"],
        "state": "reset state; no policy, no rollout, no stepping",
        "hazard_geom": {
            "id": hazard_geoms[0],
            "name": model.geom(hazard_geoms[0]).name or "",
            "type": GEOM_TYPES.get(int(model.geom_type[hazard_geoms[0]]), "?"),
            "half_extents_m": model.geom_size[hazard_geoms[0]].tolist(),
            "world_position": data.geom_xpos[hazard_geoms[0]].tolist(),
        },
        "robot_collision_geoms": len(robot_geoms),
        "measurements": measurements,
        "geoms_returning_exactly_zero": [m["name"] or m["geom_id"] for m in zeros],
        "zero_count": len(zeros),
        "distance_control_experiment": control,
        "finding": (
            f"{len(zeros)} of {len(robot_geoms)} robot collision geoms return exactly 0.0 "
            f"from the hazard box at the reset state. The offender is "
            f"{zeros[0]['name'] if zeros else 'n/a'}, whose centre is "
            f"{zeros[0]['centre_to_centre_m']:.3f} m from the bar, while neighbouring geoms "
            f"at comparable ranges return "
            f"{min(m['mj_geomDistance_m'] for m in nonzero):.3f}-"
            f"{max(m['mj_geomDistance_m'] for m in nonzero):.3f} m. The control experiment "
            f"shows the call's distmax semantics are correct, so the zero is a per-geom "
            f"failure of the distance routine, not a far-field convention."
            if zeros else "no geom returned exactly 0.0"),
        "consequence": (
            "eval_act_obstacle_safety._minimum_environment_distance takes a MINIMUM over "
            "these pairs, so this one geom pins minimum_environment_distance_m -- and hence "
            "episode minimum_clearance_m -- at <= 0 on every frame of every rollout in every "
            "condition, including the raw-head development task. Those fields remain "
            "meaningful as 'deepest penetration observed' (the minimum is still taken over "
            "genuine negative distances) but are NOT a clearance margin, and a hazard-only "
            "distance built the same way is unusable."),
        "mitigation": (
            "Safety claims in this task rest on contact classification from data.contact, "
            "which uses MuJoCo's real collision pipeline and is unaffected. The adapter is "
            "not modified: it is the audited file."),
    }
    report["report_sha256"] = canonical_hash(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")

    print(f"robot collision geoms : {len(robot_geoms)}")
    print(f"returning exactly 0.0 : {len(zeros)} -> "
          f"{[m['name'] for m in zeros]}")
    print(f"control far pair      : {control['far_pair_5m_distmax_1m']} (expect distmax 1.0)")
    print(f"control near pair     : {control['near_pair_0p3m_distmax_1m']} (expect 0.2)")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
