#!/usr/bin/env python3
"""B1b: freeze a stable, category-capped prop/mount palette for PACT place V8B."""

from __future__ import annotations

import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MOLMO = ROOT / "submodules" / "molmospaces"
if str(MOLMO) not in sys.path:
    sys.path.insert(0, str(MOLMO))

from molmo_spaces.utils.lazy_loading_utils import install_uid  # noqa: E402
from molmo_spaces.utils.object_metadata import ObjectMeta  # noqa: E402
from molmo_spaces.utils.synset_utils import get_valid_pickupable_obja_uids  # noqa: E402

OUTPUT = ROOT / "diagnostics_output/pact_place_clutter_sweep_v8b/palette_stability.json"
SHELF_TOP_Z = 0.72
SETTLE_STEPS = 300
MAX_CENTER_DRIFT_M = 0.005
MAX_ORIENTATION_CHANGE_DEG = 5.0
POSE_QUAT_WXYZ = np.asarray([2**-0.5, 2**-0.5, 0.0, 0.0], dtype=float)

# These five read as bolted wall fixtures in review: two switch plates, a sign,
# a network switch, and a shallow clock. They never enter the prop settle pool.
MOUNT_UIDS = (
    "8f601aff31c8480db54a5fe7c85b34c2",
    "80ed82d00f494be2b5a8706d3c06c832",
    "815d3d48bec04deda2a4e46dd742a366",
    "1ccab191f993433e84e3fd66778a7708",
    "RoboTHOR_alarm_clock_bajk_v",
)

# Ordered by preference. More than thirteen are measured so rejected assets do
# not silently shrink the frozen prop partition.
PROP_CANDIDATE_UIDS = (
    "Candle_1",
    "Candle_3",
    "Cellphone_1",
    "Cellphone_5",
    "Soap_Bottle_1",
    "Soap_Bottle_11",
    "021b1586089044278164dadc584abb89",
    "057658eddc6a4729af79b02ca1e68513",
    "062ac63bbf7e41c9987916d6488bef2a",
    "10f55a08aa9e4e02bacd443a064a29d5",
    "13487661582e457f9ddb9f6a9846361a",
    "14daacbaf0f94f458851e15dfd717426",
    "4e7d2fdb112b4145a8c467c8e5339dbb",
    "50e060a0d57e4e378a87d34256270fe8",
    "1c809c3aa8564665beaab1d1950cb3c7",
    "f772035734964e4994f5285d8e8b8a08",
    "31ff480523f24d2991140d46b7380081",
    "529bad643a234a7fa4c18bb8fc16e8b3",
    "RoboTHOR_book_ai2_5_v",
    "Book_14",
    "Plate_10",
    "Plate_22",
    "03fa6768245446f4bab866db37a6caec",
    "16bfd987e11444ca9c065718d49f5877",
    "521e5cb1802643d0bb7366e1f85c18a9",
    "Alarm_Clock_21",
    "a433948e9a5d41219b75f384f22a5101",
)


def _collision_bounds(model, data, root_body_id: int) -> tuple[np.ndarray, np.ndarray]:
    root_id = int(model.body_rootid[root_body_id])
    lows, highs = [], []
    for geom_id in range(int(model.ngeom)):
        body_id = int(model.geom_bodyid[geom_id])
        if int(model.body_rootid[body_id]) != root_id:
            continue
        if not (int(model.geom_contype[geom_id]) or int(model.geom_conaffinity[geom_id])):
            continue
        local_center = np.asarray(model.geom_aabb[geom_id, :3], dtype=float)
        local_half = np.asarray(model.geom_aabb[geom_id, 3:], dtype=float)
        rotation = np.asarray(data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
        center = np.asarray(data.geom_xpos[geom_id], dtype=float) + rotation @ local_center
        half = np.abs(rotation) @ local_half
        lows.append(center - half)
        highs.append(center + half)
    if not lows:
        raise RuntimeError("asset has no active collision geometry")
    return np.min(lows, axis=0), np.max(highs, axis=0)


def _annotation(uid: str) -> dict[str, Any]:
    annotation = ObjectMeta.annotation(uid) or {}
    bbox = annotation.get("boundingBox") or {}
    dimensions = [float(bbox.get(axis, 0.0)) for axis in "xyz"]
    if min(dimensions) <= 0.0 or max(dimensions) > 0.30:
        raise RuntimeError(f"unsupported dimensions for {uid}: {dimensions}")
    return {
        "uid": uid,
        "category": str(annotation.get("category") or "object"),
        "dimensions_m": dimensions,
        "max_dimension_m": max(dimensions),
    }


def settle_prop(uid: str) -> dict[str, Any]:
    spec = mujoco.MjSpec.from_file(str(install_uid(uid)))
    body = spec.worldbody.bodies[0]
    if not body.first_joint():
        body.add_joint(
            name=f"{uid}_v8b_stability_free",
            type=mujoco.mjtJoint.mjJNT_FREE,
            damping=0.05,
        )
    floor = spec.worldbody.add_geom(
        name="v8b_stability_shelf",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[1.0, 1.0, 0.05],
        pos=[0.0, 0.0, SHELF_TOP_Z],
    )
    floor.contype = 1
    floor.conaffinity = 1
    model = spec.compile()
    data = mujoco.MjData(model)
    body_id = int(model.body(body.name).id)
    joint_id = int(model.body_jntadr[body_id])
    if joint_id < 0 or int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
        raise RuntimeError(f"{uid} root is not free after compilation")
    qadr = int(model.jnt_qposadr[joint_id])
    dadr = int(model.jnt_dofadr[joint_id])
    data.qpos[qadr : qadr + 3] = [0.68, 0.0, 0.0]
    data.qpos[qadr + 3 : qadr + 7] = POSE_QUAT_WXYZ
    mujoco.mj_forward(model, data)
    low, _high = _collision_bounds(model, data, body_id)
    data.qpos[qadr + 2] += SHELF_TOP_Z - float(low[2]) + 0.002
    data.qvel[dadr : dadr + 6] = 0.0
    mujoco.mj_forward(model, data)
    initial_low, initial_high = _collision_bounds(model, data, body_id)
    initial_center = (initial_low + initial_high) / 2.0
    initial_rotation = np.asarray(data.xmat[body_id], dtype=float).reshape(3, 3).copy()
    for _ in range(SETTLE_STEPS):
        mujoco.mj_step(model, data)
    final_low, final_high = _collision_bounds(model, data, body_id)
    final_center = (final_low + final_high) / 2.0
    final_rotation = np.asarray(data.xmat[body_id], dtype=float).reshape(3, 3)
    cosine = float(np.clip((np.trace(initial_rotation.T @ final_rotation) - 1.0) / 2.0, -1.0, 1.0))
    orientation_deg = math.degrees(math.acos(cosine))
    center_drift_m = float(np.linalg.norm(final_center - initial_center))
    accepted = center_drift_m <= MAX_CENTER_DRIFT_M and orientation_deg <= MAX_ORIENTATION_CHANGE_DEG
    return {
        **_annotation(uid),
        "slot_class": "prop",
        "settle_steps": SETTLE_STEPS,
        "center_drift_m": center_drift_m,
        "orientation_change_deg": orientation_deg,
        "initial_collision_bounds_m": [initial_low.tolist(), initial_high.tolist()],
        "final_collision_bounds_m": [final_low.tolist(), final_high.tolist()],
        "collision_dimensions_m": (final_high - final_low).tolist(),
        "accepted": accepted,
        "reject_reason": None if accepted else "center_drift_or_orientation_threshold",
    }


def measure_mount(uid: str) -> dict[str, Any]:
    spec = mujoco.MjSpec.from_file(str(install_uid(uid)))
    body = spec.worldbody.bodies[0]
    for joint in list(spec.joints):
        if joint.type == mujoco.mjtJoint.mjJNT_FREE:
            spec.delete(joint)
    body.mocap = True
    model = spec.compile()
    data = mujoco.MjData(model)
    body_id = int(model.body(body.name).id)
    data.mocap_quat[int(model.body_mocapid[body_id])] = POSE_QUAT_WXYZ
    mujoco.mj_forward(model, data)
    low, high = _collision_bounds(model, data, body_id)
    return {
        "collision_dimensions_m": (high - low).tolist(),
        "collision_bounds_at_origin_m": [low.tolist(), high.tolist()],
    }


def main() -> int:
    os.environ.setdefault("MLSPACES_ASSETS_DIR", "/root/prox_learning/assets")
    pickupable = set(get_valid_pickupable_obja_uids())
    requested = set(MOUNT_UIDS) | set(PROP_CANDIDATE_UIDS)
    missing = sorted(requested - pickupable)
    if missing:
        raise SystemExit(f"palette candidates are not pickupable: {missing}")
    records: list[dict[str, Any]] = []
    for uid in MOUNT_UIDS:
        records.append(
            {
                **_annotation(uid),
                **measure_mount(uid),
                "slot_class": "mount",
                "settle_steps": 0,
                "center_drift_m": None,
                "orientation_change_deg": None,
                "accepted": True,
                "reject_reason": None,
                "settling_skipped": True,
                "reason": "kinematic_mocap_mount",
            }
        )
    for uid in PROP_CANDIDATE_UIDS:
        record = settle_prop(uid)
        records.append(record)
        print(
            f"{uid}: accepted={record['accepted']} drift={record['center_drift_m']:.6f}m "
            f"rotation={record['orientation_change_deg']:.3f}deg",
            flush=True,
        )
    selected_props: list[dict[str, Any]] = []
    category_counts: Counter[str] = Counter()
    for record in records[len(MOUNT_UIDS) :]:
        category = str(record["category"])
        if not record["accepted"] or category_counts[category] >= 3:
            continue
        selected_props.append(record)
        category_counts[category] += 1
        if len(selected_props) == 13:
            break
    if len(selected_props) != 13:
        raise SystemExit(f"only {len(selected_props)} stable category-capped props")
    selected_records = [*records[: len(MOUNT_UIDS)], *selected_props]
    palette = []
    for index, record in enumerate(selected_records):
        collision_dimensions = list(record["collision_dimensions_m"])
        palette.append(
            {
                "slot": f"{index:02d}",
                "slot_class": record["slot_class"],
                "uid": record["uid"],
                "category": record["category"],
                "dimensions_m": collision_dimensions,
                "annotation_dimensions_m": record["dimensions_m"],
                "max_dimension_m": max(collision_dimensions),
                "size_class": "small" if max(collision_dimensions) <= 0.10 else (
                    "medium" if max(collision_dimensions) <= 0.18 else "large"
                ),
                "body_prefix": f"pact_clutter_{index:02d}/",
            }
        )
    all_category_counts = Counter(item["category"] for item in palette)
    if max(all_category_counts.values()) > 3:
        raise SystemExit(f"palette category cap violated: {all_category_counts}")
    document = {
        "schema_version": "pact_place_v8b_palette_stability_v1",
        "settle_steps": SETTLE_STEPS,
        "max_center_drift_m": MAX_CENTER_DRIFT_M,
        "max_orientation_change_deg": MAX_ORIENTATION_CHANGE_DEG,
        "mount_settle_policy": "skip_kinematic_mocap_mounts",
        "records": records,
        "palette": palette,
        "palette_size": len(palette),
        "slot_class_counts": dict(Counter(item["slot_class"] for item in palette)),
        "palette_category_counts": dict(all_category_counts),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(OUTPUT), "palette_size": len(palette), "slot_class_counts": document["slot_class_counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
