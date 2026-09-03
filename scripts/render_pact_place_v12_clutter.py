#!/usr/bin/env python3
"""Render the v12 tabletop; no rollout."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
for path in (
    ROOT / "scripts",
    ROOT / "submodules" / "molmospaces",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pact_place_v12_contract import build_row, sha256_payload  # noqa: E402

SCENE = (
    ROOT
    / "custom_scenes"
    / "pact_place_corridor_v12.xml"
)
OUTPUT = ROOT / "diagnostics_output/pact_place_v12"
# THOR kitchen assets are Y-up. This 90-deg X rotation stands them on the bench.
STAND_QUAT = (2**-0.5, 2**-0.5, 0.0, 0.0)
BENCH_Z = 0.72
STANDING_KITCHEN = (
    # Screenshot: empty patch off the grasp lane, camera-right of the cup.
    {"uid": "Bottle_1", "xy": (0.80, -0.28)},
    # One extra behind the cup we grasp (Cup_10), toward the hood back.
    {"uid": "Soap_Bottle_1", "behind_grasp": True},
    {"uid": "Wine_Bottle_1"},
    {"uid": "Paper_Towel_1"},
    {"uid": "Tissue_Box_1"},
    {"uid": "Soap_Bottle_3"},
    {"uid": "Salt_Shaker_2"},
    {"uid": "Pepper_Shaker_2"},
    {"uid": "Salt_Shaker_1"},
    {"uid": "Pepper_Shaker_1"},
)
SAFE_X = (0.55, 1.24)
SAFE_Y = (-0.42, 0.36)
CLEAR_PAD_M = 0.02
# Expert inbound, grasp, and outbound live in this patch. Extras stay behind
# it or against the side walls so seed jitter does not put a bottle in the cup.
LANE_KEEP_LO = (0.48, -0.18, 0.68)
LANE_KEEP_HI = (0.98, 0.18, 1.16)
PICKUP_PAD_M = 0.05
# Distinct wall/back rests, ~10 cm apart, inset from the hood.
EMPTY_SPACE_XY = (
    (1.08, -0.28),
    (1.08, 0.20),
    (0.88, -0.30),
    (0.88, 0.24),
    (0.98, -0.30),
    (0.98, 0.22),
    (1.16, -0.20),
    (1.16, 0.16),
    (0.84, -0.22),
    (0.84, 0.18),
)
# Camera-right hood side wall is SAFE_Y[0] ≈ -0.34.
_SIT_Z: dict[str, float] = {}


def _candidate_xy(*, prefer_empty: bool = False) -> tuple[tuple[float, float], ...]:
    xs = np.arange(0.84, 1.161, 0.08)
    ys = np.arange(-0.32, 0.301, 0.08)
    points: list[tuple[float, float]] = []
    for x in xs:
        for y in ys:
            if x <= LANE_KEEP_HI[0] and abs(y) < 0.20:
                continue
            points.append((float(x), float(y)))
    points.sort(key=lambda point: (-abs(point[1]), -point[0]))
    del prefer_empty
    return tuple(dict.fromkeys((*EMPTY_SPACE_XY, *points)))
# Preview-only household edit. V10.10 keeps both soap vessels; this parks the
# outbound bottle and pulls the inbound bottle toward the robot.
PARK_HOUSEHOLD = ("Candle_1", "Candle_2", "Soap_Bottle_30")
KEEP_BOTTLE = "Soap_Bottle_11"
TOWARD_ROBOT_DX_M = -0.15
BOTTLE_MIN_X_M = 0.50


def _is_parked_household(body_name: str) -> bool:
    return body_name.split("/")[-1] in PARK_HOUSEHOLD


def _free_qpos(model: mujoco.MjModel, body_name: str) -> tuple[int, int] | None:
    body_id = int(model.body(body_name).id)
    joint_adr = int(model.body_jntadr[body_id])
    for joint_offset in range(int(model.body_jntnum[body_id])):
        joint_id = joint_adr + joint_offset
        if int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
            continue
        return int(model.jnt_qposadr[joint_id]), int(model.jnt_dofadr[joint_id])
    return None


def _park_household(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    for body_id in range(int(model.nbody)):
        body_name = model.body(body_id).name or ""
        if not _is_parked_household(body_name):
            continue
        addresses = _free_qpos(model, body_name)
        if addresses is None:
            continue
        qadr, dadr = addresses
        data.qpos[qadr : qadr + 3] = (0.0, 2.5, -1.0)
        data.qvel[dadr : dadr + 6] = 0.0


def _shift_kept_bottle_toward_robot(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    cup = _grasp_target_xy(model, data)
    for body_id in range(int(model.nbody)):
        body_name = model.body(body_id).name or ""
        if KEEP_BOTTLE not in body_name or not body_name.startswith("pact_clutter_"):
            continue
        addresses = _free_qpos(model, body_name)
        if addresses is None:
            continue
        qadr, dadr = addresses
        current_x = float(data.qpos[qadr])
        if cup is not None:
            target_x = float(cup[0]) + TOWARD_ROBOT_DX_M
        else:
            target_x = current_x + TOWARD_ROBOT_DX_M
        data.qpos[qadr] = max(BOTTLE_MIN_X_M, target_x)
        data.qvel[dadr : dadr + 6] = 0.0


def _apply_preview_household(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Park the outbound soap bottle; keep Soap_Bottle_11 on the robot side of the cup."""
    _park_household(model, data)
    _shift_kept_bottle_toward_robot(model, data)
    mujoco.mj_forward(model, data)


def _refresh_clutter_settle(task, model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Drop parked bottles from the topple gate; retarget the kept bottle."""
    scene = getattr(task, "scene_params", None) or {}
    settle = scene.get("pact_clutter_settle")
    if not isinstance(settle, dict):
        return
    updated = []
    for record in settle.get("objects") or []:
        body = str(record.get("body") or "")
        if _is_parked_household(body):
            continue
        body_id = int(model.body(body).id)
        item = dict(record)
        item["position_m"] = np.asarray(data.xpos[body_id], dtype=float).tolist()
        item["xmat"] = np.asarray(data.xmat[body_id], dtype=float).tolist()
        updated.append(item)
    settle["objects"] = updated


def _install_uid(uid: str) -> Path:
    from molmo_spaces.utils.lazy_loading_utils import install_uid

    return Path(install_uid(uid))


def _stand_quat(yaw_deg: float = 0.0) -> list[float]:
    from scipy.spatial.transform import Rotation as R

    quat = (
        R.from_euler("z", float(yaw_deg), degrees=True)
        * R.from_euler("x", 90.0, degrees=True)
    ).as_quat(scalar_first=True)
    return [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])]


def _sit_on_bench(
    uid: str, xy: tuple[float, float], *, yaw_deg: float = 0.0
) -> tuple[list[float], list[float]]:
    key = f"{uid}:{float(yaw_deg):.1f}"
    if key not in _SIT_Z:
        spec = mujoco.MjSpec.from_file(str(_install_uid(uid)))
        body = spec.worldbody.bodies[0]
        for joint in list(spec.joints):
            if joint.type == mujoco.mjtJoint.mjJNT_FREE:
                spec.delete(joint)
        body.quat = _stand_quat(yaw_deg)
        model = spec.compile()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        low, _high = _subtree_aabb(model, data, body.name, hull="primitive")
        _SIT_Z[key] = BENCH_Z - float(low[2]) + 0.001
    return [float(xy[0]), float(xy[1]), _SIT_Z[key]], _stand_quat(yaw_deg)


def _attach_standing_kitchen(spec: mujoco.MjSpec) -> list[str]:
    names: list[str] = []
    for index, item in enumerate(STANDING_KITCHEN):
        extra = mujoco.MjSpec.from_file(str(_install_uid(item["uid"])))
        body = extra.worldbody.bodies[0]
        for joint in list(extra.joints):
            if joint.type == mujoco.mjtJoint.mjJNT_FREE:
                extra.delete(joint)
        body.quat = list(STAND_QUAT)
        body.mocap = True
        # Park off the bench so household objects can settle first.
        frame = spec.worldbody.add_frame(pos=[0.0, 2.5 + 0.25 * index, -1.0])
        namespace = f"pact_preview_extra_{item['uid']}/"
        original_name = body.name
        frame.attach_body(body, namespace, "")
        names.append(namespace + original_name)
    return names


def _install_preview_contact_classes() -> None:
    """Score standing-kitchen extras as clutter, not other_environment."""
    import molmo_spaces.tasks.pact_place_contact_audit as audit

    if getattr(audit, "_pact_preview_extra_contact_patched", False):
        return
    original = audit.classify_contact
    original_is_clutter = audit._is_clutter_name

    def classify_contact(pair):
        blob = " ".join(
            str(pair.get(key, ""))
            for key in ("geom1", "geom2", "body1", "body2", "root1", "root2")
        )
        if "pact_preview_extra_" in blob:
            return "clutter"
        return original(pair)

    def is_clutter_name(name: str) -> bool:
        return original_is_clutter(name) or "pact_preview_extra_" in str(name)

    audit.classify_contact = classify_contact
    audit._is_clutter_name = is_clutter_name
    audit._pact_preview_extra_contact_patched = True


def _lane_keepout() -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(LANE_KEEP_LO, dtype=float), np.asarray(LANE_KEEP_HI, dtype=float)


def _geom_aabb(model: mujoco.MjModel, data: mujoco.MjData, geom_id: int) -> tuple[np.ndarray, np.ndarray]:
    center = np.asarray(data.geom_xpos[geom_id], dtype=float)
    if int(model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_BOX):
        axes = np.asarray(data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
        half = np.asarray(model.geom_size[geom_id], dtype=float)
        corners = [
            center + axes @ (half * np.array([sx, sy, sz], dtype=float))
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ]
        stacked = np.stack(corners)
        return stacked.min(axis=0), stacked.max(axis=0)
    radius = max(float(model.geom_rbound[geom_id]), 0.01)
    return center - radius, center + radius


def _subtree_aabb(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_name: str,
    *,
    skip_primitive: bool = False,
    hull: str = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    """AABB of a body. Default hull is PrimitiveCollider boxes (tight THOR hull)."""
    if hull == "auto":
        hull = "visual" if skip_primitive else "primitive"
    root = int(model.body(body_name).id)
    lows: list[np.ndarray] = []
    highs: list[np.ndarray] = []
    for geom_id in range(int(model.ngeom)):
        geom_name = model.geom(geom_id).name or ""
        is_prim = "PrimitiveCollider" in geom_name
        if hull == "primitive" and not is_prim:
            continue
        if hull == "visual" and is_prim:
            continue
        if hull != "primitive" and int(model.geom_contype[geom_id]) == 0:
            continue
        walk = int(model.geom_bodyid[geom_id])
        while walk and walk != root:
            walk = int(model.body_parentid[walk])
        if walk != root:
            continue
        low, high = _geom_aabb(model, data, geom_id)
        lows.append(low)
        highs.append(high)
    if not lows and hull == "primitive":
        return _subtree_aabb(model, data, body_name, hull="collision")
    if not lows:
        origin = np.asarray(data.xpos[root], dtype=float)
        return origin, origin
    return np.min(np.stack(lows), axis=0), np.max(np.stack(highs), axis=0)


def _aabbs_overlap(left_lo: np.ndarray, left_hi: np.ndarray, right_lo: np.ndarray, right_hi: np.ndarray, pad: float = 0.01) -> bool:
    return bool(np.all(left_lo < right_hi + pad) and np.all(right_lo < left_hi + pad))


def _household_boxes(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    include_lane: bool = True,
    include_pickup: bool = True,
) -> list[tuple[np.ndarray, np.ndarray]]:
    boxes: list[tuple[np.ndarray, np.ndarray]] = []
    for body_id in range(int(model.nbody)):
        name = model.body(body_id).name or ""
        if not name.startswith("pact_clutter_"):
            continue
        if _is_parked_household(name):
            continue
        if "/" not in name or name.count("/") != 1:
            continue
        boxes.append(_subtree_aabb(model, data, name, hull="primitive"))
    for geom_name in ("hood_back", "hood_side_l", "hood_side_r"):
        boxes.append(_geom_aabb(model, data, int(model.geom(geom_name).id)))
    for site_id in range(int(model.nsite)):
        name = model.site(site_id).name or ""
        if "ReceptacleCollider" not in name:
            continue
        body_name = model.body(int(model.site_bodyid[site_id])).name or ""
        if "cavity_obj_" in body_name:
            continue
        center = np.asarray(data.site_xpos[site_id], dtype=float)
        axes = np.asarray(data.site_xmat[site_id], dtype=float).reshape(3, 3)
        half = np.asarray(model.site_size[site_id], dtype=float)
        corners = [
            center + axes @ (half * np.array([sx, sy, sz], dtype=float))
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ]
        stacked = np.stack(corners)
        boxes.append((stacked.min(axis=0), stacked.max(axis=0)))
    pad = np.array([PICKUP_PAD_M, PICKUP_PAD_M, PICKUP_PAD_M], dtype=float)
    if include_pickup:
        for body_id in range(int(model.nbody)):
            name = model.body(body_id).name or ""
            if not name.startswith("cavity_obj_") or name.count("/") != 1:
                continue
            low, high = _subtree_aabb(model, data, name, hull="primitive")
            boxes.append((low - pad, high + pad))
    for geom_name in (
        "place_receptacle_floor_g",
        "place_receptacle_lip_left_g",
        "place_receptacle_lip_right_g",
    ):
        try:
            boxes.append(_geom_aabb(model, data, int(model.geom(geom_name).id)))
        except Exception:
            continue
    if include_lane:
        boxes.append(_lane_keepout())
    return boxes


def _aabb_inside_safe(low: np.ndarray, high: np.ndarray) -> bool:
    return bool(
        low[0] >= SAFE_X[0]
        and high[0] <= SAFE_X[1]
        and low[1] >= SAFE_Y[0]
        and high[1] <= SAFE_Y[1]
    )


def _snap_to_bench(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> None:
    mocap_id = int(model.body_mocapid[int(model.body(name).id)])
    mujoco.mj_forward(model, data)
    low, _high = _subtree_aabb(model, data, name, hull="primitive")
    data.mocap_pos[mocap_id, 2] += BENCH_Z + 0.001 - float(low[2])
    mujoco.mj_forward(model, data)


def _grasp_target_xy(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[float, float] | None:
    for body_id in range(int(model.nbody)):
        name = model.body(body_id).name or ""
        if not name.startswith("cavity_obj_") or name.count("/") != 1:
            continue
        pos = data.xpos[body_id]
        return float(pos[0]), float(pos[1])
    return None


def _place_standing_kitchen(model: mujoco.MjModel, data: mujoco.MjData, names: list[str]) -> list[str]:
    blocked = _household_boxes(model, data)
    placed: list[str] = []
    used_xy: set[tuple[float, float]] = set()
    for name, item in zip(names, STANDING_KITCHEN, strict=True):
        slots = _candidate_xy(prefer_empty=bool(item.get("prefer_empty")))
        mocap_id = int(model.body_mocapid[int(model.body(name).id)])
        if mocap_id < 0:
            raise RuntimeError(f"{name} is not a mocap body")
        chosen: tuple[float, float] | None = None
        yaw_deg = float(item.get("yaw_deg", 0.0))
        pinned: list[tuple[float, float]] = []
        if item.get("behind_grasp"):
            target = _grasp_target_xy(model, data)
            if target is not None:
                for dx in (0.14, 0.16, 0.12, 0.18, 0.10):
                    pinned.append((float(target[0] + dx), float(target[1])))
        if "xy" in item:
            pinned.append(tuple(item["xy"]))
        # Behind the cup must not fall back to a wall slot.
        slots = tuple(pinned) if item.get("behind_grasp") else tuple(pinned) + slots
        obstacles = (
            _household_boxes(model, data, include_lane=False, include_pickup=False)
            if item.get("behind_grasp")
            else blocked
        )
        for xy in slots:
            if xy in used_xy:
                continue
            pos, quat = _sit_on_bench(item["uid"], xy, yaw_deg=yaw_deg)
            data.mocap_pos[mocap_id][:] = np.asarray(pos, dtype=np.float64)
            data.mocap_quat[mocap_id][:] = np.asarray(quat, dtype=np.float64)
            mujoco.mj_forward(model, data)
            _snap_to_bench(model, data, name)
            low, high = _subtree_aabb(model, data, name, hull="primitive")
            if not _aabb_inside_safe(low, high):
                continue
            if any(_aabbs_overlap(low, high, other_lo, other_hi, CLEAR_PAD_M) for other_lo, other_hi in obstacles):
                continue
            chosen = xy
            break
        if chosen is None and item.get("behind_grasp") and pinned:
            xy = pinned[0]
            pos, quat = _sit_on_bench(item["uid"], xy, yaw_deg=yaw_deg)
            data.mocap_pos[mocap_id][:] = np.asarray(pos, dtype=np.float64)
            data.mocap_quat[mocap_id][:] = np.asarray(quat, dtype=np.float64)
            mujoco.mj_forward(model, data)
            _snap_to_bench(model, data, name)
            chosen = xy
        if chosen is None:
            data.mocap_pos[mocap_id][:] = np.asarray([0.0, 2.5, -1.0], dtype=np.float64)
            mujoco.mj_forward(model, data)
            continue
        used_xy.add(chosen)
        blocked.append(_subtree_aabb(model, data, name, hull="primitive"))
        placed.append(name)
    return placed


def extras_overlap_motion_lane(
    model: mujoco.MjModel, data: mujoco.MjData, names: list[str]
) -> list[str]:
    lane_lo, lane_hi = _lane_keepout()
    hits: list[str] = []
    for name in names:
        low, high = _subtree_aabb(model, data, name, hull="primitive")
        if _aabbs_overlap(low, high, lane_lo, lane_hi, 0.0):
            hits.append(name)
    return hits


def _hide_primitive_colliders(model: mujoco.MjModel) -> None:
    for geom_id in range(int(model.ngeom)):
        name = model.geom(geom_id).name or ""
        if "PrimitiveCollider" in name:
            model.geom_rgba[geom_id, 3] = 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--viewer", action="store_true", help="open an interactive MuJoCo viewer"
    )
    args = parser.parse_args()
    os.environ.setdefault("MLSPACES_ASSETS_DIR", str(ROOT / "assets"))

    from molmo_spaces.tasks.enclosure_reach import (
        PactPlaceCorridorV1010FourObjectSampler,
    )
    from run_pact_place_v12_expert import _make_config
    from run_pact_place_v12_cameras import THIRD_PERSON_FOV, third_person_pose

    row = build_row("F0_target_side_stagger", "left", "center", 0)
    row["pact_v106_scene_sha256"] = hashlib.sha256(SCENE.read_bytes()).hexdigest()
    row["pact_v1010_scene_relative"] = str(SCENE.relative_to(ROOT))
    row["environment_version"] = "pact_place_corridor_v12"
    row.pop("row_sha256", None)
    row["row_sha256"] = sha256_payload(row)

    config = _make_config(
        OUTPUT / "scratch" / "result.json",
        scene_xml=SCENE,
        sampler_class="PactPlaceCorridorV1010FourObjectSampler",
    )

    sampler = PactPlaceCorridorV1010FourObjectSampler(config)
    extra_bodies: list[str] = []
    original_add = sampler.add_auxiliary_objects

    def add_auxiliary_objects(spec: mujoco.MjSpec) -> None:
        original_add(spec)
        extra_bodies.extend(_attach_standing_kitchen(spec))

    sampler.add_auxiliary_objects = add_auxiliary_objects  # type: ignore[method-assign]
    task = None
    try:
        sampler.seed_task_sampling(int(row["task_seed_u32"]))
        sampler.set_pact_manifest_row(row)
        task = sampler.sample_task(house_index=1)
        if task is None:
            raise RuntimeError("preview sampler returned no task")
        task.reset()
        env = task.env
        model, data = env.current_model, env.current_data
        _hide_primitive_colliders(model)
        _apply_preview_household(model, data)
        _refresh_clutter_settle(task, model, data)
        extra_bodies[:] = _place_standing_kitchen(model, data, extra_bodies)
        mujoco.mj_forward(model, data)

        OUTPUT.mkdir(parents=True, exist_ok=True)
        def render_lookat(position: list[float], target: list[float]) -> np.ndarray:
            position_array = np.asarray(position, dtype=np.float64)
            forward = np.asarray(target, dtype=np.float64) - position_array
            forward /= np.linalg.norm(forward)
            right = np.cross(forward, np.asarray([0.0, 0.0, 1.0]))
            right /= np.linalg.norm(right)
            up = np.cross(right, forward)
            return np.asarray(
                env._render_frame(
                    position_array, forward, up, 52.0, segmentation=False
                )
            )

        position, forward, up = third_person_pose(env)
        table_frame = np.asarray(
            env._render_frame(
                position,
                forward,
                up,
                THIRD_PERSON_FOV,
                segmentation=False,
            )
        )
        table_view = OUTPUT / "center_F0_left_table_camera.png"
        Image.fromarray(table_frame).save(table_view)

        # Development visualization only: make the hood roof transparent for a
        # top-down layout inspection. Restore it immediately after rendering.
        hood_top = model.geom("hood_top")
        hood_top_alpha = float(hood_top.rgba[3])
        hood_top.rgba[3] = 0.0
        try:
            top_view = OUTPUT / "center_F0_left_table_topdown_cutaway.png"
            Image.fromarray(
                np.asarray(
                    env._render_frame(
                        np.asarray([0.95, 0.0, 1.70]),
                        np.asarray([0.0, 0.0, -1.0]),
                        np.asarray([1.0, 0.0, 0.0]),
                        52.0,
                        segmentation=False,
                    )
                )
            ).save(top_view)
        finally:
            hood_top.rgba[3] = hood_top_alpha

        active = list(getattr(sampler, "_pact_active_clutter_names", []))
        allowed_support = ("bench_top", "bench_body", "floor")
        geo_contacts: dict[str, list[dict[str, float | str]]] = {
            name: [] for name in extra_bodies
        }
        for contact_index in range(int(data.ncon)):
            contact = data.contact[contact_index]
            geom1 = model.geom(int(contact.geom1)).name or ""
            geom2 = model.geom(int(contact.geom2)).name or ""
            body1 = model.body(int(model.geom_bodyid[int(contact.geom1)])).name or ""
            body2 = model.body(int(model.geom_bodyid[int(contact.geom2)])).name or ""
            for name in extra_bodies:
                if name in f"{geom1} {geom2} {body1} {body2}":
                    other_geom = geom2 if name in f"{geom1} {body1}" else geom1
                    other_body = body2 if name in f"{geom1} {body1}" else body1
                    if other_geom in allowed_support or other_body in allowed_support:
                        continue
                    geo_contacts[name].append(
                        {
                            "other_geom": other_geom,
                            "other_body": other_body,
                            "distance_m": float(contact.dist),
                        }
                    )
        report = {
            "role": "development_preview_not_a_gate",
            "authorizes_collection": False,
            "scene": str(SCENE.relative_to(ROOT)),
            "scene_sha256": row["pact_v106_scene_sha256"],
            "existing_household_objects": active,
            "existing_household_object_count": len(active),
            "added_standing_kitchen_objects": extra_bodies,
            "added_standing_kitchen_uids": [item["uid"] for item in STANDING_KITCHEN],
            "extra_xy": {
                name: [
                    round(float(data.xpos[int(model.body(name).id)][0]), 3),
                    round(float(data.xpos[int(model.body(name).id)][1]), 3),
                ]
                for name in extra_bodies
            },
            "geometric_clutter_count": len(extra_bodies),
            "geometric_clutter_fixed_no_joints": {
                name: int(model.body_jntnum[int(model.body(name).id)]) == 0
                for name in extra_bodies
            },
            "geometric_object_contacts": geo_contacts,
            "geometric_object_contact_free": not any(geo_contacts.values()),
            "renders": [str(table_view), str(top_view)],
            "topdown_note": "hood_top hidden for layout inspection only",
        }
        (OUTPUT / "preview.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))

        if args.viewer:
            from mujoco import viewer as mj_viewer

            keep_id = int(model.body(f"pact_clutter_06/{KEEP_BOTTLE}").id)
            keep_xyz = [float(x) for x in data.xpos[keep_id]]
            parked = [
                name
                for name in active
                if _is_parked_household(name)
            ]
            print(
                json.dumps(
                    {
                        "viewer": "Soap_Bottle_11 toward robot",
                        "kept_bottle": KEEP_BOTTLE,
                        "kept_xyz_m": keep_xyz,
                        "parked_off_bench": PARK_HOUSEHOLD,
                    }
                ),
                flush=True,
            )
            with mj_viewer.launch_passive(model, data) as viewer:
                viewer.cam.lookat[:] = keep_xyz
                viewer.cam.distance = 1.35
                viewer.cam.azimuth = -90.0
                viewer.cam.elevation = -18.0
                viewer.sync()
                print("MuJoCo viewer open: drag to orbit, scroll to zoom, close when done")
                while viewer.is_running():
                    time.sleep(0.05)
    finally:
        sampler.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
