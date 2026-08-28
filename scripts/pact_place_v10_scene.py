#!/usr/bin/env python3
"""Pose the V10 compound-pendant geoms on a compiled MuJoCo model."""

from __future__ import annotations

from typing import Any

import numpy as np

from pact_place_v10_compound_pendant_contract import (
    ALL_GEOMS,
    ASSEMBLY_PARK_XYZ_M,
    LEGACY_MOUNT_BODIES,
    PENDANT_BODY,
)

INACTIVE_HALF_M = (0.001, 0.001, 0.001)
LEGACY_PARK_XYZ_M = {
    "pact_clutter_mount_wall_left": (0.0, 2.2, -2.0),
    "pact_clutter_mount_wall_right": (0.0, -2.2, -2.0),
    "pact_clutter_mount_ceiling": (0.0, 0.0, -2.0),
}


def _set_geom(model, name: str, item: dict[str, Any] | None) -> None:
    geom_id = int(model.geom(name).id)
    # Compiled zero-local geoms get mjSAMEFRAME_BODY and ignore later geom_pos writes.
    model.geom_sameframe[geom_id] = 0
    active = bool(item is not None and item.get("active"))
    if not active:
        model.geom_pos[geom_id] = np.zeros(3, dtype=float)
        model.geom_size[geom_id] = np.asarray(INACTIVE_HALF_M, dtype=float)
        model.geom_contype[geom_id] = 0
        model.geom_conaffinity[geom_id] = 0
        rgba = np.asarray(model.geom_rgba[geom_id], dtype=float).copy()
        rgba[3] = 0.0
        model.geom_rgba[geom_id] = rgba
        return
    model.geom_pos[geom_id] = np.asarray(item["center_m"], dtype=float)
    model.geom_size[geom_id] = np.asarray(item["half_m"], dtype=float)
    model.geom_contype[geom_id] = 8
    model.geom_conaffinity[geom_id] = 15
    rgba = np.asarray(model.geom_rgba[geom_id], dtype=float).copy()
    rgba[3] = 1.0
    model.geom_rgba[geom_id] = rgba


def pose_assembly_geoms(
    model,
    assembly: dict[str, Any] | None,
    *,
    parked: bool,
) -> None:
    by_geom = {
        item["geom"]: item for item in (assembly or {}).get("components") or []
    }
    for name in ALL_GEOMS:
        _set_geom(model, name, None if parked else by_geom.get(name))


def mocap_id(model, body_name: str) -> int:
    body_id = int(model.body(body_name).id)
    mocap = int(model.body_mocapid[body_id])
    if mocap < 0:
        raise ValueError(f"{body_name} is not a mocap body")
    return mocap


def park_legacy_mounts(model, data) -> None:
    for body, xyz in LEGACY_PARK_XYZ_M.items():
        data.mocap_pos[mocap_id(model, body)] = np.asarray(xyz, dtype=float)


def pose_assembly_on_data(
    model,
    data,
    assembly: dict[str, Any] | None,
    *,
    parked: bool,
) -> None:
    park_legacy_mounts(model, data)
    pose_assembly_geoms(model, assembly, parked=parked)
    if parked or not assembly:
        data.mocap_pos[mocap_id(model, PENDANT_BODY)] = np.asarray(
            ASSEMBLY_PARK_XYZ_M, dtype=float
        )
        return
    data.mocap_pos[mocap_id(model, PENDANT_BODY)] = np.zeros(3, dtype=float)
