"""Build a PHOTOSHOOT model: the FR3 with the blue hybrid dermis skin densely tiled with SPAD
sensor markers EVERYWHERE on the skin (farthest-point sampled across each dermis mesh, oriented
to the local surface normal) — the look of the real gentact skin for the paper figure.

This is a VISUAL model (red sensor dots as sites on the blue skin). The functional 40-sensor
model stays model_hybrid.xml. Output: assets/robots/franka_skin/model_photoshoot.xml
"""
from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_hybrid_on_franka_skin import parse_urdf, rpy_quat, FS, OLD_SKIN_BODIES  # noqa: E402

OUT = FS / "model_photoshoot.xml"
SKIN_RGBA = [0.20, 0.45, 0.88, 1.0]     # rich skin blue
DOT_RGBA = [1.0, 0.13, 0.13, 1.0]      # SPAD red
DOT_R = 0.0080                          # marker radius (m)
TARGET_SPACING = 0.020                  # ~2 cm between sensors on the surface
MAX_PER_MESH = 90


def farthest_point_sample(V, spacing, cap):
    """Even coverage: greedily pick points >= spacing apart (approx Poisson-disk), capped."""
    idx = [int(np.argmax(V[:, 2]))]
    d = np.linalg.norm(V - V[idx[0]], axis=1)
    while len(idx) < cap:
        j = int(np.argmax(d))
        if d[j] < spacing:
            break
        idx.append(j)
        d = np.minimum(d, np.linalg.norm(V - V[j], axis=1))
    return idx


def main():
    skins, sensors = parse_urdf()
    spec = mujoco.MjSpec.from_file(str(FS / "model.xml"))
    for nm in OLD_SKIN_BODIES:
        b = spec.body(nm)
        if b is not None:
            spec.delete(b)

    # blue dermis skin per link
    skin_bodies = {}
    added = set()
    for child, s in skins.items():
        if not s["mesh"]:
            continue
        mname = s["mesh"].replace(".stl", "")
        if s["mesh"] not in added:
            spec.add_mesh(name=mname, file=s["mesh"], scale=s["scale"])
            added.add(s["mesh"])
        parent = spec.body(s["parent"])
        nb = parent.add_body(name=child, pos=s["xyz"], quat=rpy_quat(s["rpy"]))
        g = nb.add_geom()
        g.type = mujoco.mjtGeom.mjGEOM_MESH
        g.meshname = mname
        g.rgba = SKIN_RGBA
        g.contype = 0
        g.conaffinity = 0
        g.group = 0
        g.mass = 1e-6
        skin_bodies[child] = nb

    # the EXACT 40 URDF sensors: position + rotation straight from the URDF joint origins.
    # Each marker is a flat SPAD module (thin red cylinder) whose face normal follows the URDF
    # rpy frame's z-axis — so both position AND orientation come from the URDF, nothing sampled.
    n_sens = 0
    for s in sensors:
        host = skin_bodies.get(s["parent"]) or spec.body(s["parent"])
        if host is None:
            print(f"WARN no host for {s['name']}")
            continue
        dg = host.add_geom()
        dg.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        dg.size = [DOT_R, 0.0016, 0]            # radius, half-thickness -> a SPAD wafer
        dg.pos = list(s["xyz"])
        dg.quat = rpy_quat(s["rpy"])            # URDF rotation, verbatim
        dg.rgba = DOT_RGBA
        dg.contype = 0
        dg.conaffinity = 0
        dg.group = 0
        dg.mass = 1e-9
        n_sens += 1

    spec.visual.global_.offwidth = 2400
    spec.visual.global_.offheight = 1800
    model = spec.compile()
    OUT.write_text(spec.to_xml())
    print(f"photoshoot model: nbody={model.nbody} sensors={n_sens} (URDF positions) -> {OUT}")


if __name__ == "__main__":
    main()
