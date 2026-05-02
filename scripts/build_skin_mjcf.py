#!/usr/bin/env python3
"""Convert a GenTact Blender export (sensor_sites.json) into MJCF camera bodies.

The Blender + GenTact toolbox produces a JSON list of dicts:
    [{"name": "link6_sensor_0", "pos": [x,y,z], "euler": [r,p,y]}, ...]

For each sensor we generate an MJCF ``<body>`` containing:
  * a tiny red sphere site (so the operator can see where the sensor is in
    the viewer),
  * a fixed camera, 8x8 resolution, 45 deg FOV, with an inner ``quat="0 1 0 0"``
    to flip MuJoCo's default -Z view direction to body +Z (outward).

The bodies are inserted directly into the appropriate ``link*_skin`` body in
the existing FR3 MJCF. We DO NOT use this script for URDF-driven builds —
that pipeline is in ``pla.sim.build_mjcf``. This script exists so the
Day 1-2 sensor-density iteration in Blender is fast (export JSON, run this,
view in MuJoCo).

Run::

    python scripts/build_skin_mjcf.py \
        --sites assets/mjcf/sensor_sites.json \
        --base-mjcf assets/mjcf/fr3_skin.xml \
        --out assets/mjcf/fr3_skin_blender.xml
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


def euler_to_quat_wxyz(euler_rpy: list[float]) -> tuple[float, ...]:
    from scipy.spatial.transform import Rotation as R

    x, y, z, w = R.from_euler("xyz", euler_rpy).as_quat()
    return (w, x, y, z)


def build_body_xml(site: dict) -> str:
    name = site["name"]
    pos = " ".join(f"{v:.6f}" for v in site["pos"])
    quat = euler_to_quat_wxyz(site["euler"])
    quat_str = " ".join(f"{v:.6f}" for v in quat)
    return (
        f'\n      <body name="{name}" pos="{pos}" quat="{quat_str}">\n'
        f'        <site name="{name}_site" type="sphere" size="0.004" rgba="1 0.2 0.2 1"/>\n'
        f'        <camera name="{name}" mode="fixed" pos="0 0 0" '
        f'quat="0 1 0 0" fovy="45" resolution="8 8"/>\n'
        f"      </body>"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sites", type=Path, required=True,
                   help="GenTact-exported sensor_sites.json")
    p.add_argument("--base-mjcf", type=Path, required=True,
                   help="Existing FR3 MJCF (e.g. assets/mjcf/fr3_skin.xml)")
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sites = json.loads(args.sites.read_text())
    text = args.base_mjcf.read_text()
    by_link: dict[str, list[dict]] = defaultdict(list)
    for s in sites:
        link = s["name"].split("_sensor_")[0]
        by_link[link].append(s)

    inserted = 0
    for link, group in by_link.items():
        bodies = "".join(build_body_xml(s) for s in group)
        # Insert before the </body> that closes the linkN_skin body.
        # Find the opening tag, then the matching closing </body> heuristically.
        pat = rf'(<body\s+name="{re.escape(link)}_skin"[^>]*>)'
        m = re.search(pat, text)
        if m is None:
            print(f"WARN: did not find {link}_skin in base MJCF; skipping {len(group)} sensors")
            continue
        # Insert immediately after the opening body tag — easiest and safe.
        idx = m.end()
        text = text[:idx] + bodies + text[idx:]
        inserted += len(group)

    args.out.write_text(text)
    print(f"wrote {args.out} with {inserted} sensor cameras inserted")


if __name__ == "__main__":
    main()
