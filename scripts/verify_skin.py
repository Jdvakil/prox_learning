#!/usr/bin/env python3
"""Verify a sensor-skin MJCF is correctly mounted.

Loads the MJCF, enumerates every camera whose name contains ``sensor``, and
renders an 8x8 depth image *in an empty scene*. A sensor that is mounted
inside the link mesh will show frac_near < 5 cm > threshold (default 5%);
those are flagged as self-hitting.

Run::

    python scripts/verify_skin.py --mjcf assets/mjcf/fr3_skin_fixed.xml \
        --frac-near-threshold 0.05 \
        --out reports/checks/skin_verify.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    import mujoco
except ImportError as e:
    raise SystemExit("mujoco>=3 required") from e


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mjcf", type=Path, required=True)
    p.add_argument("--frac-near-threshold", type=float, default=0.05,
                   help="fraction of pixels < 50mm above which we flag a sensor")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model = mujoco.MjModel.from_xml_path(str(args.mjcf))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    sensor_cams = [
        (i, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i))
        for i in range(model.ncam)
    ]
    sensor_cams = [(i, n) for i, n in sensor_cams if n and "sensor" in n]
    print(f"found {len(sensor_cams)} sensor cameras (of {model.ncam} total)")

    renderer = mujoco.Renderer(model, height=8, width=8)
    renderer.enable_depth_rendering()

    self_hits: list[dict] = []
    per_sensor: list[dict] = []
    for cam_id, name in sensor_cams:
        renderer.update_scene(data, camera=cam_id)
        depth_mm = renderer.render() * 1000.0
        frac_near = float((depth_mm < 50).mean())
        per_sensor.append({
            "name": name,
            "frac_near": frac_near,
            "min_mm": float(depth_mm.min()),
            "max_mm": float(depth_mm.max()),
            "mean_mm": float(depth_mm.mean()),
        })
        if frac_near > args.frac_near_threshold:
            self_hits.append(per_sensor[-1])
            print(f"  WARN self-hit: {name} frac_near={frac_near:.2f}")

    summary = {
        "n_cameras_total": int(model.ncam),
        "n_sensor_cameras": len(sensor_cams),
        "n_self_hits": len(self_hits),
        "self_hit_threshold_frac": args.frac_near_threshold,
        "per_sensor": per_sensor,
        "self_hits": self_hits,
    }
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2))
        print(f"wrote {args.out}")
    print(f"self-hitting sensors: {len(self_hits)} / {len(sensor_cams)}")
    print("target: < 5 self-hitting sensors")


if __name__ == "__main__":
    main()
