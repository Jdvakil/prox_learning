"""Simulated ToF sensor designs, applied post-hoc to recorded skin depths.

The recorded skin is an 8x8-zone, ~45-degree, 4-subframe depth array per sensor
in metres - essentially a VL53L5CX/L8CX-class multizone part. Each design here
degrades that recording to what a different STMicroelectronics FlightSense part
would have measured, so the same demonstrations can train one PACT variant per
sensor design with the policy architecture unchanged:

  vl53l5cx_4x4    4x4 multizone, same FoV family, 4 m   - resolution axis
  vl53l1x_1zone   single zone, ~27 deg FoV, 4 m         - resolution floor
  vl6180x_short   single zone, ~23 deg FoV, 0.6 m max   - range floor

Mechanics, in order: centre-crop the 8x8 grid for the narrower field of view
(a k-of-8 crop scales the half-angle by atan(k/8 * tan(22.5 deg)); k=5 gives
~29 deg, k=4 gives ~23 deg), min-pool the crop into the design's zone count
(a SPAD zone reports its nearest return, so min is the physical aggregation),
saturate everything beyond the design's ranging limit at that limit, add
datasheet-scale Gaussian ranging noise, and broadcast the zones back onto the
native 8x8 grid so every downstream shape - subframe pooling, the causal
window, the 40x128 encoder - is untouched. The designs therefore differ only
in the information content of the proximity input.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DESIGNS: dict[str, dict] = {
    "vl53l5cx_4x4": {
        "part": "VL53L5CX (4x4 mode)",
        "zones": 4, "fov_crop": 8, "max_range_m": 4.0,
        "noise_floor_m": 0.005, "noise_pct": 0.02,
    },
    "vl53l1x_1zone": {
        "part": "VL53L1X",
        "zones": 1, "fov_crop": 5, "max_range_m": 4.0,
        "noise_floor_m": 0.005, "noise_pct": 0.025,
    },
    "vl6180x_short": {
        "part": "VL6180X (extended-range mode)",
        "zones": 1, "fov_crop": 4, "max_range_m": 0.6,
        "noise_floor_m": 0.003, "noise_pct": 0.03,
    },
}


def apply_design(depth_m: np.ndarray, design: str | dict,
                 rng: np.random.Generator | None = None) -> np.ndarray:
    """Degrade recorded depths (any leading dims, trailing (8, 8)) to a design.

    Returns float32 with the identical shape: zones are broadcast back onto
    the 8x8 grid, so (T, S, 4, 8, 8) in means (T, S, 4, 8, 8) out.
    """
    cfg = DESIGNS[design] if isinstance(design, str) else design
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.shape[-2:] != (8, 8):
        raise ValueError(f"expected trailing (8, 8) zone grid, got {depth.shape}")
    lead = depth.shape[:-2]
    flat = depth.reshape(-1, 8, 8)

    k = int(cfg["fov_crop"])
    lo = (8 - k) // 2
    crop = flat[:, lo:lo + k, lo:lo + k]

    z = int(cfg["zones"])
    if z == 1:
        zones = crop.min(axis=(1, 2), keepdims=True)
    elif k % z == 0:
        s = k // z
        zones = crop.reshape(-1, z, s, z, s).min(axis=(2, 4))
    else:
        raise ValueError(f"fov_crop {k} not divisible into {z} zones")

    limit = np.float32(cfg["max_range_m"])
    zones = np.minimum(zones, limit)
    if rng is not None:
        sigma = np.maximum(np.float32(cfg["noise_floor_m"]),
                           np.float32(cfg["noise_pct"]) * zones)
        zones = zones + rng.normal(0.0, 1.0, zones.shape).astype(np.float32) * sigma
        zones = np.clip(zones, 0.0, limit)

    out = np.repeat(np.repeat(zones, 8 // z, axis=1), 8 // z, axis=2)
    return out.reshape(*lead, 8, 8).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--example", required=True,
                    help="a model bundle's example_observation.npz (with proximity)")
    ap.add_argument("--seed", type=int, default=3103)
    args = ap.parse_args()

    with np.load(args.example, allow_pickle=False) as z:
        prox = z["proximity"].astype(np.float32)  # (40, 4, 8, 8) metres
    report = {"input_shape": list(prox.shape),
              "input_min_m": round(float(prox.min()), 4),
              "input_mean_m": round(float(prox.mean()), 4)}
    for name in DESIGNS:
        rng = np.random.default_rng(args.seed)
        out = apply_design(prox, name, rng)
        assert out.shape == prox.shape and out.dtype == np.float32
        report[name] = {
            "min_m": round(float(out.min()), 4),
            "mean_m": round(float(out.mean()), 4),
            "saturated_frac": round(float((out >= DESIGNS[name]["max_range_m"] - 1e-6).mean()), 4),
            "unique_values_per_frame": int(len(np.unique(out[0, 0]))),
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
