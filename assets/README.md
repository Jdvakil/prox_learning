# `assets/` — robot model + reference renders

```
assets/
├── urdf/               URDF (FR3 + skin) — source of truth for sensor poses
├── mjcf/               MuJoCo XML — built from URDF by `pla.sim.build_mjcf`
└── reference_images/   PNG dumps used for sanity comparisons in viz/
```

## URDF -> MJCF flow

1. `urdf/fr3_full_skin_fixed.urdf` — the corrected URDF after Blender
   sensor placement + post-processing (sensor +Z aligned outward).
2. `python -m pla.sim.build_mjcf --urdf assets/urdf/fr3_full_skin_fixed.urdf
   --out assets/mjcf/fr3_skin_fixed.xml` — build MJCF.
3. `python scripts/verify_skin.py --mjcf assets/mjcf/fr3_skin_fixed.xml`
   — confirm <5 self-hitting sensors.

## What's *expected* to be here vs not

* `urdf/` — checked into git; small.
* `mjcf/` — checked into git; built artifact but cheap to track.
* `reference_images/` — checked into git; tiny PNGs used for asserting
  visual regressions in the viz pipeline.
* **NOT** here: STL / mesh files. They live in the upstream
  `gentact_ros_tools` package; `build_mjcf` resolves them at build time.
