# `pla/sim/` — URDF -> MJCF skin pipeline + ToF rendering

## Purpose

Turn a real-world skin design (Blender + GenTact) into a MuJoCo simulation
where every VL53L5CX gets rendered as an 8x8 depth camera, every step. This
is the bridge between hardware design and policy learning.

## What's where

| file                         | what it does                                          |
|------------------------------|-------------------------------------------------------|
| `build_mjcf.py`              | URDF -> MJCF builder; emits one camera per sensor frame |
| `fix_sensor_orientations.py` | post-processes the URDF so sensor +Z faces outward    |
| `patch_mjcf.py`              | patches an existing MJCF with corrected orientations  |
| `tof.py`                     | `ToFSensorArray` + `extend_obs_with_tof` — runtime     |

There are also two top-level scripts:

  * `scripts/build_skin_mjcf.py` — Blender JSON -> MJCF camera bodies.
    Used during the Day-1-2 sensor density iteration loop (export from
    Blender, run, view in MuJoCo, repeat).
  * `scripts/verify_skin.py` — empty-scene self-hit detector. Run after
    every skin change; <5 self-hitting sensors is the bar.

## Sensor model

  * **Hardware:** ST VL53L5CX, 8x8 SPAD multi-zone, 45 deg FOV, 20-4000 mm range.
  * **In MuJoCo:** one fixed `<camera>` per sensor, `resolution="8 8"` and
    `fovy="45"`. The camera body's `+Z` points outward (we apply the
    `quat="0 1 0 0"` flip inside the camera so MuJoCo's default `-Z` view
    direction aligns with the body's outward normal).
  * **Noise:** at runtime (`tof.py`) we add Gaussian sigma=5 mm + 5% per-zone
    dropout to saturated max range, then re-clip. Modeled on the reported
    noise floor of the VL53L5CX.

## Critical: sensor enumeration order

`ToFSensorArray.__init__` walks `range(model.ncam)` in MJCF order. Do **not**
sort by name. The build pipeline is what controls the order; if you change
which group of sensors comes first in the XML, the per-sensor token order
shifts and the trained encoder embeds the wrong sensor in the wrong slot.

The wrist-only ablation depends on this convention: it masks indices 8..31
under the assumption that `link6` sensors are first.

## Day 1 - 3 workflow

```bash
# 1. Blender: design the skin, export sensor_sites.json (see PROJECT.md §2).

# 2. Convert to MJCF camera bodies and merge into the FR3 skin model:
python scripts/build_skin_mjcf.py \
    --sites assets/mjcf/sensor_sites.json \
    --base-mjcf assets/mjcf/fr3_skin.xml \
    --out      assets/mjcf/fr3_skin_blender.xml

# 3. Verify: render every sensor in an empty scene; <5 self-hits.
python scripts/verify_skin.py \
    --mjcf assets/mjcf/fr3_skin_blender.xml \
    --out  reports/checks/skin_verify.json

# 4. Smoke-test the runtime renderer:
python -c "
import mujoco; from pla.sim.tof import ToFSensorArray
m = mujoco.MjModel.from_xml_path('assets/mjcf/fr3_skin_blender.xml')
d = mujoco.MjData(m); mujoco.mj_forward(m, d)
arr = ToFSensorArray(m); print('n_sensors:', arr.n_sensors)
print('one render:', arr.render(d).shape)   # expect (N, 8, 8)
"
```

## Why these design choices

* **MuJoCo cameras over custom raycasting.** MuJoCo's renderer gives correct
  hidden-surface removal for free. Each sensor is just an 8x8 depth render,
  which is essentially the data the VL53L5CX outputs. Custom raycasting
  would require us to implement occlusion logic that already works.
* **Cameras live inside their own body.** This makes the (xyz, rpy) of the
  sensor identical to a URDF joint origin. The patch tool
  (`patch_mjcf.py`) only has to rewrite body pos/quat — the inner
  camera orientation is invariant.
* **5 mm Gaussian noise + 5% dropout.** Both numbers come from the VL53L5CX
  datasheet noise figures. Without dropout the policy learns to expect
  consistent readings everywhere; the real sensor sometimes returns saturated
  max-range values when the SPADs miss. Dropout simulates that and keeps
  the model honest.
* **Re-clip after noise.** Without this, the model can see depths < 20 mm
  or > 4000 mm that the real sensor can't produce — a sim-to-real bug.

## Sanity-check checklist (run before each data collection run)

- [ ] `n_sensors` reported by `ToFSensorArray` matches `n_sensors` in the
      training config.
- [ ] `verify_skin.py` reports <5 self-hits.
- [ ] One rendered batch has values in `[20, 4000]` and no NaN.
- [ ] Depth varies when the robot moves (not all-max, not all-min).
- [ ] Sensor camera order matches what the docstring of `build_mjcf.py`
      promises.
