# Proximity-data Foxglove visualizer

Turns a franka_skin trajectory (`.h5`) into a Foxglove `.mcap` you can scrub through:
the robot, the **29 SPAD proximity sensors back-projected to a live 3D point cloud**,
the camera feeds, the end-effector pose, and the task/phase log.

## Install (already done in the `mlspaces` env)

```bash
/opt/conda/envs/mlspaces/bin/pip install mcap foxglove-schemas-protobuf mcap-protobuf-support foxglove-sdk
```

## Export

```bash
# one trajectory
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl /opt/conda/envs/mlspaces/bin/python scripts/foxglove_export.py \
  --h5 assets/prox_learning_data/.../house_33/trajectories_batch_1_of_1.h5 \
  --traj 0 --out house33_traj0.mcap

# every trajectory in an h5 -> mcaps/traj_0.mcap, traj_1.mcap, ...
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl /opt/conda/envs/mlspaces/bin/python scripts/foxglove_export.py \
  --h5 .../trajectories_batch_1_of_1.h5 --traj all --out-dir mcaps/
```

Useful flags: `--far 1.0` (depth→color far clip, m), `--d-max 2.0` (drop returns beyond this),
`--stride N` (every Nth frame), `--no-mesh` (skip the robot mesh for a lighter file),
`--max-frames N` (quick preview).

## View

1. Open <https://app.foxglove.dev> (or the desktop app) → **Open local file** → pick the `.mcap`.
2. Add panels (or load `scripts/foxglove_layout.json` via *Layouts → Import*):
   - **3D** panel. In its settings enable topics:
     - `/tf` (transforms — the whole robot tree)
     - `/robot` (the robot mesh; `SceneUpdate`)
     - `/proximity` (the headline point cloud). Set *Color field* = `red/green/blue` (RGBA) — it's
       pre-colored turbo: **red = near, blue = far**. Point size ~3.
     - `/proximity_by_link` (optional: same points colored by which link's sensor saw them)
     - `/tcp` (end-effector pose)
   - **Image** panel ×2 → `/camera/wrist`, `/camera/exo`
   - **Log** panel → `/task` (task description at t=0, then every policy-phase transition)
3. Press play. Watch surfaces light up around the arm as it moves.

## Topics

| topic | schema | what |
|---|---|---|
| `/tf` | FrameTransforms | every robot link + the 29 sensor frames, per frame |
| `/robot` | SceneUpdate | robot + "skin" mesh, extracted from the compiled MuJoCo model |
| `/proximity` | PointCloud | 29 sensors' 8×8 depth → one world cloud, colored by distance |
| `/proximity_by_link` | PointCloud | same points, colored per link (red=2, teal=3, dark=5, orange=6) |
| `/camera/wrist`, `/camera/exo` | CompressedImage | recorded RGB videos, frame-synced |
| `/tcp` | PoseInFrame | end-effector pose (in the robot base frame) |
| `/task` | Log | task description + policy-phase transitions |

## How it works (why it's correct)

The exporter does **not** trust any stored extrinsics. It loads the `franka_skin` MJCF, replays
the saved joint trajectory (`env_states/articulations/panda`) through `mj_forward`, and reads each
sensor camera's pose directly. The saved 8×8 depth is back-projected with the exact pinhole model
used during collection (45° FOV, MuJoCo GL convention) — the same math validated to sub-cm in
`scripts/datagen/verify_synthetic_scenes.py`. The world frame is robot-base-centric, so the
reconstruction is geometrically exact relative to the arm regardless of which scene the data is from.

---

## Engineer dashboard (foxglove_dashboard.py) — 2026-06-10

Full instrumented view, modeled on a DROID-style ops screen. Export + open:

    python scripts/foxglove_dashboard.py --h5 <batch.h5> --traj 1 --out ep1.mcap
    # Foxglove: Open local file -> ep1.mcap, then Layout -> Import from file
    #   -> scripts/foxglove_dashboard_layout.json

Panels:
- 3D: robot mesh + /tf, proximity point cloud (turbo, red=near), every SPAD's 45-deg FOV
  cone (per-link color), GROUND-TRUTH obstacle boxes from scene_params (protrusion red),
  target start (green) / goal (blue) spheres, TCP axis.
- /camera/exo, /camera/wrist: RGB feeds.
- /skin/link2|3|5|6: per-sensor 8x8 depth tiles, labeled "S<n> <min>m" (red label < 8cm,
  "--" = no return). What each sensor SEES, sensor by sensor.
- Plots: q1..q7, commanded c1..c7, tracking error e1..e7 (deg, the stall-gate signal),
  velocities v1..v7, per-link skin min distance, global min + TCP speed (the speed law,
  visibly coupled), gripper mm + phase id.
- /task log: episode θ summary, phase transitions, SUCCESS/FAIL.

Notes: base pose read from obs/extra/robot_base_pose (works for in-house episodes too);
--mount-z 0.35 for enclosure-era data (old drawer data was 0.58).
