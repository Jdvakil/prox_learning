# Overnight work report — drawer environment + Foxglove visualizer

Two asks from your advisor, both delivered:
1. **Furniture clutter that encapsulates the proximity sensors** ("reaching into a drawer") —
   a *custom* environment, not predefined houses.
2. **A data visualizer for Foxglove.**

Plus a guaranteed safety-net data-collection run so the night wasn't wasted.

---

## 1. Drawer/cabinet "reach-in" environment (custom — no predefined houses)

A hand-authored cabinet cavity. A graspable objaverse object is injected resting inside it,
and the Franka must reach IN to grasp — so the 29 SPAD proximity sensors are flanked by the
cavity walls throughout the approach/grasp. Reuses the stock privileged pick planner + grasp
DB + rollout + h5 saving unchanged.

**Files**
- `submodules/molmospaces/molmo_spaces/data_generation/custom_scenes/cabinet_cavity.xml`
  (+ `_metadata.json` sidecar) — the cavity geometry (bottom + back + 2 sides + low front lip;
  open front/top so both front and top-down grasps stay feasible).
- `submodules/molmospaces/molmo_spaces/tasks/cavity_pick_task_sampler.py` — `CavityPickTaskSampler`.
- `submodules/molmospaces/.../object_manipulation_datagen_configs.py` —
  `FrankaSkinCabinetCavitySmokeConfig` (1 pick, smoke) and `FrankaSkinCabinetCavityConfig` (full).
- `scripts/cavity_scene.py` — standalone geometry builder + encapsulation validator (no datagen needed).

**Run**
```bash
cd submodules/molmospaces
env MUJOCO_GL=egl PYOPENGL_PLATFORM=egl /opt/conda/envs/mlspaces/bin/python \
  -m molmo_spaces.data_generation.main FrankaSkinCabinetCavityConfig
# output -> assets/datagen/cabinet_cavity_v1/
```

**Validation**
- Geometry (scripts/cavity_scene.py): **25/29 sensors active, 24/29 see <30 cm, median proximity
  depth 0.19 m** when the arm is in the cavity (vs ~0.53 m in houses).
- A real collected trajectory: 71 steps, success, all 29 sensors recorded, **74% of proximity
  returns within 30 cm** — i.e. the skin is genuinely exercised, which is the whole point.

**Tuning knobs**
- Cavity size/shape: edit `cabinet_cavity.xml` (and the matching `CAVITY_CENTER`/`CAVITY_INTERIOR`
  in the sampler). Tighter = more encapsulation but lower grasp yield.
- Object variety: `scene_xml_paths=[_CAVITY_XML]*N` — N "houses", one objaverse object each
  (currently 8; raise N and lower `samples_per_house` for more object diversity).
- Per-episode randomization (object XY/yaw + robot base jitter) lives in the sampler — widen the
  `np.random.uniform` ranges for more variety.

**Non-obvious integration facts (solved here, for reference)**
- `scene_dataset="user"` + `task_sampler_config.scene_xml_paths` loads custom MJCF.
- A `<scene>_metadata.json` sidecar (`{"objects":{}}`) is **mandatory** (else a `None` crash).
- Custom scenes have no occupancy map → `place_robot_near` always fails; we set the base pose
  directly (override `_sample_and_place_robot`).
- The pipeline requests the `"ceiling"` scene variant; user mappings only have `"base"` → we map
  every variant to the file (override `_get_dataset_index_map`).
- An injected object spawns floating; `get_supporting_geom` needs it resting → we settle physics
  (isolating the object so the uncontrolled arm doesn't droop).

---

## 2. Foxglove visualizer

`scripts/foxglove_export.py` turns any trajectory `.h5` into a `.mcap` you scrub in
<https://app.foxglove.dev> (Open local file). See `scripts/FOXGLOVE_README.md` for full details;
`scripts/foxglove_layout.json` is an importable panel layout.

Topics: `/tf` (full robot tree), `/robot` (robot+skin mesh), **`/proximity`** (the 29 sensors'
8×8 depth back-projected to one world point cloud, red=near→blue=far), `/proximity_by_link`,
`/camera/wrist`, `/camera/exo`, `/tcp`, `/task` (task text + policy phases).

It replays the saved joints through the MJCF and reads each sensor's pose from forward kinematics
(doesn't trust stored extrinsics), so the cloud is geometrically exact — same back-projection
validated to sub-cm in `scripts/datagen/verify_synthetic_scenes.py`.

**Ready-to-open examples**: `assets/prox_learning_data/foxglove_examples/`
(`cavity_smoke_traj0.mcap` = the new drawer env; `house_{11,33,46}_traj0.mcap` = old house data).

```bash
env MUJOCO_GL=egl PYOPENGL_PLATFORM=egl /opt/conda/envs/mlspaces/bin/python \
  scripts/foxglove_export.py --h5 PATH.h5 --traj all --out-dir mcaps/
```

---

## 3. Data collected overnight
- **`assets/datagen/cabinet_cavity_v1/`** — the new drawer-environment dataset (target ~120,
  filling as of this writing; ~8–16 distinct objects with pose/approach variety).
- **`assets/datagen/overnight_prox_multi_house/`** — safety-net multi-house proximity-necessity
  dataset (`FrankaSkinProxOvernightConfig`), accumulating in parallel.

## 4. Suggested next steps
- Eyeball `cavity_smoke_traj0.mcap` in Foxglove to confirm the look matches what the advisor wants.
- Decide cavity tightness / object count for the "real" collection, then scale
  `FrankaSkinCabinetCavityConfig` (more `scene_xml_paths`, more workers — the box has 48 CPUs).
- Then we can move to the ACT/model side on this data.
