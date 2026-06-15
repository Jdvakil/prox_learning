# Proximity Learning Architecture (PLA)

Sim-and-real pipeline for training a **Proximity Learning Architecture** — a
manipulation policy that fuses VLM-derived task grounding with low-resolution,
high-rate, body-distributed time-of-flight depth from a sensor skin on the
Franka FR3.

The headline experiment for the CoRL 2026 deadline is **PLA vs VLM-only ACT**:
the same VLM front-end and ACT policy backbone, with and without the proximity
skin stream, on a sweep of Franka manipulation tasks across iTHOR /
ProcTHOR-derived household scenes.

This top-level repo is a thin shell that pins:

- `assets/` — robot MJCF and shared scene/object assets.
- `submodules/molmospaces/` — the data-generation, simulation, and rendering
  stack (forked / extended).
- `submodules/MolmoBot/` — the policy / molmo VLM integration (training,
  inference).
- `synthetic_verify/`, `proximity_inspect/` — verification artifacts produced
  while validating the proximity sensor pipeline (kept in-tree for repro).

The remainder of this README documents the **proximity sensor stack** end to
end: model, simulator integration, verification protocol, and the substantive
fixes that landed during validation.

---

## 1. The franka_skin robot

`assets/robots/franka_skin/model.xml` is a Franka FR3 with:

- The standard FR3 collision and visual meshes (link0 .. link7).
- A Robotiq 2F-85 gripper (`robotiq_2f85_v4/2f85.xml`).
- A **sensor skin** wrapping link2, link3, link5, link6 — defined as
  decorative `class="skin"` geoms (group=2, contype=0, conaffinity=0,
  mass=0). The skin is purely visual; it does not collide and has no inertia.
- **29 SPAD-style proximity sensors** distributed across the four skinned
  links:

| Link | # sensors |
|------|-----------|
| link2 | 7 |
| link3 | 8 |
| link5 | 6 |
| link6 | 8 |
| **Total** | **29** |

Each sensor is realized in MJCF as a body whose pose places it just above the
collision mesh at the desired skin location, with two children:

- A `<site class="skin_sensor_site">` (red sphere, group=2) — visual marker
  for the sensor location.
- A `<camera mode="fixed" pos="0 0 0" quat="0 0 1 0" fovy="45.0"
  resolution="8 8"/>` — the 8x8 proximity sensor itself.

The `quat="0 0 1 0"` is a 180° rotation about the body's local Y axis, which
preserves the outward viewing direction (along the body's +Z) while orienting
the rendered image so "up" in the image corresponds to the conventional ToF
top edge. (See §5 for why this changed from `0 1 0 0`.)

Sensor body positions and orientations were imported from the original Isaac
Sim USD placements; the per-sensor `pos` / `quat` on each `<body>` is what
defines the geometry of the array.

---

## 2. SPAD proximity sensor model

A single proximity sensor models an array-style time-of-flight depth chip
(loosely a VL53L8CX class part):

| Property | Value |
|----------|-------|
| Resolution | 8 × 8 |
| Field of view | 45° (HFOV = VFOV, square pixel grid) |
| Range | 0.05 – 4.0 m |
| Sample rate | 60 Hz (sub-stepped within the policy step) |
| Output channel | depth only (no RGB at training time) |

Because all 29 sensors share these parameters and only differ in body pose,
they are generated programmatically in
`molmo_spaces/configs/camera_configs.py::_skin_sensor_camera_specs()` (29
`MjcfCameraConfig(..., record_depth=True, is_proximity_sensor=True)` entries).

`is_proximity_sensor=True` is the flag the env layer uses to:

- skip RGB rendering for that camera,
- record depth at `proximity_sensor_period_ms` (default 16.67 ms ≈ 60 Hz),
  sub-stepping multiple frames per policy step,
- route through the dedicated 8x8 depth renderer rather than the global RGB
  renderer.

---

## 3. How proximity depth is rendered

Implemented in
`submodules/molmospaces/molmo_spaces/env/env.py::record_proximity_depths`.

Two non-obvious design decisions, both load-bearing:

### 3a. Dedicated 8×8 renderer, not the global one

The global Mujoco renderer is sized at the RGB camera resolution (e.g.
624 × 352, non-square). If we asked it to render an `fovy=45°` proximity
camera, the *vertical* FOV is 45° but the *horizontal* FOV is determined by
the aspect ratio — we'd get HFOV ≈ 72°, ~1.6× wider than the SPAD spec, and
the depth pattern would be wrong.

The fix is a separate `mujoco.Renderer(model, height=8, width=8)` used only
for proximity sensors. With a square 8×8 viewport, HFOV = VFOV = 45° as
intended.

### 3b. Hide the skin during the proximity render

The skin is a decorative shell (group=2) sitting *outside* the collision
mesh; the proximity sensor bodies are positioned *inside* that shell (above
the collision mesh, embedded in the skin volume). With the skin visible,
each sensor sees its own skin at near-zero distance and reports the wrong
thing.

The fix:

```python
self._proximity_scene_option = mujoco.MjvOption()
mujoco.mjv_defaultOption(self._proximity_scene_option)
self._proximity_scene_option.geomgroup[2] = 0  # hide skin
```

passed via `update_scene(..., scene_option=self._proximity_scene_option)`.
With group 2 hidden the renderer only considers the visual+collision links
of the robot and the surrounding scene — exactly what a real ToF chip
mounted on the skin would see.

### 3c. Sub-step buffer

Inside one policy control step, `record_proximity_depths(camera_names)` is
called multiple times (the substep loop), each call appending one 8×8
depth frame per active sensor to `self._proximity_depth_frames[camera]`.
`reset_proximity_depth_buffer()` clears the buffer at the start of each
policy step. `ProximityDepthBufferSensor` consumes the per-step list and
emits the trajectory time-series.

---

## 4. Verification protocol

We were not willing to start large-scale data collection without a
quantitative *and* visual sign-off that the 29 sensors see the world
correctly. The verification suite lives in
`submodules/molmospaces/scripts/datagen/`:

### 4a. Empty-room and flat-plane geometry tests

`verify_synthetic_scenes.py` and `verify_proximity_gt.py` place the
franka_skin robot in synthetic rooms (empty box, flat ground plane) and
compare the rendered 8×8 depth against ground-truth `mj_ray` casts (which
ignore back-face culling and rasterization quantization).

Per-sensor outputs:

- `synthetic_verify/empty_room/` — point clouds, error histograms, per-axis
  bias, GT-vs-rendered overlays.
- `synthetic_verify/flat_plane/` — quantization analysis as a function of
  sensor resolution; documents the **−44.6 mm floor bias at 8×8**
  (rasterization-driven, scales as 1/H, expected, not a bug).

### 4b. mj_ray independent classifier

A pure ray-cast diagnostic that bypasses the renderer and reports, per
sensor, whether the *body name* hit by the central ray is:

- the parent link's collision mesh (sensor pointing inward — placement bug),
- another robot link (sensor pointing across the arm — usually OK),
- the world / scene (sensor pointing outward — correct),
- nothing within range (free space).

Result: 24 / 29 sensors are correctly placed and oriented. 5 sensors
(link2_1, link2_2, link2_4, link3_0, link3_7) have a nominal viewing
direction that points toward their parent link's collision geom. With the
skin culled (§3b) and back-face culling on the collision mesh, the renderer
recovers an environment-derived depth for those rays anyway, so they are
usable in the production pipeline. The placement bug is documented and can
be fixed in a future skin revision without changing any code.

### 4c. Hi-res RGB inspection (one-off)

To go beyond histograms — an actual visual check of what every sensor sees
during a real episode — we ran a one-shot inspection mode that rendered an
extra 256×256 RGB frame from each proximity camera at every substep,
matching the depth render exactly (same camera pose, same skin-hidden scene
option). Outputs were 29 per-sensor MP4s plus a 29-tile grid MP4 over the
full episode in `assets/datagen/pick_planner_v1/inspect_prox_*/proximity_rgb/`.

The inspection MP4s confirmed:

- No sensor sees the gripper, its own link, or its own skin in normal
  motion.
- Sensor frames track the iTHOR house_1 kitchen geometry as the arm moves
  (cabinets, countertop, plants, windows visible from the appropriate
  link6 sensors).
- Image orientation was correct after the §5 quat fix.

The inspection-mode code was a verification-only path. **It has been
removed from `env.py` for production** so the env runs the canonical 8×8
depth path and nothing else (see §6).

---

## 5. Substantive fixes that landed during validation

Documented here so they don't get lost in git archaeology.

### 5a. Skin self-occlusion (the big one)

**Symptom:** in initial verification runs, ~33 % of rendered depths in an
empty room came back as `~0`, classified as self-hits.

**Root cause:** the skin shell (group=2) was visible to the proximity
renderer, and sensor bodies sit *inside* the skin volume. Every sensor saw
its own skin at distance ≈ 0.

**Fix:** hide group 2 via `MjvOption.geomgroup[2] = 0` on the proximity
renderer's scene option (see §3b). Empty-room self-hit rate dropped to
near zero for the 24 well-placed sensors.

### 5b. Failed `cx` shift, then reverted

**Symptom:** a residual −44.6 mm floor bias at 8×8 against a flat ground
plane.

**Hypothesis tried:** off-center principal point — moved `cx` from
`(W-1)/2 = 3.5` to `3.0` to "compensate" for what looked like a half-pixel
asymmetry.

**Result:** self-hit classification *increased* from 33.3 % → 36.8 %. The
shift introduced a lateral misalignment in the back-projected point cloud
without fixing the bias.

**Resolution:** reverted to `cx = (W-1)/2 = 3.5`. The −44.6 mm floor bias is
**rasterization quantization** that scales as `1/H` (verified by re-running
flat-plane tests at multiple resolutions). It is not a calibration bug; it
is a property of finite-resolution rendering and is small relative to the
SPAD's depth noise floor.

### 5c. Image orientation flip

**Symptom:** in the hi-res RGB inspection MP4s, scene content appeared
upside-down relative to the wrist / exo cameras.

**Root cause:** all 29 cameras had `quat="0 1 0 0"` — a 180° rotation about
local X. That preserves the viewing direction (still down +Z out of the
body) but flips the image vertically.

**Fix:** changed all 29 camera quats to `quat="0 0 1 0"` — a 180° rotation
about local Y. Same viewing direction, image right-side-up (and
horizontally mirrored vs the buggy state, which is the correct ToF
convention). Backup of the pre-fix model lives at
`assets/robots/franka_skin/model.xml.bak_before_orientation_fix`.

### 5d. Site-packages vs in-tree `molmo_spaces` resolution

**Symptom:** edits to `submodules/molmospaces/molmo_spaces/env/env.py`
weren't visible to the data-generation pipeline at runtime.

**Root cause:** the MolmoBot venv had both an old site-packages copy of
`molmo_spaces` (from a prior install) *and* the in-tree submodule. Python
was importing the site-packages copy.

**Fix at runtime:** invoke the pipeline with the in-tree path on
`PYTHONPATH` and the in-tree dir as cwd, e.g.

```bash
cd submodules/molmospaces \
  && PYTHONPATH=. \
       /home/jaydv/code/prox_learning/submodules/MolmoBot/MolmoBot/.venv/bin/python \
       scripts/datagen/run_pipeline.py [args...]
```

This is the canonical invocation pattern; the pipeline itself was not the
problem.

### 5e. Production cleanup

After visual sign-off, the inspection-mode patches were removed from
`env.py`. The fields `_proximity_inspect_resolution`,
`_proximity_inspect_renderer`, and `_proximity_rgb_frames`, the method
`enable_proximity_rgb_inspection`, and the conditional inspection branch
inside `record_proximity_depths` are all gone. `record_proximity_depths`
now does exactly one thing: render 8×8 depth from each proximity camera
with the skin hidden, append to the substep buffer, return.

The standalone helper scripts that were only useful during validation —
`scripts/datagen/inspect_proximity_rgb.py` and the failed
`scripts/datagen/fix_sensor_orientations.py` — have been deleted. The
analytical verifiers (`verify_synthetic_scenes.py`,
`verify_proximity_gt.py`, `visualize_proximity.py`,
`analyze_sample_episode.py`) remain.

---

## 6. Production data collection

Two registered configs in
`submodules/molmospaces/molmo_spaces/data_generation/config/object_manipulation_datagen_configs.py`
drive the franka_skin pick-and-place pipeline:

| Config name | Purpose | Scope |
|-------------|---------|-------|
| `FrankaSkinPickAndPlaceDataGenConfig` | Production class. | iTHOR / train, defaults from `PickAndPlaceTaskSamplerConfig` (`samples_per_house=20`, `house_inds=range(0,4)`). Subclass or override at the call site for the full sweep. |
| `FrankaSkinPickAndPlacePilotConfig` | 10-house pilot. Subclass of the prod config. | iTHOR / train, `house_inds=1..10`, `samples_per_house=4`, `seed=2026`. |

Both wire `FrankaSkinRobotConfig` + `FrankaSkinCameraSystem` (the 29 SPAD
sensors plus the standard exo / wrist RGB cameras) and use the planner
policy via the `PickAndPlaceDataGenConfig` base class.

Canonical invocation (named-config style — same entry point as the rest of
molmospaces):

```bash
cd /home/jaydv/code/prox_learning/submodules/molmospaces

PYTHONPATH=. \
  /home/jaydv/code/prox_learning/submodules/MolmoBot/MolmoBot/.venv/bin/python \
  -m molmo_spaces.data_generation.main FrankaSkinPickAndPlaceDataGenConfig
```

Output trajectory bundles contain, per timestep:

- Exo / wrist RGB streams.
- For each of the 29 proximity sensors, a list of 8×8 float32 depth frames
  recorded at 60 Hz across the policy step
  (`ProximityDepthBufferSensor`).
- The usual proprioception / action / language / object channels.

Trajectories land under
`assets/experiment_output/datagen/pick_and_place_skin_v1/house_<i>/` and are
combined into a training H5 with
`scripts/datagen/combine_trajs_into_h5.py`.

> **Legacy CLI alternative.** The flag-driven entry point still works:
> `scripts/datagen/run_pipeline.py --robot skin --task_type pick_and_place
> --scene_dataset ithor --data_split train --house_inds <int>
> --samples_per_house <N> --seed <S> --run_name_prefix <name>`.
> Use it when you need quick CLI overrides; otherwise prefer the named
> config.

### 6.1. Pilot run (10 iTHOR houses, pick-and-place)

Before committing compute to the full sweep, run the 10-house pilot to
surface any scene-dependent failures (sensor clipping into thin walls,
iTHOR geometry edge cases, planner timeouts on the new task family). Task
matches the molmobot evaluation task: **pick_and_place**.

Pilot parameters (baked into `FrankaSkinPickAndPlacePilotConfig`):

| Field | Value | Why |
|-------|-------|-----|
| `robot_config` | `FrankaSkinRobotConfig` | franka_skin (29 SPAD sensors). |
| `camera_config` | `FrankaSkinCameraSystem` | Skin proximity + exo/wrist RGB. |
| `task_type` | `pick_and_place` | Matches the molmobot evaluation task. |
| `scene_dataset` | `ithor` | iTHOR train kitchens (verified scene_1). |
| `data_split` | `train` | Train split, disjoint from eval. |
| `task_sampler_config.house_inds` | `1..10` | 10-house pilot. |
| `task_sampler_config.samples_per_house` | `4` | ≈40 successful episodes target. |
| `seed` | `2026` | Reproducible. Bump per re-run. |
| `output_dir` | `assets/experiment_output/datagen/pick_and_place_skin_pilot_v1` | Tagged so it doesn't collide with prod runs. |

Launch:

```bash
cd /home/jaydv/code/prox_learning/submodules/molmospaces

mkdir -p logs

PYTHONPATH=. \
  /home/jaydv/code/prox_learning/submodules/MolmoBot/MolmoBot/.venv/bin/python \
  -m molmo_spaces.data_generation.main FrankaSkinPickAndPlacePilotConfig \
  2>&1 | tee logs/pilot_skin_pickplace_v1.log
```

The pipeline iterates the 10 houses internally (no bash loop needed) and
writes per-house subdirs under
`assets/experiment_output/datagen/pick_and_place_skin_pilot_v1/house_<i>/`.

Pilot acceptance checklist (before scaling up):

- [ ] Per-house success rate ≥ ~50 % (planner solving pick-and-place in the
      kitchen). Below that, debug the planner on iTHOR pick-and-place before
      scaling.
- [ ] No crashes from the proximity render path (grep
      `logs/pilot_skin_pickplace_v1.log` for `native 8x8 render failed` —
      should be zero).
- [ ] Spot-check 1–2 trajectories with `scripts/datagen/visualize_proximity.py`
      / `analyze_sample_episode.py` to confirm depth streams look sane on
      different houses (different geometry from house_1).
- [ ] Combined H5 builds cleanly via `combine_trajs_into_h5.py`.

If all four pass, the full sweep is just running
`FrankaSkinPickAndPlaceDataGenConfig` directly (or registering a wider-house
subclass — same pattern as the pilot).

### 6.2. Low-surface pick-and-place collection (proximity-skin showcase)

The first pilot ran cleanly but the resulting dataset is dominated by
**tabletop** episodes: the default `PickAndPlaceTaskSampler` picks any
object on any supporting surface, and procthor-objaverse / iTHOR scenes are
saturated with `CounterTop` and `DiningTable` candidates. That distribution
under-exercises the proximity skin — the wrist sensors see open space the
whole way to the grasp.

To bias collection toward the cases where the skin actually pays off
(reaching down into a sink, into a low shelf, onto a chair / stool /
sofa / bed / bathtub / toilet / dresser / chest-of-drawers), three things
landed in the molmospaces submodule:

1. A new sampler config field
   `PickAndPlaceTaskSamplerConfig.source_surface_types` (case-insensitive
   prefix tuple, default `()`).
2. An override of `_get_scene_objects` on
   `PickAndPlaceReceptacleTaskSampler` that filters candidate pickups by
   walking the supporting geom's body parent chain (up to 3 ancestors) and
   keeping only objects whose support body name starts with one of the
   requested prefixes. Logs `[source_surface_types] kept N/M candidates by
   prefix; counts={...}` per scene so per-prefix yield is visible. When
   the filter empties the candidate list it raises `HouseInvalidForTask`,
   not an assertion, so the worker advances cleanly to the next house.
3. Two registered configs in `object_manipulation_datagen_configs.py`
   that wire the filter:

| Config name | Purpose | Scope |
|-------------|---------|-------|
| `FrankaSkinLowSurfacePickAndPlaceDataGenConfig` | Production class with the low-surface filter. | procthor-objaverse / train, `house_inds=range(1999)`, `samples_per_house=5`, `num_workers=4`, `source_surface_types=LOW_SURFACE_PREFIXES`. |
| `FrankaSkinLowSurfacePickAndPlacePilotConfig` | Quick pilot subclass. | Same dataset, `house_inds=range(200)`, `samples_per_house=3`. |

`LOW_SURFACE_PREFIXES = ("sink", "shelf", "bookshelf", "chair", "armchair",
"stool", "sofa", "bed", "bathtub", "toilet", "crapper", "dresser",
"chestofdrawers")`. `crapper` and `chestofdrawers` are included because
that's how the procthor-objaverse XMLs name those bodies.

Output goes to
`assets/datagen/pick_and_place_skin_low_surface_v1/` (prod) and
`assets/datagen/pick_and_place_skin_low_surface_pilot_v1/` (pilot), kept
separate from the tabletop-dominated v1 dataset.

#### Robustness fixes that also landed

These came out of the first low-surface pilot run and apply to any
pick-and-place datagen, not just the low-surface variant:

- **Worker self-termination guard relaxed.** `max_allowed_sequential_irrecoverable_failures=10000`
  on both pilot and prod skin configs. The default of 5 was treating
  "house exhausted its candidate pool after some successes" as
  irrecoverable and exiting workers after only 5 productive houses each.
  The first 47-episode pilot was actually a 2-worker × 5-house = 10-house
  cap, not a real `samples_per_house` cap.
- **`_configure_pick_and_place` assertion → raise.** The base
  `assert self.candidate_objects is not None and len(self.candidate_objects) > 0`
  in `pick_and_place_object_target_task_sampler.py` now raises
  `HouseInvalidForTask` instead, so a drained pool advances to the next
  house instead of crashing the worker.
- **Worker tracebacks logged to file.** All three
  `traceback.print_exc()` calls in
  `molmo_spaces/data_generation/pipeline.py` were going to stderr only
  (the `worker_stdout_context` redirect is a no-op in this build) and so
  worker errors never reached `running_log.log`. Replaced with
  `worker_logger.error/warning(... + traceback.format_exc())` so every
  task-sampling, rollout, and save error now shows up in the per-run
  `running_log.log`. Critical for after-the-fact debugging — without it,
  rerun-and-pray was the only diagnostic loop.

#### How to launch

```bash
cd /home/jaydv/code/prox_learning/submodules/molmospaces
/opt/conda/envs/mlspaces/bin/python molmo_spaces/data_generation/main.py FrankaSkinLowSurfacePickAndPlacePilotConfig
# once the pilot looks healthy:
/opt/conda/envs/mlspaces/bin/python molmo_spaces/data_generation/main.py FrankaSkinLowSurfacePickAndPlaceDataGenConfig
```

Watch for `[source_surface_types] kept N/M ...` lines in the per-worker
log to gauge how many scenes have qualifying surfaces. If yield is too
low, narrow or widen `LOW_SURFACE_PREFIXES`.

#### Pre-flight: scene cache must be populated

`assets/scenes/<dataset>/*.xml` are symlinks into
`~/.cache/molmo-spaces-resources/scenes/...`. If the cache is wiped (e.g.
disk pressure cleanup) the symlinks dangle silently and **every house
fails with `ParseXML: Error opening file`**. Symptom: a healthy-looking
log dump followed by hundreds of
`HouseInvalidForTask: Scene setup failed during compilation` warnings
and `Completed 0 houses, skipped N houses` at the end.

Verify before launching a long run:

```bash
ls -L /home/jaydv/code/prox_learning/assets/scenes/procthor-objaverse-train/train_0.xml
# should print a non-zero file size, NOT "No such file or directory"
```

If the cache is gone, repopulate via the molmospaces HF download:

```bash
cd /home/jaydv/code/prox_learning/submodules/molmospaces
/opt/conda/envs/mlspaces/bin/python scripts/assets/hf_download.py
```

Or fetch a specific scene/variant:

```bash
/opt/conda/envs/mlspaces/bin/python scripts/datagen/fetch_assets.py \
    scene procthor-objaverse <idx> --variant ceiling
```

Note that `task_sampler.sample_task` defaults to `variant="ceiling"`, so
both the base and ceiling XMLs need to be present for the scene to load.

#### Why per-house yield varies

`samples_per_house=N` is a target ceiling, not a quota. The pick-and-place
sampler keeps trying tasks in a house until it collects `N` successes or
the candidate pool is exhausted via `_remove_candidate_object` (called on
supporting-geom failure, robot-placement error, ≥2 grasp failures, or
`_on_candidate_selected` ValueError). Houses with few qualifying objects
will produce fewer episodes than houses with many — this is the data
distribution, not a bug. The first pilot's
`{house_4: 1, house_5: 10, house_8: 2, ...}` spread was natural variance
in candidate-pool size after filtering.

---

## 7. Repo layout (top level)

```
prox_learning/
├── assets/
│   ├── robots/franka_skin/        # franka_skin MJCF + skin meshes (the model)
│   ├── robots/franka_fr3/         # baseline FR3 (no skin)
│   ├── robots/...                 # other robots (yam, rby1, ...)
│   └── datagen/                   # rollout outputs + inspection runs
├── synthetic_verify/              # quantitative GT-vs-render verification
│   ├── empty_room/
│   ├── flat_plane/
│   └── summary.md
├── proximity_inspect/             # ad hoc proximity inspection artifacts
├── pointcloud.ipynb               # interactive proximity point-cloud notebook
├── submodules/
│   ├── molmospaces/               # data-gen + sim + rendering stack (extended)
│   └── MolmoBot/                  # VLM + policy training
├── pyproject.toml
└── README.md                      # this file
```

---

## 7.5. PLA training pipeline (`pla/`)

The headline CoRL 2026 ablation — **PLA (with proximity) vs VLM-only ACT
(without proximity)** — is implemented as a thin training/eval stack on top
of the `submodules/act` ACT model. Both variants share an identical CVAE +
transformer backbone; the only difference is whether 29 proximity tokens
are inserted into the transformer encoder context. See `TODO.md` for the
spec this stack implements.

Layout:

```
pla/
├── dataset.py            # PyTorch dataset over HDF5 trajectories + sibling MP4s
├── proximity_encoder.py  # shared MLP: (B, 29, 8, 8) → (B, 29, 512)
├── policy.py             # PLA_DETRVAE + PLAPolicy (loss + image normalization)
├── eval_policy.py        # InferencePolicy wrapper for molmospaces eval
├── train.py              # CLI entry: training loop with WandB + ckpting
├── eval.py               # CLI entry: 200-ep benchmark eval, Wilson 95% CI
└── diagnostics.py        # dataset sanity-check plots + summary.json
```

### 7.5.a Dataset format

Each `house_<i>/trajectories_batch_*.h5` holds N trajectories under
`traj_<i>/`. Per-trajectory layout:

| Path | Shape / type | Notes |
|------|--------------|-------|
| `obs/proximity/<sensor>` | `(T, n_substeps, 8, 8)` float32 | 29 sensors named `link{2,3,5,6}_sensor_{i}`; mean-pooled over substep dim. **Substep dim ≡ 1 only when `proximity_sensor_period_ms=0`, which silently disables recording (see §6.1 + the dataset bug memory).** |
| `obs/agent/qpos[t]` | `(T, 2000)` uint8 | JSON `{"arm":[7], "base":[], "gripper":[2]}` rows; we read `arm`. |
| `actions/joint_pos[t]` | `(T, 2000)` uint8 | JSON `{"arm":[7], "gripper":[1]}` rows; we read `arm`. |
| `obs_scene` | scalar bytes | JSON+pickle blob; `task_description` is the language string. |
| `success` / `fail` | `(T,)` bool | Episode is "successful" if `success[-1] == True`. |
| Sibling MP4s | per-camera | `episode_<00000000+i>_<cam>_batch_1_of_1.mp4` for `exo_camera_1`, `wrist_camera`. Read via `decord` at the chunk-start timestep. |

`pla.dataset.FrankaSkinHDF5Dataset` indexes one sample per timestep (the
canonical ACT schedule). Items: `proximity (29, 8, 8) ∈ [0,1]`, `qpos (7,)`,
`action (k=100, 7)`, `is_pad (k,)`, `image (num_cam, 3, H, W)`,
`language (str)`. The `use_proximity=False` mode zeroes the proximity tensor
so the same loader feeds both the PLA and the VLM-only baseline.

### 7.5.b Policy

`pla.policy.PLA_DETRVAE` mirrors upstream `detr_vae.DETRVAE` and reuses its
`Transformer`, `Backbone`, CVAE encoder, and sinusoidal positional tables
without modification. The single addition is a `ProximityEncoder`
(`Linear(64→128) → ReLU → Linear(128→512)`, weights shared across the 29
sensors per TODO §2) and 29 extra slots in `additional_pos_embed`. The
encoder context becomes `[latent_z, qpos, *29 proximity tokens, *image
tokens]` when `use_proximity=True`; it falls back to the upstream
`[latent_z, qpos, *image tokens]` when False.

Default hyperparameters (TODO §3): `chunk_size=100`, `hidden_dim=512`,
`enc_layers=dec_layers=7`, `kl_weight (β) = 10`. Param counts (verified):
**PLA 96.37M, baseline 96.28M** — the proximity branch is ~90k params
(the encoder + 29 extra positional embeddings).

### 7.5.c Train / eval

```bash
# Smoke runs (CPU/GPU sanity)
python -m pla.dataset <dataset_root>            # prints a sample dict
python -m pla.proximity_encoder                 # prints param count
python -m pla.policy                            # 1-step forward+backward

# Full training (per TODO §5)
python -m pla.train --use_proximity false --run_name vlm_only_act
python -m pla.train --use_proximity true  --run_name pla_v1

# Diagnostics on the dataset (gates training)
python -m pla.diagnostics --root <dataset_root> \
                          --out pla/diagnostics_output/<run_id>

# Eval on FrankaPickandPlaceHardBench (200 episodes, procthor-objaverse val)
BENCH_DIR=~/.cache/molmo-spaces-resources/benchmarks/molmospaces-bench-v2/20260415/procthor-objaverse/FrankaPickandPlaceHardBench/FrankaPickandPlaceHardBench_20260212_200ep_json_benchmark
python -m pla.eval --checkpoint runs/pla_v1/latest.pt \
                   --benchmark_dir $BENCH_DIR \
                   --run_name pla_v1 --max_episodes 200
```

Eval writes `eval_output/<run_name>/results.json` with
`{success_count, total_count, success_rate, wilson_95_ci}`.

### 7.5.d Open problems

- ~~**Gripper is not predicted by the network.**~~ **Resolved 2026-05-10**:
  `pla.dataset.FrankaSkinDatasetConfig.action_dim` defaults to 8 (7 arm joints
  + 1 normalized gripper, where the raw binary command `{0.0, 255.0}` is
  rescaled to `{0, 1}` for L1 compatibility with arm magnitudes). The eval
  policy predicts gripper directly and snaps back to `{0, 255}` for the
  controller via a threshold (`gripper_threshold=0.5`).
- ~~**Pilot dataset has zero proximity values.**~~ **Resolved 2026-05-10**:
  `proximity_sensor_period_ms` was fixed in
  `FrankaSkinPickAndPlaceDataGenConfig` (0.0 → 16.6667 ms ≡ 60 Hz). Smoke
  re-collection on 10 houses × 4 samples (`FrankaSkinPickAndPlacePilotSmokeConfig`,
  seed=2026) produced 36/36 successful trajectories with 99.94% nonzero
  proximity pixels, per-sensor mean ~1-3 m and max ~4 m clipped (see §7.5.e).
- **Eval is currently blocked: JsonBenchmark schema does not support 8×8
  proximity sensors.** Two architectural mismatches surfaced this round.
  - *First:* the cached `FrankaPickandPlaceHardBench_20260212_200ep_json_benchmark`
    is built for `franka_droid` (DROID-randomized cameras, no SPAD sensors).
    `camera_config_override` doesn't recover it because per-episode JSON
    re-installs DROID cameras.
  - *Second (the deeper blocker):* even after building a `franka_skin`
    JsonBenchmark from held-out houses 11-20 via
    `scripts/benchmarks/create_json_benchmark.py` (35 episodes generated, right
    robot, right camera names), the eval still fails. The benchmark
    `CameraSpec` schema at
    `submodules/molmospaces/molmo_spaces/evaluation/benchmark_schema.py:58-83`
    only stores `name / type / reference_body_names / camera_offset /
    lookat_offset / camera_quaternion / fov / record_depth`. It has **no
    per-camera resolution** and **no `is_proximity_sensor` flag**. A single
    global `img_resolution: tuple[int, int]` governs every camera in the
    episode. At eval time all 31 cameras (including the 29 SPAD sensors)
    render RGB at the benchmark's `[624, 352]`, so `obs["link2_sensor_0"]`
    comes back shaped `(352, 624, 3)` instead of `(8, 8)`. Our policy raises
    `ValueError: could not broadcast input array from shape (624,3) into shape (8,8)`.
    Three forward paths, in increasing rigor:
    1. **Extend the JsonBenchmark schema** (`CameraSpec`) with per-camera
       `resolution: tuple[int,int] | None` and `is_proximity_sensor: bool`,
       propagate through `camera_manager.py` setup, and re-run
       `create_json_benchmark.py`. ~Half-day upstream change.
    2. **Custom rollout script** that bypasses `JsonEvalRunner`: iterate
       `benchmark.json` episodes, set up env with the un-clobbered
       `FrankaSkinCameraSystem`, run policy, check success. Re-uses task
       specs but reads cameras from our config.
    3. **Re-train on DROID-camera data** so we can use the cached
       `FrankaPickandPlaceHardBench` directly. Throws away proximity entirely
       (DROID datagen has no SPAD sensors), so contradicts the project goal.

  Decision deferred (see README §7.5.e and TODO §6). Held-out 35-episode
  benchmark is intact at
  `assets/eval_subsets/FrankaSkinPickAndPlaceHoldout_v1/`; usable once a
  rollout path is built.
- **Language conditioning is not yet wired in.** The dataset returns
  `task_description` per episode; the policy does not consume it. A Molmo
  VLM token branch is the natural next step, deferred until the held-out
  eval shows a positive proximity signal.

### 7.5.e First validation round (2026-05-10 / 2026-05-11)

End-to-end run on the 36-trajectory smoke dataset to validate the entire
pipeline before committing compute to a 100-house+ pilot.

**Smoke dataset (36 trajectories, 9430 timesteps):**

| Quantity | Value | Source |
|----------|-------|--------|
| Success rate | 36/36 (100%) | data-gen |
| Episode length | 224-301 steps (μ=262) | `diagnostics_output/.../summary.json` |
| Proximity nonzero pixel fraction | 99.94% | per-sensor histogram |
| Per-sensor mean depth | 1-3 m | `02_proximity_per_sensor_stats.png` |
| Proximity overall max (post-clip to 4 m) | 4.000 m | clipping verified |
| Unique task descriptions | 35 / 36 episodes | language coverage plot |
| Q1 plausibility (frac in [0.05, 4.0] m) | 87.2% | `analyze_sample_episode.py` |
| Q2 temporal variance (per-sensor) | 0.50 m² mean | passes |
| Q3 phase-correlated signal | 102% variance explained by phase | passes |
| Q4 schema (29 sensors, 31 cam params, 2 RGB MP4, 2 depth MP4) | all present | passes |
| Pointcloud reconstructed (one traj) | 1.79M world-frame points | `pointcloud.ply` |

**Diagnostics + pointcloud artifacts**:
`diagnostics_output/pilot_skin_smoke_v1/episode_house2_traj0/` contains 9 PNGs
(per-sensor heatmap, sensor-grid panel, 3D pointcloud projections, RGBD samples,
qpos/action distributions, language coverage, episode-length histogram) +
`pointcloud.ply` + `report.md`. Run `python submodules/molmospaces/scripts/datagen/analyze_sample_episode.py <h5> --traj traj_0 --out <dir>` to regenerate
for any other trajectory.

**Training (20 000 steps, batch=8, lr=1e-5, num_workers=2):**

| Run | use_proximity | params | start loss | end loss | end L1 | end KL | throughput |
|-----|---------------|--------|------------|----------|--------|--------|------------|
| `smoke_pla_v3_full` | True | 96.37 M | 12.21 (step 50) | **0.0619** | 0.0321 | 0.0030 | 19.3 samp/s |
| `smoke_vlm_only_act_v3_full` | False | 96.28 M | 12.40 (step 50) | **0.0689** | 0.0390 | 0.0030 | 21.7 samp/s |

PLA's final loss is 10% lower (0.062 vs 0.069) and L1 is 17% lower (0.032 vs
0.039). At this data scale the CVAE prior collapses (`KL → 0.003`), which is
expected; the meaningful signal is the L1 gap. Both runs are on WandB
(`project=pla`, tags `backfill,smoke,validation_round`); backfilled from the
local logs via `scripts/backfill_wandb_from_log.py`. Direct links:
- PLA: <https://wandb.ai/jayluvsgeography/pla/runs/731wnt1d>
- Baseline: <https://wandb.ai/jayluvsgeography/pla/runs/gjl5aijc>

**Eval — JsonBenchmark route blocked, custom rollout completed.**

The `JsonEvalRunner` route was abandoned after discovering that the
`CameraSpec` schema has no per-camera resolution or `is_proximity_sensor`
flag, so the 29 SPAD sensors render as 624×352 RGB instead of 8×8 depth.
See §7.5.d for the three upstream-fix paths.

Instead, `pla/rollout_eval.py` was built: a thin wrapper around the
data-generation pipeline (`FrankaSkinPLARolloutConfig`) that uses the
correct `FrankaSkinCameraSystem` (preserving 8×8 SPAD depth) and swaps
the planner for `PLAInferencePolicy`. Tasks are deterministic given
(houses, seed), so PLA and baseline attempt the same set of tasks for a
head-to-head comparison.

Results — 20 episodes each on houses 11-20, seed 2028:

| Run | success | 95% CI | mean approach Δ (m) | gripper open frac | wall |
|-----|---------|--------|---------------------|-------------------|------|
| PLA   (`rollout_pla_v3_holdout`) | 0/18 | [0, 17.6%] | +0.020 | 93% | 70 min |
| Baseline (`rollout_vlm_v3_holdout`) | 0/20 | [0, 16.1%] | +0.037 | 75% | 73 min |

(PLA's `n=18` rather than 20: pipeline skipped one house with a sampling
error.)

**Both policies fail every held-out task.** Neither approaches the pickup
object meaningfully — `mean tcp→pickup` shrinks by only 2-4 cm over 301
steps (vs the >1 m the planner moves on the same tasks). Failure-mode
analysis at `analysis_output/rollout_compare_v1/comparison.md`:

- Bucket A (`baseline_fail, PLA_success`): **0 of 18**
- Bucket B (`baseline_success, PLA_fail`): **0 of 18**
- Bucket C (`both_success`): **0 of 18**
- Bucket D (`both_fail`): **18 of 18**

**Root cause: training scale + missing language conditioning.** 36
training trajectories is too few to learn anything that generalizes across
houses and objects. The training loss drop (12 → 0.06) was memorization
of those 36 specific demos. On held-out houses the policy outputs
near-zero action deltas because it has never seen this distribution. The
fact that PLA's training-loss L1 was 17% lower than baseline's didn't
translate to any behavioural difference at rollout time — neither policy
gets close enough to anything for proximity readings to matter.

**Decision (2026-05-11)**: stop here, write up. Next round needs (a)
**100-house medium pilot data** (`FrankaSkinPickAndPlacePilotMediumConfig`
already registered, num_workers=2, ~6-8 h wall time) and (b) **language
conditioning** so the policy can identify the target object. Re-train +
re-rollout after both. Anything less and we will keep producing 0/N
results.

Artifacts kept on disk (all reusable for the next round):

- `runs/smoke_pla_v3_full/` + `runs/smoke_vlm_only_act_v3_full/` —
  ten 1.16 GB checkpoints each.
- `assets/datagen/pick_and_place_skin_pilot_eval_holdout_v1/.../20260511_021228/`
  — 35/52 successful held-out trajectories (the planner's solution to
  these same 40 tasks; useful as a ceiling for future evals).
- `assets/eval_subsets/FrankaSkinPickAndPlaceHoldout_v1/` —
  patched 35-episode JsonBenchmark (`object_poses` filter +
  displacement-threshold fix applied). Useful once the `CameraSpec`
  schema is extended upstream.
- `rollout_output/{rollout_pla_v3_holdout, rollout_vlm_v3_holdout}/` —
  20 episodes each, full h5 trajectories + per-episode `results.json`.
- `analysis_output/rollout_compare_v1/comparison.{md,json}` — the
  side-by-side report above.

WandB: <https://wandb.ai/jayluvsgeography/pla> (PLA `731wnt1d`,
baseline `gjl5aijc`).

---

## 8. Status

- [x] Skin on FR3 (29 SPAD-style 8×8 proximity sensors, link2/3/5/6).
- [x] 8×8 depth render path with correct HFOV/VFOV (dedicated 8×8 renderer).
- [x] Skin self-occlusion fix (`geomgroup[2]=0`).
- [x] Camera image orientation correct (`quat="0 0 1 0"`).
- [x] Quantitative GT verification (mj_ray, empty room, flat plane).
- [x] Visual verification (per-sensor hi-res RGB inspection of a full
      house_1 pick episode).
- [x] Production cleanup (inspection-mode code removed from `env.py`).
- [x] Pilot data run (procthor-objaverse, pick-and-place) — 47 successful
      episodes across 10 houses; revealed worker self-termination guard
      and tabletop bias (see §6.1, §6.2).
- [ ] **Low-surface pick-and-place collection (skin-showcase scenes) —
      see §6.2. Blocked on scene cache repopulation.**
- [ ] Large-scale data collection across iTHOR / ProcTHOR scenes.
- [x] **PLA training pipeline (`pla/`) — dataset, encoder, policy wrapper,
      train/eval/diagnostics scripts. Code-complete; see §7.5.**
- [x] **Smoke re-collection with the proximity-period fix** — 36/36
      successful trajectories with 99.94% nonzero proximity pixels
      (2026-05-10). Full pointcloud + verification report at
      `diagnostics_output/pilot_skin_smoke_v1/episode_house2_traj0/`.
- [x] **action_dim 7→8 (gripper now predicted by the network)**,
      eval_policy wired to snap to `{0, 255}` via threshold (§7.5.d).
- [x] **Train VLM-only ACT baseline on smoke (20k steps, final loss 0.0689).**
- [x] **Train PLA on smoke (20k steps, final loss 0.0619, ~10% lower than baseline).**
- [x] **WandB backfilled** for both runs (`scripts/backfill_wandb_from_log.py`).
- [⚠️] **JsonBenchmark eval blocked on `CameraSpec` schema** — needs an
      upstream PR to add per-camera resolution + `is_proximity_sensor` flag
      (see §7.5.d). Patched 35-episode benchmark is on disk and ready when
      the schema is extended.
- [x] **Custom rollout eval (`pla/rollout_eval.py`) — built, ran on 18-20
      held-out episodes per condition. Result: PLA 0/18, baseline 0/20.**
      Both policies produce near-stationary actions on held-out tasks;
      no behavioural gap. Pipeline validated, training scale insufficient.
- [ ] **100-house medium pilot collection**
      (`FrankaSkinPickAndPlacePilotMediumConfig`, num_workers=2,
      ~6-8 h wall time). Now mandatory — 36 trajs is too few to learn
      generalizable behaviour.
- [ ] **Language conditioning via Molmo VLM tokens** — without it the
      policy has no way to identify the target object in a multi-task
      setting. Also now on the critical path, not a stretch goal.
- [ ] Re-train PLA + baseline on the medium dataset WITH language.
- [ ] Re-run `pla.rollout_eval` and `pla.rollout_compare`.
- [ ] Real-robot transfer evaluation.
