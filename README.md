# prox_learning — proximity-skin sensing & safety for the Franka FR3

A MuJoCo / MolmoSpaces project for a **40-sensor hybrid proximity skin** on a Franka FR3, and a
**Safety-CVAE** that turns skin readings into a reflexive "move away from the obstacle" motion.

The repo is organized around three things that are actively used:

1. **Fumehood / enclosure data collection** — scripted whole-arm-clearance datagen (MolmoSpaces).
2. **Safety-CVAE training** — distills an analytic potential-field teacher into a small conditional VAE.
3. **Evaluation + demos + paper figures** — render the trained reflex and the sensor's capabilities.

---

## What the Safety-CVAE is

A conditional VAE that maps the **40×8×8 skin depth image → a 7-DoF joint retreat `dq`** (which way to
pull the arm back from a nearby obstacle). It is trained by distillation from an analytic
potential-field teacher, so it outputs the *avoidance motion directly* — it is not a collision
classifier. At inference `z = 0`, giving a deterministic retreat. `SafetyHead.load()` (in
`scripts/train_safety_cvae.py`) is the inference wrapper every demo imports.

Pipeline: `datagen trajectories → safety_sweep.py (potential-field labels) → sweep.h5 → train_safety_cvae.py → cvae_*/ → demos`.

---

## Setup

```bash
conda activate mlspaces
```

All rendering is headless (offscreen EGL). Prefix render/datagen commands with:

```bash
OMP_NUM_THREADS=2 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
```

The hybrid-skin model used everywhere is `assets/robots/franka_skin/model_hybrid.xml`
(built by `scripts/build_hybrid_on_franka_skin.py`).

---

## The pipeline (verified end-to-end)

**Run outputs go to `experiments_output/<run>/` (git-ignored), one consolidated folder per run** —
`sweep.h5`, `cvae/`, `demos/`, `figures/`. With no `--out`/`--outdir` flags everything defaults to
`experiments_output/default/`. (The committed canonical weights stay at `assets/safety/cvae_v3`.)

Example dataset: `assets/datagen/hybrid_obstacle_v1/FrankaSkinHybridObstacleConfig/20260612_183855`.
(Datasets are local datagen output — git-ignored. On a fresh clone, run the datagen recipe below
first, or skip steps 1 and train straight from the committed `assets/safety/sweep_v3.h5`.)

```bash
export EGL="OMP_NUM_THREADS=2 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl"
RUN=experiments_output/v5        # pick a name; all this run's files land here together
DS=assets/datagen/hybrid_obstacle_v1/FrankaSkinHybridObstacleConfig/20260612_183855

# 1. Build the near-contact training set (renders 40 SPAD depths, computes potential-field labels)
env $EGL python scripts/safety_sweep.py --runs $DS --n 15000 --out $RUN/sweep.h5

# 2. Train the Safety-CVAE (wandb + 4 diagnostic PNGs saved next to the weights)
env $EGL python scripts/train_safety_cvae.py --data $RUN/sweep.h5 --out $RUN/cvae --epochs 60
#   add --no-wandb to skip wandb (plots still written)
#   reference numbers on this dataset: val mse 0.009, close-cos 0.93

# 3. Demos — each writes a Foxglove .mcap + an annotated .mp4
for d in flinch react sphere moving orbit; do
  env $EGL python scripts/safety_${d}_demo.py --ckpt $RUN/cvae --out $RUN/demos/${d}.mcap
done
#   common flags: --obstacle {bar,sphere}, --clean (3-metric HUD only)

# 4. Paper figures — all 29 consolidated into one script
env $EGL python scripts/figures.py --list                       # list the 29 figures
env $EGL python scripts/figures.py --all --outdir $RUN/figures   # all -> this run's folder
env $EGL python scripts/figures.py panel_coverage_behind --outdir $RUN/figures   # just one
```

---

## Fumehood / enclosure data collection

Datagen lives in the `submodules/molmospaces` submodule. Recipe:

```bash
conda activate mlspaces
cd submodules/molmospaces
OMP_NUM_THREADS=2 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    python -m molmo_spaces.data_generation.main <ConfigName>
```

Useful hybrid-skin configs (`molmo_spaces/data_generation/config/object_manipulation_datagen_configs.py`):

| Config | What it collects |
|---|---|
| `FrankaSkinHybridFumehoodSmokeConfig` | Fumehood whole-arm-clearance reach, 40-sensor skin |
| `FrankaSkinFumehoodSmokeConfig` | Same, base (8 houses × 3) |
| `FrankaSkinHybridObstacleConfig` | Obstacle near-contact set (used for the CVAE above) |
| `FrankaSkinEnclosureGenConfig`, `FrankaSkinEnclosureSmokeConfig` | General enclosure-reach (full / smoke) |

Output goes to `assets/datagen/<name>/` (git-ignored).

---

## Where things live

```
scripts/
  train_safety_cvae.py     safety_sweep.py          # CVAE train + dataset prep
  safety_{flinch,react,moving,orbit,sphere}_demo.py # demos (import each other; keep together)
  build_hybrid_on_franka_skin.py  convert_hybrid_skin_urdf.py
  hybrid_viz_lib.py  foxglove_viz.py                # shared skin viz + .mcap export
  verify_hybrid_skin_sensors.py  test_and_reconstruct_hybrid.py
  build_photoshoot_skin.py  photoshoot_sweep.py     # paper "photoshoot" renders
  cavity_scene.py  analyze_dataset.py  analyze_obstacle_dataset.py
  dataset_probes.py  enclosure_report.py  proximity_necessity.py  plot_mindist_traces.py
  decorr_heatmap.py  wandb_upload_dataset.py  assemble_overnight_report.py
  figures.py                                        # ALL 29 paper figures (python figures.py --list)
assets/                                             # committed canonical artifacts (the "big files")
  robots/franka_skin/model_hybrid.xml               # the 40-sensor skin model
  safety/                                           # canonical cvae_v3/, sweep_v*.h5, demo + eval mp4/mcap
  datagen/                                          # collected datasets (git-ignored)
experiments_output/                                 # per-run outputs (git-ignored), one folder per run
  <run>/  sweep.h5  cvae/  demos/  figures/         # e.g. experiments_output/v4/, .../default/
submodules/                                         # act, molmospaces, MolmoBot
diagnostics_output/                                 # legacy committed renders (new figures -> experiments_output/)
paper/                                              # paper draft
```
