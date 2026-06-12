# P+ACT — Proximity-conditioned Action Chunking Transformer

**Peripersonal Language-Action policies via whole-body time-of-flight proximity sensing.**

`prox_learning` is the research repo for **P+ACT**: a manipulation policy that
augments the standard [ACT](https://tonyzhaozh.github.io/aloha/) (Action
Chunking Transformer) with **body-distributed proximity sensing**. A Franka FR3
is wrapped in a sensor *skin* carrying **29 SPAD-style 8×8 time-of-flight depth
sensors** (on links 2, 3, 5, 6). A small transformer encoder — trained offline
to map each sensor's short depth-frame history into the **3-D position of the
manipulated object in that sensor's local frame** — is **frozen** and inserted
into ACT as 29 extra encoder tokens.

The headline experiment is **P+ACT vs vanilla ACT** on Franka pick-and-place in
[MolmoSpaces](https://molmospaces.allen.ai/) (iTHOR / ProcTHOR-Objaverse)
household scenes: the same ACT backbone, with and without the proximity tokens.

| Headline number                          | Value                                  |
| ---------------------------------------- | -------------------------------------- |
| Prox-encoder val mean Euclidean error    | **0.020 m** (per-axis ≈ 0.8 / 1.0 / 1.2 cm) |
| Vanilla ACT success rate (n = 10)        | **4 / 10 = 40 %**                      |
| P+ACT success rate (n = 10)              | **8 / 10 = 80 %**                      |
| Δ                                        | **+ 40 pp** (Fisher one-sided p ≈ 0.057–0.085) |
| Decoder cross-attention on prox tokens   | **21.7 % of mass on 15.2 % of tokens** (1.55× per-token vs image) |

> The deep dive on *why* P+ACT works, the full results, the attention analysis,
> and the masking ablations live in **[`pact/README.md`](pact/README.md)** — the
> single source of truth for the P+ACT model. This top-level README is the
> **operational guide**: install, what every file does, and how to run the
> whole pipeline end to end.

---

## Table of contents

1. [Architecture in one diagram](#1-architecture-in-one-diagram)
2. [Repository layout](#2-repository-layout)
3. [Installation](#3-installation)
4. [The end-to-end pipeline](#4-the-end-to-end-pipeline)
5. [Stage A — Data collection (MolmoSpaces)](#5-stage-a--data-collection-molmospaces)
6. [Stage B — Verification & sanity checks](#6-stage-b--verification--sanity-checks)
7. [Stage C — Convert to ACT format](#7-stage-c--convert-to-act-format)
8. [Stage D — Train the proximity encoder](#8-stage-d--train-the-proximity-encoder)
9. [Stage E — Train P+ACT and the ACT baseline](#9-stage-e--train-pact-and-the-act-baseline)
10. [Stage F — Rollout evaluation](#10-stage-f--rollout-evaluation)
11. [Stage G — Ablations, analysis & figures](#11-stage-g--ablations-analysis--figures)
12. [File reference — `pact/`](#12-file-reference--pact)
13. [File reference — `scripts/`](#13-file-reference--scripts)
14. [The `submodules/act` modifications](#14-the-submodulesact-modifications)
15. [Top-level output directories](#15-top-level-output-directories)
16. [Conventions, gotchas & troubleshooting](#16-conventions-gotchas--troubleshooting)

---

## 1. Architecture in one diagram

```
                 ┌──────────────────────────────────────────────┐
                 │  29 body-mounted proximity sensors           │
                 │  (link2×7, link3×8, link5×6, link6×8)        │
                 │  each: 8×8 ToF depth @ 60 Hz                 │
                 │  trailing window: W=8 control steps × 4 sub  │
                 └────────────────────┬─────────────────────────┘
                                      │ (B, 29, W·4=32, 8, 8)
                                      ▼
              ┌──────────────────────────────────────────────┐
              │  FROZEN prox-encoder  (~0.82 M params)        │
              │  transformer encoder, 8×8 depth → 3-D pos     │
              │  ckpt: pact/outputs_prox/runs/.../ckpt_best.pt│
              └────────────────────┬─────────────────────────┘
                                   │ (B, 29, 3)  object pos per sensor, metres
                                   ▼
   ┌──────────────────────────────────────────────────────────────┐
   │                       ACT encoder memory                       │
   │  [ latent(1) | proprio(1) | prox(29) | image(160) ] = 191      │
   │                              │                                 │
   │                              ▼                                 │
   │   ACT decoder cross-attention into the 191 memory tokens       │
   │            ↳ 100 action queries → 8-d action chunk             │
   └──────────────────────────────────────────────────────────────┘
```

Vanilla ACT is the same minus the `prox(29)` tokens (memory length 162). The
proximity branch adds only ~22 k params (`Linear(3→512)` + 29 positional
embeddings), so the success-rate gap is attributable to the *signal*, not model
capacity. The integration is gated behind `n_proximity_sensors=0`, so vanilla
ACT remains bit-identical when the flag is unset.

---

## 2. Repository layout

```
prox_learning/
├── pact/                     # ← the P+ACT pipeline (the code you run)
│   ├── prox_encoder/         #   proximity encoder: model + cache builder + dataset
│   ├── act_prox/             #   ACT↔proximity integration: mapping, dataset, trainer, eval
│   ├── scripts/              #   encoder CLIs (build_cache / train / evaluate)
│   ├── analysis/             #   attention visualisation + outputs
│   ├── outputs_prox/         #   encoder caches + checkpoints (gitignored)
│   └── README.md             #   the P+ACT deep dive (model, results, science)
├── scripts/                  # ← ~60 helper scripts: datagen launchers, conversion,
│                             #   eval orchestration, statistics, analysis, W&B push
├── submodules/
│   ├── act/                  #   ACT policy (forked; proximity-gated edits)
│   ├── molmospaces/          #   data-gen + sim + rendering + benchmarks
│   └── MolmoBot/             #   Molmo VLM + policy integration (pinned)
├── assets/                   #   robot MJCF (incl. franka_skin), scenes, objects, benchmarks
├── franka_assets/fr3_skin/   #   FR3 + skin meshes
├── analysis_output/          #   plots/stats from eval runs (gitignored)
├── diagnostics_output/       #   proximity audits + smoke diagnostics (gitignored)
├── synthetic_verify/         #   empty-room / flat-plane GT verification artifacts
├── logs/ wandb/              #   run logs and W&B run dirs (gitignored)
├── pointcloud.ipynb          #   interactive proximity point-cloud notebook
├── pyproject.toml            #   the `pla` package metadata (name kept for history)
└── README.md                 #   this file
```

> **History note.** An earlier all-in-one stack lived in `pla/` (documented in
> `README_OLD.md`). The current, canonical pipeline is **`pact/`** (P+ACT). The
> `pyproject.toml` package name is still `pla` for continuity.

---

## 3. Installation

The pipeline uses **two Python environments** because the policy code (ACT) and
the simulator (MolmoSpaces) have different, conflicting dependency pins.

### 3.0 Clone with submodules

```bash
git clone <this-repo> prox_learning
cd prox_learning
git submodule update --init --recursive
```

### 3.1 MolmoSpaces env (`mlspaces`) — datagen, sim, P+ACT training/eval

This is the **primary** environment; the entire `pact/` pipeline and all
MolmoSpaces datagen/rollouts run in it. (Existing setups expose it at
`/opt/conda/envs/mlspaces/bin/python`.)

```bash
conda create -n mlspaces python=3.11
conda activate mlspaces

# MolmoSpaces (data-gen + sim + rendering)
cd submodules/molmospaces
pip install -e ".[mujoco]"          # or ".[mujoco-filament]" for the Filament renderer
cd ../..

# The pact package (this repo)
pip install -e .                    # numpy, torch, h5py, mujoco, matplotlib, scipy, ...
```

Optional MolmoSpaces extras: `dev` (linting/tests), `grasp` (grasp generation),
`housegen` (iTHOR/ProcTHOR/Holodeck house generation), `curobo`
(GPU-accelerated planning — only needed for RB-Y1 tasks). See
`submodules/molmospaces/README.md` for the CuRobo CUDA build recipe.

**Headless rendering** (servers without a display) — set before any sim run:

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

### 3.2 ACT env (`aloha`) — optional, for vanilla-ACT-only workflows

The ACT submodule ships its own conda spec. The P+ACT run-book runs ACT *inside*
`mlspaces`, but the original ACT scripts and some baselines (e.g.
`scripts/train_houses13_seeds.sh`) use the `aloha` env:

```bash
conda env create -f submodules/act/conda_env.yaml   # creates env "aloha"
conda activate aloha
cd submodules/act/detr && pip install -e .          # install the DETR/ACT model package
```

Installs Python 3.9, PyTorch 2.0 (CUDA 11.8), `mujoco==2.3.3`, `dm_control`,
`einops`, `h5py`, etc. (See `submodules/act/README.md`.)

### 3.3 Assets & scene cache

MolmoSpaces auto-downloads assets on first run. To pre-install / repair the
cache (robots incl. `franka_skin`, scenes, objects, grasps, benchmarks):

```bash
cd submodules/molmospaces
export MLSPACES_ASSETS_DIR=/path/to/resources     # where symlinks live
python -m molmo_spaces.molmo_spaces_constants      # download + symlink everything
```

| Env var | Effect | Default |
|---|---|---|
| `MLSPACES_ASSETS_DIR` | Where downloaded assets are placed | `~/.cache/molmospaces/assets/<hash>` |
| `MLSPACES_FORCE_INSTALL` | Overwrite existing assets | `True` |
| `MLSPACES_PINNED_ASSETS_FILE` | JSON pinning per-asset versions | — |

Fetch a single scene/variant:

```bash
python scripts/datagen/fetch_assets.py scene procthor-objaverse 0 --split train
python scripts/datagen/fetch_assets.py default      # all default robots/objects/grasps
```

> **Dangling-symlink gotcha.** `assets/scenes/<dataset>/*.xml` are symlinks into
> the cache. If the cache is wiped, the symlinks dangle silently and **every
> house fails with `ParseXML: Error opening file`** (symptom: hundreds of
> `HouseInvalidForTask: Scene setup failed during compilation`). Verify before a
> long run: `ls -L assets/scenes/procthor-objaverse-train/train_0.xml` should
> print a non-zero size. Repair by re-running the install above with
> `MLSPACES_FORCE_INSTALL=True`.

---

## 4. The end-to-end pipeline

```
 A. Collect demos in sim            molmo_spaces.data_generation.main  →  per-house *.h5 + MP4s
 B. Verify the proximity stream     scripts/datagen/verify_*.py, visualize_proximity.py
 C. Convert to ACT format           scripts/convert_*.py                →  act_style_data/<set>/episode_*.hdf5
 D. Train the proximity encoder     pact/scripts/{build_cache,train,evaluate}.py  →  ckpt_best.pt (FROZEN)
 E. Map ACT episodes ↔ source h5    pact.act_prox.build_mapping        →  prox_mapping.json
 F. Train P+ACT  +  ACT baseline    pact.act_prox.imitate_episodes_with_prox
 G. Rollout eval + ablations        pact.act_prox.eval_act_with_prox_encoder + scripts/*
 H. Analysis & paper figures        scripts/*figure*.py, pact/analysis/, scripts/push_*_to_wandb.py
```

Every command below assumes `cd /home/jaydv/code/prox_learning` and uses the
`mlspaces` interpreter unless noted. For brevity, `PY=/opt/conda/envs/mlspaces/bin/python`.

---

## 5. Stage A — Data collection (MolmoSpaces)

Demonstrations are generated by a scripted planner in MolmoSpaces using the
`franka_skin` robot (`FrankaSkinRobotConfig`) and camera system
(`FrankaSkinCameraSystem` = 2 RGB cameras `exo_camera_1` + `wrist_camera`, plus
the 29 SPAD proximity sensors). Configs live in
`submodules/molmospaces/molmo_spaces/data_generation/config/object_manipulation_datagen_configs.py`.

### 5.1 The `franka_skin` datagen configs

| Config name | Purpose | Scene set | Houses | Samples/house |
|---|---|---|---|---|
| `FrankaSkinPickAndPlaceDataGenConfig` | Base production class (iTHOR pick-and-place) | ithor | `range(0,4)` | 20 |
| `FrankaSkinPickAndPlacePilotConfig` | Full pilot | procthor-objaverse | up to 1999 | 5 |
| `FrankaSkinPickAndPlacePilotSmokeConfig` | 10-house smoke (validate the whole path) | procthor-objaverse | 1–10 | 4 |
| `FrankaSkinPickAndPlacePilotEvalHoldoutConfig` | Held-out eval set (builds a JsonBench) | procthor-objaverse | 11–20 | 4 |
| `FrankaSkinPickAndPlaceOneHouseMugConfig` | Single-house mug-only (the main P+ACT dataset) | ithor | `[1]` | 250 |
| `FrankaSkinLowSurfacePickAndPlaceDataGenConfig` | Bias to low/enclosed surfaces (sinks, shelves, seats) | procthor-objaverse | up to 1999 | 5 |
| `FrankaSkinLowSurfacePickAndPlacePilotConfig` | Low-surface pilot | procthor-objaverse | ~200 | 3 |
| `PACT` | Medium ~500-ep pre-pilot (subclass of pilot); `disable_collision_checks=True` (collision probe) | procthor-objaverse | 11–20 | 1 |
| `PACT_LowSurface` | `PACT` on low/enclosed surfaces | procthor-objaverse | — | 2 |
| `FrankaSkinProxNecessityPilotConfig` | **Proximity-necessity regime** (vision fails, skin must carry the task) — see [§5.3](#53-collision-probe--the-proximity-necessity-regime) | procthor-objaverse | 1–50 | 3 |

`LOW_SURFACE_PREFIXES = ("sink", "shelf", "bookshelf", "chair", "armchair",
"stool", "sofa", "bed", "bathtub", "toilet", "crapper", "dresser",
"chestofdrawers")` — these bias collection toward poses where the skin actually
pays off. `crapper`/`chestofdrawers` are the procthor-objaverse body names.

> **Critical field — `proximity_sensor_period_ms`.** Must be `16.6667` (≈ 60 Hz,
> 4 substeps/policy step). Setting it to `0` **silently disables proximity
> recording entirely** (the substep dim collapses to 1 and pixels read zero).
> This is fixed in the `FrankaSkin*` configs; double-check it if you subclass.

### 5.2 Launching datagen

Canonical invocation (named-config entry point):

```bash
cd submodules/molmospaces
PYTHONPATH=. MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  /opt/conda/envs/mlspaces/bin/python -m molmo_spaces.data_generation.main \
  FrankaSkinPickAndPlacePilotSmokeConfig 2>&1 | tee ../../logs/datagen_smoke.log
```

Swap the config name for any from the table (the pipeline iterates houses
internally — no bash loop needed). Output lands under
`<MLSPACES_ASSETS_DIR>/datagen/<output_dir>/<timestamp>/house_<i>/` as
`trajectories_batch_*.h5` + sibling `episode_*_<cam>_batch_1_of_1.mp4`.

**Parallel single-house collection.** Single-house configs need
`num_workers=1` (workers collide on scene setup). To parallelise, launch N
processes with disjoint output dirs and merge afterward — wrappers do this:

```bash
# 4 workers × 63 samples of the house-1 mug task (the v3 dataset)
PY=$(echo /opt/conda/envs/mlspaces/bin/python)
source /opt/conda/etc/profile.d/conda.sh && conda activate mlspaces
export MLSPACES_ASSETS_DIR=$PWD/assets MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
python scripts/run_v3_parallel.py        --jobs 4 --samples_per_job 63   # mug, house 1
python scripts/run_house10_cup_parallel.py --jobs 4 --samples_per_job 63 # cup, house 10
python scripts/_bench_one_house_mug.py   --n 5 --workers 4               # throughput probe
```

**Legacy CLI alternative** (quick flag overrides): `scripts/datagen/run_pipeline.py
--robot skin --task_type pick_and_place --scene_dataset ithor --house_inds <i>
--samples_per_house <N> --seed <S>`.

### 5.3 Collision probe & the proximity-necessity regime

The default pick-and-place pipeline is a **bad** showcase for proximity: the task
sampler's `check_robot_placement_visibility=True` only places the robot where
`exo_camera_1` already sees the target, so vision never fails and the skin is
redundant. Measured on the `PACT` data, `vision_blind_frac = 0.000` and
proximity-necessity `= 0.000` across every house. To make proximity *load-bearing*
you have to break that guarantee and force the arm to work among surfaces.

Two pieces landed for this:

**(a) Collision probe** — `disable_collision_checks` (new field on
`MlSpacesExpConfig`, default `False`; **`True` on `PACT`**). When set, the
task-sampler's robot-placement collision rejection is bypassed (the `PLACE_ROBOT_NEAR`
loop accepts any pose) so the robot can operate in collision-prone configurations.
MuJoCo contact *detection* stays on, and a per-step **collision metric** is recorded
into every trajectory's `obs_scene["collision_metrics"]`:
`{collided, n_collision_steps, total_contacts, per_step_contacts, n_steps}`
(robot↔environment contacts, excluding the floor and the grasped object — the latter
gets welded to the robot root on grasp). Code: `env.py::count_robot_environment_contacts`,
threaded through `tasks/task.py`.

**(b) Proximity-necessity regime** — `FrankaSkinProxNecessityPilotConfig` flips three
levers vs `PACT`: `check_robot_placement_visibility=False` (drop the camera guarantee),
`source_surface_types=LOW_SURFACE_PREFIXES` (recessed targets — sink basins, shelf
interiors — where even the wrist cam loses the object), and heavy clutter packed
close (`num_added_pickups=60`, small `base_pose_sampling_radius_range`). Collisions are
left **on** so a vision-only policy must collide at rollout (the intended P+ACT-vs-ACT
contrast).

This **over-generates** candidates rather than hand-designing scenes. Curate it with
the necessity metric and keep trajectories where the skin is provably required:

```bash
# 1. collect
cd submodules/molmospaces
PYTHONPATH=. MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  /opt/conda/envs/mlspaces/bin/python -m molmo_spaces.data_generation.main \
  FrankaSkinProxNecessityPilotConfig 2>&1 | tee ../../logs/prox_necessity_pilot.log

# 2. measure / curate (keep prox_active_frac >= 0.8 AND high vision_blind_frac)
cd /home/jaydv/code/prox_learning
/opt/conda/envs/mlspaces/bin/python scripts/proximity_necessity.py \
  --glob 'assets/datagen/pick_and_place_skin_prox_necessity_pilot_v1/**/house_*/trajectories_batch_*.h5' \
  --near_m 0.15 --out diagnostics_output/prox_necessity_pilot
```

`scripts/proximity_necessity.py` reports, per trajectory and as dataset means:
`vision_blind_frac` (neither RGB cam sees the target), `prox_active_frac` (≥1 skin
sensor reads a surface < `--near_m`), and `necessity` (`vision_blind ∧ prox_active`).
A dataset is only useful for the thesis when `necessity` and `frac_meeting_prox_active_0.8`
are well above zero. Inspect any single trajectory with
`scripts/inspect_pact_trajectory.py` (qpos / actions / TCP / 29-sensor heatmaps /
collision probability — see [§13](#13-file-reference--scripts)).

---

## 6. Stage B — Verification & sanity checks

Never start a big collection without a quantitative **and** visual sign-off that
the 29 sensors see the world correctly. Verifiers live in
`submodules/molmospaces/scripts/datagen/` (run from inside `submodules/molmospaces`):

| Script | What it checks | Example |
|---|---|---|
| `verify_proximity_gt.py` | Renders native 8×8 depth at a timestep, compares to recorded values; writes `analysis/gt_compare_t<T>/grid.png` + `summary.md` | `python scripts/datagen/verify_proximity_gt.py <H5> --t 10` |
| `verify_synthetic_scenes.py` | Places the robot in an empty room / flat plane, reconstructs a point cloud from proximity, compares to known geometry | `python scripts/datagen/verify_synthetic_scenes.py` |
| `visualize_proximity.py` | Renders the 29-sensor grid as an MP4 + PNG from a trajectory | `python scripts/datagen/visualize_proximity.py <H5> --traj traj_0` |
| `analyze_sample_episode.py` | Per-episode kinematics, object tracking, success/fail, plots | `python scripts/datagen/analyze_sample_episode.py <H5> --episode 0` |
| `print_configs.py` / `compare_configs.py` | List all registered configs / diff two config JSONs | `python scripts/datagen/print_configs.py` |

Quantitative results from this protocol are checked into
[`synthetic_verify/`](synthetic_verify/) (empty-room + flat-plane error
histograms, the documented −44.6 mm 8×8 floor bias, and `summary.md`).

**Pilot acceptance checklist** before scaling up: per-house success rate
≥ ~50 %; zero `native 8x8 render failed` in the log; spot-check 1–2 trajectories
with `visualize_proximity.py`; the combined H5 builds cleanly.

Repo-side diagnostics: [`diagnostics_output/`](diagnostics_output/) holds
proximity audits and the `act_inference_probe.py` model-probe;
[`pointcloud.ipynb`](pointcloud.ipynb) is an interactive point-cloud
reconstruction notebook.

---

## 7. Stage C — Convert to ACT format

ACT consumes per-episode `episode_<idx>.hdf5` files (decoded float32 arrays +
resampled camera frames). The `scripts/convert_*` family transforms MolmoSpaces
output into `act_style_data/<set>/`:

```bash
PY=/opt/conda/envs/mlspaces/bin/python

# Single source h5 → per-episode ACT hdf5s
$PY -m scripts.convert_pla_to_act --src <…/trajectories_batch_1_of_1.h5> \
    --dst act_style_data/<set> --image_h 240 --image_w 320

# Whole mug_random_everything run (356 per-timestamp folders) → global episode indices
$PY scripts/convert_mug_random_to_act.py \
    --dst act_style_data/mug_house1_random_everything --image_h 240 --image_w 320 --resume

# Smoke (10 houses) → ACT + a mapping.json provenance sidecar
$PY -m scripts.convert_smoke_to_act --src_run_dir <…/20260510_124831> --dst act_style_data/smoke

# Merge parallel v3 worker outputs (run_w0, run_w1, …) into one ACT dataset
$PY scripts/merge_v3_to_act_style.py --src_base <…/parallel_2026…> --dst act_style_data/pla_house1_mug_v3

# Combine two ACT datasets (house 1 + house 3 → 488 eps via symlinks)
$PY scripts/build_combined_h1_h3.py
```

Other helpers: `duplicate_one_trajectory.py` (replicate one demo N× with noise
for overfit tests), `merge_chunks.py` / `merge_n20_chunks.py` (merge eval
chunks), `append_rand_object_to_ep0.py`, `lock_clutter_bins.py` (freeze clutter
bins from planner data for stratified eval). See [§13](#13-file-reference--scripts).

---

## 8. Stage D — Train the proximity encoder

The encoder is trained **once**, then frozen and reused by every P+ACT run. Each
training sample is one `(trajectory, sensor, t)` tuple, kept only when the object
is **visible** to the sensor, the gripper is **not holding** it
(`grasp_state.held == False`), and W real control steps precede `t`. Labels are
the object position in the sensor's local frame.

```bash
PY=/opt/conda/envs/mlspaces/bin/python

# 1. Build the windowed cache from source h5s (one sample = one window).
$PY pact/scripts/build_cache.py \
    --data_glob 'assets/datagen/mug_house_1_random_everything/**/trajectories_batch_*.h5' \
    --out pact/outputs_prox/cache_full.npz --window 8 --keep_every 1

# 2. Train (~30 min on a 4090). Best-by-Euclidean checkpoint is saved.
$PY pact/scripts/train.py \
    --cache pact/outputs_prox/cache_full.npz \
    --out_dir pact/outputs_prox/runs --run_name prox_encoder_v1 \
    --steps 10000 --batch_size 256 --use_wandb --wandb_project prox-encoder

# 3. Evaluate + plots (scatter, error/euclidean hists, per-sensor MAE, 3-D).
$PY pact/scripts/evaluate.py \
    --checkpoint pact/outputs_prox/runs/prox_encoder_v1/ckpt_best.pt \
    --cache pact/outputs_prox/cache_full.npz --split val
```

`train.py` writes `evaluation_metrics.json` + `predictions.npz`; `evaluate.py`
adds the plots. The shipped `ckpt_best.pt` reaches **2.0 cm mean Euclidean**
error on held-out trajectories. Use `--smoke` for a 200-step CPU/GPU sanity run.

---

## 9. Stage E — Train P+ACT and the ACT baseline

### 9.1 Build the ACT-episode ↔ source-h5 mapping (one-time per dataset)

P+ACT reads proximity from the original h5 (no re-conversion). `build_mapping.py`
links each ACT episode to its source trajectory using a **qpos signature at
timesteps {5, 10, 15, 20, 25}** (45 floats — robot init is deterministic so
t<5 is identical across trajectories). It aborts on any zero- or multi-match.

```bash
/opt/conda/envs/mlspaces/bin/python -m pact.act_prox.build_mapping \
    --act_dataset_dir act_style_data/mug_house1_random_everything
# → writes act_style_data/mug_house1_random_everything/prox_mapping.json
```

### 9.2 Train both arms (matched hyperparameters)

```bash
PY=/opt/conda/envs/mlspaces/bin/python

# Vanilla ACT baseline (no proximity).
$PY -m pact.act_prox.imitate_episodes_with_prox \
    --task_name pla_house1_mug_random --policy_class ACT \
    --ckpt_dir runs/act_mug_v1_baseline \
    --batch_size 8 --num_epochs 5000 --lr 1e-4 --seed 0 \
    --kl_weight 10 --chunk_size 100 --hidden_dim 512 --dim_feedforward 3200 \
    --use_wandb --wandb_project pact --wandb_run_name act_mug_v1_baseline

# P+ACT (frozen prox-encoder ON).
$PY -m pact.act_prox.imitate_episodes_with_prox \
    --task_name pla_house1_mug_random --policy_class ACT \
    --ckpt_dir runs/act_prox_mug_v1 \
    --batch_size 8 --num_epochs 2000 --lr 1e-4 --seed 0 \
    --kl_weight 10 --chunk_size 100 --hidden_dim 512 --dim_feedforward 3200 \
    --use_proximity \
    --prox_encoder_ckpt pact/outputs_prox/runs/prox_encoder_v1/ckpt_best.pt \
    --prox_mapping_json act_style_data/mug_house1_random_everything/prox_mapping.json \
    --use_wandb --wandb_project pact --wandb_run_name act_prox_mug_v1
```

The trainer **asserts at every step** that all encoder params have
`requires_grad == False` and that none received a gradient. It logs the standard
ACT metrics plus `prox/pred_pos_{x,y,z}_mean` and `prox/finite_frac`. Without
`--use_proximity` it runs as plain ACT (regression-tested identical to upstream).

Multi-seed baseline+P+ACT on the house-3 dataset (uses the `aloha` env):
`bash scripts/train_houses13_seeds.sh`. Medium-dataset ablation pipelines:
`scripts/launch_medium_v1.sh`, `scripts/launch_medium_ablations.sh`.

---

## 10. Stage F — Rollout evaluation

Rollouts run the trained policy inside a fresh MolmoSpaces process (so each draws
a new task — that's why absolute success counts wiggle ±10 pp; read the Wilson
95 % CIs, not point estimates). The inference policy
(`eval_act_with_prox_encoder.py`) maintains a per-sensor **ring buffer** of the
last W control steps, z-scores it with the encoder's stats, and feeds the 3-D
positions to ACT.

```bash
# 10 P+ACT rollouts, one shell command:
CKPT_DIR=runs/act_prox_mug_v1 \
PROX_ENC=pact/outputs_prox/runs/prox_encoder_v1/ckpt_best.pt \
PROX_MAP=act_style_data/mug_house1_random_everything/prox_mapping.json \
N_ROLLOUTS=10 \
  bash scripts/eval_act_prox_aggregate.sh

# 10 vanilla ACT rollouts (parallel wrapper around the ACT eval entry point):
/opt/conda/envs/mlspaces/bin/python scripts/run_act_mug_random_10x.py \
    --n_runs 10 --output_dir eval_output/act_house1_mug_random_v1_aggregate

# Build the summary + comparison + significance:
PY=/opt/conda/envs/mlspaces/bin/python
$PY scripts/aggregate_pact_eval.py --root eval_output/act_prox_mug_v1_aggregate \
    --baseline_summary eval_output/act_house1_mug_random_v1_aggregate/summary.json
$PY scripts/plot_pact_vs_baseline.py \
    --baseline_root eval_output/act_house1_mug_random_v1_aggregate \
    --pact_root     eval_output/act_prox_mug_v1_aggregate \
    --out           eval_output/act_prox_mug_v1_aggregate/comparison_plot.png
$PY scripts/significance_pact_vs_baseline.py \
    --baseline_root eval_output/act_house1_mug_random_v1_aggregate \
    --pact_root     eval_output/act_prox_mug_v1_aggregate
```

`run_act_prox_mug_10x.py` is the Python equivalent of the bash loop (wipes
per-run dirs to force re-randomisation). Other entry points in
`submodules/act/`: `eval_act_house1.py`, `eval_act_house1_dup250.py`,
`eval_act_mug_random.py`, `eval_act_house10_cup.py`, `eval_act_with_prox.py`
(see [§14](#14-the-submodulesact-modifications)). `scripts/watch_progress.sh`
prints live progress of running evals.

---

## 11. Stage G — Ablations, analysis & figures

### 11.1 Proximity masking ablations (is ACT *using* the prox tokens?)

`eval_act_with_prox_encoder.py` supports `--mask_proximity
{none,zero,mean,noise,shuffle}` and phase-localised masking `--mask_phase
{approach,pregrasp,grasp_lift,transit,place}`. The orchestrators:

```bash
PY=/opt/conda/envs/mlspaces/bin/python
# Precompute the mean-position baseline used by --mask_proximity mean:
$PY pact/act_prox/precompute_prox_mean.py --act_dataset_dir act_style_data/mug_house1_random_everything \
    --prox_mapping_json <…/prox_mapping.json> --prox_encoder_ckpt <…/ckpt_best.pt> --output prox_pos_mean.npy

# One masking condition (N rollouts, K parallel):
$PY scripts/run_pact_mask_experiment.py --n_runs 50 --parallel 4 \
    --mask_proximity zero --mask_phase none --output_dir eval_output/exp1_mask_zero_n50

# All masking experiments (Exp 1 prox masks + Exp 2 phase masks):
N=50 PARALLEL=3 bash scripts/run_pact_exp1_exp2_all.sh

# Which checkpoint epoch rolls out best:
$PY scripts/run_pact_epoch_sweep.py --ckpt_dir runs/act_prox_mug_v1 \
    --epochs 1500,1700,1900,best,last --n_runs 12 --parallel 2 --output_dir eval_output/epoch_sweep
```

`test_masking.py` is the fast unit test for the masking branches.

### 11.2 Attention & sensor analysis

```bash
# Decoder cross-attention over the 29 prox tokens (4 PNGs + raw_stats.json):
PYTHONPATH="submodules/act:.:${PYTHONPATH:-}" /opt/conda/envs/mlspaces/bin/python \
  pact/analysis/visualize_prox_attention.py \
    --ckpt_dir runs/act_prox_mug_v1 \
    --prox_encoder_ckpt pact/outputs_prox/runs/prox_encoder_v1/ckpt_best.pt \
    --prox_mapping_json act_style_data/mug_house1_random_everything/prox_mapping.json \
    --dataset_dir act_style_data/mug_house1_random_everything \
    --out_dir pact/analysis/attention_outputs --n_batches 20
```

Sensor-usage and failure analyses (all in `scripts/`): `sensor_usage_timeline.py`,
`sensor_success_vs_fail.py`, `joint_sensor_ranking.py`, `attention_vs_activity.py`,
`temporal_attention_plot.py`, `failure_taxonomy.py`, `phase_duration_analysis.py`,
`phase_transitions.py`, `action_variance_per_phase.py`, `tcp_path_visualisation.py`,
`modality_weight_comparison.py`, `plot_weight_usage.py`.

### 11.3 Paper figures & W&B push

Composite figures: `paper_figure.py`, `paper_figure_v2.py`,
`paper_master_figure.py`, `sensor_characterization_figure.py`,
`plot_three_way_act_prox_comparison.py`, `plot_visrand_ablation.py`,
`plot_mask_experiments.py`. Push results to Weights & Biases:
`push_pact_n50_to_wandb.py`, `push_exp_aggregate_to_wandb.py`,
`push_paper_analysis_to_wandb.py`, `push_taxonomy_to_wandb.py`,
`push_visrand_ablation_summary.py`, and `backfill_wandb_from_log.py` (replay a
log into W&B when a run was trained with `use_wandb=false`).

---

## 12. File reference — `pact/`

### `pact/prox_encoder/` — the proximity encoder
| File | What it does |
|---|---|
| `model.py` | `ProxEncoder` + `ProxEncoderConfig`: CNN `FrameTokenizer` per 8×8 frame → transformer encoder (d_model=128, 4 heads, 4 layers) + sinusoidal PE → 3-D position. Input `(B, T, 8, 8)` → output `(B, 3)`. |
| `cache.py` | Preprocesses source h5 trajectories into a flat windowed `.npz` cache (one sample = one `(traj, sensor, t)`). Filters on visible/not-held/full-window; stores raw fp16 windows, fp32 labels, metadata, channel-wise norm stats. CLI: `--data_glob --out --window --keep_every --max_trajs --label_clip_m`. |
| `dataset.py` | `ProxWindowDataset` over the cache (per-channel z-scoring) + `split_by_trajectory` (deterministic 90/10 hold-out by whole trajectory). |

### `pact/scripts/` — encoder CLIs
| File | What it does |
|---|---|
| `build_cache.py` | Thin CLI wrapper → `prox_encoder.cache.main`. |
| `train.py` | Trains `ProxEncoder` (MSE, cosine LR + warmup, per-eval validation in metres, best-MAE checkpoint, optional W&B, `--smoke`). |
| `evaluate.py` | Loads a checkpoint, computes per-axis/per-sensor MAE/RMSE/R²/Euclidean, writes `evaluation_metrics.json` + `predictions.npz` + 6 plots. |

### `pact/act_prox/` — ACT ⨉ proximity integration
| File | What it does |
|---|---|
| `build_mapping.py` | Builds `prox_mapping.json` (ACT episode → source h5 + traj key) via the qpos signature; self-tests; aborts on ambiguous matches. |
| `dataset.py` | `ProxAugmentedEpisodicDataset` — wraps ACT's `EpisodicDataset`, yields `(image, qpos, action, is_pad, proximity_window)`; z-scores with encoder stats; left-pads early timesteps. `make_prox_dataloaders()` factory. |
| `prox_features.py` | `FrozenProxFeatureExtractor` — wraps the ~0.82 M frozen encoder; `(B, 29, W·4, 8, 8) → (B, 29, 3)` under `no_grad`, params frozen by construction. |
| `precompute_prox_mean.py` | Samples the dataset through the frozen encoder → `(29, 3)` mean positions for the `--mask_proximity mean` baseline. |
| `imitate_episodes_with_prox.py` | ACT trainer forked from `imitate_episodes.py`; `--use_proximity` adds the frozen encoder + prox tokens; encoder-frozen assertions; W&B prox metrics. |
| `eval_act_with_prox_encoder.py` | Rollout inference policy with the per-sensor ring buffer + masking experiments (`--mask_proximity`, `--mask_phase`, phase classifier); sets `MUJOCO_GL=egl`. |
| `test_masking.py` | Fast unit tests for the masking branches and phase classifier (no external files needed). |

### `pact/analysis/`
| File | What it does |
|---|---|
| `visualize_prox_attention.py` | Hooks decoder `multihead_attn`, aggregates cross-attention over the 191 memory tokens; writes `per_sensor_attention.png`, `group_attention.png`, `per_layer_per_sensor_heatmap.png`, `temporal_per_sensor.png`, `raw_stats.json` (committed under `attention_outputs/`). |

---

## 13. File reference — `scripts/`

### Data collection, conversion & dataset building
| File | What it does |
|---|---|
| `run_v3_parallel.py` | Launch N parallel datagen workers for the house-1 mug task (disjoint dirs/seeds, staggered start). |
| `run_house10_cup_parallel.py` | Same, for the house-10 cup task. |
| `_bench_one_house_mug.py` | Throughput probe for `FrankaSkinPickAndPlaceOneHouseMugConfig` (per-success wall-clock, extrapolation). |
| `convert_pla_to_act.py` | Convert one MolmoSpaces h5 (+ sibling MP4s) → per-episode ACT `episode_<i>.hdf5`. |
| `convert_mug_random_to_act.py` | Batch-convert the 356-folder `mug_house_1_random_everything` run with global indexing (`--resume`). |
| `convert_smoke_to_act.py` | Convert the 10-house smoke run to ACT + a `mapping.json` provenance sidecar. |
| `merge_v3_to_act_style.py` | Merge `run_w*` parallel worker outputs into one ACT dataset. |
| `build_combined_h1_h3.py` | Symlink-merge house-1 + house-3 ACT datasets (488 eps) and remap `prox_mapping.json`. |
| `duplicate_one_trajectory.py` | Replicate one trajectory N× with optional per-copy noise (overfit experiments). |
| `merge_chunks.py` / `merge_n20_chunks.py` | Merge evaluation chunks → unified `results.csv` + `summary.json` (Wilson CI). |
| `append_rand_object_to_ep0.py` | Re-render episode-0 plots overlaying extra experimental runs. |
| `lock_clutter_bins.py` | Bin eval houses into low/medium/high clutter from planner data (stable stratification). |
| `build_mug_random_everything_videos.py` | First/last-frame summary MP4s of a dataset (+ optional W&B). |

### Training launchers & orchestration
| File | What it does |
|---|---|
| `train_houses13_seeds.sh` | Train vanilla ACT + P+ACT on house-3 across seeds {42, 1337, 2026} (`aloha` env). |
| `launch_medium_v1.sh` | End-to-end medium-dataset train → rollout (houses 21–30) → compare. |
| `launch_medium_ablations.sh` | Train 4 PLA/VLM ablation variants, then multi-seed eval across houses 11–20. |
| `automation_after_epoch_sweep.sh` / `automation_after_mask_mean.sh` | Post-sweep automation: re-eval best ckpt at n=50, regenerate figures, push to W&B. |
| `restart_epoch_sweep_wider.sh` | Kill a narrow epoch sweep and relaunch wider at higher parallelism. |
| `watch_progress.sh` | Live progress dashboard for running experiments (subprocess count, RAM/GPU, per-condition bars). |

### Evaluation, rollout & aggregation
| File | What it does |
|---|---|
| `eval_act_prox_aggregate.sh` | Bash loop: N sequential P+ACT rollouts → `run_NN/` layout. |
| `run_act_prox_mug_10x.py` | Python parallel wrapper for N P+ACT rollouts (+ CSV/summary/plot, Wilson CI). |
| `run_act_mug_random_10x.py` | Same for vanilla ACT (`eval_act_mug_random.py`). |
| `run_act_dup250_evals.sh` | Evaluate dup250-trained ACT checkpoints (10 rollouts each). |
| `run_vanilla_act_n50.sh` | Re-evaluate vanilla ACT at n=50 with fresh sampling. |
| `run_pact_mask_experiment.py` | Parallel runner for one mask / phase-mask condition (+ aggregation, plot, phase log). |
| `run_pact_exp1_exp2_all.sh` | Master orchestrator: Exp 1 (prox masks) + Exp 2 (phase masks). |
| `run_pact_epoch_sweep.py` | Sweep P+ACT eval across checkpoint epochs → `best_epoch.json`. |
| `run_noise_ablation.sh` | `mask_proximity=noise` and `=shuffle` fallbacks. |
| `aggregate_pact_eval.py` | Aggregate per-rollout logs → `summary.json` + `results.csv` (Wilson CI). |
| `significance_pact_vs_baseline.py` | Full stats suite: two-proportion z, Fisher exact, Newcombe CI, 20k-bootstrap. |

### Trajectory diagnostics & proximity-necessity ([§5.3](#53-collision-probe--the-proximity-necessity-regime))
| File | What it does |
|---|---|
| `inspect_pact_trajectory.py` | Full single-trajectory diagnostic report (13 PNGs + `summary.json` + `report.md`): qpos/qvel, commanded-vs-realized actions, world-frame TCP + distance-to-object, 29-sensor proximity heatmaps + 8×8 montage, manipulation phases, reward/success, and **collision probability** (`--h5 <path> [--traj traj_0] [--out <dir>]`). |
| `proximity_necessity.py` | Dataset-level metric & curation filter: per trajectory computes `vision_blind_frac`, `prox_active_frac`, and `necessity` (vision-blind ∧ prox-active); ranks trajectories, writes CSV/JSON/scatter, prints a verdict. Use to prove an environment is in the proximity-necessary regime and to select trajectories (`--glob '…/house_*/…h5'` or `--h5`, `--near_m`). |

### Analysis & plotting
| File | What it does |
|---|---|
| `sensor_usage_timeline.py` | Per-sensor activity by phase + time-normalised heatmaps + example trajectories. |
| `sensor_success_vs_fail.py` | Per-sensor activity heatmaps for success vs fail (+ difference matrix). |
| `joint_sensor_ranking.py` | Rank sensors by activity / attention / success-fail diff; rank-correlation matrix. |
| `attention_vs_activity.py` | Correlate decoder attention with physical sensor activity (Pearson/Spearman). |
| `temporal_attention_plot.py` | Attention to each sensor over normalised episode time. |
| `failure_taxonomy.py` | Classify every failed trajectory into 5 modes; χ² baseline vs P+ACT. |
| `phase_duration_analysis.py` / `phase_transitions.py` | Phase durations and Gantt-style entry-time distributions (success vs fail). |
| `action_variance_per_phase.py` | Per-phase action-delta magnitude (motion commitment), success vs fail. |
| `tcp_path_visualisation.py` | World-frame TCP paths coloured by phase (xy/xz projections). |
| `modality_weight_comparison.py` / `plot_weight_usage.py` | Input-projection weight magnitudes per modality, per layer, per sensor. |
| `plot_pact_vs_baseline.py` | Headline P+ACT vs ACT bars (per-run dots, Wilson CI, Fisher p, odds ratio). |
| `plot_mask_experiments.py` | Aggregate Exp 1/2 mask conditions → bars + significance + `all_rates.json`. |
| `plot_three_way_act_prox_comparison.py` | Vanilla vs prox-K=1 vs prox-K=6 success bars. |
| `plot_visrand_ablation.py` | Vanilla vs P+ACT across 3 visual-randomisation conditions. |
| `paper_figure.py` / `paper_figure_v2.py` / `paper_master_figure.py` | 3- / 6- / multi-panel composite paper figures. |
| `sensor_characterization_figure.py` | Single comprehensive sensor-use-case figure. |
| `visualize_mug_random_everything.py` / `visualize_skin_test_data.py` | Dataset visualisation (+ optional W&B watch mode). |

### Weights & Biases push
| File | What it does |
|---|---|
| `backfill_wandb_from_log.py` | Replay a training stdout log into a W&B run. |
| `push_pact_n50_to_wandb.py` | Log the final n=50 headline numbers + full stats suite. |
| `push_exp_aggregate_to_wandb.py` | Push Exp 1/2/3 aggregate (bars + taxonomy + rate table). |
| `push_paper_analysis_to_wandb.py` | Push the CoRL paper analysis (4 conditions, figures, markdown). |
| `push_taxonomy_to_wandb.py` | Push failure taxonomy (χ², counts, per-trajectory table). |
| `push_visrand_ablation_summary.py` | Push the visual-randomisation ablation matrix. |

---

## 14. The `submodules/act` modifications

ACT (Action Chunking Transformer) is a CVAE + DETR-style transformer that
predicts action *chunks*. Four files received small, **backwards-compatible**
additions, all gated behind `n_proximity_sensors=0` (default) so vanilla ACT is
bit-identical when proximity is off:

| File | Edit |
|---|---|
| `detr/models/detr_vae.py` | Adds `input_proj_proximity = Linear(3, hidden_dim)` and extends `additional_pos_embed` to `(2 + n_proximity_sensors·K, hidden_dim)`; `forward` accepts `proximity_positions` and raises if it's `None` when sensors are enabled. |
| `detr/models/transformer.py` | `forward` accepts `proximity_input`; concatenates the prox tokens after `[latent, proprio]` and before image tokens. |
| `policy.py` | `ACTPolicy.__call__` threads `proximity_positions` through to the DETRVAE. |
| `detr/main.py` | Declares `--n_proximity_sensors`, `--prox_tokens_per_sensor` (+ trainer flags) so the nested argparse accepts them. |

ACT rollout entry points (used by the eval wrappers; run with
`PYTHONPATH=$PWD:… MUJOCO_GL=egl`):

| Script | Evaluates |
|---|---|
| `eval_act_mug_random.py` | Randomised house-1 mug pickup (matches training distribution). |
| `eval_act_house1.py` | Standard single-house mug pickup. |
| `eval_act_house1_dup250.py` | dup250 dataset with all randomisation disabled (deterministic). |
| `eval_act_house10_cup.py` | House-10 cup pickup (different house/object). |
| `eval_act_with_prox.py` | ACT + an optional `ProximityResidualHead` action-correction. |

Upstream files: `imitate_episodes.py` (vanilla trainer), `policy.py`,
`utils.py` (data loading), `constants.py` (task/robot constants, `DT`).

---

## 15. Top-level output directories

Most are **gitignored** (`.gitignore` excludes `assets/**`, `runs/**`,
`wandb/**`, `logs/**`, `act_style_data/**`, `eval_output/**`).

| Directory | Contents |
|---|---|
| `assets/` | Robot MJCF (incl. `franka_skin`), scenes, objects, benchmarks, `eval_subsets/`, grasps, reference renders. Has its own `assets/README.md`. |
| `franka_assets/fr3_skin/` | FR3 + skin meshes (STL/OBJ) + MJCF used at compile time. |
| `analysis_output/` | Plots/stats from eval runs (`eval_medium_v1`, `rollout_compare_v1`, `per_house_corr`, `training_data_diagnostics`, …). |
| `diagnostics_output/` | Proximity audits, smoke diagnostics, `act_inference_probe.py`. |
| `synthetic_verify/` | Empty-room / flat-plane GT verification artifacts + `summary.md`. |
| `pact/outputs_prox/` | Encoder caches (`cache_*.npz`), checkpoints (`runs/`, `runs_smoke/`), `train_v1.log`. |
| `logs/` | Timestamped datagen/training/eval/rollout logs. |
| `wandb/` | Local W&B run directories. |
| `pointcloud.ipynb` | Interactive proximity point-cloud reconstruction notebook. |

---

## 16. Conventions, gotchas & troubleshooting

- **Two interpreters.** Use `/opt/conda/envs/mlspaces/bin/python` for everything
  in `pact/`, `scripts/`, and MolmoSpaces datagen/eval. The `aloha` env is only
  for the original ACT scripts / a couple of baselines.
- **Headless rendering.** Export `MUJOCO_GL=egl PYOPENGL_PLATFORM=egl` for any
  sim run on a server. `eval_act_with_prox_encoder.py` sets it itself.
- **`PYTHONPATH`.** Run MolmoSpaces from inside `submodules/molmospaces` with
  `PYTHONPATH=.` so the in-tree copy wins over any stale site-packages install.
  Attention/eval scripts that import ACT need `PYTHONPATH=submodules/act:.`.
- **`proximity_sensor_period_ms` must be 16.6667**, never 0 (0 silently disables
  proximity recording — see [§5.1](#51-the-franka_skin-datagen-configs)).
- **Proximity is decorative unless vision fails.** The task sampler's
  `check_robot_placement_visibility=True` guarantees the camera sees the target, so
  the proximity-necessity metric is 0 on the default/`PACT` data. Use
  `FrankaSkinProxNecessityPilotConfig` + curation to get a useful regime
  ([§5.3](#53-collision-probe--the-proximity-necessity-regime)).
- **`disable_collision_checks`** (default `False`; `True` on `PACT`) only bypasses the
  task-sampler placement rejection — MuJoCo contact detection stays on, and the
  per-episode collision metric is written to `obs_scene["collision_metrics"]`.
- **Worker memory.** Each datagen worker spawns a ~6–7 GB MuJoCo sim; keep
  `num_workers` at 2–4 on a 64 GB box. Single-house configs require
  `num_workers=1` (parallelise with the `run_*_parallel.py` wrappers instead).
- **Eval noise.** MolmoSpaces draws a fresh task per process, so re-running a
  10-rollout eval gives different absolute counts. Read the Wilson 95 % CIs and
  the significance tests, not point estimates.
- **The frozen encoder is OOD post-grasp** (trained only on `held == False`).
  v1 feeds post-grasp frames through unchanged and lets ACT discount them via
  attention; the masking ablations ([§11.1](#111-proximity-masking-ablations-is-act-using-the-prox-tokens))
  quantify the effect.

---

*For the model design, the full results, the attention/ablation evidence, and the
scientific argument for why P+ACT works, read **[`pact/README.md`](pact/README.md)**.
Per-project memory lives under
`.claude/projects/-home-jaydv-code-prox-learning/memory/`.*


---

# FR3 Hybrid Proximity Skin

Franka FR3 + Robotiq with the **gentact hybrid skin**: 40 SPAD proximity sensors (8×8 depth,
45° FOV, 1.5–50 cm range) tiled on links 1–6. RGB is blurred at policy-training time, so this
skin is the robot's contact-range perception.

## Models
| file | what |
|---|---|
| `assets/robots/franka_skin/model_hybrid.xml` | **functional** model — 40 working 8×8 depth cameras |
| `assets/robots/franka_skin/model_photoshoot.xml` | **visual** model — 40 SPAD markers at the URDF poses (paper figures) |
| `assets/robots/franka_skin/model.xml` | original 29-sensor skin (unchanged) |
| `assets/urdf/fr3_hybrid_skin.urdf` | source URDF (gentact) |

Built from the proven `model.xml` (real FR3 meshes + Robotiq), swapping the old skin for the
hybrid dermis + sensors. Scripts: `build_hybrid_on_franka_skin.py`, `build_photoshoot_skin.py`.

## Sensors (40)
link1:7, link2:7, link3:5, link4:5, link5_front:4, link5_back:6, link6:6.
- **Aim:** radial out of each arm-link axis (outward + surface-perpendicular); a few cramped
  sensors are raycast-repaired. URDF `<axis>` and dermis mesh normals are unreliable (wrong
  winding) — do not use them. Cameras sit 9 mm proud of the skin so they clear the dermis shell.
- **Near-field fix:** MuJoCo depth clip = `vis.map.znear × stat.extent`; with a big scene this
  silently erased every return < ~10 cm. Pinned `znear=0.0002` (clip ~2 mm). Same render path is
  used by datagen (`renderer/opengl_rendering.py`), so this matters for collected data too.

## Verification (`scripts/verify_hybrid_skin_sensors.py`)
| check | result |
|---|---|
| not buried in own arm (3 poses) | **40/40** |
| outward-facing | **40/40** |
| plate @ 0.15 m reads 0.145±0.012 m | **38/40** (2 link5-front legitimately see the wrist link) |
| range linearity | slope **1.0**, R² **0.9999999**, error sub-µm |
| back-projected cloud vs ground truth | **~4 mm** median |

## Reconstruction (back-project all 40 sensors → point cloud vs every real surface)
| scene | active | RMS | within 1 cm | script |
|---|---|---|---|---|
| primitives (box+sphere+cyl+corner) | – | 4.0 mm | 95% | `test_and_reconstruct_hybrid.py` |
| open clutter (3 obj + table) | 14/40 | 4.7 mm | 91% | `test_and_reconstruct_hybrid.py` |
| clutter, accumulated (14 poses) | – | – | 4200 pts | `test_and_reconstruct_hybrid.py` |
| **fumehood** (1 frame) | 22/40 | 4.7 mm | 91% | `test_reconstruct_fumehood.py` |
| **fumehood**, accumulated (14 poses) | – | 5.0 mm | 93% (10366 pts) | `test_reconstruct_fumehood.py` |

Every back-projected return lands on a real surface to **millimetre** accuracy → sensors are in
the right place **and** looking at the right thing.

## Datagen wiring (40 sensors flow end-to-end, verified)
- `FrankaSkinHybridRobotConfig` (→ `model_hybrid.xml`), `FrankaSkinHybridCameraSystem` (40 cams),
  `FrankaSkinHybridFumehoodSmokeConfig`.
- Run: `python -m molmo_spaces.data_generation.main FrankaSkinHybridFumehoodSmokeConfig`
- Skin geoms set to **group 2** (env hides them in proximity render).
- Fixed `save_utils.py` regex that was dropping the `link5_front/back` sensors (saved 30/40 → now
  **40/40** proximity keys in the h5).
- Not yet done: house-embedded hybrid (set `znear` in the mixin), full-scale collection, probe
  re-tuning for 40-sensor stats.

### Fumehood variations (advisor: "variate it a lot... motions that need the robot to get in more")
`diagnostics_output/20260611_fumehood_variations/` — 6 hood geometries x verified deep-insertion motions
(FK-checked hand depth, per-frame skin cloud + clearance telemetry):

| variant | interior (WxHxD, sash) | insertion (hand / TCP) | active sensors @ deep | min clearance |
|---|---|---|---|---|
| standard | 64x46x55, 30cm | 31.6cm (57% D) | 28/40 | 3.4cm |
| tall | 68x85x55, 55cm | 35cm + high-interior sweep | 29/40 | 3.5cm |
| short low-sash | 68x34x50, **17cm** | 29.3cm hand / 44.8cm TCP | 26/40 | **0.3cm** (ducks flat) |
| narrow | **32**x50x55, 32cm | 33.5cm | 25/40 | 0.4cm lateral |
| wide big | 110x65x70, 45cm | 39cm + lateral sweep +-24cm | 31/40 | 0.5cm |
| deep tunnel | 52x42x**95**, 28cm | 38cm hand / **52.6cm TCP** (whole forearm inside) | 26/40 | 0.9cm |

Note: deep-tunnel 60cm hand target is beyond the FR3's 855mm reach envelope — 52.6cm TCP is the
physical ceiling with the elbow resting on the bench lip. All motions collision-checked.

### Foxglove fumehood tour
One playable trajectory chaining all 6 hood variants (22.7s, 190 frames): 3D robot + per-chapter
hood + live 40-sensor skin cloud, exo RGB, sensor-tile mosaic, insertion-depth & skin plots,
chapter log. Build: `python scripts/foxglove_fumehood_tour.py` -> `diagnostics_output/20260611_fumehood_tour/fumehood_tour.mcap`
(open at app.foxglove.dev, import `scripts/foxglove_unified_layout.json`).

## Photoshoot (`scripts/photoshoot_sweep.py`)
`diagnostics_output/20260611_skin_photoshoot/`: hero, sensor close-up, 360° turntable (sheet +
mp4), pose sweep. Studio lighting, dark background; uses `model_photoshoot.xml`.

## Reproduce
```
# build
python scripts/build_hybrid_on_franka_skin.py        # functional model
python scripts/build_photoshoot_skin.py              # visual model
# verify + reconstruct
python scripts/verify_hybrid_skin_sensors.py
python scripts/test_and_reconstruct_hybrid.py
python scripts/test_reconstruct_fumehood.py
# render
python scripts/render_hybrid_skin_viz.py             # 3D + exo + wrist + 40 tiles
python scripts/photoshoot_sweep.py                   # paper turntable
```
All renders need EGL (handled in-script via the default-display patch).
Outputs land in `diagnostics_output/2026*`. wandb gallery: `jayluvsgeography/prox-skin-dataset`.
