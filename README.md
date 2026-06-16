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

## Method & math

The whole system is four stages: **skin → closeness input**, **analytic potential-field
teacher → labels**, **CVAE distillation**, **deterministic inference**. Constants used
throughout: $D_{\max}=0.5\,\text{m}$ (input normalization), $D_{\text{act}}=0.18\,\text{m}$
(teacher activation range), close $<0.12\,\text{m}$, far $>0.25\,\text{m}$.

### 1. Skin sensor → network input

40 SPAD-style proximity sensors, each rendering an $8\times8$ planar-z **depth** patch in
meters → raw input is $40\times8\times8 = 2560$ values. Each patch uses pinhole intrinsics
with focal length $f = 4/\tan(\text{FOVY}/2)$, $\text{FOVY}=45^\circ$, principal point
$c_x=c_y=3.5$.

Depths are mapped to a **closeness** feature so "far / no return" is exactly zero (head stays
silent when nothing is near):

$$c = \mathrm{clip}\!\left(1 - \frac{d}{D_{\max}},\; 0,\; 1\right), \qquad D_{\max}=0.5\,\text{m}$$

Pixels with $d<5\,\text{mm}$ (dead/invalid) are forced to $0$. Flattening all sensors gives
the network input $x \in [0,1]^{2560}$.

### 2. Analytic potential-field teacher (the labels)

For every sensor whose nearest pixel sees an **environment** return at range $r < D_{\text{act}}$:
back-project that pixel to a world hit-point $p_o$; let $p$ be the sensor origin. Define the
range, the unit push direction (pointing *away* from the obstacle), and a repulsion weight that
vanishes at the activation boundary and grows as the obstacle nears:

$$r = \lVert p - p_o\rVert, \qquad \hat{u} = \frac{p - p_o}{r}, \qquad w(r) = \frac{1}{r} - \frac{1}{D_{\text{act}}}$$

Each push is mapped from Cartesian to joint space through the sensor body's translational
Jacobian $J_p$, and summed over all 40 sensors. The 7 arm-DoF entries are the label:

$$dq \;=\; \sum_{s\,:\,r_s < D_{\text{act}}} J_{p,s}^{\top}\, \hat{u}_s\, \left(\frac{1}{r_s} - \frac{1}{D_{\text{act}}}\right)$$

This is whole-arm potential-field repulsion in joint space, computed **exactly from sim state**.
Self-hits (returns on the robot's own arm) are detected against the analytic scene boxes and
**excluded from the label**, but kept in the input — the trained head must cope with them at runtime.

### 3. Label scaling

Labels are normalized to unit RMS over the close subset $C$ (min depth $<0.12\,\text{m}$); the
scale $\sigma$ is stored in `meta.json` and folded back in at inference:

$$\sigma = \sqrt{\frac{1}{|C|}\sum_{i\in C}\lVert dq_i\rVert^2}, \qquad \widetilde{dq} = dq/\sigma$$

### 4. Conditional VAE (distillation)

A small MLP CVAE ($512\to256$, SiLU; latent $z\in\mathbb{R}^{8}$) that **reconstructs the
retreat vector**. At train time the encoder sees both the skin input $x$ and the target
$\widetilde{dq}$; the decoder reconstructs $\widetilde{dq}$ from $x$ and a latent $z$:

$$q(z\mid x, \widetilde{dq}) = \mathcal{N}\!\big(\mu,\, \mathrm{diag}\,e^{\log\sigma^2}\big), \qquad z = \mu + \epsilon\odot e^{\tfrac12\log\sigma^2},\ \ \epsilon\sim\mathcal{N}(0,I)$$

$$\widehat{dq} = \mathrm{Dec}([x, z]), \qquad \mathcal{L} = \underbrace{\big\lVert \widehat{dq} - \widetilde{dq}\big\rVert^2}_{\text{recon MSE}} + \beta\, D_{\mathrm{KL}}\!\big(q(z\mid x,\widetilde{dq})\,\Vert\,\mathcal{N}(0,I)\big)$$

with the closed-form Gaussian KL and a 10-epoch $\beta$ warm-up ($\beta_{\max}=10^{-2}$):

$$D_{\mathrm{KL}} = -\tfrac12\sum_{j}\big(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2\big), \qquad \beta = \beta_{\max}\cdot\min\!\Big(1,\ \tfrac{\text{ep}+1}{10}\Big)$$

The latent only shapes training — it lets the head represent the small ambiguity in *which way
to dodge*.

### 5. Inference (deterministic)

The latent is pinned to the prior mean $z=0$, so the head is a pure, deterministic
$\text{skin}\to dq$ map — **no scene geometry or object poses needed at runtime**:

$$dq = \sigma \cdot \mathrm{Dec}\big([\,x,\; \mathbf{0}\,]\big)$$

`SafetyHead.load("assets/safety/cvae_v3")` wraps exactly this; every demo calls `head(prox)`.

### 6. Metrics

- **recon MSE** — on the validation split (scaled label space). *ref ≈ 0.009.*
- **close direction cosine** — $\cos(\widehat{dq}, dq)$ on close samples (min depth $<0.12\,\text{m}$):
  is it pushing the *right way*? This is the trustworthy signal. *ref ≈ 0.93.*
- **far-quiet RMS** — $\lVert\widehat{dq}\rVert$ on far samples (min depth $>0.25\,\text{m}$):
  is it silent when clear?

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

## What each demo shows

All five demos drive the **same** skin-only head; they differ only in the obstacle motion and
how the head's output is applied. Two control modes:

- **Flinch mode** (`flinch`, `sphere`) — the arm rests at a clear posture and the head's
  baseline-subtracted output is integrated with a soft spring back to nominal, so the arm only
  reacts to the *change* a new obstacle causes.
- **Residual mode** (`react`, `moving`, `orbit`) — the head's push is low-passed and added as a
  growing/decaying residual correction on top of a nominal motion:

$$dq = \mathrm{head}(\text{skin with obstacle}) - \mathrm{head}(\text{skin, obstacle parked})$$

$$\text{correction} \mathrel{+}= (\text{gain}\cdot dq - \text{decay}\cdot\text{correction})\,\Delta t, \qquad q_{\text{exec}} = q_{\text{nom}} + \mathrm{clip}(\text{correction},\,\pm\,\text{max\_dev})$$

The baseline subtraction is key: the head fires on *any* close surface (hood walls, the arm's
own links), so subtracting its rest output makes each demo react only to the obstacle.

| Demo | What it demonstrates |
|---|---|
| `flinch` | Single bar marches down a forearm sensor's view axis; arm flinches away and relaxes home. |
| `sphere` | Identical motion with a **blue sphere** — head trained only on orange box bars reacts the same, proving it is shape- and colour-agnostic (it only ever sees an 8×8 depth blob). |
| `react`  | Reactive avoidance **layer** on a recorded reach-grasp-lift: clock never stops, arm bulges around bars in the path and rejoins — deviate-and-rejoin. |
| `moving` | Two bars patrol the whole kinematic chain (wrist→shoulder); each link tinted by its own obstacle-induced proximity, so you watch the signal travel across all links. |
| `orbit`  | One bar sweeps a half-circle over the outward face of the arm and slides elbow→wrist; sensors light up in turn as the arm pushes each part away. |

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

## ACT baseline (vanilla — RGB + qpos, NO proximity)

The control policy the proximity work is measured against: an **upstream ACT** policy
(`submodules/act/imitate_episodes.py`) trained on the one-env obstacle pick — red cup in
the fumehood, hazard bar present ~75% of episodes — using only the **exo + wrist RGB
cameras and joint qpos**. No skin / proximity channels. It is trained from the same
`hybrid_obstacle_v1` dataset the Safety-CVAE uses and is **evaluated in the exact datagen
environment** (`FrankaSkinHybridObstacleConfig`), so the success rate is directly
comparable across policies.

Shapes: `qpos = 9` (arm 7 + 2 finger joints), `action = 8` (arm 7 + 1 gripper command,
the FR3 hand actuator at {0, 255}). Cameras: `exo_camera_1`, `wrist_camera`, resized to
240×320.

One-time: the `mlspaces` env is missing two deps the ACT code imports —
```bash
conda activate mlspaces
pip install ipython          # required (ACT + DETR import it)
pip install wandb && wandb login   # training logs to wandb by default; skip + use --no_wandb to opt out
```
(`dm_control` is NOT needed — the ALOHA-only import is now optional.)

```bash
conda activate mlspaces
cd /home/jaydv/code/prox_learning
DS=assets/datagen/hybrid_obstacle_v1/FrankaSkinHybridObstacleConfig/20260612_183855

# 1. Convert the datagen run -> ACT per-episode HDF5s (RGB + qpos only; drops fail[-1]
#    episodes by default). Prints the final episode count + max length when done.
python -m scripts.convert_obstacle_to_act --src $DS --dst act_style_data/obstacle_v1 \
    --image_h 240 --image_w 320
#    -> e.g. "num_episodes=100, episode_len=169". Paste both into the
#    `obstacle_baseline` entry of submodules/act/constants.py if they differ.

# 2. Train ACT (run from the act submodule so its local `detr` package is importable).
#    chunk_size/hidden_dim/dim_feedforward/kl_weight MUST match the eval defaults below.
cd submodules/act
PYTHONPATH="$PWD:$PYTHONPATH" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
python imitate_episodes.py \
    --task_name obstacle_baseline --policy_class ACT \
    --ckpt_dir ckpts \
    --kl_weight 10 --chunk_size 100 --hidden_dim 512 --dim_feedforward 3200 \
    --batch_size 8 --lr 1e-5 --seed 0 --num_epochs 2000
#    --ckpt_dir is the ROOT; the run saves to its own dated folder:
#      ckpts/<task>/<datetime>_<runname>/   (printed at startup as "[ckpt] saving this run to ...")
#    e.g. ckpts/obstacle_baseline/20260616_153000_obstacle_baseline_2000_100_1e-05_0/
#    wandb logging is ON by default (run name = taskname_numepochs_chunk_lr_seed);
#    needs `pip install wandb` + `wandb login`. Disable with --no_wandb; override name
#    with --wandb_run_name. Writes policy_best.ckpt + dataset_stats.pkl into that folder.

# 3. Evaluate IN THE SAME ENV (FrankaSkinHybridObstacleConfig). One process = N rollouts;
#    reports success/total and saves a rollout MP4 + h5 per episode under --output_dir.
PYTHONPATH="$PWD:$PYTHONPATH" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
python eval_act_obstacle.py \
    --ckpt_dir ckpts/obstacle_baseline/<datetime>_<runname> \
    --output_dir /home/jaydv/code/prox_learning/eval_output/act_obstacle_baseline_v1 \
    --num_rollouts 25 --chunk_size 100
#    --ckpt_dir = the exact dated folder training printed (contains policy_best.ckpt +
#    dataset_stats.pkl).
#    add --temp_agg_off to disable temporal aggregation; --use_wandb to log + upload videos.

# 3b. Watch it live on your DESKTOP — add --live to open a MuJoCo viewer window.
#     Do NOT set MUJOCO_GL=osmesa or unset DISPLAY; the offscreen cameras still render
#     on EGL while the viewer opens its own GLFW window (so keep $DISPLAY set). --live
#     forces single-process and still writes the MP4s. On a headless box --live just
#     warns and falls back to offscreen. Use a few rollouts when watching:
PYTHONPATH="$PWD:$PYTHONPATH" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
python eval_act_obstacle.py \
    --ckpt_dir ckpts/obstacle_baseline/<datetime>_<runname> \
    --output_dir /home/jaydv/code/prox_learning/eval_output/act_obstacle_baseline_v1 \
    --num_rollouts 5 --chunk_size 100 --live
```

`imitate_episodes.py --eval` is the upstream ALOHA evaluator and does **not** know this
env — always evaluate with `eval_act_obstacle.py`, which swaps the ACT policy into the
real datagen pipeline. Checkpoints live in `submodules/act/ckpts/` and rollouts in
`eval_output/` (both git-ignored).

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
  convert_obstacle_to_act.py                        # datagen run -> ACT-style HDF5s (RGB+qpos baseline)
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
  act/  imitate_episodes.py (train)  eval_act_obstacle.py (in-env eval)  constants.py (obstacle_baseline)
act_style_data/                                     # converted ACT datasets (git-ignored)
eval_output/                                        # ACT in-env rollout MP4s + h5 (git-ignored)
diagnostics_output/                                 # legacy committed renders (new figures -> experiments_output/)
paper/                                              # paper draft
```
