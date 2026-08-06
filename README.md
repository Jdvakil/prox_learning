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
#    Besides success/total, eval reports a per-episode COLLISION metric: PickTask counts,
#    each step, the distinct environment bodies (cavity wall / shelf / hazard bar /
#    fumehood, excluding floor + the grasped cup) the arm penetrates. _ObstacleEvalRunner
#    reads it off the live task after each rollout and logs to stdout/running_log as
#    `[act-eval] ep### ... obstacle_contact_steps=.../...` plus a final `collision summary`
#    (contact_free_rate, mean_contact_steps split by success/fail, mean_peak). With
#    --use_wandb these go to wandb too: per-episode `episode/obstacle_*`, aggregate
#    `eval/*`, and an `eval/per_episode` table. This is the safety axis for the
#    proximity-vs-baseline comparison (vision-only wedges the arm -> high contact_steps).
#    STRICT-SAFETY criterion: pass --end_on_collision to count ANY arm<->obstacle collision
#    as a FAILURE and end the episode the moment of first contact (pick_task.is_terminal +
#    get_info, gated on config.end_on_collision; default off so standard runs/numbers are
#    unchanged). The reported success then = the strict (collision-as-failure) rate. Even
#    WITHOUT the flag, every run now also prints/logs `strict_success_rate` (= grasped+lifted
#    AND contact-free) and `collision_rate` (= 1 - contact_free_rate), so one standard run
#    yields both the raw and hedged numbers. Baseline (N=20): raw success 35/40/30% ->
#    strict 20/15/25% for 2000ep-chunk50 / 2000ep-chunk100 / 5000ep-chunk100; collision
#    rate 30/60/20%.
#    PER-CELL EVAL: pass --eval_cell {visible,invisible,free} to pin EVERY rollout to one
#    obstacle cell — it swaps the task sampler to InvisibleObstacleFumehoodPickCheckSampler
#    and forces its class attrs per cell (visible: OBSTACLE_P=1.0/INVIS_P=0.0; invisible:
#    OBSTACLE_P=1.0/INVIS_P=1.0 = bar physically present but hidden from the RGB cameras,
#    skin-only; free: OBSTACLE_P=0.0). Same sampler in all three cells keeps the object-
#    placement distribution identical, so one checkpoint is scored per cell across three
#    invocations (use one --output_dir per cell). Without the flag nothing changes (the
#    inherited ~75%-visible-bar mix; the invisible-bar sampler is never even imported, so
#    the script still runs on an older molmospaces checkout). Every run now also writes
#    <output_dir>/eval_summary.json — ckpt, eval_cell (+forced probs), sampler class,
#    headline numbers, and per-episode collision records — so results are self-describing.
#    MEMORY: ACTObstacleEvalConfig forces viz_sensor_rgb=False. The FrankaSkinHybrid* chain
#    inherits viz_sensor_rgb=True, which renders all 40 proximity sensors at 256x256 RGB +
#    256x256 depth-turbo EVERY policy step purely for cosmetic skin videos. The vision-only
#    ACT policy never reads those keys, yet the datagen pipeline retains each episode's full
#    observation history in RAM until the whole house finishes saving -> ~3 GB/episode of
#    skin frames that OOM-kill any multi-episode eval (a 30-rollout run SIGKILLed on a 62 GB
#    box). Disabling it leaves policy inputs, success, and the collision metric byte-for-byte
#    identical; memory then grows ~0.5 GB/episode (just exo/wrist frames). Rollouts stay
#    ~3 min each (the sub-stepped 8x8 SPAD depth render over 40 sensors dominates, not viz),
#    so budget ~70 min for 20 rollouts. Eval is single-worker, so several ckpts can run in
#    parallel processes (each ~8 GB base + 0.5 GB/episode -> keep concurrency x N under RAM).

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

## P+ACT (PACT) — fusing the proximity Safety-CVAE into ACT

PACT conditions the ACT policy on the proximity skin so it can *see* obstacles the
cameras miss, with the goal of beating the vision-only baseline (headline run B:
**40% success / 60% collision rate** at 2000 epochs, chunk 100). The fusion is
**architectural, not a post-hoc safety layer**: the pretrained Safety-CVAE
(`assets/safety/cvae_v3`) is loaded **frozen** and used as a skin → feature extractor;
its feature becomes extra **encoder tokens** inside the ACT transformer, so the policy
learns to attend to proximity exactly as it attends to image/qpos tokens. Only the
ACT-side projection (`input_proj_proximity`) and the extra positional embeddings train.

- **Where it injects** (`detr/models/detr_vae.py`, `transformer.py`): encoder memory
  becomes `[latent, proprio, prox_1..prox_K, image_tokens...]`. A single global skin
  feature is expanded into `K` tokens (`--prox_tokens_per_sensor`, default 8). With
  proximity OFF the model is bit-identical to vanilla ACT.
- **Two feature taps** (`prox_cvae.py`, `--prox_feature`) — we sweep both:
  - `trunk` (256-d): the CVAE decoder trunk's hidden activation at `z=0` — the richest
    skin-only representation it learned.
  - `delta` (7-d): the CVAE's actual joint retreat-direction output — the literal
    steering vector (interpretable, but lower-dim and magnitude-soft).
  - The CVAE's own *encoder* `q(z|skin,dq)` is conditional on the retreat label and so
    cannot run at policy time; the frozen **decoder** is the skin-only path.
- **One dataset, both arms.** `convert_obstacle_to_act.py --with_proximity` writes
  `/observations/proximity (T,40,8,8)` alongside identical RGB/qpos/action. Vanilla ACT
  ignores it; PACT reads it. Sensors are stacked in `cvae_v3/meta.json` order (the one
  source of truth — note **link5_back precedes link5_front**, opposite the env tuple).
- **Train/eval match is automatic.** Training writes `prox_config.json` into the ckpt
  dir; `eval_act_obstacle.py` auto-detects it and rebuilds the same extractor + token
  layout (no flags needed). The frozen CVAE gives identical features at train and eval.
- **FACTR visual curriculum** (`--blur_sigma0`, `--blur_curriculum_steps`): Gaussian-blurs
  ALL camera frames at training time with a linearly annealed strength
  `sigma_n = sigma0 * (1 - n/N)` at global train step `n` (FACTR, architecture unchanged).
  Strong blur early starves the vision shortcut and forces the policy onto the qpos +
  proximity tokens; the blur is gone by step `N` (default: half the total training steps).
  `--blur_sigma0 8` matches FACTR; `0` (default) disables it. Validation and eval always
  see sharp frames; the per-epoch sigma is logged to wandb as `train/blur_sigma`. For a
  fair comparison give every arm (vanilla control included) the same blur settings.
  `--blur_mode` picks the schedule: `curriculum` (default) is the annealed FACTR behavior
  above; `constant` holds `sigma0` on EVERY training frame for the whole run (no anneal) —
  a fixed-blur degraded-vision *training handicap* (validation/eval still see sharp frames).
  Constant mode is registered in both argparsers (`imitate_episodes.py` + `detr/main.py`),
  so the DETR re-parse accepts it. See the constant-blur baseline recipe below.
- **Modality dropout** (`--image_dropout_p`, `--prox_dropout_p`, `--image_dropout_mode`,
  `--no_zero_latent_on_drop`): per-sample hard dropout at TRAINING time, applied after
  the blur; constant p throughout (no anneal — blur is the annealed component, dropout
  is hard pressure forever). Vision-dropped samples get every camera (`all`, default) or
  one random camera (`single`) filled with the ImageNet mean → exactly-zero input after
  the policy's Normalize, so on those samples L1 can only fall through the qpos +
  proximity tokens (attacks the BC-redundancy failure mode). Two leak paths are closed:
  the CVAE style latent z (inferred from the ground-truth chunk) is zeroed on
  vision-dropped samples exactly like the inference branch (ablate with
  `--no_zero_latent_on_drop`), and prox dropout is sampled disjointly from image dropout
  so no sample is ever blind on both. Recommended arm: `--image_dropout_p 0.3
  --prox_dropout_p 0.1` on top of `--blur_sigma0 8`; keep `p <= 0.4`. Validation, eval,
  and best-ckpt selection stay clean (dropout never fires there; `eval_act_obstacle.py`
  unchanged). Watch `train/l1_img_dropped` vs `train/l1_clean` in wandb: dropped-L1
  plateauing far above a degrading clean-L1 is the degenerate-average early warning
  (lower p or switch to `--image_dropout_mode single`).

```bash
cd /home/jaydv/code/prox_learning/submodules/act
DS=/home/jaydv/code/prox_learning/assets/datagen/hybrid_obstacle_v1/FrankaSkinHybridObstacleConfig/20260612_183855

# 1. Convert WITH proximity (one dataset serves vanilla + both PACT arms). Prints
#    num_episodes/episode_len -> paste into constants.py `obstacle_pact` if they differ.
cd /home/jaydv/code/prox_learning && conda activate mlspaces
python -m scripts.convert_obstacle_to_act --src $DS \
    --dst act_style_data/obstacle_prox_v1 --with_proximity --image_h 240 --image_w 320

# 2. Three matched arms (identical seed/epochs/chunk; the only difference is proximity).
#    Run from the act submodule so its local `detr` package imports. The shared flags are
#    --task_name obstacle_pact --policy_class ACT --ckpt_dir ckpts --kl_weight 10
#    --chunk_size 100 --hidden_dim 512 --dim_feedforward 3200 --batch_size 8 --lr 1e-5
#    --seed 0 --num_epochs 2000  (matches baseline run B). Prefix each with the env vars.
cd submodules/act
ENV='PYTHONPATH="$PWD:$PYTHONPATH" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl'

# 2a. vanilla control (ignores the proximity group; same blur so the comparison is fair)
PYTHONPATH="$PWD:$PYTHONPATH" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python imitate_episodes.py \
    --task_name obstacle_pact --policy_class ACT --ckpt_dir ckpts --kl_weight 10 \
    --chunk_size 100 --hidden_dim 512 --dim_feedforward 3200 --batch_size 8 --lr 1e-5 \
    --seed 0 --num_epochs 2000 --blur_sigma0 8 --wandb_run_name vanilla_blur
# 2b. PACT, trunk feature (add --use_proximity --prox_feature trunk)
PYTHONPATH="$PWD:$PYTHONPATH" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python imitate_episodes.py \
    --task_name obstacle_pact --policy_class ACT --ckpt_dir ckpts --kl_weight 10 \
    --chunk_size 100 --hidden_dim 512 --dim_feedforward 3200 --batch_size 8 --lr 1e-5 \
    --seed 0 --num_epochs 2000 --blur_sigma0 8 --use_proximity --prox_feature trunk \
    --wandb_run_name pact_trunk_blur
# 2c. PACT, delta feature (swap --prox_feature delta)
PYTHONPATH="$PWD:$PYTHONPATH" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python imitate_episodes.py \
    --task_name obstacle_pact --policy_class ACT --ckpt_dir ckpts --kl_weight 10 \
    --chunk_size 100 --hidden_dim 512 --dim_feedforward 3200 --batch_size 8 --lr 1e-5 \
    --seed 0 --num_epochs 2000 --blur_sigma0 8 --use_proximity --prox_feature delta \
    --wandb_run_name pact_delta_blur
#   --prox_encoder_ckpt defaults to assets/safety/cvae_v3; PACT runs write prox_config.json.
#   --blur_sigma0 8 = FACTR visual curriculum (see bullets above); drop it for no-blur arms.

# 3. Eval each IN-ENV (proximity auto-detected from the ckpt's prox_config.json — no flag).
#    Quick proof = 50 rollouts/arm; same viz_sensor_rgb=False OOM guard as the baseline.
#    --temp_agg_off matters for PACT: with temporal aggregation ON (the default, m=0.01)
#    the newest chunk — the only one that saw the CURRENT skin reading — carries ~1.6% of
#    the executed action (mean staleness ~41 steps), structurally muting any reactive
#    avoidance. The eq50 runs of 2026-06-18 had it ON; rerun comparisons with it OFF.
#    BUG (found + fixed 2026-07-04): the original --temp_agg_off branch re-queried the
#    policy EVERY step and executed only chunk[0]. action[0] is nearly a copy of the
#    current qpos (the easiest thing for the L1 loss to learn), so per-step chunk[0]
#    execution converges to a fixed point: the arm creeps toward the object, freezes
#    ~30 cm short, and holds there to timeout. Symptom: 0/N success with LOW collision
#    for EVERY checkpoint (v1 and v2, any cell). Verified by A/B: v1 vanilla ctrl ckpt
#    scored 0/10 with the old --temp_agg_off but 5/10 with temp agg ON on the same env.
#    The fix makes --temp_agg_off standard ACT open-loop chunking: query once, execute
#    the whole chunk, re-query when it is exhausted (inference_model early-returns
#    chunk[k] from the pending chunk). Any --temp_agg_off numbers produced before
#    2026-07-04 are invalid — rerun them.
for D in <vanilla_dir> <pact_trunk_dir> <pact_delta_dir>; do
  PYTHONPATH="$PWD:$PYTHONPATH" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python eval_act_obstacle.py \
      --ckpt_dir ckpts/obstacle_pact/$D \
      --output_dir /home/jaydv/code/prox_learning/eval_output/$D \
      --num_rollouts 50 --chunk_size 100 --temp_agg_off
done

# 4. Honest comparison (success rate + Wilson CI + collision + Fisher exact).
cd /home/jaydv/code/prox_learning
python scripts/compare_pact.py \
    vanilla=<S>/50,<C>/50 pact_trunk=<S>/50,<C>/50 pact_delta=<S>/50,<C>/50
```

A 50-rollout pass is a **quick proof**, not a publication result: a +18pp success win is
still only a "trend" at n=50 (Fisher p≈0.11). If a tap shows a trend, re-run it across
≥3 seeds for confidence intervals before claiming "PACT beats vanilla ACT". The
collision rate mixes benign cavity-wall brushing with hazard-bar ramming; splitting out
the **hazard-bar-only** collision rate (where proximity should help most) is the natural
next refinement.

**Architecture audit (2026-07-02) — why the first PACT run tied vanilla.** The prox-token
plumbing is verified correct end-to-end at line level (sensor order, featurize parity,
frozen CVAE, token/pos-embed layout, optimizer registration, train/eval parity, and
`n_proximity_sensors=0` bit-identity). The eq50 null result traces to confirmed causes,
not bugs:

1. **The trained policies ignore the token (critical).** Sensitivity probes on the actual
   checkpoints: blanking the ENTIRE skin moves pact_trunk's predicted chunk by ~0.005
   normalized units; a synthetic obstacle 6 cm from ten link5 sensors moves it ~0.03 rad —
   comparable to a 0.05 rad qpos nudge (pact_delta: ~0.19 rad, still weak). Min val loss is
   statistically identical across all three arms (0.0781 / 0.0777 / 0.0784): the token added
   zero held-out predictive value. Root cause is BC redundancy — demos come from the
   privileged scripted planner, whose avoidance is fully predictable from vision+qpos, so
   the L1 loss exerts no pressure to use the extra token. The FACTR blur curriculum above
   attacks exactly this.
2. **Ambient saturation, no baseline subtraction (major).** In 100% of demo frames some
   fixture (bench/sash/jamb) is within D_MAX=0.5 m, and 40-60% of timesteps sit inside the
   CVAE's D_ACT=0.18 m repulsion zone while the demonstrated action is "proceed into the
   cavity" — the danger feature fires constantly in collision-free demos
   (corr(‖delta‖, min_depth) ≈ −0.7), teaching BC to discount it. The reactive demos that
   DO avoid obstacles (`safety_react_demo.py`) work because they subtract a per-frame
   obstacle-parked baseline — a sim-only privilege PACT doesn't have.
3. **Temporal-aggregation washout at eval (major).** See step-3 comment: fresh skin gets
   ~1.6% of the executed action under the default `m=0.01`. One-flag fix: `--temp_agg_off`.
4. **Signal-to-horizon mismatch (major).** One skin snapshot conditions a 100-step chunk
   under uniform L1; a <0.5 m reading informs only the first few steps, so its marginal
   predictive value is intrinsically tiny at chunk 100. Shorter chunks (50) or early-step
   loss weighting would raise it.
5. **Delta-tap scale (minor).** The 7-d delta is un-normalized (×`label_scale` 11.36,
   observed up to |36|) through a near-init Linear, and goes out-of-distribution exactly at
   near-collision states — injects arbitrary-direction noise right when it matters.

Cheap diagnostics that separate "ignored" from "stale": (a) zero/shuffle the prox batch in
`eval_train_set.py` and compare L1; (b) log prox-token attention share in `attn_heatmap.py`
(the weights exist at indices `2:2+K` of the cross-attention but are currently discarded);
(c) one matched eval rerun with `--temp_agg_off`.

**Probe gate (run BEFORE any PACT GPU spend).** `scripts/probe_prox_decodability.py`
asks the prerequisite question directly: can the frozen cvae_v3 trunk feature *linearly*
separate deflect episodes from free episodes at approach timesteps? Labels are planner
ground truth (`behavior_class` from `obs_scene` in the original
`assets/datagen/hybrid_obstacle_v1/.../20260612_183855` run, remapped to
`act_style_data/obstacle_prox_v1` episodes and verified per-episode by exact qpos match);
probes are class-balanced L2 logistic regression under 5-fold grouped CV by episode, with
a raw per-sensor peak-closeness (40-d) probe as the comparison arm. Gate: trunk
episode-level AUC ≥ 0.8. **Result on obstacle_prox_v1 (2026-07-03): FAIL** — deflect vs
free is at chance (trunk ep-AUC 0.526 / seed 1: 0.449; raw skin 0.500), while the easier
bar-present label is partially decodable (trunk 0.770 vs raw 0.618, so the trunk isn't
destroying skin information — the *deflection* signature just isn't in the skin at
approach time). This independently corroborates audit cause #2: ambient fixture returns
saturate the skin, so bar-present and bar-avoiding approaches look alike. Do not launch
PACT training runs on this dataset until a v2 datagen changes the gate to PASS
(`--label-source {datagen-dir,h5,heuristic}` covers future label plumbing; heuristic =
joint-space bow, 88% label agreement, flagged in output).

**Gate-A residual diagnostic (2026-07-03, follow-up to the FAIL).** Four rescue probes ran
on the same v1 data to ask whether the deflect signal was merely *hidden* rather than
absent (scripts preserved in the session scratchpad as `probe_residual_gateA.py`,
`probe_extra_seeds.py`, `probe_univariate.py`; results in `gateA_rows.npy`):
1. *Ambient residual* — fit a kNN map qpos→expected 40-d skin closeness on free episodes,
   probe the residual (what the skin sees beyond pose-predictable ambient). Deflect AUC
   stayed at chance (0.41–0.46); residual bar-present dropped below the plain trunk.
2. *Bar-station window* — restrict frames to |tcp_x − bar_x| ≤ window. No gain; univariate
   skin stats are near-identical inside the window (deflect median max-closeness 0.726 vs
   free 0.721; ~72% of frames have ≥1 sensor inside 0.18 m in *both* classes).
3. *qpos-incremental* — proprioception alone is ALSO at chance for deflect (qpos9 AUC
   0.34–0.56 across 5 seeds), so no single-frame feature of any kind separates the classes;
   skin adds nothing on top (qpos+raw ≈ qpos).
4. *Finer features* — the full 2560-d flattened closeness map peaks at ~0.56–0.62 on some
   seeds but is seed-unstable (0.41–0.45 on seed 4); bar-present from 2560-d is 0.67–0.73,
   i.e. no better than the 256-d trunk (0.72–0.78 across seeds).

**Verdict:** the deflection signature genuinely does not exist frame-wise in v1 skin — the
planner holds the hazard bar at the same clearance as the ambient walls, so a deflecting
approach and a free approach produce statistically identical skin frames. This is direct
evidence for audit cause #2, not a feature-engineering failure (appending raw channels,
the round-1 fallback, would not help: raw is *worse* than trunk on every label). The
decodable quantity is **bar presence** (trunk AUC 0.72–0.78) — which is exactly the label
the invisible-bar design needs the skin to carry, since there the policy's job is
"bar present → take the deflecting path" and vision/qpos are blind to bar presence by
construction. Gate reinterpretation: the ≥0.8 deflect gate tested the wrong label for the
invisible-bar arm; the operative gate is bar-present decodability on the **v2** dataset
(re-run `probe_prox_decodability.py` against `obstacle_prox_v2` after datagen; proceed to
GPU if bar-present ep-AUC ≳ 0.75, with the caveat that ~0.77 signal quality may cap how
cleanly the policy separates the eval cells). The probe script has `--gate-label bar` for
exactly this: it scores the PASS/FAIL line on bar-present instead of deflect.

**Invisible-bar v2 collection (2026-07-03).** `FrankaSkinHybridInvisObstacleConfig` ran
125/200 episodes: 3 of 4 workers were OOM-killed (15–29 GB RSS each on the 62 GB box —
the known sensor-video memory issue), so houses 1, 25 and 73 produced nothing; the
surviving worker completed 5 houses × 25. The data that landed is healthy: 105/125
successes (84%), all 125 thetas unique, cells split visible-bar 47 / invisible-bar 49 /
free 29 (matches OBSTACLE_P=0.75, INVIS_P=0.5), `bar_invisible` + full bar geometry
stamped in `obs_scene.scene_params`. 105 successes ≈ the 100-episode v1 baseline, so
training proceeds on this; to backfill the 3 missing houses, rerun the config with
`num_workers=2` (or 1) so workers fit in RAM.

**v2 probe results (2026-07-03, gateA_v2 diagnostic on obstacle_prox_v2).** The picture
INVERTED relative to v1, and both changes are informative:
- *bar-present collapsed to chance* (trunk ep-AUC 0.40–0.52, raw 0.44–0.57, seeds 0–2;
  was 0.72–0.78 on v1). The v2 sampler always draws the bar-geometry variables so object
  placement no longer leaks bar presence — meaning v1's 0.77 was largely that leak, not
  the skin literally seeing the bar. The formal `--gate-label bar --auc-gate 0.75` gate
  therefore FAILS on v2.
- *deflect became decodable* (raw40 ep-AUC 0.749/0.763/0.751, trunk 0.716–0.754; was
  chance on v1). It is NOT a pose echo: qpos alone scores 0.60–0.68 and qpos+skin adds
  nothing over skin alone. Mechanically consistent: the v2 bar enters the 0.18 m
  repulsion zone in 74% of bar episodes (median per-episode min sensor-to-bar distance
  0.145 m; 87% of deflect episodes vs 62% of bar-free episodes get inside 0.18 m).
- *cell-restricted deflect* (the discrimination each eval cell actually needs):
  invisible+free raw40 0.64–0.70, visible+free raw40 0.74–0.79, invisible-only unstable
  0.58–0.75 (n=44).
- *raw beats trunk on every v2 label* → a third feature tap `--prox_feature raw` was
  added (40-d per-sensor peak closeness, bypasses the CVAE; prox_cvae.py, math identical
  to the probe's raw_skin_feature, smoke-verified equal).

Decision status: formal gate FAIL (bar presence is not frame-decodable), but the signal
the L1 loss actually trains against — skin explaining deflection actions vision can't —
exists at ~0.65–0.75 and survives the qpos control. Training on v2 is a judgment call,
not a gate pass; the eval-cell design (visible/invisible/free) makes even a negative
rollout result attributable.

**v2 training recipe (task `obstacle_pact_v2`, constants.py updated: 105 eps, len 185).**
Same hypers as the v1 arms for comparability; blur OFF (the invisible bar IS the forcing
pressure — don't stack a second confound):
```bash
cd submodules/act
# arm 1: vanilla control        --task_name obstacle_pact_v2 (no proximity flags)
# arm 2: PACT raw   (probe-favored)  add --use_proximity --prox_feature raw
# arm 3: PACT trunk (CVAE feature)   add --use_proximity --prox_feature trunk
PYTHONPATH="$PWD:$PYTHONPATH" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python imitate_episodes.py \
    --task_name obstacle_pact_v2 --policy_class ACT --ckpt_dir ckpts --kl_weight 10 \
    --chunk_size 100 --hidden_dim 512 --dim_feedforward 3200 --batch_size 8 --lr 1e-5 \
    --seed 0 --num_epochs 2000 --wandb_run_name <vanilla_v2|pact_raw_v2|pact_trunk_v2> \
    [--use_proximity --prox_feature raw|trunk]
# eval: 3 cells x 50 rollouts, serial (41 GB RSS each), temporal aggregation OFF
for CELL in invisible free visible; do
  PYTHONPATH="$PWD:$PYTHONPATH" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python eval_act_obstacle.py \
      --ckpt_dir ckpts/obstacle_pact_v2/<run> \
      --output_dir /home/jaydv/code/prox_learning/eval_output/<run>_$CELL \
      --num_rollouts 50 --chunk_size 100 --temp_agg_off --eval_cell $CELL
done
```
Headline comparison: hazard-BAR contact rate + success, invisible cell, PACT arms vs
vanilla (vanilla-invisible ≈ its free-cell behavior by construction — cameras see no bar).

**v2 RESULTS (2026-07-05, fixed --temp_agg_off open-loop chunking, 50 rollouts/cell):**

| collision rate      | free | invisible | visible |          | success  | free | invisible | visible |
|---------------------|------|-----------|---------|----------|----------|------|-----------|---------|
| vanilla             | 60%  | 66%       | 64%     |          | vanilla  | 22%  | 36%       | 28%     |
| pact_raw            | 58%  | **40%**   | 50%     |          | pact_raw | 18%  | 30%       | 16%     |
| pact_trunk          | 64%  | 72%       | 58%     |          | trunk    | 34%  | 34%       | 32%     |

- HEADLINE: pact_raw cuts invisible-cell collisions 66%→40% (Fisher p=.016, Wilson CIs
  [.52,.78] vs [.28,.54]); success unchanged in every cell (all p≥.23) — the skin buys
  safety at no task cost. Strict success (clean AND lifted): raw 20% vs vanilla 14%.
- CONTROLS: free cell = no arm differences (p=1.0) → no general-timidity artifact; the
  drop appears ONLY where the skin-only hazard exists. Vanilla does NOT improve when the
  bar is visible (64% vs 66%) — 105 eps never taught visual avoidance, so the raw skin
  tap is the only avoidance signal that made it into any policy.
- pact_trunk (frozen CVAE 256-d feature) is inert everywhere (p≥.67) — matches the v2
  probes (raw ≥ trunk on every label) and the v1 attention audit.
- DECISION (2026-07-06): trunk arm is SCRAPPED. No further trunk training or evals; the
  trunk checkpoint/eval JSONs stay on disk as the negative-result record. All round-2
  work iterates on the raw tap only (modality dropout, smaller executed-chunk window,
  more data, blur-sigma curriculum).
- Gradient note: raw collision falls free 58% → visible 50% → invisible 40%; within-arm
  invisible-vs-free p=.109 (trend). The causal claim rests on the between-arm test + the
  free-cell null, not on the within-arm trend.
- Anomaly (unexplained): free-cell success is LOWEST for vanilla/raw despite no bar —
  same placement sampler across cells; not blocking, flagged for the writeup.
- Raw numbers: eval_output/<arm>_v2_<cell>/eval_summary.json (per-episode records inside).

### Constant-blur vision-only baseline (blur sweep)

A degraded-vision baseline: vanilla ACT (RGB + qpos, **no proximity**) trained with a
*constant* Gaussian blur on every camera frame (`--blur_mode constant`, no anneal) and
**evaluated on sharp frames**. It asks how well vision-only avoidance survives when the
policy could never lean on fine visual detail during training — the natural floor the
proximity arms are measured against. Same hypers as the `obstacle_pact_v2` arms so it
drops straight into the v2 comparison table. Trainability of `obstacle_prox_v2` was
verified (105 eps, contiguous, no NaNs, T 54–183 ≤ 185 budget, all action dims non-zero
variance). Note on sigma: at 240×320, blur saturates fast — σ=2 already removes ~98% of
high-frequency detail (variance-of-Laplacian 498→7.8) and σ=4≈σ=8; see
`eval_output/blur_sweep_preview.png`. For a graded curve prefer {1,2,4}.

```bash
cd /home/jaydv/code/prox_learning/submodules/act
# TRAIN — one vanilla arm per blur sigma (constant blur; eval stays sharp). No proximity flags.
for S in 2 4 8; do
  PYTHONPATH="$PWD:$PYTHONPATH" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python imitate_episodes.py \
      --task_name obstacle_pact_v2 --policy_class ACT --ckpt_dir ckpts --kl_weight 10 \
      --chunk_size 100 --hidden_dim 512 --dim_feedforward 3200 --batch_size 8 --lr 1e-5 \
      --seed 0 --num_epochs 2000 --blur_sigma0 $S --blur_mode constant \
      --wandb_run_name vanilla_blurC${S}_v2
done
#   each run saves to ckpts/obstacle_pact_v2/<datetime>_vanilla_blurC${S}_v2/ (printed at startup);
#   `ls -t ckpts/obstacle_pact_v2 | head` recovers the three dated folders for eval.

# EVAL — 3 cells x N rollouts per model, sharp frames (default), temporal aggregation OFF.
for D in <blurC2_dir> <blurC4_dir> <blurC8_dir>; do
  for CELL in invisible free visible; do
    PYTHONPATH="$PWD:$PYTHONPATH" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python eval_act_obstacle.py \
        --ckpt_dir ckpts/obstacle_pact_v2/$D \
        --output_dir /home/jaydv/code/prox_learning/eval_output/${D}_$CELL \
        --num_rollouts 50 --chunk_size 100 --temp_agg_off --eval_cell $CELL
  done
done
#   anchor against the sharp-trained v2 vanilla (collision free/invis/visible = 60/66/64%,
#   success 22/36/28%). Compare with scripts/compare_pact.py using each eval_summary.json.
```

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
  convert_obstacle_to_act.py                        # datagen run -> ACT HDF5s (+--with_proximity for PACT)
  compare_pact.py                                   # PACT vs vanilla success/collision + Fisher exact
  probe_prox_decodability.py                        # go/no-go gate: is deflect linearly decodable from the frozen trunk?
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
  act/  imitate_episodes.py (train, --use_proximity)  eval_act_obstacle.py (in-env eval)
        prox_cvae.py (frozen Safety-CVAE feature extractor for PACT)  constants.py (obstacle_baseline/_pact)
act_style_data/                                     # converted ACT datasets (git-ignored)
eval_output/                                        # ACT in-env rollout MP4s + h5 (git-ignored)
diagnostics_output/                                 # legacy committed renders (new figures -> experiments_output/)
paper/                                              # paper draft
```
