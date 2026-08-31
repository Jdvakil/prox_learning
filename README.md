# prox_learning — proximity-skin sensing & safety for the Franka FR3

A MuJoCo / MolmoSpaces project for a **40-sensor hybrid proximity skin** on a Franka FR3, and a
**Safety-CVAE** that turns skin readings into a reflexive "move away from the obstacle" motion.

The repo is organized around three things that are actively used:

1. **Fumehood / enclosure data collection** — scripted whole-arm-clearance datagen (MolmoSpaces).
2. **Safety-CVAE training** — distills an analytic potential-field teacher into a small conditional VAE.
3. **Evaluation + demos + paper figures** — render the trained reflex and the sensor's capabilities.

The preregistered whole-body modality comparison is documented in
[`docs/PACT_VS_ACT_FINAL_DECISION.md`](docs/PACT_VS_ACT_FINAL_DECISION.md);
its Phase 1 environment gate is
[`docs/PACT_ENVIRONMENT_ADEQUACY.md`](docs/PACT_ENVIRONMENT_ADEQUACY.md).
The independently seeded recovery protocol is frozen in
[`docs/PACT_VS_ACT_REMEDIATION_PREREGISTRATION.md`](docs/PACT_VS_ACT_REMEDIATION_PREREGISTRATION.md).
The preregistered 32-D proximity-front-end development screen is reported in
[`docs/PACT_FRONTEND_SCREEN_DECISION.md`](docs/PACT_FRONTEND_SCREEN_DECISION.md).
Its distribution-matched validity follow-up is
[`docs/PACT_VALID_ABLATION_DECISION.md`](docs/PACT_VALID_ABLATION_DECISION.md),
with the conversion and zero-support audit in
[`docs/PACT_ACT_DATA_EQUIVALENCE_AND_ZERO_SUPPORT.md`](docs/PACT_ACT_DATA_EQUIVALENCE_AND_ZERO_SUPPORT.md).
The independent-seed replication decision is reported in
[`docs/PACT_SEED_REPLICATION_DECISION.md`](docs/PACT_SEED_REPLICATION_DECISION.md).
The sharpened contact-count endpoint is frozen in
[`docs/PACT_CONTACT_ENDPOINT_PREREGISTRATION.md`](docs/PACT_CONTACT_ENDPOINT_PREREGISTRATION.md)
and its final decision will be reported in
[`docs/PACT_CONTACT_ENDPOINT_DECISION.md`](docs/PACT_CONTACT_ENDPOINT_DECISION.md).
The held-out scene-geometry generalization gate and decision are reported in
[`docs/PACT_GEOMETRY_GENERALIZATION.md`](docs/PACT_GEOMETRY_GENERALIZATION.md).
The powered zero-shot geometry follow-up is reported in
[`docs/PACT_GEOMETRY_GENERALIZATION_V3.md`](docs/PACT_GEOMETRY_GENERALIZATION_V3.md).
The preregistered inference-time RGB-blur robustness sweep is reported in
[`docs/PACT_BLUR_SWEEP.md`](docs/PACT_BLUR_SWEEP.md).

The pick-and-place corridor Phase-0 expert gate is reported in
[`docs/PACT_PLACE_CORRIDOR_GATE.md`](docs/PACT_PLACE_CORRIDOR_GATE.md).
The expert-fix re-screen (5 mm release clearance; both gates failed) is reported in
[`docs/PACT_PLACE_EXPERT_FIX.md`](docs/PACT_PLACE_EXPERT_FIX.md);
the fresh-seed gate ledger is
[`docs/PACT_PLACE_CORRIDOR_GATE_V2.md`](docs/PACT_PLACE_CORRIDOR_GATE_V2.md).
The attempt-3 screen (named repairs; still 18/24 clean) is reported in
[`docs/PACT_PLACE_ATTEMPT3.md`](docs/PACT_PLACE_ATTEMPT3.md);
the v3 gate ledger is
[`docs/PACT_PLACE_CORRIDOR_GATE_V3.md`](docs/PACT_PLACE_CORRIDOR_GATE_V3.md).
The attempt-4 screen (empty-gripper disarm + persist-3; 15/24 clean) is reported in
[`docs/PACT_PLACE_ATTEMPT4.md`](docs/PACT_PLACE_ATTEMPT4.md);
the v4 gate ledger is
[`docs/PACT_PLACE_CORRIDOR_GATE_V4.md`](docs/PACT_PLACE_CORRIDOR_GATE_V4.md).
The attempt-5 screen (relocated tray; phase-aware receptacle exemption; 22/24
clean) is reported in
[`docs/PACT_PLACE_ATTEMPT5.md`](docs/PACT_PLACE_ATTEMPT5.md);
the v5 gate ledger is
[`docs/PACT_PLACE_CORRIDOR_GATE_V5.md`](docs/PACT_PLACE_CORRIDOR_GATE_V5.md).
The attempt-6 clutter arm (fixed shelf boxes beside the cup) stopped at A0:
no declared-grid set had zero expert `pact_clutter` contact, so Phase 0 was
not frozen or run. That record is
[`docs/PACT_PLACE_ATTEMPT6.md`](docs/PACT_PLACE_ATTEMPT6.md);
the ledger is
[`docs/PACT_PLACE_CORRIDOR_GATE_V6.md`](docs/PACT_PLACE_CORRIDOR_GATE_V6.md).
The attempt-6b resite (`|y| = 0.32` after naming the contacting body as the
carried cup) passed Phase 0 at **20/24** with zero `pact_clutter` contact.
That record is
[`docs/PACT_PLACE_ATTEMPT6B.md`](docs/PACT_PLACE_ATTEMPT6B.md);
the ledger is
[`docs/PACT_PLACE_CORRIDOR_GATE_V6B.md`](docs/PACT_PLACE_CORRIDOR_GATE_V6B.md).
The 24 qpos-replay clips are produced by
`scripts/run_pact_place_v6b_replay_videos.py` into
[`diagnostics_output/pact_place_corridor_v6b_videos/CRIB.md`](diagnostics_output/pact_place_corridor_v6b_videos/CRIB.md);
the MP4s are local and are not committed.
The attempt-6c presence pass (5×10×10 cm boxes, inner face held at `|y| = 0.29`,
28 mm closest-approach margin) passed Phase 0 at **23/24** with zero
`pact_clutter` contact. That record is
[`docs/PACT_PLACE_ATTEMPT6C.md`](docs/PACT_PLACE_ATTEMPT6C.md);
the ledger is
[`docs/PACT_PLACE_CORRIDOR_GATE_V6C.md`](docs/PACT_PLACE_CORRIDOR_GATE_V6C.md).
The 24 qpos-replay clips are produced by
`scripts/run_pact_place_v6c_replay_videos.py` into
[`diagnostics_output/pact_place_corridor_v6c_videos/CRIB.md`](diagnostics_output/pact_place_corridor_v6c_videos/CRIB.md);
the MP4s are local and are not committed. Do not edit the v5 or v6b renderers.
v1–v4 are `PACT_PLACE_CORRIDOR_PHASE0_FAIL`. v5 is
`PACT_PLACE_CORRIDOR_PHASE0_PASS`. v6 is
`PACT_PLACE_CORRIDOR_PHASE0_NOT_RUN`. v6b and v6c are
`PACT_PLACE_CORRIDOR_PHASE0_PASS`. v5 clean-success is not comparable to
v1–v4, and is not comparable to the stricter cluttered v6b/v6c counts. Join place-corridor rows on `(config_sha256, role_index)`, never on
`episode_id`. The original-seed re-run is a diagnostic, not a gate.
Rows 6 and 12 of the v2 gate are an initial-state panel overlap, not inbound
scraping; see [`docs/PACT_PLACE_HAZARD_ROWS_6_12.md`](docs/PACT_PLACE_HAZARD_ROWS_6_12.md).
The eight tracking-clean, zero-IK, `gripper_width_min_m = 0` failures across
v1/v2/v3 abort on the empty-gripper branch while the pads are still on the
cup; see [`docs/PACT_PLACE_ABORT_BRANCH.md`](docs/PACT_PLACE_ABORT_BRANCH.md).
The 24 attempt-3 episodes can be watched as qpos-replay clips (no physics
step, no expert re-run) from
[`diagnostics_output/pact_place_corridor_v3_videos/CRIB.md`](diagnostics_output/pact_place_corridor_v3_videos/CRIB.md);
the renderer is `scripts/run_pact_place_v3_replay_videos.py`. The MP4s are
generated locally and are not committed.
Collection was stopped at **152** kept of 174 attempted (target 255).

That first collection ran the Phase-0 *screen* harness rather than the datagen
pipeline, so it inherited two observation reductions: the polling suite is
truncated to `["qpos", "tcp_pose"]`, and `proximity_sensor_period_ms` is set to
`0.0`, which collapses the skin buffer to a single sub-step. The result was 152
valid screen records carrying no actions, no wrist RGB and no proximity — enough
to audit the expert, not enough to train anything. All 152 were re-recorded
through the datagen pipeline and reproduced their frozen outcome exactly,
**152/152 with zero divergence**, which also measures the full sensor suite as
physics-neutral for this task. The recovery is reported in
[`docs/PACT_PLACE_V5_DEMO_RECOVERY.md`](docs/PACT_PLACE_V5_DEMO_RECOVERY.md).

The published set is the recovered datagen, at
[huggingface.co/datasets/Lundii/pact_place_corridor_v5](https://huggingface.co/datasets/Lundii/pact_place_corridor_v5):
152 `trajectory.h5` with `traj_0/actions/*`, `obs/agent/{qpos,qvel}` and 40
proximity sensors at `(T, 4, 8, 8)`, plus per-episode wrist MP4s. It is raw
datagen, not ACT HDF5 — convert before training. Note the corridor has a wrist
camera only; there is no exo camera.

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

## Hybrid obstacle collection: episode manifest v2 (seeding contract)

The 40-sensor hybrid-obstacle collection is now driven by a **committed candidate
manifest** rather than by wraparound house indices. Read
`docs/HYBRID_OBSTACLE_SEEDING_FINAL_DECISION.md` before touching any of it.

Why: the earlier collection wrote 175 trajectories that represented at most 75
distinct episodes (50 replica classes of three). `seed_task_sampling` seeded
Python/NumPy/Torch once *per worker*, `get_episode_seed` returned that same
per-worker constant, houses were claimed dynamically, and hazard presence was a
runtime `np.random.random() < OBSTACLE_P` off the same shared stream — so an
episode's content was a function of worker state and execution order. That
collection is read-only and must never be used as canonical ACT training data.

Contract (`configs/hybrid_obstacle_independent_v2.yaml`): candidate `i`'s content
depends only on manifest version, master seed **20260725**, candidate index,
named stream ID, and retry index. Never on worker ID, worker count, scheduling,
house-claim order, house alias, file ordering, or a resume boundary. Ten named
streams are derived as `SeedSequence([master_seed, candidate_index, stream_id,
retry_index])` — order-independent, no `spawn()` chain. Identities are SHA-256;
Python's `hash()` is never used for a persistent ID or seed.

```
configs/hybrid_obstacle_independent_v2.yaml          # the frozen contract
configs/hybrid_obstacle_candidate_manifest_v2.json   # 160 rows, 120 hazard / 40 clear
                                                     #   sha256 8be804057f3e0710...
configs/hybrid_obstacle_manifest_v2_smoke8.json      # 8-row bounded smoke subset
                                                     #   sha256 cb9df6e1f8dabb2e...
scripts/build_hybrid_obstacle_manifest_v2.py         # regenerate/verify (--check)
scripts/run_hybrid_obstacle_manifest_v2.py           # THE launcher (ManifestRolloutRunner)
scripts/hybrid_obstacle_manifest_v2_audit.py         # A/B/C invariance audit, frozen tolerances
scripts/hybrid_obstacle_manifest_v2_provenance.py    # runtime provenance record
diagnostics_output/hybrid_obstacle_seeding/          # execution-path audit, RNG inventory,
                                                     #   invariance report, final_decision.json
```

Run it (env exactly as the datagen recipe in CLAUDE.md; needs mujoco==3.5.0 and
warp-lang==1.11.1, enforced at startup by `runtime_compat`):

```bash
PY=/root/act_retrain_venv/bin/python
$PY scripts/build_hybrid_obstacle_manifest_v2.py --check      # manifest still frozen?
$PY scripts/run_hybrid_obstacle_manifest_v2.py \
    --output-dir <fresh dir> --workers 8                      # full 160 rows
$PY scripts/run_hybrid_obstacle_manifest_v2.py \
    --output-dir <fresh dir> --workers 4 --smoke               # 8-row smoke only
```

Things that will bite you:

- **Every row rebuilds its scene on purpose.** `sample_task`'s scene-reuse branch
  is not equivalent to its load branch (it skips the `task_config` reset and the
  `update_scene` RNG draws), so reusing the cache reintroduces a first-row/later-row
  asymmetry. Roughly 60–90 s per row. Do not "optimise" it away.
- **Hazard presence is never drawn for this config**; it comes from
  `row.hazard_present`, and the compiled geometry is checked against the row.
  `OBSTACLE_P = 0.75` stays as the documented design probability (120/160) and
  every *legacy* obstacle config keeps its Bernoulli byte for byte.
- **Episode identity is `episode_id + manifest_row_sha256`**, written into H5
  metadata — not the `traj_N` buffer index. Resume keys on it. Worker ID and
  house alias appear only as `*_descriptive` fields.
- **A failed row is a recorded result**, not a retry opportunity, and rows are
  never reused to fill a success quota.
- Selection and split are predeclared: first 75 successful hazard-present and
  first 25 hazard-absent rows by stratum rank; train 60/20, val 15/5. They never
  inspect rollout quality.

Proved on a bounded smoke (8 rows, 24 executions): 1 worker vs 4 workers vs a
SIGKILLed-and-resumed 4-worker run are **bit-identical** by episode ID, with all
rows reconciling exactly once and no replica classes. See §14–§15 of the decision
doc. The claim holds for the recorded runtime only.

## Hybrid obstacle collection: the canonical 100-episode dataset (executed)

The frozen 160-row manifest above **has been executed**, audited, and reduced to
the predeclared canonical dataset. Read
`docs/HYBRID_OBSTACLE_FULL_COLLECTION_FINAL_DECISION.md` before using or
regenerating any of it.

Result: **160/160 rows** reached a terminal outcome and reconcile exactly once by
candidate index, episode ID and manifest-row hash. **145 succeeded** (110 of 120
hazard-present, 35 of 40 hazard-absent); the 15 failures are all scientific
`task_failure` — zero sampling failures, zero infrastructure failures, zero lost
workers. Both predeclared quotas (75 present / 25 absent) pass. No duplicate
episode ID, row hash, core-trajectory hash, task-state hash or episode-spec hash
exists across the 145 successes: **the largest replica class is 1**, so the
class-of-three defect that voided the 175-file collection does not recur.

```
assets/datagen/hybrid_obstacle_independent_v2/20260725_full160_4w/   # source, READ-ONLY
                                     # 1344 files, 1.045 GiB, all 160 rows incl. failures
                                     # tree sha256 8b569d0e20804949...
configs/hybrid_obstacle_canonical_manifest_v2.json   # controlled_predeclared_canonical_subset
                                                     #   100 rows = 75 present + 25 absent
                                                     #   sha256 f49f5cd14b3c75b8...
configs/hybrid_obstacle_canonical_split_v2.json      # train 60/20, val 15/5
                                                     #   sha256 f7c2b22718f1697e...
assets/act_style_data/hybrid_obstacle_canonical_v2/conversion_A/     # 100 ACT episodes
assets/act_style_data/hybrid_obstacle_canonical_v2/conversion_B/     # byte-identical rerun
scripts/hybrid_obstacle_full_collection_audit.py          # reconcile + H5/40-sensor + hashes A-E
scripts/hybrid_obstacle_smoke8_reference_compare.py       # smoke8 vs full run, frozen tolerances
scripts/hybrid_obstacle_build_canonical_subset.py         # quota gate + selection + split
scripts/hybrid_obstacle_convert_canonical_to_act.py       # manifest-driven ACT conversion
scripts/hybrid_obstacle_full_collection_validate.py       # the 18-check offline validation
scripts/hybrid_obstacle_write_final_decision.py           # generates the decision MD + JSON
diagnostics_output/hybrid_obstacle_full_collection/       # every report + final_decision.json
```

The exact commands are in §18 of the decision doc. Things worth knowing:

- **The source collection is read-only and includes the failures.** It is the
  primary provenance record. Do not delete failed or non-canonical rows.
- **The committed ACT converter cannot be invoked directly on this layout.**
  `scripts/convert_obstacle_to_act.py` globs `house_*/trajectories*.h5` and expects
  `episode_<i>_<cam>_batch_1_of_1.mp4`; the manifest runner writes
  `rows/<episode_id>/trajectory.h5` and `episode_00000000_<cam>.mp4`, and the
  committed entry point assigns the ACT episode index from filesystem order.
  `hybrid_obstacle_convert_canonical_to_act.py` imports and reuses that module's
  decode/video functions verbatim and takes the episode set *and* index from the
  canonical manifest instead. The converter and ACT are unmodified.
- **10 of the 100 selected rows sit past the manifest's own `reserve` boundary.**
  That is the selection rule working as predeclared: when an earlier-ranked row
  fails, the next successful row by stratum rank is promoted. The split is then
  derived from position within the selected set, which reproduces the manifest's
  own `split` column exactly when nothing fails.
- **`collection_summary.json` carries a stale `warning: "Partial output
  retained…"` even on a fully successful run.** `build_final_summary` derives
  `complete` from a house-based comparison and a manifest run writes no houses, so
  it always sets that key; the runner then overrides `complete`/`status` from row
  reconciliation but never deletes it. The validated smoke reference carries the
  same string. Trust `complete`, `status`, `row_reconciliation.ok` and
  `workers.complete`, not `warning`.
- **Next step is ACT training**, in its own approved task: point
  `obstacle_baseline` at `conversion_A` with `num_episodes=100` and
  `episode_len=132` (max T = 130). Nothing in this task trained ACT or the
  Safety-CVAE, or ran any evaluation.

## Hybrid obstacle ACT baseline: the pinned nominal policy (trained)

The canonical vanilla ACT baseline has been trained on the 100-episode dataset
above and pinned. Read `docs/HYBRID_OBSTACLE_ACT_BASELINE_FINAL_DECISION.md`
before using or retraining it.

**This one checkpoint is the nominal policy for BOTH later conditions** — ACT
alone, and the same checkpoint plus the 40-sensor Safety-CVAE residual. Do not
train a second nominal ACT for the safety arm; that would confound the
comparison.

Result: seed 0, 2000 epochs, batch 8, 49m 47s on an A10. Validation loss
77.84 → **0.171486, best at epoch 1738**. Offline teacher-forced MAE on the fixed
20 validation trajectories: 0.4429 normalized, arm joints 0.0894 rad,
gripper 55.07 on the 0–255 actuator scale. Those are imitation metrics only — no
task success, collision or safety measurement was made.

```
configs/hybrid_obstacle_act_baseline_v2.yaml                 # frozen training contract
scripts/run_hybrid_obstacle_act_baseline_v2.py               # builds the command from the contract
scripts/hybrid_obstacle_act_baseline_offline_eval.py         # teacher-forced offline metrics
submodules/act/fixed_split_data.py                           # committed-split loader, train-only stats
submodules/act/tests/test_fixed_split_loader.py              # 38 tests (56 with the existing suite)
diagnostics_output/hybrid_obstacle_act_baseline/             # audit, smoke, curves, metrics, pin
/root/act_retrain_assets/act_ckpts/hybrid_obstacle_act_baseline_v2/20260725_seed0_2000ep/
                                                             # 8.8 GiB of checkpoints, NOT in git
                                                             #   policy_best.ckpt dd7cd108a64ce10e...
                                                             #   policy_last.ckpt f952f8f0887bfc5d...
```

Why the ACT loader had to be repaired first (all five were real, all confirmed by
reading `3d25c69`):

- **`utils.load_data` generated its own random 80/20 split** with
  `np.random.permutation`, ignoring the committed split entirely — and seeded it
  from `set_seed(1)`, not `--seed`.
- **Normalization statistics were computed over all 100 episodes**, so the 20
  validation trajectories leaked into the qpos/action mean and std.
- **The validation loader was shuffled**, so validation loss was not reproducible.
- **`constants.py` `obstacle_baseline` pointed at `/home/jaydv/...obstacle_v1`**
  with `episode_len` 169 — a machine-specific path to the superseded collection.
- **There was no resume**: periodic saves wrote the model state dict only, with no
  optimizer state, RNG state or epoch counter, and were not atomic.

Things worth knowing:

- **Pass `--split_manifest` to get the fixed path.** Without it, `imitate_episodes.py`
  keeps its legacy behaviour for every existing task. The manifest path takes
  `--dataset_dir`, so no machine-specific path is committed to `constants.py`.
- **The loader fails closed.** It verifies the split manifest's self-hash, every
  converted file's hash, the converted tree hash, and the on-disk tensor contract
  (sim flag, qpos 9, action 8, both cameras, horizon) before the first batch.
- **Validation windows are fixed across epochs**, unlike the fork, which redrew
  them every epoch. That makes the validation curve comparable epoch to epoch and
  stops best-checkpoint selection from rewarding a lucky window draw.
- **Bit-identical weights are not claimed.** Two fresh runs with the same seed
  already differ by ~1e-4 after two epochs from nondeterministic cuDNN kernels;
  resume was measured to add nothing beyond that. Epoch-0 validation loss *is*
  bit-identical across runs, so the data path itself is deterministic.
- **Training overfits after ~epoch 900** (train 0.068 vs val 0.18 at the end).
  That is why `policy_best.ckpt` is epoch 1738 and not the last epoch.
- **Next step is paired evaluation**, in its own approved task: verify
  `checkpoint_manifest.json`, then roll out `policy_best.ckpt` with and without
  the Safety-CVAE residual. Nothing in this task ran a rollout.

## Hybrid obstacle safety residual: what the Safety-CVAE reference must be

Four evaluation tasks have now run against the pinned checkpoint above. Read the
decision documents before touching the safety stack; each supersedes the last on
the points it covers.

| task | document | outcome |
|---|---|---|
| paired smoke | `docs/HYBRID_OBSTACLE_PAIRED_SMOKE_FINAL_DECISION.md` | adapter validated on four pairs |
| observation reference | `docs/HYBRID_OBSTACLE_OBSERVATION_REFERENCE_FINAL_DECISION.md` | `first_live_skin` is **not** canonical |
| raw head | `docs/HYBRID_OBSTACLE_RAW_HEAD_QUALIFICATION_FINAL_DECISION.md` | `RAW_HEAD_CONTROLLER_GROSS_REGRESSION` |
| parked oracle | `docs/HYBRID_OBSTACLE_ORACLE_REFERENCE_FINAL_DECISION.md` | `ORACLE_REFERENCE_VALID_CONTROLLER_VIABLE` |

The documented reference is `dq = head(skin with obstacle) − head(skin, obstacle
parked)` (README §"the safety head" above, and `scripts/safety_react_demo.py:369-379`).
It is now implemented per frame in `submodules/act/parked_obstacle_reference.py`
and driven by `submodules/act/eval_act_obstacle_oracle.py`.

Things worth knowing before writing another reference:

- **The parked oracle is privileged and is not deployable.** It moves a scene body
  the robot cannot move. It exists to measure whether the Safety-CVAE carries a
  hazard-specific differential at all — it does — not to be shipped.
- **Subtracting nothing does not work.** The raw head fires on any close surface,
  so unsubtracted it produces a persistent geometry-driven push: pooled
  hazard-present task success went 11/15 → 0/15 with large hazard-bar contact
  counts. It was not saturation; the correction never clipped.
- **Subtracting a step-0 reference does not work either**, and a *recorded*
  reference is worse: it is indexed by step into a finite array and dies when the
  recording runs out (candidate 106's demonstration is 109 frames against a
  200-step horizon). Do not pad or wrap it — that hides the real fault, which is
  that a recording does not describe the current pose.
- **Pair the two renders at the same simulator state.** MolmoSpaces' last
  proximity sub-step render lands one sim sub-step before the policy step ends, so
  the observation's "latest" proximity is at a slightly earlier pose. Pairing it
  against a counterfactual taken at the decision state invents a differential of
  up to 2.5 on a row with *no hazard at all*, where the correct pairing gives
  exactly zero.
- **Do not call `mj_forward` before the counterfactual render**, even though the
  demo does. The demo poses its scene by hand and never integrates; here,
  `mj_step` integrates `qpos` after the forward dynamics, so every body's `xpos`
  lags by one sub-step and `mj_forward` would move the whole scene (~2.5e-4 m over
  23 bodies) before the parked render. Translate the hazard mocap body's render
  state instead — no dynamics call, so the counterfactual is invisible to the
  integrator by construction.
- **`minimum_clearance_m` is not a clearance.** `mj_geomDistance` returns exactly
  0.0 for `robot_0/fr3_link7_collision`, and the audited adapter takes a minimum
  over all robot × environment pairs, so that one geom pins the reported figure at
  ≤ 0 in every condition and every prior task. Use it as "deepest penetration
  observed" only; safety claims should rest on contact classification from
  `data.contact`. Evidence:
  `scripts/hybrid_obstacle_geom_distance_probe.py`.
- **`confirmatory41` has never been executed.** The oracle evaluator hard-refuses
  any manifest whose role is `CONFIRMATORY_UNTOUCHED`.

## Hybrid obstacle safety residual: the deployable reference attempt

`docs/HYBRID_OBSTACLE_DEPLOYABLE_REFERENCE_FINAL_DECISION.md` —
`DEPLOYABLE_REFERENCE_LIVE_GROSS_REGRESSION`.

A posture+skin MLP was trained to predict the parked SafetyHead output from
runtime-observable signals only, so the oracle's subtraction could be done without
privileged information. It passed every offline gate on held-out trajectories and
then failed live.

- **Posture alone is not enough.** The predeclared KNN control over (qpos, qvel,
  gripper) gives a median oracle cosine of **−0.39** and a differential MAE worse
  than doing nothing clever. Skin information is doing the work.
- **Offline looked excellent**: the MLP reached median oracle cosine **+0.977** and
  differential MAE 0.283 against 0.660 (raw head) and 0.657 (first-live) on 20
  held-out trajectories.
- **Live it collapsed**: pooled median cosine **0.345**, magnitude over-predicted by
  up to **6.9×**, and the controller fired on 23–34 % of frames where the true
  differential was exactly zero. On the hazard-absent row it produced new
  environment contact in all five repeats.
- **Root cause is distribution shift, not architecture.** The reference trains on
  expert planner trajectories that deflect around the hazard; on all four
  development expert trajectories the predicted norm never once reached the quiet
  threshold, so the gate opened zero times offline. Live ACT trajectories drive
  oracle differentials up to 2.5 and the model extrapolates.
- **Task success did not regress** (12/15 vs ACT-only 11/15, oracle 14/15), and on
  candidate 107 the deployable reference cut environment contacts from 75–93 per
  rollout to 5–28 — better than the oracle. The mechanism is real; the
  approximation is not yet good enough.

Two measurement traps worth knowing before designing the next study:

- **The support gate is nearly always open here.** No validation frame has a
  minimum valid depth above 0.187 m, so the "far frame > 0.25 m" false-activation
  metric has *zero* frames and gate A fires on 76 % of frames. The quiet threshold
  does all the work.
- **That threshold guarantees its own gate.** `tau` is the 99.5th percentile of the
  hazard-absent validation norm, so ~0.5 % exceedance is true by construction, not
  measured. Calibrate it against an on-policy false-activation target instead.

Next: build the paired dataset **on-policy** — ACT_ONLY rollouts on the training
rows, parked counterfactual along those trajectories through the same state-neutral
seam — and retrain the same fixed architecture on the union. `confirmatory41` is
still untouched.

## Hybrid obstacle safety residual: the on-policy aggregation round

`docs/HYBRID_OBSTACLE_ON_POLICY_REFERENCE_FINAL_DECISION.md` —
`ON_POLICY_REFERENCE_OFFLINE_INVALID` (Case C).

One DAgger-style aggregation round was run against the previous task's
distribution-shift diagnosis: 200 on-policy labelling rollouts (ACT-only and
privileged-oracle, per canonical row), then 64 learner-induced rollouts, then a
retrain of the *unchanged* architecture on four equally weighted distributions.

- **The diagnosis was right and the round worked.** Round 1 halves the differential
  MAE on both on-policy distributions (0.313 → 0.208 and 0.355 → 0.151) and turns
  the median oracle cosine on ACT-only on-policy validation frames from **−0.08 to
  +0.88**.
- **It still cannot be given an activation contract.** On the 8 calibration
  trajectories no threshold satisfies the predeclared rule: holding recall ≥ 80 %
  leaves the median cosine at ~0.61, reaching cosine ≥ 0.70 drops recall to 0.72,
  and the positive-cosine fraction never reaches 80 % above 4 % recall.
- **The failure concentrates on oracle-on-policy states** (calibration cosine 0.501
  vs 0.72 elsewhere) — the states a *correct* reference creates for itself. A
  contract fitted anywhere else would be optimistic.
- No live rollout was run: the offline stage failed, so the 20-rollout schedule was
  frozen and left unexecuted, and the frozen ACT-only/oracle/V1 baselines were not
  touched.

Practical notes for the next attempt:

- **Don't run another round on this feature contract.** The round improved
  everything measurable and the contract still didn't close.
- **Predict the parked *skin*, not the parked head.** Pushing a predicted 40×8×8
  depth field through the frozen head keeps the head's structure instead of asking a
  150 k-parameter MLP to imitate its 7-number output — a target that equals its own
  input on most frames.
- The four causal skin frames are already collected in every paired shard and are
  currently unused; the per-sensor summary discards intra-patch layout entirely.
- **Calibrate on oracle-on-policy states specifically.** That is where the model is
  weakest and where a deployed reference will spend its time.
- `confirmatory41` is still untouched, and still disjoint from every partition.

## Hybrid obstacle safety residual: the parked-skin reference (blocked on data)

`docs/HYBRID_OBSTACLE_PARKED_SKIN_REFERENCE_FINAL_DECISION.md` —
`PARKED_SKIN_DATA_CONTRACT_FAILED`.

The plan was to stop predicting the 7-D parked SafetyHead output and predict the
parked 40×8×8 **field** instead, routing it through the frozen head. The task
stopped at its own data-contract audit: **the training target does not exist.**

- **0 of 60,793 paired frames** carry both a causal current skin and a parked skin.
  The parked field was never retained anywhere — only its SHA-256 and the 7-D head
  output derived from it, neither invertible to the field.
- Causal current skins exist only for the 7,993 expert frames (13%). The three
  on-policy distributions store the 40×4 per-sensor summary the *previous* model
  consumed, not the fields this one needs.
- Not corruption — a scoping consequence. Retaining both raw fields would have added
  ~2.9 GiB that the previous model never used.

**If you add a field to a paired dataset, retain the raw counterfactual too.** The
oracle already renders the parked field every frame inside
`PerFrameParkedObstacleReference`; the previous collection hashed it and threw it
away. Keeping it is free in compute and only costs disk.

Delivered anyway, so the next attempt only needs data:

- `submodules/act/parked_skin_reference.py` — `CAUSAL_PARKED_SKIN_REFERENCE_V1`,
  331,713 parameters, with `0 ≤ c_parked ≤ c_current` guaranteed **by construction**
  (verified even with the change head's bias driven to +50).
- The closeness transform (`clip(1 - depth/0.5, 0, 1)`, dead pixels <5 mm → 0), the
  four-frame causal buffer, an activity gate that gates on activity probability
  rather than predicted norm, and a strict loader.
- 54 contract tests.

To regenerate: 364 rollouts (100 expert paired, 200 on-policy labelling, 64
learner-induced), ~2.9 GiB. Cut it by storing float16 closeness instead of float32
depth, and keeping fields only for oracle-active frames plus a sample of zero
frames. Note a rerun is a *new sample* under MSAA — reuse the partition and
schedules, but expect different frames.

`confirmatory41` is still untouched.

## Hybrid obstacle safety residual: the parked-skin dataset (collected and frozen)

`docs/HYBRID_OBSTACLE_PARKED_SKIN_DATASET_FINAL_DECISION.md` —
`PARKED_SKIN_DATASET_READY_FOR_MODEL_TRAINING`.

The missing target from the section above now exists. 364 outputs / 60,793 frames,
frozen at `assets/reference_data/hybrid_obstacle_parked_skin_supervision_v1/`
(gitignored; tree sha256 `1fb68b3c…db1dba5`, every file mode 444). Zero collection
failures.

| Distribution | Files | Frames |
|---|---:|---:|
| `EXPERT_RECONSTRUCTED` | 100 | 7,993 |
| `ACT_ONLY_ON_POLICY` | 100 | 20,000 |
| `ORACLE_ON_POLICY` | 100 | 20,000 |
| `LEARNER_INDUCED_ON_POLICY` | 64 | 12,800 |

**The ~2.9 GiB estimate above was wrong by 12×, and the mitigation it proposed was
unnecessary.** Actual size is 0.23 GiB with *every* frame retained — no active-frame
filtering, no zero-frame subsampling. Two changes did it, both in
`submodules/act/parked_skin_retention.py`:

- Store closeness (`clip(1 - depth/0.5, 0, 1)`) rather than raw depth, alongside an
  explicit validity mask. Sub-5 mm readings map to 0 closeness **and** are flagged
  invalid — dead pixels and true contact otherwise look identical, and a model would
  learn to read sensor dropout as an obstacle.
- Store only the contiguous current-field sequence. The four-frame window is rebuilt on
  load: `history(t) = [t-3, t-2, t-1, t]`, left-padded by repeating the earliest frame,
  never crossing a trajectory boundary. Materialising windows holds four copies of every
  frame for no information.

So the retention lesson generalises further than it first looked: **the reason to throw
data away is usually a storage layout you haven't fixed yet.** Discarding oracle-zero
frames would have destroyed the majority class (46,382 of 60,793) — and which states are
quiet is itself the signal. Balancing belongs to the training sampler, not the collector.

Three field groups: `deployable/` (runtime-observable inputs), `privileged/` (targets —
`load_trajectory` refuses to return these without `allow_privileged=True`), `integrity/`
(hashes, state-neutrality). The split is enforced at load so a deployable loader cannot
accidentally consume its own label.

Integrity over all 364 files: 0 duplicate identities, 0 state-neutrality failures, 0
violations of `0 ≤ parked ≤ current ≤ 1` (nothing silently clamped), 0 non-causal
histories, 0 nonfinite values. Re-running the frozen SafetyHead from the *stored* fields
reproduces the recorded 7-D targets and the oracle differential at max abs delta
**exactly 0.0** — on-policy rows retain the same parked array the oracle already
rendered, because a second render lands at a different physics substep and would break
the pairing silently.

Hazard-absent frames (15,025) are an **exact** control: fields bitwise equal, changed-pixel
mask empty, differential exactly zero. That is the sharpest available test that parking
perturbs only the hazard and leaks nothing into scene state.

Two operational notes. Concurrency is capped at **two** rollout processes — five lanes
under GPU contention is what killed a shard during the on-policy round. And expert rows
are reconstructed by open-loop replay of the recorded commands, *not* by pose-setting:
the trajectory H5 stores no per-step pose for the pickup object, so restoring only the
robot leaves the target at its rest pose and pairs the parked field against a scene that
never existed.

Not yet established: that the dataset is *useful*. Nothing here shows
`CAUSAL_PARKED_SKIN_REFERENCE_V1` can learn the parked field from these inputs, or that
it would close the activation contract `V2` failed. Training, `development4` and
`confirmatory41` were all out of scope — `confirmatory41` remains untouched.

## Hybrid obstacle safety residual: the parked-skin reference (trained, offline)

`docs/CAUSAL_PARKED_SKIN_REFERENCE_V1_FINAL_DECISION.md` —
`PARKED_REFERENCE_MODEL_OVERFIT`.

**The dataset is useful and the field is learnable.** The open question above is answered:
offline SafetyHead-space differential MAE drops from the trivial baseline's 0.042062 to
**0.011337** — a **73% reduction** against a 25% gate — with median direction cosine
**0.999** against a 0.75 gate, 94–98% recall on oracle-active frames, and zero
output-constraint violations. The privileged upper bound (true parked field through the
frozen head) is 3.8e-08, so essentially all the recoverable signal is there.

7 of 8 technical gates and 7 of 8 generalization gates pass. The decision token is set by the
one failure: oracle-zero false-positive rate 2.15% on seed 0 against a ≤2% ceiling.

**Read the token carefully — this is not overfitting.** Offline-test MAE (0.0091) is *less
than half* validation MAE (0.0208); there is no validation-to-test gap. The failure is
**threshold transfer**. The activation threshold is the 99th percentile of oracle-zero norms
on the 8-episode / 3,821-frame calibration partition, which gives 1.01% there by
construction, 2.15% on offline test and 2.61% on validation. Across seeds the thresholds
themselves vary by CV 0.72. **Calibrate on more than one partition, and target a false-
positive rate below the ceiling you have to hold** — a quantile fitted on ~3.8k frames does
not transfer tightly enough to sit right on a 2% line. No retraining is needed to fix this.

Three ablations, same targets, partitions, checkpoint rule and evaluation code:

| Model | Offline MAE | Seed CV |
|---|---:|---:|
| `CURRENT_FRAME_ONLY` — **frozen primary** | **0.011337** | 0.177 |
| `FULL_CAUSAL` (4 frames) | 0.012809 | 0.381 |
| `QPOS_ONLY` (state only) | 0.048446 | 0.034 |
| `ZERO_DIFFERENTIAL` | 0.042062 | — |

**Temporal history does not help — drop it.** Four causal frames are 13% *worse* than the
current frame alone and twice as variable across seeds. The hazard is a *parked* obstacle,
static by construction, so its whole signature is in frame `t`; the extra frames buy
parameters and variance. The handoff's alternative branch applies: the simpler model is
frozen and every remaining gate was recomputed against it rather than inherited. A live
deployment can drop the four-frame buffer entirely.

**Proximity is essential, robot state is not sufficient.** `QPOS_ONLY` is *worse than
predicting no obstacle at all*. Posture tells you where the arm is, not where the hazard is.

Two implementation notes worth carrying forward. The BCE positive weight must be capped:
uncapped at `negatives/positives ≈ 1200` the mask head fires field-wide, and the frozen head
turns a ~4e-3 per-pixel error into a ~0.7 differential error — a validation MAE of 0.2739,
far worse than silence. And the state-only control must not receive the current field
through its decoder, or it quietly becomes a worse `CURRENT_FRAME_ONLY` instead of a control.

Error is confined to the distal links — `link5_back` 4.8e-04, `link6` 3.1e-04, and exactly
0.00 on `link1`, `link2`, `link4`, which never see the hazard.

Two limits on what this establishes. `LEARNER_INDUCED_ON_POLICY` exists only in the train
partition, so generalization to learner-induced states — the distribution that broke the
previous reference — is **unevaluable** from this dataset. And re-running an identical
config and seed moved validation MAE by 15.2% (non-deterministic GPU kernels), so the 17.7%
across-seed CV is substantially kernel noise.

These are offline feasibility results. No live rollouts were run, ACT and the Safety-CVAE
were untouched, and `confirmatory41` is still untouched.

## Hybrid obstacle safety residual: trajectory-aware threshold (transfer failed)

`docs/HYBRID_OBSTACLE_REFERENCE_THRESHOLD_FINAL_DECISION.md` —
`REFERENCE_THRESHOLD_TRANSFER_FAILED`, Case C.

The previous `PARKED_REFERENCE_MODEL_OVERFIT` token was a rubric artefact, not a diagnosis —
test MAE was *below* validation MAE. The real blocker was activation-threshold transfer.
This task fixed the threshold statistics properly and found the model still fails, for a
reason no threshold can fix.

**The frame-level percentile threshold was wrong and is retired.** Treating 3,821
autocorrelated frames as independent observations overstates a quantile's precision badly;
whole trajectories share a scene, a hazard pose and a policy, so the effective sample size is
nearer the trajectory count than the frame count. Its ~1% calibration exceedance was a
construction artefact, which is why it read 2.15% elsewhere.

Replaced with: metrics computed **per trajectory, never pooled first**, and a **cluster
bootstrap resampling whole episodes** (10,000 replicates, seed 20260727, one-sided 95%
bound). Episodes are the cluster, not files — the same episode appears in three source
distributions, so resampling files would smuggle the independence assumption back in.

On `threshold_calibration16` (16 episodes / 48 trajectories / 7,731 frames) the fit is
excellent: threshold **0.99960858**, bootstrap upper FPR bound **0.00000**, median recall
0.882, median cosine 1.000, **zero** false positives in all 48 trajectories.

**It fails on held-back data anyway.** Two blocking checks trip, both from one trajectory —
a *hazard-absent* episode where current and parked fields are bitwise identical and the true
differential is exactly 0.0 — on which the model fires for **7 consecutive frames** at
activity 0.999999, predicting a differential norm rising to 0.390.

**Why no threshold repairs it:** silencing that run needs a threshold above 0.99999905, which
would discard **59% of all genuinely active frames**. The false activation sits *above* most
true activations in activity.

**The mechanism is episode onset.** Across all three partitions there are 17 false-positive
frames at the selected threshold: **16 are at frame index 0–6, all 17 are in the first 10% of
their trajectory, and there are none anywhere else.** Every episode starts from the same home
posture, and in hazard-present episodes the obstacle is there from frame 0 — so the model
appears to have learned a posture-onset prior and applies it before the proximity field
becomes discriminative.

**Watch the aggregate/cluster gap.** Mean trajectory FPR on the diagnostic set is 0.30% and
the bootstrap upper bound 0.68%, both comfortably inside the 2% target. Pooled over frames
this model looks calibrated and would have gone to a live run; clustered by trajectory it
shows a sustained false activation on a provably clear scene. **Keep the trajectory-aware
machinery — it is what caught this.**

Also confirmed: frozen inference is **bit-identical** over 24 repeats (activity, parked
field, delta and head differential all exactly 0.0 drift), so the earlier training
nondeterminism does not touch the deployed forward pass.

No live rollouts were run (0 of 20 permitted) — step 9 requires stopping before live when
these checks trip. The threshold manifest is written but carries
`authorized_for_live: false`, and its strict loader refuses to hand it to an evaluator. No
model was retrained, no seed reselected, no controller constant changed. `confirmatory41`
remains untouched.

**Next change must be to the activity model or training objective, not the threshold.**

## Hybrid obstacle safety residual: proximity-only activity gate (infeasible)

`docs/HYBRID_OBSTACLE_PROX_ACTIVITY_GATE_FINAL_DECISION.md` —
`PROX_ACTIVITY_GATE_CALIBRATION_INFEASIBLE`, Case B.

**The previous task's hypothesis was wrong, and causal intervention says so.** The onset
false activations were attributed to a proprioceptive/episode-onset prior. Running eight
interventions on the frozen checkpoint refutes it: with real state and the proximity field
replaced by a clear reference, activity on the 17 known false positives collapses from
0.9999 to **0.0229**. Shuffling state or replacing it with the batch mean moves nothing
(0.9999 either way). **State alone does not fire the gate.** Classification:
`PROXIMITY_AMBIGUITY_DOMINANT`.

The structural audit still condemns the old design: **there is no dedicated activity head**.
State is summed into every sensor token before the cross-sensor transformer, the shared
decoder emits the mask-logit channel, and "activity" is a max-reduction of that same
channel. Activity and the parked field are siblings from one decoder — entangled by
construction.

So a proximity-only gate was built to spec (90,241 params, budget 250,000), trained once on
seed 0 over a fresh 40/8/8/8 nested episode partition, onset-zero frames weighted 4× with a
trajectory-level onset-**max** penalty (a mean would score seven clustered activations the
same as seven scattered ones — the clustered case is the dangerous one).

**No feasible threshold exists.** Recall ≥ 0.80 and mean trajectory FPR ≤ 0.02 are not
jointly attainable anywhere: at τ=0.50 recall 0.802 but FPR **24.7%**; at τ=0.99 FPR 1.1%
but recall **0.022**. 3,502 of 5,447 candidates clear the recall screen; **zero** are
feasible.

**This is an information limit, not underfitting** — measured, not assumed:

| Split | Gate AUROC | Old head AUROC |
|---|---:|---:|
| gate_training | 0.9893 | 0.9982 |
| threshold_calibration (held out) | **0.7859** | **0.9998** |
| nested_offline_evaluation | 0.8734 | 0.9979 |

The gate fits its training split and fails to transfer; the old head — which sees proximity
**and** state — is near-perfect on the same held-out episodes. **Current proximity alone
does not identify removable hazard evidence.** The causal audit showed proximity is
*necessary* (clearing it extinguishes activation); it never showed it *sufficient*.

Two things worth carrying forward. The old head is already at AUROC 0.998 held out — its
defect is 17 rare, clustered, onset-located false positives out of ~22,000 frames, not a
separation failure, so replacing it wholesale trades a rare failure for a common one. And
the prohibition on state input is *what makes* the isolated gate weak; any future design
that keeps state out of the activity path needs a different disambiguator — uncertainty over
the predicted parked field is the obvious candidate, since the frames that fool the gate are
exactly those where the field is compatible with both a hazard and a clear scene.

The task stops at calibration per its own rules: **0 of 20 permitted live rollouts**, no
nested evaluation, no runtime integration. The parked-field model was not retrained, no seed
reselected, no controller constant changed, dataset untouched. `confirmatory41` remains
untouched.

## Hybrid obstacle safety residual: activity identifiability (uncertainty works)

`docs/HYBRID_OBSTACLE_ACTIVITY_IDENTIFIABILITY_FINAL_DECISION.md` —
`EPISTEMIC_UNCERTAINTY_SIGNAL_PRESENT`.

**The observation contract is sufficient after all, and ensemble disagreement finds the
failures.** Read-only audit; nothing trained, no thresholds changed, no rollouts.

Disagreement among the three frozen parked-field seeds separates the 17 historical false
positives cleanly. **Changed-pixel-mask agreement rejects 17/17 while retaining 96.5% of
active frames and 94.4% of hard-true-active** (AUROC 0.979, pAUC@5% 0.783). Median agreement
is 0.167 on the failures vs 0.71–0.78 on genuine activity — the seeds disagree about *which
pixels move* exactly where the head is wrong.

**No collisions anywhere.** Across all 60,793 frames: 0 exact current-proximity, 0 exact
full-deployable-input, 0 near-identity collisions with opposite labels (tolerances 1e-5
closeness, 1e-6 state/action). No pair shares an observation and disagrees about the truth.

**The ambiguity is asymmetric, and it runs the other way.** Opposite-label fraction among the
8 nearest neighbours in raw proximity:

| Group | Opposite fraction |
|---|---:|
| **HISTORICAL_FALSE_POSITIVE** | **0.0000** |
| ONSET_ZERO | 0.0030 |
| LATE_ACTIVE | 0.4800 |
| **ONSET_ACTIVE** | **0.7165** |

The 17 failures sit in unanimously quiet neighbourhoods — their observations *do* determine
their labels; the head was simply wrong. The genuine ambiguity is onset-**active** frames
looking like quiet ones, which causes missed detections, not spurious corrections.

**A measurement trap worth remembering.** Raw between-class embedding distance is 5.6 at
onset vs 16.5 late, which reads as severe onset overlap. Against the within-class baseline it
reverses: separation ratio **1.121 onset vs 1.037 late** — onset is *better* separated. The
raw gap was pure embedding-magnitude compression. Always divide by the within-class baseline.

Two caveats on the record: **n = 17** bounds every separability estimate, and one metric
(`norm_coefficient_of_variation`) passes the gate's letter while being useless — its median
on ordinary quiet frames (0.90–1.04) exceeds its median on the failures (0.708), so it would
abstain almost everywhere. That is a hole in the gate, which constrains active recall only;
it is flagged and excluded, and any successor contract should add a quiet-frame retention
floor.

Score tails at the frozen threshold (trajectory-clustered bootstrap): FPR 0.0012
(CI 0.0003–0.0025), active recall 0.770 (CI 0.671–0.853); 99 of 108 trajectories have no
false positive, one has a run of 7.

Next: one predeclared trajectory-bootstrap ensemble, uncertainty used **only** for
abstention, seed-0 mean model retained, qualified on development4 before confirmatory41.
`confirmatory41` remains untouched.

## Hybrid obstacle safety residual: uncertainty abstention (bootstrap collapsed)

`docs/HYBRID_OBSTACLE_UNCERTAINTY_ABSTENTION_FINAL_DECISION.md` —
`UNCERTAINTY_ABSTENTION_CALIBRATION_INFEASIBLE`, Case C.

All five bootstrap members trained and strict-loaded cleanly (0 constraint violations, all
well under the 0.0715 trivial baseline). **No agreement threshold satisfies the contract**,
and the reason is that swapping seed-variance for data-variance destroyed the signal.

**Trajectory-bootstrap disagreement is not seed disagreement.** Seeds 0/1/2 were each trained
on the *full* training set, so their disagreement isolated seed variance at fixed data. The
bootstrap members see 24–28 unique clusters each, so their disagreement is dominated by how
much data they happened to get:

| Anchor-vs-member agreement | Three seeds (full data) | Trajectory bootstrap |
|---|---:|---:|
| Genuine active frames | 0.71–0.78 | **0.5467** |
| Oracle-zero frames | — | 0.6000 |
| Historical false positives | 0.167 | — |
| **Separation** | **≈0.55** | **+0.053** |

Members agree with *each other* at only 0.41. Any threshold low enough to keep genuine
activations accepts nearly everything; any threshold high enough to abstain meaningfully
throws away genuine activations too.

**A second, independent failure sits underneath.** At agreement threshold 0 — abstaining on
nothing — median active recall is **0.786**, already below the 0.80 floor. That is the frozen
activity threshold alone (range 0.152–1.000 over 18 trajectories), which this task may not
refit. *Even a perfect uncertainty metric would have failed this contract on this split.*
Both are reported, because acting on one alone would mislead.

**The anti-degeneracy floors earned their place.** The previous task recommended a
quiet-frame acceptance floor; it was added and is exactly what fails first as the threshold
rises. Final false activation is 0.000 at *every* threshold, so on false-activation grounds
alone a threshold near 0.6 would have looked attractive while abstaining on half of every
trajectory. Don't weaken these to make something feasible.

No deployment manifest written, 0 of 20 live rollouts, `development4` not executed,
`confirmatory41` untouched. Seed 0 unaltered, no averaging, no member substitution — the ACT
`ACT_PLUS_UNCERTAINTY_ABSTENTION` condition is committed but never ran.

Next: decide explicitly whether the 0.80 recall floor or the frozen threshold gives way (they
are currently incompatible), and if uncertainty is retried, **keep the data fixed and vary
only the seed** — that is the construction the evidence actually supports.

## Hybrid obstacle safety residual: full-seed joint gate (offline transfer failed)

`docs/HYBRID_OBSTACLE_FULL_SEED_JOINT_GATE_FINAL_DECISION.md` —
`FULL_SEED_JOINT_GATE_OFFLINE_TRANSFER_FAILED`, Case C.

**Joint calibration works, and it vindicates the owner's diagnosis.** Recalibrating the
activity threshold *together with* the agreement threshold — instead of inheriting one fitted
for a single-gate controller — restores full coverage without touching the 0.80 recall floor:

| | Retired standalone | Jointly calibrated |
|---|---:|---:|
| Activity threshold | 0.99960858 | **0.99154764** |
| Activity-alone active retention | 0.786 | **1.000** |
| Final active recall | — | **1.000** |

1,690 feasible (activity, agreement) pairs out of a 5,084 × 266 Cartesian grid; bootstrap
upper bound on false activation 0.00000. **The recall floor was never lowered** — the
threshold moved because the deployed system is now a two-gate controller.

**Offline transfer then fails, and only on clustering.** Nested recall 0.997, diagnostic
recall 0.970, mean executed false activation 0.0044 / 0.0026 — all comfortably inside their
ceilings. What fails: nested false-active **run 3** (>2) plus persistent correction after
support ends; diagnostic **run 7** (>5); historical regression **7/17 rejected, 10 executed**.

**The proximate cause is a definitional difference worth remembering.** The handoff specifies
anchor agreement `mean(J(s0,s1), J(s0,s2))`; the identifiability audit that produced the
validated 17/17 used the **three-pair** mean including `J(s1,s2)`. On all 17 frames
`J(s0,s2) = 0.000` and `J(s1,s2) ≈ 0`, so the audit got `(0.5+0+0)/3 = 0.167` — exactly its
reported median — while the anchor form gives `(0.5+0)/2 = 0.250`. Dropping the seed1–seed2
term raises the failures' agreement ~1.5×, above the 0.225 threshold, and 10 of 17 execute.
Rejecting all 17 needs >0.375, but only **14 of 266** agreement values survive the
quiet-frame acceptance floors at all.

Also worth noting: the retired activity threshold would have rejected **0 of 17** — these
frames always passed the activity gate, so disagreement was always the only thing that could
stop them.

Three constructions have now failed on the same observation contract (three full-data seeds,
a trajectory bootstrap, this joint gate). **Do not add another same-input ensemble.** But
before concluding, re-run this exact calibration with the three-pair metric — it is one
argument, and it is the only validated form. Every failure here was clustering or persistence,
never coverage.

0 of 20 live rollouts, `development4` not executed, `confirmatory41` untouched. Frozen
inference is bit-identical across all three seeds.

## Hybrid obstacle safety residual: three-pair joint gate (metric question closed)

`docs/HYBRID_OBSTACLE_THREE_PAIR_JOINT_GATE_FINAL_DECISION.md` —
`THREE_PAIR_JOINT_GATE_OFFLINE_TRANSFER_FAILED`, Case D.

**The previous task's hypothesis is falsified, and the metric question is now closed.** It
predicted that restoring `J(seed1,seed2)` would reverse the historical decision. Measured:
executions went **10 → 9** of 17. One frame.

The reason is arithmetic. `J(s0,s2)` and `J(s1,s2)` are **both exactly 0.000 on all 17**
frames, so restoring J12 multiplies every one of them by the same 2/3 factor — a monotone
rescaling that cannot reorder them — and the calibration then moves the threshold down by a
comparable factor (0.225 → 0.1667, ratio 0.74 vs the 2/3 rescale):

| | Two-anchor | Three-pair |
|---|---:|---:|
| Historical median / max agreement | 0.250 / 0.375 | 0.1667 / 0.2500 |
| Calibrated threshold | 0.225 | 0.166667 |
| Historical frames executing | 10/17 | **9/17** |

Golden reproduction of the identifiability audit passed exactly (three-pair median 0.1667,
anchor 0.2500). The handoff's `>= 0.5` mask vs the audit's `> 0.5` was settled empirically —
masks are **bitwise identical**, so the distinction is empty here.

**Everything non-temporal is healthy**: calibration feasible with the 0.80 floor untouched,
bootstrap upper bound 0.00000, calibration recall 1.000, nested recall 0.997, hard-active
retention 0.979, acceptance 0.977/0.999/0.980. The only nested and diagnostic failures are
temporal (run 5 with post-support persistence; run 7). But the historical regression fails on
**per-frame** grounds — 9 of 17 still execute — so this is Case D, not the temporal token.
Awarding `THREE_PAIR_TEMPORAL_CLUSTERING_REMAINS` would authorize a temporal study while nine
known false positives still execute.

**Stop pursuing agreement-metric variants.** Both definitions are now calibrated under an
identical contract and differ by a monotone rescaling on exactly the frames that matter. With
J02 and J12 identically zero, J01 is the only discriminative term, and it spans 0.385–0.750 on
frames that must all be rejected. A temporal study is now well motivated — it is the only
offline failure left once the historical frames are set aside — but it should test whether
suppressing the onset burst also removes the per-frame executions, and must not be allowed to
relax the per-frame contract.

0 of 20 live rollouts, `development4` not executed, `confirmatory41` untouched. Frozen
inference bit-identical across all three seeds including J12.

## Hybrid obstacle safety residual: three-pair joint gate, live (owner override)

`docs/HYBRID_OBSTACLE_THREE_PAIR_LIVE_FINAL_DECISION.md` —
`THREE_PAIR_LIVE_DEVELOPMENT_PASSED`. 20/20 rollouts on `development4` under
`ACT_PLUS_THREE_PAIR_JOINT_GATE`, `confirmatory41` untouched, nothing trained or
recalibrated. Provenance 67/67.

**The offline blocker was an artifact of the offline framing.** Five tasks stopped on 17
onset false positives that projected to multi-frame bursts. Live, across ~3,900 control
frames: **1** false-positive frame, burst length 1, peak arm deviation **0.000144 rad**
against the 0.35 cap, non-persistent, zero hazard-bar contact anywhere. False-positive
onset bursts do not cause closed-loop harm — they barely occur. All 10 development
criteria pass.

Three things temper the pass:

1. **The uncertainty veto never fired.** 471 executed frames, 471 `ACTIVITY_ONLY_SHADOW`
   frames, 0 vetoes. Live, this was a bare seed-0 activity gate; the agreement term from
   tasks 15–17 bought zero interventions.
2. **The empty-mask convention is backwards on the frames that matter.** The one false
   positive scored J01 = 0, J02 = 0, **J12 = 1.0** → three-pair 0.3333, passing the 0.1667
   threshold, because seeds 1 and 2 were both *empty* and empty-vs-empty is defined as
   perfect agreement. Seed 0 firing alone against two silent peers is the strongest
   disagreement evidence available and the metric scores it near-maximal. Fix this
   convention before spending more calibration effort on the veto.
3. **The 7 task failures are not the controller's.** All five 118 rollouts executed 0
   frames with `executed_action` **bitwise identical** to nominal; 108 failed r0 and r2,
   and r2 also intervened zero times. 118's other-environment contact is `[0,0,0,0,0]`,
   matching ACT-only exactly — the earlier deployable reference scored `[13,12,35,20,21]`
   and failed that gate.

Contact classes resolve at episode granularity only; per-frame contact classes are not in
the rollout schema, so a contact cannot be timestamped to a burst frame. Immaterial at one
FP frame and zero hazard contacts; it would matter at a real burst rate.

## Pick-and-place corridor v5: re-recording the kept rows as demonstrations (executed)

The v5 "collection" ran the feasibility-screen harness, not the datagen pipeline.
`scripts/run_pact_place_collection.py:33` imports `run_row` from `run_pact_place_expert_screen`,
which truncates the sampled sensor suite to `qpos`/`tcp_pose` before the rollout. The 152 kept
episodes are a valid **screen record** with no proximity, no RGB and no action arrays: the whole
tree holds 0 non-JSON files. Nothing downstream could run off them.

The kept set was already decided, so this re-records it rather than re-selecting it. Seeds, expert,
scene, success criterion and clean-success filter are unchanged; only the observation suite differs.

Contract `configs/pact_place_v5_recovery.json`
(`config_sha256 125db9ac9f1eafcbbf9ce5a741d2c684a410d46da662ce55a99de3965bf7b9ac`) freezes the 152
kept rows, each carrying the outcome its screen row recorded, so the re-run can be checked row by
row instead of trusted.

```
export MUJOCO_GL=egl PYTHONUNBUFFERED=1
export MLSPACES_ASSETS_DIR=/root/prox_learning_pact_remediation/assets
PY=/root/act_retrain_venv/bin/python

$PY scripts/pact_place_recovery_contract.py
$PY scripts/run_pact_place_recovery_datagen.py --config configs/pact_place_v5_recovery.json --workers 12
$PY scripts/verify_pact_place_recovery_keys.py --config configs/pact_place_v5_recovery.json
```

Results, 2026-08-19, 12 workers, 2h26m, 5.6 GB at ~37 MB per episode:

| Measure | Result |
|---|---|
| Rows complete / `clean_success` | 152 / 152 |
| Outcome, seed and step count reproduced | 152 / 152, zero divergences |
| Step-3 file gate | 152 / 152 pass all 14 checks |
| Total timesteps | 72,955 (T: min 243, median 480, max 634) |
| `recovery_sha256` | `a704bccf9bec8ae6d0ac377e64bd22e3e0688e7d4eaaea91e7ee19553651c2e8` |
| `keys_verified_sha256` | `b06ff8fcb3751a0eefc6adbafc569922df69f9d2c1d38d1036ff4e1dc59ea639` |

Output `assets/datagen/pact_place_corridor_v2/recovered_152/` now holds 152 `trajectory.h5` and 152
wrist MP4s. Exact reproduction across all 152 means the full sensor suite is physics-neutral here,
which was the open stop-condition.

Three things worth carrying forward, all in `docs/PACT_PLACE_V5_DEMO_RECOVERY.md`:

1. The screen's `_make_config` carries a **second** observation reduction beyond the sensor-suite
   truncation: `proximity_sensor_period_ms = 0.0`, which writes proximity as `(T, 1, 8, 8)`. The
   trainable schema needs `(T, 4, 8, 8)` and the converter rejects anything else. Copying that
   function would have wasted the whole run.
2. `save_utils.py:374` calls a bare `json.dumps` on `obs_scene`; the place sampler's
   `scene_params["cam_visible"]` is a `numpy.bool_`, which NumPy 2 names `bool` and json refuses.
   Coerced in the recovery publisher so the reference collection's producer stays untouched.
3. This container's `/sys/fs/cgroup/pids.max` is 3840 and each datagen worker otherwise costs ~319
   tasks, so 12 workers die at import with only `libgomp: Thread creation failed`. The runner now
   pins the thread-pool env vars (319 tasks -> 3, no throughput cost) and preflights cgroup headroom.

Authorized next step is the ACT conversion only. Training remains unauthorized: both the runner and
the verifier hard-code `training_authorized: false`.

**For whoever writes the v6 collection:** the v6 plan says "collect ~255 clean under v6" with no
mention of the sensor suite, and as written reproduces this failure exactly. Its collection step
must require the datagen pipeline with cameras and proximity enabled, plus a file-level key
verification, before any training is authorized. Config-level review is what failed here -- the v5
collection config was internally consistent, self-hashed, and wrong.

## Place corridor v8c: the overhead bar cannot be sited (stopped at C0)

Full report: `docs/PACT_PLACE_V8C_C0_SITING.md`.

v8b failed admission because the carried cup was closer to the clutter than any arm link in 5 of 6
episodes. v8c's answer was to exploit the band `z in [1.05, 1.40]`, which the v7 swept volume shows
is traversed by link5 and link6 but never by the cup, and hang a hazard bar there — link-primary by
construction. C0 was to site that bar by measurement over the 24 frozen v6c trajectories before
anything was built.

```
export MUJOCO_GL=egl PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1
export MLSPACES_ASSETS_DIR=/root/prox_learning/assets
PY=/root/act_retrain_venv/bin/python3

$PY scripts/run_pact_place_v8c_c0_siting.py --workers 8
$PY scripts/run_pact_place_v8c_c0_visibility_ceiling.py --workers 8
$PY scripts/run_pact_place_v8c_c0_duck_feasibility.py --workers 8
```

667 candidate bars, 24 episodes, 11,352 control steps, replay-only, exact `mj_geomDistance`.
Artifacts in `diagnostics_output/pact_place_corridor_v8c_c0/`, all `authorizes_gate: false`.

**The premise held.** `cup_is_closest_body` is **0 of 24 in all 567** hazard-class candidates, with
a minimum cup clearance of 0.134 m. The wall that stopped v6c, v7 and v8b is genuinely cleared.

**Three walls behind it, all measured, none tunable.**

| | measured |
|---|---|
| Wrist visibility of any bar in the band | **0 of 24, all 667 candidates**; the bar's centre never enters the FOV cone |
| Wrist visibility ceiling inside the enclosure | **z = 1.030 m**, against a cup swept ceiling of 1.011 m and a band bottom of 1.050 m |
| link5/link6 height above the TCP in the corridor | median **0.426 m** |
| TCP drop needed to duck a bar at z = 1.05 | median **0.277 m**; puts the TCP below the shelf top in **24 of 24** episodes |
| Admissible candidates that merely graze the arm | **0 of 392** — penetration is 26 mm at best, 99 mm median |

The first two rows are the same fact twice: the cup and the wrist camera ride the same rigid body, so
the band that is cup-free is camera-blind, to within 19 mm. `cup_is_closest_body <= 0.25` and
`non-zero visibility_at_min >= 1/3` are mutually exclusive in this corridor, and the v8c plan
retained both. The third and fourth rows close the expert: both obstacle maneuvers act on the TCP,
and the colliding body is 43 cm above it, so clearing the bar means a nullspace elbow re-pose — a
general planner, which the plan forbids.

Three clips with the chosen bar composited into the replay are in
`diagnostics_output/pact_place_corridor_v8c_c0_review/` (`run_pact_place_v8c_c0_review_videos.py`,
rows 0/4/17, both panel sides): wrist, third-person, and a corridor view with the bar tinted.
Across 1,603 frames the bar is in the wrist FOV **0** times and inside link5/link6 **763** times.
MuJoCo's contact detection flags the bar in exactly the frames `mj_geomDistance` calls negative.

Broken down by phase, both legs are obstructed — and `grasp`, `gripper-close` and `lift` are struck
in **100%** of their frames (45/45, 27/27, 57/57), as is `outbound_approach` (217/217). Those grasp
phases are not traversal segments: the TCP is pinned to the cup on the shelf, so there is no
waypoint to bow and no `z_travel` to shift. The duck does not merely overshoot the shelf there; it
has no free parameter to act on.

Two things to carry forward regardless of what happens to v8c:

1. **Only `pact_intrusion_*` scores as `hazard_bar`.** `pact_contact_audit.py:16` sets
   `HAZARD_BODY_PREFIX = "pact_intrusion_"`; the legacy `protr_s/m/l` bars fall through to
   `other_environment`. Both break `clean_success`, but the attribution differs, and the v8c plan
   assumed either would do. 100 of the 667 candidates were rejected on this alone.
2. **`mj_geomDistance` has a second false-zero mode that v8b's fallback does not cover** — scalar
   0.0 with `fromto` left untouched, at every `distmax`, for geoms 25 cm apart. It fabricates
   contacts. The C0 instrument clears the buffer per call and uses an AABB gap only to *disprove*
   such a zero, never as a clearance; 52 were rejected in this sweep.
   `scripts/measure_pact_place_v8b_realized.py:41` still carries the unhardened version.

## Place corridor v9: clutter the skin can resolve (blocked at raw admission)

Three plan documents, in execution order:

| Document | Scope |
|---|---|
| [`docs/PACT_PLACE_V9_ENVIRONMENT_PLAN.md`](docs/PACT_PLACE_V9_ENVIRONMENT_PLAN.md) | V0-V2: instrument, palette, siting, expert wiring, human review, Phase-0 gate |
| [`docs/PACT_PLACE_V9_RAW_ADMISSION_FIX_PLAN.md`](docs/PACT_PLACE_V9_RAW_ADMISSION_FIX_PLAN.md) | **Current work.** Replaces V0c siting after the raw admission failure |
| [`docs/PACT_PLACE_V9_TRAIN_EVAL_PLAN.md`](docs/PACT_PLACE_V9_TRAIN_EVAL_PLAN.md) | V3-V7: collection, conversion, training, 600-rollout evaluation, analysis |

Status: [`docs/PACT_PLACE_V96_CLUSTER_SITING_STATUS.md`](docs/PACT_PLACE_V96_CLUSTER_SITING_STATUS.md)
(current), after [`docs/PACT_PLACE_V95_LOW_WALL_STATUS.md`](docs/PACT_PLACE_V95_LOW_WALL_STATUS.md).
V9.5 stopped at its raw-first gate with **0 of 6 physics-clean variants passing**; the one variant
originally counted as a pass is the one whose source episode is not collision-free
(`diagnostics_output/pact_place_v95_v0c5_raw_prerequisite/admission_correction.json`). The clustered-hazard remedy was then measured and **does not fit the
corridor** — see below. Nothing in v9 authorizes a gate or a collection.

### Why: the skin cannot resolve household objects

Each skin sensor is 8x8 depth over a 45 deg cone (`camera_configs.py:428-435`), so the pixel pitch
at range R is `2*tan(22.5deg)/8 * R = 0.1036*R`. An object registers only when its width across the
view axis exceeds that pitch:

| object | width | 1 px at | 2 px at | 4 px at |
|---|---:|---:|---:|---:|
| intrusion panel | 0.480 | 4.64 m | 2.32 m | 1.16 m |
| soapbottle | 0.089 | 0.86 m | 0.43 m | 0.22 m |
| candle | 0.016 | 0.15 m | 0.08 m | 0.04 m |

The measured present-versus-parked causal effects on the real `[474, 40, 4, 8, 8]` tensor match that
model (`diagnostics_output/pact_place_v95_v0c5_raw_prerequisite/validation.json`, variant 6 —
**the dirty-source episode**, see the correction below):

| hazard | sensors changed | changed values | max delta |
|---|---:|---:|---:|
| intrusion panel | 11 | up to 23,004 | 2.00 m |
| outbound bottle | 2 | 448 + 60 | 0.77 m |
| inbound bottle | 1 | 40 | 0.26 m |

**Correction (E0).** Joining those variants to the smoke summary's `clean_success` shows that the
single "pass" is the one variant whose source episode is not collision-free (F3 left, 351 clutter
contacts; its right pair, 2,315). Every physics-clean variant failed, so the V9.5 headline is
**0 of 6**, not 1 of 8, and the inbound vessel's only nonzero reading in all of V9.5 comes from that
dirty episode at R = 0.11 m — a sensor nearly touching an object, not a detection at range. The
validator now fails admission on a dirty source; no V9.5 artifact was edited.

The tensor holds 4.85M values, so the inbound vessel changes **8 parts per million**. That is below
the sensor's resolving power, not a weak signal awaiting tuning, and it is why V9.4's lower fixtures
and V9.5's wider vessel could not fix it.

### Three findings worth carrying forward

1. **Sensing happens at range, from links that never enter the enclosure.** The sensors that see the
   panel are `link2_sensor_3/4/5`, `link3_sensor_1/2/4`, `link5_back_sensor_0-4`. The v7 swept-volume
   fact that links 1-4 have zero voxels inside the enclosure bounds **collision**, not sensing. Every
   siting sweep so far scored TCP or collision clearance; none scored angular subtense at the sensors.

2. **The admission bar was "any nonzero pixel."**
   `scripts/run_pact_place_v9_v0c3_causal_proximity.py:414-417` sets
   `passed = panel.changed_values > 0 and inbound.changed_values > 0 and outbound.changed_values > 0`.
   The per-value threshold is noise-floor derived and sound; the aggregate rule is not, which is how
   a 40-value single-sensor signal counted as a pass.

3. **The frozen encoder is per-sensor and shared.** `SurfaceEmbeddingEncoder.forward` takes
   `(B, CAUSAL_FRAMES, 8, 8)` with sensors carried in the batch dimension, so it has no knowledge of
   how many sensors exist. Adding skin coverage would not invalidate encoder
   `6fd2dd037e3236b5b6bf7fce8cb2709ead0cf52adcbbe9cbad1061efc2fe3206`; it would change
   `n_proximity_sensors` in the policy, which is retrained regardless. v9 keeps the suite frozen at
   40 for comparability, so the hazard adapts to the sensor rather than the reverse.

### W1: the resolving-power instrument retrodicts the measurement to r = 0.99997

[`scripts/pact_skin_resolvability.py`](scripts/pact_skin_resolvability.py) scores, per frame and per
sensor, the range, the hazard extent across the view axis, `subtense_px = W_perp / (0.1036 * R)`, and
an occlusion-aware `mj_ray` pixel count under the proximity renderer's own geom-group filter.
[`scripts/run_pact_place_v9_w1_resolvability.py`](scripts/run_pact_place_v9_w1_resolvability.py)
replays the eight frozen V9.5 trajectories with `mj_forward` only and compares it to the measured raw
counterfactual. Artifact: `diagnostics_output/pact_place_v9_w1_resolvability/resolvability.json`.

Over 24 role measurements: ordering reproduced in 8 of 8 variants, Pearson r on `changed_values`
0.99997, predicted/measured ratio 0.994-1.042, zero measured-nonzero-predicted-zero, and 1.00 recall
of the measured responding sensors. Both vessels are predicted value for value (40, 448, 60). The
prefilter's **occlusion-free** variant over-states a compact hazard by 31.8x (inbound vessel), 6.2x
(outbound) and 1.33x (panel) — so geometry-only siting scores are upper bounds, never admission.

### W2: making the hazard large works; there is nowhere to put it

[`scripts/run_pact_place_v9_w2_cluster_siting.py`](scripts/run_pact_place_v9_w2_cluster_siting.py)
scored 14,280 cluster placements — three tall vessels shoulder to shoulder, spanning 0.29-0.39 m —
against all eight frozen trajectories, recording every candidate with its reason. The best outbound
cluster (`x = 0.740`, `y = -0.280`, 90 deg, span 0.296 m) clears 2 px on 4 sensors in the worst
variant with a 2.6x side imbalance, against V9.5's single bottle at 1 sensor. The inbound leg admits
nothing, and no inbound/outbound pair is jointly feasible at all:

| | |
|---|---:|
| Usable aperture width | 0.810 m |
| Free band left by the active panel (inner face at y = 0.095) | 0.460 m |
| Loaded transport envelope + clearance | 0.340 m |
| **Max hazard width still leaving a lane, at the panel's depth** | **0.120 m** |
| **Contiguous silhouette the skin needs** | **0.250 m** |
| Inbound x outbound pairs examined / jointly feasible | 18,393 / **0** |

The reason is structural, not a siting failure. The paired-side design requires identical clutter
under both panel sides; the panel forces the arm's lane to a side that flips between the two rows of
a pair; a hazard must be lateral (`|y| >= 0.10`) to be sensed at all, because the gripper, hand and
link7 carry no sensors; and a lateral hazard does not flip, so it stands in the lane in exactly half
the rows. A hazard centred on `y = 0` flips with nothing, but the same lane test admits it at 0.12 m of
width and closes at 0.15 m — the same 0.12 m budget from the other direction.

Two measured ceilings bound the escape routes. Deeper than the panel, an inbound hazard needs to be
**0.60 m** wide before three sensors clear 2 px with a balanced response — wider than the intrusion
panel itself (0.480 m) and wider than the 0.460 m band it would have to fit inside. Under the panel,
an object clearing its `z = 0.80` underside on both sides can be at most **0.070 m** tall, which the
arm crosses at `z >= 0.788`. The status doc lists the five choices this leaves.

### W3, run once as a pipeline validation, not as an admission

[`scripts/run_pact_place_v96_cluster_causal_proximity.py`](scripts/run_pact_place_v96_cluster_causal_proximity.py)
imports the V9.5 validator's rendering path unchanged and replaces only the aggregate pass rule, with
the floor written to `config.json` before any render. Because W2 admitted nothing, it was run once on
one family purely to exercise the path and calibrate the prefilter
(`diagnostics_output/pact_place_v96_w3_pipeline_validation/`, `passed: false`).

The panel's causal effect on the twelve-slot V9.6 scene is **58,416** left and **23,508** right —
identical value for value to the V9.5 measurement of the same panel on the same frozen trajectories,
which validates the replay across a change in `nq`. Clustering then does what W1 predicted, on the
side the geometry allows:

Both sides of this comparison are F3, whose source physics is dirty, so it must not be quoted as a
clean-source result; E1b re-runs it on F0/F1/F2.

| hazard, F3 left, V9.5 decision window | V9.5 single vessel | V9.6 cluster |
|---|---|---|
| inbound | 40 changed values, 1 sensor | **2,604 changed values, 3 sensors** |
| outbound | 508 changed values, 2 sensors | **644 changed values, 3 sensors** |

On the right the inbound cluster changes 16 values on one sensor — a 162.8x imbalance against the 4x
limit — so the paired-side asymmetry that stopped V9.5 survives at cluster scale. The occlusion-free
prefilter over-predicted the measured cluster response by 2.6x (left) and 19.8x (right), in line with
W1's calibration: **no geometry-only score is ever an admission.**

### What this narrows the claim to

Because the skin needs a contiguous silhouette of roughly 0.25 m, the clustered hazards v9 builds are
sensed as a slab while reading as household clutter to the wrist camera. The supportable claim is
**"PACT avoids large obstacles the camera cannot see, some of which are built from household items"**,
not "PACT avoids clutter". Mugs and apples at 0.075-0.09 m stay RGB-only decor by physics. The v9
reports must state the resolving-power floor wherever they describe the hazards.

### V9.8 ceiling pendant gate — stopped at the paired offset selection rule

The V9.8 pendant implementation is pre-registered in
[`docs/PACT_PLACE_V98_PENDANT_PLAN.md`](docs/PACT_PLACE_V98_PENDANT_PLAN.md). The
first S1 siting sweep selected a symmetric ceiling fixture at `x=0.72`, `y=0.0`,
bottom `z=1.10`, and half-width `y=0.18`: 14 worst-case sensors at 2 px and
181 route-intrusion frames across the six physics-clean F0/F1/F2 variants.
That width lands the fixture-bow waypoint on the aperture limit with zero
slack. S1 v2 required ≥ 20 mm detour slack and selected `half_y = 0.16`
(13 sensors, 181 intrusion frames). The causal frozen-qpos smoke preserved
the 40-sensor `[40,4,8,8]` renderer and had a zero repeat baseline, but it is
not admission.

The first 24-row expert screen (0/24, all clutter contact) is a **void
measurement of a mis-wired expert** and is retained, not reused. Pinning seed
955339 and enabling the TCP-only lateral bow at `half_y = 0.18` produced
fixture-free **6/8**, pendant+bow **0/8** (clutter during a 0.14–0.16 m
inbound detour), and a pinned-seed **0/24**. That 0/8 measured a zero-slack
detour. With the bow off, a same-run V9.5 replay is still **6/8 row for
row**, and the same eight rows plus pendant are **0/8** on `mounted_fixture`
contact (clutter 0, inbound bow 0.0 m). Sweeping `half_y` 0.16 / 0.14 / 0.12
with the bow on is **0/8** at every width: unclipped waypoints and
`mounted_fixture` contact on every complete row. Offsetting the pendant to
`y = 0.100` with side-dependent ceiling envelopes (lag + 4 mm) passed the
live `_bow_segment` preflight (algebra, dispatch, and no clipping — not
swept-arm clearance) and left the V9.5 guard at **6/8**, but
neither named candidate preserved the fixture-free clean rows (wide and
conservative both 0/8 on the paired join). The 24-row gate was not run.
The aperture is not widened. Collection and training are not authorized.

A 2026-08-25 retained-qpos audit
(`diagnostics_output/pact_place_v98_offset_contact_diagnosis/`) supersedes
the earlier causal reading that "the wrist does not follow the TCP bow."
Baseline-clean failures start outside the protected ceiling-fixture
approach/pass/exit set: left in post-bow `pregrasp` (`fr3_link5_collision`),
right in vessel/cross-vessel pass (`fr3_link6_collision` where reconstructable).
The 0.208 / 0.108 lag constants are `unverified_provenance`; the
`[0.044, 0.156]` face window is `physical_input_invalid`. There is no
established claim that a wrist-height hazard is unavoidable in the 0.85 m
aperture.

Ground clutter remains RGB-only decor that counts as failure on contact, with
no claim that the skin sees it. The skin's roughly 0.25 m contiguous-silhouette
resolving floor is why the first S1-selected pendant is 0.36 m wide
(`half_y = 0.18 m`); that width is zero-slack, and the narrower widths that
have slack still fail on the pendant.

A fixture-free 24-seed sweep of the V9.5 8-row smoke averaged 4.08/8 clean
(51%). Only 4 of 24 seeds reach 7/8; 955339 is one of those four (sweep 7/8
vs smoke/guard 6/8, unreconciled). A canonical varied-seed 24-row screen
would expect ~12/24 against a bar of 20 (`narrow: true`).

### V9.9 fixed pendant qualification — permanently closed

[`docs/PACT_PLACE_V99_PENDANT_PLAN.md`](docs/PACT_PLACE_V99_PENDANT_PLAN.md)
asked whether one fixed, identical ceiling-flush pendant can force ≥ 5 cm
lateral detours on both inbound and outbound traversal while preserving the
six clean F0/F1/F2 V9.5 cells. Independent reconstruction of the frozen eight
rows succeeded (max TCP residual 0.87 mm). The conservative AABB filter
certified 0 candidates; that is only a broad-phase screen. The final float64
exact retained-qpos close-out scored all 12880 dual-transit AABB hits with
all four predicates on all six cells and found 0 survivors. Scoped
conclusion: **no survivor in the registered fixed rectangular-box lattice.**
Routing, paired screens, and collection were not run. V9.8 artifacts and
conclusions are unchanged. Collection, training, and eval are not authorized.

### V10 connected compound pendant qualification

[`docs/PACT_PLACE_V10_COMPOUND_PENDANT_PLAN.md`](docs/PACT_PLACE_V10_COMPOUND_PENDANT_PLAN.md)
replaces V9.9’s single rectangular prism with one static, side-independent
assembly of two paddles, two ceiling stems, and a crossbar. It preserves the
frozen six-cell V9.5 layout and V9.9 safety, necessity, routing, and sensing
gates. V9.9 remains permanently closed and is not reopened. The v1 siting
catalog of 8,554,036 rows
(`diagnostics_output/pact_place_v10_siting/siting.json`, payload SHA
`923c9380319b343e43e55f018080995db8a4b59e5a5f3cbd7f5d1a3be79d0eb6`;
catalog SHA
`63369af3552bbb806a61fea97d281011374ee25bb375004876704b920b6f3443`)
is a superseded robot/target prefilter; those rows did not pass posed-panel
or clutter checks. Corrected siting v2
(`diagnostics_output/pact_place_v10_siting_v2/siting.json`, payload SHA
`2e0b2a56bd4c22ecc920927dc149adf9c1bbc0d1d3ccbd3ee433ea450b187c1c`;
catalog SHA
`b84e19bf269c39cd052551639c22d4cbb5b4348eaf6188663fae0659af824d6e`)
has 150,288 panel-clear and full-environment exact two-lobe survivors
(1,779 unique union AABBs). Planning-probe v2 is the trust anchor.
Three-lobe search was not run. Offline routing
(`diagnostics_output/pact_place_v10_route/route.json`, payload SHA
`c0f1b35084d6950a88531c45e6805b06437add31c82ca5fd68bb5da4f5de3ff7`) found
zero route-feasible unions and zero route-feasible morphologies. All 21,348
union×cell×direction evaluations failed the registered ≥5 cm detour /
open-side rewrite. `stop_reason: no_two_lobe_route_survivor`. That
route-v1 result remains valid under contiguous-group-freeze and is not an
error. A separately registered endpoint-only amendment
(`diagnostics_output/pact_place_v10_route_v2/route.json`, payload SHA
`e311ba01c77c14b3a930be8dd9d4d40e9de483710521f8662d2e3a55357f71e1`)
reproduced 17,826 geometry-feasible evaluations and 1,032 unions, then
found zero route-feasible unions and zero route-feasible morphologies:
666,448 inbound identities passed sequential IK and failed strict
environment. `stop_reason: no_route_v2_ik_clearance_survivor`. That
route-v2 stop is a historical result of the flawed scalar environment
predicate and is **not** physical infeasibility. `v10_closed: true` for
the registered offline-search question. Signal screening, paired screens,
collection, training, and ACT/PACT evaluation were not run under V10 and
remain unauthorized.

### V10.1 empirical pendant qualification — stopped before Phase 0

[`docs/PACT_PLACE_V101_EMPIRICAL_QUALIFICATION_PLAN.md`](docs/PACT_PLACE_V101_EMPIRICAL_QUALIFICATION_PLAN.md)
replaces further exhaustive route search with a 12-row empirical
qualification of the frozen V10 `planning_probe_assembly()` / probe_v2
two-lobe pendant on V9.3 families F0–F2 only. The frozen route is
`rewrite_primitive=endpoint_only`,
`qualification_mode=empirical_live_contact_v1`, slab padding 0.08 m, left
lane −0.30 m, right lane +0.30 m. Empirical mode does not call the flawed
scalar strict-environment preclearance; live MuJoCo contact auditing is
the environment predicate. Historical V10 rows without the new markers
keep contiguous-group-freeze behavior. All V9.9/V10 siting and route
artifacts are preserved.

Contract `pact_place_v101_empirical_qualification_v1` SHA
`c550badd4a95bb0f46c84744ca9cb52b6fd5aa4290377edb8aacc2458087873b`.
Preflight
(`diagnostics_output/pact_place_v101_empirical_review/preflight.json`,
artifact SHA
`37f1d3739376a0fe8b84c7822ac5aeb65ac1fff61a2b84f91540bdf3ac89f7f3`)
verified the siting-v2 trust anchor and admitted the fixed route **12/12**
on V9.9 snapshot stock TCP. The 12-row review
(`diagnostics_output/pact_place_v101_empirical_review/review_manifest.json`,
artifact SHA
`3a472dc165b0053766478dbc3f9e64af54b66e324ec6abd522bde9840675c841`)
reconciled 12/12 rows with zero infrastructure failures and **0/12 clean
successes** (0/2 in every family×side cell). `eligible_for_human_review:
false`. `authorizes_gate: false`. Gallery:
`diagnostics_output/pact_place_v101_empirical_review/videos/` (12/12
three-pane videos). On the 11 complete rows, route telemetry is healthy
(no fallback/clip/wrong-way/endpoint mutation; offline strict-env not
used; zero pendant contact). Live causes: 8 `terminal_ik_cascade`, 3
`clutter_collision_stability_event`, 1 F2-left `sampling_failure` with
missing telemetry. Causal proximity
(`diagnostics_output/pact_place_v101_empirical_causal/causal.json`,
artifact SHA
`30329d737be32663b93802c88ba6ced22a121e06b727ca2baa947747018b9364`)
fail-closed with `missing_clean_cell` in all six cells and did not call
`env.step`. Failure table SHA
`74688c41674a15be909c2889b921e7809daa2fcc5a3cc75805d29e8e78fc1057`.
`human_approval.json` is absent. Phase 0 was not run. Collection,
training, and evaluation remain unauthorized.

### V10.2 raised, collision-legible pendant — stopped at Step 0

[`docs/PACT_PLACE_V102_RAISED_PENDANT_REMEDIATION_PLAN.md`](docs/PACT_PLACE_V102_RAISED_PENDANT_REMEDIATION_PLAN.md)
answers three human-review objections to the V10.1 gallery: the arm appeared to
pass through a pendant stem, the lobes sat too low against table clutter, and
the empty-arm approach looked too fast. V10.2 raises the two-lobe assembly so
its lowest point is `1.10 m` (a `0.38 m` gap above the `0.72 m` shelf top),
thickens both stems and the crossbar x face to a 12 mm square in collision *and*
visible geometry with no visual-only sleeve, keeps the frozen endpoint-only
route (`empirical_live_contact_v2`, padding 0.08 m, lanes ∓0.30 m), assigns
speed per named route piece (0.15 m/s empty-arm approach, 0.045 m/s pendant
pass/exit, inherited 0.08 m/s pregrasp, unchanged outbound transport) instead of
copying `segments[0].speed`, and renders review video at 1000/66 fps with
`frame_stride = 1` — real time, against V10.1's 1.32×. It uses a distinct
environment marker `pact_place_corridor_v10_2_raised_pendant` on the same
compiled V10 scene; V10/V10.1 dispatch and geometry are behaviorally unchanged
and every V10.1 artifact is preserved byte-for-byte.

Contract `pact_place_v102_raised_pendant_v1` SHA
`16f4c263d3b0310788b27e51303f0aa3feed0241e2c09ba254a69de25eb29a8b`;
implementation SHA
`c061bc50c4bd9a13c40250fdd081f0c0286a84e7e1ac619a0ba306b7d2f708e6`;
assembly self-SHA
`0751a8d4850994e59f0486bd46f411018608ccb37105a8c0c03e03cbecccdb27`;
speed-schedule SHA
`b9c17c5022780d8820bfff57db17f2e6715aa0d10bc2a025c25c21ce0a1e7d32`.

**Step 0 failed and V10.2 stopped there.** Preflight
(`diagnostics_output/pact_place_v102_preflight/preflight.json`, artifact SHA
`6c5079916775e8a2093defb1547a3fa85ef9b32dcc4fddcf785ffa6c3276976d`) passed
items 1–4 and failed items 5–7:

- protected artifacts and the V10 scene hash matched (17 artifacts);
- the raised assembly is panel-, clutter-, static- and initial-state-clear on
  all six frozen cells, and its derived facts match the registered table;
- exact stock-route necessity is **12/12** — the raised pendant still obstructs
  every cell and direction, so it is not decorative;
- the fixed endpoint-only route is admitted **12/12** with inbound detours
  0.103–0.120 m and outbound 0.059–0.071 m;
- complete sequential IK on the nominal route: only **5/12** cases, all of
  them inbound, with every outbound case incomplete (erratum: an earlier
  version of this paragraph said 4/12; the immutable preflight payload says
  5/12 and is unchanged);
- per-component clearance: **0/12** cases meet the 15 mm floor. Every case is
  negative. The same-side lobe is penetrated by **56–77 mm** and its stem by
  26–52 mm, inbound and outbound, on all three families. The crossbar and the
  opposite-side components are clear throughout.

Raising the lobe bottoms from `0.82/0.84 m` to `1.10 m` moved the pendant out of
the table-clutter band and into the arm's own elbow/forearm envelope. The
registered route is a **TCP** lane rewrite with the grasp-side endpoint frozen;
it has no authority over where link5/link6 sit while the TCP tracks that lane.
The V10 lattice cap `LOBE_TOP_MAX_M = 1.10` is exactly the boundary this design
crosses. The human observation that the arm passes through the pendant is
therefore confirmed as literal geometry for the raised design.

Step 0 item 7 also failed, and its root cause is recorded separately in
`diagnostics_output/pact_place_v102_preflight/contact_parity_root_cause.json`
(artifact SHA `e4e544a999534b322d177d8b296aa4c7580d9b7627bf89bf0d42630fdd0774df`,
produced by `scripts/diagnose_pact_place_v102_contact_parity.py`): with a stem
deliberately posed to penetrate `robot_0/fr3_link5_collision` by 17–69 mm across
19 poses, `data.contact` reported **nothing**, while the same scene carried 83
other contacts. The pair is collision-compatible, so the contype filter is not
the cause. `pact_place_v10_scene.pose_assembly_geoms` writes `model.geom_pos`
and `model.geom_size` at runtime but leaves `model.geom_aabb`, `geom_rbound` and
the pendant body's `model.bvh_aabb` at their compile-time 1 mm placeholder
values, so MuJoCo's broadphase never proposes a pendant/robot pair. Refreshing
only those bounds, at the same pose, makes the contact appear at
`dist = −0.016630 m`, matching hardened GJK to six decimals. **Recorded "zero
`mounted_fixture` contact" for the runtime-posed V10 pendant — including the
V10.1 review rows — is therefore not evidence of clearance.** No V10/V10.1
artifact was altered and `pose_assembly_geoms` was not repaired: that would
change V10/V10.1 runtime behaviour, which the V10.2 plan forbids. It is flagged
for the owner.

Step 1 is complete: `tests/test_pact_place_v102_raised_pendant.py` is 41 passing
behavioral tests covering marker/hash-gated dispatch, unchanged V10/V10.1
behaviour, identical 12 mm collision and visible stem geometry, parked-control
disabling, deliberate stem overlap seen by both `data.contact` and the contact
classifier, full-waypoint IK accounting with qpos restoration on
success/failure/exception paths, route-piece speeds and caps, row admission, and
renderer timing (151 policy frames → 9.966 s). No V10.2 episode was generated:
the six-row screen, the 12-row review and gallery, the causal replay, and the
24-row Phase 0 were **not run**. `human_approval.json` is absent and was not
inferred. Collection, training, and evaluation remain unauthorized.

**Owner-requested diagnostic gallery (not the gate).** After the Step-0 stop the
owner asked to see the twelve clips anyway. They were produced by
`scripts/run_pact_place_v102_diagnostic_gallery.py` into
`diagnostics_output/pact_place_v102_diagnostic_gallery/` (manifest SHA
`9d0f7b4b6c8adc51261cf58bbed26a12ec68b5b62bfd1c446ac122c94918c0a1`,
`is_registered_review: false`, `eligible_for_human_review: false`, every
authorization false). This is **not** the registered review pack, is not an
input to causal proximity or Phase 0, and must never be cited as one; the
registered review path was not written and the Step-0 stop is untouched. The
gallery runner is not part of the V10.2 implementation hash, which is unchanged.

The twelve rows ran the real expert with the live contact audit. All 12
reconciled and completed; **0/12** are clean; 11 end in `terminal_ik_cascade`
and one in `clutter_collision_contact`. Every row's minimum per-frame
robot/target-to-pendant clearance is **negative** — −0.0008 m to −0.0742 m —
against the 15 mm floor, and across all twelve rows there are **zero** live
pendant-contact frames and **zero** `mounted_fixture` contact entries. That
pairing is the two Step-0 findings reproduced on live rollouts: the arm passes
through the raised pendant, and the contact pipeline cannot see it. The speed
schedule is visible and correct in the telemetry —
`inbound_pendant_approach` commanded at 0.15 m/s and `inbound_pendant_pass` at
0.045 m/s, against an inherited 0.20 m/s for both.

The clips render every policy frame at `1000/66` fps, so they play in real time.
The registered renderer's pendant pane is aimed from `y = -1.15 m`, outside the
hood, where `hood_side_r` occludes it; the gallery overrides that camera
in-process to a pose inside the aperture and records the override in its
manifest under `pendant_side_camera_override`. Any successor version should
adopt the corrected pose.

### V10.3 static-pendant joint-route qualification — stopped at Step 0B

[`docs/PACT_PLACE_V103_STATIC_PENDANT_IK_PHASE0_PLAN.md`](docs/PACT_PLACE_V103_STATIC_PENDANT_IK_PHASE0_PLAN.md)
answers V10.2's two findings at once. The pendant becomes **static**: the
selected lobe/stem/crossbar poses and sizes are compiled into their own scene
before `MjModel` exists, the body carries no joint, freejoint, or mocap flag, and
nothing writes `model.geom_pos` or `model.geom_size` at episode runtime — so the
stale broad-phase bounds that made V10/V10.1/V10.2 pendant contact invisible
cannot recur. The route becomes a **continuous joint-space plan**: a layered
multi-branch IK graph over named control poses, with exact distance at every
0.01 rad interpolation sample, pinned to the retained qpos at both ends, and
executed as the selected joint trajectory rather than re-solved from TCP at
runtime. The only search variable is the lobe height.

**Step 0A passed.** `tests/test_pact_place_v103_static_pendant.py` is 34 passing
behavioral tests — a compiled static scene whose pendant cannot move under
repeated `mj_forward`, visible-equals-collision sizing, a clear/touching/
penetrating contact-parity fixture where hardened distance and `data.contact`
agree, the V10.2 errata read from the immutable payload, the 120-template
lattice, fixed Halton seeds, dedup, graph ranking, and duration caps. The
regression sweep before search was **302 passed**, with the same three
pre-existing stale-scene-hash failures in `tests/test_pact_place_corridor.py`
recorded rather than excluded.

**Step 0B stopped, under an owner-approved early stop.** Nine of twelve
cell/direction cases completed, **all with zero feasible routes at every one of
the four registered heights** (0.92 / 0.96 / 1.00 / 1.04 m): the six outbound
cases each evaluated all 120 registered templates and found none whose
control-pose layers were all non-empty, and the three left-inbound cases were
excluded at the pinned endpoint before any template work. The three right-inbound
cases were still running after 4 h 42 m and are recorded as `not_evaluated`. The
stop record
(`diagnostics_output/pact_place_v103_ik_search/search.json`, artifact SHA
`f06feaa3c09d5f95a006f66d00e45c8684962393967a1acc3ad40d21dc23df98`) is explicit
that `search_exhaustive: false` and that not every registered template was
evaluated.

The stop is nonetheless conclusive, and the witness is
`diagnostics_output/pact_place_v103_ik_search/endpoint_certificate.json`
(artifact SHA
`3ced3a35b71ac7a1cc9f94ab23549b0764dceef07508ab5302791e592c062fda`). Across all
three left-inbound cells × all four heights × two retained frames, **all 24
measurements are penetrations below the 0.020 m node floor**, with a
`-0.02000 m` margin in every one and both instruments agreeing — analytic exact
GJK returns 0.0 (intersecting) and hardened signed `mj_geomDistance` returns a
negative distance. The binding component is always the negative lobe. What hits
it changes with height: `robot_0/gripper/base` at 0.92 and 0.96 m
(-0.4 to -14.3 mm), `robot_0/fr3_link7_collision` at 1.00 m (-14.8 to -22.8 mm),
and `robot_0/fr3_link6_collision` at 1.04 m (-20.0 to -24.5 mm). A low lobe is
struck by the hand, a high lobe by the wrist, and no z window in the lattice
threads between them. Because that endpoint is pinned to the retained qpos and is
the first interpolation sample of every edge leaving it, no route template can
rescue those cells — so no height can route all twelve cases regardless of the
three unevaluated ones. **The V10.2 result was never specific to the 1.10 m
height**: moving the pendant 6–18 cm down reproduces it with a different arm link.

Step 0C, the six-row smoke, the twelve-row review, the causal replay, and the
24-row Phase 0 were **not run**; no episode was generated, `env.step` was never
called, and no V10.3 episode runtime was built. `human_approval.json` is absent
and was not sought. Collection, training, and evaluation remain unauthorized.

### V10.4 first-shot static pendant — Steps 0–2 passed, stopped at Step 3

[`docs/PACT_PLACE_V104_FIRST_SHOT_STATIC_PENDANT_PLAN.md`](docs/PACT_PLACE_V104_FIRST_SHOT_STATIC_PENDANT_PLAN.md)
abandons the V9.5/V10.1 route family and builds on the already-qualified **V6c**
environment and expert instead (23/24 Phase-0 with zero clutter contact). It adds
one compiled-static, symmetric two-lobe pendant *outboard* at `|y| = 0.34 m`,
lobes spanning `z = 0.98–1.04 m` — above the clutter, outside the arm's envelope,
rather than in it. That is the axis the V10.3 close-out identified as the one
that mattered. The only route change is a single registered speed cap on the
first free-space segment (`0.20 → 0.12 m/s`).

Contract `pact_place_v104_first_shot_static_pendant_v1` SHA
`455379b852c994c6e4645b5650e8c690ebbc542509b40700316c8888db977707`; production
scene `pact_place_corridor_v10_4.xml` SHA
`01d8adf34808a9f419cb3a9d07668ec1069d3a5acfa8cb01885c622ea09876f7`.

**Step 0 passed on all six items**
(`diagnostics_output/pact_place_v104_preflight/preflight.json`, SHA
`fe64e285332a3c530cab30599d2b862823a3c1e2db661a6961a0a1461f2c41d5`). Both
read-only trust anchors reproduce **exactly**: the six retained V9.5 cells give
**0.04052 m** against a 0.035 m floor and an audit value of 0.04052, and the 24
frozen V6c Phase-0 trajectories give **0.05523 m** against a 0.050 m floor and an
audit value of 0.05523 — 24/24 above floor. The eight rigid ±5 mm corners hold
(V9.5 worst 0.03337 m, V6c worst 0.04800 m). Contact parity agrees across
hardened signed `mj_geomDistance`, analytic GJK, live `data.contact`, and the
place contact audit on **30/30** fixture cases. Route preservation is exact on
**24/24** V6c manifests: identical poses, exactly one speed change at primitive 1
segment 0, V6c itself never amended, max predicted 583 steps against the 840-step
limit. The pendant body carries no joint, freejoint, or mocap, and its compiled
bounds enclose the real geometry — the V10.2 stale-broad-phase defect cannot
recur.

**Step 1 passed at 6/6 strict clean successes**
(`.../pact_place_v104_review_production/production_manifest.json`, SHA
`fdcf757b4bff512c71c6e3ac241c151742523c89ec7531b46132af715e92b3af`), 3 left and 3
right against a `≥5/6` bar, **zero pendant contact in every row**, minimum
observed clearance **61.6 mm** — about three times the 20 mm floor — and 426–628
control steps against a 1050 horizon. Six rows are a qualification check, not a
clean-rate estimate.

**Step 2 passed on both sides**
(`.../pact_place_v104_causal/causal.json`, SHA
`a30c863d61537edb58d24cc91b13291fa5d9efc47c7521b08b2304943f2f2ffc`): 23,684
changed values on the left and 12,712 on the right, 6 changed sensors each, with
`link5_back` and `link5_front`/`link6` responding — both far above the 7,209
panel-preservation floor, side ratio 1.86 within the 4× limit.

**Step 3 stopped.** With 6/6 rows clean there were no natural production
failures, so all three failure clips had to be diagnostic negative controls. The
first in the registered order, `left_lobe_contact`, cannot be made to touch: over
the full frozen inward-shift grid `0.000–0.160 m` (**161 shifts recorded**) the
maximum penetration reached is **2.579 mm**, below the registered 5–30 mm band.
The clearance is mostly vertical and longitudinal, so a purely lateral shift
closes it only asymptotically. The plan requires stopping rather than extending
the grid, so the grid was not extended, no substitute control or source row was
chosen, and the production XML was never modified
(`.../pact_place_v104_review/control_shortfall_stop.json`, SHA
`9a4abfec9992edaff3bf957dc90900c2a6dd50e430e8cb7f7cf873fe4d11103a`). The other
two controls are reachable (9.57 mm and 36.64 mm at the grid end); the order rule
is what binds. **This shortfall is a property of the control recipe, not the
environment** — it happens because the pendant is far outside the arm's envelope,
the same fact that makes Steps 0–2 pass comfortably.

No review packet was produced, `human_approval.json` is absent and was not
created, Phase 0 was not run, and collection, conversion, training, and
learned-policy evaluation remain unauthorized.

### V10.4 review-v2 — the six-video packet, published for owner review

[`docs/PACT_PLACE_V104_REVIEW_REPAIR_PLAN.md`](docs/PACT_PLACE_V104_REVIEW_REPAIR_PLAN.md)
repairs the two Step-3 defects without touching production geometry, routing,
speeds, seeds, or results. The six V10.4 episodes are **reused byte-for-byte**;
no replacement episode was generated and no `env.step` was called.

**The v1 shortfall was a search defect, not a physical one.** Two things were
wrong. The frozen grid ended at `0.160 m`, and the audited left-lobe contact
occurs at `0.175 m` — 15 mm past where v1 stopped looking. And v1 displaced only
the target component's geom rather than rigidly translating the whole assembly,
which is what a physical intrusion would do. With both repaired, all three
controls certify and every audited anchor reproduces exactly:

| control | component | source row | shift | penetration | max frame | limiting body |
|---|---|---|---:|---:|---:|---|
| `left_lobe_contact` | `lobe_0` | 0 (left) | 0.175 m | **5.044 mm** | 88 | `fr3_link7` |
| `right_lobe_contact` | `lobe_1` | 3 (right) | 0.132 m | **5.239 mm** | 245 | `gripper/base` |
| `stem_contact` | `stem_0` | 0 (left) | 0.083 m | **5.455 mm** | 212 | `fr3_link7` |

At the certified frame all four instruments agree on every control — signed
distance negative, analytic GJK reporting intersection, live `data.contact`
non-empty, and the place audit classifying `mounted_fixture` and nothing else.
Each diagnostic scene is a separate compiled bundle (V3/V5 includes plus a
metadata copy renamed to the diagnostic stem), reloaded through the real task
sampler and confirmed compiled-static with bounds that enclose the shifted
geometry — so the V10.2 stale-broad-phase defect cannot recur here either. The
production XML is byte-identical before and after every control.

**Provenance bridge.** Every file the Step-0 preflight bound still matches except
one: the review runner, `b40e5a0f… → ddf96225…`, superseded by this repair. That
single path is the entire allowlist, and the bridge forgives only that exact
old→new transition — a different new hash on the same path still fails closed.
The bridge also distinguishes the contract/implementation pair the v1 artifacts
were actually **executed** under (`eb8f1174…` / `bd135e68…`) from the later live
aggregate (`455379b8…` / `bf4af91…`), which is not what ran and is never bound.
All six rows reconcile as strict-clean, 3 left and 3 right, minimum clearance
61.6 mm, zero pendant contact.

**The packet.** `diagnostics_output/pact_place_v104_review_v2/` holds exactly six
MP4s — three complete production successes (rows 0 left, 3 right, 4 left) and
three trimmed contact-centered controls — plus `provenance_bridge.json`,
`control_certificates.json`, `review_preflight.json`, `review_manifest.json`, and
`REVIEW.md`. All six decode back at 15.1515 fps with exact frame counts. Control
windows match the registered anchors: 40–89 (50 frames), 197–260 (64), 164–227
(64). Publication is atomic — rendered and decode-verified in a temporary
directory, then moved into place in one step — and an existing final directory is
refused.

One correction to the plan's own numbers: the left-lobe stem contact that trims
that clip does **not** reach 38.15 mm at frame 90. It *begins* at frame 90 at
0.103 mm and only reaches 38.15 mm at frame 193. Both figures are real; the
attribution was not. The certificate records them as separate frames, and the
clip is trimmed at 89 so neither is shown.

**Phase0-v2 is frozen and gated.** `scripts/run_pact_place_v104_phase0_v2.py`
recomputes every binding from file bytes instead of trusting embedded self-hashes
— a tampered artifact carries a tampered self-hash too — and refuses missing,
stale, partial, agent-created, or extra-video approvals. It creates no gate
directory and no row before the approval validates; run without one it exits with
`missing owner approval`. The 24-row manifest and every threshold are unchanged
(≥20/24 clean, ≥9/12 per side, zero pendant contact, ≥15 mm per-frame clearance),
and a passing gate sets only `phase0_passed: true`.

`human_approval.json` is absent and was not created. The required owner-authored
schema is documented in the packet's `REVIEW.md`, and a test asserts that the
documented bindings are exactly what the verifier demands, so a record pasted
from it validates as written.

### V10.5 V9.5 real clutter with a static pendant — stopped at Step 2

[`docs/PACT_PLACE_V105_V95_CLUTTER_STATIC_PENDANT_PLAN.md`](docs/PACT_PLACE_V105_V95_CLUTTER_STATIC_PENDANT_PLAN.md)
restores the settled **fixture-free V9.5** household-object clutter — the V5
scene, `PactPlaceCorridorV93Sampler`, `load_v95_palette`, `build_v95_layout`,
all four layout families, movable free bodies — and asks whether the V10.4
pendant shape can be moved inboard far enough to matter while keeping a 15 mm
floor. **It cannot, anywhere in the registered lattice.**

Lineage note, because it has been confused before: V10.5 restores V9.5 real
clutter but **not** the V9.5 low wall. The 51% seed-robustness result came from
the fixture-free V9.3 sampler with the settled V9.5 palette and layout.

**Step 1 passed.** Strict-clean status was re-derived from each retained row's
own telemetry rather than imported; all 192 fragility rows agree with their
stored boolean. **98/192 = 51.0% clean**, reproducing the recorded
`mean_clean_rate` of 0.5104 exactly, with every family/side cell above the
two-per-cell floor (minimum 8). Sixteen rows replayed through the live V9.3
sampler: 16/16 reconstructed, TCP residual 0.067–0.163 mm against a 1 mm limit.

**Step 2 selected nothing.** 96 scenes (32 `(x, r)` bundles × 3 poses) were
scored against **all 98** retained strict-clean trajectories with no early
termination and no failed rows. **0 survivors.** The binding predicate is
universal: **32/32** bundles put at least one historically clean trajectory
below the 15 mm floor. The trade is stark — the bundle with the most risk-band
witnesses (`x = 0.780, r = 0.325`, **153** in the 15–35 mm band) drops to
**3.4 mm** on its worst clean row, while the bundle with the highest floor
(`x = 0.800, r = 0.320`) reaches only **9.4 mm**. Outboard raises the floor and
empties the band; inboard fills the band and causes contact. The best candidates
sit at the lattice's outer corner, and the plan forbids extending it, so it was
not extended.

Predicates 1, 2, 5, 6 and 9 rejected nothing: the pendant never intersected a
household object, panel, tray or shell, initial clearance always held, the
grasp/lift/release windows never bound, every closest-risk witness bound a
lobe or stem on its own route side, and no clean row was made unclean by
reconstruction differences.

**A predicate defect was found and corrected before reporting.** The first
aggregation counted a direction as a risk-band witness only if the row's
*overall* lobe/stem minimum was already in band. Since the loaded outbound leg
almost always passes closer than the inbound leg, that gate suppressed every
inbound witness and made the lattice look as though the inbound approach never
came near the pendant. It does — with the gate removed, `left:inbound`
witnesses appear in 22/32 bundles and `right:inbound` in 5/32. The selection was
unchanged (0 survivors either way) but the reported *reason* would have been
wrong. The uncorrected run is preserved beside the corrected one, which was
re-derived from the same stored per-row scores without re-measuring anything.

Steps 3–6 did not run. Step 3 refuses without a selected bundle and was verified
to do so. No contact certificate, causal comparison, production scene, manifest,
review packet, video, or Phase-0 row exists, and no V10.5 scene was published.

**One consequence for V10.4.** Registering `PactPlaceCorridorV105Sampler`
required editing `enclosure_reach.py` and `run_pact_place_expert_screen.py`,
both bound by the V10.4 Step-0 preflight. The V10.4 review-v2 provenance bridge
now fails — correctly. Its data checks all still pass and the six published MP4s
are byte-identical; only the scoped implementation binding moved. By V10.4's own
rule that an implementation-hash change requires a new version rather than
silent regeneration, that packet is no longer approvable as published and was
**not** regenerated here.

### V10.5 audit and erratum — the narrative was wrong in two places

The V10.5 report above was audited as untrusted input
(`scripts/audit_pact_place_v105.py`,
`diagnostics_output/pact_place_v105_audit/audit.json`). All four sealed
artifacts re-hash correctly, all **192/192** retained rows verify, and an
**independently written flat-table aggregator** reproduced 9408 evaluations and
agreed with the primary scorer on **192 checks with 0 disagreements**.

Two reported claims were wrong:

- **"21 active clutter free bodies" is incorrect — there are 8 household
  objects.** The 21 counted nested MuJoCo mesh child bodies and the four
  corridor chicane bodies as distinct objects.
- **"`x=0.800, r=0.320` is the highest-floor candidate" is incorrect.** The
  highest symmetric floor is `x=0.800, r=0.325` at **13.4388185 mm**, with
  **zero** exact contacts and only **2 of 294** evaluations below 15 mm.
  `r=0.320` is second at 9.3898271 mm. The V10.5 write-up mislabelled the
  runner-up as the best, which understated how close the symmetric family came.

98/192 strict-clean is confirmed valid (4 rows retain no trajectory; all four
are unclean, so the count is unaffected). `risk_group_counts` was an ambiguous
name and is now `band_evaluations_by_group`: per `pose_id|side` group, the
number of **(trajectory, pose) evaluations** whose lobe/stem minimum lies in the
15–35 mm band.

### V10.6 asymmetric static pendant — qualifies geometrically, stopped at Step 4b

[`docs/PACT_PLACE_V106_ASYMMETRIC_PENDANT_PLAN.md`](docs/PACT_PLACE_V106_ASYMMETRIC_PENDANT_PLAN.md)
gives the two lobes independent radii — one **global** assembly for every layout
family, not per-family placement — on a registered 9-candidate lattice
(`x = 0.800`, `r_neg ∈ {0.325, 0.330, 0.335}`, `r_pos ∈ {0.295, 0.300, 0.305}`,
three poses). The crossbar's centre and half-length are derived from the
asymmetric stem endpoints, and connectivity is asserted rather than assumed.

**The asymmetry resolves what V10.5 could not.** Scored against all 98 clean
trajectories, **9/9 candidates are admissible and 4 achieve universal ≥15 mm
clearance** — the preregistered fallback was never needed. Selected
`x = 0.800, r_neg = 0.335, r_pos = 0.305`: **18.5703 mm** absolute minimum over
294 evaluations, **0** below floor, **0** contacts, 10.0 mm to the environment.
The negative side needs 30 mm more radius than the positive side, which no
symmetric assembly can express.

Certification passed: three compiled-static scenes with enclosing bounds, and
**6/6** witnesses where recorded, compiled, signed `mj_geomDistance` and
analytic GJK agree to five decimals. Every witness binds a lobe against a robot
link during `outbound_vessel_pass`, independently confirming the audited finding
that the meaningful near-pass is loaded outbound. Raw proximity causality passed
all seven checks (4512 left / 2004 right changed values, ratio 2.251,
deterministic, link5/link6 responding).

**The stop is the contact-risk certificate.** No group reaches robot-pendant
contact within the registered 30 mm displacement cap; `neg5|right` ends
**1.018 mm** short. Three defects in that probe were found and fixed before the
result was accepted — inherited contacts counted as new, a TCP-to-centre
direction that *increased* clearance because the limiting pair is
`lobe_0 ↔ fr3_link7`, and the held cup's grasp contacts counted as collisions.
Both uncorrected runs are preserved. Per the plan, a failed Step-4 check stops
V10.6, so the 48-row review pool and the six-video packet did not run.

Whether the 30 mm straight-line TCP probe is too weak an operationalization, or
the pendant is genuinely too far to matter, is an owner decision — changing the
probe after seeing the result is what preregistration forbids.

**Resolved in V10.7 (see below): the V10.6 Step-4b outcome is *diagnostically
inconclusive*, not a demonstration of physical infeasibility.** A cardinal-axis
TCP excursion is one weak operationalization of reachability; it holds the
carried target fixed while the arm moves and explores no joint-space or
closest-point direction. Repaired, it reaches contact in 1/6 groups. The V10.6
record's causal hashes have also been corrected: it originally quoted
`a0127a5e…`/`e1f4e76c…`, which belong to the *first, defective* probe run; the
published artifact is payload `6338f85c…`, raw `b4464ea4…`.

### V10.7 qualification repair — qualified offline, stopped at the pool floor

[`docs/PACT_PLACE_V107_QUALIFICATION_REPAIR_PLAN.md`](docs/PACT_PLACE_V107_QUALIFICATION_REPAIR_PLAN.md)
treats V10.6 geometry results as historical inputs and changes two things. The
ranking becomes **risk-aligned** — universal ≥15 mm clearance first, extra
clearance demoted *below* relevance — and the cardinal-TCP contact test is
**retired as a gate**. The registered relevance test is natural exact clearance
in the 15–35 mm band for all six `pose × side` groups plus six-group causal
sensing.

An immutable `specification.json` binds 9 sealed inputs and 20 implementation
files before anything runs; every later stage re-verifies it. Score NPZs are
written before the JSON that binds their raw SHA-256. The drift guard fired for
real once, halting a chain when a test file was edited after sealing.

**Selection changed.** The risk-aligned ranking picks
`x = 0.800, r_neg = 0.330, r_pos = 0.300` — derived, with the runner asserting
its sorted ranking equals an independent argmin and a test asserting no bundle
string appears in the runner. V10.6 had picked `0.335|0.305`, the *farthest*
admissible pendant. The new pick holds 16.8435 mm minimum over 294 evaluations,
0 below floor, 0 contacts, and the most risk-band evaluations (142) of the four
universal candidates; all six group minima lie in 15–35 mm.

Certification passed on **11 witnesses** (6 minima + 5 threshold-near) with
**zero** instrument disagreements. Six-group causality passed for **all six**
groups (2012–4648 changed values, 7–9 sensors, side ratio 2.251).

**The pool is where it stops: 21/48 clean = 43.8%**, Wilson 95%
[30.7%, 57.7%], against floors of 32/48 overall and 14/24 · 8/16 · 4/8 on the
balance axes. No packet was published, no video rendered.

**The pendant caused none of it.** Across 48 episodes there were **zero**
robot-or-target pendant contacts, and clean rows held 16.05–56.08 mm of
clearance. The failures are ordinary V9.5 household clutter: 21 clutter
contacts, 12 stability events, 11 task failures.

43.8% brackets the V9.5 corpus's own **51.0%** (98/192). The 32/48 floor and the
inherited 16/24 Phase-0 bar are both **66.7%** — above what this expert achieves
on real V9.5 clutter, exactly as the V9.5 fragility artifact recorded
(~12.25/24 expected against a bar of 20). The floor did what it was registered
to do: it refused a curated six-video packet for an environment that would not
pass Phase 0. **The open question is now the Phase-0 bar itself, not the
geometry** — the environment is qualified offline and causally sensed.

Three plumbing defects were exposed only by running real episodes, each fixed
with a regression test and each failed run preserved: the scene-hash guard read
`cfg.scene_xml` instead of `task_sampler_config.scene_xml_paths` (48/48
`sampling_failure`); the retained row copies an explicit subset of `policy_info`
that omitted the V10.5/V10.6 telemetry keys (48/48 `missing_frame_telemetry`);
and the policy read `_pact_manifest_row`, a sampler attribute it does not have
(null clearance on every completed episode).

**Owner visual-review packet (review-only).**
[`diagnostics_output/pact_place_v107_owner_review/`](diagnostics_output/pact_place_v107_owner_review/)
holds six complete retained trajectories replayed from the failed pool, with
[`REVIEW.md`](diagnostics_output/pact_place_v107_owner_review/REVIEW.md) and an
immutable manifest. **The pool remains failed at 21/48**; the packet exists
solely for owner visual assessment and neither reinterprets that result nor
authorizes downstream work. Nothing was executed to produce it — no episode, no
resampling, no `env.step`, no geometry or threshold change — and the manifest
rows were rebuilt from the frozen generator with each `row_sha256` asserted
against the executed row.

Selection is derived from `pool.json`, not hand-picked: among 54,846 valid
subsets balanced one-per-pose per outcome class, three left and three right, and
≥2 layout families per class, it minimises the maximum pendant clearance (then
total, then role-index tuple), which is what makes the pendant risk visually
legible without altering an episode. Chosen: successes 6/28/8, failures
45/40/20, with episode minima 24.248 / 26.307 / 16.052 mm and 21.517 / 21.399 /
13.081 mm. **All three failures are natural; none is an induced pendant
collision.** `scripts/verify_pact_place_v107_owner_review.py` re-derives the
selection and reconciles every hash, frame count and duration independently —
0 problems.
## Place corridor v10.8: an exploratory collection under owner override (stopped early)

**This is not a Phase-0 pass.** V10.7's gate failed at 8/24 and is permanently
closed. V10.8 collected demonstrations anyway, under an explicit owner override
for scientific curiosity, and the owner stopped it early at **141 of 152**
target successes with quotas unmet.

[`diagnostics_output/pact_place_v108_collection/`](diagnostics_output/pact_place_v108_collection/)
holds the frozen contract, the 353-row ledger and the close-out. 141 accepted,
212 rejected, 39.9% yield, 5.97 h, zero accepted rows with pendant contact.

**A create-only erratum corrects the V10.8 record.**
[`V108_ERRATUM.md`](diagnostics_output/pact_place_v109_train_eval/V108_ERRATUM.md)
and its JSON sit beside `closeout.json`, which is **not** modified. Seven
corrections, each re-derived from `ledger.jsonl` and the retained row files
rather than copied from the close-out:

- **E1** "no pendant involvement anywhere" was false. Zero *accepted* rows
  contacted the pendant — that part stands.
- **E2** one rejected attempt, `1a756c9304311cdc…`, recorded **42
  `mounted_fixture` contacts, one pendant-contact frame and zero clearance**. It
  is the only such row in 353 attempts and was correctly rejected.
- **E3** `closeout.json` reports `infrastructure_halts: 0`. There were **eight**
  `BrokenProcessPool` worker deaths, caused by terminating the pool on the
  owner's stop instruction with a batch in flight — owner-stop-induced
  terminations, not zero events and not spontaneous corruption. None advanced a
  seed stream or entered the 353 scientific rows, and they are a different thing
  from the **12 `sampling_failure`** ledger rows.
- **E4** exactly **17** cells equal quota and **two** exceed it, so 19 meet or
  exceed; **five** are short. "19 at quota" overstates the balance.
- **E5** only `F3|right|neg5` (one episode) cannot appear in both splits;
  `F3|right|pos5` (two) splits 1/1.
- **E6** the encoder-health figures came from **120 windows of one episode**,
  not the corpus.
- **E7** `episode_steps` are control steps; HDF5 `T = episode_steps + 1`, so T
  runs **356 to 627** and sums to **71,511**.

## Place corridor v10.9: converting, training and evaluating the 141 demos (exploratory)

Owner-authorized conversion, ACT-vs-PACT training and a paired learned-policy
evaluation on the 141 accepted V10.8 demonstrations. Plan:
[`docs/PACT_PLACE_V109_TRAIN_EVAL_PLAN.md`](docs/PACT_PLACE_V109_TRAIN_EVAL_PLAN.md).
Artifacts:
[`diagnostics_output/pact_place_v109_train_eval/`](diagnostics_output/pact_place_v109_train_eval/)
and
[`diagnostics_output/pact_place_v109_eval/`](diagnostics_output/pact_place_v109_eval/).
**Neither V10.7 nor V10.8 is reinterpreted.** Every authorization field on every
V10.9 artifact is false; `is_phase0_pass` is false throughout.

**Source freeze.** `scripts/verify_pact_place_v109_source.py` re-derives the
population from the ledger and the retained files and passes **33/33 checks**:
353 unique rows, 141 accepted and all strict-clean, every HDF5 hash matching its
row, 40 finite non-constant `(T,4,8,8)` float32 sensors per episode, T 356–627
summing to 71,511, zero accepted pendant contacts, clearance minimum
0.008272895299859126 m, and every task seed reproduced from its frozen cell
stream. Rows are ordered canonically by registered cell, then `attempt_index`,
then `attempt_id` — never by parallel-completion order. The manifest carries the
imbalance in its own body: families **39/38/38/26**, sides **75 left / 66
right**, two cells over quota, five short.

**Conversion.** `scripts/convert_pact_place_v109_to_act.py` is a new adapter —
the existing converter's `recovered_152` path is contract-specific — reusing the
proven V5 semantics. The frozen sensor order is preserved exactly
(`2198e29b796ce63f43d8b0db50a92da7d4429895f8571f7d87b655bc265c8fe1`), including
`link5_front_*` **before** `link5_back_*`. That order is *not* alphabetical; the
encoder is weight-shared but the PACT transformer assigns positional embeddings
by sensor slot, so sorting would silently relabel every sensor. A test asserts
it.

**Embeddings.** The frozen `SurfaceEmbeddingEncoder`
(`6fd2dd03…`, schema `pact_surface_embedding_encoder_v1`, whose checkpoint
declares the same sensor-order hash) wrote `(T,40,32)` tokens over the whole
converted corpus. `scripts/verify_pact_place_v109_embeddings.py` reads them back
independently: **2,854,800 windows, all finite, 0 dead dimensions**, global std
1.1127, per-dimension std 0.249–1.393, 12.72% of readings valid. Preservation is
proved the strong way — `action`, `qpos`, `qvel`, wrist RGB, raw proximity and
the per-sensor extrinsics/intrinsics were **re-derived from the V10.8 source for
all 141 episodes** and compared element-wise. 141/141 preserved.

**Split.** One byte-identical **113/28** split, master seed 2026082901,
stratified over the 24 cells and decided only by SHA-256 of identity — never by
loss, length, clearance or any learned outcome. All 24 cells appear in training,
23 in validation; the sole-row `F3|right|neg5` goes to training and the two-row
`F3|right|pos5` splits 1/1. Tests assert the counts, the per-cell floor and
independence from input order.

**Training.** Fresh root `/root/pact_place_v108_141_pact_vs_act_chunk100_seed3101`.
V5 chunk-100 parameters exactly: seed 3101, 2000 epochs, batch 8, lr 1e-5, KL 10,
chunk 100, hidden 512, ff 3200, 7+7 layers, 8 heads, ResNet-18, wrist only,
state 9 / action 8, 4 workers, checkpoint every 200, no W&B,
`episode_horizon=635`. A parsed flag diff runs before training and fails closed:
PACT differs **only** by `--ckpt_dir` and the five proximity flags. The ACT
submodule differs from the V5 training commit `01751759` by four **added**
evaluation scripts and nothing else — 0 deletions, no training, model or loader
source touched.

| | ACT | PACT |
|---|---|---|
| epochs | 2000/2000 | 2000/2000 |
| wall clock | 72.0 min | 76.0 min |
| best epoch | 1883 | 1841 |
| best validation loss | 0.11186 | 0.11549 |
| strict reload / offline smoke | pass | pass |
| `input_proj_proximity.weight` | n/a | `[512, 32]` |

Proximity consumption is proved **causally**: the same batch run with real
versus zeroed frozen embeddings moves PACT's actions by max 1.376, mean 0.219.

**Paired evaluation — 40 held-out instances, 80 rollouts, 0 failures, 2.44 h.**
A separately versioned evaluator binds `PactPlaceCorridorV106Sampler` and the
three certified V10.7 scenes; the V2 scene and sampler are refused, and the
V10.6 sampler independently verifies each loaded scene against the row's
`pact_v106_scene_sha256`. Instances are balanced 10 per family, 20/20 by side,
14/13/13 by pose, all 24 cells present, 16 doubled by hash-ranked constrained
selection, with seeds asserted disjoint from 1,608 prior seeds.

| endpoint | ACT | PACT | PACT−ACT | 95% CI (pp) | exact McNemar |
|---|---|---|---|---|---|
| task success | 14/40 | 11/40 | −7.5 | [−20.2, +5.2] | 0.4531 |
| **collision-free task success** | **8/40** | **6/40** | **−5.0** | **[−16.9, +6.9]** | **0.6875** |
| strict-clean task success | 8/40 | 6/40 | −5.0 | [−16.9, +6.9] | 0.6875 |
| pendant-free task success | 14/40 | 11/40 | −7.5 | [−20.2, +5.2] | 0.4531 |
| intrusion-panel contact | 10/40 | 8/40 | −5.0 | [−18.8, +8.8] | 0.7266 |
| clutter contact | 17/40 | 18/40 | +2.5 | [−15.2, +20.1] | 1.0000 |

**The result is null.** PACT did not beat ACT on any endpoint; every interval
crosses zero and no McNemar p-value falls below 0.45. **Zero pendant contact in
all 80 rollouts** — the certified pendant was never touched by a learned policy.
The one directional signal is contact *volume*: PACT logged 43,845
intrusion-panel contact entries against ACT's 140,424, a 3.2× reduction, while
touching the panel in two fewer episodes — but the episode-level difference is
not significant and the entry counts are not a paired test. The V5 chunk-100
result (PACT 19/40 vs ACT 13/40) reversed sign here; the environment and the
training corpus both changed, so that comparison is **contextual only**.

Nothing here reopens V10.7, whose Phase-0 gate remains failed at 8/24 and
permanently closed. Every authorization field on every V10.9 artifact is false.

## Place corridor v10.10: four-object ACT/PACT replication

V10.10 keeps the certified V10.7 static pendant and the V9.5 layout families,
but activates exactly four household-clutter objects: `Soap_Bottle_30`,
`Plate_10`, `Plate_22`, and `Soap_Bottle_11`. The target cup, pendant,
intrusion panel, tray, and enclosure are not counted as clutter. The registered
plan is
[`docs/PACT_PLACE_V1010_FOUR_OBJECT_TRAIN_EVAL_PLAN.md`](docs/PACT_PLACE_V1010_FOUR_OBJECT_TRAIN_EVAL_PLAN.md).

The source contains 144 strict-clean demonstrations, exactly six from every
family×side×pendant-pose cell. The frozen split is 120/24: five training and one
validation episode per cell. ACT and PACT use seed 3101, 2,000 epochs, batch 8,
learning rate `1e-5`, KL weight 10, chunk size 100, hidden size 512, feed-forward
size 3,200, 7+7 transformer layers, 8 heads, ResNet-18 wrist RGB, state/action
dimensions 9/8, and no W&B. PACT differs only by the registered proximity-token
flags; the preflight compares the parsed commands and refuses any other
difference.

### What a plain Git pull does not provide

A byte-for-byte reproduction needs more than source code. Before handing this
recipe to another machine, make sure the V10.10 scripts, tests, scene changes,
and ACT submodule pointer have actually been committed and pushed. The large
artifacts below are normally not stored in Git and must either be transferred
separately or regenerated:

- `diagnostics_output/pact_place_v1010_collection/` and the 144 raw HDF5 rows;
- `assets/act_style_data/pact_place_v1010_144/`, if conversion is to be skipped;
- `/root/pact_frontend_screen_artifacts/encoder_v1/embedding_encoder_frozen.pt`;
- `/root/pact_place_v1010_144_pact_vs_act_chunk100_seed3101/`, if evaluating the
  original checkpoints without retraining;
- the frozen evaluation manifest and the small upstream seed-audit manifests if
  the exact manifest hash, rather than just the same 40 evaluation rows, is
  required.

The reference source revisions are ACT submodule
`f4f59d7975e7d1d52403df92ee8a789fbc3e14c3` and MolmoSpaces submodule
`ed045d757fe0ccbd848ba7903773e32ec99f2f29`. Run
`git submodule update --init --recursive`, then require
`git -C submodules/act status --short` to print nothing. Training preflight
fails if the ACT training/model/loader files are dirty.

Reference runtime: Python 3.11.15, PyTorch 2.7.1+cu126, MuJoCo 3.5.0,
NumPy 2.4.6, h5py 3.16.0, and an NVIDIA A10. Use the same environment when an
exact checkpoint comparison matters. Fresh retraining on another GPU can be a
scientific replication, but floating-point nondeterminism means it is not
guaranteed to reproduce the checkpoint hashes or the exact 40-row counts.

### Required artifact hashes

| artifact | SHA-256 |
|---|---|
| frozen proximity encoder | `6fd2dd037e3236b5b6bf7fce8cb2709ead0cf52adcbbe9cbad1061efc2fe3206` |
| converted 144-episode tree | `69406b77e565049e9070557c010071e8544ef055c6568db6fd4b48d85a783597` |
| split manifest | `35e5b7085eb2b2e18ec35a3624432bb95b5ec45de9f8012411b06a3ef83352f1` |
| dataset statistics | `b4f67f412b605ec92641c55a12270b9afb5b22bff8ac219155ee0a4da34acffb` |
| ACT `policy_best.ckpt` | `b14b51f48e15ef94e761284a0c373cd115f4005fbbd43d2508a4acbb2b5ff154` |
| PACT `policy_best.ckpt` | `75889ea4332666e812f60190b9f8428a597b138f23523bf68040e8bfc2d3fb87` |
| evaluation manifest | `35bcc74fa96b5bd6fe8fd161ae0ed3e44892bc38bde0eab6ec31468a316ba614` |

Verify the external encoder before doing any conversion or PACT evaluation:

```bash
cd /root/prox_learning_pact_remediation
PY=/root/act_retrain_venv/bin/python
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MLSPACES_ASSETS_DIR="$PWD/assets"

test "$(sha256sum /root/pact_frontend_screen_artifacts/encoder_v1/embedding_encoder_frozen.pt | cut -d' ' -f1)" = \
  6fd2dd037e3236b5b6bf7fce8cb2709ead0cf52adcbbe9cbad1061efc2fe3206
$PY -m unittest tests.test_pact_place_v1010_eval_infrastructure -v
```

Use a fresh clone or fresh output namespace. The manifests, terminal row
results, and reports are create-only. Do not delete or overwrite a completed
run to make room for a retry. Allow about 35 GiB free for collection through
trajectory-retaining evaluation; the training preflight alone requires 11.5
GiB of peak space.

### Optional: regenerate the 144 demonstrations

Skip this section when the verified V10.10 collection directory has been
transferred. From a clean output tree:

```bash
$PY scripts/run_pact_place_v1010_preflight.py --workers 6
$PY scripts/run_pact_place_v1010_collect.py --workers 8

$PY - <<'PY'
import json
from pathlib import Path
p = Path("diagnostics_output/pact_place_v1010_collection/closeout.json")
d = json.loads(p.read_text())
assert d["quotas_met"] is True
assert d["accepted_total"] == 144
assert d["cells_short"] == {}
print(d["payload_sha256"])
PY
```

The reference collection used 313 attempts, took 6.50 hours, and produced six
accepted rows in each of the 24 cells. Hardware timing is not a gate.

### Recommended command: conversion through final evaluation

Once the collection close-out and encoder exist, this driver verifies the
source, converts all episodes, computes and verifies `(T,40,32)` frozen
embeddings, creates the balanced split, trains ACT then PACT, verifies both
checkpoints, builds the held-out manifest, runs the smoke and full evaluations,
and finalizes the paired analysis:

```bash
LOG=/root/pact_place_v1010_repro_logs
$PY scripts/run_pact_place_v1010_pipeline.py \
  --log-dir "$LOG" \
  --workers 8 \
  --eval-workers 4
```

The driver may be restarted after a fully completed stage because it skips that
stage's immutable marker. Do not assume an interrupted evaluation stage is
automatically resumable; inspect its row artifacts and use a separately named
output root rather than deleting evidence.

### Training ACT and PACT explicitly

If conversion, embedding verification, and the 120/24 split already exist, run
the three training commands below. Preflight records the complete parsed ACT
and PACT commands in `training_preflight.json`; inspect that artifact when an
exact command line is needed for a bug report.

```bash
LOG=/root/pact_place_v1010_repro_logs
mkdir -p "$LOG"

$PY scripts/run_pact_place_v1010_train.py \
  --stage preflight --log-dir "$LOG"
$PY scripts/run_pact_place_v1010_train.py \
  --stage train --log-dir "$LOG"
$PY scripts/verify_pact_place_v1010_training.py
```

The hard-coded training root must be absent or empty before preflight:
`/root/pact_place_v1010_144_pact_vs_act_chunk100_seed3101/`. ACT trains first,
then its intermediate checkpoints are pruned; PACT trains second. Both retain
`policy_best.ckpt`, `policy_last.ckpt`, `dataset_stats.pkl`,
`run_manifest.json`, and `epoch_log.jsonl`.

The reference verification is
[`diagnostics_output/pact_place_v1010_train_eval/training_verification.json`](diagnostics_output/pact_place_v1010_train_eval/training_verification.json):

| | ACT | PACT |
|---|---:|---:|
| epochs | 2,000/2,000 | 2,000/2,000 |
| best epoch | 1,998 | 1,859 |
| best validation loss | 0.10621 | 0.10386 |
| strict reload and offline smoke | pass | pass |

PACT's proximity-consumption check must pass: on the same batch, replacing the
real embeddings with zeros changed its actions by mean absolute 0.204 and
maximum absolute 1.852 in the reference run. Validation loss is not the policy
comparison endpoint.

### Evaluate an existing ACT/PACT checkpoint pair

Put the checkpoints and the shared `dataset_stats.pkl` under the two exact
training-root directories above, verify their hashes, and run:

```bash
$PY scripts/build_pact_place_v1010_eval_manifest.py

$PY scripts/run_pact_place_v1010_eval.py \
  --stage smoke --workers 4 --save-trajectory --h5-only

$PY scripts/run_pact_place_v1010_eval.py \
  --stage full --workers 4 --save-trajectory --h5-only

$PY scripts/finalize_pact_place_v1010_eval.py
```

The evaluator hard-caps concurrency at four workers. Each process also forces
its native BLAS/OpenMP pools to one thread; raising the worker count previously
exhausted the container's PID/thread limit. `--h5-only` suppresses ffmpeg but
still retains every `trajectory.h5` and `actions.npz`. The smoke is an
infrastructure check only and never gates on policy performance. The reference
smoke took about 37 minutes; the full 80-rollout evaluation took 5.75 hours on
the A10.

For byte-level evaluation reproduction, use the transferred frozen manifest
instead of rebuilding it. Rebuilding without the older V10.7–V10.9 seed-audit
manifests produces the same registered V10.10 row stream but a different audit
payload and therefore a different manifest hash.

### Reference V10.10 result

The repaired run completed 80/80 rollouts with zero infrastructure failures.
All 40 ACT/PACT pairs had identical initial observations across all 421 retained
observation datasets.

| endpoint | ACT | PACT | PACT−ACT | paired 95% CI (pp) | exact McNemar |
|---|---:|---:|---:|---:|---:|
| task success | 11/40 | 14/40 | +7.5 | [−10.01, +25.01] | 0.5811 |
| **collision-free task success** | **7/40** | **10/40** | **+7.5** | **[−7.02, +22.02]** | **0.5078** |
| strict-clean task success | 7/40 | 10/40 | +7.5 | [−7.02, +22.02] | 0.5078 |
| clutter-contact episode | 22/40 | 17/40 | −12.5 | [−33.50, +8.50] | 0.3593 |
| pendant-contact episode | 2/40 | 0/40 | −5.0 | not preregistered | 0.5000 |

This is a directional PACT advantage, not a superiority claim: the registered
primary interval crosses zero. The verified machine-readable artifacts are
[`analysis.json`](diagnostics_output/pact_place_v1010_eval_infra_repair_01/analysis.json)
and
[`full_run.json`](diagnostics_output/pact_place_v1010_eval_infra_repair_01/full_run.json).
Their payload hashes are respectively
`2c09430f581ab2c0f57f434d15f83753ae8563fd1ffeda275478911d61ea1301`
and `c6c043e597d265dd3ba74c39fbc033724ecec332fc5e7b31df6486cfa8907009`.

Two secondary-reporting caveats do not affect the registered primary endpoint:

- the legacy evaluator did not emit `v109r_funnel`, so `touched=0` and
  `held=0` in the current `analysis.json` are unavailable values, not real
  zeros; direct reconstruction from the retained `GraspStateSensor` gives ACT
  27 touched / 12 held and PACT 28 touched / 20 held;
- contact summary mode did not retain geom-pair identities, so per-object
  contact counts cannot be reconstructed from `analysis.json`; only aggregate
  contact classes and the per-object stability events are valid.

As with V10.9, this exploratory comparison does not reopen V10.7's failed
Phase-0 gate and does not authorize downstream collection, training, or
evaluation by itself.
