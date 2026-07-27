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
