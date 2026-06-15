# PLA RUNBOOK — install, data, training, eval, analysis

End-to-end instructions for reproducing every step of the PLA experiment
on a single 24 GB GPU (no Brev / cloud). All paths are repo-relative
unless absolute. Last updated 2026-05-13.

**Contents**

1. [Hardware + Software Setup](#1-hardware--software-setup)
2. [The two-venv split (critical)](#2-the-two-venv-split-critical)
3. [Data Collection](#3-data-collection)
4. [Proximity Audit (signal diagnostic)](#4-proximity-audit-signal-diagnostic)
5. [Lock Clutter Bins](#5-lock-clutter-bins)
6. [Encoder Masking + the Five Experiment Variants](#6-encoder-masking--the-five-experiment-variants)
7. [Training (per-model commands)](#7-training-per-model-commands)
8. [Parallel-launch strategy on a single 24 GB GPU](#8-parallel-launch-strategy-on-a-single-24-gb-gpu)
9. [Eval Rollouts (the multi-seed harness)](#9-eval-rollouts-the-multi-seed-harness)
10. [Analysis + Plots](#10-analysis--plots)
11. [Statistical Pipeline (TBD)](#11-statistical-pipeline-tbd)
12. [Failure-mode Analysis (TBD)](#12-failure-mode-analysis-tbd)
13. [What each experiment must measure (CSV schema)](#13-what-each-experiment-must-measure-csv-schema)
14. [Quick-reference launch sequences](#14-quick-reference-launch-sequences)

---

## 1. Hardware + Software Setup

| Component        | Required                                                        |
|------------------|-----------------------------------------------------------------|
| GPU              | 1 × 24 GB CUDA (4090 or A100-20G+; 4090 fits everything we run) |
| CPU RAM          | ≥ 32 GB (data collection workers + MP4 decode are RAM-heavy)    |
| Disk             | ≥ 250 GB free under `runs/`, `assets/datagen/`, `analysis_output/` |
| OS               | Linux + EGL for headless MuJoCo (`MUJOCO_GL=egl`)              |

The repo lives at `/home/jaydv/code/prox_learning/`. Submodules are
populated at clone time (`git submodule update --init --recursive`).

## 2. The two-venv split (critical)

The pipeline runs under two distinct Python environments. **Mixing them
silently breaks things.**

| Env                                    | Binary                                                            | Purpose                                                    | Has                                          | Lacks                                  |
|----------------------------------------|-------------------------------------------------------------------|------------------------------------------------------------|----------------------------------------------|----------------------------------------|
| MolmoBot-Pi0 venv (TRAINING)            | `submodules/MolmoBot/MolmoBot-Pi0/.venv/bin/python`              | Training, audit, dataset analysis                          | CLIP, transformers, ACT, decord, torch       | mujoco_warp (not needed offline)       |
| mlspaces conda env (DATA + ROLLOUT)     | `/opt/conda/envs/mlspaces/bin/python`                            | Data generation + simulation rollouts                      | mujoco_warp, filament, GPU-physics renderer  | (transformers was installed 2026-05-12)|

Activation patterns:

```bash
# Training context
source submodules/MolmoBot/MolmoBot-Pi0/.venv/bin/activate
python -m pla.train ...

# Rollout / datagen context
/opt/conda/envs/mlspaces/bin/python -m pla.rollout_eval ...
# (or)
/opt/conda/envs/mlspaces/bin/python -m molmo_spaces.data_generation.main FrankaSkinPickAndPlacePilotMediumConfig
```

Both envs share `runs/`, so a ckpt saved by MolmoBot-Pi0 loads cleanly in
mlspaces at eval time. The training scripts source the right env
automatically (`scripts/launch_medium_ablations.sh`).

Required environment variables for any rollout / datagen:

```bash
export MLSPACES_ASSETS_DIR=/home/jaydv/code/prox_learning/assets
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export JAX_PLATFORMS=cpu        # we don't use JAX-on-GPU; mujoco_warp does its own CUDA
export PYTHONUNBUFFERED=1
```

## 3. Data Collection

### 3.1 The medium training dataset (already collected)

100-house pick-and-place collection with 29-sensor proximity readings.
Lives under
`assets/datagen/pick_and_place_skin_pilot_medium_v1/FrankaSkinPickAndPlacePilotMediumConfig/<timestamp>/`.
Current snapshot (2026-05-12 → 13):

- 82 house directories (18 skipped; planner could not generate viable tasks)
- 343 successful trajectories (dataset is filtered for success)
- 89,029 timesteps across 326 unique task descriptions
- 5.7 GB total, 1.37 GB of MP4 (exo + wrist, RGB + depth per episode)

To regenerate (will take 20+ hours on the 4090, GPU contention with
training):

```bash
/opt/conda/envs/mlspaces/bin/python -m molmo_spaces.data_generation.main \
    FrankaSkinPickAndPlacePilotMediumConfig
```

The config sets `proximity_sensor_period_ms=20` (verified non-zero —
see `[[dataset_zero_proximity_bug]]` for the prior incident where this
field was 0.0 and produced an empty proximity stream).

### 3.2 The planner holdout dataset (already collected)

10 held-out houses (11–20) with planner-driven trajectories. Used to
**(a) define clutter bins** and **(b) provide a deterministic comparison
target** during eval.

Path:
`assets/datagen/pick_and_place_skin_pilot_eval_holdout_v1/FrankaSkinPickAndPlacePilotEvalHoldoutConfig/20260511_021228/`

To regenerate:

```bash
/opt/conda/envs/mlspaces/bin/python -m molmo_spaces.data_generation.main \
    FrankaSkinPickAndPlacePilotEvalHoldoutConfig
```

### 3.3 Health check after collection

```bash
source submodules/MolmoBot/MolmoBot-Pi0/.venv/bin/activate
python - <<'PY'
import h5py
from pathlib import Path
root = Path("assets/datagen/pick_and_place_skin_pilot_medium_v1/FrankaSkinPickAndPlacePilotMediumConfig/20260511_210545")
ok = empty = bad = 0
for p in root.glob("house_*/trajectories_batch_*.h5"):
    try:
        with h5py.File(p, "r") as f:
            tks = [k for k in f.keys() if k.startswith("traj_")]
            if tks: ok += 1
            else:   empty += 1
    except Exception:
        bad += 1
print(f"houses: ok={ok} empty={empty} bad={bad}")
PY
```

Expected: `ok=82 empty=0 bad=0` for the current snapshot.

## 4. Proximity Audit (signal diagnostic)

Answers four pre-training questions: *is there proximity signal at all,
is it approach-biased, is it clutter-correlated, and does any sensor
self-sense?* Re-run any time the dataset changes.

```bash
source submodules/MolmoBot/MolmoBot-Pi0/.venv/bin/activate
python -m pla.audit_proximity \
    --data_root assets/datagen/pick_and_place_skin_pilot_medium_v1/FrankaSkinPickAndPlacePilotMediumConfig/20260511_210545 \
    --out diagnostics_output/proximity_audit_medium_full
```

Outputs:
```
diagnostics_output/proximity_audit_medium_full/
├── report.md              human-readable
├── summary.json           machine-readable
├── q1_per_link_activation.png
├── q1_threshold_sweep.png
├── q1b_reading_distribution.png   ← self-sensing diagnostic
├── q2_phase_split.png
├── q3_clutter_correlation.png
└── q4_near_contact.png
```

Runtime: ~12 s on 82 houses. The audit script skips any h5 file modified
in the last 10 min so it is safe to re-run while data collection is still
writing.

Headline result on the current snapshot: link2 is self-sensing (within-
trajectory std 2.27 cm — sensor stares at the robot's own torso). The
encoder masks it by default. See `paper/section3_proximity_signal_draft.md`
for the full narrative.

## 5. Lock Clutter Bins

Per-house "low / medium / high" assignments computed from planner
trajectories. Locked once and reused across **every** eval run so PLA and
VLM (and ablations) are scored against the same bins.

```bash
source submodules/MolmoBot/MolmoBot-Pi0/.venv/bin/activate
python -m scripts.lock_clutter_bins \
    --planner_root assets/datagen/pick_and_place_skin_pilot_eval_holdout_v1/FrankaSkinPickAndPlacePilotEvalHoldoutConfig/20260511_021228 \
    --out analysis_output/eval_medium_v1/clutter_bins.json
```

Output (committed):
```
{
  "11": "medium", "12": "medium", "13": "high", "14": "high", "15": "medium",
  "16": "low",    "17": "low",    "18": "high", "19": "low",  "20": "medium",
  "__meta__": {... per-house values, quantiles, source path ...}
}
```

Pass this path to every `pla.eval_harness` invocation via
`--clutter_bins_path`. The harness mirrors the file into each run's
output dir for self-containment.

## 6. Encoder Masking + the Five Experiment Variants

`pla/proximity_encoder.py:ProximityEncoder(mask_links=...)` accepts a
tuple of link names whose channels are zeroed at input. The shape stays
`(B, 29, 512)` regardless of mask, so checkpoints are interchangeable
across mask choices and the positional embeddings in `pla/policy.py` are
unaffected.

| Experiment           | `use_proximity` | `mask_links`        | Sensors carrying signal |
|----------------------|-----------------|---------------------|-------------------------|
| **PLA (headline)**   | true            | `link2`             | link3 + link5 + link6   |
| **VLM (baseline)**   | false           | n/a                 | none                    |
| **PLA-no-mask**      | true            | `()` (no mask)      | all 29 incl. broken link2 |
| **PLA-EE-only**      | true            | `link2, link3`      | link5 + link6 only      |
| **Oracle**           | n/a (different policy class) | n/a    | privileged sim state    |

### Why each ablation matters

- **PLA-no-mask** defends the headline against the reviewer challenge
  "the mask is doing the work, not the proximity signal".
- **PLA-EE-only** tests whether the forearm contributes. Audit says
  link3 has +0.38 clutter correlation; PLA-EE-only directly measures
  whether dropping it hurts policy performance.
- **Oracle** sets the upper bound. **Without oracle, PLA-vs-VLM gaps are
  uninterpretable** — a 10-pp gap is huge if oracle is at 65 %, marginal
  if oracle is at 95 %.

## 7. Training (per-model commands)

All training uses the MolmoBot-Pi0 venv. All five model commands below
are self-contained: copy-paste, edit `--run_name`, fire.

Common environment:

```bash
cd /home/jaydv/code/prox_learning
source submodules/MolmoBot/MolmoBot-Pi0/.venv/bin/activate
export PYTHONUNBUFFERED=1
DATA_ROOT=assets/datagen/pick_and_place_skin_pilot_medium_v1/FrankaSkinPickAndPlacePilotMediumConfig/20260511_210545
```

### 7.1 PLA (headline; link2 masked)

```bash
python -m pla.train \
    --use_proximity true --use_language true --mask_links link2 \
    --run_name medium_pla_v1 \
    --data_root "$DATA_ROOT" \
    --num_steps 50000 --batch_size 8 --num_workers 2 \
    --ckpt_every 5000 --log_every 50 \
    --use_wandb true --wandb_project pla \
    2>&1 | tee logs/train_medium_pla_v1.log
```

### 7.2 VLM (RGB-only baseline; no proximity)

```bash
python -m pla.train \
    --use_proximity false --use_language true \
    --run_name medium_vlm_v1 \
    --data_root "$DATA_ROOT" \
    --num_steps 50000 --batch_size 8 --num_workers 2 \
    --ckpt_every 5000 --log_every 50 \
    --use_wandb true --wandb_project pla \
    2>&1 | tee logs/train_medium_vlm_v1.log
```

### 7.3 PLA-no-mask (all 29 sensors including broken link2)

```bash
python -m pla.train \
    --use_proximity true --use_language true --mask_links none \
    --run_name medium_pla_no_mask_v1 \
    --data_root "$DATA_ROOT" \
    --num_steps 50000 --batch_size 8 --num_workers 2 \
    --ckpt_every 5000 --log_every 50 \
    --use_wandb true --wandb_project pla \
    2>&1 | tee logs/train_medium_pla_no_mask_v1.log
```

### 7.4 PLA-EE-only (mask link2 AND link3, keep link5 + link6 only)

```bash
python -m pla.train \
    --use_proximity true --use_language true --mask_links link2,link3 \
    --run_name medium_pla_ee_only_v1 \
    --data_root "$DATA_ROOT" \
    --num_steps 50000 --batch_size 8 --num_workers 2 \
    --ckpt_every 5000 --log_every 50 \
    --use_wandb true --wandb_project pla \
    2>&1 | tee logs/train_medium_pla_ee_only_v1.log
```

### 7.5 Oracle (privileged-state baseline)

Built as a sibling module (`pla/oracle.py`). Uses joint pos+vel, TCP
pose, and ground-truth object pickup/placement poses — no RGB, no
proximity. **Privileged state vector is 35-d**: qpos (7) + qvel (7) +
tcp_pose (7) + obj_start_in_base (7) + obj_end_in_base (7). The 5
modalities are split into 5 separate tokens fed to the transformer
encoder.

```bash
python -m pla.oracle train \
    --run_name medium_oracle_v1 \
    --data_root "$DATA_ROOT" \
    --num_steps 50000 --batch_size 32 --num_workers 2 \
    --ckpt_every 5000 --log_every 50 \
    --use_wandb true --wandb_project pla \
    2>&1 | tee logs/train_medium_oracle_v1.log
```

The oracle dataloader does **not decode MP4 frames** (no image, no
proximity), so throughput is much higher — comfortably batch=32 on the
4090 and ~50 samp/s observed in the smoke. Expect ~3 h for 50k steps
vs ~8 h for the image/proximity-bound models.

### Verifying any training run

After ~250 steps you should see:

- Loss dropping monotonically (~10 → ~3 → ~2 in the first 500 steps).
- No NaNs in `l1` or `kl`.
- `samp/s` ≥ 10 for image-based models, ≥ 40 for oracle.
- A checkpoint at `runs/<run_name>/step_00005000.pt`.

If loss plateaus above ~5 or `samp/s` < 5, something is wrong (most
likely dataloader bottleneck on a worker count of 0; bump to 2).

## 8. Parallel-launch strategy on a single 24 GB GPU

Important reality check: **5-way parallel doesn't fit on a 24 GB GPU**.
Per-model GPU memory at batch=8:

| Model       | ~Active GPU memory at batch 8       | Notes                                 |
|-------------|-------------------------------------|----------------------------------------|
| PLA         | ~13 GB                              | ResNet + transformer + proximity branch |
| VLM         | ~12 GB                              | Same minus proximity branch            |
| PLA-no-mask | ~13 GB                              | Same as PLA                            |
| PLA-EE-only | ~13 GB                              | Same as PLA                            |
| Oracle      | ~3 GB                               | No image / proximity; lightweight      |

What actually works:

| Strategy                                  | Wall clock | Risk                          |
|-------------------------------------------|------------|--------------------------------|
| **Sequential (current launch script)**    | ~32 h total (4 × 8 h + 3 h oracle, sequential) | none                       |
| **Oracle in parallel with one image-model** | ~26 h (max(8, 3) + 24h serial of remaining 3) | low — oracle uses ~3 GB |
| **Two image models in parallel** at batch=4 each | ~24 h (2× wallclock per model) | OOM risk, throughput halves |
| **Three+ in parallel**                    | n/a — won't fit | guaranteed OOM            |

**Recommendation**: pair `pla.oracle` with one image-based model from
the start (saves ~3 h). Do not run two image models concurrently —
batch=4 halves throughput per model and OOM hazard is real.

Concretely, run two terminals:

```bash
# Terminal A — image-based queue (sequential)
bash scripts/launch_medium_ablations.sh
# trains: pla → vlm → nomask → eeonly  (~32 h)

# Terminal B — oracle (single)
source submodules/MolmoBot/MolmoBot-Pi0/.venv/bin/activate
python -m pla.oracle train --run_name medium_oracle_v1 \
    --data_root "$DATA_ROOT" --num_steps 50000 --batch_size 32 \
    --use_wandb true --wandb_project pla \
    2>&1 | tee logs/train_medium_oracle_v1.log
# completes in ~3 h regardless of Terminal A progress
```

After both terminals finish: all 5 checkpoints are under `runs/`.

### Killing the current sequential launcher (if you want to restart)

```bash
pkill -f "scripts/launch_medium_ablations.sh"
pkill -f "pla.train"
```

Wait ~10 seconds, then verify with `pgrep -af pla.train`.

## 9. Eval Rollouts (the multi-seed harness)

After all checkpoints exist, run **one** harness invocation to roll out
every model across every (seed, scene) combination and emit one wide
CSV that everything downstream groups on.

```bash
cd /home/jaydv/code/prox_learning
export PYTHONPATH=submodules/molmospaces:.
export MLSPACES_ASSETS_DIR=/home/jaydv/code/prox_learning/assets
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1

/opt/conda/envs/mlspaces/bin/python -m pla.eval_harness \
    --models \
        pla=runs/medium_pla_v1/latest.pt:use_prox=true \
        vlm=runs/medium_vlm_v1/latest.pt:use_prox=false \
        nomask=runs/medium_pla_no_mask_v1/latest.pt:use_prox=true \
        eeonly=runs/medium_pla_ee_only_v1/latest.pt:use_prox=true \
        oracle=runs/medium_oracle_v1/latest.pt:use_prox=false \
    --seeds 2028,2029,2030,2031,2032 \
    --house_inds 11,12,13,14,15,16,17,18,19,20 \
    --samples_per_house 2 \
    --num_workers 2 \
    --clutter_bins_path analysis_output/eval_medium_v1/clutter_bins.json \
    --out_dir analysis_output/eval_medium_v1_<TS> \
    2>&1 | tee logs/eval_harness_<TS>.log
```

5 models × 5 seeds × 10 houses × 2 samples = **500 rollouts**, ~6 h on
the 4090.

> **Oracle eval routing is live (2026-05-13).** `pla.eval_policy.PLAInferencePolicy`
> now detects `policy_kind == "oracle"` in the checkpoint and routes to
> `OraclePolicy` with the privileged-state forward path. The obs→state
> conversion (`_tensors_from_obs_oracle`) extracts qpos / qvel / tcp_pose
> from the rollout obs dict (with multiple fallback paths) and pulls
> obj_start / obj_end from `self.task` if they're not in obs. First-time
> missing-field warnings are emitted via the logger so any obs-schema
> surprises surface immediately at smoke time. No changes to `rollout_eval.py`
> or `eval_harness.py` callers required — the existing `--models` argument
> works for oracle exactly like the other models.

## 10. Analysis + Plots

The harness auto-calls `pla.eval_analysis` at the end of every run. If
you want to re-analyse an existing CSV (after fixing a metric or
plotting), run it standalone:

```bash
source submodules/MolmoBot/MolmoBot-Pi0/.venv/bin/activate
python -m pla.eval_analysis \
    --csv analysis_output/eval_medium_v1_<TS>/eval_metrics.csv \
    --out_dir analysis_output/eval_medium_v1_<TS>
```

Outputs (8 plots + 1 markdown report):

```
success_by_clutter.png            ← PLA vs VLM by clutter bin, Wilson 95% CI
contact_events_total.png          ← PRIMARY mechanism number (task_info.robot_contact)
pregrasp_approach.png             ← PRIMARY approach metric (TCP→pickup, pregrasp-only)
contacts_by_link.png              ← secondary diagnostic (per-link near-contact)
contacts_vs_clutter_scatter.png
episode_len_by_clutter.png
approach_delta_box.png            ← full-episode legacy
per_house_breakdown.png
paper_report.md                   ← paste-into-paper tables
```

## 11. Statistical Pipeline (TBD)

**Not built yet.** Belongs in `pla/eval_stats.py`. From the eval CSV,
compute:

1. **Wilson 95 % CI on success rate** per (model, clutter_bin). Already
   in `summary.json`.
2. **Bootstrap CI on contact-event mean** per (model, clutter_bin). 1000
   bootstrap resamples by episode.
3. **Permutation test** for "is PLA's success rate higher than VLM's?",
   stratified by clutter bin. 10,000 permutations.
4. **Cliff's δ effect size** on contact_events_total and
   pregrasp_min_tcp_pickup_m, PLA vs VLM and PLA vs Oracle.

Approximate ~1 h to build once the CSV exists. The deferred piece is in
TODO.md (task to be added). Suggested skeleton:

```python
# pla/eval_stats.py — sketch
import scipy.stats
def bootstrap_ci(arr, n_boot=1000, ci=0.95):
    samples = [np.random.choice(arr, len(arr), replace=True).mean() for _ in range(n_boot)]
    return float(np.percentile(samples, 100*(1-ci)/2)), float(np.percentile(samples, 100*(1+ci)/2))

def permutation_p(a, b, n_perm=10000):
    diff = a.mean() - b.mean()
    combined = np.concatenate([a, b])
    null = []
    for _ in range(n_perm):
        np.random.shuffle(combined)
        null.append(combined[:len(a)].mean() - combined[len(a):].mean())
    return float((np.abs(null) >= np.abs(diff)).mean())
```

## 12. Failure-mode Analysis (TBD)

**Not built yet.** Belongs in `pla/failure_analysis.py`. Operates
post-hoc on saved h5 rollout trajectories (under
`rollout_output/<run>/datagen_raw/<config>/<TS>/house_*/`).

Want to surface for each failed trajectory:

1. **Failure category**: (a) failed to grasp (no gripper-close phase
   reached), (b) grasped but dropped, (c) misplaced, (d) collision
   stop / sim early termination.
2. **Phase where failure occurred** — last `policy_phase` value before
   `success=False && fail=True`.
3. **Per-link min reading at the failure frame** — was the policy in a
   close-range proximity event when it failed?
4. **TCP→pickup distance trajectory** — did the EE overshoot, get stuck
   far away, or oscillate?

Approximate ~2 h to build. The failure clip selection (pulling
2-second video clips around the failure moment using the saved MP4s)
goes another ~1 h.

## 13. What each experiment must measure (CSV schema)

From the eval harness CSV (`analysis_output/.../eval_metrics.csv`):

**Identifiers**

| Column          | Type   | Meaning                                                     |
|-----------------|--------|--------------------------------------------------------------|
| `model`         | string | "pla", "vlm", "nomask", "eeonly", "oracle"                  |
| `scene_id`      | int    | house index                                                 |
| `clutter_bin`   | string | "low", "medium", "high" — from locked `clutter_bins.json`   |
| `seed`          | int    | 2028, 2029, ...                                             |
| `traj`          | string | trajectory key within the h5, e.g. `traj_0`                 |

**Outcomes**

| Column           | Type | Meaning                                                                  |
|------------------|------|----------------------------------------------------------------------------|
| `success`        | int  | 1 if `success[-1]=True` else 0                                            |
| `fail`           | int  | 1 if `fail[-1]=True` else 0                                               |
| `episode_len`    | int  | T (number of recorded timesteps)                                         |

**PRIMARY mechanism metrics** (independent of the policy's input)

| Column                     | Type | Meaning                                                                |
|----------------------------|------|-------------------------------------------------------------------------|
| `n_contact_events_total`   | int  | falling-edge entries into `task_info.robot_contact == True` (MuJoCo)   |
| `contact_frames_total`     | int  | raw count of `robot_contact == True` frames                            |

**PRIMARY approach metrics** (pregrasp-phase only; frame-corrected via robot_base_pose)

| Column                          | Type  | Meaning                                                                     |
|---------------------------------|-------|------------------------------------------------------------------------------|
| `pregrasp_frames`               | int   | # frames where `policy_phase == 2` (pregrasp)                                |
| `pregrasp_min_tcp_pickup_m`     | float | min ‖TCP−obj‖ during pregrasp (in base frame)                               |
| `pregrasp_final_tcp_pickup_m`   | float | ‖TCP−obj‖ at the last pregrasp frame                                        |
| `pregrasp_progress_m`           | float | first_pregrasp_distance − pregrasp_min (positive = made progress)          |
| `time_to_grasp`                 | int   | first frame index where `policy_phase == 3` (grasp); -1 if never reached    |
| `time_to_place`                 | int   | first frame index where `policy_phase == 7` (place); -1 if never reached    |

**SECONDARY / diagnostic metrics** (per-link proximity, do NOT use as paper headline)

| Column                          | Meaning                                                                     |
|---------------------------------|------------------------------------------------------------------------------|
| `n_contacts_link{3,5,6}`        | falling-edge entries into per-link min < 5 cm                              |
| `near_contact_frames_link{3,5,6}`| raw count of per-link min < 5 cm                                            |
| `min_prox_link{3,5,6}_m`        | min per-link reading over the trajectory                                    |

**Legacy / context**

| Column                  | Meaning                                                              |
|-------------------------|------------------------------------------------------------------------|
| `approach_delta_m`      | full-episode TCP→pickup change (biased by lift/return; keep for compat) |
| `tcp_to_pickup_end_m`   | ‖TCP−obj‖ at the final frame                                         |
| `clutter_signed`        | per-trajectory clutter proxy (higher = more clutter)                   |
| `task_description`      | language string (CSV-safe truncated)                                   |

### Time-to-grasp / time-to-place (now in CSV, 2026-05-13)

`time_to_grasp` and `time_to_place` columns are populated by
`metrics_from_traj` from `obs/extra/policy_phase`: the first index of
phase 3 (grasp) and phase 7 (place). `-1` if the phase was never
reached (failure case). Verified on planner holdout: 35/35 trajectories
reach phase 3 (median step 32), 34/35 reach phase 7 (median step 113).

### Per-link colliding body (per your spec)

`task_info.robot_contact` is a single bool, not per-link. To recover
per-link MuJoCo contacts you would need to either:

(a) Modify `molmo_spaces.data_generation.pipeline` to log `mjData.contact[i].geom1/geom2` and the body it maps to. ~30 LOC change in the submodule.
(b) Post-hoc replay: load the saved env_states and step MuJoCo one frame at a time to retrieve contacts. Slow but doesn't require modifying the submodule.

For tonight's eval, **stick with the global robot_contact + the
per-link proximity proxy**. Add proper per-link contact detection as a
follow-up if reviewers ask.

## 14. Quick-reference launch sequences

### Full reproduction from scratch (data + everything)

```bash
# 0. Setup
cd /home/jaydv/code/prox_learning
source submodules/MolmoBot/MolmoBot-Pi0/.venv/bin/activate
export PYTHONUNBUFFERED=1

# 1. Data collection (skip if assets/datagen/ already populated)
/opt/conda/envs/mlspaces/bin/python -m molmo_spaces.data_generation.main \
    FrankaSkinPickAndPlacePilotMediumConfig
/opt/conda/envs/mlspaces/bin/python -m molmo_spaces.data_generation.main \
    FrankaSkinPickAndPlacePilotEvalHoldoutConfig

# 2. Audit
python -m pla.audit_proximity \
    --data_root assets/datagen/pick_and_place_skin_pilot_medium_v1/FrankaSkinPickAndPlacePilotMediumConfig/<TS> \
    --out diagnostics_output/proximity_audit_medium_full

# 3. Lock clutter bins
python -m scripts.lock_clutter_bins \
    --out analysis_output/eval_medium_v1/clutter_bins.json

# 4. Training — 5 models (recommend: 4 sequential + 1 oracle in parallel)
# In terminal A:
bash scripts/launch_medium_ablations.sh
# In terminal B (in parallel):
python -m pla.oracle train --run_name medium_oracle_v1 \
    --data_root <DATA_ROOT> --num_steps 50000 --batch_size 32

# 5. Eval (after all 5 ckpts exist)
/opt/conda/envs/mlspaces/bin/python -m pla.eval_harness \
    --models pla=runs/medium_pla_v1/latest.pt vlm=runs/medium_vlm_v1/latest.pt \
             nomask=runs/medium_pla_no_mask_v1/latest.pt \
             eeonly=runs/medium_pla_ee_only_v1/latest.pt \
             oracle=runs/medium_oracle_v1/latest.pt:use_prox=false \
    --seeds 2028,2029,2030,2031,2032 \
    --house_inds 11,12,13,14,15,16,17,18,19,20 \
    --samples_per_house 2 \
    --clutter_bins_path analysis_output/eval_medium_v1/clutter_bins.json \
    --out_dir analysis_output/eval_medium_v1
```

### Eval-only when you already have checkpoints

```bash
# Single-model eval (debug / iterate)
/opt/conda/envs/mlspaces/bin/python -m pla.eval_harness \
    --models pla=runs/medium_pla_v1/latest.pt:use_prox=true \
    --seeds 2028 --house_inds 11,12 --samples_per_house 1 \
    --clutter_bins_path analysis_output/eval_medium_v1/clutter_bins.json \
    --out_dir /tmp/eval_smoke
```

### Re-analyse an existing CSV without re-rolling

```bash
python -m pla.eval_analysis \
    --csv analysis_output/<run>/eval_metrics.csv \
    --out_dir analysis_output/<run>
```

### Smoke test individual components

```bash
# Encoder mask invariance
python -m pla.proximity_encoder

# Full PLA policy forward pass
python -m pla.policy

# Oracle policy forward pass
python -m pla.oracle smoke

# Harness analysis-only on existing planner data (no rollouts)
python -c "
import sys; sys.path.insert(0, '.')
from pathlib import Path; import h5py
from pla.eval_harness import metrics_from_traj
root = Path('assets/datagen/pick_and_place_skin_pilot_eval_holdout_v1/FrankaSkinPickAndPlacePilotEvalHoldoutConfig/20260511_021228')
n = 0
for p in sorted(root.glob('house_*/trajectories_batch_*.h5')):
    with h5py.File(p, 'r') as f:
        n += sum(1 for k in f.keys() if k.startswith('traj_'))
print(f'n_traj={n}')
"
```

---

## Appendix — what was already done before this runbook

- 4 ablation training launched 2026-05-12 23:42 (`scripts/launch_medium_ablations.sh`).
- Model #1 `medium_pla_v1_20260512_234252` complete; checkpoints under `runs/`.
- Model #2 `medium_vlm_v1_20260512_234252` in progress (step ~34k / 50k).
- Models #3 `medium_pla_no_mask_v1_…` and #4 `medium_pla_ee_only_v1_…` queued.
- Oracle (`medium_oracle_v1`) **not yet running** — the code is built
  (`pla/oracle.py`); fire it in a second terminal per §8.
- `clutter_bins.json` locked from planner holdout.
- Section 3 draft of paper: `paper/section3_proximity_signal_draft.md`.
- Related-work scan: `analysis_output/related_work_scan.md`.
- Wandb project: https://wandb.ai/jayluvsgeography/pla
