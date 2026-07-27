# Hybrid obstacle — parked-reference activation threshold, final decision

## Executive summary

A trajectory-aware threshold **was** found, and it is excellent on the data it was fitted
to: activity threshold **0.99960858**, cluster-bootstrap 95% upper bound on the mean
trajectory-level oracle-zero false-positive rate of **0.00000**, median active recall
0.882, median oracle cosine 1.000, zero false positives anywhere in the 48 calibration
trajectories. Frozen inference is bit-identical across 24 repeats.

It still fails, and it fails for a reason that no threshold can fix.

On the reused 20-episode diagnostic set, two predeclared blocking checks trip, and both
come from **the same trajectory**: episode `499eee89fb91`, `EXPERT_RECONSTRUCTED`,
**hazard-absent** — a scene where current and parked fields are bitwise identical and the
true differential is exactly 0.0 on every frame. The model fires for **7 consecutive
frames** (11.1% of that 63-frame trajectory) at activity probabilities of 0.999999,
0.999999, 0.999999, 0.999998, 0.999995, 0.999951, 0.999833, while predicting a differential
norm that climbs from 0.099 to 0.390 against a truth of zero.

Silencing that run requires a threshold above **0.99999905**, which would discard **59.1%
of all genuinely active frames**. The false activation is not separable from true activation
by any threshold. That is the finding.

Looking at where every false positive lands makes the mechanism plain. Across all three
partitions there are 17 false-positive frames at the selected threshold. **Sixteen of them
are in frames 0–6, and all 17 are within the first 10% of their trajectory. There are none
anywhere else.** This is a systematic episode-onset artifact: at the starting posture the
model is confidently wrong on scenes that are clear, and it recovers within a few frames.

Decision: **`REFERENCE_THRESHOLD_TRANSFER_FAILED`** — Case C. No live rollouts were run
(0 of the 20 permitted); the handoff requires stopping before live execution when these
checks trip. `confirmatory41` remains untouched and unexecuted.

## Why the historical `PARKED_REFERENCE_MODEL_OVERFIT` token is misleading

The prior rubric required that token, but it does not describe what happened, and carrying
it forward without correction would misdirect the next task. The evidence:

| Claim | Value |
|---|---|
| Offline-test MAE **below** validation MAE | 0.009087 vs 0.020782 |
| Improvement over trivial baseline | ~73% |
| Median oracle direction cosine | ~0.999 |
| Active recall | 94–98% |
| Only failed gate | seed-0 oracle-zero FPR **2.15%** vs a 2.00% ceiling |
| Calibration FPR | ~1% *by construction* |
| Threshold CV across seeds | ~0.72 |

A model that does better on test than on validation has no generalization gap to overfit
with. The blocker was **activation-threshold transfer instability**, which is what this task
set out to fix — and, having fixed the statistics properly, the residual failure turns out
to be in the activity signal rather than in the threshold.

## Starting and final commits

| Repo | Expected | Observed | Modified |
|---|---|---|---|
| root | `b2051ae` | `b2051ae` → branch `qualify/hybrid-obstacle-reference-threshold-v1` | code added only |
| ACT | `91fc42a` | `91fc42a` | no — no evaluator change was needed |
| MolmoSpaces | `678f2eb` | `678f2eb` | no |

Two pre-existing stashes on root left untouched. Nothing pushed. Runtime: Python 3.11.15,
torch 2.7.1+cu126, numpy 2.4.6, CUDA 12.6, NVIDIA A10.

## Immutable provenance

**49 checks, 0 failed.** Every hash was recomputed from disk rather than read from a config
field.

- Seed-0 `CURRENT_FRAME_ONLY` checkpoint `47ae5cb1206c07d3…` — matches, and the checkpoint's
  own embedded config hash, variant, seed, best epoch (53) and parameter count (838,434) all
  match the qualification record.
- Rebuilt architecture confirms `history_frames == 1` and `use_proximity == True` — the
  four-frame variant stays retired and this is not the state-only control.
- Input contract `cc87a773…`, 29-wide state, zero prohibited inputs.
- Physical counterfactual contract: `closeness = clip(1 - depth/0.5, 0, 1)`, dead pixels
  below 5 mm → 0; `0 ≤ parked ≤ current ≤ 1` with 0 recorded violations.
- Paired dataset tree `1fb68b3c…`, manifest `a88b94b3…`, partition `b06196b6…`; all 364
  files still mode 444, none writable.
- Sensor order 40 sensors, hash unchanged. SafetyHead `model.pt` `1fb2fc2b…`, `meta.json`
  `7c873756…`, `label_scale` 11.359346389770508, 2560→7.
- ACT best epoch **1738**, `policy_best.ckpt` `dd7cd108…`, `dataset_stats.pkl` `c8119b90…`.
- Residual constants gain 4.0, decay 2.2, EMA 0.75, max_dev 0.35 rad/joint, dt 66 ms,
  arm-only, gripper owned by ACT — none changed.
- `offsamples` 4. development4 rows exactly `[106, 107, 108, 118]`. confirmatory41: 41 rows,
  `executed_in_this_task: false`.

One check failed on first run and was a defect in the check, not the artifact: I compared
the closeness formula against a string missing its `closeness = ` prefix. Corrected; the
stack config was never modified.

## Frozen model and seed disposition

Seed 0 remains the production candidate **because it was the preselected model**. Seeds 1
and 2 recorded lower offline-test FPR (1.42% and 1.10% against seed 0's 2.15%), and that is
precisely why they were not chosen: selecting either now would be **post-hoc model selection
on an already-observed test result**. Their outputs were not averaged, their thresholds were
not reused, and their checkpoints do not appear anywhere in the calibration. A test asserts
that the non-zero-seed checkpoint paths appear nowhere in the calibration report.

## Retirement of the frame-level percentile threshold

The old threshold was 0.02819432708943556 — the frame-level 99th percentile of oracle-zero
predicted-differential norms over eight calibration trajectories. It is retired and not
reused.

The defect is statistical. Treating 3,821 frames as 3,821 independent observations badly
overstates the precision of that quantile: proximity fields at ~15 Hz are strongly
autocorrelated, and whole trajectories share a scene, a hazard pose and a driving policy.
The effective sample size is nearer the number of trajectories than the number of frames. Its
~1% calibration exceedance was therefore a construction artefact rather than an estimate,
which is exactly why it produced 2.15% on a different partition.

Two consequences shape everything below: metrics are computed **per trajectory and never
pooled first**, and uncertainty comes from a **cluster bootstrap over episodes**.

## `threshold_calibration16`

Manifest `configs/hybrid_obstacle_threshold_calibration16_v1.json`, sha256
`59a8570314f2c894c4d2d11fd9eb77a8992f794af42c6eb689c5fdf59d4e8d97`.

**16 episodes** — 8 from `reference_calibration` and 8 from `reference_validation` —
contributing **48 trajectories / 7,731 frames** (each episode appears in three source
distributions: expert, ACT-only, oracle).

The cluster unit is the **episode**, not the file. The same episode identity appears in all
three source distributions, so those three trajectories are not independent of one another;
resampling files would have quietly reintroduced the very independence assumption this
recalibration exists to remove.

The consumed `offline_reference_test` partition is excluded structurally: the fitting script's
`CALIBRATION_PARTITIONS` constant contains only the two calibration partitions, and a test
parses its AST to confirm the consumed set cannot enter the fit.

## Activity definition and gate rule

The frozen model has no frame-level activity head — it emits a per-pixel changed-probability
map. The frame activity probability is defined as the **maximum of that map**: the model's
confidence that at least one pixel is removable. This is a parameter-free reduction of an
output the frozen checkpoint already produces, so it involves no training, no new weights and
no architecture change. It was committed in source before any candidate was scored.

Gating on it rather than on the predicted differential norm is required by the handoff and is
also the right call: the norm conflates "something is there" with "the correction is large".
The manifest records `gates_on_differential_norm_alone: false`.

## Trajectory-level calibration and bootstrap

For each of 7,089 unique observed activity values, metrics were computed independently per
trajectory — active-frame count, zero-frame count, active recall, oracle-zero FPR, median
cosine on retained active frames, positive-cosine fraction, changed-pixel precision, maximum
consecutive false-positive run, and whether any activation persists after oracle activity
ends — preserving hazard stratum, policy distribution, episode identity and trajectory length.

**Bootstrap:** 10,000 replicates, committed seed **20260727**, resampling **whole episodes**
(all of an episode's trajectories move together), one-sided 95% upper confidence bound on the
mean trajectory-level oracle-zero FPR. The same index matrix is reused across all candidates
so the bounds are directly comparable and reproducible from the seed alone.

**Feasibility contract** (fixed before fitting, unchanged after): median active recall ≥ 0.80;
median active cosine ≥ 0.75; median positive-cosine fraction ≥ 0.85; bootstrap upper FPR bound
≤ 0.02; no hazard-absent trajectory FPR > 0.05; max consecutive false-positive run ≤ 2; no
persistent activation after the oracle returns to zero.

**Selection:** lexicographic — lowest bootstrap upper bound, then highest median active
recall, then highest threshold.

7,089 candidates → 504 passed the recall and hazard-absent screens → **138 feasible**.

### Selected threshold: `0.99960858`

| Metric | Value | Requirement |
|---|---:|---|
| Bootstrap 95% upper FPR bound | **0.00000** | ≤ 0.02 |
| Median trajectory active recall | 0.8819 | ≥ 0.80 |
| Mean trajectory zero FPR | 0.00000 | — |
| Median active cosine | 1.0000 | ≥ 0.75 |
| Median positive-cosine fraction | 1.0000 | ≥ 0.85 |
| Max hazard-absent trajectory FPR | 0.00000 | ≤ 0.05 |
| Max consecutive false-positive run | 0 | ≤ 2 |
| Trajectories with persistence | none | none |

Zero false positives across all 48 calibration trajectories, including all 12 hazard-absent
ones.

**The recall tail deserves stating, since the median hides it.** Of the 24 trajectories that
contain any oracle-active frames, recall ranges from 0.000 to 1.000. One trajectory (3 active
frames) never fires at all; four more sit at 0.258, 0.423, 0.481 and 0.569. The median of
0.882 clears the bar, but the model misses most active frames in a handful of trajectories.

## Consumed-test disposition

The 20-episode `offline_reference_test` partition was opened during the previous task, and
its 2.15% result is what motivated this one. It is therefore a **consumed diagnostic**, not an
untouched test. Every metric from it below is labelled `reused_nonconfirmatory_diagnostic`.
It was not used to fit the threshold and does not provide the final readiness gate. It can
block live testing — and it did — but passing it would not have been an independent
generalization claim. The next genuinely held-out check remains the live development4 run,
and after that `confirmatory41`.

### Results on the reused diagnostic set (20 episodes / 60 trajectories / 9,543 frames)

| Metric | Value | Blocking threshold |
|---|---:|---|
| Median trajectory active recall | 0.8963 | ≥ 0.70 ✓ |
| Mean trajectory zero FPR | 0.00297 | ≤ 0.03 ✓ |
| Bootstrap 95% upper zero FPR | 0.00684 | — |
| Median active cosine | 0.99995 | ≥ 0.70 ✓ |
| Median positive-cosine fraction | 1.0000 | — |
| Trajectories over 2% / 5% FPR | 2 / 1 | — |
| **Max hazard-absent active fraction** | **0.1111** | **≤ 0.10 ✗** |
| **Max consecutive false-positive run** | **7** | **≤ 5 ✗** |
| Constraint violations | 0 | 0 ✓ |
| Nonfinite outputs | 0 | 0 ✓ |

Note the shape of this: the **aggregate** numbers are excellent. Mean trajectory FPR is
0.30% and the cluster-bootstrap upper bound is 0.68%, both far inside the 2% target the old
threshold missed. Pooled across frames this model looks comfortably calibrated. Clustered by
trajectory, one episode produces a sustained false activation on a provably clear scene — the
concentration the frame-pooled view is structurally unable to show.

### The failing trajectory

`expert_reconstructed__499eee89fb91…`, hazard-absent, 63 frames, no oracle-active frames at
all:

| Frame | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Activity probability | 0.999999 | 0.999999 | 0.999999 | 0.999998 | 0.999995 | 0.999951 | 0.999833 |
| Predicted ‖dq‖ | 0.099 | 0.099 | 0.099 | 0.138 | 0.139 | 0.259 | 0.390 |
| True ‖dq‖ | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

The same episode's ACT-only and oracle trajectories each contribute one isolated false
positive as well, so the effect is a property of the episode rather than of the expert replay
alone.

### Why no threshold repairs this

Silencing the run needs a threshold above 0.99999905. Across the diagnostic set, genuinely
active frames have median activity 0.999992 and 10th percentile 0.991821, so that threshold
would drop **59.1%** of all true activations. The false activation sits *above* most true
activations in activity. It is not a threshold-placement problem.

### The mechanism: episode onset

At the selected threshold, across `offline_reference_test`, `reference_calibration` and
`reference_validation` combined, there are **17 false-positive frames in total**:

- **16 of 17 are at frame index 0–6**
- **17 of 17 are within the first 10% of their trajectory**
- **0 anywhere else**

Frame indices: `[0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 4, 4, 4, 5, 6, 9]`. By distribution:
expert 10, oracle 4, ACT-only 3; by hazard stratum: 9 absent, 8 present.

Every trajectory starts from the same home posture. In hazard-present episodes the obstacle is
present from frame 0, so the training distribution pairs "this starting posture" with "hazard
near". The model appears to have learned that association and applies it at onset regardless
of what the proximity field shows, recovering within a few frames once the arm moves and the
field becomes discriminative. That is a defect in the *activity model*, which is exactly what
Case C prescribes changing.

## Magnitude-support contract

Unchanged and not recalculated. The frozen construction is carried forward verbatim:

```
changed_probability = sigmoid(mask_logits)
delta_magnitude     = current_closeness * sigmoid(magnitude_logits)
predicted_delta     = changed_probability * delta_magnitude
predicted_parked    = current_closeness - predicted_delta
```

guaranteeing `0 ≤ predicted_parked ≤ current_closeness ≤ 1` by construction with no post-hoc
clipping. The new threshold changes **only the activation decision**: the predicted parked
field, predicted parked head, ungated differential, SafetyHead, correction dynamics and max
deviation are all untouched. The magnitude cap was not tuned from the consumed offline-test
set.

## Frozen inference stability

24 repeated inferences on a fixed 256-frame batch, no training kernels invoked:

| Quantity | Max absolute delta | Tolerance |
|---|---:|---|
| Activity probability | **0.0** (bit-identical) | ≤ 1e-7 |
| Parked field | **0.0** | ≤ 1e-7 |
| Predicted delta | **0.0** | ≤ 1e-7 |
| Head differential | **0.0** | ≤ 1e-7 |

Training was nondeterministic — the same configuration and seed moved validation MAE by 15%
between runs — so this was worth measuring rather than assuming. The frozen forward pass is
exactly reproducible.

## Live development: not executed

**0 of 20 permitted rollouts were run.** Handoff step 9 requires stopping before live
execution when the reused diagnostic set shows a hazard-absent trajectory above 10% active
frames or a false-active run longer than 5 frames. Both tripped.

The threshold manifest `configs/hybrid_obstacle_reference_threshold_v1.json`
(sha256 `12f7267c6c4c1a1f55ff5ebbd03b8c1101e987dc6574d070fe3ef2f47a2d4a29`) was still written —
it is the record of exactly what was fitted, and the next task should not have to re-derive
it — but it carries `authorized_for_live: false` with the blocking reason recorded, and the
strict loader **refuses to hand it to an evaluator** while that flag is false. A manifest that
looked ready would have been worse than no manifest.

The manifest pins: seed-0 checkpoint hash, model config hash, feature-contract hash, selected
threshold, calibration16 membership and hash, bootstrap seed and replicate count, bootstrap
upper bound, active-recall result, magnitude-support construction, SafetyHead hashes, ACT
checkpoint and statistics hashes, controller constants, sensor-order hash, source commits and
runtime versions. It also carries the frozen 28-field per-frame live log schema, with
`minimum_clearance_m` explicitly excluded — `mj_geomDistance` returns exactly 0.0 for
`robot_0/fr3_link7_collision`, pinning that metric at ≤ 0 regardless of true geometry.

Live approximation gates, gross-regression gates and the candidate-118 negative control were
therefore **not evaluated**. No task-success, contact or controller-saturation results exist
from this task, and none are claimed.

## Case classification

**Case C.** Trajectory-aware calibration succeeded offline — a feasible threshold exists with
a zero bootstrap upper bound on its fitting set — but false activation remains on held-back
data. The handoff's Case C conclusion applies with one clarification: the blocking evidence
here is **open-loop**, gathered before any closed-loop rollout, so "unreliable under
closed-loop deployment" is stronger than what was shown. What is shown is that the threshold
does not transfer across episodes even offline.

The prescription is unchanged and, on this evidence, well supported: **do not execute
confirmatory41, and do not retrain the same model merely to move the threshold.** The next
change must affect the activity model or the training objective.

## Confirmatory41

Untouched and unexecuted: 41 rows, `executed_in_this_task: false`, manifest sha256
`7b4500e9b4b2868e2612d7e444c34762d72c5e6e7b4b7c38bcf31f027b51b69e`. No confirmatory row
appears in the development schedule, and a test asserts the two row sets are disjoint.

## Constraints honoured

No model trained or fine-tuned. No seed reselected on observed test performance. ACT and the
Safety-CVAE untouched and unmodified. Residual-controller constants unchanged. Architecture
and inputs unchanged; causal history not reintroduced. Paired dataset unmodified and still
read-only. MSAA/`offsamples`, cameras, environment, robot, obstacle, task, horizon, collisions,
temporal aggregation and success criteria all unchanged. The 2% false-activation target was not
loosened. The prior offline-test set was not used to fit the threshold. No confirmatory41 row
executed. 0 of 20 permitted new rollouts run. Nothing pushed.

## Changed files

Added: `causal_parked_skin/threshold.py`; `scripts/hybrid_obstacle_reference_threshold_{provenance,calibrate,audit,manifest,decision}.py`;
`configs/hybrid_obstacle_threshold_calibration16_v1.json`;
`configs/hybrid_obstacle_reference_threshold_v1.json`;
`tests/test_reference_threshold_qualification.py` (44 tests);
`diagnostics_output/hybrid_obstacle_reference_threshold/*.json`; this document. Nothing under
`submodules/` was touched — no ACT evaluator change was required, because no live rollout was
authorized.

## Reproduction commands

```bash
SCR=<scratchpad>            # cache and checkpoints live outside the repo
PY=/root/act_retrain_venv/bin/python
D=diagnostics_output/hybrid_obstacle_reference_threshold

# 1. immutable provenance -> CHECKPOINT_OR_SOURCE_MISMATCH on any failure
$PY scripts/hybrid_obstacle_reference_threshold_provenance.py \
    --out $D/provenance_verification.json

# 2. build calibration16 and fit the trajectory-aware threshold
$PY scripts/hybrid_obstacle_reference_threshold_calibrate.py \
    --cache $SCR/parked_skin_cache \
    --checkpoint $SCR/final/CURRENT_FRAME_ONLY__seed0/best.pt \
    --stack configs/hybrid_safety_stack_v1.json --safety-dir assets/safety/cvae_v3 \
    --calibration-manifest configs/hybrid_obstacle_threshold_calibration16_v1.json \
    --out $D/threshold_calibration.json

# 3. reused-diagnostic audit + frozen inference repeatability
$PY scripts/hybrid_obstacle_reference_threshold_audit.py \
    --cache $SCR/parked_skin_cache \
    --checkpoint $SCR/final/CURRENT_FRAME_ONLY__seed0/best.pt \
    --stack configs/hybrid_safety_stack_v1.json --safety-dir assets/safety/cvae_v3 \
    --calibration $D/threshold_calibration.json --out $D/threshold_audit.json

# 4. freeze the manifest (authorized_for_live is derived, not asserted)
$PY scripts/hybrid_obstacle_reference_threshold_manifest.py \
    --provenance $D/provenance_verification.json \
    --calibration $D/threshold_calibration.json --audit $D/threshold_audit.json \
    --calibration-manifest configs/hybrid_obstacle_threshold_calibration16_v1.json \
    --stack configs/hybrid_safety_stack_v1.json --safety-dir assets/safety/cvae_v3 \
    --out configs/hybrid_obstacle_reference_threshold_v1.json

# 5. decision
$PY scripts/hybrid_obstacle_reference_threshold_decision.py \
    --provenance $D/provenance_verification.json \
    --calibration $D/threshold_calibration.json --audit $D/threshold_audit.json \
    --calibration-manifest configs/hybrid_obstacle_threshold_calibration16_v1.json \
    --threshold-manifest configs/hybrid_obstacle_reference_threshold_v1.json \
    --out $D/final_decision.json

# verification
$PY -m pytest tests/test_reference_threshold_qualification.py -q
$PY -m ruff check causal_parked_skin/ scripts/hybrid_obstacle_reference_threshold_*.py
```

## Next recommended task

The evidence points at one specific, testable defect rather than a general shortfall.

1. **Fix the episode-onset failure in the activity signal, not the threshold.** Every false
   positive is in the first ~7 frames. The cheapest honest probe is to check whether activity
   at onset is driven by the 29-D proprioceptive state embedding rather than the proximity
   field — the state is identical across hazard-present and hazard-absent episodes at frame 0,
   so a state-driven prior would produce exactly this signature. If confirmed, the fix is in
   the training objective: penalise activity on hazard-absent frames specifically at onset, or
   withhold state from the activity path.
2. **Keep the trajectory-aware calibration machinery.** It is correct and it is what surfaced
   this; the frame-pooled view rated the same model at 0.30% mean FPR and would have sent it
   to a live run.
3. **Do not run confirmatory41**, and do not retrain the identical model hoping for a kinder
   threshold. The activity signal cannot separate this failure mode at any operating point.

REFERENCE_THRESHOLD_TRANSFER_FAILED
