# Hybrid obstacle — proximity-evidence activity gate, final decision

## Executive summary

The previous task's hypothesis was that the onset false activations came from a
proprioceptive or episode-onset prior. **Direct causal intervention on the frozen
checkpoint refutes that.** With real proprioceptive state and the proximity field replaced
by a clear reference, activity on the 17 known false-positive frames collapses from 0.9999
to **0.0229**. Shuffling state changes nothing (0.9999); replacing it with the batch mean
changes nothing (0.9999). State alone does not fire the gate. The onset attribution
classifies as **`PROXIMITY_AMBIGUITY_DOMINANT`**.

The structural audit is nonetheless damning about the old design: there is **no dedicated
activity head at all**. Proprioceptive state is summed into every sensor token before the
cross-sensor transformer, the shared decoder emits the mask-logit channel from those tokens,
and the "activity" value is a max-reduction of that same channel. State *can* reach the
activation decision, and the activity signal is entangled with the parked-field decoder by
construction. Replacing that path with something isolated was the right move regardless of
which input turned out to be at fault.

So a proximity-only gate was built to the fixed specification (90,241 parameters, well under
the 250,000 budget), trained exactly once on seed 0 over a fresh 40/8/8/8 nested episode
partition, with onset-zero frames weighted 4× and a trajectory-level onset-max penalty.

**No feasible threshold exists.** The calibration contract requires median active recall
≥ 0.80 *and* a bootstrap upper bound on mean trajectory FPR ≤ 0.02. On the calibration
split those are not jointly attainable anywhere:

| Threshold | Median recall | Mean traj. FPR | Max hazard-absent traj. FPR | Onset FP run |
|---:|---:|---:|---:|---:|
| 0.50 | 0.802 | **0.2471** | 0.6450 | 5 |
| 0.80 | 0.751 | **0.1805** | 0.5700 | 5 |
| 0.90 | 0.580 | 0.1297 | 0.3350 | 2 |
| 0.99 | **0.022** | 0.0109 | 0.0350 | 1 |

3,502 of 5,447 candidate thresholds clear the recall screen; **zero** are feasible.

The reason is not that the gate failed to train. It fits its own training split
(AUROC 0.9893) and then does not generalize, while the old shared head — which sees
proximity *and* state — is near-perfect on the very same held-out episodes:

| Split | Gate AUROC | Old head AUROC |
|---|---:|---:|
| gate_training | 0.9893 | 0.9982 |
| threshold_calibration (held out) | **0.7859** | **0.9998** |
| nested_offline_evaluation | 0.8734 | 0.9979 |

**Current proximity alone does not identify removable hazard evidence.** This is
`PROX_ACTIVITY_GATE_CALIBRATION_INFEASIBLE`, Case B. Per handoff step 12 the task stops
before live execution: **0 of the 20 permitted rollouts were run**, `development4` was not
executed, and `confirmatory41` remains untouched.

- Root branch `qualify/hybrid-obstacle-prox-activity-gate-v1`, from `bbfdb5e`
- ACT `91fc42a`, MolmoSpaces `678f2eb` — both untouched, clean working trees
- Decision JSON sha256 `2bc43347a5675efb55a373a39214b80a5c6f1c8969ddcf0a45a28589a53fdc8f`

## Starting and final commits

| Repo | Expected | Observed | Modified |
|---|---|---|---|
| root | `bbfdb5e` | `bbfdb5e` → branch `qualify/hybrid-obstacle-prox-activity-gate-v1` | code added only |
| ACT | `91fc42a` | `91fc42a` | no — no evaluator change was required |
| MolmoSpaces | `678f2eb` | `678f2eb` | no |

Remote `origin` → `Jdvakil/prox_learning`. Two pre-existing stashes untouched. Nothing
pushed. Runtime: Python 3.11.15, torch 2.7.1+cu126, numpy 2.4.6, CUDA 12.6, NVIDIA A10.

## Immutable provenance

**59 checks, 0 failed.** Every hash recomputed from disk.

Frozen seed-0 `CURRENT_FRAME_ONLY` checkpoint `47ae5cb1206c07d3…` matches, along with its
embedded config hash, variant, seed, best epoch (53) and parameter count (838,434);
`history_frames == 1` and `use_proximity == True` on rebuild. Input contract `cc87a773…`,
29-wide state, zero prohibited inputs. Paired dataset tree `1fb68b3c…`, manifest
`a88b94b3…`, partition `b06196b6…`, all 364 files still mode 444. SafetyHead `model.pt`
`1fb2fc2b…`, `meta.json` `7c873756…`, label_scale 11.359346389770508, 2560→7. ACT best
epoch 1738, `policy_best.ckpt` `dd7cd108…`, `dataset_stats.pkl` `c8119b90…`. Residual
constants gain 4.0, decay 2.2, EMA 0.75, max_dev 0.35, dt 66 ms, arm-only, gripper owned by
ACT. `offsamples` 4. development4 rows `[106, 107, 108, 118]`. confirmatory41: 41 rows,
`executed_in_this_task: false`.

Additionally pinned this task: the previous threshold manifest self-hash, its
`authorized_for_live: false` flag, and the magnitude-support bound
(`recalculated_in_this_task: false`, `post_hoc_clipping: false`).

## Frozen parked-field model

Untouched throughout. It was never retrained, no other seed was selected, no seeds were
ensembled. The gate trainer does not import `build_model` or `load_checkpoint` — a test
asserts that by parsing its AST, so the field model cannot be constructed there even by
accident. The causal audit confirms model weights were bitwise unchanged before and after
all eight interventions, and that no dataset file was written.

## Old activity-path trace

| Stage | Shape | State reaches | Proximity reaches |
|---|---|---|---|
| current_field_encoder | [320, 64, 8, 8] | no | yes |
| sensor_token_projection | [320, 192] | no | yes |
| **state_context_encoder** | [8, 192] | **yes** | no |
| cross_sensor_transformer | [8, 40, 192] | **yes** | yes |
| shared_per_sensor_decoder | [320, 2, 8, 8] | **yes** | yes |

Reachability was established by gradient tracing, not by reading the source: a tensor is
reachable from an input exactly when a gradient flows back to it.

- **Dedicated activity head exists: no.** Activity is `sigmoid(max per-pixel mask logit)`.
- **State reaches activity: yes.** The 29-D proprioceptive vector is added to every sensor
  token *before* attention.
- **Shares context-conditioned tokens with the parked-field decoder: yes** — the mask-logit
  channel and the magnitude channel come from the same decoder output, so activity and the
  parked field are siblings, not one derived from the other.
- **Gradients connected the activity objective to the state encoder during training: yes.**
  The focal BCE on `changed_pixel_mask` backpropagates through the shared decoder into the
  tokens, and the state embedding is summed into those tokens.

## Onset causal attribution

Eight interventions, five matched frame groups (17 / 450 / 452 / 2,651 / 12,812 frames),
frozen checkpoint only. `CURRENT_FIELD_IDENTITY_CONTROL` reproduces `FULL_INPUT` **exactly**
(max delta 0.0), so the harness itself perturbs nothing.

Group A — the 17 known false-positive frames:

| Intervention | Activity | Predicted ‖dq‖ | Change-mask fraction |
|---|---:|---:|---:|
| FULL_INPUT | 0.9999 | 0.1178 | 0.00391 |
| CURRENT_FIELD_IDENTITY_CONTROL | 0.9999 | 0.1178 | 0.00391 |
| STATE_SHUFFLED_WITHIN_ONSET | 0.9999 | 0.1164 | 0.00402 |
| STATE_MEAN | 0.9999 | 0.1179 | 0.00423 |
| STATE_SWAPPED_ACROSS_HAZARD_STRATA | 0.7316 | 0.0794 | 0.00324 |
| PROX_ONLY (state zeroed) | 0.5489 | 0.0571 | 0.00076 |
| PROX_SWAPPED_ACROSS_HAZARD_STRATA | 0.3848 | 0.1570 | 0.00170 |
| **STATE_ONLY** (proximity → clear reference) | **0.0229** | **0.0008** | **0.00000** |

Across groups, mean activity under FULL_INPUT / PROX_ONLY / STATE_ONLY: onset-zero
hazard-absent 0.0691 / 0.0375 / 0.0558; onset-active hazard-present 0.9985 / 0.7166 /
0.0190; later-active 0.9603 / 0.7565 / 0.2851; later-zero 0.0840 / 0.0665 / 0.1617.

**Classification: `PROXIMITY_AMBIGUITY_DOMINANT`.** The predeclared rule required a state
intervention to drop activity by ≥ 0.5 on a *majority* of the 17 frames; the best any
managed was PROX_ONLY at 8/17 (0.4706), and the two cleanest state manipulations —
shuffling and mean-replacement — moved nothing at all. Meanwhile clearing the proximity
field while keeping real state extinguishes the activation entirely.

**This refutes the previous task's stated hypothesis, and I am recording that plainly**
rather than quietly re-framing it: the recommendation to look for a state-driven onset prior
was wrong. The onset frames are ones where the *current proximity field itself* looks like
it contains removable structure.

One caveat on PROX_ONLY: a zeroed 29-D state vector is out of distribution for the state
encoder, so its 0.5489 mixes "state removed" with "state made nonsensical". STATE_MEAN and
STATE_SHUFFLED are the better-controlled probes and both leave activity at 0.9999, and
STATE_ONLY is cleanest of all. The conclusion does not rest on the PROX_ONLY number.

## Proximity-only gate

`PROX_EVIDENCE_ACTIVITY_GATE_V1`, built exactly to the fixed specification:

- per sensor: concat(closeness, valid mask) = 128 → Linear(128→64) → SiLU → Linear(64→64)
  → SiLU → add learned 64-D sensor embedding
- cross-sensor: TransformerEncoder, 2 layers, d_model 64, 4 heads, ff 128, pre-norm,
  dropout 0
- pooling: concat(mean, max) over the 40 tokens → 128
- activity head: Linear(128→64) → SiLU → Linear(64→1)

**90,241 parameters** against a 250,000 budget. The architecture was not changed after
training started.

Input isolation is structural, not procedural: `forward(self, closeness, valid_mask)` takes
no other argument, so there is no channel through which qpos, qvel, actions, gripper state,
the predicted parked field or differential, episode step, trajectory length, task phase,
hazard state or manifest identity could arrive. `assert_gate_inputs` additionally rejects
them by name for the deployment path, and `gate_feature_hash` hashes exactly what the gate
consumed, separately from the field model's features.

## Nested partition

`configs/hybrid_obstacle_prox_activity_partition_v1.json`, sha256
`daf97291596ac38abc4056236bffa92c32a81584cdbcb04203c5bf7b1d37466f`. Deterministic: episodes
sorted by identity within each hazard stratum and dealt in a fixed order.

| Split | Episodes | Hazard + | Hazard − | Trajectories |
|---|---:|---:|---:|---:|
| gate_training | 40 | 30 | 10 | 160 |
| checkpoint_validation | 8 | 6 | 2 | 32 |
| threshold_calibration | 8 | 6 | 2 | 32 |
| nested_offline_evaluation | 8 | 6 | 2 | 32 |

All 64 `reference_train` episodes, each used exactly once. The previous calibration,
validation and offline-test trajectories stay outside entirely as reused diagnostics.
`development4` and `confirmatory41` are excluded by construction — neither appears in the
paired dataset at all.

## Sampling and loss contract

Label: `activity_target = any changed_pixel_mask value is true`. Onset:
`episode_step < max(10, ceil(0.10 * trajectory_length))`.

Sampling, in the fixed order: distribution uniform → trajectory uniform within it →
activity class 50/50 → for zero frames, 50/50 onset vs non-onset → real frame uniform within
the group. No global frame-count weighting. Training pool: 27,168 frames (7,577 active,
2,490 onset-zero); checkpoint validation 5,449 frames.

Loss: BCE with fixed weights — active 1.0, ordinary zero 1.0, **onset zero 4.0** — plus a
trajectory-level onset penalty equal to the mean over represented trajectories of the
**maximum** predicted activity on their onset-zero frames, weighted 1.0. No dynamic positive
weight, no focal loss, no per-run manual adjustment.

The penalty is a maximum rather than a mean on purpose: the failure being targeted was seven
consecutive activations inside one trajectory, and a mean would score that identically to
seven isolated errors spread across seven trajectories — a far less dangerous thing. A test
asserts the clustered case costs strictly more.

Optimization exactly as specified: seed 0, AdamW, lr 3e-4, weight decay 1e-5, batch 256,
max 80 epochs (100 batches each), grad clip 1.0, dropout 0, checkpoint on minimum
trajectory-balanced BCE over the checkpoint-validation episodes. **Trained once.** Best
epoch 8, validation 0.4215; validation degrades monotonically afterwards, so the checkpoint
rule selected an early stop rather than the final weights.

## Calibration result — infeasible

Trajectory-wise, cluster bootstrap over episodes, 10,000 replicates, seed 20260727,
one-sided 95%.

5,447 candidate thresholds → 3,502 pass the recall ≥ 0.80 screen → **0 feasible**.

The binding conflict is recall against false-positive rate; the cosine conditions were never
the problem (median cosine 1.0000 and positive-cosine fraction 1.0000 at every threshold,
since those measure the *frozen field model* on retained frames).

Full sweep on the calibration split:

| τ | Median recall | Mean traj. FPR | Max hazard-absent FPR | Onset FP run |
|---:|---:|---:|---:|---:|
| 0.10 | 0.838 | 0.3441 | 0.7150 | 5 |
| 0.30 | 0.821 | 0.2812 | 0.6700 | 5 |
| 0.50 | 0.802 | 0.2471 | 0.6450 | 5 |
| 0.70 | 0.755 | 0.2059 | 0.6050 | 5 |
| 0.80 | 0.751 | 0.1805 | 0.5700 | 5 |
| 0.90 | 0.580 | 0.1297 | 0.3350 | 2 |
| 0.95 | 0.397 | 0.0798 | 0.2889 | 2 |
| 0.99 | 0.022 | 0.0109 | 0.0350 | 1 |
| 0.999 | 0.000 | 0.0000 | 0.0000 | 0 |

No row satisfies recall ≥ 0.80 and FPR ≤ 0.02 together; a test asserts this over the
recorded sweep.

### Is this underfitting or an information limit?

The distinction matters, so it was measured rather than asserted. The gate reaches AUROC
0.9893 on its own training split — it has the capacity to fit the label — and drops to
0.7859 on held-out calibration episodes. The old shared head scores 0.9998 on those same
held-out episodes. The gate is not underfitting; it is fitting something that does not
transfer, because **the current proximity field alone does not determine whether the
obstacle contributes removable evidence.**

That is coherent with the causal audit rather than contradicting it. The audit showed that
clearing proximity extinguishes activation, so proximity is *necessary*. It never showed
proximity is *sufficient*. The old head's near-perfect separation uses proximity in a
state-conditioned context; strip the context and the same field becomes ambiguous.

**Disclosure:** the AUROC figures above include the `nested_offline_evaluation` split. They
were computed *after* calibration had already failed and the task had terminated, purely to
separate underfitting from insufficiency. No threshold, model or split selection follows
them, and the nested split provided no gate for any decision. This is recorded in the
evaluation JSON as `post_termination_diagnostics_note`.

## Stages not reached

Because handoff step 12 requires stopping when no threshold is feasible, the following did
not run and **no results are claimed for any of them**: nested offline evaluation (step 13),
reused diagnostic audit (step 14), runtime integration and strict isolation (steps 15–16),
the 20 live development rollouts (steps 17–18), live approximation gates (step 19), live
gross-regression gates (step 20). There is consequently no 20-rollout outcome table, no live
onset/overall false-activation measurement, no candidate-118 negative control, and no
task/contact outcome data from this task.

The historical 17-frame onset regression check was likewise not run against a calibrated
gate — there is no calibrated gate to run it against.

## Case classification

**Case B.** The proximity-only gate cannot separate oracle activity offline.

Conclusion, as predeclared: current proximity alone does not identify removable hazard
evidence reliably. **Do not add another threshold or another activity MLP.** The next model
must use spatial uncertainty, or jointly model activity and parked-field credibility, rather
than treating activity as a separable classification problem over the instantaneous field.

Two specific observations for whoever picks this up:

1. The old head is *already* near-perfect at this discrimination (AUROC 0.998 held out). Its
   defect is 17 rare, clustered, onset-located false positives out of ~22,000 frames — not a
   separation failure. Replacing it wholesale with a weaker isolated classifier trades a
   rare failure for a common one.
2. The prohibition on state input is what makes the isolated gate weak. If a future design
   keeps state out of the activity path, it will need a different source of disambiguation —
   uncertainty over the predicted parked field is the obvious candidate, since the frames
   that fool the gate are exactly those where the field is compatible with both a hazard and
   a clear scene.

## Confirmatory41

Untouched and unexecuted: 41 rows, `executed_in_this_task: false`, manifest sha256
`7b4500e9b4b2868e2612d7e444c34762d72c5e6e7b4b7c38bcf31f027b51b69e`. No confirmatory row
appears in the development schedule and a test asserts the row sets are disjoint.

## Constraints honoured

Parked-field predictor not retrained; no other seed selected; no seeds ensembled. ACT and
Safety-CVAE untouched. Residual controller unchanged. Parked-field physical constraint and
magnitude-support bound unchanged. Temporal proximity history not reintroduced. Paired
dataset unmodified and still read-only. MSAA, cameras, robot, obstacle, planner, task,
horizon, collisions, temporal aggregation and success criteria unchanged. The gate received
no qpos, qvel, nominal action, gripper state, episode index, timestep, phase, hazard label or
manifest identity. No hand-coded first-N-frame suppression. Not a threshold-only change.
development4 and confirmatory41 not used for gate training. 0 of 20 permitted live rollouts
executed. Nothing pushed.

## Checkpoint

Gate weights held outside git (no approved artifact policy for binary weights here):

- path: `…/scratchpad/prox_gate/gate_best.pt`
- sha256: `50613f0914e48c0f88461fdb227b5988d413d9f08a6c932603450429bb28f7f9`
- epoch 8, 90,241 parameters, seed 0

## Changed files

Added: `causal_parked_skin/activity_gate.py`;
`scripts/hybrid_obstacle_prox_activity_{path_audit,onset_audit,partition,train,evaluate,decision}.py`;
`configs/hybrid_obstacle_prox_activity_partition_v1.json`;
`tests/test_prox_activity_gate.py`;
`diagnostics_output/hybrid_obstacle_prox_activity_gate/*.json`; this document. Modified:
`scripts/hybrid_obstacle_reference_threshold_provenance.py` (extended with the previous
threshold manifest and magnitude-bound checks). Nothing under `submodules/` was touched.

## Reproduction commands

```bash
SCR=<scratchpad>
PY=/root/act_retrain_venv/bin/python
D=diagnostics_output/hybrid_obstacle_prox_activity_gate

# 1. provenance -> CHECKPOINT_OR_SOURCE_MISMATCH on any failure
$PY scripts/hybrid_obstacle_reference_threshold_provenance.py \
    --out $D/provenance_verification.json

# 2. trace the frozen activity path
$PY scripts/hybrid_obstacle_prox_activity_path_audit.py \
    --cache $SCR/parked_skin_cache \
    --checkpoint $SCR/final/CURRENT_FRAME_ONLY__seed0/best.pt \
    --stack configs/hybrid_safety_stack_v1.json \
    --out $D/current_activity_path.json

# 3. causal onset attribution (frozen weights, eight interventions)
$PY scripts/hybrid_obstacle_prox_activity_onset_audit.py \
    --cache $SCR/parked_skin_cache \
    --checkpoint $SCR/final/CURRENT_FRAME_ONLY__seed0/best.pt \
    --stack configs/hybrid_safety_stack_v1.json --safety-dir assets/safety/cvae_v3 \
    --out $D/onset_attribution.json

# 4. freeze the nested partition
$PY scripts/hybrid_obstacle_prox_activity_partition.py \
    --manifest configs/hybrid_obstacle_parked_skin_supervision_v1.json \
    --dev4 configs/hybrid_obstacle_controller_development4_v1.json \
    --conf41 configs/hybrid_obstacle_confirmatory41_v1.json \
    --out configs/hybrid_obstacle_prox_activity_partition_v1.json

# 5. train the gate once, seed 0
$PY scripts/hybrid_obstacle_prox_activity_train.py \
    --cache $SCR/parked_skin_cache \
    --partition configs/hybrid_obstacle_prox_activity_partition_v1.json \
    --checkpoint-dir $SCR/prox_gate --out $D/gate_training.json

# 6. calibrate, then (conditionally) evaluate
$PY scripts/hybrid_obstacle_prox_activity_evaluate.py \
    --cache $SCR/parked_skin_cache --gate-checkpoint $SCR/prox_gate/gate_best.pt \
    --field-checkpoint $SCR/final/CURRENT_FRAME_ONLY__seed0/best.pt \
    --qpos-checkpoint $SCR/final/QPOS_ONLY__seed0/best.pt \
    --partition configs/hybrid_obstacle_prox_activity_partition_v1.json \
    --stack configs/hybrid_safety_stack_v1.json --safety-dir assets/safety/cvae_v3 \
    --onset-audit $D/onset_attribution.json --out $D/gate_evaluation.json

# 7. decision
$PY scripts/hybrid_obstacle_prox_activity_decision.py \
    --provenance $D/provenance_verification.json \
    --path-audit $D/current_activity_path.json --onset-audit $D/onset_attribution.json \
    --partition configs/hybrid_obstacle_prox_activity_partition_v1.json \
    --training $D/gate_training.json --evaluation $D/gate_evaluation.json \
    --out $D/final_decision.json

# verification
$PY -m pytest tests/test_prox_activity_gate.py -q
$PY -m ruff check causal_parked_skin/ scripts/hybrid_obstacle_prox_activity_*.py
```

## Next recommended task

Do not iterate on the threshold or on another isolated activity MLP; both have now been
shown insufficient, by different evidence.

1. **Model activity jointly with parked-field credibility.** The frames that defeat the
   isolated gate are those where the instantaneous field is consistent with both a hazard and
   a clear scene. A credibility or uncertainty output over the predicted parked field is the
   natural discriminator, and the frozen field model already produces a per-pixel change
   probability that could be extended to an explicit uncertainty without touching the field
   prediction itself.
2. **If state must stay out of the activity path, supply a substitute disambiguator.** The
   isolated gate is weak precisely because the prohibition removes the context the old head
   was using. Spatial structure across sensors, or agreement between the change mask and the
   magnitude channel, are candidates that remain within the proximity domain.
3. **Reconsider whether wholesale replacement is the right remedy at all.** The old head is
   at AUROC 0.998 held out with 17 clustered onset failures in ~22,000 frames. Targeting
   those 17 — for example by an onset-specific credibility check layered on top — may be a
   better-conditioned problem than rebuilding the activity decision from a weaker input set.

PROX_ACTIVITY_GATE_CALIBRATION_INFEASIBLE
