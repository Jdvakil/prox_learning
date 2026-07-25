# Hybrid obstacle — deployable posture-conditioned reference: final decision

Date: 2026-07-25
Task: train and qualify a deployable reference that approximates the validated
parked-obstacle oracle **without privileged simulation information**.
Scope: this task does **not** execute `confirmatory41`.

---

## 1. Executive summary

The reference trains cleanly and passes every predeclared **offline** gate on held-out
trajectories. It then fails **live**, in a way the offline evidence did not predict.

**Offline, on 20 held-out trajectories:**

* the posture-only KNN baseline answers its question decisively — **posture alone does not
  explain the parked head**: median oracle cosine **−0.39**, differential MAE 0.946, worse
  than both the raw head (0.660) and first-live skin (0.657);
* the posture + skin MLP passes all four gates — median oracle cosine **+0.977**,
  91.7 % positive, analytic-teacher cosine +0.922, differential MAE **0.283**, hazard-absent
  false activation 0.52 %.

**Live, over 20 rollouts on the four development rows:**

* **all 12 technical gates pass** — 20/20 finalised, no privileged feature reached the
  model, the shadow oracle stayed state-neutral on every one of ~4 000 counterfactuals,
  the gripper was bitwise unchanged, the residual stayed arm-only after temporal
  aggregation;
* **oracle approximation collapses**: pooled median cosine **0.345** against the 0.70 gate,
  versus 0.977 offline;
* **magnitude is badly over-predicted** — on candidate 106 the deployable differential is
  **5.1–6.9×** the true oracle differential;
* **the controller fires when the true signal is exactly zero** — 23–34 % of shadow-zero
  frames on candidate 106, against a 2 % gate;
* **candidate 118, where the true differential is provably zero on every frame, gained
  environment contact in all five repeats** (12–35 contacts) where ACT-only had none.

Task success did **not** regress: pooled hazard-present **12/15**, above ACT-only's 11/15
and below the oracle's 14/15, with no row falling to 0/5. The regression is in unnecessary
motion and in new contact on the hazard-absent row.

**Root cause: distribution shift.** The reference is trained on expert planner
trajectories, which deflect around the hazard. On all four development expert trajectories
the predicted differential norm **never once reached `tau`** — the gate opened zero times.
Live ACT trajectories visit states the expert never does and drive oracle differentials up
to 2.5, and there the model extrapolates: right sign more often than not, wrong direction
in detail, several times too large.

**Decision: `DEPLOYABLE_REFERENCE_LIVE_GROSS_REGRESSION`.**

---

## 2. Starting and final commits

| Repo | Branch | Start | Final |
|---|---|---|---|
| root | `develop/hybrid-obstacle-deployable-reference-v1` | `4193b77` | see §16 |
| ACT | `develop/hybrid-obstacle-posture-reference-v1` | `709a22d` | see §16 |
| MolmoSpaces | `repair/hybrid-obstacle-manifest-runner-v2` | `678f2eb` | `678f2eb`, **unmodified** |

No commit was made on the oracle branches. Nothing was pushed.

---

## 3. Immutable artifact verification

**50 checks, 0 failures** (`provenance_verification.json`). Expected digests were recovered
from the oracle, raw-head and ACT-baseline decision artifacts.

| Artifact | SHA-256 |
|---|---|
| `policy_best.ckpt` (epoch 1738) | `dd7cd108a64ce10e5aab21b525dc06190f54d4e5fe446f65715b6852c49e7d36` |
| `dataset_stats.pkl` | `c8119b904bfc80d66e3d33825722fcf9bb8bf3433c956dc09c27e6517d7c4ae2` |
| Safety-CVAE `model.pt` | `1fb2fc2b6023e64d2b9cbcf67fd5a24402968ec6f902c1e8a8595690396e7405` |
| `model_hybrid.xml` | `50924661e0411f92ab529c790512b17b674e789434c592c3dbc6d2359164d4c6` |
| 40-sensor order | `c31df8c36b0011b0eaf5b2eb5ce66d2514b5d6662ba9d7684ff021cd17cec858` |
| 80/20 split manifest | `f7c2b22718f1697ea153926220a48bac1ab5876f6119d863317117d04474ccd0` |
| `confirmatory41` manifest | `7b4500e9b4b2868e2612d7e444c34762d72c5e6e7b4b7c38bcf31f027b51b69e` |

The oracle decision's own self-hash re-derives, `mj_forward_called` is `False` in the
implementation this task reuses, and the frozen controller constants are unchanged at
(4.0, 2.2, 0.75, 0.35). **All four partitions are pairwise disjoint** — reference-training
(80), reference-validation (20), live development (4), confirmatory (41) — which is what
lets the reference use the canonical split while `confirmatory41` stays independent.

---

## 4. Why long-range direction is out of teacher support

The committed analytic teacher fires only when a hazard return is a sensor's **closest**
return inside `D_act = 0.18 m`; outside that radius the repulsion weight `1/r − 1/D_act`
is undefined, and the oracle task measured a **reversed** median cosine (−0.36) on
candidate 108 at the head's own 0.5 m input radius. Direction is therefore neither
optimised nor scored outside the supported range; the residual is required to be *quiet*
there instead.

**Two measured facts make the support gate weak in this environment**, and both matter for
reading §9 and §11:

* **no validation frame has a minimum valid depth above 0.187 m.** The "far frame > 0.25 m"
  metric therefore has **zero frames** and its gate is vacuous;
* gate A (a valid return inside 0.18 m) is open on **1168 of 1543** validation frames
  (75.7 %). The quiet threshold `tau` does essentially all the gating work.

---

## 5. Paired oracle-reference dataset

Built by reusing the validated per-frame parked-obstacle counterfactual verbatim
(`PerFrameParkedObstacleReference`): both halves rendered at the **same decision state**,
no dynamics-advancing operation, no `mj_forward`, render state restored directly. State
reconstruction is deterministic open-loop replay of the recorded expert joint commands from
the verified initial state, because the trajectory H5 stores no per-step pose for the
pickup object.

| | trajectories | frames | support frames | teacher-active | oracle-nonzero |
|---|---|---|---|---|---|
| train | 80 | 6 450 | 4 742 | 453 | 1 418 |
| validation | 20 | 1 543 | 1 168 | 133 | 322 |

Tree SHA-256 `9e3a66ece0fbcf997bc3597bb3542ff871e99092211d6311b00d10947a2de6ce`.

**Pairing correctness, checked rather than asserted:**

* **0 state-neutrality failures** across all 7 993 frames;
* on **every** hazard-absent trajectory, `current_head` and `parked_head` are **bitwise
  equal** — parking is a no-op there by construction;
* privileged fields (hazard presence, obstacle pose, minimum hazard return) live in a
  separate `privileged_` namespace that no feature builder reads.

Split is by **trajectory**, never by frame. Train/validation trajectory lists are recorded
in the selection report and the final decision JSON.

One infrastructure note: shard 3 of the 5-way parallel build was killed mid-trajectory
without a traceback under GPU contention. It was resumed with `--skip-existing`; generation
is deterministic given the manifest row, so the resumed shard produced the files a fresh
one would.

---

## 6. Candidates

**Candidate A — `POSTURE_KNN_REFERENCE_V1`.** Runtime inputs: qpos(9), qvel(9), gripper
state(2) = 20 features. Deterministic k = 8 distance-weighted lookup of `parked_head` over
the 6 450 training frames, ties broken by `(distance, trajectory, timestep)` lexsort.
Purpose: an interpretable fixed-environment baseline that answers whether posture alone
explains the parked head.

**Candidate B — `POSTURE_SKIN_MLP_REFERENCE_V1`.** Runtime inputs: qpos(9), qvel(9),
nominal ACT action(8), gripper state(2), gripper command(1), current SafetyHead output(7),
and a per-sensor causal proximity summary (40 × 4: minimum valid depth, tenth-percentile
valid depth, mean closeness, fraction of pixels below 0.18 m) = **196 features**.
Architecture `196→256 SiLU →256 SiLU →128 SiLU →7`, **150 023 parameters** (budget
250 000), no dropout, no batch norm. Fixed training: seed 0, AdamW, lr 1e-3, weight decay
1e-5, batch 256, 150 epochs, SmoothL1 against `parked_head`, minimum-validation checkpoint.
Best epoch 141, validation loss 0.021257. Strict reload reproduces predictions bitwise.

Both predict the **parked head**, never the executed action. All normalization statistics
come from the 80 training trajectories only. No architecture search, no sweep, no
retraining after live results.

---

## 7. Model selection

The predeclared lexicographic rule, evaluated on the fixed 20-trajectory validation split:

| gate | KNN | MLP |
|---|---|---|
| 1 — oracle cosine ≥ 0.70, ≥ 80 % positive (133 teacher-active frames) | **FAIL** −0.390, 30.8 % | **PASS** +0.977, 91.7 % |
| 2 — analytic-teacher cosine ≥ 0.60, ≥ 75 % positive | **FAIL** −0.358, 38.3 % | **PASS** +0.922, 98.5 % |
| 3 — hazard-absent FA ≤ 1 %, far-frame FA ≤ 2 % | PASS 0.52 %, n/a | PASS 0.52 %, n/a |
| 4 — active-frame differential MAE below every baseline | **FAIL** 0.946 | **PASS** 0.283 |

Baselines: raw SafetyHead MAE **0.660**, first-live skin MAE **0.657**.

**Selected: `POSTURE_SKIN_MLP_REFERENCE_V1`.** Only one candidate passed, so no tie-break
was needed.

The KNN result is worth stating plainly: with posture-only inputs the predicted differential
points *away* from the oracle more often than towards it, and is worse than doing nothing
clever at all. Skin information is doing the work.

---

## 8. Frozen model

| field | value |
|---|---|
| reference type | `POSTURE_SKIN_MLP_REFERENCE_V1` |
| artifact SHA-256 | `see deployment_manifest.json` |
| input-statistics SHA-256 | from the 80 training trajectories only |
| **tau** | **1.4161068201065063** |
| tau rule | 99.5th percentile of predicted-differential norm on hazard-absent validation frames, frozen before live execution |
| `D_act` | 0.18 m |
| runtime inputs | qpos, qvel, nominal action, gripper state, gripper command, current head, sensor summary |
| privileged inputs | **none** |

The loader refuses a wrong artifact hash, wrong input statistics, wrong feature width, wrong
sensor order, wrong ACT checkpoint, wrong Safety-CVAE, wrong `offsamples`, wrong controller
constants, or any field outside the runtime whitelist.

**A caveat on gate 3 that belongs here.** `tau` is the 99.5th percentile of the
hazard-absent validation norm, so roughly 0.5 % exceedance on those frames is guaranteed by
*construction*. The observed 0.52 % is a calibration, not an independent measurement, and
gate 3 is correspondingly weak.

---

## 9. Offline development replay

The frozen reference and gate were run across the four development expert trajectories with
the complete frozen residual dynamics, open loop. All four gates passed — and the headline
is *why*:

| cand | hazard | frames | activations | predicted norm max | cos (teacher-active) |
|---|---|---|---|---|---|
| 106 | present | 108 | **0** | 0.767 | — (0 active) |
| 107 | present | 109 | **0** | 0.884 | +0.817 (n = 31) |
| 108 | present | 62 | **0** | 0.538 | — (0 active) |
| 118 | absent | 59 | **0** | 1.362 | — |

**The gate never opened.** Every predicted norm sat below `tau = 1.416`, so the offline
replay gates passed vacuously on everything except direction. This is the first visible
symptom of the distribution problem: the expert planner deflects around the hazard, so its
trajectories never produce a differential large enough to act on. Candidate 118's 1.362 —
on a row whose true differential is identically zero — was already within 4 % of `tau`.

---

## 10. The 20-rollout schedule

`configs/hybrid_obstacle_deployable_schedule_v1.json`, sha256 `a7980e11…`, frozen before
execution. Four development rows × `ACT_PLUS_DEPLOYABLE_REFERENCE` × 5 repeats = **20**
rollouts, budget 20/20. The 20 ACT_ONLY and 20 ORACLE rollouts were **reused after hash and
schema verification** (40/40 verified); **zero reruns**. Each rollout ran in a fresh
process, reconstructed the manifest row and accepted retry, verified the initial-state hash
and `offsamples = 4`, and reset ACT aggregation, the reference's causal state and the
residual controller.

---

## 11. ACT-only vs oracle vs deployable

| cand | hazard | ACT-only | oracle | **deployable** | activation % | hazard-bar (dep) | other-env (AO → dep) |
|---|---|---|---|---|---|---|---|
| 106 | present | 5/5 | 5/5 | **4/5** | 21–29 | `[15,0,0,0,0]` | 55–68 → 32–64 |
| 107 | present | 5/5 | 5/5 | **5/5** | 33–41 | `[0,0,0,0,0]` | 75–93 → **5–28** |
| 108 | present | 1/5 | 4/5 | **3/5** | 0–1.5 | `[0,0,0,0,0]` | 0 → `[0,10,0,0,0]` |
| 118 | absent | 0/5 | 0/5 | 0/5 | 1.5–2.0 | `[0,0,0,0,0]` | **0 → 12–35, all five** |

Pooled hazard-present: **ACT-only 11/15 · oracle 14/15 · deployable 12/15**.

### Technical gates — 12 / 12 pass

20/20 finalised · no privileged feature in the model · 0 shadow-oracle state-neutrality
failures · no nonfinite output or action · residual after temporal aggregation · arm-only ·
gripper bitwise unchanged · state reset between repeats · MSAA and camera contract unchanged
· initial state replayed exactly · 40/40 frozen baselines verified · `confirmatory41`
unexecuted.

### Live approximation gates — 4 / 9 pass

| gate | result |
|---|---|
| median deployable/oracle cosine ≥ 0.70 | **FAIL** — pooled median **0.345** |
| positive cosine fraction ≥ 80 % | PASS — 0.872 |
| norm ratio in [0.5, 1.5] | **FAIL** — pooled median 1.559; **5.1–6.9 on candidate 106** |
| activation rate when shadow oracle is zero ≤ 2 % | **FAIL** — 23–34 % on 106, 7–11 % on 107 |
| no persistent correction after support clears | **FAIL** — `cand107_deployable_r4` |
| candidate 118 activation ≤ 2 % | PASS — 1.5–2.0 % |
| candidate 118 no run longer than 2 frames | **FAIL** — runs `[2,4,3,3,3]` |
| candidate 118 no universal new hazard-bar collision | PASS |
| candidate 118 median correction ≤ 10 % of hazard-present | PASS (**vacuously** — both medians are exactly 0.0) |

### Gross-regression gates — 7 / 8 pass

Candidate 106 not 0/5 · candidate 107 not 0/5 · pooled ≥ 8/15 (12) · pooled within 20 pp of
ACT-only (**−6.7 pp**, i.e. better) · no new universal hazard-bar collision · median
saturation 0.000 · no rollout saturated over 75 % — all pass.

**FAIL — candidate 118 systematic new environment collision**: `[13, 12, 35, 20, 21]`
contacts across all five repeats against ACT-only's `[0,0,0,0,0]`.

---

## 12. Candidate 108, inside and outside teacher support

**Supported range** (valid return < 0.18 m and a nonzero shadow oracle): the row was almost
entirely inactive — 3 of 5 rollouts had **zero** activations and an identically zero
correction, so their control law was ACT-only's. Where the gate did open (r0, r2) the
supported-range cosine was +0.849 (n = 4) and +0.009 (n = 9). Task success 3/5, between
ACT-only's 1/5 and the oracle's 4/5, on the row predeclared *unstable*.

**Unsupported range**: false activation rate **0.000 in all five rollouts** — the support
gate closed the residual to zero on every far frame. The oracle task's reversed long-range
cosine (−0.36 at 0.5 m) was therefore never executed. **Support gating did its job here.**

---

## 13. Candidate 118 negative control

The row where the true parked differential is provably **zero on every frame**.

| repeat | activations / 200 | longest run | max correction | other-env contacts |
|---|---|---|---|---|
| r0 | 3 | 2 | 0.079 | 13 |
| r1 | 4 | **4** | 0.117 | 12 |
| r2 | 3 | **3** | 0.116 | 35 |
| r3 | 3 | **3** | 0.093 | 20 |
| r4 | 3 | **3** | 0.081 | 21 |

ACT-only: `[0,0,0,0,0]`. Oracle (correction provably exactly zero): `[31,0,0,0,0]`.

**Attribution, stated honestly.** The oracle condition proves that MSAA-driven rollout
variation alone can produce environment contacts on this row — it produced 31 in one repeat
with a provably zero correction. What the deployable adds is that **all five** repeats show
them while a nonzero correction of 0.08–0.12 rad is present. That is suggestive of
causation and is what the predeclared gate tests, but at n = 5, with a known
stochastic-contact channel, it is not proof.

---

## 14. Privileged shadow-oracle diagnostics

The validated parked-obstacle counterfactual ran alongside every deployable rollout as a
**non-executed** diagnostic: privileged, never touching the executed action, the
controller, the environment or any RNG. Over ~4 000 counterfactual renders there were
**0 state-neutrality failures**. It is the instrument that made §11's approximation gates
measurable at all — without it, the live cosine collapse would have been invisible.

---

## 15. Interpretation

**What worked.** The dataset is exact. The KNN control answers its question decisively.
The MLP genuinely learns the parked head on the training distribution — parked-head MAE
0.097, cancellation error 0.075 where the oracle is zero, and a median oracle cosine of
0.977 on held-out *trajectories*, not held-out frames. All the plumbing gates pass.

**What failed.** Live, the same model's cosine falls to 0.345 and its magnitude
over-predicts by up to 6.9×, and it fires on a quarter to a third of frames where the true
differential is exactly zero.

**Why.** The training distribution is the expert planner's, which deflects around the
hazard; the deployment distribution is ACT's, which does not. On all four development
expert trajectories the predicted norm never reached `tau` at all — the model was never
asked, offline, to produce a large differential. Live it is asked constantly, and
extrapolates: the sign is right more often than not (positive-cosine fraction 0.872 still
passes) but the direction is wrong in detail and the magnitude is far too large.

**Not a task regression.** 12/15 versus ACT-only's 11/15, no row at 0/5. And on candidate
107 the deployable reference cut other-environment contacts from 75–93 per rollout to 5–28,
better than the oracle — the mechanism is real even where the approximation is poor.

The predeclared gross-regression instrument nevertheless records one failure — new
environment contact in all five hazard-absent repeats — and the approximation instrument
records five. That is a live regression in unnecessary motion and contact, which is the
`DEPLOYABLE_REFERENCE_LIVE_GROSS_REGRESSION` case.

---

## 16. Changed files and commits

**ACT** (`develop/hybrid-obstacle-posture-reference-v1`), new files only:

* `deployable_reference.py` — both candidates, the runtime whitelist, the support gate, the
  loader and the runtime-contract check
* `eval_act_obstacle_deployable.py` — the `ACT_PLUS_DEPLOYABLE_REFERENCE` condition and the
  non-executed shadow oracle

**Root** (`develop/hybrid-obstacle-deployable-reference-v1`):

* `configs/hybrid_obstacle_deployable_schedule_v1.json`
* `scripts/hybrid_obstacle_deployable_provenance.py`
* `scripts/hybrid_obstacle_build_paired_reference_dataset.py`
* `scripts/hybrid_obstacle_paired_dataset_manifest.py`
* `scripts/hybrid_obstacle_train_deployable_reference.py`
* `scripts/hybrid_obstacle_deployable_offline_replay.py`
* `scripts/hybrid_obstacle_deployable_schedule.py`
* `scripts/hybrid_obstacle_deployable_analysis.py`
* `scripts/hybrid_obstacle_deployable_write_decision.py`
* `tests/test_deployable_reference_contract.py` — 51 tests
* `diagnostics_output/hybrid_obstacle_deployable_reference/*.json`
* `docs/HYBRID_OBSTACLE_DEPLOYABLE_REFERENCE_FINAL_DECISION.md`

**Not committed** (paths and hashes recorded in `final_decision.json`): the generated paired
frame data, the reference-model checkpoint, live rollout H5s and `rollout.json` payloads,
the ACT checkpoint, Safety-CVAE weights, canonical source and converted datasets. There is
no approved artifact policy in this repo for multi-hundred-megabyte binaries.

MolmoSpaces is unmodified. Nothing was pushed.

---

## 17. Reproduction

```bash
export MUJOCO_GL=egl PYTHONUNBUFFERED=1
export MLSPACES_ASSETS_DIR=/root/prox_learning_hybrid_safety/assets
export PYTHONPATH=/root/prox_learning_hybrid_safety/submodules/molmospaces
PY=/root/act_retrain_venv/bin/python
cd /root/prox_learning_hybrid_safety
D=diagnostics_output/hybrid_obstacle_deployable_reference

# 1. provenance
$PY scripts/hybrid_obstacle_deployable_provenance.py --out $D/provenance_verification.json

# 2. paired oracle-reference dataset (5 shards; --skip-existing to resume)
$PY scripts/hybrid_obstacle_build_paired_reference_dataset.py \
    --split-manifest configs/hybrid_obstacle_canonical_split_v2.json \
    --collection-manifest configs/hybrid_obstacle_candidate_manifest_v2.json \
    --stack configs/hybrid_safety_stack_v1.json --safety-dir assets/safety/cvae_v3 \
    --run-dir assets/datagen/hybrid_obstacle_independent_v2/20260725_full160_4w \
    --ckpt-dir /root/act_retrain_assets/act_ckpts/hybrid_obstacle_act_baseline_v2/20260725_seed0_2000ep \
    --out-dir /root/act_retrain_assets/paired_reference_v1 --shard <i> --shards 5
$PY scripts/hybrid_obstacle_paired_dataset_manifest.py \
    --dataset-dir /root/act_retrain_assets/paired_reference_v1 \
    --split-manifest configs/hybrid_obstacle_canonical_split_v2.json \
    --out $D/paired_dataset_manifest.json

# 3. train both candidates, select, freeze
$PY scripts/hybrid_obstacle_train_deployable_reference.py \
    --dataset-dir /root/act_retrain_assets/paired_reference_v1 \
    --dataset-manifest $D/paired_dataset_manifest.json \
    --split-manifest configs/hybrid_obstacle_canonical_split_v2.json \
    --artifact-dir /root/act_retrain_assets/deployable_reference_v1 \
    --out $D/selection_report.json --deployment-manifest $D/deployment_manifest.json

# 4. offline development replay
$PY scripts/hybrid_obstacle_deployable_offline_replay.py \
    --development-manifest configs/hybrid_obstacle_controller_development4_v1.json \
    --paired-dir /root/act_retrain_assets/paired_reference_dev4 \
    --reference-manifest $D/deployment_manifest.json --out $D/offline_replay.json

# 5. freeze the schedule, then execute it (fresh process per rollout)
$PY scripts/hybrid_obstacle_deployable_schedule.py \
    --development-manifest configs/hybrid_obstacle_controller_development4_v1.json \
    --reference-manifest $D/deployment_manifest.json \
    --act-only-root /root/act_retrain_assets/rawhead_dev_v1 \
    --oracle-root /root/act_retrain_assets/oracle_dev_v1 \
    --output-root /root/act_retrain_assets/deployable_dev_v1 \
    --out configs/hybrid_obstacle_deployable_schedule_v1.json

cd submodules/act && $PY eval_act_obstacle_deployable.py \
    --eval-manifest ../../configs/hybrid_obstacle_controller_development4_v1.json \
    --episode-id <episode> --condition ACT_PLUS_DEPLOYABLE_REFERENCE --repeat-index <r> \
    --collection-manifest ../../configs/hybrid_obstacle_candidate_manifest_v2.json \
    --reference-manifest ../../$D/deployment_manifest.json \
    --ckpt_dir /root/act_retrain_assets/act_ckpts/hybrid_obstacle_act_baseline_v2/20260725_seed0_2000ep \
    --expected-act-checkpoint-sha256 dd7cd108a64ce10e5aab21b525dc06190f54d4e5fe446f65715b6852c49e7d36 \
    --expected-dataset-stats-sha256 c8119b904bfc80d66e3d33825722fcf9bb8bf3433c956dc09c27e6517d7c4ae2 \
    --output_dir /root/act_retrain_assets/deployable_dev_v1/<tag>

# 6. analysis, decision, tests
cd /root/prox_learning_hybrid_safety
$PY scripts/hybrid_obstacle_deployable_analysis.py --schedule configs/hybrid_obstacle_deployable_schedule_v1.json \
    --development-manifest configs/hybrid_obstacle_controller_development4_v1.json \
    --reference-manifest $D/deployment_manifest.json --selection-report $D/selection_report.json \
    --offline-replay $D/offline_replay.json --out $D/development_analysis.json
$PY -m pytest tests/test_deployable_reference_contract.py -q
```

---

## 18. `confirmatory41` remains untouched

| property | value |
|---|---|
| sha256 | `7b4500e9b4b2868e2612d7e444c34762d72c5e6e7b4b7c38bcf31f027b51b69e` |
| rows | 41 (32 hazard-present + 9 hazard-absent) |
| `executed_in_this_task` | **false** |
| rows executed in this task | **0** |
| overlap with any training, validation or development partition | **none** |

The deployable evaluator hard-refuses any manifest whose role is `CONFIRMATORY_UNTOUCHED`
and accepts only `DEVELOPMENT_ONLY`. Both refusals are covered by tests.

---

## 19. Next recommended task

**Close the distribution gap before changing the model.** The reference is trained on expert
trajectories that never enter the near-hazard regime the deployed policy actually visits,
and that — not the architecture — is what the live numbers implicate. Offline the model was
never once asked to produce a differential above `tau`; live it is asked constantly.

The natural next step is a predeclared **on-policy paired dataset**: run ACT_ONLY rollouts
on the training rows, generate the parked counterfactual along *those* trajectories with the
same state-neutral seam, and retrain the same fixed architecture on the union. Two secondary
items:

1. raise the quiet threshold's basis from a percentile of hazard-absent norms to a
   calibrated false-activation target measured **on-policy** — the current `tau` guarantees
   its own gate-3 result by construction;
2. reconsider support gate A, which is open on 76 % of frames because this enclosure is
   never more than 19 cm away, and whose companion far-frame gate has zero frames to test.

Do not execute `confirmatory41` on the strength of this report.

---

DEPLOYABLE_REFERENCE_LIVE_GROSS_REGRESSION
