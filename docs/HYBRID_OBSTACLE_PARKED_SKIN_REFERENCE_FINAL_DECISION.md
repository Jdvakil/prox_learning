# Hybrid obstacle — parked-skin reference: final decision

Date: 2026-07-26
Task: replace the seven-output parked-head estimator with one fixed causal model that
predicts the full parked 40×8×8 proximity field, routes it through the frozen SafetyHead,
and conditionally runs the four-row live development test.
Scope: this task does **not** execute `confirmatory41`.

---

## 1. Executive summary

The task stops at step 3, its own data-contract audit. **The training target does not
exist.**

The model predicts the parked 40×8×8 field, so every trainable frame needs two things: the
four causal *current* frames (input) and the *parked* field at the same decision state
(target). Across all 60 793 paired frames:

| distribution | files | frames | causal current skin | **parked skin target** |
|---|---|---|---|---|
| expert | 100 | 7 993 | yes, 4×40×8×8 | **missing** |
| ACT-only on-policy | 100 | 20 000 | **no** | **missing** |
| oracle on-policy | 100 | 20 000 | **no** | **missing** |
| learner-induced on-policy | 64 | 12 800 | **no** | **missing** |

**Frames meeting both contracts: 0 of 60 793.**

The parked field was never retained anywhere — only its SHA-256 and the 7-D SafetyHead
output derived from it, neither of which is invertible to the field. And the causal current
skins exist only for the 13 % of frames that come from the expert distribution; the three
on-policy distributions store the 40×4 per-sensor summary the *previous* model consumed,
not the fields this one needs.

So step 4's model cannot be trained, step 7's sampling contract (four distributions at 25 %
each) is unsatisfiable, and steps 8–21 are all downstream of those. Regenerating the data
is what would fix it, and this task's constraints forbid it: *"Do not collect another
on-policy training dataset."*

Everything that could be delivered without the missing target was:

* provenance verified — **55 checks, 0 failures**; no immutable artifact has drifted;
* the data contract audited in full and reported per distribution, with no silent repair;
* `CAUSAL_PARKED_SKIN_REFERENCE_V1` **implemented and verified** — 331 713 parameters,
  the `0 ≤ c_parked ≤ c_current` bound proven to hold even under saturated logits;
* the closeness transform, causal-history buffer, activity gate and strict loader
  implemented;
* **54 contract tests**, all passing.

**Decision: `PARKED_SKIN_DATA_CONTRACT_FAILED`.** No case classification: cases A/B/C
classify offline and live *results*, and no result exists.

---

## 2. Starting and final commits

| Repo | Branch | Start | Final |
|---|---|---|---|
| root | `develop/hybrid-obstacle-parked-skin-reference-v1` | `5270bee` | see §11 |
| ACT | `develop/hybrid-obstacle-parked-skin-reference-v1` | `21bf05e` | see §11 |
| MolmoSpaces | `repair/hybrid-obstacle-manifest-runner-v2` | `678f2eb` | `678f2eb`, **unmodified** |

The starting commits were resolved from the prior decision artifacts as instructed. One
wrinkle worth recording: `final_decision.json` necessarily stores the commit that existed
*while it was being written*, so it names `0bca2be`/`f6301bc` — the parents of the prior
task's own commits. The verification asserts both the recorded parent and the actual branch
tips, and they are consistent.

---

## 3. Immutable provenance

**55 checks, 0 failures** (`provenance_verification.json`).

`policy_best.ckpt` (epoch 1738) `dd7cd108…` · `dataset_stats.pkl` `c8119b90…` ·
Safety-CVAE `model.pt` `1fb2fc2b…` and `meta.json` `7c873756…` · `model_hybrid.xml`
`50924661…` · camera contract `7e90b4db…` · 40-sensor order `c31df8c3…` · collection
`8be80405…` · canonical `f49f5cd1…` · split `f7c2b227…` · converted dataset tree
`a567df08…` · `offsamples = 4` · residual constants (4.0, 2.2, 0.75, 0.35) unchanged ·
oracle implementation unmodified with `mj_forward_called` still `False` · expert paired,
labelling and learner dataset manifests all re-hash to their recorded values with 0
state-neutrality failures each · V1 and V2 checkpoints re-hash · partition `b06196b6…`
reused exactly · `confirmatory41.executed_in_this_task == false` · every reference
partition free of `development4` and `confirmatory41`.

**The blocker is not a mismatch.** Everything that exists is exactly what it should be.

---

## 4. Why the seven-output target failed

Recorded so the next task does not relitigate it. The previous round improved the 7-output
model on every evaluable measure — ACT-only on-policy differential MAE 0.313 → 0.208,
oracle on-policy 0.355 → 0.151, median cosine on ACT-only on-policy validation frames
−0.082 → **+0.884** — and still admitted no activation threshold: holding recall ≥ 0.80
left the median cosine at ~0.61, and the positive-cosine fraction never reached 0.80 above
4 % recall. The binding distribution was oracle-controlled on-policy.

Two structural reasons a spatial target is the right next move:

1. **The 7-D target equals its own input on most frames**, so the loss rewards the identity
   map. A dense field target with an explicit changed-pixel mask does not.
2. **A small MLP had to re-learn the head's 2560 → 7 mapping** from seven numbers of
   supervision per frame. Predicting the field and pushing it through the *frozen* head
   keeps that structure instead of imitating its output.

That reasoning is unchanged by this task's outcome. It is the data, not the idea, that is
missing.

---

## 5. Paired-skin data contract audit

`paired_skin_data_audit.json`. The audit reports; it does not repair.

**What is missing.** The parked 40×8×8 field, in every distribution. What was retained
instead is `parked_skin_sha256` (a hash) and `parked_head` / `privileged_parked_head`
(the 7-D head output). Neither can be inverted to a field.

**What exists and verifies.** The expert shards' causal input side is well-formed: all
shapes `4×40×8×8`, all values finite, the latest causal slot matching the decision-state
summary, and the history a shifted window with zero-padding at episode start — never a
future frame.

**What could not be checked at all.** The `parked_closeness ≤ current_closeness`
constraint, the changed-pixel mask, and the per-sensor/pixel/trajectory/distribution/phase
violation breakdowns. These are reported as `evaluable: false` with `null` breakdowns
rather than as passes — there is nothing to compare against, and recording a vacuous pass
would misrepresent the state of the data.

**Why the fields are absent.** Not corruption. The previous model consumed a 40×4
per-sensor summary, so that is what its collection retained; storing both raw fields would
have added roughly 2.9 GiB across 60 793 frames for data that model never used.

---

## 6. The physical parked-closeness constraint

Implemented and tested even though it cannot yet be applied to data.

```
closeness   = clip(1 - depth / 0.5, 0, 1)      # dead pixels (<5 mm) -> 0
removable   = c_current * sigmoid(change_logits)
c_parked    = clamp(c_current - removable, 0, 1)
depth_parked = 0.5 * (1 - c_parked)             # zero closeness -> 0.5 m, far / no activation
```

`0 ≤ c_parked ≤ c_current` therefore holds **by construction**, not by penalty. Verified
numerically, including with the change head's bias driven to +50 so every sigmoid
saturates: the bound still holds and the field stays in `[0, 1]`. A pixel with no current
return cannot acquire one. The dead-pixel rule matters here — a sub-5 mm reading is the
sensor reporting nothing, and left untransformed it would become closeness 1.0, the maximum
possible apparent threat.

---

## 7. Model: implemented, verified, never trained

`CAUSAL_PARKED_SKIN_REFERENCE_V1`, exactly as specified:

| component | shape |
|---|---|
| per-sensor encoder | `Linear(4×64 → 128)` + SiLU |
| sensor embedding | learned `40 × 128` |
| state context | `Linear(29 → 128)` SiLU `Linear(128 → 128)`, added to every token |
| cross-sensor encoder | TransformerEncoder, 2 layers, d_model 128, 4 heads, ff 256, pre-norm, dropout 0 |
| output A | per-sensor 64-pixel change logits |
| output B | frame-level activity logit from mean-pooled tokens |

**331 713 parameters**, against a 1 000 000 budget. Context width 29 = qpos(9) + qvel(9) +
nominal ACT action(8) + gripper state(2) + gripper command(1). No RGB input. No privileged
input — the runtime whitelist is enforced at load time and every privileged field name is
refused by test.

Also delivered and tested: the four-frame `CausalHistory` buffer (pads at episode start,
refuses a non-monotonic step, never returns a future frame), the `ActivityGate` (gates on
the model's activity probability, **not** on predicted-differential norm, and caps
magnitude at `rho_max` while preserving direction), and the strict loader (rejects a
tampered artifact, a non-four-frame history declaration, a privileged runtime input, or any
runtime-contract mismatch).

**Not trained.** No checkpoint exists, and none is claimed.

---

## 8. Everything downstream that was not performed

| step | status | why |
|---|---|---|
| training | not performed | no target exists |
| activity calibration | not performed | requires a trained model |
| `rho_max` | not performed | requires a calibrated model |
| disjoint validation (8 trajectories) | not performed | requires a frozen contract |
| offline test (20 trajectories) | not performed | requires a frozen contract |
| causal-history ablations | not performed | requires a trained model |
| development4 offline replay | not performed | requires a frozen model |
| **live rollouts** | **0 of 20 executed** | gated behind every offline gate |

No number in this report is a model performance figure, because no model was trained.

---

## 9. Candidate-118 negative control and live outcomes

Not applicable — no live rollout ran. The frozen ACT-only, oracle, V1 and V2 results from
prior tasks were not touched, not rerun and are not restated here as if they were new.

---

## 10. `confirmatory41` remains untouched

| property | value |
|---|---|
| sha256 | `7b4500e9b4b2868e2612d7e444c34762d72c5e6e7b4b7c38bcf31f027b51b69e` |
| rows | 41 (32 hazard-present + 9 hazard-absent) |
| `executed_in_this_task` | **false** |
| rows executed in this task | **0** |
| overlap with any reference partition | **none** |

---

## 11. Changed files and commits

**ACT** (`develop/hybrid-obstacle-parked-skin-reference-v1`), one new file:

* `parked_skin_reference.py` — the model, the physical counterfactual, the closeness
  transform, the causal-history buffer, the activity gate and the strict loader

**Root** (`develop/hybrid-obstacle-parked-skin-reference-v1`):

* `scripts/hybrid_obstacle_parked_skin_provenance.py`
* `scripts/hybrid_obstacle_parked_skin_data_audit.py`
* `scripts/hybrid_obstacle_parked_skin_write_decision.py`
* `tests/test_parked_skin_reference_contract.py` — 54 tests
* `diagnostics_output/hybrid_obstacle_parked_skin_reference/*.json`
* `docs/HYBRID_OBSTACLE_PARKED_SKIN_REFERENCE_FINAL_DECISION.md`

No model checkpoint is committed because none exists. Local paths and SHA-256 values for
the existing uncommitted data artifacts are recorded in `final_decision.json`.

MolmoSpaces unmodified. Nothing pushed.

---

## 12. Reproduction

```bash
export MUJOCO_GL=egl PYTHONUNBUFFERED=1
export MLSPACES_ASSETS_DIR=/root/prox_learning_hybrid_safety/assets
export PYTHONPATH=/root/prox_learning_hybrid_safety/submodules/molmospaces
PY=/root/act_retrain_venv/bin/python
cd /root/prox_learning_hybrid_safety
D=diagnostics_output/hybrid_obstacle_parked_skin_reference

$PY scripts/hybrid_obstacle_parked_skin_provenance.py --out $D/provenance_verification.json

$PY scripts/hybrid_obstacle_parked_skin_data_audit.py \
    --partition configs/hybrid_obstacle_reference_partition_v2.json \
    --expert-paired-dir /root/act_retrain_assets/paired_reference_v1 \
    --labelling-schedule configs/hybrid_obstacle_on_policy_labelling_schedule_v2.json \
    --learner-schedule configs/hybrid_obstacle_on_policy_learner_schedule_v2.json \
    --out $D/paired_skin_data_audit.json

$PY scripts/hybrid_obstacle_parked_skin_write_decision.py \
    --provenance $D/provenance_verification.json \
    --data-audit $D/paired_skin_data_audit.json --out $D/final_decision.json

$PY -m pytest tests/test_parked_skin_reference_contract.py -q
```

---

## 13. Next recommended task

**Regenerate the paired dataset with the parked field retained, then run this task
unchanged.** The model, the physical counterfactual, the closeness transform, the causal
buffer, the activity gate, the strict loader and 54 contract tests are already implemented
and committed. The only missing ingredient is data.

Store per frame, in addition to what is stored today:

* the **current** 40×8×8 field at the decision state;
* the **parked** 40×8×8 field at the *same* decision state;
* the four causal current frames (already stored for the expert distribution).

That means re-running 364 rollouts — 100 expert paired, 200 on-policy labelling, 64
learner-induced — at roughly 2.9 GiB uncompressed. Two ways to cut that materially:

1. store float16 **closeness** rather than float32 depth — the model consumes closeness
   anyway, and the transform is lossy in the same direction;
2. retain fields only for oracle-active frames plus a uniform sample of zero frames, and
   keep summaries for the rest — roughly a fifth of the size with no loss for this
   objective, since the changed-pixel supervision only exists on active frames.

Two things worth fixing while regenerating. The oracle's parked render is already computed
per frame in `PerFrameParkedObstacleReference`, so retaining it costs rendering time
already spent — the previous collection threw it away immediately after hashing it. And a
rerun is a *new sample* under MSAA, so the regenerated dataset will not reproduce the
existing one frame for frame; the partition and schedules should be reused, but the frames
themselves will differ.

Do not execute `confirmatory41`.

---

PARKED_SKIN_DATA_CONTRACT_FAILED
