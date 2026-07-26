# Hybrid obstacle — on-policy reference aggregation round: final decision

Date: 2026-07-26
Task: one predeclared DAgger-style on-policy dataset-aggregation round for the
posture+skin reference MLP, activation recalibrated on disjoint trajectories, and the
four-row live development evaluation repeated.
Scope: this task does **not** execute `confirmatory41`.

---

## 1. Executive summary

The aggregation round worked. The model got substantially better on every distribution.
It still cannot be given an activation contract, so the task stops before live rollouts.

**What improved.** Round 1 beats V1 on every evaluable predeclared criterion, and beats
round 0 on all of them:

| metric (pooled, disjoint sets) | V1 | V2 round 1 |
|---|---|---|
| ACT-only on-policy differential MAE | 0.313 | **0.208** |
| oracle on-policy differential MAE | 0.355 | **0.151** |
| norm-ratio error | 0.585 | **0.420** |
| oracle-zero predicted norm, median (threshold-free) | 0.308 | **0.256** |

On the reference-validation trajectories the round-1 median oracle cosine on ACT-only
on-policy frames is **+0.884** with 93.9 % positive, against V1's **−0.082**. The
distribution-shift diagnosis from the previous task was correct, and on-policy
aggregation addresses it.

**What did not.** The predeclared lexicographic calibration is **infeasible**. On the 8
calibration trajectories there is no threshold that simultaneously retains the signal and
points the right way:

| tau | recall | median cosine | positive fraction |
|---|---|---|---|
| 0.00 | 1.000 | 0.612 | 0.743 |
| 0.30 | 0.922 | 0.631 | 0.737 |
| **0.50** | **0.724** | **0.699** | **0.718** |
| 0.80 | 0.489 | 0.776 | 0.720 |
| 2.00 | 0.044 | 0.815 | 0.792 |

Floors are recall ≥ 0.80, median cosine ≥ 0.70, positive fraction ≥ 0.80. To hold recall
the direction must be given up; to gain direction the recall collapses; and the positive
fraction never reaches 0.80 until recall is 4 %. This is not a marginal miss.

Because no activation contract can be established, disjoint validation and offline testing
cannot be run against a frozen tau, and the 20 live development rollouts were **not**
executed.

**Case C. Decision: `ON_POLICY_REFERENCE_OFFLINE_INVALID`.**

---

## 2. Starting and final commits

| Repo | Branch | Start | Final |
|---|---|---|---|
| root | `develop/hybrid-obstacle-on-policy-reference-v2` | `0bca2be` | see §14 |
| ACT | `develop/hybrid-obstacle-on-policy-reference-v2` | `f6301bc` | see §14 |
| MolmoSpaces | `repair/hybrid-obstacle-manifest-runner-v2` | `678f2eb` | `678f2eb`, **unmodified** |

No commit was made on the previous development branches. Nothing was pushed.

---

## 3. Immutable provenance

**49 checks, 0 failures** (`provenance_verification.json`). Expected digests recovered from
the deployable, oracle, raw-head and ACT-baseline decision artifacts.

`policy_best.ckpt` (epoch 1738) `dd7cd108…` · `dataset_stats.pkl` `c8119b90…` ·
Safety-CVAE `model.pt` `1fb2fc2b…` · `model_hybrid.xml` `50924661…` · 40-sensor order
`c31df8c3…` · 80/20 split `f7c2b227…` · `confirmatory41` `7b4500e9…` ·
`offsamples = 4` · residual constants (4.0, 2.2, 0.75, 0.35) unchanged · the oracle
implementation's `mj_forward_called` is still `False` · V1's deployment manifest and
expert paired dataset re-hash to their recorded values.

---

## 4. Reference partition

The ACT split is untouched. Within the 80 ACT-training trajectories, assignment is by the
episode's committed `predeclared_stratum_rank` — no fresh randomness, no file-order
dependence (asserted at AST level, not by substring).

| partition | hazard present | hazard absent | total |
|---|---|---|---|
| reference train | 48 | 16 | 64 |
| reference calibration | 6 | 2 | 8 |
| reference validation | 6 | 2 | 8 |
| offline reference test (= ACT validation) | 15 | 5 | 20 |

`partition_sha256 b06196b641b702e93dd2450bee0d0007d28ace29cf0fa8983bb42df3da0a827d`.
All partitions are **pairwise disjoint**, the three reference partitions cover ACT-train
exactly, and neither `development4` nor `confirmatory41` appears anywhere.

---

## 5. Rollout schedules and on-policy datasets

All three schedules were frozen and hashed before anything executed.

| schedule | rollouts | sha256 |
|---|---|---|
| labelling (100 rows × 2 conditions) | 200 | `efc0a09a…` |
| learner-induced (64 training rows) | 64 | `56615da0…` |
| live development (4 rows × 5) | 20 | `30cd9fc5…` **not executed** |

Condition order in the labelling schedule is balanced by row rank: 50 rows run
ACT-only first, 50 run the oracle first.

**Executed: 264 rollouts, zero failures.**

| distribution | rollouts | frames | oracle-active frames |
|---|---|---|---|
| expert (inherited) | 100 | 7 993 | 1 740 |
| ACT-only on-policy | 100 | 20 000 | 4 800 |
| oracle on-policy | 100 | 20 000 | 4 536 |
| learner-induced on-policy | 64 | 12 800 | 3 401 |

**Oracle pairing correctness, checked not asserted**: 0 state-neutrality failures across
all 52 800 on-policy frames; on every hazard-absent row the oracle differential is
**exactly zero** on every frame; both halves rendered at the same decision state; no
dynamics-advancing call; privileged labels confined to the `privileged_` namespace, which
no feature builder reads.

---

## 6. Round 0 and round 1

Both rounds use the **unchanged** V1 architecture (196 → 256 → 256 → 128 → 7, 150 023
parameters), the unchanged optimiser, learning rate, weight decay, batch size, epoch
budget, loss and seed, and both start from a **fresh initialisation** — neither continues
from V1 weights. No sweep, no second architecture.

Distribution weighting is equal *by distribution*, then uniform over rows, then uniform
over frames, so the shorter expert trajectories cannot be swamped by the longer on-policy
ones.

| | distributions | best epoch | validation loss |
|---|---|---|---|
| `…_V2_ROUND0` | expert, ACT-only, oracle (1/3 each) | 5 | 0.135015 |
| `…_V2_ROUND1` | + learner-induced (1/4 each) | 86 | **0.060308** |

Both reload bitwise-identically under strict loading.

**Round 0's own calibration was also infeasible**, with markedly worse direction than
round 1 (calibration ACT-only cosine 0.140 vs round 1's 0.724). The handoff requires the
learner round regardless, so round 0 was run under an explicitly labelled
**collection-only gate** — `tau = 0`, output bounded by the oracle-derived `rho_max`
5.8347 — so the learner-induced states are those an unfiltered learner actually reaches,
with no trajectory driven by an unbounded magnitude. Nothing was tuned and no validation
or offline-test data was touched. That gate was never evaluated and is not an activation
contract.

---

## 7. Calibration: infeasible

`tau` was fitted on the 8 reference-calibration trajectories only, by the predeclared
lexicographic rule. The previous task's self-evaluated percentile rule was **not** reused,
and the previous `tau` value appears nowhere in this task's code.

Oracle-active frames are those with an oracle differential above the numerical floor and,
where the analytic teacher can be evaluated at all, an active teacher: **548 active** and
**3 273 zero** frames.

The sweep in §1 is the whole result. Every candidate threshold fails at least one floor:

* recall ≥ 0.80 forces `tau ≤ ~0.4`, where the median cosine is 0.61–0.63;
* median cosine ≥ 0.70 needs `tau ≥ ~0.5`, where recall is 0.724;
* the positive fraction never reaches 0.80 above 4 % recall.

`rho_max` is well defined regardless, because it depends only on the privileged oracle:
the 99th percentile of the oracle differential norm on active calibration frames,
**5.8347**. It is recorded but has nothing to gate.

**The binding distribution is oracle-on-policy.** Per-distribution medians on the
calibration set for round 1: expert +0.722 (86.9 % positive), ACT-only on-policy +0.724
(70.8 %), **oracle on-policy +0.501 (74.8 %)**. States visited when the *true* oracle is
driving are the ones the model reads worst — which is exactly the regime a deployed
reference would create for itself.

---

## 8. Support envelope

The V1 support gate is removed, as instructed: its global-minimum-depth condition was open
on 76 % of validation frames because this enclosure is never more than ~19 cm away, so it
was dominated by static geometry and gated almost nothing. `SupportEnvelopeGate` replaces
it with a quiet threshold plus an output-support bound that preserves direction and caps
the norm at `rho_max` — the bound that would have contained V1's 5.1–6.9× over-prediction
without touching the residual controller's gain or `max_dev`.

The mechanism is implemented and unit-tested (including that a 6.9× vector is capped and
its direction preserved). It is unused here because the threshold half of the contract
could not be established.

---

## 9. Disjoint validation and offline test

**Not run against a frozen contract**, because no contract exists. Reporting them under an
invented threshold would be reporting a tuned result, so they are omitted rather than
fabricated.

What *can* be stated without a threshold is the per-distribution direction and error on the
disjoint reference-validation trajectories, which is what §10 does.

---

## 10. V1 versus V2 shift analysis

Both models evaluated on **identical frames** from the disjoint validation and offline-test
trajectories.

| set / distribution | differential MAE V1 → V2 | median cosine V1 → V2 |
|---|---|---|
| validation / expert | 0.151 → **0.104** | — → +0.843 |
| validation / ACT-only on-policy | 0.337 → **0.202** | −0.082 → **+0.884** |
| validation / oracle on-policy | 0.190 → **0.166** | −0.286 → **+0.592** |
| test / expert | 0.249 → **0.218** | +0.987 → +0.908 |
| test / ACT-only on-policy | 0.290 → **0.214** | +0.924 → +0.685 |
| test / oracle on-policy | 0.520 → **0.137** | +0.543 → +0.382 |

Required improvements: ACT-only on-policy MAE **PASS**, oracle on-policy MAE **PASS**,
norm-ratio error **PASS**. The oracle-zero false-activation comparison is **not evaluable**
— V2 has no feasible tau, so its gated activation rate would be an artefact of the
diagnostic `tau = 0` used to run the audit. The threshold-free substitute is the predicted
norm on oracle-zero frames, where V2 is also better: median **0.256** against V1's
**0.308**.

Note the mixed picture on the offline-test set: V2's MAE is uniformly better while its
median cosine on two distributions is *lower* than V1's. V1's cosines there are computed
on the few frames its depth gate let through, so the two are not measuring the same
population; MAE, which is computed on all active frames, is the comparable quantity.

Offline improvement alone does not establish that the architecture is adequate — and here
it does not, because the contract cannot be closed.

---

## 11. Live development: not executed

The 20-rollout schedule is frozen and hashed (`30cd9fc5…`) but **no live rollout was run**,
per step 10's instruction to stop before live development when the offline stage fails.
The frozen ACT-only, oracle and V1 baselines were left untouched and un-rerun.

There is consequently no live approximation data, no candidate-118 live negative control,
and no gross-regression result in this task. The prior task's live figures stand unchanged.

---

## 12. Case classification

**Case C.** V2 fails the disjoint offline on-policy stage — specifically, the activation
contract cannot be established at all. The conclusion the handoff attaches to Case C is
the one the evidence supports: *the existing feature contract cannot reliably identify
hazard-specific background cancellation*, and the task stops before live rollout.

Two qualifications belong with that.

1. **The aggregation round is not what failed.** Every evaluable comparison moved the right
   way, several of them by a factor of two. If the round had been the problem, the numbers
   would not have improved.
2. **The failure is concentrated where it matters most.** The model is worst on
   oracle-on-policy states — the states created when a *correct* reference is driving.
   A deployed reference would generate those states for itself, so a contract fitted
   anywhere else would be optimistic. This is the substantive finding, and it is why one
   more aggregation round on the same features is unlikely to close the gap.

---

## 13. `confirmatory41` remains untouched

| property | value |
|---|---|
| sha256 | `7b4500e9b4b2868e2612d7e444c34762d72c5e6e7b4b7c38bcf31f027b51b69e` |
| rows | 41 (32 hazard-present + 9 hazard-absent) |
| `executed_in_this_task` | **false** |
| rows executed in this task | **0** |
| overlap with any partition | **none** |

The on-policy evaluator hard-refuses any manifest whose role is `CONFIRMATORY_UNTOUCHED`
and accepts only `REFERENCE_PARTITION` or `DEVELOPMENT_ONLY`. Verified by dry run and test.

---

## 14. Changed files and commits

**ACT** (`develop/hybrid-obstacle-on-policy-reference-v2`):

* `eval_act_obstacle_on_policy.py` — new: the three on-policy conditions with per-frame
  oracle labelling
* `deployable_reference.py` — extended with `SupportEnvelopeGate`; V1's `SupportGate`,
  feature schema and loader are unchanged

**Root** (`develop/hybrid-obstacle-on-policy-reference-v2`):

* `configs/hybrid_obstacle_reference_partition_v2.json`
* `configs/hybrid_obstacle_on_policy_{labelling,learner,live}_schedule_v2.json`
* `scripts/hybrid_obstacle_on_policy_provenance.py`
* `scripts/hybrid_obstacle_reference_partition.py`
* `scripts/hybrid_obstacle_on_policy_schedule.py`
* `scripts/hybrid_obstacle_on_policy_dataset_audit.py`
* `scripts/hybrid_obstacle_train_on_policy_reference.py`
* `scripts/hybrid_obstacle_on_policy_shift_audit.py`
* `scripts/hybrid_obstacle_on_policy_analysis.py` (written, unused — no live rollouts)
* `scripts/hybrid_obstacle_on_policy_write_decision.py`
* `tests/test_on_policy_reference_contract.py` — 45 tests
* `diagnostics_output/hybrid_obstacle_on_policy_reference/*.json`
* `docs/HYBRID_OBSTACLE_ON_POLICY_REFERENCE_FINAL_DECISION.md`

**Not committed** (paths and SHA-256 in `final_decision.json`): the 264 rollout frame
shards and payloads, the round-0 and round-1 checkpoints, the expert paired dataset, the
ACT checkpoint, Safety-CVAE weights, canonical datasets.

MolmoSpaces unmodified. Nothing pushed.

---

## 15. Reproduction

```bash
export MUJOCO_GL=egl PYTHONUNBUFFERED=1
export MLSPACES_ASSETS_DIR=/root/prox_learning_hybrid_safety/assets
export PYTHONPATH=/root/prox_learning_hybrid_safety/submodules/molmospaces
PY=/root/act_retrain_venv/bin/python
cd /root/prox_learning_hybrid_safety
D=diagnostics_output/hybrid_obstacle_on_policy_reference

$PY scripts/hybrid_obstacle_on_policy_provenance.py --out $D/provenance_verification.json

$PY scripts/hybrid_obstacle_reference_partition.py \
    --split-manifest configs/hybrid_obstacle_canonical_split_v2.json \
    --canonical-manifest configs/hybrid_obstacle_canonical_manifest_v2.json \
    --development-manifest configs/hybrid_obstacle_controller_development4_v1.json \
    --confirmatory-manifest configs/hybrid_obstacle_confirmatory41_v1.json \
    --out configs/hybrid_obstacle_reference_partition_v2.json

for W in labelling learner live; do
  $PY scripts/hybrid_obstacle_on_policy_schedule.py \
      --partition configs/hybrid_obstacle_reference_partition_v2.json \
      --development-manifest configs/hybrid_obstacle_controller_development4_v1.json \
      --which $W --output-root /root/act_retrain_assets/on_policy_${W}_v2 \
      --out configs/hybrid_obstacle_on_policy_${W}_schedule_v2.json
done

# 200 labelling rollouts, then 64 learner rollouts (fresh process per rollout)
cd submodules/act && $PY eval_act_obstacle_on_policy.py \
    --eval-manifest ../../configs/hybrid_obstacle_reference_partition_v2.json \
    --episode-id <episode> --condition ACT_ONLY_ON_POLICY \
    --collection-manifest ../../configs/hybrid_obstacle_candidate_manifest_v2.json \
    --ckpt_dir /root/act_retrain_assets/act_ckpts/hybrid_obstacle_act_baseline_v2/20260725_seed0_2000ep \
    --expected-act-checkpoint-sha256 dd7cd108a64ce10e5aab21b525dc06190f54d4e5fe446f65715b6852c49e7d36 \
    --expected-dataset-stats-sha256 c8119b904bfc80d66e3d33825722fcf9bb8bf3433c956dc09c27e6517d7c4ae2 \
    --output_dir /root/act_retrain_assets/on_policy_v2/<tag>

cd /root/prox_learning_hybrid_safety
$PY scripts/hybrid_obstacle_on_policy_dataset_audit.py \
    --schedule configs/hybrid_obstacle_on_policy_labelling_schedule_v2.json \
    --development-manifest configs/hybrid_obstacle_controller_development4_v1.json \
    --confirmatory-manifest configs/hybrid_obstacle_confirmatory41_v1.json \
    --out $D/labelling_dataset_manifest.json

for R in 0 1; do
  $PY scripts/hybrid_obstacle_train_on_policy_reference.py \
      --partition configs/hybrid_obstacle_reference_partition_v2.json \
      --labelling-schedule configs/hybrid_obstacle_on_policy_labelling_schedule_v2.json \
      --learner-schedule configs/hybrid_obstacle_on_policy_learner_schedule_v2.json \
      --expert-paired-dir /root/act_retrain_assets/paired_reference_v1 --round $R \
      --artifact-dir /root/act_retrain_assets/on_policy_reference_v2 \
      --out $D/round${R}_training.json \
      --deployment-manifest $D/round${R}_deployment_manifest.json
done

$PY scripts/hybrid_obstacle_on_policy_shift_audit.py \
    --partition configs/hybrid_obstacle_reference_partition_v2.json \
    --labelling-schedule configs/hybrid_obstacle_on_policy_labelling_schedule_v2.json \
    --expert-paired-dir /root/act_retrain_assets/paired_reference_v1 \
    --v1-manifest diagnostics_output/hybrid_obstacle_deployable_reference/deployment_manifest.json \
    --v2-manifest $D/round1_diagnostic_gate_manifest.json --out $D/shift_audit.json

$PY -m pytest tests/test_on_policy_reference_contract.py -q
```

---

## 16. Next recommended task

**Do not run another aggregation round on this feature contract.** The round did its job
and the contract still cannot be closed; a second round would be a larger version of the
same experiment.

The evidence points at what to change. The model is asked to predict a 7-vector head
output from a 160-number per-sensor summary, and it is worst exactly where a correct
reference would put it. Two candidates, in order of directness:

1. **Predict the parked *skin*, not the parked head.** The oracle differential is
   `head(current) − head(parked)`; predicting the 40 × 8 × 8 parked depth field and pushing
   it through the frozen head keeps the head's own structure instead of asking a small MLP
   to imitate its output. It also makes the target dense and spatially supervised rather
   than a 7-number regression whose target equals its own input on most frames.
2. **A richer causal model over the causal skin stack.** The current summary discards
   spatial layout within each 8 × 8 patch and all temporal structure beyond the current
   frame; the four causal frames are already collected and unused.

Either way, calibrate on oracle-on-policy states specifically — that is where the current
model is weakest, and where a deployed reference will spend its time.

Do not execute `confirmatory41`.

---

ON_POLICY_REFERENCE_OFFLINE_INVALID
