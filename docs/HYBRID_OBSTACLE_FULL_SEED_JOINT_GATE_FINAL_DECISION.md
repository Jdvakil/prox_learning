# Hybrid obstacle — full-seed joint gate, final decision

## Executive summary

**The owner's diagnosis was right: the old standalone activity threshold was the incompatible
component, and joint calibration fixes it without touching the recall floor.** A feasible
(activity, agreement) pair exists — 1,690 of them — and the selected pair delivers median
active recall **1.000** on calibration against the preserved 0.80 floor, with a
cluster-bootstrap upper bound on false activation of **0.00000**. Activity-alone retention
rose from **0.786** under the retired threshold to **1.000** under the recalibrated one. The
recall floor was never lowered; the threshold moved because the deployed system is now a
two-gate controller.

**Offline transfer then fails on all three checks**, and the mechanism is specific rather
than diffuse.

| Check | Result |
|---|---|
| Nested offline (8 clusters) | **FAIL** — false-active run 3 (>2), persistent correction after support ends |
| Historical 17-frame regression | **FAIL** — 7/17 rejected, **10 executed** (needs ≥16 rejected, 0 executed) |
| Reused diagnostic (20 clusters) | **FAIL** — false-active run 7 (>5) |

The coverage side is excellent throughout: nested recall 0.997, diagnostic recall 0.970, mean
executed false activation 0.0044 and 0.0026 (both well under their ceilings), quiet-frame
agreement acceptance 0.968. What fails is *clustering* — the 7-frame run in the original
failing episode is back and executing.

**The proximate cause is the agreement definition, and it is worth stating exactly.** The
handoff specifies the anchor form, `mean(J(seed0,seed1), J(seed0,seed2))`. The identifiability
audit that produced the validated 17/17 result used the **three-pair** mean, which also
includes `J(seed1,seed2)`. On all 17 historical frames `J(seed0,seed2) = 0.000` and
`J(seed1,seed2)` is likewise ~0, so:

- three-pair form: `(0.5 + 0 + 0) / 3 = 0.167` — exactly the audit's reported median
- anchor form: `(0.5 + 0) / 2 = 0.250` — the observed median here

Dropping the seed1–seed2 term raises the failures' agreement by roughly 1.5×, to a median of
0.250 and a maximum of 0.375. The selected agreement threshold is 0.225, so ten of the
seventeen now clear it. A threshold above 0.375 would be needed to reject them all, and only
14 of 266 agreement values survive the quiet-frame acceptance floors at all.

Decision: **`FULL_SEED_JOINT_GATE_OFFLINE_TRANSFER_FAILED`**, Case C. The task stops before
live execution: **0 of 20 permitted rollouts**, `development4` not executed, `confirmatory41`
untouched.

- Root branch `qualify/hybrid-obstacle-full-seed-joint-gate-v1`, from `bea058c`
- ACT `69bda27`, MolmoSpaces `678f2eb` — both untouched
- Decision JSON sha256 recorded in `final_decision.json`

## Owner decision: recall floor versus activity threshold

Honoured exactly as directed, and the direction was vindicated.

| | Retired standalone | Jointly calibrated |
|---|---:|---:|
| Activity threshold | 0.99960857629776 | **0.99154764** |
| Activity-alone active retention | 0.786 | **1.000** |
| Final active recall (after uncertainty) | — | **1.000** |
| Mean executed false activation | 0.000 | 0.000 |
| Quiet-frame agreement acceptance | — | 0.967 |

Stated explicitly, as the handoff requires:

- **The 0.80 recall floor was not relaxed.** It remains the binding coverage constraint in
  the contract and in code (`MIN_MEDIAN_ACTIVE_RECALL = 0.80`), asserted by test.
- **The standalone threshold changed because the deployed system is now a two-gate
  controller.** A threshold calibrated for a single gate consumed all the available recall,
  leaving none for a veto to spend. Recalibrating it jointly restored the headroom.
- **Uncertainty did not justify lowering system-level active recall.** Final recall is 1.000
  on calibration and 0.997 on nested — higher than the retired configuration achieved before
  any veto existed.

## Starting and final commits

| Repo | Expected | Observed | Modified |
|---|---|---|---|
| root | `bea058c` | `bea058c` → branch `qualify/hybrid-obstacle-full-seed-joint-gate-v1` | code added only |
| ACT | `69bda27` | `69bda27` | no — the evaluator condition was never reached |
| MolmoSpaces | `678f2eb` | `678f2eb` | no |

Full hashes were resolved from the previous decision artifacts rather than trusted from the
abbreviated forms. Remote `origin` → `Jdvakil/prox_learning`; two pre-existing stashes
untouched; nothing pushed. Runtime: Python 3.11.15, torch 2.7.1+cu126, numpy 2.4.6, CUDA 12.6,
NVIDIA A10.

## Immutable provenance

**67 checks, 0 failed**, including both ACT artifacts verified directly on disk:
`policy_best.ckpt` `dd7cd108a64ce10e…` and `dataset_stats.pkl` `c8119b904bfc80d6…`, best
epoch 1738.

All three full-data seeds match their recorded hashes — seed 0 `47ae5cb1206c07d3…`, seed 1
`8797ad87e20f2d54…`, seed 2 `b4fa1b6daa2cf66b…` — and **share one configuration hash apart
from the seed** (`ccbef0b982aa9641…`, 838,434 parameters each), confirming they are the same
architecture, training set and preprocessing. Paired dataset tree `1fb68b3c…` unchanged, all
364 files mode 444. SafetyHead `1fb2fc2b…`. Residual constants, magnitude-support bound
(`recalculated: false`, `post_hoc_clipping: false`), sensor order and `offsamples = 4` all
unchanged. development4 rows `[106, 107, 108, 118]`; confirmatory41 41 rows,
`executed_in_this_task: false`. All 17 historical identities recoverable.

## Bootstrap-ensemble rejection

Recorded as `invalid_for_deployment_uncertainty_due_to_cluster_omission_variance`. Its
checkpoints and reports were **not deleted**. It is proven absent from threshold fitting,
nested evaluation, the runtime evaluator, the deployment manifest and any live rollout, and
`joint_gate.assert_not_bootstrap` raises on a bootstrap manifest or on any member record
carrying a `bootstrap_seed`. Tests exercise both refusal paths and confirm the calibration
script never references the ensemble manifest.

## Agreement implementation

Pinned rather than reconstructed, because two earlier modules in this project disagree:

- **Mask comparison is strict** (`probability > 0.5`), matching the identifiability audit.
  The later bootstrap module used `>=`; the audit is the measurement the owner decision rests
  on, so its form is used. A test asserts both the joint-gate source and the audit source use
  the strict form.
- **Controlling metric is the anchor form**, `mean(J(seed0,seed1), J(seed0,seed2))`, with two
  empty masks agreeing completely, exactly as the handoff specifies. The audit's three-pair
  form is computed and logged as a secondary metric and can never alter execution.

Implementation hash pinned in the deployment manifest.

## Joint threshold grid and feasible pairs

Full Cartesian product over the eight calibration clusters: **5,084 activity × 266 agreement
= 1,352,344 pairs**, with boundary values 0.0 and 1.0 included on both axes.

Evaluation order was restructured for tractability without changing the rule. Two conditions
factorise — activity-alone retention depends only on the activity axis, the agreement
acceptance floors only on the agreement axis — so those screens run first (4,630 / 5,084
activity values and **14 / 266** agreement values survive). For each surviving agreement
value the per-trajectory false-activation rate across the whole activity axis is a
searchsorted lookup, so the 10,000-replicate cluster bootstrap runs once per agreement value
rather than once per pair. **3,692 pairs** then received full per-trajectory evaluation, of
which **1,690 are feasible**.

That the agreement axis loses 252 of 266 values to the quiet-frame acceptance floors is itself
the important structural fact: only very permissive agreement thresholds survive them.

### Selected pair

| | Value |
|---|---:|
| Activity threshold | **0.99154764** |
| Agreement threshold | **0.225** |
| Bootstrap 95% upper bound, false activation | **0.00000** |
| Median active recall | 1.0000 |
| Activity-alone retention | 1.0000 |
| Mean executed false activation | 0.0000 |
| Quiet-frame agreement acceptance | 0.9671 |

Selected lexicographically: lowest bootstrap upper bound, then highest median active recall,
then highest HARD_TRUE_ACTIVE retention, then highest ordinary-zero acceptance, then highest
activity threshold, then highest agreement threshold. The ordering was fixed before results
were visible. Every calibration check passed.

Cluster bootstrap: 10,000 replicates, seed 20260727, resampling whole episodes.

## Nested offline evaluation

Opened only after the deployment manifest was written and hashed.

| Metric | Value | Gate |
|---|---:|---|
| Median active recall | 0.9972 | ≥0.75 ✓ |
| Mean executed false activation | 0.00436 | ≤0.02 ✓ |
| Quiet-frame agreement acceptance | 0.9677 | ≥0.80 ✓ |
| **Max false-active run** | **3** | **≤2 ✗** |
| **Persistent correction after support** | **yes** | **none ✗** |

Coverage and aggregate false activation are comfortably inside their gates. The failure is
that the residual false activations *cluster* and persist past the end of oracle support —
the same failure mode this chain has been chasing since the threshold task.

## Historical 17-frame regression

Evaluated after threshold freezing; never used for fitting.

| Frames | Result |
|---|---|
| Rejected | 7 of 17 |
| Rejected specifically by disagreement | 7 |
| **Executed** | **10** |
| Requirement | ≥16 rejected, ≥16 by disagreement, **0 executed** |

Frame-level detail (all 17 pass the activity gate at either threshold — activity ≥ 0.9998):

| J(s0,s1) | J(s0,s2) | Anchor | ≥0.225 | Executed | Count |
|---:|---:|---:|---|---|---:|
| 0.750 | 0.000 | 0.375 | yes | **yes** | 5 |
| 0.647 | 0.000 | 0.324 | yes | **yes** | 1 |
| 0.611 | 0.000 | 0.306 | yes | **yes** | 1 |
| 0.500 | 0.000 | 0.250 | yes | **yes** | 2 |
| 0.471 | 0.000 | 0.235 | yes | **yes** | 1 |
| 0.421 | 0.000 | 0.211 | no | no | 3 |
| 0.400 | 0.000 | 0.200 | no | no | 2 |
| 0.385 | 0.000 | 0.192 | no | no | 1 |
| 0.000 | 0.000 | 0.000 | no | no | 1 |

Two facts stand out. **`J(seed0, seed2) = 0.000` on every one of the seventeen** — seed 2's
changed-pixel mask is disjoint from seed 0's on exactly these frames, which is a clean,
reproducible signal. And the retired activity threshold would have rejected **0 of 17**: these
frames always passed the activity gate, so disagreement was always the only thing that could
stop them.

The anchor definition halves the weight of that clean seed-2 signal by averaging it with the
noisier seed-1 term instead of the audit's three-way mean, which is what lifts ten frames
above the highest agreement threshold the anti-degeneracy floors permit.

## Reused diagnostic audit

Labelled `reused_nonconfirmatory_diagnostic` throughout.

| Metric | Value | Gate |
|---|---:|---|
| Median active recall | 0.9695 | ≥0.70 ✓ |
| Mean executed false activation | 0.00255 | ≤0.03 ✓ |
| **Max false-active run** | **7** | **≤5 ✗** |

The run of 7 is the original failing episode's onset burst, executing again end to end.
Passing this set would not have been a confirmatory result; failing it blocks live execution.

## Frozen inference stability

24 repeated inferences on a fixed 256-frame batch across all three seeds:

| Quantity | Max absolute delta |
|---|---:|
| Activity | 0.0 |
| Anchor agreement | 0.0 |
| Pairwise Jaccard (both pairs) | identical |
| Execution decision | identical |

No training kernels invoked. The gate is exactly reproducible; the failure is not numerical.

## Stages not reached

Because the offline transfer checks fail, the following did not run and **no results are
claimed**: evaluator integration for `ACT_PLUS_FULL_SEED_JOINT_GATE` (step 16); the 20 live
development rollouts (steps 17–18); live approximation and abstention gates (step 19); live
gross-regression gates (step 20). There is no 20-rollout outcome table, no live retention or
abstention rate, no candidate-118 negative control, and no task or contact outcome from this
task.

## Case classification

**Case C.** Offline joint calibration passes but nested and diagnostic transfer fail.

Conclusion as predeclared: full-data seed disagreement does not transfer reliably enough for
closed-loop deployment under this contract, and **additional same-input ensembles are not
justified**. The chain has now tested three constructions on the same inputs — three full-data
seeds, a five-member trajectory bootstrap, and this joint two-gate contract — and the residual
failure mode has survived all three.

One scope note, because it bounds the conclusion fairly: this run used the handoff's anchor
agreement, not the three-pair form that produced the validated 17/17. The evidence above shows
the two are materially different on precisely the frames that matter. Whether the three-pair
form would have passed the historical regression at a threshold that also satisfies the
quiet-frame floors is **not established here**, and I would not assume it: its median on the
17 is 0.167, which is lower and therefore easier to reject, but the acceptance floors that
eliminated 252 of 266 anchor thresholds would apply to it as well.

## Confirmatory41 and development4

Both untouched. confirmatory41: 41 rows, `executed_in_this_task: false`, manifest sha256
`7b4500e9b4b2868e2612d7e444c34762d72c5e6e7b4b7c38bcf31f027b51b69e`. development4 rows
`[106, 107, 108, 118]`, not executed and not used for fitting; tests confirm neither manifest
is referenced by the calibration or decision scripts.

## Constraints honoured

No model trained or fine-tuned. No new seeds. Bootstrap members used for neither deployment
nor calibration. Seeds 1 and 2 never selected as deployment predictor — a test parses the
calibration AST and confirms the only value read from them is `changed_probability`. No
fields, heads, differentials or actions averaged; the joint-gate module's only `mean()` calls
are inside the two agreement functions. ACT and Safety-CVAE unmodified. Residual constants,
magnitude-support bound and the parked-field predictor unchanged. Paired dataset unmodified
and still read-only. The 0.80 recall floor was not lowered. The historical 17 frames,
development4 and confirmatory41 were not used for fitting. 0 of 20 permitted live rollouts.
Nothing pushed.

## Changed files

Added: `causal_parked_skin/joint_gate.py`;
`scripts/hybrid_obstacle_joint_gate_{calibrate,decision}.py`;
`configs/hybrid_obstacle_full_seed_joint_gate_v1.json`;
`tests/test_full_seed_joint_gate.py`;
`diagnostics_output/hybrid_obstacle_full_seed_joint_gate/*.json`; this document. Nothing
under `submodules/` was touched.

## Reproduction commands

```bash
SCR=<scratchpad>
PY=/root/act_retrain_venv/bin/python
D=diagnostics_output/hybrid_obstacle_full_seed_joint_gate
ACTDIR=/root/act_retrain_assets/act_ckpts/hybrid_obstacle_act_baseline_v2/20260725_seed0_2000ep

# 1. provenance, including the ACT artifacts
$PY scripts/hybrid_obstacle_reference_threshold_provenance.py \
    --act-checkpoint-dir $ACTDIR --out $D/provenance_verification.json

# 2. joint calibration, manifest freeze, nested / historical / diagnostic transfer
$PY scripts/hybrid_obstacle_joint_gate_calibrate.py \
    --cache $SCR/parked_skin_cache \
    --partition configs/hybrid_obstacle_prox_activity_partition_v1.json \
    --checkpoint-root $SCR/final --stack configs/hybrid_safety_stack_v1.json \
    --safety-dir assets/safety/cvae_v3 \
    --groups diagnostics_output/hybrid_obstacle_activity_identifiability/diagnostic_groups.json \
    --onset-audit diagnostics_output/hybrid_obstacle_prox_activity_gate/onset_attribution.json \
    --deployment-manifest-out configs/hybrid_obstacle_full_seed_joint_gate_v1.json \
    --out $D/joint_calibration.json

# 3. decision
$PY scripts/hybrid_obstacle_joint_gate_decision.py \
    --provenance $D/provenance_verification.json --calibration $D/joint_calibration.json \
    --manifest configs/hybrid_obstacle_full_seed_joint_gate_v1.json \
    --partition configs/hybrid_obstacle_prox_activity_partition_v1.json \
    --out $D/final_decision.json

# verification
$PY -m pytest tests/test_full_seed_joint_gate.py -q
$PY -m ruff check causal_parked_skin/ scripts/hybrid_obstacle_joint_gate_*.py
```

## Next recommended task

The joint recalibration did what the owner expected — it removed the activity threshold as the
blocking component and restored full coverage. What remains is narrower than before and should
be treated as such.

1. **Resolve the agreement-definition question before anything else.** The anchor form and the
   three-pair form differ by 1.5× on exactly the seventeen frames that decide this contract,
   and only the three-pair form has ever been validated. Re-running this same calibration with
   the three-pair metric costs one script argument and would settle whether the transfer
   failure is a property of seed disagreement or an artifact of dropping `J(seed1,seed2)`.
   Nothing else should be attempted until that is known.
2. **Do not add another same-input ensemble.** Three constructions have now failed on the same
   observation contract. The Case C prescription stands.
3. **If the three-pair form also fails, the run-length and persistence gates are the target,
   not the coverage gates.** Aggregate false activation is already 4–10× inside its ceiling;
   every failure in this task was a clustering or persistence failure. A contract change that
   addresses temporal clustering directly — rather than another per-frame score — is the
   remaining unexplored direction.

FULL_SEED_JOINT_GATE_OFFLINE_TRANSFER_FAILED
