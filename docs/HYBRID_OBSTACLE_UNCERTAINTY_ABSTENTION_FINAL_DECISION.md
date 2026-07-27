# Hybrid obstacle — uncertainty abstention, final decision

## Executive summary

All five bootstrap members trained and strict-loaded cleanly. **No agreement threshold
satisfies the calibration contract**, and the reason is not a marginal miss — it is that
replacing seed-variance with data-variance destroyed the signal the previous task measured.

The previous audit found changed-pixel-mask agreement across **three seeds trained on the
full training set** to be sharply discriminative: 0.167 on the 17 historical false positives
against 0.71–0.78 on genuine activity, a separation of about 0.55. This task's ensemble is
built differently by design — five members each trained on a **bootstrap resample of 40
trajectory clusters, of which only 24–28 are unique**. Against that ensemble the same metric
gives:

| Anchor-vs-member agreement | Three-seed reference | Trajectory bootstrap |
|---|---:|---:|
| Genuine oracle-active frames | 0.71 – 0.78 | **0.5467** |
| Oracle-zero frames | — | 0.6000 |
| Historical false positives | 0.167 | — |
| **Separation** | **≈ 0.55** | **+0.053** |

Mean pairwise agreement *among the members themselves* is 0.41 on active frames and 0.35 on
quiet ones — they do not agree with each other either. The disagreement is dominated by how
much data each member happened to see, not by epistemic uncertainty about particular frames.
Any threshold low enough to preserve genuine activations accepts almost everything; any
threshold high enough to abstain meaningfully rejects genuine activations too. That is
**Case C**: the original three-seed finding was not robust enough for deployment.

**A second, independent failure sits underneath it, and it would have blocked this contract
even with a perfect uncertainty metric.** At agreement threshold 0 — the gate accepting
everything — median trajectory active recall is **0.786**, already below the contract's 0.80
floor. That number is produced by the frozen activity threshold alone, which this task is
forbidden to refit. On the eight calibration clusters the frozen gate spans 0.152 to 1.000
recall across 18 trajectories with active frames.

Decision: **`UNCERTAINTY_ABSTENTION_CALIBRATION_INFEASIBLE`**, Case C. Per handoff step 9 the
task stops before the deployment manifest is frozen: **0 of the 20 permitted live rollouts**
were run, no deployment manifest was written, `development4` was not executed, and
`confirmatory41` remains untouched.

- Root branch `qualify/hybrid-obstacle-uncertainty-abstention-v1`, from `893c83a`
- ACT branch `qualify/hybrid-obstacle-uncertainty-abstention-v1`, from `91fc42a`
- MolmoSpaces `678f2eb`, untouched
- Decision JSON sha256 `0daf3a2c101615f4a891df636e8f855154f1dda0a7cad3adb71d94da47ffd944`

## Starting and final commits

| Repo | Expected | Observed | Modified |
|---|---|---|---|
| root | `893c83a` | `893c83a` → branch `qualify/hybrid-obstacle-uncertainty-abstention-v1` | code added only |
| ACT | `91fc42a` (resolved from prior artifacts) | `91fc42a` → branch `qualify/hybrid-obstacle-uncertainty-abstention-v1` | evaluator integration, +42/−3 lines |
| MolmoSpaces | `678f2eb` | `678f2eb` | no |

The ACT branch was created because step 15 requires an evaluator condition, which the handoff
explicitly permits. The integration was written and lints/parses clean, but it never ran: the
task stopped at calibration.

Remote `origin` → `Jdvakil/prox_learning`. Two pre-existing stashes untouched. Nothing pushed.
Runtime: Python 3.11.15, torch 2.7.1+cu126, numpy 2.4.6, CUDA 12.6, NVIDIA A10.

## Immutable provenance

**67 checks, 0 failed** — the first run in this chain to verify the ACT artifacts directly
rather than record their expected values. Both were located and hashed:

- `policy_best.ckpt` `dd7cd108a64ce10e…` at
  `/root/act_retrain_assets/act_ckpts/hybrid_obstacle_act_baseline_v2/20260725_seed0_2000ep/`
- `dataset_stats.pkl` `c8119b904bfc80d6…`, best epoch 1738

Also verified: seed-0 deployment checkpoint `47ae5cb1206c07d3…` with its config hash, variant,
seed and parameter count; diagnostic seeds 1 and 2; the proximity-gate checkpoint; paired
dataset tree `1fb68b3c…` with all 364 files still mode 444; SafetyHead `1fb2fc2b…`,
label_scale 11.359346389770508; residual constants gain 4.0 / decay 2.2 / EMA 0.75 /
max_dev 0.35 / arm-only / gripper owned by ACT; `offsamples` 4; the frozen activity threshold
**0.99960857629776**; the magnitude-support bound with `recalculated: false` and
`post_hoc_clipping: false`; development4 rows `[106, 107, 108, 118]`; confirmatory41 41 rows
`executed_in_this_task: false`; and all 17 historical false-positive identities recoverable.

## Identifiability result carried forward

Unchanged and not re-litigated: the deployable observation contract is identifiable. No exact
full-input opposite-label collision exists across 60,793 frames, none under the near-identity
tolerances, and the 17 historical false positives have local activity probability ≈ 0.0625
with zero active neighbours among their eight nearest. This task tests whether a *specific
ensemble construction* can exploit that, not whether the problem is solvable.

## Bootstrap ensemble

`PARKED_SKIN_TRAJECTORY_BOOTSTRAP_ENSEMBLE_V1`, manifest sha256
`5ddcbc162acd090625610ac0fe94cbb0e31bf5f17da0c58710bd8853eea4349d`.

Five members, fixed seeds 20260731 / 20260801 / 20260802 / 20260803 / 20260804. Each draws 30
hazard-present and 10 hazard-absent **trajectory clusters with replacement** from the parent
training split, includes every policy distribution of a sampled cluster, and preserves
multiplicity by index repetition rather than by reweighting. Architecture, preprocessing,
physical constraint, objective, loss weights, optimizer and schedule are identical to the
frozen seed-0 model; only the sample and the initialisation differ. Checkpoints were selected
on the eight checkpoint-validation clusters alone.

Two design points worth stating, because they are what make the negative result meaningful:

- **The cluster is the bootstrap unit.** Resampling frames would let one scene enter a
  member's training set as four correlated distribution copies, and the members would agree
  for a reason unrelated to epistemic uncertainty.
- **Resampling is hazard-stratified.** An unstratified draw could omit hazard-absent scenes
  entirely, and a member that had never seen a clear scene would disagree about clear scenes
  trivially.

| Member | Seed | Unique clusters | Out-of-bag | Checkpoint sha256 | dq MAE | Cosine | Mask F1 | Violations |
|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 0 | 20260731 | 27 | 13 | `c233b8bb7c22…` | 0.033815 | 0.9944 | 0.6346 | 0 |
| 1 | 20260801 | 28 | 12 | `00626b888463…` | 0.027204 | 0.9952 | 0.3214 | 0 |
| 2 | 20260802 | 24 | 16 | `c3d6a9a33bfc…` | 0.023495 | 0.9945 | 0.4665 | 0 |
| 3 | 20260803 | 25 | 15 | `3c924b088ec2…` | 0.050307 | 0.9799 | 0.4319 | 0 |
| 4 | 20260804 | 28 | 12 | `323ef688c463…` | 0.020421 | 0.9983 | 0.7447 | 0 |

All five are finite, physically valid, strict-reloadable, and well below the trivial
parked-field baseline (ZERO_DIFFERENTIAL validation MAE 0.071528). **No member was dropped or
replaced**, and no member-performance gate was applied — the handoff forbids removing
inconvenient members, and none was removed.

The changed-mask F1 spread across members (0.32 to 0.74) is itself informative: with only
24–28 unique clusters each, the members differ substantially in how they segment change, and
that is exactly the variance that later swamps the agreement metric.

## Anchor mask agreement

`anchor_mask_agreement` = mean Jaccard between the frozen seed-0 changed-pixel mask and each
of the five member masks, at the fixed pixel threshold 0.5, with two empty masks agreeing
completely. It is anchored on seed 0 rather than computed pairwise because an ensemble can be
internally consistent while collectively disagreeing with the model actually being deployed;
what matters is whether the *deployed* prediction is supported.

It was the only metric permitted to control abstention, and it was not swapped for another
after seeing results.

## Calibration result — infeasible

Eight uncertainty-calibration clusters, 748 candidate thresholds, **0 feasible**. The frozen
activity threshold remained active throughout and was not refit.

| τ | Median active recall | Mean final activation | Zero acceptance | Failing checks |
|---:|---:|---:|---:|---|
| 0.000 | 0.786 | 0.000 | 1.000 | recall, hard retention |
| 0.135 | 0.786 | 0.000 | 0.974 | recall, hard retention |
| 0.202 | 0.772 | 0.000 | 0.880 | + abstention cap |
| 0.262 | 0.752 | 0.000 | 0.880 | + active acceptance |
| 0.313 | 0.683 | 0.000 | 0.876 | + active acceptance |
| 0.410 | 0.567 | 0.000 | 0.735 | + zero/inactive acceptance |
| 0.467 | 0.527 | 0.000 | 0.734 | all acceptance floors |
| 0.614 | 0.227 | 0.000 | 0.518 | all acceptance floors |
| 0.710 | 0.000 | 0.000 | 0.509 | all |
| 1.000 | 0.000 | 0.000 | 0.303 | all, plus cosine |

Note the shape. Final false activation is **0.000 at every threshold** — the combined contract
never fires spuriously on this split. What fails is the other side: recall and the
anti-degeneracy acceptance floors. The gate is not too permissive; it is too destructive.

The binding failures across the sweep are: `median_active_recall`,
`median_hard_retention`, `zero_acceptance_floor`, `active_acceptance_floor`,
`inactive_acceptance_floor`, `trajectory_abstention_cap`, and at the extreme
`retained_cosine` / `retained_positive_cosine`.

### Disentangling the two failures

These are separable and both matter.

**1. The uncertainty signal collapsed.** Anchor agreement medians on the calibration split:
oracle-active **0.5467**, oracle-zero **0.6000** — a separation of **+0.053**, against roughly
0.55 for the three-seed ensemble. Members agree with each other at only 0.41 (active) and 0.35
(quiet). The metric can no longer tell a frame the model is confidently right about from one
it is confidently wrong about, because the members disagree with seed 0 almost everywhere.

The mechanism is not subtle: seeds 0/1/2 were each trained on the **full** `reference_train`,
so their disagreement isolated seed variance at fixed data. The bootstrap members see 24–28
unique clusters — a substantially smaller and different training set each — so their
disagreement is dominated by data variance. **Trajectory-bootstrap disagreement and seed
disagreement are not the same quantity**, and only the latter was ever shown to work.

**2. The frozen activity threshold alone cannot meet the recall floor on this split.** At
τ = 0, where the agreement gate accepts 100% of frames and abstains on nothing, median
trajectory active recall is 0.786 against the 0.80 requirement — range 0.152 to 1.000 over 18
trajectories. This is a property of the frozen threshold and the calibration split, not of the
ensemble. It was inherited, not introduced here, and the task is forbidden to refit it.

So even a perfect uncertainty metric would have failed this contract on this split. Both
findings are reported because acting on only one of them would mislead the next task.

## Quiet-frame anti-degeneracy result

The previous task recommended adding a quiet-frame acceptance floor, and it was added:
oracle-zero acceptance ≥ 0.80, active acceptance ≥ 0.80, acceptance on frames the activity
gate already declined ≥ 0.80, and no trajectory abstaining on more than 50% of its frames.

The floors did their job. They are the checks that fail first as τ rises — at τ = 0.202 the
trajectory abstention cap breaks, at τ = 0.262 active acceptance breaks, and by τ = 0.410 all
three acceptance floors have broken. Without them a threshold around 0.6 would have looked
attractive on false-activation grounds alone (0.000) while abstaining on roughly half of every
trajectory. The floor turned a metric that would have passed on the old contract into a
visible failure, which is what it was for.

## Stages not reached

Because handoff step 9 requires stopping when no agreement threshold is feasible, the
following did not run, and **no results are claimed for any of them**: the combined deployment
manifest freeze (step 10) — no manifest was written; nested offline evaluation (step 11); the
historical 17-frame regression (step 12); the reused diagnostic audit (step 13); frozen
inference determinism (step 14); the 20 live development rollouts (steps 16–17); live
uncertainty and approximation gates (step 18); live gross-regression gates (step 19).

There is therefore no 20-rollout outcome table, no live active-retention or abstention rate,
no candidate-118 negative control, and no task or contact outcome from this task.

The ACT evaluator integration for `ACT_PLUS_UNCERTAINTY_ABSTENTION` was written and is
committed — it registers the condition, requires `--abstention-manifest`, strict-loads the
seed-0 anchor and exactly five members, and routes members to masks only — but it was never
executed.

## Case classification

**Case C.** Trajectory-bootstrap disagreement cannot preserve active recall and ordinary quiet
acceptance simultaneously. The original three-seed finding was not robust enough for
deployment in this form.

The precise scope of that conclusion matters. What has been falsified is *the trajectory-
bootstrap construction as an uncertainty source for this contract*. The three-seed measurement
itself was not wrong — it was a measurement of a different ensemble, and this task shows it
does not transfer to a bootstrap ensemble trained on ~40% of the clusters.

## Confirmatory41 and development4

Both untouched. confirmatory41: 41 rows, `executed_in_this_task: false`, manifest sha256
`7b4500e9b4b2868e2612d7e444c34762d72c5e6e7b4b7c38bcf31f027b51b69e`. development4 rows
`[106, 107, 108, 118]`, not executed and not used for training, checkpoint selection or
threshold calibration; tests assert neither manifest is referenced by the training or
calibration scripts.

## Constraints honoured

Seed-0 deployment predictor not retrained or altered — its checkpoint hash is unchanged on
disk. Seeds 1 and 2 not selected for deployment. No parked field, head output, differential or
action averaged across models; the abstention module's only `mean()` calls are over Jaccard
scores, asserted by AST. No member can replace the seed-0 prediction — the runtime driver reads
`changed_probability` from members and nothing else. ACT and Safety-CVAE unmodified.
Residual-controller constants unchanged. Activity threshold unchanged. Magnitude-support bound
unchanged. Paired dataset unmodified and still read-only. No new training or on-policy data
collected. Temporal history not reintroduced. Uncertainty used only to turn execute into
abstain. 0 of 20 permitted live rollouts. Nothing pushed.

## Checkpoints

Ensemble weights held outside git (no approved artifact policy for binary weights here). Local
paths under `…/scratchpad/bootstrap_ensemble/member{N}_seed{SEED}/best.pt`:

| Member | Seed | sha256 |
|---:|---:|---|
| 0 | 20260731 | `c233b8bb7c22…` |
| 1 | 20260801 | `00626b888463…` |
| 2 | 20260802 | `c3d6a9a33bfc…` |
| 3 | 20260803 | `3c924b088ec2…` |
| 4 | 20260804 | `323ef688c463…` |

Full 64-character hashes and the `last.pt` companions are recorded in
`diagnostics_output/hybrid_obstacle_uncertainty_abstention/ensemble_training.json` and in the
ensemble manifest.

## Changed files

Added: `causal_parked_skin/abstention.py`;
`scripts/hybrid_obstacle_uncertainty_{ensemble_train,calibrate,decision}.py`;
`configs/hybrid_obstacle_uncertainty_ensemble_v1.json`;
`tests/test_uncertainty_abstention.py`;
`diagnostics_output/hybrid_obstacle_uncertainty_abstention/*.json`; this document.
ACT: added `submodules/act/uncertainty_abstention.py` and integrated the
`ACT_PLUS_UNCERTAINTY_ABSTENTION` condition into `eval_act_obstacle_on_policy.py`.
MolmoSpaces untouched.

## Reproduction commands

```bash
SCR=<scratchpad>
PY=/root/act_retrain_venv/bin/python
D=diagnostics_output/hybrid_obstacle_uncertainty_abstention
ACTDIR=/root/act_retrain_assets/act_ckpts/hybrid_obstacle_act_baseline_v2/20260725_seed0_2000ep

# 1. provenance, including the ACT artifacts
$PY scripts/hybrid_obstacle_reference_threshold_provenance.py \
    --act-checkpoint-dir $ACTDIR --out $D/provenance_verification.json

# 2. five bootstrap members
$PY scripts/hybrid_obstacle_uncertainty_ensemble_train.py \
    --cache $SCR/parked_skin_cache \
    --partition configs/hybrid_obstacle_prox_activity_partition_v1.json \
    --seed0-checkpoint $SCR/final/CURRENT_FRAME_ONLY__seed0/best.pt \
    --stack configs/hybrid_safety_stack_v1.json --safety-dir assets/safety/cvae_v3 \
    --checkpoint-root $SCR/bootstrap_ensemble \
    --manifest-out configs/hybrid_obstacle_uncertainty_ensemble_v1.json \
    --out $D/ensemble_training.json

# 3. calibrate the agreement threshold (stops here when infeasible)
$PY scripts/hybrid_obstacle_uncertainty_calibrate.py \
    --cache $SCR/parked_skin_cache \
    --partition configs/hybrid_obstacle_prox_activity_partition_v1.json \
    --ensemble-manifest configs/hybrid_obstacle_uncertainty_ensemble_v1.json \
    --seed0-checkpoint $SCR/final/CURRENT_FRAME_ONLY__seed0/best.pt \
    --stack configs/hybrid_safety_stack_v1.json --safety-dir assets/safety/cvae_v3 \
    --activity-threshold 0.99960857629776 \
    --groups diagnostics_output/hybrid_obstacle_activity_identifiability/diagnostic_groups.json \
    --onset-audit diagnostics_output/hybrid_obstacle_prox_activity_gate/onset_attribution.json \
    --deployment-manifest-out configs/hybrid_obstacle_uncertainty_deployment_v1.json \
    --out $D/calibration.json

# 4. decision
$PY scripts/hybrid_obstacle_uncertainty_decision.py \
    --provenance $D/provenance_verification.json \
    --ensemble-training $D/ensemble_training.json \
    --ensemble-manifest configs/hybrid_obstacle_uncertainty_ensemble_v1.json \
    --calibration $D/calibration.json \
    --partition configs/hybrid_obstacle_prox_activity_partition_v1.json \
    --agreement-diagnostic $D/agreement_diagnostic.json --out $D/final_decision.json

# verification
$PY -m pytest tests/test_uncertainty_abstention.py -q
$PY -m ruff check causal_parked_skin/ scripts/hybrid_obstacle_uncertainty_*.py
```

## Next recommended task

Two problems were found and they need different remedies; fixing one alone will not produce a
qualifiable contract.

1. **The recall floor is unreachable with the frozen activity threshold on this split.** Before
   any further uncertainty work, decide explicitly whether the 0.80 median-recall requirement or
   the frozen threshold gives way. They are currently incompatible: the threshold alone yields
   0.786. That is a contract decision, not an engineering one, and it should be made
   deliberately rather than absorbed into another calibration attempt.
2. **If uncertainty is pursued again, keep the data fixed and vary only the seed.** The
   three-seed signal came from seed variance at full data. This task shows that bootstrapping
   the data destroys it, and the mechanism — members trained on 24–28 unique clusters disagree
   with the anchor everywhere — predicts that any resampling scheme will do the same. A
   five-seed, full-data ensemble is the construction the evidence actually supports, and it
   would be a small change to this pipeline.
3. **Do not weaken the anti-degeneracy floors to make a threshold feasible.** They are what
   exposed this failure; removing them would readmit exactly the metric the previous task
   flagged as useless.

UNCERTAINTY_ABSTENTION_CALIBRATION_INFEASIBLE
