# Hybrid obstacle — activity identifiability audit, final decision

## Executive summary

The remaining activity failure **is** detectable through predictive uncertainty, and it is
**not** an identifiability failure of the deployable observation contract. Both halves of
that claim rest on measurements rather than on the absence of contrary evidence.

Disagreement among the three frozen parked-field seeds separates the 17 historical false
positives cleanly. The best metric, **changed-pixel-mask agreement**, rejects **17 of 17**
while retaining **96.5%** of oracle-active frames and **94.4%** of hard-true-active frames
(AUROC 0.979, partial AUROC at 5% FPR 0.783). Median mask agreement is 0.167 on the
failures against 0.71–0.78 on genuine activity — the failures are exactly where the seeds
disagree about *which pixels* move.

Nothing in the collision search argues the other way. Across all 60,793 frames there are
**zero** exact current-proximity collisions, **zero** exact full-deployable-input
collisions, and **zero** near-identity collisions under the predeclared tolerances. So there
is no pair of frames sharing an observation and disagreeing about the truth.

The neighbourhood evidence is the most informative single result, and it is asymmetric:

| Group | Opposite-label fraction, k=8 (raw proximity) |
|---|---:|
| **HISTORICAL_FALSE_POSITIVE** | **0.0000** |
| ONSET_ZERO | 0.0030 |
| LATE_ZERO | 0.2180 |
| LATE_ACTIVE | 0.4800 |
| **ONSET_ACTIVE** | **0.7165** |

The 17 failures sit in neighbourhoods that are **unanimously oracle-zero** in raw proximity,
in the proximity embedding, and in the full deployable input. Their observations do determine
their labels locally; the old head simply got them wrong. The genuine ambiguity in this
dataset runs the *other* way — onset-active frames sit among zero-labelled neighbours 72% of
the time — and that direction produces false negatives, not the false positives under
investigation.

Decision: **`EPISTEMIC_UNCERTAINTY_SIGNAL_PRESENT`**. All five not-identifiable triggers are
negative and all five signal-present criteria pass.

Two caveats I want on the record rather than buried, because they bound what this authorizes:

1. **n = 17.** Every separability number rests on seventeen frames. The gate is met
   decisively, but a 17-frame positive class cannot support a tight estimate of anything.
2. **One metric that passes the gate is degenerate and was excluded.**
   `norm_coefficient_of_variation` rejects 17/17 at 92% active recall, so it satisfies the
   gate as written — but its median on ordinary quiet frames (0.90–1.04) is *higher* than on
   the failures (0.708). At its operating point it would abstain on most genuinely quiet
   frames. The gate constrains active recall only, which is a hole in the gate; it is flagged
   in the decision JSON as `metrics_passing_letter_but_degenerate` and was not selected.

- Root branch `audit/hybrid-obstacle-activity-identifiability-v1`, from `db326d1`
- ACT `91fc42a`, MolmoSpaces `678f2eb` — untouched, clean
- Decision JSON sha256 `23b0b672501a68c6716ff9f20e0a80b4bc97178731ee6286b3293031733d9556`

## Corrected rejection of the state-prior hypothesis

Carried forward and reaffirmed. The task before last attributed the onset false activations
to a proprioceptive or episode-onset prior. Causal intervention on the frozen checkpoint
refuted it: with real state and the proximity field replaced by a clear reference, activity
on the 17 frames collapses from 0.9999 to **0.0229**; shuffling state or replacing it with
the batch mean leaves activity at 0.9999. State alone does not fire the head.

This audit adds a second, independent refutation from a different direction. Episode index
is never used as a feature here, and onset frames separate *at least as well* as late frames
once the comparison is done properly (below). There is no onset-specific input pathology to
find.

## Starting and final commits

| Repo | Expected | Observed | Modified |
|---|---|---|---|
| root | `db326d7` (see below) | `db326d1` → branch `audit/hybrid-obstacle-activity-identifiability-v1` | code added only |
| ACT | resolved from prior artifacts: `91fc42a1bfda2acd4b973bb53549bbf42d1fe9a6` | `91fc42a…` | no |
| MolmoSpaces | `678f2eb` | `678f2eb…` | no |

**Handoff commit discrepancy, disclosed.** The handoff states root `db326d7`. That object
does not exist in this repository (`git cat-file -t db326d7` → *Not a valid object name*).
The actual previous commit is `db326d1` — a one-character difference, and the commit that
wrote the previous decision. I treated this as a transcription slip rather than an artifact
mismatch and did **not** stop with `CHECKPOINT_OR_SOURCE_MISMATCH`, on the grounds that a
mistyped SHA in prose is not an immutable artifact, and every artifact that *can* be
hash-verified was verified independently: 66 checks, 0 failed. Had any of those failed the
decision would have been the mismatch token.

Remote `origin` → `Jdvakil/prox_learning`. Two pre-existing stashes untouched. Nothing pushed.
Runtime: Python 3.11.15, torch 2.7.1+cu126, numpy 2.4.6, CUDA 12.6, NVIDIA A10.

## Immutable provenance

**66 checks, 0 failed.** All three parked-field checkpoints match their recorded hashes —
seed 0 `47ae5cb1206c07d3…`, seed 1 `8797ad87e20f2d54…`, seed 2 `b4fa1b6daa2cf66b…` — as does
the proximity-only gate `50613f0914e48c0f…`. Paired dataset tree `1fb68b3c…` unchanged, all
364 files still mode 444. SafetyHead `1fb2fc2b…`, label_scale 11.359346389770508. ACT best
epoch 1738, `policy_best.ckpt` `dd7cd108…`, `dataset_stats.pkl` `c8119b90…`. Residual
constants unchanged. Sensor order unchanged, 40 sensors. `offsamples` 4. development4 rows
`[106, 107, 108, 118]`; confirmatory41 41 rows, `executed_in_this_task: false`.

Seeds 1 and 2 were loaded only as diagnostics. They were never averaged, never ensembled for
execution, and never became a deployment candidate; the decision artifact records seed 0 as
the candidate and tests assert it.

## Diagnostic frame groups

17,274 frames across the three consumed partitions. Zero development4 or confirmatory41
frames.

| Group | Count | Hazard + | Hazard − |
|---|---:|---:|---:|
| A HISTORICAL_FALSE_POSITIVE | 17 | 8 | 9 |
| B ONSET_ZERO | 1,359 | 909 | 450 |
| C LATE_ZERO | 12,812 | 9,029 | 3,783 |
| D ONSET_ACTIVE | 452 | 452 | 0 |
| E LATE_ACTIVE | 2,651 | 2,651 | 0 |
| F HAZARD_ABSENT_ZERO | 4,233 | 0 | 4,233 |
| G HARD_TRUE_ACTIVE | 1,958 | 1,958 | 0 |

Onset is `episode_step < max(10, ceil(0.10 × trajectory_length))`. HARD_TRUE_ACTIVE is the
set of genuinely active frames scoring below the highest-scoring historical false positive —
the frames a naive score threshold would sacrifice.

## Three-seed uncertainty metrics

Model weights verified bitwise unchanged after the audit; no optimizer is constructed in any
audit script, and tests assert that by AST.

| Metric | AUROC | pAUC@5% | Gate | Note |
|---|---:|---:|---|---|
| **changed_pixel_mask_agreement** | **0.9787** | **0.7825** | **pass** | selected |
| active_link_set_agreement | 0.9596 | 0.8037 | pass | |
| active_sensor_set_agreement | 0.9609 | 0.7975 | pass | |
| norm_coefficient_of_variation | 0.9476 | 0.5476 | pass | **degenerate — excluded** |
| max_parked_field_variance | 0.9219 | 0.6962 | fail | |
| mean_parked_field_variance | 0.9035 | 0.6653 | fail | |
| mean_pairwise_differential_cosine | 0.8968 | 0.6231 | fail | |
| min_pairwise_differential_cosine | 0.8868 | 0.5385 | fail | |
| differential_norm_variance | 0.6678 | 0.4872 | fail | |
| predicted_head_variance | 0.6231 | 0.4872 | fail | |

Selected metric by group (median mask agreement; **lower means more disagreement**):

| Group | Median |
|---|---:|
| A HISTORICAL_FALSE_POSITIVE | **0.1667** |
| G HARD_TRUE_ACTIVE | 0.7143 |
| E LATE_ACTIVE | 0.7555 |
| D ONSET_ACTIVE | 0.7778 |
| B ONSET_ZERO | 1.0000 |

All 17 failures fall in the top 5% of disagreement; 1 of 17 in the top 1%.

**On the "seeds merely differ" objection**, which the handoff explicitly forbids as a basis
for claiming usefulness. The claim here is not that the seeds differ — they differ on plenty
of frames. It is that they differ *specifically and almost exclusively* on the failure
frames while agreeing on genuine activity, which is what AUROC 0.979 with 17/17 rejection at
96.5% active retention measures. Disagreement that were merely generic would show up as a
near-chance AUROC and would force a recall collapse to reject the failures; neither happens.

Recall after rejecting the most uncertain frames:

| Reject | Active recall | Hard-true-active recall | Historical rejected |
|---:|---:|---:|---:|
| 1% | 0.9926 | 0.9883 | 1 / 17 |
| 2% | 0.9874 | 0.9801 | 1 / 17 |
| 5% | 0.9239 | 0.8820 | **17 / 17** |
| 10% | 0.9239 | 0.8820 | 17 / 17 |

The failures are concentrated between the 2nd and 5th percentile of disagreement, so an
abstention budget of roughly 5% captures all of them.

## Uncertainty separability result

All five predeclared signal-present criteria pass:

| Criterion | Result |
|---|---|
| A metric rejects ≥16/17 | PASS — 17/17 |
| Active recall ≥80% | PASS — 96.5% |
| HARD_TRUE_ACTIVE recall ≥80% | PASS — 94.4% |
| No substantial opposite-label collision around the failures | PASS — 0.0000 in all three input spaces |
| No exact full-input opposite-label collision | PASS — 0 |

## Nearest-neighbour label entropy

Pool: 43,519 `reference_train` frames. Exclusions: same trajectory, same episode identity,
duplicate scientific-state hash. 1,017 queries.

Opposite-label fraction at k=8:

| Group | Raw prox | Prox embedding | Full deployable | Model embedding |
|---|---:|---:|---:|---:|
| HISTORICAL_FALSE_POSITIVE | **0.0000** | **0.0000** | **0.0000** | 0.3603 |
| ONSET_ZERO | 0.0030 | 0.0240 | 0.0055 | 0.0410 |
| LATE_ZERO | 0.2180 | 0.1890 | 0.2095 | 0.1155 |
| LATE_ACTIVE | 0.4800 | 0.3855 | 0.4885 | 0.2865 |
| ONSET_ACTIVE | 0.7165 | 0.4990 | 0.7205 | 0.5430 |

The failures are unambiguous in every *observation* space and ambiguous only in
`D_FROZEN_MODEL_EMBEDDING` — the model's own mask-logit representation. That is a statement
about the model, not the input contract, and it is exactly what "the representation is
wrong about these frames" looks like. It is reported but deliberately not used as an
identifiability trigger.

Local activity probability (leave-one-trajectory-out, k=32, ambiguous band 0.2–0.8):

| Group | Fraction ambiguous | Median p_local |
|---|---:|---:|
| HISTORICAL_FALSE_POSITIVE | 0.2353 | 0.0625 |
| ONSET_ZERO | 0.0840 | 0.0000 |
| LATE_ZERO | 0.1440 | 0.0000 |
| ONSET_ACTIVE | 0.3800 | 0.2500 |
| LATE_ACTIVE | 0.2400 | 0.9062 |

The failures have median local activity probability 0.0625 — the neighbourhood says "quiet",
and it is right.

## Exact and near input collisions

| Search | Collisions |
|---|---:|
| Identical current-proximity hash, opposite labels | **0** |
| Identical full deployable-input hash, opposite labels | **0** |
| Near-identity within predeclared tolerances, opposite labels | **0** |

Tolerances: closeness ≤ 1e-5, qpos ≤ 1e-6, qvel ≤ 1e-6, nominal action ≤ 1e-6. Cross-episode
pairs only. Had a single exact full-input collision existed it would have been direct proof
that no deterministic model on this input contract can classify both frames correctly; none
does.

## Onset versus late comparison

Episode index was never used as a feature — it only selects which frames enter which
comparison.

Field statistics are nearly identical between classes at onset *and* late:

| Group | Mean max closeness | Mean active sensors | Mean changed-pixel fraction |
|---|---:|---:|---:|
| onset_zero | 0.6365 | 8.81 | 0.000000 |
| onset_active | 0.6360 | 8.19 | 0.001893 |
| late_zero | 0.8229 | 14.59 | 0.000000 |
| late_active | 0.7971 | 14.40 | 0.004203 |

**This is where a careless reading goes wrong, and I corrected it mid-audit.** The raw
between-class embedding distance is 5.635 at onset against 16.531 late, which looks like
severe onset-specific overlap. It is not: onset embeddings are simply smaller in magnitude.
Against the within-class baseline the picture reverses.

| | Between-class | Within-class baseline | **Separation ratio** |
|---|---:|---:|---:|
| onset | 5.635 | 5.026 | **1.1214** |
| late | 16.531 | 15.944 | **1.0368** |

Onset classes are *better* separated than late classes by this measure. Since late activity
is demonstrably identifiable — the frozen head reaches AUROC ≈ 0.998 there — an onset ratio
above the late ratio cannot support a claim of onset non-identifiability. The trigger does
not fire.

(The ratio being near 1.0 in *both* cases is a reminder that Euclidean neighbourhood
distances in a 7,680-dimensional embedding are a weak separability proxy; the head separates
these classes far better than the ratio suggests. It is reported for completeness, not as
the basis of the conclusion.)

Hazard-present versus hazard-absent onset-zero fields: embedding distance 5.159 against a
4.973 within-baseline, i.e. barely distinguishable before the privileged counterfactual is
applied. That is a genuine limitation of the raw field statistics, and it is why the head's
learned representation — not a distance in field space — is what does the discrimination.

## Score-tail and clustered-error analysis

Full-range AUROC is deliberately not the headline.

| Quantity | Value |
|---|---:|
| Oracle-zero score median | ≈ 0 |
| Oracle-zero q99.9 | 0.999824 |
| Oracle-zero max | 0.999999 |
| Oracle-active q1 | 0.017331 |
| Oracle-active min | 0.000008 |

The distributions overlap in the extreme tail: the worst zero frames outscore the weakest
1% of genuine activations by orders of magnitude. Partial ROC:

| FPR | TPR |
|---:|---:|
| 0.1% | 0.7132 |
| 0.5% | 0.9233 |
| 1% | 0.9381 |
| 2% | 0.9513 |
| 5% | 0.9671 |

At the previously frozen threshold, trajectory-clustered bootstrap (10,000 replicates,
seed 20260727, resampling whole trajectories):

- false-positive rate **0.00120** (95% CI 0.00034 – 0.00250)
- active recall **0.770** (95% CI 0.671 – 0.853)

False-positive run-length histogram across 108 trajectories: 99 with no false positive, 7
with a run of 1, 1 with a run of 3, and **1 with a run of 7** — two clustered episodes. The
aggregate rate is excellent and the clustering is the whole problem, which is consistent with
everything found in the previous two tasks.

## Identifiability conclusion

**The current deployable observation contract is sufficient to distinguish these frames.**
The 17 historical false positives are not ambiguous observations: no frame in the dataset
shares their input and carries the opposite label, their nearest neighbours in every
observation space are unanimously quiet, and their local activity probability is 0.0625. The
old head was wrong about frames that the data says are determinable.

Ensemble disagreement locates them. The three seeds diverge about which pixels move on
exactly these frames while agreeing on genuine activity, giving 17/17 rejection at 96.5%
active retention.

The real ambiguity in this dataset is the mirror image — onset-active frames sit among quiet
neighbours 72% of the time — and it would cause missed detections rather than spurious
corrections. That is a separate problem, and it is not the one this chain has been chasing.

## Permitted next scientific options

Under `EPISTEMIC_UNCERTAINTY_SIGNAL_PRESENT` the authorized next task is:

- train **one** predeclared trajectory-bootstrap ensemble;
- use uncertainty **only** as an abstention signal, never to modify the emitted correction;
- retain the seed-0 mean parked-field model unless a new model contract is explicitly
  approved;
- qualify on development4 before confirmatory41.

Three things I would carry into that task's design:

1. **Fix the gate's hole.** The current gate constrains rejection of failures and retention
   of actives, but says nothing about retention of *quiet* frames — which is how
   `norm_coefficient_of_variation` passed while being useless. Any successor contract should
   add a minimum quiet-frame retention.
2. **Treat n=17 as the binding limit.** The abstention budget that captures all 17 sits
   between the 2nd and 5th percentile of disagreement. That boundary is estimated from
   seventeen points and should be calibrated with trajectory-clustered intervals, not a point
   estimate.
3. **Expect the mirror failure.** Abstaining on high disagreement will also abstain on some
   onset-active frames, which are the genuinely ambiguous ones. The 5% operating point
   already costs 7.6 points of hard-true-active recall.

## Confirmatory41 and development4

Both untouched. confirmatory41: 41 rows, `executed_in_this_task: false`, manifest sha256
`7b4500e9b4b2868e2612d7e444c34762d72c5e6e7b4b7c38bcf31f027b51b69e`. development4 rows
`[106, 107, 108, 118]`, not executed and not used for any fitting. Neither appears in the
paired dataset, so no diagnostic group can contain their frames; tests assert the audit
scripts never reference either manifest.

## Constraints honoured

No model trained or fine-tuned — no optimizer is constructed in any audit script, asserted by
AST. Seeds 1 and 2 not selected for deployment and not ensembled for execution. No checkpoint
altered; all weights verified bitwise unchanged. ACT, Safety-CVAE and the residual controller
untouched. No activation threshold changed. Paired dataset unaltered and still mode 444. No
new simulator trajectories. No live policy evaluation. development4 and confirmatory41 not
used for fitting and not executed. Uncertainty is not claimed useful merely because seeds
differ. Nothing pushed.

## Changed files

Added: `scripts/hybrid_obstacle_activity_{ensemble_audit,collision_audit,onset_tail_audit,identifiability_decision}.py`;
`tests/test_activity_identifiability.py`;
`diagnostics_output/hybrid_obstacle_activity_identifiability/*.json`; this document.
Modified: `scripts/hybrid_obstacle_reference_threshold_provenance.py` (extended with the
diagnostic-seed and gate checks). Nothing under `submodules/` was touched.

## Reproduction commands

```bash
SCR=<scratchpad>
PY=/root/act_retrain_venv/bin/python
D=diagnostics_output/hybrid_obstacle_activity_identifiability
GATE=diagnostics_output/hybrid_obstacle_prox_activity_gate

# 1. provenance -> CHECKPOINT_OR_SOURCE_MISMATCH on any failure
$PY scripts/hybrid_obstacle_reference_threshold_provenance.py \
    --out $D/provenance_verification.json

# 2. frozen groups + three-seed ensemble + uncertainty separability
$PY scripts/hybrid_obstacle_activity_ensemble_audit.py \
    --cache $SCR/parked_skin_cache --checkpoint-root $SCR/final \
    --stack configs/hybrid_safety_stack_v1.json --safety-dir assets/safety/cvae_v3 \
    --onset-audit $GATE/onset_attribution.json \
    --groups-out $D/diagnostic_groups.json --out $D/ensemble_audit.json

# 3. collisions, neighbour ambiguity, local identifiability
$PY scripts/hybrid_obstacle_activity_collision_audit.py \
    --cache $SCR/parked_skin_cache \
    --checkpoint $SCR/final/CURRENT_FRAME_ONLY__seed0/best.pt \
    --stack configs/hybrid_safety_stack_v1.json \
    --onset-audit $GATE/onset_attribution.json --out $D/collision_audit.json

# 4. onset-vs-late and score tails
$PY scripts/hybrid_obstacle_activity_onset_tail_audit.py \
    --cache $SCR/parked_skin_cache \
    --checkpoint $SCR/final/CURRENT_FRAME_ONLY__seed0/best.pt \
    --stack configs/hybrid_safety_stack_v1.json \
    --onset-audit $GATE/onset_attribution.json --out $D/onset_tail_audit.json

# 5. decision
$PY scripts/hybrid_obstacle_activity_identifiability_decision.py \
    --provenance $D/provenance_verification.json --groups $D/diagnostic_groups.json \
    --ensemble $D/ensemble_audit.json --collision $D/collision_audit.json \
    --onset-tail $D/onset_tail_audit.json --out $D/final_decision.json

# verification
$PY -m pytest tests/test_activity_identifiability.py -q
$PY -m ruff check causal_parked_skin/ scripts/hybrid_obstacle_activity_*.py
```

EPISTEMIC_UNCERTAINTY_SIGNAL_PRESENT
