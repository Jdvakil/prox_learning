# Chunk-25 place rollout evaluation

## Decision

**CHUNK25_PARTIAL.** The gripper closed in some but not most episodes: the copy-the-state shortcut is weakened but not cleanly broken at chunk 25. This is the informative middle point the plan predeclared, not a failure.

| Arm | N | Task success | Collision-free task success | Gripper close commanded |
|---|---:|---:|---:|---:|
| ACT | 40 | 8/40 | 6/40 | 9/40 |
| PACT | 40 | 11/40 | 10/40 | 20/40 |

| Arm | Hazard-bar contact | Other-environment contact | Clutter contact | Place-receptacle contact | 900 control steps | Wall-clock seconds min / median / mean / max |
|---|---:|---:|---:|---:|---:|---:|
| ACT | 13 | 0 | 0 | 0 | 40/40 | 930.8 / 1010.3 / 1037.6 / 1271.4 |
| PACT | 6 | 0 | 0 | 0 | 40/40 | 929.8 / 1025.3 / 1053.2 / 1275.5 |

## The three-point chunk table — the durable output of this run

The 20 shared physical instances, evaluated at all three chunk sizes with the same seed-3101
checkpoints of each arm:

| Arm | Chunk | Gripper close commanded | Task success | Collision-free task success |
|---|---:|---:|---:|---:|
| ACT | 1 | 0/20 | 0/20 | 0/20 |
| ACT | 25 | 6/20 | 6/20 | 5/20 |
| ACT | 100 | 20/20 | 7/20 | 7/20 |
| PACT | 1 | 0/20 | 0/20 | 0/20 |
| PACT | 25 | 9/20 | 4/20 | 4/20 |
| PACT | 100 | 20/20 | 10/20 | 9/20 |

Rows 0–19 of the chunk-25/chunk-100 manifest are the same physical episodes as chunk 1, matched on
`task_seed_u64`, both panel jitters, and `intrusion_side`. Their `episode_id` and `row_sha256`
intentionally differ because schema version and role participate in those hashes; hash equality was
not asserted.

## Modality contrast — directional only, no interaction claim

`PACT − ACT` collision-free task success was **+10.0 pp** at chunk 25
(paired approximate 95% interval -3.7 to
+23.7 pp), beside **+7.5 pp** at chunk 100
(+7.5 pp recomputed here on the same instances, interval
-10.2 to +25.2 pp).

The reactivity hypothesis is an interaction — does `(PACT − ACT)@25` exceed `(PACT − ACT)@100`? The
instance-paired difference-in-differences is +2.5 pp
(-19.1 to +24.1 pp). **This run cannot
confirm or refute the reactivity hypothesis.** It was predeclared as unresolvable at N = 40: the
plan's independent-sample estimate gave a ±30 pp interval against a chunk-100 gap of only +7.5 pp,
and detecting a 10 pp change would need roughly 714 instances per cell. The directional number above
is reported as a directional number and nothing more.

## The arms did not break the shortcut equally — read the contrast above in this light

ACT commanded the gripper closed in 9/40 episodes, PACT in
20/40. **No episode in either arm succeeded without a
commanded close** (asserted, not assumed). So the two arms are not merely better and worse at the
same task — they attempt the grasp at very different rates, and the unconditional rates above mix
"attempts the grasp at all" with "does the task well once it attempts".

Restricted to the episodes where each arm actually closed the gripper:

| Arm | Episodes closing | Task success given close | Collision-free success given close |
|---|---:|---:|---:|
| ACT | 9 | 8/9 (89%) | 6/9 (67%) |
| PACT | 20 | 11/20 (55%) | 10/20 (50%) |

Conditional on attempting, ACT is the *higher*-scoring arm on both endpoints; PACT's larger
unconditional total comes from attempting more than twice as often. These conditional cells are
small (9 and 20
episodes) and the conditioning variable is itself an outcome, so this is not evidence that ACT is
better either — it is evidence that **the unconditional `PACT − ACT` gap cannot be read as a
proximity effect at chunk 25.** The dominant difference between these two arms is how often each
one escaped the copy-the-state shortcut, which is a training-dynamics difference, not a
sensing-modality one.

## Runtime and scope

Smoke ran 4/4 rollouts and commanded the gripper
closed in 1/4. Mean smoke time was
15.94 minutes per rollout; the re-derived full-eval
projection was 2.12 hours. Runtime decision: `RUN_FULL`.
Full dispatch completed 80/80 jobs with
10 workers and zero errors.

## Verification and limits

- Every result used `num_queries=25`; both checkpoints loaded with `strict=True` in the
  evaluator, and PACT recorded feature width 32 with projection `(512, 32)` and
  `proximity_consumed_for_action: true`.
- The instance manifest is the chunk-100 file reused **byte-identical**
  (`515bb60d00613aa4990f7a824e5aadcff9bcb56361f6b64aaab8fa8510981018`), not regenerated. Its `schema_version` and `role` still say `chunk100`;
  those fields are cosmetic and its rows encode chunk-agnostic instances, which is exactly what
  makes chunks 1, 25 and 100 paired on identical episodes.
- No new evaluator, contract or manifest was written. `eval_pact_place_chunk100_row.py` was invoked
  unchanged with `--num-queries 25`; only a dispatch launcher and this finaliser are new.
- Training commands differ from the chunk-100 pair **only** by `--chunk_size` and `--ckpt_dir`, and
  PACT differs from ACT by exactly the five proximity flags — both asserted flag-by-flag over
  sorted `(flag, value)` pairs, which is order-independent but still catches a dropped or repeated
  flag. The argv *token order* does differ from the chunk-100 pair: the chunk-25 run was relaunched
  by a script that groups the shared flags and places `--ckpt_dir` last. The parsed flag set and
  every value are identical apart from the two intended ones.
- **ACT@25 was trained twice.** The first attempt died at epoch 1800/2000 when the filesystem
  filled while writing a checkpoint. That partial run was deleted rather than resumed from its
  epoch-1600 bundle, so both chunk-25 arms have identical single-shot provenance. Disk space was
  reclaimed by deleting rollout trajectories and videos from an unrelated July 2026 run; no
  artifact this evaluation depends on was touched, as the hash checks above confirm.
- All 152 converted dataset files were re-hashed; the protected tree
  remains `b16a5a0bd221d786f54fd9f28e00d493d01316ed47d9e909c1a915d37b13e6f1`. The chunk-1 and chunk-100 checkpoints, the frozen
  encoder, and every published evaluator retained their recorded SHA-256 values.
- **Single seed 3101, N = 40 per arm.** Rate-difference precision is limited.
- **No within-model ablation this run.** `PACT_PERMUTED` was skipped because at chunk 100 it scored
  6/40 against ACT's 13/40 — below the no-proximity baseline, so it behaved as an active distractor
  rather than a clean control, and it inflated the chunk-100 headline (of the reported +25.0 pp for
  PACT − PACT_PERMUTED, only +7.5 pp is PACT − ACT). That leaves the chunk-100 anomaly undiagnosed
  and leaves this run with only the cross-model contrast, which has flipped sign between seeds in
  this project before.
- The proximity encoder is the frozen **corridor** encoder used unchanged on the place task —
  cross-task transfer.
- `place_receptacle` contacts are over-counted because a learned policy exposes no expert phase, so
  the audit phase remains `other`.

The exact chunk-100 → chunk-25 training-command diff for each arm was:

```diff
# ACT
--- --ckpt_dir /root/pact_place_152_pact_vs_act_chunk100_seed3101/act_seed3101
+++ --ckpt_dir /root/pact_place_152_pact_vs_act_chunk25_seed3101/act_seed3101
--- --chunk_size 100
+++ --chunk_size 25

# PACT
--- --ckpt_dir /root/pact_place_152_pact_vs_act_chunk100_seed3101/pact_seed3101
+++ --ckpt_dir /root/pact_place_152_pact_vs_act_chunk25_seed3101/pact_seed3101
--- --chunk_size 100
+++ --chunk_size 25
```

The exact ACT → PACT command-only differences at chunk 25 were:

```diff
--- --ckpt_dir /root/pact_place_152_pact_vs_act_chunk25_seed3101/act_seed3101
+++ --ckpt_dir /root/pact_place_152_pact_vs_act_chunk25_seed3101/pact_seed3101
+++ --use_proximity --n_proximity_sensors 40 --prox_tokens_per_sensor 1 --proximity_feature_dim 32 --proximity_encoder_sha256 6fd2dd037e3236b5b6bf7fce8cb2709ead0cf52adcbbe9cbad1061efc2fe3206
```

Machine-readable details are in `analysis.json`.

---

# V9.6 clustered-hazard siting (this run) — measurement only, nothing authorized

Separate work from the chunk-25 rollout evaluation above: no policy was trained or rolled out. This
run executed W1-W3 of
[`docs/PACT_PLACE_V9_RAW_ADMISSION_FIX_PLAN.md`](docs/PACT_PLACE_V9_RAW_ADMISSION_FIX_PLAN.md).
Full write-up: [`docs/PACT_PLACE_V96_CLUSTER_SITING_STATUS.md`](docs/PACT_PLACE_V96_CLUSTER_SITING_STATUS.md).

**Decision: STOP for review.** `authorizes_gate: false`, `authorizes_collection: false`,
`authorizes_v1b: false`.

## W1 — resolving-power instrument, retrodiction against the measured raw counterfactual

`scripts/pact_skin_resolvability.py`, driven by `scripts/run_pact_place_v9_w1_resolvability.py`.
Artifact: `diagnostics_output/pact_place_v9_w1_resolvability/resolvability.json`.

| metric | value |
|---|---:|
| Variants replayed | 8 of 8 |
| Role measurements compared | 24 |
| Ordering `panel >> outbound > inbound` reproduced | 8 / 8 |
| Pearson r on `changed_values` | 0.99997 |
| Spearman r on `changed_values` | 1.000 |
| Predicted / measured ratio (min, median, max) | 0.994, 1.006, 1.042 |
| Measured-nonzero predicted-zero | 0 |
| Recall of measured responding sensors | 1.00 |
| `retrodiction_passed` | **true** |

Occlusion-free prefilter over-prediction, for calibration: 31.8x (inbound vessel), 6.2x (outbound
vessel), 1.33x (panel).

## W2 — siting by angular subtense

`scripts/run_pact_place_v9_w2_cluster_siting.py`. Artifact:
`diagnostics_output/pact_place_v9_w2_cluster_siting/siting.json`.

| | inbound cluster | outbound cluster |
|---|---:|---:|
| Placements scored | 7,140 | 7,140 |
| Geometry-feasible | 341 | 108 |
| Sensing-admitted (>= 3 sensors at 2 px in every variant, link5/link6 responder, <= 4x imbalance) | **0** | 54 |
| Best admitted: worst-case sensors / sensor-frames / imbalance | — | 4 / 290 / 2.6x |

763 inbound placements clear the sensing floor (up to 6 sensors, 333 sensor-frames); none is
geometry-feasible.

## W2b — the corridor budget

`scripts/run_pact_place_v9_w2b_inbound_diagnostic.py`.

| quantity | value |
|---|---:|
| Usable aperture width | 0.810 m |
| Free band left by the active panel | 0.460 m |
| Loaded transport envelope + clearance | 0.340 m |
| **Max hazard width still leaving a lane** | **0.120 m** |
| **Contiguous silhouette the skin resolves at working range** | **0.250 m** |
| Shortfall | 0.130 m |
| Max bench-standing height clearing the panel's underside | 0.070 m |
| Width needed for a balanced 3-sensor inbound response deeper than the panel | 0.600 m |
| Inbound x outbound cluster pairs examined / jointly feasible | 18,393 / **0** |

## E0 — V9.5 record correction

`diagnostics_output/pact_place_v95_v0c5_raw_prerequisite/admission_correction.json`. No V9.5 artifact
was edited; all 34 are byte-identical and the correction is a new record.

| # | family | side | raw_passed | source_physics_clean |
|---|---|---|---|---|
| 0-5 | F0 / F1 / F2 | both | False | True |
| 6 | F3_aperture_side_stagger | left | True | **False** (351 clutter contacts) |
| 7 | F3_aperture_side_stagger | right | False | **False** (2,315 clutter contacts) |

**V9.5 headline corrected: 0 of 6 physics-clean variants passed**, not 1 of 8. The single recorded
pass is the dirty-source episode, and the inbound vessel's only nonzero reading in all of V9.5 (40
values, max 3.1 px at R = 0.11 m) comes from it. `run_pact_place_v9_v0c3_causal_proximity.py` now
fails admission on a dirty source. W1's retrodiction and W2's structural finding are unaffected; the
W3 comparison below used F3 as both baseline and test and is flagged accordingly.

## W3 — pipeline validation only, on a configuration W2 rejected

`scripts/run_pact_place_v96_cluster_causal_proximity.py`, floor pre-registered in `config.json`.
Artifact: `diagnostics_output/pact_place_v96_w3_pipeline_validation/validation.json` (`passed: false`).

Panel causal effect on the V9.6 twelve-slot scene: 58,416 left and 23,508 right — identical value
for value to the V9.5 measurement across a change in `nq`.

Both sides of this comparison are F3, whose source physics is dirty (see E0):

| F3 left, V9.5 decision window | V9.5 single vessel | V9.6 cluster |
|---|---|---|
| inbound | 40 changed values, 1 sensor | 2,604 changed values, 3 sensors |
| outbound | 508 changed values, 2 sensors | 644 changed values, 3 sensors |

Right side, inbound cluster: 16 changed values on 1 sensor, 162.8x left/right imbalance. Posed span
and gap, measured from renderable meshes: 0.299 m / 0.021 m and 0.296 m / 0.000 m, both inside the
0.25 m / 0.04 m contract.

---

# V9.8 ceiling pendant environment gate

The implementation and pre-registered bounds are in
[`docs/PACT_PLACE_V98_PENDANT_PLAN.md`](docs/PACT_PLACE_V98_PENDANT_PLAN.md).
This gate ended before collection or training.

| stage | result |
|---|---|
| S1 siting | 77 candidates; selected bottom 1.10 m / half-width 0.18 m |
| S1 worst-case sensing | 14 sensors at 2 px; 181 route-intrusion frames |
| S2a causal smoke | six clean variants; repeat baseline 0.0; smoke only |
| S3 expert screen | **0/24 clean**, 24/24 with clutter contact |
| frozen bar | 20/24 clean successes |
| verdict | `authorizes_gate: false`; `authorizes_collection: false` |

Artifacts: `diagnostics_output/pact_place_v98_pendant_siting/siting.json`,
`diagnostics_output/pact_place_v98_pendant_causal_smoke3/validation.json`,
`diagnostics_output/pact_place_v98_expert_gate/expert_screen.json`, and
`diagnostics_output/pact_place_v98_pendant_review/review.json`.

S2b raw admission was not run after the S3 stop. The mandatory review wrapper
records that three clean successes were unavailable. The resolving-power caveat
is unchanged: the skin resolves a contiguous silhouette of roughly 0.25 m and
nothing smaller; the pendant is 0.30 m wide because of that floor. Ground
clutter remains RGB-only decor that counts as failure on contact, with no claim
that the skin sees it.

## Why it stops

The paired-side design requires identical clutter under both panel sides; the panel forces the arm's
lane to a side that flips between the two rows of a pair; a hazard must be lateral to be sensed at
all, because the gripper, hand and link7 carry no sensors; and a lateral hazard does not flip. The
status doc lists the five choices this leaves, all of which change something the plan froze.
