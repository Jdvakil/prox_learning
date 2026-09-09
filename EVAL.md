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
Collection and training are not authorized.

The original 0/24 S3 screen is a **void measurement of a mis-wired expert**,
superseded and retained at
`diagnostics_output/pact_place_v98_expert_gate/`. Nothing in that run reached
the pendant.

The repair pinned every row to seed 955339 and first measured the TCP-only
lateral bow at the original S1 width (`half_y = 0.18`): fixture-free **6/8**,
pendant+bow **0/8** (clutter during a 0.14–0.16 m inbound detour), pinned-seed
gate **0/24**. That 0/8 mixed a zero-slack detour with the fixture. With the
bow off, a same-run V9.5 replay is still **6/8 row for row**, and the same
eight rows plus pendant are **0/8** on `mounted_fixture` contact (20–257
entries, clutter 0, inbound bow 0.0 m).

S1 v2 then required fixture-bow detour slack ≥ 20 mm and selected
`half_y = 0.16` (`diagnostics_output/pact_place_v98_pendant_siting_v2/`).
A same-run V9.5 replay remains **6/8**
(`diagnostics_output/pact_place_v95_smoke_repro_guard_v4/`). Sweeping the
paired eight at `half_y` 0.16 / 0.14 / 0.12 with the bow **on** is **0/8** at
every width: unclipped waypoints, clutter 0, `mounted_fixture` contact on
every complete row.

Contract v2 then offset the pendant to `y = 0.100` with side-dependent
ceiling envelopes (lag + 4 mm). Step 0 passed both named candidates against
live `_bow_segment` (algebra, dispatch, and no clipping — not swept-arm
clearance). The V9.5 guard after that change is still **6/8
row-for-row** (`guard_v5`). Neither offset candidate preserves every
baseline-clean row (wide `half_y=0.056` and conservative `half_y=0.045`
are both 0/8 on the paired join: unclipped, correct bow side, clutter 0,
`mounted_fixture` contact). The 24-row gate was not run. The aperture is
not widened.

A 2026-08-25 retained-qpos audit
(`diagnostics_output/pact_place_v98_offset_contact_diagnosis/diagnosis.json`)
classifies the offset stop as `route_composition_coverage_failure`:
baseline-clean contact starts outside the protected ceiling-fixture
approach/pass/exit set (left post-bow `pregrasp`; right
`inbound_vessel_pass` / `inbound_cross_vessel_pass`). Terminal phase is
not onset. The 0.208 / 0.108 constants are `unverified_provenance` and
the `[0.044, 0.156]` face window is `physical_input_invalid`. That audit
supersedes the earlier claim that the wrist failed to follow the TCP bow.
It does not establish that a wrist-height hazard is unavoidable in the
0.85 m aperture. `authorizes_gate: false`; `authorizes_collection: false`.

| stage | result |
|---|---|
| S1 siting | 77 candidates; selected bottom 1.10 m / half-width 0.18 m (zero slack) |
| S1 v2 (detour slack ≥ 20 mm) | 40/77 eligible; selected bottom 1.10 m / half-width 0.16 m; 0.18 ineligible |
| S1 worst-case sensing (v2 select) | 13 sensors at 2 px; 181 route-intrusion frames |
| S2a causal smoke | six clean variants; repeat baseline 0.0; smoke only (`half_y = 0.18`) |
| first S3 | **void**: 0/24, expert routing dead |
| V9.5 regression guard (v3, v4, v5) | **6/8**, row-for-row match |
| paired V9.5 + pendant, bow on, 0.18 | **0/8**, clutter during inbound detour |
| paired V9.5 + pendant, bow off, 0.18 | **0/8**, `mounted_fixture` contact, bow 0.0 m |
| paired, bow on, `half_y` 0.16 / 0.14 / 0.12 | **0/8** each; unclipped bow; `mounted_fixture` contact |
| Step 0 offset predictor | both named candidates pass live `_bow_segment` |
| paired offset wide (`y=0.100`, `half_y=0.056`) | **0/8**; baseline-clean rows not preserved |
| paired offset cons (`y=0.100`, `half_y=0.045`) | **0/8**; baseline-clean rows not preserved |
| retained-qpos offset diagnosis | `route_composition_coverage_failure`; lag `unverified_provenance` |
| S3 re-screen (pinned seed, bow on, 0.18) | **0/24 clean**; measures the zero-slack detour |
| frozen bar | 20/24, read as 5 of 6 cells while the seed is pinned |
| verdict | `authorizes_gate: false`; `authorizes_collection: false` |

Artifacts: `diagnostics_output/pact_place_v98_pendant_siting/siting.json`,
`diagnostics_output/pact_place_v98_pendant_siting_v2/siting.json`,
`diagnostics_output/pact_place_v98_pendant_causal_smoke3/validation.json`,
`diagnostics_output/pact_place_v98_expert_gate/expert_screen.json` (void),
`diagnostics_output/pact_place_v98_expert_gate_v2/expert_screen.json` (bow on, 0.18),
`diagnostics_output/pact_place_v98_v95_pendant_paired/paired.json` (bow on, 0.18),
`diagnostics_output/pact_place_v98_v95_pendant_paired_nobow/paired.json`,
`diagnostics_output/pact_place_v98_paired_halfy016/paired.json`,
`diagnostics_output/pact_place_v98_paired_halfy014/paired.json`,
`diagnostics_output/pact_place_v98_paired_halfy012/paired.json`,
`diagnostics_output/pact_place_v95_smoke_repro_guard_v3/guard.json`,
`diagnostics_output/pact_place_v95_smoke_repro_guard_v4/guard.json`,
`diagnostics_output/pact_place_v95_smoke_repro_guard_v5/guard.json`,
`diagnostics_output/pact_place_v98_bow_clearance_predict.json`,
`diagnostics_output/pact_place_v98_paired_offset_wide/paired.json`,
`diagnostics_output/pact_place_v98_paired_offset_cons/paired.json`,
`diagnostics_output/pact_place_v98_offset_contact_diagnosis/diagnosis.json`,
and
`diagnostics_output/pact_place_v98_pendant_review_v2/review.json`.

S2b raw admission was not run. The 24-row offset gate was not run.
The review wrapper records that three clean successes were unavailable. The
resolving-power caveat is unchanged: the skin resolves a contiguous
silhouette of roughly 0.25 m and nothing smaller; the first S1-selected
pendant is 0.36 m wide (`half_y = 0.18 m`) because of that floor, and that
width is zero-slack. Narrower widths with slack still fail on the pendant.
Ground clutter remains RGB-only decor that counts as failure on contact, with
no claim that the skin sees it.

A fixture-free 24-seed sweep of the V9.5 8-row smoke pattern, independent of
the pendant stop, averaged **4.08/8 clean (51%)**, range 0–7. Only **4 of 24
seeds reach 7/8**, including 955339 at 7/8 (the smoke and both regression
guards measured that seed at 6/8; unreconciled). A canonical varied-seed
24-row screen would expect **~12/24 against a bar of 20**, so the seed is
narrow (`narrow: true`). Artifact:
`diagnostics_output/pact_place_v95_seed_fragility/fragility.json`.

# V9.9 fixed pendant qualification

The plan is [`docs/PACT_PLACE_V99_PENDANT_PLAN.md`](docs/PACT_PLACE_V99_PENDANT_PLAN.md).
This is a new engineering qualification. It does not reopen V9.8.

Independent reconstruction of the frozen eight V9.5 rows succeeded (max TCP
residual 0.87 mm < 1 mm). Reconstruction NPZ paths and SHA-256 values are
recorded in `reconstruction.json`. The conservative AABB filter certified
**0 candidates**. AABB overlap is only a broad-phase screen.

The final float64 exact retained-qpos close-out verified reconstruction,
source-row, scene, and implementation hashes, scored all four predicates on
all six cells without early termination, and found **0 survivors** among
12880 dual-transit AABB hits (overlapping rejects: inbound_no_contact=4209,
outbound_no_contact=84, grasp_below_nominal=12880, initial_contact=0). Live
`mj_forward` contact-parity passed. Routing, paired screens, the 24-row
gate, S2b, and human review were not run. The lattice and thresholds were
not relaxed.

V9.9 is permanently closed. Scoped conclusion: **no survivor in the
registered fixed rectangular-box lattice.** `authorizes_collection: false`.
`authorizes_paired_screen: false`. Artifact:
`diagnostics_output/pact_place_v99_siting/siting.json`.

# V10 connected compound pendant qualification

The plan is [`docs/PACT_PLACE_V10_COMPOUND_PENDANT_PLAN.md`](docs/PACT_PLACE_V10_COMPOUND_PENDANT_PLAN.md).
This is a new shape-family qualification. It does not reopen V9.9 or weaken
its scoped conclusion.

V10 searches a registered two-lobe (then, only if empty, three-lobe)
connected compound pendant on the frozen six-cell V9.5 layout. The v1
8,554,036-row catalog is a superseded robot/target prefilter
(`diagnostics_output/pact_place_v10_siting/siting.json`, payload SHA
`923c9380319b343e43e55f018080995db8a4b59e5a5f3cbd7f5d1a3be79d0eb6`;
catalog SHA
`63369af3552bbb806a61fea97d281011374ee25bb375004876704b920b6f3443`).
Corrected siting v2 found 150,288 full-environment exact two-lobe survivors
after an independently reproduced 150,288 panel-clear rows (1,779 unique
union AABBs)
(`diagnostics_output/pact_place_v10_siting_v2/siting.json`, payload SHA
`2e0b2a56bd4c22ecc920927dc149adf9c1bbc0d1d3ccbd3ee433ea450b187c1c`;
catalog SHA
`b84e19bf269c39cd052551639c22d4cbb5b4348eaf6188663fae0659af824d6e`).
Planning-probe v1 is retained only as `probe_v1_invalid_panel_overlap`.
Planning-probe v2 is the trust anchor. Verified scene hashes: V5
`5ac1ebd3e04f0bf509f6b8e11f0d086ac8c43bd550349762aba6c4129aebd61c`;
V10
`360b1407a01d1447d8b440ade3115866399a1db09efc76321016aa3c04eaddf7`.
V9.9 reconstruction and snapshot hashes are unchanged. Siting v2 closed
geometry with `three_lobe.searched: false`, `routing_run: false`, and
`stop_reason: exact_survivors_route_not_run`. Offline routing then ran on
that frozen catalog
(`diagnostics_output/pact_place_v10_route/route.json`, schema
`pact_place_v10_route_v1`, payload SHA
`c0f1b35084d6950a88531c45e6805b06437add31c82ca5fd68bb5da4f5de3ff7`; mask SHA
`6c2609d11dccb69537970fbd7decc2e1b4efc8c2b83275fe3ef65275728a8274`).
Verified 150,288 rows and 1,779 unions. All 21,348 union×cell×direction
evaluations failed detour/clipping/wrong-way (lane construction 0, nominal
IK 0, strict environment 0, pendant clearance 0). Route-surviving unions:
0. Route-surviving morphologies: 0. `routing_run: true`.
`physics_stepped: false`. `episodes_run: false`. `signal_screen_run: false`.
`three_lobe_searched: false`. `v10_closed: true`. `stop_reason:
no_two_lobe_route_survivor`. Signal screening did not run: the
route-surviving population is empty, so the preregistered complete-screen
limit was not reached (`REGISTERED_COMPLETE_SIGNAL_SCREEN_LIMIT` remains
`None`; no post-hoc shortlist). Route-v1 remains a valid scoped historical
result under contiguous-group-freeze and is not relabeled as an error. A
separately registered endpoint-only amendment
(`diagnostics_output/pact_place_v10_route_v2/route.json`, schema
`pact_place_v10_route_v2_endpoint_only`, payload SHA
`e311ba01c77c14b3a930be8dd9d4d40e9de483710521f8662d2e3a55357f71e1`;
geometry SHA
`48e643e6d6b768b3a2dba491c3199c859c8ba1287a75379e504a09b5fdefc74a`; mask SHA
`a20d2e6c0ced5e0807103b3f0f7050a798a5c89c447945fe9898460f2e286bb1`)
independently reproduced 17,826 of 21,348 geometry evaluations and 1,032 of
1,779 unions on all six cells and both directions, then exhausted inbound
lane/padding identities: 666,448 nominal IK passed, 666,448 strict-
environment failed, 0 robust corners, 0 pendant-clearance evaluations,
0 alternative-route recoveries, 0 route-surviving unions, 0 route-surviving
morphologies. `stop_reason: no_route_v2_ik_clearance_survivor`. That
route-v2 stop is a historical result of the flawed scalar environment
predicate and is **not** physical infeasibility. `all_eight_corners_evaluated:
not_applicable`. `routing_run: true`. `physics_stepped: false`.
`episodes_run: false`. `signal_screen_run: false`. `three_lobe_searched:
false`. `v10_closed: true` for the registered offline-search question.
Signal screening did not run; no complete-screen limit or shortlist was
created. Paired screens, collection, training, and ACT/PACT evaluation
remain unauthorized. `authorizes_collection: false`.

# V10.1 empirical pendant qualification

The plan is [`docs/PACT_PLACE_V101_EMPIRICAL_QUALIFICATION_PLAN.md`](docs/PACT_PLACE_V101_EMPIRICAL_QUALIFICATION_PLAN.md).
This is an empirical qualification of the frozen V10 probe_v2 two-lobe
assembly under endpoint-only live-contact routing. It does not reopen V9.9
or V10 siting, does not search alternative lanes or pendants, and does not
treat route-v2 as physical infeasibility.

Contract `pact_place_v101_empirical_qualification_v1` SHA
`c550badd4a95bb0f46c84744ca9cb52b6fd5aa4290377edb8aacc2458087873b`.
Review stream `pact-place-v10.1-pendant-human-review`, master seed
`2026091002`. Gate stream `pact-place-v10.1-pendant-phase0`, master seed
`2026091001` (gate not run). Preflight artifact SHA
`37f1d3739376a0fe8b84c7822ac5aeb65ac1fff61a2b84f91540bdf3ac89f7f3`:
protected V9.9/V10 hashes matched; fixed route admitted **12/12** on V9.9
snapshot stock TCP. Review manifest SHA
`3a472dc165b0053766478dbc3f9e64af54b66e324ec6abd522bde9840675c841`.
12/12 rows reconciled, 0 infrastructure failures, 11 complete, 1
sampling failure, **0/12 clean successes**, 0/2 clean in every F0/F1/F2
family×side cell. `eligible_for_human_review: false`.
`authorizes_gate: false`. Gallery
`diagnostics_output/pact_place_v101_empirical_review/videos/` rendered all
12 rows. Complete-row route telemetry: `endpoint_only` /
`empirical_live_contact_v1`, no fallback/clip/wrong-way/endpoint mutation,
offline strict-environment preclearance intentionally unused, zero
pendant/`mounted_fixture` contact. Causes: 8 `terminal_ik_cascade`, 3
`clutter_collision_stability_event`, 1 F2-left `sampling_failure` (missing
telemetry). Causal artifact SHA
`30329d737be32663b93802c88ba6ced22a121e06b727ca2baa947747018b9364`:
`causal_passed: false`, `blocks_phase0: true`, six `missing_clean_cell`
and six `missing_side_window` codes, no `env.step`. Failure table SHA
`74688c41674a15be909c2889b921e7809daa2fcc5a3cc75805d29e8e78fc1057`.
`human_approval.json` is absent; Phase 0 was not run.
`authorizes_collection: false`. `authorizes_training: false`.
`authorizes_evaluation: false`.

# V10.2 raised, collision-legible pendant

The plan is [`docs/PACT_PLACE_V102_RAISED_PENDANT_REMEDIATION_PLAN.md`](docs/PACT_PLACE_V102_RAISED_PENDANT_REMEDIATION_PLAN.md).
V10.2 raises the frozen two-lobe pendant to a lowest point of `1.10 m`,
thickens both stems and the crossbar x face to a 12 mm square in collision and
visible geometry, assigns TCP speed per named route piece, and renders review
video in real time. It stopped at Step 0. No episode was generated.

Contract `pact_place_v102_raised_pendant_v1` SHA
`16f4c263d3b0310788b27e51303f0aa3feed0241e2c09ba254a69de25eb29a8b`.
Implementation SHA
`c061bc50c4bd9a13c40250fdd081f0c0286a84e7e1ac619a0ba306b7d2f708e6`.
Assembly self-SHA `0751a8d4850994e59f0486bd46f411018608ccb37105a8c0c03e03cbecccdb27`.
Speed-schedule SHA `b9c17c5022780d8820bfff57db17f2e6715aa0d10bc2a025c25c21ce0a1e7d32`.
Preflight artifact SHA
`6c5079916775e8a2093defb1547a3fa85ef9b32dcc4fddcf785ffa6c3276976d`.
Contact-parity root cause artifact SHA
`e4e544a999534b322d177d8b296aa4c7580d9b7627bf89bf0d42630fdd0774df`.

| Step-0 item | Required | Measured |
|---|---|---|
| 1 protected artifacts and scene hashes | all match | **17/17 match** |
| 2 raised assembly vs panel/clutter/hood/initial state | clear on 6 cells | **clear on 6/6** |
| 3 exact stock-route necessity | 12/12 obstructed | **12/12** |
| 4 fixed endpoint-only route geometry | 12/12 admitted | **12/12** |
| 5 complete sequential IK | 12/12 cases | **5/12** (all inbound) |
| 6 per-component pendant clearance ≥ 15 mm | 12/12 cases | **0/12** |
| 7 deliberate stem contact parity | contact observed | **0 contacts at 19 penetrating poses** |

Erratum recorded under V10.3: an earlier version of this table said item 5
was 4/12. The immutable preflight payload says **5/12** and is unchanged.

Item 6 is the decisive result. Minimum robot/target-to-pendant clearance along
the nominal retained-qpos route, per cell and direction, floor `0.015 m`
(negative = penetration depth):

| family | side | inbound worst (m) | outbound worst (m) | worst component |
|---|---|---:|---:|---|
| F0 | left | −0.0676 | −0.0761 | `lobe_0` |
| F0 | right | −0.0651 | −0.0561 | `lobe_1` |
| F1 | left | −0.0769 | −0.0763 | `lobe_0` |
| F1 | right | −0.0665 | −0.0564 | `lobe_1` |
| F2 | left | −0.0748 | −0.0763 | `lobe_0` |
| F2 | right | −0.0666 | −0.0565 | `lobe_1` |

Same-side stems are penetrated by 26–52 mm on the same cases. The crossbar
(0.266–0.279 m) and the opposite-side lobe and stem are clear everywhere.
Raising the lobe bottoms to `1.10 m` moved the pendant into the arm's
elbow/forearm envelope, which a frozen-endpoint TCP lane rewrite cannot
sidestep; `LOBE_TOP_MAX_M = 1.10` in the V10 lattice is exactly this boundary.

Item 7's failure is an instrument finding, verified separately: 19 deliberately
penetrating stem poses (GJK −69 mm to −17 mm) produced **0** `data.contact`
entries in a scene simultaneously carrying 83 other contacts, because
`pose_assembly_geoms` leaves `geom_aabb`, `geom_rbound` and the pendant body's
`bvh_aabb` at compile-time 1 mm placeholders. Refreshing only those bounds makes
the contact appear at `dist = −0.016630 m`, matching hardened GJK. Recorded
"zero `mounted_fixture` contact" for the runtime-posed V10 pendant, V10.1
included, is therefore not evidence of clearance. Nothing was repaired: that
would change V10/V10.1 behaviour, which the V10.2 plan forbids.

`preflight_passed: false`. `stop: true`. Six-row screen, 12-row review, causal
replay, and 24-row Phase 0 were **not run**. `human_approval.json` is absent and
was not inferred. `authorizes_gate: false`. `authorizes_collection: false`.
`authorizes_training: false`. `authorizes_evaluation: false`.

**Owner-requested diagnostic gallery (not the gate).** After the Step-0 stop the
owner asked to see the twelve clips anyway. They were produced by
`scripts/run_pact_place_v102_diagnostic_gallery.py` into
`diagnostics_output/pact_place_v102_diagnostic_gallery/` (manifest SHA
`9d0f7b4b6c8adc51261cf58bbed26a12ec68b5b62bfd1c446ac122c94918c0a1`,
`is_registered_review: false`, `eligible_for_human_review: false`, every
authorization false). This is **not** the registered review pack, is not an
input to causal proximity or Phase 0, and must never be cited as one; the
registered review path was not written and the Step-0 stop is untouched. The
gallery runner is not part of the V10.2 implementation hash, which is unchanged.

The twelve rows ran the real expert with the live contact audit. All 12
reconciled and completed; **0/12** are clean; 11 end in `terminal_ik_cascade`
and one in `clutter_collision_contact`. Every row's minimum per-frame
robot/target-to-pendant clearance is **negative** — −0.0008 m to −0.0742 m —
against the 15 mm floor, and across all twelve rows there are **zero** live
pendant-contact frames and **zero** `mounted_fixture` contact entries. That
pairing is the two Step-0 findings reproduced on live rollouts: the arm passes
through the raised pendant, and the contact pipeline cannot see it. The speed
schedule is visible and correct in the telemetry —
`inbound_pendant_approach` commanded at 0.15 m/s and `inbound_pendant_pass` at
0.045 m/s, against an inherited 0.20 m/s for both.

The clips render every policy frame at `1000/66` fps, so they play in real time.
The registered renderer's pendant pane is aimed from `y = -1.15 m`, outside the
hood, where `hood_side_r` occludes it; the gallery overrides that camera
in-process to a pose inside the aperture and records the override in its
manifest under `pendant_side_camera_override`. Any successor version should
adopt the corrected pose.

# V10.3 static-pendant joint-route qualification

The plan is [`docs/PACT_PLACE_V103_STATIC_PENDANT_IK_PHASE0_PLAN.md`](docs/PACT_PLACE_V103_STATIC_PENDANT_IK_PHASE0_PLAN.md).
V10.3 compiles the pendant statically into its own scene and plans a continuous
joint-space route pinned to the retained qpos at both ends. It stopped at Step 0B
under an owner-approved early stop. No episode was generated.

Contract `pact_place_v103_static_pendant_joint_route_v1`. Stop record
`diagnostics_output/pact_place_v103_ik_search/search.json`, artifact SHA
`f06feaa3c09d5f95a006f66d00e45c8684962393967a1acc3ad40d21dc23df98`. Endpoint
certificate `diagnostics_output/pact_place_v103_ik_search/endpoint_certificate.json`,
artifact SHA `3ced3a35b71ac7a1cc9f94ab23549b0764dceef07508ab5302791e592c062fda`.

| Stage | Required | Measured |
|---|---|---|
| Step 0A tests | pass before search | **34/34 pass** |
| Pre-search regressions | no new failures | **302 pass**, 3 pre-existing unrelated |
| Step 0B cases | 12 evaluated | **9 completed, 3 `not_evaluated`** |
| Completed cases with any feasible route | — | **0 / 9** |
| Geometry routing all 12 cases | ≥ 1 | **0** |
| Step 0C / smoke / review / causal / Phase 0 | — | **not run** |

Search lattice, searched once and never extended: heights
`0.92 / 0.96 / 1.00 / 1.04 m`; lanes `0.28 / 0.30 / 0.32 m`; staging buffers
`0.10 / 0.12 m`; pass-z `-0.06 / -0.04 / -0.02 / 0.00 m`; five pass orientations
per side (120 templates per side); 24 fixed scrambled-Halton seeds plus three
fixed seeds and both adjacent layers; node/edge floor `0.020 m`, corner floor
`0.015 m`, interpolation step `0.01 rad`.

The conclusive witness is the pinned inbound endpoint on the three left cells.
All 24 measurements (3 cells × 4 heights × 2 retained frames) penetrate the
negative lobe; node-clearance margin is `-0.02000 m` in every one. Signed
penetration at the pinned pregrasp frame, and the robot part responsible:

| height (m) | F0 left | F1 left | F2 left | contacting geom |
|---:|---:|---:|---:|---|
| 0.92 | -0.00829 | -0.00043 | -0.00220 | `robot_0/gripper/base` |
| 0.96 | -0.01431 | -0.00575 | -0.00735 | `robot_0/gripper/base` |
| 1.00 | -0.02282 | -0.01478 | -0.01633 | `robot_0/fr3_link7_collision` |
| 1.04 | -0.02445 | -0.01996 | -0.02073 | `robot_0/fr3_link6_collision` |

Analytic exact GJK returns `0.00000 m` (intersecting) for all 24; the signed
hardened `mj_geomDistance` above agrees. The last retained inbound frame is the
same or worse. Low lobes are struck by the hand, high lobes by the wrist, and the
lattice contains no z window between them — so the V10.2 failure was not specific
to the 1.10 m height.

`search_exhaustive: false`. `every_registered_template_evaluated: false`.
`stop_reason: no_static_geometry_with_twelve_joint_routes`.
`global_conclusion_conclusive: true`.
`conclusive_witness: pinned_endpoint_clearance_below_node_floor`.
`remaining_cases_cannot_change_selection: true`. `episodes_generated: 0`.
`env_step_called: false`. `runtime_built: false`. `authorizes_gate: false`.
`authorizes_collection: false`. `authorizes_training: false`.
`authorizes_evaluation: false`. `phase0_passed: false`.

# V10.4 first-shot static pendant

The plan is [`docs/PACT_PLACE_V104_FIRST_SHOT_STATIC_PENDANT_PLAN.md`](docs/PACT_PLACE_V104_FIRST_SHOT_STATIC_PENDANT_PLAN.md).
V10.4 builds on the qualified V6c environment and adds one compiled-static
two-lobe pendant outboard at `|y| = 0.34 m`, lobes at `z = 0.98–1.04 m`, plus a
single registered speed cap. Steps 0, 1 and 2 passed; Step 3 stopped on the
plan's own control rule.

| Stage | Required | Measured |
|---|---|---|
| Step 0A provenance | all match | **61/61** byte-level |
| Step 0B V9.5 anchor | ≥ 0.035 m, ≈ 0.04052 | **0.04052 m** (delta −0.00000) |
| Step 0B V6c anchor | 24/24 ≥ 0.050 m, ≈ 0.05523 | **0.05523 m**, **24/24** |
| Step 0B corners | ≥ 0.030 / ≥ 0.040 m | **0.03337 / 0.04800 m** |
| Step 0C static + initial state | no disallowed overlap | pass; only the designed `crossbar`/`hood_top` face touches |
| Step 0D contact parity | all instruments agree | **30/30** |
| Step 0E route preservation | 24/24 identical, one speed change | **24/24**, max 583 steps vs 840 |
| Step 1 production | ≥ 5/6 clean, ≥ 2/3 per side | **6/6**, 3/3 per side |
| Step 1 pendant contact | zero | **zero in all six rows** |
| Step 1 min clearance | ≥ 0.020 m | **0.06158 m** |
| Step 2 causal, per side | ≥ 448 and ≥ 7,209 changed values | **23,684** left, **12,712** right |
| Step 2 side ratio | ≤ 4× | **1.86×** |
| Step 3 review packet (v1) | 6 videos | **not produced — stopped** |
| Step 3 review packet (v2) | 6 videos | **6 published** |
| Step 4 Phase 0 | after approval only | **not run** |

Six production rows, all strict-clean:

| row | side | min pendant clearance | contact frames | steps |
|---:|---|---:|---:|---:|
| 0 | left | 0.06158 m | 0 | 542 |
| 1 | right | 0.08070 m | 0 | 426 |
| 2 | left | 0.08069 m | 0 | 607 |
| 3 | right | 0.07955 m | 0 | 455 |
| 4 | left | 0.07070 m | 0 | 628 |
| 5 | right | 0.08494 m | 0 | 438 |

Step 3 stop: with no natural production failures, all three failure clips had to
be diagnostic negative controls. `left_lobe_contact` reaches only **2.579 mm**
penetration at the end of the frozen `0.000–0.160 m` grid (161 shifts recorded),
below the registered 5–30 mm band. The grid was not extended and no substitute
control or source row was chosen.

Artifacts: preflight `fe64e285…`, production `fdcf757b…`, causal `a30c863d…`,
Step-3 stop `9a4abfec…`, contract `455379b8…`, scene `01d8adf3…`.

Tests: 32 V10.4 tests passing; regression sweep 334 passed with the three known
stale-scene-hash corridor failures, the V3 scene bytes independently confirmed
unchanged by byte-level provenance.

`eligible_for_human_review: false`. `human_approval_present: false`.
`authorizes_phase0: false`. `authorizes_gate: false`.
`authorizes_collection: false`. `authorizes_training: false`.
`authorizes_evaluation: false`. `phase0_passed: false`.

# V10.4 review-v2 diagnostic-control repair

The plan is [`docs/PACT_PLACE_V104_REVIEW_REPAIR_PLAN.md`](docs/PACT_PLACE_V104_REVIEW_REPAIR_PLAN.md).
The six V10.4 production episodes are reused byte-for-byte through a scoped
provenance bridge; no episode, `env.step`, or Phase-0 row was run.

| Check | Required | Measured |
|---|---|---|
| v1 payload hashes | 3/3 recomputed match | **3/3** |
| v1 raw file hashes | 3/3 match | **3/3** |
| scene + metadata | byte-identical | **both match** |
| scoped implementation | all bound files match | **15 bound, 14 match, 1 bridged** |
| bridge allowlist size | exactly 1 | **1** (`run_pact_place_v104_review_video.py`) |
| production rows | 6/6 strict-clean reconciled | **6/6**, 3 left / 3 right |
| replacement episodes | zero | **zero** |
| controls certified | 3/3 before any render | **3/3** |
| anchor reproduction | within 0.1 mm | **3/3 exact** |
| four-instrument parity | agree at certified frame | **3/3** |
| videos published | exactly 6 | **6** |
| decode verification | frames, fps, duration, size, SHA | **6/6 pass** |
| Phase0-v2 without approval | refuse, create nothing | **refused**, no gate dir |

Diagnostic controls, on separately compiled scenes (grid `0.000–0.200 m`, 201
points, whole assembly rigidly translated inward along y):

| control | component | source row | shift | penetration | max frame | limiting body | window |
|---|---|---|---:|---:|---:|---|---|
| left lobe | `lobe_0` | 0 (left) | 0.175 m | 5.044 mm | 88 | `fr3_link7` | 40–89 (50 f) |
| right lobe | `lobe_1` | 3 (right) | 0.132 m | 5.239 mm | 245 | `gripper/base` | 197–260 (64 f) |
| stem | `stem_0` | 0 (left) | 0.083 m | 5.455 mm | 212 | `fr3_link7` | 164–227 (64 f) |

Six published videos, stride 1 at 15.1515 fps, true time:

| video | kind | frames | duration |
|---|---|---:|---:|
| `success_00_left.mp4` | production clean success | 543 | 35.84 s |
| `success_03_right.mp4` | production clean success | 456 | 30.10 s |
| `success_04_left.mp4` | production clean success | 629 | 41.51 s |
| `control_left_lobe_contact.mp4` | diagnostic negative control | 50 | 3.30 s |
| `control_right_lobe_contact.mp4` | diagnostic negative control | 64 | 4.22 s |
| `control_stem_contact.mp4` | diagnostic negative control | 64 | 4.22 s |

Why v1 stopped and v2 does not: the v1 grid ended at 0.160 m and the left-lobe
contact occurs at 0.175 m, and v1 moved only the target geom rather than the
whole assembly. The v1 measurement of 2.579 mm was correct for the grid it
searched; the grid was the defect.

Correction to the plan's stated numbers: the left-lobe stem contact begins at
frame 90 at 0.103 mm and reaches 38.15 mm at frame 193, not 38.15 mm at frame 90.
Both values are real; the certificate records them as separate frames.

Tests: **71** review-v2 tests and **32** V10.4 tests passing. The `pact_place`
sweep is **387 passed, 3 failed**, and those three are exactly the known stale
V3/V5 hash assertions in `tests/test_pact_place_corridor.py` — reported only
after the **61/61** protected byte hashes reproduced unchanged. The full `tests/`
sweep is **1258 passed, 18 failed, 1 skipped**; the 18 are the same three plus 15
pre-existing failures in unrelated areas (checkpoint hashes, slideshow bundles,
oracle contract). No existing source file was modified by this repair — every
change is a new file — so no pre-existing failure is attributable to it, and the
clean sweep contains **zero** V10.4 failures.

`eligible_for_human_review: true` (the review manifest only).
`human_approval_present: false`. `authorizes_phase0: false`.
`authorizes_gate: false`. `authorizes_collection: false`.
`authorizes_training: false`. `authorizes_evaluation: false`.
`phase0_passed: false`.

# V10.5 V9.5 real clutter with a static pendant

The plan is [`docs/PACT_PLACE_V105_V95_CLUTTER_STATIC_PENDANT_PLAN.md`](docs/PACT_PLACE_V105_V95_CLUTTER_STATIC_PENDANT_PLAN.md).
V10.5 restores the settled fixture-free V9.5 household-object clutter (V5 scene,
`PactPlaceCorridorV93Sampler`, movable free bodies) — **not** the V9.5 low wall
— and searches a registered lattice for a pendant placement that is
behaviourally consequential while preserving a 15 mm clearance floor.
**Stopped at Step 2: no candidate survived.**

| Stage | Required | Measured |
|---|---|---|
| Step 0 contract + tests | frozen before search | **67 tests passing** |
| Step 1 corpus agreement | derived == stored | **192/192 agree**, 0 disputed |
| Step 1 clean coverage | ≥ 2 per family/side cell | **98/192 = 51.0%**, min 8/cell |
| Step 1 live replay | TCP residual ≤ 1 mm | **16/16**, 0.067–0.163 mm |
| Step 2 scenes scored | 32 × 3 = 96, no early stop | **96/96** |
| Step 2 trajectories | all recoverable strict-clean | **98/98**, 0 failed |
| Step 2 survivors | ≥ 1 three-pose bundle | **0** |
| Steps 3–6 | only after a selection | **not executed** |

Rejection counts across the 32 bundles:

| predicate | failing |
|---|---:|
| 4 — min clearance ≥ 15 mm on every clean trajectory | **32/32** |
| 8 — inbound and loaded-outbound band witness per side | 30/32 |
| 3 — no robot/carried-target pendant contact on a clean row | 27/32 |
| 7 — a 15–35 mm witness per `pose_id × side` group | 8/32 |
| 1, 2, 5, 6, 9 | 0/32 |

The trade that defeats the lattice:

| bundle | risk-band witnesses | worst clean-row clearance |
|---|---:|---:|
| `x = 0.780, r = 0.325` (most witnesses) | 153 | **3.4 mm** |
| `x = 0.800, r = 0.320` (highest floor) | 133 | **9.4 mm** |
| `x = 0.800, r = 0.325` | 126 | 13.4 mm |

Artifacts: reconstruction payload `ccace4b0…` (raw `71bcb635…`), siting payload
`4c9f5646…` (raw `56f5d6ba…`),
`stop_reason: no_complete_three_pose_bundle_survived_the_lattice`.

A predicate defect was corrected before reporting: gating direction witnesses on
the row's overall minimum suppressed every inbound witness. Corrected,
`left:inbound` appears in 22/32 bundles and `right:inbound` in 5/32. Selection
unchanged; the reported reason would have been wrong. Both runs are preserved.

Registering the V10.5 sampler modified `enclosure_reach.py` and
`run_pact_place_expert_screen.py`, which the V10.4 Step-0 preflight binds, so
the V10.4 review-v2 provenance bridge now fails by design. V10.4's data and six
MP4s are byte-identical; only the implementation binding moved. That packet was
not regenerated.

Tests: 67 V10.5 passing. Full sweep **1317 passed, 26 failed, 1 skipped** — 18
pre-existing, 8 the V10.4 bridge failures described above. `git diff --check`
clean in both repositories.

`eligible_for_human_review: false`. `human_approval_present: false`.
`authorizes_phase0: false`. `authorizes_gate: false`.
`authorizes_collection: false`. `authorizes_conversion: false`.
`authorizes_training: false`. `authorizes_evaluation: false`.
`phase0_passed: false`.

# V10.5 audit and erratum

`scripts/audit_pact_place_v105.py`, artifact
`diagnostics_output/pact_place_v105_audit/audit.json` (payload `93156a32…`,
raw `89cd499b…`). The V10.5 narrative was treated as untrusted.

| check | measured |
|---|---|
| sealed artifacts hashed (raw + canonical payload) | **4/4**, both JSON self-consistent |
| retained corpus rows verified | **192/192**, 0 problems |
| rows without a retained trajectory | 4, **all unclean** — 98/192 unaffected |
| independent aggregator evaluations | **9408** |
| agreement with primary scorer | **192 checks, 0 disagreements** |

| id | claim | status | correction |
|---|---|---|---|
| E1 | 21 active clutter free bodies | **incorrect** | **8** household objects; 21 counted nested mesh child bodies and 4 corridor chicane bodies |
| E2 | `r=0.320` is the highest floor | **incorrect** | highest is `x=0.800, r=0.325` at **13.4388185 mm**, 0 contacts, **2/294** below 15 mm; `r=0.320` is second at 9.3898271 mm |
| E3 | 98/192 strict-clean | **valid** | confirmed |
| E4 | `risk_group_counts` | **ambiguous** | renamed `band_evaluations_by_group` — count of (trajectory, pose) **evaluations** in the 15–35 mm band, per `pose\|side` group |

# V10.6 asymmetric static pendant

The plan is [`docs/PACT_PLACE_V106_ASYMMETRIC_PENDANT_PLAN.md`](docs/PACT_PLACE_V106_ASYMMETRIC_PENDANT_PLAN.md).
Global asymmetric lattice (not per-family placement) on the V10.5 shape.
**Stopped at Step 4b.**

| Stage | Required | Measured |
|---|---|---|
| Step 1 audit | exact agreement | **192/192, 0 disagreements** |
| Step 3 scenes scored | 9 × 3 = 27, all 98 trajectories | **27/27**, **98/98**, 0 failed |
| Step 3 admitted | ≥ 1 | **9/9**, **4 universal** |
| Step 3 selected floor | ≥ 15 mm universal | **18.5703 mm**, 0/294 below floor, 0 contacts |
| Step 4a compiled-static | 3 scenes | **3/3**, bounds enclose, 58 BVH each |
| Step 4a witnesses | all four instruments agree | **6/6**, agreement to 5 decimals |
| Step 4b causal | 7 registered checks | **7/7 pass** |
| Step 4b contact risk | contact ≤ 30 mm, all 6 groups | **0/6 — STOP** |
| Steps 5–6 | after Step 4 passes | **not executed** |

Selected `x = 0.800, r_neg = 0.335, r_pos = 0.305` — the negative side needs
30 mm more radius than the positive side, which no symmetric assembly can
express. The preregistered fallback was not needed.

Causal, selected scene vs compiled no-pendant control, real `[40,4,8,8]` tensor:

| side | changed values | sensors | links | onset | deterministic |
|---|---:|---:|---|---|---|
| left | 4512 | 7 | link2/3/5_back/6 | 60 frames, 3.96 s | yes |
| right | 2004 | 9 | link2/3/4/5_back/6 | 60 frames, 3.96 s | yes |

Side ratio 2.251 against a 4× limit.

Contact-risk certificate, registered 30 mm cap, remaining clearance at 30 mm:
11.904 / 5.224 / 15.953 / **1.018** / 13.054 / 9.719 mm. No group reaches
contact. Three probe defects were found and fixed first (inherited contacts
counted as new; a TCP-to-centre direction that increased clearance; the grasp
counted as a collision); both uncorrected runs are preserved.

Artifacts: audit `93156a32…`, siting `17e803e6…`, certification `ad5ca617…`,
risk/causal `a0127a5e…`.

Tests: 36 V10.6 and 67 V10.5 passing. Full sweep **1353 passed, 26 failed,
1 skipped** — 18 pre-existing, 8 the V10.4 provenance-bridge failures caused by
registering new samplers in the shared `enclosure_reach.py`. No V10.5 or V10.6
test fails. `git diff --check` clean in both repositories.

`eligible_for_human_review: false`. `human_approval_present: false`.
`authorizes_phase0: false`. `authorizes_gate: false`.
`authorizes_collection: false`. `authorizes_conversion: false`.
`authorizes_training: false`. `authorizes_evaluation: false`.
`phase0_passed: false`.

# V10.7 qualification repair

The plan is [`docs/PACT_PLACE_V107_QUALIFICATION_REPAIR_PLAN.md`](docs/PACT_PLACE_V107_QUALIFICATION_REPAIR_PLAN.md).
V10.6 geometry results are historical inputs; nothing prior was modified.
**Steps 1–6 passed. Step 7 stopped at the pool floor.**

| Stage | Required | Measured |
|---|---|---|
| Step 1 specification | bound before execution | **9 inputs + 20 files**, drift guard fired once for real |
| Step 2 selection | risk-aligned, not hardcoded | **`0.800\|0.330\|0.300`**, argmin asserted |
| Step 2 group minima | all six in 15–35 mm | **6/6**: 16.84–22.31 mm |
| Step 3 certification | 4 instruments agree, minima + near | **11 witnesses**, **0 disagreements** |
| Step 4 causality | all six groups | **6/6 pass**, ratio 2.251 |
| Step 7 pool | ≥32/48 and all balance floors | **21/48 — STOP** |
| Step 7 packet | six videos | **not published** |

Selection, four universal candidates ranked risk-first:

| bundle | abs min (mm) | band evals | selected |
|---|---:|---:|---|
| **0.800\|0.330\|0.300** | **16.8435** | **142** | **yes** |
| 0.800\|0.330\|0.305 | 17.4878 | 135 | |
| 0.800\|0.335\|0.300 | 16.8435 | 121 | |
| 0.800\|0.335\|0.305 | 18.5703 | 114 | (V10.6's pick) |

Six-group causality, selected scene vs compiled no-pendant control:

| group | changed values | sensors | onset | deterministic |
|---|---:|---:|---:|---|
| center\|left | 4592 | 7 | 60 | yes |
| center\|right | 2032 | 9 | 60 | yes |
| neg5\|left | 4648 | 7 | 60 | yes |
| neg5\|right | 2080 | 9 | 60 | yes |
| pos5\|left | 4544 | 8 | 60 | yes |
| pos5\|right | 2012 | 9 | 60 | yes |

Pool: **21/48 = 43.8%**, Wilson 95% [30.7%, 57.7%]; by side 13 left / 8 right;
by pose 8 / 6 / 7. Defects: 21 ordinary clutter contact, 12 clutter stability,
11 task, 10 grasp, 4 lift, 1 sampling failure, and **0 pendant contacts**.
Clean rows held 16.052–56.082 mm of pendant clearance.

43.8% brackets the V9.5 corpus's own **51.0%** (98/192). The 32/48 pool floor
and the 16/24 Phase-0 bar are both **66.7%** — above what this expert achieves
on real V9.5 clutter, as the V9.5 fragility artifact already recorded
(~12.25/24 expected against a bar of 20). The floor worked as registered: it
refused a curated packet for an environment that would not pass Phase 0.

Contact perturbation is now a **non-gating diagnostic**. Repaired (rigid
carried target, narrow gripper-pad allowlist, worsened-baseline tracking,
instrument agreement) it reaches contact in **1/6** groups versus 0/6 in V10.6,
and reports worsened baseline penetrations in 6/6 — which is why it does not
gate.

Artifacts: specification `4854d6b1…`, selection `5ecd3aa0…` (NPZ `d248a764…`),
certification `d31d602b…` (NPZ `cab36c63…`), causal `a916a6be…` (NPZ
`3c05c5fb…`), pool `a5cfed9d…`, diagnostic `38b0b358…`.

Three plumbing defects were exposed by running real episodes and each was fixed
with a regression test, with the failed run preserved: the scene-hash guard read
`cfg.scene_xml` instead of `task_sampler_config.scene_xml_paths` (48/48
`sampling_failure`); the retained row did not copy V10.5/V10.6 telemetry keys
(48/48 `missing_frame_telemetry`); and the policy read `_pact_manifest_row`,
which it does not have (null clearance on every episode). A fourth chain halted
correctly on hash drift.

`eligible_for_human_review: false`. `human_approval_present: false`.
`authorizes_phase0: false`. `authorizes_gate: false`.
`authorizes_collection: false`. `authorizes_conversion: false`.
`authorizes_training: false`. `authorizes_evaluation: false`.
`phase0_passed: false`.

## V10.7 owner visual-review packet (review-only)

`diagnostics_output/pact_place_v107_owner_review/` — manifest payload
`b1239ab7718b1d29…`. **Review-only. The pool remains FAILED at 21/48; this
packet does not reinterpret or overwrite that result and authorizes nothing.**

Six complete retained trajectories replayed at true time (15.1515 fps, stride 1,
untrimmed). No episode generated, no task resampled, no `env.step`, no geometry
or threshold change, no pool rerun. Rows were rebuilt from the frozen generator
and each `row_sha256` asserted against the executed row.

Selection is derived, not hardcoded: among **54,846** valid subsets meeting one
row per pendant pose in each outcome class, three left and three right overall,
and ≥2 layout families per class, minimise the maximum pendant clearance, then
total clearance, then the sorted role-index tuple. Objective: max **26.307 mm**,
total 122.605 mm, roles [6, 8, 20, 28, 40, 45].

| video | outcome | role | pose | side | family | min clearance | frames |
|---|---|---:|---|---|---|---:|---:|
| `success_role06_neg5_left.mp4` | clean | 6 | neg5 | left | F1 | 24.248 mm | 577 |
| `success_role28_center_right.mp4` | clean | 28 | center | right | F0 | 26.307 mm | 443 |
| `success_role08_pos5_left.mp4` | clean | 8 | pos5 | left | F1 | 16.052 mm | 544 |
| `failure_role45_neg5_right.mp4` | natural failure | 45 | neg5 | right | F3 | 21.517 mm | 441 |
| `failure_role40_center_right.mp4` | natural failure | 40 | center | right | F2 | 21.399 mm | 448 |
| `failure_role20_pos5_left.mp4` | natural failure | 20 | pos5 | left | F3 | 13.081 mm | 583 |

**All three failures are natural production failures; none is an induced
pendant collision** — each records zero robot-or-target pendant contact frames.

The independent verifier (`scripts/verify_pact_place_v107_owner_review.py`)
reports **0 problems**: bound artifact hashes, scene and assembly hashes, source
result/trajectory hashes, decoded frame counts, fps and durations all reconcile,
and the selection re-derives to the same six roles. 84 targeted tests pass.

One bound file drifted and is reported rather than ignored: the V10.7 plan
document, which records measured outcomes and necessarily changes after the run
it describes. Code and data drift is zero.

`eligible_for_owner_visual_review: true`. `pool_passed: false`.
`authorizes_downstream_work: false`. Every authorization field false.
`human_approval.json` absent and not created.

# V10.8 exploratory collection and V10.9 train/evaluate (owner override)

**Neither reopens V10.7.** Its Phase-0 gate remains **failed at 8/24 and
permanently closed**. V10.8 is an exploratory owner-override collection stopped
early by owner instruction at **141 of 152** target successes, quotas unmet.
V10.9 converts, trains and evaluates on those 141 under explicit owner
authorization. Every authorization field on every V10.8/V10.9 artifact is false.

## V10.8 erratum (create-only; `closeout.json` unmodified)

Seven corrections, each re-derived from `ledger.jsonl` and the retained files:
"no pendant involvement anywhere" was **false** (zero *accepted* rows contacted
it, but rejected attempt `1a756c9304311cdc…` logged 42 `mounted_fixture`
contacts, one pendant-contact frame and zero clearance); `infrastructure_halts:
0` was **false** (eight `BrokenProcessPool` worker deaths, caused by terminating
the pool on the owner's stop instruction — not spontaneous corruption, and
distinct from the 12 `sampling_failure` ledger rows); **17** cells equal quota
and **two** exceed it, five are short; only `F3|right|neg5` (1 episode) cannot
split; the encoder statistics came from **120 windows of one episode**; and
HDF5 `T = episode_steps + 1`, so T runs 356–627 summing to **71,511**.

## V10.9 pipeline verification

| stage | result |
|---|---|
| source freeze | **33/33 checks pass** — 353 rows, 141 accepted all strict-clean, every HDF5 hash matched, T 356–627 sum 71,511, 0 accepted pendant contacts, every seed reproduced from its cell stream |
| conversion | 141 episodes, sensor order `2198e29b…` preserved (front-before-back, verified non-alphabetical) |
| embeddings | 2,854,800 windows, **all finite, 0 dead dimensions**, global std 1.1127, per-dim std 0.249–1.393, 12.72% valid |
| preservation | **141/141 proved** by re-deriving action, qpos, qvel, wrist RGB, raw proximity and extrinsics/intrinsics from the V10.8 source |
| split | 113/28, 24 cells in train, 23 in validation, order-independent |
| training | both arms 2000/2000 epochs, strict reload and offline smoke pass |
| evaluation | 80/80 rollouts, 0 failures, 2.44 h |

## Training

ACT best epoch 1883, validation loss **0.11186**, 72.0 min.
PACT best epoch 1841, validation loss **0.11549**, 76.0 min.
PACT differs only by `--ckpt_dir` and the five proximity flags, verified by a
parsed flag diff that fails closed. Proximity consumption proved causally: real
versus zeroed embeddings on the same batch move PACT's actions by max **1.376**.

## Paired evaluation: 40 held-out instances, 80 rollouts

| endpoint | ACT | PACT | PACT−ACT | 95% CI (pp) | discordant P/A | exact McNemar |
|---|---|---|---|---|---|---|
| task success | 14/40 (35.0%) | 11/40 (27.5%) | −7.5 | [−20.2, +5.2] | 2/5 | 0.4531 |
| **collision-free task success** | **8/40 (20.0%)** | **6/40 (15.0%)** | **−5.0** | **[−16.9, +6.9]** | 2/4 | **0.6875** |
| strict-clean task success | 8/40 | 6/40 | −5.0 | [−16.9, +6.9] | 2/4 | 0.6875 |
| pendant-free task success | 14/40 | 11/40 | −7.5 | [−20.2, +5.2] | 2/5 | 0.4531 |
| intrusion-panel contact episodes | 10/40 | 8/40 | −5.0 | [−18.8, +8.8] | 3/5 | 0.7266 |
| clutter-contact episodes | 17/40 | 18/40 | +2.5 | [−15.2, +20.1] | 7/6 | 1.0000 |

Contact volume (descriptive, not paired): intrusion panel **43,845** entries for
PACT against **140,424** for ACT (3.2× lower); clutter 123,001 against 87,688;
other-environment 994 against 4,118; clutter-stability events 16 each. Pendant
and mounted-fixture contact: **0 frames, 0 episodes, both arms**. Gripper close
commanded in 38/40 (PACT) and 39/40 (ACT). Every episode ran the full 900
control steps (`end_on_success=false`).

Stratified, collision-free task success: by family ACT 1/3/2/2 vs PACT 0/3/1/2
(F0–F3); by side ACT 1 left / 7 right vs PACT 2 / 4; by pose ACT 0/4/4 vs
PACT 1/3/2 (center/neg5/pos5).

## Verdict

**Null.** PACT did not beat ACT on any endpoint. Every paired interval crosses
zero, no McNemar p-value falls below 0.45, and the largest discordant split is
2 versus 5 — a single seed on 40 paired instances cannot resolve a difference of
this size. The historical V5 chunk-100 result (PACT 19/40 vs ACT 13/40 task
success) reversed sign here, but the environment (certified V10.7 static pendant
on real V9.5 clutter, versus the V2 corridor) and the training corpus (141 V10.8
demonstrations, versus 152 V5) both changed, so it is **contextual only, not a
like-for-like comparison**.

The dataset is also **not balanced**: F3 holds 26 of 38 target episodes, sides
run 75 left / 66 right, five cells are short and two over quota. Any per-family
reading of the table above must account for that.

`PACT_PERMUTED` was not run. No superiority claim is made from validation loss,
from a single seed, or from an interval crossing zero.

# V10.10 four-object ACT/PACT retraining and paired evaluation

V10.10 preserves the certified V10.7 static pendant, scenes, route and 40-sensor
suite, but activates exactly four household-clutter objects: `Soap_Bottle_30`,
`Plate_10`, `Plate_22`, and `Soap_Bottle_11`. This is an exploratory owner-run
comparison; it does not reopen V10.7's failed Phase-0 gate and every downstream
authorization field remains false.

## Corpus and training

The collector produced **144/144 strict-clean demonstrations**, six in every
family×side×pose cell, from 313 attempts in 6.50 hours. The deterministic split
is **120/24**, exactly five train and one validation row per cell. Both models
used seed 3101 and the same chunk-100 settings: 2,000 epochs, batch 8, learning
rate 1e-5, KL 10, chunk 100, hidden 512, feed-forward 3,200, 7+7 layers, 8
heads, wrist ResNet-18, state/action 9/8. A parsed-command check verifies that
PACT differs only by its checkpoint directory and the five registered
proximity flags.

| | ACT | PACT |
|---|---:|---:|
| epochs | 2,000/2,000 | 2,000/2,000 |
| best epoch | 1,998 | 1,859 |
| best validation loss | 0.10621 | 0.10386 |
| strict reload / offline smoke | pass | pass |

PACT's proximity input is causally active: zeroing the frozen embeddings on the
same batch changed actions by mean absolute 0.204 and maximum absolute 1.852.
Validation loss is not an evaluation endpoint.

## Paired held-out evaluation

The repaired infrastructure run completed **80/80 rollouts with zero failures**
over 40 paired instances in 5.746 hours. All ACT/PACT pairs had byte-identical
initial observations across all 421 retained observation datasets.

| endpoint | ACT | PACT | PACT−ACT | paired 95% CI (pp) | exact McNemar |
|---|---:|---:|---:|---:|---:|
| task success | 11/40 | 14/40 | +7.5 | [−10.01, +25.01] | 0.5811 |
| **collision-free task success (primary)** | **7/40** | **10/40** | **+7.5** | **[−7.02, +22.02]** | **0.5078** |
| strict-clean task success | 7/40 | 10/40 | +7.5 | [−7.02, +22.02] | 0.5078 |
| clutter-contact episode | 22/40 | 17/40 | −12.5 | [−33.50, +8.50] | 0.3593 |
| pendant-contact episode | 2/40 | 0/40 | −5.0 | not preregistered | 0.5000 |

**Verdict: directional but inconclusive.** PACT leads by 3/40 on both task and
collision-free task success, and contacts clutter in five fewer episodes, but
the registered primary interval crosses zero. Under the preregistered rule this
does not establish PACT superiority. It is nevertheless the intended
four-object clutter environment and a useful follow-up to the negative V10.9
result.

The legacy evaluator did not emit the newer `v109r_funnel` fields, so funnel
zeros in the current machine-readable analysis mean unavailable, not zero.
Independent reconstruction from retained `GraspStateSensor` trajectories gives
ACT 27 touched / 12 held and PACT 28 touched / 20 held. Contact summary mode
also did not retain geom-pair identities, so aggregate contact classes are
valid but per-object contact attribution is unavailable.

Reference payloads: full run `c6c043e597d265dd3ba74c39fbc033724ecec332fc5e7b31df6486cfa8907009`;
analysis `2c09430f581ab2c0f57f434d15f83753ae8563fd1ffeda275478911d61ea1301`.

# V10.11 mixed mesh/primitive clutter validation

V10.11 is environment validation only. It uses three unchanged mesh clutter
assets and three runtime primitives, including two bounded near-target draws;
no cup/object dimensions or scene XML were edited.

| stage | result |
|---|---:|
| focused implementation/regression sweep | 102 tests passed |
| registered preflight | 96/96 passed |
| review generation | 12/12 complete, 0 sampling/infrastructure failures |
| deterministic packet selection | 3 strict-clean + 3 natural failures |
| decodable review MP4s / matching reconstructed layouts | 6/6 / 6/6 |

Preflight payload:
`1fbeafb185643bb530b5b9d725dd0df43526951701932f170f10df65bb24861c`.
Review-manifest payload:
`a742c7b4222f3370393c3da6887d3b027c1258a2a342f0396cf34407e9c2440b`.

The review set contains three clean task successes, one task success made
non-clean by clutter contact, one task success with a soap-bottle stability
event, and one task failure with a primitive-cylinder stability event. The packet is ready for
owner visual review. All collection, conversion, Phase-0, training, and
evaluation authorizations remain false.

The appearance claim is deliberately narrow: the 40-sensor skin has no
link-7/hand/gripper coverage, and the frozen V9.5 replay places the inbound
vessel outside every sensor cone in 7/8 variants. See
`docs/PACT_PLACE_SKIN_VISIBILITY.md` for the artifact-backed resolution and
coverage limits.

# V10.11c ACT/PACT training and paired evaluation

Completed six chunk-100 models and **300/300 raw-verified scientific rollouts**: ACT and PACT, seeds 3103/3104/3105, 50 paired instances per seed. All 150 pairs have identical non-RGB initial observations and selected environment seeds, with wrist RGB checked against the disclosed tolerance below. The smoke completed 8/8 rollouts on four separate instances, initially failed strict RGB equality, and was reconciled under that rule without rerunning any rollout; eight additional frozen smoke rows were reserved but not selected.

The owner narrowed scope on 2026-09-05 to chunk 100 only and omitted chunk 25, gripper-status analysis, Wilson intervals and McNemar tests. Results below are descriptive. They do not establish statistical superiority.

## Corpus and training

The read-only collection ledger contains **482 attempts, 99 accepted strict-clean episodes**. All 24 cells have 4–5 episodes. Row directories resolve as `attempt_id[:16]`; no closeout was required. The split is **75 train / 24 validation**, taking the lowest SHA256(split seed:attempt_id) per cell for validation. Split seed: 2026090401. Converted T=176–546, sum 37,871; 1,514,840 windows were encoded, with independent source-preservation checks.

The reused embedding verifier retains V10.9 schema labels and a V10.8 `note_on_window_count` paragraph. That legacy paragraph does not describe this corpus; the numerical episode/timestep/window fields and source reconstruction are authoritative.

Both arms used 2,000 epochs, batch 8, learning rate 1e-5, KL 10, chunk 100, hidden 512, feed-forward 3,200, 7 encoder/7 decoder layers, 8 heads, wrist ResNet-18, and state/action dimensions 9/8. Training episode horizon was 635. The command comparison allowed only the checkpoint directory and five PACT flags. The frozen encoder hash is `6fd2dd037e3236b5b6bf7fce8cb2709ead0cf52adcbbe9cbad1061efc2fe3206`. All six models passed strict checkpoint reload, offline inference and split/dataset provenance checks; PACT proximity consumption was checked using real versus zeroed embeddings.

| Seed | Arm | Epochs | Best epoch | Best validation loss | Training minutes |
|---|---|---:|---:|---:|---:|
| 3103 | ACT | 2000 | 1995 | 0.135764 | 51.0 |
| 3103 | PACT | 2000 | 1974 | 0.134507 | 54.0 |
| 3104 | ACT | 2000 | 1886 | 0.156815 | 53.4 |
| 3104 | PACT | 2000 | 1865 | 0.159752 | 56.1 |
| 3105 | ACT | 2000 | 1756 | 0.119959 | 54.0 |
| 3105 | PACT | 2000 | 1720 | 0.121472 | 56.9 |

## Paired outcomes

The overall comparison includes **all three independently trained seeds**: 150 rollouts per arm. For every episode-level rate, the pooled numerator divided by 150 is exactly the arithmetic mean of the three seed-level rates, because each seed contributes 50 instances. Frame and contact-entry columns are summed counts across the indicated seed(s), not percentages or per-rollout averages. Each arm is paired on the same instances within each seed; the three seed blocks use disjoint held-out instances. Thus variation between seed rows reflects both training randomness and held-out instance variation.

Collision-free success means final task success with zero hazard-bar, other-environment, clutter or mounted-fixture contact entries. Task success was independently read from the final trajectory `success` field. Contact episodes have at least one physics sample with that contact. A frame below is an audited physics sample (2 ms plus episode boundaries), not a contact-pair entry or a rendered video frame.

| Seed | Arm | Collision-free success | Task success | Hazard-contact episodes | Hazard frames | Hazard entries | Clutter-contact episodes |
|---|---|---:|---:|---:|---:|---:|---:|
| 3103 | ACT | 3/50 (6.0%) | 4/50 (8.0%) | 19/50 (38.0%) | 78119 | 83005 | 13/50 (26.0%) |
| 3103 | PACT | 4/50 (8.0%) | 4/50 (8.0%) | 16/50 (32.0%) | 69743 | 78783 | 12/50 (24.0%) |
| 3104 | ACT | 4/50 (8.0%) | 5/50 (10.0%) | 16/50 (32.0%) | 74220 | 77423 | 20/50 (40.0%) |
| 3104 | PACT | 4/50 (8.0%) | 6/50 (12.0%) | 21/50 (42.0%) | 114345 | 123899 | 17/50 (34.0%) |
| 3105 | ACT | 1/50 (2.0%) | 2/50 (4.0%) | 15/50 (30.0%) | 74855 | 92551 | 21/50 (42.0%) |
| 3105 | PACT | 2/50 (4.0%) | 2/50 (4.0%) | 18/50 (36.0%) | 99571 | 107220 | 10/50 (20.0%) |
| pooled | ACT | 8/150 (5.3%) | 11/150 (7.3%) | 50/150 (33.3%) | 227194 | 252979 | 54/150 (36.0%) |
| pooled | PACT | 10/150 (6.7%) | 12/150 (8.0%) | 55/150 (36.7%) | 283659 | 309902 | 39/150 (26.0%) |

## Per-object contact and stability

Slot 01 is the tall route cylinder; 08 the tall near-target cylinder; 09 the tall near-target box. An object is stable only if it never exceeds 2 cm translation or 25° rotation from its settled baseline at any retained control sample. Object contact attribution was reconstructed from retained geom/body pair identities and counts; stability was reconstructed from retained poses and rotations.

| Seed | Slot | Arm | Contact episodes | Contact frames | Contact entries | Stable episodes |
|---|---|---|---:|---:|---:|---:|
| 3103 | 01 | ACT | 1/50 (2.0%) | 1229 | 1270 | 46/50 (92.0%) |
| 3103 | 01 | PACT | 2/50 (4.0%) | 2636 | 6985 | 46/50 (92.0%) |
| 3103 | 08 | ACT | 2/50 (4.0%) | 187 | 254 | 49/50 (98.0%) |
| 3103 | 08 | PACT | 3/50 (6.0%) | 606 | 635 | 49/50 (98.0%) |
| 3103 | 09 | ACT | 2/50 (4.0%) | 136 | 136 | 50/50 (100.0%) |
| 3103 | 09 | PACT | 2/50 (4.0%) | 1065 | 1120 | 48/50 (96.0%) |
| 3104 | 01 | ACT | 11/50 (22.0%) | 8681 | 17548 | 38/50 (76.0%) |
| 3104 | 01 | PACT | 7/50 (14.0%) | 6232 | 11867 | 43/50 (86.0%) |
| 3104 | 08 | ACT | 1/50 (2.0%) | 774 | 774 | 50/50 (100.0%) |
| 3104 | 08 | PACT | 0/50 (0.0%) | 0 | 0 | 49/50 (98.0%) |
| 3104 | 09 | ACT | 2/50 (4.0%) | 822 | 1010 | 50/50 (100.0%) |
| 3104 | 09 | PACT | 1/50 (2.0%) | 18 | 18 | 49/50 (98.0%) |
| 3105 | 01 | ACT | 7/50 (14.0%) | 4433 | 10081 | 42/50 (84.0%) |
| 3105 | 01 | PACT | 4/50 (8.0%) | 14644 | 32542 | 45/50 (90.0%) |
| 3105 | 08 | ACT | 7/50 (14.0%) | 28736 | 87063 | 45/50 (90.0%) |
| 3105 | 08 | PACT | 2/50 (4.0%) | 305 | 868 | 49/50 (98.0%) |
| 3105 | 09 | ACT | 4/50 (8.0%) | 1948 | 2577 | 47/50 (94.0%) |
| 3105 | 09 | PACT | 0/50 (0.0%) | 0 | 0 | 50/50 (100.0%) |
| pooled | 01 | ACT | 19/150 (12.7%) | 14343 | 28899 | 126/150 (84.0%) |
| pooled | 01 | PACT | 13/150 (8.7%) | 23512 | 51394 | 134/150 (89.3%) |
| pooled | 08 | ACT | 10/150 (6.7%) | 29697 | 88091 | 144/150 (96.0%) |
| pooled | 08 | PACT | 5/150 (3.3%) | 911 | 1503 | 147/150 (98.0%) |
| pooled | 09 | ACT | 8/150 (5.3%) | 2906 | 3723 | 147/150 (98.0%) |
| pooled | 09 | PACT | 3/150 (2.0%) | 1083 | 1138 | 147/150 (98.0%) |

## Interpretation and provenance

The original smoke stage exited 1 after 8/8 successful rollouts because three wrist RGB pairs failed exact byte equality: `AssertionError: initial observation mismatch: observation/wrist_camera`. Those pairs differed in 15, 24 and 48 of 658,944 uint8 channel values, each by exactly one intensity level; all 449 non-RGB observation datasets matched exactly. Before scientific evaluation, the pairing rule was amended to allow at most 1/255 intensity difference in at most 0.1% of RGB channel values while requiring exact equality for every other field. The cause of the sparse RGB differences was not conclusively established. Original failures, raw images and `smoke_reconciliation.json` are retained. No smoke performance outcome influenced this change.

The outer coordinator was lost in a Codex crash during training; the six-model trainer survived. A detached coordinator reconnected without restarting a model. Its completion record explicitly marks the lost coordinator exit code as unavailable and relies on all six actual model exit codes and independent verification.

Clutter here is effectively invisible to the proximity skin: inbound vessel `max_w_perp_m = 0.000` in **7/8** variants of `diagnostics_output/pact_place_v9_w1_resolvability_full/resolvability.json`; the 40 sensors cover **link1–link6 only**. Any PACT–ACT difference is **not evidence that PACT senses the clutter**. This historical resolvability check is not a new visibility measurement on the held-out evaluation instances.

Evaluation used `molmo_spaces.tasks.enclosure_reach.PactPlaceCorridorV1011C33PctTallerPrimitiveSampler` from `experiment/pact-vs-act-remediation-v2`, with its implementation hash checked against the source freeze. The original chunk-100 temporal ensemble, 127.5 gripper threshold, 900-step horizon, disabled action noise and `end_on_success=false` were preserved. All five native thread pools were capped at 1. Initial and retry evaluation seeds were checked against 28,222 historical/collection-stream seeds and against each other.

Scientific evaluation took 8.93 hours with worker-count history [4, 10]. Full H5 trajectories, actions, initial observations and raw telemetry are retained under `diagnostics_output/pact_place_v1011c_eval/`.

Periodic training snapshots and optimizer resume bundles were omitted for the initial disk constraint. Verified best checkpoints are retained; each redundant final checkpoint was pruned after verification. Source-preservation checks rehashed 694 files. Every authorization flag remains false.

Orphan `multiprocessing.spawn` workers belonging to this worktree cleaned: 0.

Machine-readable report: `diagnostics_output/pact_place_v1011c_eval/analysis.json`. Completion was reconciled against `full_ledger.jsonl` and every raw result, not a progress summary.

The owner requested increased concurrency after observing resource headroom. The original scheduler was paused while all active rollouts finished normally. Their actual Linux exit codes and raw artifacts were retained in `scheduler_resize_01/`; only the drained scheduler was terminated (stage `11_eval_full` actual exit -9). The resumed stage retained the original frozen schedule and completed the remaining instances without repeating or discarding a scientific rollout. The interrupted stage is not reported as successful.

## Post-evaluation low-success audit

The [artifact-backed audit](/root/prox_learning_pact_remediation/diagnostics_output/pact_place_v1011c_post_eval_audit/AUDIT.md) confirms the low task scores and compares them with the complete V10.9 and repaired V10.10 runs. Current ACT/PACT never touch the target in 105/150 and 115/150 rollouts; no current early success is lost by the final frame. The wrist-camera target is never segmented in 62/150 and 75/150 rollouts, versus 2/40 and 6/40 in V10.10. This identifies a visibility/approach failure pattern, not proof that physical occlusion alone causes it.

Relative to V10.10, the environment changes target placement and clutter composition as well as height, while training drops from 120 to 75 episodes and from 30,000 to 20,000 actual optimizer steps per model despite the same 2,000 epochs. Expert task success among completed collection attempts falls from 240/303 (79.2%) to 210/482 (43.6%). These confounded changes support a harder environment/data combination, not a height-only causal conclusion. The audit changes no experiment parameters or scores and launches no new training or rollouts; its corrected machine-readable source is `diagnostics_output/pact_place_v1011c_post_eval_audit/audit_v2.json`.

### Approach and chunk-100 follow-up

The [follow-up audit](/root/prox_learning_pact_remediation/diagnostics_output/pact_place_v1011c_post_eval_audit/APPROACH_AND_CHUNK100.md) reconstructs world-frame TCP motion: 100/105 ACT and 112/115 PACT never-touch failures nevertheless pass within 10 cm XY of the tray. This includes 37 ACT and 40 PACT never-touch rollouts with no disallowed collision; their arm joints closely track the commanded motion. Many finish nearly stationary. This is consistent with an empty-handed placement-like trajectory, not solely physical blockage.

The model predicts every 66 ms, but the original temporal ensemble combines up to 100 predictions of the current command, with weighted mean observation age 2.726 s at full history. A new offline diagnostic on all six frozen models and all 24 validation demonstrations finds pre-touch arm-command error 42.8% higher for ACT and 45.8% higher for PACT under the original history-100 blend than under newest-prediction-only aggregation. All models remain trained at chunk 100; only arithmetic on predictions from recorded expert observations was varied. This supports aggregation as a plausible contributor, **not a measured improvement in closed-loop task success**. No new rollout or training was performed, and the original scientific results remain unchanged.

### Supervision timing and unused table-camera follow-up

The [deeper learning-failure diagnosis](/root/prox_learning_pact_remediation/diagnostics_output/pact_place_v1011c_post_eval_audit/learning_failure/DIAGNOSIS.md) verifies a one-control-step arm-label lag in all 99 V10.11c episodes (also all 144 V10.10 episodes): observation t is paired with the command already applied before it, rather than the forward command recorded at t+1. Two terminal status-only actions in each corpus were explicitly excluded from forward-target comparisons. No dataset was edited.

All 99 source episodes already contain the table-camera RGB stream, verified again by hashes and full video decoding, but all six models use wrist RGB only. Initial target segmentation is present in 92/99 table views versus 15/99 wrist views. This supports testing corrected action alignment and both cameras for both arms while preserving the V10.11c clutter, not claiming a measured gain from that unrun remedy.

The earlier offline comparison above scored against the existing lagged labels. A new six-model recheck against the actual next arm commands still favors history 10 over history 100 (pre-touch joint-error reductions 27.1% ACT / 28.6% PACT), but a trivial current-position predictor scores even lower. These errors are not success rates. The original scientific outcomes are unchanged; no new training or environment rollout was performed. All authorization flags remain false. The linked diagnosis records the actual initial diagnostic failures and subsequent successful rechecks.

## V10.11c one-seed aligned dual-camera pilot (rollouts finished; pairing gate failed)

Started 2026-09-05; target completion by 2026-09-06 11:03 UTC. This is a new seed-3103 ACT/PACT pilot, not a replacement or reanalysis of the completed three-seed experiment above. No new collection is needed: all 99 original demonstrations already contain the table-camera recording. The source corpus and historical artifacts remain read-only.

The new derivative has 37,869 observation/next-command transitions, wrist plus table RGB for both arms, and the same ledger-derived 75/24 episode split. Two terminal status-only transitions are excluded; the frozen encoder's remaining embeddings are reused bit-for-bit. Training is 30,000 updates per arm (3,000 epochs, batch 8, learning rate 1e-5), with exact 20k/30k snapshots and optimizer resume bundles retained. Training chunk, temporal-ensemble history and gripper threshold remain 100, 100 and 127.5. No chunk-25 or shorter-averaging experiment is included.

Four smoke pairs test infrastructure. Twelve separate development pairs select one shared checkpoint update count using a preregistered symmetric rule; the final evaluation is 50 fresh paired instances. Neither smoke nor development is pooled into the final result. Evaluation uses the collection `enclosure_reach.PactPlaceCorridorV1011C33PctTallerPrimitiveSampler` and `FrankaSkinHybridCameraSystem`, not the alternate `pact_place.py` environment. Evaluation concurrency is 12, capped at 12 by the owner's 2026-09-06 00:56 UTC amendment; the earlier optional 14-worker gate is superseded. The actual container limits are 15.36 CPU cores, 170.9 GiB RAM, and one 23,028-MiB NVIDIA A10. All thread-pool caps are 1; rollout storage is H5-only.

Initial launch status: conversion and four contract/runtime-wiring tests passed before training started. See the dated outcome below for the current status; the paired evaluation has not passed its final gate. Stage exits, per-rollout completion receipts, frozen streams, raw metrics and subsequent results are recorded under [the pilot directory](/root/prox_learning_pact_remediation/diagnostics_output/pact_place_v1011c_dualcam_aligned_s3103). The as-run conversion manifest retains several inherited parent-directory/token annotations; `conversion_metadata_clarification.json` identifies them. The training manifest and command explicitly bind the new directory and verified new data-tree hash, not the old dataset.

Clutter remains effectively invisible to the proximity skin (inbound vessel `max_w_perp_m = 0.000` in 7/8 historical variants; 40 sensors on link1–link6 only). Any PACT–ACT difference is not evidence that PACT senses clutter. Both policies receive the added RGB view. All authorization flags remain false. Wilson intervals, McNemar tests and gripper-status analysis are omitted as requested.

Training recovery, 2026-09-06 02:16 UTC: ACT finished with actual exit 0 at 30,000 updates. PACT's initial process exited **1** during epoch 1423 with `RuntimeError: can't start new thread`, raised while a data-loader queue was shutting down. The original log and failure receipt are retained. Its last atomic optimizer/RNG checkpoint is epoch 1400 / 14,010 updates. PACT is resuming from that checkpoint, replaying 220 completed updates plus any uncommitted work from the failed epoch, with a planned clean process renewal at 20k before continuing to 30k. Dataset, split, normalization, batch size and four loader workers are unchanged; explicit PyTorch intra/inter-op caps are now 1. The exact thread owner at failure was not captured, so a particular leak or external process is not asserted as the cause. `recovery_01/` preserves the failed log history, checkpoint, manifests and actual recovery-process exits. No evaluation stage was skipped or declared successful because of this failure.

Smoke repair, 2026-09-06 03:38 UTC: both models have reached 30,000 updates and all four retained checkpoints passed strict reload and input-consumption checks. The first eight smoke workers each exited **1** at result export with `AttributeError: 'DualCameraInferencePolicy' object has no attribute '_input_proj_proximity_shape'`; smoke stage exit was **1**, not a successful completion. The new adapter omitted a metadata field initialized by the inherited policy loader. The fix derives that field from the loaded model's actual projection weights and changes no model, action arithmetic, environment or seed. Five regression tests, including runtime initialization plus the inherited metadata exporter for both arms, passed. All failed attempts and receipts are preserved under `recovery_02/`; all four smoke pairs are being repeated. No development or final-test rollout had started when the repair was made.

### Outcome and unresolved gate — 2026-09-06 08:23 UTC

Both models finished 30,000 updates. Repaired smoke completed 8/8 and development completed 48/48, with actual stage exits 0. The symmetric development rule selected the 30k snapshots: at both 20k and 30k, ACT scored 2/12 and PACT 1/12 task successes; pooled hazard frames broke the tie in favor of 30k. Smoke/development outcomes are not included in the final panel.

All 100 original final rollout workers exited 0, reconciled against the ledger and raw artifacts. Nevertheless, final stage `07_final` exited **1**, because only 49/50 initial-observation pairs passed. A pre-recorded, bounded same-instance repeat of both arms also completed with worker exits 0 but failed the same gate; the repeated stage exited **1** at 08:18:14 UTC after 1,028.806 seconds. This is not a completed, verified paired evaluation. The normal `08_final_raw_audit` stage was **not run**, and no `pilot_completion.json` was issued.

The failed instance is `af02b30513b020e021abbcbdb050dd1688bafd0396ba2f783a5c9b39cb47e80d`. An exhaustive independent comparison confirms all 459 non-RGB initial datasets match exactly and table RGB is identical. Original wrist RGB had 42 channel values differing by one level and three by two; the repeat had 79 differing by one and three by two, out of 658,944 values. Both fail the unchanged maximum difference of one uint8 intensity level. No tolerance, seed, checkpoint or inference parameter was changed; no further repeat was launched. Original stage/ledger/attempts remain in `recovery_03/`, and the repeat failure is preserved in `pairing_recovery_failure.json` and `logs/07_final.log`.

A separate diagnostic recount completed with actual exit 0, reconstructing all 102 original/repeat trajectories, physics contact classes and object poses. It checked every canonical receipt against the 100-entry ledger and rechecked 703 protected-source/implementation hashes with zero mismatches. This independent diagnostic does **not** convert the failed pairing gate into a pass. Its evidence is [blocked_raw_audit.json](/root/prox_learning_pact_remediation/diagnostics_output/pact_place_v1011c_dualcam_aligned_s3103/evaluation/blocked_raw_audit.json).

The following descriptive counts use the **original first attempts**, retaining all 50 instances per arm, including the explicitly failed pair. Every episode-level count and hazard-frame total is unchanged in the repeat panel.

| Endpoint | ACT, seed 3103 | PACT, seed 3103 |
|---|---:|---:|
| Task success | 14/50 (28%) | 7/50 (14%) |
| Collision-free task success | 12/50 (24%) | 7/50 (14%) |
| Entirely collision-free episode | 23/50 (46%) | 27/50 (54%) |
| Hazard-bar contact episode | 14/50 (28%) | 14/50 (28%) |
| Clutter-contact episode | 19/50 (38%) | 13/50 (26%) |
| Hazard-bar contact frames, total | 127,472 | 49,565 |
| Hazard-bar contact frames, mean/episode | 2,549.44 | 991.30 |

Contact frames are physics-audit samples, not RGB frames or policy-control steps. Stability events mean displacement exceeding 2 cm or rotation exceeding 25 degrees at any recorded control observation. First-attempt per-object values:

| Slot | Arm | Contact episodes | Contact frames | Stable episodes |
|---|---|---:|---:|---:|
| 01 | ACT | 8/50 | 8,816 | 38/50 |
| 01 | PACT | 5/50 | 20,054 | 45/50 |
| 08 | ACT | 3/50 | 1,184 | 48/50 |
| 08 | PACT | 0/50 | 0 | 50/50 |
| 09 | ACT | 5/50 | 12,972 | 48/50 |
| 09 | PACT | 2/50 | 1,010 | 50/50 |

Repeating the affected pair changes slot-01 total contact frames to ACT 7,965 and PACT 20,065; contact/stability episode counts remain unchanged. Original and repeat maximum displacement/rotation and all pair-level endpoints are retained separately in the diagnostic JSON; attempts are not pooled. On the 49 pairs that passed the gate, task success remains ACT 14/49 versus PACT 7/49. This post-hoc subset is diagnostic, not a replacement benchmark, but shows the one pairing exception does not explain the low task scores.

Both arms touched the task object in 23/50 episodes. PACT has fewer clutter-contact episodes and fewer hazard-contact frames, but identical hazard-contact incidence, lower task success and lower collision-free task success. These data do not meet the owner's joint performance/safety goal. The combined camera/alignment/update intervention does not isolate any one cause; comparison with the original experiment uses a different test stream. This is one seed only, not a pooled three-seed result.

Clutter here remains effectively invisible to the proximity skin: inbound vessel `max_w_perp_m = 0.000` in 7 of 8 variants in `diagnostics_output/pact_place_v9_w1_resolvability_full/resolvability.json`; the 40 sensors cover link1–link6 only. Neither reduced contacts nor input sensitivity is evidence that PACT senses the clutter. No new chunk-25 experiment, shorter-averaging trial, Wilson intervals, McNemar tests or gripper-status analysis was performed. All authorization flags remain false.

The controllers and workers have stopped. Scoped `multiprocessing.spawn` workers with PPID 1 found during cleanup: **0**. The hourly monitor was sent SIGTERM after the bounded repair failed and was subsequently observed absent; its actual exit code was not captured. No artifact was deleted. Further pairing repair needs a new decision; this pilot is stopped, not declared successful.
