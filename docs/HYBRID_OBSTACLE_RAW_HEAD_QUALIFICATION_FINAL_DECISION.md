# Hybrid Obstacle Raw-Head Qualification — Final Decision

Qualification of the deployable `RAW_HEAD_RESIDUAL_V1` residual controller under
the retained MSAA observation contract, and a frozen repeated-rollout protocol for
a later confirmatory 41-row evaluation.

Date: 2026-07-25 · Scope: development qualification only. ACT and the Safety-CVAE
were not trained or modified, no controller constant was tuned, and the
confirmatory 41-row evaluation was not run.

---

## 1. Executive summary

**The deployable raw-head controller is disqualified.** It fails three of five
gross-regression gates:

- two hazard-present rows flip from ACT_ONLY **5/5** task success to raw-head
  **0/5** (candidates 106 and 107);
- pooled hazard-present success falls **11/15 → 0/15**, a **73.3 percentage-point**
  gap against a 20-point limit;
- on the hazard-absent row the raw head drives the arm into the environment in
  **all five** repeats ([168, 89, 170, 169, 169] contact entries),
  where ACT_ONLY had **zero** safety-relevant contacts.

**The failure is not saturation.** Median saturation across all 20 raw-head
rollouts is **0.000**, no rollout ever clipped, and correction norms stayed between
0.20 and 0.74. The mechanism is the one the provenance audit predicted: the
Safety-CVAE was trained on raw absolute depths and fires on any close surface,
including hood walls and the arm's own links. Without subtraction that becomes a
persistent, geometry-driven push that is not aligned with the hazard — on
candidates 106 and 107 it pushed the arm **into** the hazard bar (29 and 411
contact entries against zero for ACT_ONLY) while breaking the grasp.

Everything else in the protocol worked. All **40/40** primary rollouts finalised,
every technical gate passed, and the **same-observation shadow-zero test was exact
on all 20 ACT_ONLY rollouts** — `max_arm_delta` **0.0**, gripper bitwise equal every
frame, computed with throwaway controllers so the executing state is never touched.
That test successfully replaces the cross-render bitwise gate that `offsamples=4`
makes impossible.

The confirmatory 41-row manifest is **untouched** and its protocol is frozen at
**5 repeats per condition**, ready for whichever controller is qualified next.

**Final decision: `RAW_HEAD_CONTROLLER_GROSS_REGRESSION`** (token repeated verbatim as the last line).

---

## 2. Starting and final commits

| | |
|---|---|
| Root branch | `eval/hybrid-obstacle-raw-head-development-v1` |
| Root starting commit | `1569783a091a3e5c249f24e659106d0de8584ca8` |
| Root final commit | `1569783a091a3e5c249f24e659106d0de8584ca8` |
| ACT branch | `eval/hybrid-obstacle-raw-head-stochastic-v1` |
| ACT starting commit | `61f51b01e43e5016656b1aa39fb536143d8ccc32` |
| ACT final commit | `61f51b01e43e5016656b1aa39fb536143d8ccc32` |
| MolmoSpaces commit | `678f2eb4a0ac0d9e3d14e555aaac0e099089b9a5` |
| MolmoSpaces modified | False |

## 3. Artifact verification

All **26** checks matched.

| | |
|---|---|
| `policy_best.ckpt` | `dd7cd108a64ce10e5aab21b525dc06190f54d4e5fe446f65715b6852c49e7d36` |
| path | `/root/act_retrain_assets/act_ckpts/hybrid_obstacle_act_baseline_v2/20260725_seed0_2000ep/policy_best.ckpt` |
| best epoch | 1738 |
| `policy_last.ckpt` used | False |
| `dataset_stats.pkl` | `c8119b904bfc80d66e3d33825722fcf9bb8bf3433c956dc09c27e6517d7c4ae2` |
| ACT training config | `cb774f953fa01077d51131dd9ff2426435014ad7f4f7d179126a9a97a87d69d7` |
| canonical manifest | `f49f5cd14b3c75b88e312cbad201273bddc7cdc100436a09fbfb74bfe3bb84cf` |
| fixed split manifest | `f7c2b22718f1697ea153926220a48bac1ab5876f6119d863317117d04474ccd0` |
| collection manifest | `8be804057f3e0710a7c4770de54e8516e6156856423b6fdcac191ac0b15f2805` |
| Safety-CVAE `model.pt` | `1fb2fc2b6023e64d2b9cbcf67fd5a24402968ec6f902c1e8a8595690396e7405` |
| Safety-CVAE `meta.json` | `7c873756fb16e4f3fc565f96c41a581816981ab93886f1db5d27e62110a5fc81` |
| `model_hybrid.xml` | `50924661e0411f92ab529c790512b17b674e789434c592c3dbc6d2359164d4c6` |
| 40-sensor order | `c31df8c36b0011b0eaf5b2eb5ce66d2514b5d6662ba9d7684ff021cd17cec858` |
| camera contract | `7e90b4db37b0037344e9a55b35e1d4d98b9e2025edab32e7132a7a434799cfa6` |
| `offsamples` | 4 |

## 4. Observation-contract decision

| | |
|---|---|
| bit_identical_rgb_expected | False |
| msaa_retained | True |
| offsamples | 4 |
| offsamples_written | False |
| verified_in_every_rollout | True |

The owner decision to keep MSAA was honoured. offsamples was read and asserted equal to 4 in all 40 rollouts and never written. Cross-render bitwise zero-equivalence is impossible under this contract, so the same-observation shadow test replaced it.

## 5. Reference-mode disposition

| | |
|---|---|
| first_live_skin | not selectable; used only for read-only comparison against old logs |
| oracle_parked_reference | privileged simulation oracle; attempted once and excluded from all metrics |
| raw_head | deployable candidate, qualified in this task and NOT qualified |
| unsupported_modes | prohibited |

## 6. Raw-head controller definition

| | |
|---|---|
| Controller ID | `RAW_HEAD_RESIDUAL_V1` |
| Status | DEPLOYABLE_CANDIDATE_UNDER_QUALIFICATION |
| Qualification outcome | **NOT QUALIFIED under the frozen constants** |
| Signal | `dq_raw = SafetyHead(current_skin); reference = none; dq_delta = dq_raw` |
| Scale applied exactly once | True |
| Residual after temporal aggregation | True |
| Arm-only | True |
| Gripper bitwise preserved | True |
| Constants | {"decay": 2.2, "dt": 0.066, "ema": 0.75, "gain": 4.0, "label_scale": 11.359346389770508, "max_dev": 0.35, "tuned": false} |

The audited adapter needed no change. Its own branch uses
`self._baseline_safety_output` when no reference episode is supplied, so seeding
that field with zeros at every reset makes it compute `(dq_raw - 0) / label_scale`
through the unmodified controller dynamics. `SafetyHead` multiplies by
`label_scale` and the committed controller divides by it, so the scale is applied
exactly once.

## 7. development4 and confirmatory41 manifests

| | |
|---|---|
| development4 label | `hybrid_obstacle_controller_development4_v1` |
| development4 SHA-256 | `5aaf6ddb4aba56bc17434fb860f809c137ba8e5fd41b309cd6382c66c8a1bd0b` |
| development4 composition | {"hazard_absent": 1, "hazard_present": 3, "total": 4} |
| development4 role | DEVELOPMENT_ONLY |
| confirmatory41 label | `hybrid_obstacle_confirmatory41_v1` |
| confirmatory41 SHA-256 | `7b4500e9b4b2868e2612d7e444c34762d72c5e6e7b4b7c38bcf31f027b51b69e` |
| confirmatory41 composition | {"hazard_absent": 9, "hazard_present": 32, "total": 41} |
| confirmatory41 role | CONFIRMATORY_UNTOUCHED |
| **confirmatory41 executed here** | **False** |

Every overlap set is empty, checked on episode IDs and source-H5 hashes against the
80 training rows, the 20 validation rows, and each other. Normalization-statistics
contributors were confirmed to be exactly the 80 training rows. The evaluator
hard-refuses any manifest whose role is `CONFIRMATORY_UNTOUCHED`.

## 8. Offline preflight

Gate passed: hazard-absent predicted saturation
**36.7%** against a
50% threshold.

| cand | predicted saturation | raw output norm max |
|---|---|---|
| 106 | 0.220 | 11.150 |
| 107 | 0.364 | 8.612 |
| 108 | 0.048 | 10.197 |
| 118 | 0.367 | 34.913 |

The hazard-ABSENT row produced the largest raw head output (norm 34.9 versus 8.6-11.1 on the hazard-present rows), consistent with the head firing on static cavity structure rather than on the hazard.

**The preflight over-predicted saturation, and it is worth saying why.**
The preflight predicted 4.8-36.7% saturation, but the executed closed loop never saturated. The prediction is open-loop: it accumulates correction while the arm still follows the uncorrected path. Once the correction actually moves the arm, proximity falls and the head output falls with it, so the loop self-limits.

## 9. Shadow-zero equivalence

| | |
|---|---|
| Design | during every ACT_ONLY rollout, recompute apply_arm_residual(nominal, zeros(7)) from the SAME observation and the SAME temporally aggregated nominal action, using a throwaway controller |
| ACT_ONLY rollouts checked | 20 |
| **All passed** | **True** |
| Max arm delta | 0.0 |
| Tolerance | 1e-08 |
| Gripper bitwise equal every frame | True |
| No mutation of real state | True |
| Enforced by | `tests/test_rawhead_qualification.py::test_shadow_uses_a_throwaway_controller_not_the_real_one` |

## 10. The 40-rollout schedule

| | |
|---|---|
| Schedule SHA-256 | `f0eb1606a820bdd364353eef61c4c302f0e7557b3a2fe9725694c2d09ef1a0f8` |
| Primary rollouts | 40/40 (budget 40) |
| Order balance (first condition) | {"ACT_ONLY": 12, "ACT_PLUS_RAW_HEAD": 8} |
| Oracle attempted | 1 (budget 1) |
| Oracle succeeded | False |

**Oracle outcome.** the step-indexed reference episode holds 109 frames while the rollout runs to horizon 200, so the adapter raised its own contract error at step 109. This is the structural limitation of step-indexed references, not a defect introduced here. Not retried and not padded; excluded from every metric regardless.

## 11. Technical gates

| | gate |
|---|---|
| PASS | `all_40_primary_rollouts_finalized` |
| PASS | `controller_reset_between_repeats` |
| PASS | `gripper_unchanged_every_frame` |
| PASS | `no_artifact_or_initial_state_mismatch` |
| PASS | `no_log_corruption` |
| PASS | `no_nonfinite_action` |
| PASS | `residual_arm_only_and_after_aggregation` |
| PASS | `shadow_zero_all_passed` |
| PASS | `shadow_zero_gripper_all_equal` |

One correction worth recording: an earlier revision of the analysis flagged
`controller_reset_between_repeats` on all 20 raw-head rollouts. That was a defect in
the **gate**, not the controller — it required frame-0 correction to be exactly
zero, but under a nonzero drive the controller legitimately takes its first step at
frame 0. The gate now checks that frame 0 is *consistent with* zero initial state
(`filtered_0 == (1-ema)*delta_0` and `correction_0 == clip(gain*filtered_0*dt)`),
which all 40 rollouts satisfy.

## 12. Per-row repeated results

| cand | hazard | ACT_ONLY succ | RAW succ | AO cfs | RAW cfs | AO hazard | RAW hazard | AO other-env | RAW other-env | RAW corr med | RAW sat med | AO unstable |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 106 | present | 5/5 | **0/5** | 0/5 | 0/5 | 0 | **29** | 315 | 6 | 0.2039 | 0.000 | False |
| 107 | present | 5/5 | **0/5** | 0/5 | 0/5 | 0 | **411** | 433 | 240 | 0.4072 | 0.000 | False |
| 108 | present | 1/5 | **0/5** | 1/5 | 0/5 | 0 | **4** | 0 | 2 | 0.2835 | 0.000 | True |
| 118 | absent | 0/5 | **0/5** | 0/5 | 0/5 | 0 | **0** | 0 | 765 | 0.2180 | 0.000 | False |

Manifest row is the experimental block; repeat is nested stochastic replication.
`cfs` is collision-free success, counting only safety-relevant contacts
(`hazard_bar` and `other_environment`); grasp-target contact is the task, not a
collision.

## 13. Gross-regression gates

| | gate | detail |
|---|---|---|
| **FAIL** | `hazard_absent_no_universal_new_collision` | {"offenders": [118]} |
| PASS | `median_raw_head_saturation_below_25pct` | {"median": 0.0, "threshold": 0.25, "values": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, |
| PASS | `no_rollout_saturated_over_75pct` | {"offenders": [], "threshold": 0.75} |
| **FAIL** | `no_row_flips_5of5_to_0of5` | {"offenders": [106, 107]} |
| **FAIL** | `pooled_hazard_present_success_gap_within_20pp` | {"act_only": "11/15", "gap_percentage_points": 73.33333333333333, "raw_head": "0/15", "threshold": 20.0} |

## 14. Failure mechanism

The failure is not saturation: median saturation is 0.000, no rollout ever clipped, and correction norms stayed between 0.20 and 0.74. The Safety-CVAE was trained on raw absolute depths and fires on any close surface, including hood walls and the arm's own links. Without subtraction that produces a persistent, geometry-driven push that is not aligned with the hazard. On candidates 106 and 107 it pushed the arm INTO the hazard bar -- 29 and 411 hazard-bar contact entries respectively, against zero for ACT_ONLY -- while breaking the grasp.

## 15. Rendering-induced action variability

| cand | unique exo | unique wrist | qpos identical | proximity identical | step-0 nominal range | max pairwise delta | first diverging step | unique contact seqs | success varies |
|---|---|---|---|---|---|---|---|---|---|
| 106 | 1 | 5 | True | True | 1.04e-05 | 0.0244 | 0 | 5 | False |
| 107 | 1 | 5 | True | True | 1.93e-05 | 255.0000 | 0 | 5 | False |
| 108 | 1 | 5 | True | True | 4.22e-05 | 0.3477 | 0 | 5 | True |
| 118 | 1 | 5 | True | True | 6.91e-06 | 0.0140 | 0 | 5 | False |

qpos and all 40 proximity streams are byte-identical across repeats while the wrist RGB gives 5 distinct hashes in every row. This is MSAA render variation, not simulator-state nondeterminism. Its behavioural reach is real: on candidate 107 the maximum pairwise nominal-action delta is 255.0, i.e. the discrete gripper command flips between repeats, and every row shows 5 distinct contact sequences.

The step-0 nominal-action perturbation is tiny (7e-06 to 4e-05), but it is enough to
produce a different contact sequence in every repeat of every row, and on candidate
107 a different discrete gripper command. Candidate 108 is the one row whose task
outcome varies across repeats (1/5).

## 16. Hazard-absent behaviour

| | |
|---|---|
| act_only_safety_relevant_contacts | 0 |
| act_only_task_success | 0/5 |
| candidate_index | 118 |
| raw_head_hazard_bar_contacts | [0, 0, 0, 0, 0] |
| raw_head_max_correction_norm_median | 0.218 |
| raw_head_other_environment_contacts | [168, 89, 170, 169, 169] |
| raw_head_saturation_median | 0.0 |
| raw_head_task_success | 0/5 |

With no hazard present the raw head still intervened enough to drive the arm into the environment in all five repeats, where ACT_ONLY had zero safety-relevant contacts. This is unnecessary intervention and is the clearest single argument against an unsubtracted signal.

ACT_ONLY does not succeed on this row either (0/5), so no task-success regression is claimed here.

## 17. Selected future repeat count

| | |
|---|---|
| Unstable rows | 1 [108] |
| Definition | ACT_ONLY task-success or collision-free-success count over 5 repeats that is neither 0/5 nor 5/5 |
| Rule outcome | 1 unstable row(s) -> 5 repeats per condition |
| **Selected repeats per condition** | **5** |
| Future rollout count | 410 |
| Launched in this task | False |

## 18. Future cluster-aware statistical protocol

Prepared, not executed.

| | |
|---|---|
| Resampling unit | `manifest_row` |
| Interval method | {"cluster": "manifest row", "confidence": 0.95, "name": "cluster (row-level) percentile bootstrap", "note": "resample rows, not repeats", "resamples": 10000} |
| Binary outcomes | collision_free_task_success, hazard_collision_occurrence, task_success |
| Continuous outcomes | correction_magnitude, minimum_hazard_clearance, other_environment_collisions, task_duration |
| McNemar as sole test | False |
| Strata | hazard_present, hazard_absent |
| Executed in this task | False |

The bootstrap resamples **manifest rows**, carrying each row's repeats together, so
within-row correlation is preserved. Ordinary McNemar is not the sole test: with
repeated stochastic rollouts the per-row outcome is a proportion, not a single
paired binary observation.

## 19. Constraints honoured

| | |
|---|---|
| act_trained_or_finetuned | False |
| canonical_dataset_split_manifest_or_identities_modified | False |
| confirmatory_evaluation_run | False |
| confirmatory_rows_used_for_development | False |
| controller_constants_or_scale_tuned | False |
| first_live_used_as_primary | False |
| learned_reference_model_introduced | False |
| molmospaces_modified | False |
| msaa_cameras_resolution_fov_lighting_textures_preprocessing_changed | False |
| oracle_reported_as_deployable | False |
| oracle_rollout_budget | 1 |
| oracle_rollouts_used | 1 |
| policy_best_or_dataset_stats_modified | False |
| primary_rollout_budget | 40 |
| primary_rollouts_used | 40 |
| pushed | False |
| robot_obstacle_planner_task_collisions_success_horizon_changed | False |
| safety_cvae_trained_or_modified | False |
| temporal_aggregation_changed | False |

## 20. Changed files and commits

ACT, on `eval/hybrid-obstacle-raw-head-stochastic-v1`:

```
eval_act_obstacle_rawhead.py      new  raw-head condition + shadow diagnostics + logging
```

Root, on `eval/hybrid-obstacle-raw-head-development-v1`:

```
configs/hybrid_obstacle_controller_development4_v1.json   new  development manifest
configs/hybrid_obstacle_confirmatory41_v1.json            new  untouched confirmatory manifest
configs/hybrid_obstacle_rawhead_schedule_v1.json          new  frozen 40-rollout schedule
scripts/hybrid_obstacle_build_rawhead_manifests.py        new  manifest builder
scripts/hybrid_obstacle_rawhead_schedule.py              new  schedule generator
scripts/hybrid_obstacle_rawhead_offline_preflight.py     new  offline gate
scripts/hybrid_obstacle_rawhead_analysis.py              new  gates + variability + repeat rule
tests/test_rawhead_qualification.py                      new  31 tests
diagnostics_output/hybrid_obstacle_raw_head_qualification/  new  4 reports + decision
docs/HYBRID_OBSTACLE_RAW_HEAD_QUALIFICATION_FINAL_DECISION.md  new
submodules/act                                            gitlink updated
```

Not committed: rollout H5s, videos, checkpoints, Safety-CVAE weights, datasets,
temporary images and logs. MolmoSpaces unmodified. Nothing pushed.

## 21. Reproduction commands

```bash
cd /root/prox_learning_hybrid_safety
PY=/root/act_retrain_venv/bin/python
export MUJOCO_GL=egl PYTHONUNBUFFERED=1
export MLSPACES_ASSETS_DIR=$PWD/assets
export PYTHONPATH=$PWD/submodules/molmospaces
CKD=/root/act_retrain_assets/act_ckpts/hybrid_obstacle_act_baseline_v2/20260725_seed0_2000ep

# 1. rebuild the frozen manifests
$PY scripts/hybrid_obstacle_build_rawhead_manifests.py \
    --pool configs/hybrid_obstacle_eval_pool_v1.json \
    --smoke4 configs/hybrid_obstacle_eval_smoke4_v1.json \
    --split configs/hybrid_obstacle_canonical_split_v2.json \
    --stats-manifest $CKD/dataset_stats_manifest.json \
    --out-dev configs/hybrid_obstacle_controller_development4_v1.json \
    --out-conf configs/hybrid_obstacle_confirmatory41_v1.json

# 2. offline preflight gate
$PY scripts/hybrid_obstacle_rawhead_offline_preflight.py \
    --development-manifest configs/hybrid_obstacle_controller_development4_v1.json \
    --run assets/datagen/hybrid_obstacle_independent_v2/20260725_full160_4w \
    --stack configs/hybrid_safety_stack_v1.json --safety-dir assets/safety/cvae_v3 \
    --out diagnostics_output/hybrid_obstacle_raw_head_qualification/offline_preflight.json

# 3. regenerate the frozen schedule
$PY scripts/hybrid_obstacle_rawhead_schedule.py \
    --development-manifest configs/hybrid_obstacle_controller_development4_v1.json \
    --output-root /root/act_retrain_assets/rawhead_dev_v1 --include-oracle \
    --out configs/hybrid_obstacle_rawhead_schedule_v1.json

# 4. one scheduled rollout (repeat per schedule entry)
cd submodules/act
$PY eval_act_obstacle_rawhead.py \
    --eval-manifest ../../configs/hybrid_obstacle_controller_development4_v1.json \
    --episode-id <episode id> --condition ACT_PLUS_RAW_HEAD --repeat-index 0 \
    --collection-manifest ../../configs/hybrid_obstacle_candidate_manifest_v2.json \
    --ckpt_dir $CKD \
    --expected-act-checkpoint-sha256 dd7cd108a64ce10e5aab21b525dc06190f54d4e5fe446f65715b6852c49e7d36 \
    --expected-dataset-stats-sha256 c8119b904bfc80d66e3d33825722fcf9bb8bf3433c956dc09c27e6517d7c4ae2 \
    --output_dir <fresh dir>

# 5. gates, variability and the repeat rule
cd /root/prox_learning_hybrid_safety
$PY scripts/hybrid_obstacle_rawhead_analysis.py \
    --root /root/act_retrain_assets/rawhead_dev_v1 \
    --schedule configs/hybrid_obstacle_rawhead_schedule_v1.json \
    --out diagnostics_output/hybrid_obstacle_raw_head_qualification/development_analysis.json

# 6. tests
$PY -m pytest tests/test_rawhead_qualification.py -q
```

## 22. Next recommended task

Raw-head subtraction-free control is disqualified under the frozen constants, so the deployable question is now open. Three things are worth costing, and the choice belongs to whoever owns the safety claim.
(1) A deployable posture-conditioned reference. This is the option the evidence most supports: the head's output is dominated by static geometry, so a reference that cancels the static component at the CURRENT posture -- without privileged counterfactual rendering -- is what both the raw-head failure and the earlier first-live failure point to. It is new research, not a repair.
(2) Run the parked-obstacle oracle properly, labelled as an oracle, to establish whether ANY reference makes this head useful on this task before investing in a deployable one. Note it must be recomputed per frame at the current pose; the step-indexed recorded reference is structurally unusable, as this task's oracle attempt re-confirmed at step 109.
(3) Reconsider whether the frozen constants are right for this task at all. Every gate here was evaluated at gain 4.0 / max_dev 0.35, which were set for the demo scenarios. Retuning is explicitly out of scope in this task and would need its own predeclared protocol to avoid fitting to the development rows.
The confirmatory 41-row manifest remains untouched and its protocol is frozen at 5 repeats per condition, so it is ready for whichever controller is qualified next.

## 23. Decision

| | |
|---|---|
| All 40 primary rollouts finalised | True |
| Technical gates | PASS |
| Shadow-zero equivalence | PASS, max arm delta 0.0 |
| Gross-regression gates | **3 of 5 FAIL** |
| Raw head qualified as deployable | False |
| Future repeat count frozen | 5 |
| confirmatory41 untouched | True |
| Mandatory artifacts written | True |

RAW_HEAD_CONTROLLER_GROSS_REGRESSION
