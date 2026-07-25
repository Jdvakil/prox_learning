# Hybrid Obstacle Paired Smoke — Final Decision

Validation of the existing live ACT + 40-sensor Safety-CVAE adapter against the
newly pinned ACT baseline, followed by a bounded manifest-replayed paired smoke.

Date: 2026-07-25 · Scope: adapter validation and a bounded smoke only. ACT and the
Safety-CVAE were not trained or modified, and the full 45-row evaluation was not run.

---

## 1. Executive summary

Everything the paired stack is made of validated. The ACT adapter descends from
`3d25c69` and all four adapter files are **byte-identical** to it, so the loader
repair did not touch controller semantics. All **14 artifact hashes match**. ACT
strict-loads and consumes RGB + qpos only, with no proximity tensor and no
Safety-CVAE on its path. `SafetyHead.load` gives a deterministic 40x8x8 -> 7 head.
The held-out pool is the **45 successful rows excluded from the canonical 100**
(35 present / 10 absent), proven disjoint from the 80 training and 20 validation
episodes by both episode ID and source-H5 hash. A manifest-replay launcher now
reconstructs the accepted attempt and refuses to run unless the recompiled
initial-state hash equals the recorded one.

**The step-10 zero-equivalence preflight failed, and that is the gate.**
`ACT_PLUS_ZERO` did not reproduce `ACT_ONLY`: the nominal ACT action diverges at
**step 0** and the contact sequence differs. A control experiment settles the
cause. Two `ACT_ONLY` rollouts with **byte-identical source** on the same row
diverge in exactly the same way — the **`wrist_camera` RGB differs at step 0**
while the replayed initial state, the frozen reference skin, all 40 proximity
streams, `exo_camera_1` and `qpos` are bit-identical. Contact counts across the
three runs were 1335 /
1300 /
1427 — all different. Both logs carry the
pre-existing MolmoSpaces warning naming exactly that stream: *"Camera sensor
'wrist_camera' observation mismatch between reset and first step. Overwriting
cached observation with current state."*

So the divergence is **environment render nondeterminism**, not the residual, not
the reference pipeline, and not the launcher. The reference pipeline is in fact
validated as deterministic and unprivileged, and the residual integration is
sound: gripper bitwise preserved every frame, arm-only to
1.2e-07, all
values finite, correction clipped, and the controller equation replaying from the
logs to 4.5e-08.

The handoff directs that the remaining pairs must not be run when zero equivalence
fails, and fixing the cause would require changing the cameras or environment,
which is a hard constraint. Four primary pairs therefore did not finalize.

**Final decision: `PAIRED_SMOKE_INCOMPLETE`** (token repeated verbatim as the last line).

---

## 2. Starting and final commits

| | |
|---|---|
| Root branch | `eval/hybrid-obstacle-act-safety-smoke-v1` |
| Root starting commit | `d9de2af28be126a134fc4bb649257daafe8209bf` |
| Root final commit | `d9de2af28be126a134fc4bb649257daafe8209bf` |
| ACT branch | `eval/hybrid-obstacle-manifest-safety-v1` |
| ACT final commit | `68b73a80746f0165351a80c6e31bbc591f5b11a4` |
| ACT loader-repair base | `68b73a80746f0165351a80c6e31bbc591f5b11a4` |
| ACT adapter base | `3d25c69edd8d972afa59fec5c3edb9d13a357f92` |
| MolmoSpaces commit | `678f2eb4a0ac0d9e3d14e555aaac0e099089b9a5` |
| MolmoSpaces modified | False |

## 3. ACT adapter lineage verification

`git merge-base --is-ancestor 3d25c69 HEAD` succeeds. Every adapter file is
byte-identical to `3d25c69`:

| | |
|---|---|
| `eval_act_obstacle_safety.py` | True |
| `hybrid_safety_residual.py` | True |
| `run_paired_hybrid_safety_eval.py` | True |
| `tests/test_hybrid_safety_residual.py` | True |

The loader repair changed only `fixed_split_data.py`, `imitate_episodes.py`, `tests/test_fixed_split_loader.py`, `utils.py`,
so no controller semantics moved. No second copy of the adapter was cherry-picked.

## 4. Checkpoint and Safety-CVAE hashes

| | |
|---|---|
| `policy_best.ckpt` | `dd7cd108a64ce10e5aab21b525dc06190f54d4e5fe446f65715b6852c49e7d36` |
| `policy_best.ckpt` path | `/root/act_retrain_assets/act_ckpts/hybrid_obstacle_act_baseline_v2/20260725_seed0_2000ep/policy_best.ckpt` |
| `policy_last.ckpt` used | False |
| `dataset_stats.pkl` | `c8119b904bfc80d66e3d33825722fcf9bb8bf3433c956dc09c27e6517d7c4ae2` |
| ACT checkpoint manifest | `21678cd693f75df9ca22ad1d6f4f38474583bd6065ee624567b55b535b7525fc` |
| Training config | `cb774f953fa01077d51131dd9ff2426435014ad7f4f7d179126a9a97a87d69d7` |
| Canonical 100-row manifest | `f49f5cd14b3c75b88e312cbad201273bddc7cdc100436a09fbfb74bfe3bb84cf` |
| Fixed split manifest | `f7c2b22718f1697ea153926220a48bac1ab5876f6119d863317117d04474ccd0` |
| Converted dataset tree | `a567df08e3bea549a1f8f6ddfe06d8c2d6b0e8e7816759404312497ff36d7c47` |
| Source collection tree | `8b569d0e20804949f6cd344a761de17fb6207863275d66c8fa1aef587bc21f30` |
| Safety-CVAE `model.pt` | `1fb2fc2b6023e64d2b9cbcf67fd5a24402968ec6f902c1e8a8595690396e7405` |
| Safety-CVAE `meta.json` | `7c873756fb16e4f3fc565f96c41a581816981ab93886f1db5d27e62110a5fc81` |
| `model_hybrid.xml` | `50924661e0411f92ab529c790512b17b674e789434c592c3dbc6d2359164d4c6` |
| 40-sensor order | `c31df8c36b0011b0eaf5b2eb5ce66d2514b5d6662ba9d7684ff021cd17cec858` |
| ACT input is RGB + qpos only | True |
| Proximity into ACT | False |
| SafetyHead in / out | [40, 8, 8] -> [7] |
| SafetyHead deterministic | True |
| Safety label scale | 11.359346389770508 |

## 5. Evaluation-pool derivation and no-overlap proof

Label: **`held_out_expert_feasible_reserve_v1`** · SHA-256 `ae4143e91e81e5178e24751a7624c60822f7c2a22a7f3b6f13627564c49c326f`

Derivation: the successful collection rows excluded from the canonical 100-episode dataset. Composition
35 hazard-present + 10
hazard-absent = 45.

| | |
|---|---|
| pool_vs_canonical_selected_ids | 0 |
| pool_vs_train_ids | 0 |
| pool_vs_train_source_hashes | 0 |
| pool_vs_validation_ids | 0 |
| pool_vs_validation_source_hashes | 0 |

Every overlap set is empty, checked on both episode IDs and source-H5 hashes
against the 80 training and 20 validation episodes and the canonical 100.

**Scope caveat, recorded in the manifest itself:** Expert-feasible pool. The 15 collection rows whose expert planner failed have no accepted initial state to replay and are deliberately absent, so this pool is not an unrestricted sample of every environment attempt.

## 6. smoke4 manifest

SHA-256 `e92149d8f2f0def3fa50d0b531535cbcf2be68cebb998062b6b00eb19b26aa38` · the three lowest-ranked hazard-present reserve rows and the lowest-ranked hazard-absent reserve row, by predeclared stratum rank; frozen before any rollout

| cand | hazard | stratum rank | accepted retry | episode id |
|---|---|---|---|---|
| 106 | present | 81 | 0 | `f802a9057b18fc13615b…` |
| 107 | present | 82 | 1 | `4f85b23206489b0ffd0d…` |
| 108 | present | 83 | 0 | `2186a9afb8531ea0592d…` |
| 118 | absent | 29 | 0 | `a766240874f9fe5e7b22…` |

Candidate 107 carries accepted retry index 1, so retry replay is exercised.
Preflight row: candidate 106.

## 7. Reference-mode decision

| | |
|---|---|
| Headline mode | **`first_live_skin`** |
| Implementation | already present in the audited adapter; frozen at step 0 and reset per episode |
| Frozen reference skin SHA-256 | `e80cefb15ba316bc6a484c078bcab2057e87ccbdd9c0abb55b3bbdc123f5a5cc` |
| Reference minimum depth | 0.156448 m |
| Identical across all four rollouts | True |
| Privileged data in headline | False |
| Oracle reference run | False |
| Oracle skipped because | optional diagnostic; skipped to keep the rollout budget for the control experiment |

The adapter already implemented `first_live_observation`: it freezes
`head(first_skin)` once and clears it in `reset()`. This launcher additionally
records the raw reference tensor's hash, which is identical across all four
rollouts — so the reference is deterministic and no privileged frame-aligned H5
entered the headline condition.

## 8. Zero-equivalence result

**Verdict: FAILED**

Matched exactly:

- replayed initial-state hash
- frozen reference skin hash
- step-0 proximity
- step-0 exo_camera_1
- step-0 qpos
- frame count
- phase sequence
- task success
- gripper bitwise
- corrections identically zero in both arms

Diverged:

- step-0 wrist_camera RGB
- nominal ACT action from step 0
- executed action
- collision sequence

First diverging step: **0** ·
max abs executed-action difference **1.648e-02**

### Control experiment

Two ACT_ONLY rollouts of the same manifest row with byte-identical source, separate processes.

| | |
|---|---|
| Initial-state hash equal | True |
| Frozen reference skin equal | True |
| Step-0 proximity equal | True |
| Step-0 `exo_camera_1` equal | True |
| Step-0 `qpos` equal | True |
| **Step-0 `wrist_camera` equal** | **False** |
| Contact counts | {'act_only': 1335, 'act_only_control': 1300, 'act_plus_zero': 1427} |
| Task success | {'act_only': True, 'act_only_control': True, 'act_plus_zero': True} |

Two ACT_ONLY rollouts with byte-identical source diverge in exactly the same way as ACT_ONLY versus ACT_PLUS_ZERO: the wrist_camera RGB at step 0 differs while the replayed initial state, the frozen reference skin, the 40 proximity streams, exo_camera_1 and qpos are all bit-identical. Bitwise zero-equivalence is therefore unattainable in this environment for reasons independent of the safety stack, and the earlier annotation-only source edit is excluded as a cause.

### Alternative decisions ruled out, with evidence

| | |
|---|---|
| `SAFETY_RESIDUAL_INTEGRATION_FAILED` | SOUND -- SAFETY_RESIDUAL_INTEGRATION_FAILED does not apply |
| `REFERENCE_PIPELINE_INVALID` | DETERMINISTIC and unprivileged -- REFERENCE_PIPELINE_INVALID does not apply |
| `CHECKPOINT_OR_SOURCE_MISMATCH` | CHECKPOINT_OR_SOURCE_MISMATCH does not apply |
| `NOMINAL_ACT_LIVE_FAILED` | ACT-only succeeds on the preflight row, so NOMINAL_ACT_LIVE_FAILED does not apply |

Residual integration detail:

| | |
|---|---|
| all_logged_vectors_finite | True |
| controller_equation_replay_max_abs_error | 4.513e-08 |
| gripper_bitwise_preserved_every_frame | True |
| max_abs_correction | 0.27656 |
| max_deviation_limit | 0.35 |
| residual_is_arm_only_max_abs_error | 1.173e-07 |
| within_clip | True |

## 9. Pairs

Only the preflight row ran. **ONE pair only. The gating preflight failed, so the remaining three pairs were not run. Nothing here supports any comparison, ranking or safety claim.**

| cand | hazard | arm | task success | grasp contacts | hazard-bar | other env | max correction |
|---|---|---|---|---|---|---|---|
| 106 | present | ACT_ONLY | True | 1270 | 0 | 65 | 0.0 |
| 106 | present | ACT+SAFETY_LIVE | False | 103 | 0 | 173 | 0.29671 |

Hazard-present rows completed: 1 of 3. Hazard-absent rows completed: 0 of 1, so
no hazard-absent metrics exist and none are reported.

### Contact classification

The audited adapter logs every robot contact against a non-floor body, which includes the grasp target cavity_obj_0/<uid>. Touching the target IS the task, so contacts are partitioned here into grasp_target, hazard_bar (body protr_*) and other_environment. collision_free_success counts only the latter two; the raw all-non-floor reading is reported as collision_free_success_including_grasp. The adapter itself is unmodified.

| | |
|---|---|
| ACT_ONLY | {'grasp_target': 1270, 'hazard_bar': 0, 'other_environment': 65} |
| ACT+SAFETY_LIVE | {'grasp_target': 103, 'hazard_bar': 0, 'other_environment': 173} |
| Hazard-bar contacts, either arm | 0 |

**Clearance caveat:** minimum_environment_clearance_m comes from the adapter's mj_geomDistance sweep over all non-floor environment geoms, which also includes the grasp target, so it goes negative during a successful grasp. Per-geom distances are not logged for non-contacting bodies, so a hazard-bar-only minimum distance cannot be recovered from these logs; hazard-bar proximity is reported via contact entries and the 40-sensor activation depths instead.

On this single row the safety residual reached a correction norm of
0.2967 against the 0.35 limit,
the task went from success to failure, and other-environment contacts rose from
65 to
173
(gripper and `geom_70` against `bench_top`). Neither arm touched the hazard bar.
One pair proves nothing; it is recorded because it is what was observed.

## 10. Sensor attribution and the known 38/40 finding

| | |
|---|---|
| Sensors | `link5_front_sensor_1`, `link5_front_sensor_2` |
| Kept in the Safety-CVAE input | True |
| Geometry modified | False |
| ACT_ONLY activation counts | {'link5_front_sensor_1': 0, 'link5_front_sensor_2': 0} |
| Dominate raw or subtracted output | False |

On the preflight row both known self-return sensors recorded ZERO activations inside the 0.5 m activation depth, so they do not materially dominate the raw or subtracted safety output here. 25-28 other sensors were active on all 200 frames.

They remain in the input to match the Safety-CVAE's training contract, are logged
separately, and their raw activation is not counted as obstacle-responsive
evidence.

## 11. Rollout budget

| | |
|---|---|
| Maximum allowed | 10 |
| Used | 4 |
| Rollouts | r106 ACT_ONLY, r106 ACT_PLUS_ZERO, r106 ACT_PLUS_SAFETY_LIVE, r106 ACT_ONLY control (determinism) |
| Remaining pairs | per the handoff instruction not to proceed when zero equivalence fails |

The fourth rollout is the determinism control. It was spent deliberately: the
zero-equivalence gate is a bitwise comparison, so establishing whether the
environment is reproducible at all was worth more than an optional oracle
diagnostic.

## 12. Readiness gates

| | |
|---|---|
| act_strictly_loads_and_runs | True |
| all_four_initial_states_replay_exactly | 1 of 4 verified (only the preflight row was run) |
| all_four_primary_pairs_reached_terminal_outcomes | False |
| every_artifact_hash_matches | True |
| final_reports_complete | True |
| gripper_commands_unchanged | True |
| no_manifest_row_duplicated_or_replaced | True |
| no_nan_invalid_action_missing_sensor_or_logging_corruption | True |
| reserve_manifest_no_train_val_overlap | True |
| safetyhead_strictly_loads_and_runs | True |
| temporal_aggregation_unchanged | True |
| zero_safety_reproduces_act_only | False |

## 13. Constraints honoured

| | |
|---|---|
| act_trained_or_finetuned | False |
| canonical_dataset_or_split_changed | False |
| collection_manifest_changed | False |
| controller_constants_tuned | False |
| environment_robot_cameras_changed | False |
| evaluation_rows_from_train_or_validation | False |
| full_45_row_evaluation_run | False |
| invalid_175_file_collection_used | False |
| molmospaces_modified | False |
| policy_best_or_dataset_stats_changed | False |
| pushed | False |
| safety_cvae_trained_or_modified | False |
| second_safety_controller_implemented | False |
| sensor_geometry_modified | False |

## 14. Changed files and commits

ACT, on `eval/hybrid-obstacle-manifest-safety-v1` (base `68b73a8`):

```
eval_act_obstacle_manifest_safety.py     new  manifest-replay launcher
tests/test_manifest_safety_eval.py       new  38 static tests
```

Root, on `eval/hybrid-obstacle-act-safety-smoke-v1`:

```
configs/hybrid_obstacle_eval_pool_v1.json          new  45-row held-out pool
configs/hybrid_obstacle_eval_smoke4_v1.json        new  frozen 3/1 smoke manifest
scripts/hybrid_obstacle_build_eval_manifest.py     new  pool + smoke builder
scripts/hybrid_obstacle_paired_smoke_audit.py      new  zero-equivalence + paired audit
diagnostics_output/hybrid_obstacle_paired_smoke/   new  audit, diagnosis, decision
docs/HYBRID_OBSTACLE_PAIRED_SMOKE_FINAL_DECISION.md  new
submodules/act                                     gitlink updated
```

The audited adapter, MolmoSpaces, the checkpoints, the canonical dataset and the
split are all unmodified. Rollout outputs, videos and H5s are not committed.
Nothing was pushed.

## 15. Reproduction commands

```bash
cd /root/prox_learning_hybrid_safety
PY=/root/act_retrain_venv/bin/python
export MUJOCO_GL=egl PYTHONUNBUFFERED=1
export MLSPACES_ASSETS_DIR=$PWD/assets
export PYTHONPATH=$PWD/submodules/molmospaces

# 1. rebuild the held-out pool and the frozen smoke manifest
$PY scripts/hybrid_obstacle_build_eval_manifest.py \
    --canonical configs/hybrid_obstacle_canonical_manifest_v2.json \
    --split configs/hybrid_obstacle_canonical_split_v2.json \
    --collection-manifest configs/hybrid_obstacle_candidate_manifest_v2.json \
    --run assets/datagen/hybrid_obstacle_independent_v2/20260725_full160_4w \
    --stack configs/hybrid_safety_stack_v1.json \
    --out-pool configs/hybrid_obstacle_eval_pool_v1.json \
    --out-smoke configs/hybrid_obstacle_eval_smoke4_v1.json

# 2. static tests (94 across the ACT suite)
(cd submodules/act && $PY -m pytest tests/ -q)

# 3. one manifest-replayed rollout (repeat per condition)
cd submodules/act
$PY eval_act_obstacle_manifest_safety.py \
    --eval-manifest ../../configs/hybrid_obstacle_eval_smoke4_v1.json \
    --episode-id f802a9057b18fc13615bcb1386d13cb839ade4b5fa5a038a5cfc054b9187b8e2 \
    --condition ACT_ONLY \
    --collection-manifest ../../configs/hybrid_obstacle_candidate_manifest_v2.json \
    --ckpt_dir /root/act_retrain_assets/act_ckpts/hybrid_obstacle_act_baseline_v2/20260725_seed0_2000ep \
    --expected-act-checkpoint-sha256 dd7cd108a64ce10e5aab21b525dc06190f54d4e5fe446f65715b6852c49e7d36 \
    --expected-dataset-stats-sha256 c8119b904bfc80d66e3d33825722fcf9bb8bf3433c956dc09c27e6517d7c4ae2 \
    --output_dir <fresh dir>

# 4. zero-equivalence + paired audit
cd /root/prox_learning_hybrid_safety
$PY scripts/hybrid_obstacle_paired_smoke_audit.py \
    --root /root/act_retrain_assets/paired_smoke_v1 \
    --pairs 106:r106_act_only:r106_safety_live \
    --zero-pair r106_act_only r106_act_plus_zero \
    --out diagnostics_output/hybrid_obstacle_paired_smoke/paired_smoke_audit.json
```

## 16. Next recommended task

Make the paired comparison robust to wrist_camera render nondeterminism, in a task that is explicitly allowed to touch the observation path. Two options worth costing: (a) investigate and fix the reset/first-step wrist_camera cache mismatch that MolmoSpaces already warns about, which would restore bitwise zero-equivalence; or (b) redefine the gate as a stochastic-equivalence test over N repeats per condition, and size the 45-row evaluation for paired variance rather than bitwise identity. Everything else in this stack is validated and ready.

## 17. Decision

| | |
|---|---|
| Artifact hashes match | True |
| Adapter lineage verified | True |
| Reference pipeline deterministic and unprivileged | True |
| Residual integration sound | True |
| Reserve manifest free of train/validation overlap | True |
| Zero safety reproduces ACT-only | False |
| Four primary pairs finalized | False |

PAIRED_SMOKE_INCOMPLETE
