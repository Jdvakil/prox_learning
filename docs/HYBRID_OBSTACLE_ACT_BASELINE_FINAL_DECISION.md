# Hybrid Obstacle ACT Baseline — Final Decision

Training and pinning of the canonical vanilla ACT baseline on the recovered
hybrid-obstacle dataset, including a bounded audit and minimal repair of the ACT
dataset loader so it honours the committed trajectory split without validation
leakage.

Date: 2026-07-25 · Task scope: loader repair, one canonical seed-0 training, and
offline teacher-forced validation. **No simulator evaluation was run and the
Safety-CVAE was never loaded or integrated.**

---

## 1. Executive summary

The ACT fork could not train this dataset correctly as it stood. Its loader
generated its **own random 80/20 split**, computed normalization statistics over
**all 100 episodes** — leaking the 20 validation trajectories into the qpos and
action mean and std — **shuffled the validation loader**, pointed at a
machine-specific absolute path for the superseded `obstacle_v1` dataset with
`episode_len` 169, and had **no way to resume** an interrupted run. All five
defects were repaired in the data/training path only; the ACT architecture, loss,
KL formulation, action chunking, temporal aggregation and optimizer mathematics
are untouched.

The repaired loader consumes the committed split manifest and nothing else, and
verifies the split, canonical-manifest and converted-dataset hashes before a
single batch is read. Normalization now comes from the **80 training
trajectories only** (6450 real timesteps), and a regression test proves that
mutating a validation trajectory leaves the statistics unchanged while mutating a
training trajectory changes them.

One canonical run was executed: **seed 0, 2000 epochs, batch size 8**, completing
all 2000 epochs and 20 000 optimizer steps in **49m 47s**.
Validation loss fell from 77.8449 to a minimum of
**0.171486 at epoch 1738**. Training loss falls
below validation loss after roughly epoch 900, so best-checkpoint selection by
validation loss is doing real work; no hyperparameter was changed in response to
any curve.

Offline teacher-forced metrics on the fixed 20 validation trajectories
(1543 real timesteps): normalized action MAE **0.442933**,
denormalized **6.962633**, arm-joint MAE
**0.089441 rad**, gripper-command MAE
**55.0750** on the 0–255 actuator scale.
These are imitation metrics only — no task success, collision or safety claim is
made or implied.

All nine post-training verification checks pass: the converted dataset and the
read-only source collection are unchanged, the best checkpoint contains no
proximity and no Safety-CVAE parameter, the statistics match the contract, the
split IDs match the committed manifest exactly, all 2000 epochs are present
without repeats, and every loss is finite. Reloading `policy_best.ckpt` is
bit-deterministic on a fixed validation batch.

**Final decision: `ACT_BASELINE_TRAINED_AND_PINNED`** (token repeated verbatim as the last line).

---

## 2. Scientific role of this checkpoint

The single nominal manipulation policy for BOTH later conditions: ACT only, and the same checkpoint plus the 40-sensor Safety-CVAE residual. No separate nominal checkpoint was or should be trained for the safety condition.

## 3. Starting and final commits

| | |
|---|---|
| Root branch | `train/hybrid-obstacle-act-baseline-v2` |
| Root starting commit | `91065ace73613e2672de92f42dd52e34f42da120` |
| Root commit at training | `92371042b41eb792302727f0769a8c5fda0201a5` |
| ACT branch created | `repair/hybrid-obstacle-fixed-split-loader-v1` |
| ACT commit after repair | `68b73a80746f0165351a80c6e31bbc591f5b11a4` |
| ACT base (Jdvakil, unmodified) | `3d25c69edd8d972afa59fec5c3edb9d13a357f92` |
| MolmoSpaces commit | `678f2eb4a0ac0d9e3d14e555aaac0e099089b9a5` |
| MolmoSpaces modified | False |

## 4. Canonical dataset and split hashes

Every hash below was recomputed independently in this task, not copied from the
handoff.

| | |
|---|---|
| Source collection tree SHA-256 | `8b569d0e20804949f6cd344a761de17fb6207863275d66c8fa1aef587bc21f30` |
| Canonical 100-row manifest SHA-256 | `f49f5cd14b3c75b88e312cbad201273bddc7cdc100436a09fbfb74bfe3bb84cf` |
| Fixed split manifest SHA-256 | `f7c2b22718f1697ea153926220a48bac1ab5876f6119d863317117d04474ccd0` |
| Converted dataset tree SHA-256 | `a567df08e3bea549a1f8f6ddfe06d8c2d6b0e8e7816759404312497ff36d7c47` |
| Selected conversion | **conversion_A** — conversion_A and conversion_B were byte-identical; A chosen as the immutable input |
| Episodes | 100 (75 hazard-present / 25 hazard-absent) |
| Train | 80 (60 present / 20 absent) |
| Validation | 20 (15 present / 5 absent) |

Conversion A and B tree hashes are equal, both conversions hold the same 100
source episode IDs, every converted episode maps to exactly one canonical row, no
source episode appears twice, and there is no train/validation overlap by ACT
index, source episode ID or source file.

## 5. Current-loader audit

Full report: `diagnostics_output/hybrid_obstacle_act_baseline/current_loader_audit.json`
Verdict: **REPAIR_REQUIRED**

| question | answer | severity |
|---|---|---|
| `act_generates_its_own_random_split` | True | blocking |
| `normalization_uses_all_100_episodes` | True | blocking |
| `dataset_files_discovered_by_filesystem_order` | False | — |
| `episode_ids_assumed_contiguous` | True | tolerable_for_this_dataset |
| `validation_dataloader_is_shuffled` | True | blocking |
| `qpos_action_statistics_include_padded_timesteps` | False | — |
| `variable_length_episode_padding` | padded to num_queries (chunk size), not to episode_len | — |
| `all_batch_elements_receive_same_action_sequence_length` | True | — |
| `sim_attribute_set_correctly` | True | — |
| `simulation_actions_begin_at_same_timestep_as_observations` | True | — |
| `non_simulation_start_ts_minus_1_hack_active` | False | — |
| `state_dimension_hard_coded` | True | cosmetic |
| `action_dimension_hard_coded` | True | cosmetic |
| `camera_names_hard_coded` | True | — |
| `episode_horizon_hard_coded` | True | blocking_value_wrong |
| `checkpoint_resume_preserves_optimizer_and_rng_state` | False | blocking |
| `proximity_consumed` | False | — |
| `safety_cvae_loaded` | False | — |

### Leakage and padding defects found

1. random 80/20 split replaces any committed split (utils.py:121)
2. normalization statistics computed over all 100 episodes, leaking the 20 validation trajectories (utils.py:126)
3. validation DataLoader shuffled, making validation loss nonreproducible (utils.py:132)
4. constants.py obstacle_baseline points at a machine-specific absolute path for the superseded obstacle_v1 dataset with episode_len 169 instead of 132
5. no optimizer/RNG/epoch resume state and non-atomic checkpoint writes

### Confirmed sound, and therefore left alone

- simulation temporal alignment is correct and the start_ts-1 real-robot hack is inactive
- padding produces a uniform (num_queries, action_dim) action tensor and a matching mask, so heterogeneous lengths batch correctly
- padded timesteps do not enter the normalization statistics
- padded timesteps do not enter the L1 loss
- episode files are not discovered by filesystem order
- no proximity input and no Safety-CVAE on the training path

## 6. Loader and trainer changes

New `submodules/act/fixed_split_data.py`:

- train/validation lists come from the committed split manifest only, with its
  self-hash verified on load and optionally pinned to an expected value;
- `compute_train_only_norm_stats` reads on-disk real timesteps for the training
  episodes only, so neither validation data nor padding can contribute;
- `verify_dataset` checks every converted file's hash and the whole-tree hash
  against the conversion manifest before training;
- `verify_episode_schema` asserts the `sim` flag, qpos 9, action 8, both cameras,
  frame counts and the horizon instead of trusting hard-coded values;
- both loaders take explicit generators reseeded per epoch, and validation uses a
  constant seed so it scores a fixed window set in deterministic order.

`submodules/act/imitate_episodes.py`:

- `--dataset_dir`, `--split_manifest`, `--dataset_manifest`,
  `--expect_split_sha256`, `--expect_dataset_tree_sha256`, `--episode_horizon`,
  `--state_dim`, `--action_dim`, `--exact_ckpt_dir`, `--resume`, `--max_steps`,
  `--ckpt_every`, `--num_workers`;
- a run manifest recording the exact episode IDs and every provenance hash;
- atomic checkpoint publication and a resume bundle carrying epoch, optimizer
  state, best-so-far, and Python/NumPy/Torch/CUDA RNG state; resume refuses to
  continue if the split, statistics or dataset hash moved;
- the per-epoch train summary no longer slices a flat per-batch history with
  `(batch_idx+1)*epoch` arithmetic, which mis-sliced on any resume;
- an argv guard hides this launcher's flags from DETR's own parser, which
  re-parses the process-wide `sys.argv`. DETR is unchanged.

`submodules/act/utils.py`: one opt-in `init_probe=False` so the manifest path can
read the `sim` flag without the constructor's random draw. Default behaviour and
every legacy task are unchanged.

**Not changed:** ACT architecture, ResNet-18 backbone, transformer layer counts,
attention heads, loss definition, KL formulation, action chunking, temporal
aggregation semantics, optimizer mathematics.

## 7. Exact train and validation episode IDs

ACT episode indices, taken verbatim from the committed split manifest and printed
by the loader at startup.

**Train (80):**

```
0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94
```

**Validation (20):**

```
60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 95, 96, 97, 98, 99
```

Validation trajectories in detail:

| act idx | hazard | source episode id | real T | denorm MAE |
|---|---|---|---|---|
| 60 | present | `a048f9a55ad73ebf53c0…` | 110 | 9.5269 |
| 61 | present | `74afc9ab4a9368ff8a5c…` | 56 | 7.4207 |
| 62 | present | `72ac2c10c0c079666873…` | 59 | 7.8239 |
| 63 | present | `c8425b548fa62c4652bd…` | 61 | 5.1320 |
| 64 | present | `10d698b68ccde0dc9fd4…` | 62 | 5.7166 |
| 65 | present | `15b7751135578d34cbf0…` | 54 | 2.6381 |
| 66 | present | `08e834f65a5d8ba7ce54…` | 93 | 6.5216 |
| 67 | present | `cf6f9c259f9b375fe219…` | 59 | 9.5473 |
| 68 | present | `815db39e841c29340eea…` | 59 | 10.9379 |
| 69 | present | `5fb90f8aed511a979290…` | 128 | 6.7637 |
| 70 | present | `1ce06b5c41b9d7d3f237…` | 56 | 5.8680 |
| 71 | present | `cdaaa6c0b0971ea6cbde…` | 97 | 6.9057 |
| 72 | present | `97ba8fa5ca54e4a79551…` | 95 | 3.7446 |
| 73 | present | `3dd56c4f5180e28d3d15…` | 106 | 7.6012 |
| 74 | present | `b90df4a7b64c84c127bf…` | 62 | 7.9146 |
| 95 | absent | `abea31b28c6fe48aab09…` | 85 | 7.4170 |
| 96 | absent | `10e1cd6d45700f4957ea…` | 88 | 6.4191 |
| 97 | absent | `8032fabd9fe0211c19b1…` | 67 | 7.9736 |
| 98 | absent | `9312495a27e0bf865c43…` | 83 | 6.5413 |
| 99 | absent | `499eee89fb917c2eeb4a…` | 63 | 5.6154 |

## 8. Train-only normalization statistics

| | |
|---|---|
| Computed from | training trajectories only |
| Contributing trajectories | 80 |
| Contributing timesteps | 6450 |
| Validation included | False |
| Padding included | False |
| **Statistics SHA-256** | `5a2425252d01ee505b124e4b77a18a2b80ad54d97c04473b4b09a1438dc44a71` |

Regression tests prove the property rather than asserting it: mutating a
validation-only trajectory leaves the statistics hash unchanged, mutating a
training trajectory changes it, and the contributing-timestep count equals the
sum of real on-disk timesteps (not `n_episodes * chunk_size`, which is what a
padded computation would give).

## 9. Static and integration tests

| | |
|---|---|
| New tests | 38 |
| Pre-existing tests | 18 |
| Total passed | 56 |
| Failures | 0 |
| Test file | `submodules/act/tests/test_fixed_split_loader.py` |

Bounded full-size integration smoke (5 optimizer steps, no performance claim, no
simulator rollout):

| | |
|---|---|
| Optimizer steps | 5 |
| Finite forward loss | True |
| Finite gradients | True |
| Train-only statistics loaded | True |
| images batch | `[8, 2, 3, 240, 320]` |
| qpos batch | `[8, 9]` |
| padded actions | `[8, 100, 8]` |
| padding mask | `[8, 100]` |
| model output | `[8, 100, 8]` |
| Parameters | 83,911,817 |
| GPU peak allocated | 2.230 GiB |
| GPU peak reserved | 2.350 GiB |
| GPU total | 22.069 GiB |
| **Batch size 8 fits** | **True** with 19.7 GiB headroom |

### Resume equality — measured, not assumed

| | |
|---|---|
| fresh vs fresh, same seed | max|diff| 1.041e-04 |
| uninterrupted vs resumed | max|diff| 1.043e-04 |
| epoch-0 validation loss, all runs | [77.84487915039062, 77.84487915039062, 77.84487915039062] |
| epochs repeated on resume | none |
| global step, uninterrupted vs resumed | 20 vs 20 |
| best epoch, uninterrupted vs resumed | 1 vs 1 |

Resume introduces no divergence beyond the platform's own run-to-run nondeterminism: uninterrupted-vs-resumed (max|diff| 1.043e-04) is indistinguishable from fresh-vs-fresh with the same seed (1.041e-04). The epoch-0 validation loss is bit-identical across all three runs, so the data path, split, statistics and fixed validation windows are fully deterministic; divergence appears only after backward passes, from nondeterministic cuDNN kernels. torch.use_deterministic_algorithms was deliberately NOT enabled: the handoff forbids changing training behaviour merely to force bit-identity.

## 10. Full training configuration

Frozen in `configs/hybrid_obstacle_act_baseline_v2.yaml`
(SHA-256 `cb774f953fa01077d51131dd9ff2426435014ad7f4f7d179126a9a97a87d69d7`) **before** the run and not adjusted afterwards.

| | |
|---|---|
| policy class | ACT |
| backbone | ResNet-18 |
| visual inputs | `exo_camera_1`, `wrist_camera` |
| image size | 240 x 320 |
| qpos dim | 9 |
| action dim | 8 |
| transformer encoder layers | 4 |
| transformer decoder layers | 7 |
| attention heads | 8 |
| hidden dim | 512 |
| feed-forward dim | 3200 |
| chunk size | 100 |
| episode horizon | 132 |
| KL weight | 10 |
| batch size | 8 |
| learning rate | 1e-5 |
| epochs | 2000 |
| seed | 0 |
| train / validation trajectories | 80 / 20 |
| optimizer | AdamW, two param groups (recorded from the fork, not changed) |
| lr_backbone | 1e-5 |
| weight decay | 1e-4 |
| lr schedule | none |

## 11. Training command

```bash
cd /root/prox_learning_hybrid_safety/submodules/act && \
    /root/act_retrain_venv/bin/python \
    imitate_episodes.py \
    --task_name obstacle_baseline \
    --ckpt_dir /root/act_retrain_assets/act_ckpts/hybrid_obstacle_act_baseline_v2/20260725_seed0_2000ep \
    --exact_ckpt_dir \
    --policy_class ACT \
    --batch_size 8 \
    --seed 0 \
    --num_epochs 2000 \
    --lr 1e-05 \
    --kl_weight 10 \
    --chunk_size 100 \
    --hidden_dim 512 \
    --dim_feedforward 3200 \
    --dataset_dir /root/prox_learning_hybrid_safety/assets/act_style_data/hybrid_obstacle_canonical_v2/conversion_A \
    --split_manifest /root/prox_learning_hybrid_safety/configs/hybrid_obstacle_canonical_split_v2.json \
    --dataset_manifest /root/prox_learning_hybrid_safety/diagnostics_output/hybrid_obstacle_full_collection/conversion_A_manifest.json \
    --expect_split_sha256 f7c2b22718f1697ea153926220a48bac1ab5876f6119d863317117d04474ccd0 \
    --expect_dataset_tree_sha256 a567df08e3bea549a1f8f6ddfe06d8c2d6b0e8e7816759404312497ff36d7c47 \
    --episode_horizon 132 \
    --state_dim 9 \
    --action_dim 8 \
    --num_workers 1 \
    --ckpt_every 100 \
    --no_wandb
```

## 12. Training duration and curves

| | |
|---|---|
| Epochs completed | 2000 / 2000 |
| Optimizer steps | 20000 |
| Steps per epoch | 10 |
| Wall-clock | 49m 47s |
| Seconds per epoch | 1.494 |
| First train loss | 42.992737 |
| Final train loss | 0.070248 |
| First validation loss | 77.844879 |
| Final validation loss | 0.173852 |
| **Minimum validation loss** | **0.171486** |
| **Best epoch** | **1738** |
| Best-epoch L1 component | 0.167729 |
| Best-epoch KL component | 0.000376 |
| GPU peak reserved | 2.350 GiB |

Validation loss at milestones:

| epoch | validation loss |
|---|---|
| 0 | 77.844879 |
| 100 | 0.315890 |
| 500 | 0.214366 |
| 880 | 0.177701 |
| 1000 | 0.193705 |
| 1454 | 0.174312 |
| 1738 | 0.171486 |
| 1999 | 0.173852 |

train loss falls below validation loss after roughly epoch 900 and keeps falling to 0.0682 while validation flattens near 0.18; best-checkpoint selection by validation loss is what guards against this. No hyperparameter was changed in response.

Curve PNGs (`train_val_loss_seed_0.png`, `train_val_l1_seed_0.png`,
`train_val_kl_seed_0.png`) and the per-epoch `epoch_log.jsonl` are in the run
directory; the full arrays are in `training_summary.json`.

## 13. Offline validation metrics

Offline teacher-forced imitation metrics only. No simulator rollout, no task-success measurement, and no claim about collision avoidance, safety, or superiority over any other policy.

Protocol: every real timestep of every validation trajectory; chunk size 100;
padding never scored; CVAE in inference (prior sample, no action encoder).

| | |
|---|---|
| Validation trajectories | 20 |
| Validation real timesteps | 1543 |
| Scored action elements | 514952 |
| **Normalized action MAE** | **0.442933** |
| **Denormalized action MAE** | **6.962633** |
| Arm-joint MAE (normalized) | 0.441976 |
| Arm-joint MAE (denormalized, rad) | 0.089441 |
| Gripper-command MAE (normalized) | 0.449629 |
| Gripper-command MAE (denormalized, 0–255 scale) | 55.0750 |
| Hazard-present MAE (denormalized) | 7.010568 |
| Hazard-absent MAE (denormalized) | 6.809570 |
| Hazard-present / absent trajectories | 15 / 5 |

Per-action-dimension MAE:

| dimension | normalized MAE | denormalized MAE |
|---|---|---|
| arm_joint_0 | 0.578268 | 0.040696 |
| arm_joint_1 | 0.329458 | 0.095859 |
| arm_joint_2 | 0.227393 | 0.034779 |
| arm_joint_3 | 0.465489 | 0.161405 |
| arm_joint_4 | 0.477130 | 0.084627 |
| arm_joint_5 | 0.571206 | 0.109040 |
| arm_joint_6 | 0.444890 | 0.099681 |
| gripper_command | 0.449629 | 55.074978 |

The denormalized aggregate is dominated by the gripper command, which lives on a
0–255 actuator scale while the seven arm joints are radians; the per-dimension
table above is the honest view. Hazard-present and hazard-absent MAE are close
(7.011 vs 6.810),
but the hazard-absent stratum has only 5 trajectories, so no conclusion should be
drawn from the difference.

## 14. Checkpoint hashes and provenance

| | |
|---|---|
| `policy_best.ckpt` | `dd7cd108a64ce10e5aab21b525dc06190f54d4e5fe446f65715b6852c49e7d36` |
| `policy_best.ckpt` path | `/root/act_retrain_assets/act_ckpts/hybrid_obstacle_act_baseline_v2/20260725_seed0_2000ep/policy_best.ckpt` |
| `policy_last.ckpt` | `f952f8f0887bfc5d3b09d9d8c2a6753b40828de10751dfad516294ca09977ebe` |
| `policy_last.ckpt` path | `/root/act_retrain_assets/act_ckpts/hybrid_obstacle_act_baseline_v2/20260725_seed0_2000ep/policy_last.ckpt` |
| `dataset_stats.pkl` | `c8119b904bfc80d66e3d33825722fcf9bb8bf3433c956dc09c27e6517d7c4ae2` |
| `resume_bundle.ckpt` | `8483dc08912d27c3f1c48efc2bf2351b0dc17d436a61df618f13b153d185fd70` |
| Run directory size | 8.78 GiB |
| **Checkpoint manifest SHA-256** | `21678cd693f75df9ca22ad1d6f4f38474583bd6065ee624567b55b535b7525fc` |
| Run manifest SHA-256 | `5ffedb180d4cec1d9c42990909a60cbb246b071c8603fb912e88e4ded3cd6f48` |

Checkpoints are **not committed**: 8.8 GiB of checkpoints; no approved artifact policy in this repo. Immutable local paths and SHA-256 hashes are recorded instead.

The evaluator rejection rule is recorded in `checkpoint_manifest.json`:
recompute the statistics hash from the training split, compare the split-manifest
and converted-dataset tree hashes, and compare the `policy_best.ckpt` hash. Any
mismatch means the checkpoint was not trained on this dataset and split.

### Post-training verification

| | |
|---|---|
| all_losses_finite | **PASS** |
| converted_dataset_unchanged | **PASS** |
| epoch_bookkeeping | **PASS** |
| no_proximity_parameters | **PASS** |
| no_safety_cvae_parameters | **PASS** |
| source_collection_intact | **PASS** |
| split_ids_match_committed | **PASS** |
| statistics_match | **PASS** |
| training_log_clean | **PASS** |

All checks: **PASS**

## 15. Constraints honoured

| | |
|---|---|
| canonical_75_25_selection_unchanged | True |
| canonical_source_collection_unmodified | True |
| committed_80_20_split_unchanged | True |
| converted_datasets_unmodified | True |
| hyperparameter_sweep_run | False |
| hyperparameters_changed_after_seeing_validation | False |
| live_policy_rollouts_run | False |
| molmospaces_modified | False |
| multiple_seeds_trained | False |
| new_random_split_created | False |
| normalization_from_validation | False |
| proximity_added_to_act | False |
| pushed | False |
| robot_env_cameras_planner_task_modified | False |
| safety_cvae_loaded | False |
| task_success_evaluated | False |

## 16. Changed files and commits

ACT, on `repair/hybrid-obstacle-fixed-split-loader-v1` (base `3d25c69`):

```
fixed_split_data.py                      new
tests/test_fixed_split_loader.py         new (38 tests)
imitate_episodes.py                      manifest path, run manifest, atomic ckpt, resume
utils.py                                 opt-in init_probe=False
```

Root, on `train/hybrid-obstacle-act-baseline-v2`:

```
configs/hybrid_obstacle_act_baseline_v2.yaml               new (frozen contract)
scripts/run_hybrid_obstacle_act_baseline_v2.py             new (launcher)
scripts/hybrid_obstacle_act_baseline_offline_eval.py       new (offline metrics)
diagnostics_output/hybrid_obstacle_act_baseline/*.json     new (7 reports)
docs/HYBRID_OBSTACLE_ACT_BASELINE_FINAL_DECISION.md        new
submodules/act                                            gitlink updated
```

Not committed: checkpoints, converted dataset, source H5 data, WandB cache,
videos, temporary smoke outputs. Nothing was pushed.

## 17. Exact reproduction and resume commands

```bash
cd /root/prox_learning_hybrid_safety
PY=/root/act_retrain_venv/bin/python

# 1. static tests (56)
(cd submodules/act && $PY -m pytest tests/ -q)

# 2. show the frozen command without running it
$PY scripts/run_hybrid_obstacle_act_baseline_v2.py \
    --ckpt-dir <fresh dir> --print-only

# 3. the canonical run (seed 0, 2000 epochs)
$PY scripts/run_hybrid_obstacle_act_baseline_v2.py --ckpt-dir <fresh dir>

# 4. resume an interrupted run in the SAME directory
$PY scripts/run_hybrid_obstacle_act_baseline_v2.py --ckpt-dir <same dir> --resume

# 5. offline teacher-forced validation of the best checkpoint
$PY scripts/hybrid_obstacle_act_baseline_offline_eval.py \
    --run-dir /root/act_retrain_assets/act_ckpts/hybrid_obstacle_act_baseline_v2/20260725_seed0_2000ep \
    --dataset-dir assets/act_style_data/hybrid_obstacle_canonical_v2/conversion_A \
    --out diagnostics_output/hybrid_obstacle_act_baseline/offline_validation.json
```

The launcher refuses to start if the canonical manifest, split manifest or
converted-dataset tree hash does not match the frozen contract, and refuses to
overwrite a non-empty run directory without `--resume`.

## 18. Next recommended task

Paired offline-to-online evaluation setup: run the pinned policy_best.ckpt in the obstacle environment for the ACT-only condition, then the same checkpoint with the 40-sensor Safety-CVAE residual, in a task that explicitly approves simulator rollouts. The evaluator must verify checkpoint_manifest.json before rolling out.

Points for whoever picks it up:

1. **Verify `checkpoint_manifest.json` before rolling out.** It exists so a
   checkpoint trained on a different split or different statistics can be
   rejected mechanically.
2. **Use this one checkpoint for both arms.** Training a second nominal ACT for
   the safety condition would confound the comparison.
3. **The offline numbers here are not a success rate.** Nothing in this task
   measured task success, collision avoidance or safety.
4. Seeds 1 and 2 were deliberately not trained; a multi-seed study is its own
   approved task.

## 19. Decision

| | |
|---|---|
| Canonical dataset hashes match | True |
| Exact committed 80/20 split used | True |
| Normalization train-only | True |
| No leakage | True |
| Temporal alignment and padding correct | True |
| Full-size smoke passes | True |
| 2000-epoch seed-0 training completed | True |
| policy_best.ckpt and dataset_stats.pkl reload correctly | True |
| Checkpoint provenance complete | True |
| No proximity or Safety-CVAE input consumed | True |
| Mandatory final artifacts written | True |

ACT_BASELINE_TRAINED_AND_PINNED
