# PACT place corridor v5: re-recording the 152 kept rows as demonstrations

Executes Steps 1-3 of the recovery plan. Step 4 (ACT conversion) is **not** covered here and is
not authorized by this document.

## Why a re-record was needed

`scripts/run_pact_place_collection.py:33` imports `run_row` from `run_pact_place_expert_screen`.
That harness replaces the sampled sensor suite with a `qpos`/`tcp_pose` pair before the rollout,
so the 152 kept v5 episodes carry no proximity, no RGB and no action arrays. They are a valid
**screen record** and were never trainable demonstrations. Verified in this clone: the whole
`diagnostics_output/pact_place_corridor_v5_collection` tree contains zero non-JSON files, and
`assets/datagen/` held only `pact_collision_corridor_v2`.

The kept set was already decided. This work re-records it; it does not re-select it.

## What was built

| Path | Role |
|---|---|
| `scripts/pact_place_recovery_contract.py` | Freezes the 152 kept rows, each with the outcome its screen row recorded |
| `configs/pact_place_v5_recovery.json` | The frozen contract, `config_sha256 = 125db9ac9f1eafcbbf9ce5a741d2c684a410d46da662ce55a99de3965bf7b9ac` |
| `scripts/run_pact_place_recovery_datagen.py` | Step 1-2 producer: datagen pipeline, full sensor suite, `trajectory.h5` + MP4s |
| `scripts/verify_pact_place_recovery_keys.py` | Step 3 gate: asserts on the produced files, not on the config |

Output root: `assets/datagen/pact_place_corridor_v2/recovered_152/`. The v5 screen record is not
read-modified, only read.

The producer is modelled on `scripts/run_pact_collision_collection.py`, the proven producer for the
working reference collection. It deliberately does not import `run_row` from the screen harness.
Seed selection, the retry ladder, the initial-contact rejection filter, the expert, the scene and
the clean-success criterion are identical to the screen; only the observation suite differs.

## Two defects the plan did not anticipate

### 1. A second observation reduction, beyond the sensor-suite truncation

`run_pact_place_expert_screen._make_config` also sets `proximity_sensor_period_ms = 0.0`. With
sub-step recording disabled, `get_core_sensors` collapses `max_prox_substeps` to 1 and every
proximity dataset is written as `(T, 1, 8, 8)`.

The trainable schema requires `(T, 4, 8, 8)`, and `convert_pact_collision_to_act.extract_proximity`
raises on any other shape. Had Step 1 been implemented by copying `_make_config` as the plan
described, all 152 episodes would have produced the wrong proximity rank and the entire conversion
would have failed after the compute was already spent.

The recovery config keeps the datagen default of 16.6667 ms. Confirmed arithmetic at
`sim_dt_ms = 2.0`, `policy_dt_ms = 66.0`: `n_sim_per_prox = 8`, `total_sim_steps = 33`,
`max_substeps = 4`.

### 2. `obs_scene` is not JSON-serializable for the place task

`molmo_spaces/utils/save_utils.py:374` calls a bare `json.dumps` on `obs_scene`. The place sampler
puts `cam_visible` into `scene_params` as a `numpy.bool_`, which NumPy 2 reports as type `bool` and
`json` refuses. The collision corridor has no such flag, which is why its producer never hit this.

This aborted the first smoke episode **after** a completed 10-minute rollout and after all three
MP4s had been written. Fixed in the recovery publisher by coercing the block with `_jsonable`
before handing it to `prepare_episode_for_saving`, rather than patching the shared submodule and
disturbing the reference collection's producer.

The staging cleanup was also changed from `rmdir` to `shutil.rmtree`: the original leaves staged
MP4s behind when publication fails partway, which a retried row would then refuse to overwrite.

## An infrastructure limit, not a science result

The first 12-worker launch died instantly with every row in the batch reporting
`unreconciled_worker_failure`. The only diagnostic was
`libgomp: Thread creation failed: Resource temporarily unavailable`.

Cause: this container's `/sys/fs/cgroup/pids.max` is 3840. Each worker sized its libgomp, BLAS and
torch pools to the 128-core count, costing ~319 tasks; twelve workers exceed the ceiling before any
rollout begins. Those pools were pure overhead -- the uncapped single-worker baseline used 9m28s of
user time over 10m01s wall, about 1.07 cores.

No row crossed the scientific boundary, and the empty output tree was removed rather than resumed
onto. A related trap: killing the parent left 11 orphaned workers holding ~3500 threads, so the
ceiling stayed pinned until they were killed by PID.

Fixes in the runner:

- `THREAD_POOL_ENV` pins `OMP`/`MKL`/`OPENBLAS`/`NUMEXPR`/`VECLIB` thread counts to 1, set in the
  parent before the pool is built so spawned workers inherit capped pools from first import.
  Measured effect: **319 tasks per worker to 3**, at unchanged 101% CPU per worker.
- `_pids_headroom()` preflights the cgroup and refuses to launch a worker count that does not fit,
  reporting what it does afford. A silent mid-run collapse becomes a startup error.

Because `OMP_NUM_THREADS` changes BLAS reduction order and could in principle perturb a trajectory,
the capped configuration was checked for reproduction before the full launch rather than assumed
inert. Rows 000 and 001 reproduced exactly (532/532 and 465/465 steps).

## Step 2: reproduction

Each re-recorded row is checked against its screen row on `task_success`, `clean_success`,
`terminal_policy_phase` and `terminal_action_index`, plus selected seed, retry index and episode
step count. Divergence is reported, never substituted.

**152/152 rows reproduced exactly. Zero divergences.**

| Measure | Result |
|---|---|
| Rows complete | 152 / 152 |
| `clean_success` | 152 / 152 |
| Outcome reproduced (all four keys) | 152 / 152 |
| Selected seed reproduced | 152 / 152 |
| Episode step count reproduced | 152 / 152 |
| Divergences | none |

`recovery.json`, `recovery_sha256 = a704bccf9bec8ae6d0ac377e64bd22e3e0688e7d4eaaea91e7ee19553651c2e8`.

Every episode landed on the same seed, the same retry index, the same step count and the same
outcome as its screen row. The full sensor suite is therefore physics-neutral for this task, which
was the plan's stated stop-condition risk. It is now a measured result rather than an assumption.

Run: 12 workers, 152 episodes, 02:56 to 05:22 UTC on 2026-08-19 (2h26m), 5.6 GB of datagen output
at ~37 MB per episode. The plan's estimate of ~16 MB per episode was low; the reference episodes
are 158 steps while these run 243-634 (median 480).

## Step 3: the verification gate

The v5 collection passed every config-level check it had and still produced nothing trainable, so
this gate opens each produced file. Per episode:

- `trajectory.h5` exists and opens, with exactly one `traj_*` group
- `actions/joint_pos` decodes to a contiguous valid prefix of length T > 0, and
  `actions/commanded_action` is non-empty over that prefix
- `obs/agent/qpos` and `qvel` both decode to `(T, 9)`
- `obs/proximity` holds exactly the 40 contract sensors, each `(T, 4, 8, 8)` float32
- proximity is non-degenerate: finite everywhere and not constant (an all-zero skin would pass
  every shape check and teach nothing)
- `obs/sensor_param/<sensor>/extrinsic_cv` is `(T, 3, 4)` and `intrinsic_cv` is `(T, 3, 3)`
- exactly one `episode_*_wrist_camera.mp4`, which decodes and covers T frames

Report: `diagnostics_output/pact_place_v5_recovery/keys_verified.json`.

`conversion_authorized` flips true only on a clean pass over all 152. `training_authorized` is
hard-coded false in both the runner and the verifier.

**152/152 episodes pass all 14 checks.**

| Measure | Result |
|---|---|
| Episodes verified | 152 / 152 |
| Passed | 152 |
| Failed | 0 |
| Total timesteps | 72,955 |
| T per episode | min 243, median 480, max 634 |

Every one of the 14 checks passed on all 152 episodes: `trajectory_h5_exists`,
`trajectory_h5_opens`, `actions_joint_pos_nonempty`, `actions_commanded_nonempty`,
`qpos_qvel_present_and_aligned`, `proximity_sensor_set_exact`, `proximity_shapes_and_dtypes`,
`proximity_not_degenerate`, `obs_sensor_param_present`, `obs_sensor_data_present`,
`sensor_param_shapes`, `wrist_video_unique`, `wrist_video_decodes`, `wrist_video_covers_timesteps`.

`keys_verified_sha256 = b06ff8fcb3751a0eefc6adbafc569922df69f9d2c1d38d1036ff4e1dc59ea639`.
`conversion_authorized` is now true. `training_authorized` remains false.

152 `trajectory.h5` and 152 `episode_*_wrist_camera.mp4` exist on disk, against 0 non-JSON files
of any kind in the v5 screen tree. That contrast is the whole point of the recovery.

## Constraints honoured

- The v5 collection was not deleted or modified; recovery output went to a new root.
- `configs/pact_place_corridor_v5_collection.json` was not modified. The 12 protected artifacts are
  re-verified by `verify_protected_artifacts` at both start and end of every run.
- The expert, scene, success criterion and clean-success filter are unchanged.
- `run_row` from `run_pact_place_expert_screen` is not reused; the contract records
  `reuse_of_screen_run_row_forbidden`.

Re-verified after the run: all **12** protected artifacts hash unchanged; the v5 collection config
and summary both still pass their self-hash checks and still list 152 kept rows; the v5 tree still
contains 0 non-JSON files.

## Note for the v6 collection

The v6 plan says "collect ~255 clean under v6" without specifying the sensor suite. As written it
reproduces this failure exactly. Its collection step must require the datagen pipeline with cameras
and proximity enabled **and** a file-level key verification before any training is authorized.
Config-level review is what failed here: the v5 collection config was internally consistent,
self-hashed, and wrong.
