# Hybrid Clean Retrain Final Decision

## Executive summary

Decision: `CANONICAL_DATA_COLLECTION_FAILED`.

The frozen canonical collection was executed exactly as the approved manifest
prescribes, ran to completion, and produced 175 source trajectories of which 165
are successful — comfortably above the manifest's 100-success minimum. It did
**not** satisfy the manifest's hazard-distribution acceptance gate. Hazard
presence is **0.6900** across the 100 episodes the committed converter would
select (69 of 100; 70 required), and it is below the 0.70 floor at *every* level
of aggregation: 0.6942 at the sampler's own accepted-theta draw, 0.6857 across
all written trajectories, 0.6909 across successful trajectories.

The deficit is not a selection artefact and not a configuration change. The
committed `ObstacleFumehoodPickSampler.OBSTACLE_P` is 0.75 and the config file
hash matches the pinned value; the realized Bernoulli draw simply came in about
two sigma low (168 hazard of 242 accepted theta draws, two-sided p ≈ 0.057).
Every one of the seven collected houses lands between 0.625 and 0.708 — none
reaches 0.75.

Because the gate is stated on the accepted/selected demonstrations and the
shortfall is one episode, the only ways to cross it would be to re-run the
frozen collection and keep whichever run's hazard draw passes, or to re-select
the 100 episodes on the gated statistic. Both are selection on the very quantity
being gated, and both are forbidden by this task's no-tuning and
no-silent-restart constraints. Per the task's stop rule, conversion, statistics,
ACT training, bundling, the baseline runtime check and the paired live smoke were
**not started**.

Live rollouts used: **0 of 8**. ACT was not trained. The Safety-CVAE, residual
controller, adapter, sensor model, environment and planner were not modified.
No artifacts were deleted or overwritten and nothing was pushed or uploaded.

A second, independent defect was found and is reported below: one MolmoSpaces
worker hung and its entire house was silently lost while the pipeline still
exited 0 claiming "skipped 0 houses". This did not cause the gate failure, but it
must be fixed before any re-run.

## Starting and final branches and commits

| Component | State |
|---|---|
| Root branch | `train/hybrid-obstacle-act-clean-v1`, created from `6a15ac3b13ea329de5e46fdca80e230efb3a7b03` |
| Root commit at start | `6a15ac3b13ea329de5e46fdca80e230efb3a7b03` (clean) |
| ACT gitlink | `3d25c69edd8d972afa59fec5c3edb9d13a357f92` (clean, unmodified) |
| MolmoSpaces gitlink | `c817f07b0fffc55a0dce1577312e0a7afc473b69` (clean, unmodified) |
| MolmoBot gitlink | `4ff337dfc60708f46d5831430df807fa90014821` (unchanged) |
| Worktree | `/root/prox_learning_act_retrain` |

Not worked on, as required: `recovery/hybrid-act-canonical-artifacts`,
`repair/hybrid-safety-act-integration`, `repair/live-safety-residual-eval`. The
recovery manifest was read from the recovery branch without merging any recovery
tooling into the training branch.

## Manifest verification

`diagnostics_output/hybrid_act_artifact_recovery/clean_retraining_manifest.json`

| Check | Result |
|---|---|
| SHA-256 | `e81d7f372c5a8dae4e5ca03a2801c39c0aed8aa633a850cd65080c9dc3acb3ca` — matches expected |
| Pinned commits | root / ACT / MolmoSpaces all match the checked-out gitlinks |
| Environment / config name | `FrankaSkinHybridObstacleConfig` — imports, resets, matches |
| Demonstration target | 100 converted successful episodes; ≥100 source successes before conversion |
| Hazard requirement | target 0.75, accepted range [0.70, 0.80] |
| Conversion command | `scripts/convert_obstacle_to_act.py`, 240×320, qpos 9, action 8 |
| Split algorithm | `numpy.random.RandomState(1).permutation(100)`, 80/20 |
| Statistics algorithm | `submodules/act/utils.py:get_norm_stats` |
| ACT hyperparameters | 2000 epochs, batch 8, lr 1e-5, chunk 100, hidden 512, ff 3200, KL 10 |
| Training seed | `[0]` |
| Output conventions | `ckpts/act_obstacle_clean_retrain/obstacle_baseline/<UTC>_...` |
| Paired evaluation seeds | `[0, 1, 2, 3]`, 2 rollouts each, 8 total |

No required field was absent or in conflict with the committed source, so
`RETRAINING_MANIFEST_INCOMPLETE` does not apply.

The manifest prescribes **8 wrap-around house IDs × 25 = 200** configured
trajectories. The five-house / 125-attempt figure appears only under
`historical_reference_only` and was therefore not used.

### Canonical frozen hashes — all verified

| Artifact | SHA-256 | Match |
|---|---|---|
| Hybrid contract | `aef29d76…fc90df2b` | ✔ |
| Safety-CVAE model | `1fb2fc2b…396f7405` | ✔ |
| Safety-CVAE metadata | `7c873756…e22110a5fc81` | ✔ |
| Live adapter | `21e8ccbe…c882b8699c292ee` | ✔ |
| Residual controller | `655a2e92…c200443eaeeca` | ✔ |
| Paired launcher | `4623ce5f…172692a424e3fc2` | ✔ |
| Sensor order | `c31df8c3…f021cd17cec858` | ✔ |
| Converter | `74b60458…76994856695` | ✔ |
| `utils.py` (get_norm_stats) | `494ea056…dbc3060bbf84` | ✔ |
| MolmoSpaces datagen config | `cd8891e0…8aeeb3b8ff3f` | ✔ |
| `model_hybrid.xml` | `50924661…6d2359164d4c6` | ✔ |

The sensor-order value is a *content* hash over the canonical JSON encoding of the
40 ordered names (`hybrid_safety_residual.sensor_order_hash`), not the SHA-256 of
the manifest file; recomputing it from the contract reproduces
`c31df8c36b0011b0eaf5b2eb5ce66d2514b5d6662ba9d7684ff021cd17cec858` exactly.

## Environment

Python 3.11.15; torch 2.7.1+cu126 (CUDA 12.6, cuDNN 9.5.1); numpy 2.4.6;
h5py 3.16.0; opencv 5.0.0; einops 0.8.2; NVIDIA A10 (23 GB, driver 570.86.10);
128 CPUs; 503 GB RAM; EGL via `libEGL_nvidia`, `GL_RENDERER = NVIDIA A10/PCIe/SSE2`.

No usable Python environment existed on the host, so one was built. Two pins were
required to make the *pinned, unmodified* MolmoSpaces code run at all:

- `mujoco==3.5.0` — `mujoco-warp 3.5.0.2` requires `mjENBL_MULTICCD`, removed in 3.10.0.
- `warp-lang==1.11.1` — with 1.15.0, `warp_kinematics.ik` raises
  `RuntimeError: Incompatible array data types`, so every rollout fails at
  `task.reset()` and the run produces zero episodes.

Both are dependency-resolution pins. No repository source was changed.

Space gate: estimated remaining output 12.76 GiB, 2× requirement 25.53 GiB,
actual free 200.79 GiB — satisfied before collection started.

## Pre-collection static validation

| Check | Result |
|---|---|
| `FrankaSkinHybridObstacleConfig` imports and resets | pass |
| Uses `FrankaSkinHybridRobotConfig` | pass (`model_hybrid.xml`) |
| Uses `FrankaSkinHybridCameraSystem` | pass |
| Exactly 40 proximity sensors in canonical order | pass (source H5 order hashes to the canonical value) |
| `exo_camera_1` and `wrist_camera` exist | pass (42 cameras: 2 RGB + 40 sensors) |
| qpos dimension 9 | pass (arm 7 + gripper 2) |
| Expert action dimension 8 | pass (arm 7 + gripper 1) |
| Target is the canonical single red cup | pass — pool index 1 = `4afa0cdde045417ab31f98ae7745b039`; all 8 house IDs ≡ 1 (mod 24) |
| Hazard generation follows the committed ~75% distribution | code path confirmed: `OBSTACLE_P = 0.75` |
| Collision and penetration logging work | collision metrics in datagen; `maximum_penetration_m` in the live adapter |
| Raw source H5 retains all required fields | pass: RGB, qpos, qvel, actions, 40 proximity streams, task state, obstacle state, env states, seeds |
| Converter excludes skin from vanilla ACT inputs | pass — no proximity written, verified on converted output |

Existing suites run before collection: ACT hybrid-safety/residual tests **18
passed**; hybrid safety log-storage test **pass**; hybrid stack audit **13/13**
with `sensor_order_manifest.json` regenerated byte-identically (only
branch/commit identity differed, and the committed file was restored).

### Known physical verifier condition — reproduced

`scripts/verify_hybrid_skin_sensors.py` → **PASS 38/40**, with exactly
`link5_front_sensor_1` (self_min 0.058) and `link5_front_sensor_2` (self_min
0.035) failing. Geometry was not changed. Recorded as part of the canonical
reproduction.

Baseline subtraction was checked before any live execution: feeding identical
current and reference skin through the canonical Safety-CVAE (label_scale
11.359346) over 40 real proximity frames from collected source data yields
`max|subtracted_dq| = 0.0` and `max|correction| = 0.0` — exact zero, matching the
committed tests' `assert_array_equal(..., np.zeros(7))` tolerance. The two known
self-returning sensors therefore cancel through the reference-subtraction path.

## Collection

Command (executed verbatim, one immutable collection ID, no restarts, no
parameter changes):

```
cd /root/prox_learning_act_retrain/submodules/molmospaces && \
PYTHONHASHSEED=0 OMP_NUM_THREADS=2 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
/root/act_retrain_venv/bin/python -m molmo_spaces.data_generation.main FrankaSkinHybridObstacleConfig
```

Collection ID `20260724_183407`; 18:33:41Z → 20:54:21Z; exit 0.

| Quantity | Value |
|---|---|
| Configured | 8 houses × 25 = 200 trajectories, 250 attempts/house budget |
| Houses written | **7 of 8** (`house_1` missing — see defect below) |
| Trajectories written | 175 |
| Successful demonstrations | **165** |
| Failed demonstrations | 10 |
| Success rate | 94.29% |
| Planner rejections (not written) | 49 (41 lift-pose IK, 8 pregrasp IK) |
| Hazard present (all written) | 120 / 175 = **0.6857** |
| Hazard present (successful) | 114 / 165 = **0.6909** |
| Hazard present (selected 100) | 69 / 100 = **0.6900** |
| Target object | single UID `4afa0cdde045417ab31f98ae7745b039`, all trajectories |
| Proximity streams | all 175 trajectories expose exactly 40 sensors |
| Missing / corrupt required fields | 0 |
| Episode length T (selected) | min 53, median 87, max 195 |
| Collisions (selected) | 100/100 trajectories register contacts; 22,513 total contacts |
| Total bytes | 1,431,940,193 (1.33 GiB), 1059 files |

Per-house successful counts: `house_25` 24/25, `house_49` 24/25, `house_73`
24/25, `house_97` 23/25, `house_121` 23/25, `house_145` 23/25, `house_169` 24/25.

Per-house hazard fraction among successful trajectories: 0.696, 0.708, 0.708,
0.696, 0.696, 0.696, 0.625 — every house below the 0.75 target.

Per-file SHA-256 for all 1059 source files, the sorted source-file manifest, and
the one-to-one trajectory→file mapping are in
`/root/act_retrain_provenance/source_manifest.json` and `collection_summary.json`.

## Source provenance freeze

All writers closed (process exited 0). All 7 H5 files open and read cleanly.
Every source H5, MP4, config pickle and log hashed.

- Source content-tree SHA-256: `09c98aee08d015b3a561b08674415df9a4ed398186940207f41ef384251cdf24`
- Re-verified byte-identical after marking the tree read-only (`chmod -R a-w`).
- Conversion was never attempted in place; the source tree is untouched.

## Failing gate

**Gate**: hazard presence between 70% and 80% across the selected source
demonstrations (`acceptance_gates[1]`, and
`pinned_source.source_semantics.accepted_hazard_fraction_range = [0.7, 0.8]`).

| Level of aggregation | Hazard fraction | In [0.70, 0.80] |
|---|---|---|
| Sampler accepted-theta draws (242) | 0.6942 | ✘ |
| All written trajectories (175) | 0.6857 | ✘ |
| Successful trajectories (165) | 0.6909 | ✘ |
| **Selected first 100 (the gated set)** | **0.6900** | **✘** |

Shortfall on the gated set: **1 episode** (69 present; 70 required).

**Mechanism.** The deficit is present in the sampler's own Bernoulli draw, not
introduced downstream. The log records 242 accepted theta draws of which 168 set
`cell=bar`, a rate of 0.6942 against the configured 0.75. Under
Binomial(242, 0.75) that is z = −2.00, two-sided p ≈ 0.057. Planner rejection is
approximately hazard-neutral: of the 67 drawn-but-unwritten episodes, 48 (71.6%)
were hazard-present, close to the overall rate, so the 49 IK rejections do not
explain the shortfall. `OBSTACLE_P` is 0.75 in the committed source and the
config file hash matches the pinned value, so this is a low draw under the frozen
configuration rather than a configuration error.

**Why it is not remediable inside this task.** Crossing the gate would require
either re-running the frozen collection and keeping whichever run's hazard draw
passes, or re-selecting the 100 episodes on the gated statistic. Both are
selection on the quantity being gated and are forbidden by the no-tuning and
no-silent-restart constraints. Adding the missing `house_1` would also change the
gated set after the statistic had been inspected, and would mix a second
collection ID into the bundle.

## Independent defect: silent worker loss

Worker 0 emitted its last output at 19:03:53Z, mid-rollout on `house_1` episode
22 with 17 trajectories already collected, and never produced another line. It
never wrote `house_1/trajectories_batch_1_of_1.h5`. Workers 1–3 completed
normally. The pipeline then reported:

```
Completed 7 houses, skipped 0 houses
Success count: 165, Total count: 175
Success rate: 94.29%
```

and exited **0**. MolmoSpaces buffers a house in memory and writes it only at
house completion, so a hung or dead worker silently discards its trajectories,
and the summary line reports "skipped 0 houses" despite one house being absent.

This did not cause the gate failure — `house_1`'s hazard draw would be ~0.69 in
expectation like every other house, and 165 successes already exceed the 100
minimum. It is reported because a collection that loses an eighth of its planned
data while exiting 0 is a provenance hazard that must be fixed before any re-run.
All eight house indices are congruent to 1 (mod 24) and select the identical
red-cup task — the committed docstring states the wrap-around indices exist
purely to parallelize — so the loss did not alter the task distribution.

## Not performed

Per the task's stop rule, these were **not started**: conversion to vanilla-ACT
format, deterministic split and statistics generation, canonical ACT training,
offline artifact validation, immutable bundle creation, the pre-live safety
baseline runtime check, and the bounded paired live smoke. There is therefore no
checkpoint, no `dataset_stats.pkl`, no bundle manifest and no smoke table to
report. Live rollout use is **0 of 8**.

The converter, split, statistics, strict-load, bundle and smoke tooling was
nonetheless written and exercised end to end on preflight data, so a re-run under
a re-approved manifest can proceed immediately:

- Converter chain validated on preflight output: converted `qpos`/`action` arrays
  are **byte-identical** to independently re-decoded source rows, no proximity
  present, cameras `exo_camera_1`/`wrist_camera` at 240×320×3, and the documented
  `T = T_h5 − 1` rule (trailing empty `{}` action row) holds.
- Deterministic split reproduces the manifest's 80/20 lists exactly through the
  trainer's own code path (`set_seed(1)` → `np.random.permutation(100)`).
- Canonical ACT architecture builds and runs: chunk 100, hidden 512, ff 3200,
  qpos 9, action 8, cameras `['exo_camera_1','wrist_camera']`, KL 10,
  83,911,817 parameters, forward output `[1, 100, 8]`, finite.
- Paired launcher dry run: 4 seeds / 8 rollouts, `act_only` then `normal` per
  seed, and it correctly refuses 5 seeds.

## Validation results

| Item | Result |
|---|---|
| Hybrid stack audit | 13/13 pass |
| 40-sensor contract | pass, canonical content hash reproduced |
| Safety-CVAE + live residual tests | 18 passed |
| Data-collection/config static checks | pass |
| Converter chain | pass (preflight, byte-identical provenance) |
| Split / statistics tooling | split verified exact; statistics not computed (no converted dataset) |
| Strict ACT loading | architecture verified; no checkpoint to load |
| Offline evaluator / bundle / smoke | not run — blocked by the stop rule |
| Provenance freeze | pass, tree hash stable after read-only |
| Paired launcher dry run | pass, bounds enforced |
| Python byte compilation | pass |
| Ruff on new tooling | 3 findings (blind-except diagnostics); committed baseline scripts report 9 under the same invocation |
| JSON / Markdown validation | pass |
| `git diff --check` | clean |
| Worktree | clean apart from the new audit tooling; all three submodules clean and at pinned commits |

## Changed files and commits

Committed to `train/hybrid-obstacle-act-clean-v1` (not pushed): the five new
lightweight audit tools under `scripts/`, this report, and the machine-readable
decision. No source data, converted episodes, checkpoints, videos or archives
were committed. No prior artifact was deleted or overwritten.

Preserved outside Git:

- Source collection (read-only):
  `/root/act_retrain_assets/datagen/hybrid_obstacle_v1/FrankaSkinHybridObstacleConfig/20260724_183407`
- Provenance, logs and the append-only attempts ledger: `/root/act_retrain_provenance/`

## Reproduction commands

```bash
# environment
source /root/act_retrain_env.sh

# worktree and pins
git worktree add -b train/hybrid-obstacle-act-clean-v1 \
    /root/prox_learning_act_retrain 6a15ac3b13ea329de5e46fdca80e230efb3a7b03
git -C /root/prox_learning_act_retrain -c protocol.file.allow=always \
    submodule update --init --recursive
uv venv --python 3.11 /root/act_retrain_venv
VIRTUAL_ENV=/root/act_retrain_venv uv pip install -e submodules/molmospaces
VIRTUAL_ENV=/root/act_retrain_venv uv pip install "mujoco==3.5.0" "warp-lang==1.11.1" ipython pyquaternion

# static validation
python scripts/audit_hybrid_safety_stack.py
python scripts/verify_hybrid_skin_sensors.py          # expect PASS 38/40
(cd submodules/act && python -m pytest tests/ -q)     # expect 18 passed

# collection (one immutable collection ID)
cd submodules/molmospaces && PYTHONHASHSEED=0 OMP_NUM_THREADS=2 \
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  python -m molmo_spaces.data_generation.main FrankaSkinHybridObstacleConfig

# provenance and the gate that failed
python scripts/clean_retrain_provenance.py source-manifest <RUN_DIR>
python scripts/clean_retrain_provenance.py conversion-provenance <RUN_DIR> <DST> --max_episodes 100
```

Future upload and clean-redownload commands are recorded but deliberately **not
executed**; no bundle exists to upload because training never started.

## Exact next recommended task

Diagnose and re-approve the hazard-fraction gate before any further retraining
attempt. Either (a) amend the approved manifest so the accepted range reflects
the sampler's realized rate under the frozen configuration — measured 0.6942 over
242 accepted theta draws, with 0.75 lying about two sigma above it — or
(b) change `ObstacleFumehoodPickSampler.OBSTACLE_P` so the realized rate centres
on 0.75, which is an environment change and therefore needs its own approval.
Independently, fix the MolmoSpaces worker-loss defect: a house is buffered in
memory and written only at house completion, so a hung worker silently discards
its trajectories while the pipeline still exits 0 reporting "skipped 0 houses".
Re-run the collection only under a re-approved manifest, as one new immutable
collection ID.

Do not begin larger evaluation, additional training seeds, Safety-CVAE tuning, or
paper-result generation on the basis of this report.

CANONICAL_DATA_COLLECTION_FAILED
