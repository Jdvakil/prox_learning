# Hybrid Safety Stack Final Decision

## Executive summary

Decision: `MISSING_CANONICAL_ARTIFACTS`.

The canonical 40-sensor observation contract and canonical Safety-CVAE are present and
internally consistent. The repository did not contain a live ACT + Safety-CVAE adapter at
the start, so an isolated adapter and bounded paired launcher were implemented in ACT
without changing ACT, Safety-CVAE, MolmoSpaces, geometry, or planner architectures.

A live paired smoke was not started. The exact obstacle ACT `policy_best.ckpt`, its
`dataset_stats.pkl`, and the source `hybrid_obstacle_v1` dataset are absent locally and
were not recoverable from Git history, GitHub releases/actions, or any of the eight
public Hugging Face dataset revisions. The new evaluator deliberately rejects an
unpinned or mismatched checkpoint before policy construction.

Separately, the existing physical sensor verifier reproduces its historical `PASS 38/40`
result. `link5_front_sensor_1` and `link5_front_sensor_2` see self geometry before the
test plate. The current CSV is byte-identical to the pre-existing
`diagnostics_output/20260611_hybrid_sensor_verify/sensor_verify.csv`; no geometry was
changed.

No ACT or Safety-CVAE training, architecture/weight change, planner change, geometry
change, large collection, or live rollout was performed. No 29-sensor PACT code was
merged.

## Source state and preservation

| Component | Starting state | Final implementation state |
|---|---|---|
| Root | `deep_cavity_v2` at `37087ef11baa81613430a1f1e9709f205b52f541` | `repair/hybrid-safety-act-integration`; implementation commit `d396f6f044cfc813c866279f31b0980cc5dac22b` |
| ACT | gitlink `526dd8f475a5195dd6e238d489f16dbfcfcbf541` | `repair/live-safety-residual-eval` at `3d25c69edd8d972afa59fec5c3edb9d13a357f92` |
| MolmoSpaces | gitlink `c817f07b0fffc55a0dce1577312e0a7afc473b69` | unchanged at `c817f07b0fffc55a0dce1577312e0a7afc473b69`; no repair branch needed |
| MolmoBot | gitlink `4ff337dfc60708f46d5831430df807fa90014821` | unchanged |

The decision documents are committed immediately after the implementation commit.
Because a Git commit cannot embed its own hash, resolve the exact containing decision
commit with `git rev-parse HEAD`; the exact tip is also reported in the final handoff.

The archived transformer-PACT track remains isolated in `/root/prox_learning`. Its root,
ACT, and MolmoSpaces worktrees are clean, and the following stashes remain preserved:

- Root: `575937179ba1a5c596f56d786196331028652e25`
- Root pre-existing: `be5b715b05b21f7f98bfe29e8b6d7da4f2993b70`
- ACT: `1fdebe6b72ca9644d51ba4984c82eb7834c5f3d6`
- MolmoSpaces: `cfbb0df86ccc61339a39a82f094e3511764784d7`
- MolmoSpaces pre-existing: `7f7b0740c4f2e28e8aa77c461c8f45f835f1a694`

GitHub CLI is authenticated as `Lundii1` using HTTPS. Direct SSH still returns
`Permission denied (publickey)`, but authenticated HTTPS access works, submodules were
initialized with command-scoped URL overrides, and `.gitmodules` was not changed.

## Artifact inventory

All required code and canonical Safety-CVAE artifacts exist. Full sizes and SHA-256
digests are in
`diagnostics_output/hybrid_safety_stack/artifact_inventory.json`.

| Artifact | Status | SHA-256 |
|---|---|---|
| `assets/robots/franka_skin/model_hybrid.xml` | present | `50924661e0411f92ab529c790512b17b674e789434c592c3dbc6d2359164d4c6` |
| `assets/safety/cvae_v3/model.pt` | present | `1fb2fc2b6023e64d2b9cbcf67fd5a24402968ec6f902c1e8a8595690396e7405` |
| `assets/safety/cvae_v3/meta.json` | present | `7c873756fb16e4f3fc565f96c41a581816981ab93886f1db5d27e62110a5fc81` |
| `assets/safety/cvae_v3/config.json` | present | `c376f44316e1edfc5c198ad3a8ca64b69e1e8e54c6f3125a082020172ac46060` |
| `assets/safety/cvae_v3/history.json` | present | `0bd2910383b46d2ea411bfd1cd8b83b14e108de060b52d432421f6820510489f` |
| `assets/safety/sweep_v3.h5` | present | `4161d62eba53320ab83661adb0a1d3e24244f71e24f00b6390cffa794ac783e8` |
| `scripts/train_safety_cvae.py` | present | `c7e68bd0d20877408b58bbebcfe0394688126eb37220a366b7298c9b477a594a` |
| `scripts/safety_react_demo.py` | present | `69364a0157afda36ce0ce000948d289b8f8b8ce1e487f7eafa811f7870585baa` |
| `submodules/act/eval_act_obstacle.py` | present | `0ac9d766a4ab409bf5345722d8399fd5f98003e94378bca688910b9ce1d48ba1` |
| Other required train/sweep/demo/build/verify/convert scripts | present | recorded in inventory JSON |

The following canonical runtime artifacts are missing:

| Missing artifact | Recovered provenance | Result |
|---|---|---|
| Obstacle ACT `policy_best.ckpt` | Historical path: `/home/jaydv/code/prox_learning/submodules/act/ckpts/act_obstacle_baseline_v1/obstacle_baseline/20260616_002504_obstacle_baseline_2000_100_1e-05_0/policy_best.ckpt` | file not present |
| Matching `dataset_stats.pkl` | Same historical checkpoint directory | file not present |
| `hybrid_obstacle_v1` source dataset | `assets/datagen/hybrid_obstacle_v1/FrankaSkinHybridObstacleConfig/20260612_183855` | directory not present |

Recovery evidence:

- A `/root` and `/tmp` search found no canonical checkpoint, statistics, or source
  dataset.
- ACT commit `b89fd84332c563b0ffb9f6d6b77634d7e1762870` contains an April checkpoint, but it
  predates the June hybrid-obstacle stack and is paired with a 74-byte placeholder
  statistics file containing scalar means of 0 and standard deviations of 1.
- ACT commit `0a1f7aec4775b5b43a86539c6a9703c0b427fcb2` has March statistics but no matching
  canonical obstacle checkpoint.
- GitHub has no release, tag, or Actions artifact containing the files.
- All eight revisions of `jdvakil/prox_learning_data` were enumerated. The current
  revision contains 34 `act_obstacle_baseline_v1` evaluation files but no checkpoint,
  statistics, or source dataset.
- The Hugging Face `running_log.log` proves the canonical checkpoint once loaded from
  the path above and records a historical ACT-only result of 2/5. That historical
  output is not a paired Safety-CVAE evaluation and cannot authenticate or replace the
  missing files.
- ACT training logs scalar metrics to Weights & Biases but has no `wandb.Artifact` or
  `wandb.save` call for either required file. No local W&B API key was available.
- No repository download manifest or command exists for these three missing artifacts.

The complete recovery record is
`diagnostics_output/hybrid_safety_stack/artifact_recovery_audit.json`.

## 40-sensor observation contract

`FrankaSkinHybridObstacleConfig` uses `FrankaSkinHybridRobotConfig` and
`FrankaSkinHybridCameraSystem`; the robot config resolves to `model_hybrid.xml`.
The camera system contains two ACT RGB cameras plus exactly 40 unique proximity
cameras. `get_core_sensors()` emits exactly 40 proximity observation keys, each with
shape `(4, 8, 8)` at a 66 ms policy step and 16.6667 ms proximity period. Live safety
inference selects the latest substep, yielding exactly `(40, 8, 8)`.

The canonical inference order is the order stored in both `sweep_v3.h5["sensors"]` and
`cvae_v3/meta.json`, not raw XML declaration order:

```text
00 link1_sensor_0          01 link1_sensor_1
02 link1_sensor_2          03 link1_sensor_3
04 link1_sensor_4          05 link1_sensor_5
06 link1_sensor_6          07 link2_sensor_0
08 link2_sensor_1          09 link2_sensor_2
10 link2_sensor_3          11 link2_sensor_4
12 link2_sensor_5          13 link2_sensor_6
14 link3_sensor_0          15 link3_sensor_1
16 link3_sensor_2          17 link3_sensor_3
18 link3_sensor_4          19 link4_sensor_0
20 link4_sensor_1          21 link4_sensor_2
22 link4_sensor_3          23 link4_sensor_4
24 link5_back_sensor_0     25 link5_back_sensor_1
26 link5_back_sensor_2     27 link5_back_sensor_3
28 link5_back_sensor_4     29 link5_back_sensor_5
30 link5_front_sensor_0    31 link5_front_sensor_1
32 link5_front_sensor_2    33 link5_front_sensor_3
34 link6_sensor_0          35 link6_sensor_1
36 link6_sensor_2          37 link6_sensor_3
38 link6_sensor_4          39 link6_sensor_5
```

The order hash is
`c31df8c36b0011b0eaf5b2eb5ce66d2514b5d6662ba9d7684ff021cd17cec858`.
Link assignment is:

- indices 0–6: `link1_skin`
- indices 7–13: `link2_skin`
- indices 14–18: `link3_skin`
- indices 19–23: `link4_skin`
- indices 24–29: `link5_back_skin`
- indices 30–33: `link5_front_skin`
- indices 34–39: `link6_skin`

The machine manifest includes each name, index, link, shape, model hash, camera-config
hash, declaration orders, timing, and 13 passing consistency checks:
`diagnostics_output/hybrid_safety_stack/sensor_order_manifest.json`.

The physical plate verifier returns `PASS 38/40`. The two failures are:

| Sensor | Nearest self return | Expected plate | Measured plate read | Finding |
|---|---:|---:|---:|---|
| `link5_front_sensor_1` | 0.058 m | 0.145 m | 0.088 m | self return precedes plate |
| `link5_front_sensor_2` | 0.035 m | 0.145 m | 0.051 m | self return precedes plate |

This is not an ordering/count mismatch, but it remains an unresolved physical sensor
finding. Geometry was intentionally left unchanged.

## Canonical Safety-CVAE

The checkpoint was loaded only through:

```python
SafetyHead.load("assets/safety/cvae_v3")
```

Verified behavior:

- Input `(40, 8, 8)` produces output `(7,)`.
- Repeated `z=0` inference is bitwise deterministic.
- Preprocessing is exactly
  `closeness = clip(1 - depth / 0.5, 0, 1)`.
- Depth below `0.005 m` maps to zero closeness.
- Depth at or above `0.5 m`, including renderer far/no-return values, maps to zero
  closeness.
- All-far input has physical output norm `0.38400027`, or `0.03380479` after the
  saved label scale. An identical far reference subtracts to exactly zero.
- A `0.05 m` close patch on canonical sensor index 18 changes the physical output by
  norm `78.86090`, establishing nonzero retreat behavior.
- Metadata label scale is `11.359346389770508` and reproduces exactly.

`sweep_v3.h5` contains 15,000 samples of `(40, 8, 8)` and labels of `(7,)`. Recreating
the seed-0 validation split gives:

| Metric | Metadata | Recomputed |
|---|---:|---:|
| Best validation MSE | 0.0090693403 | 0.0090693394 |
| Close direction cosine | 0.9255079031 | 0.9255078435 |
| Far quiet norm | 0.0310112610 | 0.0310112610 |

All 15 machine checks pass in
`diagnostics_output/hybrid_safety_stack/safety_cvae_audit.json`.

## Actual integration status

At the starting commit:

- `scripts/safety_react_demo.py` was a recorded/synthetic nominal-trajectory demo. It
  directly poses MuJoCo along a saved or generated nominal trajectory and rerenders
  parked obstacles; it does not run `ACTPolicy`.
- `submodules/act/eval_act_obstacle.py` was live vanilla ACT with exo RGB, wrist RGB,
  and qpos, but had no Safety-CVAE inference or residual controller.
- Therefore, no live ACT + Safety-CVAE adapter existed.

The isolated ACT adapter is `eval_act_obstacle_safety.py`. It:

- reuses the unchanged vanilla `ACTInferencePolicy` inference, chunking, and temporal
  aggregation;
- loads the safety model only through `SafetyHead.load`;
- validates model/config/sensor/Safety hashes and requires explicit ACT checkpoint and
  statistics SHA-256 values;
- consumes all and only the 40 canonical sensor keys in pinned order;
- preserves the executable demo semantics, including the saved label-scale division:

```text
dq_delta_physical = head(current_skin) - head(reference_skin)
dq_delta = dq_delta_physical / label_scale
dq_filtered = ema * dq_filtered + (1 - ema) * dq_delta
correction += (gain * dq_filtered - decay * correction) * dt
q_exec_arm = q_nominal_arm + clip(correction, -max_dev, +max_dev)
```

- uses the existing constants `gain=4.0`, `decay=2.2`, `ema=0.75`, and
  `max_dev=0.35`; no tuning was performed;
- derives `dt=0.066 s` from the live environment clock;
- supports an explicit frame-aligned reference H5 or, when absent, the first live skin
  observation as the reference;
- changes only seven arm joints and copies ACT's gripper command verbatim;
- supports `act_only`, `normal`, `zero`, `delayed`, and `shuffled` causal controls;
- logs nominal action, raw/baseline/subtracted safety outputs, filtered correction,
  executed action, active sensors/links, minimum non-floor environment distance,
  non-floor robot/environment collision pairs and penetration, phase, and success;
- stores controller constants, artifact hashes, sensor order/hash, per-frame records,
  and episode metrics through the existing MolmoSpaces `obs_scene` path.

`run_paired_hybrid_safety_eval.py` launches exactly ACT-only then normal safety for each
unique seed, rejects more than four seeds, and therefore cannot exceed eight rollouts.
It seeds NumPy before MolmoSpaces config class defaults are evaluated, uses identical
ACT settings for both arms, rejects nonempty output directories, verifies identical
scene parameters/target object/first nominal ACT command, and computes trajectory
divergence and task-success loss.

## Bounded paired smoke

| Item | Result |
|---|---|
| Paired seeds requested | 0 |
| ACT-only rollouts | 0 |
| ACT + Safety-CVAE rollouts | 0 |
| Total live rollouts | 0 |
| Limit respected | yes, 0/8 |
| Reason skipped | canonical ACT checkpoint and matching statistics cannot be supplied or hash-pinned |

No task, collision, clearance, activation, correction, or paired-divergence result is
claimed. A four-seed dry run generated exactly eight commands and proved the launcher
limit, but it did not execute them.

## Validation results

| Validation | Result |
|---|---|
| Hybrid artifact audit | pass for required code and Safety assets; three canonical runtime artifacts missing |
| Sensor contract audit | 13/13 checks pass |
| Safety-CVAE audit | 15/15 checks pass |
| ACT residual unit tests | 18 passed |
| Safety demo CLI import/help | 5/5 pass after installing Foxglove only in the disposable test venv |
| Hybrid sensor verifier | `PASS 38/40`; two pre-existing link-5 self-return findings |
| Live evaluator import/help | pass |
| Four-seed paired dry-run plan | pass; exactly 4 pairs/8 commands |
| MolmoSpaces H5 log persistence | pass; 40 names, order hash, and representative frame round-trip exactly |
| Python byte compilation | pass |
| Ruff on changed Python files | pass |
| JSON validation | pass |
| `git diff --check` | pass |
| Strict Safety-CVAE state load | pass through `SafetyHead.load` |
| Strict ACT state load | not run; checkpoint is missing and evaluator fails explicitly before load |
| Live paired smoke | not run; missing canonical ACT artifacts |

The full demos cannot run without the missing source posture dataset; their CLI/import
surfaces and the canonical head itself were tested without generating demo output.

## Changed files and commits

ACT commit `3d25c69edd8d972afa59fec5c3edb9d13a357f92`:

- `eval_act_obstacle_safety.py`
- `hybrid_safety_residual.py`
- `run_paired_hybrid_safety_eval.py`
- `tests/test_hybrid_safety_residual.py`

Root implementation commit `d396f6f044cfc813c866279f31b0980cc5dac22b`:

- `configs/hybrid_safety_stack_v1.json`
- `scripts/audit_hybrid_safety_stack.py`
- portable-only changes to `scripts/verify_hybrid_skin_sensors.py`
- machine audit JSON, verifier CSV/plot, and starting-state record
- ACT gitlink update

Decision-artifact commit:

- `scripts/test_hybrid_safety_log_storage.py`
- `docs/HYBRID_SAFETY_STACK_FINAL_DECISION.md`
- `diagnostics_output/hybrid_safety_stack/final_decision.json`
- final refreshed audit/validation records

MolmoSpaces has no source change or commit. No existing checkpoint, stash, branch, or
experimental output was deleted, reset, rewritten, merged, popped, or dropped. Nothing
was pushed.

## Reproduction commands

Run from the root worktree with an environment containing the repository dependencies:

```bash
export PYTHONPATH="$PWD/submodules/act:$PWD/submodules/molmospaces:$PWD/scripts"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

python scripts/audit_hybrid_safety_stack.py
python -m pytest -q submodules/act/tests/test_hybrid_safety_residual.py
python scripts/verify_hybrid_skin_sensors.py
python scripts/test_hybrid_safety_log_storage.py
python submodules/act/eval_act_obstacle_safety.py --help
python submodules/act/run_paired_hybrid_safety_eval.py --help
python -m compileall -q scripts submodules/act
ruff check scripts/audit_hybrid_safety_stack.py \
  scripts/test_hybrid_safety_log_storage.py \
  scripts/verify_hybrid_skin_sensors.py \
  submodules/act/hybrid_safety_residual.py \
  submodules/act/eval_act_obstacle_safety.py \
  submodules/act/run_paired_hybrid_safety_eval.py \
  submodules/act/tests/test_hybrid_safety_residual.py
find configs diagnostics_output/hybrid_safety_stack -name '*.json' \
  -print0 | xargs -0 -n1 python -m json.tool >/dev/null
git diff --check
git submodule status
```

After the exact checkpoint and statistics are restored and SHA-256 pinned, a bounded
dry run can be inspected before any live execution:

```bash
python submodules/act/run_paired_hybrid_safety_eval.py \
  --ckpt_dir /path/to/20260616_002504_obstacle_baseline_2000_100_1e-05_0 \
  --expected-act-checkpoint-sha256 <64-hex-sha256> \
  --expected-dataset-stats-sha256 <64-hex-sha256> \
  --output_dir /new/empty/output \
  --seeds <seed1> <seed2> <seed3> <seed4> \
  --dry-run
```

Removing `--dry-run` is not recommended until artifact provenance is resolved and the
two physical link-5 verifier findings are explicitly accepted or repaired under a
separate geometry-authorized task.

## Recommended next task

Recover the exact files from the historical checkpoint directory:

1. `policy_best.ckpt`
2. `dataset_stats.pkl`

Record their SHA-256 values and provenance, and restore or separately archive the
`hybrid_obstacle_v1` source dataset. Then rerun the static suite, perform strict ACT
loading, review the two link-5 physical verifier findings without silently changing
geometry, and run at most four predeclared paired seeds using the existing launcher.
Do not retrain either model as part of that recovery task.

MISSING_CANONICAL_ARTIFACTS
