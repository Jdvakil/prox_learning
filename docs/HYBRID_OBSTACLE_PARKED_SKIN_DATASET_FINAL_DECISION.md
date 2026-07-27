# Hybrid obstacle — parked-skin supervision dataset, final decision

Data generation and integrity only. No model was trained, `development4` was not run, and
`confirmatory41` was not executed.

- Root branch `data/hybrid-obstacle-parked-skin-supervision-v1`, from `1343160`. The dataset
  was produced by `fdb217f` — the value the JSON records, since it cannot record the commit
  that carries itself; this document and that JSON land in `c6e09fd`.
- ACT branch `data/hybrid-obstacle-parked-skin-retention-v1` @ `91fc42a` (from `5e0d3b3`)
- MolmoSpaces `678f2eb`, unmodified working tree
- Machine-readable record: `diagnostics_output/hybrid_obstacle_parked_skin_dataset/final_decision.json`
  (sha256 `7aa6030d2c6330a34550837fd5220cb28a603c922a20f8bdca3d69ca1d68dd97`)

## Why the previous collection produced nothing trainable

`PARKED_SKIN_DATA_CONTRACT_FAILED` was not a shortage of data. It was a shortage of one
specific array. Of 60,793 recorded examples, **0** carried both a reconstructible four-frame
current-field history and the parked 40×8×8 field at the same decision state.

The validated per-frame oracle rendered the parked field on *every* step — it had to, to
compute its differential — and then discarded it, keeping only the resulting 7-D head output
and a hash of the array. Neither is invertible to the field. `CAUSAL_PARKED_SKIN_REFERENCE_V1`
predicts the full parked field, so its target simply did not exist on disk. The fix is
retention, not more rollouts.

## What was collected

364 outputs / 60,793 frames / 246,584,205 bytes (0.23 GiB) across four distributions:

| Distribution | Files | Frames |
|---|---:|---:|
| `EXPERT_RECONSTRUCTED` | 100 | 7,993 |
| `ACT_ONLY_ON_POLICY` | 100 | 20,000 |
| `ORACLE_ON_POLICY` | 100 | 20,000 |
| `LEARNER_INDUCED_ON_POLICY` | 64 | 12,800 |

By reference partition — `reference_train` 256 trajectories / 43,519 frames,
`offline_reference_test` 60 / 9,543, `reference_validation` 24 / 3,910,
`reference_calibration` 24 / 3,821. Hazard-present 45,768 frames, hazard-absent 15,025.

The partition itself is untouched: `configs/hybrid_obstacle_reference_partition_v2.json`
still hashes to `b061 96b6…a827d`, the value the dataset manifest recorded before collection
started. Nothing from `development4` or `confirmatory41` enters the dataset, and the
provenance check asserts that from the manifest rather than taking it on trust.

Zero failures across the whole collection. Both lanes finished on the first pass; nothing was
retried or hand-repaired.

### Expert rows are replayed, not pose-set

The trajectory H5 stores no per-step pose for the pickup object. Restoring only the robot at
each recorded state would leave the target frozen at its rest pose, and the parked field would
then be paired against a scene that never existed. Expert rows are therefore reconstructed by
deterministic open-loop replay of the recorded joint commands from the verified initial state.
ACT is never run there — the nominal-action field carries the recorded command, which is what
actually drove that trajectory. Source H5 files are opened read-only.

The 7,993 expert frames against 100 files reflect real episode lengths (the terminal step
records an empty action and is dropped); the on-policy distributions run a fixed 200-step
horizon.

## Storage contract

Three groups, `dataset_version = hybrid_obstacle_parked_skin_supervision_v1`,
manifest sha256 `a88b94b3…9c58d1b`:

- **`deployable/`** — runtime-observable inputs only: current closeness field, validity mask,
  qpos, qvel, nominal action, gripper state, episode step, control timestamp.
- **`privileged/`** — training targets: the parked field, the two head outputs, the oracle
  differential, oracle activity. A deployable loader must never read this group as input;
  `load_trajectory` refuses it unless `allow_privileged=True` is passed explicitly.
- **`integrity/`** — per-frame hashes and state-neutrality results.

Two choices in that contract are load-bearing:

**Fields are stored as closeness, with an explicit validity mask.** `clip(1 - depth/0.5, 0, 1)`,
and readings below 5 mm map to 0 closeness *and* are flagged invalid. Dead pixels and true
contact both read as extreme values; without the mask they are indistinguishable, and a model
would learn to treat sensor dropout as an obstacle.

**Only the contiguous current-field sequence is stored.** The four-frame history is
reconstructed on load by `history(t) = [t-3, t-2, t-1, t]`, left-padded by repeating the
earliest available frame, never reaching a future frame and never crossing a trajectory
boundary. Materialising the windows would hold four copies of every frame and quadruple the
dataset for no information.

Publication validates before it writes, then goes temp → `fsync` → `os.replace`. A killed
worker cannot leave a half-written trajectory that a later resume would mistake for finished.

## Integrity

Audited over **all 364 files**:

| Check | Violations |
|---|---:|
| Duplicate trajectory IDs | 0 |
| Duplicate source identities | 0 |
| State-neutrality failures | 0 |
| `0 ≤ parked ≤ current ≤ 1` (tol 1e-7) | 0 |
| Hazard-absent nonzero targets | 0 |
| Non-causal histories | 0 |
| Shape mismatches | 0 |
| Nonfinite values | 0 |
| Missing outputs | 0 |

No value was silently clamped to satisfy the physical inequality — violations were counted,
and the count is zero.

**Hazard-absent rows are an exact control, not an approximate one.** On those 15,025 frames
the current and parked fields are bitwise equal, the removable component is exactly zero, the
changed-pixel mask is empty, the two head outputs are identical, and the oracle differential is
exactly 0.0. Any leak from the parking operation into scene state would break this, so it is
the sharpest available test that the oracle perturbs only the hazard.

**Head reconstruction is exact.** Re-running the frozen SafetyHead from the *stored* closeness
fields (via `closeness_to_depth`) reproduces the recorded 7-D targets with max absolute delta
**0.0**, and the oracle differential likewise **0.0** — exact rather than within tolerance,
because the array stored is the same array the head consumed. On-policy rows record the parked
field the oracle already rendered for its own differential rather than re-rendering it: a
second render lands at a different physics substep and would silently break the very pairing
the dataset exists to establish.

Scope, stated plainly: the audit's head-reconstruction check sampled 16 trajectories; the
history smoke then verified **every frame** of one trajectory from each of the four
distributions. All other checks in the table above cover all 364 files and all 60,793 frames.

## History and model smoke

One trajectory per distribution, all frames, all causal windows:

| Distribution | T | Shape `N×4×40×8×8` | Last frame == current | No future frame | head Δ | oracle Δ |
|---|---:|---|---|---|---:|---:|
| `EXPERT_RECONSTRUCTED` | 110 | ✓ | ✓ | ✓ | 0.0 | 0.0 |
| `ACT_ONLY_ON_POLICY` | 200 | ✓ | ✓ | ✓ | 0.0 | 0.0 |
| `ORACLE_ON_POLICY` | 200 | ✓ | ✓ | ✓ | 0.0 | 0.0 |
| `LEARNER_INDUCED_ON_POLICY` | 200 | ✓ | ✓ | ✓ | 0.0 | 0.0 |

`CAUSAL_PARKED_SKIN_REFERENCE_V1` was imported and shape-checked only — 331,713 parameters,
within budget. It was never instantiated with data and never trained.

## Coverage

Median per-trajectory current-head norm 2.573. Oracle differential norm: per-trajectory max
median 0.156, overall max 16.882 — the supervision signal is small on most frames and large on
a few, which is the shape a parked-obstacle differential should have. Changed-pixel fraction
median 3.9e-5, max 9.3e-3: parking moves a small, localised part of the field, not the whole
image. All 364 trajectories report 40 active sensors.

**The natural distribution is retained.** 46,382 of 60,793 frames are oracle-zero and all are
kept; no active/zero balancing was applied at generation and no zero frames were subsampled.
Which states are quiet is itself the signal. Failed episodes are retained. Balancing, if a
future run wants it, belongs to the training sampler, not the collector.

## Schedule and concurrency

264 policy rollouts plus 100 reconstructions. Condition order is predeclared: even row rank
runs `ACT_ONLY_ON_POLICY` first, odd runs `ORACLE_ON_POLICY` first (50/50 balance); learner
rows are ordered after all 200 ACT/oracle identities. The learner checkpoint
`POSTURE_SKIN_MLP_REFERENCE_V2_ROUND0` (sha256 `f5c61334…f3796`) was loaded strictly, never
substituted.

Concurrency capped at **two** rollout processes. Five concurrent lanes under GPU contention is
what killed a shard during the on-policy round; the cap is a direct response to that.

## Constraints

No model trained. The Safety-CVAE, the residual controller, and MolmoSpaces are untouched
(MolmoSpaces working tree is clean at `678f2eb`). The ACT policy, its architecture, its weights
and its training code are unmodified; the evaluation harness in the ACT repo gained an opt-in
`--retain-parked-skin` flag — 47 lines inserted, 0 deleted, inert unless passed. The reference
partition is unchanged. `development4` and `confirmatory41` were neither used nor executed.
Only-active-frame retention and zero-frame subsampling were not applied. Fields are stored, not
hashes or head outputs in their place. Four-frame histories are reconstructed, not duplicated.
Nothing was pushed.

## Freeze

Tree sha256 `1fb68b3cdee9755881579a187c84992782bfdd5f1c65aff6a3c39ebd11b8dba5`, over the 364
dataset files by `(distribution, episode_id, file_sha256)`. Every file under the data root is
now mode 444.

The data root also holds `_rollouts/` (467 MB) — per-rollout generation byproduct
(`rollout.json`, `summary.json`, `retention.json`, `frames.npz`) kept for provenance. It is
deliberately outside the tree hash: it is regenerable and is not part of the dataset contract.
The 364 dataset files are self-contained; nothing in `_rollouts/` is needed to load them.

The dataset lives at
`assets/reference_data/hybrid_obstacle_parked_skin_supervision_v1/` and is gitignored as a bulk
artifact; the manifest, audit, smoke, provenance and decision JSONs are committed.

## Verification

98 contract tests pass (`tests/test_parked_skin_dataset_contract.py`,
`tests/test_parked_skin_reference_contract.py`). 55 provenance checks, 0 failed.

One defect was found and fixed in the audit tool itself during this task: the required-field
check was written `A | B | C - D`, which Python parses as `A | B | (C - D)`, so it reported
every deployable and privileged field as missing on all 364 files — fields the same audit had
just read successfully. The data was never affected. Re-running after the fix returns zero
problems.

## Not established by this task

The dataset is trainable, which is a different claim from useful. Nothing here shows that
`CAUSAL_PARKED_SKIN_REFERENCE_V1` can learn the parked field from these inputs, or that a
model trained on it would close the activation contract that `V2` failed on oracle-on-policy
states. The oracle differential is concentrated on a minority of frames, and whether 43,519
training frames carry enough of that signal is an open question this collection does not
answer. Training, `development4` and `confirmatory41` remain unrun by explicit instruction.

PARKED_SKIN_DATASET_READY_FOR_MODEL_TRAINING
