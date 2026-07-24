# Hybrid Obstacle Dataset Final Decision

## Executive summary

Decision: `COLLECTION_INTEGRITY_BLOCKED`.

The canonical 100-episode dataset cannot be recovered from the completed
collection, and the reason is not the selection rule. The collection stores 175
trajectories but contains only **75 distinct episodes**. Concurrent workers
replay an identical RNG stream, so 50 episodes are each stored three times, in
two clusters: `{house_25, house_49, house_73}` and
`{house_97, house_121, house_145}`. Only `house_169` is unique.

The replicas are exact. For a matched pair the `qpos`, `actions/joint_pos`, all
40 proximity streams, `env_states`, `tcp_pose`, `obj_start`, `scene_params` and
even the serialized `frozen_config` are bit-identical. They are the same episode
written three times, not three similar episodes.

That leaves **71 distinct successful trajectories** — 48 hazard-present and 23
hazard-absent — against a 75/25 target. The dataset is short by 27 hazard-present
and 2 hazard-absent, 29 in total. No selection rule can close that gap, because
the missing episodes do not exist.

Everything downstream of the data was built and shown to work: the versioned
deterministic selector produces byte-identical manifests across runs, balances
houses, refuses infeasible targets with a nonzero exit, and never consults a
quality or model metric; the committed converter is reproducible to an identical
tree hash. The blocker is the source collection alone, which is why this is
`COLLECTION_INTEGRITY_BLOCKED` and not `SELECTION_PIPELINE_FAILED`.

The worker-loss defect from the previous report has been fixed and tested
independently, and the runtime is now pinned with a startup compatibility check.
No data was recollected, no model was trained, and the source collection was not
modified.

### Correction to the previous report

The previous report concluded that the hazard draw came in "about two sigma low"
and computed a binomial test over 242 accepted theta draws. Those draws include
the replicated worker streams, so that test was run on pseudo-replicated data and
overstated its precision. On the distinct episodes the written hazard rate is
0.6667 with an exact 95% interval of **[0.5483, 0.7714]**, which contains 0.75.
The hazard draw is statistically consistent with the configured `OBSTACLE_P`;
the real defect is the duplication, not a biased sampler.

## Starting and final state

| Component | At start | Final |
|---|---|---|
| Root branch | `train/hybrid-obstacle-act-clean-v1` @ `2a47691c5d08f223e249942621e6f6a1a3bf29c5` | `repair/hybrid-obstacle-canonical-selection` |
| ACT | detached @ `3d25c69edd8d972afa59fec5c3edb9d13a357f92`, clean | unchanged, clean |
| MolmoSpaces | detached @ `c817f07b0fffc55a0dce1577312e0a7afc473b69`, clean | `repair/datagen-worker-completeness` |
| MolmoBot gitlink | `4ff337dfc60708f46d5831430df807fa90014821` | unchanged |

Root stashes `575937179ba1a5c596f56d786196331028652e25` and
`be5b715b05b21f7f98bfe29e8b6d7da4f2993b70` are intact; ACT and MolmoSpaces had no
stashes. Nothing was committed on `train/hybrid-obstacle-act-clean-v1`. Nothing
was pushed.

## Immutable source collection

| | |
|---|---|
| Path | `/root/act_retrain_assets/datagen/hybrid_obstacle_v1/FrankaSkinHybridObstacleConfig/20260724_183407` |
| Size | 1,431,940,193 bytes (1.334 GiB), 1059 files |
| Content-tree SHA-256 | `09c98aee08d015b3a561b08674415df9a4ed398186940207f41ef384251cdf24` — matches `09c98aee...1cdf24` |
| Permissions | `dr-xr-xr-x`, read-only, unmodified by this task |

## Integrity result

| Check | Result |
|---|---|
| Source tree hash matches | ✔ |
| Every H5 opens | ✔ 7/7 |
| No truncated trajectory | ✔ 175/175 streams consistent and fully readable |
| Episode IDs unique | ✔ no duplicate `house/traj` id |
| Companion metadata present | ✔ 6 videos per episode, 1050 total, none zero-byte |
| Success/failure internally consistent | ✔ `fail[-1]`, `success[-1]` and `task_info.success` agree on all 175 |
| Hazard label present for every trajectory | ✔ 175/175 |
| Hazard label matches rendered scene geometry | ✔ 175/175 |
| **No duplicate trajectory content** | **✘ 50 content classes stored 3× each** |
| No worker wrote overlapping episode IDs | ✔ each house claimed by exactly one worker |
| Episode sequences valid | ✔ each house is a contiguous `traj_0..traj_24` |
| All 40 sensor streams, expected names and shapes | ✔ `(T, 4, 8, 8)`, order hash `c31df8c3…c17cec858` on all 175 |
| RGB, qpos, action, task-state, hazard metadata present | ✔ no missing required field |
| Single canonical target UID | ✔ `4afa0cdde045417ab31f98ae7745b039` |
| Single policy clock | ✔ 66.0 ms |

Hazard labels were not taken on trust. `scene_params.protrusion_present` was
checked against the geometry actually compiled into the model: a hazard scene
carries `protr_center`/`protr_half` and exactly one extra entry in
`obstacle_aabbs` (7 boxes versus 6), and that entry matches the recorded bar box.
The sampler's documented object-placement coupling was also verified —
`_obj_rest` places the cup at `side · (bar_face_y − obj_gap)`, and the recorded
object pose agrees to a maximum of 1.5e-2 m across all hazard episodes, the
residual being physics settling. Label and geometry agree on every trajectory.

### Counts

By worker (worker 0 hung on `house_1`, which was never written):

| Worker | Houses | Trajectories | Successful | Hazard present |
|---|---|---:|---:|---:|
| 1 | `house_25`, `house_145`, `house_169` | 75 | 71 | 50 |
| 2 | `house_49`, `house_121` | 50 | 47 | 35 |
| 3 | `house_73`, `house_97` | 50 | 47 | 35 |
| 0 | `house_1` | 0 | 0 | 0 |

By house: 25 trajectories each for `house_25`, `house_49`, `house_73`,
`house_97`, `house_121`, `house_145`, `house_169`.
By success: 165 successful / 10 failed. By hazard: 120 present / 55 absent.

## Duplication

| | |
|---|---|
| Stored trajectories | 175 |
| Distinct content classes | **75** |
| Multiplicity | 25 classes ×1, 50 classes ×3 |
| Replica clusters | `{house_25, house_49, house_73}`, `{house_97, house_121, house_145}` |
| Unique house | `house_169` |
| Distinct successful | **71** (48 hazard-present, 23 hazard-absent) |

### Root cause

`FrankaSkinHybridObstacleConfig` carries a fixed seed (2026). Each worker builds
its own task sampler — `pipeline.py`: *"create task sampler once for this worker
(persists across all houses)"* — and `TaskSampler.__init__` calls
`seed_task_sampling(seed)`, which seeds `random`, `np.random` and `torch`
globally with that same value:

```python
def seed_task_sampling(self, seed) -> None:
    self.current_seed = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
```

The seed depends on neither worker id nor house index, and every configured house
index is congruent to 1 (mod 24), so all houses are the identical red-cup task.
The k-th house processed by any worker therefore replays the same episode
sequence. `house_169` is unique only because it is worker 1's *third* house and
no other worker processed a third house.

A corollary: `house_1` was worker 0's *first* house, so it would have replicated
`house_25`/`house_49`/`house_73`. **The silent worker loss cost zero distinct
episodes.** It remains a serious defect, but it is not why the dataset is short.

## Hazard-draw audit

Configured `OBSTACLE_P = 0.75`; convention `np.random.random() < self.OBSTACLE_P`;
the draw is owned by the global NumPy legacy RandomState seeded by
`TaskSampler.seed_task_sampling`; the draw site is
`ObstacleFumehoodPickSampler._draw_theta`. Rejected sampling attempts **do**
consume draws (a rejected rollout is re-sampled through `_draw_theta`). Worker
seeds overlap completely, as above. Hazard presence **does** affect rollout
success, because hazard episodes route through the deflection planner and can
fail IK — so conditional success rates are reported separately. Hazard metadata
cannot diverge from the rendered bar: the same `_draw_theta` call sets the flag
and emits the geometry, and `_apply_theta` colours that same geom. The sampler
was not altered.

Rates with exact Clopper-Pearson 95% intervals:

| Rate | As written (175, replica-inflated) | Distinct (75, independent) |
|---|---|---|
| Written hazard rate | 0.6857 [0.6113, 0.7537] (120/175) | **0.6667 [0.5483, 0.7714]** (50/75) |
| Successful hazard rate | 0.6909 [0.6144, 0.7604] (114/165) | 0.6761 [0.5545, 0.7824] (48/71) |
| Success \| hazard present | 0.9500 [0.8943, 0.9814] (114/120) | 0.9600 [0.8629, 0.9951] (48/50) |
| Success \| hazard absent | 0.9273 [0.8241, 0.9798] (51/55) | 0.9200 [0.7397, 0.9902] (23/25) |

Accepted-theta draws logged: 242, of which 168 hazard (0.6942
[0.6320, 0.7516]). With a replica multiplicity of 2.33 the effective independent
draw count is roughly 104, so even that interval is narrower than the data
justify. **Every distinct-data interval contains 0.75.** The configured hazard
probability is not contradicted by this collection.

Only the `distinct` column carries independent information. The `as_written`
intervals are reported for completeness and are not valid sampling intervals.

## Old versus new manifest

| | Old (`clean_retraining_manifest.json`) | New (`canonical_selection_v2.json`) |
|---|---|---|
| Selection order | sorted source H5 path, then numeric traj index, first 100 accepted | hash-ordered within hazard strata, round-robin over houses |
| Hazard composition | uncontrolled; whatever the draw gave | explicit 75 / 25 quotas |
| Replica awareness | none | identical-content classes collapsed before selection |
| Seed recorded | no selection seed | `hybrid-obstacle-canonical-v2`, recorded in the manifest |
| Outcome on this source | 69 hazard / 31 absent — and 96 of those 100 were replicas of 47 distinct episodes | refuses: infeasible, exit 3 |

The old manifest is untouched. The new manifest is a new versioned file.

## Deterministic selection algorithm

Fixed in advance; no downstream metric participates.

1. Drop trajectories whose `fail[-1]` is set.
2. Collapse exact-replica classes — trajectories sharing a content SHA-256 are
   the same episode stored more than once, so exactly one representative (the
   lexicographically smallest trajectory id) is eligible. This is a correctness
   requirement, not a quality filter.
3. Group the eligible representatives by hazard label.
4. Order within each group by
   `SHA-256(selection_seed ‖ trajectory_id ‖ source_file_hash)`.
5. Walk each group in a fixed round-robin over houses in sorted order, consuming
   each house's hash-ordered candidates, until the stratum quota is met.
6. Take the quota from each stratum.

Never consulted: clearance, collisions, trajectory length or quality, sensor
activation, action statistics, any model output. Selection is trajectory-level
only.

- Selector source SHA-256: `3932d31690c2e491446b2d20130f13a782b2b184d98c650cb314511163208e5f` (recorded in the manifest)
- Selection seed: `hybrid-obstacle-canonical-v2`
- Manifest SHA-256: `9d8853a260bd3108b6ea97e263444c00f5a2d466a6b72b3d8ded009d59a96317`

### Result on this collection

| Stratum | Required | Available (distinct) | Shortfall |
|---|---:|---:|---:|
| Hazard present | 75 | 48 | **27** |
| Hazard absent | 25 | 23 | **2** |
| Total | 100 | 71 | **29** |

The selector exits 3 with `feasible: false`. It does not emit a partial or
padded set.

### Positive control

To separate a data problem from a tool problem, the same selector was run at a
feasible quota (45 hazard + 15 absent) on the same source:

- selected exactly 45 + 15 = 60, all unique trajectory ids and unique content hashes;
- houses balanced 20 / 20 / 20 across `house_25`, `house_121`, `house_169`;
- run twice, byte-identical output, manifest SHA-256
  `28ef5925bb00a34b058c98ca44f328db40149f09b2baf917327f62886853aa87`.

The selector is correct. The data is not sufficient.

## Downstream converter and reproducibility

No converted canonical dataset was produced. Assembling 100 episodes from 71
distinct trajectories would require including replicas, which would fabricate
independence the source does not have. The previous converted dataset was not
overwritten.

Converter reproducibility was verified separately, on the immutable collection,
into two fresh temporary directories:

| Run | Converted tree SHA-256 |
|---|---|
| A | `5f14116e75c383d73de2d65c9e63657a28dd9344f68ceca0013a3ad239fd2760` |
| B | `5f14116e75c383d73de2d65c9e63657a28dd9344f68ceca0013a3ad239fd2760` |

Identical. Temporary outputs were removed. Conversion is deterministic.

## Worker-loss defect and fix

**Defect.** `ParallelRolloutRunner` joined its workers with
`p.join(); p.close()` and never inspected `Process.exitcode`, then built its
summary purely from shared counters. A worker that hung incremented neither
`completed_houses` nor `skipped_houses`, so its houses vanished from the summary
while the run exited 0 reporting `"Completed 7 houses, skipped 0 houses"`. Since
a house is buffered in memory and written only at completion, every trajectory
that worker held was discarded silently.

**Fix.** New module `molmo_spaces/data_generation/worker_completeness.py`,
integrated into `pipeline.py`:

- expected worker IDs are declared before launch (`WorkerRegistry`);
- every worker publishes a terminal record from a `finally` block, including on
  exception, with per-worker attempted / written / successful counts and the
  houses it wrote;
- the parent records each worker's `exitcode` as it joins;
- missing, failed and nonzero/unknown-exit workers are detected and named, and a
  missing final status is treated as silent loss even when the exit code is 0 —
  precisely the observed case;
- the run raises `WorkerCompletenessError` (nonzero exit) unless every expected
  worker finished with an approved terminal status **and** every expected house
  was written;
- the final summary is published atomically via `os.replace`, so a partial
  summary can never be mistaken for a complete one;
- partial output is retained and marked `COLLECTION_INCOMPLETE` with an explicit
  warning not to treat it as canonical.

**Tests** — `mlspaces_tests/data_generation/test_worker_completeness.py`, 17
passing: normal completion; per-worker count reporting; worker exception; worker
process death (a real spawned process calling `os._exit(3)`); missing final
status with a zero exit code; nonzero and unknown exit codes; duplicate expected
worker ID; duplicate publication; unexpected worker ID; invalid status; parent
interruption; resume after interruption; end-to-end reproduction of the observed
7-of-8-houses case; atomic publication; and no partial file left behind when
serialization fails.

The collection was **not** re-run after this fix.

## Environment pins

| | |
|---|---|
| Python | 3.11.15 |
| NumPy | 2.4.6 |
| MuJoCo | **3.5.0** |
| warp-lang | **1.11.1** |
| mujoco-warp / mujoco-mjx | 3.5.0.2 / 3.5.0 |
| torch | 2.7.1+cu126 (CUDA 12.6, cuDNN 9.5.1) |
| h5py | 3.16.0 |
| GPU / driver | NVIDIA A10, 23028 MiB, driver 570.86.10 |
| Lock file | `environment/hybrid_obstacle_collection.lock.txt`, 185 entries, SHA-256 `f5568db75aa97055a6e0c28f29c47bfa5fd04b1985c2cd97f2aec95da7d647ef` |
| Specification | `environment/hybrid_obstacle_collection.env.json`, SHA-256 `813742f16399811a45a30861f2ee5a09ce558f27185e2eae36dc985670e95423` |

Why these two are pinned: `mujoco-warp` 3.5.x reads
`mujoco.mjtEnableBit.mjENBL_MULTICCD`, which MuJoCo removed after 3.5; and
`warp-lang` after 1.11 enforces strict `wp.copy` dtype matching, which makes
`SimpleWarpKinematics.ik` raise `Incompatible array data types` at
`task.reset()` — every rollout is then rejected and the run completes with zero
episodes while still exiting successfully.

New startup check `molmo_spaces/data_generation/runtime_compat.py`
(`assert_supported_runtime`, invoked from `data_generation/main.py`) reports
unsupported versions by name with the failure each one causes, and exits nonzero.
Verified in both directions: clean on the pinned stack, and correctly raising
when 3.10.0 / 1.15.0 are simulated. Planner mathematics was **not** adjusted to
accommodate another Warp release.

## Changed files and commits

Root repository, branch `repair/hybrid-obstacle-canonical-selection`:

- `scripts/hybrid_obstacle_dataset_audit.py` — integrity auditor
- `scripts/hybrid_obstacle_hazard_audit.py` — hazard-draw auditor
- `scripts/hybrid_obstacle_select_canonical.py` — deterministic v2 selector
- `diagnostics_output/hybrid_obstacle_dataset/integrity_report.json`
- `diagnostics_output/hybrid_obstacle_dataset/hazard_audit.json`
- `diagnostics_output/hybrid_obstacle_dataset/canonical_selection_v2.json`
- `diagnostics_output/hybrid_obstacle_dataset/final_decision.json`
- `environment/hybrid_obstacle_collection.lock.txt`
- `environment/hybrid_obstacle_collection.env.json`
- `docs/HYBRID_OBSTACLE_DATASET_FINAL_DECISION.md`
- updated MolmoSpaces gitlink

MolmoSpaces, branch `repair/datagen-worker-completeness`:

- `molmo_spaces/data_generation/worker_completeness.py` (new)
- `molmo_spaces/data_generation/runtime_compat.py` (new)
- `mlspaces_tests/data_generation/test_worker_completeness.py` (new)
- `molmo_spaces/data_generation/pipeline.py` (completion contract wired in)
- `molmo_spaces/data_generation/main.py` (startup compatibility check)

Not committed: source trajectories, converted datasets, checkpoints, videos,
temporary output. `EVAL.md` was not touched.

## Reproduction commands

```bash
source /root/act_retrain_env.sh
cd /root/prox_learning_act_retrain

# integrity (read-only)
python scripts/hybrid_obstacle_dataset_audit.py integrity \
    /root/act_retrain_assets/datagen/hybrid_obstacle_v1/FrankaSkinHybridObstacleConfig/20260724_183407 \
    --pixel_sample 0 --worker_log /root/act_retrain_provenance/logs/collection.stderr \
    > integrity_report.json

# hazard draw audit
python scripts/hybrid_obstacle_hazard_audit.py \
    --integrity_report integrity_report.json \
    --worker_log /root/act_retrain_provenance/logs/collection.stderr --config_seed 2026

# deterministic selection (exits 3: infeasible on this collection)
python scripts/hybrid_obstacle_select_canonical.py \
    --integrity_report integrity_report.json \
    --output diagnostics_output/hybrid_obstacle_dataset/canonical_selection_v2.json

# worker completeness contract
cd submodules/molmospaces
python -m pytest mlspaces_tests/data_generation/test_worker_completeness.py -q
python molmo_spaces/data_generation/runtime_compat.py
```

## Next recommended task

Recollect under a **seeding** fix, not a selection fix. Derive each worker's
task-sampling seed from `(config seed, worker id, house index)` so concurrent
workers explore independent streams, while keeping the fixed base seed for
reproducibility. That changes sampler seeding, which this task explicitly
forbade, so it needs its own approval. Then re-run the 8 × 25 collection on the
pinned runtime with the new fail-loud completion contract active, assert that the
integrity auditor reports 200 distinct content classes, and apply the v2 selector
unchanged to draw the 75 / 25 canonical set. The v2 selector, the integrity
auditor, the hazard auditor and the converter require no further work.

Do not launch training or new data collection on the basis of this report.

## Independent re-verification (second agent, from scratch)

The decision above was re-derived by a second agent that did not reuse the audit
code behind it. `scripts/hybrid_obstacle_reverify.py` is a separate
implementation — its own tree hash, its own H5 traversal, its own identity
hashes, its own Clopper-Pearson routine — so agreement between the two is
evidence rather than a shared bug. It exits 3 on this collection.

- Verifier source SHA-256: `41390ae5d9b5240693b7b51a1c515f92a536f9c180dc209270af81757436436b`
- Report: `diagnostics_output/hybrid_obstacle_dataset/independent_reverification.json`,
  SHA-256 `b4e08baabe79fa808e9f010bfde5fd9177a5c57bf196facf4e20dbd9cef3057f`
- Re-running the verifier reproduces the report byte-for-byte.

| Claim under test | Independent result | Agrees |
|---|---|:--:|
| Full expected tree hash recovered from the committed clean-retrain manifest | `09c98aee08d015b3a561b08674415df9a4ed398186940207f41ef384251cdf24` | ✔ |
| Recomputed tree hash, 1059 files, 1,431,940,193 bytes | identical | ✔ |
| All 7 committed per-file H5 SHA-256 values | reproduce exactly | ✔ |
| Stored trajectories | 175 | ✔ |
| Distinct episodes | **75** | ✔ |
| Replica grouping (50 groups of 3) | **set-identical** to the committed grouping, member for member | ✔ |
| Distinct successful | **71** (48 hazard-present, 23 hazard-absent) | ✔ |
| Written / successful / failed | 175 / 165 / 10 | ✔ |
| Hazard split, written | 120 present, 55 absent | ✔ |
| Hazard split, successful | 114 present, 51 absent | ✔ |
| Hazard split, failed | 6 present, 4 absent | ✔ |
| 40 proximity streams, shape `(T, 4, 8, 8)`, order identical on all 175 | ✔ | ✔ |
| Hazard label vs compiled scene geometry | 0 disagreements / 175 | ✔ |
| Companion media (1050 files) | none missing, none zero-byte | ✔ |
| Every exact binomial interval in the hazard audit | reproduces to printed precision | ✔ |
| Source collection unchanged after the audit | all 1059 per-file hashes identical pre/post, tree still `09c98aee…1cdf24`, still `dr-xr-xr-x` | ✔ |

Source-level confirmation of the root cause, read directly rather than taken
from the earlier report: `OBSTACLE_P = 0.75` at
`molmo_spaces/tasks/enclosure_reach.py:1121`; the draw is
`if np.random.random() < self.OBSTACLE_P:` at line 1134;
`TaskSampler.seed_task_sampling` (`task_sampler.py:420`) seeds `random`,
`np.random` and `torch` from one value that depends on neither worker id nor
house index; `pipeline.py:414` creates one sampler per worker that "persists
across all houses"; and `get_episode_seed` returns `task_sampler.current_seed`
unchanged for every episode. Nothing in the path differentiates the workers.

### The shortfall does not depend on how "distinct" is defined

This strengthens the earlier finding. Three independent identity notions were
computed over the same 175 trajectories:

| Identity | Distinct written | Distinct successful | Hazard-present | Hazard-absent |
|---|---:|---:|---:|---:|
| `qpos` + `actions/joint_pos` | 75 | 71 | 48 | 23 |
| `scene_params` | 75 | 71 | 48 | 23 |
| **every leaf dataset** (most permissive) | 78 | 74 | **50** | **24** |

The permissive notion counts three replica classes as distinct because they
differ in `actions/ee_twist`, `actions/joint_pos_rel`, `policy_phase` and
projected `object_image_points` — all derived quantities — while their `qpos`
and `actions/joint_pos` are bit-identical. Even granting those, the collection
supplies at most **50 hazard-present** and **24 hazard-absent** successful
trajectories against the 75 / 25 target: short by 25 and 1. The 75 / 25 subset
is infeasible under *every* defensible definition of a distinct trajectory, not
only the strict one.

### Toolchain re-exercised

| Check | Result |
|---|---|
| v2 selector at the 75 / 25 target | exit 3, `feasible: false`, manifest SHA-256 `9d8853a2…a96317` — identical to the committed manifest |
| v2 selector at a feasible 45 / 15 quota, run twice | byte-identical, SHA-256 `28ef5925…6853aa87` — matches the documented positive control |
| Positive-control composition | 60 selected, 60 unique trajectory ids, 60 unique content hashes, houses balanced 20 / 20 / 20 |
| Selector independence | ordering key is `SHA-256(seed ‖ trajectory_id ‖ source_h5_sha256)` only; `frames` is recorded in the manifest but never read by the ordering, and no clearance / collision / sensor / quality / model field is touched |
| Recorded self-hashes | `manifest_sha256`, `selector_source_sha256`, `lock_file_sha256` and `environment_specification_sha256` all recompute exactly |
| `mlspaces_tests/data_generation/test_worker_completeness.py` | 17 passed |
| `runtime_compat.py` on the pinned stack | `runtime compatibility: OK`, exit 0 |

### Correction to the reproduction commands

The integrity-auditor command in the *Reproduction commands* section above
writes a report whose per-trajectory rows are stripped before committing (see
`trajectories_detail_note`). Feeding the committed
`diagnostics_output/hybrid_obstacle_dataset/integrity_report.json` to the
selector therefore fails with `KeyError: 'trajectories_detail'`. The selector
needs the full report, which is retained outside the repository as bulk
provenance:

```bash
python scripts/hybrid_obstacle_select_canonical.py \
    --integrity_report /root/act_retrain_provenance/dataset_repair/integrity_report.json \
    --output diagnostics_output/hybrid_obstacle_dataset/canonical_selection_v2.json
```

The selector was deliberately **not** edited to work around this, because its
source SHA-256 is bound into the committed manifest.

### Independent re-verification command

```bash
source /root/act_retrain_env.sh
cd /root/prox_learning_act_retrain
python scripts/hybrid_obstacle_reverify.py \
    --run_dir /root/act_retrain_assets/datagen/hybrid_obstacle_v1/FrankaSkinHybridObstacleConfig/20260724_183407 \
    --expected_tree_sha256 09c98aee08d015b3a561b08674415df9a4ed398186940207f41ef384251cdf24 \
    --output diagnostics_output/hybrid_obstacle_dataset/independent_reverification.json
# exits 3: source verified, canonical 75/25 target infeasible
```

The decision is unchanged and is now supported by two independent audits.

COLLECTION_INTEGRITY_BLOCKED
