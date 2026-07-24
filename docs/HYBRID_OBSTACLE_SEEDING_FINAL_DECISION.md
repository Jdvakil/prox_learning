# Hybrid Obstacle Seeding — Final Decision

Immutable episode manifest and episode-scoped RNG contract for the 40-sensor
hybrid-obstacle collection, with worker-count and resume invariance proved on a
bounded smoke test.

Date: 2026-07-24 · Task scope: contract + proof only. The full 160-row
collection was **not** launched. ACT was not trained or modified. The
Safety-CVAE was not trained, modified, or re-checkpointed.

---

## 1. Executive summary

The previous hybrid-obstacle collection was scientifically void: 175 written
trajectories represented at most 75 defensibly distinct episodes, containing 50
replica classes of three members each. The cause was structural — the "episode
seed" was a per-worker constant, houses were claimed dynamically, and hazard
presence was a runtime Bernoulli off a shared global RNG — so an episode's
content was a function of worker state and execution order rather than of any
episode identity.

This task replaces episode identity and seeding with a committed contract:

- A frozen **160-row candidate manifest** (`hybrid_obstacle_independent_v2`,
  master seed **20260725**, 120 hazard-present / 40 hazard-absent), with the
  hazard schedule, canonical-selection ranks and train/val split all predeclared
  before any simulation.
- **Ten named RNG streams** derived as
  `SeedSequence([master_seed, candidate_index, stream_id, retry_index])` —
  order-independent, and structurally incapable of consuming a worker ID, a
  worker count, or a house alias.
- A **dedicated manifest runner** that queues rows rather than houses, claims
  each exactly once atomically, installs the row's seed contract before any
  task-level draw, forces the row's hazard assignment, and finalises every row
  with an explicit terminal outcome.
- Episode identity is `episode_id + manifest_row_sha256`, written into H5
  metadata, never derived from file layout.

**Result: all three smoke runs are bit-identical.** Runs A (1 worker), B (4
workers) and C (4 workers, hard-killed after four rows finalized, then resumed)
produced the same eight episode IDs with byte-identical scientific arrays,
identical retry counts, identical planner phase paths and identical outcomes,
while worker assignments differed across runs. All 68 static tests pass,
including the 17 pre-existing worker-completeness tests.

**Final decision: `SEEDING_CONTRACT_READY_FOR_FULL_COLLECTION`** (token repeated
verbatim as the last line of this document).

One honest caveat is recorded in §13: the first A/B comparison **failed**, and
the failure was real. It exposed a residual cross-row dependency that the static
tests had not covered. It was diagnosed to a single line, fixed, and Runs A/B/C
were re-executed from scratch on fresh output directories. The passing result
above is from the second, post-fix freeze.

---

## 2. The invalid collection and its root cause

### What was wrong

| Property | Value |
|---|---|
| Trajectories written | 175 |
| Defensibly distinct episodes | at most 75 |
| Replica classes | 50, of three members each |
| Max distinct successful hazard-present | 50 |
| Max distinct successful hazard-absent | 24 |
| Recoverable 75/25 scientific dataset | **no** |

A scientifically independent 75/25 dataset cannot be recovered from it. The
collection remains read-only and must never be used as the canonical ACT
training dataset. Nothing in this task reads, modifies, converts or selects from
it.

### Verified mechanism

Traced end to end and recorded in
`diagnostics_output/hybrid_obstacle_seeding/current_execution_path.json`:

1. `BaseTaskSampler.__init__` (`task_sampler.py:313-315`) seeds from a single
   repeated `config.seed`; `seed_task_sampling` (`:420-424`) sets Python, NumPy
   and Torch globals **once per worker**.
2. Exactly one task sampler is constructed per worker
   (`pipeline.py:416-418`) and persists across every house that worker claims.
3. `get_episode_seed` (`pipeline.py:735-737`) returns that repeated sampler
   seed. The "per-episode seed" in the H5 was a worker constant.
4. Houses are claimed dynamically from a shared counter
   (`pipeline.py:431-437`), so which worker gets which house is a scheduling
   race.
5. The eight wraparound indices `[1, 25, 49, 73, 97, 121, 145, 169]` are all
   `1 (mod 24)` and therefore aliases of the *same* fumehood/red-cup task; they
   existed purely to parallelise a house-oriented runner.
6. Hazard presence was `np.random.random() < OBSTACLE_P`
   (`enclosure_reach.py:1134`) off that same order-dependent global stream.

Two draws in the failed collection were worse than order-dependent: the config
scalars `robot_object_z_offset_random_min/max` are `np.random.uniform(...)`
evaluated at **module import time**
(`object_manipulation_datagen_configs.py:2230-2231`), before any seeding, and
therefore differ between the parent and each spawned worker. The manifest config
pins both.

---

## 3. Branches and commits

Full record: `diagnostics_output/hybrid_obstacle_seeding/branch_preservation.json`.

### Handoff expectation vs. observed state

The session did **not** open in the state the handoff described. Both
differences were verified and resolved before any edit:

| | Handoff expected | Observed at session start | Resolution |
|---|---|---|---|
| Root | `repair/hybrid-obstacle-canonical-selection` @ `beb125e` | `recovery/hybrid-act-canonical-artifacts` @ `a4d6b6d` | `beb125e` exists exactly as stated but is checked out in a **separate worktree** (`/root/prox_learning_act_retrain`), so it could not be checked out here. The new branch was created and hard-reset to `beb125e`. The sibling worktree was left untouched. |
| MolmoSpaces | `repair/datagen-worker-completeness` @ `fa8e61f` | detached @ `c817f07` (= `main`) | `fa8e61f` was **absent from the local object store** — the clone's reflog showed only clone + checkout of `c817f07`. `git ls-remote` confirmed `refs/heads/repair/datagen-worker-completeness → fa8e61f` on origin; `git fetch --all` recovered it. |
| ACT | unchanged | `repair/live-safety-residual-eval` @ `3d25c69` | unchanged — **not modified by this task**. |

Root `beb125e` records gitlink `submodules/molmospaces = fa8e61f`, so the handoff
pair is internally consistent; the session's starting state was not. Intervening
commits between the shared ancestor `6a15ac3` and `beb125e` are `2a47691`
(`CANONICAL_DATA_COLLECTION_FAILED`), `108a01f`
(`COLLECTION_INTEGRITY_BLOCKED`) and `beb125e` (independent re-verification).
Neither branch contained any manifest or seeding work.

### Branches used

| Repo | Branch created | Base |
|---|---|---|
| Root | `repair/hybrid-obstacle-episode-manifest-v2` | `beb125e97607cc34c5f64645a9872ccc78e33911` |
| MolmoSpaces | `repair/hybrid-obstacle-manifest-runner-v2` | `fa8e61f40eb97e27bf3b69480c8bc65b0450f362` |

No commits were made on `repair/hybrid-obstacle-canonical-selection`,
`repair/datagen-worker-completeness`, `train/hybrid-obstacle-act-clean-v1` or
`recovery/hybrid-act-canonical-artifacts`. Nothing was pushed. The ACT gitlink
stays `3d25c69`.

Two pre-existing stashes on the root repo (`stash@{0}` PACT data-quality,
`stash@{1}` PACT frontend recovery) were left untouched.

---

## 4. Current execution-path audit

`diagnostics_output/hybrid_obstacle_seeding/current_execution_path.json` traces
16 stages from parent launch through H5 publication and names 11 independence
violations. The load-bearing findings:

| Stage | Finding |
|---|---|
| Parent work assignment | shared counter + lock; worker↔house is a scheduling race |
| Sampler lifetime | one instance per worker, spanning every house it claims |
| `seed_task_sampling` | once per worker; `reset()` is never called by the datagen loop |
| Task sampling entry | **no seeding of any kind between episodes** — this is the v2 insertion point |
| `get_episode_seed` | read *after* the scene is already sampled; a label only, seeds nothing |
| Hazard | Bernoulli drawn mid-`_draw_theta`, after the parent's theta draws |
| Episode loop | a success-quota loop; failed attempts are never recorded as identified candidates |
| Sampler mutable state | asset blacklist, synset counter and robot positions accumulate across houses |
| Output identity | H5 group is `traj_{buffer_index}` — identity is file layout |
| Resume | `should_skip = batch_file.exists()`; an interrupted house loses its entire in-memory buffer |
| Worker ledger (fa8e61f) | registry declared pre-launch, exit codes recorded, every worker publishes a terminal record, summary written atomically |

---

## 5. RNG inventory

`diagnostics_output/hybrid_obstacle_seeding/rng_inventory.json` — 24 sources, 21
on the hybrid-obstacle path, 17 duplicated in the old collection, 4 requiring an
explicit generator, 1 eliminated outright.

No repository-wide conversion from global RNGs to `Generator` objects was
attempted. The minimal safe implementation is used:

- install the row's seeds before any task-level draw;
- seed Python, the legacy NumPy global RNG and Torch (plus CUDA when available)
  consistently;
- **explicitly** seed the components that already own independent generators;
- leave legacy behavior untouched outside the manifest config.

Sources needing explicit seeding (they own `RandomState` objects built once per
scene load, which would otherwise advance across rows): the lighting, texture and
dynamics randomizers, plus the debug-retention draw.

The single largest consumer is per-step action noise
(`robots/abstract.py:144-160`), which uses `scipy.stats.truncnorm.rvs` — whose
default `random_state` **is** the NumPy global — and `np.random.randn`. It is
enabled for this config. That is precisely why the global install must precede
task sampling and why no cross-row state may leak.

A repository-wide grep found no stochastic MuJoCo or Warp draw on this path.
Physics is deterministic given the model, the initial state and the action
sequence, for a fixed `mujoco==3.5.0` / `warp-lang==1.11.1` build.

---

## 6. Manifest schema and frozen values

Contract: `configs/hybrid_obstacle_independent_v2.yaml`
Manifest: `configs/hybrid_obstacle_candidate_manifest_v2.json`
Smoke subset: `configs/hybrid_obstacle_manifest_v2_smoke8.json`

| Item | Value |
|---|---|
| Manifest version | `hybrid_obstacle_independent_v2` |
| **Master seed** | **20260725** |
| **Full 160-row manifest SHA-256** | `8be804057f3e0710a7c4770de54e8516e6156856423b6fdcac191ac0b15f2805` |
| **smoke8 subset SHA-256** | `cb9df6e1f8dabb2e8705f685a8e1a0f9673ca184d1d455995e40e06f6bb4ddbc` |
| Candidates | 160, indices 0–159 |
| Hazard schedule | 120 present / 40 absent |
| Scene-template identity | `fumehood_red_cup_v1`, house index 1 (no aliases) |
| Sensor-order SHA-256 | `c31df8c36b0011b0eaf5b2eb5ce66d2514b5d6662ba9d7684ff021cd17cec858` |
| Robot model SHA-256 | `50924661e0411f92ab529c790512b17b674e789434c592c3dbc6d2359164d4c6` |
| Fumehood scene SHA-256 | `eec36c2ae5d17fc82d1221feaa87a1de2e9bec391902dfb1b46ef27ed7ad716d` |
| Env/config SHA-256 | `190547e0fa424b17594375e5347a0d21f809b945f14bc648a936e6eb38eccd16` |
| Runtime-contract SHA-256 | `64bc65689ae3d21b8f2e8e1ddf6672732f622dbbaea14851f1664b6eace7dd70` |
| MolmoSpaces source commit | `fa8e61f40eb97e27bf3b69480c8bc65b0450f362` |
| Max retries per row | 4 |

Each row carries: manifest version, master seed, candidate index, immutable
episode ID, scene-template ID, hazard label, stratum rank, canonical-selection
rank, split rank, root `SeedSequence` entropy per stream, the named stream seeds,
the 40-sensor-order hash, robot-model hash, env/config hash, Safety-CVAE
contract references, MolmoSpaces source commit, runtime-contract hash, and its
own SHA-256.

Episode ID derivation (SHA-256 throughout; Python's `hash()` is never used for
any persistent ID or seed — enforced by an AST-level test):

```
episode_id = SHA256(manifest_version ‖ master_seed ‖ candidate_index ‖ scene_template_id)
```

with `\x1f` field separators so no two distinct field tuples share a preimage.

The manifest pins the **base** MolmoSpaces commit rather than the runner commit:
a manifest whose hash moved every time the runner was amended would not be a
freeze. The exact frozen source is proven separately by §12.

### The 120/40 hazard schedule

Exactly 120 `True` and 40 `False` values are created and deterministically
permuted from the master seed via the reserved pseudo-stream `1000`, then
committed before simulation. It is a genuine permutation, not a sorted block —
the first 40 candidates are:

```
H H H H H . H H . H H H H H H H H H . H H H H H . . H H H H H H H . H H H H H H
```

`OBSTACLE_P = 0.75` is retained as the documented design probability and
120/160 = 0.75 exactly, so design intent is preserved without the draw. **For
the manifest config the Bernoulli is never drawn.** Every legacy config keeps it
byte for byte, proven by dedicated isolation tests (§11).

---

## 7. Stream derivation

Stable integer IDs, never renumbered:

| ID | Stream | ID | Stream |
|---|---|---|---|
| 0 | global compatibility | 5 | action noise |
| 1 | task/scene sampling | 6 | camera/lighting randomization |
| 2 | robot and object placement | 7 | Python-random compatibility |
| 3 | obstacle theta and geometry jitter | 8 | Torch compatibility |
| 4 | planner/grasp stochastic choices | 9 | retry/fallback |

```python
entropy = [master_seed, candidate_index, stream_id, retry_index]
state    = np.random.SeedSequence(entropy).generate_state(2, dtype=np.uint32)
seed_u32 = int(state[0])
seed_u64 = int(state[0]) | (int(state[1]) << 32)
```

Every stream is derived independently from the **full four-tuple**. There is
deliberately no chain of `spawn()` calls, because spawn-order semantics change
the moment a stream is inserted, and no reliance on the order in which streams
are requested. A test asserts that requesting streams forwards and backwards
yields identical seeds, and that introducing a hypothetical stream ID 99 leaves
stream 5 untouched.

Installation order per row, before any task-level draw:
`random.seed` → `np.random.seed` → `torch.manual_seed` (+
`torch.cuda.manual_seed_all` when CUDA is present) → explicit reseed of the
owned lighting/texture/dynamics `RandomState` objects.

### Why worker ID and house aliases are excluded

Not by convention but **structurally**: `derive_stream_seed` accepts exactly
`(master_seed, candidate_index, stream_id, retry_index)`. There is no argument
through which a worker ID, a worker count, or a wraparound house alias could
reach a seed. A test asserts the function signature itself.

The aliases carry no scientific meaning in any case — all eight are `1 (mod 24)`
and index the same red cup. They were a scheduling device for a house-oriented
runner; the manifest runner parallelises rows, so the device is deleted rather
than reinterpreted. Worker ID and house alias are still recorded in the H5 and
outcome ledger, explicitly labelled `*_descriptive`.

---

## 8. Row and retry semantics

A manifest row is one candidate scientific episode, **not** one successful
trajectory slot.

- Hazard assignment, scene-template identity and candidate identity are fixed.
- Task sampling has a bounded retry sequence; `retry_index` starts at 0 and each
  retry derives its own independent streams, so no retry inherits a partially
  consumed state.
- Every rejected retry and its reason is logged into the row's outcome record.
- One row produces at most one accepted trajectory.
- A row that exhausts its retries is recorded as `sampling_failure`. It is never
  silently replaced by a new unidentified episode, and no row is ever reused to
  satisfy a success quota.
- Terminal outcomes: `success`, `task_failure`, `sampling_failure`,
  `infrastructure_failure`.

`filter_for_successful_trajectories` is `False` for this config. Full payloads
(H5 + videos) are retained for accepted trajectories only, for storage reasons,
but **every** manifest row keeps a complete atomic outcome record, so no
candidate identity is ever erased.

Before every row the worker clears episode-scoped sampler state — the object
synset counter, used robot positions, asset failure counts, the dynamic
blacklist, per-house sample accounting, the previous row's theta, and the cached
lighting bases — and forces a scene reload. Only the dataset index map and the
config are preserved as documented immutable caches.

---

## 9. Identity, publication and resume

Resume and deduplication key on `episode_id + manifest_row_sha256`. Identity is
never a worker-local episode counter, a batch index, a house directory, a
wraparound alias, or file ordering.

Layout under `<output_dir>/rows/<episode_id>/`:

| File | Written by | Semantics |
|---|---|---|
| `claim.json` | `O_CREAT\|O_EXCL` | the atomic claim |
| `outcome.json` | `mkstemp` + `fsync` + `os.replace` | the terminal finalisation |
| `trajectory.h5` | staged then `os.replace` | the published payload |

A row is complete iff `outcome.json` carries a terminal status. A row is
reclaimable iff it holds a claim from a *different* run with no outcome — a
properly abandoned row. A second claim within the same run raises
`DuplicateClaimError`; a second publication raises `DuplicatePublicationError`.

Every published H5 gains a `manifest` group plus root and per-trajectory
attributes carrying: manifest version and hash, candidate index, episode ID, row
hash, hazard label, stratum/canonical/split ranks, retry count and history, the
complete seed map, actual obstacle theta, selected object and grasp identity,
outcome, 40-sensor-order hash, runtime-contract hash, source commits, and
descriptive worker ID / house alias. Fields are **added**; the existing RGB,
qpos, action, task-state and proximity datasets are untouched, so the committed
offline converter stays compatible.

---

## 10. Predeclared canonical selection and split

Locked now, before collection. Nothing here inspects rollout quality.

**Canonical dataset** (after a future full 160-row collection): the first **75**
successful hazard-present rows and the first **25** successful hazard-absent
rows, by predeclared stratum rank.

**Trajectory-level split**, fixed:

| Split | Hazard-present | Hazard-absent |
|---|---|---|
| Train | 60 | 20 |
| Validation | 15 | 5 |

Selection and split never inspect clearance, collision severity, trajectory
length, action smoothness, proximity activation, image quality, planner retries,
policy phase duration, or any model or audit score. The split is a pure function
of stratum rank, asserted by test.

If a future 160-row collection does not yield enough successful rows: do not
duplicate rows, do not lower the quota, do not alter this manifest. A separately
versioned deterministic extension manifest requires its own explicitly approved
task.

---

## 11. Static test results

**68 passed, 0 failed.**

```
mlspaces_tests/data_generation/test_episode_manifest.py          42 passed
mlspaces_tests/data_generation/test_manifest_hazard_isolation.py  9 passed
mlspaces_tests/data_generation/test_worker_completeness.py       17 passed   (pre-existing, fa8e61f)
```

| Group | Proven |
|---|---|
| Manifest | 160 unique candidate indices / episode IDs / row hashes; exactly 120 + 40; deterministic regeneration reproduces the same manifest hash; master seed 20260725 recorded; canonical and split ranks fixed; tampering with a row or the manifest hash is rejected |
| Seed contract | same row reconstructs identical stream seeds; all 160×10 stream seeds distinct; streams within a row mutually distinct; derivation independent of request order; inserting a new stream ID perturbs nothing; worker ID / worker count / house alias change no seed (asserted on the signature); retry index re-derives every stream without disturbing retry 0; seeds stable across a `spawn`ed process (so `PYTHONHASHSEED` cannot reach them); no builtin `hash()` anywhere in the contract modules (AST check); installed contract reproduces identical draws and differs between rows; randomizer reseed lands on the same state as a fresh scene build |
| Execution | manifest hazard overrides the runtime Bernoulli **without consuming a draw**; legacy obstacle config still draws Bernoulli at rate 0.75; the preflight check sampler still forces the bar present; a pinned row on one instance does not leak to another or to the class; duplicate row claim fails; duplicate publication fails; failed rows remain in the ledger; a row cannot be finalised twice; non-terminal status rejected; missing manifest fails; wrong sensor/config/runtime hash each fail; reconcile reports unfinalised, unexpected and published-without-outcome rows; resume skips completed rows and reclaims only properly abandoned ones |
| Cross-row isolation | episode-scoped reset clears counters and blacklists, forces a scene reload, and drops the previous row's theta (the §13 regression) |
| Worker lifecycle | all 17 fa8e61f tests remain green; expected workers reconciled; silent worker loss fail-loud; atomic final summary intact |

The committed manifest is additionally verified by
`scripts/build_hybrid_obstacle_manifest_v2.py --check`, which regenerates it and
compares hashes — run as part of the test suite.

---

## 12. Source freeze

`diagnostics_output/hybrid_obstacle_seeding/frozen_source.json`

| Generation | Trees | Status |
|---|---|---|
| 1 | — | **INVALIDATED** by the §13 fix; its Runs A/B were discarded and its output directory renamed `smoke_runs_invalidated_v1` |
| 2 | root `b7a18d1ee69c68a20026b14713be13f82fef1a2d`, molmospaces `ce1c82f6532c25bb6f99865a6e4d6a14791b30a4` | **authoritative for Runs A, B and C** |
| 3 | — | analysis-only delta (§15) |

The 12 collection-source files hashed at generation 2 are byte-identical to
those recorded in `runtime_provenance.json` after Run C — verified
programmatically. Collection source digest:
`d5b41570164244f1b8a52346847cd1b61d7247a3b36b186fdf03fed8f04e94cd`.

---

## 13. The first A/B comparison failed — and why that matters

The pre-freeze diagnostic budget (four candidate-row executions: a 2-row
1-worker run and a 2-row 4-worker run) proved the runner reaches task sampling
and writes the required metadata. It also surfaced a divergence, which led to a
first fix: `sample_task` branches on whether the requested house is already
loaded, and the two branches are not equivalent — the load branch resets
`task_config`, resets the metadata adder, and runs the whole `update_scene` path,
which itself consumes global RNG draws. Forcing a scene reload per row was
applied and the source frozen.

**That fix was incomplete, and freeze-1's Run A vs Run B failed invariance.**
Four of eight rows diverged. The pattern was diagnostic: rows that were a
worker's *first* row in one run but a *later* row in the other diverged; rows
that held the same position in both did not.

Two non-simulating probes (no rollouts, no ledger, no published output) located
it to a single line. `EnclosureSampler._obj_rest`
(`enclosure_reach.py:243-248`) runs during scene setup, **before** the current
row's `_draw_theta`, and branches on leftover state:

```python
th = getattr(self, "_theta", None)
if not th:
    return (TUBE_X0 + 0.25, 0.0, SHELF_TOP_Z)                     # zero draws
x = TUBE_X0 + max(0.12, th["target_frac"] * th["depth"] - 0.04)
y = float(np.random.uniform(-1, 1) * (th["ap_w"] / 2 - 0.05))     # one draw
```

A worker's first row found no `_theta` and consumed **zero** draws; every later
row found the *previous row's* `_theta` and consumed **one**, using the previous
row's aperture width. The global RNG position at `_draw_theta` entry was 17 on a
first row and 19 on later ones. Which rows land first is exactly what the worker
count decides, which is how this survived as a pure worker-count invariance
failure.

The fix adds `_theta` to the episode-scoped reset. After it, the probe reported
position 17 and identical theta on every attempt, and a second probe confirmed
order-independence: sampling candidates `[1,5,2,0]` and then `[2,0,1,5]` in one
process produced identical digests for every candidate at every position.

The source was re-frozen (generation 2) and **Runs A, B and C were re-executed
from scratch on fresh output directories.** All results in §14–§15 are
post-fix.

This is recorded rather than smoothed over because it is the substantive
finding: static tests and a two-row diagnostic were *not* sufficient to
establish worker-count invariance. The A/B comparison was.

---

## 14. Runs A, B and C

Eight predeclared rows (`configs/hybrid_obstacle_manifest_v2_smoke8.json`,
hash `cb9df6e1…`): the four lowest-ranked hazard-present rows (candidates 0, 1,
2, 3) and the four lowest-ranked hazard-absent rows (candidates 5, 8, 18, 24).
The subset hash was recorded before simulation and no row was replaced after its
outcome was observed. Total: **24 row executions**, no environment tuning, no
seed changes, no replacement rows.

| Run | Workers | Rows | Interruption | Finalised | Succeeded | Reconciled | Summary complete |
|---|---|---|---|---|---|---|---|
| A | 1 | 8 | — | 8 | 8 | ✅ exactly once | ✅ |
| B | 4 | 8 | — | 8 | 8 | ✅ exactly once | ✅ |
| C | 4 | 8 | SIGKILL to the process group after 4 rows finalized, then resumed | 8 | 8 | ✅ exactly once | ✅ |

### Per-row outcomes (identical across all three runs)

| Candidate | Hazard | Stratum rank | Split | Outcome | Retries | Worker A / B / C |
|---|---|---|---|---|---|---|
| 0 | present | 0 | train | success | 0 | 0 / 2 / 1 |
| 1 | present | 1 | train | success | 0 | 0 / 0 / 0 |
| 2 | present | 2 | train | success | 0 | 0 / 3 / 3 |
| 3 | present | 3 | train | success | **1** | 0 / 1 / 2 |
| 5 | absent | 0 | train | success | 0 | 0 / 3 / 3 |
| 8 | absent | 1 | train | success | 0 | 0 / 2 / 0 |
| 18 | absent | 2 | train | success | 0 | 0 / 0 / 1 |
| 24 | absent | 3 | train | success | **1** | 0 / 1 / 2 |

Candidates 3 and 24 each required exactly one task-sampling retry — **in all
three runs**. Retry behavior is itself deterministic per row, which is the
stronger claim: the retry stream is derived from the row, not inherited from a
partially consumed state.

### Run C interruption record

`diagnostics_output/hybrid_obstacle_seeding/smoke_runs/run_c_interrupt.json`

The interrupt was a `SIGKILL` to the entire process group — job death, not a
graceful shutdown — so rows in flight left claims with no terminal outcome.

| | |
|---|---|
| Rows finalised before the kill | 4 |
| Claim files present at the kill | 8 |
| **Dangling claims left behind** | **4** |
| Resume: already finalised on entry | 4 (skipped, not re-executed) |
| Resume: abandoned claims reclaimed | 4 |
| Resume: rows executed | 4 |
| Final finalised | 8 — neither skipped nor duplicated |

---

## 15. Invariance comparison

`diagnostics_output/hybrid_obstacle_seeding/invariance_report.json`. Comparison
is **by episode ID**, never by file order.

| Pair | Same episode-ID set | Invariant | **Bit-identical** | Episodes |
|---|---|---|---|---|
| A vs B | ✅ | ✅ | ✅ | 8 |
| A vs C | ✅ | ✅ | ✅ | 8 |
| B vs C | ✅ | ✅ | ✅ | 8 |

Everything required to match did match **exactly**, with zero reliance on the
tolerance path: episode ID, manifest row hash, hazard assignment, seed map,
scene-template ID, target/object identity, obstacle theta, robot initial qpos,
object initial pose, camera/light randomization parameters, selected grasp
identity, retry count, retry-reason sequence, task success/failure, H5 field
names and shapes, the ordered 40-sensor names, and row outcome status.

Scientific arrays (qpos, actions, proximity depth, object/task state) are
**bit-identical**, so the tolerances below were never exercised. They were
encoded in the audit tool before Runs A/B/C executed and were not changed
afterwards:

| Quantity | Max absolute delta |
|---|---|
| qpos and actions | ≤ 1e-6 |
| object/robot poses | ≤ 1e-7 |
| proximity depth | ≤ 1e-5 |

A tolerance match is treated as insufficient whenever any discrete event
differs. A different selected grasp, retry count, rejection sequence, planner
phase path, success/failure outcome, hazard geometry, object identity or
collision outcome fails invariance outright.

**Audit correction (freeze generation 3).** After A/B/C executed, the audit tool
was corrected to handle NaN padding: `np.array_equal` reports `NaN != NaN`, and
`max|a−b|` over NaN yields NaN, which then slipped through a `delta > tolerance`
test because every NaN comparison is False. The frozen numeric tolerances were
**not** changed. The correction is analysis-only — it reads published output and
cannot alter what was simulated — and it made the verdict *stricter*: NaN
patterns are now required to match exactly, and the magnitude check runs over
finite entries only. Under the corrected check the runs compare bit-identical;
under the flawed one they had been reported as merely within tolerance.

### Duplicate and replica audit

| Run | Distinct episode IDs | Distinct episode-spec hashes | Distinct qpos+action trajectory hashes | Largest replica class |
|---|---|---|---|---|
| A | 8 | 8 | 8 | 1 |
| B | 8 | 8 | 8 | 1 |
| C | 8 | 8 | 8 | 1 |

No two distinct episode IDs share an immutable episode-spec hash. No two share
an exact full qpos+action trajectory hash. **No replica class analogous to the
old classes of three appears.**

### Worker-count, resume and scheduling

- **Worker-count invariance:** 1 worker vs 4 workers → bit-identical.
- **Resume invariance:** a hard-killed and resumed run → bit-identical to both
  uninterrupted runs; all 8 rows reconcile exactly once; nothing skipped,
  nothing duplicated.
- **Worker scheduling changed only descriptive metadata:** the audit records
  `worker_scheduling_changed_between_runs: true` — worker assignments genuinely
  differed across runs (e.g. candidate 0 ran on worker 0 / 2 / 1) while the
  scientific content did not change at all.

### Worker-completeness regression

All 17 fa8e61f tests remain green. The manifest runner records row lifecycle
*through* that mechanism: every worker publishes a terminal `WorkerReport`
extended with its claimed and finalised row IDs, the parent records every exit
code, and the final summary is published atomically via
`write_summary_atomically`. Row reconciliation is authoritative and the run exits
nonzero on unreconciled or silently lost work. All three runs reported
`complete: true`, `workers.complete: true`, `row_reconciliation.ok: true`.

---

## 16. Runtime provenance

`diagnostics_output/hybrid_obstacle_seeding/runtime_provenance.json`

| | |
|---|---|
| OS | Linux-6.8.0-101-generic-x86_64-with-glibc2.39 |
| Architecture | x86_64 |
| Python | 3.11.15 (`/root/act_retrain_venv`) |
| NumPy | 2.4.6 |
| Torch | 2.7.1+cu126 |
| MuJoCo | **3.5.0** (pinned) |
| Warp | **1.11.1** (pinned) |
| SciPy | 1.17.1 |
| h5py | 3.16.0 |
| CUDA toolkit (torch build) | 12.6 |
| GPU | NVIDIA A10, driver 570.86.10 |
| MUJOCO_GL | egl |
| Root commit at run | `beb125e97607cc34c5f64645a9872ccc78e33911` |
| MolmoSpaces commit at run | `fa8e61f40eb97e27bf3b69480c8bc65b0450f362` |
| ACT gitlink | `3d25c69edd8d972afa59fec5c3edb9d13a357f92` (unmodified) |
| `model_hybrid.xml` | `50924661e0411f92ab529c790512b17b674e789434c592c3dbc6d2359164d4c6` |
| Fumehood scene | `eec36c2ae5d17fc82d1221feaa87a1de2e9bec391902dfb1b46ef27ed7ad716d` |
| 40-sensor-order hash | `c31df8c36b0011b0eaf5b2eb5ce66d2514b5d6662ba9d7684ff021cd17cec858` |
| Runtime-contract hash | `64bc65689ae3d21b8f2e8e1ddf6672732f622dbbaea14851f1664b6eace7dd70` |

**No reproducibility is claimed across different software versions or hardware
from this smoke.** The tested contract is for the recorded runtime only. The
fa8e61f `runtime_compat` guard still fails the launcher fast on an unsupported
MuJoCo/Warp rather than letting it surface as an empty dataset.

---

## 17. Changed files and commits

### MolmoSpaces — `repair/hybrid-obstacle-manifest-runner-v2`

Base `fa8e61f40eb97e27bf3b69480c8bc65b0450f362` → commit `678f2eb4a0ac0d9e3d14e555aaac0e099089b9a5`.

```
molmo_spaces/data_generation/episode_manifest.py                    new
molmo_spaces/data_generation/row_ledger.py                          new
molmo_spaces/data_generation/manifest_runner.py                     new
molmo_spaces/data_generation/config/object_manipulation_datagen_configs.py   +64
molmo_spaces/tasks/enclosure_reach.py                               +32/-4
molmo_spaces/tasks/task_sampler.py                                  +23/-4
mlspaces_tests/data_generation/test_episode_manifest.py             new
mlspaces_tests/data_generation/test_manifest_hazard_isolation.py    new
```

Contains only manifest-row execution support, episode-scoped seeding, forced
hazard support, row lifecycle/output metadata, tests, and preserved
worker-completeness behavior. `ParallelRolloutRunner`'s default behavior for
unrelated configs is unchanged; both edits to existing task files are opt-in and
default to legacy behavior.

### Root — `repair/hybrid-obstacle-episode-manifest-v2`

Base `beb125e97607cc34c5f64645a9872ccc78e33911`; the commit below carries the updated MolmoSpaces gitlink.

```
configs/hybrid_obstacle_independent_v2.yaml                         new
configs/hybrid_obstacle_candidate_manifest_v2.json                  new (frozen 160 rows)
configs/hybrid_obstacle_manifest_v2_smoke8.json                     new
scripts/build_hybrid_obstacle_manifest_v2.py                        new
scripts/run_hybrid_obstacle_manifest_v2.py                          new
scripts/hybrid_obstacle_manifest_v2_audit.py                        new
scripts/hybrid_obstacle_manifest_v2_provenance.py                   new
docs/HYBRID_OBSTACLE_SEEDING_FINAL_DECISION.md                      new
diagnostics_output/hybrid_obstacle_seeding/*.json                   new (7 reports)
.gitignore                                                          smoke output ignored
submodules/molmospaces                                              gitlink updated
```

**Not committed:** smoke H5s, videos, per-row payloads, the invalid 1.33 GiB
collection, converted datasets, checkpoints, temporary outputs, unrelated
`EVAL.md` changes, 29-sensor PACT work. Nothing was pushed.

---

## 18. Exact reproduction commands

```bash
cd /root/prox_learning_hybrid_safety
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export MLSPACES_ASSETS_DIR=/root/prox_learning_hybrid_safety/assets
export PYTHONPATH=/root/prox_learning_hybrid_safety/submodules/molmospaces
PY=/root/act_retrain_venv/bin/python

# 1. Verify the frozen manifest regenerates identically
$PY scripts/build_hybrid_obstacle_manifest_v2.py --check
#    manifest OK  8be804057f3e0710a7c4770de54e8516e6156856423b6fdcac191ac0b15f2805
#    smoke8   OK  cb9df6e1f8dabb2e8705f685a8e1a0f9673ca184d1d455995e40e06f6bb4ddbc

# 2. Static tests (68)
cd submodules/molmospaces
$PY -m pytest mlspaces_tests/data_generation/test_episode_manifest.py \
              mlspaces_tests/data_generation/test_manifest_hazard_isolation.py \
              mlspaces_tests/data_generation/test_worker_completeness.py -q
cd /root/prox_learning_hybrid_safety

# 3. Run A — one worker
RUNS=diagnostics_output/hybrid_obstacle_seeding/smoke_runs
$PY scripts/run_hybrid_obstacle_manifest_v2.py --output-dir $RUNS/run_a --workers 1 --smoke

# 4. Run B — four workers
$PY scripts/run_hybrid_obstacle_manifest_v2.py --output-dir $RUNS/run_b --workers 4 --smoke

# 5. Run C — four workers, SIGKILL the process group once four rows have
#    finalized, then resume the SAME output directory
setsid $PY scripts/run_hybrid_obstacle_manifest_v2.py \
      --output-dir $RUNS/run_c --workers 4 --smoke &
#    ... wait for `ls $RUNS/run_c/rows/*/outcome.json | wc -l` to reach 4, then
#    kill -9 -<pgid>
$PY scripts/run_hybrid_obstacle_manifest_v2.py --output-dir $RUNS/run_c --workers 4 --smoke

# 6. Invariance audit (exit 0 iff invariant)
$PY scripts/hybrid_obstacle_manifest_v2_audit.py \
    --run A=$RUNS/run_a --run B=$RUNS/run_b --run C=$RUNS/run_c \
    --expected-rows 8 \
    --out diagnostics_output/hybrid_obstacle_seeding/invariance_report.json

# 7. Runtime provenance
$PY scripts/hybrid_obstacle_manifest_v2_provenance.py
```

---

## 19. Next recommended task

Launch the full 160-row canonical collection under this contract, in a task that
explicitly approves it:

```bash
$PY scripts/run_hybrid_obstacle_manifest_v2.py \
    --output-dir <fresh canonical dir> --workers 8
```

Points for whoever picks it up:

1. **Budget for the scene reload.** Every row rebuilds its scene by design — that
   is what buys the invariant. Observed cost was roughly 60–90 s per row on the
   recorded runtime; 160 rows on 8 workers is on the order of half an hour, and
   the run must not be "optimised" by restoring the scene cache.
2. **A failed row is a result, not a retry opportunity.** Do not re-run a row to
   improve the success count. If successes fall short of 75/25, the shortfall
   policy in §10 applies.
3. **Only then** apply the predeclared canonical selection and the fixed 80/20
   split, convert, and train ACT — each as its own approved task.

This task deliberately did not launch the full collection, run conversion, train
ACT, train or modify the Safety-CVAE, or evaluate any policy.

---

## 20. Decision

All conditions for the ready decision are met: the manifest and static tests
pass, worker-completeness tests remain green, Runs A and B are scientifically
invariant (in fact bit-identical), the interrupted and resumed Run C is
scientifically invariant, all eight manifest rows reconcile exactly once in every
run, no duplicate identity or replica trajectory appears, the frozen 160-row
manifest is committed, and both final artifacts are written.

SEEDING_CONTRACT_READY_FOR_FULL_COLLECTION
