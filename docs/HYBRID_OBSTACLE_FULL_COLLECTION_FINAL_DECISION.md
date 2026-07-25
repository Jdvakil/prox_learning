# Hybrid Obstacle Full Collection — Final Decision

Execution of the frozen 160-row `hybrid_obstacle_independent_v2` candidate
manifest, integrity audit of every written trajectory, construction of the
predeclared canonical 75/25 dataset, the frozen 80/20 split, and a
double-run reproducibility proof of the offline ACT conversion.

Date: 2026-07-25 · Task scope: execution and audit only. ACT was not trained or
modified. The Safety-CVAE was not trained, modified or re-checkpointed. No
policy evaluation was run.

---

## 1. Executive summary

The frozen manifest was executed exactly once, at 4 workers, into a single fresh
output directory, with no target-success count, no row substitution and no
parameter override. All **160 of 160**
manifest rows reached a terminal outcome and reconcile exactly once by candidate
index, episode ID and manifest-row hash. No claim was left unresolved and no row
was published twice.

Outcomes: **145 success**, 15 non-success
(15 task_failure).
Distinct successful rows by committed hazard label: **110 hazard-present**
of 120 and **35 hazard-absent** of 40.

The predeclared quota is 75 hazard-present and 25 hazard-absent distinct
successes. Observed: 110 / 35 — quota
**MET**.

Every successful H5 passed the full integrity audit: 145 of
145 clean, exactly 40 proximity streams each at
`(T,4,8,8)`, sensor order reproducing the committed
`c31df8c36b0011b0…`, rendered hazard geometry agreeing with the
committed label on every row, and both ACT RGB cameras present. Across all
successes there is **no duplicate episode ID, no duplicate row hash, and no two
distinct episode IDs sharing a core-trajectory, task-state, all-leaf or
episode-spec hash**. The largest replica class is 1 — the class-of-three failure
that voided the previous collection does not recur.

The eight smoke-reference rows were re-compared against the validated four-worker
smoke run using only the tolerances already frozen in the seeding audit:
**8/8 invariant, all bit-identical =
True**, with worker assignments differing between runs.

Canonical selection took the first 75 successful hazard-present and first 25
successful hazard-absent rows by predeclared stratum rank, consulting no
downstream metric, and regenerates to the same manifest hash. The split is
80 train / 20
validation at trajectory level with no episode, source-file or scientific-hash
overlap. Both ACT conversions produced 100 episodes with identical per-file
hashes and identical tree hashes.

**Final decision: `CANONICAL_DATASET_READY_FOR_ACT_TRAINING`** (token repeated verbatim as the last line).

---

## 2. Starting and final commit state

| | |
|---|---|
| Root branch | `collect/hybrid-obstacle-independent-v2` |
| **Root commit that produced the data** | `afac1d94583888a6402e48e98b0397e195b8e2e1` |
| Root base (validated handoff commit) | `afac1d94583888a6402e48e98b0397e195b8e2e1` |
| MolmoSpaces branch | `repair/hybrid-obstacle-manifest-runner-v2` |
| MolmoSpaces commit | `678f2eb4a0ac0d9e3d14e555aaac0e099089b9a5` |
| MolmoSpaces expected | `678f2eb4a0ac0d9e3d14e555aaac0e099089b9a5` |
| MolmoSpaces new commits | none — pinned exactly, source unmodified |
| ACT gitlink | `3d25c69edd8d972afa59fec5c3edb9d13a357f92` |
| ACT expected | `3d25c69edd8d972afa59fec5c3edb9d13a357f92` (unmodified) |
| Root gitlink → MolmoSpaces | matches the validated commit |

Submodule state at the end of the task:

```
4ff337dfc60708f46d5831430df807fa90014821 submodules/MolmoBot (4ff337d)
 3d25c69edd8d972afa59fec5c3edb9d13a357f92 submodules/act (heads/repair/live-safety-residual-eval)
 678f2eb4a0ac0d9e3d14e555aaac0e099089b9a5 submodules/molmospaces (heads/repair/hybrid-obstacle-manifest-runner-v2)
```

## 3. Manifest, smoke8 and contract hashes

| | |
|---|---|
| **Full 160-row manifest SHA-256** | `8be804057f3e0710a7c4770de54e8516e6156856423b6fdcac191ac0b15f2805` |
| Manifest file SHA-256 | `10e45609162a61371f1bf135b563c76c69787a2be58ce0fb245fd54a3688dff3` |
| **smoke8 subset SHA-256** | `cb9df6e1f8dabb2e8705f685a8e1a0f9673ca184d1d455995e40e06f6bb4ddbc` |
| smoke8 file SHA-256 | `8b7807f7380c5dd4d0732a11c1eb09f91ac317187ad2553fb16d2ecf38473ef7` |
| Master seed | **20260725** |
| Candidates | 160 (indices 0–159) |
| Hazard schedule | 120 present / 40 absent |
| 40-sensor-order SHA-256 | `c31df8c36b0011b0eaf5b2eb5ce66d2514b5d6662ba9d7684ff021cd17cec858` |
| `model_hybrid.xml` SHA-256 | `50924661e0411f92ab529c790512b17b674e789434c592c3dbc6d2359164d4c6` |
| Fumehood scene SHA-256 | `eec36c2ae5d17fc82d1221feaa87a1de2e9bec391902dfb1b46ef27ed7ad716d` |
| Env/config SHA-256 | `190547e0fa424b17594375e5347a0d21f809b945f14bc648a936e6eb38eccd16` |
| Runtime-contract SHA-256 | `64bc65689ae3d21b8f2e8e1ddf6672732f622dbbaea14851f1664b6eace7dd70` |
| Collection source digest | `d584f057f601000374198750f8030061625af1e682bf23d14cd451396f9bfe33` |

The manifest regenerated identically from committed source before launch
(`build_hybrid_obstacle_manifest_v2.py --check` reported both hashes OK).

## 4. Source and runtime verification

| | |
|---|---|
| OS | Linux-6.8.0-101-generic-x86_64-with-glibc2.39 |
| Kernel | 6.8.0-101-generic |
| Python | 3.11.15 (`/root/act_retrain_venv/bin/python`) |
| NumPy | 2.4.6 |
| Torch | 2.7.1+cu126 |
| **MuJoCo** | **3.5.0** (pinned) |
| **Warp** | **1.11.1** (pinned) |
| SciPy | 1.17.1 |
| h5py | 3.16.0 |
| CUDA toolkit (torch build) | 12.6 |
| GPU | NVIDIA A10, 570.86.10 |
| MUJOCO_GL | egl |

Every value matches the runtime recorded in the seeding decision report, on the
same machine. `runtime_compat.check_runtime()` returned zero issues, and the
launcher's `assert_supported_runtime(strict=True)` guard passed. All 68 static
tests pass (51 manifest/hazard-isolation + the 17 pre-existing
worker-completeness tests).

### Stream derivation, verified in committed source

The handoff requires that each scientific stream derive from an immutable key
containing at least `master_seed`, `candidate_index`, `stream_id` and
`retry_index`, and that worker ID, worker count, house alias, Python's builtin
`hash()` and a runtime-ordered `SeedSequence.spawn()` chain all be absent. This
was checked at AST level rather than from prose:

```
{
  "episode_manifest.py": {
    "builtin_hash_calls": [],
    "spawn_calls": [],
    "SeedSequence_calls": [
      214,
      268
    ]
  },
  "manifest_runner.py": {
    "builtin_hash_calls": [],
    "spawn_calls": [],
    "SeedSequence_calls": []
  },
  "row_ledger.py": {
    "builtin_hash_calls": [],
    "spawn_calls": [],
    "SeedSequence_calls": []
  },
  "stream_entropy_returns": [
    "master_seed",
    "candidate_index",
    "stream_id",
    "retry_index"
  ],
  "forbidden_tokens_in_entropy": [],
  "install_row_seed_contract_args": [
    "row",
    "retry_index",
    "task_sampler"
  ],
  "install_row_seed_contract_row_keys": [
    "camera_light",
    "candidate_index",
    "episode_id",
    "global_compat",
    "master_seed",
    "py_random",
    "seed_u32",
    "seed_u64",
    "torch"
  ]
}

STREAM_DERIVATION_OK
```

`stream_entropy` returns exactly
`[master_seed, candidate_index, stream_id, retry_index]`, and that list is passed
straight to `np.random.SeedSequence(...)` with no `spawn()` anywhere in the three
contract modules. `install_row_seed_contract` reads only `row["master_seed"]`,
`row["candidate_index"]` and its own `retry_index` argument. `worker_id` appears
in `manifest_runner.py` only as `worker_id_descriptive` — a recorded operational
label that reaches no draw.

## 5. Exact collection command

```bash
cd /root/prox_learning_hybrid_safety
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export MLSPACES_ASSETS_DIR=/root/prox_learning_hybrid_safety/assets
export PYTHONPATH=/root/prox_learning_hybrid_safety/submodules/molmospaces

/root/act_retrain_venv/bin/python scripts/run_hybrid_obstacle_manifest_v2.py \
    --output-dir assets/datagen/hybrid_obstacle_independent_v2/20260725_full160_4w \
    --workers 4
```

No `--smoke`, so the full committed 160-row manifest was used. No target-success
count, no row substitution, no row replacement after failure, no environment or
parameter override. Worker-completeness monitoring and atomic row
claim/finalisation/publication are enabled by the runner. Full stdout/stderr and
per-worker logs were retained, and the source/runtime/manifest hashes were copied
into `_provenance/` **before** launch.

**Deviation from the seeding report, stated explicitly:** §19 of
`HYBRID_OBSTACLE_SEEDING_FINAL_DECISION.md` suggests `--workers 8`. The handoff
for this task mandates exactly 4 workers, so 4 was used. 4 is also the worker
count of the validated Run B / Run C smoke references, which makes the
smoke-reference comparison in §10 a like-for-like comparison. Worker-count
invariance is a proven property of the contract, so this does not affect the
scientific content of any row.

Output path: `assets/datagen/hybrid_obstacle_independent_v2/20260725_full160_4w`

## 6. Collection duration

| | |
|---|---|
| Launched (UTC) | see _provenance/launch_time.txt |
| Wall-clock duration | 1h 5m 32s (3932 s) |
| Rows | 160 |
| Workers | 4 |
| Mean per-row cost (worker-seconds) | 98.3 s |

Per-row scene reconstruction remained enabled throughout. No scene or sampler
cache was introduced and the per-row rebuild was not optimised away.

## 7. All 160 row outcomes

Status totals:

| | |
|---|---|
| success | 145 |
| scientific task failure (`task_failure`) | 15 |
| sampling/reset failure (`sampling_failure`) | 0 |
| infrastructure failure (`infrastructure_failure`) | 0 |
| total | 160 |
| retries recorded across all rows | 40 |

Any status not listed above is zero. The four names are the runner's complete
terminal-outcome vocabulary (`episode_manifest.py:137-140`).

Hazard breakdown by outcome:

| | |
|---|---|
| success — hazard-present | 110 |
| task_failure — hazard-present | 10 |
| success — hazard-absent | 35 |
| task_failure — hazard-absent | 5 |

Success rates: hazard-present **110/120 =
91.7%**, hazard-absent
**35/40 = 87.5%**,
overall **145/160 = 90.6%**.

### Retry histogram

| | |
|---|---|
| 0 | 127 |
| 1 | 29 |
| 2 | 2 |
| 3 | 1 |
| 4 | 1 |

### Failure reason histogram

| | |
|---|---|
| rollout completed without task success | 11 |
| rollout completed without task success; earlier retries: ValueError: IK failed for lift pose | 4 |

### Failure phase histogram

| | |
|---|---|
| rollout_completed_task_not_achieved | 15 |

### Retry reason histogram

Retries are the bounded deterministic re-draws already defined by the committed
row contract (max 4 per row); each retry
re-derives every stream at a fresh `retry_index`.

| | |
|---|---|
| ValueError: IK failed for lift pose | 29 |
| ValueError: IK failed for pregrasp pose | 10 |
| ValueError: No feasible grasp found | 1 |

### Per-row detail

| cand | hazard | outcome | retries | worker | duration s | reason |
|---|---|---|---|---|---|---|
| 0 | present | success | 0 | 0 | 109.8 |  |
| 1 | present | success | 0 | 1 | 112.5 |  |
| 2 | present | success | 0 | 2 | 63.1 |  |
| 3 | present | success | 1 | 3 | 110.7 |  |
| 4 | present | success | 0 | 0 | 67.7 |  |
| 5 | absent | success | 0 | 2 | 65.9 |  |
| 6 | present | success | 1 | 1 | 108.0 |  |
| 7 | present | success | 0 | 0 | 94.0 |  |
| 8 | absent | success | 0 | 2 | 63.0 |  |
| 9 | present | success | 0 | 3 | 95.1 |  |
| 10 | present | success | 0 | 1 | 106.1 |  |
| 11 | present | success | 0 | 2 | 81.1 |  |
| 12 | present | success | 0 | 0 | 110.6 |  |
| 13 | present | success | 0 | 3 | 95.0 |  |
| 14 | present | success | 0 | 2 | 62.6 |  |
| 15 | present | success | 0 | 1 | 60.1 |  |
| 16 | present | success | 0 | 3 | 108.2 |  |
| 17 | present | success | 0 | 0 | 86.1 |  |
| 18 | absent | success | 0 | 2 | 65.0 |  |
| 19 | present | success | 1 | 1 | 135.1 |  |
| 20 | present | success | 0 | 2 | 62.6 |  |
| 21 | present | success | 2 | 0 | 145.9 |  |
| 22 | present | success | 0 | 3 | 67.3 |  |
| 23 | present | success | 0 | 2 | 75.1 |  |
| 24 | absent | success | 1 | 1 | 68.8 |  |
| 25 | absent | task_failure | 1 | 3 | 99.7 |  |
| 26 | present | success | 0 | 2 | 89.3 |  |
| 27 | present | success | 0 | 1 | 62.8 |  |
| 28 | present | success | 0 | 0 | 63.7 |  |
| 29 | present | success | 0 | 3 | 107.2 |  |
| 30 | present | success | 0 | 1 | 61.1 |  |
| 31 | present | success | 0 | 0 | 86.8 |  |
| 32 | present | success | 0 | 2 | 84.9 |  |
| 33 | absent | success | 4 | 1 | 115.9 |  |
| 34 | present | success | 1 | 3 | 73.6 |  |
| 35 | present | task_failure | 1 | 0 | 103.8 |  |
| 36 | present | success | 0 | 2 | 93.3 |  |
| 37 | present | success | 1 | 3 | 105.1 |  |
| 38 | present | success | 0 | 1 | 60.7 |  |
| 39 | present | success | 0 | 0 | 95.2 |  |
| 40 | present | success | 0 | 2 | 100.6 |  |
| 41 | absent | task_failure | 0 | 1 | 63.5 |  |
| 42 | present | success | 0 | 3 | 62.5 |  |
| 43 | present | success | 1 | 0 | 143.2 |  |
| 44 | present | success | 0 | 2 | 62.6 |  |
| 45 | present | success | 0 | 1 | 63.2 |  |
| 46 | present | task_failure | 0 | 3 | 121.2 |  |
| 47 | present | success | 0 | 2 | 124.5 |  |
| 48 | present | success | 0 | 1 | 81.8 |  |
| 49 | present | success | 0 | 0 | 84.3 |  |
| 50 | absent | success | 0 | 3 | 85.7 |  |
| 51 | present | success | 0 | 1 | 101.3 |  |
| 52 | present | task_failure | 1 | 2 | 73.9 |  |
| 53 | present | success | 0 | 0 | 106.1 |  |
| 54 | absent | success | 1 | 3 | 85.6 |  |
| 55 | present | success | 0 | 2 | 57.5 |  |
| 56 | present | success | 0 | 1 | 90.1 |  |
| 57 | present | task_failure | 0 | 3 | 125.3 |  |
| 58 | present | success | 0 | 2 | 111.5 |  |
| 59 | present | success | 0 | 0 | 92.7 |  |
| 60 | absent | success | 1 | 1 | 70.4 |  |
| 61 | absent | success | 0 | 0 | 89.1 |  |
| 62 | present | success | 1 | 1 | 95.7 |  |
| 63 | present | success | 1 | 3 | 66.0 |  |
| 64 | present | success | 0 | 2 | 85.6 |  |
| 65 | present | success | 1 | 3 | 108.7 |  |
| 66 | present | success | 0 | 0 | 57.9 |  |
| 67 | present | success | 0 | 2 | 55.7 |  |
| 68 | absent | success | 0 | 1 | 67.2 |  |
| 69 | absent | success | 0 | 0 | 60.7 |  |
| 70 | absent | success | 0 | 2 | 56.0 |  |
| 71 | absent | success | 1 | 1 | 103.1 |  |
| 72 | present | success | 0 | 3 | 124.1 |  |
| 73 | present | success | 0 | 0 | 86.7 |  |
| 74 | present | success | 0 | 2 | 101.2 |  |
| 75 | present | success | 0 | 1 | 84.3 |  |
| 76 | absent | success | 0 | 0 | 58.1 |  |
| 77 | present | success | 0 | 3 | 84.8 |  |
| 78 | present | success | 0 | 2 | 60.4 |  |
| 79 | present | success | 0 | 1 | 60.3 |  |
| 80 | present | success | 2 | 0 | 237.0 |  |
| 81 | present | success | 1 | 2 | 102.3 |  |
| 82 | present | success | 0 | 3 | 61.0 |  |
| 83 | present | task_failure | 0 | 1 | 51.4 |  |
| 84 | absent | success | 0 | 3 | 58.6 |  |
| 85 | absent | success | 0 | 1 | 65.5 |  |
| 86 | present | success | 0 | 2 | 61.2 |  |
| 87 | present | success | 0 | 3 | 62.2 |  |
| 88 | present | success | 0 | 1 | 54.2 |  |
| 89 | present | success | 0 | 2 | 89.9 |  |
| 90 | absent | success | 0 | 3 | 88.1 |  |
| 91 | absent | task_failure | 0 | 0 | 60.1 |  |
| 92 | present | success | 1 | 1 | 103.9 |  |
| 93 | absent | success | 0 | 2 | 120.5 |  |
| 94 | absent | success | 0 | 0 | 58.4 |  |
| 95 | absent | success | 0 | 3 | 84.8 |  |
| 96 | absent | task_failure | 1 | 1 | 57.5 |  |
| 97 | present | task_failure | 0 | 0 | 121.2 |  |
| 98 | present | success | 0 | 1 | 59.0 |  |
| 99 | present | success | 0 | 2 | 122.4 |  |
| 100 | present | success | 0 | 3 | 56.9 |  |
| 101 | absent | success | 0 | 1 | 85.0 |  |
| 102 | present | success | 0 | 0 | 95.0 |  |
| 103 | present | success | 1 | 3 | 98.6 |  |
| 104 | present | success | 1 | 2 | 110.1 |  |
| 105 | present | success | 0 | 1 | 62.9 |  |
| 106 | present | success | 0 | 0 | 104.0 |  |
| 107 | present | success | 1 | 3 | 110.3 |  |
| 108 | present | success | 0 | 1 | 62.7 |  |
| 109 | present | task_failure | 0 | 2 | 93.9 |  |
| 110 | present | success | 1 | 0 | 153.6 |  |
| 111 | present | success | 0 | 1 | 61.7 |  |
| 112 | present | success | 0 | 3 | 88.6 |  |
| 113 | absent | success | 0 | 2 | 85.3 |  |
| 114 | present | success | 0 | 1 | 91.0 |  |
| 115 | absent | success | 0 | 3 | 67.5 |  |
| 116 | absent | success | 0 | 2 | 82.7 |  |
| 117 | absent | success | 0 | 0 | 63.4 |  |
| 118 | absent | success | 0 | 1 | 59.6 |  |
| 119 | present | success | 0 | 3 | 61.3 |  |
| 120 | absent | success | 1 | 0 | 113.6 |  |
| 121 | present | success | 0 | 2 | 64.5 |  |
| 122 | absent | success | 0 | 1 | 84.6 |  |
| 123 | present | success | 0 | 3 | 97.7 |  |
| 124 | absent | success | 1 | 2 | 94.2 |  |
| 125 | present | success | 0 | 1 | 96.6 |  |
| 126 | present | success | 0 | 0 | 127.9 |  |
| 127 | present | success | 0 | 3 | 100.9 |  |
| 128 | present | success | 0 | 2 | 88.1 |  |
| 129 | present | success | 0 | 1 | 64.2 |  |
| 130 | absent | success | 0 | 3 | 84.4 |  |
| 131 | present | success | 0 | 0 | 89.1 |  |
| 132 | present | task_failure | 0 | 2 | 102.3 |  |
| 133 | absent | success | 1 | 1 | 87.7 |  |
| 134 | absent | task_failure | 0 | 3 | 85.2 |  |
| 135 | present | success | 0 | 0 | 103.4 |  |
| 136 | absent | success | 1 | 2 | 65.0 |  |
| 137 | present | task_failure | 0 | 1 | 48.0 |  |
| 138 | present | success | 0 | 3 | 97.4 |  |
| 139 | present | success | 0 | 1 | 84.0 |  |
| 140 | present | success | 1 | 2 | 191.2 |  |
| 141 | present | success | 0 | 0 | 103.9 |  |
| 142 | present | success | 0 | 3 | 80.2 |  |
| 143 | present | task_failure | 0 | 1 | 81.5 |  |
| 144 | present | success | 0 | 0 | 102.0 |  |
| 145 | absent | success | 3 | 1 | 102.4 |  |
| 146 | present | success | 0 | 3 | 96.6 |  |
| 147 | present | success | 0 | 2 | 94.9 |  |
| 148 | present | success | 0 | 0 | 82.8 |  |
| 149 | absent | success | 0 | 3 | 59.6 |  |
| 150 | absent | success | 0 | 1 | 59.1 |  |
| 151 | present | success | 0 | 2 | 81.7 |  |
| 152 | present | success | 0 | 3 | 59.9 |  |
| 153 | present | success | 0 | 0 | 106.8 |  |
| 154 | present | success | 0 | 1 | 61.1 |  |
| 155 | present | success | 0 | 2 | 98.5 |  |
| 156 | present | success | 0 | 3 | 62.9 |  |
| 157 | present | success | 1 | 1 | 96.7 |  |
| 158 | present | success | 0 | 0 | 95.5 |  |
| 159 | present | success | 0 | 3 | 60.4 |  |

## 8. Worker reconciliation

| | |
|---|---|
| Rows accounted for | 160/160 |
| Every candidate index exactly once | True |
| Duplicate candidate indices | 0 |
| Duplicate episode IDs | 0 |
| Duplicate manifest-row hashes | 0 |
| Unresolved claims | 0 |
| Stray row directories | 0 |
| Missing rows | 0 |
| Hazard label mismatches vs manifest | 0 |
| Row hash mismatches vs manifest | 0 |
| Collection summary published | True |
| Reconciliation verdict | **ok** |

Authoritative verdicts from the published `collection_summary.json`:

| | |
|---|---|
| `complete` | True |
| `status` | `complete` |
| `workers.complete` | True |
| `row_reconciliation.ok` | True |
| Expected / finalized rows | 160 / 160 |
| Silently lost workers | [] |
| Workers missing a final status | [] |
| Workers with failed status | [] |
| Nonzero/unknown exit codes | {} (none) |
| Worker exit codes | {'0': 0, '1': 0, '2': 0, '3': 0} |
| Rows missing an outcome | [] |
| Rows never claimed | [] |
| Published without outcome | [] |
| Unexpected row directories | [] |
| Reclaimed abandoned claims | [] |
| Rows already finalised on entry | 0 |
| Every worker has an approved final record | True |
| Parent and worker totals reconcile | True |
| Per-worker sums | {'episodes_attempted': 160, 'episodes_successful': 145, 'episodes_written': 145} |
| Parent shared counters | {'rows_expected': 160, 'rows_failed': 15, 'rows_finalized': 160, 'rows_succeeded': 145} |

Rows per worker:

| | |
|---|---|
| 0 | 36 |
| 1 | 44 |
| 2 | 40 |
| 3 | 40 |

No silent worker loss and no duplicate publication occurred; no finalized row was
rerun, and no abandoned claim needed reclaiming.

**One field must not be misread.** The summary also carries a
`warning: "Partial output retained…"` string
(present: True). That string is a stale
artifact, not a worker-loss signal. `build_final_summary`
(`worker_completeness.py:245-259`) derives `complete` from a *house*-based
comparison and inserts the warning whenever that comparison fails; a manifest run
writes no houses at all, so the house comparison always fails. The manifest
runner then overrides `houses_missing`, `houses_unexpected`, `complete` and
`status` from the authoritative row reconciliation
(`manifest_runner.py:685-700`), and raises `WorkerCompletenessError` — exiting
nonzero — if reconciliation does not pass, but it never deletes the already-written
`warning` key. The validated four-worker Run B smoke reference carries the
identical string alongside `complete: true`, which confirms this is pre-existing
behaviour and not a regression from this run. The load-bearing fields are the ones
tabulated above.

## 9. Source collection freeze, H5 and 40-sensor integrity

| | |
|---|---|
| File count | 1344 |
| Total size | 1121586684 bytes (1.045 GiB) |
| **Deterministic source tree SHA-256** | `8b569d0e20804949f6cd344a761de17fb6207863275d66c8fa1aef587bc21f30` |
| Successes audited | 145 |
| Successes clean | 145 |
| Sensor-order hash reproduced | True |
| Sensor-order formula | `sha256(json.dumps(ordered_names, separators=(',',':')))` |
| Files changed during the audit | 0 |

Per-file hashes are recorded in the committed source manifest. Every successful
H5 was verified to open, be untruncated, carry an episode ID matching the ledger,
a row hash matching the manifest, a hazard label matching the manifest, rendered
hazard geometry agreeing with that label, complete seed metadata matching the
committed per-row seed map at retry 0, target/object identity, obstacle theta,
robot initial state, qpos and actions, both ACT RGB cameras, exactly 40 proximity
streams each shaped `(T,4,8,8)`, a sensor order reproducing the committed hash,
task-state and success metadata, and non-negative initial clearance (no
penetration).

### Duplicate and replica audit

| | |
|---|---|
| Duplicate full-file hashes (A) | 0 |
| Duplicate all-leaf scientific hashes (B) | 0 |
| Duplicate core-trajectory hashes (C) | 0 |
| Duplicate task-state hashes (D) | 0 |
| Duplicate episode-spec hashes (E) | 0 |
| Replica class size histogram | {{}} (all classes size 1) |

The complete 160-row collection, including every failure, is retained as the
primary provenance record and was marked read-only after all required writes.
No failure and no non-canonical successful episode was deleted.

## 10. Smoke-reference comparison

Reference: the retained validated four-worker smoke run
(`diagnostics_output/hybrid_obstacle_seeding/smoke_runs/run_b`). Retained
reference H5s were available: **True**, so recorded
hashes were not needed as a substitute.

The comparison is delegated to the already-committed
`scripts/hybrid_obstacle_manifest_v2_audit.py`, so the tolerances, exact-match
field list and discrete-event field list are exactly those frozen during the
seeding audit. None was created or relaxed for this task.

| | |
|---|---|
| Frozen tolerances | `{"actions": 1e-06, "object_pose": 1e-07, "proximity_depth": 1e-05, "qpos": 1e-06, "robot_pose": 1e-07}` |
| Episodes compared | 8/8 |
| Episode ID sets match | True |
| All invariant | True |
| **All bit-identical** | **True** |

- candidate 5, `02ce6a616ae21ff3…` — bit-identical, smoke worker 2 → full-run worker 3
- candidate 3, `05ac82b8ad8f6fa5…` — bit-identical, smoke worker 3 → full-run worker 1
- candidate 1, `14a576f2851c7120…` — bit-identical, smoke worker 1 → full-run worker 0
- candidate 8, `2b3349e9ef97bd40…` — bit-identical, smoke worker 2 → full-run worker 2
- candidate 18, `37845503017238ff…` — bit-identical, smoke worker 2 → full-run worker 0
- candidate 2, `3cb1b467a0561ca3…` — bit-identical, smoke worker 2 → full-run worker 3
- candidate 24, `bfbb6269f1d3a2ca…` — bit-identical, smoke worker 1 → full-run worker 1
- candidate 0, `ec9d068e369e791e…` — bit-identical, smoke worker 0 → full-run worker 2

Manifest row, hazard label, obstacle theta, object identity, robot/object initial
state, selected grasp, retry count and reasons, result, and H5 field names and
shapes all match; the scientific arrays are bit-identical while worker
assignments differ.

## 11. Quota result

| | |
|---|---|
| Distinct successful hazard-present rows | **110** (required ≥ 75) |
| Distinct successful hazard-absent rows | **35** (required ≥ 25) |
| Verdict | **PASS** |

Counted from manifest identity and the recorded scientific success outcome only.
Retries are not counted as separate rows, no replica is counted, the quota was
not lowered, no hazard label was changed, and no failed scientific row was rerun.

## 12. Canonical 75/25 manifest

Label: **`controlled_predeclared_canonical_subset`**

| | |
|---|---|
| Total rows | 100 |
| Hazard-present | 75 |
| Hazard-absent | 25 |
| Selection rule | first N successful rows per hazard stratum ordered by predeclared stratum_rank; no downstream metric consulted |
| **Canonical manifest SHA-256** | `f49f5cd14b3c75b88e312cbad201273bddc7cdc100436a09fbfb74bfe3bb84cf` |
| Selection code SHA-256 | `dc34d8ad23ad099941ba4b5b77f1e79122c18e357d243ab1b3bfc3b21b609d2f` |
| Source collection tree SHA-256 | `8b569d0e20804949f6cd344a761de17fb6207863275d66c8fa1aef587bc21f30` |
| Excluded successful rows | 45 |
| Failed rows recorded | 15 |
| Rows promoted past the manifest's own reserve boundary | 10 |
| Regenerates to the same hash | True |

The manifest records, per selected row: episode ID, candidate index, row hash,
source H5 hash, hazard label, predeclared stratum and canonical ranks, selection
reason, split label and split rank. Excluded successful rows carry an explicit
exclusion reason and failed rows carry their outcome. Selection inspected no
trajectory length, retry count, clearance, collision severity, proximity
activation, action smoothness, image quality, planner phase duration or model
score.

## 13. Frozen 80/20 split

| | |
|---|---|
| Train total | 80 |
| Train hazard-present | 60 |
| Train hazard-absent | 20 |
| Validation total | 20 |
| Validation hazard-present | 15 |
| Validation hazard-absent | 5 |
| Split level | trajectory |
| **Split manifest SHA-256** | `f7c2b22718f1697ea153926220a48bac1ab5876f6119d863317117d04474ccd0` |
| Leakage free | True |

## 14. Conversion A/B

| | |
|---|---|
| Converter | `scripts/convert_obstacle_to_act.py` (unmodified, SHA-256 `74b60458754b7823…`) |
| conversion_A | `assets/act_style_data/hybrid_obstacle_canonical_v2/conversion_A` |
| conversion_B | `assets/act_style_data/hybrid_obstacle_canonical_v2/conversion_B` |
| Episodes A / B | 100 / 100 |
| Hazard present / absent | 75 / 25 |
| Train / validation | 80 / 20 |
| Episode length range | 54..130 |
| **Tree file SHA-256 A** | `a567df08e3bea549a1f8f6ddfe06d8c2d6b0e8e7816759404312497ff36d7c47` |
| **Tree file SHA-256 B** | `a567df08e3bea549a1f8f6ddfe06d8c2d6b0e8e7816759404312497ff36d7c47` |
| Tree file hashes equal | True |
| Tree semantic SHA-256 A | `8ca1d9540160fd4c7c79db1aed0aad364bc1aa3d14a4c7ac3d6aedf7dd08820f` |
| Tree semantic SHA-256 B | `8ca1d9540160fd4c7c79db1aed0aad364bc1aa3d14a4c7ac3d6aedf7dd08820f` |
| Tree semantic hashes equal | True |
| Per-file hash differences | 0 |
| Semantic hash differences | 0 |
| Episode ID differences | 0 |
| **Conversions identical** | **True** |

Each episode carries `exo_camera_1` and `wrist_camera` images, `qpos` shaped
`(T,9)` and `action` shaped `(T,8)` — the dimensions ACT's `obstacle_baseline`
task declares — plus a complete source-provenance mapping from ACT episode index
back to episode ID, candidate index, row hash and source H5 hash.

The episode set and each ACT episode index come from the canonical manifest, not
from filesystem iteration order. The committed converter's discovery helper
(`_find_h5_files`, which globs `house_*/trajectories*.h5`) and its
`episode_<i>_<cam>_batch_1_of_1.mp4` naming assumption do not match the
manifest-runner layout `rows/<episode_id>/`, and its entry point assigns the ACT
index from directory order. A thin manifest-driven wrapper therefore imports and
reuses the committed converter's decode and video functions verbatim
(`_decode_action`, `_decode_qpos_qvel`, `_video_frames`) together with its
dimension constants and output schema. `scripts/convert_obstacle_to_act.py` was
not modified, no ACT constant was changed, and the prior 69/31 conversion was not
overwritten.

## 15. Leakage audit

| | |
|---|---|
| core_trajectory_sha256 | 0 |
| duplicate_episode_ids_within_selection | 0 |
| duplicate_source_files_within_selection | 0 |
| episode_id | 0 |
| source_h5_sha256 | 0 |
| source_relpath | 0 |
| task_state_sha256 | 0 |

No episode overlap, no source-file overlap, no duplicate scientific hash across
splits, and the split is determined exclusively by the committed rank logic.
Because the split is at trajectory level and each source trajectory contributes
to exactly one ACT episode, no frame of any episode appears in both splits.

## 16. Final offline validation

| | |
|---|---|
| canonical_selection_regeneration | **PASS** |
| clean_submodule_verification | **PASS** |
| double_conversion_reproducibility | **PASS** |
| duplicate_content_audit | **PASS** |
| final_process_guard | **PASS** |
| git_diff_check | **PASS** |
| hazard_label_geometry_audit | **PASS** |
| json_yaml_parsing | **PASS** |
| manifest_identity_audit | **PASS** |
| python_byte_compilation | **PASS** |
| quota_check | **PASS** |
| ruff | **PASS** |
| sensor_schema_audit | **PASS** |
| smoke_reference_comparison | **PASS** |
| source_integrity_audit | **PASS** |
| source_unchanged_during_audit | **PASS** |
| split_leakage_audit | **PASS** |
| worker_completion_reconciliation | **PASS** |

All checks: **PASS**

No new simulation was launched after the full collection completed.

## 17. Changed files and commit

Committed on `collect/hybrid-obstacle-independent-v2`, on top of
`afac1d94583888a6402e48e98b0397e195b8e2e1` — the commit whose source produced the data. This document
records no hash for the provenance commit itself: the commit contains this file,
so any hash written here would be invalidated by the act of committing it. Resolve
it with `git log -1 --format=%H collect/hybrid-obstacle-independent-v2`.

Files in the provenance commit:

```
README.md                                          |    65 +
 configs/hybrid_obstacle_canonical_manifest_v2.json |  2977 ++++
 configs/hybrid_obstacle_canonical_split_v2.json    |  1341 ++
 .../conversion_A_manifest.json                     |  3521 +++++
 .../conversion_B_manifest.json                     |  3521 +++++
 .../final_decision.json                            |   380 +
 .../final_offline_validation.json                  |   202 +
 .../integrity_audit.json                           | 14591 +++++++++++++++++++
 .../old_invalid_collection_state.json              |    59 +
 .../pre_collection_state.txt                       |     8 +
 .../prelaunch_provenance.json                      |    68 +
 .../runtime_provenance.json                        |    53 +
 .../smoke8_revalidation.json                       |   306 +
 .../smoke8_revalidation_summary.json               |   194 +
 .../source_manifest.json                           |  6728 +++++++++
 .../stream_derivation_check.txt                    |    45 +
 ...BRID_OBSTACLE_FULL_COLLECTION_FINAL_DECISION.md |   873 ++
 scripts/hybrid_obstacle_build_canonical_subset.py  |   323 +
 .../hybrid_obstacle_convert_canonical_to_act.py    |   251 +
 scripts/hybrid_obstacle_full_collection_audit.py   |   747 +
 .../hybrid_obstacle_full_collection_validate.py    |   431 +
 .../hybrid_obstacle_smoke8_reference_compare.py    |   180 +
 scripts/hybrid_obstacle_write_final_decision.py    |   902 ++
 23 files changed, 37766 insertions(+)
```

Not committed: H5 trajectories, videos, converted ACT data, checkpoints,
temporary logs, the old invalid collection, unrelated `EVAL.md` changes, and any
MolmoSpaces or ACT source change. No new MolmoSpaces commit was created. Nothing
was pushed.

## 18. Exact reproduction commands

```bash
cd /root/prox_learning_hybrid_safety
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export MLSPACES_ASSETS_DIR=/root/prox_learning_hybrid_safety/assets
export PYTHONPATH=/root/prox_learning_hybrid_safety/submodules/molmospaces
PY=/root/act_retrain_venv/bin/python
RUN=assets/datagen/hybrid_obstacle_independent_v2/20260725_full160_4w
DIAG=diagnostics_output/hybrid_obstacle_full_collection

# 1. verify the frozen manifest regenerates
$PY scripts/build_hybrid_obstacle_manifest_v2.py --check

# 2. static tests (68)
(cd submodules/molmospaces && $PY -m pytest \
    mlspaces_tests/data_generation/test_episode_manifest.py \
    mlspaces_tests/data_generation/test_manifest_hazard_isolation.py \
    mlspaces_tests/data_generation/test_worker_completeness.py -q)

# 3. the collection (already executed once; this would be a second run)
$PY scripts/run_hybrid_obstacle_manifest_v2.py --output-dir $RUN --workers 4

# 4. integrity audit, source freeze
$PY scripts/hybrid_obstacle_full_collection_audit.py \
    --run $RUN \
    --manifest configs/hybrid_obstacle_candidate_manifest_v2.json \
    --stack configs/hybrid_safety_stack_v1.json \
    --out $DIAG/integrity_audit.json \
    --source-manifest $DIAG/source_manifest.json

# 5. smoke-reference revalidation (frozen tolerances, committed audit)
$PY scripts/hybrid_obstacle_smoke8_reference_compare.py \
    --run $RUN \
    --reference diagnostics_output/hybrid_obstacle_seeding/smoke_runs/run_b \
    --smoke-subset configs/hybrid_obstacle_manifest_v2_smoke8.json \
    --view /tmp/smoke8_view \
    --out $DIAG/smoke8_revalidation.json \
    --decision-json diagnostics_output/hybrid_obstacle_seeding/final_decision.json

# 6. quota gate, canonical 75/25 manifest, frozen 80/20 split
$PY scripts/hybrid_obstacle_build_canonical_subset.py \
    --run $RUN \
    --manifest configs/hybrid_obstacle_candidate_manifest_v2.json \
    --audit $DIAG/integrity_audit.json \
    --source-manifest $DIAG/source_manifest.json \
    --out-canonical configs/hybrid_obstacle_canonical_manifest_v2.json \
    --out-split configs/hybrid_obstacle_canonical_split_v2.json

# 7. double conversion
$PY scripts/hybrid_obstacle_convert_canonical_to_act.py \
    --run $RUN --canonical configs/hybrid_obstacle_canonical_manifest_v2.json \
    --dst assets/act_style_data/hybrid_obstacle_canonical_v2/conversion_A --manifest-out $DIAG/conversion_A_manifest.json
$PY scripts/hybrid_obstacle_convert_canonical_to_act.py \
    --run $RUN --canonical configs/hybrid_obstacle_canonical_manifest_v2.json \
    --dst assets/act_style_data/hybrid_obstacle_canonical_v2/conversion_B --manifest-out $DIAG/conversion_B_manifest.json

# 8. final offline validation
$PY scripts/hybrid_obstacle_full_collection_validate.py --run $RUN ...
```

## 19. Next recommended task

Train the vanilla ACT baseline on `conversion_A` in its own explicitly approved
task:

1. Point ACT's `obstacle_baseline` `dataset_dir` at the conversion_A directory and
   set `num_episodes=100` and `episode_len=132` (max T =
   130). Changing those two constants is the only ACT edit that should
   be needed, and it belongs to the training task, not this one.
2. Hold the 20-episode validation split out
   of training; it is already labelled in the split manifest.
3. The Safety-CVAE comparison arm needs a proximity-exporting converter; the
   40-stream source data is present and audited, but exporting it is a separate
   task.

This task did not train ACT, did not train or modify the Safety-CVAE, and ran no
policy evaluation.

## 20. Decision

| | |
|---|---|
| All 160 rows reconcile | True |
| Collection integrity passes | True |
| No replicas exist | True |
| ≥75 hazard-present and ≥25 hazard-absent successes | True |
| Canonical selection holds exactly 100 distinct trajectories | True |
| Split holds exactly 80 train and 20 validation | True |
| Both conversions identical | True |
| No leakage | True |
| All mandatory reports written | True |

CANONICAL_DATASET_READY_FOR_ACT_TRAINING
