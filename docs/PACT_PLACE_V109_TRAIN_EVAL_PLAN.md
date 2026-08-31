# V10.9 — Exploratory conversion, ACT-vs-PACT training, and paired learned-policy evaluation

**Status:** registered 2026-08-29, **executed and closed 2026-08-29**.
Close-out: `diagnostics_output/pact_place_v109_train_eval/closeout.json`
(`3247cbfd1a6b0e74…`). Outcome: **null**. On 40 paired held-out instances PACT
did not beat ACT on any endpoint; every paired interval crosses zero and no
exact McNemar p-value falls below 0.45. Zero pendant contact in all 80
rollouts.
**Authorization:** explicit owner authorization to convert, train, and evaluate the 141
accepted V10.8 demonstrations, *despite* V10.7's failed Phase-0 gate and V10.8's
owner-instructed early stop.

**What this task does not do.** It does not alter, reinterpret, rerun, or rehabilitate
either historical result. V10.7's Phase-0 gate remains **failed at 8/24 and permanently
closed**. V10.8 remains an **exploratory owner-override collection stopped early by owner
instruction at 141 of 152 target successes**, with quotas unmet. Every V5 and V10.4–V10.8
artifact is preserved byte-for-byte; every artifact produced here is newly versioned and
create-only.

---

## 0. Correct the V10.8 record

The previous V10.8 narrative is treated as untrusted. Every number below is independently
re-derived from `ledger.jsonl` and the retained row files, not copied from `closeout.json`.

A create-only erratum is written to
`diagnostics_output/pact_place_v109_train_eval/v108_erratum.json` (with a Markdown
companion). **`closeout.json` is not overwritten.** The erratum corrects:

1. **Pendant involvement.** Zero *accepted* rows contacted the pendant — that claim stands.
   The claim that there was *no pendant involvement anywhere* is false.
2. **The one pendant-contact attempt.** Rejected attempt
   `1a756c9304311cdc07091641e59af8da16b6098550aa2fc9ce9d1c0c99cb6ae8`
   (cell `F0_target_side_stagger|right|neg5`, attempt_index 10, seed 1876755499)
   recorded **42 `mounted_fixture` contacts**, **one pendant-contact frame**, and
   **`min_pendant_clearance_m = 0.0`**. It is the sole such row in all 353 attempts, and it
   was correctly rejected.
3. **The eight worker deaths.** `closeout.json` reports `infrastructure_halts: 0`. That is
   wrong. `collection_1787966003.json` records **eight** `BrokenProcessPool` worker deaths
   and one `halted_for_repair` entry naming one of those same eight. They were caused by
   **terminating the process pool on the owner's stop instruction** while a batch was in
   flight. They are **owner-stop-induced worker terminations** — not "zero infrastructure
   events", and not spontaneous data corruption. None advanced a scientific seed stream
   (`scientific_stream_advanced: false`, `row_replaced: false`) and none entered the 353
   scientific ledger rows. They must not be conflated with the **12 `sampling_failure`**
   ledger rows, which are a different, scientific outcome.
4. **Quota accounting.** Exactly **17** cells equal quota and **two** exceed it, so **19**
   meet or exceed quota; **five** cells are short. Reporting "19 at quota" without the split
   overstates the balance.
5. **Splittability.** Only `F3_aperture_side_stagger|right|neg5`, with **one** episode, is
   mathematically unable to appear in both train and validation.
   `F3_aperture_side_stagger|right|pos5`, with **two**, splits 1/1.
6. **Encoder-health provenance.** The previously reported encoder statistics came from
   **120 windows drawn from a single real episode**, not the full corpus. Corpus-wide
   statistics are computed in §3 and supersede them.
7. **Corpus size.** The exact corpus is **71,511 HDF5 timesteps** and
   **2,860,440 sensor windows** (71,511 × 40). HDF5 `T` ranges **356 to 627**
   (`episode_steps` 355–626, `T = episode_steps + 1`).

The erratum is bound to: `ledger.jsonl` SHA-256
`ca4adea083d4fd0f25eb2e0dfd39b910c36f877ad1d76309beabc563a63038f6`; the V10.8 close-out raw
and payload hashes; all 141 accepted attempt IDs and HDF5 hashes; and the owner
authorization represented by this task.

**Authoritative training population:** the **141 accepted ledger rows**. Not the stale
`collection.json` smoke record, and not a glob of the rows directory.

---

## 1. Freeze the 141-row source population

Independently verify, from the artifacts alone:

- 353 unique scientific ledger rows; 141 accepted, 212 rejected
- every accepted row is strict-clean
- 141 unique, readable HDF5 files; every HDF5 SHA-256 matches its ledger row
- exactly 40 proximity sensors per episode
- every proximity tensor finite, nonconstant, float32, shaped `(T,4,8,8)`
- `T_min = 356`, `T_max = 627`, `sum(T) = 71,511`
- zero accepted pendant-contact rows
- accepted clearance minimum `0.008272895299859126` m
- no duplicate episode, attempt, task-seed, or output identity

Write an immutable source manifest ordered canonically by **registered V10.8 cell order,
then `attempt_index`, then `attempt_id`** — never by nondeterministic parallel-completion or
ledger arrival order.

Record the underrepresentation prominently, in the manifest itself:

| axis | counts |
|---|---|
| family F0/F1/F2/F3 | 39 / 38 / 38 / **26** |
| side | 75 left / 66 right |
| over-quota cells | 2 |
| short cells | 5 |
| `F3\|right\|neg5` | **1** |
| `F3\|right\|pos5` | **2** |

All 141 accepted episodes are used. No surplus row is dropped, no F3 resampling, no
duplication, no family weights.

---

## 2. Convert into a fresh ACT-format dataset

Destination: `assets/act_style_data/pact_place_v108_141`. No V10.8 source HDF5 and no V5
converted dataset is modified.

V5 conversion semantics are reused; a **new V10.9 adapter** is required because the existing
converter's `recovered_152` path is contract-specific (it reads
`configs/pact_place_v5_recovery.json` and a 152-row key-verification gate).

**Sensor order is preserved exactly**, including `link5_front_*` before `link5_back_*`:

```
sensor-order SHA-256: 2198e29b796ce63f43d8b0db50a92da7d4429895f8571f7d87b655bc265c8fe1
```

Verified: this is *not* alphabetical order. Alphabetical sorting would place
`link5_back_*` before `link5_front_*`. The encoder is weight-shared, but the PACT
transformer assigns positional embeddings by sensor slot, so a re-sorted order would
silently relabel every sensor.

Per converted episode, verify `action (T,8)`, `qpos (T,9)`, wrist RGB `(T,240,320,3)`,
raw proximity `(T,40,4,8,8)`, correct per-sensor extrinsics/intrinsics, source and
destination hashes, semantic identity, and exact timestep preservation. Create-only
conversion and tree-hash manifests.

---

## 3. Generate frozen proximity embeddings

Encoder: `/root/pact_frontend_screen_artifacts/encoder_v1/embedding_encoder_frozen.pt`,
required SHA-256
`6fd2dd037e3236b5b6bf7fce8cb2709ead0cf52adcbbe9cbad1061efc2fe3206`.
Loaded through the sanctioned frozen loader; must report schema
`pact_surface_embedding_encoder_v1`, class `SurfaceEmbeddingEncoder`, feature dim 32.

Run over all **2,860,440** sensor windows; write `(T,40,32)` `proximity_embeddings` into
**only** the fresh converted files. Record corpus-wide finite/noncollapse statistics,
per-dimension standard deviations, the zero-dead-dimension count, the encoder hash,
per-file post-encoding hashes, and the final dataset-tree hash.

---

## 4. Freeze one 113/28 split

One byte-identical split for both arms. Split master seed **2026082901**. Stratified over
the 24 family×side×pose cells, without examining trajectory loss, length, clearance, or any
learned outcome.

1. Assign the sole `F3|right|neg5` row to **training**.
2. For every other nonempty cell, reserve one deterministically hash-ranked row for
   validation → 23 validation rows.
3. Allocate the remaining **five** validation slots by largest remainder across cells,
   capped to leave at least one training row in every cell. Ties broken by
   SHA-256 of `(split seed, cell key)`.
4. Within a cell, select rows by SHA-256 of `(split seed, attempt_id)`.

Therefore `F3|right|pos5` (2 rows) splits 1/1. Assert: exactly 113/28, no overlap, all 141
assigned once, all 24 cells represented in training, 23/24 in validation. The split manifest
stores row identities, the algorithm, the seed, counts by family/side/pose/cell, and a
self-hash.

---

## 5. Freeze and compare both training commands

Fresh root: `/root/pact_place_v108_141_pact_vs_act_chunk100_seed3101/`.

Parameters match the V5 chunk-100 experiment exactly: policy class ACT, seed 3101,
2000 epochs, batch 8, lr 1e-5, KL weight 10, chunk size 100, hidden dim 512, feedforward
3200, 7 encoder / 7 decoder layers, 8 heads, ResNet-18, wrist camera only, state dim 9,
action dim 8, 4 data workers, checkpoint every 200 epochs, no W&B,
**`episode_horizon = 635` because verified `T_max = 627`**.

Both arms share the exact converted dataset, split, normalization statistics, seed,
architecture, optimization budget, and horizon. PACT may differ **only** by its checkpoint
directory and these five flags:

```
--use_proximity
--n_proximity_sensors 40
--prox_tokens_per_sensor 1
--proximity_feature_dim 32
--proximity_encoder_sha256 6fd2dd037e3236b5b6bf7fce8cb2709ead0cf52adcbbe9cbad1061efc2fe3206
```

A parsed flag/value diff runs before training and fails closed on any other difference.

**ACT submodule provenance (verified).** `submodules/act` at `d186ec1` differs from the
original V5 training commit `01751759c49d7237f2b14ff8a16fd3c10ae4c089` by exactly three
**added** evaluation scripts — `eval_pact_place_chunk100_row.py`,
`eval_pact_place_permuted_row.py`, `eval_pact_place_row.py` — 373 insertions, **0
deletions**, clean working tree. No training, model, or loader source changed.

Disk is preflighted against the projected converted dataset plus two checkpoint trees
(reference: V5 = 7.2 GB dataset, 7.5 GB per checkpoint tree). No historical artifact is
deleted. Duplicate or partially populated output directories are refused.

---

## 6. Train and verify

Sequential: **ACT first, then PACT**. Single-shot exploratory runs; neither is tuned on the
other's result.

After each run: preserve the full epoch log and checkpoints; record start/end/runtime and
GPU information; verify 2000 epochs completed; identify best epoch and validation loss; hash
`policy_best.ckpt`, `policy_last.ckpt`, statistics, and run manifest; reload the best
checkpoint with `strict=True`; run a fixed offline batch-inference smoke test; check output
shape `(B,100,8)`, finiteness, and nonconstant actions. For PACT, assert
`input_proj_proximity.weight.shape == (512,32)` and prove proximity was consumed.

**Stop before live evaluation if either checkpoint fails strict reload or the offline
smoke test.**

---

## 7. Build a new V10.9 learned-policy evaluator

The V5/V2 evaluator is neither edited nor silently reused: it hardcodes the V2 scene and
`PactPlaceCorridorV2Sampler`. A separately versioned wrapper/contract is created using
`PactPlaceCorridorV106Sampler` and the three exact certified static scenes
`pact_place_corridor_v10_7_neg5.xml`, `pact_place_corridor_v10_7_center.xml`,
`pact_place_corridor_v10_7_pos5.xml`, bound to the exact scene hashes from V10.7
certification; real V9.5 household clutter; action noise disabled; `num_queries=100`; task
horizon 900; `end_on_success=false`; the existing `PactPlaceContactAudit`; canonical
40-sensor order and frozen encoder hash.

A test asserts each row actually loads its registered pose-specific XML and that no V2 scene
or sampler fallback occurs.

---

## 8. Freeze a paired 40-instance evaluation

Evaluation master seed **2026082902**. 40 held-out physical instances; both arms run on the
identical instances → **80 learned-policy rollouts**.

Balance: every one of the 24 cells appears at least once; 16 cells receive a second
instance; 10 instances per family; 20 left / 20 right; pose totals 14 center, 13 neg5,
13 pos5. The 16 doubled cells are chosen by hash-ranked constrained selection satisfying
those totals, frozen before any rollout.

Evaluation task seeds are asserted **disjoint** from: all 353 V10.8 scientific-attempt seeds
reconstructed from their cell streams; all 141 accepted training rows; the V10.7 pool and
Phase-0 rows; and any evaluation smoke rows. ACT and PACT share the same initial row, task
seed, pendant pose, clutter layout, and initial-state hash. No completed scientific row is
replaced, cherry-picked, or reseeded.

A separate **four-instance / eight-rollout infrastructure smoke** with disjoint seeds runs
first. Smoke *performance* does not gate the full evaluation — the smoke exists only to
check checkpoint loading, scene identity, telemetry, memory, and ETA. If infrastructure is
healthy the full evaluation runs **even if both policies perform poorly**.

---

## 9. Evaluation outputs

Each arm uses its strictly reloaded `policy_best.ckpt`. PACT computes live causal embeddings
with the same frozen encoder and sensor order used during conversion.

Reported for both arms: task success; **collision-free task success (primary endpoint,
matching V5 reporting)**; strict-clean task success; pendant-free task success;
pendant/mounted-fixture contact episodes and frames; clutter-contact episodes;
clutter-stability events; other-environment contact; place-receptacle diagnostic contact;
gripper-close-commanded rate; episode/control-step distribution; failure taxonomy; and
results by family, side, pose, and cell.

Paired PACT−ACT differences for task success and collision-free task success are reported
with a paired estimate in percentage points, a paired 95% interval, discordant pair counts,
and the exact McNemar result where defined.

The historical V5 chunk-100 result (PACT 19/40 vs ACT 13/40 task successes) is quoted for
context only, stated clearly as **contextual, not comparable**, because both the environment
and the training corpus changed.

`PACT_PERMUTED` is **not** run; this task authorizes only ACT and PACT. No superiority claim
is made from validation loss alone, from a single seed, or from a confidence interval
crossing zero.

---

## 10. Final close-out

Return the completed task list; corrected V10.8 facts; the exact 141-row source
verification; split counts and hashes; conversion and embedding hashes/statistics; the exact
ACT/PACT command diff; training time, best epoch/loss, and checkpoint hashes; offline
checkpoint-smoke results; the evaluation manifest and scene hashes; the complete 40-pair
outcome table; aggregate and stratified evaluation results; the contact/failure taxonomy;
artifact paths and SHA-256 values; disk/GPU usage; and all deviations.

Update this plan, `README.md`, and `EVAL.md`. Preserve V5 and V10.4–V10.8 artifacts
byte-for-byte. Afterwards: no additional seed, no hyperparameter tuning, no further
demonstration collection, no additional ablation.


---

## Executed result (2026-08-29)

### Training

| | ACT | PACT |
|---|---|---|
| epochs | 2000/2000 | 2000/2000 |
| wall clock | 72.0 min | 76.0 min |
| best epoch | 1883 | 1841 |
| best validation loss | 0.11186 | 0.11549 |
| strict `load_state_dict` | pass | pass |
| offline smoke `(8,100,8)` | finite, non-constant | finite, non-constant |
| `input_proj_proximity.weight` | n/a | `[512, 32]` |

Proximity consumption is proved causally, not structurally: the same fixed batch
run with real versus zeroed frozen embeddings moves PACT's actions by
**max 1.376, mean 0.219**. A policy ignoring its proximity tokens returns
identical actions.

### Evaluation — 40 paired instances, 80 rollouts, 0 failures, 2.44 h

| endpoint | ACT | PACT | PACT−ACT | 95% CI (pp) | discordant | exact McNemar |
|---|---|---|---|---|---|---|
| task success | 14/40 | 11/40 | −7.5 pp | [−20.2, +5.2] | 2/5 | p = 0.4531 |
| **collision-free task success** | **8/40** | **6/40** | **−5.0 pp** | **[−16.9, +6.9]** | 2/4 | **p = 0.6875** |
| strict-clean task success | 8/40 | 6/40 | −5.0 pp | [−16.9, +6.9] | 2/4 | p = 0.6875 |
| pendant-free task success | 14/40 | 11/40 | −7.5 pp | [−20.2, +5.2] | 2/5 | p = 0.4531 |
| intrusion-panel contact episodes | 10/40 | 8/40 | −5.0 pp | [−18.8, +8.8] | 3/5 | p = 0.7266 |
| clutter-contact episodes | 17/40 | 18/40 | +2.5 pp | [−15.2, +20.1] | 7/6 | p = 1.0000 |

**Zero pendant contact in all 80 rollouts, both arms.** The certified V10.7
asymmetric pendant was never touched by a learned policy, consistent with
V10.8's zero accepted pendant contacts.

Contact volume, not a paired test: PACT logged **43,845** intrusion-panel
contact entries against ACT's **140,424** — a 3.2× reduction — while touching
the panel in two fewer episodes. The episode-level difference is not
significant. Clutter contact ran the other way (123,001 PACT vs 87,688 ACT).

Every episode in both arms ran the full 900 control steps, because
`end_on_success=false`.

### What this does and does not license

It licenses nothing about PACT. The primary endpoint is **null**: the interval
crosses zero, the discordant counts are 2 versus 4, and a single seed with 40
paired instances cannot resolve a difference of this size. The V5 chunk-100
result (PACT 19/40 vs ACT 13/40 task success) reversed sign here, but the
environment and the training corpus both changed, so that comparison is
contextual only and is recorded as such.

It also does not reopen V10.7. Phase-0 remains **failed at 8/24 and permanently
closed**; V10.8 remains an exploratory owner-override collection stopped early
at 141 of 152. Every authorization field on every V10.9 artifact is false.

### Deviations recorded in the close-out

**D1** the converted corpus is 2,854,800 sensor windows, not 2,860,440 — the
latter counts the raw HDF5, and conversion drops each episode's trailing empty
action row. **D2** `episode_horizon=635` is unchanged, but raw `T_max` is 627
while converted `T_max` is 626. **D3** the V5 converter's `_semantic_sha256`
hashes numpy object arrays by pointer, so every `act_semantic_sha256` is
unreproducible; it was dropped as an anchor and preservation proved instead by
re-deriving all seven tensor sets from source for all 141 episodes. **D4** the
split and evaluation manifests are written as text because the immutable writer
injects a key their loaders then include when recomputing the self-hash.
**D5** the split manifest names the pre-embedding tree hash; training verified
the post-embedding one, and both are recorded.

### Defects caught before they could corrupt a result

Four superseded attempts are retained rather than deleted, each with its cause:
the split manifest that `fixed_split_data.load_split_manifest` would have
rejected (caught by loading it through the real loader before training); two
infrastructure-smoke failures, the second of which would have silently restored
the **V2 place scene** because `eval_pact_place_row.main()` reassigns its own
module globals onto the legacy evaluator after the wrapper patches them; and an
analysis whose collision-free cross-check used the V5 *converter's*
demonstration filter rather than the audit's actual
`hazard_bar + other_environment + clutter + mounted_fixture` definition.
