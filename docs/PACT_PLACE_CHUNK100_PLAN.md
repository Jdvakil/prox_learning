# Chunk-100 place run: train ACT and PACT, evaluate three arms

*Every file path, line number, constant and cost below was read from source and re-verified. Nothing
is projected except where explicitly labelled.*

## Start here

Seven steps, in order. **T2, T3 and T4 need no GPU — run them while T0/T1 train.**

| | Step | Output |
|---|---|---|
| T0/T1 | train ACT@100 then PACT@100 | `/root/pact_place_152_pact_vs_act_chunk100_seed3101/` |
| T2 | fork the manifest contract, build a 40-row manifest | `scripts/pact_place_eval_chunk100_contract.py`, `configs/pact_place_eval_chunk100_manifest.json` |
| T3 | fork the token-plan builder, build the place plan | `scripts/build_pact_place_permuted_token_plan.py` + plan dir |
| T4 | place permuted evaluator | `submodules/act/eval_pact_place_permuted_row.py` |
| T5 | smoke, 2 rollouts × 3 arms | **stop and report if no rollout closes the gripper** |
| T6 | full eval, 3 arms × 40 | 120 rollout results |
| T7 | report | `EVAL.md` in the run root |

**Four hard stops:** stop after T5 if the gripper never closes; never edit a published evaluator,
contract, manifest or the chunk-1 run directory (fork and patch instead); never assert `row_sha256`
equality between the chunk-100 and chunk-1 manifests (see T2); never claim a modality difference if
the shortcut-broken condition fails.

**Three things that will bite you, all verified:** `legacy.ARMS` excludes `PACT_PERMUTED` and argparse
uses `choices=ARMS`; legacy skips the encoder-hash check for that arm; and the manifest contract
hardcodes 20 rows with a 10/10 side split.

## Context

The chunk-1 run collapsed: **0/20 task successes for both arms**, all 40 rollouts ran the full
900-step horizon, and **neither policy ever commanded the gripper closed**. The investigation showed
this is not a harness bug and not a checkpoint bug.

**The harness was correct.** PACT consumed 32-D proximity in all 20 rollouts
(`input_proj_proximity_shape [512, 32]`, `proximity_consumed_for_action: true`), `num_queries=1`,
`PactPlaceContactAudit` attached, images 240×320 RGB matching training, all 40 model-output traces
distinct.

**The policy was correct on-distribution.** Loaded at `num_queries=1` under `strict=True` and run on
its own training observations: `true 0 → −4.85`, `true 255 → 259.14`, arm joints to ~0.01 rad.

**The cause is a gripper-state shortcut.** Input-swap tests showed qpos, not the image, drives the
gripper decision. In the training data:

```
corr(observed qpos[7], commanded action[7])                        = 0.9934
accuracy of "command 255 iff qpos[7] >= 0.382" over 29,051 steps   = 0.9983
fraction of steps that are transitions (the only informative ones) = 0.0017
```

The policy learned to **echo its own gripper state**. Every episode starts at `qpos[7] ≈ 0.003`; it
reads that, commands open, the fingers stay low, it reads it again. **Self-locking, with no way to
bootstrap into the closed regime.**

**Why chunk 100 should break it:** **37.3%** of 100-step windows contain a gripper transition. A
chunk-100 model must emit the open→close transition inside its predicted sequence; it cannot satisfy
that by copying the current state.

**The primary question of this run is not PACT vs ACT.** It is whether chunk 100 produces policies
that grip and place at all.

### Decisions

| | |
|---|---|
| Arms | **ACT, PACT, PACT_PERMUTED** — the permuted arm is eval-only, no extra training |
| Episodes | **40 per arm** — matches the frozen permuted-token contract (`ROWS = 40`) |
| Instances | rows 0–19 are the **same physical episodes** as chunk-1 (see T2 caveat) |
| Budget | **~6.3 h** with overlap (see Schedule) — 0.8 h over the 5.5 h approved |
| Seed | **3101** both arms |
| Encoder | the frozen corridor encoder, unchanged from @1 — cross-task transfer |

`PACT_PERMUTED` matters because every prior study found cross-model PACT-vs-ACT seed-unstable
(+25.0 → −7.5 pp) while PACT-vs-PERMUTED held its sign on both seeds.

## Measured costs

**Training** — from the direct chunk-100 reference
(`/root/pact_contact_endpoint_artifacts/seed3103_training.log`): 2000 epochs in **1:52:03** on a
**199-episode** train split = 24.9 iterations/epoch at batch 8 → **135 ms/iteration**. The place split
has **122** train episodes → 15.25 iterations/epoch → **~69 min per arm**, ~2.3 h for both.

**Eval** — chunk-1 actuals: **40 rollouts in 68.5 min on 10 workers** (00:39:27 → 01:47:56),
mean 17.1 min (ACT) / 17.3 min (PACT) per rollout. 3 arms × 40 = 120 rollouts → **~3.4 h**.

**Caveat to re-measure at T5:** chunk 100 runs a 100-query decoder and averages up to 100 pending
chunks per step versus exactly one at chunk 1, so per-rollout cost may exceed 17 min. Rendering
(40 sensors × 4 substeps = 160 depth renders per control step) should still dominate. Re-derive the
schedule from the smoke timings before committing to the full run.

`end_on_success` stays **False**, as in the demos and the chunk-1 eval, so every rollout runs 900
steps. Do not change it — it would break comparability.

### Schedule — and an honest overrun

Summed sequentially the work is **7.3 h**, which exceeds the approved 5.5 h. T2, T3 and T4 need no
GPU, so run them while training occupies it:

```
t=0.0   start T0/T1 training (GPU, 2.3 h)  ─┐
t=0.0   T2 manifest + forked contract       │  no GPU
t=0.3   T3 place token plan                 │  -> overlapped, saves 1.0 h
t=0.7   T4 permuted evaluator              ─┘
t=2.3   T5 smoke, 6 rollouts (0.3 h)  <- re-measure per-rollout cost here
t=2.6   T6 full eval, 120 rollouts (3.4 h)
t=6.0   T7 report (0.25 h)
        ------------------------------------
        ~6.3 h
```

**Decision: proceed at full scope and accept the ~0.8 h overrun.** Do not stop to ask. All three arms
and the 20 shared instances are worth more than the 48 minutes.

**Runtime cut rule, if you fall behind.** Evaluate this once, at the end of T5, using the smoke's
measured per-rollout cost:

```
projected_T6_hours = 120 * measured_minutes_per_rollout / 10 / 60
if elapsed_since_start + projected_T6_hours > 7.5:
    drop PACT_PERMUTED from T6, run ACT and PACT at 40 each, and say so in EVAL.md
```

Drop the permuted arm rather than shrinking N: it can be added later on the same checkpoints, the same
manifest and the same token plan, whereas a smaller N cannot be extended without re-running everything
at a different `INSTANCE_COUNT`. **Never cut the smoke step.**

## What exists — reuse, do not rebuild

| Purpose | File |
|---|---|
| training command | `/root/pact_place_152_pact_vs_act_seed3101/{act,pact}_seed3101/run_manifest.json` |
| place evaluator | `submodules/act/eval_pact_place_row.py` |
| permuted pattern | `submodules/act/eval_pact_valid_ablation_row.py:72-147` |
| token-plan builder | `scripts/build_pact_permuted_token_plan.py` |
| manifest contract | `scripts/pact_place_eval_chunk1_contract.py` |
| chunk-1 manifest | `configs/pact_place_eval_chunk1_manifest.json` |
| dataset | `assets/act_style_data/pact_place_corridor_v2_recovered_152/` |

**Proximity embeddings already exist** — `observations/proximity_embeddings` at `(T, 40, 32)` in every
converted episode. No encoding pass is needed.

**How `eval_pact_place_row.py` actually works** (verified): `PactPlaceInferencePolicy` subclasses
`frontend.PactFrontendScreenInferencePolicy` (`:47`), takes `--num-queries` as a required pre-parsed
CLI arg (`:168`), and `main()` (`:165-181`) **replaces** rather than extends:

```python
legacy.PactCollisionInferencePolicy        = PactPlaceInferencePolicy
legacy.PactCollisionPolicyConfig           = policy_config_factory
legacy.FrankaSkinPACTCollisionCorridorConfig = PactPlaceEvalConfig
legacy.PactContactAudit                    = PactPlaceContactAudit
legacy.load_eval_manifest                  = load_manifest
legacy.retry_seed_for                      = retry_seed
legacy.task_sampler_class_for              = task_sampler_class_for   # replaced, not an allowlist edit
```

## T0 / T1 — train, ACT first

Copy the @1 command verbatim; change **only** `--chunk_size 1` → `--chunk_size 100` and `--ckpt_dir`.
Everything else stays fixed: seed 3101, 2000 epochs, batch 8, lr 1e-5, kl_weight 10, hidden 512,
ff 3200, 7/7 layers, wrist camera, `--episode_horizon 635`, state 9 / action 8, `--num_workers 4`,
`--ckpt_every 200`, same split and dataset manifests, both `expect_*_sha256`.

New root `/root/pact_place_152_pact_vs_act_chunk100_seed3101/`. **Do not write into the chunk-1
directory.** Record start/end timestamps.

These are the exact commands, derived from the chunk-1 run manifests. Run from
`submodules/act/` with the interpreter and env vars in Constraints. The token-level diff against the
chunk-1 commands is exactly `--chunk_size 1 -> 100` and the `--ckpt_dir` path — verified.

```
# ACT
python imitate_episodes.py --task_name obstacle_baseline \
  --ckpt_dir /root/pact_place_152_pact_vs_act_chunk100_seed3101/act_seed3101 --exact_ckpt_dir \
  --policy_class ACT --batch_size 8 --seed 3101 --num_epochs 2000 --lr 1e-5 --kl_weight 10 \
  --chunk_size 100 --hidden_dim 512 --dim_feedforward 3200 --enc_layers 7 --dec_layers 7 \
  --camera_names wrist_camera \
  --dataset_dir  /root/prox_learning_pact_remediation/assets/act_style_data/pact_place_corridor_v2_recovered_152 \
  --split_manifest   /root/prox_learning_pact_remediation/diagnostics_output/pact_place_152_pact_vs_act/split_manifest.json \
  --dataset_manifest /root/prox_learning_pact_remediation/diagnostics_output/pact_place_152_pact_vs_act/conversion_manifest_encoded.json \
  --expect_split_sha256        bd3b246da6d140f8b4a493789876cad8fe8c7a0e51fa208b104a4ce81041b823 \
  --expect_dataset_tree_sha256 b16a5a0bd221d786f54fd9f28e00d493d01316ed47d9e909c1a915d37b13e6f1 \
  --episode_horizon 635 --state_dim 9 --action_dim 8 --num_workers 4 --ckpt_every 200 --no_wandb

# PACT = the same, with --ckpt_dir .../pact_seed3101 and these five flags appended:
  --use_proximity --n_proximity_sensors 40 --prox_tokens_per_sensor 1 \
  --proximity_feature_dim 32 \
  --proximity_encoder_sha256 6fd2dd037e3236b5b6bf7fce8cb2709ead0cf52adcbbe9cbad1061efc2fe3206
```

The PACT-vs-ACT token diff is exactly the `--ckpt_dir` path plus those five flags — verified against
the chunk-1 manifests.

## T2 — 40-row manifest, and a forked contract

**The contract must be forked too.** `scripts/pact_place_eval_chunk1_contract.py` hardcodes
`INSTANCE_COUNT = 20` (`:18`) and `validate_manifest` rejects anything else: `len(rows) != INSTANCE_COUNT`
(`:145`), side balance exactly `{"left": 10, "right": 10}` (`:147`), `role_index` contiguous over
`range(INSTANCE_COUNT)` (`:149`), and `role_counts`/`total_candidates` equal to `INSTANCE_COUNT` (`:136-137`).

Fork to `scripts/pact_place_eval_chunk100_contract.py` with `INSTANCE_COUNT = 40`,
`SCHEMA_VERSION = "pact_place_eval_chunk100_manifest_v1"`, `ROLE = "place_chunk100_eval"`, and side
balance `{"left": 20, "right": 20}`. **Keep `MASTER_SEED = 2026082101` and all four stream IDs.**

### The caveat that corrects an earlier draft

**Rows 0–19 cannot be byte-identical, and must not be asserted as such.** `row_sha256` is
`sha256_payload(row)` over a row containing `schema_version` and `role` (`:76-96`), and
`episode_id_for()` (`:59-63`) hashes `SCHEMA_VERSION` into its preimage. Changing either field changes
both values for every row.

What **is** preserved, verified by reproducing it against the chunk-1 manifest:

```
task_seed_u32/u64   = derive_seed(candidate_index, TASK_STREAM_ID)       -> reproduces exactly
panel_x/face_jitter = derive_seed(candidate_index, GEOMETRY_STREAM_ID)   -> reproduces exactly
intrusion_side      = shuffle(["left"]*10+["right"]*10, derive_seed(0, SIDE_STREAM_ID))  -> reproduces exactly
```

None of the three depends on `SCHEMA_VERSION` or `ROLE`. So rows 0–19 are the **same physical
episodes**; only bookkeeping identifiers differ.

**Side construction must be blockwise, not a 40-element shuffle.** A single shuffle over
`["left"]*20 + ["right"]*20` would re-order the first 20 positions and change their sides. Build:

- rows 0–19: `shuffle(["left"]*10 + ["right"]*10, seed=derive_seed(0,  SIDE_STREAM_ID))` — the chunk-1 call, unchanged
- rows 20–39: `shuffle(["left"]*10 + ["right"]*10, seed=derive_seed(20, SIDE_STREAM_ID))`

giving 20/20 overall.

**Assert on the physical instance**, not the hashes: for `candidate_index` 0–19 the new manifest's
`task_seed_u64`, `panel_x_jitter_m`, `panel_face_jitter_m` and `intrusion_side` must equal the chunk-1
manifest's. Record that check.

Also assert all 40 `task_seed_u64` are disjoint from the 152 training seeds — the contract already
enforces this two ways (`:163-164`, `:166-169`).

## T3 — place permuted-token plan

**Fork** `scripts/build_pact_permuted_token_plan.py` to
`scripts/build_pact_place_permuted_token_plan.py`. Do not edit the original; it stands behind the
published valid-ablation result.

One substantive change, at `:135-136`:

```python
if len(train_indices) != 199:
    raise ValueError("valid ablation requires the frozen 199-episode train split")
```

That 199 is the corridor split — confirmed, `full_act_split_encoded_v2.json` has exactly 199 train.
The place split has **122**. Parameterize the expected count and assert it against the split manifest
rather than a literal.

Everything else already matches and needs no change: `ROWS = 40`, `MAX_CONTROL_STEPS = 900`,
`SENSORS = 40`, `FEATURE_DIM = 32` (`:20-23`); it reads `observations/proximity_embeddings` (`:144`),
which the place dataset has; and `load_split` (`:57-63`) only verifies the manifest self-hash, so the
place split's `hybrid_obstacle_canonical_split_v2` schema passes.

Sampling is from the **train** partition only, so tokens stay distribution-matched to place data.
122 train episodes give **58,602** frames — ample for 40 rows × 900 frames with within-row
selection without replacement and cross-row reuse allowed.

Output tensor must be `(40, 900, 40, 32)` float32, ~184 MB. The evaluator enforces the full contract
at `eval_pact_valid_ablation_row.py:39-67`.

## T4 — place permuted evaluator

New `submodules/act/eval_pact_place_permuted_row.py`. Subclass **the place policy**
(`eval_pact_place_row.PactPlaceInferencePolicy`) with the four overrides from
`eval_pact_valid_ablation_row.py:77-122`:

- `prepare_model` — set `self.pc.arm = "PACT"` around `super()`, restore in `finally`. **Required**:
  `PactPlaceInferencePolicy.prepare_model` gates the 32-D encoder on
  `if self.pc.arm in ("PACT", "PACT_ZERO")` (`eval_pact_place_row.py:57`), which excludes
  `PACT_PERMUTED`; without the swap it would take the ACT path and fail to load the checkpoint.
- `_surface_positions` — return `TOKEN_FRAMES[self._step]` instead of live embeddings
- `inference_model` — same arm swap
- `get_info` — `arm: PACT_PERMUTED`, `live_proximity_aligned_with_action: False`, token-plan hash,
  frames consumed

`main()` must pre-parse `--num-queries`, `--token-plan-manifest`, `--token-plan-row`, call
`load_token_plan(...)`, then apply **both** patch sets — the seven place patches listed above **plus**:

```python
legacy.ARMS = (*legacy.ARMS, "PACT_PERMUTED")
```

**This is mandatory.** `legacy.ARMS = ("ACT", "PACT", "PACT_ZERO")` (`eval_pact_collision_row.py:63`)
and argparse uses `choices=ARMS` (`:487`), so `--arm PACT_PERMUTED` is rejected without it.

### Encoder hash is not verified for this arm

`eval_pact_collision_row.py:544-548` guards the encoder path and its SHA-256 with
`if args.arm in ("PACT", "PACT_ZERO")` — **`PACT_PERMUTED` skips both checks.** The permuted run must
still pass `--surface-encoder` and `--surface-encoder-sha256` (the config stores the path regardless,
`:566-568`), and **the new module must verify the hash itself**, since legacy will not.

## T5 — smoke

**2 rollouts per arm, all three arms.** Assert before committing to the full run:

- both checkpoints load at **`--num-queries 100`** under `strict=True`
- PACT and PACT_PERMUTED report `input_proj_proximity_shape [512, 32]` and
  `proximity_consumed_for_action: true`
- PACT_PERMUTED reports a non-null `token_plan_sha256`, `live_proximity_aligned_with_action: False`,
  and its encoder SHA-256 verified by the new module
- **`gripper_close_commanded` is true in at least one smoke rollout.** This is the run's whole point;
  if it is false in all six, **stop and report** rather than spending 3.4 h.
- record per-rollout wall-clock and re-derive the schedule from it

## T6 — full eval

3 arms × 40 = **120 rollouts**, 10 workers. Ten workers is proven; the pid ceiling
(`pids.max = 3840`, 382 in use) is not binding, but the single A10 is the likely bottleneck, so do not
raise the worker count without measuring throughput first.

## T7 — report

`EVAL.md` in the run root. Per arm: task success, collision-free task success,
`gripper_close_commanded` count, contact episodes by class, control-step counts, wall-clock.

**Primary endpoint:** `collision_free_task_success`. **Decision-bearing contrast:**
`PACT − PACT_PERMUTED`.

**Freeze before the first rollout:**

- **Shortcut broken** iff `gripper_close_commanded` is true in **≥ 50%** of episodes for both trained
  arms. A chunk-size result, not a modality result.
- **Collapse floor:** if either trained arm scores **≤ 2/40** collision-free task success, report
  `CHUNK100_COLLAPSE` and treat all arms as uninformative for modality.
- **No cross-arm claim** unless the shortcut-broken condition holds. At N = 40 the interval on a rate
  difference is roughly ±15 pp; expect nothing to clear zero. Report `PACT − PACT_PERMUTED` with its
  interval and label it directional.

**State in EVAL.md:** single seed 3101; N = 40; cross-task encoder transfer; rows 0–19 are the same
physical episodes as chunk-1 but carry different `episode_id`/`row_sha256` because those hash the
schema version and role; `place_receptacle` contacts over-counted because a learned policy exposes no
expert phase, so the audit stays at `"other"`.

**Compare @100 to @1 on the shared 20 instances** — matched on `task_seed_u64`, jitters and side. That
is the cleanest available test of the gripper-shortcut mechanism.

## Verification

- Training commands differ from the chunk-1 pair **only** by `--chunk_size` and `--ckpt_dir`; paste
  the diff. Both arms share split hash, dataset tree hash and seed; PACT differs from ACT by exactly
  the five proximity flags.
- Manifest: 40 rows, 20/20 sides, `role_index` contiguous 0–39, zero seed intersection with training.
  For candidate_index 0–19, `task_seed_u64` + both jitters + `intrusion_side` equal the chunk-1
  manifest — **assert these, not `row_sha256`.**
- Token plan validates against the frozen contract; tensor `(40, 900, 40, 32)` float32; built from the
  **122-episode place train split**.
- `num_queries == 100` in every result; both checkpoints loaded `strict=True`.
- PACT and PACT_PERMUTED both record feature dim 32 and `[512, 32]`; PACT_PERMUTED additionally
  records the token-plan hash, `live_proximity_aligned_with_action: False`, and a self-verified
  encoder SHA-256.
- 120/120 rollouts complete, each row reconciling to exactly one result.
- Unmodified, verified by hash: chunk-1 run directory and both chunk-1 checkpoints,
  `eval_pact_collision_row.py`, `eval_pact_frontend_screen_row.py`, `eval_pact_valid_ablation_row.py`,
  `eval_pact_place_row.py`, `scripts/build_pact_permuted_token_plan.py`,
  `scripts/pact_place_eval_chunk1_contract.py`, `configs/pact_place_eval_chunk1_manifest.json`,
  and the 152 dataset.

## Constraints

- Do not edit the chunk-1 run directory, the published evaluators, the chunk-1 contract or manifest,
  or the original token-plan builder. Fork, subclass and patch.
- Do not change `end_on_success`, `task_horizon`, the encoder, the split, or the seed.
- Do not raise worker count above 10 without measuring throughput.
- Do not claim a modality difference if the shortcut-broken condition fails.
- **Stop after T5** if no smoke rollout commands the gripper closed.
- Interpreter `/root/act_retrain_venv/bin/python3`; `MUJOCO_GL=egl`,
  `MLSPACES_ASSETS_DIR=/root/prox_learning/assets`, `PYTHONPATH` → repo `submodules/molmospaces`.
  Work in `/root/prox_learning_pact_remediation`. A10 23 GB free; 128 cores; 327 GB RAM free.
- `pgrep -fc "python.*(eval_pact_place|imitate_episodes)"` — plain `pgrep -fc eval_` self-matches the
  checking shell.
