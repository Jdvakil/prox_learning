# Rollout eval for the chunk-1 place policies (ACT@1 vs PACT@1)

## Context

Two policies finished training on the 152 recovered place demonstrations, seed 3101, **chunk size 1
for both arms** (matched — the command diff is exactly the five proximity flags). Offline result:
ACT best validation loss **0.042982 @ epoch 1954**, PACT **0.047391 @ epoch 1773**.

No rollouts have been run. `EVAL.md` correctly claims nothing about task success or collisions.

Two things make a rollout eval worth the window:

1. **Chunk 1 is per-step behaviour cloning over 244–635-step episodes.** The open question is whether
   either policy functions at all. A collapse is decisive and cheap to detect.
2. **The offline number is a generalization artifact, not a modality verdict.** PACT's final train L1
   is *lower* than ACT's (0.057682 vs 0.058775) while its val L1 is *higher* (0.077836 vs 0.067853) —
   a generalization gap of **0.0202 vs 0.0091**, 2.2×. On 122 training episodes, 40 extra sensor
   inputs give more capacity to memorize. Validation loss cannot settle what happens in rollout.

**There is no place eval harness.** This plan builds one.

## THE 32-D FRONT END — read this first

**The policies consume 32-D proximity embeddings, not 3-D surface points.** Verified directly:

| Check | Value |
|---|---|
| PACT@1 checkpoint `model.input_proj_proximity.weight` | **(512, 32)** |
| ACT@1 checkpoint | no proximity tensors |
| Frozen encoder `schema_version` | `pact_surface_embedding_encoder_v1` |
| variant / `policy_feature_dim` | `embedding32_reconstruction_auxiliary_surface` / **32** |
| encoder architecture input | `[32, 8, 8]` — 8-frame causal window × 4 substeps |
| encoder SHA-256 | `6fd2dd037e3236b5b6bf7fce8cb2709ead0cf52adcbbe9cbad1061efc2fe3206` |
| PACT run manifest `policy_config.proximity_feature_dim` | **32** |

`SURFACE_EMBEDDING_DIM = 32` (`surface_proximity_encoder.py:24`).

**The corridor evaluator is on the old 3-D path and cannot load these checkpoints.** It imports
`load_frozen_surface_encoder` (`eval_pact_collision_row.py:49`), calls it at `:234`, and calls
`.predict()` at `:264` which returns 3-D positions. It fails *before* any shape mismatch, because the
loader rejects on schema first:

```python
# surface_proximity_encoder.py:309-316
def load_frozen_surface_encoder(...):
    if payload.get("schema_version") != "pact_surface_encoder_v1":
        raise ValueError("not a pact_surface_encoder_v1 checkpoint")
```

Our encoder is `pact_surface_embedding_encoder_v1`. Loud failure at encoder load, not a silent 3-D
fallback. Good, but it means the corridor evaluator is the wrong parent.

## Build on the frontend-screen evaluator, not the corridor one

`eval_pact_frontend_screen_row.py` **already implements the whole 32-D path** in 218 lines:

| What | Where |
|---|---|
| `load_frozen_surface_embedding_encoder` | `:48` (loader at `surface_proximity_encoder.py:327`, schema guard `:334`) |
| `feature_dim` from the payload, checked against `SURFACE_EMBEDDING_DIM` | `:52-54` |
| cross-check against the checkpoint's `run_manifest.json` | `:55-64` — `run_manifest["policy_config"]["proximity_feature_dim"]` must equal the encoder's |
| `proximity_feature_dim` passed into the model config | `:81` |
| `.policy_features(...)` instead of 3-D `.predict(...)` | `:109` (`SurfaceEmbeddingEncoder.policy_features` at `surface_proximity_encoder.py:298`) |
| 8-frame causal window | `_proximity_history` = `deque(maxlen=8)`, inherited from `eval_pact_collision_row.py:196` |

It reaches this by **subclass + monkey-patch + delegate**, which is the established pattern to follow:

```python
class PactFrontendScreenInferencePolicy(legacy.PactCollisionInferencePolicy)   # :35
class PactFrontendScreenPolicyConfig(legacy.PactCollisionPolicyConfig)         # :203
def main():                                                                    # :207
    legacy.PactCollisionInferencePolicy = PactFrontendScreenInferencePolicy    # :209
    legacy.PactCollisionPolicyConfig    = PactFrontendScreenPolicyConfig       # :210
    legacy.load_eval_manifest           = load_screen_manifest                 # :211
    return legacy.main()                                                       # :214
```

**`eval_pact_place_row.py` subclasses the frontend-screen classes.** The 32-D path is inherited, not
copied, so it stays defined in one place.

## `num_queries` is hardcoded in BOTH evaluators

This is the chunk-1 blocker and it has two sites, not one:

| File | Line | Content |
|---|---|---|
| `eval_pact_collision_row.py` | **210** | `"num_queries": 100,` |
| `eval_pact_frontend_screen_row.py` | **67** | `"num_queries": 100,` — its own `prepare_model` |

Both are followed by a strict load, which is what turns the mismatch into a hard failure rather than
silent corruption:

| File | Line | Content |
|---|---|---|
| `eval_pact_collision_row.py` | **228** | `policy.load_state_dict(state, strict=True)` |
| `eval_pact_frontend_screen_row.py` | **88** | `policy.load_state_dict(state, strict=True)` |

Because the place evaluator inherits from the *frontend-screen* policy, the site that matters is
**`eval_pact_frontend_screen_row.py:67`**, and the fix belongs in the **new place subclass's own
`prepare_model` override** — take `num_queries` from a CLI argument, default 100, pass **1** for these
checkpoints. Do not edit either existing evaluator; both stand behind published results.

At `num_queries=1` the decoder has a single query, `query_embed` is `nn.Embedding(1, hidden_dim)` and
`pos_table` is sized `1+1+1`. A checkpoint trained at 1 cannot load into a model built at 100 under
`strict=True`, and vice versa.

## The place environment

**The config recipe already exists and is known-good.** `scripts/run_pact_place_recovery_datagen.py:156-208`
(`make_recovery_config`) is the path that produced the 152 demos:

```python
config = FrankaSkinPACTCollisionCorridorConfig(output_dir=..., num_workers=1)
config.task_type   = "pick_and_place"
config.task_horizon = 900
config.end_on_success = False
config.task_config   = PickAndPlaceTaskConfig(task_cls=PactPlaceCorridorTask)
config.policy_config = PactPlaceCorridorPolicyConfig()      # EXPERT; eval replaces this
config.task_sampler_config.task_sampler_class = PactPlaceCorridorV2Sampler
config.task_sampler_config.scene_xml_paths    = [pact_place_corridor_v2.xml] * 2
config.robot_config.action_noise_config.enabled = False
```

For eval, keep all of it except `policy_config`, which becomes the place subclass of
`PactFrontendScreenPolicyConfig`.

**Scene confirmed as v2** from a row's own `scene_params`:
`pact_place_environment_version: pact_place_corridor_v2`,
`task_success_criterion: PickAndPlaceTask.supported_released_receptacle_stable`,
`place_receptacle_name: place_receptacle` — matching the training data exactly.

**The contact audit is a drop-in.** `PactPlaceContactAudit`
(`molmo_spaces/tasks/pact_place_contact_audit.py:116`) exposes the same `reset` / `observe(env, step)`
/ `summary()` interface as the shared `PactContactAudit` and provides both keys the evaluator reads —
`contact_class_totals` (`eval_pact_collision_row.py:467`) and `collision_free` (`:724`). Attach it the
same way, via `task._contact_audit_hook` (`:197-198`).

`place_environment_contact_pairs` additionally admits carried-cup-vs-clutter pairs that the shared
`robot_environment_contact_pairs` filter drops. The v2 scene has no clutter, so that path stays empty
here — but using the place audit now means the harness already works for v8b.

**`collision_free` is phase-independent** — `hazard_bar + other_environment + clutter == 0`. It does
not consult the per-phase tables, so the missing expert phase cannot corrupt it.

**Phase defaults to `"other"`** (`pact_place_contact_audit.py:137`), which is *not* in
`PLACEMENT_EXEMPT_TRAVERSAL_PHASES` (`:28`). A learned policy has no expert phase, so `set_phase()` is
never called and every `place_receptacle` contact counts as outside-placement. **That over-counts
rather than under-counts** — report it as a separate diagnostic, never fold it into the endpoint.

## Change list

1. **`prepare_model` override** in the new place subclass of `PactFrontendScreenInferencePolicy` —
   `num_queries` from CLI (**1** here). Inherits the 32-D encoder path, the run-manifest feature-width
   cross-check, and the 8-frame causal window.
2. **`PactPlaceContactAudit`** in place of `PactContactAudit`.
3. **Register `PactPlaceCorridorV2Sampler`** in `task_sampler_class_for`
   (`eval_pact_collision_row.py:87-113`). Keep the allowlist — it exists so a manifest cannot import
   arbitrary code.
4. **Place config overrides** per `make_recovery_config` above.
5. **Place manifest loader**, patched in the way `load_screen_manifest` is at `:211`.

## Manifest

Copy the schema of `configs/pact_contact_endpoint_manifest_v1.json` — `environment_version`,
`master_seed`, `scene_template_house_index`, `scene_template_id`, `sensor_names`,
`sensor_order_sha256`, `rows`, `manifest_sha256`.

Fields the v2 place sampler reads, verified in `enclosure_reach.py`:

```
hazard_present        line 1150   required, bool
intrusion_side        lines 1327, 1340   required, "left" | "right"
panel_x_jitter_m      line 1345   optional, defaults 0.0
panel_face_jitter_m   line 1347   optional, defaults 0.0
```

Plus what the evaluator requires: `episode_id`, `task_seed_u32`, `task_seed_u64`,
`scene_template_house_index`, `max_sampling_retries`, `role`, `role_index`, `row_sha256`.
Clutter fields are V3+ only and must be absent.

**Instances must be held out.** All 152 demonstrations are training data (122 train / 30 val), so
generate rows from a **new master seed** reproducing none of the 152 `task_seed_u64` values — assert
the intersection is empty and record it. Balance `intrusion_side` 50/50.

## Execution

**E0 — build.** New `eval_pact_place_row.py` on the frontend-screen classes. Do not edit
`eval_pact_collision_row.py` or `eval_pact_frontend_screen_row.py`; both stand behind published
results.

**E1 — smoke, and measure.** 2 rollouts per arm. Check the episode completes, the audit summary is
populated, `task_success` is a bool, and **both checkpoints load at `num_queries=1` under
`strict=True`**. Assert the PACT policy's `input_proj_proximity` is `(512, 32)` after load.
**Record wall-clock per rollout and re-derive N before committing.**

The current figure is extrapolated, not measured: the contact-endpoint eval ran **1200 rollouts in
27.57 h on 10 workers** (`full_launcher_receipt.json` 2026-08-01T11:14:54 →
`full_execution_summary.json` 2026-08-02T14:49:18) = **13.8 min per rollout** at ~195-step corridor
episodes. Place episodes are ~480 steps, so **~34 min per rollout** is a projection. Cost is dominated
by proximity rendering — 40 sensors × 4 substeps = 160 depth renders per control step — and ACT gets
no discount, because `n_sensors = 0` (`eval_pact_collision_row.py:207`,
`eval_pact_frontend_screen_row.py:40`) only shrinks the *model* config while the simulator still
renders.

**E2 — run.** Default **N = 20 per arm**, 10 workers (proven; 12 hit the cgroup `pids.max = 3840`
ceiling before). ~2.3 h for both arms at the projected cost. Adjust N from the E1 measurement.

## What this can and cannot establish

**Can:** whether the chunk-1 policies function at all. If both score 0/20 that is decisive, and it
means the chunk-1 pair is uninformative regardless of the val-loss difference — worth knowing before
anyone builds a narrative on +0.004410.

**Cannot:** any comparison between the arms. At N = 20 the interval on a rate difference is roughly
±22 pp against an established contact effect of −8.3 pp at ~100 effective instances. **Nothing will
clear zero, and no PACT-vs-ACT difference may be claimed.**

**Predeclare the collapse floor now, before any rollout:** if either arm's collision-free task success
is **≤ 1/20**, report `CHUNK1_COLLAPSE` and treat *both* arms as uninformative. That is a statement
about chunk size, not proximity, and must not be mined for a modality result.

## Verification

- PACT loaded through `load_frozen_surface_embedding_encoder`; `policy_feature_dim == 32`; the
  run-manifest cross-check passed; `.policy_features()` used, never 3-D `.predict()`.
- Encoder SHA-256 equals `6fd2dd03…`, matching the training manifest.
- `num_queries` came from the CLI and equals **1**; both checkpoints load under `strict=True` with no
  shape error; checkpoint SHA-256 match `EVAL.md` (`cd95d805…` ACT, `4404138b…` PACT).
- Config differs from `make_recovery_config` only in `policy_config`; the diff is printed into the report.
- Scene is `pact_place_corridor_v2.xml` with `PactPlaceCorridorV2Sampler`, matching training.
- `PactPlaceContactAudit` attached; `contact_class_totals` and `collision_free` present in every result.
- Eval `task_seed_u64` values intersect the 152 training seeds in **zero** rows — asserted, recorded.
- `intrusion_side` balanced 50/50; every row reconciles to exactly one result.
- Per-rollout wall-clock recorded; N justified against the E1 measurement, not the projection.
- Report states: single seed, N per arm, no cross-arm claim, temporal ensembling inert at chunk 1
  (`eval_pact_collision_row.py:391-406` keeps chunks while `age < len(value)`; at chunk 1 exactly one
  survives at weight 1.0), and `place_receptacle` over-counted because phase is unavailable.
- `eval_pact_collision_row.py`, `eval_pact_frontend_screen_row.py`, the 152 dataset, both checkpoints,
  and all v5/v6b/v6c artifacts unmodified.

## Constraints

- Do not edit either existing evaluator; subclass and patch, as `eval_pact_frontend_screen_row.py:207-214` does.
- Do not use the 3-D `load_frozen_surface_encoder` path — it rejects this encoder on schema.
- Do not claim a PACT-vs-ACT difference from N = 20.
- Do not run the @100 pair's eval until the @1 eval has been reported.
- Keep the sampler allowlist; register `PactPlaceCorridorV2Sampler` explicitly.
- 10 workers maximum (cgroup `pids.max = 3840`).
- Interpreter `/root/act_retrain_venv/bin/python3`; `MUJOCO_GL=egl`,
  `MLSPACES_ASSETS_DIR=/root/prox_learning/assets`, `PYTHONPATH` → repo `submodules/molmospaces`.
- `pgrep -fc "python.*(eval_pact_place|imitate_episodes)"` — plain `pgrep -fc eval_` self-matches the
  checking shell.
