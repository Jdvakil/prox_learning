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

**There is no place eval harness.** Everything under `submodules/act/eval_pact_*_row.py` binds to the
collision corridor. This plan builds one.

## Verified findings — the harness is a fork, not a rewrite

Everything below was read from source, not assumed.

**The config recipe already exists.** `scripts/run_pact_place_recovery_datagen.py:156-208`
(`make_recovery_config`) builds the place environment from the corridor config by overriding six
fields. That is the datagen path that produced the 152 demos, so it is known-good:

```python
config = FrankaSkinPACTCollisionCorridorConfig(output_dir=..., num_workers=1)
config.task_type   = "pick_and_place"
config.task_horizon = 900
config.end_on_success = False
config.task_config   = PickAndPlaceTaskConfig(task_cls=PactPlaceCorridorTask)
config.policy_config = PactPlaceCorridorPolicyConfig()      # <-- EXPERT; eval replaces this
config.task_sampler_config.task_sampler_class = PactPlaceCorridorV2Sampler
config.task_sampler_config.scene_xml_paths    = [pact_place_corridor_v2.xml] * 2
config.robot_config.action_noise_config.enabled = False
```

For eval, keep all of it except `policy_config`, which becomes `PactCollisionPolicyConfig(...)` —
the ACT-policy wrapper the corridor evaluator already uses
(`eval_pact_collision_row.py:557-572`).

**The scene is confirmed v2.** A row's `scene_params` reads
`pact_place_environment_version: pact_place_corridor_v2`,
`task_success_criterion: PickAndPlaceTask.supported_released_receptacle_stable`,
`place_receptacle_name: place_receptacle`. So `PactPlaceCorridorV2Sampler` and
`pact_place_corridor_v2.xml`, matching the training data exactly.

**The contact audit is a drop-in.** `PactPlaceContactAudit`
(`molmo_spaces/tasks/pact_place_contact_audit.py:116`) exposes the same `reset` / `observe(env, step)`
/ `summary()` interface as the shared `PactContactAudit`, and its `summary()` provides both keys the
evaluator reads — `contact_class_totals` (line 467) and `collision_free` (line 724). It is attached
the same way, through `task._contact_audit_hook` (`eval_pact_collision_row.py:197-198`).

It additionally records `clutter` and `place_receptacle` classes, and
`place_environment_contact_pairs` adds carried-cup-vs-clutter pairs that the shared
`robot_environment_contact_pairs` filter drops. The v2 scene has no clutter, so that path stays
empty here — but using the place audit now means the harness already works for v8b later.

**`collision_free` is phase-independent.** It is `hazard_bar + other_environment + clutter == 0`
(summary, line ~185). It does **not** consult the per-phase tables, so the missing expert phase
during eval cannot corrupt it.

**Phase defaults to `"other"`** (`reset()`, line 137), and `"other"` is *not* in
`PLACEMENT_EXEMPT_TRAVERSAL_PHASES` (line 28). A learned policy has no expert phase to report, so
`set_phase()` is never called and every `place_receptacle` contact is counted as outside-placement.
**That over-counts rather than under-counts**, which is the safe direction — report it as a separate
diagnostic, never fold it into the primary endpoint.

**Three things must change in the forked evaluator:**

| Line | Now | Change |
|---|---|---|
| `eval_pact_collision_row.py:210` | `"num_queries": 100` hardcoded | take from a CLI arg; a chunk-1 checkpoint fails to load otherwise (shape mismatch on `query_embed` and `pos_table`) |
| `:197` | `PactContactAudit()` | `PactPlaceContactAudit()` |
| `:87-113` | allowlist of four `PactCollisionCorridor*` samplers | add `PactPlaceCorridorV2Sampler`; keep the allowlist — it exists so a manifest cannot import arbitrary code |

**Temporal ensembling degenerates at chunk 1, and that is expected.** The loop
(`:391-406`) keeps chunks while `self._step - start < len(value)` and weights them `exp(-0.01·age)`.
At chunk 1, `len(value) == 1`, so exactly one value survives at weight 1.0. No code change needed;
record it in the report so nobody reads the passthrough as a bug.

## Manifest

Copy the schema of `configs/pact_contact_endpoint_manifest_v1.json` — top level
`environment_version`, `master_seed`, `scene_template_house_index`, `scene_template_id`,
`sensor_names`, `sensor_order_sha256`, `rows`, `manifest_sha256`.

Fields the v2 place sampler actually reads, verified in `enclosure_reach.py`:

```
hazard_present        line 1150   required, bool
intrusion_side        lines 1327, 1340   required, "left" | "right"
panel_x_jitter_m      line 1345   optional, defaults 0.0
panel_face_jitter_m   line 1347   optional, defaults 0.0
```

Plus what the evaluator itself requires: `episode_id`, `task_seed_u32`, `task_seed_u64`,
`scene_template_house_index`, `max_sampling_retries`, `role`, `role_index`, `row_sha256`.
Clutter fields are V3+ only and must be absent.

**Instances must be held out.** All 152 demonstrations are training data (122 train / 30 val), so
generate rows from a **new master seed** that reproduces none of the 152 `task_seed_u64` values —
assert the intersection is empty and record it. Balance `intrusion_side` 50/50.

## Execution

**Order matters — the chunk-1 arms are the ones that may collapse, and that is the question.**

**E0 — build.** Fork `eval_pact_collision_row.py` to `eval_pact_place_row.py` with the three changes
above. Do not edit the corridor evaluator or any `eval_pact_*` file that produced a published result.

**E1 — smoke, and measure.** Run **2 rollouts per arm** and check: the episode completes, the audit
summary is populated, `task_success` is a bool, the checkpoint loads at `num_queries=1`.
**Record wall-clock per rollout and re-derive N from it before committing.**

The current estimate is extrapolated, not measured: the contact-endpoint eval ran **1200 rollouts in
27.57 h on 10 workers** (`full_launcher_receipt.json` 2026-08-01T11:14:54 →
`full_execution_summary.json` 2026-08-02T14:49:18) = **13.8 min per rollout** at ~195-step corridor
episodes. Place episodes are ~480 steps, so **~34 min per rollout** is a projection. Cost is
dominated by proximity rendering — 40 sensors × 4 substeps = 160 depth renders per control step —
and ACT gets no discount, because `n_sensors = 0` at line 207 only shrinks the *model* config while
the simulator still renders.

**E2 — run.** Default **N = 20 per arm**, 10 workers (proven; 12 workers hit the cgroup
`pids.max = 3840` ceiling before). At 34 min/rollout that is ~2.3 h for both arms. Adjust N from the
E1 measurement.

## What this can and cannot establish

**Can:** whether the chunk-1 policies function at all. If both score 0/20, that is decisive, and it
means the chunk-1 pair is uninformative regardless of the val-loss difference — a fact worth having
before anyone builds a narrative on +0.004410.

**Cannot:** any comparison between the arms. At N = 20 the interval on a rate difference is roughly
±22 pp. The established contact effect was −8.3 pp at ~100 effective instances. **Nothing here will
clear zero, and no PACT-vs-ACT difference may be claimed from it.**

**Predeclare the collapse floor now, before any rollout:** if either arm's collision-free task
success is **≤ 1/20**, the run is reported as `CHUNK1_COLLAPSE` and *both* arms are treated as
uninformative. That outcome is a statement about chunk size, not about proximity, and it must not be
mined for a modality result.

## Verification

- Config differs from `make_recovery_config` only in `policy_config`; the diff is printed and pasted
  into the report.
- Scene is `pact_place_corridor_v2.xml` with `PactPlaceCorridorV2Sampler`, matching training.
- `PactPlaceContactAudit` attached; `contact_class_totals` and `collision_free` both present in every
  result.
- `num_queries` came from the CLI and equals **1**; both checkpoints load with no shape error and
  their SHA-256 match `EVAL.md` (`cd95d805…` ACT, `4404138b…` PACT).
- Eval `task_seed_u64` values intersect the 152 training seeds in **zero** rows — asserted, recorded.
- `intrusion_side` balanced 50/50; every row reconciles to exactly one result.
- Per-rollout wall-clock recorded, and N justified against the E1 measurement rather than the
  projection.
- Report states: single seed, N per arm, no cross-arm claim, temporal ensembling inert at chunk 1,
  `place_receptacle` contacts over-counted because phase is unavailable.
- Corridor evaluators, the 152 dataset, both checkpoints, and all v5/v6b/v6c artifacts unmodified.

## Constraints

- Do not edit `eval_pact_collision_row.py` or any evaluator behind a published result.
- Do not claim a PACT-vs-ACT difference from N = 20.
- Do not run the @100 pair's eval until the @1 eval has been reported.
- Keep the sampler allowlist; register `PactPlaceCorridorV2Sampler` explicitly.
- 10 workers maximum (cgroup `pids.max = 3840`).
- Interpreter `/root/act_retrain_venv/bin/python3`; `MUJOCO_GL=egl`,
  `MLSPACES_ASSETS_DIR=/root/prox_learning/assets`, `PYTHONPATH` → repo `submodules/molmospaces`.
- `pgrep -fc "python.*(eval_pact_place|imitate_episodes)"` — plain `pgrep -fc eval_` self-matches the
  checking shell.
