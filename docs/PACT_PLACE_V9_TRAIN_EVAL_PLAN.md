# v9 part 2 of 2: collect, train, and evaluate PACT vs ACT with clutter

**Scope: V3 through V7 — collection, conversion, training, evaluation, analysis.**

**Precondition: the Phase-0 gate in
[`docs/PACT_PLACE_V9_ENVIRONMENT_PLAN.md`](PACT_PLACE_V9_ENVIRONMENT_PLAN.md) has passed at
>= 20/24 and has been approved.** That gate is a separate authorization. If it has not passed, stop
here. Do not collect against an ungated environment; the v5 collection already produced 152
untrainable episodes that way.

*You may be picking this up cold. Every number below was read from source or from a recorded
artifact.*

## Context

40 skin proximity sensors on links 1-6, each an 8x8 depth patch through a frozen encoder to a
**32-D** embedding per sensor. `PACT_PERMUTED` is the valid ablation; `PACT_ZERO` is invalid,
because an all-zero skin is out of distribution.

The question this run answers: **does proximity make the policy safer, not just better at the
task?** Part 1 now uses staggered route-blocker families and a privileged collision-free expert.
Before this part may start, Part 1 must separately prove that the blocker causes non-degenerate,
phase-aligned proximity in the real 40-sensor observation path. The expert's access to geometry is
not evidence that PACT receives useful signal; the raw causal-proximity gate and the later
`PACT_PERMUTED` arm provide that evidence.

**The comparison to beat**, from the no-clutter chunk-100 run, one seed, N=40 per arm
(`/root/pact_place_chunk100_eval_seed3101/EVAL.md`):

| arm | task success | collision-free task success | hazard-bar contact |
|---|---:|---:|---:|
| ACT | 13/40 | 13/40 | 13 |
| PACT | 19/40 | 16/40 | 12 |
| PACT_PERMUTED | 7/40 | 6/40 | 8 |

PACT - PACT_PERMUTED was +25.0 pp CFTS, so proximity carries real signal. But PACT was better at
the *task*, not *safer*: 12 vs 13 contacts. The reach corridor's -17.9 pp contact|failure advantage
has no place-task counterpart yet, and the place CI of [-28, +29] pp contains it. That is a power
problem as much as a design one -- roughly 111 failures per arm are needed, against the 21-27
available at N=40.

Hence 3 seeds x 100 episodes per arm here.

## V3 -- collection, 152 episodes

Match the existing dataset size so the no-clutter result stays comparable.

**Use the datagen pipeline. Never the screen harness.**
`scripts/run_pact_place_collection.py:33` imports `run_row` from `run_pact_place_expert_screen`,
whose `_make_config` truncates `expert_rollout_sensor_polling` to `["qpos", "tcp_pose"]` **and**
sets `proximity_sensor_period_ms = 0.0`. Both reductions are silent. That is how 152 v5 episodes
were collected, passed every config-level check, and turned out to contain no proximity, no RGB and
no actions.

Model the producer on `scripts/run_pact_place_recovery_datagen.py`, which is the proven path. Keep
the datagen default `proximity_sensor_period_ms = 16.6667`; with `sim_dt_ms = 2.0` and
`policy_dt_ms = 66.0` that yields `max_substeps = 4`, and proximity lands as `(T, 4, 8, 8)`. Any
other rank makes `convert_pact_collision_to_act.extract_proximity` raise **after** the compute is
spent.

Two defects already found on this path, both fixed in the recovery producer -- carry the fixes:

- `save_utils.py:374` calls a bare `json.dumps` on `obs_scene`, and the place sampler puts
  `cam_visible` into `scene_params` as a `numpy.bool_`. Coerce with `_jsonable` before
  `prepare_episode_for_saving`; do not patch the shared submodule.
- staging cleanup must be `shutil.rmtree`, not `rmdir`, or a partially published row leaves MP4s
  behind that the retry then refuses to overwrite.

Runner hygiene: pin `OMP`/`MKL`/`OPENBLAS`/`NUMEXPR`/`VECLIB` to 1 **in the parent before the pool
is built**, so workers inherit capped pools from first import. Measured effect: 319 tasks per worker
down to 3, at unchanged 101% CPU. Preflight the cgroup with `_pids_headroom()`; `pids.max` is 3840
and twelve unpinned workers exceed it before any rollout begins. If the parent is killed, kill the
orphaned workers by PID or the ceiling stays pinned.

Expect ~2.5-3.5 h at 12 workers and ~37 MB per episode (~5.6 GB). Clutter and free bodies will make
it slower than the 2h26m the no-clutter recovery took.

## V4 -- conversion and the file-level gate

Run `scripts/verify_pact_place_recovery_keys.py`. It **opens each produced file** rather than
checking the config, which is the whole point -- the v5 collection config was internally consistent,
self-hashed, and wrong. Per episode:

- `trajectory.h5` exists and opens, with exactly one `traj_*` group
- `actions/joint_pos` decodes to a contiguous valid prefix of length T > 0, and
  `actions/commanded_action` is non-empty over that prefix
- `obs/agent/qpos` and `qvel` both decode to `(T, 9)`
- `obs/proximity` holds exactly the 40 contract sensors, each `(T, 4, 8, 8)` float32
- proximity is finite everywhere **and not constant** -- an all-zero skin passes every shape check
  and teaches nothing

Then convert with `scripts/convert_pact_collision_to_act.py` into
`assets/act_style_data/pact_place_corridor_v9_<n>/`, alongside the existing
`pact_place_corridor_v2_recovered_152`. Do not overwrite that tree; it backs every prior result.

Record the split manifest and the source-collection tree hash. **Stop and report** if any episode
fails a key check -- do not substitute, pad, or drop it silently.

## V5 -- training, 6 runs

3 seeds x {ACT, PACT}. Entry point `submodules/act/imitate_episodes.py`.

Held fixed across all six: `--num_epochs 2000`, `--chunk_size 100`, batch 8, hidden 512,
dim_feedforward 3200, resnet18 backbone, 7 encoder / 7 decoder layers, lr 1e-5, kl_weight 10,
`camera_names = ["wrist_camera"]`.

PACT differs from ACT by its checkpoint directory and exactly **five** proximity flags:

```
--use_proximity
--n_proximity_sensors 40
--prox_tokens_per_sensor 1
--proximity_feature_dim 32
--proximity_encoder_sha256 6fd2dd037e3236b5b6bf7fce8cb2709ead0cf52adcbbe9cbad1061efc2fe3206
```

`proximity_feature_dim` is **32**, not 3. The 3-D value is the legacy `SurfaceProximityEncoder`;
this study uses `SurfaceEmbeddingEncoder` (schema `pact_surface_embedding_encoder_v1`), loaded by
`load_frozen_surface_embedding_encoder`. Mixing them raises at load, which is the intended
behaviour -- do not work around it.

Write a `run_manifest.json` per run mirroring the existing ones, and diff each against
`/root/pact_place_152_pact_vs_act_chunk100_seed3101/*/run_manifest.json`: the only intended
differences are seed, checkpoint dir, and dataset dir.

Measured cost on the A10 (23 GB): 76 min ACT, 78 min PACT. Six runs ~8 h sequential.

## V6 -- evaluation, 600 rollouts

3 seeds x 100 episodes x {ACT, PACT}.

**A new evaluator variant is required.** `eval_pact_place_row.py:38-43` hardcodes
`PLACE_SCENE = pact_place_corridor_v2.xml` and imports `PactPlaceCorridorV2Sampler`; both must
point at the v9 scene and sampler. Follow `eval_pact_place_chunk100_row.py` -- a 17-line wrapper
with no hardcoded scene or chunk references. Do not edit the v2 evaluator; it backs the comparison
baseline.

Held-out instances: a fresh master seed, zero `task_seed_u64` intersection with the 152 training
seeds, contiguous manifest indices, and 50/50 left/right `intrusion_side` within each arm. Each
manifest row must reconcile to exactly one result per arm.

**Pipeline evaluation behind training** -- start seed 1's eval while seed 2 trains. This is what
makes the deadline: ~28 h serial becomes ~23 h. Run the smoke first and read **actual** GPU memory
before choosing worker count; 10 eval workers plus a training run may not fit in 23 GB. Record the
choice in `runtime_decision.json` as the chunk-100 run did, and re-apply the `pids.max` preflight
and thread pinning from V3.

Expect ~18.5 min per rollout at chunk 100, so ~20 h for 600 at 10 workers, longer with clutter.

**PACT_PERMUTED on seed 1 only** (+100 rollouts, ~3.3 h) as the validity check that proximity is
being consumed rather than ignored. **This is the first thing to cut if V6 is running behind** --
drop it and say so in the report rather than shortening the primary arms.

## V7 -- analysis

**Primary endpoint:** `clean_success` rate, PACT vs ACT, paired on physical instance, pooled across
the 3 seeds with seed as a blocking factor.

**Report contact | failure alongside the raw contact rate.** Raw counts are confounded by competing
risks: ACT fails earlier and so has fewer opportunities to collide, which is exactly why chunk-100
showed 12 vs 13 hazard contacts while task success differed by 6. The reach corridor's -17.9 pp was
measured on contact | failure, and that is the quantity comparable to it.

**Pre-register the secondary split before looking:** clutter contact alone, panel contact alone.
Report both whatever they show. If clutter contact is 0 in both arms, say so plainly -- it would
mean the environment still does not put clutter in the policies' path, which is a finding about
part 1's siting and not a null result about proximity.

State the limits that hold regardless of outcome: a cross-task frozen corridor encoder, one
architecture, and `place_receptacle` contacts over-counted because a learned policy exposes no
expert phase so the audit stays in phase `other`.

## Verification

- Collection used the datagen pipeline; `proximity_sensor_period_ms` left at 16.6667.
- Every episode passed the file-level gate; proximity `(T, 4, 8, 8)`, finite, non-constant.
- `pact_place_corridor_v2_recovered_152` and its `act_style_data` tree are byte-identical
  afterwards.
- Six `run_manifest.json` files differ from the chunk-100 references only in seed, checkpoint dir
  and dataset dir.
- Both arms load `strict=True`; PACT confirms feature width 32 and projection shape `(512, 32)`,
  and independently re-verifies the encoder SHA-256.
- Eval seeds disjoint from all 152 training seeds; left/right balanced per arm; every manifest row
  reconciles to one result per arm.
- Checkpoints and the frozen encoder re-hash to their pre-run values after the run.
- `runtime_decision.json` records the measured smoke timing and the worker count actually used.
- Primary and both secondary endpoints reported, including any that came out null.

## Constraints

- **Do not start without the Phase-0 gate passed and approved.**
- Do not use the screen harness for collection.
- Do not modify `assets/datagen/pact_place_corridor_v2/recovered_152/`,
  `assets/act_style_data/pact_place_corridor_v2_recovered_152/`, `eval_pact_place_row.py`, or
  `eval_pact_place_chunk100_row.py`.
- Do not change `chunk_size` from 100. Chunk 1 collapses both arms to 0/20 via a gripper-state
  copycat shortcut; chunk 25 is partial. That is settled and is not being re-litigated here.
- Do not substitute, pad or drop an episode that fails a key check -- stop and report.
- Do not shorten the primary arms to save time; cut `PACT_PERMUTED` first.
- Do not touch `inter_finger_dist` or re-score any hybrid-obstacle result.
- Work in `/root/prox_learning_pact_remediation`; interpreter `/root/act_retrain_venv/bin/python3`;
  `MUJOCO_GL=egl`, `MLSPACES_ASSETS_DIR=/root/prox_learning/assets`, `PYTHONPATH` -> repo
  `submodules/molmospaces`, `OPENBLAS_NUM_THREADS=1` (the cgroup `pids.max` is 3840 and otherwise
  kills numpy imports outright).

## Schedule

```
V3  collection, 152 episodes @ 12 workers        ~3 h
V4  conversion + verification gate               ~1 h
V5  training, 6 runs @ ~2.6 h                    ~8 h   GPU
V6  evaluation, 600 rollouts @ 10 workers       ~20 h   pipelined behind V5
V7  analysis + report                            ~2 h
                                          ~23 h pipelined
```

Part 1 is ~10 h plus review time, so the pair is ~33 h of machine time against a 48-72 h budget.
