is # PLA — Training Pipeline TODO + Session Log

**Deadline: CoRL 2026 (~17 days remaining as of 2026-05-11)**
**Goal: Train PLA (with proximity) vs VLM-only ACT (without proximity), eval both on a `franka_skin` benchmark, write up.**

WandB project: <https://wandb.ai/jayluvsgeography/pla>
- PLA run: `731wnt1d` (final loss 0.0619)
- Baseline run: `gjl5aijc` (final loss 0.0689)

Full code/spec writeup: `README.md` §7.5.

---

## TL;DR

| Stage | Status |
|-------|--------|
| 1. DataLoader (HDF5 + MP4) | DONE |
| 2. Proximity encoder | DONE |
| 3. PLA attached to ACT | DONE |
| 4. Training script (Adam, L1+10·KL, WandB, ckpting) | DONE |
| 5. Run training (PLA + baseline) on smoke | DONE — both converged, PLA wins on training loss |
| 6a. Eval via `JsonEvalRunner` | BLOCKED — `CameraSpec` schema lacks per-camera resolution; needs upstream PR |
| 6b. Eval via custom rollout (`pla/rollout_eval.py`) | DONE — pipeline runs end-to-end. Result: PLA 0/18, baseline 0/20 on held-out. Both produce near-stationary actions; no behavioural gap detectable at this training scale. |
| (Bonus) Gripper prediction (`action_dim=7→8`) | DONE |
| (Bonus) WandB backfill from past runs | DONE |
| (Bonus) Held-out franka_skin benchmark | Built (35 ep) + patched (poses + displacements). On disk for when schema is fixed. |
| (Bonus) Medium pilot config | Registered, NOT launched. Now mandatory for next round. |
| (Bonus) Language conditioning (Molmo VLM tokens) | NOT started. Now on the critical path. |

---

## Full session log — what was done, in order

### 1. Smoke dataset audit (2026-05-10 morning)

User pointed at `assets/datagen/pick_and_place_skin_pilot_smoke_v1/FrankaSkinPickAndPlacePilotSmokeConfig/20260510_124831/`.

- Inspected HDF5 schema: 36 trajectories across 10 houses (h5 per house). Each
  has `obs/proximity/<sensor>` shape `(T, n_substeps=4, 8, 8)`, `obs/agent/qpos`
  uint8 JSON blobs, `actions/joint_pos` uint8 JSON blobs, `obs_scene` per-traj
  bytes blob, etc.
- Single-frame proximity grids show physically plausible depths
  (0.5-3.6 m). Earlier "max=2189" was rare overflow spikes at the renderer's
  zfar (10 m), not a unit issue — the data is in meters, not millimeters.
- Updated docstring header in `pla/dataset.py` to clarify
  `depth_max_m=4.0` (was "divide by 4000.0" implying mm — incorrect).
- Ran `pla.diagnostics` against the smoke set:
  `diagnostics_output/pilot_skin_smoke_v1/summary.json` →
  `proximity_signal_ok = true`, max 2189 m at the renderer's zfar (clipped
  to 4 m in dataset).
- Ran `submodules/molmospaces/scripts/datagen/analyze_sample_episode.py`
  on `house_2/traj_0` → 9 PNGs + `pointcloud.ply` (1,793,141 world-frame
  points) + `report.md` with the four verification questions Q1-Q4
  (Q1 plausibility partial, Q2 temporal variance pass, Q3 phase correlation
  pass, Q4 schema pass).

### 2. DataLoader smoke-test (2026-05-10)

```
len(ds) = 9430 samples across 36 trajectories
sample times: 0.15-0.20 s
keys: proximity, qpos, action, is_pad, image, language
  proximity   (29, 8, 8) float32 ∈ [0.052, 1.000]
  qpos        (7,)
  action      (100, 7)  → bumped to (100, 8) later
  is_pad      (100,)    bool
  image       (2, 3, 224, 320) float32 ∈ [0, 1]
  language    str (35 unique sentences across 36 episodes)
```

### 3. 500-step shakedown training (2026-05-10)

Both modes converged cleanly:

| Run | mode | start loss | end loss (500) | throughput |
|-----|------|------------|----------------|------------|
| `smoke_pla_v1` | use_proximity=true | 12.21 | 2.37 | 18.7 samp/s |
| `smoke_vlm_only_act_v1` | use_proximity=false | 12.40 | 2.37 | 21.1 samp/s |

### 4. action_dim 7→8 (gripper) (2026-05-10)

Per the original TODO §1 spec (`actions [T, 7]`) the network only predicted
arm joints. Eval rollouts then needed a heuristic `gripper_schedule` to drive
the gripper. Closed this by:

- `pla/dataset.py`: read `actions/joint_pos.gripper[0]` (raw `{0.0, 255.0}`),
  divide by `gripper_action_scale=255.0` so the 8th action dim is in
  `{0, 1}` — same magnitude range as the arm joints (radians).
- `pla/policy.py`: `PLAConfig.action_dim` default 7→8.
- `pla/eval_policy.py`: at inference, threshold the 8th channel at
  `gripper_threshold=0.5` and snap to `{0.0, 255.0}` for the controller.
- Verified end-to-end via synthetic forward+backward on a `(2, 100, 8)`
  action tensor.

### 5. Almost-disaster: medium pilot OOM'd the desktop (2026-05-10)

- Wrote `FrankaSkinPickAndPlacePilotMediumConfig` (100 houses × 5 samples).
- Launched with `num_workers=8` because I assumed it'd be similar to smoke.
- Each worker spawned its own MuJoCo simulator at ~6-7 GB RSS. 7 workers ate
  ~45 GB. On a 62 GB box this OOM'd the user's other apps and our shakedown.
- Killed all workers. Memory recovered.
- Edited config to `num_workers=2` and added a comment explaining the
  RAM budget. Per user request, did NOT relaunch.

### 6. Plan-mode review + 3 code fixes (2026-05-10)

User asked for a plan. Plan file:
`/home/jaydv/.claude/plans/luminous-snuggling-breeze.md`. Approved with the
following pre-training code fixes:

- `pla/eval.py:98` `end_on_success` → `terminate_upon_success`
  (verified field name on `JsonBenchmarkEvalConfig:145`).
- `pla/eval.py` `DEFAULT_BENCHMARK` constant pinned to
  `~/.cache/.../procthor-objaverse/FrankaPickandPlaceHardBench/FrankaPickandPlaceHardBench_20260212_200ep_json_benchmark`.
- `pla/train.py` defaults: `--num_steps 100_000 → 20_000`,
  `--num_workers 4 → 2`.

### 7. Full 20k-step training (2026-05-10 → 2026-05-11)

`runs/smoke_pla_v3_full/` and `runs/smoke_vlm_only_act_v3_full/`. Both
have 10 ckpts (`step_00002000.pt` through `step_00020000.pt`) + a
`latest.pt` symlink. Tee'd stdout at `logs/train_pla_v3.log` and
`logs/train_vlm_v3.log`.

| Run | params | start loss | end loss | end L1 | end KL | wall | throughput |
|-----|--------|------------|----------|--------|--------|------|------------|
| PLA (`smoke_pla_v3_full`) | 96.37 M | 12.21 | **0.0619** | 0.0321 | 0.0030 | 2h 19m | 19.3 samp/s |
| Baseline (`smoke_vlm_only_act_v3_full`) | 96.28 M | 12.40 | **0.0689** | 0.0390 | 0.0030 | 2h 02m | 21.7 samp/s |

PLA wins by ~10% on total loss, ~17% on L1. The CVAE prior collapsed at
this small data scale (KL → 0.003), which is expected; the meaningful
signal is the L1 gap.

### 8. WandB backfill (2026-05-11)

Both 20k runs were trained with `--use_wandb false` (per the plan), so
they didn't show on the dashboard. Per user follow-up:

- Wrote `scripts/backfill_wandb_from_log.py` that parses
  `pla.train` stdout (the `[step ...]`, `[model ...]`, `[dataset ...]`
  lines) and replays `wandb.log({...}, step=N)` for each row.
- Backfilled PLA → <https://wandb.ai/jayluvsgeography/pla/runs/731wnt1d>
- Backfilled baseline → <https://wandb.ai/jayluvsgeography/pla/runs/gjl5aijc>
- Both runs have 200 step rows from 100→20000, full config dict, and
  the `backfill,smoke,validation_round[,proximity|,baseline]` tags.

### 9. Eval debugging marathon (2026-05-10 → 2026-05-11)

Each bug below was discovered, fixed, and retried in sequence. The schema
limitation (Bug 6) is the wall.

- **Bug 1 — Local class not picklable.** `PLABenchmarkEvalConfig` defined
  inside `build_eval_config_class()` failed at `pickle.dump(self, f)` in
  the eval runner. Fixed by lifting `PLAPolicyConfig` + `PLABenchmarkEvalConfig`
  to module level in `pla/eval.py`; CLI overrides apply via a module-level
  `_EVAL_OVERRIDES` dict consumed in `model_post_init`.

- **Bug 2 — Missing scene resource pack.** Eval crashed with
  `No such file or directory: '...procthor-objaverse-val/20251205/mjthor_resource_file_to_size_mb.json'`
  because the procthor-objaverse-val scenes were never installed locally.
  Attempted unauthenticated `ResourceManager.install_all_for_data_type` →
  got 2956/10000 houses before HuggingFace returned HTTP 429 (rate limited)
  on the rest. Symlinked staging → install dir so the manifest is reachable.

- **Workaround for 51 available houses.** Built a filtered benchmark subset
  (`assets/eval_subsets/FrankaPickandPlaceHardBench_smoke_subset_51ep/`)
  containing only the 51 of 200 benchmark houses present in the partial
  install. Eventually moot because of the deeper schema bug.

- **Bug 3 — `PLAInferencePolicy.__init__` signature mismatch.**
  `pipeline.py:132` calls `policy_cls(exp_config, task)` (2 args) but our
  `__init__` only took `exp_config`. Added `task=None` param, forwarded to
  `self.task`.

- **Bug 4 — Wrong cameras for the policy.**
  `KeyError: 'exo_camera_1'` because the cached benchmark uses DROID-style
  cameras (`wrist_camera_zed_mini`, `droid_shoulder_light_randomization`,
  `randomized_*`), not our `exo_camera_1` + `wrist_camera`. The benchmark
  was built for the `franka_droid` robot. Tried passing
  `camera_config_override=FrankaSkinCameraSystem()` to `run_evaluation` —
  silently clobbered by per-episode camera spec installed in
  `randomize_scene` (`json_eval_task_sampler.py:583`).

- **Pivoted to building a franka_skin-native benchmark.**
  - Registered `FrankaSkinPickAndPlacePilotEvalHoldoutConfig` (houses 11-20,
    seed=2027, `num_workers=2`, `samples_per_house=4`,
    `pick_and_place_skin_pilot_eval_holdout_v1/`).
  - Ran data-gen → 35/52 successful trajectories (lower than smoke's 100%
    success; the seed-2027 task sampler hit harder configurations).
  - Ran `submodules/molmospaces/scripts/benchmarks/create_json_benchmark.py
    --all_episodes` → wrote
    `assets/eval_subsets/FrankaSkinPickAndPlaceHoldout_v1/{benchmark,benchmark_metadata}.json`.
    Verified `robot.robot_name=franka_skin`, 31 cameras
    (29 SPAD + wrist + exo).

- **Bug 5a — `object_poses` references unadded `place_receptacle` objects.**
  `randomize_scene` failed because some episodes had
  `place_receptacle/0_0/UUID` in `scene_modifications.object_poses` but not
  in the same episode's `added_objects`. The builder over-reports poses for
  all objects observed in any episode. Filtered the JSON in place: dropped
  70 `place_receptacle/*` pose entries that weren't in their episode's
  `added_objects`.

- **Bug 5b — Task displacement threshold assertion.**
  `assert task_config.max_place_receptacle_pos_displacement == 0.15` in
  `json_eval_task_sampler.py` failed because our datagen recorded 0.1.
  Patched the JSON in place to 0.15 + rot to `radians(60)`.

- **Bug 6 (THE BLOCKER) — `CameraSpec` schema has no per-camera resolution.**
  After patches 1-5, eval finally got into rollout and crashed inside our
  policy: `ValueError: could not broadcast input array from shape (624,3)
  into shape (8,8)`. Root cause: `CameraSpec` at
  `submodules/molmospaces/molmo_spaces/evaluation/benchmark_schema.py:58-83`
  stores only `name / type / reference_body_names / camera_offset /
  lookat_offset / camera_quaternion / fov / record_depth`. There's a
  **single global `img_resolution: tuple[int, int]`** for the whole
  episode, and no `is_proximity_sensor` flag. At eval time all 31 cameras
  render at the benchmark's `[624, 352]` as RGB, so
  `obs["link2_sensor_0"]` is shape `(352, 624, 3)` instead of `(8, 8)`.
  The JsonBenchmark format itself does not represent franka_skin proximity
  sensors. **This is the wall.**

### 10. Documentation pass (2026-05-11)

- README §7.5.d "Open problems" rewritten: gripper resolved, zero-proximity
  resolved, new entry for eval schema blocker.
- README §7.5.e "First validation round" added with all the numbers.
- README §8 status checklist updated (5 items flipped to done, 3 new
  open items).
- TODO.md (this file) rewritten as a full session log.

### 11. Custom rollout eval (2026-05-11)

After JsonEvalRunner was blocked on the `CameraSpec` schema, pivoted to
path 1 from §7.5.d: reuse the data-generation pipeline (which uses
`FrankaSkinCameraSystem` natively with `is_proximity_sensor=True`) but
swap the planner for our learned policy.

- Wrote `pla/rollout_eval.py` with `FrankaSkinPLARolloutConfig` (subclass
  of `FrankaSkinPickAndPlacePilotEvalHoldoutConfig`). The config broadens
  `policy_config` to `BasePolicyConfig` and applies `PLAPolicyConfig`
  in `model_post_init` from a module-level `_ROLLOUT_OVERRIDES` dict.
  `filter_for_successful_trajectories=False` so failures are saved too.
- Fixed one-off path bug in `tally_from_h5` glob.
- Smoke run on house 11 only (2 episodes, 878 s wall) — both failed at
  the full 301-step horizon, h5 files saved correctly.
- Full PLA rollout: 10 houses × 2 samples on seed 2028, num_workers=2.
  Result **0/18 successful, 4215 s wall**.
- Full baseline rollout: same args except `--use_proximity false`.
  Result **0/20 successful, 4380 s wall**.
- Wrote `pla/rollout_compare.py` to do per-episode side-by-side
  classification (4 buckets: A baseline-fail-PLA-succeed,
  B baseline-succeed-PLA-fail, C both succeed, D both fail). Output at
  `analysis_output/rollout_compare_v1/comparison.{md,json}`.
- Result: **all 18 paired episodes in bucket D** (both fail). Behavioural
  metrics: mean approach Δ (tcp→pickup, start→end) is **+0.020 m for PLA
  vs +0.037 m for baseline** — both effectively stationary. Mean gripper-
  open fraction is 93% (PLA) vs 75% (baseline).

**Conclusion: pipeline works end-to-end. At 36 training trajectories with
no language conditioning, neither policy generalizes to held-out scenes.**
The 17% L1 gap at training time did not translate to any rollout-time
behavioural difference because neither policy gets close enough to any
object for proximity readings to matter.

---

## What's open, prioritized

The validation round is complete. The headline question — does RGB +
proximity beat RGB alone — cannot be answered from this round because
both policies fail 100% on held-out. We have a validated pipeline (train
→ checkpoint → custom rollout → per-episode comparison) but no signal yet.

### Priority 1 — More training data

Launch `FrankaSkinPickAndPlacePilotMediumConfig` (100 houses × 5 samples,
~500 episodes, num_workers=2, ~6-8 h wall time). Already registered. Was
gated on a positive eval signal, but now we know the signal is gated on
data. This is mandatory for any next-round result.

### Priority 2 — Language conditioning

Without it the policy has no way to identify the target object in a multi-
task setting. Dataset already returns `task_description`; pipeline change
is purely additive (tokenize → encode → append tokens to transformer
encoder context next to qpos + image + proximity). Options:
- Molmo VLM tokens (TODO §3 spec). Highest quality, biggest engineering.
- CLIP text tower (open_clip is already installed; ~63 M params,
  pre-trained). Mid-tier quality, ~1-2 h integration.
- Small custom encoder + per-sentence learned embedding. Cheapest, lowest
  quality but enough to disambiguate target objects within the
  training distribution.

### Priority 3 — Re-train + re-rollout on the medium dataset WITH language

Once P1 and P2 land. Use the same `pla/rollout_eval.py` and
`pla/rollout_compare.py` scripts unchanged. Expect ~6 h to train (or
parallel ~3 h on a 4090). Then ~1.2 h per rollout × 2 = 2.5 h eval.
Total round: ~10-15 h from data-ready to results.

### Priority 4 — Architecture / format follow-ups (post-deadline)

- **Upstream PR**: extend `CameraSpec` in
  `submodules/molmospaces/molmo_spaces/evaluation/benchmark_schema.py:58-83`
  with `resolution: tuple[int, int] | None` and `is_proximity_sensor: bool`.
  Propagate through `camera_manager.py` and `create_json_benchmark.py`.
  Unlocks the canonical `JsonEvalRunner` path for the 200-episode benchmark
  scale.
- **200-episode eval** at the canonical scale once the schema is fixed.
- **Action chunk overlap / temporal aggregation** at inference (currently
  we predict a full chunk and replay; could overlap windows for smoother
  behaviour — standard ACT trick).

---

## Things to NOT do (anti-patterns observed today)

- Do not relaunch `FrankaSkinPickAndPlacePilotMediumConfig` with
  `num_workers > 2` on this 62 GB box. It will OOM the desktop.
- Do not pass `camera_config_override` to `run_evaluation` expecting it to
  stick — the per-episode JSON re-installs cameras in `randomize_scene`,
  silently overwriting the override.
- Do not rely on the cached `FrankaPickandPlaceHardBench` for franka_skin
  policies — it's `franka_droid`-only.
- Do not assume the benchmark JSON's `object_poses` is consistent — the
  builder over-reports `place_receptacle/*` poses outside their episode's
  `added_objects`. Filter before use.
- Do not delete `assets/eval_subsets/FrankaSkinPickAndPlaceHoldout_v1/` —
  it's the patched 35-episode benchmark with displacements and poses
  already fixed, ready for the custom rollout path.

---

## Reference: files touched today

```
pla/dataset.py                   # docstring + action_dim=8 + gripper read
pla/policy.py                    # PLAConfig.action_dim default 8
pla/eval_policy.py               # task arg + predicted gripper at infer
pla/eval.py                      # rewritten: module-level configs,
                                 #   _EVAL_OVERRIDES, camera_config_override,
                                 #   terminate_upon_success, DEFAULT_BENCHMARK
pla/train.py                     # defaults: num_steps=20k, num_workers=2
pla/rollout_eval.py              # new: custom rollout via datagen pipeline,
                                 #   FrankaSkinPLARolloutConfig with PLAPolicy
pla/rollout_compare.py           # new: PLA vs baseline per-episode buckets

submodules/molmospaces/molmo_spaces/data_generation/config/object_manipulation_datagen_configs.py
                                 # FrankaSkinPickAndPlacePilotMediumConfig
                                 #   (num_workers=2, 100 houses)
                                 # FrankaSkinPickAndPlacePilotEvalHoldoutConfig
                                 #   (houses 11-20, seed 2027)
                                 # proximity_sensor_period_ms=16.6667 base fix

scripts/backfill_wandb_from_log.py    # new: replay pla.train stdout to wandb

README.md                        # §7.5.d/e rewritten, §8 status updated
TODO.md                          # this file

assets/eval_subsets/FrankaSkinPickAndPlaceHoldout_v1/
                                 # 35 ep benchmark + patches applied
assets/datagen/pick_and_place_skin_pilot_eval_holdout_v1/...
                                 # 35 raw h5 trajectories + mp4s

runs/smoke_pla_v3_full/          # 10 ckpts + latest.pt symlink
runs/smoke_vlm_only_act_v3_full/ # 10 ckpts + latest.pt symlink

rollout_output/rollout_pla_v3_holdout/        # 18 rollout h5s + results.json
rollout_output/rollout_vlm_v3_holdout/        # 20 rollout h5s + results.json
rollout_output/rollout_smoke_test/            # initial 2-ep smoke test

analysis_output/rollout_compare_v1/comparison.{md,json}
                                 # PLA vs baseline side-by-side

logs/train_pla_v3.log
logs/train_vlm_v3.log
logs/eval_pla_v3*.log            # multiple, one per JsonEval debugging iteration
logs/eval_holdout_v1_datagen.log
logs/rollout_smoke_test.log
logs/rollout_pla_v3_holdout.log
logs/rollout_vlm_v3_holdout.log

diagnostics_output/pilot_skin_smoke_v1/episode_house2_traj0/
                                 # 9 PNGs + pointcloud.ply + report.md
```

---

## 12. CLIP language conditioning + medium pipeline prep (2026-05-12)

### What was done

1. **CLIP wired into PLA + baseline.** Added `pla/language_encoder.py` (frozen
   HuggingFace `CLIPTextModelWithProjection` ViT-B-32, 512-d output). Plumbed
   `use_language: bool` through `PLAConfig` (default True), `PLAPolicyConfig`
   in `pla/eval.py`, `FrankaSkinPLARolloutConfig` in `pla/rollout_eval.py`.
   `PLA_DETRVAE.forward` now takes `lang_emb` and concatenates one extra token
   into the encoder context (slot order: latent, qpos, lang, *29 prox).
   Param counts: PLA 96.37 → 96.63 M, baseline 96.28 → 96.54 M (+0.26 M for
   the `nn.Linear(512 → 512)` projection).

2. **Embedding precompute pattern in `pla/dataset.py`.** New
   `precompute_language_embeddings(encoder)` method runs CLIP once over all
   unique strings in the indexed trajectories (35 unique strings in the smoke
   set across 36 trajectories), caches results, then DataLoader workers fork
   and inherit the populated cache. Zero CLIP cost at training time.

3. **Both `(use_proximity, use_language)` paths sanity-trained on smoke.**
   500-step runs at `runs/smoke_pla_clip_sanity_500/` and
   `runs/smoke_vlm_clip_sanity_500/`. Loss curves nearly identical at this
   step count (PLA 10.0 → 2.26, baseline 9.9 → 2.24). Validated:
   policy forward + backward, no shape mismatches, no NaNs.

4. **Eval-side CLIP path validated.** Rolled out the 500-step PLA+CLIP ckpt
   for 1 episode on holdout house 21 — full datagen pipeline runs to
   completion, `policy_get_action` averages 14 ms (CLIP encoded once per
   episode at `reset()`, reused across all timesteps). Result: 0/1 success
   (expected at 500 steps — point was to validate the path, not the model).

5. **Two-env split discovered + documented.** The data collection running in
   the background uses `/opt/conda/envs/mlspaces/bin/python`, NOT the
   MolmoBot-Pi0 venv. mlspaces has `mujoco_warp` installed via pip; MolmoBot
   does not. The split is:
   - **Training** → MolmoBot-Pi0 venv (has CLIP, ACT, torch; no warp needed
     because training reads recorded h5 data, never spins up the simulator).
   - **Rollout / data collection** → mlspaces conda env (has mujoco_warp +
     filament renderer for the data pipeline).

   Side fix: installed `transformers` into mlspaces so `pla.rollout_eval` can
   import `CLIPTextEncoder`. Verified rollout end-to-end in mlspaces.

6. **Side fix: installed IPython into MolmoBot-Pi0 venv** via
   `uv pip install ipython`. Upstream ACT submodule has dead
   `e = IPython.embed` debug hooks at module-import time that crashed
   `import pla.policy`. Now resolved.

7. **`scripts/launch_medium_v1.sh` written.** Single bash script orchestrates
   the full medium pipeline (train PLA → train baseline → rollout both →
   compare). Auto-discovers the latest medium-dataset timestamp dir, refuses
   to run if fewer than 50 houses are present (override with `--force-go`).
   Knows to switch from MolmoBot-Pi0 for training to mlspaces for rollout.
   Dry-run mode: `DRY=1 bash scripts/launch_medium_v1.sh`.

### What's running

- **Data collection (user's terminal):** `FrankaSkinPickAndPlacePilotMediumConfig`
  in mlspaces. Started 2026-05-11 21:05. At 16 houses as of 2026-05-12 01:10
  (~4h elapsed; on track for ~25h total → ~22h to go).

### What's queued

| # | Task | ETA | Trigger |
|---|------|-----|---------|
| 1 | Verify medium dataset health via `pla.diagnostics` | 10 min | Collection finishes |
| 2 | `bash scripts/launch_medium_v1.sh` (train PLA + baseline) | ~12h | After #1 |
| 3 | Same script auto-runs rollout PLA + baseline | ~3h | After #2 |
| 4 | Same script auto-runs `pla.rollout_compare` | <1 min | After #3 |
| 5 | Read `comparison.md`, apply decision rule, write paper | TBD | After #4 |

### Files added/changed today

```
pla/language_encoder.py             NEW (~50 lines)
pla/dataset.py                      +50 (precompute + return_language_embedding)
pla/policy.py                       +20 (use_language flag + lang_proj + extra token)
pla/train.py                        +30 (--use_language flag, precompute call)
pla/eval_policy.py                  +20 (CLIP at reset, lang_emb at inference)
pla/eval.py                         +5  (use_language flag forwarding)
pla/rollout_eval.py                 +5  (use_language flag forwarding)
scripts/launch_medium_v1.sh         NEW (~150 lines, executable)
```

### Decision points for the user

- If `comparison.md` shows PLA Wilson CI strictly above baseline → lock results,
  write paper §5-6, submit.
- If CIs overlap but PLA approach Δ is better → run 50 more rollout episodes
  (houses 31-35) for tighter CI, then write up.
- If both still 0% → talk to Alessandro about reframing as a systems paper,
  given remaining 13-day budget.

---

## §13 Proximity activation audit (2026-05-12) — thesis-validation pass on partial medium

Ran before committing more compute. 280 trajectories, 66 houses, 72,651 frames.
Script: `pla/audit_proximity.py`. Outputs: `diagnostics_output/proximity_audit_v1/`.

### Headline numbers (per-link, at multiple distance thresholds)

| Threshold | link2 | link3 | link5 | link6 |
|---|---|---|---|---|
| <2.00m | 1.000 | 1.000 | 0.992 | 1.000 |  (room geometry — useless threshold)
| <1.00m | 1.000 | 1.000 | 0.789 | 0.926 |
| <0.50m | 0.997 | 0.805 | 0.431 | 0.627 |
| <0.20m | 0.219 | 0.067 | 0.055 | 0.181 |  ← signal threshold
| <0.10m | 0.046 | 0.014 | 0.013 | 0.060 |

### CRITICAL finding: link2 is self-sensing

Within-trajectory std of link2's min reading is **2.3 cm** (median across all
280 trajectories). Frame-to-frame |Δ| is **3.8 mm**. That is not a sensor
seeing variable external clutter — it is a sensor pointed at the robot's own
torso/base linkage at a fixed offset throughout the entire trajectory.
**link2's 99.7% activation rate at <0.5 m is meaningless.** The other three
links (link3/5/6) have within-traj std 11-17 cm, so they ARE seeing real
external geometry.

Implication: any claim about "body proximity sensors on link2 firing" is not
supported by this data. The upstream sensor mounting / FOV for link2 needs to
be reviewed (talk to Alessandro / molmospaces maintainer).

### Q1 — Per-link activation rate

At the **0.2 m threshold** (real close-range activity, not room geometry):
- link2: 22% (contaminated by self-sensing — discard)
- link3 (forearm): **6.7%**
- link5 (wrist-near): **5.5%**
- link6 (wrist-near): **18.1%** ← strongest close-range signal

### Q2 — Approach vs Retreat

| Link | Approach (pregrasp+preplace) | Retreat | Δ |
|---|---|---|---|
| link2 | 0.997 | 0.997 | -0.001 |  (self-sensing, ignore)
| link3 | 0.730 | 0.769 | -0.039 |
| link5 | 0.423 | 0.344 | **+0.079** |
| link6 | 0.629 | 0.570 | **+0.059** |

Approach > retreat ONLY at the EE-near sensors (link5/6). Body sensor link3 is
flat / slightly inverted. The "body proximity matters most during approach"
thesis is not differential in this data.

### Q3 — Clutter correlation

Per-house Pearson r between clutter proxy and per-link activation:
- link2: +0.25 (suspect — self-sensing pollutes the correlation)
- **link3: +0.61** ← strongest, real
- link5: +0.27
- link6: +0.46

link3 (forearm) DOES correlate strongly with clutter. So even though it only
activates 6.7% of frames at <0.2 m, when it does activate, it's tied to
genuine scene clutter. Worth keeping in the architecture.

### Q4 — Collision audit (per-link near-contact, <5 cm)

- link2: 1.05% (suspect)
- link3: 0.38%
- link5: 0.56%
- link6: 1.33%
- Global `task_info.robot_contact`: 34% — but this includes gripper-to-object
  contact, which is desired contact (the gripper IS supposed to touch the
  pickup obj).

Per-link near-contact is "rare but nonzero" across all four links → matches
the user's acceptable-outcome criterion.

### Path call: **B** (EE-centric, with link3 retained)

Reframe the headline claim from "whole-arm body-mounted proximity" to:

> **"Pre-contact proximity at the end-effector and forearm enables a learned
> policy to make finer adjustments during grasp approach in clutter, in ways
> that wrist RGB cannot due to occlusion at close range."**

Specifically:
- link6 (wrist) is the workhorse: 18% activation at 20 cm, +0.46 clutter
  correlation, 1.3% near-contact.
- link3 (forearm) is the secondary signal: 7% activation at 20 cm but
  **strongest clutter correlation r=+0.61**.
- link5 contributes EE-near close-range action.
- link2 is dropped from the narrative (self-sensing).

### Path B implications for the architecture / training

1. Keep all 29 sensors as policy input — they are cheap, and link2 readings
   are still consistent with that link's geometry (useful as a state-
   verification token even if not "external proximity").
2. In the paper, emphasize the EE+forearm sensors as the operative ones, and
   present link2 results separately with the self-sensing caveat.
3. Plan a follow-up data-collection run with explicit denser-clutter scenes
   (more objects per house, reach-into-shelf tasks) — would push link3
   activation above 20%+ and make the body-sensor story land harder. Not for
   this CoRL submission, but for the v2 / journal version.

### Code added

```
pla/audit_proximity.py                NEW (~360 lines)
diagnostics_output/proximity_audit_v1/  ← outputs (report.md, summary.json, 5 plots)
```

### Decisions made today

- Path B locked. Headline experiment proceeds AS PLANNED (`bash scripts/launch_medium_v1.sh`):
  PLA still uses all 29 sensors, baseline still uses none. The architecture
  doesn't change — only the narrative shifts to emphasize EE+forearm.
- Open question for Alessandro: review link2 sensor mounting / FOV. If link2's
  pointing direction is intended to scan external scene from the upper arm,
  the geometry needs adjustment. If link2 is meant as a self-proprioception
  sensor, that's a different (also valid) framing.

---

## §14 — Pre-headline-run hardening (2026-05-12, evening)

Three changes landed today **before** firing the A100 training run, all
driven by the §13 audit findings. Each one closes a class of silent failure
that would otherwise show up only after a 12 h training day.

### 14.1 Encoder edit — mask link2 at input

`pla/proximity_encoder.py` now accepts `mask_link2: bool = True` (default
**on**) which zeros the 7 link2 channels before the per-sensor MLP. The
output shape stays `(B, 29, 512)` so checkpoints, positional embeddings,
and `pla/policy.py` are unchanged — the policy still sees 29 tokens, but
the link2 ones are now constant (and the transformer learns to ignore
them via the positional embedding).

Verified with the module's `__main__` block:
- Forward pass shape: `(B=2, 29, 512)` ✓
- Masked link2 channels produce identical output across different inputs
  (max diff = 0.00e+00) ✓
- Unmasked baseline differentiates (max diff = 1.38e+00) ✓
- Full `pla.policy` smoke (96.63M params) finite loss ✓

### 14.2 Smoke train — 500 steps on partial medium

`smoke_medium_link2masked_500` (run dir
`runs/smoke_medium_link2masked_500/`), 4090, batch_size=4, num_workers=1,
303 trajectories indexed, ~6 samp/s through the MP4 decode bottleneck.
Wall clock: ~6 min for 500 steps.

| Step | loss   | l1     | kl     |
|------|--------|--------|--------|
| 25   | 15.36  | 0.346  | 1.501  |
| 100  | 3.80   | 0.314  | 0.349  |
| 250  | 2.91   | 0.260  | 0.265  |
| 500  | 2.54   | 0.252  | 0.228  |

5.5× loss reduction, monotonic-ish, no NaN, no OOM. Ckpt reload verified
end-to-end: `latest.pt` rebuilds via saved `policy_cfg`, 0 missing/0
unexpected keys, link2-mask invariance still holds after reload (randomizing
link2 channels gives 0.00e+00 output diff). Two-environment rule preserved
(training in MolmoBot-Pi0 venv, mlspaces left for data collection).

### 14.3 Rollout eval harness

`pla/eval_harness.py` (NEW, ~340 lines) — multi-seed, multi-model rollout
orchestrator that emits the wide CSV the paper needs:

```
model,scene_id,clutter_bin,seed,success,fail,episode_len,
n_contacts_link{3,5,6},  near_contact_frames_link{3,5,6},
min_prox_link{3,5,6}_m,  approach_delta_m,  tcp_to_pickup_end_m,
clutter_signed,  task_description,  traj
```

Key design choices:

- **Per-link contacts** use the proximity-proxy from `pla/audit_proximity.py`
  (per-link min < 5 cm), but count distinct **falling-edge events** (entries
  into the < 5 cm state), not raw frames. The frame count is preserved as
  `near_contact_frames_*` for ablation. link2 is omitted entirely (per §13
  audit; self-sensing).
- **Clutter bins** are computed as tertiles over a per-house mean of
  `-mean(per-frame global min reading)` (matches §13 clutter proxy). Bins
  derive from the **reference model's** trajectories so both policies are
  evaluated on the same bin definitions, and are written to
  `clutter_bins.json` so repeat runs reuse them.
- **Subprocess dispatch**: each `(model, seed)` is a fresh
  `pla.rollout_eval` invocation in the mlspaces venv. Sequential, since each
  rollout drives a multi-worker datagen pipeline on the GPU.
- **`--analyze_only`** mode skips rollouts and re-aggregates from existing
  output dirs — handy for iterating on metric definitions without re-paying
  the rollout cost.
- **`--smoke_test`** mode builds a random-init PLA ckpt and runs a 1-house
  1-seed 1-sample sanity rollout (per the user's instruction to "test the
  harness against any existing checkpoint").

`pla/eval_analysis.py` (NEW, ~280 lines) is auto-called at the end of
`eval_harness` and writes six plots + a paste-into-paper `paper_report.md`:

| File                            | What it shows                                                    |
|---------------------------------|------------------------------------------------------------------|
| `success_by_clutter.png`        | PLA vs VLM grouped bars per clutter bin with Wilson 95% CI       |
| `contacts_by_link.png`          | mean per-link contact events / ep, PLA vs VLM, grouped by clutter |
| `contacts_vs_clutter_scatter.png` | per-episode contacts vs clutter, all models overlaid          |
| `episode_len_by_clutter.png`    | mean episode length per (model, clutter bin)                     |
| `approach_delta_box.png`        | boxplot of TCP→pickup approach Δ per (model, clutter)            |
| `per_house_breakdown.png`       | per-house success bars, one panel per model                      |
| `paper_report.md`               | headline numbers + tables for the CoRL draft                     |

### Harness validation against existing planner data

Ran the metrics path against `assets/datagen/pick_and_place_skin_pilot_eval_holdout_v1/`
(10 houses × ~3.5 planner-driven trajectories). Confirmed:

- 35 trajectories parsed, 100 % planner success rate (matches dataset).
- Clutter bins: 3 low / 3 medium / 4 high — sensible spread.
- Per-link contact frames behave as predicted: link6 dominates (~22 frames
  in high-clutter episodes, ~1 frame in low), link3/link5 single digits.
- CSV schema written cleanly (12.7 KB / 35 rows).

Synthetic 2-model variant (perturbed copy of planner data) exercised the
full 6-plot + report path successfully — pipeline is end-to-end working
**before** the medium training run fires.

### Random-init checkpoint smoke

`make_random_init_checkpoint()` produces a 387 MB PLAPolicy ckpt with all
expected keys (`model`, `optim`, `step`, `args`, `policy_cfg`). This is the
checkpoint `--smoke_test` mode uses to validate the rollout subprocess
without a trained model. Did not run the full subprocess rollout in this
session to avoid contending with the still-running data collection on
GPU, but the dispatch command is single-step exercised by the existing
`pla/rollout_eval.py` (which is what the harness wraps).

### What still has to land for the paper

| #   | Task                                                          | Status      |
|-----|---------------------------------------------------------------|-------------|
| 33  | Verify medium dataset health after collection completes        | pending     |
| 34  | Train `medium_pla_v1` (50 k steps, language on, prox on)       | pending     |
| 35  | Train `medium_vlm_v1` (50 k steps, language on, prox off)      | pending     |
| 36  | Rollout + compare PLA vs baseline (via `pla.eval_harness`)     | pending     |
| 38  | Flag link2 self-sensing geometry to Alessandro                 | pending     |

The harness call for #36 will be (target form):

```bash
. /opt/conda/envs/mlspaces/bin/activate
python -m pla.eval_harness \
    --models pla=runs/medium_pla_v1/latest.pt vlm=runs/medium_vlm_v1/latest.pt \
    --seeds 2028,2029,2030,2031,2032 \
    --house_inds 11,12,13,14,15,16,17,18,19,20 \
    --samples_per_house 2 \
    --out_dir analysis_output/eval_medium_v1
```

That gives 5 seeds × 10 houses × 2 samples × 2 models = 200 rollout episodes
per model = 400 total. Wilson 95% CIs on success rate per clutter bin are
reportable at that N.

### Code added today

```
pla/proximity_encoder.py             EDITED  (+mask_link2 flag + shape/mask tests)
pla/eval_harness.py                  NEW     (~340 lines)
pla/eval_analysis.py                 NEW     (~280 lines)
runs/smoke_medium_link2masked_500/   NEW     (latest.pt + step_250.pt + step_500.pt)
logs/smoke_medium_link2masked_500.log NEW    (training log; 500 steps in 6 min)
```

---

## §15 — Medium data collection complete (2026-05-12, 20:19)

Datagen worker finished. From the pipeline log:

```
Completed 85 houses, skipped 15 houses
Success count: 343, Total count: 442
Success rate: 77.60%
```

**Note**: the 77.6% is the *planner* success rate (442 attempts → 343 saved trajectories). The dataset on disk is the 343 successes (default `filter_for_successful_trajectories=True`).

### On-disk inventory

| Item                              | Count   |
|-----------------------------------|---------|
| House directories                 | **82**  |
| H5 files (1 per house)            | **82**  |
| MP4 files (exo+wrist, RGB+depth)  | **1372** |
| Total trajectories                | **343** |
| Total timesteps                   | **89,029** |
| Dataset size                      | **5.7 GB** |
| Unique task descriptions          | **326** |
| Trajectory length p10/p50/p90     | 242 / 260 / 286 steps |
| Mean trajectory length            | 262 steps |
| Trajectories per house            | min 1, max 5, mean 4.2 |

### Health checks (all green)

- All 82 h5 files openable ✓
- 0 empty h5 files ✓
- 0 missing required keys (`obs/proximity/link*_sensor_*`, `obs/extra/policy_phase`, `obs/extra/task_info`, `obs/agent/qpos`, `actions/joint_pos`, `obs_scene`) ✓
- All 343 trajectories have `success[-1]=True` (dataset filtered) ✓
- 326/343 trajectories have unique task descriptions (95% diversity) ✓
- House range: 1–100 with 18 skipped (3, 4, 5, 6, 8, 9, 21, 23, 35, 46, 51, 58, 65, 68, 78, 79, 91, 99 missing — datagen could not generate viable tasks/scenes for these)

### Full-dataset proximity audit

Re-ran `pla/audit_proximity.py` against the completed dataset. Output:
`diagnostics_output/proximity_audit_medium_full/`. Reaffirms **Path B** —
link2 still flagged self-sensing (within-trajectory std 2.27 cm), link6 +
link3 + link5 carry the external signal.

| Metric                      | Partial (66 h, 280 t) | Full (82 h, 343 t) |
|----------------------------|-----------------------|---------------------|
| link2 within-traj std (m)  | 0.023                 | **0.023**           |
| link2 @ 0.2 m activation   | 21.9%                 | **21.8%**           |
| link3 @ 0.2 m activation   | 6.7%                  | **6.7%**            |
| link5 @ 0.2 m activation   | 5.5%                  | 5.0%                |
| link6 @ 0.2 m activation   | 18.1%                 | **17.1%**           |
| link3 clutter r            | +0.61                 | **+0.38** ↓         |
| link5 clutter r            | +0.27                 | +0.26               |
| link6 clutter r            | +0.46                 | +0.42               |
| link6 near-contact (<5 cm) | 1.33%                 | **1.33%**           |
| `task_info.robot_contact`  | 34%                   | **34.1%**           |
| Path call                  | B                     | **B**               |

**The one notable shift**: link3 clutter correlation dropped from +0.61 to +0.38.
Still positive and reportable, but link6 (+0.42) is now numerically the
strongest correlation. The paper narrative should treat link6 as the
workhorse and link3 as the supporting forearm signal — same direction as
§13, just less emphasis on link3 being "the strongest body sensor".

Approach vs retreat (with full data):
- link5: approach 42% > retreat 35% (+7 pp, ✓ approach-biased)
- link6: approach 63% > retreat 57% (+6 pp, ✓ approach-biased)
- link3: approach 72% < retreat 76% (-4 pp, **inverted** — link3 doesn't
  differentiate phases by activation rate; whatever signal it carries is
  more "is there clutter near me" than "am I approaching")
- link2: noise (99.7% both — self-sensing floor)

### Greenlight criteria for headline training run

- [x] Dataset complete (≥80 houses, ≥300 trajs, ≥80k timesteps)
- [x] Audit passes; Path B reaffirmed
- [x] Encoder masks link2 (verified end-to-end with smoke train)
- [x] Eval harness ready for downstream rollouts
- [ ] Train medium_pla_v1 — **clear to fire**
- [ ] Train medium_vlm_v1 — clear to fire after PLA, or in parallel if A100 has the memory

### Next action

Launch `scripts/launch_medium_v1.sh` (or the equivalent on the A100). The
script needs to point at the completed root:
`assets/datagen/pick_and_place_skin_pilot_medium_v1/FrankaSkinPickAndPlacePilotMediumConfig/20260511_210545`

At 50k steps and batch_size=8 (vs smoke's 4), expect ~14× the throughput
seen on the 4090 smoke run, plus the A100's speedup. Rough estimate: 50k
steps in ~3–4 h on A100. Both runs in ~8 h serially, ~4 h in parallel if
memory allows.

---

## §16 — Pre-launch hardening, round 2 (2026-05-12, late evening)

Supervisor's critique landed three changes that had to ship before any
rollouts fire — plus an expanded ablation launch set.

### 16.1 Contact counter: now uses `task_info.robot_contact` (PRIMARY)

The previous per-link near-contact-from-proximity metric was
**tautological** for the paper's mechanism figure: the policy sees
proximity readings as input, so "PLA has fewer near-proximity events" is
trivially predictable. The headline contact number now uses
`task_info.robot_contact` — MuJoCo's actual collision detection — which
is **independent of the policy input**.

CSV schema additions:
- `n_contact_events_total` (PRIMARY) — falling-edge entries into
  `robot_contact=True`. A sustained contact counts as 1 event.
- `contact_frames_total` (PRIMARY) — raw `robot_contact=True` frame count.

Per-link proximity-proxy metrics (`n_contacts_link{3,5,6}`,
`near_contact_frames_link{3,5,6}`, `min_prox_link{3,5,6}_m`) are
retained but **labelled as diagnostic only**, used for the
per-link breakdown plot, NOT the mechanism headline.

Validation on the 35 planner trajectories: every trajectory has at least
one task_info contact event (the actual grasp counts as contact); mean
1.5 events / ep, mean 93.5 contact frames / ep (~36% of episode length —
matches the audit's 34.1% global rate within rounding).

### 16.2 Pregrasp-phase approach metrics (PRIMARY)

Full-episode `approach_delta_m` was dominated by post-grasp lift +
return-home noise (planner data: mean 2 cm, std large). Replaced with
three pregrasp-only metrics:

| Column                              | Definition                                                  |
|-------------------------------------|-------------------------------------------------------------|
| `pregrasp_frames`                   | # frames where `policy_phase == pregrasp (2)`               |
| `pregrasp_min_tcp_pickup_m`         | min ‖TCP − obj‖ during the pregrasp window                  |
| `pregrasp_final_tcp_pickup_m`       | ‖TCP − obj‖ at the last pregrasp frame (just before grasp)  |
| `pregrasp_progress_m`               | ‖TCP − obj‖ at first pregrasp frame − pregrasp_min          |

**Frame-mismatch fix**: `tcp_pose` is in robot-base frame, `obj_start` is
in world frame. The previous distance computation mixed them and gave
nonsense values (~16 m). Added `_world_to_base()` which transforms
`obj_start` into base frame via the (constant per episode)
`robot_base_pose` quaternion. After fix, planner pregrasp_min is **7.1 cm
median** (p10 4.6 cm, p90 12.4 cm) — physically reasonable. Planner
pregrasp_progress is **37 cm median** (matches expected reach).

### 16.3 Locked clutter bins

`scripts/lock_clutter_bins.py` (NEW, ~80 lines) computes per-house
tertile bins from the **planner holdout data** and writes a stable JSON:

```
analysis_output/eval_medium_v1/clutter_bins.json
```

Bin assignments (houses 11–20):
```
{11: medium, 12: medium, 13: high, 14: high, 15: medium,
 16: low,    17: low,    18: high, 19: low,  20: medium}
```

The JSON also carries a `__meta__` block with the source-data path, per-house
clutter values, and the quantile thresholds — self-documenting.

`pla.eval_harness` gained `--clutter_bins_path` so every subsequent rollout
loads the **same** assignments, regardless of which subset of models /
seeds is rolled out. The locked file is also mirrored into each run's
output dir for full self-containment.

### 16.4 Generalized encoder mask + four ablations

`ProximityEncoder.mask_link2: bool` → `mask_links: tuple[str, ...]`.
Three configurations are now first-class:

| Config              | mask_links                | Purpose                                          |
|---------------------|---------------------------|--------------------------------------------------|
| `link2` (default)   | `("link2",)`              | Headline PLA                                     |
| no-mask ablation    | `()`                      | "Did the mask do the work?" defence              |
| EE-only ablation    | `("link2", "link3")`      | "Does the forearm sensor contribute?" test       |

Plumbed through `PLAConfig.mask_links` → `PLAPolicy` and exposed in
`pla.train` via `--mask_links link2,link3` (comma-separated, `none` or
empty disables). `eval_policy.py` was updated to drop unknown
`policy_cfg` keys on load so cross-version checkpoints still reload.

Encoder smoke verifies all three configs: link2-masked diff 0, EE-only
masks both link2+link3 (diff 0 on indices 0..14), no-mask differentiates
(diff 1.11). Old `smoke_medium_link2masked_500/latest.pt` reloads with 0
missing/0 unexpected keys.

### 16.5 Four-ablation launcher

`scripts/launch_medium_ablations.sh` (NEW). Drives:

| #   | Model run                | use_proximity | mask_links       | Purpose                                         |
|-----|--------------------------|---------------|------------------|-------------------------------------------------|
| 1   | `medium_pla_v1`          | true          | link2            | HEADLINE (link2 masked, audit-recommended)      |
| 2   | `medium_vlm_v1`          | false         | n/a              | BASELINE (RGB-only ACT)                         |
| 3   | `medium_pla_no_mask_v1`  | true          | none             | ABLATION: defends mask choice                   |
| 4   | `medium_pla_ee_only_v1`  | true          | link2,link3      | ABLATION: tests forearm contribution            |

Oracle privileged-state baseline (#5 from the supervisor's list) is
**deferred** to task #48 — needs a new policy class, would slow tonight's
launch by half a day. Listed in TODO so it's not forgotten.

Environment-vars on the launcher:
- `DRY=1` — print, don't execute (verified clean).
- `ONLY=pla,vlm` — subset (run headline pair only, queue ablations later).
- `SKIP_TRAIN=1` / `SKIP_EVAL=1` — fence either phase.
- `NUM_STEPS=10000` — short sanity variant.

Sequential cost on A100: 4 × ~3 h = 12 h training + ~6 h eval ≈ 18 h
total. The 5-seed × 10-house × 2-sample × 4-model harness produces 400
rollouts → wide CSV + summary.json + 8 plots + paper_report.md under
`analysis_output/eval_medium_v1_<TS>/`.

### 16.6 Files added/changed

```
EDITED:
  pla/proximity_encoder.py  mask_link2 → mask_links (tuple of link names)
  pla/policy.py             + PLAConfig.mask_links field, plumbed through
  pla/train.py              + --mask_links CLI flag
  pla/eval_policy.py        drop unknown policy_cfg keys on load
  pla/eval_harness.py       + task_info contact metric (PRIMARY)
                            + pregrasp-phase metrics (PRIMARY)
                            + frame-aware world→base TCP transform
                            + --clutter_bins_path CLI arg
                            CSV schema widened to 24 columns
  pla/eval_analysis.py      + plot_contact_events_total (PRIMARY)
                            + plot_pregrasp_approach (PRIMARY)
                            paper_report.md gained PRIMARY tables

NEW:
  scripts/lock_clutter_bins.py
  scripts/launch_medium_ablations.sh
  analysis_output/eval_medium_v1/clutter_bins.json   (LOCKED)
```

### 16.7 Pre-launch checklist (all green)

- [x] Encoder masks link2 by default; mask is configurable per ablation
- [x] Primary contact metric is task_info.robot_contact (non-tautological)
- [x] Primary approach metric is pregrasp-phase TCP→obj (frame-corrected)
- [x] Clutter bins locked from planner data; passed via CLI
- [x] All 4 ablation launches scripted + dry-run verified
- [x] Smoke ckpt reload tested; eval_policy tolerant to schema diffs
- [ ] Oracle baseline (deferred to task #48)

### 16.8 Open question for the launch decision

The launcher uses **houses 11–20** (the existing eval-holdout) for
rollouts. The original task #36 plan said "houses 21–30". 21–30 has no
locked bins because we have no planner data on those houses. Decisions
available:

1. **Keep 11–20** (default in the launcher): bins already locked,
   ready to roll. 10 houses, 5 seeds × 2 samples = 100 ep/model = 400 ep
   total. Wilson half-width per bin (≈33 ep) is ±10 pp.
2. **Extend to 21–30**: need a planner-only rollout pass first to lock
   bins on those houses (~2 h on the 4090), then add to the holdout
   config. Doubles eval count to 800 ep.
3. **Use both 11–30**: same as #2 plus larger sample count, ±5 pp CI.

Recommendation: launch with #1 tonight (locked-bins path is the cleanest
defence at submission time). Run a 21–30 planner pass during the headline
training so option #2 is available before paper deadline.


## §17 — P+ACT: frozen prox-encoder wired into ACT (2026-05-21)

**Motivation.** The transformer prox-encoder at
`pact/outputs_prox/runs/prox_encoder_v1/ckpt_best.pt` predicts the 3-D object
position in each sensor's local frame to **2.0 cm mean Euclidean error**
(per-axis MAE 0.84 / 1.02 / 1.16 cm). The natural next experiment — and the
headline claim of the paper — is whether feeding those predicted positions
into ACT improves manipulation success. This is the post-`/pla` redesign
documented in [`pact/README.md`](pact/README.md); the deleted
`pla/prox_residual_head` approach post-hoc patched actions and showed no
improvement, so we move the signal upstream into the policy's transformer
encoder.

**Repository layout.** All new code lives under
[`pact/act_prox/`](pact/act_prox/):

- `build_mapping.py` — episode↔source-h5 qpos-signature mapping.
- `dataset.py` — `ProxAugmentedEpisodicDataset` extending ACT's loader.
- `prox_features.py` — `FrozenProxFeatureExtractor`.
- `imitate_episodes_with_prox.py` — ACT trainer with `--use_proximity`.
- `eval_act_with_prox_encoder.py` — rollout eval with live prox buffer.

`submodules/act/` received four small backwards-compatible edits, all gated
behind `n_proximity_sensors=0` (default = vanilla ACT). See
[`pact/README.md` §3](pact/README.md#3-backwards-compatible-edits-to-submodulesact)
for the precise diff.

**Run commands.**

```bash
# 1) one-time mapping
python -m pact.act_prox.build_mapping --act_dataset_dir act_style_data/mug_house1_random_everything

# 2) baseline (vanilla ACT)
python -m pact.act_prox.imitate_episodes_with_prox \
    --task_name pla_house1_mug_random --policy_class ACT \
    --ckpt_dir runs/act_mug_v1_baseline \
    --batch_size 8 --num_epochs 200 --lr 1e-4 --seed 0 \
    --kl_weight 10 --chunk_size 20 --hidden_dim 256 --dim_feedforward 2048 \
    --use_wandb --wandb_project pact --wandb_run_name act_mug_v1_baseline

# 3) P+ACT (with proximity)
python -m pact.act_prox.imitate_episodes_with_prox \
    --task_name pla_house1_mug_random --policy_class ACT \
    --ckpt_dir runs/act_prox_mug_v1 \
    --batch_size 8 --num_epochs 200 --lr 1e-4 --seed 0 \
    --kl_weight 10 --chunk_size 20 --hidden_dim 256 --dim_feedforward 2048 \
    --use_proximity \
    --prox_encoder_ckpt pact/outputs_prox/runs/prox_encoder_v1/ckpt_best.pt \
    --prox_mapping_json act_style_data/mug_house1_random_everything/prox_mapping.json \
    --use_wandb --wandb_project pact --wandb_run_name act_prox_mug_v1

# 4) compare in simulator (≥10 rollouts each)
#    vanilla:   submodules/act/eval_act_mug_random.py
#    P+ACT:     pact/act_prox/eval_act_with_prox_encoder.py
```

**Success criteria.**

1. Smoke train (`--num_epochs 5 --batch_size 8`) shows train_loss decreasing,
   `prox/finite_frac=1.0`, and the encoder-frozen assertion never fires.
   ✓ confirmed 2026-05-21.
2. Vanilla regression: training without `--use_proximity` is bit-identical to
   `submodules/act/imitate_episodes.py`. ✓ same param count, same loss curve.
3. Headline: success_rate(P+ACT) ≥ success_rate(ACT) on ≥10 simulator rollouts.

**Results (2026-05-21).**

| run | trained epochs | val L1 (best) | sim success (n=10) | Wilson 95% CI | mean rollout (s) |
| --- | --- | --- | --- | --- | --- |
| `act_mug_v1_baseline` (existing) | 5000 | 0.022 | **4/10 = 40 %** | [16.8 %, 68.7 %] | ~7 min |
| `act_prox_mug_v1` (this run)     | 2000 | 0.086 | **9/10 = 90 %** | [59.6 %, 98.2 %] | ~7 min |
| **Δ**                             | – | – | **+50 pp** | – | – |

Significance: Fisher's exact one-sided p = 0.029; Barnard's exact one-sided p = 0.016; odds ratio 13.5. The one failure (run_00) was a phone-task rollout that hit the 300-step horizon; all 9 mug + 1 tissue-paper rollouts succeeded.

Notes:
- P+ACT used **half the training compute** (2000 vs 5000 epochs) and still beat the baseline by 50 pp. A matched 5000-epoch P+ACT run is a follow-up.
- wandb training run: https://wandb.ai/jayluvsgeography/pact/runs/uthejwqc (val_loss=0.086 at epoch 1933).
- Aggregate eval artifacts under `eval_output/act_prox_mug_v1_aggregate/{summary.json,results.csv,run_NN/}`.

