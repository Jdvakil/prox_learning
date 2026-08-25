# CURSOR.md — session change log

Living log of agent edits in this checkout. Read this before new work. Append
a dated block after every non-trivial change. Do not replace history; add
below.

**Science / method notes** live in [`PACT.md`](PACT.md). **Paper-agent brief**
lives in [`paper.md`](paper.md). **How to run the repo** lives in
[`README.md`](README.md). This file is only *what the agent changed, why, how, when*.

How to append:

```
## YYYY-MM-DD — short title
- **When:** ...
- **Why:** ...
- **What:** files + behavior
- **How:** implementation notes
- **Not done:** leftover for the user or a later session
```

Newest session at the top.

---

## 2026-08-24 — dropped Safety-CVAE weights; PACT-raw does not need them

- **When:** 2026-08-24 ~22:20 America/Denver.
- **Why:** User: keep cvae_v3 only if tests prove it worth it, else delete and use
  new data. Honest call: **not worth it for PACT.** Raw never ran `model.pt`.
  Trunk/delta already lost. Reflex is a different method.
- **What:** `hybrid_skin_sensors.py` holds the 40-name order. `ProxCVAEEncoder`
  `feature=raw` loads no weights. Convert default no longer reads meta.json.
  Deleted `assets/safety/cvae_v{1,2,3}/`. Tests: `tests/test_prox_raw.py`.
- **How:** leftover `sweep_v*.h5` + reflex demo mp4/mcap still in `assets/safety/`.
  `scripts/safety_*_demo.py` will 404 until someone retrains a head.
- **Not done:** gate-bar collect.

---

## 2026-08-24 — user wiped datagen / ckpts / eval; leftover converted hdf5

- **When:** 2026-08-24 ~22:11 America/Denver.
- **Why:** User deleted data, ckpts, eval for a clean slate.
- **What is gone:** `assets/datagen/` (all source collections, including
  hybrid_obstacle_v1 and hybrid_invis_obstacle_v1), `submodules/act/ckpts/`,
  `eval_output/`. July and avoid-v1 **cannot** be reproduced from this disk.
- **What is still here (do not mix into the next run):**
  `act_style_data/obstacle_prox_v2` (918M) and `obstacle_prox_avoid_v1` (1.5G).
  `assets/safety/cvae_v3` — keep; PACT frozen encoder.
- **How / next:** gate-bar from zero (README §12.2). Do not reconvert avoid-v1.
  `obstacle_gate_v1` counts still 0 until convert. Wipe leftover hdf5 if they
  want the disk to match the story.
- **Not done:** preflight / collect / convert / train / eval.

---

## 2026-08-24 — avoid-v1 n=50 grid **complete** (visible PACT finished)

- **When:** `pact_raw_persensor_idrop03_s0_visible/eval_summary.json` 2026-08-24 15:22 MDT.
  Eval process gone; tmux idle at prompt.
- **Why:** User asked updates.
- **What:** last cell filled. Full n=50, `--temp_agg_off`, prox on/off checked:

| cell | vanilla succ / coll / strict | PACT succ / coll / strict | Fisher coll / succ |
|---|---|---|---|
| invisible | 21/50 42% / 20/50 **40%** / 14/50 | 12/50 24% / 15/50 **30%** / 7/50 | p=0.40 / 0.088 |
| free | 21/50 42% / 24/50 48% / 12/50 | 15/50 30% / 22/50 44% / 8/50 | p=0.84 / 0.30 |
| visible | 20/50 40% / 34/50 68% / 5/50 | 14/50 28% / 28/50 **56%** / 6/50 | p=0.30 / 0.29 |

Collision Δ all favor PACT (−10 / −4 / −12 pts). None significant. Success worse every cell. Verdict unchanged: **failed**.
- **Not done:** none on this grid.

---

## 2026-08-24 — avoid-v1 declared FAILED; gate-bar (v3) data design + per-body collision metric prepared

- **When:** 2026-08-24 afternoon (America/Denver), after the n=50 avoid-v1 grid missed
  the ≥15 pt / p<0.05 bar (40%→30%, p≈0.40) and lost success (42%→24%).
- **Why:** User: collect new data that makes PACT win honestly; set up data, model
  config, and every experiment; MVP results needed by tomorrow. Root cause of both
  failed rounds, verified in sampler code: (1) avoid-v1 trained on *visible* bars;
  (2) `_obj_rest` coupled the cup to the bar's side, so cameras could read the bar off
  the cup even in the invis set; (3) bar face 0.14–0.24 m off-center meant one
  "always bow" path cleared every bar — the 3× upsample taught it to vanilla.
- **What (parent repo):**
  - `README.md` — §12.2 gate-bar recipe (design, Monte-Carlo numbers, preflight
    criteria, convert/train/eval deltas), §12 config-table row, §6 `--eval_sampler`
    flag row, §16 decision-log entries (avoid-v1 FAILED + gate design).
  - `paper.md` — avoid-v1 numbers moved into do-not-claim + failed-experiments with
    the honest wording. 2026-07-05 grid stays the headline.
  - `CURSOR.md` — this entry.
- **What (`submodules/molmospaces`, dirty):**
  - `tasks/enclosure_reach.py` — `GateObstacleFumehoodPickSampler` (+`Check`):
    OBSTACLE_P=0.75, INVIS_P=1.0, signed BAR_FACE_Y=(-0.06,0.22), GATE_X_WORLD=
    (0.47,0.58), AP_W_RANGE=(0.66,0.85), `_obj_rest` fully decoupled (cup y=±U(0.08,
    0.14)). Expert unchanged (strict superset; bows fit with 0% waypoint clipping in a
    20k-draw sim; blind-policy best case still hits 36% of poles, mean path 57%).
  - `tasks/pick_task.py` — `_accumulate_obstacle_diag` now records contact BODY NAMES
    per step (`_obstacle_diag_bodies`; world-rooted geoms named by geom), episode log
    line gains `bodies=...`. Additive; datagen/eval behavior otherwise unchanged.
  - `data_generation/config/object_manipulation_datagen_configs.py` —
    `FrankaSkinHybridGateBarCheckConfig` (4 eps, viz on) +
    `FrankaSkinHybridGateBarConfig` (8×25=200 eps, 4 workers, `viz_sensor_rgb=False`
    so no v2-style OOM) → `assets/datagen/hybrid_gate_bar_v1`.
- **What (`submodules/act`, dirty):**
  - `eval_act_obstacle.py` — `--eval_sampler {invis,gate}` picks which check sampler
    provides `--eval_cell`; per-episode records gain `hit_bar` / `bar_contact_steps` /
    `contact_bodies`; summary gains `bar_hit_rate` / `nonbar_collision_rate` /
    `mean_bar_contact_steps`; eval_summary.json records `eval_sampler`; wandb table
    extended. Old ckpts/records stay compatible (missing keys default).
  - `constants.py` — `obstacle_gate_v1` task (counts = 0 placeholders until convert
    prints them); avoid-v1 entry annotated with its failed result.
  - `imitate_episodes.py` — `obstacle_gate_v1` added to the state_dim=9/action_dim=8
    task tuple (else it falls to the 14-dim default and crashes).
- **How verified:** all files py_compile; samplers+configs import in `mlspaces`
  (sampler class, INVIS_P, viz flag, output dir checked); `_apply_eval_cell` swaps to
  `GateObstacleFumehoodPickCheckSampler` and pins per-cell probs; summary math checked
  on synthetic records; 20k-draw Monte-Carlo of the gate geometry (deflect 80% of bar
  eps, need p90 26 cm, corr(cup y, pole y)=+0.02).
- **Not done (user runs, workflow constraint):** gate-bar preflight + 200-ep collection,
  convert (+ paste counts into `constants.py`), train vanilla + PACT at chunk 50 (NO
  image dropout on the headline arm), eval n=50 invisible+free with
  `--eval_sampler gate`, `compare_pact.py` on collisions AND bar-hits. PACT-visible
  avoid-v1 eval may still be running in tmux — let it finish or kill it; its result
  cannot change the avoid-v1 verdict.

---

## 2026-08-24 — n=50 avoid-v1 eval: 5/6 cells done; last job PACT-visible

- **When:** 2026-08-24 ~13:13 America/Denver. Grid started 2026-08-23 21:10.
- **Why:** User asked for updates.
- **What (from `eval_summary.json`, n=50, `--temp_agg_off`):**

| cell | vanilla succ / coll / strict | PACT-raw per-sensor succ / coll / strict |
|---|---|---|
| **invisible** (paper) | 21/50 (42%) / **20/50 (40%)** / 14/50 | 12/50 (24%) / **15/50 (30%)** / 7/50 |
| free | 21/50 (42%) / 24/50 (48%) / 12/50 | 15/50 (30%) / 22/50 (44%) / 8/50 |
| visible | 20/50 (40%) / 34/50 (68%) / 5/50 | **still running** (~17/50 at 13:14) |

Invisible collisions 40%→30% is **10 pts**, Fisher p≈0.40. Does **not** meet ≥15 pts / p<0.05. Success 42%→24% (p≈0.088). Free collisions similar (48% vs 44%) so not a statue on the crash axis; lift still worse. Old 66%→40% grid stays the published headline until they decide otherwise.
- **How:** PACT jobs `use_proximity=true`, `raw`/`per_sensor` live. Vanilla `prox=False`. Invisible `INVIS_P=1`.
- **Not done:** `pact_raw_persensor_idrop03_s0_visible` (~2 h left). Do not write visible-cell PACT numbers yet.

---

## 2026-08-23 — user away ~10–12 h; n=50 grid in tmux; paper from July-5 numbers

- **When:** 2026-08-23 ~21:13 America/Denver.
- **Why:** User leaving overnight to write the paper. Eval loop started ~21:10.
- **What:** six jobs serial: ACT/PACT × invisible/free/visible. First job live:
  vanilla `act_avoid_s0_invisible`. Full grid ~18 h → at +12 h expect some
  `eval_summary.json` files, maybe not all six.
- **How / when back:** glob `eval_output/{act_avoid_s0,pact_raw_persensor_idrop03_s0}_{invisible,free,visible}/eval_summary.json`. Paper cell = both **invisible** n=50. Do not paste avoid-v1 rates into `paper.md` until those exist. Paper text tonight = 2026-07-05 66%→40% grid only.
- **Not done:** any n=50 avoid-v1 number.

---

## 2026-08-23 — smoke eval OK (n=2, not a result)

- **When:** 2026-08-23 20:57–21:04 America/Denver.
- **Why:** User ran 2-rollout PACT invisible smoke; asked to watch for completion/issues.
- **What:** `eval_output/smoke_pact_invisible/eval_summary.json`. Wiring good:
  `eval_cell=invisible`, sampler `InvisibleObstacleFumehoodPickCheckSampler`,
  `OBSTACLE_P=1 INVIS_P=1`, `[InvisBar] geom group 4`, PACT `raw` / `per_sensor` /
  40 sensors / K=1, `proximity ON (40,8,8)`, `--temp_agg_off`, full 201-step chunks
  (not frozen). 0/2 lift, collisions 1/2. n=2 = noise.
- **How:** Harmless log noise: `cam_visible=True` is a physics raycast not RGB;
  wrist reset mismatch; D405 depth-range on RGB-D mp4s; ffmpeg 470→480 pad.
- **Not done:** n=50 grid, both arms, three cells.

---

## 2026-08-23 — avoid-v1 train finished; eval is next

- **When:** 2026-08-23 ~20:49 America/Denver. Both 2000-epoch jobs done (serial in tmux).
- **Why:** User asked what is next after logs looked done.
- **What (verified on disk, not a result yet):**
  - vanilla: `ckpts/obstacle_pact_avoid_v1/20260823_183123_act_avoid_s0/`
    `policy_best.ckpt` = epoch **1914**, min val **0.0402**. ~59 min train.
  - PACT: `ckpts/obstacle_pact_avoid_v1/20260823_193046_pact_raw_persensor_idrop03_s0/`
    `policy_best.ckpt` = epoch **1850**, min val **0.0438**. ~64 min train.
    `prox_config.json`: `raw` / `per_sensor` / `min` / `n_sensors=40` / K=1.
    Last-epoch `l1_img_dropped` 0.069 vs `l1_clean` 0.039 → image dropout actually hurt the dropped samples (good).
- **How / next:** eval both ckpts, three cells, n=50, `--temp_agg_off --eval_cell`. Auto-detect PACT from `prox_config.json`. Never `imitate_episodes.py --eval`. ~3.5 min × 50 × 6 ≈ 18 h serial (~41 GB RSS). Paper cell first (`invisible`). Then `scripts/compare_pact.py`.
- **Not done:** any rollout numbers. Do not update `paper.md` tables until `eval_summary.json` exists.

---

## 2026-08-23 — paper.md (paper-agent claim fence)

- **When:** 2026-08-23 ~18:36 America/Denver.
- **Why:** User asked for the paper-writing brief as a file, easier to copy-paste.
- **What:** new `paper.md` (allowed claims, do-not-claim list, method, dataset, traps,
  2026-07-05 tables, in-flight avoid-v1 flagged as unmeasured). README routing row +
  intro pointer. This log header now cites `paper.md`.
- **How:** dump of the chat brief; numbers from `STATUS.md` / `PACT.md` / README §14.
  Not a paper draft.
- **Not done:** actual paper text. Avoid-v1 eval still pending.

---

## 2026-08-23 — user started avoid-v1 ACT + PACT train in tmux

- **When:** 2026-08-23 ~18:26 America/Denver.
- **Why:** Headline comparison on collision-aware set (`obstacle_pact_avoid_v1`, 151 eps).
- **What (user, not agent):** two `imitate_episodes.py` jobs in tmux under `submodules/act`:
  1. vanilla ACT — `--wandb_run_name act_avoid_s0` (no proximity)
  2. PACT — `--use_proximity --prox_feature raw --prox_layout per_sensor --image_dropout_p 0.3 --prox_dropout_p 0.1 --wandb_run_name pact_raw_persensor_idrop03_s0`
- **How:** ckpts land in `submodules/act/ckpts/obstacle_pact_avoid_v1/<YYYYMMDD_HHMMSS>_<runname>/`. If both `python` lines run in **one** pane, PACT waits until ACT finishes (~2000 epochs). Two tmux windows = both GPUs/jobs in parallel (only if VRAM allows; otherwise keep serial).
- **Not done:** eval. After `policy_best.ckpt` exists: `eval_act_obstacle.py --temp_agg_off --eval_cell invisible`, n=50, report collisions.

---

## 2026-08-23 — convert avoid-v1 counts pasted into constants

- **When:** 2026-08-23 ~18:23 America/Denver. User ran convert; agent pasted printed counts.
- **Why:** `imitate_episodes.py` reads `TASK_CONFIGS['obstacle_pact_avoid_v1']['num_episodes']`. Left at 0, train would see an empty set.
- **What:** `submodules/act/constants.py` → `num_episodes=151`, `episode_len=140`.
- **How / result:** User command (visible-bar `hybrid_obstacle_v1` `20260612_183855`, not the invis run):

```
python -m scripts.convert_obstacle_to_act \
    --src assets/datagen/hybrid_obstacle_v1/FrankaSkinHybridObstacleConfig/20260612_183855 \
    --dst act_style_data/obstacle_prox_avoid_v1 \
    --with_proximity --prox_pool min \
    --skip_approach_collision --keep_deflect_collisions --upsample_deflect 3 \
    --image_h 240 --image_w 320
```

`convert_meta.json`: fail skip 25, collision skip 13 (non-deflect scrapes), not_deflect 0. Written **deflect=96** (32 unique × 3) + **free=55** = **151**. max T=138 → episode_len=140. `prox_pool=min`.
- **Not done:** train vanilla ACT + PACT-raw-per_sensor; eval invisible cell n=50. Optional later: same convert on `hybrid_invis_obstacle_v1` if the camera-blind cell should be *in training* too.

---

## 2026-08-23 — PACT audit dump, collision-aware convert, encoder layout

- **When:** 2026-08-23 afternoon–evening (America/Denver). Follow-on to the
  same-day visualizer / CVAE-writeup / PACT-audit chat.
- **Why:** Success rate does not move because BC copies demos that already
  complete the pick with the bar present. The paper number is **invisible-cell
  collision rate**. Published win is `--prox_feature raw` (66% → 40%
  collisions, Fisher p = 0.016). CLI default `trunk` *raised* collisions
  (66% → 72%). Dataset also teaches scrapes: ~40% of episodes have approach
  (arm-vs-env) contacts; deflect demos still mean ~5 contacts. User asked to
  (1) dump canvases + audit into `PACT.md`, (2) fix those shortcomings, (3)
  introspect the dataset, (4) add collisions in a way that shows PACT beats
  ACT at avoidance — **without** injecting random crashes into BC.

### What landed (uncommitted)

Parent repo:

| file | change |
|---|---|
| `PACT.md` | **new.** Canvases (Safety-CVAE + PACT audit), dataset table, convert/train/eval commands, “incredible” success criteria. |
| `CURSOR.md` | **new (this file).** Agent change log. |
| `README.md` | Route to `PACT.md`. Recipe C pointer. Flag table: default `--prox_feature raw`, new `--prox_layout per_sensor`, recommend `--image_dropout_p 0.3`. Decision-log bullet 2026-08-23. Audit “next experiments” shortened to the three high-leverage runs. |
| `scripts/convert_obstacle_to_act.py` | Collision-aware convert (see How). |

`submodules/act` (dirty submodule, not committed):

| file | change |
|---|---|
| `prox_cvae.py` | `resolve_prox_layout`, `per_sensor` vs `global`, min/mean substep pool on live stack. Default tap `raw`. Encoder returns `(B, n_act_sensors, act_feat_dim)`. |
| `imitate_episodes.py` | Defaults `raw` + `per_sensor`. Hard-exit `--eval` + `--use_proximity`. Writes `prox_layout` / `prox_pool` / `n_proximity_sensors` into `prox_config.json`. Reads `convert_meta.json` for pool. Task list includes `obstacle_pact_avoid_v1`. |
| `eval_act_obstacle.py` | Rebuilds encoder from `prox_config.json` layout (missing key → `global`, so old ckpts still load). Live stack uses saved `prox_pool`. |
| `attn_heatmap.py` | Same layout rebuild as eval. |
| `detr/main.py` | No-op argparse for `--prox_layout` / `--prox_pool` so DETR re-parse does not explode. |
| `constants.py` | New task `obstacle_pact_avoid_v1` → `act_style_data/obstacle_prox_avoid_v1`, `num_episodes=0` until convert prints counts. |

### How (implementation)

**Do not teach colliding.** Convert does not invent crash labels. It *filters*
and *reweights* existing planner demos.

`convert_obstacle_to_act.py`:

- `--prox_pool {mean,min}` — `min` keeps a 1-of-4 substep graze (old path was mean, 4× dilution).
- `--skip_approach_collision` — drop eps with arm-vs-env contacts in pregrasp/grasp (same phase split as `analyze_obstacle_dataset.py`: phases `{2,3}`, inbound = before gripper-close phase 4). Tags come from `obs_scene` JSON (`behavior_class`, `collision_metrics.per_step_contacts`).
- `--keep_deflect_collisions` — exception: keep `behavior_class=deflect` even if they grazed. Needed because p10 of deflect contacts is 0; dropping *all* contacts would leave ~5 bows.
- `--only_deflect` / `--upsample_deflect N` — optional pure-avoidance set / copy each deflect hdf5 N times.
- Writes `convert_meta.json` (counts, flags, `prox_pool`) next to `episode_*.hdf5`.

`ProxCVAEEncoder`:

- `layout=global` (old published raw): `(B, 1, 40)`, `n_proximity_sensors=1`, K=8.
- `layout=per_sensor` (new default, `raw` only): `(B, 40, 1)`, `n_proximity_sensors=40`, K clamped 8→1 so 40 tokens do not drown ~160 image tokens.
- Live `stack_obs_proximity(..., pool=)` matches convert.

`imitate_episodes.py --eval --use_proximity` now `SystemExit`. That path never passed `proximity_positions`. Real eval is `eval_act_obstacle.py --temp_agg_off`.

Old PACT ckpts: no `prox_layout` in `prox_config.json` → eval defaults `global` + `n_proximity_sensors=1`. Compatible.

### Dataset facts used (already on disk)

Source analysis: `diagnostics_output/obstacle_analysis/` from
`scripts/analyze_obstacle_dataset.py` on `assets/datagen/hybrid_obstacle_v1`
(151 eps).

| fact | value |
|---|---|
| Bar present | 75% (113 / 151) |
| Deflect / free | 49 / 102 (only 43% of bar eps actually bow) |
| Lateral bow, bar-deflect | mean 3.8 cm |
| Approach collision | **40% of episodes** |
| Approach contacts, bar-deflect | mean **5.0** |
| Skin close (<0.10 m), bar vs no-bar | 86% vs 74% (ambient saturation) |
| `act_style_data/obstacle_prox_v2` | still on disk, 105 eps (old train set) |

### Not done (user runs)

Convert / train / eval were **prepared, not executed** (repo workflow: agent
edits, user runs).

1. Convert avoid set (`PACT.md` §5). Paste `num_episodes` / `episode_len` into
   `constants.py` `obstacle_pact_avoid_v1`.
2. Train vanilla ACT and PACT-raw-per_sensor + `--image_dropout_p 0.3` on that set.
3. Eval n=50, `--temp_agg_off`, cells `invisible` / `free` / `visible`. Report
   **collisions**, not just lift-success.
4. Optional later: more **clean** deflect datagen (`approach_contacts=0`),
   `FrankaSkinHybridInvisObstacleConfig`. Residual `SafetyHead` at eval is a
   **different method** (ACT+reflex), not PACT.

`submodules/molmospaces` shows dirty in `git status`. Not part of this PACT
code pass (earlier visualizer work). Do not mix it into a PACT commit.

Nothing committed. No push.

---

## 2026-08-23 (earlier same day) — visualizer plates, CVAE write-up, PACT audit

- **When:** earlier 2026-08-23, before the `PACT.md` / convert-filter pass.
- **Why:** README needed the new visualizer behavior and paper-style plates.
  User then asked for a full CVAE description and a PACT wiring audit for
  “why no success-rate gain.”
- **What (already in tree / README before this log existed):**
  - `submodules/molmospaces/scripts/datagen/visualize_environment.py` — text-free
    1920×1080 plates, scored cameras, presentation headlight, `--attempts`,
    `--format both`.
  - README §12 hero/preview images under
    `experiments_output/default/environment_viz/...` (pngs gitignored; `git add -f`
    to publish).
  - README §10 Safety-CVAE tensors; §13 audit “why PACT success does not move.”
  - Canvases:
    `~/.cursor/projects/home-jaydv-code-prox-learning/canvases/pact-audit.canvas.tsx`
    and `safety-cvae-explained.canvas.tsx`.
- **How:** audit conclusion was wiring OK (metres, one featurize, no z-score on
  skin, sensor order from `cvae_v3/meta.json`). Method fights data: wrong
  metric, default `trunk` negative control, CVAE ⊥ BC, 40 sensors mashed to 1
  vector, one skin frame vs 100-step chunk, image dropout 0, temporal agg
  washes newest skin, mean-pool of 4 substeps.
- **Not done then:** `PACT.md` dump, convert filters, `raw`/`per_sensor`
  defaults — those are the block above.

---

## Backlog (do not lose)

- Paste convert counts into `obstacle_pact_avoid_v1` after first convert.
- Headline train: `--use_proximity --prox_feature raw --prox_layout per_sensor --image_dropout_p 0.3 --prox_dropout_p 0.1`.
- Eval: `eval_act_obstacle.py --temp_agg_off --eval_cell invisible`, n=50.
- Collect ≥150 clean deflect eps if reconverted set is too small.
- Optional third arm: ACT + residual SafetyHead at eval (not PACT).
- Do not commit `PACT.md` / `CURSOR.md` / act submodule together with unrelated
  molmospaces visualizer dirt unless that is the intended PR.
