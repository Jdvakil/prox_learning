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
