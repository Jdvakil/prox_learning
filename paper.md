# Paper-writing brief — prox_learning / P+ACT

**Paste this whole file into a paper-writing agent.** It is a claim fence, not a draft.
Science lives in `STATUS.md` and `PACT.md`. How to run lives in `README.md`. This file
tells the writer what is true, what is in flight, and what must not be invented.

**Date of brief:** 2026-08-23. Published headline numbers are from **2026-07-05**.

---

## 1. Attach these files (in this order)

1. This file (`paper.md`).
2. `STATUS.md` — story, tables, blur failure, why n=50.
3. `PACT.md` — method traps, Safety-CVAE vs PACT, dataset facts, what a “win” means.
4. `README.md` §10 (CVAE tensors), §13–16 (eval cells, numbers, traps, decision log).
5. `reports/2026-08-14/report.md` and its pngs — already-written figures.
6. `CURSOR.md` — what is **in flight** vs published. Do not treat it as a result.
7. After eval: each `eval_output/*/eval_summary.json` and counts from `scripts/compare_pact.py`.

Do **not** use `paper/section3_proximity_signal_draft.md` as source of truth (it cites deleted scripts).

GitHub will **not** contain checkpoints, `act_style_data/`, or most datagen. The writer cannot
train from a clone. Do not wait for weights to start the paper.

---

## 2. One-sentence claim (the only headline)

A full-body proximity skin, fused as raw per-sensor closeness into an ACT imitation policy
(**PACT-raw**), cuts **collision rate** from **66% to 40%** (n=50, Fisher p = 0.016) when a
hazard bar is physically present but **hidden from the cameras**. Task **lift-success is
unchanged** (noise). The claim is **safer when cameras cannot see the obstacle**, not a
better picker.

Tone: safer when cameras fail. Not SOTA pick-and-place.

---

## 3. Suggested paper spine

1. Enclosure reach + camera occlusion. Skin = 40 ToF-like cones, not vision.
2. Imitation ACT; three eval cells; causal test = invisible.
3. Result: collisions drop only where cameras are blind; success flat; trunk fails; raw works.
4. Ablation: trunk negative control; free-cell control; strict success vs statue policy.
5. Honest limitations: blunt metric, one seed, BC copies scrapes, CVAE ⊥ BC.
6. Related work: ACT (Zhao et al.), tactile/proximity skins, CVAE safety as **reflex not
   representation**.

---

## 4. Claims you may write **now** (2026-07-05 grid, n=50)

**Question.** Does a full-body proximity skin make a vision-based imitation policy safer
when cameras cannot see the obstacle?

**Not** “better picker.” Lift-success is designed to be flat because the expert already
completes the pick with the bar present. Headline metric = **invisible-cell collision rate**.

### Collisions (lower is better)

| policy | free | invisible | visible |
|---|---|---|---|
| vanilla ACT (RGB + qpos) | 60% | **66%** | 64% |
| PACT-raw (same + 40 peak closeness) | 58% | **40%** | 50% |
| PACT-trunk (frozen CVAE decoder tap) | 64% | **72%** | 58% |

### Task success (higher is better; all diffs are noise)

| policy | free | invisible | visible |
|---|---|---|---|
| vanilla ACT | 22% | 36% | 28% |
| PACT-raw | 18% | 30% | 16% |
| PACT-trunk | 34% | 34% | 32% |

- Fisher on invisible collisions 66% vs 40%: **p = 0.016**. Success diffs p ≥ 0.23.
- Strict success (lift **and** contact-free), invisible: raw **20%** vs vanilla **14%**.
- Graded benefit **2 / 14 / 26** points (free / visible / invisible) is the causal pattern.
- Vanilla does **not** avoid even when the bar is visible (64% vs 66%). 105 demos never
  taught visual avoidance, so the skin is not a backup sense — it is the only avoidance
  signal in any trained policy.
- Free cell ~unchanged (60% → 58%, p = 1.0) → not a statue / general freeze.
- Trunk is **worse** than vanilla on the causal cell. Do not sell the Safety-CVAE as the
  PACT encoder.

**Setup.** Franka FR3 in MuJoCo. **40** proximity sensors, each **8×8** planar-z metres,
45° cones. Two RGB cameras `exo_camera_1`, `wrist_camera` at 240×320. Policy = ACT
(chunk size 100, hidden dim 512, feedforward 3200, KL weight 10). Task = fumehood cup
pick; bar present ~75% of training episodes (`OBSTACLE_P=0.75`).

**Eval protocol (only valid one).** `eval_act_obstacle.py --temp_agg_off --eval_cell
{invisible,free,visible}`, n=50. Never `imitate_episodes.py --eval` (that path does not
pass skin). Invisible bar = MuJoCo geom group 4: camera renderer skips it, skin renderer
includes it. Physics is unchanged.

| cell | bar present | cameras see bar | skin feels bar | what it tests |
|---|---|---|---|---|
| `visible` | yes | yes | yes | ordinary case |
| `invisible` | yes | **no** | yes | **causal / paper cell** |
| `free` | no | — | — | background brushing; statue check |

Compare rates with `scripts/compare_pact.py` (Wilson CI, rate difference, Fisher exact).
You type counts; it reads no files. Example:
`vanilla=18/50,33/50 pact_raw=15/50,20/50` (success/N, collisions/N).

---

## 5. Do **not** claim (until a new `eval_summary.json` says otherwise)

- Avoid-v1 / per-sensor / image-dropout training (tmux, 2026-08-23) **has no numbers yet**.
- Multi-seed. Published grid = **one seed, one dataset**.
- Real robot / hardware skin. Sim only.
- PACT uses a trained CVAE **encoder** at runtime. Runtime latent `z = 0`. The CVAE
  encoder `q(z | skin, dq)` is **train-only** (needs the retreat target). The PACT arm
  that worked **bypasses** the CVAE: `--prox_feature raw` = per-sensor peak closeness.
- Safety-CVAE is a skin autoencoder. It never reconstructs 2560 depth pixels. It
  reconstructs a **7-DoF joint retreat `dq`**. Skin is the *condition*.
- ACT + residual `SafetyHead` at eval = a **different method** (“ACT+reflex”). Not PACT.
- Train-time camera blur as evidence that cameras fail. That sweep is **null**; n=25
  noise is ±40 points. Retract any “blur ladder” story.
- Collision counter = “hit the bar.” It counts **any arm–environment contact** except
  floor and grasped cup. Background (no bar) is ~**60%**. The bar itself adds only
  ~**4–6** points. A 26-point drop is larger than the bar increment, so the skin is
  likely making the arm **generally more careful in the cavity**, strongest where the
  skin fires. Say that honestly.
- v1 PACT “worked.” Round 1 (2026-06-18) tied vanilla. Formal probe gate on
  bar-presence **failed** on v2 (chance). Deflection became decodable from raw (~0.75).
- Any `--temp_agg_off` number from **before 2026-07-04**. Invalid (arm froze ~30 cm short).
- Injecting crashes into behavior cloning. Convert **filters and upweights** existing
  bows; it does not invent collisions.

If avoid-v1 PACT does not beat vanilla by ≥15 collision points, Fisher p < 0.05, **keep
the 2026-07-05 table** as the result and treat per-sensor/dropout as a follow-up. Do not
overwrite 66→40 with a failed rerun.

---

## 6. Method (get the tensors right)

**PACT** = ACT + proximity tokens in the transformer memory:
`[latent z, qpos, prox tokens, ~160 image tokens]`.

**Published win (use this in the paper unless a later eval beats it):**

- `--prox_feature raw`
- **global** mash: 40 sensors → one 40-d vector → 8 anonymous tokens
- `n_proximity_sensors = 1`

**In flight (not a result — do not write as if measured):**

- `--prox_layout per_sensor` (40 tokens of dim 1; K clamped 8→1)
- `--image_dropout_p 0.3 --prox_dropout_p 0.1`
- convert `--prox_pool min --skip_approach_collision --keep_deflect_collisions
  --upsample_deflect 3`
- dataset `obstacle_pact_avoid_v1`: **151** episodes (96 deflect copies = 32 unique × 3,
  plus 55 free), `episode_len=140`

### Safety-CVAE vs PACT wrapper (two different “encoders”)

| | Job A — reflex head | Job B — frozen PACT wrapper |
|---|---|---|
| Who | `SafetyHead` / `scripts/safety_*_demo.py` | `submodules/act/prox_cvae.py` `ProxCVAEEncoder` |
| In | `(40, 8, 8)` metres | `(B, 40, 8, 8)` metres |
| Out | `(7,)` rad retreat | 1 feature, then K ACT tokens |
| CVAE encoder run at policy time? | **no** (`z = 0`) | **no** (decoder trunk / delta / raw) |

Closeness: `clip(1 − d / 0.5)`; dead range `< 5 mm → 0`. Decoder taps: `trunk` 256
(old CLI default, **negative control**), `delta` 7, `raw` 40 (skips CVAE). `cvae_v3`
ended with ~1 of 8 latent dims alive. The “C” is mostly a train regulariser.

Sensor order is `assets/safety/cvae_v3/meta.json`. `link5_back` precedes `link5_front`.
Never use the env’s `_HYBRID_SKIN_SENSOR_NAMES` tuple.

Wiring that was audited clean: metres stay metres; featurize once; `dataset_stats` never
z-scores skin; convert and live eval both stack in `cvae_v3` order.

### Why success stays flat (method vs data, not a shape bug)

1. Wrong metric: BC copies demos that succeed *with the bar present*.
2. CLI default `trunk` is a retreat prior orthogonal to BC; it *raised* collisions 66%→72%.
3. 40 sensors mashed to 1 vector then 8 anonymous tokens (published raw).
4. One skin frame conditions a 100-step action chunk under uniform L1.
5. `--image_dropout_p` default 0; ~160 image tokens vs 8 prox; vision can fit demos alone.
6. Mean-pool of 4 skin substeps dilutes a 1-substep graze 4× (avoid-v1 convert uses min).

Do **not** freeze the Safety-CVAE as the policy encoder for the headline method. Use `raw`.
Keep `trunk` / `delta` as negative controls.

---

## 7. Dataset (train is not “clean avoidance”)

The expert is a **scripted planner that cheats** (knows geometry). BC copies it.

Source analysis (`scripts/analyze_obstacle_dataset.py` on `assets/datagen/hybrid_obstacle_v1`,
151 episodes → `diagnostics_output/obstacle_analysis/`):

| fact | value |
|---|---|
| Bar present | 75% (113 / 151) |
| `behavior_class` | 49 deflect / 102 free (only 43% of bar episodes actually bow) |
| Lateral bow, bar-deflect | mean **3.8 cm** (p90 7.1 cm) |
| Lateral bow, bar-free | mean **0.5 cm** |
| Skin close (`< 0.10 m`) on bar / no-bar | **86%** / **74%** (ambient saturation) |
| Approach (arm-vs-env) collision | **40% of episodes** |
| Approach contacts, bar-deflect | mean **5.0** |
| Task success in source | 81% (convert already drops `fail[-1]`) |

Published PACT train set (`act_style_data/obstacle_prox_v2`): **105** demos
(47 visible / 49 hidden / 29 none). Prediction error: raw **0.0595** vs cameras 0.0755
vs trunk 0.0830.

Do not write “clean collision-free avoidance demonstrations.”

Avoid-v1 convert (2026-08-23, not yet evaluated): drop non-deflect inbound scrapes, keep
deflect even if they grazed, 3× upsample bows, min-pool skin. Unique kept ≈ 87 from 125
source trajs. **Do not** drop every approach contact — that would leave ~5 bows.

---

## 8. Hardware / sensing facts (for intro)

- Skin: 40 sensors, 8×8 depth, 45° cone (`model_hybrid.xml`).
- Directional coverage: **83%** of directions vs **10%** for the wrist camera.
- Skin accuracy (sensor proofs): linear response; 5.6 mm error reconstructing a pipe.
- Skin under blur / darkness: readings **bit-for-bit identical**; cameras collapse.
- Reflex net (honest split): direction accuracy 0.924; magnitude 69% worse than the
  cheating split; when obstacle is closest, outputs only 64% of needed size. Lookup
  table ties direction (0.923).
- Skin under a pick-and-place visibility constraint adds ~nothing (`proximity_necessity.py`);
  that is why the project pivoted to enclosures.

v1 → v2: 29-sensor 4-link skin replaced by 40-sensor hybrid. Anything citing
`LINKS = 2/3/5/6` is pre-pivot.

---

## 9. Limitations / reviewer ammo (put in the paper)

1. **One seed, one dataset.** Unresolved whether the main result holds with another seed.
2. **Blunt collision metric.** Cannot tell “rammed the bar” from “brushed the cavity wall.”
   High fixture-brush floor (~60%). A per-body counter is still missing.
3. **Sim only.** Invisible cell is a **renderer privilege** (geom group 4).
4. **Demos subtract an obstacle-parked baseline that PACT cannot use** (README trap 4).
   In 100% of demo frames some fixture is within `D_MAX = 0.5 m`; 40–60% of timesteps sit
   inside `D_ACT = 0.18 m` while the demonstrated action is “proceed into the cavity.”
   `corr(‖delta‖, min_depth) ≈ −0.7` on successful demos, so BC learns to ignore retreat.
5. **One skin frame, 100-step chunk.**
6. **n = 25 is inside the noise band** (±40 points, measured on the blur grid). Floor for
   any claim is **n = 50**.
7. **Low collisions can mean broken**, not careful (blur σ=2: fewest collisions, almost
   never finishes). Always report **strict success**.
8. **Training loss does not predict behaviour** (blur training-error ladder vs null robot
   behaviour). Never ship a behavioural claim from wandb alone.
9. Still missing for a strong submission: second seed; per-body “hit bar vs hit wall”;
   optional **test-time** camera degrade with skin held fixed (tool exists; not the
   headline yet). Train-time blur is not that experiment.

---

## 10. Failed / retracted experiments (mention as negative results, not as support)

**PACT v1 (2026-06-18).** Nothing beat cameras-only (p = 0.76). Blanking the entire skin
moved the predicted chunk by ~0.005. Causes: policies ignore the token, ambient
saturation, temporal-aggregation washout, signal-to-horizon mismatch.

**Probe gate.** v1: deflection not decodable (chance from trunk, raw, and qpos). v2:
bar-presence collapsed to chance (0.40–0.52; v1’s 0.72–0.78 was largely a placement
leak). Deflection *became* decodable from raw40 (0.749–0.763) and survived a qpos
control. Formal bar-presence gate still FAILED; training on v2 was a judgment call.

**Trunk arm (scrapped 2026-07-06).** Inert or worse (invisible collisions 72%). Consistent
with v2 probes and v1 attention audit.

**Train-time blur sweep (2026-07-24 → 2026-08-10).** Three constant-blur vanilla arms,
225 rollouts, 13.4 h. Training error: 0.0755 / 0.0836 / 0.0948 / 0.1100 (σ = 0/2/4/8).
Behaviour: no ladder. σ=2 ≈ statue (0% hidden success). σ=4 recovered success and
*raised* collisions. σ=8 free-cell collisions 28% vs 68% with a bar — same policy, so
that 40-point swing is **noise at n=25**. Two of nine tests crossed p=0.05 in opposite
directions. Keep the lessons; drop the blur-as-camera-failure claim.

---

## 11. Eval traps the writer must not launder into the paper

1. Temporal aggregation on (default `m=0.01`): newest chunk (the only one that saw
   *current* skin) carries ~1.6% of the executed action. Structurally mutes reactive
   avoidance. Paper evals use `--temp_agg_off` **after** the 2026-07-04 fix (open-loop
   chunking, not execute-`chunk[0]`-every-step).
2. `imitate_episodes.py --eval` never passes `proximity_positions`. Current code
   `SystemExit`s if `--use_proximity` is set.
3. `viz_sensor_rgb` OOM if left on; eval forces it off.
4. `scene_params["cell"]` is not a label on obstacle runs (always `"bar"`).
5. Test throughput ~3.5–3.6 min/rollout; ~8 GB + 0.5 GB/episode. A 50-rollout cell is
   ~3 hours.

---

## 12. After the 2026-08-23 tmux train finishes

Train (user, serial in one pane unless split):

```bash
# A. vanilla ACT
python imitate_episodes.py --task_name obstacle_pact_avoid_v1 --policy_class ACT \
    --ckpt_dir ckpts --kl_weight 10 --chunk_size 100 --hidden_dim 512 \
    --dim_feedforward 3200 --batch_size 8 --lr 1e-5 --seed 0 --num_epochs 2000 \
    --wandb_run_name act_avoid_s0

# B. PACT-raw per-sensor + vision dropout
python imitate_episodes.py --task_name obstacle_pact_avoid_v1 --policy_class ACT \
    --ckpt_dir ckpts --kl_weight 10 --chunk_size 100 --hidden_dim 512 \
    --dim_feedforward 3200 --batch_size 8 --lr 1e-5 --seed 0 --num_epochs 2000 \
    --use_proximity --prox_feature raw --prox_layout per_sensor \
    --image_dropout_p 0.3 --prox_dropout_p 0.1 \
    --wandb_run_name pact_raw_persensor_idrop03_s0
```

Ckpts: `submodules/act/ckpts/obstacle_pact_avoid_v1/<datetime>_<runname>/`.

Eval (both arms, three cells). Feed **new** counts only from `eval_summary.json`:

```bash
python eval_act_obstacle.py \
    --ckpt_dir ckpts/obstacle_pact_avoid_v1/<dated>_<arm> \
    --output_dir /home/jaydv/code/prox_learning/eval_output/<arm>_<cell> \
    --num_rollouts 50 --chunk_size 100 --temp_agg_off --eval_cell <cell>
```

What “incredible” would mean for a *new* table (do not write this until measured):

- Invisible-cell **collision**: PACT-raw per-sensor + image dropout beats vanilla by
  ≥15 points, n=50, Fisher p < 0.05. (Already 26 points with old global-raw — need that
  to hold after the data filter, or grow.)
- Invisible-cell **strict success** not worse than vanilla (no statue).
- Free-cell collisions similar.
- Ablations: `trunk` ≥ vanilla collisions; dropout should help vs no dropout.

Optional later third method, **not PACT**: residual `SafetyHead` on vanilla ACT at eval.

---

## 13. Related-work positioning (do not overclaim)

- **ACT** is off-the-shelf. Nothing about the transformer is ours except the proximity
  token path (`input_proj_proximity`, extra positional embeddings). Only those (plus
  dropout) train when proximity is on; the Safety-CVAE is frozen.
- Position against: wrist-only tactile, vision-only IL in clutter, learned safety
  filters / CVAEs that output joint retreat. Our CVAE *works as a reflex* and *fails as
  a PACT feature*. That contrast is a result, not a footnote.
- Do not cite deleted `pla/audit_proximity.py`. Sensor-proof numbers live in README §14
  and `STATUS.md`.

---

## 14. Copy-paste stats one-liner

> On a Franka FR3 in MuJoCo with 40 full-body proximity sensors, an ACT policy trained
> from 105 scripted enclosure-pick demonstrations collides in 66% of rollouts (n=50)
> when a hazard bar is hidden from the cameras. Adding a raw-skin token (PACT-raw) cuts
> that to 40% (Fisher p = 0.016) without changing lift-success. A frozen Safety-CVAE
> “retreat” embedding (PACT-trunk) does not help (72%). The camera-only policy does not
> avoid the bar even when it is visible (64% vs 66%). Background contact with no bar is
> ~60%; the collision counter cannot yet separate bar hits from cavity brushes. One
> seed. Sim only.

---

## 15. File map for the writer (do not dump code)

| path | role |
|---|---|
| `submodules/act/imitate_episodes.py` | train |
| `submodules/act/eval_act_obstacle.py` | **only** evaluator |
| `submodules/act/prox_cvae.py` | `ProxCVAEEncoder` |
| `submodules/act/detr/models/detr_vae.py` | ACT + prox tokens |
| `scripts/convert_obstacle_to_act.py` | datagen → ACT hdf5 |
| `scripts/analyze_obstacle_dataset.py` | source-set stats |
| `scripts/compare_pact.py` | Fisher / Wilson |
| `scripts/train_safety_cvae.py` | reflex CVAE |
| `assets/safety/cvae_v3/` | frozen CVAE weights + `meta.json` |
| `reports/2026-08-14/` | figures already made |
| `scripts/figures.py --list` | more paper figures |
