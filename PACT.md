# PACT — why success is flat, and how to beat ACT on collision avoidance

This file is the working note for **P+ACT** (ACT plus a proximity-skin token).
It dumps the two canvases from 2026-08-23, the wiring audit, the dataset facts,
and the exact next runs. The repo manual stays in [`README.md`](README.md).
Recipe C still lives in README §13; this file is the *why it stalled* and *what
to change* companion.

**Punchline.** The tensor path is fine. Success stays flat because the method
fights the data. The paper number is **invisible-cell bar-hit**, not
lift-success.

**Now (2026-08-24 late):** gate-bar paused. Reproduce coauthor
`data/pact_place_corridor_v5` (152 recovered pick-and-place rows). Convert +
train vanilla vs PACT-raw per-sensor, chunk 50, no image dropout. Eval needs
molmospaces worktree at `977acd6` (`pact_place_corridor_v2.xml`). Do not write coauthor win
numbers into `paper.md` until local `eval_summary.json` exists.

The July 66%→40% grid is a published number on deleted data; avoid-v1 failed.
Gate-bar v3.1 (visible check first) stays parked.

The CLI default `--prox_feature trunk` is a **negative control**.

---

## 1. Safety-CVAE — what gets encoded (canvas)

The Safety-CVAE is **not** a skin autoencoder. It never rebuilds the 2560 depth
pixels. It rebuilds a **7-DoF joint retreat** `dq`. Skin is the *condition*.
Latent `z` is a train-time dodge knob, then pinned to `0`.

| | Job A — reflex head | Job B — frozen PACT encoder |
|---|---|---|
| Who | `scripts/safety_*_demo.py` via `SafetyHead` | `submodules/act/prox_cvae.py` `ProxCVAEEncoder` |
| In | `(40, 8, 8)` metres | `(B, 40, 8, 8)` metres |
| Out | `(7,)` rad | 1 feature, then K ACT tokens |
| Encoder run? | **no** (`z = 0`) | **no** (decoder trunk / delta / raw) |

Two different words named "encoder":

- **CVAE encoder** (`SafetyCVAE.enc`) — train only. `q(z | skin, dq)`. Needs the
  target retreat. Cannot run while the arm is acting. Output is an 8-d Gaussian,
  not a skin embedding.
- **PACT wrapper** (`ProxCVAEEncoder`) — runtime. Frozen **decoder** at `z = 0`.
  Taps: `trunk` 256 (old default), `delta` 7, or `raw` 40 (skips CVAE).

### Every tensor

| name | shape | units | where |
|---|---|---|---|
| raw SPAD depths | `(B, 40, 8, 8)` | metres | env / h5 |
| closeness `x` | `(B, 2560)` | `[0, 1]` | `clip(1 − d/0.5)`, dead `<5 mm → 0` |
| encoder in (train only) | `(B, 2567)` | concat(`x`, `dq̃`) | 2560 + 7 |
| `μ`, `log σ²` | `(B, 8)` each | latent posterior | CVAE encoder |
| `z` train / runtime | `(B, 8)` | sample / **zeros** | runtime pins `z = 0` |
| decoder in | `(B, 2568)` | concat(`x`, `z`) | 2560 + 8 |
| decoder trunk | `(B, 256)` | SiLU hidden | old PACT default tap |
| `dq̂̃` then `dq` | `(B, 7)` | scaled, then `× σ≈11.36` rad | reflex / delta tap |
| ACT prox tokens (global) | `(K, B, hidden)` | `K=8`, `hidden=512` | `input_proj_proximity` |
| ACT memory | ~170 tokens | 1 latent + 1 proprio + 8 prox + ~160 image | two 240×320 cams |

`cvae_v3` ended with **1 of 8** latent dims alive. The "C" is mostly a train
regulariser.

---

## 2. PACT audit — why success does not move (canvas)

### What is actually wired

```
h5 /observations/proximity     (T, 40, 8, 8) metres
        │  one random frame per ACT sample
        ▼
ProxCVAEEncoder.featurize      (B, 2560) closeness, once
        │  old default tap = trunk  → (B, 1, 256)
        │  useful tap        = raw   → (B, 1, 40)   or per-sensor (B, 40, 1)
        ▼
n_proximity_sensors = 1        Linear(feat → K·hidden) → K tokens
        ▼
encoder memory                 [z, qpos, 8 prox, ~160 image]
```

Checked clean: metres stay metres; featurize once; `dataset_stats` never
z-scores skin; convert and live eval both stack by `HYBRID_SKIN_SENSOR_ORDER`
(`link5_back` before `link5_front`).

### Findings, highest leverage first

| rank | finding | effect |
|---|---|---|
| 1 | **Wrong metric.** BC copies demos that succeed *with the bar present*. | Lift-success is designed to be flat. Report **invisible-cell collisions**. |
| 2 | **CLI default `trunk` is a negative control.** | Trunk *raised* invisible collisions 66%→72%. Raw cut them to **40%**. |
| 3 | **Safety-CVAE objective ⊥ BC.** Retreat labels vs "keep going". | `‖delta‖` anti-correlates with min depth (~−0.7) on *successful* demos. BC ignores the token. |
| 4 | **40 sensors → 1 vector → 8 anonymous tokens.** | No per-link identity in the transformer. |
| 5 | **One skin frame, 100-step action chunk.** Uniform L1. | Late actions barely depend on t=0 closeness. |
| 6 | **`--image_dropout_p` default 0.** ~160 image tokens vs 8 prox. | Vision fits the demos alone. v1 blanking moved the chunk by ~0.005. |
| 7 | **`imitate_episodes.py --eval` never passes `proximity_positions`.** | Not a PACT eval. Use `eval_act_obstacle.py --temp_agg_off`. |
| 8 | **Mean-pool of 4 substeps.** | A 1-substep graze is diluted 4×. |

Published grid (n=50, invisible cell):

| arm | success | collisions | note |
|---|---|---|---|
| vanilla ACT | 36% | **66%** | cameras + qpos |
| PACT `raw` | 30% | **40%** | Fisher p = 0.016 vs vanilla collisions |
| PACT `trunk` | — | **72%** | worse than vanilla |

Success 36% vs 30% is noise. Collisions are the story.

---

## 3. Dataset introspection (already measured)

Source: `python scripts/analyze_obstacle_dataset.py --root assets/datagen/hybrid_obstacle_v1`
→ `diagnostics_output/obstacle_analysis/summary.json` (151 episodes).

| fact | value | why it matters |
|---|---|---|
| Bar present | 75% (113 / 151) | matches `OBSTACLE_P=0.75` |
| `behavior_class` | 49 deflect / 102 free | only **43% of bar episodes actually bow** |
| Lateral bow, bar-deflect | mean **3.8 cm** (p90 7.1 cm) | this is the avoidance motion BC should copy |
| Lateral bow, bar-free | mean **0.5 cm** | almost a straight line — skin has nothing to imitate |
| Skin close (<0.10 m) on bar eps | **86%** | bar *does* hit the skin |
| Skin close on no-bar eps | **74%** | fixtures also light up the skin (`D_MAX=0.5 m`) |
| Min approach depth, bar vs no-bar | 6.3 cm vs 7.5 cm | tiny gap — ambient saturation |
| **Approach (arm-vs-env) collision** | **40% of episodes** | BC is also copying *scrapes* |
| Approach contacts, bar-deflect | mean **5.0** | "deflect" demos still rub the bar |
| Approach contacts, bar-free | mean **0.47** | |
| Task success | 81% overall | filter already drops `fail[-1]` |

So the training set is **not** "clean avoidance plus a hidden bar". It is:

1. mostly straight-in picks (free),
2. a minority of bows that still contact (~5 contacts/ep),
3. skin that fires on walls/sash even with no bar.

You cannot get "incredible" collision-avoidance from imitating (1)+(2) unless
you **filter and upsample**, then **force the policy to use skin**.

`act_style_data/` was deleted 2026-08-16. Retrain needs a reconvert (commands
below). Checkpoints under `ckpts/` still exist for the published numbers.

---

## 4. What we are changing in code

| change | where | why |
|---|---|---|
| Default `--prox_feature raw` | `imitate_episodes.py` | only tap that ever beat ACT |
| `--prox_layout {global,per_sensor}` | encoder + DETRVAE | 40 named tokens, not one mashed vector |
| Hard-stop `imitate_episodes.py --eval` when proximity is on | train entry | that path never feeds skin |
| Convert `--skip_approach_collision` | `convert_obstacle_to_act.py` | stop teaching scrapes |
| Convert `--upsample_deflect N` | same | overweight the 49 bows |
| Convert `--only_deflect` | same | optional pure-avoidance set |
| Convert `--prox_pool {mean,min}` | same + live `stack_obs_proximity` | min keeps the graze |

Do **not** freeze the Safety-CVAE as the policy encoder for the headline run.
Use `raw` (or per-sensor raw). Keep `trunk` / `delta` as negative controls.

A residual `SafetyHead` *at eval* (ACT for the pick, CVAE for joint retreat) is
a **different method**. It is the right use of the CVAE. Track it as "ACT+reflex",
not PACT.

---

## 5. Commands — rebuild data, train, eval

`act_style_data/obstacle_prox_v2` is still on disk (105 eps). Rebuild a **collision-aware** set for the avoidance claim. **Do not drop every deflect graze** — that leaves ~5 episodes. Keep deflect even if they rubbed, drop free/no-bar scrapes.

```bash
conda activate mlspaces
cd /home/jaydv/code/prox_learning

# 0. Re-print the audit (optional; already in diagnostics_output/)
python scripts/analyze_obstacle_dataset.py \
    --root assets/datagen/hybrid_obstacle_v1 \
    --out diagnostics_output/obstacle_analysis

# 1. Convert: successes only, drop non-deflect inbound scrapes, 3× the deflect bows,
#    min-pool the 4 skin substeps, keep proximity.
# DONE 2026-08-23: 151 eps (96 deflect = 32×3, 55 free), episode_len=140, prox_pool=min.
# Skipped fail=25 collision=13. Counts already in TASK_CONFIGS['obstacle_pact_avoid_v1'].
python -m scripts.convert_obstacle_to_act \
    --src assets/datagen/hybrid_obstacle_v1/FrankaSkinHybridObstacleConfig/20260612_183855 \
    --dst act_style_data/obstacle_prox_avoid_v1 \
    --with_proximity --prox_pool min \
    --skip_approach_collision --keep_deflect_collisions --upsample_deflect 3 \
    --image_h 240 --image_w 320
```

Invisible-bar v2 source, if you want the camera-blind cell in the *training*
distribution too:

```bash
python -m scripts.convert_obstacle_to_act \
    --src assets/datagen/hybrid_invis_obstacle_v1/FrankaSkinHybridInvisObstacleConfig/20260703_095653 \
    --dst act_style_data/obstacle_prox_invis_avoid_v1 \
    --with_proximity --prox_pool min \
    --skip_approach_collision --keep_deflect_collisions --upsample_deflect 3
```

Train **two** arms on the same converted set (byte-identical RGB/qpos/action):

```bash
cd /home/jaydv/code/prox_learning/submodules/act
export PYTHONPATH="$PWD:$PYTHONPATH"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl

# A. vanilla ACT (no skin) — the thing we must beat on collisions
python imitate_episodes.py \
    --task_name obstacle_pact_avoid_v1 --policy_class ACT \
    --ckpt_dir ckpts --kl_weight 10 --chunk_size 100 \
    --hidden_dim 512 --dim_feedforward 3200 --batch_size 8 --lr 1e-5 \
    --seed 0 --num_epochs 2000 --wandb_run_name act_avoid_s0

# B. PACT: raw per-sensor tokens + vision dropout (forces skin use)
python imitate_episodes.py \
    --task_name obstacle_pact_avoid_v1 --policy_class ACT \
    --ckpt_dir ckpts --kl_weight 10 --chunk_size 100 \
    --hidden_dim 512 --dim_feedforward 3200 --batch_size 8 --lr 1e-5 \
    --seed 0 --num_epochs 2000 \
    --use_proximity --prox_feature raw --prox_layout per_sensor \
    --image_dropout_p 0.3 --prox_dropout_p 0.1 \
    --wandb_run_name pact_raw_persensor_idrop03_s0
```

Eval. **Always** `eval_act_obstacle.py --temp_agg_off`. Invisible cell is the
causal test (cameras cannot see the bar; skin can).

```bash
for ARM in act_avoid_s0 pact_raw_persensor_idrop03_s0; do
  for CELL in invisible free visible; do
    python eval_act_obstacle.py \
      --ckpt_dir ckpts/obstacle_pact_avoid_v1/<dated>_${ARM} \
      --output_dir /home/jaydv/code/prox_learning/eval_output/${ARM}_${CELL} \
      --num_rollouts 50 --chunk_size 100 --temp_agg_off --eval_cell $CELL
  done
done

cd /home/jaydv/code/prox_learning
python scripts/compare_pact.py \
    vanilla=<S>/50,<C>/50 pact_raw=<S>/50,<C>/50
```

Never: `python imitate_episodes.py --eval` on a PACT ckpt. That call is
`policy(qpos, image)` with no skin. The entry now exits if you try.

---

## 6. How to *add* collisions (and still teach avoidance)

Do **not** inject random crashes into BC. That teaches colliding.

Do this instead:

1. **Drop inbound scrapes at convert** (`--skip_approach_collision --keep_deflect_collisions`). Stops the 40% of *free* demos that smear fixtures from becoming the policy, without deleting the 49 bows (most bows still graze; p10 contacts = 0).
2. **Upsample real bows** (`--upsample_deflect 3`). The 49 deflect episodes are
   the only ones whose TCP path proves avoidance (~4 cm lateral bow).
3. **Collect more deflect + abort** if the reconverted set is too small after
   filtering. Same configs, more houses, keep `behavior_class=deflect` and
   `approach_contacts=0`. Target: ≥150 clean deflect episodes.
4. **Eval cell `invisible`, n=50, report collisions.** That is the paper table.
   Strict success (lift AND contact-free) is the hedge against "policy that
   barely moves".
5. **Optional third method, not PACT:** residual `SafetyHead` on top of vanilla
   ACT at eval. Uses the CVAE for its trained job (joint retreat). Compare as
   ACT vs PACT-raw vs ACT+reflex.

If you need *more* near-miss diversity (tighter bars, more invis), that is a
datagen rerun of `FrankaSkinHybridInvisObstacleConfig`, not a fake collision
labeler.

---

## 7. What "incredible" looks like

A result worth writing:

- Invisible-cell **collision** : PACT-raw (per-sensor, image dropout) **beats**
  vanilla ACT by ≥15 points, n=50, Fisher p<0.05. (Already 26 points with the
  old global-raw tap — we need that to hold after the data filter, and ideally
  grow.)
- Invisible-cell **strict success** not worse than vanilla (no "statue" policy).
- Free cell collisions similar (proves we did not just freeze the arm).
- Ablations: `trunk` ≥ vanilla collisions (negative control); `raw` + no image
  dropout weaker than `raw` + dropout (skin actually used).

Success rate matching vanilla is OK. The claim is **safer when cameras cannot
see the bar**, not "better picker".

---

## 8. Coauthor place-corridor (reproduce, do not cite yet)

HF `Lundii/pact_place_corridor_v5` → `data/pact_place_corridor_v5`. 152 clean
pick-and-place rows. Wrist RGB only. Scene XML `pact_place_corridor_v2`.
Commands: README §13.1. Train vanilla vs `--use_proximity --prox_feature raw
--prox_layout per_sensor`, chunk 50, no image dropout. Eval
`eval_act_place_corridor.py --temp_agg_off`. Metric: place-success +
`bar_hit_rate`. Local n=50 (2026-08-27): place-success **28% vs 42%**
(ACT vs PACT-raw, Fisher p = 0.21), bar hit **34% vs 36%** (p = 1.0).
**No safety win.** Success gap is noise. **Not a paper number.** n=20 smoke
(15% vs 35%, bar 30% vs 20%) was luck. Files:
`eval_output/place_corridor_vanilla_s0_n50/` and `place_corridor_raw_s0_n50/`.

Local 32-d surface embedding (2026-08-25): compressor gate passed on 15 test
rows (100% validity, 20.6 mm XYZ, recon pixel 87/95%). That is a reconstruction
score, not a policy score. Geometry sees 11.8% of tiles inside 20 cm; PACT-raw
fires 41% inside 50 cm. Headline arm stays **raw peak closeness**. Embedding
tokens are an ablation after ACT reuses the encoder `split_manifest.json`.
