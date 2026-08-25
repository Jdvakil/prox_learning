# Chunk-25 place run: ACT vs PACT, reactivity test

*Every path, constant and cost below was read from source or measured from the completed chunk-1 and
chunk-100 runs. Nothing is projected except where labelled.*

## Start here

**Almost nothing new is built.** The evaluator, wrapper, manifest and contract all exist and are
reused verbatim. The only new artifacts are two checkpoints and a report.

| | Step | Output |
|---|---|---|
| T0/T1 | train ACT@25 then PACT@25 | `/root/pact_place_152_pact_vs_act_chunk25_seed3101/` |
| T2 | smoke, 2 rollouts × 2 arms | stop if the gripper never closes |
| T3 | full eval, 2 arms × 40 | 80 rollout results |
| T4 | report | `EVAL.md` in the eval root |

Reused unchanged: `submodules/act/eval_pact_place_chunk100_row.py` (pass `--num-queries 25`),
`configs/pact_place_eval_chunk100_manifest.json`, `scripts/pact_place_eval_chunk100_contract.py`.

## Context

Three chunk sizes, two already measured:

| chunk | gripper close | CFTS (ACT / PACT) | status |
|---|---|---|---|
| **1** | 0/40 | 0/20 / 0/20 | `CHUNK1_COLLAPSE` — copy-the-state shortcut |
| **25** | ? | ? | this run |
| **100** | 40/40 | 13/40 / 16/40 | `FUNCTIONAL_CHUNK100` |

Chunk 1 collapsed because the policy learned to **echo its own gripper state**
(`corr(qpos[7], action[7]) = 0.9934`; the rule "command 255 iff `qpos[7] >= 0.382`" reproduces the
demonstrator on 99.83% of steps). Chunk 100 broke it: on the 20 shared instances, gripper close went
**0/20 → 20/20** for both arms.

### What chunk 25 buys, and what it risks

**Reactivity** — the eval ensembles pending chunks with `exp(-0.01·age)`
(`eval_pact_collision_row.py:391-406`), so chunk size sets how much the current proximity reading
influences the executed action:

| chunk | weight on the current-step prediction | median executed-action age |
|---|---|---|
| 100 | 1.57% | 37 steps |
| **25** | **4.50%** | **11 steps** |
| 1 | 100% | 0 steps |

That is the teammate's hypothesis: proximity is a *reactive* signal and chunking dilutes it, so PACT's
advantage should widen as the chunk shrinks.

**The risk** — measured over all 152 episodes, the fraction of K-step windows containing a gripper
transition (the only pressure that defeats the shortcut):

```
chunk   1 :  0.0%   -> collapsed (confirmed)
chunk  25 : 10.4%   -> unknown
chunk 100 : 38.1%   -> functional (confirmed)
```

Every episode has exactly **2** transitions in ~480 steps. At 10.4% a **partial** break is plausible —
the gripper closes sometimes but not reliably. That is a real possible outcome and is predeclared below.

## The primary hypothesis cannot be resolved at N = 40 — read this before running

The reactivity claim is an **interaction**: does `(PACT − ACT)@25` exceed `(PACT − ACT)@100`? That is a
difference-in-differences, and at N = 40 per cell:

```
(PACT - ACT)@100 = +7.5 pp,  approx 95% [-13.5, +28.5]     <- already crosses zero
SE of the difference-in-differences  ~ 15.2 pp  ->  95% width ~ +/-30 pp
```

**The interaction would have to change the gap by more than ~30 pp to clear zero, while the gap itself
is only +7.5 pp.** Detecting a 10 pp change would need ~714 instances per cell; 20 pp needs ~179.

**So this run cannot confirm or refute the reactivity hypothesis.** It is still worth running, but the
report must say so plainly rather than presenting a directional number as support.

### What the run *can* deliver

1. **Answerable:** does the shortcut stay broken at 10.4% transition coverage? Gripper-close rate at
   N = 40 settles this — it was 0/40 and 40/40 at the two ends, so anything in between is informative.
2. **Answerable:** do the chunk-25 policies function at all (task success, collision-free success)?
3. **Directional only:** `PACT − ACT` at chunk 25, reported beside +7.5 pp at chunk 100, explicitly
   labelled as unable to resolve the interaction.

## Decisions

| | |
|---|---|
| Arms | **ACT and PACT only.** No `PACT_PERMUTED` |
| Episodes | **40 per arm**, the same 40 instances as chunk 100 (20 shared with chunk 1) |
| Instances | **reuse `configs/pact_place_eval_chunk100_manifest.json` verbatim** |
| Seed | **3101**, both arms, as at chunk 1 and 100 |
| Encoder | the frozen corridor encoder, unchanged — cross-task transfer |

**On skipping `PACT_PERMUTED`:** at chunk 100 it scored 6/40 (15%) against ACT's 13/40 (32.5%) — far
*below* the no-proximity baseline, so it behaved as an active distractor rather than a clean control.
That inflated the chunk-100 headline: of the reported +25.0 pp for PACT − PACT_PERMUTED, only +7.5 pp
is PACT − ACT and +17.5 pp is the permuted arm sitting below ACT. Leaving it out is defensible, but it
means **the chunk-100 anomaly stays undiagnosed** and this run has no within-model ablation — so the
only contrast available is the cross-model one this project has repeatedly shown to be seed-unstable.
Say that in the report.

## Measured costs

```
training, chunk-100 actuals   ACT 76 min, PACT 78 min = 2.6 h    (a 25-query decoder should be <=)
eval rollout, chunk-100       18.46 min mean, 10 workers
80 rollouts                   2.5 h
smoke + report                0.5 h
                              ------
TOTAL                         ~5.5 h
```

## T0 / T1 — train, ACT first

Copy the chunk-100 command; change **only** `--chunk_size 100` → `--chunk_size 25` and the
`--ckpt_dir`. New root `/root/pact_place_152_pact_vs_act_chunk25_seed3101/`.

```
# ACT
python imitate_episodes.py --task_name obstacle_baseline \
  --ckpt_dir /root/pact_place_152_pact_vs_act_chunk25_seed3101/act_seed3101 --exact_ckpt_dir \
  --policy_class ACT --batch_size 8 --seed 3101 --num_epochs 2000 --lr 1e-5 --kl_weight 10 \
  --chunk_size 25 --hidden_dim 512 --dim_feedforward 3200 --enc_layers 7 --dec_layers 7 \
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

Record start/end timestamps to `training_timing.json`, as the chunk-100 run did. **Do not write into
the chunk-1 or chunk-100 directories.**

## T2 — smoke

**2 rollouts per arm.** Use `eval_pact_place_chunk100_row.py` with `--num-queries 25`. Assert:

- both checkpoints load at `num_queries=25` under `strict=True`
- PACT reports `input_proj_proximity_shape [512, 32]` and `proximity_consumed_for_action: true`
- every rollout records `num_queries == 25` and 900 control steps
- **at least one smoke rollout commands the gripper closed.** If none does across all four, the
  shortcut likely survives at 10.4% coverage — that is a legitimate finding, so **stop and report it**
  rather than spending 2.5 h to confirm a collapse.
- record per-rollout wall-clock and re-derive the T3 schedule from it

## T3 — full eval

2 arms × 40 rows = **80 rollouts**, 10 workers, ~2.5 h. Reuse
`configs/pact_place_eval_chunk100_manifest.json` unchanged.

**Note the naming honestly in the report:** the manifest is labelled `chunk100` in its
`schema_version` and `role`, but those are cosmetic. Its rows encode instances — `task_seed_u64`,
panel jitters, `intrusion_side` — which are chunk-agnostic. Reusing it is what makes chunk 1, 25 and
100 paired on identical episodes. Do not rename or rebuild it.

## T4 — report

`EVAL.md` in the eval root. Per arm: task success, collision-free task success, `gripper_close_commanded`
count, contact episodes by class, control-step counts, wall-clock.

**Freeze before the first rollout:**

- **Shortcut broken** iff `gripper_close_commanded` ≥ 20/40 for both arms.
- **Partial break** if either arm lands in 1–19/40. Report as `CHUNK25_PARTIAL` — an informative
  middle point, not a failure, and the most likely outcome given 10.4% coverage.
- **Collapse floor:** if either arm scores ≤ 2/40 CFTS, report `CHUNK25_COLLAPSE`; all modality
  numbers are then uninformative.
- **No interaction claim.** Report `(PACT − ACT)@25` beside `+7.5 pp` at chunk 100 and state that the
  difference-in-differences has a ~±30 pp interval at this N and therefore cannot resolve the
  reactivity hypothesis either way.

**Also report the three-point table** — gripper close and CFTS at chunks 1, 25, 100 on the shared
instances. That table is the durable output of this run regardless of what the modality contrast does.

**State in EVAL.md:** single seed 3101; N = 40; no within-model ablation this run, so the only
contrast is the cross-model one that has flipped sign between seeds before; cross-task encoder
transfer; `place_receptacle` contacts over-counted because a learned policy exposes no expert phase.

## Verification

- Training commands differ from the chunk-100 pair **only** by `--chunk_size` and `--ckpt_dir`; paste
  the token diff. PACT differs from ACT by exactly the five proximity flags.
- `num_queries == 25` in every result; both checkpoints load `strict=True`; PACT records `[512, 32]`.
- The manifest file is **byte-identical** to the one the chunk-100 run used — verify by sha256, do not
  regenerate it.
- 80/80 rollouts complete, each row reconciling to exactly one result; 900 control steps throughout.
- The 20 chunk-1 shared instances are matched on `task_seed_u64`, both jitters and `intrusion_side` —
  assert those, **never `row_sha256`**.
- Unmodified, verified by hash: chunk-1 and chunk-100 run directories and all four checkpoints;
  `eval_pact_place_row.py`, `eval_pact_place_chunk100_row.py`, `eval_pact_place_permuted_row.py`,
  `eval_pact_collision_row.py`, `eval_pact_frontend_screen_row.py`, `eval_pact_valid_ablation_row.py`;
  `configs/pact_place_eval_chunk100_manifest.json`; `scripts/pact_place_eval_chunk100_contract.py`;
  the 152 dataset.

## Constraints

- **Write no new evaluator, contract or manifest.** Everything needed exists; pass `--num-queries 25`.
- Do not edit any chunk-1 or chunk-100 artifact, or any published evaluator.
- Do not change `end_on_success`, `task_horizon`, the encoder, the split, or the seed.
- Do not raise worker count above 10 without measuring throughput.
- **Do not present the reactivity hypothesis as confirmed or refuted.** The N does not support it.
- Stop after T2 if no smoke rollout commands the gripper closed.
- Interpreter `/root/act_retrain_venv/bin/python3`; `MUJOCO_GL=egl`,
  `MLSPACES_ASSETS_DIR=/root/prox_learning/assets`, `PYTHONPATH` → repo `submodules/molmospaces`.
  Work in `/root/prox_learning_pact_remediation`. A10 23 GB; 128 cores.
- `pgrep -fc "python.*(eval_pact_place|imitate_episodes)"` — plain `pgrep -fc eval_` self-matches the
  checking shell.
