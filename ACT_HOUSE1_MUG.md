# Vanilla ACT on PLA House-1 Mug Pickup

End-to-end recipe: convert the molmospaces PLA dataset → train upstream ACT
(`submodules/act/imitate_episodes.py`) with wandb logging → roll the trained
checkpoint out in the same molmospaces environment to produce success
metrics + per-episode rollout videos.

Everything lives inside `submodules/act/` and `scripts/`. Nothing depends on
`pla/`.

---

## 1. Convert the dataset (one-time)

Source: a single `trajectories_batch_1_of_1.h5` produced by datagen, with one
group per trajectory and sibling MP4s per camera. The converter writes one
`episode_<i>.hdf5` per trajectory in the layout `submodules/act/utils.py`
expects (`/action`, `/observations/qpos`, `/observations/qvel`,
`/observations/images/<cam>`).

```bash
conda activate prox
cd /home/jaydv/code/prox_learning

python -m scripts.convert_pla_to_act \
  --src /home/jaydv/code/prox_learning/assets/datagen/pick_and_place_one_house_mug_dup250_v1/FrankaSkinPickAndPlaceOneHouseMugDup250Config/20260515_010000/house_1/trajectories_batch_1_of_1.h5 \
  --dst /home/jaydv/code/prox_learning/act_style_data/pla_house1_mug_v1 \
  --image_h 240 --image_w 320
```

Output: 250 files `episode_0.hdf5 … episode_249.hdf5`, each ~261 frames at
240×320 with two RGB cameras (`exo_camera_1`, `wrist_camera`). qpos = arm(7)
+ 2 finger joints (9-d); action = arm(7) + gripper_cmd(1) (8-d).

The dataset path is already registered in
`submodules/act/constants.py:TASK_CONFIGS['pla_house1_mug']`. Change `--dst`
to use a different location, but then keep the constants entry in sync.

---

## 2. Install the ACT submodule (one-time)

The ACT repo's `detr/` package needs to be importable. From the same env:

```bash
cd /home/jaydv/code/prox_learning/submodules/act/detr
pip install -e .
```

If `imitate_episodes.py` still can't find `util.misc`, run training with
`PYTHONPATH=$PWD/detr:$PYTHONPATH` from inside `submodules/act`.

---

## 3. Train

```bash
cd /home/jaydv/code/prox_learning/submodules/act
mkdir -p ckpts/act_house1_mug_v1

python imitate_episodes.py \
  --task_name pla_house1_mug \
  --ckpt_dir ckpts/act_house1_mug_v1 \
  --policy_class ACT \
  --kl_weight 10 \
  --chunk_size 100 \
  --hidden_dim 512 \
  --batch_size 8 \
  --dim_feedforward 3200 \
  --num_epochs 2000 \
  --lr 1e-5 \
  --seed 0 \
  --use_wandb \
  --wandb_project act-pla-house1 \
  --wandb_run_name act_house1_mug_v1
```

What changed vs. the upstream ACT README command:

* `--task_name pla_house1_mug` — registered in `constants.py`, points at the
  converted dataset, two cameras, episode_len=261.
* `imitate_episodes.py` now sets `state_dim=9, action_dim=8` for this task
  and propagates both to the DETR-VAE so the encoder/action head dims are
  correct. (The patch to `detr/models/detr_vae.py` adds an `action_dim`
  argument that defaults to `state_dim` — Aloha-style symmetric runs are
  unaffected.)
* `--use_wandb` enables `wandb.init()` inside `train_bc` and logs
  `train/{l1,kl,loss}`, `val/{l1,kl,loss}`, and `min_val_loss` per epoch.

Outputs in `ckpt_dir`:

* `policy_best.ckpt` — lowest val loss.
* `policy_last.ckpt` — final epoch.
* `policy_epoch_<E>_seed_0.ckpt` — every 100 epochs.
* `dataset_stats.pkl` — qpos/action mean+std (needed at eval).
* `train_val_*_seed_0.png` — loss curves.

Note: ACT is on the upstream layout, so `set_seed(1)` at the top of
`main()` reshuffles the data split each run; the `--seed` flag controls
weight init / training, not the train/val split.

---

## 4. Evaluate (rollouts in the same molmospaces env)

`submodules/act/eval_act_house1.py` extends `FrankaSkinPickAndPlaceOneHouseMugConfig`
with an `ACTInferencePolicy` (built in this file — no `pla/` imports) and
runs it via `ParallelRolloutRunner`. Rollouts use temporal ensembling
(Zhao et al. 2023) on top of the chunked ACT output. The script:

* Forces offscreen rendering (`MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl`,
  unsets `DISPLAY`) before any mujoco / molmospaces import, so it runs
  cleanly over SSH.
* Bumps `task_sampler_config.max_total_attempts_multiplier` to 100 — the
  scripted teleop planner is used as a feasibility filter on each
  randomized scene and rejects most attempts as `HouseInvalidForTask`. A
  higher multiplier means more sampling tries before giving up on
  collecting `num_rollouts` rolloutable scenes.
* Optionally logs to wandb (config + live per-episode signal + per-camera
  MP4 uploads at the end).

```bash
cd /home/jaydv/code/prox_learning/submodules/act

PYTHONPATH="$PWD:$PWD/detr:$PYTHONPATH" \
python eval_act_house1.py \
  --ckpt_dir ckpts/act_house1_mug_v1 \
  --ckpt_name policy_best.ckpt \
  --output_dir /home/jaydv/code/prox_learning/eval_output/act_house1_mug_v1 \
  --num_rollouts 10 \
  --task_horizon 300 \
  --chunk_size 100 \
  --kl_weight 10 \
  --hidden_dim 512 \
  --dim_feedforward 3200 \
  --image_h 240 --image_w 320 \
  --seed 0 \
  --use_wandb \
  --wandb_project act-pla-house1-eval \
  --wandb_run_name act_house1_mug_v1_eval
```

Output layout (set by `ParallelRolloutRunner`):

```
eval_output/act_house1_mug_v1/
  experiment_config_<timestamp>.pkl
  running_log.log
  house_1/
    trajectories_batch_1_of_1.h5
    episode_00000000_exo_camera_1_batch_1_of_1.mp4
    episode_00000000_wrist_camera_batch_1_of_1.mp4
    ...
```

The terminal prints `success {hit}/{total}` from the molmospaces task
evaluator. Failed episodes are kept (we set
`filter_for_successful_trajectories=False`) so you can watch what the
policy got wrong. MP4s + h5 are written **after the full batch completes**,
not per-episode — tail `running_log.log` to watch progress while it runs.

### wandb logging (when `--use_wandb` is passed)

* On startup the script logs all eval hyperparams (ckpt, horizon, chunk
  size, KL weight, model dims, temp-agg settings, seed).
* During eval, every time the policy resets (= start of a new rollout),
  the previous rollout's step count is logged as `rollout/episode_length`,
  keyed on `rollout/episode_idx`. This is the live signal — until videos
  upload at the end, this is how you watch progress in the wandb UI.
* After the runner finishes:
  * `eval/success`, `eval/total`, `eval/success_rate` (also mirrored to
    `wandb.summary` so they show up in the run list view).
  * Every per-camera rollout MP4 in `<output_dir>/house_*/` is uploaded
    as a `wandb.Video` under `videos/<house>/ep<i>/<camera>`.

If `--use_wandb` is omitted, eval still runs normally — wandb is purely
opt-in. Extra CLI flags (`--wandb_project`, `--wandb_run_name`,
`--wandb_entity`) let you target a specific project / run name / entity.

### Tips

* Disable temporal ensembling (use raw chunked playback) with
  `--temp_agg_off`. Useful when chasing whether the timing artifact in
  the memory `eval_chunk_timing_memorization.md` is biting — if success
  rate is similar with vs. without ensembling, the policy isn't reacting
  to the obs, it's playing back a memorized open-loop chunk.
* For a quick sanity run, set `--num_rollouts 2 --task_horizon 200`.
* Hyperparams passed to eval (`--kl_weight`, `--hidden_dim`,
  `--dim_feedforward`, `--chunk_size`) **must match** the training run —
  they determine model architecture.
* DETR's internal `argparse` requires `--policy_class / --task_name /
  --num_epochs` that the eval CLI doesn't expose. The script injects a
  minimal valid argv for DETR via `_detr_argv` while building the model,
  so you don't need to pass them. The eval CLI also tolerates them via
  `parse_known_args` in case they get pasted from a training command.

---

## What I touched

* `scripts/convert_pla_to_act.py` (new): h5 + MP4 → per-episode hdf5.
* `submodules/act/constants.py`: added `pla_house1_mug` TASK_CONFIGS entry.
* `submodules/act/imitate_episodes.py`:
  * branch for `pla_house1_mug` setting `state_dim=9, action_dim=8`;
  * plumbed `action_dim` into `policy_config`;
  * added wandb init/log inside `train_bc`;
  * added `--use_wandb / --wandb_project / --wandb_run_name` flags.
* `submodules/act/detr/main.py`: added the new flags + `action_dim` to the
  unused-arg parser stub so argparse doesn't reject them.
* `submodules/act/detr/models/detr_vae.py`: `DETRVAE` now takes
  `action_dim` (defaults to `state_dim`) — used to size `action_head` and
  `encoder_action_proj` independently.
* `submodules/act/eval_act_house1.py` (new): standalone eval entry point.
  Offscreen-render env-var pinning, `_detr_argv` shim, sampler attempt
  multiplier bump, and opt-in wandb logging (live `rollout/*` keys +
  final `eval/*` metrics + per-camera `wandb.Video` uploads).
