#!/usr/bin/env bash
# ./scripts/exp/eval_v1011d.sh
# Wired: repo-root eval_act_v1011d.py. Default CKPT_DIR is the existing raw run, not the train_exp RUN name.
set -e
REPO=/home/jaydv/code/prox_learning
cd "$REPO"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export MLSPACES_ASSETS_DIR="${MLSPACES_ASSETS_DIR:-$REPO/assets}"

EXP="${EXP:-PACT_RAW}"
SEED="${SEED:-0}"
TASK="${TASK:-pact_pick_n_place_v2}"
CHUNK_SIZE="${CHUNK_SIZE:-50}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-1e-5}"
EPOCHS="${EPOCHS:-2000}"
RUN="${TASK}_${EXP}_s${SEED}_bs${BATCH_SIZE}_cs${CHUNK_SIZE}_lr${LR}_e${EPOCHS}"

CKPT_DIR="${CKPT_DIR:-$REPO/submodules/act/ckpts/pact_pick_n_place_v2/20260903_171108_pact_pick_n_place_v2_v1011d_s0}"
CKPT_NAME="${CKPT_NAME:-policy_best.ckpt}"
CAMERAS="${CAMERAS:-exo_camera_1 wrist_camera}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-2}"
SEED_BASE="${SEED_BASE:-2026}"
SKIN="${SKIN:-egl}"
HISTORY="${HISTORY:-query}"
SKIN_SUBSTEPS="${SKIN_SUBSTEPS:-snapshot}"
CLUTTER_XY_SCALE="${CLUTTER_XY_SCALE:-1.0}"
SAVE_VIDEO="${SAVE_VIDEO:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO/eval_output/${TASK}_${EXP}_n${NUM_ROLLOUTS}}"

extra=
if [ "$SAVE_VIDEO" = 1 ]; then
  extra="--save_video"
fi

echo "ckpt=$CKPT_DIR"
echo "out=$OUTPUT_DIR"

python eval_act_v1011d.py \
  --ckpt_dir "$CKPT_DIR" \
  --ckpt_name "$CKPT_NAME" \
  --cameras $CAMERAS \
  --num_rollouts "$NUM_ROLLOUTS" \
  --seed_base "$SEED_BASE" \
  --skin "$SKIN" \
  --history "$HISTORY" \
  --skin_substeps "$SKIN_SUBSTEPS" \
  --clutter_xy_scale "$CLUTTER_XY_SCALE" \
  --output_dir "$OUTPUT_DIR" \
  $extra \
  "$@"
