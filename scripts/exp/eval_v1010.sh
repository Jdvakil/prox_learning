#!/usr/bin/env bash
# ./scripts/exp/eval_v1010.sh
# Not a suite yet. eval_act.py --task v1010 exits. Edit CKPT_DIR after train. Do not use hallway eval.
set -e
REPO=/home/jaydv/code/prox_learning
cd "$REPO"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export MLSPACES_ASSETS_DIR="${MLSPACES_ASSETS_DIR:-$REPO/assets}"

EXP="${EXP:-PACT_READOUT}"
SEED="${SEED:-1}"
TASK="${TASK:-pact_place_corridor_v1010}"
CHUNK_SIZE="${CHUNK_SIZE:-50}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-1e-5}"
EPOCHS="${EPOCHS:-2000}"
RUN="${TASK}_${EXP}_s${SEED}_bs${BATCH_SIZE}_cs${CHUNK_SIZE}_lr${LR}_e${EPOCHS}"

CKPT_DIR="${CKPT_DIR:-$REPO/submodules/act/ckpts/$TASK/$RUN}"
CKPT_NAME="${CKPT_NAME:-policy_best.ckpt}"
EVAL_TASK="${EVAL_TASK:-v1010}"
CAMERAS="${CAMERAS:-table_camera wrist_camera}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-2}"
HOUSE_IND="${HOUSE_IND:-1}"
SEED_BASE="${SEED_BASE:-2026}"
SKIN="${SKIN:-egl}"
HISTORY="${HISTORY:-query}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO/eval_output/${TASK}_${EXP}_s${SEED}_n${NUM_ROLLOUTS}}"

echo "ckpt=$CKPT_DIR"
echo "out=$OUTPUT_DIR"
echo "closed-loop --task $EVAL_TASK is not wired (README §7). This call will exit."

python eval_act.py \
  --ckpt_dir "$CKPT_DIR" \
  --ckpt_name "$CKPT_NAME" \
  --task "$EVAL_TASK" \
  --cameras $CAMERAS \
  --num_rollouts "$NUM_ROLLOUTS" \
  --house_ind "$HOUSE_IND" \
  --seed_base "$SEED_BASE" \
  --skin "$SKIN" \
  --history "$HISTORY" \
  --output_dir "$OUTPUT_DIR" \
  "$@"
