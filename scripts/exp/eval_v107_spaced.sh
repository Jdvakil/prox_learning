#!/usr/bin/env bash
# ./scripts/exp/eval_v107_spaced.sh
set -e
cd /home/jaydv/code/prox_learning
export OMP_NUM_THREADS=2
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MLSPACES_ASSETS_DIR=/home/jaydv/code/prox_learning/assets

EXP=ACT #ACT, PACT_RAW, PACT_READOUT
SEED=0
TASK=pact_place_corridor_v107_spaced
CHUNK_SIZE=50
BATCH_SIZE=8
LR=1e-5
EPOCHS=2000
NUM_ROLLOUTS=50
SEED_BASE=2026
SKIN=egl
HISTORY=query
SKIN_SUBSTEPS=snapshot
CAMERAS="table_camera wrist_camera"
CKPT_NAME=policy_best.ckpt
WANDB_PROJECT=PC_ACT_experiments

# Brace every var. Bare $TASK_ is an empty name, not "$TASK" + "_".
RUN="${TASK}_${EXP}_s${SEED}_bs${BATCH_SIZE}_cs${CHUNK_SIZE}_lr${LR}_e${EPOCHS}"
CKPT_DIR=/home/jaydv/code/prox_learning/submodules/act/ckpts/$TASK/$RUN
OUTPUT_DIR=/home/jaydv/code/prox_learning/eval_output/$RUN

echo "$RUN"

python eval_act_v107spaced.py \
  --ckpt_dir "$CKPT_DIR" \
  --ckpt_name "$CKPT_NAME" \
  --cameras $CAMERAS \
  --num_rollouts "$NUM_ROLLOUTS" \
  --seed_base "$SEED_BASE" \
  --skin "$SKIN" \
  --history "$HISTORY" \
  --skin_substeps "$SKIN_SUBSTEPS" \
  --output_dir "$OUTPUT_DIR" \
  --wandb_project "$WANDB_PROJECT" \
  --wandb_run_name "${RUN}_eval"
