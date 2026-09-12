#!/usr/bin/env bash
# ./scripts/exp/eval_v1011d.sh
set -e
cd /home/jaydv/code/prox_learning
export OMP_NUM_THREADS=2
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MLSPACES_ASSETS_DIR=/home/jaydv/code/prox_learning/assets

EXP=PACT_RAW #ACT, PACT_RAW, PACT_READOUT
SEED=0
TASK=pact_pick_n_place_v2
CHUNK_SIZE=50
BATCH_SIZE=8
LR=1e-5
EPOCHS=2000
NUM_ROLLOUTS=2
SEED_BASE=2026
SKIN=egl
HISTORY=query
SKIN_SUBSTEPS=snapshot
CLUTTER_XY_SCALE=1.0
CAMERAS="exo_camera_1 wrist_camera"
CKPT_NAME=policy_best.ckpt
CKPT_DIR=/home/jaydv/code/prox_learning/submodules/act/ckpts/pact_pick_n_place_v2/20260903_171108_pact_pick_n_place_v2_v1011d_s0

# Brace every var. Bare $TASK_ is an empty name, not "$TASK" + "_".
RUN="${TASK}_${EXP}_s${SEED}_bs${BATCH_SIZE}_cs${CHUNK_SIZE}_lr${LR}_e${EPOCHS}"
OUTPUT_DIR=/home/jaydv/code/prox_learning/eval_output/$RUN

echo "$RUN"

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
  --output_dir "$OUTPUT_DIR"
