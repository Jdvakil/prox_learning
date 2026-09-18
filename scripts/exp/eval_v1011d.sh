#!/usr/bin/env bash
# ./scripts/exp/eval_v1011d.sh
# Edit knobs below. One run. No sweep.
set -e
cd /home/jaydv/code/prox_learning
export OMP_NUM_THREADS=2
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MLSPACES_ASSETS_DIR=/home/jaydv/code/prox_learning/assets

EXP=ACT #ACT, PACT_RAW, PACT_READOUT
SEED=0
TASK=pact_pick_n_place_v2_v1011d
CHUNK_SIZE=50
BATCH_SIZE=8
LR=1e-5
EPOCHS=2000
NUM_ROLLOUTS=50
SEED_BASE=0
SKIN=egl
HISTORY=consecutive
SKIN_SUBSTEPS=snapshot
CLUTTER_XY_SCALE=0.25
CAMERAS="exo_camera_1 wrist_camera"
CKPT_NAME=policy_best.ckpt
WANDB_PROJECT=PC_ACT_experiments_eval
# Per-link keep rate. 1=all, 0=drop all, 0.5=Poisson 50% on each link.
SENSOR_KEEP_FRAC=1
SAVE_FIRST_FRAME=1
SENSOR_MASK_FIXED=0

# Brace every var. Bare $TASK_ is an empty name, not "$TASK" + "_".
RUN="${TASK}_${EXP}_s${SEED}_bs${BATCH_SIZE}_cs${CHUNK_SIZE}_lr${LR}_e${EPOCHS}"
CKPT_DIR=/home/jaydv/code/prox_learning/submodules/act/ckpts/$TASK/$RUN
PCT=$(python -c "print(int(round(100*float('${SENSOR_KEEP_FRAC}'))))")
OUT_TAG=""
if [ "$PCT" -ne 100 ]; then
  OUT_TAG="_keep${PCT}"
fi
if [ "${SENSOR_MASK_FIXED}" = "1" ]; then
  OUT_TAG="${OUT_TAG}_fixed"
fi
OUTPUT_DIR=/home/jaydv/code/prox_learning/eval_output/${RUN}${OUT_TAG}

EXTRA=(--sensor_keep_frac "${SENSOR_KEEP_FRAC}")
if [ "${SAVE_FIRST_FRAME}" = "1" ]; then
  EXTRA+=(--save_first_frame)
fi
if [ "${SENSOR_MASK_FIXED}" = "1" ]; then
  EXTRA+=(--sensor_mask_fixed)
fi

echo "${RUN}${OUT_TAG} p=${SENSOR_KEEP_FRAC}"

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
  --wandb_project "$WANDB_PROJECT" \
  --wandb_run_name "${RUN}${OUT_TAG}_eval" \
  "${EXTRA[@]}"
