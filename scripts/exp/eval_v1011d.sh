#!/usr/bin/env bash
# ./scripts/exp/eval_v1011d.sh
# Keep-fraction flags match hallway. Headline paired table is hallway H-B, not this shell.
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
WANDB_PROJECT=PC_ACT_experiments
SENSOR_KEEP_FRACS="${SENSOR_KEEP_FRACS:-1}"
SAVE_FIRST_FRAME="${SAVE_FIRST_FRAME:-0}"
SENSOR_MASK_FIXED="${SENSOR_MASK_FIXED:-0}"
SENSOR_MASK_SEED="${SENSOR_MASK_SEED:-}"

# Brace every var. Bare $TASK_ is an empty name, not "$TASK" + "_".
RUN="${TASK}_${EXP}_s${SEED}_bs${BATCH_SIZE}_cs${CHUNK_SIZE}_lr${LR}_e${EPOCHS}"

echo "$RUN keep_fracs=${SENSOR_KEEP_FRACS}"

for P in ${SENSOR_KEEP_FRACS}; do
  PCT=$(python -c "print(int(round(100*float('${P}'))))")
  OUT_TAG=""
  if [ "$PCT" -ne 100 ] || [ "${SENSOR_MASK_FIXED}" = "1" ]; then
    OUT_TAG="_keep${PCT}"
    if [ "${SENSOR_MASK_FIXED}" = "1" ]; then
      OUT_TAG="${OUT_TAG}_fixed"
    fi
  fi
  OUTPUT_DIR=/home/jaydv/code/prox_learning/eval_output/${RUN}${OUT_TAG}
  EXTRA=()
  EXTRA+=(--sensor_keep_frac "${P}")
  if [ "${SAVE_FIRST_FRAME}" = "1" ]; then
    EXTRA+=(--save_first_frame)
  fi
  if [ "${SENSOR_MASK_FIXED}" = "1" ]; then
    EXTRA+=(--sensor_mask_fixed)
  fi
  if [ -n "${SENSOR_MASK_SEED}" ]; then
    EXTRA+=(--sensor_mask_seed "${SENSOR_MASK_SEED}")
  fi
  echo "${RUN}${OUT_TAG} p=${P}"
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
done
