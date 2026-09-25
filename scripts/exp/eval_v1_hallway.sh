#!/usr/bin/env bash
# ./scripts/exp/eval_v1_hallway.sh
# Default: keep-fraction sweep 75/50/25/0 (skip 100 — use existing H-B dirs).
# Same seeds 2026+i, house 1. Paired ablation. 0% keep is the required control.
# SENSOR_KEEP_FRACS=1 for a full-skin run (writes ..._keep100, does not overwrite H-B).
set -e
cd /home/jaydv/code/prox_learning
export OMP_NUM_THREADS=2
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MLSPACES_ASSETS_DIR=/home/jaydv/code/prox_learning/assets

EXP="${EXP:-PACT_RAW}"  # ACT, PACT_RAW, PACT_READOUT
SEED="${SEED:-0}"
TASK=pact_place_corridor_v5
CHUNK_SIZE=50
BATCH_SIZE=8
LR=1e-5
EPOCHS=2000
NUM_ROLLOUTS="${NUM_ROLLOUTS:-50}"
HOUSE_IND=1
SEED_BASE=2026
SKIN=egl
HISTORY=query
CAMERAS="wrist_camera"
CKPT_NAME=policy_best.ckpt
WANDB_PROJECT=PC_ACT_experiments
SENSOR_KEEP_FRACS="${SENSOR_KEEP_FRACS:-0.75 0.5 0.25 0}"
SAVE_FIRST_FRAME="${SAVE_FIRST_FRAME:-1}"
SENSOR_MASK_FIXED="${SENSOR_MASK_FIXED:-0}"
SENSOR_MASK_SEED="${SENSOR_MASK_SEED:-}"
# Policy cameras fed as black frames at inference (skin + qpos unchanged): all, or names
# from CAMERAS, space-separated. Output dir gets _blind (all) or _no<cam> (exo_camera_1 -> _noexo).
BLANK_CAMERAS="${BLANK_CAMERAS:-}"
# 1 = original slow path: per-substep pose refresh of the 40 skin cameras and
# per-step object_image_points segmentation renders. A/B only. Not a protocol field.
EAGER_CAMERAS="${EAGER_CAMERAS:-0}"

# Brace every var. Bare $TASK_ is an empty name, not "$TASK" + "_".
RUN="${TASK}_${EXP}_s${SEED}_bs${BATCH_SIZE}_cs${CHUNK_SIZE}_lr${LR}_e${EPOCHS}"
CKPT_DIR=/home/jaydv/code/prox_learning/submodules/act/ckpts/$TASK/$RUN

echo "$RUN keep_fracs=${SENSOR_KEEP_FRACS} fixed=${SENSOR_MASK_FIXED}"

if [ ! -f "$CKPT_DIR/$CKPT_NAME" ]; then
  echo "MISSING $CKPT_DIR/$CKPT_NAME"; exit 4
fi

for P in ${SENSOR_KEEP_FRACS}; do
  PCT=$(python -c "print(int(round(100*float('${P}'))))")
  OUT_TAG="_keep${PCT}"
  if [ "${SENSOR_MASK_FIXED}" = "1" ]; then
    OUT_TAG="${OUT_TAG}_fixed"
  fi
  for C in ${BLANK_CAMERAS}; do
    if [ "$C" = "all" ]; then OUT_TAG="${OUT_TAG}_blind"; else OUT_TAG="${OUT_TAG}_no${C%%_*}"; fi
  done
  OUTPUT_DIR=/home/jaydv/code/prox_learning/eval_output/${RUN}${OUT_TAG}
  EXTRA=()
  EXTRA+=(--sensor_keep_frac "${P}")
  if [ "${SAVE_FIRST_FRAME}" = "1" ]; then
    EXTRA+=(--save_first_frame)
  fi
  if [ "${SENSOR_MASK_FIXED}" = "1" ]; then
    EXTRA+=(--sensor_mask_fixed)
  fi
  if [ "${EAGER_CAMERAS}" = "1" ]; then
    EXTRA+=(--eager_cameras --keep_export_sensors)
  fi
  if [ -n "${BLANK_CAMERAS}" ]; then
    EXTRA+=(--blank_cameras ${BLANK_CAMERAS})
  fi
  if [ -n "${SENSOR_MASK_SEED}" ]; then
    EXTRA+=(--sensor_mask_seed "${SENSOR_MASK_SEED}")
  fi
  echo "${RUN}${OUT_TAG} p=${P}"
  if [ -f "$OUTPUT_DIR/eval_summary.json" ] || [ -s "$OUTPUT_DIR/episodes.jsonl" ]; then
    DONE_N=$(python -c "import json,sys,os; p=sys.argv[1]; print(json.load(open(p)).get('completed',0) if os.path.isfile(p) else 0)" "$OUTPUT_DIR/eval_summary.json")
    if [ "$DONE_N" -ge "$NUM_ROLLOUTS" ]; then
      echo "DONE $OUTPUT_DIR ($DONE_N/$NUM_ROLLOUTS)"; continue
    fi
    echo "REFUSE $OUTPUT_DIR is partial ($DONE_N/$NUM_ROLLOUTS). No resume: new dir."; continue
  fi
  python eval_act.py \
    --ckpt_dir "$CKPT_DIR" \
    --ckpt_name "$CKPT_NAME" \
    --task hallway \
    --cameras $CAMERAS \
    --num_rollouts "$NUM_ROLLOUTS" \
    --house_ind "$HOUSE_IND" \
    --seed_base "$SEED_BASE" \
    --skin "$SKIN" \
    --history "$HISTORY" \
    --output_dir "$OUTPUT_DIR" \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_name "${RUN}${OUT_TAG}_eval" \
    "${EXTRA[@]}"
done
