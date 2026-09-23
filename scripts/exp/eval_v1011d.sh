#!/usr/bin/env bash
# ./scripts/exp/eval_v1011d.sh
# Edit knobs below. One run. No sweep.
set -e
cd /home/jaydv/code/prox_learning
export OMP_NUM_THREADS=2
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MLSPACES_ASSETS_DIR=/home/jaydv/code/prox_learning/assets

EXP="${EXP:-ACT}"  # ACT, PACT_RAW, PACT_READOUT
SEED="${SEED:-0}"
TASK=pact_pick_n_place_v2_v1011d
CHUNK_SIZE=50
BATCH_SIZE=8
LR=1e-5
EPOCHS=2000
NUM_ROLLOUTS="${NUM_ROLLOUTS:-50}"
SEED_BASE=0
SKIN=egl
HISTORY=consecutive
SKIN_SUBSTEPS=snapshot
# 0.25 = easy (T-1011d three-arm dirs, untagged). 1 = full V10.11d randomize -> dir tag _xy1.
CLUTTER_XY_SCALE="${CLUTTER_XY_SCALE:-0.25}"
CAMERAS="exo_camera_1 wrist_camera"
CKPT_NAME=policy_best.ckpt
WANDB_PROJECT=PC_ACT_experiments_eval
# Per-link keep rate. 1=all, 0=drop all, 0.5=Poisson 50% on each link.
SENSOR_KEEP_FRAC=1
SAVE_FIRST_FRAME=1
SENSOR_MASK_FIXED=0
# 1 = original slow path: per-substep pose refresh of the 40 skin cams +
# per-step object_image_points segmentation renders. A/B only.
EAGER_CAMERAS=0

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
XY_TAG=$(python -c "s=float('${CLUTTER_XY_SCALE}'); print('' if abs(s-0.25)<1e-9 else '_xy%g' % s)")
OUT_TAG="${OUT_TAG}${XY_TAG}"
OUTPUT_DIR=/home/jaydv/code/prox_learning/eval_output/${RUN}${OUT_TAG}

EXTRA=(--sensor_keep_frac "${SENSOR_KEEP_FRAC}")
if [ "${SAVE_FIRST_FRAME}" = "1" ]; then
  EXTRA+=(--save_first_frame)
fi
if [ "${SENSOR_MASK_FIXED}" = "1" ]; then
  EXTRA+=(--sensor_mask_fixed)
fi

echo "${RUN}${OUT_TAG} p=${SENSOR_KEEP_FRAC} clutter_xy_scale=${CLUTTER_XY_SCALE}"

if [ ! -f "$CKPT_DIR/$CKPT_NAME" ]; then
  echo "MISSING $CKPT_DIR/$CKPT_NAME (not trained yet)"; exit 4
fi
if [ -f "$OUTPUT_DIR/eval_summary.json" ] || [ -s "$OUTPUT_DIR/episodes.jsonl" ]; then
  DONE_N=$(python -c "import json,sys,os; p=sys.argv[1]; print(json.load(open(p)).get('completed',0) if os.path.isfile(p) else 0)" "$OUTPUT_DIR/eval_summary.json")
  if [ "$DONE_N" -ge "$NUM_ROLLOUTS" ]; then
    echo "DONE $OUTPUT_DIR ($DONE_N/$NUM_ROLLOUTS)"; exit 0
  fi
  echo "REFUSE $OUTPUT_DIR is partial ($DONE_N/$NUM_ROLLOUTS). No resume: new dir."; exit 3
fi

if [ "${EAGER_CAMERAS}" = "1" ]; then
  EXTRA+=(--eager_cameras --keep_export_sensors)
fi

run_eval() {
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
    "${EXTRA[@]}" "$@"
}

run_eval
