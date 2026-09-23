#!/usr/bin/env bash
# ./scripts/exp/eval_v5_ext.sh
# v5 ext (pact_place_corridor/data/v5, 193 eps, wrist-only train) on the hallway protocol:
# eval_act.py --task hallway, house 1, seeds 2026+i, query history, gated EGL (H-B settings).
# One run. EXP / SEED / NUM_ROLLOUTS / TAG from the environment. Skips a finished dir;
# refuses a partial one (no resume).
set -e
cd /home/jaydv/code/prox_learning
export OMP_NUM_THREADS=2
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MLSPACES_ASSETS_DIR=/home/jaydv/code/prox_learning/assets

EXP="${EXP:-PACT_READOUT}"  # ACT, PACT_RAW, PACT_READOUT
SEED="${SEED:-0}"
TASK=pact_place_corridor_v5_ext
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
WANDB_PROJECT=PC_ACT_experiments_eval
SAVE_FIRST_FRAME="${SAVE_FIRST_FRAME:-1}"
EAGER_CAMERAS="${EAGER_CAMERAS:-0}"
TAG="${TAG:-}"

# Brace every var. Bare $TASK_ is an empty name, not "$TASK" + "_".
RUN="${TASK}_${EXP}_s${SEED}_bs${BATCH_SIZE}_cs${CHUNK_SIZE}_lr${LR}_e${EPOCHS}"
CKPT_DIR=/home/jaydv/code/prox_learning/submodules/act/ckpts/$TASK/$RUN
OUTPUT_DIR=/home/jaydv/code/prox_learning/eval_output/${RUN}${TAG}

echo "${RUN}${TAG} n=${NUM_ROLLOUTS}"

if [ ! -f "$CKPT_DIR/$CKPT_NAME" ]; then
  echo "MISSING $CKPT_DIR/$CKPT_NAME (not trained yet)"; exit 4
fi
if [ -f "$OUTPUT_DIR/eval_summary.json" ] || [ -s "$OUTPUT_DIR/episodes.jsonl" ]; then
  DONE_N=$(python -c "import json,sys,os; p=sys.argv[1]; print(json.load(open(p)).get('completed',0) if os.path.isfile(p) else 0)" "$OUTPUT_DIR/eval_summary.json")
  if [ "$DONE_N" -ge "$NUM_ROLLOUTS" ]; then
    echo "DONE $OUTPUT_DIR ($DONE_N/$NUM_ROLLOUTS)"; exit 0
  fi
  echo "REFUSE $OUTPUT_DIR is partial ($DONE_N/$NUM_ROLLOUTS). No resume: rerun with TAG=_rerun into a new dir."; exit 3
fi

EXTRA=()
if [ "${SAVE_FIRST_FRAME}" = "1" ]; then
  EXTRA+=(--save_first_frame)
fi
if [ "${EAGER_CAMERAS}" = "1" ]; then
  EXTRA+=(--eager_cameras --keep_export_sensors)
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
  --wandb_run_name "${RUN}${TAG}_eval" \
  "${EXTRA[@]}" "$@"
