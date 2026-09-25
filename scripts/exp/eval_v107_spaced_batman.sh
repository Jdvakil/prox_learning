#!/usr/bin/env bash
# ./scripts/exp/eval_v107_spaced_batman.sh
# T-107 checkpoints (batman v107_spaced, 210 eps, table_camera + wrist) re-evaluated on the
# eval_act_place.py protocol with the train-matched camera. Dir tag _cam58 keeps the original
# T-107 dirs (eval_act_v107spaced.py, fov 45 rename, query history) untouched.
# table_camera = hybrid exo_camera_1 pose at fovy 58 (batman publish render). The old rename
# rendered it at 45 (zoomed in); TABLE_CAMERA_FOV=45 reproduces that (dir tag _cam45). README §0.1.
# One run. Knobs below; EXP / SEED / NUM_ROLLOUTS / SEED_BASE / HISTORY / TAG can be set from the
# environment (EXP=ACT ./scripts/exp/eval_v107_spaced_batman.sh). Protocol = eval_act_v1011d.py (fast path) with the
# v107_spaced world swapped in by eval_act_place.py. Skips a finished dir; refuses a partial one (no resume).
set -e
cd /home/jaydv/code/prox_learning
export OMP_NUM_THREADS=2
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MLSPACES_ASSETS_DIR=/home/jaydv/code/prox_learning/assets

ENV=v107_spaced
EXP="${EXP:-PACT_READOUT}"  # ACT, PACT_RAW, PACT_READOUT
SEED="${SEED:-0}"
TASK=pact_place_corridor_v107_spaced
CHUNK_SIZE=50
BATCH_SIZE=8
LR=1e-5
EPOCHS=2000
NUM_ROLLOUTS="${NUM_ROLLOUTS:-50}"
SEED_BASE="${SEED_BASE:-2026}"
SKIN=egl
# consecutive = 8-step causal skin window at each query (readout train-matched; T-1011d protocol).
HISTORY="${HISTORY:-consecutive}"
SKIN_SUBSTEPS=snapshot
CAMERAS="table_camera wrist_camera"
# 58 = train-matched batman table camera. 45 = old renamed exo view (T-107 path).
TABLE_CAMERA_FOV="${TABLE_CAMERA_FOV:-58}"
CKPT_NAME=policy_best.ckpt
WANDB_PROJECT=PC_ACT_experiments_eval
# Per-link keep rate. 1=all, 0=drop all, 0.5=Poisson 50% on each link.
SENSOR_KEEP_FRAC="${SENSOR_KEEP_FRAC:-1}"
# Policy cameras fed as black frames at inference (skin + qpos unchanged): all, or names
# from CAMERAS, space-separated. Output dir gets _blind (all) or _no<cam> (exo_camera_1 -> _noexo).
BLANK_CAMERAS="${BLANK_CAMERAS:-}"
SAVE_FIRST_FRAME="${SAVE_FIRST_FRAME:-1}"
SENSOR_MASK_FIXED=0
# 1 = original slow path (per-substep skin-camera poses + object_image_points). A/B only.
EAGER_CAMERAS="${EAGER_CAMERAS:-0}"
# Extra output-dir suffix for a second run of the same ckpt (e.g. TAG=_rerun2).
TAG="${TAG:-}"

# Brace every var. Bare $TASK_ is an empty name, not "$TASK" + "_".
RUN="${TASK}_${EXP}_s${SEED}_bs${BATCH_SIZE}_cs${CHUNK_SIZE}_lr${LR}_e${EPOCHS}"
CKPT_DIR=/home/jaydv/code/prox_learning/submodules/act/ckpts/$TASK/$RUN
PCT=$(python -c "print(int(round(100*float('${SENSOR_KEEP_FRAC}'))))")
OUT_TAG=""
if [ "$PCT" -ne 100 ]; then
  OUT_TAG="_keep${PCT}"
fi
for C in ${BLANK_CAMERAS}; do
  if [ "$C" = "all" ]; then OUT_TAG="${OUT_TAG}_blind"; else OUT_TAG="${OUT_TAG}_no${C%%_*}"; fi
done
OUT_TAG="${OUT_TAG}_cam${TABLE_CAMERA_FOV}${TAG}"
OUTPUT_DIR=/home/jaydv/code/prox_learning/eval_output/${RUN}${OUT_TAG}

echo "${RUN}${OUT_TAG} env=${ENV} n=${NUM_ROLLOUTS} p=${SENSOR_KEEP_FRAC}"

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

EXTRA=(--sensor_keep_frac "${SENSOR_KEEP_FRAC}" --table_camera_fov "${TABLE_CAMERA_FOV}")
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

python eval_act_place.py --env "$ENV" \
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
  --wandb_run_name "${RUN}${OUT_TAG}_eval" \
  "${EXTRA[@]}" "$@"
