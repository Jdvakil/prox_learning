#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Constant-blur vision-only ACT baselines on obstacle_prox_v2.
#
# Trains ONE ACT policy per blur sigma. Each policy is RGB + qpos only (the
# dataset carries a proximity channel but no --use_proximity flag is passed, so
# the skin is ignored -> vision-only baseline). Blur is CONSTANT: the same
# Gaussian sigma is applied to every camera frame for the whole run (no anneal);
# evaluation later sees sharp frames. Runs are serial (one finishes before the
# next starts).
#
# Usage:
#   conda activate mlspaces        # (or set PYTHON=/path/to/env/python below)
#   ./train_blur_baseline.sh                # default sweep: sigma 2 4 8
#   ./train_blur_baseline.sh 1 2 4          # custom sweep
#   PYTHON=/opt/conda/envs/mlspaces/bin/python ./train_blur_baseline.sh
#
# Note: at 240x320, blur saturates fast — sigma=2 already erases ~98% of fine
# detail and sigma=4 ~= sigma=8. For a graded curve prefer: ./train_blur_baseline.sh 1 2 4
# ---------------------------------------------------------------------------
set -uo pipefail

ACT_DIR="/home/jaydv/code/prox_learning/submodules/act"
PYTHON="${PYTHON:-/opt/conda/envs/mlspaces/bin/python}"

# blur sigmas: command-line args, else the default sweep
SIGMAS=("$@")
if [ ${#SIGMAS[@]} -eq 0 ]; then SIGMAS=(2 4 8); fi

cd "$ACT_DIR" || { echo "[blur-baseline] cannot cd to $ACT_DIR"; exit 1; }

# preflight: python + torch reachable, dataset present
if ! "$PYTHON" -c "import torch" 2>/dev/null; then
  echo "[blur-baseline] ERROR: '$PYTHON' has no torch. 'conda activate mlspaces' first,"
  echo "                or run: PYTHON=/opt/conda/envs/mlspaces/bin/python $0 ${SIGMAS[*]}"
  exit 1
fi
DATA="/home/jaydv/code/prox_learning/act_style_data/obstacle_prox_v2"
if [ ! -d "$DATA" ]; then
  echo "[blur-baseline] ERROR: dataset not found: $DATA"; exit 1
fi

LOGDIR="$ACT_DIR/ckpts/obstacle_pact_v2/_blur_logs"
mkdir -p "$LOGDIR"

echo "[blur-baseline] python : $PYTHON"
echo "[blur-baseline] sigmas : ${SIGMAS[*]}"
echo "[blur-baseline] task   : obstacle_pact_v2 (vision-only, 105 eps) | epochs=2000 chunk=100 seed=0"
echo "[blur-baseline] logs   : $LOGDIR/<run>.log"

declare -a RESULTS
for S in "${SIGMAS[@]}"; do
  RUN="vanilla_blurC${S}_v2"
  LOG="$LOGDIR/${RUN}.log"
  echo
  echo "==================================================================="
  echo "[blur-baseline] TRAIN sigma=$S  ->  run name: $RUN"
  echo "[blur-baseline] log: $LOG"
  echo "==================================================================="

  PYTHONPATH="$ACT_DIR:${PYTHONPATH:-}" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    "$PYTHON" imitate_episodes.py \
      --task_name obstacle_pact_v2 --policy_class ACT --ckpt_dir ckpts --kl_weight 10 \
      --chunk_size 100 --hidden_dim 512 --dim_feedforward 3200 --batch_size 8 --lr 1e-5 \
      --seed 0 --num_epochs 2000 --blur_sigma0 "$S" --blur_mode constant \
      --wandb_run_name "$RUN" 2>&1 | tee "$LOG"
  status=${PIPESTATUS[0]}

  CDIR=$(grep -oP '(?<=\[ckpt\] saving this run to ).*' "$LOG" | head -1)
  if [ "$status" -ne 0 ]; then
    echo "[blur-baseline] !! sigma=$S FAILED (exit $status) — see $LOG"
    RESULTS+=("sigma=$S  FAILED  ($LOG)")
  else
    echo "[blur-baseline] sigma=$S DONE -> ${CDIR:-<ckpt dir not parsed; check $LOG>}"
    RESULTS+=("sigma=$S  ${CDIR:-?}")
  fi
done

echo
echo "==================================================================="
echo "[blur-baseline] ALL RUNS DONE"
for r in "${RESULTS[@]}"; do echo "   $r"; done
echo
echo "[blur-baseline] newest blurC ckpt folders (paste into the eval loop):"
ls -t "$ACT_DIR/ckpts/obstacle_pact_v2" 2>/dev/null | grep blurC | head -3
echo "==================================================================="
