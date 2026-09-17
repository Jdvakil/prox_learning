#!/usr/bin/env bash
# Hub v6 n=48 on the frozen v1011d-protocol evaluator (eval_act_v6.py).
# Sequential arms, 8 workers each (proven stable). New output dir so old
# eval_act_pact_pick_n_place rates are not mixed.
#
# readout uses --history consecutive (8 control-step skin window).
# raw/ACT use --history query (one snapshot per chunk).
set -euo pipefail

PY="${PY:-/home/ekshan/miniconda3/envs/mlspaces-warp111/bin/python}"
ROOT="${ROOT:-/home/ekshan/prox_learning_v6_train}"
ACT="$ROOT/submodules/act"
MOLMO="${MOLMO:-/home/ekshan/prox_learning_v1011d_n100/submodules/molmospaces}"
CKPT_ROOT="$ACT/ckpts/pact_pick_n_place_v6"
OUT_ROOT="$ROOT/eval_output/two_object_v6proto"
LOG_ROOT="$ROOT/logs/v6_eval_v6proto"
CPU_CORES="${CPU_CORES:-0-19}"
N=48
NWORK=8
STAGGER_S="${STAGGER_S:-5}"

ACT_CKPT="$CKPT_ROOT/20260915_113546_act_pick_n_place_v6_s0"
RAW_CKPT="$CKPT_ROOT/20260915_114210_pact_pick_n_place_v6_raw_s0"
READOUT_CKPT="$CKPT_ROOT/20260915_115435_pact_pick_n_place_v6_readout_s0"

run_arm() {
  local arm="$1" ckpt="$2" history="$3"
  local out="$OUT_ROOT/${arm}_n48"
  local log="$LOG_ROOT/$arm"
  mkdir -p "$log"
  echo "[v6proto] arm=$arm history=$history workers=$NWORK $(date -u +%FT%TZ)"
  for i in $(seq 0 $((NWORK - 1))); do
    local s=$(( i * N / NWORK ))
    local e=$(( (i + 1) * N / NWORK ))
    local dest="$out/shards/${s}_${e}"
    mkdir -p "$dest"
    local gpu=$(( i % 2 ))
    CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_EGL_DEVICE_ID=0 \
      PYTHONPATH="$MOLMO:$ACT:$ROOT:$ROOT/scripts" \
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
      MOLMOSPACES_PACT_V1010="$MOLMO" MOLMOSPACES_PACT_PLACE="$MOLMO" \
      MLSPACES_ASSETS_DIR=/home/ekshan/prox_learning/assets WANDB_MODE=offline \
      taskset -c "$CPU_CORES" "$PY" -u "$ROOT/eval_act_v6.py" \
        --ckpt_dir "$ckpt" --output_dir "$dest" \
        --molmo "$MOLMO" --num_rollouts "$N" --spread_cells \
        --history "$history" --skin rays --skin_substeps snapshot \
        --row_start "$s" --row_end "$e" \
        > "$log/shard_${s}_${e}.log" 2>&1 &
    echo $! > "$log/shard_${s}_${e}.pid"
    echo "[v6proto] $arm shard ${s}_${e} gpu=$gpu pid $!"
    sleep "$STAGGER_S"
  done
  wait
  echo "[v6proto] $arm done $(date -u +%FT%TZ)"
}

mkdir -p "$LOG_ROOT"
echo "[v6proto] host=$(hostname) start $(date -u +%FT%TZ)" | tee "$LOG_ROOT/launch.log"
run_arm readout "$READOUT_CKPT" consecutive
run_arm raw "$RAW_CKPT" query
run_arm act "$ACT_CKPT" query
echo "[v6proto] all arms done $(date -u +%FT%TZ)" | tee -a "$LOG_ROOT/launch.log"
