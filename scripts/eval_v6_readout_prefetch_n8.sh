#!/usr/bin/env bash
# v6 readout on eval_act_v6.py with the 8-step causal window (prefetch, not
# every control step). Same worker layout as causal-fix n8: 8 workers, 4/GPU, 5s stagger.
set -euo pipefail

PY="${PY:-/home/ekshan/miniconda3/envs/mlspaces-warp111/bin/python}"
ROOT="${ROOT:-/home/ekshan/prox_learning_v6_train}"
ACT="$ROOT/submodules/act"
MOLMO="${MOLMO:-/home/ekshan/prox_learning_v1011d_n100/submodules/molmospaces}"
CKPT="$ACT/ckpts/pact_pick_n_place_v6/20260915_115435_pact_pick_n_place_v6_readout_s0"
OUT="$ROOT/eval_output/two_object_v6proto/readout_n48"
LOG="$ROOT/logs/v6_eval_v6proto/readout"
CPU_CORES="${CPU_CORES:-0-19}"
N=48
NWORK=8
STAGGER_S="${STAGGER_S:-5}"

mkdir -p "$LOG"
echo "[v6proto-n8] host=$(hostname) workers=$NWORK (4/gpu) cpus=$CPU_CORES stagger=${STAGGER_S}s history=consecutive prefetch=8 $(date -u +%FT%TZ)"

for i in $(seq 0 $((NWORK - 1))); do
  s=$(( i * N / NWORK ))
  e=$(( (i + 1) * N / NWORK ))
  dest="$OUT/shards/${s}_${e}"
  mkdir -p "$dest"
  gpu=$(( i % 2 ))
  CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_EGL_DEVICE_ID=0 \
    PYTHONPATH="$MOLMO:$ACT:$ROOT:$ROOT/scripts" \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    MOLMOSPACES_PACT_V1010="$MOLMO" MOLMOSPACES_PACT_PLACE="$MOLMO" \
    MLSPACES_ASSETS_DIR=/home/ekshan/prox_learning/assets WANDB_MODE=offline \
    taskset -c "$CPU_CORES" "$PY" -u "$ROOT/eval_act_v6.py" \
      --ckpt_dir "$CKPT" --output_dir "$dest" \
      --molmo "$MOLMO" --num_rollouts "$N" --spread_cells \
      --history consecutive --skin rays --skin_substeps snapshot \
      --row_start "$s" --row_end "$e" \
      > "$LOG/shard_${s}_${e}.log" 2>&1 &
  echo $! > "$LOG/shard_${s}_${e}.pid"
  echo "[v6proto-n8] shard ${s}_${e} gpu=$gpu pid $!"
  sleep "$STAGGER_S"
done
wait
echo "[v6proto-n8] done $(date -u +%FT%TZ)"
