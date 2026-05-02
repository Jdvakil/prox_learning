#!/usr/bin/env bash
# Eval every checkpoint x every task (PROJECT.md §4.1).
# Output: reports/eval/<method>_<task>.json, then a summary table.
set -euo pipefail
cd "$(dirname "$0")/.."

TASKS=("pnp" "near_contact" "pnp_color" "pnp_next_to")
N=${N_EPISODES:-100}
SEED=${SEED_BASE:-42}
mkdir -p reports/eval

declare -A METHODS=(
  [PLA]="runs/pla_concat_v1/best.pt"
  ["VLM-only ACT"]="runs/vlm_only_baseline_v1/best.pt"
  [WristOnly]="runs/ablation_wrist_only_v1/best.pt"
  [Handcrafted]="runs/ablation_handcrafted_v1/best.pt"
  [Conv2D]="runs/ablation_conv2d_v1/best.pt"
  [CrossAttn]="runs/ablation_cross_attn_v1/best.pt"
)

for name in "${!METHODS[@]}"; do
  ckpt=${METHODS[$name]}
  if [[ ! -f "$ckpt" ]]; then
    echo "skip $name (no checkpoint at $ckpt)"
    continue
  fi
  for task in "${TASKS[@]}"; do
    safe_name=$(echo "$name" | tr ' /' '__')
    out="reports/eval/${safe_name}_${task}.json"
    echo "=== $name on $task -> $out"
    python -m pla.eval.run_eval \
      --checkpoint "$ckpt" \
      --task "$task" \
      --n-episodes "$N" \
      --seed-base "$SEED" \
      --method-name "$name" \
      --out "$out"
  done
done

echo
echo "Aggregate table:"
python -m pla.eval.run_eval --print-table reports/eval/
