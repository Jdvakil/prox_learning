#!/usr/bin/env bash
# Ablation ladder (PROJECT.md §4.3, Day 8-9):
#   ablation_wrist_only      — keep only link6 sensors (sensor_mask)
#   ablation_handcrafted     — encoder_type=handcrafted
#   ablation_conv2d          — encoder_type=conv2d
#   ablation_cross_attn      — fusion_type=cross_attn
set -euo pipefail
cd "$(dirname "$0")/.."

python -m pla.ablations.run_ablations "$@"
