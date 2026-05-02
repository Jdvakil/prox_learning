#!/usr/bin/env bash
# Train the VLM-only ACT baseline. PropOnlyMLP is a sanity floor only and
# is invoked separately (it does not use the train_loop pipeline).
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pla.train.train --config configs/train/act_baseline.yaml "$@"
