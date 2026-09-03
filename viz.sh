#!/usr/bin/env bash
# Incremental dataset dashboard. No `conda activate` — bash scripts do not
# get that hook. Call the env python instead.
set -euo pipefail
cd "$(dirname "$0")"
exec /opt/conda/envs/mlspaces/bin/python scripts/dataset_viz.py \
    --data data/ --each --cam3d --no-mcap --stride 2 "$@"
