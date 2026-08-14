#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Evaluate the constant-blur vision-only ACT baselines IN-ENV, on SHARP frames.
#
# For every ckpt folder matching *blurC*_v2 under ckpts/obstacle_pact_v2/, runs
# eval_act_obstacle.py across the three obstacle cells (visible / invisible /
# free), N rollouts each, temporal aggregation OFF (the README-correct setting).
# Eval always renders sharp frames — the blur was a training-only handicap.
# These ckpts carry no proximity, so eval auto-detects vision-only (no flags).
#
# Then prints a table (success + collision per cell per sigma) and, if present,
# the sharp vanilla_v2 anchor from eval_output/vanilla_v2_<cell>/.
#
# Usage:
#   conda activate mlspaces            # or rely on the hardcoded PYTHON below
#   ./eval_blur_baseline.sh                 # N=50 rollouts, cells: invisible free visible
#   ./eval_blur_baseline.sh 25              # N=25 (recommended first pass; ~1/2 the time/RAM)
#   ./eval_blur_baseline.sh 15 invisible free   # custom N + subset of cells
#
# COST: ~3 min/rollout. 3 models x 3 cells x N. At N=50 that is ~22 h serial;
# start with N=15-25. Runs SERIAL on purpose — each rollout retains obs history
# (~0.5 GB/episode), so a 50-rollout process peaks ~25 GB RSS; don't parallelize
# blindly on this box.
# ---------------------------------------------------------------------------
set -uo pipefail

ACT_DIR="/home/jaydv/code/prox_learning/submodules/act"
REPO="/home/jaydv/code/prox_learning"
PYTHON="${PYTHON:-/opt/conda/envs/mlspaces/bin/python}"

# Preflight: a driver/library version mismatch (e.g. the kernel module upgraded
# without a reboot) makes every rollout raise cudaGetDeviceCount error 804 while
# the loop happily continues, burning hours and writing no eval_summary.json.
# Fail loudly up front instead.
if ! "$PYTHON" -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
  echo "[blur-eval] FATAL: torch.cuda.is_available() is False -- refusing to run."
  echo "[blur-eval] Most likely an NVIDIA driver/library version mismatch."
  echo "[blur-eval]   loaded module : $(cat /proc/driver/nvidia/version 2>/dev/null | head -1)"
  echo "[blur-eval]   nvidia-smi    : $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>&1 | head -1)"
  echo "[blur-eval] Fix: sudo reboot   (or reload the modules, then re-run this script)"
  exit 1
fi

N="${1:-50}"
shift || true
CELLS=("$@")
if [ ${#CELLS[@]} -eq 0 ]; then CELLS=(invisible free visible); fi

cd "$ACT_DIR" || { echo "[blur-eval] cannot cd to $ACT_DIR"; exit 1; }

mapfile -t DIRS < <(ls -d ckpts/obstacle_pact_v2/*blurC*_v2 2>/dev/null | sort)
if [ ${#DIRS[@]} -eq 0 ]; then
  echo "[blur-eval] ERROR: no *blurC*_v2 ckpt folders under ckpts/obstacle_pact_v2/"; exit 1
fi

echo "[blur-eval] python : $PYTHON"
echo "[blur-eval] N      : $N rollouts | cells: ${CELLS[*]} | temp_agg OFF | sharp frames"
echo "[blur-eval] models :"; for d in "${DIRS[@]}"; do echo "             $d"; done

for D in "${DIRS[@]}"; do
  RUN=$(basename "$D")
  for CELL in "${CELLS[@]}"; do
    OUT="$REPO/eval_output/${RUN}_${CELL}"
    echo
    echo "==================================================================="
    echo "[blur-eval] $RUN  cell=$CELL  N=$N  -> $OUT"
    echo "==================================================================="
    PYTHONPATH="$ACT_DIR:${PYTHONPATH:-}" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
      "$PYTHON" eval_act_obstacle.py \
        --ckpt_dir "$D" \
        --output_dir "$OUT" \
        --num_rollouts "$N" --chunk_size 100 --temp_agg_off --eval_cell "$CELL"
    [ "${PIPESTATUS[0]}" -ne 0 ] && echo "[blur-eval] !! $RUN cell=$CELL FAILED"
  done
done

echo
echo "==================================================================="
echo "[blur-eval] SUMMARY (success% / collision%, sharp eval)"
echo "==================================================================="
"$PYTHON" - "$REPO" "${CELLS[@]}" <<'PY'
import json, sys, glob, os, re
repo = sys.argv[1]; cells = sys.argv[2:]
def load(run, cell):
    f = os.path.join(repo, "eval_output", f"{run}_{cell}", "eval_summary.json")
    if not os.path.exists(f): return None
    d = json.load(open(f))
    c = d.get("collision", {})
    return (d.get("success_rate"), c.get("collision_rate"), c.get("strict_success_rate"), d.get("total"))
# blur models, sorted by sigma. NOTE: run dirs carry a timestamp prefix
# (20260724_010935_vanilla_blurC2_v2), so the glob must not be anchored on
# "vanilla" and the captured run name must include that prefix -- it is part of
# the eval_output directory name that load() reconstructs.
runs = sorted(
    {re.match(r"(.*blurC\d+_v2)_(?:" + "|".join(map(re.escape, cells)) + r")$",
              os.path.basename(p)).group(1)
     for p in glob.glob(os.path.join(repo, "eval_output", "*blurC*_v2_*"))
     if re.match(r"(.*blurC\d+_v2)_(?:" + "|".join(map(re.escape, cells)) + r")$",
                 os.path.basename(p))},
    key=lambda r: int(re.search(r"blurC(\d+)", r).group(1)))
w = max([22] + [len(r) + 2 for r in runs])
hdr = "model".ljust(w) + "".join(f"{c:>26}" for c in cells)
print(hdr); print("-"*len(hdr))
def fmt(v):
    s,c,ss,n = v if v else (None,)*4
    if s is None: return "--"
    return f"succ {s*100:3.0f}% coll {c*100:3.0f}% n={n}"
for r in runs + ["vanilla_v2"]:
    label = r + ("  (sharp anchor)" if r=="vanilla_v2" else "")
    print(label.ljust(w) + "".join(f"{fmt(load(r,c)):>26}" for c in cells))
print("\n(strict_success = grasped+lifted AND contact-free; see each eval_summary.json)")
print("README sharp vanilla_v2 @50: collision free/invis/visible = 60/66/64%, success 22/36/28%")
PY
echo "==================================================================="
