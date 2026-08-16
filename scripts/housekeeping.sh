#!/usr/bin/env bash
# Repo housekeeping for prox_learning.
#
# Reclaims disk in independent tiers. DRY RUN BY DEFAULT — nothing is deleted until
# you pass --apply. Every tier prints what it would remove and how much it would save.
#
#   scripts/housekeeping.sh --tier1                 # preview the safe tier
#   scripts/housekeeping.sh --tier1 --apply         # actually do it
#   scripts/housekeeping.sh --tier1 --tier2 --apply
#   scripts/housekeeping.sh --all --apply           # tiers 1-4 (NOT --gitgc, NOT --venvs)
#
# Tiers, smallest risk first:
#   --tier1   caches + dead output dirs .............. ~1.6 GB   zero risk
#   --tier2   intermediate training checkpoints ...... ~89 GB    no read path exists
#   --tier3   eval rollout .h5 / .mp4 blobs .......... ~14 GB    keeps every eval_summary.json
#   --tier4   regenerable converted datasets ......... ~1.6 GB   rebuild with convert_obstacle_to_act.py
#   --assets  regenerable asset caches ............... ~3.9 GB   re-downloads / rebuilds on demand
#   --venvs   two unused MolmoBot virtualenvs ........ ~9 GB     only if you never run MolmoBot
#   --gitgc   repack + prune loose git objects ....... ~4 GB     safe, does not rewrite history
#
# Modifiers:
#   --keep-every-500   tier2 keeps epoch checkpoints at multiples of 500 (saves ~72 GB not ~89 GB)
#   --keep-headline    tier3 keeps the MP4s for the two headline invisible-cell runs
#
# Not automated on purpose (see README "Housekeeping"): rewriting git history to drop the
# ~20 GB of deleted .h5 blobs still carried in .git. That needs git-filter-repo and a
# force-push across 8 branches.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

APPLY=0
T1=0; T2=0; T3=0; T4=0; ASSETS=0; VENVS=0; GITGC=0
KEEP500=0; KEEP_HEADLINE=0

for arg in "$@"; do
  case "$arg" in
    --apply)          APPLY=1 ;;
    --tier1)          T1=1 ;;
    --tier2)          T2=1 ;;
    --tier3)          T3=1 ;;
    --tier4)          T4=1 ;;
    --assets)         ASSETS=1 ;;
    --venvs)          VENVS=1 ;;
    --gitgc)          GITGC=1 ;;
    --all)            T1=1; T2=1; T3=1; T4=1 ;;
    --keep-every-500) KEEP500=1 ;;
    --keep-headline)  KEEP_HEADLINE=1 ;;
    -h|--help)        sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown flag: $arg" >&2; exit 2 ;;
  esac
done

if [ $((T1+T2+T3+T4+ASSETS+VENVS+GITGC)) -eq 0 ]; then
  sed -n '2,30p' "$0"
  exit 0
fi

# reap() runs on the right-hand side of a pipe, i.e. in a subshell, so the running
# total has to live in a file rather than a variable.
TOTAL_FILE="$(mktemp)"
trap 'rm -f "$TOTAL_FILE"' EXIT
echo 0 > "$TOTAL_FILE"

add_total() { echo $(( $(cat "$TOTAL_FILE") + $1 )) > "$TOTAL_FILE"; }

hr()   { printf '\n\033[1m%s\033[0m\n' "$*"; }
note() { printf '  %s\n' "$*"; }

# human_bytes <bytes>
human_bytes() { numfmt --to=iec --suffix=B "${1:-0}" 2>/dev/null || echo "${1:-0}B"; }

# Sum the sizes of a NUL-separated file list on stdin, then delete it if --apply.
# usage: <producer> | reap "<label>"
reap() {
  local label="$1" bytes=0 count=0 tmp
  tmp="$(mktemp)"
  cat > "$tmp"
  count=$(tr -cd '\0' < "$tmp" | wc -c)
  if [ "$count" -gt 0 ]; then
    bytes=$(du -sc --files0-from="$tmp" 2>/dev/null | tail -1 | cut -f1)
    bytes=$((bytes * 1024))
  fi
  add_total "$bytes"
  printf '  %-56s %6s items  %10s\n' "$label" "$count" "$(human_bytes "$bytes")"
  if [ "$APPLY" -eq 1 ] && [ "$count" -gt 0 ]; then
    xargs -0 rm -rf < "$tmp"
  fi
  rm -f "$tmp"
}

if [ "$APPLY" -eq 1 ]; then
  hr "APPLYING — files will be deleted"
else
  hr "DRY RUN — nothing will be deleted (add --apply)"
fi

# ---------------------------------------------------------------- tier 1: caches + dead dirs
if [ "$T1" -eq 1 ]; then
  hr "tier 1 — caches and dead output directories"

  find . -path ./.git -prune -o -type d -name __pycache__ -print0 \
    | reap "__pycache__ directories"

  find . -path ./.git -prune -o -type d -name '*.egg-info' -print0 \
    | reap "*.egg-info directories"

  # Aborted smoke tests and n=10 debug runs. None carry eval_summary.json; none are
  # referenced by any doc or script. n=50 is the declared floor for real results.
  find eval_output -maxdepth 1 \( -name '_smoke_*' -o -name 'smoke_pla_v3_full*' \
       -o -name '*_diag10' -o -name '*_diag10_fixed' -o -name 'logs_v2' \) -print0 2>/dev/null \
    | reap "eval_output smoke + diag10 + stdout logs"

  # Abandoned mug task (see README history) and the pre-PACT baseline whose numbers
  # are already transcribed into the README results table.
  find eval_output -maxdepth 1 \( -name 'act_house1_mug_v*' \
       -o -name 'act_obstacle_baseline_v1_*' \) -print0 2>/dev/null \
    | reap "eval_output abandoned mug + pre-PACT baseline"

  # v1 PACT evals: superseded by the v2 grid, and the eq50 family ran with temporal
  # aggregation ON, which the README explicitly flags as invalid.
  find eval_output -maxdepth 1 \( -name 'eq50_*' -o -name 'eval_vanilla' \
       -o -name 'eval_trunk_strict' -o -name 'eval_delta_strict' \) -print0 2>/dev/null \
    | reap "eval_output superseded v1 PACT runs"

  # Local wandb mirrors. Newest run is 2026-06-17, all pre-v2, all synced to the cloud.
  find . -maxdepth 3 -type d -name wandb \
       \( -path './wandb' -o -path './submodules/act/wandb' \) -print0 2>/dev/null \
    | reap "local wandb run mirrors (cloud copies remain)"

  # Orphaned checkpoints at the ckpts root: epochs 0/100/200 with no best/last.
  find submodules/act/ckpts -maxdepth 2 -name 'policy_epoch_*.ckpt' -print0 2>/dev/null \
    | reap "orphaned checkpoints from an aborted run"

  find .git/objects/pack -maxdepth 1 -name 'tmp_pack_*' -print0 2>/dev/null \
    | reap "leftover git temp packs"
fi

# ---------------------------------------------------------------- tier 2: checkpoint pruning
if [ "$T2" -eq 1 ]; then
  hr "tier 2 — intermediate training checkpoints"
  note "Nothing in this repo reads policy_epoch_*: every evaluator hardcodes"
  note "policy_best.ckpt, and all 24 eval_summary.json files record policy_best.ckpt."
  note "policy_best.ckpt and policy_last.ckpt are always kept."
  if [ "$KEEP500" -eq 1 ]; then
    note "--keep-every-500: keeping epochs at multiples of 500."
    find submodules/act/ckpts -mindepth 2 -name 'policy_epoch_*_seed_*.ckpt' -print0 \
      | while IFS= read -r -d '' f; do
          n="$(basename "$f" | sed -E 's/policy_epoch_([0-9]+)_seed_.*/\1/')"
          [ $((n % 500)) -eq 0 ] || printf '%s\0' "$f"
        done \
      | reap "policy_epoch_*.ckpt (keeping multiples of 500)"
  else
    find submodules/act/ckpts -mindepth 2 -name 'policy_epoch_*_seed_*.ckpt' -print0 \
      | reap "policy_epoch_*.ckpt (keeping best + last only)"
  fi
fi

# ---------------------------------------------------------------- tier 3: eval blobs
if [ "$T3" -eq 1 ]; then
  hr "tier 3 — eval rollout blobs"
  note "eval_summary.json holds every per-episode metric and is never touched here."
  note "Only raw trajectory h5 and rollout videos are removed."

  find eval_output -name 'trajectories_batch_*.h5' -print0 2>/dev/null \
    | reap "eval_output trajectory .h5"

  if [ "$KEEP_HEADLINE" -eq 1 ]; then
    note "--keep-headline: keeping MP4s under vanilla_v2_invisible + pact_raw_v2_invisible."
    find eval_output -name '*.mp4' \
         -not -path 'eval_output/vanilla_v2_invisible/*' \
         -not -path 'eval_output/pact_raw_v2_invisible/*' -print0 2>/dev/null \
      | reap "eval_output rollout .mp4 (headline runs kept)"
  else
    find eval_output -name '*.mp4' -print0 2>/dev/null \
      | reap "eval_output rollout .mp4"
  fi
fi

# ---------------------------------------------------------------- tier 4: regenerable datasets
if [ "$T4" -eq 1 ]; then
  hr "tier 4 — regenerable converted datasets"
  note "Rebuild either with scripts/convert_obstacle_to_act.py from"
  note "assets/datagen/hybrid_obstacle_v1/FrankaSkinHybridObstacleConfig/20260612_183855."
  note "obstacle_prox_v2 (the live 105-episode training set) is NOT touched."
  find act_style_data -maxdepth 1 \( -name 'obstacle_v1' -o -name 'obstacle_prox_v1' \) \
       -print0 2>/dev/null \
    | reap "act_style_data/{obstacle_v1,obstacle_prox_v1}"
fi

# ---------------------------------------------------------------- regenerable asset caches
if [ "$ASSETS" -eq 1 ]; then
  hr "regenerable asset caches"
  note "assets/ is the MolmoSpaces asset root (MLSPACES_ASSETS_DIR points here)."
  note "Both items below are caches that rebuild themselves; the working trees stay."

  # assets/prox_learning_data is a clone of the HuggingFace dataset repo
  # git@hf.co:datasets/jdvakil/prox_learning_data. Its LFS object store duplicates
  # the checked-out files byte for byte. Re-clone if you ever need the history.
  find assets/prox_learning_data -maxdepth 1 -name '.git' -print0 2>/dev/null \
    | reap "assets/prox_learning_data/.git (HF LFS object store)"

  # MolmoSpaces rebuilds these LMDB caches on demand from assets/{scenes,objects,grasps}.
  find assets -maxdepth 1 -name '.lmdb' -print0 2>/dev/null \
    | reap "assets/.lmdb (MolmoSpaces LMDB caches)"
fi

# ---------------------------------------------------------------- venvs
if [ "$VENVS" -eq 1 ]; then
  hr "MolmoBot virtualenvs"
  note "The MolmoBot submodule is not imported anywhere in this project."
  note "Recreate with 'uv sync' inside each package if you ever need them."
  find submodules/MolmoBot -maxdepth 3 -type d -name '.venv' -print0 2>/dev/null \
    | reap "submodules/MolmoBot/*/.venv"
fi

# ---------------------------------------------------------------- git gc
if [ "$GITGC" -eq 1 ]; then
  hr "git gc — repack and prune unreachable objects"
  note "Does NOT rewrite history. Reclaims loose objects and leftover temp packs."
  before=$(du -sk .git | cut -f1)
  if [ "$APPLY" -eq 1 ]; then
    git reflog expire --expire=now --all
    git gc --prune=now --aggressive
    after=$(du -sk .git | cut -f1)
    saved=$(( (before - after) * 1024 ))
    add_total "$saved"
    printf '  %-56s %10s\n' ".git shrank by" "$(human_bytes "$saved")"
  else
    printf '  %-56s %10s\n' ".git currently" "$(human_bytes $((before * 1024)))"
    note "(run with --apply to repack; expect roughly 4 GB back)"
  fi
fi

hr "total: $(human_bytes "$(cat "$TOTAL_FILE")")"
if [ "$APPLY" -eq 0 ]; then
  echo "  dry run — re-run with --apply to delete"
fi
