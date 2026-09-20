#!/usr/bin/env bash
# ./scripts/exp/test_lazy_cameras.sh <1|2|3|4|5>
# Gate for scripts/pact_eval_lazy_cameras.py. Run inside `conda activate mlspaces`.
#   1  cProfile, 100 steps, eager  -> is update_pose the hot spot?   (~3 min)
#   2  2 episodes, lazy (new default)                                 (~10 min)
#   3  2 episodes, eager (original), same seeds                       (~45 min)
#   4  compare 2 vs 3: episodes.jsonl + first_frames must be identical
#   5  compare 2 vs the running seed_base=0 READOUT paper eval (ep 0,1)
set -e
cd /home/jaydv/code/prox_learning
export OMP_NUM_THREADS=2
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MLSPACES_ASSETS_DIR=/home/jaydv/code/prox_learning/assets

RUN=pact_pick_n_place_v2_v1011d_PACT_READOUT_s0_bs8_cs50_lr1e-5_e2000
CK=submodules/act/ckpts/pact_pick_n_place_v2_v1011d/${RUN}
PAPER_DIR=eval_output/${RUN}
COMMON=(
  --ckpt_dir "$CK"
  --cameras exo_camera_1 wrist_camera
  --seed_base 0
  --skin egl
  --history consecutive
  --skin_substeps snapshot
  --clutter_xy_scale 0.25
  --no_wandb
)

case "${1:-}" in
  1)
    python -m cProfile -o /tmp/ev.prof eval_act_v1011d.py "${COMMON[@]}" \
      --num_rollouts 1 --horizon 100 --eager_cameras --keep_export_sensors \
      --output_dir eval_output/_prof_eager
    python -c "import pstats; pstats.Stats('/tmp/ev.prof').sort_stats('cumtime').print_stats(30)" \
      | tee eval_output/_prof_eager/profile.txt
    ;;
  2)
    python eval_act_v1011d.py "${COMMON[@]}" --num_rollouts 2 --save_first_frame \
      --output_dir eval_output/_ab_lazy 2>&1 | tee eval_output/_ab_lazy.log
    grep -E "lazy prox-camera|omit export|wall=" eval_output/_ab_lazy.log
    ;;
  3)
    python eval_act_v1011d.py "${COMMON[@]}" --num_rollouts 2 --save_first_frame \
      --eager_cameras --keep_export_sensors --output_dir eval_output/_ab_eager 2>&1 | tee eval_output/_ab_eager.log
    grep -E "eager_cameras|keep_export|wall=" eval_output/_ab_eager.log
    ;;
  4)
    python - <<'PY'
import json
a, b = (
    [json.loads(l) for l in open(f"eval_output/{d}/episodes.jsonl")]
    for d in ("_ab_eager", "_ab_lazy")
)
bad = [
    (x["episode_idx"], k, x[k], y.get(k))
    for x, y in zip(a, b)
    for k in x
    if k not in ("lazy_prox_cameras", "export_sensors_dropped") and x[k] != y.get(k)
]
# Contact-frame COUNT fields drift ~1% run to run even on identical code.
# Outcome fields and first_contact_step are the gate.
print("IDENTICAL" if not bad and len(a) == len(b) == 2 else f"DIFFERS {bad}")
PY
    if diff <(cd eval_output/_ab_eager/first_frames && sha256sum *) \
            <(cd eval_output/_ab_lazy/first_frames && sha256sum *); then
      echo FRAMES_IDENTICAL
    else
      echo FRAMES_MISMATCH
    fi
    ;;
  5)
    PAPER_DIR="$PAPER_DIR" python - <<'PY'
import json, os
# Paper jsonl can hold stale records from another --seed_base. Match on (idx, seed).
paper = {
    (r["episode_idx"], r["seed"]): r
    for r in map(json.loads, open(os.environ["PAPER_DIR"] + "/episodes.jsonl"))
}
new = [json.loads(l) for l in open("eval_output/_ab_lazy/episodes.jsonl")]
old = [paper[(r["episode_idx"], r["seed"])] for r in new]
keys = (
    "seed", "success", "ever_success", "hit_bar", "bar_contact_frames",
    "other_environment_frames", "clutter_frames", "collision_free",
    "first_contact_step", "snapshot_renders",
)
bad = [
    (o["episode_idx"], k, o.get(k), n.get(k))
    for o, n in zip(old, new)
    for k in keys
    if o.get(k) != n.get(k)
]
print("MATCHES PAPER RUN" if not bad and len(new) == 2 else f"MISMATCH {bad}")
PY
    ;;
  *)
    sed -n 2,9p "$0"
    exit 1
    ;;
esac
