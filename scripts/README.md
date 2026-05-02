# `scripts/` — shell wrappers and one-off Python tools

Thin convenience scripts that call into `pla.*` modules. Each shell wrapper
exists so the typical CLI invocation is one line. Edit the underlying
Python module if you need to change behavior — the shell wrappers should
stay parameter-thin.

## Files

| file                  | role                                                    |
|-----------------------|---------------------------------------------------------|
| `collect_data.sh`     | data collection: `bash scripts/collect_data.sh near_contact 1000` |
| `train_pla.sh`        | one-shot PLA training launch                            |
| `train_baselines.sh`  | VLM-only + prop-only training                           |
| `run_ablations.sh`    | runs all 4 ablations (sequentially)                     |
| `eval_all.sh`         | runs eval on every checkpoint x every task              |
| `build_skin_mjcf.py`  | Blender JSON -> MJCF camera bodies                      |
| `verify_skin.py`      | empty-scene self-hit detector                           |

## Run order over the timeline

```bash
# Day 1-2: skin
python scripts/build_skin_mjcf.py --sites assets/mjcf/sensor_sites.json \
    --base-mjcf assets/mjcf/fr3_skin.xml --out assets/mjcf/fr3_skin_blender.xml
python scripts/verify_skin.py --mjcf assets/mjcf/fr3_skin_blender.xml \
    --out reports/checks/skin_verify.json

# Day 3-4: collect + verify + normalize
bash scripts/collect_data.sh near_contact 1000
python -m pla.data.verify --data-dir data/raw/near_contact --strict
python -m pla.data.normalize --data-dir data/raw/near_contact --out stats.json

# Day 4-7: train + smoke-eval
bash scripts/train_baselines.sh
bash scripts/train_pla.sh

# Day 8-12: ablations + sensor importance
bash scripts/run_ablations.sh
python -m pla.eval.sensor_importance \
    --checkpoint runs/pla_concat_v1/best.pt --task near_contact \
    --n-episodes 50 --out reports/tables/sensor_importance.json

# Day 13-14: full eval
bash scripts/eval_all.sh
python -m pla.eval.run_eval --print-table reports/eval/
```
