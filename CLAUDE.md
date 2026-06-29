# Workflow constraint

ALWAYS DOCUMENT AND REFER TO THE README IN prox_learning/README.md NO NEW README FILES (you can make new markdown files for reports).

## Datagen run recipe (pref 8 to 12 workers)
```
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export MLSPACES_ASSETS_DIR='/root/prox_learning/assets'
export PYTHONPATH='/root/prox_learning/submodules/molmospaces'

/root/old/.venv/bin/python '/root/prox_learning/scratch/run_prox_necessity_pilot.py'
```
