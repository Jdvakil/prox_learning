# prox_learning — PLA: Peripersonal Language-Action

Code, configs, and assets for the CoRL 2026 submission **PLA: Peripersonal
Language-Action Policies via Whole-Body Time-of-Flight Proximity Sensing**.

- **Researcher:** Jay Vakil — HIRO Lab, CU Boulder
- **Advisor:** Alessandro Roncone
- **Submission deadline:** 2026-05-28

The full project description and 26-day plan live in
[`docs/PROJECT.md`](docs/PROJECT.md) and [`docs/TIMELINE.md`](docs/TIMELINE.md).

## How this repo is structured

Every subfolder has its own `README.md` with: scientific motivation, file
inventory, sanity-check checklist, and run commands. **Read the README in
the folder you are touching before doing any work in it.** The READMEs are
the contract; if code drifts from a README, treat it as a bug.

```
pla/                      importable Python package — all code
├── data/                 collection harness, HDF5 schema, normalization, DataLoader
├── sim/                  URDF/MJCF skin builder, ToF sensor rendering
├── models/               ProximityEncoder, FrozenMolmo2, ModalityFusion, ACT, PLA
├── train/                training entry point + losses + grad-norm checks
├── eval/                 eval runner, bootstrap stats, sensor importance, failure
├── ablations/            ablation orchestration
├── checks/               pre-training sanity checks (forward pass, grad norm, replay)
└── viz/                  paper figures, composite videos, heatmaps

assets/                   MJCF, URDF, reference renders
configs/                  data/, train/, eval/ YAML configs
scripts/                  shell wrappers (collect, train, eval, ablations) + skin tools
docs/                     PROJECT.md, TIMELINE.md, DATASET.md, etc.
reports/                  paper-bound figures, tables, logs (see reports/README.md)
paper/                    LaTeX submission
runs/                     gitignored — training outputs
data/                     gitignored — HDF5 trajectory shards
submodules/               MolmoBot, molmospaces, ACT (git submodules)
```

## Quick start

```bash
# Install the package and submodules.
git submodule update --init --recursive
pip install -e .

# 1. Build + verify the sensor skin.
python scripts/build_skin_mjcf.py --sites assets/mjcf/sensor_sites.json \
    --base-mjcf assets/mjcf/fr3_skin.xml --out assets/mjcf/fr3_skin_blender.xml
python scripts/verify_skin.py --mjcf assets/mjcf/fr3_skin_blender.xml \
    --out reports/checks/skin_verify.json

# 2. Smoke-test the package (no GPU, no real VLM).
python -m pla.checks.forward_pass
python -m pla.checks.grad_norm --config configs/train/pla.yaml --steps 50

# 3. Collect 1000 near-contact trajectories.
bash scripts/collect_data.sh near_contact 1000

# 4. Verify dataset (must show >=30% proximity-informative).
python -m pla.data.verify --data-dir data/raw/near_contact --strict

# 5. Compute training-only normalization stats.
python -m pla.data.normalize --data-dir data/raw/near_contact --out stats.json

# 6. Train baselines first, then PLA. (one-flag-difference!)
bash scripts/train_baselines.sh   # configs/train/act_baseline.yaml
bash scripts/train_pla.sh         # configs/train/pla.yaml

# 7. Ablations.
bash scripts/run_ablations.sh

# 8. Evaluate everything on all 4 tasks (100 episodes each).
bash scripts/eval_all.sh
python -m pla.eval.run_eval --print-table reports/eval/
```

## The headline experiment

The paper's primary scientific claim (PROJECT.md §8) is tested by the delta
between PLA and the **VLM-only ACT** baseline on the **near-contact** task.
That comparison is **one config flag** different from PLA — see
`configs/train/act_baseline.yaml` (`vlm_only: true`). Without this baseline
run, the proximity-sensing claim cannot be substantiated.

The eval harness pairs episode seeds across methods so the paired bootstrap
p-value is valid. The decision rule:

  * `delta > 10 pp` and `p < 0.05` -> strong result
  * `delta < 5 pp` -> debug: dataset coverage, encoder grad norm, task design

## Submodules

| Path                       | Purpose                                                 |
|----------------------------|---------------------------------------------------------|
| `submodules/MolmoBot`      | Vision-language-action backbone + TAMP planning         |
| `submodules/molmospaces`   | procthor-objaverse simulation + FrankaPickandPlace eval |
| `submodules/act`           | Action Chunking Transformer reference                   |
