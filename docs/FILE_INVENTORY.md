# File inventory

Every file in the repo (excluding `submodules/`, `runs/`, `data/`, build
artifacts) with: purpose, line count (Day 1 baseline), and which doc/test
covers it. Use this as the index when looking for "where is the function
that does X?".

For a higher-level view see [pla/README.md](../pla/README.md) and the
per-subfolder READMEs.

---

## Top-level

| file                  | LOC | purpose                                                          |
|-----------------------|-----|------------------------------------------------------------------|
| `README.md`           | -   | repo entry point, quick start, layout                            |
| `pyproject.toml`      | -   | setuptools build config + deps                                   |
| `.gitignore`          | -   | runs/, data/, build artifacts excluded                           |
| `MEMORY.md`           | -   | (auto-memory index, Claude session state)                        |

## docs/

| file                       | purpose                                                                        |
|----------------------------|--------------------------------------------------------------------------------|
| `README.md`                | docs index — points at the files below                                         |
| `PROJECT.md`               | scientific brief; central reference for claims and references                  |
| `TIMELINE.md`              | 26-day plan + risk register                                                    |
| `STATUS.md`                | live status of what's done vs. not                                             |
| `IMPLEMENTATION_LOG.md`    | append-only chronicle of what was implemented and why                           |
| `SANITY_CHECKS.md`         | every check command + verbatim output                                          |
| `ARCHITECTURE.md`          | tensor shapes, module contracts, invariants                                    |
| `DESIGN_DECISIONS.md`      | every non-obvious choice with rationale and alternatives                       |
| `FILE_INVENTORY.md`        | this file                                                                      |
| `DATASET.md`               | (legacy) skin_pick_fixed_v1 stats                                              |
| `SKIN_PIPELINE.md`         | (legacy) URDF/MJCF skin pipeline reference                                     |
| `CVAE.md`                  | (legacy) skin-proximity CVAE pretrain proof                                    |

## pla/ (Python package)

```
pla/
├── __init__.py
├── README.md                     -> guide to package layout
├── data/         (collection, schema, normalization, DataLoader)
├── sim/          (URDF/MJCF skin builder + ToF rendering)
├── models/       (encoder, VLM, fusion, ACT, PLA, baselines)
├── train/        (training loop + losses)
├── eval/         (eval harness, bootstrap stats, sensor importance)
├── ablations/    (ablation orchestration)
├── checks/       (pre-training sanity checks)
└── viz/          (heatmaps, composite videos, dataset plots)
```

### `pla/data/` (data layer)

| file              | LOC | purpose                                                                  |
|-------------------|-----|--------------------------------------------------------------------------|
| `__init__.py`     | 26  | exports: PLADataset, collate_pla, compute_stats, verify_dataset, ...     |
| `README.md`       | -   | scientific motivation + sanity-check checklist                           |
| `collect.py`      | 202 | episode HDF5 writer + `--dry-run` synthetic mode                         |
| `dataset.py`      | 143 | sliding-window PLADataset                                                 |
| `normalize.py`    | 107 | per-channel mean/std on training files                                   |
| `verify.py`       | 133 | Day-2 schema + prox-informative checker                                  |
| `schema.py`       | 68  | structural validator + proximity_informative_fraction                    |
| `stats.py`        | 141 | dataset-level summary statistics + plot helpers                          |
| `cvae_dataset.py` | 112 | (legacy) skin-CVAE pretrain Dataset                                      |

### `pla/sim/` (simulation layer)

| file                          | LOC | purpose                                                       |
|-------------------------------|-----|---------------------------------------------------------------|
| `__init__.py`                 | 0   | (empty)                                                       |
| `README.md`                   | -   | skin pipeline contract + sanity-check checklist               |
| `tof.py`                      | 149 | `ToFSensorArray` (cached renderer) + `extend_obs_with_tof`     |
| `build_mjcf.py`               | 632 | URDF -> MJCF builder; emits one camera per sensor frame       |
| `fix_sensor_orientations.py`  | 170 | post-process URDF so sensor +Z faces outward                  |
| `patch_mjcf.py`               | 76  | patch existing MJCF with corrected orientations                |

### `pla/models/` (model layer)

| file                     | LOC | purpose                                                          |
|--------------------------|-----|------------------------------------------------------------------|
| `__init__.py`            | 36  | exports: PLA, VLMOnlyACT, ProximityEncoder, ACTDecoder, ...      |
| `README.md`              | -   | dataflow diagram + design rationale + sanity-check checklist     |
| `proximity_encoder.py`   | 130 | shared MLP encoder + Handcrafted + Conv2D ablation variants      |
| `vlm_backbone.py`        | 163 | FrozenMolmo2 + DummyVLBackbone                                    |
| `fusion.py`              | 74  | ModalityFusion: concat (primary) + cross_attn (ablation)         |
| `act.py`                 | 151 | ACTDecoder — Zhao 2023 (CVAE encoder + transformer decoder)       |
| `pla.py`                 | 189 | unified PLA class (vlm_only, encoder_type, sensor_mask flags)    |
| `baselines.py`           | 55  | VLMOnlyACT alias + PropOnlyMLP floor                              |
| `cvae.py`                | 85  | (legacy) skin-CVAE pretrain proof                                 |

### `pla/train/` (training layer)

| file                  | LOC | purpose                                                          |
|-----------------------|-----|------------------------------------------------------------------|
| `__init__.py`         | 10  | exports: train_loop, build_model_from_cfg, act_loss              |
| `README.md`           | -   | training loop walkthrough + grad-norm interpretation              |
| `train.py`            | 287 | unified training entry point — handles every config               |
| `losses.py`           | 27  | act_loss (L1 + 10*KL) library helper                              |
| `train_pla.py`        | 42  | (legacy) thin wrapper around `pla.train.train`                    |
| `train_baseline.py`   | 37  | (legacy) thin wrapper                                             |
| `train_cvae.py`       | 147 | (legacy) skin-CVAE pretrain                                       |

### `pla/eval/` (evaluation layer)

| file                   | LOC | purpose                                                          |
|------------------------|-----|------------------------------------------------------------------|
| `__init__.py`          | 32  | exports: bootstrap_ci, paired_bootstrap_p, REGISTRY, ...          |
| `README.md`            | -   | evaluation protocol + statistical claim + sanity-check checklist |
| `run_eval.py`          | 228 | per-method/per-task eval + `print_results_table`                  |
| `bootstrap.py`         | 53  | bootstrap_ci (10000 resamples) + paired_bootstrap_p              |
| `tasks.py`             | 81  | TaskSpec REGISTRY for the 4 tasks                                 |
| `sensor_importance.py` | 171 | post-hoc per-sensor masking sweep                                 |
| `failure_analysis.py`  | 80  | rule-based failure-mode categorizer + outcome serializer          |

### `pla/ablations/`

| file                  | LOC | purpose                                                          |
|-----------------------|-----|------------------------------------------------------------------|
| `__init__.py`         | 16  | docs only                                                        |
| `README.md`           | -   | ablation matrix + sanity-check checklist                          |
| `run_ablations.py`    | 65  | sequential subprocess launcher / tmux command printer            |

### `pla/checks/`

| file                          | LOC | purpose                                                       |
|-------------------------------|-----|---------------------------------------------------------------|
| `__init__.py`                 | 12  | exports: grad_norm, assert_learning, per_param_grad_norms     |
| `README.md`                   | -   | sanity-check menu + when-to-run table                         |
| `forward_pass.py`             | 89  | end-to-end PLA forward+backward+inference test                 |
| `grad_norm.py`                | 119 | grad-norm library API + CLI driver                            |
| `depth_reconstruction.py`     | 263 | (legacy) reconstruct known scene from sensor readings          |
| `replay_mjcf.py`              | 173 | (legacy) replay recorded trajectory against MJCF               |

### `pla/viz/`

| file                | LOC | purpose                                                            |
|---------------------|-----|--------------------------------------------------------------------|
| `__init__.py`       | 0   | (empty — module is import-light by design)                          |
| `README.md`         | -   | paper figure menu + style requirements                             |
| `heatmap.py`        | 31  | ToF heatmap sequence + sensor-importance heatmap helpers           |
| `composite.py`      | 142 | composite trajectory videos                                        |
| `pointcloud.py`     | 187 | sensor pointcloud reconstruction (3D)                              |
| `dataset_plots.py`  | 237 | depth histograms, coverage stats, per-sensor stats                 |
| `cvae_plots.py`     | 213 | (legacy) CVAE pretrain plots                                       |

## configs/

| file                                          | LOC | purpose                                              |
|-----------------------------------------------|-----|------------------------------------------------------|
| `README.md`                                   | -   | flag reference + flat schema convention              |
| `train/pla.yaml`                              | 39  | PLA — headline config                                |
| `train/act_baseline.yaml`                     | 33  | VLM-only baseline (one-flag-different from pla.yaml) |
| `train/ablation_wrist_only.yaml`              | 36  | wrist-only sensor mask ablation                      |
| `train/ablation_handcrafted.yaml`             | 33  | handcrafted encoder ablation                         |
| `train/ablation_conv2d.yaml`                  | 33  | conv2d encoder ablation                              |
| `train/ablation_cross_attn.yaml`              | 32  | cross-attention fusion ablation                      |
| `train/cvae.yaml`                             | 18  | (legacy) CVAE pretrain config                        |
| `data/near_contact.yaml`                      | 19  | near-contact task definition (PRIMARY)                |
| `data/standard_pnp.yaml`                      | 14  | standard PnP task                                     |
| `eval/default.yaml`                           | 19  | seed_base=42, n_episodes=100                          |

## scripts/

| file                  | purpose                                                                 |
|-----------------------|-------------------------------------------------------------------------|
| `README.md`           | shell-wrapper menu + run order                                          |
| `collect_data.sh`     | tmux launcher for `pla.data.collect`                                    |
| `train_pla.sh`        | one-line `pla.train.train --config configs/train/pla.yaml`               |
| `train_baselines.sh`  | one-line baseline launcher                                              |
| `run_ablations.sh`    | wraps `pla.ablations.run_ablations`                                     |
| `eval_all.sh`         | matrix eval (every method × every task) + table aggregator              |
| `build_skin_mjcf.py`  | Blender JSON -> MJCF camera bodies                                       |
| `verify_skin.py`      | empty-scene self-hit detector                                            |

## assets/

| dir                  | purpose                                                                  |
|----------------------|--------------------------------------------------------------------------|
| `README.md`          | URDF/MJCF flow + what's tracked vs not                                   |
| `urdf/`              | URDF source-of-truth for sensor poses                                    |
| `mjcf/`              | built MJCF + sensor_sites.json + reference XMLs                          |
| `reference_images/`  | small PNGs for visual-regression tests                                   |

## reports/

| dir                | purpose                                                                    |
|--------------------|----------------------------------------------------------------------------|
| `README.md`        | what's tracked vs gitignored; how to regenerate                            |
| `figures/`         | paper figures (PDF) — tracked                                              |
| `tables/`          | results tables (JSON, LaTeX) — tracked                                     |
| `videos/`          | composite trajectory MP4s — gitignored, regenerable                        |
| `logs/`            | training/collection logs — gitignored                                      |
| `eval/`            | per-method/task JSON output — gitignored                                   |
| `checks/`          | sanity-check JSON output — gitignored                                      |

## Aggregate stats

| metric                                                | value           |
|-------------------------------------------------------|-----------------|
| Python files in `pla/` + `scripts/`                   | 49              |
| Python LOC in `pla/`                                  | ~5,800          |
| Markdown files (READMEs + docs)                       | 21              |
| Per-folder READMEs                                    | 12              |
| Smoke-importable modules                              | 29 / 29 (100%)  |
| AST-parseable files                                   | 49 / 49 (100%)  |
| YAML configs                                          | 11 (276 LOC)    |
| Shell scripts                                         | 5               |

## What's *not* in this inventory (and why)

- `submodules/` — vendored MolmoBot, MolmoSpaces, ACT. Tracked via git
  submodule; not part of `pla/`.
- `runs/` — training outputs. Gitignored. Regenerable via training scripts.
- `data/` — collected trajectories. Gitignored. Regenerable via collection
  scripts.
- `build/`, `dist/`, `*.egg-info/` — Python build artifacts. Gitignored.
- `__pycache__/`, `*.pyc` — bytecode. Gitignored.
- `paper/` — LaTeX source for the submission. Tracked, but not Python.
