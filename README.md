# prox_learning — proximity-skin sensing & safety for the Franka FR3

A Franka FR3 wearing **40 proximity sensors** in MuJoCo, plus the policies and analysis that
answer one question: *does a proximity skin make a robot arm safer when the cameras cannot see
the obstacle?*

**Current answer (2026-07-05, 50 rollouts per cell):** yes, on the safety axis. A vision-only ACT
policy collides in **66%** of episodes when a hazard bar is hidden from the cameras. The same
policy with a raw-skin token collides in **40%** (Fisher p = .016). Success rate is unchanged.
Everything else in this repo is either the machinery that produced that number, or a follow-up
that did not work.

For the plain-language version of the science, read **`STATUS.md`**. For the latest written
report, read **`reports/2026-08-14/report.md`**. This file is the manual: what every file is,
what to run, and what will bite you.

---

## Contents

1. [Routing table — "I want to…"](#1-routing-table--i-want-to)
2. [How the three pipelines fit together](#2-how-the-three-pipelines-fit-together)
3. [Setup and environment](#3-setup-and-environment)
4. [Repo map](#4-repo-map)
5. [File-by-file: `scripts/`](#5-file-by-file-scripts)
6. [File-by-file: `submodules/act` (the policy)](#6-file-by-file-submodulesact-the-policy)
7. [File-by-file: `submodules/molmospaces` (the simulator)](#7-file-by-file-submodulesmolmospaces-the-simulator)
8. [File-by-file: models and assets](#8-file-by-file-models-and-assets)
9. [Data formats](#9-data-formats)
10. [Method & math — the Safety-CVAE](#10-method--math--the-safety-cvae)
11. [Recipe A — Safety-CVAE and the demos](#11-recipe-a--safety-cvae-and-the-demos)
12. [Recipe B — datagen](#12-recipe-b--datagen)
13. [Recipe C — ACT and PACT](#13-recipe-c--act-and-pact)
14. [Every number in one place](#14-every-number-in-one-place)
15. [Traps](#15-traps)
16. [Decision log](#16-decision-log)
17. [Repo state and housekeeping](#17-repo-state-and-housekeeping)

---

## 1. Routing table — "I want to…"

| I want to… | run | section |
|---|---|---|
| understand the result in plain English | read `STATUS.md` | — |
| read the latest progress report | `reports/2026-08-14/report.md` | — |
| collect new demonstrations | `python -m molmo_spaces.data_generation.main <Config>` | [12](#12-recipe-b--datagen) |
| convert demos into training files | `python -m scripts.convert_obstacle_to_act` | [13](#13-recipe-c--act-and-pact) |
| train a policy | `submodules/act/imitate_episodes.py` | [13](#13-recipe-c--act-and-pact) |
| test a policy | `submodules/act/eval_act_obstacle.py` | [13](#13-recipe-c--act-and-pact) |
| compare two result folders | `scripts/compare_pact.py` | [13](#13-recipe-c--act-and-pact) |
| train the reflex net | `scripts/safety_sweep.py` → `scripts/train_safety_cvae.py` | [11](#11-recipe-a--safety-cvae-and-the-demos) |
| make a demo video | `scripts/safety_*_demo.py` | [11](#11-recipe-a--safety-cvae-and-the-demos) |
| make paper figures | `scripts/figures.py --list` | [5](#5-file-by-file-scripts) |
| rebuild the 40-sensor arm | `scripts/build_hybrid_on_franka_skin.py` | [8](#8-file-by-file-models-and-assets) |
| know what a file is | — | [5](#5-file-by-file-scripts)–[8](#8-file-by-file-models-and-assets) |
| free disk space | `scripts/housekeeping.sh` | [17](#17-repo-state-and-housekeeping) |

---

## 2. How the three pipelines fit together

```
             submodules/molmospaces              scripts/              submodules/act/
             ──────────────────────              ────────              ───────────────
 MJCF arm ─► <Config> datagen run ─► assets/datagen/<run>/*.h5
   │                                        │
   │                                        ├─► safety_sweep.py ─► sweep.h5 ─► train_safety_cvae.py
   │                                        │                                     └─► assets/safety/cvae_v3/
   │                                        │                                           │ (frozen)
   │                                        │                                           ▼
   │                                        └─► convert_obstacle_to_act.py ─► act_style_data/<ds>/
   │                                                   (--with_proximity)          │
   │                                                                               ▼
   └──────────────── same env reused for eval ◄─── imitate_episodes.py ─► ckpts/<task>/<run>/
                                  │                                               │
                                  └──────────► eval_act_obstacle.py ◄─────────────┘
                                                        │
                                                        ▼
                                        eval_output/<run>_<cell>/eval_summary.json
                                                        │
                                                        ▼
                                                 compare_pact.py
```

Three things share one dataset: the **Safety-CVAE** (a reflex net trained by distillation), the
**vanilla ACT baseline** (cameras + joints only), and **PACT** (ACT plus a proximity token).

---

## 3. Setup and environment

```bash
conda activate mlspaces
```

All rendering is headless offscreen EGL. Prefix every render/datagen/train/eval command with:

```bash
OMP_NUM_THREADS=2 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
```

Two facts about the environment that are not obvious and matter a lot:

- **`~/.bashrc` exports `MLSPACES_ASSETS_DIR=/home/jaydv/code/prox_learning/assets`.** This repo's
  `assets/` directory *is* the MolmoSpaces asset root. That is why datagen output lands inside
  the repo, and why `assets/{scenes,objects,grasps,test_data,benchmarks}` are symlink farms into
  `~/.cache/molmo-spaces-resources/`.
- **`pyproject.toml` is not installed and is not meant to be.** It declares
  `[tool.setuptools] py-modules = []` — "a collection of scripts + assets, not an importable
  package". The `mlspaces` conda env supplies the dependencies. Scripts import each other by
  doing `sys.path.insert(0, Path(__file__).parent)` and then importing siblings as top-level
  modules, which is what creates `scripts/__pycache__`.

One-time extras the `mlspaces` env is missing for ACT:

```bash
pip install ipython          # ACT + DETR import it
pip install wandb && wandb login   # training logs to wandb by default; --no_wandb opts out
```

`dm_control` is not needed — the ALOHA-only import is optional.

---

## 4. Repo map

```
README.md            this manual
STATUS.md            the same project in plain language — read this first if you are lost
CLAUDE.md            agent working agreement
pyproject.toml       dependency declaration (NOT installed; see §3)

scripts/             29 analysis / training / figure scripts + housekeeping.sh   §5
submodules/act/      ACT fork — trains and evaluates the policies                §6
submodules/molmospaces/  the simulator + demonstration collection                §7
submodules/MolmoBot/ unused; nothing in this project imports it

assets/              MolmoSpaces asset root AND this project's artifacts         §8
  robots/franka_skin/model_hybrid.xml    the 40-sensor arm — the canonical model
  safety/cvae_v3/                        the reflex net weights + sensor order
  datagen/                               collected demonstration datasets
franka_assets/       mesh store; reached only through symlinks — DO NOT DELETE   §8
act_style_data/      converted ACT training sets
eval_output/         rollout videos + eval_summary.json per test
experiments_output/  per-run output folders (sweep.h5, cvae/, demos/, figures/)
diagnostics_output/  committed legacy renders (242 tracked files)
reports/2026-08-14/  the current progress report
paper/               one section draft
synthetic_verify/    proximity ground-truth check (29-sensor era)

train_blur_baseline.sh   trains one vanilla arm per constant blur sigma
eval_blur_baseline.sh    the 9-condition blur test + summary table
blur_eval.log            log from the 2026-08-09/10 run
pointcloud.ipynb         DEAD — reads a deleted dataset, unrelated kernel
```

---

## 5. File-by-file: `scripts/`

All 29 files compile and every sibling import resolves. Status is one of **ACTIVE** (part of a
current recipe), **ARCHIVE** (worked, but its era is over), **DEAD** (broken or unreachable).

### Core pipeline

| file | what it does | entry | status |
|---|---|---|---|
| `safety_sweep.py` | Renders 40 SPAD depths over replayed postures with hazard bars planted next to random sensors, and computes the analytic potential-field retreat labels. Writes the CVAE training set. | `--runs <datagen_dir> [--n 15000] [--out sweep.h5]` | ACTIVE |
| `train_safety_cvae.py` | Trains the Safety-CVAE (40×8×8 depth → 7-DoF retreat). Writes `model.pt`, `meta.json`, 4 diagnostic PNGs. | `--data <sweep.h5> [--out cvae/] [--epochs 60] [--no-wandb]` | ACTIVE |
| `safety_flinch_demo.py` | Bar marches down a forearm sensor's view axis; arm flinches and relaxes home. | `[--ckpt assets/safety/cvae_v3] [--obstacle bar\|sphere] [--clean]` | ACTIVE |
| `safety_react_demo.py` | Reactive avoidance layered on a recorded reach-grasp-lift — deviate and rejoin. | same + `[--mode sweep\|pick]` | ACTIVE |
| `safety_moving_demo.py` | Two bars patrol wrist→shoulder; each link tinted by its own proximity. | same + `[--secs 18]` | ACTIVE |
| `safety_orbit_demo.py` | One bar sweeps a half-circle over the arm's outward face, sliding elbow→wrist. | same + `[--arc 180] [--passes 3]` | ACTIVE |
| `safety_sphere_demo.py` | Flinch demo with a blue sphere — shape/colour-agnosticism proof. | `[--ckpt …] [--radius …]` | ACTIVE but redundant¹ |
| `figures.py` | All 29 paper figures. 6672 lines. | `--list \| --all \| <name> [--outdir DIR]` | ACTIVE |
| `hybrid_viz_lib.py` | Shared MuJoCo/EGL helpers: model build, 8×8 depth render, back-projection, scene primitives, the dark `STYLE` dict. Sets `MUJOCO_GL=egl` on import. | library — the most-imported module here | ACTIVE |
| `foxglove_viz.py` | Exports a datagen `.h5` (or a whole run dir) to one `.mcap`: robot meshes, proximity cloud, RGB video, joints, phase logs. | `--h5 <path_or_dir> [--out viz.mcap]` | ACTIVE |
| `build_hybrid_on_franka_skin.py` | Builds `model_hybrid.xml`, the canonical 40-sensor arm. No argparse — paths hardcoded. | `python scripts/build_hybrid_on_franka_skin.py` | ACTIVE |
| `housekeeping.sh` | Tiered disk cleanup. Dry-run by default. | `--tier1 [--apply]`, see §17 | ACTIVE |

¹ `safety_sphere_demo.py` is 294 lines reproducing `safety_flinch_demo.py --obstacle sphere`, and
it lacks `--clean`. The only thing keeping it alive is that Recipe A's loop names `sphere`.

**Demo import chain** (deliberate, not accidental duplication):
`safety_sweep` ← `safety_flinch_demo` ← `safety_react_demo` ← `safety_moving_demo` ← `safety_orbit_demo`.
Flinch is the base library (posture pick, aperture, render, mosaic, HUD). Keep them together.

### ACT / PACT tooling

| file | what it does | entry | status |
|---|---|---|---|
| `convert_obstacle_to_act.py` | Datagen run → ACT per-episode HDF5. `--with_proximity` adds the 40-sensor group. Prints the `num_episodes`/`episode_len` you paste into `constants.py`. | `--src <run_dir> --dst <out> [--with_proximity] [--image_h 240] [--image_w 320]` | ACTIVE |
| `probe_prox_decodability.py` | Go/no-go gate: is the planner's deflection linearly decodable from the frozen trunk feature? Prints AUC + PASS/FAIL. | `[--act-dir …] [--gate-label deflect\|bar] [--auc-gate 0.8]` | ACTIVE |
| `compare_pact.py` | Small-N statistics: Wilson CI, rate difference, Fisher exact. Reads no files — you type the counts. | `vanilla=11/50,30/50 pact_raw=9/50,20/50` | ACTIVE |
| `train_blur_baseline.sh` | One vanilla arm per constant blur sigma on `obstacle_prox_v2`. | `./train_blur_baseline.sh [2 4 8]` | ACTIVE |
| `eval_blur_baseline.sh` | Evaluates every `*blurC*_v2` ckpt across three cells, prints a summary table. | `./eval_blur_baseline.sh [N=50] [cells…]` | ACTIVE |

### Model build and verification

| file | what it does | status |
|---|---|---|
| `test_and_reconstruct_hybrid.py` | Four-part reconstruction suite. **Its `main()` is archive, but `figures.py` imports `gt_boxes`/`cloud_error`/`cloud_panel` from it at module scope** — do not delete. | ACTIVE as library |
| `verify_hybrid_skin_sensors.py` | Per-sensor QA of all 40 SPADs: self-hit, outward normal, known-distance plate, cloud-vs-geometry error. Re-run after any sensor change. | ARCHIVE |
| `convert_hybrid_skin_urdf.py` | URDF → visual-only MJCF with 40 cameras. Superseded by `build_hybrid_on_franka_skin.py`; its output model has no consumer. | ARCHIVE |
| `build_photoshoot_skin.py` | Cosmetic model with red SPAD dots for paper renders. | ARCHIVE |
| `photoshoot_sweep.py` | Studio renders of `model_photoshoot.xml`: turntable, pose strip, hero shot. | ARCHIVE |

### Dataset analysis (all pre-`figures.py` era)

| file | what it does | status |
|---|---|---|
| `analyze_obstacle_dataset.py` | Six-check validation of `hybrid_obstacle_v1`: bar rate, deflect-vs-free, prox separation, contact safety. | ARCHIVE |
| `analyze_dataset.py` | Generic 6-domain sweep of a pick-and-place run. **29-sensor era** (`LINKS = 2/3/5/6`). | ARCHIVE |
| `dataset_probes.py` | Four advisor-spec probes on the enclosure dataset. **29-sensor era.** Superseded by `probe_prox_decodability.py`. | ARCHIVE |
| `decorr_heatmap.py` | Per-scene correlation heatmap: hidden geometry vs camera-visible params. Hardcoded three-run study. | ARCHIVE |
| `plot_mindist_traces.py` | Min-skin-distance traces per behavior class. **Same three hardcoded runs and same output dir as `decorr_heatmap.py`** — these two should be one script. | ARCHIVE |
| `proximity_necessity.py` | How often vision loses the target while the skin still has signal. Its own docstring records the conclusion that pushed the project from pick-and-place to enclosures. | ARCHIVE |
| `enclosure_report.py` | Date-tagged report bundle for an enclosure run. Shells out to `dataset_probes.py`. | ARCHIVE |
| `cavity_scene.py` | Cabinet/drawer-cavity MJCF generator. Loads the **29-sensor** `model.xml`. Its geometry logic was absorbed into the MolmoSpaces cavity sampler. | ARCHIVE |
| `wandb_upload_dataset.py` | One-time upload of the frozen 2026-06-10 dataset to wandb. | ARCHIVE |
| `assemble_overnight_report.py` | Stitches the 2026-06-11 renders into contact sheets. **DEAD:** line 50 reads `20260610_hybrid_skin_rich/` but the directory is `20260611_…`, and two of its inputs have no generator anywhere in the repo. | DEAD |
| `pointcloud.ipynb` (top level) | **DEAD:** reads `assets/datagen/pick_planner_v1/…` which does not exist, uses the `ilab` kernel, superseded by `foxglove_viz.py`. | DEAD |

### `figures.py` — the 334 KB file

Structurally trivial despite the size: **29 top-level `fig_*()` functions + a `REGISTRY` dict +
`main()`**. Every figure function is entirely self-contained — all its constants, scene geometry
and helpers are local to the function body (200–390 lines each). There is no shared internal
layer, which is the whole reason for the file size.

It was assembled by concatenating ~29 standalone scripts: lines 27–89 hold **24 separate
`from hybrid_viz_lib import …` statements**, `import sys` three times, and `from collections
import defaultdict` twice. Harmless, but that is the fingerprint.

**All 29 registry keys map to real functions, and all 30 output PNGs exist in
`experiments_output/v4/figures/`. Nothing in it is dead.**

CLI: `--list`, `--all`, positional names, `--outdir` (rebinds the module-global `_FIGROOT`, which
every function reads as its `OUT`). Undocumented env override: `PROX_FIG_OUT`. Unknown keys warn
and are skipped rather than erroring. With no arguments at all it falls through to `--list`.

**Naming is inconsistent in 12 of 29 cases** — the registry key, the function name and the output
filename disagree, and the `make_` / `fig_` / `panel_` / `env_` / `proof_` / `viz_` / `render_`
prefixes carry no meaning. You cannot predict the output filename from the CLI key. Examples:

| CLI key | output PNG |
|---|---|
| `proof_whole_arm_clearance` | `use_whole_arm_clearance.png` |
| `make_pipe_tunnel_fig` | `env_pipe_tunnel.png` |
| `viz_peg_forest` | `env_peg_forest.png` |
| `make_acc_range_linearity` | `acc_range_linearity_perlink.png` |
| `fig_hood_narrow` | `hood_narrow.png` (function is `fig_fig_hood_narrow()`) |

Groups: sensor characterisation (`panel_single_sensor_anatomy`, `panel_plane_distance_sweep`,
`panel_plane_tilt_sweep`, `panel_range_accuracy_scatter`, `acc_angular_resolution`,
`make_acc_range_linearity`, `proof_acc_repeat`) · why skin is needed (`panel_coverage_behind`,
`panel_need_vision_vs_skin`, `panel_need_blur_and_dark`) · use cases (`proof_whole_arm_clearance`,
`panel_clearance_controller`) · reconstruction (`panel_known_shapes_cloud`,
`panel_cavity_reconstruction_3d`, `test_reconstruct_fumehood` — the only two-output figure) ·
environments (`make_pipe_tunnel_fig`, `panel_env_cluttered_shelf`, `env_corner_cavity_hero`,
`env_narrow_slot`, `env_overhang_fig`, `viz_peg_forest`) · fume hoods (`fig_hood_narrow`,
`fig_hood_tall`, `fumehood_std_fig`, `fumehood_short_low_sash_fig`,
`fumehood_var_hood_deep_tunnel`) · galleries (`panel_sensor_gallery`, `render_hybrid_skin_rich`,
`render_hybrid_skin_viz`).

### Known duplication in `scripts/`

| group | what actually differs | verdict |
|---|---|---|
| `safety_sphere_demo.py` vs `safety_flinch_demo.py --obstacle sphere` | Nothing meaningful. Sphere-demo uses a dedicated mocap body; flinch converts bar geoms in place and also supports `--clean`. | delete the sphere script |
| `decorr_heatmap.py` + `plot_mindist_traces.py` | Identical hardcoded run dict, identical output dir, both re-implement `decode_scene`/`decode_blob`. Only the plot differs. | merge behind `--plot {decorr,mindist}` |
| `analyze_dataset.py` + `analyze_obstacle_dataset.py` | ~120 lines of copy-pasted HDF5 decode helpers. The checks genuinely differ. | extract the helpers, keep both |
| `build_hybrid_on_franka_skin.py` + `convert_hybrid_skin_urdf.py` | Both parse the same URDF and place 40 cameras. The converter builds visual-only from scratch; the builder grafts onto the proven functional base. | converter is superseded |
| depth-render helpers | `hybrid_viz_lib`, `verify_hybrid_skin_sensors`, `photoshoot_sweep`, `safety_sweep`, `foxglove_viz` each carry their own `build`/`set_pose`/`backproject`. | low priority — refactoring risks the demos |

### Dead references (verified by `stat`, not inferred)

- `pointcloud.ipynb` → `assets/datagen/pick_planner_v1/…` — gone.
- `assemble_overnight_report.py:50` → `20260610_hybrid_skin_rich/` — actual dir is `20260611_…`.
- `assemble_overnight_report.py` → `fov_coverage_map.png`, `use_cloud_accumulation.png` — the PNGs
  exist but **no generator survives anywhere in the repo**; these two figures cannot be rebuilt.
- `foxglove_viz.py:5` docstring → `scripts/foxglove_layout.json` — does not exist (docs only).
- `safety_sweep.py:32` usage example → `assets/datagen/hybrid_pnp5_mass/…` — the data actually
  lives under `assets/prox_learning_data/`.
- `scripts/__pycache__/` holds `.pyc` for 7 deleted scripts (`convert_pla_to_act`,
  `convert_smoke_to_act`, `foxglove_dashboard`, `lock_clutter_bins`,
  `panel_env_cluttered_shelf`, `sensor_usage_timeline`, `visualize_skin_test_data`) — fossils of
  the June cleanup.

---

## 6. File-by-file: `submodules/act` (the policy)

Fork of Tony Zhao's ACT. Upstream ends at `742c753` (2024-01-28); every commit after it belongs to
this project. The fork adds proximity fusion (PACT), a real in-env evaluator, and the blur /
modality-dropout curricula.

> **The parent repo's submodule pointer is stale:** it records `ec44793`, the checkout is at
> `830fc2b`. That is the `M submodules/act` in `git status`. Fix with `git add submodules/act`.

### Fork-touched files

| file | what it is | status |
|---|---|---|
| `imitate_episodes.py` | **Training entry point.** All ACT + PACT training. | modified (+360) |
| `eval_act_obstacle.py` | **Canonical in-env evaluator.** | fork-new |
| `prox_cvae.py` | Frozen Safety-CVAE → skin feature extractor. | fork-new |
| `constants.py` | `TASK_CONFIGS` lookup table. | modified (+102) |
| `utils.py` | `EpisodicDataset`, `get_norm_stats`, `load_data`. Pads to `num_queries`, not `episode_len`, so variable-length episodes work. | modified |
| `policy.py` | `ACTPolicy` wrapper; gains `proximity_positions=` and `image_dropped=`. | modified (+31) |
| `detr/main.py` | DETR argparse. **Must re-declare every fork flag as a no-op**, because `build_ACT_model_and_optimizer` re-parses all of `sys.argv`. | modified (+50) |
| `detr/models/detr_vae.py` | The CVAE policy; `input_proj_proximity`, sized `additional_pos_embed`. | modified (+135) |
| `detr/models/transformer.py` | Where prox tokens are concatenated into encoder memory. | modified (+17) |
| `detr/models/backbone.py`, `position_encoding.py` | Cosmetic: relative imports, optional IPython. | modified |
| `eval_train_set.py` | Open-loop train-set L1 — separates underfitting from covariate shift. | fork-new, works |
| `attn_heatmap.py` | Decoder cross-attention overlays on failure rollouts. | fork-new, works |

### Upstream ALOHA files, unused here

`sim_env.py`, `ee_sim_env.py`, `scripted_policy.py`, `record_sim_episodes.py`,
`visualize_episodes.py` (only `save_videos` is imported), `detr/models/__init__.py`,
`detr/setup.py`, `detr/util/*`.

### The eval scripts — only one is current

| file | verdict |
|---|---|
| **`eval_act_obstacle.py`** | **CANONICAL.** The only one with PACT support, collision metrics, `--eval_cell`, `--eval_blur_sigma`, `--end_on_collision`, the `viz_sensor_rgb=False` OOM guard, `eval_summary.json` output, and the fixed `--temp_agg_off`. |
| `eval_act_house1.py` | Historic (mug task). Still imported by `eval_act_house1_dup250.py`. |
| `eval_act_house10_cup.py` | Historic copy of house1, different house/object. No `TASK_CONFIGS` entry at all. |
| `eval_act_house1_dup250.py` | Dead one-off; its dataset is gone. |
| `eval_act_mug_random.py` | Historic. `samples_per_house=1`, so one rollout per process — you launch N and aggregate via `--wandb_group`. |
| `eval_act_with_prox.py` | **BROKEN.** Imports `pla.prox_residual_head`; no `pla/` package exists. This was the *old* design (a residual head added to the action chunk), superseded by in-transformer conditioning. |

### How proximity actually enters the network

1. `(B, 40, 8, 8)` raw depths **in metres** come out of the dataloader.
2. `ProxCVAEEncoder` featurises to closeness `c = clip(1 − d/0.5, 0, 1)`, with `c[d < 0.005] = 0`.
3. One of three taps:

| `--prox_feature` | computation | shape | dim |
|---|---|---|---|
| `raw` | per-sensor peak closeness — **bypasses the CVAE entirely** | `(B,1,40)` | 40 |
| `trunk` | CVAE decoder trunk hidden activation at `z=0` | `(B,1,256)` | 256 |
| `delta` | CVAE output × `label_scale` — the literal 7-DoF joint retreat | `(B,1,7)` | 7 |

4. `input_proj_proximity = Linear(feat_dim, K*hidden_dim)` reshaped to K tokens
   (`--prox_tokens_per_sensor`, default 8).
5. `transformer.py` **concatenates** them into encoder memory:
   `[latent, proprio, prox_1..prox_K, image_tokens…]`. With two 240×320 cameras the image side is
   160 tokens, so proximity is 8 of 170.
6. Position info is a **learned** `additional_pos_embed = Embedding(2 + N*K, hidden)`, added to
   queries and keys only (standard DETR), never to values.

Only `input_proj_proximity` and the extra positional-embedding rows train. The CVAE is frozen,
`eval()`, `requires_grad_(False)`, and lives *outside* the policy checkpoint. With
`n_proximity_sensors=0` the model is bit-identical to vanilla ACT. Training writes
`prox_config.json` into the checkpoint dir and `eval_act_obstacle.py` auto-detects it, so
train/eval parity needs no flags.

### `TASK_CONFIGS` — what actually resolves

| task name | dataset dir | eps | len | on disk? |
|---|---|---|---|---|
| `obstacle_baseline` | `act_style_data/obstacle_v1` | 100 | 169 | **yes** |
| `obstacle_pact` | `act_style_data/obstacle_prox_v1` | 100 | 168 | **yes** |
| `obstacle_pact_v2` | `act_style_data/obstacle_prox_v2` | 105 | 185 | **yes** |
| `test`, `proximity_learning`, `pla_house1_mug`, `pla_smoke`, `pla_house1_mug_random`, `pla_house3_mug_random`, `pla_houses_1_3_mug_random` | old `pla_*` paths | — | — | **no — all dead** |
| `SIM_TASK_CONFIGS` (4 ALOHA tasks) | `DATA_DIR` | — | — | **no — dead** |

`episode_len` is only read by the dead `--eval` path; the dataloader pads to `chunk_size`. The
169-vs-168 discrepancy for the same source run is a harmless off-by-one.

### Training flags worth knowing

| flag | default | what it does |
|---|---|---|
| `--use_proximity` | off | turns on the PACT path |
| `--prox_feature` | `trunk` | `raw` / `trunk` / `delta` (see above) |
| `--prox_encoder_ckpt` | `assets/safety/cvae_v3` | which frozen CVAE |
| `--prox_tokens_per_sensor` | 8 | K |
| `--blur_sigma0` | 0 | Gaussian blur strength on camera frames at training time |
| `--blur_mode` | `curriculum` | `curriculum` anneals `σ·(1 − n/N)`; `constant` holds σ all run |
| `--blur_curriculum_steps` | half of total | N |
| `--image_dropout_p` | 0 | per-sample hard vision dropout |
| `--prox_dropout_p` | 0 | per-sample skin dropout, sampled disjointly from vision |
| `--image_dropout_mode` | `all` | `all` cameras or a `single` random one |
| `--no_zero_latent_on_drop` | off | ablation: keep the CVAE style latent on dropped samples |
| `--no_wandb` | off | wandb is ON by default |

Blur and dropout are **training-only**. Validation, best-checkpoint selection and eval always see
clean sharp frames.

### Eval flags worth knowing

| flag | default | what it does |
|---|---|---|
| `--ckpt_dir` | baseline path | the dated run folder holding `policy_best.ckpt` + `dataset_stats.pkl` |
| `--num_rollouts` | 25 | episodes in this process |
| `--eval_cell` | none | `visible` / `invisible` / `free` — pins every rollout to one obstacle cell |
| `--temp_agg_off` | off | open-loop chunking (see the trap in §15) |
| `--temp_agg_m` | 0.01 | temporal-aggregation weight when it is on |
| `--eval_blur_sigma` | 0 | blurs cameras at **inference**; skin and qpos untouched |
| `--end_on_collision` | off | strict safety: any contact is a failure and ends the episode |
| `--live` | off | opens a MuJoCo viewer (desktop only, forces single-process) |
| `--house_ind` | 1 | ProcTHOR house; 1 (≡ 1 mod 24) is the red cup the data used |
| `--task_horizon` | 200 | max policy steps |

### Breakage and gotchas in the fork

1. `eval_act_with_prox.py` — import error on startup (`pla/` gone).
2. 8 of 10 `TASK_CONFIGS` dataset dirs do not exist.
3. `constants.py` cites `scripts/build_combined_h1_h3.py` — that script is gone.
4. `imitate_episodes.py --eval` is dead for every fork task: it takes the real-robot branch and
   ImportErrors, and it never passes `proximity_positions`. **Always use `eval_act_obstacle.py`.**
5. `detr/main.py` comments reference a `pact.act_prox` package that does not exist;
   `--prox_mapping_json` is a pure no-op consumed by nobody.
6. `prox_cvae.py`'s docstring says "two feature taps" — there are three; `raw` was added later.
7. Both cameras share **one** ResNet18 backbone (`build()` appends one, `forward` uses
   `backbones[0]`, marked `# HARDCODED`). Upstream behaviour, not a fork bug.
8. Upstream per-epoch train-summary slice bug at `imitate_episodes.py:626`.

---

## 7. File-by-file: `submodules/molmospaces` (the simulator)

A large library. You need about four of its sixteen packages.

| package | what you need from it |
|---|---|
| **`data_generation/`** | `main.py` (entry), `pipeline.py` (workers, rollout loop, saving), `config_registry.py`, **`config/object_manipulation_datagen_configs.py` — every config in this project**, `custom_scenes/` (the MJCF scenes: `fumehood.xml`, `enclosure_param.xml`, `panel_slalom.xml`, `cubby_overreach.xml`) |
| **`tasks/`** | `enclosure_reach.py` (samplers + experts + the obstacle policy), `house_embed.py`, `cavity_pick_task_sampler.py`, `pick_task.py`, `task.py` (step loop / sub-step recorder). The other ~27 files are unrelated tasks. |
| **`env/`** | `env.py` (`CPUMujocoEnv`, owns the proximity renderers), `sensors_cameras.py`, `sensors.py` |
| **`configs/`** | `abstract_exp_config.py` (global flags incl. `viz_sensor_rgb`), `camera_configs.py` (skin layouts), `robot_configs.py`, `task_sampler_configs.py`, `base_pick_config.py`, `policy_configs.py` |

Ignore: `molmo_spaces_isaac/`, `molmo_spaces_maniskill/`, `mlspaces_tests/`, `docs/`, `bin/`,
and within `molmo_spaces/`: `planner/`, `robots/`, `renderer/`, `controllers/`, `kinematics/`,
`grasp_generation/`, `housegen/`, `resources/`.

### Datagen configs relevant to this project

All live in `config/object_manipulation_datagen_configs.py`. **Every registered config is
concrete and runnable** — the "base" ones are simply configs that others also subclass.

**The live line — 40-sensor hybrid skin:**

| config | line | inherits | what it collects | output dir |
|---|---|---|---|---|
| `FrankaSkinHybridFumehoodSmokeConfig` | 2088 | `FrankaSkinFumehoodSmokeConfig` | fumehood reach; **switches to the 40-sensor hybrid skin**; turns on `viz_sensor_rgb` @256×256 | `datagen/hybrid_fumehood_smoke` |
| `FrankaSkinHybridPnP5Config` | 2104 | ↑ | fumehood pick with grasp-file grasps | `datagen/hybrid_pnp5` |
| `FrankaSkinHybridPnP5MassConfig` | 2153 | ↑ | 24 objects × 10 — fed `sweep_v1.h5` | `datagen/hybrid_pnp5_mass` |
| `FrankaSkinHybridObstacleCheckConfig` | 2190 | `HybridPnP5` | preflight, bar forced on | `datagen/hybrid_obstacle_check` |
| **`FrankaSkinHybridObstacleConfig`** | 2222 | ↑ | **the main ACT dataset.** 8 house indices × 25 | `datagen/hybrid_obstacle_v1` |
| `FrankaSkinHybridInvisObstacleCheckConfig` | 2262 | `HybridObstacleCheck` | preflight, bar present and invisible | `datagen/hybrid_invis_obstacle_check` |
| **`FrankaSkinHybridInvisObstacleConfig`** | 2293 | ↑ | **the v2 invisible-bar dataset** | `datagen/hybrid_invis_obstacle_v1` |

Full chain: `HybridObstacle ← HybridObstacleCheck ← HybridPnP5 ← HybridFumehoodSmoke ←
FumehoodSmoke ← EnclosureSmoke ← CabinetCavitySmoke ← PickBaseConfig`.

**The 29-sensor task-shape line** (`FrankaSkinEnclosureSmokeConfig` :1886,
`FrankaSkinEnclosureGenConfig` :1916, `FrankaSkinFumehoodSmokeConfig` :1944,
`FrankaSkinPanelSlalomSmokeConfig` :1969, `FrankaSkinCubbySmokeConfig` :1993, plus the three
`House*` variants at :2040/:2054/:2068 that embed the same task inside ProcTHOR rooms) produced the
June diagnostic datasets. Historic.

**Older lines still registered but with no data and no references:** the cavity/shelf/clutter/
pillar/real-table family (:1458–:1828) and the iTHOR pick-and-place family (:238–:743). Only
`FrankaSkinProxNecessityPilotConfig` from that era still has data on disk.

### Task samplers — the knobs you override

Chain: `PickTaskSampler → CavityPickTaskSampler → EnclosureReachSampler → FumehoodSampler →
BigFumehoodPickSampler → ObstacleFumehoodPickSampler → InvisibleObstacleFumehoodPickSampler`.
All in `tasks/enclosure_reach.py` unless noted.

| sampler | line | key class attributes (defaults) |
|---|---|---|
| `EnclosureReachSampler` | 113 | `MIXTURE=(("free",.28),("hidden",.33),("visible",.28),("abort",.11))`, `POOL_SIZE=24` |
| `FumehoodSampler` | 694 | `MIXTURE=(("free",.40),("hidden",.30),("visible",.15),("abort",.15))` |
| `BigFumehoodPickSampler` | 1025 | `MIXTURE=(("free",1.0),)`, `BASE_FWD=0.08`, `PICK_CATEGORIES=(mug,cup,apple,…)` |
| **`ObstacleFumehoodPickSampler`** | 1109 | **`OBSTACLE_P=0.75`**, `BAR_FACE_Y=(0.14,0.24)`, `BAR_X_FRAC=(0.20,0.55)`, `OBJ_GAP=(0.12,0.20)` |
| `ObstacleFumehoodPickCheckSampler` | 1169 | `OBSTACLE_P=1.0` |
| **`InvisibleObstacleFumehoodPickSampler`** | 1175 | **`INVIS_P=0.5`** (inherits `OBSTACLE_P=0.75`); hides bars by moving the geom to **group 4**; decouples object placement from bar presence (the v1 leak fix) |
| `InvisibleObstacleFumehoodPickCheckSampler` | 1237 | `OBSTACLE_P=1.0`, `INVIS_P=1.0` — this is what `--eval_cell` drives |

The matching expert is `ObstacleAwarePickPlannerPolicy` (`enclosure_reach.py:1249`):
`GRIP_HALF=0.10`, `SAFE_GAP=0.08`, `PASS_SPEED=0.05`. It reads `scene_params["protr_center"]` and
`["protr_half"]` — **never pixels** — bows the approach with two bracketing waypoints, and stamps
`behavior_class = "deflect"`; otherwise `"free"`.

### The sensor pipeline

1. **Declaration.** Each sensor is an MJCF camera with `is_proximity_sensor=True`
   (`configs/camera_configs.py:28`), `fov=45.0`, `record_depth=True`.
   - 29-sensor skin: `_SKIN_SENSOR_LINK_COUNTS = {2:7, 3:8, 5:6, 6:8}` (`camera_configs.py:389`).
   - **40-sensor hybrid:** `_HYBRID_SKIN_SENSOR_NAMES` (`:417`) = link1×7, link2×7, link3×5,
     link4×5, link5_front×4, link5_back×6, link6×6.
2. **Sub-step recording.** `_n_sim_steps_per_proximity = round(proximity_sensor_period_ms /
   sim_dt_ms)` (`tasks/task.py:56`). With `policy_dt_ms=66`, `sim_dt_ms=2`,
   `proximity_sensor_period_ms=16.6667` → 33 sim steps per policy step ÷ 8 = **4 sub-steps**.
   That is the `4` in the stored shape.
3. **Render.** `CPUMujocoEnv.record_proximity_depths` (`env/env.py:376`) uses a **dedicated 8×8
   renderer**, deliberately bypassing the 624×352 global one (wrong FOV on a non-square aspect).
   Scene option: `geomgroup[2]=0` (hide the cosmetic skin so sensors do not see themselves) and
   **`geomgroup[4]=1`** — which is exactly the mechanism the invisible-bar sampler exploits.
4. **Packaging.** `ProximityDepthBufferSensor` stacks to `(max_substeps, 8, 8)` float32 metres,
   left-padding by repeating the earliest frame rather than zeros (a zero would read as contact
   at 0 m).
5. **Written to** `obs/proximity/<sensor_name>` with shape `(T, 4, 8, 8)`.

### `viz_sensor_rgb` — the memory bomb

Defined at `configs/abstract_exp_config.py:57` (default `False`), turned **on** at
`object_manipulation_datagen_configs.py:2094` with `viz_sensor_resolution=(256,256)` — and
therefore inherited by every config in the hybrid/obstacle chain.

It adds 40 extra 256×256 RGB renders plus 40 depth reads **per policy step**, purely to produce
cosmetic skin mosaic videos. **It adds nothing to the HDF5** (verified: identical key sets with it
on and off). The pipeline retains each episode's full observation history in RAM until the whole
house finishes saving, so this costs **~3 GB/episode instead of ~0.5 GB** — enough to OOM-kill a
30-rollout eval on a 62 GB box. `eval_act_obstacle.py` forces it off; datagen does not, which is
why `num_workers <= 2` is the standing advice.

### Entry point

`main.py` takes **exactly one positional argument**, the config name — either `ConfigName` (which
imports every config module so the decorators fire) or `module.path:ClassName` (faster, avoids
unrelated import side-effects). **There is no way to override a config field from the command
line today**; edit the class or subclass it.

Output lands at `<output_dir>/<ConfigName>/<YYYYmmdd_HHMMSS>/`, plus
`experiment_config_<ts>.pkl` and `running_log.log`. Parallelism is `exp_config.num_workers`, and
work is distributed **at house granularity** — one house start-to-finish per worker. That is why
the obstacle configs list 8 wrap-around house indices (`1, 25, 49, …`, all ≡ 1 mod 24) for what is
really a single task: it is the only way to use more than one worker.

`setup_house_dirs` **skips a house whose h5 already exists** — that is the resume mechanism, and
also why re-running into an existing timestamped dir silently does nothing.

---

## 8. File-by-file: models and assets

### Robot models — verified by compiling each with MuJoCo, not by grep

| file | sensors | what makes it different | verdict |
|---|---|---|---|
| **`assets/robots/franka_skin/model_hybrid.xml`** | **40** (nsite 42, ncam 42, ngeom 71, nu 8) | The gentact hybrid dermis on links 1–6 with the link5 front/back split. Inlined Robotiq 2F-85. | **LIVE — the canonical model** |
| `assets/robots/franka_skin/model.xml` | 29 | The older skin (links 2/3/5/6). Attaches Robotiq by reference. | LIVE (legacy path — all non-hybrid configs) |
| `assets/robots/franka_skin/model_photoshoot.xml` | 0 cameras, 40 red decal geoms | Visual-only paper render. | LIVE, single consumer |
| `assets/robots/fr3_hybrid_skin/model.xml` | 40 cams, **nu=0**, no gripper | URDF→MJCF intermediate with hardcoded absolute `meshdir`. Its `meshes/skin/` is live build input; the model itself is loaded by nothing. | build intermediate |
| `assets/urdf/fr3_hybrid_skin.urdf` | — | Source of truth for the 40 sensor poses. | LIVE (build input) |
| `assets/robots/franka_skin/model.xml.bak_before_orientation_fix` | 29 | Byte-identical to `model.xml` except one attribute on all 29 cameras: `quat="0 1 0 0"` → `quat="0 0 1 0"`. That is the entire "orientation fix". | DEAD (keep as provenance) |
| `assets/mjcf/fr3_skin.xml`, `fr3_robotiq.xml` | 29 | **Fail to compile** — meshes point at a deleted `.mesh_cache`. | DEAD / broken |
| `assets/urdf/fr3_full_skin.urdf`, `fr3_full_skin_fixed.urdf`, `robotiq.urdf` | — | 29-sensor era. Zero references. | DEAD |
| `franka_assets/fr3_skin/model.xml` | 29 | Pre-orientation-fix copy with local mesh paths. | DEAD |
| `franka_assets/fr3_skin/scene.xml` | — | **Fails to compile** (self-referential relative path). | DEAD / broken |
| **`franka_assets/fr3_skin/{assets,skin_meshes,robotiq_2f85_v4}/`** | — | The actual 117 MB mesh store. | **LIVE — see warning below** |

> **`franka_assets/` has zero textual references in the entire repo.** It is reached only through
> symlinks: `assets/robots/franka_skin/{assets,skin_meshes,robotiq_2f85_v4}` →
> `../../../franka_assets/fr3_skin/*`. Any grep-based cleanup will conclude it is dead. Deleting it
> breaks the 40-sensor model.

**Build chain:** `assets/urdf/fr3_hybrid_skin.urdf` → `convert_hybrid_skin_urdf.py` →
`assets/robots/fr3_hybrid_skin/` (dermis STLs) → `build_hybrid_on_franka_skin.py` (rewrites
`franka_skin/model.xml`, deleting the four old `link*_skin` bodies) → **`model_hybrid.xml`** →
`build_photoshoot_skin.py` → `model_photoshoot.xml`.

*Open question:* `build_photoshoot_skin.py`'s docstring claims dense farthest-point tiling
(`TARGET_SPACING=0.02`, `MAX_PER_MESH=90` → hundreds of dots), but the committed
`model_photoshoot.xml` has exactly 40 and is smaller than `model_hybrid.xml`. Either the file
predates the dense version or the sampler silently fell back. Needs a re-run to settle.

### The Safety-CVAE weights

All three `model.pt` are 11,598,939 bytes with distinct md5s — same architecture
(`n_in=2560`, `n_out=7`, `z_dim=8`, `d_max_input=0.5`), different training runs.

| version | trained on | `label_scale` | best val MSE | close-cos | far-quiet |
|---|---|---|---|---|---|
| `cvae_v1` | `sweep_v1.h5` | 16.7436 | 0.011126 | 0.8912 | 0.03258 |
| `cvae_v2` | `sweep_v2.h5` | 14.6776 | 0.014584 | 0.8654 | 0.03603 |
| **`cvae_v3`** | `sweep_v3.h5` | **11.3593** | **0.009069** | **0.9255** | 0.03101 |

`cvae_v3/config.json`: `epochs=60, bs=512, lr=1e-3, beta=0.01, z_dim=8, d_max=0.5, seed=0,
n_train=13500, n_val=1500, active_latent_dims=1`.

**`cvae_v3` is the universal default — zero code paths default to v1 or v2** (they appear only in
stale docstring examples). It is the default in `prox_cvae.py:45`,
`convert_obstacle_to_act.py:64`, `probe_prox_decodability.py:100`, all five demos,
`imitate_episodes.py:708`, and `eval_act_obstacle.py:592`.

> **`assets/safety/cvae_v3/meta.json["sensors"]` is the authoritative 40-sensor stacking order for
> the entire ACT/PACT pipeline.** It is NOT the env's `_HYBRID_SKIN_SENSOR_NAMES` order —
> `link5_back` precedes `link5_front`. Both `stack_obs_proximity()` and
> `convert_obstacle_to_act.py` read meta.json. Never hand-roll the order.

Sweep provenance, read from the h5 attrs (all 15,000 samples, `d_act=0.18`, `mount_z=0.35`):

| sweep | source datagen run(s) |
|---|---|
| `sweep_v1.h5` | `prox_learning_data/FrankaSkinHybridPnP5MassConfig/20260612_023111` |
| `sweep_v2.h5` | `hybrid_obstacle_v1/…/20260612_183855` + `…/20260612_162142` + the PnP5Mass run |
| `sweep_v3.h5` | `hybrid_obstacle_v1/FrankaSkinHybridObstacleConfig/20260612_183855` only |

Demo artifacts in `assets/safety/`: each demo writes `<out>.mcap` plus a matching `.mp4`.
`flinch_demo_v1/_v2.mcap` are the same demo run against the v1/v2 weights. The four
`eval_*_sphere.*` files are the flinch/react/moving/orbit demos run with `--obstacle sphere`
(inferred — no script emits that prefix, but exactly those four declare the flag, which is also
why there is no `eval_sphere_sphere`).

### Datasets on disk

| dir | size | what | verdict |
|---|---|---|---|
| `assets/datagen/hybrid_obstacle_v1` | 1.19 GB | **The hot dataset.** 125 trajectories. Referenced by all five demos, `safety_sweep.py`, `convert_obstacle_to_act.py`, `probe_prox_decodability.py`, `constants.py`, and every README recipe. | **IN USE** |
| `assets/datagen/hybrid_invis_obstacle_v1` | 0.99 GB | The invisible-bar v2 collection (125 collected, 105 usable). | **IN USE** |
| `assets/prox_learning_data/FrankaSkinHybridPnP5MassConfig` | 1.26 GB | Source of `sweep_v1.h5` and part of `sweep_v2.h5` — both superseded. | historic |
| `assets/datagen/{cubby,panel_slalom,fumehood}_smoke` | 3.24 GB | June diagnostic three-scene study. | orphaned |
| `assets/datagen/{enclosure,house_fumehood,hybrid_fumehood,house_panel}_smoke` | 1.66 GB | Same era. | orphaned |
| `assets/datagen/enclosure_v1` | 0.86 MB | **Contains zero `.h5`** — only `.pkl`/`.log`. The 2160-episode run never produced trajectories. | failed collection |
| `assets/datagen/hybrid_{obstacle,invis_obstacle}_check` | 0.04 GB | Two-episode preflights. | orphaned |
| `assets/prox_learning_data/FrankaSkinProxNecessityPilotConfig` | 0.63 GB | Source for the `paper/` draft — whose analysis script `pla/audit_proximity.py` is deleted. | orphaned |
| `assets/prox_learning_data/.git` | **2.60 GB** | This directory is a clone of the HuggingFace dataset repo `git@hf.co:datasets/jdvakil/prox_learning_data`. Its LFS object store duplicates its own working tree. | re-cloneable |
| `assets/.lmdb` | **1.3 GB** | MolmoSpaces LMDB caches (objathor metadata 1.1 GB, scenes 215 MB). | rebuilt on demand |

### Everything else

| path | what it is |
|---|---|
| `assets/{scenes,objects,grasps,test_data,benchmarks}` | MolmoSpaces asset farm — real dirs or symlinks into `~/.cache/molmo-spaces-resources/`. **Live, keep.** |
| `assets/benchmark` (singular) | A different thing entirely from `benchmarks`: 57 MB of local datagen output from June. Orphaned. |
| `assets/eval_subsets` | Two hand-cut benchmark JSON subsets. Zero references. |
| `assets/mjthor_data_type_to_source_to_versions.json` | Asset-version manifest (live). The `.bak` beside it is byte-identical — delete. |
| `assets/README.md` | **Entirely stale** — describes a flow through `pla.sim.build_mjcf`, `pla.viz.sensor_overlay`, `scripts/verify_skin.py`, all three deleted. |
| `synthetic_verify/` | Proximity ground-truth check: ~40 mm mean abs error, ~−12 mm bias. Output dir of a molmospaces script. **29-sensor era**, not hybrid. |
| `paper/section3_proximity_signal_draft.md` | CoRL section draft. Cites `pla/audit_proximity.py`, which is deleted — **unreproducible as written**. |
| `.claude/` | Empty directory. |
| `.vscode/settings.json` | Two lines; the dir is gitignored anyway. |

---

## 9. Data formats

### Datagen HDF5 (what MolmoSpaces writes)

One file per house per batch: `house_<id>/trajectories_batch_1_of_1.h5`, top-level groups
`traj_0 … traj_{N-1}`. `T` = episode length, `S` = 4 sub-steps.

```
traj_<i>/
  obs/
    agent/qpos, agent/qvel                    (T, 2000)     uint8   ← zero-padded JSON
    extra/obj_start|obj_end|tcp_pose|
          grasp_pose|robot_base_pose          (T, 7)        float32  xyz + wxyz quat
    extra/grasp_state_pickup_obj              (T, 2000)     uint8   ← JSON
    extra/task_info                           (T, 4000)     uint8   ← JSON
    extra/policy_phase                        (T,)          int64    legend in obs_scene
    extra/policy_num_retries                  (T,)          int64
    extra/object_image_points/…/points        (T, 10, 2)    float32  84 subgroups
    proximity/<sensor_name>                   (T, 4, 8, 8)  float32  × 40  ← THE SKIN
    sensor_param/<cam>/intrinsic_cv           (T, 3, 3)     float64  × 42
    sensor_param/<cam>/extrinsic_cv           (T, 3, 4)     float64  × 42
    sensor_param/<cam>/cam2world_gl           (T, 4, 4)     float64  × 42
  actions/commanded_action|ee_pose|ee_twist|
          joint_pos|joint_pos_rel             (T, 2000)     uint8   ← JSON
  env_states/articulations/panda              (T, 31)       float32
  obs_scene                                   scalar        bytes   ← JSON, ~20 KB
  success|fail|terminated|truncated           (T,)          bool
  rewards                                     (T,)          float32
```

Three things that will bite you:

- **The `uint8 (T, 2000)` datasets are zero-padded UTF-8 JSON, not arrays.** Decode with
  `bytes(row).rstrip(b"\x00")` then `json.loads`.
- **There is no RGB or camera depth in the HDF5** — deliberate. Images live only as sibling MP4s
  in the same house dir.
- **`obs_scene` is where all the labels live**: `scene_params` (the sampler's θ — including
  `protrusion_present`, `bar_face_y`, `protr_center`, `protr_half`, `cam_visible`, and
  `bar_invisible` on invis runs), `behavior_class` ∈ `{free, deflect, abort}`, and
  `collision_metrics`. **`scene_params["cell"]` is NOT a usable label on obstacle runs** — the
  sampler sets it to `"bar"` to skip a raycast rejection loop. Use `protrusion_present` /
  `bar_invisible` / `behavior_class` instead.

### ACT-style HDF5 (what `convert_obstacle_to_act.py` writes)

`<dst>/episode_<g>.hdf5`, `g` contiguous across houses.

| key | shape | note |
|---|---|---|
| attr `sim` | bool | |
| `/action` | `(T, 8)` | arm 7 + gripper command (FR3 hand actuator at {0, 255}) |
| `/observations/qpos` | `(T, 9)` | arm 7 + 2 fingers |
| `/observations/qvel` | `(T, 9)` | read but unused |
| `/observations/images/exo_camera_1` | `(T, 240, 320, 3)` uint8 | |
| `/observations/images/wrist_camera` | `(T, 240, 320, 3)` uint8 | |
| `/observations/proximity` | `(T, 40, 8, 8)` float32 | only with `--with_proximity`; **raw metres**, un-normalised |

Sensors are stacked in `cvae_v3/meta.json` order. The dataloader reads a single random frame per
sample, not the whole episode.

---

## 10. Method & math — the Safety-CVAE

A conditional VAE mapping the **40×8×8 skin depth image → a 7-DoF joint retreat `dq`**. It is
trained by distillation from an analytic potential-field teacher, so it outputs the *avoidance
motion directly* — it is not a collision classifier. At inference `z = 0`, giving a deterministic
retreat. `SafetyHead.load()` in `scripts/train_safety_cvae.py` is the wrapper every demo imports.

Constants used throughout: $D_{\max}=0.5\,\text{m}$ (input normalization),
$D_{\text{act}}=0.18\,\text{m}$ (teacher activation range), close $<0.12\,\text{m}$,
far $>0.25\,\text{m}$.

### 1. Skin sensor → network input

40 SPAD-style sensors, each rendering an $8\times8$ planar-z depth patch in metres → raw input is
$40\times8\times8 = 2560$ values. Pinhole intrinsics with focal length $f = 4/\tan(\text{FOVY}/2)$,
$\text{FOVY}=45^\circ$, principal point $c_x=c_y=3.5$.

Depths map to a **closeness** feature so "far / no return" is exactly zero (the head stays silent
when nothing is near):

$$c = \mathrm{clip}\!\left(1 - \frac{d}{D_{\max}},\; 0,\; 1\right), \qquad D_{\max}=0.5\,\text{m}$$

Pixels with $d<5\,\text{mm}$ (dead/invalid) are forced to $0$. Flattening gives
$x \in [0,1]^{2560}$.

### 2. Analytic potential-field teacher (the labels)

For every sensor whose nearest pixel sees an **environment** return at range $r < D_{\text{act}}$:
back-project that pixel to a world hit-point $p_o$; let $p$ be the sensor origin.

$$r = \lVert p - p_o\rVert, \qquad \hat{u} = \frac{p - p_o}{r}, \qquad w(r) = \frac{1}{r} - \frac{1}{D_{\text{act}}}$$

Each push is mapped from Cartesian to joint space through the sensor body's translational
Jacobian $J_p$, and summed over all 40 sensors. The 7 arm-DoF entries are the label:

$$dq \;=\; \sum_{s\,:\,r_s < D_{\text{act}}} J_{p,s}^{\top}\, \hat{u}_s\, \left(\frac{1}{r_s} - \frac{1}{D_{\text{act}}}\right)$$

Self-hits (returns on the robot's own arm) are detected against the analytic scene boxes and
**excluded from the label**, but kept in the input — the trained head must cope with them at
runtime.

### 3. Label scaling

Labels are normalized to unit RMS over the close subset $C$ (min depth $<0.12\,\text{m}$); the
scale $\sigma$ is stored in `meta.json` and folded back in at inference:

$$\sigma = \sqrt{\frac{1}{|C|}\sum_{i\in C}\lVert dq_i\rVert^2}, \qquad \widetilde{dq} = dq/\sigma$$

### 4. Conditional VAE (distillation)

A small MLP CVAE ($512\to256$, SiLU; latent $z\in\mathbb{R}^{8}$). At train time the encoder sees
both the skin input $x$ and the target $\widetilde{dq}$; the decoder reconstructs $\widetilde{dq}$
from $x$ and a latent $z$:

$$q(z\mid x, \widetilde{dq}) = \mathcal{N}\!\big(\mu,\, \mathrm{diag}\,e^{\log\sigma^2}\big), \qquad z = \mu + \epsilon\odot e^{\tfrac12\log\sigma^2},\ \ \epsilon\sim\mathcal{N}(0,I)$$

$$\widehat{dq} = \mathrm{Dec}([x, z]), \qquad \mathcal{L} = \underbrace{\big\lVert \widehat{dq} - \widetilde{dq}\big\rVert^2}_{\text{recon MSE}} + \beta\, D_{\mathrm{KL}}\!\big(q(z\mid x,\widetilde{dq})\,\Vert\,\mathcal{N}(0,I)\big)$$

with the closed-form Gaussian KL and a 10-epoch $\beta$ warm-up ($\beta_{\max}=10^{-2}$):

$$D_{\mathrm{KL}} = -\tfrac12\sum_{j}\big(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2\big), \qquad \beta = \beta_{\max}\cdot\min\!\Big(1,\ \tfrac{\text{ep}+1}{10}\Big)$$

The latent only shapes training — it lets the head represent the small ambiguity in *which way to
dodge*. Note `cvae_v3` ended with **`active_latent_dims=1`**: the latent is nearly collapsed, which
is consistent with the retreat direction being almost fully determined by the skin.

### 5. Inference (deterministic)

The latent is pinned to the prior mean $z=0$, so the head is a pure, deterministic
$\text{skin}\to dq$ map — **no scene geometry or object poses needed at runtime**:

$$dq = \sigma \cdot \mathrm{Dec}\big([\,x,\; \mathbf{0}\,]\big)$$

### 6. Metrics

- **recon MSE** — validation split, scaled label space. *ref ≈ 0.009.*
- **close direction cosine** — $\cos(\widehat{dq}, dq)$ on close samples: is it pushing the *right
  way*? This is the trustworthy signal. *ref ≈ 0.93.*
- **far-quiet RMS** — $\lVert\widehat{dq}\rVert$ on far samples: is it silent when clear?

---

## 11. Recipe A — Safety-CVAE and the demos

Run outputs go to `experiments_output/<run>/`, one consolidated folder per run. With no
`--out`/`--outdir` everything defaults to `experiments_output/default/`. The committed canonical
weights stay at `assets/safety/cvae_v3`.

```bash
export EGL="OMP_NUM_THREADS=2 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl"
RUN=experiments_output/v5
DS=assets/datagen/hybrid_obstacle_v1/FrankaSkinHybridObstacleConfig/20260612_183855

# 1. Build the near-contact training set (renders 40 SPAD depths + potential-field labels)
env $EGL python scripts/safety_sweep.py --runs $DS --n 15000 --out $RUN/sweep.h5

# 2. Train the Safety-CVAE (wandb + 4 diagnostic PNGs next to the weights)
env $EGL python scripts/train_safety_cvae.py --data $RUN/sweep.h5 --out $RUN/cvae --epochs 60
#   --no-wandb skips wandb (plots still written)
#   reference on this dataset: val mse 0.009, close-cos 0.93

# 3. Demos — each writes a Foxglove .mcap + an annotated .mp4
for d in flinch react sphere moving orbit; do
  env $EGL python scripts/safety_${d}_demo.py --ckpt $RUN/cvae --out $RUN/demos/${d}.mcap
done
#   common flags: --obstacle {bar,sphere}, --clean (3-metric HUD only)

# 4. Paper figures
env $EGL python scripts/figures.py --list
env $EGL python scripts/figures.py --all --outdir $RUN/figures
env $EGL python scripts/figures.py panel_coverage_behind --outdir $RUN/figures
```

On a fresh clone you can skip step 1 and train straight from the committed
`assets/safety/sweep_v3.h5`.

### What each demo shows

All five drive the **same** skin-only head; they differ only in obstacle motion and how the
head's output is applied.

- **Flinch mode** (`flinch`, `sphere`) — the arm rests at a clear posture and the head's
  baseline-subtracted output is integrated with a soft spring back to nominal, so the arm reacts
  only to the *change* a new obstacle causes.
- **Residual mode** (`react`, `moving`, `orbit`) — the head's push is low-passed and added as a
  growing/decaying correction on top of a nominal motion:

$$dq = \mathrm{head}(\text{skin with obstacle}) - \mathrm{head}(\text{skin, obstacle parked})$$

$$\text{correction} \mathrel{+}= (\text{gain}\cdot dq - \text{decay}\cdot\text{correction})\,\Delta t, \qquad q_{\text{exec}} = q_{\text{nom}} + \mathrm{clip}(\text{correction},\,\pm\,\text{max\_dev})$$

The baseline subtraction is the key trick: the head fires on *any* close surface (hood walls, the
arm's own links), so subtracting its rest output makes each demo react only to the obstacle.
**This is also a sim-only privilege that PACT does not have** — see trap 4 in §15.

| demo | what it demonstrates |
|---|---|
| `flinch` | Single bar marches down a forearm sensor's view axis; arm flinches away and relaxes home. |
| `sphere` | Identical motion with a blue sphere — the head, trained only on orange box bars, reacts the same, proving it is shape- and colour-agnostic (it only ever sees an 8×8 depth blob). |
| `react` | Reactive avoidance layered on a recorded reach-grasp-lift: the clock never stops, the arm bulges around bars and rejoins. |
| `moving` | Two bars patrol the whole chain (wrist→shoulder); each link tinted by its own obstacle-induced proximity. |
| `orbit` | One bar sweeps a half-circle over the outward face and slides elbow→wrist; sensors light up in turn. |

---

## 12. Recipe B — datagen

```bash
conda activate mlspaces
cd submodules/molmospaces
OMP_NUM_THREADS=2 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    python -m molmo_spaces.data_generation.main <ConfigName>
```

Keep `num_workers <= 2` or workers get OOM-killed (15–29 GB RSS each on a 62 GB box — the
`viz_sensor_rgb` issue in §7). Output lands under `assets/datagen/<name>/<ConfigName>/<timestamp>/`.

| config | what it collects |
|---|---|
| `FrankaSkinHybridObstacleConfig` | The main obstacle dataset (v1) |
| `FrankaSkinHybridInvisObstacleConfig` | The invisible-bar dataset (v2) |
| `FrankaSkinHybridFumehoodSmokeConfig` | Fumehood whole-arm-clearance reach, 40-sensor skin |
| `FrankaSkinEnclosureGenConfig`, `FrankaSkinEnclosureSmokeConfig` | General enclosure reach (full / smoke) |

To backfill houses that a crashed run missed, re-run the config with `num_workers` lowered —
`setup_house_dirs` skips houses that already have an h5, so it resumes rather than restarting.
The corollary: re-running into a directory that is already complete does nothing at all.

---

## 13. Recipe C — ACT and PACT

The control policy is upstream ACT trained on the one-env obstacle pick — red cup in the
fumehood, hazard bar present ~75% of episodes — using only the exo + wrist RGB cameras and joint
qpos. PACT adds the proximity token. Both are **evaluated in the exact datagen environment**, so
success rates are directly comparable.

Shapes: `qpos = 9` (arm 7 + 2 fingers), `action = 8` (arm 7 + gripper). Cameras `exo_camera_1`,
`wrist_camera` at 240×320.

```bash
conda activate mlspaces
cd /home/jaydv/code/prox_learning
DS=assets/datagen/hybrid_obstacle_v1/FrankaSkinHybridObstacleConfig/20260612_183855

# 1. Convert. One dataset serves vanilla + every PACT arm.
python -m scripts.convert_obstacle_to_act --src $DS \
    --dst act_style_data/obstacle_prox_v2 --with_proximity --image_h 240 --image_w 320
#    prints num_episodes / episode_len -> paste into submodules/act/constants.py if they differ

# 2. Train. Run from the act submodule so its local `detr` package imports.
cd submodules/act
PYTHONPATH="$PWD:$PYTHONPATH" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python imitate_episodes.py \
    --task_name obstacle_pact_v2 --policy_class ACT --ckpt_dir ckpts --kl_weight 10 \
    --chunk_size 100 --hidden_dim 512 --dim_feedforward 3200 --batch_size 8 --lr 1e-5 \
    --seed 0 --num_epochs 2000 --wandb_run_name <name> \
    [--use_proximity --prox_feature raw]      # PACT arm
    [--blur_sigma0 4 --blur_mode constant]    # degraded-vision arm
#    saves to ckpts/<task>/<datetime>_<runname>/ (printed at startup)
#    chunk_size/hidden_dim/dim_feedforward/kl_weight MUST match the eval defaults

# 3. Evaluate, one cell at a time. Proximity is auto-detected from prox_config.json.
for CELL in invisible free visible; do
  PYTHONPATH="$PWD:$PYTHONPATH" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python eval_act_obstacle.py \
      --ckpt_dir ckpts/obstacle_pact_v2/<run> \
      --output_dir /home/jaydv/code/prox_learning/eval_output/<run>_$CELL \
      --num_rollouts 50 --chunk_size 100 --temp_agg_off --eval_cell $CELL
done

# 4. Statistics.
cd /home/jaydv/code/prox_learning
python scripts/compare_pact.py vanilla=<S>/50,<C>/50 pact_raw=<S>/50,<C>/50
```

### The three eval cells

`--eval_cell` swaps in `InvisibleObstacleFumehoodPickCheckSampler` and forces its class attributes:

| cell | `OBSTACLE_P` | `INVIS_P` | meaning |
|---|---|---|---|
| `visible` | 1.0 | 0.0 | bar always present and rendered to RGB |
| `invisible` | 1.0 | 1.0 | bar physically present but hidden from the cameras — **skin-only** |
| `free` | 0.0 | — | no bar |

The same sampler runs in all three, so the object-placement distribution is identical and one
checkpoint is scored per cell across three invocations. Without the flag nothing changes and the
invisible-bar sampler is never even imported.

Every run writes `<output_dir>/eval_summary.json` — checkpoint, cell, sampler class, headline
numbers, and per-episode collision records. **That file is the result**; the `.h5` and `.mp4`
beside it are never read by anything.

### What eval reports

Besides success/total: a per-episode **collision** metric counting the distinct environment bodies
the arm penetrates each step (cavity wall, shelf, hazard bar, fumehood — excluding floor and the
grasped cup). Every run prints `strict_success_rate` (grasped and lifted **and** contact-free) and
`collision_rate` (= 1 − contact-free rate), so one standard run yields both the raw and hedged
numbers. `--end_on_collision` additionally makes any contact an immediate failure.

Budget: rollouts are ~3.5 min each, dominated by the sub-stepped 8×8 render over 40 sensors.
Memory is ~8 GB base + 0.5 GB per episode (41 GB RSS for a 50-rollout run), so run cells serially.

---

## 14. Every number in one place

| thing | value | source |
|---|---|---|
| Skin sensors | 40, each 8×8 depth, 45° cone | `model_hybrid.xml`, verified by compile |
| Skin directional coverage | 83% of all directions vs 10% for the wrist camera | sensor proofs, 06-11 |
| Skin accuracy | linear response; 5.6 mm error reconstructing a pipe | sensor proofs |
| Skin under blur / darkness | readings bit-for-bit identical; cameras collapse | sensor proofs |
| Reflex net direction accuracy (honest split) | 0.924 | audit, 06-13 |
| Reflex net error magnitude (honest split) | 69% worse than the cheating split | audit |
| Reflex net when obstacle is closest | outputs only 64% of the needed size | audit |
| Lookup-table baseline | ties the reflex net on direction (0.923) | audit |
| Camera-only baseline, best-success run | 40% success / 60% collisions (n=20) | June sweep |
| First skin attempt (v1), 50 rollouts | nothing beats cameras alone, p = 0.76 | 06-18 |
| Deflection decodable from skin? v1 | no — chance from every feature | v1 probes |
| Deflection decodable from skin? v2 | yes, ~0.75 | v2 probes |
| Bar presence decodable? v2 | no — chance. The formal gate FAILED. | v2 probes |
| Dataset v2 | 105 demos (47 visible / 49 hidden / 29 none) | `obstacle_prox_v2` |
| Prediction error: raw skin / cameras / reflex | **0.0595** / 0.0755 / 0.0830 | wandb |
| Cameras-only collisions (free / invisible / visible) | 60% / 66% / 64% | main grid, n=50 |
| Cameras-only success | 22% / 36% / 28% | main grid |
| **Raw-skin collisions** | 58% / **40%** / 50% | main grid |
| Raw-skin success | 18% / 30% / 16% | main grid |
| Trunk-feature collisions | 64% / 72% / 58% | main grid |
| **The headline** | **26 points better, Fisher p = 0.016** | main grid |
| Strict success (clean AND lifted), invisible | raw 20% vs vanilla 14% | main grid |
| Background brushing rate (no bar present) | ~60% | main grid |
| How much the bar itself adds to collisions | only ~4–6 points | main grid |
| Blur training-error ladder (σ = 0/2/4/8) | 0.0755 / 0.0836 / 0.0948 / 0.1100 | 07-24 |
| Blur robot behaviour | no pattern; noise dominates | 08-10 |
| **Measured noise at 25 rollouts** | **±40 points** | blur grid |
| Blur saturation | σ = 2 removes ~98% of fine detail | `blur_sweep_preview.png` |
| Test throughput | 3.56 min/rollout over 225 rollouts | blur grid |
| Test memory | 8 GB base + 0.5 GB/rollout | measured |

---

## 15. Traps

Every one of these has already cost real time.

1. **Temporal aggregation, and the `--temp_agg_off` bug.** With temporal aggregation on (default,
   `m=0.01`) the newest chunk — the only one that saw the *current* skin reading — carries ~1.6%
   of the executed action (mean staleness ~41 steps), structurally muting any reactive avoidance.
   But the *original* `--temp_agg_off` branch re-queried the policy every step and executed only
   `chunk[0]`, which is nearly a copy of the current qpos. That converges to a fixed point: the
   arm creeps toward the object, freezes ~30 cm short, and holds to timeout. Symptom: **0/N
   success with LOW collision for every checkpoint.** Fixed 2026-07-04 to standard open-loop
   chunking. **Any `--temp_agg_off` number produced before 2026-07-04 is invalid.**
2. **`viz_sensor_rgb` OOM.** Inherited `True` from the datagen chain; renders 40 × 256×256 RGB per
   policy step for cosmetic videos the policy never reads, at ~3 GB/episode. `eval_act_obstacle.py`
   forces it off. Datagen does not — keep `num_workers <= 2`.
3. **Sensor order.** `cvae_v3/meta.json["sensors"]` is authoritative, and `link5_back` precedes
   `link5_front`, opposite the env's tuple. Never hand-roll it.
4. **Baseline subtraction is a sim-only privilege.** The demos work because they subtract a
   per-frame obstacle-parked baseline. PACT cannot do that. In 100% of demo frames some fixture is
   within `D_MAX = 0.5 m`, and 40–60% of timesteps sit inside the `D_ACT = 0.18 m` repulsion zone
   while the demonstrated action is "proceed into the cavity" — so the danger feature fires
   constantly in collision-free demos (corr(‖delta‖, min_depth) ≈ −0.7), teaching BC to discount it.
5. **A low collision rate can mean "broken", not "careful."** The mildly-blurred policy has the
   fewest collisions of any trained, and also almost never finishes the task — it learned to barely
   move. Only the strict score (did the job **and** touched nothing) separates careful from broken.
6. **Training error does not predict behaviour.** The blurred arms' training errors form a perfect
   ladder while their actual behaviour is contradictory. Never ship a behavioural claim supported
   only by training numbers.
7. **`scene_params["cell"]` is not a label on obstacle runs.** It reads `"bar"` regardless, to skip
   a raycast rejection loop.
8. **Datagen resume is silent.** A house with an existing `.h5` is skipped. Re-running a complete
   run does nothing and prints nothing alarming.
9. **`imitate_episodes.py --eval` cannot evaluate anything in this project.** Use
   `eval_act_obstacle.py`.
10. **`franka_assets/` looks dead to grep and is not.** It is symlinked into
    `assets/robots/franka_skin/`.
11. **n = 25 is inside the noise band.** Measured noise at 25 rollouts is ±40 points. 50 is the
    declared floor for any real result.

---

## 16. Decision log

Compressed history. The full narrative for the most recent stretch is in `STATUS.md` and
`reports/2026-08-14/report.md`.

- **Pick-and-place → enclosures (June).** `proximity_necessity.py` showed that under a
  pick-and-place visibility constraint the skin adds ~nothing, because the target must stay
  visible. The project pivoted to enclosure/obstacle scenes where geometry is genuinely hidden.
- **29 → 40 sensors.** The hybrid gentact skin (`model_hybrid.xml`) replaced the 4-link,
  29-sensor model. Anything referencing `LINKS = 2/3/5/6` is pre-pivot.
- **Safety-CVAE v1 → v3.** Retraining on the obstacle dataset alone (`sweep_v3.h5`) gave the best
  head: val MSE 0.009, close-cos 0.926. v3 is the universal default.
- **PACT round 1 (v1 data) tied vanilla (2026-06-18).** The prox-token plumbing was audited
  line-by-line and is correct. The null result traced to four confirmed causes: the trained
  policies *ignore* the token (blanking the entire skin moves the predicted chunk by ~0.005
  normalized units; val loss identical across all three arms), ambient saturation with no baseline
  subtraction, temporal-aggregation washout at eval, and a signal-to-horizon mismatch (one skin
  snapshot conditions a 100-step chunk under uniform L1).
- **Probe gate on v1: FAIL (2026-07-03).** Deflect-vs-free is at chance from the trunk (ep-AUC
  0.526), from raw skin (0.500), and — crucially — from qpos alone (0.34–0.56). Four rescue probes
  (ambient residual, bar-station window, qpos-incremental, full 2560-d) all failed. The planner
  holds the hazard bar at the same clearance as the ambient walls, so deflecting and free
  approaches produce statistically identical skin frames. Not a feature-engineering failure.
- **v2 invisible-bar collection (2026-07-03).** `FrankaSkinHybridInvisObstacleConfig`; 3 of 4
  workers OOM-killed, 125 of 200 episodes landed, 105 usable. The v2 sampler always draws the
  bar-geometry variables, which removed a placement leak.
- **v2 probes inverted the picture.** Bar-presence collapsed to chance (0.40–0.52, was 0.72–0.78
  on v1 — so v1's number was largely that leak), while deflection *became* decodable (raw40
  0.749–0.763) and survived a qpos control. **`raw` beat `trunk` on every v2 label**, which is why
  the third feature tap exists. The formal gate still FAILED; training on v2 was a judgment call,
  justified by the eval-cell design making even a negative result attributable.
- **v2 results (2026-07-05).** The headline. `pact_raw` cuts invisible-cell collisions 66% → 40%
  (p = .016) with success unchanged in every cell. Controls hold: the free cell shows no
  difference between arms (p = 1.0), so this is not general timidity. Vanilla does *not* improve
  when the bar is visible (64% vs 66%) — 105 episodes never taught visual avoidance, so the raw
  skin tap is the only avoidance signal in any policy.
- **Trunk arm scrapped (2026-07-06).** Inert everywhere (p ≥ .67), consistent with both the v2
  probes and the v1 attention audit. Its checkpoints and eval JSONs stay on disk as the
  negative-result record. All round-2 work iterates on the raw tap only.
- **Blur sweep (2026-07-24 → 2026-08-10): no usable pattern.** Three constant-blur vanilla arms,
  225 test runs, 13.4 hours. Training error formed a clean ladder; behaviour did not. The run
  established the ±40-point noise band at n = 25 and the "low collisions can mean broken" lesson,
  and that is what it is worth.

### Unresolved

- Whether the main result holds with a different seed. One seed, one dataset.
- How much of any collision number is actually the hazard. The counter cannot tell hazard from
  wall, and the wall-brushing floor is ~60%.
- Whether the policy actually attends to the skin — inferred from indirect tests, never measured.
  **The instrument for this exists on `origin/encoder_eval` and was never merged (see §17).**
- Why the skin arm collides *less* with a hidden bar (40%) than with no bar at all (58%).
- Whether 105 demonstrations is the binding limit. The 200-episode version was never collected.

---

## 17. Repo state and housekeeping

### Git

- Branch `main`. **`submodules/act` pointer is one commit stale** — `git add submodules/act` fixes
  the `M submodules/act` in `git status`.
- **Untracked:** `reports/2026-08-14/` (the current progress report and its 9 figures — the only
  unbacked-up deliverable in the repo; commit it before any cleanup) and `check.md`, a Jetson AGX
  security-audit checklist unrelated to this project.
- **Fully merged into `main`, safe to delete:** `Cleanup`, `clean` (local); `origin/Cleanup`,
  `origin/clean`, `origin/cleanup`, `origin/main_old` (remote).
- **Two branches carry work that is NOT on `main`:**

| branch | unique commits | what is on it |
|---|---|---|
| `origin/encoder_eval` | 3 | **1756 lines of Safety-CVAE ablation infrastructure**: `safety_eval_lib.py` (546 L, batched closed-loop evaluator), `safety_ablation_eval.py` (matched-seed collision-course benchmark), `safety_openloop_ablation.py` (arm frozen, encoder direction vs the analytic oracle), `safety_sweep_ablation.py`, `safety_ablation_stats.py` (Wilson CI + **McNemar paired test**), `plot_safety_ablation.py` (figures + LaTeX table), `safety_probe_head.py`, plus a patched `safety_sweep.py` (+81/−34) |
| `origin/environments` | 8 | `scripts/audit_sensor_activation.py` (1482 L, read-only per-sensor activation audit) + activation audits for six datagen configs, from a collaborator |

`origin/encoder_eval` is the direct answer to the open question "does the head actually use the
skin, and is proximity what makes it work". It is worth merging deliberately.
`origin/environments` should be **cherry-picked** — merging it whole drags back `analysis_output/`
blobs and bumps the molmospaces submodule to a different commit.

### Disk — 166 GB total

| path | size | what |
|---|---|---|
| `submodules/act/ckpts` | 97 G | 12 runs × ~28 checkpoints each |
| `.git` | 29 G | history still carries deleted `.h5` datasets |
| `eval_output` | 15 G | 46 run dirs; 14.74 of 14.78 GB is `.h5` + `.mp4` that nothing reads |
| `submodules/MolmoBot/*/.venv` | 15 G | two virtualenvs for a submodule nothing imports |
| `assets/datagen` | 6.7 G | demo datasets |
| `assets/prox_learning_data` | 4.5 G | older datasets + a 2.6 GB HF-LFS `.git` |
| `act_style_data` | 2.5 G | 3 converted ACT datasets |
| `assets/.lmdb` | 1.3 G | regenerable MolmoSpaces cache |
| 1584 `__pycache__` dirs | 283 M | cache |

### `scripts/housekeeping.sh`

Tiered, **dry-run by default**, prints exactly what it would remove and how much it saves.

```bash
scripts/housekeeping.sh --tier1                 # preview
scripts/housekeeping.sh --tier1 --apply         # do it
scripts/housekeeping.sh --all --apply           # tiers 1-4
```

| tier | reclaims | what | risk |
|---|---|---|---|
| `--tier1` | 6.7 GB | `__pycache__`, `.egg-info`, smoke + `diag10` eval dirs, abandoned mug + pre-PACT runs, superseded v1 PACT evals, local wandb mirrors, orphaned checkpoints, leftover git temp packs | none |
| `--tier2` | 90 GB | intermediate `policy_epoch_*.ckpt`. **Nothing in the repo reads them**: every evaluator hardcodes `policy_best.ckpt` and all 24 `eval_summary.json` record it. Only two write-sites exist. `--keep-every-500` saves 72 GB instead and keeps a coarse ladder. | none proven; irreversible if you ever want a mid-training probe |
| `--tier3` | 15 GB | eval `.h5` and `.mp4`. `eval_summary.json` (0.21 MB total) holds every metric and is untouched. `--keep-headline` keeps the two invisible-cell demo videos. | loses qualitative video and the ability to re-derive new metrics from old rollouts |
| `--tier4` | 1.7 GB | `act_style_data/{obstacle_v1,obstacle_prox_v1}` — regenerable from the datagen run | breaks two `constants.py` entries until regenerated |
| `--assets` | 3.9 GB | `assets/prox_learning_data/.git` (re-cloneable from HF) and `assets/.lmdb` (rebuilds on demand) | none |
| `--venvs` | 9 GB | the two MolmoBot virtualenvs | only if you never run MolmoBot |
| `--gitgc` | ~4 GB | `git gc --prune=now`; does **not** rewrite history | none |

Total: **125 GB of 166 GB.**

### Not automated, on purpose

- **Rewriting git history** to drop the ~20 GB of deleted `.h5` blobs still in `.git`
  (`rollout_output/`, `analysis_output/`, `reports/demo_pnp/`, old `eval_output/` — all from
  commits `a5ffcf4 add eval output` → `6cfb928 remove eval output`). Needs `git-filter-repo` and a
  force-push across 8 branches. Real 20 GB, real risk. Your call.
- **Deleting git-tracked dead files.** These are repo changes, not disk cleanup:
  `assets/mjcf/*` (broken), `assets/urdf/fr3_full_skin*.urdf`, `assets/urdf/robotiq.urdf`,
  `assets/robots/franka_skin/model.xml.bak_before_orientation_fix`,
  `assets/mjthor_data_type_to_source_to_versions.json.bak` (byte-identical to the live file),
  `assets/reference_images/annotated/_fr3_skin_patched.xml`, `pointcloud.ipynb`,
  `scripts/assemble_overnight_report.py`, `submodules/act/eval_act_with_prox.py`.
  `diagnostics_output/` is 225 MB of tracked files — deleting it changes the repo for very little.
- **`assets/README.md`** describes a deleted `pla/` package end to end. It should be replaced by a
  pointer to §8 of this file rather than left to mislead.

### Do not delete

`franka_assets/` (symlink target of the live model) · `assets/datagen/hybrid_obstacle_v1` ·
`assets/datagen/hybrid_invis_obstacle_v1` · `assets/safety/` · `assets/robots/franka_skin/` ·
`assets/robots/fr3_hybrid_skin/meshes/skin/` · `assets/urdf/fr3_hybrid_skin.urdf` ·
`act_style_data/obstacle_prox_v2` · `reports/2026-08-14/` (commit it first).
