# prox_learning — proximity-skin sensing & safety for the Franka FR3

A Franka FR3 wearing **40 proximity sensors** in MuJoCo, plus the policies and analysis that
answer one question: *does a proximity skin make a robot arm safer when the cameras cannot see
the obstacle?*

<p align="center">
  <a href="experiments_output/default/environment_viz/FrankaSkinCabinetCavitySmokeConfig/cabinet_cavity_house_0/sample_00/01_robot_scene.png">
    <img src="experiments_output/default/environment_viz/FrankaSkinCabinetCavitySmokeConfig/cabinet_cavity_house_0/sample_00/01_robot_scene.png"
         alt="Franka FR3 with full-body proximity skin reaching into a cabinet cavity"
         width="100%">
  </a>
</p>
<p align="center">
  <sub>Canonical 40-sensor hybrid skin in a cabinet-cavity task — three automatically selected, text-free views.</sub>
</p>

**Current answer (2026-07-05, 50 rollouts per cell):** yes, on the safety axis. A vision-only ACT
policy collides in **66%** of episodes when a hazard bar is hidden from the cameras. The same
policy with a raw-skin token (**PACT-raw**) collides in **40%** (Fisher p = 0.016). Task success
is unchanged. Safer, not a better picker.

This file is the whole project document: what we built, what the numbers say, what you may claim,
how to run every experiment, and what will bite you. Figures with PNGs from the 2026-08-14 weekly
writeup live in [`reports/2026-08-14/report.md`](reports/2026-08-14/report.md). Agent protocol is
[`CLAUDE.md`](CLAUDE.md); session log is [`CURSOR.md`](CURSOR.md).

---

## Contents

1. [Now — disk truth, 2026-08-27](#1-now--disk-truth-2026-08-27)
2. [Routing table — "I want to…"](#2-routing-table--i-want-to)
3. [Setup](#3-setup)
4. [How to run](#4-how-to-run)
5. [What this is](#5-what-this-is)
6. [Headline result](#6-headline-result)
7. [Every experiment, one line](#7-every-experiment-one-line)
8. [Paper claims](#8-paper-claims)
9. [Failed / retracted](#9-failed--retracted)
10. [Method](#10-method)
11. [Repo map](#11-repo-map)
12. [Recipe detail](#12-recipe-detail)
13. [Every number in one place](#13-every-number-in-one-place)
14. [Traps](#14-traps)
15. [Decision log](#15-decision-log)
16. [Housekeeping](#16-housekeeping)

---

<a id="1-now--disk-truth-2026-08-27"></a>
## 1. Now — disk truth, 2026-08-27

| Item | Status |
|---|---|
| **Headline 66% → 40%** (invisible-cell collisions, n=50, p = 0.016) | **The paper number.** Source datagen (`hybrid_obstacle_v1` / `hybrid_invis_obstacle_v1`), converted `obstacle_prox_v2`, and July ckpts were wiped 2026-08-24. Archived metrics: `reports/eval_summaries/`. Not retrainable from this checkout until you collect again. |
| Hallway `pact_place_corridor_v5` n=50 | **Done, not a paper number.** Place-success 28% vs 42% (ACT vs PACT-raw, Fisher p = 0.21); bar hit 34% vs 36% (p = 1.0). No safety win. Success gap is noise. n=20 smoke (15% vs 35%, bar 30% vs 20%) was luck. Data + ckpts + eval on disk. |
| Gate-bar v3.1 collect | **Parked.** Only `assets/datagen/hybrid_gate_bar_check` (and clutter check). Do not collect 200 until the Visible check shows a tall pole in the doorway and an ~18 cm veer. |
| Surface-embedding bake into ACT | **Parked** as an ablation. Compressor gate passed (20.6 mm XYZ). Do not bake 32-d HDF5 tokens. |
| Surface readout finetune | **Live path.** Unfreeze the pretrained geometry net. ACT sees 128-d CLS readout tokens, same at train and eval. `--finetune_prox_encoder`. No numbers yet. Headline arm stays PACT-raw until this eval exists. |
| Safety-CVAE `cvae_v3/model.pt` | **Deleted 2026-08-24.** PACT-raw never needed it. Retrain from `assets/safety/sweep_v3.h5` if you want the reflex demos. `--prox_feature trunk` / `delta` need those weights and are negative controls. |
| Live training set | `act_style_data/pact_place_corridor_v5` (152 eps). Not `obstacle_prox_v2`. |

Proximity is redundant when vision already explains the demonstration. It helps when cameras
cannot. That is the thesis. Nominal hallway pick-and-place did not produce a safety win. The
hidden-bar cell did.

---

<a id="2-routing-table--i-want-to"></a>
## 2. Routing table — "I want to…"

| I want to… | jump |
|---|---|
| run anything | [§3 Setup](#3-setup) then [§4 How to run](#4-how-to-run) |
| reproduce hallway ACT vs PACT | [§4.3](#43-live--hallway-act-vs-pact) |
| run Amine's 40-row place protocol on local ckpts | [§4.3.1](#431-live--amine-40-row-place-protocol) |
| finetune the skin encoder into ACT | [§4.4](#44-live--corridor-skin-fire--compress-skin) |
| collect the next obstacle set (gate-bar) | [§4.7](#47-parked--gate-bar-v31) |
| understand the 66% → 40% result | [§6](#6-headline-result) |
| see every test in one line | [§7](#7-every-experiment-one-line) |
| paste a paper-writing brief | [§8](#8-paper-claims) |
| know the tensors | [§10](#10-method) |
| inspect a scene before collecting | [§4.2](#42-live--inspect-scenes) |
| visualize a cloned h5 folder | [§4.2.1](#421-live--visualize-a-dataset-folder) |
| browse every viz as a dashboard | [§4.2.1](#421-live--visualize-a-dataset-folder) |
| free disk | [§16](#16-housekeeping) |

---

<a id="3-setup"></a>
## 3. Setup

```bash
conda activate mlspaces
cd /home/jaydv/code/prox_learning
export OMP_NUM_THREADS=2 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
```

Prefix every render / datagen / train / eval command with that EGL pair. One GPU. Eval is
**serial** (one arm, one cell at a time).

`~/.bashrc` exports `MLSPACES_ASSETS_DIR=/home/jaydv/code/prox_learning/assets`. This repo's
`assets/` **is** the MolmoSpaces asset root. Datagen output lands here.
`assets/{scenes,objects,grasps,test_data,benchmarks}` are symlink farms into
`~/.cache/molmo-spaces-resources/`.

`pyproject.toml` is **not installed**. It declares an empty package — scripts plus assets. The
`mlspaces` conda env supplies dependencies. Scripts import siblings via `sys.path.insert`.

One-time extras `mlspaces` is missing for ACT:

```bash
pip install ipython
pip install wandb && wandb login    # --no_wandb opts out
```

Sanity:

```bash
conda activate mlspaces
cd /home/jaydv/code/prox_learning
python -m encoders                                          # names + dummy-tensor shapes
python -m pytest tests/test_encoders.py tests/test_prox_raw.py
```

Never `python imitate_episodes.py --eval` on a PACT checkpoint. That path calls
`policy(qpos, image)` with no skin and now `SystemExit`s if `--use_proximity` is set. Real eval is
`eval_act_obstacle.py` or `eval_act_place_corridor.py`.

---

<a id="4-how-to-run"></a>
## 4. How to run

Shared flags first. Then copy-paste per experiment. Each block is marked **live** (data on this
disk), **parked** (commands ready; do not skip preflight), or **needs datagen** (source wiped;
collect or restore first).

### 4.0 Shared ACT train / eval / stats

From `submodules/act`:

```bash
conda activate mlspaces
cd /home/jaydv/code/prox_learning/submodules/act
export PYTHONPATH="$PWD:$PYTHONPATH"
export OMP_NUM_THREADS=2 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl

python imitate_episodes.py --task_name <TASK> --policy_class ACT --ckpt_dir ckpts \
    --kl_weight 10 --chunk_size <50|100> --hidden_dim 512 --dim_feedforward 3200 \
    --batch_size 8 --lr 1e-5 --seed 0 --num_epochs 2000 --wandb_run_name <name>
# PACT: add  --use_proximity --prox_feature raw --prox_layout per_sensor
```

CLI default is `--prox_feature raw` and `--prox_layout per_sensor`. **Train `raw`.** `trunk` /
`delta` are negative controls and need deleted CVAE weights.

Obstacle eval (fumehood pick, three cells):

```bash
python eval_act_obstacle.py \
    --ckpt_dir ckpts/<task>/<dated>_<run> \
    --output_dir /home/jaydv/code/prox_learning/eval_output/<run>_<cell> \
    --num_rollouts 50 --chunk_size <50|100> --temp_agg_off \
    --eval_cell <visible|invisible|free>
# gate-bar ckpts also need:  --eval_sampler gate
```

`--eval_cell` pins every rollout:

| cell | bar present | cameras see bar | skin feels bar | what it tests |
|---|---|---|---|---|
| `visible` | yes | yes | yes | ordinary case |
| `invisible` | yes | **no** | yes | **causal / paper cell** |
| `free` | no | — | — | background brushing; statue check |

Invisible bar = MuJoCo geom group 4: camera renderer skips it, skin renderer includes it. Physics
is unchanged.

Stats (you type counts; it reads no files):

```bash
cd /home/jaydv/code/prox_learning
python scripts/compare_pact.py vanilla=<S>/50,<C>/50 pact_raw=<S>/50,<C>/50
```

Floor for any claim: **n = 50**. n = 25 is inside a measured ±40-point noise band. Always report
**strict success** (task done **and** contact-free) next to collisions. A low collision rate can
mean "broken" (barely moves), not "careful."

### 4.1 Live — tests

```bash
conda activate mlspaces
cd /home/jaydv/code/prox_learning
python -m pytest tests/test_encoders.py tests/test_prox_raw.py
```

<a id="42-live--inspect-scenes"></a>
### 4.2 Live — inspect scenes

Visualizer lives in the submodule, not `scripts/`. It samples the same config as collection, then
forces the 40-sensor hybrid skin. `--show-hidden` reveals geom group 4 (invisible bars).

```bash
conda activate mlspaces
cd /home/jaydv/code/prox_learning/submodules/molmospaces

python scripts/datagen/visualize_environment.py --list --scope project

OMP_NUM_THREADS=2 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    python scripts/datagen/visualize_environment.py \
    FrankaSkinHybridClutterPnPCheckConfig \
    --format both --show-hidden

OMP_NUM_THREADS=2 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    python scripts/datagen/visualize_environment.py --all \
    --format both --show-hidden --attempts 10
```

Outputs: `experiments_output/default/environment_viz/<Config>/<scene>_house_<id>/sample_00/`.
Open `gallery.html` after `--all`. Useful switches: `--format png|mp4|both`,
`--keep-scene-lighting`, `--keep-config-robot`, `--show-sensors`, `--dry-run --all`.

<a id="421-live--visualize-a-dataset-folder"></a>
### 4.2.1 Live — visualize a dataset folder

`scripts/dataset_viz.py` reads a folder of trajectory HDF5s and writes one Foxglove `.mcap`
(3D FK, RGB, skin heatmap, joint plots — still one concatenated timeline) plus **one short
tiled MP4 per episode** and an `index.html` that browses them. The MP4 layout is
wrist | table | **live prox 3D**, then the 8×8 mosaic **fills** the leftover left
panel (tiles stretch; no blank band). qpos / qvel / HUD sit below. Missing wrist /
table / heatmap tiles are **dropped** (no slate). Leftover RGB grows; heatmap takes
the whole left column when no RGB. Hallway v5 has **no table / exo** — that slot is
gone, wrist uses the full left width. **v10.10** (`data/pact_place_corridor/data/v1010`)
**does** ship table + wrist; the sidecar names are `episode_{sha}_{wrist,table}_camera.mp4`,
not `episode_00000000_exo_camera_1.mp4`. A glob that only knew the padded-int / `exo`
names dropped both RGB tiles (heatmap + 3D only). `glob_mp4` now accepts the hash
name and `table_camera`. Skin comes from the tensor, not the sidecar
heatmap mp4. dt defaults to `obs_scene.policy_dt_ms` or **66 ms** (not ACT's train
`DT=0.02`).
ACT hdf5 has no saved `cam2world`; the 3D panel uses MuJoCo FK camera poses. Datagen /
HF rows use saved `cam2world_gl` when present.

All live clones sit under `data/`. Do **not** pass that parent without `--list` / `--each` —
it is a mixed tree (hallway rows + the `molmo-pi0-eval-videos` dump). `--list` prints one
row per dataset (`DUP` = nested copy of openfront 52 / raw_h5). `--each` writes one viz
dir per **unique** row and rebuilds the root dashboard (`index.html` + `audit.json`).
`results/` eval rollouts stay out unless `--include-eval`. `--keep-dups` keeps the
copies. Full-folder audit skips Foxglove (`--no-mcap`) and uses `--stride 2` so encode
finishes. Browse with `--serve`, not a Cursor preview of every mp4.

**The output location is fixed — there is no `--out` flag (removed 2026-08-31).** Every run
writes under `experiments_output/default/dataset_viz/`, in a folder that mirrors the dataset
path with the `data/` prefix removed:

| dataset path | output folder |
|---|---|
| `data/molmo-pi0-eval-videos/data/fumehood/pick` | `…/dataset_viz/molmo-pi0-eval-videos/data/fumehood/pick/` |
| `data/pact_place_corridor_v5` | `…/dataset_viz/pact_place_corridor_v5/` |
| `act_style_data/foo` (in the checkout, not under `data/`) | `…/dataset_viz/act_style_data/foo/` |
| `/mnt/scratch/ds` (outside the checkout) | `…/dataset_viz/mnt/scratch/ds/` |

A single `.h5` file maps to `<parent>/<stem>/`. The root
`experiments_output/default/dataset_viz/index.html` is a **dashboard**, not a
table of preloaded videos. It reads `audit.json` (one row of metadata per dataset)
and loads **one** clip plus `timeline.json` plots when you click a row. Opening
the HTML as `file://` shows the baked catalog and plays clips; Plotly plots and
live refresh need a tiny HTTP server because browsers block `fetch` on
`file://`:

```bash
python scripts/dataset_viz.py --dashboard   # rewrite catalog only, no encode
python scripts/dataset_viz.py --serve       # http://127.0.0.1:8765/
```

`--serve` re-scans nested `audit.json` files on each catalog request, so a
parallel `--each` run shows new datasets within a few seconds. `--each` also
rewrites the catalog after every dataset. Do not open the root `index.html` in
Cursor's simple preview if you want plots — use `--serve`. Folders written
before 2026-08-31 use the old flat `a_b_c` slug; they still show in the index —
delete them or re-run to get the nested layout. `--force` redoes a dataset that
already has an audit plus at least one video.

**Video is one clip per episode, filed by trajectory type (changed 2026-08-31).** The old
single hour-long `dataset.mp4` is gone by default. Each dataset folder now holds:

```
<dataset>/episodes/<trajectory type>/<idx>_<episode label>.mp4   # e.g. free/0007_house_1_traj_3.mp4
<dataset>/index.html                                             # clip list, grouped by type
<dataset>/dataset.mcap                                           # still the whole timeline
```

The type is the first attribute the episode carries out of `behavior_class`,
`intrusion_side`, `has_bar`, `clean_success`; with none of them it falls back to
success / fail, then to `all`. `--group-by ATTR` names the folder from a different
attribute (`--group-by traj` gives one folder per trajectory index). `--one-video`
restores the old single concatenated `dataset.mp4`.

`index.html` lists the clips under a sticky per-type header. Clicking one loads that clip
and zooms the qpos / qvel / skin plots to that episode; the playhead still reads global
dataset time, so the plots line up with the `.mcap`.

Clip playback is now **wall-clock real time**: the writer divides the frame rate by
`--stride` (before, `--stride 2` silently played at 2× speed).

**`--cam3d` puts the returns in the real scene (added 2026-08-31, opt-in).** By default the
right panel is `render_view3d`: the skin returns floating beside a stick-figure arm, which
does not say *what* a point hit. With `--cam3d` the panel becomes `render_cam3d`: the table
(exo) camera's own frame with the returns drawn on top, so a point sits on the pixels of the
wall, shelf or object it bounced off. Nothing is replaced — `render_view3d` is untouched and
stays the default; drop the flag to get it back.

```bash
python scripts/dataset_viz.py --data data/molmo-pi0-eval-videos/data/fumehood/pick_and_place     --cam3d --max-episodes 2 --stride 4 --no-mcap --force
```

How it projects: `obs/sensor_param/<cam>/extrinsic_cv` (world→camera) and `intrinsic_cv` (K)
are read per frame into `Episode.cam_params`, and the world points from
`proximity_world_points` go through `K · (R·p + t)`. **Trap:** the stored `intrinsic_cv` is
the calibration of a **480×480** render (`cx = cy = 240`), but the sidecar mp4 is **624×352**
with the same *vertical* field of view. `camera_matrices` rescales the focal length by
`H_frame / (2·cy)` and re-centres the principal point; without that the points sit ~35 px low
and spread too wide. The correction is a no-op when the frame already matches the stored size.

Three limits, by design:
- **No occlusion test.** A point behind the hood wall still draws over the wall. The panel
  shows where a return *projects*, not whether the camera could see it.
- **Datagen h5 only.** ACT `episode_*.hdf5` stores no camera calibration, so `render_cam3d`
  returns `None` and the old 3D panel is drawn instead. No flag change, no error.
- **Letterboxed.** A 624×352 frame in a 400×480 panel leaves bars above and below. Ask if you
  want it cropped to the points instead.

```bash
conda activate mlspaces
cd /home/jaydv/code/prox_learning

# catalog everything under data/ (no write)
python scripts/dataset_viz.py --data /home/jaydv/code/prox_learning/data --list

# rewrite the dashboard catalog after encodes (cheap)
python scripts/dataset_viz.py --dashboard

# browse stats + one clip + plots (plots need this, not file://)
python scripts/dataset_viz.py --serve

# smoke one viz per dataset (2 eps each)
python scripts/dataset_viz.py --data /home/jaydv/code/prox_learning/data \
    --each --max-episodes 2

# audit: every unique dataset, all episodes, H.264 + 3D panel (no Foxglove)
python scripts/dataset_viz.py --data /home/jaydv/code/prox_learning/data \
    --each --no-mcap --stride 2

# hallway HF rows (152)   -> .../dataset_viz/pact_place_corridor_v5/
python scripts/dataset_viz.py --data data/pact_place_corridor_v5 \
    --max-episodes 2

# v10.10 four-object (hash mp4 ids, table+wrist). --cam3d = wrist overlay
# (no table calib in this dump). Full set: drop --max-episodes.
#   -> .../dataset_viz/pact_place_corridor/data/v1010/accepted/
python scripts/dataset_viz.py \
    --data data/pact_place_corridor/data/v1010/accepted \
    --cam3d --no-mcap --stride 2 --force

# fumehood pick houses (datagen, exo+wrist)
#   -> .../dataset_viz/molmo-pi0-eval-videos/data/fumehood/pick/
python scripts/dataset_viz.py \
    --data data/molmo-pi0-eval-videos/data/fumehood/pick \
    --max-episodes 2

# open-front ACT hdf5 (52)
#   -> .../dataset_viz/molmo-pi0-eval-videos/data/openfrontcluttered/act_style_52/data/act_style/
python scripts/dataset_viz.py \
    --data data/molmo-pi0-eval-videos/data/openfrontcluttered/act_style_52/data/act_style \
    --max-episodes 2
```

Open `http://127.0.0.1:8765/` after `--serve`, or any `episodes/<type>/*.mp4` in Cursor / VS Code (H.264
`yuv420p` — MPEG-4 `mp4v` will not play in the IDE). Foxglove: open `dataset.mcap` in
[Foxglove](https://app.foxglove.dev) and import `foxglove_layout.json` from the same out dir.
`--no-mcap` / `--no-video` skip one side. `--stride 2` halves cost and keeps real-time
playback. `--reencode DIR` H.264-remuxes every `.mp4` under `DIR`. `--include-sensor-rgb` pulls the 256²
mosaic sidecar (large). `foxglove_viz.py` remains the older datagen-only exporter.

<p align="center">
  <a href="experiments_output/default/environment_viz/FrankaSkinCabinetCavitySmokeConfig/cabinet_cavity_house_0/sample_00/02_sensor_cones.png">
    <img src="experiments_output/default/environment_viz/FrankaSkinCabinetCavitySmokeConfig/cabinet_cavity_house_0/sample_00/02_sensor_cones.png"
         alt="Colour-coded proximity-sensor fields of view around the Franka arm"
         width="49%">
  </a>
  <a href="experiments_output/default/environment_viz/FrankaSkinCabinetCavitySmokeConfig/cabinet_cavity_house_0/sample_00/03_cameras_and_sensors.png">
    <img src="experiments_output/default/environment_viz/FrankaSkinCabinetCavitySmokeConfig/cabinet_cavity_house_0/sample_00/03_cameras_and_sensors.png"
         alt="Exocentric and wrist cameras with paired proximity-depth and RGB atlases"
         width="49%">
  </a>
</p>

<a id="43-live--hallway-act-vs-pact"></a>
### 4.3 Live — hallway ACT vs PACT

Source: HuggingFace `Lundii/pact_place_corridor_v5` cloned to `data/pact_place_corridor_v5`.
152 recovered pick-and-place demos (`clean_success`). Wrist RGB only. Scene XML
`pact_place_corridor_v2`. Eval env is **not** molmospaces `main` — pin worktree `977acd6`.

Local n=50 (2026-08-27, metrics-only): **not a paper number.**

| arm | ckpt dir | place-success | bar hit | collision-free |
|---|---|---|---|---|
| ACT | `20260825_161821_act_place_corridor_s0` | 14/50 (**28%**) | 17/50 (**34%**) | 33/50 (66%) |
| PACT-raw | `20260825_215846_pact_place_corridor_raw_s0` | 21/50 (**42%**) | 18/50 (**36%**) | 32/50 (64%) |

Fisher two-sided: success p = 0.21, bar-hit p = 1.0. PACT does **not** cut bar hits.

```bash
conda activate mlspaces
cd /home/jaydv/code/prox_learning
export OMP_NUM_THREADS=2 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl

# 0. Eval env (once). Leaves submodules/molmospaces on main.
git -C submodules/molmospaces worktree add \
    /home/jaydv/code/molmospaces-pact-place \
    977acd6719a8c05b688d3e70da356d61dd32d259

# 1. Convert (already done: 152 eps, episode_len=636). Re-run only if hdf5 gone.
python -m scripts.convert_pact_place_to_act \
    --src data/pact_place_corridor_v5 \
    --dst act_style_data/pact_place_corridor_v5 \
    --with_proximity --prox_pool min --image_h 240 --image_w 320
# paste printed num_episodes / episode_len into
# TASK_CONFIGS['pact_place_corridor_v5'] in submodules/act/constants.py

# 2. Train. Wrist-only, chunk 50, NO image dropout. One GPU — serial.
cd submodules/act
export PYTHONPATH="$PWD:$PYTHONPATH"

python imitate_episodes.py \
    --task_name pact_place_corridor_v5 --policy_class ACT --ckpt_dir ckpts \
    --kl_weight 10 --chunk_size 50 --hidden_dim 512 --dim_feedforward 3200 \
    --batch_size 8 --lr 1e-5 --seed 0 --num_epochs 2000 \
    --wandb_run_name act_place_corridor_s0

python imitate_episodes.py \
    --task_name pact_place_corridor_v5 --policy_class ACT --ckpt_dir ckpts \
    --kl_weight 10 --chunk_size 50 --hidden_dim 512 --dim_feedforward 3200 \
    --batch_size 8 --lr 1e-5 --seed 0 --num_epochs 2000 \
    --use_proximity --prox_feature raw --prox_layout per_sensor \
    --wandb_run_name pact_place_corridor_raw_s0

# 3. Eval. Metrics-only by default (no MP4/HDF5). PYTHONPATH worktree FIRST.
#    Vanilla skips 40-sensor 60 Hz depth. --temp_agg_off gates PACT 8×8 skin
#    to chunk queries (not every control step).
PYTHONPATH="/home/jaydv/code/molmospaces-pact-place:$PWD:$PYTHONPATH" \
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python eval_act_place_corridor.py \
    --ckpt_dir ckpts/pact_place_corridor_v5/20260825_161821_act_place_corridor_s0 \
    --output_dir /home/jaydv/code/prox_learning/eval_output/place_corridor_vanilla_s0_n50 \
    --num_rollouts 50 --chunk_size 50 --temp_agg_off --task_horizon 800

PYTHONPATH="/home/jaydv/code/molmospaces-pact-place:$PWD:$PYTHONPATH" \
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python eval_act_place_corridor.py \
    --ckpt_dir ckpts/pact_place_corridor_v5/20260825_215846_pact_place_corridor_raw_s0 \
    --output_dir /home/jaydv/code/prox_learning/eval_output/place_corridor_raw_s0_n50 \
    --num_rollouts 50 --chunk_size 50 --temp_agg_off --task_horizon 800
```

`--save_trajectories` restores the slow datagen path (~16 min/ep, OOM near 30). Default
metrics-only: ACT ~1 min/ep, PACT-raw ~15.5 min/ep. Place-corridor is **not** the obstacle
`--eval_cell` loop; the bar is in the sampler.

<a id="431-live--amine-40-row-place-protocol"></a>
### 4.3.1 Live — Amine 40-row place protocol on local ckpts

Amine's launcher (`run_pact_place_eval_chunk100.py` on his box) is a hashed contract:
chunk **100**, frozen **32-d** embeddings, `run_manifest.json`, encoder sha
`6fd2dd03…`, seed 3101, 10 workers, plus `PACT_PERMUTED` from a `(40, 900, 40, 32)`
token plan. That worker is `amine/act/eval_pact_place_chunk100_row.py`. It **cannot**
`strict=True` load the three local hallway ckpts (chunk 50; raw K=8 dim 1; readout
128-d CLS). `PACT_PERMUTED` stays off until a 32-d plan exists for these models.

What we **do** reuse: his frozen **40 scenes** (20 left / 20 right, master seed
`2026082101`, `PactPlaceCorridorV2Sampler`, house 1). Same XML, same contact audit.
Policy load stays `eval_act_place_corridor.py` (`prox_config.json`, `--temp_agg_off`).
Horizon 900 (his eval length). Skin stack is still `HYBRID_SKIN_SENSOR_ORDER`
(link5_back before front). His manifest names front first; that list is **not** used
to stack tokens.

One GPU. `--workers 10` is clamped to 2. Smoke = rows 0 and 1. Full = 40 per arm.

```bash
conda activate mlspaces
cd /home/jaydv/code/prox_learning
export OMP_NUM_THREADS=2 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl

# smoke first (2 rows × arms). Stop if gripper never closes.
python scripts/run_pact_place_eval_chunk100.py \
    --manifest configs/pact_place_eval_chunk100_manifest.json \
    --output-root /home/jaydv/code/prox_learning/eval_output/pact_place_chunk100_jay \
    --mode smoke \
    --workers 1 \
    --arms ACT PACT PACT_READOUT

# full 40-row protocol
python scripts/run_pact_place_eval_chunk100.py \
    --manifest configs/pact_place_eval_chunk100_manifest.json \
    --output-root /home/jaydv/code/prox_learning/eval_output/pact_place_chunk100_jay \
    --mode full \
    --workers 1 \
    --arms ACT PACT PACT_READOUT
```

`--arms ACT PACT PACT_PERMUTED` (his paste) runs ACT + PACT-raw and **skips**
permuted. PACT here is **PACT-raw**, not his 32-d screen PACT. Readout is the extra
arm `PACT_READOUT`. Kill-safe: existing `result.json` with `status=complete` is
skipped. Primary local endpoints stay place-success, `collision_free_task_success`,
bar-hit, `gripper_close_commanded`. **Not a paper number.** Do not mix with the
random-house n=50 in [§4.3](#43-live--hallway-act-vs-pact) or with his seed-3101
chunk-100 table.

**Why PACT is 12–15 h/model.** Measured 2026-08-29 smoke: ACT 119.5 s / 2 eps.
PACT-raw **2121 s / 2 eps (~18 min/ep)** with `renders=19 skip=883`. Chunk-gate
worked. The leftover tax is 19 × 40 EGL `update_scene` (~1.4 s/camera on this
house). Default eval skin is now `mj_multiRay` (center pixel, geom group 2
hidden). `--egl-prox` is the old rasterizer; do not use it on the inner loop.
`--fast-prox-rays` is already the default.

**18-day loop.** Ctrl+C the current PACT_READOUT EGL run. Same 18 min/ep. Logs
now stream. Inner loop: one arm, smoke (rays default). Direction:
`--mode full --limit-rows 10`. 40-row table also rays unless you pass
`--egl-prox` on a 2-row check. Train 2000 ep is a separate 12 h.

```bash
# kill the EGL readout first (Ctrl+C), then:
python scripts/run_pact_place_eval_chunk100.py \
    --manifest configs/pact_place_eval_chunk100_manifest.json \
    --output-root /home/jaydv/code/prox_learning/eval_output/pact_place_chunk100_jay \
    --mode smoke --workers 1 --arms PACT_READOUT

# 10-row direction cut
python scripts/run_pact_place_eval_chunk100.py \
    --manifest configs/pact_place_eval_chunk100_manifest.json \
    --output-root /home/jaydv/code/prox_learning/eval_output/pact_place_chunk100_jay_fast10 \
    --mode full --limit-rows 10 --workers 1 --arms PACT PACT_READOUT
```

<a id="44-live--corridor-skin-fire--compress-skin"></a>
### 4.4 Live — corridor skin fire + compress skin

Probe native rows `(T, 40, 4, 8, 8)` metres. No ACT convert needed. Without `--checkpoint` this
scores the analytic 20 cm nearest-surface **target** vs PACT-raw 50 cm peak closeness.

```bash
conda activate mlspaces
cd /home/jaydv/code/prox_learning

python -m encoders.probe \
    --src data/pact_place_corridor_v5 \
    --out experiments_output/default/surface_encoder_probe/pact_place_corridor_v5

python -m encoders.train \
    --src data/pact_place_corridor_v5 \
    --out experiments_output/default/surface_encoder_train/pact_place_corridor_v5 \
    --kind embedding --device cuda \
    --epochs 20 --batch-size 512 --stride 4 --num-workers 8

CKPT=experiments_output/default/surface_encoder_train/pact_place_corridor_v5/pact_surface_embedding_encoder_v1.pt
python -m encoders.probe \
    --src data/pact_place_corridor_v5 \
    --checkpoint "$CKPT" --split test \
    --kind embedding --device cuda --untrained-episodes 0
```

**152-episode run (2026-08-25):** 100% balanced validity and recall; recon pixel P/R 87.4 / 95.3%;
XYZ MAE **20.6 mm** (0.6 mm over the 20 mm preference). Hard gate pass. That grades the
compressor, not the robot.

**Do not bake 32-d tokens.** Frozen embedding HDF5 is an ablation. The live geometry arm
finetunes the encoder and feeds the **128-d CLS readout** (one token per sensor) into ACT at
train and at eval. Same forward. No `encode_tokens`. Live layout is `raw_causal` (last 8 pooled
steps). ACT still shuffles 80/20 vs the encoder `split_manifest.json` — this is a policy
experiment, not an honest compressor test.

```bash
conda activate mlspaces
cd /home/jaydv/code/prox_learning/submodules/act
export PYTHONPATH="$PWD:$PYTHONPATH"
export OMP_NUM_THREADS=2 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl

CKPT=/home/jaydv/code/prox_learning/experiments_output/default/surface_encoder_train/pact_place_corridor_v5/pact_surface_embedding_encoder_v1.pt

python imitate_episodes.py \
    --task_name pact_place_corridor_v5 --policy_class ACT --ckpt_dir ckpts \
    --kl_weight 10 --chunk_size 50 --hidden_dim 512 --dim_feedforward 3200 \
    --batch_size 8 --lr 1e-5 --seed 0 --num_epochs 2000 \
    --use_proximity --prox_feature surface_embedding --prox_layout per_sensor \
    --prox_encoder_ckpt "$CKPT" \
    --finetune_prox_encoder \
    --wandb_run_name pact_place_corridor_readout_s0
```

Eval (after the run dir exists). Loads `prox_encoder_best.pt` from that dir, not the pretrain
file. `--temp_agg_off` is required. With it on, RGB/skin EGL run only on chunk
queries (~16 / 800 steps). Same actions as the old every-step render; n=50 PACT
should finish under 4 h, not ~12 h.

```bash
conda activate mlspaces
cd /home/jaydv/code/prox_learning/amine/act
export OMP_NUM_THREADS=2 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl

python eval_pact_collision_row.py \
    --checkpoint-dir /home/jaydv/code/prox_learning/submodules/act/ckpts/pact_place_corridor_v5/20260828_003136_pact_place_corridor_readout_s0 \
    --output-dir /home/jaydv/code/prox_learning/eval_output/place_corridor_readout_s0_n50_fast \
    --num-rollouts 50

# same eval, from the ACT submodule:
cd /home/jaydv/code/prox_learning/submodules/act
PYTHONPATH="/home/jaydv/code/molmospaces-pact-place:$PWD:$PYTHONPATH" \
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python eval_act_place_corridor.py \
    --ckpt_dir ckpts/pact_place_corridor_v5/20260828_003136_pact_place_corridor_readout_s0 \
    --output_dir /home/jaydv/code/prox_learning/eval_output/place_corridor_readout_s0_n50_fast \
    --num_rollouts 50 --chunk_size 50 --temp_agg_off --task_horizon 800
```

`--prox_policy_tap readout` is implied by `--finetune_prox_encoder`. Do **not** pass baked
`--prox_feature surface_embedding` without `--finetune_prox_encoder` until ACT reuses
`split_manifest.json`. Headline corridor arm remains vanilla vs PACT-raw until this eval
exists.

Do **not** mix closeness maps: peak-closeness `D_MAX = 0.5 m`; surface geometry
`MAX_SURFACE_RANGE_M = 0.20 m` (trap 16).

<a id="45-live--reflex-retrain-and-demos"></a>
### 4.5 Live — reflex retrain and demos

Canonical `cvae_v3` weights are gone. Retrain from committed `assets/safety/sweep_v3.h5`. This is
the reflex-head path, **not** the PACT default.

```bash
conda activate mlspaces
cd /home/jaydv/code/prox_learning
export EGL="OMP_NUM_THREADS=2 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl"
RUN=experiments_output/default

env $EGL python scripts/train_safety_cvae.py \
    --data assets/safety/sweep_v3.h5 --out $RUN/cvae --epochs 60
# reference on the original obstacle sweep: val mse 0.009, close-cos 0.93

for d in flinch react sphere moving orbit; do
  env $EGL python scripts/safety_${d}_demo.py --ckpt $RUN/cvae --out $RUN/demos/${d}.mcap
done
```

To rebuild labels from a **new** obstacle datagen run (old `hybrid_obstacle_v1` path is gone):

```bash
env $EGL python scripts/safety_sweep.py --runs <datagen_run_dir> --n 15000 --out $RUN/sweep.h5
```

Demos all drive the same skin-only head. `react` is the one to show people — the arm swerves
around bars mid-motion and rejoins, clock never stopping. `sphere` proves shape/colour
agnosticism (trained on orange bars, reacts to a blue sphere). Baseline subtraction in the demos
is a **sim-only privilege PACT does not have** (trap 4).

### 4.6 Live — paper figures

```bash
conda activate mlspaces
cd /home/jaydv/code/prox_learning
OMP_NUM_THREADS=2 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python scripts/figures.py --list
OMP_NUM_THREADS=2 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    python scripts/figures.py --all --outdir experiments_output/default/figures
```

CLI keys, function names, and PNG filenames disagree in 12 of 29 cases. `--list` is the map.

<a id="47-parked--gate-bar-v31"></a>
### 4.7 Parked — gate-bar v3.1

Next *obstacle* collect. Both earlier obstacle datasets let cameras explain the bows. Avoid-v1
failed for that reason ([§9](#9-failed--retracted)). v3.1 snaps a **44 cm pole** onto the live
TCP line (~18 cm bow, sign = wall coin-flip, `INVIS_P=1` on collect). Detail: [§12.2](#122-gate-bar-v31).

```bash
conda activate mlspaces
cd /home/jaydv/code/prox_learning/submodules/molmospaces
export OMP_NUM_THREADS=2 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl

# 1. geometry debug — pole RENDERED. You must SEE it in the doorway and a ~18 cm veer.
python -m molmo_spaces.data_generation.main FrankaSkinHybridGateBarVisibleCheckConfig
# 2. invisible preflight — same geometry, pole hidden. Videos have no pole; log still DEFLECTs.
python -m molmo_spaces.data_generation.main FrankaSkinHybridGateBarCheckConfig
# 3. full run — ONLY after both checks pass. 8 houses × 25 = 200. viz_sensor_rgb OFF.
python -m molmo_spaces.data_generation.main FrankaSkinHybridGateBarConfig
```

**Visible-check pass:** orange pole in every exo MP4, in the doorway not a stub; `GATE SNAP` then
`DEFLECT` ~18 cm, **both signs**; grasp still works.

**Invisible-check pass:** no pole in MP4s; `[InvisBar] geom group 4`; same SNAP / DEFLECT / signs.

Then convert / train / eval:

```bash
cd /home/jaydv/code/prox_learning
python -m scripts.convert_obstacle_to_act \
    --src assets/datagen/hybrid_gate_bar_v1/FrankaSkinHybridGateBarConfig/<timestamp> \
    --dst act_style_data/obstacle_gate_v1 \
    --with_proximity --prox_pool min --skip_approach_collision \
    --image_h 240 --image_w 320
# paste num_episodes / episode_len into TASK_CONFIGS['obstacle_gate_v1']

cd submodules/act
export PYTHONPATH="$PWD:$PYTHONPATH" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl

python imitate_episodes.py --task_name obstacle_gate_v1 --policy_class ACT \
    --ckpt_dir ckpts --kl_weight 10 --chunk_size 50 --hidden_dim 512 \
    --dim_feedforward 3200 --batch_size 8 --lr 1e-5 --seed 0 --num_epochs 2000 \
    --wandb_run_name act_gate_s0

python imitate_episodes.py --task_name obstacle_gate_v1 --policy_class ACT \
    --ckpt_dir ckpts --kl_weight 10 --chunk_size 50 --hidden_dim 512 \
    --dim_feedforward 3200 --batch_size 8 --lr 1e-5 --seed 0 --num_epochs 2000 \
    --use_proximity --prox_feature raw --prox_layout per_sensor \
    --wandb_run_name pact_gate_raw_s0
# no --image_dropout_p on this arm — avoid-v1 dropout taxed the pick

for ARM in act_gate_s0 pact_gate_raw_s0; do
  for CELL in invisible free visible; do
    python eval_act_obstacle.py \
        --ckpt_dir ckpts/obstacle_gate_v1/<dated>_$ARM \
        --output_dir /home/jaydv/code/prox_learning/eval_output/${ARM}_${CELL} \
        --num_rollouts 50 --chunk_size 50 --temp_agg_off \
        --eval_cell $CELL --eval_sampler gate
  done
done
```

Metric: `bar_hit_rate` (contact with `protr_*`) next to blunt `collision_rate`. Pre-registered:
≥15 points on invisible-cell bar-hit, n=50, Fisher p < 0.05; free-cell similar; strict success
not worse.

Keep `num_workers <= 2` on datagen. `pkill -9 -f data_generation.main` between runs.

<a id="48-parked--test-time-camera-blur"></a>
### 4.8 Parked — test-time camera blur

Freeze a **finished** brain. Blur only RGB at eval (`--eval_blur_sigma`). Skin and qpos
untouched. Needs an obstacle ckpt; none on disk until gate-bar or a recollect.

```bash
cd /home/jaydv/code/prox_learning/submodules/act
export PYTHONPATH="$PWD:$PYTHONPATH" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl

python eval_act_obstacle.py \
    --ckpt_dir ckpts/<task>/<run> \
    --output_dir /home/jaydv/code/prox_learning/eval_output/<run>_visible_blur4 \
    --num_rollouts 50 --chunk_size 50 --temp_agg_off \
    --eval_cell visible --eval_blur_sigma 4
```

Pilot σ ∈ {0, 1, 2, 4} cheaply before committing n=50 everywhere. Train-time blur is a **different,
failed** experiment ([§4.12](#412-needs-datagen--train-time-blur)).

<a id="49-needs-datagen--headline-act-vs-pact-66--40"></a>
### 4.9 Needs datagen — headline ACT vs PACT (66% → 40%)

**Cannot run until you collect again.** `hybrid_obstacle_v1`, `hybrid_invis_obstacle_v1`, and
`obstacle_prox_v2` are gone. Published numbers live in `reports/eval_summaries/`.

To rebuild a hidden-bar set:

```bash
conda activate mlspaces
cd /home/jaydv/code/prox_learning/submodules/molmospaces
OMP_NUM_THREADS=2 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    python -m molmo_spaces.data_generation.main FrankaSkinHybridInvisObstacleConfig
# num_workers <= 2. Prefer gate-bar (§4.7) over repeating this v2 design.

cd /home/jaydv/code/prox_learning
python -m scripts.convert_obstacle_to_act \
    --src assets/datagen/hybrid_invis_obstacle_v1/FrankaSkinHybridInvisObstacleConfig/<timestamp> \
    --dst act_style_data/obstacle_prox_v2 \
    --with_proximity --image_h 240 --image_w 320

cd submodules/act
export PYTHONPATH="$PWD:$PYTHONPATH" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
# vanilla: no --use_proximity; PACT: --use_proximity --prox_feature raw
# published win used global mash (n_proximity_sensors=1, K=8), chunk 100
python eval_act_obstacle.py ... --num_rollouts 50 --chunk_size 100 --temp_agg_off --eval_cell invisible
```

<a id="410-needs-datagen--dataset-contents--dodge-probes"></a>
### 4.10 Needs datagen — dataset contents / dodge probes

Needs a datagen root with `obs_scene` labels (`behavior_class`, `protrusion_present`).

```bash
python scripts/analyze_obstacle_dataset.py \
    --root assets/datagen/<obstacle_run> \
    --out diagnostics_output/obstacle_analysis

python scripts/probe_prox_decodability.py   # defaults assume wiped hybrid_obstacle_v1
```

Historical: v1 deflection from skin = coin flip. v2: swerve ~0.75, bar-presence still chance.
`trunk` probe needs CVAE weights (deleted). Raw-skin probe does not.

<a id="411-needs-datagen--avoid-v1-already-failed-do-not-treat-as-headline"></a>
### 4.11 Needs datagen — avoid-v1 (already failed; do not treat as headline)

Filter scrapes, 3× upsample bows, min-pool skin, image dropout 0.3. Invisible collisions 40% vs
30% (p ≈ 0.40); success 42% vs 24%. Bows were learnable from vision.

```bash
python -m scripts.convert_obstacle_to_act \
    --src <visible_bar_datagen> \
    --dst act_style_data/obstacle_prox_avoid_v1 \
    --with_proximity --prox_pool min \
    --skip_approach_collision --keep_deflect_collisions --upsample_deflect 3 \
    --image_h 240 --image_w 320
# PACT train extras: --image_dropout_p 0.3 --prox_dropout_p 0.1
```

Do **not** drop every deflect graze — that leaves ~5 bows.

<a id="412-needs-datagen--train-time-blur"></a>
### 4.12 Needs datagen — train-time blur

Needs `act_style_data/obstacle_prox_v2`. Negative result: no behavioural ladder; ±40 points at
n=25. If you rerun, use n=50.

```bash
./train_blur_baseline.sh 2 4 8
./eval_blur_baseline.sh 50
```

<a id="413-needs-datagen--open-table-skin-necessity"></a>
### 4.13 Needs datagen — open-table skin necessity

```bash
python scripts/proximity_necessity.py \
    --glob 'assets/datagen/**/house_*/trajectories_batch_*.h5'
```

Historical: ~0% extra value when cameras already see the object. That is why the project moved
into enclosures.

<a id="414-sensor-qa-after-xml-edits"></a>
### 4.14 Sensor QA after XML edits

No argparse. Hardcoded `assets/robots/franka_skin/model_hybrid.xml`. Re-run after **any** sensor
change. Also check depth `znear` — at room scale the default silently drops readings closer than
~10–20 cm.

```bash
python scripts/verify_hybrid_skin_sensors.py
python scripts/build_hybrid_on_franka_skin.py    # rebuilds model_hybrid.xml
```

<a id="415-cluttered-bay-collect"></a>
### 4.15 Cluttered-bay collect

Loads the whole skin, then makes the arm travel (hood → cart). Detail: [§12.1](#121-cluttered-bay-pick-and-place).

```bash
cd submodules/molmospaces
MUJOCO_GL=egl python -m molmo_spaces.data_generation.main FrankaSkinHybridClutterPnPCheckConfig
MUJOCO_GL=egl python -m molmo_spaces.data_generation.main FrankaSkinHybridClutterPnPConfig
```

### 4.16 Do not rerun as if new

June camera-only chunk-size sweep, PACT v1 skin-blanking (~0.005 action change), trunk vs raw
grid. Numbers in [§13](#13-every-number-in-one-place). JSON in `reports/eval_summaries/`.

---

<a id="5-what-this-is"></a>
## 5. What this is

**The problem.** A robot arm reaches into a tight space. Cameras fail: self-occlusion, darkness,
a thin obstacle in a blind spot. A human stops looking and starts *feeling*.

**What we built.** 40 ToF-like distance sensors on the whole arm (forearm, upper arm, wrist).
Each is an 8×8 depth image, 45° cone. They answer "how close is something, right here?" — not
colour, texture, or shape.

**The question.** Does adding this skin make the robot *safer* than cameras alone? Fewer crashes.
Not smarter, not faster.

**The answer so far: yes, with an asterisk.** The skin helps a lot when it is the only sense that
can perceive the obstacle. It helps less when cameras can also see the obstacle, and not at all
when there is no obstacle.

**One surprise.** A small network that turns skin into a "flinch" joint motion works on its own.
Feeding that polished reflex into the policy did **nothing**. Feeding the 40 raw distances worked.
The dumb version won.

| Word | Meaning here |
|---|---|
| **policy** / **brain** | Network that looks at senses and decides how to move |
| **demonstration** | Recorded example from a scripted expert that can "cheat" (knows geometry) |
| **imitation learning** | Copy demonstrations. The brain uses a sense only if the demos *cannot be explained without it* |
| **ACT** | Off-the-shelf action-chunking transformer. Cameras + joints. Nothing about it is ours except the proximity token path |
| **vanilla** | Cameras only. The comparison point |
| **PACT-raw** | Same brain + 40 peak-closeness numbers. **The one that works** |
| **PACT-readout** | Same brain + 40 live 128-d CLS tokens; encoder finetuned with ACT. No number yet |
| **PACT-trunk** | Same brain + processed reflex embedding. Did nothing. Abandoned |
| **success rate** | Fraction that completed the task (lift or place) |
| **collision rate** | Fraction that touched something they should not. Lower is better |
| **strict success** | Task done **and** never touched anything |
| **n** | Attempts behind a number. Floor is 50 |
| **p-value** | Chance of a gap this big by luck. Below 0.05 is the usual bar |

**Why "free" matters.** With no bar, every recorded collision is the arm brushing the cavity.
Background brushing is ~60%. That is the instrument's floor.

---

<a id="6-headline-result"></a>
## 6. Headline result

**Adding raw skin readings cuts collisions from 66% to 40% where only the skin can sense the
hazard — and the robot is no worse at the task.**

Measured 2026-07-05, 50 attempts per cell. One seed, one dataset.

Collisions (lower is better):

| Brain | no bar (free) | bar hidden (invisible) | bar visible |
|---|---|---|---|
| cameras only (`vanilla`) | 60% | **66%** | 64% |
| **cameras + raw skin (`pact_raw`)** | 58% | **40%** | 50% |
| cameras + reflex signal (`pact_trunk`) | 64% | **72%** | 58% |

Task success (higher is better; diffs are noise):

| Brain | no bar | bar hidden | bar visible |
|---|---|---|---|
| cameras only | 22% | 36% | 28% |
| cameras + raw skin | 18% | 30% | 16% |
| cameras + reflex signal | 34% | 34% | 32% |

Fisher on invisible collisions 66% vs 40%: **p = 0.016**. Success diffs p ≥ 0.23. Strict success
(invisible): raw **20%** vs vanilla **14%**.

### Why we believe it

1. **Not luck.** p = 0.016.
2. **Not just timidity.** Free cell 60% → 58% (p = 1.0). Improvement appears where there is
   something to avoid.
3. **Graded with how much the skin is needed:** 2 / 14 / 26 points (free / visible / invisible).
4. **Vanilla does not avoid even when the bar is visible** (64% vs 66%). 105 demos never taught
   visual avoidance. The skin is not a backup sense here — it is the only avoidance signal in any
   trained policy.
5. **Raw skin predicts the expert 21% better** (val error 0.0595 vs 0.0755 vs 0.0830 for raw /
   cameras / trunk). Supporting, not the headline: training error does not reliably predict
   robot behaviour.

### Why it might be wrong

**The collision counter is blunt.** It counts *any* arm–environment contact except floor and
grasped cup. It cannot tell "rammed the bar" from "brushed the cavity wall." Background brushing
~60%; the bar itself adds only **~4–6 points**. A 26-point drop is larger than the bar increment,
so the skin is likely making the arm **generally more careful in the cavity**, strongest where the
skin fires. Say that honestly.

Per-body names are now recorded at eval (`hit_bar`, `bar_hit_rate`). That split cannot be
recovered from old runs. Gate-bar eval uses it.

**One training run.** One random seed, one dataset.

---

<a id="7-every-experiment-one-line"></a>
## 7. Every experiment, one line

**ACT** = cameras only. **PACT** = cameras + skin. Results `x% vs y%`.

| Experiment | Description | Results | Run |
|---|---|---|---|
| ACT vs PACT (headline) | 105 demos; test 50× with bar cameras cannot see | Crashes **66% vs 40%** hidden (p = 0.016); free 60% vs 58%; visible 64% vs 50%; pick 36% vs 30% noise | [§4.9](#49-needs-datagen--headline-act-vs-pact-66--40) |
| Do the skin sensors work? | Coverage, distance, blur/dark | 83% vs 10% of directions (skin vs wrist cam); 5.6 mm pipe error; skin bit-identical under blur/dark | [§4.14](#414-sensor-qa-after-xml-edits) |
| Skin on an open table? | Extra distance when cameras already see the object | ~0% extra value; project moved to tight boxes | [§4.13](#413-needs-datagen--open-table-skin-necessity) |
| Does the flinch network work? | Skin → "move away" joints, honest split | Direction 0.924 vs 0.926 (fair vs easy); closest-obstacle push only 64% of needed size | [§4.5](#45-live--reflex-retrain-and-demos) |
| Which flinch net is best? | Three trains, keep best | Error 0.009 vs 0.011 vs 0.015 (v3 / v1 / v2) | [§4.5](#45-live--reflex-retrain-and-demos) |
| How far ahead should cameras plan? | Short vs long chunk; less vs more practice | Crashes 30% vs 60% (chunk 50 vs 100); 60% vs 20% (2k vs 5k epochs). n=20 | archived |
| First try: add skin | v1 obstacle set, 50 attempts | 0% gain, p = 0.76 | archived |
| Does the brain use the skin? | Zero the skin, watch the planned move | Chunk moves ~0.005; brain ignored skin | archived |
| Can skin show a dodge? (set 1) | Swerve vs straight from skin only | Coin flip (0.50–0.56) | [§4.10](#410-needs-datagen--dataset-contents--dodge-probes) |
| Can skin show a dodge? (set 2) | Same on hidden-bar data | Swerve ~0.75; bar-presence still chance | [§4.10](#410-needs-datagen--dataset-contents--dodge-probes) |
| Skin help guess the next move? | Val error: raw vs cameras vs trunk | 0.0595 vs 0.0755 vs 0.0830; raw ~21% better | wandb / archived |
| Processed flinch instead of raw | Trunk token vs cameras | Hidden crashes 66% vs **72%**; trunk worse | archived |
| Train with blurry cameras | Train blurry, test sharp | Hidden success 36 / 0 / 40 / 24% (σ=0/2/4/8); ±40 pts at n=25; no pattern | [§4.12](#412-needs-datagen--train-time-blur) |
| What is in the training videos? | Swerves, scrapes, close skin | 43% of bar runs swerve; 40% scrape inbound; close skin 86% vs 74% | [§4.10](#410-needs-datagen--dataset-contents--dodge-probes) |
| Second try, cleaned data | Drop scrapes, 3× bows, hide cameras 30% | Hidden crashes 40% vs 30% (p = 0.40); pick 42% vs 24%. Failed | [§4.11](#411-needs-datagen--avoid-v1-already-failed-do-not-treat-as-headline) |
| Is the doorway pole in the way? | Short XML peg vs gripper path | Sideslip 1.6–7.2 cm; often misses; did not collect | [§4.7](#47-parked--gate-bar-v31) |
| Corridor skin fire | 20 cm vs 50 cm hits on hallway rows | 11% vs 40% of tiles; `link1_sensor_5` on 100% (self-view) | [§4.4](#44-live--corridor-skin-fire--compress-skin) |
| Compress the skin | 32-d embedding + XYZ | Validity 100%; XYZ 20.6 mm; pixel 87/95%. Compressor grade, not policy | [§4.4](#44-live--corridor-skin-fire--compress-skin) |
| Finetune readout into ACT | Unfreeze encoder; 128-d CLS tokens live at train/eval | **No policy number yet** | [§4.4](#44-live--corridor-skin-fire--compress-skin) |
| Hallway pick-and-place | 152 coauthor demos, n=50 | Place **28% vs 42%** (p = 0.21); bar **34% vs 36%** (p = 1.0). **Not a paper number** | [§4.3](#43-live--hallway-act-vs-pact) |
| Collect a taller doorway pole | 44 cm pole on TCP line | 0 examples collected | [§4.7](#47-parked--gate-bar-v31) |
| Blur cameras only at test time | Freeze policy, blur RGB, leave skin | 0 of these tests run | [§4.8](#48-parked--test-time-camera-blur) |

---

<a id="8-paper-claims"></a>
## 8. Paper claims

**One-sentence claim (the only headline).** A full-body proximity skin, fused as raw per-sensor
closeness into an ACT imitation policy (**PACT-raw**), cuts **collision rate** from **66% to 40%**
(n=50, Fisher p = 0.016) when a hazard bar is physically present but **hidden from the cameras**.
Lift-success is unchanged (noise). Safer when cameras cannot see the obstacle, not a better picker.

Tone: safer when cameras fail. Not SOTA pick-and-place. Avoid "safe" in the title unless you
provide formal guarantees.

**Setup.** Franka FR3 in MuJoCo. 40 sensors, 8×8 planar-z metres, 45° cones. Two RGB cameras
`exo_camera_1`, `wrist_camera` at 240×320 (hallway: wrist only). Policy = ACT (published grid:
chunk 100, hidden 512, FF 3200, KL 10). Task = fumehood cup pick; bar ~75% of training
(`OBSTACLE_P=0.75`). Eval: `eval_act_obstacle.py --temp_agg_off --eval_cell …`, n=50.

Copy-paste stats line:

> On a Franka FR3 in MuJoCo with 40 full-body proximity sensors, an ACT policy trained from 105
> scripted enclosure-pick demonstrations collides in 66% of rollouts (n=50) when a hazard bar is
> hidden from the cameras. Adding a raw-skin token (PACT-raw) cuts that to 40% (Fisher p = 0.016)
> without changing lift-success. A frozen Safety-CVAE retreat embedding (PACT-trunk) does not help
> (72%). The camera-only policy does not avoid the bar even when it is visible (64% vs 66%).
> Background contact with no bar is ~60%; the collision counter cannot yet separate bar hits from
> cavity brushes. One seed. Sim only.

### You may write now (2026-07-05 grid)

The tables in [§6](#6-headline-result). Graded benefit 2 / 14 / 26. Trunk is worse on the causal
cell. Do not sell the Safety-CVAE as the PACT encoder.

### Do not claim until a new `eval_summary.json` says otherwise

- Avoid-v1 / per-sensor / image-dropout: **measured 2026-08-24 and FAILED** (40% vs 30%, p ≈ 0.40;
  success down). Do not write "PACT cut collisions 40→30". Gate-bar may replace the headline only
  after its own summaries exist.
- Place-corridor local n=50: **do not cite.** No safety win. Do not replace 66→40. Coauthor
  "PACT beats ACT" on this hallway is not reproduced.
- Finetuned CLS-readout PACT: **no eval yet.** Do not cite. Commands in [§4.4](#44-live--corridor-skin-fire--compress-skin).
- Multi-seed. Real robot / hardware skin. Sim only.
- PACT uses a trained CVAE **encoder** at runtime. Runtime `z = 0`. The encoder `q(z | skin, dq)`
  is train-only. The arm that worked **bypasses** the CVAE: `--prox_feature raw`.
- Safety-CVAE is a skin autoencoder. It reconstructs 7-DoF `dq`, not 2560 pixels.
- ACT + residual `SafetyHead` at eval = a **different method** ("ACT+reflex"), not PACT.
- Train-time camera blur as evidence cameras fail. That sweep is **null**.
- Collision counter = "hit the bar." It is any contact. Say the 60% floor.
- v1 PACT "worked." Round 1 tied vanilla. Formal bar-presence probe **failed** on v2.
- Any `--temp_agg_off` number from **before 2026-07-04**. Invalid (arm froze ~30 cm short).
- Injecting crashes into behaviour cloning. Convert filters and upweights existing bows.

Limitations a reviewer will use: one seed; blunt metric; sim-only invisible cell (renderer
privilege); demos subtract a parked-obstacle baseline PACT cannot use; one skin frame vs a
100-step chunk; n=25 is noise; low collisions can mean broken; training loss does not predict
behaviour.

Related work: ACT is off-the-shelf. Position against wrist-only tactile, vision-only IL in
clutter, and CVAE safety filters that output joint retreat. Our CVAE *works as a reflex* and
*fails as a PACT feature*. That contrast is a result.

---

<a id="9-failed--retracted"></a>
## 9. Failed / retracted

**PACT v1 (2026-06-18).** Nothing beat cameras-only (p = 0.76). Blanking the entire skin moved the
predicted chunk by ~0.005. Causes: policies ignore the token, ambient saturation, temporal
aggregation washout, one skin frame vs a 100-step chunk.

**Probe gate.** v1: deflection not decodable (chance from trunk, raw, and qpos). v2: bar-presence
collapsed to chance (0.40–0.52). Deflection *became* decodable from raw (~0.75) and survived a
qpos control. Formal bar-presence gate still **failed**; training on v2 was a judgment call.

**Trunk arm (scrapped 2026-07-06).** Invisible collisions **72%**. CLI default used to be `trunk`;
it is now `raw`. Keep trunk as a negative control if you retrain CVAE weights.

**Train-time blur (2026-07-24 → 2026-08-10).** Three constant-blur vanilla arms, 225 rollouts.
Training error: 0.0755 / 0.0836 / 0.0948 / 0.1100 (σ = 0/2/4/8). Behaviour: no ladder. σ=2 ≈
statue (0% hidden success, fewest collisions). σ=8 free-cell collisions 28% vs 68% with a bar —
same policy, so that 40-point swing is **noise at n=25**. Keep the lessons; drop blur-as-camera-failure.

**Avoid-v1 (2026-08-24).** Invisible 40% vs 30% (p ≈ 0.40), lift 42% vs 24%. Cause verified in
sampler code: bar visible in training, cup coupled to bar side, bar pose barely varied. 3×
upsampled bows taught *vanilla* to bow. Motivated gate-bar.

**v3.0 gate pegs (2026-08-24 night).** XML pegs 20–24 cm; expert only added 1.6–7.2 cm. Task too
easy. Do not collect that. v3.1 is the fix.

**Hallway n=20 smoke.** 15% vs 35% place, 30% vs 20% bar. Luck. n=50 killed the bar story.

---

<a id="10-method"></a>
## 10. Method

**PACT** = ACT + proximity tokens in transformer memory:
`[latent z, qpos, prox tokens, ~160 image tokens]`.

**Published win** (use unless a later eval beats it): `--prox_feature raw`, **global** mash
(40 sensors → one 40-d vector → 8 anonymous tokens), `n_proximity_sensors = 1`, chunk 100.

**Current defaults** (not the published win): `--prox_feature raw --prox_layout per_sensor`
(40 tokens of dim 1; K clamped 8→1). Avoid-v1 used this plus image dropout 0.3 and **failed**.
Gate-bar uses per-sensor, chunk 50, **no** image dropout.

```
h5 /observations/proximity     (T, 40, 8, 8) metres
        │  one random frame per ACT sample
        ▼
PeakClosenessEncoder.featurize  closeness, once
        │  raw   → (B, 1, 40) global  or  (B, 40, 1) per_sensor
        │  trunk → (B, 1, 256)   needs cvae weights, negative control
        │  delta → (B, 1, 7)     needs cvae weights, negative control
        ▼
Linear(feat → K·hidden) → K tokens
        ▼
encoder memory                 [z, qpos, prox, ~160 image]
```

Audited clean: metres stay metres; featurize once; `dataset_stats` never z-scores skin; convert
and live eval both stack by `HYBRID_SKIN_SENSOR_ORDER` (`link5_back` before `link5_front`). Never
use the env's `_HYBRID_SKIN_SENSOR_NAMES` tuple.

Closeness: \(c = \mathrm{clip}(1 - d / 0.5,\, 0,\, 1)\); dead \(d < 5\,\mathrm{mm} \to 0\).

With proximity off, the model is bit-identical to vanilla ACT. Training writes `prox_config.json`;
eval auto-detects it. Only `input_proj_proximity` and extra positional-embedding rows train when
proximity is on.

### Why success stays flat (method vs data, not a shape bug)

1. **Wrong metric.** BC copies demos that succeed *with the bar present*. Report **invisible-cell
   collisions**, not lift-success.
2. **`trunk` is a retreat prior orthogonal to BC.** It *raised* collisions 66%→72%.
3. **40 sensors mashed to 1 vector** in the published raw run (anonymous tokens).
4. **One skin frame, 100-step chunk.** Uniform L1. Late actions barely depend on t=0 closeness.
5. **`--image_dropout_p` default 0.** ~160 image tokens vs 8 prox. Vision can fit the demos alone.
6. **Mean-pool of 4 substeps** dilutes a 1-substep graze 4×. Convert `--prox_pool min` keeps it.

Do **not** freeze the Safety-CVAE as the policy encoder for the headline method.

### Safety-CVAE (reflex, not PACT)

Not a skin autoencoder. Reconstructs a **7-DoF joint retreat** `dq`. Skin is the *condition*.
Latent `z` is a train-time dodge knob, then pinned to `0`. The CVAE encoder `q(z | skin, dq)`
**cannot run while acting** — it needs the target retreat.

| | Job A — reflex head | Job B — PACT wrapper |
|---|---|---|
| Who | `scripts/safety_*_demo.py` via `SafetyHead` | `encoders/peak_closeness.py` (`prox_cvae.ProxCVAEEncoder` shim) |
| In | `(40, 8, 8)` metres | `(B, 40, 8, 8)` metres |
| Out | `(7,)` rad | feature → K ACT tokens |
| Encoder run at policy time? | **no** (`z = 0`) | **no** |

```mermaid
flowchart LR
  subgraph infer ["Runtime: CVAE encoder is dead, z = 0"]
    X2["x closeness 2560"]
    Z0["z = 0"]
    Dec2["decoder MLP"]
    T["trunk 256"]
    D["dq 7 rad"]
    X2 --> Dec2
    Z0 --> Dec2
    Dec2 --> T
    Dec2 --> D
  end
```

Teacher: for every sensor with an environment return \(r < D_{\mathrm{act}}=0.18\,\mathrm{m}\),
push away in Cartesian, map through the translational Jacobian, sum over 40 sensors. Self-hits
excluded from the **label**, kept in the **input**. Labels scaled to unit RMS on the close subset
(\(\sigma \approx 11.359\) for v3). Loss = recon MSE + \(\beta\) KL, \(\beta_{\max}=10^{-2}\),
10-epoch warm-up. v3 ended with **1 of 8** latent dims alive — the "C" is mostly a regulariser.

Honest split (pose-grouped): direction 0.924. Magnitude 69% worse than the cheating split. When
the obstacle is closest, output is 64% of the needed size. A nearest-neighbour table ties
direction (0.923). Do not claim fine spatial resolution.

### Skin encoders (`encoders/`)

Two front-ends, named by **job**. Same live tensor `(B, 40, 8, 8)` metres. Run from repo root.

| name | file | job | out | weights |
|---|---|---|---|---|
| `peak_closeness` | `encoders/peak_closeness.py` | per-sensor peak closeness, 50 cm cap | `(B, 40, 1)` in `[0, 1]` | none (headline PACT-raw) |
| `cvae_trunk` / `cvae_delta` | same | frozen Safety-CVAE retreat taps | `(B, 1, 256)` / `(B, 1, 7)` | `model.pt` (deleted) |
| `nearest_surface` | `encoders/surface_geometry.py` | nearest in-range XYZ, 20 cm cap | `(B, 40, 3)` metres | `pact_surface_encoder_v1` (frozen default) |
| `surface_embedding` | same | frozen 32-d embedding **or** 128-d CLS readout | `(B, 40, 32)` / `(B, 40, 128)` | pretrained `pact_surface_embedding_encoder_v1`; finetune writes `prox_encoder_best.pt` |

The **live geometry arm** skips bake. `--finetune_prox_encoder` loads the pretrained stem,
unfreezes it, and injects the CLS hidden state (`encoded[:, 0]`, 128-d) as `proximity_positions`
`(B, 40, 128)`. Auxiliary 32-d embedding / XYZ / recon heads stay on the net for the compressor
trainer; ACT does not use them. Eval loads `prox_encoder_best.pt` from the ACT run dir and runs
the same readout.

```python
from encoders import load_encoder

prox = ...  # (B, 40, 8, 8) metres
raw = load_encoder("peak_closeness")
feat = raw.policy_features(prox)   # (B, 40, 1)

live = load_encoder(
    "surface_embedding",
    checkpoint="…/pact_surface_embedding_encoder_v1.pt",
    frozen=False,
    policy_tap="readout",
)
tok = live.policy_features(prox)   # (B, 40, 128) CLS readout; grads on
```

Aliases: `raw` → `peak_closeness`, `xyz` → `nearest_surface`, `embedding` → `surface_embedding`.
Without a geometry checkpoint the conv-transformer is random (shapes still work).
`submodules/act/prox_cvae.py` is a shim to `encoders/peak_closeness.py`.

Bake frozen 32-d tokens (ablation only, after the split-manifest wire):

```bash
python -m encoders.encode_tokens \
    --dataset-dir act_style_data/pact_place_corridor_v5 \
    --checkpoint experiments_output/default/surface_encoder_train/pact_place_corridor_v5/pact_surface_embedding_encoder_v1.pt \
    --kind embedding
```

### Dataset facts (historical obstacle source)

From `analyze_obstacle_dataset.py` on wiped `hybrid_obstacle_v1` (151 episodes):

| fact | value |
|---|---|
| Bar present | 75% (113 / 151) |
| `behavior_class` | 49 deflect / 102 free — only **43% of bar episodes actually bow** |
| Lateral bow, bar-deflect / free | mean **3.8 cm** / **0.5 cm** |
| Skin close (`<0.10 m`), bar / no-bar | **86%** / **74%** (ambient saturation) |
| Approach (arm-vs-env) collision | **40% of episodes** |
| Approach contacts, bar-deflect | mean **5.0** |

Not "clean avoidance plus a hidden bar." Convert `--skip_approach_collision
--keep_deflect_collisions --upsample_deflect 3` tries to stop teaching scrapes without deleting
the bows.

Published PACT train set: **105** demos (47 visible / 49 hidden / 29 none).

### Training / eval flags

| flag | default | what |
|---|---|---|
| `--use_proximity` | off | turns on PACT |
| `--prox_feature` | `raw` | `raw` / `trunk` / `delta` / geometry names |
| `--finetune_prox_encoder` | off | unfreeze geometry stem; CLS readout; live `raw_causal` |
| `--prox_policy_tap` | (implied) | `embedding` (frozen 32-d) / `readout` (128-d CLS) / `xyz` |
| `--prox_encoder_lr` | same as `--lr` | encoder param group when finetuning |
| `--prox_layout` | `per_sensor` | `per_sensor` = 40 named tokens; `global` = published mash |
| `--prox_tokens_per_sensor` | 8 | K; `per_sensor` clamps 8→1 |
| `--chunk_size` | — | 100 on published grid; **50** on hallway and gate-bar |
| `--image_dropout_p` | 0 | per-sample hard vision dropout |
| `--blur_sigma0` / `--blur_mode` | 0 / `curriculum` | train-time camera blur |
| `--temp_agg_off` | off | **required** at eval after 2026-07-04 |
| `--eval_cell` | none | `visible` / `invisible` / `free` |
| `--eval_sampler` | `invis` | `gate` for gate-bar ckpts |
| `--eval_blur_sigma` | 0 | blur cameras at **inference**; skin untouched |
| `--num_rollouts` | 25 | use **50** |
| `--no_wandb` | off | wandb is on by default |

Blur and dropout are training-only. Validation and eval see sharp frames unless you pass
`--eval_blur_sigma`.

`detr/main.py` must re-declare every fork flag as a no-op: `build_ACT_model_and_optimizer`
re-parses `sys.argv`. Eval is exempt (it shields that parser).

---

<a id="11-repo-map"></a>
## 11. Repo map

```
README.md            this file — science, claims, run cookbook
CLAUDE.md            agent working agreement
CURSOR.md            session change log (not a result)
pyproject.toml       not installed; see §3

scripts/             analysis / training / figures / housekeeping.sh
encoders/            peak closeness + surface geometry
tests/               encoder + PACT-raw unit tests
configs/             frozen eval manifests (Amine 40-row place protocol)
submodules/act/      ACT fork — train and eval
submodules/molmospaces/  simulator + demonstration collection
submodules/MolmoBot/ unused

assets/              MolmoSpaces asset root AND this project's artifacts
  robots/franka_skin/model_hybrid.xml    40-sensor arm — canonical model
  safety/            sweep_v*.h5 + leftover demo mp4/mcap (weights deleted)
  datagen/           check runs only (obstacle sources wiped 2026-08-24)
franka_assets/       mesh store via symlinks — DO NOT DELETE
data/pact_place_corridor_v5/   coauthor hallway rows (live)
act_style_data/pact_place_corridor_v5/   converted ACT hdf5 (live)
eval_output/         gitignored rollouts
experiments_output/  encoder trains, viz, figures
diagnostics_output/  committed legacy renders
reports/2026-08-14/  weekly report with PNGs
reports/eval_summaries/  archived eval_summary.json — only durable published numbers
train_blur_baseline.sh / eval_blur_baseline.sh
```

**`franka_assets/` has zero textual references.** It is reached only through
`assets/robots/franka_skin/{assets,skin_meshes,robotiq_2f85_v4}` →
`../../../franka_assets/fr3_skin/*`. Grep will call it dead. Deleting it breaks the 40-sensor
model.

Canonical model: `assets/robots/franka_skin/model_hybrid.xml` (40 sensors, ncam 42, nq 13).
Older `model.xml` is the 29-sensor skin. `model.xml.bak_before_orientation_fix` is provenance.
`assets/mjcf/fr3_skin.xml` fails to compile (deleted `.mesh_cache`).

### Scripts (active)

| file | job |
|---|---|
| `safety_sweep.py` | 40 SPAD depths + potential-field labels → `sweep.h5` |
| `train_safety_cvae.py` | skin → 7-DoF retreat CVAE |
| `safety_{flinch,react,moving,orbit,sphere}_demo.py` | reflex videos |
| `figures.py` | 29 paper figures (`--list`) |
| `convert_obstacle_to_act.py` | datagen → ACT hdf5 |
| `convert_pact_place_to_act.py` | hallway rows → ACT hdf5 |
| `probe_prox_decodability.py` | swerve linear probe |
| `compare_pact.py` | Wilson CI + Fisher |
| `analyze_obstacle_dataset.py` | bar / deflect / scrape stats |
| `proximity_necessity.py` | vision-vs-skin coverage |
| `verify_hybrid_skin_sensors.py` | per-sensor QA |
| `build_hybrid_on_franka_skin.py` | builds `model_hybrid.xml` |
| `housekeeping.sh` | tiered disk cleanup, dry-run default |
| `dataset_viz.py` | folder of h5 → one MCAP + tiled MP4 + HTML (ACT / HF / datagen) |
| `foxglove_viz.py` | datagen h5 → `.mcap` (older; prefer `dataset_viz.py`) |
| `hybrid_viz_lib.py` | shared MuJoCo/EGL helpers |
| `run_pact_place_eval_chunk100.py` | Amine 40-row place protocol on local ACT/PACT ckpts |
| `pact_place_eval_chunk100_contract.py` | frozen 40-row scene hashes (vendored) |

Visualizer: `scripts/dataset_viz.py` for a folder of h5 (MCAP + HTML). Scene inspect:
`submodules/molmospaces/scripts/datagen/visualize_environment.py`.

ARCHIVE (era over, still imported or historic): `test_and_reconstruct_hybrid.py` (library for
`figures.py` — do not delete), `analyze_dataset.py` / `dataset_probes.py` (29-sensor),
`convert_hybrid_skin_urdf.py`, `build_photoshoot_skin.py`.

### ACT fork

Upstream ends at `742c753`. This project adds proximity fusion, in-env eval, blur / dropout.
`imitate_episodes.py` trains. `eval_act_obstacle.py` and `eval_act_place_corridor.py` evaluate.
`--manifest` on the place eval runs Amine's frozen 40-row protocol ([§4.3.1](#431-live--amine-40-row-place-protocol)).
`constants.py` `TASK_CONFIGS`:

| task | dataset | eps / len | on disk? |
|---|---|---|---|
| `obstacle_pact_v2` | `obstacle_prox_v2` | 105 / 185 | **no** (wiped) |
| `obstacle_pact_avoid_v1` | `obstacle_prox_avoid_v1` | 151 / 140 | **no** |
| `obstacle_gate_v1` | `obstacle_gate_v1` | 0 / 0 until convert | **no** |
| `pact_place_corridor_v5` | `pact_place_corridor_v5` | 152 / 636 | **yes** |

`SIM_TASK_CONFIGS` (ALOHA) is dead, kept because three upstream files import the name.

Both cameras share **one** ResNet18 (`# HARDCODED`). Upstream behaviour.

### MolmoSpaces

You need `data_generation/`, `tasks/` (`enclosure_reach.py`, `pick_task.py`, `fumehood_clutter.py`),
`env/`, `configs/`. Ignore isaac / maniskill / housegen / planner.

Datagen: `python -m molmo_spaces.data_generation.main <ConfigName>`. One positional argument.
No CLI field overrides — edit the class. Output:
`<output_dir>/<ConfigName>/<YYYYmmdd_HHMMSS>/`. Work is **one house per worker**.
`setup_house_dirs` skips a house whose h5 exists (resume **and** silent no-op if complete).

Live hybrid configs (all 40-sensor):

| config | what |
|---|---|
| `FrankaSkinHybridObstacleConfig` | old main ACT set (source wiped) |
| `FrankaSkinHybridInvisObstacleConfig` | v2 hidden-bar (source wiped) |
| `FrankaSkinHybridGateBarVisibleCheckConfig` | pole **rendered** preflight |
| `FrankaSkinHybridGateBarCheckConfig` | invisible preflight |
| **`FrankaSkinHybridGateBarConfig`** | **v3.1 headline collect** |
| `FrankaSkinHybridClutterPnPCheckConfig` / `…PnPConfig` | cluttered bay |

`viz_sensor_rgb` is on for the hybrid chain: 40 extra 256×256 RGB renders per step, **nothing
in the hdf5**, ~3 GB/episode. Eval forces it off. Collection configs for gate-bar and clutter
force it off. Preflights may leave it on — `num_workers <= 2`.

Sensor pipeline: 4 sub-steps per policy step (`(T, 4, 8, 8)` native). Dedicated 8×8 renderer.
`geomgroup[2]=0` (hide cosmetic skin); `geomgroup[4]=1` (invisible-bar trick). Zero-padding would
read as contact at 0 m, so the buffer left-pads by repeating the earliest frame.

Expert `ObstacleAwarePickPlannerPolicy` reads `protr_center` / `protr_half` — **never pixels** —
and stamps `behavior_class ∈ {deflect, free, abort}`.

### Data formats

Datagen hdf5: one file per house, groups `traj_*`. Skin at `obs/proximity/<sensor_name>`
`(T, 4, 8, 8)` metres. **No RGB in the hdf5** — images are sibling MP4s. `uint8 (T, 2000)`
datasets are zero-padded UTF-8 JSON. Labels live in `obs_scene`: `behavior_class`,
`protrusion_present`, `bar_invisible`, `collision_metrics`. **`scene_params["cell"]` is not a
label** on obstacle runs (always `"bar"`).

ACT hdf5: `episode_<g>.hdf5` with `/action (T, 8)`, `/observations/qpos (T, 9)`, images 240×320,
optional `/observations/proximity (T, 40, 8, 8)` raw metres, stacked in
`HYBRID_SKIN_SENSOR_ORDER`. Hallway convert writes wrist only. Dataloader reads **one random
frame** per sample. Dump a whole folder to one video / MCAP with `scripts/dataset_viz.py`
([§4.2.1](#421-live--visualize-a-dataset-folder)).

---

<a id="12-recipe-detail"></a>
## 12. Recipe detail

<a id="121-cluttered-bay-pick-and-place"></a>
### 12.1 Cluttered-bay pick-and-place

The obstacle line loads a handful of wrist sensors for a second. This task loads all six links,
then makes the arm *travel*: cup from the hood onto a rolling cart, sweeping a cluttered bay.

Scene: `fumehood_clutter.xml` — shelf left, cabinet rear, cart right, 16 mocap clutter items
re-posed per episode. Clutter is rejection-sampled off the expert path (`KEEPOUT` volumes + TCP
polyline). Expert never touches clutter; a policy that ignores the skin drifts and hits
something.

**Reach constraint.** Cart `(0.32, -0.56)` and `TRANSPORT_Z = 0.78` keep the wrist inside the
FR3's 0.855 m envelope (shoulder at `(0.08, 0, 0.35)`). A first draft at `(0.30, -0.62)`, z=1.00
was 0.925 m out — **every** episode died with `IK failed`. Change cart pose in **two** places:
the XML and `CART_XY` in `tasks/fumehood_clutter.py`.

**Segment names** must be `gripper-open / pregrasp / grasp / gripper-close / lift / preplace /
place / retreat / go_home`. Anything else writes `-1` into `obs/policy_phase`.

Success is scored at the **destination** (`PLACE_TOL = 0.12 m`), not at lift.
`[ObstacleDiag] success=False` next to `completed with success=True` is correct: inherited
diag still wants "lifted in gripper."

Check `[ClutterPnP] contacts by body:`. Any `clut_*` is `*** CLUTTER TOUCHED ***`. Sash grazes
are pre-existing hood behaviour.

Knobs on the sampler: `OBSTACLE_P` (0.60), `INVIS_P` (0.5), `N_CLUTTER` (9–15), `CLUT_MIN_GAP`,
`PLACE_TOL`.

Commands: [§4.15](#415-cluttered-bay-collect).

<a id="122-gate-bar-v31"></a>
### 12.2 Gate-bar v3.1

**Why.** Avoid-v1 failed because vision could explain the bows: cup sat on the bar's side;
bar face 0.14–0.24 m off-center so one "always bow" path cleared every bar.

**v3.1** (`GateObstacleFumehoodPickSampler`):

- `GATE_HALF_Z=0.22` (44 cm). `gate_block` snaps the pole's inner face onto the home→pregrasp TCP
  at t=0.40. Straight gripper envelope intersects the geom. Expert bow ≈ **18 cm**.
- Bow **sign** = `protr_wall` coin-flip. Cameras see the cup (where the line is), not which way
  is open. Cup y independent (`±U(0.08, 0.14)`).
- Collect `INVIS_P = 1.0`. Wide jambs `ap_w ∈ 0.66–0.85`.

100% of bar episodes deflect. Blind straight hits the pole. Fixed-side bow hits the other half.

Commands and pass criteria: [§4.7](#47-parked--gate-bar-v31). `--chunk_size 50` so the policy
re-reads skin mid-approach (chunk 100 gives ~2 blind looks; June sweep: chunk 50 halves
collisions).

---

<a id="13-every-number-in-one-place"></a>
## 13. Every number in one place

| thing | value | source |
|---|---|---|
| Skin sensors | 40 × 8×8 depth, 45° cone | `model_hybrid.xml` |
| Directional coverage | 83% vs 10% wrist camera | sensor proofs, 06-11 |
| Skin accuracy | linear; 5.6 mm pipe reconstruction | sensor proofs |
| Skin under blur / darkness | bit-for-bit identical; cameras collapse | sensor proofs |
| Reflex direction (honest split) | 0.924 | audit, 06-13 |
| Reflex magnitude (honest split) | 69% worse than cheating split | audit |
| Reflex when obstacle closest | 64% of needed size | audit |
| Lookup-table baseline | ties direction 0.923 | audit |
| Camera-only June sweep (n=20) | 40% success / 60% collisions (chunk 100, 2k ep) | June |
| First skin attempt v1, n=50 | nothing beats cameras, p = 0.76 | 06-18 |
| Deflection from skin v1 / v2 | chance / ~0.75 | probes |
| Bar presence from skin v2 | chance. Formal gate FAILED | probes |
| Dataset v2 | 105 demos (47 vis / 49 hid / 29 none) | wiped `obstacle_prox_v2` |
| Val error raw / cameras / trunk | **0.0595** / 0.0755 / 0.0830 | wandb |
| Vanilla collisions free / hid / vis | 60% / **66%** / 64% | 07-05, n=50 |
| Vanilla success | 22% / 36% / 28% | 07-05 |
| **PACT-raw collisions** | 58% / **40%** / 50% | 07-05 |
| PACT-raw success | 18% / 30% / 16% | 07-05 |
| Trunk collisions | 64% / **72%** / 58% | 07-05 |
| **Headline** | **26 points, Fisher p = 0.016** | 07-05 |
| Strict success, invisible | raw 20% vs vanilla 14% | 07-05 |
| Background brushing (no bar) | ~60% | 07-05 |
| Bar's own add to collisions | ~4–6 points | 07-05 |
| Blur train-error σ=0/2/4/8 | 0.0755 / 0.0836 / 0.0948 / 0.1100 | 07-24 |
| Blur robot behaviour | no pattern | 08-10 |
| **Noise at 25 rollouts** | **±40 points** | blur grid |
| Blur saturation | σ=2 removes ~98% fine detail | preview |
| Obstacle eval throughput | ~3.56 min/rollout | blur grid |
| Obstacle eval memory | 8 GB + 0.5 GB/rollout | measured |
| Avoid-v1 invisible coll / succ | 40% vs 30% (p≈0.40) / 42% vs 24% | 08-24 **failed** |
| Place-corridor n=50 ACT vs PACT | place **28% vs 42%**; bar **34% vs 36%**; p = 0.21 / 1.0 | 08-27 **not paper** |
| Place-corridor eval time | ACT 119.5 s / 2 eps. PACT-raw **2121 s / 2 eps** with `renders=19 skip=883` (EGL). Default eval skin is now `mj_multiRay`. `--egl-prox` restores the 18 min/ep rasterizer | smoke 2026-08-29 |
| Surface encoder test XYZ | 20.6 mm; validity 100%; pixel 87.4 / 95.3% | 08-25 |
| Corridor 20 cm / 50 cm tile hit | 11% / 40%; `link1_sensor_5` 100% at 20 cm | probe |

n=20 hallway smoke (15% vs 35% place, 30% vs 20% bar) is superseded luck, not a result row.

---

<a id="14-traps"></a>
## 14. Traps

Every one of these has already cost real time.

1. **`--temp_agg_off` was broken until 2026-07-04.** Default temporal agg (`m=0.01`): newest chunk
   (the only one that saw current skin) carries ~1.6% of the executed action. The *original*
   `--temp_agg_off` re-queried every step and executed only `chunk[0]` (nearly a copy of current
   qpos). Arm creeps, freezes ~30 cm short, 0 successes with *low* collisions. Fixed to open-loop
   chunking. **Any `--temp_agg_off` number from before 2026-07-04 is invalid.**
2. **`viz_sensor_rgb` OOM.** ~3 GB/episode of cosmetic mosaics the policy never reads. Eval forces
   it off. Datagen: `num_workers <= 2`.
3. **Sensor order.** `hybrid_skin_sensors.HYBRID_SKIN_SENSOR_ORDER` is law. `link5_back` before
   `link5_front`.
4. **Baseline subtraction is sim-only.** Demos subtract a parked-obstacle rest reading. PACT
   cannot. In 100% of demo frames some fixture is within `D_MAX = 0.5 m`; 40–60% of steps sit
   inside `D_ACT = 0.18 m` while the action is "keep going." `corr(‖delta‖, min_depth) ≈ −0.7` on
   successful demos. BC learns to ignore retreat.
5. **Low collisions can mean broken**, not careful (blur σ=2). Report strict success.
6. **Training error does not predict behaviour.** Blur ladder vs null robot. Never ship a
   behavioural claim from wandb alone.
7. **`scene_params["cell"]` is not a label** on obstacle runs (always `"bar"`).
8. **Datagen resume is silent.** Existing h5 → skip. Re-running a complete dir does nothing.
9. **`imitate_episodes.py --eval` cannot evaluate this project.** Use `eval_act_obstacle.py` or
   `eval_act_place_corridor.py`.
10. **`franka_assets/` looks dead to grep and is not.**
11. **n = 25 is inside the noise band.** ±40 points. 50 is the floor. Match sample sizes.
12. **Any new waypoint vs 0.855 m reach.** `‖p − (0.08, 0, 0.35)‖` before trusting a pose.
13. **Planner segment names are a closed vocabulary.** Unknown names write `-1` into
    `obs/policy_phase`.
14. **Deleting `assets/.lmdb` costs ~10 minutes** on the next datagen run (looks hung).
15. **Killing datagen can leave the MuJoCo worker alive** (10–12 GB). Next run slows then exits
    0 with no h5. `pkill -9 -f data_generation.main`.
16. **Two closeness maps.** Peak-closeness 50 cm + dead `<5 mm`. Surface geometry 20 cm, farther
    = **invalid**, not a regression target. Never mix `featurize_*` and `depth_to_closeness` on
    the same tensor.
17. **Two argument parsers.** New train flags must also be no-ops in `detr/main.py`.
18. **Baked 32-d tokens skip the encoder.** `proximity_layout=embeddings` sets the train-time
    encoder to `None`. `--finetune_prox_encoder` forces `raw_causal` and keeps the net. Eval of a
    finetuned run must load `prox_encoder_best.pt`, not the pretrain file.
19. **Depth minimum range** scales with scene size. At room scale the default deletes the whole
    band this project studies. Set it small and re-check datasets.
20. **Hiding something from cameras hides it from depth too** if you use transparency or the
    wrong display layer. Invisible-to-cameras, visible-to-skin needs separate geom-group masks —
    the hidden-bar mechanism.
21. **Driver mismatch is silent until GPU use.** Preflight `torch.cuda.is_available()`; reboot if
    false. A 2026-07-28 overnight run failed every attempt.
22. **Test the summary step of a long script** before the long part. Blur grid ran 13 h then
    printed an empty table (timestamp prefix vs glob).
23. **Place-corridor eval with videos on OOMs** near ~30/50 (~2 GB/ep). Metrics-only is the path.
24. **Amine `eval_pact_place_chunk100_row.py` will not load local hallway ckpts.** That worker
    wants chunk 100, frozen 32-d embeddings, `run_manifest.json`, and hashed encoder paths.
    Local models are chunk 50 (raw K=8 dim 1, or readout 128-d). Use
    `scripts/run_pact_place_eval_chunk100.py` ([§4.3.1](#431-live--amine-40-row-place-protocol)).
    `PACT_PERMUTED` needs his `(40,900,40,32)` token plan; it is not on this disk. One GPU:
    do not pass `--workers 10`.
25. **PACT 12–15 h is 40 EGL `update_scene`, even gated.** Smoke 2026-08-29: PACT-raw
    `renders=19 skip=883` and still **2121 s / 2 eps**. Default skin is `mj_multiRay`.
    `--egl-prox` is the slow rasterizer. Kill any in-flight EGL readout.
26. **v10 hallway RGB names are not v5 names.** `data/pact_place_corridor/data/v1010`
    writes `episode_{sha256}_wrist_camera.mp4` and `episode_{sha256}_table_camera.mp4`.
    Lundii v5 writes `episode_00000000_wrist_camera.mp4` and has **no table**. A viz
    glob that only knew `episode_{idx:08d}_exo_camera_1.mp4` reported `no wrist RGB`
    / `no table RGB` even though both files sat next to `trajectory.h5`. Re-run
    `scripts/dataset_viz.py --force` after the glob fix. `obs/sensor_param` on this
    dump still has wrist only — `--cam3d` projects onto wrist, not table.
27. **Root `dataset_viz/index.html` is not a video wall.** The old audit page
    put a `<video preload>` in every row and the browser fetched every clip.
    The dashboard fetches `audit.json` plus one `timeline.json` and one mp4.
    `file://` cannot `fetch` those JSON files — use
    `python scripts/dataset_viz.py --serve`. `--dashboard` is the no-encode
    catalog rebuild when you are not serving.

---

<a id="15-decision-log"></a>
## 15. Decision log

- **Pick-and-place → enclosures (June).** `proximity_necessity.py`: under a visibility constraint
  the skin adds ~nothing. Pivoted to enclosure/obstacle scenes.
- **29 → 40 sensors.** Hybrid gentact skin (`model_hybrid.xml`). Anything citing `LINKS = 2/3/5/6`
  is pre-pivot.
- **Safety-CVAE v1 → v3.** Retrain on obstacle-only `sweep_v3.h5`: val MSE 0.009, close-cos 0.926.
  Weights deleted 2026-08-24; PACT-raw never used them.
- **PACT round 1 tied vanilla (2026-06-18).** Plumbing audited correct. Four causes: ignore token,
  ambient saturation, temp-agg washout, one snapshot vs 100-step chunk.
- **Probe gate v1 FAIL (2026-07-03).** Deflect-vs-free chance from trunk, raw, **and qpos**.
  Planner holds the bar at wall clearance.
- **v2 invisible-bar collection (2026-07-03).** 3 of 4 workers OOM; 105 usable of 200. Object
  placement decoupled from bar (removed a leak).
- **v2 probes inverted the picture.** Bar-presence → chance (v1's 0.72–0.78 was the leak).
  Deflection became decodable from raw. Formal gate still failed. Training was a judgment call.
- **v2 results (2026-07-05).** Headline. `pact_raw` 66% → 40% (p = 0.016). Free cell flat. Vanilla
  does not improve when the bar is visible.
- **`--temp_agg_off` fixed (2026-07-04).** Pre-fix numbers invalid.
- **Trunk scrapped (2026-07-06).** Inert or worse.
- **Blur sweep (2026-07-24 → 2026-08-10):** no usable pattern. Established ±40-point noise at n=25
  and "low collisions can mean broken."
- **Collision-aware convert + per-sensor layout (2026-08-23).** Defaults `raw` + `per_sensor`.
  Hard-stop `imitate_episodes.py --eval` with proximity. Paper number stays invisible collisions.
- **avoid-v1 FAILED (2026-08-24).** 40% vs 30% (p ≈ 0.40), success down. Vision-predictable bows.
- **Gate-bar v3.1 (2026-08-24).** Tall pole on TCP line. Visible check first. Collect still parked.
- **User wipe (2026-08-24).** `assets/datagen/` obstacle runs, `ckpts/`, `eval_output/` cleared.
  July grid not reproducible from this disk.
- **Place-corridor local n=50 (2026-08-27).** 28% vs 42% place, 34% vs 36% bar. No safety win.
  **Not a paper number.** Do not replace 66→40.
- **Readout finetune (2026-08-28).** Drop frozen 32-d bake for the live geometry arm. ACT consumes
  128-d CLS tokens; encoder trains with BC. No eval yet. Not a claim.
- **Amine 40-row place protocol (2026-08-29).** Reuse his frozen scenes, not his policy loader.
  Local launcher `scripts/run_pact_place_eval_chunk100.py`. `PACT_PERMUTED` skipped. Not a
  paper number. Do not mix with random-house n=50.

### Unresolved

- Whether the main result holds with a different seed.
- How much of any old collision number is actually the hazard (60% wall floor). Gate-bar eval
  now has `bar_hit_rate`; that grid has not been run.
- Whether the policy attends to the skin (inferred, never measured). Instrument on
  `origin/encoder_eval`, never merged.
- Why the skin arm collides *less* with a hidden bar (40%) than with no bar (58%).
- Whether 105 demonstrations is the binding limit. The 200-episode version was never collected.
- Whether this task needs sharp vision at all (blur-4 training did not hurt success).

---

<a id="16-housekeeping"></a>
## 16. Housekeeping

### What is on disk now (2026-08-27)

| path | ~size | what |
|---|---|---|
| `data/pact_place_corridor_v5` | 11 G | coauthor hallway rows — **keep** |
| `submodules/act/ckpts` | 15 G | includes corridor vanilla + PACT-raw |
| `act_style_data/pact_place_corridor_v5` | 4.4 G | converted ACT hdf5 — **keep** |
| `assets/safety` | 342 M | `sweep_v*.h5` + demo mp4/mcap; **no `cvae_v3/`** |
| `assets/datagen` | 39 M | `hybrid_gate_bar_check`, `hybrid_clutter_pnp_check` only |
| `eval_output` | small | hallway n=50 summaries; gitignored |
| `reports/` | 2.6 M | weekly report + archived `eval_summary.json` |
| `.git` | tens of GB | history still carries deleted `.h5` blobs |

**Do not delete:** `franka_assets/` · `assets/robots/franka_skin/` ·
`assets/robots/fr3_hybrid_skin/meshes/skin/` · `assets/urdf/fr3_hybrid_skin.urdf` ·
`data/pact_place_corridor_v5` · `act_style_data/pact_place_corridor_v5` · `reports/`
(especially `reports/eval_summaries/`) · `policy_best.ckpt` / `policy_last.ckpt`.

**`policy_best.ckpt` and `policy_epoch_<best>.ckpt` are not byte-identical.** Intermediate epoch
files were purged.

**Never delete `reports/eval_summaries/`.** `eval_output/` is gitignored; those JSON files are
the surviving record of the published 66→40 grid.

### 2026-08-16 cleanup (history)

Committed the 2026-08-14 report; archived 24 `eval_summary.json` into `reports/eval_summaries/`;
purged caches; deleted dead `eval_output/` dirs and five dead ACT evaluators; pruned
`TASK_CONFIGS`; hardened `.gitignore` (`*.ckpt`, `ckpts/`, `*.log`). `housekeeping.sh` tiers 1–4
were applied.

### Commands

```bash
scripts/housekeeping.sh --tier1                 # preview
scripts/housekeeping.sh --tier1 --apply
scripts/housekeeping.sh --all --apply           # tiers 1–4, not --gitgc, not --venvs
```

`--gitgc` does not shrink blobs that are still reachable from old commits. Reclaiming ~20 GB of
deleted `.h5` means `git-filter-repo` + force-push across every branch. Deliberate non-action.

The ACT fork is a **second repository** (`git@github.com:jdvakil/act.git`). A parent commit that
bumps the submodule pointer is broken for every fresh clone until the fork itself is pushed:

```bash
cd submodules/act && git push origin main && cd ../..
git push origin main
```

`main` is the trunk. `encoder_eval` still holds Safety-CVAE ablation infrastructure (the unused
"does the head actually use the skin" instrument). `environments` should be cherry-picked, not
merged whole (drags `analysis_output/` blobs).

Pipelines:

```
MJCF arm → <Config> datagen → assets/datagen/<run>/*.h5
                ├─ safety_sweep.py → sweep.h5 → train_safety_cvae.py → reflex demos
                └─ convert_*_to_act.py → act_style_data/<ds>/
                                              ↓
                                    imitate_episodes.py → ckpts/<task>/<run>/
                                              ↓
                         eval_act_obstacle.py / eval_act_place_corridor.py
                         scripts/run_pact_place_eval_chunk100.py  (40 frozen rows)
                                              ↓
                                    eval_output/.../eval_summary.json
                                              ↓
                                      compare_pact.py
```
