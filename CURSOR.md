# CURSOR.md — session change log

Living log of agent edits in this checkout. Read this before new work. Append
a dated block after every non-trivial change. Do not replace history; add
below.

Science, claims, and how to run live in [`README.md`](README.md). This file is
only *what the agent changed, why, how, when*.

How to append:

```
## YYYY-MM-DD — short title
- **When:** ...
- **Why:** ...
- **What:** files + behavior
- **How:** implementation notes
- **Not done:** leftover for the user or a later session
```

Newest session at the top.

---

## 2026-09-05 — diagnose failed v12 reference/optimized parity

- **Evidence:** User's offline validation and runtime/check commands passed. Both
  101-step pairs in `4c5fd12983d9c514` completed, but saved input hashes differ at
  reset and subsequent actions diverge (max arm difference 0.000487/0.002596 rad).
  Success flags and contact summaries agree. The subprocess traceback merely
  propagates the failed verification status; training did not fail.
- **What:** Added per-input-field hashes and detailed comparison reports, including
  first differing sensor/state field and numeric deltas. Verification stops after
  the first failed pair while requiring all planned pairs for a pass. Preserved
  exact hash and 1e-6 numeric comparisons. Wrapper now reports child failures without
  a redundant traceback and preserves their exit codes. README records evidence,
  diagnostic fields, remaining uncertainty and the rerun command.
- **Validation:** Focused tests cover sensor mismatch localization, legacy trace
  diagnostics and fail-fast/all-pairs verification behavior. Existing saved traces
  inspected without modifying their results. No training or GPU rollout launched.
- **Unresolved:** Original records lack per-component hashes; CUDA is unavailable
  in the agent execution environment. The next user-run verification is required
  to identify the differing input source. No root-cause fix or parity pass claimed.

---

## 2026-09-05 — comprehensive current-workflow README audit

- **Why:** User asks for a thorough README containing the complete workflow without
  relying on the chat history.
- **What:** Expanded §4.21 with full readout tensor flow, pretrained initialization,
  gradient behavior and an explicit GPU training command; expanded §4.22 with the
  arm comparison. Added §4.23 for complete CLI syntax, checkpoint pair semantics,
  metric/artifact definitions, result inspection, batch evaluation, failure recovery,
  new dataset onboarding, provenance and the focused validation command.
- **Consistency:** Added a current-workflow notice and routing links, included v12
  in the profile table, updated the repo map, separated direct/legacy CLI defaults
  from wrapper defaults, and qualified old results and timings with their historical
  query-history protocol. Kept dated historical measurements and prior log entries.
- **Validation:** Checked command flags against current parsers, shell/Python example
  syntax, current-workflow anchors and whitespace. No training/evaluation launched;
  the 69-test result is identified as the prior implementation check, not a new run.
- **Scope:** Documentation only; implementation and existing staged edits preserved.

---

## 2026-09-05 — enable the actual finetuned PACT-readout architecture

- **Why:** User wants v12 numbers from the full model used in the existing readout
  experiment, rather than the peak-closeness baseline.
- **What:** `pact.py train --arm readout` is now the default. It uses the existing
  hallway pretrained surface encoder, 40 × 128-d CLS features, one token per sensor,
  live causal inputs and joint encoder/ACT finetuning. Added encoder checkpoint/LR
  options; W&B display names now match local run names. Baselines remain explicit.
- **Evaluation fix:** Shared inference previously accumulated geometry history only
  at chunk queries. It now records consecutive control frames before returning a
  cached action. Contract gating retains skin every step while caching unused RGB;
  legacy gates retain fresh observations for history encoders. Readout is accepted
  only with the expected configuration and matching finetuned weights. Verification
  hashes every consumed skin frame using saved sensor names.
- **Checkpoints:** Added `pact_checkpoint.py`, paired best/last/periodic encoder
  saves and pair hashes. Simulation and offline loaders select the corresponding
  encoder and reject missing/mismatched pairs; evaluation identity includes encoder
  weights. Historical best pairs without a hash index remain loadable.
- **Docs:** Updated README §4.20–4.22 and batch examples for readout; marked old
  hallway numbers/timings as belonging to the historical query-history protocol.
- **Validation:** 69 focused tests passed. Loaded the actual pretrained checkpoint
  on CPU, verified `(1, 40, 128)` features and nonzero encoder gradients. The v12
  dry-run passed against the existing 132/33 manifest. Checked Python/Bash syntax
  and whitespace. Preserved prior edits; no training, package install or physics
  evaluation launched. Full-horizon smoke, live trace parity and judge controls
  remain necessary before reporting new simulation results.

---

## 2026-09-05 — full wrapper reference and batch-training recipes

- **Why:** User wants to understand what the wrapper does, why it exists and how
  to use it for multiple training jobs.
- **What:** Added README §4.22 and a routing-table link: all commands, settings
  sources/defaults, manifest lifecycle, trainer handoff, artifacts, checkpoint
  selection, staged evaluation, serial batches and explicit two-GPU jobs.
- **How:** Matched the reference to current wrapper/trainer/evaluator code. Called
  out blocking execution, absent training resume/scheduling, W&B names differing
  from local run names, shared-setup concurrency limits and evaluation identity.
- **Validation:** Bash syntax checked for all new command blocks; CLI help/list
  checked without launching training, conversion, package installation or rollouts.
- **Not changed:** Preserved the existing wrapper edit and training behavior.

---

## 2026-09-05 — document the training wrapper and direct-training differences

- **Why:** User asks whether README and CURSOR include the workflow explanation.
- **What:** Added the explicit `pact.py train` versus `imitate_episodes.py`
  comparison to README §4.20: shared trainer/model/loss/optimizer implementation,
  dataset selection, grouped splits, training-only normalization, saved metadata,
  output directories, defaults, dry-run command and legacy direct invocation.
- **Correction:** Current `pact.py` no longer passes `--no_wandb`; documentation
  reflects the trainer's default logging behavior. Preserved that existing code edit.
- **Validation:** Checked documentation against the current wrapper and trainer;
  documentation-only changes, no training or evaluation launched.

---

## 2026-09-05 — separate v12 training and the actual collection scene variant

- **Why:** User switches to v12 after an OOD V1011d speed run scored 0/50 in
  5.85 hours. The new result still used FourObject/F0-left-center and no overlay.
- **What:** Added v12 registry profile, guarded `pact.py convert`, exact historical
  preview XML export, `pact_v12_adapter.py`, and evaluator overlay/camera/scene
  binding. Prepared contracts fingerprint overlay files; v12 test suite has 48
  center-pose rows, not 16 or a 24-category pose sweep. README §4.21 has commands.
- **Evidence:** Dataset manifest says onebottle + standing kitchen extras while
  raw scene_params retain FourObject. Located the exact cb6be07e... XML in git
  history. Verified nominal layout/jitter inputs against all 165 raw rows.
  Expected split 132/33 with both repeated selected-seed pairs kept together;
  rare F3-left category has only two examples.
- **Validation:** 39 focused tests pass. All eight v12 sampler configurations,
  preview XML hashes, pre-overlay object poses and pinned overlay imports checked.
- **Not done:** No dataset conversion, training, package installation or live
  policy evaluation. User runs those commands. The selected runtime and geometry
  integration still require live validation; no higher-success-rate promise.

---

## 2026-09-04 — dataset-bound training and evaluation workflow

- **Why:** User needs multiple dataset/environment pairs, rapid checkpoint iteration,
  and an evaluation protocol checked against robot-learning references.
- **What:** `configs/pact_datasets.json`, `scripts/pact.py`, `pact_workflow.py`,
  `pact_eval_protocol.py`, ACT `eval_pact.py`; manifest-aware training arguments and
  grouped split / train-only normalization; native-depth chunk gating and tests.
  README §4.20 documents exact commands, ACT/robomimic comparisons and limitations.
- **Behavior:** Fixed smoke/dev/test suites, pinned code exports, isolated runtime
  setup option, saved run contracts, legacy checkpoint pointers, trace parity checks,
  success-ever plus terminal success, full-horizon contact audit, incomplete/error
  handling and identity-bound resume. Initial adapter accepts ACT and raw PACT only.
- **Validation:** 28 focused tests; configuration construction and parameter draws
  for V1011D (56) and hallway (50). Prepared both contracts/code exports. Bound the
  existing V10.11d checkpoint as `v1011d_existing_s0`. Runtime preflight correctly
  rejects installed MuJoCo 3.6 / Warp 1.13 for the collection's 3.5 / 1.11 pin.
- **Not done:** No package installation, training, policy inference, physics rollout,
  live parity benchmark or expert/judge calibration. User runs setup/verification
  commands; no fault-free or measured speedup claim. Existing weights/stats preserved.

---

## 2026-09-04 — remove dataset-only segmentation from metrics-only evaluation

- **Why:** User asks why evaluation is slow and how to speed it up. Actual V10.11d
  logs show 40,338.65 s sensor polling, 19,500.19 s physics/control/audit, and 41.21 s
  policy action calls across 48 rollouts. Previous camera-only attribution was incomplete.
- **What:** Shared `eval_place_fast_hooks.py` filters `ObjectImagePointsSensor` and
  `EnvStateSensor` from metrics-only suites. Segmentation annotations had bypassed
  the RGB/proximity chunk gate on every step. Save-trajectories mode keeps them.
  V10.11d evaluator records backend, filter mode, and elapsed/sensor-query timing.
  `tests/test_eval_place_fast_hooks.py`; README §4.19 has evidence and a one-episode command.
- **Validation:** Eight focused tests passed; syntax and whitespace checks passed.
- **Not done:** User runs the benchmark. No measured speedup or rollout-equivalence
  claim. Physics/control/audit remains a separate cost. V1011D matching-sampler adapter
  is still pending. Removed annotations consumed RNG; use controlled scene seeds.

---

## 2026-09-04 — zero-success audit and simulator-free checkpoint diagnostic

- **Why:** User confirms V10.11d is the failing checkpoint, wants usable dataset
  splits and faster iteration, and plans more checkpoints later.
- **What:** `submodules/act/attn_heatmap.py` infers chunk length from the selected
  checkpoint and accepts camera order. `eval_train_set.py` reads conversion camera
  metadata, selects legacy train/val or explicit IDs, caps episodes after splitting,
  and saves masked deployed action/arm/gripper metrics plus provenance as JSON.
  `tests/test_eval_train_set.py`; root README §4.18 contains commands and split advice.
- **Findings:** Existing evaluator remains V1010/OOD. Only 1/48 target contacts at
  horizon 1050, not a measured successful grasp. Raw V1011D metadata shows four
  duplicated selected-seed/layout pairs; converted IDs 2/22 cross legacy train/val.
  Statistics use all episodes; validation misses four cells. Gating is already
  active, and existing summaries cannot identify the proximity backend or bottleneck.
- **Validation:** Focused unit tests; no training, simulation, or model inference.
- **Not done:** Matching V1011D sampler adapter, grouped training split and train-only
  statistics, conversion provenance retention, runtime benchmarking. Existing checkpoint
  statistics and data were preserved. User runs diagnostic commands from README.

---

## 2026-09-04 — v1011d 0/48 is OOD eval, not a broken judge

- **When:** 2026-09-04. User: 1050 JSON place 0%; thinks eval script is the bug.
- **Why:** Success counter is `PickAndPlaceTask.judge_success` (cup on tray and released).
  47/48 never touch the cup. 1/48 grasps and holds. Horizon 800 already 0/48. Train dump
  is V10.11d randomized clutter (6 bodies + primitives). Script binds V10.10 four-object
  from `origin/main`. V1011D sampler is `70dedc0` only.
- **What:** README §1 / §4.17 / trap 32 / decision log. Eval script docstring + OOD print +
  grasp/tray tallies in summary. Archived
  `reports/eval_summaries/pact_pick_n_place_v2_v1011d_raw_s0_n48_horizon1050.json`.
- **How:** Read train `obs_scene` vs eval JSON `eval_env`. Did not re-run eval. Did not
  rewrite the script onto `70dedc0` (needs hallway-style import rewrite; `70dedc0` has no
  `pact_place_datagen_configs.py`).
- **Not done:** Wire `PactPlaceCorridorV1011DRandomizedLayoutSampler`. User runs worktree
  add + smoke after that lands. Do not cite 0/48.

---

## 2026-09-03 — do not dismiss readout place-success

- **When:** 2026-09-03. User: calling 40% place "noise" is wrong; the new model beat ACT on
  success **and** collision avoidance.
- **Why:** 20/50 vs 14/50 is a real +12 point place gain on the eval they ran. Fisher p = 0.29
  only says n=50 is underpowered for a 0.05 star on place — it does not make 40% fake.
- **What:** README lede, §1, §4.3, §4.4, §6, §8. Report both axes. Drop "safer, not a better
  placer" / "success is not a significant improvement."
- **Not done:** second seed. User writes the paper.

---

## 2026-09-03 — hallway readout n=50 is the paper MVP

- **When:** 2026-09-03 America/Denver. User: `eval_output/place_corridor_readout_s0_n50_fast`
  shows 88% collision-free and 40% success; make that the paper MVP in README.
- **Why:** Live checkout can reproduce hallway readout. The 2026-07-05 66%→40% grid is wiped.
  Hallway PACT-raw still has no safety win; the gain is finetuned CLS readout.
- **What:** `README.md` lede, §1, §4.3 table, §4.4, §5–§8, §13, decision log. Archived JSON
  copies: `reports/eval_summaries/place_corridor_{vanilla,raw,readout_s0_n50_fast}.json`.
  Numbers: place 20/50 (40%), bar 6/50 (12%), free 44/50 (88%). Fisher bar vs ACT p = 0.016,
  vs raw p = 0.009. Place vs ACT p = 0.29 (noise).
- **How:** Read `eval_summary.json`. Did not re-run eval. Do not mix with invisible-cell 66→40.
- **Not done:** multi-seed. User writes the paper. Do not overwrite the `n50_fast` eval dir.

---

## 2026-09-03 — README: new clones + train/eval walkthrough

- **When:** 2026-09-03 ~00:40 America/Denver. User: dump the chat brief into README so
  the other desktop has it in the morning without this chat.
- **Why:** Sep HF clones (v1010, v12, mixed v10.11c, …) landed; README disk truth was
  still 2026-08-27. Train/eval is v5-only. User asked how to train the new envs; answer
  was "do not copy-paste; wire first." That has to live in the cookbook.
- **What:** `README.md` only (no new markdown).
  - §1 disk truth → 2026-09-03. New-clone row. Live train set still v5 only.
  - §2 routing: train new clones / skeptic walkthrough → §4.17; leftover readout → §4.4.
  - §4 intro: viz-only warning.
  - **§4.17** inventory, fork A vs B, wiring list, convert→train→eval skeptic
    walkthrough, v5 leftover eval cmd, morning next.
  - §7 one-line row. §10 token-budget lie (320 vs “40”). `TASK_CONFIGS` table. Scripts
    map. Traps 29–30. Decision log + unresolved. §16 disk table + pipeline diagram.
- **How:** Disk counts from `data/` + `convert_meta.json` + `constants.py` + live
  `prox_config.json`. No convert/train/eval code change.
- **Not done:** Wire v12 (recommended) convert + TASK_CONFIGS + eval sampler. User picks
  the set in the morning. Do not start 2000-epoch jobs on the new clones.

---

## 2026-09-02 — dashboard collected time

- **When:** 2026-09-02. User wants collection date/time on the viz dashboard if saved.
- **Why:** Most h5 dumps store no time attr. Datagen folders are `YYYYMMDD_HHMMSS`.
  Some sidecars have `started_utc`. File mtime is clone time, not collection.
- **What:** `infer_collected_at` reads attrs/scene keys, closeout/result JSON,
  then path stamps. Dashboard card + list + episode buttons. `--dashboard`
  backfills old audits from `src`. README §4.2.1. Tests.
- **How:** No h5 scan. No mtime as "collected".
- **Not done:** user `--dashboard` (or next `--each`) to refresh HTML.

---

## 2026-09-02 — viz.sh skips conda activate

- **When:** 2026-09-02. User: `conda activate mlspaces` fails inside `viz.sh`.
- **Why:** `conda activate` is a bash function from `conda.sh`. Non-interactive
  scripts do not load it.
- **What:** `viz.sh` calls `/opt/conda/envs/mlspaces/bin/python` and `cd`s to
  the repo. Extra flags pass through `"$@"`. README §4.2.1.
- **How:** `exec` env python. No conda hook.
- **Not done:** user `chmod +x viz.sh` if the bit is off, then `./viz.sh`.

---

## 2026-09-02 — viz incremental (no --force)

- **When:** 2026-09-02. User: `--each --force` on `data/` re-encodes a,b,c,d when
  they only added e. Git pull / clone into `data/` should update the dashboard
  with the new set only.
- **Why:** `--force` means redo every dataset that already has output. Daily
  command must omit it. Also needed grow-path when an existing clone gains
  episodes.
- **What:** `viz_action` skip / grow / run. `--each` skips finished datasets,
  encodes new folders, appends new episode clips (keep old mp4 + timeline).
  `--force` still full redo. README §4.2.1 + trap 28. Tests in
  `tests/test_dataset_viz_mp4.py`.
- **How:** Compare `audit.n_eps_exported` to catalog `n_eps`. Append matches
  episode labels and skips clips that still exist.
- **Not done:** User drops `--force` on the daily command.

---

## 2026-09-01 — dashboard: no HTTP server

- **When:** 2026-09-01. User on SSH, previews HTML from VS Code. Kill `--serve`.
- **Why:** `--serve` is a local `http.server`. Useless over SSH. `fetch` of
  JSON also dies in VS Code Simple Browser / `file://`.
- **What:** Drop `--serve` / `serve_dashboard`. Catalog stays baked in
  `index.html`. Plots load via `timeline.js` (`window.DATASET_TIMELINE = …`).
  `--dashboard` writes those js files from existing `timeline.json`. README
  §4.2.1 + trap 27.
- **How:** `<script src="slug/timeline.js">` works without an HTTP server.
- **Not done:** User `--dashboard` then Simple-Browser the root `index.html`.

---

## 2026-09-01 — dataset viz dashboard

- **When:** 2026-09-01. User: root
  `experiments_output/default/dataset_viz/index.html` is a crude table; make a
  real dashboard (stats, plots, clips) that stays cheap as data arrives fast.
- **Why:** Old audit HTML baked one `<video>` per dataset. Browser preloaded
  every clip. No filter, no Plotly, no live catalog.
- **What:** `scripts/dataset_viz.py` writes a SPA dashboard. `--dashboard`
  rebuilds catalog only. `--each` rewrites the catalog after every dataset.
  New audits get `skin_min_min` / `skin_min_mean`. README §4.2.1 + trap 27.
- **How:** Catalog is metadata only. One player (`preload=none`). Bootstrap
  JSON inside `index.html`.
- **Not done:** superseded by the no-server block above.

---

## 2026-09-01 — dataset viz dashboard

- **When:** 2026-09-01. User: root
  `experiments_output/default/dataset_viz/index.html` is a crude table; make a
  real dashboard (stats, plots, clips) that stays cheap as data arrives fast.
- **Why:** Old audit HTML baked one `<video>` per dataset. Browser preloaded
  every clip. No filter, no Plotly, no live catalog.
- **What:** `scripts/dataset_viz.py` writes a SPA dashboard. `--dashboard`
  rebuilds catalog only. `--serve [PORT]` (default 8765) serves
  `_OUT_BASE` and re-scans audits on each `audit.json` GET. `--each` rewrites
  the catalog after every dataset. New audits get `skin_min_min` /
  `skin_min_mean`. README §4.2.1 + trap 27.
- **How:** Catalog is metadata only. One player (`preload=none`). Timeline
  fetched per click and cached. Bootstrap JSON inside `index.html` so
  `file://` still lists datasets. Poll 4 s.
- **Not done:** User runs `--dashboard` then `--serve`. Old per-dataset
  `index.html` pages stay.

---

## 2026-09-01 — v1010 viz missed wrist/table RGB

- **When:** 2026-09-01. User: viz of
  `experiments_output/default/dataset_viz/pact_place_corridor/data/v1010/accepted`
  shows no wrist or table camera.
- **Why:** `glob_mp4` only matched `episode_{vid_id:08d}_{stem}.mp4` with
  `RGB_STEMS = (wrist_camera, exo_camera_1)`. HF discover always sets `vid_id=0`.
  v10.10 sidecars are `episode_{sha}_{wrist,table}_camera.mp4`. Audit: `cams=[]`,
  gaps `no wrist RGB` / `no table RGB`. Pixels were on disk.
- **What:** `glob_mp4` also accepts folder-sha names and a sole `episode_*_{stem}.mp4`.
  `RGB_STEMS` includes `table_camera`. README §4.2.1 + trap 26. Test
  `tests/test_dataset_viz_mp4.py`.
- **Not done:** user `--force` regen. 215 eps. `--cam3d` still wrist-only on this
  dump (`sensor_param` has no table cam).

---

## 2026-08-31 — heatmap fills leftover panel

- **When:** 2026-08-31. User: pick `episodes/free/0000_house_1_traj_0.mp4` 8x8 grid
  tiny, blank band to the right.
- **Why:** `render_heatmap` used square `min(w,h)` cells (~20 px) left-aligned. RGB
  strip was 240 px so heatmap only got 240 px height.
- **What:** Tiles stretch per-row to fill width; RGB strip 160 px, heatmap 320 px.
  README §4.2.1.
- **Not done:** user `--force` regen (this clip, or whole pick).

---

## 2026-08-31 — drop empty RGB tiles in dataset.mp4

- **When:** 2026-08-31. User: Check_r5 (and same layout) blank RGB box steals space
  from heatmap / prox 3D.
- **Why:** `compose_frame` always reserved wrist|table. Missing cam drew "no table RGB"
  slate (hallway, sweep).
- **What:** `_present_rgb` only draws cams that have pixels. 1 cam → full 640-wide
  tile. 0 RGB → heatmap fills left 640×480. No prox → no heatmap slate; RGB grows
  tall. README §4.2.1.
- **Not done:** user `--force` regen of hallway / Check_r5 (Check_r5 already has
  both RGB so layout stays two tiles).

---

## 2026-08-31 — audit compilation videos + live prox 3D panel

- **When:** 2026-08-31. User: compilation video for every unique dataset (audit what
  exists vs still collect); extra panel with proximity returns in 3D as the robot moves.
- **Why:** `--each` mixed in DUP copies (`pact_20260622/data/openfrontcluttered_52_act`
  and `raw_openfrontcluttered`). MP4 had no world-frame skin cloud. `--no-mcap` skipped
  FK so 3D would have been empty if we only logged in the MCAP path.
- **What:** MP4 layout wrist|table + prox-3D (FK skeleton + turbo back-projected
  returns). `--each` skips DUP copies (`--keep-dups` to keep). Writes per-dataset
  `audit.json` and parent `experiments_output/default/dataset_viz/index.html`.
  `export_episode` always FK+3D even with `--no-mcap`. README §4.2.1.
- **How:** `proximity_world_points` uses saved `cam2world_gl` or MuJoCo `cam_xmat`.
  Numpy look-at projector (no EGL). H.264 remux unchanged.
- **Not done:** none. Videos at `experiments_output/default/dataset_viz/`. Open
  `index.html`. Stride 2. 3 DUP copies skipped. Some fumehood pick h5 chunks
  LZF-fail (skipped, not missing on purpose).

---

## 2026-08-31 — H.264 so Cursor plays dataset.mp4

- **When:** 2026-08-31. User: encode videos for VS Code / Cursor IDE preview.
- **Why:** OpenCV `mp4v` is MPEG-4 Part 2. Chromium previewer wants H.264 + yuv420p.
- **What:** `encode_h264_ide` after write (`libx264`, `+faststart`). Existing
  `experiments_output/default/dataset_viz/*/dataset.mp4` re-encoded.
- **Not done:** user re-opens the mp4 in the editor.

---

## 2026-08-31 — data/ catalog + --each

- **When:** 2026-08-31. User: all data lives in `/home/jaydv/code/prox_learning/data`
  (hallway clone + `Likerener/molmo-pi0-eval-videos`).
- **Why:** pointing `--data data/` mixed 112 ACT hdf5 from nested openfront copies and
  skipped hallway / fumehood. The HF dump is many datasets, not one.
- **What:** `scan_dataset_roots` + `--list` catalog + `--each` (one viz per dataset).
  Skip `results/` / `eval/` / `videos/` unless `--include-eval`. README §4.2.1 uses
  `data/` paths.
- **Not done:** user `--list` then smoke `--each --max-episodes 2`.

---

## 2026-08-30 — dataset folder visualizer

- **When:** 2026-08-30 America/Denver. User: folder of h5 → one trajectory/dataset video
  (MCAP or web), wrist + table RGB, sensors, joint pos/vel, anything else; clone more
  datasets later; just the script.
- **Why:** `foxglove_viz.py` is datagen-only (`traj_*` + `_batch_*.mp4`, exo-id glob).
  ACT `episode_*.hdf5` and HF `rows/*/trajectory.h5` (wrist-only, no `_batch_`) did not
  load. `visualize_episodes.py` is one-episode RGB + wrong ALOHA joint names, no skin.
- **What:** `scripts/dataset_viz.py` auto-detects ACT / HF / datagen, concatenates every
  episode onto one timeline. Writes `dataset.mcap` + generated `foxglove_layout.json`
  (Foxglove) and `dataset.mp4` + `index.html` + `timeline.json` (browser). Topics: wrist
  / table RGB, 8×8 skin mosaic from the proximity tensor, optional embeddings, FK `/tf`
  + `/proximity` cloud, `/joints` q/v/action/skin-min, `/task` attrs. README §4.2.1.
- **How:** one episode in RAM at a time. Heatmap is numpy turbo mosaic (no matplotlib).
  Reuses `foxglove_viz` FK / backproject / scene markers. Table cam absent → slate.
- **Not done:** user runs the commands below. Full 152-ep corridor dump is long; smoke
  with `--max-episodes 2` first.

---

## 2026-08-29 — rays default; EGL was 18 min/ep gated

- **When:** 2026-08-29 America/Denver. User: 35 min wait ridiculous.
- **Why:** PACT smoke **2121.4 s / 2 rows**. `renders=19 skip=883` — gate worked.
  Tax is 19×40 EGL `update_scene` (~1.4 s/cam). PACT_READOUT still on that path.
- **What:** `mj_multiRay` default for PACT eval. `--egl-prox` opt-in for the
  rasterizer. Launcher streams already. README §4.3.1 / trap 25.
- **Not done:** user Ctrl+C readout, rerun `--arms PACT_READOUT` (ACT+PACT json
  skip). Rays ≠ EGL pixels.

---

## 2026-08-29 — mj_multiRay signature + transformers spam

- **When:** 2026-08-29 America/Denver. Readout smoke died: `TypeError: mj_multiRay()`.
  Wall of HuggingFace `ProximityDepthBufferSensor` alias warnings.
- **Why:** This MuJoCo wants `normal=None` plus column arrays `[m,1]`. Patch
  `getattr` on every `sys.modules` entry poked transformers lazy image processors.
- **What:** Column-shaped `mj_multiRay` + `normal`. Patch only `molmo_spaces.*`.
  Launcher `--no-skip-existing` (row 000 was leftover EGL readout).
- **Not done:** user reruns readout smoke with `--no-skip-existing`.

---

## 2026-08-29 — PACT eval 12–15 h; stream + skip + rays

- **When:** 2026-08-29 America/Denver. User: PACT not loading faster; 18-day
  deadline; 12–15 h/model too slow.
- **Why:** Cost is 40 EGL 8×8 `update_scene` calls, not ckpt I/O. ACT smoke
  already `renders=19 skip=883` in 119 s/2 eps. PACT smoke sat ~15 min on row 0
  with launcher `capture_output=True` (no live `renders=`). Two remaining
  taxes: skip missing `ProximityDepthBufferSensor` across duplicate
  `molmo_spaces` modules, and 19×40 EGL even when skip works.
- **What:**
  - `eval_act_place_corridor.py`: skip by class name + patch every
    `ProximityDepthBufferSensor.get_observation`; batch 40-cam
    `record_proximity_depths` on chunk query; heartbeat `skin query #N`;
    `--fast_prox_rays` (`mj_multiRay`, group 2 hidden).
  - `scripts/run_pact_place_eval_chunk100.py`: stream worker logs,
    `PYTHONUNBUFFERED=1`, `--fast-prox-rays`.
  - README §4.3.1 18-day loop; trap 25.
- **Not done:** user Ctrl+C current smoke; rerun PACT-only with
  `--fast-prox-rays` then EGL 40-row table. Rays ≠ paper table.

---

## 2026-08-29 — Amine 40-row place protocol on local ACT/PACT ckpts

- **When:** 2026-08-29 America/Denver. User pasted Amine
  `run_pact_place_eval_chunk100.py --arms ACT PACT PACT_PERMUTED` and asked to
  eval local models with his scripts (`amine/act/eval_pact_place_chunk100_row.py`).
- **Why:** His worker is a hashed chunk-100 / 32-d / `run_manifest.json` pipeline.
  Local ckpts are chunk 50 (vanilla, PACT-raw K=8 dim 1, readout 128-d). `strict=True`
  load dies. `PACT_PERMUTED` needs his `(40,900,40,32)` token plan (not on disk).
  Original `eval_pact_collision_row.py` in `amine/act` is still the fast wrapper.
- **What:** Reuse his **40 frozen scenes**, keep Jay policy load.
  - Vendored `scripts/pact_place_eval_chunk100_contract.py` +
    `configs/pact_place_eval_chunk100_manifest.json` from
    `origin/experiment/pact-valid-ablation-followup-v1`.
  - `eval_act_place_corridor.py --manifest` pins `set_pact_manifest_row` /
    `task_seed_u32`. `--temp_agg_off`. Horizon 900.
  - `scripts/run_pact_place_eval_chunk100.py` launches ACT / PACT-raw /
    PACT_READOUT. `--workers 10` clamped to 2. `PACT_PERMUTED` skipped.
  - Gripper-close flag on `ACTInferencePolicy`. README §4.3.1, trap 24.
- **Not done:** user runs smoke then full. No eval number. Not a claim. Do not
  mix with random-house n=50.

---

## 2026-08-29 — fast place-corridor eval (chunk-gated skin)

- **When:** 2026-08-29 America/Denver. User: convert `amine/act/eval_pact_collision_row.py`; 12 h/model too slow; want <4 h.
- **Why:** n=50 PACT profile was ~0.75 s/step sensor EGL × 800 (40×8×8 every control step). `--temp_agg_off` already ignores those frames. Rendering them was waste.
- **What:** `ACTInferencePolicy.needs_fresh_policy_observation`. `eval_act_place_corridor.py` monkeypatches `SensorSuite.get_observations` so RGB/skin EGL run only on chunk queries. Physics + 2 ms contact audit unchanged. `amine/act/eval_pact_collision_row.py` is now the hallway entry: `--checkpoint-dir` → `policy_best.ckpt`, forces `--temp_agg_off`. README §4.4.
- **How:** Stale last observation reused on idle steps; prox depth buffer cleared before each real query so skin is live at chunk boundaries. Bit-identical executed actions vs old temp_agg_off path.
- **Not done:** user runs n=50 readout. Old incomplete `eval_output/place_corridor_readout_s0_n50` (36/50) is not this run.

---

## 2026-08-28 — unfreeze geometry encoder; CLS readout at train/eval

- **When:** 2026-08-28 America/Denver. User: no frozen encoder; finetune with readout
  tokens at inference.
- **Why:** Frozen 32-d embedding bake is a compressor tap. Policy never trained the
  stem. User wants the CLS hidden state as the ACT token, same forward train and eval,
  grads on.
- **What:**
  - `SurfaceEmbeddingEncoder` / `SurfaceProximityEncoder`: `encode_sequence` +
    `readout_tokens()` → `(B, 1, 128)` CLS. Pretrain heads unchanged.
  - `SurfaceGeometryEncoder(frozen=False, policy_tap="readout")` → `(B, S, 128)`,
    grads on. Default still frozen 32-d / XYZ.
  - `--finetune_prox_encoder` on `imitate_episodes.py`: needs
    `--prox_feature surface_embedding` + `--prox_encoder_ckpt`. Forces
    `raw_causal`. Adds encoder param group. Saves `prox_encoder_best.pt`.
  - Eval / heatmap load that file, tap=readout, `.eval()`.
  - Do not bake tokens for this arm.
- **Files:** `encoders/surface_geometry.py`, `encoders/pact.py`, `encoders/__init__.py`,
  `encoders/__main__.py`, `tests/test_encoders.py`, `submodules/act/imitate_episodes.py`,
  `detr/main.py`, `eval_act_obstacle.py`, `attn_heatmap.py`, `eval_train_set.py`,
  `README.md`.
- **Not done:** user runs pytest, then the train command in README §4.4. No eval
  number. Not a claim. Headline stays PACT-raw.

---

## 2026-08-27 14:07 MDT — docs: hallway n=50 ACT vs PACT-raw

- **When:** 2026-08-27 ~14:07 America/Denver. User: write n=50 into markdown; they push.
- **Result:** ACT 14/50 (28%) place, 17/50 (34%) bar, 33/50 collision-free.
  PACT-raw 21/50 (42%) place, 18/50 (36%) bar, 32/50 collision-free.
  Fisher success p = 0.21, bar p = 1.0. No safety win. Success gap noise.
  n=20 smoke was luck. Files: `eval_output/place_corridor_vanilla_s0_n50/` and
  `place_corridor_raw_s0_n50/`.
- **Wrote:** `experiments.md`, README §13.1 + §14 + decision log, `PACT.md` §8,
  `STATUS.md` header, `paper.md` do-not-cite. Headline stays 66→40.
- **Not done:** user commits and pushes. Encoder bake and gate-bar still parked.

---

## 2026-08-27 00:52 MDT — PACT n=50 RAM: leave running

- **When:** 2026-08-27 ~00:52 America/Denver. User: do not let RAM kill overnight PACT n=50.
- **Live:** PID 65821. Metrics-only on (`cameras=41`, proximity ON). Tree RSS **8.1 GB**. MemAvailable **53 GB**. GPU 2.1 / 24 GB. Same ~6 GB footprint as n=20. Did **not** restart.
- **Watch:** `eval_output/place_corridor_raw_s0_n50/rss_watch.log` every 5 min.
- **Code (next run only):** `eval_act_place_corridor.py` now drops `task.sensor_suite` in metrics-only `get_history` so 50 eps cannot pin 50 camera suites. Current PID already imported the old hooks.
- **Not done:** no sudo, cannot add swap or lower oom_score_adj.

---

## 2026-08-27 00:43 MDT — docs: local n=20 hallway ACT vs PACT-raw

- **When:** 2026-08-27 ~00:43 America/Denver. User: update markdown including
  `experiments.md`, then next steps.
- **What:** Wrote the 2026-08-26 n=20 smoke into `experiments.md`, README
  §13.1 + §14 + decision log, `PACT.md` §8, `STATUS.md` header, `paper.md`
  do-not-cite. Numbers: ACT 15% place / 30% bar vs PACT-raw 35% / 20%;
  Fisher p = 0.27 / 0.72. Fence: not a paper number; 66→40 stays headline.
- **Next (user):** optional `--num_rollouts 50` for power (PACT ~13 h);
  park encoder bake and gate-bar collect.

---

## 2026-08-26 22:31 MDT — PACT-raw n=20 done; both smoke evals in

- **When:** 2026-08-26 22:08 America/Denver finish. User: "updates." GPU idle.
- **PACT-raw n=20:** place-success **7/20 (35%)**, bar_hit **4/20 (20%)**,
  collision-free 16/20 (80%). `use_proximity=true`. ~15 min/ep, ~5 h wall.
  Files: `eval_output/place_corridor_raw_s0/eval_summary.json`.
- **Vs vanilla n=20:** success 15% vs 35%, bar 30% vs 20%. Fisher two-sided
  p = 0.27 (success) and 0.72 (bar). **Not a paper number.**
- **Wrote:** `experiments.md` hallway row as local n=20 smoke, not a cite.
- **Next if more power:** same eval, `--num_rollouts 50`.

---

## 2026-08-26 17:08 MDT — PACT-raw still on episode 0

- **When:** 2026-08-26 ~17:08 America/Denver. User: "updates."
- **Job:** PID 14153, ~11 min, CPU ~100%, GPU ~13%, RSS 6 GB (flat). Log last
  write 16:58:32. Still `collected=0/20`. Not hung: first 800-step rollout
  with 40 × 8×8 renders. ~11 min/ep ⇒ n=20 ~3–4 h. No jsonl yet.
- **Do not kill.** Vanilla was fast because skin cameras were stripped.

---

## 2026-08-26 17:02 MDT — ACT vs PACT headline row

- **When:** 2026-08-26 ~17:02 America/Denver. User: add PACT vs ACT; those are the headline numbers.
- **Why:** Outsider table buried 66% vs 40% as "cameras vs cameras + skin".
- **What:** `experiments.md` now defines ACT/PACT up top. First table row is **ACT vs PACT (headline)** with 66% vs 40% hidden-bar crashes. Dropped the duplicate main-test row.
- **Not done:** User runs nothing.


## 2026-08-26 16:59 MDT — PACT-raw n=20 eval running

- **When:** 2026-08-26 ~16:58 America/Denver. User: same tmux.
- **Job:** PID 14153. Skin ON (`proximity ON | (40, 8, 8)`). 41 cameras.
  RSS ~6 GB. Episode 0/20. Do not kill. Then compare both
  `eval_summary.json`. n=20 is smoke.

---

## 2026-08-26 16:56 MDT — simpler descriptions for outsiders

- **When:** 2026-08-26 ~16:56 America/Denver. User: make description simpler for people who do not know the project.
- **Why:** Names like PACT, CVAE, AUC, trunk were blocking a non-expert read.
- **What:** Rewrote `experiments.md` names + descriptions in plain words. Kept x% vs y% results. Short intro: arm, cameras, 40-sensor skin.
- **Not done:** User runs nothing.


## 2026-08-26 16:56 MDT — vanilla n=20 eval done

- **When:** 2026-08-26 16:52 America/Denver. GPU idle. Prompt back.
- **Vanilla (cameras only, n=20, ~49 s/ep mean, 16 min wall):**
  place-success **3/20 (15%)**, bar_hit **6/20 (30%)**, collision-free
  14/20 (70%). `use_proximity=false`. Sides L9/R11. Success eps 8,17,19.
  Files: `eval_output/place_corridor_vanilla_s0/eval_summary.json` +
  `episodes.jsonl`.
- **Not a paper number.** n=20 smoke. Do not paste into `paper.md` /
  `experiments.md` as a result.
- **Next:** PACT-raw n=20. Same eval flags, raw ckpt dir.

---

## 2026-08-26 16:49 MDT — experiments.md results as x% vs y%

- **When:** 2026-08-26 ~16:49 America/Denver. User: results need more numerical x% vs y% plus some text.
- **Why:** First pass was STE prose with few paired rates.
- **What:** Rewrote the Results column in `experiments.md`. Same tests. Each result is a comparison plus a short note. Numbers still from README §14 / STATUS / PACT / paper.
- **Not done:** User runs nothing.


## 2026-08-26 16:44 MDT — experiments.md one-line table

- **When:** 2026-08-26 ~16:44 America/Denver. User: table of experiment / description / results, one-liner, ASD-STE100, write `experiments.md`.
- **Why:** Docs already hold the numbers (`README.md` §14, `STATUS.md`, `PACT.md`, `paper.md`, `CURSOR.md`). Need one short table.
- **What:** New `experiments.md`. README intro + routing row point at it. Full numbers stay in README §14.
- **How:** STE sentences. Completed tests plus three incomplete rows (place-corridor, gate-bar collect, test-time blur). No new claims.
- **Not done:** User runs nothing. Do not treat place-corridor partial eval as a result.


## 2026-08-26 16:37 MDT — fast vanilla eval running (n=20)

- **When:** 2026-08-26 ~16:37 America/Denver. User: "ran the evaluation."
- **Job:** metrics-only vanilla, PID 9084. Wrist camera only. ~37 s/ep
  (was ~16 min). RSS ~6 GB, GPU ~1 GB. 4/20 done at check. No
  `eval_summary.json` yet. `episodes.jsonl` live.
- **Partial (not a result):** 0/4 place-success, 2/4 bar hit.
- **Next:** let n=20 finish, then PACT-raw n=20. Same flags, raw ckpt dir.

---

## 2026-08-26 15:26 MDT — kill slow eval; metrics-only fast path

- **When:** 2026-08-26 ~15:26 America/Denver. User: eval too slow for a
  week; will kill processes; needs a better way.
- **Why:** Place-corridor eval reused the datagen pipeline: 40 sensors at
  60 Hz, keep every RGB/depth frame until the house ends, reload the 321 MB
  ACT net every episode. ~16 min/ep, OOM at ~34/50. That is not the policy
  being "bad"; it is eval cost.
- **What:** `eval_act_place_corridor.py` default is now metrics-only:
  skip MP4/HDF5, drop cached frames, load policy once, vanilla strips skin
  cameras, PACT skin at policy rate not 60 Hz. Writes `episodes.jsonl` each
  rollout. `--save_trajectories` restores the old path.
- **Week path:** encoder already trained; **do not** bake tokens this week.
  Headline = vanilla vs PACT-raw. n=20 both arms, then n=50 if numbers move.
- **User:** kill the old eval, then run README §13.1 fast eval cmds.

---

## 2026-08-26 15:23 MDT — vanilla eval 16/50, RAM climbing

- **When:** 2026-08-26 ~15:23 America/Denver. User: "updates."
- **Job:** same vanilla eval PID 1898751, ~4h37m. Episode **16/50** in
  progress. ~16.3 min/ep. GPU ~1.7 GB. No `eval_summary.json`.
- **Partial (not a result):** 16 done, place-success **2/16**. Bar-hit not
  in `running_log` (stdout only). Videos not landing in output dir.
- **RAM:** RSS **32 GB** (49%). Available ~23 GB. Swap 1.8/2.0 Gi. Same
  leak that SIGKILL'd the last run at 34/50. Linear extra ~2 GB/ep will
  likely die again before 50. Do not start PACT eval on this GPU.

---

## 2026-08-26 10:47 MDT — vanilla eval killed at 34/50; user restarted

- **When:** 2026-08-26 ~10:47 America/Denver. User: "it was killed... restarting."
- **Killed run:** started 01:11, SIGKILL ~10:30 after **34/50** rollouts.
  ~16 min/ep. No `eval_summary.json`. Swap was 1.9/2.0 Gi — likely OOM.
  Output dir only configs + `running_log.log` (no kept h5/mp4 in that folder).
- **Restart:** same vanilla cmd, same dir, from episode 0. PID 1898751.
  GPU python attached. Do not kill.
- **Risk:** same 50-ep + `save_videos=True` may OOM again near ~30+. Watch
  `free -h`. If it dies, next run needs videos off (no CLI flag yet).
- **ETA:** ~13 h for 50 eps if it lives. Then PACT-raw eval.

---

## 2026-08-26 01:07 MDT — both corridor trains done; eval next

- **When:** 2026-08-26 ~01:07 America/Denver. User: "both jobs are done."
  GPU idle.
- **Vanilla:** `20260825_161821_act_place_corridor_s0` — best val 0.061355
  @ epoch 1853. No `prox_config.json`.
- **PACT-raw:** `20260825_215846_pact_place_corridor_raw_s0` — finished
  22:55 MDT. Best val 0.067665 @ epoch 1916. `prox_config.json` present
  (raw / per_sensor / min). Wandb `nle5g3e0`.
- **Next (user runs, serial, one GPU):** `eval_act_place_corridor.py
  --temp_agg_off` on each dir, 50 rollouts, horizon 800. Worktree
  `977acd6` already on disk. Place-corridor has **no** `--eval_cell`
  loop (bar is in the sampler). Headline: `eval_summary.json`
  place-success + `bar_hit_rate`. Never `imitate_episodes.py --eval`.
- **Not this:** surface-embedding ACT. Encoder bake still blocked on
  split-manifest wire.

---

## 2026-08-25 21:58 MDT — PACT-raw train running

- **When:** 2026-08-25 ~21:58 America/Denver. User: "running it now."
- **What:** `imitate_episodes.py` with `--use_proximity --prox_feature raw
  --prox_layout per_sensor`. PID 1841920, GPU ~3.5 GB. Dir
  `submodules/act/ckpts/pact_place_corridor_v5/20260825_215846_pact_place_corridor_raw_s0/`.
  Wandb `pact_place_corridor_raw_s0`. Epochs already ticking (~1.75 s/ep).
- **Do not:** kill, start a second GPU job, bake encoder tokens, or
  `imitate_episodes.py --eval`.
- **After finish:** eval both vanilla + this dir with
  `eval_act_place_corridor.py --temp_agg_off`.

---

## 2026-08-25 21:57 MDT — vanilla ACT done; PACT-raw is next

- **When:** 2026-08-25 ~21:57 America/Denver. User: "training done, what next."
- **Vanilla:** 2000/2000 finished 17:13 MDT. Best val loss **0.061355 @ epoch
  1853**. Dir
  `submodules/act/ckpts/pact_place_corridor_v5/20260825_161821_act_place_corridor_s0/`
  (`policy_best.ckpt`, `policy_last.ckpt`). No `prox_config.json` (cameras
  only). GPU idle. Wandb `act_place_corridor_s0`.
- **Next (user runs):** PACT-raw, same hypers, `--use_proximity
  --prox_feature raw --prox_layout per_sensor`. Not the 32-d surface
  encoder. Do not eval yet (one GPU). Do not `imitate_episodes.py --eval`.
- **After PACT-raw:** `eval_act_place_corridor.py --temp_agg_off` on both
  ckpt dirs. Worktree already `977acd6`.

---

## 2026-08-25 16:19 MDT — vanilla ACT corridor train running

- **When:** 2026-08-25 ~16:18 America/Denver. User: "okay, it is training."
- **What:** `imitate_episodes.py` vanilla ACT (no `--use_proximity`). PID
  1793684, GPU ~2.5 GB. Ckpt dir
  `submodules/act/ckpts/pact_place_corridor_v5/20260825_161821_act_place_corridor_s0/`.
  Wandb `act_place_corridor_s0`. ~1.65 s/epoch, 2000 epochs → ~55 min.
- **Not this job:** surface encoder (already done). PACT-raw. Embedding tokens.
- **Do not:** kill tmux, start a second GPU train, bake tokens, or
  `imitate_episodes.py --eval` on a future PACT ckpt.
- **After finish:** PACT-raw cmd in README §13.1. Then eval
  `eval_act_place_corridor.py --temp_agg_off`.

---

## 2026-08-25 15:37 MDT — test probe agrees with trainer

- **When:** 2026-08-25 ~15:37 America/Denver. User ran `encoders.probe
  --split test` and pasted the log.
- **Why:** Independent check before calling the 20-epoch ckpt useful.
- **What:** Probe on the 15 held-out rows matches `test_metrics.json`
  exactly. Representation `pooled`. 8109 GT-valid tiles (11.786%).
  - validity acc / balanced / P / R: **100%**
  - XYZ MAE both-valid **and** all-GT-valid: **20.621 mm** (zero FNs)
  - recon MSE 0.000367; foreground MAE 0.058; pixel P/R **87.4 / 95.3%**
  - per-episode XYZ MAE 17.5–24.0 mm
  - 50 cm peak-closeness hit 41.2% vs 20 cm geometry 11.8%
  - side AUC: peak closeness 0.64, 20 cm valid-frac **0.48** (chance)
  - self-view still owns 20 cm: `link1_sensor_5` 100%, `link2_sensor_3` 97%
- **Gate:** hard validity/recall **pass**. Recon pixel P and R **pass**.
  XYZ preferably <20 mm: **0.6 mm over**. Call the compressor **bake-allowed
  as an ablation**, not a policy win. Do not train
  `--prox_feature surface_embedding` until ACT `load_data` reuses
  `split_manifest.json` (still 80/20 shuffle). Headline corridor arm remains
  vanilla vs PACT-raw (README §13.1, `PACT.md` §8).
- **Wrote:** README §4 bake/split note; `PACT.md` §8 encoder fence.

---

## 2026-08-25 13:56 MDT — surface encoder 20-epoch run finished

- **When:** 2026-08-25 12:45–12:57 America/Denver. User job in tmux `0`.
  Agent checked after "it is training." GPU idle. Shell back at prompt.
- **Why:** Judge the 152-episode embedding train, not the smoke ckpt.
- **What:** `python -m encoders.train` on `data/pact_place_corridor_v5`
  finished 20/20. Split 122/15/15 episodes (`split_sha256`
  `a360742269244884845da6ec720002871e48cbbc924c30f813fcdbf602fe718a`).
  Best epoch 20. Files:
  `experiments_output/default/surface_encoder_train/pact_place_corridor_v5/`
  (`pact_surface_embedding_encoder_v1.pt`, `last.pt`, `history.json`,
  `test_metrics.json`, `curves.png`, `split_manifest.json`, `config.json`).
- **Held-out TEST (trainer, pooled val/test):** balanced validity 100%,
  P/R 100/100%, XYZ MAE **20.6 mm**, recon pixel P/R **87.4 / 95.3%**,
  foreground MAE 0.058. Val XYZ 19.7 mm. Empty-base still ~88%.
- **Gate (README §4):** validity ≥95% and recall ≥90% **pass**. XYZ
  preferably <20 mm: val pass, test **0.6 mm over**. Recon pixel P and R
  both ≥80% **pass**. Validity 100% is cheap: latest 8×8 still sits in the
  32-frame window, so occupancy is almost a copy. XYZ is the real score.
  Do **not** bake ACT tokens until `encoders.probe --split test` agrees.
- **Not done:** independent probe on test rows. `encode_tokens` still
  blocked. ACT must reuse this `split_manifest.json`. Headline PACT arm
  remains raw peak closeness (`PACT.md` §8) until policy ablations.

---

## 2026-08-25 13:06 MDT — place-corridor convert done; train is next

- **When:** 2026-08-25 ~13:06 America/Denver. User: update `CURSOR.md` and
  say what to run next.
- **Why:** Coauthor `data/pact_place_corridor_v5` is the current test, not
  gate-bar. Convert finished in an earlier turn; training never started.
  `constants.py` still had `num_episodes=0` so `imitate_episodes.py` would
  see an empty set.
- **Done (do not rerun convert / worktree):**
  - `scripts/convert_pact_place_to_act.py` wrote 152 ACT HDF5s to
    `act_style_data/pact_place_corridor_v5` (wrist 240×320, proximity
    min-pool, 0 skips). Meta: `num_episodes=152`, `episode_len=636`,
    max T=634, left=72 / right=80.
  - `TASK_CONFIGS['pact_place_corridor_v5']` now has those counts.
  - Molmospaces worktree already at `/home/jaydv/code/molmospaces-pact-place`
    (`977acd6`, `pact_place_corridor_v2.xml`). Submodule `main` untouched.
  - Eval entry: `submodules/act/eval_act_place_corridor.py`. Recipe:
    README §13.1.
- **Not done:** vanilla ACT train, PACT-raw train, eval. No local
  `eval_summary.json` yet — do not put coauthor "PACT beats ACT" in
  `paper.md`.
- **User next:** one GPU; run vanilla first, then PACT. Commands in the
  chat reply and README §13.1. `--chunk_size 50`, no `--image_dropout_p`.
  Never `imitate_episodes.py --eval` on the PACT ckpt.
- **Still parked:** gate-bar 200-ep collect.

---

## 2026-08-25 00:31 MDT — self-train surface embedding encoder from corridor rows

- **When:** 2026-08-25 00:17–00:31 America/Denver.
- **Why:** No coauthor `pact_surface_*_v1.pt` exists on this machine. User:
  "I need to train my own encoder from the data."
- **What:** Added a complete native-row trainer:
  - `python -m encoders.train`
  - Source: `data/pact_place_corridor_v5/rows/*/trajectory.h5`, native
    `(T, 40, 4, 8, 8)` metres. No ACT convert needed.
  - Default model: `SurfaceEmbeddingEncoder`, 837,700 parameters, shared across
    all 40 sensors. Input is 32 causal frames (8 control steps × 4 subframes).
  - Targets are generated from the latest native tile: nearest XYZ inside
    20 cm, valid/empty, and 8×8 20 cm closeness reconstruction. XYZ is
    normalized by 0.20 m for loss.
  - Episode-level 80/10/10 train/validation/test split. No frame leakage.
    Validation selects the validity threshold and best checkpoint. Test is
    touched once after selection.
  - Default sampler gives valid and empty samples 50/50 training mass because
    natural valid rate is only ~11%. Validation/test retain natural prior.
    `--no-balance-valid` and optional `--sensor-balance` are ablations.
  - Metrics: raw and balanced validity accuracy, precision, recall, F1,
    specificity, XYZ MAE on all GT-valid tiles, reconstruction MSE, and
    always-invalid baseline. Never accept raw accuracy alone (~89% for
    always-invalid).
  - Best checkpoint selection: highest validation balanced accuracy; validation
    loss breaks ties. Calibrated threshold stored in checkpoint and honored by
    `SurfaceGeometryEncoder` / token writer / probe.
  - Output checkpoint schema is exactly loadable by existing ACT glue:
    `pact_surface_embedding_encoder_v1`, `frozen=True`,
    `policy_feature_dim=32`. Also writes `last.pt`, `config.json`,
    `history.json`, `test_metrics.json`, and `curves.png`.
- **Files:**
  - Added `encoders/train.py` and `encoders/rows.py`.
  - Updated `encoders/surface_geometry.py`: schema constants, frozen payload
    writer, calibrated threshold loading/use, selected-timestep encoding.
  - Updated `encoders/probe.py`: shared row loader, `--split
    all|train|val|test`, precision/recall/balanced accuracy, true-positive XYZ
    MAE plus end-to-end GT-valid XYZ MAE (false negatives are zero).
  - Updated `encoders/__init__.py`, `tests/test_encoders.py`, README §4 and
    routing table.
- **Verification:**
  - `pytest tests/test_encoders.py tests/test_prox_raw.py` → **32 passed**.
  - Final smoke: 4 episodes, stride 16, 3 epochs, CUDA. Held-out test:
    95.6% raw validity accuracy, 84.2% balanced accuracy, 88.4% precision,
    69.5% recall, raw-head XYZ MAE 63.2 mm, recon MSE 0.186. This only proves
    train/val/test/checkpoint/probe wiring; it is deliberately undertrained.
  - Probe of saved smoke test row used checkpoint threshold and matched:
    95.61% / 84.19% raw/balanced validity, 88.37% / 69.51% precision/recall,
    57.8 mm XYZ MAE on true positives, 102.9 mm end-to-end GT-valid error.
  - Smoke checkpoint:
    `experiments_output/default/surface_encoder_train/smoke_final/`
- **Full user run (not launched by agent):**
  ```
  conda activate mlspaces
  cd /home/jaydv/code/prox_learning
  python -m encoders.train \
      --src data/pact_place_corridor_v5 \
      --out experiments_output/default/surface_encoder_train/pact_place_corridor_v5 \
      --kind embedding --device cuda \
      --epochs 20 --batch-size 512 --stride 4 --num-workers 8
  ```
  Then probe honest test rows with the produced checkpoint and `--split test`.
  Initial gate: test balanced accuracy ≥95%, recall ≥90%, XYZ MAE preferably
  <20 mm. Fail means do not bake/use tokens yet.
- **Not done:** Full 152-episode training. User runs command above. Do not treat
  smoke checkpoint as trained encoder. ACT token baking and ACT policy training
  remain after the quality gate.

---

## 2026-08-25 — start coauthor place-corridor convert / train / eval glue

- **When:** 2026-08-25, after the surface-encoder probe writeup. User asked to
  start tests on `data/pact_place_corridor_v5`.
- **Why:** Coauthor reports PACT beating ACT here. Reproduce locally. Probe
  already showed the skin is not empty; convert was still unrun.
- **What:**
  - `scripts/convert_pact_place_to_act.py`
  - `TASK_CONFIGS['pact_place_corridor_v5']` + 9/8 dims in `imitate_episodes.py`
  - `submodules/act/eval_act_place_corridor.py` (needs molmospaces worktree
    `977acd6`, scene XML `pact_place_corridor_v2`)
  - README §13.1, PACT.md §8
  - Gate-bar 200-ep collect still parked
- **How:** `--with_proximity --prox_pool min`, wrist 240×320, chunk 50, no
  image dropout. Never `imitate_episodes.py --eval` on PACT.
- **Not done at write time:** convert of 152, paste of `num_episodes` /
  `episode_len`, the two train jobs, eval.

---

## 2026-08-25 00:07 MDT — what the corridor probe actually found (plain language)

- **When:** 2026-08-25 00:07 America/Denver. Probe itself finished ~00:02 MDT
  (`experiments_output/default/surface_encoder_probe/pact_place_corridor_v5/`).
- **Why:** User asked for the findings in simple language, and a full log of this
  encoder session so later chats can pick it up without re-deriving.
- **What the probe is:** it does **not** grade a trained neural net (no
  `pact_surface_*_v1.pt` on this machine). It reads the 152 corridor episodes'
  raw skin depths and asks two questions the net would be trained on:
  1. How often is something within **20 cm** of a sensor? (geometry encoder
     target — nearest pixel XYZ, farther tiles are *invalid*, not a number.)
  2. How often is something within **50 cm**? (PACT-raw peak closeness.)
- **Headline numbers (152 eps, every 4th control step, last of 4 native subframes):**
  - **11.4%** of (timestep × sensor) tiles have a 20 cm hit. **83,608** such
    points. Typical nearest-surface distance: **9 cm / 16 cm / 19 cm**
    (10th / 50th / 90th percentile). So when the 20 cm encoder *does* fire, it
    is seeing mid-range skin, not a graze.
  - **40.3%** of tiles fire under the 50 cm PACT-raw cap. The extra ~29 points
    of percent are "I see something 20–50 cm away" — the geometry encoder
    **throws those away**.
  - Split: 72 left-intrusion / 80 right. Using one number per episode (max peak
    closeness, or 20 cm hit rate) to guess left vs right is **weak** (AUC 0.66
    and 0.61). Skin max closeness is not a side label.
- **The important catch — a lot of 20 cm signal is the robot seeing itself:**
  - `link1_sensor_5`: **100%** of probed steps have a 20 cm hit. Always on.
  - `link2_sensor_3`: **96%**. Almost always on.
  - Then a taper: `link2_sensor_0` 58%, `_6` 41%, `_4` 34%, `link5_back_sensor_4`
    30%, a few link6 sensors 12–16%. Many sensors are **zero** at 20 cm.
  - `link1_sensor_3` and `link1_sensor_4`: **74%** fire at 50 cm, **0%** at 20 cm.
    PACT-raw would keep those; the geometry encoder would output invalid/zeros.
    That is trap 16 on this dataset, not a hypothetical.
- **Untrained net (2 episodes, random weights, CUDA):** XYZ error **151 mm**,
  validity accuracy **12%** (random net mostly says "valid"; true rate is 11%).
  Reconstruction MSE 0.25. This only proves the forward pass runs on real
  `(T, 40, 4, 8, 8)` rows. **It is not evidence the encoder is good or bad.**
- **Simple verdict:** corridor skin is **not empty** inside 20 cm, so a trained
  geometry encoder would have something to chew. But a big chunk of that 20 cm
  signal is **always-on self geometry** on a couple of proximal sensors, and
  PACT-raw's 50 cm map lights up a lot of tiles the 20 cm encoder will ignore.
  Cannot say if the coauthor net is accurate until someone drops
  `pact_surface_embedding_encoder_v1.pt` (or the XYZ v1 file) and re-runs
  `python -m encoders.probe --checkpoint ...`.
- **Artifacts:**
  - `experiments_output/default/surface_encoder_probe/pact_place_corridor_v5/probe.json`
  - `sensor_hit_rates.png`, `valid_z_hist.png`, `episode_20_vs_50.png`
- **Not done:** trained XYZ MAE / validity / recon. Convert to ACT hdf5 not run
  (`act_style_data/` still empty for this task). `constants.py` still has
  `num_episodes=0` placeholders for `pact_place_corridor_v5`.

---

## 2026-08-24 ~23:55–00:02 MDT — encoder package, ACT wire-up, corridor probe tool

- **When:** 2026-08-24 evening through 2026-08-25 00:02 America/Denver (same
  chat as the probe). Earlier in the chat: compare coauthor vs PACT-raw,
  fold `amine/` into `encoders/`, delete `amine/`, then probe the HF corridor
  dump. This block is the code trail; the 00:07 block above is the science.
- **Why:** User wanted both encoders in one function-named folder, then the
  coauthor ACT drop files merged, then `data/pact_place_corridor_v5` run
  through the new encoder.
- **What (package):** `encoders/`
  - `peak_closeness.py` — PACT-raw. All 40 sensors, one snapshot `(B,40,8,8)` m.
    Cap **0.5 m**. Dead pixel `<5 mm → 0`. Headline: per-sensor peak closeness
    `(B,40,1)`. Alias `ProxCVAEEncoder`.
  - `surface_geometry.py` — coauthor conv+transformer (~0.82M), weight-shared
    across sensors. Wants **32 causal 8×8 frames** (8 control steps × 4 native
    60 Hz subframes). Cap **0.20 m**. v1 XYZ / v2 32-d embedding. Wrapper
    `SurfaceGeometryEncoder(kind='xyz'|'embedding')`. Inner weights always
    `requires_grad=False`. Transformer `enable_nested_tensor=False`.
    Helpers: `nearest_surface_target`, `nearest_surface_target_batch`,
    `causal_sensor_window`, `to_causal_closeness`, `as_subframe_episode`
    (repeats pooled `(T,S,8,8)` into fake `(T,S,4,8,8)`), `encode_episode` /
    `encode_episode_full` / `encode_episode_at_times`.
  - `__init__.py` — `load_encoder(name, checkpoint=..., device=...)`. Aliases
    `raw`/`xyz`/`embedding`. Geometry kwargs `layout`/`tokens_per_sensor` ignored.
  - `pact.py` — ACT glue: `build_pact_encoder`, `hdf5_proximity_layout`,
    `encode_for_act`, `is_geometry_feature`, `causal_pooled_window`.
  - `encode_tokens.py` — bake frozen tokens into ACT hdf5.
  - `probe.py` — `python -m encoders.probe --src data/pact_place_corridor_v5`.
  - `__main__.py` — `python -m encoders` dummy shapes.
- **What (ACT, from deleted `amine/`):**
  - `submodules/act/utils.py` `proximity_layout`: `raw` | `raw_causal`
    (last 8 pooled steps) | `embeddings` `(40,32)` | `positions` `(40,3)`.
  - `detr/models/detr_vae.py`: `prox_feat_dim` and alias `proximity_feature_dim`;
    kept this repo's `image_dropped` path.
  - `imitate_episodes.py`, `eval_act_obstacle.py` (8-step live history),
    `detr/main.py`, `attn_heatmap.py`. Geometry default **K=1** if argparse
    default was 8. Without `--prox_encoder_ckpt`, geometry net is frozen-random.
  - `submodules/act/prox_cvae.py` is a shim to `encoders.peak_closeness`.
- **What (probe run, 2026-08-25 ~00:02 MDT):**
  ```
  conda activate mlspaces
  cd /home/jaydv/code/prox_learning
  python -m encoders.probe \
      --src data/pact_place_corridor_v5 \
      --out experiments_output/default/surface_encoder_probe/pact_place_corridor_v5 \
      --device cuda
  ```
  Source rows already have native `(T, 4, 8, 8)` per sensor — no convert.
  Tests: `pytest tests/test_encoders.py tests/test_prox_raw.py` → 29 passed.
- **How (do not mix closeness maps):** peak closeness `D_MAX=0.5 m`; surface
  geometry `MAX_SURFACE_RANGE_M=0.20 m`. README trap 16. Dataset
  `Lundii/pact_place_corridor_v5` cloned at `data/pact_place_corridor_v5`
  (152 `rows/*/trajectory.h5`, wrist-only, scene XML name
  `pact_place_corridor_v2`). `recovery.json` has `training_authorized: false`
  / `conversion_authorized: false` — recovery metadata, not a code lock.
- **Not done:** frozen coauthor checkpoint still missing (searched disk, none).
  Hunt for `pact_surface_encoder_v1.pt` or `pact_surface_embedding_encoder_v1.pt`.
  Do not treat untrained XYZ/embeddings as a quality verdict.

---

## 2026-08-24 — gate-bar v3.0 too easy; v3.1 snaps a tall pole onto the TCP line

- **When:** 2026-08-24 ~23:30 America/Denver, after user watched
  `hybrid_gate_bar_check/.../20260824_231030` exo videos.
- **Why:** User: no pole in videos (correct — INVIS_P=1) but the pick is too simple
  to collect. Log bows were 1.6–7.2 cm around XML pegs (20–24 cm tall, 3.5 cm
  thick) that often missed the gripper line. Straight vanilla would not hit.
- **What:** Do **not** launch `FrankaSkinHybridGateBarConfig` on v3.0.
  v3.1: `GATE_HALF_Z=0.22` (44 cm), `gate_block` snaps inner face onto the live
  TCP at t=0.40, bow sign = `protr_wall` coin-flip, `geom_rbound` updated so the
  taller box actually collides. New
  `FrankaSkinHybridGateBarVisibleCheckConfig` (INVIS_P=0) so a human can see
  the pole in the doorway. README §12.2 rewritten.
- **How:** `enclosure_reach.py` Gate sampler + `ObstacleAwarePickPlannerPolicy._repose_gate_pole`.
  Visible check first, then invisible check, then 200.
- **Not done:** user runs visible check. Do not collect 200 until that passes
  (pole in doorway, DEFLECT ~18 cm, both signs, grasp still works).

---

## 2026-08-24 — dropped Safety-CVAE weights; PACT-raw does not need them

- **When:** 2026-08-24 ~22:20 America/Denver.
- **Why:** User: keep cvae_v3 only if tests prove it worth it, else delete and use
  new data. Honest call: **not worth it for PACT.** Raw never ran `model.pt`.
  Trunk/delta already lost. Reflex is a different method.
- **What:** `hybrid_skin_sensors.py` holds the 40-name order. `ProxCVAEEncoder`
  `feature=raw` loads no weights. Convert default no longer reads meta.json.
  Deleted `assets/safety/cvae_v{1,2,3}/`. Tests: `tests/test_prox_raw.py`.
- **How:** leftover `sweep_v*.h5` + reflex demo mp4/mcap still in `assets/safety/`.
  `scripts/safety_*_demo.py` will 404 until someone retrains a head.
- **Not done:** gate-bar collect.

---

## 2026-08-24 — user wiped datagen / ckpts / eval; leftover converted hdf5

- **When:** 2026-08-24 ~22:11 America/Denver.
- **Why:** User deleted data, ckpts, eval for a clean slate.
- **What is gone:** `assets/datagen/` (all source collections, including
  hybrid_obstacle_v1 and hybrid_invis_obstacle_v1), `submodules/act/ckpts/`,
  `eval_output/`. July and avoid-v1 **cannot** be reproduced from this disk.
- **What is still here (do not mix into the next run):**
  `act_style_data/obstacle_prox_v2` (918M) and `obstacle_prox_avoid_v1` (1.5G).
  `assets/safety/cvae_v3` — keep; PACT frozen encoder.
- **How / next:** gate-bar from zero (README §12.2). Do not reconvert avoid-v1.
  `obstacle_gate_v1` counts still 0 until convert. Wipe leftover hdf5 if they
  want the disk to match the story.
- **Not done:** preflight / collect / convert / train / eval.

---

## 2026-08-24 — avoid-v1 n=50 grid **complete** (visible PACT finished)

- **When:** `pact_raw_persensor_idrop03_s0_visible/eval_summary.json` 2026-08-24 15:22 MDT.
  Eval process gone; tmux idle at prompt.
- **Why:** User asked updates.
- **What:** last cell filled. Full n=50, `--temp_agg_off`, prox on/off checked:

| cell | vanilla succ / coll / strict | PACT succ / coll / strict | Fisher coll / succ |
|---|---|---|---|
| invisible | 21/50 42% / 20/50 **40%** / 14/50 | 12/50 24% / 15/50 **30%** / 7/50 | p=0.40 / 0.088 |
| free | 21/50 42% / 24/50 48% / 12/50 | 15/50 30% / 22/50 44% / 8/50 | p=0.84 / 0.30 |
| visible | 20/50 40% / 34/50 68% / 5/50 | 14/50 28% / 28/50 **56%** / 6/50 | p=0.30 / 0.29 |

Collision Δ all favor PACT (−10 / −4 / −12 pts). None significant. Success worse every cell. Verdict unchanged: **failed**.
- **Not done:** none on this grid.

---

## 2026-08-24 — avoid-v1 declared FAILED; gate-bar (v3) data design + per-body collision metric prepared

- **When:** 2026-08-24 afternoon (America/Denver), after the n=50 avoid-v1 grid missed
  the ≥15 pt / p<0.05 bar (40%→30%, p≈0.40) and lost success (42%→24%).
- **Why:** User: collect new data that makes PACT win honestly; set up data, model
  config, and every experiment; MVP results needed by tomorrow. Root cause of both
  failed rounds, verified in sampler code: (1) avoid-v1 trained on *visible* bars;
  (2) `_obj_rest` coupled the cup to the bar's side, so cameras could read the bar off
  the cup even in the invis set; (3) bar face 0.14–0.24 m off-center meant one
  "always bow" path cleared every bar — the 3× upsample taught it to vanilla.
- **What (parent repo):**
  - `README.md` — §12.2 gate-bar recipe (design, Monte-Carlo numbers, preflight
    criteria, convert/train/eval deltas), §12 config-table row, §6 `--eval_sampler`
    flag row, §16 decision-log entries (avoid-v1 FAILED + gate design).
  - `paper.md` — avoid-v1 numbers moved into do-not-claim + failed-experiments with
    the honest wording. 2026-07-05 grid stays the headline.
  - `CURSOR.md` — this entry.
- **What (`submodules/molmospaces`, dirty):**
  - `tasks/enclosure_reach.py` — `GateObstacleFumehoodPickSampler` (+`Check`):
    OBSTACLE_P=0.75, INVIS_P=1.0, signed BAR_FACE_Y=(-0.06,0.22), GATE_X_WORLD=
    (0.47,0.58), AP_W_RANGE=(0.66,0.85), `_obj_rest` fully decoupled (cup y=±U(0.08,
    0.14)). Expert unchanged (strict superset; bows fit with 0% waypoint clipping in a
    20k-draw sim; blind-policy best case still hits 36% of poles, mean path 57%).
  - `tasks/pick_task.py` — `_accumulate_obstacle_diag` now records contact BODY NAMES
    per step (`_obstacle_diag_bodies`; world-rooted geoms named by geom), episode log
    line gains `bodies=...`. Additive; datagen/eval behavior otherwise unchanged.
  - `data_generation/config/object_manipulation_datagen_configs.py` —
    `FrankaSkinHybridGateBarCheckConfig` (4 eps, viz on) +
    `FrankaSkinHybridGateBarConfig` (8×25=200 eps, 4 workers, `viz_sensor_rgb=False`
    so no v2-style OOM) → `assets/datagen/hybrid_gate_bar_v1`.
- **What (`submodules/act`, dirty):**
  - `eval_act_obstacle.py` — `--eval_sampler {invis,gate}` picks which check sampler
    provides `--eval_cell`; per-episode records gain `hit_bar` / `bar_contact_steps` /
    `contact_bodies`; summary gains `bar_hit_rate` / `nonbar_collision_rate` /
    `mean_bar_contact_steps`; eval_summary.json records `eval_sampler`; wandb table
    extended. Old ckpts/records stay compatible (missing keys default).
  - `constants.py` — `obstacle_gate_v1` task (counts = 0 placeholders until convert
    prints them); avoid-v1 entry annotated with its failed result.
  - `imitate_episodes.py` — `obstacle_gate_v1` added to the state_dim=9/action_dim=8
    task tuple (else it falls to the 14-dim default and crashes).
- **How verified:** all files py_compile; samplers+configs import in `mlspaces`
  (sampler class, INVIS_P, viz flag, output dir checked); `_apply_eval_cell` swaps to
  `GateObstacleFumehoodPickCheckSampler` and pins per-cell probs; summary math checked
  on synthetic records; 20k-draw Monte-Carlo of the gate geometry (deflect 80% of bar
  eps, need p90 26 cm, corr(cup y, pole y)=+0.02).
- **Not done (user runs, workflow constraint):** gate-bar preflight + 200-ep collection,
  convert (+ paste counts into `constants.py`), train vanilla + PACT at chunk 50 (NO
  image dropout on the headline arm), eval n=50 invisible+free with
  `--eval_sampler gate`, `compare_pact.py` on collisions AND bar-hits. PACT-visible
  avoid-v1 eval may still be running in tmux — let it finish or kill it; its result
  cannot change the avoid-v1 verdict.

---

## 2026-08-24 — n=50 avoid-v1 eval: 5/6 cells done; last job PACT-visible

- **When:** 2026-08-24 ~13:13 America/Denver. Grid started 2026-08-23 21:10.
- **Why:** User asked for updates.
- **What (from `eval_summary.json`, n=50, `--temp_agg_off`):**

| cell | vanilla succ / coll / strict | PACT-raw per-sensor succ / coll / strict |
|---|---|---|
| **invisible** (paper) | 21/50 (42%) / **20/50 (40%)** / 14/50 | 12/50 (24%) / **15/50 (30%)** / 7/50 |
| free | 21/50 (42%) / 24/50 (48%) / 12/50 | 15/50 (30%) / 22/50 (44%) / 8/50 |
| visible | 20/50 (40%) / 34/50 (68%) / 5/50 | **still running** (~17/50 at 13:14) |

Invisible collisions 40%→30% is **10 pts**, Fisher p≈0.40. Does **not** meet ≥15 pts / p<0.05. Success 42%→24% (p≈0.088). Free collisions similar (48% vs 44%) so not a statue on the crash axis; lift still worse. Old 66%→40% grid stays the published headline until they decide otherwise.
- **How:** PACT jobs `use_proximity=true`, `raw`/`per_sensor` live. Vanilla `prox=False`. Invisible `INVIS_P=1`.
- **Not done:** `pact_raw_persensor_idrop03_s0_visible` (~2 h left). Do not write visible-cell PACT numbers yet.

---

## 2026-08-23 — user away ~10–12 h; n=50 grid in tmux; paper from July-5 numbers

- **When:** 2026-08-23 ~21:13 America/Denver.
- **Why:** User leaving overnight to write the paper. Eval loop started ~21:10.
- **What:** six jobs serial: ACT/PACT × invisible/free/visible. First job live:
  vanilla `act_avoid_s0_invisible`. Full grid ~18 h → at +12 h expect some
  `eval_summary.json` files, maybe not all six.
- **How / when back:** glob `eval_output/{act_avoid_s0,pact_raw_persensor_idrop03_s0}_{invisible,free,visible}/eval_summary.json`. Paper cell = both **invisible** n=50. Do not paste avoid-v1 rates into `paper.md` until those exist. Paper text tonight = 2026-07-05 66%→40% grid only.
- **Not done:** any n=50 avoid-v1 number.

---

## 2026-08-23 — smoke eval OK (n=2, not a result)

- **When:** 2026-08-23 20:57–21:04 America/Denver.
- **Why:** User ran 2-rollout PACT invisible smoke; asked to watch for completion/issues.
- **What:** `eval_output/smoke_pact_invisible/eval_summary.json`. Wiring good:
  `eval_cell=invisible`, sampler `InvisibleObstacleFumehoodPickCheckSampler`,
  `OBSTACLE_P=1 INVIS_P=1`, `[InvisBar] geom group 4`, PACT `raw` / `per_sensor` /
  40 sensors / K=1, `proximity ON (40,8,8)`, `--temp_agg_off`, full 201-step chunks
  (not frozen). 0/2 lift, collisions 1/2. n=2 = noise.
- **How:** Harmless log noise: `cam_visible=True` is a physics raycast not RGB;
  wrist reset mismatch; D405 depth-range on RGB-D mp4s; ffmpeg 470→480 pad.
- **Not done:** n=50 grid, both arms, three cells.

---

## 2026-08-23 — avoid-v1 train finished; eval is next

- **When:** 2026-08-23 ~20:49 America/Denver. Both 2000-epoch jobs done (serial in tmux).
- **Why:** User asked what is next after logs looked done.
- **What (verified on disk, not a result yet):**
  - vanilla: `ckpts/obstacle_pact_avoid_v1/20260823_183123_act_avoid_s0/`
    `policy_best.ckpt` = epoch **1914**, min val **0.0402**. ~59 min train.
  - PACT: `ckpts/obstacle_pact_avoid_v1/20260823_193046_pact_raw_persensor_idrop03_s0/`
    `policy_best.ckpt` = epoch **1850**, min val **0.0438**. ~64 min train.
    `prox_config.json`: `raw` / `per_sensor` / `min` / `n_sensors=40` / K=1.
    Last-epoch `l1_img_dropped` 0.069 vs `l1_clean` 0.039 → image dropout actually hurt the dropped samples (good).
- **How / next:** eval both ckpts, three cells, n=50, `--temp_agg_off --eval_cell`. Auto-detect PACT from `prox_config.json`. Never `imitate_episodes.py --eval`. ~3.5 min × 50 × 6 ≈ 18 h serial (~41 GB RSS). Paper cell first (`invisible`). Then `scripts/compare_pact.py`.
- **Not done:** any rollout numbers. Do not update `paper.md` tables until `eval_summary.json` exists.

---

## 2026-08-23 — paper.md (paper-agent claim fence)

- **When:** 2026-08-23 ~18:36 America/Denver.
- **Why:** User asked for the paper-writing brief as a file, easier to copy-paste.
- **What:** new `paper.md` (allowed claims, do-not-claim list, method, dataset, traps,
  2026-07-05 tables, in-flight avoid-v1 flagged as unmeasured). README routing row +
  intro pointer. This log header now cites `paper.md`.
- **How:** dump of the chat brief; numbers from `STATUS.md` / `PACT.md` / README §14.
  Not a paper draft.
- **Not done:** actual paper text. Avoid-v1 eval still pending.

---

## 2026-08-23 — user started avoid-v1 ACT + PACT train in tmux

- **When:** 2026-08-23 ~18:26 America/Denver.
- **Why:** Headline comparison on collision-aware set (`obstacle_pact_avoid_v1`, 151 eps).
- **What (user, not agent):** two `imitate_episodes.py` jobs in tmux under `submodules/act`:
  1. vanilla ACT — `--wandb_run_name act_avoid_s0` (no proximity)
  2. PACT — `--use_proximity --prox_feature raw --prox_layout per_sensor --image_dropout_p 0.3 --prox_dropout_p 0.1 --wandb_run_name pact_raw_persensor_idrop03_s0`
- **How:** ckpts land in `submodules/act/ckpts/obstacle_pact_avoid_v1/<YYYYMMDD_HHMMSS>_<runname>/`. If both `python` lines run in **one** pane, PACT waits until ACT finishes (~2000 epochs). Two tmux windows = both GPUs/jobs in parallel (only if VRAM allows; otherwise keep serial).
- **Not done:** eval. After `policy_best.ckpt` exists: `eval_act_obstacle.py --temp_agg_off --eval_cell invisible`, n=50, report collisions.

---

## 2026-08-23 — convert avoid-v1 counts pasted into constants

- **When:** 2026-08-23 ~18:23 America/Denver. User ran convert; agent pasted printed counts.
- **Why:** `imitate_episodes.py` reads `TASK_CONFIGS['obstacle_pact_avoid_v1']['num_episodes']`. Left at 0, train would see an empty set.
- **What:** `submodules/act/constants.py` → `num_episodes=151`, `episode_len=140`.
- **How / result:** User command (visible-bar `hybrid_obstacle_v1` `20260612_183855`, not the invis run):

```
python -m scripts.convert_obstacle_to_act \
    --src assets/datagen/hybrid_obstacle_v1/FrankaSkinHybridObstacleConfig/20260612_183855 \
    --dst act_style_data/obstacle_prox_avoid_v1 \
    --with_proximity --prox_pool min \
    --skip_approach_collision --keep_deflect_collisions --upsample_deflect 3 \
    --image_h 240 --image_w 320
```

`convert_meta.json`: fail skip 25, collision skip 13 (non-deflect scrapes), not_deflect 0. Written **deflect=96** (32 unique × 3) + **free=55** = **151**. max T=138 → episode_len=140. `prox_pool=min`.
- **Not done:** train vanilla ACT + PACT-raw-per_sensor; eval invisible cell n=50. Optional later: same convert on `hybrid_invis_obstacle_v1` if the camera-blind cell should be *in training* too.

---

## 2026-08-23 — PACT audit dump, collision-aware convert, encoder layout

- **When:** 2026-08-23 afternoon–evening (America/Denver). Follow-on to the
  same-day visualizer / CVAE-writeup / PACT-audit chat.
- **Why:** Success rate does not move because BC copies demos that already
  complete the pick with the bar present. The paper number is **invisible-cell
  collision rate**. Published win is `--prox_feature raw` (66% → 40%
  collisions, Fisher p = 0.016). CLI default `trunk` *raised* collisions
  (66% → 72%). Dataset also teaches scrapes: ~40% of episodes have approach
  (arm-vs-env) contacts; deflect demos still mean ~5 contacts. User asked to
  (1) dump canvases + audit into `PACT.md`, (2) fix those shortcomings, (3)
  introspect the dataset, (4) add collisions in a way that shows PACT beats
  ACT at avoidance — **without** injecting random crashes into BC.

### What landed (uncommitted)

Parent repo:

| file | change |
|---|---|
| `PACT.md` | **new.** Canvases (Safety-CVAE + PACT audit), dataset table, convert/train/eval commands, “incredible” success criteria. |
| `CURSOR.md` | **new (this file).** Agent change log. |
| `README.md` | Route to `PACT.md`. Recipe C pointer. Flag table: default `--prox_feature raw`, new `--prox_layout per_sensor`, recommend `--image_dropout_p 0.3`. Decision-log bullet 2026-08-23. Audit “next experiments” shortened to the three high-leverage runs. |
| `scripts/convert_obstacle_to_act.py` | Collision-aware convert (see How). |

`submodules/act` (dirty submodule, not committed):

| file | change |
|---|---|
| `prox_cvae.py` | `resolve_prox_layout`, `per_sensor` vs `global`, min/mean substep pool on live stack. Default tap `raw`. Encoder returns `(B, n_act_sensors, act_feat_dim)`. |
| `imitate_episodes.py` | Defaults `raw` + `per_sensor`. Hard-exit `--eval` + `--use_proximity`. Writes `prox_layout` / `prox_pool` / `n_proximity_sensors` into `prox_config.json`. Reads `convert_meta.json` for pool. Task list includes `obstacle_pact_avoid_v1`. |
| `eval_act_obstacle.py` | Rebuilds encoder from `prox_config.json` layout (missing key → `global`, so old ckpts still load). Live stack uses saved `prox_pool`. |
| `attn_heatmap.py` | Same layout rebuild as eval. |
| `detr/main.py` | No-op argparse for `--prox_layout` / `--prox_pool` so DETR re-parse does not explode. |
| `constants.py` | New task `obstacle_pact_avoid_v1` → `act_style_data/obstacle_prox_avoid_v1`, `num_episodes=0` until convert prints counts. |

### How (implementation)

**Do not teach colliding.** Convert does not invent crash labels. It *filters*
and *reweights* existing planner demos.

`convert_obstacle_to_act.py`:

- `--prox_pool {mean,min}` — `min` keeps a 1-of-4 substep graze (old path was mean, 4× dilution).
- `--skip_approach_collision` — drop eps with arm-vs-env contacts in pregrasp/grasp (same phase split as `analyze_obstacle_dataset.py`: phases `{2,3}`, inbound = before gripper-close phase 4). Tags come from `obs_scene` JSON (`behavior_class`, `collision_metrics.per_step_contacts`).
- `--keep_deflect_collisions` — exception: keep `behavior_class=deflect` even if they grazed. Needed because p10 of deflect contacts is 0; dropping *all* contacts would leave ~5 bows.
- `--only_deflect` / `--upsample_deflect N` — optional pure-avoidance set / copy each deflect hdf5 N times.
- Writes `convert_meta.json` (counts, flags, `prox_pool`) next to `episode_*.hdf5`.

`ProxCVAEEncoder`:

- `layout=global` (old published raw): `(B, 1, 40)`, `n_proximity_sensors=1`, K=8.
- `layout=per_sensor` (new default, `raw` only): `(B, 40, 1)`, `n_proximity_sensors=40`, K clamped 8→1 so 40 tokens do not drown ~160 image tokens.
- Live `stack_obs_proximity(..., pool=)` matches convert.

`imitate_episodes.py --eval --use_proximity` now `SystemExit`. That path never passed `proximity_positions`. Real eval is `eval_act_obstacle.py --temp_agg_off`.

Old PACT ckpts: no `prox_layout` in `prox_config.json` → eval defaults `global` + `n_proximity_sensors=1`. Compatible.

### Dataset facts used (already on disk)

Source analysis: `diagnostics_output/obstacle_analysis/` from
`scripts/analyze_obstacle_dataset.py` on `assets/datagen/hybrid_obstacle_v1`
(151 eps).

| fact | value |
|---|---|
| Bar present | 75% (113 / 151) |
| Deflect / free | 49 / 102 (only 43% of bar eps actually bow) |
| Lateral bow, bar-deflect | mean 3.8 cm |
| Approach collision | **40% of episodes** |
| Approach contacts, bar-deflect | mean **5.0** |
| Skin close (<0.10 m), bar vs no-bar | 86% vs 74% (ambient saturation) |
| `act_style_data/obstacle_prox_v2` | still on disk, 105 eps (old train set) |

### Not done (user runs)

Convert / train / eval were **prepared, not executed** (repo workflow: agent
edits, user runs).

1. Convert avoid set (`PACT.md` §5). Paste `num_episodes` / `episode_len` into
   `constants.py` `obstacle_pact_avoid_v1`.
2. Train vanilla ACT and PACT-raw-per_sensor + `--image_dropout_p 0.3` on that set.
3. Eval n=50, `--temp_agg_off`, cells `invisible` / `free` / `visible`. Report
   **collisions**, not just lift-success.
4. Optional later: more **clean** deflect datagen (`approach_contacts=0`),
   `FrankaSkinHybridInvisObstacleConfig`. Residual `SafetyHead` at eval is a
   **different method** (ACT+reflex), not PACT.

`submodules/molmospaces` shows dirty in `git status`. Not part of this PACT
code pass (earlier visualizer work). Do not mix it into a PACT commit.

Nothing committed. No push.

---

## 2026-08-23 (earlier same day) — visualizer plates, CVAE write-up, PACT audit

- **When:** earlier 2026-08-23, before the `PACT.md` / convert-filter pass.
- **Why:** README needed the new visualizer behavior and paper-style plates.
  User then asked for a full CVAE description and a PACT wiring audit for
  “why no success-rate gain.”
- **What (already in tree / README before this log existed):**
  - `submodules/molmospaces/scripts/datagen/visualize_environment.py` — text-free
    1920×1080 plates, scored cameras, presentation headlight, `--attempts`,
    `--format both`.
  - README §12 hero/preview images under
    `experiments_output/default/environment_viz/...` (pngs gitignored; `git add -f`
    to publish).
  - README §10 Safety-CVAE tensors; §13 audit “why PACT success does not move.”
  - Canvases:
    `~/.cursor/projects/home-jaydv-code-prox-learning/canvases/pact-audit.canvas.tsx`
    and `safety-cvae-explained.canvas.tsx`.
- **How:** audit conclusion was wiring OK (metres, one featurize, no z-score on
  skin, sensor order from `cvae_v3/meta.json`). Method fights data: wrong
  metric, default `trunk` negative control, CVAE ⊥ BC, 40 sensors mashed to 1
  vector, one skin frame vs 100-step chunk, image dropout 0, temporal agg
  washes newest skin, mean-pool of 4 substeps.
- **Not done then:** `PACT.md` dump, convert filters, `raw`/`per_sensor`
  defaults — those are the block above.

---

## Backlog (do not lose)

- Paste convert counts into `obstacle_pact_avoid_v1` after first convert.
- Headline train: `--use_proximity --prox_feature raw --prox_layout per_sensor --image_dropout_p 0.3 --prox_dropout_p 0.1`.
- Eval: `eval_act_obstacle.py --temp_agg_off --eval_cell invisible`, n=50.
- Collect ≥150 clean deflect eps if reconverted set is too small.
- Optional third arm: ACT + residual SafetyHead at eval (not PACT).
- Do not commit `PACT.md` / `CURSOR.md` / act submodule together with unrelated
  molmospaces visualizer dirt unless that is the intended PR.
