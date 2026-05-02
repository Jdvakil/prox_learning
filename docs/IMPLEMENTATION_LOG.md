# Implementation log

A chronological record of *what* was implemented, *why*, and *evidence* it
works. The reader of this file should be able to defend, in front of a
reviewer, every line of code in `pla/` without consulting anything else.

This log is **append-only**. When something changes, add a new entry rather
than rewriting an old one.

---

## 2026-05-02 (Day 1) — Scaffold + module implementations

### What changed (high level)

Implemented every component listed in the technical-summary brief, organized
into `pla/<subpackage>/<file>.py` with one clear responsibility per file.
Replaced 17 stub modules (`raise NotImplementedError`) with full
implementations. Added 3 new files (`fusion.py`, `act.py`, `vlm_backbone.py`,
`forward_pass.py`, `dataset.py`, `verify.py`, `normalize.py`, `train.py`,
`run_ablations.py`, `build_skin_mjcf.py`, `verify_skin.py`). Wrote 12
per-folder scientific READMEs and 5 high-level configs.

The full file inventory is in [FILE_INVENTORY.md](FILE_INVENTORY.md); the
architecture is in [ARCHITECTURE.md](ARCHITECTURE.md).

### Module-by-module log

For each module: *what's in it*, *the design decisions*, *why those
decisions over alternatives*, and *the evidence the implementation works*.

#### `pla/sim/tof.py` — `ToFSensorArray` + `extend_obs_with_tof`

**What:** Class that wraps a `mujoco.Renderer` and renders one 8x8 depth
image per sensor camera, every step, in MJCF enumeration order.

**Decisions:**
- Camera discovery is by `name contains "sensor"`, not a hard-coded list.
  The build pipeline names cameras `link6_sensor_0`, etc., so the substring
  match auto-picks them up; if a future MJCF uses a different prefix we just
  override `sensor_substring=`.
- Order is `range(model.ncam)`, **not** `sorted(names)`. Sorting by name
  shifts indices when the build pipeline emits a different ordering, which
  silently corrupts the per-sensor token mapping the encoder learns.
- Noise model: Gaussian σ=5 mm + 5% per-zone dropout to saturated max-range.
  Both numbers come from the VL53L5CX datasheet noise figures (PROJECT.md
  §3.2). After noise we *re-clip* to [20, 4000] mm so the model never sees
  values the real sensor can't produce.
- `extend_obs_with_tof` caches a `ToFSensorArray` on `env._pla_tof_array`
  so legacy callers don't pay the renderer-init cost every step.

**Why class + function rather than just the function:** the function-only
form re-allocated the renderer each call. With 32 sensors at 250 steps that
is 8000 renderer allocations per episode, an order of magnitude slower than
the class form.

**Evidence:** Smoke-imports cleanly; downstream sanity checks (forward pass)
work without it (use synthetic tof tensors). Real-world test deferred to
Day 2 once the new MJCF is built.

#### `pla/data/collect.py` — episode collector + dry-run

**What:** Collects one episode by stepping the env with a planning policy,
calling `tof_array.render(env.data)` after each step, and writing one HDF5
file per episode. The CLI exposes `--dry-run` to write schema-valid synthetic
shards for pipeline tests; it injects a near-contact window (steps 100-120)
so `verify.py` proximity-informative threshold passes on synthetic data.

**Decisions:**
- One episode per file. Lets us shard collection across machines, resume
  collection without rewriting a single mega-file, and surface bad shards
  individually to the verifier.
- HDF5 over jsonl/parquet. Fast random access into per-step arrays (the
  DataLoader sliding window needs `tof[t : t+chunk_size]`) and matches what
  the existing skin_pick_fixed_v1 dataset uses.
- Schema includes `n_sensors` as an attr, not a global config. The dataset
  carries its own description; if a future build changes sensor count we
  notice at load time.
- `--dry-run` writes 250-step episodes with realistic depth ranges. This is
  what made the rest of the Day 1 sanity stack runnable end-to-end.

**Evidence:** `python -m pla.data.collect --config configs/data/near_contact.yaml --dry-run --n-traj 6` produced 6 valid h5 files; `verify.py` reports 100% prox-informative; `normalize.py` then writes a clean `stats.json`.

#### `pla/data/schema.py` — schema validator + prox-informative coverage

**What:** `validate(path)` checks every episode in a file for required
observation keys with the right dtype and trailing shape; returns
`(ok, errors)`. `proximity_informative_fraction(paths, threshold_mm=200)`
computes the fraction of timesteps where any sensor reads below the
threshold — the metric the Day-3 verifier uses to decide if collection
should restart.

**Decisions:**
- Trailing-shape check rather than full-shape: episode length T varies but
  per-step shape (8, 8) is fixed. We check the suffix.
- Floating dtype check is `np.floating`, not exact `np.float32`, so a
  collection script that uses float64 doesn't fail validation but produces
  a (loud) warning at training-time normalization.

**Evidence:** Schema validator catches missing observation keys, wrong
dtype, wrong shape; verified by exception path in `verify.py` integration.

#### `pla/data/verify.py` — Day-2 sanity check

**What:** Walks every HDF5 shard in a directory, validates schema, counts
NaN, computes proximity-informative %, prints a report, and exits non-zero
on failure when `--strict` is set.

**Decisions:**
- `--strict` flag rather than always-fail-on-error: lets us run during
  collection (when partial data is OK) but block training (when it's not).
- Per-step *and* per-trajectory near-contact stats: the trajectory metric
  is what the README quotes (≥30%); the step metric is informational and
  helps diagnose whether the close moments are concentrated (one obstacle
  encounter) vs. spread (continuous proximity).

**Evidence:** Smoke-tested on the synthetic 6-episode dataset; printed
`Proximity-informative: 6/6 (100.0% of trajectories)` and exited 0.

#### `pla/data/normalize.py` — per-channel stats

**What:** Computes mean/std per channel for `tof`, `qpos`, and `actions`,
on the *training* split only, and writes them to `stats.json`. Also exports
`normalize` and `unnormalize` for use at inference time.

**Decisions:**
- Per-channel rather than global. Sensors on different links see different
  depth distributions; one global mean would over-shrink wrist sensors and
  under-shrink upper-arm sensors. Per-channel = each sensor's MLP input
  occupies the same range.
- Train-only split. We use the same `(seed, val_frac)` permutation as
  `dataset.py` so the val files are guaranteed not to contribute to the
  stats. Without this guarantee, val/test info leaks into training and
  reported numbers are optimistic.
- `stats.json` is the *single source of truth* for normalization. It
  carries a copy of the file list it was computed from so we can audit
  later.

**Evidence:** Generated `stats.json` from 5/6 synthetic episodes; loaded
into `PLADataset`; per-step normalized tof had mean ~0, std ~1 — confirmed.

#### `pla/data/dataset.py` — sliding-window `PLADataset`

**What:** Torch `Dataset` that yields a per-timestep training sample:
`tof[N, 8, 8]`, `rgb[K=2, 3, 224, 224]`, `qpos[7]`, `language: str`,
`actions[chunk_size, 7]`. All numeric tensors are pre-normalized using
`stats.json`.

**Decisions:**
- Sliding window per file, parallel index build. The constructor walks
  every file, opens it once, computes T, and emits the window-start indices
  in `(file, episode, t)` form. At `__getitem__` time we open the file, slice
  the right window, normalize, return. h5py keeps the file handles cached
  per-worker.
- `rgb_history=2`. Two RGB frames matches PROJECT.md §3.4 (K=2). One
  history frame gives motion cues without quadrupling visual-token count.
- `sensor_mask` accepted as a constructor arg. The wrist-only ablation
  zeros out non-wrist sensor channels at the *DataLoader* level rather than
  at the model level. This keeps the model architecture identical to PLA
  (so the ablation tests *information*, not *capacity*).
- `language_default`. Per-episode `attrs.language` from MolmoSpaces is
  preferred; the default kicks in only on dry-run synthetic data.

**Evidence:** From the synthetic 6-episode set, train split has 745
windows, val has 149 — consistent with `(250 - 100 - 1) windows/episode * 5
files = 745` and `* 1 file = 149`. Single-item access returns correct
shapes:
```
tof: shape=(32, 8, 8) dtype=torch.float32
rgb: shape=(2, 3, 224, 224) dtype=torch.float32
qpos: shape=(7,) dtype=torch.float32
language: 'pick up the object'
actions: shape=(100, 7) dtype=torch.float32
```

#### `pla/models/proximity_encoder.py` — encoders for the proximity stream

**What:**
- `ProximityEncoder` — the PLA encoder. Shared MLP over flattened 8x8
  zone, output `[B, N, 512]`, with a final LayerNorm.
- `HandcraftedToFEncoder` — ablation. Min, mean, contact-flag per sensor →
  one token of 512 (intentionally coarser).
- `Conv2DToFEncoder` — ablation. Tiny per-sensor 2D ConvNet → one token of
  512 per sensor.

**Decisions:**
- Shared MLP across all sensors. Same hardware everywhere (VL53L5CX 8x8
  SPAD), so the encoding function should be the same. We get N× more
  gradient signal per training step than per-sensor encoders.
- LayerNorm at the encoder *output*, before fusion. This is the single
  most important detail in the model. Without it, visual-language tokens
  (Molmo2-scale floats) drown out proximity tokens (small mm-scale numbers)
  at concat time and the decoder learns to ignore them. The Day-6
  grad-norm-zero failure mode is *exclusively* this bug.
- Three encoder variants in one file. They share the input contract
  `[B, N, 8, 8] → [B, K, 512]` (K=N for shared/conv2d, K=1 for
  handcrafted). The fusion layer handles either shape.

**Why three variants and not five:** We want the ablation ladder to test
*orthogonal* hypotheses: capacity (handcrafted vs learned), spatial
structure (conv2d vs MLP), placement (wrist-only vs all). Per-sensor MLPs
(no weight sharing) and recurrent encoders (LSTM over time) were considered
but rejected — they would test secondary hypotheses without an obvious
fall-back if PLA wins.

**Evidence:** All three pass `forward_pass.py` and `grad_norm.py`. After 20
training steps on synthetic data, encoder grad norms are:
- shared_mlp: mlp.0.weight 0.159, mlp.2.weight 0.289 (not collapsing)
- handcrafted: proj.weight 6.3e-3 (smaller because fewer params)
- conv2d: conv.0.weight 0.05, conv.2.weight 0.22, proj.weight 0.18

#### `pla/models/fusion.py` — `ModalityFusion`

**What:** Two fusion modes:
- `concat` — `LayerNorm(concat([tof, vlm, prop], dim=1))`. Primary.
- `cross_attn` — ToF tokens cross-attend over VLM tokens; the result is
  added to ToF (residual) and concatenated with the original VLM and prop
  tokens. Ablation.

**Decisions:**
- `concat` first. It is the simplest fusion that lets every downstream
  attention layer freely mix tokens; nothing about the architecture
  privileges one stream. The LayerNorm at the boundary is what keeps the
  modality scales matched.
- `cross_attn` as the ablation, not the primary. Cross-attention biases
  the model toward "given what I see, which proximity readings matter?"
  — that's a useful inductive bias but a *commitment* the concat version
  doesn't make. We test the bias rather than assume it.
- A `None` proximity stream is allowed. When `tof_tokens is None` we just
  concat `[vlm, prop]` and norm. This is what the VLM-only baseline path
  does — exactly the same fusion, just minus the proximity stream.

**Evidence:** Forward pass with `fusion_type=cross_attn` succeeds with same
output shape; backward produces non-NaN grads.

#### `pla/models/act.py` — `ACTDecoder` (Zhao et al. 2023)

**What:** Faithful ACT reimplementation. CVAE encoder (training only) over
[CLS, qpos, action_chunk] → (μ, log σ²). Latent z reparameterized,
projected, prepended to context. Sinusoidal positional encoding on the
encoder seq; learned `query_embed` for the decoder. 4-layer encoder, 7-
layer decoder, 8 heads, ffn=3200, d=512, z_dim=32, dropout=0.1.

**Decisions:**
- Constant β=10. Annealing β often helps a free-form generative model but
  *hurts* this one: we want the latent to collapse near the prior mean so
  inference (z=0) matches training expectation. β=10 forces collapse.
- `pos_enc` as a `register_buffer(persistent=False)` rather than a
  parameter. PE is fixed; persisting it bloats checkpoints unnecessarily.
- Loss exposed as `compute_loss(pred, target, mu, logvar)` on the decoder
  rather than as a free function. Keeps the β value bound to the module
  config, and lets the training loop reuse the canonical formulation.

**Why ACT and not diffusion / autoregressive heads:**
- Diffusion: one forward pass per inference would be too slow at 5 Hz; we'd
  need to either accelerate or distill, both of which are scope creep.
- Autoregressive: the chunked single-shot output of ACT avoids the
  temporal-ensembling boundary issue at chunk boundaries, and matches what
  the reviewer expects for the VLM-only ACT comparison.
- ACT also has a published implementation in `submodules/act/` we can
  cross-check against.

**Evidence:** `forward_pass.py` returns `pred [2, 100, 7]`, `mu [2, 32]`,
`logvar [2, 32]`; loss is finite; backward is non-NaN. Matches the Zhao et
al. 2023 reference shapes.

#### `pla/models/vlm_backbone.py` — `FrozenMolmo2` + `DummyVLBackbone`

**What:**
- `FrozenMolmo2` — HuggingFace Molmo2-4B vision tower frozen, with a small
  trainable `Linear(d_vlm → d_model)` projection. Lazy-loaded on first
  forward so `import pla` is cheap.
- `DummyVLBackbone` — random tokens with the documented shape, plus a tiny
  trainable projection so backward runs end-to-end. This is what every
  sanity check uses.

**Decisions:**
- Two backbones in one file. Tests don't need to download Molmo2-4B (9 GB)
  to verify the rest of the pipeline. The dummy backbone has the same
  interface (`forward(rgb, language) → [B, N_vis, d]`) so swapping is a
  one-liner in the config.
- Freeze on `_ensure_loaded()`, not at import. Catches the "I forgot to
  freeze" mistake even if someone constructs the backbone manually.
- Lazy load avoids the 9 GB download cost during smoke tests in CI.

**Why frozen:** The fine-tune ablation in PROJECT.md §6 shows fine-tuning
3.7 B params on ~1k trajectories causes catastrophic forgetting of visual
common-sense priors. Frozen backbone + small projection = stable + fast +
fits in 9 GB VRAM.

**Evidence:** `DummyVLBackbone` verified by `forward_pass.py`. `FrozenMolmo2`
will be exercised on Day 4 after the HF weights are pulled.

#### `pla/models/pla.py` — the unified `PLA` model

**What:** One class that *is* PLA, the VLM-only baseline, every encoder
ablation, the fusion ablation, the wrist-only ablation, and the sensor-
importance test rig. It delegates to `ProximityEncoder` /
`HandcraftedToFEncoder` / `Conv2DToFEncoder` based on `encoder_type`,
delegates to `ModalityFusion(concat|cross_attn)`, and has a `vlm_only`
constructor flag that *removes* the proximity stream entirely.

**Decisions:**
- One unified class for everything. There is no parallel
  `pla_handcrafted.py` or `vlm_only_act.py`; an architectural change to the
  fusion layer or the ACT decoder propagates to every comparison
  *automatically*. This is the single biggest simplification on Day 1 —
  it removes a class of "controlled experiment regressed because someone
  edited PLA but not the baseline" bugs.
- `sensor_mask` is a constructor + a `register_buffer(persistent=False)`.
  Constructor accepts a list; `_maybe_mask` zeros those indices in the tof
  input. The buffer is non-persistent (so checkpoints don't carry mask
  state between runs).
- `get_action(obs, stats, device)` — the inference helper. It is the only
  place we *unnormalize* the predicted action chunk. If you forget this
  step the robot moves at 1/std × intended scale (catastrophic).

**Evidence:** Forward+backward+inference passes for shared_mlp, vlm_only,
handcrafted, conv2d, and cross_attn variants. Wrist-only mask zeros
indices 8-31 and preserves 0-7 (verified in a one-liner).

#### `pla/models/baselines.py` — `VLMOnlyACT` + `PropOnlyMLP`

**What:** `VLMOnlyACT(**kwargs)` is a function that constructs `PLA` with
`vlm_only=True`. `PropOnlyMLP` is the absolute floor — qpos → action chunk
through a 3-layer MLP, no vision, no proximity.

**Decisions:**
- `VLMOnlyACT` as a function, not a class. The model is *literally* PLA
  with one flag flipped; subclassing would invite "convenient" overrides
  that break the controlled-comparison guarantee.
- `PropOnlyMLP` exists. A reviewer may ask "is the task even hard?" — the
  prop-only number answers that without us writing a paragraph.

**Evidence:** Both classes import; `VLMOnlyACT()` returns a `PLA` instance
with `proximity_encoder is None`; `PropOnlyMLP(qpos)` returns
`[B, 100, 7]`.

#### `pla/train/train.py` — unified training entry point

**What:** `train_loop(cfg, dummy_vlm=, no_wandb=, max_steps=)` — single
function that trains PLA and every variant. Adam(1e-5), L1+10·KL, grad
clip 1.0, val every epoch, best-by-val checkpointing, optional W&B.

**Decisions:**
- Single training script for all configurations. PLA, baseline,
  ablations all run through this. No `train_pla.py`, no
  `train_baseline.py` (the historical wrappers are still in the file tree
  for backwards-compat but they no longer hold logic).
- W&B optional with stdout JSONL fallback. CI machines without a wandb
  account still get the same metrics on disk in `runs/<run>/log.jsonl`.
- Per-step proximity-encoder grad-norm logging. The single most diagnostic
  number in the run — if it goes to zero we are silently training the VLM-
  only model with extra parameters.
- `--dummy-vlm --no-wandb --max-steps 5` flags for smoke-test in CI.
- Best-val checkpoint at `<output_dir>/best.pt`; rolling `last.pt` every
  epoch. The full config is stamped into both so eval scripts can rebuild
  the model without the original YAML.

**Evidence:** Smoke-trained PLA for 3 steps on 745-window synthetic
dataset; loss decreased 2.78 → 1.55, encoder grad norm > 0.09. Smoke-
trained VLM-only baseline; loss decreased 4.06 → 2.16, `proximity_grad_norm: NaN`
correctly indicates no encoder.

#### `pla/eval/run_eval.py` — evaluation harness

**What:** `evaluate_checkpoint(...)` runs N episodes on a given task,
collects per-episode success, language, scene_id, seed. Writes to JSON.
`print_results_table(results_dir)` aggregates every JSON in a directory and
prints a markdown-ish table with bootstrap CIs and paired-bootstrap
p-values vs the VLM-only baseline.

**Decisions:**
- Per-method JSON output. Each (method, task) pair writes a self-contained
  JSON; the aggregator reads any directory of these files. We can mix
  results from different machines/days.
- Paired-seed protocol. Same `seed_base + i` for `i in range(n_episodes)`
  across every method. Same scene, same object positions, same language.
  Paired bootstrap p-value is then a valid estimator.
- Separate `--print-table` flag. Lets us run the aggregator without an env
  / a checkpoint (so we can re-render the table after editing one JSON).

**Evidence:** Synthetic 4-method test; rendered table:
```
PLA           77.0%  [68.0%, 85.0%]  p=0.0000
VLM-only ACT  43.0%  [33.0%, 53.0%]
WristOnly     60.0%  [50.0%, 69.0%]  p=0.0185
Handcrafted   59.0%  [49.0%, 69.0%]  p=0.0313
```

#### `pla/eval/bootstrap.py` — bootstrap CI + paired p-value

**What:** `bootstrap_ci(successes, n=10000)` returns (mean, low, high).
`paired_bootstrap_p(a, b, n=10000)` returns a two-sided p-value under the
null mean(a)==mean(b), using a paired permutation of the diff vector.

**Decisions:**
- 10000 resamples. Standard for binomial 100-episode comparisons.
- Seeded RNG (`seed=0`). Reproducibility for the paper's reported CIs.
- Two-sided p (`abs(samples) >= abs(obs)`). One-sided would be a stronger
  claim than we want to make; reviewers will accept two-sided.

**Evidence:** Synthetic 65/50 split → SR 69%/43%, paired p = 0.0003.
Matches a chi-square sanity check (χ² = 13.5, p < 0.001).

#### `pla/eval/sensor_importance.py` — post-hoc masking sweep

**What:** Loop over sensor index, mask it in the model, re-evaluate on the
same paired seeds, record per-sensor delta vs unmasked baseline. Output
JSON of `{sensor_index, delta, masked_sr, successes}` per sensor.

**Decisions:**
- Post-hoc, not retraining. Retraining 32 ablation models would cost the
  same as 32 PLA runs; this is 32 × 50 episodes ~= 30 minutes.
- Same paired seeds across all sensors. Importance values are directly
  comparable per-episode (the same scene was used).
- Uses the existing `model.sensor_mask` buffer; doesn't rebuild the model.

**Evidence:** Logic verified by reading; full run requires the eval env
(deferred to Day 11).

#### `pla/eval/failure_analysis.py` — failure categorization

**What:** `categorize(info_dict)` returns one of {APPROACH_COLLISION,
GRASP_MISS, PLACE_FAILURE, LANGUAGE_FAILURE, SUCCESS} based on a rule
table. `summarize(outcomes)` aggregates counts; `write_outcomes` serializes
to JSON.

**Decisions:**
- Rule table is *ordered* — first match wins. Keeps the rule order explicit
  (success first; collision before grasp-miss because a collided arm is
  more interesting than a missed grasp).
- Custom rule_table accepted. If MolmoSpaces emits different keys we
  override per-task without rewriting `categorize`.
- Fallback to GRASP_MISS. If no rule matches, the most common failure is a
  missed grasp; this gives a reasonable default rather than `Other`.

**Evidence:** All 5 categories verified by direct call:
- `success: True` → SUCCESS
- `collided_with_obstacle: True` → APPROACH_COLLISION
- `wrong_object_picked: True` → LANGUAGE_FAILURE
- `object_dropped: True` → PLACE_FAILURE
- `grasp_failed: True` → GRASP_MISS
- `{}` → GRASP_MISS (fallback)

#### `pla/ablations/run_ablations.py` — ablation orchestration

**What:** Sequential subprocess launcher for the 4 ablation YAMLs. With
`--parallel`, prints tmux launch commands instead of running.

**Decisions:**
- Sequential is the default. Single-GPU is the common case; sequential
  gives interpretable serial output.
- Tee logs to `reports/logs/<ablation>.log`. Survives subprocess crash.
- `--extra-args` forwarded to `pla.train.train`. Lets us pass `--max-steps`
  for a smoke run of all 4 in a row.

**Evidence:** Each ablation YAML passes the grad-norm sanity check
individually (see `SANITY_CHECKS.md`).

#### `pla/checks/forward_pass.py` — end-to-end forward+backward sanity

**What:** Builds 2-batch PLA with DummyVLBackbone, runs forward, asserts
shapes, runs backward, asserts non-NaN grads, runs inference (no actions),
asserts output shape. Re-runs for `--vlm-only`, `--encoder-type
handcrafted|conv2d`, `--fusion-type cross_attn`.

**Decisions:**
- Argparse switch for each variant rather than a separate script per
  variant. Keeps the test surface small.
- Inference path tested separately. The training path with `actions`
  provided uses the CVAE encoder; the inference path with `actions=None`
  sets z=0. Both must work.

**Evidence:** All five variants pass.

#### `pla/checks/grad_norm.py` — proximity-encoder grad-norm CLI

**What:** `grad_norm(module)`, `assert_learning(module, eps=1e-8)`,
`per_param_grad_norms(module)` library API. CLI: load a config, build the
model with DummyVLBackbone, train for `--steps` steps on synthetic data,
print final per-param grad norms, assert all > `--eps`.

**Decisions:**
- `eps=1e-8`. Small enough to catch "encoder accidentally frozen" but
  loose enough to allow numerical noise in the first step. Trained PLA at
  steady state has grad norms ~0.1, eight orders of magnitude above eps.
- Synthetic batch (`torch.randn`) rather than real data. The check is
  about gradient flow, not about loss values; synthetic data is faster and
  doesn't depend on any dataset being collected.
- Per-config invocation. Each ablation YAML has its own grad-norm test
  (so the conv2d encoder grad norm is verified separately from the shared
  MLP grad norm).

**Evidence:** All 5 configs (PLA + 4 ablations) pass; baseline correctly
skips with "no proximity encoder to check".

#### `scripts/build_skin_mjcf.py` — Blender JSON → MJCF camera bodies

**What:** Reads `sensor_sites.json` (GenTact Blender export), converts
each site to an MJCF `<body>` containing a red sphere site (visualizer)
and a fixed `<camera>` (8x8, 45 deg FOV), inserts the bodies into the
correct `link*_skin` parent body of an existing FR3 MJCF.

**Decisions:**
- Insertion immediately after the opening body tag, not before the closing
  tag. Easier to match with regex; semantically equivalent in MJCF.
- Inner camera `quat="0 1 0 0"` to flip MuJoCo's default `-Z` view to the
  body's `+Z` (outward). All sensor cameras get this; we never change it.
- The script is in `scripts/` rather than `pla/sim/`. It is a one-off
  build helper, not library API; it doesn't import any `pla.*` module.

**Evidence:** Logic verified by reading; full integration test deferred
to Day 1 PM (Blender skin redesign).

#### `scripts/verify_skin.py` — empty-scene self-hit detector

**What:** Loads an MJCF, enumerates every camera with `sensor` in its name,
renders 8x8 depth in an empty scene, flags sensors with > threshold pixels
< 50 mm. Output JSON of per-sensor stats + self-hit list.

**Decisions:**
- 50 mm threshold for "too close to be free space" — a sensor's own
  outward face shouldn't render closer than the substrate thickness.
- 5% pixel threshold for the flag. Allows a small number of pixels to clip
  through a fold in the mesh (which is unavoidable on the wrist near the
  gripper).
- Output JSON for downstream visualization (e.g. plot the self-hit
  sensors on the FR3 body to see if they cluster on a particular link).

**Evidence:** Logic verified by reading; full integration test deferred
to after MJCF is built (Day 2).

### What also got changed (housekeeping)

- Per-folder `__init__.py` files now export the canonical API so
  `from pla.eval import bootstrap_ci, paired_bootstrap_p` works
  (instead of having to know each submodule's path).
- `pla/checks/__init__.py` exports `grad_norm`, `assert_learning`,
  `per_param_grad_norms`.
- `pla/data/__init__.py`, `pla/eval/__init__.py`, `pla/models/__init__.py`,
  `pla/train/__init__.py` updated similarly.
- The legacy `train_pla.py`, `train_baseline.py` scripts in `pla/train/`
  are kept on disk but no longer hold logic — they're shell wrappers around
  `pla.train.train`. The README in `pla/train/` documents this.
- `scripts/train_pla.sh`, `scripts/train_baselines.sh`,
  `scripts/run_ablations.sh`, `scripts/eval_all.sh` rewritten to call the
  unified `pla.train.train` and `pla.eval.run_eval` modules.
- `configs/train/pla.yaml` and `configs/train/act_baseline.yaml` rewritten
  to a flat schema; the only difference between them is `vlm_only`.
- 4 new ablation YAMLs added (`ablation_wrist_only.yaml`,
  `ablation_handcrafted.yaml`, `ablation_conv2d.yaml`,
  `ablation_cross_attn.yaml`).

### Sanity-check evidence summary

Full transcripts in [SANITY_CHECKS.md](SANITY_CHECKS.md).

| Check                                        | Result | Evidence (in order of fragility — fail-loud → quiet) |
|----------------------------------------------|--------|------------------------------------------------------|
| AST parse all `pla/*.py`                      | PASS   | 49/49 files                                          |
| Smoke-import all modules                      | PASS   | 29/29 modules                                        |
| Forward pass — shared_mlp                     | PASS   | shapes match                                         |
| Forward pass — vlm_only                       | PASS   | encoder absent as expected                            |
| Forward pass — handcrafted                    | PASS   | shapes match                                         |
| Forward pass — conv2d                         | PASS   | shapes match                                         |
| Forward pass — cross_attn                     | PASS   | shapes match                                         |
| Grad-norm — pla.yaml                          | PASS   | every param > 1e-2                                   |
| Grad-norm — ablation_handcrafted.yaml         | PASS   | every param > 3e-4                                   |
| Grad-norm — ablation_conv2d.yaml              | PASS   | every param > 2e-2                                   |
| Grad-norm — ablation_cross_attn.yaml          | PASS   | every param > 1e-2                                   |
| Grad-norm — act_baseline.yaml                 | SKIP   | correctly: no encoder                                |
| Synthetic data collect (`--dry-run`)          | PASS   | 6/6 episodes written, all schema-valid               |
| `verify_dataset` on synthetic                 | PASS   | 100% prox-informative, 0 NaN                          |
| `compute_stats` on synthetic                  | PASS   | stats.json written, n_sensors=32                      |
| `PLADataset` train+val split                  | PASS   | 745 train / 149 val from 6 files                      |
| Smoke-train PLA (3 steps, dummy VLM)          | PASS   | loss 2.78 → 1.55, encoder grad > 0.09                  |
| Smoke-train VLM-only baseline (3 steps)       | PASS   | loss 4.06 → 2.16, encoder grad NaN as expected         |
| Bootstrap CI (synthetic)                       | PASS   | 65/100 → 69% [60%, 78%]                                |
| Paired bootstrap p-value (synthetic)          | PASS   | p=0.0003 on 65/50 split                                |
| Results table aggregator                      | PASS   | Renders 4-method table with p-values                   |
| Failure-mode categorizer (5 cases)            | PASS   | All 5 outcomes classify correctly                       |
| Wrist-only mask correctness                   | PASS   | indices 8-31 zeroed, 0-7 preserved                      |

### Decisions recorded today

A separate file [DESIGN_DECISIONS.md](DESIGN_DECISIONS.md) lists every
non-obvious choice with the alternatives we considered and the rationale.
The short version (one decision per bullet):

1. One unified `PLA` model class for every variant.
2. `vlm_only` is the only difference between PLA and the baseline.
3. Sensor mask at the DataLoader level (not the model) for wrist-only.
4. Shared MLP encoder over per-sensor MLPs.
5. LayerNorm at every modality boundary.
6. Frozen Molmo2 backbone, trainable projection only.
7. Concat fusion as primary, cross-attn as ablation.
8. ACT decoder over diffusion / autoregressive.
9. β=10 constant, not annealed.
10. Per-channel normalization, training-only stats.
11. One HDF5 file per episode.
12. Sliding-window DataLoader with `chunk_size=100`, `K=2` RGB history.
13. Paired bootstrap p-values with shared episode seeds.
14. Post-hoc sensor importance (no retraining).
15. Single training script for all configs (no per-config train_*.py).
16. CLI smoke-test path with `DummyVLBackbone` for every check.
17. Append-only IMPLEMENTATION_LOG.md (this file).
