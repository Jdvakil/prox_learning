# Design decisions

Every non-obvious choice in the codebase, with: the alternative we
considered, the rationale, and the consequences (good and bad). This
document is meant for a reviewer or a co-author who asks "why is the
codebase shaped this way?".

For the *what* (shapes, interfaces) see [ARCHITECTURE.md](ARCHITECTURE.md).
For the *when* (how decisions arose chronologically) see
[IMPLEMENTATION_LOG.md](IMPLEMENTATION_LOG.md).

---

## D1. One unified `PLA` model class for every variant

**Choice:** A single `pla.models.PLA` class implements PLA *and* every
ablation. Variants are selected at construction time via:
- `vlm_only: bool` — disables the proximity stream entirely.
- `encoder_type: 'shared_mlp' | 'handcrafted' | 'conv2d'`.
- `fusion_type: 'concat' | 'cross_attn'`.
- `sensor_mask: list[int]` — zero those sensor indices at the encoder input.

**Alternatives considered:**
- (a) Separate classes (`PLA`, `PLAHandcrafted`, `PLAConv2D`, `VLMOnlyACT`).
- (b) Subclasses (`class VLMOnlyACT(PLA): ...`).

**Why this:**
- Architectural changes to PLA propagate to every variant *automatically*.
  When we add a tweak to the fusion layer next week, every ablation
  inherits it.
- The "controlled experiment regressed because PLA was edited but the
  baseline wasn't" failure mode is structurally impossible.
- The YAML configs differ by **one flag**. A reviewer can verify the
  controlled-comparison guarantee by `diff configs/train/pla.yaml configs/train/act_baseline.yaml`.

**Consequences:**
- (+) Verified: `diff` between `pla.yaml` and `act_baseline.yaml` shows
  only `run_name`, `output_dir`, and `vlm_only`. (See SANITY_CHECKS.md §10
  for the parameter-count proof.)
- (-) The `PLA.__init__` signature is wider than a single-purpose class
  would be. Mitigated by named-only kwargs and config-driven instantiation.

---

## D2. `vlm_only` is the *only* difference between PLA and the baseline

**Choice:** PLA and the VLM-only ACT baseline share the same backbone, the
same ACT hyperparameters, the same chunk size, the same dataset, the same
seed. Only `vlm_only: true|false` changes.

**Alternatives considered:**
- (a) Different per-config hyperparameter tuning ("the baseline benefits
  from a different learning rate").
- (b) Different chunk size for the baseline ("ACT vs PLA-ACT").

**Why this:**
- Any per-config tuning would let a reviewer ask "did you cherry-pick
  hyperparameters for PLA?". Answer is "no, we used the *same* config
  with one flag flipped".
- Any chunk-size difference would conflate the proximity-vs-no-proximity
  test with a chunk-size test.

**Consequences:**
- (+) Strongest possible controlled comparison.
- (-) The baseline might *underperform its own paper* if Zhao et al. 2023
  used different hyperparams. Mitigated by tracking baseline numbers in
  reports/eval and noting any discrepancy in the paper.

---

## D3. Sensor mask at the *DataLoader* level (or model level), not architecture surgery

**Choice:** The wrist-only ablation is implemented by zeroing input tof
channels for indices 8..31, *not* by re-instantiating the model with
`n_sensors=8`.

**Alternatives considered:**
- (a) `n_sensors=8` reduces model parameters too — different capacity.
- (b) Explicit "wrist-only" model class.

**Why this:**
- The ablation tests *information* (what does the model gain from
  whole-body coverage?), not *capacity*. With (a), a worse number could
  be explained by either smaller model or less information.
- Identical model architecture means the optimizer state, weight init,
  even the random seed all stay the same.

**Consequences:**
- (+) Apples-to-apples.
- (-) The wrist-only model has 32 input channels, 24 of which are always
  zero. Slight inference inefficiency that doesn't matter (sub-1ms).

---

## D4. Shared MLP encoder over per-sensor MLPs

**Choice:** A single `Linear(64) -> ReLU -> Linear(d_model) -> LayerNorm`
applied to every sensor's flattened 8x8 patch.

**Alternatives considered:**
- (a) Per-sensor MLP (32 separate encoders, no weight sharing).
- (b) Per-link MLP (4 encoders for link2/3/5/6 + EE).
- (c) Pooled (one MLP applied after concatenating all sensors → loses
  per-sensor identity).

**Why this:**
- Same hardware everywhere. The encoding function from "8x8 SPAD readings"
  to "useful proximity feature" should be the same regardless of which
  sensor we're reading. Inductive bias matches physics.
- Gradient signal: with N sensors, the shared MLP gets N× the gradient
  per training step compared to (a). This is what lets us train with
  ~1k trajectories instead of ~32k.
- (c) loses per-sensor identity, which the decoder needs in order to map
  "the wrist sensor sees something close" → "the gripper is about to
  contact". Per-sensor positional info comes naturally from the per-sensor
  token slot.

**Consequences:**
- (+) Trains on small data; right inductive bias.
- (-) Cannot specialize per-sensor (e.g. wrist sensors have different
  expected distributions than upper-arm sensors). Mitigated by the LayerNorm
  on the encoder output; per-sensor normalization on the input via
  `stats.json`; and per-sensor positional encoding in the decoder.

---

## D5. LayerNorm at every modality boundary

**Choice:** Every encoder output (proximity, vision, proprio) ends with
`nn.LayerNorm`. The fusion layer adds another LayerNorm after concat.

**Alternatives considered:**
- (a) No LayerNorm — let the magnitudes be what they are.
- (b) BatchNorm — depends on batch statistics.
- (c) RMSNorm.

**Why this:**
- The *single most common silent failure mode* in multi-modal models is
  one stream's tokens having a different magnitude than another's. The
  decoder learns to ignore the smaller stream because attention's
  softmax saturates.
- LayerNorm is per-token, deterministic, and has learnable affine
  parameters — so the model can still upweight a stream if it really
  wants to. It just doesn't *start* with one drowned out.
- BatchNorm is broken for transformer-style sequence models because
  attention can mix across tokens.
- RMSNorm is fine, but LayerNorm is the published ACT choice; we don't
  deviate without reason.

**Consequences:**
- (+) Day-1 sanity checks confirm encoder grad norms > 0.01 across all
  variants. Without the encoder LayerNorm we observed grad collapse (~1e-12)
  in early prototypes — this is the bug we are guarding against.
- (-) Negligible compute cost.

---

## D6. Frozen Molmo2 backbone, trainable projection only

**Choice:** All Molmo2 weights have `requires_grad = False`. Only a
`Linear(d_vlm → d_model)` projection on the output is trainable.

**Alternatives considered:**
- (a) Full fine-tune of Molmo2-4B.
- (b) LoRA adapters on Molmo2 attention layers.
- (c) Last-N-layer fine-tune.

**Why this:**
- Fine-tuning a 4 B-param backbone on ~1k trajectories causes catastrophic
  forgetting of the visual common-sense priors. PROJECT.md §6 reports the
  delta: PLA + frozen vs PLA + fine-tune is ~12 pp on near-contact.
- VRAM: full fine-tune adds ~16 GB of optimizer state. Frozen backbone
  fits in <9 GB total, leaving room for batch size 8.
- LoRA is a reasonable middle ground but doubles the engineering surface
  (which adapters? what rank? does it work with Molmo2's multi-modal
  fusion?). We deferred LoRA to a future ablation if the basic claim
  doesn't go through.

**Consequences:**
- (+) Stable, fast, fits in 9 GB. The proximity stream is the only thing
  *learning* visual-side; that's exactly what the paper argues for.
- (-) If Molmo2 is missing some feature (e.g. fine-grained object
  localization for `pnp_color`), we can't fix it via fine-tune.

---

## D7. Concat fusion as primary, cross-attention as ablation

**Choice:** Primary fusion is `LayerNorm(concat(tof, vlm, prop))`. The
cross-attention variant exists only as an ablation.

**Alternatives considered:**
- (a) Cross-attn primary: ToF queries VLM (or vice versa).
- (b) FiLM-style modulation of one stream by the other.
- (c) Token-level co-attention.

**Why this:**
- Concat is the simplest fusion that *doesn't commit*. The decoder's
  attention layers can freely mix any subset of tokens; nothing about
  the fusion biases it toward "ToF is conditioned on VLM" or vice versa.
- Cross-attn introduces an inductive bias ("given what I see, which
  proximity readings matter?"). That bias might help — but we want to
  *test* the bias, not assume it. So cross-attn is the ablation.
- FiLM and co-attention are interesting but introduce more design
  parameters (which stream modulates which? at what depth?) without
  obvious wins on the headline metric.

**Consequences:**
- (+) Fewest assumptions in the primary architecture.
- (-) Concat does not give us interpretable cross-modality attention
  weights. The cross-attn ablation does (used for figure 3 in the paper
  if it wins; otherwise we drop the figure).

---

## D8. ACT decoder over diffusion / autoregressive heads

**Choice:** Action Chunking Transformer (Zhao et al. 2023) — single forward
pass produces `chunk_size=100` joint deltas.

**Alternatives considered:**
- (a) Diffusion policy (Chi et al. 2023).
- (b) Autoregressive token-by-token decoder.

**Why this:**
- Inference speed. ACT is one forward pass; diffusion is 50-100. At 5 Hz
  control we can't afford 50× compute per timestep.
- Multimodality regularization. The CVAE bottleneck collapses multimodal
  expert demos to the mean trajectory under β=10. Demos from a heuristic
  TAMP planner have multiple acceptable execution traces; we want the
  policy to *commit* to one rather than sample from a multimodal posterior.
- Reviewer expectations. The VLM-only ACT baseline is ACT minus the
  proximity stream; the comparison is cleanest if both sides are ACT.

**Consequences:**
- (+) Fast inference, simple training, established baseline.
- (-) ACT can underperform diffusion on highly stochastic tasks. Mitigated
  by our task choices being deterministic-modulo-TAMP-tie-breaking.

---

## D9. β=10 constant, not annealed

**Choice:** `kl_weight=10.0`. Not annealed across training.

**Alternatives considered:**
- (a) β-VAE annealing: 0 → 10 over the first N steps.
- (b) β=1.

**Why this:**
- We *want* the latent to collapse near the prior so inference at z=0
  matches what the encoder produced during training. β=10 forces collapse.
- Annealing helps free-form generative models (where you want the latent
  to explain variation in the data); for a policy where we want a single
  modal trajectory, annealing is counterproductive.
- β=1 lets the latent escape and the model relies on z; at inference z=0
  diverges from training expectation and performance drops.

**Consequences:**
- (+) Stable training; matches Zhao et al. 2023.
- (-) Latent provides little stochasticity; the policy is essentially
  deterministic. Acceptable for our task suite.

---

## D10. Per-channel normalization, training-only stats

**Choice:** `pla.data.normalize` computes mean/std *per* sensor channel
(and per qpos joint, per action dim). Stats come from training files
*only*; val/test files use the same stats but never contribute.

**Alternatives considered:**
- (a) Global single-mean / single-std across all sensors.
- (b) Stats over the full dataset (no train/val separation).

**Why this:**
- Each ToF sensor has a different expected depth distribution — the
  wrist sensor sees the gripper and the object, the upper-arm sensor sees
  free space mostly. A global mean over-shrinks one and under-shrinks
  the other.
- Train-only stats prevent val/test info from leaking into the input
  distribution. A network that's seen normalized val data during stat
  computation has been shown the val set; reported numbers are optimistic.

**Consequences:**
- (+) Each sensor's MLP input lives in O(1) std. Faster convergence.
- (-) `stats.json` carries N×8×8 floats — small (KB), but check it into
  `runs/<name>/` along with the checkpoint.

---

## D11. One HDF5 file per episode

**Choice:** `data/raw/<task>/episode_NNNNNN.h5`, one file per rollout.

**Alternatives considered:**
- (a) One mega-HDF5 with all episodes as groups.
- (b) Parquet / Zarr / WebDataset.

**Why this:**
- Sharded collection: tmux session per worker writes to its own files;
  no locking, no merge step.
- Resumable: if the box crashes mid-collection we don't lose anything but
  the in-flight episode.
- Verification: bad shards fail the verifier individually; we can `rm` the
  bad ones and continue.
- HDF5 specifically: fast random slice access (the DataLoader's sliding
  window needs `tof[t : t+chunk_size]`), wide tooling support, matches
  the existing `skin_pick_fixed_v1` dataset.

**Consequences:**
- (+) Easy to reason about. Bad data is per-file, not per-row.
- (-) ~1000 files in a directory. `ls` is slow but `glob` is fine.

---

## D12. Sliding-window DataLoader, `chunk_size=100`, `K=2` RGB history

**Choice:** Each `__getitem__` yields a `(tof_t, rgb[t-1:t+1], qpos_t,
actions[t : t+chunk_size])` window. `chunk_size=100`, `K=2`.

**Alternatives considered:**
- (a) `chunk_size=50` (faster, smaller context, less stable).
- (b) `K=1` RGB (no history).

**Why this:**
- `chunk_size=100` matches Zhao et al. 2023 ACT defaults. At ~20 Hz
  control that's ~5 s of action, enough for a full PnP sub-trajectory.
- `K=2` gives one frame of motion context (object velocity, hand
  approach speed). `K=1` works for static scenes but the policy can't
  distinguish a moving object from a stationary one.
- Larger K makes each visual-token batch larger; we'd need to drop batch
  size or shrink d_model. Not worth it.

**Consequences:**
- (+) Match published ACT setup; one frame of motion context.
- (-) Each window opens the h5 once. Cached file handles per worker keep
  this fast.

---

## D13. Paired bootstrap p-values, shared episode seeds

**Choice:** Eval runs every method with the *same* `seed_base + i` for
`i in range(n_episodes)`. Paired bootstrap uses the per-episode diff vector.

**Alternatives considered:**
- (a) Independent samples + unpaired bootstrap / t-test.
- (b) Permutation test on the full data.

**Why this:**
- The variance reduction from pairing is large for binomial outcomes:
  paired CIs are ~30% tighter for the same N. We need fewer episodes for
  the same statistical power.
- Pairing requires shared scene/object placement, which requires a fixed
  seed schedule.
- Bootstrap is robust to the binomial nature of the outcomes (no
  normality assumption).

**Consequences:**
- (+) Fewer episodes, tighter CIs, valid p-values.
- (-) The eval harness cannot easily run two methods with different
  `seed_base`. We document this in `eval/README.md`.

---

## D14. Post-hoc sensor importance via masking, not retraining

**Choice:** For each sensor i, mask its 8x8 input channel and re-evaluate
on the same paired seeds.

**Alternatives considered:**
- (a) Train one PLA per masked subset (32 retrainings).
- (b) Saliency / attention-rollout.

**Why this:**
- Compute. 32 retrainings = 32 × full PLA training cost. Masking + 50
  eval episodes per sensor is ~30 minutes total.
- Apples-to-apples. The 32 masked evals all use the same trained model,
  so importance differences are clearly *information* differences, not
  retraining variance.
- Saliency on transformers is brittle and hard to validate; the masking
  experiment is causal (we *intervened* and measured the consequence).

**Consequences:**
- (+) Cheap, causal, paired.
- (-) Importance values understate the long-run effect (a retrained model
  could learn to lean on remaining sensors). For the figure we want, the
  masking ranking is what matters.

---

## D15. Single training script for every config

**Choice:** `pla/train/train.py` with `train_loop(cfg, ...)` is the only
training entry point. The historical `train_pla.py`, `train_baseline.py`,
`train_cvae.py` are kept for back-compat but no longer hold logic.

**Alternatives considered:**
- (a) `train_pla.py` plus parallel `train_baseline.py`, `train_*.py` for
  each ablation.

**Why this:**
- Keeps the loss formula, optimizer schedule, gradient-clipping, and
  logging setup *exactly* the same across all comparisons. Any change to
  `train_loop` propagates to every ablation.
- Per-script duplication tends to drift over weeks. We avoid the drift.

**Consequences:**
- (+) Reproducible across configs. One bug fix in `train_loop` fixes
  every variant.
- (-) The shell wrappers (`scripts/train_*.sh`) are essentially just
  `python -m pla.train.train --config <yaml>` — they exist for shortcut
  ergonomics, not logic.

---

## D16. CLI smoke-test path with `DummyVLBackbone`

**Choice:** Every sanity-check CLI (`forward_pass.py`, `grad_norm.py`)
defaults to a `DummyVLBackbone` that returns random tokens with the right
shape. Real Molmo2 is opt-in.

**Alternatives considered:**
- (a) Always use real Molmo2 (download 9 GB on every test run).
- (b) Mock the backbone with a constant zero output.

**Why this:**
- CI / new-laptop / offline-dev cases. We can verify the full graph
  without 9 GB of weights and without an Internet connection.
- Random tokens (with a learnable Linear projection inside the dummy)
  exercise the autograd graph end-to-end, including the projection-
  through-the-backbone path that real Molmo2 also takes.
- Constant tokens (option b) miss bugs where the model relies on the
  variation — unlikely but possible.

**Consequences:**
- (+) Sanity checks run in seconds on any machine.
- (-) Real Molmo2 forward pass is verified separately on Day 4 (after
  the first real training start).

---

## D17. Append-only `IMPLEMENTATION_LOG.md`

**Choice:** When something changes, add a new dated entry. Don't rewrite
old entries.

**Alternatives considered:**
- (a) Replace old entries with current state.
- (b) git history is the log.

**Why this:**
- We have 27 days. Things will be re-decided. The trail of decisions
  ("we tried X on Day 6 and it didn't work, so on Day 7 we did Y")
  matters when writing the discussion section of the paper.
- git history is the *what*; the log is the *why*. Reviewers don't read
  git history; they read the paper, and the paper draws on this log.

**Consequences:**
- (+) Permanent record of the project's intellectual evolution.
- (-) The file grows. Mitigated by reverse-chronological order; Day 1
  entries shrink in relevance once Day 14 lands.

---

## D18. Doc-first contracts (READMEs are the spec)

**Choice:** Every subfolder has its own README.md. The README is the
contract; code that drifts from a README is a bug, regardless of which
side is "right".

**Alternatives considered:**
- (a) Single big top-level README + docstrings.
- (b) Docstrings only (no README).

**Why this:**
- New collaborators (and future-self) need a roadmap, not just per-function
  documentation. The README answers "where do I start? what's the
  intent here?"; docstrings answer "what does *this* function do?".
- The per-folder README puts the science (motivation, decisions, sanity
  checks) next to the code that implements it.

**Consequences:**
- (+) One-stop docs per subfolder. `cat pla/models/README.md` tells you
  everything you need before reading code.
- (-) READMEs can drift. We mitigate with the explicit "if a README claim
  is wrong, update it" rule and routine README review during status
  updates.
