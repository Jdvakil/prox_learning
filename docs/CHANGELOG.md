# Changelog

User-facing summary of changes. Newest at the top.

For the deep "what was implemented and why" log see
[IMPLEMENTATION_LOG.md](IMPLEMENTATION_LOG.md). For verification of
specific claims see [SANITY_CHECKS.md](SANITY_CHECKS.md).

---

## 2026-05-02 (Day 1)

### Added
- **Paper draft (`paper/`)**: `main.tex` (full CoRL-style LaTeX with TikZ
  architecture figure + bibliography), `references.bib` (15 entries),
  `Makefile` (pdflatex + bibtex pipeline), `main.md` (markdown mirror),
  `paper/README.md`. Methods + protocol locked; results table is
  pre-registered placeholder pending Day-14 numbers.
- **Full model layer**: ProximityEncoder (shared MLP), HandcraftedToFEncoder,
  Conv2DToFEncoder, ModalityFusion (concat + cross_attn), full ACTDecoder
  (Zhao 2023), FrozenMolmo2 + DummyVLBackbone, unified PLA class with
  `vlm_only` flag and `sensor_mask`.
- **Data layer**: `pla.data.dataset.PLADataset` (sliding-window),
  `pla.data.normalize` (per-channel training-only stats),
  `pla.data.verify` (Day-2 sanity), expanded `pla.data.collect`.
- **Training**: unified `pla.train.train` for every config; W&B optional
  with stdout JSONL fallback; per-step encoder grad-norm logging.
- **Evaluation**: `pla.eval.run_eval` (per-method/per-task runner +
  results-table aggregator), bootstrap stats helpers, sensor-importance
  sweep, failure-mode categorizer.
- **Sim**: `ToFSensorArray` class (cached renderer, MJCF-order sensor list).
- **Skin pipeline scripts**: `scripts/build_skin_mjcf.py` (Blender JSON →
  MJCF camera bodies), `scripts/verify_skin.py` (empty-scene self-hit).
- **Sanity checks**: `pla.checks.forward_pass` (5 variants),
  `pla.checks.grad_norm` (CLI driver).
- **Configs**: 4 ablation YAMLs (wrist_only, handcrafted, conv2d,
  cross_attn). Rewrote `pla.yaml` and `act_baseline.yaml` to flat schema —
  the only difference between them is `vlm_only`.
- **Per-folder READMEs**: `pla/`, `pla/data/`, `pla/sim/`, `pla/models/`,
  `pla/train/`, `pla/eval/`, `pla/checks/`, `pla/viz/`, `pla/ablations/`,
  `assets/`, `configs/`, `scripts/`, `reports/`.
- **Tracking docs**: `docs/STATUS.md`, `docs/IMPLEMENTATION_LOG.md`,
  `docs/SANITY_CHECKS.md`, `docs/ARCHITECTURE.md`, `docs/DESIGN_DECISIONS.md`,
  `docs/FILE_INVENTORY.md`, `docs/STATISTICAL_PROTOCOL.md`,
  `docs/CHANGELOG.md`.

### Verified
- 49/49 Python files parse cleanly.
- 29/29 modules smoke-import.
- Forward+backward+inference passes for: PLA, vlm_only, handcrafted,
  conv2d, cross_attn variants.
- Grad-norm > 1e-2 across all 4 ablation configs; baseline correctly skips.
- Synthetic data pipeline end-to-end: collect → verify → normalize →
  PLADataset → train (3 steps) — loss decreases, encoder grads non-zero.
- Bootstrap CI + paired bootstrap p-value match independent χ² check.
- Wrist-only mask zeros indices 8-31 and preserves 0-7 element-wise.

### Removed / replaced
- Stub `NotImplementedError` bodies in: `pla.data.collect`, `pla.eval.run_eval`,
  `pla.eval.sensor_importance`, `pla.eval.failure_analysis`.
- Legacy `train_pla.py` / `train_baseline.py` are now thin wrappers
  around `pla.train.train`; logic moved.

### Not yet done (deferred to Day 2+)
- Real Molmo2 forward pass (HF download).
- Built MJCF skin from Day-1-PM Blender redesign.
- 1000-trajectory near-contact dataset.
- Any real training run with real data.
- Paper figures.

---

## 2026-04-21 .. 2026-05-01 (Day 0 — pre-project setup)

### Added
- Initial repo structure with `pla/` package skeleton.
- Legacy `docs/PROJECT.md`, `docs/TIMELINE.md`, `docs/SKIN_PIPELINE.md`,
  `docs/CVAE.md`, `docs/DATASET.md`.
- Submodules: MolmoBot, molmospaces, ACT.
- URDF + MJCF for the legacy 10-traj sanity dataset
  (`skin_pick_fixed_v1`).
