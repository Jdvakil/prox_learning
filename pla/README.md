# `pla/` — the importable package

This is the only place code goes. Everything outside (`assets/`, `configs/`,
`scripts/`, `paper/`, `runs/`, `data/`, `reports/`) is either inputs or
outputs to the modules here.

```
pla/
├── data/        Collection harness, HDF5 schema, normalization, DataLoader
├── sim/         URDF -> MJCF skin builder, ToF sensor rendering
├── models/      ProximityEncoder, FrozenMolmo2, ModalityFusion, ACT, PLA
├── train/       Training entry point, losses, grad-norm checks
├── eval/        Eval runner, bootstrap stats, sensor importance
├── ablations/   Ablation orchestration
├── checks/      Pre-training sanity checks (depth recon, replay, grad-norm)
└── viz/         Heatmaps, point clouds, composite figures
```

## How to read this package

For **every task in TIMELINE.md**, before writing or running anything,
re-read the README of the subfolder you are touching. The READMEs explain:

  * What the module is for (scientific motivation).
  * Why the code is shaped that way (key design decisions).
  * What sanity checks must pass before downstream work proceeds.
  * Concrete CLI commands you can copy-paste.

If a README claim is wrong, update it. The READMEs are the contract — code
that drifts from its README is a bug, regardless of which one is "right".

## Top-level invariants

These hold across the whole package; if you ever see a violation, treat it
as a P0 bug:

1. **Shapes.** Throughout the code:
    * `tof:     [B, N_sensors, 8, 8]`  float32, **mm** at I/O, normalized in
      the model.
    * `rgb:     [B, K, 3, H, W]`        in `[0, 1]` for the model;
      `uint8 [0,255]` on disk.
    * `qpos:    [B, 7]`
    * `actions: [B, chunk_size=100, 7]`
    * `context: [B, N_ctx, d_model=512]`

2. **Sensor order.** Cameras enumerated in MJCF order (NOT lexicographic).
   The DataLoader writes the same order; the model reads the same order.
   See `pla/sim/tof.py:ToFSensorArray.__init__`.

3. **Statistics.** `stats.json` is computed on training files only
   (`pla/data/normalize.py`). Validation and test splits MUST share the same
   stats but never contribute to them.

4. **Reproducibility.** `--seed` and `--split-seed` propagate through the
   whole stack. The same seed at eval time recovers the same scene
   distribution; this is what makes paired-bootstrap p-values valid.

5. **One-flag baseline.** PLA vs VLM-only ACT differs by **exactly one**
   YAML key: `vlm_only: true|false`. Both runs use the same dataset, the
   same hyperparameters, the same backbone, the same chunk size. If you
   find yourself adding any other difference, stop and rethink.
