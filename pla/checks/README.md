# `pla/checks/` — pre-training sanity checks

## Purpose

Cheap, deterministic scripts that fail loud when an upstream piece is
broken. Each one corresponds to a specific failure mode that can silently
ruin a training run; run the relevant check before kicking off any
expensive job.

## Files

| file                       | what it checks                                   | when to run |
|----------------------------|--------------------------------------------------|-------------|
| `forward_pass.py`          | full PLA forward pass with `DummyVLBackbone`     | Day 4 (and after any model edit) |
| `grad_norm.py`             | `assert_learning(module)` — non-zero grads       | Day 6 (every PLA training start) |
| `depth_reconstruction.py`  | reconstructs a known scene from sensor readings  | Day 2 (after MJCF build) |
| `replay_mjcf.py`           | replays a recorded trajectory to validate cams   | Day 2 (after MJCF build) |

## Run

```bash
# Day 2 — after MJCF skin build
python -m pla.checks.depth_reconstruction
python -m pla.checks.replay_mjcf

# Day 4 — after first model edit / before training
python -m pla.checks.forward_pass

# Day 6 — every PLA training launch
python -m pla.checks.grad_norm \
    --config configs/train/pla.yaml \
    --steps 50
```

## What each check guards against

### `forward_pass.py`
Builds a 2-batch `PLA` with the dummy backbone and runs
`forward(...)`. Asserts:
  * `pred.shape == (2, 100, 7)`
  * `mu.shape == (2, 32)`, `logvar.shape == (2, 32)`
  * One backward pass produces non-NaN grads on every trainable parameter.
This catches: shape regressions, NaN init, missing modules, broken
`vlm_only=True` path.

### `grad_norm.py`
Trains for N steps (default 50) on dummy data and asserts
`proximity_encoder` grad-norm > `eps`. Catches the LayerNorm-kills-gradient
bug, missing autograd edges, and "encoder accidentally frozen" mistakes.

### `depth_reconstruction.py`
Renders a fixed scene from every sensor camera, reconstructs the obstacle
positions in world coordinates from the depth + camera pose, and checks
they match the ground-truth obstacle. Catches MJCF orientation bugs (the
infamous "sensor +Z points the wrong way" issue from PROJECT.md §2 / the
patch_mjcf workaround).

### `replay_mjcf.py`
Loads a real recorded trajectory and replays it against the MJCF, comparing
the rendered ToF stream to the recorded ToF stream. Differences > 50 mm RMS
indicate a sensor-position drift between the URDF used for collection and
the MJCF used for training.

## Sanity-check checklist (every training day)

- [ ] `forward_pass.py` exits 0
- [ ] `grad_norm.py` reports non-zero norms on every parameter
- [ ] If using a freshly built MJCF: `depth_reconstruction.py` and
      `replay_mjcf.py` both pass
