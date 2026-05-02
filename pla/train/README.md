# `pla/train/` — training entry point

## Purpose

A single training loop driven by a YAML config. The same `train_loop`
function trains PLA, VLM-only ACT, and every encoder/fusion ablation.

## Files

| file        | role                                                              |
|-------------|-------------------------------------------------------------------|
| `train.py`  | `main()` + `train_loop` — Adam(lr=1e-5), L1+10*KL, grad clip      |
| `losses.py` | `act_loss(pred, gt, mu, logvar, beta=10)` standalone helper       |
| `train_pla.py`, `train_baseline.py`, `train_cvae.py` | thin wrappers (kept for backwards compat) |

## Run

```bash
# PLA
python -m pla.train.train --config configs/train/pla.yaml

# VLM-only baseline (one-flag difference; this is the comparison run)
python -m pla.train.train --config configs/train/act_baseline.yaml

# Smoke / shape test (no GPU, no real VLM weights):
python -m pla.train.train --config configs/train/pla.yaml \
    --dummy-vlm --no-wandb --max-steps 5
```

## What the loop does, every step

1. Move batch to device.
2. Forward: `model(rgb, language, tof, qpos, actions)` -> `pred, mu, logvar`.
3. Loss: `total = L1(pred, gt) + 10 * KL(mu, logvar)`.
4. Backward.
5. Compute the proximity-encoder L2 grad norm (across ALL params in
   `model.proximity_encoder`). If `< min_grad_norm` and we are not in
   VLM-only mode, **WARN**. This is the single most diagnostic number in
   the whole training run; see PROJECT.md §6.
6. `clip_grad_norm_(trainable, 1.0)`.
7. `optimizer.step()`.
8. Every `log_every` steps, log to W&B (or stdout JSONL).

## Logging

Every step logs:
  * `train/loss_total`, `train/loss_l1`, `train/loss_kl`
  * `train/proximity_grad_norm` (the diagnostic number)

Every epoch logs:
  * `val/loss`
  * `epoch_seconds`

Best `val/loss` is saved to `<output_dir>/best.pt`; a rolling `last.pt` is
overwritten every epoch.

## Why these design choices

* **One training script, all ablations.** PLA, baseline, and ablations
  differ by **YAML flags only**. There are no parallel training scripts.
  This is the single most important simplification — it makes the VLM-only
  comparison automatically controlled.
* **Adam + lr=1e-5.** ACT uses these exact values. Higher lr destroys the
  KL term faster than the L1 term can recover.
* **`grad_clip=1.0`.** ACT is a transformer with a stochastic latent. KL
  loss occasionally spikes when `logvar` blows up; clipping to 1.0 is the
  cheap fix.
* **`min_grad_norm=1e-8`.** This is the threshold below which we believe
  the proximity encoder is silently disconnected. A loose threshold here
  catches the LayerNorm placement bug AND a missing edge in the autograd
  graph (e.g. someone wrapped the encoder output in a `with torch.no_grad()`).
* **W&B optional.** If wandb isn't installed (or `--no-wandb`), we fall
  back to JSONL on disk; the same metrics get logged so the same checks
  work offline.

## Sanity-check checklist (Day 4-7)

- [ ] First 100 steps: `train/loss_total` is finite; `train/loss_l1`
      decreases monotonically (no spikes).
- [ ] `train/proximity_grad_norm > 1e-6` after first 50 steps.
- [ ] At step 0, `enc_grad_norm` for `vlm_only=true` run is `nan` — this
      confirms the encoder really isn't built in baseline mode.
- [ ] Memory peak < 18 GB on A100 (PLA) or < 12 GB (baseline).
- [ ] After 1 epoch, `val/loss` < `train/loss_total` (else val split overlaps
      with train — this should be impossible given the seeded split, but
      check anyway).
- [ ] Best checkpoint loadable via `torch.load(...)['config']` reproduces
      the run config.

## What goes wrong

Failure                          | Most likely cause
---------------------------------|----------------------------------------
`enc_grad_norm == 0`             | LayerNorm killed gradients OR the model is in vlm_only mode
NaN loss                         | `logvar.exp()` overflow; reduce lr, increase clip
L1 oscillates                    | Action stats wrong (re-run normalize); or beta too high
val/loss flat                    | Dataset too small or sliding window not configured
peak GPU usage > VRAM            | Reduce batch_size, then n_decoder_layers
