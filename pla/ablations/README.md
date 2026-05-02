# `pla/ablations/` — ablation orchestration

## Purpose

Run every ablation **with the same training script**. There is no
`train_ablation.py` — every ablation is a YAML in `configs/train/ablation_*.yaml`
that flips a single switch from `pla.yaml`.

## The ablations

| name                   | what changes from PLA                              | what it tests |
|------------------------|----------------------------------------------------|---------------|
| `ablation_wrist_only`  | `sensor_mask` zeros all non-wrist sensors          | Does whole-body coverage matter, or is the wrist enough? |
| `ablation_handcrafted` | `encoder_type: handcrafted`                        | Does the *learned* encoder beat min/mean/contact stats? |
| `ablation_conv2d`      | `encoder_type: conv2d`                             | Does the 8x8 grid's spatial structure carry signal? |
| `ablation_cross_attn`  | `fusion_type: cross_attn`                          | Does the "ToF queries VLM" inductive bias help? |

Sensor importance is **post-hoc** (no retraining), via
`pla.eval.sensor_importance`.

## Run

```bash
# Sequential (single GPU):
python -m pla.ablations.run_ablations

# Print the tmux launch commands instead (multi-GPU):
python -m pla.ablations.run_ablations --parallel

# Or launch one at a time:
python -m pla.train.train --config configs/train/ablation_wrist_only.yaml
```

## Why these design choices

* **Single training script.** If we had `train_ablation_handcrafted.py`,
  every ablation would diverge from PLA over time. Driving everything from
  YAML means an architectural change to PLA propagates to every ablation
  automatically (good); equivalently, a controlled comparison is
  *guaranteed by construction* (better).
* **Wrist-only via DataLoader mask, not architecture surgery.** We don't
  reduce `n_sensors` to 8; we keep all 32 input channels and zero out 24
  of them. That keeps the model architecture identical to PLA, so the
  ablation tests **information**, not **capacity**.
* **Sensor importance via masking, not retraining.** Retraining 32 ablation
  models costs the same as 32 full PLA runs. Masking is a 50-episode eval
  per sensor — about 30 min total. The numbers don't tell exactly the
  same story (importance under retraining could be larger), but the rank
  correlation is what matters for the heatmap figure.

## Sanity-check checklist (Day 8-12)

- [ ] All 4 ablation configs trained to convergence (val/loss plateau).
- [ ] On the near-contact task (50 episode quick eval), the ranking is:
      `pla >= ablation_cross_attn >= ablation_conv2d >= ablation_handcrafted >= ablation_wrist_only >= vlm_only_baseline >= prop_only`.
      If any inversion is large (>5 pp), investigate before running 100-ep eval.
- [ ] Sensor importance heatmap is non-uniform (else PLA isn't using
      proximity).
