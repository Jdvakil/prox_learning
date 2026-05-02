# `pla/models/` — model layer

## Purpose

The neural net. PLA = frozen VLM (Molmo2) + learned ProximityEncoder +
proprio projection -> ACT decoder. Everything in this folder is shape-only
plumbing on top of those four pieces; the heavy lifting is the
ProximityEncoder (small, learned) and Molmo2 (large, frozen).

## Files

| file                    | what's in it                                              |
|-------------------------|-----------------------------------------------------------|
| `proximity_encoder.py`  | `ProximityEncoder` (PLA) + `HandcraftedToFEncoder`,       |
|                         | `Conv2DToFEncoder` (ablations)                            |
| `vlm_backbone.py`       | `FrozenMolmo2` + `DummyVLBackbone` (for tests)            |
| `fusion.py`             | `ModalityFusion` — concat (PLA) and cross-attn (ablation) |
| `act.py`                | `ACTDecoder` — Zhao et al. 2023 ACT (CVAE + decoder)       |
| `pla.py`                | `PLA` — full model. `vlm_only=True` -> baseline mode.      |
| `baselines.py`          | `VLMOnlyACT` alias + `PropOnlyMLP` floor                  |
| `cvae.py`               | legacy: skin-CVAE pretrain proof (kept for reference)     |

## Dataflow

```
                                +--------------+
       rgb [B,K,3,H,W] -------> | FrozenMolmo2 | -> [B, N_vis, 512]
                                +--------------+        |
                                                        v
                              +-------------------+     concat
       tof [B,N,8,8] -------> | ProximityEncoder | -> [B,N,512]
                              +-------------------+
                                                        v
       qpos [B,7] --[Linear]-> [B,1,512]                concat
                                                        v
                                              context [B, ~225, 512]
                                                        |
                                                        v
                                                +---------------+
       actions [B,k,7] (training only) -------> |  ACTDecoder   | -> pred [B,k,7], mu, logvar
                                                +---------------+
```

## The `vlm_only` flag

`PLA(vlm_only=True)` skips the ProximityEncoder entirely and concats only
`[vlm_tokens, prop_token]`. This is the headline baseline. The flag is the
**only** difference between PLA and VLM-only ACT. Same backbone, same ACT
hyperparameters, same chunk size, same dataset.

## ACT hyperparameters (Zhao et al. 2023 exact)

| param              | value      |
|--------------------|------------|
| `d_model`          | 512        |
| `n_heads`          | 8          |
| `n_encoder_layers` | 4          |
| `n_decoder_layers` | 7          |
| `ffn_dim`          | 3200       |
| `chunk_size`       | 100        |
| `z_dim`            | 32         |
| `kl_weight` (beta) | 10 (constant) |
| `dropout`          | 0.1        |

We do not anneal beta. With beta=10 the latent collapses to a near-Dirac
during training, which is what we want at inference (we feed `z = 0`).

## Why these design choices

* **Shared MLP across sensors.** Same hardware everywhere -> same encoding
  function. We get N x more gradient signal per training step than per-sensor
  encoders would, and the inductive bias is correct.
* **LayerNorm at every modality boundary.** Without it, the
  visual-language tokens (Molmo2-scale floats) drown out the proximity
  tokens (small mm-scale numbers) and the decoder learns to ignore them.
  This is the root cause of the most common silent failure mode (encoder
  grad norm = 0).
* **Frozen Molmo2.** Fine-tuning the 3.7B-param backbone on ~1k trajectories
  causes catastrophic forgetting of the visual common-sense priors. The
  small projection layer is the only trainable part of the VLM path, which
  preserves backbone knowledge and saves >9 GB of optimizer state.
* **Concat fusion.** Simplest possible fusion that lets the decoder freely
  mix any subset of tokens. Cross-attention biases the decoder toward "ToF
  queries VLM" — see the cross-attn ablation for the test.
* **One-shot chunked actions.** Predicting 100 future joint deltas in a
  single forward pass is fast at inference and avoids the temporal
  ensembling boundary issue at the chunk boundary. We re-plan from scratch
  every 100 steps; for FR3 this is ~5 Hz which is plenty for PnP.

## Sanity-check checklist (run after any model edit)

- [ ] `python -m pla.checks.forward_pass` succeeds with `DummyVLBackbone`.
      Asserts pred shape `[2, 100, 7]`, mu/logvar shapes `[2, 32]`.
- [ ] After one backward pass on dummy data, `model.proximity_encoder`
      grad norm > 1e-6 (else LayerNorm or fusion is broken).
- [ ] `model.vl` parameters all have `requires_grad=False`.
- [ ] `model.act_decoder.compute_loss(...).total` is finite.
- [ ] `vlm_only=True`: `model.proximity_encoder is None`. Forward pass
      ignores `tof`.
- [ ] `sensor_mask=[…]`: indices in the mask appear as zeros in the
      proximity input the encoder sees.
