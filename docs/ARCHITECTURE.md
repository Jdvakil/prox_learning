# Architecture reference

Tensor shapes, dataflow, and every module's I/O contract. This document is
the canonical place to look when debugging "why does my batch have shape
[B, 32, 8, 8] when the encoder expects [B, 8, 8, 32]?" type questions.

Companion docs:
- [PROJECT.md](PROJECT.md) — scientific motivation, references, claims
- [STATUS.md](STATUS.md) — what's built vs. what's not
- [DESIGN_DECISIONS.md](DESIGN_DECISIONS.md) — *why* the architecture is this shape

---

## 0. Layered view

```
┌──────────────────────────────────────────────────────────────────────────┐
│ HARDWARE LAYER                                                           │
│   29-32 VL53L5CX 8x8 SPAD sensors on FR3 (GenTact skin, link2/3/5/6 + EE)│
│   45° FOV, 20-4000 mm range, ~5 mm noise floor                           │
└──────────────────────────────────────────────────────────────────────────┘
                              │ MJCF cameras
                              ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ SIMULATION LAYER (pla/sim/)                                              │
│   ToFSensorArray: walks MJCF cameras, renders 8x8 depth per step         │
│   tof_array.render(data) -> [N_sensors, 8, 8] float32 mm                 │
│   extend_obs_with_tof(obs, env) mutates obs['tof']                       │
└──────────────────────────────────────────────────────────────────────────┘
                              │ HDF5
                              ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ DATA LAYER (pla/data/)                                                   │
│   collect.py    -> data/raw/<task>/episode_*.h5                          │
│   verify.py     -> >=30% prox-informative, 0 NaN, schema OK              │
│   normalize.py  -> stats.json (training-only per-channel mean/std)       │
│   dataset.py    -> PLADataset: sliding-window, normalized                │
└──────────────────────────────────────────────────────────────────────────┘
                              │ DataLoader batch
                              ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ MODEL LAYER (pla/models/)                                                │
│                                                                          │
│   ┌─────────────────────┐                                                │
│   │  ProximityEncoder   │  tof[B,N,8,8]  -> [B,N,512]                    │
│   │  (shared MLP+LN)    │  (or [B,1,512] for handcrafted ablation)       │
│   └─────────────────────┘                                                │
│                                                                          │
│   ┌─────────────────────┐                                                │
│   │  FrozenMolmo2       │  rgb[B,K=2,3,224,224] + lang -> [B,N_vis,512]  │
│   │  (frozen, +Linear)  │  (DummyVLBackbone for tests)                   │
│   └─────────────────────┘                                                │
│                                                                          │
│   ┌─────────────────────┐                                                │
│   │  ModalityFusion     │  concat | cross_attn -> [B, N_ctx, 512]        │
│   │  (+ proprio token)  │                                                │
│   └─────────────────────┘                                                │
│                                                                          │
│   ┌─────────────────────┐                                                │
│   │  ACTDecoder         │  context, (actions, qpos) ->                   │
│   │  (CVAE+transformer) │    pred[B, 100, 7], mu/logvar[B, 32]           │
│   └─────────────────────┘                                                │
└──────────────────────────────────────────────────────────────────────────┘
                              │ pred actions
                              ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ TRAINING LAYER (pla/train/)                                              │
│   loss = L1(pred, gt) + 10 * KL(N(mu,exp(logvar)) || N(0,I))             │
│   Adam lr=1e-5, batch=8, grad_clip=1.0                                   │
│   per-step proximity-encoder grad-norm logging                           │
└──────────────────────────────────────────────────────────────────────────┘
                              │ best.pt
                              ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ EVAL LAYER (pla/eval/)                                                   │
│   N=100 episodes per (method x task) cell                                │
│   Bootstrap 95% CIs + paired bootstrap p-values vs VLM-only ACT          │
│   Sensor-importance via post-hoc masking                                  │
│   Failure-mode categorization (collision/grasp/place/language)           │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Canonical shapes

### 1.1 At I/O (HDF5 on disk)

| key                          | dtype   | shape                  | unit / scale          |
|------------------------------|---------|------------------------|------------------------|
| `obs/tof`                    | float32 | `[T, N, 8, 8]`          | millimetres in [20, 4000] |
| `obs/rgb`                    | uint8   | `[T, 3, 224, 224]`      | bytes [0, 255]            |
| `obs/qpos`                   | float32 | `[T, 7]`                | radians (FR3 joints)      |
| `actions`                    | float32 | `[T, 7]`                | joint deltas              |
| `policy_phase` (optional)    | int32   | `[T]`                   | TAMP stage id              |
| `attrs.success`              | bool    | scalar                 | episode success            |
| `attrs.n_sensors`            | int     | scalar                 | matches per-step N         |
| `attrs.language` (optional)  | str     | scalar                 | instruction string         |

### 1.2 At DataLoader output

`PLADataset.__getitem__` yields a dict with the values below; `collate_pla`
stacks numerical tensors and keeps `language` as a list.

| key       | dtype   | shape                    | unit / scale            |
|-----------|---------|--------------------------|--------------------------|
| `tof`     | float32 | `[N, 8, 8]`               | normalized (z-scored)    |
| `rgb`     | float32 | `[K=2, 3, 224, 224]`      | [0, 1]                   |
| `qpos`    | float32 | `[7]`                     | normalized               |
| `language`| str     | scalar                   | (per-episode constant)   |
| `actions` | float32 | `[chunk_size=100, 7]`      | normalized               |

### 1.3 Inside the model

| stage                          | shape                           |
|--------------------------------|---------------------------------|
| input `tof`                    | `[B, N, 8, 8]`                   |
| `ProximityEncoder.forward`     | `[B, N, 512]`                    |
| `HandcraftedToFEncoder.forward`| `[B, 1, 512]`                    |
| `Conv2DToFEncoder.forward`     | `[B, N, 512]`                    |
| `FrozenMolmo2.forward(rgb)`    | `[B, N_vis ≈ 192, 512]`           |
| `ModalityFusion(concat)`       | `[B, N + N_vis + 1, 512]`         |
| `ModalityFusion(cross_attn)`   | `[B, N + N_vis + 1, 512]` + attn  |
| ACTDecoder context input       | `[B, N_ctx, 512]`                  |
| ACTDecoder.encode (training)   | `(mu, logvar) = ([B,32], [B,32])` |
| ACTDecoder.decode              | `pred = [B, 100, 7]`               |
| `loss.backward()`              | grads on every trainable param    |

---

## 2. Module contracts (I/O signatures)

Every module that takes or returns tensors has a hard contract. Violations
should fail loud (assertion or shape mismatch from the next layer).

### 2.1 `pla.sim.tof.ToFSensorArray`

```python
arr = ToFSensorArray(model)                                    # mjModel
depth = arr.render(data, add_noise=True)                        # mjData
                                                               # depth: [N, 8, 8] mm
```

- `arr.n_sensors == len(arr.sensor_cam_ids) == len(arr.sensor_cam_names)`.
- `arr.sensor_cam_names` is in MJCF order, *not* lexicographic.
- Re-clipped to `[ZNEAR_MM, ZFAR_MM]` after Gaussian noise + dropout.

### 2.2 `pla.data.PLADataset`

```python
ds = PLADataset(data_dir, stats_path, chunk_size=100, split='train|val', val_frac=0.1)
sample = ds[i]   # dict with the keys/shapes documented in 1.2
```

- Splits: `seed`+`val_frac` permutation. Same seed in `normalize.py` -> no leak.
- Sliding window: every `t` such that `(t - K + 1) >= 0` and `(t + chunk_size) <= T`.
- All numeric outputs already normalized.

### 2.3 `pla.models.ProximityEncoder`

```python
enc = ProximityEncoder(n_sensors=32, d_model=512)
tokens = enc(tof)                              # tof: [B, N, 8, 8]
                                              # tokens: [B, N, 512]
```

- Asserts `tof.shape[-2:] == (8, 8)`.
- Raises `ValueError` if `tof.shape[1] != n_sensors`.
- LayerNorm on output (this is essential — see ARCHITECTURE §3.1).

### 2.4 `pla.models.PLA`

```python
m = PLA(n_sensors=32, d_model=512, chunk_size=100, vl_backbone=...)
# Training
pred, mu, logvar = m(rgb, language, tof, qpos, actions)
# Inference (z=0 path)
pred = m(rgb, language, tof, qpos, None)
# Inference helper that unnormalizes
acts = m.get_action(obs, stats)              # acts: numpy [chunk_size, 7]
```

- `vl_backbone` callable: `(rgb, language) -> [B, N_vis, d_model]`.
- `vlm_only=True`: `proximity_encoder is None`; `tof` is ignored.
- `sensor_mask=[indices]`: zero those indices in `tof` before encoding.

### 2.5 `pla.models.ACTDecoder`

```python
dec = ACTDecoder(action_dim=7, chunk_size=100, d_model=512)
# Training
pred, mu, logvar = dec(context, actions=actions, qpos=qpos)
# Inference
pred = dec(context)
total, l1, kl = dec.compute_loss(pred, target_actions, mu, logvar)
```

- `context: [B, N_ctx, 512]` — must come post-LayerNorm from fusion.
- `pred: [B, chunk_size=100, 7]`.
- `mu, logvar: [B, z_dim=32]`.
- `total = l1 + kl_weight * kl`, `kl_weight=10` by default.

### 2.6 `pla.eval.bootstrap`

```python
mean, lo, hi = bootstrap_ci(successes, n_resamples=10000, alpha=0.05, seed=0)
p = paired_bootstrap_p(successes_a, successes_b, n_resamples=10000, seed=0)
```

- `successes_*` are 0/1 numpy arrays of equal length (paired across methods).
- `p` is two-sided.

---

## 3. Critical invariants

These hold throughout the whole codebase. Violations almost always indicate
a bug rather than a deliberate design choice.

### 3.1 **LayerNorm at every modality boundary.**

Every module that produces tokens (proximity, vision, proprio) ends with a
`nn.LayerNorm`. Without this, the magnitudes of the streams differ by 10×+
and the decoder learns to ignore the smaller stream. The LayerNorm scales
are *learned*, so the model can still upweight a stream if it wants to —
but it doesn't *start* with one stream drowned out.

### 3.2 **Sensor enumeration order is MJCF order.**

`ToFSensorArray.__init__` walks `range(model.ncam)` in MJCF source-file
order. Sorting by name corrupts the encoder's per-sensor positional
mapping. The DataLoader does *not* re-sort; the model does *not* re-sort.

### 3.3 **Stats are computed on training files only.**

`pla.data.normalize.compute_stats` and `pla.data.dataset.PLADataset` use
the *same* `(seed, val_frac)` permutation. The val/test split is therefore
deterministic, and the stats never see val data.

### 3.4 **The VLM backbone is frozen.**

`FrozenMolmo2._ensure_loaded` sets `requires_grad = False` on every
backbone parameter. The *only* trainable VLM-side parameter is the
`Linear(d_vlm -> d_model)` projection.

### 3.5 **Action chunks are unnormalized at inference.**

`PLA.get_action` ends with `pred * act_std + act_mean`. If you ever build
a custom inference loop, the unnormalize step is non-negotiable.

### 3.6 **The VLM-only baseline differs from PLA by exactly one flag.**

Same dataset, same hyperparameters, same backbone, same chunk_size, same
seed-base for eval. Only `vlm_only: false` vs `true` in the YAML.

### 3.7 **Paired bootstrap requires shared episode seeds.**

`run_eval.py` uses `seed_base + i` for `i in range(n_episodes)`. If you
run two methods with different `seed_base`, the paired p-value is
*invalid* (the comparison is over different scenes). The aggregator does
*not* check for this; the responsibility is yours when launching evals.

---

## 4. Token budget at the ACT decoder

Approximate per-batch token count entering the decoder context:

| stream      | tokens   | dim | total floats per sample (B=1) |
|-------------|----------|-----|--------------------------------|
| proximity   | 32       | 512 | 16,384                         |
| visual-lang | ~192     | 512 | 98,304                         |
| proprio     | 1        | 512 | 512                            |
| latent z    | 1        | 512 | 512                            |
| **total**   | ~226     | 512 | ~115,712                        |

At batch size 8 the decoder sees ~1.8 M floats per forward. The 7-layer
decoder × 8 heads × 226 tokens fits comfortably in <2 GB activations on
top of the ~7 GB Molmo2 weights.

---

## 5. Loss anatomy

```
L = L1(pred, gt) + beta * KL
  = mean_over_batch_chunk(|pred - gt|)
  + 10 * mean_over_batch( -0.5 * sum(1 + logvar - mu^2 - exp(logvar)) )
```

At convergence we expect:
- `L1` ~ 0.05 - 0.15 (normalized action units; a 10 mm joint delta becomes
  ~1 std of action-std).
- `KL` ~ 0.001 - 0.01 (latent collapsed near prior). With `beta=10` the KL
  term contributes 0.01 - 0.1 to total loss — comparable to L1.
- `total` ~ 0.06 - 0.25.

If `KL` stays > 1 after a few epochs the latent is not collapsing — usually
because beta is too low or the encoder is too expressive for the dataset.

If `L1` is flat across epochs the encoder is not learning useful
representations — check `train/proximity_grad_norm`.

---

## 6. Dataflow at inference

```
real env step
   │
   ├─ env.step(action) -> obs[rgb, qpos, ...]  (no tof yet)
   ├─ tof_array.render(env.data) -> tof[N, 8, 8]
   │
   ├─ obs_norm = normalize(obs, stats)
   │     - tof_norm  = (tof  - tof_mean)  / tof_std
   │     - qpos_norm = (qpos - qpos_mean) / qpos_std
   │     - rgb_norm  = rgb / 255.0
   │
   ├─ pred_norm = model(rgb=rgb_norm, language=instr, tof=tof_norm,
   │                     qpos=qpos_norm, actions=None)
   │     - context = fusion( prox_enc(tof_norm), vlm(rgb_norm, instr), qpos_norm )
   │     - pred_norm = act_decoder.decode(z=0, context)        # [1, 100, 7]
   │
   ├─ pred = pred_norm * act_std + act_mean                    # unnormalize!
   │
   └─ for action in pred[0]: env.step(action)
```

The **unnormalize step is not optional**. Forgetting it produces a robot
that moves at `1 / action_std` of the intended scale.

---

## 7. Reading order for new collaborators

1. [PROJECT.md](PROJECT.md) — what we're trying to prove and why.
2. [TIMELINE.md](TIMELINE.md) — the 26-day schedule.
3. [ARCHITECTURE.md](ARCHITECTURE.md) (this file) — the shapes.
4. `pla/README.md` — the package layout.
5. The README in the subfolder you're touching.
6. [DESIGN_DECISIONS.md](DESIGN_DECISIONS.md) — why we chose these shapes.
7. [SANITY_CHECKS.md](SANITY_CHECKS.md) — the contract you must not break.

If you only read one file other than PROJECT.md: read this one.
