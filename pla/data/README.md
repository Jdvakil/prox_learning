# `pla/data/` — collection, schema, normalization, DataLoader

## Purpose

Turn raw MolmoSpaces rollouts into a dataset that PyTorch can train on. The
contract is fixed: we always write **HDF5 shards, one episode per file**,
with the schema below. Every downstream module (training, eval, viz) reads
through `PLADataset` so the schema is checked exactly once, at load time.

## HDF5 schema (per episode)

```
episode_0/
  observations/
    tof:    [T, N_sensors, 8, 8]   float32   millimetres, clipped [20, 4000]
    rgb:    [T, 3, 224, 224]       uint8     standard image bytes
    qpos:   [T, 7]                 float32   FR3 joint positions (rad)
  actions:  [T, 7]                 float32   joint deltas
  policy_phase: [T]                int32     (optional) which TAMP stage
  attrs:
    success:    bool
    n_sensors:  int
    task:       str (optional)
    language:   str (optional, used by DataLoader)
    seed:       int
```

Validators live in `schema.py` (`validate`, `proximity_informative_fraction`).

## Files

| file                | what it does                                                   |
|---------------------|----------------------------------------------------------------|
| `collect.py`        | drives MolmoSpaces TAMP rollouts; writes HDF5 shards           |
| `schema.py`         | structural validator + proximity-informative coverage          |
| `verify.py`         | Day-2 sanity check; targets 30%+ near-contact, 0 NaN, schema OK |
| `normalize.py`      | per-channel mean/std; writes `stats.json`                      |
| `dataset.py`        | `PLADataset` — sliding-window DataLoader (chunk_size=100)       |
| `cvae_dataset.py`   | legacy: skin-CVAE pretrain dataset (kept for the encoder proof) |
| `stats.py`          | dataset-level summary statistics + plot helpers                 |

## Order of operations (TIMELINE.md Days 3-5)

```bash
# 1. Collect (Day 3): launches MolmoSpaces TAMP, writes data/raw/<task>/*.h5
bash scripts/collect_data.sh near_contact 1000

# 2. Verify (Day 3 evening, again at end of run): schema + proximity coverage.
python -m pla.data.verify --data-dir data/raw/near_contact --strict

# 3. Normalize (Day 4): compute training-only stats. THIS MUST COME FIRST.
python -m pla.data.normalize --data-dir data/raw/near_contact --out stats.json

# 4. Smoke-test the loader (Day 4):
python -c "
from pla.data import PLADataset
ds = PLADataset('data/raw/near_contact', 'stats.json', chunk_size=100, split='train')
print(len(ds), ds[0]['tof'].shape, ds[0]['actions'].shape)
"
```

## Why these design choices

* **One episode per file.** Lets us shard and resume collection without
  rewriting; lets the verifier surface bad shards individually; lets the
  DataLoader build its sliding-window index in parallel.
* **mm on disk, normalized in the model.** Storing raw mm makes verification
  trivial (`tof < 200` is a literal "below 20 cm"), preserves the physical
  meaning across machines, and only costs one extra subtract+divide on
  load. Networks learn faster on standardized inputs (PROJECT.md §3.2).
* **Training-only stats.** Computing stats on the full set leaks val/test
  info into the input distribution; reported numbers would be optimistic.
  The val split is held out using the same `(seed, val_frac)` tuple in both
  `normalize.py` and `dataset.py` so the splits agree by construction.
* **Sliding window with `chunk_size=100`.** Matches Zhao et al. 2023; the
  CVAE encoder needs the whole future chunk at training, the decoder needs
  the same at inference. The window starts at `t=K-1` where `K=2` so we
  always have two RGB frames of history (PROJECT.md §3.4 uses K=2).
* **`proximity_informative_fraction >= 30%`.** Without near-contact frames
  the proximity stream has no signal during PLA training. The verify step
  fails loud rather than letting us train a model on uninformative data.
  This is the most common failure mode for the headline experiment.

## Sanity-check checklist (run before training each day)

- [ ] `pla.data.verify` exits 0 (schema OK, NaN-free, ≥30% prox-informative)
- [ ] `stats.json` exists and `n_sensors` matches the model config
- [ ] `len(PLADataset('train'))` > 10 * batch_size (else val/train split is broken)
- [ ] `(ds[0]['tof'].mean(), ds[0]['tof'].std())` are O(1) — confirms normalization
- [ ] First batch loads in < 1 s on warm cache (else IO is the bottleneck)
