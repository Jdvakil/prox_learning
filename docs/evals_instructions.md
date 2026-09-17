# Eval instructions

Branch `feat/pact-place-v6`. New files for FrozenACT eval. Open-loop chunk 50, `--skin rays`, terminal `judge_success`. n=48, `--spread_cells`.

## New files

| File | What it does |
|---|---|
| `eval_act_v1011d.py` | v1011d-world FrozenACT eval. On this branch, `--history consecutive` is the 8-step causal window (last 7 idle chunk steps plus the query), matching readout training. On `main`, `consecutive` still samples skin every control step. |
| `eval_act_v6.py` | Same FrozenACT protocol on the v6 two-bottle world (slots 01/06 live, plates parked). `--prox_ablate zero` keeps 40×128 skin tokens but feeds zeros. |
| `eval_place_v1010_scene.py` | Helper used only by `eval_act_v6.py`: 24-cell table, pose XMLs, scene hashes, spread counts. Not an eval. |
| `scripts/eval_v6_protocol_n48.sh` | Shard launcher: ACT, PACT-raw, readout n=48. |
| `scripts/eval_v6_readout_prefetch_n8.sh` | Shard launcher: readout prefetch-8 only. |
| `scripts/eval_v6_readout_zeroskin.sh` | Shard launcher: readout `--prox_ablate zero`. |

The `.sh` files only launch `eval_act_v6.py` (8 workers). They are not a different protocol.

## v6 runs

Ckpts under `submodules/act/ckpts/pact_pick_n_place_v6/`.

| Arm | Ckpt | Flags |
|---|---|---|
| ACT | `20260915_113546_act_pick_n_place_v6_s0` | `--history query` |
| PACT-raw | `20260915_114210_pact_pick_n_place_v6_raw_s0` | `--history query` |
| PACT-readout prefetch 8 | `20260915_115435_pact_pick_n_place_v6_readout_s0` | `--history consecutive` |
| PACT-readout zeroskin | same readout ckpt | `--history consecutive --prox_ablate zero` |

```bash
python eval_act_v6.py \
  --ckpt_dir submodules/act/ckpts/pact_pick_n_place_v6/<ckpt> \
  --num_rollouts 48 --spread_cells \
  --history query \
  --skin rays --skin_substeps snapshot \
  --output_dir eval_output/v6/act_n48
```

Readout uses `--history consecutive`. Zeroskin adds `--prox_ablate zero`.

v1011d world (not v6):

```bash
python eval_act_v1011d.py \
  --ckpt_dir <v1011d_ckpt> \
  --history consecutive --skin rays --skin_substeps snapshot \
  --num_rollouts 48 \
  --output_dir eval_output/v1011d/readout_prefetch8
```

Train stays Hub HDF5. Do not use `eval_act_pact_pick_n_place.py` for these runs. Do not mix every-control-step skin with prefetch-8.

## Results (v6, n=48)

FrozenACT, prefetch-8 readout, `--skin rays`. Place success:

| Arm | Place | Strict | Collision-free |
|---|---|---|---|
| ACT | 32/48 (66.7%) | 16/48 | 20/48 |
| PACT-raw | 34/48 (70.8%) | 20/48 | 30/48 |
| Readout + skin (prefetch-8) | 20/48 (41.7%) | 12/48 | 25/48 |
| Readout skin = 0 | 5/48 (10.4%) | 2/48 | 23/48 |

Raw is the strongest place rate. Readout uses the 40×128 embedding; zeroing those tokens at eval drops place 41.7% → 10.4% while collision-free stays about the same, so the policy still moves but the two-bottle squeeze fails. Every-control-step skin was 17/48 and is not this suite.
