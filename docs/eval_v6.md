# V6 evals

`eval_act_v1011d.py` on this branch differs from Jay `main`: `--history consecutive`
is the 8-step causal window (prefetch last 7 idle chunk steps plus the query),
matching readout training. Main still samples skin on every control step.

`eval_act_v6.py` is unchanged. Same prefetch-8 protocol on the v6 world.

Four v6 runs on `eval_act_v6.py`:

| Arm | Ckpt | Flags |
|---|---|---|
| ACT | `20260915_113546_act_pick_n_place_v6_s0` | `--history query` |
| PACT-raw | `20260915_114210_pact_pick_n_place_v6_raw_s0` | `--history query` |
| PACT-readout prefetch 8 | `20260915_115435_pact_pick_n_place_v6_readout_s0` | `--history consecutive` |
| PACT-readout zeroskin | same readout ckpt | `--history consecutive --prox_ablate zero` |

n=48, `--spread_cells`. Train stays Hub HDF5.
