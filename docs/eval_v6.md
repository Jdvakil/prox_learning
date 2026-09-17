# V6 evals

`eval_act_v1011d.py` on this branch is the prefetch-8 FrozenACT evaluator from
[`Jdvakil/prox_learning` main](https://github.com/Jdvakil/prox_learning/blob/main/eval_act_v1011d.py).
Do not edit it. `--history consecutive` is the 8-step causal window.

`eval_act_v6.py` is the same protocol with the v6 world. Do not edit
`eval_act_v1011d.py` to add v6.

Four v6 runs on `eval_act_v6.py`:

| Arm | Ckpt | Flags |
|---|---|---|
| ACT | `20260915_113546_act_pick_n_place_v6_s0` | `--history query` |
| PACT-raw | `20260915_114210_pact_pick_n_place_v6_raw_s0` | `--history query` |
| PACT-readout prefetch 8 | `20260915_115435_pact_pick_n_place_v6_readout_s0` | `--history consecutive` |
| PACT-readout zeroskin | same readout ckpt | `--history consecutive --prox_ablate zero` |

n=48, `--spread_cells`. Train stays Hub HDF5.
