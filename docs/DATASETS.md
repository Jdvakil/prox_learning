# Published datasets and the environments behind them

Dataset: [Ekshan267/pact_pick_n_place_v2](https://huggingface.co/datasets/Ekshan267/pact_pick_n_place_v2).

Hub folder names are labels, not environment versions. Read this table before
pointing an evaluation at any of these checkpoints.

| Hub split | Episodes | Environment version | Sampler | Repo package |
|---|---|---|---|---|
| `data/v12` | 165 | `pact_place_corridor_v10_11_preview_onebottle` | `PactPlaceCorridorV1010FourObjectSampler` | `environments/hf_v12` |
| `data/v12.1` | 5 | `pact_place_corridor_v10_11_preview_tablecam` | `PactPlaceCorridorV1010FourObjectSampler` | not shipped |
| `data/v1011d` | 200 | `pact_place_corridor_v10_11d_randomized_clutter` | `PactPlaceCorridorV1011DRandomizedLayoutSampler` | `environments/hf_v1011d` |

Everything above is read off the collect code that produced each dump.

## Two traps

**Hub `v12` is not `pact_place_corridor_v12`.** The 165 episodes came from the
V10.11 one-bottle preview environment. `pact_place_corridor_v12`, already in
this repo under `scripts/pact_place_v12_*.py`, is a different environment and
has no published dataset. Nothing about it is wrong; it just is not the source
of any data on the hub.

**The published `v1011d` metadata understates its environment.** The folder
manifest shortens the environment to `pact_place_corridor_v10_11d`, and the
folder README credits `PactPlaceCorridorV1010FourObjectSampler`. Both were
written by hand at publish time. The n200 collect imports
`pact_place_v1011d_contract`, which records
`pact_place_corridor_v10_11d_randomized_clutter` and
`PactPlaceCorridorV1011DRandomizedLayoutSampler`. Evaluating a v1011d
checkpoint against the four-object sampler is a domain mismatch: the policy was
trained on randomized clutter layouts and would be tested on a fixed one.

## Running them

```bash
python scripts/verify_hf_env.py --online      # confirm the checkout matches the data
python environments/hf_v12/collect.py --target 2 --max-attempts 4
python environments/hf_v1011d/collect.py --smoke-only --smoke-attempts 2
```

Prefix with the usual `OMP_NUM_THREADS=2 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl`.

## Why the submodule pin moved

Both dumps were collected against molmospaces
`experiment/pact-vs-act-remediation-v2` (`70dedc07`), and the pin now points
there. molmospaces `main` cannot replay either one: it has no
`PactPlaceCorridorV1011DRandomizedLayoutSampler`, it dropped
`data_generation/runtime_compat.py` that the v12 scripts import, and its
corridor policy is a trimmed rewrite. The previous pin (`89696ed`) predates the
corridor work entirely and carries no `PactPlaceCorridor*` classes at all, which
is why nothing in `scripts/pact_place_v12_*.py` could run from a clean clone.

Porting the corridor stack forward onto molmospaces `main` is worth doing, but
it needs a seed-equivalence test against these dumps and should not block
evaluation.

## Scenes

Scene files live in `custom_scenes/` and are byte-identical to their molmospaces
counterparts, so the sha256 values recorded in published rows still match. The
one-bottle preview scene, `pact_place_corridor_v10_11_center_preview.xml`, was
never committed to molmospaces; it is vendored here for the first time. Only its
recorded path changes, not its bytes.
