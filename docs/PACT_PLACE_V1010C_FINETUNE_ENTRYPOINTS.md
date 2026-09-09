# 128D PACT fine-tuning: training and evaluation entry points

These are the scripts used for the completed ACT / PACT-Finetune / PACT-Frozen comparison on seeds 3103, 3104 and 3105.

| Purpose | Seed 3103 | Seeds 3104 and 3105 |
|---|---|---|
| Training worker | [pact_v1010c_train.py](../scripts/pact_v1010c_train.py) | [pact_v1010c_multi_train.py](../scripts/pact_v1010c_multi_train.py) |
| Evaluation worker | [pact_v1010c_eval.py](../scripts/pact_v1010c_eval.py) | [pact_v1010c_multi_eval.py](../scripts/pact_v1010c_multi_eval.py) |
| Encoder, dataset and policy integration | [pact_v1010c_core.py](../scripts/pact_v1010c_core.py) | [pact_v1010c_multi_core.py](../scripts/pact_v1010c_multi_core.py) |
| Training/evaluation supervisor | [pact_v1010c_run.py](../scripts/pact_v1010c_run.py) | [pact_v1010c_multi_run.py](../scripts/pact_v1010c_multi_run.py) |

The multi-seed scripts select the seed through `PACT_FINETUNE_SEED=3104` or `PACT_FINETUNE_SEED=3105`. The training worker takes `--stop-updates`; the completed training target was 60,000 updates. The evaluation worker takes `--job`, pointing to a prepared evaluation-job JSON. The supervisor prepares the jobs, runs training and evaluation, and records observed worker exits. The multi-seed supervisor's `pipeline` stage covers both additional seeds.

The core adapters construct the trainable proximity encoder, configure the policy's 128-dimensional proximity input, and execute the copied main-branch training definitions. The evaluation worker produces the same live causal 128D features from the checkpoint's paired encoder. The byte-identical upstream sources are retained in [the upstream directory](../diagnostics_output/pact_place_v1010c_readout_s3103/upstream), with their hashes in [upstream_manifest.json](../diagnostics_output/pact_place_v1010c_readout_s3103/upstream_manifest.json).

These scripts use the retained experiment layout, original dataset and pretrained encoder. A fresh checkout also needs the dataset, model artifacts and runtime dependencies restored at the configured paths. Generated checkpoints, raw HDF5 trajectories, datasets, videos and runtime logs remain local under the repository's artifact exclusions. Source, reports and selected supporting metadata are committed.

The retained fine-tuned model pairs are `policy_last.ckpt` and `prox_encoder.pt` in these local directories:

- Seed 3103: `diagnostics_output/pact_place_v1010c_readout_s3103/checkpoint/`
- Seed 3104: `diagnostics_output/pact_place_v1010c_readout_s3_v1/seed3104/checkpoint/`
- Seed 3105: `diagnostics_output/pact_place_v1010c_readout_s3_v1/seed3105/checkpoint/`

[Three-method results](PACT_PLACE_V1010C_THREE_METHOD_COMPARISON.md) · [Three-seed execution plan](PACT_PLACE_V1010C_THREE_SEED_PLAN.md)
