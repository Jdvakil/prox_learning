# PACT follow-up data-equivalence and zero-support audit

## Decision

The reused ACT checkpoint is not confounded by a different non-proximity
training payload. ACT retraining is not required.

This audit does not change the valid-ablation decision:
`VALID_ABLATION_WEAK_SIGNAL` does not clear Step 1, so no fresh confirmatory run
is authorized.

## ACT versus 32-D PACT conversion

The old ACT conversion and new embedding conversion contain the same 255
episode filenames and identical per-episode lengths. The only HDF5 dataset
added by the new conversion is `observations/proximity_embeddings`.

All four non-proximity training payloads are exactly equal after decoding in
every episode:

| Dataset | Old/new aggregate payload SHA-256 |
|---|---|
| `action` | `edbac49d55bcca09c7353d4f802ca744d0b4613b61da77e9014af323d467638c` |
| `observations/images/wrist_camera` | `618e9796477c88576500a3adf70f7b175069a41b61a7a4f3cb63a99a004c009a` |
| `observations/qpos` | `0cb4910e5c56bf12a9656e0967addaf0f08e914141843b96d405e2112e6b80d3` |
| `observations/qvel` | `fd20c7194a82f64835131b7292056fc8447bb3c663418a4b0640df4816936096` |

Normalization-statistics files are byte-identical and semantically identical,
SHA-256
`1fff47c6d6e75fce68d953bfef5029ffbad5794d08854ea9d0f7dafadc7be6ec`.
The train/validation indices and episode IDs are identical, as is the full
training recipe. The split-manifest hashes differ (`04684e80…` versus
`7d25e884…`), but this is metadata/versioning rather than a different split:
all 199 train and 56 validation assignments match exactly.

## Old 3-D versus new 32-D zero support

The same full 199-episode training partition contains 1,247,040 sensor tokens
in each representation.

| Quantity | Old 3-D token | New 32-D embedding |
|---|---:|---:|
| Exact zero vectors | 1,184,764 (95.0%) | 0 (0.0%) |
| Norm below 0.1 | 1,186,598 | 0 |
| Norm mean / median / minimum | 0.0090 / 0.0000 / 0.0000 | 6.3123 / 6.3182 / 6.0157 |
| Zero distance from mean, median / max | 0.13σ / 0.22σ | 2.20σ / 5.89σ |
| Dimensions beyond 3σ | 0/3 | 10/32 |

Every one of the 1,184,764 invalid old 3-D tokens is exactly zero, while none
of the 62,276 valid tokens is zero. Thus the old zero ablation represented the
ordinary in-distribution “no valid surface” state. In the new embedding space,
zero never occurs and lies far outside the tight norm shell. The old and new
PACT_ZERO experiments are therefore not comparable instruments.

The machine-readable audit is
[`diagnostics_output/pact_valid_ablation/data_and_zero_support_audit.json`](../diagnostics_output/pact_valid_ablation/data_and_zero_support_audit.json),
SHA-256 self-hash
`4a4c9f57f63067528390a302acbf4ae9eef4eeda301e1a1dc4bac5537e641eb5`.
