# PACT valid-ablation lossless storage contract

The 40-row valid-ablation dispatch predeclares lossless storage compaction before
the smoke rollout. Schedule rows 1 through 38 are selected solely by fixed index,
independent of arm or outcome. Rows 0 and 39 remain fully unpacked.

For each eligible completed row, the original `result.json` and `trajectory.h5`
are archived byte-for-byte with zstd level 1, verified by streaming decompression
against the original SHA-256, and retained in recoverable form. The compact
`result.json` keeps every field used by the frozen analyzer. Videos and process
provenance remain unchanged.

The transform does not change or inspect the schedule, token plan, scene,
checkpoint, endpoint, analysis, or decision rule. The exact schedule, dispatch,
analyzer, compactor, wrapper, and output-root hashes are frozen in
`configs/pact_valid_ablation_storage_amendment_v1.json`.
