# PACT front-end screen outcome-blind storage amendment

## Status

The required detached smoke completed as schedule row 0. Its scientific payload
occupies approximately 805 MiB, while the filesystem had approximately 16 GiB
free. The preregistration estimate of approximately 53 MiB per row was therefore
too small, and continuing the 120-row dispatch without a storage transform would
exhaust the filesystem.

No task-success, collision-free-success, contact, failure-taxonomy, or other
endpoint value was inspected before this amendment was frozen. The storage
decision used file sizes and protocol identity only.

The frozen dispatch already declared that raw schedule rows 0 and 119 would be
preserved. This amendment implements that predeclared storage requirement before
the full eight-worker dispatch begins.

## Frozen transform

For schedule rows 1 through 118, after a row appears in the outcome-blind
completion ledger:

1. Preserve the original `result.json` byte-for-byte as
   `result.full.json.zst`.
2. Preserve the original `trajectory.h5` byte-for-byte as
   `trajectory.h5.zst`.
3. Verify both archives by streaming decompression and comparing the restored
   SHA-256 with the source SHA-256.
4. Publish a small `result.json` containing every field required by the already
   frozen screen analyzer, plus archive provenance.
5. Remove the unpacked trajectory only after its verified archive exists.
6. Retain all videos and process provenance unmodified.
7. Write a self-hashed `storage_archive.json` containing sizes and hashes.

Selection is by fixed schedule index only and is independent of arm or outcome.
The compactor does not print or summarize endpoint values. Rows 0 and 119 remain
fully unpacked, so the smoke and final scheduled row can be inspected directly.

## Scientific invariants

This amendment changes storage representation only. It does not change the
scene, instances, schedule, row order, checkpoints, endpoint, contact taxonomy,
worker count, evaluator, frozen analysis, or decision rule. It does not authorize
replacement or outcome-based reruns.

The full original result and trajectory remain byte-exactly recoverable from
verified archives. The compact analyzer view retains task success,
collision-free task success, failure taxonomy, contact-class totals, and all
row/checkpoint identity fields.

The exact frozen bindings are recorded in
`configs/pact_frontend_screen_storage_amendment_v1.json`.
