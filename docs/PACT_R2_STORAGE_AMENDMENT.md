# PACT R2 outcome-blind storage amendment

## Status

The R2 confirmatory dispatch was reversibly paused as a single isolated process
group after 175 of 960 scheduled rows completed. The filesystem had approximately
13 GB free, while the completed R2 artifacts occupied approximately 65 GB.
Continued execution without a storage change would have exhausted the filesystem.
No R2 endpoint result was inspected to make this decision.

The user explicitly authorized lossless compaction on 2026-07-30 and requested
that the final scheduled row remain directly analyzable without restoration. This
document freezes that operational amendment before any R2 scientific payload is
changed.

## Frozen transform

For schedule rows 1 through 958, once a row is present in the outcome-blind
completion ledger:

1. Preserve the original `result.json` byte-for-byte as
   `result.full.json.zst`.
2. Preserve the original `trajectory.h5` byte-for-byte as
   `trajectory.h5.zst`.
3. Verify each archive by streaming decompression and comparing the restored
   SHA-256 with the source SHA-256.
4. Publish a small `result.json` containing every field required by the already
   frozen analyzer, plus archive provenance.
5. Remove the unpacked trajectory only after its verified archive exists.
6. Retain all videos and process provenance unmodified.
7. Write a self-hashed per-row `storage_archive.json` recording sizes and hashes.

The transform applies to every eligible row by schedule index, independent of
arm, success, contacts, or any other endpoint value. The compactor neither prints
nor summarizes endpoint values.

Schedule row 0 remains fully unpacked because it is the already hash-frozen smoke
row. Schedule row 959 remains fully unpacked so the final row can be analyzed
directly immediately after completion, as requested. Both exclusions are fixed
before compaction.

## Scientific invariants

This amendment changes storage representation only. It does not change the scene,
instances, schedule, row order, checkpoints, endpoint, contact taxonomy, worker
count, evaluation code, frozen analysis, or decision rule. It does not authorize
replacement or outcome-based reruns. The frozen analyzer remains:

`fd3c7f2e91a1737e248fc3ebe803018dcb4f9455d2b4e413d56946a4aebe25be`

The compact analyzer view retains task success, collision-free task success,
failure taxonomy, contact-class totals, and all row/checkpoint identity fields.
The full original scientific result and trajectory remain byte-exactly
recoverable from their verified archives.

## Frozen binding

- Schedule payload SHA-256:
  `35e1377c9029f4934ff816b2d04c15f9134f232c7baa7136545565ea6b0057ad`
- Dispatch contract SHA-256:
  `c660694b9f2e6915c5fb5543508ad3f84f943c4edc8c7f3487b45315a730a173`
- Compactor file SHA-256:
  `1f4f5d149fe18d79c5a265cb31df8ffb9a4932827eb0f7444898f8ce6b305fc6`
- Storage amendment payload SHA-256:
  `ee7c2c08112449d406511b71fdc15eacd7a7e15f2a419261292cf34bf78c27b5`
- Output root:
  `/root/pact_remediation_artifacts_v2/confirmatory_r2_35e1377c`

After tests and commit, the existing completed backlog will be compacted while
the evaluator remains paused. The same eight stopped evaluator processes will be
continued only after archive verification and adequate free-space recovery.
