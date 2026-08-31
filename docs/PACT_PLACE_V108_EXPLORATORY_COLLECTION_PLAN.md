# PACT place V10.8: exploratory demonstration collection

> **This is an explicit owner override for scientific curiosity. It is NOT a
> Phase-0 pass.** V10.7's Phase-0 gate failed at **8/24** and is permanently
> closed. Nothing here changes, reinterprets, reruns, or supersedes that
> result. V10.8 uses new contracts, streams, manifests and output directories.

## Goal

Collect exactly **152 strict-clean, trainable demonstrations** in the unchanged
V10.7 environment: same static pendant geometry and three starting poses, same
expert policy, routes, speeds, sensors, and clean-success definition.

## Pipeline

The **full datagen pipeline** (`prepare_episode_for_saving` /
`save_trajectories`), never the expert-screen harness — that harness truncates
the sensor suite and is what left an earlier collection without proximity, RGB
or actions. Every accepted episode must carry:

- all five action arrays;
- `obs/agent/qpos` and `obs/agent/qvel`;
- wrist RGB, published by the pipeline as `episode_*_wrist_camera.mp4` and
  validated by decoding it and matching its frame count to `T`;
- all **40** proximity sensors as finite, non-constant `(T, 4, 8, 8)` float32.

Episodes are fresh. No V10.7 pool or Phase-0 row is reused.

## Quotas — frozen before the first attempt

Base **6** per cell over the 24 `family × side × pose` cells = 144, plus a
seventh for eight registered cells:

| family | +1 left | +1 right |
|---|---|---|
| F0 | neg5 | center |
| F1 | center | pos5 |
| F2 | pos5 | neg5 |
| F3 | center | pos5 |

Totals, asserted in code: **152** overall, **38/family**, **76/side**,
**50 neg5 / 51 center / 51 pos5**.

Quotas are never relaxed or redistributed. A cell leaves the rotation only when
its own quota is met; no other cell inherits its shortfall.

## Scheduling and seeds

Each cell has its **own** deterministic seed stream, frozen before anything
runs, so a cell needing many attempts never consumes another's seeds and the
schedule is reproducible in any visit order. Cells with unmet quotas are
scheduled in deterministic round-robin.

## Budget — hard stop, never extended

**900** scientific attempts or **16** wall-clock hours, whichever comes first.
If quotas remain incomplete the run stops and reports; it does not extend.

## Attempt ledger

Every attempt is appended to a durable, fsynced JSONL ledger — seed, cell,
outcome, defects, contact class totals, stability events, steps, and minimum
pendant clearance — **before** its heavy staging is pruned. Only accepted
successes keep full trainable files; a rejected attempt keeps its compact
record and its heavy artifacts are dropped. A torn final line from a crash is
ignored on resume rather than corrupting the tally.

## Preflight, before any attempt

Disk headroom; cgroup PID/CPU/memory; thread-pool pinning **before** any worker
is created; scene/geometry hash verification against the V10.7 certification;
trainable-schema validation; and resume/crash-safety that continues the frozen
attempt stream rather than restarting it.

## Schema smoke and infrastructure defects

The first registered attempt runs as an official schema smoke. Every success —
the first included — is fully schema-validated before it is counted.

A **schema or infrastructure defect is not a scientific outcome**: it is
recorded in a separate infrastructure list, the cell's scientific stream is
**not** advanced, the row is **not** replaced, and execution halts for repair.
The frozen contract authorizes no infrastructure retry.

## Final gate

`scripts/verify_pact_place_v108_dataset.py` independently re-reads every
accepted episode from disk, revalidates the full schema, re-checks quota
satisfaction cell by cell and the family/side/pose totals, and rejects corrupt,
partial, duplicate or non-clean episodes.

## Boundaries

Stop after the verified 152-success collection. **Do not convert, train, or
evaluate.** No historical artifact is deleted or modified to make space. Every
authorization field stays false.
