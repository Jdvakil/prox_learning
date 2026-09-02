# V10.11c 100-Demonstration Collection

Status: authorized by the owner on 2026-09-02. The earlier 144-success target
was superseded before collection began. This plan ends at 100 accepted rows.

## Objective

Collect exactly 100 strict-clean expert demonstrations from the validated
V10.11c environment (`pact_place_corridor_v10_11c_33pct_taller_primitives`).
This collection does not claim a Phase-0 pass and does not authorize
conversion, training, or evaluation.

## Frozen population

- Use all 24 family × side × pendant-pose cells.
- Collect four successes in every cell (96 total), then one bonus success in
  each of four preregistered cells. This produces 25 per family, 50 per side,
  and pose totals 33 `neg5`, 34 `center`, and 33 `pos5`.
- Permit at most one in-flight attempt per cell. Never redistribute a quota or
  retain a row after its cell is full.
- Use the new deterministic collection stream and seeds in
  `pact_place_v1011c_collection_contract.py`; do not reuse preflight, review,
  V10.8, or V10.10 seeds.

## Admission and observations

An accepted row must pass the inherited strict-clean predicate: successful
task, grasp, lift, and place; no disallowed contact; no clutter stability
event; no infrastructure failure. It must also pass the training-schema
validator and the table-camera validator.

Each accepted row must contain:

- wrist-camera RGB;
- `exo_camera_1` RGB;
- per-frame `extrinsic_cv`, `cam2world_gl`, and `intrinsic_cv` for
  `exo_camera_1`;
- all 40 raw `(T, 4, 8, 8)` proximity tensors;
- actions and robot state required by the existing ACT/PACT converter.

Use summary-only contact-audit storage. This changes only diagnostic payload
size, not live contact classification or clean-row admission. Retain complete
accepted rows and prune heavy artifacts from rejected attempts only after an
append-only ledger entry is fsynced.

## Execution and stops

1. Re-hash and verify the V10.11c contract, visibility gate, 96-row preflight,
   six-video review manifest, three static scenes, implementation files, seed
   disjointness, and exact 100-row quota.
2. Run a disjoint one-row hybrid-camera smoke. Confirm its HDF5 schema, MP4
   decodability, and projected storage before the scientific collection.
3. Run with at most eight workers, 900 scientific attempts, 16 hours, and a
   2 GiB terminal free-space reserve.
4. Halt on the first infrastructure/schema defect. Scientific sampling or
   clean-row failures consume their registered attempt and continue.
5. Stop immediately when every frozen quota is met (100/100), or on the first
   budget/storage stop. Do not continue toward 144.

The close-out must report the full per-cell accepted/attempted table, failure
taxonomy, dataset and ledger hashes, table-camera validation, remaining disk,
and exact stop reason.
