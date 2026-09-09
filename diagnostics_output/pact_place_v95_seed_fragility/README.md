# V9.5 seed fragility — expert-screen rows removed 2026-09-02

Deleted to reclaim disk during the V10.11c 100-episode collection:

- `expert_screen_rows/` (1.3 GB) — per-seed expert screen rollouts

Retained: `fragility.json` and `fragility_progress.json`.

`fragility.json` is deliberately kept: `pact_place_v106_contract.py:458` hashes
it into the V10.6 specification contract's `immutable_inputs`. Removing it would
not raise -- `file_hashes` records `"absent"` -- but it would silently change
that contract's payload hash.

Consequence: `scripts/audit_pact_place_v105.py` and the `index_corpus` step in
`scripts/run_pact_place_v105_reconstruct.py:334` both read the deleted rows
directory and can no longer run without re-generating it via
`scripts/run_pact_place_v95_seed_fragility.py`.
