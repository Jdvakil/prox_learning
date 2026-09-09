# V10.10 wrist280 dataset amendment — September 6, 2026

The user requested stopping collection with the available accepted demonstrations
instead of waiting hours to fill the last cell quotas. The live check found 280
accepted episodes, so this continuation selects those 280 and stops new attempts.
The original wrist288 plan, configuration, seed registry, code, raw artifacts and
full collection ledger remain preserved in their original locations.

## Frozen cohort and split

- Use the first **280 strict-clean accepted episodes with actual parent-observed
  zero exit receipts**, in committed ledger order. Freeze their ledger prefix in
  `amendments/wrist280/selected_ledger.jsonl` before conversion or training.
- Let already-running collection jobs finish. Preserve their raw data and exit
  receipts; episodes accepted after the frozen cutoff are outside this cohort.
- All 24 cells are represented: 21 have 12 episodes; F2 left center, left neg5 and
  left pos5 have 8, 9 and 11, respectively. This is an explicit departure from the
  original balanced collection design, motivated by observed collection yield.
- Split **240 train / 40 validation**. Allocate validation counts in proportion
  to the selected cell counts using largest remainders of `40*n_cell/280`, with
  at least one per cell. Break equal remainders by SHA256 of the original namespace,
  `wrist280_validation_apportionment`, existing cell split-seed derivation and cell.
  Rank episodes within each cell using the original frozen split-hash rule.
- Freeze exact episode identities, split labels, per-cell counts, selected raw
  hashes, amendment code/doc hashes and an effective configuration before training.
- Compute statistics from the same 240 training episodes for both ACT and PACT.

## Training and evaluation

Keeping 240 training episodes preserves **30 updates per epoch at batch size 8**,
**2,000 epochs**, and **60,000 actual optimizer updates** for every model. The
3,000-update resume schedule, 30,000-update restart/snapshot, 900-update benchmark
segments, model architecture, optimizer settings and all seed pairings remain as
specified in the original plan.

Validation has 40 episodes (one or two per cell) rather than 48. The smaller and
less balanced validation set must be reported. Evaluation still uses the final
60,000-update checkpoint. The 24-pair development screen, common-history rule,
PACT >=13/24 and >=3 advantage gate, three training seeds, frozen final 150 pairs,
pairing tolerances, success/safety metrics, deadline and reporting reserve remain
unchanged. This amendment does not authorize relaxing clean-data acceptance,
changing the environment or expert, replacing policy failures, or claiming that a
historical qualification passed. All `authorizes_*` flags stay false.

## Continuation and monitoring

Archive the original supervisor's documented drain/closure before restarting the
same experiment under `scripts/pact_wrist280_amendment.py`. Original frozen inputs
must still verify, and the effective configuration adds the new cohort and adapter
bindings. Resume conversion/training only after all collection exits are observed.
The capacity checker can stop because collection is closed; evaluation retains
its original 14 -> 12 -> 10 resource policy. Retry a transient resource-query fork
failure up to twice before surfacing it; worker failures retain their original
pause/receipt behavior.

Keep the standalone Python hourly audit and parent-chat one-hour command-line
waits. The separate Codex monitoring task stays disabled and archived.
