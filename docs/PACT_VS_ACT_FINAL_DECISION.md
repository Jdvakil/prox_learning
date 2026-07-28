# PACT versus ACT final decision

## Outcome

Decision: `PACT_EXPERIMENT_INCOMPLETE`.

The remediation-v2 collision environment passed the surface-signal test and produced enough clean expert demonstrations. The fixed pilot ACT evaluation nevertheless produced no scientific policy outcomes because every fresh evaluator subprocess failed before loading the manifest. The preregistered process-outcome rule makes those 64 ledger entries terminal, so none was rerun and the experiment stopped before scale.

This run therefore does not establish whether PACT beats ACT or PACT_ZERO.

## Environment evidence

| Measure | Result | Frozen threshold |
|---|---:|---:|
| Active-panel signal inside 20 cm | 54.0% (95% bootstrap 46.0%–61.9%) | >=30% |
| Active-panel signal inside 12 cm | 18.1% (95% bootstrap 13.5%–22.8%) | >=5% |
| Active scientific expert episodes | 62/62 | >=5/6 |
| Usable clean demonstrations | 58/64 | >=48 |
| Expert no-outcome rate | 2/64 (3.1%) | <5% |

Surface observability was robustly adequate and passed every leave-one-episode-out check. Gate A was not applicable because the predeclared route targets nearby surface geometry, not object position.

## Pilot policy and terminal evaluation ledger

Vision-only ACT seed 1101 completed 2,000 epochs. Its frozen best checkpoint was epoch 1849 with validation loss 0.093538 and SHA-256 `4fca3b0b0542d6ae65c7d44f1fd562cd376199532f91b08aaf5722109a858db6`.

The immutable pilot schedule contains 64 ACT rows, uses eight workers, and has SHA-256 `e0515adf10a12cca22412d349d37b56ec5400446894b450b0e84edbe139b564e`. All 64 driver entries are `invocation_failure`; there are zero scientific `result.json` files. Each evaluator was launched from the ACT submodule while receiving the relative manifest path `configs/pact_collision_candidate_manifest_v2.json`, which was not resolvable from that working directory.

The runner now resolves manifest and output paths before changing the evaluator working directory, with a focused regression test. That repair was made only after the terminal ledger existed and was not used to rerun any row.

## What was not run

Gates B and C are inconclusive because 0/64 rows have scientific ACT outcomes; the frozen minimum was 61. The full train/validation collection, surface encoder, full ACT and PACT seeds, PACT_ZERO ablation, 960-rollout confirmatory schedule, Fisher tests, and instance bootstrap were not run.

The pilot checkpoint is retained as provenance, but it is not one of the three final-arm checkpoints requested for a completed comparison. No claim is made from its validation loss.

## Machine-readable artifacts

- `diagnostics_output/pact_vs_act/provenance.json`
- `diagnostics_output/pact_vs_act/schedule.json`
- `diagnostics_output/pact_vs_act/analysis.json`
- `diagnostics_output/pact_vs_act/final_decision.json`
- `diagnostics_output/pact_vs_act/environment_gate.json`

Analysis SHA-256: `27734a416142372043c9d8f8511eb035b0dd609779551eb742b61e479d2d1c25`. Final-decision SHA-256: `08060e5faacedc66d6dd61f465a467e632d62053106c8586cae74484792b6533`.

## Decision

The last line is the exact allowed decision token.

PACT_EXPERIMENT_INCOMPLETE
