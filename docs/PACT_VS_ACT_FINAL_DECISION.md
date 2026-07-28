# PACT versus ACT final decision

## Outcome

Decision: `PACT_ENVIRONMENT_INADEQUATE`.

The frozen collision-route environment did not pass its Phase 1 solvability prerequisite. The protocol therefore stopped before dataset conversion, proximity-encoder training, ACT/PACT training, or any three-arm evaluation. This is a valid preregistered early stop, not an unreconciled confirmatory schedule.

## Frozen Phase 1 result

| Check | Result | Threshold | Pass |
|---|---:|---:|---:|
| Expert ordinary task success | 20/24 | >=20/24 | True |
| Expert collision-free task success | 19/24 | >=20/24 | False |
| Episodes with panel signal | 20/24 | >=20/24 | True |
| Pre-grasp frames inside 20 cm | 1284/2495 (51.5%) | >=30% | True |
| Pre-grasp frames inside 12 cm | 492/2495 (19.7%) | >=5% | True |

The 24-row ledger reconciled as 20 successes, 1 sampling failure, and 3 infrastructure failures. All terminal rows count; none was replaced.

The sole outcome-bearing collision was pilot expert row 21 (`2ac0f1e4dccbc07e81b1d36ae5a29568fdcbf679f4a7078cf2f7a2dd7f42b7d8`): it completed the task but recorded 58 `hazard_bar` entries across 58 frames. `grasp_target` contact remained exempt and there were 0 `other_environment` entries.

## What was not run

Because one applicable prerequisite failed, Gates B/C were not run. There is no pilot ACT checkpoint, full dataset, frozen surface encoder, ACT checkpoint, PACT checkpoint, PACT_ZERO evaluation, Wilson interval, Fisher test, or paired bootstrap result. Running those steps would violate the frozen stop rule.

The planned 80-instance × 3-arm schedule is retained only as the preregistered design. Its checkpoint-bound rows were never instantiated and no confirmatory outcome was seen. The stopped schedule record has SHA-256 `1f6969d0101ebea6d045c1a49a476976b97da560bf1047db2bef0e7442f83da0`.

## Interpretation

The surface-signal guard passed, so this is not a repeat of the fridge scene's no-signal failure. The environment nevertheless missed its joint adequacy requirement: it was not sufficiently robustly solvable by the expert under the fixed seeds and contact endpoint. Consequently this run cannot establish whether PACT beats ACT or PACT_ZERO.

## Machine-readable artifacts

- `diagnostics_output/pact_vs_act/environment_gate.json`
- `diagnostics_output/pact_vs_act/schedule.json`
- `diagnostics_output/pact_vs_act/analysis.json`
- `diagnostics_output/pact_vs_act/final_decision.json`
- `diagnostics_output/pact_vs_act/provenance.json`

Analysis SHA-256: `0671b5492ce35c455d4703c70e3d06f0b73eaf664e20ac03eca622cbcc13f87b`. Final-decision SHA-256: `355704bf5054133c67edec58a37826fae7fb7c1cf668a7de31e09cdaea8c1d60`.

## Decision

The final line is the exact allowed decision token.

PACT_ENVIRONMENT_INADEQUATE
