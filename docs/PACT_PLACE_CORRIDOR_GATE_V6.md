# PACT pick-and-place corridor: Phase 0 attempt-6 (not run)

This would have been the cluttered-shelf follow-up to the passing v5 gate.
It is **not a FAIL of a 24-row screen**. The official contract was never
frozen, and no official episode was run.

v5 remains the last Phase-0 decision:
[`docs/PACT_PLACE_CORRIDOR_GATE_V5.md`](PACT_PLACE_CORRIDOR_GATE_V5.md)
(`PACT_PLACE_CORRIDOR_PHASE0_PASS`, 22/24). The four earlier FAIL records
are unchanged.

The attempt narrative is [`docs/PACT_PLACE_ATTEMPT6.md`](PACT_PLACE_ATTEMPT6.md).

## Decision

`PACT_PLACE_CORRIDOR_PHASE0_NOT_RUN`

A0 required an 8-episode expert probe with **zero** `pact_clutter` contact.
The closest eligible set (`|y| = 0.22`) had clutter on 6/8 rows. The farthest
eligible set on the declared grid (`|y| = 0.28`) had clutter on 3/8 rows. All
clutter was outbound carry. The plan forbids shrinking boxes, moving the tray
or target, or widening the corridor, so work stops here.

Collection, encoder work, policy training, and learned-policy evaluation are
not authorized by this document.

There is no `configs/pact_place_corridor_v6.json`.

## What was measured (diagnostics only)

| Measure | Closest set (A0 13) | Farthest set (A0 14) |
|---|---:|---:|
| Slot `|y|` (m) | 0.22 | 0.28 |
| Height (m) | 0.06 | 0.06 |
| Completed probe rows | 8/8 | 8/8 |
| Task success | 7/8 | 7/8 |
| Clean under v6 rule | 1/8 | 4/8 |
| `pact_clutter` contact | **6/8** | **3/8** |
| Inbound clutter | 0/8 | 0/8 |
| Hazard contact | 0/8 | 0/8 |
| Other-environment | 0/8 | 0/8 |
| Tray outside placement | 0/8 | 0/8 |

Config SHA-256 values are diagnostic dumps, not frozen gates:

- `|y| = 0.22`: `0f4d80580e1aa2963906378c377e7d61e4bee65afa719e74c7ca9e799b9778d8`
- `|y| = 0.28`: `eea681436be34f03e3ccffa6db2cbf7a8845ad15492363fa8f8e617f7a8430be`

A0 static sweep SHA-256 (unchanged):
`e34038b9e4a32e5b84729f62d5dc1a851b40c3ad2aa11b6d79bccc461c3526ae`.

## Clean-success rule that was never gated

Had Phase 0 run, a clean success would have meant task success **and** zero
`hazard_bar` **and** zero `other_environment` **and** zero `clutter` **and**
zero `place_receptacle` contact outside placement (`preplace` counted as
placement). That rule is stricter than v5. Do not compare a future cluttered
count to v5's 22/24 as if they were the same endpoint.

The drafted prediction band was 19–22 of 24, bar 20. It was never recorded in
a frozen config.

## Next action

None from this gate. A different clutter design (not a shrink of these boxes)
would be a new attempt with a new A0 grid, not a continuation of this freeze.
