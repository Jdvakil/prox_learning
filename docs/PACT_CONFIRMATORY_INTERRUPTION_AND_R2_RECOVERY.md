# PACT confirmatory interruption and R2 recovery

## Executive conclusion

The launch-smoke rollout completed successfully and passed its integrity
audit. It was not the process that failed. After the smoke passed and the full
eight-worker pool was released, eight additional schedule rows accepted their
initial observations, but the evaluator process group disappeared before any
of those rows wrote a scientific result.

Closing the controlling screen is consistent with this synchronized process
loss and is a plausible explanation. It cannot be established from the
retained artifacts, however, so the exact external initiator is recorded as
unknown. The audit found no kernel out-of-memory event or GPU Xid in the
relevant window.

The interrupted experiment is `PACT_EXPERIMENT_INCOMPLETE`, not a statistical
null and not evidence that PACT failed. Under its frozen boundary rule, the
eight interrupted rows are terminal and cannot be rerun as part of the same
confirmatory experiment. A valid new comparison remains possible through a
separately preregistered R2 experiment using fresh held-out instances and the
unchanged frozen environment, policies, endpoint, and analysis.

## What completed successfully

Before confirmatory evaluation:

- The environment passed the adequacy gate:
  `PACT_ENVIRONMENT_ADEQUATE`.
- Pilot ACT collision-free task success was 23/64 (35.9%), leaving measurable
  headroom.
- ACT contacted the intrusion in a separate, disjoint set of 23/64 pilot
  episodes.
- The full dataset, frozen surface encoder, and two ACT plus two PACT policy
  seeds completed.
- The 960-row schedule was frozen at 160 instances × 3 arms × 2 checkpoint
  seeds, with eight workers.
- The analysis script, decision rule, checkpoint identities, and output root
  were bound by a pre-outcome dispatch contract.

The launch smoke itself then completed as intended:

| Audit item | Result |
|---|---|
| Schedule index | 0 |
| Arm / checkpoint seed | ACT / 3101 |
| Invocations | 1 |
| Attempts | 1 |
| Initial observation accepted | Yes |
| Scientific result written | Yes |
| Driver status / return code | `complete` / 0 |
| Pre-observation infrastructure failures | 0 |
| Endpoint inspected during smoke audit | No |

Its boundary marker, result, driver, rollout identity, schedule-row hash,
checkpoint hash, and recorded artifact hashes all reconcile. The launch-smoke
artifact SHA-256 is
`dfd038f42133242862fea75555360128784da8875b4da549b7fb69b73eb3188a`.

## What failed

After the full dispatch began, rows 1–8 all wrote durable
`initial_observation_accepted.json` markers. None wrote a scientific
`result.json`. Their process logs stop immediately after initial-observation
acceptance, without a traceback. The parent runner and all eight evaluators
then disappeared, and no normal execution summary was produced at the time.
The later outcome-blind reconciliation wrote a summary that explicitly
records the interruption rather than presenting it as a normal completed run.

The outcome-blind reconciliation found:

| Ledger category | Count |
|---|---:|
| Frozen schedule rows | 960 |
| Rows dispatched | 9 |
| Complete scientific rows | 1 |
| Terminal post-boundary failures | 8 |
| Never-started rows | 951 |
| Rows rerun | 0 |
| Rows launched after irrecoverability was known | 0 |

The generic frozen analyzer describes 959 rows as lacking valid results. The
more specific incident ledger resolves those as eight terminal
post-boundary rows plus 951 never-started rows.

## Why the eight rows cannot simply be rerun

The boundary rule was frozen before evaluation:

1. A failure strictly before acceptance of the initial observation is an
   infrastructure failure and may be retried.
2. Once an initial observation is accepted, the row is outcome-bearing and
   terminal. No subsequent exception, contact, task failure, success, process
   loss, or other event may cause replacement or rerun.

All eight interrupted rows crossed the second boundary. The rule deliberately
does not contain an exception for an accidental screen closure or a
content-independent parent-process failure.

This strictness prevents survivorship bias. Runtime and completion before a
simultaneous interruption can depend on arm behavior, trajectory duration,
contact behavior, rendering load, or simulator state. Even when the most
likely cause is unrelated to the endpoint, selectively rerunning only the
rows that did not survive would make the original schedule no longer the
predeclared single evaluation.

Rerunning the same eight rows—or restarting all 960 rows while retaining the
same confirmatory label—would also violate these frozen requirements:

- no outcome-bearing row is rerun or replaced;
- the schedule and worker count are fixed before the first rollout;
- the smoke row is part of the schedule and is reconciled, not rerun;
- the analysis and decision are not changed after outcomes begin.

An exact rerun of the old rows could be reported only as exploratory evidence.
It could not validly award `PACT_BENEFIT_ESTABLISHED`,
`PACT_NO_CONFIRMED_BENEFIT`, or `PACT_WORSE_THAN_ACT` under the interrupted
preregistration.

## Why the current result is incomplete rather than negative

Only one of 960 scheduled results exists. The frozen analyzer therefore
refuses to calculate or interpret:

- pooled collision-free task success;
- Wilson intervals by arm;
- Fisher exact comparisons;
- instance-clustered bootstrap intervals;
- task-success or contact comparisons.

Consequently, the interruption says nothing about whether PACT beats ACT or
PACT_ZERO. It also does not reverse the adequate environment gate. The
slightly lower PACT validation losses remain training diagnostics, not endpoint
evidence.

The exact decision for this execution is:

`PACT_EXPERIMENT_INCOMPLETE`

## Valid recovery: a fresh confirmatory R2

A new R2 can remain confirmatory if it is declared as a new experiment rather
than a repair or continuation of the interrupted rows.

### Frozen elements to carry forward unchanged

- Corridor scene, sampling distributions, sensor layout, and contact taxonomy
- Collision-free task success as the primary endpoint
- Ordinary task success as the secondary endpoint
- The frozen 819,172-parameter surface encoder
- The two ACT and two PACT checkpoint files and their SHA-256 identities
- PACT_ZERO as PACT with proximity zeroed only at inference
- Two-sided Wilson intervals and Fisher exact tests
- Whole-instance deterministic bootstrap with 20,000 replicates
- Decision rules and allowed tokens
- Eight evaluation workers

No retraining, scene tuning, threshold tuning, checkpoint selection, or
analysis change is needed or permitted.

### New elements required before R2 begins

1. Generate a new independently seeded population of 160 held-out
   confirmatory instances. None may be an interrupted schedule instance.
2. Freeze a new balanced 960-row schedule and record its detectable-effect
   statement before any R2 rollout.
3. Freeze a new dispatch contract binding the fresh manifest, unchanged
   checkpoint hashes, unchanged analysis-script hash, and a new empty output
   root.
4. Quarantine the interrupted output. It must not be pooled with, substituted
   into, or used to select R2 rows.
5. Run one predeclared R2 smoke row. It remains part of the R2 schedule.
6. Launch the full pool under a persistent supervisor that survives loss of
   the interactive screen, records a durable parent log and PID, and exposes a
   read-only heartbeat.
7. Reconcile all 960 rows before running the already frozen analysis.

The analysis script that was frozen for the interrupted schedule has SHA-256
`fd3c7f2e91a1737e248fc3ebe803018dcb4f9455d2b4e413d56946a4aebe25be`.
R2 should reuse that exact file unless a new preregistration explicitly
identifies and justifies a non-statistical harness correction before any R2
outcome.

### Estimated R2 duration

With eight workers:

- Fresh manifest, schedule, contract, and validation: approximately 1–3 hours
- One launch-smoke row: approximately 10–16 minutes
- Remaining evaluation: approximately 24–32 hours
- Reconciliation, analysis, tests, and reporting: approximately 1–2 hours

Expected end-to-end R2 duration is therefore approximately 27–37 hours. The
dominant uncertainty is simulator rollout throughput.

## Provenance

- Finalized interruption commit: `fc8ea67`
- Frozen schedule SHA-256:
  `b6d9b3f7a87fef328a87db725e405ab4c48fe88d720ef3f7da094fda05110a8f`
- Dispatch contract SHA-256:
  `81e99fe9e78cab1e19b019db396b149c396a7037a2121cf435b1ddea01f68151`
- Interruption record SHA-256:
  `901868dd06ada31c119f9bad3ca4882b0c72e9c43cd2d7a1cbfd1b74f5a64d54`

Authoritative artifacts:

- `diagnostics_output/pact_vs_act/confirmatory_interruption_v1.json`
- `diagnostics_output/pact_vs_act/analysis.json`
- `diagnostics_output/pact_vs_act/final_decision.json`
- `diagnostics_output/pact_vs_act/provenance.json`
- `docs/PACT_VS_ACT_FINAL_DECISION.md`

No R2 rollout has been authorized or launched by this report.
