# PACT held-out geometry generalization — attempt 3 execution record

Attempt 3 completed and reconciled. It replaces the abandoned, uninterpreted attempt-2 policy run
with a more efficient and better-powered allocation. No v2 policy result was opened, no v2 outcome
was analyzed, and all v2 artifacts remain preserved.

## Frozen allocation

| Field | v3 |
|---|---:|
| Conditions | C0, C2, `Z_093` |
| Arms | PACT, PACT_PERMUTED |
| Seeds | 3101, 3102, 3103 |
| Fresh instances per condition | 40 |
| Total rollouts | 720 |
| Fixed workers | 12 |
| Bootstrap replicates | 20,000 |

The effective sample size per condition is 40 instance clusters. The power-only approximation puts
the C0 interval half-width near 7.35 percentage points, compared with about 9.3 points at v2's 25
instances. ACT and `HALF_Y_030` were removed for the preregistered reasons recorded in
`PACT_GEOMETRY_GENERALIZATION_V3_PREREGISTRATION.md`.

## Frozen bindings

- Manifest: `800339f5b651664d4ed9ea5a3cbed2822c4426033d4b1c2e15c9c6b47460891c`
- Schedule: `00a4190a6ba4af021dd2e6f4cc1f0bd3b79d79fcfd36e3547935d0e0f98f1f79`
- Dispatch: `d8f8bc9b108f3170f38ab7d7bdb1349b0fffa12ce30ae53f34d6b9dd4c7e9e3a`
- Worker sizing: `f1e9557ac43be9d559ebad810b5c3327cdc2578f288ce7d1677d5a13d19bf3b8`
- Detachment proof: `19126c0483f178a734501b8c5beb2001210a5c55504c5dde865a0123f108161e`

Phase 0 was reused without rerun: C0 passed 11/12, C2 passed 12/12, and `Z_093` passed
11/12 clean expert episodes. The dropped `HALF_Y_030` condition remains recorded as an 11/12 pass.
The frozen checkpoint, encoder, contact endpoint, scene, and taxonomy hashes all verified before
dispatch.

## Launch audit

The detached smoke passed and its one scientific row was reconciled into the full schedule without
rerun. The full cohort started at 2026-08-07 01:38:41 UTC with supervisor PID 3282775, compactor PID
3282776, and throughput-monitor PID 3282777.

At simultaneous startup, ten workers initialized normally and two hit the container's 3,840-task
limit before accepting an observation. The two result-free, pre-observation processes were retried
under the frozen boundary rule after future child affinity was limited to 32 logical CPUs. No
post-observation process was stopped. All twelve full-run workers then accepted their initial
observations. GPU use was 18,576 MiB, below the 19,456 MiB ceiling; the fixed worker count remains
12. The machine-readable amendment and audit contain no endpoint fields.

At this launch audit the state was 1/720 complete (the smoke), 12 active, 707 pending,
`abort_reason=null`.

The fixed outcome-blind 20-minute window completed at 01:58:41 UTC. Ten full-run rows completed in
the window, or 0.50 rows/minute. With 709 rows remaining at the cutoff, the measured execution
projection is 23.63 hours; allow roughly 24–25 hours end to end for execution, reconciliation,
frozen analysis, the final report, and commits. The measurement used completion identities and
timestamps only and records `endpoint_fields_read=false` and `schedule_changed=false`.

## Completion

All 720/720 rows reconciled at 2026-08-07 16:53:47 UTC with no abort and no missing schedule cell.
Execution took approximately 15 hours 15 minutes from full dispatch. The preregistered analysis
then returned `GEOMETRY_GENERALIZES`: C0 reproduced, C2 and `Z_093` both favored PACT on both
contact endpoints, and both pooled-shifted modality confidence intervals excluded zero below zero.
The final report is `docs/PACT_GEOMETRY_GENERALIZATION_V3.md`.

EVALUATION_COMPLETE
