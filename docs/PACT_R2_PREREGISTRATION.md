# PACT versus ACT confirmatory R2 preregistration

## Authorization and scope

R2 is a new confirmatory experiment authorized by
`/root/prox_learning_hybrid_safety/docs/PACT_R2_AUTHORIZATION.md`, SHA-256
`0f6c44eadbdfbb799041f3aa9a0809db80a7c615a3b04bb300be92432bbe1300`.
It does not repair, continue, pool, replace, or reinterpret the interrupted R1
schedule.

R1 remains `PACT_EXPERIMENT_INCOMPLETE`. Its output root is read-only and
quarantined. The surviving R1 smoke endpoint has not been inspected and will
not be loaded during R2 setup, execution, or analysis.

## Carried-forward scientific contract

The following remain unchanged:

- the `pact_collision_corridor_v1` scene and sampling distributions;
- collision-free task success as the primary endpoint;
- ordinary task success as a secondary endpoint;
- the contact taxonomy and grasp-target exemption;
- the frozen 819,172-parameter surface encoder, SHA-256
  `d0ba9db102f2f7abe6eb46054f1a79ee292d9084bf7d1e38235b65fe600f9e6d`;
- ACT seeds 3101 and 3102, checkpoint SHA-256s
  `a5ebbf3d5537315337e17e0f28951de068ce6960974d0f282b77fcfcca672eb1`
  and
  `e98d98bad87e2762cef37eb953d9ab55fcb65ed6355d2d8e9a881f38ef48c8d4`;
- PACT seeds 3101 and 3102, checkpoint SHA-256s
  `a90cd4087c5666c665235f49b24ce8c9635475a9b9a40c213926d93a8c4e11cd`
  and
  `880f01b5a2cb6056bc522012c3cf261d0acde16575df402b072522b3dee34d86`;
- PACT_ZERO as the corresponding PACT checkpoint with all proximity tokens
  zeroed only at inference;
- 160 held-out instances, three arms, two checkpoint seeds, and 960 rows;
- exactly eight evaluation workers;
- Wilson 95% intervals, two-sided Fisher exact tests, and a deterministic
  20,000-replicate whole-instance bootstrap;
- the decision rules and allowed decision tokens;
- frozen analysis script SHA-256
  `fd3c7f2e91a1737e248fc3ebe803018dcb4f9455d2b4e413d56946a4aebe25be`.

There is no retraining, checkpoint reselection, scene tuning, threshold
tuning, endpoint change, or analysis change.

## Fresh held-out population

R2 uses a new independently seeded 160-instance confirmatory population.

- Manifest schema: `pact_confirmatory_r2_manifest_v1`
- Manifest master seed: `2026073001`
- Schedule seed: `2026073002`
- Bootstrap seed: `2026072902`

Every R2 episode ID must be disjoint from all 624 episode IDs in the R1
candidate manifest. Intrusion side remains exactly balanced 80 left / 80
right. Geometry jitter ranges, task-seed derivation, scene-template identity,
and maximum task-sampling retries remain unchanged.

If deterministic generation or validation cannot produce 160 disjoint rows,
R2 stops before rollout. The authorization permits a same-instance R2 fallback,
but this preregistration deliberately does not invoke it because fresh
capacity is available.

## Amended result and recovery boundary

A row is terminal once it produces a valid scientific `result.json`.

An in-flight row lost to an indiscriminate external process termination is an
infrastructure failure and is rerun. Whenever such a loss occurs, every row
in flight at that moment is rerun, never a selected subset.

A qualifying indiscriminate termination is identified without endpoint data
by either of these mechanical conditions:

1. the live supervisor observes every evaluator PID in its atomically frozen
   active cohort exit within a five-second window, with no valid scientific
   result for any cohort row; or
2. after supervisor restart, the durable active-cohort record contains no live
   evaluator PID and no valid scientific result for any cohort row.

Before a qualifying cohort is rerun, the supervisor must atomically write a
self-hashed recovery event containing the complete active cohort, evaluator
PIDs, attempt indices, boundary-marker presence, result absence, and process
log hashes. Every row in that event is then requeued. A row that already has a
valid result is never rerun.

An isolated or otherwise non-qualifying post-observation evaluator loss does
not authorize a selected retry. It stops R2 as
`PACT_EXPERIMENT_INCOMPLETE`. Pre-observation failures remain individually
retryable because they contain no accepted observation or scientific result.

The R1 boundary rule remains unchanged for R1. This amendment applies only to
the not-yet-started R2 schedule.

## Detached supervisor requirement

R2 runs through a dedicated supervisor started with `setsid` and `nohup`, with
stdin detached. It must persist:

- its PID and session identity;
- a durable parent log;
- an atomically updated read-only heartbeat;
- the complete active cohort and child PIDs;
- per-row attempt ledgers;
- recovery events and execution summary.

Before full dispatch, the predeclared R2 smoke row is launched through this
supervisor. While the smoke is in flight, the launching shell is killed. Full
dispatch is forbidden unless:

- the supervisor remains alive and is reparented independently of that shell;
- the evaluator remains active or completes normally;
- the heartbeat continues to advance;
- exactly one valid smoke result is produced;
- the smoke row is reconciled into the full schedule and is not rerun.

## Evaluation and throughput

The fixed schedule uses one fresh evaluator subprocess per row and eight
concurrent workers. Arm/seed order is balanced. No output row is replaced
based on task, contact, or policy outcome.

Throughput is measured exactly 20 minutes after full dispatch begins using
only completion timestamps and row identities. No task-success, contact, arm
endpoint, or failure-taxonomy field is read. The measurement reports completed
rows per elapsed minute and revises the ETA without changing the schedule,
worker count, or analysis.

## Frozen analysis and decision

Analysis begins only after all 960 R2 rows reconcile to valid scientific
results. It reports pooled collision-free task success, Wilson intervals,
ordinary task success, contacts and failures by taxonomy, two-sided Fisher
tests, and whole-instance bootstrap intervals.

The decision rule is unchanged:

- `PACT_BENEFIT_ESTABLISHED`: PACT beats ACT and PACT_ZERO, both paired
  bootstrap lower bounds exceed zero, and both Fisher p-values are below 0.05.
- `PACT_WORSE_THAN_ACT`: the PACT-minus-ACT paired-bootstrap upper bound is
  below zero and the two-sided Fisher p-value is below 0.05.
- `PACT_NO_CONFIRMED_BENEFIT`: a reconciled experiment that meets neither
  rule.
- `PACT_EXPERIMENT_INCOMPLETE`: the R2 schedule does not reconcile.

No R2 rollout may begin until this preregistration, its machine-readable
counterpart, the supervisor implementation and tests, the fresh manifest, the
schedule, and the dispatch contract are committed.
