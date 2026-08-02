# R2 authorization: proceed with four changes

Your interruption report is accepted: `PACT_EXPERIMENT_INCOMPLETE` is the right call, the
environment gate stands, and the incident ledger is sound. Proceed to R2 with four changes.

## 1. Amend the boundary rule before any R2 rollout — this is the priority

As written, ~8 rows are in flight at any instant across a ~30-hour run, so any process-group
loss makes them terminal and voids the entire R2. That turns one stray signal into 30 hours of
lost compute. Replace it with:

> A row is terminal once it produces a scientific result. An in-flight row lost to
> indiscriminate external process termination — one that kills all live rows without regard to
> their content — is an infrastructure failure and is re-run. Whenever such a loss occurs,
> **every** row in flight at that moment is re-run, never a selected subset.

This preserves the anti-survivorship intent exactly: if all in-flight rows are re-run and none
had a result, no selection is possible.

## 2. Your terminality reading was stricter than the statistics require

Your own audit records `Endpoint inspected during smoke audit: No`, and the 8 interrupted rows
wrote no result — so **zero endpoint outcomes have been observed**. With nothing observed there
is nothing to select on, and re-running all 960 rows on the same instances under a new
preregistration would also be valid.

Fresh instances are still cleaner, so keep that plan **if** the held-out pool can supply 160
non-overlapping instances; if it cannot, re-run the same 960 under a new preregistration rather
than blocking R2.

## 3. Harden the supervisor and prove it

`setsid`/`nohup` so the pool reparents to init, durable parent PID and log on disk, read-only
heartbeat. Then **verify by killing the launching shell during the smoke row** and confirming
the pool keeps running. Do not take survival on faith.

## 4. Measure throughput before planning around 35 hours

A comparable 8-worker rollout pool on this same A10 sustains **1.7 rollouts/min**, which would
put 959 rollouts near 9.4 hours — your 24–32 hour estimate implies 0.5–0.67/min. Measure over
the first 20 minutes of the full dispatch and revise the ETA from data.

## Also

Do not inspect the surviving smoke row's endpoint. It is the only confirmatory datum that
exists and your audit correctly avoided it — keep that intact through R2 setup.

## Carried forward frozen and unchanged

Scene, endpoint, encoder, the four checkpoint hashes, PACT_ZERO as inference-time zeroing,
Wilson + Fisher + 20,000-replicate instance bootstrap, decision tokens, 8 workers, analysis
script `fd3c7f2e91a1737e248fc3ebe803018dcb4f9455d2b4e413d56946a4aebe25be`.
