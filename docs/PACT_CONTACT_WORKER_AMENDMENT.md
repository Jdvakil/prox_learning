# Pre-dispatch amendment: raise the worker count from 8 to 12

## What changes

**Worker count only: 8 → 12.** Nothing else. Not the instances, arms, seeds, endpoints, episode
cap, analyzer, decision rule, or any threshold.

## Why this is legitimate right now, and only right now

No rollout has executed. Seed 3103 was still training when this was written, and the artifact tree
contains **zero** `result.json` files. The preregistration freezes the worker count *before the
first row*, so amending it before dispatch is a pre-outcome change of the same class as the
horizon-mismatch fix the previous smoke caught.

**Verify this before touching anything:**

```bash
find /root/pact_contact_endpoint_artifacts -name result.json | wc -l   # must be 0
```

If that returns anything other than 0, **stop** — the run has begun, the count is frozen, and this
amendment is void. Do not restart a partially executed schedule to change worker count.

## Why 12

At 8 workers the previous runs used **12–14 GB of 22.6 GB at ~52% GPU utilisation**, with 128 CPUs
available. The card was roughly half idle, so throughput should scale close to linearly.

| Workers | Projected throughput | 1,200 rollouts |
|---:|---:|---:|
| 8 (current) | 0.40 rows/min | ~50 h |
| 12 | ~0.60 rows/min | **~33 h** |

Episode length is the reason this run is slow at all — 900 control steps against conf41's 200,
which matches the 4.25× per-rollout time difference almost exactly. That cost is structural and
cannot be reduced without changing the endpoint, so parallelism is the only lever left.

## Choose the count from measured memory, not from this document

The 12 figure is a projection. Per-worker memory has been observed between **1,538 MiB** (12,304 /
8) and **1,771 MiB** (14,170 / 8), and those two projections differ materially at 12 workers —
18.5 GB versus 21.3 GB, the second of which leaves under 1.4 GB of headroom on a 22.6 GB card.

So compute it rather than assume it:

1. Take the **peak** GPU memory observed at 8 workers from the prior runs' records — use the
   largest value, not the average.
2. `per_worker = peak_8 / 8`
3. `N = floor(19000 MiB / per_worker)` — the 19 GB ceiling leaves ~3.5 GB of headroom
4. **Use `N`, capped at 12.** If `N` comes out below 10, stay at 8 and record why.

An OOM 20 hours into a 33-hour run is far worse than finishing in 40 hours instead of 33. Bias
toward headroom.

## Procedure

1. Confirm zero `result.json` files exist.
2. Compute `N` as above. Record `peak_8`, `per_worker`, and `N` in the amendment artifact.
3. Write a pre-outcome amendment record containing: the old count, the new count, the memory
   calculation, the reason, a statement that no outcome had been observed, and a SHA-256 self-hash.
4. Re-freeze the dispatch contract with the new worker count and record the new contract hash.
5. Run the launch smoke as already specified — one row, assert a scientific `result.json`, verify
   the horizon and token plan, and prove detachment by killing the launching shell.
6. During the smoke and the first minutes of the full pool, **sample GPU memory** and record the
   observed peak. If it exceeds 20 GB, abort before more rows finalize, drop to the next lower
   count, and re-freeze once.
7. **Re-measure the outcome-blind first-20-minute throughput** and record it. The 0.4 rows/min
   baseline no longer applies, and the reported ETA must come from the new measurement.

## What must not change

- 100 fresh instances; 4 arms (ACT, PACT, PACT_ZERO, PACT_PERMUTED); 3 seeds; 1,200 rollouts
- 900 control-step episode cap — contact frames accumulate over the episode, so shortening it
  changes the co-primary endpoint's scale. This is a measurement, not an optimisation target.
- Both co-primary endpoints, every secondary, the diagnostic conditioning, and all thresholds
- The frozen analyzer `e2d9a5061e3a26599fa03a9d4f147ceda8386eb8ad87481f5674d7022f681589`
- The decision rule and its tokens
- The dropped occlusion subset — the partition was degenerate at 285/285 and stays dropped
- The boundary rule: terminal once a scientific result exists; rows lost to indiscriminate
  external termination are infrastructure failures, and **all** in-flight rows are re-run

## Do not use this as cover for other changes

Worker count is the only field this amendment touches. Do not take the opportunity to adjust
instance count, arms, seeds, or thresholds — power is already marginal (MDE 890.96 hazard frames
against a historical effect of 896.83), and any change to the design under an amendment banner
would compromise the preregistration.

## Artifacts

- `diagnostics_output/pact_contact_endpoint/worker_amendment_v1.json` — old count, new count,
  memory calculation, zero-results proof, reason, self-hash
- updated dispatch contract with the new worker count and a fresh contract hash
- `throughput_first_20_minutes.json` regenerated under the new count
- the revised ETA, stated from the re-measured throughput rather than the projection above
