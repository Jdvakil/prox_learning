# Wrist280 evaluation capacity amendment — September 7, 2026

The user requested raising evaluation concurrency to 12 or 14 to shorten the run.
Use **12 evaluation workers**, retaining the original resource thresholds and the
settled 12 -> 10 fallback. Measured 14-worker VRAM was 95.2%, above the 85% guard;
12 workers project to approximately 81.6%. Verify the full running pool.

Ten-worker RAM peaked near 79% while Linux retained completed data in file cache.
A scoped `POSIX_FADV_DONTNEED` probe released approximately 21.8 GiB of file cache
from closed experiment-owned raw/converted data. This operation preserves file
contents. The continuation releases cache only for completed experiment-owned
HDF5 files, before evaluation startup, after hourly validation, and at minute
resource samples with RAM >=73%. Model checkpoints and environment assets retain
their caches. Keep the original 80% RAM pressure threshold; do not subtract cache
from reported usage or raise any limit.

Drain the current pool normally, preserve actual observed exit receipts, archive
the closed supervisor's reporting artifacts, and adopt every completed rollout.
Resume pending work using `scripts/pact_wrist280_eval_capacity.py`. Frozen original
code and both prior amendments remain unchanged. The new contract binds this
adapter, tests, document, user request and prior recovery contract.

Use the current target (12, or 10 after resource backoff) for the deadline estimate.
Measure actual rollout time; linear scaling is an estimate, not a promised gain.
Training budgets/concurrency, weights, seeds, scientific instances, scoring,
pairing audits, gate thresholds and deadline remain as previously specified.
The separate Codex task stays disabled; continue parent-chat hourly waits.
