# Superseded V10.10 full evaluation, attempt 01

Retained, not deleted. 3 of 80 rollouts completed in 105 minutes before the
run collapsed. No `full_run.json` was written, so nothing downstream consumed
it. The 30 partial rollout directories are preserved here under
`partial_rollouts/`; each holds only a `rollout.log`, no result and no accepted
initial observation.

## Cause

Resource exhaustion, not the SSH disconnection. From the rollout logs:

    libgomp: Thread creation failed: Resource temporarily unavailable

followed by CUDA initialisation failures on the surviving workers.

The cgroup allows **3840 PIDs**. Each rollout process starts MuJoCo, PyTorch and
warp, and with no pinning OpenMP sizes its pool to the 128 visible CPUs. Ten
concurrent rollouts therefore ask for well over a thousand threads before any
real work begins, and the pool broke. The runner's own counter then jumped from
75 to 80 in a single second because every outstanding future resolved at once
when the process pool died -- the "80/80" line is an artifact of that collapse,
not eighty completed rollouts.

`run_pact_place_v108_collect.py` has always pinned these pools through
`THREAD_POOL_ENV`; the evaluation runner inherited from V10.9 never did. It went
unnoticed because the four-instance smoke runs only eight rollouts at four
workers, which stays under the budget.

## Fix

The evaluation worker environment now pins `OPENBLAS_NUM_THREADS`,
`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS` and
`VECLIB_MAXIMUM_THREADS` to 1, matching the collector, and the resumed run uses
fewer workers. The three completed rollouts are kept and skipped on resume.
