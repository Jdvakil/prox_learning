# Wrist288 three-seed execution

Started 2026-09-06 09:39 UTC; fresh collection started 09:59 UTC.

- Final precollection binding: `config.json` (SHA256 payload `c3a2a2440ee04e9d924e98d738ea833bbd3b51121f077ddc3f3f9fdb1c0307e0`).
- Environment preflight: 24/24 cells passed. Contract tests: 14 passed.
- Supervisor: current PID in `state.json`; log in `supervisor.log` (initial PID 139439).
- Resource/hourly monitor: current PID in `monitoring/monitor.json` (initial PID 139438).
- Requested monitoring task: `hourly_run_check`, thread `01a0762a-7ca1-7b53-882d-542324aeedca`.
- Pipeline: strict-clean collection 288 -> conversion/split 240/48 -> seed 3103 training -> four smoke pairs -> 96 development rollouts -> prespecified gate -> remaining seeds and 300 final rollouts only if the gate passes.
- Checkpoints: 60,000 actual optimizer updates; 30,000-update snapshot and 3,000-update resume bundles retained. No test-selected checkpoint.
- Reporting reserve starts September 7, 23:00 UTC; hard cutoff September 8, 00:00 UTC.

`EVAL.md` contains the experiment verdict/report. Until the supervisor publishes a verified completion or safe shutdown, the experiment is running and the target is unmeasured. `SAFE_SHUTDOWN.json` marks closure; its verdict must be read, not inferred from file existence.

The `precollection_revision_00/` directory preserves the brief precollection pause used to finish hourly receipt audits and thread caps. It contains no collection attempts. Environment, seeds, quotas and scientific settings were unchanged; the final code/configuration was frozen before the first fresh collection launch.

Historical results and both submodules are unchanged. All experiment `authorizes_*` flags remain false.

User-authorized collection capacity amendment (September 6, 10:20 UTC): keep 14 through 11:00 UTC, then assess 16; assess 18 only after another 15-minute observation period with sufficient measured headroom. The controller is PID 164650; its current decision is in `monitoring/capacity_status.json`, with the immutable policy/code binding in `monitoring/capacity_amendment.json`. Existing pressure, disk and deadline guards remain active.

The original supervisor has no live resize interface. If 16 fits after 11:00, the controller lets all current attempts finish, verifies actual exit receipts, archives the temporary closure in `monitoring/capacity_handoff_archive/`, and resumes the same collection ledger with the new collection-only pool. The hourly monitor restarts as part of this handoff. During this documented handoff, the temporary root closure is not the final experiment verdict; consult `monitoring/capacity_status.json`. Every upsize/downsize is recorded in `monitoring/concurrency_changes.jsonl`. Scientific bindings, quotas, seeds, training and evaluation remain unchanged.

Monitoring amendment (September 6, 13:03 UTC): the separate Codex `hourly_run_check` task was asked to stop at the user’s request to conserve usage. The parent chat now waits one hour between reports. The standalone Python hourly audit, worker/crash supervision and resource/capacity controller continue unchanged; this supersedes the earlier statement that a separate Codex task remains active. See `monitoring/monitoring_mode.json`.

User-authorized dataset amendment (September 6, 19:22 UTC): collection is closed at a frozen cohort of **280 accepted episodes**, split **240 training / 40 validation**. Exact identities, per-cell train/validation counts, and new code bindings are in `amendments/wrist280/cohort.json` and `effective_config.json`. The latter is the active training binding (SHA da3388ab016166721679bb2f5b66d5c218ecfff24d348b3b3634aba9dc89cdcc); original `config.json` remains the preserved precollection binding. Later in-flight outcomes are retained in the full raw ledger outside the selected cohort.

The dataset now contains 12 episodes in 21 cells and 8/9/11 in the three F2-left cells; validation has one or two episodes per cell. The 60,000-update budget, 240 training episodes, paired seeds, development gate, final manifests and inference/safety rules are unchanged. See `docs/PACT_PLACE_V1010_WRIST280_AMENDMENT.md`. The old closure is preserved under `amendments/wrist280/prior_closure/`. The capacity controller has stopped because collection is closed; the amended Python monitor runs normally, and the parent chat retains one-hour command-line waits.

September 7 resource recovery: the first development batch at 14 workers reached 95.2% VRAM and 92.8% RAM. The supervisor reduced its target while 14 jobs were still finishing, then paused based on those same in-flight jobs. All 14 completions were preserved. Evaluation now resumes at the authorized **10-worker fallback**, and further backoff waits for existing jobs to reach the previous target. The scientific configuration/checkpoints are unchanged. Recovery code/tests and evidence are bound in `amendments/resource_recovery_20260907/contract.json`; the old closure is archived alongside it. The existing measured training benchmark is adopted during continuation.

2026-09-07T04:38:20.915867+00:00: user-requested 12-worker evaluation continuation, with scoped closed-artifact file-cache release and original resource thresholds. See `amendments/eval_capacity_20260907/contract.json`.

2026-09-07T05:47:33.606513+00:00: Explicit user-directed continuation after development gate miss: 50 total final pairs, 17/17/16 across seeds; remaining models retain 60,000 updates. See `amendments/final50_20260907/contract.json`.

2026-09-07T16:00:52.720538+00:00: Final evaluation cache coverage corrected for .hdf5 and unused fully-trained checkpoint files; resumed remaining 68 rollouts with all completed data preserved. Probe released 31.4 GiB. See `amendments/final_cache_recovery_20260907/contract.json`.

2026-09-07T18:09:10.256377+00:00: Uniform infrastructure retry of 12 unreceipted final-3105 rollouts after fork exhaustion killed monitoring/receipt collection; all originals quarantined. Remaining 8 unstarted rollouts unchanged. GPU queries now use in-process NVML and resource-query errors preserve exit polling. See `amendments/receipt_recovery_20260907/contract.json`.

2026-09-07T19:00:54.893467+00:00: User requested seed 3105 expanded to n=50 pairs. Reuse 16 completed pairs, evaluate remaining 34 from original frozen manifest, preserving prior three-seed report. Extension report: `amendments/seed3105_full50_20260907/EVAL.md`.
