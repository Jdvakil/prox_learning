# One-seed V10.11c pilot

Current status, 2026-09-06 08:23 UTC: **stopped at failed pairing gate**. Both models finished 30k updates and all 100 original final workers exited 0. One of 50 initial-image pairs failed the gate; the single pre-recorded both-arm repeat failed it again. No further rollouts or tolerance changes are authorized by this recovery plan. The normal stage 08 and successful completion report were not run/issued. An independent diagnostic recount of all 102 original/repeat trajectories completed, preserving the failure, in `evaluation/blocked_raw_audit.json`. Original task outcomes are ACT 14/50 and PACT 7/50, unchanged by the repeat. Root `EVAL.md` contains full descriptive endpoints, limitations and actual failures. The hourly monitor has been stopped; no scoped PPID-1 spawn workers were found. Ask for direction before further repair. The design and earlier progress log below are historical.

Owner request: implement the audited remedies, test ACT and PACT on one training seed, aim to finish within twelve hours. Start: 2026-09-05 23:03 UTC. Deadline target: 2026-09-06 11:03 UTC.

## Frozen design

- Seed 3103 for both arms; new weights, not a continuation of the old models.
- Original 99 strict-clean demonstrations, ledger-derived 75/24 split; no new collection.
- Correct forward targets: observation[t] -> commanded_action[t+1]. Two terminal status-only transitions excluded. New derivative: 37,869 transitions, 28,564 training transitions.
- Both policies receive wrist and exo_camera_1 RGB at 240x320. Reuse the recorded videos, not newly rendered training images.
- Frozen 32-D encoder unchanged; 40 sensors, one token each, link1–link6 only. Existing embeddings copied exactly, excluding the two invalid terminal transitions.
- Chunk 100, original history-100 temporal ensemble, decay 0.01, threshold 127.5. No chunk 25 or averaging ablation.
- 30,000 updates/arm, 3,000 epochs, batch 8, learning rate 1e-5, KL 10, hidden 512, feed-forward 3200, seven encoder/seven decoder layers. Resume bundles retained. Exact snapshots at 20k and 30k updates.
- Four infrastructure-smoke pairs; twelve development pairs at both snapshots; fifty distinct final-test pairs at the shared selected snapshot. Total expected: 156 rollouts, of which 100 are the final test.
- Symmetric checkpoint selection: total two-arm development task successes, then collision-free successes, then fewer hazard frames, then earlier update count. Final-test outcomes cannot influence selection.
- Exact collection sampler/module/branch and hybrid camera system. 900 control steps, end_on_success false, action noise false. H5 only.
- Eight native thread-pool caps set to 1. Evaluation concurrency 12. Owner amendment at 2026-09-06 00:56 UTC caps this at 12 (10 is a fallback), superseding the original optional 14-worker gate.
- Source corpus and historical artifacts remain read-only. All authorization flags remain false.

## Durable processes and handoffs

`training_launch.json` identifies the detached two-arm training supervisor; `pilot_launch.json` identifies the detached completion supervisor. Do not launch duplicates. Actual model process exits are recorded in `models/`; subsequent actual stage exits are in `stages/`. Training and pilot supervisors print to `training.log` and `pilot.log`; per-stage logs are in `logs/`.

Completion supervisor waits for both training receipts, reruns unit tests, strictly reloads all four milestone checkpoints, verifies both cameras and proximity affect their appropriate policies, then runs smoke, development, selection, final evaluation and raw-artifact auditing. It stops on actual failures instead of declaring skipped stages successful. `pilot_failure.json` records a failure; `pilot_completion.json` records completed raw-verified results. The final report is `evaluation/REPORT.md`; append the completed findings to the root `EVAL.md` after review.

Per-rollout `worker_completion.json` receipts and role ledgers are authoritative for process completion. `result.json` and progress lines alone are insufficient. The scheduler reconciles completed receipts on a restart but refuses to silently retry a failed or already-accepted episode. Inspect and preserve any failed stage before an explicit repair.

Conversion completed and independently verified every source hash and output tensor. Its detached launch did not retain a parent exit-code receipt; do not invent that exit code. `conversion_verification.json` records the completed verification, and both trainers independently hash-check all converted H5 files. `conversion_metadata_clarification.json` discloses inherited parent-directory/token annotations in the immutable as-run conversion manifest; actual training commands/manifests bind the correct new directory and tree.

Expected first measured training speed: about 0.195 seconds/update, approximately 1.6–1.7 hours/arm. Evaluation ETA remains provisional until measured in the two-camera runtime. Container limits, not host totals: 15.36 CPU cores, 170.9 GiB RAM, one NVIDIA A10 with 23,028 MiB VRAM, pids.max 3840.

When finished, verify source preservation, append actual final numbers to `EVAL.md`, and check for scoped orphan multiprocessing.spawn workers with PPID 1. Never alter or remove the original datasets/checkpoints, and never cd to the parent worktree.

## Recovery 01 — 2026-09-06 02:16 UTC

ACT completed 30,000 updates (actual exit 0, 104.3 minutes). PACT failed in data-loader queue shutdown at epoch 1423 with actual exit 1, `RuntimeError: can't start new thread`. The original completion supervisor stopped on that failure. All failure evidence is retained in `recovery_01/`, including the original 1,423-line epoch log and the atomic checkpoint at epoch 1400 / step 14,010. The canonical epoch log is reconciled to that checkpoint before appending resumed epochs; 220 completed updates and any partial failed-epoch work are replayed.

`recovery_launch_01.json` identifies the recovery supervisor. It resumes the same PACT model/optimizer/RNG to 20k, renews the process, then resumes to 30k. Data, sampling configuration and four loader workers remain unchanged. Explicit PyTorch intra/inter-op caps are 1, with process/cgroup thread counts logged at each 1,000-step milestone. Numerical bit-for-bit continuation is not claimed. `pilot_launch_02.json` identifies the replacement completion supervisor, still capped at 12 evaluation workers.

## Recovery 02 — 2026-09-06 03:38 UTC

Both models are now fully trained at 30,000 updates. Strict reload, both-camera sensitivity and PACT proximity sensitivity passed at 20k and 30k. The first smoke stage failed: all eight workers exited 1 during metadata export because the new adapter did not initialize `_input_proj_proximity_shape`, required by inherited `get_info()`. No complete results were published. The metadata-only fix derives the shape from loaded weights; actions, models, sampler and seeds are unchanged. Five regression tests passed, including this exact runtime loader/exporter boundary. All failed smoke attempts, logs and receipts are under `recovery_02/`, with explicit old-to-archive path mappings. `pilot_launch_03.json` identifies the restarted supervisor; it repeats all four smoke pairs before development and final evaluation. Failed attempts are not counted as successful rollouts.

## Hourly monitoring and pairing exception — 2026-09-06 06:57 UTC

The read-only hourly verifier is `scripts/monitor_pact_place_v1011c_dualcam.py`; its PID and immutable reports are under `hourly_monitor/`. It writes local reports, not chat notifications, and does not automatically retry workers. The 04:52 and 05:52 checks passed. Development completed 48/48 with actual exit 0 and selected the shared 30k snapshots by the frozen symmetric rule. At 06:52, 60/100 final rollouts had exit-0 receipts, but the hourly initial-image pairing audit failed for instance `af02b30513b020e021abbcbdb050dd1688bafd0396ba2f783a5c9b39cb47e80d`. All 459 non-RGB initial datasets match exactly, as does table RGB. Wrist RGB differs in 45/658,944 channel values: 42 by one level and three by two levels. The existing maximum is one level; it is not relaxed. See `pairing_audit_20260906T0654.json` for the complete 30-pair read-only audit.

The remaining scheduled rollouts continue. `pairing_recovery_plan_01.json` records a bounded same-instance, both-arm infrastructure repeat after the schedule finishes, preserving original failure evidence and reporting both attempts. No new seeds, checkpoint changes, outcome-based eligibility, relaxed threshold, or repeated-until-pass loop is permitted. If the repeat fails, request direction rather than claiming verified completion.
