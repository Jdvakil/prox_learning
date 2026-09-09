# Seed 3104 extension to 50 paired instances

The user requested completing n=50 for seed 3104 after the seed-3105 extension
finished, and explicitly extended the original deadline by three hours.

- Adopt the 17 completed seed-3104 pairs with their original jobs, observed
  zero-exit receipts, results, and initial-state bindings.
- Evaluate the remaining 33 pairs from the original frozen 50-instance
  `final_3104.json` manifest: 66 additional rollouts, 100 total for this seed.
- Use the existing ACT and PACT 60,000-update checkpoints and history 100.
  Training data, models, inference, environment, metrics, and pairing rules
  remain unchanged. No further training or outcome-based replacement occurs.
- Use the separate full-stage name `final_3104_full50_h100`, while worker
  identities and directories retain `final_3104_h100`. Preserve the original
  17-pair schedule, ledger, and completion file, the original three-seed
  report, and the completed seed-3105 extension report and artifacts.
- Target 12 rollout workers with the existing settled fallback to 10, resource
  thresholds, disk reserve, scoped cache release, and in-process NVML queries.
- Apply the user-authorized hard deadline of September 8, 2026, 03:00 UTC.
  Stop new launches at 02:00 UTC to reserve an hour for final validation.
  Apply these times to both the supervisor and monitor in this adapter;
  preserve the original frozen common module and experiment contracts.
- Keep the separate Codex task disabled. Continue the existing local supervisor
  and monitor, with parent-chat hourly command-line waits and user updates.
- Require 100 unique valid rollout completions and 50 passing initial-state
  pairing audits. Report prior 17, additional 33, and full 50 results separately
  under `amendments/seed3104_full50_20260907/`.

The development gate remains missed. This is a user-directed follow-up; all
historical qualification and `authorizes_*` fields remain unchanged.
