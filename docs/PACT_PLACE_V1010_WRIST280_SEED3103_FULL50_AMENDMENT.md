# Seed 3103 extension to 50 paired instances

On September 8 the user requested removing the environment audit and completing
seed 3103 to n=50. The prior three-hour deadline extension remains in force.

- Adopt the 17 original completed pairs with their unchanged jobs, observed
  zero-exit receipts, and results. Evaluate the remaining 33 pairs from the
  original frozen `final_3103.json` manifest: 66 new rollouts, 100 for this seed.
- Use the existing 60,000-update ACT and PACT checkpoints with averaging history
  100. No training, model changes, new instance selection, or outcome-based
  replacement is included.
- Use the separate stage `final_3103_full50_h100`, retaining the original
  `final_3103_h100` worker identities and directories. Preserve the original
  17-pair schedule, ledger, and completion file, the original 50-pair three-seed
  report, and both completed 50-pair seed-3104 and seed-3105 extensions.
- Target 12 workers with the existing resource guards and settled fallback to
  10. Retain scoped cache release, in-process NVML monitoring, and exit receipts.
- Keep the hard deadline at September 8, 03:00 UTC and stop new launches at
  02:00 UTC. Healthy in-flight rollouts finish normally. Admission estimates
  must place the last new launch before 02:00 and completion plus validation
  before 03:00; do not incorrectly treat the launch stop as a worker kill time.
- Keep the separate Codex task disabled. Use the existing local supervisor and
  monitor, and parent-chat hourly command-line waits and updates.
- Validate 100 unique completions and all 50 initial-state pairing audits.
  Report prior 17, additional 33, and full 50 results separately under
  `amendments/seed3103_full50_20260908/`. Once all seeds are complete, derive a
  separate full 150-pair summary without overwriting the original n=50 report.

The environment-audit directory and its four generated files were removed as
requested. Historical experiment artifacts and qualification flags are unchanged;
the original development gate remains missed and all `authorizes_*` flags remain
false.
