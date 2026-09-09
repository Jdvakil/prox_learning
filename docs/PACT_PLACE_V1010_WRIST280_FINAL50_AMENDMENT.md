# Wrist280 user-directed 50-pair follow-up — September 7, 2026

After the development gate missed (PACT 10/24, ACT 8/24), the user explicitly
requested proceeding with final evaluation and clarified: **50 pairs total across
three seeds**. Preserve the missed gate and its report. Continue by user direction;
do not change its recorded result or represent this continuation as gate-qualified.

Train both arms for seeds 3104 and 3105 to 60,000 updates, with the same data,
statistics, checkpoints, architecture, optimizer settings and serial scheduling.
Adopt the completed 3103 models. Use the already-selected common history of 100.

Freeze **17 / 17 / 16 pairs** for seeds **3103 / 3104 / 3105**, respectively,
before any final evaluation. This is 50 distinct physical instances, each evaluated
by ACT and PACT, yielding **100 rollouts**. Select from the original frozen final
manifests, leaving all source rows and sampling/retry bindings unchanged.

The deterministic subset gives every one of the 24 cells two pairs globally and
adds two extras on opposite intrusion sides and different pendant poses. Hash-sort
the cells using the experiment namespace; distribute two rounds cyclically across
seeds with a one-seed rotation between rounds. Add one previously unassigned cell
to each of seeds 3103 and 3104 using a deterministic hash tie-break. Select each
instance within its assigned cell by a separate namespace/hash ranking. No outcome
enters this selection. Freeze exact identities and source manifest hashes.

Keep 12 evaluation workers with settled fallback to 10 and scoped release of
completed-artifact file cache. Keep the 900-step horizon, no action noise,
end_on_success=False, raw HDF5 retention, strict paired initial-state/RGB audits,
resource guards and September 7 23:00 UTC stopping time / reporting reserve.

Report per-seed counts before pooling, with three-seed mean and pooled success,
collision-free success, hazard/contact/stability and failure-stage metrics.
Label this a reduced, user-directed follow-up. The original 150-pair final target
is not assessed at its specified sample size; do not claim it passed. Preserve
the existing sensor limitations and all false authorization/qualification flags.

Keep the separate Codex task disabled and continue parent-chat hourly waits.
