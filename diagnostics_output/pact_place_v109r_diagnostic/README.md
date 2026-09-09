# V10.9R diagnostic — bulk rollout data removed 2026-09-02

Deleted to reclaim disk during the V10.11c 100-episode collection:

- `event/`  (977 MB) — event-decoder rollout trajectories and videos
- `legacy/` (977 MB) — legacy-averaging rollout trajectories and videos

Retained: `diagnostic_run.json` and `diagnostic_analysis.json`, which carry the
per-episode results and the comparison the diagnostic was run to produce.

Consequence: the numbers survive, the raw rollouts do not. Re-deriving anything
not already in the two JSON files requires re-running
`scripts/run_pact_place_v109r_diagnostic.py`.
