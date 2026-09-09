# Superseded V10.9 infrastructure smoke, attempt 01

Retained, not deleted. `infrastructure_healthy: false`, 0 of 8 rollouts
completed, so the full evaluation was correctly refused and no evaluation
instance was ever touched.

Cause, from every rollout log: `episode ID resolved to 0 rows`.

The frozen manifest deliberately keeps the 40 evaluation instances under `rows`
and the four infrastructure-smoke instances under `smoke.rows`, so that neither
a later reader nor the analysis can mistake a smoke rollout for an evaluation
instance. But both the V10.9 wrapper and the inherited evaluator resolve an
episode by scanning `rows`, so no smoke episode id was ever findable.

Fix: `load_manifest_all_rows` merges the two lists **in memory only** when the
evaluator resolves a row. The stored manifest is unchanged, the analysis still
reads the 40 evaluation instances from `rows` alone, and episode ids stay
distinct because the role is part of their preimage (44 rows, 44 unique ids).

This is exactly what the smoke stage exists to catch: an infrastructure defect
found in 1.0 minutes of wall clock instead of after 80 rollouts.
