# PACT valid-ablation pre-outcome horizon amendment

The first detached smoke exposed an execution-feasibility defect before any
scientific result existed. The fixed collision-corridor config inherits a
900-step task horizon, while the initial PACT_PERMUTED token tensor contained
only 512 frames per row. Attempt 0 therefore stopped at token index 512 with
`RuntimeError: rollout exceeded frozen 512-step token plan` and wrote neither
`result.json` nor `driver_result.json`. The supervisor's generic recovery
started attempt 1; it was terminated after the deterministic defect was
identified, also before either terminal artifact existed.

No PACT_PERMUTED endpoint was available or inspected. The old schedule,
dispatch, token plan, and result-less output root are retained as failure
provenance and retired.

Before a replacement smoke, the token horizon is amended to the already-fixed
900-step environment horizon. Seed 2026073105 is unchanged. Each row samples
900 full 40-sensor × 32-D frames without replacement within that row; reuse
across different rows is permitted because 40 × 900 exceeds the 31,176-frame
training population. Adjacent frames still come from different source episodes.
Every supplied token remains an actual frozen-training-set token, and live-scene
alignment remains destroyed.

This changes no arm, checkpoint, instance, scene, worker count, endpoint,
contrast, analysis code, threshold, or interpretation. A new token-plan hash,
schedule hash, dispatch hash, and empty output root must be frozen, and the
detachment smoke must pass again before the full pool starts.
