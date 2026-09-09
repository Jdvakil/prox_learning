# Execution review

Initial launch review completed, 2026-09-08. The offline audit is complete; this file does not certify execution completion.

The execution subagent is `/root/v1010b_execution`, GPT-6 Astra, xhigh. It owns the new V10.10b scripts and output directory. The parent is reviewing scientific identity, instrumentation, continuation, and gate behavior before accepting execution results.

Confirmed from the actual inherited source:

- The evaluated baseline is `policy_update_60000.ckpt`; the unused best checkpoint remains unused.
- Direct construction of the new configuration retains the wrist288 configuration, V10.10 sampler, 900-step horizon, no action noise, and active 32-D frontend method.
- Both fixed data loaders receive `chunk_size=100`. `episode_horizon=635` is a schema/configuration bound. The parent's earlier message suggesting 635-step loader padding was incorrect and was explicitly withdrawn before execution.
- The command record attributes `_v109_arm` and `_v109_gripper` are the actual inherited attributes. The original publisher is available directly without calling an old main function.
- Scalar and vector exponential age-weight calculations were checked locally and are exactly equal for ages 0–99.

Issues sent to the execution agent for correction before contract freeze:

- Parse the raw split manifest through its loader; raw storage uses `episodes` with split labels.
- Include keyword tensors in recorded model-input hashes, including PACT proximity.
- Report translation-only reference rim depth for ordinary trajectories as well as measured dynamic geometry where available.
- Use each vertical wall box's thin axis for its normal and distinguish the base collider.
- Require complete stage/pairing identities and denominators; an empty collection must not pass an `all()` gate.
- Include root H5 attributes in physical-state pairing.
- Avoid repeatedly loading all six checkpoint/resume bundles in each rollout process; stage-level verification and per-job selected-model checks suffice.

An independent call to the new metrics implementation on retained PACT-3104 scene `98c64a358f22eddaba4754f97d40325e9f3e9151f6968d553123681be884a33f` completed with exit 0. It reproduced failure, exclusive pickup failure, 29,701 physics samples, closure command 148, and eligible early depth. Its requested-versus-achieved closure gap was 17.9824 mm and maximum FK discrepancy 0.4558 mm. This checks compatibility with one historical mechanism case; it does not validate new replay instrumentation or establish population prevalence.

An independent real-data p0 fixture passed for both ACT and PACT, using converted training episode 0 and 16 NumPy seeds per arm. Every returned tensor and the post-sample NumPy RNG state matched the original `EpisodicDataset` exactly; action tensors were 100×8. During fixture setup, the parent caught a remaining label-builder mismatch: the normalized split loader calls the identity `source_episode_id`, not `episode_id`. This was sent to the execution agent. The passing dataset fixture used a directly verified single-row label, so it does not conceal that separate builder issue.

The execution agent corrected the reported split, keyword-input, rim, wall-normal, pairing-attribute and denominator issues. Its current preflight suite passed 18 tests, with a retained receipt at `../pact_place_v1010b_grasp_v1/preflight_tests.json`. The parent checked that receipt, the expected stage counts, and infrastructure amendment `01_prelaunch_parent_lifecycle`, which preserves the prior contract and runner. The amendment registers a child immediately after launch and routes the hard deadline through owned-process-group TERM then KILL escalation; healthy workers drain with observed exits on parent exceptions.

The parent released serial A1 smoke execution under effective contract SHA256 `4189d5aba788988a6e2f48f25d01fb1d96d67a703e8c18b725cfcef6845ca79a`. T0 is 2026-09-08 04:04:46 UTC; the hard deadline is 2026-09-10 04:04:46 UTC. Stage counts are A1 12, A2 72 analyzed/48 new, B 216 analyzed/180 new, C 216 new, and D 600 new. The initial full-path disk projection passes with approximately 80 MiB margin above the mandatory reserve and therefore remains a live stage gate. The executor was instructed to report first-pair action/outcome comparison and any mismatch immediately, then continue through the fixed gates autonomously.

Final stage status must be read from the new run's observed exit receipts, metrics, gates, and `parent_closure.json`. No training or final-result claim is made here.
