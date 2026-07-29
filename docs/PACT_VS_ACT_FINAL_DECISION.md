# PACT versus ACT final decision

## Experiment status

Decision: `PACT_EXPERIMENT_INCOMPLETE`

## Environment adequacy gate

Phase 1 decision: `PACT_ENVIRONMENT_ADEQUATE`. Science classifications: {'gate_b': 'adequate', 'gate_c': 'adequate', 'surface_observability': 'adequate'}.

- Usable clean expert demonstrations: 58/64; ordinary task success: 59/64.
- Active-panel surface signal: 54.0% of pre-grasp steps inside 20 cm, 18.1% inside 12 cm; 100.0% of scientific episodes active.
- Pilot ACT collision-free task success: 23/64; ordinary task success: 23/64; intrusion contact in 23/64.

## Frozen proximity front-end

The route-matched surface encoder has 819,172 parameters and checkpoint SHA-256 `d0ba9db102f2f7abe6eb46054f1a79ee292d9084bf7d1e38235b65fe600f9e6d`.

- Held-out mean / median Euclidean error: 3.26 cm / 1.88 cm.
- Within 2 cm: 51.3%; validity precision: 99.9%; recall: 99.9%.

## Policy training

| Arm | Seed | Best epoch | Validation loss | Checkpoint SHA-256 |
|---|---:|---:|---:|---|
| ACT | 3101 | 1904 | 0.084780 | `a5ebbf3d5537315337e17e0f28951de068ce6960974d0f282b77fcfcca672eb1` |
| ACT | 3102 | 1829 | 0.099043 | `e98d98bad87e2762cef37eb953d9ab55fcb65ed6355d2d8e9a881f38ef48c8d4` |
| PACT | 3101 | 1968 | 0.083446 | `a90cd4087c5666c665235f49b24ce8c9635475a9b9a40c213926d93a8c4e11cd` |
| PACT | 3102 | 1793 | 0.097197 | `880f01b5a2cb6056bc522012c3cf261d0acde16575df402b072522b3dee34d86` |

PACT_ZERO was not separately trained; it uses the corresponding PACT checkpoint with all 40 proximity tokens zeroed at inference.

## Frozen confirmatory design

160 held-out instances × 3 arms × 2 checkpoint seeds = 960 rollouts, 8 fixed workers, one fresh subprocess per row. Schedule SHA-256: `b6d9b3f7a87fef328a87db725e405ab4c48fe88d720ef3f7da094fda05110a8f`.

Using the frozen pilot ACT collision-free rate 0.359, 160 independent instances per arm have approximately 80% power at two-sided alpha=0.05 for a 0.155 absolute increase under a conservative unpaired normal approximation. The second checkpoint seed per instance and pairing receive no power credit.

The preregistered schedule did not reconcile, so no endpoint comparison was interpreted.

- Expected rows: 960
- Valid rows: 1
- Missing: 959
- Non-complete driver rows: 0
- Invalid rows: 0

## Dispatch integrity and interruption

The predeclared launch-smoke row passed in one invocation and one attempt. Its boundary, result, driver, schedule-row, checkpoint, and recorded hashes reconcile. The smoke endpoint was not interpreted.

After the full pool was released, eight additional rows accepted initial observations, but the evaluator process group disappeared before any of them wrote a scientific result. All eight logs stop after initial-observation acceptance without a traceback. No kernel OOM or GPU Xid was observed in the audit window; the exact external initiator is unknown.

Under the frozen boundary rule, those eight rows are terminal post-boundary failures and cannot be replaced or rerun. The remaining 951 rows were never started after irrecoverability was known, because further rollouts could not restore a valid confirmatory decision.

The frozen analyzer sees 1 valid row and 959 rows without valid results. The incident ledger resolves that latter count into 8 terminal post-boundary rows and 951 never-started rows. No endpoint comparison, Fisher test, or instance bootstrap was interpreted.

Incident SHA-256: `901868dd06ada31c119f9bad3ca4882b0c72e9c43cd2d7a1cbfd1b74f5a64d54`.

## Decision

The final line is the exact allowed decision token.

PACT_EXPERIMENT_INCOMPLETE
