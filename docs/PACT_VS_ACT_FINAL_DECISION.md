# PACT versus ACT final decision

## Experiment status

Decision: `PACT_NO_CONFIRMED_BENEFIT`

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

160 held-out instances × 3 arms × 2 checkpoint seeds = 960 rollouts, 8 fixed workers, one fresh subprocess per row. Schedule SHA-256: `35e1377c9029f4934ff816b2d04c15f9134f232c7baa7136545565ea6b0057ad`.

Using the frozen pilot ACT collision-free rate 0.359, 160 independent instances per arm have approximately 80% power at two-sided alpha=0.05 for a 0.155 absolute increase under a conservative unpaired normal approximation. The second checkpoint seed per instance and pairing receive no power credit.

## Primary and secondary endpoints

| Arm | Collision-free task success | Wilson 95% CI | Ordinary task success | Wilson 95% CI |
|---|---:|---:|---:|---:|
| ACT | 170/320 (53.1%) | [47.7%, 58.5%] | 177/320 (55.3%) | [49.8%, 60.7%] |
| PACT | 159/320 (49.7%) | [44.2%, 55.1%] | 169/320 (52.8%) | [47.3%, 58.2%] |
| PACT_ZERO | 160/320 (50.0%) | [44.6%, 55.4%] | 169/320 (52.8%) | [47.3%, 58.2%] |

## Preregistered comparisons

- PACT_minus_ACT: -3.4%, paired-instance bootstrap 95% CI [-8.4%, +1.6%], two-sided Fisher p=0.429036.
- PACT_minus_PACT_ZERO: -0.3%, paired-instance bootstrap 95% CI [-1.6%, +0.9%], two-sided Fisher p=1.

## Contact totals

- ACT: pair entries {'grasp_target': 70569690, 'hazard_bar': 1029374, 'other_environment': 771}; episodes {'grasp_target': 275, 'hazard_bar': 67, 'other_environment': 1}.
- PACT: pair entries {'grasp_target': 72087757, 'hazard_bar': 1191065, 'other_environment': 1120}; episodes {'grasp_target': 280, 'hazard_bar': 69, 'other_environment': 2}.
- PACT_ZERO: pair entries {'grasp_target': 70913878, 'hazard_bar': 1196616, 'other_environment': 1131}; episodes {'grasp_target': 278, 'hazard_bar': 71, 'other_environment': 2}.

## Failure taxonomy

- ACT: {'collision_free_task_success': 170, 'target_contact_without_task_success': 81, 'hazard_bar_contact': 67, 'other_environment_contact': 1, 'task_failure_after_gripper_close': 1}.
- PACT: {'target_contact_without_task_success': 88, 'collision_free_task_success': 159, 'hazard_bar_contact': 69, 'other_environment_contact': 2, 'task_failure_after_gripper_close': 2}.
- PACT_ZERO: {'target_contact_without_task_success': 85, 'collision_free_task_success': 160, 'hazard_bar_contact': 71, 'other_environment_contact': 2, 'task_failure_after_gripper_close': 2}.

## Decision

The final line is the exact allowed decision token.

PACT_NO_CONFIRMED_BENEFIT
