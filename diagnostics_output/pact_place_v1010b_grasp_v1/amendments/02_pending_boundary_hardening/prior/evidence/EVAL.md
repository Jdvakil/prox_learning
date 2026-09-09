# V10.10b bounded grasp-acquisition experiment

Status: **STOPPED_AT_GATE_A**.

frozen replay action/outcome difference remains unexplained; intervention blocked

Completed new rollouts: 2/1056 maximum main path. Final comparison: 0/600.
Completed continuation branches: 0/12; new optimizer updates: 0/36000.

| Stage | New rollouts completed | Main-path expected |
|---|---:|---:|
| A1 | 2 | 12 |
| A2 | 0 | 48 |
| B | 0 | 180 |
| C | 0 | 216 |
| D | 0 | 600 |

The unchanged update60000 ACT/PACT baselines remain runnable. This execution does not alter historical qualification or authorization flags.
Scene blocks and training seeds are separate identities. Curated replays are selected for mechanism inspection; crossed historical scenes are exposed diagnostics.
A prerequisite stop completes the bounded experiment without demonstrating a successful fix. Missing/infrastructure endpoints are not imputed or removed from scientific denominators.
All writes are confined to the new scripts/tests and this experiment directory. The pre-existing root EVAL.md is preserved.

smoke_replay_comparison.json:
```json
{
  "comparison_eligible": false,
  "comparisons": [
    {
      "actions": {
        "arm": {
          "exact": false,
          "first_different_step": 0,
          "max_abs": 0.022661924362182617
        },
        "gripper": {
          "exact": false,
          "first_different_step": 407,
          "max_abs": 255.0
        },
        "model_output": {
          "exact": false,
          "first_different_step": 0,
          "max_abs": 32.06278991699219
        }
      },
      "arm": "ACT",
      "failure_stage_exact": true,
      "initial_pairing_passed": true,
      "original_directory": "/root/prox_learning_pact_remediation/diagnostics_output/pact_place_v1010_wrist288_s3_v1/evaluation/final_3104_h100/final_3104_h100_ACT_98c64a358f22edda",
      "outcome_exact": true,
      "replay_directory": "/root/prox_learning_pact_remediation/diagnostics_output/pact_place_v1010b_grasp_v1/rollouts/A1/A1_ACT_3104_frozen60000_98c64a358f22edda/attempt_00",
      "scene_id": "98c64a358f22eddaba4754f97d40325e9f3e9151f6968d553123681be884a33f"
    },
    {
      "actions": {
        "arm": {
          "exact": false,
          "first_different_step": 0,
          "max_abs": 0.005877137184143066
        },
        "gripper": {
          "exact": true,
          "first_different_step": null,
          "max_abs": 0.0
        },
        "model_output": {
          "exact": false,
          "first_different_step": 0,
          "max_abs": 2.2701034545898438
        }
      },
      "arm": "PACT",
      "failure_stage_exact": true,
      "initial_pairing_passed": true,
      "original_directory": "/root/prox_learning_pact_remediation/diagnostics_output/pact_place_v1010_wrist288_s3_v1/evaluation/final_3104_h100/final_3104_h100_PACT_98c64a358f22edda",
      "outcome_exact": true,
      "replay_directory": "/root/prox_learning_pact_remediation/diagnostics_output/pact_place_v1010b_grasp_v1/rollouts/A1/A1_PACT_3104_frozen60000_98c64a358f22edda/attempt_00",
      "scene_id": "98c64a358f22eddaba4754f97d40325e9f3e9151f6968d553123681be884a33f"
    }
  ],
  "count": 2,
  "no_demonstrated_defect": false,
  "resolution": "Action/outcome mismatch requires explicit diagnosis; no causal eligibility inferred.",
  "schema": "pact_v1010b_grasp_v1"
}
```
