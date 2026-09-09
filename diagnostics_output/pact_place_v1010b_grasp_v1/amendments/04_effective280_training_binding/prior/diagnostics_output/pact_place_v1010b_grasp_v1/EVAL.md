# V10.10b bounded grasp-acquisition experiment

Status: **INCOMPLETE_INFRASTRUCTURE_OR_BUDGET**.

invalid completion: train_ACT_3103_uniform63000: worker exit1; see /root/prox_learning_pact_remediation/diagnostics_output/pact_place_v1010b_grasp_v1/checkpoints/act_seed3103_uniform63000/worker.log

Completed new rollouts: 60/1056 maximum main path. Final comparison: 0/600.
Completed continuation branches: 0/12; new optimizer updates: 0/36000.

| Stage | New rollouts completed | Main-path expected |
|---|---:|---:|
| A1 | 12 | 12 |
| A2 | 48 | 48 |
| B | 0 | 180 |
| C | 0 | 216 |
| D | 0 | 600 |

The unchanged update60000 ACT/PACT baselines remain runnable. This execution does not alter historical qualification or authorization flags.
Scene blocks and training seeds are separate identities. Curated replays are selected for mechanism inspection; crossed historical scenes are exposed diagnostics.
A prerequisite stop completes the bounded experiment without demonstrating a successful fix. Missing/infrastructure endpoints are not imputed or removed from scientific denominators.
All writes are confined to the new scripts/tests and this experiment directory. The pre-existing root EVAL.md is preserved.

Gate A: PASS. Failed checks: 

```json
{
  "by_checkpoint": {
    "3103": {
      "avoidance_percent": 98.85217108290405,
      "cfts": 6,
      "failure_stages": {
        "held_without_lift": 1,
        "lifted_without_placement": 5,
        "no_target_interaction": 1,
        "success": 8,
        "touched_without_hold": 9
      },
      "lifted_without_placement": 5,
      "n": 24,
      "physics_samples": 712824,
      "pickup": 9,
      "success": 8,
      "union_frames": 8182
    },
    "3104": {
      "avoidance_percent": 98.90674275838074,
      "cfts": 6,
      "failure_stages": {
        "held_without_lift": 1,
        "lifted_without_placement": 2,
        "no_target_interaction": 2,
        "success": 11,
        "touched_without_hold": 8
      },
      "lifted_without_placement": 2,
      "n": 24,
      "physics_samples": 712824,
      "pickup": 8,
      "success": 11,
      "union_frames": 7793
    },
    "3105": {
      "avoidance_percent": 98.04075059201149,
      "cfts": 12,
      "failure_stages": {
        "lifted_without_placement": 1,
        "no_target_interaction": 2,
        "success": 16,
        "touched_without_hold": 5
      },
      "lifted_without_placement": 1,
      "n": 24,
      "physics_samples": 712824,
      "pickup": 5,
      "success": 16,
      "union_frames": 13966
    }
  },
  "by_scene": {
    "018fa741fcfb4e08480be5079c183a596fd44c73e98f29354ad151439eee8f74": {
      "3103": {
        "pickup": false,
        "success": false
      },
      "3104": {
        "pickup": true,
        "success": false
      },
      "3105": {
        "pickup": false,
        "success": true
      }
    },
    "087e3e01ba4383b2527036497f7f7e65f2670b6ad679ec3a88ac88d48de74707": {
      "3103": {
        "pickup": true,
        "success": false
      },
      "3104": {
        "pickup": false,
        "success": true
      },
      "3105": {
        "pickup": false,
        "success": false
      }
    },
    "161209836332d0b752bbcc085d93a5a96a8f3f6af177c46d709d1d0781df9bfe": {
      "3103": {
        "pickup": true,
        "success": false
      },
      "3104": {
        "pickup": false,
        "success": false
      },
      "3105": {
        "pickup": false,
        "success": true
      }
    },
    "1e897063291fac78e63058ecc6f86f7154bc6caf30b6ddf258b50dff60601d8e": {
      "3103": {
        "pickup": false,
        "success": true
      },
      "3104": {
        "pickup": true,
        "success": false
      },
      "3105": {
        "pickup": false,
        "success": true
      }
    },
    "244723319760dcd8a6a9d3881e4400ed3c6a0b6baf03c48afaa99258769cc731": {
      "3103": {
        "pickup": true,
        "success": false
      },
      "3104": {
        "pickup": false,
        "success": false
      },
      "3105": {
        "pickup": true,
        "success": false
      }
    },
    "285217c4a0fcff2bb81c5dc3885ba9418c6ad82742b8bec50ebcd6af4df7a467": {
      "3103": {
        "pickup": false,
        "success": false
      },
      "3104": {
        "pickup": false,
        "success": true
      },
      "3105": {
        "pickup": false,
        "success": true
      }
    },
    "2ece7101c50d1c9b6dd015a86b010abd1b6a930710b660c7e2529a0e09d63afe": {
      "3103": {
        "pickup": false,
        "success": true
      },
      "3104": {
        "pickup": false,
        "success": true
      },
      "3105": {
        "pickup": false,
        "success": true
      }
    },
    "32da0d3b4fd0c1d858eb4de70bb90d2dcf2fb85b839fdb7e6e0309fdb3164f58": {
      "3103": {
        "pickup": true,
        "success": false
      },
      "3104": {
        "pickup": false,
        "success": true
      },
      "3105": {
        "pickup": false,
        "success": true
      }
    },
    "3da6f20aa5942a4dd803b50bc324e767c3f99b8d56a69b0b0aba6ab68ddf7655": {
      "3103": {
        "pickup": false,
        "success": false
      },
      "3104": {
        "pickup": false,
        "success": true
      },
      "3105": {
        "pickup": false,
        "success": true
      }
    },
    "3e0486b142e7abfe31573d8bc245f5c3d33ce018167ee222f38898b6dbe4d30a": {
      "3103": {
        "pickup": false,
        "success": true
      },
      "3104": {
        "pickup": false,
        "success": true
      },
      "3105": {
        "pickup": false,
        "success": true
      }
    },
    "457f3bce6855a0cd60ea9e428e34649df102adb18a4cf4ff4935e2c362685914": {
      "3103": {
        "pickup": true,
        "success": false
      },
      "3104": {
        "pickup": true,
        "success": false
      },
      "3105": {
        "pickup": true,
        "success": false
      }
    },
    "46146eec4fc59c9f16f1f6e587acd9abc5f898db5e5baa7cbfa9d406c05d0262": {
      "3103": {
        "pickup": false,
        "success": true
      },
      "3104": {
        "pickup": false,
        "success": true
      },
      "3105": {
        "pickup": false,
        "success": true
      }
    },
    "5d38f87da4547184133a1b6f83836477b50f301362dbc2a01eb68048e52fa3f7": {
      "3103": {
        "pickup": false,
        "success": false
      },
      "3104": {
        "pickup": false,
        "success": false
      },
      "3105": {
        "pickup": false,
        "success": false
      }
    },
    "6cddf5fef14baa4a4f294ae404e9c32464657c75c4b207fc8cd0bdf31c4a5299": {
      "3103": {
        "pickup": true,
        "success": false
      },
      "3104": {
        "pickup": false,
        "success": true
      },
      "3105": {
        "pickup": true,
        "success": false
      }
    },
    "86e5f130dcb86ffc23287faea454a5dd887f22b59d422d8bfad045a4c3703677": {
      "3103": {
        "pickup": true,
        "success": false
      },
      "3104": {
        "pickup": false,
        "success": false
      },
      "3105": {
        "pickup": false,
        "success": true
      }
    },
    "88b1363b23e900b04f1dab9a5dc549bb0c5606cd6e76f23ec1792d63353d5dc9": {
      "3103": {
        "pickup": false,
        "success": true
      },
      "3104": {
        "pickup": false,
        "success": true
      },
      "3105": {
        "pickup": false,
        "success": true
      }
    },
    "98c64a358f22eddaba4754f97d40325e9f3e9151f6968d553123681be884a33f": {
      "3103": {
        "pickup": false,
        "success": false
      },
      "3104": {
        "pickup": true,
        "success": false
      },
      "3105": {
        "pickup": false,
        "success": true
      }
    },
    "a05dc0de4ac2bc8144b012f2547e3bba70066cff1ea7ed16c213865de30ea5d8": {
      "3103": {
        "pickup": false,
        "success": true
      },
      "3104": {
        "pickup": true,
        "success": false
      },
      "3105": {
        "pickup": false,
        "success": true
      }
    },
    "a20477c29f9f5942fc0ab4c864f2e3f13b29c0862021c4d174c3d5c77563e728": {
      "3103": {
        "pickup": true,
        "success": false
      },
      "3104": {
        "pickup": true,
        "success": false
      },
      "3105": {
        "pickup": true,
        "success": false
      }
    },
    "bc1ca4f96ce731444743a05bc2a30b2167388c373ddd872354fa8e92a6dcd2aa": {
      "3103": {
        "pickup": false,
        "success": true
      },
      "3104": {
        "pickup": true,
        "success": false
      },
      "3105": {
        "pickup": false,
        "success": true
      }
    },
    "c3523463d0cbe3f0bdba635ff7f1d25458c357694d5747eef1aa5b4770300427": {
      "3103": {
        "pickup": false,
        "success": false
      },
      "3104": {
        "pickup": false,
        "success": true
      },
      "3105": {
        "pickup": false,
        "success": true
      }
    },
    "cd1fe7234437c3b4899188e972b51f32dd1d75047088974d0e923431e43de0d7": {
      "3103": {
        "pickup": false,
        "success": false
      },
      "3104": {
        "pickup": false,
        "success": false
      },
      "3105": {
        "pickup": false,
        "success": false
      }
    },
    "e208ca8dbbea2c3026541085eb086e5f9523e5a8cf9d8a444772faa5689fe645": {
      "3103": {
        "pickup": false,
        "success": true
      },
      "3104": {
        "pickup": false,
        "success": true
      },
      "3105": {
        "pickup": false,
        "success": true
      }
    },
    "f7576495b092b1fbbc5cd4fe69d36176d4496863efcd139bdddf8739c0178d11": {
      "3103": {
        "pickup": true,
        "success": false
      },
      "3104": {
        "pickup": true,
        "success": false
      },
      "3105": {
        "pickup": true,
        "success": false
      }
    }
  },
  "failure_stages": {
    "held_without_lift": 2,
    "lifted_without_placement": 8,
    "no_target_interaction": 5,
    "touched_without_hold": 22
  },
  "restricted_replay_scopes": [
    {
      "checks": {
        "held_from_one_gripper_contact": true,
        "neither_lifted_1cm": true,
        "one_declared_held_observation": true,
        "only_declared_difference": true,
        "original_never_held": true,
        "outside_cross24": true,
        "same_declared_pair": true,
        "sensor_reconstruction_exact": true
      },
      "id": "PACT3105_0f25bb_single_instantaneous_held_observation",
      "interpretation": "The unchanged instantaneous held heuristic changes for one recorded observation with no1cm lift. Both categories are retained. This pair is ineligible for original-category equivalence and excluded from the Stage A supporting-case count; new chronology remains descriptive.",
      "manifest_sha256": "7040591294654c437c780a2dfda89ddd6a066b8465e7d86f051b943e408af689",
      "new_chronology_eligible": true,
      "original_category_comparison_eligible": false,
      "use_for_stage_A_support": false
    }
  ],
  "supported_ids": [
    "98c64a358f22eddaba4754f97d40325e9f3e9151f6968d553123681be884a33f",
    "621d8bd12acc6e7256e2a671b85f49c0168e469e53ac02193dc8e0ba09be2d11"
  ]
}
```

smoke_replay_comparison.json:
```json
{
  "all_differences_resolved": true,
  "comparison_eligible": true,
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
      "comparison_eligible": true,
      "diagnosis_checks": {
        "both_first_queries_exact": true,
        "continuous_action_difference_starts_at_zero": true,
        "delta_exactly_reproduced": true,
        "different_rgb": true,
        "exclusive_stage_unchanged": true,
        "initial_pairing": true,
        "outcome_unchanged": true,
        "repeated_inference_exact": true,
        "same_proximity": true,
        "same_qpos": true
      },
      "difference_resolved": true,
      "failure_stage_exact": true,
      "initial_pairing_passed": true,
      "original_directory": "/root/prox_learning_pact_remediation/diagnostics_output/pact_place_v1010_wrist288_s3_v1/evaluation/final_3104_h100/final_3104_h100_ACT_98c64a358f22edda",
      "outcome_exact": true,
      "reconstruction_path": "/root/prox_learning_pact_remediation/diagnostics_output/pact_place_v1010b_grasp_v1/replay_query_reconstruction_v2/ACT_98c64a358f22eddaba4754f97d40325e9f3e9151f6968d553123681be884a33f.json",
      "reconstruction_sha256": "d33bae96193ecb094caaebab017035d26d0c5897083c0a83fcc3250b95e27ea1",
      "replay_directory": "/root/prox_learning_pact_remediation/diagnostics_output/pact_place_v1010b_grasp_v1/rollouts/A1/A1_ACT_3104_frozen60000_98c64a358f22edda/attempt_00",
      "resolution": "First-action divergence reproduced exactly from tolerated raster differences. New chronology is eligible only if all diagnosis checks pass; later closed-loop differences are not individually attributed because historical later images are unavailable.",
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
      "comparison_eligible": true,
      "diagnosis_checks": {
        "both_first_queries_exact": true,
        "continuous_action_difference_starts_at_zero": true,
        "delta_exactly_reproduced": true,
        "different_rgb": true,
        "exclusive_stage_unchanged": true,
        "initial_pairing": true,
        "outcome_unchanged": true,
        "repeated_inference_exact": true,
        "same_proximity": true,
        "same_qpos": true
      },
      "difference_resolved": true,
      "failure_stage_exact": true,
      "initial_pairing_passed": true,
      "original_directory": "/root/prox_learning_pact_remediation/diagnostics_output/pact_place_v1010_wrist288_s3_v1/evaluation/final_3104_h100/final_3104_h100_PACT_98c64a358f22edda",
      "outcome_exact": true,
      "reconstruction_path": "/root/prox_learning_pact_remediation/diagnostics_output/pact_place_v1010b_grasp_v1/replay_query_reconstruction_v2/PACT_98c64a358f22eddaba4754f97d40325e9f3e9151f6968d553123681be884a33f.json",
      "reconstruction_sha256": "c6bbc8518cb0d99dcf86b4fec66ecbac1f08e3b3d3226c1f3f2da2f73b3a84d7",
      "replay_directory": "/root/prox_learning_pact_remediation/diagnostics_output/pact_place_v1010b_grasp_v1/rollouts/A1/A1_PACT_3104_frozen60000_98c64a358f22edda/attempt_00",
      "resolution": "First-action divergence reproduced exactly from tolerated raster differences. New chronology is eligible only if all diagnosis checks pass; later closed-loop differences are not individually attributed because historical later images are unavailable.",
      "scene_id": "98c64a358f22eddaba4754f97d40325e9f3e9151f6968d553123681be884a33f"
    }
  ],
  "count": 2,
  "no_demonstrated_defect": true,
  "resolution": "Originals remain separate. Exact trace, checked raster onset, or an explicitly hash-bound restricted comparison scope; no full-trajectory identity or aggregation-causality claim.",
  "schema": "pact_v1010b_grasp_v1"
}
```

replay_comparison.json:
```json
{
  "all_differences_resolved": true,
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
      "comparison_eligible": true,
      "diagnosis_checks": {
        "both_first_queries_exact": true,
        "continuous_action_difference_starts_at_zero": true,
        "delta_exactly_reproduced": true,
        "different_rgb": true,
        "exclusive_stage_unchanged": true,
        "initial_pairing": true,
        "outcome_unchanged": true,
        "repeated_inference_exact": true,
        "same_proximity": true,
        "same_qpos": true
      },
      "difference_resolved": true,
      "failure_stage_exact": true,
      "initial_pairing_passed": true,
      "original_directory": "/root/prox_learning_pact_remediation/diagnostics_output/pact_place_v1010_wrist288_s3_v1/evaluation/final_3104_h100/final_3104_h100_ACT_98c64a358f22edda",
      "outcome_exact": true,
      "reconstruction_path": "/root/prox_learning_pact_remediation/diagnostics_output/pact_place_v1010b_grasp_v1/replay_query_reconstruction_v2/ACT_98c64a358f22eddaba4754f97d40325e9f3e9151f6968d553123681be884a33f.json",
      "reconstruction_sha256": "d33bae96193ecb094caaebab017035d26d0c5897083c0a83fcc3250b95e27ea1",
      "replay_directory": "/root/prox_learning_pact_remediation/diagnostics_output/pact_place_v1010b_grasp_v1/rollouts/A1/A1_ACT_3104_frozen60000_98c64a358f22edda/attempt_00",
      "resolution": "First-action divergence reproduced exactly from tolerated raster differences. New chronology is eligible only if all diagnosis checks pass; later closed-loop differences are not individually attributed because historical later images are unavailable.",
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
      "comparison_eligible": true,
      "diagnosis_checks": {
        "both_first_queries_exact": true,
        "continuous_action_difference_starts_at_zero": true,
        "delta_exactly_reproduced": true,
        "different_rgb": true,
        "exclusive_stage_unchanged": true,
        "initial_pairing": true,
        "outcome_unchanged": true,
        "repeated_inference_exact": true,
        "same_proximity": true,
        "same_qpos": true
      },
      "difference_resolved": true,
      "failure_stage_exact": true,
      "initial_pairing_passed": true,
      "original_directory": "/root/prox_learning_pact_remediation/diagnostics_output/pact_place_v1010_wrist288_s3_v1/evaluation/final_3104_h100/final_3104_h100_PACT_98c64a358f22edda",
      "outcome_exact": true,
      "reconstruction_path": "/root/prox_learning_pact_remediation/diagnostics_output/pact_place_v1010b_grasp_v1/replay_query_reconstruction_v2/PACT_98c64a358f22eddaba4754f97d40325e9f3e9151f6968d553123681be884a33f.json",
      "reconstruction_sha256": "c6bbc8518cb0d99dcf86b4fec66ecbac1f08e3b3d3226c1f3f2da2f73b3a84d7",
      "replay_directory": "/root/prox_learning_pact_remediation/diagnostics_output/pact_place_v1010b_grasp_v1/rollouts/A1/A1_PACT_3104_frozen60000_98c64a358f22edda/attempt_00",
      "resolution": "First-action divergence reproduced exactly from tolerated raster differences. New chronology is eligible only if all diagnosis checks pass; later closed-loop differences are not individually attributed because historical later images are unavailable.",
      "scene_id": "98c64a358f22eddaba4754f97d40325e9f3e9151f6968d553123681be884a33f"
    },
    {
      "actions": {
        "arm": {
          "exact": false,
          "first_different_step": 0,
          "max_abs": 0.019238561391830444
        },
        "gripper": {
          "exact": true,
          "first_different_step": null,
          "max_abs": 0.0
        },
        "model_output": {
          "exact": false,
          "first_different_step": 0,
          "max_abs": 7.3208770751953125
        }
      },
      "arm": "ACT",
      "comparison_eligible": true,
      "diagnosis_checks": {
        "both_first_queries_exact": true,
        "continuous_action_difference_starts_at_zero": true,
        "delta_exactly_reproduced": true,
        "different_rgb": true,
        "exclusive_stage_unchanged": true,
        "initial_pairing": true,
        "outcome_unchanged": true,
        "repeated_inference_exact": true,
        "same_proximity": true,
        "same_qpos": true
      },
      "difference_resolved": true,
      "failure_stage_exact": true,
      "initial_pairing_passed": true,
      "original_directory": "/root/prox_learning_pact_remediation/diagnostics_output/pact_place_v1010_wrist288_s3_v1/evaluation/final_3103_h100/final_3103_h100_ACT_621d8bd12acc6e72",
      "outcome_exact": true,
      "reconstruction_path": "/root/prox_learning_pact_remediation/diagnostics_output/pact_place_v1010b_grasp_v1/replay_query_reconstruction_v2/ACT_621d8bd12acc6e7256e2a671b85f49c0168e469e53ac02193dc8e0ba09be2d11.json",
      "reconstruction_sha256": "44ab66e04b4d4f777eccd49570e968483d33a2062af1d28018138a9e657910e9",
      "replay_directory": "/root/prox_learning_pact_remediation/diagnostics_output/pact_place_v1010b_grasp_v1/rollouts/A1/A1_ACT_3103_frozen60000_621d8bd12acc6e72/attempt_00",
      "resolution": "First-action divergence reproduced exactly from tolerated raster differences. New chronology is eligible only if all diagnosis checks pass; later closed-loop differences are not individually attributed because historical later images are unavailable.",
      "scene_id": "621d8bd12acc6e7256e2a671b85f49c0168e469e53ac02193dc8e0ba09be2d11"
    },
    {
      "actions": {
        "arm": {
          "exact": false,
          "first_different_step": 0,
          "max_abs": 0.026110529899597168
        },
        "gripper": {
          "exact": false,
          "first_different_step": 415,
          "max_abs": 255.0
        },
        "model_output": {
          "exact": false,
          "first_different_step": 0,
          "max_abs": 50.22749328613281
        }
      },
      "arm": "PACT",
      "comparison_eligible": true,
      "diagnosis_checks": {
        "both_first_queries_exact": true,
        "continuous_action_difference_starts_at_zero": true,
        "delta_exactly_reproduced": true,
        "different_rgb": true,
        "exclusive_stage_unchanged": true,
        "initial_pairing": true,
        "outcome_unchanged": true,
        "repeated_inference_exact": true,
        "same_proximity": true,
        "same_qpos": true
      },
      "difference_resolved": true,
      "failure_stage_exact": true,
      "initial_pairing_passed": true,
      "original_directory": "/root/prox_learning_pact_remediation/diagnostics_output/pact_place_v1010_wrist288_s3_v1/evaluation/final_3103_h100/final_3103_h100_PACT_621d8bd12acc6e72",
      "outcome_exact": true,
      "reconstruction_path": "/root/prox_learning_pact_remediation/diagnostics_output/pact_place_v1010b_grasp_v1/replay_query_reconstruction_v2/PACT_621d8bd12acc6e7256e2a671b85f49c0168e469e53ac02193dc8e0ba09be2d11.json",
      "reconstruction_sha256": "5dce018098706d2b9f3da36f81b7debe808581830f80101003d47ce62001de88",
      "replay_directory": "/root/prox_learning_pact_remediation/diagnostics_output/pact_place_v1010b_grasp_v1/rollouts/A1/A1_PACT_3103_frozen60000_621d8bd12acc6e72/attempt_00",
      "resolution": "First-action divergence reproduced exactly from tolerated raster differences. New chronology is eligible only if all diagnosis checks pass; later closed-loop differences are not individually attributed because historical later images are unavailable.",
      "scene_id": "621d8bd12acc6e7256e2a671b85f49c0168e469e53ac02193dc8e0ba09be2d11"
    },
    {
      "actions": {
        "arm": {
          "exact": false,
          "first_different_step": 0,
          "max_abs": 0.006852149963378906
        },
        "gripper": {
          "exact": true,
          "first_different_step": null,
          "max_abs": 0.0
        },
        "model_output": {
          "exact": false,
          "first_different_step": 0,
          "max_abs": 6.041084289550781
        }
      },
      "arm": "ACT",
      "comparison_eligible": true,
      "diagnosis_checks": {
        "both_first_queries_exact": true,
        "continuous_action_difference_starts_at_zero": true,
        "delta_exactly_reproduced": true,
        "different_rgb": true,
        "exclusive_stage_unchanged": true,
        "initial_pairing": true,
        "outcome_unchanged": true,
        "repeated_inference_exact": true,
        "same_proximity": true,
        "same_qpos": true
      },
      "difference_resolved": true,
      "failure_stage_exact": true,
      "initial_pairing_passed": true,
      "original_directory": "/root/prox_learning_pact_remediation/diagnostics_output/pact_place_v1010_wrist288_s3_v1/evaluation/final_3105_h100/final_3105_h100_ACT_c0ff4fd9273e6de8",
      "outcome_exact": true,
      "reconstruction_path": "/root/prox_learning_pact_remediation/diagnostics_output/pact_place_v1010b_grasp_v1/replay_query_reconstruction_v2/ACT_c0ff4fd9273e6de8a16efdb2c68d4827f22a2bf32772633c610c90a732ec735b.json",
      "reconstruction_sha256": "53c1db45080005a730e2bab52602ece43c5ec4f6f081d80cad83b0ba0692671d",
      "replay_directory": "/root/prox_learning_pact_remediation/diagnostics_output/pact_place_v1010b_grasp_v1/rollouts/A1/A1_ACT_3105_frozen60000_c0ff4fd9273e6de8/attempt_00",
      "resolution": "First-action divergence reproduced exactly from tolerated raster differences. New chronology is eligible only if all diagnosis checks pass; later closed-loop differences are not individually attributed because historical later images are unavailable.",
      "scene_id": "c0ff4fd9273e6de8a16efdb2c68d4827f22a2bf32772633c610c90a732ec735b"
    },
    {
      "actions": {
        "arm": {
          "exact": false,
          "first_different_step": 0,
          "max_abs": 0.01414477825164795
        },
        "gripper": {
          "exact": true,
          "first_different_step": null,
          "max_abs": 0.0
        },
        "model_output": {
          "exact": false,
          "first_different_step": 0,
          "max_abs": 5.6348419189453125
        }
      },
      "arm": "PACT",
      "comparison_eligible": true,
      "diagnosis_checks": {
        "both_first_queries_exact": true,
        "continuous_action_difference_starts_at_zero": true,
        "delta_exactly_reproduced": true,
        "different_rgb": true,
        "exclusive_stage_unchanged": true,
        "initial_pairing": true,
        "outcome_unchanged": true,
        "repeated_inference_exact": true,
        "same_proximity": true,
        "same_qpos": true
      },
      "difference_resolved": true,
      "failure_stage_exact": true,
      "initial_pairing_passed": true,
      "original_directory": "/root/prox_learning_pact_remediation/diagnostics_output/pact_place_v1010_wrist288_s3_v1/evaluation/final_3105_h100/final_3105_h100_PACT_c0ff4fd9273e6de8",
      "outcome_exact": true,
      "reconstruction_path": "/root/prox_learning_pact_remediation/diagnostics_output/pact_place_v1010b_grasp_v1/replay_query_reconstruction_v2/PACT_c0ff4fd9273e6de8a16efdb2c68d4827f22a2bf32772633c610c90a732ec735b.json",
      "reconstruction_sha256": "886f00a8526319edce8e07ee601371236e39c61eddedad7738335125708c4901",
      "replay_directory": "/root/prox_learning_pact_remediation/diagnostics_output/pact_place_v1010b_grasp_v1/rollouts/A1/A1_PACT_3105_frozen60000_c0ff4fd9273e6de8/attempt_00",
      "resolution": "First-action divergence reproduced exactly from tolerated raster differences. New chronology is eligible only if all diagnosis checks pass; later closed-loop differences are not individually attributed because historical later images are unavailable.",
      "scene_id": "c0ff4fd9273e6de8a16efdb2c68d4827f22a2bf32772633c610c90a732ec735b"
    },
    {
      "actions": {
        "arm": {
          "exact": false,
          "first_different_step": 0,
          "max_abs": 0.010895438492298126
        },
        "gripper": {
          "exact": true,
          "first_different_step": null,
          "max_abs": 0.0
        },
        "model_output": {
          "exact": false,
          "first_different_step": 0,
          "max_abs": 5.31976318359375
        }
      },
      "arm": "ACT",
      "comparison_eligible": true,
      "diagnosis_checks": {
        "both_first_queries_exact": true,
        "continuous_action_difference_starts_at_zero": true,
        "delta_exactly_reproduced": true,
        "different_rgb": true,
        "exclusive_stage_unchanged": true,
        "initial_pairing": true,
        "outcome_unchanged": true,
        "repeated_inference_exact": true,
        "same_proximity": true,
        "same_qpos": true
      },
      "difference_resolved": true,
      "failure_stage_exact": true,
      "initial_pairing_passed": true,
      "original_directory": "/root/prox_learning_pact_remediation/diagnostics_output/pact_place_v1010_wrist288_s3_v1/evaluation/final_3103_h100/final_3103_h100_ACT_fecc06a7e3f15eec",
      "outcome_exact": true,
      "reconstruction_path": "/root/prox_learning_pact_remediation/diagnostics_output/pact_place_v1010b_grasp_v1/replay_query_reconstruction_v2/ACT_fecc06a7e3f15eecc7f68a0d5344cfa1806f17d55033aa9198bfcae6a8994729.json",
      "reconstruction_sha256": "58b09fb90010b7d7f8fcb1160ebf9125a30e1fc7633af695c042a1ed2dc29910",
      "replay_directory": "/root/prox_learning_pact_remediation/diagnostics_output/pact_place_v1010b_grasp_v1/rollouts/A1/A1_ACT_3103_frozen60000_fecc06a7e3f15eec/attempt_00",
      "resolution": "First-action divergence reproduced exactly from tolerated raster differences. New chronology is eligible only if all diagnosis checks pass; later closed-loop differences are not individually attributed because historical later images are unavailable.",
      "scene_id": "fecc06a7e3f15eecc7f68a0d5344cfa1806f17d55033aa9198bfcae6a8994729"
    },
    {
      "actions": {
        "arm": {
          "exact": false,
          "first_different_step": 0,
          "max_abs": 0.044881463050842285
        },
        "gripper": {
          "exact": false,
          "first_different_step": 411,
          "max_abs": 255.0
        },
        "model_output": {
          "exact": false,
          "first_different_step": 0,
          "max_abs": 45.082855224609375
        }
      },
      "arm": "PACT",
      "comparison_eligible": true,
      "diagnosis_checks": {
        "both_first_queries_exact": true,
        "continuous_action_difference_starts_at_zero": true,
        "delta_exactly_reproduced": true,
        "different_rgb": true,
        "exclusive_stage_unchanged": true,
        "initial_pairing": true,
        "outcome_unchanged": true,
        "repeated_inference_exact": true,
        "same_proximity": true,
        "same_qpos": true
      },
      "difference_resolved": true,
      "failure_stage_exact": true,
      "initial_pairing_passed": true,
      "original_directory": "/root/prox_learning_pact_remediation/diagnostics_output/pact_place_v1010_wrist288_s3_v1/evaluation/final_3103_h100/final_3103_h100_PACT_fecc06a7e3f15eec",
      "outcome_exact": true,
      "reconstruction_path": "/root/prox_learning_pact_remediation/diagnostics_output/pact_place_v1010b_grasp_v1/replay_query_reconstruction_v2/PACT_fecc06a7e3f15eecc7f68a0d5344cfa1806f17d55033aa9198bfcae6a8994729.json",
      "reconstruction_sha256": "60c679f894e5e274658c67065929520ce7aa3a0d6c1dddbbdfea8c46aba8e6bd",
      "replay_directory": "/root/prox_learning_pact_remediation/diagnostics_output/pact_place_v1010b_grasp_v1/rollouts/A1/A1_PACT_3103_frozen60000_fecc06a7e3f15eec/attempt_00",
      "resolution": "First-action divergence reproduced exactly from tolerated raster differences. New chronology is eligible only if all diagnosis checks pass; later closed-loop differences are not individually attributed because historical later images are unavailable.",
      "scene_id": "fecc06a7e3f15eecc7f68a0d5344cfa1806f17d55033aa9198bfcae6a8994729"
    },
    {
      "actions": {
        "arm": {
          "exact": false,
          "first_different_step": 0,
          "max_abs": 0.014948487281799316
        },
        "gripper": {
          "exact": false,
          "first_different_step": 311,
          "max_abs": 255.0
        },
        "model_output": {
          "exact": false,
          "first_different_step": 0,
          "max_abs": 11.8551025390625
        }
      },
      "arm": "ACT",
      "comparison_eligible": true,
      "diagnosis_checks": {
        "both_first_queries_exact": true,
        "continuous_action_difference_starts_at_zero": true,
        "delta_exactly_reproduced": true,
        "different_rgb": true,
        "exclusive_stage_unchanged": true,
        "initial_pairing": true,
        "outcome_unchanged": true,
        "repeated_inference_exact": true,
        "same_proximity": true,
        "same_qpos": true
      },
      "difference_resolved": true,
      "failure_stage_exact": true,
      "initial_pairing_passed": true,
      "original_directory": "/root/prox_learning_pact_remediation/diagnostics_output/pact_place_v1010_wrist288_s3_v1/evaluation/final_3104_h100/final_3104_h100_ACT_93cb60c3de7743f9",
      "outcome_exact": true,
      "reconstruction_path": "/root/prox_learning_pact_remediation/diagnostics_output/pact_place_v1010b_grasp_v1/replay_query_reconstruction_v2/ACT_93cb60c3de7743f97c670dd14629678476f6c54c459c1db3ac1e92565699c41d.json",
      "reconstruction_sha256": "7a0f17fecf886d18a67d053b6436e9f20ec8dbe563d049590bf6f0c7dc0c3623",
      "replay_directory": "/root/prox_learning_pact_remediation/diagnostics_output/pact_place_v1010b_grasp_v1/rollouts/A1/A1_ACT_3104_frozen60000_93cb60c3de7743f9/attempt_00",
      "resolution": "First-action divergence reproduced exactly from tolerated raster differences. New chronology is eligible only if all diagnosis checks pass; later closed-loop differences are not individually attributed because historical later images are unavailable.",
      "scene_id": "93cb60c3de7743f97c670dd14629678476f6c54c459c1db3ac1e92565699c41d"
    },
    {
      "actions": {
        "arm": {
          "exact": false,
          "first_different_step": 0,
          "max_abs": 0.038857460021972656
        },
        "gripper": {
          "exact": false,
          "first_different_step": 132,
          "max_abs": 255.0
        },
        "model_output": {
          "exact": false,
          "first_different_step": 0,
          "max_abs": 42.44580078125
        }
      },
      "arm": "PACT",
      "comparison_eligible": true,
      "diagnosis_checks": {
        "both_first_queries_exact": true,
        "continuous_action_difference_starts_at_zero": true,
        "delta_exactly_reproduced": true,
        "different_rgb": true,
        "exclusive_stage_unchanged": true,
        "initial_pairing": true,
        "outcome_unchanged": true,
        "repeated_inference_exact": true,
        "same_proximity": true,
        "same_qpos": true
      },
      "difference_resolved": true,
      "failure_stage_exact": true,
      "initial_pairing_passed": true,
      "original_directory": "/root/prox_learning_pact_remediation/diagnostics_output/pact_place_v1010_wrist288_s3_v1/evaluation/final_3104_h100/final_3104_h100_PACT_93cb60c3de7743f9",
      "outcome_exact": true,
      "reconstruction_path": "/root/prox_learning_pact_remediation/diagnostics_output/pact_place_v1010b_grasp_v1/replay_query_reconstruction_v2/PACT_93cb60c3de7743f97c670dd14629678476f6c54c459c1db3ac1e92565699c41d.json",
      "reconstruction_sha256": "3fafe7fe77bdc72bf5ee5e303650950dfcd39be13d59093765df5c561e436672",
      "replay_directory": "/root/prox_learning_pact_remediation/diagnostics_output/pact_place_v1010b_grasp_v1/rollouts/A1/A1_PACT_3104_frozen60000_93cb60c3de7743f9/attempt_00",
      "resolution": "First-action divergence reproduced exactly from tolerated raster differences. New chronology is eligible only if all diagnosis checks pass; later closed-loop differences are not individually attributed because historical later images are unavailable.",
      "scene_id": "93cb60c3de7743f97c670dd14629678476f6c54c459c1db3ac1e92565699c41d"
    },
    {
      "actions": {
        "arm": {
          "exact": false,
          "first_different_step": 0,
          "max_abs": 0.015516623854637146
        },
        "gripper": {
          "exact": true,
          "first_different_step": null,
          "max_abs": 0.0
        },
        "model_output": {
          "exact": false,
          "first_different_step": 0,
          "max_abs": 0.8741378784179688
        }
      },
      "arm": "ACT",
      "comparison_eligible": true,
      "diagnosis_checks": {
        "both_first_queries_exact": true,
        "continuous_action_difference_starts_at_zero": true,
        "delta_exactly_reproduced": true,
        "different_rgb": true,
        "exclusive_stage_unchanged": true,
        "initial_pairing": true,
        "outcome_unchanged": true,
        "repeated_inference_exact": true,
        "same_proximity": true,
        "same_qpos": true
      },
      "difference_resolved": true,
      "failure_stage_exact": true,
      "initial_pairing_passed": true,
      "original_directory": "/root/prox_learning_pact_remediation/diagnostics_output/pact_place_v1010_wrist288_s3_v1/evaluation/final_3105_h100/final_3105_h100_ACT_0f25bb0b1a72d962",
      "outcome_exact": true,
      "reconstruction_path": "/root/prox_learning_pact_remediation/diagnostics_output/pact_place_v1010b_grasp_v1/replay_query_reconstruction_v2/ACT_0f25bb0b1a72d962671d75fd44b7551c352e8528bcae9acf9ba8588d483c18f3.json",
      "reconstruction_sha256": "bf828beb256ed3f01239d069c877061ef4b526af65a85d295453dc1a4da3c514",
      "replay_directory": "/root/prox_learning_pact_remediation/diagnostics_output/pact_place_v1010b_grasp_v1/rollouts/A1/A1_ACT_3105_frozen60000_0f25bb0b1a72d962/attempt_00",
      "resolution": "First-action divergence reproduced exactly from tolerated raster differences. New chronology is eligible only if all diagnosis checks pass; later closed-loop differences are not individually attributed because historical later images are unavailable.",
      "scene_id": "0f25bb0b1a72d962671d75fd44b7551c352e8528bcae9acf9ba8588d483c18f3"
    },
    {
      "actions": {
        "arm": {
          "exact": false,
          "first_different_step": 0,
          "max_abs": 0.02924954891204834
        },
        "gripper": {
          "exact": false,
          "first_different_step": 152,
          "max_abs": 255.0
        },
        "model_output": {
          "exact": false,
          "first_different_step": 0,
          "max_abs": 17.294906616210938
        }
      },
      "arm": "PACT",
      "comparison_eligible": false,
      "diagnosis_checks": {
        "both_first_queries_exact": true,
        "continuous_action_difference_starts_at_zero": true,
        "delta_exactly_reproduced": true,
        "different_rgb": true,
        "exclusive_stage_unchanged": false,
        "initial_pairing": true,
        "outcome_unchanged": true,
        "repeated_inference_exact": true,
        "same_proximity": true,
        "same_qpos": true
      },
      "difference_resolved": true,
      "failure_stage_exact": false,
      "initial_pairing_passed": true,
      "original_directory": "/root/prox_learning_pact_remediation/diagnostics_output/pact_place_v1010_wrist288_s3_v1/evaluation/final_3105_h100/final_3105_h100_PACT_0f25bb0b1a72d962",
      "outcome_exact": true,
      "reconstruction_path": "/root/prox_learning_pact_remediation/diagnostics_output/pact_place_v1010b_grasp_v1/replay_query_reconstruction_v2/PACT_0f25bb0b1a72d962671d75fd44b7551c352e8528bcae9acf9ba8588d483c18f3.json",
      "reconstruction_sha256": "ff52fe131f600561c2925833aacb8c92dddc967e70c9aeb4f3cff706f86aff68",
      "replay_directory": "/root/prox_learning_pact_remediation/diagnostics_output/pact_place_v1010b_grasp_v1/rollouts/A1/A1_PACT_3105_frozen60000_0f25bb0b1a72d962/attempt_00",
      "resolution": "First-action divergence reproduced exactly from tolerated raster differences. New chronology is eligible only if all diagnosis checks pass; later closed-loop differences are not individually attributed because historical later images are unavailable.",
      "scene_id": "0f25bb0b1a72d962671d75fd44b7551c352e8528bcae9acf9ba8588d483c18f3",
      "scope_resolution": {
        "checks": {
          "held_from_one_gripper_contact": true,
          "neither_lifted_1cm": true,
          "one_declared_held_observation": true,
          "only_declared_difference": true,
          "original_never_held": true,
          "outside_cross24": true,
          "same_declared_pair": true,
          "sensor_reconstruction_exact": true
        },
        "id": "PACT3105_0f25bb_single_instantaneous_held_observation",
        "interpretation": "The unchanged instantaneous held heuristic changes for one recorded observation with no1cm lift. Both categories are retained. This pair is ineligible for original-category equivalence and excluded from the Stage A supporting-case count; new chronology remains descriptive.",
        "manifest_sha256": "7040591294654c437c780a2dfda89ddd6a066b8465e7d86f051b943e408af689",
        "new_chronology_eligible": true,
        "original_category_comparison_eligible": false,
        "use_for_stage_A_support": false
      }
    }
  ],
  "count": 12,
  "no_demonstrated_defect": true,
  "resolution": "Originals remain separate. Exact trace, checked raster onset, or an explicitly hash-bound restricted comparison scope; no full-trajectory identity or aggregation-causality claim.",
  "schema": "pact_v1010b_grasp_v1"
}
```
