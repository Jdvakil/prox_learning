# Seed3103: main fine-tuned readout versus frozen PACT

Completed60000 new PACT updates and50 matched readout evaluations. The original frozen50 outcomes were verified and reused; one extra frozen replay checked the unchanged evaluator.

| Endpoint | Frozen32 | Fine-tuned128 |
|---|---:|---:|
| Task success /50 | 20 | 27 |
| Collision-free task success /50 | 16 | 24 |
| Exclusive pickup failures /50 | 16 | 8 |
| Hazard/clutter union frames | 59086 | 29000 |
| Frame avoidance % | 96.0212787448234 | 98.04720379785192 |

Paired wins/losses: {"task_success": {"candidate_only": 12, "control_only": 5}, "pickup_failure": {"candidate_only": 3, "control_only": 11}, "collision_free_task_success": {"candidate_only": 11, "control_only": 3}}

This uses the same50 exposed seed3103 scenes, so it is a regression comparison. It changes fine-tuning,32→128 readout and pooling together, as required to reproduce main. It does not establish an unfreezing-only effect or performance on other seeds.

The exact upstream modules are retained with byte hashes. The original dataset split, non-proximity inputs, action semantics, task geometry and history100 controller were preserved. Same-update policy and encoder hashes were validated before inference. Full raw endpoints and comparisons are in comparison.json.
