# V10.11c ACT/PACT training and paired evaluation

Completed six chunk-100 models and **300/300 raw-verified scientific rollouts**: ACT and PACT, seeds 3103/3104/3105, 50 paired instances per seed. All 150 pairs have identical non-RGB initial observations and selected environment seeds, with wrist RGB checked against the disclosed tolerance below. The smoke completed 8/8 rollouts on four separate instances, initially failed strict RGB equality, and was reconciled under that rule without rerunning any rollout; eight additional frozen smoke rows were reserved but not selected.

The owner narrowed scope on 2026-09-05 to chunk 100 only and omitted chunk 25, gripper-status analysis, Wilson intervals and McNemar tests. Results below are descriptive. They do not establish statistical superiority.

## Corpus and training

The read-only collection ledger contains **482 attempts, 99 accepted strict-clean episodes**. All 24 cells have 4–5 episodes. Row directories resolve as `attempt_id[:16]`; no closeout was required. The split is **75 train / 24 validation**, taking the lowest SHA256(split seed:attempt_id) per cell for validation. Split seed: 2026090401. Converted T=176–546, sum 37,871; 1,514,840 windows were encoded, with independent source-preservation checks.

The reused embedding verifier retains V10.9 schema labels and a V10.8 `note_on_window_count` paragraph. That legacy paragraph does not describe this corpus; the numerical episode/timestep/window fields and source reconstruction are authoritative.

Both arms used 2,000 epochs, batch 8, learning rate 1e-5, KL 10, chunk 100, hidden 512, feed-forward 3,200, 7 encoder/7 decoder layers, 8 heads, wrist ResNet-18, and state/action dimensions 9/8. Training episode horizon was 635. The command comparison allowed only the checkpoint directory and five PACT flags. The frozen encoder hash is `6fd2dd037e3236b5b6bf7fce8cb2709ead0cf52adcbbe9cbad1061efc2fe3206`. All six models passed strict checkpoint reload, offline inference and split/dataset provenance checks; PACT proximity consumption was checked using real versus zeroed embeddings.

| Seed | Arm | Epochs | Best epoch | Best validation loss | Training minutes |
|---|---|---:|---:|---:|---:|
| 3103 | ACT | 2000 | 1995 | 0.135764 | 51.0 |
| 3103 | PACT | 2000 | 1974 | 0.134507 | 54.0 |
| 3104 | ACT | 2000 | 1886 | 0.156815 | 53.4 |
| 3104 | PACT | 2000 | 1865 | 0.159752 | 56.1 |
| 3105 | ACT | 2000 | 1756 | 0.119959 | 54.0 |
| 3105 | PACT | 2000 | 1720 | 0.121472 | 56.9 |

## Paired outcomes

The overall comparison includes **all three independently trained seeds**: 150 rollouts per arm. For every episode-level rate, the pooled numerator divided by 150 is exactly the arithmetic mean of the three seed-level rates, because each seed contributes 50 instances. Frame and contact-entry columns are summed counts across the indicated seed(s), not percentages or per-rollout averages. Each arm is paired on the same instances within each seed; the three seed blocks use disjoint held-out instances. Thus variation between seed rows reflects both training randomness and held-out instance variation.

Collision-free success means final task success with zero hazard-bar, other-environment, clutter or mounted-fixture contact entries. Task success was independently read from the final trajectory `success` field. Contact episodes have at least one physics sample with that contact. A frame below is an audited physics sample (2 ms plus episode boundaries), not a contact-pair entry or a rendered video frame.

| Seed | Arm | Collision-free success | Task success | Hazard-contact episodes | Hazard frames | Hazard entries | Clutter-contact episodes |
|---|---|---:|---:|---:|---:|---:|---:|
| 3103 | ACT | 3/50 (6.0%) | 4/50 (8.0%) | 19/50 (38.0%) | 78119 | 83005 | 13/50 (26.0%) |
| 3103 | PACT | 4/50 (8.0%) | 4/50 (8.0%) | 16/50 (32.0%) | 69743 | 78783 | 12/50 (24.0%) |
| 3104 | ACT | 4/50 (8.0%) | 5/50 (10.0%) | 16/50 (32.0%) | 74220 | 77423 | 20/50 (40.0%) |
| 3104 | PACT | 4/50 (8.0%) | 6/50 (12.0%) | 21/50 (42.0%) | 114345 | 123899 | 17/50 (34.0%) |
| 3105 | ACT | 1/50 (2.0%) | 2/50 (4.0%) | 15/50 (30.0%) | 74855 | 92551 | 21/50 (42.0%) |
| 3105 | PACT | 2/50 (4.0%) | 2/50 (4.0%) | 18/50 (36.0%) | 99571 | 107220 | 10/50 (20.0%) |
| pooled | ACT | 8/150 (5.3%) | 11/150 (7.3%) | 50/150 (33.3%) | 227194 | 252979 | 54/150 (36.0%) |
| pooled | PACT | 10/150 (6.7%) | 12/150 (8.0%) | 55/150 (36.7%) | 283659 | 309902 | 39/150 (26.0%) |

## Per-object contact and stability

Slot 01 is the tall route cylinder; 08 the tall near-target cylinder; 09 the tall near-target box. An object is stable only if it never exceeds 2 cm translation or 25° rotation from its settled baseline at any retained control sample. Object contact attribution was reconstructed from retained geom/body pair identities and counts; stability was reconstructed from retained poses and rotations.

| Seed | Slot | Arm | Contact episodes | Contact frames | Contact entries | Stable episodes |
|---|---|---|---:|---:|---:|---:|
| 3103 | 01 | ACT | 1/50 (2.0%) | 1229 | 1270 | 46/50 (92.0%) |
| 3103 | 01 | PACT | 2/50 (4.0%) | 2636 | 6985 | 46/50 (92.0%) |
| 3103 | 08 | ACT | 2/50 (4.0%) | 187 | 254 | 49/50 (98.0%) |
| 3103 | 08 | PACT | 3/50 (6.0%) | 606 | 635 | 49/50 (98.0%) |
| 3103 | 09 | ACT | 2/50 (4.0%) | 136 | 136 | 50/50 (100.0%) |
| 3103 | 09 | PACT | 2/50 (4.0%) | 1065 | 1120 | 48/50 (96.0%) |
| 3104 | 01 | ACT | 11/50 (22.0%) | 8681 | 17548 | 38/50 (76.0%) |
| 3104 | 01 | PACT | 7/50 (14.0%) | 6232 | 11867 | 43/50 (86.0%) |
| 3104 | 08 | ACT | 1/50 (2.0%) | 774 | 774 | 50/50 (100.0%) |
| 3104 | 08 | PACT | 0/50 (0.0%) | 0 | 0 | 49/50 (98.0%) |
| 3104 | 09 | ACT | 2/50 (4.0%) | 822 | 1010 | 50/50 (100.0%) |
| 3104 | 09 | PACT | 1/50 (2.0%) | 18 | 18 | 49/50 (98.0%) |
| 3105 | 01 | ACT | 7/50 (14.0%) | 4433 | 10081 | 42/50 (84.0%) |
| 3105 | 01 | PACT | 4/50 (8.0%) | 14644 | 32542 | 45/50 (90.0%) |
| 3105 | 08 | ACT | 7/50 (14.0%) | 28736 | 87063 | 45/50 (90.0%) |
| 3105 | 08 | PACT | 2/50 (4.0%) | 305 | 868 | 49/50 (98.0%) |
| 3105 | 09 | ACT | 4/50 (8.0%) | 1948 | 2577 | 47/50 (94.0%) |
| 3105 | 09 | PACT | 0/50 (0.0%) | 0 | 0 | 50/50 (100.0%) |
| pooled | 01 | ACT | 19/150 (12.7%) | 14343 | 28899 | 126/150 (84.0%) |
| pooled | 01 | PACT | 13/150 (8.7%) | 23512 | 51394 | 134/150 (89.3%) |
| pooled | 08 | ACT | 10/150 (6.7%) | 29697 | 88091 | 144/150 (96.0%) |
| pooled | 08 | PACT | 5/150 (3.3%) | 911 | 1503 | 147/150 (98.0%) |
| pooled | 09 | ACT | 8/150 (5.3%) | 2906 | 3723 | 147/150 (98.0%) |
| pooled | 09 | PACT | 3/150 (2.0%) | 1083 | 1138 | 147/150 (98.0%) |

## Interpretation and provenance

The original smoke stage exited 1 after 8/8 successful rollouts because three wrist RGB pairs failed exact byte equality: `AssertionError: initial observation mismatch: observation/wrist_camera`. Those pairs differed in 15, 24 and 48 of 658,944 uint8 channel values, each by exactly one intensity level; all 449 non-RGB observation datasets matched exactly. Before scientific evaluation, the pairing rule was amended to allow at most 1/255 intensity difference in at most 0.1% of RGB channel values while requiring exact equality for every other field. The cause of the sparse RGB differences was not conclusively established. Original failures, raw images and `smoke_reconciliation.json` are retained. No smoke performance outcome influenced this change.

The outer coordinator was lost in a Codex crash during training; the six-model trainer survived. A detached coordinator reconnected without restarting a model. Its completion record explicitly marks the lost coordinator exit code as unavailable and relies on all six actual model exit codes and independent verification.

Clutter here is effectively invisible to the proximity skin: inbound vessel `max_w_perp_m = 0.000` in **7/8** variants of `diagnostics_output/pact_place_v9_w1_resolvability_full/resolvability.json`; the 40 sensors cover **link1–link6 only**. Any PACT–ACT difference is **not evidence that PACT senses the clutter**. This historical resolvability check is not a new visibility measurement on the held-out evaluation instances.

Evaluation used `molmo_spaces.tasks.enclosure_reach.PactPlaceCorridorV1011C33PctTallerPrimitiveSampler` from `experiment/pact-vs-act-remediation-v2`, with its implementation hash checked against the source freeze. The original chunk-100 temporal ensemble, 127.5 gripper threshold, 900-step horizon, disabled action noise and `end_on_success=false` were preserved. All five native thread pools were capped at 1. Initial and retry evaluation seeds were checked against 28,222 historical/collection-stream seeds and against each other.

Scientific evaluation took 8.93 hours with worker-count history [4, 10]. Full H5 trajectories, actions, initial observations and raw telemetry are retained under `diagnostics_output/pact_place_v1011c_eval/`.

Periodic training snapshots and optimizer resume bundles were omitted for the initial disk constraint. Verified best checkpoints are retained; each redundant final checkpoint was pruned after verification. Source-preservation checks rehashed 694 files. Every authorization flag remains false.

Orphan `multiprocessing.spawn` workers belonging to this worktree cleaned: 0.

Machine-readable report: `diagnostics_output/pact_place_v1011c_eval/analysis.json`. Completion was reconciled against `full_ledger.jsonl` and every raw result, not a progress summary.

The owner requested increased concurrency after observing resource headroom. The original scheduler was paused while all active rollouts finished normally. Their actual Linux exit codes and raw artifacts were retained in `scheduler_resize_01/`; only the drained scheduler was terminated (stage `11_eval_full` actual exit -9). The resumed stage retained the original frozen schedule and completed the remaining instances without repeating or discarding a scientific rollout. The interrupted stage is not reported as successful.
