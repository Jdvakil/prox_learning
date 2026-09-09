# Full three-seed evaluation: 50 pairs per seed

Completed and verified at 2026-09-08T02:14:56.092705+00:00: 150 physical instances, 300 ACT/PACT rollouts, all 150 initial-state pairing audits passed. All models used 60,000-update checkpoints and history 100.

| Seed | ACT task success | PACT task success | PACT−ACT | ACT collision-free success | PACT collision-free success |
|---|---:|---:|---:|---:|---:|
| 3103 | 19/50 (38.0%) | 20/50 (40.0%) | +2.0 pp | 16/50 (32.0%) | 16/50 (32.0%) |
| 3104 | 22/50 (44.0%) | 19/50 (38.0%) | -6.0 pp | 15/50 (30.0%) | 10/50 (20.0%) |
| 3105 | 23/50 (46.0%) | 35/50 (70.0%) | +24.0 pp | 15/50 (30.0%) | 25/50 (50.0%) |
| Pooled | 64/150 (42.7%) | 74/150 (49.3%) | +6.7 pp | 46/150 (30.7%) | 51/150 (34.0%) |

The equal-size three-seed mean equals the pooled rate. The original target was not met: PACT needed at least 76/150 successes and at least 15 more successes than ACT. The observed counts are 74/150 and 10 more, respectively. These are descriptive outcomes; no confidence intervals or significance tests are included.

| Arm | Collision-free rollouts | Hazard-contact rollouts | Hazard-free rollouts | Hazard frames | Mean hazard frames/rollout |
|---|---:|---:|---:|---:|---:|
| ACT | 77/150 | 33/150 | 117/150 | 232733 | 1551.55 |
| PACT | 90/150 | 13/150 | 137/150 | 44673 | 297.82 |

Paired successes: ACT-only 24; PACT-only 34.

| Arm | Target touch | Contact-only hold | Lift ≥1 cm | Tray support |
|---|---:|---:|---:|---:|
| ACT | 108/150 | 71/150 | 74/150 | 66/150 |
| PACT | 133/150 | 86/150 | 87/150 | 75/150 |

| Arm | Slot | Contact rollouts | Contact frames | Stability-event rollouts | Stable rollouts |
|---|---|---:|---:|---:|---:|
| ACT | 01 | 38/150 | 98202 | 25/150 | 125/150 |
| ACT | 03 | 1/150 | 18 | 0/150 | 150/150 |
| ACT | 04 | 0/150 | 0 | 0/150 | 150/150 |
| ACT | 06 | 32/150 | 124888 | 27/150 | 123/150 |
| PACT | 01 | 40/150 | 39394 | 14/150 | 136/150 |
| PACT | 03 | 0/150 | 0 | 0/150 | 150/150 |
| PACT | 04 | 0/150 | 0 | 0/150 | 150/150 |
| PACT | 06 | 26/150 | 67792 | 18/150 | 132/150 |

Slots 08/09 are absent. The 40 proximity sensors cover link1–link6; the historical sensor audit does not establish clutter visibility.

The original reduced n=50 report and all three extension reports remain unchanged. The development gate remains missed and all authorization/qualification flags remain false. Extensions reused the original frozen full manifests, preserving every valid failure.

Source hashes, all rows and pairing-audit bindings are in `three_seed_full150.json`. Seed 3103 completed its final rollout by 01:54:59 UTC and finalized reporting by 01:56:40 UTC; the final monitor audit is clean and both supervisor and monitor have exited.
