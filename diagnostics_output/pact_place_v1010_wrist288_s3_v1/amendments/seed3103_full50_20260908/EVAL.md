# Seed 3103 — full 50-pair evaluation

Verdict: **SEED 3103 FULL 50-PAIR EVALUATION COMPLETE**.

Completed 50 valid paired instances / 100 rollouts: 17 existing pairs adopted and 33 additional pairs evaluated.

Updated: 2026-09-08T01:55:35.023520+00:00. ACT and PACT use their existing 60,000-update checkpoints, history 100, and the original frozen final-3103 manifest.

| Instances | ACT success | PACT success | PACT−ACT | ACT collision-free success | PACT collision-free success |
|---|---:|---:|---:|---:|---:|
| Previously completed 17 | 9/17 (52.9%) | 6/17 (35.3%) | -17.6 pp | 7/17 | 6/17 |
| Additional 33 | 10/33 (30.3%) | 14/33 (42.4%) | +12.1 pp | 9/33 | 10/33 |
| Full 50 | 19/50 (38.0%) | 20/50 (40.0%) | +2.0 pp | 16/50 | 16/50 |

Descriptive 50-pair goals: PACT ≥26 successes: **False**; PACT at least five successes ahead of ACT: **False**.

| Arm | Collision-free rollouts | Hazard-contact rollouts | Hazard frames | Mean hazard frames | Touch | Hold | Lift ≥1 cm | Receptacle support |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ACT | 25/50 | 11/50 | 66395 | 1327.90 | 36 | 24 | 25 | 20 |
| PACT | 30/50 | 5/50 | 18183 | 363.66 | 43 | 25 | 26 | 20 |

| Arm | Slot | Contact rollouts | Contact frames | Stability-event rollouts | Stable rollouts |
|---|---|---:|---:|---:|---:|
| ACT | 01 | 17/50 | 50264 | 12/50 | 38/50 |
| ACT | 03 | 0/50 | 0 | 0/50 | 50/50 |
| ACT | 04 | 0/50 | 0 | 0/50 | 50/50 |
| ACT | 06 | 13/50 | 75465 | 12/50 | 38/50 |
| PACT | 01 | 15/50 | 22246 | 7/50 | 43/50 |
| PACT | 03 | 0/50 | 0 | 0/50 | 50/50 |
| PACT | 04 | 0/50 | 0 | 0/50 | 50/50 |
| PACT | 06 | 10/50 | 29721 | 9/50 | 41/50 |

Paired successes: ACT-only 9; PACT-only 10.

Failure stages: {"ACT": {"held_without_lift": 1, "lifted_without_placement": 5, "no_target_interaction": 12, "success": 19, "supported_without_final_success": 1, "touched_without_hold": 12}, "PACT": {"held_without_lift": 1, "lifted_without_placement": 6, "no_target_interaction": 7, "success": 20, "touched_without_hold": 16}}.

The user selected seed 3103 for expansion after reviewing its initial 17-pair result. This is a follow-up on that selected seed; the earlier three-seed report remains unchanged. The original development gate remains missed.

The user extended the deadline by three hours: September 8 03:00 UTC, with new launches stopping at 02:00 UTC for the final reporting hour. Healthy jobs launched earlier finish normally; the admission estimate must fit all starts before 02:00 and completion plus validation before 03:00. The original experiment contracts remain preserved.

All 50 pairs use the original frozen instances and full initial-state/RGB pairing audits. Prior completed pairs are adopted without rerunning them. Report all failures; no outcome-based replacement or checkpoint selection is permitted.

Slots 08/09 are absent. The 40 proximity sensors cover link1–link6; the prior audit found the inbound vessel outside sensor visibility in 7/8 variants. A PACT advantage does not establish clutter sensing. All `authorizes_*` flags remain false.

Artifacts: `contract.json`, `analysis.json`, `adoption.json`; canonical full-stage ledger and pairing audits under `../../evaluation/`.
