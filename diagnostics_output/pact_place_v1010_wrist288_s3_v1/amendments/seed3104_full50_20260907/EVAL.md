# Seed 3104 — full 50-pair evaluation

Verdict: **SEED 3104 FULL 50-PAIR EVALUATION COMPLETE**.

Completed 50 valid paired instances / 100 rollouts: 17 existing pairs adopted and 33 additional pairs evaluated.

Updated: 2026-09-07T23:28:23.898933+00:00. ACT and PACT use their existing 60,000-update checkpoints, history 100, and the original frozen final-3104 manifest.

| Instances | ACT success | PACT success | PACT−ACT | ACT collision-free success | PACT collision-free success |
|---|---:|---:|---:|---:|---:|
| Previously completed 17 | 9/17 (52.9%) | 6/17 (35.3%) | -17.6 pp | 7/17 | 4/17 |
| Additional 33 | 13/33 (39.4%) | 13/33 (39.4%) | +0.0 pp | 8/33 | 6/33 |
| Full 50 | 22/50 (44.0%) | 19/50 (38.0%) | -6.0 pp | 15/50 | 10/50 |

Descriptive 50-pair goals: PACT ≥26 successes: **False**; PACT at least five successes ahead of ACT: **False**.

| Arm | Collision-free rollouts | Hazard-contact rollouts | Hazard frames | Mean hazard frames | Touch | Hold | Lift ≥1 cm | Receptacle support |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ACT | 26/50 | 12/50 | 140601 | 2812.02 | 35 | 23 | 23 | 22 |
| PACT | 29/50 | 5/50 | 14271 | 285.42 | 43 | 24 | 23 | 20 |

| Arm | Slot | Contact rollouts | Contact frames | Stability-event rollouts | Stable rollouts |
|---|---|---:|---:|---:|---:|
| ACT | 01 | 10/50 | 24993 | 6/50 | 44/50 |
| ACT | 03 | 0/50 | 0 | 0/50 | 50/50 |
| ACT | 04 | 0/50 | 0 | 0/50 | 50/50 |
| ACT | 06 | 7/50 | 31929 | 5/50 | 45/50 |
| PACT | 01 | 13/50 | 5953 | 3/50 | 47/50 |
| PACT | 03 | 0/50 | 0 | 0/50 | 50/50 |
| PACT | 04 | 0/50 | 0 | 0/50 | 50/50 |
| PACT | 06 | 7/50 | 34913 | 6/50 | 44/50 |

Paired successes: ACT-only 12; PACT-only 9.

Failure stages: {"ACT": {"lifted_without_placement": 1, "no_target_interaction": 15, "success": 22, "touched_without_hold": 12}, "PACT": {"held_without_lift": 2, "lifted_without_placement": 3, "no_target_interaction": 7, "success": 19, "supported_without_final_success": 1, "touched_without_hold": 18}}.

The user selected seed 3104 for expansion after reviewing its initial 17-pair result. This is a follow-up on that selected seed; the earlier three-seed report remains unchanged. The original development gate remains missed.

The user extended the deadline by three hours: September 8 03:00 UTC, with new launches stopping at 02:00 UTC to reserve one hour for reporting. The original experiment contracts remain preserved.

All 50 pairs use the original frozen instances and full initial-state/RGB pairing audits. Prior completed pairs are adopted without rerunning them. Report all failures; no outcome-based replacement or checkpoint selection is permitted.

Slots 08/09 are absent. The 40 proximity sensors cover link1–link6; the prior audit found the inbound vessel outside sensor visibility in 7/8 variants. A PACT advantage does not establish clutter sensing. All `authorizes_*` flags remain false.

Artifacts: `contract.json`, `analysis.json`, `adoption.json`; canonical full-stage ledger and pairing audits under `../../evaluation/`.
