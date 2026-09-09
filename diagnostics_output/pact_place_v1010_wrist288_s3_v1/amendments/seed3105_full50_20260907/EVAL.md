# Seed 3105 — full 50-pair evaluation

Verdict: **SEED 3105 FULL 50-PAIR EVALUATION COMPLETE**.

Completed 50 valid paired instances / 100 rollouts: 16 existing pairs adopted and 34 additional pairs evaluated.

Updated: 2026-09-07T20:57:24.423495+00:00. ACT and PACT use their existing 60,000-update checkpoints, history 100, and the original frozen final-3105 manifest.

| Instances | ACT success | PACT success | PACT−ACT | ACT collision-free success | PACT collision-free success |
|---|---:|---:|---:|---:|---:|
| Previously completed 16 | 8/16 (50.0%) | 11/16 (68.8%) | +18.8 pp | 7/16 | 9/16 |
| Additional 34 | 15/34 (44.1%) | 24/34 (70.6%) | +26.5 pp | 8/34 | 16/34 |
| Full 50 | 23/50 (46.0%) | 35/50 (70.0%) | +24.0 pp | 15/50 | 25/50 |

Descriptive 50-pair goals: PACT ≥26 successes: **True**; PACT at least five successes ahead of ACT: **True**.

| Arm | Collision-free rollouts | Hazard-contact rollouts | Hazard frames | Mean hazard frames | Touch | Hold | Lift ≥1 cm | Receptacle support |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ACT | 26/50 | 10/50 | 25737 | 514.74 | 37 | 24 | 26 | 24 |
| PACT | 31/50 | 3/50 | 12219 | 244.38 | 47 | 37 | 38 | 35 |

| Arm | Slot | Contact rollouts | Contact frames | Stability-event rollouts | Stable rollouts |
|---|---|---:|---:|---:|---:|
| ACT | 01 | 11/50 | 22945 | 7/50 | 43/50 |
| ACT | 03 | 1/50 | 18 | 0/50 | 50/50 |
| ACT | 04 | 0/50 | 0 | 0/50 | 50/50 |
| ACT | 06 | 12/50 | 17494 | 10/50 | 40/50 |
| PACT | 01 | 12/50 | 11195 | 4/50 | 46/50 |
| PACT | 03 | 0/50 | 0 | 0/50 | 50/50 |
| PACT | 04 | 0/50 | 0 | 0/50 | 50/50 |
| PACT | 06 | 9/50 | 3158 | 3/50 | 47/50 |

Paired successes: ACT-only 3; PACT-only 15.

Failure stages: {"ACT": {"lifted_without_placement": 2, "no_target_interaction": 13, "success": 23, "supported_without_final_success": 1, "touched_without_hold": 11}, "PACT": {"held_without_lift": 1, "lifted_without_placement": 3, "no_target_interaction": 3, "success": 35, "touched_without_hold": 8}}.

The user selected seed 3105 for expansion after reviewing its initial 16-pair result. This is a follow-up on that selected seed; the earlier three-seed report remains unchanged. The original development gate remains missed.

All 50 pairs use the original frozen instances and full initial-state/RGB pairing audits. Prior completed pairs are adopted without rerunning them. Report all failures; no outcome-based replacement or checkpoint selection is permitted.

Slots 08/09 are absent. The 40 proximity sensors cover link1–link6; the prior audit found the inbound vessel outside sensor visibility in 7/8 variants. A PACT advantage does not establish clutter sensing. All `authorizes_*` flags remain false.

Artifacts: `contract.json`, `analysis.json`, `adoption.json`; canonical full-stage ledger and pairing audits under `../../evaluation/`.
