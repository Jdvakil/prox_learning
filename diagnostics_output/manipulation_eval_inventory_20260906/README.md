# Retained ACT/PACT manipulation evaluations — 2026-09-06 audit

Scope: completed learned ACT-versus-PACT evaluations referenced by this worktree's reports and their linked artifact roots. This is not a claim to inventory every unregistered experiment ever run on the machine. No historical data or experiment was changed. Collection/expert qualification, smoke panels, qualitative selections, interrupted R1, and comparisons lacking ACT are not eligible for the headline ranking. The aligned dual-camera V10.11c pilot is excluded: all original rollouts finished, but one of 50 pairs failed its initial-image gate again on the bounded repeat. Its independently recounted descriptive task counts are ACT 14/50 and PACT 7/50; it is not a verified paired benchmark. See root `EVAL.md` and the pilot's `evaluation/blocked_raw_audit.json` for the actual failure and separately retained attempts.

Task success means the recorded unconditional task endpoint, not success conditional on contact, grasp attempt, or another outcome. Differences below are PACT minus ACT in percentage points. Each experiment has its own scene/data/checkpoints; ordering scores across experiments is descriptive, not a controlled comparison of recipes.

## Full pickup, transport and placement

Naming correction from the subsequent sampler audit: the workflow called `V5 / recovered-152` used **`PactPlaceCorridorV2Sampler`, with no added household clutter**, for both its recovered training corpus and the chunk-100 evaluation. It is not the distinct `PactPlaceCorridorV5Sampler` Objaverse-clutter environment. The recovery config's `scene.sampler_class` and the actual evaluation manifest establish this; the saved rollout commands and result manifest hashes bind that manifest. Chunk 25 reused the frozen chunk-100 manifest, and the chunk-1 manifest also declares V2. The earlier shorthand in this inventory must not be read as evidence of 47.5% success in a cluttered V5 scene. Scores are unchanged.

The place evaluator explicitly sets `task_type = "pick_and_place"` and uses `PactPlaceCorridorTask`, which inherits the support/release/stability criterion from `PickAndPlaceTask`. See `submodules/act/eval_pact_place_row.py:137` and `submodules/molmospaces/molmo_spaces/tasks/enclosure_reach.py:1560`.

| Evaluation | Training seeds | Rollouts/arm | ACT task | PACT task | Task gap | ACT collision-free success | PACT collision-free success |
|---|---:|---:|---:|---:|---:|---:|---:|
| V2 / recovered-152 (workflow called V5), chunk 100 | 1 | 40 | 13/40 (32.5%) | 19/40 (47.5%) | +15.0 pp | 13/40 (32.5%) | 16/40 (40.0%) |
| V10.10, chunk 100 | 1 | 40 | 11/40 (27.5%) | 14/40 (35.0%) | +7.5 pp | 7/40 (17.5%) | 10/40 (25.0%) |
| V2 / recovered-152, historical chunk 25 | 1 | 40 | 8/40 (20.0%) | 11/40 (27.5%) | +7.5 pp | 6/40 (15.0%) | 10/40 (25.0%) |
| V10.9, chunk 100 | 1 | 40 | 14/40 (35.0%) | 11/40 (27.5%) | -7.5 pp | 8/40 (20.0%) | 6/40 (15.0%) |
| Original V10.11c, chunk 100 | 3 | 150 | 11/150 (7.3%) | 12/150 (8.0%) | +0.7 pp | 8/150 (5.3%) | 10/150 (6.7%) |
| V2 / recovered-152, historical chunk 1 | 1 | 20 | 0/20 (0%) | 0/20 (0%) | 0.0 pp | 0/20 (0%) | 0/20 (0%) |

**Winner for both PACT task success and positive task-success gap among these completed full-place evaluations: V2/recovered-152, chunk 100.** None has PACT task success above 60%. V10.10 is the highest PACT task-success result among the added-clutter full-place comparisons in this table. The historical chunk-25 row is included only for this requested inventory; no new chunk-25 run was launched.

Safety is not interchangeable with successful completion. V2/recovered-152, chunk 100 has ACT 27/40 versus PACT 25/40 entirely collision-free episodes, despite PACT's higher collision-free-success score. Hazard-contact episodes are ACT 13/40 versus PACT 12/40. This does not establish an overall collision-avoidance advantage for PACT. V10.10 has ACT 14/40 versus PACT 16/40 entirely collision-free episodes, alongside its smaller positive task-success gap.

Recount sources: `recheck_01/place_chunk100.json`, `place_chunk25.json`, `place_chunk1.json`, `place_v109.json`, `place_v1010.json`, and `place_v1011c.json`. Original V5 report: `/root/pact_place_chunk100_eval_seed3101/EVAL.md`.

## Pickup/extraction corridor — a different task

The older collision-corridor configuration uses `PickTaskSamplerConfig` and `PactCollisionCorridorSampler(BigFumehoodPickSampler)`, not the place task. See `submodules/molmospaces/molmo_spaces/data_generation/config/object_manipulation_datagen_configs.py:2328` and `submodules/molmospaces/molmo_spaces/tasks/enclosure_reach.py:1301`. The scene includes an intrusion panel; it is not the six-clutter-object V10.11c placement benchmark.

| Evaluation / condition | Seeds | Instances × seeds per arm | ACT task | PACT task | Task gap | ACT collision-free success | PACT collision-free success |
|---|---:|---:|---:|---:|---:|---:|---:|
| 32-D front-end development screen | 1 | 40 × 1 | 19/40 (47.5%) | 29/40 (72.5%) | +25.0 pp | 19/40 (47.5%) | 29/40 (72.5%) |
| Same screen plus independent-seed replication | 2 | 40 × 2 | 43/80 (53.8%) | 51/80 (63.8%) | +10.0 pp | 43/80 (53.8%) | 50/80 (62.5%) |
| Contact-endpoint study | 3 | 100 × 3 | 169/300 (56.3%) | 183/300 (61.0%) | +4.7 pp | 159/300 (53.0%) | 171/300 (57.0%) |
| Blur study, unmodified RGB (sigma 0) | 3 | 25 × 3 | 37/75 (49.3%) | 45/75 (60.0%) | +10.7 pp | 36/75 (48.0%) | 44/75 (58.7%) |
| Blur study, sigma 0.5 | 3 | 25 × 3 | 35/75 (46.7%) | 44/75 (58.7%) | +12.0 pp | 33/75 (44.0%) | 42/75 (56.0%) |
| Blind-RGB study, sighted control | 3 | 25 × 3 | 37/75 (49.3%) | 44/75 (58.7%) | +9.3 pp | 36/75 (48.0%) | 43/75 (57.3%) |
| Original 3-D front-end R2 | 2 | 160 × 2 | 177/320 (55.3%) | 169/320 (52.8%) | -2.5 pp | 170/320 (53.1%) | 159/320 (49.7%) |
| Blur study, sigma 1 | 3 | 25 × 3 | 31/75 (41.3%) | 35/75 (46.7%) | +5.3 pp | 30/75 (40.0%) | 34/75 (45.3%) |
| Blur study, sigma 2 | 3 | 25 × 3 | 14/75 (18.7%) | 16/75 (21.3%) | +2.7 pp | 13/75 (17.3%) | 14/75 (18.7%) |
| Blind-RGB study, blinded | 3 | 25 × 3 | 1/75 (1.3%) | 1/75 (1.3%) | 0.0 pp | 0/75 (0%) | 1/75 (1.3%) |

The 72.5% screen is the highest whole-panel PACT task rate in the retained ACT/PACT comparisons above. It is exploratory, not replicated proof: seed 3102 scored PACT 22/40 (55%) versus ACT 24/40 (60%), reversing the task-success gap to -5 pp. Its collision-free-success gap is -7.5 pp. The valid-ablation and replication reports reuse the seed-3101 screen; they are not independent replications of its 29/40.

If individual seed-by-ablation cells are allowed, the largest positive task gap found is **+32 pp** in the blur study at sigma 0.5, seed 3101: PACT 18/25 (72%) versus ACT 10/25 (40%). This is a small, selected cell under inference-time vision degradation, not a larger replicated headline. The full three-seed sigma-0.5 result is 58.7% versus 46.7% (+12 pp), below the requested >60% PACT target. No inference-time blur was newly applied in this audit.

The contact-endpoint study is the most relevant existing three-seed result for the user's numerical criteria. Its per-seed task successes (ACT/PACT) are 55/62, 53/58 and 61/63 out of 100. It has ACT 67/300 versus PACT 42/300 hazard-contact episodes and ACT 233/300 versus PACT 257/300 completely collision-free episodes. Thus its aggregate point estimates meet >60% PACT task success, a positive task gap, and lower contact incidence. However, this is pickup/extraction rather than full placement; the published task-success uncertainty includes zero advantage, so the task gap is not established as a reliable improvement. The contact reduction is better supported than the success-rate difference. See `docs/PACT_CONTACT_ENDPOINT_DECISION.md`.

Recount sources under `recheck_01/`: `frontend_screen.json`, `seed_replication_new_seed.json`, `contact_endpoint.json`, `blur_sweep.json`, `blind_rgb.json`, `pact_vs_act_r2.json`.

## Exclusions and verification limits

- Geometry-generalization V3 has PACT 75/120 (62.5%) on C0, but its arms are PACT and PACT_PERMUTED, with no ACT. It cannot supply a PACT-minus-ACT gap. Earlier geometry gates did not run a qualifying learned-policy comparison.
- V10.9R is a ten-instance development diagnostic: both legacy and event decoders report ACT 4/10 versus PACT 1/10 task successes. Bulk trajectories were deleted; the retained row-level diagnostic fields and summary remain. It is not a stronger result or a new training/evaluation benchmark.
- V1–V10.11 environment-siting/collection/expert gates are not learned ACT-versus-PACT success rates. Smoke and qualitative subsets are not ranked as full evaluations.
- Recounted 2,960 ACT/PACT rollout records across the twelve audit files, using per-rollout scientific result fields and actual-exit receipts. Collision-free success was recomputed from the required contact totals; missing fields failed validation rather than becoming zeros. H5 endpoint checks were performed where `trajectory.h5` remains in the row directory (382 trajectories). The original V10.11c block also reran the complete contact/pose reconstruction on all 300 trajectories. No missing historical trajectory is claimed as freshly physics-verified.
- The first inventory-script pass incorrectly required later optional blur/blind metadata in older schemas. It recorded errors and was stopped; it is not verification evidence. The corrected pass records absent optional metadata as null and successfully reconstructs all included result counts. `recheck_01/` contains the corrected evidence.
- Historical artifacts stayed read-only. No training, environment or inference configuration was changed by this inventory. All authorization flags remain false.
