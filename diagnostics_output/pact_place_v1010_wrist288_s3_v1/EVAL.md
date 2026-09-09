# V10.10 wrist280 three-seed experiment (amended cohort)

Verdict: **REDUCED FINAL EVALUATION COMPLETE**.

User-discussed 50-pair goals were **not met**: PACT achieved **23/50 (46%)**, below 26/50; ACT achieved **26/50 (52%)**, so PACT−ACT was **−3 successes / −6 percentage points**, below the requested +5 successes / +10 points. All 50 canonical pairs passed pairing audits. Twelve unreceipted outputs were retained separately and excluded uniformly; their same-instance retries are included among the 100 verified canonical rollouts.

Completed all 50 valid pairs / 100 rollouts across three seeds. Pooled successes: {'ACT': 26, 'PACT': 23}. The original development gate was overridden by explicit user instruction; the original 150-pair target was not assessed.

Updated: 2026-09-07T18:45:34.514844+00:00. All `authorizes_*` flags remain false; historical qualification outcomes are unchanged.

Completed collection: 280/280.
Converted episodes: 280/280.

- act_seed3103: 60000/60,000 logged optimizer updates.
- act_seed3104: 60000/60,000 logged optimizer updates.
- act_seed3105: 60000/60,000 logged optimizer updates.
- pact_seed3103: 60000/60,000 logged optimizer updates.
- pact_seed3104: 60000/60,000 logged optimizer updates.
- pact_seed3105: 60000/60,000 logged optimizer updates.

| Seed | ACT task success | PACT task success | PACT−ACT | ACT collision-free success | PACT collision-free success |
|---|---:|---:|---:|---:|---:|
| 3103 | 9/17 | 6/17 | -17.6 pp | 7/17 | 6/17 |
| 3104 | 9/17 | 6/17 | -17.6 pp | 7/17 | 4/17 |
| 3105 | 8/16 | 11/16 | +18.8 pp | 7/16 | 9/16 |

| Arm | Pooled success | Collision-free rollouts | Hazard-contact rollouts | Hazard frames | Mean frames/rollout |
|---|---:|---:|---:|---:|---:|
| ACT | 26/50 | 30/50 | 9/50 | 77962 | 1559.24 |
| PACT | 23/50 | 30/50 | 6/50 | 16553 | 331.06 |

Paired successes: ACT-only 10; PACT-only 7.

| Arm | Target touch | Held | Lift ≥1 cm | Receptacle support |
|---|---:|---:|---:|---:|
| ACT | 38 | 29 | 30 | 27 |
| PACT | 43 | 28 | 26 | 24 |

| Arm | Slot | Contact rollouts | Contact frames | Stability-event rollouts | Stable rollouts |
|---|---|---:|---:|---:|---:|
| ACT | 01 | 12 | 34218 | 9 | 41 |
| ACT | 03 | 0 | 0 | 0 | 50 |
| ACT | 04 | 0 | 0 | 0 | 50 |
| ACT | 06 | 10 | 43621 | 10 | 40 |
| PACT | 01 | 10 | 8229 | 4 | 46 |
| PACT | 03 | 0 | 0 | 0 | 50 |
| PACT | 04 | 0 | 0 | 0 | 50 |
| PACT | 06 | 9 | 39844 | 8 | 42 |

Slots 08/09 are absent; they are not zero-contact observations.

Three-seed mean task success: ACT 51.96%; PACT 46.45%; difference -5.51 pp.

Clutter was effectively invisible to the proximity skin in the prior resolvability audit: inbound vessel `max_w_perp_m = 0.000` in 7 of 8 variants. The 40 sensors cover link1–link6 only. A PACT advantage is not evidence that it senses the clutter.

Action alignment follows the [MolmoSpaces data format](https://allenai.github.io/molmospaces/data_format/) and the [original ACT recorder](https://github.com/tonyzhaozh/act/blob/main/record_sim_episodes.py). Its effect is measured by this experiment.

Artifacts: `config.json`, `manifests/`, `collection/ledger.jsonl`, `raw/`, `converted/`, `checkpoints/`, `evaluation/`, and `monitoring/hourly_run_check.jsonl`.

Dataset amendment: 280 selected strict-clean episodes; 240 training / 40 validation. All 24 cells are represented; collected totals are 12 in 21 cells and 8/9/11 in the three F2-left cells. This replaces the original balanced 288-episode / 48-validation design at the user’s request, before any model training. The 60,000-update budget, development gate, final manifests and evaluation rules are unchanged.

| Cell | Selected | Train | Validation |
|---|---:|---:|---:|
| F0_target_side_stagger / left / center | 12 | 11 | 1 |
| F0_target_side_stagger / left / neg5 | 12 | 10 | 2 |
| F0_target_side_stagger / left / pos5 | 12 | 10 | 2 |
| F0_target_side_stagger / right / center | 12 | 10 | 2 |
| F0_target_side_stagger / right / neg5 | 12 | 10 | 2 |
| F0_target_side_stagger / right / pos5 | 12 | 10 | 2 |
| F1_inner_panel_stagger / left / center | 12 | 10 | 2 |
| F1_inner_panel_stagger / left / neg5 | 12 | 10 | 2 |
| F1_inner_panel_stagger / left / pos5 | 12 | 10 | 2 |
| F1_inner_panel_stagger / right / center | 12 | 10 | 2 |
| F1_inner_panel_stagger / right / neg5 | 12 | 11 | 1 |
| F1_inner_panel_stagger / right / pos5 | 12 | 10 | 2 |
| F2_outer_panel_stagger / left / center | 8 | 7 | 1 |
| F2_outer_panel_stagger / left / neg5 | 9 | 8 | 1 |
| F2_outer_panel_stagger / left / pos5 | 11 | 10 | 1 |
| F2_outer_panel_stagger / right / center | 12 | 10 | 2 |
| F2_outer_panel_stagger / right / neg5 | 12 | 10 | 2 |
| F2_outer_panel_stagger / right / pos5 | 12 | 11 | 1 |
| F3_aperture_side_stagger / left / center | 12 | 10 | 2 |
| F3_aperture_side_stagger / left / neg5 | 12 | 10 | 2 |
| F3_aperture_side_stagger / left / pos5 | 12 | 11 | 1 |
| F3_aperture_side_stagger / right / center | 12 | 11 | 1 |
| F3_aperture_side_stagger / right / neg5 | 12 | 10 | 2 |
| F3_aperture_side_stagger / right / pos5 | 12 | 10 | 2 |

Bindings: `amendments/wrist280/effective_config.json`, `cohort.json`, `selected_ledger.jsonl`; original `config.json` and full collection ledger retained.

Operational recovery: evaluation resumes at 10 workers after the 14-worker pool exceeded resource limits. Further resource backoff waits until existing jobs reach the reduced target. Trained weights, data/splits, inference parameters, scientific instances and completed outcomes remain unchanged. The measured training benchmark is adopted rather than recreated from empty pools. See `amendments/resource_recovery_20260907/contract.json`.

Evaluation capacity amendment: user requested 12 or 14 workers. Target is 12; measured VRAM excludes 14 under the original 85% guard. Release clean file cache for completed experiment-owned HDF5 artifacts before evaluation startup, after audits, and at minute samples above 73% RAM. The 80% RAM guard and settled 12 -> 10 backoff remain active. See `amendments/eval_capacity_20260907/contract.json`.

User-directed follow-up: the original development gate remains missed (PACT 10/24, ACT 8/24). The user explicitly requested proceeding with 50 final pairs total across all three seeds, allocated 17/17/16. History 100 remains selected. The subset was frozen without final outcomes; all 24 cells have two pairs globally, with two balanced extra pairs. This reduced follow-up does not assess the original 150-pair target at its specified sample size. See `amendments/final50_20260907/contract.json`.

Final evaluation cache recovery: include converted `.hdf5` files and unused checkpoint files from fully completed training in scoped file-cache release. Inference checkpoints stay cached. All disk contents, memory thresholds, selected instances and completed results are preserved. See `amendments/final_cache_recovery_20260907/contract.json`.

Infrastructure retry: system thread/memory pressure prevented the old supervisor and monitor from spawning nvidia-smi. Twelve finished-looking rollouts lacked parent-observed exit receipts and were excluded uniformly. Their original files were quarantined and the same six frozen pairs rerun; no policy outcome determined retry selection. Eight additional rollouts were still unstarted. The intended cohort remains 50 pairs. GPU queries now use in-process NVML, and resource-query errors pause launches while preserving worker exit collection. See `amendments/receipt_recovery_20260907/quarantine.json`.
