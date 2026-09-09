# V10.10 wrist280 three-seed experiment (amended cohort)

Verdict: **DEVELOPMENT GATE MISSED**.

Prespecified development gate missed: {'history': 100, 'selection_scores': {'100': [18, 12, -27217, 1], '10': [0, 0, -423473, 0]}, 'successes': {'ACT': 8, 'PACT': 10}, 'passed': False}. Seeds 3104/3105 and final evaluation were not launched.

Updated: 2026-09-07T05:30:46.120811+00:00. All `authorizes_*` flags remain false; historical qualification outcomes are unchanged.

Completed collection: 280/280.
Converted episodes: 280/280.

- act_seed3103: 60000/60,000 logged optimizer updates.
- pact_seed3103: 60000/60,000 logged optimizer updates.

| Seed | ACT task success | PACT task success | PACT−ACT | ACT collision-free success | PACT collision-free success |
|---|---:|---:|---:|---:|---:|
| 3103 | 8/24 | 10/24 | +8.3 pp | 6/24 | 6/24 |

| Arm | Pooled success | Collision-free rollouts | Hazard-contact rollouts | Hazard frames | Mean frames/rollout |
|---|---:|---:|---:|---:|---:|
| ACT | 8/24 | 10/24 | 8/24 | 17545 | 731.04 |
| PACT | 10/24 | 14/24 | 4/24 | 9672 | 403.00 |

Paired successes: ACT-only 1; PACT-only 3.

| Arm | Target touch | Held | Lift ≥1 cm | Receptacle support |
|---|---:|---:|---:|---:|
| ACT | 17 | 12 | 12 | 8 |
| PACT | 18 | 10 | 11 | 10 |

| Arm | Slot | Contact rollouts | Contact frames | Stability-event rollouts | Stable rollouts |
|---|---|---:|---:|---:|---:|
| ACT | 01 | 5 | 8876 | 3 | 21 |
| ACT | 03 | 0 | 0 | 0 | 24 |
| ACT | 04 | 0 | 0 | 0 | 24 |
| ACT | 06 | 3 | 3559 | 2 | 22 |
| PACT | 01 | 4 | 1580 | 1 | 23 |
| PACT | 03 | 1 | 243 | 0 | 24 |
| PACT | 04 | 0 | 0 | 0 | 24 |
| PACT | 06 | 1 | 890 | 1 | 23 |

Slots 08/09 are absent; they are not zero-contact observations.


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
