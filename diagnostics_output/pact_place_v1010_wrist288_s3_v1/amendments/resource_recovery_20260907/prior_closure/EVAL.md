# V10.10 wrist280 three-seed experiment (amended cohort)

Verdict: **INCOMPLETE**.

Paused: three consecutive resource pressure samples

Updated: 2026-09-07T01:50:07.757838+00:00. All `authorizes_*` flags remain false; historical qualification outcomes are unchanged.

Completed collection: 280/280.
Converted episodes: 280/280.

- act_seed3103: 60000/60,000 logged optimizer updates.
- pact_seed3103: 60000/60,000 logged optimizer updates.

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
