# Results — every closed-loop eval on disk

Snapshot 2026-09-22. Built from `eval_output/*/episodes.jsonl` (recounted per episode) and
`reports/eval_summaries/*.json`. Refresh the ledger with `python scripts/exp_tracker.py`
(writes `reports/experiments_evals.csv`). Facts only; statistics and caveats are in `PAPER.md` §3.

## Environment codes

| code | environment | dataset / TASK | cams | horizon |
|---|---|---|---|---|
| v1 | fume-hood obstacle pick, first version (June 2026) | `obstacle_pact` (deleted) | – | 200 |
| v2 | fume-hood obstacle pick, camera-hidden bar study (July 2026) | `obstacle_pact_v2` / `obstacle_prox_v2` (deleted 2026-08-24) | – | 200 |
| v5 | hallway place corridor | `pact_place_corridor_v5` (shells `*_v1_hallway.sh`) | wrist | 800 |
| v107_spaced | spaced bench (batman, 210 eps) | `pact_place_corridor_v107_spaced` | table + wrist | 1050 |
| v1010 | four-object bench | `pact_place_corridor_v1010` | table + wrist | 1050 |
| v1011c | clutter geometry, 99 eps | `pact_place_corridor_v10_11c_100` | exo + wrist | 1050 |
| v1011d | randomised clutter | `pact_pick_n_place_v2_v1011d` | exo + wrist | 1050 |
| v6 | two-object V10.10 | `pact_pick_n_place_v2_v6` | exo + wrist | 1050 |
| v12 | kitchen | `pact_pick_n_place_v2_v12` | exo + wrist | 1050 |

`v1` / `v2` are the fume-hood task-directory suffixes; the full map for v5 onward is README §0.1.

## Metrics

- **Success** — task done at the final step (placement for v5+; pick for v1 / v2).
- **Strict** — success and zero counted collisions in the episode.
- **Bar hit** — any robot–hazard-bar contact. Hazard avoidance = 1 − bar hit.
- **Coll-free** — zero counted contacts (collision avoidance rate). v1 / v2 count any
  robot–environment contact (no bar-only split), so the bar-hit column is "–".
- **Ever** — placement judge true at any step.

Counts are `k/n (%)`. All rows are chunk 50 unless the chunk column says otherwise.

## All results

| code | set | arm (train seed) | chunk | n | success | strict | bar hit | coll-free | ever | source |
|---|---|---|---|---|---|---|---|---|---|---|
| v1 | A, 2000 epochs | ACT | 50 | 20 | 7 (35%) | 4 (20%) | – | 14 (70%) | – | wandb `rq7cwlqy` (no JSON) |
| v1 | B, 2000 epochs | ACT | 100 | 20 | 8 (40%) | 3 (15%) | – | 8 (40%) | – | wandb `c7v2nugl` (no JSON) |
| v1 | C, 5000 epochs | ACT | 100 | 20 | 6 (30%) | 5 (25%) | – | 16 (80%) | – | wandb `san7y9rp` (no JSON) |
| v2 | O-INV, no bar | ACT | 100 | 50 | 11 (22%) | 4 (8%) | – | 20 (40%) | – | `vanilla_v2_free.json` |
| v2 | O-INV, visible bar | ACT | 100 | 50 | 14 (28%) | 8 (16%) | – | 18 (36%) | – | `vanilla_v2_visible.json` |
| v2 | O-INV, camera-hidden bar | ACT | 100 | 50 | 18 (36%) | 7 (14%) | – | 17 (34%) | – | `vanilla_v2_invisible.json` |
| v2 | O-INV, no bar | PACT-raw (old design) | 100 | 50 | 9 (18%) | 4 (8%) | – | 21 (42%) | – | `pact_raw_v2_free.json` |
| v2 | O-INV, visible bar | PACT-raw (old design) | 100 | 50 | 8 (16%) | 5 (10%) | – | 25 (50%) | – | `pact_raw_v2_visible.json` |
| v2 | O-INV, camera-hidden bar | PACT-raw (old design) | 100 | 50 | 15 (30%) | 10 (20%) | – | 30 (60%) | – | `pact_raw_v2_invisible.json` |
| v2 | O-INV, no bar | PACT-trunk | 100 | 50 | 17 (34%) | 8 (16%) | – | 18 (36%) | – | `pact_trunk_v2_free.json` |
| v2 | O-INV, visible bar | PACT-trunk | 100 | 50 | 16 (32%) | 9 (18%) | – | 21 (42%) | – | `pact_trunk_v2_visible.json` |
| v2 | O-INV, camera-hidden bar | PACT-trunk | 100 | 50 | 17 (34%) | 5 (10%) | – | 14 (28%) | – | `pact_trunk_v2_invisible.json` |
| v2 | train-blur σ=2, no bar | ACT | 100 | 25 | 1 (4%) | 0 (0%) | – | 13 (52%) | – | `*_blurC2_v2_free.json` |
| v2 | train-blur σ=2, visible | ACT | 100 | 25 | 2 (8%) | 2 (8%) | – | 12 (48%) | – | `*_blurC2_v2_visible.json` |
| v2 | train-blur σ=2, hidden | ACT | 100 | 25 | 0 (0%) | 0 (0%) | – | 13 (52%) | – | `*_blurC2_v2_invisible.json` |
| v2 | train-blur σ=4, no bar | ACT | 100 | 25 | 10 (40%) | 3 (12%) | – | 4 (16%) | – | `*_blurC4_v2_free.json` |
| v2 | train-blur σ=4, visible | ACT | 100 | 25 | 6 (24%) | 2 (8%) | – | 4 (16%) | – | `*_blurC4_v2_visible.json` |
| v2 | train-blur σ=4, hidden | ACT | 100 | 25 | 10 (40%) | 1 (4%) | – | 3 (12%) | – | `*_blurC4_v2_invisible.json` |
| v2 | train-blur σ=8, no bar | ACT | 100 | 25 | 6 (24%) | 4 (16%) | – | 18 (72%) | – | `*_blurC8_v2_free.json` |
| v2 | train-blur σ=8, visible | ACT | 100 | 25 | 6 (24%) | 4 (16%) | – | 8 (32%) | – | `*_blurC8_v2_visible.json` |
| v2 | train-blur σ=8, hidden | ACT | 100 | 25 | 6 (24%) | 2 (8%) | – | 8 (32%) | – | `*_blurC8_v2_invisible.json` |
| v5 | H-A, legacy evaluator, random house | ACT (0) | 50 | 50 | 14 (28%) | 13 (26%) | 17 (34%) | 33 (66%) | 14 | `place_corridor_vanilla_s0_n50` |
| v5 | H-A | PACT-raw (0) | 50 | 50 | 21 (42%) | 17 (34%) | 18 (36%) | 32 (64%) | 21 | `place_corridor_raw_s0_n50` |
| v5 | H-A | PACT-readout (0) | 50 | 50 | 20 (40%) | 20 (40%) | 6 (12%) | 44 (88%) | 20 | `place_corridor_readout_s0_n50_fast` |
| v5 | H-A, early n=20 | ACT (0) | 50 | 20 | 3 (15%) | 3 (15%) | 6 (30%) | 14 (70%) | 3 | `place_corridor_vanilla_s0` |
| v5 | H-A, early n=20 | PACT-raw (0) | 50 | 20 | 7 (35%) | 6 (30%) | 4 (20%) | 16 (80%) | 7 | `place_corridor_raw_s0` |
| v5 | H-A, early n=20 | PACT-readout (0) | 50 | 20 | 5 (25%) | 4 (20%) | 6 (30%) | 14 (70%) | 5 | `place_corridor_readout_s0_n20` |
| v5 | H-A, stopped at 36/50 | PACT-readout (0) | 50 | 36 | 13 (36%) | 13 (36%) | 12 (33%) | 24 (67%) | 13 | `place_corridor_readout_s0_n50` |
| v5 | H-A, `n8_test` random house | PACT-readout (0) | 50 | 20 | 3 (15%) | 2 (10%) | 13 (65%) | 7 (35%) | 3 | `place_corridor_readout_s0_n8_test` |
| v5 | H-B, `eval_act.py`, house 1 | ACT (1) | 50 | 50 | 15 (30%) | 12 (24%) | 20 (40%) | 30 (60%) | 15 | `pact_place_corridor_v5_ACT_s1_*` |
| v5 | H-B | PACT-raw (0) | 50 | 50 | 21 (42%) | 19 (38%) | 13 (26%) | 37 (74%) | 21 | `pact_place_corridor_v5_PACT_RAW_s0_*` |
| v5 | H-B | PACT-readout (0) | 50 | 50 | 21 (42%) | 19 (38%) | 9 (18%) | 41 (82%) | 24 | `pact_place_corridor_v5_PACT_READOUT_s0_*` |
| v5 | H-B, sensor keep 0.75 | PACT-readout (0) | 50 | 50 | 23 (46%) | 22 (44%) | 14 (28%) | 36 (72%) | 23 | `…_PACT_READOUT_s0_*_keep75` |
| v5 | H-B, sensor keep 0.5 (partial) | PACT-readout (0) | 50 | 6 | 2 (33%) | 2 (33%) | 2 (33%) | 4 (67%) | 2 | `…_PACT_READOUT_s0_*_keep50` |
| v5 | H-B | ACT (0) | 50 | 2 | 1 (50%) | 1 (50%) | 0 (0%) | 2 (100%) | 1 | `pact_place_corridor_v5_ACT_s0_*` |
| v5 | H-C, Aug ckpt on `eval_act.py` | PACT-readout (0) | 50 | 50 | 18 (36%) | 18 (36%) | 7 (14%) | 43 (86%) | 19 | `simple_hallway_n50` |
| v107_spaced | T-107 (eval cam fov 45, train 58) | ACT (0) | 50 | 50 | 13 (26%) | 7 (14%) | 8 (16%) | 19 (38%) | 14 | `pact_place_corridor_v107_spaced_ACT_s0_*` |
| v107_spaced | T-107 | PACT-raw (0) | 50 | 50 | 0 (0%) | 0 (0%) | 9 (18%) | 19 (38%) | 0 | `pact_place_corridor_v107_spaced_PACT_RAW_s0_*` |
| v107_spaced | T-107 | PACT-readout (0) | 50 | 50 | 2 (4%) | 2 (4%) | 6 (12%) | 20 (40%) | 3 | `pact_place_corridor_v107_spaced_PACT_READOUT_s0_*` |
| v1011d | T-1011d, clutter 0.25 | ACT (0) | 50 | 50 | 13 (26%) | 9 (18%) | 18 (36%) | 15 (30%) | 13 | `pact_pick_n_place_v2_v1011d_ACT_s0_*` |
| v1011d | T-1011d, clutter 0.25 | PACT-raw (0) | 50 | 50 | 12 (24%) | 11 (22%) | 3 (6%) | 26 (52%) | 12 | `pact_pick_n_place_v2_v1011d_PACT_RAW_s0_*` |
| v1011d | T-1011d, clutter 0.25 | PACT-readout (0) | 50 | 50 | 13 (26%) | 11 (22%) | 10 (20%) | 22 (44%) | 14 | `pact_pick_n_place_v2_v1011d_PACT_READOUT_s0_*` |
| v1011d | old Sep 3 ckpt, clutter 1.0 | PACT-raw (0) | 50 | 50 | 7 (14%) | 4 (8%) | 10 (20%) | 20 (40%) | 10 | `simple_v1011d_smoke_video` |
| v1011d | old Sep 3 ckpt, clutter 0.25 | PACT-raw (0) | 50 | 50 | 14 (28%) | 11 (22%) | 3 (6%) | 22 (44%) | 14 | `simple_v1011d_easy025_n50` |
| v1011d | old Sep 3 ckpt, wrist only (partial) | PACT-raw (0) | 50 | 4 | 0 (0%) | 0 (0%) | 3 (75%) | 1 (25%) | 0 | `simple_v1011d_wrist_only_n50` |
| v1011d | rerun noise, episodes 0–23, run a | PACT-readout (0) | 50 | 24 | 4 (17%) | 4 (17%) | 6 (25%) | 11 (46%) | 4 | `_ab_lazy24_a` |
| v1011d | rerun noise, episodes 0–23, run b | PACT-readout (0) | 50 | 24 | 2 (8%) | 2 (8%) | 6 (25%) | 11 (46%) | 2 | `_ab_lazy24_b` |
| v1011d → v1010 | OOD: v1011d ckpt on v1010 sampler, horizon 800 | PACT-raw (0) | 50 | 48 | 0 (0%) | 0 (0%) | 3 (6%) | 32 (67%) | 0 | `pact_pick_n_place_v2_v1011d_raw_s0_n48` |
| v1011d → v1010 | OOD, horizon 1050 | PACT-raw (0) | 50 | 48 | 0 (0%) | 0 (0%) | 0 (0%) | 37 (77%) | 0 | `…_raw_s0_n48_horizon1050` |
| v1011d → v1010 | OOD, ray-based proximity | PACT-raw (0) | 50 | 50 | 0 (0%) | 0 (0%) | 0 (0%) | 28 (56%) | 0 | `v1011d_speedcheck_n50` |
| v1010 | — | 3 arms × 2 seeds trained | 50 | 0 | not evaluated | | | | | `eval_v1010.sh` |
| v1011c | — | 3 arms × s0 trained | 50 | 0 | not evaluated | | | | | `eval_v1011c.sh` |
| v6 | — | trained | 50 | 0 | not evaluated | | | | | `eval_v6.sh` |
| v12 | — | trained | 50 | 0 | not evaluated | | | | | `eval_v12.sh` |

Source column: plain names are `eval_output/<name>/`; `*.json` are in `reports/eval_summaries/`.

## Notes

- v1011d cup side: 21 of the 50 T-1011d cups start on the bar side; every arm placed 0/21 there
  (demos only have the cup away from the bar). `--target_support train` fixes the sampler.
- v2 blur ladder: null at n = 25 (±40 points noise). v2 checkpoints and data were deleted
  2026-08-24; JSONs only.
- v1 numbers live only in wandb (`act-obstacle-baseline-eval`); no JSON on disk.
- Excluded: `_verify_*` / `_parity_*` / `_smoke_*` wiring checks (horizon 1–5 steps),
  n = 10 `*_diag10` diagnostics, DIRTY `…_keep0_smoke` / `…_keep50_smoke`, empty dirs
  (v5 raw s1, readout s1, raw keep75). Avoid-v1 (2026-08-24) has README prose only, no JSON.
