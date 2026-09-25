# Results

Updated 2026-09-25 08:45. Every number is recounted from `eval_output/<run>/episodes.jsonl`
(copies in `reports/eval_summaries/`). Every row is 50 test runs; counts are out of 50. Inside one
environment, all three models see the same 50 scenes.

Three models per environment, same recipe (batch 8, lr 1e-5, 2000 epochs, chunk 50; v6 chunk 100):

- **ACT** — cameras only.
- **PACT-raw** — cameras + raw skin readings.
- **PACT-readout** — cameras + skin through a small learned encoder.

## Status

| env | clutter | what it is | demos | ACT | PACT-raw | PACT-readout | ablations |
|---|---|---|---|---|---|---|---|
| v5 | 0 (one side bar) | hallway corridor, wrist camera | 152 | done | done | done | skin off, skin 75 %, cameras off done; readout 2nd seed (normal, skin off, cameras off) done |
| v5_ext | 0 (one side bar) | same hallway, more demos | 193 | done | done | done | — |
| v6 | 2 bottles | corridor bench, two bottles on the route | 200 | done | done | done | skin off, skin 50 %, skin history, cameras off done |
| v107_spaced | 8 tall objects | spaced bench (hub recording) | 200 | done | done | done | — |
| v107_spaced batman | 8 tall objects | same bench, own recording, table camera | 210 | trained; eval invalid (camera fov mismatch) | same | same | rerun at fov 58 not started |
| v1010 | 4 objects + pendant | bench with household objects, table camera | 215 | done | done | done | ACT 2nd seed done; raw, readout 2nd seed not evaluated |
| v1011c | 6 | mixed clutter, fixed seats | 99 | done | done | done | skin off, skin history, ACT 2nd seed done |
| v1011d | 6 | same clutter, positions randomised | 200 | done | done | done | skin off, cup-side check, ACT 2nd seed done |
| v12 | 11 (1 bottle + 10 kitchen items) | kitchen bench | 165 | done | done | done | raw skin off done |
| v107 | pendant + clutter | 48 demos, no evaluator exists | 48 | nothing | nothing | nothing | — |
| v12.1, table_smoke | — | 5 / 10 demo test sets | 5 / 10 | nothing | nothing | nothing | — |

done = trained and evaluated on 50 runs. trained = checkpoint exists, no eval. nothing = not trained.

## Metrics

- **Success** — cup is on the tray at the end of the run. Bumps along the way don't matter.
- **Collision-free** — the robot touched nothing it shouldn't (bar, clutter, walls, fixtures).
  Touching the cup, tray, floor or itself is fine.
- **Strict** — success and collision-free in the same run.
- **Contact time** — seconds per run with the robot (or the carried cup) touching bar, clutter or
  walls. Checked every 2 ms over the whole run (69 s bench, 53 s hallway). In brackets: change vs
  ACT in the same environment.
- **vs ACT** — metrics where the PACT model differs from ACT on the same 50 scenes with p < 0.05
  (paired exact McNemar for counts, Wilcoxon for contact time). ↑ = more, ↓ = less.

## Main results

### Bench environments

| env | model | success | strict | collision-free | contact time (s/run) | vs ACT |
|---|---|---|---|---|---|---|
| v12 | ACT | 30 | 23 | 34 | 2.1 | – |
| v12 | PACT-raw | 26 | 24 | 44 | 1.2 (−41 %) | collision-free ↑ (p = 0.013), contact ↓ (p = 0.010) |
| v12 | PACT-readout | 26 | 19 | 36 | 1.9 (−11 %) | none |
| v6 | ACT | 20 | 13 | 28 | 7.8 | – |
| v6 | PACT-raw | 21 | 11 | 29 | 1.6 (−79 %) | contact ↓ (p = 0.037) |
| v6 | PACT-readout | 21 | 19 | 38 | 3.0 (−61 %) | collision-free ↑ (p = 0.006), contact ↓ (p = 0.004) |
| v1010 | ACT | 25 | 18 | 32 | 2.5 | – |
| v1010 | PACT-raw | 23 | 17 | 35 | 0.8 (−68 %) | none |
| v1010 | PACT-readout | 31 | 21 | 34 | 1.6 (−35 %) | none |
| v1011c | ACT | 12 | 8 | 18 | 8.7 | – |
| v1011c | PACT-raw | 12 | 8 | 25 | 4.9 (−44 %) | contact ↓ (p = 0.002) |
| v1011c | PACT-readout | 11 | 5 | 27 | 6.0 (−31 %) | collision-free ↑ (p = 0.035), contact ↓ (p = 0.015) |
| v107_spaced | ACT | 24 | 8 | 20 | 1.9 | – |
| v107_spaced | PACT-raw | 26 | 14 | 29 | 2.7 (+44 %) | none |
| v107_spaced | PACT-readout | 29 | 16 | 26 | 0.7 (−64 %) | contact ↓ (p = 0.012) |
| v1011d | ACT | 21 | 16 | 24 | 7.2 | – |
| v1011d | PACT-raw | 18 | 16 | 31 | 2.0 (−73 %) | contact ↓ (p < 0.001) |
| v1011d | PACT-readout | 19 | 17 | 32 | 2.6 (−65 %) | contact ↓ (p = 0.002) |

v1011d: cup always on the side the demos used (see ablation below). v1010: table camera at the
training field of view (58°).

### Hallway (v5) — different eval script, wrist camera only

| env | model | success | strict | collision-free | contact time (s/run) | vs ACT |
|---|---|---|---|---|---|---|
| v5 | ACT (training seed 1) | 15 | 12 | 30 | 4.3 | – |
| v5 | PACT-raw | 21 | 19 | 37 | 0.7 (−84 %) | collision-free ↑ (p = 0.039), contact ↓ (p = 0.002) |
| v5 | PACT-readout | 21 | 19 | 41 | 0.6 (−87 %) | collision-free ↑ (p < 0.001), contact ↓ (p < 0.001) |
| v5 | PACT-readout (training seed 1) | 22 | 21 | 44 | 0.3 (−93 %) | strict ↑ (p = 0.049), collision-free ↑ (p < 0.001), contact ↓ (p < 0.001) |
| v5_ext | ACT | 14 | 12 | 32 | 5.6 | – |
| v5_ext | PACT-raw | 22 | 20 | 35 | 1.9 (−66 %) | contact ↓ (p = 0.014) |
| v5_ext | PACT-readout | 21 | 19 | 38 | 0.8 (−85 %) | collision-free ↑ (p = 0.031), contact ↓ (p < 0.001) |

v5 ACT is training seed 1 (its seed-0 eval has only 2 runs); PACT models are seed 0, except the
readout seed-1 row, which is seed-matched to ACT: collision-free 14 vs 0 (p < 0.001), strict 13 vs 4
(p = 0.049), contact time Wilcoxon p < 0.001. In the
hallway, the only thing to hit is the bar, so collision-free = runs that never touched the bar.

## Ablations

### Skin switched off at test time

Same trained model, same 50 scenes, every skin sensor blanked to "nothing near" before the policy
sees it. ACT row for reference.

| env | model | skin | success | strict | collision-free | contact time (s/run) |
|---|---|---|---|---|---|---|
| v1011c | ACT | – | 12 | 8 | 18 | 8.7 |
| v1011c | PACT-raw | on | 12 | 8 | 25 | 4.9 |
| v1011c | PACT-raw | off | 12 | 7 | 26 | 4.1 |
| v1011c | PACT-readout | on | 11 | 5 | 27 | 6.0 |
| v1011c | PACT-readout | off | 8 | 5 | 30 | 3.2 |
| v6 | ACT | – | 20 | 13 | 28 | 7.8 |
| v6 | PACT-raw | on | 21 | 11 | 29 | 1.6 |
| v6 | PACT-raw | off | 24 | 15 | 32 | 3.0 |
| v6 | PACT-readout | on | 21 | 19 | 38 | 3.0 |
| v6 | PACT-readout | off | 22 | 18 | 32 | 4.8 |
| v5 hallway | ACT | – | 15 | 12 | 30 | 4.3 |
| v5 hallway | PACT-readout | on | 21 | 19 | 41 | 0.6 |
| v5 hallway | PACT-readout | off | 14 | 12 | 30 | 4.7 |
| v5 hallway | PACT-readout, 2nd training seed | on | 22 | 21 | 44 | 0.3 |
| v5 hallway | PACT-readout, 2nd training seed | off | 16 | 15 | 34 | 1.7 |
| v1011d | ACT | – | 21 | 16 | 24 | 7.2 |
| v1011d | PACT-raw | on | 18 | 16 | 31 | 2.0 |
| v1011d | PACT-raw | off | 22 | 19 | 32 | 3.7 |
| v1011d | PACT-readout | on | 19 | 17 | 32 | 2.6 |
| v1011d | PACT-readout | off | 23 | 17 | 30 | 2.2 |
| v12 | ACT | – | 30 | 23 | 34 | 2.1 |
| v12 | PACT-raw | on | 26 | 24 | 44 | 1.2 |
| v12 | PACT-raw | off | 24 | 20 | 39 | 2.5 |

Read: hallway readout without skin scores the same as ACT on every metric (collision-free 41 → 30,
paired p = 0.007). The 2nd readout training seed repeats it: 44 → 34 (10 vs 0,
p = 0.002), then level with ACT on collision-free (p = 0.42). v1011c and v1011d models keep their numbers with the skin blanked and still have
less contact time than ACT (paired Wilcoxon p ≤ 0.006, all four); collision-free vs ACT stays
significant on v1011c (raw p = 0.039, readout p < 0.001), not on v1011d (p = 0.15 / 0.18). v6
readout loses 6 collision-free runs (p = 0.11) and its contact time goes up. v12 raw without skin:
contact time doubles (1.2 → 2.5 s, Wilcoxon p = 0.016), collision-free 44 → 39 (5 vs 0, p = 0.062);
no longer different from ACT on collision-free (p = 0.27) or contact time (p = 0.067).

### Skin partly switched off

| env | model | sensors kept | success | strict | collision-free | contact time (s/run) |
|---|---|---|---|---|---|---|
| v5 hallway | PACT-readout | 100 % | 21 | 19 | 41 | 0.6 |
| v5 hallway | PACT-readout | 75 % | 23 | 22 | 36 | 1.7 |
| v5 hallway | PACT-readout | 0 % | 14 | 12 | 30 | 4.7 |
| v6 | PACT-raw | 100 % | 21 | 11 | 29 | 1.6 |
| v6 | PACT-raw | 50 % | 22 | 12 | 30 | 5.2 |
| v6 | PACT-raw | 0 % | 24 | 15 | 32 | 3.0 |
| v6 | PACT-readout | 100 % | 21 | 19 | 38 | 3.0 |
| v6 | PACT-readout | 50 % | 19 | 16 | 34 | 4.1 |
| v6 | PACT-readout | 0 % | 22 | 18 | 32 | 4.8 |

v6 50 % vs 100 %, same scenes: no paired difference (readout collision-free 4 vs 0, p = 0.125). Sensors
kept averaged 0.507. Hallway 75 % vs 100 %: no paired difference (collision-free 1 vs 6, p = 0.125;
success 5 vs 7, p = 0.77). The sensor mask is redrawn per run (random per-link keep).

### Skin history at test time

Bench models normally see the last 8 skin readings (as in training). This run gives them only the
current reading.

| env | model | skin readings | success | strict | collision-free | contact time (s/run) |
|---|---|---|---|---|---|---|
| v1011c | PACT-readout | last 8 | 11 | 5 | 27 | 6.0 |
| v1011c | PACT-readout | current only | 15 | 8 | 28 | 5.7 |
| v6 | PACT-readout | last 8 | 21 | 19 | 38 | 3.0 |
| v6 | PACT-readout | current only | 23 | 18 | 34 | 1.6 |

Current only vs last 8, same scenes: no paired difference (v1011c success 0 vs 4, p = 0.125; v6
collision-free 4 vs 0, p = 0.125).

### Second ACT training seed

Does ACT do as badly with a different training seed? Same recipe, training seed 1, same 50 scenes.
PACT rows are the seed-0 models from the main table; % is contact time vs ACT seed 1.

| env | model | success | strict | collision-free | contact time (s/run) |
|---|---|---|---|---|---|
| v1011c | ACT seed 0 | 12 | 8 | 18 | 8.7 |
| v1011c | ACT seed 1 | 10 | 7 | 24 | 8.0 |
| v1011c | PACT-raw | 12 | 8 | 25 | 4.9 (−39 %) |
| v1011c | PACT-readout | 11 | 5 | 27 | 6.0 (−25 %) |
| v1011d | ACT seed 0 | 21 | 16 | 24 | 7.2 |
| v1011d | ACT seed 1 | 17 | 13 | 24 | 5.1 |
| v1011d | PACT-raw | 18 | 16 | 31 | 2.0 (−61 %) |
| v1011d | PACT-readout | 19 | 17 | 32 | 2.6 (−49 %) |
| v1010 | ACT seed 0 | 25 | 18 | 32 | 2.5 |
| v1010 | ACT seed 1 | 25 | 17 | 36 | 0.3 |
| v1010 | PACT-raw | 23 | 17 | 35 | 0.8 (+167 %) |
| v1010 | PACT-readout | 31 | 21 | 34 | 1.6 (+433 %) |

Read: v1011d repeats: PACT contact time still below ACT seed 1 (Wilcoxon raw p = 0.004, readout
p = 0.020; collision-free readout 11 vs 3, p = 0.057). v1011c does not: ACT seed 1 reaches 24
collision-free, and PACT vs ACT seed 1 is n.s. on every metric (contact time raw p = 0.094, readout
p = 0.24). v1010: seed 1 has almost no contact; no difference vs PACT is significant. No success or
strict difference is significant anywhere. v1010 table camera at 58° for both seeds.

### Cameras switched off at test time (blind)

Same trained model, same 50 scenes, every policy camera fed as a black frame (`BLANK_CAMERAS=all`);
skin and joint angles unchanged. Nothing can be placed without a camera, so read collision-free
and contact time only.

| env | model | cameras | success | strict | collision-free | contact time (s/run) |
|---|---|---|---|---|---|---|
| v5 hallway | ACT | on | 15 | 12 | 30 | 4.3 |
| v5 hallway | ACT | off | 0 | 0 | 23 | 4.8 |
| v5 hallway | PACT-raw | on | 21 | 19 | 37 | 0.7 |
| v5 hallway | PACT-raw | off | 0 | 0 | 5 | 32.3 |
| v5 hallway | PACT-readout | on | 21 | 19 | 41 | 0.6 |
| v5 hallway | PACT-readout | off | 0 | 0 | 43 | 2.3 |
| v5 hallway | PACT-readout, 2nd training seed | on | 22 | 21 | 44 | 0.3 |
| v5 hallway | PACT-readout, 2nd training seed | off | 0 | 0 | 14 | 24.5 |
| v6 | ACT | on | 20 | 13 | 28 | 7.8 |
| v6 | ACT | off | 0 | 0 | 0 | 62.9 |
| v6 | PACT-raw | on | 21 | 11 | 29 | 1.6 |
| v6 | PACT-raw | off | 0 | 0 | 3 | 55.0 |
| v6 | PACT-readout | on | 21 | 19 | 38 | 3.0 |
| v6 | PACT-readout | off | 0 | 0 | 0 | 43.1 |

Read: blind hallway readout stays collision-free as often as with cameras (41 → 43, paired 7 vs 9,
p = 0.80) and beats blind ACT (25 vs 5, p < 0.001; contact time Wilcoxon p = 0.009). Blind raw
collapses (37 → 5, p < 0.001), worse than blind ACT. v6: every model fails blind (0–3 of 50
collision-free); the skin alone does not carry the bench task.

The blind hallway result does not repeat on the 2nd readout training seed: 14 collision-free vs 43
for seed 0 (paired 1 vs 30, p < 0.001), and not better than blind ACT (11 vs 20, p = 0.15).

### Cup position, v1011d

The v1011d scene generator puts the cup on either side of the bar, but 199 of 200 demos have it
away from the bar. With the cup under the bar no model ever placed it (0 of 21 for all three).

| cup | model | success | strict | collision-free | contact time (s/run) |
|---|---|---|---|---|---|
| either side (21 of 50 under the bar) | ACT | 13 | 9 | 15 | 11.8 |
| either side | PACT-raw | 12 | 11 | 26 | 3.8 |
| either side | PACT-readout | 13 | 11 | 22 | 7.1 |
| demo side only (main table) | ACT | 21 | 16 | 24 | 7.2 |
| demo side only | PACT-raw | 18 | 16 | 31 | 2.0 |
| demo side only | PACT-readout | 19 | 17 | 32 | 2.6 |

### Hallway repeats (older checkpoints and evaluator)

| set | model | success | strict | collision-free | contact time (s/run) |
|---|---|---|---|---|---|
| H-A: Aug checkpoints, old evaluator, random scenes | ACT | 14 | 13 | 33 | 5.3 |
| H-A | PACT-raw | 21 | 17 | 32 | 1.0 |
| H-A | PACT-readout | 20 | 20 | 44 | 0.4 |
| H-C: Aug readout checkpoint on the current evaluator | PACT-readout | 18 | 18 | 43 | 0.7 |
| H-B (main table): Sep 10 checkpoints | ACT | 15 | 12 | 30 | 4.3 |
| H-B | PACT-raw | 21 | 19 | 37 | 0.7 |
| H-B | PACT-readout | 21 | 19 | 41 | 0.6 |

H-A has no scene seeds, so its three models did not see the same scenes. Earlier H-A runs with 20
test runs each: ACT 3 / 3 / 14, PACT-raw 7 / 6 / 16, PACT-readout 5 / 4 / 14 (success / strict /
collision-free).

### Camera mismatch (invalid, do not cite)

v107_spaced batman (210 demos, table camera): eval rendered the table camera at 45°, training
used 58°, and gave the skin models query-spaced history. Rerun at 58° (`eval_v107_spaced_batman.sh`)
not started.

| env | model | success | strict | collision-free | contact time (s/run) |
|---|---|---|---|---|---|
| v107_spaced batman | ACT | 13 | 7 | 19 | 11.4 |
| v107_spaced batman | PACT-raw | 0 | 0 | 19 | 4.8 |
| v107_spaced batman | PACT-readout | 2 | 2 | 20 | 11.4 |

### Older and side runs (not in the main tables)

| run | model | runs | success | strict | collision-free | contact time (s/run) | why not main |
|---|---|---|---|---|---|---|---|
| v1011d, Sep 3 checkpoint, clutter spread 0.25, query-spaced history | PACT-raw | 50 | 14 | 11 | 22 | 7.8 | old checkpoint; raw only |
| v1011d, Sep 3 checkpoint, clutter spread 1.0 (full), query-spaced history | PACT-raw | 50 | 7 | 4 | 20 | 5.3 | old checkpoint; raw only |
| v1011d Sep 3 checkpoint on the v1010 scene generator, horizon 800 | PACT-raw | 48 | 0 | 0 | 32 | 0.8 | wrong scenes (out of distribution); no seeds |
| same, horizon 1050 | PACT-raw | 48 | 0 | 0 | 37 | 1.0 | same |
| same, ray-cast skin | PACT-raw | 50 | 0 | 0 | 28 | 1.6 | same |
| v1011d cup either side, first 24 scenes, run 1 | PACT-readout | 24 | 5 | 4 | 11 | 6.5 | repeat-run noise check |
| same, run 2 | PACT-readout | 24 | 4 | 4 | 11 | 5.7 | same |
| same, run 3 | PACT-readout | 24 | 2 | 2 | 11 | 6.0 | same |

Noise check: the same checkpoint on the same 24 scenes three times gives success 5 / 4 / 2 and
collision-free 11 every time. Treat success gaps of ±3 in 24 runs as noise.

Not counted anywhere (too short or broken): hallway ACT seed 0 (2 runs), hallway readout keep 50 %
(stopped at 6), hallway readout H-A stopped at 36 (13 / 13 / 24), hallway `n8_test` (20 runs, random
house: 3 / 2 / 7), v1011d wrist-only (stopped at 4), `*_keep0_smoke` / `*_keep50_smoke` (duplicate
runs), and every `_verify_*`, `_parity_*`, `_equiv_*`, `_ab_*`, `_iso_*`, `_port_*`, `_smoke_*`
wiring check (1–24 runs).

### Old task (fume-hood pick, June–July 2026, deleted)

Different task and metric (any robot–environment contact); success = pick. 50 runs each.

| condition | model | success | collision-free |
|---|---|---|---|
| no bar | ACT | 11 | 20 |
| no bar | PACT-raw (old design) | 9 | 21 |
| no bar | PACT-trunk | 17 | 18 |
| bar visible | ACT | 14 | 18 |
| bar visible | PACT-raw (old design) | 8 | 25 |
| bar visible | PACT-trunk | 16 | 21 |
| bar hidden from camera | ACT | 18 | 17 |
| bar hidden from camera | PACT-raw (old design) | 15 | 30 |
| bar hidden from camera | PACT-trunk | 17 | 14 |

Camera blur on ACT (σ = 2 / 4 / 8, 25 runs each): success 1 / 10 / 6 (no bar), 2 / 6 / 6
(visible), 0 / 10 / 6 (hidden). First v1 baseline (20 runs, W&B only): success 7 / 8 / 6 for
chunk 50 / chunk 100 / 5000 epochs.

## Running now (08:10, 2026-09-25)

Nothing.

## Not run

- v1010 raw and readout, 2nd training seed (checkpoints exist).
- Hallway raw 2nd seed.
- v107_spaced batman rerun at fov 58.
- v1011d with full clutter randomisation (`CLUTTER_XY_SCALE=1`).
- Hallway with 8-reading skin history; hallway skin 50 % / 25 %.
- v1011c skin 50 % (skin off already changed nothing there).
- v5_ext skin off / cameras off (`eval_v5_ext.sh` has no knob for either).

## Run folders

Main results: `eval_output/<TASK>_<ARM>_s0_bs8_cs50_lr1e-5_e2000` (v6: `cs100`; v1011c and v1011d
end in `_tstrain`). Ablations add `_keep0`, `_keep50`, `_keep75`, `_hquery`, `_blind`; 2nd
training seed = `_s1_`; hallway seed-1 readout runs end in `_keep100` / `_keep0` / `_keep100_blind`. Hallway H-A / H-C:
`place_corridor_*_s0_n50*`, `simple_hallway_n50`. Old task: `reports/eval_summaries/*_v2_*.json`.
Env code → dataset → script map: README §0.1. Ledger: `python scripts/exp_tracker.py`.
