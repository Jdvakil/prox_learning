# PACT place V10.7: qualification repair

**Status: Steps 1–6 passed; Step 7 stopped at the pool floor (21/48 vs 32/48).
No packet published.**

Successor to V10.6. V10.6 geometry results are historical inputs; no existing
artifact is modified.

## What is being repaired

V10.6 qualified an environment and then stopped on a contact-perturbation test
that measured the reach of a straight-line TCP displacement rather than
physical reachability. Two things follow.

1. **The ranking was not risk-aligned.** V10.6 ranked additional clearance
   above relevance and therefore selected the *farthest* admissible pendant.
   V10.7 keeps universal ≥15 mm clearance as the first key and demotes extra
   clearance below risk relevance.
2. **The cardinal-TCP contact test is retired as a gate.** The registered
   relevance test is natural exact clearance in the 15–35 mm band for all six
   `pose × side` groups, plus six-group causal sensing. A repaired contact
   perturbation may still run as a **diagnostic**, and its outcome gates
   nothing.

## 1. Immutable specification before execution

`specification.json` is written before any stage runs and binds, by raw
SHA-256, every runner, audit script, contract, geometry file, test, sealed
input JSON/NPZ, and base scene. Every later stage re-verifies it and raises
`HashDriftError` on any drift.

Score NPZs are written **before** the manifest JSON that binds them, so each
manifest can record the NPZ's raw SHA-256.

## 2. Risk-aligned re-selection from the sealed V10.6 scores

Keys, most significant first:

1. universal ≥15 mm clearance;
2. all six group minima inside 15–35 mm;
3. **more** risk-band evaluations;
4. smaller mean group minimum;
5. only then, more absolute clearance;
6. deterministic radii tie-break.

The winning bundle is not written into any runner. The selection runner
computes the ranking and asserts its own choice equals an independently
recomputed argmin. Agreement with any externally expected bundle is recorded as
an observation, never as a gate. The selected candidate's six historical group
minima must all lie in 15–35 mm.

## 3. Recompile and certify

The three selected static scenes are recompiled and certified at all six group
minima **and every threshold-near witness** (≤20 mm). Analytic GJK, hardened
signed `mj_geomDistance`, live `data.contact`, and the place contact audit must
agree; any disagreement fails closed.

## 4. Six-group raw proximity causality

Causality runs separately for **all six** `pose × side` groups, not one witness
per side. Each group must independently meet the registered signal, sensor,
link5/link6, onset, determinism and balance rules. A hash-bound NPZ stores
per-sensor and per-frame changed counts and the thresholds used, sufficient for
independent reaggregation.

## 5. Contact perturbation, demoted to a diagnostic

Retained with four repairs: the carried target moves rigidly with the gripper;
only actual gripper-pad-to-carried-target grasp contacts are allowlisted;
worsening baseline penetrations are tracked; and GJK, `mj_geomDistance` and
live contact must agree before contact is called. The artifact records
`diagnostic_only: true` and `gates_qualification: false`, and no downstream
stage reads it.

## 6. Behavioral tests

Cover the real certification and causal runners, hash drift, six-group
coverage, distance disagreement, rigid carried-target motion, and the narrow
allowlist.

## 7. Frozen pool, then six videos

If Steps 1–6 pass, the frozen 48-row pool executes and must meet the scaled
floors — ≥32/48 overall, ≥14/24 per side, ≥8/16 per pose, ≥4/8 per
`side × pose` — **before** any video is rendered. The packet is six complete
production episodes: three natural strict-clean successes and three natural
failures, one per pose in each class, three left and three right overall, at
least two layout families per class. Then stop.

The agent does not create `human_approval.json` and does not run Phase 0.

---

# Execution record

**Status: Steps 1–6 passed. Step 7 stopped: the pool yielded 21/48 against a
32/48 floor, so no packet was published.**

Executed 2026-08-28. Measured outcomes only.

## Steps 1–2 — specification and risk-aligned selection: passed

| artifact | raw SHA-256 | payload SHA-256 |
|---|---|---|
| `specification.json` | `6f96a0e29064d132…` | `4854d6b1ad671a39…` |
| `selection.json` | `2d8600f581d8e80f…` | `5ecd3aa0921b18ec…` |
| `selection_scores.npz` | `d248a764cf007f35…` | — |

The specification binds 9 sealed inputs and 20 implementation files. Every
later stage re-verified it; the drift guard fired for real once, halting a run
when a test file was edited after the specification was sealed.

**Selected: `x = 0.800, r_neg = 0.330, r_pos = 0.300`** — derived, not
hardcoded. The runner asserts its sorted ranking equals an independently
recomputed argmin, and a test asserts no bundle string appears in the runner.
This matches the externally expected bundle, which is recorded as an
observation and not used as a gate.

Absolute minimum 16.8435 mm, 0/294 below floor, 0 contacts, 142 risk-band
evaluations (the highest of the four universal candidates). All six group
minima lie in 15–35 mm: 19.720, 18.570, 22.307, 16.844, 17.488, 20.494 mm.

The risk-aligned ranking selects a *different* bundle from V10.6's, which chose
`0.335|0.305` at 18.5703 mm — the farthest admissible pendant. Demoting extra
clearance below relevance is the whole difference.

## Step 3 — certification: passed

`certification.json` raw `faff1dd0a2e6f16d…`, payload `d31d602b990df273…`;
`certification_scores.npz` raw `cab36c63eb1a6797…`.

Three scenes recompiled (`pact_place_corridor_v10_7_{neg5,center,pos5}.xml`)
plus a no-pendant counterfactual, all compiled-static with enclosing bounds.
**11 witnesses certified — 6 group minima and 5 threshold-near (≤20 mm) — with
0 instrument disagreements.** All six groups covered.

## Step 4 — six-group causality: passed

`causal.json` raw `6f7d640c28b66bce…`, payload `a916a6be722c7494…`;
`causal_scores.npz` raw `3c05c5fb4cb433a8…`.

| group | changed values | sensors | onset | deterministic |
|---|---:|---:|---:|---|
| center\|left | 4592 | 7 | 60 | yes |
| center\|right | 2032 | 9 | 60 | yes |
| neg5\|left | 4648 | 7 | 60 | yes |
| neg5\|right | 2080 | 9 | 60 | yes |
| pos5\|left | 4544 | 8 | 60 | yes |
| pos5\|right | 2012 | 9 | 60 | yes |

Side totals 13784 left / 6124 right, ratio **2.251** against a 4× limit. Every
group independently clears the 448-value, 3-sensor, link5/link6, onset and
determinism rules. The NPZ stores per-sensor and per-frame changed counts and
each group's threshold, sufficient for independent reaggregation.

## Step 7 — pool: 21/48, below the floor, no packet published

`pool.json` raw `7df042aec02588a2…`, payload `a5cfed9d0b12157c…`.

**21/48 strict-clean = 43.8%**, Wilson 95% [30.7%, 57.7%]. Floors demanded
32/48 overall, 14/24 per side, 8/16 per pose, 4/8 per cell; observed 13 left /
8 right, 8 neg5 / 6 center / 7 pos5. All four floors missed.

| defect class | rows |
|---|---:|
| ordinary clutter contact | 21 |
| clutter stability event | 12 |
| task not successful | 11 |
| place phase failed | 11 |
| grasp phase failed | 10 |
| cup not lifted | 4 |
| sampling failure | 1 |
| **robot/target pendant contact** | **0** |

**The pendant caused no failure at all.** Across 48 episodes there were zero
robot-or-target pendant contacts, and clean rows held 16.052–56.082 mm of
pendant clearance. Every failure is the ordinary V9.5 household-clutter
environment.

### What this measures

43.8% [30.7%, 57.7%] brackets the V9.5 corpus's own **51.0%** (98/192) clean
rate. The scaled pool floor of 32/48 and the inherited Phase-0 bar of 16/24 are
both **66.7%** — far above what this expert achieves on real V9.5 clutter. The
V9.5 fragility artifact already recorded this: a canonical varied-seed 24-row
screen expects ~12.25/24 against a bar of 20.

So the floor did exactly what it was registered to do: it refused to publish a
curated six-video packet for an environment that would not pass Phase 0. The
limiting factor is expert yield on real household clutter, not the pendant.

## Defects found and fixed, each preserved

Three plumbing defects were exposed by running real episodes, each fixed with a
regression test, each with its failed run preserved:

1. **Scene-hash guard read the wrong attribute** — `cfg.scene_xml` instead of
   `task_sampler_config.scene_xml_paths`, so it refused every task and all 48
   rows returned `sampling_failure`. Preserved:
   `..._pool_attempt_01_scene_guard_defect/`.
2. **Telemetry not passed through** — the retained row copies an explicit
   subset of `policy_info`, and the V10.5/V10.6 keys were absent, so every row
   read as `missing_frame_telemetry`. Preserved:
   `..._pool_attempt_02_telemetry_not_passed_through/`.
3. **Policy read a sampler attribute** — `_pact_manifest_row` does not exist on
   `PactPlaceCorridorPolicy`, so clearance telemetry was null on every
   completed episode. The assembly parameters now travel through
   `scene_params`. Preserved:
   `..._pool_attempt_03_policy_lacked_manifest_row/`.

A fourth run halted correctly on hash drift after a test file was edited
post-specification; that chain is preserved as `..._03_halted_on_drift`.

## Contact diagnostic — non-gating

Run separately with `diagnostic_only: true` and `gates_qualification: false`.
No downstream stage reads it, and a test asserts that.

## Not done

No Phase-0 row, no packet, no video, no `human_approval.json`, no collection,
conversion, training or evaluation. `pact_place_v107_review/` does not exist.
No V10.4, V10.5 or V10.6 artifact was modified.
