# PACT place V10.6: V9.5 real clutter with an asymmetric static pendant

**Status: stopped at Step 4b. The contact-risk certificate did not reach
contact within its registered 30 mm cap. That outcome is
*diagnostically inconclusive* — it measured the reach of a straight-line TCP
displacement, not physical reachability, and does not establish that contact is
infeasible. Superseded by V10.7, which retires that test as a gate.**

Successor to V10.5. The V10.5 narrative is treated as untrusted; its artifacts
are preserved unmodified and re-verified by an independent audit before
anything here runs.

## Why a successor exists

V10.5 scored a symmetric lattice and selected nothing. An audit of that run
(`diagnostics_output/pact_place_v105_audit/audit.json`) confirmed the sealed
numbers but corrected four reporting claims — see the erratum section below.
Two audited facts drive this successor:

1. The best symmetric candidate, `x = 0.800, r = 0.325`, reaches a
   **13.4388185 mm** floor with **zero** exact contacts and only **2 of 294**
   evaluations below 15 mm. That is far closer to admissible than the V10.5
   narrative conveyed.
2. The meaningful near-pass is **predominantly loaded outbound**, and the two
   intrusion sides bind at different radii. A symmetric assembly must therefore
   compromise one side to satisfy the other.

V10.6 keeps everything else frozen and gives the two lobes independent radii.

## 1. Audit and erratum (complete, sealed)

`scripts/audit_pact_place_v105.py` recomputes raw and canonical payload hashes
for `reconstruction.json`, `corpus_index.npz`, `siting.json` and
`per_row_scores.npz`; independently verifies all 192 retained result and
trajectory payloads; and re-aggregates every bundle statistic with a **second,
independently written flat-table aggregator** required to agree exactly with
the primary scorer.

Corrections to the record:

| id | claim as reported | status | correction |
|---|---|---|---|
| E1 | 21 active clutter free bodies | **incorrect** | **8** household objects. The 21 counted nested MuJoCo mesh child bodies and the four corridor chicane bodies as distinct objects. |
| E2 | `x=0.800, r=0.320` is the highest-floor candidate | **incorrect** | The highest symmetric floor is `x=0.800, r=0.325` at **13.4388185 mm**, 0 contacts, 2/294 below 15 mm. `r=0.320` is second at 9.3898271 mm. |
| E3 | 98/192 strict-clean rows | **valid** | Confirmed. 4 rows retain no trajectory; all four are unclean, so the count is unaffected. |
| E4 | `risk_group_counts` | **ambiguous name** | Renamed `band_evaluations_by_group`: per `pose_id\|side` group, the number of **(trajectory, pose) evaluations** whose lobe/stem minimum lies in the 15–35 mm band. Never a count of groups, trajectories, or frames. Companion field `evaluations_ge_floor_by_group` reports how many evaluations sit at or above the 15 mm floor. |

## 2. Registered asymmetric global lattice

One global assembly for every layout family — **not** per-family placement.

```text
x                    = 0.800 m
negative-side radius = {0.325, 0.330, 0.335} m
positive-side radius = {0.295, 0.300, 0.305} m
whole-assembly d     = {-0.005, 0.000, +0.005} m   (pose IDs neg5/center/pos5)
```

9 candidates × 3 poses = 27 scenes. Shape, height, lobe/stem/crossbar
dimensions and the compiled-static discipline are inherited unchanged: lobe
bottom z = 0.98, crossbar top flush with `hood_top` at 1.515.

The crossbar is no longer centred on `d`. Its centre and half-length are
derived from the two asymmetric stem centres so that it overlaps the full
12 mm width of both stems; connectivity is asserted, not assumed. Every
production scene is compiled-static — no joint, freejoint, mocap flag or
actuator, and no runtime write to `geom_pos`, `geom_size`, `geom_aabb`,
`geom_rbound` or BVH fields.

## 3. Rescoring and preregistered admission

Every candidate is scored against **all 98** historical clean trajectories and
all three poses — 294 evaluations per bundle — with no early termination.

A candidate with **universal ≥15 mm clearance** is preferred. If none exists,
the fallback below may admit a candidate. It was written into
`scripts/pact_place_v106_contract.py` before any V10.6 score existed and is not
edited afterwards:

- zero exact pendant contacts across all 294 evaluations;
- absolute minimum clearance ≥ 10 mm;
- ≥ 90% of all evaluations ≥ 15 mm;
- ≥ 80% of every `pose × side` group ≥ 15 mm;
- grasp/lift/release and initial-state clearance universally ≥ 15 mm;
- every `pose × side` group has a 15–35 mm lobe/stem witness;
- loaded-outbound risk present on both sides.

**Inbound risk is not required.** The sealed V10.5 evidence shows the
meaningful near-pass is predominantly loaded outbound.

## 4. Certification before any episode

The selected compiled scenes are certified at every minimum and every
threshold-near witness with analytic GJK, signed/true `mj_geomDistance`, live
`data.contact`, and the place contact audit, all required to agree. Compiled
AABB/BVH bounds must enclose the true geometry, the body must be static, and
state restoration is verified. The existing contact-risk and raw-proximity
causal checks then run. Any failure stops V10.6.

## 5. Review pool with scaled yield floors

The frozen 48-row pool must clear floors scaled from the 16/24 Phase-0 bar
**before** any packet is published:

- ≥ **32/48** overall;
- ≥ **14/24** per side;
- ≥ **8/16** per pose;
- ≥ **4/8** per `side × pose`.

This prevents presenting a curated six-video packet for an environment already
unlikely to pass Phase 0.

## 6. Packet and stop

If the pool passes, publish exactly six **complete production** videos — three
natural strict-clean successes and three natural failures, balanced as
previously specified — then stop. The agent does not create
`human_approval.json` and does not run Phase 0.

## Boundaries

No V10.4 or V10.5 artifact is modified. New files and new output directories
only. Collection, conversion, training and learned-policy evaluation remain
unauthorized regardless of outcome.

---

# Execution record

**Status: stopped at Step 4b. Certification and causality passed; the
contact-risk certificate did not reach contact within its registered 30 mm cap.**

Executed 2026-08-28. Measured outcomes only.

## Step 1 — audit and erratum: passed

`diagnostics_output/pact_place_v105_audit/audit.json`, payload
`93156a32…`, raw `89cd499b…`.

Sealed V10.5 artifacts, recomputed:

| artifact | raw SHA-256 | canonical payload SHA-256 |
|---|---|---|
| `reconstruction.json` | `71bcb635…` | `ccace4b0…` |
| `corpus_index.npz` | `4e26d38a…` | `20538135…` |
| `siting.json` | `56f5d6ba…` | `4c9f5646…` |
| `per_row_scores.npz` | `a717be69…` | `8dba91ba…` |

Both JSON artifacts are self-consistent. All **192/192** retained rows verify,
0 problems. Four rows retain no trajectory (`06_d83398b4`, `07_06f17809`,
`07_99b68628`, `07_9e0a5487`); **all four are unclean**, so the 98/192 count is
unaffected — that is checked, not assumed.

The independent flat-table aggregator reproduced **9408** evaluations and agreed
with the primary scorer on **192/192** checks with **0 disagreements**.

Errata are recorded in the audit artifact (E1 object count, E2 highest-floor
candidate, E3 confirmed valid, E4 field-name definition).

## Step 2/3 — asymmetric lattice: a candidate qualifies

`diagnostics_output/pact_place_v106_siting/siting.json`, payload `17e803e6…`,
raw `34949f17…`. 27 scenes scored against **98/98** clean trajectories, no
failed rows, no early termination.

**9/9 candidates admitted; 4 achieve universal ≥15 mm clearance.**

| x \| r_neg \| r_pos | abs min (mm) | below 15 mm | contacts | frac ≥15 mm | basis |
|---|---:|---:|---:|---:|---|
| 0.800 \| 0.325 \| 0.295 | 13.4388 | 3 | 0 | 0.9898 | fallback |
| 0.800 \| 0.325 \| 0.300 | 13.4388 | 2 | 0 | 0.9932 | fallback |
| 0.800 \| 0.325 \| 0.305 | 13.4388 | 2 | 0 | 0.9932 | fallback |
| 0.800 \| 0.330 \| 0.295 | 14.7889 | 1 | 0 | 0.9966 | fallback |
| 0.800 \| 0.330 \| 0.300 | 16.8435 | 0 | 0 | 1.0000 | **universal** |
| 0.800 \| 0.330 \| 0.305 | 17.4878 | 0 | 0 | 1.0000 | **universal** |
| 0.800 \| 0.335 \| 0.295 | 14.7889 | 1 | 0 | 0.9966 | fallback |
| 0.800 \| 0.335 \| 0.300 | 16.8435 | 0 | 0 | 1.0000 | **universal** |
| **0.800 \| 0.335 \| 0.305** | **18.5703** | **0** | **0** | **1.0000** | **universal (selected)** |

Selected `x = 0.800, r_neg = 0.335, r_pos = 0.305`: 294 evaluations, absolute
minimum **18.5703 mm**, 0 below floor, 0 contacts, 0 environment intersections,
minimum pendant-to-environment clearance 10.0000 mm, windows and initial state
universally ≥15 mm, loaded-outbound band witnesses on both sides, band
evaluations per group 9–37.

The asymmetry is what made this possible: the negative side needs 0.335 m and
the positive side only 0.305 m — a 30 mm difference no symmetric assembly can
express. The preregistered fallback was not needed.

## Step 4a — certification: passed

`diagnostics_output/pact_place_v106_certification/certification.json`, payload
`ad5ca617…`, raw `d60119b4…`. Three production scenes and one no-pendant
counterfactual published (created, not overwritten):

| pose | scene SHA-256 |
|---|---|
| neg5 | `d2ea1b5796b79995…` |
| center | `d4b2e6c136a98a6d…` |
| pos5 | `8ebad1f1cd01811e…` |

All three compile static (dofnum 0, jntnum 0, mocapid < 0), bounds enclose the
true geometry, 58 BVH nodes each, all finite, collision enabled on all five
geoms.

**6/6 witnesses certified**, all four instruments agreeing to five decimals:

| pose | side | recorded | compiled | signed | GJK | limiting pair | phase |
|---|---|---:|---:|---:|---:|---|---|
| center | left | 22.30658 | 22.30658 | 22.30658 | 22.30658 | `lobe_0 ↔ fr3_link7` | outbound_vessel_pass |
| center | right | 20.49394 | 20.49394 | 20.49394 | 20.49394 | `lobe_1 ↔ gripper/left_coupler` | outbound_vessel_pass |
| neg5 | left | 25.18797 | 25.18797 | 25.18797 | 25.18797 | `lobe_0 ↔ fr3_link7` | outbound_vessel_pass |
| neg5 | right | 18.57030 | 18.57030 | 18.57030 | 18.57030 | `lobe_1 ↔ gripper/left_coupler` | outbound_vessel_pass |
| pos5 | left | 19.72019 | 19.72019 | 19.72019 | 19.72019 | `lobe_0 ↔ fr3_link7` | outbound_vessel_pass |
| pos5 | right | 23.26553 | 23.26553 | 23.26553 | 23.26553 | `lobe_1 ↔ gripper/left_coupler` | outbound_vessel_pass |

State restored and verified at every witness. Every witness falls in
`outbound_vessel_pass`, independently confirming the audited finding that the
meaningful near-pass is loaded outbound.

Threshold-near witnesses (≤20 mm, registered before the run): **0** beyond the
six minima. Verified from the raw scores — only 2 of 294 evaluations are
≤20 mm, and both are group minima.

## Step 4b — raw proximity causality: passed

`diagnostics_output/pact_place_v106_causal/risk_causal.json`, payload
`6338f85cbac5a681…`, raw `b4464ea48dadd116…`.

**Hash correction.** The first version of this record quoted
`a0127a5e…`/`e1f4e76c…`, which are the hashes of the *first, defective* probe
run, not of the published artifact. The three runs are:

| run | payload SHA-256 | raw SHA-256 |
|---|---|---|
| attempt 01 (probe defects) | `a0127a5e12a902e9…` | `e1f4e76cfcfbca94…` |
| attempt 02 (grasp counted as collision) | `01a1ee2b2b0066e8…` | `b8ef9d14b0f4eb9e…` |
| **published** | **`6338f85cbac5a681…`** | **`b4464ea48dadd116…`** |

All three record `causal_passed: true`; they differ only in the contact probe. Selected scene versus the compiled no-pendant
control at byte-identical state, real `[40, 4, 8, 8]` tensor.

| side | changed values | sensors | responding links | onset | deterministic |
|---|---:|---:|---|---|---|
| left | **4512** | 7 | link2, link3, link5_back, link6 | 60 frames / 3.96 s | yes |
| right | **2004** | 9 | link2, link3, link4, link5_back, link6 | 60 frames / 3.96 s | yes |

Side ratio **2.251** against a 4× limit. All seven registered causal checks pass.

## Step 4b — contact-risk certificate: inconclusive, and this was the stop

No group reaches robot-pendant contact within the registered 30 mm displacement
cap.

| pose | side | baseline (mm) | remaining at 30 mm (mm) |
|---|---|---:|---:|
| center | left | 22.307 | 11.904 |
| center | right | 20.494 | 5.224 |
| neg5 | left | 25.188 | 15.953 |
| neg5 | right | 18.570 | **1.018** |
| pos5 | left | 19.720 | 13.054 |
| pos5 | right | 23.266 | 9.719 |

The probe is sound as an instrument: IK solves at every magnitude, no spurious
collisions, and the closest case ends 1.018 mm short. The measured reduction is
sub-linear and saturating — a straight-line TCP displacement does not translate
the limiting link one-for-one, so 30 mm of TCP travel buys 6–13 mm of approach.

**This result is diagnostically inconclusive, not a demonstration of physical
infeasibility.** A cardinal-axis TCP excursion is one particular, weak
operationalization of "can the robot reach the pendant"; it holds the carried
target fixed while the arm moves, and it explores no joint-space or
closest-point direction. It does not license any claim that contact is
unreachable. V10.7 retires it as a gate and keeps a repaired version as a
non-gating diagnostic.

Per this plan as written, a failed Step-4 check stopped V10.6, so Steps 5 and 6
did not run here. The environment itself was never disqualified.

### Three probe defects found and fixed before this result was accepted

All three were mine, in the instrument, and each would have produced a wrong
conclusion:

1. **Pre-existing contacts counted as new.** Every robot contact was treated as
   a new collision, including the 7 inherited from the retained state, so the
   probe aborted at the first magnitude.
2. **Wrong displacement direction.** The TCP-to-lobe-centre vector *increased*
   clearance on the left side, because the limiting pair is
   `lobe_0 ↔ fr3_link7`, not the TCP. Replaced with a measured descent
   direction chosen by probing the six axes and keeping whichever actually
   reduces the exact distance.
3. **The grasp counted as a collision.** All six witnesses are loaded-outbound
   frames, so the cup is held; solving IK while the recorded cup pose stays put
   perturbs the pad-to-cup contact set. Those pairs are the grasp, not a
   collision, and are now excluded.

The uncorrected runs are preserved at
`..._causal_attempt_01_probe_defects/` and
`..._causal_attempt_02_grasp_counted_as_collision/`.

### The open question for the owner

The environment is geometrically qualified with universal ≥15 mm clearance and
a decisively passing causal signal, but the pendant is not reachable by a 30 mm
straight-line TCP displacement. Two readings, and choosing between them is an
owner decision because changing the probe after seeing results is exactly what
preregistration forbids:

- **The operationalization is too weak.** A closest-point direction between the
  limiting geoms, or a joint-space perturbation, would likely reach contact —
  `neg5|right` misses by 1.018 mm.
- **The environment is genuinely too far.** If a 30 mm TCP excursion cannot
  touch the pendant, the hazard may not be behaviourally consequential enough
  for the intended comparison, which is the question V10.5 was built to answer.

**Resolved in V10.7:** the first reading was correct. The relevance test is now
natural exact clearance in the 15–35 mm band for all six groups plus six-group
causal sensing, and the contact perturbation is a non-gating diagnostic.

## Not done

No production episode, `env.step`, review-pool row, Phase-0 row, collection,
conversion, training, or evaluation occurred. `pact_place_v106_review_pool/`,
`..._review/` and `..._phase0/` do not exist. `human_approval.json` was neither
created nor required. No V10.4 or V10.5 artifact was modified.
