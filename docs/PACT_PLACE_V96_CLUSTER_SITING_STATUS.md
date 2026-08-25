# V9.6 clustered-hazard siting: W1 passed, W2 found no admissible siting

**MANDATORY STOP. `authorizes_gate: false`, `authorizes_collection: false`, `authorizes_v1b: false`.**

Executed from [`docs/PACT_PLACE_V9_RAW_ADMISSION_FIX_PLAN.md`](PACT_PLACE_V9_RAW_ADMISSION_FIX_PLAN.md).
W1 and W2 ran. W3 ran once as a pipeline validation only — W2 admitted no configuration to confirm,
and the plan forbids lowering the floor after seeing a result. What follows is the measurement and
the reason.

## The result in one paragraph

The plan's diagnosis was right and its remedy does not fit. The skin needs a contiguous silhouette of
about **0.25 m** to resolve a hazard at working range; the corridor, once the intrusion panel has
taken its half of the aperture, has room for a hazard **0.12 m** wide before the loaded arm envelope
no longer fits past it. Every one of **18,393** inbound/outbound cluster pairs drawn from the sweep
closes the arm's lane under one panel side or the other. The obstruction is not the recipe, the
depth, or the sensor budget: it is that the panel forces the lane to a side which *flips* between
paired left/right rows, while a laterally sited cluster does not flip.

## W1 — the resolving-power instrument, and its retrodiction

New instrument: [`scripts/pact_skin_resolvability.py`](../scripts/pact_skin_resolvability.py).
Driver: [`scripts/run_pact_place_v9_w1_resolvability.py`](../scripts/run_pact_place_v9_w1_resolvability.py).
Artifact: `diagnostics_output/pact_place_v9_w1_resolvability/resolvability.json`.

For every frame of the eight frozen V9.5 trajectories and each of the 40 sensors it computes range,
the hazard's extent across the view axis, `subtense_px = W_perp / (0.1036 * R)`, and two predicted
pixel counts — an occlusion-free ray/box count, and an occlusion-aware `mj_ray` count against the
whole scene under the proximity renderer's own geom-group filter. Replay only: `mj_forward`, no
physics step, no render.

**`retrodiction_passed: true`.** Against the measured V9.5 raw counterfactual, over all 24 role
measurements in 8 variants:

| | |
|---|---|
| Ordering `panel >> outbound > inbound` reproduced | 8 of 8 variants |
| Pearson r on `changed_values` | **0.99997** |
| Spearman r on `changed_values` | 1.000 |
| Predicted / measured ratio (min, median, max) | 0.994, 1.006, 1.042 |
| Measured-nonzero, predicted-zero | **0** |
| Mean recall of the measured responding sensors | **1.00** |

Both vessels are predicted exactly. In variant 6 — the variant V9.5 recorded as its only pass, and
since corrected to a dirty-source episode (see the E0 note below) — the model
returns 40 changed values on `link5_back_sensor_4` for the inbound vessel and 448 + 60 on
`link3_sensor_2` + `link5_back_sensor_4` for the outbound vessel, matching the measurement value for
value. The panel is predicted to within 1.8%, the residual being pixel-centre versus rasteriser edge
cases. The plan's prediction table is met:

| hazard | plan expected | model returned (variant 6, V9.5 decision window) |
|---|---|---|
| panel | many sensors, high subtense | 9 sensors clear 2 px, 1,051 sensor-frames, max 18.0 px |
| outbound bottle | 1-2 sensors marginally over 1 px | 2 sensors clear 1 px, 1 clears 2 px, 63 sensor-frames |
| inbound bottle | at or below 1 px almost everywhere | 1 sensor, 10 sensor-frames of 3,600, max 3.1 px at R = 0.11 m |

One calibration matters downstream: the **occlusion-free** predictor over-states a compact hazard
badly — 31.8x for the inbound vessel, 6.2x for the outbound, 1.33x for the panel. Any siting score
built on geometry alone is an upper bound, which is why W3 exists at all.

### E0 correction: the V9.5 baseline this section is measured against

`diagnostics_output/pact_place_v95_v0c5_raw_prerequisite/admission_correction.json`. Joining the
V9.5 admission variants to the smoke summary's `clean_success`: variants 0-5 (F0/F1/F2, both sides)
are physics-clean and all failed; variants 6 and 7 (F3, both sides) are **not** collision-free — 351
and 2,315 clutter contacts — and variant 6 is the one recorded as a pass. **0 of 6 physics-clean
variants passed**, not 1 of 8.

The inbound vessel's only nonzero reading in all of V9.5 — the 40 values quoted throughout this
document — comes from that dirty episode, at R = 0.11 m with the arm already contacting clutter.
Treat it as a sensor nearly touching an object rather than a detection at range.

**W1 is unaffected.** It predicts the renderer's output from posed geometry; that is a claim about
the sensor model, not about whether the pose was reached without contact. r = 0.99997 stands.
**W2's structural finding is unaffected and slightly strengthened**: its floor required passing in
*every* variant, so the two dirty episodes made admission harder, not easier, and removing them
cannot create an admission that did not exist.

## W2 — the subtense siting sweep

Sweep: [`scripts/run_pact_place_v9_w2_cluster_siting.py`](../scripts/run_pact_place_v9_w2_cluster_siting.py).
Contract: [`scripts/pact_place_v96_cluster_contract.py`](../scripts/pact_place_v96_cluster_contract.py).
Artifact: `diagnostics_output/pact_place_v9_w2_cluster_siting/siting.json`.

14,280 cluster placements — two recipes x two legs x 17 depths x 35 lateral offsets x 6 line angles
— each scored on all eight frozen trajectories and each recorded with its scores and, when rejected,
its reason. Clusters are three tall vessels from the accepted `palette_v9_1.json` records, standing
shoulder to shoulder with a 25 mm gap, spanning 0.29-0.39 m. Scoring uses the physical legs
(everything before the grasp; lift to release) rather than V9.5's phase labels, which name the
segment where *that* layout's vessel sat; the V9.5 decision-window numbers are recorded alongside
for comparability. No admission decision uses TCP clearance, collision clearance, or an AABB gap.

Admission floor, fixed before the sweep: >= 3 distinct sensors clearing 2 px in **every** variant,
a link5/link6 response in every variant, and left/right imbalance <= 4x.

| role | candidates | geometry-feasible | sensing-admitted |
|---|---:|---:|---:|
| inbound cluster | 7,140 | 341 | **0** |
| outbound cluster | 7,140 | 108 | 54 |

The best admitted outbound cluster is `C3_wide` at `x = 0.740`, `y = -0.280`, line angle 90 deg,
span 0.296 m: 4 sensors clear 2 px in the worst of the eight variants (33 of the 54 admitted
placements reach 5), 290 sensor-frames, side imbalance 2.6x, predicted changed values 1,696-16,716
per variant against a floor of 448. **Making
the hazard large works.** It is the placement, not the size, that fails.

The best geometry-feasible inbound cluster reaches 2 sensors and a 12.8x side imbalance. **763**
inbound placements do clear the sensing floor, up to 6 sensors and 333 sensor-frames — and **not one
of them is geometry-feasible**: they sit at `x = 0.55-0.75`, and they are rejected because they
intersect the panel's own volume (771 rejections) or escape the bench workspace (667).

## W2b — why, quantitatively

Diagnostic: [`scripts/run_pact_place_v9_w2b_inbound_diagnostic.py`](../scripts/run_pact_place_v9_w2b_inbound_diagnostic.py).
Artifact: `diagnostics_output/pact_place_v9_w2b_inbound_diagnostic/inbound_diagnostic.json`.

Three measured ceilings, all replay-only against the W1 cache.

**1. The corridor has no room.** The usable aperture is 0.810 m. The active panel's inner face sits
at `y = 0.095` worst case, so with clearance it removes 0.350 m, leaving a free band of 0.460 m. The
loaded transport envelope is 0.300 m wide and wants 0.040 m of clearance. What is left for a hazard
at the panel's depth is:

| | |
|---|---:|
| Max hazard width that still leaves a lane, at the panel's depth | **0.120 m** |
| Contiguous silhouette the skin needs at working range | **0.250 m** |
| Shortfall | **0.130 m** |

**2. Siting deeper than the panel does not rescue it.** At `x = 0.75-0.80`, `|y| = 0.24-0.28` — the
band the panel leaves free — an inbound hazard must be **0.60 m** wide before three sensors clear
2 px with a balanced response, which is wider than the intrusion panel itself (0.480 m) and wider
than the 0.460 m free band it would have to fit inside. On the `+y` side even 0.80 m of width leaves
the imbalance at 4.6-8.9x. The inbound leg is 90 decision frames against the outbound leg's 352, and
at that depth the arm is extended with the wrist near the target, where lateral coverage is thin and
strongly side-dependent.

**3. Going under the panel abandons the hazard.** The panel's underside is at `z = 0.80` and the
bench top at `z = 0.72`, so an object that clears it on both panel sides can be at most **0.070 m**
tall. Fifteen such placements do clear the sensing floor (3-4 sensors, up to 176 sensor-frames,
imbalance as low as 1.95x) — a wide, flat cluster of plates and bowls is genuinely visible to the
skin. It is also 70 mm tall on a bench the arm crosses at `z >= 0.788`, which is not a hazard. This
is recorded as a measured option, not a recommendation.

## The structural reason

Enumerated exhaustively over every geometry-feasible pair
(`scripts/select_pact_place_v96_w3_configuration.py`, artifact
`configs/pact_place_v96_w3_configuration.json`):

| | |
|---|---:|
| Inbound x outbound combinations examined | **18,393** |
| Jointly feasible | **0** |
| Rejected on the corridor lane alone | 7,712 |
| Rejected on the lane and an object overlap | 10,681 |
| Lane closed under the left panel / right panel | 13,749 / 13,851 |

The V9.3 corridor test collapses the corridor to one scalar, which is only correct when the panel
and the blocker share a depth. A cluster is deliberately sited at a different depth, so this uses a
slice-by-slice lane test instead: at each depth the obstacles spanning it remove a band of `y`, and a
lane is admitted only if a free band one envelope wide persists and stays connected from the aperture
to the target. The test admits the settled V9.5 layout with 0.315 m to spare, so it is not simply
strict.

What it exposes is a design contradiction that predates this plan:

- the paired-side design requires **identical clutter geometry** under the left and right panel, so
  the panel side cannot leak through RGB;
- the panel therefore forces the arm's lane to a side **that flips** between the two rows of a pair;
- a hazard must be sited laterally at `|y| >= 0.10` to be sensed at all, because the gripper, hand
  and link7 carry no sensors and nothing covers the centreline;
- but a lateral hazard **does not flip**, so it stands in the lane in exactly the half of the rows
  whose panel pushes the arm to its side.

A hazard symmetric about `y = 0` would flip with nothing, and it closes too: run through the same
lane test at the panel's depth, a centred hazard admits a lane at 0.12 m of width (narrowest free
band 0.305 m) and closes it at 0.15 m (0.290 m, ten millimetres short of the envelope). The 0.12 m
budget is the same number from either direction.

## What this narrows the claim to

Carrying the plan's instruction forward: the skin resolves a contiguous silhouette of roughly 0.25 m
at working range and nothing smaller. A cluster of bottles is, to this sensor, **a panel with a mug
texture** — household clutter to the wrist camera, a slab to the skin. Mugs and apples at
0.075-0.09 m sit below that floor at any useful range; they are RGB-only decor **by physics, not by
choice**, and no v9 report may describe the clustered hazards as "clutter the skin senses" without
also stating the floor that forced them to be that size. The supportable claim remains
**"PACT avoids large obstacles the camera cannot see, some of which are built from household items"**,
not "PACT avoids clutter".

## W3 — run once as a pipeline validation, on a configuration W2 rejected

W2 admitted nothing to confirm, and the plan forbids lowering the floor after seeing a result, so W3
is **not** an admission here. It was run once, on one family, to exercise the V9.6 raw path end to
end and to calibrate the occlusion-free prefilter against the real tensor for a clustered hazard.
The pair rendered — the best admitted outbound cluster with the best geometry-feasible inbound
cluster — closes the corridor lane under one panel side and must never be used as an admission; the
configuration file says so in its own `role` field.

Configuration: `configs/pact_place_v96_w3_pipeline_validation.json`.
Artifact: `diagnostics_output/pact_place_v96_w3_pipeline_validation/validation.json`
(`passed: false`, `authorizes_gate: false`, `authorizes_collection: false`, `authorizes_v1b: false`).

Three things came out of it.

**The replay is exact.** The panel's causal effect on the V9.6 scene is **58,416** changed values on
the left and **23,508** on the right — identical, value for value, to the V9.5 measurement of the
same panel on the same frozen trajectories, despite the scene now carrying twelve prop slots instead
of eight and a different `nq`. The renderer's repeat baseline is again 0.0.

**Clustering does what W1 predicted it would.** On the V9.5 decision window, F3 left — noting that
F3 is the dirty-source family on both sides, so this comparison uses a dirty episode as both baseline
and test and must not be quoted as a clean-source result. E1b re-runs it on F0/F1/F2:

| hazard | V9.5 single vessel | V9.6 cluster |
|---|---|---|
| inbound | 40 changed values, 1 sensor | **2,604 changed values, 3 sensors** (link3, link5_back) |
| outbound | 508 changed values, 2 sensors | **644 changed values, 3 sensors** (link1, link4, link5_front) |

Both clear the pre-registered floor — 3 distinct sensors, 448 changed values, a link5/link6
responder — on that side. Posed geometry, measured from the renderable meshes rather than the
palette's nominal dimensions, gives realized spans of 0.299 m and 0.296 m with maximum gaps of
0.021 m and 0.000 m, inside the 0.25 m / 0.04 m contract.

**The paired-side asymmetry survives at cluster scale.** On the right the inbound cluster changes
16 values on one sensor (`link2`), a 162.8x left/right imbalance against the 4x limit. Scored on the
physical leg instead of V9.5's phase labels it recovers to 660 values on 3 sensors, so much of the
decision-window gap is that those labels name where the *old* vessel sat — but the asymmetry itself
is real and is the same one that stopped V9.5.

The prefilter over-predicted the measured cluster response by 2.6x on the left and 19.8x on the
right, in line with W1's calibration. **No geometry-only score should ever be treated as admission.**

## What is implemented and idle behind the stop

- `PactPlaceCorridorV96ClusterSampler` (`enclosure_reach.py`): twelve prop slots, three cluster
  members per leg, six RGB-only decor items, span and gap contracts enforced at layout time. The
  40-sensor suite, the encoder SHA-256
  `6fd2dd037e3236b5b6bf7fce8cb2709ead0cf52adcbbe9cbad1061efc2fe3206`, the `[T, 40, 4, 8, 8]` tensor,
  the panel geometry and `inter_finger_dist` are all untouched.
- [`scripts/run_pact_place_v96_cluster_causal_proximity.py`](../scripts/run_pact_place_v96_cluster_causal_proximity.py):
  the W3 raw confirmation. It imports the V9.5 validator's rendering path unchanged and replaces only
  the aggregate pass rule, with the floor — 3 distinct sensors, 448 changed values per role and side,
  4x paired ratio, a link5/link6 responder, posed-span >= 0.25 m and posed-gap <= 0.04 m verified
  from the renderable meshes — written to `config.json` before any render. It has been run once, as
  the pipeline validation above, and never as an admission.

## The choices this leaves

Every one of these changes something the plan froze, which is why it stops here rather than picking.

1. **Widen the aperture.** `APERTURE_WIDTH` is 0.85. The corridor needs about 0.13 m more to hold a
   0.25 m hazard beside the panel. This invalidates the v5/v6b/v6c/v7/v8b/v9 corridor lineage.
2. **Shorten the panel's lateral reach.** Its inner face at `y = 0.100` is what removes the aperture's
   usable half. The plan lists the panel's geometry as untouchable, and it is the one hazard with a
   measured effect (-17.9 pp contact|failure in the reach corridor).
3. **Make the hazard flip with the panel.** Site the cluster on the panel's own side so it is always
   out of the lane. This leaks the panel side through RGB-visible clutter unless the target, tray and
   route are mirrored as well — a redesign of the paired-side control, not a siting change.
4. **Give the skin hand coverage.** The plan's own diagnosis says the gripper, hand and link7 carry
   no sensors and that this is why a centreline hazard is invisible. Adding sensors is compatible
   with the encoder (it is per-sensor and shared) but breaks 40-sensor comparability with every
   result to date.
5. **Accept that the place task cannot reproduce the panel result** and report the reach-corridor
   panel finding alone. The place task has never reproduced it (12 vs 13 hazard contacts at N = 40),
   and this measurement gives a geometric reason why.

**No further work should start until one of these is chosen.**
