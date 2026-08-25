# V9.5 low-wall status: implementation complete, raw prerequisite failed

V9.5 is implemented as an isolated, wall-only successor to V9.4. The ceiling
fixture is always parked. A low wall fixture must lie wholly within x
`[0.58, 0.86]`, have bottom z in `[0.87, 0.98]`, top z in `[1.06, 1.15]`, and
extend at least 50 mm below the measured wrist-visibility ceiling at z `1.03`.

The agreed raw-first prerequisite stopped execution before any V9.5 fixture
episode or video was generated. The authoritative result is:

`diagnostics_output/pact_place_v95_v0c5_raw_prerequisite/validation.json`

## Raw result

- Full matrix: 8 variants.
- Passing variants: **0 of 6 physics-clean variants**. As originally published this line
  read "1 of 8"; see the correction below.
- Inbound vessel causal activation: left 1/4, right 0/4.
- Outbound vessel causal activation: left 4/4, right 2/4.
- Panel causal activation: nonzero in 8/8.
- Overall `passed`: false.

### Correction (E0): the null is stronger than first published

`diagnostics_output/pact_place_v95_v0c5_raw_prerequisite/admission_correction.json`.

Joining the raw admission variants to the smoke summary's `clean_success` shows that the single
passing variant is the one whose source episode is **not collision-free**:

| # | family | side | raw_passed | source_physics_clean |
|---|---|---|---|---|
| 0-5 | F0 / F1 / F2 | both | False | True |
| 6 | F3_aperture_side_stagger | left | **True** | **False** (351 clutter contacts) |
| 7 | F3_aperture_side_stagger | right | False | **False** (2,315 clutter contacts) |

**Every physics-clean variant failed.** The headline is **0 of 6**, not 1 of 8. The inbound vessel's
only nonzero reading in all of V9.5 — 40 changed values on `link5_back_sensor_4`, max 3.1 px at
R = 0.11 m — comes from that dirty episode; with the arm already contacting clutter that is a sensor
nearly touching an object, not a detection at range. The retained `validation.json` carries
`source_physics_clean: null` because the guard postdates it; the guard is now in
`scripts/run_pact_place_v9_v0c3_causal_proximity.py` and a dirty source is a hard admission failure.
No V9.5 artifact was edited — all 34 are byte-identical and the correction is a new record.

The raw measurement uses the production `[40, 4, 8, 8]` tensor and frozen-qpos
present-versus-parked counterfactuals. Geometry proxies do not authorize this
gate. A proxy siting sweep and three real-tensor coordinate probes were also
retained; none produced paired-side admission.

## Implemented behind the blocker

- `PactPlaceCorridorV95LowWallSampler` with no active ceiling fixture.
- A wider independently settled inbound vessel variant isolated from V9.3.
- Exact IK plus MuJoCo geom-distance lateral fixture-bow search.
- Post-rollout exact cup/finger/hand/link5/link6/link7 clearance validation.
- Fixture-present versus fixture-parked raw PACT causality, including a
  link5/link6-specific activation requirement.
- Four-cell panel-side x wall-side runner that refuses to start unless the raw
  vessel prerequisite passes and refuses to render until fixture admission
  passes.

No V9.5 artifact authorizes a gate or collection. The next valid action is to
resolve the right-side vessel sensing asymmetry; adding or lowering fixtures
cannot substitute for that result.
