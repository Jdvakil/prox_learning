# V10.1 Empirical Pendant Qualification Plan

Status: **stopped after review and fail-closed causal proximity.**
The 12-row review pack is not eligible for human review (0/12 clean
successes). Causal proximity did not replay tensors because no family×side
cell had a clean row. Phase 0 was not run. The agent did not infer owner
approval. Collection, training, and evaluation remain unauthorized.

V10.1 replaces further exhaustive route search with a short empirical
qualification of the registered V10 `planning_probe_assembly()` /
probe_v2 two-lobe pendant. Route-v1 remains a valid scoped historical
result under contiguous-group-freeze. Route-v2 remains a historical result
of its flawed scalar environment predicate and is **not** cited as physical
infeasibility. All V9.9/V10 siting and route artifacts are preserved and
are not overwritten.

Success of this qualification supports only a human decision about whether
the frozen probe_v2 assembly, under the frozen F0–F2 V9.5 layout and the
frozen endpoint-only empirical route, produces clean expert execution and
real causal proximity. It does not authorize collection.

## Frozen design

Sampler: `PactPlaceCorridorV10CompoundPendantSampler`.
Scene: `pact_place_corridor_v10.xml`.
Assembly: exact `planning_probe_assembly()`:

- negative lobe center `[0.70, -0.18, 0.86]`, half `[0.01, 0.04, 0.04]`
- positive lobe center `[0.70, 0.22, 0.86]`, half `[0.01, 0.02, 0.02]`
- unchanged derived stems and ceiling crossbar

Families: V9.3/V9.5 F0–F2 only, with existing paired-side-identical
clutter, jitter bounds, active panel, V9.5 palette, and target
distribution. F3, three-lobe search, catalog search, alternative lanes,
and alternative pendants are not authorized.

Frozen route for every row:

- `rewrite_primitive`: `endpoint_only`
- `qualification_mode`: `empirical_live_contact_v1`
- slab padding `0.08` m inbound and outbound
- left lane `-0.30` m, right lane `+0.30` m, inbound and outbound
- left-panel rows use the left lane; right-panel rows use the right lane
- unchanged start/end poses, ≥5 cm registered detour, ≤5 mm translation
  densification, ≤2° rotation densification

Runtime dispatch is conditional. Historical V10 rows without the new route
markers keep contiguous-group-freeze `plan_lane` and the offline split
clearance. Endpoint-only rows use `plan_lane_endpoint_only`. Empirical mode
does not call the flawed scalar strict-environment preclearance. It still
rejects missing parameters, clipping, wrong-way travel, insufficient
detour, malformed paths, and endpoint changes. Actual MuJoCo contact
auditing during rollout is the environment predicate. The prior 25/20 mm
offline pendant-clearance floors are not Phase-0 requirements.

## Contract

`pact_place_v101_empirical_qualification_v1`

- Review stream `pact-place-v10.1-pendant-human-review`, master seed
  `2026091002`
- Gate stream `pact-place-v10.1-pendant-phase0`, master seed `2026091001`
- Zero review/gate task-seed intersection
- Self-hashed rows, assembly, route, scene, implementation, and protected
  historical artifacts

## Execution sequence

### 1. Preflight and 12-row review

Before episodes, verify the frozen V10 siting-v2 trust anchor, V10 scene
hash, and byte hashes of historical V9.9/V10 artifacts. Reproduce
endpoint-only geometry admission for the fixed route on all six F0/F1/F2
side cells and both directions. Require 12/12. Do not search alternative
lanes.

Run targeted and regression tests before generating rows.

Generate exactly 12 review rows: two independent paired repeats for every
F0/F1/F2 family, six left and six right. Run every row with the real
expert and contact audit. Render every row — success, task failure,
sampling failure, and infrastructure failure — as a three-pane review
video with route, phase, contact, clutter-drift, endpoint, and pendant
telemetry.

The review manifest is immutable and carries `authorizes_gate: false` and
`authorizes_collection: false`.

Automatic eligibility for human review requires:

- all 12 rows reconcile
- zero infrastructure failures
- at least 10/12 clean successes
- at least 1/2 clean success in each family×side cell
- no fallback, clipping, wrong-way route, missing telemetry, or endpoint
  mutation
- clean rows satisfy task success, zero pendant/panel/clutter/other-
  environment contact, zero clutter-stability event, and no receptacle
  contact outside the allowed placement window

Any code, geometry, route, threshold, or manifest change invalidates the
complete review pack and requires a new versioned pack.

### 2. Causal-proximity validation and human stop

From the lowest-role-index clean review row in each family×side cell,
replay retained qpos without `env.step`. Render the real production
40-sensor tensor for pendant-present, present-repeat, and whole-assembly-
parked worlds. Evaluate inbound and outbound decision windows separately.

Require in every cell and direction:

- at least 3 distinct changed sensors
- at least 448 changed values above `max(ABS_DELTA_FLOOR_M, 10 × repeat noise)`
- at least one responding link5/link6 sensor

For each family and direction, the left/right changed-value ratio must be
at most 4×. A silent/missing cell or failed balance rule blocks Phase 0.

Stop and return the complete review gallery, review manifest, causal
artifact, and concise failure table to the owner. The agent must not infer
approval.

### 3. Phase-0 gate after approval only

Record the owner decision in a self-hashed `human_approval.json` with
`decision: approve_phase0` and the exact review, causal, and contract
hashes. Freeze the Phase-0 manifest before row 0. Run exactly 24 untouched
rows: eight per F0/F1/F2 family and four per family×side cell.

Permit process resumption only for an unfinished exact row. Never replace
a terminal row, change its seed, substitute another row, tune the route,
or retry the gate under the same version.

Pass requires all 24 rows reconcile, zero infrastructure failures, every
clean row meeting the review clean-success definition, and complete
hash-consistent route/pendant telemetry.

On failure, write a permanent stop artifact and leave every authorization
false. On pass, write `phase0_passed: true` and
`eligible_for_separate_collection_authorization: true`, but keep
`authorizes_collection`, `authorizes_training`, and
`authorizes_evaluation` false.

## Measured status

The plan file existed before episodes. Hashes below are the measured
review and causal close-out. Phase 0 remains unauthorized.

| Stage | Status | Artifact | Hash |
| --- | --- | --- | --- |
| Contract | frozen | `pact_place_v101_empirical_qualification_v1` | `c550badd4a95bb0f46c84744ca9cb52b6fd5aa4290377edb8aacc2458087873b` |
| Implementation digest | frozen with the review pack | `implementation_sha256` | `2a716960c82a90f6983eeb6ef881f324c180c4d123a9f70eedb419a04e05dde4` |
| Historical artifact preflight | passed; protected V9.9/V10 hashes unchanged | `diagnostics_output/pact_place_v101_empirical_review/preflight.json` | `37f1d3739376a0fe8b84c7822ac5aeb65ac1fff61a2b84f91540bdf3ac89f7f3` |
| Fixed-route geometry 12/12 | **12/12 admitted** on V9.9 snapshot stock TCP; no alternative lanes searched | same preflight | same |
| 12-row review | 12/12 reconciled; 0 infrastructure; **0/12 clean**; not eligible | `diagnostics_output/pact_place_v101_empirical_review/review_manifest.json` | `3a472dc165b0053766478dbc3f9e64af54b66e324ec6abd522bde9840675c841` |
| Review summary | same eligibility | `diagnostics_output/pact_place_v101_empirical_review/summary.json` | `5da45b65a3cbc0820a900b6562df2d1c9fcec25e22cf7fd4a3fef74f00718618` |
| Review config | frozen before row 0 | `diagnostics_output/pact_place_v101_empirical_review/config.json` | `529ec206a4808019f8b049b97845d23edeb2f717ad45590cc8c4fa6685d482a0` |
| Review gallery | 12/12 three-pane videos, including the sampling-failure row | `diagnostics_output/pact_place_v101_empirical_review/videos/` | (12 mp4 files named in the summary) |
| Causal proximity | **fail-closed**; no `env.step`; no tensors | `diagnostics_output/pact_place_v101_empirical_causal/causal.json` | `30329d737be32663b93802c88ba6ced22a121e06b727ca2baa947747018b9364` |
| Failure table | companion to the causal artifact | `diagnostics_output/pact_place_v101_empirical_causal/failure_table.json` | `74688c41674a15be909c2889b921e7809daa2fcc5a3cc75805d29e8e78fc1057` |
| Owner Phase-0 approval | **not inferred** | `human_approval.json` | absent |
| 24-row Phase 0 | **not run** | — | — |
| Collection / training / eval | unauthorized | — | — |

Siting-v2 trust anchor payload SHA
`2e0b2a56bd4c22ecc920927dc149adf9c1bbc0d1d3ccbd3ee433ea450b187c1c`.
V10 scene SHA
`360b1407a01d1447d8b440ade3115866399a1db09efc76321016aa3c04eaddf7`.
Route-v1 and route-v2 payload hashes remain
`c0f1b35084d6950a88531c45e6805b06437add31c82ca5fd68bb5da4f5de3ff7` and
`e311ba01c77c14b3a930be8dd9d4d40e9de483710521f8662d2e3a55357f71e1`.

`eligible_for_human_review: false`. `causal_passed: false`.
`blocks_phase0: true`. `phase0_passed: false`.
`authorizes_gate: false`. `authorizes_collection: false`.
`authorizes_training: false`. `authorizes_evaluation: false`.

### Review failure table

On the 11 complete rows, endpoint-only empirical telemetry is present and
healthy: `rewrite_primitive=endpoint_only`,
`qualification_mode=empirical_live_contact_v1`, no fallback, no clipping,
no wrong-way travel, endpoints preserved, inbound detour ≈9–12 cm,
outbound detour ≈5.1–6.7 cm, and offline strict-environment preclearance
intentionally not used. Pendant/`mounted_fixture`, `hazard_bar`, and
`other_environment` contact are zero. Live failures are expert IK cascade
and clutter-stability events, not pendant collision.

| role | family | side | status | task success | cause |
| ---: | --- | --- | --- | --- | --- |
| 0 | F0 | left | complete | false | `terminal_ik_cascade` |
| 1 | F0 | right | complete | false | `clutter_collision_stability_event` (clutter 394) |
| 2 | F1 | left | complete | false | `clutter_collision_stability_event` (clutter 253) |
| 3 | F1 | right | complete | false | `terminal_ik_cascade` |
| 4 | F2 | left | sampling_failure | false | 13 pre-boundary attempts: clutter/target overlap, pregrasp/lift IK, clutter drift; missing telemetry |
| 5 | F2 | right | complete | false | `terminal_ik_cascade` |
| 6 | F2 | left | complete | false | `terminal_ik_cascade` |
| 7 | F2 | right | complete | false | `terminal_ik_cascade` |
| 8 | F0 | left | complete | true | `clutter_collision_stability_event` (clutter 2875); not clean |
| 9 | F0 | right | complete | false | `terminal_ik_cascade` |
| 10 | F1 | left | complete | false | `terminal_ik_cascade` |
| 11 | F1 | right | complete | false | `terminal_ik_cascade` |

Cause counts: 8 `terminal_ik_cascade`, 3 `clutter_collision_stability_event`,
1 `sampling_failure`. Clean successes by family×side cell: 0/2 in every
cell. Eligibility failures: six `cell_clean_shortfall` codes plus
`missing_telemetry` on role 4.

Causal proximity therefore recorded six `missing_clean_cell` codes and six
`missing_side_window` codes. It did not bind present/repeat/parked tensors.
A silent/missing cell blocks Phase 0.

Any code, geometry, route, threshold, or manifest change invalidates this
pack. Phase 0 still requires an explicit owner `approve_phase0` bound to
review-manifest SHA
`3a472dc165b0053766478dbc3f9e64af54b66e324ec6abd522bde9840675c841`,
causal-artifact SHA
`30329d737be32663b93802c88ba6ced22a121e06b727ca2baa947747018b9364`, and
contract SHA
`c550badd4a95bb0f46c84744ca9cb52b6fd5aa4290377edb8aacc2458087873b`.
That decision is absent. Do not infer it from this close-out.
