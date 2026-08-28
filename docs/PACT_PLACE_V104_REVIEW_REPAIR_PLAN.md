# V10.4 Review-V2 Diagnostic-Control Repair

**Status: complete. Six videos published; stopped for owner review.**

## Summary

Preserve the successful V10.4 environment, six production episodes, causal
tensors, and all V10.4-v1 artifacts. Build a separately versioned review-v2
packet that repairs the diagnostic controls and provenance chain without
changing production geometry, routing, speeds, seeds, or results.

The review and Phase-0 code is completed and frozen before any artifact is
produced. Exactly six review videos are generated, then execution stops for
owner review. Phase 0 remains unauthorized until the owner supplies a valid
approval file.

## Why a v2 packet exists

V10.4 Steps 0, 1 and 2 passed. Step 3 stopped because the first registered
diagnostic control, `left_lobe_contact`, reached only 2.579 mm of penetration
across the v1 frozen grid `0.000-0.160 m`, below the registered 5-30 mm band.

Two v1 defects are repaired here, and neither touches production:

1. **Grid too short.** The v1 grid ended at 0.160 m. The audited left-lobe
   contact occurs at 0.175 m. The v2 grid is `0.000-0.200 m` inclusive in
   0.001 m increments (201 points).
2. **Partial assembly translation.** v1 displaced only the target component's
   geom. v2 rigidly translates the complete diagnostic assembly inward along
   y, which is what a physical intrusion would do, and audits every pendant
   component for secondary contact along the full retained path.

A third, non-defect change: the v1 review runner was bound by the Step-0
preflight at `b40e5a0f...` and now hashes `ddf96225...`. That single file is
bridged explicitly rather than silently accepted.

## 1. Preserve and bridge V10.4-v1

- Never overwrite the existing preflight, production, causal, review-stop,
  scene, or trajectory artifacts.
- Contract `pact_place_v104_review_packet_v2`, new output roots:
  - `diagnostics_output/pact_place_v104_review_v2/`
  - `diagnostics_output/pact_place_v104_phase0_v2/`
- Verify both embedded payload hashes and raw file SHA-256 values:

  | input | payload SHA-256 | raw file SHA-256 |
  |---|---|---|
  | preflight | `fe64e285...` | `134f79cf...` |
  | production | `fdcf757b...` | `13707b19...` |
  | causal | `a30c863d...` | `5216216e...` |
  | production scene | - | `01d8adf3...` |
  | scene metadata | - | `7df36c5e...` |

- Distinguish the executed v1 contract (`eb8f1174...`, implementation
  `bd135e68...`) from the later live aggregate (`455379b8...`, implementation
  `bf4af91...`). The executed pair is what the artifacts were produced under
  and is what the bridge binds.
- Scoped provenance bridge: every file listed by the old preflight still
  matches except the superseded review runner
  (`b40e5a0f...` -> `ddf96225...`). That single path is the entire allowlist.
- Fail closed if any production-affecting file, scene, metadata, result,
  trajectory, row binding, config binding, or causal NPZ differs.
- Reconcile all six retained rows as strict-clean and preserve their original
  outcome statistics. No replacement episodes are generated.
- review-v2 and Phase0-v2 live in new modules; the historical v1 runners are
  untouched.

## 2. Certify all three controls before rendering

Deterministic success selection, unchanged:

- production successes: roles 0 left, 3 right, 4 left;
- left-lobe control: role 0, `lobe_0`;
- right-lobe control: role 3, `lobe_1`;
- stem control: role 0, `stem_0`.

For controls:

- Rigidly translate the complete diagnostic assembly inward along y.
- Frozen grid `0.000-0.200 m` inclusive, 0.001 m increments.
- Evaluate every admissible retained frame; the candidate list is never capped.
- Select the first shift whose target component reaches 5-30 mm signed
  penetration.
- Reproduce these audited anchors within 0.1 mm:

  | control | shift | penetration | maximum frame | limiting robot body |
  |---|---|---|---|---|
  | left lobe | 0.175 m | 5.044 mm | 88 | `fr3_link7` |
  | right lobe | 0.132 m | 5.239 mm | 245 | `gripper/base` |
  | stem | 0.083 m | 5.455 mm | 212 | `fr3_link7` |

Before accepting a control:

- Build a complete temporary scene bundle with the required V3/V5 includes and
  a copy of the V10.4 metadata renamed to the diagnostic scene stem.
- Reload the scene through the real task sampler.
- Confirm the assembly is still compiled-static: no joint, freejoint, mocap, or
  runtime bound repair.
- Confirm compiled AABBs/rbounds enclose the diagnostic geometry.
- At the certified frame, require agreement among signed distance, analytic GJK
  intersection, live `data.contact`, and `mounted_fixture` classification.
- Audit the full retained trajectory for every pendant component and record all
  secondary contacts.
- Confirm the production XML remains byte-identical.

All three controls must pass before any MP4 is rendered. Otherwise write a
review-v2 stop artifact and stop with zero published videos.

## 3. Render the review packet atomically

Successes render as complete retained trajectories. Controls render as
true-time contact-centered windows:

- start 45 control frames before the target component's first live contact;
- end 15 frames after its maximum penetration, or one frame before the first
  non-target component contact, whichever comes first;
- always include the maximum-penetration frame.

Expected windows:

| control | inclusive frames | frames |
|---|---|---|
| left lobe | 40-89 | 50 |
| right lobe | 197-260 | 64 |
| stem | 164-227 | 64 |

The left-lobe full trajectory later reaches a 38.15 mm stem contact at frame
90. That is recorded in the certificate and excluded from the left-lobe clip.

Every control retains the rigid assembly and displays:

- `DIAGNOSTIC NEGATIVE CONTROL`
- `TRIMMED CONTACT WINDOW`
- `NOT PRODUCTION GEOMETRY - NOT AN EPISODE`
- source role, global retained frame, target component, shift, signed
  penetration, limiting pair, and contact classification.

All six videos use stride 1 at the 66 ms control period (1000/66 fps), an
untinted wrist view, a third-person view, and a pendant view, with
commanded/realized speed, phase, target-held state, clearance, and contact
overlay.

Render into a temporary directory. Decode every completed MP4 and verify frame
count, FPS, duration, nonzero size, and SHA-256 before atomically publishing
`pact_place_v104_review_v2`. Refuse an existing final directory.

Immutable outputs: `provenance_bridge.json`, `control_certificates.json`,
`review_preflight.json`, `review_manifest.json`, `REVIEW.md`, and exactly six
MP4 files. The manifest reports three production successes, three diagnostic
controls, `eligible_for_human_review: true`, and every authorization field
false. `human_approval.json` remains absent. Return six clickable video paths
and stop.

## 4. Freeze the Phase0-v2 gate before review

Phase0-v2 is implemented and tested before the review packet is generated, so
no code change is needed after approval.

The approval verifier must:

- recompute payload and raw-file hashes instead of trusting embedded hashes;
- validate the provenance bridge, production eligibility, causal pass, control
  certificates, review eligibility, and exact six-video inventory;
- bind the v2 contract, scoped production/runtime hash, review implementation
  hash, gate implementation hash, scene and metadata hashes, all artifact
  hashes, and every MP4 hash;
- reject missing, stale, partial, agent-created, or extra-video approvals;
- create no gate directory or rows before approval passes.

The agent must not create `human_approval.json`. The required owner-authored
schema and exact bindings are documented in `REVIEW.md`.

After a later owner approval, Phase0-v2 may run exactly the frozen 24-row
manifest, with no substitutions, reseeding, threshold changes, or extra rows.
Gate requirements are preserved: at least 20/24 strict-clean, at least 9/12 per
side, zero pendant contacts, complete per-frame clearance at or above 15 mm. A
passing gate sets only `phase0_passed: true`; all downstream authorization
remains false.

Behavioral tests cover payload/raw-hash tampering and the one-file bridge
allowlist; a control whose true worst frame occurs after candidate 64; exact
source-role/component mappings and the 201-point grid; all three audited
anchors; diagnostic metadata discovery through the real task sampler;
compiled-static bounds and live `mounted_fixture` parity; full-path
secondary-contact reporting and trimmed-window exclusion; exact six-video
selection and decoded frame counts; atomic/create-only publication; and
approval rejection for every stale or incomplete binding.

Also re-run the existing 32 V10.4 tests and the previous regression sweep. The
three known stale V3/V5 hash assertions may be reported only after the 61
protected byte hashes reproduce unchanged. Run `git diff --check` in both
repositories. Confirm no production episode, `env.step`, Phase-0 row,
collection, training, or evaluation occurred during the repair.

## Assumptions

- The six existing successful episodes are reused through the scoped provenance
  bridge, per owner choice.
- Diagnostic controls use trimmed contact-centered windows, per owner choice.
- The 0.200 m grid applies only to separately compiled diagnostic scenes; it is
  not a production geometry search.
- V10.4 is qualified through Steps 0-2 but is not Phase0-qualified until owner
  review and the later 24-row gate.

## Measured status

Executed 2026-08-28. All four sections complete; stopped for owner review with
six videos published and Phase 0 unauthorized.

**Provenance bridge — passed.** All three v1 payload hashes recompute to their
audited values (`fe64e285…`, `fdcf757b…`, `a30c863d…`) and all three raw files
match (`134f79cf…`, `13707b19…`, `5216216e…`). Scene and metadata are
byte-identical. Of the 15 files the Step-0 preflight bound, 14 match and exactly
one is bridged — the review runner, `b40e5a0f…` → `ddf96225…`. All six rows
reconcile strict-clean, 3 left / 3 right, zero pendant contact, minimum clearance
0.06158 m, no replacement episode generated.

One semantic correction made while building the bridge: `config_sha256` is a
self-hash computed *before* the key is inserted, so it is neither the raw file
hash nor the payload hash of the file as it now stands. The verifier recomputes
it the way the writer made it.

**Controls — 3/3 certified before any frame was rendered.** Every audited anchor
reproduced exactly:

| control | shift | penetration | max frame | limiting body | window |
|---|---:|---:|---:|---|---|
| left lobe | 0.175 m | 5.044 mm | 88 | `fr3_link7` | 40–89 (50 f) |
| right lobe | 0.132 m | 5.239 mm | 245 | `gripper/base` | 197–260 (64 f) |
| stem | 0.083 m | 5.455 mm | 212 | `fr3_link7` | 164–227 (64 f) |

At the selected left-lobe shift the search measured 289 candidate frames — far
past the 64 v1 would have kept, which is precisely why v1 could not see this
contact. All four instruments agree at each certified frame, every diagnostic
scene reloaded through the real sampler as compiled-static with enclosing bounds,
and the production XML was byte-identical before and after each control.

**Packet — six videos published atomically.** All six decode back at 15.1515 fps
with exact frame counts, matching SHA-256, and nonzero size. `REVIEW.md`
documents the owner approval schema; a test asserts the documented bindings are
exactly what the verifier demands and that the documented record validates as
written.

**Phase0-v2 — frozen, and it refuses.** Run without an approval it exits with
`missing owner approval` and creates no gate directory and no row.

### Corrections to this plan's own figures

The plan states the left-lobe trajectory "reaches a 38.15 mm stem contact at
frame 90". Measurement separates these: the stem first touches at frame **90** at
**0.103 mm**, and only reaches **38.15 mm** at frame **193**. Both numbers are
real; the attribution to one frame was not. Frame 90 is what trims the clip at
89; the certificate records both as `secondary_summary`.

The plan's binding list did not include `v1_contract_sha256`, and an early draft
of the verifier bound it. It was removed: it is the *live* contract aggregate,
which drifts whenever unrelated implementation files change, and the meaningful
value — the contract the artifacts were executed under — is already bound as
`executed_v1_contract_sha256`.

### Preserved superseded artifact

`diagnostics_output/pact_place_v104_review_v2_superseded_01_incomplete_approval_schema/`
is a complete, correct packet from the first publication whose `REVIEW.md`
omitted `review_manifest_sha256` and the six per-video bindings that the verifier
requires — an owner following it verbatim would have been rejected for a missing
binding. It is preserved rather than deleted, in the same spirit as the V10.4-v1
preflight attempts. The published packet was regenerated by the corrected, frozen
code; no control result, shift, penetration, frame, or window changed between the
two.

### Test results

71 review-v2 tests and 32 V10.4 tests pass. The `pact_place` sweep is 387 passed
/ 3 failed, those three being the known stale V3/V5 hash assertions, reported
after the 61/61 protected byte hashes reproduced unchanged. The full `tests/`
sweep is 1258 passed / 18 failed / 1 skipped; the 18 are those three plus 15
pre-existing failures in unrelated areas. `git diff --check` is clean in both
repositories. This repair modified no existing source file — every change is a
new file — so no pre-existing failure is attributable to it.

### Not done

No production episode, `env.step`, Phase-0 row, collection, conversion, training,
or learned-policy evaluation occurred. `human_approval.json` was not created.
Every authorization field is false; only the review manifest carries
`eligible_for_human_review: true`.
