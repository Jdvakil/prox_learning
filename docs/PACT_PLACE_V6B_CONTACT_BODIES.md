# v6b Step 0: which body hits the clutter

Diagnostic only. `role: diagnostic_not_a_gate`. Authorizes nothing.
Re-ran set-13 rows **0** (left) and **4** (right) with
`PACT_CONTACT_AUDIT_SUMMARY_ONLY=0`. The v6 stop record is unchanged.

## Decision for A0b

The contacting body is the **carried cup**, not a robot link. Contact is
lateral: the cup occupies the outbound bow at about `|y| = 0.19` and strikes
the opposite-side inner box (`r0` on left intrusion, `l0` on right). Inner
faces at 0.19 overlap that sweep. Move clutter **laterally** onto
`|y| ∈ {0.32, 0.36, 0.40}`. Do not shrink the boxes. Height stays
`{0.06, 0.10}` unless that grid also fails.

TCP z stays above the 0.78 m box tops; the cup hangs below the gripper and
skims. That is why a TCP-only envelope understated the hit.

## Rows

| | Row 0 | Row 4 |
|---|---|---|
| Intrusion | left | right |
| Task / clean | success / not clean | success / not clean |
| Clutter pairs / frames | 2497 / 1990 | 2486 / 2406 |
| First–last step | 224–290 | 239–315 |
| Traversal | outbound only | outbound only |
| Policy phases | `outbound_approach` then `outbound_pass` | same |
| Pair | `Cup_10` vs `pact_clutter_r0_g` | `Cup_10` vs `pact_clutter_l0_g` |
| Robot-link pairs | **0** | **0** |
| Max penetration | 1.32 mm at step 271 | 0.55 mm at step 265 |

Output: `diagnostics_output/pact_place_corridor_v6b_contact_bodies/`.
The retained `contact_frames` payloads are large and are not committed.
