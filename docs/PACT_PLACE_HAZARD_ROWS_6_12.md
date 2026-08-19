# v2 rows 6 and 12: hazard contact at the accepted initial observation

This is the Step 1 diagnostic for Phase 0 attempt 3. It re-ran **only** rows 6 and 12 of
the frozen v2 contract (`config_sha256`
`a0f30725e325a73b5584895a07fa18000fe3645cb63ebd1b4e5a6746bc201c31`) with
`PACT_CONTACT_AUDIT_SUMMARY_ONLY=0`. No expert, sampler, threshold, or gate number was
changed. Both v1 and v2 `PACT_PLACE_CORRIDOR_PHASE0_FAIL` records stand.

**Outcome 1.** The robot is against the panel at the accepted initial observation, before
commanded arm motion. Route A exists as a principled pre-boundary resample. This is not
inbound scraping by the frozen corridor pick expert.

## What was re-run

| Item | Value |
|---|---|
| Config | `configs/pact_place_corridor_v2.json` |
| Rows | 6 and 12 only |
| Output | `diagnostics_output/pact_place_corridor_v2_hazard_rows_6_12_frames/` |
| Role | `diagnostic_not_a_gate` |
| Sampling retries | 0 / 0 (same first-draw seeds as the v2 gate) |
| Videos | not recorded — Phase 0 still strips RGB; contact frames answer the question |

## Bodies, timing, penetration

Both episodes are the same geometry of overlap.

| | Row 6 | Row 12 |
|---|---:|---:|
| Task / clean | success / not clean | success / not clean |
| Hazard frames | 82 | 171 |
| First / last control step | 0 … 3 | 0 … 6 |
| First / last sim time | 0.360 … 0.530 s | 0.360 … 0.722 s |
| Max penetration | 5.54 mm, **on the first frame** | 20.46 mm, **on the first frame** |
| Pair | `pact_intrusion_right_g` vs `robot_0/fr3_link7_collision` | same |
| Roots | `pact_intrusion_right` / `robot_0/base` | same |
| Policy phases | `gripper-open` then `pregrasp` | same |
| Traversal bucket | inbound 82 / outbound 0 | inbound 171 / outbound 0 |
| First `grasp_target` contact | step 143 | step 135 |
| Hazard after last listed frame | none | none |

`place_receptacle` is classified first, and `hazard_bar` requires `pact_intrusion_`. The
pair is a robot link against the panel body. The tray cannot be misclassified as this.

The first policy observation is at `sim_time = 0.360 s` in both rows. That clock value is
the post-reset settle time (identical across instances), not motion into the corridor.
`gripper-open` is the first primitive; it commands the gripper, not a TCP translation.
Penetration is already maximal at that observation, then falls as the contact solver
separates `fr3_link7` from the panel. Contact is gone before the arm has reached the cup,
and it does not return for the rest of the ~400–500 step pick-and-place.

## Which of the three outcomes this is

1. **The robot is against the panel at reset — yes.** The scientific boundary wrote
   `initial_observation_accepted.json` on a state that already had robot–panel contact.
   20.5 mm is a packed overlap, not a one-step flicker. The fix is to treat that contact
   as a pre-boundary sampling failure and draw another seed, the same way IK-infeasible
   resets are already retried.
2. **Real contact during the inbound approach — no.** After t = 0.53 s / 0.72 s there are
   zero hazard frames. The inbound path is still
   `PactCollisionCorridorPolicy._compute_trajectory()`, and it does not scrape the panel
   on these episodes. Attempt 3 does **not** pause Phase 0 for a frozen-expert finding.
3. **A transient initialization overlap that resolves immediately — related, not the
   label.** Physics does peel the overlap in ~0.2–0.4 s. That is how an illegal initial
   state presents, not a reason to drop the frames. Silently excluding them would hide
   the acceptance bug.

## What this does not license

- Do not raise `tcp_pos_err_threshold`, widen `max_sequential_ik_failures`, or filter
  grasps.
- Do not reintroduce the planning-time bow IK check.
- Do not retune inbound bow constants. They are unused on this class; inbound is the
  frozen pick expert, and these rows never contacted the panel on that path.
