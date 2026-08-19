# Attempt-3 place-corridor replay crib

These clips restore the recorded full-model `qpos` and render. They are
not a re-run: physics is not stepped, the expert is not executed, and the
Phase-0 FAIL records are unchanged. Step 0 after `mj_forward` matches the
recording at 1e-6 m (qpos indexing). Later frames can drift by tens of
microns because xpos/TCP are derived after `mj_forward`, not `mj_step`.

Directory: `/root/prox_learning_pact_remediation/diagnostics_output/pact_place_corridor_v5_videos`
Playback: **1×** of the 66 ms control period (`fps = 15.1515`).
Layout: wrist (left) and the existing render-only third-person camera (right).

Named failure clips:

- `row03_FAIL_outbound_approach_ik_cascade.mp4`
- `row10_FAIL_lift_cup_dropped.mp4`

Clean successes are `rowXX_clean_success.mp4`.

## Watch the six failures against a stated claim

| Row | Steps | Phase | Recorded | Watch for |
|---:|---:|---|---|---|
| 10 | 160 | `lift` | never reached 1 cm (cup max z 0.8004 vs 0.7937 start); cup ended z 0.7527, BELOW its start; empty_gripper branch fired | Grasp closes but the cup is knocked down rather than lifted. Does it get pushed off the shelf? |
| 3 | 238 | `outbound_approach` | lifted 1 cm and carried, then 8 sequential IK failures; stalled at y 0.1067 reaching for y 0.1314 (2.5 cm short laterally) | The genuine reach limit. The arm freezes mid-bow with the cup still held and nothing blocking it. |

Row 9 is the one worth the most attention: `outbound_approach` was subdivided
into ~4 cm Cartesian pieces for attempt 3 and the class still appeared. If the
video shows the arm climbing anyway, the remaining candidate is IK branch
continuity across the bow — the plan's second-ranked repair, never applied.

v2 hazard rows 6 and 12 have no `trajectory.json` and are out of scope.
