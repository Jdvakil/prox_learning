# Attempt-6c place-corridor replay crib

These clips restore the recorded full-model `qpos` and render. They are
not a re-run: physics is not stepped, the expert is not executed, and the
Phase-0 PASS record is unchanged. Step 0 after `mj_forward` matches the
recording at 1e-6 m (qpos indexing). Later frames can drift by tens of
microns because xpos/TCP are derived after `mj_forward`, not `mj_step`.

The renderer refuses to run unless the scene is `pact_place_corridor_v3.xml`
and every `pact_clutter_*` body matches the frozen config plus row jitter.

Directory: `/root/prox_learning_pact_remediation/diagnostics_output/pact_place_corridor_v6c_videos`
Playback: **1×** of the 66 ms control period (`fps = 15.1515`).
Layout: wrist (left) and the existing render-only third-person camera (right).

Named failure clips:

- `row07_FAIL_outbound_approach_ik_cascade.mp4`

Clean successes are `rowXX_clean_success.mp4`.

## Watch the one failure against a stated claim

| Row | Steps | Phase | Recorded | Watch for |
|---:|---:|---|---|---|
| 7 | 185 | `outbound_approach` | lifted and carried, then 8 sequential IK failures; cup still held | Arm freezes mid-bow. Clutter is not in contact on this row. |

The one official failure has no `pact_clutter` contact. Row 7 is the same
`outbound_approach` / `ik_cascade` class as v5 row 3 and v6b rows 9/22.
The number that matters is 28 mm: the gap between the carried cup and the
nearest box at closest approach, not the 26 cm it looks like on screen.
