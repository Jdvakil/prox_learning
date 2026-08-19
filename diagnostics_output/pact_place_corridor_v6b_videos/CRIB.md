# Attempt-6b place-corridor replay crib

These clips restore the recorded full-model `qpos` and render. They are
not a re-run: physics is not stepped, the expert is not executed, and the
Phase-0 PASS record is unchanged. Step 0 after `mj_forward` matches the
recording at 1e-6 m (qpos indexing). Later frames can drift by tens of
microns because xpos/TCP are derived after `mj_forward`, not `mj_step`.

The renderer refuses to run unless the scene is `pact_place_corridor_v3.xml`
and every `pact_clutter_*` body matches the frozen config plus row jitter.

Directory: `/root/prox_learning_pact_remediation/diagnostics_output/pact_place_corridor_v6b_videos`
Playback: **1×** of the 66 ms control period (`fps = 15.1515`).
Layout: wrist (left) and the existing render-only third-person camera (right).

Named failure clips:

- `row02_FAIL_lift_cup_dropped.mp4`
- `row09_FAIL_outbound_approach_ik_cascade.mp4`
- `row14_FAIL_lift_cup_dropped.mp4`
- `row22_FAIL_outbound_approach_ik_cascade.mp4`

Clean successes are `rowXX_clean_success.mp4`.

## Watch the four failures against a stated claim

| Row | Steps | Phase | Recorded | Watch for |
|---:|---:|---|---|---|
| 2 | 139 | `lift` | empty_gripper on lift; gripper_width_min_m = 0; cup max z 0.8018 vs 0.7937 start; ended z 0.7527 | Grasp closes but the cup is knocked down rather than lifted. |
| 9 | 247 | `outbound_approach` | lifted and carried, then 8 sequential IK failures; cup still held | Arm freezes mid-bow. Clutter is not in contact on this row. |
| 14 | 144 | `lift` | empty_gripper on lift; gripper_width_min_m = 0; cup max z 0.7995 vs 0.7937 start; ended z 0.7527 | Same cup-drop as row 2. Clutter is not in contact. |
| 22 | 204 | `outbound_approach` | lifted and carried, then 8 sequential IK failures; cup still held | Same reach/IK class as row 9 and v5 row 3. |

None of the four official failures has `pact_clutter` contact. Rows 2 and 14
are the v5 row-10 cup-drop class. Rows 9 and 22 are the v5 row-3 IK-cascade
class. The clutter resite did not create a new failure mode on this seed.
