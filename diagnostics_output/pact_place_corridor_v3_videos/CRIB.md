# Attempt-3 place-corridor replay crib

These clips restore the recorded full-model `qpos` and render. They are
not a re-run: physics is not stepped, the expert is not executed, and the
Phase-0 FAIL records are unchanged. Step 0 after `mj_forward` matches the
recording at 1e-6 m (qpos indexing). Later frames can drift by tens of
microns because xpos/TCP are derived after `mj_forward`, not `mj_step`.

Directory: `/root/prox_learning_pact_remediation/diagnostics_output/pact_place_corridor_v3_videos`
Playback: **1×** of the 66 ms control period (`fps = 15.1515`).
Layout: wrist (left) and the existing render-only third-person camera (right).

Named failure clips:

- `row02_FAIL_outbound_approach_cup_lost.mp4`
- `row06_FAIL_placement_descent_not_released.mp4`
- `row09_FAIL_outbound_approach_tracking_12.6cm.mp4`
- `row12_FAIL_pregrasp_never_grasped.mp4`
- `row20_FAIL_lift_cup_dropped.mp4`
- `row21_FAIL_outbound_pass_cup_lost.mp4`

Clean successes are `rowXX_clean_success.mp4`.

## Watch the six failures against a stated claim

| Row | Steps | Phase | Recorded | Watch for |
|---:|---:|---|---|---|
| 12 | 14 | `pregrasp` | gripper stayed open 84.0 mm; tracking error 10.5 cm | Dies in under half a second. Does the arm even start toward the cup? |
| 20 | 133 | `lift` | cup ended z 0.753 vs 0.794 start; gripper emptied | Grasp closes but slips — does the cup get knocked off the shelf? |
| 2 | 193 | `outbound_approach` | lifted 1 cm then gripper → 0.000; tracking error only 9.5 mm | Cup lost while tracking fine. Where does it slip out? |
| 21 | 281 | `outbound_pass` | gripper → 0.000; cup abandoned at x 0.72 | Same class as row 2 but later — does it clip the aperture? |
| 9 | 234 | `outbound_approach` | tracking error **12.58 cm**; actual z 0.9866 vs target 0.9062, plus a 9.4 cm y shortfall | The claim is the arm climbs out of the carry plane. Does it visibly lift? |
| 6 | 388 | `placement_descent` | cup on the tray, `supported_by_receptacle` true, `robot_contact` still true, place error 20.9 mm | The release defect recurring once. Is the cup actually placed and the gripper just still on it? |

Row 9 is the one worth the most attention: `outbound_approach` was subdivided
into ~4 cm Cartesian pieces for attempt 3 and the class still appeared. If the
video shows the arm climbing anyway, the remaining candidate is IK branch
continuity across the bow — the plan's second-ranked repair, never applied.

v2 hazard rows 6 and 12 have no `trajectory.json` and are out of scope.
