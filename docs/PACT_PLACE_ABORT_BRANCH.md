# Diagnostic: which `check_failure` branch aborted the eight suspect rows?

**This is a diagnostic, not a gate.** No expert behaviour, threshold, or gate record
was changed. Artifacts are marked `role: diagnostic_not_a_gate` and
`authorizes_collection: false`. No fix was applied.

**Outcome 1.** Branch 1 (`empty_gripper`) fired on all eight reproducing rows
while the cup was still in the pads. Position and rotation tracking were inside
tolerance. IK was 0.

## Step 0 — Cup_10 wall thickness is not 2 mm

The empty-gripper trip point is `inter_finger_dist_range[0] + 0.002` = **2.0 mm**
(Franka range `[0.0, 0.08]`). Cup_10's collision wall at the eight recorded
`adjusted_grasp_object_local_position_m` points is **8.4–10.7 mm**, measured by
raycasting the cup collision geoms along the local XZ radial through each grasp
point.

| Screen | Row | Grasp radius (mm) | Inner (mm) | Outer (mm) | Wall (mm) | vs 2 mm trip |
|---|---:|---:|---:|---:|---:|---|
| v1 | 13, 16 | 36.3 | 31.5 | 41.8 | 10.3 | 5.2× |
| v1 | 21, 22 | 36.7 | 32.7 | 43.3 | 10.7 | 5.3× |
| v2 | 3 | 33.3 | 27.0 | 35.5 | 8.4 | 4.2× |
| v3 | 2 | 33.4 | 27.1 | 35.6 | 8.5 | 4.2× |
| v3 | 6 | 36.3 | 31.5 | 41.8 | 10.3 | 5.2× |
| v3 | 21 | 33.3 | 27.0 | 35.5 | 8.4 | 4.2× |

A correct pinch of this cup is **not** geometrically indistinguishable from an
empty gripper. During holding, every one of these episodes sat at
**~8.3–8.8 mm** finger separation — matching the wall — until a single control
step reported `inter_finger_dist = 0.0`.

## Reproduction

Telemetry was added read-only. Each row was re-run under the producing tree
(v1 parent `8459489` / submodule `b00dc35`; v2 parent `9fb0406` / submodule
`2828751`; v3 current dirty `enclosure_reach.py` matching
`3155ac3e…` plus the telemetry overlay) and matched the frozen row exactly on
`task_success`, `clean_success`, `terminal_policy_phase`,
`terminal_action_index`, and `position_error_m`. **11/11 reproduced. 0 discarded.**

## What fired

`TCPMoveSequence.check_failure` order: empty gripper, then `pos_err > 0.1 m`,
then `rot_err > 30°`. All eight suspects:

| Screen | Row | Phase | Branch | Finger (mm) | Trip (mm) | posErr (mm) | rot (°) | Pad–cup contact | On tray |
|---|---:|---|---|---:|---:|---:|---:|---|---|
| v1 | 13 | `placement_descent` | `empty_gripper` | 0.00 | 2.0 | 10.7 | 1.54 | yes | yes |
| v1 | 16 | `placement_descent` | `empty_gripper` | 0.00 | 2.0 | 8.7 | 1.30 | yes | yes |
| v1 | 21 | `placement_descent` | `empty_gripper` | 0.00 | 2.0 | 4.7 | 0.77 | yes | yes |
| v1 | 22 | `placement_descent` | `empty_gripper` | 0.00 | 2.0 | 5.5 | 0.93 | yes | yes |
| v2 | 3 | `outbound_pass` | `empty_gripper` | 0.00 | 2.0 | 6.4 | 0.27 | yes | no |
| v3 | 2 | `outbound_approach` | `empty_gripper` | 0.00 | 2.0 | 6.6 | 0.32 | yes | no |
| v3 | 6 | `placement_descent` | `empty_gripper` | 0.00 | 2.0 | 5.6 | 0.86 | yes | yes |
| v3 | 21 | `outbound_pass` | `empty_gripper` | 0.00 | 2.0 | 6.6 | 0.30 | yes | no |

No row hit `pos_err`, `rot_err`, `ik_cascade`, or `sequence_complete`.
`gripper_width_min_m = 0` on the frozen rows is uninformative: the episode
starts with a closed gripper, so the minimum is always 0.

The empty predicate is armed because these rows die inside the holding
`TCPMoveSequence` (`action_index = 4`), which includes outbound *and*
`placement_descent`.

## Controls (v3 rows 5, 15, 17)

All three succeeded, reproduced, and **never** satisfied `empty_gripper` while
`is_holding_object` was true. Holding width stayed at 7.8–9.4 mm through
`placement_descent`. After the release primitive, holding is false, so the
empty check is not armed. The race “dip after release” did not occur.

## How to read this, per the three named outcomes

1. **Branch 1 with the cup held — this is what happened.** The object is in the
   pads at abort (negative pad–cup distance; TCP–object offset unchanged on the
   abort step). Two different follow-ons, both true:
   - **`placement_descent` (v1 13/16/21/22, v3 6):** category error. The next
     primitive is the intended release. The empty-gripper check is testing for
     a drop on a segment whose job is to put the cup down. The cup is already
     supported by the receptacle.
   - **Transport (v2 3, v3 2, v3 21):** the check is right in principle — a
     real drop during carry should abort — but the *measurement* is a false
     empty. Finger joints sit at ~8.5 mm for tens to hundreds of steps, then
     one control step reports 0.0 mm while the pads are still on the cup.
     That is not “the 2 mm threshold is too tight for this wall.” The wall is
     8–10 mm; the pinch was already at 8.5 mm, well above 2 mm.
2. **Branch 3 (`rot_err`) — did not fire.** Peak abort rotation among the eight
   is 1.54°, against a 30° tolerance.
3. **Something else — no.** Elimination on v3 rows 2 and 21 was correct.

A row is counted as “cup held” here only because the pads are still in contact
with Cup_10 at the abort instant, not because counting it that way would move
a gate number.

## What this does not license

- Do not change `gripper_empty_threshold`, `tcp_pos_err_threshold`,
  `tcp_rot_err_threshold`, `max_sequential_ik_failures`, or
  `MIN_CLEAN_SUCCESSES`.
- Do not edit the expert trajectory, grasp selection, or `RELEASE_CLEARANCE_M`.
- Do not rewrite the three `PACT_PLACE_CORRIDOR_PHASE0_FAIL` records. If the
  eight were later treated as non-failures, the arithmetic would be v1 22/24,
  v2 21/24, v3 21/24 — that is a reason the measurement had to be careful, not
  a decision this diagnostic is allowed to make.

## Artifacts

- `diagnostics_output/pact_place_abort_branch/role.json`
- `diagnostics_output/pact_place_abort_branch/analysis.json`
- `diagnostics_output/pact_place_abort_branch/wall_thickness.json`
- Per-row `abort_branch_telemetry.json` next to the diagnostic `result.json`
  (new output root; frozen Phase-0 `result.json` files were not touched)

`terminal_tracking` now records `check_failure_branch`, both tracking errors,
`inter_finger_dist_m`, and the trip point. The 12 protected frozen artifacts
were hash-verified 12/12 after the run.
