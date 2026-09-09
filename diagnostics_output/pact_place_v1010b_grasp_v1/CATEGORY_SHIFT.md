# Restricted replay comparison: PACT3105, scene0f25bb…

The original remains `touched_without_hold`; the new replay remains `held_without_lift`. Both fail the unchanged final task predicate. Neither label is replaced by the other.

| Measurement | Retained original | New instrumented replay |
|---|---:|---:|
| First close command | 153 | 152 |
| Raw gripper score at152 (threshold127.5) | 126.079 | 129.297 |
| Observations labeled held | None | 153 only |
| Maximum cup-origin lift | 5.080mm | 5.492mm |
| Longest continuous bilateral contact | 0.872s | 0.980s |

At observation153 the new cup rises1.634mm and has exactly one recorded contact, with `left_pad2`. The unchanged simulator heuristic defines held as touching a gripper while touching no non-gripper geometry; it requires neither sustained engagement nor a minimum lift. The raw contact reconstruction reproduces all50 held/touching observations from130 through179 exactly. The single held flag therefore does not establish a stable grasp.

Both original and new first-query actions reconstruct exactly from their saved inputs, with identical qpos/proximity and tolerated RGB differences. This establishes the action-difference onset. Later closed-loop differences are not individually attributed because later historical images are unavailable; the one-step closure change and instantaneous contact-label change are directly observed.

This pair remains in the12-replay denominator and its new geometry remains descriptive. It is **ineligible for original-category equivalence and contributes no Stage A supporting-case count**. It is outside cross24. The other required3103 and3104 failure cases retain their original categories and supply the requested-versus-achieved discrepancy evidence.

The frozen case definition and exact source hashes are in [replay_scope_resolutions.json](replay_scope_resolutions.json). Raw sensor reconstruction is in [category_shift_0f25bb_contact_reconstruction.json](category_shift_0f25bb_contact_reconstruction.json). Prior stop reports and source versions remain under `amendments/03_single_held_observation_scope/prior/`.
