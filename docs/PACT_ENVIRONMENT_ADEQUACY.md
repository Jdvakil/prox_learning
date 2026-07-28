# PACT environment adequacy

## Status

Phase 1 is adjudicated. The frozen expert/surface pilot failed its expert
collision-free solvability guard: ordinary task success was 20/24, but
collision-free task success was 19/24 against a minimum of 20/24. The
surface-observability guards passed. Per the preregistration, the experiment
stopped before dataset conversion, pilot ACT training, full collection,
front-end training, or ACT/PACT policy training.

The machine-readable preregistration is
`configs/pact_collision_environment_v1.json`. The deterministic candidate
manifest is `configs/pact_collision_candidate_manifest_v1.json`.

## Deliberate route choice

This experiment takes the **collision route**.

The recorded fridge result rules out treating these sensors as useful
grasp-object localizers in that scene: there were no target sightings inside
20 cm and none at grasp. In contrast, the skin cameras do measure nearby
surfaces. The primary endpoint is collision-free task success, so this
environment tests surface geometry directly rather than assigning the modality
an unsupported object-position job. Gate A's object-signal criterion is
therefore not applicable. A collision-route surface-observability guard is
added in its place.

## Environment

`pact_collision_corridor_v1` retains the fumehood pick task and
removes the exocentric RGB camera. Both learned policies receive wrist RGB and
9-D Franka proprioception. PACT additionally receives all 40 skin streams.

One wall-colored matte panel enters the wrist/link-6 passage from either the
left or the right. Its nominal center is `(0.615, +/-0.340, 0.89)` m with half
extents `(0.055, 0.240, 0.090)` m. Its side is exactly balanced in every
manifest role. The aperture width is fixed at 0.85 m, leaving the opposite-side
expert bow physically open. Small x and inner-face jitters are committed per row. The red
grasp-validated `Cup_10` target has a 7.0 by 7.3 cm cross-section, safely
inside the Franka's 8.5 cm finger span; it is independently sampled near
`(0.76, y, shelf_top)`, with
`y in [-0.04, 0.04]` m. Thus target pixels do not disclose the required
left-versus-right avoidance direction. The robot-base forward placement is
0.14 m, keeping the target in the same proven relative reach envelope as the
parent fumehood pick. A privileged expert reads the panel geometry and bows
away from it with a 0.08 m nominal surface-clearance margin.
The sash aperture is fixed at 0.70 m so incidental link-5/sash contact cannot
contaminate the panel-avoidance endpoint.

The design targets the wrist/link-6 swept volume rather than the grasp target.
This is the intended whole-body necessity: the wrist camera can keep the target
visible while a lateral arm intrusion remains outside its useful view.
Opposite intrusion sides make a single memorized safe detour impossible.

### Frozen development screen

The final eight reserved development rows were run from manifest
`418e2b919c620c66422d3b3b3da2afdfc21fb245ba8a7b60a66f5464ba1063d4`
at root commit `9a79e16` and MolmoSpaces commit `84c54a9`. The privileged
expert achieved 8/8 ordinary and collision-free task success with zero
non-target contact entries. All 8/8 rows saw the active panel before grasp:
533/982 frames (54.3%) had a panel return inside 20 cm and 229/982 (23.3%)
inside 12 cm.

As a causal geometry control, the avoidance bow was disabled on development
row 5 without changing the task seed or scene. The straight path still
completed the task but produced 192 `hazard_bar` contact entries across 164
frames. These are development measurements, not the Phase 1 gate result. They
justify freezing the candidate and starting the independently seeded pilot.
The machine-readable record is
`diagnostics_output/pact_vs_act/development_screen.json`.

## Endpoint and contact taxonomy

The primary endpoint is exactly:

`task_success AND hazard_bar == 0 AND other_environment == 0`

Only robot-to-environment contacts at non-positive MuJoCo distance are counted.
The audit runs after every 2 ms control/physics step, as well as at episode
boundaries, so contacts that begin and resolve inside the 66 ms policy interval
are retained. Robot self-contact and floor contact are excluded. Every counted
pair is partitioned into:

- `grasp_target`: any pair rooted at `cavity_obj_`; recorded and exempt;
- `hazard_bar`: either `pact_intrusion_left` or `pact_intrusion_right`;
- `other_environment`: every other non-floor scene contact.

Counts are reported both as contact-pair entries and frames with contact.
Ordinary task success is a separate secondary endpoint.

## Frozen Phase 1 protocol

Eight `development` rows are reserved for render, kinematic, and expert smoke
tests. They may cause a newly versioned scene. Once the first `pilot_train` row
starts, the scene and this gate freeze.

Each manifest row has one initial construction seed and four deterministic
retry seeds. A retry is allowed only when task sampling or the initial
`task.reset()`/expert trajectory construction fails before an initial
observation is accepted and before any action. Every such failure and seed is
retained. Immediately after a successful initial reset the row becomes
outcome-bearing: no later exception, contact, task failure, or success can
cause a replacement or rerun.

The pilot contains 24 expert `pilot_train` rows and 24 independent
`pilot_eval` instances. A wrist-RGB-plus-proprio ACT baseline is trained on only
the collision-free successful expert rows, in frozen role order, with an 80/20
train/validation split and seed 1101. It uses the requested ACT recipe and the
minimum-validation-loss checkpoint from the fixed 2000-epoch run. Each pilot
evaluation row is run once in a fresh subprocess.

All of the following must pass:

| Guard or gate | Frozen pass condition |
|---|---|
| Expert solvability | At least 20/24 expert rows achieve ordinary success and collision-free success |
| Surface observability | Active intrusion is seen by a skin ray inside 20 cm in at least 30% of pre-grasp frames, inside 12 cm in at least 5%, and in at least 20/24 episodes |
| Gate A | Not applicable: collision route |
| Gate B | ACT collision-free task success is 8–16/24, inclusive; ordinary task success is at least 12/24 |
| Gate C | ACT has any non-target contact in at least 6/24 rows and specifically contacts the intrusion in at least 6/24 |

“Pre-grasp” ends at the earlier of the first grasp-target contact or first
closing command. An intrusion sighting is a valid depth ray whose reconstructed
world point lies inside the active panel AABB with 1 cm tolerance. This prevents
nearby fumehood walls or robot self-returns from being misreported as panel
signal.

Failure of any applicable row stops the experiment before full collection or
policy training. No threshold may move after pilot outcomes are visible.

## Frozen front-end target

The collision route uses nearest-surface geometry, not object position. The
weight-shared frozen encoder consumes 32 causal 8x8 depth frames per sensor
(eight control steps by four subframes) and predicts the nearest valid surface
point in that sensor's local frame plus validity. Far-plane and missing-depth
sentinels are excluded from regression targets and encoded as invalid.

Before policy training, held-out mean and median Euclidean error, within-2-cm
rate, validity precision, and validity recall will be reported. The encoder is
then frozen. The converted ACT dataset retains the raw 40-sensor stream, ordered
sensor names, intrinsics, and per-timestep world-to-sensor extrinsics.

## Measurements

| Quantity | Result |
|---|---:|
| Expert terminal-row ledger | 20 success, 1 sampling failure, 3 infrastructure failures |
| Expert ordinary task success | 20/24 (pass; minimum 20) |
| Expert collision-free task success | **19/24 (fail; minimum 20)** |
| Pre-grasp frames with intrusion inside 20 cm | 1284/2495 (51.5%; pass) |
| Pre-grasp frames with intrusion inside 12 cm | 492/2495 (19.7%; pass) |
| Episodes with an intrusion sighting | 20/24 (pass) |
| Pilot ACT ordinary task success | Not run: failed expert prerequisite |
| Pilot ACT collision-free task success | Not run: failed expert prerequisite |
| Pilot ACT rows with any non-target contact | Not run: failed expert prerequisite |
| Pilot ACT rows with intrusion contact | Not run: failed expert prerequisite |

Pilot expert row 21 completed the task but recorded 58 `hazard_bar`
contact-pair entries across 58 contact frames; it had no
`other_environment` contact. Four other rows were terminal construction/IK
failures and count as failures without replacement. Thus the 24-row ledger is
complete, but the expert collision-free guard misses its threshold by one.

## Decision

The environment is not adequate under the jointly frozen criteria. Although
the proximity modality carried strong panel signal, the task was not robustly
solvable enough by the expert at the primary endpoint. Gates B and C were not
run because any failed applicable prerequisite mandates a stop before policy
training. The machine-readable adjudication is
`diagnostics_output/pact_vs_act/environment_gate.json`.

PACT_ENVIRONMENT_INADEQUATE
