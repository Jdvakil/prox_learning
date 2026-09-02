# V10.11 — two randomized near-target household objects

**Status:** proposed, not implemented. V10.7–V10.10 and all of their artifacts
remain immutable. V10.11 is a new sampler/config/data lineage.

## Objective

Start from the successful V10.10 four-object environment and activate two more
household objects. The two additions must:

- be visibly near the target cup and randomized per episode;
- remain ordinary movable clutter, not fixtures or policy-controlled objects;
- start settled, contained, and free of contact with the target or other clutter;
- avoid blocking the scripted grasp and placement route.

They do **not** have to force avoidance, enter a proximity-risk band, or produce
a measurable PACT advantage. This version tests a modest increase in nearby
visual/physical clutter, not a new hazard mechanism.

## Corrected frozen facts

- V10.10 compiles the complete eight-object V9.5 palette on every task. It
  activates slots `01,03,04,06` and parks `00,02,05,07`. Therefore V10.11 can
  reactivate two existing slots without adding assets or changing the runtime
  model topology.
- The settled `Cup_10` collision box in its task orientation is approximately
  `70.47 x 72.97 x 145.74 mm`. The earlier 7.3 cm figure was its footprint, not
  its height.
- The target is not fixed at `(0.75, 0.02)`. The inherited distribution is
  equivalent to:

  ```text
  x = 0.760 + Uniform(-0.015, +0.015) m
  y = Uniform(-0.040, +0.040) + Uniform(-0.015, +0.015) m
  ```

- The expert route is side-dependent: left intrusion rows bow toward `-y` and
  right intrusion rows bow toward `+y`. No fixed `+y` region is globally safe.
- The verified V10.10 collection yield was `144 / 313 = 46.0%`.

The three V10.7 source XML files and their byte hashes remain unchanged. That
does not let V10.11 inherit a clearance/qualification result automatically;
the new object poses still require their own preflight.

### Geometry freeze — non-negotiable

Every existing physical dimension is read-only. V10.11 may translate and yaw
the two reactivated free bodies, but it must not resize or rescale the cup, mug,
can, existing clutter, pendant, panel, tray, enclosure, robot, collision proxy,
mesh, or sensor geometry. The measurements written in this plan are descriptive
inputs for separation checks only; they are not proposed geometry edits.

No V10.11 path may write `model.geom_size`, alter mesh scale, change asset
metadata, or generate a modified scene XML. Preflight must hash the protected
geometry inputs before and after execution and fail on any difference.

## Object choice

Reactivate the following existing parked palette entries:

| slot | object | compiled collision dimensions | reason |
|---|---|---:|---|
| `02` | `Mug_2` | `109.15 x 85.85 x 103.64 mm` | tall, stable V9.5 decor, shorter than the cup |
| `05` | can `663b5edc92a543668c1b602981e724a4` | `75.29 x 75.28 x 115.44 mm` | tall, compact, stable V9.5 decor, shorter than the cup |

Do not use `Potato_1` or `Potato_26`. They passed a short settling probe but
were later observed rolling without robot contact in the V9.1 live smoke.

V10.11 therefore keeps the same eight compiled palette entries:

```text
active:   01, 02, 03, 04, 05, 06   (six movable clutter objects)
parked:   00, 07                    (two objects)
```

The two route-bearing soap bottles and both V10.10 plates remain active.

## Registered random placement

The mug and can are sampled relative to the episode's pre-sampled target XY.
Their intended region is behind the target in `+x`, which is away from the
front-side grasp approach for both intrusion sides.

### Deterministic random source

Derive independent local RNG seeds from the frozen row seed and labels
`v1011_target`, `v1011_slot_02`, and `v1011_slot_05`. Do not consume the global
RNG for placement retries. Reconstructing the same row must reproduce the
target XY, both placements, and every rejection count exactly.

Pre-sample the full inherited target XY distribution shown above. In V10.11
only, make `_obj_rest()` return that recorded XY and set the inherited second
XY jitter to zero; otherwise the target would be randomized a second time after
the near-target placements had already been chosen. Initial compilation may use
the nominal rest pose before the row-specific layout exists.

### Annular sector

For each new object, sample uniformly by **area** in the target-relative sector:

```text
angle:  -65 to +65 degrees about world +x
r_max:  0.220 m
r_min:  target XY collision circumscribed radius
        + object XY collision circumscribed radius
        + 0.020 m separation margin
```

Using the measured collision boxes, the nominal lower bounds are approximately
`0.1402 m` for the mug and `0.1240 m` for the can. Compute them from the frozen
dimensions in code; do not hardcode rounded values as the source of truth.

Use `r = sqrt(r_min^2 + u * (r_max^2 - r_min^2))` so the distribution is uniform
over area rather than biased toward the inner edge.

The planned collision-center height is always:

```text
center_z = SHELF_TOP_Z (0.720 m) + object_collision_half_height
```

Do not set the object center itself to `z = 0.720 m`.

### Candidate rejection

Try at most `NEAR_TARGET_POSITION_ATTEMPTS = 64` candidates per object, in slot
order `02` then `05`. This is an internal placement budget and is separate from
the existing full-task `MAX_SAMPLING_RETRIES = 12`.

Reject a candidate if its compiled/planned collision bounds, with the stated
margin, would:

1. escape the enclosure or shelf support;
2. overlap the target, tray, panel, pendant, any of the four existing active
   objects, or the other new object;
3. enter the frozen pregrasp, grasp, lift, or loaded-outbound swept keep-out
   volume for that cell and intrusion side.

Use collision dimensions and exact compiled collision checks. Annotation boxes
and a single 2-D TCP polyline are not sufficient. There is no upper-clearance or
sensor-response requirement: a placement may be benign.

After all movable objects settle, recheck exact collision AABBs, `data.contact`,
containment, XY drift, linear speed, and angular speed. A rejected or unstable
layout becomes a normal sampling failure; never silently drop one of the two
objects from an episode.

## Sampler and contract implementation

Create `PactPlaceCorridorV1011NearTargetSampler` as a new V10.6/V10.10-lineage
sampler. Do not modify the V10.10 sampler or artifacts.

Implementation requirements:

1. Inherit the existing eight-entry palette unchanged; do not add slots 08/09
   and do not load any new asset.
2. Build the layout from the unfiltered `PactPlaceCorridorV106Sampler` layout,
   then select slots `01,02,03,04,05,06`. Do not call the V10.10 four-object
   filter and then override its constants; that path has four-object assertions.
3. Apply the registered target/placement sampling above and publish:
   `near_target_slots`, target XY, per-slot center, radius, angle, candidate
   index, rejection counts/reasons, RNG derivation, six-object identity hash,
   and six-object layout hash.
4. Set a distinct environment marker and add it to every V10.6-lane allowlist
   used by expert routing, the speed amendment, frame telemetry, runner sampler
   resolution, and evaluation. Add behavioral tests proving V10.11 receives the
   same route/speed/telemetry behavior as V10.10 apart from its two added poses.
5. Require exactly six active clutter bodies and exactly two parked bodies.
6. Extend per-object contact/stability summaries, render labels, collection
   manifests, conversion verification, and evaluation reporting to slots 02 and
   05. Classification must remain name-based rather than relying on geom IDs.
7. Use new disjoint collection, split, and evaluation seeds and new immutable
   output roots. Never overwrite V10.10 data, checkpoints, or evaluation.

## Step 0 — tests before qualification

Add tests for:

- the unchanged eight-entry palette and exact six/two activation split;
- correct mug/can identities and collision dimensions;
- no geometry-size, mesh-scale, asset-metadata, or scene-XML mutation;
- target-relative annular-sector bounds and uniform-area formula;
- target XY sampled once, recorded, and reproduced;
- deterministic retry behavior and independent RNG streams;
- correct shelf-center Z;
- target/object, object/object, shell, panel, pendant, and route rejection;
- fail-closed behavior after 64 candidates;
- exact settled-contact and stability rejection;
- V10.11 expert route, speed, telemetry, and contact-audit dispatch;
- preservation of V10.10 tests and protected artifact hashes.

Run the focused V10.11 suite and the V10.10/V10.7/V9.5/contact-audit
regressions before generating an episode.

## Step 1 — placement and route preflight

Use the 24 frozen `family x side x pendant-pose` cells and four deterministic
placement replicas per cell (96 layouts total). Setup settling/`mj_forward` is
allowed; these are not retained training episodes.

Every layout must satisfy:

- six active and two parked objects with exact identity/hash reconciliation;
- both new objects within the registered sector around the recorded target;
- exact containment and zero forbidden initial contact after settling;
- both new objects under the existing drift/speed stability limits;
- complete scripted route construction and IK;
- zero robot/target contact with either new object along the scripted route;
- no regression in pendant, panel, or existing-clutter contact accounting.

Record per-layout placements, settled bounds, minimum route clearance and the
limiting geom pair. Stop if any of the 96 layouts fails; do not relax the sector,
margin, or object identities inside the same version.

## Step 2 — visual and small live check

Render a contact sheet containing all 24 cells plus six representative complete
episodes (three per intrusion side). The owner checks only that:

- the mug and can are visibly near, but not touching, the target;
- both positions vary across episodes;
- all six clutter objects are present and look physically settled;
- the target remains visible/graspable and the route looks normal.

This review is not a claim that the new objects create avoidance pressure.

Then run a preregistered 48-attempt pilot balanced across the 24 cells. Continue
to full collection only if:

- at least `12 / 48` attempts are strict-clean successes (a 25% yield keeps a
  144-success collection feasible under the existing 900-attempt ceiling);
- there are zero infrastructure/schema failures;
- there are zero spontaneous mug/can stability failures without robot contact;
- there are zero accepted episodes with mug/can contact.

Report the observed yield against V10.10's 46.0%; do not require equality.

## Step 3 — fresh collection, training, and evaluation

If Steps 0–2 pass:

1. Collect 144 strict-clean episodes, six per 24-way cell, with the existing
   900-attempt and wall-clock ceilings.
2. Convert and verify a fresh 120-train / 24-validation split, five/one per cell.
3. Train fresh ACT and PACT arms using the exact V10.10 hyperparameters,
   encoder, chunk-100 decoder, seed policy, and checkpoint-selection rule.
4. Run a fresh 40-instance paired ACT/PACT evaluation on a disjoint frozen
   stream using the V10.10 endpoint definitions plus per-object mug/can contact
   and stability summaries.

Retraining is required because the observation and state distribution now
contains two additional active objects near the target. V10.10 checkpoints may
be used only as historical context, not as the V10.11 comparison.

## Required close-out

At every stop or completion, report:

- exact tests and regressions;
- sampler/config/scene/implementation hashes;
- placement and rejection distributions by slot and cell;
- preflight and pilot tables;
- contact and stability taxonomy;
- every artifact path and SHA-256;
- explicit episode/collection/training/evaluation authorization fields;
- `git diff --check`.

No stage authorizes a later stage unless its stated conditions pass. Historical
V10.7–V10.10 artifacts remain read-only throughout.
