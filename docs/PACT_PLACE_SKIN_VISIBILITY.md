# PACT place: what the proximity skin can see

This note separates two limits that are easy to conflate: spatial resolution
inside a sensor cone and whether an object enters any cone at all. The numbers
below come from the frozen V9.5 replay in
`diagnostics_output/pact_place_v9_w1_resolvability_full/resolvability.json`;
they are not estimates from V10.11 videos.

## Sensor geometry

The hybrid skin contains 40 robot-mounted depth cameras: 7 on link 1, 7 on
link 2, 5 on link 3, 5 on link 4, 4 on the front and 6 on the back of link 5,
and 6 on link 6. Each produces an 8 by 8 depth image over a 45 degree field of
view. There are no link-7, hand, finger, or gripper sensors.

At range `R`, one pixel spans approximately `0.10355339 * R`. The artifact's
registered examples are:

| range | one pixel | two pixels |
|---:|---:|---:|
| 0.50 m | 51.8 mm | 103.6 mm |
| 0.75 m | 77.7 mm | 155.3 mm |

An 89 mm soap bottle therefore spans two pixels only inside approximately
0.43 m. Increasing an object's visible width improves this resolution limit
linearly, provided the object is already inside a sensor cone.

## The dominant place-corridor limit is coverage

The complete eight-variant replay found:

- inbound vessel maximum visible transverse width was exactly 0 in 7/8
  variants; the remaining, physics-dirty source variant reached 37.6 mm;
- outbound vessel maximum visible width ranged from 0 to 231.8 mm, depending
  on the panel side and route;
- the intrusion panel was visible in every variant at 358.0--388.2 mm.

The small tabletop objects are therefore usually missed because the distal
hand/gripper region passes them without sensor coverage, not merely because
their return is weak. V10.11 is consequently an appearance/camera robustness
environment. It does not establish that PACT's proximity channel is
shape-agnostic over household clutter.

## Practical ways to increase proximity relevance

From least to most disruptive:

1. Validate the existing low-wall family. A wall-mounted object enters from
   the corridor edge at forearm height, where the panel evidence shows that the
   present skin has coverage.
2. Widen a route obstacle, after re-deriving the corridor clearance budget.
   Wider silhouettes are easier to resolve but directly consume the open lane;
   old budget figures must not be reused without a new measurement.
3. Add link-7/hand/gripper sensors. This fixes the structural coverage gap, but
   changes the 40-sensor input contract and invalidates existing datasets and
   checkpoints for direct comparison.

Any future claim that an obstacle is proximity-visible should be gated with
`scripts/pact_skin_resolvability.py`, including its occlusion-aware ray test,
rather than with collision geometry or angular subtense alone.
