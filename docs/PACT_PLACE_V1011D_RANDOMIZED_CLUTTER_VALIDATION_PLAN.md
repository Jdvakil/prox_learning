# V10.11d — V10.11c clutter, randomized positions

**Status:** environment built; 96-layout preflight in progress. Collection,
training and evaluation are out of scope and delegated.

## Why

V10.11c is settled in terms of *what* the clutter is. It is not randomized in
terms of *where* it stands. Measured across all eight family/side combinations
of the frozen V9.5 layout:

| slot | object | V10.11c position |
|---|---|---|
| `03` | Plate_10 | **(0.980, −0.220) in every combination** |
| `04` | Plate_22 | **(1.090, +0.300) in every combination** |
| `01` | primitive cylinder (outbound vessel) | x 0.680–0.690, y ±0.010, plus ±20 mm x / ±5 mm y jitter |
| `06` | Soap_Bottle_11 (inbound vessel) | x 0.565–0.580, y ±0.010, plus ±5 mm x / ±10 mm y jitter |

The two plates never move at all, and the two vessels move by millimetres.
Slots `08`/`09` already draw from a target-relative annulus and are unchanged.

V10.11d keeps the palette, primitive shapes, heights and identity hash
byte-identical to V10.11c — the contract asserts the identity hash matches —
and randomizes the centres of slots `01`, `06`, `03`, `04` per episode.

## How placement is decided

Proposal boxes bound the draw; they do **not** define admissibility. Each
candidate is rejected unless it clears every check below, up to 96 candidates
per slot, in the order `01 → 06 → 03 → 04` (most constrained first, so a
rejected vessel does not waste the plates' draws).

| slot | proposal box x (m) | proposal box y (m) |
|---|---|---|
| `01` | 0.650 – 0.740 | ±0.055 |
| `06` | 0.545 – 0.600 | ±0.050 |
| `03` | 0.920 – 1.240 | ±0.320 |
| `04` | 0.920 – 1.240 | ±0.320 |

Checks per candidate:

1. containment inside `CLUTTER_WORKSPACE_LOW/HIGH`;
2. planar separation from every already-placed body (`NEAR_OBJECT_GAP_M`);
3. for slot `01` only, both `route_blocker_metrics` and
   `panel_corridor_metrics` must still report `detour_admitted`.

Slots `08`/`09` are sampled afterwards from the inherited annulus and see the
final occupancy, so the near-target logic is unchanged but now reacts to moved
base slots.

### Two bounds that are not arbitrary

**Slot `01`'s x floor is 0.650, not lower.** The route predicates alone admit
slot `01` down to about x 0.50 (left panel) / 0.55 (right). But the two vessels
need 98 mm of x separation and slot `06` cannot go below x 0.545 without leaving
the bench shell, so a lower floor for `01` starves `06` and the whole layout
fails. Measured over 800 synthetic layouts: a 0.600 floor placed all four slots
in **68.5%** of draws, with every failure being slot `06` exhausting against
slot `01`. The 0.650 floor gives **100%**.

**No target-clearance rule is applied to these four slots.** In a measured
V10.11c preflight row the cup's AABB overlaps slot `01`'s in all three axes —
55 mm in x, 15 mm in y — while the row still records zero forbidden initial
contact, because the cup mesh and the cylinder do not actually touch. Any
conservative planar separation rule would therefore reject V10.11c's own
working layout. The runtime settle and initial-contact check remain the
authority for cup/clutter overlap, exactly as in V10.11c.

## What changed in shared code

`PactPlaceCorridorV1011MixedClutterSampler` gains one hook,
`_randomize_base_slot_centers(by_slot, layout)`, called after the target draw
and before the near-target annulus. Its default returns `{}` and the layout key
is written only when non-empty, so V10.11, V10.11b and V10.11c layouts keep
exactly the key set and values they had before. The 20-test V10.11 suite passes
unchanged.

`pact_place_corridor_v10_11d_randomized_clutter` joins
`PACT_PLACE_V106_LANE_ENVIRONMENT_VERSIONS`, which is what `_v9_enabled`,
`_v106_enabled` and the inbound hazard-role set all read, so the expert routing,
speed amendment and frame telemetry follow automatically. The sampler is also
registered in the expert screen's import list, horizon allow-list and dispatch
ladder.

As in V10.11c, the two stale allow-lists at `enclosure_reach.py:5690-5701` and
`:5768-5780` are deliberately **not** joined: they stop at V10.2 and already
exclude the whole V10.6 lane, so joining them would make V10.11d diverge from
the V10.11c baseline it extends.

## Verification

- 96-layout preflight (24 cells × 4 replicates), requiring all rows to pass
  containment, settle stability, full IK and zero forbidden initial contact.
- Position spread reported per slot, to confirm every clutter item actually
  moves and the plates are no longer static.
- V10.11 regression suite green; V10.11c artifacts byte-unchanged.
- Owner review packet of 3 clean successes and 3 failures, as for V10.11c.
