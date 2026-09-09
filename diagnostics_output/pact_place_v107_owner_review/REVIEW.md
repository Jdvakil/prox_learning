# V10.7 owner visual-review packet

> **The production pool FAILED at 21/48** against a 32/48 floor.
> **These videos are provided solely for owner visual assessment.**
> **Publishing them does not make the pool pass and does not authorize any downstream work.**

## Status

| check | result |
|---|---|
| offline certification | **passed** |
| offline six-group causality | **passed** |
| production pool | **FAILED — 21/48** |
| Phase 0 | not run |

The offline geometry, certification and causal checks all passed. The
pool did not: it yielded 21/48 strict-clean, Wilson 95% [0.307, 0.577], missing every registered
floor. `pool_passed` remains **false** and is not reinterpreted here.

## What these clips are

Six **complete retained trajectories** replayed from the pool at true
time (15.1515 fps, stride 1). No episode was generated, no
task resampled, no `env.step` called, no geometry or threshold changed,
and the pool was not rerun. Nothing is trimmed around the interesting
event.

**All three failures are natural production failures. None is an
induced pendant collision** — every selected failure records zero
robot-or-target pendant contact frames.

## Selection

Deterministic and derived, not hand-picked: among subsets with one row
per pendant pose in each outcome class, three left and three right
overall, and at least two layout families per class, minimise the
maximum pendant clearance, then total clearance, then the sorted
role-index tuple. 54846 valid subsets were
considered.

| # | outcome | role | pose | side | family | min clearance | frames | duration |
|---|---|---:|---|---|---|---:|---:|---:|
| 1 | clean success | 6 | neg5 | left | F1_inner_panel_stagger | 24.248 mm | 577 | 38.08 s |
| 2 | clean success | 28 | center | right | F0_target_side_stagger | 26.307 mm | 443 | 29.24 s |
| 3 | clean success | 8 | pos5 | left | F1_inner_panel_stagger | 16.052 mm | 544 | 35.90 s |
| 4 | natural failure | 45 | neg5 | right | F3_aperture_side_stagger | 21.517 mm | 441 | 29.11 s |
| 5 | natural failure | 40 | center | right | F2_outer_panel_stagger | 21.399 mm | 448 | 29.57 s |
| 6 | natural failure | 20 | pos5 | left | F3_aperture_side_stagger | 13.081 mm | 583 | 38.48 s |

## Panes and overlay

Untinted wrist RGB policy view; wide third-person view showing the
household clutter, robot, target, panel and pendant; pendant-side close
view; and a per-component proximity bar derived from retained state.
The overlay carries outcome, role, family, side, pendant pose, phase,
commanded and realized speed, current/running/episode-minimum pendant
clearance, limiting component and body, clutter contact and stability
state, task outcome, and pendant-contact state.

## Authorization

`eligible_for_owner_visual_review: true`, `pool_passed: false`, and
every authorization field false. `human_approval.json` is absent and
was not created. Phase 0 has not run and is not authorized.

Verify this packet independently with:

```
python scripts/verify_pact_place_v107_owner_review.py
```
