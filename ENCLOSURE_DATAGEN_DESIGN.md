# Parameterized enclosure-reach data generation — design spec

Source: advisor's design directive (2026-06-09). One **parameterized scene generator** (not bespoke
environments), an **observation-realizable scripted expert**, **decorrelation by construction**, and
**dataset probes before any training**.

## 1. Primary generator: enclosure-reach (shelf cubby)

One MJCF scene whose enclosure is assembled from oversized slabs on **mocap bodies**, re-posed per
episode (no recompile): aperture = gap between slabs, protrusion = slab inserted vs parked away.

Per-episode parameters θ (all logged):

| param | distribution | notes |
|---|---|---|
| `clearance_cm` | 0.7·U(1,5) + 0.3·U(5,8) | aperture = gripper bbox + clearance; mass in the 1–5 cm regime where proximity drives behavior |
| `depth_m` | U(0.25, 0.55) | enclosure depth |
| `target_depth_frac` | U(0.5, 0.9) | target pose along depth; + lateral U over interior width; + yaw U(0,2π) |
| `interior_margin_cm` | U(0, 6) | interior cross-section beyond aperture |
| `protrusion_present` | Bernoulli per mixture cell | see §3 |
| `protrusion_wall` | U{left, right, top} | independent of everything visible |
| `protrusion_pos_frac` | U(0.25, 0.75) | along depth axis |
| `protrusion_size_cm` | U(3.5, 7) | cross-section; ≥ a few SPAD zones at decision distance (zone ≈ 45°/8 ≈ 5.6° → ~1.2 cm @ 12 cm) |
| `protrusion_intrude` | sized so residual gap ∈ [grip+0.5, grip+4] cm (feasible) or < grip (abort cell) | |
| `light_scale` | log-U(0.02, 1.0) | per-episode lighting; NOT a separate split |
| `target_uid` | U(graspable pool, eggs excluded) | nuisance |
| derived: `cam_visible` | raycast/segmentation check from exo+wrist at t=0 | LOGGED, not sampled — needed for stratified eval, unrecoverable later |

**Decorrelation by construction:** protrusion params drawn independently of (clearance, depth,
lighting, object id). Tight-but-clear episodes included so narrow aperture ⇏ obstacle. Verified
post-hoc by the correlation-matrix probe (§5.3).

## 2. Expert: ScriptedEnclosurePolicy (observation-realizable)

The privileged planner reacts to hidden geometry at t=0 — unlearnable from student observations
(and the early deflection leaks obstacle presence into RGB via the robot's own pose). Fix: the
expert reacts to hidden geometry **only after it enters the skin's FOV/range envelope**
(detection-gated; detection distance derated near concave corners for multipath).

Three behavior classes (each a measurable signature of sensor use):
1. **Graded speed modulation** — v ∝ min clearance (continuous signature).
2. **Deflection-side selection** — re-route around the protrusion only after detection
   (discrete signature, large trajectory divergence).
3. **Abort/retreat** — when residual gap < gripper + margin (discovered at detection time):
   retreat cleanly and stop. "Don't proceed" is a behavior class; counts as success for its cell.

Phases: approach aperture → align → insert (speed-modulated) → [on detection: deflect | abort]
→ grasp → retract with object.

## 3. Mixture (per advisor; dynamic-intrusion deferred)

~28% obstacle-free (full clearance range incl. tight-but-clear) · ~33% hidden obstacle ·
~28% visible obstacle · ~11% abort/infeasible. Lighting sampled across all cells. Every eval
cell gets honest training mass; no factor pair correlated.

## 4. Scale & holdouts

2–5k episodes. Holdouts defined on the logged params (post-hoc split lists):
- unseen protrusion position/size bands (interpolation),
- aperture extrapolation band: train ≥2.5 cm clearance, test 1.5–2.5 cm,
- optional novel-shape split (L-cubby / cylindrical bin) later.

## 5. Dataset probes (BEFORE any training)

1. Regress expert EE speed on min skin reading → strong negative slope, or the expert isn't
   modulating and no training trick will fix it.
2. Logistic probe: deflection side from left/right zone asymmetry → ≫ chance.
3. Correlation matrix: hidden-obstacle params × all visible params ≈ 0.
4. Signal distribution: fraction of timesteps with any zone < 8 cm (too small ⇒ clearance
   distribution too generous).

## 6. Secondary generator: bin/drawer descent

Different occlusion mechanism (gripper self-occludes target in the last 5–10 cm; near-wall targets
force sub-cm wall proximity RGB can't judge metrically). Build AFTER the primary works. Two
mechanisms is the generality claim; resist a fifth environment.

## 7. Pipeline integration notes (filled by research)

- Custom policy class wired via `policy_config` (replaces PickPlannerPolicy for this collection).
- Custom task success: reach-and-lift OR clean-abort; `filter_for_successful_trajectories`
  semantics must keep abort-class episodes.
- Per-episode θ + `cam_visible` + behavior label logged into the h5 (task_info/obs_scene path).
- Lighting randomization verified per episode (texture randomization off must NOT disable it).
- Per-step ground-truth min-clearance recorded as an extra observation for probes 1 & 4.
