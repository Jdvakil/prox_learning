# Hybrid Obstacle Observation & Reference Contract — Final Decision

Resolution of the two blockers that stopped the paired ACT + Safety-CVAE smoke:
deterministic wrist-camera replay, and the scientifically correct Safety-CVAE
reference contract.

Date: 2026-07-25 · Scope: audit and diagnosis only. ACT and the Safety-CVAE were
not trained or modified, no rendering or camera semantics were changed, and the
smoke4 evaluation was not continued.

---

## 1. Executive summary

**Both blockers are resolved as findings, and both require a decision that is not
this task's to make.**

The wrist-camera nondeterminism has an exact cause: **4x multisample
anti-aliasing is enabled** (`mjModel.vis.quality.offsamples = 4`), and the
classic OpenGL multisample resolve is not bit-reproducible on this driver. Ten
back-to-back renders of a byte-identical scene produced ten distinct wrist images,
differing in 13–19 of
219648 pixels by exactly **one least-significant bit**.
Setting `offsamples = 0` makes the view bit-identical over 8 renders. Every allowed
fix category — state restoration, forward refresh, camera-pose update, cache
clearing, cache keys, render order, renderer scene reset — was examined against the
evidence and **none of them is the defect**; the one candidate with a plausible
mechanism was tested and falsified. The only known route to determinism changes
rendered-image semantics, which this task is forbidden to do, so **no fix was
applied**.

One earlier inference needed correcting: `exo_camera_1` is **not** deterministic
either. It appeared stable in probes A and C, but under probe L's identical sweep
it diverges 4 of 8 times. Its earlier stability was incidental to that view. The
40 proximity streams are stable because they are *depth* renders, which the MSAA
resolve does not perturb.

The reference audit overturns the previous task's headline choice.
**`first_live_skin` is not canonical.** It appears in no README passage, no
committed demo and no paper text — it exists only as the else-branch at
`eval_act_obstacle_safety.py:213-215`. The documented contract (README:174,
`safety_react_demo.py:368-379`) recomputes the baseline **every frame** by parking
the bars at the *current* pose and re-rendering, which requires privileged
simulation state. And the Safety-CVAE was trained on **raw absolute depths with no
subtraction at all**, so subtraction is an inference-time device the demos added,
not part of the learned model.

That also explains the earlier saturation without any tuning: the frozen reference
sits at the step-0 posture while active sensors grow **10 → 22** as the arm enters
the cavity. There were zero hazard-bar contacts and the two known self-return
sensors were active in zero of 200 frames, so the correction tracked posture drift,
not hazard proximity.

**Final decision: `OBSERVATION_CONTRACT_CHANGED`** (token repeated verbatim as the last line).

---

## 2. Starting and final commits

| | |
|---|---|
| Root branch | `repair/hybrid-obstacle-observation-reference-v1` |
| Root starting commit | `75b2347cb4e550cf5c0976b2c9b7114cb6de1e81` |
| Root final commit | `75b2347cb4e550cf5c0976b2c9b7114cb6de1e81` |
| ACT branch | `eval/hybrid-obstacle-manifest-safety-v1` |
| ACT commit | `61f51b01e43e5016656b1aa39fb536143d8ccc32` |
| ACT changed this task | False |
| MolmoSpaces commit | `678f2eb4a0ac0d9e3d14e555aaac0e099089b9a5` |
| MolmoSpaces modified | False |

Branches deliberately not created: `repair/wrist-camera-replay-determinism-v1 (no MolmoSpaces fix was warranted)`, `repair/paired-reference-contract-audit-v1 (no ACT change was needed)`.

### Artifact verification

| | |
|---|---|
| `policy_best.ckpt` | `dd7cd108a64ce10e5aab21b525dc06190f54d4e5fe446f65715b6852c49e7d36` |
| best epoch | 1738 |
| `dataset_stats.pkl` | `c8119b904bfc80d66e3d33825722fcf9bb8bf3433c956dc09c27e6517d7c4ae2` |
| Safety-CVAE `model.pt` | `1fb2fc2b6023e64d2b9cbcf67fd5a24402968ec6f902c1e8a8595690396e7405` |
| Safety-CVAE `meta.json` | `7c873756fb16e4f3fc565f96c41a581816981ab93886f1db5d27e62110a5fc81` |
| 40-sensor order | `c31df8c36b0011b0eaf5b2eb5ce66d2514b5d6662ba9d7684ff021cd17cec858` |
| `camera_configs.py` | `7e90b4db37b0037344e9a55b35e1d4d98b9e2025edab32e7132a7a434799cfa6` |
| `model_hybrid.xml` | `50924661e0411f92ab529c790512b17b674e789434c592c3dbc6d2359164d4c6` |
| All matched | True |

## 3. Exact wrist-camera divergence cause

4x multisample anti-aliasing is enabled in the compiled model (mjModel.vis.quality.offsamples = 4). The classic OpenGL renderer's MSAA resolve is not bit-reproducible on this GPU/driver, so two renders of a byte-identical scene can differ by one least-significant bit on edge pixels. It is view-dependent only in likelihood, not in kind: the wrist view diverged on every render and the exo view diverged on half of them. Depth renders are unaffected, which is why all 40 proximity streams were bit-identical in the prior paired runs.

| | |
|---|---|
| `offsamples` (MSAA samples) | 4 |
| Renderer | MjOpenGLRenderer |
| Resolution | [624, 352] |
| Differing pixels | [13, 19] of 219648 |
| Fraction of pixels | 8.65e-05 |
| Max abs RGB difference | 1 |

Note both cameras are **registry-only** — neither has a MuJoCo `<camera>` element
(`model_camera_id = -1`), so their pose lives solely in `CameraManager.registry`
and `render_rgb_frame` renders a free camera from it.

## 4. Pre-fix probe matrix

| probe | result | note |
|---|---|---|
| A_render_wrist_twice_same_env_same_state | DIVERGES | 10 back-to-back renders of an unchanged state gave 10 distinct images |
| B_restore_exact_state_and_render | DIVERGES | qpos/qvel/ctrl/mocap/warmstart restored, mj_forward + update_all_cameras run each cycle |
| C_fresh_environment_same_manifest_state | DIVERGES for wrist, IDENTICAL for exo | across 5 independent constructions exo RGB is bit-identical and state/pose are identical, yet wrist RGB never  |
| D_fresh_process | SUBSUMED | probe A already shows divergence between two consecutive renders inside one process, so a cross-process compar |
| EF_render_order | ORDER-INDEPENDENT DIVERGENCE | wrist differs whether rendered before or after exo, so render order is not the cause |
| GH_cache_clearing | NOT APPLICABLE | _render_frame calls renderer.update(data, cam) on every call and sets the GL camera pose explicitly, so no per |
| IJ_reset_restore_sequence | DIVERGES | mj_forward followed by update_all_cameras does not make the wrist render reproducible |

### First-divergence localization

| component | verdict |
|---|---|
| cache_key_collision | RULED OUT - same reason |
| derived_mujoco_kinematics | RULED OUT - xpos/xmat digests identical |
| observation_cache_reuse | RULED OUT - two consecutive render_rgb_frame calls bypass any cache and still diverge |
| render_order | RULED OUT - probe EF |
| renderer_context_initialization | RULED OUT - divergence persists on renders 2..10 within one initialized renderer, so it is not a first-render warm-up |
| robot_mounted_camera_pose_update | RULED OUT - wrist registry pos/forward/up/fov identical across all probes; sample_task already ends with update_all_cameras (task_sampler.py:1157) |
| simulator_state_restoration | RULED OUT - qpos/qvel/ctrl/mocap/warmstart digests identical; divergence occurs with no state change at all |

The divergence originates inside the render call itself. With byte-identical simulator state, byte-identical camera pose and the same renderer instance, each render_rgb_frame('wrist_camera') returns a different image differing in 13-19 of 219648 pixels by exactly 1 LSB. exo_camera_1 is bit-identical under the same treatment, so the effect is view-dependent, consistent with unstable tie-breaking among near-coincident fragments in the close-up wrist view. Depth rendering is unaffected, which is why all 40 proximity streams were bit-identical in the prior paired runs.

### Probe K — scene-camera hypothesis, falsified

`_render_frame` passes a degenerate `MjvCamera` to `renderer.update()` with a
source comment claiming it "is not actually required". Since `mjv_updateScene`
uses the camera for culling and transparent-geom depth sorting, an ill-defined
camera could have produced unstable draw order. Tested by monkey-patching a
correctly configured camera, with the GL camera pose still overridden identically:
**REFUTED** — the wrist view stayed 8/8 distinct.
The hypothesis was falsified rather than asserted.

### Probe L — multisample hypothesis, confirmed

| | |
|---|---|
| wrist, MSAA as configured | 8/8 distinct, identical=False |
| wrist, MSAA disabled | 1/8 distinct, identical=True |
| exo, MSAA as configured | 4/8 distinct, identical=False |
| exo, MSAA disabled | 1/8 distinct, identical=True |

**Correction to an earlier inference.** Probes A and C showed exo_camera_1 as bit-identical, which suggested the effect was specific to the wrist camera. Probe L, which renders both cameras through the same sweep, shows exo diverging 4 of 8 times. exo's earlier stability was incidental to that view, not a property of the camera. The defect is general to MSAA'd RGB rendering.

## 5. The exact fix — and why none was applied

| | |
|---|---|
| clear_stale_per_camera_observation_caches | not the cause - back-to-back render calls bypass every cache and still diverge |
| correct_an_aliasing_cache_key | not the cause - same reason |
| explicit_robot_mounted_camera_pose_update | not the cause - registry pose identical; sample_task already calls update_all_cameras |
| first_render_only_after_state_and_camera_sync | not the cause - renders 2..10 diverge equally |
| mujoco_forward_refresh_after_restore | not the cause - state and derived kinematics were already identical |
| reset_renderer_scene_state | tested as probe K via a correctly configured scene camera - REFUTED |

| | |
|---|---|
| Narrow fix applied | no |
| Only known route | `set mjModel.vis.quality.offsamples = 0, i.e. disable anti-aliasing` |

Every allowed fix category was examined against the evidence and none addresses the cause. Applying one anyway would be a speculative change, which step 7 forbids. The one allowed-category hypothesis with a plausible mechanism (probe K) was tested and falsified.

It changes rendered-image semantics, which this task's hard constraints forbid ('Do not alter ... image preprocessing'; step 7 disallows 'changing images or preprocessing'). It is also a scientific decision rather than a bug fix: the 100-episode ACT training dataset was rendered with MSAA enabled through this same path, so evaluating with MSAA off would introduce a train/eval observation mismatch for the pinned checkpoint.

## 6. Proof camera semantics did not change

| | |
|---|---|
| camera_config_py_hash_unchanged | True |
| model_hybrid_xml_hash_unchanged | True |
| molmospaces_worktree_clean | True |
| resolution | [624, 352] |
| wrist_registry_fov | 56.74 |
| wrist_registry_pose | [0.3199982556767919, -0.08106286650310322, 0.8945843201687841] |

No camera pose, FOV, intrinsic, resolution, lighting, texture or preprocessing value was altered. The MSAA experiment in probe L mutated offsamples in-process only, restored it in a finally block, and never touched source.

## 7. Post-fix determinism matrix, open-loop replay, and the rollouts

Not run, and deliberately so.

| | |
|---|---|
| step_7_narrow_fix | not applied - every allowed category was examined and none addresses the cause; the one plausible candidate was tested and falsified (probe K) |
| step_8_determinism_acceptance | not run - it requires bit-identical wrist RGB after a frozen fix, which cannot pass without the forbidden semantic change |
| step_9_zero_equivalence_rollouts | not run - 0 of 3 permitted rollouts used; gated on step 8, and with determinism unrestored they could only re-demonstrate the known divergence without changing the decision |

| | |
|---|---|
| Full policy rollouts permitted | 3 |
| Full policy rollouts used | 0 |
| Probe env constructions permitted | 20 |
| Probe env constructions used | 9 |

Running the three rollouts would have re-demonstrated a divergence already
characterised at its root and could not have altered the decision, so the budget
was left unspent.

## 8. Canonical reference provenance

| id | mode | code path | demo | live ACT | privileged | frame-aligned | deployable | trained on | new choice |
|---|---|---|---|---|---|---|---|---|---|
| A | no subtraction (raw SafetyHead) | `SafetyHead.__call__ used directly; no committed consumer d` | False | False | False | True | True | True | False |
| B | first_live_skin (frozen step-0 reference) | `eval_act_obstacle_safety.py:213-215 (else-branch)` | False | True | False | False | True | False | True |
| C | clear/rest-pose skin | `safety_flinch_demo.py:293-296` | True | False | True | True | False | False | False |
| D | frame-aligned obstacle-parked skin (per frame) | `safety_react_demo.py:368-379 (also moving/orbit demos)` | True | False | True | True | False | False | False |
| E | obstacle-removed same-pose counterfactual | `not separately implemented; equivalent in effect to D (par` | False | False | True | True | False | False | False |
| F | step-indexed recorded-trajectory reference H5 | `eval_act_obstacle_safety.py:206-212 via --baseline_referen` | False | True | True | False | False | False | True |

### Citations

- **README.md:174** — the documented residual-mode formula
  > `dq = head(skin with obstacle) - head(skin, obstacle parked)`
- **README.md:178-179** — states the PURPOSE of subtraction: suppress static clutter and self returns
  > `The baseline subtraction is key: the head fires on *any* close surface (hood walls, the arm's own links), so subtracting its rest output makes each de`
- **scripts/safety_react_demo.py:12** — docstring explicitly says PER-FRAME
  > `dq_raw = head(skin_with_bars) - head(skin_bars_parked)   # per-frame baseline -> the bars' push`
- **scripts/safety_react_demo.py:368-379** — THE canonical implementation: at the CURRENT pose, park the bars, re-render, subtract, restore. Two renders per frame and privileged mocap manipulation.
  > `depths=render_all(...); for name in bar_mocaps: data.mocap_pos[mid[name]]=sw.PARK; mj_forward(); depths_rest=render_all(...); restore bars; mj_forward`
- **scripts/safety_flinch_demo.py:293-296** — computed ONCE, but valid there only because the flinch demo holds the arm at a FIXED rest posture (line 85, 109). Not transferable to a moving arm.
  > `# rest baseline: head output with the bar parked (subtracted every frame) / rest_depths=render_all(...); dq_rest=head(rest_depths)`
- **scripts/train_safety_cvae.py:2-3** — training maps RAW skin -> retreat delta
  > `joint-space retreat delta, distilled from the analytic potential-field labels produced by scripts/safety_sweep.py`
- **scripts/train_safety_cvae.py:89-94** — the network input is absolute closeness, NOT a difference. No subtraction anywhere in training.
  > `def featurize(prox): d=prox; c=clip(1-d/D_MAX,0,1); c[d<0.005]=0; return c.reshape(len(c),-1)`
- **scripts/train_safety_cvae.py:83-86** — z=0 determinism confirmed
  > `def act(self,x): z=torch.zeros(...); return self.dec(cat([x,z],-1))  # Deterministic head: decode at z = 0`
- **assets/safety/cvae_v3/meta.json:-** — the head is quiet only when nothing is near; inside a fumehood cavity walls and self returns are always near, so the raw head always fires
  > `far_quiet 0.0310, close_cos 0.9255, label_scale 11.359346389770508`
- **submodules/act/hybrid_safety_residual.py:5, 346-350** — the controller EQUATION is faithful; the module makes no claim about how the reference is obtained
  > `implements the controller equation already used by scripts/safety_react_demo.py`
- **submodules/act/eval_act_obstacle_safety.py:206-212** — mode F: step-indexed recorded-trajectory reference
  > `if self._reference_proximity is not None: baseline = self._head(self._reference_proximity[self._step])`
- **submodules/act/eval_act_obstacle_safety.py:213-215** — mode B: first_live_skin, frozen once at step 0. This is the ONLY place it exists.
  > `if self._baseline_safety_output is None: self._baseline_safety_output = self._head(current_proximity); baseline = self._baseline_safety_output`
- **paper/section3_proximity_signal_draft.md:-** — the paper establishes no reference contract
  > `no mention of baseline subtraction, parked obstacle, or reference`

## 9. First-live versus parked/counterfactual

first_live_skin is NOT canonical. It appears in no README passage, no demo, and no paper text. It exists only as the else-branch fallback in the live ACT adapter (eval_act_obstacle_safety.py:213-215). The documented contract is a PER-FRAME, FRAME-ALIGNED, obstacle-parked counterfactual, which requires privileged simulation state.

| | |
|---|---|
| Documented canonical | mode D - per-frame frame-aligned obstacle-parked counterfactual (README:174, safety_react_demo.py:368-379) |
| `first_live_skin` status | mode B - NOVEL_UNVALIDATED, present only at eval_act_obstacle_safety.py:213-215 |
| Safety-CVAE trained on | raw absolute depths, no subtraction anywhere (train_safety_cvae.py:2-3, 89-94) |

Three distinctions that matter:

- A frame-aligned parked or obstacle-removed reference is a useful **simulation
  oracle** but uses privileged counterfactual information, so results obtained
  with it cannot be presented as achievable on hardware.
- A frozen first-live reference **is** deployable but is **not** equivalent to the
  documented parked-obstacle subtraction, and the measured behaviour shows the two
  differ materially.
- The recorded-trajectory `react` demo does not transfer: it recomputes its
  baseline per-frame at the current pose, so it never depends on posture agreement
  with a recording. A step-indexed recorded reference is self-defeating for a
  residual controller, because the residual's whole purpose is to make the robot
  deviate from that recording.

## 10. Existing one-pair saturation attribution

| | |
|---|---|
| Primary cause | posture drift away from the frozen step-0 reference |
| Hazard-driven | False |
| Self-return-driven | False |
| Frozen reference norm | 0.806553 |
| Active-sensor growth | 10 -> 22 as the arm enters the cavity |
| Hazard-bar contact frames | 0 |
| Known self-return frames active | 0 |
| raw head-norm range | [0.1116, 2.1338] |
| correction ever clipped | False |
| corr(subtracted, n_active_sensors) | 0.364 |

The correction growth is attributable to posture drift away from the frozen step-0 reference, not to hazard proximity. The Safety-CVAE was trained on raw absolute skin depths, so its output is a strong function of how much static structure is in view; the number of active sensors more than doubles as the arm enters the fumehood cavity. Subtracting a reference captured at the step-0 posture therefore leaves a large residual that reflects the posture change. This is a direct consequence of using mode B (first_live_skin) rather than the documented mode D (per-frame frame-aligned parked-obstacle counterfactual), which by construction cancels the static contribution at the CURRENT posture.

**ONE episode. This attributes a mechanism; it does not measure an effect size and supports no claim about the controller's merit.**

## 11. Allowed scientific options for the next task

| label | controller | why |
|---|---|---|
| DEPLOYABLE | raw SafetyHead, no subtraction (mode A) | Needs only the live 40-sensor reading. It is also the ONLY option whose input distribution matches Safety-CVAE training, which used raw absolute depth |
| NOVEL_UNVALIDATED | frozen first-live skin (mode B) | Deployable, but established by nothing: it appears in no README passage, no committed demo and no paper text, existing only as the else-branch at eval |
| HISTORICAL_DEMO_ONLY | clear/rest-pose skin (mode C) | Used by safety_flinch_demo.py:293-296, where a single baseline is valid only because the arm is held at a fixed rest posture (lines 85, 109). |
| SIMULATION_ORACLE | per-frame frame-aligned obstacle-parked counterfactual (modes D/E) | THE documented canonical contract: README:174 and safety_react_demo.py:368-379, which parks the bars at the CURRENT pose, re-renders, subtracts and re |
| UNSUPPORTED | step-indexed recorded-trajectory reference H5 (mode F) | Available at eval_act_obstacle_safety.py:206-212 via --baseline_reference_h5, but it is indexed by step, so it is frame-aligned only while the live ro |

**A. Reproduce the documented counterfactual controller and label it an oracle.**

- cost: one extra 40-sensor render per control step; requires privileged mocap parking
- gives: a faithful reproduction of the published method and an upper bound on residual benefit
- does not give: a hardware-deployable controller

**B. Design and validate a deployable posture-conditioned reference.**

- cost: new design plus its own validation campaign
- gives: a deployable controller that cancels static structure at the current posture
- does not give: anything already established; this is new research, not a repair

**C. Evaluate the raw SafetyHead without subtraction.**

- cost: none beyond the evaluation itself
- gives: the only variant whose inputs match Safety-CVAE training, and a clean measurement of what the head does unaided in a cluttered cavity
- does not give: the static-clutter suppression the demos rely on

**D. Retain first-live subtraction as a newly defined controller and validate it separately.**

- cost: an explicit scope decision plus its own validation
- gives: a deployable controller that can be honestly named as new work
- does not give: continuity with the published method; it must not be reported as the documented parked-obstacle subtraction


Not chosen here. Selecting among A-D changes what the safety result means scientifically, so it belongs to whoever owns the claim, not to this repair task.

## 12. Tests and validation

| | |
|---|---|
| Test file | `tests/test_observation_reference_contract.py` |
| Passed | 22 |
| Failed | 0 |
| Ruff | clean |
| Byte compilation | clean |
| JSON validation | all 4 reports parse |
| `git diff --check` | clean |
| Process guard | no simulator or eval processes |

The suite locks in the artifact pins, the controller constants and label scale,
the provenance labels, the rule that a privileged mode can never be relabelled
DEPLOYABLE, the MSAA root cause, and the correction about `exo_camera_1`.

## 13. Constraints honoured

| | |
|---|---|
| act_trained_or_modified | False |
| camera_pose_fov_intrinsics_resolution_lighting_textures_preprocessing_altered | False |
| canonical_dataset_split_or_manifests_changed | False |
| controller_constants_tuned | False |
| full_45_row_evaluation_run | False |
| future_observations_shared_between_conditions | False |
| molmospaces_modified | False |
| mujoco_egl_cuda_driver_renderer_versions_changed | False |
| obstacle_robot_planner_task_collisions_success_horizon_changed | False |
| policy_best_or_dataset_stats_changed | False |
| pushed | False |
| recorded_wrist_image_injected | False |
| safety_cvae_trained_or_modified | False |
| zero_equivalence_redefined_statistically | False |

## 14. Changed files and commits

Root, on `repair/hybrid-obstacle-observation-reference-v1`:

```
scripts/hybrid_obstacle_wrist_determinism_probe.py            new  probe matrix launcher
tests/test_observation_reference_contract.py                  new  22 tests
diagnostics_output/hybrid_obstacle_observation_reference/      new  4 reports + decision
docs/HYBRID_OBSTACLE_OBSERVATION_REFERENCE_FINAL_DECISION.md   new
```

ACT and MolmoSpaces are unchanged, so no gitlink moved and neither optional repair
branch was created. Rollout videos, H5s, checkpoints and temporary images are not
committed. Nothing was pushed.

## 15. Reproduction commands

```bash
cd /root/prox_learning_hybrid_safety
PY=/root/act_retrain_venv/bin/python
export MUJOCO_GL=egl PYTHONUNBUFFERED=1
export MLSPACES_ASSETS_DIR=$PWD/assets
EID=f802a9057b18fc13615bcb1386d13cb839ade4b5fa5a038a5cfc054b9187b8e2

# probe matrix A/B/EF/IJ in one environment
$PY scripts/hybrid_obstacle_wrist_determinism_probe.py --mode in-process \
    --collection-manifest configs/hybrid_obstacle_candidate_manifest_v2.json \
    --eval-manifest configs/hybrid_obstacle_eval_smoke4_v1.json \
    --episode-id $EID --stack configs/hybrid_safety_stack_v1.json \
    --repeat-renders 10 --out /tmp/probe_inprocess.json

# probe C across 5 fresh environment constructions
$PY scripts/hybrid_obstacle_wrist_determinism_probe.py --mode fresh-env --fresh-envs 5 \
    --collection-manifest configs/hybrid_obstacle_candidate_manifest_v2.json \
    --eval-manifest configs/hybrid_obstacle_eval_smoke4_v1.json \
    --episode-id $EID --stack configs/hybrid_safety_stack_v1.json \
    --out /tmp/probe_freshenv.json

# the audit tests
$PY -m pytest tests/test_observation_reference_contract.py -q
```

Probes K and L are diagnostic scripts kept in the scratch directory; both
monkey-patch the render path in-process and neither modifies MolmoSpaces. Probe L
reads `mjModel.vis.quality.offsamples`, temporarily forces it to 0 to measure
stability, and restores it in a `finally` block.

## 16. Next recommended task

Two independent decisions, both needing an owner rather than more engineering.
(1) Observation determinism. Either accept a rendering-semantics change - set offsamples=0 for BOTH data collection and evaluation so training and evaluation stay matched, which means re-rendering the converted dataset or accepting a train/eval mismatch for the pinned checkpoint - or keep MSAA and replace exact bitwise zero-equivalence with a repeat-based equivalence gate, which the present task was explicitly forbidden from doing. A cheaper third option worth costing is to keep MSAA for the ACT observation path and note that paired comparisons must then be designed around per-render noise of 1 LSB on order 10-30 pixels.
(2) Reference contract. Choose among options A-D in reference_contract_classification.json. Note that the documented canonical controller is a simulation oracle, so a deployable claim cannot be made with it, and that first-live subtraction is new work that must be named as such.

## 17. Decision

| | |
|---|---|
| Wrist-camera replay deterministic | False |
| Root cause identified exactly | True |
| Fix available within the allowed categories | False |
| Determinism obtainable only by changing rendering semantics | True |
| ACT-only repeat equivalence | not run - gated |
| ACT-plus-zero equivalence | not run - gated |
| Reference contract audited | True |
| Non-privileged canonical reference established by provenance | False |
| Reference choice escalated | True |

OBSERVATION_CONTRACT_CHANGED
