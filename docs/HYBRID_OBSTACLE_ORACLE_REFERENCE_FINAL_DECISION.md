# Hybrid obstacle — per-frame parked-obstacle oracle reference: final decision

Date: 2026-07-25
Task: implement and qualify the documented per-frame parked-obstacle oracle reference
(`ORACLE_PARKED_REFERENCE_V1`) on the four development rows.
Scope: this task does **not** develop a deployable reference, retune the controller, or
execute `confirmatory41`.

> **The oracle is privileged and is not deployable.** It moves a scene body the robot
> cannot move and observes a world that does not exist. Everything below measures whether
> the Safety-CVAE contains a usable hazard-specific differential. Nothing below is a claim
> about what is achievable on hardware.

---

## 1. Executive summary

The documented reference works, and the frozen controller consumes it without gross
regression.

* **The counterfactual isolates the hazard exactly.** On the hazard-absent row the
  differential is **bitwise zero on every frame of all five rollouts**, the correction is
  exactly zero, and the executed action equals the nominal action bit for bit. On the
  hazard-present rows, **every** frame with a nonzero differential has at least one sensor
  patch the hazard actually changes — 0 unattributable frames out of 105. The static
  enclosure geometry that dominated the raw head cancels by construction.
* **The residual is small, stable and never clipped.** Median saturation **0.000** across
  all 20 rollouts, nothing ever reached the ±0.35 limit, and peak correction norms are
  0.036–0.20 rad.
* **Pooled hazard-present task success is 14/15, against ACT-only 11/15.** No row flipped
  from 5/5 to 0/5. Hazard-bar contact occurred in **1 of 15** hazard-present oracle
  rollouts (ACT-only: 0 of 15).
* **Direction agrees with the analytic teacher where the hazard is close**: median cosine
  **+0.941** on the committed teacher's 31 active frames, 87.1 % positive.
* All **12 technical**, **6 controller** and **6 direction** gates pass.

Two findings changed the implementation and are load-bearing:

1. The observation's "latest" proximity sub-step is rendered **one sim sub-step before the
   policy step ends**. Pairing it against a counterfactual rendered at the decision state
   produced a differential whose **median contamination ratio was 1.0** — on a typical
   frame the entire apparent signal was pose lag, not the hazard. On the hazard-absent row
   it produced a spurious differential of up to **2.50** where the correct pairing gives
   exactly **0**. The oracle therefore pairs a re-render of the live scene at the decision
   state against the parked render of that same state.
2. The demo's `mj_forward` before the counterfactual render is **not** transplantable to a
   stepped environment: after `mj_step`, every body's `xpos` lags its `qpos` by one physics
   sub-step, and calling `mj_forward` would move the whole scene by up to
   **2.5 × 10⁻⁴ m** before the parked render.

**Decision: `ORACLE_REFERENCE_VALID_CONTROLLER_VIABLE` (Case A).**

---

## 2. Starting and final commits

| Repo | Branch | Start | Final |
|---|---|---|---|
| root | `eval/hybrid-obstacle-oracle-reference-v1` | `5a16963` | see §14 |
| ACT | `eval/hybrid-obstacle-per-frame-oracle-v1` | `68713a1` | see §14 |
| MolmoSpaces | `repair/hybrid-obstacle-manifest-runner-v2` | `678f2eb` | `678f2eb`, **unmodified** |

No commit was made on an existing development or repair branch. Nothing was pushed.

---

## 3. Artifact verification

41 checks, **0 failures** (`provenance_verification.json`). Expected digests were recovered
from the prior decision artifacts, not recomputed from the files being checked.

| Artifact | SHA-256 |
|---|---|
| `policy_best.ckpt` (epoch 1738) | `dd7cd108a64ce10e5aab21b525dc06190f54d4e5fe446f65715b6852c49e7d36` |
| `dataset_stats.pkl` | `c8119b904bfc80d66e3d33825722fcf9bb8bf3433c956dc09c27e6517d7c4ae2` |
| Safety-CVAE `model.pt` | `1fb2fc2b6023e64d2b9cbcf67fd5a24402968ec6f902c1e8a8595690396e7405` |
| `model_hybrid.xml` | `50924661e0411f92ab529c790512b17b674e789434c592c3dbc6d2359164d4c6` |
| 40-sensor order | `c31df8c36b0011b0eaf5b2eb5ce66d2514b5d6662ba9d7684ff021cd17cec858` |
| `development4` manifest | `5aaf6ddb4aba56bc17434fb860f809c137ba8e5fd41b309cd6382c66c8a1bd0b` |
| `confirmatory41` manifest | `7b4500e9b4b2868e2612d7e444c34762d72c5e6e7b4b7c38bcf31f027b51b69e` |
| oracle schedule | `d5b483a0bd977bb246e9b507dc140017c66c207fb68ac5244944433867fad72a` |

`offsamples = 4` verified at scene build in all 20 rollouts. `label_scale` =
11.359346389770508, applied exactly once. The four source trajectory H5s re-hash to their
manifest values. `confirmatory41.executed_in_this_task == false`.

---

## 4. Exact documented oracle provenance

Extracted from source by AST, not from prose (`current_oracle_audit.json`).

**Formula** — `README.md:174`:

```
dq = head(skin with obstacle) − head(skin, obstacle parked)
```

**Canonical implementation** — `scripts/safety_react_demo.py:369-379`:

```python
depths = render_all(model, data, rd, opt, sensors)
# per-frame baseline: same pose, ALL bars parked -> isolate the bars' marginal push
for name in bar_mocaps:
    data.mocap_pos[mid[name]] = sw.PARK
mujoco.mj_forward(model, data)
depths_rest = render_all(model, data, rd, opt, sensors)
for name, _half, wp in placed:
    data.mocap_pos[mid[name]] = wp
mujoco.mj_forward(model, data)

dq_raw = (head(depths) - head(depths_rest)) / max(head.scale, 1e-6)
```

**Which body moves.** The hazard is a MuJoCo **mocap body** — `protr_s`, `protr_m`,
`protr_l` (`enclosure_reach.py:46`). Parking writes `data.mocap_pos` only;
`mocap_quat`, `qpos`, `qvel` and `ctrl` are never touched (`_mocap_set`,
`enclosure_reach.py:180-183`).

**The committed parked pose** — `enclosure_reach.py:203-204`:

```python
for k, (px, py) in zip(PROTR, ((0.0, 0.8), (0.0, 1.2), (0.0, 1.6))):
    self._mocap_set(env, k, [px, py, -2.0])
```

| body | committed parked position |
|---|---|
| `protr_s` | `[0.0, 0.8, −2.0]` |
| `protr_m` | `[0.0, 1.2, −2.0]` |
| `protr_l` | `[0.0, 1.6, −2.0]` |

`_apply_theta` parks **all three** and then re-places only the chosen one, so on a
hazard-absent row every bar already sits at its committed parked pose — parking is a
bitwise no-op there **by construction**, which is what makes §8's negative control exact.
`manifest_runner._PARKED_Z_THRESHOLD = −1.0` is the runner's own test for "a hazard is
compiled into the scene".

**When each observation is captured.** MolmoSpaces renders the 40 sensors at 8×8 with the
cosmetic skin hidden (`geomgroup[2] = 0`), depth rendering enabled, skybox off
(`env.py:374-409`), once every `proximity_sensor_period_ms` during the sub-step loop
(`task.py:376-386`). The observation stacks those frames and
`extract_latest_proximity` takes `[-1]`.

**Controller constants and timing** — unchanged and untouched: gain 4.0, decay 2.2,
EMA 0.75, max_dev 0.35, dt 0.066 s, `label_scale` 11.359346389770508.

**Did the demo use prerecorded nominal motion?** **Yes.** `safety_react_demo.pick_episode()`
loads a recorded successful reach-grasp-lift, resamples and smooths it, and advances a
scalar phase `s` (`safety_react_demo.py:246-249, 386`). The nominal motion is open-loop
playback. The **reference is nevertheless recomputed every frame at the current executed
pose** — the per-frame property is a property of the reference, not of the nominal.

**Existing parked utilities are not reusable here.** Every demo builds its own scene via
`safety_sweep.build_model()` with a hard-coded `ROBOT_XML` path that does not exist in this
checkout, and parks `bar_s/m/l` at `PARK = [0, 0, −3.0]`. The manifest scene's hazard is
`protr_s/m/l` parked at z = −2.0 with distinct x, y. The mechanism transfers; the
coordinates and body names do not.

---

## 5. The old 109-frame failure

**Mechanism** — `eval_act_obstacle_safety.py:207-211`:

```python
if self._step >= len(self._reference_proximity):
    raise HybridSafetyContractError(
        "Reference ACT-only proximity ended before the live safety rollout"
    )
baseline = self._head(self._reference_proximity[self._step])
```

`load_proximity_sequence_h5` returns a `(T, 40, 8, 8)` array whose `T` is the **recorded
length of the expert demonstration**, not the live horizon:

| row | recorded real timesteps | live horizon |
|---|---|---|
| cand 106 | **109** | 200 |
| cand 107 | 110 | 200 |
| cand 108 | 63 | 200 |
| cand 118 | 60 | 200 |

The prior oracle attempt used candidate 106 and died at step 109 — exactly its recording's
length.

**Two independent faults, not one.** (a) The reference was a *prerecorded trajectory*
rather than a counterfactual of the current state, so once the residual moved the arm the
reference no longer described the same pose at all. (b) It was *indexed by step into a
finite array*. Padding or wrapping would have silenced the exception while leaving fault
(a) untouched, and is prohibited.

**A per-frame counterfactual has neither.** There is no array and no index: the reference
is a function of the state at the instant of the call, so no horizon can exhaust it.

---

## 6. The new per-frame implementation

`submodules/act/parked_obstacle_reference.py` — `PerFrameParkedObstacleReference`.

Per control timestep:

1. capture the live ACT observation and nominal action (unchanged);
2. **re-render the live scene** at the decision state → `current_skin_t`;
3. snapshot `mjSTATE_INTEGRATION`, `mocap_pos`, `xpos`, `xipos`, `geom_xpos`;
4. translate each hazard mocap body to its committed parked pose;
5. render → `parked_skin_t`, through MolmoSpaces' own `record_proximity_depths`;
6. write every snapshotted array back;
7. verify 21 hashed state fields and the `mjSTATE_INTEGRATION` buffer;
8. `dq_oracle_t = SafetyHead(current_skin_t) − SafetyHead(parked_skin_t)`;
9. frozen controller dynamics, after ACT temporal aggregation, arm only, gripper copied
   bit for bit.

**The audited adapter is not modified.** Its own `else` branch consumes
`self._baseline_safety_output`; seeding that field with the fresh parked-head output makes
it compute the differential through the committed controller. `current_skin_t` reaches the
head through the adapter's own `ProximityIntervention.select` seam — its documented
mechanism for choosing which proximity the safety head sees, independently of what ACT
sees. **ACT's observation is untouched.**

### 6.1 Why `current_skin_t` is a re-render and not the observation

The last proximity sub-step render lands one sim sub-step before the policy step ends, so
the observation's "latest" proximity is at a slightly earlier pose than the state the
counterfactual must be rendered from. Measured offline over the four rows:

| row | observation-paired max ‖Δ‖ | pose-consistent max ‖Δ‖ | median contamination ratio |
|---|---|---|---|
| 106 | 0.8759 | 0.8759 | 1.00 |
| 107 | 1.6397 | 1.6420 | 0.026 |
| 108 | 0.3757 | 0.3117 | 0.019 |
| **118 (no hazard)** | **2.5018** | **0.0000** | 1.00 |

On the hazard-absent row the observation-paired form invents a differential of up to
**2.50** where the correct pairing gives exactly zero. Had the observation been used, the
negative control in §8 would have failed outright and the whole oracle would have been
uninterpretable.

### 6.2 Why `mj_forward` is not called

`mj_step` runs the forward dynamics at `qpos_t` and only then integrates, so after a policy
step every body's `xpos` lags its `qpos` by one physics sub-step — and that lagged pose is
the one the live proximity observation was rendered from. Calling `mj_forward` before the
counterfactual render would advance **23 bodies** by up to **2.5 × 10⁻⁴ m** (proximity
cameras up to 2.2 × 10⁻⁴ m), so `parked_skin` would differ from `current_skin` by a
whole-scene pose change as well as by the hazard.

Measured on a scratch `MjData` seeded from the live integration state, so the live data —
including `qacc_warmstart`, which `mj_forward` would overwrite — is never touched. Because
the hazard is a mocap body translated with no rotation, its render state is updated exactly
and minimally instead. **No dynamics function is called at all**, so the counterfactual is
invisible to the integrator by construction rather than by repair.

---

## 7. State-neutrality evidence

21 fields hashed before and after **every one of the 4 000 counterfactual renders**
(20 rollouts × 200 steps), plus the full `mjSTATE_INTEGRATION` buffer compared byte for
byte.

| requirement | result |
|---|---|
| simulation-time delta | **0.0**, every reference |
| `qpos`, `qvel`, `act`, `ctrl` | unchanged |
| `qacc_warmstart` | unchanged (never written) |
| obstacle pose after restore | exact, every reference |
| `xpos`, `xipos`, `xquat`, `xmat`, `geom_xpos`, `geom_xmat`, `cam_xpos`, `cam_xmat` | restored |
| contact count and geom pairs | identical |
| `sensordata` | identical |
| RNG state | untouched (no dynamics call) |
| observation proximity buffer | truncated back to its exact prior contents |
| dynamics functions called | **0** |
| **neutrality failures** | **0 / 4 000** |

The check is not decorative: a deliberately sabotaged render that perturbs `qpos` by 1e-9
is caught and raises (`test_mutation_is_detected_and_raised`). The first implementation,
which followed the demo and called `mj_forward`, **was rejected by this gate** — that is
how §6.2 was found.

---

## 8. Hazard-absent negative control (candidate 118)

| requirement | result |
|---|---|
| current and parked scientific scenes equivalent | yes — all three bars already parked |
| `current_skin` and `parked_skin` bit-identical | **every frame, all 5 rollouts** |
| SafetyHead outputs bit-identical | **every frame** |
| oracle differential max abs | **0.0** (tolerance 1e-7) |
| filtered output | **exactly zero** |
| accumulated correction | **exactly zero** |
| executed action equals nominal | **yes, bit for bit** |
| gripper bitwise identical | **yes** |

This is exact rather than approximate because `_apply_theta` parks all three bars and
re-places only the chosen one, so on a hazard-absent row parking writes values that are
already there.

**One honest note.** Rollout `cand118_oracle_r0` recorded 31 `other_environment` contacts
where the ACT-only repeat recorded none. Its correction was provably **exactly zero** on
every frame and its executed action equalled the nominal every frame, so that variation is
MSAA-driven rollout stochasticity, **not** the safety controller. Per the handoff, task and
contact outcomes on this row need not match a specific prior ACT-only repeat; the safety
correction itself is what must remain inactive, and it did.

---

## 9. Offline differential and teacher analysis

Method: the four recorded expert action sequences are replayed **open-loop** through the
live environment from the verified initial state, with a per-frame counterfactual at every
timestep. No ACT, no residual controller in the loop, no policy.

Replay rather than pose-setting because the trajectory H5 stores **no per-step pose for the
pickup object** (`env_states/actors` is empty), so setting only the robot would leave the
target frozen at its rest pose.

**Reconstruction fidelity** (re-rendered skin vs the recording): mean pixel agreement
0.39–0.49 at 1 mm, median absolute depth difference **1.1–3.3 mm**. The reconstruction
tracks the recording closely but is not bitwise; max-abs is dominated by single pixels
straddling a depth discontinuity. This affects *which* states are sampled, not the validity
of the differential at each state, since both renders are of the same reconstructed state.

**Teacher.** The committed analytic potential-field repulsion
(`safety_sweep.py:322-351`), restricted to returns landing on the hazard box —
which needs no global scene-surface inventory and so is exactly computable in the live
manifest scene, and is precisely the quantity the differential should recover.

| row | teacher-active frames | median cosine | fraction positive | min hazard return |
|---|---|---|---|---|
| 106 | 0 | — | — | — |
| **107** | **32** | **+0.941** | **0.871** | 6.6 cm |
| 108 | 0 | — | — | — |
| 118 | 0 | — | — | — |

The committed teacher fires only when a hazard return is a sensor's **closest** return
inside `D_ACT = 0.18 m`; on rows 106 and 108 the hazard never becomes one.

**Supplementary geometric direction audit** (reported, **not** a predeclared gate) at the
head's own 0.5 m input radius, so direction can be inspected on all three hazard rows:

| row | active frames | median cosine | fraction positive |
|---|---|---|---|
| 106 | 7 | **+0.990** | 1.000 |
| 107 | 43 | **+0.941** | 0.738 |
| **108** | 37 | **−0.362** | **0.054** |

Direction is established where the hazard is close and is **reversed at long range on
candidate 108**. This is a real qualification on the result and is carried into §13.

### 9.1 Offline viability gates (step 8)

| gate | result |
|---|---|
| hazard-absent differential ≤ 1e-7 | **PASS** (0.0) |
| hazard-present differential nonzero on active frames | **PASS** |
| median cosine > 0.5 on active frames | **PASS** (+0.941) |
| ≥ 70 % active frames positive | **PASS** (87.1 %) |
| activation attributable to the hazard, not static geometry | **PASS** (0 / 105 unattributable; 7 + 57 + 41 frames on rows 106/107/108) |
| the two known self-return sensors do not dominate | **PASS** (share 0.000) |

The attribution gate is the direct answer to the raw head's failure mode: every frame with
a nonzero differential has at least one sensor patch that the hazard actually changes.

---

## 10. The 20-rollout schedule

`configs/hybrid_obstacle_oracle_schedule_v1.json`, sha256 `d5b483a0…`, frozen before
execution. Four development rows × `ACT_PLUS_ORACLE` × 5 repeats = **20** privileged
rollouts. Budget 20/20 used. **No entry was replaced, rerun or added**, and no failed
execution was retried — all 20 exited 0 on their first and only attempt.

The 20 ACT-only baselines were **reused, not rerun**, after verifying condition, episode
id, repeat index, initial-state hash, `offsamples = 4` and shadow-zero equivalence on each:
20/20 verified.

**One ACT-only compatibility rollout** was run on candidate 106 (budget 1/1) to prove the
new evaluator did not alter nominal execution: exit 0, 200 frames, success, initial-state
hash matched, shadow-zero passed. It is excluded from every baseline statistic.

Each rollout used a fresh isolated process, reconstructed the manifest row and accepted
retry, verified the initial-state hash and `offsamples`, reset temporal aggregation and the
residual controller, and wrote its own condition/repeat manifest.

---

## 11. Per-row outcomes

| cand | hazard | ACT-only succ | oracle succ | AO cf-succ | OR cf-succ | AO hazard-bar | OR hazard-bar | OR other-env | max corr ‖·‖ | sat |
|---|---|---|---|---|---|---|---|---|---|---|
| 106 | present | 5/5 | **5/5** | 0/5 | 0/5 | `[0,0,0,0,0]` | `[0,0,0,0,0]` | `[33,35,32,27,32]` | 0.086–0.200 | 0.000 |
| 107 | present | 5/5 | **5/5** | 0/5 | 0/5 | `[0,0,0,0,0]` | `[0,0,0,1,0]` | `[77,90,84,90,82]` | 0.055 | 0.000 |
| 108 | present | 1/5 | **4/5** | 1/5 | 4/5 | `[0,0,0,0,0]` | `[0,0,0,0,0]` | `[0,0,0,0,0]` | 0.000–0.040 | 0.000 |
| 118 | absent | 0/5 | 0/5 | 0/5 | 0/5 | `[0,0,0,0,0]` | `[0,0,0,0,0]` | `[31,0,0,0,0]` | **0.000** | 0.000 |

For contrast, candidate 106's ACT-only other-environment contacts were `[61,68,63,68,55]`
against the oracle's `[33,35,32,27,32]` — five-versus-five with no overlap, with the
correction active on 36–40 % of frames. Grasp retained at the end of the episode: 5/5, 5/5,
4/5, 0/5.

First oracle activation: step 0 on rows 106 and 107 (phase `act_live`), steps 145 and 69 on
the two candidate-108 rollouts where the differential was nonzero at all.

### Pooled hazard-present (15 executions)

| | ACT-only | oracle |
|---|---|---|
| task success | 11/15 | **14/15** |
| collision-free success | 1/15 | 4/15 |
| rollouts with ≥ 1 hazard-bar contact | 0/15 | 1/15 |

No confirmatory statistical testing was performed.

---

## 12. Correction and saturation analysis

| statistic | value |
|---|---|
| median saturation fraction over 20 rollouts | **0.000** |
| rollouts saturated > 75 % of timesteps | **0** |
| rollouts that ever reached the ±0.35 clip | **0** |
| peak correction norm range (nonzero rollouts) | 0.036 – 0.200 rad |
| candidate 118 correction | exactly 0.000, all 5 rollouts |
| references generated | 200 per rollout, 4 000 total |
| nonfinite actions | 0 |

The controller is nowhere near its limits. The failure mode that disqualified the raw head
— a persistent unsubtracted push — does not appear: the correction decays back towards
nominal, with final norms of 0.000–0.175.

### 12.1 A metric that must not be trusted

`mujoco.mj_geomDistance` returns **exactly 0.0** for `robot_0/fr3_link7_collision` against
the hazard box at the reset state, while every geometrically comparable neighbour returns
0.21–0.29 m and a synthetic box-pair control confirms the call's `distmax` semantics are
correct. Because the audited adapter's `_minimum_environment_distance` takes a **minimum**
over all such pairs, that one geom pins the reported clearance at ≤ 0 on every frame of
every rollout **in every condition, including the raw-head task**.

Consequence: `minimum_clearance_m` is usable as "deepest penetration observed" but **is not
a clearance margin**, and a hazard-only distance built the same way is unusable. No safety
claim in this report rests on either; they rest on contact classification from
`data.contact`, which uses MuJoCo's real collision pipeline and is unaffected. The adapter
is not modified — it is the audited file. Evidence: `geom_distance_defect.json`.

---

## 13. Interpretation, and the Case A/B/C classification

**The reference.** The parked-obstacle counterfactual does what the documentation claims:
it cancels the static enclosure exactly and leaves the hazard's marginal contribution. The
hazard-absent row is bitwise zero; every nonzero differential on the hazard-present rows is
attributable to a sensor patch the hazard changes. This is the direct contrast with the raw
head, whose failure was diagnosed as a persistent geometry-driven push from an unsubtracted
signal.

**The controller.** Under the frozen constants the residual is small, never clipped, and
does not degrade the task: pooled hazard-present success 14/15 against 11/15, no row
flipped, one hazard-bar contact in 15.

**Case A — differential and live controller both work.**

Four qualifications belong with that verdict:

1. **Candidate 108's improvement is not evidence of a safety effect.** Three of its five
   oracle rollouts had an *identically zero* differential, so their control law was the
   ACT-only one; the remaining two had peak corrections of 0.036 and 0.040. The 1/5 → 4/5
   change is rollout stochasticity on the row predeclared *unstable* in the raw-head task,
   not a demonstrated effect.
2. **Direction is established at short range only.** The committed teacher activates on one
   of three hazard rows. The supplementary audit agrees strongly on 106 (+0.99) and 107
   (+0.94) and is **reversed on 108 (−0.36, 5.4 % positive)**, where the hazard never comes
   within the committed activation radius.
3. **Candidate 107 gained a hazard-bar contact** in 1 of 5 repeats where ACT-only had none
   (and the only negative `minimum_hazard_distance` reading in the set). One repeat is not
   a trend, but it is not nothing either.
4. **Four rows and five repeats cannot establish an effect size.** This qualifies the
   reference and the constants for further research; it is not a result.

---

## 14. Changed files and commits

**ACT** (`eval/hybrid-obstacle-per-frame-oracle-v1`), new files only:

* `parked_obstacle_reference.py` — the per-frame state-neutral counterfactual
* `eval_act_obstacle_oracle.py` — `ACT_PLUS_ORACLE` evaluator and `PoseConsistentProximity`

**Root** (`eval/hybrid-obstacle-oracle-reference-v1`):

* `configs/hybrid_obstacle_oracle_schedule_v1.json`
* `scripts/hybrid_obstacle_oracle_provenance.py`
* `scripts/hybrid_obstacle_oracle_source_audit.py`
* `scripts/hybrid_obstacle_oracle_offline_signal.py`
* `scripts/hybrid_obstacle_oracle_schedule.py`
* `scripts/hybrid_obstacle_oracle_analysis.py`
* `scripts/hybrid_obstacle_oracle_write_decision.py`
* `scripts/hybrid_obstacle_geom_distance_probe.py`
* `tests/test_oracle_reference_contract.py` — 46 tests
* `diagnostics_output/hybrid_obstacle_oracle_reference/*.json`
* `docs/HYBRID_OBSTACLE_ORACLE_REFERENCE_FINAL_DECISION.md`

Not committed: rollout H5s, `rollout.json` payloads, videos, checkpoints, Safety-CVAE
weights, canonical data, temporary renders. MolmoSpaces is unmodified. Nothing was pushed.

---

## 15. Reproduction

```bash
export MUJOCO_GL=egl PYTHONUNBUFFERED=1
export MLSPACES_ASSETS_DIR=/root/prox_learning_hybrid_safety/assets
export PYTHONPATH=/root/prox_learning_hybrid_safety/submodules/molmospaces
PY=/root/act_retrain_venv/bin/python
cd /root/prox_learning_hybrid_safety

# 1. provenance and source audit
$PY scripts/hybrid_obstacle_oracle_provenance.py \
    --out diagnostics_output/hybrid_obstacle_oracle_reference/provenance_verification.json
$PY scripts/hybrid_obstacle_oracle_source_audit.py \
    --development-manifest configs/hybrid_obstacle_controller_development4_v1.json \
    --out diagnostics_output/hybrid_obstacle_oracle_reference/current_oracle_audit.json

# 2. offline differential audit (open-loop expert replay, no policy)
$PY scripts/hybrid_obstacle_oracle_offline_signal.py \
    --development-manifest configs/hybrid_obstacle_controller_development4_v1.json \
    --collection-manifest configs/hybrid_obstacle_candidate_manifest_v2.json \
    --stack configs/hybrid_safety_stack_v1.json --safety-dir assets/safety/cvae_v3 \
    --out diagnostics_output/hybrid_obstacle_oracle_reference/offline_signal_audit.json

# 3. freeze the schedule, then execute it (fresh process per rollout)
$PY scripts/hybrid_obstacle_oracle_schedule.py \
    --development-manifest configs/hybrid_obstacle_controller_development4_v1.json \
    --baseline-root /root/act_retrain_assets/rawhead_dev_v1 \
    --output-root /root/act_retrain_assets/oracle_dev_v1 \
    --out configs/hybrid_obstacle_oracle_schedule_v1.json

cd submodules/act && $PY eval_act_obstacle_oracle.py \
    --eval-manifest ../../configs/hybrid_obstacle_controller_development4_v1.json \
    --episode-id <episode> --condition ACT_PLUS_ORACLE --repeat-index <r> \
    --collection-manifest ../../configs/hybrid_obstacle_candidate_manifest_v2.json \
    --ckpt_dir /root/act_retrain_assets/act_ckpts/hybrid_obstacle_act_baseline_v2/20260725_seed0_2000ep \
    --expected-act-checkpoint-sha256 dd7cd108a64ce10e5aab21b525dc06190f54d4e5fe446f65715b6852c49e7d36 \
    --expected-dataset-stats-sha256 c8119b904bfc80d66e3d33825722fcf9bb8bf3433c956dc09c27e6517d7c4ae2 \
    --output_dir /root/act_retrain_assets/oracle_dev_v1/<tag>

# 4. analysis, defect probe, decision, tests
cd /root/prox_learning_hybrid_safety
$PY scripts/hybrid_obstacle_geom_distance_probe.py \
    --development-manifest configs/hybrid_obstacle_controller_development4_v1.json \
    --collection-manifest configs/hybrid_obstacle_candidate_manifest_v2.json \
    --out diagnostics_output/hybrid_obstacle_oracle_reference/geom_distance_defect.json
$PY scripts/hybrid_obstacle_oracle_analysis.py \
    --schedule configs/hybrid_obstacle_oracle_schedule_v1.json \
    --development-manifest configs/hybrid_obstacle_controller_development4_v1.json \
    --offline-signal diagnostics_output/hybrid_obstacle_oracle_reference/offline_signal_audit.json \
    --compat-rollout /root/act_retrain_assets/oracle_dev_v1/cand106_act_only_compat \
    --geom-distance-defect diagnostics_output/hybrid_obstacle_oracle_reference/geom_distance_defect.json \
    --out diagnostics_output/hybrid_obstacle_oracle_reference/development_analysis.json
$PY -m pytest tests/test_oracle_reference_contract.py -q
```

---

## 16. `confirmatory41` remains untouched

| property | value |
|---|---|
| manifest | `configs/hybrid_obstacle_confirmatory41_v1.json` |
| sha256 | `7b4500e9b4b2868e2612d7e444c34762d72c5e6e7b4b7c38bcf31f027b51b69e` |
| rows | 41 (32 hazard-present + 9 hazard-absent) |
| `executed_in_this_task` | **false** |
| rows executed in this task | **0** |
| distinct episodes executed | 4, all from `development4` |

The evaluator hard-refuses any manifest whose role is `CONFIRMATORY_UNTOUCHED`, and accepts
only `DEVELOPMENT_ONLY`. Both refusals are covered by tests.

---

## 17. Next recommended task

**Develop a deployable posture-conditioned reference.** The oracle establishes that the
Safety-CVAE does carry a usable hazard-specific differential, and that the frozen constants
consume it without gross regression. The open problem is producing the parked render's
cancelling effect without privileged information.

Before any confirmatory run, characterise the **long-range direction reversal on candidate
108**: a deployable reference that reproduces it would push the arm the wrong way when the
hazard is far. A predeclared study on more rows, with the committed teacher's activation
radius as the stratifier, is the right instrument.

Do not execute `confirmatory41`, tune controller constants, or implement a deployable
reference on the strength of this report alone.

---

ORACLE_REFERENCE_VALID_CONTROLLER_VIABLE
