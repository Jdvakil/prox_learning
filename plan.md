# The core diagnosis

You are closer than it feels. You already have:

* A sensorized robot and fume-hood environment.
* A 150-episode dataset.
* A strong camera-only baseline.
* A multimodal implementation attempt.
* A useful preliminary result: **nominal ACT and your method perform similarly**.

What you lack is not an entire MVP. You lack a **decisive experiment that creates a need for proximity sensing**, and there may also be an inference-path problem.

The official ICRA 2027 deadline is **September 15, 2026 at 11:59 PST**. The complete submission, including references, is limited to eight pages. ([IEEE ICRA 2027][1])

## Three technical issues to check immediately

### 1. Is proximity available to the deployed policy?

In ACT, the CVAE encoder is used during training to infer the latent style variable and is discarded at test time. The deployed policy is the CVAE decoder conditioned on current observations. Therefore, if proximity appears only in ACT’s CVAE encoder, the robot cannot use live proximity during execution. ([ar5iv][2])

The sensor signal must enter the **policy observation path at every inference query**, alongside images and joint state.

### 2. Are you using measured proximity or predicted proximity?

If your ACT policy receives proximity predicted from the camera, then it is still fundamentally a camera-only policy. Predicted proximity may improve representation learning, but it introduces no new environmental measurement at inference.

To demonstrate that proximity sensors are a useful modality, you need to compare:

* Camera-only ACT.
* Camera plus **camera-predicted** proximity.
* Camera plus **live measured** proximity.

Your existing proximity-prediction CVAE is not wasted. It becomes a very useful diagnostic and baseline.

### 3. Can action chunking react quickly enough?

A near-field sensor is valuable because it provides rapidly changing local feedback. If your implementation predicts a long action chunk and executes it without interruption, the sensor may arrive too late to affect behavior. Recent visual-tactile work has explicitly identified limited responsiveness during chunk execution and introduced slow-fast policies to address it. ([Robotics Proceedings][3])

Audit whether you:

* Query ACT at every control step or only once per chunk.
* Use temporal ensembling.
* Can interrupt or shorten a chunk when the robot enters the sensor’s near range.

---

# Recommended paper narrative

## Working title

**When Does Proximity Help? Characterizing Pre-Contact Sensing for Imitation-Learned Manipulation in Constrained Workspaces**

A more method-oriented alternative, if your new model produces a clear gain:

**Proximity-Conditioned Action Chunking for Collision-Aware Manipulation under Visual Occlusion**

Avoid putting “safe” in the title unless you provide formal safety guarantees. “Collision-aware,” “risk-sensitive,” or “safer in our benchmark” is easier to defend.

## One-sentence thesis

> Cameras provide global scene context, while proximity sensors provide local pre-contact geometry. Proximity is therefore redundant in visually clear, wide-clearance tasks, but becomes valuable during near-contact manipulation under occlusion, appearance ambiguity, and narrow clearances.

That makes your current equal result part of the story rather than an embarrassment:

> In nominal settings, the camera already contains enough information, so adding proximity produces no material improvement. The benefit emerges specifically when near-field geometry is difficult or impossible to infer visually.

This is more scientifically interesting than claiming that adding another sensor always improves performance.

## Proposed introduction arc

1. Vision-based imitation policies are capable in structured manipulation.
2. Near contact, the end effector, object, fixture, or enclosure can obscure task-relevant geometry.
3. Contact sensing is reactive: it tells the robot after interaction begins. Proximity sensing can provide local information before contact.
4. However, an additional modality should not help when its measurements are already predictable from vision.
5. We therefore ask **when proximity supplies nonredundant information**, rather than whether it universally improves average task success.
6. We introduce a proximity-conditioned ACT policy and a controlled evaluation varying visual observability and geometric clearance.
7. We show nominal parity but improved collision-free performance in sensor-relevant regimes, and use prediction and masking studies to explain where the improvement originates.

Generic “vision plus touch beats vision” is already a crowded claim: recent multimodal manipulation papers have reported substantial gains from visual-tactile fusion. Your distinctive angle should be **non-contact pre-touch sensing, controlled sensor-utility characterization, constrained workspaces, and closed-loop use within action chunking**. ([arXiv][4])

## Proposed contributions

1. **A controlled sensor-utility benchmark** for manipulation in a constrained fume-hood environment, varying visibility and clearance independently.
2. **Proximity-Conditioned ACT**, which conditions the deployed policy on a short history of live proximity measurements and optionally increases replanning near obstacles.
3. **A characterization of when proximity helps**, using measured-versus-predicted proximity, sensor masking, and phase-specific analysis.
4. **Operational safety results**, using collision-free success and independently measured contact or clearance metrics.

---

# Research questions and hypotheses

## Main research question

> Under what combinations of visual observability and geometric clearance does live proximity feedback improve the task success and operational safety of an action-chunking imitation policy?

## Supporting questions

**RQ1 — Information:**
Does measured proximity improve near-field state or collision-risk prediction beyond what can be inferred from camera images and robot state?

**RQ2 — Control:**
Does conditioning ACT on live proximity improve collision-free success under occlusion, tight clearances, or unseen obstacle configurations?

**RQ3 — Integration:**
Is direct closed-loop proximity conditioning more effective than using proximity only as auxiliary supervision or a predicted latent representation?

**RQ4 — Localization:**
During which part of the task does proximity matter: global reaching, final approach, contact initiation, or object transport?

## Testable hypotheses

**H1: Nominal parity.**
When visibility is high and clearance is generous, camera-only ACT and proximity-conditioned ACT perform similarly.

**H2: Condition-dependent benefit.**
The advantage of measured proximity increases as visibility and clearance decrease.

**H3: Measured beats predicted under ambiguity.**
Camera-predicted proximity and measured proximity perform similarly in nominal conditions, but measured proximity performs better under visual occlusion or hidden geometry.

**H4: Causal sensor use.**
Shuffling, delaying, or masking proximity at inference eliminates most of the hard-condition improvement.

**H5: Phase-specific benefit.**
Masking proximity during the final approach hurts performance more than masking it during the early, global reaching phase.

---

# Minimum viable method

Call the minimum method **P-ACT**, for Proximity-Conditioned ACT.

At time (t), use:

[
o_t = {I_t,\ q_t,\ p_{t-H:t},\ \Delta p_{t-H:t},\ m_{t-H:t}},
]

where:

* (I_t) is the camera observation.
* (q_t) is robot state.
* (p_{t-H:t}) is a short history of live proximity readings.
* (\Delta p) captures whether distance is closing or opening.
* (m) is a validity or out-of-range mask.

Use a small temporal MLP or one-dimensional convolution to turn the sensor history into one or several tokens:

[
e^p_t=f_\phi(p_{t-H:t},\Delta p_{t-H:t},m_{t-H:t}).
]

Append those tokens to the **deployed ACT policy’s Transformer observation stream**:

[
\hat a_{t:t+k} =
\pi_\theta(I_t,q_t,e^p_t,z).
]

The proximity tokens must be present at inference. Do not put environmental information solely in ACT’s training-only posterior encoder.

## Keep the method small

For the MVP:

* Use a history covering approximately 0.25–0.5 seconds.
* Normalize each sensor separately.
* Include a valid-measurement mask.
* Keep the ACT image encoder, optimizer, dataset, and action representation unchanged.
* Requery every control step if computationally possible.
* If actions are executed open-loop, interrupt or shorten the chunk when proximity crosses a predefined near-field threshold.

Do not begin with a large fusion architecture search. First establish that direct live conditioning works.

## Reuse your proximity CVAE

Your existing proximity predictor can support the central characterization.

Define camera-based proximity prediction error for condition (c):

[
R(c)=\mathbb{E}_{c}\left[\lVert p_t-\hat p_t(I_t,q_t)\rVert\right].
]

Define sensor policy utility:

[
U(c)=
\mathrm{CFS}*{V+P}(c)-\mathrm{CFS}*{V}(c),
]

where CFS is collision-free success.

Your most interesting analysis may be a plot showing that conditions with high (R(c))—where proximity is poorly inferable from vision—also have high (U(c)).

That would explain both results:

* Low prediction error in nominal conditions → little policy improvement.
* High prediction error under occlusion → measured sensor becomes useful.

Treat this as an empirical relationship, not as a formal information-theoretic proof.

## Optional method extension

Only add this after direct fusion works:

**Future-clearance auxiliary head**

Predict either:

* Minimum proximity over the predicted action chunk.
* Probability of a prohibited contact in the next (k) steps.
* Time to enter the near-field threshold.

Use:

[
\mathcal L =
\mathcal L_{\mathrm{ACT}}
+\lambda\mathcal L_{\mathrm{clearance}}.
]

This encourages the representation to encode actionable sensor dynamics.

A second optional extension is **proximity-triggered replanning**: use normal chunking in free space but increase query frequency or reduce effective chunk length near obstacles.

---

# The decisive experiment

Use a simple factorial experiment rather than several unrelated tasks.

## Experimental axes

### Visual observability

1. **High visibility:** nominal camera view.
2. **Low visibility:** realistic target, fixture, or side-wall occlusion.

The low-visibility condition should model a plausible deployment issue, such as gripper self-occlusion, enclosure geometry, glare, or an object hidden behind the manipulated item. Avoid arbitrary full-image blackouts unless camera loss is a realistic failure mode.

### Geometric clearance

1. **Wide clearance:** nominal workspace.
2. **Tight clearance:** obstacle or enclosure wall placed within the useful proximity-sensor range.

This gives four cells:

| Condition | Visibility | Clearance | Expected result               |
| --------- | ---------: | --------: | ----------------------------- |
| A         |       High |      Wide | Similar performance           |
| B         |        Low |      Wide | Small or moderate sensor gain |
| C         |       High |     Tight | Moderate sensor gain          |
| D         |        Low |     Tight | Largest sensor gain           |

The central paper figure should be a heatmap of the performance difference between measured-proximity and camera-only policies across this grid.

## Policies to compare

| Policy             | Inputs                                    | What it tests                                  |
| ------------------ | ----------------------------------------- | ---------------------------------------------- |
| **V-ACT**          | Camera and robot state                    | Main baseline                                  |
| **V+(\hat P)-ACT** | Camera, state, camera-predicted proximity | Whether auxiliary representation helps         |
| **V+P-ACT**        | Camera, state, live measured proximity    | Whether the physical modality adds information |
| **V+P-shuffled**   | Same model, wrong proximity at inference  | Whether the model causally uses the sensor     |

Optional safety baseline:

| Policy             | Description                                                             |
| ------------------ | ----------------------------------------------------------------------- |
| **V-ACT + shield** | Camera-only ACT with threshold-based slowing or stopping from proximity |

The shield separates two questions:

1. Is the sensor useful at all?
2. Does learning use it better than a simple reactive rule?

## Primary metrics

### Collision-free success

[
\mathrm{CFS}
============

\mathbf{1}[
\text{task completed}
\land
\text{no prohibited contact}
].
]

This should be the headline metric because ordinary task success can hide unsafe trajectories.

### Prohibited-contact rate

Report the percentage of rollouts containing contact with:

* Fume-hood walls.
* Fixtures.
* Non-target objects.
* Fragile or restricted regions.

Define prohibited contacts before the final evaluation.

### Secondary metrics

* Task success.
* Minimum ground-truth clearance.
* Number of contact events.
* End-effector speed at first contact or closest approach.
* Time spent inside a near-collision region.
* Completion time.
* Emergency interventions or stops.
* Failure mode: collision, misalignment, drop, timeout, or wrong grasp.

Do not use the same proximity signal as both the policy input and the sole ground truth for safety. In simulation, use collision geometry or signed distance. In real experiments, use contact sensing, force, external tracking, or blinded video annotation.

If the sensors cover only the gripper, restrict the claim to **near-field end-effector collision risk**, not whole-arm or human safety.

## Evaluation protocol

* Use the same demonstrations and hyperparameters for V-ACT and V+P-ACT.
* Use identical randomized initial states for paired policy comparisons.
* Start with ten paired rollouts per cell for the pilot.
* For final results, target approximately 20–30 paired rollouts per method and condition where practical.
* Train multiple random seeds if training time allows; three is a useful target.
* Report effect sizes and 95% confidence intervals, not just percentages.
* For binary paired outcomes, use McNemar’s test or a logistic model with policy, condition, and their interaction.
* For clearance and completion time, use paired bootstrap intervals.
* The most important statistical result is the **policy × condition interaction**: proximity should help more in low-visibility or tight-clearance settings.

---

# Audit the existing 150 episodes before collecting more

Do not immediately collect another large nominal dataset.

Produce these plots first:

1. Distribution of each proximity sensor’s readings.
2. Distribution of minimum proximity over all timesteps.
3. Fraction of frames where at least one sensor is within its informative range.
4. Number of far-to-near transitions.
5. Sensor reading versus end-effector motion.
6. Camera-to-sensor and action-to-sensor timing offsets.
7. Proximity-prediction error by task phase.
8. Expert actions conditioned on similar images but different proximity values.

The fundamental requirement is that your dataset contain examples where:

> Two states look similar from the camera but have different nearby geometry and therefore require different safe actions.

If these paired situations do not occur, the network has no reason to use proximity.

## Targeted data collection

If near-field or corrective states are sparse, collect a smaller set of targeted demonstrations rather than more nominal demonstrations:

* Start close to walls or fixtures.
* Randomize hidden obstacle positions.
* Vary clearance while keeping global appearance similar.
* Include successful corrections from near-collision states.
* Run the camera-only policy, intervene before a collision, and demonstrate recovery.
* Oversample training windows near the sensor’s operating range.

Thirty carefully targeted corrective episodes may be more valuable than another hundred nominal successes.

If this is a simulation-to-real project, characterize the physical sensor’s noise, range, latency, invalid returns, and relevant surface/angle failures, then reproduce those effects in simulation. Recent learning-based proximity work similarly treats sensor noise and latency characterization as a central part of the system evaluation. ([arXiv][5])

---

# Your 48-hour MVP

This is what should exist before doing more method research.

## 1. Verify the deployment path

Log the live sensor vector immediately before the ACT forward pass during rollout.

Confirm that changing the measured sensor changes the observation tensor sent to the deployed policy.

## 2. Run an action-sensitivity unit test

Take a fixed near-obstacle image and robot state. Replace proximity with several values taken from real dataset examples.

Plot:

[
\left|
\hat a^{P_1}_{t}
----------------

\hat a^{P_2}_{t}
\right|.
]

If the predicted first action does not materially change, the policy is ignoring the sensor or the sensor is not in the inference path.

This is a debugging test, not a final scientific result.

## 3. Plot data coverage

Generate the sensor histograms and identify whether demonstrations contain meaningful near-field behavior.

## 4. Implement the four evaluation cells

You need only the current task, with:

* Normal versus occluded view.
* Wide versus tight clearance.

## 5. Run the first pilot

Run V-ACT and direct V+P-ACT on ten paired configurations in each cell.

The first advisor-facing result should be a single plot containing:

* Collision-free success.
* Collision rate.
* Four conditions.
* Two policies.

That is your paper MVP.

---

# Heilmeier Catechism

DARPA’s Heilmeier Catechism asks researchers to define the objective, current practice, novelty, impact, risks, cost, time, and measurable exams. ([darpa.mil][6])

## 1. What are you trying to do?

We want a robot working inside a constrained enclosure to notice nearby surfaces before hitting them and use that information to complete manipulation tasks with fewer collisions when the camera view is poor.

## 2. How is it done today, and what are the limits?

Learning-based manipulation policies commonly use cameras and robot joint state. Cameras give broad information about objects and the scene, but near the end effector the target or surrounding geometry can be hidden by the gripper, object, enclosure, or lighting effects.

Camera-only policies may still succeed in nominal configurations, but they lack a direct measurement of near-field clearance. Contact or force sensing can identify interaction after contact begins, whereas proximity can provide information before contact.

## 3. What is new, and why should it work?

We will condition an action-chunking imitation policy directly on a short history of live proximity measurements and evaluate it over controlled levels of visual observability and geometric clearance.

The new element is not simply attaching another sensor. It is:

* Identifying when proximity contains information that vision cannot recover.
* Delivering that information to the policy during deployment.
* Measuring whether the policy uses it causally.
* Characterizing where in the task and under which conditions it improves performance.

It should work because nearby geometry directly changes proximity readings even when the corresponding obstacle is visually hidden or difficult to estimate.

## 4. Who cares, and what difference would success make?

The result matters for learned manipulation in fume hoods, gloveboxes, cabinets, shelves, industrial cells, and other enclosed workspaces where collisions may damage equipment, disturb objects, or interrupt an automated process.

A successful study would provide both:

* A practical method for adding low-dimensional pre-contact feedback to imitation policies.
* A design rule explaining when the additional sensor is worth its hardware and integration cost.

## 5. What are the risks?

The main risks are:

* The task is fully observable from the camera, making proximity redundant.
* The policy ignores the low-dimensional modality.
* The training data contains too few corrective near-field actions.
* Long action chunks prevent fast reactions.
* Sensor placement or range does not cover the relevant collision direction.
* Sensor readings are noisy, delayed, or surface-dependent.
* Apparent safety gains are artifacts of using the sensor itself as the evaluation metric.
* Results apply only to one task or simulated setting.

Each risk has a corresponding test: controlled occlusion, causal sensor masking, targeted data, chunk-interruption experiments, sensor characterization, independent collision ground truth, and at least one second task or physical check if time permits.

## 6. How much will it cost?

Most hardware and environment costs are already sunk.

The remaining cost is primarily:

* One direct-fusion implementation.
* Up to 30–50 targeted demonstrations if coverage is insufficient.
* Two or three core policy variants.
* Multiple training seeds.
* Paired evaluation rollouts.
* Approximately one week of writing and figure preparation.

Cut extra architectures and additional unrelated tasks before cutting the controlled visibility-clearance experiment.

## 7. How long will it take?

The remaining work is structured as:

* One week to verify the method and obtain a pilot.
* One week for targeted data and final training.
* One week for full evaluation and ablations.
* One week for analysis, writing, revision, and submission.

## 8. What are the midterm and final exams?

### Midterm exam: approximately August 22

Pass if:

* Live measured proximity is verified in the deployed policy path.
* Changing proximity changes the predicted action in near-field states.
* The dataset contains, or has been augmented with, meaningful near-field corrective behavior.
* A four-condition pilot shows either reduced collisions or improved risk prediction in at least one sensor-relevant condition.

### Final exam

Internal planning targets:

* Nominal success remains within approximately five percentage points of camera-only ACT.
* Collision-free success improves by at least 15 percentage points in one or more hard conditions, or prohibited-contact rate falls by at least 30–50% relatively.
* The effect appears in more than one randomized configuration.
* Shuffling or masking the sensor eliminates a substantial portion of the improvement.
* The benefit is concentrated in the final approach or tight-clearance phase.
* Results use independent collision or clearance measurements.

These are go/no-go targets for project management, not numbers to force through selective evaluation.

---

# Thirty-day execution plan

## August 15–17: lock the question and audit the system

* Freeze the main research question.
* Verify the inference path.
* Check sensor synchronization, normalization, validity masks, and update rate.
* Determine ACT query frequency and whether chunks can be interrupted.
* Produce dataset-coverage plots.
* Define prohibited contact and collision-free success.

**Deliverable:** one-page study design and an action-sensitivity test.

## August 18–22: build the paper MVP

* Implement direct V+P-ACT.
* Train one seed.
* Build the visibility × clearance evaluation grid.
* Run ten paired trials per cell for V-ACT and V+P-ACT.
* Analyze why failures occur.

**Decision gate:** continue with the method only if the sensor is live, causally influences actions, and shows a plausible hard-condition benefit.

## August 23–27: repair data coverage

* Collect targeted near-field and recovery demonstrations only where needed.
* Retrain V-ACT and V+P-ACT on identical data.
* Train the V+(\hat P) baseline using your existing predictor.
* Train additional random seeds.
* Add a simple safety shield baseline if collision reduction is central.

**Freeze the architecture by August 27.**

## August 28–September 3: complete the main evaluation

* Run the full paired evaluation.
* Capture videos during the same rollouts.
* Record failure categories.
* Calculate collision-free success, contact rate, clearance, and completion time.
* Run the measured-versus-predicted proximity comparison.
* Plot sensor prediction error versus policy benefit.

## September 4–6: ablations and statistics

Prioritize:

1. Sensor shuffled at inference.
2. Sensor masked only during final approach.
3. No sensor history versus short history.
4. Nominal versus low-visibility condition.
5. Sensor latency or noise ablation.

Do not perform a large fusion-architecture sweep.

**Freeze results by September 6.**

## September 7–10: full manuscript and video

* Write the entire paper, including limitations.
* Produce final system, architecture, heatmap, and rollout-trace figures.
* Prepare the accompanying video.
* Obtain a complete advisor review.

The first ICRA video-upload window closes September 9; a second window is scheduled for September 17–22. ([IEEE ICRA 2027][1])

## September 11–13: revise

* Address advisor feedback.
* Rerun only broken, missing, or clearly underpowered experimental cells.
* Remove unsupported claims.
* Verify that every plot can be reconstructed from saved logs.

## September 14

* Double-anonymous check.
* Eight-page check, including references.
* PDF compliance check.
* Final proofread.
* Upload a submission-quality version rather than waiting for the final hour.

## September 15

Submit well before the official 11:59 PST deadline. ([IEEE ICRA 2027][1])

---

# Fallback narratives

## Plan A: strong policy improvement

**Claim:** Live proximity conditioning improves collision-free success in visually ambiguous, tight-clearance manipulation.

This supports the method-oriented P-ACT title.

## Plan B: nominal parity, hard-condition improvement

**Claim:** Proximity is not universally beneficial; its value is determined by visual observability and clearance.

This is probably the most defensible and interesting outcome. Your current result already supports the first half.

## Plan C: learned policy shows little gain, but a proximity shield works

**Claim:** Imitation policies trained only on successful nominal demonstrations do not reliably learn collision avoidance from sparse proximity signals, but a simple sensor-based runtime layer reduces prohibited contacts without sacrificing task success.

Compare:

* V-ACT.
* V-ACT plus shield.
* Learned V+P-ACT.

This becomes a study of learned versus explicit use of local sensing.

## Plan D: proximity does not help anywhere

First determine whether:

* Sensor geometry is wrong.
* Sensor range does not cover the decision point.
* The task is visually fully observable.
* The demonstrations never require sensor-conditioned decisions.

Do not manufacture a positive claim. A systematic negative characterization may still be useful, but it carries the highest submission risk unless it produces a strong general design insight.

---

# Recommended paper figures

**Figure 1 — System and motivation**
Fume-hood environment, sensor placement, camera field of view, and an example of geometry hidden from the camera but visible to proximity.

**Figure 2 — P-ACT architecture**
Clearly distinguish the training-only CVAE encoder from the deployed policy observation path.

**Figure 3 — Sensor-utility heatmap**
Improvement in collision-free success across visibility × clearance.

**Figure 4 — Why the sensor helps**
Proximity-prediction error versus policy improvement, plus a rollout trace showing sensor reading, clearance, action, and collision event.

**Table 1 — Main comparison**
V-ACT, V+(\hat P)-ACT, and V+P-ACT across conditions.

**Table 2 — Causal ablations**
Shuffled, delayed, near-phase masked, and no-history proximity.

---

# Abstract scaffold

> Vision-based imitation policies can perform constrained manipulation reliably when task geometry remains visible, but their observations become ambiguous near contact due to self-occlusion and narrow clearances. We investigate when local non-contact proximity sensing provides information that cannot be recovered from camera observations alone. We introduce Proximity-Conditioned ACT, an action-chunking policy that conditions each inference query on a short history of measured proximity signals. Using a fume-hood manipulation environment, we compare camera-only, camera-predicted-proximity, and measured-proximity policies over controlled variations in visual observability and geometric clearance. **[Insert main quantitative result.]** In nominal conditions, proximity provides little improvement, whereas under **[hard conditions]** it improves collision-free success by **[X]** and reduces prohibited contacts by **[Y]**. Sensor-shuffling and phase-specific masking show that the improvement depends on live proximity during the final approach. These results characterize when pre-contact sensing is worth adding to learned manipulation systems and provide practical guidance for multimodal policy design.

---

# A concise update for your advisor

> I have completed the sensorized environment, collected 150 demonstrations, trained the ACT baseline, and obtained an initial multimodal result. The current nominal evaluation shows parity, which suggests that proximity is redundant when the task is fully visually observable. I also identified a potential architecture issue: proximity must enter ACT’s deployed policy path rather than only its training-time CVAE encoder. I am reframing the paper around when proximity provides nonredundant pre-contact information. By August 22, I will deliver a direct live-sensor ACT baseline and a paired visibility-by-clearance pilot reporting collision-free success and sensor ablations.

One submission-policy detail: ICRA 2027 requires disclosure when AI-generated content is used directly in an article, while ordinary editing and grammar enhancement are generally treated differently. Use this response as research-planning material, rewrite manuscript language in your own voice, and follow the conference’s disclosure instructions for any generated content that is retained. ([IEEE ICRA 2027][1])

The next deliverable is not another complicated model. It is a four-cell pilot proving that the proximity measurement is **live, informative, and causally used**.

[1]: https://2027.ieee-icra.org/contribute/call-for-icra-2027-papers-now-accepting-submissions/ "IEEE ICRA 2027 | Call for Technical Papers"
[2]: https://ar5iv.labs.arxiv.org/html/2304.13705?utm_source=chatgpt.com "[2304.13705] Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware"
[3]: https://www.roboticsproceedings.org/rss21/p052.html?utm_source=chatgpt.com "Robotics: Science and Systems XXI - Online Proceedings"
[4]: https://arxiv.org/abs/2512.09851?utm_source=chatgpt.com "[2512.09851] Simultaneous Tactile-Visual Perception for Learning Multimodal Robot Manipulation"
[5]: https://arxiv.org/abs/2606.31912?utm_source=chatgpt.com "[2606.31912] Learning Locomotion on Discrete Terrain via Minimal Proximity Sensing"
[6]: https://www.darpa.mil/about/heilmeier-catechism "The Heilmeier Catechism"
