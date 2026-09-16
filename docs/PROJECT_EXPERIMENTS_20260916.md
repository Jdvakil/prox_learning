# Proximity Learning / PC-ACT: Complete Recoverable Experiment Record

**Inventory date: 16 September 2026.** This document follows the numbered narrative, results, evidence, limitations, and appendix structure of the project's [PAPER.md][paper]. It records completed experiments, negative results, partial evaluations, environment qualification, and analyses of existing rollouts. Proposed experiments are listed separately at the end.

**Scope:** the available project records, including this worktree, other local project worktrees and artifact roots, and the public `main` branch at commit `a5de7a4dd42a075d99b8f5865ec7baa9479a1fdd`. This is a project inventory, not an attribution of every run to one researcher. Some historical experiments were performed by collaborators. It is not possible to recover undocumented experiments or missing collaborator artifacts from these sources.

**Evidence standard:** saved evaluation JSONs and final audited reports take precedence over older progress notes. Remote-only results are identified explicitly. An archived summary can establish what was reported without establishing that its demonstrations, checkpoint, or raw trajectories are still available. No training or simulation was run to prepare this document.

**Reading guide:** §7 contains policy experiments and their results; §4 contains sensor/representation measurements; Appendix A contains the environment-development experiments; Appendix B records configurations and naming; Appendix C explains statistics; Appendix D indexes evidence and existing figures; Appendix E defines terms; Appendix F lists work without recovered results.

---

## 1. The story in one page

The project asks whether body-mounted proximity measurements help learned manipulation, and what representation and training conditions make that information useful. The record contains several distinct answers, rather than one uniformly positive result.

| Experiment family | Main observation | Scope of the evidence |
|---|---|---|
| Latest local V10.10/V10.10c placement, three seeds | Finetuned 128-D PACT: **83/150 task successes, 63/150 collision-free task successes**, versus ACT **64/150 and 46/150** | Best current local placement comparison; exposed regression scenes, not a fresh generalization test |
| Historical hallway, H-A | Readout hazard contact **6/50**, ACT **17/50**, scalar PACT **18/50** | Separate 152-demo, chunk-50 experiment; one trained checkpoint per method; unmatched scene realizations |
| Frozen 32-D pickup contact study | Live PACT hazard contact **42/300**, ACT **67/300**, unrelated-frame control **70/300** | Three seeds; strongest recovered matched intervention study, but not the current 128-D model |
| Frozen 32-D geometry transfer | Live proximity retains a contact advantage over the unrelated-frame control under two shifts | No ACT arm; task performance falls on the shifts |
| Vision degradation | Proximity reduces contact in several conditions, but task completion collapses under complete RGB replacement | Lower contact does not establish successful manipulation without vision |
| Placement transfer/development | Several harder-clutter comparisons are negative or inconclusive | Spaced-bench, V10.9, V10.11c, and the aligned dual-camera pilot belong in the record |
| Completion time | On 50 scenes where both latest policies succeed, PACT averages **29.73 s**, ACT **31.24 s** | Reanalysis of existing rollouts; conditional on shared success, not inference-speed measurement |
| Hybrid ACT + Safety-CVAE | Several residual controllers fail offline or live qualification; one bounded 20-rollout version passes development | Different architecture; no recovered confirmatory41 evaluation |

There is **no valid grand total** from adding every table: some analyses reuse entire rollout records, some evaluate the same checkpoint repeatedly, and some development runs repeat four scenes many times. The inventory preserves these relationships.

## 2. Introduction: how the experiments developed

Early open-table and fridge studies measured sensor activation, target visibility, and whether the skin observed information absent from the cameras. They exposed an important distinction: seeing nearby walls is different from seeing the manipulated object. Increasing sensor activation alone did not yield better grasping.

The obstacle studies then tested raw proximity, a Safety-CVAE-derived representation, and learned surface representations. Some early policies barely reacted to proximity removal. A frozen 32-D representation produced a promising screen, but its large advantage over ACT did not replicate on the second training seed. Larger matched studies subsequently found a more consistent reduction in hazardous contact than an increase in task success.

The placement program developed collision-audited experts, clutter layouts, and static pendants before collecting multiple datasets. The current jointly finetuned 128-D result improves task and strict success on the local three-seed V10.10 comparison. Other placement environments remain difficult, and neither those improvements nor the historical hallway results establish arbitrary-layout or new-task generalization.

## 3. System overview: distinguish the methods before comparing results

| Name used here | Proximity pathway | Experiments using it |
|---|---|---|
| ACT | RGB and robot state | Baseline throughout, with different cameras/controllers across studies |
| Archived obstacle raw PACT | Global proximity values projected into tokens | O-INV and early obstacle work |
| Archived obstacle trunk PACT | Features from the Safety-CVAE pathway | O-INV negative trunk control |
| Hallway scalar/raw PACT | Peak-closeness scalars, 50 cm range; historical implementation has expanded tokenization | H-A, H-B, T-107, T-1011d |
| Local 3-D PACT | Learned surface-coordinate features | Original pickup R2 |
| Local frozen 32-D PACT | Frozen per-sensor learned embedding | Pickup screen/ablation/contact/geometry/blur/blind; earlier local placement |
| PACT-128D-finetune | Per-sensor temporal CLS readouts; encoder optimized jointly with ACT | Latest local V10.10c; a separate historical implementation in H-A/H-B |
| ACT + Safety-CVAE residual | External controller adds a proximity-derived joint correction to ACT | Hybrid-controller studies only |

The 40-sensor skin has sensors on links 1–6; it does not instrument the hand, fingers, or link 7. Forty is a particular skin layout, not a sensor-count optimum established by these experiments. A completed 40-versus-20 experiment was not recovered.

## 4. What the skin makes available: measured geometry and representation studies

These are measurements or offline learning studies. They are not additional policy-success trials.

| ID / experiment | Setup and result | Interpretation / evidence |
|---|---|---|
| S01 — early episode instrumentation | A 29-sensor episode has 270 steps, final task success, and 3 environment-contact steps; `fail_any=true` also appears | Demonstrates why final outcome and transient flags need separate interpretation. [Episode report](../diagnostics_output/pact_house11_traj0/report.md) |
| S02 — initial collection smoke and partial collection | Retained summaries cover 36 smoke trajectories and 68 partial-medium trajectories | Data/observation characterization, not ACT-versus-PACT evidence. [Smoke](../diagnostics_output/pilot_skin_smoke_v1/summary.json), [partial](../diagnostics_output/medium_v1_partial/summary.json) |
| S03 — proximity activation audit, initial and expanded | Initial: 280 trajectories, 66 houses, 72,651 frames. Expanded: 343 trajectories, 82 houses, 89,029 frames. At 0.5 m, expanded link activation is 99.7% / 79.2% / 43.2% / 62.9% for links 2/3/5/6 | Link-level activation is not the percentage of the robot surface covered. Older sensor contract; expanded corpus overlaps earlier audit. [Initial](../diagnostics_output/proximity_audit_v1/report.md), [expanded](../diagnostics_output/proximity_audit_medium_full/report.md) |
| S04 — open-table necessity pilot | 16 trajectories: vision-blind fraction 1.08%, hard necessity fraction 0.151%. Six-trajectory PACT-data audit: both fractions 0% | These target-visibility measurements motivated confined scenes; they do not establish a universal zero benefit from proximity. [Pilot](../diagnostics_output/prox_necessity_pilot/necessity_summary.json), [PACT](../diagnostics_output/prox_necessity_pact/necessity_summary.json) |
| S05 — synthetic reconstruction | Empty-room mean absolute surface errors 39.51–42.70 mm; flat-plane mean absolute z error 23.32 mm, p99 49.21 mm | Specific older reconstruction pipeline. Do not replace these with the unrelated historical 5.6 mm pipe claim. [Summary](../synthetic_verify/summary.md) |
| S06 — physical sensor verification | Plate check passes 38/40 sensors; two link-5 front sensors return nearer self geometry | Geometry/visibility issue, not proof that all 40 sensors measure arbitrary hazards. [Audit](HYBRID_SAFETY_STACK_FINAL_DECISION.md) |
| S07 — historical direction coverage / pipe / lighting characterization | Remote notes report 83% skin versus 10% wrist directional coverage, 5.6 mm pipe error, and bit-identical skin readings under blur/darkness | **Reported, not fully traceable here.** `PAPER.md` explicitly lacks the producing coverage script. Do not publish as verified collision-surface coverage or current encoder accuracy. [Remote paper][paper], [remote research log][readme] |
| S08 — Safety-CVAE training variants | Remote record gives MSE approximately 0.011 / 0.015 / 0.009 for v1/v2/v3. Honest-split direction cosine 0.924 versus easier split 0.926; posture lookup 0.923; near-obstacle correction magnitude only about 64% of teacher | Offline reflex prediction is not closed-loop safety. [Remote research log][readme]; canonical v3 audit in [local Safety-CVAE record](../diagnostics_output/hybrid_safety_stack/safety_cvae_audit.json) |
| S09 — deflection / bar-presence probes | v1 deflection scores around 0.50–0.56; v2 raw deflection around 0.75 with a qpos control, but bar-presence remains around 0.40–0.52 | The formal presence probe fails. Decoding a swerve is not proof of hazard detection or policy use. [Remote research log][readme] |
| S10 — obstacle demonstration audit | 151 recorded episodes, 113 with bar; 49/113 bar episodes deflect (43.4%). Remote analysis also reports frequent inbound scrapes and proximity activation in no-bar episodes | Expert trajectories and background geometry can confound proximity features. [Saved analysis](../diagnostics_output/obstacle_analysis/summary.json), [remote log][readme] |
| S11 — hallway range/encoder characterization | At 20/50 cm, about 11%/40% of sensor tiles activate; one sensor is continuously active. Historical encoder valid-target error 20.62 mm, validity F1 1.0, reconstruction precision/recall 0.874/0.953 | Particular historical pretrained encoder, not the later local 32-D or finetuned encoder. [Remote paper][paper] |
| S12 — local surface encoder comparison | Frozen 32-D encoder: mean/median error 3.20/1.69 cm, 52.9% within 2 cm; validity precision/recall 99.4%/99.9%. Earlier 3-D: 3.26/1.88 cm, 51.3% within 2 cm | Representation validation does not measure policy success. [Frontend decision](PACT_FRONTEND_SCREEN_DECISION.md) |
| S13 — ACT data equality and zero support | All 255 episodes match on non-proximity observations/actions and 199/56 split. Exact-zero tokens: 95.0% for old 3-D, **0/1,247,040** for 32-D | Zeroing the 32-D pathway is outside its observed training support. [Audit](PACT_ACT_DATA_EQUIVALENCE_AND_ZERO_SUPPORT.md) |
| S14 — V9.5 skin visibility and counterfactual replay | Eight variants: inbound vessel has zero visibility in 7/8; panel visible in all eight. Physics-clean raw-admission result is **0/6**, correcting the earlier 1/8 headline | The sole nominal passing variant had a colliding source trajectory. Does not measure all later clutter instances. [Visibility](PACT_PLACE_SKIN_VISIBILITY.md), [correction](PACT_PLACE_V95_LOW_WALL_STATUS.md) |
| S15 — sensor resolving-power validation | Eight variants, 24 role comparisons: predicted/measured changed values Pearson r=0.99997, Spearman r=1.0; responding-sensor recall 1.0 | Retrodiction on measured geometry, followed by the siting experiments in Appendix A. [Status](PACT_PLACE_V96_CLUSTER_SITING_STATUS.md) |
| S16 — sensor point-cloud visualization | Remote paper reconstructs 384 points per snapshot from six link-6 sensors across 106 approach steps | Descriptive geometry visualization; no new task-success denominator. [Remote paper][paper] |
| S17 — v12 RGB framebuffer repeatability | Twenty repeat renders per configuration: dithering on/off yields 7/6 unique images; single-sample framebuffer yields one image per camera. Two live reference/optimized pairs agree for 101 observations, with zero arm/gripper/qpos differences | Completed renderer/evaluator-equivalence diagnostic, not a task-success benchmark. [Remote audit](https://github.com/Jdvakil/prox_learning/blob/a5de7a4dd42a075d99b8f5865ec7baa9479a1fdd/reports/diagnostics/v12_rgb_framebuffer_audit.json) |

## 5. Method: what was changed in the experiments

### 5.1 Tasks and demonstrations

The record includes open-table pick-and-place, a two-level fridge, obstacle pickup/extraction, and multiple fumehood placement environments. Collection success, accepted demonstrations, training/validation split sizes, and evaluation rollouts are different quantities. For example, the original local V10.10 collected 144 accepted demonstrations from 313 attempts; the later wrist280 comparison used 280 demonstrations split 240/40.

### 5.2 Representation and modality interventions

The recovered local `PACT_PERMUTED` control supplies **whole real 40×32 frames from unrelated training episodes/timesteps**. It destroys correspondence with the current scene and temporal coherence. Despite its name, it is **not an isolated permutation of sensor identities**. The live and intervention arms use the same trained policy.

`PACT_ZERO` is retained as a failure probe. For the frozen 32-D model it is not a validated in-distribution null measurement. The local frozen-versus-finetuned comparison changes feature dimensionality and pooling as well as encoder optimization; it cannot isolate the causal effect of unfreezing alone.

### 5.3 Controller and history

Historical hallway runs use chunk-50 open-loop execution and horizon 800. The latest local placement comparison uses chunk 100, a history-100 temporal ensemble, prediction every control step, and horizon 900. Historical remote evaluation has documented query-spaced proximity history and single-subframe preprocessing mismatches; those limitations must not be silently attached to the later local implementation, which maintains control-frequency history.

### 5.4 Source selection and retention

Historical results include best-validation checkpoints, while the latest local three-method result uses final policy checkpoints and corresponding finetuned encoders. The referenced final reports identify actual checkpoint paths, seeds, normalizers, and hashes. A filename containing “v5,” “v1010,” or “128-D” does not by itself identify a common experiment.

## 6. Evaluation protocol and interpretation

- **Task success:** the experiment's recorded task-judge endpoint. Pickup and placement endpoints are not interchangeable. “Ever successful” is distinct from final success.
- **Strict success / collision-free task completion:** task success and no forbidden contact during the rollout. Some expert gates additionally require clutter stability or other checks; retain their original definitions.
- **Collision-free:** no forbidden contact, whether or not the task finishes.
- **Hazard incidence:** episodes with any contact with the designated hazard.
- **Contact samples:** audited physics samples with contact, distinct from policy frames, RGB frames, contact-pair entries, and independent trials.

Intended grasp and receptacle contacts are excluded only where the evaluator's contact taxonomy explicitly says so. The archived visibility study instead uses a broad environment-contact measure with a high no-hazard background rate.

Matched-scene comparisons require matching actual initial state, not merely equal random seed labels. The original H-A scenario lists do not match; H-B uses matched scenarios but different training seeds between ACT and PACT. Local multi-seed placement has matched methods within each seed and disjoint scene blocks across seeds. Those different designs require different statistical interpretations.

## 7. Results

### 7.1 Archived obstacle studies: visibility, representation, training blur

#### O01 — First obstacle PACT and camera-only tuning

**Question:** does adding proximity improve the first obstacle policy?

The archived June-18 comparison, n=50, reports no advantage over ACT (p=0.76). Zeroing proximity changes predicted chunks by about 0.005. Earlier camera-only development runs, n=20, report contact rates of 30% versus 60% for chunk 50 versus 100, and 60% versus 20% for 2,000 versus 5,000 epochs. These are historical development comparisons with limited retained provenance, not a clean factorial study or current-model results.

The remote log invalidates pre-July-4 evaluations using the broken `--temp_agg_off` path. Appendix D preserves the individual diagnostic JSONs, including no-progress runs and repaired/aggregation-on runs. [Evidence: remote log][readme].

#### O02 / O-INV — Free, visible, and camera-hidden obstacle

**Setup:** old pickup policy; one checkpoint per method; 50 trials per cell; horizon 200. Each cell below gives **task success / strict success / episodes with environment contact**, all out of 50.

| Condition | ACT | Archived raw PACT | Archived trunk PACT |
|---|---:|---:|---:|
| No hazard | 11 / 4 / 30 | 9 / 4 / 29 | 17 / 8 / 32 |
| Visible hazard | 14 / 8 / 32 | 8 / 5 / 25 | 16 / 9 / 29 |
| Camera-hidden hazard | 18 / 7 / 33 | 15 / 10 / 20 | 17 / 5 / 36 |

**Observation:** hidden-condition environment contact is 66% for ACT and 40% for raw PACT; the reported two-sided Fisher p-value is 0.016. The trunk representation instead reaches 72% contact in that condition.

**Interpretation:** a historical contact-reduction result with an important negative representation control. The 2/14/26-point free/visible/hidden pattern is descriptive; no verified interaction test establishes a visibility-dependent effect. Task-success nonsignificance does not demonstrate retention. The counter includes background enclosure contact, and training/checkpoint artifacts were deleted. An inconsistent training-composition count in the old README is not repeated as fact here.

**Evidence:** nine retained `vanilla_v2_*`, `pact_raw_v2_*`, and `pact_trunk_v2_*` [remote JSONs][remote-evals]; [historical report][aug-report]. This is not a recovered current-128-D visibility experiment.

#### O03 — Train-time constant blur, evaluated with clear images

**Setup:** three separately trained camera-only policies; sigma 2, 4, 8; three visibility conditions; 25 trials per cell, 225 total. Entries are **task / strict / environment-contact episodes**, out of 25.

| Training blur | Free | Visible | Camera-hidden |
|---|---:|---:|---:|
| Sigma 2 | 1 / 0 / 12 | 2 / 2 / 13 | 0 / 0 / 12 |
| Sigma 4 | 10 / 3 / 21 | 6 / 2 / 21 | 10 / 1 / 22 |
| Sigma 8 | 6 / 4 / 7 | 6 / 4 / 17 | 6 / 2 / 17 |

Validation losses rose monotonically (unblurred/2/4/8: 0.0755/0.0836/0.0948/0.1100), while behavior did not. The unblurred comparator in O02 uses n=50, not 25. The mildly blurred policy's low contact coexists with almost no task success. This is a completed negative training intervention, distinct from test-time blur in P08. A difference between physically different cells is not an estimate of sampling noise by itself. [Nine saved blur summaries][remote-evals], [historical report][aug-report].

#### O04 — Collision filtering, avoidance upsampling, and dropout

**Setup:** `avoid-v1` filters scraping demonstrations, upsamples bows 3×, uses image dropout 0.3 and proximity dropout 0.1. Remote notes report hidden-condition contact ACT 40% versus PACT 30% (p≈0.40), but task success 42% versus 24%.

**Interpretation:** inconclusive contact difference with worse observed task completion. Training also exposed visible bars and coupled cup/bar-side structure, allowing visual shortcuts. Exact retained per-episode outcomes were not found among the remote evaluation JSONs; retain these as report-level results. [Remote log][readme].

#### O05 — Gate-bar geometry probe

Short 20–24 cm pegs induced only about 1.6–7.2 cm lateral detours and often missed the relevant path. A taller 44 cm pole was proposed, but no completed collection/evaluation is established by that proposal. Geometry development, not a PACT benchmark. [Remote log][readme].

### 7.2 Historical hallway and remote clutter evaluations

#### H01 / H-A — Original three-method hallway comparison

**Setup:** 152 demonstrations, wrist RGB, chunk 50, horizon 800, n=50 per arm, one checkpoint per method.

| Method | Place success | Strict success | Hazard contact | Collision-free |
|---|---:|---:|---:|---:|
| ACT | 14/50 (28%) | 13/50 (26%) | 17/50 (34%) | 33/50 (66%) |
| Scalar/raw PACT | 21/50 (42%) | 17/50 (34%) | 18/50 (36%) | 32/50 (64%) |
| 128-D readout, finetuned | 20/50 (40%) | 20/50 (40%) | 6/50 (12%) | 44/50 (88%) |

**Observation:** readout reduces hazard contact by 22 percentage points versus ACT; Fisher p≈0.016. Versus raw, reduction is 24 points, p≈0.009. Placement improvement versus ACT is uncertain (reported p≈0.29). Collision-free **88% is not an 88% successful-placement rate**.

**Limits:** random scene realizations are not paired; intrusion sides mismatch in 24/50 indexed ACT/readout pairs. The raw/readout implementations also differ in range, temporal encoding, tokenization, and capacity. The six readout contacts consist of five right-side and one left-side approach, a small post-hoc diagnostic. Preceding n=20 smoke results (ACT/raw/readout place 15%/35%/25%, hazard 30%/20%/30%) are development, not an independent confirmatory replication.

**Evidence:** `place_corridor_{vanilla,raw}_s0_n50.json`, `place_corridor_readout_s0_n50_fast.json` in [remote summaries][remote-evals]; [paper discussion][paper].

#### H02 / H-B — September hallway retraining with frozen evaluation

**Setup:** same hallway family, 50 matched scenarios, evaluation seeds 2026–2075, house 1, horizon 800. ACT training seed 1; raw/readout seed 0.

| Method | Place success | Strict success | Hazard contact | Collision-free |
|---|---:|---:|---:|---:|
| ACT | 15/50 (30%) | 12/50 (24%) | 20/50 (40%) | 30/50 (60%) |
| Scalar/raw PACT | 21/50 (42%) | 19/50 (38%) | 13/50 (26%) | 37/50 (74%) |
| 128-D readout, finetuned | 21/50 (42%) | 19/50 (38%) | 9/50 (18%) | 41/50 (82%) |

Readout-versus-ACT hazard discordance is 11 versus 0, exact McNemar p≈0.00098; raw-versus-ACT 8 versus 1, p≈0.039; readout-versus-raw 6 versus 2, p≈0.29. These paired tests differ from the unpaired Fisher numbers in some progress notes. They do not resolve training-seed variation. Other trained seeds lack completed n=50 evaluations in this remote record; ACT seed 0 has only two recorded evaluations. [Three `pact_place_corridor_v5_*` summaries][remote-evals], [paper][paper].

#### H03 / H-C — Old readout checkpoint, new fixed evaluation

The H-A readout checkpoint is evaluated again on the H-B scenario schedule: **18/50 final placements, 19/50 ever placed, 18/50 strict, 7/50 hazard contacts, 43/50 collision-free**. This is an evaluator/scenario rerun of the same trained model, not another training seed. [`simple_hallway_n50.json`][remote-evals], [paper][paper].

#### H04 / T-107 — Spaced-bench three-method comparison

**Setup:** 210 demonstrations, table+wrist RGB, 24 condition indices, chunk 50, horizon 1,050, one training seed, 50 trials per method.

| Method | Place success | Strict success | Hazard contact | Collision-free |
|---|---:|---:|---:|---:|
| ACT | 13/50 (26%) | 7/50 (14%) | 8/50 (16%) | 19/50 (38%) |
| Scalar/raw PACT | 0/50 | 0/50 | 9/50 (18%) | 19/50 (38%) |
| Readout PACT | 2/50 (4%) | 2/50 (4%) | 6/50 (12%) | 20/50 (40%) |

**Interpretation:** negative placement result for both proximity methods. Readout versus ACT placement Fisher p≈0.004; lower observed bar contact does not offset the task collapse. This environment is not a successful robustness replication of H-A. [Three `v107_spaced` summaries][remote-evals].

#### H05 / T-1011d — Randomized clutter, scalar PACT only

**Setup:** 200 demonstrations, exo+wrist policy, horizon 1,050. No matched ACT/readout comparison was recovered.

| Evaluation | Final placement | Strict success | Hazard contact | Collision-free |
|---|---:|---:|---:|---:|
| Original randomization, n=50 | 7/50 (14%) | 4/50 (8%) | 10/50 (20%) | 20/50 (40%) |
| Reduced xy randomization, scale 0.25, n=50 | 14/50 (28%) | 11/50 (22%) | 3/50 (6%) | 22/50 (44%) |

The easier distribution improves point estimates, but is not evidence of robustness to a harder unseen distribution. A wrist-only evaluation stopped at **4/50 planned**, with 0/4 successes and 3/4 bar contacts; it is incomplete. Mismatched FourObject-sampler evaluations have **0/48 successes** at horizons 800 and 1,050, respectively (bar 3/48 and 0/48); these are not source-distribution scores. A separate ray-proximity speed-check run is reported as 0/50 in the remote log. [Saved summaries][remote-evals], [remote log][readme].

#### H06 — Trained remote models without completed evaluation

Remote T-1010 has six trained checkpoints (three methods × two seeds, 215 demonstrations); remote T-1011c has three checkpoints on a 99-demo corpus. The remote inventory calls their evaluations unwired. These are training records, not additional rollout results, and are distinct from the local V10.10/V10.11c experiments below. Later v12 commands and a framebuffer diagnostic likewise do not establish a completed policy benchmark. [Remote inventory][readme].

### 7.3 Local pickup/extraction: representation, ablations, contact, and robustness

#### P00 — Environment adequacy and the interrupted first confirmatory run

The adequacy screen has **58/64 clean expert demonstrations, 59/64 ordinary expert successes**. Active-panel proximity falls inside 20 cm on 54.0% of pregrasp steps and inside 12 cm on 18.1%. A separate ACT pilot has **23/64 task and strict successes**, with intrusion contact on another 23/64 episodes. These precede the main comparison. [Adequacy report](PACT_ENVIRONMENT_ADEQUACY.md), [final decision](PACT_VS_ACT_FINAL_DECISION.md).

The original 960-row confirmatory launch completed one scientific row; eight more crossed initial-observation acceptance but lost their processes before results, and 951 never started. This is **incomplete**, not a null. R2 used a separately frozen fresh-instance schedule. [Interruption ledger](PACT_CONFIRMATORY_INTERRUPTION_AND_R2_RECOVERY.md).

#### P01 — Original learned 3-D representation, confirmatory R2

**Setup:** 160 shared instances × two training seeds, 320 rollouts per arm. This is pickup/extraction, not full placement.

| Method | Task success | Strict success |
|---|---:|---:|
| ACT | 177/320 (55.3%) | 170/320 (53.1%) |
| 3-D PACT | 169/320 (52.8%) | 159/320 (49.7%) |
| PACT_ZERO | 169/320 (52.8%) | 160/320 (50.0%) |

PACT−ACT strict gap is −3.4 points, instance-bootstrap 95% CI [−8.4, +1.6]. **Completed negative/inconclusive comparison.** The interrupted earlier confirmatory execution and its recovery are not an extra successful replication. [Final decision](PACT_VS_ACT_FINAL_DECISION.md), [interruption/recovery](PACT_CONFIRMATORY_INTERRUPTION_AND_R2_RECOVERY.md).

#### P02–P04 — Frozen 32-D screen, valid ablation, and second seed

| ID / stage | ACT task / strict | Live PACT task / strict | Unrelated-frame control task / strict | Zeroed PACT |
|---|---:|---:|---:|---:|
| P02: seed 3101 screen, n=40 | 19 / 19 | 29 / 29 | Added in P03 | 2 task / 1 strict |
| P03: valid intervention, same seed/scenes | Reuses P02 | Reuses P02 | 24 / 24 | Not the validated control |
| P04: seed 3102 replication, n=40 | 24 / 24 | 22 / 21 | 15 / 15 | Not required |

All counts are out of 40. P03 adds a new control arm, not 40 new live PACT episodes. Its live-control strict gap is +12.5 points, paired CI [−2.5, +27.5], McNemar p=0.2266. Seed-3102 live-control gap is +15 points, CI [2.5, 27.5], exact McNemar p≈0.0703; the different inferential procedures need not agree at the threshold.

Across the two seeds, live-control strict gap is +13.75 points, clustered CI [3.75, 23.75]. PACT−ACT is +8.75 points, CI [−5.0, 22.5]. **The large screen advantage over ACT did not replicate.** Zero-input collapse is not by itself a valid physical-information claim, as S13 establishes.

Evidence: [screen](PACT_FRONTEND_SCREEN_DECISION.md), [valid ablation](PACT_VALID_ABLATION_DECISION.md), [replication](PACT_SEED_REPLICATION_DECISION.md).

#### P05 — Larger matched contact-endpoint study

**Setup:** frozen 32-D PACT; 100 shared instances × three training seeds × four arms = 1,200 rollouts. The same-checkpoint unrelated-frame intervention preserves real-frame values while breaking correspondence.

| Method | Task success | Strict success | Any hazard contact | Mean hazard-contact samples |
|---|---:|---:|---:|---:|
| ACT | 169/300 (56.3%) | 159/300 (53.0%) | 67/300 (22.3%) | 3,023.4 |
| Live PACT | 183/300 (61.0%) | 171/300 (57.0%) | 42/300 (14.0%) | 1,678.2 |
| Unrelated-frame PACT | 171/300 (57.0%) | 159/300 (53.0%) | 70/300 (23.3%) | 3,658.0 |
| Zeroed PACT | 24/300 (8.0%) | 20/300 (6.7%) | 107/300 (35.7%) | 3,912.6 |

Live−control hazard incidence: **−9.33 points**, CI [−14.33, −5.0]. Mean sample difference: **−1,979.8**, CI [−3,152.9, −965.2], a 54.1% point-estimate reduction. Live−ACT task gain remains uncertain: +4.7 points [−2.3, +11.7]; strict gain +4.0 [−2.3, +10.3]. The live-control mean-contact gap favors live in all three seeds.

This supports lower contact burden from live scene-corresponding proximity for this frozen 32-D model. It does not isolate sensor identity, temporal freshness, or finetuning. [Decision](PACT_CONTACT_ENDPOINT_DECISION.md), [analysis](../diagnostics_output/pact_contact_endpoint/analysis.json).

#### P06 — Contact-tail and target-engagement reanalyses

Reuses P05; no new trials.

| Analysis | ACT | Live PACT | Unrelated-frame control |
|---|---:|---:|---:|
| Episodes exceeding 500 hazard samples | 59/300 | 33/300 | 58/300 |
| Hazard contact with no target touch | 34/300 | 3/300 | 30/300 |
| Fewer than 50 target samples within high-contact tail | 34/59 | 5/33 | 33/58 |

All 67 no-target-touch hazard episodes fail. Mean hazard samples **conditional on entering the high-contact tail** are about 15,363 ACT versus 15,236 PACT, so the evidence concerns fewer entries into that regime rather than faster escape. Among both-successful live/control pairs, the mean hazard-sample difference is −0.8 [−19.8, 22.0], not a measurable gain. [Tail analysis](PACT_TAIL_CHARACTERIZATION.md), [engagement analysis](PACT_ABSORBING_FAILURE_CHARACTERIZATION.md), [contact decision](PACT_CONTACT_ENDPOINT_DECISION.md).

#### P07 — Zero-shot geometry study V3

**Setup:** fixed frozen 32-D policies, live and unrelated-frame arms only; 40 instances × three seeds × three conditions × two arms = 720 rollouts. C0 uses aperture 0.85 m, panel inner face y=0.10 m, panel z=0.89 m. C2 narrows the aperture to 0.70 m and moves the inner face to y=0.07 m; Z_093 raises the panel to z=0.93 m. The [frozen config](../configs/pact_geometry_generalization_v3.json) defines jitter and the remaining fixed geometry.

| Condition / method | Task | Strict | Hazard incidence | Mean hazard samples |
|---|---:|---:|---:|---:|
| C0 live | 75/120 | 70/120 | 10/120 | 610.4 |
| C0 control | 66/120 | 63/120 | 22/120 | 1,903.7 |
| C2 live | 55/120 | 50/120 | 23/120 | 2,047.4 |
| C2 control | 46/120 | 41/120 | 37/120 | 3,793.7 |
| Z_093 live | 67/120 | 55/120 | 20/120 | 2,049.4 |
| Z_093 control | 66/120 | 56/120 | 34/120 | 3,795.2 |

Pooled shifted incidence gap: −11.7 points [−18.3, −5.4]; mean hazard-sample gap: −1,746.1 [−2,794.3, −811.8]. Contact advantage survives relative to this control, but live task/strict performance drops from C0, and Z_093 strict success does not favor live. **No ACT arm; no current-128-D generalization claim.** [V3 report](PACT_GEOMETRY_GENERALIZATION_V3.md).

Earlier geometry V1 completed 48 expert feasibility rows: clean C0 11/12, C1 4/12, C2 12/12, C3 5/12. Only one shifted condition passed, so no learned policy ran. V3 carries forward C0/C2 qualification without rerunning it and uses Z_093's later 11/12 clean screen. V2 stopped at 9/900 policy rows and was abandoned before interpretation. These are predecessors, not independent positive replications. [V1](PACT_GEOMETRY_GENERALIZATION.md), [V2 progress](PACT_GEOMETRY_GENERALIZATION_V2_PROGRESS.md).

#### P08 — Test-time RGB blur sweep

**Setup:** frozen policies; 25 instances × three seeds × three arms × four blur levels = 900 rollouts. This is different from training new blurred-image policies in O03.

| Test sigma | ACT strict | Live PACT strict | Control strict | ACT / live / control mean hazard samples |
|---|---:|---:|---:|---:|
| 0 | 36/75 | 44/75 | 36/75 | 3,333 / 679 / 1,946 |
| 0.5 | 33/75 | 42/75 | 37/75 | 3,319 / 658 / 2,575 |
| 1 | 30/75 | 34/75 | 31/75 | 4,245 / 895 / 3,163 |
| 2 | 13/75 | 14/75 | 15/75 | 4,507 / 2,748 / 5,705 |

At sigma 2, ACT/live hazard incidence is 25/75 versus 12/75. Contact burden improves, but **the predeclared task-robustness hypothesis fails**: every positive-sigma live−ACT strict-success interval includes zero; slope interaction −5.3 points/sigma [−12.6, +1.4]. [Report](PACT_BLUR_SWEEP.md).

#### P09 — Complete RGB replacement

**Setup:** sighted/blind, 25 instances × three seeds × three arms = 450 rollouts. Replacing all RGB is not selective hiding of the hazard.

| Condition / method | Task | Strict | Hazard incidence | Mean hazard samples |
|---|---:|---:|---:|---:|
| Sighted ACT | 37/75 | 36/75 | 16/75 | 2,967.9 |
| Sighted PACT | 44/75 | 43/75 | 4/75 | 693.6 |
| Sighted control | 37/75 | 37/75 | 14/75 | 1,932.8 |
| Blind ACT | 1/75 | 0/75 | 48/75 | 12,289.4 |
| Blind PACT | 1/75 | 1/75 | 37/75 | 8,113.7 |
| Blind control | 0/75 | 0/75 | 46/75 | 9,934.5 |

Blind PACT−ACT mean-contact gap is −4,175.7 [−7,356.3, −1,263.0]; live−control −1,820.8 [−2,999.0, −784.8]. **Contact benefit survives while manipulation collapses.** Sighted counts differ from P08's sigma-zero execution; retain each experiment's own records. [Report](PACT_BLIND_RGB.md).

### 7.4 Local full placement: chunking, clutter, and latest readout

#### L01–L03 — Recovered-152 chunk comparisons

The recovered “V5” dataset uses `PactPlaceCorridorV2Sampler`, with no added household clutter. It is not the later cluttered V5 sampler. All 152 expert recordings were recovered without divergence. [Recovery](PACT_PLACE_V5_DEMO_RECOVERY.md), [sampler inventory](../diagnostics_output/manipulation_eval_inventory_20260906/README.md).

The preceding chunk-1 training-only comparison completed 2,000 epochs at seed 3101: best validation loss **0.042982 ACT versus 0.047391 PACT**, at epochs 1,954 and 1,773. Its original report correctly says no rollout had yet run; the subsequent L01 evaluation below supplies the behavioral outcome. [Training report](/root/pact_place_152_pact_vs_act_seed3101/EVAL.md).

| ID / chunk | ACT task / strict | Frozen PACT task / strict | Other findings |
|---|---:|---:|---|
| L01: chunk 1, n=20 | 0 / 0 | 0 / 0 | Neither policy commands gripper closure |
| L02: chunk 25, n=40 | 8 / 6 | 11 / 10 | Closure attempted in ACT 9/40, PACT 20/40; interaction CI includes zero |
| L03: chunk 100, n=40 | 13 / 13 | 19 / 16 | Unrelated-frame control 7 task / 6 strict; closure in all 40 for all arms |

These are one-seed development comparisons, not three independent demonstrations that proximity improves performance. The zero-success chunk-1 run cannot establish safe task execution. [Chunk 1](/root/pact_place_chunk1_eval_seed3101/EVAL.md), [chunk 25](../EVAL.md), [chunk 100](/root/pact_place_chunk100_eval_seed3101/EVAL.md).

#### L04–L07 — First clutter datasets and decoder diagnostic

| ID / experiment | Dataset and evaluation | ACT task / strict | PACT task / strict | Observation |
|---|---|---:|---:|---|
| L04: V10.9 household clutter | 141 demonstrations, one seed, n=40 each | 14 / 8 | 11 / 6 | Strict gap −5.0 points [−16.9, +6.9]; no success advantage |
| L05: V10.9R decoder | Ten development instances; legacy versus event decoder | 4/10 task under both | 1/10 task under both | No task gain; bulk trajectories deleted, retained summaries only |
| L06: initial V10.10 four-object | 144 demos, 120/24 split, one seed, n=40 each | 11 / 7 | 14 / 10 | Strict +7.5 points [−7.02, +22.02]; inconclusive |
| L07: V10.11c taller primitives | 99 demos, 75/24 split, three seeds, n=150 each | 11 / 8 | 12 / 10 | Severe task collapse; no overall success/safety win |

L04 hazard-contact episodes are ACT 10/40 versus PACT 8/40; clutter contact 17/40 versus 18/40. L06 clutter contact is 22/40 versus 17/40, pendant contact 2/40 versus 0/40. L07 hazard episodes are 50/150 versus 55/150; hazard samples 227,194 versus 283,659; clutter episodes 54/150 versus 39/150.

The detailed L07 object audit finds lower PACT contact incidence for the route cylinder (19→13/150), near-target cylinder (10→5/150), and near-target box (8→3/150), but route-cylinder contact samples increase (14,343→23,512). This does not rescue the overall task result. [V10.9](PACT_PLACE_V109_TRAIN_EVAL_PLAN.md), [decoder diagnostic](../diagnostics_output/pact_place_v109r_diagnostic/diagnostic_analysis.json), [V10.10](PACT_PLACE_V1010_FOUR_OBJECT_TRAIN_EVAL_PLAN.md), [V10.11c report](../diagnostics_output/pact_place_v1011c_eval/report.md), [per-object table](../EVAL.md).

#### L08 — Aligned dual-camera V10.11c pilot

One seed, 50 original rollouts per method. ACT task/strict **14/50 and 12/50**; PACT **7/50 and 7/50**; hazard incidence **14/50 for both**. All original attempts finished, but initial-scene pairing failed on one instance and again on a bounded repeat. These are descriptive original-attempt counts, **not a validated paired benchmark**. [Evaluation record](../EVAL.md), [pairing audit](../diagnostics_output/pact_place_v1011c_dualcam_aligned_s3103/evaluation/blocked_raw_audit.json).

#### L09 / L11 — V10.10 wrist280, frozen versus finetuned readout, three seeds

**Question:** does the complete finetuned 128-D method improve placement and contact outcomes on the current four-object environment?

**Setup:** 280 demonstrations, 240 training/40 validation; seeds 3103–3105; 60,000 updates; wrist RGB; chunk/history 100; horizon 900; final policy checkpoints and corresponding encoders; 50 matched scenarios per seed and method. Three scene blocks are disjoint and previously exposed regression scenes.

| Method | Task success | Strict success | Any hazard contact | Collision-free |
|---|---:|---:|---:|---:|
| ACT | 64/150 (42.7%) | 46/150 (30.7%) | 33/150 (22.0%) | 77/150 (51.3%) |
| Frozen 32-D PACT | 74/150 (49.3%) | 51/150 (34.0%) | 13/150 (8.7%) | 90/150 (60.0%) |
| **Finetuned 128-D PACT** | **83/150 (55.3%)** | **63/150 (42.0%)** | 17/150 (11.3%) | **93/150 (62.0%)** |

| Method | Forbidden-contact physics samples | Hazard samples | Clutter samples |
|---|---:|---:|---:|
| ACT | 376,556 (8.452%) | 232,733 (5.224%) | 160,802 (3.609%) |
| Frozen PACT | 135,897 (3.050%) | 44,673 (1.003%) | See source contact breakdown |
| Finetuned PACT | 85,933 (1.929%) | 54,634 (1.226%) | 32,207 (0.723%) |

Denominator: **4,455,150 audited physics samples per method**. Categories overlap; forbidden is their union including other environment and mounted fixtures. The older ACT hazard/clutter-only union 371,293 is not the complete forbidden count.

| Seed | ACT task / strict | Finetuned PACT task / strict |
|---|---:|---:|
| 3103 | 19/50 / 16/50 | 27/50 / 24/50 |
| 3104 | 22/50 / 15/50 | 26/50 / 18/50 |
| 3105 | 23/50 / 15/50 | 30/50 / 21/50 |

**Observation:** finetuned PACT versus ACT gains 12.7 points task success and 11.3 points strict success, with 77.2% fewer forbidden-contact samples and 76.5% fewer hazard samples. Task and strict-success counts favor finetuned PACT in all three seeds.

**Limits:** relative to frozen PACT, the finetuned method gains 6.0 points task and 8.0 points strict success, but has 22.3% more hazard samples; frozen PACT wins task and strict success in seed 3105. This is a complete-method comparison, not an isolated encoder-unfreezing ablation. Different scene blocks limit attribution of between-seed differences. It is not the historical 152-demo hallway experiment or the unrun new zero-shot suite.

L09 is the original frozen comparison; L11 adds the 128-D arm. The standalone seed-3103 readout result, partial 40/50 milestones, pause/recovery reports, and final three-seed table are **constituents and revisions of this result**, not independent replications. The frozen result did not meet its predefined >50% task and ≥10-point advantage target.

**Evidence:** [three-method report](PACT_PLACE_V1010C_THREE_METHOD_COMPARISON.md), [final review](../diagnostics_output/pact_place_v1010c_readout_s3_v1/FINAL_REVIEW.md), [450 audited rows](../diagnostics_output/pact_place_v1010c_readout_s3_v1/comparison.json), [contact union reconstruction](../diagnostics_output/pact_place_v1010c_readout_s3_v1/root_review/frame_avoidance.json).

#### L10 — Acquisition-window training continuation

**Setup:** three frozen-encoder PACT seeds; same 12 exposed diagnostic scenes per seed; three arms: original frozen checkpoint, equal-update uniform continuation, acquisition-window continuation. New training adds 3,000 updates to 60,000.

| Arm | Task success | Strict success |
|---|---:|---:|
| Original checkpoint | 16/36 | 11/36 |
| Uniform continuation | 22/36 | 14/36 |
| Acquisition-window continuation | 20/36 | 13/36 |

The targeted sampler underperforms equal-update uniform continuation and raises contact samples 86.3% versus uniform. **Rejected at Gate B; later stages not run.** [Final review](../diagnostics_output/pact_place_v1010b_grasp_v1/FINAL_REVIEW.md).

#### L12 — Completion time of the latest successful rollouts

**Setup:** reanalysis of all 300 ACT/finetuned-PACT rollouts from L11; no new experiment execution. Time measured from simulator timestamps, 0.066 s per action.

| Comparison | ACT mean | PACT mean | PACT time reduction |
|---|---:|---:|---:|
| All final-success episodes, 64 ACT / 83 PACT | 31.609 s | 29.060 s | 2.549 s (8.06%) |
| Same 50 scenes, both policies successful | 31.240 s | 29.726 s | 1.514 s (4.85%) |
| Start of final uninterrupted success, shared-success subset | See source statistics | See source statistics | 1.494 s |

On the shared-success subset, PACT is faster in 32 pairs, ACT in 17, with one tie. Mean paired saving conditional bootstrap 95% CI: **[0.408, 2.862] s**. Pooled successful-only means use different scene populations. Both-success conditioning also limits generalization; the interval does not capture uncertainty across new training seeds. This measures simulated task completion, not inference latency or wall-clock rollout speed. [Report](../diagnostics_output/pact_place_v1010c_completion_time_20260916/REPORT.md), [all 147 successful rollouts](../diagnostics_output/pact_place_v1010c_completion_time_20260916/all_147_successful_rollouts.csv).

#### L13 — Failure, observation, and supervision audits

| Analysis | Measured result | Scope |
|---|---|---|
| V10.11c target engagement | No target touch in ACT 105/150 and PACT 115/150; target absent from wrist segmentation throughout 62/150 and 75/150 | Failure characterization, not causal proof that occlusion alone caused failure |
| Empty-handed transport | Of never-touch failures, 100/105 ACT and 112/115 PACT approach within 10 cm XY of the tray | Supports a placement-like motion without acquisition |
| Label timing | One-step arm-label lag in all 99 V10.11c and 144 initial V10.10 demos | Audit of those datasets, not automatically every subsequent dataset |
| Unused table camera | Initial target segmented in 92/99 table views versus 15/99 wrist views | Motivated a remedy; visibility alone does not establish policy improvement |
| Aggregation age | History-100 weighted mean observation age 2.726 s; oldest 6.534 s | Offline controller analysis |
| Forward-command aggregation comparison | History 10 reduces pre-touch joint error by 27.1% ACT / 28.6% PACT versus history 100; current-position baseline is lower still | Not a measured closed-loop gain |
| Frozen wrist280 grasp diagnosis | Seeds 3104/3105: 19/50 versus 35/50 task, 23/50 versus 38/50 ≥1 cm lift, 18 versus 8 touch-without-hold failures | Different scene blocks; “held” contact flag is not validated stable grasp |

Evidence: [failure audit](../diagnostics_output/pact_place_v1011c_post_eval_audit/AUDIT.md), [approach/aggregation](../diagnostics_output/pact_place_v1011c_post_eval_audit/APPROACH_AND_CHUNK100.md), [learning diagnosis](../diagnostics_output/pact_place_v1011c_post_eval_audit/learning_failure/DIAGNOSIS.md), [pickup audit](../diagnostics_output/pact_place_v1010_pickup_audit_20260906/AUDIT.md), [300-rollout grasp diagnosis](../diagnostics_output/pact_place_v1010_wrist288_s3_v1/geometric_failure_20260908/GEOMETRIC_DIAGNOSIS_300.md).

### 7.5 Two-level fridge experiments

All entries here are recovered from the [fridge EVAL record](/root/prox_learning/EVAL.md). The policy experiment uses handcrafted 3-D proximity features, not the current finetuned encoder.

| ID | Experiment | Result / interpretation |
|---|---|---|
| F01 | Original open-top fridge dataset audit | 74 valid demos, mean 282 frames; 29.7% sensor activation below 0.5 m, 13.8% below 0.2 m, average 11.9/40 active sensors; target visible to exo in all 20,838 frames. Does not establish full geometric observability |
| F02 | Physical front-visor occlusion prototype | Two executed episodes out of three planned; 0/2 successes; neither camera sees target before grasp. Visor also physically blocks the arm: confounded visibility manipulation |
| F03 | Low/side exo-camera reposition | Initial smoke 3/3 placements; subsequent four-episode audit about 50% of pregrasp frames blind to both cameras; physical scene unchanged |
| F04 | Scaled exo-blind collection | 196 successful demos / 213 attempts (92.0%); 195 complete pick-place trajectories; 53.1% of pregrasp frames blind to both cameras |
| F05 | Handcrafted-token ACT/PACT training and interim evaluation | Best validation losses ACT 0.078 / PACT 0.072; n=6 each, 0/6 task success for both; early gripper closure around step 27. Planned full n=50×3 not completed |
| F06 | Object visibility to skin / encoder-label feasibility | Only 99 sensor-frame target sightings across the 196-demo audit, from three sensors; nearest 33.8 cm, none within 20 cm. No useful in-range target-position supervision recovered |
| F07 | Matched safety reevaluation with category logging | ACT, PACT, PACT-zero all 0/6 success. Static-environment contact entries: 0 / 835 / 955; static-contact steps: 0 / 417 / 500. Negative safety result for these handcrafted-token policies |

F05/F07 are bounded studies of the same checkpoints and six-house setting; do not count their task scores as independent replication. A launched larger ACT evaluation was stopped before any HDF5 rollouts were saved. The wall-sensing/target-sensing distinction is a measured limitation of this configuration, not proof that all proximity-based manipulation requires seeing the target.

### 7.6 Hybrid ACT + Safety-CVAE residual-controller program

This program uses an explicit residual controller. Its results must not be labeled PACT-128D or proximity-token ACT results.

#### Y01–Y08 — Artifact recovery, data integrity, and baseline qualification

| ID / study | Executed work and result | Evidence |
|---|---|---|
| Y01: stack recovery | Sensor/Safety-CVAE contract checked; live evaluation blocked by missing canonical ACT/statistics/data; no policy rollouts | [Decision](HYBRID_SAFETY_STACK_FINAL_DECISION.md) |
| Y02: initial clean-retrain collection | 175 trajectories, 165 reported successes; selected hazard fraction 69/100 misses 70% floor; training stopped | [Original decision](HYBRID_CLEAN_RETRAIN_FINAL_DECISION.md) |
| Y03: dataset duplication audit | Those 175 files represent **75 distinct episodes, 71 distinct successes**; 50 content classes stored three times | Supersedes independence implied by Y02. [Integrity decision](HYBRID_OBSTACLE_DATASET_FINAL_DECISION.md) |
| Y04: independent-seeding proof | Episode-scoped manifest and worker-count/resume invariance checked in bounded smoke runs | Engineering validation, not policy performance. [Decision](HYBRID_OBSTACLE_SEEDING_FINAL_DECISION.md) |
| Y05: repaired collection | 160 manifest rows completed, 145 successes; canonical 100 selected as 75 hazard/25 absent, split 80/20; duplicate conversion verified | [Collection decision](HYBRID_OBSTACLE_FULL_COLLECTION_FINAL_DECISION.md) |
| Y06: canonical ACT training | One seed-0, 2,000-epoch baseline; repaired split/statistics handling; best epoch 1,738; offline validation and strict reload | No simulator comparison in this task. [Training decision](HYBRID_OBSTACLE_ACT_BASELINE_FINAL_DECISION.md) |
| Y07: bounded paired smoke | Adapter/hash checks passed, smoke stopped on observation/reference-contract blockers; no full 45-row evaluation | [Smoke decision](HYBRID_OBSTACLE_PAIRED_SMOKE_FINAL_DECISION.md) |
| Y08: observation/reference diagnosis | Repeated identical-state renders differed in sparse one-level pixels under 4× MSAA; reference semantics investigated | Instrument finding, not a task-success estimate. [Decision](HYBRID_OBSTACLE_OBSERVATION_REFERENCE_FINAL_DECISION.md) |

#### Y09–Y11 — Live residual-controller comparisons

**Setup:** four development candidates, five repeats each. Three candidates have hazards; one is hazard-absent. Repeats are not 20 independent environments.

| ID / controller | Hazard-present task result | Other observation | Decision |
|---|---|---|---|
| Y09: raw SafetyHead residual | ACT 11/15 → residual 0/15 | Introduces environment contact in all 5/5 hazard-absent repeats | Rejected |
| Y10: per-frame parked-obstacle oracle | ACT 11/15 → oracle 14/15 | Hazard contact ACT 0/15 versus oracle 1/15 | Bounded development passes; privileged reference |
| Y11: learned deployable reference | ACT 11/15 → reference 12/15 | New environment contact on 5/5 hazard-absent repeats; oracle cosine 0.977 offline → 0.345 live | Fails live transfer/safety gate |

Evidence: [raw head](HYBRID_OBSTACLE_RAW_HEAD_QUALIFICATION_FINAL_DECISION.md), [oracle](HYBRID_OBSTACLE_ORACLE_REFERENCE_FINAL_DECISION.md), [deployable reference](HYBRID_OBSTACLE_DEPLOYABLE_REFERENCE_FINAL_DECISION.md). The oracle moves an obstacle out of the scene to obtain a counterfactual measurement; it is not deployable sensing.

#### Y12–Y21 — Reference learning, activity calibration, and uncertainty

| ID / experiment | Measurement and outcome | Evidence |
|---|---|---|
| Y12: one on-policy aggregation round | Differential MAE improves 0.313→0.208 on ACT-only and 0.355→0.151 on oracle trajectories, but no valid activation contract; stops before new live development | [Decision](HYBRID_OBSTACLE_ON_POLICY_REFERENCE_FINAL_DECISION.md) |
| Y13: parked-field data-contract audit | 60,793 paired frames lack required parked 40×8×8 supervision; proposed model cannot be trained from these records | [Decision](HYBRID_OBSTACLE_PARKED_SKIN_REFERENCE_FINAL_DECISION.md) |
| Y14: parked-skin supervision generation | Missing arrays are collected/reconstructed and audited; no policy/model performance in this stage | [Dataset decision](HYBRID_OBSTACLE_PARKED_SKIN_DATASET_FINAL_DECISION.md) |
| Y15: causal parked-field models, three seeds and history comparison | Overall reference MAE 0.011337 versus zero 0.042062 (73.0% lower); seed-0 false-positive rate 2.15% exceeds 2% gate, others 1.42%/1.10%. Four-frame history MAE 0.012809 is worse than current-frame 0.011337 | Threshold-transfer failure; not evidence of closed-loop success. [Decision](CAUSAL_PARKED_SKIN_REFERENCE_V1_FINAL_DECISION.md) |
| Y16: trajectory-aware threshold calibration | Threshold 0.99960858 passes calibration; diagnostic hazard-absent sequence has seven consecutive false positives | Failed transfer. [Decision](HYBRID_OBSTACLE_REFERENCE_THRESHOLD_FINAL_DECISION.md) |
| Y17: proximity-only activity and causal attribution | On 17 old false positives, clear proximity lowers activity 0.9999→0.0229; state shuffling/mean replacement leaves 0.9999. No feasible proximity-only calibration threshold | Supports proximity ambiguity on these examples, not a state-only cause. [Decision](HYBRID_OBSTACLE_PROX_ACTIVITY_GATE_FINAL_DECISION.md) |
| Y18: activity identifiability / ensemble diagnostic | Changed-pixel agreement rejects 17/17 historical false positives while retaining 96.5% active frames (AUROC 0.979) | Diagnostic separation on reused failures; not final calibration success. [Decision](HYBRID_OBSTACLE_ACTIVITY_IDENTIFIABILITY_FINAL_DECISION.md) |
| Y19: trajectory-bootstrap uncertainty | Five models each see 24–28 unique clusters from 40; median agreement active 0.5467 versus zero 0.6000; no feasible threshold | Bootstrap data variation does not reproduce the seed-ensemble diagnostic. [Decision](HYBRID_OBSTACLE_UNCERTAINTY_ABSTENTION_FINAL_DECISION.md) |
| Y20: full-seed joint calibration | 1,690 feasible calibration pairs; median active recall 1.0 and zero calibration upper-bound false activation; offline transfer fails all three checks | Failed qualification. [Decision](HYBRID_OBSTACLE_FULL_SEED_JOINT_GATE_FINAL_DECISION.md) |
| Y21: three-pair agreement repair | Historical false-positive executions 10/17→9/17, threshold 0.225→0.166667; transfer still fails | Restoring the third pair does not fix the regression. [Decision](HYBRID_OBSTACLE_THREE_PAIR_JOINT_GATE_FINAL_DECISION.md) |

#### Y22 — Three-pair gate, bounded live development

Twenty rollouts on the same four-candidate/five-repeat design: task successes **5/5, 5/5, 3/5, 0/5**; no hazard-bar contacts. One false-positive control frame, and the uncertainty veto fires zero times. This passes its live development criteria, but does not demonstrate benefit from the unused uncertainty veto. **Confirmatory41 remains unrun in the recovered record.** [Final decision](HYBRID_OBSTACLE_THREE_PAIR_LIVE_FINAL_DECISION.md).

## 8. Limitations and reproducibility

1. **Coverage of this inventory.** “Complete recoverable” refers to the sources indexed here. Colleague-only current-128-D visibility/intervention and representation studies were mentioned in conversation, but their result files were not supplied or recovered. Their absence here is not evidence they were never run.
2. **Heterogeneous versions.** Sensor feature dimensions, datasets, environments, cameras, controller cadence, horizons, and contact definitions change. Pooling them into a single ACT-versus-PACT success rate would be misleading.
3. **Reused evidence.** P02/P03, the P05 reanalyses, L09/L11 constituents, and L12 reuse observations. Repeated checkpoints/scenes also occur in remote reruns and hybrid development.
4. **Selection and exposure.** Development screens and regression sets are exposed; negative studies remain in the inventory. Lower validation loss, a selected video, or an expert qualification pass does not establish a policy improvement.
5. **Correlated measurements.** Physics samples are highly correlated within trajectories. Repeated seeds on a scene and repeated runs of a single development candidate are not independent environments.
6. **Historical corrections.** The initial hybrid collection duplicated trajectories; some early evaluation paths froze the arm; runtime-posed pendant geometry had stale collision bounds; some counts were corrected after source reconstruction. Corrected records take precedence while original artifacts remain traceable.
7. **Sensor claims.** Simulation-only results do not establish real ToF robustness. Forty sensors omit the hand. Broad “whole-body coverage” percentages and arbitrary-layout generalization are not established by the recovered tests.
8. **Audit depth.** Remote JSON outcome fields were read directly; local final reports, audited row analyses, and linked diagnostics were reconciled at report level. This inventory does not claim a fresh re-read of every raw HDF5 trajectory or successful reload of every historical checkpoint.

## 9. Conclusion

The recovered record supports a current three-seed placement improvement for the complete finetuned 128-D method, historical hallway contact reductions, and a larger matched information-intervention result for frozen 32-D pickup PACT. It also records failures to improve task completion in several environments, limits under visual degradation, and substantial environment/data/controller development. Those distinctions are part of the results and should remain visible when selecting evidence for the paper.

---

## Appendix A. Environment qualification and collection experiments

These experiments created and tested the task infrastructure. They are **expert, geometry, sensing, or data-collection results**, not learned-policy benchmarks. A failed gate is retained even when a later explicitly separate experiment proceeded.

### A.1 Placement expert, clutter, and sensor admission

| ID / stage | Executed result | Source |
|---|---|---|
| E01: original placement gate | 18/24 clean and task successes; zero hazard/environment contact; placement reliability limiting | [Gate](PACT_PLACE_CORRIDOR_GATE.md) |
| E02: release-clearance expert fix | 18/24 clean, 20/24 task, two inbound hazard episodes; gate fails | [Gate V2](PACT_PLACE_CORRIDOR_GATE_V2.md), [fix](PACT_PLACE_EXPERT_FIX.md) |
| E03: initial-contact reject / outbound subdivision | 18/24 clean/task; 19/24 grasp; gate fails | [Gate V3](PACT_PLACE_CORRIDOR_GATE_V3.md) |
| E04: abort-branch diagnosis | Eight suspect rows reproduce `empty_gripper` abort; measured cup wall thickness 8.4–10.7 mm, not assumed 2 mm | [Diagnosis](PACT_PLACE_ABORT_BRANCH.md) |
| E05: placement-phase abort repair | 15/24 clean/task; gate fails | [Gate V4](PACT_PLACE_CORRIDOR_GATE_V4.md) |
| E06: tray relocation / contact exemption | 22/24 clean/task; gate passes | [Gate V5](PACT_PLACE_CORRIDOR_GATE_V5.md) |
| E07: demonstration recovery | 152/152 expert rerecordings reproduce outcomes without divergence; restores trainable observations/actions | [Recovery](PACT_PLACE_V5_DEMO_RECOVERY.md) |
| E08: V6 clutter probe | Two eight-row probes: task 7/8 each; clean 1/8 closest, 4/8 farthest; clutter contacts 6/8 and 3/8. Full gate not run | [V6](PACT_PLACE_CORRIDOR_GATE_V6.md) |
| E09: V6b outward clutter | 20/24 clean/task, no clutter contact; gate passes | [V6b](PACT_PLACE_CORRIDOR_GATE_V6B.md) |
| E10: V6c larger boxes | 23/24 clean/task, no clutter contact; gate passes | [V6c](PACT_PLACE_CORRIDOR_GATE_V6C.md) |
| E11: V7 swept-volume / clutter siting | Fixed 16-box candidate pool and swept-volume replay measured; later designs supersede it | [Swept-volume results](../diagnostics_output/pact_place_swept_volume_v7/analysis.json), [clutter results](../diagnostics_output/pact_place_clutter_sweep_v7/analysis.json) |
| E12: V8 real-object admission | Candidate layouts still only two objects, floor-only placement; realized cup closest 5/6 despite admission prediction 0/24 | [Measured shortcomings](PACT_PLACE_V8B_CLUTTER_PLAN.md), [analysis](../diagnostics_output/pact_place_clutter_sweep_v8/analysis.json) |
| E13: V8b dense-clutter admission | Realized cup closest 5/6, only 34 link-clearance<10 cm frames, mean 0.5 distinct links exposed, zero visibility-at-min; fails | [Results](../diagnostics_output/pact_place_clutter_sweep_v8b/analysis.json), [successor's result summary](PACT_PLACE_V8C_CLUTTER_PLAN.md) |
| E14: V8c overhead C0 siting | 567 candidates all keep cup from being closest; subsequent 667-candidate visibility scan remains 0/24 visible; chosen replay has substantial arm intrusion | Offline siting does not prove safe execution. [Siting](PACT_PLACE_V8C_C0_SITING.md) |
| E15: V9 panel and bottle revisions | Original V9 review: 0/24 clean, 8/24 task successes; redesigned review captures eight clean successes and three failures in 11 attempts. Outcome-selected review packet, not a population rate; followed by V9.3 raw-admission repairs | [Original review](../diagnostics_output/pact_place_v9_v1b/review_manifest.json), [redesign](../diagnostics_output/pact_place_v9_v1b_redesign/review_manifest.json), [raw-fixed smoke](../diagnostics_output/pact_place_v93_panel_smoke_rawfix/summary.json) |
| E16: V9.5 low-wall raw admission | Eight variants; all six physics-clean ones fail; sole nominal pass is a colliding F3 source | [Corrected status](PACT_PLACE_V95_LOW_WALL_STATUS.md) |
| E17: V9.6 cluster siting W2 | 7,140 placements each inbound/outbound; geometry-feasible 341/108; sensing-admitted 0/54. Of 18,393 paired clusters, none jointly feasible | [Status](PACT_PLACE_V96_CLUSTER_SITING_STATUS.md) |
| E18: V9.6 corridor-budget and W3 replay | Available hazard width 0.120 m versus approximately 0.250 m required contiguous silhouette; W3 left improves sensing but right imbalance 162.8×; dirty-source diagnostic, not admitted design | [Status](PACT_PLACE_V96_CLUSTER_SITING_STATUS.md) |
| E19: V9.5 seed fragility | 24 seeds average 4.08/8 clean (51%), range 0–7; four seeds reach 7/8. Smoke/replay discrepancy for one seed remains disclosed | [Evaluation ledger](../EVAL.md), [fragility](../diagnostics_output/pact_place_v95_seed_fragility/fragility.json) |

### A.2 Pendant geometry and expert routing

| ID / stage | Executed result and disposition | Source |
|---|---|---|
| E20: V9.8 ceiling pendant | First 0/24 screen void due to miswired expert; repaired zero-slack screen also 0/24 clean. Paired width and offset variants each 0/8; fixture-free guard 6/8. Later audit identifies route-composition coverage failure, not universal physical infeasibility | [Ledger](../EVAL.md), [plan/results](PACT_PLACE_V98_PENDANT_PLAN.md) |
| E21: V9.9 fixed rectangular pendant | Retained-qpos reconstruction residual ≤0.87 mm; zero exact survivors among 12,880 dual-transit AABB hits; no physics rollouts | Scoped lattice rejection. [Plan/results](PACT_PLACE_V99_PENDANT_PLAN.md) |
| E22: V10 compound pendant siting / two routing searches | Corrected siting yields 150,288 two-lobe survivors. First routing: 0/21,348 union×cell×direction passes. Endpoint-only revision admits 666,448 nominal IK cases but scalar environment check rejects all | Second rejection later identified as flawed predicate, not physical impossibility; no episodes. [Ledger](../EVAL.md), [plan](PACT_PLACE_V10_COMPOUND_PENDANT_PLAN.md) |
| E23: V10.1 empirical qualification | 12 attempts, 11 complete + one sampling failure, 0/12 clean; eight IK cascades, three clutter/stability failures | Runtime “zero pendant contact” later invalidated as clearance evidence by E24. [Summary](../diagnostics_output/pact_place_v101_empirical_review/summary.json) |
| E24: V10.2 raised pendant / collision instrumentation | Only 5/12 sequential IK cases pass; 0/12 meet clearance. Nineteen penetrating poses generate zero contacts due to stale bounds. Separate requested gallery: 0/12 clean, all with negative geometric clearance | Instrument defect is measured; regular gate stops before episodes. [Ledger](../EVAL.md), [plan/results](PACT_PLACE_V102_RAISED_PENDANT_REMEDIATION_PLAN.md) |
| E25: V10.3 static pendant joint routing | Nine of 12 cases evaluated, none feasible; pinned endpoints provide a rejecting witness for the registered selection rule; no episodes | Not an exhaustive global path-infeasibility result. [Plan/results](PACT_PLACE_V103_STATIC_PENDANT_IK_PHASE0_PLAN.md) |
| E26: V10.4 static outboard pendant | 6/6 expert episodes clean; minimum clearance 61.58 mm; causal changed values 23,684 left / 12,712 right; review control grid fails before formal Phase 0 | [Plan/results](PACT_PLACE_V104_FIRST_SHOT_STATIC_PENDANT_PLAN.md), [ledger](../EVAL.md) |
| E27: V10.4 review-control repair | Reuses same six episodes; three contact controls certified and six videos published; no new policy/expert rollout | [Review repair](PACT_PLACE_V104_REVIEW_REPAIR_PLAN.md) |
| E28: V10.5 static pendant on real clutter | Audits 192 source rows, 98 clean; scores all 96 scenes × 98 trajectories; zero surviving bundles against clearance/witness criteria | [Plan/results](PACT_PLACE_V105_V95_CLUTTER_STATIC_PENDANT_PLAN.md), [corrected audit](../diagnostics_output/pact_place_v105_audit/audit.json) |
| E29: V10.6 asymmetric pendant | Nine admitted candidates, four universally clear; selected minimum 18.5703 mm; causal tests 7/7 pass, contact-risk criterion 0/6 fails | [Plan/results](PACT_PLACE_V106_ASYMMETRIC_PENDANT_PLAN.md) |
| E30: V10.7 qualification repair | Six-group causal tests pass; expert pool 21/48 clean (43.8%) fails 32/48 floor; later Phase-0 record 8/24 fails. Review clips do not overturn failures | [Plan/results](PACT_PLACE_V107_QUALIFICATION_REPAIR_PLAN.md), [ledger](../EVAL.md) |

### A.3 Collections, validation, and later environment revisions

| ID / stage | Recorded result | Source |
|---|---|---|
| E31: V10.8 exploratory collection | 141 accepted strict-clean demos from 353 ledger rows; stopped before 152 target, five cells short. Eight worker deaths occurred on requested stop. Accepted rows had no pendant contact; a rejected row did | [V10.9 source audit](PACT_PLACE_V109_TRAIN_EVAL_PLAN.md), [ledger erratum](../EVAL.md) |
| E32: initial V10.10 four-object collection | 144 accepted strict-clean demos from 313 attempts, six per cell, split 120/24 | [Plan/results](PACT_PLACE_V1010_FOUR_OBJECT_TRAIN_EVAL_PLAN.md) |
| E33: V10.11 mixed mesh/primitive review | 96/96 preflight checks; 12 completed review attempts; selected packet three clean successes and three natural failures | Packet is not a 50% population success estimate. [Ledger](../EVAL.md) |
| E34: V10.11b/c taller-primitive development and collection | Retained review revisions precede V10.11c corpus of 99 accepted episodes from 482 attempts, split 75/24 | [Training/evaluation record](../diagnostics_output/pact_place_v1011c_eval/report.md), [project log](../README.md) |
| E35: V10.11d environment review | Randomized primitive/environment review artifacts retained; separate from user's intended local V10.10c source experiment and remote T-1011d policy evaluation | [Review](../diagnostics_output/pact_place_v1011d_review/REVIEW.md) |
| E36: V10.10 table-camera validation | Ten-demo validation/collection closeout retained | Observation/data qualification, not paired task advantage. [Closeout](../diagnostics_output/pact_place_v1010_tablecam_validation10/closeout.json) |
| E37: wrist280 collection and corpus expansion | Actual retained corpus 280, train/validation 240/40 despite `wrist288` directory name | Constituent data stage of L09/L11. [Three-method report](PACT_PLACE_V1010C_THREE_METHOD_COMPARISON.md) |
| E38: manuscript's six collection families | Manuscript lists Empty 313, Spaced-8 200, Clutter-4 144, Prim-fixed 100, Prim-random 200, Dense-10 165; sum 1,122 | **Manuscript-reported inventory, not six verified policy benchmarks.** These counts cannot be mapped indiscriminately to local attempts, remote converted packs, or Y02's duplicated 165-success count |
| E39: early enclosure/cubby/fumehood scene prototypes | CSV/probe results are detailed in A.4; other dated directories contain sensor-render and expert-debug artifacts | Engineering prototypes, not learned-policy finetuning despite some directory names |
| E40: V9.4 mounted-fixture previews | Success/clean counts across smoke revisions: 1/2 and 1/2; then 1/1 and 0/1; then 5/6 and 4/6. Development revisions with distinct geometry, not pooled trials | [First smoke](../diagnostics_output/pact_place_v94_mounted_preview_smoke), [second](../diagnostics_output/pact_place_v94_mounted_preview_smoke2), [third](../diagnostics_output/pact_place_v94_mounted_preview_smoke3) |
| E41: V9.7 E1b raw confirmation | Six physics-clean source variants; overall admission fails and posed span/gap contract fails; physics regeneration required for admission | [Validation](../diagnostics_output/pact_place_v97_e1b_raw_confirmation/validation.json) |
| E42: initial hazard-contact diagnosis | Two original V2 rows: contact already at first observation, lasting 82/171 physics frames; initial maximum penetration 5.54/20.46 mm | Distinguishes invalid reset from inbound avoidance failure. [Diagnosis](PACT_PLACE_HAZARD_ROWS_6_12.md) |
| E43: V6b contact-body diagnosis | Two diagnostic reruns have 1,990/2,406 clutter-contact frames, all from carried cup; zero robot-link pairs | Explains outward clutter relocation. [Diagnosis](PACT_PLACE_V6B_CONTACT_BODIES.md) |

The table's E38 counts are retained to make the manuscript discrepancy visible: the remote T-107 converted pack is 210 demonstrations, the remote T-1010 pack 215, and local V10.10 has both an initial 144-demo corpus and a later 280-demo corpus. Dataset names are insufficient to merge these records.

### A.4 Early enclosure iterations: retained CSV outcomes

These June-10 development datasets contain mixed free/visible/hidden/abort cells. The table counts the CSV's `success` flag, which is **not necessarily ordinary pick-and-place completion**, particularly in abort cells. Repeated house content and independence have not been revalidated, so these counts should not become paper success-rate comparisons. The first interim export overlaps its later export.

| Artifact / development revision | Recorded rows | `success=true` | Speed–nearest-distance correlation in saved probe |
|---|---:|---:|---:|
| [Initial enclosure interim](../diagnostics_output/20260610_0340_enclosure_finetune1_interim/episodes.csv) | 15 | 5 | +0.154 |
| [Initial enclosure final export](../diagnostics_output/20260610_0346_enclosure_finetune1/episodes.csv) | 18 | 7 | +0.159 |
| [Front-frame revision](../diagnostics_output/20260610_0354_enclosure_finetune2_frontframe/episodes.csv) | 0 | — | No saved episode outcomes |
| [Fumehood 1](../diagnostics_output/20260610_0424_enclosure_fumehood1/episodes.csv) | 24 | 7 | +0.054 |
| [Panel 1](../diagnostics_output/20260610_0457_enclosure_panel1/episodes.csv) | 24 | 1 | +0.272 |
| [Cubby 1](../diagnostics_output/20260610_0536_enclosure_cubby1/episodes.csv) | 24 | 15 | +0.403 |
| [Fumehood 2](../diagnostics_output/20260610_0737_enclosure_fumehood2/episodes.csv) | 24 | 8 | −0.042 |
| [Panel 2](../diagnostics_output/20260610_0805_enclosure_panel2/episodes.csv) | 24 | 6 | −0.008 |
| [Cubby 2](../diagnostics_output/20260610_0854_enclosure_cubby2/episodes.csv) | 24 | 11 | +0.059 |
| [Fumehood 3](../diagnostics_output/20260610_0925_enclosure_fumehood3/episodes.csv) | 24 | 6 | +0.088 |
| [Panel 3](../diagnostics_output/20260610_0957_enclosure_panel3/episodes.csv) | 24 | 7 | −0.047 |
| [Cubby 3](../diagnostics_output/20260610_1040_enclosure_cubby3/episodes.csv) | 24 | 5 | +0.251 |

The adjacent `probes.txt` files record speed-response, deflection decoding, visual-shortcut correlation, and clearance-distribution checks. Most revisions fail the registered speed-response check; several have too few deflection examples, and cubby revisions are flagged for overly generous clearance. A successful geometry/debug recording is not a learned-policy result.

## Appendix B. Exact configuration and naming map

| Family | Data / training | Evaluation / important distinction |
|---|---|---|
| Archived O-INV | Reported 105 demos; raw/trunk obstacle policies | Pickup, horizon 200, n=50 per cell; old broad collision counter; checkpoints/data deleted |
| H-A/H-B | 152 demos; ACT hidden 512, FF 3,200, KL 10, batch 8, lr 1e−5, 2,000 epochs; chunk 50 | Wrist RGB 240×320, open-loop chunks, horizon 800; remote best-checkpoint convention |
| Historical hallway encoder | 837,700 parameters; temporal width 128, four layers, four heads, FF 256; 8 pooled control frames repeated fourfold; 20 cm geometry range | Readout 40 CLS tokens; scalar baseline's historical expansion uses 320 tokens. Query-history/subframe mismatch is disclosed in remote paper |
| Historical pretraining split | 122/15/15 encoder split; policy split 121/31; remote audit says 24/31 policy-validation episodes entered encoder pretraining and policy normalization used all episodes | Leakage caveat belongs to that historical pipeline; do not assume identical handling in later local runs |
| Local pickup P02–P09 | Frozen 32-D encoder; training seeds 3101–3103 as applicable | Shared physical instances across seeds in the larger studies; original 512-frame control plan amended to horizon 900 before scientific outcomes |
| Local initial V10.10 | 144 demos, 120/24; seed 3101; 2,000 epochs, chunk 100 | Horizon 900, temporal ensemble, wrist only |
| Local V10.11c | 99 demos, 75/24; seeds 3103–3105; 2,000 epochs | Taller-primitive task with low success; not the latest four-object corpus |
| Local latest V10.10/V10.10c | 280 demos, 240/40; seeds 3103–3105; 60,000 updates; final checkpoints | Same source study the user clarified after calling it “v1010d”; 900 actions, 901 observations, 59.4 simulated seconds |
| `wrist288` | Directory/run label | Actual final corpus 280; do not cite 288 as actual retained training size |
| Recovered “V5” | 152 demos, `PactPlaceCorridorV2Sampler` | Different from cluttered V5-family scene samplers |
| Remote T-1011d | Scalar/raw PACT, randomized clutter | Different source from local three-seed V10.10c and from an environment-review-only artifact |

For exact local checkpoint/encoder pairing, normalization hashes, source commits and controller arguments, use the experiment's linked final report and manifests. This inventory does not invent a single canonical checkpoint path for all studies.

## Appendix C. Statistics and denominators

The quoted confidence intervals and p-values are those in the source analyses, except direct counts/strict-success intersections read from remote JSONs. They are not new significance tests selected across the whole inventory.

- Original unmatched hallway binary comparisons use two-sided Fisher tests; the paper gives Wilson intervals for marginal rates.
- H-B's matched binary contrasts use discordant pairs and exact McNemar tests. Its mixed training seeds still prevent a clean seed-replicated method comparison.
- The local pickup contact, geometry, blur, and blind analyses use whole-instance cluster bootstrap intervals: repeated seeds and methods for an instance move together.
- Local three-seed placement has different scene blocks per seed. Pooled rates are descriptive averages over three observed checkpoint/block pairs, not evidence from 150 independent trained policies.
- Timing confidence intervals resample shared-success scene pairs within observed seed blocks; they condition on both success and the three observed trained models.
- A CI including zero is inconclusive, not proof of equivalence. A nonsignificant decline does not establish performance retention. Contact reduction alongside collapse in task completion is not successful safe manipulation.
- Post-hoc subsets such as high-contact tails, shared successes, and specific failure directions are labeled analyses; selection into those subsets limits causal conclusions.

## Appendix D. Evidence and figure inventory

### D.1 Main evidence and existing figures

| Asset | Purpose |
|---|---|
| [Remote PAPER.md at the audited commit][paper] | Requested document format; historical experiment grouping and caveats |
| [Remote README at the audited commit][readme] | Historical research log and training-only/partial-run inventory |
| [Remote evaluation summaries][remote-evals] | Per-arm counts and episode records for hallway, spaced bench, visibility, training blur, and diagnostic evaluations |
| [Earlier missing-results audit](PAPER_MISSING_RESULTS_AUDIT_20260912.md) | Local cross-study reconciliation and retained-artifact pointers |
| [Local EVAL.md](../EVAL.md) | Placement development, expert gates, corpus construction, and earlier policy evaluations |
| [Latest three-method report](PACT_PLACE_V1010C_THREE_METHOD_COMPARISON.md) | Current 128-D/frozen/ACT outcome comparison |
| [Completion-time figure PDF](../diagnostics_output/pact_place_v1010c_completion_time_20260916/completion_time_comparison.pdf) | Existing conditional completion-time comparison; also [PNG](../diagnostics_output/pact_place_v1010c_completion_time_20260916/completion_time_comparison.png) |
| [Historical August report][aug-report] | Visibility and training-blur figures, with interpretation corrected by this inventory |
| [Sensor/trajectory example](../diagnostics_output/pact_house11_traj0/report.md) | Early 29-sensor visualization, explicitly a different sensor contract |

Scientific failure records, invalid initial attempts, repair reports, and later successful infrastructure retries should remain together in any release. A corrected summary supersedes an erroneous interpretation without deleting the original attempt.

### D.2 Machine-readable source manifest

The companion [source manifest](PROJECT_EXPERIMENTS_20260916.sources.json) records paths, SHA-256 hashes, source type, and remote commit for the material indexed during this audit. A listed file is an evidence pointer, not automatically a distinct experiment or a verified completed rollout. The readable source tables below include retained report families and remote per-arm summaries; all raw trajectory files are intentionally not expanded into this document.

### D.3 Remote per-arm results, read directly from saved JSONs

Counts below preserve each file separately. **Any contact** and **hazard contact** are different endpoints. Legacy diagnostic rows can be invalid for scientific comparison (O01); a zero-success technical diagnostic is not a new negative policy benchmark. Strict counts are stored directly or reconstructed as final success intersected with collision-free in the saved episode records.

| Saved result file | n | Task | Strict | Hazard contact | Any contact |
|---|---:|---:|---:|---:|---:|
| [20260724_010935_vanilla_blurC2_v2_free.json](https://github.com/Jdvakil/prox_learning/blob/a5de7a4dd42a075d99b8f5865ec7baa9479a1fdd/reports/eval_summaries/20260724_010935_vanilla_blurC2_v2_free.json) | 25 | 1 | 0 | — | 12 |
| [20260724_010935_vanilla_blurC2_v2_invisible.json](https://github.com/Jdvakil/prox_learning/blob/a5de7a4dd42a075d99b8f5865ec7baa9479a1fdd/reports/eval_summaries/20260724_010935_vanilla_blurC2_v2_invisible.json) | 25 | 0 | 0 | — | 12 |
| [20260724_010935_vanilla_blurC2_v2_visible.json](https://github.com/Jdvakil/prox_learning/blob/a5de7a4dd42a075d99b8f5865ec7baa9479a1fdd/reports/eval_summaries/20260724_010935_vanilla_blurC2_v2_visible.json) | 25 | 2 | 2 | — | 13 |
| [20260724_015648_vanilla_blurC4_v2_free.json](https://github.com/Jdvakil/prox_learning/blob/a5de7a4dd42a075d99b8f5865ec7baa9479a1fdd/reports/eval_summaries/20260724_015648_vanilla_blurC4_v2_free.json) | 25 | 10 | 3 | — | 21 |
| [20260724_015648_vanilla_blurC4_v2_invisible.json](https://github.com/Jdvakil/prox_learning/blob/a5de7a4dd42a075d99b8f5865ec7baa9479a1fdd/reports/eval_summaries/20260724_015648_vanilla_blurC4_v2_invisible.json) | 25 | 10 | 1 | — | 22 |
| [20260724_015648_vanilla_blurC4_v2_visible.json](https://github.com/Jdvakil/prox_learning/blob/a5de7a4dd42a075d99b8f5865ec7baa9479a1fdd/reports/eval_summaries/20260724_015648_vanilla_blurC4_v2_visible.json) | 25 | 6 | 2 | — | 21 |
| [20260724_024536_vanilla_blurC8_v2_free.json](https://github.com/Jdvakil/prox_learning/blob/a5de7a4dd42a075d99b8f5865ec7baa9479a1fdd/reports/eval_summaries/20260724_024536_vanilla_blurC8_v2_free.json) | 25 | 6 | 4 | — | 7 |
| [20260724_024536_vanilla_blurC8_v2_invisible.json](https://github.com/Jdvakil/prox_learning/blob/a5de7a4dd42a075d99b8f5865ec7baa9479a1fdd/reports/eval_summaries/20260724_024536_vanilla_blurC8_v2_invisible.json) | 25 | 6 | 2 | — | 17 |
| [20260724_024536_vanilla_blurC8_v2_visible.json](https://github.com/Jdvakil/prox_learning/blob/a5de7a4dd42a075d99b8f5865ec7baa9479a1fdd/reports/eval_summaries/20260724_024536_vanilla_blurC8_v2_visible.json) | 25 | 6 | 4 | — | 17 |
| [pact_pick_n_place_v2_v1011d_raw_s0_n48_horizon1050.json](https://github.com/Jdvakil/prox_learning/blob/a5de7a4dd42a075d99b8f5865ec7baa9479a1fdd/reports/eval_summaries/pact_pick_n_place_v2_v1011d_raw_s0_n48_horizon1050.json) | 48 | 0 | 0 | 0 | 11 |
| [pact_pick_n_place_v2_v1011d_raw_s0_n48_horizon800.json](https://github.com/Jdvakil/prox_learning/blob/a5de7a4dd42a075d99b8f5865ec7baa9479a1fdd/reports/eval_summaries/pact_pick_n_place_v2_v1011d_raw_s0_n48_horizon800.json) | 48 | 0 | 0 | 3 | 16 |
| [pact_place_corridor_v107_spaced_ACT_s0_bs8_cs50_lr1e-5_e2000.json](https://github.com/Jdvakil/prox_learning/blob/a5de7a4dd42a075d99b8f5865ec7baa9479a1fdd/reports/eval_summaries/pact_place_corridor_v107_spaced_ACT_s0_bs8_cs50_lr1e-5_e2000.json) | 50 | 13 | 7 | 8 | 31 |
| [pact_place_corridor_v107_spaced_PACT_RAW_s0_bs8_cs50_lr1e-5_e2000.json](https://github.com/Jdvakil/prox_learning/blob/a5de7a4dd42a075d99b8f5865ec7baa9479a1fdd/reports/eval_summaries/pact_place_corridor_v107_spaced_PACT_RAW_s0_bs8_cs50_lr1e-5_e2000.json) | 50 | 0 | 0 | 9 | 31 |
| [pact_place_corridor_v107_spaced_PACT_READOUT_s0_bs8_cs50_lr1e-5_e2000.json](https://github.com/Jdvakil/prox_learning/blob/a5de7a4dd42a075d99b8f5865ec7baa9479a1fdd/reports/eval_summaries/pact_place_corridor_v107_spaced_PACT_READOUT_s0_bs8_cs50_lr1e-5_e2000.json) | 50 | 2 | 2 | 6 | 30 |
| [pact_place_corridor_v5_ACT_s1_bs8_cs50_lr1e-5_e2000.json](https://github.com/Jdvakil/prox_learning/blob/a5de7a4dd42a075d99b8f5865ec7baa9479a1fdd/reports/eval_summaries/pact_place_corridor_v5_ACT_s1_bs8_cs50_lr1e-5_e2000.json) | 50 | 15 | 12 | 20 | 20 |
| [pact_place_corridor_v5_PACT_RAW_s0_bs8_cs50_lr1e-5_e2000.json](https://github.com/Jdvakil/prox_learning/blob/a5de7a4dd42a075d99b8f5865ec7baa9479a1fdd/reports/eval_summaries/pact_place_corridor_v5_PACT_RAW_s0_bs8_cs50_lr1e-5_e2000.json) | 50 | 21 | 19 | 13 | 13 |
| [pact_place_corridor_v5_PACT_READOUT_s0_bs8_cs50_lr1e-5_e2000.json](https://github.com/Jdvakil/prox_learning/blob/a5de7a4dd42a075d99b8f5865ec7baa9479a1fdd/reports/eval_summaries/pact_place_corridor_v5_PACT_READOUT_s0_bs8_cs50_lr1e-5_e2000.json) | 50 | 21 | 19 | 9 | 9 |
| [pact_raw_v2_free.json](https://github.com/Jdvakil/prox_learning/blob/a5de7a4dd42a075d99b8f5865ec7baa9479a1fdd/reports/eval_summaries/pact_raw_v2_free.json) | 50 | 9 | 4 | — | 29 |
| [pact_raw_v2_invisible.json](https://github.com/Jdvakil/prox_learning/blob/a5de7a4dd42a075d99b8f5865ec7baa9479a1fdd/reports/eval_summaries/pact_raw_v2_invisible.json) | 50 | 15 | 10 | — | 20 |
| [pact_raw_v2_visible.json](https://github.com/Jdvakil/prox_learning/blob/a5de7a4dd42a075d99b8f5865ec7baa9479a1fdd/reports/eval_summaries/pact_raw_v2_visible.json) | 50 | 8 | 5 | — | 25 |
| [pact_trunk_v2_free.json](https://github.com/Jdvakil/prox_learning/blob/a5de7a4dd42a075d99b8f5865ec7baa9479a1fdd/reports/eval_summaries/pact_trunk_v2_free.json) | 50 | 17 | 8 | — | 32 |
| [pact_trunk_v2_invisible.json](https://github.com/Jdvakil/prox_learning/blob/a5de7a4dd42a075d99b8f5865ec7baa9479a1fdd/reports/eval_summaries/pact_trunk_v2_invisible.json) | 50 | 17 | 5 | — | 36 |
| [pact_trunk_v2_visible.json](https://github.com/Jdvakil/prox_learning/blob/a5de7a4dd42a075d99b8f5865ec7baa9479a1fdd/reports/eval_summaries/pact_trunk_v2_visible.json) | 50 | 16 | 9 | — | 29 |
| [place_corridor_raw_s0_n50.json](https://github.com/Jdvakil/prox_learning/blob/a5de7a4dd42a075d99b8f5865ec7baa9479a1fdd/reports/eval_summaries/place_corridor_raw_s0_n50.json) | 50 | 21 | 17 | 18 | 18 |
| [place_corridor_readout_s0_n50_fast.json](https://github.com/Jdvakil/prox_learning/blob/a5de7a4dd42a075d99b8f5865ec7baa9479a1fdd/reports/eval_summaries/place_corridor_readout_s0_n50_fast.json) | 50 | 20 | 20 | 6 | 6 |
| [place_corridor_vanilla_s0_n50.json](https://github.com/Jdvakil/prox_learning/blob/a5de7a4dd42a075d99b8f5865ec7baa9479a1fdd/reports/eval_summaries/place_corridor_vanilla_s0_n50.json) | 50 | 14 | 13 | 17 | 17 |
| [simple_hallway_n50.json](https://github.com/Jdvakil/prox_learning/blob/a5de7a4dd42a075d99b8f5865ec7baa9479a1fdd/reports/eval_summaries/simple_hallway_n50.json) | 50 | 18 | 18 | 7 | 7 |
| [simple_v1011d_easy025_n50.json](https://github.com/Jdvakil/prox_learning/blob/a5de7a4dd42a075d99b8f5865ec7baa9479a1fdd/reports/eval_summaries/simple_v1011d_easy025_n50.json) | 50 | 14 | 11 | 3 | 28 |
| [simple_v1011d_smoke_video.json](https://github.com/Jdvakil/prox_learning/blob/a5de7a4dd42a075d99b8f5865ec7baa9479a1fdd/reports/eval_summaries/simple_v1011d_smoke_video.json) | 50 | 7 | 4 | 10 | 30 |
| [v1_vanilla_freecell_diag10.json](https://github.com/Jdvakil/prox_learning/blob/a5de7a4dd42a075d99b8f5865ec7baa9479a1fdd/reports/eval_summaries/v1_vanilla_freecell_diag10.json) | 10 | 0 | 0 | — | 0 |
| [v1_vanilla_nocell_aggON_diag10.json](https://github.com/Jdvakil/prox_learning/blob/a5de7a4dd42a075d99b8f5865ec7baa9479a1fdd/reports/eval_summaries/v1_vanilla_nocell_aggON_diag10.json) | 10 | 5 | 3 | — | 5 |
| [v1_vanilla_nocell_diag10.json](https://github.com/Jdvakil/prox_learning/blob/a5de7a4dd42a075d99b8f5865ec7baa9479a1fdd/reports/eval_summaries/v1_vanilla_nocell_diag10.json) | 10 | 0 | 0 | — | 0 |
| [vanilla_v2_free.json](https://github.com/Jdvakil/prox_learning/blob/a5de7a4dd42a075d99b8f5865ec7baa9479a1fdd/reports/eval_summaries/vanilla_v2_free.json) | 50 | 11 | 4 | — | 30 |
| [vanilla_v2_free_diag10.json](https://github.com/Jdvakil/prox_learning/blob/a5de7a4dd42a075d99b8f5865ec7baa9479a1fdd/reports/eval_summaries/vanilla_v2_free_diag10.json) | 10 | 0 | 0 | — | 0 |
| [vanilla_v2_free_diag10_fixed.json](https://github.com/Jdvakil/prox_learning/blob/a5de7a4dd42a075d99b8f5865ec7baa9479a1fdd/reports/eval_summaries/vanilla_v2_free_diag10_fixed.json) | 10 | 4 | 1 | — | 7 |
| [vanilla_v2_invisible.json](https://github.com/Jdvakil/prox_learning/blob/a5de7a4dd42a075d99b8f5865ec7baa9479a1fdd/reports/eval_summaries/vanilla_v2_invisible.json) | 50 | 18 | 7 | — | 33 |
| [vanilla_v2_nocell_diag10.json](https://github.com/Jdvakil/prox_learning/blob/a5de7a4dd42a075d99b8f5865ec7baa9479a1fdd/reports/eval_summaries/vanilla_v2_nocell_diag10.json) | 10 | 0 | 0 | — | 2 |
| [vanilla_v2_visible.json](https://github.com/Jdvakil/prox_learning/blob/a5de7a4dd42a075d99b8f5865ec7baa9479a1fdd/reports/eval_summaries/vanilla_v2_visible.json) | 50 | 14 | 8 | — | 32 |

### D.4 Local output directory catalog

This audit found **278 immediate output directories**. They are grouped below so that small smokes, visualizations, retries, and superseded artifacts remain discoverable. **This is not a count of 278 independent experiments.** The manifest indexes report files within these folders; interpretation and corrected results are in the sections above.

**Clutter observability and raw-admission development (E15–E19, E40–E41)**

| Output directory | Retained direct report files indexed |
|---|---:|
| [pact_place_v93_panel_smoke](../diagnostics_output/pact_place_v93_panel_smoke) | 1 |
| [pact_place_v93_panel_smoke_rawfix](../diagnostics_output/pact_place_v93_panel_smoke_rawfix) | 1 |
| [pact_place_v93_success_episode_review](../diagnostics_output/pact_place_v93_success_episode_review) | 1 |
| [pact_place_v93_v0c4_causal_proximity](../diagnostics_output/pact_place_v93_v0c4_causal_proximity) | 1 |
| [pact_place_v94_mounted_preview](../diagnostics_output/pact_place_v94_mounted_preview) | 0 |
| [pact_place_v94_mounted_preview_smoke](../diagnostics_output/pact_place_v94_mounted_preview_smoke) | 1 |
| [pact_place_v94_mounted_preview_smoke2](../diagnostics_output/pact_place_v94_mounted_preview_smoke2) | 1 |
| [pact_place_v94_mounted_preview_smoke3](../diagnostics_output/pact_place_v94_mounted_preview_smoke3) | 1 |
| [pact_place_v95_raw_smoke](../diagnostics_output/pact_place_v95_raw_smoke) | 1 |
| [pact_place_v95_seed_fragility](../diagnostics_output/pact_place_v95_seed_fragility) | 3 |
| [pact_place_v95_smoke_repro](../diagnostics_output/pact_place_v95_smoke_repro) | 0 |
| [pact_place_v95_smoke_repro_guard](../diagnostics_output/pact_place_v95_smoke_repro_guard) | 1 |
| [pact_place_v95_smoke_repro_guard_v3](../diagnostics_output/pact_place_v95_smoke_repro_guard_v3) | 1 |
| [pact_place_v95_smoke_repro_guard_v4](../diagnostics_output/pact_place_v95_smoke_repro_guard_v4) | 1 |
| [pact_place_v95_smoke_repro_guard_v5](../diagnostics_output/pact_place_v95_smoke_repro_guard_v5) | 1 |
| [pact_place_v95_v0c5_candidate01_raw](../diagnostics_output/pact_place_v95_v0c5_candidate01_raw) | 1 |
| [pact_place_v95_v0c5_candidate02_f2](../diagnostics_output/pact_place_v95_v0c5_candidate02_f2) | 1 |
| [pact_place_v95_v0c5_candidate03_f2](../diagnostics_output/pact_place_v95_v0c5_candidate03_f2) | 1 |
| [pact_place_v95_v0c5_raw_prerequisite](../diagnostics_output/pact_place_v95_v0c5_raw_prerequisite) | 2 |
| [pact_place_v95_v0c5_raw_siting](../diagnostics_output/pact_place_v95_v0c5_raw_siting) | 1 |
| [pact_place_v96_cluster_review](../diagnostics_output/pact_place_v96_cluster_review) | 1 |
| [pact_place_v96_w3_pipeline_validation](../diagnostics_output/pact_place_v96_w3_pipeline_validation) | 1 |
| [pact_place_v97_e1b_raw_confirmation](../diagnostics_output/pact_place_v97_e1b_raw_confirmation) | 1 |
| [pact_place_v9_e1_subtense_siting](../diagnostics_output/pact_place_v9_e1_subtense_siting) | 0 |
| [pact_place_v9_layout_redesign](../diagnostics_output/pact_place_v9_layout_redesign) | 1 |
| [pact_place_v9_panel_gallery](../diagnostics_output/pact_place_v9_panel_gallery) | 0 |
| [pact_place_v9_panel_redesign](../diagnostics_output/pact_place_v9_panel_redesign) | 1 |
| [pact_place_v9_panel_smoke](../diagnostics_output/pact_place_v9_panel_smoke) | 1 |
| [pact_place_v9_smoke_redesign](../diagnostics_output/pact_place_v9_smoke_redesign) | 0 |
| [pact_place_v9_v0a](../diagnostics_output/pact_place_v9_v0a) | 1 |
| [pact_place_v9_v0b](../diagnostics_output/pact_place_v9_v0b) | 0 |
| [pact_place_v9_v0c](../diagnostics_output/pact_place_v9_v0c) | 1 |
| [pact_place_v9_v0c3_causal_proximity](../diagnostics_output/pact_place_v9_v0c3_causal_proximity) | 1 |
| [pact_place_v9_v1b](../diagnostics_output/pact_place_v9_v1b) | 1 |
| [pact_place_v9_v1b_redesign](../diagnostics_output/pact_place_v9_v1b_redesign) | 1 |
| [pact_place_v9_visual_gallery](../diagnostics_output/pact_place_v9_visual_gallery) | 0 |
| [pact_place_v9_w1_resolvability](../diagnostics_output/pact_place_v9_w1_resolvability) | 1 |
| [pact_place_v9_w1_resolvability_full](../diagnostics_output/pact_place_v9_w1_resolvability_full) | 1 |
| [pact_place_v9_w2_cluster_siting](../diagnostics_output/pact_place_v9_w2_cluster_siting) | 0 |
| [pact_place_v9_w2b_inbound_diagnostic](../diagnostics_output/pact_place_v9_w2b_inbound_diagnostic) | 0 |

**Earlier placement, clutter, expert gates, recovery (L01–L03 / E01–E14)**

| Output directory | Retained direct report files indexed |
|---|---:|
| [pact_place_152_pact_vs_act](../diagnostics_output/pact_place_152_pact_vs_act) | 1 |
| [pact_place_abort_branch](../diagnostics_output/pact_place_abort_branch) | 1 |
| [pact_place_clutter_sweep](../diagnostics_output/pact_place_clutter_sweep) | 2 |
| [pact_place_clutter_sweep_v6b](../diagnostics_output/pact_place_clutter_sweep_v6b) | 1 |
| [pact_place_clutter_sweep_v6c](../diagnostics_output/pact_place_clutter_sweep_v6c) | 1 |
| [pact_place_clutter_sweep_v7](../diagnostics_output/pact_place_clutter_sweep_v7) | 1 |
| [pact_place_clutter_sweep_v8](../diagnostics_output/pact_place_clutter_sweep_v8) | 1 |
| [pact_place_clutter_sweep_v8b](../diagnostics_output/pact_place_clutter_sweep_v8b) | 3 |
| [pact_place_corridor](../diagnostics_output/pact_place_corridor) | 2 |
| [pact_place_corridor_v2](../diagnostics_output/pact_place_corridor_v2) | 2 |
| [pact_place_corridor_v2_diagnostic_original_seeds](../diagnostics_output/pact_place_corridor_v2_diagnostic_original_seeds) | 1 |
| [pact_place_corridor_v2_hazard_rows_6_12_frames](../diagnostics_output/pact_place_corridor_v2_hazard_rows_6_12_frames) | 1 |
| [pact_place_corridor_v3](../diagnostics_output/pact_place_corridor_v3) | 2 |
| [pact_place_corridor_v3_videos](../diagnostics_output/pact_place_corridor_v3_videos) | 1 |
| [pact_place_corridor_v4](../diagnostics_output/pact_place_corridor_v4) | 2 |
| [pact_place_corridor_v5](../diagnostics_output/pact_place_corridor_v5) | 1 |
| [pact_place_corridor_v5_clearance_probe](../diagnostics_output/pact_place_corridor_v5_clearance_probe) | 1 |
| [pact_place_corridor_v5_collection](../diagnostics_output/pact_place_corridor_v5_collection) | 1 |
| [pact_place_corridor_v5_videos](../diagnostics_output/pact_place_corridor_v5_videos) | 1 |
| [pact_place_corridor_v6_clearance_probe](../diagnostics_output/pact_place_corridor_v6_clearance_probe) | 1 |
| [pact_place_corridor_v6_clearance_probe_y028](../diagnostics_output/pact_place_corridor_v6_clearance_probe_y028) | 1 |
| [pact_place_corridor_v6b](../diagnostics_output/pact_place_corridor_v6b) | 1 |
| [pact_place_corridor_v6b_clearance_probe](../diagnostics_output/pact_place_corridor_v6b_clearance_probe) | 1 |
| [pact_place_corridor_v6b_contact_bodies](../diagnostics_output/pact_place_corridor_v6b_contact_bodies) | 1 |
| [pact_place_corridor_v6b_videos](../diagnostics_output/pact_place_corridor_v6b_videos) | 1 |
| [pact_place_corridor_v6c](../diagnostics_output/pact_place_corridor_v6c) | 1 |
| [pact_place_corridor_v6c_clearance_probe](../diagnostics_output/pact_place_corridor_v6c_clearance_probe) | 1 |
| [pact_place_corridor_v6c_videos](../diagnostics_output/pact_place_corridor_v6c_videos) | 1 |
| [pact_place_corridor_v7_design_review](../diagnostics_output/pact_place_corridor_v7_design_review) | 1 |
| [pact_place_corridor_v8_family_review](../diagnostics_output/pact_place_corridor_v8_family_review) | 2 |
| [pact_place_corridor_v8_scoring_check](../diagnostics_output/pact_place_corridor_v8_scoring_check) | 0 |
| [pact_place_corridor_v8b_mount_scoring_check](../diagnostics_output/pact_place_corridor_v8b_mount_scoring_check) | 0 |
| [pact_place_corridor_v8b_pass2](../diagnostics_output/pact_place_corridor_v8b_pass2) | 0 |
| [pact_place_corridor_v8b_pass2b](../diagnostics_output/pact_place_corridor_v8b_pass2b) | 0 |
| [pact_place_corridor_v8b_pass2c](../diagnostics_output/pact_place_corridor_v8b_pass2c) | 0 |
| [pact_place_corridor_v8c_c0](../diagnostics_output/pact_place_corridor_v8c_c0) | 1 |
| [pact_place_corridor_v8c_c0_review](../diagnostics_output/pact_place_corridor_v8c_c0_review) | 1 |
| [pact_place_reachability_sweep](../diagnostics_output/pact_place_reachability_sweep) | 1 |
| [pact_place_swept_volume_v7](../diagnostics_output/pact_place_swept_volume_v7) | 1 |
| [pact_place_v5_paper_views](../diagnostics_output/pact_place_v5_paper_views) | 0 |
| [pact_place_v5_recovery](../diagnostics_output/pact_place_v5_recovery) | 0 |
| [pact_place_v5_thread_probe](../diagnostics_output/pact_place_v5_thread_probe) | 0 |
| [pact_place_v8_baseline](../diagnostics_output/pact_place_v8_baseline) | 1 |

**Early scene, sensor, and expert development (E39 / S01–S07)**

| Output directory | Retained direct report files indexed |
|---|---:|
| [20260610_0340_enclosure_finetune1_interim](../diagnostics_output/20260610_0340_enclosure_finetune1_interim) | 2 |
| [20260610_0346_enclosure_finetune1](../diagnostics_output/20260610_0346_enclosure_finetune1) | 2 |
| [20260610_0354_enclosure_finetune2_frontframe](../diagnostics_output/20260610_0354_enclosure_finetune2_frontframe) | 2 |
| [20260610_0424_enclosure_fumehood1](../diagnostics_output/20260610_0424_enclosure_fumehood1) | 2 |
| [20260610_0457_enclosure_panel1](../diagnostics_output/20260610_0457_enclosure_panel1) | 2 |
| [20260610_0536_enclosure_cubby1](../diagnostics_output/20260610_0536_enclosure_cubby1) | 2 |
| [20260610_0600_noglass_proof](../diagnostics_output/20260610_0600_noglass_proof) | 0 |
| [20260610_0737_enclosure_fumehood2](../diagnostics_output/20260610_0737_enclosure_fumehood2) | 2 |
| [20260610_0805_enclosure_panel2](../diagnostics_output/20260610_0805_enclosure_panel2) | 2 |
| [20260610_0854_enclosure_cubby2](../diagnostics_output/20260610_0854_enclosure_cubby2) | 2 |
| [20260610_0925_enclosure_fumehood3](../diagnostics_output/20260610_0925_enclosure_fumehood3) | 2 |
| [20260610_0957_enclosure_panel3](../diagnostics_output/20260610_0957_enclosure_panel3) | 2 |
| [20260610_1040_enclosure_cubby3](../diagnostics_output/20260610_1040_enclosure_cubby3) | 2 |
| [20260610_foxglove_dashboard](../diagnostics_output/20260610_foxglove_dashboard) | 0 |
| [20260610_hybrid_skin_viz](../diagnostics_output/20260610_hybrid_skin_viz) | 0 |
| [20260610_plots_v3](../diagnostics_output/20260610_plots_v3) | 0 |
| [20260611_exo_check](../diagnostics_output/20260611_exo_check) | 0 |
| [20260611_fumehood_tour](../diagnostics_output/20260611_fumehood_tour) | 0 |
| [20260611_fumehood_variations](../diagnostics_output/20260611_fumehood_variations) | 0 |
| [20260611_hybrid_fumehood_reconstruct](../diagnostics_output/20260611_hybrid_fumehood_reconstruct) | 1 |
| [20260611_hybrid_overnight](../diagnostics_output/20260611_hybrid_overnight) | 0 |
| [20260611_hybrid_sensor_verify](../diagnostics_output/20260611_hybrid_sensor_verify) | 0 |
| [20260611_hybrid_skin_REPORT](../diagnostics_output/20260611_hybrid_skin_REPORT) | 0 |
| [20260611_hybrid_skin_rich](../diagnostics_output/20260611_hybrid_skin_rich) | 0 |
| [20260611_hybrid_test_reconstruct](../diagnostics_output/20260611_hybrid_test_reconstruct) | 1 |
| [20260611_hybrid_viz_suite](../diagnostics_output/20260611_hybrid_viz_suite) | 0 |
| [20260611_pick_place_mug_bowl](../diagnostics_output/20260611_pick_place_mug_bowl) | 0 |
| [20260611_pnp5_debug](../diagnostics_output/20260611_pnp5_debug) | 0 |
| [20260611_skin_photoshoot](../diagnostics_output/20260611_skin_photoshoot) | 0 |

**Hybrid residual-controller program (Y01–Y22)**

| Output directory | Retained direct report files indexed |
|---|---:|
| [causal_parked_skin_reference_v1](../diagnostics_output/causal_parked_skin_reference_v1) | 3 |
| [hybrid_clean_retrain](../diagnostics_output/hybrid_clean_retrain) | 3 |
| [hybrid_obstacle_act_baseline](../diagnostics_output/hybrid_obstacle_act_baseline) | 5 |
| [hybrid_obstacle_activity_identifiability](../diagnostics_output/hybrid_obstacle_activity_identifiability) | 5 |
| [hybrid_obstacle_dataset](../diagnostics_output/hybrid_obstacle_dataset) | 4 |
| [hybrid_obstacle_deployable_reference](../diagnostics_output/hybrid_obstacle_deployable_reference) | 4 |
| [hybrid_obstacle_full_collection](../diagnostics_output/hybrid_obstacle_full_collection) | 5 |
| [hybrid_obstacle_full_seed_joint_gate](../diagnostics_output/hybrid_obstacle_full_seed_joint_gate) | 3 |
| [hybrid_obstacle_observation_reference](../diagnostics_output/hybrid_obstacle_observation_reference) | 1 |
| [hybrid_obstacle_on_policy_reference](../diagnostics_output/hybrid_obstacle_on_policy_reference) | 3 |
| [hybrid_obstacle_oracle_reference](../diagnostics_output/hybrid_obstacle_oracle_reference) | 5 |
| [hybrid_obstacle_paired_smoke](../diagnostics_output/hybrid_obstacle_paired_smoke) | 3 |
| [hybrid_obstacle_parked_skin_dataset](../diagnostics_output/hybrid_obstacle_parked_skin_dataset) | 3 |
| [hybrid_obstacle_parked_skin_reference](../diagnostics_output/hybrid_obstacle_parked_skin_reference) | 3 |
| [hybrid_obstacle_prox_activity_gate](../diagnostics_output/hybrid_obstacle_prox_activity_gate) | 2 |
| [hybrid_obstacle_raw_head_qualification](../diagnostics_output/hybrid_obstacle_raw_head_qualification) | 3 |
| [hybrid_obstacle_reference_threshold](../diagnostics_output/hybrid_obstacle_reference_threshold) | 4 |
| [hybrid_obstacle_seeding](../diagnostics_output/hybrid_obstacle_seeding) | 2 |
| [hybrid_obstacle_three_pair_joint_gate](../diagnostics_output/hybrid_obstacle_three_pair_joint_gate) | 3 |
| [hybrid_obstacle_three_pair_live](../diagnostics_output/hybrid_obstacle_three_pair_live) | 3 |
| [hybrid_obstacle_uncertainty_abstention](../diagnostics_output/hybrid_obstacle_uncertainty_abstention) | 3 |
| [hybrid_safety_stack](../diagnostics_output/hybrid_safety_stack) | 4 |

**Pendant qualification and routing (E20–E30)**

| Output directory | Retained direct report files indexed |
|---|---:|
| [pact_place_v101_empirical_causal](../diagnostics_output/pact_place_v101_empirical_causal) | 0 |
| [pact_place_v101_empirical_review](../diagnostics_output/pact_place_v101_empirical_review) | 3 |
| [pact_place_v102_diagnostic_gallery](../diagnostics_output/pact_place_v102_diagnostic_gallery) | 0 |
| [pact_place_v102_preflight](../diagnostics_output/pact_place_v102_preflight) | 1 |
| [pact_place_v103_ik_search](../diagnostics_output/pact_place_v103_ik_search) | 1 |
| [pact_place_v104_causal](../diagnostics_output/pact_place_v104_causal) | 0 |
| [pact_place_v104_preflight](../diagnostics_output/pact_place_v104_preflight) | 4 |
| [pact_place_v104_review](../diagnostics_output/pact_place_v104_review) | 1 |
| [pact_place_v104_review_production](../diagnostics_output/pact_place_v104_review_production) | 0 |
| [pact_place_v104_review_v2](../diagnostics_output/pact_place_v104_review_v2) | 4 |
| [pact_place_v104_review_v2_superseded_01_incomplete_approval_schema](../diagnostics_output/pact_place_v104_review_v2_superseded_01_incomplete_approval_schema) | 4 |
| [pact_place_v105_audit](../diagnostics_output/pact_place_v105_audit) | 1 |
| [pact_place_v105_audit_attempt_01_missing_trajectory_misclassified](../diagnostics_output/pact_place_v105_audit_attempt_01_missing_trajectory_misclassified) | 1 |
| [pact_place_v105_reconstruction](../diagnostics_output/pact_place_v105_reconstruction) | 1 |
| [pact_place_v105_reconstruction_attempt_01_instrument_failure](../diagnostics_output/pact_place_v105_reconstruction_attempt_01_instrument_failure) | 1 |
| [pact_place_v105_reconstruction_attempt_02_instrument_failure](../diagnostics_output/pact_place_v105_reconstruction_attempt_02_instrument_failure) | 1 |
| [pact_place_v105_siting](../diagnostics_output/pact_place_v105_siting) | 1 |
| [pact_place_v105_siting_attempt_01_direction_predicate_defect](../diagnostics_output/pact_place_v105_siting_attempt_01_direction_predicate_defect) | 1 |
| [pact_place_v105_siting_pilot_attempt_01_direction_predicate_defect](../diagnostics_output/pact_place_v105_siting_pilot_attempt_01_direction_predicate_defect) | 1 |
| [pact_place_v106_causal](../diagnostics_output/pact_place_v106_causal) | 0 |
| [pact_place_v106_causal_attempt_01_probe_defects](../diagnostics_output/pact_place_v106_causal_attempt_01_probe_defects) | 0 |
| [pact_place_v106_causal_attempt_02_grasp_counted_as_collision](../diagnostics_output/pact_place_v106_causal_attempt_02_grasp_counted_as_collision) | 0 |
| [pact_place_v106_certification](../diagnostics_output/pact_place_v106_certification) | 0 |
| [pact_place_v106_siting](../diagnostics_output/pact_place_v106_siting) | 1 |
| [pact_place_v107_causal](../diagnostics_output/pact_place_v107_causal) | 0 |
| [pact_place_v107_causal_01_superseded](../diagnostics_output/pact_place_v107_causal_01_superseded) | 0 |
| [pact_place_v107_causal_02_superseded](../diagnostics_output/pact_place_v107_causal_02_superseded) | 0 |
| [pact_place_v107_causal_04_superseded](../diagnostics_output/pact_place_v107_causal_04_superseded) | 0 |
| [pact_place_v107_certification](../diagnostics_output/pact_place_v107_certification) | 0 |
| [pact_place_v107_certification_01_superseded](../diagnostics_output/pact_place_v107_certification_01_superseded) | 0 |
| [pact_place_v107_certification_02_superseded](../diagnostics_output/pact_place_v107_certification_02_superseded) | 0 |
| [pact_place_v107_certification_03_halted_on_drift](../diagnostics_output/pact_place_v107_certification_03_halted_on_drift) | 0 |
| [pact_place_v107_certification_04_superseded](../diagnostics_output/pact_place_v107_certification_04_superseded) | 0 |
| [pact_place_v107_contact_diagnostic](../diagnostics_output/pact_place_v107_contact_diagnostic) | 0 |
| [pact_place_v107_owner_review](../diagnostics_output/pact_place_v107_owner_review) | 2 |
| [pact_place_v107_phase0](../diagnostics_output/pact_place_v107_phase0) | 0 |
| [pact_place_v107_pool](../diagnostics_output/pact_place_v107_pool) | 0 |
| [pact_place_v107_pool_attempt_01_scene_guard_defect](../diagnostics_output/pact_place_v107_pool_attempt_01_scene_guard_defect) | 0 |
| [pact_place_v107_pool_attempt_02_telemetry_not_passed_through](../diagnostics_output/pact_place_v107_pool_attempt_02_telemetry_not_passed_through) | 0 |
| [pact_place_v107_pool_attempt_03_policy_lacked_manifest_row](../diagnostics_output/pact_place_v107_pool_attempt_03_policy_lacked_manifest_row) | 0 |
| [pact_place_v107_selection](../diagnostics_output/pact_place_v107_selection) | 0 |
| [pact_place_v107_selection_01_superseded](../diagnostics_output/pact_place_v107_selection_01_superseded) | 0 |
| [pact_place_v107_selection_02_superseded](../diagnostics_output/pact_place_v107_selection_02_superseded) | 0 |
| [pact_place_v107_selection_03_halted_on_drift](../diagnostics_output/pact_place_v107_selection_03_halted_on_drift) | 0 |
| [pact_place_v107_selection_04_superseded](../diagnostics_output/pact_place_v107_selection_04_superseded) | 0 |
| [pact_place_v107_specification](../diagnostics_output/pact_place_v107_specification) | 0 |
| [pact_place_v107_specification_01_superseded](../diagnostics_output/pact_place_v107_specification_01_superseded) | 0 |
| [pact_place_v107_specification_02_superseded](../diagnostics_output/pact_place_v107_specification_02_superseded) | 0 |
| [pact_place_v107_specification_03_halted_on_drift](../diagnostics_output/pact_place_v107_specification_03_halted_on_drift) | 0 |
| [pact_place_v107_specification_04_superseded](../diagnostics_output/pact_place_v107_specification_04_superseded) | 0 |
| [pact_place_v10_route](../diagnostics_output/pact_place_v10_route) | 0 |
| [pact_place_v10_route_v2](../diagnostics_output/pact_place_v10_route_v2) | 0 |
| [pact_place_v10_siting](../diagnostics_output/pact_place_v10_siting) | 1 |
| [pact_place_v10_siting_v2](../diagnostics_output/pact_place_v10_siting_v2) | 1 |
| [pact_place_v98_control_parked](../diagnostics_output/pact_place_v98_control_parked) | 1 |
| [pact_place_v98_control_present](../diagnostics_output/pact_place_v98_control_present) | 1 |
| [pact_place_v98_expert_gate](../diagnostics_output/pact_place_v98_expert_gate) | 2 |
| [pact_place_v98_expert_gate_v2](../diagnostics_output/pact_place_v98_expert_gate_v2) | 2 |
| [pact_place_v98_expert_smoke](../diagnostics_output/pact_place_v98_expert_smoke) | 0 |
| [pact_place_v98_expert_smoke2](../diagnostics_output/pact_place_v98_expert_smoke2) | 0 |
| [pact_place_v98_offset_contact_diagnosis](../diagnostics_output/pact_place_v98_offset_contact_diagnosis) | 1 |
| [pact_place_v98_paired_halfy012](../diagnostics_output/pact_place_v98_paired_halfy012) | 0 |
| [pact_place_v98_paired_halfy014](../diagnostics_output/pact_place_v98_paired_halfy014) | 0 |
| [pact_place_v98_paired_halfy016](../diagnostics_output/pact_place_v98_paired_halfy016) | 0 |
| [pact_place_v98_paired_offset_cons](../diagnostics_output/pact_place_v98_paired_offset_cons) | 0 |
| [pact_place_v98_paired_offset_wide](../diagnostics_output/pact_place_v98_paired_offset_wide) | 0 |
| [pact_place_v98_pendant_causal_smoke](../diagnostics_output/pact_place_v98_pendant_causal_smoke) | 1 |
| [pact_place_v98_pendant_causal_smoke2](../diagnostics_output/pact_place_v98_pendant_causal_smoke2) | 1 |
| [pact_place_v98_pendant_causal_smoke3](../diagnostics_output/pact_place_v98_pendant_causal_smoke3) | 1 |
| [pact_place_v98_pendant_preview](../diagnostics_output/pact_place_v98_pendant_preview) | 0 |
| [pact_place_v98_pendant_review](../diagnostics_output/pact_place_v98_pendant_review) | 1 |
| [pact_place_v98_pendant_review_v2](../diagnostics_output/pact_place_v98_pendant_review_v2) | 1 |
| [pact_place_v98_pendant_siting](../diagnostics_output/pact_place_v98_pendant_siting) | 1 |
| [pact_place_v98_pendant_siting_v2](../diagnostics_output/pact_place_v98_pendant_siting_v2) | 1 |
| [pact_place_v98_v95_pendant_paired](../diagnostics_output/pact_place_v98_v95_pendant_paired) | 0 |
| [pact_place_v98_v95_pendant_paired_nobow](../diagnostics_output/pact_place_v98_v95_pendant_paired_nobow) | 0 |
| [pact_place_v99_baseline_reconstruction](../diagnostics_output/pact_place_v99_baseline_reconstruction) | 1 |
| [pact_place_v99_siting](../diagnostics_output/pact_place_v99_siting) | 1 |

**Pickup/extraction policy program (P00–P09)**

| Output directory | Retained direct report files indexed |
|---|---:|
| [pact_blind_rgb](../diagnostics_output/pact_blind_rgb) | 5 |
| [pact_blur_sweep](../diagnostics_output/pact_blur_sweep) | 5 |
| [pact_contact_endpoint](../diagnostics_output/pact_contact_endpoint) | 2 |
| [pact_frontend_screen](../diagnostics_output/pact_frontend_screen) | 2 |
| [pact_geometry_generalization](../diagnostics_output/pact_geometry_generalization) | 4 |
| [pact_geometry_generalization_v2](../diagnostics_output/pact_geometry_generalization_v2) | 1 |
| [pact_geometry_generalization_v3](../diagnostics_output/pact_geometry_generalization_v3) | 4 |
| [pact_seed_replication](../diagnostics_output/pact_seed_replication) | 2 |
| [pact_valid_ablation](../diagnostics_output/pact_valid_ablation) | 3 |
| [pact_vs_act](../diagnostics_output/pact_vs_act) | 5 |
| [pact_vs_act_r2](../diagnostics_output/pact_vs_act_r2) | 0 |

**Sensor, data, and cross-experiment audits**

| Output directory | Retained direct report files indexed |
|---|---:|
| [manipulation_eval_inventory_20260906](../diagnostics_output/manipulation_eval_inventory_20260906) | 1 |
| [medium_v1_partial](../diagnostics_output/medium_v1_partial) | 1 |
| [obstacle_analysis](../diagnostics_output/obstacle_analysis) | 1 |
| [pact_house11_traj0](../diagnostics_output/pact_house11_traj0) | 2 |
| [pact_paper_readiness_audit](../diagnostics_output/pact_paper_readiness_audit) | 1 |
| [pilot_skin_smoke_v1](../diagnostics_output/pilot_skin_smoke_v1) | 1 |
| [prox_necessity_pact](../diagnostics_output/prox_necessity_pact) | 1 |
| [prox_necessity_pilot](../diagnostics_output/prox_necessity_pilot) | 1 |
| [proximity_audit_medium_full](../diagnostics_output/proximity_audit_medium_full) | 2 |
| [proximity_audit_v1](../diagnostics_output/proximity_audit_v1) | 2 |

**V10.10 placement, training interventions, and analyses (L06, L09–L13)**

| Output directory | Retained direct report files indexed |
|---|---:|
| [pact_place_v1010_collection](../diagnostics_output/pact_place_v1010_collection) | 1 |
| [pact_place_v1010_eval](../diagnostics_output/pact_place_v1010_eval) | 1 |
| [pact_place_v1010_eval_infra_repair_01](../diagnostics_output/pact_place_v1010_eval_infra_repair_01) | 1 |
| [pact_place_v1010_paper_views](../diagnostics_output/pact_place_v1010_paper_views) | 0 |
| [pact_place_v1010_pickup_audit_20260906](../diagnostics_output/pact_place_v1010_pickup_audit_20260906) | 2 |
| [pact_place_v1010_preflight](../diagnostics_output/pact_place_v1010_preflight) | 1 |
| [pact_place_v1010_storage](../diagnostics_output/pact_place_v1010_storage) | 1 |
| [pact_place_v1010_tablecam_validation10](../diagnostics_output/pact_place_v1010_tablecam_validation10) | 1 |
| [pact_place_v1010_train_eval](../diagnostics_output/pact_place_v1010_train_eval) | 3 |
| [pact_place_v1010_wrist288_s3_v1](../diagnostics_output/pact_place_v1010_wrist288_s3_v1) | 5 |
| [pact_place_v1010_wrist288_s3_v1_unobserved_exits_20260907](../diagnostics_output/pact_place_v1010_wrist288_s3_v1_unobserved_exits_20260907) | 0 |
| [pact_place_v1010b_grasp_audit](../diagnostics_output/pact_place_v1010b_grasp_audit) | 7 |
| [pact_place_v1010b_grasp_v1](../diagnostics_output/pact_place_v1010b_grasp_v1) | 13 |
| [pact_place_v1010c_completion_time_20260916](../diagnostics_output/pact_place_v1010c_completion_time_20260916) | 2 |
| [pact_place_v1010c_readout_s3103](../diagnostics_output/pact_place_v1010c_readout_s3103) | 3 |
| [pact_place_v1010c_readout_s3_v1](../diagnostics_output/pact_place_v1010c_readout_s3_v1) | 2 |

**V10.11 environments, placement, and diagnostics (L07–L08 / E33–E35)**

| Output directory | Retained direct report files indexed |
|---|---:|
| [pact_place_v1011_contract](../diagnostics_output/pact_place_v1011_contract) | 0 |
| [pact_place_v1011_preflight](../diagnostics_output/pact_place_v1011_preflight) | 1 |
| [pact_place_v1011_preflight_superseded_01_box_radius_underbounded](../diagnostics_output/pact_place_v1011_preflight_superseded_01_box_radius_underbounded) | 1 |
| [pact_place_v1011_review](../diagnostics_output/pact_place_v1011_review) | 2 |
| [pact_place_v1011_review_superseded_01_box_radius_underbounded](../diagnostics_output/pact_place_v1011_review_superseded_01_box_radius_underbounded) | 2 |
| [pact_place_v1011_storage](../diagnostics_output/pact_place_v1011_storage) | 0 |
| [pact_place_v1011b_contract](../diagnostics_output/pact_place_v1011b_contract) | 0 |
| [pact_place_v1011b_pipeline](../diagnostics_output/pact_place_v1011b_pipeline) | 0 |
| [pact_place_v1011b_preflight](../diagnostics_output/pact_place_v1011b_preflight) | 1 |
| [pact_place_v1011b_review](../diagnostics_output/pact_place_v1011b_review) | 2 |
| [pact_place_v1011b_visibility](../diagnostics_output/pact_place_v1011b_visibility) | 0 |
| [pact_place_v1011c_collection_100](../diagnostics_output/pact_place_v1011c_collection_100) | 0 |
| [pact_place_v1011c_collection_100_smoke](../diagnostics_output/pact_place_v1011c_collection_100_smoke) | 0 |
| [pact_place_v1011c_collection_100_smoke2](../diagnostics_output/pact_place_v1011c_collection_100_smoke2) | 0 |
| [pact_place_v1011c_contract](../diagnostics_output/pact_place_v1011c_contract) | 0 |
| [pact_place_v1011c_dualcam_aligned_s3103](../diagnostics_output/pact_place_v1011c_dualcam_aligned_s3103) | 4 |
| [pact_place_v1011c_eval](../diagnostics_output/pact_place_v1011c_eval) | 2 |
| [pact_place_v1011c_pipeline](../diagnostics_output/pact_place_v1011c_pipeline) | 0 |
| [pact_place_v1011c_post_eval_audit](../diagnostics_output/pact_place_v1011c_post_eval_audit) | 6 |
| [pact_place_v1011c_preflight](../diagnostics_output/pact_place_v1011c_preflight) | 1 |
| [pact_place_v1011c_review](../diagnostics_output/pact_place_v1011c_review) | 2 |
| [pact_place_v1011c_train_eval](../diagnostics_output/pact_place_v1011c_train_eval) | 3 |
| [pact_place_v1011c_visibility](../diagnostics_output/pact_place_v1011c_visibility) | 0 |
| [pact_place_v1011d_preflight](../diagnostics_output/pact_place_v1011d_preflight) | 1 |
| [pact_place_v1011d_review](../diagnostics_output/pact_place_v1011d_review) | 2 |

**V10.8/V10.9 collection, placement, and decoder (L04–L05 / E31)**

| Output directory | Retained direct report files indexed |
|---|---:|
| [pact_place_v108_collection](../diagnostics_output/pact_place_v108_collection) | 1 |
| [pact_place_v109_eval](../diagnostics_output/pact_place_v109_eval) | 3 |
| [pact_place_v109_train_eval](../diagnostics_output/pact_place_v109_train_eval) | 5 |
| [pact_place_v109r_diagnostic](../diagnostics_output/pact_place_v109r_diagnostic) | 2 |

### D.5 Other local artifact roots

These paths are machine-local. Their presence does not imply all contained checkpoints/trajectories were audited, nor that the folders contain independent runs. The main text links the specific external reports used.

- [pact_blind_rgb_artifacts](/root/pact_blind_rgb_artifacts)
- [pact_blur_sweep_artifacts](/root/pact_blur_sweep_artifacts)
- [pact_contact_endpoint_artifacts](/root/pact_contact_endpoint_artifacts)
- [pact_frontend_screen_artifacts](/root/pact_frontend_screen_artifacts)
- [pact_geometry_generalization_v2_artifacts](/root/pact_geometry_generalization_v2_artifacts)
- [pact_geometry_generalization_v3_artifacts](/root/pact_geometry_generalization_v3_artifacts)
- [pact_place_152_pact_vs_act_chunk100_seed3101](/root/pact_place_152_pact_vs_act_chunk100_seed3101)
- [pact_place_152_pact_vs_act_chunk25_seed3101](/root/pact_place_152_pact_vs_act_chunk25_seed3101)
- [pact_place_152_pact_vs_act_seed3101](/root/pact_place_152_pact_vs_act_seed3101)
- [pact_place_chunk100_eval_seed3101](/root/pact_place_chunk100_eval_seed3101)
- [pact_place_chunk1_eval_seed3101](/root/pact_place_chunk1_eval_seed3101)
- [pact_place_chunk25_eval_seed3101](/root/pact_place_chunk25_eval_seed3101)
- [pact_place_v1010_144_pact_vs_act_chunk100_seed3101](/root/pact_place_v1010_144_pact_vs_act_chunk100_seed3101)
- [pact_place_v108_141_pact_vs_act_chunk100_seed3101](/root/pact_place_v108_141_pact_vs_act_chunk100_seed3101)
- [pact_remediation_artifacts_v2](/root/pact_remediation_artifacts_v2)
- [pact_seed_replication_artifacts](/root/pact_seed_replication_artifacts)
- [pact_slideshow_bundle](/root/pact_slideshow_bundle)
- [pact_slideshow_bundle_pre_act_success](/root/pact_slideshow_bundle_pre_act_success)
- [pact_slideshow_bundle_pre_clips_v2](/root/pact_slideshow_bundle_pre_clips_v2)
- [pact_slideshow_bundle_pre_seed3102](/root/pact_slideshow_bundle_pre_seed3102)
- [pact_valid_ablation_artifacts](/root/pact_valid_ablation_artifacts)
- [prox_learning](/root/prox_learning)
- [prox_learning_act_retrain](/root/prox_learning_act_retrain)
- [prox_learning_hybrid_safety](/root/prox_learning_hybrid_safety)
- [prox_learning_pact_remediation](/root/prox_learning_pact_remediation)
- [proximity](/root/proximity)


## Appendix E. Glossary

| Term | Meaning in this inventory |
|---|---|
| Completed | The stated bounded measurement/evaluation finished; does not imply a positive result |
| Partial / stopped | Some work or attempts exist, but the planned scientific comparison did not finish |
| Invalid / superseded | A technical or provenance issue prevents use as the intended evidence; a later record may correct it |
| Report-level | Numerical result retained in a report, without a fresh raw-trajectory reconstruction in this audit |
| Remote-only | Evidence retrieved from the public repository; local raw/checkpoint availability not established |
| Reanalysis | New measurement of existing trajectories, not new independently executed trials |
| Unrelated-frame control | Live input replaced by real frames from unrelated data; not isolated sensor-ID permutation |
| Strict success | Successful task completion plus experiment-defined forbidden-contact exclusion |
| Physics sample | Contact-audit sample; not an independent Bernoulli trial |
| Zero-shot | No model updating during evaluation; unseen scene instances alone do not establish distribution shift |

## Appendix F. Proposed or externally reported work without recovered results

These entries must **not** be turned into completed-result tables on the basis of plans, scripts, or conversation alone.

| Study | Status at this inventory cutoff |
|---|---|
| Current 128-D source / clutter-free / held-out-geometry study | [Plan exists](PACT_V1010C_ZERO_SHOT_GENERALIZATION_PLAN_20260916.md); new configuration/evaluation scripts are present, but no completed result report was recovered |
| 40→20 sensor-count reduction (or 15 sensors) | Discussed/planned; no verified completed count-sweep results recovered |
| Current 128-D visible versus camera-hidden plus sensor interventions | User reports colleague completed it; actual current-model counts/artifacts unavailable here. O02 is an older study and cannot substitute for it |
| Matched spatial/history/pretraining ablations for current 128-D | User believes completed by colleague; current-model result files not recovered. Older Y15 history comparison is a different model/objective |
| Held-out geometry plus a second policy such as Diffusion Policy | User previously said not completed; no such result found |
| Training-only proximity, demonstration efficiency, layout transfer, matched RGB-D comparisons | Research/protocol proposals in the ICLR strategy documents; no execution inferred |
| Small range-bias, motion-quality, and regional/stage analyses | [Supporting-study proposal](ICLR2027_SMALL_SUPPORTING_STUDIES_20260916.md); L12 timing is already completed, but does not complete all those proposals |
| Hybrid confirmatory41 | Development/calibration records repeatedly state not executed |
| Hardware evaluation | No physical-skin evaluation recovered; user reports hardware is not ready |

This appendix distinguishes missing evidence from a proven absence of work. Add collaborator result files, checkpoint identifiers, and evaluator metadata before changing those entries to completed.

[paper]: https://github.com/Jdvakil/prox_learning/blob/a5de7a4dd42a075d99b8f5865ec7baa9479a1fdd/PAPER.md
[readme]: https://github.com/Jdvakil/prox_learning/blob/a5de7a4dd42a075d99b8f5865ec7baa9479a1fdd/README.md
[remote-evals]: https://github.com/Jdvakil/prox_learning/tree/a5de7a4dd42a075d99b8f5865ec7baa9479a1fdd/reports/eval_summaries
[aug-report]: https://github.com/Jdvakil/prox_learning/blob/a5de7a4dd42a075d99b8f5865ec7baa9479a1fdd/reports/2026-08-14/report.md
