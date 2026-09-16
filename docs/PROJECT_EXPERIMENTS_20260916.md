# Proximity Learning / PC-ACT: Complete Recoverable Experiment Record

**Inventory date: 16 September 2026.** This document follows the numbered narrative, results, evidence, limitations, and appendix structure of the project's [PAPER.md][paper]. It records completed experiments, negative results, partial evaluations, environment qualification, and analyses of existing rollouts. Proposed experiments are listed separately at the end.

**Scope:** the available project records, including this worktree, other local project worktrees and artifact roots, and the public `main` branch at commit `a5de7a4dd42a075d99b8f5865ec7baa9479a1fdd`. This is a project inventory, not an attribution of every run to one researcher. Some historical experiments were performed by collaborators. It is not possible to recover undocumented experiments or missing collaborator artifacts from these sources.

**Evidence standard:** saved evaluation JSONs and final audited reports take precedence over older progress notes. Remote-only results are identified explicitly. An archived summary can establish what was reported without establishing that its demonstrations, checkpoint, or raw trajectories are still available. No training or simulation was run to prepare this document.

**Reading guide:** §7 contains policy experiments and their results; §4 contains sensor/representation measurements; Appendix A contains the environment-development experiments; Appendix B records configurations and naming; Appendix C explains statistics; Appendix D indexes evidence and existing figures; Appendix E defines terms; Appendix F lists work without recovered results.

**Branch covered explicitly:** `experiment/pact-valid-ablation-followup-v1`, including the scientific record through `e0d6a40`. The branch's ablations are completed experiments, not just proposed controls. The sections below now give their hypotheses, interventions, sample sizes, results, uncertainty, and limitations rather than relying on a directory listing.

<a id="branch-ablations"></a>
### Where to find the ablation experiments from this branch

| Experiment / intervention | What was actually compared | Where the full results appear |
|---|---|---|
| Original 3-D proximity removal | ACT, learned 3-D PACT, and the same PACT with zero tokens; 960 rollouts | [P01](#p01-results) |
| Frozen 32-D front-end and zero-input screen | ACT, 32-D PACT, and zeroed PACT; 120 rollouts, followed by a training-support audit | [P02 and S13](#p02-results) |
| `PACT_PERMUTED`, first valid-input follow-up | Same seed-3101 checkpoint, live versus unrelated real proximity frames; 40 matched pairs | [P03](#p03-results) |
| `PACT_PERMUTED`, independent training seed | Three arms on the same 40 instances at seeds 3101 and 3102; 240 records including reused seed-3101 results | [P04](#p04-results) |
| `PACT_PERMUTED` and `PACT_ZERO`, larger contact study | Four arms, three seeds, 100 shared instances; 1,200 rollouts | [P05](#p05-results) |
| Geometry × live/unrelated proximity | Fixed 32-D PACT under C0, C2, and Z_093; 720 rollouts | [P07](#p07-results) |
| Blur × live/unrelated proximity | ACT, live PACT, and `PACT_PERMUTED` at four inference-time blur levels; 900 rollouts | [P08](#p08-results) |
| RGB removal × live/unrelated proximity | Sighted and constant-image versions of three arms; 450 rollouts | [P09](#p09-results) |
| Placement action-chunk size and `PACT_PERMUTED` | Chunks 1, 25, and 100; the chunk-100 test includes the same-checkpoint unrelated-frame control | [L01–L03](#placement-chunk-ablations) |
| Frozen 32-D versus finetuned 128-D pathway | Complete-method comparison with ACT, three seeds, 450 rollouts | [L09/L11](#readout-method-comparison); several pathway changes, not unfreezing alone |
| Equal-update training/sampling control | Original, uniform continuation, and acquisition-window continuation for both ACT and PACT; 216 Stage-B comparisons | [L10](#training-sampler-ablation) |
| Temporal-history and learned proximity-conditioning ablations | `CURRENT_FRAME_ONLY`, `FULL_CAUSAL`, `QPOS_ONLY`, zero-differential control, and privileged reference; nine trained models | [Y15](#parked-reference-ablations); separate Safety-CVAE reference model |
| Proximity/state interventions in the activity gate | Clear proximity, shuffled state, and mean state on the same historical false positives | [Y17–Y21](#activity-gate-ablations); offline diagnostics and calibration |

The first valid-input follow-up, independent-seed replication, and larger contact study are present in branch commits [0e6e985](https://github.com/Jdvakil/prox_learning/commit/0e6e98574fcf9b7153dbfe063441fbb8aadc5f10), [dc1be08](https://github.com/Jdvakil/prox_learning/commit/dc1be08c96c6d734a73709ab5aeee8e1daef5a87), and [d110ad8](https://github.com/Jdvakil/prox_learning/commit/d110ad8c59fa7c12597191503b6416777903fee1), respectively. Their `PACT_PERMUTED` results use the **older frozen 32-D encoder**. They must not be relabeled as ablations of the later jointly finetuned 128-D encoder.

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

**What `PACT_PERMUTED` actually does.** The intervention replaces the live proximity representation at every control step with a **complete, previously recorded 40-sensor × 32-D embedding frame** from an unrelated training timestep. RGB, robot state, action decoding, and the policy checkpoint remain the same. The policy still receives 40 proximity tokens in their original sensor order, and all 32 feature coordinates of each token remain intact. It therefore tests dependence on proximity that corresponds to the current scene; it does not test a smaller skin or shuffle sensor identities.

For the initial follow-up, [the token-plan builder](../scripts/build_pact_permuted_token_plan.py), specifically `select_sources()` and `main()`, uses the frozen **199-episode training partition**, containing **31,176 complete frames / 1,247,040 sensor embeddings**. With seed `2026073105`, it chooses 900 distinct source frames per rollout without replacement within that rollout. Adjacent source frames must come from different episodes. Frames may recur across different rollouts because 40 × 900 exceeds the available frame population. The saved float32 tensor has shape `(40, 900, 40, 32)`; the builder verifies shape, finite values, and hashes before use.

In [`PactPermutedInferencePolicy`](../submodules/act/eval_pact_valid_ablation_row.py), `prepare_model()` loads the ordinary PACT weights, while `_surface_positions()` ignores the live raw sensor array and returns the selected frozen frame for that control step. `load_token_plan()` verifies the token-plan and tensor hashes. Recorded flags identify the pathway as consumed, nonzeroed, and unaligned with live geometry. The larger contact study uses [`load_contact_token_plan()`](../submodules/act/eval_pact_contact_endpoint_row.py) with the same intervention semantics and its own larger frozen plan. The placement follow-up has a separate source dataset and plan; its result is recorded under L03.

| Intervention | Preserved | Changed / removed | Appropriate interpretation |
|---|---|---|---|
| Live proximity | Scene-corresponding sensor frames and normal policy inputs | Nothing | Full method |
| `PACT_PERMUTED` | Same checkpoint and token dimensions; real whole-frame embeddings; within-frame sensor order and cross-sensor relationships | Correspondence with current RGB, robot state, and geometry; temporal coherence between supplied frames | Sensitivity to unrelated, temporally incoherent measurements |
| `PACT_ZERO` | Same checkpoint and pathway dimensions | Every proximity feature set to zero | For 32-D PACT, an out-of-distribution input-failure probe |
| Constant RGB / blind | Proximity, robot state, scene, controller, and checkpoints | Wrist image replaced by the ImageNet mean | Complete visual-input removal at inference |
| Test-time blur | Same scene and checkpoints | RGB fidelity at the specified sigma | Robustness to an inference-time visual distribution shift |

**Limits of the word “distribution-matched.”** It describes empirical support of individual proximity frames, not the joint distribution of RGB, state, proximity, and history. A sequence of real but unrelated frames can still be a strong mismatch and an active distractor. A live-versus-permuted difference supports the utility of scene-corresponding measurements under this intervention, but cannot separately identify sensor-ID alignment, history, latency, or a pure capacity effect. Similarly, `PACT_PERMUTED − ACT` mixes architecture/training differences with possible harm from wrong sensor information.

The zero-support audit in P02/S13 is essential: zero is common for old invalid 3-D sensor tokens but never occurs in the new 32-D training embeddings. Even in the 3-D case, per-token support alone does not prove that simultaneously zeroing all 40 sensors is a typical whole observation. The latest frozen-versus-finetuned comparison also changes feature width, readout, and pooling; it cannot isolate the causal effect of unfreezing alone.

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

<a id="p01-results"></a>
#### P01 — Original learned 3-D representation and zero-input ablation, confirmatory R2

**Question:** does the original learned surface-coordinate representation improve collision-free pickup, and does the trained policy depend on its proximity input?

**Design:** 160 held-out physical instances × seeds 3101/3102 × ACT, PACT, and `PACT_ZERO` = **960 completed rollouts**, with 320 per arm. This is pickup/extraction, not full placement. ACT and PACT were trained separately on the matched data; `PACT_ZERO` uses each seed's PACT checkpoint with all 40 tokens zeroed during inference. Checkpoints were selected by validation loss. ACT best epochs were 1,904/1,829; PACT best epochs were 1,968/1,793. The frozen 819,172-parameter encoder had mean/median held-out surface error 3.26/1.88 cm and 51.3% of valid targets within 2 cm.

| Method | Task success | Strict success | Any hazard contact | Other-environment contact | Target contact |
|---|---:|---:|---:|---:|---:|
| ACT | 177/320 (55.3%) | 170/320 (53.1%) | 67/320 (20.9%) | 1/320 | 275/320 |
| 3-D PACT | 169/320 (52.8%) | 159/320 (49.7%) | 69/320 (21.6%) | 2/320 | 280/320 |
| PACT_ZERO | 169/320 (52.8%) | 160/320 (50.0%) | 71/320 (22.2%) | 2/320 | 278/320 |

| Strict-success comparison | Difference | Paired-instance bootstrap 95% CI | Reported two-sided Fisher p |
|---|---:|---:|---:|
| Live PACT − ACT | −3.44 pp | [−8.4, +1.6] pp | 0.4290 |
| Live PACT − PACT_ZERO | −0.31 pp | [−1.6, +0.9] pp | 1.0000 |

Hazard contact-pair entries total **1,029,374 / 1,191,065 / 1,196,616** for ACT/live/zero respectively. These count contact pairs at audit samples, not distinct collision events. The failure taxonomy records target-contact-without-success in **81 / 88 / 85** rollouts. Saved pooled Wilson intervals for strict success are [47.7%, 58.5%], [44.2%, 55.1%], and [44.6%, 55.4%]; because the same 160 instances occur at both seeds, the paired-instance intervals are the more relevant comparison uncertainty.

**Finding:** the complete 3-D method did not improve success or contact, and zeroing its features barely changed the measured outcome. This is a completed negative result, recorded as `PACT_NO_CONFIRMED_BENEFIT`, not a missing experiment. It motivated investigation of a richer representation. It does not prove that any proximity representation is useless; nor does a near-zero contrast establish equivalence beyond the resolution of this design.

**Evidence:** [final decision, checkpoint hashes, and contact taxonomy](PACT_VS_ACT_FINAL_DECISION.md), [R2 recovery and earlier interruption](PACT_CONFIRMATORY_INTERRUPTION_AND_R2_RECOVERY.md). The interrupted first execution is not another completed 960-rollout study.

<a id="p02-results"></a>
#### P02 — Frozen 32-D front-end screen and the invalid zero-input ablation

**Question:** does a richer per-sensor learned embedding produce a policy that can use proximity more effectively than the original 3-D representation?

**Design:** seed 3101; 40 matched instances per arm; **120 rollouts** of ACT, live PACT, and `PACT_ZERO`. PACT uses the frozen 837,700-parameter 32-D encoder, checkpoint hash `6fd2dd037e3236b5b6bf7fce8cb2709ead0cf52adcbbe9cbad1061efc2fe3206`. PACT's selected checkpoint is epoch 1,796, validation loss 0.085737, hash `9db867f5b2cf059f5fad56f2eebd2e0e27024bb511ee0d526ea50692c4cf1457`. Encoder validation gives mean/median surface error 3.20/1.69 cm, 52.9% within 2 cm, and validity precision/recall 99.4%/99.9%. The encoder remains frozen during policy training and evaluation.

| Arm | Task success | Strict success, Wilson 95% CI | Any hazard contact | Target contact | Other-environment contact |
|---|---:|---:|---:|---:|---:|
| ACT | 19/40 (47.5%) | 19/40 (47.5%); [32.9, 62.5]% | 6/40 (15.0%) | 35/40 | 0/40 |
| Live PACT | 29/40 (72.5%) | 29/40 (72.5%); [57.2, 83.9]% | 2/40 (5.0%) | 37/40 | 0/40 |
| PACT_ZERO | 2/40 (5.0%) | 1/40 (2.5%); [0.4, 12.9]% | 21/40 (52.5%) | 27/40 | 0/40 |

The original decision-bearing live-minus-zero strict-success contrast was **+70.0 pp**, paired bootstrap 95% CI **[+55.0, +82.5] pp**, with 28 live-only successes and zero zero-only successes; exact McNemar p = **7.451 × 10⁻⁹**. The live-minus-ACT secondary contrast was **+25.0 pp**, paired CI **[+7.5, +42.5] pp**. The original report gives an unpaired Fisher p = 0.0392; P04's paired reanalysis of these same scenes gives McNemar p = 0.01294. These are different tests on reused data, not two replications.

| Arm | Target contact-pair entries | Hazard contact-pair entries | Failure taxonomy: strict / hazard / target-touch-without-success / failure-after-close |
|---|---:|---:|---|
| ACT | 7,573,898 | 163,549 | 19 / 6 / 15 / 0 |
| Live PACT | 10,622,992 | 135,165 | 29 / 2 / 8 / 1 |
| PACT_ZERO | 637,970 | 215,196 | 1 / 21 / 13 / 5 |

**S13 — Why the zero result was subsequently disqualified as modality evidence.** The audit compared the complete 199-episode training partition in both representations:

| Training-support measurement | Original 3-D tokens | New 32-D embeddings |
|---|---:|---:|
| Sensor tokens examined | 1,247,040 | 1,247,040 |
| Exactly zero vectors | 1,184,764 (95.0%) | 0 (0.0%) |
| Vectors with norm below 0.1 | 1,186,598 | 0 |
| Norm mean / median / minimum | 0.0090 / 0 / 0 | 6.3123 / 6.3182 / 6.0157 |
| Zero's coordinate-wise standardized distance: median / maximum | 0.13σ / 0.22σ | 2.20σ / 5.89σ |
| Coordinates where zero is more than 3σ from the mean | 0/3 | 10/32 |

For the new embedding, an all-zero observation is far outside the observed norm range. The enormous live-minus-zero gap measures the response to an unsupported input, not a calibrated removal of physical information. The audit also found all **255** old/new converted episodes identical in length, action, wrist RGB, qpos, and qvel; **199/56** train/validation assignments and normalization statistics match exactly. This ruled out a different non-proximity training dataset as a confound in reusing ACT.

**Finding:** `FRONTEND_SCREEN_SIGNAL_PRESENT` remains the historical output of the original rule, with an explicit validity amendment. The promising ACT comparison required replication, while the zero-input primary required a replacement instrument. P03 supplies that replacement; it does not inherit the +70-point claim. All 120 records reconciled; 118 preselected payloads were losslessly compacted, with the first and last rows retained unpacked.

**Evidence:** [screen decision and amendment](PACT_FRONTEND_SCREEN_DECISION.md), [data equality and zero-support audit](PACT_ACT_DATA_EQUIVALENCE_AND_ZERO_SUPPORT.md), [machine-readable audit](../diagnostics_output/pact_valid_ablation/data_and_zero_support_audit.json).

<a id="p03-results"></a>
#### P03 — `PACT_PERMUTED`: first scene-correspondence ablation

**Question:** after replacing the invalid zero-input control with real sensor embeddings, does the *same trained policy* perform better with proximity that corresponds to its current environment?

**Design:** the seed-3101 PACT checkpoint and its 40 completed live rollouts from P02 are held fixed. The only new execution is **40 `PACT_PERMUTED` rollouts** on those same instances. At each of 900 control steps, the intervention supplies an unrelated complete training frame, as specified in §5.2. Thus 80 live/control observations enter the paired analysis, but only 40 new rollouts were collected. Camera/state input, normalization, policy weights, sensor order, action interpretation, and task geometry are preserved. There is no separately trained “permuted model.”

The source's original preregistration says 512 token frames. The smoke exposed exhaustion at step 512 in a 900-step rollout **before any terminal scientific result existed**. The documented pre-outcome amendment froze a new 900-frame plan and output root, with the seed, checkpoint, instances, metrics, and thresholds unchanged. The replacement smoke passed on attempt 0. The executed experiment is the amended 900-step version, not the stale 512-frame text.

| Endpoint | Live PACT | PACT_PERMUTED |
|---|---:|---:|
| Ordinary task success | 29/40 (72.5%) | 24/40 (60.0%) |
| Collision-free task success | 29/40 (72.5%) | 24/40 (60.0%) |
| Strict-success Wilson 95% interval | [57.2%, 83.9%] | [44.6%, 73.7%] |
| Any hazard contact | 2/40 (5.0%) | 5/40 (12.5%) |
| Any target contact | 37/40 (92.5%) | 36/40 (90.0%) |
| Other-environment contact | 0/40 | 0/40 |
| Hazard contact-pair entries, total | 135,165 | 137,282 |
| Target contact-pair entries, total | 10,622,992 | 8,994,813 |

The matched strict-success table is:

| Same physical instance | PACT_PERMUTED succeeds | PACT_PERMUTED fails | Total |
|---|---:|---:|---:|
| Live PACT succeeds | 21 | 8 | 29 |
| Live PACT fails | 3 | 8 | 11 |
| Total | 24 | 16 | 40 |

The effect is **+12.5 pp**, paired whole-instance bootstrap 95% CI **[−2.5, +27.5] pp**, using 20,000 replicates. Exact two-sided McNemar p = **0.2265625**, based on the 8 versus 3 discordant pairs. Failure-taxonomy counts for live/control are strict success **29/24**, hazard contact **2/5**, target-touch-without-success **8/10**, and failure-after-close **1/1**.

**Predeclared decision and result:** a positive signal required at least +10 pp **and** a CI lower bound above zero. At least +5 pp without satisfying that stronger rule was a weak signal. The result is **`VALID_ABLATION_WEAK_SIGNAL`**, because the interval crosses zero. The study did not pass its own stronger evidence gate. The small change in total contact-pair entries, despite fewer hazard-contact episodes, also warns against treating incidence and contact duration/burden as interchangeable.

**What this establishes:** a suggestive within-checkpoint advantage for live scene-corresponding proximity, with substantial uncertainty. It does not establish a sensor-ID effect or identify whether spatial mismatch versus temporal incoherence causes the difference. The later larger contact study addresses precision on contact outcomes. All 40 new scientific results and 40 driver records reconciled; 38 payloads were losslessly compacted and rows 0/39 remain unpacked.

**Evidence:** [preregistration](PACT_VALID_ABLATION_PREREGISTRATION.md), [horizon amendment](PACT_VALID_ABLATION_HORIZON_AMENDMENT.md), [decision](PACT_VALID_ABLATION_DECISION.md), [analysis](../diagnostics_output/pact_valid_ablation/analysis.json), [token-plan builder](../scripts/build_pact_permuted_token_plan.py).

<a id="p04-results"></a>
#### P04 — `PACT_PERMUTED`: independent-seed replication of the screen

**Question:** is the seed-3101 advantage over ACT reproducible under a second independently trained PACT checkpoint, and is live proximity still better than unrelated frames?

**Design:** seeds 3101 and 3102, ACT/live PACT/`PACT_PERMUTED`, and the **same 40 physical instances at both seeds**. The 120 seed-3101 records are reused from P02/P03. Seed 3102 adds 120 new rollout records; its ACT checkpoint can be reused because the equality audit confirms the matched training payload. PACT is trained at seed 3102 with the same frozen 32-D encoder and recipe. The same control-frame plan is used across corresponding instances. `PACT_ZERO` is excluded from the evidence-bearing replication.

| Training seed | Arm | Task success | Strict success | Hazard-contact episodes | Target-contact episodes |
|---|---|---:|---:|---:|---:|
| 3101 | ACT | 19/40 (47.5%) | 19/40 (47.5%) | 6/40 | 35/40 |
| 3101 | Live PACT | 29/40 (72.5%) | 29/40 (72.5%) | 2/40 | 37/40 |
| 3101 | PACT_PERMUTED | 24/40 (60.0%) | 24/40 (60.0%) | 5/40 | 36/40 |
| 3102 | ACT | 24/40 (60.0%) | 24/40 (60.0%) | 6/40 | 37/40 |
| 3102 | Live PACT | 22/40 (55.0%) | 21/40 (52.5%) | 4/40 | 38/40 |
| 3102 | PACT_PERMUTED | 15/40 (37.5%) | 15/40 (37.5%) | 6/40 | 33/40 |

No arm has other-environment contact in this replication matrix. Seed-3102 strict-success Wilson intervals are ACT **[44.6%, 73.7%]**, live **[37.5%, 67.1%]**, and permuted **[24.2%, 53.0%]**. Its hazard contact-pair totals are ACT **94,565**, live **58,824**, and permuted **116,597**.

| Seed | Strict-success contrast | Difference | Paired 95% CI | First-only / second-only successes | Exact McNemar p |
|---|---|---:|---:|---:|---:|
| 3101 | Live − permuted | +12.5 pp | [−2.5, +27.5] pp | 8 / 3 | 0.2266 |
| 3101 | Permuted − ACT | +12.5 pp | [−7.5, +32.5] pp | 11 / 6 | 0.3323 |
| 3101 | Live − ACT | +25.0 pp | [+7.5, +42.5] pp | 12 / 2 | 0.01294 |
| 3102 | Live − permuted | +15.0 pp | [+2.5, +27.5] pp | 7 / 1 | 0.07031 |
| 3102 | Permuted − ACT | −22.5 pp | [−40.0, −5.0] pp | 3 / 12 | 0.03516 |
| 3102 | Live − ACT | −7.5 pp | [−25.0, +12.5] pp | 6 / 9 | 0.6072 |

The seed-3102 live-versus-permuted bootstrap interval and exact test differ at the conventional 0.05 threshold; both are retained rather than selecting the more favorable procedure.

| Pooled arm / contrast | Task successes | Strict successes or difference | Whole-instance clustered 95% CI for difference |
|---|---:|---:|---:|
| ACT | 43/80 (53.75%) | 43/80 (53.75%) | — |
| Live PACT | 51/80 (63.75%) | 50/80 (62.50%) | — |
| PACT_PERMUTED | 39/80 (48.75%) | 39/80 (48.75%) | — |
| Live − permuted | — | +13.75 pp | [+3.75, +23.75] pp |
| Permuted − ACT | — | −5.00 pp | [−20.0, +10.0] pp |
| Live − ACT | — | +8.75 pp | [−5.0, +22.5] pp |

Pooled resampling moves both seed outcomes for an instance together: this is **40 distinct environments**, not 80 independent environments. Uncertainty is conditional on these two trained checkpoints.

**Finding:** the live-versus-permuted direction repeats, but the large live-versus-ACT task advantage does not. Seed 3102 reverses the ACT comparison from +25 to −7.5 pp. The predeclared replication decision is **`SEED_REPLICATION_FAILED`**. Summarizing only the positive pooled modality result would hide the failed ACT replication. These results motivated the larger, contact-focused study below.

**Evidence:** [replication decision](PACT_SEED_REPLICATION_DECISION.md), [preregistration](PACT_SEED_REPLICATION_PREREGISTRATION.md), [complete analysis](../diagnostics_output/pact_seed_replication/analysis.json).

<a id="p05-results"></a>
#### P05 — `PACT_PERMUTED` and zero-input ablations in the 1,200-rollout contact study

**Question:** does live, scene-corresponding proximity reduce hazardous contact compared with realistic but unrelated sensor inputs, and does that benefit coexist with task progress?

**Design:** the unchanged `pact_collision_corridor_v1` pickup environment; frozen 32-D encoder; **100 fresh physical instances × three training seeds × four arms = 1,200 rollouts**. Every instance is evaluated under ACT, live PACT, `PACT_PERMUTED`, and `PACT_ZERO` at seeds 3101–3103. The existing 3101/3102 models are retained; seed 3103 adds independently initialized ACT/PACT using the same 2,000-epoch recipe. Permuted and zero arms use their seed's ordinary PACT weights; neither is trained separately. The scientific unit is a matched instance, observed at three trained seeds, not an individual physics sample.

The co-primary endpoints are strict success and the number of physics audit samples containing hazard-bar contact. Strict success requires task success and zero hazard-bar/other-environment contact entries. Intended target contacts are allowed. The decision-bearing information contrast is **live PACT − PACT_PERMUTED on hazard-contact samples**. `PACT_ZERO` remains an out-of-distribution failure diagnostic. Whole-instance bootstrap resampling uses 20,000 replicates, moving every arm and seed for a sampled instance together.

**Results by training seed.** Each row contains 100 rollouts; every arm's median hazard-contact count is zero.

| Seed | Arm | Task success | Strict success | Any hazard contact | Mean hazard-contact samples per rollout |
|---|---|---:|---:|---:|---:|
| 3101 | ACT | 55/100 | 52/100 | 23/100 | 3,885.81 |
| 3101 | Live PACT | 62/100 | 60/100 | 9/100 | 380.53 |
| 3101 | PACT_PERMUTED | 59/100 | 57/100 | 18/100 | 2,936.89 |
| 3101 | PACT_ZERO | 4/100 | 3/100 | 37/100 | 3,970.39 |
| 3102 | ACT | 53/100 | 50/100 | 21/100 | 1,763.39 |
| 3102 | Live PACT | 58/100 | 51/100 | 20/100 | 3,062.77 |
| 3102 | PACT_PERMUTED | 59/100 | 52/100 | 30/100 | 4,729.89 |
| 3102 | PACT_ZERO | 11/100 | 9/100 | 35/100 | 3,989.20 |
| 3103 | ACT | 61/100 | 57/100 | 23/100 | 3,421.13 |
| 3103 | Live PACT | 63/100 | 60/100 | 13/100 | 1,591.43 |
| 3103 | PACT_PERMUTED | 53/100 | 50/100 | 22/100 | 3,307.24 |
| 3103 | PACT_ZERO | 9/100 | 8/100 | 35/100 | 3,778.28 |

| Seed | Live − permuted mean hazard samples, 95% paired-instance CI | Live − ACT strict success, 95% paired-instance CI |
|---|---:|---:|
| 3101 | −2,556.4 [−4,121.9, −1,184.1] | +8.0 pp [−2.0, +18.0] |
| 3102 | −1,667.1 [−2,854.8, −702.7] | +1.0 pp [−10.0, +12.0] |
| 3103 | −1,715.8 [−3,070.0, −552.1] | +3.0 pp [−5.0, +11.0] |

This separates two findings that a pooled headline would obscure. Live PACT has fewer mean hazard-contact samples than its permuted version in **all three seeds**, with each interval below zero. Against ACT, however, live PACT has **more** mean hazard-contact samples at seed 3102, despite marginally fewer contact episodes. Task/strict point estimates favor PACT over ACT at every seed, but their individual strict-success intervals all cross zero.

**Pooled outcomes.** Each arm has 300 rollouts over the same 100 physical instances. Strict-success and mean-contact intervals below are whole-instance clustered bootstrap intervals, not 300-independent-trial intervals.

| Method | Task success | Strict success, clustered 95% CI | Any hazard contact | Mean hazard samples, clustered 95% CI |
|---|---:|---:|---:|---:|
| ACT | 169/300 (56.3%) | 159/300 (53.0%); [45.3, 60.7]% | 67/300 (22.3%) | 3,023.4; [1,854.8, 4,348.9] |
| Live PACT | 183/300 (61.0%) | 171/300 (57.0%); [48.7, 65.0]% | 42/300 (14.0%) | 1,678.2; [954.3, 2,497.5] |
| PACT_PERMUTED | 171/300 (57.0%) | 159/300 (53.0%); [45.0, 61.0]% | 70/300 (23.3%) | 3,658.0; [2,235.8, 5,268.9] |
| PACT_ZERO | 24/300 (8.0%) | 20/300 (6.7%); [4.0, 9.3]% | 107/300 (35.7%) | 3,912.6; [2,586.2, 5,391.9] |

**Primary and important secondary contrasts.** Differences are first arm minus second; negative values favor the first arm for contacts, positive values for success. Intervals are paired and clustered by physical instance.

| Contrast | Endpoint | Difference | 95% CI |
|---|---|---:|---:|
| Live − permuted | Mean hazard-contact samples | **−1,979.8** | **[−3,152.9, −965.2]** |
| Live − permuted | Any hazard-contact rate | **−9.33 pp** | **[−14.33, −5.00] pp** |
| Live − permuted | Strict success | +4.0 pp | [0.0, +8.0] pp |
| Live − permuted | Task success | +4.0 pp | [−0.7, +8.7] pp |
| Live − permuted | Mean hazard contact-pair entries | −3,230.6 | [−5,573, −1,354] |
| Live − permuted | Mean rollout maximum hazard penetration | −0.340 mm | [−0.546, −0.159] mm |
| Live − ACT | Mean hazard-contact samples | **−1,345.2** | **[−2,521, −279]** |
| Live − ACT | Any hazard-contact rate | **−8.33 pp** | **[−13.33, −3.7] pp** |
| Live − ACT | Strict success | +4.0 pp | [−2.3, +10.3] pp |
| Live − ACT | Task success | +4.7 pp | [−2.3, +11.7] pp |
| Live − ACT | Mean hazard contact-pair entries | −1,085.2 | [−2,404, +181.9] |
| Live − ACT | Mean rollout maximum hazard penetration | −0.294 mm | [−0.486, −0.127] mm |
| Permuted − ACT | Mean hazard-contact samples | +634.6 | [−234.1, +1,570] |
| Permuted − ACT | Any hazard-contact rate | +1.0 pp | [−2.7, +4.7] pp |
| Permuted − ACT | Strict success | 0.0 pp | [−6.0, +6.3] pp |
| Permuted − ACT | Task success | +0.7 pp | [−5.7, +7.0] pp |
| Live − zero, OOD diagnostic | Mean hazard-contact samples | −2,234.4 | [−3,397, −1,240] |
| Live − zero, OOD diagnostic | Any hazard-contact rate | −21.7 pp | [−28.3, −15.3] pp |
| Live − zero, OOD diagnostic | Strict success | +50.3 pp | [+42.3, +58.3] pp |

Relative to the permuted control, live proximity has **54.1% fewer mean hazard-contact samples** and **40.0% fewer hazard-contact episodes**. Relative to ACT, the corresponding reductions are **44.5%** and **37.3%**. These are point estimates; the difference intervals above carry the uncertainty. The strict-success interval against permuted reaches zero and the interval against ACT crosses zero. Neither supports describing the task-success improvement as statistically established.

| Additional pooled contact measurement | ACT | Live PACT | PACT_PERMUTED | PACT_ZERO |
|---|---:|---:|---:|---:|
| Mean hazard contact-pair entries per rollout | 3,352.5 | 2,267.3 | 5,498.0 | 4,554.1 |
| Mean of per-rollout maximum hazard penetration | 0.417 mm | 0.123 mm | 0.463 mm | 0.400 mm |
| Mean other-environment contact samples per rollout | 2.01 | 2.53 | 0.40 | 2.51 |

Penetration means include zero-contact rollouts. They are not the penetration depth of a typical collision. Other-environment contacts are rare; the principal measured benefit concerns the hazard bar. The original report also supplies Fisher exact p-values, explicitly labeled as cluster-unaware descriptive tests. They do not replace the paired instance bootstrap.

**Predeclared decision:** the result is `CONTACT_REDUCTION_WITH_TASK_BENEFIT`. Its exact rule requires a pooled live-minus-permuted contact interval strictly below zero, a negative mean contact gap in each seed, and *positive point estimates* of live-minus-ACT strict success pooled and in every seed. The label does **not** require a strict-success CI above zero. The defensible reading is established contact reduction under this intervention plus positive but uncertain task-success differences.

**Design and audit qualifications:**

- The original camera-visibility partition put **285/285 eligible recorded episodes** in the vision-disadvantaged subset. The subset analysis was dropped before rollout outcomes; this study does not contain a selective visible-versus-hidden-hazard comparison.
- The 100-instance allocation used a historical contact-effect power approximation: about 99 instances for the previous contact effect, versus 108 for the previous binary effect. This was a design calculation, not a guarantee of significance.
- The fixed 1,200-row schedule reconciled. Endpoints and hashes were retained for every row, but most trajectory/video payloads were deleted under an outcome-blind storage rule. Rows 0 and 1199 were selected for full retention. The contact payload is summary-only, which constrains the temporal reanalyses in P06.
- This is frozen **32-D** PACT. Neither this study nor the decision token establishes a causal benefit of jointly finetuning a 128-D encoder.

**Evidence:** [preregistration with exact decision rules](../configs/pact_contact_endpoint_preregistration_v1.json), [final decision and full contrast family](PACT_CONTACT_ENDPOINT_DECISION.md), [all seed/arm summaries and contrasts](../diagnostics_output/pact_contact_endpoint/analysis.json), [`load_contact_token_plan()` and evaluator](../submodules/act/eval_pact_contact_endpoint_row.py).

#### P06 — Contact-tail, entry, and target-engagement reanalyses

**Question:** is the P05 contact benefit mainly fewer trajectories entering prolonged collision, or faster recovery once contact starts? Does the benefit correspond to useful target engagement?

**Design:** reanalysis of P05's existing 1,200 records; **no new policy trials**. A diagnostic high-contact tail is defined as more than 500 hazard-contact samples. Tail-conditioned quantities describe selected episodes and must not be presented as unconditional randomized comparisons.

| Analysis | ACT | Live PACT | PACT_PERMUTED |
|---|---:|---:|---:|
| Episodes with more than 500 hazard samples | 59/300 (19.7%) | 33/300 (11.0%) | 58/300 (19.3%) |
| Mean hazard samples within this high-contact tail | About 15,363 | About 15,236 | About 18,894 |
| Hazard contact with no target touch | 34/300 | 3/300 | 30/300 |
| Fewer than 50 target samples within the high-contact tail | 34/59 | 5/33 | 33/58 |
| Tail maximum penetration: median | 0.813 mm | 0.551 mm | 0.893 mm |
| Tail maximum penetration: mean | 1.997 mm | 0.887 mm | 2.255 mm |
| Tail maximum penetration: largest observed | 10.850 mm | 8.955 mm | 10.460 mm |
| Middle 50% of recorded first hazard-contact steps | 47–148 | 142–344 | 54–147 |

Summing the three seed outcomes within each of the 100 physical instances, live PACT has lower/higher/equal hazard samples than ACT on **24/13/63** instances, and lower/higher/equal samples than permuted on **26/4/70**. The mean summed differences are approximately −4,036 and −5,939 samples respectively; both medians are zero. These sparse, heavy-tailed outcomes explain why means and medians tell different parts of the story.

Of the 33 live-PACT high-contact episodes, **31** are also high-contact under the matched permuted arm. All **20** physical instances that produce a live-PACT high-contact episode fall within the **29** such permuted instances. This is consistent with a concentration of residual failures in shared difficult geometry rather than entirely new failure locations.

**Finding:** live proximity reduces entry into the measured high-contact regime. Once in that regime, live PACT's mean burden is close to ACT's, so these summaries do not establish faster escape. All **67** hazard episodes without any target touch across ACT/live/permuted fail the task. Live PACT has far fewer of them. Conversely, restricting to pairs in which both live and permuted succeed yields a mean contact difference of **−0.8 samples**, CI **[−19.8, +22.0]**, with no resolved benefit in that conditioned subset.

**Logging limit:** all 1,200 contact records lack a retained full per-physics-step contact sequence. Contact-pair entries count simultaneous geometry contacts over samples, not contact-onset transitions. The summaries therefore cannot reconstruct precise collision episode duration, escape latency, or impulse. Lower contact totals must not be rewritten as faster collision recovery.

**Evidence:** [tail characterization](PACT_TAIL_CHARACTERIZATION.md), [target-engagement characterization](PACT_ABSORBING_FAILURE_CHARACTERIZATION.md), [both-successful contrast](PACT_CONTACT_ENDPOINT_DECISION.md).

<a id="p07-results"></a>
#### P07 — Geometry × proximity ablation: zero-shot geometry study V3

**Question:** does the contact benefit of scene-corresponding proximity survive changes to corridor geometry without retraining?

**Design:** fixed frozen 32-D PACT checkpoints at seeds 3101–3103, each evaluated with live and unrelated-frame inputs; **40 instances × three seeds × three conditions × two arms = 720 rollouts**. There is no ACT arm. Geometry, rather than token dimensions or model weights, changes between conditions. All arms, seeds, and conditions for each instance move together in the 20,000-replicate bootstrap.

| Condition | Geometry relative to source condition | Purpose / qualification |
|---|---|---|
| C0 | Aperture 0.85 m; panel inner face y=0.10 m; panel z=0.89 m | In-distribution contact-effect reproduction |
| C2 | Aperture reduced to 0.70 m; panel inner face moved to y=0.07 m; z retained | Narrower clearance; 12/12 clean expert successes in the feasibility screen |
| Z_093 | Panel raised to z=0.93 m, otherwise source geometry | Height shift; 11/12 clean expert successes in the later screen |

The expert qualification rule requires at least **10/12** task successes without hazard or other-environment contact, independently of learned-policy performance. The training support fixes aperture width at **0.85 m** and panel z at **0.89 m**, with panel inner-face y in **[0.095, 0.105] m**. C2's 0.70 m aperture / 0.07 m inner face and Z_093's 0.93 m height are therefore outside that support. C2 also fixes face jitter at zero, whereas C0/Z_093 retain ±0.005 m; all retain panel-x jitter ±0.015 m. These details come from the [frozen configuration](../configs/pact_geometry_generalization_v3.json). No policy, encoder, normalization, or checkpoint is adjusted using these outcomes.

| Condition / arm | Task success | Strict success | Any hazard contact | Mean hazard-contact samples |
|---|---:|---:|---:|---:|
| C0 live | 75/120 (62.5%) | 70/120 (58.3%) | 10/120 (8.3%) | 610.4 |
| C0 PACT_PERMUTED | 66/120 (55.0%) | 63/120 (52.5%) | 22/120 (18.3%) | 1,903.7 |
| C2 live | 55/120 (45.8%) | 50/120 (41.7%) | 23/120 (19.2%) | 2,047.4 |
| C2 PACT_PERMUTED | 46/120 (38.3%) | 41/120 (34.2%) | 37/120 (30.8%) | 3,793.7 |
| Z_093 live | 67/120 (55.8%) | 55/120 (45.8%) | 20/120 (16.7%) | 2,049.4 |
| Z_093 PACT_PERMUTED | 66/120 (55.0%) | 56/120 (46.7%) | 34/120 (28.3%) | 3,795.2 |

**Seed-level outcomes.** Task/strict/hazard entries below are counts out of 40 in that order; contact means include all 40 rollouts.

| Condition | Seed | Live: task / strict / hazard | Permuted: task / strict / hazard | Live / permuted mean hazard samples |
|---|---|---:|---:|---:|
| C0 | 3101 | 29 / 28 / 2 | 27 / 26 / 6 | 215.9 / 2,838.1 |
| C0 | 3102 | 20 / 18 / 5 | 19 / 18 / 7 | 882.5 / 1,042.4 |
| C0 | 3103 | 26 / 24 / 3 | 20 / 19 / 9 | 732.8 / 1,830.6 |
| C2 | 3101 | 25 / 22 / 6 | 23 / 20 / 12 | 1,342.3 / 4,318.1 |
| C2 | 3102 | 13 / 13 / 11 | 8 / 8 / 13 | 3,260.2 / 3,389.2 |
| C2 | 3103 | 17 / 15 / 6 | 15 / 13 / 12 | 1,539.8 / 3,673.8 |
| Z_093 | 3101 | 26 / 21 / 6 | 28 / 24 / 9 | 1,862.2 / 3,778.3 |
| Z_093 | 3102 | 19 / 16 / 7 | 18 / 15 / 12 | 1,482.2 / 3,195.2 |
| Z_093 | 3103 | 22 / 18 / 7 | 20 / 17 / 13 | 2,803.8 / 4,412.0 |

| Live − permuted contrast | Any hazard contact, 95% CI | Mean hazard samples, 95% CI | Strict success, 95% CI |
|---|---:|---:|---:|
| C0 | −10.0 pp [−19.2, −1.7] | −1,293.3 [−2,755.7, −139.8] | +5.8 pp [0.0, +11.7] |
| C2 | −11.7 pp [−20.8, −3.3] | −1,746.3 [−3,403.8, −374.8] | +7.5 pp [−4.2, +19.2] |
| Z_093 | −11.7 pp [−21.7, −2.5] | −1,745.8 [−3,354.1, −415.4] | −0.8 pp [−10.8, +10.0] |
| Both shifted conditions pooled | −11.7 pp [−18.3, −5.4] | −1,746.1 [−2,794.3, −811.8] | Not the decision-bearing endpoint |

**Finding:** the recorded decision is `GEOMETRY_GENERALIZES`, referring specifically to preservation of the *contact advantage over the permuted control*. C0 reproduces the earlier contact-effect direction, and both shifted conditions favor live proximity on the two contact measures. Absolute live strict success nevertheless falls **16.7 pp under C2** and **12.5 pp under Z_093** relative to C0. Z_093 strict success does not favor live over permuted. Thus this is not evidence of unchanged placement performance, superiority to ACT under the shifts, or generalization of the later 128-D method.

**Earlier versions and non-results:** V1 completed 48 expert feasibility rows: clean C0 **11/12**, C1 **4/12**, C2 **12/12**, C3 **5/12**. Only one shifted condition passed, so no learned-policy evaluation ran in V1. The later feasibility records also contain **11/12** clean successes for `HALF_Y_030`, a condition omitted from the final V3 policy matrix. V2 stopped at **9/900** policy rows and was abandoned before interpretation. V3 reuses C0/C2 expert qualification and adds Z_093's screen; these are predecessors, not independent positive replications.

**Evidence:** [V3 report](PACT_GEOMETRY_GENERALIZATION_V3.md), [full V3 analysis](../diagnostics_output/pact_geometry_generalization_v3/analysis.json), [V1 feasibility](PACT_GEOMETRY_GENERALIZATION.md), [incomplete V2](PACT_GEOMETRY_GENERALIZATION_V2_PROGRESS.md).

<a id="p08-results"></a>
#### P08 — Test-time RGB blur × `PACT_PERMUTED` ablation

**Question:** as the image degrades, does live proximity preserve collision-free task completion better than ACT or unrelated proximity?

**Design:** the same **25 instances × three trained seeds × three arms × four blur levels = 900 rollouts**. The arms are ACT, frozen 32-D live PACT, and `PACT_PERMUTED`; Gaussian blur sigma is 0, 0.5, 1, or 2 at inference. No blur-aware policy is trained here. This is distinct from O03's train-time blur experiment. Scenes, checkpoints, sensor processing, and contact taxonomy remain fixed; conditions are paired by physical instance. The predeclared collapse floor is 10% strict success.

| Sigma | Arm | Task success | Strict success | Any hazard contact | Mean hazard-contact samples |
|---|---|---:|---:|---:|---:|
| 0 | ACT | 37/75 | 36/75 | 18/75 | 3,333.00 |
| 0 | Live PACT | 45/75 | 44/75 | 4/75 | 678.93 |
| 0 | PACT_PERMUTED | 36/75 | 36/75 | 14/75 | 1,945.99 |
| 0.5 | ACT | 35/75 | 33/75 | 17/75 | 3,319.07 |
| 0.5 | Live PACT | 44/75 | 42/75 | 4/75 | 657.80 |
| 0.5 | PACT_PERMUTED | 37/75 | 37/75 | 14/75 | 2,575.44 |
| 1 | ACT | 31/75 | 30/75 | 21/75 | 4,245.39 |
| 1 | Live PACT | 35/75 | 34/75 | 6/75 | 894.55 |
| 1 | PACT_PERMUTED | 31/75 | 31/75 | 18/75 | 3,162.53 |
| 2 | ACT | 14/75 | 13/75 | 25/75 | 4,507.28 |
| 2 | Live PACT | 16/75 | 14/75 | 12/75 | 2,748.24 |
| 2 | PACT_PERMUTED | 17/75 | 15/75 | 22/75 | 5,705.21 |

| Sigma | Live − ACT strict-success gap, 95% CI | Live − permuted strict-success gap, 95% CI |
|---|---:|---:|
| 0 | +10.7 pp [−5.3, +26.7] | +10.7 pp [+4.0, +17.3] |
| 0.5 | +12.0 pp [−2.7, +26.7] | +6.7 pp [−1.3, +14.7] |
| 1 | +5.3 pp [−9.3, +20.0] | +4.0 pp [−4.0, +12.0] |
| 2 | +1.3 pp [−10.7, +13.3] | −1.3 pp [−9.3, +6.7] |

| Sigma | Seed | ACT strict | Live PACT strict | PACT_PERMUTED strict |
|---|---|---:|---:|---:|
| 0 | 3101 | 11/25 | 18/25 | 15/25 |
| 0 | 3102 | 12/25 | 11/25 | 10/25 |
| 0 | 3103 | 13/25 | 15/25 | 11/25 |
| 0.5 | 3101 | 10/25 | 18/25 | 15/25 |
| 0.5 | 3102 | 11/25 | 11/25 | 10/25 |
| 0.5 | 3103 | 12/25 | 13/25 | 12/25 |
| 1 | 3101 | 12/25 | 15/25 | 11/25 |
| 1 | 3102 | 6/25 | 9/25 | 8/25 |
| 1 | 3103 | 12/25 | 10/25 | 12/25 |
| 2 | 3101 | 5/25 | 4/25 | 3/25 |
| 2 | 3102 | 4/25 | 0/25 | 0/25 |
| 2 | 3103 | 4/25 | 10/25 | 12/25 |

**Contact contrasts.** Mean-sample differences and 95% whole-instance intervals:

| Sigma | Live − ACT | Live − permuted |
|---|---:|---:|
| 0 | −2,654 [−5,552, −364] | −1,267 [−2,935, +60] |
| 0.5 | −2,661 [−5,192, −484] | −1,918 [−4,104, −228] |
| 1 | −3,351 [−6,427, −592] | −2,268 [−4,818, −279] |
| 2 | −1,759 [−3,811, −22] | −2,957 [−5,474, −899] |

**Finding:** `NO_BLUR_ROBUSTNESS`. No positive-sigma live-minus-ACT strict-success interval has a lower bound above zero. The within-instance arm-by-sigma strict-success interaction is **−5.3 pp per sigma**, CI **[−12.6, +1.4]**; the task advantage does not demonstrably widen as vision degrades. At sigma 2, pooled strict success is **17.3% ACT / 18.7% live / 20.0% permuted**, with striking seed heterogeneity, including zero successes for both PACT arms at seed 3102.

The contact result is more favorable: live PACT has fewer hazard-contact episodes than ACT and permuted at every sigma, with the reported incidence-gap intervals excluding zero. Mean contact burden also favors live PACT at every positive sigma. This supports retained contact-reduction ability under blur, while the preregistered task-robustness claim fails. It does not show that an inert policy has solved the task or that a policy trained without vision would perform similarly.

**Evidence:** [blur report](PACT_BLUR_SWEEP.md), [absolute results, per-seed data, slopes, and paired intervals](../diagnostics_output/pact_blur_sweep/analysis.json).

<a id="p09-results"></a>
#### P09 — Complete RGB removal with live and unrelated proximity

**Question:** what remains of the proximity benefit when the camera provides no task image at all?

**Design:** frozen ACT, 32-D PACT, and `PACT_PERMUTED`, each sighted and blind; **25 shared instances × three seeds × three arms × two image conditions = 450 rollouts**. Blind replaces wrist RGB with the **ImageNet mean**, not a black scene or a selectively hidden hazard. Proximity, robot state, physics, and checkpoints remain unchanged. All 450 rows reconcile and the recorded intervention flags match the schedule. No blind-policy training occurs.

| Condition / method | Task success | Strict success | Any hazard contact | Mean hazard-contact samples |
|---|---:|---:|---:|---:|
| Sighted ACT | 37/75 (49.3%) | 36/75 (48.0%) | 16/75 (21.3%) | 2,967.9 |
| Sighted live PACT | 44/75 (58.7%) | 43/75 (57.3%) | 4/75 (5.3%) | 693.6 |
| Sighted PACT_PERMUTED | 37/75 (49.3%) | 37/75 (49.3%) | 14/75 (18.7%) | 1,932.8 |
| Blind ACT | 1/75 (1.3%) | 0/75 (0.0%) | 48/75 (64.0%) | 12,289.4 |
| Blind live PACT | 1/75 (1.3%) | 1/75 (1.3%) | 37/75 (49.3%) | 8,113.7 |
| Blind PACT_PERMUTED | 0/75 (0.0%) | 0/75 (0.0%) | 46/75 (61.3%) | 9,934.5 |

**Seed-level blind outcomes.** Each row is 25 rollouts:

| Seed | Arm | Task / strict | Any hazard contact | Mean hazard-contact samples |
|---|---|---:|---:|---:|
| 3101 | ACT | 0 / 0 | 17/25 | 10,216.92 |
| 3101 | Live PACT | 1 / 1 | 12/25 | 7,212.64 |
| 3101 | PACT_PERMUTED | 0 / 0 | 17/25 | 10,343.60 |
| 3102 | ACT | 1 / 0 | 16/25 | 12,341.56 |
| 3102 | Live PACT | 0 / 0 | 12/25 | 8,986.72 |
| 3102 | PACT_PERMUTED | 0 / 0 | 13/25 | 10,156.60 |
| 3103 | ACT | 0 / 0 | 15/25 | 14,309.68 |
| 3103 | Live PACT | 0 / 0 | 13/25 | 8,141.72 |
| 3103 | PACT_PERMUTED | 0 / 0 | 16/25 | 9,303.40 |

Under blindness, live-minus-ACT mean contact burden is **−4,175.7 samples**, clustered 95% CI **[−7,356.3, −1,263.0]**; live-minus-permuted is **−1,820.8**, CI **[−2,999.0, −784.8]**. Both strict-success contrasts are only **+1.3 pp**, CI **[0.0, +4.0] pp**.

| Blind minus sighted, same arm | Strict-success degradation, paired 95% CI | Change in mean hazard samples, paired 95% CI |
|---|---:|---:|
| ACT | −48.0 pp [−64.0, −32.0] | +9,321.5 [+4,441.4, +14,340.8] |
| Live PACT | −56.0 pp [−69.3, −42.7] | +7,420.1 [+3,993.7, +11,141.4] |
| PACT_PERMUTED | −49.3 pp [−60.0, −37.3] | +8,001.7 [+4,234.6, +11,887.7] |

**Finding:** `PROXIMITY_STANDALONE_CONTACT_BENEFIT` is the saved contact decision. Proximity retains a relative contact benefit, but **task completion collapses in every blind arm**, and blind live PACT still contacts the hazard on almost half its rollouts. The numerical result supports “lower contact under camera failure,” not “safe manipulation without vision.” It also cannot substitute for an observability-controlled experiment that hides only the hazard while preserving visual task information.

The sighted counts differ slightly from P08's sigma-zero execution. Those are separately executed reference rows; this inventory retains each experiment's own outcomes rather than replacing them with whichever reference looks more favorable.

**Evidence:** [blind-RGB report](PACT_BLIND_RGB.md), [seed-level results and paired sighted/blind degradation](../diagnostics_output/pact_blind_rgb/analysis.json).

### 7.4 Local full placement: chunking, clutter, and latest readout

<a id="placement-chunk-ablations"></a>
#### L01–L03 — Placement action-chunk ablations and the chunk-100 `PACT_PERMUTED` test

**Question:** do longer action chunks overcome the failure to attempt a grasp, and, once the policies are functional, does live proximity help full pick-and-place?

**Common setup:** the recovered 152-demonstration corpus uses `PactPlaceCorridorV2Sampler` / `pact_place_corridor_v2.xml`, no added household clutter, a 900-step horizon, and training seed 3101. This recovered “V5” is not the later cluttered V5 sampler. All 152 expert recordings were recovered without divergence. The frozen 32-D corridor encoder is reused for placement. Each chunk length has its own trained ACT/PACT pair; these are not just three evaluation settings on one checkpoint. Training-command comparisons verify changes to chunk size and checkpoint directory, with the intended proximity flags distinguishing the two methods.

The original chunk-1 training completed 2,000 epochs. Best validation losses were **0.042982 ACT** and **0.047391 PACT**, at epochs 1,954 and 1,773. Those offline losses did not predict a functional grasping policy.

| Experiment | Arm | N | Task success | Strict success | Gripper-close command | Hazard-contact episodes | Other-environment contact |
|---|---|---:|---:|---:|---:|---:|---:|
| L01: chunk 1 | ACT | 20 | 0/20 | 0/20 | 0/20 | 11/20 | 0/20 |
| L01: chunk 1 | PACT | 20 | 0/20 | 0/20 | 0/20 | 2/20 | 0/20 |
| L02: chunk 25 | ACT | 40 | 8/40 | 6/40 | 9/40 | 13/40 | 0/40 |
| L02: chunk 25 | PACT | 40 | 11/40 | 10/40 | 20/40 | 6/40 | 0/40 |
| L03: chunk 100 | ACT | 40 | 13/40 | 13/40 | 40/40 | 13/40 | 0/40 |
| L03: chunk 100 | Live PACT | 40 | 19/40 | 16/40 | 40/40 | 12/40 | 3/40 |
| L03: chunk 100 | PACT_PERMUTED | 40 | 7/40 | 6/40 | 40/40 | 8/40 | 5/40 |

**L01 — Chunk 1 collapse.** All 40 rollouts reach the horizon, and neither policy ever commands closure. Collision-free episodes are ACT **9/20**, PACT **18/20**, but collision-free task completion is zero for both. The preregistered collapse threshold is at most 1/20 strict successes in either arm; both meet it. Decision: **`CHUNK1_COLLAPSE`**. Lower contact from a policy that never attempts the grasp is not task-level safety evidence.

**L02 — Chunk 25 partial recovery.** Both policies sometimes close the gripper, but at substantially different rates. The strict gap is **+10.0 pp** for PACT, approximate paired 95% interval **[−3.7, +23.7] pp**. Among episodes that command closure, ACT task/strict success is **8/9 and 6/9**, versus PACT **11/20 and 10/20**. No episode succeeds without a close command. PACT's larger unconditional count therefore includes a higher probability of attempting the grasp; the conditional comparison itself is selected by policy behavior and does not establish ACT superiority. Decision: **`CHUNK25_PARTIAL`**.

The same 20 physical instances are available at all three chunk lengths:

| Arm | Chunk | Close command | Task success | Strict success |
|---|---:|---:|---:|---:|
| ACT | 1 | 0/20 | 0/20 | 0/20 |
| ACT | 25 | 6/20 | 6/20 | 5/20 |
| ACT | 100 | 20/20 | 7/20 | 7/20 |
| PACT | 1 | 0/20 | 0/20 | 0/20 |
| PACT | 25 | 9/20 | 4/20 | 4/20 |
| PACT | 100 | 20/20 | 10/20 | 9/20 |

For the 40 instances shared by chunks 25 and 100, the difference-in-differences `(PACT − ACT)@25 − (PACT − ACT)@100` is **+2.5 pp**, approximate paired 95% interval **[−19.1, +24.1] pp**. This is not evidence that a shorter chunk increases the proximity advantage. Geometry matching uses task seed, jitters, and intrusion side; manifest role/version hashes legitimately differ for some shared scenes.

**L03 — The completed full-placement `PACT_PERMUTED` ablation.** At chunk 100 every arm commands closure in all 40 episodes, satisfying the predefined functional-policy gate. The live and permuted arms share the same PACT checkpoint. The permuted version consumes 900 complete frozen 40×32 frames per rollout, using the placement-specific token plan `5bf2ea3125842b77a4360c429cd06987fe1f0ef9c26fb33c670cb90b1e0c2eff`. Encoder identity is independently checked against `6fd2dd…3206`. The 40-scene manifest contains 20 left and 20 right approaches and has zero task-seed overlap with the 152 demonstrations.

| Chunk-100 comparison | Task-success difference | Strict-success difference | Recorded approximate paired 95% strict interval |
|---|---:|---:|---:|
| Live PACT − ACT | +15.0 pp | +7.5 pp | [−10.2, +25.2] pp, recomputed in L02 |
| Live PACT − PACT_PERMUTED | +30.0 pp | +25.0 pp | [+8.2, +41.8] pp |
| PACT_PERMUTED − ACT | −15.0 pp | −17.5 pp | Not supplied in the summary |

**Finding:** live proximity improves the measured placement outcome relative to this same-checkpoint intervention, while the ACT comparison remains imprecise. The permuted control is worse than ACT on task completion but has fewer hazard-contact episodes (**8 versus 13**). Its apparent contact advantage therefore does not imply better manipulation. Wrong sensor frames can actively derail behavior, so this result should not be described as an architecture-only control. Decision: **`FUNCTIONAL_CHUNK100`**, a bounded development result at one seed.

**Execution and limits:** chunk 1, 25, and 100 dispatches reconcile **40/40**, **80/80**, and **120/120** scientific jobs respectively, separately from their smoke checks. Chunk-25 ACT had an earlier training attempt fail at epoch 1,800 due to disk exhaustion; the completed pair was retrained and audited as recorded in its report. All scientific rollouts reach 900 steps. No household clutter is present. The older receptacle-contact diagnostic lacks an expert phase under a learned policy and can overclassify contact outside placement; retain that caveat rather than importing later contact semantics. Neither the chunk-size study nor L03 is a current-128-D encoder ablation.

**Evidence:** [demo recovery](PACT_PLACE_V5_DEMO_RECOVERY.md), [chunk-1 training](/root/pact_place_152_pact_vs_act_seed3101/EVAL.md), [chunk-1 evaluation](/root/pact_place_chunk1_eval_seed3101/EVAL.md), [chunk-25 evaluation and matched chunk contrasts](../EVAL.md), [chunk-100 evaluation](/root/pact_place_chunk100_eval_seed3101/EVAL.md).

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

<a id="readout-method-comparison"></a>
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
| Frozen PACT | 135,897 (3.050%) | 44,673 (1.003%) | 92,866 (2.084%) |
| Finetuned PACT | 85,933 (1.929%) | 54,634 (1.226%) | 32,207 (0.723%) |

Denominator: **4,455,150 audited physics samples per method**. Categories overlap; forbidden is their union including other environment and mounted fixtures. The older ACT hazard/clutter-only union 371,293 is not the complete forbidden count.

**Per-seed results for all three methods.** Every row represents 50 rollouts and 1,485,050 audited physics samples.

| Seed | Method | Task success | Strict success | Forbidden-contact samples | Hazard-contact samples |
|---|---|---:|---:|---:|---:|
| 3103 | ACT | 19/50 (38%) | 16/50 (32%) | 141,834 (9.551%) | 66,395 (4.471%) |
| 3103 | Frozen PACT | 20/50 (40%) | 16/50 (32%) | 59,086 (3.979%) | 18,183 (1.224%) |
| 3103 | Finetuned PACT | 27/50 (54%) | 24/50 (48%) | 29,000 (1.953%) | 20,148 (1.357%) |
| 3104 | ACT | 22/50 (44%) | 15/50 (30%) | 179,136 (12.063%) | 140,601 (9.468%) |
| 3104 | Frozen PACT | 19/50 (38%) | 10/50 (20%) | 52,204 (3.515%) | 14,271 (0.961%) |
| 3104 | Finetuned PACT | 26/50 (52%) | 18/50 (36%) | 32,329 (2.177%) | 19,606 (1.320%) |
| 3105 | ACT | 23/50 (46%) | 15/50 (30%) | 55,586 (3.743%) | 25,737 (1.733%) |
| 3105 | Frozen PACT | 35/50 (70%) | 25/50 (50%) | 24,607 (1.657%) | 12,219 (0.823%) |
| 3105 | Finetuned PACT | 30/50 (60%) | 21/50 (42%) | 24,604 (1.657%) | 14,880 (1.002%) |

**What changed in this representation comparison:** both proximity methods begin from the original pretrained encoder. The finetuned version uses a trainable stem/transformer, 128-D CLS readout, and minimum pooling. The earlier method uses a frozen 32-D pathway. Model width, readout, preprocessing, and joint optimization therefore change together. No recovered same-width, same-pooling, frozen-128-D control isolates the effect of unfreezing in this matrix.

**Matched outcome counts.** These numbers expose wins and losses on the same scenarios; they are not additional rollouts or an independent statistical test.

| Comparison, first versus second | First-only task successes | Second-only task successes | First-only strict successes | Second-only strict successes |
|---|---:|---:|---:|---:|
| Finetuned PACT versus ACT | 33 | 14 | 30 | 13 |
| Frozen PACT versus ACT | 34 | 24 | 28 | 23 |
| Finetuned PACT versus frozen PACT | 30 | 21 | 29 | 17 |

**Failure-stage analysis from the same 450 trajectories.** The mutually exclusive recorded hierarchy gives:

| Outcome / failure stage | ACT | Frozen PACT | Finetuned PACT |
|---|---:|---:|---:|
| Final task success | 64 | 74 | 83 |
| No target interaction | 40 | 17 | 20 |
| Touched without hold | 35 | 42 | 26 |
| Held without lift | 1 | 4 | 6 |
| Lifted without placement | 8 | 12 | 14 |
| Supported without final success | 2 | 1 | 1 |
| Total | 150 | 150 | 150 |

Independent geometric diagnostics count ≥1 cm target lift in **74 ACT / 87 frozen / 98 finetuned** rollouts and sustained lift for at least 15 observations in **69 / 79 / 91**. The contact-based “held” flag is not itself a validated stable-grasp measurement. These diagnostics indicate where behavior changes, without making a causal claim that any single encoder change fixed grasping.

**Observation:** finetuned PACT versus ACT gains 12.7 points task success and 11.3 points strict success, with 77.2% fewer forbidden-contact samples and 76.5% fewer hazard samples. Task and strict-success counts favor finetuned PACT in all three seeds.

**Limits:** relative to frozen PACT, the finetuned method gains 6.0 points task and 8.0 points strict success, but has 22.3% more hazard samples; frozen PACT wins task and strict success in seed 3105. This is a complete-method comparison, not an isolated encoder-unfreezing ablation. Different scene blocks limit attribution of between-seed differences. It is not the historical 152-demo hallway experiment or the unrun new zero-shot suite.

L09 is the original frozen comparison; L11 adds the 128-D arm. The standalone seed-3103 readout result, partial 40/50 milestones, pause/recovery reports, and final three-seed table are **constituents and revisions of this result**, not independent replications. The frozen result did not meet its predefined >50% task and ≥10-point advantage target.

**Evidence:** [three-method report](PACT_PLACE_V1010C_THREE_METHOD_COMPARISON.md), [final review](../diagnostics_output/pact_place_v1010c_readout_s3_v1/FINAL_REVIEW.md), [450 audited rows](../diagnostics_output/pact_place_v1010c_readout_s3_v1/comparison.json), [contact union reconstruction](../diagnostics_output/pact_place_v1010c_readout_s3_v1/root_review/frame_avoidance.json).

<a id="training-sampler-ablation"></a>
#### L10 — Acquisition-window sampling versus equal-update training controls

**Question:** can increased exposure to acquisition windows repair frozen PACT's grasping failures, beyond any benefit from simply doing more training?

**Design:** ACT and frozen-encoder PACT at seeds 3103–3105, with three variants per arm: the original 60,000-update checkpoint, a uniform-sampling continuation to 63,000 updates, and an acquisition-window continuation to 63,000. This is **12 trained continuation branches**, not three new PACT models alone. Stage B crosses both arms, three seeds, three variants, and the **same 12 exposed physical scenes**, giving **216 comparisons**. Of these, 36 frozen-PACT records are reused from Stage A and 180 are new Stage-B rollouts. Scene repetition across seeds and variants must be retained in interpreting the counts.

The exposure intervention worked as intended: acquisition-window starts rose from approximately **12.3–12.6%** under uniform sampling to **34.3–34.5%** under the candidate, with matched ACT/PACT start streams within each seed/variant. Both continuations receive exactly 3,000 committed updates, making uniform continuation the essential training-budget control.

| Arm | Training variant | Task success /36 | Strict success /36 | Exclusive pickup failures /36 | Hazard-or-clutter samples |
|---|---|---:|---:|---:|---:|
| ACT | Original 60,000 | 18 | 13 | 6 | 53,331 |
| ACT | Uniform 63,000 | 18 | 13 | 7 | 50,282 |
| ACT | Acquisition 63,000 | 18 | 11 | 3 | 62,179 |
| PACT | Original 60,000 | 16 | 11 | 9 | 24,296 |
| PACT | Uniform 63,000 | 22 | 14 | 7 | 17,371 |
| PACT | Acquisition 63,000 | 20 | 13 | 8 | 32,362 |

Each row has **1,069,236 audited physics samples**. Hazard-or-clutter is a union, counting simultaneous contact once. Exclusive pickup failure refers to touched-without-hold after applying the recorded failure hierarchy, not every failed grasp attempt.

| Arm | Seed | Original task / strict | Uniform task / strict | Acquisition task / strict |
|---|---|---:|---:|---:|
| ACT | 3103 | 6/12 / 4/12 | 5/12 / 4/12 | 6/12 / 3/12 |
| ACT | 3104 | 6/12 / 4/12 | 7/12 / 5/12 | 7/12 / 5/12 |
| ACT | 3105 | 6/12 / 5/12 | 6/12 / 4/12 | 5/12 / 3/12 |
| PACT | 3103 | 3/12 / 2/12 | 7/12 / 6/12 | 5/12 / 4/12 |
| PACT | 3104 | 6/12 / 4/12 | 7/12 / 4/12 | 8/12 / 4/12 |
| PACT | 3105 | 7/12 / 5/12 | 8/12 / 4/12 | 7/12 / 5/12 |

| Acquisition candidate versus control | PACT task wins / losses on matched cases | ACT task wins / losses | PACT qualifying acquisition repairs |
|---|---:|---:|---:|
| Original checkpoint | 8 / 4 | 3 / 3 | 5 |
| Uniform continuation | 1 / 3 | 3 / 3 | 1 |

A qualifying repair requires a control pickup failure, candidate target lift of at least 1 cm, and a continuous bilateral-contact interval of at least one second. The frozen gate requires three repairs against **each** control; only one qualifies against uniform. The candidate reduces PACT pickup failures by just one versus original, below the required three, and increases them versus uniform.

**Finding:** the candidate gains four PACT task successes over the original checkpoint but loses two against equal-update uniform continuation. It increases PACT contact samples **33.2% versus original** and **86.3% versus uniform**, exceeding the allowed 10% increase. For ACT, it loses two strict successes against both controls and raises contact samples **16.6% / 23.7%**. Seed-specific no-loss conditions also fail. Decision: **`STOPPED_AT_GATE_B`**; Stages C/D are unrun. Uniform continuation was better in this exposed development screen but was not promoted through a fresh final evaluation.

**Execution accounting:** the whole bounded program contains **240 valid new rollouts**: 12 in A1, 48 in A2, and 180 in B. The 216-row Stage-B table includes reused measurements and must not be added to 240 as new data. All 12 continuations finish, totaling 36,000 committed updates; two diagnosed training retries include 240 additional unsaved optimizer updates that were replayed. Independent review reconstructs the 216 Stage-B trajectories and preserves the original baseline selection.

**Evidence:** [final review and complete seed table](../diagnostics_output/pact_place_v1010b_grasp_v1/FINAL_REVIEW.md), [Stage-B gate](../diagnostics_output/pact_place_v1010b_grasp_v1/gates/B.json), [realized sampling exposure](../diagnostics_output/pact_place_v1010b_grasp_v1/root_review/realized_training_exposure.json).

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
| Y15: nine trained current-frame / four-frame / state-conditioned ablations | Mean head-space MAE **0.011337 / 0.012809 / 0.048446** respectively, versus zero-differential **0.042062**; current-frame seed-0 false-positive rate **2.15%** fails the 2% gate | Full per-seed, calibration, and model-selection results are in [the detailed Y15 section](#parked-reference-ablations) |
| Y16: trajectory-aware threshold calibration | Threshold 0.99960858 passes calibration; diagnostic hazard-absent sequence has seven consecutive false positives | Failed transfer. [Decision](HYBRID_OBSTACLE_REFERENCE_THRESHOLD_FINAL_DECISION.md) |
| Y17: proximity-only activity and causal attribution | On 17 old false positives, clear proximity lowers activity 0.9999→0.0229; state shuffling/mean replacement leaves 0.9999. No feasible proximity-only calibration threshold | Supports proximity ambiguity on these examples, not a state-only cause. [Decision](HYBRID_OBSTACLE_PROX_ACTIVITY_GATE_FINAL_DECISION.md) |
| Y18: activity identifiability / ensemble diagnostic | Changed-pixel agreement rejects 17/17 historical false positives while retaining 96.5% active frames (AUROC 0.979) | Diagnostic separation on reused failures; not final calibration success. [Decision](HYBRID_OBSTACLE_ACTIVITY_IDENTIFIABILITY_FINAL_DECISION.md) |
| Y19: trajectory-bootstrap uncertainty | Five models each see 24–28 unique clusters from 40; median agreement active 0.5467 versus zero 0.6000; no feasible threshold | Bootstrap data variation does not reproduce the seed-ensemble diagnostic. [Decision](HYBRID_OBSTACLE_UNCERTAINTY_ABSTENTION_FINAL_DECISION.md) |
| Y20: full-seed joint calibration | 1,690 feasible calibration pairs; median active recall 1.0 and zero calibration upper-bound false activation; offline transfer fails all three checks | Failed qualification. [Decision](HYBRID_OBSTACLE_FULL_SEED_JOINT_GATE_FINAL_DECISION.md) |
| Y21: three-pair agreement repair | Historical false-positive executions 10/17→9/17, threshold 0.225→0.166667; transfer still fails | Restoring the third pair does not fix the regression. [Decision](HYBRID_OBSTACLE_THREE_PAIR_JOINT_GATE_FINAL_DECISION.md) |

<a id="parked-reference-ablations"></a>
#### Y15 — Temporal-history, state-conditioning, and zero-differential ablations: nine trained models

**Question:** can a model infer the proximity field that would be measured if the removable obstacle were parked away, and does that prediction require recent history or learned conditioning on the measured proximity field?

This is a completed ablation program in the branch's **hybrid ACT + Safety-CVAE reference-learning work**. It is a different model and objective from PACT-128D. The model predicts a counterfactual 40×8×8 parked field, which feeds a frozen SafetyHead. The evaluated correction is `head(current) − head(predicted_parked)`, compared with the corresponding privileged target. Errors below are **SafetyHead-space differential MAE**, not collision rates, geometric error in millimeters, or robot task-success rates. No live policy evaluation occurs in Y15.

**Inputs and fair comparison.** Deployable inputs are proximity closeness/validity plus a 29-D state: qpos 9, qvel 9, nominal action 8, gripper state 2, and gripper command 1. No RGB, step index, timestamp, oracle differential, or parked target enters as an input. The shared model uses sensor/link identity embeddings, per-sensor convolutions, two width-192 cross-sensor transformer blocks with four attention heads, and a per-sensor decoder. The SafetyHead stays frozen; gradients pass through it to the reference model, with its output scale applied once.

| Ablation arm | Learned conditioning / history | Parameters | What it tests |
|---|---|---:|---|
| `CURRENT_FRAME_ONLY` | Current field and validity plus 29-D state | 838,434 | Single-frame reference model |
| `FULL_CAUSAL` | Four causal field/validity frames plus state | 840,162 | Whether adding history improves reference prediction |
| `QPOS_ONLY` | Learned logits from the 29-D state and learned sensor/pixel priors | 762,818 | Removal of measured proximity from the learned feature/decoder-logit path |
| `ZERO_DIFFERENTIAL` | Predicted parked field equals current field | 0 trained parameters | Predict no obstacle-specific correction |
| Privileged true-parked reference | Actual counterfactual parked field | Not a deployable model | Numerical ceiling for reproducing the target |

**Important `QPOS_ONLY` detail:** the name is shorthand; its state includes more than qpos. It is also not completely independent of proximity. All learned variants apply the same physical output constraint using current closeness: `predicted_delta = sigmoid(mask_logits) × current_closeness × sigmoid(magnitude_logits)`, then `predicted_parked = current_closeness − predicted_delta`. Thus state alone drives `QPOS_ONLY`'s learned logits, but current proximity still constrains its output. [`CausalParkedSkinReferenceV1.forward()` and `build_model()`](../causal_parked_skin/model.py) document this explicitly. The ablation isolates learned proximity conditioning, not every possible use of the current field.

**Dataset and partitioning.** The frozen dataset contains **364 trajectory files / 60,793 frames but only 100 unique episode identities**; source modes reuse identities. All copies of an episode remain in the same partition, with no crossing on five audited identity keys.

| Partition | Trajectory files | Frames | Unique episode identities | Oracle-active frames |
|---|---:|---:|---:|---:|
| Training | 256 | 43,519 | 64 | 11,308 (26.0%) |
| Validation | 24 | 3,910 | 8 | 810 (20.7%) |
| Calibration | 24 | 3,821 | 8 | 549 (14.4%) |
| Offline test | 60 | 9,543 | 20 | 1,744 (18.3%) |

All **46,382 oracle-zero frames** remain in the corpus. Training draws about 50% active frames per batch; evaluation uses the natural unmodified partitions. Learner-induced on-policy data exists **only in training**, so generalization to that source mode is not measurable on this test set.

**Validation-only architecture/loss selection, before final ablations.** Six seed-0 candidates were compared without opening offline test:

| Candidate | Hidden width / blocks | Parameters | Best validation head MAE | Selected epoch |
|---|---:|---:|---:|---:|
| `c3_active_heavy`, selected | 192 / 2 | 840,162 | 0.019381 | 39 |
| `c2_quiet_heavy` | 192 / 2 | 840,162 | 0.019994 | 51 |
| `c4_one_block` | 192 / 1 | 543,138 | 0.020122 | 44 |
| `c5_wide` | 256 / 2 | 1,379,106 | 0.021185 | 34 |
| `c6_narrow` | 128 / 2 | 440,482 | 0.022313 | 32 |
| `c1_balanced` | 192 / 2 | 840,162 | 0.025964 | 16 |

Four earlier short diagnostics are disclosed separately. An uncapped class weight of roughly 1,200 made the mask overactivate and produced validation MAE **0.2739**, worse than the zero baseline. Capping it at 32 corrected that failure. These development fits and six validation candidates are not additional independent test replications.

**Final protocol:** each of the three learned variants is fitted at seeds **0, 1, and 2**, for **nine models**. Maximum training is 100 epochs with patience 12; each checkpoint is selected by validation MAE. Calibration thresholds are then fixed from calibration oracle-zero norms. Offline test is opened after all nine fits and calibrations are finalized. The temporal-history gate allows retaining the simpler current-frame model if history does not deliver the required improvement.

| Variant | Test MAE, seed 0 | Test MAE, seed 1 | Test MAE, seed 2 | Mean test MAE | Across-seed CV |
|---|---:|---:|---:|---:|---:|
| **CURRENT_FRAME_ONLY** | 0.009087 | 0.013956 | 0.010968 | **0.011337** | 0.177 |
| FULL_CAUSAL | 0.008874 | 0.019692 | 0.009861 | 0.012809 | 0.381 |
| QPOS_ONLY | 0.049031 | 0.046296 | 0.050012 | 0.048446 | 0.034 |
| ZERO_DIFFERENTIAL | — | — | — | 0.042062 | — |
| Privileged true-parked reference | — | — | — | 3.84 × 10⁻⁸ | — |

**All nine checkpoint and activation outcomes.** Recall and false positives use each model's separately frozen calibration threshold; low false-positive rate without recall is not success.

| Variant | Seed | Selected epoch | Test active recall | Test oracle-zero false-positive rate | Test active median direction cosine |
|---|---|---:|---:|---:|---:|
| CURRENT_FRAME_ONLY | 0 | 53 | 98.28% | 2.154% | 0.99917 |
| CURRENT_FRAME_ONLY | 1 | 61 | 94.32% | 1.423% | 0.99789 |
| CURRENT_FRAME_ONLY | 2 | 82 | 97.19% | 1.103% | 0.99888 |
| FULL_CAUSAL | 0 | 60 | 97.99% | 1.334% | 0.99895 |
| FULL_CAUSAL | 1 | 16 | 83.60% | 2.718% | 0.99167 |
| FULL_CAUSAL | 2 | 46 | 91.34% | 2.885% | 0.99793 |
| QPOS_ONLY | 0 | 9 | 0.11% | 0.000% | 0.21553 |
| QPOS_ONLY | 1 | 1 | 0.57% | 0.090% | 0.35195 |
| QPOS_ONLY | 2 | 6 | 0.00% | 0.000% | 0.34307 |

**What the ablations show:**

- The current-frame model's mean MAE is **73.0% lower than zero differential** and **76.6% lower than QPOS_ONLY**. The state-conditioned model's mean error is **15.2% worse than zero differential**. Learned proximity conditioning helps this offline objective under the shared output constraint.
- Four-frame history has **13.0% higher mean MAE** and greater seed variability than the current frame. However, it has lower MAE at **two of three seeds**; its worse average is driven by seed 1. The supported conclusion is that the tested temporal model did not provide a reliable average improvement here, not that temporal information can never help proximity policies.
- The zero-differential control cannot detect active corrections. The state-conditioned arm also has almost no active recall at its calibration thresholds. Their quiet output must not be mistaken for correct hazard detection.

**Calibration and the failed readiness gate.** The current-frame model's thresholds and transfer results are:

| Seed | Frozen threshold | Calibration false-positive rate | Validation false-positive rate | Offline-test false-positive rate | ≤2% test gate |
|---|---:|---:|---:|---:|---|
| 0 | 0.0281943 | 1.01% | 2.61% | 2.15% | Fail |
| 1 | 0.0554084 | 1.01% | 1.35% | 1.42% | Pass |
| 2 | 0.0348688 | 1.01% | 1.39% | 1.10% | Pass |

The saved decision is **`PARKED_REFERENCE_MODEL_OVERFIT`**, imposed by the failure of the seed-0 false-positive gate. More precisely, the observed problem is threshold transfer: seed-0 differential MAE improves from **0.020782 validation → 0.009914 calibration → 0.009087 test**, so a simple “test error rose” account would be wrong. The report's later threshold-CV sentence uses **0.0313 / 0.1394 / 0.0368**, which `final_training.json` identifies as the **FULL_CAUSAL** thresholds; it must not be attributed to CURRENT_FRAME_ONLY's threshold table.

**Additional test measurements, current-frame seed 0:** changed-mask precision **0.797**, recall **0.903**, F1 **0.847**, AUPRC **0.900**, with changed-pixel prevalence **0.000798**. All-valid parked-field MAE is **0.0000934**, but MAE on the changed pixels is **0.07354**; reporting only the all-pixel score would hide the rare-pixel difficulty. There are zero nonfinite outputs and zero physical-bound violations. Predicted-versus-oracle correction-norm correlation is **0.934**, oracle-zero RMS **0.00628**, and hazard-absent RMS **0.00844**, compared with raw SafetyHead RMS **2.1764** on those absent frames.

| Offline source mode, seed 0 | Frames | Active frames | Differential MAE |
|---|---:|---:|---:|
| ACT-only on-policy | 4,000 | 781 | 0.007285 |
| Oracle on-policy | 4,000 | 641 | 0.008801 |
| Expert reconstructed | 1,543 | 322 | 0.014497 |
| Learner-induced on-policy | 0 | — | Not evaluable |

The validity-mask agreement entry 0.119 in the frozen training JSON is a documented obsolete metric definition. The corrected agreement is 1.000 because all 24,430,080 test pixels have true current and parked validity; that result is uninformative about handling dead sensors. The JSON is preserved rather than silently rewritten. No current-128-D history claim, closed-loop safety gain, or learner-induced-state generalization should be inferred from Y15.

**Evidence:** [full final decision and corrections](CAUSAL_PARKED_SKIN_REFERENCE_V1_FINAL_DECISION.md), [all nine trained models and test metrics](../diagnostics_output/causal_parked_skin_reference_v1/final_training.json), [model and ablation definitions](../causal_parked_skin/model.py), [validation selection](../diagnostics_output/causal_parked_skin_reference_v1/selection.json).

<a id="activity-gate-ablations"></a>
#### Y17–Y21 — Proximity/state interventions and uncertainty-control follow-ups

**Y17 question:** were the parked-reference model's early false activations caused by a robot-state/onset prior, or by how it interpreted proximity? The experiment holds the seed-0 `CURRENT_FRAME_ONLY` checkpoint fixed and applies eight interventions across five matched frame groups. The key group consists of **17 previously identified false-positive frames**, not 17 new independent episodes. Its activity score is `sigmoid(max mask logit)` from the shared decoder.

| Intervention on the same 17 frames | Mean activity | Mean predicted correction norm | Predicted changed-mask fraction |
|---|---:|---:|---:|
| Full input | 0.9999 | 0.1178 | 0.00391 |
| Identity control, current field re-supplied | 0.9999 | 0.1178 | 0.00391 |
| State shuffled within onset frames | 0.9999 | 0.1164 | 0.00402 |
| State replaced by its batch mean | 0.9999 | 0.1179 | 0.00423 |
| State swapped across hazard strata | 0.7316 | 0.0794 | 0.00324 |
| Proximity retained, state zeroed | 0.5489 | 0.0571 | 0.00076 |
| Proximity swapped across hazard strata | 0.3848 | 0.1570 | 0.00170 |
| Real state retained, proximity replaced by clear reference | **0.0229** | **0.0008** | **0.00000** |

The identity control reproduces the full-input prediction exactly. Shuffling or mean-replacing state leaves the false activation almost unchanged, while clearing proximity suppresses it. The recorded diagnostic classification is **`PROXIMITY_AMBIGUITY_DOMINANT`**. This local intervention evidence contradicts the earlier simple state-prior explanation. It does not prove formal non-identifiability from proximity in all states. Zeroing the whole state vector is itself out of distribution and is a weaker control than shuffling or mean replacement.

A separate **90,241-parameter proximity-only activity gate** was then trained once at seed 0 with a nested **40/8/8/8 episode** train/validation/calibration/test split, onset-zero weighting of 4×, and an onset-max penalty. Its inputs are only closeness and validity. It failed the joint calibration requirement of median active recall ≥0.80 and a bootstrap upper bound on mean trajectory false-positive rate ≤0.02.

| Calibration threshold | Median active recall | Mean trajectory false-positive rate | Maximum hazard-absent trajectory false-positive rate | Longest onset false-positive run |
|---|---:|---:|---:|---:|
| 0.50 | 0.802 | 0.2471 | 0.6450 | 5 |
| 0.80 | 0.751 | 0.1805 | 0.5700 | 5 |
| 0.90 | 0.580 | 0.1297 | 0.3350 | 2 |
| 0.99 | 0.022 | 0.0109 | 0.0350 | 1 |

Of **5,447** candidate thresholds, 3,502 clear the recall screen but **zero** satisfy the whole contract. Gate AUROC is **0.9893 training / 0.7859 calibration / 0.8734 nested test**, versus **0.9982 / 0.9998 / 0.9979** for the old shared head on those same splits. Decision: **`PROX_ACTIVITY_GATE_CALIBRATION_INFEASIBLE`**. No live rollouts are run in this stage.

**Y18 question:** can disagreement among the three already trained current-frame models identify those failures? The models are loaded as fixed diagnostics. Several uncertainty metrics are compared; none is silently promoted to a live controller by this audit.

| Diagnostic uncertainty measure | AUROC | Partial AUROC at 5% false-positive rate | Recorded interpretation |
|---|---:|---:|---|
| Changed-pixel-mask agreement | **0.9787** | 0.7825 | Selected diagnostic |
| Active-link-set agreement | 0.9596 | 0.8037 | Passes diagnostic criterion |
| Active-sensor-set agreement | 0.9609 | 0.7975 | Passes diagnostic criterion |
| Correction-norm coefficient of variation | 0.9476 | 0.5476 | Excluded: would also reject many ordinary quiet frames |
| Maximum parked-field variance | 0.9219 | 0.6962 | Fails criterion |
| Mean parked-field variance | 0.9035 | 0.6653 | Fails criterion |
| Mean pairwise correction cosine | 0.8968 | 0.6231 | Fails criterion |
| Minimum pairwise correction cosine | 0.8868 | 0.5385 | Fails criterion |
| Correction-norm variance | 0.6678 | 0.4872 | Fails criterion |
| Predicted-head variance | 0.6231 | 0.4872 | Fails criterion |

At the selected diagnostic operating point, mask agreement rejects **17/17** historical false positives while retaining **96.5%** of active and **94.4%** of hard-active frames. Median mask agreement is **0.1667** on historical failures versus **0.7143 / 0.7555 / 0.7778** for hard-active/late-active/onset-active groups. This operating point is distinct from the report's fixed top-5%-rejection analysis, which retains **92.39%** active and **88.20%** hard-active frames while also rejecting 17/17.

No exact current-proximity or full-input opposite-label collision was found in the 60,793-frame search under its recorded rules. Absence of such a pair in a finite corpus does not prove global observability. The immediate conclusion is useful diagnostic separation on reused examples, recorded as **`EPISTEMIC_UNCERTAINTY_SIGNAL_PRESENT`**; 17 correlated historical frames provide limited evidence of future false-positive rejection.

**Y19–Y21 test whether the diagnostic can become a calibrated gate, and record the failures:**

| Follow-up | Controlled change | Result |
|---|---|---|
| Y19: trajectory-bootstrap ensemble | Train five models on bootstrap samples of 40 episode clusters; each sees 24–28 unique clusters | Median mask agreement active **0.5467**, zero **0.6000**; no jointly feasible activation threshold. Data-bootstrap variation does not reproduce the three-seed diagnostic separation |
| Y20: full-seed joint gate | Use the original full-data seeds with joint calibration | **1,690** calibration pairs feasible, median active recall **1.0**, zero upper-bound calibration false activation; nevertheless all three offline transfer checks fail |
| Y21: three-pair agreement repair | Restore the missing pair in the three-seed agreement calculation | Historical false-positive executions improve only **10/17 → 9/17**; threshold changes **0.225 → 0.166667**; offline transfer still fails |

These negative controls prevent presenting Y18's 17/17 diagnostic rejection as a validated deployable safety mechanism. Y22 below is a separate bounded live development result; its unused uncertainty veto cannot establish that the uncertainty component caused any live benefit.

**Evidence:** [all eight input interventions and one-fit gate](HYBRID_OBSTACLE_PROX_ACTIVITY_GATE_FINAL_DECISION.md), [ten uncertainty diagnostics and identifiability audit](HYBRID_OBSTACLE_ACTIVITY_IDENTIFIABILITY_FINAL_DECISION.md), [trajectory-bootstrap ensemble](HYBRID_OBSTACLE_UNCERTAINTY_ABSTENTION_FINAL_DECISION.md), [joint calibration](HYBRID_OBSTACLE_FULL_SEED_JOINT_GATE_FINAL_DECISION.md), [three-pair repair](HYBRID_OBSTACLE_THREE_PAIR_JOINT_GATE_FINAL_DECISION.md).

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

### D.2 Source access and local audit index

This Markdown contains the numerical results, interpretations, and source links needed to read the inventory. The optional local index `docs/PROJECT_EXPERIMENTS_20260916.sources.json` records paths, hashes, source types, and the remote commit from the initial audit; it is **not included in this Markdown-only commit**. The readable source tables below remain part of this file. A listed directory is an evidence pointer, not automatically a distinct experiment or a completed rollout.

Repository-relative links point to retained code, reports, and result summaries where available. Absolute `/root/...` links identify local-only artifact roots and will require that workspace; the experiment's principal results are reproduced above so that these paths are not required to understand the conclusion. Plans and other uncommitted local documents are not evidence of completed experiments. Raw trajectory files are not exhaustively listed here.

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
