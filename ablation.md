# PACT ablation experiments

Last updated: 2026-09-18.

This is the central register for the project's ablations: the question each experiment tests, the intervention, measured results, and the claims those results support. The completed current-model experiments are recorded in full below. Earlier ablations are indexed with results and links to their detailed records. Proposed experiments remain explicitly marked as planned until their evaluation artifacts are available.

The current model is **PACT-128D with a jointly fine-tuned proximity encoder**, trained for the local **v1010/v1010c fumehood pick-and-place task**. “Frozen” in an inference intervention means keeping the already fine-tuned policy and encoder unchanged during evaluation. It does not mean using the older frozen-encoder 32-D model.

## 1. Experiment register

| ID | Experiment | Model / scope | Status | Main question |
|---|---|---|---|---|
| A01 | Live versus unrelated embeddings | Current 128-D, seed 3103 | **Completed; positive primary result** | Does the trained policy benefit from proximity corresponding to its ongoing execution? |
| A02 | Sensor-location assignment | Current 128-D, seed 3103 | **Completed; both primary dependence comparisons positive** | Does the policy depend on which body location produced each embedding? |
| A03 | History content, ordering, and freshness | Current 128-D | **Planned; no results** | Does the policy depend on past geometry, its ordering, and fresh measurements? |
| A04 | Reduce the sensor count from 40 to 20 | Current-model study discussed with the team | **Planned; execution/results not verified here** | How does reducing instrumentation affect performance? |
| S01 | VL53L4CD single-zone transfer | Current 128-D, seed 3103; 50 matched scenes | **Completed; strict success 22%; negative transfer** | Can one distance per sensor replace the original grid without retraining? |
| S02 | VL53L7CX 8×8 transfer | Current 128-D, seed 3103; 50 matched scenes | **Completed; strict success 48%; retention inconclusive** | Does the fixed model tolerate this different multizone profile? |
| S03 | VL53L7CX native 4×4 resolution | Current 128-D, seed 3103; 50 matched scenes | **Completed; strict success 50%; retention inconclusive** | Can sixteen native values replace sixty-four with other L7CX settings fixed? |
| S04 | VL53L1CB sequential 2×2 transfer | Current 128-D, seed 3103; 50 matched scenes | **Completed; strict success 16%; loss exceeds margin** | Does the fixed model tolerate four sequential ranges and narrower coverage? |
| S05 | L7CX 4×4 → 2×2 software aggregation | Current 128-D, seed 3103; 50 matched scenes | **Completed; strict success 26%; negative compression result** | Can four quadrant minima preserve performance with physical acquisition unchanged? |
| H01–H13 | Earlier ablations and related controls | Multiple older models and tasks | **Recorded separately in Section 5** | Historical evidence; preserve each experiment's model and task identity |
| External | Visibility/intervention and representation ablations held by colleague | Current 128-D, according to the team | **Visibility/interventions reported completed; representation ablations believed completed; numerical records unavailable here** | Reconcile exact conditions before duplicating or making claims |

A02 is the completed spatial-identity follow-up to A01. A03 remains planned. S01–S05 are documented in Section 8, including both reused ACT/PACT references and every completed sensor-transfer result. Current status and the retained scientific artifacts determine whether a result is complete; an implemented intervention alone is not a measured finding.

## 2. A01 — Live versus unrelated 128-D embeddings

**Status:** completed on 2026-09-16. Saved decision: `POSITIVE_PRIMARY`. All **50/50 matched pairs** are present, with no missing scenarios and no contact-censored policy failures.

### Question and intervention

Does the seed-3103 PACT policy perform better when its proximity inputs correspond to the current scene and robot execution than when it receives realistic embeddings from unrelated demonstrations?

The experiment uses **the same policy weights, fine-tuned encoder, normalization, and controller in both arms**. There is no separately trained “permuted model.”

| Arm | Input consumed by the policy |
|---|---|
| Live PACT-128D | The ordinary 40 × 128 readout from the current causal proximity history |
| `PACT_PERMUTED_128D` / unrelated embeddings | A complete 40 × 128 readout from an unrelated training episode, substituted immediately before the policy's proximity projection at every control step |

Despite the historical name `PACT_PERMUTED`, this intervention **does not permute sensor IDs or embedding coordinates**. Each donor frame preserves all 40 sensors in their original order. What is disrupted is correspondence with the current scene, RGB, robot state, and the preceding donor frames.

The donor bank uses the **final fine-tuned 128-D encoder**, not the older cached 32-D representation. Its source population is the original 240 training demonstrations, containing 117,966 control timesteps. RNG seed `2026091603` selects 900 distinct source frames for each scenario; adjacent selected frames come from different donor episodes. Reuse across scenarios is allowed. The bank has shape `(50, 900, 40, 128)` and contains 37,635 unique selected history windows.

### Fixed experimental configuration

| Item | Executed setting |
|---|---|
| Task | Local v1010/v1010c fumehood full pick-and-place |
| Training seed | 3103 |
| Checkpoint selection | Original final update **60,000**, unchanged |
| Policy and encoder | Fixed throughout evaluation; no retraining or normalization refit |
| Proximity | 40 sensors, 128-D readout per sensor |
| Encoder history | Eight consecutive control frames; minimum pooling over four raw subframes; initial-frame padding |
| Visual input | Wrist RGB, 240 × 320 |
| Action prediction | Chunk length 100 |
| Execution | Policy queried every control step; temporal aggregation history 100 |
| Horizon | 900 actions; no early stop after success; no added action noise |
| Pairing | Identical initial physical scenarios, checked against the saved live initial snapshots |
| Scientific comparison | 50 historical live rollouts and 50 new unrelated-embedding rollouts |
| Separate preflight | One live replay, excluded from the 50-pair comparison |
| Audited physics samples | 29,701 per complete rollout; 1,485,050 per arm |

This is the local clutter evaluation protocol. It is different from the manuscript's 50-action-chunk, 800-step hallway evaluation.

The original checkpoint directory is [pact_place_v1010c_readout_s3103/checkpoint](diagnostics_output/pact_place_v1010c_readout_s3103/checkpoint/).

| Artifact | SHA-256 |
|---|---|
| `policy_last.ckpt` | `3d01957cd3d86bb95db13bb4b96db05ebe07794134e4f5f64b683205afd46dcd` |
| `prox_encoder.pt` | `c1127d13cc195197c7375b5bb03081b65f9a44dabe477cb28d9408b5308e7cbc` |
| `dataset_stats.pkl` | `c15e9673e5c619ba1c86d54ab45d8a90843ec296ed768dfeb4e588808aaae107` |

### Outcome definitions

- **Placement success:** the retained evaluator's final task-success outcome.
- **Collision-free placement / strict success:** placement success and no forbidden contact anywhere during the entire rollout.
- **Any forbidden contact:** at least one audited contact in `hazard_bar`, `clutter`, `other_environment`, or `mounted_fixture`.
- **Collision-free rollout:** no forbidden contact, regardless of whether placement succeeds.
- **Contact exposure:** the fraction of audited physics samples with any forbidden contact. Simultaneous forbidden categories count once in this union.
- **Allowed contact classes:** `grasp_target` and `place_receptacle`, under the original evaluator's taxonomy. These do not count as forbidden contacts.

Contact physics samples are repeated measurements within a rollout, not independent experimental trials. Contact categories may overlap and must not be summed to reconstruct the union.

### Complete outcome results

Differences below are **live minus unrelated**. Positive differences favor live input for success/progress; negative differences favor live input for contacts. “pp” means percentage points.

| Outcome | Live PACT-128D | Unrelated embeddings | Difference |
|---|---:|---:|---:|
| **Collision-free placement — primary** | **24/50 (48%)** | **6/50 (12%)** | **+36 pp** |
| Placement success | 27/50 (54%) | 15/50 (30%) | +24 pp |
| Any forbidden contact | 18/50 (36%) | 34/50 (68%) | −32 pp |
| Collision-free rollout | 32/50 (64%) | 16/50 (32%) | +32 pp |
| Target touch | 43/50 (86%) | 29/50 (58%) | +28 pp |
| Target lift ≥1 cm | 33/50 (66%) | 16/50 (32%) | +34 pp |
| Forbidden-contact samples, total | 29,000 | 214,149 | −185,149 |
| Forbidden-contact samples, mean per rollout | 580.00 | 4,282.98 | −3,702.98 |
| Forbidden-contact samples, median per rollout | 0 | 978 | −978 |
| Forbidden-contact exposure | 1.953% | 14.420% | −12.468 pp |
| Audited physics samples | 1,485,050 | 1,485,050 | Same denominator |

The collision-free counts are the complement of forbidden-contact incidence. Rounded exposure percentages come from the exact audited counts. Unrelated embeddings produce **7.38×** the total forbidden-contact exposure of live input; equivalently, live input has **86.5% less** exposure relative to the unrelated arm. These ratios are descriptive point estimates.

### Paired statistical results

The frozen protocol specifies 20,000 paired bootstrap resamples, RNG seed `2026091604`, and 95% intervals. The sampling unit is the matched scenario. The primary binary comparison uses a two-sided exact McNemar test. A positive primary result requires both a difference interval strictly above zero and exact-test `p < 0.05`.

| Binary endpoint | Live − unrelated | Paired 95% CI | Exact McNemar p |
|---|---:|---:|---:|
| **Collision-free placement — primary** | **+36 pp** | **[+22, +50] pp** | **0.0000401** |
| Placement success | +24 pp | [+8, +40] pp | 0.01182 |
| Any forbidden contact | −32 pp | [−46, −18] pp | 0.0001450 |
| Hazard-bar contact | −16 pp | [−28, −4] pp | 0.03857 |
| Clutter contact | −34 pp | [−48, −20] pp | 0.00007629 |
| Target touch | +28 pp | [+14, +42] pp | 0.0005188 |
| Target lift ≥1 cm | +34 pp | [+18, +50] pp | 0.0002213 |

All rows except collision-free placement are **exploratory secondary analyses**; their p-values are unadjusted for multiple comparisons. They do not constitute seven independent confirmatory findings.

The primary paired contingency table is:

| Same initial scenario | Unrelated succeeds strictly | Unrelated fails strictly | Total |
|---|---:|---:|---:|
| Live succeeds strictly | 5 | 19 | 24 |
| Live fails strictly | 1 | 25 | 26 |
| Total | 6 | 44 | 50 |

Thus **19 scenarios succeed strictly only with live input**, versus **one only with unrelated input**. The exact primary p-value is `4.00543212890625e-05`.

The mean forbidden-contact difference is **−3,702.98 samples per rollout**, paired 95% CI **[−5,431.64, −2,165.80]**. The exposure-fraction difference is **−12.468 pp**, paired 95% CI **[−18.288, −7.292] pp**.

### Contact breakdown

| Contact class | Live episodes /50 | Unrelated episodes /50 | Live physics samples | Unrelated physics samples |
|---|---:|---:|---:|---:|
| Hazard bar | 6 (12%) | 14 (28%) | 20,148 | 87,222 |
| Clutter | 13 (26%) | 30 (60%) | 9,251 | 128,504 |
| Other environment | 0 | 0 | 0 | 0 |
| Mounted fixture | 0 | 0 | 0 | 0 |
| **Any forbidden contact, union** | **18 (36%)** | **34 (68%)** | **29,000** | **214,149** |

Both hazard-bar and clutter contact increase under the intervention. The union is smaller than the sum of categories because some contacts overlap.

### Implications for the paper

1. **The deployed PACT policy depends on its proximity input.** Architecture, policy weights, and the fine-tuned encoder are fixed. Replacing the consumed readouts substantially changes closed-loop outcomes, so the proximity pathway is behaviorally consequential at inference.
2. **Live correspondence contributes to successful collision avoidance under this intervention.** Live input produces both more placements and fewer forbidden contacts. Its advantage includes collision-free task completion, rather than collision avoidance without task progress.
3. **The result addresses an architecture-only explanation within this checkpoint.** The intervention changes no model capacity or trained weights. It supports a role for the supplied measurements beyond merely having trained an extra pathway. It does not quantify how much of PACT's separate advantage over ACT comes from information versus training effects.
4. **The next ablations should isolate distinct properties of the input.** A01 changes scene correspondence and temporal coherence together. A02 tests location assignment using current live embeddings; A03 tests history and freshness while preserving sensor identities.

Interpretation limits:

- This is **one user-selected checkpoint, one training seed, 50 previously evaluated scenes, and one fixed donor plan**. The result is a post hoc dependence diagnostic, not a fresh-test generalization result or a three-seed replication.
- Donor embeddings are genuine encoder outputs, but their sequence and combination with current RGB/state can be outside the joint training distribution. The comparison does not measure the optimal performance of a policy intentionally trained without proximity.
- The intervention jointly disrupts scene agreement, task-phase agreement, and temporal continuity. It does not isolate sensor identity, temporal order, spatial grid structure, or any single physical variable.
- It does not establish that encoder fine-tuning causes the benefit. That claim requires an appropriately matched training ablation.
- The results concern simulated contact under the retained evaluator, not physical-robot safety guarantees.

**Suggested manuscript wording:**

> To test whether the trained PACT-128D policy uses its proximity input during execution, we held the seed-3103 policy and its jointly fine-tuned encoder fixed and replaced each live 40 × 128 readout with a complete readout from an unrelated training episode. Across 50 matched initial scenarios, collision-free placement decreased from 48% to 12% under this intervention, a paired difference of 36 percentage points (95% bootstrap CI [22, 50]; exact two-sided McNemar p = 4.01 × 10⁻⁵). Forbidden-contact incidence increased from 36% to 68%. These results support a dependence on scene-corresponding proximity inputs for this checkpoint. Because replacement also disrupts temporal and cross-modal consistency, the experiment does not isolate their individual contributions.

### Verification and execution qualifications

The live preflight matched all **557 non-image initial fields**. Of 658,944 RGB channel values, 48 differed by one intensity level, within the retained rendering tolerance. The historical and replay task/strict outcomes agreed. Forbidden-contact samples were 10 historically and 18 on replay, so the replay demonstrates physical/input and functional compatibility but **not bit-identical closed-loop contact trajectories**. The replay is excluded from the scientific comparison. Offline checks on identical inputs reproduced live actions and history features exactly and verified unchanged model tensors.

The original supervisor stopped after seven accepted scientific rollouts. Six already-started workers completed their 900-action trajectories without recorded process exit codes. Their original attempts were admitted only after an outcome-independent audit of frozen inputs, scene matching, donor consumption, finite actions, complete trajectories/telemetry, recomputed outcomes, and completion messages. Their exit codes remain unknown. The remaining 37 scenarios used the unchanged frozen evaluator. **No completed scientific failure was rerun.** This is a documented deviation from the original observed-exit acceptance rule, not a fabricated successful exit. The study retains the recovery evidence.

### Evidence and implementation

- [Detailed execution protocol](docs/PACT_V1010C_PERMUTED_128D_3103.md).
- [Frozen machine-readable protocol](diagnostics_output/pact_place_v1010c_permuted_s3103_v1/protocol.json).
- [Saved results report](diagnostics_output/pact_place_v1010c_permuted_s3103_v1/RESULTS.md), [complete analysis](diagnostics_output/pact_place_v1010c_permuted_s3103_v1/analysis.json), and [paired records](diagnostics_output/pact_place_v1010c_permuted_s3103_v1/paired_results.csv).
- [Source manifest](diagnostics_output/pact_place_v1010c_permuted_s3103_v1/source_manifest.json), [donor plan](diagnostics_output/pact_place_v1010c_permuted_s3103_v1/donor_plan.json), and [preflight report](diagnostics_output/pact_place_v1010c_permuted_s3103_v1/preflight/report.json).
- [Recovery audit](diagnostics_output/pact_place_v1010c_permuted_s3103_v1/recovery/accepted.json).
- [Study orchestration and analysis](scripts/pact_v1010c_permuted_128d.py) and [inference intervention](scripts/pact_v1010c_permuted_128d_eval.py), especially `PermutedReadoutPolicy._surface_positions()`.
- [Results figure, PNG](diagnostics_output/pact_place_v1010c_permuted_s3103_v1/results.png) and [PDF](diagnostics_output/pact_place_v1010c_permuted_s3103_v1/results.pdf).

These evidence links refer to local experiment artifacts. Some are untracked and will not automatically be available in a GitHub-only checkout; preserve or separately distribute the artifacts when sharing the document.

## 3. A02 — Spatial assignment ablation

**Status: completed on 2026-09-17 at 00:31:35 UTC, seed 3103. All 100/100 scientific rollouts and the separate live preflight passed the final artifact audit. Both primary comparisons meet the frozen `POSITIVE_DEPENDENCE` criterion.** There are no missing scenarios, technical retries, or contact-censored policy failures. See the [execution protocol](docs/PACT_V1010C_SPATIAL_IDENTITY_128D_3103.md) and [completed health report](diagnostics_output/pact_place_v1010c_spatial_identity_s3103_v1/health/STATUS.md).

**Question:** does the trained policy rely on the correspondence between a sensor's embedding and its assigned body location?

| Condition | Implemented change | Question isolated |
|---|---|---|
| `LIVE` | Original assignment | Reference |
| `WITHIN_LINK_PERMUTED` | Reassign current live embeddings among sensor slots on the same robot link | Reliance on mounting location/orientation within a link |
| `ACROSS_LINK_PERMUTED` | Reassign current live embeddings to slots belonging to other links | Reliance on body-region assignment |

The policy's slot positional embeddings remain unchanged: measurements are deliberately assigned to the wrong identities. Reassignment occurs **after the ordinary encoder**, immediately before the proximity projection. Each 128-D vector, the complete current 40-sensor set, and all live histories are preserved.

The run reuses A01's exact seed-3103 final policy, paired fine-tuned encoder, normalization, 50 historical live scenarios, and full 900-action controller. It adds **50 within-link and 50 across-link rollouts**, each with all 40 slots reassigned. Three predefined mapping pairs are allocated to 17, 17, and 16 scenes, balanced by approach side; each scene uses one pair for both arms. Each mapping remains fixed for the entire rollout. This is not a full mapping-by-scene factorial design.

The physical-link groups contain 7, 7, 5, 5, 10, and 6 sensors. Link 5 front/back share a group; the canonical training order retains front before back. Mapping, allocation, schedule, and bootstrap seeds are `2026091605`, `2026091606`, `2026091607`, and `2026091608`, respectively. The [mapping plan](diagnostics_output/pact_place_v1010c_spatial_identity_s3103_v1/mapping_plan.json) records every moved sensor ID and mapping hash.

The identity adapter reproduced original CUDA actions/readouts exactly across 105 input steps, including history and aggregation boundaries. All six mappings passed checks at the actual projection input, inverse recovery, unchanged weights, and reset parity. The new and prior ablation test suites passed 13/13 tests. The full 900-action identity replay passed infrastructure checks, and the [scientific protocol](diagnostics_output/pact_place_v1010c_spatial_identity_s3103_v1/protocol.json) was frozen before intervention execution.

**Primary analysis:** collision-free placement, with paired exact McNemar comparisons of live versus each intervention and Holm adjustment across these two primary tests. Report pointwise paired 95% bootstrap intervals, placement/contact outcomes separately, and mapping subgroups descriptively. The within-versus-across comparison and secondary endpoints are exploratory. One selected training seed and previously evaluated scenes limit the scope of inference.

### Complete outcome results

Each arm contains the same 50 initial scenarios. The live reference is reused from the original evaluation; the two reassignment arms are new rollouts.

| Outcome | Live PACT-128D | Within-link permutation | Across-link permutation |
|---|---:|---:|---:|
| **Collision-free placement — primary** | **24/50 (48%)** | **6/50 (12%)** | **6/50 (12%)** |
| Placement success | 27/50 (54%) | 11/50 (22%) | 14/50 (28%) |
| Any forbidden contact | 18/50 (36%) | 34/50 (68%) | 33/50 (66%) |
| Collision-free rollout | 32/50 (64%) | 16/50 (32%) | 17/50 (34%) |
| Target touch | 43/50 (86%) | 25/50 (50%) | 28/50 (56%) |
| Target lift ≥1 cm | 33/50 (66%) | 17/50 (34%) | 16/50 (32%) |
| Forbidden-contact samples | 29,000 | 211,658 | 147,653 |
| Mean forbidden-contact samples per rollout | 580.00 | 4,233.16 | 2,953.06 |
| Forbidden-contact exposure | 1.953% | 14.253% | 9.943% |
| Audited physics samples | 1,485,050 | 1,485,050 | 1,485,050 |

Both interventions reduce collision-free placement by **36 percentage points**, a 75% relative decline from the live rate. They also reduce placement and increase contact; the live benefit therefore includes task progress as well as contact avoidance. Contact samples are repeated measurements within rollouts, not independent experimental trials.

### Primary paired statistics

Differences are **live minus intervention**. Intervals are pointwise paired 95% bootstrap intervals from 20,000 scenario resamples; p-values are exact two-sided McNemar tests, with Holm adjustment across the two primary contrasts.

| Primary contrast | Difference | Paired 95% CI | Exact p | Holm-adjusted p | Frozen decision |
|---|---:|---:|---:|---:|---|
| Live − within-link | +36 pp | [+22, +50] pp | 0.00004005 | 0.00008011 | `POSITIVE_DEPENDENCE` |
| Live − across-link | +36 pp | [+20, +52] pp | 0.00012112 | 0.00012112 | `POSITIVE_DEPENDENCE` |

| Matched strict-success outcomes | Both succeed | Live only | Intervention only | Neither succeeds |
|---|---:|---:|---:|---:|
| Live / within-link | 5 | 19 | 1 | 25 |
| Live / across-link | 4 | 20 | 2 | 24 |

The live advantage is supported by 19 versus 1 discordant successes for within-link reassignment, and 20 versus 2 for across-link reassignment. The two comparisons share the live reference and are not independent replications.

### Secondary paired results

These analyses are exploratory; binary p-values below are unadjusted. Positive differences favor live input for task/progress, while negative differences favor live input for contacts.

| Endpoint | Live − within-link (95% CI) | Exact p | Live − across-link (95% CI) | Exact p |
|---|---:|---:|---:|---:|
| Placement | +32 pp [+16, +48] | 0.0008554 | +26 pp [+10, +42] | 0.007197 |
| Any forbidden contact | −32 pp [−46, −18] | 0.0001450 | −30 pp [−44, −16] | 0.0002747 |
| Hazard-bar contact | −24 pp [−36, −12] | 0.0004883 | −22 pp [−34, −12] | 0.0009766 |
| Clutter contact | −14 pp [−28, 0] | 0.09229 | −14 pp [−26, −4] | 0.03906 |
| Target touch | +36 pp [+22, +50] | 0.000007629 | +30 pp [+18, +44] | 0.00006104 |
| Target lift ≥1 cm | +32 pp [+16, +46] | 0.0004025 | +34 pp [+18, +50] | 0.0002213 |

Mean forbidden-contact samples per rollout differ by **−3,653.16 [−5,719.08, −1,878.63]** for live minus within-link and **−2,373.06 [−3,590.24, −1,298.78]** for live minus across-link. The corresponding exposure differences are **−12.300 pp [−19.256, −6.325]** and **−7.990 pp [−12.088, −4.373]**.

### Contact breakdown

| Contact class | Live episodes /50 | Within-link episodes /50 | Across-link episodes /50 | Live samples | Within-link samples | Across-link samples |
|---|---:|---:|---:|---:|---:|---:|
| Hazard bar | 6 | 18 | 17 | 20,148 | 143,899 | 106,820 |
| Clutter | 13 | 20 | 20 | 9,251 | 69,786 | 40,896 |
| Other environment | 0 | 1 | 0 | 0 | 422 | 0 |
| Mounted fixture | 0 | 1 | 2 | 0 | 517 | 19 |
| **Any forbidden contact, union** | **18** | **34** | **33** | **29,000** | **211,658** | **147,653** |

Contact categories can overlap, so category counts must not be summed to reconstruct the union. Intended grasp-target and receptacle contacts retain their original allowed classification.

### Within-link versus across-link and mapping subgroups

The two interventions have the same aggregate strict-success count, with a paired within-minus-across difference of **0 pp [−10, +10]**, exact `p = 1`. This does not establish equivalence. Placement differs by **−6 pp [−18, +6]**, `p = 0.5488`, and any forbidden contact by **+2 pp [−12, +16]**, `p = 1`.

Within-link contact exposure is numerically higher, with a within-minus-across difference of **+4.310 pp [−0.176, +9.953]**. Its interval includes zero. **The study does not establish that across-link reassignment is more damaging than within-link reassignment**, or a reliable ordering between the conditions. Moving the same number of slots also does not guarantee equal physical/numerical perturbation magnitude.

| Predefined mapping pair | Matched scenarios | Live strict | Within-link strict | Across-link strict |
|---|---:|---:|---:|---:|
| 0 | 17 | 8/17 (47.1%) | 2/17 (11.8%) | 3/17 (17.6%) |
| 1 | 17 | 10/17 (58.8%) | 2/17 (11.8%) | 2/17 (11.8%) |
| 2 | 16 | 6/16 (37.5%) | 2/16 (12.5%) | 1/16 (6.3%) |

All three predefined subgroups show lower strict success under each intervention. These are descriptive checks on disjoint scene subsets, not independent training-seed replications or estimates of isolated mapping effects.

### Implications and limits

1. **Correct spatial assignment contributes to this policy's performance.** Both interventions retain the current complete set of live vectors and each sensor's causal history, while altering sensor-to-slot correspondence. Degradation therefore goes beyond A01's unrelated-scene and temporally incoherent donor stream.
2. **The policy depends on assignment within a physical link as well.** Reassignment restricted to sensors on the same link is sufficient to reduce collision-free placement. The result does not separately identify mounting position versus orientation, nor prove the policy has recovered an explicit geometric coordinate system.
3. **Together, A01 and A02 support deployed use of structured proximity input.** Architecture, model weights, and encoder are fixed. The evidence is consistent with useful measurement correspondence rather than capacity alone, but it does not apportion PACT's separate advantage over ACT between information and training effects.

The conclusions apply to **one selected training seed, 50 previously evaluated scenes, and the predefined assignment schedule**. The same live reference is reused in A01 and A02. Permutation can produce unfamiliar joint inputs even though the vectors and temporal histories remain intact. This is not evidence of arbitrary physical sensor-layout transfer, superiority over a separately trained permutation-invariant policy, generalization to new environments, or a causal benefit of encoder fine-tuning itself. A03's temporal interventions remain unrun.

**Suggested manuscript wording:**

> With the seed-3103 PACT-128D policy and its jointly fine-tuned encoder held fixed, we reassigned complete live proximity readouts to incorrect sensor slots using fixed within-link or across-link permutations. Across 50 matched initial scenarios per condition, collision-free placement fell from 48% with correct assignment to 12% in both intervention conditions, a 36-point decrease (paired 95% CIs: 22–50 and 20–52 points; Holm-adjusted exact McNemar p-values: 0.000080 and 0.000121). Because the interventions preserve the live readout set and causal sensor histories, these results support dependence on spatial sensor-to-slot correspondence in this checkpoint. The data do not establish a difference between within-link and across-link effects.

### Completion audit and evidence

All **101 new executions** (100 scientific plus one preflight) completed on their original attempt, with observed zero exit codes. Every scientific rollout passed all 900 projection-input and inverse-recovery checks, had numerically changed assignments at every control step, and matched the historical physical initialization. All 900-action trajectories and their contact telemetry were retained; no policy failures were rerun and no scenes were dropped. The independent hourly monitor recorded clean completion and stopped. The existing 50 live trajectories were reaudited for the final analysis.

- [Final results report](diagnostics_output/pact_place_v1010c_spatial_identity_s3103_v1/RESULTS.md), [complete analysis](diagnostics_output/pact_place_v1010c_spatial_identity_s3103_v1/analysis.json), and [paired records](diagnostics_output/pact_place_v1010c_spatial_identity_s3103_v1/paired_results.csv).
- [Frozen protocol](diagnostics_output/pact_place_v1010c_spatial_identity_s3103_v1/protocol.json), [mapping plan](diagnostics_output/pact_place_v1010c_spatial_identity_s3103_v1/mapping_plan.json), and [source manifest](diagnostics_output/pact_place_v1010c_spatial_identity_s3103_v1/source_manifest.json).
- [Preflight report](diagnostics_output/pact_place_v1010c_spatial_identity_s3103_v1/preflight/report.json), [accepted execution ledger](diagnostics_output/pact_place_v1010c_spatial_identity_s3103_v1/valid_ledger.jsonl), and [hash-bound completion marker](diagnostics_output/pact_place_v1010c_spatial_identity_s3103_v1/finished.json).
- [Figure PNG](diagnostics_output/pact_place_v1010c_spatial_identity_s3103_v1/results.png) and [PDF](diagnostics_output/pact_place_v1010c_spatial_identity_s3103_v1/results.pdf).

## 4. A03 — Temporal content, ordering, and freshness ablation

**Status: planned; no measured results.** The colleague may already have related current-model representation ablations; reconcile exact conditions before repeating them.

**Question:** does the trained policy use previous geometry, its temporal order, or fresh sensor feedback?

Let the original encoder window be `[P(t−7), …, P(t−1), P(t)]`, where each `P` is a min-pooled control frame. Keep the 40 sensor identities, image/state inputs, tensor shapes, and all model weights unchanged.

| Condition | Proposed input | Question isolated |
|---|---|---|
| `LIVE_HISTORY8` | Original causal eight-frame window | Reference |
| `CURRENT_REPEATED8` | Repeat `P(t)` into all eight history positions | Dependence on historical content beyond the current measurement |
| `PAST_REVERSED` | `[P(t−1), P(t−2), …, P(t−7), P(t)]` | Dependence on past ordering, preserving the frame set and newest frame's position |
| `READOUT_DELAY4` | Supply the ordinary complete encoder readout from four control steps earlier | Dependence on freshness of the proximity representation |

Repeat/reorder the **pooled control frames before encoding**, retaining the encoder's existing subframe construction. For delay, maintain the ordinary live history/encoder updates and deliver a causally buffered readout. Define initial padding before evaluation; use the first available frame/readout until sufficient history exists, with no future access. Record delay in control steps and its realized simulation time.

**Interpretation:** a drop under repeated-current input supports dependence on historical information; an additional ordering effect supports sensitivity to temporal structure. A delay effect measures sensitivity to stale feedback. The perturbations can create unfamiliar inputs, so their effects establish dependence of the trained system, not that an architecture trained without history must perform worse. Nonsignificant differences do not establish equivalence.

**Results to add:** exact window traces and padding checks; per-seed success/contact counts; paired differences; task-stage/contact diagnostics if available. Preserve null or negative results.

## 5. Earlier ablations and related controls

These are completed historical studies unless noted. Their full records are in the [project experiment inventory](docs/PROJECT_EXPERIMENTS_20260916.md#branch-ablations). Counts, contact definitions, tasks, and representations differ across studies; do not pool them into A01 or label them as current-128-D replications. “Strict” below retains each source experiment's definition.

| ID / original record | Intervention and scope | Results and implications | Detailed record |
|---|---|---|---|
| H01 / P01 | Learned 3-D PACT versus ACT and zero proximity; 960 rollouts | Original representation/removal study; preserve its original endpoint and decision separately from later embedding controls | [P01](docs/PROJECT_EXPERIMENTS_20260916.md#p01-results) |
| H02 / P02 | Frozen 32-D front-end and zero-input screen; 120 rollouts | Zero-input strict success was 1/40. The zero vector was absent from 1,247,040 training embeddings, making this an out-of-distribution failure diagnostic rather than a clean measurement-removal comparison | [P02 and support audit](docs/PROJECT_EXPERIMENTS_20260916.md#p02-results) |
| H03 / P03 | Frozen 32-D live versus unrelated frames; seed 3101; 40 pairs | Strict 29/40 versus 24/40; +12.5 pp, paired CI [−2.5, +27.5], exact p=0.2266. Suggestive, inconclusive primary signal | [P03](docs/PROJECT_EXPERIMENTS_20260916.md#p03-results) |
| H04 / P04 | Same control, independent seed 3102; shared 40 instances across two seeds | Seed-3102 strict 21/40 live versus 15/40 unrelated. Pooled live−unrelated +13.75 pp [3.75, 23.75]; live−ACT +8.75 pp [−5.0, 22.5]. The broader ACT-advantage replication failed | [P04](docs/PROJECT_EXPERIMENTS_20260916.md#p04-results) |
| H05 / P05 | Frozen 32-D live/unrelated/zero plus ACT; 100 scenes × three seeds × four arms | Live versus unrelated strict 171/300 versus 159/300; gap +4.0 pp [0.0, 8.0]. Hazard episodes 42/300 versus 70/300; mean hazard samples 1,678.2 versus 3,658.0, difference −1,979.8 [−3,152.9, −965.2]. Stronger contact evidence than task-success evidence | [P05](docs/PROJECT_EXPERIMENTS_20260916.md#p05-results) |
| H06 / P07 | Geometry × live/unrelated inputs; frozen 32-D; 720 rollouts; no ACT arm | Live retains a contact advantage under narrower/raised hazards, but absolute strict performance declines; raised-hazard strict is 55/120 live versus 56/120 unrelated. Does not establish current-model or ACT-comparative generalization | [P07](docs/PROJECT_EXPERIMENTS_20260916.md#p07-results) |
| H07 / P08 | RGB blur × live/unrelated inputs plus ACT; frozen 32-D; 900 rollouts | At sigma 2, strict is 13/75 ACT, 14/75 live, 15/75 unrelated. Contact benefits persist, but the predefined task-robustness claim fails (`NO_BLUR_ROBUSTNESS`) | [P08](docs/PROJECT_EXPERIMENTS_20260916.md#p08-results) |
| H08 / P09 | Complete RGB removal × live/unrelated inputs plus ACT; frozen 32-D; 450 rollouts | Blind strict is 0/75 ACT, 1/75 live, 0/75 unrelated. Live reduces contact relative to controls, but task completion collapses | [P09](docs/PROJECT_EXPERIMENTS_20260916.md#p09-results) |
| H09 / L01–L03 | Separately trained chunk lengths 1, 25, 100; chunk-100 unrelated-input control | Chunk 1 yields no placements in either policy. Chunk-100 strict is 13/40 ACT, 16/40 live, 6/40 unrelated; live−unrelated +25 pp, recorded approximate paired CI [8.2, 41.8]. Older 32-D placement environment | [Chunk and placement ablations](docs/PROJECT_EXPERIMENTS_20260916.md#placement-chunk-ablations) |
| H10 / L09/L11 | ACT, frozen 32-D PACT, and fine-tuned 128-D PACT; three seeds | Fine-tuned versus frozen PACT gains 6.0 pp task and 8.0 pp strict, with 22.3% more hazard samples. Several pathway changes occur together: this is a complete-method comparison, not an isolated unfreezing ablation | [Readout comparison](docs/PROJECT_EXPERIMENTS_20260916.md#readout-method-comparison) |
| H11 / L10 | Equal-update acquisition-window versus uniform training continuation | PACT task counts: original 16/36, uniform 22/36, acquisition 20/36; strict 11/36, 14/36, 13/36. Acquisition raises contact relative to both controls and fails Gate B; later stages unrun | [Training-sampler control](docs/PROJECT_EXPERIMENTS_20260916.md#training-sampler-ablation) |
| H12 / Y15 | Current-frame, causal-history, and state-conditioned parked-reference models; nine fits | Mean SafetyHead-space MAE 0.011337 / 0.012809 / 0.048446; zero-differential baseline 0.042062. History does not improve this offline objective; one calibration gate fails. Separate hybrid Safety-CVAE system, not PACT-128D | [History and conditioning ablations](docs/PROJECT_EXPERIMENTS_20260916.md#parked-reference-ablations) |
| H13 / Y17–Y21 | Proximity/state interventions and uncertainty/calibration controls | Clear-proximity, shuffled-state, and mean-state diagnostics on historical false positives; uncertainty/gating follow-ups include failed deployment gates. These do not establish a current-PACT closed-loop safety gain | [Activity-gate ablations](docs/PROJECT_EXPERIMENTS_20260916.md#activity-gate-ablations) |

Related archived studies include [free/visible/camera-hidden obstacles and training-time blur/data controls](docs/PROJECT_EXPERIMENTS_20260916.md#71-archived-obstacle-studies-visibility-representation-training-blur) and [hallway/raw-scalar/readout comparisons](docs/PROJECT_EXPERIMENTS_20260916.md#72-historical-hallway-and-remote-clutter-evaluations). Their source configurations and limitations remain in those records. The colleague's current-model studies require their own checkpoint identifiers, protocols, and numerical outcomes before adding completed entries here.

## 6. Shared protocol for the proposed current-model ablations

1. Reconcile A02/A03 with the colleague's completed controls. An experiment absent from this repository may already exist elsewhere.
2. Use the original final checkpoint and corresponding fine-tuned encoder for each seed. Start with 3103; plan replication at 3104 and 3105 using their own encoder pairs. Do not select weights, mappings, or delays using the ablation outcomes.
3. Preserve RGB, robot-state normalization, sensor count/order except the intended assignment change, action decoding, aggregation, querying cadence, and the 900-action horizon. Use isolated adapters so A01's frozen evaluator and artifacts remain intact.
4. Freeze matched scenario manifests and verify physical initialization across arms. A planning starting point is 50 scenarios per seed and condition; determine the final budget before scientific runs. Reuse historical live outcomes only after compatibility checks and disclose that reuse. Multiple mappings or repeated seeds on one scene are repeated observations, not additional independent scenes.
5. Retain collision-free placement as primary; report task completion, any forbidden contact, and contact exposure alongside it. Report per-seed estimates. Pair/cluster uncertainty by scenario and preserve repeated mappings/seeds in resampling. With only three training seeds, limit claims about training-run variability.
6. Predeclare the primary contrasts and a multiple-comparison procedure for each executed study. A02 uses Holm adjustment across its two spatial-intervention-versus-live primary tests; A03's family must be defined before it is run. Keep exploratory diagnostics labeled. Do not retrofit the later rule to A01's already frozen single-primary protocol.
7. Preserve all outcomes and technical attempts. Apply documented technical retries symmetrically; never rerun completed policy failures to obtain successes. Report implementation mismatches, recovery deviations, and incomplete conditions.
8. Append results only after checking counts against retained analysis artifacts. Report negative and inconclusive outcomes with the same prominence as positive ones. Architecture-benefit claims need matched training controls; input perturbations alone establish properties of an already trained system.

## 7. Template for each new completed entry

- **ID, title, date, status, and scientific question.**
- **Exact model/task identity:** policy and encoder checkpoints, hashes, training seed, source-data/checkpoint selection, preprocessing and controller.
- **Intervention and controls:** what changes, what is fixed, maps/donor seeds/history rules, and whether training occurs.
- **Evaluation:** matched scenario IDs, counts per seed/arm, reused versus new rollouts, contact taxonomy, rejection/retry rules.
- **Results:** numerator/denominator for every rate, paired effect sizes and confidence intervals, per-seed results, declared statistical decisions, contact exposure denominator.
- **Implications and limits:** claims supported, alternative explanations, negative/inconclusive outcomes, and scope of generalization.
- **Evidence:** frozen protocol, structured analysis, paired records, figure, implementation entry points, and execution deviations.

Update the register when an experiment changes status. Preserve the original evidence and distinguish later reanalysis from new independent experiments.

## 8. Sensor-model transfer and spatial-resolution experiments

**Status at this update (2026-09-18 UTC): five completed sensor interventions, 50 scientific rollouts each.** The original PACT and ACT results below are reused reference evaluations, not new sensor experiments. All PACT conditions retain **40 mounted sensors**. Changing 8×8 to 4×4 or 2×2 changes spatial information per sensor; it does not execute A04's proposed reduction from 40 physical sensors to 20.

These are **zero-shot inference evaluations of simulated measurement profiles**: the previously jointly fine-tuned policy and its corresponding encoder receive no further updates, and normalization is not refit. “Zero-shot” describes adaptation to the changed measurement stream; the task and these 50 initial scenarios were already evaluated. The studies do not establish unseen-environment, new-task, or physical-hardware generalization.

### Shared configuration, references, and outcome definitions

All PACT rows use the seed-3103 final-update-60,000 checkpoint, paired 128-D encoder, and normalization hashes recorded in A01. They retain wrist RGB at 240×320, the canonical 40-sensor order, the original encoder calibration/preprocessing, eight causal control frames, and minimum pooling over four observation polls. The controller predicts chunks of 100, queries every control step, uses temporal aggregation history 100, and executes all 900 actions without early success termination. Physics runs at 2 ms; the control interval is 66 ms. A repeated value or held poll is not an additional independent measurement.

Each completed intervention uses all **50 identical source scenarios**, including the original selected sampling retry, XML, object poses, and robot initialization. Initial physical observations and wrist RGB are checked against the saved references; only the declared proximity values/metadata and, where optics change, proximity-specific diagnostic projections may differ. No scenes are removed because a policy fails. Original policy/encoder/stats hashes and unchanged inference weights are audited.

The original PACT reference is the **VL53L5CX-like 8×8 simulator**, with a 45°×45° view and four simulated acquisitions at 16, 32, 48, and 64 ms per 66 ms control interval. Its nominal 60 Hz 8×8 acquisition exceeds the real L5CX's documented 8×8 maximum of 15 Hz. This source distribution is preserved as historical evidence; it is not presented as a hardware-accurate L5CX implementation. The L7CX studies use 15 Hz native acquisition. See the [L5CX driver manual](https://www.st.com/resource/en/user_manual/um2884-a-guide-to-using-the-vl53l5cx-multizone-timeofflight-ranging-sensor-with-wide-field-of-view-ultra-lite-driver-uld-stmicroelectronics.pdf) and [L7CX driver manual](https://www.st.com/resource/en/user_manual/um3038-a-guide-to-using-the-vl53l7cx-timeofflight-multizone-ranging-sensor-with-90-fov-stmicroelectronics.pdf).

The **ACT reference is the camera-only seed-3103 ACT** from the [audited three-method comparison](diagnostics_output/pact_place_v1010c_readout_s3_v1/comparison.json), using the same 50 scene identities and controller horizon. Its policy hash is `9abe289b329b78ec622f1c22984dd995fdd471aecebcd6ca6fdfa64de7fddf81`. Do not substitute the older frozen-encoder 32-D PACT comparator for ACT. On this documentation update, all 50 ACT results and raw contact telemetry were reread; their result hashes and physical scenario identities matched the source records.

**Primary outcome:** collision-free placement, requiring successful task completion and no forbidden contact anywhere in the full rollout. Forbidden classes are `hazard_bar`, `clutter`, `other_environment`, and `mounted_fixture`; intended `grasp_target` and `place_receptacle` contacts remain allowed. Collision-free rollouts include unsuccessful rollouts, so they must not be described as task success. Every complete arm has **1,485,050 audited physics samples** (50 × 29,701). Contact exposure counts the union of forbidden classes once per sample. Categories overlap; their counts must not be added to reconstruct the union.

### Complete cross-profile outcome table

All completed rows below have 50 rollouts. ACT is descriptive context; each intervention's prespecified paired comparator is identified in the statistical table.

| Policy / simulated profile | Placement | Collision-free placement | Collision-free rollout | Any forbidden contact ↓ |
|---|---:|---:|---:|---:|
| ACT — RGB + joints | 19/50 (38%) | **16/50 (32%)** | 25/50 (50%) | 25/50 (50%) |
| Original PACT — L5CX-like 8×8 | 27/50 (54%) | **24/50 (48%)** | 32/50 (64%) | 18/50 (36%) |
| S01 — VL53L4CD single-zone | 20/50 (40%) | **11/50 (22%)** | 17/50 (34%) | 33/50 (66%) |
| S02 — VL53L7CX 8×8 | 28/50 (56%) | **24/50 (48%)** | 33/50 (66%) | 17/50 (34%) |
| S03 — VL53L7CX 4×4 | 29/50 (58%) | **25/50 (50%)** | 35/50 (70%) | 15/50 (30%) |
| S04 — VL53L1CB 2×2 sequential | 19/50 (38%) | **8/50 (16%)** | 18/50 (36%) | 32/50 (64%) |
| S05 — L7CX 4×4 → 2×2 software aggregation | 19/50 (38%) | **13/50 (26%)** | 33/50 (66%) | 17/50 (34%) |

| Policy / simulated profile | Target touch | Lift ≥1 cm | Forbidden-contact samples | Contact exposure ↓ |
|---|---:|---:|---:|---:|
| ACT — RGB + joints | 36/50 (72%) | 25/50 (50%) | 141,834 | 9.551% |
| Original PACT — L5CX-like 8×8 | 43/50 (86%) | 33/50 (66%) | 29,000 | 1.953% |
| S01 — VL53L4CD single-zone | 35/50 (70%) | 25/50 (50%) | 114,359 | 7.701% |
| S02 — VL53L7CX 8×8 | 44/50 (88%) | 32/50 (64%) | 25,337 | 1.706% |
| S03 — VL53L7CX 4×4 | 43/50 (86%) | 32/50 (64%) | 25,163 | 1.694% |
| S04 — VL53L1CB 2×2 sequential | 35/50 (70%) | 22/50 (44%) | 113,468 | 7.641% |
| S05 — L7CX 4×4 → 2×2 software aggregation | 41/50 (82%) | 23/50 (46%) | 67,503 | 4.546% |

ACT's all-forbidden total is **141,834 samples (9.551%)**. The older hazard-plus-clutter union of 139,167 excludes other-environment contacts and must not replace this total. The 2×2 L1CB condition ties ACT's placement rate (38%) and has lower contact exposure, but has lower collision-free placement (16% versus 32%) and more rollouts with any forbidden contact (64% versus 50%). Therefore “worse than ACT” must identify the metric.

### Contact-class detail

Cells show **rollouts with the contact / 50; audited samples**. Overlapping categories explain why totals can exceed the union.

| Policy / simulated profile | Hazard bar | Clutter | Other environment | Mounted fixture |
|---|---:|---:|---:|---:|
| ACT — RGB + joints | 11/50; 66,395 | 21/50; 94,445 | 2/50; 2,667 | 0/50; 0 |
| Original PACT — L5CX-like 8×8 | 6/50; 20,148 | 13/50; 9,251 | 0/50; 0 | 0/50; 0 |
| S01 — VL53L4CD single-zone | 11/50; 61,210 | 25/50; 53,660 | 0/50; 0 | 0/50; 0 |
| S02 — VL53L7CX 8×8 | 4/50; 10,153 | 13/50; 15,184 | 0/50; 0 | 0/50; 0 |
| S03 — VL53L7CX 4×4 | 5/50; 13,461 | 10/50; 11,702 | 0/50; 0 | 0/50; 0 |
| S04 — VL53L1CB 2×2 sequential | 12/50; 58,855 | 24/50; 55,559 | 0/50; 0 | 0/50; 0 |
| S05 — L7CX 4×4 → 2×2 software aggregation | 5/50; 7,834 | 14/50; 59,694 | 0/50; 0 | 0/50; 0 |

### Paired statistics and the performance-retention rule

Differences are **new condition minus its specified reference** for collision-free placement. The saved analyses use exact two-sided McNemar tests and descriptive paired bootstrap 95% intervals with 20,000 scenario resamples. Bootstrap seeds are `2026091701` through `2026091704` for S01–S04; S05 uses `2026091705`. Inference conditions on one trained checkpoint pair and the matched scenes. These are sequential exploratory follow-ups sharing references, not independent replications; the listed p-values are unadjusted across sensor studies and should not be presented as a jointly controlled confirmatory family.

Each study froze a **10-percentage-point maximum acceptable loss** before its scientific outcomes. A conservative one-sided 95% lower bound above −10 pp supports retention; a conservative one-sided upper bound below −10 pp establishes a loss exceeding the margin. Otherwise retention is inconclusive. The bounds combine one-sided 97.5% Clopper–Pearson bounds for paired gain/loss probabilities. The lower and upper bounds listed below are separate one-sided bounds, not a two-sided 95% interval. A nonsignificant difference alone never establishes retention.

| Study | Primary reference | Strict-success difference | Paired bootstrap 95% CI | Exact McNemar p | Reference-only / new-only successes |
|---|---|---:|---:|---:|---:|
| S01 — VL53L4CD single-zone | Original PACT 8×8 | -26 pp | [-40, -12] pp | 0.0023498535 | 15 / 2 |
| S02 — VL53L7CX 8×8 | Original PACT 8×8 | +0 pp | [-14, +14] pp | 1 | 7 / 7 |
| S03 — VL53L7CX 4×4 | L7CX 8×8 (S02) | +2 pp | [-6, +10] pp | 1 | 2 / 3 |
| S04 — VL53L1CB 2×2 sequential | Original PACT 8×8 | -32 pp | [-46, -20] pp | 3.0517578e-05 | 16 / 0 |
| S05 — L7CX 4×4 → 2×2 software aggregation | L7CX 4×4 (S03) | -24 pp | [-38, -12] pp | 0.0018310547 | 13 / 1 |

| Study | One-sided lower bound | One-sided upper bound | Saved retention decision |
|---|---:|---:|---|
| S01 | -44.12 pp | -4.15 pp | `INCONCLUSIVE_RETENTION` |
| S02 | -20.92 pp | +20.92 pp | `INCONCLUSIVE_RETENTION` |
| S03 | -12.46 pp | +16.06 pp | `INCONCLUSIVE_RETENTION` |
| S04 | -46.70 pp | -12.41 pp | `LOSS_EXCEEDS_MARGIN` |
| S05 | -40.29 pp | -3.98 pp | `INCONCLUSIVE_RETENTION` |

S01 and S05 have statistically detectable declines versus zero, while their conservative bounds do not determine whether the loss exceeds 10 pp. That is why their saved decision is `INCONCLUSIVE_RETENTION`; it does not mean the observed degradation is absent. S02 and S03 have encouraging point estimates, but their prespecified retention tests remain inconclusive. S04 establishes a loss exceeding the margin.

### S01 — VL53L4CD single-zone transfer

**Status:** completed 2026-09-17 at 04:35 UTC; 50/50 scientific rollouts, zero technical retries, zero contact-censored failures.

**Question:** can the fixed policy and fine-tuned encoder operate with one range value per mounted sensor, supplied through a fixed scalar-to-8×8 adapter?

**Executed measurement model:** a nominal 18° circular cone, nearest visible geometric ray from a 33×33 internal integration grid, optical-axis depth converted to radial distance, a 1 mm–1.2 m envelope, and 1 mm rounding. Each scalar is broadcast across all 64 input cells. The internal 33×33 values are discarded before inference, so this is genuinely one spatial value per sensor. Returns below the envelope saturate at 1 mm; above-range values become 1.2 m. Validity is saved for audit, not supplied as a new policy channel. Retain the original four acquisitions per 66 ms; ST's up-to-100-Hz capability is not treated as a mandatory rate. The fixed cone and nearest-ray rule are declared simulation approximations, not ST firmware or calibrated reflectance/ambient-light behavior. [ST L4CD datasheet](https://www.st.com/resource/en/datasheet/vl53l4cd.pdf).

**Results:** placement falls from 27/50 to 20/50 (54% → 40%); collision-free placement falls from 24/50 to 11/50 (48% → 22%); forbidden-contact rollouts rise from 18/50 to 33/50 (36% → 66%). Exposure rises from 29,000 to 114,359 samples (1.953% → 7.701%). Strict-success difference: **−26 pp [−40, −12]**, exact `p = 0.00234985`, with 15 reference-only and two transfer-only successes.

**Implication:** this specific single-zone transfer fails to preserve the original observed safety/task performance. The result limits transfer under the full changed sensing profile; it does not prove that a policy trained for single-zone sensors cannot work. Resolution, coverage, distance convention and range limits change together. The result is retained as negative evidence, alongside the L7CX results.

**Evidence:** [protocol](docs/PACT_V1010C_VL53L4CD_128D_3103.md), [results](diagnostics_output/pact_place_v1010c_vl53l4cd_s3103_v1/RESULTS.md), [analysis](diagnostics_output/pact_place_v1010c_vl53l4cd_s3103_v1/analysis.json), [paired records](diagnostics_output/pact_place_v1010c_vl53l4cd_s3103_v1/paired_results.csv), and [sensor implementation](scripts/pact_vl53l4cd_sensor.py). Orchestrator/worker: `scripts/pact_v1010c_vl53l4cd_128d.py` and its `_eval.py` counterpart.

### S02 — VL53L7CX 8×8 transfer

**Status:** completed 2026-09-17 at 12:32 UTC; 50/50 scientific rollouts, zero technical retries, zero contact-censored failures.

**Question:** does the frozen policy tolerate a different multizone profile while retaining an 8×8 spatial grid?

**Executed measurement model:** native 8×8 ideal depth rendering, 60°×60° field of view, 2 cm–3.5 m clipping, 1 mm rounding, optical-axis depth, and 15 Hz full-grid acquisition. Native observations are held between acquisitions and polled using the original four-poll control interface; held polls are not independent captures. All forty mounts and their ordering remain fixed. The simulator uses the datasheet's 60°×60° square view directly rather than deriving a pinhole angle from the advertised 90° diagonal. Range, optics, and native rate differ from the original source, so this is a combined profile transfer. [ST L7CX datasheet](https://www.st.com/resource/en/datasheet/vl53l7cx.pdf).

**Results:** placement is 28/50 (56%), collision-free placement 24/50 (48%), collision-free rollouts 33/50 (66%), and forbidden-contact rollouts 17/50 (34%). Contact exposure is 25,337/1,485,050 (1.706%). Relative to original PACT, strict success is unchanged in aggregate: **0 pp [−14, +14]**, exact `p = 1`; seven scenes improve and seven regress.

**Implication:** the frozen policy remains useful under this simulated profile, with point estimates close to its source performance. Identical aggregate success does not imply identical behavior or established noninferiority. The one-sided lower retention bound is −20.92 pp, so formal retention within 10 pp is inconclusive. Do not describe this as proven equivalence or real-hardware transfer.

**Evidence:** [protocol](docs/PACT_V1010C_VL53L7CX_128D_3103.md), [results](diagnostics_output/pact_place_v1010c_vl53l7cx_s3103_v1/RESULTS.md), [analysis](diagnostics_output/pact_place_v1010c_vl53l7cx_s3103_v1/analysis.json), [paired records](diagnostics_output/pact_place_v1010c_vl53l7cx_s3103_v1/paired_results.csv), and [sensor implementation](scripts/pact_vl53l7cx_sensor.py). Orchestrator/worker: `scripts/pact_v1010c_vl53l7cx_128d.py` and its `_eval.py` counterpart.

### S03 — VL53L7CX native 4×4 resolution ablation

**Status:** completed 2026-09-17 at 17:11 UTC; 50/50 scientific rollouts, zero technical retries, zero contact-censored failures.

**Question:** can the same policy operate with sixteen native spatial measurements instead of sixty-four when the L7CX profile's other settings remain fixed?

**Executed intervention:** render a native 4×4 grid, then repeat each value into a 2×2 block of the expected 8×8 tensor. Hold S02's 60°×60° optics, 2 cm–3.5 m clipping, optical-axis distance convention, 1 mm rounding and 15 Hz native acquisition fixed. Although ST supports up to 60 Hz in 4×4 mode, this run retains 15 Hz to avoid changing resolution and rate simultaneously. Its primary reference is **S02's L7CX 8×8**, not the original L5CX-like source. Both modes are ideal per-zone rays rather than calibrated physical zone integration.

**Results:** placement is 29/50 (58%), collision-free placement 25/50 (50%), collision-free rollouts 35/50 (70%), and forbidden-contact rollouts 15/50 (30%). Exposure is 25,163/1,485,050 (1.694%). Relative to S02, strict success changes by **+2 pp [−6, +10]**, exact `p = 1`, with three 4×4-only and two 8×8-only successes.

**Implication:** substantial useful performance remains with sixteen native values per sensor. The two-point increase is one additional safe completion and is not evidence that fewer zones improve performance. The prespecified retention lower bound is −12.46 pp, so retention within 10 pp remains inconclusive. This is the cleanest completed within-profile resolution comparison, with the fixed shape adapter included in the intervention.

**Evidence:** [protocol](docs/PACT_V1010C_VL53L7CX_4X4_128D_3103.md), [results](diagnostics_output/pact_place_v1010c_vl53l7cx_4x4_s3103_v1/RESULTS.md), [analysis](diagnostics_output/pact_place_v1010c_vl53l7cx_4x4_s3103_v1/analysis.json), [paired records](diagnostics_output/pact_place_v1010c_vl53l7cx_4x4_s3103_v1/paired_results.csv), and [sensor implementation](scripts/pact_vl53l7cx_4x4_sensor.py). Orchestrator/worker: `scripts/pact_v1010c_vl53l7cx_4x4_128d.py` and its `_eval.py` counterpart.

### S04 — VL53L1CB 2×2 sequential-zone transfer

**Status:** completed 2026-09-17 at 22:32 UTC; 50/50 scientific rollouts, zero technical retries, zero contact-censored failures. Saved primary decision: **`LOSS_EXCEEDS_MARGIN`**.

**Question:** can the fixed policy operate with four sequentially acquired quadrant distances from a narrower-view sensor profile?

**Executed measurement model:** ST VL53L1, ordering code **VL53L1CBV0FY/1**, approximated by four quadrant-center rays. The nominal 27° diagonal field is modeled as a square pinhole with **19.2695° horizontal and vertical** fields. Depth is converted to Euclidean line-of-sight distance, rounded to 1 mm, floored at 4 cm, and uses a nominal 2.5 m detection envelope. Returns beyond the envelope map to the documented 8191 mm no-target code. The 2.5 m choice comes from a typical white-target corner-ROI specification at 16 ms; it is not the device's advertised 8 m extended-ranging mode. Below-range saturation is an explicit unvalidated model assumption. [ST L1 datasheet](https://www.st.com/resource/en/datasheet/vl53l1.pdf) and [multizone/timing manual](https://www.st.com/resource/en/user_manual/um2133-timeofflight-longdistance-ranging-sensor-with-advanced-multizone-and-multiobject-detection-stmicroelectronics.pdf).

The model updates one quadrant every approximately 16.7 ms: **60 zone updates/s, 15 complete sweeps/s**. Other quadrants retain their last value and timestamp. All four reset values are initially filled from the unchanged stationary reset state. Each of the four values is repeated into a 4×4 block of the 8×8 input. The simulation does not integrate all returns over a physical ROI, model SPAD angular response, or reproduce ST target-selection firmware.

**Results:** placement is 19/50 (38%), collision-free placement 8/50 (16%), collision-free rollouts 18/50 (36%), and forbidden-contact rollouts 32/50 (64%). Exposure is 113,468/1,485,050 (7.641%). Relative to original PACT, strict success changes by **−32 pp [−46, −20]**, exact `p = 0.0000305176`. Sixteen scenes succeed strictly only with the original profile; none succeed strictly only with this transfer. The one-sided upper difference bound is **−12.41 pp**, establishing a loss greater than the 10 pp margin.

**Implication:** this is a clear negative transfer result for the complete simulated profile. It does not isolate a failure caused by 2×2 resolution. Relative to S03, coverage narrows from 60°×60° to about 19.3°×19.3°, sixteen distances become four, simultaneous capture becomes sequential, and close-range/distance encoding changes. Reduced coverage, sparse-ray approximation, mixed acquisition times and mismatch with the frozen encoder's spatial assumptions are plausible explanations, not experimentally isolated causes. A proximity-dependent policy can be misled by unfamiliar input; ACT does not have an automatic fallback role inside PACT. Physical L1CB hardware performance has not been measured.

**Evidence:** [protocol](docs/PACT_V1010C_VL53L1_2X2_128D_3103.md), [results](diagnostics_output/pact_place_v1010c_vl53l1_2x2_s3103_v1/RESULTS.md), [analysis](diagnostics_output/pact_place_v1010c_vl53l1_2x2_s3103_v1/analysis.json), [paired records](diagnostics_output/pact_place_v1010c_vl53l1_2x2_s3103_v1/paired_results.csv), and [sensor implementation](scripts/pact_vl53l1_2x2_sensor.py). Orchestrator/worker: `scripts/pact_v1010c_vl53l1_2x2_128d.py` and its `_eval.py` counterpart.

### S05 — L7CX 4×4-to-2×2 software aggregation

**Status: completed 2026-09-18 at 02:16 UTC; all 50/50 scientific rollouts passed the final audit, with zero technical retries and zero contact-censored failures.** Launched 2026-09-17 at 23:39 UTC. The ten-step infrastructure preflight is excluded from the fifty scientific rollouts. Policy, encoder, normalization, native acquisition and evaluation scenarios remain unchanged.

**Question:** can the policy retain useful performance with four derived spatial values per sensor while preserving S03's native acquisition, field of view, distance encoding and timing?

**Executed intervention:** continue acquiring the same native 4×4 L7CX grid, then take the minimum in each nonoverlapping 2×2 block. For encoded native grid `D`, `Q[y,x] = min(D[2*y:2*y+2, 2*x:2*x+2])`. Repeat each of these four values into a 4×4 block of the encoder's 8×8 input. The original 2 cm floor, 3.5 m above-range value, 1 mm rounding and 15 Hz acquisition remain unchanged. A below-range return retains its 2 cm floor rather than being discarded as free space. Minimum aggregation was fixed before scientific outcomes; no pooling operator is selected by performance.

This is **software aggregation of sixteen measurements into four values**, not a native ST 2×2 mode, a four-ray renderer, a physical sensor-count reduction, or a hardware-cost experiment. The primary comparator is S03's native 4×4 condition. Comparison with S04 can provide descriptive context, but differs in several sensor properties and cannot isolate their individual causes.

**Verified before launch:** ten targeted tests passed; the original controller reproduced actions/readouts exactly for 105 steps; ten synthetic aggregated-history checks and ten real-scene interface steps passed; model weights remained unchanged; the initial inputs matched exact aggregation of the saved 4×4 reference; all 50 reference outcomes were reaudited. The frozen protocol uses the same 10 pp retention margin and 20,000 paired bootstrap resamples, seed `2026091705`.

The run records both native `(892,40,4,4)` and aggregated `(892,40,2,2)` streams for each full rollout, plus acquisition/poll timestamps, delivered-input hashes, complete trajectories and contact telemetry. The final audit independently recomputed quadrant minima and reconstructed the hash of all 900 consumed encoder inputs for every rollout. Initial physical pairing, native acquisition, metrics and process receipts all passed. Scientific failures remain in the results; the first full rollout was a technical gate included in the fifty, followed by up to six workers. The supervisor exited successfully and final health checks reported no issues.

**Results:** placement falls from 29/50 to 19/50 (58% → 38%); collision-free placement falls from 25/50 to 13/50 (50% → 26%). The paired primary difference is **−24 pp [−38, −12]**, with exact McNemar `p = 0.0018310547`: thirteen scenes succeed strictly only with native 4×4, and one only with software-aggregated 2×2. The conservative one-sided bounds are −40.29 pp and −3.98 pp; the prespecified 10 pp retention decision remains `INCONCLUSIVE_RETENTION`. This is a statistically detectable decline versus zero, not established retention; the conservative margin test cannot determine whether the decline exceeds 10 pp.

Any forbidden contact increases from 15/50 to 17/50 (30% → 34%; paired difference +4 pp, 95% bootstrap CI [−4, +12], exact `p = 0.625`). Exposure rises from 25,163 to 67,503 audited samples (1.694% → 4.546%). Hazard-contact incidence remains 5/50 in the same five scenes, while hazard samples decrease from 13,461 to 7,834. Clutter-contact incidence rises from 10/50 to 14/50, and clutter samples rise from 11,702 to 59,694. Categories overlap. Target touch remains relatively close (43/50 → 41/50), but lift falls from 32/50 to 23/50. Collision-free rollouts alone remain 33/50 (66%) under aggregation, illustrating why avoidance without task completion is an insufficient primary outcome.

**Implication:** this fixed four-value compression loses useful information or introduces an input mismatch for the frozen policy/encoder, despite keeping coverage, native timing, range handling and physical acquisition unchanged. Narrower coverage and sequential acquisition are therefore not necessary for a 2×2 intervention to degrade this frozen system. This does not establish the cause of the earlier L1CB result, prove that four spatial values are inherently insufficient, or measure how a policy trained on 2×2 inputs would perform. Quadrant minimum pooling and repetition are part of the intervention, so resolution cannot be separated from that adapter. The detailed failure mechanism remains unmeasured.

**Evidence:** [execution protocol](docs/PACT_V1010C_VL53L7CX_2X2_AGGREGATE_128D_3103.md), [frozen protocol](diagnostics_output/pact_place_v1010c_vl53l7cx_2x2_aggregate_s3103_v1/protocol.json), [results](diagnostics_output/pact_place_v1010c_vl53l7cx_2x2_aggregate_s3103_v1/RESULTS.md), [analysis](diagnostics_output/pact_place_v1010c_vl53l7cx_2x2_aggregate_s3103_v1/analysis.json), [paired records](diagnostics_output/pact_place_v1010c_vl53l7cx_2x2_aggregate_s3103_v1/paired_results.csv), [final audit](diagnostics_output/pact_place_v1010c_vl53l7cx_2x2_aggregate_s3103_v1/completion_review.json), [preflight](diagnostics_output/pact_place_v1010c_vl53l7cx_2x2_aggregate_s3103_v1/preflight/report.json), and [adapter](scripts/pact_vl53l7cx_2x2_aggregate_sensor.py). Orchestrator/worker: `scripts/pact_v1010c_vl53l7cx_2x2_aggregate_128d.py` and its `_eval.py` counterpart. Final output root: `diagnostics_output/pact_place_v1010c_vl53l7cx_2x2_aggregate_s3103_v1/`.

### Evidence audit and fixed artifact identities

The five completed interventions contain **250 new scientific rollouts**, in addition to reused references. All five final analysis hashes match their completion markers and reviews; none has missing scenes, technical retries, or contact-censored failures. S01 and S02 additionally ran separate complete live preflights; these are excluded from the outcome tables. All studies performed controller/history and real-scene interface checks. Saved native input provenance, scenario matching, weight hashes and recomputed outcomes support the declared interventions; they do not establish hardware fidelity or perfectly deterministic closed-loop trajectories.

Each completed output root contains `protocol.json`, `sensor_profile.json`, `source_manifest.json`, `preflight/report.json`, `valid_ledger.jsonl`, `analysis.json`, `paired_results.csv`, `RESULTS.md`, `results.png/pdf`, `completion_review.json` and `finished.json`. Original attempts and process-exit receipts are preserved. Earlier results and hash-bound implementations were not overwritten.

| Study | Final analysis file SHA-256 | Frozen protocol self-hash |
|---|---|---|
| S01 | `66796e182e1331551386aaee658b6bb03abfec71a92470c80579bcfdeba073b6` | `84f557b78e3ff75cebd6d0ee2b9b8937cbc2a2e4a70d5d846dc4513498bf043a` |
| S02 | `ee949e7a685b67a2c1f55e1f02eda229afead939792d50a6b769eddadd6522f2` | `bbc67a2b51078f6dbdd580dfb0b48670e46aa3c1fdb6802131d4f063fffd23fb` |
| S03 | `84763f6a90734a17d9c379bf9ccf11df36cbdbe9d5b2709debc9e977bba3ad23` | `cbd06e21b5f50be1d7ce43af2097ca060a423ba022c7ba7344d14cb714526112` |
| S04 | `f6a5de550270728e9057f293c371aa1442ff78141a7255484295ffa2d3048e7a` | `c5231b8e4242d5205eae5df0745db7ca77c13169df88f62030e188edb40517d9` |
| S05 | `856cbd20c02635317c9297e3693fedd1faef0ae998b9c431abbdb69235446278` | `6b387a8718bae44f7f3d15f45f6bf9d657788ba23f5e538fd233ec690d4c423f` |

Linked scripts, protocols and raw experiment artifacts are local evidence and are not all tracked on GitHub. This documentation-only update publishes the complete numerical account in `ablation.md`; it does not upload checkpoints, trajectories or the untracked sensor evaluation code.

### Combined implications and manuscript wording

The completed evidence supports **selective tolerance to sensing changes, with substantial failures under other profiles**. The L7CX 8×8 and 4×4 runs retain useful observed collision-free placement, while the single-zone L4CD and sequential 2×2 L1CB profiles degrade it. These experiments complement A01/A02: a policy that demonstrably depends on proximity should also be evaluated when that measurement stream changes. They do not show universal sensor interchangeability, prove encoder fine-tuning causes transfer, or establish that two-dimensional grid resolution alone determines success.

S05 adds a more controlled negative result: reducing the sixteen native L7CX values to four quadrant minima lowers collision-free placement from 50% to 26% while physical acquisition remains unchanged. This strengthens the evidence that the frozen system is sensitive to the spatial information delivered to it, with the compression adapter included in the intervention. More collision-free rollouts than ACT alone (66% versus 50%) would be misleading without also showing lower collision-free placement (26% versus 32%); these cross-policy point estimates are descriptive.

> With the seed-3103 policy and its jointly fine-tuned encoder held fixed, we evaluated idealized sensor-profile changes on 50 matched source scenarios. Collision-free placement was 48% with the original L5CX-like stream, 48% with an L7CX-like 8×8 stream, and 50% with an L7CX-like 4×4 stream. Single-zone L4CD-like and sequential four-zone L1CB-like profiles reduced this outcome to 22% and 16%, respectively. The L7CX point estimates indicate useful performance under these simulated changes, although the prespecified 10-point retention tests remained inconclusive. The L1CB-like transfer exceeded that loss margin. Because profile changes also alter coverage, timing and range encoding, these results describe sensitivity of the complete frozen system rather than an isolated effect of resolution or physical hardware validation.

> In a further within-profile control, retaining L7CX native 4×4 acquisition while reducing each quadrant to its minimum lowered collision-free placement from 50% to 26% (paired difference −24 points, descriptive bootstrap 95% interval [−38, −12]; exact McNemar p = 0.00183). Coverage, acquisition timing, range encoding and model weights were held fixed. This result identifies a limitation of the frozen system under this particular software spatial-compression adapter; it does not evaluate native 2×2 hardware or policies trained for four-zone input.

Report both positive-looking point estimates and negative transfers; do not select only the successful profiles for the paper.
