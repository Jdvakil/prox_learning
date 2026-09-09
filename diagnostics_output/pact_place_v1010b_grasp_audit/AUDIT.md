# V10.10b grasp audit

Completed 2026-09-08. This is an offline audit of the completed wrist280 experiment, followed by a bounded remediation plan. No simulator rollout, collection or training was launched during the audit. Subsequent execution belongs to the separate V10.10b directory and is authorized by the user's follow-up instruction.

**Recommendation:** first resolve checkpoint-versus-scene confounding with the frozen 24-scene matrix and record a few synchronized pickup replays. Then, if acquisition remains the leading loss, test one change: a 25% mixture of acquisition-window sampling during a 3,000-update continuation. Apply it to ACT and PACT with an equal-update uniform-sampling control. Do not deploy a tracking-error grasp gate: the retained successes contradict its specificity.

The evidence supports unreliable acquisition of a usable cup-wall pinch. It does not identify a unique training defect. The proposed sampling change is an engineering hypothesis with explicit failure gates, not an established cure.

## Sources, scope and reproduction

Paths are relative to `/root/prox_learning_pact_remediation`:

- `W = diagnostics_output/pact_place_v1010_wrist288_s3_v1`
- `G = W/geometric_failure_20260908`
- `S = W/amendments/seed3103_full50_20260908`
- `A = diagnostics_output/pact_place_v1010b_grasp_audit`

The audit reads `G/analysis_300.json`, the numbered geometry/contact/command files, both `_300` figures, `GEOMETRIC_DIAGNOSIS_300.md`, `final_verification.json`, `S/three_seed_full150.json`, the three extension closures/reports, original experiment/amendment plans, frozen cohort/split/statistics, six checkpoint directories, and all full-stage schedules, ledgers, receipts, trajectories and initial observations. The figures were inspected as plots, **not as RGB rollout videos**. None of the final trajectories retains its wrist video.

[evidence.json](evidence.json) contains full case identities, source SHA256s, definitions, raw per-case measurements and summaries. [rollout_audit.json](rollout_audit.json) records independent reconciliation of all 300 raw trajectories, contact classes and action traces, verified file/exit hashes, offline FK and 150 fresh initial-state pair checks. [data_audit.json](data_audit.json) records all 280 converted/raw comparisons and all six checkpoint bindings. [validation_errors.json](validation_errors.json) records 1,440 offline predictions: six phase anchors × 40 validation episodes × six frozen models. Every model loads strictly at update 60,000; no validation or final outcome selects a different checkpoint.

```bash
cd /root/prox_learning_pact_remediation
PYTHONDONTWRITEBYTECODE=1 /root/act_retrain_venv/bin/python diagnostics_output/pact_place_v1010b_grasp_audit/audit.py rollouts
PYTHONDONTWRITEBYTECODE=1 /root/act_retrain_venv/bin/python diagnostics_output/pact_place_v1010b_grasp_audit/audit.py data
PYTHONDONTWRITEBYTECODE=1 /root/act_retrain_venv/bin/python diagnostics_output/pact_place_v1010b_grasp_audit/validation_errors.py
PYTHONDONTWRITEBYTECODE=1 /root/act_retrain_venv/bin/python diagnostics_output/pact_place_v1010b_grasp_audit/audit.py assemble
PYTHONDONTWRITEBYTECODE=1 /root/act_retrain_venv/bin/python diagnostics_output/pact_place_v1010b_grasp_audit/test_audit.py
```

Only audit outputs are written. Historical scripts were inspected, not executed into `G`. The source snapshot stays fixed at 300 rollouts. The initial data-audit attempt caught an incorrect audit assumption that `policy_best.ckpt` must equal `policy_update_60000.ckpt`; the corrected audit follows the actual adapter and ledger. Both attempt logs are retained. This was an audit-script defect, not an experiment defect.

The measured raw audit took 267 seconds, data audit 108 seconds, and offline model evaluation 14 seconds, excluding inspection, writing and the superseded attempt. Budget before expensive follow-up: at most 30 CPU minutes and 15 GPU minutes for this audit, no rollout budget; the numerical work fits. New-run timing and disk estimates are in the fix plan.

## Reconciled baseline

280 accepted demonstrations, **240 train / 40 validation**, 60,000 updates per model, wrist RGB, chunk 100, history 100, identical train-only normalization and frozen 40×32 proximity embeddings. `selected_ledger.jsonl` is a **735-attempt prefix**, containing 280 accepted and 455 rejected attempts. It must be filtered by acceptance, not treated as 280 lines. The 120-demo/reduced-final/V10.11c experiments are not this baseline.

| Seed | Arm | Task success | Collision-free task success | Exclusive touch-without-hold | Raw touched, never held | Hazard/clutter frame avoidance |
|---|---|---:|---:|---:|---:|---:|
| 3103 | ACT | 19/50 | 16/50 | 12/50 | 12/50 | 90.6288% |
| 3103 | PACT | 20/50 | 16/50 | 16/50 | 18/50 | 96.0213% |
| 3104 | ACT | 22/50 | 15/50 | 12/50 | 12/50 | 87.9374% |
| 3104 | PACT | 19/50 | 10/50 | 18/50 | 19/50 | 96.4847% |
| 3105 | ACT | 23/50 | 15/50 | 11/50 | 13/50 | 96.4318% |
| 3105 | PACT | 35/50 | 25/50 | 8/50 | 10/50 | 98.3430% |

ACT task success is **64/150**, PACT **74/150**, difference **+10/150 = +6.67 percentage points**. Collision-free success is 46/150 versus 51/150. The original performance objective, PACT ≥76/150 and ≥15 successes above ACT, remains missed. Historical development/qualification flags stay false.

The exclusive hierarchy is final success → any tray support → any lift ≥1 cm → any contact-only held → any touch → no interaction. Thus PACT-3103's 43 touched minus 25 held is 18 raw episodes, but only 16 exclusive pickup failures. Two reach a later milestone. The same distinction matters for PACT-3104 and 3105.

Frame avoidance uses the **union**, not summed contact counts:

| Arm/seed | Samples | Hazard OR clutter frames | Overlap frames that must not be double-counted |
|---|---:|---:|---:|
| ACT 3103 | 1,485,050 | 139,167 | 21,673 |
| PACT 3103 | 1,485,050 | 59,086 | 1,087 |
| ACT 3104 | 1,485,050 | 179,136 | 32 |
| PACT 3104 | 1,485,050 | 52,204 | 355 |
| ACT 3105 | 1,485,050 | 52,990 | 537 |
| PACT 3105 | 1,485,050 | 24,607 | 200 |

The denominator includes recorded boundary samples. Collision-free task success additionally excludes other-environment and mounted-fixture contacts under the unchanged predicate. Whole-episode collision-free rates are a separate metric. Raw pair identities reproduce all contact class entries. Cup–clutter is included in official `clutter`; it is not necessarily direct robot obstruction.

## What reproduced, and what limits its interpretation

| Observation | Reconciliation | Interpretation and counterevidence |
|---|---|---|
| PACT 3105 gains 16 successes over 3104 | Lift ≥1 cm rises 23→38; success 19→35 | Most of the count difference arises before lift. Different checkpoint and scene block remain confounded. |
| PACT-3104 pickup failures are shallow | 16/18 have no achieved TCP below the translated, initial-orientation rim in the early window | One other case goes below the reference; one has no eligible near-cup sample. A TCP height alone is not pad insertion. |
| Hand rises during closure despite requested descent | PACT-3104 failure medians: actual +14.1 mm; command −2.7 mm over two steps | Immediate upward motion is not evidence of commanded lift. Contact forces and passive finger states are missing. |
| Successful wall pinches | Same-wall bilateral contact ≥1 s in 18/19 PACT-3104 and 32/35 PACT-3105 successes | Same-wall contact also lasts ≥1 s in 5/31 and 2/15 failures; engagement does not guarantee placement. PACT-3103 has six such failures. |
| Centre targeting is more common in PACT-3104 | Command XY offset <20 mm: 20/50 versus 5/50 in 3105 | Eight successful 3104 cases are under this threshold. It is not a valid-grasp boundary. |
| Full-rim projected span can exceed nominal opening | 21/35 successful PACT-3105 grasps exceed 87 mm | A wall pinch remains possible; full-rim enclosure is not required. |
| Paired ACT establishes solvability | 3104: 12 ACT-only successes, 10 corresponding PACT pickup failures; PACT minimum grasp height +18.5 mm median | Same-scene control is stronger than comparing separate seed blocks, but arms have different policies. |
| Strict blocked-descent signature recurs | PACT 11/50, 14/50, 8/50 for 3103/4/5 | 3103 additionally has six lifted-without-placement failures. One mechanism cannot describe all losses. |
| Wall alignment alone classifies failure poorly | PACT median angle, success vs pickup failure: 20.4°/19.9°, 25.8°/26.7°, 21.6°/17.8° | Position, angle, depth and contact chronology must be considered together. |

The depth denominator is explicit: PACT pickup groups3103/3104/3105 have14/16,17/18,8/8 eligible depth measurements. The remaining2,1,0 are unavailable, not evidence of a successful descent. Shallow counts10,16,7 are reported over the full pickup groups while medians omit unavailable measurements. Every count uses all eligible cases. Full counterexample identities are stored under `groups.*.centered_success_ids`, `wide_rim_success_ids` and `samewall_1s_failure_ids`. For example, centred PACT-3104 success `228610657102c64a8508aca9b7df8c20763ac3ad20e07363d582e59e942215e7` contradicts a universal 20 mm exclusion rule. PACT-3103 failure `83ec989cad54c17102287d7165a315e02a176c100b66b4e57754cf9a5489c8d8` contradicts treating sustained same-wall contact as final success.

### Frames, geometry and timing

TCP and robot-base poses use `xyz + wxyz`. World TCP is `R_base × TCP_base + base_translation`. Offline FK uses the actual cached Franka Droid asset and the verified **local +0.35 m mount offset**. The offset must be rotated with the base. FK over all recorded joint states reproduces the command reconstruction; maximum error around closure remains 0.978 mm, versus 2.319 mm over all states. Submillimetre effects are unresolved at this accuracy.

`actions.npz[k]` exactly matches raw `commanded_action[k+1]`, including all seven joint targets and the unchanged 127.5 gripper threshold. It follows observation k and precedes observation k+1. The two-step comparison uses command positions at k and k+2 and observed positions at k and k+2; these are clearly distinct from next-observation tracking error.

All five Cup_10 primitive collision boxes, their local offsets, sizes and quaternion conventions match saved model arrays. The initial upper edge is 71.98 mm above the cup origin. TCP local +Y is the closing axis; +Z supplies approach tilt. Cup-wall normals and projected opening span use the initial cup orientation. The Robotiq driver joints are **hinge angles in radians**; nominal inter-finger range 0–87 mm is a different quantity. The converter's assertion message calling finger states metres is inaccurate documentation. It neither rescales those states nor changes labels; the audit corrects the interpretation without modifying a frozen converter.

The rim follows measured cup translation but assumes initial orientation. Recomputing all box corners after hypothetical tilts about 72 horizontal axes gives a top-edge shift of approximately 1.0–1.4 mm at 2°, 2.4–3.4 mm at 5°, and 4.6–6.7 mm at 10°. For PACT-3104 pickup failures, 16/18 remain above the rim for every tested 2° tilt; at 5° that count is 13/18 and at 10° 2/18. These are sensitivity scenarios, **not observed rotations or probabilistic bounds**. The contact/command discrepancy survives as evidence; precise rim-side classification is weaker as rotation increases.

The grasp-state sensor inspects MuJoCo contacts without a distance filter, while the contact audit retains nonpositive-distance pairs; a control-observation touch and a raw contact onset need not be identical. The early window is all observations through close+60, filtered to TCP/cup XY distance <60 mm and cup lift <10 mm. No eligible observations gives null, not a successful descent. A minimum above zero is descriptive; it does not prove a mechanically impossible grasp. The trajectory's 31-wide `env_states/articulations/panda` record packs move-group qpos/qvel and pads with zeros; it does not restore the unrecorded passive finger joints. Final H5s lack dynamic cup actor poses. Pad penetration, normal force and slip force cannot be reconstructed exactly.

[supplement.json](supplement.json) stratifies every eligible trajectory by layout family, intrusion side, pendant pose, and outcome-independent quartiles of actual initial cup X/yaw. PACT-3104 versus3105 success is4/12 vs8/12 in F0,4/14 vs11/12 in F1,7/12 vs8/14 in F2 and4/12 vs8/12 in F3. The difference is not confined to one family. Actual pose still matters: in the highest initial-X quartile, PACT succeeds2/10 versus4/11, compared with4/10 versus15/17 in the third quartile. These uneven strata are descriptive; they do not separate checkpoint and scene effects. The same supplement checks every retained nonterminal applied move-group target against its commanded action and finds no mismatch. Requiring one fixed wall for the full1 s instead of any simultaneous same-wall mask changes zero of the300 duration classifications.

### Event reconstruction of required cases

Control steps below are observation indices except the explicit close command. Raw first-contact substeps can precede a control-observation contact by a sample; both are retained in JSON.

| Scene / policy | First cup touch | Close command | First achieved below-rim (eligible) | First held | First ≥1 cm lift | Longest same-wall contact | Outcome |
|---|---:|---:|---:|---:|---:|---:|---|
| 3104 `98c64a…` ACT | 150 | 148 | 146 | 159 | 162 | 6.070 s | success |
| 3104 `98c64a…` PACT | 144 | 148 | none | none | none | 0.566 s | pickup failure |
| 3103 `621d8b…` ACT | 121 | 126 | 128 | 133 | 138 | 12.472 s | success |
| 3103 `621d8b…` PACT | 131 | 139 | none | none | none | 0.566 s | pickup failure |
| 3105 `c0ff4f…` PACT | 153 (raw 152) | 150 | 147 | 153 | 165 | 6.952 s | success |
| 3105 `c0ff4f…` ACT | 141 | 139 | 135 | 144 | 153 | 7.122 s | success |

Full identities: `98c64a358f22eddaba4754f97d40325e9f3e9151f6968d553123681be884a33f`, `621d8bd12acc6e7256e2a671b85f49c0168e469e53ac02193dc8e0ba09be2d11`, and `c0ff4fd9273e6de8a16efdb2c68d4827f22a2bf32772633c610c90a732ec735b`.

In the 3104 pair, the meaningful divergence is already in the approach: at closure PACT requests a TCP offset 1.97 mm from the cup origin versus ACT 44.92 mm. PACT touches before closure while still above the rim; ACT descends below the reference before its first cup contact. Both close at step 148. PACT's actual/command rim heights are +5.01/−12.97 mm; ACT reaches −13.65 mm minimum and lifts. There is no robot–clutter, cup–clutter or hazard contact in either trajectory. Later empty transport cannot account for the earlier acquisition failure.

In the 3103 pair, contact precedes closure for both policies, so early contact itself is not diagnostic. ACT achieves −19.65 mm minimum and lifts; PACT stays at +3.89 mm despite requesting −11.23 mm at closure. Both share an initial brief clutter contact; neither has concurrent clutter interference at closure. The contact-to-engagement transition is where the mechanisms separate most clearly. This also counters a blanket rule forbidding every close command with a positive tracking gap.

The 3105 illustration was selected as the closest successful initial cup XY in the 3104 example's nominal cell, 17.78 mm away. It is not an identical-state control. Its paired ACT was inspected here as well.

## Dataset, training and inference audit

All 240 train and 40 validation identities are unique and agree across accepted cohort, conversion ledger and frozen split. Every converted action component exactly matches observation[t] → raw commanded_action[t+1]; the dummy and terminal status-only rows are excluded by actual fields. `sim=True` selects `utils.EpisodicDataset`'s no-additional-shift path. Statistics recomputed from all and only the 240 training episodes are byte-for-byte array-equal to the frozen statistics, shared by all six models.

The split is cell-stratified, not grasp-mode-stratified. Nearest initial cup-wall centre to the achieved closure TCP is a **geometric mode proxy**, not recorded pad contact:

| Split | n | Wall 0 | Wall 1 | Wall 3 | Wall 4 | Any recorded retry | Second closure attempt |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 240 | 87 | 34 | 43 | 76 | 0 | 0 |
| Validation | 40 | 8 | 3 | 9 | 20 | 0 | 0 |

All modes appear in both splits, but proportions differ. Do not relabel the split after observing test failures. Per-cell counts, nearest-wall labels, closure poses and numeric expert phase counts are in `data_audit.json`. Median initial-rim-relative closure TCP height is −19.58 mm in train, −19.99 mm in validation. Median episode lengths are 474.5 and 477 transitions. All accepted episodes succeed; zero planner retries and zero second closures provide no demonstrated failed-approach recovery coverage. This does not prove all small continuous corrections are absent.

Uniform timestep sampling gives the ±30-step closure window only **12.687%** of training query starts on average. The retained loss masks padded labels and then averages over the full tensor, so near-terminal windows have lower effective action-loss weight; the pad head is neither supervised by this loss nor consulted at inference. This is a shared training convention, not a demonstrated PACT-only defect. Do not change masking, KL balance and sampling together.

Actual inference method resolution is:

`Wrist288InferencePolicy → PactPlaceV109InferencePolicy → PactPlaceInferencePolicy → PactFrontendScreenInferencePolicy → PactCollisionInferencePolicy → InferencePolicy`.

The **32-D frontend override**, not the older 3-D `_surface_positions` implementation, supplies the active inference method. It uses eight causal raw sensor frames, left-padding the initial frame, and `policy_features()` from the frozen embedding encoder. The PACT projection is 512×32. ACT uses the same wrist RGB/proprioception preprocessing with no proximity projection. The class/shape checks and 900-call instrumentation in the worker agree with saved results.

The wrist adapter retains H−1 previous chunks, then the frontend adds the current chunk. At control step k it uses contributor query j at index k−j, for ages 0–99, weighted by `exp(−0.01 age)`. Newer predictions receive greater weight. There is no zero-value sentinel that drops valid rows. De-normalization precedes aggregation; affine normalization commutes with the weighted mean before the final binary gripper decoder. The source-level tests exercise age expiry, retained zero-valued predictions, decode threshold and RGB tolerance boundaries.

`Wrist288` rewrites the legacy entry point to load `policy_update_60000.ckpt`. Each ledger checkpoint hash matches that file. `policy_best.ckpt` is a different, unused artifact; its filename must not be used to infer the evaluated checkpoint. Each run has 2,000 epoch records, 60,000 optimizer updates and matching ACT/PACT data/architecture settings aside from proximity. No demonstrated binding, decode, alignment or controller-unit defect explains the seed gap. Commands are seven absolute joint-radian targets plus one 0/255 gripper actuator target; they are not TCP displacement commands.

The development history-10 condition already scored **0/24 for both arms**; history100 scored ACT 8/24, PACT 10/24. The gate was missed. Short history is not an untested quick remedy. Retained final action traces do not contain individual predicted chunks, and final RGB frames cannot be reconstructed from a missing video. Offline replay of final network predictions is therefore unavailable.

### New phase-local frozen-model comparison

For every validation demonstration, six predeclared query anchors are close−30, close−10, close, close+10, midpoint(close,T), and T−30, clipped to valid bounds. Each frozen checkpoint predicts a chunk using the retained expert observation, normalized exactly as in inference with prior latent and its actual proximity path. This is **teacher-forced offline inference**, not the final closed-loop distribution or temporal ensemble. FK error below compares predicted actions with expert commanded actions; it is not achieved pad clearance.

| Model | Median FK error at the close query, n=40 | Median close-target XY error predicted 10 steps earlier | Median valid chunk L1 at close | Median valid chunk L1 at transport |
|---|---:|---:|---:|---:|
| ACT 3103 | 12.6 mm | 9.0 mm | 0.1015 | 0.0474 |
| PACT 3103 | 17.0 mm | 16.2 mm | 0.1028 | 0.0504 |
| ACT 3104 | 13.4 mm | 14.7 mm | 0.0910 | 0.0413 |
| PACT 3104 | 17.1 mm | 6.4 mm | 0.0780 | 0.0397 |
| ACT 3105 | 13.7 mm | 9.3 mm | 0.0902 | 0.0500 |
| PACT 3105 | 11.6 mm | 8.3 mm | 0.0782 | 0.0369 |

Acquisition errors are a reasonable target for a limited supervision experiment. Counterexample: PACT-3104's preclose XY error is **lower** than PACT-3105's, and their close-query chunk losses are nearly identical. Thus neither mean validation loss nor a universal bad-localization story establishes the final 3104 failure cause. Exact closure-step gripper disagreement is transition-sensitive and cannot justify threshold retuning on its own. No checkpoint was selected using these results.

## Ranked competing explanations

| Rank / hypothesis | Support | Counterexample / alternative | Smallest distinguishing observation or intervention |
|---|---|---|---|
| 1. Unsuitable achieved target-relative grasp | Raw depth/contact split, ACT-only paired recoverability, success usually maintains a wall pinch | Centred grasps can work; angle-only groups overlap; initial-rim approximation has uncertainty | Same-scene frozen checkpoint matrix; synchronized actual cup/pad pose and contact chronology on six paired examples |
| 2. Contact limits requested descent | Commands continue downward while measured hand rises; target contact precedes shallow closure in representative failure | Positive demand gap appears in many successful grasps; normal controller lag contributes | Record substep arm target/ctrl/qpos and contact normals/forces; compare before and after closure without altering controls |
| 3. Acquisition supervision/generalization and mode coverage | Four geometric approaches, no recorded retries, query probability only 12.7%, phase errors exceed transport errors | All modes have successful demos; validation 3104 preclose error can beat 3105; weighting may amplify ambiguous modes | A single acquisition-sampling continuation against identical-update uniform continuation, same scenes and all seeds |
| 4. Temporal averaging mixes approaches | Long overlapping chunks and multimode demonstrations make it plausible | No saved chunk contributors; history10 already fails; aggregation indexing tests pass | Log each original chunk once and reconstruct contributors/weights; compare individual predicted grasp dispersion with aggregate and actual contact |
| 5. Wrist perception / proximity conditioning changes localization | Closed-loop ACT/PACT diverge at acquisition; target pixel visibility does not establish pose information | Most pickup failures saw some target pixels; sensor visibility cannot explain every failure; no retained RGB video | Six short paired wrist clips tied to a defined occlusion/pose question, plus input hashes and frozen-chunk traces; no sensor change |
| 6. Runtime binding, decode or controller discrepancy | Plausible implementation class requiring audit | All raw action alignment, checkpoint hashes, shared statistics and active 32-D method path reconcile | Existing invariant tests; only change an adapter if a reproducible defect appears |

An observed force path is still missing. The audit therefore says **contact-consistent obstruction**, not a measured upward contact force or a precise penetration depth. A zero-proximity trial would be a distribution-shift diagnostic, not evidence of sensor causality.

A candidate observable signal was screened without applying it: first close requested while actual TCP is >5 mm above commanded TCP. It triggers in PACT 38/50, 40/50, 42/50 cases, including **19/20, 16/19, 31/35 successes**, while capturing 12/16, 17/18, 7/8 exclusive pickup failures. It is far too nonspecific to label grasp failure. Waiting on it could disrupt seed3105. Do not prescribe more downward force, longer idle time, a simulator `held` gate or an oracle wall offset.

## Disposition

No historical scores or geometry need correction. Corrections are interpretive and confined here: filter the mixed attempt ledger, use hinge-angle units, resolve the active frontend/checkpoint adapter, deduplicate contact overlaps, separate raw versus exclusive pickup failures, and attach rotation sensitivity to millimetre-level rim claims. The original diagnosis is preserved and its principal observable failure is supported with the counterexamples above.

The executable [fix plan](../../docs/PACT_PLACE_V1010B_GRASP_FIX_PLAN.md) names the next adapters, frozen scene rule, matching contract, training branches, numerical gates, storage/compute budget and rollback. The first action is the small frozen cross-checkpoint diagnostic, followed conditionally by acquisition-window sampling. Evidence that would change this choice: failure disappears when checkpoint and scene are crossed; synchronized data shows an upstream binding/control defect; acquisition is not the leading loss; or a few coherent individual chunks expose aggregation as the specific source of invalid grasp positions. In those cases, stop the proposed sampling experiment and retain the unchanged baseline with a documented finding rather than inventing another intervention.
