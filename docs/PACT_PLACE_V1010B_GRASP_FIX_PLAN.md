# V10.10b: bounded grasp-acquisition remediation

Version 1, 2026-09-08. Execution root: `/root/prox_learning_pact_remediation`.

## 1. Decision, authorization and baseline

**First action:** implement the read-only instrumentation and new matching contract below, then run the 24-scene frozen PACT cross-checkpoint diagnostic. **First intervention, conditional on its acquisition gate:** continue each frozen ACT/PACT model for exactly **3,000 optimizer updates** with a **25% acquisition-window sampling mixture**, compared with an equal-update uniform-sampling control.

This changes the distribution of supervised query starts only. It tests whether increasing exposure to successful approach/closure/early-lift examples improves achieved acquisition. It does not presume sampling is the unique cause. A runtime tracking-gap gate is rejected: it triggers in 19/20, 16/19 and 31/35 historical successful PACT rollouts. Neither commanded downward force nor a simulator grasp flag is a justified correction.

The conditional fallback is **restore/retain the frozen baseline and stop**, with a causal or negative-result report. Use it on a failed numerical/scientific gate, unexplained replay mismatch, or exhausted budget. There is no second intervention, recollection round, sensor change or parameter sweep in this version. A demonstrated alternative mechanism produces a new bounded proposal rather than silently changing this experiment.

The user explicitly requested that this completed plan be delegated and executed by **GPT-6 Astra, xhigh**. That authorizes implementation and the stages that pass their prerequisites. It does not turn historical `authorizes_*` or failed qualification flags true; leave them false. Do not restart the separate Codex monitoring task or send external messages.

Read the [audit](../diagnostics_output/pact_place_v1010b_grasp_audit/AUDIT.md) and [machine-readable evidence](../diagnostics_output/pact_place_v1010b_grasp_audit/evidence.json) first. Define:

- `W = diagnostics_output/pact_place_v1010_wrist288_s3_v1`
- `A = diagnostics_output/pact_place_v1010b_grasp_audit`
- `B = diagnostics_output/pact_place_v1010b_grasp_v1`

Baseline: 280 demos, 240 train / 40 validation, three seeds 3103/3104/3105, 60,000 updates, wrist RGB, chunk100/history100, frozen 40×32 proximity encoder and shared train-only stats. Final historical success is ACT 64/150 and PACT 74/150; it missed ≥76 and ≥15 advantage. Keep this objective; do not promise it.

The baseline remains runnable through its unchanged worker and `policy_update_60000.ckpt` files. Do not run an old supervisor whose deadline has expired or whose `WORK` path points into completed results.

## 2. Exact implementation

Create new modules; preserve historical code and artifacts. Store all outputs under `B`, with schema `pact_v1010b_grasp_v1`, explicit variant `frozen60000`, `uniform63000`, or `acquisition63000` and manifest/config/code hashes.

| New file | Required interface / responsibility |
|---|---|
| `scripts/pact_v1010b_contract.py` | `build_contract(start_utc)`, `verify_contract()`, `derive_scene(role,block,cell,ordinal,retry)`, `select_cross24()`, `build_jobs(stage)`; freeze paths, parameter values, hashes, counts and deadlines before work |
| `scripts/pact_v1010b_dataset.py` | `AcquisitionWindowDataset` and deterministic `choose_start(T,close,episode_id,epoch,draw_index,p)`; labels/preprocessing remain identical to the ACT loader |
| `scripts/pact_v1010b_train.py` | `--arm --training-seed --variant --max-updates 63000 --output-dir`; clone frozen 60000 resume state into a new output directory and perform exactly 3000 additional steps |
| `scripts/pact_v1010b_eval_worker.py` | `Wrist1010bInferencePolicy`, configuration and explicit CLI with `--training-seed`, `--scene-id`, `--scene-block`, `--checkpoint-path`, `--checkpoint-sha256`, `--manifest`, `--variant`, `--instrumentation none\|grasp`, `--output-dir` |
| `scripts/pact_v1010b_pairing.py` | `audit_initial_group(rows,destination)`; intentional model differences separated from exact physical/configuration equality, original RGB tolerance retained |
| `scripts/pact_v1010b_metrics.py` | `recompute(directory)`, `compare_matched(stage)`, `evaluate_gate(stage)`; raw contacts, exclusive/raw pickup counts, sustained engagement, durations and paired outcomes |
| `scripts/pact_v1010b_run.py` | `prepare`, `diagnostic`, `train`, `mechanism`, `development`, `final`, `verify`, `supervise`; observed child exits, resource guards, staged gates and closure |
| `tests/test_pact_v1010b.py` | The mechanism/invariance/failure tests in §8 |

Reuse classes/functions from `pact_wrist288_eval_worker.py`, `pact_wrist288_metrics.py`, `pact_wrist288_analysis.py`, `pact_wrist288_train.py`, `fixed_split_data.py`, `utils.py` and the actual frontend inheritance chain. Rebinding must be confined to the new process and verified by module/class/function identity. Do not call historical `main()` unmodified: it rewrites checkpoint filenames, fixes `OUTPUT` under `W/evaluation`, and assumes old checkpoint/scene identities. New code must explicitly resolve `Wrist288 → V109 → Place → FrontendScreen → Collision`, retaining the 32-D frontend method.

### Sampling intervention

For each accepted training demo, read first close index `c` from the **converted, correctly aligned** `action[:,7] >=127.5`. All 240 have one closure. Freeze a label manifest containing episode identity, converted hash, c and interval `[max(0,c−30), min(T−1,c+30)]`, inclusive. Validation identities remain untouched. These are offline action labels, not new runtime signals.

Uniform control: `p=0`, original uniform query-start sampling over `[0,T−1]`.

Candidate: `p=0.25`; with probability 0.25 draw uniformly in the interval, otherwise uniformly over the whole trajectory. Keep the full 100-action chunk starting at the selected observation, existing padding/masks, L1 denominator, KL10, optimizer, augmentation, cameras and normalized actions unchanged. There is no importance reweighting on top. This changes training exposure by design. From measured mean baseline interval probability 0.126874, the candidate gives `0.25 + 0.75×0.126874 = 0.345156`, approximately 2.72 times as many acquisition-window starts. The 75% uniform branch protects other phases, including 3103's placement losses.

Choose 25% as a modest fixed mixture; do not search 10/25/50%, change the interval or tune a gripper threshold after looking at mechanism results. The ±30 control-step interval spans about ±1.98 s and includes approach, closure and early lift, targeting the observed sequence rather than only the binary transition.

The dataset must expose epoch and per-episode draw index. Preserve the original uniform draw/RNG stream even when the candidate overrides that result, and use a separate deterministic generator for the mixture decision/window index, domain-separated by training seed, epoch, episode ID and draw index. This keeps episode order and model/CVAE RNG streams paired as far as architectures permit. ACT and PACT within a training seed use exactly the same sampled starts per variant. Record a hash of the complete sampled `(epoch,episode,start)` stream. At `p=0`, samples/tensors and RNG progression must match the old loader.

### Training branches and budget

First test existing checkpoints in the offline audit and Stage A below. No inference-only helper is justified by that evidence. The chosen hypothesis requires changing supervised exposure; training is conditional, small, and begins only after Stage A.

Fork **12 branches**: 3 seeds × 2 arms × 2 sampling variants. Each starts from that arm/seed's original update60000 model **and optimizer/RNG state**, then runs updates60001–63000 inclusive. Total is **36,000 new updates**; each branch is 100 epochs ×30 updates at batch8. Seed names remain 3103/4/5; suffixes denote continuation, not new independent training seeds. Assert fork model tensors equal the evaluated update60000 checkpoint and optimizer step counters equal60000. Never load an unused validation-selected `policy_best` checkpoint.

Use learning rate1e−5, backbone1e−5, existing AdamW/config, 7/7 transformer layers, hidden512, feedforward3200, 8 heads, chunk100, KL10, 9-state/8-action dimensions, unchanged wrist preprocessing, frozen encoder and the exact 240/40 split/statistics. PACT encoder stays frozen. No demonstration, diagnostic scene or final scene enters the data split. Save local resumable bundles every300 updates and final model at63000; retain actual step-count receipts and strict reload verification. Keep a final model for every branch, never choose among intermediate checkpoints.

Stage training serially unless a measured two-trainer pilot using the first300 updates per branch gives ≥20% aggregate throughput gain without crossing resource limits. Retain pilot updates. Do not overlap training and rollouts. Keep at most two temporary optimizer/RNG bundles concurrently. After a branch has a zero-exit receipt, valid63000 model, verified optimizer-step/RNG metadata and hash receipt, its explicitly marked **new-run temporary** bundles may be removed; retain final model and audit metadata. If interrupted before completion, preserve its latest bundle. Original bundles are never removed.

This is an explicit new experimental allowance for short additional training, rather than the old experiment's frozen60000 evaluation. It does not alter the old result or silently grant a recovery-data or gripper-retuning exception.

### Runtime inputs and action semantics

Runtime uses only the unchanged wrist RGB, 7 arm joint positions, 2 gripper driver-joint angles, and the existing PACT proximity stream/encoder; ACT receives no proximity projection. No cup pose, pad contact, task held/success, future trajectory or expert phase is supplied to the policy. The sampling labels exist only in training.

Actions remain seven **absolute joint targets in radians**, plus one gripper target decoded to0 or255 at127.5. Control timing stays observation k → command k → observation k+1. History100 retains ages0–99 with `exp(−0.01 age)`, normalized over valid contributors. Do not train/use the untrained pad head to gate inference. Gripper joint angles are not jaw gap in metres.

There is no new runtime grasp gate or recovery state machine. Timeout remains the unchanged900-control-step horizon; `end_on_success=False`. No branch may gain success by ending early, extending time or waiting indefinitely. Capture wrist clips from the already acquired policy observation; do not re-render or request another environment observation for recording. Any telemetry fault pauses scientific execution; it is not a policy input or controller recovery.

## 3. Physical identity and instrumentation

Keep sampler `PactPlaceCorridorV1010FourObjectSampler`; target Cup_10 and its **five primitive boxes**; four clutter identities/slots01,03,04,06; assets, placement distribution, friction, limits, success predicate, wrist camera/sensor contract, horizon900 and no action noise unchanged. No V10.11 geometry or alternative mesh collider.

Every job has independent fields:

- `training_seed` and `checkpoint_sha256` identify the model;
- `scene_id`, `scene_block`, complete physical row, `task_seed_u32/u64`, successful sampling retry and sampler/config hashes identify physics;
- `variant`, instrumentation version and worker code identify execution.

Changing `training_seed` must not redraw target pose, robot start, clutter, lighting or sampling/retry choice. Feed the original physical row to the sampler unchanged; metadata such as `source_seed` in the selection file is a wrapper, not part of the physical row. For retained-scene replay, use the original successful sampling-seed entry and retry index. Failure to reproduce it is a failed replay, not permission to choose the next seed.

The new group pairing audit allows intentional checkpoint/model and variant differences **only in the separate provenance section**. Compare dataset names, shapes, dtypes and exact bytes for all initial physics qpos/qvel/act/ctrl/mocap, non-RGB policy observations, model arrays and physical configuration. RGB max absolute difference ≤2 uint8 levels and changed-channel fraction ≤0.001; retain both initial images and diff arrays. Compare source model/geom assets and controller/camera/sensor contracts too. Do not weaken the historical ACT/PACT pair checker or ignore arbitrary configuration fields.

### Synchronized diagnostic recording

On all **new Stage A PACT matrix runs**, record each predicted normalized chunk once, shape900×100×8 float32, its query control index, de-normalized or reconstructible statistics hash, and per-step contributor query IDs/ages/weights. Preserve the original post-aggregation trace and applied actions. This costs about2.75 MiB/chunk trace per rollout before compression. Do not store the same full chunk at every contributing step. Hooks must return the original tensors without modifying outputs or RNG.

Select **six curated instrumented scenes**, all from the original final manifests: the required3103 and3104 failures; the nearest-cell3105 success `c0ff4f…`; and one SHA256-ranked PACT success from each of3103/3104 plus one SHA256-ranked strict blocked-descent PACT failure from3105. Exclude already selected identities. Use digest of `pact_place_v1010b_grasp_v1/instrument/` + episode ID for these rankings. This is an explicitly outcome-selected mechanism set, not an estimator of prevalence. Run both original ACT/PACT checkpoints on each: **12 additional replays**, retaining failures.

For these12, record at2 ms physics sampling from reset through the first closure+60 control steps, capped at300 steps; if no closure occurs, record to300. Retain control-rate measurements thereafter. Record:

- actual cup world position/quaternion; pad world position/quaternion and complete driver/passive finger qpos/qvel;
- contact pair geom/body IDs, point, normal, signed distance and normal/tangential forces obtained at the physics callback after the solver state is valid;
- requested arm targets, applied ctrl and actual arm qpos/qvel; command/observation/physics timestamps;
- predicted chunks, contributor weights and original aggregate;
- wrist RGB from close−30 through close+60, retaining a ring buffer; cap150frames,240×320,15fps. For no closure retain the final150frames within the instrumentation cap.

Use poses/forces only in diagnostics. Define and test force-frame/sign convention. Retained original H5s cannot establish that convention. Treat clips as new replay clips, never as the original final-rollout video.

First run one curated scene's two-arm replay serially as instrumentation smoke, included in the12. Require exact non-RGB initial pairing, RGB tolerance, 900 actions/901 observations, finite logs and post-hook tensor equality. Compare each replay's action trace and outcomes to original. If actions differ, retain both and diagnose determinism, bindings or hooks before using the replay causally. Do not relabel the original. A replay changing outcomes blocks the intervention stage until explained; instrumentation must not alter policy inputs.

Compute dynamic rotated-box rim, closing-axis/wall-normal alignment, pad opening and contact geometry on these cases. Compare with initial-orientation approximations and the original chronology; they may overturn a precise rim claim without invalidating a raw success count.

## 4. Stages and complete rollout arithmetic

All stage manifests/gates are frozen before reading their outcomes. Scientific failures remain in every denominator. Invalid infrastructure completions have a separate ledger and original artifacts. An infrastructure retry is allowed only for a diagnosed infrastructure defect with the same physical/model identity, at most one retry per job and at most12 retries overall. Budget retries separately. Pairing/replay nondeterminism is not solved by retrying until agreement.

| Stage | Matrix | New rollouts | Reuse / total analyzed |
|---|---|---:|---|
| A1: replay/instrument smoke and chronology | 6 curated scenes × original2 arms ×1 original model per arm | 12 | Original12 retained separately; no substitution |
| A2: crossed frozen PACT matrix | 24 scenes ×3 PACT checkpoints ×1 frozen variant | 48 if eligible | 24 original same-checkpoint outcomes +48 new =72 |
| B: mechanism | 12 selected matrix scenes ×3 seeds ×2 arms ×3 variants | 180 | 36 frozen PACT from A +36 new frozen ACT +72 uniform +72 candidate =216 |
| C: fresh development | 12 scenes per seed ×3 seeds ×2 arms ×3 variants | 216 | All fresh, no historical baseline reuse |
| D: one frozen final comparison | 50 scenes per seed ×3 seeds ×2 arms ×2 variants (frozen/candidate) | 600 | 300 frozen controls +300 candidate, paired by physical scene |
| **Maximum main path** | | **1,056** | A1 originals and A2 reused rows are additional reference observations, not new runs |

If an A2 original fails **binding/availability** eligibility, budget a replacement measurement separately: at most24 additional runs, bringing the bound to1,080. The failed/missing original remains documented. Measurement requirements for the common A2 matrix are the original H5/contact/command summaries; old rows cannot be reused as if they had chunks, dynamic cup rotation or RGB. The new48 and A1 instrumented cases support those extra analyses with explicit missingness. Reuse is zero for new instrumentation.

Training does not begin before A passes. B's first **12 candidate/control rollouts** (one of its12 scenes ×3 seeds ×2 arms ×2 trained variants) are the trained-policy smoke, included in B's144 trained rollouts. Validate them before releasing remaining B jobs. Frozen ACT controls are still included in the36 budget. No separate showcase smoke is added to the total.

### Stage A selection and causal gate

Use [cross24_scenes.json](../diagnostics_output/pact_place_v1010b_grasp_audit/cross24_scenes.json), selected independently of outcomes from all150 original scenes: within each nominal cell choose the minimum SHA256 of UTF8 `pact_place_v1010b_grasp_v1/cross24/` + full episode ID. The frozen selection has source blocks3103:7,3104:11,3105:6. Keep all three training checkpoints on all24 scenes, even when a source block differs. These are exposed diagnostic/regression scenes permanently; never call them untouched final tests.

Analyze checkpoint marginal success/pickup counts, scene marginal counts and matched scene-wise changes. A checkpoint tendency persisting on common scenes supports a checkpoint contribution; a changed ranking implicates scene interactions. Neither is a population significance test.

Before training require all of:

1. All72 matrix combinations valid, all new initial-state group audits pass, all12 instrumented replays accounted for; any replay action/outcome difference is explained and comparison eligibility explicitly resolved.
2. Across the72 PACT combinations, ≥12 exclusive pickup failures, spanning ≥2 checkpoints, and pickup failures are the largest exclusive failure category. This supplies a nontrivial target for the proposed intervention.
3. At least4 instrumented PACT/ACT trajectories have measurable approach/contact/closure/lift chronology. At least2 failed PACT acquisitions retain requested-versus-achieved depth discrepancy >5 mm with target contact and insufficient sustained engagement; confirm the discrepancy exceeds FK error and is not solely a rotated-rim classification artifact.
4. No demonstrated checkpoint/stats/input/decoder/controller defect. No evidence that logging itself changes policy inputs. If such a defect exists, stop this intervention and report the concrete correction needed.

If the matrix indicates acquisition is not the leading loss or instrumented geometry contradicts the proposed target, use the baseline fallback. If chunk analysis specifically demonstrates averaging of incompatible approaches, record it as a reason to commission a different one-change plan; do not switch this plan to a medoid/short-history experiment.

### Stage B matched mechanism test

Choose12 of the frozen24 scenes by the smallest SHA256 of `pact_place_v1010b_grasp_v1/mechanism12/` + scene ID, before any new results; run every seed on the same12. All24 nominal cells are already covered in A; B is a bounded mechanism screen, not a full coverage result.

Test all seeds and both arms, including successes and later failures. Compare candidate against both uniform63000 and frozen60000 using identical physical scenes. Record per-seed and pooled win/loss pairs for success, pickup failure and collision-free success. This controls both scene differences and extra optimizer updates.

Proceed only if all conditions hold, evaluated against **each control separately**:

- PACT candidate task successes increase by ≥3/36 and exclusive pickup failures decrease by ≥3/36.
- PACT seed3105 has no task-success loss; seeds3103/3104 lose at most1/12 each. PACT pooled CFTS does not fall; no seed loses >1 CFTS.
- ACT pooled task success and CFTS do not fall, and neither drops >1/12 in any ACT seed. The same sampling change is not allowed to weaken the ACT comparison arm silently.
- Among control pickup failures, at least3 candidate cases newly lift≥1 cm and show ≥1 s continuous bilateral engagement. Improved requested depth alone fails this test. If none of these cases has precise pad pose, use same-wall/bilateral identities plus actual lift and label the geometric limit.
- PACT-3103 lifted-without-placement failures increase by at most1/12; seed3105 strengths are explicitly retained in the paired table.
- For each arm pooled across seeds, hazard/clutter union frame count ≤110% of each control; no seed's frame avoidance falls by >0.5 percentage points. Contact duration for hazard/clutter cannot be hidden by class-overlap counting.

A three-case threshold is 8.33 percentage points at n36; it is a descriptive screen, not statistical proof. Failure stops the candidate. Do not repair a few selected cases and bypass the denominator.

### Stage C fresh development selection

Freeze36 fresh scenes before B outcomes:12 per training-seed block. For each block use four families ×three pendant poses, choosing one intrusion side by parity of block index+family index+pose index, with ordered lists from the original contract. This gives6 scenes per side/block and includes all24 cells across the three blocks. Derive actual task seeds independently with role `development_v1`; reject collisions with the entire frozen historical/new seed inventory and every data/smoke/diagnostic/final stream.

Evaluate frozen, uniform and candidate for both arms:216 fresh rollouts. Same gates as B for task/pickup/safety/per-seed outcomes, **except** mechanism-depth improvements are reported rather than required on a selected repair subset. Require PACT candidate ≥3/36 more successes and ≥3/36 fewer exclusive pickup failures against each control; ACT and PACT no pooled CFTS loss; per-seed protections and contact limits as in B. Also require median prior-inference close-query FK error on the unchanged40-demo validation diagnostic not to worsen >10% versus uniform in any PACT seed; do not choose checkpoints using this criterion.

This is the only fresh development selection. Pass freezes the candidate (all six exact model hashes, p0.25, interval±30, history100, controller/metrics). Fail ends the plan with baseline retained. Do not reuse these exposed scenes as a final test or run an alternate p/window.

### Stage D frozen final comparison

Before C outcomes, create150 **fresh, disjoint** final scenes:50 per training-seed block, two per each24 cells plus two extras. Select six distinct extra cells by global SHA256 ranking under `final_extras_v1`, assigning two to each block in order; verify side/pose balance and record any unavoidable allocation imbalance. Derive task seeds under independent role `final_v1`. Exclude all training/validation, A/B/C, prior final and historical seeds/identities. Freeze complete manifests and hashes; do not inspect their policy outcomes until C passes.

Evaluate **both frozen and candidate ACT/PACT** on the same150 scenes:600 rollouts, 50 ACT/PACT pairs per seed per variant. Uniform control is tested in B/C, not added to this frozen final matrix. Final improvement versus frozen includes the 3000 updates; the specific sampling attribution relies on the uniform controls already reported, and that limit must accompany final claims.

Final acceptance requires:

- PACT candidate ≥76/150 successes and ≥15 more successes than candidate ACT (original objective).
- Against the matched frozen PACT controls: candidate ≥9/150 more successes and ≥9/150 fewer exclusive pickup failures (6 percentage points); no PACT seed loses >1/50 task success, and seed3105 has no loss.
- PACT pooled CFTS does not decrease; no seed loses >1/50. ACT candidate pooled task success/CFTS do not decrease; no ACT seed loses >1/50 on either metric.
- Hazard/clutter union frames ≤110% of matched frozen counts per arm pooled, no seed frame-avoidance loss >0.5 percentage points. Report other-environment, mounted-fixture, cup and receptacle contacts separately.
- PACT-3103 lifted-without-placement failures increase by at most1/50. Report target interaction, maximum lift, stable engagement and empty-hand transport; success must not come from longer horizons or extra idle time.

Report all raw counts, denominators and paired wins/losses per seed. A target miss remains a miss even if some mechanism cases improve. Omit confidence intervals and McNemar tests. No final-test tuning or selection of a different checkpoint, seed or cell subset.

## 5. Metrics and missing data

Use the unchanged final task predicate at control step900. CFTS is task success plus zero disallowed contacts under the historical predicate. Keep the exclusive pickup hierarchy and also raw touched-but-never-held. Preserve the difference between contact-only held, measured lift and successful placement.

Frame avoidance: `100 × (1 − count(hazard OR clutter)/recorded_physics_samples)`, actual denominator per run; count overlapping classes once. Contact-duration descriptor is sample count × measured median physics dt; additionally report timestamp-integrated contact duration with an explicit terminal-sample convention for new data. Both arms use the same convention. Whole-episode avoidance is separate.

For mechanism outcomes use: closure=first applied255 command; near-cup XY<60 mm; early window observations0…min(close+60,900) with lift<10 mm; null if no eligible sample. Requested TCP FK uses command k and the correctly rotated local mount offset. Achieved TCP is observation k, and next-step tracking separately k+1. Rim depth using dynamic rotation is available only where instrumented; report the original translation-only reference on all rows for comparability. Mark unavailable force/pad geometry as null.

Continuous engagement=simultaneous bilateral pad contact lasting≥1.0 s; same-wall primitive tracked separately, including which wall. Do not join discontinuous runs or sum alternating pads. Record continuous contact on a fixed wall as well as the historical any-same-wall mask. Lift≥1 cm uses actual cup translation from its settled start; also report sustained lift≥1 cm for15 control observations. Empty-hand transport means TCP enters within100 mm XY of tray with no measured≥1 cm lift; report that definition separately from the historical narrower no-touch descriptor. Post-lift failure is a final failure with lift≥1 cm, split by tray support.

No-close, no-touch, no-eligible-depth and absent chunks/rotations are explicit states. Never substitute0 for missing geometry. Do not impute outcomes for crashed jobs. Missing required endpoints block completion rather than shrinking scientific denominators.

## 6. Budget, storage and bounded schedule

Freeze T0 in `B/contract.json`. Set **hard deadline T0+48 hours**, stop launching work whose measured completion would extend beyondT0+47h, and reserve the last hour for verification/closure. This replaces the completed experiment's expired absolute deadline. Planning budgets: implementation/test ≤4h, A≤3h, training≤3h, B≤6h, C≤7h, D≤20h, reporting1h:44h, leaving4h for startup/retry variance. Each stage re-estimates from actual throughput; no deadline is a guaranteed result.

Measured recent per-rollout duration is median1041.9 s in seed3103; pooled median1084.9 s and p90 1236.8 s. Start with **12 workers**, estimate `ceil(new_runs/12)×1085s`, and use1237s plus20% startup/drain reserve for conservative planning:

| Stage | New runs | 12-worker median estimate | Conservative estimate |
|---|---:|---:|---:|
| A | 60 | 1.51 h | 2.06 h |
| B | 180 | 4.52 h | 6.19 h |
| C | 216 | 5.42 h | 7.42 h |
| D | 600 | 15.07 h | 20.62 h |
| All rollout stages | 1056 | 26.52 h | 36.29 h |

Sequential smoke and actual resource pressure can increase these estimates. The per-stage targets above are launch-planning targets; the48h hard boundary and observed remaining-time gate govern. Twelve possible infrastructure retries add about0.34h conservatively, plus actual lost failed-run time. Twenty-four ineligible historical reuses add about0.69h. Do not assume linear gains from more workers or launch14 by default.

Recent 30,000-update segments took approximately4201–4234s ACT and4456–4481s PACT. Twelve3000-update branches cost about1.45h serial extrapolated; budget3h including loading/saving/validation. These are short continuations, **not six fresh60k retrains**.

At audit time the A10 had23,028 MiB VRAM,15.36 CPU cores of quota,183,497,654,272 bytes memory limit, pids.max3840 and about52 GiB disk free. Recheck actual availability at execution; historical OOM counters alone are not current errors. Preserve the existing thread caps before imports, including PyTorch intra/inter-op1. Sample resources every minute. After three consecutive samples >85%VRAM, >80%cgroup RAM or >75%PID limit, drain12→10; immediate OOM/thread failure pauses launching and records/reduces capacity. Persistent pressure at10 stops for diagnosis. High utilization alone is not an error. Recompute projected deadline on every change.

Historical retained rollouts average approximately48.23 MiB including initial state and telemetry;1056 new rollouts would need about49.7 GiB, before models/reserve. That **does not fit** the observed disk budget. Use a new-run-only **lossless H5 storage writer** with gzip4+shuffle for numeric arrays. A measured representative repack retained all437 datasets' shapes/dtypes/exact bytes while reducing trajectory size48,046,379→32,084,971 bytes in2.15s; [storage_probe.json](../diagnostics_output/pact_place_v1010b_grasp_audit/storage_probe.json) records this. Keep strings/scalars exact and all attributes. This is a storage change, not observation/sensor thinning. Validate full numerical equality before publishing hashes and write result/ledger hashes against the final repacked artifact. Do not repack or delete historical trajectories.

A six-case outcome-independent lossless probe measured32.32 MiB mean and32.60 MiB maximum including standard telemetry/actions/initial state; see `A/storage_probe_six.json`. Budget **34 MiB per new standard rollout** until measured on12 new runs, approximately35.1 GiB at1056 runs, plus≤5 GiB final model files, ≤3 GiB concurrent temporary training bundles, and≤1 GiB instrumented extras. Not all temporary bundles coexist with final rollout accumulation: verify peak usage stage-by-stage. The10 GiB reserve is mandatory. Before each stage require `free_bytes − projected_stage_bytes − remaining_required_artifacts >= 10 GiB`. If measured compression/storage cannot fit the remaining full matrix, stop and report capacity needed; do not silently omit raw data, delete old results, or reduce final counts. Re-estimate from observed mean and maximum artifact sizes, not only the one-file probe. Maximum instrument storage is100 MiB per curated run; pause instrumentation on overflow and preserve the partial recording.

## 7. Execution, resume and rollback

The commands below are the interfaces the implementation agent must provide and test; they are **not pre-existing commands**. Implement them before execution. Use `/root/act_retrain_venv/bin/python`, installed NumPy/SciPy/h5py/MuJoCo/PyTorch/OpenCV/imageio, existing cached robot/object assets and frozen encoder. Do not upgrade dependencies or download substitute assets. Record interpreter/library/GPU versions.

```bash
cd /root/prox_learning_pact_remediation
/root/act_retrain_venv/bin/python scripts/pact_v1010b_run.py prepare --output diagnostics_output/pact_place_v1010b_grasp_v1 --workers 12 --hours 48
/root/act_retrain_venv/bin/python -m pytest -q tests/test_pact_v1010b.py
/root/act_retrain_venv/bin/python scripts/pact_v1010b_run.py supervise --output diagnostics_output/pact_place_v1010b_grasp_v1
/root/act_retrain_venv/bin/python scripts/pact_v1010b_run.py verify --output diagnostics_output/pact_place_v1010b_grasp_v1
```

`supervise` follows A→training→B→C→D, refusing unmet gates. Stage subcommands permit scoped resume but may not bypass prerequisites. Keep the supervising parent alive locally, with minute resource sampling and stage/low-frequency progress records; the separate monitoring task remains disabled. Poll children and obtain their real exit codes promptly. Do not infer success from a log string, orphaned PID, output-file existence or a printed counter.

Required artifacts: frozen `contract.json`; source/checkpoint/encoder/stats hash manifest; `scene_manifests/`; `training_labels.json`; training sample-stream hashes; stage schedules; jobs and parent-observed `exit_receipt.json`; append-only valid/invalid ledgers; original/full/repacked-file provenance; initial observations/group audits; raw H5/contacts/actions; predicted chunks and synchronized instrumented traces/clips; training update logs/final checkpoint hashes; stage metrics/gates; `EVAL.md`; `parent_closure.json`.

Resume only if contract, code, assets, models, stats, split and job identities match, every completed row has a matching observed zero exit/result hash, and pending work is exactly the unsatisfied schedule. Preserve partial attempts under unique attempt directories. A scientifically valid failed rollout is complete and never replaced. An implementation repair must produce a versioned amendment naming its effect, preserve failures/old contracts, and rerun only checks/data invalidated by the repair; a physical or policy change invalidates scientific comparisons and requires a new bounded plan.

Rollback means select original `W/checkpoints/{act,pact}_seed{seed}/policy_update_60000.ckpt` and the original history100 worker recipe, stop B-owned launches, drain healthy workers, and leave all results readable. Terminate only verified experiment-owned orphan processes identified by job identity and process start time. Never kill unrelated workloads. Never alter/delete W, G, original reports or the user's existing dirty files.

Closure must say one of: `FINAL_ACCEPTED`, `FINAL_TARGET_MISSED`, `STOPPED_AT_GATE_<stage>`, or `INCOMPLETE_INFRASTRUCTURE_OR_BUDGET`, with exact completed/expected counts and the actual reason. A scientifically justified gate stop is completion of this bounded experiment, not evidence of a successful fix. Never claim the600-rollout final completed if a prerequisite stopped the plan.

## 8. Required tests and independent verification

Before any simulator jobs:

1. Frame transforms: wxyz/xyzw conversion, rotated0.35m mount, FK agreement with recorded states, correct command-k/observation-k+1 indexing; reject incorrectly world-added mount offsets.
2. Data invariance: all eight forward-aligned labels; no extra loader shift; p0 sample tensor/RNG equivalence; candidate-only start distribution; unchanged padding, stats, frozen encoder and train/validation membership; no diagnostic/final seed leakage. Verify exact ACT/PACT sampled-start matching per seed/variant.
3. Decoder/aggregation: active 32-D override; chunk age indexing/expiry; valid zeros retained; normalized weights; exact127.5 boundary; no current/future chunk misindexing; post-hook actions identical; padding prediction is unused.
4. Contact/geometry accounting: overlapping hazard/clutter deduplication; distinguish robot–cup, robot–clutter and cup–clutter; discontinuous/alternating pads cannot become continuous bilateral engagement; fixed-wall versus any-wall durations distinguished; lift alone and held alone cannot produce final success.
5. False grasp signals: include positive tracking-gap successful historical examples and held-without-lift counterexamples. No runtime gate may be introduced as a convenience. No-close and no-eligible-depth produce explicit states. Instrument cap≤300 and RGB cap≤150 must preserve the900-step policy horizon and never change controls.
6. Pairing: changing only model provenance is permitted; physical seed/row/model-array changes rejected; RGB tolerance boundary accepted/rejected exactly; cross-checkpoint changes cannot resample the scene. Test the first six cross-checkpoint initial states before filling A2.
7. Completion/resume: missing H5 fields, wrong hashes, nonzero or unobserved exits, duplicate IDs, absent pair member and shortened trajectory all block completion. Recovering an infra crash preserves the original attempt and cannot count twice.
8. Continuation: original60000 model/optimizer fork invariant, finite loss, exact3000 new updates, resume equivalence over an interrupted short fixture, p0 control semantics, no accidental write to W, final strict checkpoint reload and correct model/stat binding.
9. Storage: exact dataset names/shapes/dtypes/bytes and group/dataset attributes before/after new-run repack; original file immutable; publish final hashes only after verified repack; enforce budget/reserve and interrupted-publication recovery.

After each passing stage run only checks newly required by changes or unresolved failures. Finish with an independent raw recomputation of all included outcomes and group pairings, audit report/source hashes, and observed process closure. The execution subagent should report findings back to the parent at each gate and immediately on a blocking mismatch. The parent should challenge any claimed mechanism or completion that the raw evidence does not support.
