# V10.10: wrist-only ACT/PACT training and three-seed evaluation

## 1. Objective and fixed boundaries

Target **PACT task success >50%, with at least a 10-percentage-point advantage over ACT**, averaged across training seeds **3103, 3104 and 3105**. Report safety separately.

Final evaluation is **50 paired instances per seed: 150 instances per arm, 300 rollouts total**. Meeting the target requires PACT to succeed on at least **76/150**, with at least **15 more successes than ACT**. These are experimental targets, not guaranteed outcomes or claims of statistical significance.

- Work exclusively in `/root/prox_learning_pact_remediation`; never change to its parent.
- Create an isolated experiment under `diagnostics_output/pact_place_v1010_wrist288_s3_v1/`, containing manifests, raw and converted data, checkpoints, logs and evaluations.
- Preserve historical code, datasets and results. Implement new adapters rather than modifying historical experiment contracts.
- Keep every `authorizes_*` flag false. Do not represent this experiment as a previously failed qualification passing.
- Use **wrist RGB only** and **chunk-100 training only**. No table camera, chunk-25 experiment, gripper retuning, recovery-data round or broad hyperparameter sweep.
- Target completion in **32–36 hours**, with a hard cutoff of **September 8, 2026, 00:00 UTC**.

Freeze configuration, code/asset hashes, seed derivation, selection rules and output locations before collection. Resume only when these bindings match; file existence alone does not prove stage completion.

## 2. Environment, collection and corrected supervision

### Preserve the actual V10.10 environment

Use `PactPlaceCorridorV1010FourObjectSampler` from `molmo_spaces.tasks.enclosure_reach`, on `experiment/pact-vs-act-remediation-v2`, inspected at commit `70dedc07f34ed7f8335aed7f694ddef7ef823d3d`.

Verify the historical V10.10 scene/XML bindings, expert, task-success predicate, target distribution and four active clutter slots **01, 03, 04 and 06**. Do not substitute V2, V10.11c or the alternative environment port.

Preserve the wrist-only camera system, original image preprocessing, canonical 40-sensor order and frozen proximity encoder:

`6fd2dd037e3236b5b6bf7fce8cb2709ead0cf52adcbbe9cbad1061efc2fe3206`

### Collect 288 fresh strict-clean demonstrations

- Cover all **24 family × intrusion-side × pendant-pose cells**, accepting **12 episodes per cell**.
- Derive the split from the accepted ledger: deterministically rank episodes within each cell, assigning **2 validation and 10 training**. Totals: **48 validation / 240 training**.
- Use fresh, domain-separated collection, split, smoke, development, final-evaluation and retry streams. Derive seeds deterministically from the experiment namespace, role, cell, ordinal and retry index; reject collisions with recorded historical and new task seeds before use.
- Target **14 collection rollout workers**, with fallback to **12, then 10** under the resource rules below. Allow at most one attempt in flight per cell.
- Record rejected, failed and accepted attempts; do not silently exceed quotas or relax strict-clean acceptance.
- Preserve HDF5, required wrist RGB recordings and sensor provenance. Avoid redundant trajectory JSON and unnecessary diagnostic videos.
- Verify accepted episodes from raw artifacts and ledger identities before conversion. Resolve row directories explicitly; do not assume a ledger contains `row_dir`.

### Correct conversion—not the upstream recorder

**Decision: change the new dataset’s labels.** MolmoSpaces explicitly documents that state `i` corresponds to action `i+1`; its last-action recording convention is intentional. Original ACT simulation data instead pairs observations with the action subsequently applied. The conversion boundary must translate between these conventions. [MolmoSpaces data format](https://allenai.github.io/molmospaces/data_format/), [original ACT recorder](https://github.com/tonyzhaozh/act/blob/main/record_sim_episodes.py)

For each valid transition:

`input = observation[t]`  
`target = raw commanded_action[t+1]`

- Keep wrist RGB, proprioception and proximity aligned at `t`.
- Exclude the initial dummy action and terminal status-only transitions. Inspect actual fields; do not blindly discard a fixed number of trailing rows.
- Validate all eight action components and their units, without changing the **127.5 gripper threshold**.
- Mark converted data as simulation data; verify the ACT loader introduces no second shift.
- Compute normalization from the **240 training episodes only**, shared identically by both arms.
- Encode proximity into `(T, 40, 32)` tokens using the frozen encoder.

This corrects a documented alignment mismatch; its effect on success must still be measured.

## 3. Training parameters and development gate

Train seed **3103 first**, for both arms.

| Parameter | Setting |
|---|---|
| Training budget | **60,000 actual optimizer updates per model** |
| Batch size | 8 |
| Epoch count | 2,000: 240 training episodes → 30 updates/epoch |
| Prediction chunk | 100 |
| Learning rate | `1e-5`, including backbone |
| KL weight | 10 |
| Transformer | Hidden 512, feedforward 3200, 7 encoder / 7 decoder layers, 8 heads |
| RGB input | Wrist only, 240×320, existing ResNet18 preprocessing |
| State / action dimensions | 9 / 8 |
| PACT input | 40 frozen 32-dimensional proximity tokens |
| DataLoader workers | 4 per trainer |
| Training padding horizon | `max(635, converted_T_max + 8)` |

Preserve remaining V10.10 optimizer, loss and architecture settings. Verify matching episode sampling and training parameters within each ACT/PACT seed pair; only the proximity-specific inputs/model components may differ.

- Use the **60,000-update checkpoint** for evaluation, not a test-selected checkpoint.
- Retain the 30,000-update snapshot for provenance, without an additional rollout sweep.
- Save atomic optimizer/RNG resume bundles every **3,000 updates** and restart training processes at the 30,000-update boundary.
- Confirm completed update counts, finite losses, strict checkpoint reload, shared statistics and actual PACT proximity consumption.

### Development experiment

After infrastructure smoke tests, evaluate the seed-3103 checkpoints on **24 fresh paired development instances**, one per cell:

- Chunk-100 predictions with averaging history **100**.
- The same checkpoints with averaging history **10**.

This is **96 development rollouts**, targeting **14 workers**, with fallback to 12 or 10. Both modes query the policy every control step. Preserve age weighting `exp(-0.01 × age)` and change only the maximum number of overlapping predictions retained.

Choose **one common averaging history for both arms** using:

1. Highest combined ACT+PACT task-success count.
2. Then highest combined collision-free-success count.
3. Then lowest combined hazard-contact frame count.
4. Exact tie: retain history 100.

Do **not** select the history that maximizes PACT’s advantage.

Continue only if the selected configuration achieves **PACT ≥13/24 successes and at least 3 more successes than ACT**. This is a small development-screening gate, not proof of the final target.

If it misses, pause, report raw outcomes and diagnose the failure stage. Do not automatically launch the other seeds, extend training or inspect the final test set. If it passes, freeze the inference recipe and train both arms for seeds **3104 and 3105** to the same 60,000-update budget.

## 4. Final paired evaluation and reporting

Freeze the final instance manifest independently of development outcomes.

- Evaluate **50 instances for each training seed**, with no instance shared across seed blocks or development/data streams.
- Give each seed two instances per cell, plus two predeclared extras. Distribute the six extras across distinct cells, balancing sides and pendant poses globally.
- ACT and PACT must use the same physical instance and sampling/retry bindings within each pair.
- Target **14 rollout workers**, falling back to **12, then 10** if necessary.
- Use the selected common averaging history, a **900-step horizon**, no action noise and `end_on_success=False`.
- Use `--save-trajectory --h5-only`; retain the initial observations needed for pairing verification.
- Never replace a valid policy failure. Infrastructure retries must preserve the original artifacts and instance identity; do not repeatedly rerun until a favorable outcome appears.

Apply the agreed **new-run-only** pairing contract:

- Exact non-RGB initial state, configuration, shapes and dtypes.
- Initial wrist RGB: maximum absolute difference **≤2 uint8 levels**, with at most **0.1% of channel values changed**.
- Retain both images and the full difference audit. A violation blocks paired analysis; it is not silently waived.

Recompute results from raw trajectories after confirming field names and semantics. Verify **300 unique, valid rollout completions** against worker exit receipts, ledger bindings and trajectory lengths—not a printed completion counter.

Write the experiment report to its own `EVAL.md`, with per-seed results before pooling:

- Task success and PACT−ACT difference.
- Collision-free success and collision-free rollout rate.
- Hazard-bar contact rate, total contact frames and mean frames per rollout.
- Per-object contact and stability for active slots **01, 03, 04 and 06**. Mark slots 08/09 as absent, not zero-contact observations.
- Target-touch, lift and placement-stage outcomes to distinguish non-interaction from successful safe behavior.
- Three-seed mean, pooled counts and paired ACT-only/PACT-only successes.

Omit Wilson intervals, McNemar tests and detailed gripper-status analysis, as requested. Report whether the target was met without concealing negative seeds or failed stages.

Include the sensor limitation: clutter is effectively invisible to the proximity skin in the prior resolvability audit—inbound vessel `max_w_perp_m = 0.000` in **7 of 8 variants**, with 40 sensors covering **link1–link6 only**. A PACT advantage is **not evidence that it senses the clutter**.

## 5. Resources, hourly monitoring, tests and deadline

The previously inspected machine provides an **A10 with approximately 23 GiB VRAM**, a **15.36-core CPU quota**, approximately **171 GiB usable RAM**, and approximately **113 GiB free disk**. Recheck these at launch rather than relying on host-wide specifications.

For every worker, set before numerical-library imports:

`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`, `VECLIB_MAXIMUM_THREADS=1`.

Also cap `OMP_THREAD_LIMIT`, `BLIS_NUM_THREADS`, `RAYON_NUM_THREADS`, and PyTorch intra/inter-op threads at 1.

### Rollout concurrency: 14 → 12 → 10

- Start collection and evaluation with a **target concurrency of 14**, staggering process startup.
- A lightweight supervisor samples resources every minute. Reduce concurrency one level after three consecutive samples above **85% VRAM**, **80% usable RAM**, or **75% of the cgroup PID/thread limit**.
- OOM or thread-creation errors trigger an immediate launch pause and reduction, preserving failed-job evidence.
- Stop issuing excess jobs and retire idle workers; let healthy in-flight rollouts finish. Never drop, replace or renumber scientific instances to resize the pool.
- High CPU/GPU utilization alone is not a failure. Keep productive utilization; investigate falling throughput or resource pressure.
- If pressure persists at 10 workers, pause and diagnose rather than silently reducing further.
- Record every concurrency change and its reason. Maintain a **10 GiB disk reserve**; crossing it pauses new work.

### Required task: `hourly_run_check`

Start this monitoring task when execution begins and repeat **every hour until completion or safe shutdown**. Each check must:

- Verify the supervisor and expected workers are alive; inspect actual exit codes and new errors.
- Count accepted collection episodes by cell, training updates by seed/arm, and valid completed rollouts against their ledgers.
- Check GPU/VRAM, cgroup CPU/RAM/PID usage, disk space and experiment-owned orphan workers.
- Confirm progress has advanced; investigate stalls without treating an unchanged summary counter as proof of failure or success.
- Recalculate remaining ETA from measured throughput.
- Append a timestamped monitoring record and provide a concise user update, including any corrective action.

Worker crashes should be detected immediately by the supervisor, **not left until the hourly check**. Full progress verification and user updates remain hourly.

### Training throughput and scheduling

- Benchmark the first **900 updates per pilot arm serially**, then the next **900 per arm concurrently**, retaining those updates and resuming at epoch boundaries.
- Keep two concurrent trainers only if aggregate throughput improves by **at least 20%**, peak VRAM stays below **85%**, RAM below **80%**, and neither process errors. Otherwise resume serial training.
- Do not overlap GPU training with rollout evaluation.
- Historical timings suggested approximately **35–40 hours with the earlier 12-worker schedule**, before demonstrated training-concurrency gains. Measure the benefit of 14 workers; do not assume linear speedup or promise the deadline.

### Required tests and safe completion

Before expensive stages, test environment/encoder bindings; ledger quotas and split/seed disjointness; forward-action alignment and terminal handling; train-only normalization; matching arm settings; resume equivalence and optimizer-step accounting; both averaging implementations; pairing-tolerance boundaries; missing-field failures; and false-completion detection.

Run **four disjoint smoke pairs** with valid raw artifacts before development evaluation.

Reserve the final hour for validation and reporting. If measured remaining work no longer fits before **September 8, 00:00 UTC**, checkpoint safely and report incomplete stages and actual error/output. Do not silently reduce episodes, updates, seeds or evaluation size.

On completion or safe shutdown, clean up only verified experiment-owned orphaned `multiprocessing.spawn` workers with `PPID 1`, preserving unrelated processes. Finish with the final hourly-check record, artifact locations, completed-stage counts and an honest target-met/target-missed/incomplete verdict.
