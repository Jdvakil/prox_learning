# CAUSAL_PARKED_SKIN_REFERENCE_V1 — final decision

## Executive summary

The parked proximity field **is learnable** from deployable inputs, and by a wide margin.
The frozen model cuts the SafetyHead-space differential MAE from the trivial baseline's
0.042062 to **0.011337** on offline test — a **73.0% reduction** against a 25% gate — with a
median direction cosine of **0.999** against a 0.75 gate, 94–98% recall on oracle-active
frames, and **zero** output-constraint violations anywhere.

One gate fails. Using the threshold frozen on calibration, the offline-test oracle-zero
false-positive rate is 2.15% on seed 0 against a **≤2%** requirement (seeds 1 and 2 give
1.42% and 1.10%). Every other technical and generalization gate passes. Under the handoff's
rubric that single offline-test failure forces `PARKED_REFERENCE_MODEL_OVERFIT`.

**That token is the rubric's, and it fits the letter rather than the phenomenon.** This is
not overfitting in the usual sense, and the report should not be read as saying it is:

| | validation | calibration | offline test |
|---|---:|---:|---:|
| Differential MAE (seed 0) | 0.020782 | 0.009914 | **0.009087** |

Offline-test error is *less than half* the validation error. There is no validation-to-test
degradation to overfit with. What actually fails is **threshold transfer**: the activation
threshold is fixed at the 99th percentile of the calibration partition's oracle-zero norms,
which yields 1.01% on calibration by construction, 2.15% on offline test, and 2.61% on
validation. The calibration partition is 8 episodes / 3,821 frames, and its oracle-active
prevalence (14.4%) is the lowest of the four partitions. A quantile estimated on that slice
does not transfer tightly enough to hold a 2% ceiling elsewhere.

A second finding is independently important: **temporal history does not help.** Four causal
frames are *worse* than the current frame alone (offline MAE 0.012809 vs 0.011337, a −13.0%
"gain"), and less stable across seeds (CV 0.381 vs 0.177). The hazard is a *parked* obstacle
— static by construction — so its signature is fully present in frame `t`, and the extra
frames add parameters and variance without information. Per the handoff's alternative
branch, the simpler model is frozen and every remaining gate was recomputed against it
rather than inherited from the discarded one.

Proximity itself is essential: the state-only control lands at 0.048446, **worse than
predicting no obstacle at all**.

- Root branch `train/causal-parked-skin-reference-v1`, from `9c4703a`
- ACT `91fc42a`, MolmoSpaces `678f2eb` — both untouched, clean working trees
- Decision JSON sha256 `759a01f5e626387854f877e8fcdb1b202d930c14fc476fd5b4136b6cc8de55d8`

## Starting and final commits

| Repo | Expected | Observed | Modified |
|---|---|---|---|
| root | `9c4703a` | `9c4703a` → branch `train/causal-parked-skin-reference-v1` | code added only |
| ACT | `91fc42a` | `91fc42a` | no |
| MolmoSpaces | `678f2eb` | `678f2eb` | no |

Two pre-existing stashes on root were left untouched. No commits were pushed.

## Dataset and partition verification

The frozen dataset was re-hashed from the manifest and matches the freeze exactly: tree
sha256 `1fb68b3cdee9755881579a187c84992782bfdd5f1c65aff6a3c39ebd11b8dba5` over 364 files,
all mode 444. Nothing was written to it; the loader opens every file `"r"` and a test parses
the loader's AST to assert that.

Partition independence was checked on **five** identity keys — episode id, source H5 hash,
manifest-row hash, trajectory id, and initial-state hash — with **0 crossings** on every one.
This mattered more than usual here: the same episode identity is reused across all four
source distributions by design (64 episodes appear in 4 files, 36 in 3), so per-file
uniqueness would have proved nothing. What had to hold is that every copy of an episode
lands in one partition, and it does.

| Partition | Trajectories | Frames | Episodes | Oracle-active | Hazard +/− |
|---|---:|---:|---:|---:|---|
| `reference_train` | 256 | 43,519 | 64 | 11,308 (26.0%) | 32,727 / 10,792 |
| `reference_validation` | 24 | 3,910 | 8 | 810 (20.7%) | 2,987 / 923 |
| `reference_calibration` | 24 | 3,821 | 8 | 549 (14.4%) | 2,897 / 924 |
| `offline_reference_test` | 60 | 9,543 | 20 | 1,744 (18.3%) | 7,157 / 2,386 |

**Coverage limitation, stated up front.** `LEARNER_INDUCED_ON_POLICY` exists **only in
`reference_train`** (64 trajectories). Validation, calibration and offline test contain no
learner-induced rows at all. The handoff asks for per-source-mode results including that
mode and for a gate on it; that gate is therefore *unevaluable*, not passed. Whether the
model generalizes to learner-induced states cannot be answered from this frozen dataset, and
no rearrangement of it would answer the question without changing the partitions, which is
forbidden and would have been wrong anyway.

## Causal input contract

Contract sha256 `cc87a773b2a06d7ca6bd4be5a0783be1d576887c74efa3e13ea94dd6d92d7704`. All 23
stored fields were inventoried from the files themselves; live availability was read out of
the live evaluator's AST rather than assumed.

**Model inputs (7 fields, all deployable and all live):** `current_closeness` (40×8×8,
float32, closeness in [0,1]), `current_valid_mask` (40×8×8, bool), `qpos` (9; rad + m),
`qvel` (9; rad/s + m/s), `nominal_action` (8), `gripper_state` (2), `gripper_command` (1) —
a 29-wide state vector plus the field and its mask.

Live availability was proven, not asserted: the evaluator's `runtime` dict supplies `qpos`,
`qvel`, `nominal_action`, `gripper_state`, `gripper_command`, and `render_current_skin()`
produces the full 40×8×8 field upstream of it. An earlier version of the audit matched "the
largest dict containing qpos", which found the per-frame *diagnostic record* — that dict
legitimately logs privileged fields, and would have certified `privileged_oracle_dq` as
live-available. Targeting the assignment to `runtime` specifically fixed it.

**Excluded although live:** `episode_step` and `control_timestamp`. Both are trivially
available at runtime, and both were deliberately withheld: every on-policy row runs a fixed
200-step horizon, so a step index lets the model learn *when* the hazard tends to be near
instead of reading it from the field.

**Prohibited and confirmed unused:** parked closeness/validity, removable closeness, changed
mask, parked head, oracle differential, oracle activity, teacher outputs, and `current_head`.
Zero prohibited inputs were used; zero inputs were non-live. No RGB was added.

## Model architecture

`CausalParkedSkinReferenceV1` — shared per-sensor spatiotemporal encoder (Conv 2F→32→64 over
the 8×8 tile, closeness and validity as explicit channels), mean+max pooling to a token,
learned sensor-ID and link-ID embeddings, a 29→hidden state embedding, two cross-sensor
transformer blocks (d=192, 4 heads, pre-norm), and a shared per-sensor spatial decoder
producing two logit maps.

| Variant | Parameters | Budget |
|---|---:|---|
| `FULL_CAUSAL` (4 frames) | 840,162 | < 3,000,000 |
| `CURRENT_FRAME_ONLY` (1 frame) — **frozen primary** | 838,434 | < 3,000,000 |
| `QPOS_ONLY` (state only) | 762,818 | < 3,000,000 |

Output parameterization, identical across every learned variant:

```
changed_probability = sigmoid(mask_logits)
delta_magnitude     = current_closeness * sigmoid(magnitude_logits)
predicted_delta     = changed_probability * delta_magnitude
predicted_parked    = current_closeness - predicted_delta
```

Both factors lie in [0,1] and the magnitude is scaled by the current field, so
`0 ≤ predicted_parked ≤ current_closeness ≤ 1` holds **by construction**. Nothing is clipped
afterwards — a clamp would hide violations and the gate that counts them would be measuring
the clamp. A test drives the output bias to ±60 and confirms the bounds still hold.

The two-headed factorization is deliberate: a single sigmoid conflates "this pixel changes"
with "by how much", and at a changed-pixel prevalence of 8e-4 the magnitude head would spend
nearly all its gradient on pixels that never move.

## Frozen SafetyHead path

The shipped `SafetyHead.__call__` takes numpy depths and `SafetyCVAE.act` is decorated
`@torch.no_grad()`, so neither can carry a gradient into the parked field. The decoder is
called directly at `z = 0`, exactly as `act` does, with `label_scale = 11.359346389770508`
applied **once** at the end. The dataset stores closeness and `featurize` maps depth to that
same closeness, so no conversion happens — converting to depth and back would apply the
transform twice.

Verified: head parameters have `requires_grad=False` and receive no gradient; head weights
are bitwise unchanged after backward passes; a fingerprint of the decoder parameters is
identical before and after all nine training runs; `train(True)` cannot take it out of eval;
and applying the scale twice is explicitly distinguishable from applying it once. Gradients
do reach the model through the head, which is the point.

Reconstruction check: `head(current) − head(parked)` reproduces the stored `oracle_dq`, and
the stored `current_head − parked_head` matches `oracle_dq` to **exactly 0.0**.

## Losses and sampling

Six components. Weights were selected on validation only.

| Component | Form | Weight (frozen) |
|---|---|---:|
| changed-mask | focal BCE, class-balanced, valid pixels only | 2.0 |
| active-delta | Huber on `current − parked` over changed pixels | 30.0 |
| all-valid field | L1 over valid target pixels | 5.0 |
| frozen-head consistency | Huber on the 7-D differential | 20.0 |
| oracle-zero quiet | MSE penalty on predicted differential, zero frames | 20.0 |
| temporal smoothness | causal, onset-gated | 0.0 (off) |

`focal_gamma` 2.0, `huber_beta` 0.01, `mask_pos_weight_cap` 32.

**All 46,382 oracle-zero frames stay in the dataset and remain addressable.** Training uses a
deterministic stratified sampler over `{active, zero} × {hazard present, absent} × source
mode`, drawing 50% oracle-active frames per batch against a natural prevalence of 26.0%.
Nothing is removed or subsampled, and **evaluation always runs the natural, unmodified
partition distribution** — all reported metrics are population-weighted, never
sampler-weighted. The sampler is deterministic given `(seed, epoch)`; a test asserts two
instances produce identical batches and that different seeds and epochs produce different
ones.

Note the `active_hazard_absent` stratum is empty by construction: hazard-absent frames are
exact zero controls, so they are never oracle-active.

## Bounded validation-only selection

Six seed-0 candidates, varying only loss weights, block count and hidden width. Input
fields, causal history length, partitions, target definition, output constraints, the
SafetyHead, the checkpoint rule, `max_epochs` and `patience` were identical across all six.
**The selection script never loads the offline-test partition at all** — a test parses its
AST to confirm the lock is structural rather than a matter of discipline.

| Candidate | Hidden | Blocks | Params | Val head MAE | Best epoch |
|---|---:|---:|---:|---:|---:|
| **c3_active_heavy** (selected) | 192 | 2 | 840,162 | **0.019381** | 39 |
| c2_quiet_heavy | 192 | 2 | 840,162 | 0.019994 | 51 |
| c4_one_block | 192 | 1 | 543,138 | 0.020122 | 44 |
| c5_wide | 256 | 2 | 1,379,106 | 0.021185 | 34 |
| c6_narrow | 128 | 2 | 440,482 | 0.022313 | 32 |
| c1_balanced | 192 | 2 | 840,162 | 0.025964 | 16 |

Selected config hash `cab280dff314ef17e5e544408429aa714c73a76015ea1b6a7f51a406c71a1e08`.
ZERO_DIFFERENTIAL validation MAE for reference: 0.040890.

**Disclosed pre-candidate diagnostics.** Four short runs preceded the candidate budget and
are recorded in `selection.json` rather than hidden. They existed to find a defect, not to
pick a model: the BCE positive weight was uncapped at `negatives/positives ≈ 1200`, which
drove the mask head to fire on essentially every pixel; the frozen head then turned a ~4e-3
per-pixel error into a ~0.7 differential error, giving a validation MAE of 0.2739 — far
*worse* than predicting nothing. Capping the weight at 32 produced the first result that
beat the baseline. A fourth run established that the 12-epoch probes had been schedule-
limited rather than weight-limited.

## Calibration

Threshold on ‖predicted_oracle_dq‖ at the 99th percentile of the **calibration** partition's
oracle-zero norms, targeting ≤1% false activation. Frozen before offline test was opened;
`offline_test_used: false` is recorded per run. A test confirms the threshold is computed
from oracle-zero frames only and is unchanged when active-frame magnitudes are scaled 500×.

| Seed | Threshold | Calibration FP | Validation FP | **Offline-test FP** |
|---|---:|---:|---:|---:|
| 0 | 0.02819 | 1.01% | 2.61% | **2.15%** ✗ |
| 1 | 0.05541 | 1.01% | 1.35% | **1.42%** ✓ |
| 2 | 0.03487 | 1.01% | 1.39% | **1.10%** ✓ |

The thresholds themselves vary by a coefficient of variation of **0.72** across seeds
(0.0313 / 0.1394 / 0.0368) — the flagged `calibration_instability` hypothesis. Seed 1 needs a
4× higher threshold than seed 0 to hit the same calibration false-positive rate, which is
what a 3,821-frame quantile estimate buys.

## Three-seed results

Seeds 0, 1, 2; max 100 epochs, patience 12 validation epochs; best-on-validation checkpoint
selection; atomic publication with full optimizer/scheduler/RNG resume bundles. Offline test
was loaded **once**, after all nine runs and all thresholds were final.

### SafetyHead-space, offline test

| Model | seed 0 | seed 1 | seed 2 | mean | CV |
|---|---:|---:|---:|---:|---:|
| **CURRENT_FRAME_ONLY** (frozen) | 0.009087 | 0.013956 | 0.010968 | **0.011337** | 0.177 |
| FULL_CAUSAL | 0.008874 | 0.019692 | 0.009861 | 0.012809 | 0.381 |
| QPOS_ONLY | 0.049031 | 0.046296 | 0.050012 | 0.048446 | 0.034 |
| ZERO_DIFFERENTIAL | — | — | — | 0.042062 | — |
| privileged true-parked upper bound | — | — | — | 3.84e-08 | — |

The privileged upper bound is nonzero only by float32 rounding, since routing the true
parked field through the frozen head is exactly how the stored target was produced.

Frozen-primary detail (seed 0, offline test): per-joint MAE
`[0.0177, 0.0151, 0.0178, 0.0104, 0.0013, 0.0008, 0.0006]` — error concentrates in the
proximal three joints; arm-vector norm error 0.0225; predicted-vs-true norm correlation
0.934; active-frame RMS 0.213; oracle-zero RMS 0.0063; hazard-absent RMS 0.0084 against a
raw SafetyHead RMS of 2.176 on the same frames, a ratio of **0.0039** against a ≤0.25 gate.

### Pixel-space, offline test (frozen primary, seed 0)

| Metric | Value |
|---|---:|
| All-valid parked-closeness MAE | 9.34e-05 |
| Changed-pixel parked-closeness MAE | 0.0735 |
| Differential MAE | 9.34e-05 |
| Changed-mask precision / recall / F1 | 0.797 / 0.903 / 0.847 |
| Changed-mask AUPRC | 0.900 |
| Changed-pixel prevalence | 7.98e-04 |
| Prevalence-normalized AUPRC | **1128×** (gate ≥3×) |
| Constraint violations | **0** |
| Validity-mask agreement | **1.000** (see below) |

**On validity-mask agreement.** The `validity_mask_agreement` field recorded in
`final_training.json` (0.119) used a placeholder definition that measured "fraction of pixels
the model left unsaturated", which is a different quantity. The correct value is **1.000**,
and it is uninformative here: across all 24,430,080 offline-test pixels, `current_valid` and
`parked_valid` are both **entirely True** — there is not a single dead pixel — so the masks
are identical and any model inheriting current validity agrees trivially. The model cannot
turn a live pixel dead in any case, because that needs closeness above 0.99 and no stored
closeness exceeds 0.902. The definition has been corrected in `engine.py`; the frozen
training record was left as recorded rather than regenerated.

### By source mode, offline test (frozen primary, seed 0)

| Source mode | Frames | Active | Differential MAE | Median direction cosine |
|---|---:|---:|---:|---:|
| `ACT_ONLY_ON_POLICY` | 4,000 | 781 | 0.007285 | 0.9991 |
| `ORACLE_ON_POLICY` | 4,000 | 641 | 0.008801 | 0.9985 |
| `EXPERT_RECONSTRUCTED` | 1,543 | 322 | 0.014497 | 0.9996 |
| `LEARNER_INDUCED_ON_POLICY` | 0 | — | **not evaluable** | **not evaluable** |

Worst-over-best MAE ratio across the three available modes is 1.99, just under the 2.0 mark
that would flag distribution shift. Expert rows are the hardest, which is consistent with
their being open-loop replays rather than closed-loop rollouts.

### Per-sensor and per-link attribution

Field MAE is almost entirely confined to the distal links:

| Link | Sensors | Field MAE |
|---|---:|---:|
| `link5_back` | 6 | 4.75e-04 |
| `link6` | 6 | 3.06e-04 |
| `link3` | 5 | 4.4e-05 |
| `link5_front` | 4 | 8e-06 |
| `link1`, `link2`, `link4` | 19 | 0.00e+00 |

Worst single sensors: `link5_back_sensor_4` (2.82e-03), `link6_sensor_0` (1.12e-03),
`link6_sensor_3` (4.51e-04). This is the expected geometry — the parked hazard is reachable
by the wrist and forearm, and the proximal links never see it, so they have nothing to
predict and no error to make.

## Oracle-zero and hazard-absent behaviour

Hazard-absent frames are an exact control in the data: current and parked fields are bitwise
equal, the changed mask is empty, and the oracle differential is exactly 0.0. The frozen
model's hazard-absent RMS is 0.0084 against a raw SafetyHead RMS of 2.176 on those same
frames — it is quiet where the scene is genuinely clear, at 0.39% of the raw signal.

Oracle-zero RMS is 0.0063. Recall on oracle-active frames is 94–98%. The model is not
winning by predicting the majority class: the `model_predicts_the_majority_zero_class`
hypothesis is explicitly **unsupported** (recall 0.983 ≫ 0.10 threshold).

## Readiness gates

**Technical — 8/8 pass**

| Gate | Result |
|---|---|
| No partition leakage (5 identity keys) | PASS — 0 crossings |
| Input contract valid | PASS — 0 prohibited, 0 non-live |
| Dataset unchanged | PASS — tree hash re-verified |
| No nonfinite output | PASS — 0 |
| Zero constraint violations | PASS — 0, counted not clamped |
| Frozen SafetyHead unchanged | PASS — fingerprint identical |
| Checkpoint reload deterministic | PASS — bitwise identical on a fixed batch |
| Offline test opened once after freeze | PASS |

**Generalization — 7/8 pass**

| Gate | Required | Observed (frozen primary, all seeds) | Result |
|---|---|---|---|
| Offline MAE beats ZERO | ≥25% | 78.4% / 66.8% / 73.9% | PASS |
| Temporal-history value | 10% or freeze simpler | history −13.0%; simpler model frozen | PASS (alt branch) |
| Median direction cosine (active) | ≥0.75 | 0.9992 / 0.9979 / 0.9989 | PASS |
| **Oracle-zero false-positive rate** | **≤2%** | **2.15%** / 1.42% / 1.10% | **FAIL** |
| Hazard-absent RMS ratio | ≤25% | 0.39% | PASS |
| Changed-mask AUPRC vs prevalence | ≥3× | 1128× | PASS |
| Per-source-mode direction cosine | ≥0.5 | 0.9985–0.9996 (3 modes; 1 unevaluable) | PASS |
| Seed coefficient of variation | <20% | 17.7% | PASS |

## Failure analysis

Two of the nine hypotheses are supported by the recorded evidence:

- **`static_state_sufficient_history_unnecessary` — supported.** Four frames underperform
  one (−13.0%) and are twice as variable across seeds. The parked hazard is static, so its
  signature is instantaneous. Proximity is still essential — removing it entirely
  (`QPOS_ONLY`) costs 76.6% and lands worse than the trivial baseline.
- **`calibration_instability` — supported.** Threshold CV 0.72 across seeds; this is the
  direct cause of the one failing gate.

Explicitly **not** supported: signal not learnable (73% improvement), majority-zero
collapse (98% recall), field-accurate-but-head-wrong, head-accurate-but-localization-weak
(mask F1 0.847), source-mode shift (ratio 1.99, just under threshold), oracle-active events
too sparse.

**`learner_induced_distribution_causes_failure` is not evaluable**, as recorded above — that
distribution is absent from every partition except train.

One further caveat on the seed-CV gate. Re-running the *identical* configuration and seed
gave 0.019381 during selection and 0.022328 during final training — a 15.2% same-seed spread
from non-deterministic GPU kernels. The 17.7% across-seed CV is therefore substantially
kernel noise rather than seed sensitivity, and the gate passes with less headroom than the
number alone suggests.

## Checkpoints

Held outside git; no approved artifact policy exists for binary weights in this repository.
All nine are pinned in `final_decision.json` with local path, sha256, config hash, root/ACT/
MolmoSpaces commits, Safety-CVAE hashes, dataset tree hash, partition hash, input-contract
hash, seed, best epoch, validation metric and calibrated threshold.

Frozen primary (`CURRENT_FRAME_ONLY`), under
`…/scratchpad/final/CURRENT_FRAME_ONLY__seed{N}/best.pt`:

| Seed | sha256 (first 16) | Best epoch | Validation MAE | Threshold |
|---|---|---:|---:|---:|
| 0 | `47ae5cb1206c07d3` | 53 | 0.020782 | 0.02819 |
| 1 | `8797ad87e20f2d54` | 61 | 0.019808 | 0.05541 |
| 2 | `b4fa1b6daa2cf66b` | 82 | 0.020277 | 0.03487 |

## Changed files

Added: `causal_parked_skin/` (`data.py`, `model.py`, `losses.py`, `metrics.py`, `engine.py`,
`gates.py`), six `scripts/causal_parked_skin_*.py`,
`tests/test_causal_parked_skin_reference.py` (59 tests),
`diagnostics_output/causal_parked_skin_reference_v1/*.json`, this document. Nothing under
`submodules/` was touched.

## Exact commands

```bash
SCR=<scratchpad>            # cache and checkpoints live outside the repo
PY=/root/act_retrain_venv/bin/python

# 1. contract and partition audits
$PY scripts/causal_parked_skin_input_contract_audit.py \
    --manifest configs/hybrid_obstacle_parked_skin_supervision_v1.json \
    --stack configs/hybrid_safety_stack_v1.json \
    --out diagnostics_output/causal_parked_skin_reference_v1/input_contract_audit.json
$PY scripts/causal_parked_skin_partition_independence.py \
    --manifest configs/hybrid_obstacle_parked_skin_supervision_v1.json \
    --partition-config configs/hybrid_obstacle_reference_partition_v2.json \
    --out diagnostics_output/causal_parked_skin_reference_v1/partition_independence.json

# 2. build the read-only cache (never writes to the frozen dataset)
$PY -c "from causal_parked_skin.data import build_cache; from pathlib import Path; \
        build_cache(Path('configs/hybrid_obstacle_parked_skin_supervision_v1.json'), \
                    Path('$SCR/parked_skin_cache'))"

# 3. bounded validation-only selection (never loads offline test)
$PY scripts/causal_parked_skin_select.py --cache $SCR/parked_skin_cache \
    --stack configs/hybrid_safety_stack_v1.json --safety-dir assets/safety/cvae_v3 \
    --checkpoint-root $SCR/candidates --max-epochs 100 --patience 12 \
    --batches-per-epoch 300 \
    --out diagnostics_output/causal_parked_skin_reference_v1/selection.json

# 4. three seeds, calibration, then offline test exactly once
$PY scripts/causal_parked_skin_train_final.py --cache $SCR/parked_skin_cache \
    --stack configs/hybrid_safety_stack_v1.json --safety-dir assets/safety/cvae_v3 \
    --selection diagnostics_output/causal_parked_skin_reference_v1/selection.json \
    --checkpoint-root $SCR/final --max-epochs 100 --patience 12 --batches-per-epoch 300 \
    --out diagnostics_output/causal_parked_skin_reference_v1/final_training.json

# 5. attribution and decision
$PY scripts/causal_parked_skin_failure_analysis.py \
    --final-training diagnostics_output/causal_parked_skin_reference_v1/final_training.json \
    --stack configs/hybrid_safety_stack_v1.json \
    --out diagnostics_output/causal_parked_skin_reference_v1/failure_analysis.json
$PY scripts/causal_parked_skin_write_decision.py \
    --input-contract diagnostics_output/causal_parked_skin_reference_v1/input_contract_audit.json \
    --partition-report diagnostics_output/causal_parked_skin_reference_v1/partition_independence.json \
    --selection diagnostics_output/causal_parked_skin_reference_v1/selection.json \
    --final-training diagnostics_output/causal_parked_skin_reference_v1/final_training.json \
    --dataset-decision diagnostics_output/hybrid_obstacle_parked_skin_dataset/final_decision.json \
    --safety-dir assets/safety/cvae_v3 \
    --out diagnostics_output/causal_parked_skin_reference_v1/final_decision.json

# verification
$PY -m pytest tests/test_causal_parked_skin_reference.py -q
$PY -m ruff check causal_parked_skin/ scripts/causal_parked_skin_*.py
```

Resume from a checkpoint: `causal_parked_skin.engine.load_checkpoint(path, device)` rebuilds
the model exactly as trained; `restore_rng(payload["rng"])` restores the torch, CUDA and
numpy streams; optimizer and scheduler states are in the same bundle.

## Constraints honoured

Frozen dataset not modified and still mode 444. Partitions unchanged. Offline test never used
for architecture, loss, threshold, epoch or checkpoint selection, and loaded exactly once.
No oracle-zero frames removed or subsampled. ACT not trained or modified. Safety-CVAE not
trained or modified; SafetyHead weights frozen and verified unchanged. No `development4`,
no `confirmatory41`, no simulator or policy rollouts. No parked skin, oracle differential,
future frames, future actions or obstacle geometry used as input. No RGB added. Nothing
pushed.

## Next recommended task

The blocking issue is threshold transfer, not model quality. In order:

1. **Re-derive the activation threshold with an uncertainty margin.** Fit it on calibration
   *and* validation pooled (7,731 frames, both disjoint from offline test), and take a
   conservative lower quantile — e.g. target 0.5% rather than 1% — so the ≤2% ceiling holds
   with margin. This needs no retraining; the checkpoints already exist. On the recorded
   numbers a modest threshold increase clears the gate on all three seeds.
2. **Freeze `CURRENT_FRAME_ONLY` as the reference architecture** and drop the four-frame
   buffer from the live contract. It is better, more stable, and removes a live buffer the
   deployed system would otherwise have to maintain.
3. **Only then** consider a live development round. Note this task establishes offline
   feasibility only; nothing here demonstrates a live safety improvement, and the
   learner-induced distribution — the one that broke the previous reference — remains
   unmeasured because the frozen dataset carries it in train alone.

PARKED_REFERENCE_MODEL_OVERFIT
