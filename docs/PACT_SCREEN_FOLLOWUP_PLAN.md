# Next steps: the screen's primary endpoint is invalid, but the ACT comparison may be real

## What the screen actually showed

| Arm | Collision-free success |
|---|---:|
| ACT | 19/40 (47.5%) |
| PACT | 29/40 (72.5%) |
| PACT_ZERO | **1/40 (2.5%)** |

The awarded token was `FRONTEND_SCREEN_SIGNAL_PRESENT`, on a primary endpoint of
PACT − PACT_ZERO = +70.0 pp.

**That endpoint does not measure modality use, and the +70 pp must not be reported as if it
did.** PACT_ZERO at 2.5% is not "PACT without proximity" — it is far *below* ACT's 47.5%, and a
policy that merely lost an input should degrade toward a vision-only policy, not collapse to
near-zero.

## The evidence: the zeroed token is far out of distribution

Sampling 37,880 real proximity embeddings from the training set:

| Quantity | Value |
|---|---:|
| Real token L2 norm — mean / min / 1st pct | 6.31 / 6.02 / 6.10 |
| Zeroed token L2 norm | 0.00 |
| Real tokens with norm < 0.1 | **0 of 37,880** |
| Zero vector distance from per-dim mean | median 2.12σ, max 5.59σ |
| Dimensions beyond 3σ | 9 of 32 |

Real embeddings live in a tight shell around norm 6.3. The all-zeros vector never occurs — not
once in 37,880 samples. Zeroing does not remove information; it injects a token the network has
never seen.

The behavioural diagnostics confirm global derailment rather than selective loss of avoidance:
PACT_ZERO's `grasp_target` pair entries are 637,970 against PACT's 10,622,992 (16× fewer), it
touches the target in 27/40 episodes against PACT's 37/40, and it fails after gripper close 5
times against PACT's 1. It is not failing to dodge the panel — it is failing to do the task at
all.

**This also means the old and new ablations are not comparable.** Under the previous 3-D point
representation, a zero vector plausibly sat near the in-distribution "nothing detected" region,
which is likely why PACT_ZERO ≈ PACT then. Verify that explicitly with the same norm test on
the old representation before drawing any contrast between the two runs.

## Step 1 — replace the ablation with a distribution-matched one

Do this before anything else. It is cheap: one arm, 40 rollouts, same instances, same
checkpoint, no retraining.

Primary replacement: **permuted proximity tokens.** At each timestep feed proximity embeddings
drawn from a different randomly chosen timestep/episode of the same dataset. This destroys the
information content while preserving the marginal distribution exactly, so the input stays in
distribution.

Report alongside it, as cheap sanity arms if budget allows:

- **mean-token**: every sensor receives the dataset-mean embedding (in-distribution, zero
  information, no variance)
- keep **PACT_ZERO** as a diagnostic *only*, explicitly labelled an OOD-robustness probe rather
  than a modality ablation

Predeclare which one is primary before running. The modality question is then:

**PACT − PACT_PERMUTED**, paired by instance on identical weights.

If that gap is large, proximity is genuinely being used. If it collapses toward zero, the +70 pp
was an OOD artifact and the modality is still unused — the same conclusion as the previous run.

## Step 2 — clear the ACT confound before spending anything on a confirmatory run

PACT − ACT = **+25.0 pp**, paired bootstrap CI [+7.5, +42.5], Fisher p = 0.0392. This comparison
is *not* affected by the zeroing problem — it compares two independently trained policies on the
same 40 instances. It is the real candidate signal.

But it must clear one confound first. ACT is the reused `act_seed3101` checkpoint, trained on the
earlier conversion; PACT was trained on `dataset_embedding32_v1`. The split manifest is the same
(`full_act_split_v2.json`, `7d25e884…`), which is good. Verify the rest:

- identical episode set and per-episode lengths
- byte-identical `observations/images`, `qpos`, `qvel`, and `action` payloads between the two
  conversions, with only the proximity channel added
- identical normalization statistics
- identical recipe (already confirmed: batch 8, lr 1e-5, 2000 epochs, chunk 100, hidden 512,
  enc/dec 7, wrist_camera, state_dim 9, action_dim 8)

If anything but proximity differs, the +25 pp is confounded by training data and the ACT arm must
be retrained on the new conversion.

Also note the encoder's geometric accuracy barely moved: 3.26 → 3.20 cm mean, 51.3% → 52.9%
within 2 cm. So any real gain comes from the richer 32-D representation, not better surface
estimates. Worth stating explicitly — it is the interesting mechanistic claim.

## Step 3 — only then, a confirmatory run

Proceed only if Step 1 shows a real gap under a valid ablation **and** Step 2 clears the
confound. Preregister fresh:

- 160 held-out instances, none used in the screen or prior runs
- arms: ACT, PACT, PACT_PERMUTED (plus PACT_ZERO as a labelled diagnostic if desired)
- **≥2 seeds per arm** — the screen ran one seed each, so PACT − ACT currently carries
  uncontrolled seed noise, which is exactly what defeated the previous confirmatory run
- primary endpoint and analysis frozen before the first rollout
- **budget from measured throughput: 0.4 rows/min was observed**, so ~960 rollouts is ~40 hours,
  not 24–32. Plan and communicate against the measured rate.

## Step 4 — amend the screen report

Record in `PACT_FRONTEND_SCREEN_DECISION.md` that the primary endpoint was invalidated
post hoc by an out-of-distribution ablation, with the norm evidence above. Keep the token — it
was awarded honestly under the frozen rule — but annotate that it rests on an invalid instrument
and that the decision to proceed now rests on PACT − ACT plus Step 1.

Do not silently re-award a different token, and do not delete the original finding.

## What was done well and should carry forward

Execution integrity was excellent and should be kept verbatim: 120/120 rows reconciled with 120
scientific results and 120 driver records, the detached smoke proof passed, the frozen
eight-worker count held, throughput was measured outcome-blind in the first 20 minutes, and the
storage compaction preserved byte-exact recoverability while emitting no endpoint value. The
supervisor problem from the interruption is properly fixed.

## Constraints

- Do not retrain PACT to chase the result; Step 1 requires no retraining.
- Do not move the go/no-go rule after seeing outcomes.
- Do not report +70 pp as evidence of modality benefit anywhere.
- A screen still cannot award `PACT_BENEFIT_ESTABLISHED`, `PACT_NO_CONFIRMED_BENEFIT`, or
  `PACT_WORSE_THAN_ACT`.
- Work in `/root/prox_learning_pact_remediation`; do not touch `/root/prox_learning_hybrid_safety`
  or `confirmatory41`.
- Do not commit rollout H5s, videos, or checkpoints. Do not push.
