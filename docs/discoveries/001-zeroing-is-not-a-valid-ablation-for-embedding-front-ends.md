# 001 — Zeroing is not a valid ablation once the front-end emits learned embeddings

*Discovered 2026-07-29, during the PACT wider-front-end screen.*

## The lesson, stated generally

**An input ablation is only valid if the ablated input stays inside the training
distribution.** Zeroing a modality is the standard trick, and it is safe when zero is a value
the network actually sees — a raw sensor reading of "nothing detected," a masked patch, a
padding token. It becomes invalid the moment the modality is delivered as a *learned embedding*,
because a learned embedding almost never has the zero vector in its range.

When that happens, the ablation stops measuring "the policy without this modality" and starts
measuring "the policy fed an impossible input." Those two quantities can differ by an enormous
margin, and the second one looks exactly like a spectacular positive result.

## How it showed up

Two runs of the same comparison, with the proximity front-end changed between them and the
ablation method left untouched.

| | Run 1 (3-D point front-end) | Run 2 (32-D embedding front-end) |
|---|---:|---:|
| ACT | 53.1% | 47.5% |
| PACT | 49.7% | 72.5% |
| PACT_ZERO | 50.0% | **2.5%** |
| PACT − PACT_ZERO | −0.3% [−1.6, +0.9] | **+70.0 pp** [+55.0, +82.5] |

Run 1 said the policy ignored proximity entirely. Run 2 appeared to say the policy depended on
it almost totally. The front-end change looked like a triumph.

The tell was that **PACT_ZERO (2.5%) fell far below ACT (47.5%)**. ACT is a policy that never
had proximity at all. Removing an input cannot leave you *worse* than never having had it — at
worst you degrade toward the vision-only policy. Falling 45 points below it means the input was
not removed; it was corrupted.

## The evidence

Sampling 37,880 real proximity embeddings from the training set:

| Quantity | Value |
|---|---:|
| Real token L2 norm — mean / min / 1st percentile | 6.31 / 6.02 / 6.10 |
| Zeroed token L2 norm | 0.00 |
| Real tokens with norm < 0.1 | **0 of 37,880** |
| Zero vector distance from per-dim mean | median 2.12σ, max 5.59σ |
| Dimensions beyond 3σ | 9 of 32 |

Real embeddings occupy a tight shell around norm 6.3. The origin is empty — not one sample in
37,880 lands near it. The encoder has no reason to produce a zero vector and never does.

The behavioural signature confirms derailment rather than modality loss. PACT_ZERO's
`grasp_target` contact entries were 637,970 against PACT's 10,622,992 — sixteen times fewer. It
touched the target in 27/40 episodes against 37/40, and failed after gripper close 5 times
against 1. It was not failing to dodge the obstacle. It was failing to do the task at all.

## Why Run 1's ablation was probably fine

Under the 3-D point representation, each sensor contributed a position plus a validity flag. An
all-zeros vector there plausibly sits inside the training distribution — it reads as "no surface
detected," which is a state the network sees constantly. That is likely why PACT_ZERO ≈ PACT in
Run 1, and why that −0.3% remains a *valid* measurement.

So the two runs' ablations are not comparable, and the swing from −0.3% to +70 pp is not
evidence that the front-end fix worked. One number came from a working instrument and the other
did not.

*(This is a strong inference, not yet verified. Running the same norm test on the 3-D
representation would confirm it.)*

## The correct ablation

Replace zeroing with an intervention that destroys **information** while preserving the
**distribution**:

1. **Permuted tokens (preferred).** Feed real proximity embeddings drawn from a different
   randomly chosen timestep or episode. The marginal distribution is exactly preserved; the
   correlation with the current scene is destroyed.
2. **Mean-token replacement.** Every sensor receives the dataset-mean embedding. In
   distribution, zero information, but also zero variance — a weaker control.
3. **Train-time modality dropout.** Randomly zero the modality during training so the zero
   vector becomes in-distribution by construction. This makes zeroing valid, but it changes the
   trained policy, so it is a design decision rather than a post-hoc fix.

Keep zeroing if you like, but label it what it is: an **OOD-robustness probe**, not a modality
ablation.

## How to catch this in future

Cheap checks, in order of cost:

- **Sanity floor.** Does the ablated arm fall below a baseline that never had the modality? If
  yes, suspect the ablation before believing the result.
- **Norm/percentile test.** Compare the ablated input's norm against the distribution of real
  inputs. If it sits outside the observed range, the ablation is invalid.
- **Behavioural signature.** Check whether the ablated policy fails *selectively* (loses the
  capability the modality supports) or *globally* (stops doing the task). Global collapse is
  the fingerprint of a broken input, not a missing one.

The general rule: **whenever you change how a modality is represented, re-validate the ablation
that measures it.** The instrument silently stops working, and it fails in the direction that
flatters the hypothesis.

## What remains open

- The permuted-token ablation had not yet run when this was written. It decides whether the
  wider front-end genuinely made the policy use proximity, or whether the modality is still
  unused and the +70 pp was entirely artifact.
- The surviving candidate signal is **PACT − ACT = +25.0 pp** (paired bootstrap [+7.5, +42.5],
  Fisher p = 0.0392). It is unaffected by the ablation problem, and its main confound is
  cleared: both arms trained on identical demonstrations and an identical 199/56 split
  (`split_rule`, `canonical_manifest_sha256`, episode order and assignment all match; only a
  dataset tree hash differs, because the encoded set carries extra proximity arrays).
- That +25 pp is **n=40 with one seed per arm**. Single-seed comparison is what defeated the
  previous confirmatory run. It is a strong hint, not a result.
- Notably, the encoder's geometric accuracy barely changed: 3.26 → 3.20 cm mean error, 51.3% →
  52.9% within 2 cm. If the gain is real, it comes from passing **surface structure** rather
  than a better point estimate — the old front-end was discarding the part that mattered.

## Convention for this folder

Numbered, one discovery per file, named for the transferable claim rather than the experiment
that produced it. Record the evidence and the numbers inline so the finding survives without
the surrounding project context.
