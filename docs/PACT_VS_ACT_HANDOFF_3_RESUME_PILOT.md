# Handoff 3: re-run the pilot evaluation and finally measure Gates B and C

Working tree: `/root/prox_learning_pact_remediation` (keep it; the isolation worked —
provenance records `confirmatory41_touched_by_pact_work: false`).

## What this handoff is

`PACT_EXPERIMENT_INCOMPLETE` was awarded on a **path bug**, not on a scientific result. All 64
pilot ACT evaluation rows died as `invocation_failure` with the recorded root cause
*"relative manifest path was invalid from evaluator cwd"* — every subprocess died before
loading the manifest. You already fixed it (path resolution before the working-directory
change, with a regression test) and then declined to use the fix.

This handoff unblocks that. It does not reopen the environment question and does not loosen
any scientific threshold.

## What already passed — do not re-measure

Phase 1 environment adequacy is settled. Carry it forward unchanged and do not spend budget
re-running it:

| Measure | Result | Threshold |
|---|---:|---:|
| Active-panel signal inside 20 cm | 54.0% (95% bootstrap 46.0–61.9) | ≥30% |
| Active-panel signal inside 12 cm | 18.1% (95% bootstrap 13.5–22.8) | ≥5% |
| Active scientific expert episodes | 62/62 | ≥5/6 |
| Usable clean demonstrations | 58/64 | ≥48 |
| Expert no-outcome rate | 2/64 (3.1%) | <5% |

Surface observability survived every leave-one-episode-out check. The expert fix worked
(90.6% clean demonstrations, CI 81.0–95.6). The expert-side infrastructure fix worked (17% →
3.1% no-outcome). Vision-only ACT seed 1101 trained cleanly: 2000 epochs, best epoch 1849,
validation loss 0.093538, checkpoint `4fca3b0b…`.

Reuse that pilot checkpoint. Do not retrain it.

## Step 1 — amend the terminal-outcome rule, and freeze the amendment first

This is the central correction. Write it into the preregistration and hash it **before**
launching anything.

> A schedule row becomes outcome-bearing at the moment an initial observation is accepted.
> Any failure strictly before that point — invocation failure, import error, unresolvable
> path, missing file, startup OOM, CUDA init failure — is an **infrastructure failure**. It is
> retryable without limit, is recorded with its cause and retry count, is reported separately
> from scientific outcomes, and can never fail a science gate or award a decision token.
>
> Once an initial observation is accepted, the row is terminal: no later exception, contact,
> task failure, or success may cause a replacement or rerun.

Your existing protocol already said retries are allowed for failures *"before an initial
observation is accepted and before any action."* An invocation failure occurs strictly earlier
than that — the process died before reading the manifest. The rule was right; it was applied
to a class of failure it was never meant to cover.

## Step 2 — justify the re-run explicitly in the report

Re-running the 64 rows is legitimate, and the report must say why rather than assert it:

- zero observations, zero actions, zero scientific outcomes were produced;
- the fix was made before any result was known;
- path resolution is **content-independent** — it cannot influence which rows succeed or what
  they produce, so it cannot bias the outcome distribution.

Re-running rows that produced no outcome is not outcome-based replacement, because there was
no outcome to select on. Outcome-based replacement remains forbidden.

## Step 3 — add a launch smoke test before any schedule dispatch

Before releasing a full schedule, launch **one** row, wait for it, and assert it wrote a
scientific `result.json`. Abort the dispatch if it did not. Record the smoke row's identity in
the schedule artifact.

This is cheap and it is what this run needed: a single 4-minute check would have caught the
path bug instead of consuming all 64 rows. The confirmatory experiment in the neighbouring
repo used exactly this discipline and it caught a sealed-set guard on the first pair rather
than 410 times.

Make this a standing requirement for every future schedule, including the 960-rollout
confirmatory one.

## Step 4 — re-run the pilot evaluation and adjudicate Gates B and C

Run the 64-row pilot ACT evaluation against the frozen schedule
(`e0515adf10a12cca…`) using the repaired runner and the retained pilot checkpoint.

Then adjudicate the gates that have **never been measured** in three attempts:

- **Gate B** — vision-only ACT collision-free success in a band that is neither floor nor
  ceiling (the frozen 8–16/24 intent, expressed as a proportion with an interval), and
  ordinary task success above its floor.
- **Gate C** — the baseline actually collides: non-target contact, and specifically intrusion
  contact, in a meaningful fraction of rows.

Report both with intervals and with the one-outcome-perturbation stability check you already
implemented.

Expected, from evidence in hand: Gate C likely passes — the causal control that disabled the
expert's avoidance bow produced 192 `hazard_bar` entries across 164 frames, so a policy that
does not avoid collides heavily. **Gate B is the genuine unknown.**

If Gate B fails high (ACT near ceiling), the scene needs difficulty, not signal. If it fails
low (ACT near zero), the task is not learnable as posed — stop and report
`PACT_ENVIRONMENT_INADEQUATE`, which is now correctly reserved for real environment failures.

## Step 5 — only if Gates B and C pass

Proceed as already preregistered: full train/validation collection, frozen nearest-surface
encoder (report held-out mean/median Euclidean error, within-2-cm rate, validity precision and
recall), ACT and PACT training on an identical recipe differing only in proximity tokens, then
the three-arm evaluation — ACT, PACT, PACT_ZERO — with Wilson intervals, Fisher's exact, and
instance-clustered bootstrap.

Train ≥2 seeds per arm if budget allows; one seed per arm cannot separate a modality effect
from initialization noise. State the detectable effect size before running.

## Allowed final decisions

- `PACT_BENEFIT_ESTABLISHED`
- `PACT_NO_CONFIRMED_BENEFIT`
- `PACT_WORSE_THAN_ACT` — the recorded prior on collisions was ACT 0 static contacts vs PACT
  835; this must stay reportable
- `PACT_ENVIRONMENT_INADEQUATE` — reserved for Gate B, Gate C, or surface-observability
  failure; never for demonstrator cleanliness or harness faults
- `PACT_EXPERIMENT_INCOMPLETE` — reserved for a genuinely unreconciled schedule, **not** for
  an infrastructure fault that a retry can clear

## Constraints

- Do not re-measure or re-litigate Phase 1 environment adequacy.
- Do not retrain the pilot ACT checkpoint.
- Do not move any scientific threshold; the only rule change permitted is the
  infrastructure/outcome boundary in Step 1, frozen before use.
- Do not replace or re-run any row that produced a scientific outcome.
- Stay in `/root/prox_learning_pact_remediation`. Do not touch
  `/root/prox_learning_hybrid_safety`, its submodule checkouts, `confirmatory41`, or the
  safety-residual chain.
- Check `pgrep -fc eval_act_obstacle_on_policy.py` before heavy GPU use; a paused 410-rollout
  confirmatory run resumes on the same A10.
- Do not commit rollout H5s, videos, or checkpoints. Do not push.

## A note on what to keep

Stopping rather than improvising is correct behaviour and is why all three of these runs are
recoverable instead of quietly wrong. The problem was never the discipline — it was one
mis-scoped rule. Fix the rule, keep the discipline.
