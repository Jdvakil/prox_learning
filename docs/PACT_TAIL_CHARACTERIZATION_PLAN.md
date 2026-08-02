# Plan: characterize the contact tail, then push

No GPU. Pure analysis of the 1,200 rollouts already collected under the contact-endpoint run.

## Why

The confirmed result has an unexplained shape. Contact drops from 22.3% to 14.0% of episodes and
hazard frames from 3,023 to 1,678, yet:

- the **median is 0.0 hazard frames for every arm** — most episodes never touch the obstacle
- the difference **conditioned on both arms succeeding at manipulation is −0.8, CI [−19.8, +22.0]** —
  indistinguishable from nothing

So the entire effect comes from a minority of episodes. The write-up currently *infers* that these
are entrapment cases where the arm becomes stuck against the panel. **Nobody has looked.** That is
the first question a reader will ask, and it is answerable from data already on disk.

## Status of this work

**Exploratory and descriptive.** It must not change the awarded token
(`CONTACT_REDUCTION_WITH_TASK_BENEFIT`), the frozen endpoints, or the preregistered analysis. Do
not present any new comparison here as confirmatory, and do not attach decision-bearing p-values
to post-hoc groupings. Label every output as post-hoc characterization.

## Analyses to run

Work per rollout from the existing `result.json` files, keyed by instance, arm, and seed.

**1. Shape of the distribution.** Per arm: fraction of rollouts with zero hazard frames, and the
percentiles (50/75/90/95/99/max) of the nonzero remainder. Establish concretely how heavy the tail
is.

**2. Concentration.** Sort rollouts by hazard frames and report what share of all contact comes
from the top 1%, 5%, and 10% of episodes, per arm. If the top 5% carries most of it, say so with
the number.

**3. Is it the scene or the policy?** For each instance, check whether the tail episodes are the
*same instances* across arms. If certain instances blow up for every arm, the tail is a property of
the scene; if they differ by arm, it is a property of the policy. **This is the single most
informative analysis in the list** — report the overlap explicitly.

**4. Entry versus escape.** Decompose the effect into two questions:
   - how often does each arm *enter* a high-contact regime — P(hazard frames > threshold), for a
     threshold stated up front, e.g. 500 frames
   - *given* entry, how long does contact persist — mean and median frames among entered episodes

   This distinguishes "proximity avoids the trap" from "proximity escapes the trap faster." They
   are different mechanisms and the write-up should not guess between them.

**5. Is it entrapment?** Test the stuck hypothesis directly. Compute the frames-to-entries ratio
(high ratio = sustained contact, low = repeated brushing) and, if per-step contact timing is
available in the audit records, the longest contiguous contact run. Check whether
`pact_contact_audit.py` retains per-step contact classes; if only episode totals exist, say so and
use the ratio plus `first_contact_step` as the proxy.

**6. What else is true of tail episodes.** For high-contact rollouts: manipulation success rate,
collision-free success rate, where contact begins (`first_contact_step`) relative to the 900-step
horizon, and maximum penetration depth. Are these episodes that were already failing, or ones the
contact itself derailed?

**7. Paired per-instance view.** For each of the 100 instances, the PACT-versus-ACT and
PACT-versus-PACT_PERMUTED contact difference. Count instances where PACT is better, worse, and
tied. Report the discordance rather than only the mean — with a heavy tail, the mean is a poor
summary.

## Reporting

Write `docs/PACT_TAIL_CHARACTERIZATION.md` with the tables and a plain statement of which
mechanism the data supports:

- proximity *prevents entry* into the entrapment regime, or
- proximity *shortens* entrapment once it starts, or
- the tail is instance-driven and proximity changes which instances trigger it

If the data does not clearly distinguish these, say that. An honest "cannot separate these from
episode totals alone" is a better outcome than a confident guess, and it identifies exactly what
per-step logging a future run should retain.

Machine-readable output to
`diagnostics_output/pact_contact_endpoint/tail_characterization.json`, with a self-hash.

## Then: pull the discoveries and push

Three discovery documents were written in the sibling clone and pushed to GitHub, but are not in
this worktree. Bring them in so the findings live alongside the work that produced them:

```bash
cd /root/prox_learning_pact_remediation
git fetch origin qualify/hybrid-obstacle-three-pair-live-v1
git checkout FETCH_HEAD -- docs/discoveries/
```

That pulls only `docs/discoveries/` — three files:

- `001-zeroing-is-not-a-valid-ablation-for-embedding-front-ends.md`
- `002-modality-effects-are-invisible-to-cross-model-success-comparisons.md`
- `003-proximity-reduces-obstacle-contact-in-the-tail-not-in-routine-operation.md`

Read 003 before writing the characterization — it states the current inference about the tail, and
your job is to confirm, refine, or overturn it. **If the data contradicts it, update 003 rather
than working around it.**

### Push

The default credential helper is the OpenResearch bot, which is denied on this remote (403).
Credential helpers are additive, so the chain must be reset first:

```bash
git -c credential.helper= -c credential.helper='!gh auth git-credential' \
    push origin experiment/pact-valid-ablation-followup-v1
```

Verify with `gh api user --jq .login` — it should report `Lundii1`.

Commit the characterization document, the JSON artifact, the pulled discoveries, and any updates
to 003. **Do not commit** rollout H5s, videos, checkpoints, or the raw artifact tree under
`/root/pact_contact_endpoint_artifacts/`.

## Constraints

- Do not run any rollouts; this is analysis of existing artifacts only.
- Do not change the awarded token, the frozen endpoints, or the preregistered analysis.
- Do not present post-hoc groupings as confirmatory evidence.
- Do not touch `/root/prox_learning_hybrid_safety`, its submodule checkouts, or `confirmatory41`.
- Do not retrain or re-evaluate anything.
