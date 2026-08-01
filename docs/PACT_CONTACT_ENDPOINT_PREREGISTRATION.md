# PACT contact-endpoint preregistration

This document freezes the contact-endpoint experiment before its first rollout. The scene, frozen surface encoder, learned 32-D representation, existing seeds, and distribution-matched permutation ablation remain unchanged. Seed 3103 adds one independently initialized matched ACT/PACT pair.

The run has 100 fresh instances, three policy seeds, four arms, one repeat, and 1,200 total rollouts at a fixed eight workers. The arms are ACT, PACT, PACT_ZERO, and PACT_PERMUTED. PACT_ZERO is explicitly an out-of-distribution sensor-failure probe; it is never modality evidence. PACT_PERMUTED is the decision-bearing modality-information instrument.

## Frozen populations

The geometry-only 50% wrist-camera criterion classified 285/285 eligible recorded episodes as vision-disadvantaged. This is a degenerate partition, so subset analysis is dropped without changing the threshold. Only the full fresh-instance distribution is analyzed. Partition SHA-256: `a3ac51bf15aa4ecc1f31dfca3f1f6d59a446bdd7cf2323395ecb06fd6b73bdc2`.

## Endpoints and power

The co-primary operational endpoint is task success with zero hazard-bar and zero other-environment contact entries. The co-primary magnitude endpoint is hazard-bar contact frames per rollout. Hazard frames conditioned on both compared arms achieving manipulation success are diagnostic. All declared secondary contact, penetration, and manipulation metrics remain reported.

The paired-normal design calculation used prior outcomes only for sizing. At n=100, the 80%-power, two-sided 5% MDE is 890.955 hazard frames. The historical absolute effect was 896.825; the approximation required 99 instances for the count endpoint versus 108 for binary any-contact. Thus the count endpoint saves only about nine instances, not a material reduction.

## Frozen analysis and decisions

Every paired difference uses 20,000 deterministic bootstrap replicates. Whole instances are clusters: all arms and all seeds move together. Seeds are shown separately before pooling, and medians accompany heavy-tailed contact-frame means.

Two-sided Fisher exact tests accompany binary contrasts, including PACT versus ACT and PACT versus PACT_ZERO. Pooled Fisher values are explicitly cluster-unaware descriptive tests; the whole-instance bootstrap remains the cluster-aware inference.

`CONTACT_REDUCTION_ESTABLISHED` requires the pooled PACT-minus-PERMUTED hazard-frame 95% CI to be strictly below zero and a negative difference in every seed. `CONTACT_REDUCTION_WITH_TASK_BENEFIT` additionally requires positive PACT-minus-ACT collision-free task success pooled and in every seed. A CI strictly above zero yields `CONTACT_INCREASE`. A CI including zero, or a pooled reduction with inconsistent seed signs, yields `NO_CONTACT_REDUCTION`. The subset-only token is unavailable because the partition was dropped. An unreconciled schedule yields `CONTACT_EXPERIMENT_INCOMPLETE`.

Frozen analyzer SHA-256: `cdccb37357619c7cbce558e9761a10889bfca7aa48bbf3b31429b9ebc6940754`. Preregistration SHA-256: `108a1299aebc81b45cbc01daadd2ddce4b236028512322300fafb1fbe17d615c`.
