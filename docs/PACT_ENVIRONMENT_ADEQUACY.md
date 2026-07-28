# PACT environment adequacy

## Remediation-v2 decision

Decision: `PACT_ENVIRONMENT_ADEQUATE`.

This is a new, independently seeded Phase 1 experiment. The historical v1 pilot is neither rescored nor pooled. Its preregistered `PACT_ENVIRONMENT_INADEQUATE` decision remains final under v1.

The v2 manifest SHA-256 is `cba7ff8879565e72eb667da3419267104dc9c3141a459de78d635fe760a41784` and its master seed is `2026072901`. The route remains `collision` and the physical scene remains `pact_collision_corridor_v1`.

## What changed—and what did not

The corridor XML, panel geometry, aperture, target distribution, balanced intrusion sides, and target/side independence are unchanged. The expert clearance is 0.10 m, and mid-rollout trajectory replans are disabled. Initial construction retries remain pre-action and deterministic.

Every fixed attempt remains in the ledger. A demonstration is usable only when the task succeeds with zero `hazard_bar` and zero `other_environment` contact. Target contact is exempt.

## Expert demonstrations and infrastructure

| Quantity | Result |
|---|---:|
| Fixed expert attempts | 64 |
| Scientific outcomes | 62 |
| Ordinary task success | 59/64 |
| Usable clean demonstrations | 58/64 |
| Clean-demo Wilson 95% CI | [81.0%, 95.6%] |
| Frozen usable-demo floor | 48 |
| No scientific outcome | 2/64 |
| No-outcome Wilson 95% CI | [0.9%, 10.7%] |

Demonstrator count and infrastructure health are progression criteria, not science gates. They cannot produce `PACT_ENVIRONMENT_INADEQUATE`.

## Surface observability

| Quantity | Point estimate | 95% interval | Frozen minimum |
|---|---:|---:|---:|
| Active scientific episodes | 100.0% | [94.2%, 100.0%] | 83.3% |
| Pre-grasp frames with panel inside 20 cm | 54.0% | [46.0%, 61.9%] | 30.0% |
| Pre-grasp frames with panel inside 12 cm | 18.1% | [13.5%, 22.8%] | 5.0% |

Robust classification: `adequate`. All leave-one-episode-out point estimates pass: `True`.

## Handoff-3 infrastructure recovery

The original dispatch made 64 evaluator attempts but accepted zero initial observations, executed zero actions, and produced zero scientific outcomes. All 64 failed before manifest load because the relative manifest path was invalid from the evaluator working directory.

The absolute-path fix was committed before any policy result existed. Path resolution is content-independent: it cannot select rows by contact or task outcome and cannot change policy actions after startup. Re-executing these unchanged rows is therefore pre-observation infrastructure recovery, not outcome-based replacement.

Predeclared launch-smoke row 0 (`76c59c423483c58cbcf2b5161dcacf690457d180ba84e99e4adaa7a02129af97`) passed once before full dispatch. The repaired ledger contains 64 scientific results, 0 post-boundary terminal failures, and 0 retryable pre-observation failures. Scientific rows rerun: 0.

The expert and surface measurements above are carried forward byte-for-byte from the settled Phase 1 gate; they were not recomputed for this resume.

## Gate B — vision alone is insufficient but solvable

ACT collision-free task success is 23/64 (35.9%), Wilson 95% CI [25.3%, 48.2%]. The target point band is [33.3%, 66.7%], with the interval required inside [20%, 80%].

Robust classification: `adequate`; one-outcome stable: `True`.

Ordinary ACT task success is 23/64 (35.9%), Wilson 95% CI [25.3%, 48.2%]. This is the secondary endpoint.

## Gate C — the baseline contacts the intrusion

ACT contacted the panel in 23/64 scientific rows (35.9%), Wilson 95% CI [25.3%, 48.2%]. The frozen point minimum is 25%, with Wilson lower bound above 10%.

Robust classification: `adequate`; one-outcome stable: `True`. `other_environment` contact occurred in 0 rows and does not substitute for panel contact.

## Decision

All three environment science gates robustly pass, and the predeclared demo/infrastructure progression requirements are met. The experiment may proceed to full collection and training.

The last line is the exact allowed decision token.

PACT_ENVIRONMENT_ADEQUATE
