# Seed 3103: main fine-tuned encoder versus frozen PACT

The main fine-tuned method improved task success and collision-free task success on the same 50 seed-3103 regression scenes. The new policy completed 60,000 committed updates. Independent reading of all 100 old/new trajectories exactly confirmed the reported endpoints.

| Endpoint | Frozen 32-D | Fine-tuned 128-D |
|---|---:|---:|
| Task success | 20/50 (40%) | 27/50 (54%) |
| Collision-free task success | 16/50 (32%) | 24/50 (48%) |
| Exclusive pickup failures | 16/50 | 8/50 |
| Raw touched but never held | 18/50 | 8/50 |
| Ever held | 25/50 | 35/50 |
| Lifted at least 1 cm | 26/50 | 33/50 |
| Lift sustained for at least 15 observations | 22/50 | 30/50 |
| Continuous bilateral contact for at least 1 s | 27/50 | 34/50 |
| Post-lift placement failures | 6/50 | 6/50 |
| Held without lift failures | 1/50 | 2/50 |
| No target interaction failures | 7/50 | 7/50 |
| Hazard/clutter union contact frames | 59,086 | 29,000 |
| Frame avoidance | 96.021% | 98.047% |

Task success increased by 14 percentage points and collision-free task success by 16 points. Exclusive pickup failures halved, and combined hazard/clutter contact frames fell by 50.9%. The improvement is concentrated in acquisition; post-lift placement failures stayed at six.

The hazard-bar result is a remaining weakness: hazard-bar contact frames increased from 18,183 to 20,148 (+10.8%). Clutter contact frames decreased from 41,990 to 9,251. The combined reduction should not be described as a reduction in every contact class.

## Paired scene changes

The fine-tuned policy gained 12 task successes and lost five previous successes. It gained 11 collision-free successes and lost three. Eleven scenes stopped being exclusive pickup failures, while three became new pickup failures. For that last endpoint, a candidate-only failure is a regression, not a win.

Ten original pickup-failure scenes now show a lift and at least one second of continuous bilateral contact. Their timing is retained in comparison.json; that aggregate does not by itself prove that the measured contact interval caused the lift.

[Per-scene comparison](paired_scene_comparison.csv) lists all 50 identities, both outcomes, failure categories, contact counts and source directories.

## Contact detail

| Class | Frozen frames | Fine-tuned frames |
|---|---:|---:|
| Hazard bar | 18,183 | 20,148 |
| Clutter | 41,990 | 9,251 |
| Grasp target | 308,054 | 366,834 |
| Other environment | 0 | 0 |
| Mounted fixture | 0 | 0 |
| Place receptacle | 0 | 0 |

Episodes with hazard-bar contact: 5 → 6. Episodes with clutter contact: 18 → 13.

Each arm contains 1,485,050 contact samples (29,701 per rollout, nominal 2 ms spacing), 900 actions and 901 observations per scene. Union frames count simultaneous hazard/clutter contact once. Grasp-target contact is reported separately from the forbidden contact classes. Exclusive pickup failure follows the existing failure-stage priority; raw touched-but-never-held is a separate, nonexclusive descriptor.

## Source and integration

The source is prox_learning main at af286905719c939c74e6f7c9cf1cc2ed7a9e5e64 and its pinned ACT fork at d956cbcab832a39e83f7face8b533a0c1bfad06c. All 18 retained upstream files match their Git source bytes. The encoder, causal-window helpers, live forward path, encoder optimizer group and encoder serialization execute the upstream definitions. See [the source manifest](upstream_manifest.json) and [integration adapter](../../scripts/pact_v1010c_core.py).

This is the full main method: trainable encoder stem/transformer, 128-D CLS readout and minimum pooling of the four native subframes. It is not an isolated unfreezing experiment. The same existing pretrained encoder used by frozen PACT initialized this run; initialization provenance is in [initialization.json](initialization.json). The separately documented pretrained checkpoint on main was unavailable locally.

The policy trained from scratch with seed 3103, the original 240/40 split, uniform starts, batch 8 and the original action/state statistics. Policy, image-backbone and encoder learning rates were 1e-5. The existing wrist 240×320 input, chunk/history 100 controller, 7/7 transformer layers, hidden size 512, feedforward size 3200, eight heads, KL weight 10, nine state values and eight action values were preserved. Main’s eight-frame causal proximity window retains the existing canonical sensor slots. [The plan](../../docs/PACT_PLACE_V1010C_READOUT_3103_PLAN.md) records the full intervention.

The evaluated policy and encoder are the fixed final pair at update 60,000. Validation did not select a different pair. All 50 original scene identities and successful sampling retries were retained, with matching physical initial states, the existing RGB tolerance, task geometry, action decoder and full horizon. These scenes had already been exposed by earlier audits; results establish a regression improvement for seed 3103, not performance on other seeds or a fresh confirmatory result.

## Verification and infrastructure

Preflight verified source bytes, pretrained loading, nonzero encoder gradients and real stem updates, unused pretraining heads, original non-proximity samples and RNG, causal train/eval parity, checkpoint pairing and resource guards. The final policy and encoder passed strict reload. Every evaluated readout policy consumed 40×128 proximity features on all 900 control queries.

The root’s independent reader used the raw trajectory, action and contact files without importing the experiment metrics module. It confirmed all 100 old/new primary, pickup and lift endpoints, aggregate counts, paired changes, action decoding and horizons. All initial match checks and protected source/input hashes passed. All operational recovery source hashes also match their launch records. See [independent verification](root_review/final_review.json) and [supervisor verification](final_verification.json).

The original frozen 50 outcomes were reused after raw/hash verification. One extra frozen smoke was kept outside that denominator. Its tiny RGB rendering differences explained the first-query mismatch: both retained first vectors were reproduced exactly from their own images with identical non-RGB policy inputs. The complete replay action trace was not bit-identical; its task outcome and gripper commands matched. [Diagnosis](amendments/02_frozen_render_diagnosis/NOTES.md).

Two training workers were interrupted by shared-host Git-process pressure. Joint checkpoints retained updates 15,600 and 20,700. The first interruption discarded 60 logged updates and up to 30 further unlogged updates; the second occurred after the 20,700 checkpoint, before the next training epoch. Final committed training remains 60,000 updates. Restored state/RNG were verified, but subsequent GPU training replay losses were not bit-identical; the precise source of that numerical difference was not isolated. [Recovery records](recoveries/03_thread_capacity/NOTES.md).

The captured pressure came from thousands of app-server Git diffs of generated artifacts. A data-loader capacity wait preserved all 35 real epoch batches and parent RNG in testing, and successfully waited through a later 114-second surge. A new diagnostics_output/.gitignore keeps generated payloads out of background diffs while retaining source code and Markdown reports for review. Files remain on disk. [Capacity recovery](recoveries/04_capacity_wait/NOTES.md).

The first eight evaluations completed under a RAM-guard drain. Read-only cache advice released unused training-file pages, and the remaining 42 evaluations ran with six workers. All 50 evaluations had observed exit 0 and none were replaced. The ledger contains 53 valid worker exits (50 readout evaluations, one frozen smoke and two completed training segments) plus the two disclosed training failures. The final supervisor exited 0, observed by the root through session 60074. [Evaluation recovery](recoveries/05_eval_capacity/NOTES.md).

## Saved model pair

- Policy: [policy_last.ckpt](checkpoint/policy_last.ckpt), SHA256 `3d01957cd3d86bb95db13bb4b96db05ebe07794134e4f5f64b683205afd46dcd`.
- Encoder: [prox_encoder.pt](checkpoint/prox_encoder.pt), SHA256 `c1127d13cc195197c7375b5bb03081b65f9a44dabe477cb28d9408b5308e7cbc`.
- Pair index: [checkpoint_pairs.json](checkpoint/checkpoint_pairs.json).
- Complete raw metrics and matched comparisons: [comparison.json](comparison.json).
