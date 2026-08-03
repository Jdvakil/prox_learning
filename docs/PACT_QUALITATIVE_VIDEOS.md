# PACT qualitative-video determinism-gate report

> These are illustrative examples selected by a rule fixed before rendering. They are not
> evidence. The quantitative result is contact entry falling from 19.7% to 11.0% of rollouts, in
> `PACT_CONTACT_ENDPOINT_DECISION.md`.

**Status: stopped at the predeclared determinism gate.** No paired qualitative video is being
published. The one completed raw render is retained only as an unpublished determinism probe; it
must not be described as footage of the analyzed rollout. The frozen contact analysis and awarded
token were not rerun or changed.

## Determinism check

The first selected ACT row was rerun once with the offscreen third-person renderer. The initial
observation boundary SHA-256 matched the original exactly, as did task/manipulation success and all
first-contact steps. The required `contact_class_totals` comparison did not match exactly:

| Required field | Original analyzed row | Render rerun | Exact? |
|---|---:|---:|---:|
| `grasp_target` contact-pair samples | 0 | 0 | yes |
| `hazard_bar` contact-pair samples | **29,069** | **29,074** | **no** |
| `other_environment` contact-pair samples | 0 | 0 | yes |
| Task success | false | false | yes |
| Manipulation success (represented by task success) | false | false | yes |
| First contact step | `{grasp_target: null, hazard_bar: 21, other_environment: null}` | same | yes |

Descriptively, the rerun had 29,024 hazard contact frames versus 29,022 originally and maximum
hazard penetration of 0.8429 mm versus 0.8326 mm. Both audits contained exactly 29,701 physics
samples and both retained the same `hazard_bar_contact` failure taxonomy. These close agreements do
not satisfy an exact-match gate. This check cannot distinguish whether the extra rendering
perturbed the execution or the evaluation runtime is slightly nondeterministic.

Per the frozen plan, execution stopped before the remaining nine reruns. Producing the five paired
videos after this mismatch would require labelling them independent draws from the same instances,
not recordings of the analyzed rows. That alternative was not authorized by the plan.

The probe itself is a valid 901-frame, 624×352 MPEG-4 stream at 15.1515 fps, but its overlay does
not carry an independent-draw warning, so it is not a deliverable and should not be published:

- Path: `/root/pact_contact_endpoint_artifacts/qualitative_videos/raw/video_01_act.mp4`
- File SHA-256: `f4d07e0bb5b0d4d69d69a2484bdd17ed799ddf90a4a491b06a6e95d826d1320e`
- Determinism record: `/root/pact_contact_endpoint_artifacts/qualitative_videos/determinism_check.json`
- Determinism-record file SHA-256: `4dccfed76a99444c1b6cee46ee60482485330bd4733eeb88580dec1aecd4fade`
- Determinism-record self-hash: `4ce93fb15cc4958aea84b9028bfd1da34d9f454430a79727b601142d2d3e9a19`

## Frozen selection rule

The following text is verbatim from the pre-render selection contract:

- **Mechanism:** among instance-seed pairs where ACT hazard frames > 500 and PACT = 0, take the
  three with the largest ACT hazard-frame total
- **Routine:** among pairs where both arms have 0 hazard frames and both succeed at the task, take
  the lowest instance ID
- **Counterexample:** among pairs where PACT hazard frames ≥ ACT, take the one with the largest PACT
  total

The selections were generated mechanically and committed before the probe was rendered. No footage
was viewed before they were frozen. Ties were broken by ascending full instance ID and then ascending
policy seed.

## Mechanically selected pairs

The table records the original analyzed outcomes used for selection. It does not claim that final
videos exist.

| Planned video | Category | Full instance ID | Seed | ACT hazard frames / task | PACT hazard frames / task | Selection intent |
|---:|---|---|---:|---:|---:|---|
| 1 | Mechanism | `54a6272f66ca3c7bb57dc603550a4c29d35605e3fe65aaef627a9a06bad00b6f` | 3101 | 29,022 / fail | 0 / success | Contrast ACT's largest qualifying high-contact episode with a clean PACT episode. This ACT row became the failed determinism probe. |
| 2 | Mechanism | `d564bfef2efbd5b2b5edf383521d08025dae87e21eda9819b4233e13d2b3155a` | 3101 | 28,736 / fail | 0 / fail | Illustrate contact avoidance independently of ordinary task success. Not rerun. |
| 3 | Mechanism | `36a756384dd5640149b1301d36d83fd32cd42d86b0d0e4936de9a674d6512934` | 3101 | 28,551 / fail | 0 / fail | Illustrate a second high-contact ACT / clean PACT mechanism case. Not rerun. |
| 4 | Routine | `05c9a8fa2688509c45538819605be47112a1d84b82f9d80748f74578bc4191c1` | 3101 | 0 / success | 0 / success | Show that both policies can complete a routine clean instance. Not rerun. |
| 5 | Counterexample | `356e8f7dc4ca1c9a126eeec6bc169a7664826492c05724e3392433fe92437396` | 3101 | 10,477 / fail | 27,852 / fail | Preserve the required case where PACT is worse rather than showing only favorable examples. Not rerun. |

## Render-only contract audit

The third-person view was implemented as a direct MuJoCo free-camera render using a fixed
robot-base-relative pose. It was not registered with the camera manager, did not add an observation
key, and was not named in the policy configuration. ACT remained configured for wrist RGB and qpos;
PACT remained configured for wrist RGB, qpos, and the 40 skin streams. The probe result records
`policy_camera_names: ["wrist_camera"]`, 901 rendered frames, and zero added RNG calls. The policy
check was corrected after the probe to normalize the evaluator's one-element observation wrapper
before recording observation key names; this metadata-only correction does not change the failed
scientific equality check and no second probe was run.

## Machine-readable record

The self-hashed manifest is
`diagnostics_output/pact_contact_endpoint/qualitative_video_manifest.json`. Its status is
`aborted_determinism_mismatch`, its five frozen selections are unchanged, it records zero completed
paired videos, and it binds the unpublished probe and external determinism record by SHA-256.

Manifest self-hash: `709a93881475a50b888bb0df09ccc8d3c0e880eedbbd4a5b23dfea07455f615d`
