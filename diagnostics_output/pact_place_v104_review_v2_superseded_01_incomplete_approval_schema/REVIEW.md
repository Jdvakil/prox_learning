# V10.4 review-v2 packet

Three production clean successes and three diagnostic negative controls.
Nothing in this packet is a new episode. The six production episodes were
run under V10.4-v1 and are reused byte-for-byte through the provenance
bridge; the controls are separately compiled diagnostic scenes and are
**not** production geometry.

## Videos

| # | kind | label | frames | duration | window |
|---|---|---|---|---|---|
| 1 | production_clean_success | `success_00_left.mp4` | 543 | 35.84 s | complete retained trajectory |
| 2 | production_clean_success | `success_03_right.mp4` | 456 | 30.10 s | complete retained trajectory |
| 3 | production_clean_success | `success_04_left.mp4` | 629 | 41.51 s | complete retained trajectory |
| 4 | diagnostic_negative_control | `control_left_lobe_contact.mp4` | 50 | 3.30 s | frames 40-89 (trimmed) |
| 5 | diagnostic_negative_control | `control_right_lobe_contact.mp4` | 64 | 4.22 s | frames 197-260 (trimmed) |
| 6 | diagnostic_negative_control | `control_stem_contact.mp4` | 64 | 4.22 s | frames 164-227 (trimmed) |

All six are stride 1 at the 66 ms control period (15.1515 fps), true time, with an untinted wrist pane, a
third-person pane, and a pendant pane.

## Diagnostic controls

| control | component | source role | shift | penetration | max frame | limiting body |
|---|---|---|---|---|---|---|
| left_lobe_contact | `lobe_0` | 0 (left) | 0.175 m | 5.044 mm | 88 | `fr3_link7` |
| right_lobe_contact | `lobe_1` | 3 (right) | 0.132 m | 5.239 mm | 245 | `gripper/base` |
| stem_contact | `stem_0` | 0 (left) | 0.083 m | 5.455 mm | 212 | `fr3_link7` |

Each control rigidly translates the complete assembly inward along y on
a separately compiled scene, reloaded through the real task sampler and
confirmed compiled-static with enclosing bounds. At the certified frame
signed distance, analytic GJK, live `data.contact`, and the place audit's
`mounted_fixture` classification all agree.

The production scene XML is byte-identical before and after every control
(`01d8adf34808a9f419cb3a9d07668ec1069d3a5acfa8cb01885c622ea09876f7`).

## What this packet does not authorize

Every authorization field is false and `human_approval.json` is absent.
Phase 0 has not run. No episode, `env.step`, collection, training, or
evaluation occurred while this packet was built.

## Owner approval schema

To authorize Phase0-v2, write `human_approval.json` into
`diagnostics_output/pact_place_v104_review_v2/` yourself, containing exactly:

```json
{
  "decision": "approve_phase0",
  "created_by_agent": false,
  "reviewed_videos": [
    "success_00_left.mp4",
    "success_03_right.mp4",
    "success_04_left.mp4",
    "control_left_lobe_contact.mp4",
    "control_right_lobe_contact.mp4",
    "control_stem_contact.mp4"
  ],
  "contract_version_v2": "pact_place_v104_review_packet_v2",
  "control_certificates_sha256": "0f04dfc70f630b8c0ea57c8b5cc449fe3ba9e20d9fd7426501a126a107d2bdae",
  "executed_v1_contract_sha256": "eb8f1174142976561495827b4cd3a8609569465fbce23c7a46b4a53885fe875e",
  "executed_v1_implementation_sha256": "bd135e68303618ceefbe57f1ad8a6e6a5d81ae2d29930f641f04648d4847ec90",
  "gate_v2_implementation_sha256": "d270329f99fda8e8e75c2152fcbc435a9bb98d4c9193d882e6a20fe75efface0",
  "production_scene_sha256": "01d8adf34808a9f419cb3a9d07668ec1069d3a5acfa8cb01885c622ea09876f7",
  "provenance_bridge_sha256": "d06b73b1102c6556a67cc2d81fc6a781787d5b3d92833178dd3b9cad9d999433",
  "review_preflight_sha256": "520d955ea061aebe1d9873a5781154f8c8f87e688ca1e943d89a8844f11d04f6",
  "review_v2_implementation_sha256": "c926c5a9d11d0b92f9b5da18aced88cbaa33a5128c25891ccdc8163ac24f63db",
  "scene_metadata_sha256": "7df36c5e26364f9b5bd6da98e59108d7745c2dbd1270cc3ca73d307a656b809c",
  "scoped_production_sha256": "92a9eae33370a020be200fb44af27b18741949d4219a9a89128bc927f7665ece"
}
```

The verifier recomputes every one of these from file bytes rather than
trusting an embedded value, requires the video list to be exactly the six
names above with no extras, and refuses a record with
`created_by_agent: true`. A passing gate sets only `phase0_passed: true`;
collection, training, and evaluation stay unauthorized.

Then run:

```
python scripts/run_pact_place_v104_phase0_v2.py
```
