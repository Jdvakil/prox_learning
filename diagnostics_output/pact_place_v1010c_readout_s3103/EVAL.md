# Seed 3103: main fine-tuned encoder versus frozen PACT

Completed 60,000 committed PACT updates and 50 matched readout evaluations. Independent raw verification confirmed all 100 frozen/new endpoints. The original frozen 50 outcomes were retained; the extra frozen binding smoke is outside both denominators.

| Endpoint | Frozen 32-D | Fine-tuned 128-D |
|---|---:|---:|
| Task success | 20/50 (40%) | 27/50 (54%) |
| Collision-free task success | 16/50 (32%) | 24/50 (48%) |
| Exclusive pickup failures | 16/50 | 8/50 |
| Hazard/clutter union frames | 59,086 | 29,000 |
| Hazard-bar frames | 18,183 | 20,148 |
| Clutter frames | 41,990 | 9,251 |

The fine-tuned method improved both success metrics and halved pickup failures. Combined hazard/clutter contact frames fell 50.9%, while hazard-bar frames increased 10.8%. Task success changed on 17 scenes: 12 gains and five losses. Collision-free success had 11 gains and three losses. Pickup failures disappeared on 11 scenes and appeared on three.

This tests the full main method—fine-tuning, 128-D readout and minimum pooling—on the same 50 exposed seed-3103 scenes. It does not isolate unfreezing or establish results on other seeds. Two training infrastructure interruptions and a RAM-related evaluation pause are disclosed in the full review; all 50 evaluations completed successfully.

[Full review](FINAL_REVIEW.md) · [Per-scene CSV](paired_scene_comparison.csv) · [Raw comparison](comparison.json) · [Independent verification](root_review/final_review.json)
