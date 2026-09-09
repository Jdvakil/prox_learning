# V10.10b grasp experiment: final root review

**STOPPED_AT_GATE_B — the bounded experiment is complete; the acquisition-sampling candidate is rejected.** The supervisor closed at 17:28 UTC on 2026-09-08. Root observed its actual exit code 0 at the 17:32 hourly check. No evaluation workers remain. The unchanged update60000 baselines remain selected.

All 12 continuation branches completed their prescribed 3,000 updates. Stage A passed. Stage B completed all 216 scheduled comparisons: 180 new rollouts and 36 frozen PACT measurements reused from Stage A. These are the same 12 exposed diagnostic scenes crossed with three checkpoints and both arms; the 36 outcomes per arm/variant are not 36 independent physical scenes. Stages C and D were not launched because the frozen Stage B gate failed. No final qualification claim is made.

| Arm | Variant | Success /36 | Collision-free success /36 | Exclusive pickup failures /36 | Hazard/clutter union frames |
|---|---|---:|---:|---:|---:|
| ACT | Frozen 60000 | 18 | 13 | 6 | 53,331 |
| ACT | Uniform 63000 | 18 | 13 | 7 | 50,282 |
| ACT | Acquisition 63000 | 18 | 11 | 3 | 62,179 |
| PACT | Frozen 60000 | 16 | 11 | 9 | 24,296 |
| PACT | Uniform 63000 | 22 | 14 | 7 | 17,371 |
| PACT | Acquisition 63000 | 20 | 13 | 8 | 32,362 |

Every row has 1,069,236 physics samples. Union frames count hazard OR clutter once. Exclusive pickup failure means touched-without-hold after applying the unchanged failure hierarchy; held-without-lift and other failure categories remain separate in the full report.

The candidate gained four PACT successes over the frozen baseline, but lost two against equal-update uniform training. PACT pickup failures decreased by only one versus frozen (the requirement was at least three) and increased by one versus uniform. Candidate PACT union contact frames increased 33.2% versus frozen and 86.3% versus uniform, exceeding the allowed 10%. Candidate ACT lost two collision-free successes against each control and increased union frames 16.6% and 23.7%, respectively. These independent failures require stopping even though some individual cases improved.

| Arm | Seed | Frozen success/CFTS | Uniform success/CFTS | Candidate success/CFTS |
|---|---:|---:|---:|---:|
| ACT | 3103 | 6/12; 4/12 | 5/12; 4/12 | 6/12; 3/12 |
| ACT | 3104 | 6/12; 4/12 | 7/12; 5/12 | 7/12; 5/12 |
| ACT | 3105 | 6/12; 5/12 | 6/12; 4/12 | 5/12; 3/12 |
| PACT | 3103 | 3/12; 2/12 | 7/12; 6/12 | 5/12; 4/12 |
| PACT | 3104 | 6/12; 4/12 | 7/12; 4/12 | 8/12; 4/12 |
| PACT | 3105 | 7/12; 5/12 | 8/12; 4/12 | 7/12; 5/12 |

PACT candidate seed3103 lost two successes and two collision-free successes versus uniform; seed3105 lost one success, violating its no-loss requirement. Full per-seed contact, failure-category and paired counts are retained in [the gate](gates/B.json) and [the supervisor report](EVAL.md).

| Candidate versus control | PACT success wins/losses | ACT success wins/losses | PACT acquisition-repair count |
|---|---:|---:|---:|
| Frozen 60000 | 8/4 | 3/3 | 5 |
| Uniform 63000 | 1/3 | 3/3 | 1 |

Repair counts require a control pickup failure, candidate actual lift of at least 1 cm, and at least one continuous bilateral-contact interval of 1 second. The gate required three repairs against each control. Five met the count against frozen, but only one against uniform. Contact/lift chronologies are retained; the existence of both endpoints alone does not establish that the contact interval caused the lift.

The intervention changed exposure as intended: acquisition-window starts rose from approximately 12.3–12.6% under uniform sampling to 34.3–34.5%, with exactly matched ACT/PACT start streams for every seed/variant. Increased exposure did not satisfy the required net acquisition and contact protections. Uniform performed better than the candidate on PACT task success in this exposed screen, but it has not passed fresh development or final evaluation and is not promoted.

Execution accounting: 240 valid new rollouts (A1 12, A2 48, B 180); 12 completed training branches; 36,000 committed continuation updates. Two diagnosed infrastructure failures consumed two training retries. The second discarded/replayed 240 unsaved optimizer updates, so total optimizer work exceeded the committed budget by those 240 updates. One later supervisor-monitor fork failure was repaired through operational amendment 06 without repeating completed evaluations. All attempts and amendments remain recorded.

Verification passed: the supervisor recomputed all 240 new rollouts from raw data, rechecked included initial-state groups and protected hashes, and reconciled 252 valid zero-exit records. Root independently read all 216 Stage B raw trajectories/contact arrays, checked the complete scene × checkpoint × arm × variant matrix, and exactly reproduced every saved primary endpoint and pooled/per-seed gate aggregate. Root also rehashed all six frozen baseline models. Existing root EVAL.md and protected historical artifacts remain unchanged.

Hourly manual checks at 14:36, 15:32, 16:32 and 17:32 UTC are recorded in [hourly_checks.jsonl](root_review/hourly_checks.jsonl). They observed Stage B at 82, 120, 152 and 180 completed new rollouts. The last check confirmed terminal closure, so hourly monitoring is finished.

Evidence: [independent root review](root_review/final_review.json), [review source](root_review/final_review.py), [final verification](final_verification.json), [parent closure](parent_closure.json), [training exposure](root_review/realized_training_exposure.json).
