# Seed 3105 — full 50-pair evaluation

Verdict: **INCOMPLETE**.

Paused: disk/deadline/manual guard

Updated: 2026-09-07T19:58:15.944450+00:00. ACT and PACT use their existing 60,000-update checkpoints, history 100, and the original frozen final-3105 manifest.


The user selected seed 3105 for expansion after reviewing its initial 16-pair result. This is a follow-up on that selected seed; the earlier three-seed report remains unchanged. The original development gate remains missed.

All 50 pairs use the original frozen instances and full initial-state/RGB pairing audits. Prior completed pairs are adopted without rerunning them. Report all failures; no outcome-based replacement or checkpoint selection is permitted.

Slots 08/09 are absent. The 40 proximity sensors cover link1–link6; the prior audit found the inbound vessel outside sensor visibility in 7/8 variants. A PACT advantage does not establish clutter sensing. All `authorizes_*` flags remain false.

Artifacts: `contract.json`, `analysis.json`, `adoption.json`; canonical full-stage ledger and pairing audits under `../../evaluation/`.
