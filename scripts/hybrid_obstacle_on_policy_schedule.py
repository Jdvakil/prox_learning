#!/usr/bin/env python3
"""Predeclare the frozen on-policy rollout schedules.

Handoff steps 4 and 7 and 12. Three schedules, each frozen and hashed before anything
executes:

``labelling``  100 reference rows x {ACT_ONLY_ON_POLICY, ACT_PLUS_ORACLE_ON_POLICY} = 200
``learner``    the 64 reference-training rows x ACT_PLUS_REFERENCE_V2 = 64
``live``       the 4 development rows x ACT_PLUS_REFERENCE_V2 x 5 repeats = 20

Condition order in the labelling schedule is balanced deterministically by row rank --
even rank runs ACT_ONLY first, odd rank runs the oracle first -- so neither condition is
systematically first.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

LABELLING_CONDITIONS = ("ACT_ONLY_ON_POLICY", "ACT_PLUS_ORACLE_ON_POLICY")
LEARNER_CONDITION = "ACT_PLUS_REFERENCE_V2"
LIVE_REPEATS = 5


def canonical_hash(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--partition", required=True, type=Path)
    ap.add_argument("--development-manifest", required=True, type=Path)
    ap.add_argument("--which", required=True, choices=("labelling", "learner", "live"))
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    partition = json.loads(args.partition.read_text())
    development = json.loads(args.development_manifest.read_text())
    if development.get("role") != "DEVELOPMENT_ONLY":
        raise SystemExit(f"refusing a manifest whose role is {development.get('role')!r}")

    entries: list[dict] = []
    if args.which == "labelling":
        rows = [row for name in ("reference_train", "reference_calibration",
                                 "reference_validation", "offline_reference_test")
                for row in partition["partitions"][name]]
        rows.sort(key=lambda r: (r["partition"], not r["hazard_present"],
                                 r["predeclared_stratum_rank"]))
        for rank, row in enumerate(rows):
            order = (LABELLING_CONDITIONS if rank % 2 == 0
                     else tuple(reversed(LABELLING_CONDITIONS)))
            for condition in order:
                tag = f"{row['partition']}_cand{row['candidate_index']}_{condition.lower()}"
                entries.append({
                    "execution_order": len(entries),
                    "row_rank": rank,
                    "episode_id": row["episode_id"],
                    "candidate_index": row["candidate_index"],
                    "hazard_present": row["hazard_present"],
                    "partition": row["partition"],
                    "condition": condition,
                    "repeat_index": 0,
                    "tag": tag,
                    "output_dir": f"{args.output_root.rstrip('/')}/{tag}",
                })
        expected = 200
    elif args.which == "learner":
        rows = sorted(partition["partitions"]["reference_train"],
                      key=lambda r: (not r["hazard_present"],
                                     r["predeclared_stratum_rank"]))
        for row in rows:
            tag = f"learner_cand{row['candidate_index']}"
            entries.append({
                "execution_order": len(entries),
                "episode_id": row["episode_id"],
                "candidate_index": row["candidate_index"],
                "hazard_present": row["hazard_present"],
                "partition": "reference_train",
                "condition": LEARNER_CONDITION,
                "repeat_index": 0,
                "tag": tag,
                "output_dir": f"{args.output_root.rstrip('/')}/{tag}",
            })
        expected = 64
    else:
        rows = sorted(development["rows"], key=lambda r: (not r["hazard_present"],
                                                          r["predeclared_stratum_rank"]))
        for repeat in range(LIVE_REPEATS):
            for row in rows:
                tag = f"cand{row['candidate_index']}_referencev2_r{repeat}"
                entries.append({
                    "execution_order": len(entries),
                    "episode_id": row["episode_id"],
                    "candidate_index": row["candidate_index"],
                    "hazard_present": bool(row["hazard_present"]),
                    "partition": "development4",
                    "condition": LEARNER_CONDITION,
                    "repeat_index": repeat,
                    "accepted_retry_index": row["accepted_retry_index"],
                    "initial_state_sha256": row["initial_state_sha256"],
                    "tag": tag,
                    "output_dir": f"{args.output_root.rstrip('/')}/{tag}",
                    "frozen_act_only_baseline":
                        f"/root/act_retrain_assets/rawhead_dev_v1/"
                        f"cand{row['candidate_index']}_act_only_r{repeat}",
                    "frozen_oracle_baseline":
                        f"/root/act_retrain_assets/oracle_dev_v1/"
                        f"cand{row['candidate_index']}_oracle_r{repeat}",
                    "frozen_v1_baseline":
                        f"/root/act_retrain_assets/deployable_dev_v1/"
                        f"cand{row['candidate_index']}_deployable_r{repeat}",
                })
        expected = 20

    if len(entries) != expected:
        raise SystemExit(f"{args.which} schedule must hold {expected} entries, "
                         f"got {len(entries)}")

    development_ids = {r["episode_id"] for r in development["rows"]}
    scheduled = {e["episode_id"] for e in entries}
    if args.which in ("labelling", "learner") and scheduled & development_ids:
        raise SystemExit("a development row leaked into a reference-model schedule")
    if args.which == "live" and scheduled != development_ids:
        raise SystemExit("the live schedule must cover exactly the development rows")

    firsts: dict[str, str] = {}
    for entry in entries:
        firsts.setdefault(entry["episode_id"], entry["condition"])
    counts = {c: sum(1 for v in firsts.values() if v == c)
              for c in {e["condition"] for e in entries}}

    schedule = {
        "schema": f"hybrid_obstacle_on_policy_{args.which}_schedule_v2",
        "which": args.which,
        "partition_sha256": partition["partition_sha256"],
        "development_manifest_sha256": development["manifest_sha256"],
        "rollouts": len(entries),
        "rollout_budget": expected,
        "conditions": sorted({e["condition"] for e in entries}),
        "order_balance": {
            "rule": ("even row rank runs ACT_ONLY_ON_POLICY first, odd runs the oracle "
                     "first" if args.which == "labelling" else "single condition"),
            "first_condition_counts": counts,
        },
        "immutability": ("frozen and hashed before execution; no entry may be replaced, "
                         "rerun or added after an outcome is observed"),
        "confirmatory_rows_included": False,
        "development_rows_included": args.which == "live",
        "entries": entries,
    }
    schedule["schedule_sha256"] = canonical_hash(schedule)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(schedule, indent=2, sort_keys=True) + "\n")

    print(f"{args.which}: {len(entries)} rollouts (budget {expected})")
    print(f"  first-condition counts {counts}")
    print(f"  schedule sha256 {schedule['schedule_sha256']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
