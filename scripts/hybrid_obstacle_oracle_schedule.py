#!/usr/bin/env python3
"""Predeclare the frozen 20-rollout oracle schedule.

Handoff step 10. Four development rows x one condition (ACT_PLUS_ORACLE) x five repeats
= 20 privileged rollouts. The five ACT_ONLY repeats per row are **not** rerun: they come
from the raw-head development task and are reused after hash and schema verification.

The schedule is written and hashed before any rollout runs. No entry may be replaced,
rerun or added after an outcome is observed, and a failed oracle execution is not retried.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REPEATS = 5
CONDITION = "ACT_PLUS_ORACLE"


def canonical_hash(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--development-manifest", required=True, type=Path)
    ap.add_argument("--baseline-root", required=True,
                    help="directory holding the frozen ACT_ONLY raw-head rollouts")
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    dev = json.loads(args.development_manifest.read_text())
    if dev.get("role") != "DEVELOPMENT_ONLY":
        raise SystemExit(f"refusing a manifest whose role is {dev.get('role')!r}")
    rows = sorted(dev["rows"], key=lambda r: (not r["hazard_present"],
                                              r["predeclared_stratum_rank"]))
    if len(rows) != 4:
        raise SystemExit(f"expected 4 development rows, got {len(rows)}")

    entries = []
    for repeat in range(REPEATS):
        for row in rows:
            tag = f"cand{row['candidate_index']}_oracle_r{repeat}"
            entries.append({
                "execution_order": len(entries),
                "episode_id": row["episode_id"],
                "candidate_index": row["candidate_index"],
                "hazard_present": bool(row["hazard_present"]),
                "accepted_retry_index": row["accepted_retry_index"],
                "initial_state_sha256": row["initial_state_sha256"],
                "condition": CONDITION,
                "repeat_index": repeat,
                "privileged": True,
                "deployable": False,
                "tag": tag,
                "output_dir": f"{args.output_root.rstrip('/')}/{tag}",
                "frozen_act_only_baseline":
                    f"{args.baseline_root.rstrip('/')}/cand{row['candidate_index']}"
                    f"_act_only_r{repeat}",
            })
    if len(entries) != 20:
        raise SystemExit(f"schedule must hold exactly 20 oracle rollouts, got {len(entries)}")

    schedule = {
        "schema": "hybrid_obstacle_oracle_schedule_v1",
        "reference_id": "ORACLE_PARKED_REFERENCE_V1",
        "controller_id": "ORACLE_PARKED_RESIDUAL_V1",
        "privileged": True,
        "deployable": False,
        "development_manifest_sha256": dev["manifest_sha256"],
        "development_manifest_label": dev["label"],
        "development_manifest_role": dev["role"],
        "rows": 4,
        "conditions": [CONDITION],
        "repeats": REPEATS,
        "oracle_rollouts": len(entries),
        "oracle_rollout_budget": 20,
        "act_only_compatibility_rollout_budget": 1,
        "act_only_baselines": {
            "rerun": False,
            "source_task": "hybrid_obstacle_raw_head_qualification",
            "root": args.baseline_root,
            "rule": ("reused after hash and schema verification; rerun only if that "
                     "verification fails"),
        },
        "immutability": ("frozen before execution; no entry may be replaced, rerun or added "
                         "after an outcome is observed, and a failed oracle execution is not "
                         "retried"),
        "confirmatory_rows_included": False,
        "entries": entries,
    }
    schedule["schedule_sha256"] = canonical_hash(schedule)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(schedule, indent=2, sort_keys=True) + "\n")

    print(f"oracle rollouts : {len(entries)} (budget 20)")
    print(f"rows            : {[r['candidate_index'] for r in rows]}")
    print(f"schedule sha256 : {schedule['schedule_sha256']}")
    for entry in entries[:4]:
        print(f"  {entry['execution_order']:3d} cand{entry['candidate_index']:>4} "
              f"r{entry['repeat_index']} -> {entry['tag']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
