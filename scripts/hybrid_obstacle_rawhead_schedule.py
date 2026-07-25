#!/usr/bin/env python3
"""Predeclare the frozen 40-rollout repeated schedule.

Handoff step 8. Four development rows x two conditions x five repeats = 40 primary
rollouts, plus one optional privileged oracle rollout on the first hazard-present
row.

Condition order is balanced deterministically so neither condition is always first:

    even repeat index -> ACT_ONLY then ACT_PLUS_RAW_HEAD
    odd  repeat index -> ACT_PLUS_RAW_HEAD then ACT_ONLY

The schedule is written and hashed before any rollout runs. Rows are never
replaced, rerun or added after an outcome is observed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

PRIMARY_CONDITIONS = ("ACT_ONLY", "ACT_PLUS_RAW_HEAD")
REPEATS = 5


def canonical_hash(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--development-manifest", required=True, type=Path)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--include-oracle", action="store_true")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    dev = json.loads(args.development_manifest.read_text())
    rows = sorted(dev["rows"], key=lambda r: (not r["hazard_present"],
                                              r["predeclared_stratum_rank"]))
    if len(rows) != 4:
        raise SystemExit(f"expected 4 development rows, got {len(rows)}")

    entries = []
    order = 0
    for repeat in range(REPEATS):
        conds = (PRIMARY_CONDITIONS if repeat % 2 == 0
                 else tuple(reversed(PRIMARY_CONDITIONS)))
        for row in rows:
            for cond in conds:
                tag = f"cand{row['candidate_index']}_{cond.lower()}_r{repeat}"
                entries.append({
                    "execution_order": order,
                    "episode_id": row["episode_id"],
                    "candidate_index": row["candidate_index"],
                    "hazard_present": bool(row["hazard_present"]),
                    "accepted_retry_index": row["accepted_retry_index"],
                    "initial_state_sha256": row["initial_state_sha256"],
                    "condition": cond,
                    "repeat_index": repeat,
                    "privileged": False,
                    "tag": tag,
                    "output_dir": f"{args.output_root.rstrip('/')}/{tag}",
                })
                order += 1

    if len(entries) != 40:
        raise SystemExit(f"schedule must hold exactly 40 primary rollouts, got {len(entries)}")

    oracle = None
    if args.include_oracle:
        first_present = next(r for r in rows if r["hazard_present"])
        tag = f"cand{first_present['candidate_index']}_oracle_parked_r0"
        oracle = {
            "execution_order": order,
            "episode_id": first_present["episode_id"],
            "candidate_index": first_present["candidate_index"],
            "hazard_present": True,
            "accepted_retry_index": first_present["accepted_retry_index"],
            "initial_state_sha256": first_present["initial_state_sha256"],
            "condition": "ORACLE_PARKED_REFERENCE",
            "repeat_index": 0,
            "privileged": True,
            "privileged_note": ("uses privileged simulation information; excluded from every "
                                "readiness metric and headline table; never deployable"),
            "tag": tag,
            "output_dir": f"{args.output_root.rstrip('/')}/{tag}",
        }

    # order balance check
    firsts = {}
    for e in entries:
        key = (e["candidate_index"], e["repeat_index"])
        firsts.setdefault(key, e["condition"])
    counts = {c: sum(1 for v in firsts.values() if v == c) for c in PRIMARY_CONDITIONS}

    sched = {
        "schema": "hybrid_obstacle_rawhead_schedule_v1",
        "development_manifest_sha256": dev["manifest_sha256"],
        "development_manifest_label": dev["label"],
        "rows": 4, "conditions": list(PRIMARY_CONDITIONS), "repeats": REPEATS,
        "primary_rollouts": len(entries),
        "primary_rollout_budget": 40,
        "oracle_rollouts": 1 if oracle else 0,
        "oracle_rollout_budget": 1,
        "order_balance": {
            "rule": ("even repeat index runs ACT_ONLY first, odd runs ACT_PLUS_RAW_HEAD first"),
            "first_condition_counts": counts,
            "balanced": abs(counts["ACT_ONLY"] - counts["ACT_PLUS_RAW_HEAD"]) <= 4,
        },
        "immutability": ("frozen before execution; no row may be replaced, rerun or added after an "
                         "outcome is observed"),
        "entries": entries,
        "oracle_entry": oracle,
    }
    sched["schedule_sha256"] = canonical_hash(sched)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(sched, indent=2, sort_keys=True) + "\n")

    print(f"primary rollouts : {len(entries)} (budget {sched['primary_rollout_budget']})")
    print(f"oracle rollouts  : {sched['oracle_rollouts']} (budget 1)")
    print(f"first-condition  : {counts}")
    print(f"schedule sha256  : {sched['schedule_sha256']}")
    print("first 6 entries:")
    for e in entries[:6]:
        print(f"  {e['execution_order']:3d} cand{e['candidate_index']:>4} "
              f"{e['condition']:<18} r{e['repeat_index']} -> {e['tag']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
