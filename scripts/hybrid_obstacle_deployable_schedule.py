#!/usr/bin/env python3
"""Predeclare the frozen 20-rollout deployable schedule.

Handoff step 12. Four development rows x ``ACT_PLUS_DEPLOYABLE_REFERENCE`` x five repeats
= 20 policy rollouts. The 20 ACT_ONLY and 20 ORACLE rollouts are **reused** after hash and
schema verification; a rerun of either requires stopping and reporting, not silent
replacement.

Frozen and hashed before any rollout runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REPEATS = 5
CONDITION = "ACT_PLUS_DEPLOYABLE_REFERENCE"


def canonical_hash(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--development-manifest", required=True, type=Path)
    ap.add_argument("--reference-manifest", required=True, type=Path)
    ap.add_argument("--act-only-root", required=True)
    ap.add_argument("--oracle-root", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    dev = json.loads(args.development_manifest.read_text())
    if dev.get("role") != "DEVELOPMENT_ONLY":
        raise SystemExit(f"refusing a manifest whose role is {dev.get('role')!r}")
    reference = json.loads(args.reference_manifest.read_text())
    rows = sorted(dev["rows"], key=lambda r: (not r["hazard_present"],
                                              r["predeclared_stratum_rank"]))
    if len(rows) != 4:
        raise SystemExit(f"expected 4 development rows, got {len(rows)}")

    entries = []
    for repeat in range(REPEATS):
        for row in rows:
            tag = f"cand{row['candidate_index']}_deployable_r{repeat}"
            entries.append({
                "execution_order": len(entries),
                "episode_id": row["episode_id"],
                "candidate_index": row["candidate_index"],
                "hazard_present": bool(row["hazard_present"]),
                "accepted_retry_index": row["accepted_retry_index"],
                "initial_state_sha256": row["initial_state_sha256"],
                "condition": CONDITION,
                "repeat_index": repeat,
                "privileged": False,
                "deployable": True,
                "tag": tag,
                "output_dir": f"{args.output_root.rstrip('/')}/{tag}",
                "frozen_act_only_baseline":
                    f"{args.act_only_root.rstrip('/')}/cand{row['candidate_index']}"
                    f"_act_only_r{repeat}",
                "frozen_oracle_baseline":
                    f"{args.oracle_root.rstrip('/')}/cand{row['candidate_index']}"
                    f"_oracle_r{repeat}",
            })
    if len(entries) != 20:
        raise SystemExit(f"schedule must hold exactly 20 rollouts, got {len(entries)}")

    schedule = {
        "schema": "hybrid_obstacle_deployable_schedule_v1",
        "condition": CONDITION,
        "controller_id": "DEPLOYABLE_POSTURE_REFERENCE_RESIDUAL_V1",
        "privileged": False,
        "deployable": True,
        "reference_type": reference["reference_type"],
        "reference_manifest_sha256": reference["manifest_sha256"],
        "tau": reference["tau"],
        "development_manifest_sha256": dev["manifest_sha256"],
        "development_manifest_role": dev["role"],
        "rows": 4,
        "repeats": REPEATS,
        "rollouts": len(entries),
        "rollout_budget": 20,
        "reused_baselines": {
            "act_only": {"rerun": False, "root": args.act_only_root,
                         "source_task": "hybrid_obstacle_raw_head_qualification"},
            "oracle": {"rerun": False, "root": args.oracle_root,
                       "source_task": "hybrid_obstacle_oracle_reference"},
            "rule": ("reused after hash and schema verification; a rerun requires stopping "
                     "and reporting rather than silently replacing them"),
        },
        "shadow_oracle": {"enabled": True, "privileged": True, "executed": False,
                          "note": "diagnostic only; never changes the deployable action"},
        "immutability": ("frozen before execution; no entry may be replaced, rerun or "
                         "added after an outcome is observed"),
        "confirmatory_rows_included": False,
        "entries": entries,
    }
    schedule["schedule_sha256"] = canonical_hash(schedule)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(schedule, indent=2, sort_keys=True) + "\n")

    print(f"rollouts       : {len(entries)} (budget 20)")
    print(f"reference      : {reference['reference_type']} tau={reference['tau']}")
    print(f"rows           : {[r['candidate_index'] for r in rows]}")
    print(f"schedule sha256: {schedule['schedule_sha256']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
