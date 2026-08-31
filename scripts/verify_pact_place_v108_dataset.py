#!/usr/bin/env python3
"""Independent final gate over the V10.8 dataset.

Re-derives everything rather than trusting the collection summary: it re-reads
every accepted episode from disk, revalidates the full trainable schema, checks
quota satisfaction cell by cell, and refuses corrupt, partial, duplicate or
non-clean episodes. Read-only.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "scripts", ROOT / "submodules" / "molmospaces"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from pact_place_v108_contract import (  # noqa: E402
    COLLECTION_ROOT, DATASET_ROOT, TARGET_SUCCESSES, cell_key, quota_totals,
    quotas, sha256_file,
)
from run_pact_place_v108_collect import Ledger, validate_trainable  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection-root", type=Path,
                        default=ROOT / COLLECTION_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / DATASET_ROOT)
    args = parser.parse_args()
    collection_root = args.collection_root.resolve()
    rows_root = args.dataset_root.resolve() / "rows"
    problems: list[str] = []

    ledger = Ledger(collection_root / "ledger.jsonl").read()
    accepted_records = [r for r in ledger if r.get("accepted")]

    seen_ids = Counter(r["attempt_id"] for r in accepted_records)
    duplicates = sorted(a for a, n in seen_ids.items() if n > 1)
    if duplicates:
        problems.append(f"duplicate accepted attempt ids: {duplicates[:4]}")

    episodes: list[dict[str, Any]] = []
    for record in accepted_records:
        directory = rows_root / record["attempt_id"][:16]
        entry: dict[str, Any] = {
            "attempt_id": record["attempt_id"], "cell": record["cell"],
            "dir": str(directory.relative_to(ROOT)) if directory.exists() else None,
        }
        if not directory.is_dir():
            problems.append(f"missing episode directory: {record['attempt_id'][:16]}")
            entry["present"] = False
            episodes.append(entry)
            continue
        if not record.get("clean_success"):
            problems.append(f"accepted a non-clean episode: {record['attempt_id'][:16]}")
        validation = validate_trainable(directory)
        entry.update({
            "present": True, "schema_passed": validation["passed"],
            "n_frames": validation["detail"].get("n_frames"),
            "n_proximity": validation["detail"].get("n_proximity"),
            "wrist_frames": validation["detail"].get("wrist_frames"),
        })
        if not validation["passed"]:
            problems.append(
                f"{record['attempt_id'][:16]}: {validation['problems'][:2]}")
        h5 = directory / "trajectory.h5"
        if h5.is_file():
            observed = sha256_file(h5)
            entry["h5_sha256"] = observed
            entry["h5_bytes"] = h5.stat().st_size
            if record.get("trajectory_h5_sha256") and observed != record[
                "trajectory_h5_sha256"
            ]:
                problems.append(f"h5 drifted: {record['attempt_id'][:16]}")
        else:
            problems.append(f"missing trajectory.h5: {record['attempt_id'][:16]}")
        episodes.append(entry)

    by_cell = Counter(r["cell"] for r in accepted_records)
    q = quotas()
    shortfall = {k: q[k] - by_cell.get(k, 0) for k in q if by_cell.get(k, 0) < q[k]}
    excess = {k: by_cell.get(k, 0) - q[k] for k in q if by_cell.get(k, 0) > q[k]}
    if shortfall:
        problems.append(f"cells short of quota: {shortfall}")
    if excess:
        problems.append(f"cells over quota: {excess}")
    if len(accepted_records) != TARGET_SUCCESSES:
        problems.append(
            f"{len(accepted_records)} accepted, need {TARGET_SUCCESSES}")

    totals = quota_totals()
    by_family = Counter(r["family_id"] for r in accepted_records)
    by_side = Counter(r["intrusion_side"] for r in accepted_records)
    by_pose = Counter(r["pose_id"] for r in accepted_records)
    for label, observed, expected in (
        ("family", dict(by_family), totals["by_family"]),
        ("side", dict(by_side), totals["by_side"]),
        ("pose", dict(by_pose), totals["by_pose"]),
    ):
        if observed != expected:
            problems.append(f"{label} totals {observed} != {expected}")

    contact_rows = [r for r in accepted_records
                    if (r.get("pendant_contact_frames") or 0) > 0]
    if contact_rows:
        problems.append(f"{len(contact_rows)} accepted rows show pendant contact")

    total_bytes = sum(e.get("h5_bytes", 0) for e in episodes)
    print(json.dumps({
        "verified": not problems,
        "n_problems": len(problems),
        "problems": problems[:12],
        "n_accepted": len(accepted_records),
        "n_ledger_records": len(ledger),
        "target": TARGET_SUCCESSES,
        "by_family": dict(by_family), "by_side": dict(by_side),
        "by_pose": dict(by_pose),
        "cells_at_quota": sum(1 for k in q if by_cell.get(k, 0) == q[k]),
        "n_cells": len(q),
        "accepted_rows_with_pendant_contact": len(contact_rows),
        "dataset_h5_gb": round(total_bytes / 1e9, 3),
    }, indent=2))
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
