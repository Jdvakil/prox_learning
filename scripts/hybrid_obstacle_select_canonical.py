#!/usr/bin/env python3
"""Deterministic canonical-selection manifest builder (version v2).

Selects a stratified canonical subset of successful trajectories from an
immutable datagen run. This writes a *new* versioned manifest; it never edits an
earlier manifest and never touches the source collection.

Algorithm (fixed in advance, no downstream metric participates)
--------------------------------------------------------------
1. Drop trajectories whose ``fail[-1]`` is set.
2. Collapse exact-replica classes: trajectories with identical content SHA-256
   are the same episode stored more than once, so exactly one representative --
   the lexicographically smallest trajectory id -- is eligible. This is a
   correctness requirement, not a quality filter.
3. Group the eligible representatives by hazard label.
4. Within each hazard group, order by
   ``SHA-256(selection_seed || "\\0" || trajectory_id || "\\0" || source_file_hash)``.
5. Walk each hazard group in a fixed round-robin over house strata (houses in
   sorted order, each contributing its next hash-ordered candidate) until the
   stratum quota is met. The round robin is applied before the hash ordering is
   consumed, so worker/house balance is deterministic and decided in advance.
6. Take the quota from each hazard group.

Selection never consults clearance, collisions, trajectory length, sensor
activation, action statistics, or any model output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

SCHEMA = "hybrid_obstacle_canonical_selection_v2"


def order_key(selection_seed: str, trajectory_id: str, source_hash: str) -> str:
    payload = f"{selection_seed}\0{trajectory_id}\0{source_hash}".encode()
    return hashlib.sha256(payload).hexdigest()


def canonical_hash(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def round_robin(candidates: list[dict[str, Any]], quota: int) -> list[dict[str, Any]]:
    """Take ``quota`` candidates, cycling houses in sorted order."""
    by_house: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in candidates:
        by_house[c["house"]].append(c)
    for house in by_house:
        by_house[house].sort(key=lambda c: c["order_key"])
    houses = sorted(by_house)
    picked: list[dict[str, Any]] = []
    cursor = {h: 0 for h in houses}
    while len(picked) < quota:
        progressed = False
        for house in houses:
            if len(picked) >= quota:
                break
            idx = cursor[house]
            if idx < len(by_house[house]):
                picked.append(by_house[house][idx])
                cursor[house] = idx + 1
                progressed = True
        if not progressed:
            break
    return picked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--integrity_report", required=True)
    parser.add_argument("--selection_seed", default="hybrid-obstacle-canonical-v2")
    parser.add_argument("--hazard_quota", type=int, default=75)
    parser.add_argument("--clear_quota", type=int, default=25)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--allow_replicas",
        action="store_true",
        help="DANGEROUS: permit multiple trajectories from the same identical-content "
        "class. Off by default; replicas are not independent episodes.",
    )
    args = parser.parse_args()

    report = json.loads(Path(args.integrity_report).read_text())
    rows = report["trajectories_detail"]

    successful = [r for r in rows if r["successful"]]
    by_content: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in successful:
        by_content[r["content_sha256"]].append(r)

    eligible: list[dict[str, Any]] = []
    suppressed: list[dict[str, Any]] = []
    for content, group in by_content.items():
        group = sorted(group, key=lambda r: r["trajectory_id"])
        keep = group if args.allow_replicas else group[:1]
        for r in keep:
            eligible.append(r)
        for r in group[len(keep) :]:
            suppressed.append(
                {
                    "trajectory_id": r["trajectory_id"],
                    "content_sha256": content,
                    "reason": "exact replica of " + group[0]["trajectory_id"],
                }
            )

    for r in eligible:
        r["order_key"] = order_key(
            args.selection_seed, r["trajectory_id"], r["source_h5_sha256"]
        )

    hazard = sorted(
        (r for r in eligible if r["hazard_recorded"]), key=lambda r: r["order_key"]
    )
    clear = sorted(
        (r for r in eligible if not r["hazard_recorded"]), key=lambda r: r["order_key"]
    )

    feasible = len(hazard) >= args.hazard_quota and len(clear) >= args.clear_quota

    selected_hazard = round_robin(hazard, args.hazard_quota) if feasible else []
    selected_clear = round_robin(clear, args.clear_quota) if feasible else []
    selected = selected_hazard + selected_clear

    def compose(sample: list[dict[str, Any]], key: str) -> dict[str, int]:
        out: dict[str, int] = defaultdict(int)
        for r in sample:
            out[str(r[key])] += 1
        return dict(sorted(out.items()))

    selected_ids = {r["trajectory_id"] for r in selected}
    excluded = [
        {
            "trajectory_id": r["trajectory_id"],
            "reason": (
                "failed (fail[-1] set)"
                if not r["successful"]
                else "exact replica suppressed"
                if any(s["trajectory_id"] == r["trajectory_id"] for s in suppressed)
                else "not drawn within stratum quota"
            ),
            "hazard_present": r["hazard_recorded"],
            "successful": r["successful"],
        }
        for r in rows
        if r["trajectory_id"] not in selected_ids
    ]

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA,
        "supersedes": "clean_retraining_manifest.json selection_order "
        "(sorted source H5 path, then numeric traj index, first 100 accepted)",
        "source_run_dir": report["run_dir"],
        "source_collection_id": report["collection_id"],
        "source_content_tree_sha256": (
            "09c98aee08d015b3a561b08674415df9a4ed398186940207f41ef384251cdf24"
        ),
        "selection_seed": args.selection_seed,
        "selector_source_sha256": sha256_file(Path(__file__).resolve()),
        "integrity_report_sha256": sha256_file(Path(args.integrity_report).resolve()),
        "target_composition": {
            "total": args.hazard_quota + args.clear_quota,
            "hazard_present": args.hazard_quota,
            "hazard_absent": args.clear_quota,
        },
        "algorithm": {
            "step_1": "drop fail[-1] trajectories",
            "step_2": "collapse identical-content replica classes to one representative "
            "(lexicographically smallest trajectory id)",
            "step_3": "group by hazard label",
            "step_4": "order within group by SHA-256(selection_seed || trajectory_id || "
            "source_file_hash)",
            "step_5": "fixed round-robin over houses in sorted order, consuming each "
            "house's hash-ordered candidates",
            "step_6": "take the stratum quota",
            "excluded_signals": [
                "clearance",
                "collisions",
                "trajectory length or quality",
                "sensor activation",
                "action statistics",
                "any downstream model metric",
            ],
            "granularity": "trajectory-level only",
        },
        "availability": {
            "stored_trajectories": len(rows),
            "successful_trajectories": len(successful),
            "distinct_successful_after_replica_collapse": len(eligible),
            "eligible_hazard_present": len(hazard),
            "eligible_hazard_absent": len(clear),
            "replicas_suppressed": len(suppressed),
            "allow_replicas": args.allow_replicas,
        },
        "feasible": feasible,
        "shortfall": {
            "hazard_present": max(0, args.hazard_quota - len(hazard)),
            "hazard_absent": max(0, args.clear_quota - len(clear)),
            "total": max(0, args.hazard_quota + args.clear_quota - len(eligible)),
        },
        "selected": [
            {
                "canonical_index": i,
                "trajectory_id": r["trajectory_id"],
                "house": r["house"],
                "worker": r["worker"],
                "traj_key": r["traj_key"],
                "source_h5": r["source_h5"],
                "source_h5_sha256": r["source_h5_sha256"],
                "content_sha256": r["content_sha256"],
                "hazard_present": r["hazard_recorded"],
                "frames": r["frames"],
                "order_key": r["order_key"],
                "reason": "selected: hazard stratum quota, round-robin over houses, "
                "hash-ordered",
            }
            for i, r in enumerate(selected)
        ],
        "selected_composition": {
            "total": len(selected),
            "hazard_present": sum(1 for r in selected if r["hazard_recorded"]),
            "hazard_absent": sum(1 for r in selected if not r["hazard_recorded"]),
            "by_house": compose(selected, "house") if selected else {},
            "by_worker": compose(selected, "worker") if selected else {},
        },
        "replicas_suppressed_detail": suppressed,
        "excluded": excluded,
    }
    manifest["manifest_sha256"] = canonical_hash(
        {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    )

    out = Path(args.output)
    if out.exists():
        raise SystemExit(f"refusing to overwrite existing manifest: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    summary = {
        "output": str(out),
        "feasible": feasible,
        "eligible_hazard_present": len(hazard),
        "eligible_hazard_absent": len(clear),
        "shortfall": manifest["shortfall"],
        "selected": len(selected),
        "manifest_sha256": manifest["manifest_sha256"],
    }
    json.dump(summary, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    if not feasible:
        sys.exit(3)


if __name__ == "__main__":
    main()
