#!/usr/bin/env python3
"""Freeze the nested activity-gate partition over the 64 reference-training episodes.

Handoff step 6. The split is deterministic from a predeclared hazard-stratum rank: episodes
are sorted by identity within each hazard stratum and dealt out in a fixed order, so the
partition is reproducible from the manifest hashes alone and nothing about gate performance
could have influenced it.

The previous calibration, validation and offline-test trajectories stay outside this
partition entirely. They are reused diagnostics and may not enter gate fitting or threshold
calibration. development4 and confirmatory41 are excluded by construction: neither appears
in the paired dataset.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from causal_parked_skin import threshold as thr

SPLITS = (
    ("gate_training", 30, 10),
    ("checkpoint_validation", 6, 2),
    ("threshold_calibration", 6, 2),
    ("nested_offline_evaluation", 6, 2),
)
SOURCE_PARTITION = "reference_train"
EXCLUDED = ("reference_validation", "reference_calibration", "offline_reference_test")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--dev4", required=True, type=Path)
    ap.add_argument("--conf41", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text())
    dev4 = json.loads(args.dev4.read_text())
    conf41 = json.loads(args.conf41.read_text())

    entries = [e for e in manifest["entries"] if e["partition"] == SOURCE_PARTITION]
    episodes: dict[str, dict] = {}
    for entry in entries:
        block = episodes.setdefault(entry["episode_id"], {
            "episode_id": entry["episode_id"], "hazard_present": entry["hazard_present"],
            "trajectories": []})
        block["trajectories"].append({"trajectory_id": entry["trajectory_id"],
                                      "distribution": entry["distribution"],
                                      "output": entry["output"]})
    if len(episodes) != 64:
        raise SystemExit(f"expected 64 training episodes, found {len(episodes)}")

    present = sorted(e for e, b in episodes.items() if b["hazard_present"])
    absent = sorted(e for e, b in episodes.items() if not b["hazard_present"])
    if (len(present), len(absent)) != (48, 16):
        raise SystemExit(f"expected 48/16 hazard split, found {len(present)}/{len(absent)}")

    assignment: dict[str, list[str]] = {}
    p_cursor = a_cursor = 0
    for name, n_present, n_absent in SPLITS:
        assignment[name] = (present[p_cursor:p_cursor + n_present]
                            + absent[a_cursor:a_cursor + n_absent])
        p_cursor += n_present
        a_cursor += n_absent
    if p_cursor != 48 or a_cursor != 16:
        raise SystemExit("split does not consume every episode exactly once")

    seen: set[str] = set()
    for name, members in assignment.items():
        overlap = seen & set(members)
        if overlap:
            raise SystemExit(f"{name} overlaps an earlier split: {sorted(overlap)}")
        seen |= set(members)
    if seen != set(episodes):
        raise SystemExit("partition does not cover every training episode")

    payload = {
        "name": "hybrid_obstacle_prox_activity_partition_v1",
        "source_partition": SOURCE_PARTITION,
        "unit": "episode",
        "assignment_rule": ("episodes sorted by identity within each hazard stratum and "
                            "dealt in the fixed order gate_training, "
                            "checkpoint_validation, threshold_calibration, "
                            "nested_offline_evaluation"),
        "splits": {
            name: {
                "episodes": members,
                "episode_count": len(members),
                "hazard_present": sum(1 for e in members if episodes[e]["hazard_present"]),
                "hazard_absent": sum(1 for e in members
                                     if not episodes[e]["hazard_present"]),
                "trajectories": sorted(
                    t["trajectory_id"] for e in members
                    for t in episodes[e]["trajectories"]),
                "trajectory_count": sum(len(episodes[e]["trajectories"])
                                        for e in members),
            } for name, members in assignment.items()},
        "excluded_partitions": {
            "reused_diagnostics_only": list(EXCLUDED),
            "reason": ("already opened in earlier tasks; they may not enter gate fitting "
                       "or threshold calibration"),
        },
        "development4": {
            "rows": [r["candidate_index"] for r in dev4["rows"]],
            "in_paired_dataset": False,
            "used_for_gate_training": False,
        },
        "confirmatory41": {
            "rows": len(conf41["rows"]),
            "in_paired_dataset": False,
            "used_for_gate_training": False,
            "executed": bool(conf41.get("executed_in_this_task", False)),
        },
        "dataset_manifest_sha256": manifest["manifest_sha256"],
    }
    payload["manifest_sha256"] = thr.canonical_hash(payload)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    for name, block in payload["splits"].items():
        print(f"  {name:<28} episodes={block['episode_count']:>2} "
              f"(haz+ {block['hazard_present']}, haz- {block['hazard_absent']})  "
              f"trajectories={block['trajectory_count']}")
    print(f"manifest sha256: {payload['manifest_sha256']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
