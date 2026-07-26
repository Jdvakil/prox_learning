#!/usr/bin/env python3
"""Freeze the reference-model partition of the 80 canonical ACT-training trajectories.

Handoff step 3. The ACT split is untouched. Within the 80 ACT-training trajectories a
deterministic stratified partition is cut:

    reference train        48 hazard-present + 16 hazard-absent = 64
    reference calibration   6 hazard-present +  2 hazard-absent =  8
    reference validation    6 hazard-present +  2 hazard-absent =  8

The 20 ACT-validation trajectories (15 present + 5 absent) become the offline reference
test set. development4 and confirmatory41 take no part.

Assignment is by ``predeclared_stratum_rank`` -- the rank the episode already carries
from the canonical selection -- so the partition is a function of committed data alone,
with no fresh randomness and no dependence on file order.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

COMPOSITION = {
    "reference_train": {"hazard_present": 48, "hazard_absent": 16},
    "reference_calibration": {"hazard_present": 6, "hazard_absent": 2},
    "reference_validation": {"hazard_present": 6, "hazard_absent": 2},
}
#: Calibration first, then validation, then train. Taking the small sets from the front
#: of the rank order makes them a fixed, inspectable prefix rather than a leftover.
ORDER = ("reference_calibration", "reference_validation", "reference_train")


def canonical_hash(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split-manifest", required=True, type=Path)
    ap.add_argument("--canonical-manifest", required=True, type=Path)
    ap.add_argument("--development-manifest", required=True, type=Path)
    ap.add_argument("--confirmatory-manifest", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    split = json.loads(args.split_manifest.read_text())
    canonical = json.loads(args.canonical_manifest.read_text())
    development = json.loads(args.development_manifest.read_text())
    confirmatory = json.loads(args.confirmatory_manifest.read_text())

    ranks = {row["episode_id"]: int(row["predeclared_stratum_rank"])
             for row in canonical["selected"]}
    episodes = {e["episode_id"]: e for e in split["episodes"]}
    for episode_id in episodes:
        if episode_id not in ranks:
            raise SystemExit(f"episode {episode_id} has no predeclared stratum rank")

    train_pool = [e for e in split["episodes"] if e["split"] == "train"]
    test_pool = [e for e in split["episodes"] if e["split"] == "validation"]
    if len(train_pool) != 80 or len(test_pool) != 20:
        raise SystemExit(f"expected 80/20, got {len(train_pool)}/{len(test_pool)}")

    strata = {
        "hazard_present": sorted((e for e in train_pool if e["hazard_present"]),
                                 key=lambda e: (ranks[e["episode_id"]], e["episode_id"])),
        "hazard_absent": sorted((e for e in train_pool if not e["hazard_present"]),
                                key=lambda e: (ranks[e["episode_id"]], e["episode_id"])),
    }
    if len(strata["hazard_present"]) != 60 or len(strata["hazard_absent"]) != 20:
        raise SystemExit("ACT-training strata are not 60 present / 20 absent")

    cursors = {"hazard_present": 0, "hazard_absent": 0}
    partitions: dict[str, list[dict]] = {}
    for name in ORDER:
        rows = []
        for stratum, count in sorted(COMPOSITION[name].items()):
            start = cursors[stratum]
            for episode in strata[stratum][start:start + count]:
                rows.append({
                    "episode_id": episode["episode_id"],
                    "candidate_index": episode["candidate_index"],
                    "hazard_present": bool(episode["hazard_present"]),
                    "predeclared_stratum_rank": ranks[episode["episode_id"]],
                    "act_split": episode["split"],
                    "source_h5_sha256": episode["source_h5_sha256"],
                    "source_relpath": episode["source_relpath"],
                    "core_trajectory_sha256": episode["core_trajectory_sha256"],
                    "row_sha256": episode["row_sha256"],
                    "partition": name,
                })
            cursors[stratum] = start + count
        partitions[name] = sorted(rows, key=lambda r: (not r["hazard_present"],
                                                       r["predeclared_stratum_rank"]))

    partitions["offline_reference_test"] = sorted(
        [{
            "episode_id": e["episode_id"], "candidate_index": e["candidate_index"],
            "hazard_present": bool(e["hazard_present"]),
            "predeclared_stratum_rank": ranks[e["episode_id"]],
            "act_split": e["split"], "source_h5_sha256": e["source_h5_sha256"],
            "source_relpath": e["source_relpath"],
            "core_trajectory_sha256": e["core_trajectory_sha256"],
            "row_sha256": e["row_sha256"], "partition": "offline_reference_test",
        } for e in test_pool],
        key=lambda r: (not r["hazard_present"], r["predeclared_stratum_rank"]))

    # ---- disjointness, in every direction --------------------------------- #
    identifiers = {name: {r["episode_id"] for r in rows} for name, rows in partitions.items()}
    identifiers["development4"] = {r["episode_id"] for r in development["rows"]}
    identifiers["confirmatory41"] = {r["episode_id"] for r in confirmatory["rows"]}
    overlaps = {}
    names = sorted(identifiers)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            shared = sorted(identifiers[left] & identifiers[right])
            if shared:
                overlaps[f"{left}|{right}"] = shared
    if overlaps:
        raise SystemExit(f"partition overlap: {overlaps}")

    covered = set().union(*(identifiers[n] for n in
                            ("reference_train", "reference_calibration",
                             "reference_validation")))
    if covered != {e["episode_id"] for e in train_pool}:
        raise SystemExit("the three reference partitions do not exactly cover ACT-train")

    manifest = {
        "schema": "hybrid_obstacle_reference_partition_v2",
        "rule": ("deterministic stratified assignment by the episode's committed "
                 "predeclared_stratum_rank, calibration then validation then train; "
                 "no fresh randomness, no dependence on file order"),
        "act_split_manifest_sha256": split["split_manifest_sha256"],
        "canonical_manifest_sha256": canonical["manifest_sha256"],
        "development4_manifest_sha256": development["manifest_sha256"],
        "confirmatory41_manifest_sha256": confirmatory["manifest_sha256"],
        "act_split_unchanged": True,
        "composition": {
            name: {"hazard_present": sum(1 for r in rows if r["hazard_present"]),
                   "hazard_absent": sum(1 for r in rows if not r["hazard_present"]),
                   "total": len(rows)}
            for name, rows in partitions.items()},
        "expected_composition": {
            **COMPOSITION,
            "offline_reference_test": {"hazard_present": 15, "hazard_absent": 5}},
        "excluded": {
            "development4": sorted(identifiers["development4"]),
            "confirmatory41_rows": len(identifiers["confirmatory41"]),
            "note": ("development4 is the live development set and confirmatory41 takes no "
                     "part in any action in this task; neither may appear in any "
                     "reference-model dataset"),
        },
        "pairwise_overlaps": overlaps,
        "all_pairwise_disjoint": not overlaps,
        "partitions": partitions,
    }
    for name, expected in manifest["expected_composition"].items():
        actual = manifest["composition"][name]
        if actual["hazard_present"] != expected["hazard_present"] or \
                actual["hazard_absent"] != expected["hazard_absent"]:
            raise SystemExit(f"{name}: {actual} != {expected}")
    manifest["partition_sha256"] = canonical_hash(manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    for name in ("reference_train", "reference_calibration", "reference_validation",
                 "offline_reference_test"):
        composition = manifest["composition"][name]
        print(f"  {name:<24} {composition['hazard_present']:>3} present + "
              f"{composition['hazard_absent']:>2} absent = {composition['total']:>3}")
    print(f"  all pairwise disjoint    {manifest['all_pairwise_disjoint']}")
    print(f"  partition sha256         {manifest['partition_sha256']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
