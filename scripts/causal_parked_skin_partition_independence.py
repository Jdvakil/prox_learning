#!/usr/bin/env python3
"""Prove trajectory-level isolation across train / validation / calibration / offline test.

Handoff step 3. The same episode identity is deliberately reused across all four source
distributions -- expert, ACT-only, oracle and learner-induced rows are the *same* scene
driven differently -- so per-file uniqueness proves nothing. What has to hold is that every
copy of an episode lands in one partition. If episode X's expert row trained the model and
its oracle row scored it, the offline-test number would be measuring memorisation of a
scene, not generalisation.

Five independent identity keys are checked, because any one of them could coincide for a
benign reason: episode id, source H5 hash, manifest-row hash, trajectory id, and the
initial-state hash of the replayed/rolled-out episode.

Writes diagnostics_output/causal_parked_skin_reference_v1/partition_independence.json.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

import numpy as np

PARTITIONS = ("reference_train", "reference_validation", "reference_calibration",
              "offline_reference_test")

# identity key -> attribute in the stored file (None means it comes from the manifest)
IDENTITY_KEYS = {
    "episode_id": "episode_id",
    "source_h5_sha256": "source_h5_sha256",
    "manifest_row_id": "manifest_row_id",
    "trajectory_id": "trajectory_id",
    "initial_state_sha256": None,
}


def canonical_hash(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--partition-config", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    import h5py

    manifest = json.loads(args.manifest.read_text())
    partition_config = json.loads(args.partition_config.read_text())

    # identity value -> set of partitions it appears in
    appearances: dict[str, dict[str, set]] = {k: collections.defaultdict(set)
                                              for k in IDENTITY_KEYS}
    composition: dict[str, dict] = {p: {
        "trajectories": 0, "frames": 0,
        "source_mode": collections.Counter(), "source_mode_frames": collections.Counter(),
        "hazard_present_trajectories": 0, "hazard_absent_trajectories": 0,
        "hazard_present_frames": 0, "hazard_absent_frames": 0,
        "oracle_active_frames": 0, "oracle_zero_frames": 0,
        "episode_ids": set(),
    } for p in PARTITIONS}

    unknown_partitions: list[str] = []
    for entry in manifest["entries"]:
        path = Path(entry["output"])
        with h5py.File(path, "r") as handle:
            attrs = {k: handle.attrs[k] for k in handle.attrs}
            frames = int(attrs["frames"])
            oracle_active = int(np.asarray(handle["privileged/oracle_active"][()]).sum())
        partition = str(attrs["partition"])
        if partition not in composition:
            unknown_partitions.append(f"{entry['episode_id']}: {partition}")
            continue
        distribution = str(attrs["distribution"])
        hazard = bool(attrs["hazard_present"])

        for key, attr in IDENTITY_KEYS.items():
            if attr is not None:
                value = str(attrs[attr])
            else:
                # expert rows record the replayed initial state; on-policy rows record
                # the rolled-out one under a different attribute name
                value = str(attrs.get("replayed_initial_state_sha256")
                            or attrs.get("initial_state_sha256") or "")
                if not value:
                    continue
            appearances[key][value].add(partition)

        block = composition[partition]
        block["trajectories"] += 1
        block["frames"] += frames
        block["source_mode"][distribution] += 1
        block["source_mode_frames"][distribution] += frames
        block["oracle_active_frames"] += oracle_active
        block["oracle_zero_frames"] += frames - oracle_active
        block["episode_ids"].add(str(attrs["episode_id"]))
        if hazard:
            block["hazard_present_trajectories"] += 1
            block["hazard_present_frames"] += frames
        else:
            block["hazard_absent_trajectories"] += 1
            block["hazard_absent_frames"] += frames

    crossings = {}
    for key, table in appearances.items():
        crossings[key] = sorted(
            {"identity": value, "partitions": sorted(parts)}
            for value, parts in table.items() if len(parts) > 1)

    # episode identities are reused across distributions by design; confirm the reuse is
    # within a partition and quantify it so the report is not silent about it
    reuse = collections.Counter()
    for value, parts in appearances["episode_id"].items():
        reuse[len(parts)] += 1
    per_episode_files = collections.Counter()
    for entry in manifest["entries"]:
        per_episode_files[entry["episode_id"]] += 1

    report_composition = {}
    for name, block in composition.items():
        report_composition[name] = {
            "trajectories": block["trajectories"],
            "frames": block["frames"],
            "distinct_episodes": len(block["episode_ids"]),
            "source_mode_trajectories": dict(sorted(block["source_mode"].items())),
            "source_mode_frames": dict(sorted(block["source_mode_frames"].items())),
            "hazard_present_trajectories": block["hazard_present_trajectories"],
            "hazard_absent_trajectories": block["hazard_absent_trajectories"],
            "hazard_present_frames": block["hazard_present_frames"],
            "hazard_absent_frames": block["hazard_absent_frames"],
            "oracle_active_frames": block["oracle_active_frames"],
            "oracle_zero_frames": block["oracle_zero_frames"],
            "oracle_active_prevalence": round(
                block["oracle_active_frames"] / block["frames"], 6)
            if block["frames"] else None,
        }

    total_crossings = sum(len(v) for v in crossings.values())
    report = {
        "schema": "causal_parked_skin_partition_independence_v1",
        "dataset_manifest_sha256": manifest["manifest_sha256"],
        "partition_config_sha256": partition_config["partition_sha256"],
        "identity_keys_checked": sorted(IDENTITY_KEYS),
        "crossings": crossings,
        "total_crossings": total_crossings,
        "unknown_partitions": unknown_partitions,
        "composition": report_composition,
        "episode_reuse_across_distributions": {
            "episodes_by_partition_count": {str(k): v for k, v in sorted(reuse.items())},
            "files_per_episode_histogram": dict(
                sorted(collections.Counter(per_episode_files.values()).items())),
            "note": ("an episode identity is reused across source distributions by "
                     "design; every copy must land in one partition"),
        },
        "valid": bool(total_crossings == 0 and not unknown_partitions),
    }
    report["report_sha256"] = canonical_hash(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")

    for name in PARTITIONS:
        block = report_composition[name]
        print(f"{name:<24} traj={block['trajectories']:>3} "
              f"frames={block['frames']:>6} eps={block['distinct_episodes']:>3} "
              f"active={block['oracle_active_frames']:>5} "
              f"({block['oracle_active_prevalence']:.4f}) "
              f"haz+/-={block['hazard_present_frames']}/{block['hazard_absent_frames']}")
        print(f"{'':<24} modes={block['source_mode_trajectories']}")
    for key, items in crossings.items():
        print(f"crossings[{key}]: {len(items)}")
    histogram = report["episode_reuse_across_distributions"][
        "files_per_episode_histogram"]
    print(f"files per episode: {histogram}")
    print(f"valid: {report['valid']}")
    print(f"wrote {args.out}")
    return 0 if report["valid"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
