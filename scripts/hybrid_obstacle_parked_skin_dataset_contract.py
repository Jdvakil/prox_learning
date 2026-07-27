#!/usr/bin/env python3
"""Freeze the parked-skin supervision dataset contract and execution schedule.

Handoff steps 3-4. Every row and distribution is predeclared and hashed before any
generation runs:

    A EXPERT_RECONSTRUCTED        100 canonical trajectories, reconstructed from the
                                  immutable source H5 states -- no new policy rollout
    B ACT_ONLY_ON_POLICY          100 manifest rows, one fresh rollout each
    C ORACLE_ON_POLICY            the same 100 rows, one fresh rollout each
    D LEARNER_INDUCED_ON_POLICY   the 64 reference-training rows, driven by the exact
                                  frozen round-0 learner

264 new policy rollouts. Condition order for the ACT/oracle pairs alternates by
predeclared row rank, and the learner rows are ordered after all 200 ACT/oracle
identities so they cannot be interleaved into that pairing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET_VERSION = "hybrid_obstacle_parked_skin_supervision_v1"
PAIR_CONDITIONS = ("ACT_ONLY_ON_POLICY", "ORACLE_ON_POLICY")
LEARNER_CONDITION = "LEARNER_INDUCED_ON_POLICY"
MAX_CONCURRENT_ROLLOUTS = 2


def canonical_hash(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--partition", required=True, type=Path)
    ap.add_argument("--development-manifest", required=True, type=Path)
    ap.add_argument("--confirmatory-manifest", required=True, type=Path)
    ap.add_argument("--learner-manifest", required=True, type=Path)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    partition = json.loads(args.partition.read_text())
    development = json.loads(args.development_manifest.read_text())
    confirmatory = json.loads(args.confirmatory_manifest.read_text())
    learner_model = json.loads(args.learner_manifest.read_text())

    canonical_rows = [row for name in ("reference_train", "reference_calibration",
                                       "reference_validation", "offline_reference_test")
                      for row in partition["partitions"][name]]
    canonical_rows.sort(key=lambda r: (r["partition"], not r["hazard_present"],
                                       r["predeclared_stratum_rank"]))
    if len(canonical_rows) != 100:
        raise SystemExit(f"expected 100 canonical rows, got {len(canonical_rows)}")
    training_rows = sorted(partition["partitions"]["reference_train"],
                           key=lambda r: (not r["hazard_present"],
                                          r["predeclared_stratum_rank"]))
    if len(training_rows) != 64:
        raise SystemExit(f"expected 64 reference-training rows, got {len(training_rows)}")

    excluded = ({r["episode_id"] for r in development["rows"]}
                | {r["episode_id"] for r in confirmatory["rows"]})
    scheduled = {r["episode_id"] for r in canonical_rows}
    if scheduled & excluded:
        raise SystemExit("a development4 or confirmatory41 row entered the schedule")

    def row_identity(row, distribution, rank=None):
        return {
            "dataset_version": DATASET_VERSION,
            "distribution": distribution,
            "partition": row["partition"],
            "trajectory_id": f"{distribution.lower()}__{row['episode_id']}",
            "manifest_row_id": row["row_sha256"],
            "episode_id": row["episode_id"],
            "candidate_index": row["candidate_index"],
            "hazard_present": bool(row["hazard_present"]),
            "predeclared_stratum_rank": row["predeclared_stratum_rank"],
            "source_h5_sha256": row["source_h5_sha256"],
            **({"row_rank": rank} if rank is not None else {}),
        }

    # ---- distribution A: reconstruction, no policy rollout ----------------- #
    expert = [{**row_identity(row, "EXPERT_RECONSTRUCTED"),
               "policy_condition": "EXPERT_RECONSTRUCTED",
               "requires_policy_rollout": False,
               "output": f"{args.data_root.rstrip('/')}/expert/"
                         f"{row['episode_id']}.h5"}
              for row in canonical_rows]

    # ---- distributions B and C: alternating condition order ---------------- #
    pairs = []
    for rank, row in enumerate(canonical_rows):
        order = PAIR_CONDITIONS if rank % 2 == 0 else tuple(reversed(PAIR_CONDITIONS))
        for condition in order:
            pairs.append({
                **row_identity(row, condition, rank),
                "policy_condition": condition,
                "requires_policy_rollout": True,
                "output": f"{args.data_root.rstrip('/')}/{condition.lower()}/"
                          f"{row['episode_id']}.h5",
                "rollout_dir": f"{args.data_root.rstrip('/')}/_rollouts/"
                               f"{condition.lower()}__{row['episode_id']}",
            })

    # ---- distribution D: after every ACT/oracle identity is frozen --------- #
    learner = [{**row_identity(row, LEARNER_CONDITION),
                "policy_condition": LEARNER_CONDITION,
                "requires_policy_rollout": True,
                "learner_checkpoint_sha256": learner_model["artifact_file_sha256"],
                "learner_manifest_sha256": learner_model["manifest_sha256"],
                "learner_label": learner_model["label"],
                "output": f"{args.data_root.rstrip('/')}/learner_induced_on_policy/"
                          f"{row['episode_id']}.h5",
                "rollout_dir": f"{args.data_root.rstrip('/')}/_rollouts/"
                               f"learner__{row['episode_id']}"}
               for row in training_rows]

    entries = []
    for group in (expert, pairs, learner):
        for item in group:
            entries.append({"execution_order": len(entries), **item})

    policy_rollouts = sum(1 for e in entries if e["requires_policy_rollout"])
    if policy_rollouts != 264:
        raise SystemExit(f"expected 264 policy rollouts, got {policy_rollouts}")

    firsts: dict[str, str] = {}
    for entry in entries:
        if entry["distribution"] in PAIR_CONDITIONS:
            firsts.setdefault(entry["episode_id"], entry["distribution"])
    balance = {c: sum(1 for v in firsts.values() if v == c) for c in PAIR_CONDITIONS}

    manifest = {
        "schema": "hybrid_obstacle_parked_skin_supervision_manifest_v1",
        "dataset_version": DATASET_VERSION,
        "data_root": args.data_root,
        "partition_sha256": partition["partition_sha256"],
        "partition_composition": partition["composition"],
        "development4_manifest_sha256": development["manifest_sha256"],
        "confirmatory41_manifest_sha256": confirmatory["manifest_sha256"],
        "development4_excluded": True,
        "confirmatory41_excluded": True,
        "excluded_episode_count": len(excluded),
        "distributions": {
            "EXPERT_RECONSTRUCTED": {
                "rows": len(expert), "policy_rollouts": 0,
                "method": "reconstruct each recorded decision state from the immutable "
                          "source H5; ACT is never run and the source is never altered"},
            "ACT_ONLY_ON_POLICY": {
                "rows": len(canonical_rows), "policy_rollouts": len(canonical_rows),
                "method": "nominal ACT drives; the parked oracle is a non-executed shadow"},
            "ORACLE_ON_POLICY": {
                "rows": len(canonical_rows), "policy_rollouts": len(canonical_rows),
                "method": "the validated oracle residual controller drives; the stored "
                          "parked field is the exact field that produced the action"},
            LEARNER_CONDITION: {
                "rows": len(training_rows), "policy_rollouts": len(training_rows),
                "method": "the exact frozen round-0 seven-output learner drives",
                "checkpoint_sha256": learner_model["artifact_file_sha256"],
                "label": learner_model["label"]},
        },
        "total_policy_rollouts": policy_rollouts,
        "total_reconstructions": len(expert),
        "condition_order_rule": ("even predeclared row rank runs ACT_ONLY_ON_POLICY first, "
                                 "odd runs ORACLE_ON_POLICY first; learner rows are ordered "
                                 "after all 200 ACT/oracle identities"),
        "condition_order_balance": balance,
        "max_concurrent_rollout_processes": MAX_CONCURRENT_ROLLOUTS,
        "concurrency_rationale": ("the prior collection lost a shard to GPU contention at "
                                  "five concurrent processes"),
        "retention_rule": ("every real timestep is retained -- active, zero, grasp-contact, "
                           "post-contact, failure and clear alike; no balancing and no "
                           "subsampling at generation time, so natural class prevalence, "
                           "false-positive rates and temporal persistence stay measurable"),
        "storage_rule": ("one contiguous (T,40,8,8) current sequence and one paired "
                         "(T,40,8,8) parked sequence per trajectory; the four-frame causal "
                         "history is reconstructed on demand, never stored duplicated"),
        "entries": entries,
    }
    manifest["manifest_sha256"] = canonical_hash(manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(f"expert reconstructions : {len(expert)}")
    print(f"ACT/oracle rollouts    : {len(pairs)} (order balance {balance})")
    print(f"learner rollouts       : {len(learner)}")
    print(f"total policy rollouts  : {policy_rollouts}")
    print(f"manifest sha256        : {manifest['manifest_sha256']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
