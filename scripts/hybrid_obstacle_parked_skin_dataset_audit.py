#!/usr/bin/env python3
"""Integrity audit of the frozen parked-skin supervision dataset.

Handoff steps 11-13 and 17-19. Every retained trajectory is re-opened and checked against
the contract, with the frozen SafetyHead re-run from the *stored* fields so the recorded
7-D targets are reproduced rather than trusted.

Predeclared tolerances (frozen before generation, in ``parked_skin_retention.TOLERANCES``):

    head output max abs delta          <= 1e-6
    oracle differential max abs delta  <= 1e-6
    closeness inequality               <= 1e-7

A material violation stops the task. Nothing is silently clamped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submodules" / "act"))

from parked_skin_retention import (
    CHANGED_PIXEL_EPSILON,
    DEPLOYABLE_FIELDS,
    INTEGRITY_FIELDS,
    PRIVILEGED_FIELDS,
    TOLERANCES,
    closeness_to_depth,
    reconstruct_all_histories,
)


def canonical_hash(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--safety-dir", required=True, type=Path)
    ap.add_argument("--head-check-trajectories", type=int, default=12,
                    help="how many trajectories get the full SafetyHead reconstruction")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    for name in ("manifest", "safety_dir", "out"):
        setattr(args, name, Path(getattr(args, name)).resolve())

    import h5py
    import torch
    from train_safety_cvae import SafetyHead

    manifest = json.loads(args.manifest.read_text())
    head = SafetyHead.load(str(args.safety_dir),
                           device="cuda" if torch.cuda.is_available() else "cpu")

    files, problems = [], []
    counts: dict[str, Counter] = {"distribution": Counter(), "partition": Counter()}
    frames_by_distribution: Counter = Counter()
    frames_by_partition: Counter = Counter()
    totals = {"frames": 0, "active": 0, "zero": 0, "hazard_present": 0,
              "hazard_absent": 0, "bytes": 0}
    neutrality_failures = 0
    inequality_violations = 0
    hazard_absent_nonzero = 0
    noncausal = 0
    nonfinite = 0
    shape_mismatch = 0
    duplicate_identity = 0
    head_checked = 0
    head_delta = 0.0
    oracle_delta = 0.0
    coverage = {"changed_pixel_fraction": [], "current_head_norm": [],
                "oracle_norm": [], "active_sensors": Counter()}
    seen_identity: set[str] = set()
    seen_trajectory: set[str] = set()

    scheduled = [e for e in manifest["entries"]]
    for index, entry in enumerate(scheduled):
        path = Path(entry["output"])
        if not path.is_file():
            problems.append(f"missing output: {entry['distribution']} {entry['episode_id']}")
            continue
        with h5py.File(path, "r") as handle:
            attrs = dict(handle.attrs)
            frames = int(attrs["frames"])
            current = handle["deployable/current_closeness"][()]
            parked = handle["privileged/parked_closeness"][()]
            current_valid = handle["deployable/current_valid_mask"][()]
            removable = handle["privileged/removable_closeness"][()]
            changed = handle["privileged/changed_pixel_mask"][()]
            stored_current_head = handle["privileged/current_head"][()]
            stored_parked_head = handle["privileged/parked_head"][()]
            stored_oracle = handle["privileged/oracle_dq"][()]
            steps = handle["deployable/episode_step"][()]
            stamps = handle["deployable/control_timestamp"][()]
            qpos = handle["deployable/qpos"][()]
            neutral = handle["integrity/state_neutral"][()]
            # NOTE the parentheses: `-` binds tighter than `|`, so writing
            # `A | B | C - D` computes `A | B | (C - D)` and reports every field in
            # A and B as missing regardless of what the file holds.
            required = ({f"deployable/{f}" for f in DEPLOYABLE_FIELDS}
                        | {f"privileged/{f}" for f in PRIVILEGED_FIELDS}
                        | {f"integrity/{f}" for f in INTEGRITY_FIELDS})
            present = {f"{g}/{k}" for g in handle for k in handle[g]}
            missing_fields = sorted(required - present)

        identity = canonical_hash({"d": entry["distribution"], "e": entry["episode_id"],
                                   "c": entry["policy_condition"]})
        if identity in seen_identity:
            duplicate_identity += 1
            problems.append(f"duplicate identity: {entry['distribution']} "
                            f"{entry['episode_id']}")
        seen_identity.add(identity)
        if attrs.get("trajectory_id") in seen_trajectory:
            problems.append(f"duplicate trajectory_id {attrs.get('trajectory_id')}")
        seen_trajectory.add(attrs.get("trajectory_id"))
        if missing_fields:
            problems.append(f"{entry['episode_id']}: missing {missing_fields[:4]}")

        # ---- shapes, finiteness ------------------------------------------ #
        if current.shape != (frames, 40, 8, 8) or parked.shape != current.shape:
            shape_mismatch += 1
            problems.append(f"{entry['episode_id']}: shape {current.shape}")
        if not (np.isfinite(current).all() and np.isfinite(parked).all()):
            nonfinite += 1
            problems.append(f"{entry['episode_id']}: non-finite field")

        # ---- sequence and causality -------------------------------------- #
        if len(steps) != frames or steps[0] != 0 or not np.array_equal(
                steps, np.arange(frames)):
            problems.append(f"{entry['episode_id']}: episode_step not contiguous from 0")
        if frames > 1 and not np.all(np.diff(stamps) > 0):
            problems.append(f"{entry['episode_id']}: timestamps not monotonic")
        if len(qpos) != frames:
            problems.append(f"{entry['episode_id']}: qpos length {len(qpos)} != {frames}")
        _, sources = reconstruct_all_histories(current)
        if (sources > np.arange(frames)[:, None]).any():
            noncausal += 1
            problems.append(f"{entry['episode_id']}: a history reads a future frame")
        if not np.array_equal(sources[:, -1], np.arange(frames)):
            noncausal += 1
            problems.append(f"{entry['episode_id']}: history does not end at t")

        # ---- physical pairing -------------------------------------------- #
        violations = int((parked > current + TOLERANCES["closeness_inequality"]).sum())
        inequality_violations += violations
        if violations:
            problems.append(f"{entry['episode_id']}: {violations} parked>current pixels")
        if float(current.max()) > 1.0 + 1e-6 or float(current.min()) < -1e-6:
            problems.append(f"{entry['episode_id']}: closeness outside [0,1]")
        if not np.array_equal(changed, removable > CHANGED_PIXEL_EPSILON):
            problems.append(f"{entry['episode_id']}: changed mask disagrees with removable")

        # ---- hazard-absent exact control --------------------------------- #
        if not entry["hazard_present"]:
            if not np.array_equal(current, parked):
                hazard_absent_nonzero += 1
                problems.append(f"{entry['episode_id']}: hazard-absent fields differ")
            if float(np.abs(removable).max()) != 0.0 or bool(changed.any()):
                hazard_absent_nonzero += 1
                problems.append(f"{entry['episode_id']}: hazard-absent removable nonzero")
            if not np.array_equal(stored_current_head, stored_parked_head) or \
                    float(np.abs(stored_oracle).max()) != 0.0:
                hazard_absent_nonzero += 1
                problems.append(f"{entry['episode_id']}: hazard-absent heads differ")

        failures = int((~neutral).sum())
        neutrality_failures += failures
        if failures:
            problems.append(f"{entry['episode_id']}: {failures} state-neutrality failures")

        # ---- SafetyHead reconstruction from the STORED fields ------------- #
        if head_checked < args.head_check_trajectories:
            recomputed_current = np.stack(
                [head(closeness_to_depth(current[t])) for t in range(frames)])
            recomputed_parked = np.stack(
                [head(closeness_to_depth(parked[t])) for t in range(frames)])
            head_delta = max(head_delta,
                             float(np.abs(recomputed_current - stored_current_head).max()),
                             float(np.abs(recomputed_parked - stored_parked_head).max()))
            oracle_delta = max(oracle_delta, float(np.abs(
                (recomputed_current - recomputed_parked) - stored_oracle).max()))
            head_checked += 1

        active = int((np.linalg.norm(stored_oracle, axis=1) > 0).sum())
        counts["distribution"][entry["distribution"]] += 1
        counts["partition"][entry["partition"]] += 1
        frames_by_distribution[entry["distribution"]] += frames
        frames_by_partition[entry["partition"]] += frames
        totals["frames"] += frames
        totals["active"] += active
        totals["zero"] += frames - active
        totals["hazard_present" if entry["hazard_present"] else "hazard_absent"] += frames
        totals["bytes"] += path.stat().st_size
        coverage["changed_pixel_fraction"].append(float(changed.mean()))
        coverage["current_head_norm"].append(
            float(np.median(np.linalg.norm(stored_current_head, axis=1))))
        coverage["oracle_norm"].append(float(np.abs(stored_oracle).max()))
        coverage["active_sensors"][int(np.median(current_valid.any(axis=(2, 3)).sum(1)))] += 1

        files.append({
            "distribution": entry["distribution"], "partition": entry["partition"],
            "episode_id": entry["episode_id"], "candidate_index": entry["candidate_index"],
            "hazard_present": bool(entry["hazard_present"]), "frames": frames,
            "oracle_active_frames": active, "file": str(path.relative_to(
                Path(manifest["data_root"]))), "file_sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        })
        if (index + 1) % 50 == 0:
            print(f"  audited {index + 1}/{len(scheduled)}", flush=True)

    head_ok = head_delta <= TOLERANCES["head_output_max_abs_delta"]
    oracle_ok = oracle_delta <= TOLERANCES["oracle_differential_max_abs_delta"]

    report = {
        "schema": "hybrid_obstacle_parked_skin_dataset_audit_v1",
        "dataset_version": manifest["dataset_version"],
        "manifest_sha256": manifest["manifest_sha256"],
        "partition_sha256": manifest["partition_sha256"],
        "scheduled_outputs": len(scheduled),
        "outputs_present": len(files),
        "counts": {
            "files_per_distribution": dict(counts["distribution"]),
            "trajectories_per_partition": dict(counts["partition"]),
            "frames_per_distribution": dict(frames_by_distribution),
            "frames_per_partition": dict(frames_by_partition),
            "total_frames": totals["frames"],
            "oracle_active_frames": totals["active"],
            "oracle_zero_frames": totals["zero"],
            "hazard_present_frames": totals["hazard_present"],
            "hazard_absent_frames": totals["hazard_absent"],
            "total_bytes": totals["bytes"],
            "total_gib": round(totals["bytes"] / (1024 ** 3), 3),
        },
        "integrity": {
            "duplicate_trajectory_ids": len(files) - len(seen_trajectory),
            "duplicate_source_identities": duplicate_identity,
            "state_neutrality_failures": neutrality_failures,
            "physical_inequality_violations": inequality_violations,
            "hazard_absent_nonzero_targets": hazard_absent_nonzero,
            "noncausal_histories": noncausal,
            "shape_mismatches": shape_mismatch,
            "nonfinite_values": nonfinite,
            "missing_outputs": len(scheduled) - len(files),
        },
        "tolerances": TOLERANCES,
        "head_reconstruction": {
            "trajectories_checked": head_checked,
            "max_abs_head_delta": head_delta,
            "within_tolerance": head_ok,
            "max_abs_oracle_delta": oracle_delta,
            "oracle_within_tolerance": oracle_ok,
            "method": ("SafetyHead re-run from the STORED closeness fields via "
                       "closeness_to_depth, compared against the recorded 7-D targets"),
        },
        "coverage": {
            "changed_pixel_fraction": {
                "median": float(np.median(coverage["changed_pixel_fraction"]))
                if coverage["changed_pixel_fraction"] else None,
                "max": max(coverage["changed_pixel_fraction"], default=None)},
            "current_head_norm_median_per_trajectory": {
                "median": float(np.median(coverage["current_head_norm"]))
                if coverage["current_head_norm"] else None},
            "oracle_norm_max_per_trajectory": {
                "median": float(np.median(coverage["oracle_norm"]))
                if coverage["oracle_norm"] else None,
                "max": max(coverage["oracle_norm"], default=None)},
            "median_active_sensor_count_histogram": dict(coverage["active_sensors"]),
        },
        "natural_distribution_retained": {
            "active_zero_balancing_at_generation": False,
            "zero_frames_subsampled": False,
            "failures_retained": True,
            "note": "balancing belongs to the future training sampler",
        },
        "files": files,
        "tree_sha256": canonical_hash([{k: f[k] for k in
                                        ("distribution", "episode_id", "file_sha256")}
                                       for f in sorted(files, key=lambda f: (
                                           f["distribution"], f["episode_id"]))]),
        "problems": problems,
        "valid": bool(not problems and len(files) == len(scheduled)
                      and head_ok and oracle_ok),
        "decision_if_invalid": "PARKED_SKIN_DATASET_CONTRACT_FAILED",
    }
    report["report_sha256"] = canonical_hash(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")

    print(f"outputs           : {len(files)}/{len(scheduled)}")
    for name, value in report["counts"]["frames_per_distribution"].items():
        print(f"  {name:<28} {counts['distribution'][name]:>4} files, {value:>7} frames")
    print(f"total frames      : {totals['frames']}  "
          f"(active {totals['active']}, zero {totals['zero']})")
    print(f"size              : {report['counts']['total_gib']} GiB")
    for key, value in report["integrity"].items():
        print(f"  {key:<34} {value}")
    print(f"head reconstruction: max delta {head_delta:.3e} "
          f"(tol {TOLERANCES['head_output_max_abs_delta']:.0e}) -> {head_ok}")
    print(f"oracle reconstruction: max delta {oracle_delta:.3e} -> {oracle_ok}")
    print(f"tree sha256       : {report['tree_sha256']}")
    for problem in problems[:8]:
        print(f"  PROBLEM {problem}")
    print(f"wrote {args.out}")
    return 0 if report["valid"] else 9


if __name__ == "__main__":
    raise SystemExit(main())
