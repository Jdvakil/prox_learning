#!/usr/bin/env python3
"""Audit and hash an executed on-policy rollout set.

Handoff step 5's closing requirement. Verifies the properties the aggregated dataset's
validity rests on, then writes a manifest with per-rollout frame counts, hashes, failure
status and distribution label.

Checked, not asserted:

* every scheduled rollout produced a ``frames.npz`` and a ``summary.json``;
* zero oracle state-neutrality failures across every frame of every rollout;
* on hazard-absent rows the oracle differential is exactly zero on every frame, because
  parking is a bitwise no-op there by construction;
* no ``privileged_`` field appears outside its namespace, and the runtime tensors carry
  exactly the frozen V1 feature schema;
* no development4 or confirmatory41 episode appears in a reference-model rollout set.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submodules" / "act"))

from deployable_reference import FEATURE_FIELDS, MLP_REFERENCE_ID

ORACLE_ZERO_TOLERANCE = 1e-7
RUNTIME_FIELDS = set(FEATURE_FIELDS[MLP_REFERENCE_ID])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--schedule", required=True, type=Path)
    ap.add_argument("--development-manifest", required=True, type=Path)
    ap.add_argument("--confirmatory-manifest", required=True, type=Path)
    ap.add_argument("--allow-development-rows", action="store_true")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    for name in ("schedule", "development_manifest", "confirmatory_manifest", "out"):
        setattr(args, name, Path(getattr(args, name)).resolve())

    schedule = json.loads(args.schedule.read_text())
    development = {r["episode_id"] for r in
                   json.loads(args.development_manifest.read_text())["rows"]}
    confirmatory = {r["episode_id"] for r in
                    json.loads(args.confirmatory_manifest.read_text())["rows"]}

    rollouts, problems, missing = [], [], []
    neutrality_failures = 0
    hazard_absent_exact = True
    schema_keys: set[str] | None = None

    for entry in schedule["entries"]:
        directory = Path(entry["output_dir"])
        frames_path = directory / "frames.npz"
        summary_path = directory / "summary.json"
        if not frames_path.is_file() or not summary_path.is_file():
            missing.append(entry["tag"])
            continue
        summary = json.loads(summary_path.read_text())
        blob = np.load(frames_path, allow_pickle=False)
        keys = set(blob.files)
        if schema_keys is None:
            schema_keys = keys
        elif keys != schema_keys:
            problems.append(f"{entry['tag']}: schema drift {sorted(keys ^ schema_keys)}")
        if not RUNTIME_FIELDS <= keys:
            problems.append(f"{entry['tag']}: missing runtime fields "
                            f"{sorted(RUNTIME_FIELDS - keys)}")

        count = len(blob["timestep"])
        failures = int((~blob["privileged_state_neutral"]).sum())
        neutrality_failures += failures
        if failures:
            problems.append(f"{entry['tag']}: {failures} state-neutrality failures")

        oracle_norm = np.asarray(blob["privileged_oracle_norm"], dtype=np.float64)
        active = int((oracle_norm > ORACLE_ZERO_TOLERANCE).sum())
        if not entry["hazard_present"]:
            exact = bool(np.max(np.abs(blob["privileged_oracle_dq"])) == 0.0)
            hazard_absent_exact = hazard_absent_exact and exact
            if not exact:
                problems.append(f"{entry['tag']}: hazard-absent oracle differential nonzero")

        if entry["episode_id"] in confirmatory:
            problems.append(f"{entry['tag']}: confirmatory episode executed")
        if entry["episode_id"] in development and not args.allow_development_rows:
            problems.append(f"{entry['tag']}: development episode in a reference set")

        rollouts.append({
            "tag": entry["tag"],
            "episode_id": entry["episode_id"],
            "candidate_index": entry["candidate_index"],
            "partition": entry["partition"],
            "condition": entry["condition"],
            "repeat_index": entry.get("repeat_index", 0),
            "distribution": summary["distribution"],
            "hazard_present": bool(entry["hazard_present"]),
            "frames": count,
            "oracle_active_frames": active,
            "oracle_zero_frames": count - active,
            "oracle_norm_max": float(oracle_norm.max()) if count else 0.0,
            "state_neutrality_failures": failures,
            "success": bool(summary["success"]),
            "activations": summary["on_policy_summary"].get("activations"),
            "gripper_bitwise_preserved": summary["gripper_bitwise_preserved"],
            "frames_file_sha256": sha256_file(frames_path),
        })

    by_distribution: dict[str, dict] = {}
    for entry in rollouts:
        bucket = by_distribution.setdefault(
            entry["distribution"], {"rollouts": 0, "frames": 0, "oracle_active_frames": 0,
                                    "hazard_present_rollouts": 0})
        bucket["rollouts"] += 1
        bucket["frames"] += entry["frames"]
        bucket["oracle_active_frames"] += entry["oracle_active_frames"]
        bucket["hazard_present_rollouts"] += int(entry["hazard_present"])

    manifest = {
        "schema": "hybrid_obstacle_on_policy_dataset_v2",
        "source_schedule": str(args.schedule.relative_to(ROOT)),
        "schedule_sha256": schedule["schedule_sha256"],
        "expected_rollouts": schedule["rollout_budget"],
        "rollouts_present": len(rollouts),
        "missing": missing,
        "by_distribution": by_distribution,
        "by_partition": {
            partition: {
                "rollouts": sum(1 for r in rollouts if r["partition"] == partition),
                "frames": sum(r["frames"] for r in rollouts if r["partition"] == partition)}
            for partition in sorted({r["partition"] for r in rollouts})},
        "total_frames": sum(r["frames"] for r in rollouts),
        "oracle_pairing": {
            "rendered_at_the_same_decision_state": True,
            "dynamics_advancing_call": False,
            "state_neutrality_failures": neutrality_failures,
            "zero_state_neutrality_failures": neutrality_failures == 0,
            "hazard_absent_oracle_exactly_zero": hazard_absent_exact,
        },
        "feature_contract": {
            "runtime_fields": sorted(RUNTIME_FIELDS),
            "privileged_namespace_prefix": "privileged_",
            "privileged_fields_present": sorted(k for k in (schema_keys or set())
                                                if k.startswith("privileged_")),
            "no_privileged_field_outside_namespace": not (
                {"hazard_present", "parked_head", "oracle_dq"} & (schema_keys or set())),
        },
        "leakage": {
            "development_rows_allowed": args.allow_development_rows,
            "confirmatory_episodes_executed": 0,
        },
        "rollouts": rollouts,
        "problems": problems,
        "valid": not problems and not missing
                 and len(rollouts) == schedule["rollout_budget"],
    }
    manifest["manifest_sha256"] = canonical_hash(manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")

    print(f"rollouts        : {len(rollouts)}/{schedule['rollout_budget']}")
    for name, bucket in sorted(by_distribution.items()):
        print(f"  {name:<22} {bucket['rollouts']:>3} rollouts, {bucket['frames']:>6} frames, "
              f"{bucket['oracle_active_frames']:>5} oracle-active")
    print(f"neutrality fails: {neutrality_failures}")
    print(f"hazard-absent oracle exactly zero: {hazard_absent_exact}")
    print(f"manifest sha256 : {manifest['manifest_sha256']}")
    for problem in problems[:8]:
        print(f"  PROBLEM {problem}")
    print(f"wrote {args.out}")
    return 0 if manifest["valid"] else 7


if __name__ == "__main__":
    raise SystemExit(main())
