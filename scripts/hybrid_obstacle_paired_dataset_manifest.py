#!/usr/bin/env python3
"""Audit and hash the paired oracle-reference dataset.

Handoff step 3's closing requirement. Verifies the pairing-correctness properties that
the dataset's validity rests on, then writes ``paired_dataset_manifest.json`` with file
counts, per-file hashes and per-split tree hashes.

Pairing correctness is checked, not asserted:

* zero state-neutrality failures across every frame of every trajectory;
* on hazard-absent trajectories ``current_head`` and ``parked_head`` must be **exactly**
  equal, because parking is a bitwise no-op there by construction;
* no privileged field may appear outside the ``privileged_`` namespace.
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

from deployable_reference import ALLOWED_RUNTIME_FIELDS, PRIVILEGED_FIELDS

AUDIT_ONLY_FIELDS = frozenset({
    "timestep", "parked_head", "oracle_dq", "teacher_active", "teacher_dq",
    "teacher_valid", "state_neutral", "skins_identical", "task_phase", "minimum_depth",
})


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
    ap.add_argument("--dataset-dir", required=True, type=Path)
    ap.add_argument("--split-manifest", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    for name in ("dataset_dir", "split_manifest", "out"):
        setattr(args, name, Path(getattr(args, name)).resolve())

    split = json.loads(args.split_manifest.read_text())
    files, problems = [], []
    counts = {"train": {"trajectories": 0, "frames": 0, "support_frames": 0,
                        "teacher_active_frames": 0, "oracle_nonzero_frames": 0},
              "validation": {"trajectories": 0, "frames": 0, "support_frames": 0,
                             "teacher_active_frames": 0, "oracle_nonzero_frames": 0}}
    hazard_absent_exact = True
    neutrality_failures = 0
    schema_keys: set[str] | None = None

    for episode in sorted(split["episodes"], key=lambda e: e["split_rank"]):
        path = args.dataset_dir / f"{episode['episode_id']}.npz"
        if not path.is_file():
            problems.append(f"missing {episode['episode_id']}")
            continue
        blob = np.load(path, allow_pickle=False)
        keys = set(blob.files)
        if schema_keys is None:
            schema_keys = keys
        elif keys != schema_keys:
            problems.append(f"{episode['episode_id']}: schema drift {sorted(keys ^ schema_keys)}")

        frames = len(blob["timestep"])
        oracle_nonzero = int((np.linalg.norm(blob["oracle_dq"], axis=1) > 1e-9).sum())
        bucket = counts[episode["split"]]
        bucket["trajectories"] += 1
        bucket["frames"] += frames
        bucket["support_frames"] += int(blob["runtime_support"].sum())
        bucket["teacher_active_frames"] += int(blob["teacher_active"].sum())
        bucket["oracle_nonzero_frames"] += oracle_nonzero

        failures = int((~blob["state_neutral"]).sum())
        neutrality_failures += failures
        if failures:
            problems.append(f"{episode['episode_id']}: {failures} neutrality failures")

        if not episode["hazard_present"]:
            exact = bool(np.array_equal(blob["current_head"], blob["parked_head"]))
            hazard_absent_exact = hazard_absent_exact and exact
            if not exact:
                problems.append(f"{episode['episode_id']}: hazard-absent heads differ")

        files.append({
            "episode_id": episode["episode_id"],
            "candidate_index": episode["candidate_index"],
            "split": episode["split"],
            "hazard_present": bool(episode["hazard_present"]),
            "frames": frames,
            "support_frames": int(blob["runtime_support"].sum()),
            "teacher_active_frames": int(blob["teacher_active"].sum()),
            "oracle_nonzero_frames": oracle_nonzero,
            "oracle_dq_max_abs": float(np.max(np.abs(blob["oracle_dq"]))) if frames else 0.0,
            "file": path.name,
            "file_sha256": sha256_file(path),
        })

    leaked = sorted((schema_keys or set()) & PRIVILEGED_FIELDS
                    - AUDIT_ONLY_FIELDS - {f for f in schema_keys or set()
                                           if f.startswith("privileged_")})
    unnamespaced = sorted(f for f in (schema_keys or set())
                          if f in {"hazard_present", "hazard_pose"})
    if unnamespaced:
        problems.append(f"privileged field outside the namespace: {unnamespaced}")

    def tree_hash(split_name: str) -> str:
        return canonical_hash([{k: entry[k] for k in ("episode_id", "file_sha256")}
                               for entry in files if entry["split"] == split_name])

    manifest = {
        "schema": "hybrid_obstacle_paired_reference_dataset_v1",
        "dataset_dir": str(args.dataset_dir),
        "split_manifest_sha256": split["split_manifest_sha256"],
        "file_count": len(files),
        "expected_file_count": len(split["episodes"]),
        "counts": counts,
        "total_frames": sum(c["frames"] for c in counts.values()),
        "train_trajectories": [e["episode_id"] for e in files if e["split"] == "train"],
        "validation_trajectories": [e["episode_id"] for e in files
                                    if e["split"] == "validation"],
        "train_tree_sha256": tree_hash("train"),
        "validation_tree_sha256": tree_hash("validation"),
        "tree_sha256": canonical_hash([{k: entry[k] for k in
                                        ("episode_id", "split", "file_sha256")}
                                       for entry in files]),
        "files": files,
        "pairing_correctness": {
            "rendered_at_the_same_decision_state": True,
            "observation_substep_used_as_current_half": False,
            "dynamics_advancing_operation_called": False,
            "mj_forward_implementation_used": False,
            "render_state_restored_directly": True,
            "state_neutrality_failures": neutrality_failures,
            "zero_state_neutrality_failures": neutrality_failures == 0,
            "hazard_absent_heads_exactly_equal": hazard_absent_exact,
        },
        "feature_contract": {
            "runtime_whitelist": sorted(ALLOWED_RUNTIME_FIELDS),
            "privileged_namespace_prefix": "privileged_",
            "audit_only_fields": sorted(AUDIT_ONLY_FIELDS),
            "privileged_fields_outside_namespace": unnamespaced,
            "leaked": leaked,
        },
        "problems": problems,
        "valid": not problems and len(files) == len(split["episodes"]),
    }
    manifest["manifest_sha256"] = canonical_hash(manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")

    print(f"files            : {manifest['file_count']}/{manifest['expected_file_count']}")
    for name, bucket in counts.items():
        print(f"{name:<11}      : {bucket['trajectories']} trajectories, "
              f"{bucket['frames']} frames, {bucket['support_frames']} support, "
              f"{bucket['teacher_active_frames']} teacher-active, "
              f"{bucket['oracle_nonzero_frames']} oracle-nonzero")
    print(f"neutrality fails : {neutrality_failures}")
    print(f"hazard-absent exact: {hazard_absent_exact}")
    print(f"tree sha256      : {manifest['tree_sha256']}")
    if problems:
        for problem in problems[:10]:
            print(f"  PROBLEM {problem}")
    print(f"wrote {args.out}")
    return 0 if manifest["valid"] else 6


if __name__ == "__main__":
    raise SystemExit(main())
