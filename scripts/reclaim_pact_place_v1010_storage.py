#!/usr/bin/env python3
"""V10.10 storage reclamation, verified before and after.

Nothing is removed until every artifact that must survive has been hashed and
matched against the V10.9 close-out. The deletion manifest is written first, so
what was removed is recoverable knowledge even though the bytes are not.

Authorized for removal, and nothing else:
  * diagnostics_output/pact_place_v109_eval_traj/
  * assets/act_style_data/pact_place_v108_141/   (regenerable from the datagen rows)
  * resume_bundle.ckpt, policy_epoch_*.ckpt, policy_last.ckpt in both training dirs
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from pact_place_v109_contract import (  # noqa: E402
    TRAINING_ROOT, canonical_payload_sha256, empty_authorization, sha256_file,
    write_immutable_create_only,
)

REQUIRED_FREE_GIB = 30.0
RETAIN = ("policy_best.ckpt", "dataset_stats.pkl", "dataset_stats_manifest.json",
          "run_manifest.json", "epoch_log.jsonl", "train_episodes.txt")
PRUNE_GLOBS = ("resume_bundle.ckpt", "policy_epoch_*.ckpt", "policy_last.ckpt")
TREES = ("diagnostics_output/pact_place_v109_eval_traj",
         "assets/act_style_data/pact_place_v108_141")


def tree_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    closeout = json.loads(
        (ROOT / "diagnostics_output/pact_place_v109_train_eval/closeout.json").read_text())
    problems: list[str] = []
    retained: dict[str, dict[str, str]] = {}
    for arm in ("act", "pact"):
        directory = Path(TRAINING_ROOT) / f"{arm}_seed3101"
        recorded = closeout["training"]["arms"][arm]["hashes"]
        for name in RETAIN:
            path = directory / name
            if not path.is_file():
                problems.append(f"{arm}/{name} is missing before reclamation")
                continue
            digest = sha256_file(path)
            retained[f"{arm}/{name}"] = {"path": str(path), "sha256": digest}
            if name in recorded and recorded[name] != digest:
                problems.append(
                    f"{arm}/{name} hash {digest} != close-out {recorded[name]}")
    if problems:
        print(json.dumps({"verified": False, "problems": problems}, indent=2))
        return 1

    removals: list[dict[str, Any]] = []
    for relative in TREES:
        path = ROOT / relative
        if path.is_dir():
            removals.append({"kind": "tree", "path": relative,
                             "bytes": tree_bytes(path),
                             "files": sum(1 for _ in path.rglob("*") if _.is_file())})
    for arm in ("act", "pact"):
        directory = Path(TRAINING_ROOT) / f"{arm}_seed3101"
        for pattern in PRUNE_GLOBS:
            for path in sorted(directory.glob(pattern)):
                removals.append({"kind": "file", "path": str(path),
                                 "bytes": path.stat().st_size,
                                 "sha256": sha256_file(path)})
    projected = sum(r["bytes"] for r in removals)
    free_before = shutil.disk_usage("/root").free

    document: dict[str, Any] = {
        **empty_authorization(),
        "schema_version": "pact_place_v1010_deletion_manifest_v1",
        "role": "authorized storage reclamation before the V10.10 collection",
        "verified_against": "diagnostics_output/pact_place_v109_train_eval/closeout.json",
        "closeout_payload_sha256": closeout["payload_sha256"],
        "retained_verified": retained,
        "retained_count": len(retained),
        "removals": removals,
        "removal_count": len(removals),
        "projected_reclaim_bytes": projected,
        "projected_reclaim_gib": round(projected / 2**30, 2),
        "free_before_gib": round(free_before / 2**30, 2),
        "projected_free_after_gib": round((free_before + projected) / 2**30, 2),
        "required_free_gib": REQUIRED_FREE_GIB,
        "regenerable": {
            "assets/act_style_data/pact_place_v108_141":
                "regenerable from assets/datagen/pact_place_corridor_v10_8 via "
                "scripts/convert_pact_place_v109_to_act.py plus "
                "scripts/encode_pact_embedding_tokens.py",
        },
        "applied": bool(args.apply),
    }

    if not args.apply:
        document["payload_sha256"] = canonical_payload_sha256(document)
        print(json.dumps({k: document[k] for k in (
            "retained_count", "removal_count", "projected_reclaim_gib",
            "free_before_gib", "projected_free_after_gib", "required_free_gib")},
            indent=2))
        return 0

    write_immutable_create_only(
        ROOT / "diagnostics_output/pact_place_v1010_storage/deletion_manifest.json",
        {**document, "payload_sha256": canonical_payload_sha256(document)})

    for entry in removals:
        path = Path(entry["path"]) if entry["kind"] == "file" else ROOT / entry["path"]
        if entry["kind"] == "tree":
            shutil.rmtree(path)
        elif path.is_file():
            path.unlink()

    after = shutil.disk_usage("/root").free
    recheck = {k: sha256_file(Path(v["path"])) for k, v in retained.items()}
    drift = [k for k in retained if recheck[k] != retained[k]["sha256"]]
    report = {
        "applied": True,
        "free_after_gib": round(after / 2**30, 2),
        "required_free_gib": REQUIRED_FREE_GIB,
        "sufficient": after / 2**30 >= REQUIRED_FREE_GIB,
        "reclaimed_gib": round((after - free_before) / 2**30, 2),
        "retained_hash_drift": drift,
        "retained_reverified": len(recheck),
    }
    write_immutable_create_only(
        ROOT / "diagnostics_output/pact_place_v1010_storage/reclamation_report.json",
        {**empty_authorization(), **report,
         "schema_version": "pact_place_v1010_reclamation_report_v1"})
    print(json.dumps(report, indent=2))
    return 0 if report["sufficient"] and not drift else 1


if __name__ == "__main__":
    raise SystemExit(main())
