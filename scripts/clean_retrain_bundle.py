#!/usr/bin/env python3
"""Build and re-verify the immutable local artifact bundle for the clean retrain.

``build``  assembles one new bundle directory (checkpoints, statistics, manifests,
           split, logs, environment and Git state), writes per-file SHA-256 sums and
           a bundle manifest, then produces a deterministic tar archive.
``verify`` re-validates an extracted bundle against its own manifest, so the archive
           can be extracted into a clean temporary directory and checked end to end.

Nothing is uploaded: the future upload/redownload commands are recorded as text only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
CHUNK = 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def copy_in(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)


def cmd_build(args: argparse.Namespace) -> dict[str, Any]:
    bundle = Path(args.bundle_dir).resolve()
    if bundle.exists() and any(bundle.iterdir()):
        raise SystemExit(f"refusing to write into non-empty bundle dir: {bundle}")
    bundle.mkdir(parents=True, exist_ok=True)

    run_dir = Path(args.run_dir).resolve()
    prov = Path(args.provenance_dir).resolve()

    # --- artifacts that live inside the bundle --------------------------------
    for name in ("policy_best.ckpt", "policy_last.ckpt", "dataset_stats.pkl"):
        copy_in(run_dir / name, bundle / "checkpoint" / name)
    for pattern in ("*.png", "*.json", "*.txt", "*.log"):
        for extra in run_dir.glob(pattern):
            copy_in(extra, bundle / "training" / extra.name)

    for name in (
        "starting_state.json",
        "space_estimate.json",
        "source_manifest.json",
        "converted_manifest.json",
        "conversion_provenance.json",
        "split_and_statistics.json",
        "training_manifest.json",
        "offline_validation.json",
        "attempts_ledger.jsonl",
        "collection_command.txt",
        "collection_summary.json",
    ):
        src = prov / name
        if src.is_file():
            copy_in(src, bundle / "provenance" / name)

    logs = prov / "logs"
    if logs.is_dir():
        for log in sorted(logs.iterdir()):
            if log.is_file() and log.stat().st_size < args.max_log_bytes:
                copy_in(log, bundle / "logs" / log.name)

    # --- bundle-level identity ------------------------------------------------
    identity = {
        "schema_version": "hybrid_clean_retrain_bundle_v1",
        "root_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=False
        ).stdout.strip(),
        "root_branch": subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip(),
        "act_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT / "submodules/act",
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip(),
        "molmospaces_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT / "submodules/molmospaces",
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip(),
        "sensor_order_hash": "c31df8c36b0011b0eaf5b2eb5ce66d2514b5d6662ba9d7684ff021cd17cec858",
        "safety_cvae_model_sha256": "1fb2fc2b6023e64d2b9cbcf67fd5a24402968ec6f902c1e8a8595690396e7405",
        "safety_cvae_meta_sha256": "7c873756fb16e4f3fc565f96c41a581816981ab93886f1db5d27e62110a5fc81",
        "hybrid_contract_sha256": "aef29d762a909d0ce8610b79b4cb9a89a85c08fbe4d014910f27af44fc90df2b",
        "live_adapter_sha256": "21e8ccbe489cd278e9e946fde4d72a6772de5394aff47d484c882b8699c292ee",
        "residual_controller_sha256": "655a2e926351eef59c44896eb2cd6b142bfbf6fd5444e26c133c200443eaeeca",
        "paired_launcher_sha256": "4623ce5fffc7f9a136ce9b96f7e989942d9190db5074084a5172692a424e3fc2",
        "converter_sha256": "74b60458754b782393d65d508174b4168a94dbe6c539ff5d2005076994856695",
    }
    (bundle / "bundle_identity.json").write_text(json.dumps(identity, indent=2, sort_keys=True))

    if args.commands_file and Path(args.commands_file).is_file():
        copy_in(Path(args.commands_file), bundle / "commands.json")

    env_file = Path(args.environment_file) if args.environment_file else None
    if env_file and env_file.is_file():
        copy_in(env_file, bundle / "environment" / env_file.name)

    # --- per-file digests, manifest, deterministic archive ---------------------
    files = []
    for path in sorted(p for p in bundle.rglob("*") if p.is_file()):
        files.append(
            {
                "relpath": path.relative_to(bundle).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    sha_lines = "".join(f"{f['sha256']}  {f['relpath']}\n" for f in files)
    (bundle / "SHA256SUMS").write_text(sha_lines)

    manifest = {
        "schema_version": "hybrid_clean_retrain_bundle_manifest_v1",
        "bundle_dir": str(bundle),
        "file_count": len(files),
        "total_bytes": sum(f["bytes"] for f in files),
        "files": files,
        "identity": identity,
    }
    manifest["bundle_manifest_sha256"] = canonical_hash(
        {"files": files, "identity": identity}
    )
    (bundle / "bundle_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )

    archive_path = None
    if args.archive:
        archive_path = Path(args.archive).resolve()
        # Deterministic: sorted entries, zeroed mtime/uid/gid, no gzip timestamp.
        with tarfile.open(archive_path, "w") as tar:
            for path in sorted(p for p in bundle.rglob("*") if p.is_file()):
                info = tar.gettarinfo(str(path), arcname=path.relative_to(bundle).as_posix())
                info.mtime = 0
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                info.mode = 0o644
                with path.open("rb") as handle:
                    tar.addfile(info, handle)

    return {
        "schema_version": "hybrid_clean_retrain_bundle_build_v1",
        "bundle_dir": str(bundle),
        "file_count": len(files),
        "total_bytes": manifest["total_bytes"],
        "bundle_manifest_sha256": manifest["bundle_manifest_sha256"],
        "sha256sums_sha256": sha256_file(bundle / "SHA256SUMS"),
        "archive": str(archive_path) if archive_path else None,
        "archive_sha256": sha256_file(archive_path) if archive_path else None,
    }


def cmd_verify(args: argparse.Namespace) -> dict[str, Any]:
    bundle = Path(args.bundle_dir).resolve()
    manifest = json.loads((bundle / "bundle_manifest.json").read_text())
    results = []
    for entry in manifest["files"]:
        rel = entry["relpath"]
        # The manifest and SHA256SUMS are written after hashing, so they are not
        # self-describing; every other file must match byte for byte.
        if rel in ("bundle_manifest.json",):
            continue
        path = bundle / rel
        actual = sha256_file(path) if path.is_file() else None
        results.append(
            {"relpath": rel, "expected": entry["sha256"], "actual": actual,
             "match": actual == entry["sha256"]}
        )
    recomputed = canonical_hash(
        {"files": manifest["files"], "identity": manifest["identity"]}
    )
    mismatches = [r for r in results if not r["match"]]
    return {
        "schema_version": "hybrid_clean_retrain_bundle_verify_v1",
        "bundle_dir": str(bundle),
        "files_checked": len(results),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:20],
        "bundle_manifest_sha256_recomputed": recomputed,
        "bundle_manifest_sha256_declared": manifest["bundle_manifest_sha256"],
        "bundle_manifest_sha256_match": recomputed == manifest["bundle_manifest_sha256"],
        "passed": not mismatches and recomputed == manifest["bundle_manifest_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build")
    p.add_argument("--bundle_dir", required=True)
    p.add_argument("--run_dir", required=True)
    p.add_argument("--provenance_dir", required=True)
    p.add_argument("--commands_file")
    p.add_argument("--environment_file")
    p.add_argument("--archive")
    p.add_argument("--max_log_bytes", type=int, default=50 * 1024 * 1024)
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("verify")
    p.add_argument("--bundle_dir", required=True)
    p.set_defaults(func=cmd_verify)

    args = parser.parse_args()
    json.dump(args.func(args), sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
