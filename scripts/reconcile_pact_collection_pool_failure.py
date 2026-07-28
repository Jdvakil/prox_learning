#!/usr/bin/env python3
"""Freeze and apply conservative reconciliation for one collection-pool loss."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pact_collision_contract import canonical_json, load_manifest, rows_for_role


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _contract_sha256(value: dict[str, Any]) -> str:
    unhashed = {key: item for key, item in value.items() if key != "contract_sha256"}
    return hashlib.sha256(canonical_json(unhashed).encode()).hexdigest()


def build_contract(
    manifest_path: Path,
    collection: Path,
    role: str,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    summary_path = collection / f"{role}_summary.json"
    summary = json.loads(summary_path.read_text())
    if summary.get("complete") is not False or summary.get("role") != role:
        raise ValueError("expected an incomplete summary for the requested role")
    if summary.get("manifest_sha256") != manifest["manifest_sha256"]:
        raise ValueError("summary manifest hash mismatch")
    nonterminal = set(summary.get("nonterminal", []))
    rows = [row for row in rows_for_role(manifest, role) if row["episode_id"] in nonterminal]
    if len(rows) != len(nonterminal):
        raise ValueError("summary contains unknown nonterminal rows")

    terminalize = []
    first_attempt = []
    for row in rows:
        row_dir = collection / "rows" / row["episode_id"]
        if row_dir.exists():
            contents = sorted(path.name for path in row_dir.iterdir())
            if contents:
                raise ValueError(f"ambiguous row directory is not empty: {row_dir}")
            terminalize.append(
                {
                    "episode_id": row["episode_id"],
                    "role_index": int(row["role_index"]),
                    "row_sha256": row["row_sha256"],
                    "worker_assignment_evidence": {
                        "empty_row_directory": str(row_dir.resolve()),
                        "directory_mtime_utc": datetime.fromtimestamp(
                            row_dir.stat().st_mtime, tz=timezone.utc
                        ).isoformat(),
                    },
                    "action": "terminal_infrastructure_failure_no_retry",
                }
            )
        else:
            first_attempt.append(
                {
                    "episode_id": row["episode_id"],
                    "role_index": int(row["role_index"]),
                    "row_sha256": row["row_sha256"],
                    "action": "eligible_first_attempt_never_assigned",
                }
            )

    contract = {
        "schema_version": "pact_collection_pool_recovery_v1",
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": manifest["manifest_sha256"],
        "collection": str(collection.resolve()),
        "role": role,
        "failed_summary_path": str(summary_path.resolve()),
        "failed_summary_sha256": sha256_file(summary_path),
        "failed_status_counts": summary.get("status_counts", {}),
        "terminalize_assigned_ambiguous_rows": terminalize,
        "first_attempt_never_assigned_rows": first_attempt,
        "outcome_independence": (
            "classification uses only worker assignment evidence and empty directory "
            "state; no contact or task outcome exists for these rows"
        ),
        "conservative_rule": (
            "every assigned ambiguous row is terminalized without rerun, even if its "
            "failure may have preceded observation acceptance"
        ),
        "scene_endpoint_thresholds_changed": False,
        "scientific_rows_rerun": 0,
    }
    contract["contract_sha256"] = _contract_sha256(contract)
    return contract


def apply_contract(contract_path: Path) -> None:
    contract = json.loads(contract_path.read_text())
    if contract.get("contract_sha256") != _contract_sha256(contract):
        raise ValueError("recovery contract self-hash mismatch")
    summary_path = Path(contract["failed_summary_path"])
    if sha256_file(summary_path) != contract["failed_summary_sha256"]:
        raise ValueError("failed collection summary changed after contract freeze")
    collection = Path(contract["collection"])
    for entry in contract["terminalize_assigned_ambiguous_rows"]:
        row_dir = collection / "rows" / entry["episode_id"]
        result_path = row_dir / "result.json"
        if result_path.exists():
            result = json.loads(result_path.read_text())
            if result.get("recovery_contract_sha256") != contract["contract_sha256"]:
                raise ValueError(f"unexpected existing result: {result_path}")
            continue
        if sorted(row_dir.iterdir()):
            raise ValueError(f"refusing non-empty ambiguous row: {row_dir}")
        result = {
            "schema_version": "pact_collision_collection_result_v2",
            "status": "infrastructure_failure",
            "episode_id": entry["episode_id"],
            "row_sha256": entry["row_sha256"],
            "manifest_sha256": contract["manifest_sha256"],
            "role": contract["role"],
            "role_index": entry["role_index"],
            "rollout_started": "unknown_conservatively_terminal",
            "error": (
                "worker pool broke after assignment; durable boundary state was "
                "unavailable, so the row is terminalized without rerun"
            ),
            "task_success": False,
            "collision_free_task_success": False,
            "recovery_contract_sha256": contract["contract_sha256"],
            "scientific_row_rerun": False,
        }
        _write_json_atomic(result_path, result)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--collection", required=True, type=Path)
    parser.add_argument("--role", required=True)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.apply:
        apply_contract(args.contract)
        print(json.loads(args.contract.read_text())["contract_sha256"])
        return 0
    contract = build_contract(args.manifest, args.collection, args.role)
    _write_json_atomic(args.contract, contract)
    print(contract["contract_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
