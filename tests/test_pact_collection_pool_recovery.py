from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "reconcile_pact_collection_pool_failure",
        ROOT / "scripts" / "reconcile_pact_collection_pool_failure.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


recovery = _load()


def test_contract_hash_excludes_only_its_self_hash():
    value = {
        "schema_version": "pact_collection_pool_recovery_v1",
        "rows": ["a", "b"],
    }
    digest = recovery._contract_sha256(value)
    value["contract_sha256"] = digest
    assert recovery._contract_sha256(value) == digest


def test_apply_terminalizes_without_claiming_scientific_outcome(tmp_path):
    collection = tmp_path / "collection"
    row_dir = collection / "rows" / ("e" * 64)
    row_dir.mkdir(parents=True)
    summary = collection / "full_train_summary.json"
    summary.write_text("{}")
    contract = {
        "schema_version": "pact_collection_pool_recovery_v1",
        "manifest_sha256": "m" * 64,
        "collection": str(collection),
        "role": "full_train",
        "failed_summary_path": str(summary),
        "failed_summary_sha256": recovery.sha256_file(summary),
        "terminalize_assigned_ambiguous_rows": [
            {
                "episode_id": "e" * 64,
                "role_index": 226,
                "row_sha256": "r" * 64,
            }
        ],
        "first_attempt_never_assigned_rows": [],
        "scientific_rows_rerun": 0,
    }
    contract["contract_sha256"] = recovery._contract_sha256(contract)
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract))
    recovery.apply_contract(path)
    result = json.loads((row_dir / "result.json").read_text())
    assert result["status"] == "infrastructure_failure"
    assert result["rollout_started"] == "unknown_conservatively_terminal"
    assert result["scientific_row_rerun"] is False
    assert result["task_success"] is False
