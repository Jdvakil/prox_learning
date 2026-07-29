from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


freeze = _load(
    "freeze_pact_confirmatory_dispatch",
    ROOT / "scripts" / "freeze_pact_confirmatory_dispatch.py",
)
runner = _load(
    "run_pact_confirmatory_schedule_contract",
    ROOT / "scripts" / "run_pact_confirmatory_schedule.py",
)


def test_outcome_artifacts_refuse_contract_freeze(tmp_path):
    output_root = tmp_path / "confirmatory"
    row = output_root / "rows" / "000"
    row.mkdir(parents=True)
    (row / "result.json").write_text("{}")
    with pytest.raises(ValueError, match="already contains outcome"):
        freeze.assert_no_outcomes(output_root)


def test_runner_binds_manifest_and_output_root(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    output_root = tmp_path / "out"
    row = {
        "schedule_index": 0,
        "rollout_id": "rollout",
        "instance_episode_id": "episode",
        "schedule_row_sha256": "row",
        "output_relpath": "rows/000",
    }
    schedule = {"schedule_sha256": "schedule", "workers": 8, "rows": [row]}
    contract = {
        "schema_version": "pact_confirmatory_dispatch_v2",
        "scientific_schedule": {
            "schedule_sha256": "schedule",
            "rows": 1,
            "workers": 8,
            "rows_changed": 0,
            "manifest_path": str(manifest),
            "manifest_sha256": runner.sha256_file(manifest),
        },
        "execution": {"output_root": str(output_root)},
        "launch_smoke": {
            "required_before_full_dispatch": True,
            "schedule_index": 0,
            "rollout_id": "rollout",
            "instance_episode_id": "episode",
            "schedule_row_sha256": "row",
            "output_relpath": "rows/000",
        },
    }
    contract["dispatch_contract_sha256"] = runner.canonical_hash(contract)
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract))
    loaded = runner.load_dispatch_contract(
        contract_path,
        schedule,
        manifest_path=manifest,
        output_root=output_root,
    )
    assert loaded["dispatch_contract_sha256"] == contract[
        "dispatch_contract_sha256"
    ]
    with pytest.raises(RuntimeError, match="output root"):
        runner.load_dispatch_contract(
            contract_path,
            schedule,
            manifest_path=manifest,
            output_root=tmp_path / "wrong",
        )
