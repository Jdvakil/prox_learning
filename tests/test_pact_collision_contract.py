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


contract = _load("pact_collision_contract", ROOT / "scripts" / "pact_collision_contract.py")
contacts = _load(
    "pact_contact_audit",
    ROOT
    / "submodules"
    / "molmospaces"
    / "molmo_spaces"
    / "tasks"
    / "pact_contact_audit.py",
)


def test_role_counts_are_balanced_and_minimum_eval_is_exceeded():
    assert contract.ROLE_COUNTS["pilot_train"] == 24
    assert contract.ROLE_COUNTS["pilot_eval"] == 24
    assert contract.ROLE_COUNTS["confirmatory_eval"] == 80
    assert contract.ROLE_COUNTS["confirmatory_eval"] >= 50
    for role, count in contract.ROLE_COUNTS.items():
        sides = contract._balanced_sides(role, count)
        assert sides.count("left") == count // 2
        assert sides.count("right") == count // 2


def test_seed_is_repeatable_and_retry_specific():
    row0 = contract.derive_seed(9, contract.TASK_STREAM_ID)
    assert row0 == contract.derive_seed(9, contract.TASK_STREAM_ID)
    assert row0 != contract.derive_seed(10, contract.TASK_STREAM_ID)
    row = {"candidate_index": 9, "master_seed": contract.MASTER_SEED}
    assert contract.retry_seed(row, 0) != contract.retry_seed(row, 1)


def test_manifest_round_trip_and_tamper_detection(tmp_path):
    source_hashes = {"scene_xml": "a" * 64}
    sensors = [f"sensor_{index}" for index in range(40)]
    first = contract.build_manifest(source_hashes=source_hashes, sensor_names=sensors)
    second = contract.build_manifest(source_hashes=source_hashes, sensor_names=sensors)
    assert first == second
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(first))
    assert contract.load_manifest(path) == first

    damaged = json.loads(json.dumps(first))
    damaged["rows"][0]["intrusion_side"] = "right"
    with pytest.raises(contract.PactContractError):
        contract.validate_manifest(damaged)


def test_manifest_identity_excludes_worker_and_output_layout():
    source_hashes = {"scene_xml": "b" * 64}
    sensors = [f"sensor_{index}" for index in range(40)]
    document = contract.build_manifest(source_hashes=source_hashes, sensor_names=sensors)
    forbidden = {"worker_id", "worker_count", "output_dir", "file_path"}
    for row in document["rows"]:
        assert forbidden.isdisjoint(row)


def test_contact_taxonomy_exempts_only_target_and_separates_intrusion():
    target = {"root2": "cavity_obj_0/red_cup"}
    panel = {"body1": "pact_intrusion_left"}
    wall = {"body2": "fumehood_wall"}
    assert contacts.classify_contact(target) == "grasp_target"
    assert contacts.classify_contact(panel) == "hazard_bar"
    assert contacts.classify_contact(wall) == "other_environment"
