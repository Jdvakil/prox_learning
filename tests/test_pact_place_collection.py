from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import pact_place_collection_contract as collection  # noqa: E402
import pact_place_corridor_contract as contract  # noqa: E402


def test_collection_contract_is_balanced_and_pins_v5() -> None:
    document = collection.build_collection_contract()
    collection.validate_collection_contract(document)
    assert document["master_seed"] == 2026082301
    assert len(document["collection_rows"]) == 310
    assert sum(row["intrusion_side"] == "left" for row in document["collection_rows"]) == 155
    assert document["collection"]["target_clean"] == 255
    assert document["collection"]["do_not_lower_the_filter"] is True
    assert document["collection"]["encoder_training_eval_not_authorized"] is True
    assert document["screen_config_sha256"] == (
        "bd47f1c97d2815657211085590657f5211ca847b776f6039c9617f990da9c1f1"
    )
    assert document["scene"]["xml"].endswith("pact_place_corridor_v2.xml")
    assert document["collection"]["yield_floor"] == pytest.approx(22 / 24 - 0.10)
    screen_ids = {
        row["episode_id"]
        for row in json.loads((ROOT / "configs/pact_place_corridor_v5.json").read_text())[
            "expert_screen_rows"
        ]
    }
    collect_ids = {row["episode_id"] for row in document["collection_rows"]}
    assert collect_ids.isdisjoint(screen_ids)


def test_frozen_collection_contract_if_present() -> None:
    path = ROOT / "configs/pact_place_corridor_v5_collection.json"
    if not path.exists():
        pytest.skip("collection contract has not been generated")
    saved = json.loads(path.read_text())
    collection.validate_collection_contract(saved)
    assert saved["master_seed"] == 2026082301
    assert len(saved["collection_rows"]) == 310
    assert saved["screen_config_sha256"] == (
        "bd47f1c97d2815657211085590657f5211ca847b776f6039c9617f990da9c1f1"
    )
    assert saved["config_sha256"] == (
        "ea69183efe388eaeb051a654859d93a6120955a8751294cd34ce06772e469cb3"
    )
    live = collection.build_collection_contract()
    assert live["screen_config_sha256"] == saved["screen_config_sha256"]
    for relative, digest in saved["protected_artifact_sha256_before"].items():
        assert contract.sha256_file(ROOT / relative) == digest
