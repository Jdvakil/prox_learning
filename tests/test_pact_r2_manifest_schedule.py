from __future__ import annotations

import importlib.util
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


contract = _load("pact_r2_contract_test", ROOT / "scripts/pact_r2_contract.py")
schedule = _load(
    "build_pact_r2_schedule_test",
    ROOT / "scripts/build_pact_r2_schedule.py",
)


def test_fresh_manifest_is_balanced_and_disjoint_from_r1():
    r1 = json.loads(
        (ROOT / "configs/pact_collision_candidate_manifest_v2.json").read_text()
    )
    r1_ids = {row["episode_id"] for row in r1["rows"]}
    document = contract.build_manifest(
        source_hashes={"unit": "a" * 64},
        sensor_names=[f"sensor_{index}" for index in range(40)],
        r1_episode_ids=r1_ids,
        r1_manifest_sha256=r1["manifest_sha256"],
    )
    assert len(document["rows"]) == 160
    assert not ({row["episode_id"] for row in document["rows"]} & r1_ids)
    assert Counter(row["intrusion_side"] for row in document["rows"]) == {
        "left": 80,
        "right": 80,
    }


def test_r2_condition_order_is_deterministic_and_balanced():
    first = schedule.condition_orders(160)
    assert first == schedule.condition_orders(160)
    assert all(set(order) == set(schedule.CONDITIONS) for order in first)
    for position in range(6):
        counts = Counter(order[position] for order in first)
        assert max(counts.values()) - min(counts.values()) <= 1
