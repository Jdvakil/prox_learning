from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "build_pact_act_split", ROOT / "scripts" / "build_pact_act_split.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


builder = _load()


def _episodes(roles):
    return [
        {
            "act_episode_index": index,
            "episode_id": str(index),
            "candidate_index": index + 100,
            "role": role,
            "source_h5_sha256": "a" * 64,
        }
        for index, role in enumerate(roles)
    ]


def test_pilot_split_is_frozen_eighty_twenty_order():
    conversion = {
        "roles": ["pilot_train"],
        "episodes": _episodes(["pilot_train"] * 10),
        "source_manifest_sha256": "m",
        "converted_tree_semantic_sha256": "t",
    }
    split = builder.build(conversion, "pilot")
    assert split["counts"]["train"]["total"] == 8
    assert split["counts"]["validation"]["total"] == 2
    assert [episode["act_episode_index"] for episode in split["episodes"][:8]] == list(
        range(8)
    )


def test_full_split_uses_predeclared_roles():
    roles = ["full_train"] * 3 + ["full_validation"] * 2
    conversion = {
        "roles": ["full_train", "full_validation"],
        "episodes": _episodes(roles),
        "source_manifest_sha256": "m",
        "converted_tree_semantic_sha256": "t",
    }
    split = builder.build(conversion, "full")
    assert split["counts"]["train"]["total"] == 3
    assert split["counts"]["validation"]["total"] == 2


def test_full_split_accepts_post_token_file_tree_hash():
    roles = ["full_train"] * 3 + ["full_validation"] * 2
    conversion = {
        "roles": ["full_train", "full_validation"],
        "episodes": _episodes(roles),
        "source_manifest_sha256": "m",
        "converted_tree_file_sha256": "f" * 64,
    }
    split = builder.build(conversion, "full")
    assert split["source_collection_tree_sha256"] == "f" * 64
    assert split["source_collection_tree_hash_kind"] == "file"
