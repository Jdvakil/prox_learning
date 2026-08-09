from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_pact_geometry_v3_video_manifest.py"


def load_builder():
    spec = importlib.util.spec_from_file_location("geometry_video_builder", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_geometry_video_selection_and_gate_are_fixed() -> None:
    builder = load_builder()
    document = builder.build_document()
    assert document["status"] == "selection_and_gate_frozen_pre_render"
    assert document["selection_contract"]["qualifying_counts"] == {
        "C2": 14,
        "Z_093": 18,
    }
    selections = {
        (item["condition_id"], item["selection_rank"]): item
        for item in document["ranked_selections"]
    }
    assert selections[("C2", 1)]["arms"]["PACT"]["schedule_index"] == 269
    assert selections[("C2", 1)]["arms"]["PACT_PERMUTED"]["schedule_index"] == 268
    assert selections[("Z_093", 1)]["arms"]["PACT"]["schedule_index"] == 607
    assert selections[("Z_093", 1)]["arms"]["PACT_PERMUTED"]["schedule_index"] == 606
    gate = document["determinism_gate"]
    assert gate["task_success"]["comparison"] == "exact"
    assert gate["first_hazard_bar_contact_step"]["comparison"] == "exact"
    assert gate["first_grasp_target_contact_step"]["tolerance_steps"] == 2
    assert gate["contact_pair_sample_counts"]["comparison"] == "informational_only"
    payload = dict(document)
    observed = payload.pop("qualitative_video_manifest_sha256")
    assert observed == builder.canonical_hash(payload)


def test_ranked_candidates_are_complete_matched_pairs() -> None:
    builder = load_builder()
    document = builder.build_document()
    assert len(document["ranked_selections"]) == 6
    for item in document["ranked_selections"]:
        assert set(item["arms"]) == {"PACT", "PACT_PERMUTED"}
        pact = item["arms"]["PACT"]["outcome"]
        perm = item["arms"]["PACT_PERMUTED"]["outcome"]
        assert pact["hazard_contact_frames"] == 0
        assert perm["hazard_contact_frames"] > 500


def test_scientific_record_hashes_are_bound() -> None:
    builder = load_builder()
    document = builder.build_document()
    sources = document["sources"]
    assert sources["report"]["sha256"] == "1434977069bb45d3d0548d25a0f6fd7ce73d5dad6523d383226f58d4a59c2d49"
    assert sources["analysis"]["sha256"] == "29957d18f55139e7dc993358e4d77b39202e535a868e6adf96483f3d8bb4eaa1"
    assert sources["final_decision"]["sha256"] == "ce8bb6106c47e08947670d6958e6df02876e6c417b1c754450695aa840c7670c"
