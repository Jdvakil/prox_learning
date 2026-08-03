from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "submodules/act"))
sys.path.insert(0, str(ROOT / "submodules/molmospaces"))

import build_pact_qualitative_video_manifest as selection
import eval_pact_qualitative_row as qualitative


ARTIFACT = (
    ROOT / "diagnostics_output/pact_contact_endpoint/qualitative_video_manifest.json"
)


def pair(episode: str, seed: int, act: int, pact: int, *, success: bool = True):
    def arm_record(frames: int):
        return {
            "original_outcome": {
                "frames_with_contact": {"hazard_bar": frames},
                "task_success": success,
            }
        }

    return {
        "episode_id": episode,
        "policy_seed": seed,
        "instance_role_index": 0,
        "arms": {"ACT": arm_record(act), "PACT": arm_record(pact)},
    }


def test_selection_rule_is_mechanical_and_includes_counterexample() -> None:
    records = {
        ("m1", 1): pair("m1", 1, 700, 0),
        ("m2", 1): pair("m2", 1, 900, 0),
        ("m3", 1): pair("m3", 1, 800, 0),
        ("routine_b", 1): pair("routine_b", 1, 0, 0),
        ("routine_a", 2): pair("routine_a", 2, 0, 0),
        ("counter", 1): pair("counter", 1, 5, 1200),
    }
    observed = selection.select(records)
    assert [item["episode_id"] for item in observed[:3]] == ["m2", "m3", "m1"]
    assert observed[3]["episode_id"] == "routine_a"
    assert observed[4]["episode_id"] == "counter"
    assert observed[4]["category"] == "counterexample"


def test_frozen_selection_manifest_is_self_hashed_and_has_ten_rows() -> None:
    document = json.loads(ARTIFACT.read_text())
    payload = dict(document)
    observed = payload.pop("qualitative_video_manifest_sha256")
    assert observed == selection.canonical_hash(payload)
    assert document["status"] == "aborted_determinism_mismatch"
    assert document["illustrative_only"] is True
    assert document["decision_bearing"] is False
    assert len(document["selections"]) == 5
    assert [item["category"] for item in document["selections"]] == [
        "mechanism",
        "mechanism",
        "mechanism",
        "routine",
        "counterexample",
    ]
    rows = [
        record["schedule_index"]
        for item in document["selections"]
        for record in item["arms"].values()
    ]
    assert len(rows) == len(set(rows)) == 10
    assert document["determinism_check"]["exact_match"] is False
    assert document["determinism_check"]["action"] == (
        "stopped before the remaining nine reruns"
    )
    assert document["completion"] == {
        "requested_paired_videos": 5,
        "completed_paired_videos": 0,
        "remaining_selected_reruns_launched": 0,
        "stopped_by_predeclared_gate": True,
        "scientific_results_or_token_changed": False,
    }


def test_render_contract_keeps_third_person_camera_out_of_policy() -> None:
    document = json.loads(ARTIFACT.read_text())
    contract = document["render_contract"]
    assert contract["registered_sensor_or_observation_camera"] is False
    assert contract["observation_key_added"] is False
    assert contract["policy_camera_names"] == ["wrist_camera"]
    assert contract["rng_calls_added"] == 0
    assert qualitative.CAMERA_REFERENCE_BODY == "robot_0/fr3_link0"
    assert qualitative.EXPECTED_RESOLUTION == (624, 352)


def test_overlay_renders_required_identity_and_contact_state() -> None:
    frame = np.zeros((352, 624, 3), dtype=np.uint8)
    observed = qualitative.overlay_frame(
        frame,
        arm="PACT",
        step=123,
        active=True,
        cumulative_hazard_frames=456,
        episode_id="a" * 64,
        policy_seed=3101,
    )
    assert observed.shape == frame.shape
    assert observed.dtype == np.uint8
    assert np.count_nonzero(observed) > 0
