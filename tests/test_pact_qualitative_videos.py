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
CLIPS_V2_ARTIFACT = (
    ROOT
    / "diagnostics_output/pact_contact_endpoint/qualitative_clips_v2_manifest.json"
)
CLIPS_V3_ARTIFACT = (
    ROOT
    / "diagnostics_output/pact_contact_endpoint/qualitative_clips_v3_manifest.json"
)
CLIP3_FALLBACK_ARTIFACT = (
    ROOT
    / "diagnostics_output/pact_contact_endpoint/qualitative_clip3_fallback_manifest.json"
)
CLIP3_FALLBACK2_ARTIFACT = (
    ROOT
    / "diagnostics_output/pact_contact_endpoint/qualitative_clip3_fallback_rank2_manifest.json"
)
ACT_SUCCESS_SUPPLEMENT_ARTIFACT = (
    ROOT
    / "diagnostics_output/pact_contact_endpoint/qualitative_act_success_supplement.json"
)
RECORDED_ACT_SUCCESS_ARTIFACT = (
    ROOT
    / "diagnostics_output/pact_contact_endpoint/qualitative_recorded_act_success_supplement.json"
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


def test_v2_manifest_freezes_four_fixed_single_arm_clips() -> None:
    document = json.loads(CLIPS_V2_ARTIFACT.read_text())
    payload = dict(document)
    observed = payload.pop("qualitative_clips_v2_manifest_sha256")
    assert observed == selection.canonical_hash(payload)
    assert document["status"] == "presentation_release_incomplete_determinism_drop"
    assert document["selection_candidate_counts"] == {
        "instance_a": 48,
        "instance_b": 34,
    }
    assert [clip["clip_id"] for clip in document["clips"]] == [
        "clip1_54a6272f66ca_pact_success",
        "clip2_54a6272f66ca_act_failure",
        "clip3_e99dc657bfa7_act_success",
        "clip4_e99dc657bfa7_pact_failure",
    ]
    assert [clip["original_outcome"]["hazard_frames"] for clip in document["clips"]] == [
        0,
        29022,
        19757,
        17609,
    ]
    assert [clip["original_outcome"]["task_success"] for clip in document["clips"]] == [
        True,
        False,
        True,
        False,
    ]
    assert document["render_contract"]["playback_speed_factor"] == 3.0
    assert document["selection_frozen_manifest_sha256"] == (
        "febdaffb8ca9b6b9c9eb7b39ad7557eb4f44cca7ad815fbf17214a3e528b1351"
    )
    assert document["determinism_summary"]["required_exact_fields_failed"] == 1
    assert document["determinism_summary"]["all_four_clips_retained"] is False
    assert document["determinism_summary"]["dropped_clip_ids"] == [
        "clip3_e99dc657bfa7_act_success"
    ]
    assert len(document["render_outputs"]) == 3


def test_v2_overlay_renders_frozen_outcome_and_running_contact_count() -> None:
    frame = np.zeros((352, 624, 3), dtype=np.uint8)
    observed = qualitative.overlay_frame_v2(
        frame,
        arm="ACT",
        cumulative_hazard_frames=29022,
        episode_id="54a6272f66ca" + "0" * 52,
        policy_seed=3101,
        task_success=False,
        any_hazard_contact=True,
        maximum_hazard_penetration_m=0.0008325859297544908,
        playback_speed_factor=3.0,
    )
    assert observed.shape == frame.shape
    assert observed.dtype == np.uint8
    assert np.count_nonzero(observed) > 0


def test_v3_manifest_honestly_drops_seed3102_pair_under_frozen_gate() -> None:
    document = json.loads(CLIPS_V3_ARTIFACT.read_text())
    payload = dict(document)
    observed = payload.pop("qualitative_clips_v3_manifest_sha256")
    assert observed == selection.canonical_hash(payload)
    assert document["status"] == "presentation_release_incomplete_gate_drop"
    assert document["selection_and_gate_frozen_manifest_sha256"] == (
        "4d0440f0feef4b2c39bdf1ca5e81da5d5402e9891cfd680f3bd6c936ca092b26"
    )
    gate = document["determinism_gate"]
    assert gate["declared_before_render"] is True
    assert gate["task_success"] == {"comparison": "exact"}
    assert gate["first_hazard_bar_contact_step"] == {"comparison": "exact"}
    assert gate["first_grasp_target_contact_step"]["tolerance_steps"] == 2
    assert gate["contact_pair_sample_counts"]["comparison"] == (
        "informational_only"
    )
    clips = document["clips"]
    assert [(clip["arm"], clip["checkpoint_seed"]) for clip in clips] == [
        ("ACT", 3102),
        ("PACT", 3102),
    ]
    assert len({clip["episode_id"] for clip in clips}) == 1
    assert [clip["hazard_frames"] for clip in clips] == [18447, 14675]
    assert [clip["grasp_target_frames"] for clip in clips] == [24923, 809]
    assert [clip["task_success"] for clip in clips] == [True, False]
    summary = document["determinism_summary"]
    assert summary["declared_gate_failed"] == 2
    assert summary["all_clips_retained"] is False
    assert len(document["render_outputs"]) == 0
    for check in summary["per_clip_checks"].values():
        assert check["declared_gate_passed"] is False
        comparisons = check["required_gate_comparisons"]
        assert comparisons["first_hazard_bar_contact_step"]["passed"] is False
        assert comparisons["first_grasp_target_contact_step"]["passed"] is True


def test_clip3_fallback_rank1_is_dropped_by_exact_hazard_gate() -> None:
    document = json.loads(CLIP3_FALLBACK_ARTIFACT.read_text())
    payload = dict(document)
    observed = payload.pop("qualitative_clip3_fallback_manifest_sha256")
    assert observed == selection.canonical_hash(payload)
    assert document["status"] == "fallback_rank1_dropped_gate_failure"
    assert document["selection_and_gate_frozen_manifest_sha256"] == (
        "1a2bd102f6aa081a04dead53a6e36b1456fff48f218b1d59d5006971fb9042e8"
    )
    assert document["fallback_rank"] == 1
    assert document["pairing_claim_allowed"] is False
    clip = document["clip"]
    assert clip["clip_id"] == "clip3_3fe3a173f2bf_act_success_s3103"
    assert clip["source_directory"] == "1133_c39a3a8fdd8c6ae4_act_s3103"
    assert clip["hazard_frames"] == 16739
    assert clip["grasp_target_frames"] == 25126
    assert clip["task_success"] is True
    check = document["determinism_check"]
    assert check["declared_gate_passed"] is False
    comparisons = check["required_gate_comparisons"]
    assert comparisons["task_success"]["passed"] is True
    assert comparisons["first_hazard_bar_contact_step"]["original"] == 393
    assert comparisons["first_hazard_bar_contact_step"]["rerun"] == 392
    assert comparisons["first_hazard_bar_contact_step"]["passed"] is False
    assert comparisons["first_grasp_target_contact_step"]["passed"] is True
    assert document["render_output"] is None


def test_clip3_fallback_rank2_passes_and_drops_pairing_claim() -> None:
    document = json.loads(CLIP3_FALLBACK2_ARTIFACT.read_text())
    payload = dict(document)
    observed = payload.pop("qualitative_clip3_fallback_rank2_manifest_sha256")
    assert observed == selection.canonical_hash(payload)
    assert document["status"] == "presentation_release_verified_unmatched_fallback"
    assert document["selection_and_gate_frozen_manifest_sha256"] == (
        "ab299127500652285b3b32db4128ee04f6c73dd9f5e2dd795719e599a6b436b3"
    )
    assert document["fallback_rank"] == 2
    assert document["pairing_claim_allowed"] is False
    clip = document["clip"]
    assert clip["clip_id"] == "clip3_178a8383cda2_act_success_s3101"
    assert clip["source_directory"] == "0282_dfbef2c46b0cf55b_act_s3101"
    assert clip["hazard_frames"] == 12087
    assert clip["grasp_target_frames"] == 24686
    assert clip["task_success"] is True
    check = document["determinism_check"]
    assert check["declared_gate_passed"] is True
    comparisons = check["required_gate_comparisons"]
    assert comparisons["task_success"]["passed"] is True
    assert comparisons["first_hazard_bar_contact_step"]["passed"] is True
    assert comparisons["first_grasp_target_contact_step"]["passed"] is True
    check_document = json.loads(
        Path(document["render_output"]["determinism_check_path"]).read_text()
    )
    assert check_document["informational_contact_frame_deltas"]["hazard_bar"] == {
        "original": 12087,
        "rerun": 12140,
        "signed_delta": 53,
    }
    duration = float(
        document["render_output"]["release_ffprobe"]["format"]["duration"]
    )
    assert 15.0 <= duration <= 25.0


def test_contact_endpoint_act_success_attempt_is_honestly_dropped() -> None:
    document = json.loads(ACT_SUCCESS_SUPPLEMENT_ARTIFACT.read_text())
    payload = dict(document)
    observed = payload.pop("act_success_supplement_manifest_sha256")
    assert observed == selection.canonical_hash(payload)
    assert document["status"] == "dropped_determinism_mismatch"
    assert document["selection_frozen_manifest_sha256"] == (
        "97097d1702683365afeb8ee0f3dd0b684f55156c343e348d32234cc05ab7534e"
    )
    assert document["selection_candidate_count"] == 169
    clip = document["clip"]
    assert clip["clip_id"] == "clip3_25d96dd30260_act_success"
    assert clip["source_directory"] == "0011_429ae21412b1a318_act_s3103"
    assert clip["schedule_index"] == 11
    assert clip["task_success"] is True
    assert clip["hazard_frames"] == 0
    assert clip["grasp_target_frames"] == 24576
    assert document["determinism_check"]["required_exact_match"] is False
    comparisons = document["determinism_check"]["required_exact_comparisons"]
    assert comparisons["task_success"]["exact_match"] is True
    assert comparisons["first_contact_step"]["original"]["grasp_target"] == 154
    assert comparisons["first_contact_step"]["rerun"]["grasp_target"] == 153
    assert document["render_output"] is None


def test_recorded_act_success_uses_original_frozen_footage() -> None:
    document = json.loads(RECORDED_ACT_SUCCESS_ARTIFACT.read_text())
    payload = dict(document)
    observed = payload.pop("recorded_act_success_supplement_manifest_sha256")
    assert observed == selection.canonical_hash(payload)
    assert document["status"] == "presentation_release_verified_original_recording"
    assert document["selection_frozen_manifest_sha256"] == (
        "b36c7dbcca32e2eca3bb3ed1f321daffae3436847ddb8dbeead2b34188dcc903"
    )
    assert document["selection_candidate_count"] == 19
    clip = document["clip"]
    assert clip["clip_id"] == "clip3_5b4288fea187_act_success_wrist"
    assert clip["source_directory"] == "000_00683ce9e651981d_act"
    assert clip["schedule_index"] == 0
    assert clip["task_success"] is True
    assert clip["hazard_frames"] == 0
    assert clip["grasp_target_frames"] == 24769
    assert document["render_contract"]["policy_rerun"] is False
    render = document["render_output"]
    assert render["source_video_used_byte_for_byte_before_overlay"] is True
    assert 15.0 <= float(render["release_ffprobe"]["format"]["duration"]) <= 25.0
