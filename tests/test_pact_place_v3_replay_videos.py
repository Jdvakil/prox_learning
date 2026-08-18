from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_pact_place_v3_replay_videos as replay


def test_failure_clip_names_match_the_plan() -> None:
    assert replay.clip_stem(2, clean_success=False) == (
        "row02_FAIL_outbound_approach_cup_lost"
    )
    assert replay.clip_stem(6, clean_success=False) == (
        "row06_FAIL_placement_descent_not_released"
    )
    assert replay.clip_stem(9, clean_success=False) == (
        "row09_FAIL_outbound_approach_tracking_12.6cm"
    )
    assert replay.clip_stem(12, clean_success=False) == (
        "row12_FAIL_pregrasp_never_grasped"
    )
    assert replay.clip_stem(20, clean_success=False) == (
        "row20_FAIL_lift_cup_dropped"
    )
    assert replay.clip_stem(21, clean_success=False) == (
        "row21_FAIL_outbound_pass_cup_lost"
    )
    assert replay.clip_stem(0, clean_success=True) == "row00_clean_success"
    with pytest.raises(ValueError):
        replay.clip_stem(3, clean_success=False)


def test_overlay_renders_the_five_diagnostic_fields() -> None:
    wrist = np.zeros((352, 624, 3), dtype=np.uint8)
    third = np.zeros((352, 624, 3), dtype=np.uint8)
    wrist[:] = (10, 20, 30)
    third[:] = (40, 50, 60)
    observed = replay.overlay_composite(
        wrist,
        third,
        role_index=9,
        clean_success=False,
        step=233,
        n_steps=234,
        policy_phase="outbound_approach",
        gripper_width_m=0.019,
        object_dz_m=0.012,
        tcp_plane_dz_m=0.0804,
        hazard_bar_frames=0,
        terminal=True,
    )
    assert observed.shape == (352, 1248, 3)
    assert np.count_nonzero(observed) > 0
    assert replay.OVERLAY_FIELDS == (
        "policy_phase + step",
        "gripper width (m)",
        "object z minus start z",
        "TCP z minus carry-plane z",
        "running hazard_bar frames",
    )


def test_replay_loop_is_forward_only() -> None:
    source = Path(replay.__file__).read_text()
    replay.assert_replay_source_is_forward_only(source)
    tree = ast.parse(source)
    imports = [
        ast.unparse(node)
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    dumped = "\n".join(imports)
    assert "ParallelRolloutRunner" not in dumped
    assert "setup_policy" not in dumped


def test_crib_states_the_six_failure_claims() -> None:
    text = replay.crib_markdown(output_dir=replay.DEFAULT_OUTPUT, fps=replay.FPS)
    assert "row09_FAIL_outbound_approach_tracking_12.6cm.mp4" in text
    assert "row12_FAIL_pregrasp_never_grasped.mp4" in text
    assert "arm climbs out of the carry plane" in text
    assert "v2 hazard rows 6 and 12" in text
    assert "out of scope" in text
    assert "not a re-run" in text


def test_faithfulness_tolerance_and_spot_checks() -> None:
    assert replay.INDEXING_TOLERANCE_M == pytest.approx(1e-6)
    assert replay.DERIVED_TOLERANCE_M == pytest.approx(1e-3)
    assert replay.spot_check_indices(14) == [0, 3, 7, 10, 13]
    assert replay.spot_check_indices(2) == [0, 1]
    steps = [
        {"policy_phase": "pregrasp", "tcp_position_m": [0.0, 0.0, 0.80]},
        {"policy_phase": "lift", "tcp_position_m": [0.0, 0.0, 0.90]},
        {"policy_phase": "lift", "tcp_position_m": [0.0, 0.0, 0.9062]},
        {"policy_phase": "outbound_approach", "tcp_position_m": [0.0, 0.0, 0.9866]},
    ]
    assert replay.carry_plane_z_m(steps) == pytest.approx(0.9062)


def test_v3_phase0_and_protected_hashes_are_untouched() -> None:
    contract = json.loads((ROOT / "configs/pact_place_corridor_v3.json").read_text())
    assert contract["config_sha256"] == replay.V3_CONFIG_SHA256
    from run_pact_place_expert_screen import verify_protected_artifacts

    verify_protected_artifacts(contract)
    for relative, digest in contract["protected_artifact_sha256_before"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest
    v3 = json.loads(
        (ROOT / "diagnostics_output/pact_place_corridor_v3/expert_screen.json").read_text()
    )
    assert v3["decision"] == "PACT_PLACE_CORRIDOR_PHASE0_FAIL"
    v2 = ROOT / "diagnostics_output/pact_place_corridor_v2"
    assert not list(v2.rglob("trajectory.json"))


def test_manifest_if_present_is_replay_only() -> None:
    path = replay.DEFAULT_OUTPUT / "manifest.json"
    if not path.exists():
        pytest.skip("replay videos have not been rendered yet")
    document = json.loads(path.read_text())
    payload = dict(document)
    observed = payload.pop("manifest_sha256")
    from pact_place_corridor_contract import sha256_payload

    if document.get("n") != 24:
        pytest.skip("full 24-clip replay has not been rendered yet")
    assert observed == sha256_payload(payload)
    assert document["replay_only"] is True
    assert document["physics_stepped"] is False
    assert document["expert_rerun"] is False
    assert document["phase0_reopened"] is False
    assert document["n"] == 24
    assert document["v2_hazard_rows_6_12"] == "out_of_scope"
    names = [clip["clip"] for clip in document["clips"]]
    assert "row09_FAIL_outbound_approach_tracking_12.6cm.mp4" in names
    assert "row00_clean_success.mp4" in names
    for clip in document["clips"]:
        step0 = clip["faithfulness"]["residuals"][0]
        assert step0["object_position_residual_m"] == 0.0
        assert step0["tcp_position_residual_m"] == 0.0
