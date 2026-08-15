import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "diagnostics_output/pact_blur_sweep/media_manifest.json"


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_media_selection_and_gate_are_frozen() -> None:
    document = json.loads(MANIFEST.read_text())
    payload = dict(document)
    observed = payload.pop("media_manifest_sha256")
    assert observed == canonical_hash(payload)
    assert document["status"] in {
        "selection_and_gate_frozen_pre_render",
        "presentation_release_verified",
        "partial_gate_drop",
    }
    assert document["qualifying_instance_seed_count"] == 62
    assert document["selection"]["episode_id"].startswith("65f2ab175bed")
    assert document["selection"]["checkpoint_seed"] == 3101
    assert [row["schedule_index"] for row in document["selection"]["rows"]] == [
        2,
        16,
        22,
        28,
    ]
    assert document["determinism_gate"] == {
        "declared_before_render": True,
        "task_success": {"comparison": "exact"},
        "manipulation_success": {
            "comparison": "exact",
            "represented_by": "task_success",
        },
        "first_hazard_bar_contact_step": {"comparison": "exact"},
        "first_grasp_target_contact_step": {
            "comparison": "absolute_step_delta_lte",
            "tolerance_steps": 2,
        },
        "contact_pair_sample_counts": {
            "comparison": "informational_only",
            "record_delta": True,
        },
        "on_breach": "drop_clip_without_retry",
    }


def test_calibration_figures_match_manifest() -> None:
    document = json.loads(MANIFEST.read_text())
    for figure in document["figures"].values():
        for record in figure.values():
            path = Path(record["path"])
            assert path.exists()
            assert file_hash(path) == record["sha256"]
    calibration = json.loads(
        (ROOT / "diagnostics_output/pact_blur_sweep/calibration.json").read_text()
    )
    retained = {
        float(row["sigma"]): round(
            100.0 * float(row["retained_fraction_of_sharp"]), 1
        )
        for row in calibration["measurements"]
        if float(row["sigma"]) in {0.0, 0.5, 1.0, 2.0}
    }
    assert retained == {0.0: 100.0, 0.5: 46.2, 1.0: 9.3, 2.0: 1.7}
    assert calibration["measurements"][0]["output_rgb_sha256"] == (
        "45082f1aaec76023434378b4d6b784d5ab1c253c14f3cb1dc64705d23470f9e3"
    )


def test_media_runtime_preserves_two_pane_contract() -> None:
    evaluator = (ROOT / "submodules/act/eval_pact_blur_media_row.py").read_text()
    runner = (ROOT / "scripts/run_pact_blur_media.py").read_text()
    assert "class BlurMediaInferencePolicy" in evaluator
    assert "blurred[0, 0]" in evaluator
    assert "third_person_camera_registered_in_observation" in evaluator
    assert '"resolution_width_height": list(COMPOSITE_RESOLUTION)' in evaluator
    assert "drop_clip_without_retry" in runner
    assert "setpts=PTS/3.0" in runner
    assert "WORKERS = 4" in runner


def test_protected_blur_artifacts_match_frozen_manifest() -> None:
    document = json.loads(MANIFEST.read_text())
    protected = document["sources"]["protected_scientific_artifacts"]
    for path, expected in protected.items():
        assert file_hash(Path(path)) == expected
