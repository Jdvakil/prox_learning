from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "diagnostics_output/pact_slideshow_bundle_manifest.json"
BUNDLE = Path("/root/pact_slideshow_bundle")
SPEC = importlib.util.spec_from_file_location(
    "pact_slideshow_bundle_builder",
    ROOT / "scripts/build_pact_slideshow_bundle.py",
)
assert SPEC is not None and SPEC.loader is not None
slideshow = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(slideshow)


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def test_committed_manifest_is_self_hashed_and_records_presentation_rerenders() -> None:
    document = load_manifest()
    payload = dict(document)
    observed = payload.pop("bundle_manifest_sha256")
    assert observed == slideshow.canonical_hash(payload)
    assert document["schema_version"] == "pact_slideshow_bundle_manifest_v1"
    assert document["bundle_root"] == str(BUNDLE)
    assert document["scientific_artifacts_modified"] is False
    assert document["gpu_work_performed"] is True
    assert document["rollouts_or_training_performed"] is True
    assert document["presentation_only_rollout_rerenders"] == 9
    assert document["training_performed"] is False


def test_committed_manifest_has_required_figures_and_honest_media_counts() -> None:
    document = load_manifest()
    entries = document["entries"]
    paths = [entry["path"] for entry in entries]
    assert len(paths) == len(set(paths)) == 70
    assert document["figure_concepts"] == 10
    assert document["figure_files"] == 20
    assert document["video_files"] == 6
    assert document["paired_video_files"] == 2
    assert document["matched_single_arm_clip_files"] == 3
    assert document["shipped_qualitative_single_arm_clip_files"] == 4
    assert document["supplemental_act_success_clip_files"] == 1
    assert document["complete_matched_instance_pairs"] == 1
    assert document["determinism_dropped_clip_files"] == 1
    assert document["unpaired_independent_probe_files"] == 0
    assert "explicitly unpaired" in document["paired_video_release"]
    assert document["optional_figure_10"] == (
        "omitted_no_frozen_source_field_and_no_new_analysis_allowed"
    )

    figure_paths = [path for path in paths if path.startswith("figures/")]
    stems = {str(Path(path).with_suffix("")) for path in figure_paths}
    assert len(stems) == 10
    for stem in stems:
        assert f"{stem}.png" in figure_paths
        assert f"{stem}.svg" in figure_paths
    assert not any("fig10" in path for path in paths)

    required = {
        "INDEX.md",
        "KEY_NUMBERS.md",
        "ONE_PAGE_SUMMARY.md",
        "VIDEO_SHOT_LIST.md",
        "data/analysis.json",
        "data/tail_characterization.json",
        "data/qualitative_video_manifest.json",
        "data/qualitative_clips_v2_manifest.json",
        "data/qualitative_clips_v3_manifest.json",
        "data/qualitative_clip3_fallback_manifest.json",
        "data/qualitative_clip3_fallback_rank2_manifest.json",
        "data/qualitative_clip3_fallback_rank2_check.json",
        "videos/matched_pairs/README.md",
        "videos/matched_pairs/clip1_54a6272f66ca_pact_success.mp4",
        "videos/matched_pairs/clip2_54a6272f66ca_act_failure.mp4",
        "videos/matched_pairs/clip3_178a8383cda2_act_success_s3101.mp4",
        "videos/matched_pairs/clip4_e99dc657bfa7_pact_failure.mp4",
        "videos/sensor_heatmap/sensor_heatmap_40_skin_streams.mp4",
        "videos/expert_demo/expert_clean_demo_wrist_view.mp4",
    }
    assert required <= set(paths)


@pytest.mark.skipif(not BUNDLE.is_dir(), reason="external slideshow bundle absent")
def test_external_bundle_matches_every_committed_manifest_entry() -> None:
    document = load_manifest()
    assert (BUNDLE / "BUNDLE_MANIFEST.json").read_bytes() == MANIFEST.read_bytes()
    for entry in document["entries"]:
        path = BUNDLE / entry["path"]
        assert path.is_file(), entry["path"]
        assert path.stat().st_size == entry["size_bytes"], entry["path"]
        assert slideshow.file_hash(path) == entry["sha256"], entry["path"]
    assert sum(entry["size_bytes"] for entry in document["entries"]) == document[
        "total_payload_size_bytes_excluding_manifest"
    ]


@pytest.mark.skipif(not BUNDLE.is_dir(), reason="external slideshow bundle absent")
def test_external_bundle_preserves_sources_and_frozen_states() -> None:
    for name, source in slideshow.SOURCE_COPIES.items():
        assert (BUNDLE / "data" / name).read_bytes() == source.read_bytes(), name
    for name, source in slideshow.REPORT_COPIES.items():
        assert (BUNDLE / "reports" / name).read_bytes() == source.read_bytes(), name
    for name, source in slideshow.TRAINING_LOGS.items():
        assert (BUNDLE / "data" / "training_logs" / name).read_bytes() == source.read_bytes(), name

    assert json.loads((BUNDLE / "data/final_decision.json").read_text())[
        "decision"
    ] == "CONTACT_REDUCTION_WITH_TASK_BENEFIT"
    assert json.loads(
        (BUNDLE / "data/qualitative_video_manifest.json").read_text()
    )["status"] == "aborted_determinism_mismatch"
    assert json.loads(
        (BUNDLE / "data/qualitative_clips_v2_manifest.json").read_text()
    )["status"] == "presentation_release_incomplete_determinism_drop"
    assert json.loads(
        (BUNDLE / "data/qualitative_clips_v3_manifest.json").read_text()
    )["status"] == "presentation_release_incomplete_gate_drop"
    assert json.loads(
        (BUNDLE / "data/qualitative_clip3_fallback_manifest.json").read_text()
    )["status"] == "fallback_rank1_dropped_gate_failure"
    assert json.loads(
        (
            BUNDLE
            / "data/qualitative_clip3_fallback_rank2_manifest.json"
        ).read_text()
    )["status"] == "presentation_release_verified_unmatched_fallback"


@pytest.mark.skipif(not BUNDLE.is_dir(), reason="external slideshow bundle absent")
def test_external_figures_are_exact_16_by_9_and_caveats_are_present() -> None:
    pngs = sorted((BUNDLE / "figures").glob("*.png"))
    assert len(pngs) == 10
    for path in pngs:
        with Image.open(path) as image:
            assert image.size == (3200, 1800), path.name

    index = (BUNDLE / "INDEX.md").read_text()
    key_numbers = (BUNDLE / "KEY_NUMBERS.md").read_text()
    one_page = (BUNDLE / "ONE_PAGE_SUMMARY.md").read_text()
    matched_readme = (BUNDLE / "videos/matched_pairs/README.md").read_text()
    shot_list = (BUNDLE / "VIDEO_SHOT_LIST.md").read_text()
    assert "task success directionally positive, not confirmed" in index
    assert "not modality evidence" in key_numbers
    assert "directionally positive but **not confirmed**" in one_page
    assert "Instance A — episode" in matched_readme
    assert "Unpaired ACT scraping success" in matched_readme
    assert "clip3_178a8383cda2_act_success_s3101.mp4" in shot_list
    assert "clip4_e99dc657bfa7_pact_failure.mp4" in shot_list
    assert "front-end-screen wrist-camera ACT success was retired" in shot_list
    assert "absolute delta ≤2 steps" in shot_list
    assert "must not be described as a pair" in shot_list
    assert "maximum hazard penetration, when available" in shot_list
    assert "Separate hybrid-skin safety/CVAE work" in shot_list
