"""V10.10 scene helpers for pact_pick_n_place_v2 eval. No molmospaces."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "submodules" / "act"))

from eval_place_v1010_scene import (  # noqa: E402
    CUSTOM_SCENES,
    INCLUDE_CHAIN,
    N_V1010_CELLS,
    V1010_SCENE_BY_POSE,
    V12_INCLUDE_STEM,
    V12_XML,
    assert_v1010_include_chain,
    assert_v1010_scene_hashes,
    assert_v12_wraps_center,
    resolve_v1010_scenes_dir,
    rewrite_v12_include,
    spread_episode_count,
    v1010_cell,
    v1010_scene_paths,
    v12_include_target,
)


def test_v1010_cell_house_1_is_f0_left_center():
    assert v1010_cell(0) == ("F0_target_side_stagger", "left", "neg5")
    assert v1010_cell(1) == ("F0_target_side_stagger", "left", "center")
    assert v1010_cell(2) == ("F0_target_side_stagger", "left", "pos5")
    assert v1010_cell(3) == ("F0_target_side_stagger", "right", "neg5")


def test_v1010_scene_paths_align_to_pose(tmp_path: Path):
    paths = v1010_scene_paths(tmp_path)
    assert len(paths) == N_V1010_CELLS
    assert paths[0].name == "pact_place_corridor_v10_7_neg5.xml"
    assert paths[1].name == "pact_place_corridor_v10_7_center.xml"
    assert paths[2].name == "pact_place_corridor_v10_7_pos5.xml"
    assert paths[1].parent == tmp_path


def test_spread_episode_count():
    assert spread_episode_count(2) == (1, 24)
    assert spread_episode_count(48) == (2, 48)
    assert spread_episode_count(50) == (2, 48)
    assert spread_episode_count(72) == (3, 72)


def test_v12_xml_on_disk_wraps_local_center():
    assert V12_XML.is_file()
    assert_v12_wraps_center(V12_XML)
    assert v12_include_target(V12_XML) == V12_INCLUDE_STEM


def test_custom_scenes_has_hashed_v10_7_and_include_chain():
    assert_v1010_scene_hashes(CUSTOM_SCENES)
    assert_v1010_include_chain(CUSTOM_SCENES)
    for name in INCLUDE_CHAIN:
        assert (CUSTOM_SCENES / name).is_file()
    for meta in V1010_SCENE_BY_POSE.values():
        xml = CUSTOM_SCENES / meta["filename"]
        assert xml.is_file()
        sidecar = xml.with_name(xml.stem + "_metadata.json")
        assert sidecar.is_file(), sidecar
    assert (CUSTOM_SCENES / "pact_place_corridor_v10_metadata.json").is_file()
    assert resolve_v1010_scenes_dir() == CUSTOM_SCENES


def test_rewrite_v12_include(tmp_path: Path):
    src = tmp_path / "pact_place_corridor_v12.xml"
    src.write_text(
        '<mujoco model="pact_place_corridor_v12">\n'
        f'  <include file="../submodules/molmospaces/molmo_spaces/'
        f'data_generation/custom_scenes/{V12_INCLUDE_STEM}"/>\n'
        "</mujoco>\n"
    )
    include = tmp_path / "scenes" / V12_INCLUDE_STEM
    include.parent.mkdir(parents=True)
    include.write_text("<mujoco/>\n")
    dst = tmp_path / "out" / "rewritten.xml"
    rewrite_v12_include(src, include, dst)
    text = dst.read_text()
    assert str(include.resolve()) in text
    assert "../submodules/molmospaces" not in text
