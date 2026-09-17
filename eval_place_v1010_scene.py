"""V10.10 / v12 eval scene helpers. No molmospaces import."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
CUSTOM_SCENES = _REPO / "custom_scenes"
MOLMOSPACES_V1010_SHA = "4bba4cbcea49ca8dbaee44fb9a376568b1b3cc82"

V95_LAYOUT_FAMILY_IDS = (
    "F0_target_side_stagger",
    "F1_inner_panel_stagger",
    "F2_outer_panel_stagger",
    "F3_aperture_side_stagger",
)
INTRUSION_SIDES = ("left", "right")
POSE_IDS = ("neg5", "center", "pos5")
N_V1010_CELLS = 24

V1010_SCENE_BY_POSE = {
    "neg5": {
        "filename": "pact_place_corridor_v10_7_neg5.xml",
        "sha256": "df50679c749c6ad771d00023e73a08e0bfaf59d5391df9b42cf05de4ed7893a7",
    },
    "center": {
        "filename": "pact_place_corridor_v10_7_center.xml",
        "sha256": "b5a41d0d8934240b078f1cdbf3a6991b2e94a46558ddf1c9eae0119c8b8e138a",
    },
    "pos5": {
        "filename": "pact_place_corridor_v10_7_pos5.xml",
        "sha256": "762a5a4662a8fc0d31a3a0ee1135b347d6dd2c882daf4e65c2f706ab2d6fe565",
    },
}

INCLUDE_CHAIN = (
    "pact_place_corridor_v10_7_neg5.xml",
    "pact_place_corridor_v10_7_center.xml",
    "pact_place_corridor_v10_7_pos5.xml",
    "pact_place_corridor_v5.xml",
    "pact_place_corridor_v3.xml",
)
V12_INCLUDE_STEM = "pact_place_corridor_v10_7_center.xml"
V12_XML = CUSTOM_SCENES / "pact_place_corridor_v12.xml"


def v1010_cell(index: int) -> tuple[str, str, str]:
    cells = [
        (family, side, pose)
        for family in V95_LAYOUT_FAMILY_IDS
        for side in INTRUSION_SIDES
        for pose in POSE_IDS
    ]
    return cells[int(index) % len(cells)]


def spread_episode_count(n: int) -> tuple[int, int]:
    """(per_cell, total). n=50 still runs 48. n=2 with spread is 24 (1/cell)."""
    if n <= 24:
        return 1, 24
    per = max(2, n // 24)
    return per, per * 24


def v1010_scene_paths(scenes_dir: Path) -> list[Path]:
    return [
        Path(scenes_dir) / V1010_SCENE_BY_POSE[v1010_cell(i)[2]]["filename"]
        for i in range(N_V1010_CELLS)
    ]


def resolve_v1010_scenes_dir(molmo_root: Path | None = None) -> Path:
    if CUSTOM_SCENES.is_dir() and (CUSTOM_SCENES / "pact_place_corridor_v10_7_center.xml").is_file():
        return CUSTOM_SCENES
    if molmo_root is not None:
        alt = Path(molmo_root) / "molmo_spaces" / "data_generation" / "custom_scenes"
        if (alt / "pact_place_corridor_v10_7_center.xml").is_file():
            return alt
    return CUSTOM_SCENES


def assert_v1010_scene_hashes(scenes_dir: Path) -> None:
    for meta in V1010_SCENE_BY_POSE.values():
        path = Path(scenes_dir) / meta["filename"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != meta["sha256"]:
            raise AssertionError(f"{path.name} sha256 {digest} != {meta['sha256']}")


def assert_v1010_include_chain(scenes_dir: Path) -> None:
    root = Path(scenes_dir)
    for name in INCLUDE_CHAIN:
        if not (root / name).is_file():
            raise AssertionError(f"missing include-chain file {name}")
    center = (root / "pact_place_corridor_v10_7_center.xml").read_text()
    if "pact_place_corridor_v5.xml" not in center:
        raise AssertionError("v10_7_center does not include v5")
    v5 = (root / "pact_place_corridor_v5.xml").read_text()
    if "pact_place_corridor_v3.xml" not in v5:
        raise AssertionError("v5 does not include v3")


def v12_include_target(xml: Path) -> str:
    for line in Path(xml).read_text().splitlines():
        if "<include" in line and "file=" in line:
            file = line.split("file=", 1)[1].split('"', 2)[1]
            return Path(file).name
    raise ValueError(f"no include in {xml}")


def assert_v12_wraps_center(xml: Path) -> None:
    if v12_include_target(xml) != V12_INCLUDE_STEM:
        raise AssertionError(f"{xml} does not wrap {V12_INCLUDE_STEM}")


def rewrite_v12_include(src: Path, include: Path, dst: Path) -> None:
    import re

    text = Path(src).read_text()
    new = re.sub(r'file="[^"]+"', f'file="{Path(include).resolve()}"', text, count=1)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(new)
