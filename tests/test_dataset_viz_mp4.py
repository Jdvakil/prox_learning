"""Sidecar mp4 lookup for dataset_viz (padded-int, batch, and hashed v10 names)."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))

from dataset_viz import RGB_STEMS, glob_mp4, write_audit_index  # noqa: E402


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def test_rgb_stems_include_table_and_wrist():
    assert "wrist_camera" in RGB_STEMS
    assert "table_camera" in RGB_STEMS
    assert "exo_camera_1" in RGB_STEMS


def test_glob_padded_hf_v5(tmp_path: Path):
    p = _touch(tmp_path / "episode_00000000_wrist_camera.mp4")
    _touch(tmp_path / "episode_00000000_wrist_camera_depth.mp4")
    assert glob_mp4(tmp_path, 0, "wrist_camera") == p
    assert glob_mp4(tmp_path, 0, "table_camera") is None


def test_glob_datagen_batch_picks_matching_id(tmp_path: Path):
    _touch(tmp_path / "episode_00000000_wrist_camera_batch_1_of_1.mp4")
    want = _touch(tmp_path / "episode_00000003_wrist_camera_batch_1_of_1.mp4")
    assert glob_mp4(tmp_path, 3, "wrist_camera") == want
    assert glob_mp4(tmp_path, 1, "wrist_camera") is None


def test_glob_hashed_v1010_folder_name(tmp_path: Path):
    sha = "00ad0da55639dbb3c35b29f99c4ed45aeec7c9b70eb803892e8a9e11b95e40d1"
    row = tmp_path / sha
    wrist = _touch(row / f"episode_{sha}_wrist_camera.mp4")
    table = _touch(row / f"episode_{sha}_table_camera.mp4")
    _touch(row / f"episode_{sha}_wrist_depth.mp4")
    _touch(row / f"episode_{sha}_review_2x2.mp4")
    assert glob_mp4(row, 0, "wrist_camera") == wrist
    assert glob_mp4(row, 0, "table_camera") == table
    assert glob_mp4(row, 0, "exo_camera_1") is None


def test_glob_hashed_loose_when_folder_renamed(tmp_path: Path):
    row = tmp_path / "renamed"
    wrist = _touch(row / "episode_abc123_wrist_camera.mp4")
    assert glob_mp4(row, 0, "wrist_camera") == wrist


def test_glob_loose_refuses_two_matches(tmp_path: Path):
    _touch(tmp_path / "episode_aaa_wrist_camera.mp4")
    _touch(tmp_path / "episode_bbb_wrist_camera.mp4")
    assert glob_mp4(tmp_path, 0, "wrist_camera") is None


def test_write_audit_index_dashboard(tmp_path):
    import json

    ds = tmp_path / "molmo" / "pick"
    ds.mkdir(parents=True)
    (ds / "audit.json").write_text(json.dumps({
        "kind": "datagen",
        "n_eps_exported": 2,
        "n_eps_total": 2,
        "n_videos": 2,
        "gaps": [],
        "has_prox": True,
        "has_wrist": True,
        "has_table": True,
        "groups": ["free"],
        "video": "episodes/free/0000_house_1_traj_0.mp4",
    }))
    write_audit_index(tmp_path, quiet=True)
    html = (tmp_path / "index.html").read_text()
    cat = json.loads((tmp_path / "audit.json").read_text())
    assert "%%BOOTSTRAP%%" not in html
    assert "dataset viz" in html
    assert "molmo/pick" in html
    assert cat["n"] == 1
    assert cat["n_videos"] == 2
    assert cat["rows"][0]["slug"] == "molmo/pick"
