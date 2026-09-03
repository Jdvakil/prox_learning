"""Sidecar mp4 lookup for dataset_viz (padded-int, batch, and hashed v10 names)."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))

from dataset_viz import (  # noqa: E402
    RGB_STEMS,
    glob_mp4,
    wanted_n,
    viz_action,
    write_audit_index,
    _usable_done,
    _cat_series,
)


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
    (ds / "timeline.json").write_text(json.dumps({
        "title": "pick", "t": [0.0], "qpos": {}, "qvel": {}, "skin_min": [],
        "episodes": [], "n_episodes": 2, "duration_s": 1.0,
    }))
    write_audit_index(tmp_path, quiet=True)
    html = (tmp_path / "index.html").read_text()
    cat = json.loads((tmp_path / "audit.json").read_text())
    js = (ds / "timeline.js").read_text()
    assert "%%BOOTSTRAP%%" not in html
    assert "dataset viz" in html
    assert "molmo/pick" in html
    assert "fetch(" not in html
    assert js.startswith("window.DATASET_TIMELINE")
    assert cat["n"] == 1
    assert cat["n_videos"] == 2
    assert cat["rows"][0]["slug"] == "molmo/pick"


def test_wanted_n_respects_max_and_start():
    from argparse import Namespace
    assert wanted_n(10, Namespace(max_episodes=None, start_episode=0)) == 10
    assert wanted_n(10, Namespace(max_episodes=2, start_episode=0)) == 2
    assert wanted_n(10, Namespace(max_episodes=None, start_episode=8)) == 2
    assert wanted_n(10, Namespace(max_episodes=5, start_episode=8)) == 2


def test_viz_action_skip_new_grow(tmp_path):
    from argparse import Namespace
    dest = tmp_path / "out"
    dest.mkdir()
    (dest / "audit.json").write_text(
        '{"n_eps_exported": 4, "cam3d": true, "stride": 2}'
    )
    clip = dest / "episodes" / "free" / "0000_x.mp4"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"")
    skip_args = Namespace(force=False, max_episodes=None, start_episode=0, one_video=False)
    assert viz_action(dest, 4, skip_args) == "skip"
    assert viz_action(dest, 6, skip_args) == "grow"
    assert viz_action(dest, 4, Namespace(force=True, max_episodes=None,
                                         start_episode=0, one_video=False)) == "run"
    empty = tmp_path / "empty"
    empty.mkdir()
    assert viz_action(empty, 3, skip_args) == "run"


def test_usable_done_drops_missing_clip(tmp_path):
    (tmp_path / "episodes" / "free").mkdir(parents=True)
    keep = tmp_path / "episodes" / "free" / "0000_a.mp4"
    keep.write_bytes(b"")
    tl = {
        "episodes": [
            {"label": "a", "video": "episodes/free/0000_a.mp4"},
            {"label": "b", "video": "episodes/free/0001_b.mp4"},
        ]
    }
    done = _usable_done(tmp_path, tl)
    assert set(done) == {"a"}


def test_cat_series_dicts_and_lists():
    assert _cat_series([1], [2]) == [1, 2]
    assert _cat_series({"q1": [1]}, {"q1": [2], "q2": [3]}) == {"q1": [1, 2], "q2": [3]}
