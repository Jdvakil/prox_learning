"""RGB discovery + ACT image keys for convert_pact_place_to_act."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from scripts.convert_pact_place_to_act import (  # noqa: E402
    _row_is_clean,
    convert,
    find_rgb_mp4,
    find_rgb_mp4s,
    ordered_camera_names,
)


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def _json_blob(obj: dict, n: int = 2000) -> np.ndarray:
    raw = json.dumps(obj).encode("utf-8")
    buf = np.zeros(n, dtype=np.uint8)
    buf[: len(raw)] = np.frombuffer(raw, dtype=np.uint8)
    return buf


def _write_mini_traj(path: Path, t: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    act = np.stack([_json_blob({"arm": [0.0] * 7, "gripper": [0.0]}) for _ in range(t)])
    q = np.stack([_json_blob({"arm": [0.0] * 7, "gripper": [0.0, 0.0]}) for _ in range(t)])
    with h5py.File(path, "w") as handle:
        grp = handle.create_group("traj_0")
        grp.create_dataset("fail", data=np.zeros(t, dtype=bool))
        grp.create_dataset("actions/joint_pos", data=act)
        grp.create_dataset("obs/agent/qpos", data=q)
        grp.create_dataset("obs/agent/qvel", data=q)


def test_find_padded_exo_and_wrist_skips_depth(tmp_path: Path):
    wrist = _touch(tmp_path / "episode_00000000_wrist_camera.mp4")
    exo = _touch(tmp_path / "episode_00000000_exo_camera_1.mp4")
    _touch(tmp_path / "episode_00000000_wrist_camera_depth.mp4")
    _touch(tmp_path / "episode_00000000_exo_camera_1_depth.mp4")
    _touch(tmp_path / "episode_00000000_sensors_depth8_heatmap.mp4")
    found = find_rgb_mp4s(tmp_path)
    assert found["wrist_camera"] == wrist
    assert found["exo_camera_1"] == exo
    assert "table_camera" not in found
    assert find_rgb_mp4(tmp_path, "wrist_camera") == wrist
    assert ordered_camera_names(found) == ["exo_camera_1", "wrist_camera"]


def test_find_hashed_table_camera(tmp_path: Path):
    sha = "00ad0da55639dbb3c35b29f99c4ed45aeec7c9b70eb803892e8a9e11b95e40d1"
    row = tmp_path / sha
    wrist = _touch(row / f"episode_{sha}_wrist_camera.mp4")
    table = _touch(row / f"episode_{sha}_table_camera.mp4")
    found = find_rgb_mp4s(row)
    assert found["wrist_camera"] == wrist
    assert found["table_camera"] == table
    assert ordered_camera_names(found) == ["table_camera", "wrist_camera"]


def test_row_is_clean_reads_v108_and_accepted():
    assert _row_is_clean({"v108_clean_success": True, "task_success": True}) is True
    assert _row_is_clean({"v108_clean_success": False}) is False
    assert _row_is_clean({"accepted": True}) is True
    assert _row_is_clean({"accepted": False}) is False
    assert _row_is_clean({"clean_success": False, "accepted": True}) is False
    assert _row_is_clean({}) is True


def test_convert_writes_exo_and_wrist_image_keys(tmp_path: Path, monkeypatch):
    src = tmp_path / "src"
    row = src / "rows" / "000_abc"
    _write_mini_traj(row / "trajectory.h5", t=3)
    _touch(row / "episode_00000000_wrist_camera.mp4")
    _touch(row / "episode_00000000_exo_camera_1.mp4")
    (row / "result.json").write_text(
        json.dumps({"v108_clean_success": True, "task_success": True, "accepted": True})
    )

    def fake_frames(path: Path, image_h, image_w):
        del path
        return np.zeros((3, image_h, image_w, 3), dtype=np.uint8)

    monkeypatch.setattr(
        "scripts.convert_pact_place_to_act._video_frames", fake_frames
    )
    dst = tmp_path / "dst"
    convert(
        src=src,
        dst_dir=dst,
        image_h=8,
        image_w=8,
        with_proximity=False,
        prox_pool="min",
        max_episodes=None,
        require_clean=True,
        task_name="pact_pick_n_place_v2",
    )
    out = dst / "episode_0.hdf5"
    assert out.is_file()
    with h5py.File(out, "r") as handle:
        cams = set(handle["observations/images"].keys())
        assert cams == {"exo_camera_1", "wrist_camera"}
        assert handle["observations/images/exo_camera_1"].shape == (3, 8, 8, 3)
        assert handle["observations/images/wrist_camera"].shape == (3, 8, 8, 3)
        assert "proximity" not in handle["observations"]
    meta = json.loads((dst / "convert_meta.json").read_text())
    assert meta["camera_names"] == ["exo_camera_1", "wrist_camera"]
    assert meta["num_episodes"] == 1


def test_convert_wrist_only_when_no_exo(tmp_path: Path, monkeypatch):
    src = tmp_path / "src"
    row = src / "rows" / "000_v5"
    _write_mini_traj(row / "trajectory.h5", t=2)
    _touch(row / "episode_00000000_wrist_camera.mp4")
    (row / "result.json").write_text(json.dumps({"clean_success": True}))

    monkeypatch.setattr(
        "scripts.convert_pact_place_to_act._video_frames",
        lambda path, h, w: np.zeros((2, h, w, 3), dtype=np.uint8),
    )
    dst = tmp_path / "dst"
    convert(
        src=src,
        dst_dir=dst,
        image_h=4,
        image_w=4,
        with_proximity=False,
        prox_pool="min",
        max_episodes=None,
        require_clean=True,
    )
    with h5py.File(dst / "episode_0.hdf5", "r") as handle:
        assert list(handle["observations/images"].keys()) == ["wrist_camera"]
