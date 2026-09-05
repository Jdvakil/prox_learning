"""Offline diagnostics must use checkpoint chunk size and the intended episode split."""
from pathlib import Path
import json
import pickle
import sys

import numpy as np
import pytest
import torch

ACT = Path(__file__).resolve().parents[1] / "submodules" / "act"
sys.path.insert(0, str(ACT))

from eval_train_set import resolve_cameras, select_episode_ids


def test_legacy_partitions_match_training_loader_and_limit_after_split():
    expected = np.random.RandomState(1).permutation(200)
    train = select_episode_ids(200, "train")
    val = select_episode_ids(200, "val")
    np.testing.assert_array_equal(train, expected[:160])
    np.testing.assert_array_equal(val, expected[160:])
    assert not set(train) & set(val)
    np.testing.assert_array_equal(select_episode_ids(200, "val", limit=8), val[:8])


@pytest.mark.parametrize("ids", [[1, 1], [-1], [200], [0.5], [], [[1]]])
def test_bad_explicit_episode_ids_rejected(ids):
    with pytest.raises(ValueError):
        select_episode_ids(200, "train", explicit_ids=ids)


def test_explicit_partition_preserves_order():
    np.testing.assert_array_equal(
        select_episode_ids(200, "val", explicit_ids=[5, 2, 7]), [5, 2, 7],
    )


def test_camera_order_requires_metadata_or_explicit_training_order(tmp_path):
    with pytest.raises(ValueError, match="camera order missing"):
        resolve_cameras(tmp_path)
    (tmp_path / "convert_meta.json").write_text(json.dumps({"camera_names": ["wrist_camera"]}))
    assert resolve_cameras(tmp_path) == ["wrist_camera"]
    assert resolve_cameras(tmp_path, ["exo_camera_1", "wrist_camera"]) == ["exo_camera_1", "wrist_camera"]


@pytest.mark.parametrize("chunk,cameras", [(50, ["wrist_camera"]), (100, ["exo_camera_1", "wrist_camera"])])
def test_policy_builder_reads_selected_checkpoint_chunk(tmp_path, monkeypatch, chunk, cameras):
    import attn_heatmap

    state = {"model.query_embed.weight": torch.zeros(chunk, 512)}
    torch.save(state, tmp_path / "selected.ckpt")
    with (tmp_path / "dataset_stats.pkl").open("wb") as handle:
        pickle.dump({"sentinel": 1}, handle)
    configs = []

    class FakePolicy:
        def __init__(self, config):
            configs.append(config)

        def load_state_dict(self, loaded):
            assert loaded["model.query_embed.weight"].shape == (chunk, 512)

        def to(self, device):
            return self

        def eval(self):
            return self

    monkeypatch.setattr(attn_heatmap, "ACTPolicy", FakePolicy)
    _, stats, _, _, _ = attn_heatmap.build_policy(
        tmp_path, "cpu", camera_names=cameras, ckpt_name="selected.ckpt",
    )
    assert configs[0]["num_queries"] == chunk
    assert configs[0]["camera_names"] == cameras
    assert stats == {"sentinel": 1}


def test_offline_metrics_ignore_padding_and_save_actual_ids(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import attn_heatmap
    import eval_train_set

    class FakePolicy:
        model = SimpleNamespace(query_embed=SimpleNamespace(num_embeddings=2))

        def __call__(self, qpos, image, proximity_positions=None):
            return torch.zeros(qpos.shape[0], 2, 8)

    class FakeDataset:
        def __init__(self, ids, *args, **kwargs):
            assert ids.tolist() == [0]

        def __len__(self):
            return 1

        def __getitem__(self, index):
            return (
                torch.zeros(1, 3, 4, 4), torch.zeros(9),
                torch.tensor([[1.0] * 8, [1000.0] * 8]),
                torch.tensor([False, True]),
            )

    monkeypatch.setattr(attn_heatmap, "build_policy", lambda *a, **kw: (
        FakePolicy(), {"action_std": np.full(8, 2.0)}, None, 0, "vanilla",
    ))
    monkeypatch.setattr(eval_train_set, "EpisodicDataset", FakeDataset)
    output = tmp_path / "metrics.json"
    monkeypatch.setattr(sys, "argv", [
        "eval_train_set.py", "--ckpt_dir", str(tmp_path), "--data_dir", str(tmp_path),
        "--num_episodes", "1", "--split", "all", "--passes", "1", "--device", "cpu",
        "--camera_names", "wrist_camera", "--output", str(output),
    ])
    eval_train_set.main()
    result = json.loads(output.read_text())
    assert result["episode_ids"] == [0]
    assert result["chunk_size"] == 2
    assert result["valid_predicted_steps"] == 1
    assert result["normalized_action_l1"] == pytest.approx(1)
    assert result["arm_joint_mae_rad"] == pytest.approx(2)
    assert result["gripper_normalized_l1"] == pytest.approx(1)
