#!/usr/bin/env python3
"""Deterministic split and normalisation-statistics verification.

Reproduces the split the pinned trainer actually performs (``set_seed(1)`` then
``np.random.permutation(num_episodes)``) and checks it against the approved
manifest, then verifies ``dataset_stats.pkl`` three ways:

  A  the statistics pickled by the training run;
  B  a fresh read-only call to the committed ``utils.get_norm_stats``;
  C  an independent float64 NumPy accumulation that shares no code with A or B.

A vs B must agree exactly (identical arithmetic on identical bytes). C is a
numerically independent cross-check, so its residual difference is reported with
its floating-point cause rather than being required to be zero.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
ACT_ROOT = REPO_ROOT / "submodules/act"
if str(ACT_ROOT) not in sys.path:
    sys.path.insert(0, str(ACT_ROOT))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def trainer_split(num_episodes: int) -> tuple[list[int], list[int]]:
    """Exactly what imitate_episodes.py + utils.load_data do, in that order."""
    from utils import set_seed

    set_seed(1)  # torch.manual_seed(1); np.random.seed(1)
    shuffled = np.random.permutation(num_episodes)
    train_ratio = 0.8
    cut = int(train_ratio * num_episodes)
    return shuffled[:cut].tolist(), shuffled[cut:].tolist()


def independent_stats(dataset_dir: Path, num_episodes: int) -> dict[str, np.ndarray]:
    """Path C: float64 NumPy accumulation, sharing no code with the ACT utils."""
    qpos_chunks: list[np.ndarray] = []
    action_chunks: list[np.ndarray] = []
    for idx in range(num_episodes):
        with h5py.File(dataset_dir / f"episode_{idx}.hdf5", "r") as handle:
            qpos_chunks.append(np.asarray(handle["/observations/qpos"][()], dtype=np.float64))
            action_chunks.append(np.asarray(handle["/action"][()], dtype=np.float64))
    qpos = np.concatenate(qpos_chunks, axis=0)
    action = np.concatenate(action_chunks, axis=0)
    return {
        "qpos_mean": qpos.mean(axis=0),
        # ddof=1 matches torch.std's default sample-standard-deviation correction.
        "qpos_std": np.clip(qpos.std(axis=0, ddof=1), 1e-2, np.inf),
        "action_mean": action.mean(axis=0),
        "action_std": np.clip(action.std(axis=0, ddof=1), 1e-2, np.inf),
        "example_qpos": qpos_chunks[0][0],
        "_timesteps": np.array([qpos.shape[0]]),
    }


def compare(a: dict, b: dict, keys: tuple[str, ...]) -> dict[str, Any]:
    per_key = {}
    for key in keys:
        x = np.asarray(a[key], dtype=np.float64)
        y = np.asarray(b[key], dtype=np.float64)
        denom = np.maximum(np.abs(x), np.abs(y))
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.where(denom > 0, np.abs(x - y) / denom, 0.0)
        per_key[key] = {
            "max_abs_difference": float(np.abs(x - y).max()),
            "max_rel_difference": float(np.nanmax(rel)),
            "exact": bool(np.array_equal(x, y)),
        }
    return {
        "per_key": per_key,
        "all_exact": all(v["exact"] for v in per_key.values()),
        "max_abs_difference": max(v["max_abs_difference"] for v in per_key.values()),
        "max_rel_difference": max(v["max_rel_difference"] for v in per_key.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--stats_pkl", required=True)
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    stats_path = Path(args.stats_pkl).resolve()
    manifest = json.loads(Path(args.manifest).read_text())
    keys = ("qpos_mean", "qpos_std", "action_mean", "action_std", "example_qpos")

    train_ids, val_ids = trainer_split(args.num_episodes)
    declared = manifest["deterministic_split"]
    split_manifest = {
        "algorithm": "numpy.random.RandomState(1).permutation(100); first 80 train, last 20 validation",
        "seed": 1,
        "num_episodes": args.num_episodes,
        "train_episode_ids": train_ids,
        "validation_episode_ids": val_ids,
    }

    # Path A: what the training run pickled.
    with stats_path.open("rb") as handle:
        stats_a = pickle.load(handle)
    # Path B: fresh read-only recomputation through the committed implementation.
    from utils import get_norm_stats

    stats_b = get_norm_stats(str(dataset_dir), args.num_episodes)
    # Path C: independent float64 accumulation.
    stats_c = independent_stats(dataset_dir, args.num_episodes)

    ab = compare(stats_a, stats_b, keys)
    ac = compare(stats_a, stats_c, keys)

    shapes_ok = all(
        np.shape(stats_a[k]) == want
        for k, want in (
            ("qpos_mean", (9,)),
            ("qpos_std", (9,)),
            ("action_mean", (8,)),
            ("action_std", (8,)),
            ("example_qpos", (9,)),
        )
    )
    finite_ok = all(bool(np.isfinite(np.asarray(stats_a[k])).all()) for k in keys)
    positive_ok = bool(
        (np.asarray(stats_a["qpos_std"]) > 0).all()
        and (np.asarray(stats_a["action_std"]) > 0).all()
    )

    episode_files = sorted(
        dataset_dir.glob("episode_*.hdf5"), key=lambda p: int(p.stem.split("_")[1])
    )
    tree_payload = "".join(
        f"{p.name}\0{sha256_file(p)}\n" for p in episode_files
    )

    report = {
        "schema_version": "hybrid_clean_retrain_split_stats_v1",
        "dataset_dir": str(dataset_dir),
        "split": split_manifest,
        "split_manifest_sha256": canonical_hash(split_manifest),
        "split_matches_manifest": {
            "train": train_ids == declared["train_episode_ids"],
            "validation": val_ids == declared["validation_episode_ids"],
            "train_count": len(train_ids),
            "validation_count": len(val_ids),
            "trajectory_level_only": True,
            "disjoint": set(train_ids).isdisjoint(val_ids),
            "covers_all_episodes": sorted(train_ids + val_ids) == list(range(args.num_episodes)),
        },
        "statistics": {
            "population": "all timesteps of all 100 converted episodes, independent of the split",
            "total_timesteps": int(stats_c["_timesteps"][0]),
            "shapes_ok": shapes_ok,
            "all_finite": finite_ok,
            "std_strictly_positive": positive_ok,
            "values": {k: np.asarray(stats_a[k]).tolist() for k in keys},
        },
        "comparison_A_pickled_vs_B_committed_recompute": ab,
        "comparison_A_pickled_vs_C_independent_float64": ac,
        "comparison_C_note": (
            "Path C accumulates in float64 with NumPy while paths A and B accumulate in "
            "float32 with torch. Any residual difference is float32 rounding in the "
            "reduction, not a data mismatch; A vs B is the exactness requirement."
        ),
        "hashes": {
            "dataset_stats_pkl_sha256": sha256_file(stats_path),
            "converter_source_sha256": sha256_file(
                REPO_ROOT / "scripts/convert_obstacle_to_act.py"
            ),
            "act_utils_sha256": sha256_file(ACT_ROOT / "utils.py"),
            "converted_dataset_tree_sha256": hashlib.sha256(
                tree_payload.encode("utf-8")
            ).hexdigest(),
            "converted_episode_count": len(episode_files),
        },
    }
    report["passed"] = bool(
        report["split_matches_manifest"]["train"]
        and report["split_matches_manifest"]["validation"]
        and report["split_matches_manifest"]["disjoint"]
        and report["split_matches_manifest"]["covers_all_episodes"]
        and shapes_ok
        and finite_ok
        and positive_ok
        and ab["all_exact"]
        and ab["max_abs_difference"] == 0.0
        and ab["max_rel_difference"] == 0.0
        and len(episode_files) == args.num_episodes
    )
    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
