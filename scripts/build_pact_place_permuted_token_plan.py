#!/usr/bin/env python3
"""Freeze distribution-matched proximity frames for the place ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import h5py
import numpy as np

SEED = 2026073105
ROWS = 40
MAX_CONTROL_STEPS = 900
SENSORS = 40
FEATURE_DIM = 32


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_split(path: Path) -> dict[str, Any]:
    split = json.loads(path.read_text())
    payload = dict(split)
    observed = payload.pop("split_manifest_sha256", None)
    if observed != canonical_hash(payload):
        raise ValueError("split manifest self-hash mismatch")
    return split


def select_sources(
    episode_indices: np.ndarray,
    timesteps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if len(episode_indices) < MAX_CONTROL_STEPS:
        raise ValueError("training split has too few unique per-row token frames")
    rng = np.random.default_rng(SEED)
    selected_rows = np.stack(
        [
            rng.choice(
                len(episode_indices),
                size=MAX_CONTROL_STEPS,
                replace=False,
            )
            for _ in range(ROWS)
        ]
    )
    # Preserve sampling without replacement within each rollout while
    # preventing adjacent source frames from coming from the same episode.
    # Cross-row reuse is necessary because 40 * the frozen 900-step task
    # horizon exceeds the 31,176-frame training population.
    for row in range(ROWS):
        selected = selected_rows[row]
        for position in range(1, MAX_CONTROL_STEPS):
            previous_episode = episode_indices[selected[position - 1]]
            if episode_indices[selected[position]] != previous_episode:
                continue
            replacement = next(
                (
                    candidate
                    for candidate in range(position + 1, MAX_CONTROL_STEPS)
                    if episode_indices[selected[candidate]] != previous_episode
                ),
                None,
            )
            if replacement is None:
                # The corridor fork only searched forward, which cannot repair
                # an equal-episode pair at the final position.  Swap with an
                # earlier frame whose neighborhood also remains separated.
                replacement = next(
                    (
                        candidate
                        for candidate in range(1, position - 1)
                        if episode_indices[selected[candidate]] != previous_episode
                        and episode_indices[selected[candidate - 1]]
                        != episode_indices[selected[position]]
                        and episode_indices[selected[candidate + 1]]
                        != episode_indices[selected[position]]
                    ),
                    None,
                )
            if replacement is None:
                raise ValueError("could not separate consecutive episodes")
            selected[position], selected[replacement] = (
                selected[replacement],
                selected[position],
            )
    selected_episodes = episode_indices[selected_rows]
    selected_timesteps = timesteps[selected_rows]
    if any(np.any(row[1:] == row[:-1]) for row in selected_episodes):
        raise AssertionError("consecutive source episode invariant failed")
    return selected_episodes, selected_timesteps


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-train-count", required=True, type=int)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit("token-plan output directory is not empty")
    output.mkdir(parents=True, exist_ok=True)
    split = load_split(args.split_manifest)
    dataset_manifest = json.loads(args.dataset_manifest.read_text())
    train_indices = [
        int(row["act_episode_index"]) for row in split["episodes"] if row["split"] == "train"
    ]
    if len(train_indices) != args.expected_train_count:
        raise ValueError(
            "place ablation train count differs from the split contract: "
            f"{len(train_indices)} != {args.expected_train_count}"
        )

    flat_episodes: list[np.ndarray] = []
    flat_timesteps: list[np.ndarray] = []
    lengths: dict[int, int] = {}
    for episode_index in train_indices:
        path = args.dataset_dir / f"episode_{episode_index}.hdf5"
        with h5py.File(path, "r") as episode:
            dataset = episode["observations/proximity_embeddings"]
            if dataset.shape[1:] != (SENSORS, FEATURE_DIM):
                raise ValueError(f"unexpected embedding shape in {path}")
            length = int(dataset.shape[0])
        lengths[episode_index] = length
        flat_episodes.append(np.full(length, episode_index, dtype=np.int16))
        flat_timesteps.append(np.arange(length, dtype=np.int16))
    episode_indices = np.concatenate(flat_episodes)
    timesteps = np.concatenate(flat_timesteps)
    selected_episodes, selected_timesteps = select_sources(episode_indices, timesteps)

    episode_path = output / "source_episode_indices.npy"
    timestep_path = output / "source_timesteps.npy"
    token_path = output / "permuted_tokens.npy"
    np.save(episode_path, selected_episodes, allow_pickle=False)
    np.save(timestep_path, selected_timesteps, allow_pickle=False)
    tokens = np.lib.format.open_memmap(
        token_path,
        mode="w+",
        dtype=np.float32,
        shape=(ROWS, MAX_CONTROL_STEPS, SENSORS, FEATURE_DIM),
    )
    with ExitStack() as stack:
        episodes = {
            index: stack.enter_context(h5py.File(args.dataset_dir / f"episode_{index}.hdf5", "r"))
            for index in train_indices
        }
        for row in range(ROWS):
            for step in range(MAX_CONTROL_STEPS):
                episode_index = int(selected_episodes[row, step])
                timestep = int(selected_timesteps[row, step])
                tokens[row, step] = episodes[episode_index]["observations/proximity_embeddings"][
                    timestep
                ]
    tokens.flush()
    del tokens

    observed = np.load(token_path, mmap_mode="r")
    if (
        observed.shape
        != (
            ROWS,
            MAX_CONTROL_STEPS,
            SENSORS,
            FEATURE_DIM,
        )
        or observed.dtype != np.float32
    ):
        raise ValueError("published token tensor shape or dtype changed")
    if not np.isfinite(observed).all():
        raise ValueError("published token tensor contains non-finite values")
    norms = np.linalg.norm(observed.reshape(-1, FEATURE_DIM), axis=1)
    manifest: dict[str, Any] = {
        "schema_version": "pact_permuted_token_plan_v2",
        "ablation": "PACT_PERMUTED",
        "seed": SEED,
        "rows": ROWS,
        "max_control_steps": MAX_CONTROL_STEPS,
        "token_shape": [SENSORS, FEATURE_DIM],
        "source_partition": "train",
        "source_train_episode_count": len(train_indices),
        "source_train_frame_count": int(len(episode_indices)),
        "selected_frame_count": ROWS * MAX_CONTROL_STEPS,
        "selection_without_replacement_within_row": True,
        "cross_row_source_reuse_allowed": True,
        "global_selection_without_replacement": False,
        "consecutive_source_episode_differs": True,
        "live_scene_alignment_destroyed": True,
        "split_manifest_path": str(args.split_manifest.resolve()),
        "split_manifest_sha256": split["split_manifest_sha256"],
        "dataset_manifest_path": str(args.dataset_manifest.resolve()),
        "dataset_manifest_file_sha256": file_hash(args.dataset_manifest),
        "dataset_tree_sha256": dataset_manifest["converted_tree_file_sha256"],
        "files": {
            "tokens": {
                "path": str(token_path),
                "sha256": file_hash(token_path),
                "size_bytes": token_path.stat().st_size,
            },
            "source_episode_indices": {
                "path": str(episode_path),
                "sha256": file_hash(episode_path),
                "size_bytes": episode_path.stat().st_size,
            },
            "source_timesteps": {
                "path": str(timestep_path),
                "sha256": file_hash(timestep_path),
                "size_bytes": timestep_path.stat().st_size,
            },
        },
        "selected_sensor_token_norm": {
            "count": int(norms.size),
            "mean": float(np.mean(norms)),
            "minimum": float(np.min(norms)),
            "p01": float(np.percentile(norms, 1)),
            "maximum": float(np.max(norms)),
            "count_below_0_1": int(np.sum(norms < 0.1)),
        },
    }
    manifest["token_plan_sha256"] = canonical_hash(manifest)
    write_json_atomic(output / "token_plan.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
