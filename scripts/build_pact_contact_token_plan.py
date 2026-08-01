#!/usr/bin/env python3
"""Freeze 100 rows of distribution-matched PACT permutation tokens."""

from __future__ import annotations

import argparse
import heapq
import hashlib
import json
import os
import tempfile
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import h5py
import numpy as np


SEED = 2026080103
ROWS = 100
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
    episode_indices: np.ndarray, timesteps: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if len(episode_indices) < MAX_CONTROL_STEPS:
        raise ValueError("training split has too few token frames")
    rng = np.random.default_rng(SEED)
    selected_rows = np.stack(
        [
            rng.choice(len(episode_indices), size=MAX_CONTROL_STEPS, replace=False)
            for _ in range(ROWS)
        ]
    )
    for row in range(ROWS):
        selected = selected_rows[row]
        groups: dict[int, list[int]] = {}
        for index in selected:
            groups.setdefault(int(episode_indices[index]), []).append(int(index))
        for values in groups.values():
            rng.shuffle(values)
        heap = [
            (-len(values), float(rng.random()), episode)
            for episode, values in groups.items()
        ]
        heapq.heapify(heap)
        ordered = []
        held: tuple[int, float, int] | None = None
        while heap:
            count, _tie, episode = heapq.heappop(heap)
            ordered.append(groups[episode].pop())
            count += 1
            if held is not None:
                heapq.heappush(heap, held)
            held = (count, float(rng.random()), episode) if count < 0 else None
        if held is not None or len(ordered) != MAX_CONTROL_STEPS:
            raise ValueError("could not separate consecutive source episodes")
        selected_rows[row] = np.asarray(ordered, dtype=selected_rows.dtype)
    selected_episodes = episode_indices[selected_rows]
    selected_timesteps = timesteps[selected_rows]
    if any(np.any(row[1:] == row[:-1]) for row in selected_episodes):
        raise AssertionError("consecutive source episode invariant failed")
    return selected_episodes, selected_timesteps


def build(
    *,
    dataset_dir: Path,
    split_manifest: Path,
    dataset_manifest: Path,
    data_dir: Path,
    manifest_output: Path,
) -> dict[str, Any]:
    if data_dir.exists() and any(data_dir.iterdir()):
        raise ValueError(f"token data directory is not empty: {data_dir}")
    if manifest_output.exists():
        raise ValueError(f"token manifest already exists: {manifest_output}")
    data_dir.mkdir(parents=True, exist_ok=True)
    split = load_split(split_manifest)
    dataset = json.loads(dataset_manifest.read_text())
    train_indices = [
        int(row["act_episode_index"])
        for row in split["episodes"]
        if row["split"] == "train"
    ]
    if len(train_indices) != 199:
        raise ValueError("frozen train split must contain 199 episodes")
    flat_episodes = []
    flat_timesteps = []
    for episode_index in train_indices:
        path = dataset_dir / f"episode_{episode_index}.hdf5"
        with h5py.File(path, "r") as episode:
            proximity = episode["observations/proximity_embeddings"]
            if proximity.shape[1:] != (SENSORS, FEATURE_DIM):
                raise ValueError(f"unexpected embedding shape in {path}")
            length = int(proximity.shape[0])
        flat_episodes.append(np.full(length, episode_index, dtype=np.int16))
        flat_timesteps.append(np.arange(length, dtype=np.int16))
    episode_indices = np.concatenate(flat_episodes)
    timesteps = np.concatenate(flat_timesteps)
    selected_episodes, selected_timesteps = select_sources(episode_indices, timesteps)
    episode_path = data_dir / "source_episode_indices.npy"
    timestep_path = data_dir / "source_timesteps.npy"
    token_path = data_dir / "permuted_tokens.npy"
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
            index: stack.enter_context(
                h5py.File(dataset_dir / f"episode_{index}.hdf5", "r")
            )
            for index in train_indices
        }
        for row in range(ROWS):
            for step in range(MAX_CONTROL_STEPS):
                episode_index = int(selected_episodes[row, step])
                timestep = int(selected_timesteps[row, step])
                tokens[row, step] = episodes[episode_index][
                    "observations/proximity_embeddings"
                ][timestep]
    tokens.flush()
    del tokens
    observed = np.load(token_path, mmap_mode="r")
    if observed.shape != (ROWS, MAX_CONTROL_STEPS, SENSORS, FEATURE_DIM):
        raise ValueError("token tensor shape changed")
    if observed.dtype != np.float32 or not np.isfinite(observed).all():
        raise ValueError("token tensor is invalid")
    norms = np.linalg.norm(observed.reshape(-1, FEATURE_DIM), axis=1)
    row_sha256s = [
        hashlib.sha256(np.asarray(observed[row]).tobytes(order="C")).hexdigest()
        for row in range(ROWS)
    ]
    manifest: dict[str, Any] = {
        "schema_version": "pact_contact_permuted_token_plan_v1",
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
        "same_instance_token_row_used_across_all_policy_seeds": True,
        "row_payload_hash_algorithm": "sha256 of C-order float32 payload bytes",
        "row_payload_sha256s": row_sha256s,
        "live_scene_alignment_destroyed": True,
        "split_manifest_path": str(split_manifest.resolve()),
        "split_manifest_sha256": split["split_manifest_sha256"],
        "dataset_manifest_path": str(dataset_manifest.resolve()),
        "dataset_manifest_file_sha256": file_hash(dataset_manifest),
        "dataset_tree_sha256": dataset["converted_tree_file_sha256"],
        "files": {
            "tokens": {
                "path": str(token_path.resolve()),
                "sha256": file_hash(token_path),
                "size_bytes": token_path.stat().st_size,
            },
            "source_episode_indices": {
                "path": str(episode_path.resolve()),
                "sha256": file_hash(episode_path),
                "size_bytes": episode_path.stat().st_size,
            },
            "source_timesteps": {
                "path": str(timestep_path.resolve()),
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
    write_json_atomic(manifest_output, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--manifest-output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build(
        dataset_dir=args.dataset_dir,
        split_manifest=args.split_manifest,
        dataset_manifest=args.dataset_manifest,
        data_dir=args.data_dir,
        manifest_output=args.manifest_output,
    )
    print(manifest["token_plan_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
