#!/usr/bin/env python3
"""Add frozen 32-D per-sensor embeddings to the rematerialized ACT dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
ACT = ROOT / "submodules" / "act"
sys.path.insert(0, str(ACT))

from surface_proximity_encoder import (
    SURFACE_EMBEDDING_DIM,
    causal_sensor_window,
    load_frozen_surface_embedding_encoder,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protected_eval_processes() -> list[int]:
    matches = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if b"eval_act_obstacle_on_policy.py" in command:
            matches.append(int(entry.name))
    return matches


def encode_episode(
    path: Path,
    *,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    checkpoint_sha256: str,
) -> dict:
    with h5py.File(path, "r+") as handle:
        observations = handle["observations"]
        for name in (
            "proximity_embeddings",
            "proximity_positions",
            "proximity_valid",
            "proximity_valid_probability",
        ):
            if name in observations:
                raise RuntimeError(f"{path}: refusing to replace existing {name}")
        proximity = observations["proximity"]
        timesteps, sensors = proximity.shape[:2]
        embeddings = observations.create_dataset(
            "proximity_embeddings",
            shape=(timesteps, sensors, SURFACE_EMBEDDING_DIM),
            dtype="float32",
            chunks=(1, sensors, SURFACE_EMBEDDING_DIM),
            compression="gzip",
            compression_opts=4,
        )
        positions = observations.create_dataset(
            "proximity_positions",
            shape=(timesteps, sensors, 3),
            dtype="float32",
            chunks=(1, sensors, 3),
            compression="gzip",
            compression_opts=4,
        )
        validity = observations.create_dataset(
            "proximity_valid",
            shape=(timesteps, sensors),
            dtype="bool",
            chunks=(1, sensors),
            compression="gzip",
            compression_opts=4,
        )
        probabilities = observations.create_dataset(
            "proximity_valid_probability",
            shape=(timesteps, sensors),
            dtype="float32",
            chunks=(1, sensors),
            compression="gzip",
            compression_opts=4,
        )
        pending_windows = []
        pending_indices = []
        feature_sum = np.zeros(SURFACE_EMBEDDING_DIM, dtype=np.float64)
        feature_square_sum = np.zeros(
            SURFACE_EMBEDDING_DIM, dtype=np.float64
        )
        feature_count = 0

        def flush() -> None:
            nonlocal feature_count
            if not pending_windows:
                return
            tensor = torch.from_numpy(np.stack(pending_windows)).to(device)
            (
                embedding,
                xyz,
                valid,
                probability,
                _reconstruction,
            ) = model.predict(tensor)
            embedding_np = embedding.cpu().numpy()
            xyz_np = xyz.cpu().numpy()
            valid_np = valid.cpu().numpy()
            probability_np = probability.cpu().numpy()
            for output_index, (timestep, sensor) in enumerate(
                pending_indices
            ):
                embeddings[timestep, sensor] = embedding_np[output_index]
                positions[timestep, sensor] = xyz_np[output_index]
                validity[timestep, sensor] = valid_np[output_index]
                probabilities[timestep, sensor] = probability_np[output_index]
            feature_sum[:] += embedding_np.sum(axis=0, dtype=np.float64)
            feature_square_sum[:] += np.square(embedding_np).sum(
                axis=0, dtype=np.float64
            )
            feature_count += len(embedding_np)
            pending_windows.clear()
            pending_indices.clear()

        for timestep in range(timesteps):
            start = max(0, timestep - 7)
            raw_block = np.asarray(
                proximity[start : timestep + 1], dtype=np.float32
            )
            if len(raw_block) < 8:
                raw_block = np.concatenate(
                    (
                        np.repeat(
                            raw_block[:1], 8 - len(raw_block), axis=0
                        ),
                        raw_block,
                    )
                )
            for sensor in range(sensors):
                pending_windows.append(
                    causal_sensor_window(raw_block, 7, sensor).astype(
                        np.float32
                    )
                )
                pending_indices.append((timestep, sensor))
                if len(pending_windows) >= batch_size:
                    flush()
        flush()
        handle.attrs["pact_surface_encoder_sha256"] = checkpoint_sha256
        handle.attrs["pact_surface_tokens_frozen"] = True
        handle.attrs[
            "pact_frontend_schema"
        ] = "pact_surface_embedding_encoder_v1"
        handle.attrs["pact_proximity_feature_dim"] = SURFACE_EMBEDDING_DIM
        valid_fraction = float(np.mean(validity[()]))
        mean = feature_sum / feature_count
        variance = np.maximum(
            feature_square_sum / feature_count - np.square(mean), 0.0
        )
        std = np.sqrt(variance)
    return {
        "path": str(path),
        "timesteps": timesteps,
        "sensors": sensors,
        "feature_dim": SURFACE_EMBEDDING_DIM,
        "valid_fraction": valid_fraction,
        "embedding_dimension_std_min": float(std.min()),
        "embedding_dimension_std_median": float(np.median(std)),
        "embedding_dimension_std_max": float(std.max()),
        "post_encoding_sha256": sha256_file(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--conversion-manifest", required=True, type=Path)
    parser.add_argument(
        "--updated-conversion-manifest-out", required=True, type=Path
    )
    parser.add_argument("--report-out", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    active = protected_eval_processes()
    if active:
        raise SystemExit(
            "protected confirmatory evaluation is still active; refusing "
            f"embedding encoding (PIDs {active})"
        )
    checkpoint_sha256 = sha256_file(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, payload = load_frozen_surface_embedding_encoder(
        args.checkpoint, map_location=device
    )
    model.to(device)
    files = sorted(
        args.dataset_dir.glob("episode_*.hdf5"),
        key=lambda path: int(path.stem.split("_")[1]),
    )
    conversion = json.loads(args.conversion_manifest.read_text())
    expected_files = [episode["act_file"] for episode in conversion["episodes"]]
    if [path.name for path in files] != expected_files:
        raise SystemExit("dataset files differ from conversion manifest")
    episodes = []
    for path in files:
        result = encode_episode(
            path,
            model=model,
            device=device,
            batch_size=args.batch_size,
            checkpoint_sha256=checkpoint_sha256,
        )
        episodes.append(result)
        print(
            f"{path.name}: T={result['timesteps']} "
            f"valid={100 * result['valid_fraction']:.2f}% "
            f"median_dim_std={result['embedding_dimension_std_median']:.3f}",
            flush=True,
        )
    report = {
        "schema_version": "pact_embedding_token_encoding_v1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "encoder_best_epoch": payload["best_epoch"],
        "encoder_heldout_metrics": payload["heldout_metrics"],
        "policy_feature_dim": payload["policy_feature_dim"],
        "dataset_dir": str(args.dataset_dir),
        "episode_count": len(episodes),
        "episodes": episodes,
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )

    by_name = {
        Path(episode["path"]).name: episode for episode in episodes
    }
    updated = json.loads(json.dumps(conversion))
    tree = hashlib.sha256()
    for episode in updated["episodes"]:
        post_hash = by_name[episode["act_file"]]["post_encoding_sha256"]
        episode["pre_embedding_act_file_sha256"] = episode[
            "act_file_sha256"
        ]
        episode["act_file_sha256"] = post_hash
        episode["act_h5_sha256"] = post_hash
        tree.update(f"{episode['act_file']}\x1f{post_hash}\n".encode())
    updated["pre_embedding_converted_tree_file_sha256"] = updated[
        "converted_tree_file_sha256"
    ]
    updated["converted_tree_file_sha256"] = tree.hexdigest()
    updated["embedding_token_encoding"] = {
        "schema_version": report["schema_version"],
        "checkpoint_sha256": checkpoint_sha256,
        "policy_feature_dim": SURFACE_EMBEDDING_DIM,
        "report_path": str(args.report_out),
    }
    updated["proximity_contract"]["embedding_tokens_present"] = True
    args.updated_conversion_manifest_out.parent.mkdir(
        parents=True, exist_ok=True
    )
    args.updated_conversion_manifest_out.write_text(
        json.dumps(updated, indent=2, sort_keys=True) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
