#!/usr/bin/env python3
"""Write frozen per-sensor geometry tokens into an ACT episode dataset.

Uses ``encoders.surface_geometry`` and this repo's pooled ``(T, 40, 8, 8)`` convert
layout (native ``(T, 40, 4, 8, 8)`` still works).

    python -m encoders.encode_tokens \\
        --dataset-dir act_style_data/obstacle_prox_v2 \\
        --checkpoint path/to/pact_surface_embedding_encoder_v1.pt \\
        --kind embedding
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from encoders.surface_geometry import (
    SURFACE_EMBEDDING_DIM,
    SurfaceGeometryEncoder,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _episode_paths(dataset_dir: Path) -> list[Path]:
    files = list(dataset_dir.glob("episode_*.hdf5"))
    files.sort(key=lambda path: int(path.stem.split("_")[1]))
    return files


def encode_episode_file(
    path: Path,
    *,
    model: SurfaceGeometryEncoder,
    batch_size: int,
    checkpoint_sha256: str,
    overwrite: bool,
) -> dict:
    with h5py.File(path, "r+") as handle:
        observations = handle["observations"]
        names = (
            "proximity_embeddings",
            "proximity_positions",
            "proximity_valid",
            "proximity_valid_probability",
        )
        existing = [name for name in names if name in observations]
        if existing and not overwrite:
            raise RuntimeError(
                f"{path}: refusing to replace {existing}. Pass --overwrite."
            )
        for name in existing:
            del observations[name]
        if "proximity" not in observations:
            raise RuntimeError(f"{path}: missing /observations/proximity")
        proximity = np.asarray(observations["proximity"][()], dtype=np.float32)
        packed = model.encode_episode_full(proximity, batch_size=batch_size)
        timesteps, sensors = packed["xyz_m"].shape[:2]
        if "embedding" in packed:
            observations.create_dataset(
                "proximity_embeddings",
                data=packed["embedding"].numpy().astype(np.float32),
                chunks=(1, sensors, SURFACE_EMBEDDING_DIM),
                compression="gzip",
                compression_opts=4,
            )
            feature = packed["embedding"].numpy()
            feature_dim = SURFACE_EMBEDDING_DIM
        else:
            feature = packed["xyz_m"].numpy()
            feature_dim = 3
        observations.create_dataset(
            "proximity_positions",
            data=packed["xyz_m"].numpy().astype(np.float32),
            chunks=(1, sensors, 3),
            compression="gzip",
            compression_opts=4,
        )
        observations.create_dataset(
            "proximity_valid",
            data=packed["valid"].numpy().astype(bool),
            chunks=(1, sensors),
            compression="gzip",
            compression_opts=4,
        )
        observations.create_dataset(
            "proximity_valid_probability",
            data=packed["probabilities"].numpy().astype(np.float32),
            chunks=(1, sensors),
            compression="gzip",
            compression_opts=4,
        )
        handle.attrs["pact_surface_encoder_sha256"] = checkpoint_sha256
        handle.attrs["pact_surface_tokens_frozen"] = True
        handle.attrs["pact_frontend_schema"] = (
            "pact_surface_embedding_encoder_v1"
            if model.kind == "embedding"
            else "pact_surface_encoder_v1"
        )
        handle.attrs["pact_proximity_feature_dim"] = int(feature_dim)
        valid_fraction = float(packed["valid"].float().mean())
        feature_count = feature.shape[0] * feature.shape[1]
        mean = feature.reshape(feature_count, feature_dim).mean(axis=0)
        std = feature.reshape(feature_count, feature_dim).std(axis=0)
    return {
        "path": str(path),
        "timesteps": int(timesteps),
        "sensors": int(sensors),
        "feature_dim": int(feature_dim),
        "valid_fraction": valid_fraction,
        "embedding_dimension_std_min": float(np.min(std)),
        "embedding_dimension_std_median": float(np.median(std)),
        "embedding_dimension_std_max": float(np.max(std)),
        "post_encoding_sha256": sha256_file(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--kind",
        choices=("embedding", "xyz"),
        default="embedding",
        help="surface_embedding (32-d) or nearest_surface (XYZ).",
    )
    parser.add_argument("--conversion-manifest", type=Path, default=None)
    parser.add_argument("--updated-conversion-manifest-out", type=Path, default=None)
    parser.add_argument("--report-out", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing proximity_embeddings / proximity_positions groups",
    )
    args = parser.parse_args()

    checkpoint_sha256 = sha256_file(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SurfaceGeometryEncoder(
        kind=args.kind, checkpoint=args.checkpoint, device=str(device)
    )
    files = _episode_paths(args.dataset_dir)
    if not files:
        raise SystemExit(f"no episode_*.hdf5 under {args.dataset_dir}")
    conversion = None
    if args.conversion_manifest is not None:
        conversion = json.loads(args.conversion_manifest.read_text())
        expected = [episode["act_file"] for episode in conversion["episodes"]]
        if [path.name for path in files] != expected:
            raise SystemExit("dataset files differ from conversion manifest")

    episodes = []
    for path in files:
        result = encode_episode_file(
            path,
            model=model,
            batch_size=args.batch_size,
            checkpoint_sha256=checkpoint_sha256,
            overwrite=args.overwrite,
        )
        episodes.append(result)
        print(
            f"{path.name}: T={result['timesteps']} "
            f"valid={100 * result['valid_fraction']:.2f}% "
            f"median_dim_std={result['embedding_dimension_std_median']:.3f}",
            flush=True,
        )

    payload = model.payload or {}
    report = {
        "schema_version": "pact_embedding_token_encoding_v1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "encoder_best_epoch": payload.get("best_epoch"),
        "encoder_heldout_metrics": payload.get("heldout_metrics"),
        "policy_feature_dim": int(model.act_feat_dim),
        "dataset_dir": str(args.dataset_dir),
        "episode_count": len(episodes),
        "episodes": episodes,
    }
    report_path = args.report_out or (args.dataset_dir / "encode_tokens_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    meta_path = args.dataset_dir / "convert_meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        meta["embedding_token_encoding"] = {
            "schema_version": report["schema_version"],
            "checkpoint_sha256": checkpoint_sha256,
            "policy_feature_dim": int(model.act_feat_dim),
            "kind": args.kind,
            "report_path": str(report_path),
        }
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    if conversion is not None:
        by_name = {Path(episode["path"]).name: episode for episode in episodes}
        updated = json.loads(json.dumps(conversion))
        tree = hashlib.sha256()
        for episode in updated["episodes"]:
            post_hash = by_name[episode["act_file"]]["post_encoding_sha256"]
            episode["pre_embedding_act_file_sha256"] = episode.get("act_file_sha256")
            episode["act_file_sha256"] = post_hash
            episode["act_h5_sha256"] = post_hash
            tree.update(f"{episode['act_file']}\x1f{post_hash}\n".encode())
        updated["pre_embedding_converted_tree_file_sha256"] = updated.get(
            "converted_tree_file_sha256"
        )
        updated["converted_tree_file_sha256"] = tree.hexdigest()
        updated["embedding_token_encoding"] = {
            "schema_version": report["schema_version"],
            "checkpoint_sha256": checkpoint_sha256,
            "policy_feature_dim": int(model.act_feat_dim),
            "report_path": str(report_path),
        }
        updated.setdefault(
            "proximity_contract",
            {"raw_channel_present": True, "shape": [40, 8, 8]},
        )["embedding_tokens_present"] = True
        out = args.updated_conversion_manifest_out or args.conversion_manifest
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(updated, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
