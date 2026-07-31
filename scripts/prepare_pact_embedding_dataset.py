#!/usr/bin/env python3
"""Re-materialize the proximity-preserving ACT dataset without legacy tokens."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import h5py
import numpy as np

RAW_OBSERVATION_DATASETS = (
    "qpos",
    "qvel",
    "proximity",
    "proximity_extrinsic_cv",
    "proximity_intrinsic_cv",
    "proximity_sensor_names",
)
LEGACY_TOKEN_DATASETS = {
    "observations/proximity_positions",
    "observations/proximity_valid",
    "observations/proximity_valid_probability",
    "observations/proximity_embeddings",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _update_array_content(
    digest: Any, array: np.ndarray
) -> None:
    if array.dtype.kind not in {"O", "S", "U"}:
        digest.update(array.tobytes())
        return
    for value in array.reshape(-1):
        if isinstance(value, bytes):
            encoded = value
        elif isinstance(value, np.bytes_):
            encoded = bytes(value)
        else:
            encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)


def semantic_sha256(
    path: Path, *, ignored_datasets: set[str] | None = None
) -> str:
    """Hash values deterministically, including variable-length strings.

    The original converter called ``ndarray.tobytes()`` on HDF5 object
    arrays. That serializes process-local object pointers, so its recorded
    semantic hash cannot be regenerated. This canonical version length-frames
    string values and supports a raw view of an already tokenized file.
    """
    digest = hashlib.sha256()
    ignored = ignored_datasets or set()
    with h5py.File(path, "r") as handle:
        digest.update(f"sim={bool(handle.attrs['sim'])}".encode())
        names: list[str] = []
        handle.visititems(
            lambda name, obj: (
                names.append(name)
                if isinstance(obj, h5py.Dataset) and name not in ignored
                else None
            )
        )
        for name in sorted(names):
            array = np.ascontiguousarray(handle[name][()])
            digest.update(name.encode())
            digest.update(str(array.dtype).encode())
            digest.update(str(array.shape).encode())
            _update_array_content(digest, array)
    return digest.hexdigest()


def copy_episode(source_path: Path, destination: Path) -> dict[str, Any]:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with (
            h5py.File(source_path, "r") as source,
            h5py.File(temporary, "w") as output,
        ):
            for name, value in source.attrs.items():
                if name not in {
                    "pact_surface_encoder_sha256",
                    "pact_surface_tokens_frozen",
                }:
                    output.attrs[name] = value
            source.copy("action", output)
            observations = output.create_group("observations")
            source_observations = source["observations"]
            for name in RAW_OBSERVATION_DATASETS:
                source_observations.copy(name, observations)
            images = observations.create_group("images")
            source_observations["images"].copy("wrist_camera", images)
            if "pact_provenance" in source:
                source.copy("pact_provenance", output)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    with h5py.File(destination, "r") as handle:
        return {
            "timesteps": int(handle["action"].shape[0]),
            "action_shape": list(handle["action"].shape),
            "qpos_shape": list(handle["observations/qpos"].shape),
            "image_shape": list(
                handle["observations/images/wrist_camera"].shape
            ),
            "proximity_shape": list(
                handle["observations/proximity"].shape
            ),
            "extrinsic_shape": list(
                handle["observations/proximity_extrinsic_cv"].shape
            ),
            "intrinsic_shape": list(
                handle["observations/proximity_intrinsic_cv"].shape
            ),
            "raw_channel_present": True,
            "legacy_surface_tokens_present": (
                "proximity_positions" in handle["observations"]
            ),
            "embedding_tokens_present": (
                "proximity_embeddings" in handle["observations"]
            ),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--base-manifest", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--manifest-out", required=True, type=Path)
    args = parser.parse_args()
    if args.destination.exists() and any(args.destination.iterdir()):
        raise SystemExit(f"refusing non-empty destination {args.destination}")
    args.destination.mkdir(parents=True, exist_ok=True)
    base = json.loads(args.base_manifest.read_text())
    episodes = []
    tree_file = hashlib.sha256()
    tree_semantic = hashlib.sha256()
    for source_record in sorted(
        base["episodes"], key=lambda item: int(item["act_episode_index"])
    ):
        index = int(source_record["act_episode_index"])
        filename = source_record["act_file"]
        source_path = args.source_dir / filename
        destination = args.destination / filename
        details = copy_episode(source_path, destination)
        if (
            details["legacy_surface_tokens_present"]
            or details["embedding_tokens_present"]
        ):
            raise RuntimeError(f"{destination}: token-free copy failed")
        source_raw_semantic = semantic_sha256(
            source_path, ignored_datasets=LEGACY_TOKEN_DATASETS
        )
        semantic = semantic_sha256(destination)
        if semantic != source_raw_semantic:
            raise RuntimeError(
                f"{destination}: canonical raw semantic hash {semantic} != "
                f"source raw view {source_raw_semantic}"
            )
        file_hash = sha256_file(destination)
        tree_file.update(f"{filename}\x1f{file_hash}\n".encode())
        tree_semantic.update(f"{filename}\x1f{semantic}\n".encode())
        episodes.append(
            {
                **source_record,
                **details,
                "act_file_sha256": file_hash,
                "act_h5_sha256": file_hash,
                "act_semantic_sha256": semantic,
                "source_encoded_raw_semantic_sha256": (
                    source_raw_semantic
                ),
                "legacy_recorded_semantic_sha256": source_record[
                    "act_semantic_sha256"
                ],
                "legacy_encoded_source_file_sha256": sha256_file(
                    source_path
                ),
            }
        )
        print(
            f"[{index:03d}] {filename} T={details['timesteps']} "
            "raw+extrinsics+intrinsics retained",
            flush=True,
        )
    document = {
        "schema_version": "pact_embedding_act_conversion_v1",
        "semantic_hash_schema": (
            "canonical_h5_values_v1_length_framed_strings"
        ),
        "legacy_semantic_hash_note": (
            "The original converter hashed object-array pointer bytes, so "
            "its per-file semantic hashes are process-dependent and cannot "
            "be reproduced. Each destination instead equals a deterministic "
            "raw-data view of its preserved encoded source."
        ),
        "source_manifest_sha256": base["source_manifest_sha256"],
        "sensor_order_sha256": base["sensor_order_sha256"],
        "sensor_names": base["sensor_names"],
        "roles": base["roles"],
        "requested_count": base["requested_count"],
        "included_count": len(episodes),
        "excluded_count": base["excluded_count"],
        "selection_rule": base["selection_rule"],
        "included_count_by_role": base["included_count_by_role"],
        "excluded": base["excluded"],
        "episodes": episodes,
        "converted_tree_file_sha256": tree_file.hexdigest(),
        "converted_tree_semantic_sha256": tree_semantic.hexdigest(),
        "base_conversion_manifest": str(args.base_manifest),
        "base_conversion_manifest_sha256": sha256_file(
            args.base_manifest
        ),
        "source_encoded_dataset": str(args.source_dir),
        "destination": str(args.destination),
        "proximity_contract": {
            "raw_shape_per_timestep": [40, 4, 8, 8],
            "ordered_sensor_names": True,
            "intrinsics_shape_per_timestep": [40, 3, 3],
            "world_to_sensor_extrinsics_shape_per_timestep": [40, 3, 4],
            "legacy_tokens_removed": True,
            "embedding_tokens_present": False,
        },
        "converter_sha256": sha256_file(Path(__file__)),
    }
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"prepared {len(episodes)} episodes; "
        f"semantic_tree={document['converted_tree_semantic_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
