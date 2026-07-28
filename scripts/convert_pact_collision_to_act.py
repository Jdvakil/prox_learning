#!/usr/bin/env python3
"""Convert PACT corridor rows to ACT HDF5 while preserving all skin data.

Output schema per episode::

    /action                                      (T, 8) float32
    /observations/qpos                           (T, 9) float32
    /observations/qvel                           (T, 9) float32
    /observations/images/wrist_camera            (T,H,W,3) uint8
    /observations/proximity                      (T,40,4,8,8) float32
    /observations/proximity_extrinsic_cv         (T,40,3,4) float64
    /observations/proximity_intrinsic_cv         (T,40,3,3) float64
    /observations/proximity_sensor_names         (40,) UTF-8

Only predeclared rows with collision-free expert task success are eligible.
Exclusions remain in the conversion manifest and no replacement rows are drawn.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from pact_collision_contract import load_manifest, rows_for_role, sha256_file

from scripts.convert_obstacle_to_act import (
    ACTION_DIM,
    _decode_action,
    _decode_qpos_qvel,
    _video_frames,
)

DEFAULT_MANIFEST = ROOT / "configs" / "pact_collision_candidate_manifest_v2.json"
PILOT_USABLE_DEMO_FLOOR = 48


def _trajectory_group(handle: h5py.File) -> h5py.Group:
    keys = [key for key in handle if key.startswith("traj_")]
    if len(keys) != 1:
        raise RuntimeError(f"expected one trajectory group, found {keys}")
    return handle[keys[0]]


def _decode_controls(group: h5py.Group) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    source = group["actions/joint_pos"]
    actions = np.zeros((len(source), ACTION_DIM), dtype=np.float32)
    valid = np.zeros(len(source), dtype=bool)
    for step in range(len(source)):
        actions[step], valid[step] = _decode_action(source[step])
    count = int(valid.sum())
    if count == 0 or not np.all(valid[:count]) or np.any(valid[count:]):
        raise RuntimeError("valid joint actions are not one contiguous prefix")
    qpos = np.stack(
        [_decode_qpos_qvel(group["obs/agent/qpos"][step]) for step in range(count)]
    ).astype(np.float32)
    qvel = np.stack(
        [_decode_qpos_qvel(group["obs/agent/qvel"][step]) for step in range(count)]
    ).astype(np.float32)
    return actions[:count], qpos, qvel, count


def extract_proximity(
    group: h5py.Group,
    sensor_names: list[str],
    timesteps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observed = list(group["obs/proximity"].keys())
    missing = sorted(set(sensor_names) - set(observed))
    unexpected = sorted(set(observed) - set(sensor_names))
    if missing or unexpected:
        raise RuntimeError(
            f"proximity sensor mismatch: missing={missing}, unexpected={unexpected}"
        )
    proximity = np.stack(
        [
            np.asarray(group[f"obs/proximity/{name}"][:timesteps], dtype=np.float32)
            for name in sensor_names
        ],
        axis=1,
    )
    extrinsic = np.stack(
        [
            np.asarray(
                group[f"obs/sensor_param/{name}/extrinsic_cv"][:timesteps],
                dtype=np.float64,
            )
            for name in sensor_names
        ],
        axis=1,
    )
    intrinsic = np.stack(
        [
            np.asarray(
                group[f"obs/sensor_param/{name}/intrinsic_cv"][:timesteps],
                dtype=np.float64,
            )
            for name in sensor_names
        ],
        axis=1,
    )
    expected = (timesteps, len(sensor_names), 4, 8, 8)
    if proximity.shape != expected:
        raise RuntimeError(f"proximity shape {proximity.shape} != {expected}")
    if extrinsic.shape != (timesteps, len(sensor_names), 3, 4):
        raise RuntimeError(f"extrinsic shape is {extrinsic.shape}")
    if intrinsic.shape != (timesteps, len(sensor_names), 3, 3):
        raise RuntimeError(f"intrinsic shape is {intrinsic.shape}")
    return proximity, extrinsic, intrinsic


def _find_wrist_video(row_dir: Path) -> Path:
    videos = [
        path
        for path in sorted(row_dir.glob("episode_*_wrist_camera.mp4"))
        if "_depth" not in path.name
    ]
    if len(videos) != 1:
        raise RuntimeError(f"{row_dir}: expected one wrist RGB MP4, found {len(videos)}")
    return videos[0]


def _is_usable_clean_demo(result: dict[str, Any]) -> bool:
    """Apply the frozen, endpoint-only demonstration filter."""
    audit = result.get("contact_audit", {})
    totals = audit.get("contact_class_totals", {})
    computed = bool(
        result.get("task_success")
        and int(totals.get("hazard_bar", 0)) == 0
        and int(totals.get("other_environment", 0)) == 0
    )
    recorded = result.get("collision_free_task_success")
    if recorded is not None and bool(recorded) != computed:
        raise RuntimeError(
            "recorded collision_free_task_success disagrees with the frozen "
            "contact-taxonomy recomputation"
        )
    return computed


def _semantic_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with h5py.File(path, "r") as handle:
        digest.update(f"sim={bool(handle.attrs['sim'])}".encode())
        names: list[str] = []
        handle.visititems(
            lambda name, obj: names.append(name) if isinstance(obj, h5py.Dataset) else None
        )
        for name in sorted(names):
            array = np.ascontiguousarray(handle[name][()])
            digest.update(name.encode())
            digest.update(str(array.dtype).encode())
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
    return digest.hexdigest()


def _write_episode(
    *,
    destination: Path,
    row: dict[str, Any],
    result: dict[str, Any],
    source_h5: Path,
    sensor_names: list[str],
    sensor_order_sha256: str,
    image_h: int,
    image_w: int,
) -> dict[str, Any]:
    with h5py.File(source_h5, "r") as source:
        group = _trajectory_group(source)
        action, qpos, qvel, timesteps = _decode_controls(group)
        proximity, extrinsic, intrinsic = extract_proximity(
            group, sensor_names, timesteps
        )
    wrist = _video_frames(_find_wrist_video(source_h5.parent), image_h, image_w)
    if len(wrist) < timesteps:
        raise RuntimeError(f"wrist video has {len(wrist)} frames, requires {timesteps}")
    wrist = wrist[:timesteps]

    file_handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    os.close(file_handle)
    temporary = Path(temporary_name)
    try:
        with h5py.File(temporary, "w") as output:
            output.attrs["sim"] = True
            output.attrs["pact_episode_id"] = row["episode_id"]
            output.attrs["pact_row_sha256"] = row["row_sha256"]
            output.attrs["pact_sensor_order_sha256"] = sensor_order_sha256
            output.attrs["pact_intrusion_side"] = row["intrusion_side"]
            output.create_dataset("action", data=action, dtype="float32")
            observations = output.create_group("observations")
            observations.create_dataset("qpos", data=qpos, dtype="float32")
            observations.create_dataset("qvel", data=qvel, dtype="float32")
            images = observations.create_group("images")
            images.create_dataset(
                "wrist_camera",
                data=wrist,
                dtype="uint8",
                chunks=(1, image_h, image_w, 3),
                compression="gzip",
                compression_opts=4,
            )
            observations.create_dataset(
                "proximity",
                data=proximity,
                dtype="float32",
                chunks=(1, 1, 4, 8, 8),
                compression="gzip",
                compression_opts=4,
            )
            observations.create_dataset(
                "proximity_extrinsic_cv",
                data=extrinsic,
                dtype="float64",
                chunks=(1, 1, 3, 4),
                compression="gzip",
                compression_opts=4,
            )
            observations.create_dataset(
                "proximity_intrinsic_cv",
                data=intrinsic,
                dtype="float64",
                chunks=(1, 1, 3, 3),
                compression="gzip",
                compression_opts=4,
            )
            string_type = h5py.string_dtype(encoding="utf-8")
            observations.create_dataset(
                "proximity_sensor_names",
                data=np.asarray(sensor_names, dtype=object),
                dtype=string_type,
            )
            provenance = output.create_group("pact_provenance")
            provenance.create_dataset("row", data=json.dumps(row, sort_keys=True).encode())
            provenance.create_dataset(
                "collection_result", data=json.dumps(result, sort_keys=True).encode()
            )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()

    act_file_sha256 = sha256_file(destination)
    return {
        "timesteps": timesteps,
        "action_shape": list(action.shape),
        "qpos_shape": list(qpos.shape),
        "image_shape": list(wrist.shape),
        "proximity_shape": list(proximity.shape),
        "extrinsic_shape": list(extrinsic.shape),
        "intrinsic_shape": list(intrinsic.shape),
        "source_h5_sha256": sha256_file(source_h5),
        "act_h5_sha256": act_file_sha256,
        "act_file_sha256": act_file_sha256,
        "act_semantic_sha256": _semantic_sha256(destination),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", required=True, type=Path)
    parser.add_argument(
        "--role",
        required=True,
        action="append",
        choices=("pilot_train", "full_train", "full_validation"),
        help="repeat for a combined full_train + full_validation dataset",
    )
    parser.add_argument("--dst", required=True, type=Path)
    parser.add_argument("--manifest-out", required=True, type=Path)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST, type=Path)
    parser.add_argument("--image-h", default=240, type=int)
    parser.add_argument("--image-w", default=320, type=int)
    args = parser.parse_args()

    if args.dst.exists() and any(args.dst.iterdir()):
        raise SystemExit(f"refusing non-empty destination {args.dst}")
    args.dst.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(args.manifest)
    roles = list(dict.fromkeys(args.role))
    requested = [
        row
        for role in roles
        for row in rows_for_role(manifest, role)
    ]
    included: list[tuple[dict[str, Any], dict[str, Any]]] = []
    excluded: list[dict[str, Any]] = []
    for row in requested:
        result_path = args.collection / "rows" / row["episode_id"] / "result.json"
        if not result_path.exists():
            raise SystemExit(f"unreconciled row: missing {result_path}")
        result = json.loads(result_path.read_text())
        if result.get("row_sha256") != row["row_sha256"]:
            raise SystemExit(f"row identity mismatch at {result_path}")
        if _is_usable_clean_demo(result):
            included.append((row, result))
        else:
            excluded.append(
                {
                    "episode_id": row["episode_id"],
                    "role_index": row["role_index"],
                    "status": result.get("status"),
                    "task_success": result.get("task_success"),
                    "collision_free_task_success": result.get(
                        "collision_free_task_success"
                    ),
                }
            )

    if roles == ["pilot_train"] and len(included) < PILOT_USABLE_DEMO_FLOOR:
        raise SystemExit(
            "pilot demonstration floor not met: "
            f"{len(included)} usable clean rows < {PILOT_USABLE_DEMO_FLOOR}; "
            "rows are not replaced or rerun"
        )

    episodes = []
    for act_index, (row, result) in enumerate(included):
        source_h5 = args.collection / "rows" / row["episode_id"] / "trajectory.h5"
        destination = args.dst / f"episode_{act_index}.hdf5"
        details = _write_episode(
            destination=destination,
            row=row,
            result=result,
            source_h5=source_h5,
            sensor_names=manifest["sensor_names"],
            sensor_order_sha256=manifest["sensor_order_sha256"],
            image_h=args.image_h,
            image_w=args.image_w,
        )
        episodes.append(
            {
                "act_episode_index": act_index,
                "act_file": destination.name,
                "episode_id": row["episode_id"],
                "candidate_index": row["candidate_index"],
                "role": row["role"],
                "role_index": row["role_index"],
                "intrusion_side": row["intrusion_side"],
                "row_sha256": row["row_sha256"],
                **details,
            }
        )
        print(
            f"[{act_index:03d}] role={row['role_index']:03d} "
            f"{row['episode_id'][:12]} T={details['timesteps']}"
        )

    tree_semantic = hashlib.sha256()
    tree_file = hashlib.sha256()
    for episode in episodes:
        tree_semantic.update(
            f"episode_{episode['act_episode_index']}.hdf5"
            f"\x1f{episode['act_semantic_sha256']}\n".encode()
        )
        tree_file.update(
            f"episode_{episode['act_episode_index']}.hdf5"
            f"\x1f{episode['act_h5_sha256']}\n".encode()
        )
    conversion = {
        "schema_version": "pact_collision_act_conversion_v2",
        "source_manifest_sha256": manifest["manifest_sha256"],
        "sensor_order_sha256": manifest["sensor_order_sha256"],
        "sensor_names": manifest["sensor_names"],
        "roles": roles,
        "requested_count": len(requested),
        "included_count": len(episodes),
        "excluded_count": len(excluded),
        "selection_rule": (
            "task_success == true and hazard_bar == 0 and other_environment == 0; "
            "all fixed attempts retained in provenance, included rows ordered by frozen "
            "role_index, no replacement or rerun"
        ),
        "pilot_usable_demo_floor": (
            PILOT_USABLE_DEMO_FLOOR if roles == ["pilot_train"] else None
        ),
        "excluded": excluded,
        "episodes": episodes,
        "converted_tree_file_sha256": tree_file.hexdigest(),
        "converted_tree_semantic_sha256": tree_semantic.hexdigest(),
        "converter_sha256": sha256_file(Path(__file__)),
    }
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(json.dumps(conversion, indent=2, sort_keys=True) + "\n")
    print(
        f"converted {len(episodes)}/{len(requested)} rows; "
        f"tree={conversion['converted_tree_semantic_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
