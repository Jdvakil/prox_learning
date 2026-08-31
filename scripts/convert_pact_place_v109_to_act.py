#!/usr/bin/env python3
"""V10.9 step 2: convert the 141 accepted V10.8 rows into a fresh ACT dataset.

A new adapter is required because ``convert_pact_collision_to_act.py``'s
``recovered_152`` path is contract-specific: it reads
``configs/pact_place_v5_recovery.json`` and gates on a 152-row key verification.
The *conversion semantics* here are the proven V5 ones -- same decoders, same
output schema, same hashing -- driven by the V10.9 source manifest instead.

Output schema per episode (identical to the V5 converted dataset)::

    /action                                 (T, 8)          float32
    /observations/qpos                      (T, 9)          float32
    /observations/qvel                      (T, 9)          float32
    /observations/images/wrist_camera       (T,240,320,3)   uint8
    /observations/proximity                 (T,40,4,8,8)    float32
    /observations/proximity_extrinsic_cv    (T,40,3,4)      float64
    /observations/proximity_intrinsic_cv    (T,40,3,3)      float64
    /observations/proximity_sensor_names    (40,)           UTF-8

Neither the V10.8 source HDF5 files nor any V5 converted dataset is modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from convert_obstacle_to_act import (  # noqa: E402
    ACTION_DIM,
    _decode_action,
    _decode_qpos_qvel,
    _video_frames,
)
from pact_place_v109_contract import (  # noqa: E402
    CANONICAL_SENSOR_NAMES,
    CONTRACT_VERSION_V109,
    CONVERTED_DATASET_ROOT,
    EPISODE_HORIZON,
    N_ACCEPTED,
    N_SENSORS,
    SENSOR_ORDER_SHA256,
    T_MAX,
    WORK_ROOT,
    canonical_payload_sha256,
    empty_authorization,
    sha256_file,
    write_immutable_create_only,
)

IMAGE_H = 240
IMAGE_W = 320
QPOS_DIM = 9


def _trajectory_group(handle: h5py.File) -> h5py.Group:
    keys = [k for k in handle if k.startswith("traj_")]
    if len(keys) != 1:
        raise RuntimeError(f"expected one trajectory group, found {keys}")
    return handle[keys[0]]


def _decode_controls(group: h5py.Group) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """V5 semantics: the trailing `{}` action row is dropped, T = raw_T - 1."""
    source = group["actions/joint_pos"]
    raw_timesteps = len(source)
    actions = np.zeros((raw_timesteps, ACTION_DIM), dtype=np.float32)
    valid = np.zeros(raw_timesteps, dtype=bool)
    for step in range(raw_timesteps):
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
    return actions[:count], qpos, qvel, count, raw_timesteps


def extract_proximity(group: h5py.Group, timesteps: int):
    """Stack per-sensor tensors in the frozen canonical order.

    Never ``sorted()``: alphabetical order would place ``link5_back_*`` before
    ``link5_front_*`` and silently relabel every sensor slot the PACT
    transformer assigns a positional embedding to.
    """
    names = list(CANONICAL_SENSOR_NAMES)
    observed = list(group["obs/proximity"].keys())
    missing = sorted(set(names) - set(observed))
    unexpected = sorted(set(observed) - set(names))
    if missing or unexpected:
        raise RuntimeError(
            f"proximity sensor mismatch: missing={missing}, unexpected={unexpected}"
        )
    proximity = np.stack(
        [np.asarray(group[f"obs/proximity/{n}"][:timesteps], dtype=np.float32) for n in names],
        axis=1,
    )
    extrinsic = np.stack(
        [
            np.asarray(group[f"obs/sensor_param/{n}/extrinsic_cv"][:timesteps], dtype=np.float64)
            for n in names
        ],
        axis=1,
    )
    intrinsic = np.stack(
        [
            np.asarray(group[f"obs/sensor_param/{n}/intrinsic_cv"][:timesteps], dtype=np.float64)
            for n in names
        ],
        axis=1,
    )
    if proximity.shape != (timesteps, len(names), 4, 8, 8):
        raise RuntimeError(f"proximity shape {proximity.shape}")
    if extrinsic.shape != (timesteps, len(names), 3, 4):
        raise RuntimeError(f"extrinsic shape {extrinsic.shape}")
    if intrinsic.shape != (timesteps, len(names), 3, 3):
        raise RuntimeError(f"intrinsic shape {intrinsic.shape}")
    return proximity, extrinsic, intrinsic


def _find_wrist_video(row_dir: Path) -> Path:
    videos = [p for p in sorted(row_dir.glob("episode_*_wrist_camera.mp4"))
              if "_depth" not in p.name]
    if len(videos) != 1:
        raise RuntimeError(f"{row_dir}: expected one wrist RGB MP4, found {len(videos)}")
    return videos[0]


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


def convert_one(task: tuple[dict[str, Any], str]) -> dict[str, Any]:
    """Convert a single row. Runs in a worker process."""
    row, destination_dir = task
    source_h5 = Path(row["trajectory_h5"])
    if not source_h5.is_absolute():
        source_h5 = ROOT / source_h5
    source_sha = sha256_file(source_h5)
    if source_sha != row["trajectory_h5_sha256"]:
        raise RuntimeError(f"source hash drift for {row['attempt_id'][:16]}: {source_sha}")

    with h5py.File(source_h5, "r") as handle:
        group = _trajectory_group(handle)
        action, qpos, qvel, timesteps, raw_timesteps = _decode_controls(group)
        proximity, extrinsic, intrinsic = extract_proximity(group, timesteps)

    # exact timestep preservation, stated rather than assumed
    if raw_timesteps != int(row["timesteps"]):
        raise RuntimeError(
            f"raw T {raw_timesteps} != source manifest T {row['timesteps']}")
    if timesteps != int(row["episode_steps"]):
        raise RuntimeError(
            f"converted T {timesteps} != episode_steps {row['episode_steps']}")
    if timesteps != raw_timesteps - 1:
        raise RuntimeError(f"converted T {timesteps} != raw T - 1 ({raw_timesteps - 1})")

    wrist = _video_frames(_find_wrist_video(source_h5.parent), IMAGE_H, IMAGE_W)
    wrist_frames_decoded = len(wrist)
    if wrist_frames_decoded != raw_timesteps:
        raise RuntimeError(
            f"wrist video decoded {wrist_frames_decoded} frames, raw T is {raw_timesteps}")
    wrist = wrist[:timesteps]

    if action.shape != (timesteps, ACTION_DIM):
        raise RuntimeError(f"action shape {action.shape}")
    if qpos.shape != (timesteps, QPOS_DIM):
        raise RuntimeError(f"qpos shape {qpos.shape}")
    if wrist.shape != (timesteps, IMAGE_H, IMAGE_W, 3) or wrist.dtype != np.uint8:
        raise RuntimeError(f"wrist shape {wrist.shape} dtype {wrist.dtype}")
    if not np.isfinite(action).all() or not np.isfinite(qpos).all():
        raise RuntimeError("non-finite action or qpos")
    if not np.isfinite(proximity).all():
        raise RuntimeError("non-finite proximity")

    act_file = f"episode_{int(row['act_episode_index'])}.hdf5"
    destination = Path(destination_dir) / act_file
    handle_fd, temporary_name = tempfile.mkstemp(prefix=f".{act_file}.", dir=destination_dir)
    os.close(handle_fd)
    temporary = Path(temporary_name)
    try:
        with h5py.File(temporary, "w") as output:
            output.attrs["sim"] = True
            output.attrs["pact_episode_id"] = row["attempt_id"]
            output.attrs["pact_row_sha256"] = row["row_sha256"]
            output.attrs["pact_sensor_order_sha256"] = SENSOR_ORDER_SHA256
            output.attrs["pact_intrusion_side"] = row["intrusion_side"]
            output.attrs["pact_cell"] = row["cell"]
            output.attrs["pact_family_id"] = row["family_id"]
            output.attrs["pact_pose_id"] = row["pose_id"]
            output.attrs["pact_attempt_index"] = int(row["attempt_index"])
            output.attrs["pact_task_seed_u32"] = int(row["task_seed_u32"])
            output.attrs["pact_source_version"] = "pact_place_v10_8_exploratory_collection"
            output.create_dataset("action", data=action, dtype="float32")
            observations = output.create_group("observations")
            observations.create_dataset("qpos", data=qpos, dtype="float32")
            observations.create_dataset("qvel", data=qvel, dtype="float32")
            images = observations.create_group("images")
            images.create_dataset(
                "wrist_camera", data=wrist, dtype="uint8",
                chunks=(1, IMAGE_H, IMAGE_W, 3), compression="gzip", compression_opts=4,
            )
            observations.create_dataset(
                "proximity", data=proximity, dtype="float32",
                chunks=(1, 1, 4, 8, 8), compression="gzip", compression_opts=4,
            )
            observations.create_dataset(
                "proximity_extrinsic_cv", data=extrinsic, dtype="float64",
                chunks=(1, 1, 3, 4), compression="gzip", compression_opts=4,
            )
            observations.create_dataset(
                "proximity_intrinsic_cv", data=intrinsic, dtype="float64",
                chunks=(1, 1, 3, 3), compression="gzip", compression_opts=4,
            )
            string_type = h5py.string_dtype(encoding="utf-8")
            observations.create_dataset(
                "proximity_sensor_names",
                data=np.asarray(list(CANONICAL_SENSOR_NAMES), dtype=object),
                dtype=string_type,
            )
            provenance = output.create_group("pact_provenance")
            provenance.create_dataset(
                "row", data=json.dumps(row, sort_keys=True).encode())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()

    with h5py.File(destination, "r") as check:
        names = [n.decode() if isinstance(n, bytes) else str(n)
                 for n in check["observations/proximity_sensor_names"][()]]
    if names != list(CANONICAL_SENSOR_NAMES):
        raise RuntimeError("written sensor names differ from the canonical order")
    order_hash = hashlib.sha256(
        json.dumps(names, separators=(",", ":")).encode()).hexdigest()
    if order_hash != SENSOR_ORDER_SHA256:
        raise RuntimeError(f"sensor order hash {order_hash} != {SENSOR_ORDER_SHA256}")

    act_sha = sha256_file(destination)
    return {
        "act_episode_index": int(row["act_episode_index"]),
        "act_file": act_file,
        "episode_id": row["attempt_id"],
        "candidate_index": int(row["attempt_index"]),
        "role_index": int(row["act_episode_index"]),
        "cell": row["cell"],
        "family_id": row["family_id"],
        "intrusion_side": row["intrusion_side"],
        "pose_id": row["pose_id"],
        "task_seed_u32": int(row["task_seed_u32"]),
        "row_sha256": row["row_sha256"],
        "timesteps": timesteps,
        "raw_timesteps": raw_timesteps,
        "wrist_frames_decoded": wrist_frames_decoded,
        "action_shape": list(action.shape),
        "qpos_shape": list(qpos.shape),
        "image_shape": list(wrist.shape),
        "proximity_shape": list(proximity.shape),
        "extrinsic_shape": list(extrinsic.shape),
        "intrinsic_shape": list(intrinsic.shape),
        "sensor_order_sha256": order_hash,
        "source_h5": str(source_h5.relative_to(ROOT)),
        "source_h5_sha256": source_sha,
        "act_file_sha256": act_sha,
        "act_h5_sha256": act_sha,
        "act_semantic_sha256": _semantic_sha256(destination),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path,
                        default=ROOT / WORK_ROOT / "source_manifest.json")
    parser.add_argument("--dst", type=Path, default=ROOT / CONVERTED_DATASET_ROOT)
    parser.add_argument("--manifest-out", type=Path,
                        default=ROOT / WORK_ROOT / "conversion_manifest.json")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--expected-episodes", type=int, default=None,
                        help="override the V10.9 count when reused by a later version")
    args = parser.parse_args()

    source = json.loads(args.source_manifest.read_text())
    if not source.get("verified"):
        raise SystemExit("source manifest is not verified; refusing to convert")
    rows = source["rows"]
    expected = args.expected_episodes or N_ACCEPTED
    if len(rows) != expected:
        raise SystemExit(f"source manifest holds {len(rows)} rows, expected {expected}")
    if [r["act_episode_index"] for r in rows] != list(range(len(rows))):
        raise SystemExit("source manifest act_episode_index is not a dense 0..N-1 range")
    if source["sensor_order_sha256"] != SENSOR_ORDER_SHA256:
        raise SystemExit("source manifest sensor order differs from the contract")

    if args.dst.exists() and any(args.dst.iterdir()):
        raise SystemExit(f"refusing non-empty destination {args.dst}")
    args.dst.mkdir(parents=True, exist_ok=True)

    print(f"converting {len(rows)} episodes -> {args.dst} with {args.workers} workers",
          flush=True)
    episodes: dict[int, dict[str, Any]] = {}
    failures: list[str] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(convert_one, (row, str(args.dst))): row["act_episode_index"]
            for row in rows
        }
        for done, future in enumerate(as_completed(futures), start=1):
            index = futures[future]
            try:
                episodes[index] = future.result()
            except Exception as exc:  # noqa: BLE001 - a conversion failure is fatal
                failures.append(f"episode_{index}: {exc!r}")
            if done % 20 == 0 or done == len(rows):
                print(f"  {done}/{len(rows)}", flush=True)
    if failures:
        for line in failures[:10]:
            print("FAIL", line)
        raise SystemExit(f"{len(failures)} conversion failure(s)")

    ordered = [episodes[i] for i in sorted(episodes)]
    lengths = [e["timesteps"] for e in ordered]
    tree = hashlib.sha256()
    for episode in ordered:
        tree.update(f"{episode['act_file']}\x1f{episode['act_file_sha256']}\n".encode())
    semantic_tree = hashlib.sha256()
    for episode in ordered:
        semantic_tree.update(
            f"{episode['act_file']}\x1f{episode['act_semantic_sha256']}\n".encode())

    if len({e["act_file_sha256"] for e in ordered}) != len(ordered):
        raise SystemExit("duplicate converted file hashes")
    if len({e["episode_id"] for e in ordered}) != len(ordered):
        raise SystemExit("duplicate episode ids in the conversion manifest")
    if max(lengths) > EPISODE_HORIZON:
        raise SystemExit(
            f"converted T_max {max(lengths)} exceeds episode_horizon {EPISODE_HORIZON}")

    document: dict[str, Any] = {
        **empty_authorization(),
        "schema_version": "pact_place_v109_conversion_manifest_v1",
        "contract_version": CONTRACT_VERSION_V109,
        "role": "fresh ACT-format conversion of the 141 accepted V10.8 demonstrations",
        "roles": ["v108_accepted_141"],
        "is_phase0_pass": False,
        "dataset_dir": str(args.dst.relative_to(ROOT)),
        "source_manifest": str(args.source_manifest.relative_to(ROOT)),
        "source_manifest_payload_sha256": source["payload_sha256"],
        "canonical_manifest_sha256": source["payload_sha256"],
        "source_collection_ledger_sha256": source["ledger_sha256"],
        "converter_module_sha256": sha256_file(Path(__file__)),
        "sensor_order_sha256": SENSOR_ORDER_SHA256,
        "sensor_names": list(CANONICAL_SENSOR_NAMES),
        "sensor_order_is_alphabetical":
            list(CANONICAL_SENSOR_NAMES) == sorted(CANONICAL_SENSOR_NAMES),
        "image_h": IMAGE_H,
        "image_w": IMAGE_W,
        "n_sensors": N_SENSORS,
        "episode_count": len(ordered),
        "timesteps": {
            "converted_t_min": min(lengths),
            "converted_t_max": max(lengths),
            "converted_t_sum": sum(lengths),
            "raw_t_min": min(e["raw_timesteps"] for e in ordered),
            "raw_t_max": max(e["raw_timesteps"] for e in ordered),
            "raw_t_sum": sum(e["raw_timesteps"] for e in ordered),
            "rule": "converted T = raw T - 1; the trailing empty action row is dropped "
                    "(proven V5 semantics). Converted T equals the ledger episode_steps.",
            "episode_horizon": EPISODE_HORIZON,
            "episode_horizon_exceeds_converted_t_max": EPISODE_HORIZON > max(lengths),
            "episode_horizon_exceeds_raw_t_max": EPISODE_HORIZON > T_MAX,
        },
        "sensor_windows_for_embedding": sum(lengths) * N_SENSORS,
        "converted_tree_file_sha256": tree.hexdigest(),
        "converted_tree_semantic_sha256": semantic_tree.hexdigest(),
        "episodes": ordered,
    }
    document["payload_sha256"] = canonical_payload_sha256(document)
    written = write_immutable_create_only(args.manifest_out, document)
    print(json.dumps({
        "episodes": len(ordered),
        "converted_t_min": min(lengths),
        "converted_t_max": max(lengths),
        "converted_t_sum": sum(lengths),
        "sensor_windows": sum(lengths) * N_SENSORS,
        "converted_tree_file_sha256": document["converted_tree_file_sha256"],
        "converted_tree_semantic_sha256": document["converted_tree_semantic_sha256"],
        "payload_sha256": document["payload_sha256"],
        "raw_file_sha256": written.get("raw_file_sha256"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
