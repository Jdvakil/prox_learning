#!/usr/bin/env python3
"""V10.9 step 3 verification: corpus-wide embedding statistics and integrity.

Independent of the writer. Re-derives every statistic by reading the encoded
files back, rather than trusting the numbers ``encode_pact_embedding_tokens.py``
reported while writing them.

Two integrity properties are proved here:

* the four embedding datasets exist with the right shape and dtype and are
  finite and non-collapsed corpus-wide;
* encoding did not disturb anything that existed before it -- the semantic hash
  recomputed over only the pre-embedding dataset names must still equal the
  value recorded at conversion time.

Writes the final create-only ``conversion_manifest_encoded.json``, refreshing
the fields the raw encoder pass leaves stale.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "submodules" / "act"))

from pact_place_v109_contract import (  # noqa: E402
    CANONICAL_SENSOR_NAMES,
    CONTRACT_VERSION_V109,
    CONVERTED_DATASET_ROOT,
    ENCODER_CLASS,
    ENCODER_PATH,
    ENCODER_SCHEMA,
    ENCODER_SHA256,
    N_ACCEPTED,
    N_SENSORS,
    PROXIMITY_FEATURE_DIM,
    SENSOR_ORDER_SHA256,
    WORK_ROOT,
    canonical_payload_sha256,
    empty_authorization,
    sha256_file,
    write_immutable_create_only,
)

PRE_EMBEDDING_DATASETS = {
    "action",
    "observations/qpos",
    "observations/qvel",
    "observations/images/wrist_camera",
    "observations/proximity",
    "observations/proximity_extrinsic_cv",
    "observations/proximity_intrinsic_cv",
    "observations/proximity_sensor_names",
    "pact_provenance/row",
}
EMBEDDING_DATASETS = {
    "observations/proximity_embeddings": ("float32", 3),
    "observations/proximity_positions": ("float32", 3),
    "observations/proximity_valid": ("bool", 2),
    "observations/proximity_valid_probability": ("float32", 2),
}


def deterministic_semantic_sha256(path: Path, allowed: set[str]) -> str:
    """A semantic hash that is actually reproducible.

    The V5 converter's ``_semantic_sha256`` hashes every dataset with
    ``np.ascontiguousarray(...).tobytes()``. For a variable-length UTF-8 string
    dataset numpy returns an *object* array whose ``tobytes()`` serialises
    pointers, not characters, so two reads of the same unmodified file produce
    different digests. ``observations/proximity_sensor_names`` is such a
    dataset, which makes every recorded ``act_semantic_sha256`` unreproducible.
    Object arrays are encoded by their contents here instead.
    """
    digest = hashlib.sha256()
    with h5py.File(path, "r") as handle:
        digest.update(f"sim={bool(handle.attrs['sim'])}".encode())
        names: list[str] = []
        handle.visititems(
            lambda name, obj: names.append(name) if isinstance(obj, h5py.Dataset) else None
        )
        for name in sorted(names):
            if name not in allowed:
                continue
            array = np.ascontiguousarray(handle[name][()])
            digest.update(name.encode())
            digest.update(str(array.dtype).encode())
            digest.update(str(array.shape).encode())
            if array.dtype == object:
                for item in array.ravel():
                    digest.update(
                        item if isinstance(item, bytes) else str(item).encode())
            else:
                digest.update(array.tobytes())
    return digest.hexdigest()


def preservation_check(task: tuple[str, str, str]) -> dict[str, Any]:
    """Prove encoding disturbed nothing, by re-deriving from the V10.8 source.

    Runs in a worker process. Compares ``action``, ``qpos``, ``qvel``, wrist RGB,
    raw proximity and the per-sensor extrinsics/intrinsics element-wise against a
    fresh extraction from the source HDF5 and MP4 -- not against the
    unreproducible conversion-time semantic hash.
    """
    import sys as _sys

    _sys.path.insert(0, str(ROOT))
    _sys.path.insert(0, str(ROOT / "scripts"))
    from convert_pact_place_v109_to_act import (  # noqa: PLC0415
        _decode_controls, _find_wrist_video, _trajectory_group, extract_proximity,
    )
    from convert_obstacle_to_act import _video_frames  # noqa: PLC0415

    act_file, encoded_path, source_path = task
    mismatches: list[str] = []
    with h5py.File(source_path, "r") as source:
        group = _trajectory_group(source)
        action, qpos, qvel, timesteps, _raw = _decode_controls(group)
        proximity, extrinsic, intrinsic = extract_proximity(group, timesteps)
    wrist = _video_frames(_find_wrist_video(Path(source_path).parent), 240, 320)[:timesteps]

    expected = {
        "action": action,
        "observations/qpos": qpos,
        "observations/qvel": qvel,
        "observations/images/wrist_camera": wrist,
        "observations/proximity": proximity,
        "observations/proximity_extrinsic_cv": extrinsic,
        "observations/proximity_intrinsic_cv": intrinsic,
    }
    with h5py.File(encoded_path, "r") as encoded:
        for name, want in expected.items():
            got = np.asarray(encoded[name][()])
            if got.shape != want.shape:
                mismatches.append(f"{name}: shape {got.shape} != {want.shape}")
            elif not np.array_equal(got, want):
                mismatches.append(f"{name}: contents differ")
        names = [n.decode() if isinstance(n, bytes) else str(n)
                 for n in encoded["observations/proximity_sensor_names"][()]]
    if names != list(CANONICAL_SENSOR_NAMES):
        mismatches.append("proximity_sensor_names differ from the canonical order")
    return {"act_file": act_file, "preserved": not mismatches, "mismatches": mismatches}


def verify_encoder() -> dict[str, Any]:
    import torch  # noqa: PLC0415 - deferred, torch import is slow

    from surface_proximity_encoder import (  # noqa: PLC0415
        SURFACE_EMBEDDING_DIM,
        load_frozen_surface_embedding_encoder,
    )

    path = Path(ENCODER_PATH)
    digest = sha256_file(path)
    if digest != ENCODER_SHA256:
        raise SystemExit(f"encoder hash {digest} != required {ENCODER_SHA256}")
    model, payload = load_frozen_surface_embedding_encoder(path, map_location="cpu")
    frozen = all(not p.requires_grad for p in model.parameters())
    return {
        "path": str(path),
        "sha256": digest,
        "sha256_matches_contract": True,
        "class": type(model).__name__,
        "class_matches_contract": type(model).__name__ == ENCODER_CLASS,
        # The checkpoint declares its schema under ``schema_version``.
        "schema": payload.get("schema_version"),
        "schema_matches_contract": payload.get("schema_version") == ENCODER_SCHEMA,
        "declared_frozen": bool(payload.get("frozen")),
        "checkpoint_sensor_order_sha256": payload.get("sensor_order_sha256"),
        "sensor_order_matches_contract":
            payload.get("sensor_order_sha256") == SENSOR_ORDER_SHA256,
        "variant": payload.get("variant"),
        "seed": payload.get("seed"),
        "feature_dim": int(SURFACE_EMBEDDING_DIM),
        "feature_dim_matches_contract": int(SURFACE_EMBEDDING_DIM) == PROXIMITY_FEATURE_DIM,
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "all_parameters_frozen": bool(frozen),
        "best_epoch": payload.get("best_epoch"),
        "policy_feature_dim": payload.get("policy_feature_dim"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / CONVERTED_DATASET_ROOT)
    parser.add_argument("--conversion-manifest", type=Path,
                        default=ROOT / WORK_ROOT / "conversion_manifest.json")
    parser.add_argument("--raw-encoded-manifest", type=Path, required=True)
    parser.add_argument("--raw-encoding-report", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path,
                        default=ROOT / WORK_ROOT / "conversion_manifest_encoded.json")
    parser.add_argument("--report-out", type=Path,
                        default=ROOT / WORK_ROOT / "embedding_report.json")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--expected-episodes", type=int, default=None,
                        help="override the V10.9 count when reused by a later version")
    args = parser.parse_args()

    conversion = json.loads(args.conversion_manifest.read_text())
    raw_encoded = json.loads(args.raw_encoded_manifest.read_text())
    raw_report = json.loads(args.raw_encoding_report.read_text())
    problems: list[str] = []

    encoder = verify_encoder()
    for key in ("sha256_matches_contract", "class_matches_contract",
                "schema_matches_contract", "feature_dim_matches_contract",
                "sensor_order_matches_contract", "declared_frozen",
                "all_parameters_frozen"):
        if not encoder[key]:
            problems.append(f"encoder: {key} is false ({encoder})")
    if raw_report.get("checkpoint_sha256") != ENCODER_SHA256:
        problems.append(
            f"encoding report used checkpoint {raw_report.get('checkpoint_sha256')}")

    episodes = sorted(conversion["episodes"], key=lambda e: int(e["act_episode_index"]))
    expected_episodes = args.expected_episodes or N_ACCEPTED
    if len(episodes) != expected_episodes:
        problems.append(f"conversion manifest holds {len(episodes)} episodes, "
                        f"expected {expected_episodes}")

    total = np.zeros(PROXIMITY_FEATURE_DIM, dtype=np.float64)
    total_square = np.zeros(PROXIMITY_FEATURE_DIM, dtype=np.float64)
    minimum = np.full(PROXIMITY_FEATURE_DIM, np.inf, dtype=np.float64)
    maximum = np.full(PROXIMITY_FEATURE_DIM, -np.inf, dtype=np.float64)
    count = 0
    nonfinite_windows = 0
    valid_true = 0
    per_file: list[dict[str, Any]] = []

    print(f"reading back {len(episodes)} encoded episodes ...", flush=True)
    for position, episode in enumerate(episodes, start=1):
        path = args.dataset_dir / episode["act_file"]
        with h5py.File(path, "r") as handle:
            observations = handle["observations"]
            timesteps = int(handle["action"].shape[0])
            for name, (dtype, ndim) in EMBEDDING_DATASETS.items():
                key = name.split("/", 1)[1]
                if key not in observations:
                    problems.append(f"{episode['act_file']}: missing {name}")
                    continue
                dataset = observations[key]
                if str(dataset.dtype) != dtype:
                    problems.append(f"{episode['act_file']}: {name} dtype {dataset.dtype}")
                if len(dataset.shape) != ndim or dataset.shape[0] != timesteps \
                        or dataset.shape[1] != N_SENSORS:
                    problems.append(f"{episode['act_file']}: {name} shape {dataset.shape}")
            embeddings = np.asarray(observations["proximity_embeddings"][()], dtype=np.float32)
            if embeddings.shape != (timesteps, N_SENSORS, PROXIMITY_FEATURE_DIM):
                problems.append(f"{episode['act_file']}: embedding shape {embeddings.shape}")
            finite = np.isfinite(embeddings)
            if not finite.all():
                nonfinite_windows += int((~finite).any(axis=2).sum())
            flat = embeddings.reshape(-1, PROXIMITY_FEATURE_DIM).astype(np.float64)
            total += flat.sum(axis=0)
            total_square += np.square(flat).sum(axis=0)
            minimum = np.minimum(minimum, flat.min(axis=0))
            maximum = np.maximum(maximum, flat.max(axis=0))
            count += flat.shape[0]
            valid_true += int(np.asarray(observations["proximity_valid"][()]).sum())
            names = [n.decode() if isinstance(n, bytes) else str(n)
                     for n in observations["proximity_sensor_names"][()]]
            if names != list(CANONICAL_SENSOR_NAMES):
                problems.append(f"{episode['act_file']}: sensor order changed")
            attrs = dict(handle.attrs)
        if attrs.get("pact_surface_encoder_sha256") != ENCODER_SHA256:
            problems.append(f"{episode['act_file']}: encoder attr "
                            f"{attrs.get('pact_surface_encoder_sha256')}")
        if attrs.get("pact_frontend_schema") != ENCODER_SCHEMA:
            problems.append(f"{episode['act_file']}: schema attr "
                            f"{attrs.get('pact_frontend_schema')}")
        if int(attrs.get("pact_proximity_feature_dim", -1)) != PROXIMITY_FEATURE_DIM:
            problems.append(f"{episode['act_file']}: feature dim attr")
        if not bool(attrs.get("pact_surface_tokens_frozen", False)):
            problems.append(f"{episode['act_file']}: tokens not marked frozen")
        if attrs.get("pact_sensor_order_sha256") != SENSOR_ORDER_SHA256:
            problems.append(f"{episode['act_file']}: sensor order attr")

        preserved = deterministic_semantic_sha256(path, PRE_EMBEDDING_DATASETS)
        post_sha = sha256_file(path)
        per_file.append({
            "act_episode_index": int(episode["act_episode_index"]),
            "act_file": episode["act_file"],
            "episode_id": episode["episode_id"],
            "timesteps": timesteps,
            "sensor_windows": timesteps * N_SENSORS,
            "pre_embedding_act_file_sha256": episode["act_file_sha256"],
            "deterministic_pre_embedding_semantic_sha256": preserved,
            "post_encoding_sha256": post_sha,
            "source_h5": episode["source_h5"],
        })
        if position % 20 == 0 or position == len(episodes):
            print(f"  {position}/{len(episodes)}", flush=True)

    print(f"re-deriving all {len(episodes)} episodes from the V10.8 source ...",
          flush=True)
    preservation: list[dict[str, Any]] = []
    tasks = [
        (e["act_file"], str(args.dataset_dir / e["act_file"]),
         str(ROOT / e["source_h5"]))
        for e in episodes
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(preservation_check, t): t[0] for t in tasks}
        for done, future in enumerate(as_completed(futures), start=1):
            try:
                preservation.append(future.result())
            except Exception as exc:  # noqa: BLE001 - a failure is the finding
                preservation.append({"act_file": futures[future], "preserved": False,
                                     "mismatches": [f"raised: {exc!r}"]})
            if done % 20 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)}", flush=True)
    not_preserved = [p for p in preservation if not p["preserved"]]
    for entry in not_preserved:
        problems.append(f"{entry['act_file']}: {entry['mismatches'][:2]}")

    mean = total / count
    variance = np.maximum(total_square / count - np.square(mean), 0.0)
    std = np.sqrt(variance)
    dead = [int(i) for i in np.flatnonzero(std <= 1e-8)]
    global_std = float(np.sqrt(np.maximum(
        total_square.sum() / (count * PROXIMITY_FEATURE_DIM)
        - (total.sum() / (count * PROXIMITY_FEATURE_DIM)) ** 2, 0.0)))

    expected_windows = sum(e["timesteps"] for e in conversion["episodes"]) * N_SENSORS
    if count != expected_windows:
        problems.append(f"encoded {count} windows, manifest implies {expected_windows}")
    if nonfinite_windows:
        problems.append(f"{nonfinite_windows} windows carry non-finite embeddings")
    if dead:
        problems.append(f"dead embedding dimensions: {dead}")

    tree = hashlib.sha256()
    for entry in per_file:
        tree.update(f"{entry['act_file']}\x1f{entry['post_encoding_sha256']}\n".encode())
    tree_hex = tree.hexdigest()
    if raw_encoded.get("converted_tree_file_sha256") != tree_hex:
        problems.append(
            f"encoder pass recorded tree {raw_encoded.get('converted_tree_file_sha256')}, "
            f"read-back gives {tree_hex}")

    statistics = {
        "windows_encoded": count,
        "windows_expected": expected_windows,
        "feature_dim": PROXIMITY_FEATURE_DIM,
        "all_finite": nonfinite_windows == 0,
        "nonfinite_windows": nonfinite_windows,
        "dead_dimensions": dead,
        "dead_dimension_count": len(dead),
        "collapsed": bool(dead) or global_std <= 1e-8,
        "global_std": global_std,
        "per_dimension_std": [float(x) for x in std],
        "per_dimension_std_min": float(std.min()),
        "per_dimension_std_median": float(np.median(std)),
        "per_dimension_std_max": float(std.max()),
        "per_dimension_mean_min": float(mean.min()),
        "per_dimension_mean_max": float(mean.max()),
        "value_min": float(minimum.min()),
        "value_max": float(maximum.max()),
        "proximity_valid_true_fraction": valid_true / count,
    }

    report: dict[str, Any] = {
        **empty_authorization(),
        "schema_version": "pact_place_v109_embedding_report_v1",
        "contract_version": CONTRACT_VERSION_V109,
        "role": "independent read-back verification of the frozen proximity embeddings",
        "dataset_dir": str(args.dataset_dir.relative_to(ROOT)),
        "encoder": encoder,
        "corpus_statistics": statistics,
        "preservation": {
            "method": "element-wise re-derivation of action, qpos, qvel, wrist RGB, "
                      "raw proximity and per-sensor extrinsics/intrinsics from the "
                      "V10.8 source HDF5 and MP4",
            "episodes_checked": len(preservation),
            "episodes_preserved": sum(1 for e in preservation if e["preserved"]),
            "failures": not_preserved,
        },
        "inherited_semantic_hash_defect": {
            "affected_fields": ["episodes[].act_semantic_sha256",
                                "converted_tree_semantic_sha256"],
            "cause": "the V5 converter's _semantic_sha256 hashes numpy object arrays "
                     "with tobytes(), which serialises pointers rather than string "
                     "contents; observations/proximity_sensor_names is such a dataset",
            "consequence": "those two fields are not reproducible and are dropped "
                           "from the encoded manifest rather than carried forward",
            "downstream_impact": "none -- training verifies act_file_sha256 and "
                                 "converted_tree_file_sha256, which are raw-file "
                                 "byte digests and are deterministic",
        },
        "note_on_window_count":
            "The raw V10.8 HDF5 corpus is 71,511 timesteps / 2,860,440 sensor windows. "
            "The converted ACT corpus is one timestep shorter per episode because the "
            "trailing empty action row is dropped (proven V5 semantics), giving "
            "71,370 timesteps / 2,854,800 windows. Embeddings are generated over the "
            "converted corpus, which is what training reads.",
        "per_file": per_file,
        "converted_tree_file_sha256": tree_hex,
        "problems": problems,
        "verified": not problems,
    }
    report["payload_sha256"] = canonical_payload_sha256(report)
    write_immutable_create_only(args.report_out, report)

    updated = json.loads(json.dumps(conversion))
    by_index = {e["act_episode_index"]: e for e in per_file}
    for episode in updated["episodes"]:
        entry = by_index[int(episode["act_episode_index"])]
        episode["pre_embedding_act_file_sha256"] = episode["act_file_sha256"]
        episode["act_file_sha256"] = entry["post_encoding_sha256"]
        episode["act_h5_sha256"] = entry["post_encoding_sha256"]
        episode["pre_embedding_act_semantic_sha256"] = episode["act_semantic_sha256"]
        episode.pop("act_semantic_sha256", None)
    updated["pre_embedding_converted_tree_file_sha256"] = \
        conversion["converted_tree_file_sha256"]
    updated["converted_tree_file_sha256"] = tree_hex
    updated["pre_embedding_converted_tree_semantic_sha256"] = \
        conversion["converted_tree_semantic_sha256"]
    updated.pop("converted_tree_semantic_sha256", None)
    updated["schema_version"] = "pact_place_v109_conversion_manifest_encoded_v1"
    updated["embedding_token_encoding"] = {
        "encoder_path": ENCODER_PATH,
        "encoder_sha256": ENCODER_SHA256,
        "encoder_schema": ENCODER_SCHEMA,
        "encoder_class": ENCODER_CLASS,
        "feature_dim": PROXIMITY_FEATURE_DIM,
        "windows_encoded": count,
        "embedding_report": str(args.report_out.relative_to(ROOT)),
        "embedding_report_payload_sha256": report["payload_sha256"],
    }
    updated["proximity_contract"] = {
        "raw_channel_present": True,
        "shape": [N_SENSORS, 4, 8, 8],
        "embedding_tokens_present": True,
        "embedding_shape": [N_SENSORS, PROXIMITY_FEATURE_DIM],
        "sensor_order_sha256": SENSOR_ORDER_SHA256,
    }
    updated.pop("payload_sha256", None)
    updated["payload_sha256"] = canonical_payload_sha256(updated)
    written = write_immutable_create_only(args.manifest_out, updated)

    print(json.dumps({
        "verified": report["verified"],
        "problems": problems[:8],
        "windows_encoded": count,
        "all_finite": statistics["all_finite"],
        "episodes_preserved": report["preservation"]["episodes_preserved"],
        "dead_dimension_count": statistics["dead_dimension_count"],
        "global_std": statistics["global_std"],
        "per_dimension_std_min": statistics["per_dimension_std_min"],
        "per_dimension_std_max": statistics["per_dimension_std_max"],
        "proximity_valid_true_fraction": statistics["proximity_valid_true_fraction"],
        "converted_tree_file_sha256": tree_hex,
        "encoded_manifest_payload_sha256": updated["payload_sha256"],
        "encoded_manifest_raw_sha256": written.get("raw_file_sha256"),
    }, indent=2))
    return 0 if report["verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
