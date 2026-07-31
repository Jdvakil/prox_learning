#!/usr/bin/env python3
"""Audit the ACT/PPACT conversion confound and zero-token support."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np


CORE_DATASETS = (
    "action",
    "observations/images/wrist_camera",
    "observations/qpos",
    "observations/qvel",
)
RECIPE_FIELDS = (
    "batch_size",
    "camera_names",
    "chunk_size",
    "dim_feedforward",
    "episode_horizon",
    "hidden_dim",
    "kl_weight",
    "lr",
    "num_epochs",
    "policy_class",
    "state_dim",
    "action_dim",
)


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


def episode_files(directory: Path) -> list[Path]:
    paths = list(directory.glob("episode_*.hdf5"))
    return sorted(paths, key=lambda path: int(path.stem.split("_")[-1]))


def dataset_paths(handle: h5py.File) -> set[str]:
    paths: set[str] = set()

    def visit(name: str, value: h5py.Dataset | h5py.Group) -> None:
        if isinstance(value, h5py.Dataset):
            paths.add(name)

    handle.visititems(visit)
    return paths


def iter_first_axis(dataset: h5py.Dataset, rows: int = 8) -> Iterable[np.ndarray]:
    if dataset.ndim == 0:
        yield np.asarray(dataset[()])
        return
    for start in range(0, dataset.shape[0], rows):
        yield np.asarray(dataset[start : start + rows])


def compare_core_payloads(
    old_dir: Path,
    new_dir: Path,
) -> dict[str, Any]:
    old_files = episode_files(old_dir)
    new_files = episode_files(new_dir)
    old_names = [path.name for path in old_files]
    new_names = [path.name for path in new_files]
    names_equal = old_names == new_names
    common = sorted(set(old_names) & set(new_names), key=lambda x: int(Path(x).stem.split("_")[-1]))
    digests = {
        path: {"old": hashlib.sha256(), "new": hashlib.sha256()}
        for path in CORE_DATASETS
    }
    shape_or_dtype_mismatches: list[dict[str, Any]] = []
    payload_mismatches: list[dict[str, Any]] = []
    length_mismatches: list[dict[str, Any]] = []
    key_mismatches: list[dict[str, Any]] = []
    old_lengths: dict[str, int] = {}
    new_lengths: dict[str, int] = {}
    for name in common:
        with h5py.File(old_dir / name, "r") as old, h5py.File(
            new_dir / name, "r"
        ) as new:
            old_keys = dataset_paths(old)
            new_keys = dataset_paths(new)
            if old_keys - new_keys or new_keys - old_keys != {
                "observations/proximity_embeddings"
            }:
                key_mismatches.append(
                    {
                        "episode": name,
                        "only_old": sorted(old_keys - new_keys),
                        "only_new": sorted(new_keys - old_keys),
                    }
                )
            old_lengths[name] = int(old["action"].shape[0])
            new_lengths[name] = int(new["action"].shape[0])
            if old_lengths[name] != new_lengths[name]:
                length_mismatches.append(
                    {
                        "episode": name,
                        "old": old_lengths[name],
                        "new": new_lengths[name],
                    }
                )
            for path in CORE_DATASETS:
                old_dataset = old[path]
                new_dataset = new[path]
                metadata = (
                    name,
                    path,
                    tuple(old_dataset.shape),
                    old_dataset.dtype.str,
                )
                for side in ("old", "new"):
                    digests[path][side].update(repr(metadata).encode())
                if (
                    old_dataset.shape != new_dataset.shape
                    or old_dataset.dtype != new_dataset.dtype
                ):
                    shape_or_dtype_mismatches.append(
                        {
                            "episode": name,
                            "dataset": path,
                            "old_shape": list(old_dataset.shape),
                            "new_shape": list(new_dataset.shape),
                            "old_dtype": old_dataset.dtype.str,
                            "new_dtype": new_dataset.dtype.str,
                        }
                    )
                    continue
                for chunk_index, (old_chunk, new_chunk) in enumerate(
                    zip(iter_first_axis(old_dataset), iter_first_axis(new_dataset))
                ):
                    old_bytes = np.ascontiguousarray(old_chunk).tobytes()
                    new_bytes = np.ascontiguousarray(new_chunk).tobytes()
                    digests[path]["old"].update(old_bytes)
                    digests[path]["new"].update(new_bytes)
                    if not np.array_equal(old_chunk, new_chunk):
                        payload_mismatches.append(
                            {
                                "episode": name,
                                "dataset": path,
                                "chunk_index": chunk_index,
                            }
                        )
                        break
    digest_report = {
        path: {
            "old_sha256": pair["old"].hexdigest(),
            "new_sha256": pair["new"].hexdigest(),
            "equal": pair["old"].digest() == pair["new"].digest(),
        }
        for path, pair in digests.items()
    }
    passed = bool(
        names_equal
        and len(common) == len(old_files) == len(new_files)
        and not length_mismatches
        and not key_mismatches
        and not shape_or_dtype_mismatches
        and not payload_mismatches
        and all(value["equal"] for value in digest_report.values())
    )
    return {
        "passed": passed,
        "old_episode_count": len(old_files),
        "new_episode_count": len(new_files),
        "episode_filenames_identical": names_equal,
        "episode_lengths_identical": old_lengths == new_lengths,
        "only_added_dataset_is_proximity_embeddings": not key_mismatches,
        "length_mismatches": length_mismatches[:20],
        "key_mismatches": key_mismatches[:20],
        "shape_or_dtype_mismatches": shape_or_dtype_mismatches[:20],
        "payload_mismatches": payload_mismatches[:20],
        "core_payload_digests": digest_report,
    }


def summarize_tokens(
    directory: Path,
    episode_indices: list[int],
    dataset_path: str,
    valid_path: str | None = None,
) -> dict[str, Any]:
    sums: np.ndarray | None = None
    sums_sq: np.ndarray | None = None
    norms: list[np.ndarray] = []
    count = 0
    exact_zero = 0
    valid_count = 0
    invalid_count = 0
    valid_exact_zero = 0
    invalid_exact_zero = 0
    feature_dim = 0
    for index in episode_indices:
        with h5py.File(directory / f"episode_{index}.hdf5", "r") as handle:
            dataset = handle[dataset_path]
            for start in range(0, dataset.shape[0], 32):
                values = np.asarray(dataset[start : start + 32], dtype=np.float64)
                flat = values.reshape(-1, values.shape[-1])
                feature_dim = flat.shape[1]
                if sums is None:
                    sums = np.zeros(feature_dim, dtype=np.float64)
                    sums_sq = np.zeros(feature_dim, dtype=np.float64)
                sums += flat.sum(axis=0)
                sums_sq += np.square(flat).sum(axis=0)
                chunk_norms = np.linalg.norm(flat, axis=1)
                norms.append(chunk_norms.astype(np.float32))
                zero = np.all(flat == 0.0, axis=1)
                exact_zero += int(zero.sum())
                count += int(len(flat))
                if valid_path is not None:
                    valid = np.asarray(
                        handle[valid_path][start : start + 32]
                    ).reshape(-1)
                    valid_count += int(valid.sum())
                    invalid_count += int((~valid).sum())
                    valid_exact_zero += int(np.sum(zero & valid))
                    invalid_exact_zero += int(np.sum(zero & ~valid))
    assert sums is not None and sums_sq is not None and count > 0
    all_norms = np.concatenate(norms)
    mean = sums / count
    variance = np.maximum(sums_sq / count - np.square(mean), 0.0)
    std = np.sqrt(variance)
    z = np.divide(
        np.abs(mean),
        std,
        out=np.full_like(mean, np.inf),
        where=std > 0,
    )
    report: dict[str, Any] = {
        "dataset_path": dataset_path,
        "episode_count": len(episode_indices),
        "feature_dim": feature_dim,
        "token_count": count,
        "norm": {
            "mean": float(all_norms.mean()),
            "minimum": float(all_norms.min()),
            "p01": float(np.percentile(all_norms, 1)),
            "median": float(np.median(all_norms)),
            "maximum": float(all_norms.max()),
            "count_below_0_1": int(np.sum(all_norms < 0.1)),
        },
        "exact_zero_vector_count": exact_zero,
        "exact_zero_vector_fraction": exact_zero / count,
        "zero_vector_distance_from_per_dim_mean": {
            "median_abs_z": float(np.median(z)),
            "maximum_abs_z": float(np.max(z)),
            "dimensions_beyond_3_sigma": int(np.sum(z > 3.0)),
        },
    }
    if valid_path is not None:
        report["validity"] = {
            "valid_count": valid_count,
            "invalid_count": invalid_count,
            "valid_exact_zero_count": valid_exact_zero,
            "invalid_exact_zero_count": invalid_exact_zero,
        }
    return report


def normalization_stats_equal(old_run: dict[str, Any], new_run: dict[str, Any]) -> dict[str, Any]:
    old_path = Path(old_run["dataset_stats_pkl"])
    new_path = Path(new_run["dataset_stats_pkl"])
    old_hash = file_hash(old_path)
    new_hash = file_hash(new_path)
    with old_path.open("rb") as stream:
        old_stats = pickle.load(stream)
    with new_path.open("rb") as stream:
        new_stats = pickle.load(stream)
    semantic_equal = (
        old_stats.keys() == new_stats.keys()
        and all(np.array_equal(old_stats[key], new_stats[key]) for key in old_stats)
    )
    return {
        "old_path": str(old_path),
        "new_path": str(new_path),
        "old_sha256": old_hash,
        "new_sha256": new_hash,
        "file_sha256_identical": old_hash == new_hash,
        "semantic_payload_identical": bool(semantic_equal),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-act-run", required=True, type=Path)
    parser.add_argument("--new-pact-run", required=True, type=Path)
    parser.add_argument("--old-pact-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    old_act = json.loads(args.old_act_run.read_text())
    new_pact = json.loads(args.new_pact_run.read_text())
    old_pact = json.loads(args.old_pact_run.read_text())
    old_dir = Path(old_act["dataset_dir"])
    new_dir = Path(new_pact["dataset_dir"])
    core = compare_core_payloads(old_dir, new_dir)
    normalization = normalization_stats_equal(old_act, new_pact)
    split = {
        "old_manifest_sha256": old_act["split_manifest_sha256"],
        "new_manifest_sha256": new_pact["split_manifest_sha256"],
        "manifest_hashes_identical": (
            old_act["split_manifest_sha256"]
            == new_pact["split_manifest_sha256"]
        ),
        "train_act_indices_identical": (
            old_act["train_act_indices"] == new_pact["train_act_indices"]
        ),
        "val_act_indices_identical": (
            old_act["val_act_indices"] == new_pact["val_act_indices"]
        ),
        "train_episode_ids_identical": (
            old_act["train_episodes"] == new_pact["train_episodes"]
        ),
        "val_episode_ids_identical": (
            old_act["val_episodes"] == new_pact["val_episodes"]
        ),
    }
    recipe = {
        field: {
            "old": old_act[field],
            "new": new_pact[field],
            "identical": old_act[field] == new_pact[field],
        }
        for field in RECIPE_FIELDS
    }
    train_indices = [int(value) for value in old_act["train_act_indices"]]
    old_zero = summarize_tokens(
        Path(old_pact["dataset_dir"]),
        train_indices,
        "observations/proximity_positions",
        "observations/proximity_valid",
    )
    new_zero = summarize_tokens(
        new_dir,
        train_indices,
        "observations/proximity_embeddings",
    )
    equivalence_passed = bool(
        core["passed"]
        and normalization["file_sha256_identical"]
        and normalization["semantic_payload_identical"]
        and all(
            split[key]
            for key in (
                "train_act_indices_identical",
                "val_act_indices_identical",
                "train_episode_ids_identical",
                "val_episode_ids_identical",
            )
        )
        and all(value["identical"] for value in recipe.values())
    )
    report: dict[str, Any] = {
        "schema_version": "pact_act_data_and_zero_support_audit_v1",
        "old_act_run_manifest": str(args.old_act_run.resolve()),
        "new_pact_run_manifest": str(args.new_pact_run.resolve()),
        "old_pact_run_manifest": str(args.old_pact_run.resolve()),
        "old_dataset": str(old_dir.resolve()),
        "new_dataset": str(new_dir.resolve()),
        "core_data_equivalence": core,
        "normalization_statistics": normalization,
        "split_equivalence": split,
        "recipe_equivalence": recipe,
        "equivalence_passed": equivalence_passed,
        "act_retraining_required": not equivalence_passed,
        "zero_support": {
            "old_3d_representation": old_zero,
            "new_32d_representation": new_zero,
            "old_zero_empirically_observed": (
                old_zero["exact_zero_vector_count"] > 0
            ),
            "new_zero_empirically_observed": (
                new_zero["exact_zero_vector_count"] > 0
            ),
        },
        "interpretation": {
            "step_1_decision_is_not_changed_by_this_audit": True,
            "confirmatory_run_authorized": False,
            "reason": "VALID_ABLATION_WEAK_SIGNAL does not clear Step 1",
        },
    }
    report["audit_sha256"] = canonical_hash(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "audit_sha256": report["audit_sha256"],
        "equivalence_passed": equivalence_passed,
        "old_zero_count": old_zero["exact_zero_vector_count"],
        "new_zero_count": new_zero["exact_zero_vector_count"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
