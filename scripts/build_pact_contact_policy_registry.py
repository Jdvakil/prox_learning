#!/usr/bin/env python3
"""Freeze normalized ACT/PACT records for three policy seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SEEDS = (3101, 3102, 3103)
ARMS = ("ACT", "PACT", "PACT_ZERO", "PACT_PERMUTED")
EXPECTED_RECIPE = {
    "backbone": "resnet18",
    "encoder_layers": 7,
    "decoder_layers": 7,
    "heads": 8,
    "hidden_dim": 512,
    "chunk": 100,
    "learning_rate": 1e-5,
    "batch": 8,
    "epochs": 2000,
    "kl_beta": 10,
}
ENCODER_SHA256 = "6fd2dd037e3236b5b6bf7fce8cb2709ead0cf52adcbbe9cbad1061efc2fe3206"


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


def record(
    *,
    checkpoint: str,
    checkpoint_sha256: str,
    stats: str,
    stats_sha256: str,
    encoder: str | None,
    encoder_sha256: str | None,
    feature_dim: int,
    training_source: str,
) -> dict[str, Any]:
    output = {
        "checkpoint_path": checkpoint,
        "checkpoint_sha256": checkpoint_sha256,
        "dataset_stats_path": stats,
        "dataset_stats_sha256": stats_sha256,
        "surface_encoder_path": encoder,
        "surface_encoder_sha256": encoder_sha256,
        "proximity_feature_dim": feature_dim,
        "training_source": training_source,
    }
    for path_key, hash_key in (
        ("checkpoint_path", "checkpoint_sha256"),
        ("dataset_stats_path", "dataset_stats_sha256"),
        ("surface_encoder_path", "surface_encoder_sha256"),
    ):
        path = output[path_key]
        if path is not None and file_hash(Path(path)) != output[hash_key]:
            raise ValueError(f"training artifact changed: {path}")
    return output


def prior_seed_record(summary: dict[str, Any], path: Path) -> dict[str, Any]:
    seed = int(summary["seed"])
    if summary.get("recipe") != EXPECTED_RECIPE:
        raise ValueError(f"PACT recipe changed for seed {seed}")
    if summary.get("encoder_sha256") != ENCODER_SHA256:
        raise ValueError(f"encoder changed for seed {seed}")
    pact = record(
        checkpoint=summary["checkpoint"],
        checkpoint_sha256=summary["checkpoint_sha256"],
        stats=summary["dataset_stats"],
        stats_sha256=summary["dataset_stats_sha256"],
        encoder=summary["encoder"],
        encoder_sha256=summary["encoder_sha256"],
        feature_dim=32,
        training_source=str(path),
    )
    act_info = summary["reused_act"]
    act_checkpoint = Path(act_info["checkpoint"])
    act_stats = act_checkpoint.parent / "dataset_stats.pkl"
    act = record(
        checkpoint=str(act_checkpoint),
        checkpoint_sha256=act_info["checkpoint_sha256"],
        stats=str(act_stats),
        stats_sha256=file_hash(act_stats),
        encoder=None,
        encoder_sha256=None,
        feature_dim=0,
        training_source=str(path),
    )
    return normalized_seed(act, pact)


def seed3103_record(summary: dict[str, Any], path: Path) -> dict[str, Any]:
    if summary.get("schema_version") != "pact_contact_seed3103_training_v1":
        raise ValueError("seed-3103 training schema changed")
    if summary.get("seed") != 3103 or summary.get("recipe") != EXPECTED_RECIPE:
        raise ValueError("seed-3103 recipe changed")
    if summary.get("encoder_sha256") != ENCODER_SHA256:
        raise ValueError("seed-3103 encoder changed")
    records = {}
    for arm in ("ACT", "PACT"):
        source = summary["arms"][arm]
        if source["recipe"] != EXPECTED_RECIPE or source["seed"] != 3103:
            raise ValueError(f"seed-3103 {arm} recipe changed")
        records[arm] = record(
            checkpoint=source["checkpoint"],
            checkpoint_sha256=source["checkpoint_sha256"],
            stats=source["dataset_stats"],
            stats_sha256=source["dataset_stats_sha256"],
            encoder=source["surface_encoder"],
            encoder_sha256=source["surface_encoder_sha256"],
            feature_dim=0 if arm == "ACT" else 32,
            training_source=str(path),
        )
    return normalized_seed(records["ACT"], records["PACT"])


def normalized_seed(act: dict[str, Any], pact: dict[str, Any]) -> dict[str, Any]:
    zero = dict(pact)
    zero["ablation"] = "all_zero_32d_embedding_ood_sensor_failure_probe"
    permuted = dict(pact)
    permuted["ablation"] = "distribution_matched_scene_misaligned_tokens"
    return {
        "ACT": act,
        "PACT": pact,
        "PACT_ZERO": zero,
        "PACT_PERMUTED": permuted,
    }


def build(seed3101_path: Path, seed3102_path: Path, seed3103_path: Path) -> dict[str, Any]:
    paths = {3101: seed3101_path, 3102: seed3102_path, 3103: seed3103_path}
    summaries = {seed: json.loads(path.read_text()) for seed, path in paths.items()}
    if summaries[3101].get("seed") != 3101 or summaries[3102].get("seed") != 3102:
        raise ValueError("prior PACT seed identity changed")
    seeds = {
        "3101": prior_seed_record(summaries[3101], paths[3101]),
        "3102": prior_seed_record(summaries[3102], paths[3102]),
        "3103": seed3103_record(summaries[3103], paths[3103]),
    }
    document: dict[str, Any] = {
        "schema_version": "pact_contact_policy_registry_v1",
        "checkpoint_seeds": list(SEEDS),
        "arms": list(ARMS),
        "recipe": EXPECTED_RECIPE,
        "encoder_sha256": ENCODER_SHA256,
        "source_training_summaries": {
            str(seed): {
                "path": str(path),
                "file_sha256": file_hash(path),
            }
            for seed, path in paths.items()
        },
        "PACT_ZERO_label": "OOD sensor-failure probe; never modality evidence",
        "PACT_PERMUTED_label": "distribution-matched modality-information instrument",
        "seeds": seeds,
    }
    document["policy_registry_sha256"] = canonical_hash(document)
    return document


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed3101", required=True, type=Path)
    parser.add_argument("--seed3102", required=True, type=Path)
    parser.add_argument("--seed3103", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    document = build(args.seed3101, args.seed3102, args.seed3103)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(document["policy_registry_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
