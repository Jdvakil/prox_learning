#!/usr/bin/env python3
"""Train the single PACT policy for the frozen wider-front-end screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
ACT = ROOT / "submodules" / "act"
PYTHON = Path("/root/act_retrain_venv/bin/python")
ACT_CHECKPOINT = Path(
    "/root/pact_remediation_artifacts_v2/full/policies_v2/"
    "act_seed3101/policy_best.ckpt"
)
ACT_CHECKPOINT_SHA256 = (
    "a5ebbf3d5537315337e17e0f28951de068ce6960974d0f282b77fcfcca672eb1"
)
POLICY_SEED = 3101
POLICY_FEATURE_DIM = 32
ENCODER_QUALITY_THRESHOLDS = {
    "mean_euclidean_error_m_max": 0.05,
    "validity_precision_min": 0.98,
    "validity_recall_min": 0.98,
    "active_pixel_reconstruction_mae_max": 0.15,
}


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


def training_command(
    *,
    dataset_dir: Path,
    split_manifest: Path,
    dataset_manifest: Path,
    output_dir: Path,
    split_sha256: str,
    dataset_tree_sha256: str,
    episode_horizon: int,
    encoder_sha256: str,
) -> list[str]:
    return [
        str(PYTHON),
        "imitate_episodes.py",
        "--task_name",
        "obstacle_baseline",
        "--ckpt_dir",
        str(output_dir),
        "--exact_ckpt_dir",
        "--policy_class",
        "ACT",
        "--batch_size",
        "8",
        "--seed",
        str(POLICY_SEED),
        "--num_epochs",
        "2000",
        "--lr",
        "1e-5",
        "--kl_weight",
        "10",
        "--chunk_size",
        "100",
        "--hidden_dim",
        "512",
        "--dim_feedforward",
        "3200",
        "--enc_layers",
        "7",
        "--dec_layers",
        "7",
        "--camera_names",
        "wrist_camera",
        "--dataset_dir",
        str(dataset_dir),
        "--split_manifest",
        str(split_manifest),
        "--dataset_manifest",
        str(dataset_manifest),
        "--expect_split_sha256",
        split_sha256,
        "--expect_dataset_tree_sha256",
        dataset_tree_sha256,
        "--episode_horizon",
        str(episode_horizon),
        "--state_dim",
        "9",
        "--action_dim",
        "8",
        "--num_workers",
        "4",
        "--ckpt_every",
        "2000",
        "--no_wandb",
        "--use_proximity",
        "--n_proximity_sensors",
        "40",
        "--prox_tokens_per_sensor",
        "1",
        "--proximity_feature_dim",
        str(POLICY_FEATURE_DIM),
        "--proximity_encoder_sha256",
        encoder_sha256,
    ]


def best_validation(output_dir: Path) -> tuple[int, float]:
    rows = [
        json.loads(line)
        for line in (output_dir / "epoch_log.jsonl").read_text().splitlines()
        if line.strip()
    ]
    best = min(rows, key=lambda row: float(row["val"]["loss"]))
    return int(best["epoch"]), float(best["val"]["loss"])


def verify_encoder_quality(payload: dict) -> dict:
    metrics = payload.get("heldout_metrics", {})
    numeric_keys = (
        "mean_euclidean_error_m",
        "median_euclidean_error_m",
        "within_2cm_rate",
        "validity_precision",
        "validity_recall",
        "active_pixel_reconstruction_mae",
    )
    finite = all(
        metrics.get(key) is not None
        and math.isfinite(float(metrics[key]))
        for key in numeric_keys
    )
    checks = {
        "all_reported_metrics_finite": finite,
        "mean_euclidean_error_m": (
            finite
            and float(metrics["mean_euclidean_error_m"])
            <= ENCODER_QUALITY_THRESHOLDS[
                "mean_euclidean_error_m_max"
            ]
        ),
        "validity_precision": (
            finite
            and float(metrics["validity_precision"])
            >= ENCODER_QUALITY_THRESHOLDS["validity_precision_min"]
        ),
        "validity_recall": (
            finite
            and float(metrics["validity_recall"])
            >= ENCODER_QUALITY_THRESHOLDS["validity_recall_min"]
        ),
        "active_pixel_reconstruction_mae": (
            finite
            and float(metrics["active_pixel_reconstruction_mae"])
            <= ENCODER_QUALITY_THRESHOLDS[
                "active_pixel_reconstruction_mae_max"
            ]
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": ENCODER_QUALITY_THRESHOLDS,
        "heldout_metrics": metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--encoder", required=True, type=Path)
    parser.add_argument("--summary-out", required=True, type=Path)
    args = parser.parse_args()
    active = protected_eval_processes()
    if active:
        raise SystemExit(
            "protected confirmatory evaluation is active; refusing policy "
            f"training (PIDs {active})"
        )
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"refusing non-empty output directory {args.output_dir}")
    if sha256_file(ACT_CHECKPOINT) != ACT_CHECKPOINT_SHA256:
        raise SystemExit("reused ACT checkpoint hash changed")
    encoder_payload = torch.load(args.encoder, map_location="cpu")
    if (
        encoder_payload.get("schema_version")
        != "pact_surface_embedding_encoder_v1"
        or encoder_payload.get("frozen") is not True
        or encoder_payload.get("policy_feature_dim") != POLICY_FEATURE_DIM
    ):
        raise SystemExit("encoder does not satisfy the frozen screen contract")
    encoder_sha256 = sha256_file(args.encoder)
    encoder_quality_gate = verify_encoder_quality(encoder_payload)
    if not encoder_quality_gate["passed"]:
        raise SystemExit(
            "primary embedding encoder failed the frozen training-clean "
            f"quality gate: {encoder_quality_gate['checks']}"
        )
    split = json.loads(args.split_manifest.read_text())
    dataset = json.loads(args.dataset_manifest.read_text())
    lengths = [int(episode["timesteps"]) for episode in dataset["episodes"]]
    episode_horizon = max(lengths) + 2
    args.output_dir.mkdir(parents=True, exist_ok=True)
    command = training_command(
        dataset_dir=args.dataset_dir,
        split_manifest=args.split_manifest,
        dataset_manifest=args.dataset_manifest,
        output_dir=args.output_dir,
        split_sha256=split["split_manifest_sha256"],
        dataset_tree_sha256=dataset["converted_tree_file_sha256"],
        episode_horizon=episode_horizon,
        encoder_sha256=encoder_sha256,
    )
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=ACT, check=True)
    checkpoint = args.output_dir / "policy_best.ckpt"
    stats = args.output_dir / "dataset_stats.pkl"
    best_epoch, best_loss = best_validation(args.output_dir)
    run_manifest = args.output_dir / "run_manifest.json"
    run = json.loads(run_manifest.read_text())
    if (
        run["policy_config"]["proximity_feature_dim"]
        != POLICY_FEATURE_DIM
        or run["surface_encoder_sha256"] != encoder_sha256
        or run["proximity_consumed"] is not True
    ):
        raise SystemExit("trained policy manifest violates screen contract")
    report = {
        "schema_version": "pact_frontend_screen_policy_training_v1",
        "arm": "PACT",
        "seed": POLICY_SEED,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "dataset_stats": str(stats),
        "dataset_stats_sha256": sha256_file(stats),
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "encoder": str(args.encoder),
        "encoder_sha256": encoder_sha256,
        "encoder_quality_gate": encoder_quality_gate,
        "policy_feature_dim": POLICY_FEATURE_DIM,
        "dataset_dir": str(args.dataset_dir),
        "dataset_manifest": str(args.dataset_manifest),
        "dataset_tree_sha256": dataset["converted_tree_file_sha256"],
        "split_manifest": str(args.split_manifest),
        "split_manifest_sha256": split["split_manifest_sha256"],
        "run_manifest": str(run_manifest),
        "run_manifest_sha256": sha256_file(run_manifest),
        "command": command,
        "recipe": {
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
        },
        "pact_zero": {
            "separately_trained": False,
            "checkpoint_alias": str(checkpoint),
            "zeroed_feature_dim": POLICY_FEATURE_DIM,
        },
        "reused_act": {
            "seed": POLICY_SEED,
            "checkpoint": str(ACT_CHECKPOINT),
            "checkpoint_sha256": ACT_CHECKPOINT_SHA256,
            "retrained": False,
        },
    }
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
