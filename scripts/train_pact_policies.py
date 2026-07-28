#!/usr/bin/env python3
"""Run the frozen ACT/PACT training recipe and record checkpoint provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ACT = ROOT / "submodules" / "act"
PYTHON = Path("/root/act_retrain_venv/bin/python")


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
            command = (entry / "cmdline").read_bytes().replace(b"\x00", b" ")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if b"eval_act_obstacle_on_policy.py" in command:
            matches.append(int(entry.name))
    return matches


def training_command(
    *,
    arm: str,
    seed: int,
    dataset_dir: Path,
    split_manifest: Path,
    dataset_manifest: Path,
    output_dir: Path,
    split_sha256: str,
    dataset_tree_sha256: str,
    episode_horizon: int,
    surface_encoder_sha256: str | None,
) -> list[str]:
    command = [
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
        str(seed),
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
        "100",
        "--no_wandb",
    ]
    if arm == "PACT":
        if not surface_encoder_sha256:
            raise ValueError("PACT requires a frozen surface encoder SHA-256")
        command.extend(
            [
                "--use_proximity",
                "--n_proximity_sensors",
                "40",
                "--prox_tokens_per_sensor",
                "1",
                "--proximity_encoder_sha256",
                surface_encoder_sha256,
            ]
        )
    elif arm != "ACT":
        raise ValueError(arm)
    return command


def _best_validation(output_dir: Path) -> tuple[int | None, float | None]:
    path = output_dir / "epoch_log.jsonl"
    if not path.exists():
        return None, None
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        return None, None
    best = min(rows, key=lambda row: float(row["val"]["loss"]))
    return int(best["epoch"]), float(best["val"]["loss"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pilot", "full"), required=True)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--surface-encoder", type=Path)
    args = parser.parse_args()
    active = protected_eval_processes()
    if active:
        raise SystemExit(
            "protected confirmatory evaluation is still active; refusing GPU training "
            f"(PIDs {active})"
        )
    split = json.loads(args.split_manifest.read_text())
    dataset = json.loads(args.dataset_manifest.read_text())
    split_sha256 = split["split_manifest_sha256"]
    tree_sha256 = dataset["converted_tree_file_sha256"]
    lengths = [int(episode["timesteps"]) for episode in dataset["episodes"]]
    episode_horizon = max(lengths) + 2
    surface_sha256 = (
        sha256_file(args.surface_encoder) if args.surface_encoder else None
    )
    if args.mode == "full" and surface_sha256 is None:
        raise SystemExit("--surface-encoder is required in full mode")
    arms = ("ACT",) if args.mode == "pilot" else ("ACT", "PACT")
    seeds = (1101,) if args.mode == "pilot" else (3101, 3102)
    records = []
    for arm in arms:
        for seed in seeds:
            active = protected_eval_processes()
            if active:
                raise SystemExit(
                    f"protected evaluation became active before {arm} seed {seed}"
                )
            output_dir = args.output_root / f"{arm.lower()}_seed{seed}"
            if output_dir.exists() and any(output_dir.iterdir()):
                raise SystemExit(f"refusing non-empty checkpoint directory {output_dir}")
            output_dir.mkdir(parents=True, exist_ok=True)
            command = training_command(
                arm=arm,
                seed=seed,
                dataset_dir=args.dataset_dir,
                split_manifest=args.split_manifest,
                dataset_manifest=args.dataset_manifest,
                output_dir=output_dir,
                split_sha256=split_sha256,
                dataset_tree_sha256=tree_sha256,
                episode_horizon=episode_horizon,
                surface_encoder_sha256=surface_sha256,
            )
            print(" ".join(command), flush=True)
            subprocess.run(command, cwd=ACT, check=True)
            checkpoint = output_dir / "policy_best.ckpt"
            stats_path = output_dir / "dataset_stats.pkl"
            best_epoch, best_loss = _best_validation(output_dir)
            records.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "dataset_stats": str(stats_path),
                    "dataset_stats_sha256": sha256_file(stats_path),
                    "best_epoch": best_epoch,
                    "best_validation_loss": best_loss,
                    "command": command,
                    "pact_zero_checkpoint_alias": (
                        str(checkpoint) if arm == "PACT" else None
                    ),
                }
            )
    summary = {
        "schema_version": "pact_policy_training_summary_v1",
        "mode": args.mode,
        "dataset_dir": str(args.dataset_dir),
        "dataset_manifest": str(args.dataset_manifest),
        "dataset_tree_sha256": tree_sha256,
        "split_manifest": str(args.split_manifest),
        "split_manifest_sha256": split_sha256,
        "surface_encoder": str(args.surface_encoder) if args.surface_encoder else None,
        "surface_encoder_sha256": surface_sha256,
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
        "records": records,
    }
    (args.output_root / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
