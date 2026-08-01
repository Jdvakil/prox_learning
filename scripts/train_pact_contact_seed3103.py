#!/usr/bin/env python3
"""Train and freeze ACT/PACT seed 3103 with the established recipes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
ACT_ROOT = ROOT / "submodules/act"
PYTHON = Path("/root/act_retrain_venv/bin/python")
POLICY_SEED = 3103
ENCODER = Path(
    "/root/pact_frontend_screen_artifacts/encoder_v1/embedding_encoder_frozen.pt"
)
ENCODER_SHA256 = "6fd2dd037e3236b5b6bf7fce8cb2709ead0cf52adcbbe9cbad1061efc2fe3206"
RECIPE = {
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
DATASETS = {
    "ACT": {
        "dataset_dir": ROOT
        / "assets/act_style_data/pact_collision_corridor_v2_full_cba7ff88",
        "split_manifest": ROOT
        / "diagnostics_output/pact_vs_act/full_act_split_encoded_v2.json",
        "dataset_manifest": ROOT
        / "diagnostics_output/pact_vs_act/full_conversion_encoded_v2.json",
        "split_sha256": "04684e802c2de63fd3a070680b51f90665f5e276821b05307897666ae231befc",
        "dataset_tree_sha256": "516472510c65632243c42d121f3f8eb9714cfe2a561e66552c0b78ea88b2065e",
    },
    "PACT": {
        "dataset_dir": Path(
            "/root/pact_frontend_screen_artifacts/dataset_embedding32_v1"
        ),
        "split_manifest": ROOT / "diagnostics_output/pact_vs_act/full_act_split_v2.json",
        "dataset_manifest": Path(
            "/root/pact_frontend_screen_artifacts/manifests/encoded_conversion_v1.json"
        ),
        "split_sha256": "7d25e88445cb4608238f71ddb0ea850ac78041f9d1a5dfdf252f16a27717a486",
        "dataset_tree_sha256": "7a95581dff2907da1720f17425b67244fd20cc934a88a83cb9b66e2ee1d6ce97",
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


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


def validate_dataset(arm: str) -> None:
    record = DATASETS[arm]
    split = json.loads(record["split_manifest"].read_text())
    dataset = json.loads(record["dataset_manifest"].read_text())
    if split["split_manifest_sha256"] != record["split_sha256"]:
        raise ValueError(f"{arm} split changed")
    if dataset["converted_tree_file_sha256"] != record["dataset_tree_sha256"]:
        raise ValueError(f"{arm} dataset changed")


def command_for(arm: str, output_dir: Path) -> list[str]:
    record = DATASETS[arm]
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
        str(record["dataset_dir"]),
        "--split_manifest",
        str(record["split_manifest"]),
        "--dataset_manifest",
        str(record["dataset_manifest"]),
        "--expect_split_sha256",
        record["split_sha256"],
        "--expect_dataset_tree_sha256",
        record["dataset_tree_sha256"],
        "--episode_horizon",
        "195",
        "--state_dim",
        "9",
        "--action_dim",
        "8",
        "--num_workers",
        "4",
        "--ckpt_every",
        "2000",
        "--no_wandb",
    ]
    if arm == "PACT":
        command.extend(
            [
                "--use_proximity",
                "--n_proximity_sensors",
                "40",
                "--prox_tokens_per_sensor",
                "1",
                "--proximity_feature_dim",
                "32",
                "--proximity_encoder_sha256",
                ENCODER_SHA256,
            ]
        )
    return command


def best_validation(output_dir: Path) -> tuple[int, float]:
    rows = [
        json.loads(line)
        for line in (output_dir / "epoch_log.jsonl").read_text().splitlines()
        if line.strip()
    ]
    best = min(rows, key=lambda row: float(row["val"]["loss"]))
    return int(best["epoch"]), float(best["val"]["loss"])


def clean_duplicate_checkpoints(output_dir: Path) -> list[dict[str, Any]]:
    removed = []
    best = output_dir / "policy_best.ckpt"
    for path in sorted(output_dir.glob("*.ckpt")):
        if path == best:
            continue
        removed.append({"name": path.name, "size_bytes": path.stat().st_size})
        path.unlink()
    return removed


def train_arm(arm: str, output_dir: Path) -> dict[str, Any]:
    validate_dataset(arm)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"refusing non-empty training directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    command = command_for(arm, output_dir)
    started = utc_now()
    print(f"[{started}] training {arm}: {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=ACT_ROOT, check=True)
    checkpoint = output_dir / "policy_best.ckpt"
    stats = output_dir / "dataset_stats.pkl"
    run_manifest = output_dir / "run_manifest.json"
    best_epoch, best_loss = best_validation(output_dir)
    checkpoint_sha = sha256_file(checkpoint)
    stats_sha = sha256_file(stats)
    run = json.loads(run_manifest.read_text())
    expected_sensors = 0 if arm == "ACT" else 40
    if int(run["policy_config"]["n_proximity_sensors"]) != expected_sensors:
        raise ValueError(f"{arm} proximity sensor count changed")
    if arm == "PACT" and (
        int(run["policy_config"]["proximity_feature_dim"]) != 32
        or run.get("surface_encoder_sha256") != ENCODER_SHA256
        or run.get("proximity_consumed") is not True
    ):
        raise ValueError("PACT run manifest changed")
    removed = clean_duplicate_checkpoints(output_dir)
    return {
        "arm": arm,
        "seed": POLICY_SEED,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "dataset_stats": str(stats),
        "dataset_stats_sha256": stats_sha,
        "run_manifest": str(run_manifest),
        "run_manifest_sha256": sha256_file(run_manifest),
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "dataset_dir": str(DATASETS[arm]["dataset_dir"]),
        "dataset_manifest": str(DATASETS[arm]["dataset_manifest"]),
        "dataset_tree_sha256": DATASETS[arm]["dataset_tree_sha256"],
        "split_manifest": str(DATASETS[arm]["split_manifest"]),
        "split_manifest_sha256": DATASETS[arm]["split_sha256"],
        "surface_encoder": str(ENCODER) if arm == "PACT" else None,
        "surface_encoder_sha256": ENCODER_SHA256 if arm == "PACT" else None,
        "recipe": RECIPE,
        "command": command,
        "started_utc": started,
        "completed_utc": utc_now(),
        "storage_cleanup": {
            "rule": "retain policy_best.ckpt; remove duplicate epoch/last/resume ckpt files after best SHA-256 is recorded",
            "removed": removed,
        },
    }


def encoder_quality() -> dict[str, Any]:
    if sha256_file(ENCODER) != ENCODER_SHA256:
        raise ValueError("frozen encoder changed")
    payload = torch.load(ENCODER, map_location="cpu")
    metrics = payload["heldout_metrics"]
    keys = (
        "mean_euclidean_error_m",
        "validity_precision",
        "validity_recall",
        "active_pixel_reconstruction_mae",
    )
    if not all(math.isfinite(float(metrics[key])) for key in keys):
        raise ValueError("encoder metrics are non-finite")
    return {
        "passed": bool(
            float(metrics["mean_euclidean_error_m"]) <= 0.05
            and float(metrics["validity_precision"]) >= 0.98
            and float(metrics["validity_recall"]) >= 0.98
            and float(metrics["active_pixel_reconstruction_mae"]) <= 0.15
        ),
        "heldout_metrics": metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--summary-out", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    active = protected_eval_processes()
    if active:
        raise SystemExit(f"protected shared evaluation is active: {active}")
    quality = encoder_quality()
    if not quality["passed"]:
        raise SystemExit("frozen encoder no longer passes its quality gate")
    document = {
        "schema_version": "pact_contact_seed3103_training_v1",
        "seed": POLICY_SEED,
        "recipe": RECIPE,
        "encoder": str(ENCODER),
        "encoder_sha256": ENCODER_SHA256,
        "encoder_quality_gate": quality,
        "arms": {},
    }
    for arm in ("ACT", "PACT"):
        document["arms"][arm] = train_arm(
            arm, args.output_root / f"{arm.lower()}_seed{POLICY_SEED}"
        )
        write_json_atomic(args.summary_out, document)
    document["PACT_ZERO"] = {
        "separately_trained": False,
        "checkpoint_alias": document["arms"]["PACT"]["checkpoint"],
        "label": "OOD sensor-failure probe",
    }
    document["PACT_PERMUTED"] = {
        "separately_trained": False,
        "checkpoint_alias": document["arms"]["PACT"]["checkpoint"],
        "label": "distribution-matched modality-information instrument",
    }
    write_json_atomic(args.summary_out, document)
    print(f"completed ACT and PACT seed {POLICY_SEED}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
