#!/usr/bin/env python3
"""Train, validate, and freeze the 32-D proximity geometry front-end."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from train_pact_surface_encoder import (
    ROOT,
    SurfaceSampleDataset,
    _protected_eval_processes,
    _seed_all,
    sha256_file,
)

ACT = ROOT / "submodules" / "act"
sys.path.insert(0, str(ACT))

from surface_proximity_encoder import (
    MAX_SURFACE_RANGE_M,
    SURFACE_EMBEDDING_DIM,
    SurfaceEmbeddingEncoder,
    parameter_count,
)

RECONSTRUCTION_WEIGHT = 2.0
ACTIVE_PIXEL_WEIGHT = 4.0
XYZ_WEIGHT = 5.0


def losses(
    model: SurfaceEmbeddingEncoder,
    frames: torch.Tensor,
    target_xyz: torch.Tensor,
    target_valid: torch.Tensor,
) -> tuple[
    torch.Tensor,
    dict[str, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]:
    embedding, predicted_xyz, logits, reconstruction = model(frames)
    del embedding
    mask = target_valid.bool()
    validity = F.binary_cross_entropy_with_logits(logits, target_valid)
    xyz = (
        F.smooth_l1_loss(predicted_xyz[mask], target_xyz[mask])
        if mask.any()
        else torch.zeros((), device=frames.device)
    )
    target_closeness = frames[:, -1]
    pixel_weights = 1.0 + ACTIVE_PIXEL_WEIGHT * (
        target_closeness > 0.0
    ).float()
    reconstruction_loss = (
        F.smooth_l1_loss(
            reconstruction,
            target_closeness,
            reduction="none",
        )
        * pixel_weights
    ).mean()
    total = (
        validity
        + XYZ_WEIGHT * xyz
        + RECONSTRUCTION_WEIGHT * reconstruction_loss
    )
    return (
        total,
        {
            "validity_bce": validity,
            "xyz_smooth_l1": xyz,
            "reconstruction_weighted_smooth_l1": reconstruction_loss,
        },
        (predicted_xyz, logits, reconstruction),
    )


@torch.no_grad()
def evaluate(
    model: SurfaceEmbeddingEncoder,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    totals = {
        "loss": 0.0,
        "validity_bce": 0.0,
        "xyz_smooth_l1": 0.0,
        "reconstruction_weighted_smooth_l1": 0.0,
    }
    count = 0
    errors: list[np.ndarray] = []
    reconstruction_absolute: list[np.ndarray] = []
    reconstruction_active_absolute: list[np.ndarray] = []
    tp = fp = fn = tn = 0
    for frames, target_xyz, target_valid in loader:
        frames = frames.to(device, non_blocking=True)
        target_xyz = target_xyz.to(device, non_blocking=True)
        target_valid = target_valid.to(device, non_blocking=True)
        total, components, outputs = losses(
            model, frames, target_xyz, target_valid
        )
        predicted_xyz, logits, reconstruction = outputs
        batch_size = len(frames)
        totals["loss"] += float(total) * batch_size
        for key, value in components.items():
            totals[key] += float(value) * batch_size
        count += batch_size
        mask = target_valid.bool()
        prediction_valid = torch.sigmoid(logits) >= 0.5
        tp += int(torch.sum(prediction_valid & mask))
        fp += int(torch.sum(prediction_valid & ~mask))
        fn += int(torch.sum(~prediction_valid & mask))
        tn += int(torch.sum(~prediction_valid & ~mask))
        if mask.any():
            error = (
                torch.linalg.vector_norm(
                    predicted_xyz[mask] - target_xyz[mask], dim=1
                )
                * MAX_SURFACE_RANGE_M
            )
            errors.append(error.cpu().numpy())
        target_closeness = frames[:, -1]
        absolute = torch.abs(reconstruction - target_closeness)
        reconstruction_absolute.append(absolute.cpu().numpy().reshape(-1))
        active = target_closeness > 0.0
        if active.any():
            reconstruction_active_absolute.append(
                absolute[active].cpu().numpy()
            )
    all_errors = (
        np.concatenate(errors)
        if errors
        else np.asarray([], dtype=np.float32)
    )
    all_reconstruction = np.concatenate(reconstruction_absolute)
    active_reconstruction = (
        np.concatenate(reconstruction_active_absolute)
        if reconstruction_active_absolute
        else np.asarray([], dtype=np.float32)
    )
    metrics: dict[str, Any] = {
        key: value / max(count, 1) for key, value in totals.items()
    }
    metrics.update(
        {
            "sample_count": count,
            "valid_count": int(tp + fn),
            "invalid_count": int(fp + tn),
            "mean_euclidean_error_m": (
                float(np.mean(all_errors)) if len(all_errors) else None
            ),
            "median_euclidean_error_m": (
                float(np.median(all_errors)) if len(all_errors) else None
            ),
            "within_2cm_rate": (
                float(np.mean(all_errors <= 0.02))
                if len(all_errors)
                else None
            ),
            "validity_precision": tp / (tp + fp) if tp + fp else 0.0,
            "validity_recall": tp / (tp + fn) if tp + fn else 0.0,
            "validity_specificity": tn / (tn + fp) if tn + fp else 0.0,
            "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
            "latest_closeness_reconstruction_mae": float(
                np.mean(all_reconstruction)
            ),
            "active_pixel_reconstruction_mae": (
                float(np.mean(active_reconstruction))
                if len(active_reconstruction)
                else None
            ),
        }
    )
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--conversion-manifest", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--report-out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=4201)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    args = parser.parse_args()
    active = _protected_eval_processes()
    if active:
        raise SystemExit(
            "protected confirmatory evaluation is still active; refusing "
            f"embedding encoder training (PIDs {active})"
        )
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"refusing non-empty output directory {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    conversion = json.loads(args.conversion_manifest.read_text())
    split = json.loads(args.split_manifest.read_text())
    if split["canonical_manifest_sha256"] != conversion[
        "source_manifest_sha256"
    ]:
        raise SystemExit("surface split/conversion manifest identity mismatch")
    train_indices = {
        int(episode["act_episode_index"])
        for episode in split["episodes"]
        if episode["split"] == "train"
    }
    validation_indices = {
        int(episode["act_episode_index"])
        for episode in split["episodes"]
        if episode["split"] == "validation"
    }
    if (
        train_indices & validation_indices
        or not train_indices
        or not validation_indices
    ):
        raise SystemExit("encoder requires disjoint nonempty splits")

    train_dataset = SurfaceSampleDataset(
        args.dataset_dir,
        seed=args.seed,
        negative_to_positive_ratio=1.0,
        episode_indices=train_indices,
    )
    validation_dataset = SurfaceSampleDataset(
        args.dataset_dir,
        seed=args.seed + 10,
        negative_to_positive_ratio=None,
        episode_indices=validation_indices,
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    model = SurfaceEmbeddingEncoder().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    best_loss = float("inf")
    best_epoch = None
    best_state = None
    history: list[dict[str, Any]] = []
    for epoch in range(args.epochs):
        model.train()
        total_sum = 0.0
        count = 0
        for frames, target_xyz, target_valid in train_loader:
            frames = frames.to(device, non_blocking=True)
            target_xyz = target_xyz.to(device, non_blocking=True)
            target_valid = target_valid.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            total, _components, _outputs = losses(
                model, frames, target_xyz, target_valid
            )
            if not torch.isfinite(total):
                raise RuntimeError("non-finite embedding encoder loss")
            total.backward()
            optimizer.step()
            total_sum += float(total.detach()) * len(frames)
            count += len(frames)
        validation = evaluate(model, validation_loader, device)
        entry = {
            "epoch": epoch,
            "train_loss": total_sum / max(count, 1),
            "validation": validation,
        }
        history.append(entry)
        print(
            f"epoch={epoch:03d} train={entry['train_loss']:.6f} "
            f"val={validation['loss']:.6f} "
            f"xyz_cm={100 * (validation['mean_euclidean_error_m'] or 0):.3f} "
            f"active_recon={validation['active_pixel_reconstruction_mae']:.4f}",
            flush=True,
        )
        if float(validation["loss"]) < best_loss:
            best_loss = float(validation["loss"])
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    final_metrics = evaluate(model, validation_loader, device)
    checkpoint = {
        "schema_version": "pact_surface_embedding_encoder_v1",
        "frozen": True,
        "variant": "embedding32_reconstruction_auxiliary_surface",
        "policy_feature_dim": SURFACE_EMBEDDING_DIM,
        "model_state_dict": best_state,
        "parameter_count": parameter_count(model),
        "architecture": {
            "input": [32, 8, 8],
            "conv_channels": [1, 32, 32],
            "d_model": 128,
            "layers": 4,
            "heads": 4,
            "feedforward": 256,
            "embedding_dim": SURFACE_EMBEDDING_DIM,
            "policy_output": "32-D per-sensor geometry embedding",
            "auxiliary_output": (
                "sensor-local XYZ normalized by 0.20m plus validity logit"
            ),
            "reconstruction_output": "latest 8x8 closeness map",
        },
        "loss": {
            "validity_bce_weight": 1.0,
            "xyz_smooth_l1_weight": XYZ_WEIGHT,
            "reconstruction_smooth_l1_weight": RECONSTRUCTION_WEIGHT,
            "active_pixel_extra_weight": ACTIVE_PIXEL_WEIGHT,
        },
        "max_surface_range_m": MAX_SURFACE_RANGE_M,
        "sensor_order_sha256": conversion["sensor_order_sha256"],
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "heldout_metrics": final_metrics,
        "conversion_manifest_sha256": sha256_file(
            args.conversion_manifest
        ),
        "split_manifest_sha256": split["split_manifest_sha256"],
    }
    checkpoint_path = args.output_dir / "embedding_encoder_frozen.pt"
    torch.save(checkpoint, checkpoint_path)
    report = {
        key: value
        for key, value in checkpoint.items()
        if key != "model_state_dict"
    }
    report.update(
        {
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "train_samples": len(train_dataset),
            "train_valid_samples": train_dataset.positive_count,
            "train_invalid_available": train_dataset.negative_count_available,
            "validation_samples": len(validation_dataset),
            "baseline_surface_metrics": {
                "mean_euclidean_error_m": 0.03261842578649521,
                "median_euclidean_error_m": 0.018779858946800232,
                "within_2cm_rate": 0.5125677389599907,
                "validity_precision": 0.9990777034816694,
                "validity_recall": 0.9991928974979822,
            },
            "history": history,
        }
    )
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(f"frozen_checkpoint={checkpoint_path}")
    print(f"sha256={report['checkpoint_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
