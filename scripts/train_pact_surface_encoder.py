#!/usr/bin/env python3
"""Train, validate, and freeze the collision-route surface front-end."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
ACT = ROOT / "submodules" / "act"
sys.path.insert(0, str(ACT))

from surface_proximity_encoder import (
    MAX_SURFACE_RANGE_M,
    SurfaceProximityEncoder,
    causal_sensor_window,
    nearest_surface_target,
    parameter_count,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def episode_files(directory: Path) -> list[Path]:
    files = sorted(
        directory.glob("episode_*.hdf5"),
        key=lambda path: int(path.stem.split("_")[1]),
    )
    if not files:
        raise ValueError(f"no episode_*.hdf5 files under {directory}")
    return files


class SurfaceSampleDataset(Dataset):
    """Causal per-sensor samples indexed without loading all frames into RAM."""

    def __init__(
        self,
        directory: Path,
        *,
        seed: int,
        negative_to_positive_ratio: float | None,
        episode_indices: set[int] | None = None,
    ) -> None:
        files = episode_files(directory)
        self.files = (
            files
            if episode_indices is None
            else [
                path
                for path in files
                if int(path.stem.split("_")[1]) in episode_indices
            ]
        )
        if not self.files:
            raise ValueError("the selected surface-encoder split has no episodes")
        self._handles: dict[int, h5py.File] = {}
        positives: list[tuple[int, int, int, bool]] = []
        negatives: list[tuple[int, int, int, bool]] = []
        sensor_order = None
        for file_index, path in enumerate(self.files):
            with h5py.File(path, "r") as handle:
                names = [
                    value.decode() if isinstance(value, bytes) else str(value)
                    for value in handle["observations/proximity_sensor_names"][()]
                ]
                if sensor_order is None:
                    sensor_order = names
                elif names != sensor_order:
                    raise ValueError(f"{path}: sensor order differs")
                proximity = handle["observations/proximity"]
                if proximity.shape[1:] != (40, 4, 8, 8):
                    raise ValueError(f"{path}: bad proximity shape {proximity.shape}")
                # Target is the most recent subframe, per the frozen contract.
                latest = np.asarray(proximity[:, :, -1], dtype=np.float32)
                valid = np.any(
                    np.isfinite(latest)
                    & (latest > 0.0)
                    & (latest <= MAX_SURFACE_RANGE_M),
                    axis=(2, 3),
                )
                for timestep, sensor_index in np.argwhere(valid):
                    positives.append(
                        (file_index, int(timestep), int(sensor_index), True)
                    )
                for timestep, sensor_index in np.argwhere(~valid):
                    negatives.append(
                        (file_index, int(timestep), int(sensor_index), False)
                    )
        self.sensor_names = sensor_order or []
        self.positive_count = len(positives)
        self.negative_count_available = len(negatives)
        if negative_to_positive_ratio is None:
            selected_negatives = negatives
        else:
            requested = min(
                len(negatives),
                math.ceil(len(positives) * negative_to_positive_ratio),
            )
            rng = np.random.default_rng(seed)
            choice = (
                np.sort(rng.choice(len(negatives), size=requested, replace=False))
                if requested
                else np.asarray([], dtype=int)
            )
            selected_negatives = [negatives[int(index)] for index in choice]
        self.samples = positives + selected_negatives
        order_rng = np.random.default_rng(seed + 1)
        order_rng.shuffle(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_handles"] = {}
        return state

    def _handle(self, file_index: int) -> h5py.File:
        handle = self._handles.get(file_index)
        if handle is None:
            handle = h5py.File(self.files[file_index], "r")
            self._handles[file_index] = handle
        return handle

    def __getitem__(self, index: int):
        file_index, timestep, sensor_index, expected_valid = self.samples[index]
        proximity = self._handle(file_index)["observations/proximity"]
        start = max(0, timestep - 7)
        raw = np.asarray(
            proximity[start : timestep + 1, sensor_index], dtype=np.float32
        )
        if len(raw) < 8:
            raw = np.concatenate((np.repeat(raw[:1], 8 - len(raw), axis=0), raw))
        window = causal_sensor_window(
            raw[:, None], timestep=7, sensor_index=0
        ).astype(np.float32)
        point, valid = nearest_surface_target(raw[-1, -1])
        if bool(valid) != bool(expected_valid):
            raise RuntimeError("sample validity changed after index construction")
        return (
            torch.from_numpy(window),
            torch.from_numpy(point / MAX_SURFACE_RANGE_M),
            torch.tensor(float(valid), dtype=torch.float32),
        )


@torch.no_grad()
def evaluate(
    model: SurfaceProximityEncoder,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float | int | None]:
    model.eval()
    loss_sum = 0.0
    count = 0
    errors: list[np.ndarray] = []
    tp = fp = fn = tn = 0
    for frames, target_xyz, target_valid in loader:
        frames = frames.to(device, non_blocking=True)
        target_xyz = target_xyz.to(device, non_blocking=True)
        target_valid = target_valid.to(device, non_blocking=True)
        predicted_xyz, logits = model(frames)
        bce = F.binary_cross_entropy_with_logits(logits, target_valid)
        mask = target_valid.bool()
        xyz_loss = (
            F.smooth_l1_loss(predicted_xyz[mask], target_xyz[mask])
            if mask.any()
            else torch.zeros((), device=device)
        )
        loss = bce + 5.0 * xyz_loss
        batch_size = len(frames)
        loss_sum += float(loss) * batch_size
        count += batch_size
        prediction_valid = torch.sigmoid(logits) >= 0.5
        tp += int(torch.sum(prediction_valid & mask))
        fp += int(torch.sum(prediction_valid & ~mask))
        fn += int(torch.sum(~prediction_valid & mask))
        tn += int(torch.sum(~prediction_valid & ~mask))
        if mask.any():
            error = (
                torch.linalg.vector_norm(predicted_xyz[mask] - target_xyz[mask], dim=1)
                * MAX_SURFACE_RANGE_M
            )
            errors.append(error.cpu().numpy())
    all_errors = np.concatenate(errors) if errors else np.asarray([], dtype=np.float32)
    return {
        "loss": loss_sum / max(count, 1),
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
            float(np.mean(all_errors <= 0.02)) if len(all_errors) else None
        ),
        "validity_precision": tp / (tp + fp) if tp + fp else 0.0,
        "validity_recall": tp / (tp + fn) if tp + fn else 0.0,
        "validity_specificity": tn / (tn + fp) if tn + fp else 0.0,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _protected_eval_processes() -> list[int]:
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--conversion-manifest", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=2201)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    args = parser.parse_args()
    active = _protected_eval_processes()
    if active:
        raise SystemExit(
            "protected confirmatory evaluation is still active; refusing surface "
            f"encoder training (PIDs {active})"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    conversion_manifest = json.loads(args.conversion_manifest.read_text())
    split_manifest = json.loads(args.split_manifest.read_text())
    if (
        split_manifest["canonical_manifest_sha256"]
        != conversion_manifest["source_manifest_sha256"]
    ):
        raise SystemExit("surface split/conversion manifest identity mismatch")
    train_indices = {
        int(episode["act_episode_index"])
        for episode in split_manifest["episodes"]
        if episode["split"] == "train"
    }
    validation_indices = {
        int(episode["act_episode_index"])
        for episode in split_manifest["episodes"]
        if episode["split"] == "validation"
    }
    if train_indices & validation_indices or not train_indices or not validation_indices:
        raise SystemExit("surface encoder requires disjoint nonempty train/validation splits")

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

    model = SurfaceProximityEncoder().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    best_loss = float("inf")
    best_epoch = None
    best_state = None
    history: list[dict[str, Any]] = []
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        count = 0
        for frames, target_xyz, target_valid in train_loader:
            frames = frames.to(device, non_blocking=True)
            target_xyz = target_xyz.to(device, non_blocking=True)
            target_valid = target_valid.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            predicted_xyz, logits = model(frames)
            mask = target_valid.bool()
            bce = F.binary_cross_entropy_with_logits(logits, target_valid)
            xyz_loss = (
                F.smooth_l1_loss(predicted_xyz[mask], target_xyz[mask])
                if mask.any()
                else torch.zeros((), device=device)
            )
            loss = bce + 5.0 * xyz_loss
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(frames)
            count += len(frames)
        validation = evaluate(model, validation_loader, device)
        entry = {
            "epoch": epoch,
            "train_loss": total / max(count, 1),
            "validation": validation,
        }
        history.append(entry)
        print(
            f"epoch={epoch:03d} train={entry['train_loss']:.6f} "
            f"val={validation['loss']:.6f} "
            f"xyz_cm={100 * (validation['mean_euclidean_error_m'] or 0):.3f}"
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
        "schema_version": "pact_surface_encoder_v1",
        "frozen": True,
        "model_state_dict": best_state,
        "parameter_count": parameter_count(model),
        "architecture": {
            "input": [32, 8, 8],
            "conv_channels": [1, 32, 32],
            "d_model": 128,
            "layers": 4,
            "heads": 4,
            "feedforward": 256,
            "output": "sensor-local XYZ normalized by 0.20m plus validity logit",
        },
        "max_surface_range_m": MAX_SURFACE_RANGE_M,
        "sensor_order_sha256": conversion_manifest["sensor_order_sha256"],
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "heldout_metrics": final_metrics,
        "conversion_manifest_sha256": sha256_file(args.conversion_manifest),
        "split_manifest_sha256": split_manifest["split_manifest_sha256"],
    }
    checkpoint_path = args.output_dir / "surface_encoder_frozen.pt"
    torch.save(checkpoint, checkpoint_path)
    report = {
        key: value for key, value in checkpoint.items() if key != "model_state_dict"
    }
    report.update(
        {
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "train_samples": len(train_dataset),
            "train_valid_samples": train_dataset.positive_count,
            "train_invalid_available": train_dataset.negative_count_available,
            "validation_samples": len(validation_dataset),
            "history": history,
        }
    )
    (args.output_dir / "surface_encoder_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(f"frozen_checkpoint={checkpoint_path}")
    print(f"sha256={report['checkpoint_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
