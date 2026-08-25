"""Train a frozen surface-geometry encoder from native pact_place skin rows.

Labels are self-supervised: analytic nearest axial XYZ (5 mm–20 cm), validity,
foreground-weighted latest-map reconstruction, occupancy, and next-step pooled
map prediction. Training mixes native 4-subframe windows with ACT's min-pooled
repeat-four adapter. Weight-shared across 40 sensors. Writes a
``pact_surface_*_v1.pt`` that probe / ACT load.

    python -m encoders.train \\
        --src data/pact_place_corridor_v5 \\
        --out experiments_output/default/surface_encoder_train/pact_place_corridor_v5 \\
        --kind embedding --device cuda
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from encoders.peak_closeness import HYBRID_SKIN_SENSOR_ORDER
from encoders.rows import load_episode_proximity, row_dirs
from encoders.surface_geometry import (
    MAX_SURFACE_RANGE_M,
    SurfaceEmbeddingEncoder,
    SurfaceProximityEncoder,
    as_subframe_episode,
    causal_sensor_window,
    depth_to_closeness,
    nearest_surface_target_batch,
    parameter_count,
    save_frozen_checkpoint,
)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _pooled_causal_window(
    episode: np.ndarray,
    timestep: int,
    sensor_index: int,
) -> np.ndarray:
    """ACT-deployment view: min-pool 4 subframes, then repeat each four times."""
    start = max(0, timestep - 7)
    block = episode[start : timestep + 1, sensor_index].min(axis=1)
    if len(block) < 8:
        block = np.concatenate((np.repeat(block[:1], 8 - len(block), axis=0), block))
    repeated = np.repeat(block[:, None], 4, axis=1).reshape(32, 8, 8)
    return depth_to_closeness(repeated)


class SurfaceEmbeddingTrainingModel(torch.nn.Module):
    """Checkpoint-compatible encoder plus training-only next-frame head."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = SurfaceEmbeddingEncoder()
        self.future_head = torch.nn.Sequential(
            torch.nn.Linear(32, 128),
            torch.nn.GELU(),
            torch.nn.Linear(128, 64),
        )

    def forward(
        self, frames: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        embedding, xyz, validity_logit, reconstruction = self.encoder(frames)
        future = torch.sigmoid(self.future_head(embedding)).reshape(-1, 8, 8)
        return embedding, xyz, validity_logit, reconstruction, future


def _checkpoint_model(model: torch.nn.Module, kind: str) -> torch.nn.Module:
    if kind == "embedding":
        if not isinstance(model, SurfaceEmbeddingTrainingModel):
            raise TypeError(type(model))
        return model.encoder
    return model


class SkinWindowDataset(Dataset):
    """One (episode, timestep, sensor) causal window + 20 cm targets."""

    def __init__(
        self,
        episodes: list[np.ndarray],
        index: np.ndarray,
        input_mode: str,
    ) -> None:
        self.episodes = episodes
        self.index = np.asarray(index, dtype=np.int32)
        if input_mode not in ("native", "pooled", "mixed"):
            raise ValueError(input_mode)
        self.input_mode = input_mode

    def __len__(self) -> int:
        return int(self.index.shape[0])

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        ep_i, timestep, sensor_i = (int(x) for x in self.index[i])
        prox = self.episodes[ep_i]
        pooled = self.input_mode == "pooled" or (
            self.input_mode == "mixed" and np.random.random() < 0.5
        )
        if pooled:
            frames = _pooled_causal_window(prox, timestep, sensor_i)
            latest = prox[timestep, sensor_i].min(axis=0)
        else:
            frames = causal_sensor_window(prox, timestep, sensor_i)
            latest = prox[timestep, sensor_i, -1]
        future_timestep = min(timestep + 1, prox.shape[0] - 1)
        future = prox[future_timestep, sensor_i].min(axis=0)
        xyz, valid = nearest_surface_target_batch(latest[None])
        xyz = xyz[0]
        valid_f = np.float32(valid[0])
        return {
            "frames": torch.from_numpy(np.ascontiguousarray(frames)),
            "xyz_norm": torch.from_numpy(xyz / np.float32(MAX_SURFACE_RANGE_M)),
            "valid": torch.tensor(valid_f),
            "latest_close": torch.from_numpy(depth_to_closeness(latest)),
            "future_close": torch.from_numpy(depth_to_closeness(future)),
            "has_future": torch.tensor(
                np.float32(future_timestep > timestep)
            ),
            "sensor": torch.tensor(sensor_i, dtype=torch.int64),
        }


def _build_index(
    episodes: list[np.ndarray],
    stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (N,3) index [ep,t,s], valid mask (N,), sensor ids (N,)."""
    rows = []
    valid_flags = []
    sensors = []
    for ep_i, prox in enumerate(episodes):
        times = np.arange(0, prox.shape[0], max(1, stride), dtype=np.int32)
        # Split/sampler statistics use the deployment representation.
        last = prox[times].min(axis=2)
        _xyz, valid = nearest_surface_target_batch(last)
        n_sensors = int(prox.shape[1])
        for step_i, timestep in enumerate(times):
            for sensor_i in range(n_sensors):
                rows.append((ep_i, int(timestep), sensor_i))
                valid_flags.append(bool(valid[step_i, sensor_i]))
                sensors.append(sensor_i)
    index = np.asarray(rows, dtype=np.int32)
    valid_mask = np.asarray(valid_flags, dtype=bool)
    sensor_ids = np.asarray(sensors, dtype=np.int32)
    return index, valid_mask, sensor_ids


def _sample_weights(
    valid_mask: np.ndarray,
    sensor_ids: np.ndarray,
    n_sensors: int,
    *,
    balance_valid: bool,
    sensor_balance: bool,
) -> np.ndarray | None:
    n = int(valid_mask.shape[0])
    if n == 0 or not (balance_valid or sensor_balance):
        return None
    n_valid = int(valid_mask.sum())
    n_invalid = n - n_valid
    if n_valid == 0 or n_invalid == 0:
        return None

    valid_mass = 0.5 if balance_valid else n_valid / n
    invalid_mass = 1.0 - valid_mass
    weights = np.ones(n, dtype=np.float64)
    weights[~valid_mask] = invalid_mass / n_invalid
    if sensor_balance:
        counts = np.bincount(sensor_ids[valid_mask], minlength=n_sensors).astype(
            np.float64
        )
        counts = np.maximum(counts, 1.0)
        # Square-root balance avoids repeating a handful of rare positives.
        valid_weights = 1.0 / np.sqrt(counts[sensor_ids[valid_mask]])
        weights[valid_mask] = valid_weights * (
            valid_mass / float(valid_weights.sum())
        )
    else:
        weights[valid_mask] = valid_mass / n_valid
    weights *= n / max(float(weights.sum()), 1e-12)
    return weights


def _reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    foreground_weight: float,
    occupancy_weight: float,
    sample_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    foreground = target > 0
    if sample_mask is None:
        mask = torch.ones_like(target)
    else:
        mask = sample_mask[:, None, None].expand_as(target)
    weights = mask * (1.0 + foreground_weight * foreground.float())
    depth_loss = ((prediction - target).square() * weights).sum() / weights.sum().clamp_min(1)
    occupancy_loss = (
        F.binary_cross_entropy(
            prediction.clamp(1e-6, 1.0 - 1e-6),
            foreground.float(),
            reduction="none",
        )
        * mask
    ).sum() / mask.sum().clamp_min(1)
    return (
        depth_loss + occupancy_weight * occupancy_loss,
        depth_loss,
        occupancy_loss,
    )


def _losses(
    model: torch.nn.Module,
    kind: str,
    batch: dict[str, torch.Tensor],
    *,
    pos_weight: torch.Tensor,
    xyz_weight: float,
    recon_weight: float,
    future_weight: float,
    foreground_weight: float,
    occupancy_weight: float,
    valid_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    frames = batch["frames"]
    xyz_norm = batch["xyz_norm"]
    valid = batch["valid"] > 0.5
    if kind == "embedding":
        _embedding, xyz_hat, logit, recon, future = model(frames)
        loss_recon, loss_recon_depth, loss_recon_occupancy = _reconstruction_loss(
            recon,
            batch["latest_close"],
            foreground_weight=foreground_weight,
            occupancy_weight=occupancy_weight,
        )
        loss_future, loss_future_depth, loss_future_occupancy = _reconstruction_loss(
            future,
            batch["future_close"],
            foreground_weight=foreground_weight,
            occupancy_weight=occupancy_weight,
            sample_mask=batch["has_future"],
        )
    else:
        xyz_hat, logit = model(frames)
        loss_recon = frames.new_zeros(())
        loss_recon_depth = frames.new_zeros(())
        loss_recon_occupancy = frames.new_zeros(())
        loss_future = frames.new_zeros(())
        loss_future_depth = frames.new_zeros(())
        loss_future_occupancy = frames.new_zeros(())
    loss_valid = F.binary_cross_entropy_with_logits(
        logit, batch["valid"], pos_weight=pos_weight
    )
    if valid.any():
        loss_xyz = F.mse_loss(xyz_hat[valid], xyz_norm[valid])
    else:
        loss_xyz = xyz_hat.sum() * 0.0
    loss = (
        valid_weight * loss_valid
        + xyz_weight * loss_xyz
        + recon_weight * loss_recon
        + future_weight * loss_future
    )
    with torch.no_grad():
        pred_valid = torch.sigmoid(logit) >= 0.5
        acc = float((pred_valid == valid).float().mean().item())
        stats = {
            "loss": float(loss.item()),
            "loss_valid": float(loss_valid.item()),
            "loss_xyz": float(loss_xyz.item()),
            "loss_recon": float(loss_recon.item()) if kind == "embedding" else 0.0,
            "loss_recon_depth": float(loss_recon_depth.item()),
            "loss_recon_occupancy": float(loss_recon_occupancy.item()),
            "loss_future": float(loss_future.item()),
            "loss_future_depth": float(loss_future_depth.item()),
            "loss_future_occupancy": float(loss_future_occupancy.item()),
            "acc": acc,
        }
        if valid.any():
            err_m = torch.linalg.norm(
                xyz_hat[valid] * MAX_SURFACE_RANGE_M
                - xyz_norm[valid] * MAX_SURFACE_RANGE_M,
                dim=-1,
            )
            stats["xyz_mae_mm"] = float(err_m.mean().item() * 1000.0)
        else:
            stats["xyz_mae_mm"] = float("nan")
    return loss, stats


def _validity_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    truth = np.asarray(labels, dtype=bool)
    pred = np.asarray(probabilities) >= float(threshold)
    tp = int((pred & truth).sum())
    tn = int((~pred & ~truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    return {
        "acc": (tp + tn) / max(len(truth), 1),
        "balanced_acc": 0.5 * (recall + specificity),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
    }


def _best_validity_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[float, dict[str, float]]:
    """Choose validation threshold by balanced accuracy, then F1."""
    candidates = np.unique(
        np.concatenate(
            (
                np.linspace(0.01, 0.99, 99),
                np.quantile(probabilities, np.linspace(0.01, 0.99, 99)),
            )
        )
    )
    best_threshold = 0.5
    best_metrics = _validity_metrics(labels, probabilities, best_threshold)
    for threshold in candidates:
        metrics = _validity_metrics(labels, probabilities, float(threshold))
        score = (metrics["balanced_acc"], metrics["f1"])
        best_score = (best_metrics["balanced_acc"], best_metrics["f1"])
        if score > best_score:
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics


@torch.no_grad()
def _evaluate(
    model: torch.nn.Module,
    kind: str,
    loader: DataLoader,
    device: torch.device,
    *,
    pos_weight: torch.Tensor,
    xyz_weight: float,
    recon_weight: float,
    future_weight: float,
    foreground_weight: float,
    occupancy_weight: float,
    valid_weight: float,
    validity_threshold: float | None = None,
) -> dict[str, float]:
    model.eval()
    totals = {
        "n": 0,
        "valid_bce_sum": 0.0,
        "xyz_sq_sum": 0.0,
        "xyz_elements": 0,
        "xyz_sum": 0.0,
        "xyz_n": 0,
        "gt_valid": 0,
    }
    for prefix in ("recon", "future"):
        totals[f"{prefix}_weighted_sq"] = 0.0
        totals[f"{prefix}_weight"] = 0.0
        totals[f"{prefix}_occupancy_bce"] = 0.0
        totals[f"{prefix}_pixels"] = 0
        totals[f"{prefix}_foreground_abs"] = 0.0
        totals[f"{prefix}_foreground_n"] = 0
        totals[f"{prefix}_pixel_tp"] = 0
        totals[f"{prefix}_pixel_fp"] = 0
        totals[f"{prefix}_pixel_fn"] = 0
    validity_probabilities = []
    validity_labels = []

    def accumulate_reconstruction(
        prefix: str,
        prediction: torch.Tensor,
        target: torch.Tensor,
        sample_mask: torch.Tensor | None,
    ) -> None:
        foreground = target > 0
        if sample_mask is None:
            mask = torch.ones_like(target)
        else:
            mask = sample_mask[:, None, None].expand_as(target)
        weights = mask * (1.0 + foreground_weight * foreground.float())
        totals[f"{prefix}_weighted_sq"] += float(
            ((prediction - target).square() * weights).sum().item()
        )
        totals[f"{prefix}_weight"] += float(weights.sum().item())
        occupancy_bce = F.binary_cross_entropy(
            prediction.clamp(1e-6, 1.0 - 1e-6),
            foreground.float(),
            reduction="none",
        )
        totals[f"{prefix}_occupancy_bce"] += float(
            (occupancy_bce * mask).sum().item()
        )
        totals[f"{prefix}_pixels"] += int(mask.sum().item())
        totals[f"{prefix}_foreground_abs"] += float(
            ((prediction - target).abs() * foreground * mask).sum().item()
        )
        totals[f"{prefix}_foreground_n"] += int((foreground * mask).sum().item())
        predicted_foreground = prediction >= 0.10
        active = mask > 0
        totals[f"{prefix}_pixel_tp"] += int(
            (predicted_foreground & foreground & active).sum().item()
        )
        totals[f"{prefix}_pixel_fp"] += int(
            (predicted_foreground & ~foreground & active).sum().item()
        )
        totals[f"{prefix}_pixel_fn"] += int(
            (~predicted_foreground & foreground & active).sum().item()
        )

    for batch in loader:
        batch = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        bsz = int(batch["valid"].shape[0])
        totals["n"] += bsz
        valid = batch["valid"] > 0.5
        if kind == "embedding":
            _embedding, xyz_hat, logit, recon, future = model(batch["frames"])
            accumulate_reconstruction(
                "recon", recon, batch["latest_close"], sample_mask=None
            )
            accumulate_reconstruction(
                "future",
                future,
                batch["future_close"],
                sample_mask=batch["has_future"],
            )
        else:
            xyz_hat, logit = model(batch["frames"])
        totals["valid_bce_sum"] += float(
            F.binary_cross_entropy_with_logits(
                logit,
                batch["valid"],
                pos_weight=pos_weight,
                reduction="sum",
            ).item()
        )
        validity_probabilities.append(torch.sigmoid(logit).cpu().numpy())
        validity_labels.append(valid.cpu().numpy())
        totals["gt_valid"] += int(valid.sum().item())
        if valid.any():
            difference = xyz_hat[valid] - batch["xyz_norm"][valid]
            totals["xyz_sq_sum"] += float(difference.square().sum().item())
            totals["xyz_elements"] += int(difference.numel())
            err_m = torch.linalg.norm(
                difference * MAX_SURFACE_RANGE_M,
                dim=-1,
            )
            totals["xyz_sum"] += float(err_m.sum().item())
            totals["xyz_n"] += int(valid.sum().item())
    n = max(totals["n"], 1)
    probabilities = np.concatenate(validity_probabilities)
    labels = np.concatenate(validity_labels)
    if validity_threshold is None:
        threshold, validity = _best_validity_threshold(labels, probabilities)
    else:
        threshold = float(validity_threshold)
        validity = _validity_metrics(labels, probabilities, threshold)
    loss_valid = totals["valid_bce_sum"] / n
    loss_xyz = totals["xyz_sq_sum"] / max(totals["xyz_elements"], 1)

    def reconstruction_metrics(prefix: str) -> dict[str, float]:
        depth_loss = totals[f"{prefix}_weighted_sq"] / max(
            totals[f"{prefix}_weight"], 1.0
        )
        occupancy_loss = totals[f"{prefix}_occupancy_bce"] / max(
            totals[f"{prefix}_pixels"], 1
        )
        tp = totals[f"{prefix}_pixel_tp"]
        fp = totals[f"{prefix}_pixel_fp"]
        fn = totals[f"{prefix}_pixel_fn"]
        return {
            f"{prefix}_loss": depth_loss + occupancy_weight * occupancy_loss,
            f"{prefix}_depth_loss": depth_loss,
            f"{prefix}_occupancy_loss": occupancy_loss,
            f"{prefix}_foreground_mae": totals[f"{prefix}_foreground_abs"]
            / max(totals[f"{prefix}_foreground_n"], 1),
            f"{prefix}_pixel_precision": tp / max(tp + fp, 1),
            f"{prefix}_pixel_recall": tp / max(tp + fn, 1),
        }

    recon_metrics = reconstruction_metrics("recon")
    future_metrics = reconstruction_metrics("future")
    loss_recon = recon_metrics["recon_loss"] if kind == "embedding" else 0.0
    loss_future = future_metrics["future_loss"] if kind == "embedding" else 0.0
    loss = (
        valid_weight * loss_valid
        + xyz_weight * loss_xyz
        + recon_weight * loss_recon
        + future_weight * loss_future
    )
    out = {
        "loss": loss,
        "loss_valid": loss_valid,
        "loss_xyz": loss_xyz,
        "loss_recon": loss_recon,
        "loss_future": loss_future,
        **recon_metrics,
        **future_metrics,
        **validity,
        "validity_threshold": threshold,
        "acc_at_0_5": _validity_metrics(labels, probabilities, 0.5)["acc"],
        "gt_valid_frac": totals["gt_valid"] / n,
        "pred_valid_frac": float((probabilities >= threshold).mean()),
        "xyz_mae_mm": (
            1000.0 * totals["xyz_sum"] / totals["xyz_n"]
            if totals["xyz_n"]
            else float("nan")
        ),
        "xyz_endpoint_error_mm": (
            1000.0 * totals["xyz_sum"] / totals["xyz_n"]
            if totals["xyz_n"]
            else float("nan")
        ),
        "always_invalid_acc": 1.0 - totals["gt_valid"] / n,
    }
    return out


def _plot_history(history: list[dict], path: Path) -> None:
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="val")
    axes[0].set_title("loss")
    axes[0].legend()
    axes[1].plot(epochs, [row["val_acc"] for row in history], label="val acc")
    axes[1].plot(
        epochs,
        [row["val_balanced_acc"] for row in history],
        label="balanced acc",
    )
    axes[1].plot(
        epochs,
        [row["val_always_invalid_acc"] for row in history],
        label="always-invalid",
        linestyle="--",
    )
    axes[1].set_title("validity accuracy")
    axes[1].legend()
    axes[2].plot(epochs, [row["val_xyz_mae_mm"] for row in history])
    axes[2].set_title("val XYZ MAE (mm), GT-valid")
    for ax in axes:
        ax.set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def load_episodes(
    src: Path,
    sensor_order: list[str],
    max_episodes: int | None,
) -> tuple[list[Path], list[np.ndarray]]:
    try:
        rows = row_dirs(src)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    if max_episodes is not None:
        rows = rows[: int(max_episodes)]
    episodes = []
    for row in rows:
        prox = as_subframe_episode(load_episode_proximity(row, sensor_order))
        episodes.append(prox)
        print(f"loaded {row.name}  {tuple(prox.shape)}", flush=True)
    return rows, episodes


def _episode_split(
    rows: list[Path],
    *,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    """Episode split stratified by left/right intrusion when data is large enough."""
    rng = np.random.default_rng(seed)
    if len(rows) < 10:
        order = rng.permutation(len(rows))
        n_val = max(1, int(round(len(rows) * val_frac)))
        n_test = max(1, int(round(len(rows) * test_frac)))
        if n_val + n_test >= len(rows):
            n_val = n_test = 1
        test_ids = [int(i) for i in order[:n_test]]
        val_ids = [int(i) for i in order[n_test : n_test + n_val]]
    else:
        groups: dict[str, list[int]] = {}
        for row_id, row in enumerate(rows):
            result_path = row / "result.json"
            side = "unknown"
            if result_path.is_file():
                result = json.loads(result_path.read_text())
                side = str(
                    (result.get("scene_params") or {}).get(
                        "pact_intrusion_side", "unknown"
                    )
                )
            groups.setdefault(side, []).append(row_id)
        val_ids = []
        test_ids = []
        for ids in groups.values():
            shuffled = rng.permutation(ids)
            n_val = max(1, int(round(len(ids) * val_frac)))
            n_test = max(1, int(round(len(ids) * test_frac)))
            if n_val + n_test >= len(ids):
                raise SystemExit(
                    f"not enough episodes in split stratum of size {len(ids)}"
                )
            test_ids.extend(int(i) for i in shuffled[:n_test])
            val_ids.extend(int(i) for i in shuffled[n_test : n_test + n_val])
    held_out = set(val_ids) | set(test_ids)
    train_ids = [i for i in range(len(rows)) if i not in held_out]
    return sorted(train_ids), sorted(val_ids), sorted(test_ids)


def train(
    src: Path,
    out_dir: Path,
    *,
    kind: str,
    device: str,
    epochs: int,
    batch_size: int,
    stride: int,
    val_frac: float,
    test_frac: float,
    seed: int,
    lr: float,
    xyz_weight: float,
    recon_weight: float,
    future_weight: float,
    foreground_weight: float,
    occupancy_weight: float,
    valid_weight: float,
    train_input_mode: str,
    min_balanced_acc: float,
    min_recall: float,
    balance_valid: bool,
    sensor_balance: bool,
    max_episodes: int | None,
    num_workers: int,
) -> dict:
    _set_seed(seed)
    sensor_order = list(HYBRID_SKIN_SENSOR_ORDER)
    n_sensors = len(sensor_order)
    rows, episodes = load_episodes(src, sensor_order, max_episodes)
    if len(episodes) < 3:
        raise SystemExit("need at least 3 episodes for train/val/test splits")
    if val_frac <= 0 or test_frac <= 0 or val_frac + test_frac >= 1:
        raise SystemExit("--val-frac and --test-frac must be >0 and sum to <1")

    train_ids, val_ids_list, test_ids_list = _episode_split(
        rows, val_frac=val_frac, test_frac=test_frac, seed=seed
    )

    train_eps = [episodes[i] for i in train_ids]
    val_eps = [episodes[i] for i in val_ids_list]
    test_eps = [episodes[i] for i in test_ids_list]
    train_index, train_valid, train_sensors = _build_index(train_eps, stride)
    val_index, val_valid, _val_sensors = _build_index(val_eps, stride)
    test_index, test_valid, _test_sensors = _build_index(test_eps, stride)

    n_train_valid = int(train_valid.sum())
    n_val_valid = int(val_valid.sum())
    n_test_valid = int(test_valid.sum())
    pos_rate = n_train_valid / max(len(train_index), 1)
    # Sampling controls class exposure. BCE stays unweighted; validation on
    # the natural prior supplies the deployment threshold.
    pos_weight_value = 1.0

    weights = _sample_weights(
        train_valid,
        train_sensors,
        n_sensors,
        balance_valid=balance_valid,
        sensor_balance=sensor_balance,
    )
    train_ds = SkinWindowDataset(train_eps, train_index, input_mode=train_input_mode)
    # Selection and final test match converted/live ACT: min-pool each native
    # 4-subframe control step, then repeat it four times.
    val_ds = SkinWindowDataset(val_eps, val_index, input_mode="pooled")
    test_ds = SkinWindowDataset(test_eps, test_index, input_mode="pooled")
    sampler = None
    shuffle = True
    if weights is not None and (balance_valid or sensor_balance):
        sampler = WeightedRandomSampler(
            torch.from_numpy(weights).double(),
            num_samples=len(train_ds),
            replacement=True,
        )
        shuffle = False
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
    )

    torch_device = torch.device(device)
    model: torch.nn.Module = (
        SurfaceEmbeddingTrainingModel()
        if kind == "embedding"
        else SurfaceProximityEncoder()
    )
    model.to(torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    pos_weight = torch.tensor(pos_weight_value, device=torch_device)

    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = (
        "pact_surface_embedding_encoder_v1.pt"
        if kind == "embedding"
        else "pact_surface_encoder_v1.pt"
    )
    best_path = out_dir / ckpt_name
    last_path = out_dir / "last.pt"
    best_score = (-float("inf"), -float("inf"))
    history: list[dict] = []

    config = {
        "src": str(src.resolve()),
        "kind": kind,
        "epochs": epochs,
        "batch_size": batch_size,
        "stride": stride,
        "val_frac": val_frac,
        "test_frac": test_frac,
        "split_stratification": "pact_intrusion_side",
        "seed": seed,
        "lr": lr,
        "xyz_weight": xyz_weight,
        "recon_weight": recon_weight if kind == "embedding" else 0.0,
        "future_weight": future_weight if kind == "embedding" else 0.0,
        "foreground_weight": foreground_weight,
        "occupancy_weight": occupancy_weight,
        "valid_weight": valid_weight,
        "train_input_mode": train_input_mode,
        "validation_input_mode": "pooled_min_repeat4",
        "deployment_pool": "min",
        "min_balanced_acc": min_balanced_acc,
        "min_recall": min_recall,
        "balance_valid": balance_valid,
        "sensor_balance": sensor_balance,
        "n_train_episodes": len(train_ids),
        "n_val_episodes": len(val_ids_list),
        "n_test_episodes": len(test_ids_list),
        "n_train_windows": int(len(train_index)),
        "n_val_windows": int(len(val_index)),
        "n_test_windows": int(len(test_index)),
        "n_train_valid": n_train_valid,
        "n_val_valid": n_val_valid,
        "n_test_valid": n_test_valid,
        "train_valid_frac": float(pos_rate),
        "pos_weight": pos_weight_value,
        "parameter_count": parameter_count(_checkpoint_model(model, kind)),
        "training_parameter_count": parameter_count(model),
        "train_rows": [rows[i].name for i in train_ids],
        "val_rows": [rows[i].name for i in val_ids_list],
        "test_rows": [rows[i].name for i in test_ids_list],
    }
    config["split_sha256"] = hashlib.sha256(
        json.dumps(
            {
                "train": config["train_rows"],
                "val": config["val_rows"],
                "test": config["test_rows"],
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    split_manifest = {
        "schema_version": "pact_surface_episode_split_v1",
        "source": config["src"],
        "seed": seed,
        "sha256": config["split_sha256"],
        "train_rows": config["train_rows"],
        "val_rows": config["val_rows"],
        "test_rows": config["test_rows"],
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (out_dir / "split_manifest.json").write_text(
        json.dumps(split_manifest, indent=2) + "\n"
    )
    print(
        f"train windows={len(train_index)} valid={100 * pos_rate:.2f}%  "
        f"val windows={len(val_index)} valid={100 * n_val_valid / max(len(val_index), 1):.2f}%  "
        f"test windows={len(test_index)} valid={100 * n_test_valid / max(len(test_index), 1):.2f}%  "
        f"params={config['parameter_count']}",
        flush=True,
    )

    for epoch in range(1, epochs + 1):
        model.train()
        running = {"loss": 0.0, "n": 0}
        for batch in train_loader:
            batch = {
                key: value.to(torch_device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            optimizer.zero_grad(set_to_none=True)
            loss, stats = _losses(
                model,
                kind,
                batch,
                pos_weight=pos_weight,
                xyz_weight=xyz_weight,
                recon_weight=recon_weight,
                future_weight=future_weight,
                foreground_weight=foreground_weight,
                occupancy_weight=occupancy_weight,
                valid_weight=valid_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            bsz = int(batch["valid"].shape[0])
            running["loss"] += stats["loss"] * bsz
            running["n"] += bsz
        scheduler.step()
        train_loss = running["loss"] / max(running["n"], 1)
        val_stats = _evaluate(
            model,
            kind,
            val_loader,
            torch_device,
            pos_weight=pos_weight,
            xyz_weight=xyz_weight,
            recon_weight=recon_weight,
            future_weight=future_weight,
            foreground_weight=foreground_weight,
            occupancy_weight=occupancy_weight,
            valid_weight=valid_weight,
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_stats["loss"],
            "val_acc": val_stats["acc"],
            "val_balanced_acc": val_stats["balanced_acc"],
            "val_precision": val_stats["precision"],
            "val_recall": val_stats["recall"],
            "val_validity_threshold": val_stats["validity_threshold"],
            "val_always_invalid_acc": val_stats["always_invalid_acc"],
            "val_xyz_mae_mm": val_stats["xyz_mae_mm"],
            "val_recon": val_stats["loss_recon"],
            "val_future": val_stats["loss_future"],
            "val_recon_pixel_recall": val_stats["recon_pixel_recall"],
            "val_future_pixel_recall": val_stats["future_pixel_recall"],
            "lr": float(scheduler.get_last_lr()[0]),
        }
        history.append(row)
        print(
            f"epoch {epoch:03d}/{epochs}  train_loss={train_loss:.4f}  "
            f"val_loss={val_stats['loss']:.4f}  val_acc={100 * val_stats['acc']:.1f}%  "
            f"bal_acc={100 * val_stats['balanced_acc']:.1f}%  "
            f"P/R={100 * val_stats['precision']:.1f}/{100 * val_stats['recall']:.1f}%  "
            f"thr={val_stats['validity_threshold']:.3f}  "
            f"(empty-base {100 * val_stats['always_invalid_acc']:.1f}%)  "
            f"xyz_err={val_stats['xyz_endpoint_error_mm']:.1f}mm  "
            f"recon={val_stats['loss_recon']:.4f}  "
            f"future={val_stats['loss_future']:.4f}",
            flush=True,
        )
        extra = {
            "metrics": val_stats,
            "epoch": epoch,
            "best_epoch": epoch,
            "heldout_metrics": val_stats,
            "config": config,
            "validity_threshold": val_stats["validity_threshold"],
            "split_sha256": config["split_sha256"],
            "training_input_mode": train_input_mode,
            "deployment_input_mode": "pooled_min_repeat4",
            "validity_score_calibrated": False,
        }
        if kind == "embedding":
            assert isinstance(model, SurfaceEmbeddingTrainingModel)
            extra["training_future_head_state_dict"] = (
                model.future_head.state_dict()
            )
        save_frozen_checkpoint(
            last_path, _checkpoint_model(model, kind), kind, extra
        )
        passes_gate = (
            val_stats["balanced_acc"] >= min_balanced_acc
            and val_stats["recall"] >= min_recall
        )
        # Before validity works, rank by balanced accuracy. Once it passes the
        # gate, optimize the full XYZ/reconstruction/future validation loss.
        score = (
            int(passes_gate),
            -val_stats["loss"] if passes_gate else val_stats["balanced_acc"],
        )
        if score > best_score:
            best_score = score
            save_frozen_checkpoint(
                best_path, _checkpoint_model(model, kind), kind, extra
            )
            print(
                f"  saved best -> {best_path}  "
                f"bal_acc={100 * val_stats['balanced_acc']:.2f}%  "
                f"val_loss={val_stats['loss']:.4f}",
                flush=True,
            )

    (out_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    if history:
        _plot_history(history, out_dir / "curves.png")
    if not best_path.is_file():
        save_frozen_checkpoint(
            best_path,
            _checkpoint_model(model, kind),
            kind,
            {"metrics": history[-1] if history else {}},
        )
    best_payload = torch.load(best_path, map_location=torch_device)
    _checkpoint_model(model, kind).load_state_dict(
        best_payload["model_state_dict"]
    )
    if kind == "embedding":
        assert isinstance(model, SurfaceEmbeddingTrainingModel)
        model.future_head.load_state_dict(
            best_payload["training_future_head_state_dict"]
        )
    test_stats = _evaluate(
        model,
        kind,
        test_loader,
        torch_device,
        pos_weight=pos_weight,
        xyz_weight=xyz_weight,
        recon_weight=recon_weight,
        future_weight=future_weight,
        foreground_weight=foreground_weight,
        occupancy_weight=occupancy_weight,
        valid_weight=valid_weight,
        validity_threshold=float(best_payload.get("validity_threshold", 0.5)),
    )
    best_payload["test_metrics"] = test_stats
    torch.save(best_payload, best_path)
    (out_dir / "test_metrics.json").write_text(
        json.dumps(test_stats, indent=2) + "\n"
    )
    print(
        "held-out TEST  "
        f"acc={100 * test_stats['acc']:.1f}%  "
        f"bal_acc={100 * test_stats['balanced_acc']:.1f}%  "
        f"P/R={100 * test_stats['precision']:.1f}/{100 * test_stats['recall']:.1f}%  "
        f"xyz_err={test_stats['xyz_endpoint_error_mm']:.1f}mm  "
        f"recon/future={test_stats['loss_recon']:.4f}/{test_stats['loss_future']:.4f}",
        flush=True,
    )
    print(f"best checkpoint: {best_path}")
    return {
        "best": str(best_path),
        "last": str(last_path),
        "best_validation_metrics": best_payload.get("heldout_metrics"),
        "test_metrics": test_stats,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=Path("data/pact_place_corridor_v5"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "experiments_output/default/surface_encoder_train/pact_place_corridor_v5"
        ),
    )
    parser.add_argument("--kind", choices=("xyz", "embedding"), default="embedding")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--xyz-weight", type=float, default=5.0)
    parser.add_argument("--recon-weight", type=float, default=1.0)
    parser.add_argument("--future-weight", type=float, default=0.5)
    parser.add_argument("--foreground-weight", type=float, default=10.0)
    parser.add_argument("--occupancy-weight", type=float, default=0.1)
    parser.add_argument("--valid-weight", type=float, default=1.0)
    parser.add_argument(
        "--train-input-mode",
        choices=("native", "pooled", "mixed"),
        default="mixed",
        help="Mixed trains on native subframes and ACT min-pooled/repeated windows.",
    )
    parser.add_argument("--min-balanced-acc", type=float, default=0.95)
    parser.add_argument("--min-recall", type=float, default=0.90)
    parser.add_argument(
        "--no-balance-valid",
        action="store_true",
        help="Disable default 50/50 valid/empty sampling.",
    )
    parser.add_argument(
        "--sensor-balance",
        action="store_true",
        help="Give each sensor equal mass among valid samples (off by default).",
    )
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    train(
        args.src,
        args.out,
        kind=args.kind,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        stride=args.stride,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
        lr=args.lr,
        xyz_weight=args.xyz_weight,
        recon_weight=args.recon_weight,
        future_weight=args.future_weight,
        foreground_weight=args.foreground_weight,
        occupancy_weight=args.occupancy_weight,
        valid_weight=args.valid_weight,
        train_input_mode=args.train_input_mode,
        min_balanced_acc=args.min_balanced_acc,
        min_recall=args.min_recall,
        balance_valid=not args.no_balance_valid,
        sensor_balance=args.sensor_balance,
        max_episodes=args.max_episodes,
        num_workers=args.num_workers,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
