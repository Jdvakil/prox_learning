"""Training and evaluation engine for the parked-skin reference.

Checkpoints are published atomically and carry a complete resume bundle (model, optimizer,
scheduler and every RNG stream). A half-written checkpoint that still loads is worse than
one that fails loudly, because a later resume would silently continue from a corrupted
state.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .data import SOURCE_MODES, Partition, StratifiedBatchSampler
from .losses import LossWeights, compute_losses
from .metrics import (
    average_precision,
    binary_scores,
    constraint_violations,
    direction_cosine,
    direction_cosine_included,
    rms,
    safe_norm,
)
from .model import (
    BASELINE_FULL,
    BASELINE_ZERO,
    build_model,
    zero_differential,
)


@dataclass
class TrainConfig:
    variant: str = BASELINE_FULL
    hidden: int = 192
    blocks: int = 2
    dropout: float = 0.0
    batch_size: int = 64
    active_fraction: float = 0.5
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    max_epochs: int = 100
    patience: int = 12
    batches_per_epoch: int = 300
    grad_clip: float = 1.0
    seed: int = 0
    weights: LossWeights = None

    def __post_init__(self) -> None:
        if self.weights is None:
            self.weights = LossWeights()

    def as_dict(self) -> dict:
        payload = {k: v for k, v in asdict(self).items() if k != "weights"}
        payload["weights"] = self.weights.as_dict()
        return payload

    def config_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.as_dict(), sort_keys=True,
                                         separators=(",", ":")).encode()).hexdigest()


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_batch(partition: Partition, index: np.ndarray, device: str,
               *, targets: bool = True) -> dict:
    """Assemble one batch. Deployable inputs and targets are kept in separate keys."""
    history_index = np.asarray(partition["history"])[index]
    current = np.asarray(partition["current"])
    valid = np.asarray(partition["current_valid"])
    batch = {
        "history": torch.from_numpy(np.ascontiguousarray(current[history_index])),
        "history_valid": torch.from_numpy(np.ascontiguousarray(valid[history_index])),
        "state": torch.from_numpy(np.ascontiguousarray(
            np.asarray(partition["state"])[index])),
        "current_valid": torch.from_numpy(np.ascontiguousarray(valid[index])),
    }
    if targets:
        batch.update({
            "parked": torch.from_numpy(np.ascontiguousarray(
                np.asarray(partition["parked"])[index])),
            "parked_valid": torch.from_numpy(np.ascontiguousarray(
                np.asarray(partition["parked_valid"])[index])),
            "changed": torch.from_numpy(np.ascontiguousarray(
                np.asarray(partition["changed"])[index])),
            "oracle_dq": torch.from_numpy(np.ascontiguousarray(
                np.asarray(partition["oracle_dq"])[index])),
            "oracle_active": torch.from_numpy(np.ascontiguousarray(
                np.asarray(partition["oracle_active"])[index])),
        })
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def previous_window(partition: Partition, index: np.ndarray) -> np.ndarray:
    """Frame index of t-1 inside the same trajectory, clamped at the trajectory start."""
    return np.asarray(partition["history"])[index][:, -2]


def forward(model, batch: dict, variant: str) -> dict:
    if variant == BASELINE_ZERO:
        return zero_differential(batch["history"])
    return model(batch["history"], batch["history_valid"], batch["state"])


def atomic_save(payload: dict, path: Path) -> str:
    """Write, fsync, then rename. A reader never sees a partial checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with open(temporary, "wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rng_bundle() -> dict:
    return {"torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "numpy": np.random.get_state()}


def restore_rng(bundle: dict) -> None:
    torch.set_rng_state(bundle["torch"])
    if bundle.get("cuda") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(bundle["cuda"])
    np.random.set_state(bundle["numpy"])


# --------------------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(model, partition: Partition, head, device: str, variant: str,
             *, batch_size: int = 256, threshold: float | None = None,
             collect_fields: bool = True) -> dict:
    """Full-partition evaluation on the natural, unmodified distribution.

    Never subsamples and never reweights: the sampler's balancing is a training-time
    device, and applying it here would report a prevalence the deployed system never sees.
    """
    if model is not None:
        model.eval()
    total = len(partition)
    order = np.arange(total)

    predicted_dq_all = np.zeros((total, 7), dtype=np.float64)
    current_dq_all = np.zeros((total, 7), dtype=np.float64)
    field_abs_error_sum = 0.0
    field_valid_count = 0
    changed_abs_error_sum = 0.0
    changed_count = 0
    delta_abs_error_sum = 0.0
    delta_count = 0
    violations = {"parked_above_current": 0, "parked_below_zero": 0,
                  "parked_above_one": 0, "total": 0}
    nonfinite = 0
    mask_scores: list[np.ndarray] = []
    mask_labels: list[np.ndarray] = []
    mask_predicted: list[np.ndarray] = []
    validity_agreement_sum = 0.0
    validity_count = 0
    per_sensor_error = np.zeros(40, dtype=np.float64)
    per_sensor_count = np.zeros(40, dtype=np.float64)

    for start in range(0, total, batch_size):
        index = order[start:start + batch_size]
        batch = make_batch(partition, index, device)
        output = forward(model, batch, variant)
        parked = output["parked"]
        current = output["current"]

        if not torch.isfinite(parked).all():
            nonfinite += int((~torch.isfinite(parked)).sum())

        current_dq = head(current)
        predicted_dq = current_dq - head(parked)
        predicted_dq_all[index] = predicted_dq.double().cpu().numpy()
        current_dq_all[index] = current_dq.double().cpu().numpy()

        valid = (batch["parked_valid"] & batch["current_valid"])
        error = (parked - batch["parked"]).abs()
        field_abs_error_sum += float((error * valid).sum())
        field_valid_count += int(valid.sum())

        changed = batch["changed"] & valid
        changed_abs_error_sum += float((error * changed).sum())
        changed_count += int(changed.sum())

        delta_true = current - batch["parked"]
        delta_error = (output["delta"] - delta_true).abs()
        delta_abs_error_sum += float((delta_error * valid).sum())
        delta_count += int(valid.sum())

        counted = constraint_violations(parked.double().cpu().numpy(),
                                        current.double().cpu().numpy())
        for key in violations:
            violations[key] += counted[key]

        # per-sensor absolute field error, valid pixels only
        sensor_error = (error * valid).sum(dim=(0, 2, 3)).double().cpu().numpy()
        sensor_count = valid.sum(dim=(0, 2, 3)).double().cpu().numpy()
        per_sensor_error += sensor_error
        per_sensor_count += sensor_count

        if collect_fields:
            probability = output["changed_probability"]
            mask_scores.append(probability[valid].float().cpu().numpy())
            mask_labels.append(batch["changed"][valid].cpu().numpy())
            mask_predicted.append((probability[valid] > 0.5).cpu().numpy())

        # Validity-mask agreement. A pixel is invalid only when its depth falls below
        # 5 mm, i.e. closeness above 0.99. The constraint gives 0 <= parked <= current,
        # and no stored closeness reaches 0.99, so the model can never turn a live pixel
        # dead: its implied parked validity is exactly the current validity. The earlier
        # definition here compared parked_valid against "the model left this pixel
        # unsaturated", which is a different quantity and reported ~0.12 on masks that
        # are in fact identical.
        predicted_valid = batch["current_valid"]
        validity_agreement_sum += float(
            (predicted_valid == batch["parked_valid"]).sum())
        validity_count += int(predicted_valid.numel())

    oracle_dq = np.asarray(partition["oracle_dq"], dtype=np.float64)
    active = np.asarray(partition["oracle_active"]).astype(bool)
    hazard = np.asarray(partition["hazard_present"]).astype(bool)
    modes = np.asarray(partition["source_mode"])

    dq_error = predicted_dq_all - oracle_dq
    dq_abs = np.abs(dq_error)
    predicted_norm = safe_norm(predicted_dq_all)
    true_norm = safe_norm(oracle_dq)

    cosine = direction_cosine(predicted_dq_all, oracle_dq)
    included = active & direction_cosine_included(oracle_dq)

    result = {
        "partition": partition.name,
        "frames": total,
        "variant": variant,
        "nonfinite_outputs": nonfinite,
        "constraint_violations": violations,
        "pixel": {
            "all_valid_parked_mae": field_abs_error_sum / max(field_valid_count, 1),
            "changed_pixel_parked_mae": (changed_abs_error_sum / changed_count
                                         if changed_count else None),
            "differential_mae": delta_abs_error_sum / max(delta_count, 1),
            "valid_pixels": field_valid_count,
            "changed_pixels": changed_count,
            "changed_pixel_prevalence": (changed_count / field_valid_count
                                         if field_valid_count else None),
            "validity_mask_agreement": validity_agreement_sum / max(validity_count, 1),
        },
        "head": {
            "differential_mae": float(dq_abs.mean()),
            "per_joint_mae": [float(v) for v in dq_abs.mean(axis=0)],
            "arm_vector_norm_error": float(np.abs(predicted_norm - true_norm).mean()),
            "active_frame_rms": rms(dq_error[active]) if active.any() else None,
            "oracle_zero_rms": rms(dq_error[~active]) if (~active).any() else None,
            "hazard_absent_rms": rms(dq_error[~hazard]) if (~hazard).any() else None,
            "hazard_absent_raw_head_rms": rms(current_dq_all[~hazard])
            if (~hazard).any() else None,
            "hazard_present_mae": float(dq_abs[hazard].mean()) if hazard.any() else None,
            "hazard_absent_mae": float(dq_abs[~hazard].mean())
            if (~hazard).any() else None,
            "median_direction_cosine_active": float(np.median(cosine[included]))
            if included.any() else None,
            "direction_cosine_frames": int(included.sum()),
            "norm_correlation": float(np.corrcoef(predicted_norm, true_norm)[0, 1])
            if total > 1 and predicted_norm.std() > 0 and true_norm.std() > 0 else None,
        },
        "predicted_norm": predicted_norm,
        "oracle_active": active,
        "direction_cosine": cosine,
        "direction_cosine_included": included,
    }

    if threshold is not None:
        # strictly greater: the calibrated threshold is the 99th percentile of the
        # oracle-zero norms, and ZERO_DIFFERENTIAL puts every norm at exactly 0.0, so
        # ">=" would score it as firing on 100% of frames instead of 0%
        fired = predicted_norm > threshold
        result["activation"] = {
            "threshold": float(threshold),
            "oracle_active_recall": float(fired[active].mean()) if active.any() else None,
            "oracle_zero_false_positive_rate": float(fired[~active].mean())
            if (~active).any() else None,
            "hazard_absent_false_positive_rate": float(fired[~hazard].mean())
            if (~hazard).any() else None,
        }

    if collect_fields and mask_scores:
        scores = np.concatenate(mask_scores)
        labels = np.concatenate(mask_labels)
        predicted_mask = np.concatenate(mask_predicted)
        prevalence = float(labels.mean())
        auprc = average_precision(scores, labels)
        result["mask"] = {
            **binary_scores(predicted_mask, labels),
            "auprc": auprc,
            "prevalence": prevalence,
            "prevalence_normalized_auprc": (auprc / prevalence
                                            if prevalence > 0 else None),
        }

    # per source mode and per sensor / link
    result["by_source_mode"] = {}
    for index, name in enumerate(SOURCE_MODES):
        hits = modes == index
        if not hits.any():
            result["by_source_mode"][name] = {"frames": 0, "available": False}
            continue
        mode_included = included & hits
        result["by_source_mode"][name] = {
            "frames": int(hits.sum()),
            "available": True,
            "differential_mae": float(dq_abs[hits].mean()),
            "active_frames": int((active & hits).sum()),
            "median_direction_cosine_active": float(np.median(cosine[mode_included]))
            if mode_included.any() else None,
            "oracle_zero_rms": rms(dq_error[hits & ~active])
            if (hits & ~active).any() else None,
        }

    result["per_sensor_mae"] = [
        float(e / c) if c else None
        for e, c in zip(per_sensor_error, per_sensor_count)]
    return result


def strip_arrays(result: dict) -> dict:
    """Drop the per-frame arrays so the result can be serialised."""
    return {k: v for k, v in result.items()
            if k not in ("predicted_norm", "oracle_active", "direction_cosine",
                         "direction_cosine_included")}


# --------------------------------------------------------------------------- training
def train(config: TrainConfig, train_partition: Partition,
          validation_partition: Partition, head, device: str,
          checkpoint_dir: Path, *, log_every: int = 25,
          progress: bool = True) -> dict:
    """Bounded training with validation-only checkpoint selection."""
    from .data import sensor_link_ids

    seed_everything(config.seed)
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    link_ids, links = sensor_link_ids(SENSOR_NAMES)
    model = build_model(config.variant, hidden=config.hidden, blocks=config.blocks,
                        link_ids=torch.from_numpy(link_ids), link_count=len(links),
                        dropout=config.dropout).to(device)
    parameters = model.parameter_count()

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.max_epochs)
    sampler = StratifiedBatchSampler(
        train_partition, config.batch_size, active_fraction=config.active_fraction,
        seed=config.seed, batches_per_epoch=config.batches_per_epoch)

    history = []
    best = {"epoch": -1, "metric": float("inf")}
    since_improvement = 0
    started = time.time()

    for epoch in range(config.max_epochs):
        model.train()
        epoch_parts: dict[str, float] = {}
        for step, index in enumerate(sampler.epoch(epoch)):
            batch = make_batch(train_partition, index, device)
            if config.weights.temporal_smoothness > 0.0:
                previous = previous_window(train_partition, index)
                previous_batch = make_batch(train_partition, previous, device)
                with torch.no_grad():
                    previous_output = forward(model, previous_batch, config.variant)
                batch["previous_parked"] = previous_output["parked"]
                batch["previous_oracle_dq"] = previous_batch["oracle_dq"]
            output = forward(model, batch, config.variant)
            terms = compute_losses(output, batch, head, config.weights)
            optimizer.zero_grad(set_to_none=True)
            terms.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            for key, value in terms.parts.items():
                epoch_parts[key] = epoch_parts.get(key, 0.0) + value
        scheduler.step()
        epoch_parts = {k: v / max(sampler.batches_per_epoch, 1)
                       for k, v in epoch_parts.items()}

        scored = evaluate(model, validation_partition, head, device, config.variant,
                          collect_fields=False)
        # selection metric: the quantity the controller consumes, on the natural
        # validation distribution. Never offline test.
        metric = scored["head"]["differential_mae"]
        history.append({"epoch": epoch, "train": epoch_parts,
                        "validation_head_mae": metric,
                        "validation_active_rms": scored["head"]["active_frame_rms"],
                        "validation_zero_rms": scored["head"]["oracle_zero_rms"],
                        "lr": float(scheduler.get_last_lr()[0])})

        improved = metric < best["metric"] - 1e-9
        if improved:
            best = {"epoch": epoch, "metric": metric}
            since_improvement = 0
            best_hash = atomic_save({
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "epoch": epoch,
                "config": config.as_dict(), "config_hash": config.config_hash(),
                "validation_metric": metric, "rng": rng_bundle(),
                "parameter_count": parameters,
            }, checkpoint_dir / "best.pt")
            best["sha256"] = best_hash
        else:
            since_improvement += 1

        if progress and (epoch % log_every == 0 or improved or
                         since_improvement >= config.patience):
            print(f"    epoch {epoch:>3} val_head_mae={metric:.6f} "
                  f"{'*' if improved else ' '} "
                  f"active_rms={scored['head']['active_frame_rms']:.4f} "
                  f"zero_rms={scored['head']['oracle_zero_rms']:.4f} "
                  f"total={epoch_parts.get('total', 0):.4f}", flush=True)

        if since_improvement >= config.patience:
            break

    last_hash = atomic_save({
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "epoch": len(history) - 1,
        "config": config.as_dict(), "config_hash": config.config_hash(),
        "validation_metric": history[-1]["validation_head_mae"], "rng": rng_bundle(),
        "parameter_count": parameters,
    }, checkpoint_dir / "last.pt")

    return {
        "config": config.as_dict(),
        "config_hash": config.config_hash(),
        "parameter_count": parameters,
        "epochs_run": len(history),
        "best_epoch": best["epoch"],
        "best_validation_head_mae": best["metric"],
        "best_checkpoint": str(checkpoint_dir / "best.pt"),
        "best_checkpoint_sha256": best.get("sha256"),
        "last_checkpoint": str(checkpoint_dir / "last.pt"),
        "last_checkpoint_sha256": last_hash,
        "history": history,
        "sampler": sampler.prevalence_report(),
        "wall_seconds": time.time() - started,
        "early_stopped": since_improvement >= config.patience,
    }


def load_checkpoint(path: Path, device: str):
    """Rebuild a model from a checkpoint exactly as trained."""
    from .data import sensor_link_ids

    payload = torch.load(path, map_location=device, weights_only=False)
    config = payload["config"]
    link_ids, links = sensor_link_ids(SENSOR_NAMES)
    model = build_model(config["variant"], hidden=config["hidden"],
                        blocks=config["blocks"],
                        link_ids=torch.from_numpy(link_ids), link_count=len(links),
                        dropout=config.get("dropout", 0.0)).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, payload


SENSOR_NAMES: list[str] = []


def set_sensor_names(names) -> None:
    """Pin the frozen sensor order once, at process start."""
    global SENSOR_NAMES
    SENSOR_NAMES = list(names)
