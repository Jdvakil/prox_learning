"""Unified training entry point.

Drives PLA, VLM-only ACT, and the encoder ablations from a single YAML
config. The model is constructed by ``build_model_from_cfg`` so the only
difference between PLA and VLM-only is one flag.

Logging
    Each step writes total / L1 / KL loss and the **proximity encoder
    L2 grad norm** (PROJECT.md §6 / TIMELINE.md Day 6 — if it's zero, you
    are silently training a VLM-only model with extra parameters). When W&B
    is unavailable the script falls back to stdout JSONL logging in
    ``runs/<run_name>/log.jsonl`` so the same checks still work offline.

Checkpointing
    Best-val checkpoint is written to ``runs/<run_name>/best.pt`` and a
    rolling ``last.pt`` is updated every epoch. The full config is stamped
    into the checkpoint so eval scripts can reconstruct the model without
    needing the original YAML.

Run::

    python -m pla.train.train --config configs/train/pla.yaml
    python -m pla.train.train --config configs/train/act_baseline.yaml

The two configs differ only in ``vlm_only: true|false`` and ``run_name``.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader

from pla.data.dataset import PLADataset, collate_pla
from pla.models import PLA, DummyVLBackbone, FrozenMolmo2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--dummy-vlm", action="store_true",
                   help="use DummyVLBackbone — for rig + smoke tests only.")
    p.add_argument("--max-steps", type=int, default=None,
                   help="cap training steps (used by smoke test).")
    return p.parse_args()


def build_vl_backbone(cfg: dict, dummy: bool) -> torch.nn.Module:
    if dummy or cfg.get("dummy_vlm", False):
        return DummyVLBackbone(d_model=cfg["d_model"])
    return FrozenMolmo2(
        model_name=cfg.get("vlm_model_name", "allenai/Molmo-4B-D-0924"),
        d_model=cfg["d_model"],
    )


def build_model_from_cfg(cfg: dict, vl_backbone: torch.nn.Module) -> PLA:
    return PLA(
        n_sensors=cfg["n_sensors"],
        d_model=cfg["d_model"],
        chunk_size=cfg["chunk_size"],
        encoder_type=cfg.get("encoder_type", "shared_mlp"),
        fusion_type=cfg.get("fusion_type", "concat"),
        vlm_only=bool(cfg.get("vlm_only", False)),
        vl_backbone=vl_backbone,
        kl_weight=cfg.get("beta_kl", 10.0),
        sensor_mask=cfg.get("sensor_mask"),
    )


class _StdoutLogger:
    """Fallback logger when W&B is not installed / disabled."""

    def __init__(self, run_dir: Path) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        self.f = (run_dir / "log.jsonl").open("a")

    def log(self, payload: dict[str, Any]) -> None:
        self.f.write(json.dumps(payload) + "\n")
        self.f.flush()

    def finish(self) -> None:
        self.f.close()


def _make_logger(cfg: dict, run_dir: Path, no_wandb: bool):
    if no_wandb or cfg.get("no_wandb"):
        return _StdoutLogger(run_dir)
    try:
        import wandb  # type: ignore
    except ImportError:
        print("wandb not installed; falling back to stdout JSONL logger")
        return _StdoutLogger(run_dir)
    wandb.init(
        project=cfg.get("wandb_project", "pla-corl"),
        name=cfg["run_name"],
        config=cfg,
        dir=str(run_dir),
    )
    return wandb


def _proximity_encoder_grad_norm(model: PLA) -> float:
    if model.proximity_encoder is None:
        return float("nan")
    sq = 0.0
    for p in model.proximity_encoder.parameters():
        if p.grad is not None:
            sq += float(p.grad.norm().item() ** 2)
    return sq ** 0.5


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {
        "tof": batch["tof"].to(device),
        "rgb": batch["rgb"].to(device),
        "qpos": batch["qpos"].to(device),
        "actions": batch["actions"].to(device),
        "language": batch["language"],
    }


def train_loop(cfg: dict, *, dummy_vlm: bool, no_wandb: bool, max_steps: int | None) -> None:
    run_dir = Path(cfg.get("output_dir", f"runs/{cfg['run_name']}"))
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config_used.yaml").write_text(yaml.safe_dump(cfg))
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    train_ds = PLADataset(
        data_dir=cfg["data_dir"],
        stats_path=cfg["stats_path"],
        chunk_size=cfg["chunk_size"],
        split="train",
        val_frac=cfg.get("val_frac", 0.1),
        seed=cfg.get("split_seed", 0),
        sensor_mask=cfg.get("sensor_mask"),
    )
    val_ds = PLADataset(
        data_dir=cfg["data_dir"],
        stats_path=cfg["stats_path"],
        chunk_size=cfg["chunk_size"],
        split="val",
        val_frac=cfg.get("val_frac", 0.1),
        seed=cfg.get("split_seed", 0),
        sensor_mask=cfg.get("sensor_mask"),
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg.get("num_workers", 4), pin_memory=True,
        collate_fn=collate_pla,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg.get("num_workers", 4), pin_memory=True,
        collate_fn=collate_pla,
    )

    vl = build_vl_backbone(cfg, dummy=dummy_vlm)
    model = build_model_from_cfg(cfg, vl).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"trainable params: {n_train:,}    frozen: {n_frozen:,}")

    optimizer = optim.Adam(trainable, lr=cfg["lr"])
    logger = _make_logger(cfg, run_dir, no_wandb)
    grad_clip = cfg.get("grad_clip", 1.0)

    best_val = float("inf")
    step = 0
    for epoch in range(cfg["n_epochs"]):
        model.train()
        ep_start = time.time()
        for batch in train_loader:
            batch = _move_batch(batch, device)
            pred, mu, logvar = model(
                rgb=batch["rgb"],
                language=batch["language"],
                tof=batch["tof"],
                qpos=batch["qpos"],
                actions=batch["actions"],
            )
            total, l1, kl = model.act_decoder.compute_loss(
                pred, batch["actions"], mu, logvar
            )
            optimizer.zero_grad()
            total.backward()

            enc_grad = _proximity_encoder_grad_norm(model)
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=grad_clip)
            optimizer.step()

            if step % cfg.get("log_every", 100) == 0:
                payload = {
                    "step": step,
                    "epoch": epoch,
                    "train/loss_total": float(total.item()),
                    "train/loss_l1": float(l1.item()),
                    "train/loss_kl": float(kl.item()),
                    "train/proximity_grad_norm": enc_grad,
                }
                if hasattr(logger, "log"):
                    logger.log(payload)
                else:
                    print(payload)
                if (
                    not cfg.get("vlm_only", False)
                    and enc_grad < cfg.get("min_grad_norm", 1e-6)
                ):
                    print(
                        f"WARN: proximity encoder grad norm {enc_grad:.2e} "
                        f"below threshold; encoder may be disconnected"
                    )

            step += 1
            if max_steps is not None and step >= max_steps:
                break

        if max_steps is not None and step >= max_steps:
            break

        # Validation.
        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for batch in val_loader:
                batch = _move_batch(batch, device)
                pred, mu, logvar = model(
                    rgb=batch["rgb"],
                    language=batch["language"],
                    tof=batch["tof"],
                    qpos=batch["qpos"],
                    actions=batch["actions"],
                )
                total, _, _ = model.act_decoder.compute_loss(
                    pred, batch["actions"], mu, logvar
                )
                val_losses.append(float(total.item()))
        val_loss = sum(val_losses) / max(len(val_losses), 1)
        elapsed = time.time() - ep_start

        payload = {
            "step": step,
            "epoch": epoch,
            "val/loss": val_loss,
            "epoch_seconds": elapsed,
        }
        if hasattr(logger, "log"):
            logger.log(payload)
        else:
            print(payload)

        ckpt = {
            "epoch": epoch,
            "step": step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_loss": val_loss,
            "config": cfg,
        }
        torch.save(ckpt, run_dir / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, run_dir / "best.pt")
            print(f"epoch {epoch}: new best val={val_loss:.4f}")

    if hasattr(logger, "finish"):
        logger.finish()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    train_loop(cfg, dummy_vlm=args.dummy_vlm, no_wandb=args.no_wandb, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
