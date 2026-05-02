"""Gradient norm sanity check.

Day 6 of the timeline (PROJECT.md §6): if grad-norm at the proximity encoder
is zero during PLA training, the encoder is not learning — debug before
proceeding.

Library API
    Use ``grad_norm(model.proximity_encoder)`` to read; use
    ``assert_learning(...)`` to fail loud at training start.

CLI
    ``python -m pla.checks.grad_norm --config configs/train/pla.yaml --steps 50``

    Builds the configured model with ``DummyVLBackbone``, runs ``--steps``
    train iterations on synthetic batches, and asserts every parameter has
    a non-zero grad norm. Exits non-zero on failure so CI can gate on it.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import yaml


def grad_norm(module: nn.Module, *, p: int = 2) -> float:
    """L_p norm of all parameter gradients in ``module``."""
    total = 0.0
    for param in module.parameters():
        if param.grad is None:
            continue
        total += param.grad.detach().data.norm(p).item() ** p
    return total ** (1.0 / p)


def per_param_grad_norms(module: nn.Module, *, p: int = 2) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, param in module.named_parameters():
        if param.grad is None:
            out[name] = float("nan")
        else:
            out[name] = float(param.grad.detach().data.norm(p).item())
    return out


def assert_learning(module: nn.Module, *, eps: float = 1e-8) -> None:
    gn = grad_norm(module)
    if gn <= eps:
        raise AssertionError(
            f"{type(module).__name__} grad norm {gn:.2e} <= {eps:.0e}: "
            "the module is not learning. Check loss path and detach() calls."
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--eps", type=float, default=1e-8)
    return p.parse_args()


def main() -> None:
    """CLI entry point — synthesize a tiny batch and verify learning."""
    from pla.models import DummyVLBackbone
    from pla.train.train import build_model_from_cfg

    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    cfg["dummy_vlm"] = True
    cfg["batch_size"] = min(cfg.get("batch_size", 8), 4)
    vl = DummyVLBackbone(d_model=cfg["d_model"])
    model = build_model_from_cfg(cfg, vl)
    model.train()
    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=cfg.get("lr", 1e-5)
    )

    n = cfg["n_sensors"]
    chunk = cfg["chunk_size"]
    rng = torch.Generator().manual_seed(0)

    last_norms: dict[str, float] = {}
    for step in range(args.steps):
        rgb = torch.rand(2, 2, 3, 224, 224, generator=rng)
        tof = torch.randn(2, n, 8, 8, generator=rng).abs()
        qpos = torch.randn(2, 7, generator=rng)
        actions = torch.randn(2, chunk, 7, generator=rng) * 0.01
        pred, mu, logvar = model(rgb, ["pick"] * 2, tof, qpos, actions)
        total, l1, kl = model.act_decoder.compute_loss(pred, actions, mu, logvar)
        opt.zero_grad()
        total.backward()

        if model.proximity_encoder is not None:
            last_norms = per_param_grad_norms(model.proximity_encoder)
        opt.step()

    if model.proximity_encoder is None:
        print("vlm_only=True: no proximity encoder to check. (skipping)")
        return

    print("Final per-param grad norms (proximity_encoder):")
    bad = []
    for name, gn in last_norms.items():
        flag = "" if gn > args.eps else "  <-- ZERO"
        print(f"  {name:<40s} {gn:.3e}{flag}")
        if gn <= args.eps:
            bad.append(name)
    if bad:
        raise SystemExit(
            f"FAIL: {len(bad)} parameter(s) have grad norm <= {args.eps}: {bad}"
        )
    print("PASS: all parameters have non-zero grad norms.")


if __name__ == "__main__":
    main()
