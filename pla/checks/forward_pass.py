"""End-to-end forward-pass sanity test.

Builds a 2-batch ``PLA`` with the dummy backbone, runs forward and backward,
and asserts:

    * pred shape  == [2, chunk_size, action_dim]
    * mu / logvar == [2, z_dim]
    * loss is finite and backward produces non-NaN grads on every trainable
      parameter
    * ``vlm_only=True`` path returns the same shapes with ``model.proximity_encoder is None``

Run::

    python -m pla.checks.forward_pass
    python -m pla.checks.forward_pass --vlm-only
"""
from __future__ import annotations

import argparse

import torch

from pla.models import PLA, DummyVLBackbone


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--vlm-only", action="store_true")
    p.add_argument("--n-sensors", type=int, default=32)
    p.add_argument("--chunk-size", type=int, default=100)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--encoder-type", default="shared_mlp",
                   choices=["shared_mlp", "handcrafted", "conv2d"])
    p.add_argument("--fusion-type", default="concat",
                   choices=["concat", "cross_attn"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(0)
    vl = DummyVLBackbone(d_model=args.d_model)
    model = PLA(
        n_sensors=args.n_sensors,
        d_model=args.d_model,
        chunk_size=args.chunk_size,
        encoder_type=args.encoder_type,
        fusion_type=args.fusion_type,
        vlm_only=args.vlm_only,
        vl_backbone=vl,
    )
    model.train()

    B = 2
    rgb = torch.rand(B, 2, 3, 224, 224)
    tof = torch.randn(B, args.n_sensors, 8, 8).abs()
    qpos = torch.randn(B, 7)
    actions = torch.randn(B, args.chunk_size, 7) * 0.01

    pred, mu, logvar = model(rgb, ["pick"] * B, tof, qpos, actions)
    assert pred.shape == (B, args.chunk_size, 7), pred.shape
    assert mu.shape == (B, model.act_decoder.z_dim), mu.shape
    assert logvar.shape == (B, model.act_decoder.z_dim), logvar.shape

    total, l1, kl = model.act_decoder.compute_loss(pred, actions, mu, logvar)
    assert torch.isfinite(total).item(), "loss is NaN/Inf"
    total.backward()

    nan_params: list[str] = []
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None and not torch.isfinite(p.grad).all():
            nan_params.append(name)
    if nan_params:
        raise SystemExit(f"FAIL: {len(nan_params)} params have non-finite grad: {nan_params}")

    if args.vlm_only:
        assert model.proximity_encoder is None
        print("PASS (vlm_only): forward + backward OK; encoder absent as expected.")
    else:
        assert model.proximity_encoder is not None
        # Inference-mode forward (no actions provided): z=0 path.
        with torch.no_grad():
            pred_inf = model(rgb, ["pick"] * B, tof, qpos, None)
        assert pred_inf.shape == (B, args.chunk_size, 7), pred_inf.shape
        print("PASS: forward + backward + inference OK.")


if __name__ == "__main__":
    main()
