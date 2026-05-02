"""Training layer.

Top-level exports:
    train_loop, build_model_from_cfg  — training entry points
    act_loss                           — L1 + beta * KL convenience wrapper
"""
from pla.train.losses import act_loss
from pla.train.train import build_model_from_cfg, train_loop

__all__ = ["act_loss", "build_model_from_cfg", "train_loop"]
