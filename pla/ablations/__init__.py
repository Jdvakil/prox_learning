"""Ablations.

Each ablation is a YAML in ``configs/train/ablation_*.yaml`` that flips a
single switch from the PLA config. ``run_ablations.py`` launches them in
sequence (or in parallel via tmux).

The ablations:

  ablation_wrist_only      — sensor_mask zeros all non-wrist sensors.
  ablation_handcrafted     — encoder_type=handcrafted.
  ablation_conv2d          — encoder_type=conv2d.
  ablation_cross_attn      — fusion_type=cross_attn.

Sensor importance is post-hoc and uses ``pla.eval.sensor_importance`` (no
retraining required).
"""
