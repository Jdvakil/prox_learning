"""Orchestrate the ablation training runs.

Sequentially trains each ablation YAML on the current GPU. If ``--parallel``
is passed it instead prints the recommended tmux launch commands so a user
with multiple GPUs can copy/paste them.

Subprocess output is teed to ``reports/logs/<ablation>.log`` so a failure
mid-run is recoverable.
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path

DEFAULT_ABLATIONS = [
    "configs/train/ablation_wrist_only.yaml",
    "configs/train/ablation_handcrafted.yaml",
    "configs/train/ablation_conv2d.yaml",
    "configs/train/ablation_cross_attn.yaml",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--configs", nargs="+", default=DEFAULT_ABLATIONS)
    p.add_argument("--parallel", action="store_true",
                   help="print tmux launch commands instead of running.")
    p.add_argument("--extra-args", nargs=argparse.REMAINDER,
                   help="extra args forwarded to pla.train.train")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    extra = args.extra_args or []
    log_dir = Path("reports/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    if args.parallel:
        print("# Recommended tmux launches (one GPU per ablation):")
        for cfg in args.configs:
            session = Path(cfg).stem
            cmd = (
                f"tmux new -d -s {session} "
                f"\"python -m pla.train.train --config {cfg} "
                f"{' '.join(extra)} 2>&1 | tee reports/logs/{session}.log\""
            )
            print(cmd)
        return

    for cfg in args.configs:
        log = log_dir / (Path(cfg).stem + ".log")
        cmd = ["python", "-m", "pla.train.train", "--config", cfg, *extra]
        print(f"$ {' '.join(shlex.quote(c) for c in cmd)}")
        with open(log, "w") as f:
            proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False)
        if proc.returncode != 0:
            print(f"FAILED: {cfg}; see {log}")
            return


if __name__ == "__main__":
    main()
