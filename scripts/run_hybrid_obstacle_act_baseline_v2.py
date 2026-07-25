#!/usr/bin/env python3
"""Launcher for the canonical hybrid-obstacle ACT baseline training run.

Reads the frozen contract ``configs/hybrid_obstacle_act_baseline_v2.yaml`` and
builds the exact ``imitate_episodes.py`` command from it, so no hyperparameter is
retyped at the shell and the run cannot silently drift from the committed
configuration.

The launcher never changes a hyperparameter. It fails if the dataset, canonical
manifest or split manifest hashes do not match the contract, because the trained
checkpoint is meaningless if the data underneath it moved.

Usage:
    python scripts/run_hybrid_obstacle_act_baseline_v2.py --ckpt-dir <fresh dir>
    python scripts/run_hybrid_obstacle_act_baseline_v2.py --ckpt-dir <same dir> --resume
    python scripts/run_hybrid_obstacle_act_baseline_v2.py --ckpt-dir <dir> --print-only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "hybrid_obstacle_act_baseline_v2.yaml"
ACT_DIR = ROOT / "submodules" / "act"
PYTHON = "/root/act_retrain_venv/bin/python"


def canonical_hash(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def verify_contract_inputs(cfg: dict) -> dict:
    """Fail loudly if any committed input hash moved."""
    ds = cfg["dataset"]
    checks = {}

    canon = json.loads((ROOT / ds["canonical_manifest"]).read_text())
    stored = canon.pop("manifest_sha256")
    if canonical_hash(canon) != stored or stored != ds["canonical_manifest_sha256"]:
        raise SystemExit("canonical manifest hash does not match the frozen contract")
    checks["canonical_manifest_sha256"] = stored

    split = json.loads((ROOT / ds["split_manifest"]).read_text())
    stored_s = split.pop("split_manifest_sha256")
    if canonical_hash(split) != stored_s or stored_s != ds["split_manifest_sha256"]:
        raise SystemExit("split manifest hash does not match the frozen contract")
    checks["split_manifest_sha256"] = stored_s

    conv = json.loads((ROOT / ds["conversion_manifest"]).read_text())
    if conv["converted_tree_file_sha256"] != ds["converted_tree_sha256"]:
        raise SystemExit("conversion manifest tree hash does not match the frozen contract")
    checks["converted_tree_sha256"] = conv["converted_tree_file_sha256"]

    dataset_dir = ROOT / ds["dataset_dir"]
    if not dataset_dir.is_dir():
        raise SystemExit(f"dataset directory missing: {dataset_dir}")
    n = len(list(dataset_dir.glob("episode_*.hdf5")))
    if n != ds["episodes_total"]:
        raise SystemExit(f"dataset holds {n} episodes, contract says {ds['episodes_total']}")
    checks["episode_files"] = n
    return checks


def build_command(cfg: dict, ckpt_dir: Path, resume: bool, extra: list[str]) -> list[str]:
    ds, pol, tr = cfg["dataset"], cfg["policy"], cfg["training"]
    cmd = [
        PYTHON, "imitate_episodes.py",
        "--task_name", "obstacle_baseline",
        "--ckpt_dir", str(ckpt_dir),
        "--exact_ckpt_dir",
        "--policy_class", pol["policy_class"],
        "--batch_size", str(tr["batch_size"]),
        "--seed", str(tr["seed"]),
        "--num_epochs", str(tr["epochs"]),
        "--lr", repr(float(tr["learning_rate"])),
        "--kl_weight", str(pol["kl_weight"]),
        "--chunk_size", str(pol["chunk_size"]),
        "--hidden_dim", str(pol["hidden_dim"]),
        "--dim_feedforward", str(pol["dim_feedforward"]),
        "--dataset_dir", str(ROOT / ds["dataset_dir"]),
        "--split_manifest", str(ROOT / ds["split_manifest"]),
        "--dataset_manifest", str(ROOT / ds["conversion_manifest"]),
        "--expect_split_sha256", ds["split_manifest_sha256"],
        "--expect_dataset_tree_sha256", ds["converted_tree_sha256"],
        "--episode_horizon", str(pol["episode_horizon"]),
        "--state_dim", str(pol["state_dim"]),
        "--action_dim", str(pol["action_dim"]),
        "--num_workers", "1",
        "--ckpt_every", "100",
        "--no_wandb",
    ]
    if resume:
        cmd.append("--resume")
    return cmd + extra


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt-dir", required=True, type=Path)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--print-only", action="store_true")
    ap.add_argument("extra", nargs="*", default=[])
    args = ap.parse_args()

    cfg = yaml.safe_load(CONTRACT.read_text())
    checks = verify_contract_inputs(cfg)
    print("[contract] frozen inputs verified:")
    for k, v in checks.items():
        print(f"  {k:32} {v}")

    ckpt_dir = args.ckpt_dir.resolve()
    if ckpt_dir.exists() and not args.resume and any(ckpt_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite a non-empty run directory: {ckpt_dir}\n"
                         f"pass --resume to continue it, or choose a fresh directory")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    cmd = build_command(cfg, ckpt_dir, args.resume, list(args.extra))
    # pair each flag with its value so the printed command is readable and
    # copy-pasteable rather than one token per line
    pairs, i = [], 0
    while i < len(cmd):
        if cmd[i].startswith("--") and i + 1 < len(cmd) and not cmd[i + 1].startswith("--"):
            pairs.append(f"{cmd[i]} {cmd[i + 1]}")
            i += 2
        else:
            pairs.append(cmd[i])
            i += 1
    printable = " \\\n    ".join(pairs)
    print(f"\n[command]\ncd {ACT_DIR} && \\\n    {printable}\n")

    # Persist the frozen config and the exact command inside the run directory.
    (ckpt_dir / "frozen_config.yaml").write_bytes(CONTRACT.read_bytes())
    (ckpt_dir / "command.txt").write_text(f"cd {ACT_DIR} && \\\n    {printable}\n")
    (ckpt_dir / "contract_checks.json").write_text(
        json.dumps(checks, indent=2, sort_keys=True) + "\n")

    if args.print_only:
        return 0

    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    return subprocess.run(cmd, cwd=ACT_DIR, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
