"""ACT trainer with optional proximity tokens (P+ACT).

Forked from `submodules/act/imitate_episodes.py`. With `--use_proximity` set, the
training loop runs the frozen prox-encoder on a per-sensor temporal window and
feeds the resulting `(B, N_sensors, 3)` 3-D positions into the ACT policy as
`proximity_positions=`. Without `--use_proximity`, the trainer is equivalent to
vanilla `imitate_episodes.py` (regression-tested by the smoke run in
`pact/README.md` §7 step 7).

Differences vs vanilla:
  * Optional `--use_proximity` activates the proximity dataloader + extractor.
  * `policy_config` gains `n_proximity_sensors`, threaded through to DETRVAE.
  * `forward_pass` and the train loop unpack a 5-tuple (image, qpos, action,
    is_pad, prox_window) when proximity is on. Backwards-compat for the 4-tuple.
  * wandb panels: `prox/pred_pos_{x,y,z}_mean`, `prox/finite_frac`, and an
    encoder grad-norm assertion at every step.

Run:
    /opt/conda/envs/mlspaces/bin/python -m pact.act_prox.imitate_episodes_with_prox \
        --task_name pla_house1_mug_random --policy_class ACT \
        --ckpt_dir runs/act_prox_mug_v1 \
        --batch_size 8 --num_epochs 6 --lr 1e-4 --seed 0 \
        --kl_weight 10 --chunk_size 20 --hidden_dim 256 --dim_feedforward 2048 \
        --use_proximity \
        --prox_encoder_ckpt pact/outputs_prox/runs/prox_encoder_v1/ckpt_best.pt \
        --prox_mapping_json act_style_data/mug_house1_random_everything/prox_mapping.json \
        --use_wandb --wandb_project pact --wandb_run_name act_prox_mug_v1
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Make ACT's modules importable.
_ACT_DIR = Path(__file__).resolve().parents[2] / "submodules" / "act"
if str(_ACT_DIR) not in sys.path:
    sys.path.insert(0, str(_ACT_DIR))
# Make pact's modules importable.
_PACT_DIR = Path(__file__).resolve().parents[2]
if str(_PACT_DIR) not in sys.path:
    sys.path.insert(0, str(_PACT_DIR))

from utils import load_data, compute_dict_mean, set_seed, detach_dict  # noqa: E402
from policy import ACTPolicy, CNNMLPPolicy                              # noqa: E402

from pact.act_prox.dataset import make_prox_dataloaders                  # noqa: E402
from pact.act_prox.prox_features import FrozenProxFeatureExtractor       # noqa: E402

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


# ----- policy + optimizer construction --------------------------------------


def make_policy(policy_class: str, policy_config: dict):
    if policy_class == "ACT":
        return ACTPolicy(policy_config)
    if policy_class == "CNNMLP":
        return CNNMLPPolicy(policy_config)
    raise NotImplementedError(policy_class)


# ----- forward + assertions --------------------------------------------------


def _forward_vanilla(batch, policy):
    image_data, qpos_data, action_data, is_pad = batch
    image_data = image_data.cuda(non_blocking=True)
    qpos_data = qpos_data.cuda(non_blocking=True)
    action_data = action_data.cuda(non_blocking=True)
    is_pad = is_pad.cuda(non_blocking=True)
    return policy(qpos_data, image_data, action_data, is_pad)


def _forward_prox(batch, policy, extractor: FrozenProxFeatureExtractor):
    image_data, qpos_data, action_data, is_pad, prox_window = batch
    image_data = image_data.cuda(non_blocking=True)
    qpos_data = qpos_data.cuda(non_blocking=True)
    action_data = action_data.cuda(non_blocking=True)
    is_pad = is_pad.cuda(non_blocking=True)
    prox_window = prox_window.cuda(non_blocking=True)
    with torch.no_grad():
        prox_pos = extractor(prox_window)                              # (B, N, 3)
    loss_dict = policy(qpos_data, image_data, action_data, is_pad,
                       proximity_positions=prox_pos)
    # Attach prox diagnostics for the caller to log.
    loss_dict["_prox_pos"] = prox_pos.detach()
    return loss_dict


def _assert_encoder_frozen(extractor: FrozenProxFeatureExtractor) -> None:
    for n, p in extractor.encoder.named_parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            raise AssertionError(f"prox encoder param {n} received non-zero gradient")
        if p.requires_grad:
            raise AssertionError(f"prox encoder param {n} has requires_grad=True")


# ----- main entry ------------------------------------------------------------


def main(args: dict) -> None:
    set_seed(args["seed"])

    task_name = args["task_name"]
    if task_name[:4] == "sim_":
        from constants import SIM_TASK_CONFIGS                          # noqa: E402
        task_config = SIM_TASK_CONFIGS[task_name]
    else:
        from constants import TASK_CONFIGS                              # noqa: E402
        task_config = TASK_CONFIGS[task_name]

    dataset_dir = task_config["dataset_dir"]
    num_episodes = task_config["num_episodes"]
    camera_names = task_config["camera_names"]

    # Franka / Aloha state-action plumbing.
    if task_name in ("pla_house1_mug", "pla_smoke", "pla_house1_mug_random"):
        state_dim, action_dim = 9, 8
    elif task_name in ("test", "proximity_learning"):
        state_dim = action_dim = 9
    else:
        state_dim = action_dim = 14

    use_proximity = bool(args.get("use_proximity", False))
    n_proximity_sensors = 0
    extractor = None
    if use_proximity:
        if args.get("prox_encoder_ckpt") is None or args.get("prox_mapping_json") is None:
            raise ValueError("--use_proximity requires --prox_encoder_ckpt and --prox_mapping_json")
        with open(args["prox_mapping_json"], "r") as f:
            mapping = json.load(f)
        n_proximity_sensors = int(mapping["n_sensors"])
        extractor = FrozenProxFeatureExtractor(args["prox_encoder_ckpt"], device=torch.device("cuda"))
        print(f"[init] proximity ON: {n_proximity_sensors} sensors, encoder ckpt = {args['prox_encoder_ckpt']}")
    else:
        print("[init] proximity OFF (vanilla ACT)")

    # ACT hyperparameters (match imitate_episodes.py defaults).
    policy_class = args["policy_class"]
    lr_backbone = 1e-5
    backbone = "resnet18"
    if policy_class == "ACT":
        policy_config = {
            "lr": args["lr"],
            "num_queries": args["chunk_size"],
            "kl_weight": args["kl_weight"],
            "hidden_dim": args["hidden_dim"],
            "dim_feedforward": args["dim_feedforward"],
            "lr_backbone": lr_backbone,
            "backbone": backbone,
            "enc_layers": 4,
            "dec_layers": 7,
            "nheads": 8,
            "camera_names": camera_names,
            "state_dim": state_dim,
            "action_dim": action_dim,
            "n_proximity_sensors": n_proximity_sensors,
        }
    elif policy_class == "CNNMLP":
        raise NotImplementedError("CNNMLP + proximity not supported.")
    else:
        raise NotImplementedError(policy_class)

    config = {
        "num_epochs": args["num_epochs"],
        "ckpt_dir": args["ckpt_dir"],
        "state_dim": state_dim,
        "action_dim": action_dim,
        "lr": args["lr"],
        "policy_class": policy_class,
        "policy_config": policy_config,
        "task_name": task_name,
        "seed": args["seed"],
        "camera_names": camera_names,
        "use_wandb": bool(args.get("use_wandb", False)),
        "wandb_project": args.get("wandb_project", "pact"),
        "wandb_run_name": args.get("wandb_run_name"),
        "use_proximity": use_proximity,
        "n_proximity_sensors": n_proximity_sensors,
    }

    # Data.
    if use_proximity:
        train_loader, val_loader, stats, _, _ = make_prox_dataloaders(
            dataset_dir=dataset_dir,
            num_episodes=num_episodes,
            camera_names=camera_names,
            batch_size_train=args["batch_size"],
            batch_size_val=args["batch_size"],
            num_queries=args["chunk_size"],
            prox_mapping_json=args["prox_mapping_json"],
            prox_ckpt_path=args["prox_encoder_ckpt"],
            val_ratio=0.2,
            num_workers=args.get("num_workers", 1),
            seed=args["seed"],
        )
    else:
        train_loader, val_loader, stats, _ = load_data(
            dataset_dir, num_episodes, camera_names,
            args["batch_size"], args["batch_size"], args["chunk_size"],
        )

    os.makedirs(args["ckpt_dir"], exist_ok=True)
    with open(os.path.join(args["ckpt_dir"], "dataset_stats.pkl"), "wb") as f:
        pickle.dump(stats, f)

    train_bc(train_loader, val_loader, config, extractor)


def train_bc(train_loader, val_loader, config, extractor):
    num_epochs = config["num_epochs"]
    ckpt_dir = config["ckpt_dir"]
    seed = config["seed"]
    policy_config = config["policy_config"]
    use_wandb = config["use_wandb"] and _WANDB_AVAILABLE
    use_proximity = config["use_proximity"]

    set_seed(seed)

    if use_wandb:
        wandb.init(
            project=config["wandb_project"],
            name=config["wandb_run_name"],
            dir=ckpt_dir,
            config={
                "task_name": config["task_name"],
                "policy_class": config["policy_class"],
                "num_epochs": num_epochs,
                "seed": seed,
                "state_dim": config["state_dim"],
                "action_dim": config["action_dim"],
                "camera_names": config["camera_names"],
                "use_proximity": use_proximity,
                "n_proximity_sensors": config["n_proximity_sensors"],
                **{k: v for k, v in policy_config.items() if isinstance(v, (int, float, str, list, tuple, bool))},
            },
        )

    policy = make_policy(config["policy_class"], policy_config).cuda()
    optimizer = policy.configure_optimizers()

    min_val_loss = float("inf")
    best_ckpt_info = None
    global_step = 0

    for epoch in tqdm(range(num_epochs), desc="epoch"):
        # ---- validation ------------------------------------------------
        with torch.inference_mode():
            policy.eval()
            val_dicts = []
            for batch in val_loader:
                if use_proximity:
                    fd = _forward_prox(batch, policy, extractor)
                else:
                    fd = _forward_vanilla(batch, policy)
                val_dicts.append({k: v for k, v in fd.items() if not k.startswith("_")})
            val_summary = compute_dict_mean(val_dicts)
            epoch_val_loss = val_summary["loss"].item()
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
        print(f"[epoch {epoch}] val_loss={epoch_val_loss:.5f}  min_val={min_val_loss:.5f}")

        # ---- training --------------------------------------------------
        policy.train()
        train_dicts = []
        prox_pos_sums = torch.zeros(3)
        prox_pos_finite = 0
        prox_pos_count = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            if use_proximity:
                fd = _forward_prox(batch, policy, extractor)
                pp = fd.pop("_prox_pos")                                       # (B, N, 3)
                prox_pos_sums += pp.detach().cpu().sum(dim=(0, 1))
                prox_pos_finite += int(torch.isfinite(pp).sum().item())
                prox_pos_count += int(pp.numel())
            else:
                fd = _forward_vanilla(batch, policy)
            loss = fd["loss"]
            loss.backward()
            if use_proximity:
                _assert_encoder_frozen(extractor)
            optimizer.step()
            train_dicts.append(detach_dict({k: v for k, v in fd.items() if not k.startswith("_")}))
            global_step += 1

        train_summary = compute_dict_mean(train_dicts)
        epoch_train_loss = train_summary["loss"].item()
        print(f"[epoch {epoch}] train_loss={epoch_train_loss:.5f}")

        if use_wandb:
            log = {"epoch": epoch, "min_val_loss": float(min_val_loss),
                   "global_step": global_step}
            for k, v in train_summary.items():
                log[f"train/{k}"] = float(v.item())
            for k, v in val_summary.items():
                log[f"val/{k}"] = float(v.item())
            if use_proximity and prox_pos_count > 0:
                mean_xyz = (prox_pos_sums / max(1, prox_pos_count // 3)).numpy()
                log["prox/pred_pos_x_mean"] = float(mean_xyz[0])
                log["prox/pred_pos_y_mean"] = float(mean_xyz[1])
                log["prox/pred_pos_z_mean"] = float(mean_xyz[2])
                log["prox/finite_frac"] = float(prox_pos_finite / prox_pos_count)
            wandb.log(log, step=epoch)

        # checkpoints
        if epoch % 100 == 0:
            torch.save(policy.state_dict(), os.path.join(ckpt_dir, f"policy_epoch_{epoch}_seed_{seed}.ckpt"))

    torch.save(policy.state_dict(), os.path.join(ckpt_dir, "policy_last.ckpt"))
    if best_ckpt_info is not None:
        best_epoch, _, best_state_dict = best_ckpt_info
        torch.save(best_state_dict, os.path.join(ckpt_dir, "policy_best.ckpt"))
        print(f"[done] best val_loss={min_val_loss:.5f} at epoch {best_epoch}")
    if use_wandb:
        wandb.finish()


# ----- CLI ------------------------------------------------------------------


def _parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--task_name", required=True)
    p.add_argument("--policy_class", required=True)
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--batch_size", type=int, required=True)
    p.add_argument("--num_epochs", type=int, required=True)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--seed", type=int, required=True)
    # ACT-specific
    p.add_argument("--kl_weight", type=int, required=False, default=10)
    p.add_argument("--chunk_size", type=int, required=False, default=20)
    p.add_argument("--hidden_dim", type=int, required=False, default=256)
    p.add_argument("--dim_feedforward", type=int, required=False, default=2048)
    # P+ACT
    p.add_argument("--use_proximity", action="store_true",
                   help="Enable proximity input branch (forked from vanilla ACT).")
    p.add_argument("--prox_encoder_ckpt", type=str, default=None,
                   help="Frozen prox-encoder checkpoint path.")
    p.add_argument("--prox_mapping_json", type=str, default=None,
                   help="Output of pact.act_prox.build_mapping; required when --use_proximity.")
    # wandb
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="pact")
    p.add_argument("--wandb_run_name", type=str, default=None)
    # workers
    p.add_argument("--num_workers", type=int, default=1)
    return vars(p.parse_args(argv))


if __name__ == "__main__":
    main(_parse_args())
