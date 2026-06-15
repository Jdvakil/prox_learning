"""Visualize how ACT's decoder cross-attention distributes over the 29
proximity tokens vs the other memory tokens (latent / proprio / image).

Loads the trained P+ACT checkpoint and the frozen prox-encoder, runs a few
hundred batches drawn from ProxAugmentedEpisodicDataset, and records the
per-layer (B, L_query, S_memory) cross-attention weights via a forward
hook on each decoder layer's `multihead_attn` module.

Memory token layout (matches submodules/act/detr/models/transformer.py
when n_proximity_sensors=29):

    index 0       : latent token
    index 1       : proprio token
    indices 2..30 : 29 proximity tokens (one per sensor, in `sensor_names`
                    order from prox_mapping.json)
    indices 31..  : image feature tokens (ResNet18 backbone output, two
                    cameras concatenated along the width dim)

Outputs (all PNGs under --out_dir):
    per_sensor_attention.png        - mean attention received by each sensor
    group_attention.png             - latent vs proprio vs prox vs image
    per_layer_per_sensor_heatmap.png- 7×29 heatmap (decoder layers × sensors)
    temporal_per_sensor.png         - sensor attention as a function of
                                      progress through the episode
    raw_stats.json                  - all the raw numbers behind the plots

Usage:
    /opt/conda/envs/mlspaces/bin/python pact/analysis/visualize_prox_attention.py \
        --ckpt_dir runs/act_prox_mug_v1 \
        --prox_encoder_ckpt pact/outputs_prox/runs/prox_encoder_v1/ckpt_best.pt \
        --prox_mapping_json act_style_data/mug_house1_random_everything/prox_mapping.json \
        --dataset_dir act_style_data/mug_house1_random_everything \
        --out_dir pact/analysis/attention_outputs
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Make pact + ACT importable.
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "submodules" / "act"))

from pact.act_prox.prox_features import FrozenProxFeatureExtractor    # noqa: E402

CAMERAS = ["exo_camera_1", "wrist_camera"]
STATE_DIM, ACTION_DIM = 9, 8


@contextmanager
def _detr_argv(ckpt_dir: str, seed: int, n_proximity_sensors: int):
    """Hide this script's CLI flags from detr/main.py's nested argparse."""
    orig = sys.argv
    sys.argv = [
        orig[0] if orig else "visualize_prox_attention.py",
        "--ckpt_dir", ckpt_dir,
        "--policy_class", "ACT",
        "--task_name", "pla_house1_mug_random",
        "--seed", str(seed),
        "--num_epochs", "1",
        "--n_proximity_sensors", str(n_proximity_sensors),
    ]
    try:
        yield
    finally:
        sys.argv = orig


# ---------- model loading ---------------------------------------------------


def build_policy(ckpt_dir: str, hidden_dim: int, dim_feedforward: int,
                 chunk_size: int, n_sensors: int, seed: int):
    config = dict(
        lr=1e-4,
        num_queries=chunk_size,
        kl_weight=10,
        hidden_dim=hidden_dim,
        dim_feedforward=dim_feedforward,
        lr_backbone=1e-5,
        backbone="resnet18",
        enc_layers=4,
        dec_layers=7,
        nheads=8,
        camera_names=CAMERAS,
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        n_proximity_sensors=n_sensors,
    )
    with _detr_argv(ckpt_dir, seed, n_sensors):
        from policy import ACTPolicy  # noqa: WPS433 - inside argv shim
        policy = ACTPolicy(config)
    state = torch.load(Path(ckpt_dir) / "policy_best.ckpt", map_location="cuda",
                       weights_only=False)
    policy.load_state_dict(state)
    policy = policy.cuda().eval()
    return policy, config


# ---------- forward hook to grab cross-attention weights --------------------


class CrossAttnRecorder:
    """One forward-pass call into the decoder pushes one (B, L, S) weight
    tensor onto `self.captured[layer_idx]` for each decoder layer."""

    def __init__(self, policy):
        self.captured: Dict[int, List[torch.Tensor]] = defaultdict(list)
        self._handles: List = []
        for i, layer in enumerate(policy.model.transformer.decoder.layers):
            self._handles.append(
                layer.multihead_attn.register_forward_hook(self._make_hook(i))
            )

    def _make_hook(self, layer_idx: int):
        def hook(_module, _inputs, output):
            # output = (attn_output, attn_output_weights[B, L, S]). PyTorch
            # default averages weights across heads, which is exactly what
            # we want for the spatial summary plot.
            w = output[1]
            if w is not None:
                self.captured[layer_idx].append(w.detach().cpu())
        return hook

    def clear(self) -> None:
        self.captured.clear()

    def close(self) -> None:
        for h in self._handles:
            h.remove()


# ---------- batch sampling --------------------------------------------------


def _load_norm_stats(ckpt_dir: str) -> Dict[str, np.ndarray]:
    with open(Path(ckpt_dir) / "dataset_stats.pkl", "rb") as f:
        return pickle.load(f)


def _open_prox_h5(mapping: dict, episode_idx: int) -> Tuple[h5py.File, str]:
    entry = mapping["episodes"][str(int(episode_idx))]
    return h5py.File(entry["source_h5"], "r"), entry["traj_key"]


def _load_proximity_window(traj, sensor_names, start_ts: int, W: int,
                           prox_mean: np.ndarray, prox_std: np.ndarray
                           ) -> np.ndarray:
    lo = max(0, start_ts - W + 1)
    hi = start_ts + 1
    n_real = hi - lo
    n_pad = W - n_real
    sensor_windows = np.empty((len(sensor_names), W, 4, 8, 8), dtype=np.float32)
    for s_idx, sn in enumerate(sensor_names):
        full = traj[f"obs/proximity/{sn}"][lo:hi]
        if n_pad > 0:
            pad = np.repeat(full[:1], n_pad, axis=0)
            full = np.concatenate([pad, full], axis=0)
        sensor_windows[s_idx] = full
    sensor_windows = (sensor_windows - prox_mean[None, None]) / prox_std[None, None]
    return sensor_windows.reshape(len(sensor_names), W * 4, 8, 8)


def _load_act_sample(dataset_dir: str, episode_idx: int, start_ts: int,
                     cameras, qpos_mean, qpos_std):
    path = Path(dataset_dir) / f"episode_{episode_idx}.hdf5"
    with h5py.File(path, "r") as root:
        qpos = root["/observations/qpos"][start_ts].astype(np.float32)
        episode_len = root["/action"].shape[0]
        images = []
        for cam in cameras:
            images.append(root[f"/observations/images/{cam}"][start_ts])
    image = torch.from_numpy(np.stack(images, axis=0)).float().permute(0, 3, 1, 2) / 255.0
    qpos = (qpos - qpos_mean) / qpos_std
    return image, torch.from_numpy(qpos).float(), int(episode_len)


def _build_batch(
    dataset_dir: str,
    mapping: dict,
    norm_stats: dict,
    prox_mean: np.ndarray,
    prox_std: np.ndarray,
    window: int,
    episode_ids: List[int],
    start_fracs: List[float] | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Tuple[int, int]]]:
    """Return (image (B,2,3,H,W), qpos (B,9), prox (B,29,W*4,8,8), info)
    for the requested episode_ids. start_fracs (per episode) selects a
    fraction-of-episode timestep deterministically; if None, picks the
    middle of the episode."""
    images, qposes, proxes, info = [], [], [], []
    sensor_names = list(mapping["sensor_names"])
    for k, ep in enumerate(episode_ids):
        h5, traj_key = _open_prox_h5(mapping, ep)
        try:
            traj = h5[traj_key]
            T_src = traj[f"obs/proximity/{sensor_names[0]}"].shape[0]
            # ACT episode length and source-h5 length differ slightly because
            # ACT trims; clip to whatever is shortest.
            img, qpos, T_act = _load_act_sample(
                dataset_dir, ep, 0, CAMERAS,
                norm_stats["qpos_mean"], norm_stats["qpos_std"],
            )
            T = min(T_src, T_act)
            frac = (start_fracs[k] if start_fracs is not None else 0.5)
            t = int(min(max(0, frac * (T - 1)), T - 1))
            img, qpos, _ = _load_act_sample(
                dataset_dir, ep, t, CAMERAS,
                norm_stats["qpos_mean"], norm_stats["qpos_std"],
            )
            prox = _load_proximity_window(traj, sensor_names, t, window,
                                          prox_mean, prox_std)
            images.append(img)
            qposes.append(qpos)
            proxes.append(torch.from_numpy(prox))
            info.append((ep, t))
        finally:
            h5.close()
    return (
        torch.stack(images, dim=0),
        torch.stack(qposes, dim=0),
        torch.stack(proxes, dim=0),
        info,
    )


# ---------- analysis --------------------------------------------------------


def _stack_layer_weights(captured: Dict[int, List[torch.Tensor]]
                         ) -> torch.Tensor:
    """captured[i] is a list of (B, L, S) tensors collected over batches.
    Returns (n_layers, n_batches*B, L, S)."""
    n_layers = max(captured.keys()) + 1
    per_layer = []
    for i in range(n_layers):
        per_layer.append(torch.cat(captured[i], dim=0))  # (sum B, L, S)
    return torch.stack(per_layer, dim=0)                  # (n_layers, B*, L, S)


def _summarise(weights: torch.Tensor, n_sensors: int
               ) -> Dict[str, np.ndarray]:
    """weights: (n_layers, B, L, S)."""
    n_layers, B, L, S = weights.shape
    # mean over batch & queries — per (layer, src_token)
    per_layer_per_token = weights.mean(dim=(1, 2))         # (n_layers, S)
    per_token = weights.mean(dim=(0, 1, 2))                # (S,)

    prox_slice = slice(2, 2 + n_sensors)
    per_sensor = per_token[prox_slice].numpy()             # (n_sensors,)
    per_layer_per_sensor = per_layer_per_token[:, prox_slice].numpy()

    groups = {
        "latent":  float(per_token[0].item()),
        "proprio": float(per_token[1].item()),
        "prox":    float(per_token[prox_slice].sum().item()),
        "image":   float(per_token[2 + n_sensors:].sum().item()),
    }
    # Sanity: should sum to ~1.
    return {
        "per_sensor": per_sensor,
        "per_layer_per_sensor": per_layer_per_sensor,
        "groups": groups,
        "total_attention_check": float(per_token.sum().item()),
        "n_memory_tokens": int(S),
        "n_action_queries": int(L),
        "n_layers": int(n_layers),
        "n_samples": int(B),
    }


def _link_name(sensor_name: str) -> str:
    return sensor_name.split("_")[0]


# ---------- plotting --------------------------------------------------------


def plot_per_sensor(per_sensor: np.ndarray, sensor_names: List[str],
                    out: Path) -> None:
    n = len(sensor_names)
    links = [_link_name(s) for s in sensor_names]
    palette = {"link2": "#a6cee3", "link3": "#1f78b4",
               "link5": "#b2df8a", "link6": "#33a02c"}
    colors = [palette.get(l, "#999999") for l in links]

    fig, ax = plt.subplots(figsize=(10.5, 4.4), dpi=140)
    x = np.arange(n)
    bars = ax.bar(x, per_sensor, color=colors, edgecolor="black", linewidth=0.4)
    uniform = 1.0 / max(1, per_sensor.shape[0])
    ax.axhline(uniform, color="red", linestyle="--", linewidth=0.9,
               label=f"uniform 1/{n} = {uniform:.4f}")
    ax.set_xticks(x)
    ax.set_xticklabels(sensor_names, rotation=70, ha="right", fontsize=7.5)
    ax.set_ylabel("Mean cross-attention weight  (queries × layers × batch)")
    ax.set_title("P+ACT decoder cross-attention — per-sensor attention received")
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=c, ec="black")
                      for c in palette.values()]
    ax.legend(legend_handles + [plt.Line2D([0], [0], color="red", ls="--")],
              list(palette.keys()) + ["uniform"], loc="upper left",
              fontsize=9, ncol=5)
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_groups(groups: Dict[str, float], n_memory_tokens: int,
                n_sensors: int, out: Path) -> None:
    image_tokens = n_memory_tokens - 2 - n_sensors
    sizes = {"latent": 1, "proprio": 1, "prox": n_sensors, "image": image_tokens}
    keys = ["latent", "proprio", "prox", "image"]
    totals = [groups[k] for k in keys]
    per_token = [groups[k] / sizes[k] for k in keys]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=140)

    ax = axes[0]
    bars = ax.bar(keys, totals, color=["#888a91", "#bcbd22", "#3b7dd8", "#f1a340"],
                  edgecolor="black", linewidth=0.4)
    ax.set_ylabel("Total cross-attention mass")
    ax.set_title("Where does ACT attend?  (group totals)")
    for b, v, sz in zip(bars, totals, [sizes[k] for k in keys]):
        ax.text(b.get_x() + b.get_width() / 2, v + max(totals) * 0.01,
                f"{v:.3f}\n({sz} tokens)",
                ha="center", va="bottom", fontsize=9)
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    ax.set_axisbelow(True)

    ax = axes[1]
    bars = ax.bar(keys, per_token, color=["#888a91", "#bcbd22", "#3b7dd8", "#f1a340"],
                  edgecolor="black", linewidth=0.4)
    uniform = 1.0 / n_memory_tokens
    ax.axhline(uniform, color="red", linestyle="--", linewidth=0.9,
               label=f"uniform {uniform:.4f}")
    ax.set_ylabel("Mean cross-attention per token")
    ax.set_title("Per-token attention  (corrected for group size)")
    for b, v in zip(bars, per_token):
        ax.text(b.get_x() + b.get_width() / 2, v + max(per_token) * 0.01,
                f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    ax.set_axisbelow(True)

    fig.suptitle("P+ACT decoder cross-attention — group comparison",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_per_layer_heatmap(per_layer_per_sensor: np.ndarray,
                           sensor_names: List[str], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.2), dpi=140)
    im = ax.imshow(per_layer_per_sensor, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(sensor_names)))
    ax.set_xticklabels(sensor_names, rotation=70, ha="right", fontsize=7.5)
    ax.set_yticks(range(per_layer_per_sensor.shape[0]))
    ax.set_yticklabels([f"layer {i}" for i in range(per_layer_per_sensor.shape[0])])
    ax.set_title("Per-decoder-layer attention received by each proximity sensor")
    fig.colorbar(im, ax=ax, label="mean attention weight")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_temporal(temporal: Dict[float, np.ndarray], sensor_names: List[str],
                  out: Path) -> None:
    """temporal[t_frac] = (29,) mean per-sensor attention at that fraction
    of the episode."""
    fracs = sorted(temporal.keys())
    M = np.stack([temporal[f] for f in fracs], axis=0)        # (n_fracs, 29)
    fig, ax = plt.subplots(figsize=(11, 4.4), dpi=140)
    im = ax.imshow(M, aspect="auto", cmap="magma",
                   extent=(-0.5, len(sensor_names) - 0.5, fracs[-1], fracs[0]))
    ax.set_xticks(range(len(sensor_names)))
    ax.set_xticklabels(sensor_names, rotation=70, ha="right", fontsize=7.5)
    ax.set_ylabel("Fraction of episode (0 = start, 1 = end)")
    ax.set_title("How does per-sensor attention evolve as the gripper approaches?")
    fig.colorbar(im, ax=ax, label="mean attention weight")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# ---------- main ------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--prox_encoder_ckpt", required=True)
    p.add_argument("--prox_mapping_json", required=True)
    p.add_argument("--dataset_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n_batches", type=int, default=20,
                   help="Number of forward passes for the spatial summary.")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--chunk_size", type=int, default=100)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--dim_feedforward", type=int, default=3200)
    p.add_argument("--n_temporal_buckets", type=int, default=10)
    p.add_argument("--n_temporal_eps", type=int, default=24,
                   help="Number of episodes pooled into each temporal bucket.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # ---- load encoder ckpt extras ---------------------------------------
    enc_ckpt = torch.load(args.prox_encoder_ckpt, map_location="cpu",
                          weights_only=False)
    prox_mean = np.asarray(enc_ckpt["prox_mean"], dtype=np.float32)
    prox_std = np.asarray(enc_ckpt["prox_std"], dtype=np.float32)
    window = int(enc_ckpt["window"])

    with open(args.prox_mapping_json) as f:
        mapping = json.load(f)
    sensor_names = list(mapping["sensor_names"])
    n_sensors = int(mapping["n_sensors"])
    ep_ids = [int(k) for k in mapping["episodes"].keys()]
    rng.shuffle(ep_ids)

    norm_stats = _load_norm_stats(args.ckpt_dir)

    # ---- build policy + extractor ---------------------------------------
    policy, _cfg = build_policy(args.ckpt_dir, args.hidden_dim,
                                args.dim_feedforward, args.chunk_size,
                                n_sensors, args.seed)
    extractor = FrozenProxFeatureExtractor(args.prox_encoder_ckpt,
                                           device=torch.device("cuda"))
    recorder = CrossAttnRecorder(policy)

    print(f"[viz] sensors={n_sensors}  episodes_total={len(ep_ids)}  "
          f"batches={args.n_batches} of size {args.batch_size}")

    # ---- pass 1: spatial summary ----------------------------------------
    with torch.inference_mode():
        seen = 0
        for b_i in range(args.n_batches):
            batch_eps = [ep_ids[(seen + j) % len(ep_ids)]
                         for j in range(args.batch_size)]
            seen += args.batch_size
            try:
                img, qpos, prox, info = _build_batch(
                    args.dataset_dir, mapping, norm_stats,
                    prox_mean, prox_std, window, batch_eps,
                )
            except Exception as e:
                print(f"[viz] batch {b_i} skip: {e}")
                continue
            img = img.cuda(non_blocking=True)
            qpos = qpos.cuda(non_blocking=True)
            prox = prox.cuda(non_blocking=True)
            prox_pos = extractor(prox)                            # (B, 29, 3)
            _ = policy(qpos, img, proximity_positions=prox_pos)   # inference (no action)
            if b_i % 5 == 0:
                print(f"[viz] batch {b_i+1}/{args.n_batches} captured "
                      f"(seen {seen} samples).")

    weights = _stack_layer_weights(recorder.captured)            # (L, B, Q, S)
    summary = _summarise(weights, n_sensors)
    print(f"[viz] memory token count = {summary['n_memory_tokens']}")
    print(f"[viz] action query count = {summary['n_action_queries']}")
    print(f"[viz] attention sums to {summary['total_attention_check']:.4f} (≈1)")
    print(f"[viz] group attention totals = {summary['groups']}")
    top_idx = np.argsort(-summary["per_sensor"])[:5]
    print("[viz] top-5 sensors by attention:")
    for k in top_idx:
        print(f"        {sensor_names[k]:20s}  {summary['per_sensor'][k]:.5f}")

    plot_per_sensor(summary["per_sensor"], sensor_names,
                    out_dir / "per_sensor_attention.png")
    plot_groups(summary["groups"], summary["n_memory_tokens"], n_sensors,
                out_dir / "group_attention.png")
    plot_per_layer_heatmap(summary["per_layer_per_sensor"], sensor_names,
                           out_dir / "per_layer_per_sensor_heatmap.png")

    # ---- pass 2: temporal sweep ----------------------------------------
    recorder.clear()
    temporal: Dict[float, np.ndarray] = {}
    fracs = np.linspace(0.0, 1.0, args.n_temporal_buckets)
    n_ep = min(args.n_temporal_eps, len(ep_ids))
    eps_temp = ep_ids[:n_ep]
    with torch.inference_mode():
        for frac in fracs:
            recorder.clear()
            for chunk_lo in range(0, n_ep, args.batch_size):
                batch_eps = eps_temp[chunk_lo: chunk_lo + args.batch_size]
                if not batch_eps:
                    continue
                try:
                    img, qpos, prox, _ = _build_batch(
                        args.dataset_dir, mapping, norm_stats,
                        prox_mean, prox_std, window, batch_eps,
                        start_fracs=[float(frac)] * len(batch_eps),
                    )
                except Exception as e:
                    print(f"[viz] temporal frac={frac:.2f} skip: {e}")
                    continue
                img = img.cuda(non_blocking=True)
                qpos = qpos.cuda(non_blocking=True)
                prox = prox.cuda(non_blocking=True)
                prox_pos = extractor(prox)
                _ = policy(qpos, img, proximity_positions=prox_pos)
            w = _stack_layer_weights(recorder.captured)          # (L, B, Q, S)
            per_sensor = w.mean(dim=(0, 1, 2))[2:2 + n_sensors].numpy()
            temporal[float(frac)] = per_sensor
            print(f"[viz] temporal frac={frac:.2f}  mean_sens={per_sensor.mean():.5f}")

    plot_temporal(temporal, sensor_names, out_dir / "temporal_per_sensor.png")

    # ---- raw stats ------------------------------------------------------
    stats = {
        "per_sensor": summary["per_sensor"].tolist(),
        "per_layer_per_sensor": summary["per_layer_per_sensor"].tolist(),
        "groups": summary["groups"],
        "total_attention_check": summary["total_attention_check"],
        "n_memory_tokens": summary["n_memory_tokens"],
        "n_action_queries": summary["n_action_queries"],
        "n_layers": summary["n_layers"],
        "n_samples_pass1": summary["n_samples"],
        "sensor_names": sensor_names,
        "temporal_per_sensor": {f"{k:.3f}": v.tolist() for k, v in temporal.items()},
    }
    (out_dir / "raw_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"[viz] wrote {out_dir}/*.png + raw_stats.json")

    recorder.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
