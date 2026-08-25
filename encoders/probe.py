"""Probe geometry encoder (and its analytic target) on pact_place corridor rows.

Reads native ``obs/proximity/<sensor>`` ``(T, 4, 8, 8)`` from
``data/pact_place_corridor_v5/rows/*/trajectory.h5``. No ACT convert needed.

Without ``--checkpoint`` this scores the *target* the net is trained to hit
(nearest in-range XYZ, 20 cm) vs PACT-raw peak closeness (50 cm). That answers
"does this dataset even have geometry-encoder signal?"

With a frozen ``pact_surface_*_v1`` file it also scores XYZ error and validity
against that target. ``--untrained-episodes N`` runs the random net on N rows
as a negative control (do not treat that MAE as a quality verdict).

    python -m encoders.probe \\
        --src data/pact_place_corridor_v5 \\
        --out experiments_output/default/surface_encoder_probe/pact_place_corridor_v5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from encoders.peak_closeness import D_MAX, DEAD_PIXEL_M, HYBRID_SKIN_SENSOR_ORDER
from encoders.surface_geometry import (
    MAX_SURFACE_RANGE_M,
    nearest_surface_target_batch,
)


def _row_dirs(src: Path) -> list[Path]:
    rows = src / "rows" if (src / "rows").is_dir() else src
    dirs = [p for p in rows.iterdir() if p.is_dir() and (p / "trajectory.h5").is_file()]
    dirs.sort(key=lambda p: p.name)
    if not dirs:
        raise SystemExit(f"no rows/*/trajectory.h5 under {src}")
    return dirs


def _stack_native(grp, sensor_order: list[str]) -> np.ndarray:
    prox = grp["obs/proximity"]
    chans = []
    for name in sensor_order:
        if name not in prox:
            raise SystemExit(f"missing sensor {name!r} in {list(prox.keys())[:8]}")
        chans.append(np.asarray(prox[name], dtype=np.float32))
    return np.stack(chans, axis=1)


def _auc(y: np.ndarray, scores: np.ndarray) -> float | None:
    y = np.asarray(y, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)
    if y.min() == y.max() or len(y) < 4:
        return None
    order = np.argsort(scores)
    n_pos = float(y.sum())
    n_neg = float(len(y) - n_pos)
    ranks = np.empty(len(y), dtype=np.float64)
    ranks[order] = np.arange(1, len(y) + 1)
    sum_pos = float(ranks[y == 1].sum())
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _peak_closeness(depth: np.ndarray) -> np.ndarray:
    """``(..., 8, 8)`` metres -> peak closeness ``(...)`` in [0, 1], 50 cm cap."""
    closeness = np.clip(1.0 - depth / D_MAX, 0.0, 1.0)
    closeness = np.where(depth < DEAD_PIXEL_M, 0.0, closeness)
    return closeness.max(axis=(-2, -1))


def probe(
    src: Path,
    out_dir: Path,
    *,
    stride: int,
    max_episodes: int | None,
    checkpoint: Path | None,
    kind: str,
    device: str,
    batch_size: int,
    untrained_episodes: int,
) -> dict:
    import h5py

    from encoders.surface_geometry import SurfaceGeometryEncoder

    sensor_order = list(HYBRID_SKIN_SENSOR_ORDER)
    n_sensors = len(sensor_order)
    rows = _row_dirs(src)
    if max_episodes is not None:
        rows = rows[: int(max_episodes)]

    model = None
    net_mode = "skip"
    net_budget = 0
    if checkpoint is not None:
        model = SurfaceGeometryEncoder(
            kind=kind, checkpoint=checkpoint, device=device
        )
        net_mode = "checkpoint"
        net_budget = len(rows)
    elif untrained_episodes > 0:
        model = SurfaceGeometryEncoder(kind=kind, device=device)
        net_mode = "untrained"
        net_budget = min(int(untrained_episodes), len(rows))

    hits_20 = np.zeros(n_sensors, dtype=np.int64)
    hits_50 = np.zeros(n_sensors, dtype=np.int64)
    seen = np.zeros(n_sensors, dtype=np.int64)
    z_valid: list[float] = []
    per_ep = []
    xyz_err = []
    valid_match = []
    recon_mse = []

    for ep_index, row in enumerate(rows):
        res = {}
        result_path = row / "result.json"
        if result_path.is_file():
            res = json.loads(result_path.read_text())
        sp = res.get("scene_params") or {}
        side = str(sp.get("pact_intrusion_side") or "unknown")
        inbound_bow = float(
            ((res.get("grasp_diagnostics") or {}).get("bow_diagnostics") or {})
            .get("inbound", {})
            .get("accepted_bow_m")
            or 0.0
        )

        with h5py.File(row / "trajectory.h5", "r") as handle:
            grp = handle["traj_0"]
            prox = _stack_native(grp, sensor_order)

        n_steps = prox.shape[0]
        times = np.arange(0, n_steps, max(1, stride), dtype=np.int64)
        last = prox[times, :, -1]
        gt_xyz, gt_valid = nearest_surface_target_batch(last)
        peak = _peak_closeness(last)
        n_pairs = int(last.shape[0] * n_sensors)
        ep_valid = int(gt_valid.sum())
        hits_20 += gt_valid.sum(axis=0).astype(np.int64)
        hits_50 += (peak > 0.0).sum(axis=0).astype(np.int64)
        seen += np.full(n_sensors, last.shape[0], dtype=np.int64)
        if ep_valid:
            z_valid.extend(gt_xyz[gt_valid, 2].tolist())

        net_valid_frac = None
        net_xyz_mae_mm = None
        net_valid_acc = None
        if model is not None and ep_index < net_budget:
            packed = model.encode_episode_at_times(
                prox, times, batch_size=batch_size
            )
            pred_xyz = packed["xyz_m"].numpy()
            pred_valid = packed["valid"].numpy().astype(bool)
            both = gt_valid & pred_valid
            if np.any(both):
                err = np.linalg.norm(pred_xyz[both] - gt_xyz[both], axis=-1)
                xyz_err.extend(err.tolist())
                net_xyz_mae_mm = float(np.mean(err) * 1000.0)
            acc = float((pred_valid == gt_valid).mean())
            valid_match.append(acc)
            net_valid_acc = acc
            net_valid_frac = float(pred_valid.mean())
            if "reconstruction" in packed:
                from encoders.surface_geometry import depth_to_closeness

                gt_close = depth_to_closeness(last)
                recon = packed["reconstruction"].numpy()
                recon_mse.append(float(np.mean((recon - gt_close) ** 2)))

        per_ep.append(
            {
                "row": row.name,
                "side": side,
                "inbound_bow_m": inbound_bow,
                "n_steps": int(n_steps),
                "n_probed": n_pairs,
                "valid_20cm_frac": float(ep_valid / max(n_pairs, 1)),
                "peak_closeness_max": float(peak.max() if peak.size else 0.0),
                "peak_closeness_mean": float(peak.mean() if peak.size else 0.0),
                "net_valid_frac": net_valid_frac,
                "net_xyz_mae_mm": net_xyz_mae_mm,
                "net_valid_accuracy": net_valid_acc,
            }
        )
        extra = ""
        if net_xyz_mae_mm is not None:
            extra = f" net_mae={net_xyz_mae_mm:.1f}mm"
        print(
            f"{row.name}: T={n_steps} valid20={100 * ep_valid / max(n_pairs, 1):.2f}% "
            f"peak_max={float(peak.max()):.3f} side={side}{extra}",
            flush=True,
        )

    hit_rate_20 = hits_20 / np.maximum(seen, 1)
    hit_rate_50 = hits_50 / np.maximum(seen, 1)
    known = np.array([ep["side"] in ("left", "right") for ep in per_ep])
    sides = np.array(
        [1 if ep["side"] == "left" else 0 for ep in per_ep], dtype=np.int32
    )
    auc_side_peak = None
    auc_side_valid = None
    if known.any() and len(set(sides[known].tolist())) == 2:
        auc_side_peak = _auc(
            sides[known],
            np.array([ep["peak_closeness_max"] for ep in per_ep])[known],
        )
        auc_side_valid = _auc(
            sides[known],
            np.array([ep["valid_20cm_frac"] for ep in per_ep])[known],
        )

    summary = {
        "src": str(src.resolve()),
        "n_episodes": len(per_ep),
        "stride": int(stride),
        "sensor_order": sensor_order,
        "geometry_range_m": MAX_SURFACE_RANGE_M,
        "peak_closeness_range_m": D_MAX,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "net_mode": net_mode,
        "n_net_episodes": int(net_budget if model is not None else 0),
        "valid_20cm_frac_overall": float(hits_20.sum() / max(int(seen.sum()), 1)),
        "peak_50cm_hit_frac_overall": float(hits_50.sum() / max(int(seen.sum()), 1)),
        "n_valid_20cm_points": int(hits_20.sum()),
        "z_valid_mm_p50": float(np.median(z_valid) * 1000) if z_valid else None,
        "z_valid_mm_p10": float(np.percentile(z_valid, 10) * 1000) if z_valid else None,
        "z_valid_mm_p90": float(np.percentile(z_valid, 90) * 1000) if z_valid else None,
        "per_sensor_hit_rate_20cm": {
            name: float(hit_rate_20[i]) for i, name in enumerate(sensor_order)
        },
        "per_sensor_hit_rate_50cm": {
            name: float(hit_rate_50[i]) for i, name in enumerate(sensor_order)
        },
        "side_counts": {
            side: int(sum(1 for ep in per_ep if ep["side"] == side))
            for side in sorted({ep["side"] for ep in per_ep})
        },
        "auc_intrusion_side_from_peak_closeness": auc_side_peak,
        "auc_intrusion_side_from_valid20": auc_side_valid,
        "net_xyz_mae_mm": float(np.mean(xyz_err) * 1000) if xyz_err else None,
        "net_valid_accuracy": float(np.mean(valid_match)) if valid_match else None,
        "net_recon_mse": float(np.mean(recon_mse)) if recon_mse else None,
        "episodes": per_ep,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "probe.json").write_text(json.dumps(summary, indent=2) + "\n")
    _plots(out_dir, sensor_order, hit_rate_20, hit_rate_50, z_valid, per_ep)
    _print_gate(summary, out_dir)
    return summary


def _print_gate(summary: dict, out_dir: Path) -> None:
    v20 = summary["valid_20cm_frac_overall"]
    v50 = summary["peak_50cm_hit_frac_overall"]
    print("\n=== surface encoder probe ===")
    print(f"episodes={summary['n_episodes']}  stride={summary['stride']}")
    print(f"20cm geometry-valid fraction: {100 * v20:.3f}%  (encoder target)")
    print(f"50cm peak-closeness hit fraction: {100 * v50:.3f}%  (PACT-raw)")
    if summary["z_valid_mm_p50"] is not None:
        print(
            "valid z mm p10/p50/p90: "
            f"{summary['z_valid_mm_p10']:.1f} / {summary['z_valid_mm_p50']:.1f} / "
            f"{summary['z_valid_mm_p90']:.1f}"
        )
    top = sorted(
        summary["per_sensor_hit_rate_20cm"].items(), key=lambda kv: kv[1], reverse=True
    )[:8]
    fired = [f"{n}={100 * r:.2f}%" for n, r in top if r > 0]
    print("top 20cm sensors:", ", ".join(fired) if fired else "NONE")
    if summary["auc_intrusion_side_from_peak_closeness"] is not None:
        print(
            "AUC side L/R from peak closeness: "
            f"{summary['auc_intrusion_side_from_peak_closeness']:.3f}"
        )
        print(
            "AUC side L/R from 20cm valid frac: "
            f"{summary['auc_intrusion_side_from_valid20']:.3f}"
        )
    if summary["net_mode"] == "checkpoint":
        print(f"net XYZ MAE (valid both): {summary['net_xyz_mae_mm']}")
        print(f"net validity accuracy: {summary['net_valid_accuracy']}")
        if summary["net_recon_mse"] is not None:
            print(f"net recon MSE (latest 8x8 closeness): {summary['net_recon_mse']:.6f}")
    elif summary["net_mode"] == "untrained":
        print(
            f"UNTRAINED net on {summary['n_net_episodes']} eps — wiring only, not quality."
        )
        print(f"untrained XYZ MAE (valid both): {summary['net_xyz_mae_mm']}")
        print(f"untrained validity accuracy: {summary['net_valid_accuracy']}")
    else:
        print("NO checkpoint — net not scored. Analytic target only.")
        print("Pass --checkpoint path/to/pact_surface_*_v1.pt to test the trained net.")
    print(f"wrote {out_dir / 'probe.json'}")


def _plots(out_dir: Path, names, hit20, hit50, z_valid, per_ep) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 4))
    x = np.arange(len(names))
    ax.bar(x - 0.2, 100 * hit20, 0.4, label="20 cm geometry valid")
    ax.bar(x + 0.2, 100 * hit50, 0.4, label="50 cm peak-closeness hit")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_ylabel("% of probed tiles")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "sensor_hit_rates.png", dpi=120)
    plt.close(fig)

    if z_valid:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(np.asarray(z_valid) * 1000.0, bins=40)
        ax.set_xlabel("nearest-surface z (mm), 20 cm cap")
        ax.set_ylabel("count")
        fig.tight_layout()
        fig.savefig(out_dir / "valid_z_hist.png", dpi=120)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    for side, color in (("left", "C0"), ("right", "C1"), ("unknown", "0.5")):
        xs = [ep["peak_closeness_max"] for ep in per_ep if ep["side"] == side]
        ys = [100 * ep["valid_20cm_frac"] for ep in per_ep if ep["side"] == side]
        if xs:
            ax.scatter(xs, ys, s=18, alpha=0.7, label=side, c=color)
    ax.set_xlabel("episode max peak closeness (50 cm)")
    ax.set_ylabel("episode 20 cm valid %")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "episode_20_vs_50.png", dpi=120)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=Path("data/pact_place_corridor_v5"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "experiments_output/default/surface_encoder_probe/pact_place_corridor_v5"
        ),
    )
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--kind", choices=("xyz", "embedding"), default="embedding")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--untrained-episodes",
        type=int,
        default=2,
        help="If no --checkpoint, run random net on this many rows (0 = skip net).",
    )
    args = parser.parse_args()
    if args.device is None:
        import torch

        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = args.checkpoint if args.checkpoint and args.checkpoint.is_file() else None
    if args.checkpoint is not None and checkpoint is None:
        raise SystemExit(f"--checkpoint not a file: {args.checkpoint}")
    probe(
        args.src,
        args.out,
        stride=args.stride,
        max_episodes=args.max_episodes,
        checkpoint=checkpoint,
        kind=args.kind,
        device=args.device,
        batch_size=args.batch_size,
        untrained_episodes=args.untrained_episodes,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
