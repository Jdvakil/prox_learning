"""Train the proximity safety head: a small CVAE mapping the 40x8x8 skin depths to a
joint-space retreat delta, distilled from the analytic potential-field labels produced
by scripts/safety_sweep.py.

At runtime the head needs ONLY raw skin readings — no scene geometry, no object poses:
    head = SafetyHead.load("assets/safety/cvae_v1")
    dq = head(prox)          # prox: (40, 8, 8) float meters -> dq: (7,) joint delta

Inputs are converted to "closeness" c = clip(1 - d / D_MAX, 0, 1) (D_MAX = 0.5 m), so
far/no-return pixels are exactly zero and the head is quiet when nothing is near.
Labels are scaled to unit RMS over the close-encounter subset; the scale is stored in
meta.json and folded back in at inference.

Reported metrics:
  recon MSE          on the validation split (scaled label space)
  direction cosine   on close samples (min depth < 0.12 m) — does it push the right way?
  quiet RMS          on far samples (min depth > 0.25 m)   — is it silent when clear?

Usage:
    /opt/conda/envs/mlspaces/bin/python scripts/train_safety_cvae.py \
      --data assets/safety/sweep_v1.h5 --out assets/safety/cvae_v1 --epochs 60
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

D_MAX = 0.5     # closeness normalization range (m)
Z_DIM = 8


class SafetyCVAE(nn.Module):
    def __init__(self, n_in: int, n_out: int = 7, z_dim: int = Z_DIM) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.enc = nn.Sequential(
            nn.Linear(n_in + n_out, 512), nn.SiLU(),
            nn.Linear(512, 256), nn.SiLU(),
            nn.Linear(256, 2 * z_dim),
        )
        self.dec = nn.Sequential(
            nn.Linear(n_in + z_dim, 512), nn.SiLU(),
            nn.Linear(512, 256), nn.SiLU(),
            nn.Linear(256, n_out),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        mu, logvar = self.enc(torch.cat([x, y], -1)).chunk(2, -1)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.dec(torch.cat([x, z], -1)), mu, logvar

    @torch.no_grad()
    def act(self, x: torch.Tensor) -> torch.Tensor:
        """Deterministic head: decode at z = 0 (the prior mean)."""
        z = torch.zeros(x.shape[0], self.z_dim, device=x.device)
        return self.dec(torch.cat([x, z], -1))


def featurize(prox: np.ndarray) -> np.ndarray:
    """(N, S, 8, 8) depths in meters -> (N, S*64) closeness in [0, 1]."""
    d = prox.astype(np.float32)
    c = np.clip(1.0 - d / D_MAX, 0.0, 1.0)
    c[d < 0.005] = 0.0   # dead/invalid pixels
    return c.reshape(len(c), -1)


class SafetyHead:
    """Inference wrapper: raw (40, 8, 8) skin depths -> (7,) joint retreat delta."""

    def __init__(self, model: SafetyCVAE, scale: float, device: str = "cpu") -> None:
        self.model = model.to(device).eval()
        self.scale = scale
        self.device = device

    @classmethod
    def load(cls, ckpt_dir: str | Path, device: str = "cpu") -> "SafetyHead":
        ckpt_dir = Path(ckpt_dir)
        meta = json.loads((ckpt_dir / "meta.json").read_text())
        model = SafetyCVAE(meta["n_in"], meta["n_out"], meta["z_dim"])
        model.load_state_dict(torch.load(ckpt_dir / "model.pt", map_location=device))
        return cls(model, meta["label_scale"], device)

    def __call__(self, prox: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(featurize(prox[None])).to(self.device)
        return (self.model.act(x)[0].cpu().numpy() * self.scale).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path,
                    default=Path("/home/jaydv/code/prox_learning/assets/safety/cvae_v1"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--beta", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    with h5py.File(args.data, "r") as h:
        X = featurize(h["prox"][:])
        Y = h["label_dq"][:].astype(np.float32)
        md = h["min_depth"][:].astype(np.float32)
        sensors = [s.decode() for s in h["sensors"][:]]
    md_min = np.nanmin(np.where(np.isfinite(md), md, np.nan), axis=1)
    md_min = np.where(np.isfinite(md_min), md_min, np.inf)
    close = md_min < 0.12
    far = md_min > 0.25
    norms = np.linalg.norm(Y[close], axis=1)
    scale = float(np.sqrt(np.mean(norms**2))) if close.any() else 1.0
    Ys = Y / scale
    print(f"N={len(X)}  close={close.sum()}  far={far.sum()}  label scale={scale:.3f}")

    idx = np.random.permutation(len(X))
    n_val = max(256, len(X) // 10)
    vi, ti = idx[:n_val], idx[n_val:]
    Xt = torch.from_numpy(X[ti]).to(dev)
    Yt = torch.from_numpy(Ys[ti]).to(dev)
    Xv = torch.from_numpy(X[vi]).to(dev)
    Yv = torch.from_numpy(Ys[vi]).to(dev)

    model = SafetyCVAE(X.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best = np.inf
    args.out.mkdir(parents=True, exist_ok=True)

    for ep in range(args.epochs):
        model.train()
        beta = args.beta * min(1.0, (ep + 1) / 10)
        perm = torch.randperm(len(Xt), device=dev)
        tot = 0.0
        for k in range(0, len(Xt), args.bs):
            b = perm[k:k + args.bs]
            pred, mu, logvar = model(Xt[b], Yt[b])
            rec = F.mse_loss(pred, Yt[b])
            kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(-1)).mean()
            loss = rec + beta * kl
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(b)
        sched.step()

        model.eval()
        with torch.no_grad():
            pv = model.act(Xv)
            mse = float(F.mse_loss(pv, Yv))
            vc = torch.from_numpy(close[vi]).to(dev)
            vf = torch.from_numpy(far[vi]).to(dev)
            cos = float(F.cosine_similarity(pv[vc], Yv[vc], dim=-1).mean()) if vc.any() else float("nan")
            quiet = float(pv[vf].norm(dim=-1).mean()) if vf.any() else float("nan")
        print(f"ep {ep + 1:3d}  train {tot / len(Xt):.4f}  val mse {mse:.4f}  "
              f"close-cos {cos:.3f}  far-quiet {quiet:.4f}")
        if mse < best:
            best = mse
            torch.save(model.state_dict(), args.out / "model.pt")
            (args.out / "meta.json").write_text(json.dumps(dict(
                n_in=X.shape[1], n_out=7, z_dim=Z_DIM, label_scale=scale,
                d_max_input=D_MAX, sensors=sensors, data=str(args.data),
                best_val_mse=best, close_cos=cos, far_quiet=quiet), indent=2))

    print(f"best val mse {best:.4f} -> {args.out}/model.pt")


if __name__ == "__main__":
    main()
