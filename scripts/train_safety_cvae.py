"""Train the proximity safety head: a small CVAE mapping the 40x8x8 skin depths to a
joint-space retreat delta, distilled from the analytic potential-field labels produced
by scripts/safety_sweep.py.

At runtime the head needs ONLY raw skin readings — no scene geometry, no object poses:
    head = SafetyHead.load("assets/safety/cvae_v3")
    dq = head(prox)          # prox: (40, 8, 8) float meters -> dq: (7,) joint delta

Inputs are converted to "closeness" c = clip(1 - d / D_MAX, 0, 1) (D_MAX = 0.5 m), so
far/no-return pixels are exactly zero and the head is quiet when nothing is near.
Labels are scaled to unit RMS over the close-encounter subset; the scale is stored in
meta.json and folded back in at inference.

This is a CONDITIONAL VAE that RECONSTRUCTS the retreat vector:
    encoder q(z | skin, dq) -> (mu, logvar)        # sees skin AND the target dq at train time
    decoder p(dq | skin, z) -> dq_hat              # reconstructs dq from skin + a latent z
    loss = reconstruction_MSE(dq_hat, dq) + beta * KL(q || N(0, I))
At inference z is pinned to the prior mean (z = 0), so the head is a deterministic
skin -> dq map; the latent only shapes training and lets the head represent the small
ambiguity in "which way to dodge".

Logging (everything mirrored to wandb AND saved next to the weights):
    model.pt          best-val checkpoint
    meta.json         architecture + label scale + best metrics (used by SafetyHead.load)
    config.json       every hyper-parameter + dataset path for this run
    history.json      per-epoch arrays (loss, recon, kl, val mse, cos, quiet, lr, beta)
    curves.png        loss / reconstruction+KL / direction-cos+quiet / lr+beta over epochs
    encoder_latent.png  per-latent-dim KL (posterior-collapse view) + 2-D latent map
    pred_vs_true.png    decoder reconstruction quality, per joint (val close set)
    head_diagnostics.png  per-joint MSE, direction-cosine histogram, |dq| vs min skin depth

Reported metrics:
  recon MSE          on the validation split (scaled label space)
  direction cosine   on close samples (min depth < 0.12 m) — does it push the right way?
  quiet RMS          on far samples (min depth > 0.25 m)   — is it silent when clear?

Usage:
    OMP_NUM_THREADS=2 /opt/conda/envs/mlspaces/bin/python scripts/train_safety_cvae.py \
      --data assets/safety/sweep_v3.h5 --out assets/safety/cvae_v3 --epochs 60 \
      --wandb-project prox-safety-cvae
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

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


# ----------------------------------------------------------------------------- logging

def _per_dim_kl(mu: np.ndarray, logvar: np.ndarray) -> np.ndarray:
    """Mean KL contribution of each latent dim — near 0 means that dim collapsed to the
    prior (unused); large means the encoder is storing information there."""
    return np.mean(-0.5 * (1 + logvar - mu**2 - np.exp(logvar)), axis=0)


@torch.no_grad()
def collect_diag(model: SafetyCVAE, Xv: torch.Tensor, Yv: torch.Tensor,
                 close: np.ndarray, md_min: np.ndarray) -> dict:
    """Everything the plots need from the best model on the validation split."""
    model.eval()
    mu, logvar = model.enc(torch.cat([Xv, Yv], -1)).chunk(2, -1)
    pred = model.act(Xv)
    mu, logvar, pred = mu.cpu().numpy(), logvar.cpu().numpy(), pred.cpu().numpy()
    true = Yv.cpu().numpy()
    cos = (np.sum(pred * true, 1) /
           (np.linalg.norm(pred, axis=1) * np.linalg.norm(true, axis=1) + 1e-9))
    return dict(mu=mu, logvar=logvar, pred=pred, true=true, cos=cos,
                close=close, md_min=md_min,
                pred_norm=np.linalg.norm(pred, axis=1),
                per_joint_mse=np.mean((pred - true) ** 2, axis=0),
                kl_per_dim=_per_dim_kl(mu, logvar))


def plot_curves(hist: dict, path: Path) -> None:
    ep = hist["epoch"]
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    ax[0, 0].plot(ep, hist["train_loss"], label="train total")
    ax[0, 0].plot(ep, hist["val_mse"], label="val recon MSE")
    ax[0, 0].set_title("loss"); ax[0, 0].set_xlabel("epoch"); ax[0, 0].legend()
    ax[0, 0].set_yscale("log")

    ax[0, 1].plot(ep, hist["train_recon"], label="train recon MSE")
    ax[0, 1].plot(ep, hist["train_kl"], label="train KL")
    ax[0, 1].set_title("reconstruction vs KL"); ax[0, 1].set_xlabel("epoch"); ax[0, 1].legend()
    ax[0, 1].set_yscale("log")

    ax[1, 0].plot(ep, hist["close_cos"], color="tab:green")
    ax[1, 0].set_ylabel("close direction cosine", color="tab:green")
    ax[1, 0].set_ylim(-0.1, 1.0); ax[1, 0].set_xlabel("epoch")
    q = ax[1, 0].twinx()
    q.plot(ep, hist["far_quiet"], color="tab:red")
    q.set_ylabel("far-quiet RMS", color="tab:red")
    ax[1, 0].set_title("direction accuracy (green) + quiet-when-clear (red)")

    ax[1, 1].plot(ep, hist["lr"], color="tab:blue", label="lr")
    ax[1, 1].set_ylabel("lr", color="tab:blue"); ax[1, 1].set_xlabel("epoch")
    b = ax[1, 1].twinx()
    b.plot(ep, hist["beta"], color="tab:orange", label="beta")
    b.set_ylabel("KL weight beta", color="tab:orange")
    ax[1, 1].set_title("lr schedule + beta warm-up")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_encoder_latent(diag: dict, path: Path) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    kl = diag["kl_per_dim"]
    ax[0].bar(np.arange(len(kl)), kl, color="tab:purple")
    ax[0].axhline(0.01, ls="--", c="gray", lw=1)
    ax[0].set_title(f"per-latent-dim KL  ({int((kl > 0.01).sum())}/{len(kl)} active)")
    ax[0].set_xlabel("latent dim"); ax[0].set_ylabel("mean KL (nats)")

    mu = diag["mu"]
    if mu.shape[1] >= 2:
        m = mu - mu.mean(0)
        _, _, vt = np.linalg.svd(m, full_matrices=False)
        emb = m @ vt[:2].T
    else:
        emb = np.column_stack([mu[:, 0], np.zeros(len(mu))])
    close, md = diag["close"], diag["md_min"]
    far = md > 0.25
    mid = ~close & ~far
    ax[1].scatter(emb[mid, 0], emb[mid, 1], s=4, c="lightgray", label="mid")
    ax[1].scatter(emb[far, 0], emb[far, 1], s=4, c="tab:blue", label="far/clear")
    ax[1].scatter(emb[close, 0], emb[close, 1], s=6, c="tab:red", label="close")
    ax[1].set_title("latent posterior mean (top-2 PCA)")
    ax[1].set_xlabel("pc1"); ax[1].set_ylabel("pc2"); ax[1].legend()
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_pred_vs_true(diag: dict, path: Path) -> None:
    pred, true, close = diag["pred"], diag["true"], diag["close"]
    fig, ax = plt.subplots(3, 3, figsize=(11, 10))
    for j in range(7):
        a = ax[j // 3, j % 3]
        a.scatter(true[~close, j], pred[~close, j], s=3, c="lightgray", alpha=0.5)
        a.scatter(true[close, j], pred[close, j], s=5, c="tab:red", alpha=0.6)
        lim = max(1e-3, np.abs(true[:, j]).max(), np.abs(pred[:, j]).max())
        a.plot([-lim, lim], [-lim, lim], "k--", lw=1)
        a.set_title(f"joint {j + 1}"); a.set_xlabel("true dq"); a.set_ylabel("pred dq")
    for j in (7, 8):
        ax[j // 3, j % 3].axis("off")
    fig.suptitle("decoder reconstruction (scaled label space) — red = close samples")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_head_diagnostics(diag: dict, path: Path) -> None:
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    ax[0].bar(np.arange(1, 8), diag["per_joint_mse"], color="tab:cyan")
    ax[0].set_title("per-joint reconstruction MSE"); ax[0].set_xlabel("joint")

    close = diag["close"]
    ax[1].hist(diag["cos"][close], bins=30, range=(-1, 1), color="tab:green")
    ax[1].axvline(float(np.mean(diag["cos"][close])) if close.any() else 0,
                  c="k", ls="--", lw=1)
    ax[1].set_title("direction cosine, close samples"); ax[1].set_xlabel("cosine")

    md, pn = diag["md_min"], diag["pred_norm"]
    finite = np.isfinite(md)
    ax[2].scatter(md[finite], pn[finite], s=4, c="tab:orange", alpha=0.5)
    ax[2].set_title("retreat magnitude vs nearest obstacle")
    ax[2].set_xlabel("min skin depth (m)"); ax[2].set_ylabel("|predicted dq| (scaled)")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path,
                    default=Path("/home/jaydv/code/prox_learning/assets/safety/cvae_v3"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--beta", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wandb-project", default="prox-safety-cvae",
                    help="wandb project; logging is on by default")
    ap.add_argument("--wandb-entity", default=None)
    ap.add_argument("--no-wandb", action="store_true", help="disable wandb (plots still saved)")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

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
    close_v, md_v = close[vi], md_min[vi]

    # wandb (default on; degrade gracefully if unavailable)
    wb = None
    if not args.no_wandb:
        try:
            import wandb
            wb = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                            name=args.out.name, dir=str(args.out),
                            config=dict(data=str(args.data), epochs=args.epochs, bs=args.bs,
                                        lr=args.lr, beta=args.beta, z_dim=Z_DIM,
                                        d_max=D_MAX, n_in=X.shape[1], n_out=7, seed=args.seed,
                                        n_train=len(ti), n_val=len(vi), label_scale=scale))
        except Exception as e:  # noqa: BLE001
            print(f"[wandb disabled: {e}]")
            wb = None

    model = SafetyCVAE(X.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best = np.inf
    hist = {k: [] for k in ("epoch", "train_loss", "train_recon", "train_kl",
                            "val_mse", "close_cos", "far_quiet", "lr", "beta")}

    for ep in range(args.epochs):
        model.train()
        beta = args.beta * min(1.0, (ep + 1) / 10)
        perm = torch.randperm(len(Xt), device=dev)
        tot = rec_tot = kl_tot = 0.0
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
            rec_tot += float(rec) * len(b)
            kl_tot += float(kl) * len(b)
        sched.step()

        model.eval()
        with torch.no_grad():
            pv = model.act(Xv)
            mse = float(F.mse_loss(pv, Yv))
            vc = torch.from_numpy(close_v).to(dev)
            vf = torch.from_numpy(far[vi]).to(dev)
            cos = float(F.cosine_similarity(pv[vc], Yv[vc], dim=-1).mean()) if vc.any() else float("nan")
            quiet = float(pv[vf].norm(dim=-1).mean()) if vf.any() else float("nan")
        lr_now = opt.param_groups[0]["lr"]
        print(f"ep {ep + 1:3d}  train {tot / len(Xt):.4f}  val mse {mse:.4f}  "
              f"close-cos {cos:.3f}  far-quiet {quiet:.4f}")
        for k, v in (("epoch", ep + 1), ("train_loss", tot / len(Xt)),
                     ("train_recon", rec_tot / len(Xt)), ("train_kl", kl_tot / len(Xt)),
                     ("val_mse", mse), ("close_cos", cos), ("far_quiet", quiet),
                     ("lr", lr_now), ("beta", beta)):
            hist[k].append(v)
        if wb is not None:
            wb.log(dict(epoch=ep + 1, train_loss=tot / len(Xt),
                        train_recon=rec_tot / len(Xt), train_kl=kl_tot / len(Xt),
                        val_mse=mse, close_cos=cos, far_quiet=quiet, lr=lr_now, beta=beta),
                   step=ep + 1)

        if mse < best:
            best = mse
            torch.save(model.state_dict(), args.out / "model.pt")
            (args.out / "meta.json").write_text(json.dumps(dict(
                n_in=X.shape[1], n_out=7, z_dim=Z_DIM, label_scale=scale,
                d_max_input=D_MAX, sensors=sensors, data=str(args.data),
                best_val_mse=best, close_cos=cos, far_quiet=quiet), indent=2))

    print(f"best val mse {best:.4f} -> {args.out}/model.pt")

    # diagnostics + plots from the BEST checkpoint
    model.load_state_dict(torch.load(args.out / "model.pt", map_location=dev))
    diag = collect_diag(model, Xv, Yv, close_v, md_v)

    (args.out / "history.json").write_text(json.dumps(hist, indent=2))
    (args.out / "config.json").write_text(json.dumps(dict(
        data=str(args.data), out=str(args.out), epochs=args.epochs, bs=args.bs,
        lr=args.lr, beta=args.beta, z_dim=Z_DIM, d_max=D_MAX, seed=args.seed,
        n_in=int(X.shape[1]), n_out=7, n_train=int(len(ti)), n_val=int(len(vi)),
        label_scale=scale, best_val_mse=float(best),
        active_latent_dims=int((diag["kl_per_dim"] > 0.01).sum())), indent=2))

    figs = {"curves.png": plot_curves, "encoder_latent.png": plot_encoder_latent,
            "pred_vs_true.png": plot_pred_vs_true, "head_diagnostics.png": plot_head_diagnostics}
    for name, fn in figs.items():
        p = args.out / name
        fn(hist if name == "curves.png" else diag, p)
        print(f"wrote {p}")
        if wb is not None:
            import wandb
            wb.log({f"plot/{name[:-4]}": wandb.Image(str(p))})

    if wb is not None:
        wb.summary["best_val_mse"] = float(best)
        wb.summary["active_latent_dims"] = int((diag["kl_per_dim"] > 0.01).sum())
        wb.finish()


if __name__ == "__main__":
    main()
