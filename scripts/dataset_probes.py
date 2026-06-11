"""The four pre-training dataset probes for the enclosure-reach dataset (advisor spec).

1. Speed↔clearance: regress expert EE speed on the min skin reading. The expert must move
   slower when closer — if absent, no training trick will conjure sensor reliance.
2. Deflection-side probe: logistic prediction of deflection side from the 29-sensor min-reading
   pattern around detection. Should beat chance by a wide margin.
3. Decorrelation matrix: hidden-obstacle parameters x camera-visible parameters ≈ 0.
4. Signal distribution: fraction of timesteps with any zone reading below 8 cm.

Usage:
  python scripts/dataset_probes.py --glob 'assets/datagen/enclosure_v1/**/trajectories_batch_*.h5'
"""
from __future__ import annotations

import argparse
import glob as globlib
import json

import h5py
import numpy as np

SENSORS = [f"link{l}_sensor_{s}" for (l, n) in [(2, 7), (3, 8), (5, 6), (6, 8)] for s in range(n)]
PROX_VALID = (0.051, 3.99)


def load_episode(t):
    T = t["rewards"].shape[0]
    # min valid reading per sensor per step (mean over substeps first)
    mins = np.full((T, len(SENSORS)), np.inf, dtype=np.float64)
    for j, s in enumerate(SENSORS):
        d = t[f"obs/proximity/{s}"][:].mean(axis=1).reshape(T, -1)
        valid = (d > PROX_VALID[0]) & (d < PROX_VALID[1])
        dv = np.where(valid, d, np.inf)
        mins[:, j] = dv.min(axis=1)
    tcp = t["obs/extra/tcp_pose"][:, :3].astype(np.float64)
    raw = t["obs_scene"]
    raw = raw[()] if raw.shape == () else raw[0]
    s = raw.tobytes().decode("utf-8", "ignore").rstrip("\x00") if isinstance(raw, np.ndarray) \
        else raw.decode("utf-8", "ignore").rstrip("\x00")
    meta = json.loads(s)
    return mins, tcp, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True)
    ap.add_argument("--near_cm", type=float, default=8.0)
    args = ap.parse_args()

    eps = []
    for path in sorted(globlib.glob(args.glob, recursive=True)):
        with h5py.File(path, "r") as f:
            for k in sorted(k for k in f if k.startswith("traj_")):
                try:
                    eps.append(load_episode(f[k]))
                except Exception as e:
                    print(f"skip {path}:{k}: {e}")
    print(f"episodes loaded: {len(eps)}")
    if not eps:
        return

    # ---------- probe 1: speed vs min reading ----------
    speeds, dists = [], []
    for mins, tcp, meta in eps:
        dt = float(meta.get("policy_dt_ms", 66.0)) / 1000.0
        v = np.linalg.norm(np.diff(tcp, axis=0), axis=1) / dt
        m = mins.min(axis=1)[1:]
        ok = np.isfinite(m) & (v < 1.0)
        speeds.append(v[ok]); dists.append(m[ok])
    v = np.concatenate(speeds); m = np.concatenate(dists)
    r = float(np.corrcoef(m, v)[0, 1])
    slope = float(np.polyfit(m, v, 1)[0])
    print(f"\n[1] speed vs min-skin-distance: n={len(v)}  pearson r={r:+.3f}  slope={slope:+.3f} (m/s per m)")
    print(f"    expert moves slower when closer -> expect r strongly POSITIVE (speed grows with distance)")
    print(f"    VERDICT: {'PASS' if r > 0.3 else 'FAIL — expert is not modulating speed on clearance'}")

    # ---------- probe 2: deflection side from zone asymmetry ----------
    X, y = [], []
    for mins, tcp, meta in eps:
        sp = meta.get("scene_params", {})
        if meta.get("behavior_class") != "deflect" or sp.get("protr_wall") not in ("left", "right"):
            continue
        feat = np.where(np.isfinite(mins), mins, 0.5).min(axis=0)  # per-sensor min over episode
        X.append(feat); y.append(1 if sp["protr_wall"] == "left" else 0)
    if len(y) >= 10 and len(set(y)) == 2:
        X = np.array(X); y = np.array(y)
        # simple logistic via least squares on standardized features (no sklearn dependency)
        Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
        w, *_ = np.linalg.lstsq(np.c_[Xs, np.ones(len(y))], (y * 2 - 1), rcond=None)
        acc = float((((np.c_[Xs, np.ones(len(y))] @ w) > 0).astype(int) == y).mean())
        print(f"\n[2] deflection-side probe: n={len(y)}  train-acc={acc:.2f} (chance 0.5)")
        print(f"    VERDICT: {'PASS' if acc > 0.75 else 'WEAK — side not decodable from zone pattern'}")
    else:
        print(f"\n[2] deflection-side probe: insufficient deflect episodes ({len(y)}) — collect more")

    # ---------- probe 3: decorrelation matrix ----------
    rows = []
    for _, _, meta in eps:
        sp = meta.get("scene_params", {})
        if not sp:
            continue
        # residual_margin is the DECORRELATED hidden quantity (gap the arm has left after the
        # obstacle), drawn independently of clearance; intrusion is a derived placement detail
        # that is clearance-coupled by construction, so it is NOT the decorrelation target.
        # residual/protr_pos are NaN when no obstacle is present (undefined) — masked per-cell.
        def _g(key):
            v = sp.get(key, None)
            return float(v) if v is not None else float("nan")
        rows.append([
            float(sp.get("protrusion_present", False)),
            _g("residual_margin"),
            _g("protr_pos_frac"),
            _g("clearance"),
            _g("depth"),
            _g("light_scale"),
            _g("target_frac"),
        ])
    A = np.array(rows)
    if len(A) >= 10:
        names_h = ["protr_present", "residual", "protr_pos"]
        names_v = ["clearance", "depth", "light", "target_frac"]

        def pair_corr(x, y):
            # complete-case Pearson: drop rows where either value is NaN (e.g. free cells have
            # no residual/protr_pos). With true independence, |r| noise ~ 1/sqrt(n_eff-3).
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 8 or np.std(x[mask]) < 1e-9 or np.std(y[mask]) < 1e-9:
                return float("nan"), int(mask.sum())
            return float(np.corrcoef(x[mask], y[mask])[0, 1]), int(mask.sum())

        worst, worst_n = 0.0, 0
        print(f"\n[3] decorrelation (hidden x visible), n={len(A)}:")
        for i, nh in enumerate(names_h):
            for j, nv in enumerate(names_v):
                c, neff = pair_corr(A[:, i], A[:, 3 + j])
                if np.isfinite(c) and abs(c) > worst:
                    worst, worst_n = abs(c), neff
                cs = f"{c:+.3f}" if np.isfinite(c) else "  n/a"
                print(f"    corr({nh:13s},{nv:11s}) = {cs}  (n={neff})")
        # noise floor for the worst cell given its effective sample size (1.96 SE, ~95%)
        band = 1.96 / np.sqrt(max(worst_n - 3, 1)) if worst_n else float("inf")
        ok = worst < max(0.15, band)
        print(f"    max |corr| = {worst:.3f} (n_eff={worst_n}, noise band ±{band:.2f})  "
              f"VERDICT: {'PASS' if ok else 'FAIL — visual shortcut risk'}")
    else:
        print("\n[3] decorrelation: insufficient episodes")

    # ---------- probe 4: signal distribution ----------
    frac_near = []
    for mins, _, _ in eps:
        m = mins.min(axis=1)
        frac_near.append(float((m < args.near_cm / 100.0).mean()))
    fn = float(np.mean(frac_near))
    print(f"\n[4] signal distribution: mean fraction of steps with any zone < {args.near_cm:.0f}cm = {fn:.2f}")
    print(f"    VERDICT: {'PASS' if fn > 0.3 else 'TOO GENEROUS — clearance distribution gives the skin little to say'}")

    # cell composition sanity
    cells = {}
    for _, _, meta in eps:
        c = meta.get("scene_params", {}).get("cell", "?")
        b = meta.get("behavior_class", "?")
        cells[(c, b)] = cells.get((c, b), 0) + 1
    print(f"\ncell x behavior composition: {cells}")


if __name__ == "__main__":
    main()
