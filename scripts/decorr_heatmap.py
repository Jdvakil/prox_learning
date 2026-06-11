#!/usr/bin/env python
"""Decorrelation matrix (advisor requirement).

Per scene, build a heatmap of |Pearson r| between HIDDEN params
(residual_margin, protr_pos_frac [fallback protr_center[0]], protrusion_present)
and VISIBLE params (clearance, depth, light_scale, target_frac).

Complete-case analysis: drop rows with any NaN among the used params
(e.g. free-cell residual_margin is NaN). Annotate each cell with signed r.
Title each subplot with max|r| and PASS/FAIL vs a noise band 1.96/sqrt(n_eff-3).
Low everywhere = good: vision (RGB) cannot shortcut-predict the hidden geometry,
so the policy is forced to use the proximity skin.
"""
import os, json, glob
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTDIR = "/home/jaydv/code/prox_learning/diagnostics_output/20260610_plots_v3"
os.makedirs(OUTDIR, exist_ok=True)

RUNS = {
    "fumehood": "/home/jaydv/code/prox_learning/assets/datagen/fumehood_smoke/FrankaSkinFumehoodSmokeConfig/20260610_090517",
    "panel":    "/home/jaydv/code/prox_learning/assets/datagen/panel_slalom_smoke/FrankaSkinPanelSlalomSmokeConfig/20260610_092607",
    "cubby":    "/home/jaydv/code/prox_learning/assets/datagen/cubby_smoke/FrankaSkinCubbySmokeConfig/20260610_095739",
}

HIDDEN = ["residual_margin", "protr_pos_frac", "protrusion_present"]
HIDDEN_LBL = ["residual_margin\n(m)", "protr_pos_frac\n(frac)", "protrusion_present\n(0/1)"]
VISIBLE = ["clearance", "depth", "light_scale", "target_frac"]
VISIBLE_LBL = ["clearance\n(m)", "depth\n(m)", "light_scale\n(-)", "target_frac\n(-)"]


def decode_scene(t):
    """Decode obs_scene JSON blob from a trajectory group; return scene_params dict."""
    blob = t["obs_scene"][()]
    if getattr(blob, "shape", ()) != () and not hasattr(blob, "tobytes"):
        blob = blob[0]
    s = (blob.tobytes() if hasattr(blob, "tobytes") else blob)
    if isinstance(s, bytes):
        s = s.decode("utf-8", "ignore")
    s = s.rstrip("\x00")
    return json.loads(s).get("scene_params", {})


def collect(run_dir):
    """One row per trajectory: dict of param->float (NaN if missing/None)."""
    rows = []
    h5s = sorted(glob.glob(os.path.join(run_dir, "house_*", "trajectories_batch_*.h5")))
    for path in h5s:
        try:
            with h5py.File(path, "r") as h:
                for tname in h.keys():
                    try:
                        sp = decode_scene(h[tname])
                    except Exception:
                        continue
                    row = {}
                    # hidden
                    rm = sp.get("residual_margin", np.nan)
                    row["residual_margin"] = float(rm) if rm is not None else np.nan
                    if "protr_pos_frac" in sp and sp["protr_pos_frac"] is not None:
                        row["protr_pos_frac"] = float(sp["protr_pos_frac"])
                    else:
                        pc = sp.get("protr_center", None)
                        row["protr_pos_frac"] = float(pc[0]) if pc else np.nan
                    pp = sp.get("protrusion_present", np.nan)
                    row["protrusion_present"] = float(bool(pp)) if pp is not None else np.nan
                    # visible
                    for k in VISIBLE:
                        v = sp.get(k, np.nan)
                        try:
                            row[k] = float(v) if v is not None else np.nan
                        except (TypeError, ValueError):
                            row[k] = np.nan
                    rows.append(row)
        except Exception as e:
            print(f"  skip {path}: {e}")
    return rows


def to_matrix(rows):
    cols = HIDDEN + VISIBLE
    M = np.array([[r.get(c, np.nan) for c in cols] for r in rows], dtype=float)
    return M, cols


def main():
    scenes = list(RUNS.keys())
    fig, axes = plt.subplots(1, len(scenes), figsize=(6.2 * len(scenes), 5.6))
    if len(scenes) == 1:
        axes = [axes]

    summary = {}
    for ax, scene in zip(axes, scenes):
        rows = collect(RUNS[scene])
        M, cols = to_matrix(rows)
        hi_idx = [cols.index(c) for c in HIDDEN]
        vi_idx = [cols.index(c) for c in VISIBLE]

        R = np.full((len(HIDDEN), len(VISIBLE)), np.nan)
        Nuse = np.zeros_like(R, dtype=int)
        for i, hi in enumerate(hi_idx):
            for j, vi in enumerate(vi_idx):
                x = M[:, hi]
                y = M[:, vi]
                # complete-case for this pair
                m = np.isfinite(x) & np.isfinite(y)
                xs, ys = x[m], y[m]
                Nuse[i, j] = xs.size
                if xs.size >= 4 and np.std(xs) > 1e-12 and np.std(ys) > 1e-12:
                    R[i, j] = np.corrcoef(xs, ys)[0, 1]

        absR = np.abs(R)
        # n_eff = min pairwise complete-case count across the matrix (conservative band)
        valid_counts = Nuse[np.isfinite(R)]
        n_eff = int(valid_counts.min()) if valid_counts.size else 0
        band = 1.96 / np.sqrt(max(n_eff - 3, 1)) if n_eff > 3 else np.inf
        maxr = np.nanmax(absR) if np.isfinite(absR).any() else np.nan
        passed = np.isfinite(maxr) and maxr <= band
        summary[scene] = (maxr, band, n_eff, passed, len(rows))

        im = ax.imshow(absR, cmap="magma", vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_xticks(range(len(VISIBLE)))
        ax.set_xticklabels(VISIBLE_LBL, fontsize=9)
        ax.set_yticks(range(len(HIDDEN)))
        ax.set_yticklabels(HIDDEN_LBL, fontsize=9)
        ax.set_xlabel("VISIBLE params (RGB-inferable)", fontsize=10)
        if scene == scenes[0]:
            ax.set_ylabel("HIDDEN params (geometry behind walls)", fontsize=10)

        for i in range(len(HIDDEN)):
            for j in range(len(VISIBLE)):
                if np.isfinite(R[i, j]):
                    txt = f"{R[i, j]:+.2f}\nn={Nuse[i,j]}"
                    col = "white" if absR[i, j] < 0.6 else "black"
                else:
                    txt = "n/a"
                    col = "0.6"
                ax.text(j, i, txt, ha="center", va="center", color=col, fontsize=8)

        verdict = "PASS" if passed else "FAIL"
        vcolor = "#1b7837" if passed else "#b2182b"
        ttl = (f"{scene}   max|r|={maxr:.2f}  vs band={band:.2f}  [{verdict}]\n"
               f"n_eff(complete-case)={n_eff}, episodes={len(rows)}")
        ax.set_title(ttl, fontsize=10.5, color=vcolor)

    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("|Pearson r|  (0 = decorrelated, good)", fontsize=10)

    fig.suptitle(
        "Hidden-vs-Visible Decorrelation Matrix  (low everywhere => RGB cannot shortcut hidden geometry)\n"
        "PASS = max|r| <= noise band 1.96/sqrt(n_eff-3); cells show signed r and complete-case n",
        fontsize=12, y=1.02,
    )

    out = os.path.join(OUTDIR, "decorr_heatmap.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\n=== SUMMARY ===")
    for sc, (mr, bd, ne, ps, ep) in summary.items():
        print(f"{sc:9s}: max|r|={mr:.3f} band={bd:.3f} n_eff={ne} episodes={ep} -> {'PASS' if ps else 'FAIL'}")
    sz = os.path.getsize(out)
    print(f"\nWROTE {out}  ({sz} bytes)")
    return out, summary


if __name__ == "__main__":
    main()
