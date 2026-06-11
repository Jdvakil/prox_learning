#!/usr/bin/env python
"""
PROBE 2 - deflection-side selection.

For deflect-behavior episodes the protrusion sits on protr_wall (left/right/top).
We measure the arm's lateral (world-y) deflection during the deflect phase and ask:
does the expert steer AWAY from the wall?

Frame fact (verified from the data, protr_center is task-local):
  wall=left  -> protrusion on +y side  -> expert should deflect toward -y (away)
  wall=right -> protrusion on -y side  -> expert should deflect toward +y (away)
  wall=top   -> protrusion is VERTICAL  -> no defined left/right side (excluded, noted)
"""
import os, glob, json
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUTDIR = "/home/jaydv/code/prox_learning/diagnostics_output/20260610_plots_v3"
os.makedirs(OUTDIR, exist_ok=True)
OUTPNG = os.path.join(OUTDIR, "deflect_side.png")

ROOTS = {
    "fumehood": "/home/jaydv/code/prox_learning/assets/datagen/fumehood_smoke/FrankaSkinFumehoodSmokeConfig/20260610_090517",
    "panel":    "/home/jaydv/code/prox_learning/assets/datagen/panel_slalom_smoke/FrankaSkinPanelSlalomSmokeConfig/20260610_092607",
    "cubby":    "/home/jaydv/code/prox_learning/assets/datagen/cubby_smoke/FrankaSkinCubbySmokeConfig/20260610_095739",
}

# deflect + extract_deflect phase ids (lateral steering happens here)
DEFLECT_PHASE_IDS = {13, 16}


def decode_json(t, key):
    blob = t[key][()]
    if not hasattr(blob, "tobytes") and getattr(blob, "shape", ()) != ():
        blob = t[key][0]
    s = (blob.tobytes() if hasattr(blob, "tobytes") else blob).decode("utf-8", "ignore").rstrip("\x00")
    return json.loads(s)


def collect():
    """Return list of dicts: one per deflect episode."""
    rows = []
    for scene, root in ROOTS.items():
        for hp in sorted(glob.glob(os.path.join(root, "house_*", "trajectories_batch_*.h5"))):
            try:
                f = h5py.File(hp, "r")
            except Exception:
                continue
            for tn in f.keys():
                try:
                    t = f[tn]
                    sc = decode_json(t, "obs_scene")
                    if sc.get("behavior_class") != "deflect":
                        continue
                    sp = sc.get("scene_params", {})
                    wall = sp.get("protr_wall")
                    tcp = t["obs/extra/tcp_pose"][()]
                    if tcp.ndim != 2 or tcp.shape[0] < 2:
                        continue
                    ph = t["obs/extra/policy_phase"][()]
                    y = tcp[:, 1].astype(float)
                    # baseline = first deflect-phase sample if present, else episode start.
                    mask = np.isin(ph, list(DEFLECT_PHASE_IDS))
                    used_deflect_phase = bool(mask.any())
                    if used_deflect_phase:
                        idx = np.where(mask)[0]
                        y0 = y[idx[0]]
                        seg = y[idx]
                    else:
                        # fallback: whole-episode excursion vs start
                        y0 = y[0]
                        seg = y
                    dy = seg - y0
                    if dy.size == 0:
                        continue
                    k = int(np.argmax(np.abs(dy)))
                    excursion = float(dy[k])              # signed peak lateral excursion (m)
                    sign = int(np.sign(excursion))         # +1 -> moved +y, -1 -> moved -y
                    pc = sp.get("protr_center")
                    pcy = float(pc[1]) if pc else np.nan
                    rows.append(dict(scene=scene, wall=wall, excursion=excursion,
                                     sign=sign, pcy=pcy,
                                     used_deflect_phase=used_deflect_phase))
                except Exception:
                    continue
            f.close()
    return rows


def away_correct(wall, sign):
    """True if observed deflect sign is AWAY from the wall.
    left wall (+y) -> away is -y (sign<0); right wall (-y) -> away is +y (sign>0)."""
    if wall == "left":
        return sign < 0
    if wall == "right":
        return sign > 0
    return None  # top: undefined


def main():
    rows = collect()
    n_total = len(rows)

    # ---- confusion: rows = wall side, cols = observed deflect direction ----
    walls = ["left", "right", "top"]
    # observed lateral direction buckets
    dir_labels = ["deflect +y\n(toward right side)", "deflect -y\n(toward left side)"]
    conf = np.zeros((len(walls), 2), dtype=int)  # [wall, (+y, -y)]
    for r in rows:
        wi = walls.index(r["wall"]) if r["wall"] in walls else None
        if wi is None:
            continue
        col = 0 if r["sign"] >= 0 else 1
        conf[wi, col] += 1

    # ---- agreement: only left/right walls have a defined "away" side ----
    lat_rows = [r for r in rows if r["wall"] in ("left", "right")]
    n_lat = len(lat_rows)
    n_agree = sum(1 for r in lat_rows if away_correct(r["wall"], r["sign"]))
    agree_frac = (n_agree / n_lat) if n_lat else float("nan")
    n_top = sum(1 for r in rows if r["wall"] == "top")
    n_fallback = sum(1 for r in rows if not r["used_deflect_phase"])

    THIN = n_lat < 15  # flag if data is thin

    # =====================  FIGURE  =====================
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 6.2))

    # ---------- LEFT PANEL: confusion matrix ----------
    im = axL.imshow(conf, cmap="Blues", aspect="auto", vmin=0)
    axL.set_xticks(range(2)); axL.set_xticklabels(dir_labels)
    axL.set_yticks(range(len(walls)))
    axL.set_yticklabels([f"wall = {w}" for w in walls])
    axL.set_xlabel("Observed lateral deflection direction (world +y / -y)")
    axL.set_ylabel("Protrusion wall side")
    axL.set_title("Wall side  vs  observed deflect direction\n(green outline = AWAY from wall = correct)")

    # highlight the "away from wall" cell for left/right and annotate counts
    # away cell: left->-y (col1); right->+y (col0)
    away_cell = {"left": 1, "right": 0}
    for wi, w in enumerate(walls):
        for col in range(2):
            val = conf[wi, col]
            is_away = (w in away_cell and away_cell[w] == col)
            txtcolor = "white" if val > conf.max() * 0.6 else "black"
            axL.text(col, wi, str(val), ha="center", va="center",
                     fontsize=18, fontweight="bold", color=txtcolor)
            if is_away:
                axL.add_patch(plt.Rectangle((col - 0.5, wi - 0.5), 1, 1,
                              fill=False, edgecolor="green", lw=3.5))
            elif w in away_cell:  # the toward-wall (wrong) cell
                axL.add_patch(plt.Rectangle((col - 0.5, wi - 0.5), 1, 1,
                              fill=False, edgecolor="red", lw=2.0, linestyle="--"))
    # mark top row as N/A (vertical protrusion)
    axL.text(0.5, walls.index("top"), "(vertical wall: no defined L/R side)",
             ha="center", va="center", fontsize=8.5, style="italic",
             color="dimgray", transform=axL.transData, alpha=0.0)  # placeholder; real note below
    fig.colorbar(im, ax=axL, fraction=0.046, pad=0.04, label="episode count")

    # ---------- RIGHT PANEL: signed excursion scatter / strip ----------
    # x category by wall, y signed peak excursion (cm); color = correct/incorrect
    cm_per_m = 100.0
    cat_x = {"left": 0, "right": 1, "top": 2}
    rng = np.random.default_rng(0)
    plotted_corr = plotted_inc = plotted_na = False
    for r in rows:
        x = cat_x[r["wall"]] + rng.uniform(-0.16, 0.16)
        yv = r["excursion"] * cm_per_m
        corr = away_correct(r["wall"], r["sign"])
        marker = "o" if r["used_deflect_phase"] else "^"
        if corr is True:
            c = "tab:green"; lbl = "AWAY from wall (correct)" if not plotted_corr else None; plotted_corr = True
        elif corr is False:
            c = "tab:red"; lbl = "TOWARD wall (incorrect)" if not plotted_inc else None; plotted_inc = True
        else:
            c = "tab:gray"; lbl = "top wall (no L/R side)" if not plotted_na else None; plotted_na = True
        axR.scatter(x, yv, s=70, color=c, edgecolor="black", linewidth=0.6,
                    marker=marker, alpha=0.85, label=lbl, zorder=3)

    axR.axhline(0, color="black", lw=1.0)
    axR.set_xticks([0, 1, 2])
    axR.set_xticklabels(["wall = left\n(protr on +y)", "wall = right\n(protr on -y)", "wall = top\n(vertical)"])
    axR.set_ylabel("Signed peak lateral excursion  $\\Delta y$  (cm, world frame)")
    axR.set_title("Lateral excursion per episode\n(+y = up, -y = down; expert should flee the wall)")
    axR.grid(axis="y", alpha=0.3)
    # arrows showing the 'away' direction expected for left/right
    yl = axR.get_ylim()
    span = yl[1] - yl[0]
    axR.annotate("away\n(-y)", xy=(0, yl[0] + 0.10 * span), ha="center",
                 fontsize=9, color="green", fontweight="bold")
    axR.annotate("away\n(+y)", xy=(1, yl[1] - 0.10 * span), ha="center",
                 fontsize=9, color="green", fontweight="bold")
    # legend with marker meaning
    handles, labels = axR.get_legend_handles_labels()
    extra = [plt.Line2D([], [], marker="o", color="w", markerfacecolor="gray",
                        markeredgecolor="k", markersize=8, label="deflect-phase samples"),
             plt.Line2D([], [], marker="^", color="w", markerfacecolor="gray",
                        markeredgecolor="k", markersize=8, label="fallback: whole episode")]
    axR.legend(handles=handles + extra, loc="lower right", fontsize=8.5, framealpha=0.9)

    # ---------- super title + summary annotation ----------
    fig.suptitle("PROBE 2 - Deflection-side selection: does the expert steer AWAY from the wall?",
                 fontsize=14, fontweight="bold")

    summary = (f"Lateral (left/right) deflect episodes: n = {n_lat}    "
               f"AWAY-from-wall agreement: {n_agree}/{n_lat} = "
               f"{agree_frac*100:.0f}%" if n_lat else "No lateral deflect episodes.")
    notes = (f"Total deflect episodes: {n_total}  "
             f"(left {sum(r['wall']=='left' for r in rows)}, "
             f"right {sum(r['wall']=='right' for r in rows)}, top {n_top} excluded: vertical).  "
             f"{n_fallback} used whole-episode fallback (no deflect-phase samples).")
    boxcolor = "mistyrose" if (THIN or (n_lat and agree_frac < 0.6)) else "honeydew"
    fig.text(0.5, 0.015,
             summary + "\n" + notes +
             ("\n*** DATA IS THIN (n<15 lateral episodes): treat agreement as indicative only. ***" if THIN else ""),
             ha="center", va="bottom", fontsize=10,
             bbox=dict(boxstyle="round,pad=0.5", facecolor=boxcolor, edgecolor="gray"))

    fig.tight_layout(rect=[0, 0.085, 1, 0.95])
    fig.savefig(OUTPNG, dpi=150)
    plt.close(fig)

    # ---- verify ----
    sz = os.path.getsize(OUTPNG)
    print(f"WROTE {OUTPNG}  ({sz} bytes)")
    print(f"n_total={n_total} n_lat={n_lat} agree={n_agree} frac={agree_frac:.3f} top={n_top} fallback={n_fallback} THIN={THIN}")
    return OUTPNG, sz, n_lat, n_agree, agree_frac, n_total, n_top, THIN


if __name__ == "__main__":
    main()
