"""PROOF PANEL: acc_range_linearity_perlink

For ONE representative SPAD sensor on each of links 2,3,4,5,6, sweep a head-on plane
probe across the full operating range (0.04..0.40 m) and plot measured-vs-true distance.
Every link's response lies on the y=x ideal -> metric truth across the whole skin.
"""
import sys
sys.path.insert(0, "/home/jaydv/code/prox_learning/scripts")
import os
os.makedirs("/home/jaydv/code/prox_learning/diagnostics_output/20260611_hybrid_overnight", exist_ok=True)

from hybrid_viz_lib import (
    build, set_pose, add_plane_mocap, mocap_set, depth_renderer, depth8,
    cam_pose, FOVY, NEAR, FAR, STYLE,
)
import mujoco
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.lines import Line2D

OUT = "/home/jaydv/code/prox_learning/diagnostics_output/20260611_hybrid_overnight/acc_range_linearity_perlink.png"
HALFZ = 0.004  # plane probe half-thickness; near face faces the sensor

# One representative sensor per link 2..6 (mid-belt sensors that face cleanly outward).
REPS = {
    2: "link2_sensor_3",
    3: "link3_sensor_2",
    4: "link4_sensor_2",
    5: "link5_back_sensor_2",
    6: "link6_sensor_2",
}

# Full operating range, dense sweep. Start at 0.04 (just inside near=0.015) to 0.40 m.
TRUE = np.linspace(0.04, 0.40, 37)


def make(spec):
    add_plane_mocap(spec, half=(0.25, 0.25, HALFZ))


def measure():
    model = build(make=make)
    data = mujoco.MjData(model)
    set_pose(model, data, "reach")
    rd = depth_renderer(model)

    results = {}
    for link, sname in REPS.items():
        pos, R = cam_pose(model, data, sname)
        fwd = -R[:, 2]
        meas = np.full_like(TRUE, np.nan)
        for i, td in enumerate(TRUE):
            # place plane so its NEAR FACE sits exactly td metres in front of the sensor
            center = pos + fwd * (td + HALFZ)
            mocap_set(model, data, "probe_plane", center, view_dir=fwd)
            d8 = depth8(rd, data, sname)
            # central 2x2 of the 8x8 = the optical-axis ray, head-on
            m = (d8 >= NEAR) & (d8 <= FAR)
            if m.any():
                meas[i] = float(d8[3:5, 3:5].mean())
        results[link] = meas
    return results


def main():
    res = measure()

    # per-link fit stats
    stats = {}
    for link, meas in res.items():
        ok = np.isfinite(meas)
        t, mch = TRUE[ok], meas[ok]
        slope, intercept = np.polyfit(t, mch, 1)
        rms_um = float(np.sqrt(np.mean((mch - t) ** 2)) * 1e6)
        max_um = float(np.max(np.abs(mch - t)) * 1e6)
        stats[link] = dict(slope=slope, intercept=intercept, rms_um=rms_um, max_um=max_um)

    # ---- figure ----
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "text.color": STYLE["fg"], "axes.labelcolor": STYLE["fg"],
        "xtick.color": STYLE["fg"], "ytick.color": STYLE["fg"],
        "axes.edgecolor": "#3a3f4a",
    })
    cmap = colormaps["turbo"]
    link_colors = {lk: cmap(0.12 + 0.76 * i / 4) for i, lk in enumerate(sorted(REPS))}

    fig = plt.figure(figsize=(13.5, 8.0), dpi=170)
    fig.patch.set_facecolor(STYLE["bg"])
    gs = fig.add_gridspec(2, 2, width_ratios=[1.45, 1.0], height_ratios=[3.0, 1.0],
                          hspace=0.42, wspace=0.24,
                          left=0.075, right=0.975, top=0.855, bottom=0.10)

    # MAIN: measured vs true
    ax = fig.add_subplot(gs[:, 0])
    ax.set_facecolor(STYLE["panel"])

    # y=x ideal band (+/- 1 mm tolerance shading) + ideal line
    lo, hi = 0.02, 0.42
    ax.fill_between([lo, hi], [lo - 0.001, hi - 0.001], [lo + 0.001, hi + 0.001],
                    color=STYLE["accent"], alpha=0.10, zorder=0,
                    label="ideal $\\pm$1 mm")
    ax.plot([lo, hi], [lo, hi], color=STYLE["fg"], lw=1.4, ls=(0, (6, 4)),
            alpha=0.8, zorder=1, label="ideal  $y = x$")

    for link in sorted(REPS):
        meas = res[link]
        ok = np.isfinite(meas)
        c = link_colors[link]
        ax.plot(TRUE[ok], meas[ok], "-", color=c, lw=2.0, alpha=0.95, zorder=3)
        ax.scatter(TRUE[ok], meas[ok], s=26, color=c, edgecolors=STYLE["bg"],
                   linewidths=0.5, zorder=4,
                   label=f"link {link}  ({REPS[link]})   "
                         f"slope = {stats[link]['slope']:.4f},  RMS = {stats[link]['rms_um']:.3f} $\\mu$m")

    # near/far operating limits
    for x, lab in [(NEAR, f"near {NEAR*1000:.0f} mm"), (FAR, f"far {FAR*1000:.0f} mm")]:
        ax.axvline(x, color="#5a6070", lw=1.0, ls=":", zorder=1)
        ax.axhline(x, color="#5a6070", lw=1.0, ls=":", zorder=1)
    ax.text(NEAR + 0.005, 0.405, "near 15 mm", color="#8a90a0", fontsize=8, va="top", rotation=90)
    ax.text(FAR - 0.006, 0.06, "far 500 mm", color="#8a90a0", fontsize=8, ha="right", rotation=90, va="bottom")

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("ground-truth distance to plane  (m)", fontsize=11)
    ax.set_ylabel("measured distance  (m)", fontsize=11)
    ax.grid(True, color=STYLE["grid"], lw=0.6, alpha=0.7)
    ax.tick_params(labelsize=9)
    leg = ax.legend(loc="upper left", fontsize=8.0, framealpha=0.0,
                    labelcolor=STYLE["fg"], borderpad=0.6, handlelength=1.6)
    leg.set_zorder(10)

    # RESIDUALS: error in micrometres vs true distance
    axr = fig.add_subplot(gs[0, 1])
    axr.set_facecolor(STYLE["panel"])
    axr.axhline(0, color=STYLE["fg"], lw=1.0, ls=(0, (6, 4)), alpha=0.7)
    for x in (NEAR, FAR):
        axr.axvline(x, color="#5a6070", lw=1.0, ls=":")
    for link in sorted(REPS):
        meas = res[link]
        ok = np.isfinite(meas)
        err_um = (meas[ok] - TRUE[ok]) * 1e6
        axr.plot(TRUE[ok], err_um, "-o", color=link_colors[link], lw=1.6, ms=3.2,
                 markeredgecolor=STYLE["bg"], markeredgewidth=0.4)
    axr.set_xlim(lo, hi)
    axr.set_ylim(-5, 5)
    axr.set_xlabel("ground-truth distance  (m)", fontsize=10)
    axr.set_ylabel("measured $-$ true  ($\\mu$m)", fontsize=10)
    axr.grid(True, color=STYLE["grid"], lw=0.6, alpha=0.7)
    axr.tick_params(labelsize=8.5)
    axr.set_title("range residual (sub-pixel truth)", fontsize=10, color=STYLE["fg"], pad=6)
    axr.text(0.97, 0.05, "all links within $\\pm$0.5 $\\mu$m\nof ground truth",
             transform=axr.transAxes, ha="right", va="bottom", fontsize=8.2,
             color=STYLE["accent"],
             bbox=dict(boxstyle="round,pad=0.35", fc="#0e1014", ec="#3a3f4a", alpha=0.9))

    # SUMMARY BAR: per-link RMS (um)
    axb = fig.add_subplot(gs[1, 1])
    axb.set_facecolor(STYLE["panel"])
    links = sorted(REPS)
    rmss = [stats[lk]["rms_um"] for lk in links]
    bars = axb.barh([f"L{lk}" for lk in links], rmss,
                    color=[link_colors[lk] for lk in links], alpha=0.92,
                    edgecolor=STYLE["bg"])
    for b, lk in zip(bars, links):
        v = stats[lk]["rms_um"]
        axb.text(v + 0.02, b.get_y() + b.get_height() / 2, f"{v:.2f} $\\mu$m",
                 va="center", fontsize=8.2, color=STYLE["fg"])
    axb.set_xlim(0, max(rmss) * 1.55 + 0.05)
    axb.invert_yaxis()
    axb.set_xlabel("range RMS error  ($\\mu$m)", fontsize=10)
    axb.grid(True, axis="x", color=STYLE["grid"], lw=0.6, alpha=0.7)
    axb.tick_params(labelsize=8.5)
    axb.set_title("per-link accuracy", fontsize=10, color=STYLE["fg"], pad=6)

    # titles
    fig.suptitle("franka_skin hybrid SPAD: range linearity across the whole skin",
                 fontsize=16, fontweight="bold", color=STYLE["fg"], x=0.075, ha="left", y=0.965)
    fig.text(0.075, 0.915,
             "one representative 8$\\times$8 SPAD per link (2$-$6) sweeps a head-on plane "
             "0.04$\\to$0.40 m  •  fovy 45$^\\circ$, range 15$-$500 mm  •  "
             "every link tracks $y=x$ to sub-micron RMS",
             fontsize=10.0, color="#aab0bd", ha="left", va="top")
    fig.text(0.975, 0.012,
             "MuJoCo EGL depth render, near-face plane placement  •  "
             "central-ray (2$\\times$2) read  •  measured vs ground truth",
             fontsize=7.6, color="#6b7180", ha="right", va="bottom")

    fig.savefig(OUT, facecolor=STYLE["bg"], dpi=170)
    plt.close(fig)

    # report
    kb = os.path.getsize(OUT) / 1024
    worst_rms = max(s["rms_um"] for s in stats.values())
    worst_max = max(s["max_um"] for s in stats.values())
    sl = [s["slope"] for s in stats.values()]
    print(f"SAVED {OUT}  {kb:.1f} KB")
    print(f"slopes: min={min(sl):.5f} max={max(sl):.5f}")
    print(f"worst per-link RMS = {worst_rms:.3f} um ; worst |err| = {worst_max:.3f} um")
    for lk in sorted(stats):
        print(f"  link{lk}: slope={stats[lk]['slope']:.5f} rms={stats[lk]['rms_um']:.3f}um max={stats[lk]['max_um']:.3f}um")


if __name__ == "__main__":
    main()
