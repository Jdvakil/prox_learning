"""range_accuracy_scatter — prove the franka_skin HYBRID 8x8 SPAD proximity sensors read
metric truth. For ~12 sensors x 6 head-on plane distances (0.05..0.35 m) we place a flat
mocap plane perpendicular to each sensor's optical axis at a known true distance, read the
center-pixel depth, and scatter measured vs true on a y=x plot. Line fit + residual histogram.
"""
import os, sys
sys.path.insert(0, "/home/jaydv/code/prox_learning/scripts")
from hybrid_viz_lib import (build, set_pose, sensors, depth8, cam_pose,
                            add_plane_mocap, mocap_set, depth_renderer,
                            FOVY, NEAR, FAR, STYLE)
import mujoco, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

OUTDIR = "/home/jaydv/code/prox_learning/diagnostics_output/20260611_hybrid_viz_suite"
os.makedirs(OUTDIR, exist_ok=True)
OUT = os.path.join(OUTDIR, "range_accuracy_scatter.png")

HALF_T = 0.004                       # probe-plane half thickness; surface sits HALF_T in front of center
TRUE = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35])   # 6+ head-on distances (m)
N_SENSORS = 12


def make(spec):
    # The SPAD honest range is 0.015-0.5 m, but MuJoCo's default offscreen near-clip
    # (znear*extent ~ 0.106 m) silently clips anything closer. Pull znear in so the depth
    # buffer faithfully resolves the full proximity band instead of dropping near returns.
    spec.visual.map.znear = 0.0008
    add_plane_mocap(spec, "probe_plane", half=(0.30, 0.30, HALF_T), rgba=(0.20, 0.85, 0.40, 1))


def center_depth(d8):
    """Center-distance readout: mean of the central 2x2 of the 8x8 SPAD frame (m)."""
    return float(d8[3:5, 3:5].mean())


def measure():
    model = build(make)
    data = mujoco.MjData(model)
    set_pose(model, data, "reach")
    names = sensors(model)
    rd = depth_renderer(model)

    # pick a spread of sensors across the arm links so it's not all from one body
    pick = [names[i] for i in np.linspace(0, len(names) - 1, N_SENSORS).round().astype(int)]
    pick = list(dict.fromkeys(pick))   # dedupe, keep order

    rows = []   # (sensor, link_idx, true, meas)
    for si, name in enumerate(pick):
        pos, R = cam_pose(model, data, name)
        fwd = -R[:, 2]                 # outward optical axis
        link = name.split("_sensor_")[0]
        for td in TRUE:
            # place plane so its FRONT FACE (not center) is exactly td from the sensor
            mocap_set(model, data, "probe_plane", pos + fwd * (td + HALF_T), view_dir=fwd)
            d8 = depth8(rd, data, name)
            if d8.min() > FAR:         # no return (clipped/occluded) -> skip honestly
                continue
            rows.append((name, link, td, center_depth(d8)))
    return pick, rows


def main():
    pick, rows = measure()
    true = np.array([r[2] for r in rows])
    meas = np.array([r[3] for r in rows])
    links = np.array([r[1] for r in rows])
    resid_mm = (meas - true) * 1000.0

    # least-squares line fit meas = a*true + b
    a, b = np.polyfit(true, meas, 1)
    pred = a * true + b
    ss_res = np.sum((meas - pred) ** 2)
    ss_tot = np.sum((meas - meas.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    rms_mm = float(np.sqrt(np.mean(resid_mm ** 2)))
    max_abs_mm = float(np.max(np.abs(resid_mm)))
    n_sens = len(set(r[0] for r in rows))

    # ---- figure ----------------------------------------------------------
    s = STYLE
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "text.color": s["fg"], "axes.labelcolor": s["fg"],
        "xtick.color": s["fg"], "ytick.color": s["fg"],
        "axes.edgecolor": s["grid"],
    })
    fig = plt.figure(figsize=(13.5, 7.2), facecolor=s["bg"])
    gs = GridSpec(2, 2, width_ratios=[1.55, 1.0], height_ratios=[1.0, 0.62],
                  wspace=0.24, hspace=0.42, left=0.075, right=0.965,
                  top=0.84, bottom=0.10)

    # main scatter: measured vs true, colored by true distance (turbo_r)
    ax = fig.add_subplot(gs[:, 0])
    ax.set_facecolor(s["panel"])
    cmap = matplotlib.colormaps[s["cmap"]]
    norm = matplotlib.colors.Normalize(vmin=TRUE.min(), vmax=TRUE.max())

    lo, hi = 0.0, 0.40
    ax.plot([lo, hi], [lo, hi], color=s["fg"], lw=1.3, ls="--", alpha=0.55,
            zorder=1, label="ideal  y = x")
    xx = np.linspace(lo, hi, 50)
    ax.plot(xx, a * xx + b, color=s["accent"], lw=2.0, zorder=2,
            label=f"fit  y = {a:.4f}x {b*1e3:+.3f} mm")

    sc = ax.scatter(true * 1000, meas * 1000, c=true, cmap=cmap, norm=norm,
                    s=95, edgecolor="#0c0e12", linewidth=0.8, zorder=4, alpha=0.95)
    ax.set_xlim(0, 400); ax.set_ylim(0, 400)
    ax.set_aspect("equal")
    ax.set_xlabel("true plane distance  (mm)")
    ax.set_ylabel("measured center distance  (mm)")
    ax.grid(True, color=s["grid"], lw=0.6, alpha=0.7)
    for sp in ax.spines.values():
        sp.set_color(s["grid"])
    leg = ax.legend(loc="upper left", framealpha=0.0, fontsize=10.5)
    for t in leg.get_texts():
        t.set_color(s["fg"])
    ax.set_title(f"{n_sens} SPAD sensors x {len(TRUE)} head-on distances  "
                 f"(N = {len(rows)} readings)", color=s["fg"], fontsize=12, pad=9)

    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
                      ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("distance (m)", color=s["fg"])
    cb.ax.yaxis.set_tick_params(color=s["fg"])
    cb.outline.set_edgecolor(s["grid"])
    plt.setp(plt.getp(cb.ax, "yticklabels"), color=s["fg"])

    # residual histogram (measured - true), in micrometers (residuals are sub-micron)
    resid_um = resid_mm * 1000.0
    axh = fig.add_subplot(gs[0, 1])
    axh.set_facecolor(s["panel"])
    rng = max(np.abs(resid_um).max() * 1.15, 1e-2)
    bins = np.linspace(-rng, rng, 31)
    axh.hist(resid_um, bins=bins, color=s["accent"], edgecolor="#0c0e12", alpha=0.9)
    axh.axvline(0, color=s["fg"], lw=1.1, ls="--", alpha=0.6)
    axh.set_xlabel("residual  measured - true  (µm)")
    axh.set_ylabel("count")
    axh.grid(True, color=s["grid"], lw=0.5, alpha=0.6, axis="y")
    for sp in axh.spines.values():
        sp.set_color(s["grid"])
    axh.set_title("residual distribution", color=s["fg"], fontsize=11, pad=6)
    axh.text(0.97, 0.92,
             f"mean {resid_um.mean():+.3f} µm\nstd  {resid_um.std():.3f} µm",
             transform=axh.transAxes, ha="right", va="top", fontsize=9.5,
             color=s["fg"], family="monospace",
             bbox=dict(boxstyle="round", fc=s["bg"], ec=s["grid"], alpha=0.85))

    # residual vs distance (shows there's no range-dependent bias)
    axr = fig.add_subplot(gs[1, 1])
    axr.set_facecolor(s["panel"])
    axr.scatter(true * 1000, resid_um, c=true, cmap=cmap, norm=norm,
                s=42, edgecolor="#0c0e12", linewidth=0.5, alpha=0.9)
    axr.axhline(0, color=s["fg"], lw=1.0, ls="--", alpha=0.55)
    axr.set_xlabel("true distance  (mm)")
    axr.set_ylabel("resid (µm)")
    axr.set_xlim(0, 400)
    axr.grid(True, color=s["grid"], lw=0.5, alpha=0.6)
    for sp in axr.spines.values():
        sp.set_color(s["grid"])
    axr.set_title("residual vs range", color=s["fg"], fontsize=10.5, pad=5)

    fig.suptitle("franka_skin HYBRID  —  SPAD proximity sensors read metric truth",
                 color=s["fg"], fontsize=16, fontweight="bold", x=0.5, y=0.965)
    fig.text(0.5, 0.905,
             f"slope = {a:.5f}    R² = {r2:.8f}    RMS error = {rms_mm*1000:.3f} µm    "
             f"max |err| = {max_abs_mm*1000:.3f} µm   |   8×8 depth, fovy={FOVY:g}°, "
             f"range {NEAR}-{FAR} m  (residual at depth-buffer floor)",
             color=s["accent"], fontsize=12, ha="center", family="monospace")

    fig.savefig(OUT, dpi=170, facecolor=s["bg"], bbox_inches="tight")
    plt.close(fig)

    print("SLOPE", a, "R2", r2, "RMS_mm", rms_mm, "MAX_mm", max_abs_mm,
          "N", len(rows), "SENSORS", n_sens)
    sz = os.path.getsize(OUT)
    print("WROTE", OUT, sz, "bytes")
    return OUT, sz, a, r2, rms_mm, max_abs_mm, len(rows), n_sens


if __name__ == "__main__":
    main()
