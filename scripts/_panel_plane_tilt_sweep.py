"""plane_tilt_sweep panel: one SPAD sensor, a flat plane at fixed 0.18 m tilted
0/15/30/45 deg about the sensor's local x-axis. Proves the 8x8 depth gradient grows
with tilt and the back-projected cloud reconstructs a true flat tilted plane (RMS ~ 0)."""
import os, sys
sys.path.insert(0, "/home/jaydv/code/prox_learning/scripts")
from hybrid_viz_lib import (build, set_pose, sensors, depth8, backproject, cam_pose,
    add_plane_mocap, mocap_set, depth_renderer, fit_plane, FOVY, NEAR, FAR, STYLE)
import mujoco, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors as mcolors
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

OUTDIR = "/home/jaydv/code/prox_learning/diagnostics_output/20260611_hybrid_viz_suite"
os.makedirs(OUTDIR, exist_ok=True)
OUT = os.path.join(OUTDIR, "plane_tilt_sweep.png")

TILTS = [0, 15, 30, 45]
D = 0.18                     # fixed plane distance (m)
SENSOR = "link6_sensor_0"    # a SPAD that gets a full clean return at 0.18 m


def rodrigues(v, k, a):
    """Rotate vector v about unit axis k by angle a (rad)."""
    k = k / (np.linalg.norm(k) + 1e-12)
    return v * np.cos(a) + np.cross(k, v) * np.sin(a) + k * np.dot(k, v) * (1 - np.cos(a))


def make(spec):
    add_plane_mocap(spec, "probe_plane", half=(0.13, 0.13, 0.004), rgba=(0.22, 0.85, 0.42, 1))


def main():
    model = build(make)
    data = mujoco.MjData(model)
    set_pose(model, data, "reach")
    assert len(sensors(model)) == 40, "expected 40 SPAD sensors"
    rd = depth_renderer(model)

    pos, R = cam_pose(model, data, SENSOR)
    fwd = -R[:, 2]            # camera optical axis (looks along -z)
    xax = R[:, 0]            # sensor local x-axis -> the tilt hinge

    # ---- collect data for all tilts (shared color scale across the row) ----
    runs = []
    dmin_all, dmax_all = np.inf, -np.inf
    for tilt in TILTS:
        a = np.deg2rad(tilt)
        view = rodrigues(fwd, xax, a)               # tilt the plane normal about x
        center = pos + fwd * D                      # plane centre stays at 0.18 m
        mocap_set(model, data, "probe_plane", center, view_dir=view)
        d8 = depth8(rd, data, SENSOR)
        mask = (d8 >= NEAR) & (d8 <= FAR)
        pts, d = backproject(d8, pos, R)
        c, n, rms = fit_plane(pts)
        dmin_all = min(dmin_all, d.min())
        dmax_all = max(dmax_all, d.max())
        runs.append(dict(tilt=tilt, d8=d8, mask=mask, pts=pts, d=d,
                         c=c, n=n, rms=rms, npix=int(mask.sum())))

    norm = mcolors.Normalize(vmin=dmin_all, vmax=dmax_all)
    cmap = matplotlib.colormaps[STYLE["cmap"]]

    # ---- figure scaffolding (dark robotics-lab theme) ----
    plt.rcParams.update({
        "text.color": STYLE["fg"], "axes.labelcolor": STYLE["fg"],
        "xtick.color": STYLE["fg"], "ytick.color": STYLE["fg"],
        "axes.edgecolor": STYLE["grid"], "font.family": "DejaVu Sans",
    })
    fig = plt.figure(figsize=(16.5, 8.6), facecolor=STYLE["bg"])
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.35],
                          left=0.045, right=0.92, top=0.86, bottom=0.07,
                          hspace=0.30, wspace=0.28)

    for j, run in enumerate(runs):
        # ===== TOP ROW: 8x8 depth heatmap =====
        axh = fig.add_subplot(gs[0, j], facecolor=STYLE["panel"])
        disp = np.where(run["mask"], run["d8"], np.nan)
        im = axh.imshow(disp, cmap=cmap, norm=norm, origin="upper",
                        interpolation="nearest")
        axh.set_xticks([]); axh.set_yticks([])
        for s in axh.spines.values():
            s.set_color(STYLE["grid"])
        # annotate each cell with its depth in mm
        for r in range(8):
            for ccol in range(8):
                if run["mask"][r, ccol]:
                    val = run["d8"][r, ccol]
                    tc = "#0a0a0a" if 0.30 < norm(val) < 0.78 else STYLE["fg"]
                    axh.text(ccol, r, f"{val*1000:.0f}", ha="center", va="center",
                             fontsize=5.4, color=tc)
        span = (run["d"].max() - run["d"].min()) * 1000
        axh.set_title(f"tilt {run['tilt']}°   ·   {run['npix']}/64 px",
                      color=STYLE["accent"], fontsize=12.5, pad=7, fontweight="bold")
        axh.set_xlabel(f"depth span {span:.0f} mm", fontsize=9.5,
                       color=STYLE["fg"], labelpad=3)

        # ===== BOTTOM ROW: back-projected cloud lying ON the tilted plane (3D) =====
        ax3 = fig.add_subplot(gs[1, j], projection="3d", facecolor=STYLE["panel"])
        pts, dd = run["pts"], run["d"]
        cols = cmap(norm(dd))

        # draw the fitted plane patch so the eye sees the cloud lies on a flat tilt
        c, n = run["c"], run["n"]
        if n @ (pos - c) < 0:    # orient normal toward the sensor
            n = -n
        # build an in-plane basis
        t0 = np.array([1.0, 0, 0])
        if abs(n @ t0) > 0.9:
            t0 = np.array([0, 1.0, 0])
        e1 = np.cross(n, t0); e1 /= np.linalg.norm(e1)
        e2 = np.cross(n, e1)
        # extent of the cloud in plane coords
        rel = pts - c
        s1 = (rel @ e1); s2 = (rel @ e2)
        pad = 0.012
        g1 = np.linspace(s1.min() - pad, s1.max() + pad, 2)
        g2 = np.linspace(s2.min() - pad, s2.max() + pad, 2)
        G1, G2 = np.meshgrid(g1, g2)
        PX = c[0] + G1 * e1[0] + G2 * e2[0]
        PY = c[1] + G1 * e1[1] + G2 * e2[1]
        PZ = c[2] + G1 * e1[2] + G2 * e2[2]
        ax3.plot_surface(PX, PY, PZ, color="#3a4150", alpha=0.28,
                         linewidth=0, shade=False, zorder=1)

        ax3.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=cols, s=26,
                    depthshade=False, edgecolors="#0c0e12", linewidths=0.25, zorder=3)
        # short optical-ray stub from the cloud centroid back toward the sensor,
        # so the viewer can read which way the sensor looks
        ray_to = c - 0.05 * (c - pos) / (np.linalg.norm(c - pos) + 1e-9)
        ax3.plot([c[0], ray_to[0]], [c[1], ray_to[1]], [c[2], ray_to[2]],
                 color=STYLE["accent"], lw=1.4, ls="--", alpha=0.7, zorder=4)

        # tight equal-ish cube around the cloud
        ctr = pts.mean(0)
        rng = max(0.045, (pts.max(0) - pts.min(0)).max() * 0.62)
        ax3.set_xlim(ctr[0]-rng, ctr[0]+rng)
        ax3.set_ylim(ctr[1]-rng, ctr[1]+rng)
        ax3.set_zlim(ctr[2]-rng, ctr[2]+rng)
        ax3.set_box_aspect((1, 1, 1))
        # view roughly edge-on to the sensor x-axis so the plane tilt is visible
        ax3.view_init(elev=10, azim=-72)
        ax3.set_title(f"cloud RMS = {run['rms']*1000:.3f} mm",
                      color=STYLE["fg"], fontsize=12, pad=-2, fontweight="bold")
        for axis in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
            axis.set_pane_color((0.09, 0.10, 0.12, 1.0))
            axis.line.set_color(STYLE["grid"])
            axis.label.set_color(STYLE["fg"])
        ax3.tick_params(colors=STYLE["fg"], labelsize=6.5, pad=-2)
        ax3.grid(True, color=STYLE["grid"], alpha=0.4)
        ax3.set_xlabel("x (m)", fontsize=8, labelpad=-6)
        ax3.set_ylabel("y (m)", fontsize=8, labelpad=-6)
        ax3.set_zlabel("z (m)", fontsize=8, labelpad=-6)

    # shared colorbar
    cax = fig.add_axes([0.935, 0.07, 0.013, 0.79])
    sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("distance (m)", color=STYLE["fg"], fontsize=11)
    cb.ax.yaxis.set_tick_params(color=STYLE["fg"], labelsize=9)
    plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color=STYLE["fg"])
    cb.outline.set_edgecolor(STYLE["grid"])

    rms_max = max(r["rms"] for r in runs) * 1000
    fig.suptitle(
        "franka_skin HYBRID SPAD  ·  plane-tilt sweep:  one 8×8 depth sensor (fovy 45°, "
        "0.015–0.5 m)  vs.  a flat plane at fixed 0.18 m",
        color=STYLE["fg"], fontsize=15.5, fontweight="bold", y=0.975)
    fig.text(0.045, 0.915,
             f"sensor '{SENSOR}'  ·  plane tilted 0–45° about the sensor's local x-axis  ·  "
             f"top: per-cell depth (mm) — gradient grows with tilt   bottom: back-projected cloud "
             f"reconstructs a true flat tilted plane  (max fit RMS {rms_max:.3f} mm)",
             color="#aab0bb", fontsize=10.5)

    fig.savefig(OUT, dpi=170, facecolor=STYLE["bg"], bbox_inches="tight")
    plt.close(fig)

    sz = os.path.getsize(OUT)
    spans = [(r["d"].max()-r["d"].min())*1000 for r in runs]
    print("SAVED", OUT, sz)
    print("spans_mm", [round(s, 1) for s in spans])
    print("rms_mm", [round(r["rms"]*1000, 4) for r in runs])
    print("npix", [r["npix"] for r in runs])
    return OUT, sz, runs


if __name__ == "__main__":
    main()
