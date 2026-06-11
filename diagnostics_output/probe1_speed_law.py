#!/usr/bin/env python
"""PROBE 1 - the speed law.

For every timestep of every episode across all three scenes (fumehood/panel/cubby):
  - TCP speed (cm/s) from consecutive tcp_pose positions / dt
  - min skin distance (m) = min over all 29 sensors of that sensor's (8,8) min,
    ignoring no-return values (>2.0 m).

Scatter speed (y) vs min-dist (x), one subplot per scene + one pooled.
Point color = behavior_class. Linear fit + Pearson r + slope in each subplot title.

Headline evidence: "arm slows when skin sees something close".
"""
import os
import glob
import json
import warnings

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

OUTDIR = "/home/jaydv/code/prox_learning/diagnostics_output/20260610_plots_v3"
os.makedirs(OUTDIR, exist_ok=True)
OUT_PNG = os.path.join(OUTDIR, "speed_vs_prox.png")

RUN_DIRS = {
    "fumehood": "/home/jaydv/code/prox_learning/assets/datagen/fumehood_smoke/FrankaSkinFumehoodSmokeConfig/20260610_090517",
    "panel":    "/home/jaydv/code/prox_learning/assets/datagen/panel_slalom_smoke/FrankaSkinPanelSlalomSmokeConfig/20260610_092607",
    "cubby":    "/home/jaydv/code/prox_learning/assets/datagen/cubby_smoke/FrankaSkinCubbySmokeConfig/20260610_095739",
}

NO_RETURN = 2.0  # values > this mean no return -> mask out
DEFAULT_DT = 0.066  # s

# behavior_class -> color
BEHAV_COLORS = {"free": "#2ca02c", "deflect": "#ff7f0e", "abort": "#d62728"}
BEHAV_ORDER = ["free", "deflect", "abort"]


def decode_scene(t):
    """Decode the obs_scene JSON blob -> dict (or {} on failure)."""
    ds = t["obs_scene"]
    blob = ds[()] if ds.shape == () else ds[0]
    s = (blob.tobytes() if hasattr(blob, "tobytes") else blob).decode("utf-8", "ignore").rstrip("\x00")
    return json.loads(s)


def min_skin_distance_series(t, sensor_names):
    """Return (T,) array of min skin distance (m) over all sensors, masking no-return."""
    prox = t["obs/proximity"]
    T = None
    per_sensor_min = []
    for sn in sensor_names:
        if sn not in prox:
            continue
        arr = prox[sn][()]  # (T,4,8,8)
        if T is None:
            T = arr.shape[0]
        frame = arr.mean(axis=1)  # (T,8,8) average the 4 sub-frames
        # mask no-return
        masked = np.where(frame > NO_RETURN, np.nan, frame)
        # min over the 8x8 grid for each timestep -> (T,)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            smin = np.nanmin(masked.reshape(masked.shape[0], -1), axis=1)
        per_sensor_min.append(smin)
    if not per_sensor_min:
        return None
    stacked = np.vstack(per_sensor_min)  # (n_sensors, T)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mind = np.nanmin(stacked, axis=0)  # (T,) ; nan where ALL sensors no-return
    return mind


def tcp_speed_series(t, dt_s):
    """Return (T,) array of TCP speed in cm/s. speed[0] aligned to mind[1:] semantics:
    we compute speed between t-1 and t, assign to step t (so speed[0]=nan)."""
    tcp = t["obs/extra/tcp_pose"][()]  # (T,7) [x,y,z,qw,qx,qy,qz]
    pos = tcp[:, :3]  # meters
    d = np.linalg.norm(np.diff(pos, axis=0), axis=1)  # (T-1,) meters between consecutive
    speed_cm_s = (d / dt_s) * 100.0  # cm/s
    out = np.full(pos.shape[0], np.nan)
    out[1:] = speed_cm_s  # assign to the later timestep
    return out


def collect():
    """Walk all h5 files; return dict scene-> dict(behav-> dict(x=[], y=[]))."""
    data = {sc: {b: {"x": [], "y": []} for b in BEHAV_ORDER + ["other"]} for sc in RUN_DIRS}
    counts = {sc: {"trajs": 0, "skipped": 0, "points": 0} for sc in RUN_DIRS}

    # canonical sensor list from schema
    sensor_names = (
        [f"link2_sensor_{i}" for i in range(7)]
        + [f"link3_sensor_{i}" for i in range(8)]
        + [f"link5_sensor_{i}" for i in range(6)]
        + [f"link6_sensor_{i}" for i in range(8)]
    )

    for scene, rundir in RUN_DIRS.items():
        h5files = sorted(glob.glob(os.path.join(rundir, "house_*", "trajectories_batch_*.h5")))
        for hf in h5files:
            try:
                h = h5py.File(hf, "r")
            except Exception as e:
                print(f"[WARN] cannot open {hf}: {e}")
                continue
            with h:
                for key in list(h.keys()):
                    if not key.startswith("traj_"):
                        continue
                    try:
                        t = h[key]
                        scene_meta = decode_scene(t)
                        behav = scene_meta.get("behavior_class", "other")
                        if behav not in BEHAV_COLORS:
                            behav = "other"
                        dt_s = scene_meta.get("policy_dt_ms", None)
                        dt_s = (dt_s / 1000.0) if dt_s else DEFAULT_DT

                        mind = min_skin_distance_series(t, sensor_names)  # (T,) m
                        if mind is None:
                            counts[scene]["skipped"] += 1
                            continue
                        speed = tcp_speed_series(t, dt_s)  # (T,) cm/s

                        n = min(len(mind), len(speed))
                        mind = mind[:n]
                        speed = speed[:n]

                        # valid: finite speed AND a real (non-nan) min distance
                        valid = np.isfinite(speed) & np.isfinite(mind)
                        x = mind[valid]
                        y = speed[valid]
                        if x.size == 0:
                            counts[scene]["trajs"] += 1
                            continue
                        data[scene][behav]["x"].append(x)
                        data[scene][behav]["y"].append(y)
                        counts[scene]["trajs"] += 1
                        counts[scene]["points"] += int(x.size)
                    except Exception as e:
                        counts[scene]["skipped"] += 1
                        print(f"[WARN] skip {hf}:{key}: {e}")
    # concatenate
    for sc in data:
        for b in data[sc]:
            if data[sc][b]["x"]:
                data[sc][b]["x"] = np.concatenate(data[sc][b]["x"])
                data[sc][b]["y"] = np.concatenate(data[sc][b]["y"])
            else:
                data[sc][b]["x"] = np.array([])
                data[sc][b]["y"] = np.array([])
    return data, counts


def fit_and_stats(xall, yall):
    """Return (r, slope, intercept) or (nan,nan,nan)."""
    m = np.isfinite(xall) & np.isfinite(yall)
    x = xall[m]
    y = yall[m]
    if x.size < 3 or np.allclose(x, x[0]):
        return np.nan, np.nan, np.nan
    slope, intercept = np.polyfit(x, y, 1)
    try:
        r, _ = pearsonr(x, y)
    except Exception:
        r = np.nan
    return r, slope, intercept


def panel(ax, title, behav_dict):
    xs_all, ys_all = [], []
    for b in BEHAV_ORDER + ["other"]:
        if b not in behav_dict:
            continue
        x = behav_dict[b]["x"]
        y = behav_dict[b]["y"]
        if x.size == 0:
            continue
        color = BEHAV_COLORS.get(b, "#7f7f7f")
        ax.scatter(x, y, s=6, alpha=0.25, c=color, edgecolors="none", rasterized=True)
        xs_all.append(x)
        ys_all.append(y)
    if not xs_all:
        ax.set_title(title + "  (no data)")
        return
    xall = np.concatenate(xs_all)
    yall = np.concatenate(ys_all)
    r, slope, intercept = fit_and_stats(xall, yall)
    if np.isfinite(slope):
        xline = np.linspace(np.nanmin(xall), np.nanmax(xall), 100)
        ax.plot(xline, slope * xline + intercept, "k-", lw=2.0, zorder=5,
                label=f"fit: y={slope:.0f}x+{intercept:.0f}")
    # 8cm reference line (zone threshold)
    ax.axvline(0.08, color="purple", ls="--", lw=1.2, alpha=0.7, zorder=4)
    ax.text(0.08, ax.get_ylim()[1] * 0.96, " 8 cm zone", color="purple",
            fontsize=8, va="top", ha="left")
    rtxt = f"r={r:.2f}" if np.isfinite(r) else "r=n/a"
    stxt = f"slope={slope:.0f} cm/s per m" if np.isfinite(slope) else "slope=n/a"
    ax.set_title(f"{title}\n{rtxt},  {stxt},  N={xall.size:,}", fontsize=11)
    ax.set_xlabel("min skin distance to obstacle (m)")
    ax.set_ylabel("TCP speed (cm/s)")
    ax.grid(True, alpha=0.3)
    if np.isfinite(slope):
        ax.legend(loc="upper right", fontsize=8, framealpha=0.85)


def main():
    data, counts = collect()
    print("Collection counts:")
    for sc, c in counts.items():
        print(f"  {sc}: {c}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.ravel()

    scenes = ["fumehood", "panel", "cubby"]
    for ax, sc in zip(axes[:3], scenes):
        panel(ax, f"{sc}", data[sc])

    # pooled
    pooled = {b: {"x": [], "y": []} for b in BEHAV_ORDER + ["other"]}
    for sc in scenes:
        for b in pooled:
            pooled[b]["x"].append(data[sc][b]["x"])
            pooled[b]["y"].append(data[sc][b]["y"])
    for b in pooled:
        pooled[b]["x"] = np.concatenate(pooled[b]["x"]) if any(a.size for a in pooled[b]["x"]) else np.array([])
        pooled[b]["y"] = np.concatenate(pooled[b]["y"]) if any(a.size for a in pooled[b]["y"]) else np.array([])
    panel(axes[3], "POOLED (all scenes)", pooled)

    # shared legend for behavior classes
    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="", color=BEHAV_COLORS[b],
               label=b, markersize=8, alpha=0.8)
        for b in BEHAV_ORDER
    ]
    legend_handles.append(Line2D([0], [0], color="k", lw=2, label="linear fit"))
    legend_handles.append(Line2D([0], [0], color="purple", ls="--", lw=1.2, label="8 cm zone"))
    fig.legend(handles=legend_handles, loc="lower center", ncol=5, fontsize=10,
               frameon=True, bbox_to_anchor=(0.5, -0.005))

    fig.suptitle(
        "PROBE 1 - The Speed Law: TCP speed vs nearest proximity-skin return\n"
        "Arm slows as the 29-SPAD skin sees an obstacle approach (color = commanded behavior class)",
        fontsize=14, y=0.995,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(OUT_PNG, dpi=150)
    plt.close(fig)
    print(f"Saved {OUT_PNG}")


if __name__ == "__main__":
    main()
