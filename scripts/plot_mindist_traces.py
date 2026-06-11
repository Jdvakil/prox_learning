#!/usr/bin/env python
"""Min-skin-distance time traces, up to 2 example episodes per behavior class
(free / deflect / abort) pooled across the three scenes (fumehood, panel, cubby).

Each episode shows:
  - min skin distance (m) vs time (s)
  - 8 cm "skin alert" dashed line
  - phase-transition vertical lines labelled with policy_phases names
  - gripper-close moment marker (phase == 'gripper-close') if present

One subplot per behavior so the distinct signatures stand out:
  free   = stays open (never really dips near skin)
  deflect= dips then recovers
  abort  = dips then retreats (rises away)
"""
import os, json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTDIR = "/home/jaydv/code/prox_learning/diagnostics_output/20260610_plots_v3"
os.makedirs(OUTDIR, exist_ok=True)

SCENES = {
    "fumehood": "/home/jaydv/code/prox_learning/assets/datagen/fumehood_smoke/FrankaSkinFumehoodSmokeConfig/20260610_090517",
    "panel":    "/home/jaydv/code/prox_learning/assets/datagen/panel_slalom_smoke/FrankaSkinPanelSlalomSmokeConfig/20260610_092607",
    "cubby":    "/home/jaydv/code/prox_learning/assets/datagen/cubby_smoke/FrankaSkinCubbySmokeConfig/20260610_095739",
}

NO_RETURN = 2.0       # depth values > 2.0 m mean "no return" -> mask
SKIN_ALERT = 0.08     # 8 cm
DEFAULT_DT = 0.066

import h5py


def decode_blob(blob):
    if getattr(blob, "shape", ()) != ():
        blob = blob[0]
    s = (blob.tobytes() if hasattr(blob, "tobytes") else blob)
    if isinstance(s, bytes):
        s = s.decode("utf-8", "ignore")
    return s.rstrip("\x00")


def min_skin_dist_series(traj):
    """Return (T,) array of min skin distance over all sensors, ignoring no-return."""
    prox = traj["obs/proximity"]
    sensors = list(prox.keys())
    T = None
    per_sensor_min = []
    for s in sensors:
        arr = prox[s][:]                 # (T,4,8,8)
        depth = arr.mean(axis=1)         # (T,8,8) average over 4 sub-frames
        valid = depth <= NO_RETURN
        masked = np.where(valid, depth, np.nan)
        # min over the 8x8 grid per timestep, ignoring nan
        with np.errstate(all="ignore"):
            smin = np.nanmin(masked.reshape(masked.shape[0], -1), axis=1)
        per_sensor_min.append(smin)
        T = smin.shape[0]
    stack = np.vstack(per_sensor_min)    # (n_sensors, T)
    with np.errstate(all="ignore"):
        mind = np.nanmin(stack, axis=0)  # (T,)
    return mind


def scan_episodes():
    """Collect metadata + min-dist series for every traj in every scene."""
    eps = []  # dicts
    for scene, rundir in SCENES.items():
        for h5path in sorted(glob.glob(os.path.join(rundir, "house_*", "trajectories_batch_*.h5"))):
            try:
                h = h5py.File(h5path, "r")
            except Exception as e:
                print("skip file", h5path, e)
                continue
            for tname in list(h.keys()):
                try:
                    t = h[tname]
                    scene_meta = json.loads(decode_blob(t["obs_scene"][()]))
                    behavior = scene_meta.get("behavior_class", "unknown")
                    phase_name_to_id = scene_meta.get("policy_phases", {})
                    id_to_name = {v: k for k, v in phase_name_to_id.items()}
                    dt = scene_meta.get("policy_dt_ms", DEFAULT_DT * 1000) / 1000.0
                    mind = min_skin_dist_series(t)
                    if mind.size == 0 or np.all(np.isnan(mind)):
                        continue
                    phase = t["obs/extra/policy_phase"][:].astype(int)
                    success = bool(t["success"][-1])
                    # signature metrics
                    finite = mind[np.isfinite(mind)]
                    gmin = float(np.nanmin(mind))
                    last_third = mind[int(0.66 * len(mind)):]
                    recover = float(np.nanmax(last_third) - gmin) if last_third.size else 0.0
                    eps.append(dict(
                        scene=scene, behavior=behavior, traj=tname, file=h5path,
                        mind=mind, dt=dt, phase=phase, id_to_name=id_to_name,
                        phase_name_to_id=phase_name_to_id, success=success,
                        gmin=gmin, recover=recover, T=len(mind),
                    ))
                except Exception as e:
                    # robustness: skip bad traj
                    continue
            h.close()
    return eps


def pick_examples(eps, behavior, n=2):
    """Pick up to n episodes for a behavior, prefer distinct scenes and a clear,
    representative signature (real dip for deflect/abort; clear retreat for abort)."""
    cands = [e for e in eps if e["behavior"] == behavior]
    if not cands:
        return []

    if behavior == "free":
        # free should stay open: prefer episodes that NEVER breach the 8 cm alert,
        # then by highest global min (furthest from skin the whole time).
        cands.sort(key=lambda e: (1 if e["gmin"] >= SKIN_ALERT else 0, e["gmin"]),
                   reverse=True)
    elif behavior == "deflect":
        # dips then recovers: want a real dip (gmin below alert) AND recovery
        def score(e):
            dipped = e["gmin"] < SKIN_ALERT
            return (1 if dipped else 0, e["recover"])
        cands.sort(key=score, reverse=True)
    elif behavior == "abort":
        # dips then retreats (rises) at the end: want dip + strong late recovery (retreat)
        def score(e):
            dipped = e["gmin"] < SKIN_ALERT
            return (1 if dipped else 0, e["recover"])
        cands.sort(key=score, reverse=True)
    else:
        cands.sort(key=lambda e: -e["gmin"])

    # prefer distinct scenes among the top picks
    chosen, seen_scenes = [], set()
    for e in cands:
        if e["scene"] not in seen_scenes:
            chosen.append(e); seen_scenes.add(e["scene"])
        if len(chosen) >= n:
            break
    if len(chosen) < n:
        for e in cands:
            if e not in chosen:
                chosen.append(e)
            if len(chosen) >= n:
                break
    return chosen[:n]


def phase_transitions(phase):
    """Indices where phase id changes (the new phase starts at that index)."""
    trans = []
    for i in range(1, len(phase)):
        if phase[i] != phase[i - 1]:
            trans.append(i)
    return trans


def plot():
    eps = scan_episodes()
    print(f"scanned {len(eps)} episodes")
    from collections import Counter
    print("behavior counts:", Counter(e["behavior"] for e in eps))

    behaviors = ["free", "deflect", "abort"]
    sig_desc = {
        "free": "stays open: at most brief grazes, no sustained dip",
        "deflect": "sustained dip below 8 cm, then recovers",
        "abort": "dips to contact, then retreats (monotonic rise)",
    }

    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5), sharey=True)
    line_colors = ["#1f77b4", "#d62728"]

    # Shared y cap: keep the 8 cm action legible. Lines that exceed the cap are
    # clipped (annotated) since the engineering-relevant zone is near the skin.
    YCAP = 0.32

    for ax, beh in zip(axes, behaviors):
        picks = pick_examples(eps, beh, n=2)
        clipped_any = False
        for ci, e in enumerate(picks):
            mind = e["mind"]
            dt = e["dt"]
            t_s = np.arange(len(mind)) * dt
            color = line_colors[ci % len(line_colors)]
            finite = mind[np.isfinite(mind)]
            if finite.size and np.nanmax(finite) > YCAP:
                clipped_any = True
            lbl = (f"{e['scene']} {e['traj']} "
                   f"[{'success' if e['success'] else 'fail'}, min={e['gmin']*100:.1f} cm]")
            ax.plot(t_s, mind, color=color, lw=2.0, label=lbl, zorder=3)
            ax.scatter([t_s[np.nanargmin(np.where(np.isfinite(mind), mind, np.inf))]],
                       [e["gmin"]], marker="o", s=45, facecolor="white",
                       edgecolor=color, linewidth=1.5, zorder=6)

            # phase transitions with staggered, de-duplicated labels
            id_to_name = e["id_to_name"]
            trans = phase_transitions(e["phase"])
            for j, ti in enumerate(trans):
                xt = ti * dt
                ax.axvline(xt, color=color, ls=":", lw=0.9, alpha=0.40, zorder=1)
                pname = id_to_name.get(int(e["phase"][ti]), str(int(e["phase"][ti])))
                # stagger vertically so labels do not collide
                yfrac = 0.96 - 0.085 * (j % 5) - (0.0 if ci == 0 else 0.43)
                ax.text(xt, YCAP * yfrac, pname, rotation=90, va="top", ha="right",
                        fontsize=7, color=color, alpha=0.85, zorder=4)

            # gripper-close moment
            gc_id = e["phase_name_to_id"].get("gripper-close")
            if gc_id is not None:
                gc_idx = np.where(e["phase"] == gc_id)[0]
                if gc_idx.size:
                    gi = gc_idx[0]
                    gx = gi * dt
                    gy = mind[gi] if np.isfinite(mind[gi]) else (e["gmin"])
                    gy = min(gy, YCAP)
                    ax.scatter([gx], [gy], marker="v", s=130, color=color,
                               edgecolor="black", zorder=7,
                               label="gripper-close" if ci == 0 else None)

        # 8 cm alert line + shaded danger zone
        ax.axhspan(0, SKIN_ALERT, color="red", alpha=0.05, zorder=0)
        ax.axhline(SKIN_ALERT, color="k", ls="--", lw=1.6, alpha=0.85,
                   label="8 cm skin alert", zorder=2)
        ax.set_ylim(0, YCAP)
        ax.set_xlim(left=0)
        ttl = f"{beh.upper()}\nsignature: {sig_desc[beh]}"
        if clipped_any:
            ttl += "  (trace clipped at 32 cm)"
        ax.set_title(ttl, fontsize=12, fontweight="bold")
        ax.set_xlabel("time (s)")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.92)

    axes[0].set_ylabel("min skin distance over 29 SPAD sensors (m)")
    fig.suptitle("Min-skin-distance time traces by behavior class "
                 "(proximity 'skin' steering the ACT policy)\n"
                 "29 SPAD sensors, no-return (>2 m) masked; up to 2 example episodes per class across scenes",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(OUTDIR, "mindist_traces.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


if __name__ == "__main__":
    out = plot()
    sz = os.path.getsize(out) if os.path.exists(out) else 0
    print("WROTE", out, sz, "bytes", "OK" if sz > 5120 else "TOO_SMALL")
