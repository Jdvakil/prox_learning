#!/usr/bin/env python
"""Grasp-reach diagnostic: the kinematic yield ceiling.

For episodes that attempt a grasp (behavior_class in {free, deflect}),
at the LAST timestep compute delta = obj_end[0][:3] - tcp_pose[-1][:3].
Panel A: histogram of delta_z (m).
Panel B: scatter delta_z vs object x, colored by success.
Panel C: histogram of horizontal miss sqrt(dx^2+dy^2).
"""
import os, json, glob
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/home/jaydv/code/prox_learning/diagnostics_output/20260610_plots_v3"
os.makedirs(OUT, exist_ok=True)

RUNS = {
    "fumehood": "/home/jaydv/code/prox_learning/assets/datagen/fumehood_smoke/FrankaSkinFumehoodSmokeConfig/20260610_090517",
    "panel":    "/home/jaydv/code/prox_learning/assets/datagen/panel_slalom_smoke/FrankaSkinPanelSlalomSmokeConfig/20260610_092607",
    "cubby":    "/home/jaydv/code/prox_learning/assets/datagen/cubby_smoke/FrankaSkinCubbySmokeConfig/20260610_095739",
}


def decode_scene(ds):
    blob = ds[()] if ds.shape == () else ds[0]
    s = (blob.tobytes() if hasattr(blob, "tobytes") else blob).decode("utf-8", "ignore").rstrip("\x00")
    return json.loads(s)


# Collect: delta_x, delta_y, delta_z, obj_x, success, behavior, scene
dz, dh, objx, succ, beh, scene_lbl = [], [], [], [], [], []
n_total, n_attempt = 0, 0

for scene, run in RUNS.items():
    for h5path in sorted(glob.glob(os.path.join(run, "house_*", "trajectories_batch_*.h5"))):
        try:
            h = h5py.File(h5path, "r")
        except Exception:
            continue
        for tk in list(h.keys()):
            try:
                t = h[tk]
                n_total += 1
                d = decode_scene(t["obs_scene"])
                bc = d.get("behavior_class")
                if bc not in ("free", "deflect"):
                    continue
                n_attempt += 1
                obj0 = np.asarray(t["obs/extra/obj_end"][0][:3], dtype=float)
                tcpL = np.asarray(t["obs/extra/tcp_pose"][-1][:3], dtype=float)
                delta = obj0 - tcpL  # obj minus tcp
                s_ok = bool(np.asarray(t["success"])[-1])
                dz.append(delta[2])
                dh.append(float(np.hypot(delta[0], delta[1])))
                objx.append(obj0[0])
                succ.append(s_ok)
                beh.append(bc)
                scene_lbl.append(scene)
            except Exception:
                continue
        h.close()

dz = np.asarray(dz); dh = np.asarray(dh); objx = np.asarray(objx)
succ = np.asarray(succ, dtype=bool)
N = dz.size
print(f"total traj={n_total} grasp-attempts={n_attempt} usable={N}")
print(f"delta_z: mean={dz.mean():.3f} median={np.median(dz):.3f} m  | success rate={succ.mean():.2f}")

# ---- Figure ----
fig, axes = plt.subplots(1, 3, figsize=(18, 5.6))

# Panel A: histogram of delta_z
axA = axes[0]
bins = np.linspace(min(dz.min(), -0.05), dz.max() + 0.02, 30)
axA.hist(dz[succ], bins=bins, color="#2ca02c", alpha=0.7, label=f"success (n={succ.sum()})", edgecolor="white", linewidth=0.4)
axA.hist(dz[~succ], bins=bins, color="#d62728", alpha=0.65, label=f"failure (n={(~succ).sum()})", edgecolor="white", linewidth=0.4)
axA.axvline(0.0, color="k", ls="--", lw=1.2, label="object level (dz=0)")
med = float(np.median(dz))
axA.axvline(med, color="#1f77b4", ls="-", lw=1.6, label=f"median dz = {med*100:.1f} cm")
axA.set_xlabel(r"$\Delta z = z_{obj} - z_{tcp}$  (m)")
axA.set_ylabel("episodes")
axA.set_title("A. Vertical reach gap at final step\n(positive = object sits ABOVE the gripper)")
axA.legend(fontsize=9, loc="upper right")
axA.grid(alpha=0.25)

# Panel B: scatter delta_z vs object x, colored by success
axB = axes[1]
axB.scatter(objx[succ], dz[succ], c="#2ca02c", s=42, alpha=0.8, edgecolor="k", linewidth=0.3, label="success")
axB.scatter(objx[~succ], dz[~succ], c="#d62728", s=42, alpha=0.8, edgecolor="k", linewidth=0.3, label="failure")
axB.axhline(0.0, color="k", ls="--", lw=1.0)
# trend line (kinematic ceiling: gap grows with reach distance x)
if N >= 2:
    m, b = np.polyfit(objx, dz, 1)
    xs = np.linspace(objx.min(), objx.max(), 50)
    axB.plot(xs, m * xs + b, color="#1f77b4", lw=2.0, ls="-",
             label=f"trend: dz = {m:.2f}·x {b:+.2f}")
    corr = np.corrcoef(objx, dz)[0, 1]
    axB.text(0.03, 0.96, f"corr(x, dz) = {corr:+.2f}", transform=axB.transAxes,
             va="top", ha="left", fontsize=10, bbox=dict(boxstyle="round", fc="white", alpha=0.8))
axB.set_xlabel(r"object reach distance  $x_{obj}$  (m, world)")
axB.set_ylabel(r"$\Delta z$  (m)")
axB.set_title("B. Reach gap vs. object distance\n(farther objects -> larger vertical shortfall)")
axB.legend(fontsize=9, loc="lower right")
axB.grid(alpha=0.25)

# Panel C: histogram of horizontal miss
axC = axes[2]
binsC = np.linspace(0, dh.max() + 0.01, 30)
axC.hist(dh[succ] * 100, bins=binsC * 100, color="#2ca02c", alpha=0.7, label=f"success (n={succ.sum()})", edgecolor="white", linewidth=0.4)
axC.hist(dh[~succ] * 100, bins=binsC * 100, color="#d62728", alpha=0.65, label=f"failure (n={(~succ).sum()})", edgecolor="white", linewidth=0.4)
medh = float(np.median(dh)) * 100
axC.axvline(medh, color="#1f77b4", ls="-", lw=1.6, label=f"median miss = {medh:.1f} cm")
axC.set_xlabel(r"horizontal miss  $\sqrt{\Delta x^2 + \Delta y^2}$  (cm)")
axC.set_ylabel("episodes")
axC.set_title("C. In-plane miss distance at final step")
axC.legend(fontsize=9, loc="upper right")
axC.grid(alpha=0.25)

# Story title
hi_frac = float(np.mean(dz > 0.03))
medh_all = float(np.median(dh)) * 100
fig.suptitle(
    "Grasp-reach diagnostic: the kinematic yield ceiling   |   "
    f"grasp-attempt episodes (free+deflect) n={N}, success={succ.mean()*100:.0f}%\n"
    f"Reach falls short: {hi_frac*100:.0f}% of episodes leave the object >3 cm HIGH (median dz = {med*100:+.1f} cm) "
    f"AND the gripper stops ~{medh_all:.0f} cm short in-plane -- the arm tops out before closing on far/occluded targets.",
    fontsize=13, y=1.02,
)

plt.tight_layout(rect=[0, 0, 1, 0.97])
outpng = os.path.join(OUT, "grasp_diag.png")
fig.savefig(outpng, dpi=150, bbox_inches="tight")
plt.close(fig)

sz = os.path.getsize(outpng)
print(f"WROTE {outpng} ({sz} bytes, {sz/1024:.1f} KB) exists={os.path.exists(outpng)}")
