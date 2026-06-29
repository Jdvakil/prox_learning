"""Combined fridge audit: proximity activation + per-phase target vision visibility.
Usage: audit_fridge.py <datagen_root_dir>
"""
import glob, sys
import numpy as np
import h5py

ROOT = sys.argv[1] if len(sys.argv) > 1 else \
    "/root/prox_learning/assets/datagen/fridge_two_level_v2/FridgeTwoLevelPnPV2Config"
ACTIVE_M, NEAR, FAR = 0.50, 0.02, 4.5
GROUPS = {"approach(0)": {0}, "pregrasp(2)": {2}, "grasp(3)": {3}, "close(4)": {4},
          "transit(5-6)": {5, 6}, "place(7)": {7}, "all": set(range(10))}

files = sorted(glob.glob(f"{ROOT}/**/trajectories_batch_*.h5", recursive=True))
print(f"ROOT: {ROOT}\n# demos: {len(files)}")
if not files:
    sys.exit("(no h5 files)")
with h5py.File(files[0], "r") as f:
    t0 = list(f.keys())[0]
    sensors = sorted(f[f"{t0}/obs/proximity"].keys())
    oip = list(f[f"{t0}/obs/extra/object_image_points"].keys())
TARGET = "pickup_obj" if "pickup_obj" in oip else oip[0]
S = len(sensors)

# per-phase accumulators
ph_frames = {}; ph_exo = {}; ph_wri = {}; ph_neither = {}; ph_act = {}
n_traj = 0; reached_place = 0
for fp in files:
    with h5py.File(fp, "r") as f:
        for tk in f.keys():
            g = f[tk]
            if "obs/proximity" not in g:
                continue
            ph = g["obs/extra/policy_phase"][:].reshape(-1)
            T = len(ph)
            md = np.full((S, T), np.inf, np.float32)
            for i, s in enumerate(sensors):
                a = g[f"obs/proximity/{s}"][:].reshape(T, -1)
                md[i] = np.where((a > NEAR) & (a < FAR), a, np.inf).min(axis=1)
            na = (md < ACTIVE_M).sum(axis=0)
            o = f"obs/extra/object_image_points/{TARGET}"
            ev = g[f"{o}/exo_camera_1/num_points"][:].reshape(-1) > 0
            wv = g[f"{o}/wrist_camera/num_points"][:].reshape(-1) > 0
            for p in np.unique(ph):
                m = ph == p; k = int(p)
                ph_frames[k] = ph_frames.get(k, 0) + int(m.sum())
                ph_exo[k] = ph_exo.get(k, 0) + int(ev[m].sum())
                ph_wri[k] = ph_wri.get(k, 0) + int(wv[m].sum())
                ph_neither[k] = ph_neither.get(k, 0) + int((~(ev | wv))[m].sum())
                ph_act[k] = ph_act.get(k, 0.0) + float(na[m].sum())
            if 7 in ph:
                reached_place += 1
            n_traj += 1

def grp(d, phs):
    return sum(d.get(p, 0) for p in phs)

print(f"# trajectories: {n_traj}   reached place phase: {reached_place}/{n_traj}")
print(f"\n{'group':14s} {'frames':>7s} {'exo':>7s} {'wrist':>7s} {'NEITHER(blind)':>15s} {'mean#active':>12s}")
for name, phs in GROUPS.items():
    fr = grp(ph_frames, phs)
    if fr == 0:
        print(f"{name:14s} {0:7d}      --      --              --          --"); continue
    print(f"{name:14s} {fr:7d} {grp(ph_exo,phs)/fr*100:6.1f}% {grp(ph_wri,phs)/fr*100:6.1f}% "
          f"{grp(ph_neither,phs)/fr*100:13.1f}% {grp(ph_act,phs)/fr:11.1f}")
