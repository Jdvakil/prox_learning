"""Audit proximity activation for the FridgeTwoLevelPnPV2Config dataset.

Activation convention follows the repo (README closeness, Dmax=0.5 m): a sensor is
"active" at a frame if its closest valid return is < 0.5 m. Per sensor per frame we take
the min depth over the (4 substeps x 8 x 8) tile, ignoring invalid (<2 cm) / far (>4.5 m).
Reports overall + per-sensor + per-phase activation and closeness.
"""
import glob, sys
import numpy as np
import h5py

ROOT = "/root/prox_learning/assets/datagen/fridge_two_level_v2/FridgeTwoLevelPnPV2Config"
ACTIVE_M, CLOSE_M, NEAR_INVALID, FAR = 0.50, 0.20, 0.02, 4.5

files = sorted(glob.glob(f"{ROOT}/**/trajectories_batch_*.h5", recursive=True))
print(f"# demos (h5 files): {len(files)}")
if not files:
    sys.exit("no files")

# discover sensor names + inspect encoding/phase on the first file
with h5py.File(files[0], "r") as f:
    t0 = list(f.keys())[0]
    sensors = sorted(f[f"{t0}/obs/proximity"].keys())
    s0 = f[f"{t0}/obs/proximity/{sensors[0]}"][:]
    print(f"# sensors: {len(sensors)}   per-sensor shape: {s0.shape}  dtype {s0.dtype}")
    pct = np.percentile(s0.ravel(), [0, 1, 50, 90, 99, 100])
    print("proximity value percentiles [0,1,50,90,99,100]:", np.round(pct, 3),
          "-> raw metric depth (m)" if pct[-1] > 1.5 else "-> looks like closeness 0..1")
    ph = f[f"{t0}/obs/extra/policy_phase"][:]
    print(f"policy_phase shape {ph.shape} dtype {ph.dtype}; sample {ph[:8].ravel()}")

S = len(sensors)
# accumulators
sensor_active = np.zeros(S); sensor_close = np.zeros(S); frames_total = 0
nactive_hist = np.zeros(S + 1, dtype=np.int64)
phase_nactive = {}; phase_frames = {}; phase_mindepth = {}
min_depth_all = []  # active-sensor min depths (subsample)
zero_frac_accum = 0.0; n_traj = 0
lengths = []

for fp in files:
    with h5py.File(fp, "r") as f:
        for tkey in f.keys():
            g = f[tkey]
            if "obs/proximity" not in g:
                continue
            T = g[f"obs/proximity/{sensors[0]}"].shape[0]
            md = np.full((S, T), np.inf, dtype=np.float32)  # min valid depth per sensor/frame
            raw_nonzero = 0.0
            for i, s in enumerate(sensors):
                a = g[f"obs/proximity/{s}"][:].reshape(T, -1)  # (T, 4*8*8)
                raw_nonzero += np.mean(a > NEAR_INVALID)
                valid = (a > NEAR_INVALID) & (a < FAR)
                aa = np.where(valid, a, np.inf)
                md[i] = aa.min(axis=1)
            active = md < ACTIVE_M                  # (S, T)
            close = md < CLOSE_M
            sensor_active += active.sum(axis=1)
            sensor_close += close.sum(axis=1)
            frames_total += T
            na = active.sum(axis=0)                 # (T,) #sensors active per frame
            for v in na:
                nactive_hist[int(v)] += 1
            d_active = md[active]
            if d_active.size:
                min_depth_all.append(d_active[::7].astype(np.float32))
            # phase-resolved
            ph = g["obs/extra/policy_phase"][:].reshape(T, -1)[:, 0]
            for p in np.unique(ph):
                m = ph == p
                key = p.item() if hasattr(p, "item") else p
                phase_nactive[key] = phase_nactive.get(key, 0.0) + na[m].sum()
                phase_frames[key] = phase_frames.get(key, 0) + int(m.sum())
                pm = md[:, m]; pm = pm[np.isfinite(pm) & (pm < ACTIVE_M)]
                phase_mindepth.setdefault(key, []).append(pm[::5].astype(np.float32))
            zero_frac_accum += raw_nonzero / S
            n_traj += 1
            lengths.append(T)

print(f"\n# trajectories analyzed: {n_traj}   mean length: {np.mean(lengths):.0f} frames")
print(f"sanity — mean fraction of NONZERO proximity pixels/frame across sensors: "
      f"{zero_frac_accum / n_traj:.3f}  (≈0 would mean the zero-proximity bug)")

overall = sensor_active.sum() / (S * frames_total)
print(f"\n==== OVERALL ====")
print(f"activation rate (sensor sees <{ACTIVE_M} m): {overall*100:.1f}% of all sensor-frames")
print(f"close rate     (sensor sees <{CLOSE_M} m): {sensor_close.sum()/(S*frames_total)*100:.1f}%")
mean_na = (np.arange(S + 1) * nactive_hist).sum() / nactive_hist.sum()
print(f"mean # sensors active per frame: {mean_na:.1f} / {S}")
md_all = np.concatenate(min_depth_all) if min_depth_all else np.zeros(1)
print(f"median min-depth among active sensors: {np.median(md_all)*100:.1f} cm")
# fraction of frames with >= k sensors active
cum = nactive_hist[::-1].cumsum()[::-1] / nactive_hist.sum()
for k in (1, 8, 16, 24, 32):
    print(f"  frac frames with >= {k:2d} sensors active: {cum[k]*100:.1f}%")

print(f"\n==== PER-SENSOR activation rate (sorted) ====")
rates = sensor_active / frames_total
order = np.argsort(rates)[::-1]
print(" TOP 8:", [f"{sensors[i]}={rates[i]*100:.0f}%" for i in order[:8]])
print(" BOT 8:", [f"{sensors[i]}={rates[i]*100:.0f}%" for i in order[-8:]])
print(f" sensors active >50% of the time: {(rates>0.5).sum()}/{S}")
print(f" sensors active >10% of the time: {(rates>0.1).sum()}/{S}")

print(f"\n==== PER-PHASE (mean #sensors active, mean active-depth) ====")
for k in sorted(phase_frames, key=lambda x: -phase_nactive[x]/max(phase_frames[x],1)):
    fr = phase_frames[k]; mna = phase_nactive[k]/max(fr,1)
    dd = np.concatenate(phase_mindepth[k]) if phase_mindepth[k] else np.zeros(1)
    print(f"  phase {str(k):>10s}: frames={fr:6d}  mean#active={mna:5.1f}/{S}  "
          f"median active-depth={np.median(dd)*100:4.1f}cm")
