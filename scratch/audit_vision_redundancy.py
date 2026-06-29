"""Quantify how much VISION is available during the trajectory (esp. the grasp window),
to gauge proximity<->camera redundancy. Uses stored object_image_points: num_points>0 for a
camera means the target object projects into that camera's image at that frame.
"""
import glob, sys
import numpy as np
import h5py

ROOT = "/root/prox_learning/assets/datagen/fridge_two_level_v2/FridgeTwoLevelPnPV2Config"
PHASE = {0:"approach",1:"grip-open",2:"pregrasp",3:"grasp",4:"grip-close",
         5:"lift",6:"preplace",7:"place",8:"retreat",9:"go_home"}
files = sorted(glob.glob(f"{ROOT}/**/trajectories_batch_*.h5", recursive=True))
print(f"# demos: {len(files)}")

# discover camera keys under object_image_points (exclude the proximity link*_sensor_* entries)
with h5py.File(files[0], "r") as f:
    t0 = list(f.keys())[0]
    base = f[f"{t0}/obs/extra/object_image_points"]
    grp = "pickup_obj"                               # the TARGET object (not 'gripper')
    allk = list(base[grp].keys())
    cams = [k for k in allk if "camera" in k]
    print("camera keys:", cams)

OIP = f"obs/extra/object_image_points/{grp}"
# accumulators: per camera -> per phase [visible_frames, total_frames, sum_points]
acc = {c: {} for c in cams}
both = {}; either = {}; neither = {}; ptot = {}
for fp in files:
    with h5py.File(fp, "r") as f:
        for tk in f.keys():
            g = f[tk]
            if f"{OIP}/{cams[0]}/num_points" not in g:
                continue
            ph = g["obs/extra/policy_phase"][:].reshape(-1)
            T = ph.shape[0]
            vis = {}
            for c in cams:
                np_pts = g[f"{OIP}/{c}/num_points"][:].reshape(-1).astype(np.int64)
                vis[c] = np_pts > 0
                for p in np.unique(ph):
                    d = acc[c].setdefault(int(p), [0,0,0])
                    m = ph == p
                    d[0] += int(vis[c][m].sum()); d[1] += int(m.sum())
                    d[2] += int(np_pts[m].sum())
            ev = np.zeros(T, bool); bv = np.ones(T, bool)
            for c in cams:
                ev |= vis[c]; bv &= vis[c]
            for p in np.unique(ph):
                m = ph == p; key = int(p)
                either[key]  = either.get(key,0)  + int(ev[m].sum())
                both[key]    = both.get(key,0)    + int(bv[m].sum())
                neither[key] = neither.get(key,0) + int((~ev[m]).sum())
                ptot[key]    = ptot.get(key,0)    + int(m.sum())

def rate(c, p):
    d = acc[c].get(p); return (d[0]/d[1]*100, d[2]/d[1]) if d and d[1] else (0,0)

print("\n==== TARGET VISIBILITY by phase (% frames the object projects into the cam | mean #pts) ====")
hdr = "  phase        frames " + "".join(f"{c[:11]:>14s}" for c in cams) + "   either  both  neither"
print(hdr)
order = sorted(ptot, key=lambda p: p)
for p in order:
    row = f"  {PHASE.get(p,p):>10s} {ptot[p]:7d} "
    for c in cams:
        r,mp = rate(c,p); row += f"  {r:5.0f}% ({mp:4.1f})"
    row += f"   {either[p]/ptot[p]*100:5.0f}% {both[p]/ptot[p]*100:4.0f}% {neither[p]/ptot[p]*100:5.0f}%"
    print(row)

# overall + grasp-window (pregrasp+grasp+close+lift = 2,3,4,5)
def agg(phs):
    tot=sum(ptot[p] for p in phs if p in ptot)
    out={}
    for c in cams:
        vf=sum(acc[c][p][0] for p in phs if p in acc[c]); out[c]=vf/tot*100 if tot else 0
    e=sum(either[p] for p in phs if p in either); n=sum(neither[p] for p in phs if p in neither)
    return tot, out, e/tot*100 if tot else 0, n/tot*100 if tot else 0

for label,phs in [("ALL", list(ptot)), ("GRASP WINDOW (pregrasp..lift)", [2,3,4,5])]:
    tot,out,e,n = agg(phs)
    print(f"\n{label}: frames={tot}")
    for c in cams: print(f"   {c:14s} target-visible: {out[c]:5.1f}% of frames")
    print(f"   visible in EITHER cam: {e:5.1f}%   |   in NEITHER (vision blind): {n:5.1f}%")
