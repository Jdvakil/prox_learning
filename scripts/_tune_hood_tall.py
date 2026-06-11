import sys
sys.path.insert(0, "/home/jaydv/code/prox_learning/scripts")
from hybrid_viz_lib import (build, set_pose, skin_cloud, nice_lights, add_box, FAR)
import mujoco, numpy as np

W, H, D, SASH = 0.34, 0.85, 0.55, 0.55
BZ, X0 = 0.585, 0.35

def mk(s):
    nice_lights(s)
    add_box(s, "bench", [X0 + D / 2, 0, BZ - 0.015], [D / 2 + 0.05, W + 0.05, 0.015], [0.62, 0.55, 0.45, 1])
    add_box(s, "bench_body", [X0 + D / 2, 0, BZ / 2 - 0.02], [D / 2, W, BZ / 2 - 0.02], [0.55, 0.5, 0.44, 1])
    add_box(s, "wall_l", [X0 + D / 2, W, BZ + H / 2], [D / 2, 0.012, H / 2], [0.78, 0.8, 0.84, 0.30])
    add_box(s, "wall_r", [X0 + D / 2, -W, BZ + H / 2], [D / 2, 0.012, H / 2], [0.78, 0.8, 0.84, 0.30])
    add_box(s, "back", [X0 + D, 0, BZ + H / 2], [0.012, W, H / 2], [0.72, 0.7, 0.66, 1])
    add_box(s, "top", [X0 + D / 2, 0, BZ + H], [D / 2, W, 0.012], [0.78, 0.8, 0.84, 0.30])
    add_box(s, "sash", [X0, 0, BZ + SASH + 0.028], [0.012, W, 0.028], [0.62, 0.64, 0.66, 1])
    add_box(s, "target", [X0 + 0.7 * D, 0.0, BZ + 0.045], [0.04, 0.04, 0.045], [0.85, 0.5, 0.3, 1])

model = build(mk)
data = mujoco.MjData(model)
HAND = "gripper/base"
BID = model.body(HAND).id

def fk(q):
    set_pose(model, data, q)
    return data.xpos[BID].copy()

JIDS = [model.joint(f"fr3_joint{i}").id for i in (2, 4, 6)]
DADR = [model.joint(f"fr3_joint{i}").dofadr[0] for i in (2, 4, 6)]
QIDX = [1, 3, 5]

def ik(seed, tx, tz, iters=300):
    q = np.array(seed, float)
    for _ in range(iters):
        set_pose(model, data, q)
        p = data.xpos[BID]
        err = np.array([tx - p[0], tz - p[2]])
        if np.linalg.norm(err) < 5e-5:
            break
        jacp = np.zeros((3, model.nv)); jacr = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jacp, jacr, BID)
        J = jacp[np.ix_([0, 2], DADR)]
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-5 * np.eye(2), err)
        for k in range(3):
            lo, hi = model.jnt_range[JIDS[k]]
            q[QIDX[k]] = np.clip(q[QIDX[k]] + 0.6 * dq[k], lo, hi)
    p = fk(q)
    return q, p, np.linalg.norm([tx - p[0], tz - p[2]])

targets = {
    "entry":     ([0, -0.35, 0, -2.30, 0, 2.05, 0.79], 0.40, 0.82),
    "mid":       ([0,  0.05, 0, -1.75, 0, 1.85, 0.79], 0.55, 0.76),
    "deep_low":  ([0,  0.35, 0, -1.25, 0, 1.62, 0.79], 0.70, 0.70),
    "deep_high": ([0,  0.35, 0, -1.25, 0, 1.62, 0.79], 0.64, 0.98),
}
need_x = X0 + 0.55 * D
print(f"need hand_x >= {need_x:.4f}  (depth >= {(need_x-X0)*100:.1f} cm); band z=[{BZ+0.05:.3f},{BZ+SASH-0.05:.3f}]")
for name, (seed, tx, tz) in targets.items():
    q, p, res = ik(seed, tx, tz)
    pts, dd, mins = skin_cloud(model, data)
    act = sum(1 for v in mins.values() if v < FAR)
    mn = min(mins.values())
    print(f"{name:10s} q={np.round(q,3).tolist()}  hand=({p[0]:.3f},{p[1]:.3f},{p[2]:.3f}) "
          f"res={res*1000:.2f}mm depth={(p[0]-X0)*100:.1f}cm  pts={len(pts)} act={act}/40 min={mn*100:.1f}cm")
