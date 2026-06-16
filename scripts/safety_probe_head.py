"""Open-loop probe: how well does the Safety-CVAE's baseline-subtracted retreat match the
analytic oracle as an obstacle approaches, with the arm held FIXED at the rest posture?

This isolates the head from the closed-loop feedback. For a few generated collision-course
scenarios it marches the obstacle in (arm frozen at q0) and, per frame, compares
    head_dq   = (head(skin_obstacle) - head(skin_parked)) / scale
    oracle_dq = (analytic(skin_obstacle) - analytic(skin_parked)) / scale
reporting magnitude ratio and direction cosine vs the obstacle's true surface clearance.

Reading:
  * high cosine in the engaged band (clearance ~0.02-0.20 m) => the head is good open-loop and
    a poor closed-loop result is a CONTROL issue (gain/overshoot/feedback), not the head.
  * low cosine even in the engaged band => the head genuinely mis-directs on these encounters
    (distribution mismatch vs the bench-bar training data) -- a real finding, fix the scenarios
    or report it.
  * |head|/|oracle| >> 1 near contact => the head over-fires at point-blank (OOD flooding).

Usage:
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python scripts/safety_probe_head.py \
      --runs <posture_run> --ckpt assets/safety/cvae_v3 --n 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import safety_eval_lib as L  # noqa: E402
import safety_sweep as sw  # noqa: E402
import mujoco  # noqa: E402
from train_safety_cvae import SafetyHead  # noqa: E402


def probe_one(ctx, sc, head, scale):
    model, data, rd, opt = ctx.model, ctx.data, ctx.rd, ctx.opt
    sensors, cam_ids = ctx.sensors, ctx.cam_ids
    obs_gid = L.obstacle_geom_id(model, sc.obstacle_name)
    jacp = np.zeros((3, model.nv))
    fromto = np.zeros(6)

    data.mocap_pos[ctx.mid[f"{L.NS}base"]] = sc.base[:3]
    data.mocap_quat[ctx.mid[f"{L.NS}base"]] = sc.base[3:7]
    L.apply_aperture(data, ctx.mid, sc.ap_w, sc.ap_h)
    L.park_obstacles(data, ctx.mid)
    L.set_arm(data, ctx.arm_qadr, sc.q0)           # arm FIXED at rest the whole time
    L.set_grip(data, ctx.finger_qadr, sc.grip)
    mujoco.mj_forward(model, data)
    base_boxes = L.frame_boxes(data, ctx.mid, sc.ap_w, sc.ap_h)

    rows = []
    for p in sc.path:
        if not np.isfinite(p).all():
            continue
        data.mocap_pos[ctx.mid[sc.obstacle_name]] = p
        mujoco.mj_forward(model, data)
        clr = L.link_clearance(model, data, obs_gid, ctx.arm_gids, fromto)
        d = L.render_all(model, data, rd, opt, sensors)
        data.mocap_pos[ctx.mid[sc.obstacle_name]] = sw.PARK
        mujoco.mj_forward(model, data)
        r = L.render_all(model, data, rd, opt, sensors)
        data.mocap_pos[ctx.mid[sc.obstacle_name]] = p

        boxes_w = base_boxes + [(np.asarray(p, float), sc.obstacle_half)]
        ow, _ = sw.analytic_retreat(model, data, sensors, cam_ids, d, boxes_w, ctx.arm_dofadr, jacp)
        orr, _ = sw.analytic_retreat(model, data, sensors, cam_ids, r, base_boxes, ctx.arm_dofadr, jacp)
        hd, hr = head(d), head(r)
        odq = (ow - orr) / max(scale, 1e-6)
        hdq = (hd - hr) / max(scale, 1e-6)
        nh, no = float(np.linalg.norm(hdq)), float(np.linalg.norm(odq))

        def _cos(a, b):
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            return float(np.dot(a, b) / (na * nb)) if na > 1e-9 and nb > 1e-9 else float("nan")

        cos_sub = _cos(hdq, odq)            # baseline-subtracted head vs oracle (what the eval uses)
        cos_raw = _cos(hd, ow)              # RAW head vs full-scene analytic label (training objective)
        nhr = float(np.linalg.norm(hr)) / max(scale, 1e-6)   # how hard the head fires at REST
        rows.append((clr, nh, no, cos_sub, cos_raw, nhr))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, default=Path("assets/safety/cvae_v3"))
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--engaged", type=float, nargs=2, default=(0.05, 0.25),
                    help="clearance band (m) considered an active near-encounter (the head's "
                         "training range; excludes the sub-5cm OOD endgame)")
    args = ap.parse_args()

    head = SafetyHead.load(args.ckpt)
    scale = float(head.scale)
    ctx = L.build_ctx()
    scs = L.make_scenarios(ctx, args.runs, args.n, args.seed, scale, head)

    lo, hi = args.engaged
    eng_sub, eng_raw = [], []
    for i, sc in enumerate(scs):
        a = np.array(probe_one(ctx, sc, head, scale))   # cols: clr,|head|,|oracle|,cosSub,cosRaw,|hRest|
        print(f"\n=== scenario {i}  link={sc.meta.get('link')}  sensor={sc.target_sensor} ===")
        print(f"{'clr_cm':>7} {'|head|':>7} {'|oracle|':>8} {'cosSub':>7} {'cosRaw':>7} {'|hRest|':>7}")
        for j in np.argsort(-a[:, 0])[:: max(1, len(a) // 12)]:
            clr, nh, no, cs, cr, hrn = a[j]
            print(f"{clr*100:7.1f} {nh:7.2f} {no:8.2f} {cs:7.2f} {cr:7.2f} {hrn:7.2f}")
        eng = a[(a[:, 0] >= lo) & (a[:, 0] <= hi)]
        if len(eng):
            cs = eng[:, 3][np.isfinite(eng[:, 3])]
            cr = eng[:, 4][np.isfinite(eng[:, 4])]
            ratio = eng[:, 1] / np.maximum(eng[:, 2], 1e-6)
            print(f"  engaged [{lo:.2f},{hi:.2f}]m n={len(eng)}: "
                  f"cos_sub={np.nanmean(cs):.2f}  cos_raw={np.nanmean(cr):.2f}  "
                  f"|head|/|oracle|={np.median(ratio):.2f}  |head_rest|={np.nanmean(eng[:, 5]):.2f}")
            eng_sub.append(cs); eng_raw.append(cr)

    if eng_sub:
        s, r = np.concatenate(eng_sub), np.concatenate(eng_raw)
        print(f"\nOVERALL engaged band:")
        print(f"  cos_sub (eval uses): mean={np.nanmean(s):.2f}  median={np.nanmedian(s):.2f}")
        print(f"  cos_raw (head vs label): mean={np.nanmean(r):.2f}  median={np.nanmedian(r):.2f}")
        print("  -> if cos_raw is high but cos_sub is low, the baseline subtraction is the problem.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
