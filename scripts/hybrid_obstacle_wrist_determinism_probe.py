#!/usr/bin/env python3
"""Localize the wrist-camera replay nondeterminism without running a policy.

Handoff steps 4-6. Builds the accepted state of one manifest row, captures the
full simulator / camera / render / RNG contract, then runs the probe matrix and
reports where the first divergence appears.

No ACT policy is constructed and no rollout is executed. Every render here is a
diagnostic probe.

Modes
-----
``in-process``  probes A, B, E, F, G, H, I, J plus the repeated-render sweep, all
                inside a single environment construction where possible.
``fresh-env``   probe C: N fresh environment constructions in one process.
``one-env``     probe D helper: one construction, print the step-0 hashes, so the
                caller can compare across separate processes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.pop("DISPLAY", None)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submodules" / "molmospaces"))

import mujoco
from molmo_spaces.data_generation.config.object_manipulation_datagen_configs import (
    FrankaSkinHybridObstacleManifestV2Config,
)
from molmo_spaces.data_generation.episode_manifest import install_row_seed_contract
from molmo_spaces.data_generation.manifest_runner import (
    extract_row_observations,
    reset_episode_scoped_sampler_state,
)

WRIST = "wrist_camera"
EXO = "exo_camera_1"


def ah(a) -> str:
    arr = np.ascontiguousarray(np.asarray(a))
    h = hashlib.sha256()
    h.update(str(arr.dtype.str).encode())
    h.update(str(arr.shape).encode())
    h.update(arr.tobytes())
    return h.hexdigest()


def rng_state_digest() -> dict[str, str]:
    import torch
    return {
        "python": hashlib.sha256(repr(random.getstate()).encode()).hexdigest()[:16],
        "numpy": hashlib.sha256(repr(np.random.get_state()).encode()).hexdigest()[:16],
        "torch_cpu": ah(torch.get_rng_state().numpy()),
        "torch_cuda": (ah(torch.cuda.get_rng_state_all()[0].numpy())
                       if torch.cuda.is_available() else None),
    }


def capture_state(env, task, sensor_names) -> dict[str, Any]:
    """Everything step 4 requires, captured before any step-0 render."""
    m, d = env.current_model, env.current_data
    reg = env.camera_manager.registry
    cams: dict[str, Any] = {}
    for name in (WRIST, EXO):
        if name in reg:
            cam = reg[name]
            cams[name] = {
                "registry_pos": [float(x) for x in np.asarray(cam.pos).ravel()],
                "registry_forward": [float(x) for x in np.asarray(cam.forward).ravel()],
                "registry_up": [float(x) for x in np.asarray(cam.up).ravel()],
                "registry_fov": float(cam.fov),
            }
        try:
            cid = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, name))
        except Exception:  # noqa: BLE001
            cid = -1
        cams.setdefault(name, {})["model_camera_id"] = cid
        if cid >= 0:
            cams[name]["cam_xpos"] = [float(x) for x in d.cam_xpos[cid]]
            cams[name]["cam_xmat"] = [float(x) for x in d.cam_xmat[cid].ravel()]
            cams[name]["model_fovy"] = float(m.cam_fovy[cid])
    prox_cams = {}
    for s in sensor_names:
        cid = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, s))
        if cid >= 0:
            prox_cams[s] = {"cam_xpos": ah(d.cam_xpos[cid]), "cam_xmat": ah(d.cam_xmat[cid])}
    res = (list(env.config.camera_config.img_resolution)
           if getattr(env.config, "camera_config", None) is not None else None)
    return {
        "mujoco_state": {
            "time": float(d.time), "qpos": ah(d.qpos), "qvel": ah(d.qvel),
            "act": ah(d.act) if d.act.size else None, "ctrl": ah(d.ctrl),
            "mocap_pos": ah(d.mocap_pos), "mocap_quat": ah(d.mocap_quat),
            "userdata": ah(d.userdata) if d.userdata.size else None,
            "qacc_warmstart": ah(d.qacc_warmstart),
            "nq": int(m.nq), "nv": int(m.nv), "ncon": int(d.ncon),
        },
        "derived": {"xpos": ah(d.xpos), "xmat": ah(d.xmat)},
        "cameras": cams,
        "proximity_camera_poses_digest": ah(
            np.array([[*np.frombuffer(bytes.fromhex(v["cam_xpos"])[:8], dtype=np.uint8)]
                      for v in prox_cams.values()], dtype=np.uint8)
        ) if prox_cams else None,
        "proximity_camera_count": len(prox_cams),
        "render_contract": {
            "resolution": res,
            "renderer_class": type(env._renderer).__name__,
            "renderer_id": id(env._renderer),
            "model_id": id(env.current_model),
            "data_id": id(env.current_data),
            "global_fovy": float(m.vis.global_.fovy),
        },
        "rng": rng_state_digest(),
    }


def build_env(row, retry_index):
    cfg = FrankaSkinHybridObstacleManifestV2Config()
    cfg.task_horizon = 200
    sampler = cfg.task_sampler_config.task_sampler_class(cfg)
    reset_episode_scoped_sampler_state(sampler)
    install_row_seed_contract(row, retry_index, task_sampler=sampler)
    sampler.set_manifest_row(row, retry_index)
    task = sampler.sample_task(house_index=row["scene_template_house_index"])
    if task is None:
        raise SystemExit("sample_task returned None")
    obs = extract_row_observations(task, row)
    return cfg, sampler, task, obs


def render(env, name):
    return np.asarray(env.render_rgb_frame(name))


def pixdiff(a, b) -> dict[str, Any]:
    a = np.asarray(a).astype(np.int32)
    b = np.asarray(b).astype(np.int32)
    if a.shape != b.shape:
        return {"shape_mismatch": [list(a.shape), list(b.shape)]}
    d = np.abs(a - b)
    nz = np.argwhere(d.any(axis=-1)) if d.ndim == 3 else np.argwhere(d != 0)
    return {
        "identical": bool(d.max() == 0),
        "differing_pixels": len(nz),
        "total_pixels": int(np.prod(d.shape[:2])),
        "max_abs_diff": int(d.max()),
        "mean_abs_diff": float(d.mean()),
        "first_differing_pixel": [int(x) for x in nz[0]] if len(nz) else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", required=True, choices=["in-process", "fresh-env", "one-env"])
    ap.add_argument("--collection-manifest", required=True, type=Path)
    ap.add_argument("--eval-manifest", required=True, type=Path)
    ap.add_argument("--episode-id", required=True)
    ap.add_argument("--stack", required=True, type=Path)
    ap.add_argument("--fresh-envs", type=int, default=5)
    ap.add_argument("--repeat-renders", type=int, default=10)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    coll = json.loads(args.collection_manifest.read_text())
    row = next(r for r in coll["rows"] if r["episode_id"] == args.episode_id)
    ev = json.loads(args.eval_manifest.read_text())
    evrow = next(r for r in ev["rows"] if r["episode_id"] == args.episode_id)
    retry = int(evrow["accepted_retry_index"])
    sensors = json.loads(args.stack.read_text())["sensor_contract"]["ordered_names"]

    report: dict[str, Any] = {
        "schema": "hybrid_obstacle_wrist_determinism_probe_v1",
        "mode": args.mode,
        "episode_id": args.episode_id,
        "candidate_index": row["candidate_index"],
        "accepted_retry_index": retry,
        "pid": os.getpid(),
        "probes": {},
        "environment_constructions": 0,
    }

    if args.mode in ("in-process", "one-env"):
        _cfg, sampler, task, _obs = build_env(row, retry)
        report["environment_constructions"] = 1
        env = task.env
        report["captured_state"] = capture_state(env, task, sensors)

        # ---- repeated-render sweep in ONE environment (probe A) ------------
        seq = [render(env, WRIST) for _ in range(args.repeat_renders)]
        hashes = [ah(x) for x in seq]
        first_stable = next((i for i in range(1, len(hashes))
                             if len(set(hashes[i:])) == 1), None)
        report["probes"]["A_repeated_renders_same_env"] = {
            "count": len(hashes),
            "hashes": hashes,
            "all_identical": len(set(hashes)) == 1,
            "distinct_hashes": len(set(hashes)),
            "first_index_from_which_stable": first_stable,
            "render_1_vs_2": pixdiff(seq[0], seq[1]),
            "render_2_vs_3": pixdiff(seq[1], seq[2]) if len(seq) > 2 else None,
            "render_1_vs_last": pixdiff(seq[0], seq[-1]),
        }
        stable_wrist = hashes[-1]

        # exo + proximity baseline for non-regression
        exo = [ah(render(env, EXO)) for _ in range(3)]
        report["probes"]["exo_repeated_renders"] = {
            "hashes": exo, "all_identical": len(set(exo)) == 1}

        if args.mode == "one-env":
            report["step0_contract"] = {
                "wrist_first_render": hashes[0],
                "wrist_stable_render": stable_wrist,
                "exo_first_render": exo[0],
                "qpos": report["captured_state"]["mujoco_state"]["qpos"],
                "cam_xpos_wrist": report["captured_state"]["cameras"][WRIST].get("cam_xpos"),
            }
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
            print(json.dumps(report["step0_contract"], indent=2, sort_keys=True))
            return 0

        # ---- probe E/F: render order ---------------------------------------
        w_first = ah(render(env, WRIST)); _ = render(env, EXO)
        _ = render(env, EXO); w_after = ah(render(env, WRIST))
        report["probes"]["EF_render_order"] = {
            "wrist_before_exo": w_first, "wrist_after_exo": w_after,
            "order_independent": w_first == w_after,
            "both_equal_stable": w_first == stable_wrist == w_after}

        # ---- probe B: restore exact state, render again --------------------
        m, d = env.current_model, env.current_data
        snap = {k: np.array(getattr(d, k), copy=True)
                for k in ("qpos", "qvel", "ctrl", "mocap_pos", "mocap_quat", "qacc_warmstart")}
        t0 = float(d.time)
        restores = []
        for _ in range(5):
            d.qpos[:] = snap["qpos"]; d.qvel[:] = snap["qvel"]; d.ctrl[:] = snap["ctrl"]
            d.mocap_pos[:] = snap["mocap_pos"]; d.mocap_quat[:] = snap["mocap_quat"]
            d.qacc_warmstart[:] = snap["qacc_warmstart"]; d.time = t0
            mujoco.mj_forward(m, d)
            env.camera_manager.registry.update_all_cameras(env)
            restores.append(ah(render(env, WRIST)))
        report["probes"]["B_restore_and_render"] = {
            "hashes": restores, "all_identical": len(set(restores)) == 1,
            "equal_to_stable": all(h == stable_wrist for h in restores)}

        # ---- probe J: forward refresh + camera update only ----------------
        mujoco.mj_forward(m, d)
        env.camera_manager.registry.update_all_cameras(env)
        report["probes"]["J_forward_plus_camera_update"] = {
            "wrist": ah(render(env, WRIST)), "equal_to_stable": ah(render(env, WRIST)) == stable_wrist}

        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
        pa = report["probes"]["A_repeated_renders_same_env"]
        print(f"A repeated renders   : {pa['distinct_hashes']} distinct of {pa['count']}, "
              f"stable from index {pa['first_index_from_which_stable']}")
        print(f"  render 1 vs 2      : {pa['render_1_vs_2']}")
        print(f"  render 2 vs 3      : {pa['render_2_vs_3']}")
        print(f"exo repeated         : identical={report['probes']['exo_repeated_renders']['all_identical']}")
        print(f"EF order independent : {report['probes']['EF_render_order']['order_independent']}")
        print(f"B restore identical  : {report['probes']['B_restore_and_render']['all_identical']}")
        print(f"wrote {args.out}")
        return 0

    # ---- probe C: fresh environment constructions -------------------------
    firsts, stables, exos, qposes, campos = [], [], [], [], []
    for _ in range(args.fresh_envs):
        _cfg, sampler, task, _obs = build_env(row, retry)
        report["environment_constructions"] += 1
        env = task.env
        seq = [ah(render(env, WRIST)) for _ in range(args.repeat_renders)]
        firsts.append(seq[0]); stables.append(seq[-1])
        exos.append(ah(render(env, EXO)))
        st = capture_state(env, task, sensors)
        qposes.append(st["mujoco_state"]["qpos"])
        campos.append(st["cameras"][WRIST].get("cam_xpos"))
        del task, sampler, env
    report["probes"]["C_fresh_environments"] = {
        "count": args.fresh_envs,
        "first_render_hashes": firsts, "first_render_all_identical": len(set(firsts)) == 1,
        "stable_render_hashes": stables, "stable_render_all_identical": len(set(stables)) == 1,
        "exo_hashes": exos, "exo_all_identical": len(set(exos)) == 1,
        "qpos_all_identical": len(set(qposes)) == 1,
        "wrist_cam_xpos_all_identical": len({json.dumps(c) for c in campos}) == 1}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    c = report["probes"]["C_fresh_environments"]
    print(f"C fresh envs         : {c['count']}")
    print(f"  first render same  : {c['first_render_all_identical']}")
    print(f"  stable render same : {c['stable_render_all_identical']}")
    print(f"  exo same           : {c['exo_all_identical']}")
    print(f"  qpos same          : {c['qpos_all_identical']}")
    print(f"  wrist cam_xpos same: {c['wrist_cam_xpos_all_identical']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
