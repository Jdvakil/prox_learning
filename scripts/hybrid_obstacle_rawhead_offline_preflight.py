#!/usr/bin/env python3
"""Offline raw-head preflight over the four development source trajectories.

Handoff step 7. Runs the canonical SafetyHead across every stored frame of the
four development episodes and predicts what the frozen residual controller would
have commanded, without touching the simulator.

The prediction is exact for the *signal* path -- same SafetyHead, same
``label_scale``, same gain/decay/EMA/max_dev/dt, same clipping -- but it is
open-loop: it replays the expert trajectory's proximity stream, so it does not
model the closed-loop feedback that would occur once the correction moves the arm.
It is a screening gate, not a prediction of the live result.

Gate: fail with RAW_HEAD_CONTROLLER_GROSS_REGRESSION when the hazard-absent
development trajectory predicts correction saturation on more than 50% of its real
timesteps.

Nothing is tuned from this analysis.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "submodules" / "act"))

from hybrid_safety_residual import (
    DEFAULT_ACTIVATION_DEPTH_M,
    DEFAULT_DEAD_PIXEL_BELOW_M,
    DEFAULT_DECAY,
    DEFAULT_EMA,
    DEFAULT_GAIN,
    DEFAULT_MAX_DEVIATION,
    ResidualSafetyController,
    sensor_link,
)

KNOWN_SELF_RETURN = ("link5_front_sensor_1", "link5_front_sensor_2")


def canonical_hash(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def load_proximity(h5_path: Path, sensors: list[str]) -> np.ndarray:
    """(T, 40, 8, 8) in canonical order, latest substep per step."""
    with h5py.File(h5_path, "r") as f:
        tk = [k for k in f if k.startswith("traj_")]
        if len(tk) != 1:
            raise SystemExit(f"{h5_path}: expected 1 traj group, found {len(tk)}")
        g = f[tk[0]]["obs/proximity"]
        frames = []
        for s in sensors:
            v = np.asarray(g[s][()], dtype=np.float32)
            frames.append(v[:, -1] if v.ndim == 4 else v)
        return np.stack(frames, axis=1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--development-manifest", required=True, type=Path)
    ap.add_argument("--run", required=True, type=Path, help="source collection run dir")
    ap.add_argument("--stack", required=True, type=Path)
    ap.add_argument("--safety-dir", required=True, type=Path)
    ap.add_argument("--prior-safety-rollout", type=Path, default=None,
                    help="the previously logged first-live rollout, for descriptive comparison")
    ap.add_argument("--dt", type=float, default=0.066)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    import torch
    from train_safety_cvae import SafetyHead

    dev = json.loads(args.development_manifest.read_text())
    stack = json.loads(args.stack.read_text())
    sensors = list(stack["sensor_contract"]["ordered_names"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    head = SafetyHead.load(str(args.safety_dir), device=device)

    rows_out = []
    for row in dev["rows"]:
        h5 = args.run / "rows" / row["episode_id"] / "trajectory.h5"
        prox = load_proximity(h5, sensors)
        T = len(prox)

        ctrl = ResidualSafetyController(
            label_scale=head.scale, dt=args.dt, gain=DEFAULT_GAIN, decay=DEFAULT_DECAY,
            ema=DEFAULT_EMA, max_deviation=DEFAULT_MAX_DEVIATION)
        raw_out, filt, corr, sat, per_joint = [], [], [], [], []
        min_depth, n_active, link_hits, known_hits = [], [], {}, 0
        for t in range(T):
            raw = np.asarray(head(prox[t]), dtype=np.float64)
            step = ctrl.step(raw.astype(np.float32), np.zeros(7, dtype=np.float32))
            raw_out.append(float(np.linalg.norm(raw)))
            per_joint.append(raw.tolist())
            filt.append(float(np.linalg.norm(step.filtered_safety_dq)))
            corr.append(float(np.linalg.norm(step.correction)))
            sat.append(bool(np.max(np.abs(step.correction)) >= DEFAULT_MAX_DEVIATION - 1e-9))
            # sensor attribution on the stored frame
            patch = prox[t]
            valid = patch >= DEFAULT_DEAD_PIXEL_BELOW_M
            mins = np.where(valid.any(axis=(1, 2)),
                            np.where(valid, patch, np.inf).min(axis=(1, 2)), np.inf)
            active = [sensors[i] for i in range(40) if mins[i] < DEFAULT_ACTIVATION_DEPTH_M]
            n_active.append(len(active))
            min_depth.append(float(mins.min()) if np.isfinite(mins).any() else None)
            for s in active:
                link_hits[sensor_link(s)] = link_hits.get(sensor_link(s), 0) + 1
            if set(active) & set(KNOWN_SELF_RETURN):
                known_hits += 1

        pj = np.asarray(per_joint)
        sat_frac = float(np.mean(sat))
        rows_out.append({
            "candidate_index": row["candidate_index"],
            "episode_id": row["episode_id"],
            "hazard_present": bool(row["hazard_present"]),
            "real_timesteps": T,
            "raw_output_norm": {"min": min(raw_out), "max": max(raw_out),
                                "mean": float(np.mean(raw_out)),
                                "median": float(np.median(raw_out))},
            "raw_output_per_joint_abs_max": np.abs(pj).max(axis=0).tolist(),
            "raw_output_per_joint_mean": pj.mean(axis=0).tolist(),
            "predicted_filtered_norm": {"max": max(filt), "mean": float(np.mean(filt))},
            "predicted_correction_norm": {"max": max(corr), "mean": float(np.mean(corr)),
                                          "final": corr[-1]},
            "predicted_saturated_steps": int(np.sum(sat)),
            "predicted_saturation_fraction": sat_frac,
            "first_saturated_step": next((i for i, s in enumerate(sat) if s), None),
            "minimum_proximity_depth_m": {"min": min(d for d in min_depth if d is not None),
                                          "median": float(np.median([d for d in min_depth
                                                                     if d is not None]))},
            "active_sensor_count": {"min": min(n_active), "max": max(n_active),
                                    "mean": float(np.mean(n_active))},
            "link_activation_counts": dict(sorted(link_hits.items(), key=lambda kv: -kv[1])),
            "known_self_return_active_frames": known_hits,
            "correction_norm_series_downsampled": corr[::10],
        })

    present = [r for r in rows_out if r["hazard_present"]]
    absent = [r for r in rows_out if not r["hazard_present"]]

    prior = None
    if args.prior_safety_rollout and args.prior_safety_rollout.is_file():
        pb = json.loads(args.prior_safety_rollout.read_text())["hybrid_safety_stack"]
        pf = pb["frames"]
        prior = {
            "source": str(args.prior_safety_rollout),
            "reference_mode": pb["controller"]["baseline"],
            "note": ("the previously logged first-live controller, development evidence only; "
                     "shown for descriptive comparison and not used to tune anything"),
            "correction_norm_max": max(float(np.linalg.norm(f["correction"])) for f in pf),
            "correction_norm_mean": float(np.mean([np.linalg.norm(f["correction"]) for f in pf])),
            "raw_norm_max": max(float(np.linalg.norm(f["raw_safety_dq"])) for f in pf),
            "reference_norm_constant": float(np.linalg.norm(pf[0]["baseline_safety_output"])),
            "task_success": pb["episode_metrics"]["task_success"],
        }

    gate_row = absent[0] if absent else None
    gate_failed = bool(gate_row and gate_row["predicted_saturation_fraction"] > 0.50)

    report = {
        "schema": "hybrid_obstacle_rawhead_offline_preflight_v1",
        "controller": "RAW_HEAD_RESIDUAL_V1",
        "controller_status": "DEPLOYABLE_CANDIDATE_UNDER_QUALIFICATION",
        "signal": "dq_raw = SafetyHead(current_skin); reference = none; dq_delta = dq_raw",
        "frozen_constants": {"gain": DEFAULT_GAIN, "decay": DEFAULT_DECAY, "ema": DEFAULT_EMA,
                             "max_dev": DEFAULT_MAX_DEVIATION, "dt": args.dt,
                             "label_scale": float(head.scale)},
        "scale_applied_once": ("SafetyHead multiplies by label_scale and the committed controller "
                               "divides by it, so the net drive is normalized label space"),
        "method_caveat": ("Open-loop screening over the stored expert proximity streams. It models "
                          "the signal path exactly but not the closed-loop feedback that arises once "
                          "the correction moves the arm, so it predicts a tendency, not the live "
                          "result."),
        "development_manifest_sha256": dev["manifest_sha256"],
        "rows": rows_out,
        "hazard_present_summary": {
            "rows": len(present),
            "raw_output_norm_max": max(r["raw_output_norm"]["max"] for r in present),
            "predicted_saturation_fraction": [r["predicted_saturation_fraction"] for r in present],
            "predicted_correction_norm_max": [r["predicted_correction_norm"]["max"] for r in present],
        },
        "hazard_absent_summary": {
            "rows": len(absent),
            "raw_output_norm_max": max(r["raw_output_norm"]["max"] for r in absent),
            "predicted_saturation_fraction": [r["predicted_saturation_fraction"] for r in absent],
            "predicted_correction_norm_max": [r["predicted_correction_norm"]["max"] for r in absent],
        },
        "prior_first_live_comparison": prior,
        "gate": {
            "rule": ("fail when the hazard-absent development trajectory predicts correction "
                     "saturation on more than 50% of its real timesteps"),
            "hazard_absent_predicted_saturation_fraction":
                gate_row["predicted_saturation_fraction"] if gate_row else None,
            "threshold": 0.50,
            "failed": gate_failed,
            "decision_if_failed": "RAW_HEAD_CONTROLLER_GROSS_REGRESSION",
        },
        "tuning_performed": False,
    }
    report["report_sha256"] = canonical_hash(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")

    print(f"label_scale {head.scale:.6f} | device {device}")
    print(f"{'cand':>5} {'hazard':>8} {'T':>4} {'raw|max':>9} {'corr|max':>9} "
          f"{'sat%':>6} {'firstSat':>9} {'nAct':>5} {'known':>6}")
    for r in rows_out:
        print(f"{r['candidate_index']:5d} {'present' if r['hazard_present'] else 'absent':>8} "
              f"{r['real_timesteps']:4d} {r['raw_output_norm']['max']:9.4f} "
              f"{r['predicted_correction_norm']['max']:9.5f} "
              f"{100*r['predicted_saturation_fraction']:6.1f} "
              f"{r['first_saturated_step']!s:>9} "
              f"{r['active_sensor_count']['max']:5d} {r['known_self_return_active_frames']:6d}")
    print()
    print(f"hazard-absent predicted saturation: "
          f"{100*(gate_row['predicted_saturation_fraction'] if gate_row else 0):.1f}% "
          f"(threshold 50%)")
    print("GATE FAILED -> RAW_HEAD_CONTROLLER_GROSS_REGRESSION" if gate_failed
          else "GATE PASSED -> live execution permitted")
    print(f"wrote {args.out}")
    return 2 if gate_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
