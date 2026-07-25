#!/usr/bin/env python3
"""Offline full-sequence replay of the selected deployable reference.

Handoff step 11. Before any live rollout, the frozen reference and support gate are run
across the four development source trajectories with the complete frozen residual
dynamics, open loop. This is a screening gate, not a prediction: it replays the expert
proximity stream, so it models the signal and controller paths exactly but not the
closed-loop feedback that arises once the correction moves the arm.

Fails with ``DEPLOYABLE_REFERENCE_OFFLINE_INVALID`` when

* candidate 118 predicts a nonzero correction on more than 2% of frames,
* median teacher-active deployable/oracle cosine is below 0.70,
* any rollout predicts saturation on more than 75% of frames, or
* the predicted correction is primarily active outside the <0.18 m support range.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for extra in (str(ROOT / "scripts"), str(ROOT / "submodules" / "act")):
    if extra not in sys.path:
        sys.path.insert(0, extra)

from deployable_reference import (
    D_ACT,
    FEATURE_BUILDERS,
    load_reference,
)
from hybrid_safety_residual import (
    DEFAULT_DECAY,
    DEFAULT_EMA,
    DEFAULT_GAIN,
    DEFAULT_MAX_DEVIATION,
    ResidualSafetyController,
)

LABEL_SCALE = 11.359346389770508
DT = 0.066


def canonical_hash(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def build_pseudo_split(development_manifest: Path, out_path: Path) -> Path:
    """A split-manifest-shaped view of the four development rows, for the generator.

    The development rows are not part of the canonical 100, so the paired dataset
    generator needs an episode list for them. Marked ``development`` so it can never be
    mistaken for a training or validation partition.
    """
    dev = json.loads(development_manifest.read_text())
    if dev.get("role") != "DEVELOPMENT_ONLY":
        raise SystemExit(f"refusing a manifest whose role is {dev.get('role')!r}")
    payload = {
        "schema": "hybrid_obstacle_development4_paired_episodes_v1",
        "derived_from": str(development_manifest),
        "development_manifest_sha256": dev["manifest_sha256"],
        "episodes": [{
            "episode_id": row["episode_id"],
            "candidate_index": row["candidate_index"],
            "hazard_present": bool(row["hazard_present"]),
            "source_h5_sha256": row["source_h5_sha256"],
            "accepted_retry_index": row["accepted_retry_index"],
            "split": "development",
            "split_rank": index,
        } for index, row in enumerate(sorted(dev["rows"],
                                             key=lambda r: r["candidate_index"]))],
    }
    payload["episodes_sha256"] = canonical_hash(payload["episodes"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--development-manifest", required=True, type=Path)
    ap.add_argument("--paired-dir", required=True, type=Path)
    ap.add_argument("--reference-manifest", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    for name in ("development_manifest", "paired_dir", "reference_manifest", "out"):
        setattr(args, name, Path(getattr(args, name)).resolve())

    dev = json.loads(args.development_manifest.read_text())
    model, gate, manifest = load_reference(args.reference_manifest, device="cpu")
    builder = FEATURE_BUILDERS[manifest["reference_type"]]

    rows_out = []
    for row in sorted(dev["rows"], key=lambda r: r["candidate_index"]):
        path = args.paired_dir / f"{row['episode_id']}.npz"
        if not path.is_file():
            raise SystemExit(f"paired development example missing: {path}")
        blob = np.load(path, allow_pickle=False)
        count = len(blob["timestep"])

        features = builder({k: blob[k] for k in blob.files if k in
                            {"qpos", "qvel", "nominal_action", "gripper_state",
                             "gripper_command", "current_head", "sensor_summary"}})
        predicted_parked = np.asarray(model.predict(features), dtype=np.float32)
        current = blob["current_head"]
        oracle = blob["oracle_dq"]
        predicted_dq = (current - predicted_parked).astype(np.float32)

        controller = ResidualSafetyController(
            label_scale=LABEL_SCALE, dt=DT, gain=DEFAULT_GAIN, decay=DEFAULT_DECAY,
            ema=DEFAULT_EMA, max_deviation=DEFAULT_MAX_DEVIATION)
        corrections, saturated, activations, gate_logs = [], [], [], []
        for index in range(count):
            gated, log = gate(predicted_dq[index], float(blob["minimum_depth"][index]))
            step = controller.step(gated.astype(np.float32), np.zeros(7, dtype=np.float32))
            corrections.append(float(np.linalg.norm(step.correction)))
            saturated.append(bool(np.max(np.abs(step.correction))
                                  >= DEFAULT_MAX_DEVIATION - 1e-9))
            activations.append(bool(log["activated"]))
            gate_logs.append(log)

        activations = np.asarray(activations)
        supported = blob["minimum_depth"] < D_ACT
        teacher_active = blob["teacher_active"].astype(bool)
        oracle_active = np.linalg.norm(oracle, axis=1) > 1e-9
        norms = np.linalg.norm(predicted_dq, axis=1)
        oracle_norms = np.linalg.norm(oracle, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            cosines = np.where(
                (norms > 1e-12) & (oracle_norms > 1e-12),
                (predicted_dq * oracle).sum(axis=1) / (norms * oracle_norms), np.nan)
        active_cosines = cosines[teacher_active & np.isfinite(cosines)]
        oracle_cosines = cosines[oracle_active & np.isfinite(cosines)]

        runs, current_run = [], 0
        for flag in activations:
            if flag:
                current_run += 1
            elif current_run:
                runs.append(current_run)
                current_run = 0
        if current_run:
            runs.append(current_run)

        rows_out.append({
            "candidate_index": row["candidate_index"],
            "episode_id": row["episode_id"],
            "hazard_present": bool(row["hazard_present"]),
            "frames": int(count),
            "correction_norm": {"max": max(corrections), "mean": float(np.mean(corrections)),
                                "median": float(np.median(corrections)),
                                "final": corrections[-1]},
            "saturation_fraction": float(np.mean(saturated)),
            "activation_rate": float(np.mean(activations)),
            "activations": int(activations.sum()),
            "intervention_runs": runs,
            "longest_intervention_run": max(runs, default=0),
            "predicted_intervention_duration_s": float(activations.sum() * DT),
            "gripper_preserved_by_construction": True,
            "activation_inside_support_range": (
                float(np.mean(supported[activations])) if activations.any() else None),
            "activation_outside_support_range": (
                float(np.mean(~supported[activations])) if activations.any() else None),
            "teacher_active_frames": int(teacher_active.sum()),
            "oracle_nonzero_frames": int(oracle_active.sum()),
            "cosine_teacher_active": {
                "n": int(active_cosines.size),
                "median": float(np.median(active_cosines)) if active_cosines.size else None,
                "fraction_positive": (float(np.mean(active_cosines > 0))
                                      if active_cosines.size else None)},
            "cosine_oracle_nonzero": {
                "n": int(oracle_cosines.size),
                "median": float(np.median(oracle_cosines)) if oracle_cosines.size else None,
                "fraction_positive": (float(np.mean(oracle_cosines > 0))
                                      if oracle_cosines.size else None)},
            "far_range_activation_rate": (
                float(np.mean(activations[~supported])) if (~supported).any() else None),
            "predicted_norm": {"max": float(norms.max()),
                               "median": float(np.median(norms)),
                               "p99": float(np.percentile(norms, 99))},
            "correction_norm_series_downsampled": corrections[::5],
        })

    absent = [r for r in rows_out if not r["hazard_present"]]
    present = [r for r in rows_out if r["hazard_present"]]
    pooled_cosines = [c for r in present
                      for c in ([r["cosine_teacher_active"]["median"]]
                                if r["cosine_teacher_active"]["median"] is not None else [])]

    gates = {
        "hazard_absent_activation_within_2pct": {
            "rule": "candidate 118 must not predict a nonzero correction on more than 2% "
                    "of frames",
            "values": {r["candidate_index"]: r["activation_rate"] for r in absent},
            "threshold": 0.02,
            "passed": all(r["activation_rate"] <= 0.02 for r in absent),
        },
        "median_teacher_active_cosine_at_least_0p70": {
            "rule": "median teacher-active deployable/oracle cosine >= 0.70",
            "per_row": {r["candidate_index"]: r["cosine_teacher_active"]["median"]
                        for r in present},
            "pooled_median": (float(np.median(pooled_cosines)) if pooled_cosines else None),
            "evaluable": bool(pooled_cosines),
            "threshold": 0.70,
            "passed": bool(pooled_cosines) and float(np.median(pooled_cosines)) >= 0.70,
        },
        "no_rollout_saturated_over_75pct": {
            "values": {r["candidate_index"]: r["saturation_fraction"] for r in rows_out},
            "threshold": 0.75,
            "passed": all(r["saturation_fraction"] <= 0.75 for r in rows_out),
        },
        "correction_primarily_inside_support_range": {
            "rule": "the predicted correction must not be primarily active outside the "
                    "<0.18 m support range",
            "values": {r["candidate_index"]: r["activation_outside_support_range"]
                       for r in rows_out},
            "passed": all((r["activation_outside_support_range"] or 0.0) <= 0.5
                          for r in rows_out),
        },
    }
    all_passed = all(g["passed"] for g in gates.values())

    report = {
        "schema": "hybrid_obstacle_deployable_offline_replay_v1",
        "reference_type": manifest["reference_type"],
        "reference_manifest_sha256": manifest["manifest_sha256"],
        "tau": manifest["tau"],
        "d_act": D_ACT,
        "frozen_constants": {"gain": DEFAULT_GAIN, "decay": DEFAULT_DECAY,
                             "ema": DEFAULT_EMA, "max_dev": DEFAULT_MAX_DEVIATION,
                             "dt": DT, "label_scale": LABEL_SCALE},
        "method_caveat": ("open-loop screening over the recorded expert proximity streams; "
                          "it models the signal and controller paths exactly but not the "
                          "closed-loop feedback once the correction moves the arm"),
        "development_manifest_sha256": json.loads(
            args.development_manifest.read_text())["manifest_sha256"],
        "rows": rows_out,
        "gates": gates,
        "all_gates_passed": all_passed,
        "decision_if_failed": "DEPLOYABLE_REFERENCE_OFFLINE_INVALID",
        "tuning_performed": False,
    }
    report["report_sha256"] = canonical_hash(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")

    print(f"{'cand':>5} {'hazard':>8} {'T':>4} {'act%':>6} {'corr|max':>9} {'sat%':>6} "
          f"{'cos(act)':>9} {'n':>4} {'far%':>6}")
    for entry in rows_out:
        print(f"{entry['candidate_index']:5d} "
              f"{'present' if entry['hazard_present'] else 'absent':>8} "
              f"{entry['frames']:4d} {100*entry['activation_rate']:6.1f} "
              f"{entry['correction_norm']['max']:9.5f} "
              f"{100*entry['saturation_fraction']:6.1f} "
              f"{entry['cosine_teacher_active']['median']!s:>9} "
              f"{entry['cosine_teacher_active']['n']:4d} "
              f"{100*(entry['far_range_activation_rate'] or 0):6.1f}")
    print()
    for name, gate in gates.items():
        print(f"  [{'PASS' if gate['passed'] else 'FAIL'}] {name}")
    print(f"\n{'all offline replay gates passed' if all_passed else 'GATE FAILED -> DEPLOYABLE_REFERENCE_OFFLINE_INVALID'}")
    print(f"wrote {args.out}")
    return 0 if all_passed else 5


if __name__ == "__main__":
    raise SystemExit(main())
