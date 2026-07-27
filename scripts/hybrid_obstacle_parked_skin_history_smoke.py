#!/usr/bin/env python3
"""Four-frame history and SafetyHead reconstruction smoke on the finished dataset.

Handoff step 18. Loads at least one trajectory from every distribution, reconstructs all
causal windows, and pushes the stored current and parked fields back through the frozen
SafetyHead to confirm the recorded 7-D targets.

``CAUSAL_PARKED_SKIN_REFERENCE_V1`` is imported and shape-checked only. It is never
instantiated with data or trained here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submodules" / "act"))

from parked_skin_retention import (
    CAUSAL_FRAMES,
    TOLERANCES,
    closeness_to_depth,
    load_trajectory,
    reconstruct_all_histories,
)


def canonical_hash(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--safety-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    import torch
    from train_safety_cvae import SafetyHead

    manifest = json.loads(Path(args.manifest).read_text())
    head = SafetyHead.load(str(args.safety_dir),
                           device="cuda" if torch.cuda.is_available() else "cpu")

    checked = []
    for distribution in sorted({e["distribution"] for e in manifest["entries"]}):
        entry = next((e for e in manifest["entries"]
                      if e["distribution"] == distribution
                      and Path(e["output"]).is_file()), None)
        if entry is None:
            checked.append({"distribution": distribution, "available": False})
            continue
        loaded = load_trajectory(Path(entry["output"]), allow_privileged=True)
        current = loaded["current_closeness"]
        parked = loaded["parked_closeness"]
        frames = len(current)

        windows, sources = reconstruct_all_histories(current)
        head_current = np.stack([head(closeness_to_depth(current[t]))
                                 for t in range(frames)])
        head_parked = np.stack([head(closeness_to_depth(parked[t]))
                                for t in range(frames)])
        head_delta = max(float(np.abs(head_current - loaded["current_head"]).max()),
                         float(np.abs(head_parked - loaded["parked_head"]).max()))
        oracle_delta = float(np.abs((head_current - head_parked)
                                    - loaded["oracle_dq"]).max())

        checked.append({
            "distribution": distribution,
            "available": True,
            "episode_id": entry["episode_id"],
            "frames": frames,
            "history_shape": list(windows.shape),
            "history_shape_is_N4x40x8x8": list(windows.shape) == [frames, CAUSAL_FRAMES,
                                                                  40, 8, 8],
            "last_history_frame_equals_current": bool(
                np.array_equal(windows[:, -1], current)),
            "no_future_frame": bool((sources <= np.arange(frames)[:, None]).all()),
            "first_window_sources": sources[0].tolist(),
            "fourth_window_sources": sources[min(3, frames - 1)].tolist(),
            "trajectory_scoped": bool(sources.max() < frames),
            "head_reconstruction_max_abs_delta": head_delta,
            "head_within_tolerance": head_delta <= TOLERANCES["head_output_max_abs_delta"],
            "oracle_reconstruction_max_abs_delta": oracle_delta,
            "oracle_within_tolerance":
                oracle_delta <= TOLERANCES["oracle_differential_max_abs_delta"],
        })

    # import-and-shape only; never instantiated with data, never trained
    from parked_skin_reference import PARAMETER_BUDGET, build_model, parameter_count

    model_check = {
        "imported": True,
        "instantiated_with_data": False,
        "trained": False,
        "parameters": parameter_count(build_model()),
        "within_budget": parameter_count(build_model()) < PARAMETER_BUDGET,
    }

    report = {
        "schema": "hybrid_obstacle_parked_skin_history_smoke_v1",
        "manifest_sha256": manifest["manifest_sha256"],
        "distributions_checked": checked,
        "all_distributions_available": all(c["available"] for c in checked),
        "all_histories_correct": all(
            c.get("history_shape_is_N4x40x8x8") and c.get("last_history_frame_equals_current")
            and c.get("no_future_frame") and c.get("trajectory_scoped")
            for c in checked if c["available"]),
        "all_head_targets_reproduce": all(
            c.get("head_within_tolerance") and c.get("oracle_within_tolerance")
            for c in checked if c["available"]),
        "model_check": model_check,
        "tolerances": TOLERANCES,
    }
    report["report_sha256"] = canonical_hash(report)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True,
                                         default=str) + "\n")

    for entry in checked:
        if not entry["available"]:
            print(f"  {entry['distribution']:<28} NOT AVAILABLE")
            continue
        print(f"  {entry['distribution']:<28} T={entry['frames']:>4} "
              f"hist={entry['history_shape_is_N4x40x8x8']} "
              f"last==cur={entry['last_history_frame_equals_current']} "
              f"causal={entry['no_future_frame']} "
              f"headΔ={entry['head_reconstruction_max_abs_delta']:.2e} "
              f"oracleΔ={entry['oracle_reconstruction_max_abs_delta']:.2e}")
    print(f"model imported, params {model_check['parameters']}, trained=False")
    print(f"wrote {args.out}")
    return 0 if (report["all_distributions_available"]
                 and report["all_histories_correct"]
                 and report["all_head_targets_reproduce"]) else 10


if __name__ == "__main__":
    raise SystemExit(main())
