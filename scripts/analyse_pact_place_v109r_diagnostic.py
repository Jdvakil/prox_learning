#!/usr/bin/env python3
"""V10.9R step 6: compare the original and repaired decoders on the F2 rows.

Touch and hold are read from the retained trajectories, not from the policy's
own funnel telemetry, because that telemetry did not resolve the grasp-state
sensor on this run. Reading the trajectory is the same method used to establish
the V10.9 baseline, so the two are directly comparable.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from pact_place_v109_contract import (  # noqa: E402
    canonical_payload_sha256, empty_authorization, write_immutable_create_only,
)
from pact_place_v109r_contract import (  # noqa: E402
    ARM_SLEW_LIMIT_L2, CONTRACT_VERSION_V109R, DIAGNOSTIC_ROOT, V109_BASELINE,
)

ARMS = ("act", "pact")
DECODERS = ("legacy", "event")


def grasp_state(group) -> tuple[np.ndarray, np.ndarray]:
    blobs = group["obs/extra/grasp_state_pickup_obj"][()]
    touching = np.zeros(len(blobs), bool)
    held = np.zeros(len(blobs), bool)
    for i, b in enumerate(blobs):
        raw = bytes(b).split(b"\x00", 1)[0]
        if not raw:
            continue
        try:
            state = json.loads(raw)
        except Exception:  # noqa: BLE001
            continue
        for entry in state.values():
            if isinstance(entry, dict):
                touching[i] |= bool(entry.get("touching"))
                held[i] |= bool(entry.get("held"))
    return touching, held


def analyse_row(directory: Path) -> dict[str, Any]:
    result = json.loads((directory / "result.json").read_text())
    info = result["policy_info"]
    summary = info.get("v109r_decoder_summary", {})
    payload = np.load(directory / "raw_chunks.npz")
    decoded = payload["decoded_command"]
    legacy = payload["legacy_command"]
    grip = decoded[:, 7]
    closed = np.flatnonzero(grip >= 127.5)
    with h5py.File(directory / "trajectory.h5", "r") as handle:
        group = handle[[k for k in handle if k.startswith("traj_")][0]]
        touching, held = grasp_state(group)
        tcp = np.asarray(group["obs/extra/tcp_pose"][:, :3], dtype=float)
    t = np.flatnonzero(touching)
    touch = int(t[0]) if t.size else None
    close = int(closed[0]) if closed.size else None
    travel = None
    if touch is not None and close is not None and close > touch:
        travel = round(1000 * float(
            np.linalg.norm(np.diff(tcp[touch:close + 1], axis=0), axis=1).sum()), 1)
    arm_deltas = np.linalg.norm(np.diff(decoded[:, :7], axis=0), axis=1)
    return {
        "candidate_index": int(directory.name.split("_")[0]),
        "touched": bool(t.size), "held": bool(held.any()),
        "held_steps": int(held.sum()),
        "task_success": bool(result["task_success"]),
        "strict_clean": bool(result["task_success"]) and all(
            int(result["contact_audit"]["contact_class_totals"].get(k, 0)) == 0
            for k in ("clutter", "mounted_fixture", "hazard_bar", "other_environment")),
        "first_touch_step": touch, "close_step": close,
        "touch_to_close_controls": (close - touch) if (touch is not None and close is not None) else None,
        "tcp_travel_after_touch_mm": travel,
        "released": summary.get("released"),
        "release_step": summary.get("release_step"),
        "closure_without_release": summary.get("closure_without_release"),
        "max_arm_step_delta": float(arm_deltas.max()) if arm_deltas.size else 0.0,
        "slew_clamped_steps": summary.get("slew_clamped_steps"),
        "max_legacy_vs_decoded_delta": info.get("v109r_max_legacy_vs_decoded_command_delta"),
        "raw_chunks_retained": int(payload["raw_chunks"].shape[0]),
        "legacy_close_step": (lambda c: int(c[0]) if c.size else None)(
            np.flatnonzero(legacy[:, 7] >= 127.5)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / DIAGNOSTIC_ROOT)
    args = parser.parse_args()
    cells: dict[str, dict[str, list]] = {}
    for decoder in DECODERS:
        cells[decoder] = {}
        for arm in ARMS:
            rows = [analyse_row(d) for d in sorted((args.root / decoder / arm).iterdir())
                    if (d / "result.json").is_file()]
            cells[decoder][arm] = sorted(rows, key=lambda r: r["candidate_index"])

    def agg(rows: list[dict[str, Any]]) -> dict[str, Any]:
        delays = [r["touch_to_close_controls"] for r in rows
                  if r["touch_to_close_controls"] is not None and r["touch_to_close_controls"] > 0]
        travels = [r["tcp_travel_after_touch_mm"] for r in rows if r["tcp_travel_after_touch_mm"]]
        failed = [r for r in rows if not r["held"]]
        failed_delays = [r["touch_to_close_controls"] for r in failed
                         if r["touch_to_close_controls"] is not None and r["touch_to_close_controls"] > 0]
        failed_travels = [r["tcp_travel_after_touch_mm"] for r in failed
                          if r["tcp_travel_after_touch_mm"]]
        closed = [r for r in rows if r["close_step"] is not None]
        return {
            "n": len(rows),
            "touched": sum(1 for r in rows if r["touched"]),
            "held": sum(1 for r in rows if r["held"]),
            "task_success": sum(1 for r in rows if r["task_success"]),
            "strict_clean": sum(1 for r in rows if r["strict_clean"]),
            "closed": len(closed),
            "released": sum(1 for r in rows if r["released"]),
            "closure_without_release": sum(1 for r in rows if r["closure_without_release"]),
            "touch_to_close_median": int(st.median(delays)) if delays else None,
            "touch_to_close_median_failed": int(st.median(failed_delays)) if failed_delays else None,
            "tcp_travel_median_mm": round(st.median(travels), 1) if travels else None,
            "tcp_travel_median_failed_mm": round(st.median(failed_travels), 1) if failed_travels else None,
            "max_arm_step_delta": round(max((r["max_arm_step_delta"] for r in rows), default=0.0), 6),
            "arm_slew_limit_l2": ARM_SLEW_LIMIT_L2,
            "arm_discontinuity_exceeded_limit": any(
                r["max_arm_step_delta"] > ARM_SLEW_LIMIT_L2 + 1e-6 for r in rows),
            "slew_clamped_steps_total": sum(r["slew_clamped_steps"] or 0 for r in rows),
        }

    summary = {d: {a: agg(cells[d][a]) for a in ARMS} for d in DECODERS}
    stop = []
    for arm in ARMS:
        e = summary["event"][arm]
        if e["arm_discontinuity_exceeded_limit"]:
            stop.append(f"{arm}: arm discontinuity exceeded the training maximum")
        if e["closure_without_release"] and e["closure_without_release"] == e["closed"] and e["closed"]:
            stop.append(f"{arm}: every closure failed to release")
    document = {
        **empty_authorization(),
        "schema_version": "pact_place_v109r_diagnostic_analysis_v1",
        "contract_version": CONTRACT_VERSION_V109R,
        "role": "F2 development diagnostic, original vs repaired decoder",
        "is_development_only": True,
        "not_evidence_for_the_repair": True,
        "v109_baseline": V109_BASELINE,
        "summary": summary,
        "rows": cells,
        "stop_conditions_triggered": stop,
    }
    document["payload_sha256"] = canonical_payload_sha256(document)
    write_immutable_create_only(args.root / "diagnostic_analysis.json", document)

    print(f"{'':7s} {'arm':5s} {'touch':>6s} {'held':>5s} {'succ':>5s} {'clean':>6s} "
          f"{'closed':>7s} {'rel':>4s} {'t2c':>5s} {'t2c_f':>6s} {'trav_f':>7s} {'maxΔ':>8s}")
    for decoder in DECODERS:
        for arm in ARMS:
            s = summary[decoder][arm]
            print(f"{decoder:7s} {arm:5s} {s['touched']:6d} {s['held']:5d} "
                  f"{s['task_success']:5d} {s['strict_clean']:6d} {s['closed']:7d} "
                  f"{s['released']:4d} {str(s['touch_to_close_median']):>5s} "
                  f"{str(s['touch_to_close_median_failed']):>6s} "
                  f"{str(s['tcp_travel_median_failed_mm']):>7s} {s['max_arm_step_delta']:8.4f}")
    print(f"\nstop conditions triggered: {stop or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
