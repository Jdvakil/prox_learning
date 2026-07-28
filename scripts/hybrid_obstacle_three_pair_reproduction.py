#!/usr/bin/env python3
"""Golden regression: reproduce the identifiability audit's agreement values.

Handoff step 4. Before any calibration, the three-pair metric must reproduce what the audit
recorded on the 17 historical frames. If it cannot, the metric being calibrated is not the
metric that was validated, and the whole premise of this task fails.

The handoff specifies a ``>= 0.5`` mask comparison while the audit used ``> 0.5``. Both are
computed here and compared bitwise: the distinction only matters for probabilities exactly
equal to 0.5, which is why the reproduction is checked rather than assumed either way.

These frames are used for verification only and never to select a threshold.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from causal_parked_skin import threshold as thr
from causal_parked_skin.data import load_partition
from causal_parked_skin.engine import load_checkpoint, make_batch, set_sensor_names
from causal_parked_skin.joint_gate import (
    PIXEL_MASK_THRESHOLD,
    anchor_mask_agreement,
    changed_mask,
    jaccard,
    three_pair_agreement,
)
from causal_parked_skin.model import BASELINE_CURRENT, FrozenSafetyHead

CONSUMED_PARTITIONS = ("offline_reference_test", "reference_calibration",
                       "reference_validation")

# values recorded by the identifiability audit and the two-anchor joint-gate task
EXPECTED_THREE_PAIR_MEDIAN = 0.167
EXPECTED_ANCHOR_MEDIAN = 0.250
MEDIAN_TOLERANCE = 0.01
NEAR_ZERO = 1e-9


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", required=True, type=Path)
    ap.add_argument("--checkpoint-root", required=True, type=Path)
    ap.add_argument("--stack", required=True, type=Path)
    ap.add_argument("--safety-dir", required=True, type=Path)
    ap.add_argument("--onset-audit", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    import torch

    stack = json.loads(args.stack.read_text())
    set_sensor_names(stack["sensor_contract"]["ordered_names"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    FrozenSafetyHead.load(args.safety_dir, device=device)

    models = []
    for seed in (0, 1, 2):
        path = args.checkpoint_root / f"CURRENT_FRAME_ONLY__seed{seed}" / "best.pt"
        model, payload = load_checkpoint(path, device)
        if payload["config"]["seed"] != seed or \
                payload["config"]["variant"] != BASELINE_CURRENT:
            raise SystemExit(f"seed {seed} checkpoint mismatch")
        models.append(model)

    historical = json.loads(args.onset_audit.read_text())["known_false_positive_frames"]
    wanted = {(r["trajectory_id"], r["step"]) for r in historical}
    if len(wanted) != 17:
        raise SystemExit(f"expected 17 historical frames, found {len(wanted)}")

    rows = []
    strict_vs_inclusive_identical = True
    for name in CONSUMED_PARTITIONS:
        partition = load_partition(args.cache, name)
        trajectory_index = np.asarray(partition["trajectory"])
        for index, trajectory_id in enumerate(partition.trajectory_ids):
            steps = [s for (t, s) in wanted if t == trajectory_id]
            if not steps:
                continue
            frame_rows = np.flatnonzero(trajectory_index == index)
            for step in sorted(steps):
                batch = make_batch(partition, frame_rows[step:step + 1], device)
                with torch.no_grad():
                    probabilities = [
                        m(batch["history"], batch["history_valid"],
                          batch["state"])["changed_probability"].cpu().numpy()
                        for m in models]
                strict = [changed_mask(p) for p in probabilities]
                inclusive = [p.reshape(1, -1) >= PIXEL_MASK_THRESHOLD
                             for p in probabilities]
                if not all(np.array_equal(a, b) for a, b in zip(strict, inclusive)):
                    strict_vs_inclusive_identical = False
                m0, m1, m2 = strict
                rows.append({
                    "trajectory_id": trajectory_id, "step": step,
                    "seed0_activity": float(probabilities[0].reshape(1, -1).max()),
                    "seed0_mask_pixels": int(m0.sum()),
                    "seed1_mask_pixels": int(m1.sum()),
                    "seed2_mask_pixels": int(m2.sum()),
                    "jaccard_01": float(jaccard(m0, m1)[0]),
                    "jaccard_02": float(jaccard(m0, m2)[0]),
                    "jaccard_12": float(jaccard(m1, m2)[0]),
                    "anchor_agreement": float(anchor_mask_agreement(m0, m1, m2)[0]),
                    "three_pair_agreement": float(three_pair_agreement(m0, m1, m2)[0]),
                })
    if len(rows) != 17:
        raise SystemExit(f"scored {len(rows)} historical frames, expected 17")

    j02 = np.array([r["jaccard_02"] for r in rows])
    j12 = np.array([r["jaccard_12"] for r in rows])
    anchor = np.array([r["anchor_agreement"] for r in rows])
    three = np.array([r["three_pair_agreement"] for r in rows])

    checks = {
        "seventeen_frames_scored": len(rows) == 17,
        "j02_zero_on_all": bool((j02 <= NEAR_ZERO).all()),
        "j12_near_zero_on_all": bool((j12 <= NEAR_ZERO).all()),
        "three_pair_median_matches_0_167": bool(
            abs(float(np.median(three)) - EXPECTED_THREE_PAIR_MEDIAN)
            <= MEDIAN_TOLERANCE),
        "anchor_median_matches_0_250": bool(
            abs(float(np.median(anchor)) - EXPECTED_ANCHOR_MEDIAN) <= MEDIAN_TOLERANCE),
        "three_pair_below_anchor_everywhere": bool((three <= anchor + 1e-12).all()),
    }
    report = {
        "schema": "hybrid_obstacle_three_pair_reproduction_v1",
        "purpose": "implementation verification only; never used to select thresholds",
        "frames": rows,
        "frame_count": len(rows),
        "observed": {
            "jaccard_02_max": float(j02.max()),
            "jaccard_12_max": float(j12.max()),
            "anchor_median": float(np.median(anchor)),
            "anchor_min": float(anchor.min()), "anchor_max": float(anchor.max()),
            "three_pair_median": float(np.median(three)),
            "three_pair_min": float(three.min()), "three_pair_max": float(three.max()),
        },
        "expected": {"three_pair_median": EXPECTED_THREE_PAIR_MEDIAN,
                     "anchor_median": EXPECTED_ANCHOR_MEDIAN,
                     "tolerance": MEDIAN_TOLERANCE},
        "mask_comparison": {
            "used": "strict greater-than, matching the identifiability audit",
            "handoff_specified": ">= 0.5",
            "strict_and_inclusive_masks_identical_on_these_frames":
                strict_vs_inclusive_identical,
            "note": ("the two differ only where a probability is exactly 0.5; they were "
                     "compared bitwise rather than assumed equivalent"),
        },
        "checks": checks,
        "reproduced": all(checks.values()),
        "decision_if_failed": "THREE_PAIR_AGREEMENT_REPRODUCTION_FAILED",
    }
    report["report_sha256"] = thr.canonical_hash(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")

    print(f"{'J01':>7}{'J02':>7}{'J12':>7}{'anchor':>9}{'3-pair':>9}")
    for r in sorted(rows, key=lambda x: -x["three_pair_agreement"]):
        print(f"{r['jaccard_01']:>7.3f}{r['jaccard_02']:>7.3f}{r['jaccard_12']:>7.3f}"
              f"{r['anchor_agreement']:>9.3f}{r['three_pair_agreement']:>9.3f}")
    print(f"\nanchor  median {np.median(anchor):.4f} (expected ~{EXPECTED_ANCHOR_MEDIAN})")
    print(f"3-pair  median {np.median(three):.4f} "
          f"(expected ~{EXPECTED_THREE_PAIR_MEDIAN})")
    print(f"3-pair  range  {three.min():.4f} .. {three.max():.4f}")
    for name, ok in checks.items():
        print(f"  [{'ok  ' if ok else 'FAIL'}] {name}")
    print(f"strict/inclusive masks identical: {strict_vs_inclusive_identical}")
    print(f"reproduced: {report['reproduced']}")
    print(f"wrote {args.out}")
    return 0 if report["reproduced"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
