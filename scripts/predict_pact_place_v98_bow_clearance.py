#!/usr/bin/env python3
"""Pre-flight gate: offset-pendant clearance from live ``_bow_segment``.

No expert episode runs until both named candidates pass. The waypoint is
taken from the real planner, not a hand-copied formula.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "scripts", ROOT / "submodules" / "molmospaces"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pact_place_v98_pendant_contract import (  # noqa: E402
    CONTRACT_VERSION,
    OFFSET_CANDIDATES,
    WRIST_LAG_NEG_M,
    WRIST_LAG_POS_M,
    WRIST_LAG_PROVENANCE,
    build_pendant_fixture,
    ceiling_fixture_bow_policy_constants,
    pendant_admissible_faces_m,
    pendant_faces_m,
    validate_pendant_geometry,
)

CROSS_Y_SAMPLES_M = (-0.05, 0.0, 0.05)
CLIP_EPS_M = 1e-9


def _synthetic_policy(ap_w: float):
    from molmo_spaces.policy.solvers.object_manipulation.base_object_manipulation_planner_policy import (  # noqa: E402
        TCPMoveSegment,
    )
    from molmo_spaces.tasks.enclosure_reach import PactPlaceCorridorPolicy

    policy = PactPlaceCorridorPolicy.__new__(PactPlaceCorridorPolicy)
    policy.task = SimpleNamespace(
        scene_params={
            "ap_w": float(ap_w),
            "pact_place_environment_version": "pact_place_corridor_v9_8_pendant",
        }
    )
    policy._pact_place_bow_diagnostics = (
        PactPlaceCorridorPolicy._empty_bow_diagnostics()
    )
    policy.policy_config = SimpleNamespace(speed_fast=0.12, speed_slow=0.045)
    return policy, TCPMoveSegment, PactPlaceCorridorPolicy


def _segment(TCPMoveSegment, *, start_y: float, end_y: float):
    start = np.eye(4)
    start[:3, 3] = [0.40, float(start_y), 0.885]
    end = np.eye(4)
    end[:3, 3] = [0.95, float(end_y), 0.885]
    return TCPMoveSegment(
        name="synthetic_inbound",
        start_pose=start,
        end_pose=end,
        speed=0.12,
    )


def predict_candidate(
    *,
    name: str,
    center_y_m: float,
    half_y_m: float,
    bottom_z_m: float,
) -> dict[str, Any]:
    constants = ceiling_fixture_bow_policy_constants()
    a_min, b_max = pendant_admissible_faces_m(constants)
    a_face, b_face = pendant_faces_m(center_y_m, half_y_m)
    gap = float(constants["safe_gap_m"])
    env_neg = float(constants["envelope_half_y_neg_m"])
    env_pos = float(constants["envelope_half_y_pos_m"])
    lag_neg = float(constants["wrist_lag_neg_m"])
    lag_pos = float(constants["wrist_lag_pos_m"])
    ap_w = float(constants["aperture_width_m"])
    inside_window = (a_face + CLIP_EPS_M >= a_min) and (b_face - CLIP_EPS_M <= b_max)
    fixture_ok = True
    fixture_error = None
    try:
        fixture = build_pendant_fixture(
            bottom_z_m=bottom_z_m,
            half_y_m=half_y_m,
            center_y_m=center_y_m,
        )
        validate_pendant_geometry(fixture["center_m"], fixture["half_m"])
    except ValueError as error:
        fixture_ok = False
        fixture_error = str(error)
        fixture = None

    policy, TCPMoveSegment, Policy = _synthetic_policy(ap_w)
    center = np.array(
        [0.72, float(center_y_m), (1.515 + float(bottom_z_m)) / 2.0],
        dtype=float,
    )
    half = np.array(
        [0.10, float(half_y_m), (1.515 - float(bottom_z_m)) / 2.0],
        dtype=float,
    )
    expected_neg = a_face - gap - env_neg
    expected_pos = b_face + gap + env_pos
    samples: list[dict[str, Any]] = []
    waypoint_matches = True
    unclipped = True
    wrist_clear = True
    for side, expected, envelope, lag, face, gap_limit in (
        (
            -1.0,
            expected_neg,
            env_neg,
            lag_neg,
            a_face,
            a_face - gap,
        ),
        (
            1.0,
            expected_pos,
            env_pos,
            lag_pos,
            b_face,
            b_face + gap,
        ),
    ):
        for cross_y in CROSS_Y_SAMPLES_M:
            policy._pact_place_bow_diagnostics = Policy._empty_bow_diagnostics()
            pieces, bowed = policy._bow_segment(
                _segment(TCPMoveSegment, start_y=cross_y, end_y=cross_y),
                prefix="inbound_ceiling_fixture",
                envelope_half_y=Policy._ceiling_fixture_envelope_half_y(side),
                safe_gap=gap,
                center=center,
                half=half,
                preferred_waypoint_side=side,
            )
            waypoint_y = float(pieces[0].end_pose[1, 3])
            diag = policy._pact_place_bow_diagnostics["inbound_ceiling_fixture"]
            planned = float(diag.get("planned_bow_m") or 0.0)
            accepted = float(diag.get("accepted_bow_m") or 0.0)
            clipped = accepted + CLIP_EPS_M < planned
            match = abs(waypoint_y - expected) <= 1e-9
            if side < 0.0:
                wrist_y = waypoint_y + lag
                clear = wrist_y <= gap_limit + CLIP_EPS_M
            else:
                wrist_y = waypoint_y - lag
                clear = wrist_y >= gap_limit - CLIP_EPS_M
            waypoint_matches = waypoint_matches and match and bowed
            unclipped = unclipped and (not clipped)
            wrist_clear = wrist_clear and clear
            samples.append(
                {
                    "waypoint_side": side,
                    "cross_y_m": float(cross_y),
                    "envelope_half_y_m": float(envelope),
                    "expected_waypoint_y_m": float(expected),
                    "bow_segment_waypoint_y_m": waypoint_y,
                    "waypoint_matches_algebra": match,
                    "bowed": bool(bowed),
                    "planned_bow_m": planned,
                    "accepted_bow_m": accepted,
                    "clipped": clipped,
                    "predicted_wrist_y_m": float(wrist_y),
                    "safe_gap_limit_y_m": float(gap_limit),
                    "wrist_satisfies_safe_gap": clear,
                }
            )
    passed = bool(
        inside_window
        and fixture_ok
        and waypoint_matches
        and unclipped
        and wrist_clear
    )
    return {
        "name": name,
        "center_y_m": float(center_y_m),
        "half_y_m": float(half_y_m),
        "bottom_z_m": float(bottom_z_m),
        "neg_face_y_m": float(a_face),
        "pos_face_y_m": float(b_face),
        "admissible_neg_face_y_m": float(a_min),
        "admissible_pos_face_y_m": float(b_max),
        "inside_window": inside_window,
        "contract_validates": fixture_ok,
        "contract_error": fixture_error,
        "waypoint_matches_algebra_every_cross_y": waypoint_matches,
        "unclipped": unclipped,
        "wrist_satisfies_safe_gap": wrist_clear,
        "passed": passed,
        "samples": samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "diagnostics_output/pact_place_v98_bow_clearance_predict.json",
    )
    args = parser.parse_args()
    constants = ceiling_fixture_bow_policy_constants()
    a_min, b_max = pendant_admissible_faces_m(constants)
    predictions = [
        predict_candidate(name=name, **spec) for name, spec in OFFSET_CANDIDATES.items()
    ]
    document = {
        "schema_version": "pact_place_v9_8_bow_clearance_predict_v1",
        "role": "preflight_gate_not_an_episode",
        "authorizes_collection": False,
        "authorizes_gate": False,
        "contract_version": CONTRACT_VERSION,
        "wrist_lag_provenance": WRIST_LAG_PROVENANCE,
        "live_constants": constants,
        "admissible_window_y_m": [a_min, b_max],
        "predictions": predictions,
        "all_passed": all(item["passed"] for item in predictions),
    }
    path = args.output.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: document[k] for k in (
        "all_passed", "admissible_window_y_m", "live_constants"
    )} | {
        "candidates": [
            {
                "name": item["name"],
                "passed": item["passed"],
                "inside_window": item["inside_window"],
                "contract_validates": item["contract_validates"],
                "unclipped": item["unclipped"],
                "wrist_satisfies_safe_gap": item["wrist_satisfies_safe_gap"],
                "faces": [item["neg_face_y_m"], item["pos_face_y_m"]],
            }
            for item in predictions
        ],
        "path": str(path),
    }, indent=2))
    if not document["all_passed"]:
        print("REJECT: a candidate is outside the live window; no episodes.", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
