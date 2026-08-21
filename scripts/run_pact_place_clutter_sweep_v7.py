#!/usr/bin/env python3
"""A0e: derive the v7 clutter lattice from the A0d swept-volume measurement.

No new rollouts. Do not overwrite the v6/v6b/v6c sweeps. Coordinates come from
the measured free voxels, not from hand-picked corners.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MOLMO = ROOT / "submodules" / "molmospaces"
for search_path in (ROOT / "scripts", MOLMO):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from pact_place_corridor_contract import sha256_payload  # noqa: E402
from run_pact_place_swept_volume_v7 import (  # noqa: E402
    INTERIOR_X,
    INTERIOR_Y,
    INTERIOR_Z,
    LINK_KEYS,
    OUTPUT_DIR as A0D_DIR,
    VOXEL_M,
    aabb_distance,
    closest_on_aabb,
    point_in_camera_fov,
)

OUTPUT = ROOT / "diagnostics_output/pact_place_clutter_sweep_v7/analysis.json"
C_M = 0.030
JITTER_M = 0.020
Y_LIMIT_M = 0.425
CEILING_TOP_M = 0.72 + 0.70 - 0.02  # z0 + ap_h - 0.02
SHELF_TOP_Z = 0.72
TUBE_X0 = 0.58
BACK_WALL_OFFSET_M = 0.02
PANEL_X_SPAN = (0.556, 0.666)
PANEL_LEFT = {
    "min_m": np.array([0.615 - 0.055, 0.340 - 0.240, 0.89 - 0.090], dtype=float),
    "max_m": np.array([0.615 + 0.055, 0.340 + 0.240, 0.89 + 0.090], dtype=float),
}
PANEL_RIGHT = {
    "min_m": np.array([0.615 - 0.055, -0.340 - 0.240, 0.89 - 0.090], dtype=float),
    "max_m": np.array([0.615 + 0.055, -0.340 + 0.240, 0.89 + 0.090], dtype=float),
}
TARGET_LO = np.array([0.745, -0.09, 0.72], dtype=float)
TARGET_HI = np.array([0.775, 0.09, 0.80], dtype=float)
TARGET_CLEARANCE_M = 0.04
TRAY_LO = np.array([0.25, 0.22, 0.00], dtype=float)
TRAY_HI = np.array([0.45, 0.42, 0.10], dtype=float)
KEEP_OUT_LINKS = (
    "link5",
    "link6",
    "link7",
    "hand",
    "left_finger",
    "right_finger",
    "cup",
)
PASSAGE_TOKENS = ("link4_sensor", "link5_front", "link5_back", "link6_sensor")
BASE_TOKENS = ("link1_sensor", "link2_sensor")
X_BINS = ((0.60, 0.66), (0.66, 0.72), (0.72, 0.78))
Z_BINS = ((0.72, 0.88), (0.88, 1.04), (1.04, 1.40))
TARGET_N = 12
MIN_N = 10
MAX_N = 14


def unpack_voxel(code: int) -> tuple[int, int, int]:
    return code & 0x3FF, (code >> 10) & 0x3FF, (code >> 20) & 0x3FF


def load_occupancy_xyz() -> np.ndarray:
    payload = json.loads((A0D_DIR / "occupancy_voxels_per_geom.json").read_text())
    used: set[int] = set()
    for key in KEEP_OUT_LINKS:
        used.update(payload[key])
    origin = np.asarray([INTERIOR_X[0], INTERIOR_Y[0], INTERIOR_Z[0]], dtype=float)
    points = []
    for code in used:
        ix, iy, iz = unpack_voxel(int(code))
        points.append(origin + (np.array([ix, iy, iz], dtype=float) + 0.5) * VOXEL_M)
    return np.asarray(points, dtype=float)


def min_occupancy_distance(lo: np.ndarray, hi: np.ndarray, occ_xyz: np.ndarray) -> float:
    closest = np.clip(occ_xyz, lo, hi)
    return float(np.linalg.norm(occ_xyz - closest, axis=1).min())


def expand_xy(lo: np.ndarray, hi: np.ndarray, extra: float) -> tuple[np.ndarray, np.ndarray]:
    lo = lo.copy()
    hi = hi.copy()
    lo[0] -= extra
    lo[1] -= extra
    hi[0] += extra
    hi[1] += extra
    return lo, hi


def aabb_overlap(a_lo, a_hi, b_lo, b_hi, extra: float = 0.0) -> bool:
    return aabb_distance(a_lo - extra, a_hi + extra, b_lo, b_hi) == 0.0


def x_bin_of(x: float) -> int | None:
    for i, (lo, hi) in enumerate(X_BINS):
        if lo <= x < hi or (i == len(X_BINS) - 1 and lo <= x <= hi):
            return i
    return None


def z_bin_of(z: float) -> int | None:
    for i, (lo, hi) in enumerate(Z_BINS):
        if lo <= z < hi or (i == len(Z_BINS) - 1 and lo <= z <= hi):
            return i
    return None


def generate_candidates(shallowest_back_wall: float) -> list[dict[str, Any]]:
    outer_x_limit = min(shallowest_back_wall, TUBE_X0 + 0.18 + BACK_WALL_OFFSET_M) - JITTER_M
    y_outer_limit = Y_LIMIT_M - JITTER_M
    sizes = (
        ("floor_small", 0.025, 0.040, 0.050, "floor"),
        ("floor_narrow", 0.025, 0.025, 0.050, "floor"),
        ("wall_block", 0.025, 0.020, 0.040, "wall"),
        ("overhead_thin", 0.040, 0.030, 0.015, "ceiling"),
        ("overhead_wide", 0.030, 0.040, 0.015, "ceiling"),
    )
    xs = [0.62, 0.65, 0.68, 0.70, 0.72]
    candidates = []
    cid = 0
    for size_name, hx, hy, hz, support in sizes:
        if support == "floor":
            zc = SHELF_TOP_Z + hz
            y_abs_values = [y_outer_limit - hy, 0.36, 0.33]
        elif support == "ceiling":
            zc = CEILING_TOP_M - hz
            y_abs_values = [y_outer_limit - hy, 0.36, 0.32]
        else:
            zc_values = [1.00, 1.12, 1.24]
            y_abs_values = [y_outer_limit - hy]
        z_list = [zc] if support != "wall" else zc_values
        for x in xs:
            for zc_i in z_list:
                for y_abs in y_abs_values:
                    for sign, side in ((1.0, "left"), (-1.0, "right")):
                        y = sign * y_abs
                        lo = np.array([x - hx, y - hy, zc_i - hz], dtype=float)
                        hi = np.array([x + hx, y + hy, zc_i + hz], dtype=float)
                        candidates.append(
                            {
                                "candidate_id": cid,
                                "size_name": size_name,
                                "support": support,
                                "side": side,
                                "center_m": [float(x), float(y), float(zc_i)],
                                "half_m": [float(hx), float(hy), float(hz)],
                                "min_m": lo.tolist(),
                                "max_m": hi.tolist(),
                                "outer_x_limit_m": float(outer_x_limit),
                                "y_outer_limit_m": float(y_outer_limit),
                            }
                        )
                        cid += 1
    return candidates


def reject_reason(
    cand: dict[str, Any],
    occ_xyz: np.ndarray,
    shallowest_back_wall: float,
) -> tuple[str | None, dict[str, Any]]:
    lo = np.asarray(cand["min_m"], dtype=float)
    hi = np.asarray(cand["max_m"], dtype=float)
    hx, hy, hz = cand["half_m"]
    x, y, zc = cand["center_m"]
    jittered_lo, jittered_hi = expand_xy(lo, hi, JITTER_M)
    geom: dict[str, Any] = {
        "outer_face_x_m": float(hi[0]),
        "outer_face_abs_y_m": float(abs(y) + hy),
        "top_z_m": float(hi[2]),
        "bottom_z_m": float(lo[2]),
        "inner_face_abs_y_m": float(abs(y) - hy),
    }
    if abs(y) + hy - 1e-12 > Y_LIMIT_M:
        return "outside_enclosure_y", geom
    if hi[2] - 1e-12 > CEILING_TOP_M:
        return "above_ceiling_margin", geom
    if lo[2] + 1e-12 < SHELF_TOP_Z and cand["support"] == "floor":
        return "below_shelf", geom
    if hi[0] - 1e-12 > cand["outer_x_limit_m"]:
        return "outer_x_beyond_jittered_shallowest_wall", geom
    if hi[0] - 1e-12 > shallowest_back_wall:
        return "outer_x_beyond_shallowest_episode_wall", geom
    if cand["support"] == "floor" and abs(lo[2] - SHELF_TOP_Z) > 1e-9:
        return "floor_box_not_on_shelf", geom
    if cand["support"] == "ceiling" and abs(hi[2] - CEILING_TOP_M) > 1e-9:
        return "overhead_not_on_ceiling", geom
    if cand["support"] == "wall" and abs(abs(y) + hy - cand["y_outer_limit_m"]) > 1e-6:
        return "wall_box_not_on_side_wall", geom
    panel_x_hit = (lo[0] < PANEL_X_SPAN[1]) and (hi[0] > PANEL_X_SPAN[0])
    panel_z_hit = (lo[2] < 0.98) and (hi[2] > 0.80)
    if panel_x_hit and panel_z_hit:
        return "overlaps_panel_x_span", geom
    if aabb_overlap(lo, hi, TRAY_LO, TRAY_HI, extra=0.01):
        return "overlaps_tray", geom
    target_gap = aabb_distance(lo, hi, TARGET_LO, TARGET_HI)
    geom["target_envelope_gap_m"] = target_gap
    if target_gap + 1e-12 < TARGET_CLEARANCE_M:
        return "too_close_to_target_envelope", geom
    # Overhead above the rest pose would hide the cup from the wrist.
    if cand["support"] == "ceiling" and abs(y) < 0.18 and x >= 0.70:
        return "overhead_would_occlude_target_at_rest", geom
    occ_dist = min_occupancy_distance(jittered_lo, jittered_hi, occ_xyz)
    geom["occupancy_clearance_m"] = occ_dist
    if occ_dist + 1e-12 < C_M:
        return "inside_swept_volume_plus_C", geom
    x_bin = x_bin_of(x)
    z_bin = z_bin_of(zc)
    if x_bin is None:
        return "outside_x_dispersion_range", geom
    if z_bin is None:
        return "outside_z_dispersion_range", geom
    geom["x_bin"] = x_bin
    geom["z_bin"] = z_bin
    geom["y_sign"] = 1 if y > 0 else -1
    return None, geom


def load_tracks() -> dict[str, Any]:
    cam_pos = []
    cam_fwd = []
    wrist_pos = []
    wrist_fwd = []
    names = None
    for path in sorted((A0D_DIR / "tracks").glob("row*.npz")):
        data = np.load(path)
        cam_pos.append(data["cam_pos"])
        cam_fwd.append(data["cam_fwd"])
        wrist_pos.append(data["wrist_pos"])
        wrist_fwd.append(data["wrist_fwd"])
        if names is None:
            names = [str(name) for name in data["camera_names"]]
    cam_pos = np.concatenate(cam_pos, axis=0)
    cam_fwd = np.concatenate(cam_fwd, axis=0)
    return {
        "cam_pos": cam_pos,
        "cam_fwd": cam_fwd,
        "wrist_pos": np.concatenate(wrist_pos, axis=0),
        "wrist_fwd": np.concatenate(wrist_fwd, axis=0),
        "camera_names": names,
        "is_passage": np.array(
            [any(token in name for token in PASSAGE_TOKENS) for name in names],
            dtype=bool,
        ),
        "is_base": np.array(
            [any(token in name for token in BASE_TOKENS) for name in names],
            dtype=bool,
        ),
    }


def _in_fov_mask(
    points: np.ndarray,
    origins: np.ndarray,
    forwards: np.ndarray,
    fov_deg: float,
    far_m: float,
) -> np.ndarray:
    delta = points - origins
    dist = np.linalg.norm(delta, axis=-1)
    valid = dist > 1e-9
    cosine = np.zeros_like(dist)
    cosine[valid] = np.sum(
        (delta[valid] / dist[valid, None]) * forwards[valid], axis=-1
    )
    half = np.deg2rad(fov_deg) / 2.0
    return valid & (dist <= far_m) & (cosine >= np.cos(half))


def score_slot(
    cand: dict[str, Any],
    tracks: dict[str, Any],
    fov_deg: float,
    far_m: float,
) -> dict[str, float]:
    lo = np.asarray(cand["min_m"], dtype=float)
    hi = np.asarray(cand["max_m"], dtype=float)
    cam_pos = tracks["cam_pos"]
    closest = np.clip(cam_pos, lo, hi)
    engaged = _in_fov_mask(closest, cam_pos, tracks["cam_fwd"], fov_deg, far_m)
    passage = tracks["is_passage"]
    base = tracks["is_base"]
    wrist_closest = np.clip(tracks["wrist_pos"], lo, hi)
    vis = _in_fov_mask(
        wrist_closest, tracks["wrist_pos"], tracks["wrist_fwd"], 56.74, far_m
    )
    all_pairs = engaged.size
    passage_pairs = int(passage.sum() * cam_pos.shape[0])
    return {
        "skin_engagement": float(engaged.mean()),
        "passage_engagement": float(engaged[:, tracks["is_passage"]].mean())
        if tracks["is_passage"].any()
        else 0.0,
        "base_only_engagement": float(
            (engaged & base[None, :] & ~passage[None, :]).mean()
        ),
        "wrist_fov_visibility": float(vis.mean()),
        "n_passage_engaged": float(engaged[:, tracks["is_passage"]].sum()),
        "n_engaged": float(engaged.sum()),
    }


def lattice_metrics(
    slots: list[dict[str, Any]],
    tracks: dict[str, Any],
    fov_deg: float,
    far_m: float,
) -> dict[str, float]:
    engaged = np.zeros(tracks["cam_pos"].shape[:2], dtype=bool)
    vis = np.zeros(tracks["wrist_pos"].shape[0], dtype=bool)
    for slot in slots:
        lo = np.asarray(slot["min_m"], dtype=float)
        hi = np.asarray(slot["max_m"], dtype=float)
        closest = np.clip(tracks["cam_pos"], lo, hi)
        engaged |= _in_fov_mask(
            closest, tracks["cam_pos"], tracks["cam_fwd"], fov_deg, far_m
        )
        wrist_closest = np.clip(tracks["wrist_pos"], lo, hi)
        vis |= _in_fov_mask(
            wrist_closest, tracks["wrist_pos"], tracks["wrist_fwd"], 56.74, far_m
        )
    return {
        "skin_engagement": float(engaged.mean()),
        "passage_engagement": float(engaged[:, tracks["is_passage"]].mean())
        if tracks["is_passage"].any()
        else 0.0,
        "wrist_fov_visibility": float(vis.mean()),
        "n_boxes": float(len(slots)),
    }


def dispersion(slots: list[dict[str, Any]]) -> dict[str, Any]:
    x_bins = {x_bin_of(s["center_m"][0]) for s in slots}
    z_bins = {z_bin_of(s["center_m"][2]) for s in slots}
    high_z = any(s["center_m"][2] > 0.95 for s in slots)
    y_by_x: dict[int, set[int]] = {}
    for slot in slots:
        xb = x_bin_of(slot["center_m"][0])
        y_by_x.setdefault(xb, set()).add(1 if slot["center_m"][1] > 0 else -1)
    both_y = all(y_by_x.get(i, set()) == {1, -1} for i in range(len(X_BINS)))
    return {
        "n_x_bins": len([b for b in x_bins if b is not None]),
        "n_z_bins": len([b for b in z_bins if b is not None]),
        "has_tier_above_z_0_95": bool(high_z),
        "both_y_signs_in_every_x_bin": bool(both_y),
        "y_signs_by_x_bin": {str(k): sorted(v) for k, v in y_by_x.items()},
        "ok": bool(
            len([b for b in x_bins if b is not None]) >= 3
            and len([b for b in z_bins if b is not None]) >= 3
            and high_z
            and both_y
        ),
    }


def select_slots(admitted: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(
        admitted,
        key=lambda item: (
            -item["score"]["passage_engagement"],
            item["score"]["wrist_fov_visibility"],
            -item["geom"]["occupancy_clearance_m"],
            item["candidate_id"],
        ),
    )
    chosen: list[dict[str, Any]] = []

    def overlaps_chosen(cand: dict[str, Any]) -> bool:
        lo = np.asarray(cand["min_m"], dtype=float)
        hi = np.asarray(cand["max_m"], dtype=float)
        for other in chosen:
            if aabb_overlap(
                lo,
                hi,
                np.asarray(other["min_m"], dtype=float),
                np.asarray(other["max_m"], dtype=float),
                extra=0.01,
            ):
                return True
        return False

    for cand in ranked:
        if cand["score"]["n_passage_engaged"] <= 0:
            continue
        if cand["score"]["passage_engagement"] <= 0 and cand["score"]["base_only_engagement"] > 0:
            continue
        if overlaps_chosen(cand):
            continue
        chosen.append(cand)
        if len(chosen) >= TARGET_N and dispersion(chosen)["ok"]:
            break
        if len(chosen) >= MAX_N:
            break

    def missing_constraints(current: list[dict[str, Any]]) -> list[str]:
        report = dispersion(current)
        missing = []
        if report["n_x_bins"] < 3:
            missing.append("x_bins")
        if report["n_z_bins"] < 3:
            missing.append("z_bins")
        if not report["has_tier_above_z_0_95"]:
            missing.append("high_z")
        if not report["both_y_signs_in_every_x_bin"]:
            missing.append("y_signs")
        return missing

    for cand in ranked:
        if len(chosen) >= MAX_N:
            break
        if cand in chosen or overlaps_chosen(cand):
            continue
        before = missing_constraints(chosen)
        if not before:
            break
        probe = chosen + [cand]
        after = missing_constraints(probe)
        if len(after) < len(before):
            chosen.append(cand)
    if len(chosen) > MAX_N:
        chosen = chosen[:MAX_N]
    return chosen


def main() -> int:
    a0d = json.loads((A0D_DIR / "analysis.json").read_text())
    shallowest = float(a0d["shallowest_episode_back_wall_x_m"])
    fov_deg = float(a0d["sensor_config"]["fov_deg"])
    far_m = float(a0d["sensor_config"]["clip_far_m"])
    occ_xyz = load_occupancy_xyz()
    tracks = load_tracks()
    records = []
    for cand in generate_candidates(shallowest):
        reason, geom = reject_reason(cand, occ_xyz, shallowest)
        record = {**cand, "geom": geom, "reject_reason": reason, "admitted": reason is None}
        if record["admitted"]:
            record["score"] = score_slot(cand, tracks, fov_deg, far_m)
            if record["score"]["n_passage_engaged"] <= 0:
                record["admitted"] = False
                record["reject_reason"] = "no_link4_5_6_engagement"
        records.append(record)
        print(
            f"cand={cand['candidate_id']:03d} {cand['support']:7s} "
            f"{cand['center_m']} {record['reject_reason'] or 'ADMITTED'}",
            flush=True,
        )
    admitted = [item for item in records if item["admitted"]]
    chosen = select_slots(admitted)
    for index, slot in enumerate(chosen):
        slot["body"] = f"pact_clutter_{index:02d}"
        slot["slot_name"] = f"{index:02d}"
    predicted = lattice_metrics(chosen, tracks, fov_deg, far_m) if chosen else {}
    disp = dispersion(chosen)
    analysis = {
        "schema_version": "pact_place_clutter_sweep_v7",
        "no_rollouts": True,
        "a0d_analysis_sha256": a0d["analysis_sha256"],
        "clearance_C_m": C_M,
        "jitter_m": JITTER_M,
        "keep_out": "per_geom_occupancy_voxels_of_link5_6_7_hand_fingers_cup",
        "shallowest_episode_back_wall_x_m": shallowest,
        "n_candidates": len(records),
        "n_admitted": len(admitted),
        "n_rejected": len(records) - len(admitted),
        "reject_counts": {},
        "chosen_n": len(chosen),
        "chosen_slots": [
            {
                "slot_name": slot["slot_name"],
                "body": slot["body"],
                "support": slot["support"],
                "size_name": slot["size_name"],
                "center_m": slot["center_m"],
                "half_m": slot["half_m"],
                "score": slot["score"],
                "occupancy_clearance_m": slot["geom"]["occupancy_clearance_m"],
            }
            for slot in chosen
        ],
        "dispersion": disp,
        "predicted_metrics_on_v6c_tracks": predicted,
        "v6c_baseline_official": a0d["v6c_baseline"],
        "selection_rule": (
            "rank by passage skin engagement, then lower wrist FOV visibility, "
            "then higher occupancy clearance; enforce 10-14 boxes and dispersion"
        ),
        "candidates": [
            {
                k: v
                for k, v in item.items()
                if k != "score" or item["admitted"]
            }
            for item in records
        ],
    }
    counts: dict[str, int] = {}
    for item in records:
        key = item["reject_reason"] or "admitted"
        counts[key] = counts.get(key, 0) + 1
    analysis["reject_counts"] = counts
    analysis["analysis_sha256"] = sha256_payload(
        {k: v for k, v in analysis.items() if k != "analysis_sha256"}
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    print(OUTPUT)
    print(
        json.dumps(
            {
                "n_candidates": analysis["n_candidates"],
                "n_admitted": analysis["n_admitted"],
                "reject_counts": counts,
                "chosen_n": analysis["chosen_n"],
                "dispersion": disp,
                "predicted_metrics_on_v6c_tracks": predicted,
                "v6c_skin_engagement": a0d["v6c_baseline"]["skin_engagement"],
                "v6c_wrist_visibility": a0d["v6c_baseline"]["wrist_visibility"],
                "analysis_sha256": analysis["analysis_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
