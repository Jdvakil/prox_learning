#!/usr/bin/env python3
"""V0a: validate the hardened geom-distance instrument before anything uses it.

Four checks:

  A. **GJK agrees with MuJoCo where MuJoCo works.** Over a v6c replay, every pair
     for which mj_geomDistance returns a valid positive scalar is recomputed with
     GJK. If the two disagree, the fallback cannot be trusted on the pairs MuJoCo
     fails, so this runs first and gates the rest.
  B. **fr3_link7_collision defect** -- the geom that returns exactly 0.0 at 0.287 m
     in diagnostics_output/hybrid_obstacle_oracle_reference/geom_distance_defect.json.
     The hardened instrument must return positive.
  C. **The C0 cup case** -- a BOX collider ~0.25 m clear of the bar reported as
     touching. The hardened instrument must return that distance, and it is
     bracketed independently by a surface-sample upper bound and an AABB lower
     bound so GJK is not marking its own homework.
  D. **Stale-buffer assertion** -- a far pair queried immediately after a touching
     pair must return the far distance, not the touching pair's leftover segment.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MOLMO = ROOT / "submodules" / "molmospaces"
for _path in (ROOT / "scripts", MOLMO):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

OUTPUT = ROOT / "diagnostics_output/pact_place_v9_v0a/instrument_validation.json"
AGREEMENT_TOL_M = 1e-4


def _box_surface_samples(model, data, gid: int, n: int = 26) -> np.ndarray:
    """Dense points on an oriented box's surface: an upper bound on distance."""
    pos = np.asarray(data.geom_xpos[gid], dtype=np.float64)
    mat = np.asarray(data.geom_xmat[gid], dtype=np.float64).reshape(3, 3)
    size = np.asarray(model.geom_size[gid], dtype=np.float64)
    grid = np.linspace(-1.0, 1.0, n)
    pts = []
    for axis in range(3):
        for sign in (-1.0, 1.0):
            u, v = np.meshgrid(grid, grid, indexing="ij")
            local = np.zeros((u.size, 3))
            other = [i for i in range(3) if i != axis]
            local[:, axis] = sign * size[axis]
            local[:, other[0]] = u.ravel() * size[other[0]]
            local[:, other[1]] = v.ravel() * size[other[1]]
            pts.append(local)
    local = np.concatenate(pts, axis=0)
    return local @ mat.T + pos


def main() -> int:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("MLSPACES_ASSETS_DIR", str(ROOT / "assets"))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.pop("DISPLAY", None)

    import mujoco
    from molmo_spaces.data_generation.pipeline import cleanup_episode_resources
    from pact_place_corridor_contract import sha256_payload
    from run_pact_place_expert_screen import write_json_atomic
    from run_pact_place_v6c_replay_videos import (
        CONFIG_PATH, _prepare_task, apply_recorded_qpos, row_directory,
    )
    from run_pact_place_swept_volume_v7 import geom_groups
    from run_pact_place_v8_baseline import _physical_geoms, _target_geoms
    import pact_geom_distance as inst

    config = json.loads(CONFIG_PATH.read_text())
    rows = {int(r["role_index"]): r for r in config["expert_screen_rows"]}
    row = rows[8]
    directory = row_directory(8, row["episode_id"])
    result = json.loads((directory / "result.json").read_text())
    steps = list(json.loads((directory / "trajectory.json").read_text())["steps"])

    report: dict[str, Any] = {
        "schema_version": "pact_place_v9_v0a_instrument_validation_v1",
        "role": "v0a_instrument_validation",
        "authorizes_gate": False,
        "mujoco_version": mujoco.__version__,
    }
    task = sampler = scratch = None
    try:
        task, sampler, scratch = _prepare_task(row, result["selected_seed"])
        model, data = task.env.current_model, task.env.current_data
        groups = geom_groups(model)
        spare = ("pact_intrusion_right"
                 if str(result["intrusion_side"]) == "left" else "pact_intrusion_left")
        spare_body = model.body(spare)
        spare_gid = next(g for g in range(int(model.ngeom))
                         if int(model.geom_bodyid[g]) == spare_body.id)
        cup_gids = _physical_geoms(model, _target_geoms(model))
        robot_gids = [
            g for key in ("link5", "link6", "link7", "hand", "left_finger", "right_finger")
            for g in _physical_geoms(model, list(groups[key]))
        ]
        link7_gid = _physical_geoms(model, list(groups["link7"]))[0]

        # ---------- A. GJK vs MuJoCo where MuJoCo works ----------
        deltas = []
        n_compared = 0
        by_type: dict[str, list[float]] = {}
        for frame in range(0, len(steps), 12):
            apply_recorded_qpos(task.env, steps[frame]["qpos"])
            for probe in ([0.69, 0.0, 1.20], [0.66, -0.10, 0.95], [0.72, 0.12, 0.86]):
                data.geom_xpos[spare_gid] = probe
                data.geom_xmat[spare_gid] = np.eye(3).ravel()
                for gid in robot_gids + cup_gids:
                    segment = np.zeros(6)
                    raw = float(mujoco.mj_geomDistance(
                        model, data, int(gid), int(spare_gid), 10.0, segment))
                    if raw <= 0.0 or raw >= 10.0:
                        continue
                    gjk = inst.exact_pair_distance(model, data, int(gid), int(spare_gid))
                    if gjk is None:
                        continue
                    n_compared += 1
                    delta = abs(gjk - raw)
                    deltas.append(delta)
                    key = f"{int(model.geom_type[gid])}_vs_box"
                    by_type.setdefault(key, []).append(delta)
        deltas_arr = np.asarray(deltas)
        report["A_gjk_agreement"] = {
            "n_pairs_compared": n_compared,
            "max_abs_difference_m": float(deltas_arr.max()) if deltas_arr.size else None,
            "median_abs_difference_m": float(np.median(deltas_arr)) if deltas_arr.size else None,
            "tolerance_m": AGREEMENT_TOL_M,
            "by_geom_type_max_m": {
                k: float(np.max(v)) for k, v in sorted(by_type.items())
            },
            "pass": bool(deltas_arr.size and deltas_arr.max() < AGREEMENT_TOL_M),
        }

        # ---------- B. fr3_link7_collision defect ----------
        apply_recorded_qpos(task.env, steps[0]["qpos"])
        link7_pos = np.asarray(data.geom_xpos[link7_gid], dtype=np.float64)
        protr_body = model.body("protr_s")
        protr_gid = next(g for g in range(int(model.ngeom))
                         if int(model.geom_bodyid[g]) == protr_body.id)
        found = None
        n_probes = 0
        rng = np.random.default_rng(20260822)
        frames_b = list(range(0, len(steps), 37))
        for frame in frames_b:
            apply_recorded_qpos(task.env, steps[frame]["qpos"])
            link7_pos = np.asarray(data.geom_xpos[link7_gid], dtype=np.float64)
            for _ in range(3000):
                n_probes += 1
                direction = rng.normal(size=3)
                direction /= np.linalg.norm(direction)
                probe = link7_pos + direction * rng.uniform(0.15, 0.50)
                quat = rng.normal(size=4)
                quat /= np.linalg.norm(quat)
                rot = np.zeros(9)
                mujoco.mju_quat2Mat(rot, quat)
                data.geom_xpos[protr_gid] = probe
                data.geom_xmat[protr_gid] = rot
                segment = np.zeros(6)
                raw = float(mujoco.mj_geomDistance(
                    model, data, int(link7_gid), int(protr_gid), 10.0, segment))
                if not (raw == 0.0 and np.linalg.norm(segment[3:] - segment[:3]) <= 1e-9):
                    continue
                hardened = inst.pair_distance(model, data, int(link7_gid), int(protr_gid))
                found = {
                    "frame": frame,
                    "probe_position_m": probe.tolist(),
                    "probe_xmat": rot.tolist(),
                    "centre_separation_m": float(np.linalg.norm(probe - link7_pos)),
                    "raw_mj_geomDistance_m": raw,
                    "raw_fromto_span_m": float(np.linalg.norm(segment[3:] - segment[:3])),
                    "hardened_m": hardened,
                    "aabb_lower_bound_m": inst.aabb_gap(
                        model, data, int(link7_gid), int(protr_gid)),
                }
                break
            if found is not None:
                break
        report["B_link7_defect"] = {
            "geom": model.geom(link7_gid).name,
            "reference_artifact": (
                "diagnostics_output/hybrid_obstacle_oracle_reference/geom_distance_defect.json"
            ),
            "reproduced": found is not None,
            "n_probes": n_probes,
            "case": found,
            "pass": bool(found is not None and found["hardened_m"] > 0.0
                         and found["hardened_m"] >= found["aabb_lower_bound_m"] - 1e-9),
            "note": (
                "searched random protr_s poses and orientations 0.15-0.50 m from "
                "the link7 collision geom, across replay frames, for a raw scalar "
                "zero with an unwritten fromto"
            ),
        }

        # ---------- C. the C0 cup case, independently bracketed ----------
        apply_recorded_qpos(task.env, steps[239]["qpos"])
        center = np.array([0.6901168024274587, -0.1, 1.1825])
        data.geom_xpos[spare_gid] = center
        data.geom_xmat[spare_gid] = np.eye(3).ravel()
        cup_case = None
        for gid in cup_gids:
            segment = np.zeros(6)
            raw = float(mujoco.mj_geomDistance(
                model, data, int(gid), int(spare_gid), 10.0, segment))
            if raw != 0.0 or np.linalg.norm(segment[3:] - segment[:3]) > 1e-9:
                continue
            hardened = inst.pair_distance(model, data, int(gid), int(spare_gid))
            sample_a = _box_surface_samples(model, data, int(gid))
            sample_b = _box_surface_samples(model, data, int(spare_gid))
            upper = float(np.sqrt(
                ((sample_a[:, None, :] - sample_b[None, :, :]) ** 2).sum(-1).min()
            ))
            lower = inst.aabb_gap(model, data, int(gid), int(spare_gid))
            cup_case = {
                "geom": model.geom(gid).name,
                "raw_mj_geomDistance_m": raw,
                "hardened_m": hardened,
                "independent_upper_bound_m": upper,
                "aabb_lower_bound_m": lower,
                "within_bracket": bool(lower - 1e-6 <= hardened <= upper + 1e-6),
            }
            break
        report["C_c0_cup_case"] = {
            "row": 8, "frame": 239, "bar_center_m": center.tolist(),
            "reproduced": cup_case is not None,
            "case": cup_case,
            "pass": bool(cup_case is not None and cup_case["within_bracket"]
                         and cup_case["hardened_m"] > 0.1),
        }

        # ---------- D. stale-buffer assertion ----------
        # A far pair that MuJoCo answers correctly cannot demonstrate this defect,
        # so use the pair B found: raw scalar 0.0 with fromto untouched. Query a
        # touching pair first so the shared buffer holds a real segment, then the
        # defective far pair. A caller reusing the buffer reads the touching pair's
        # span as this pair's distance.
        stale = None
        if found is not None:
            apply_recorded_qpos(task.env, steps[int(found["frame"])]["qpos"])
            near = np.asarray(data.geom_xpos[link7_gid], dtype=np.float64)
            far = np.asarray(found["probe_position_m"], dtype=np.float64)
            far_mat = np.asarray(found["probe_xmat"], dtype=np.float64)

            shared = np.zeros(6, dtype=np.float64)   # reused, never cleared
            data.geom_xpos[spare_gid] = near + np.array([0.0, 0.0, 0.02])
            data.geom_xmat[spare_gid] = np.eye(3).ravel()
            touch = float(mujoco.mj_geomDistance(
                model, data, int(link7_gid), int(spare_gid), 10.0, shared))
            touch_span = float(np.linalg.norm(shared[3:] - shared[:3]))

            data.geom_xpos[protr_gid] = far
            data.geom_xmat[protr_gid] = far_mat
            raw_far = float(mujoco.mj_geomDistance(
                model, data, int(link7_gid), int(protr_gid), 10.0, shared))
            naive_span = float(np.linalg.norm(shared[3:] - shared[:3]))
            naive_reported = raw_far if raw_far != 0.0 else naive_span
            hardened_far = inst.pair_distance(model, data, int(link7_gid), int(protr_gid))
            gjk_far = inst.exact_pair_distance(model, data, int(link7_gid), int(protr_gid))
            stale = {
                "priming_touching_query": {
                    "scalar_m": touch, "fromto_span_m": touch_span,
                },
                "defective_far_pair_raw_scalar_m": raw_far,
                "naive_reused_buffer_reports_m": naive_reported,
                "hardened_reports_m": hardened_far,
                "gjk_reference_m": gjk_far,
                "naive_is_wrong": bool(
                    gjk_far is not None
                    and abs(naive_reported - gjk_far) > AGREEMENT_TOL_M
                ),
                "hardened_is_right": bool(
                    gjk_far is not None
                    and abs(hardened_far - gjk_far) < AGREEMENT_TOL_M
                ),
            }
        report["D_stale_buffer"] = {
            "constructed_from_case_B": found is not None,
            "case": stale,
            "pass": bool(stale is not None and stale["hardened_is_right"]),
            "note": (
                "naive_is_wrong records whether the uncleaned buffer actually "
                "misreported this pair; the assertion that matters is that the "
                "hardened path returns the true distance either way"
            ),
        }
        report["counters"] = inst.counters()
        report["all_pass"] = bool(
            report["A_gjk_agreement"]["pass"]
            and report["B_link7_defect"]["pass"]
            and report["C_c0_cup_case"]["pass"]
            and report["D_stale_buffer"]["pass"]
        )
    finally:
        cleanup_episode_resources(
            task=task, policy=None, task_sampler=sampler,
            preloaded_policy=None, close_task_sampler=sampler is not None,
        )
        if scratch is not None:
            shutil.rmtree(scratch, ignore_errors=True)

    report["analysis_sha256"] = sha256_payload(report)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(OUTPUT, report)
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("analysis_sha256",)}, indent=2, default=str))
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
