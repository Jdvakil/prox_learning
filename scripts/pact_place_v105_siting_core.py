#!/usr/bin/env python3
"""V10.5 siting core: snapshot once, score every candidate exactly.

The expensive part of scoring 96 candidate scenes against ~98 retained
trajectories is not the distance maths — it is re-sampling the task and
re-forwarding MuJoCo. So each source row is loaded exactly once, its retained
frames are replayed once, and the world poses of every probe and environment
geom are snapshotted. All 96 candidate assemblies are then scored against that
snapshot analytically, with an AABB screen as broad phase and hardened exact
GJK as the only decision instrument.

Nothing here compiles a pendant into a scene. Candidates are posed boxes with
known analytic geometry, so scoring cannot perturb the source environment; the
selected bundle is compiled and re-verified separately.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "scripts", ROOT / "submodules" / "molmospaces"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from pact_geom_distance import GeomShape, gjk_distance, mesh_vertices  # noqa: E402
from pact_place_v105_geometry import (  # noqa: E402
    CLEARANCE_FLOOR_M,
    POSE_IDS,
    POSE_OFFSETS_M,
    RISK_BAND_M,
    build_assembly,
    lattice_candidates,
)

ROBOT_PREFIX = "robot_0/"
SCREEN_MARGIN_M = 0.12

INBOUND_PHASES = ("pregrasp", "inbound", "grasp", "gripper-close")
GRASP_WINDOW_PHASES = ("grasp", "gripper-close")
LIFT_WINDOW_PHASES = ("lift",)
RELEASE_WINDOW_PHASES = ("placement_descent", "gripper-open", "preplace")


def traversal_direction(policy_phase: str, target_held: bool) -> str:
    phase = str(policy_phase or "")
    if phase.startswith("outbound") or phase in {
        "preplace", "placement_descent", "gripper-open", "retreat", "lift"
    }:
        return "loaded_outbound" if target_held else "outbound"
    return "inbound"


def window_of(policy_phase: str) -> str | None:
    phase = str(policy_phase or "")
    if phase in GRASP_WINDOW_PHASES:
        return "grasp_close"
    if phase in LIFT_WINDOW_PHASES:
        return "lift"
    if phase in RELEASE_WINDOW_PHASES:
        return "release"
    return None


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------
def snapshot_row(
    row_dir: Path,
    family_id: str,
    intrusion_side: str,
    seed_u32: int,
    *,
    base_scene: Path,
) -> dict[str, Any]:
    """Replay one retained trajectory and capture every geom pose per frame.

    Returns posed-shape data, not distances: the candidate-independent half of
    the computation, done once.
    """
    import mujoco
    from molmo_spaces.data_generation.pipeline import cleanup_episode_resources
    from pact_place_v105_contract import v95_row_payload
    from run_pact_place_expert_screen import _make_config
    from run_pact_place_v7_replay_videos import apply_recorded_qpos

    steps = json.loads((row_dir / "trajectory.json").read_text())["steps"]
    payload = v95_row_payload(family_id, intrusion_side)
    row = {
        "role_index": 0,
        "episode_id": row_dir.name,
        "intrusion_side": intrusion_side,
        "task_seed_u32": int(seed_u32),
        "task_seed_u64": int(seed_u32),
        "sampler_class": "PactPlaceCorridorV93Sampler",
        **payload,
    }
    scratch = Path(tempfile.mkdtemp(prefix="v105_snap_"))
    task = sampler = None
    try:
        config = _make_config(
            scratch / "d.json",
            scene_xml=base_scene,
            sampler_class="PactPlaceCorridorV93Sampler",
        )
        sampler = config.task_sampler_config.task_sampler_class(config)
        sampler.seed_task_sampling(int(seed_u32))
        sampler.set_pact_manifest_row(row)
        task = sampler.sample_task(house_index=int(row["scene_template_house_index"]))
        env = task.env
        model, data = env.current_model, env.current_data

        from pact_place_v105_clearance import (
            robot_collision_geom_ids,
            target_collision_geom_ids,
        )

        robot_ids = robot_collision_geom_ids(model)
        target_ids = target_collision_geom_ids(task)
        probe_ids = list(robot_ids) + list(target_ids)
        env_ids = []
        for geom_id in range(int(model.ngeom)):
            if geom_id in set(probe_ids):
                continue
            body = str(model.body(int(model.geom_bodyid[geom_id])).name or "")
            if body.startswith(ROBOT_PREFIX):
                continue
            if (
                int(model.geom_contype[geom_id]) == 0
                and int(model.geom_conaffinity[geom_id]) == 0
            ):
                continue
            env_ids.append(int(geom_id))

        def static_of(ids):
            return {
                int(g): (
                    int(model.geom_type[g]),
                    np.asarray(model.geom_size[g], dtype=float).copy(),
                    mesh_vertices(model, int(g)),
                    float(model.geom_rbound[g]),
                    str(model.geom(g).name or f"geom_{g}"),
                    str(model.body(int(model.geom_bodyid[g])).name or ""),
                )
                for g in ids
            }

        probe_static = static_of(probe_ids)
        env_static = static_of(env_ids)
        n = len(steps)
        probe_pos = np.zeros((n, len(probe_ids), 3), dtype=np.float64)
        probe_mat = np.zeros((n, len(probe_ids), 9), dtype=np.float64)
        env_pos = np.zeros((n, len(env_ids), 3), dtype=np.float64)
        env_mat = np.zeros((n, len(env_ids), 9), dtype=np.float64)
        phases: list[str] = []
        held: list[bool] = []
        for index, step in enumerate(steps):
            apply_recorded_qpos(env, step["qpos"])
            mujoco.mj_forward(model, data)
            for local, geom_id in enumerate(probe_ids):
                probe_pos[index, local] = data.geom_xpos[geom_id]
                probe_mat[index, local] = np.asarray(
                    data.geom_xmat[geom_id], dtype=float
                ).reshape(9)
            for local, geom_id in enumerate(env_ids):
                env_pos[index, local] = data.geom_xpos[geom_id]
                env_mat[index, local] = np.asarray(
                    data.geom_xmat[geom_id], dtype=float
                ).reshape(9)
            phases.append(str(step.get("policy_phase") or ""))
            held.append(bool(index >= _first_close_index(steps)))
        return {
            "ok": True,
            "n_frames": n,
            "probe_ids": probe_ids,
            "probe_static": probe_static,
            "probe_pos": probe_pos,
            "probe_mat": probe_mat,
            "target_ids": list(target_ids),
            "env_ids": env_ids,
            "env_static": env_static,
            "env_pos": env_pos,
            "env_mat": env_mat,
            "phases": phases,
            "held": held,
        }
    finally:
        cleanup_episode_resources(
            task=task, policy=None, task_sampler=sampler,
            preloaded_policy=None, close_task_sampler=sampler is not None,
        )
        shutil.rmtree(scratch, ignore_errors=True)


def _first_close_index(steps: Sequence[dict[str, Any]]) -> int:
    for index, step in enumerate(steps):
        if str(step.get("policy_phase") or "") == "gripper-close":
            return index
    return len(steps)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def _boxes_for(assembly: dict[str, Any]):
    out = []
    for item in assembly["components"]:
        center = np.asarray(item["center_m"], dtype=float)
        half = np.asarray(item["half_m"], dtype=float)
        out.append(
            {
                "name": item["name"],
                "geom": item["geom"],
                "role": item["role"],
                "side": item.get("side", ""),
                "center": center,
                "half": half,
                "shape": GeomShape.posed_axis_aligned_box(center, half),
            }
        )
    return out


def score_candidate_against_snapshot(
    assembly: dict[str, Any], snap: dict[str, Any], intrusion_side: str
) -> dict[str, Any]:
    """Exact per-frame clearance for one assembly against one snapshot.

    The AABB screen is conservative and is used only to skip frames that are
    provably far; every frame that survives it is measured exactly.
    """
    boxes = _boxes_for(assembly)
    risk_side = "negative" if intrusion_side == "left" else "positive"
    n = snap["n_frames"]
    probe_pos = snap["probe_pos"]
    probe_static = snap["probe_static"]
    probe_ids = snap["probe_ids"]
    target_set = set(snap["target_ids"])
    rbound = np.asarray(
        [probe_static[g][3] for g in probe_ids], dtype=float
    )
    centers_all = np.asarray([box["center"] for box in boxes])
    halves_all = np.asarray([box["half"] for box in boxes])

    best = {"m": float("inf"), "frame": None, "box": None, "probe": None}
    best_risk = {"m": float("inf"), "frame": None, "box": None, "probe": None}
    windows = {"grasp_close": float("inf"), "lift": float("inf"),
               "release": float("inf")}
    initial = {"robot": float("inf"), "target": float("inf")}
    contact = False
    risk_direction: dict[str, float] = {}

    for frame in range(n):
        pos = probe_pos[frame]
        # Conservative AABB screen. `lower` is a proven lower bound on the true
        # distance for every (probe, box) pair, so a pair above the margin can
        # be recorded from the bound instead of measured.
        delta = np.maximum(
            np.abs(pos[:, None, :] - centers_all[None, :, :]) - halves_all[None, :, :],
            0.0,
        )
        lower = np.linalg.norm(delta, axis=2) - rbound[:, None]
        window = window_of(snap["phases"][frame])
        direction = traversal_direction(snap["phases"][frame], snap["held"][frame])
        for bi, box in enumerate(boxes):
            column = lower[:, bi]
            near = np.flatnonzero(column <= SCREEN_MARGIN_M)
            is_risk = box["role"] in ("lobe", "stem") and box["side"] == risk_side
            if near.size == 0:
                # Proven far. The bound itself is a valid conservative value
                # for every floor we need to clear.
                bound = float(np.min(column))
                if bound < best["m"]:
                    best = {"m": bound, "frame": frame, "box": box["name"],
                            "probe": None, "probe_body": None,
                            "role": box["role"], "side": box["side"],
                            "phase": snap["phases"][frame], "exact": False}
                if is_risk:
                    if bound < best_risk["m"]:
                        best_risk = {"m": bound, "frame": frame,
                                     "box": box["name"], "probe": None,
                                     "probe_body": None, "role": box["role"],
                                     "side": box["side"],
                                     "phase": snap["phases"][frame],
                                     "exact": False}
                    if bound < risk_direction.get(direction, float("inf")):
                        risk_direction[direction] = bound
                if window is not None and bound < windows[window]:
                    windows[window] = bound
                if frame == 0:
                    for key in ("robot", "target"):
                        if bound < initial[key]:
                            initial[key] = bound
                continue
            for local in near.tolist():
                geom_id = probe_ids[local]
                gtype, size, verts, _, gname, bname = probe_static[geom_id]
                shape = GeomShape(
                    gtype, pos[local], snap["probe_mat"][frame, local], size, verts
                )
                if not shape.supported:
                    continue
                distance = float(gjk_distance(shape, box["shape"]))
                if distance <= 0.0:
                    contact = True
                if distance < best["m"]:
                    best = {"m": distance, "frame": frame, "box": box["name"],
                            "probe": gname, "probe_body": bname,
                            "role": box["role"], "side": box["side"],
                            "phase": snap["phases"][frame], "exact": True}
                if is_risk:
                    if distance < best_risk["m"]:
                        best_risk = {
                            "m": distance, "frame": frame, "box": box["name"],
                            "probe": gname, "probe_body": bname,
                            "role": box["role"], "side": box["side"],
                            "phase": snap["phases"][frame], "exact": True,
                        }
                    if distance < risk_direction.get(direction, float("inf")):
                        risk_direction[direction] = distance
                if window is not None and distance < windows[window]:
                    windows[window] = distance
                if frame == 0:
                    key = "target" if geom_id in target_set else "robot"
                    if distance < initial[key]:
                        initial[key] = distance
    return {
        "min_clearance_m": None if best["frame"] is None else best["m"],
        "min_witness": None if best["frame"] is None else best,
        "min_lobe_stem_m": None if best_risk["frame"] is None else best_risk["m"],
        "risk_witness": None if best_risk["frame"] is None else best_risk,
        "risk_by_direction_m": risk_direction,
        "window_min_m": {k: (None if v == float("inf") else v)
                         for k, v in windows.items()},
        "initial_min_m": {k: (None if v == float("inf") else v)
                          for k, v in initial.items()},
        "contact": contact,
    }


def probe_mat_row(snap: dict[str, Any], frame: int, local: int) -> np.ndarray:
    return snap["probe_mat"][frame, local]


def environment_candidate_geoms(
    snap: dict[str, Any], assemblies: Sequence[dict[str, Any]]
) -> list[int]:
    """Environment geoms that could ever come near ANY candidate pendant.

    A conservative whole-trajectory screen: a geom whose bounding sphere never
    reaches within SCREEN_MARGIN_M of the union AABB of every candidate cannot
    be the limiting pair for any of them, at any frame. Dropping it is sound,
    and it removes the room shell, bench, and floor-level clutter that make up
    almost all 496 environment geoms.
    """
    if not snap["env_ids"]:
        return []
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for assembly in assemblies:
        for item in assembly["components"]:
            centre = np.asarray(item["center_m"], dtype=float)
            half = np.asarray(item["half_m"], dtype=float)
            lo = np.minimum(lo, centre - half)
            hi = np.maximum(hi, centre + half)
    centre = (lo + hi) / 2.0
    half = (hi - lo) / 2.0
    kept: list[int] = []
    env_pos = snap["env_pos"]
    for local, geom_id in enumerate(snap["env_ids"]):
        rbound = float(snap["env_static"][geom_id][3])
        delta = np.maximum(np.abs(env_pos[:, local, :] - centre) - half, 0.0)
        if float(np.min(np.linalg.norm(delta, axis=1)) - rbound) <= SCREEN_MARGIN_M:
            kept.append(int(geom_id))
    return kept


def environment_clearance(
    assembly: dict[str, Any],
    snap: dict[str, Any],
    *,
    allow_hood_top: bool = True,
    env_ids: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Pendant-to-environment clearance; must stay positive throughout.

    The designed crossbar/hood_top flush face is the single allowlisted
    contact, and it is allowlisted only here -- it never exempts robot or
    carried-target contact.
    """
    boxes = _boxes_for(assembly)
    ids = list(snap["env_ids"] if env_ids is None else env_ids)
    env_static = snap["env_static"]
    if not ids:
        return {"min_m": None, "witness": None, "intersects": False,
                "n_env_geoms_tested": 0}
    index_of = {int(g): i for i, g in enumerate(snap["env_ids"])}
    locals_ = [index_of[int(g)] for g in ids]
    rbound = np.asarray([env_static[int(g)][3] for g in ids], dtype=float)
    centers_all = np.asarray([box["center"] for box in boxes])
    halves_all = np.asarray([box["half"] for box in boxes])
    best = {"m": float("inf"), "frame": None}
    intersects = False
    # The pendant is static, so a static environment geom has a
    # frame-independent clearance and is measured once. Only geoms that
    # actually move across the retained trajectory need every frame.
    moving = np.zeros(len(locals_), dtype=bool)
    for i, local in enumerate(locals_):
        track = snap["env_pos"][:, local, :]
        moving[i] = bool(np.max(np.abs(track - track[0])) > 1e-9)
    frames = list(range(snap["n_frames"]))
    for frame in frames:
        if frame > 0 and not moving.any():
            break
        pos = snap["env_pos"][frame][locals_]
        delta = np.maximum(
            np.abs(pos[:, None, :] - centers_all[None, :, :]) - halves_all[None, :, :],
            0.0,
        )
        lower = np.linalg.norm(delta, axis=2) - rbound[:, None]
        if float(np.min(lower)) > SCREEN_MARGIN_M:
            bound = float(np.min(lower))
            if bound < best["m"]:
                best = {"m": bound, "frame": frame, "box": None,
                        "env_geom": None, "env_body": None, "exact": False}
            continue
        for bi, box in enumerate(boxes):
            for local in np.flatnonzero(lower[:, bi] <= SCREEN_MARGIN_M).tolist():
                if frame > 0 and not moving[local]:
                    continue          # already measured at frame 0, cannot change
                geom_id = int(ids[local])
                gtype, size, verts, _, gname, bname = env_static[geom_id]
                if (
                    allow_hood_top
                    and box["name"] == "crossbar"
                    and gname == "hood_top"
                ):
                    continue
                shape = GeomShape(
                    gtype,
                    snap["env_pos"][frame][locals_[local]],
                    snap["env_mat"][frame][locals_[local]],
                    size,
                    verts,
                )
                if not shape.supported:
                    continue
                distance = float(gjk_distance(shape, box["shape"]))
                if distance <= 0.0:
                    intersects = True
                if distance < best["m"]:
                    best = {"m": distance, "frame": frame, "box": box["name"],
                            "env_geom": gname, "env_body": bname, "exact": True}
    return {
        "min_m": None if best["frame"] is None else best["m"],
        "witness": None if best["frame"] is None else best,
        "intersects": intersects,
        "hood_top_flush_allowlisted": allow_hood_top,
        "n_env_geoms_tested": len(ids),
        "n_moving_env_geoms": int(moving.sum()),
        "static_geoms_measured_once": True,
    }


__all__ = [
    "CLEARANCE_FLOOR_M",
    "POSE_IDS",
    "POSE_OFFSETS_M",
    "RISK_BAND_M",
    "SCREEN_MARGIN_M",
    "build_assembly",
    "environment_candidate_geoms",
    "environment_clearance",
    "lattice_candidates",
    "score_candidate_against_snapshot",
    "snapshot_row",
    "traversal_direction",
    "window_of",
]
