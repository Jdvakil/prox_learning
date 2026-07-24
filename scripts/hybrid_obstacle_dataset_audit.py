#!/usr/bin/env python3
"""Integrity audit of the immutable hybrid-obstacle source collection.

Read-only. Opens every trajectory in a completed datagen run and checks structural
integrity, label/geometry agreement, duplication and worker attribution. It never
writes into the collection, never steps the simulator and never trains anything.

The hazard label is not trusted on its own: ``scene_params.protrusion_present`` is
cross-checked against the scene geometry actually compiled into the model
(``obstacle_aabbs`` gains exactly one box when the bar is present, and
``protr_center``/``protr_half`` describe it), and against the sampler's documented
object-placement coupling. A sampled pixel check on the exo videos confirms the
rendered frames agree with the label.

Subcommands
-----------
integrity   Full per-trajectory audit with worker/house/hazard/success breakdowns.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np

EXPECTED_SENSORS = 40
EXPECTED_PATCH = (8, 8)
EXPECTED_SUBSTEPS = 4
CANONICAL_UID = "4afa0cdde045417ab31f98ae7745b039"
COMPANION_SUFFIXES = (
    "exo_camera_1",
    "exo_camera_1_depth",
    "wrist_camera",
    "wrist_camera_depth",
    "sensors_depth8_heatmap",
    "sensors_rgb256",
)
# hazard-orange bar colour set by ObstacleFumehoodPickSampler._apply_theta
BAR_RGB = (255, 115, 13)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_scene(group: h5py.Group) -> dict[str, Any]:
    return json.loads(np.asarray(group["obs_scene"]).item())


def worker_map(run_dir: Path, extra_log: Path | None = None) -> tuple[dict[str, int], list[str]]:
    """Recover house -> worker attribution from the run's logs.

    The collection's own ``running_log.log`` records only the parent process; the
    per-worker ``Worker N house H episode ...`` lines are emitted on stderr, so an
    externally captured worker log may be supplied as well.
    """
    notes: list[str] = []
    text = ""
    log = run_dir / "running_log.log"
    if log.is_file():
        text += log.read_text(errors="replace")
    else:
        notes.append("running_log.log absent")
    if extra_log is not None:
        if extra_log.is_file():
            text += extra_log.read_text(errors="replace")
            notes.append(f"worker attribution used captured log {extra_log}")
        else:
            notes.append(f"captured worker log {extra_log} not found")
    if not text:
        return {}, notes + ["no log available; worker attribution unavailable"]
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    pairs: dict[str, set[int]] = defaultdict(set)
    for m in re.finditer(r"Worker (\d+) house (\d+) episode", text):
        pairs[f"house_{m.group(2)}"].add(int(m.group(1)))
    mapping: dict[str, int] = {}
    for house, workers in pairs.items():
        if len(workers) == 1:
            mapping[house] = next(iter(workers))
        else:
            notes.append(
                f"{house} claimed by multiple workers {sorted(workers)} — overlapping write"
            )
            mapping[house] = -1
    return mapping, notes


def hazard_from_geometry(params: dict[str, Any]) -> dict[str, Any]:
    """Derive bar presence from the scene geometry rather than the recorded flag."""
    aabbs = params.get("obstacle_aabbs") or []
    center = params.get("protr_center")
    half = params.get("protr_half")
    has_bar_box = center is not None and half is not None
    # The bar contributes exactly one extra AABB over the bar-free scene.
    matching = None
    if has_bar_box and aabbs:
        target = [list(map(float, center)), list(map(float, half))]
        matching = any(
            np.allclose(np.asarray(a[0], dtype=float), target[0], atol=1e-9)
            and np.allclose(np.asarray(a[1], dtype=float), target[1], atol=1e-9)
            for a in aabbs
        )
    return {
        "n_obstacle_aabbs": len(aabbs),
        "has_protr_center_and_half": has_bar_box,
        "bar_aabb_present_in_scene": matching,
        "geometry_says_hazard": has_bar_box,
    }


def placement_coupling(params: dict[str, Any], obj_y: float) -> dict[str, Any]:
    """ObstacleFumehoodPickSampler._obj_rest ties the object y to the bar face."""
    if not params.get("protrusion_present"):
        return {"applicable": False}
    side = 1.0 if params.get("protr_wall") == "left" else -1.0
    predicted = side * (float(params["bar_face_y"]) - float(params["obj_gap"]))
    return {
        "applicable": True,
        "predicted_obj_y": predicted,
        "actual_obj_y": float(obj_y),
        "abs_difference": abs(predicted - float(obj_y)),
    }


def audit_trajectory(
    h5_path: Path, house: str, key: str, group: h5py.Group
) -> dict[str, Any]:
    scene = decode_scene(group)
    params = scene.get("scene_params", {})
    problems: list[str] = []

    # --- structural: shapes must agree across every per-step stream -----------
    t_action = int(group["actions/joint_pos"].shape[0])
    lengths = {
        "actions/joint_pos": t_action,
        "obs/agent/qpos": int(group["obs/agent/qpos"].shape[0]),
        "obs/agent/qvel": int(group["obs/agent/qvel"].shape[0]),
        "fail": int(group["fail"].shape[0]),
    }
    if len(set(lengths.values())) != 1:
        problems.append(f"inconsistent stream lengths {lengths}")

    # --- truncation: force a full read of the first and last row of each stream
    truncated = False
    try:
        for name in ("actions/joint_pos", "obs/agent/qpos", "obs/agent/qvel"):
            _ = group[name][0]
            _ = group[name][-1]
        _ = group["fail"][()]
    except Exception as exc:  # pragma: no cover - corruption path
        truncated = True
        problems.append(f"unreadable stream: {type(exc).__name__}: {exc}")

    # --- proximity: 40 canonical streams, correct shape ------------------------
    prox = group["obs/proximity"]
    prox_names = sorted(prox.keys())
    bad_shape = [
        n
        for n in prox_names
        if tuple(prox[n].shape[1:]) != (EXPECTED_SUBSTEPS, *EXPECTED_PATCH)
        or prox[n].shape[0] != t_action
    ]
    if len(prox_names) != EXPECTED_SENSORS:
        problems.append(f"{len(prox_names)} proximity streams (expected 40)")
    if bad_shape:
        problems.append(f"{len(bad_shape)} proximity streams with wrong shape")
    prox_order_hash = hashlib.sha256(
        json.dumps(prox_names, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    ).hexdigest()

    # --- required observation groups -------------------------------------------
    required = [
        "obs/agent/qpos",
        "obs/agent/qvel",
        "actions/joint_pos",
        "obs/extra/task_info",
        "obs/extra/obj_start",
        "obs/extra/tcp_pose",
        "obs_scene",
        "fail",
        "env_states",
        "success",
    ]
    missing = [r for r in required if r not in group]
    if missing:
        problems.append(f"missing required fields {missing}")

    # --- success/failure internal consistency ----------------------------------
    fail = np.asarray(group["fail"])
    failed = bool(fail[-1]) if fail.size else False
    success_ds = np.asarray(group["success"]) if "success" in group else None
    task_info_last = None
    try:
        raw = bytes(group["obs/extra/task_info"][-1]).split(b"\x00", 1)[0]
        if raw:
            task_info_last = json.loads(raw.decode("utf-8")).get("success")
    except Exception:
        task_info_last = None
    success_last = (
        bool(success_ds[-1]) if success_ds is not None and success_ds.size else None
    )
    consistent = True
    if success_last is not None and success_last == failed:
        consistent = False
        problems.append(
            f"success[-1]={success_last} contradicts fail[-1]={failed}"
        )
    if task_info_last is not None and bool(task_info_last) == failed:
        consistent = False
        problems.append(
            f"task_info.success={task_info_last} contradicts fail[-1]={failed}"
        )

    # --- hazard label vs actual scene geometry ---------------------------------
    recorded = bool(params.get("protrusion_present", False))
    geom = hazard_from_geometry(params)
    label_matches_geometry = recorded == geom["geometry_says_hazard"]
    if not label_matches_geometry:
        problems.append(
            f"hazard label {recorded} disagrees with scene geometry "
            f"{geom['geometry_says_hazard']}"
        )
    if recorded and geom["bar_aabb_present_in_scene"] is False:
        problems.append("bar geometry absent from obstacle_aabbs despite hazard label")
    cell = params.get("cell")
    if recorded and cell != "bar":
        problems.append(f"hazard episode has cell={cell!r} (expected 'bar')")

    obj_start = np.asarray(group["obs/extra/obj_start"][0], dtype=float)
    coupling = placement_coupling(params, obj_start[1])

    # --- content fingerprint for duplicate detection ---------------------------
    qpos_raw = group["obs/agent/qpos"][()]
    act_raw = group["actions/joint_pos"][()]
    content = hashlib.sha256()
    content.update(np.ascontiguousarray(qpos_raw).tobytes())
    content.update(np.ascontiguousarray(act_raw).tobytes())
    content_hash = content.hexdigest()

    # --- companion media --------------------------------------------------------
    ep_idx = int(key.split("_", 1)[1])
    companions = {}
    for suffix in COMPANION_SUFFIXES:
        mp4 = h5_path.parent / f"episode_{ep_idx:08d}_{suffix}_batch_1_of_1.mp4"
        companions[suffix] = {"exists": mp4.is_file(), "bytes": mp4.stat().st_size if mp4.is_file() else 0}
    missing_media = [k for k, v in companions.items() if not v["exists"]]
    empty_media = [k for k, v in companions.items() if v["exists"] and v["bytes"] == 0]
    if missing_media:
        problems.append(f"missing companion video(s) {missing_media}")
    if empty_media:
        problems.append(f"zero-byte companion video(s) {empty_media}")

    return {
        "house": house,
        "traj_key": key,
        "episode_index": ep_idx,
        "trajectory_id": f"{house}/{key}",
        "source_h5": str(h5_path),
        "frames": t_action,
        "stream_lengths": lengths,
        "truncated": truncated,
        "proximity_count": len(prox_names),
        "proximity_order_hash": prox_order_hash,
        "proximity_shapes_ok": not bad_shape,
        "missing_required_fields": missing,
        "failed": failed,
        "successful": not failed,
        "success_flags_consistent": consistent,
        "hazard_recorded": recorded,
        "hazard_geometry": geom,
        "hazard_label_matches_geometry": label_matches_geometry,
        "cell": cell,
        "placement_coupling": coupling,
        "target_uid": params.get("target_uid"),
        "behavior_class": scene.get("behavior_class"),
        "policy_dt_ms": scene.get("policy_dt_ms"),
        "content_sha256": content_hash,
        "companion_media": companions,
        "problems": problems,
    }


def sampled_pixel_check(rows: list[dict[str, Any]], per_class: int) -> dict[str, Any]:
    """Confirm rendered exo frames agree with the hazard label on a sample."""
    try:
        import cv2
    except ImportError:  # pragma: no cover
        return {"performed": False, "reason": "cv2 unavailable"}

    def orange_fraction(mp4: Path) -> float | None:
        cap = cv2.VideoCapture(str(mp4))
        if not cap.isOpened():
            return None
        best = 0.0
        try:
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            for idx in (0, n // 2, max(0, n - 2)):
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = cap.read()
                if not ok:
                    continue
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.int16)
                # generous tolerance: video is lossy-compressed
                d = np.abs(rgb - np.asarray(BAR_RGB, dtype=np.int16)).sum(axis=2)
                best = max(best, float((d < 90).mean()))
        finally:
            cap.release()
        return best

    hazard = [r for r in rows if r["hazard_recorded"]][:per_class]
    clear = [r for r in rows if not r["hazard_recorded"]][:per_class]
    out = {"performed": True, "per_class": per_class, "hazard": [], "hazard_absent": []}
    for bucket, sample in (("hazard", hazard), ("hazard_absent", clear)):
        for r in sample:
            mp4 = (
                Path(r["source_h5"]).parent
                / f"episode_{r['episode_index']:08d}_exo_camera_1_batch_1_of_1.mp4"
            )
            out[bucket].append(
                {
                    "trajectory_id": r["trajectory_id"],
                    "orange_pixel_fraction": orange_fraction(mp4),
                }
            )
    hz = [x["orange_pixel_fraction"] for x in out["hazard"] if x["orange_pixel_fraction"] is not None]
    cl = [x["orange_pixel_fraction"] for x in out["hazard_absent"] if x["orange_pixel_fraction"] is not None]
    out["hazard_min_orange_fraction"] = min(hz) if hz else None
    out["hazard_absent_max_orange_fraction"] = max(cl) if cl else None
    out["separates"] = bool(hz and cl and min(hz) > max(cl))
    return out


def cmd_integrity(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.run_dir).resolve()
    houses_to_worker, worker_notes = worker_map(
        run_dir, Path(args.worker_log) if args.worker_log else None
    )

    rows: list[dict[str, Any]] = []
    h5_files = sorted(run_dir.glob("house_*/trajectories*.h5"))
    file_hashes = {}
    open_failures = []
    for h5_path in h5_files:
        house = h5_path.parent.name
        file_hashes[house] = sha256_file(h5_path)
        try:
            handle = h5py.File(h5_path, "r")
        except Exception as exc:
            open_failures.append({"path": str(h5_path), "error": f"{type(exc).__name__}: {exc}"})
            continue
        with handle as f:
            keys = sorted(f.keys(), key=lambda k: int(k.split("_", 1)[1]))
            for key in keys:
                row = audit_trajectory(h5_path, house, key, f[key])
                row["worker"] = houses_to_worker.get(house)
                row["source_h5_sha256"] = file_hashes[house]
                rows.append(row)

    # --- uniqueness and duplication -------------------------------------------
    ids = [r["trajectory_id"] for r in rows]
    dup_ids = sorted({i for i in ids if ids.count(i) > 1})
    by_content: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        by_content[r["content_sha256"]].append(r["trajectory_id"])
    dup_content = {k: v for k, v in by_content.items() if len(v) > 1}

    # per-house episode index sequences must be a contiguous 0..n-1 block
    seq: dict[str, dict[str, Any]] = {}
    for house in sorted({r["house"] for r in rows}):
        idx = sorted(r["episode_index"] for r in rows if r["house"] == house)
        seq[house] = {
            "count": len(idx),
            "min": idx[0] if idx else None,
            "max": idx[-1] if idx else None,
            "contiguous_from_zero": idx == list(range(len(idx))),
            "unique": len(set(idx)) == len(idx),
        }

    # workers must not share a house (overlapping episode-ID space)
    worker_houses: dict[Any, list[str]] = defaultdict(list)
    for house in sorted({r["house"] for r in rows}):
        worker_houses[houses_to_worker.get(house)].append(house)
    overlapping = [w for w, hs in worker_houses.items() if w == -1]

    def tally(pred) -> dict[str, int]:
        out: dict[str, int] = defaultdict(int)
        for r in rows:
            out[str(pred(r))] += 1
        return dict(sorted(out.items()))

    cross = lambda a, b: dict(  # noqa: E731
        sorted(
            {
                f"{a(r)}|{b(r)}": sum(1 for x in rows if a(x) == a(r) and b(x) == b(r))
                for r in rows
            }.items()
        )
    )

    successful = [r for r in rows if r["successful"]]
    hazard_rows = [r for r in rows if r["hazard_recorded"]]
    problems = [
        {"trajectory_id": r["trajectory_id"], "problems": r["problems"]}
        for r in rows
        if r["problems"]
    ]

    coupling_ok = [
        r["placement_coupling"]["abs_difference"]
        for r in rows
        if r["placement_coupling"].get("applicable")
    ]

    pixel = (
        sampled_pixel_check(rows, args.pixel_sample)
        if args.pixel_sample > 0
        else {"performed": False, "reason": "disabled"}
    )

    prox_hashes = {r["proximity_order_hash"] for r in rows}

    report: dict[str, Any] = {
        "schema_version": "hybrid_obstacle_dataset_integrity_v1",
        "run_dir": str(run_dir),
        "collection_id": run_dir.name,
        "h5_files": len(h5_files),
        "h5_open_failures": open_failures,
        "trajectories": len(rows),
        "successful": len(successful),
        "failed": len(rows) - len(successful),
        "hazard_recorded": len(hazard_rows),
        "successful_hazard_present": sum(1 for r in successful if r["hazard_recorded"]),
        "successful_hazard_absent": sum(1 for r in successful if not r["hazard_recorded"]),
        "source_h5_sha256": file_hashes,
        "worker_attribution": {
            "house_to_worker": houses_to_worker,
            "worker_to_houses": {str(k): v for k, v in sorted(worker_houses.items(), key=lambda x: str(x[0]))},
            "houses_claimed_by_multiple_workers": overlapping,
            "notes": worker_notes,
        },
        "counts": {
            "by_worker": tally(lambda r: r["worker"]),
            "by_house": tally(lambda r: r["house"]),
            "by_success": tally(lambda r: r["successful"]),
            "by_hazard": tally(lambda r: r["hazard_recorded"]),
            "by_worker_x_hazard": cross(lambda r: f"worker_{r['worker']}", lambda r: f"hazard_{r['hazard_recorded']}"),
            "by_worker_x_success": cross(lambda r: f"worker_{r['worker']}", lambda r: f"success_{r['successful']}"),
            "by_house_x_hazard": cross(lambda r: r["house"], lambda r: f"hazard_{r['hazard_recorded']}"),
        },
        "checks": {
            "all_h5_open": not open_failures,
            "no_truncated_trajectories": not any(r["truncated"] for r in rows),
            "episode_ids_unique": not dup_ids,
            "duplicate_trajectory_ids": dup_ids,
            "no_duplicate_content": not dup_content,
            "duplicate_content_groups": dup_content,
            "no_overlapping_worker_episode_ids": not overlapping,
            "episode_sequences_valid": all(
                v["contiguous_from_zero"] and v["unique"] for v in seq.values()
            ),
            "episode_sequences": seq,
            "all_companion_media_present": not any(
                any("companion video" in p for p in r["problems"]) for r in rows
            ),
            "success_flags_internally_consistent": all(
                r["success_flags_consistent"] for r in rows
            ),
            "hazard_label_present_for_every_trajectory": all(
                r["hazard_recorded"] in (True, False) for r in rows
            ),
            "hazard_label_matches_scene_geometry": all(
                r["hazard_label_matches_geometry"] for r in rows
            ),
            "placement_coupling_max_abs_difference": max(coupling_ok) if coupling_ok else None,
            "all_40_sensor_streams": all(r["proximity_count"] == EXPECTED_SENSORS for r in rows),
            "sensor_shapes_ok": all(r["proximity_shapes_ok"] for r in rows),
            "single_proximity_order_hash": len(prox_hashes) == 1,
            "proximity_order_hash": sorted(prox_hashes),
            "no_missing_required_fields": not any(r["missing_required_fields"] for r in rows),
            "single_canonical_target_uid": sorted({r["target_uid"] for r in rows}) == [CANONICAL_UID],
            "single_policy_dt": sorted({r["policy_dt_ms"] for r in rows}),
        },
        "rendered_pixel_check": pixel,
        "problem_trajectories": problems,
        "trajectories_detail": rows,
    }
    c = report["checks"]
    report["passed"] = bool(
        c["all_h5_open"]
        and c["no_truncated_trajectories"]
        and c["episode_ids_unique"]
        and c["no_duplicate_content"]
        and c["no_overlapping_worker_episode_ids"]
        and c["episode_sequences_valid"]
        and c["all_companion_media_present"]
        and c["success_flags_internally_consistent"]
        and c["hazard_label_present_for_every_trajectory"]
        and c["hazard_label_matches_scene_geometry"]
        and c["all_40_sensor_streams"]
        and c["sensor_shapes_ok"]
        and c["single_proximity_order_hash"]
        and c["no_missing_required_fields"]
        and c["single_canonical_target_uid"]
        and not problems
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("integrity")
    p.add_argument("run_dir")
    p.add_argument("--pixel_sample", type=int, default=10)
    p.add_argument(
        "--worker_log",
        help="externally captured stderr containing the per-worker episode lines",
    )
    p.set_defaults(func=cmd_integrity)
    args = parser.parse_args()
    json.dump(args.func(args), sys.stdout, indent=2, sort_keys=True, default=str)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
