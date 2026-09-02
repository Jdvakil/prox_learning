#!/usr/bin/env python3
"""Render table-visible multi-POV clips for twelve V10.10 four-object episodes.

**These are re-simulations, not replays, and that difference is real.** The V5
rows carry a `trajectory.json` holding the full generalized position at every
step, so a V5 clip can restore recorded state and is exactly the collected
episode. The V10.10 collector emitted no such file — its `trajectory.h5` stores
only the panda articulation, and `env_states/actors` is empty, so the cup and
clutter poses per frame were never retained. The uploaded `trajectory.json` on
HuggingFace is a compact index that says so itself.

Each episode is therefore re-run from its frozen seed with the same expert
planner, which is state-based rather than vision-based and so is expected to
reproduce. That expectation is checked rather than assumed: every clip records
the re-run's step count, success and contact totals against the values in the
collection ledger, and the manifest reports any divergence.

Camera poses were chosen by measurement, not by eye. A segmentation sweep over
azimuth, elevation and distance counted how many distinct table entities -- the
target cup and the four active clutter bodies -- are visible in each pose. Only
three azimuths reach full 5/5 visibility; the enclosure walls and the arm
occlude at least one object everywhere else. All six shipped views come from
those three azimuths, and per-clip visibility is re-measured during the rollout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from pact_place_v1010_contract import (  # noqa: E402
    ACTIVE_CLUTTER_SLOTS, ACTIVE_CLUTTER_UIDS, COLLECTION_ROOT, OBJECT_LABELS,
    SCENE_BY_POSE, build_row, cell_key, cells,
)

DEFAULT_OUT = ROOT / "diagnostics_output/pact_place_v1010_paper_views"
N_EPISODES = 12
FPS = 1000.0 / 66.0
FRAME_WH = (960, 540)
AIM = [0.664, 0.006, 0.82]          # measured: median of the active objects and the cup
SEG_EVERY = 25                       # segmentation sample cadence during the rollout

# (azimuth deg, elevation deg, distance m, vertical FOV). Every entry measured
# at 5/5 table-entity visibility in the sweep.
VIEWS: dict[str, tuple[float, float, float, float]] = {
    "table_front_high": (195, 45, 1.35, 46.0),
    "table_front":      (195, 35, 1.35, 46.0),
    "table_left_high":  (210, 45, 1.35, 46.0),
    "table_left":       (225, 25, 1.35, 46.0),
    "table_left_mid":   (225, 35, 1.35, 46.0),
    "table_wide":       (195, 45, 2.10, 46.0),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def accepted_rows() -> list[dict[str, Any]]:
    ledger = ROOT / COLLECTION_ROOT / "ledger.jsonl"
    return [json.loads(x) for x in ledger.read_text().splitlines()
            if x.strip() and json.loads(x)["accepted"]]


def select_episodes() -> list[dict[str, Any]]:
    """One episode for every (family, pose) pair, with sides alternating.

    That is exactly twelve cells: all four layout families, all three pendant
    poses, and six left / six right, so no axis of the environment is
    over-represented. Within a cell the lowest attempt_index is used, which is
    the first success that cell produced.
    """
    by_cell: dict[str, list[dict[str, Any]]] = {}
    for row in accepted_rows():
        by_cell.setdefault(row["cell"], []).append(row)
    combos = sorted({(f, p) for f, _s, p in cells()})
    picked = []
    for index, (family, pose) in enumerate(combos):
        side = "left" if index % 2 == 0 else "right"
        key = cell_key(family, side, pose)
        members = sorted(by_cell.get(key, []), key=lambda r: int(r["attempt_index"]))
        if not members:
            raise SystemExit(f"cell {key} has no accepted row")
        picked.append(members[0])
    if len(picked) != N_EPISODES:
        raise SystemExit(f"selected {len(picked)} episodes")
    return picked


def camera_pose(aim, az_deg, el_deg, dist):
    aim = np.asarray(aim, dtype=float)
    a, e = np.radians(az_deg), np.radians(el_deg)
    position = aim + dist * np.array(
        [np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    forward = aim - position
    forward /= np.linalg.norm(forward)
    up_hint = np.array([0.0, 0.0, 1.0])
    if abs(float(forward @ up_hint)) > 0.98:
        up_hint = np.array([1.0, 0.0, 0.0])
    right = np.cross(forward, up_hint)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)
    return position, forward, up


def render_episode(job: dict[str, Any]) -> dict[str, Any]:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("MLSPACES_ASSETS_DIR", str(ROOT / "assets"))
    os.environ.setdefault("PACT_CONTACT_AUDIT_SUMMARY_ONLY", "1")
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = "1"
    os.environ.pop("DISPLAY", None)
    for path in (ROOT / "scripts", ROOT / "submodules" / "molmospaces"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    import cv2
    import mujoco
    from molmo_spaces.data_generation.pipeline import (
        ParallelRolloutRunner, cleanup_episode_resources, setup_policy,
    )
    from run_pact_place_expert_screen import (
        _make_config, disallowed_initial_contacts,
        initial_robot_environment_contacts,
    )

    ledger_row = job["ledger_row"]
    family, side, pose_id = ledger_row["cell"].split("|")
    row = build_row(family, side, pose_id, int(ledger_row["attempt_index"]))
    if row["attempt_id"] != ledger_row["attempt_id"]:
        raise RuntimeError("rebuilt row does not match the ledger attempt")

    out_dir = Path(job["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    width, height = FRAME_WH
    scratch = Path(tempfile.mkdtemp(prefix=f"v1010paper_{ledger_row['attempt_id'][:8]}_"))
    sampler = task = policy = None
    writers: dict[str, Any] = {}
    try:
        scene = ROOT / row["pact_v1010_scene_relative"]
        config = _make_config(scratch / "d.json", scene_xml=scene,
                              sampler_class="PactPlaceCorridorV1010FourObjectSampler")
        config.output_dir = scratch
        sampler_cls = config.task_sampler_config.task_sampler_class
        if sampler_cls.__name__ != "PactPlaceCorridorV1010FourObjectSampler":
            raise RuntimeError(f"resolved {sampler_cls.__name__}")
        # Mirror the collection's sampling-retry loop exactly. Some cells only
        # produce a usable draw after a retry -- IK can fail for the pregrasp
        # pose, or settled clutter can overlap the target -- and the ledger does
        # not record which retry won, so the loop has to be replayed. Retry
        # seeds come from the V10.8 contract because that is what the V10.10
        # collection actually called; using the V10.10 stream here would sample
        # a different scene from the recorded one.
        from pact_place_v108_contract import cell_seed as v108_cell_seed

        retries = int(row["max_sampling_retries"])
        attempts_log: list[str] = []
        task = policy = None
        initial_reset_result = None
        for retry_index in range(retries + 1):
            if retry_index == 0:
                seed = int(row["task_seed_u32"])
            else:
                seed = int(v108_cell_seed(
                    row["family_id"], row["intrusion_side"], row["pose_id"],
                    int(row["attempt_index"]) * 1000 + retry_index)["seed_u32"])
            sampler = sampler_cls(config)
            sampler.seed_task_sampling(seed)
            sampler.set_pact_manifest_row(row)
            try:
                task = sampler.sample_task(house_index=1)
                if task is None:
                    raise RuntimeError("sample_task returned None")
                policy = setup_policy(config, task, None, None)
                initial_reset_result = task.reset()
                rejected = disallowed_initial_contacts(
                    initial_robot_environment_contacts(task.env))
                if rejected:
                    raise RuntimeError(f"initial contact n={len(rejected)}")
            except Exception as error:  # noqa: BLE001 - pre-rollout only
                attempts_log.append(f"retry {retry_index}: "
                                    f"{type(error).__name__}: {error}"[:160])
                cleanup_episode_resources(
                    task=task, policy=policy, task_sampler=sampler,
                    preloaded_policy=None, close_task_sampler=True)
                task = policy = sampler = None
                continue
            break
        if task is None:
            raise RuntimeError(f"sampling exhausted {retries} retries: {attempts_log}")

        env = task.env
        model = env.current_model
        active = list(getattr(sampler, "_pact_active_clutter_names", []))
        if len(active) != 4:
            raise RuntimeError(f"{len(active)} active clutter bodies, expected 4")

        # entity map for measuring visibility during the rollout
        def root_name(geom):
            body = int(model.geom_bodyid[geom])
            return model.body(int(model.body_rootid[body])).name or ""
        entities = sorted({root_name(g) for g in range(model.ngeom)
                           if "cavity_obj" in root_name(g)
                           or any(f"pact_clutter_{s}" in root_name(g)
                                  for s in ACTIVE_CLUTTER_SLOTS)})
        entity_index = {n: i for i, n in enumerate(entities)}
        emap = np.array([entity_index.get(root_name(g), -1)
                         for g in range(model.ngeom)], dtype=np.int32)

        stem = f"{ledger_row['cell'].replace('|', '_')}"
        for view in job["views"]:
            path = out_dir / f"{stem}__{view}.mp4"
            writers[view] = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(FPS), (width, height))

        poses = {v: camera_pose(AIM, *VIEWS[v][:3]) for v in job["views"]}
        fovs = {v: VIEWS[v][3] for v in job["views"]}
        seen_counts: dict[str, list[int]] = {v: [] for v in job["views"]}
        frames = {"n": 0}

        original_get_action = policy.get_action

        def hooked(observation):
            action = original_get_action(observation)
            index = frames["n"]
            for view in job["views"]:
                position, forward, up = poses[view]
                image = np.asarray(env._render_frame(
                    position, forward, up, fovs[view], segmentation=False))
                if image.shape[:2] != (height, width):
                    image = cv2.resize(image, (width, height),
                                       interpolation=cv2.INTER_AREA)
                writers[view].write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                if index % SEG_EVERY == 0:
                    seg = np.asarray(env._render_frame(
                        position, forward, up, fovs[view], segmentation=True))
                    geom = np.where(seg[:, :, 1] == mujoco.mjtObj.mjOBJ_GEOM,
                                    seg[:, :, 0], -1)
                    ent = np.where(geom >= 0,
                                   emap[np.clip(geom, 0, model.ngeom - 1)], -1)
                    seen_counts[view].append(
                        sum(1 for i in range(len(entities))
                            if int((ent == i).sum()) >= 150))
            frames["n"] = index + 1
            return action

        policy.get_action = hooked
        task_success = bool(ParallelRolloutRunner.run_single_rollout(
            episode_seed=int(row["task_seed_u64"]), task=task, policy=policy,
            end_on_success=False, initial_reset_result=initial_reset_result))
        info = policy.get_info()
        info.pop("trajectory", None)
        audit = info.get("pact_contact_audit") or {}
        totals = audit.get("contact_class_totals") or {}

        for writer in writers.values():
            writer.release()
        writers = {}

        clips = []
        for view in job["views"]:
            path = out_dir / f"{stem}__{view}.mp4"
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"{path.name} not written")
            counts = seen_counts[view]
            clips.append({
                "view": view, "file": path.name, "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "azimuth_deg": VIEWS[view][0], "elevation_deg": VIEWS[view][1],
                "distance_m": VIEWS[view][2], "fov_deg": VIEWS[view][3],
                "entities_total": len(entities),
                "entities_seen_min": int(min(counts)) if counts else None,
                "entities_seen_mean": round(float(np.mean(counts)), 2) if counts else None,
                "segmentation_samples": len(counts),
            })

        recorded_steps = int(ledger_row["episode_steps"])
        reproduced = {
            "recorded_episode_steps": recorded_steps,
            "rerun_frames": frames["n"],
            "steps_match": abs(frames["n"] - recorded_steps) <= 1,
            "recorded_task_success": bool(ledger_row["task_success"]),
            "rerun_task_success": task_success,
            "success_match": task_success == bool(ledger_row["task_success"]),
            "recorded_contacts": ledger_row.get("contact_class_totals"),
            "rerun_contacts": {k: int(v) for k, v in totals.items()},
        }
        reproduced["faithful"] = bool(
            reproduced["steps_match"] and reproduced["success_match"])
        return {
            "attempt_id": ledger_row["attempt_id"], "cell": ledger_row["cell"],
            "family_id": family, "intrusion_side": side, "pose_id": pose_id,
            "attempt_index": int(ledger_row["attempt_index"]),
            "task_seed_u32": int(row["task_seed_u32"]),
            "sampling_retries_used": len(attempts_log),
            "sampling_rejections": attempts_log,
            "row_sha256": row["row_sha256"],
            "active_clutter_uids": dict(ACTIVE_CLUTTER_UIDS),
            "entities_tracked": entities,
            "frames": frames["n"], "duration_s": round(frames["n"] / FPS, 2),
            "reproduction": reproduced, "clips": clips, "ok": True,
        }
    except Exception as exc:  # noqa: BLE001
        for writer in writers.values():
            try:
                writer.release()
            except Exception:  # noqa: BLE001
                pass
        return {"attempt_id": ledger_row.get("attempt_id"), "ok": False,
                "error": f"{type(exc).__name__}: {exc}"[:300]}
    finally:
        try:
            cleanup_episode_resources(task=task, policy=policy, task_sampler=sampler,
                                      preloaded_policy=None, close_task_sampler=True)
        except Exception:  # noqa: BLE001
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--views", nargs="+", default=sorted(VIEWS))
    args = parser.parse_args()
    unknown = sorted(set(args.views) - set(VIEWS))
    if unknown:
        raise SystemExit(f"unknown views: {unknown}")

    episodes = select_episodes()
    print(f"{len(episodes)} episodes x {len(args.views)} views on {args.workers} workers",
          flush=True)
    for row in episodes:
        print(f"  {row['cell']:44s} attempt {row['attempt_index']:2d} "
              f"steps {row['episode_steps']}", flush=True)

    jobs = [{"ledger_row": r, "out_dir": str(args.out / "videos"),
             "views": args.views} for r in episodes]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(render_episode, j): j for j in jobs}
        for done, future in enumerate(as_completed(futures), 1):
            try:
                results.append(future.result())
            except Exception as exc:  # noqa: BLE001
                results.append({"ok": False, "error": f"{type(exc).__name__}: {exc}"[:300]})
            print(f"  {done}/{len(jobs)} done", flush=True)
    results.sort(key=lambda r: r.get("cell") or "")
    failed = [r for r in results if not r.get("ok")]
    unfaithful = [r for r in results
                  if r.get("ok") and not r["reproduction"]["faithful"]]

    manifest = {
        "schema_version": "pact_place_v1010_paper_views_v1",
        "role": "table-visible multi-POV clips of twelve V10.10 four-object episodes",
        "resimulation_not_replay":
            "the V10.10 collector emitted no full-state trajectory (its "
            "trajectory.h5 stores only the panda articulation and "
            "env_states/actors is empty), so each episode is re-run from its "
            "frozen seed with the same state-based expert planner; every clip "
            "records the re-run against the collection ledger",
        "active_clutter_slots": list(ACTIVE_CLUTTER_SLOTS),
        "active_clutter_uids": dict(ACTIVE_CLUTTER_UIDS),
        "object_labels": dict(OBJECT_LABELS),
        "camera_selection":
            "segmentation sweep over azimuth, elevation and distance counting "
            "distinct visible table entities (target cup plus the four active "
            "clutter bodies); only azimuths 195, 210 and 225 reach 5/5, so all "
            "shipped views come from those",
        "aim_point_m": AIM, "fps": FPS, "frame_wh": list(FRAME_WH),
        "views": {v: {"azimuth_deg": VIEWS[v][0], "elevation_deg": VIEWS[v][1],
                      "distance_m": VIEWS[v][2], "fov_deg": VIEWS[v][3]}
                  for v in args.views},
        "episodes_rendered": len(results) - len(failed),
        "clips_written": sum(len(r.get("clips", [])) for r in results),
        "failures": failed,
        "unfaithful_reruns": [r["cell"] for r in unfaithful],
        "episodes": results,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: manifest[k] for k in
                      ("episodes_rendered", "clips_written", "unfaithful_reruns")},
                     indent=2))
    print(f"failures: {len(failed)}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
