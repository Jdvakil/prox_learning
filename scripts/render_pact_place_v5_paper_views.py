#!/usr/bin/env python3
"""Render multi-POV replay videos for twelve V5 rows, for the paper.

Each clip is a **replay**, not a re-run: ``apply_recorded_qpos`` restores the
recorded generalized position frame by frame and calls ``mj_forward``, so what
you see is the episode that was collected, rendered from a different camera. No
physics is re-stepped and no seed is re-drawn.

Selection is stated rather than eyeballed. All 152 recovered rows are clean
successes, so the twelve are chosen to span the environment's geometry instead
of its outcomes: six per intrusion side, taken at evenly spaced ranks after
sorting each side by (panel_face_jitter, panel_x_jitter, role_index). That
covers the corridor-tightness axis the paper argues about, and it is
reproducible from the frozen recovery config alone.
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
MOLMO = ROOT / "submodules" / "molmospaces"
SCENE = (MOLMO / "molmo_spaces/data_generation/custom_scenes"
         / "pact_place_corridor_v2.xml")
RECOVERY = ROOT / "configs/pact_place_v5_recovery.json"
ROWS = ROOT / "assets/datagen/pact_place_corridor_v2/recovered_152/rows"
DEFAULT_OUT = ROOT / "diagnostics_output/pact_place_v5_paper_views"

N_ROWS = 12
PER_SIDE = 6
FPS = 1000.0 / 66.0
FRAME_WH = (960, 540)
CAMERA_REFERENCE_BODY = "robot_0/fr3_link0"

# Camera poses in the robot-base frame: (offset, look-at, vertical FOV).
# Chosen after rendering candidates and inspecting them: the enclosure has a
# roof and rear walls, so overhead and rear angles are occluded and are not
# included. Every view here frames the shelf, the tray and the target.
VIEWS: dict[str, tuple[list[float], list[float], float]] = {
    "table_review":  ([-1.05, -0.55, 1.30], [0.55, 0.00, 0.45], 58.0),
    "front":         ([-1.20,  0.00, 1.00], [0.60, 0.00, 0.40], 52.0),
    "left_oblique":  ([-0.95, -1.10, 1.15], [0.55, 0.00, 0.40], 55.0),
    "right_front":   ([-1.40,  0.62, 1.05], [0.60, 0.00, 0.40], 55.0),
    "high_front":    ([-0.85, -0.20, 1.80], [0.62, 0.00, 0.35], 55.0),
    "low_angle":     ([-1.10, -0.30, 0.85], [0.55, 0.00, 0.45], 55.0),
    "workspace":     ([-0.80, -0.62, 1.22], [0.78, 0.06, 0.38], 46.0),
    "over_shoulder": ([-1.45, -0.25, 1.55], [0.60, 0.00, 0.40], 50.0),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_rows() -> list[dict[str, Any]]:
    """Six per side at evenly spaced geometry ranks. Deterministic."""
    rows = json.loads(RECOVERY.read_text())["recovery_rows"]
    chosen: list[dict[str, Any]] = []
    for side in ("left", "right"):
        pool = sorted(
            (r for r in rows if r["intrusion_side"] == side),
            key=lambda r: (float(r["panel_face_jitter_m"]),
                           float(r["panel_x_jitter_m"]), int(r["role_index"])),
        )
        # evenly spaced ranks across the sorted pool, endpoints included
        ranks = [round(i * (len(pool) - 1) / (PER_SIDE - 1)) for i in range(PER_SIDE)]
        chosen.extend(pool[r] for r in ranks)
    if len(chosen) != N_ROWS:
        raise SystemExit(f"selected {len(chosen)} rows, expected {N_ROWS}")
    if len({r["role_index"] for r in chosen}) != N_ROWS:
        raise SystemExit("selection produced a duplicate row")
    return chosen


def row_directory(role_index: int, episode_id: str) -> Path:
    return ROWS / f"{int(role_index):03d}_{episode_id[:16]}"


def camera_pose(env, offset, lookat):
    from molmo_spaces.env.data_views import create_mlspaces_body

    body = create_mlspaces_body(env.current_data, CAMERA_REFERENCE_BODY)
    rotation = np.asarray(body.pose[:3, :3], dtype=float)
    translation = np.asarray(body.pose[:3, 3], dtype=float)
    position = rotation @ np.asarray(offset, dtype=float) + translation
    target = rotation @ np.asarray(lookat, dtype=float) + translation
    forward = target - position
    forward /= np.linalg.norm(forward)
    up_hint = rotation @ np.asarray([0.0, 0.0, 1.0])
    if abs(float(np.dot(forward, up_hint))) > 0.98:
        up_hint = rotation @ np.asarray([1.0, 0.0, 0.0])
    right = np.cross(forward, up_hint)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)
    return position, forward, up


def render_row(job: dict[str, Any]) -> dict[str, Any]:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("MLSPACES_ASSETS_DIR", str(ROOT / "assets"))
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = "1"
    os.environ.pop("DISPLAY", None)
    for path in (ROOT / "scripts", MOLMO):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    import cv2

    from run_pact_place_expert_screen import _make_config
    from run_pact_place_v6c_replay_videos import apply_recorded_qpos

    row = job["row"]
    role = int(row["role_index"])
    directory = row_directory(role, row["episode_id"])
    trajectory = json.loads((directory / "trajectory.json").read_text())
    result = json.loads((directory / "result.json").read_text())
    steps = trajectory["steps"]
    if trajectory["row_sha256"] != row["row_sha256"]:
        raise RuntimeError(f"role {role}: trajectory row hash does not match the config")

    out_dir = Path(job["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    width, height = FRAME_WH
    scratch = Path(tempfile.mkdtemp(prefix=f"v5paper_{role:03d}_"))
    sampler = task = None
    writers: dict[str, Any] = {}
    try:
        config = _make_config(scratch / "d.json", scene_xml=SCENE, sampler_class=None)
        sampler_cls = config.task_sampler_config.task_sampler_class
        if sampler_cls.__name__ != "PactPlaceCorridorV2Sampler":
            raise RuntimeError(f"resolved {sampler_cls.__name__}, expected the V2 sampler")
        sampler = sampler_cls(config)
        sampler.seed_task_sampling(int(row["task_seed_u32"]))
        sampler.set_pact_manifest_row(row)
        task = sampler.sample_task(house_index=int(row["scene_template_house_index"]))
        env = task.env

        stem = f"role{role:03d}_{row['intrusion_side']}"
        for view in job["views"]:
            path = out_dir / f"{stem}__{view}.mp4"
            writers[view] = (cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(FPS),
                (width, height)), path)

        for step in steps:
            apply_recorded_qpos(env, step["qpos"])
            for view in job["views"]:
                offset, lookat, fov = VIEWS[view]
                position, forward, up = camera_pose(env, offset, lookat)
                frame = np.asarray(env._render_frame(
                    position, forward, up, fov, segmentation=False))
                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height),
                                       interpolation=cv2.INTER_AREA)
                writers[view][0].write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        for writer, _ in writers.values():
            writer.release()
        writers = {}

        clips = []
        for view in job["views"]:
            path = out_dir / f"{stem}__{view}.mp4"
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"{path.name} was not written")
            clips.append({"view": view, "file": path.name,
                          "bytes": path.stat().st_size,
                          "sha256": sha256_file(path)})
        return {
            "role_index": role, "episode_id": row["episode_id"],
            "intrusion_side": row["intrusion_side"],
            "panel_x_jitter_m": float(row["panel_x_jitter_m"]),
            "panel_face_jitter_m": float(row["panel_face_jitter_m"]),
            "task_seed_u32": int(row["task_seed_u32"]),
            "row_sha256": row["row_sha256"],
            "frames": len(steps),
            "episode_steps": int(result["episode_steps"]),
            "clean_success": bool(result["clean_success"]),
            "task_success": bool(result["task_success"]),
            "duration_s": round(len(steps) / FPS, 2),
            "clips": clips, "ok": True,
        }
    except Exception as exc:  # noqa: BLE001
        for writer, _ in writers.values():
            try:
                writer.release()
            except Exception:  # noqa: BLE001
                pass
        return {"role_index": role, "ok": False,
                "error": f"{type(exc).__name__}: {exc}"[:300]}
    finally:
        try:
            if sampler is not None:
                sampler.close()
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

    chosen = select_rows()
    print(f"selected {len(chosen)} rows; rendering {len(args.views)} views each "
          f"on {args.workers} workers", flush=True)
    for row in chosen:
        print(f"  role {int(row['role_index']):3d} {row['intrusion_side']:5s} "
              f"panel_x {float(row['panel_x_jitter_m']):+.4f} "
              f"face {float(row['panel_face_jitter_m']):+.4f} "
              f"steps {row['screen_episode_steps']}", flush=True)

    jobs = [{"row": r, "out_dir": str(args.out / "videos"), "views": args.views}
            for r in chosen]
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(render_row, j): j for j in jobs}
        for done, future in enumerate(as_completed(futures), 1):
            try:
                results.append(future.result())
            except Exception as exc:  # noqa: BLE001
                results.append({"role_index": int(futures[future]["row"]["role_index"]),
                                "ok": False, "error": f"{type(exc).__name__}: {exc}"[:300]})
            print(f"  {done}/{len(jobs)} done", flush=True)
    results.sort(key=lambda r: r["role_index"])
    failed = [r for r in results if not r.get("ok")]

    manifest = {
        "schema_version": "pact_place_v5_paper_views_v1",
        "role": "multi-POV replay clips of twelve V5 rows, for the paper",
        "replay_not_rerun":
            "each frame restores the recorded qpos and calls mj_forward; no "
            "physics is re-stepped and no seed is re-drawn",
        "source_rows": str(ROWS.relative_to(ROOT)),
        "recovery_config": str(RECOVERY.relative_to(ROOT)),
        "recovery_config_sha256": sha256_file(RECOVERY),
        "scene": str(SCENE.relative_to(ROOT)),
        "scene_sha256": sha256_file(SCENE),
        "sampler_class": "PactPlaceCorridorV2Sampler",
        "selection_rule":
            "all 152 recovered rows are clean successes, so the twelve span "
            "geometry rather than outcome: six per intrusion side, taken at "
            "evenly spaced ranks after sorting each side by "
            "(panel_face_jitter_m, panel_x_jitter_m, role_index)",
        "views": {name: {"offset_m": off, "lookat_m": look, "fov_deg": fov,
                         "frame": "robot-base (robot_0/fr3_link0)"}
                  for name, (off, look, fov) in VIEWS.items()
                  if name in args.views},
        "fps": FPS, "frame_wh": list(FRAME_WH),
        "rows_rendered": len(results) - len(failed),
        "clips_written": sum(len(r.get("clips", [])) for r in results),
        "failures": failed,
        "rows": results,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"rows_rendered": manifest["rows_rendered"],
                      "clips_written": manifest["clips_written"],
                      "failures": len(failed),
                      "out": str(args.out.relative_to(ROOT))}, indent=2))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
