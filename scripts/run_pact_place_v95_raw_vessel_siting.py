#!/usr/bin/env python3
"""Proxy-only shortlist for paired-side raw vessel remediation.

This stage never authorizes admission. It vectorizes the real 40 sensor-camera
poses over retained trajectories and ranks panel-independent vessel coordinates;
the selected coordinates must subsequently pass the real rendered tensor gate.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MOLMO = ROOT / "submodules/molmospaces"
for path in (ROOT / "scripts", MOLMO):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pact_place_v9_contract import PALETTE_PATH, build_layout, load_palette, sha256_payload
from run_pact_place_expert_screen import _make_config
from run_pact_place_v9_panel_smoke import _row
from run_pact_place_v9_v0c3_causal_proximity import INBOUND_DECISION_PHASES

DEFAULT_SMOKE_ROOT = ROOT / "diagnostics_output/pact_place_v93_panel_smoke_rawfix"
DEFAULT_OUTPUT = ROOT / "diagnostics_output/pact_place_v95_v0c5_raw_siting/proxy_siting.json"
SCENE_XML = ROOT / "submodules/molmospaces/molmo_spaces/data_generation/custom_scenes/pact_place_corridor_v5.xml"
SENSOR_RANGE_M = 0.85
HALF_FOV_COS = float(np.cos(np.deg2rad(22.5)))
AABB_SAMPLES = np.asarray(
    [
        (-1.0, 0.0, 0.0), (-1.0, 0.7, 0.0), (-1.0, -0.7, 0.0),
        (-1.0, 0.0, 0.7), (-1.0, 0.0, -0.7), (0.0, 0.9, 0.0),
        (0.0, -0.9, 0.0), (0.0, 0.0, -0.9),
    ],
    dtype=float,
)


def _episode_dir(root: Path, episode_id: str) -> Path:
    for directory in (root / "expert_screen_rows").glob("*"):
        path = directory / "result.json"
        if path.is_file() and json.loads(path.read_text()).get("episode_id") == episode_id:
            return directory
    raise FileNotFoundError(episode_id)


def _camera_track(item, smoke_root: Path, palette):
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("MLSPACES_ASSETS_DIR", str(ROOT / "assets"))
    os.environ.pop("DISPLAY", None)
    import mujoco
    from molmo_spaces.data_generation.pipeline import cleanup_episode_resources

    directory = _episode_dir(smoke_root, item["episode_id"])
    result = json.loads((directory / "result.json").read_text())
    steps = json.loads((directory / "trajectory.json").read_text())["steps"]
    row = _row(
        index=int(item["role_index"]),
        family_id=item["family_id"],
        side=item["intrusion_side"],
        palette_document=palette,
        implementation_sha256=json.loads((smoke_root / "summary.json").read_text())["implementation_sha256"],
        seed=json.loads((smoke_root / "summary.json").read_text())["seed"],
    )
    scratch = Path(tempfile.mkdtemp(prefix="pact_v95_proxy_siting_"))
    task = sampler = None
    try:
        config = _make_config(scratch / "dummy.json", scene_xml=SCENE_XML, sampler_class=row["sampler_class"])
        sampler = config.task_sampler_config.task_sampler_class(config)
        sampler.seed_task_sampling(int(result["selected_seed"]["seed_u32"]))
        sampler.set_pact_manifest_row(row)
        task = sampler.sample_task(house_index=int(row["scene_template_house_index"]))
        task.reset()
        model, data = task.env.current_model, task.env.current_data
        cam_ids = [i for i in range(int(model.ncam)) if "_sensor_" in (model.camera(i).name or "")]
        indices = [
            i for i, step in enumerate(steps)
            if str(step.get("policy_phase")) in INBOUND_DECISION_PHASES
        ][::4]
        positions, forwards = [], []
        for index in indices:
            data.qpos[:] = np.asarray(steps[index]["qpos"], dtype=float)
            mujoco.mj_forward(model, data)
            positions.append(np.asarray(data.cam_xpos[cam_ids], dtype=float).copy())
            matrices = np.asarray(data.cam_xmat[cam_ids], dtype=float).reshape(-1, 3, 3)
            forwards.append(-matrices[:, :, 2])
        hazard = next(
            h for h in result["scene_params"]["pact_v9_hazards"]
            if h.get("role") == "inbound_vessel"
        )
        return {
            "family_id": item["family_id"],
            "side": item["intrusion_side"],
            "positions": np.stack(positions),
            "forwards": np.stack(forwards),
            "half": np.asarray(hazard["half"], dtype=float),
            "z": float(hazard["center"][2]),
            "n_frames": len(indices),
        }
    finally:
        cleanup_episode_resources(task=task, policy=None, task_sampler=sampler, preloaded_policy=None, close_task_sampler=sampler is not None)
        shutil.rmtree(scratch, ignore_errors=True)


def _detect_count(track, x: float, y: float) -> int:
    center = np.asarray([x, y, track["z"]], dtype=float)
    points = center + AABB_SAMPLES * track["half"]
    delta = points[None, None, :, :] - track["positions"][:, :, None, :]
    distance = np.linalg.norm(delta, axis=-1)
    unit = delta / np.maximum(distance[..., None], 1e-12)
    cosine = np.einsum("fcpi,fci->fcp", unit, track["forwards"])
    active = (distance <= SENSOR_RANGE_M) & (cosine > HALF_FOV_COS)
    return int(np.count_nonzero(np.any(active, axis=(1, 2))))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-root", type=Path, default=DEFAULT_SMOKE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    smoke_root = args.smoke_root.resolve()
    summary = json.loads((smoke_root / "summary.json").read_text())
    palette = load_palette(PALETTE_PATH)
    tracks = [_camera_track(item, smoke_root, palette) for item in summary["results"]]
    candidates = [
        (round(float(x), 3), round(float(y), 3))
        for x in np.arange(0.545, 0.616, 0.005)
        for y in np.arange(-0.050, 0.051, 0.005)
    ]
    families = {}
    for family in sorted({track["family_id"] for track in tracks}):
        pair = {track["side"]: track for track in tracks if track["family_id"] == family}
        ranking = []
        for x, y in candidates:
            values = {side: _detect_count(track, x, y) for side, track in pair.items()}
            low, high = min(values.values()), max(values.values())
            ratio = float(high / low) if low else None
            ranking.append(
                {
                    "center_xy_m": [x, y],
                    "proxy_detected_frames_by_side": values,
                    "minimum_detected_frames": low,
                    "maximum_to_minimum_ratio": ratio,
                }
            )
        ranking.sort(
            key=lambda item: (
                -item["minimum_detected_frames"],
                item["maximum_to_minimum_ratio"] if item["maximum_to_minimum_ratio"] is not None else float("inf"),
                abs(item["center_xy_m"][1]),
                item["center_xy_m"][0],
            )
        )
        families[family] = {"selected_proxy_candidate": ranking[0], "top_candidates": ranking[:20]}
    document = {
        "schema_version": "pact_place_v9_5_raw_vessel_proxy_siting_v1",
        "role": "proxy_shortlist_not_admission",
        "authorizes_gate": False,
        "authorizes_collection": False,
        "uses_geometry_proxy": True,
        "requires_real_tensor_followup": True,
        "source_smoke_root": str(smoke_root.relative_to(ROOT)),
        "candidate_count": len(candidates),
        "families": families,
    }
    document["document_sha256"] = sha256_payload(document)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(output)
    print(json.dumps({family: block["selected_proxy_candidate"] for family, block in families.items()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
