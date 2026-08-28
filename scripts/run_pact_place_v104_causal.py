#!/usr/bin/env python3
"""V10.4 Step-2: fresh two-row panel causal preservation check.

Deliberately panel-only. The static pendant is not required to produce its own
causal proximity effect; this check exists to prove the V6c panel signal is
preserved under V10.4. Replays retained qpos without stepping physics.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "scripts", ROOT / "submodules" / "molmospaces"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from pact_place_corridor_contract import sha256_file  # noqa: E402
from pact_place_v104_contract import (  # noqa: E402
    CAUSAL_MAX_SIDE_RATIO,
    CAUSAL_MIN_CHANGED_SENSORS,
    CAUSAL_MIN_CHANGED_VALUES,
    CAUSAL_PANEL_PRESERVATION_FLOOR,
    CAUSAL_ROOT,
    CONTRACT_VERSION,
    PRODUCTION_ROOT,
    REVIEW_MIN_CLEARANCE_M,
    SAMPLER_CLASS,
    empty_authorization,
    implementation_sha256,
    is_clean_success,
    write_immutable_create_only,
)
from pact_place_v104_geometry import SCENE_XML_RELATIVE_V104  # noqa: E402
from run_pact_place_expert_screen import _make_config, _result_path  # noqa: E402
from run_pact_place_v9_v0c3_causal_proximity import (  # noqa: E402
    ABS_DELTA_FLOOR_M,
    INBOUND_DECISION_PHASES,
    OUTBOUND_DECISION_PHASES,
    _causal_metrics,
    _render_observation,
)

SCENE_XML = ROOT / SCENE_XML_RELATIVE_V104
CORRIDOR_LINK_TOKENS = ("link5", "link6")


def _decision_mask(phases: list[str]) -> np.ndarray:
    window = set(INBOUND_DECISION_PHASES) | set(OUTBOUND_DECISION_PHASES)
    return np.asarray([phase in window for phase in phases], dtype=bool)


def _run_side(row: dict[str, Any], result_path: Path, output_root: Path) -> dict[str, Any]:
    import mujoco
    from molmo_spaces.data_generation.pipeline import cleanup_episode_resources
    from run_pact_place_v7_replay_videos import apply_recorded_qpos

    result = json.loads(result_path.read_text())
    steps = json.loads((result_path.parent / "trajectory.json").read_text())["steps"]
    phases = [str(step.get("policy_phase") or "") for step in steps]
    mask = _decision_mask(phases)
    if not np.any(mask):
        raise RuntimeError("no decision-window frames in the retained trajectory")
    indices = np.flatnonzero(mask)

    scratch = Path(tempfile.mkdtemp(prefix="v104_causal_"))
    task = sampler = None
    try:
        config = _make_config(
            scratch / "d.json", scene_xml=SCENE_XML, sampler_class=SAMPLER_CLASS
        )
        config.proximity_sensor_period_ms = 16.6667
        sampler = config.task_sampler_config.task_sampler_class(config)
        sampler.seed_task_sampling(int(result["selected_seed"]["seed_u32"]))
        sampler.set_pact_manifest_row(row)
        task = sampler.sample_task(house_index=int(row["scene_template_house_index"]))
        if task is None:
            raise RuntimeError("sample_task returned None")
        task.reset()
        model, data = task.env.mj_model, task.env.current_data
        sensor_names = list(task._proximity_camera_names)
        if len(sensor_names) != 40 or len(set(sensor_names)) != 40:
            raise RuntimeError("V10.4 requires the frozen 40-sensor suite")
        scene = getattr(task, "scene_params", {}) or {}
        panel_body = str(scene.get("protr_name") or "")
        if not panel_body:
            raise RuntimeError("no active intrusion panel recorded for this row")
        panel_id = int(model.body(panel_body).id)
        mocap = int(model.body_mocapid[panel_id])
        if mocap < 0:
            raise RuntimeError(f"{panel_body} is not a mocap body")
        present_pos = np.asarray(data.mocap_pos[mocap], dtype=float).copy()
        parked_pos = np.array([0.0, 1.8 if "left" in panel_body else -1.8, -2.0])

        present, repeat, parked = [], [], []
        for index in indices.tolist():
            apply_recorded_qpos(task.env, steps[index]["qpos"])
            data.mocap_pos[mocap] = present_pos
            mujoco.mj_forward(model, data)
            present.append(_render_observation(task, sensor_names))
            data.mocap_pos[mocap] = present_pos
            mujoco.mj_forward(model, data)
            repeat.append(_render_observation(task, sensor_names))
            data.mocap_pos[mocap] = parked_pos
            mujoco.mj_forward(model, data)
            parked.append(_render_observation(task, sensor_names))
        data.mocap_pos[mocap] = present_pos
        mujoco.mj_forward(model, data)

        stacked = {
            "present": np.stack(present).astype(np.float32),
            "present_repeat": np.stack(repeat).astype(np.float32),
            "panel_parked": np.stack(parked).astype(np.float32),
        }
        frame_shape = tuple(stacked["present"].shape[1:])
        if frame_shape[0] != 40 or frame_shape[-2:] != (8, 8):
            raise RuntimeError(f"unexpected tensor contract {frame_shape}")
        noise = float(np.max(np.abs(stacked["present"] - stacked["present_repeat"])))
        threshold = max(float(ABS_DELTA_FLOOR_M), 10.0 * noise)
        metrics = _causal_metrics(
            stacked["present"],
            stacked["panel_parked"],
            sensor_names,
            indices.astype(np.int32),
            [phases[i] for i in indices.tolist()],
            threshold,
        )
        links = {
            str(item["link"])
            for item in metrics.get("per_sensor") or []
            if int(item.get("changed_values") or 0) > 0
        }
        responding = sorted(
            link for link in links if any(token in link for token in CORRIDOR_LINK_TOKENS)
        )
        raw_path = output_root / "raw" / f"{row['intrusion_side']}.npz"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            raw_path,
            present=stacked["present"],
            present_repeat=stacked["present_repeat"],
            panel_parked=stacked["panel_parked"],
            trajectory_indices=indices.astype(np.int32),
            policy_phases=np.asarray([phases[i] for i in indices.tolist()], dtype="U40"),
            sensor_names=np.asarray(sensor_names, dtype="U40"),
        )
        changed = int(metrics["changed_values"])
        return {
            "intrusion_side": str(row["intrusion_side"]),
            "role_index": int(row["role_index"]),
            "episode_id": str(result["episode_id"]),
            "panel_body": panel_body,
            "tensor_contract": list(frame_shape),
            "substeps": int(frame_shape[1]),
            "n_window_frames": int(len(indices)),
            "repeat_noise_m": noise,
            "threshold_m": threshold,
            "changed_values": changed,
            "changed_sensors": int(metrics["changed_sensors"]),
            "responding_corridor_links": responding,
            "raw_path": str(raw_path.relative_to(ROOT)),
            "raw_sha256": sha256_file(raw_path),
            "meets_min_sensors": bool(
                int(metrics["changed_sensors"]) >= CAUSAL_MIN_CHANGED_SENSORS
            ),
            "meets_min_changed_values": bool(changed >= CAUSAL_MIN_CHANGED_VALUES),
            "meets_panel_preservation_floor": bool(
                changed >= CAUSAL_PANEL_PRESERVATION_FLOOR
            ),
            "meets_responding_link": bool(responding),
            "zero_repeat_noise_drift": bool(noise <= 10.0 * float(ABS_DELTA_FLOOR_M)),
            "passed": bool(
                int(metrics["changed_sensors"]) >= CAUSAL_MIN_CHANGED_SENSORS
                and changed >= CAUSAL_MIN_CHANGED_VALUES
                and changed >= CAUSAL_PANEL_PRESERVATION_FLOOR
                and responding
            ),
        }
    finally:
        cleanup_episode_resources(
            task=task, policy=None, task_sampler=sampler,
            preloaded_policy=None, close_task_sampler=sampler is not None,
        )
        shutil.rmtree(scratch, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--production-root", type=Path, default=ROOT / PRODUCTION_ROOT)
    parser.add_argument("--output-root", type=Path, default=ROOT / CAUSAL_ROOT)
    args = parser.parse_args()
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("MLSPACES_ASSETS_DIR", str(ROOT / "assets"))
    os.environ.pop("DISPLAY", None)

    production_root = args.production_root.resolve()
    manifest = json.loads((production_root / "production_manifest.json").read_text())
    rows = {int(row["role_index"]): row for row in manifest["rows"]}
    selected: dict[str, dict[str, Any]] = {}
    for entry in sorted(manifest["results"], key=lambda item: int(item["role_index"])):
        result_path = _result_path(production_root, rows[int(entry["role_index"])])
        result = json.loads(result_path.read_text())
        if not is_clean_success(result, min_clearance_m=REVIEW_MIN_CLEARANCE_M):
            continue
        side = str(rows[int(entry["role_index"])]["intrusion_side"])
        selected.setdefault(side, {"row": rows[int(entry["role_index"])], "path": result_path})

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, Any]] = []
    sides: list[dict[str, Any]] = []
    for side in ("left", "right"):
        if side not in selected:
            failures.append({"code": "missing_clean_side", "side": side})
            continue
        record = _run_side(selected[side]["row"], selected[side]["path"], output_root)
        sides.append(record)
        if not record["passed"]:
            failures.append({"code": "side_floor_failed", "side": side})
    ratio = None
    if len(sides) == 2:
        values = [max(int(item["changed_values"]), 1) for item in sides]
        ratio = float(max(values) / min(values))
        if ratio > CAUSAL_MAX_SIDE_RATIO + 1e-12:
            failures.append({"code": "side_imbalance", "ratio": ratio})
    document = {
        "schema_version": "pact_place_v104_causal_v1",
        "contract_version": CONTRACT_VERSION,
        "production_manifest_sha256": manifest.get("artifact_sha256"),
        "implementation_sha256": implementation_sha256(),
        "panel_only_admission": True,
        "pendant_causal_effect_required": False,
        "calls_env_step": False,
        "floors": {
            "min_changed_sensors": CAUSAL_MIN_CHANGED_SENSORS,
            "min_changed_values": CAUSAL_MIN_CHANGED_VALUES,
            "panel_preservation_floor": CAUSAL_PANEL_PRESERVATION_FLOOR,
            "max_side_ratio": CAUSAL_MAX_SIDE_RATIO,
            "abs_delta_floor_m": ABS_DELTA_FLOOR_M,
        },
        "sides": sides,
        "side_changed_value_ratio": ratio,
        "failures": failures,
        "causal_passed": not failures and len(sides) == 2,
        **empty_authorization(),
    }
    digest = write_immutable_create_only(output_root / "causal.json", document)
    print(json.dumps({
        "causal_passed": document["causal_passed"],
        "sides": [
            {k: item[k] for k in ("intrusion_side", "changed_values", "changed_sensors",
                                  "responding_corridor_links", "passed")}
            for item in sides
        ],
        "ratio": ratio,
        "failures": failures,
        "artifact_sha256": digest,
    }, indent=2))
    return 0 if document["causal_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
