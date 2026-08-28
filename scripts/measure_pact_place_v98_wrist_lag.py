#!/usr/bin/env python3
"""Measure TCP-to-wrist lateral lag by forward kinematics on retained qpos.

Restores recorded ``qpos``, calls ``mj_forward``, and does not step physics
or run the expert. Default wrist point is ``robot_0/fr3_link6`` body origin.
On the centred half_y=0.16 source rows that quantity is ~0.05 m, not the
design 0.208 / 0.108 (those are a different wrist point). Re-run on any new
bow geometry before treating either number as current.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "scripts", ROOT / "submodules" / "molmospaces"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pact_place_corridor_contract import sha256_payload  # noqa: E402
from pact_place_v98_pendant_contract import (  # noqa: E402
    WRIST_LAG_NEG_M,
    WRIST_LAG_POS_M,
    WRIST_LAG_PROVENANCE,
)
from run_pact_place_expert_screen import _make_config  # noqa: E402
from run_pact_place_v7_replay_videos import (  # noqa: E402
    apply_recorded_qpos,
    tcp_position_m,
)

SCENE_XML = ROOT / (
    "submodules/molmospaces/molmo_spaces/data_generation/custom_scenes/"
    "pact_place_corridor_v5.xml"
)
SOURCE_SUMMARY = ROOT / "diagnostics_output/pact_place_v95_raw_smoke/summary.json"
WRIST_BODY_CANDIDATES = (
    "robot_0/fr3_link6",
    "robot_0/fr3_link7",
    "robot_0/gripper/wrist_cam_body",
    "robot_0/link6",
    "robot_0/link7",
)
CEILING_PHASES = (
    "inbound_ceiling_fixture_approach",
    "inbound_ceiling_fixture_pass",
    "inbound_ceiling_fixture_exit",
)


def _body_y(model, data, name: str) -> float | None:
    try:
        return float(data.body(name).xpos[1])
    except (KeyError, ValueError):
        return None


def _resolve_wrist_body(model) -> str:
    names = [model.body(i).name for i in range(int(model.nbody))]
    for candidate in WRIST_BODY_CANDIDATES:
        if candidate in names:
            return candidate
    for name in names:
        lowered = name.lower()
        if "wrist_cam" in lowered or lowered.endswith("fr3_link6"):
            return name
    raise RuntimeError(f"no wrist body among {names[:40]}...")


def _prepare_task(row: dict[str, Any]):
    from molmo_spaces.data_generation.runtime_compat import assert_supported_runtime

    assert_supported_runtime(strict=True)
    scratch = Path(tempfile.mkdtemp(prefix="pact_place_v98_lag_"))
    config = _make_config(
        scratch / "dummy.json",
        scene_xml=SCENE_XML,
        sampler_class=row.get("sampler_class"),
    )
    sampler = config.task_sampler_config.task_sampler_class(config)
    sampler.seed_task_sampling(int(row["task_seed_u32"]))
    sampler.set_pact_manifest_row(row)
    task = sampler.sample_task(house_index=int(row["scene_template_house_index"]))
    if task is None:
        raise RuntimeError("sample_task returned None")
    task.reset()
    return task, sampler, scratch


def measure_row(
    env,
    task,
    wrist_body: str,
    trajectory: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    import mujoco

    model = env.current_model
    data = env.current_data
    side = str(result.get("intrusion_side") or "")
    samples = []
    for step in trajectory.get("steps") or []:
        phase = str(step.get("policy_phase") or "")
        if phase not in CEILING_PHASES:
            continue
        apply_recorded_qpos(env, step["qpos"])
        tcp = tcp_position_m(env)
        recorded = step.get("tcp_position_m")
        wrist_y = _body_y(model, data, wrist_body)
        if wrist_y is None:
            continue
        tcp_y = float(tcp[1])
        samples.append(
            {
                "step": step.get("step"),
                "policy_phase": phase,
                "tcp_y_m": tcp_y,
                "tcp_x_m": float(tcp[0]),
                "wrist_y_m": float(wrist_y),
                "lag_toward_centreline_m": float(wrist_y - tcp_y)
                if tcp_y < 0.0
                else float(tcp_y - wrist_y),
                "recorded_tcp_residual_m": None
                if recorded is None
                else float(np.linalg.norm(tcp - np.asarray(recorded, dtype=float))),
            }
        )
    if not samples:
        return {
            "role_index": result.get("role_index"),
            "intrusion_side": side,
            "status": result.get("status"),
            "n_ceiling_frames": 0,
        }
    peak = max(samples, key=lambda item: abs(float(item["tcp_y_m"])))
    return {
        "role_index": result.get("role_index"),
        "intrusion_side": side,
        "status": result.get("status"),
        "n_ceiling_frames": len(samples),
        "peak_abs_tcp_y": peak,
        "lag_at_peak_tcp_m": float(peak["lag_toward_centreline_m"]),
        "bow_sign": -1.0 if float(peak["tcp_y_m"]) < 0.0 else 1.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows-root", type=Path, required=True)
    parser.add_argument("--source-summary", type=Path, default=SOURCE_SUMMARY)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("MLSPACES_ASSETS_DIR", str(ROOT / "assets"))
    rows_root = args.rows_root.resolve()
    source = json.loads(args.source_summary.resolve().read_text())
    template = dict(source["manifest_rows"][0])
    from pact_place_v98_pendant_contract import (
        CONTRACT_VERSION,
        OFFSET_CANDIDATES,
        SAMPLER_CLASS,
        build_pendant_fixture,
    )

    spec = OFFSET_CANDIDATES["wide"]
    template["sampler_class"] = SAMPLER_CLASS
    template["pact_mounted_ceiling_fixture"] = build_pendant_fixture(
        bottom_z_m=spec["bottom_z_m"],
        half_y_m=spec["half_y_m"],
        center_y_m=spec["center_y_m"],
    )
    template["pact_v98_contract_version"] = CONTRACT_VERSION
    template["pact_v98_pendant_lateral_bow"] = True
    task, sampler, scratch = _prepare_task(template)
    try:
        env = task.env
        wrist_body = _resolve_wrist_body(env.current_model)
        row_dirs = sorted(path for path in rows_root.iterdir() if path.is_dir())
        measurements = []
        for row_dir in row_dirs:
            result_path = row_dir / "result.json"
            traj_path = row_dir / "trajectory.json"
            if not result_path.is_file() or not traj_path.is_file():
                continue
            result = json.loads(result_path.read_text())
            if result.get("status") != "complete":
                continue
            trajectory = json.loads(traj_path.read_text())
            measurements.append(
                measure_row(env, task, wrist_body, trajectory, result)
            )
    finally:
        from molmo_spaces.data_generation.pipeline import cleanup_episode_resources
        import shutil

        cleanup_episode_resources(
            task=task,
            policy=None,
            task_sampler=sampler,
            preloaded_policy=None,
            close_task_sampler=True,
        )
        shutil.rmtree(scratch, ignore_errors=True)

    by_sign: dict[str, list[float]] = defaultdict(list)
    for item in measurements:
        peak = item.get("peak_abs_tcp_y") or {}
        tcp_y = peak.get("tcp_y_m")
        lag = item.get("lag_at_peak_tcp_m")
        if tcp_y is None or lag is None:
            continue
        key = "neg" if float(tcp_y) < 0.0 else "pos"
        by_sign[key].append(float(lag))
    summary = {
        "lag_neg_m": {
            "n": len(by_sign["neg"]),
            "min": min(by_sign["neg"]) if by_sign["neg"] else None,
            "max": max(by_sign["neg"]) if by_sign["neg"] else None,
            "design": WRIST_LAG_NEG_M,
        },
        "lag_pos_m": {
            "n": len(by_sign["pos"]),
            "min": min(by_sign["pos"]) if by_sign["pos"] else None,
            "max": max(by_sign["pos"]) if by_sign["pos"] else None,
            "design": WRIST_LAG_POS_M,
        },
    }
    document = {
        "schema_version": "pact_place_v9_8_wrist_lag_v1",
        "role": "forward_kinematics_instrument_not_a_gate",
        "authorizes_collection": False,
        "wrist_body": wrist_body,
        "rows_root": str(rows_root.relative_to(ROOT)),
        "provenance_template": WRIST_LAG_PROVENANCE,
        "summary": summary,
        "rows": measurements,
        "document_sha256": None,
    }
    document["document_sha256"] = sha256_payload(document)
    output = args.output or (rows_root.parent / "wrist_lag.json")
    output = output.resolve()
    output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "wrist_body": wrist_body,
                "summary": summary,
                "path": str(output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
