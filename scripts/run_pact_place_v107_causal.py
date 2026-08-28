#!/usr/bin/env python3
"""V10.7 Step 4: raw proximity causality for all six pose x side groups.

V10.6 ran one witness per side. V10.7 runs every group, because a pose that
never produced a causal witness was never shown to be sensed at all.

The numeric artifact is written before its manifest and is sufficient for
independent reaggregation: per-group per-sensor changed counts, per-frame
changed counts, and the thresholds used.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "scripts", ROOT / "submodules" / "molmospaces"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from pact_place_v107_contract import (  # noqa: E402
    CAUSAL_LINK_TOKENS,
    CAUSAL_MAX_SIDE_RATIO,
    CAUSAL_MIN_CHANGED_SENSORS,
    CAUSAL_MIN_CHANGED_VALUES,
    CAUSAL_MIN_ONSET_FRAMES,
    CAUSAL_MIN_ONSET_SECONDS,
    CAUSAL_ROOT,
    CAUSAL_WINDOW_FRAMES,
    CERT_ROOT,
    CONTRACT_VERSION_V107,
    ENVIRONMENT_VERSION,
    INTRUSION_SIDES,
    N_GROUPS,
    POLICY_TIMESTEP_MS,
    SPEC_ROOT,
    assert_no_drift,
    empty_authorization,
    recompute_payload_sha256,
    sha256_file,
    v95_row_payload,
    write_immutable_create_only,
)
from run_pact_place_v9_v0c3_causal_proximity import ABS_DELTA_FLOOR_M  # noqa: E402

POLICY_PERIOD_S = POLICY_TIMESTEP_MS / 1000.0


def _link_of(sensor_name: str) -> str:
    from run_pact_place_v9_v0c3_causal_proximity import _link_name

    return str(_link_name(sensor_name))


def render_group(witness: dict[str, Any], scene: Path, control: Path):
    """Present vs no-pendant tensors at byte-identical state, plus a repeat."""
    import mujoco
    from molmo_spaces.data_generation.pipeline import cleanup_episode_resources
    from run_pact_place_expert_screen import _make_config
    from run_pact_place_v9_v0c3_causal_proximity import _render_observation
    from run_pact_place_v7_replay_videos import apply_recorded_qpos

    steps = json.loads(
        (ROOT / witness["row_dir"] / "trajectory.json").read_text()
    )["steps"]
    closest = int(witness["frame"])
    indices = list(range(max(0, closest - CAUSAL_WINDOW_FRAMES), closest + 1))
    phases = [str(steps[i].get("policy_phase") or "") for i in indices]

    stacks: dict[str, np.ndarray] = {}
    sensor_names: list[str] = []
    for label, path in (("present", scene), ("no_pendant", control)):
        scratch = Path(tempfile.mkdtemp(prefix=f"v107_causal_{label}_"))
        task = sampler = None
        try:
            row = {
                "role_index": 0, "episode_id": "v107causal",
                "intrusion_side": witness["intrusion_side"],
                "task_seed_u32": int(witness["seed_u32"]),
                "task_seed_u64": int(witness["seed_u32"]),
                "sampler_class": "PactPlaceCorridorV93Sampler",
                **v95_row_payload(witness["family_id"], witness["intrusion_side"]),
            }
            config = _make_config(scratch / "d.json", scene_xml=path,
                                  sampler_class="PactPlaceCorridorV93Sampler")
            config.proximity_sensor_period_ms = 16.6667
            sampler = config.task_sampler_config.task_sampler_class(config)
            sampler.seed_task_sampling(int(witness["seed_u32"]))
            sampler.set_pact_manifest_row(row)
            task = sampler.sample_task(house_index=1)
            task.reset()
            model, data = task.env.mj_model, task.env.current_data
            names = list(task._proximity_camera_names)
            if len(names) != 40 or len(set(names)) != 40:
                raise RuntimeError("V10.7 requires the frozen 40-sensor suite")
            sensor_names = names
            frames, repeat = [], []
            for index in indices:
                apply_recorded_qpos(task.env, steps[index]["qpos"])
                mujoco.mj_forward(model, data)
                frames.append(_render_observation(task, names))
                if label == "present":
                    mujoco.mj_forward(model, data)
                    repeat.append(_render_observation(task, names))
            stacks[label] = np.stack(frames).astype(np.float32)
            if label == "present":
                stacks["present_repeat"] = np.stack(repeat).astype(np.float32)
        finally:
            cleanup_episode_resources(
                task=task, policy=None, task_sampler=sampler,
                preloaded_policy=None, close_task_sampler=sampler is not None,
            )
            shutil.rmtree(scratch, ignore_errors=True)
    return stacks, sensor_names, indices, phases


def analyse(stacks, sensor_names, indices, phases, witness) -> dict[str, Any]:
    shape = tuple(stacks["present"].shape[1:])
    if shape[0] != 40 or shape[-2:] != (8, 8):
        raise RuntimeError(f"unexpected tensor contract {shape}")
    repeat_delta = float(
        np.max(np.abs(stacks["present"] - stacks["present_repeat"]))
    )
    threshold = max(float(ABS_DELTA_FLOOR_M), 10.0 * repeat_delta)
    delta = np.abs(
        stacks["present"].astype(np.float64)
        - stacks["no_pendant"].astype(np.float64)
    )
    mask = delta > threshold
    per_sensor = mask.reshape(len(indices), 40, -1).sum(axis=(0, 2))
    per_frame = mask.reshape(len(indices), -1).sum(axis=1)
    responding = [
        {"sensor_index": i, "sensor_name": sensor_names[i],
         "link": _link_of(sensor_names[i]), "changed_values": int(per_sensor[i])}
        for i in range(40) if int(per_sensor[i]) > 0
    ]
    links = sorted({item["link"] for item in responding})
    first = next((i for i, c in enumerate(per_frame.tolist()) if c > 0), None)
    onset_frames = None if first is None else int(len(indices) - 1 - first)
    return {
        "group": witness["group"],
        "pose_id": witness["pose_id"],
        "intrusion_side": witness["intrusion_side"],
        "row_dir": witness["row_dir"],
        "closest_frame": int(witness["frame"]),
        "n_frames": len(indices),
        "tensor_shape": list(shape),
        "deterministic_repeat_max_abs_delta": repeat_delta,
        "deterministic": repeat_delta == 0.0,
        "threshold_m": threshold,
        "changed_values": int(mask.sum()),
        "changed_sensors": len(responding),
        "responding_sensors": responding,
        "responding_links": links,
        "has_link5_or_link6": bool(
            any(token in link for link in links for token in CAUSAL_LINK_TOKENS)
        ),
        "onset_frames_before_closest": onset_frames,
        "onset_seconds_before_closest": (
            None if onset_frames is None else onset_frames * POLICY_PERIOD_S
        ),
        "per_frame_changed_counts": per_frame.astype(int).tolist(),
        "per_sensor_changed_counts": per_sensor.astype(int).tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=ROOT / CAUSAL_ROOT)
    parser.add_argument("--certification", type=Path,
                        default=ROOT / CERT_ROOT / "certification.json")
    parser.add_argument("--specification", type=Path,
                        default=ROOT / SPEC_ROOT / "specification.json")
    args = parser.parse_args()
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("MLSPACES_ASSETS_DIR", str(ROOT / "assets"))
    os.environ.pop("DISPLAY", None)
    started = time.time()

    spec = json.loads(args.specification.resolve().read_text())
    drift = assert_no_drift(spec)
    cert_path = args.certification.resolve()
    cert = json.loads(cert_path.read_text())
    if not cert.get("certification_passed"):
        raise SystemExit("V10.7 certification did not pass")
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise SystemExit(f"refusing to overwrite {output_root}")
    output_root.mkdir(parents=True)

    control = ROOT / cert["no_pendant_scene"]["relative"]
    minima = [w for w in cert["witnesses"] if w.get("role") == "group_minimum"]
    if len(minima) != N_GROUPS:
        raise SystemExit(f"expected {N_GROUPS} group minima, got {len(minima)}")

    records: list[dict[str, Any]] = []
    for witness in sorted(minima, key=lambda w: w["group"]):
        scene = ROOT / cert["published_scenes"][witness["pose_id"]]["relative"]
        stacks, names, indices, phases = render_group(witness, scene, control)
        record = analyse(stacks, names, indices, phases, witness)
        records.append(record)
        print(json.dumps({
            "group": record["group"], "changed": record["changed_values"],
            "sensors": record["changed_sensors"],
            "links": record["responding_links"],
            "onset": record["onset_frames_before_closest"],
            "deterministic": record["deterministic"],
        }), flush=True)

    by_side: dict[str, list[int]] = {s: [] for s in INTRUSION_SIDES}
    for record in records:
        by_side[record["intrusion_side"]].append(record["changed_values"])
    side_totals = {s: sum(v) for s, v in by_side.items()}
    ratio = (
        max(side_totals.values()) / min(side_totals.values())
        if min(side_totals.values()) > 0 else float("inf")
    )

    npz_out = output_root / "causal_scores.npz"
    np.savez_compressed(
        npz_out,
        group=np.array([r["group"] for r in records], dtype=object),
        pose_id=np.array([r["pose_id"] for r in records], dtype=object),
        intrusion_side=np.array(
            [r["intrusion_side"] for r in records], dtype=object),
        row_dir=np.array([r["row_dir"] for r in records], dtype=object),
        closest_frame=np.array([r["closest_frame"] for r in records],
                               dtype=np.int64),
        n_frames=np.array([r["n_frames"] for r in records], dtype=np.int64),
        threshold_m=np.array([r["threshold_m"] for r in records],
                             dtype=np.float64),
        repeat_max_abs_delta=np.array(
            [r["deterministic_repeat_max_abs_delta"] for r in records],
            dtype=np.float64),
        changed_values=np.array([r["changed_values"] for r in records],
                                dtype=np.int64),
        changed_sensors=np.array([r["changed_sensors"] for r in records],
                                 dtype=np.int64),
        onset_frames=np.array(
            [-1 if r["onset_frames_before_closest"] is None
             else r["onset_frames_before_closest"] for r in records],
            dtype=np.int64),
        per_sensor_changed_counts=np.array(
            [r["per_sensor_changed_counts"] for r in records], dtype=np.int64),
        per_frame_changed_counts=np.array(
            [r["per_frame_changed_counts"] for r in records], dtype=np.int64),
        allow_pickle=True,
    )
    npz_sha = sha256_file(npz_out)

    checks = {
        "six_groups_evaluated": len(records) == N_GROUPS,
        "deterministic_control_repeat": all(r["deterministic"] for r in records),
        "min_changed_values_every_group": all(
            r["changed_values"] >= CAUSAL_MIN_CHANGED_VALUES for r in records),
        "min_changed_sensors_every_group": all(
            r["changed_sensors"] >= CAUSAL_MIN_CHANGED_SENSORS for r in records),
        "link5_or_link6_every_group": all(
            r["has_link5_or_link6"] for r in records),
        "onset_frames_every_group": all(
            (r["onset_frames_before_closest"] or 0) >= CAUSAL_MIN_ONSET_FRAMES
            for r in records),
        "onset_seconds_every_group": all(
            (r["onset_seconds_before_closest"] or 0.0) >= CAUSAL_MIN_ONSET_SECONDS
            for r in records),
        "side_balance": ratio <= CAUSAL_MAX_SIDE_RATIO,
    }
    document = {
        "schema_version": "pact_place_v107_causal_v1",
        "contract_version": CONTRACT_VERSION_V107,
        "environment_version": ENVIRONMENT_VERSION,
        "specification_payload_sha256": recompute_payload_sha256(
            args.specification.resolve()),
        "drift_check": drift,
        "certification_payload_sha256": recompute_payload_sha256(cert_path),
        "certification_raw_file_sha256": sha256_file(cert_path),
        "selected_key": cert["selected_key"],
        "per_group": True,
        "n_groups_evaluated": len(records),
        "groups": records,
        "changed_values_by_side": side_totals,
        "side_ratio": ratio,
        "causal_scores_npz": "causal_scores.npz",
        "causal_scores_raw_file_sha256": npz_sha,
        "npz_supports_independent_reaggregation": True,
        "npz_contents": [
            "group", "pose_id", "intrusion_side", "row_dir", "closest_frame",
            "n_frames", "threshold_m", "repeat_max_abs_delta", "changed_values",
            "changed_sensors", "onset_frames", "per_sensor_changed_counts",
            "per_frame_changed_counts",
        ],
        "thresholds": {
            "min_changed_values": CAUSAL_MIN_CHANGED_VALUES,
            "min_changed_sensors": CAUSAL_MIN_CHANGED_SENSORS,
            "required_link_tokens": list(CAUSAL_LINK_TOKENS),
            "min_onset_frames": CAUSAL_MIN_ONSET_FRAMES,
            "min_onset_seconds": CAUSAL_MIN_ONSET_SECONDS,
            "max_side_ratio": CAUSAL_MAX_SIDE_RATIO,
        },
        "checks": checks,
        "creates_episode": False, "calls_env_step": False,
        "elapsed_s": time.time() - started,
        **empty_authorization(),
        "causal_passed": all(checks.values()),
    }
    hashes = write_immutable_create_only(output_root / "causal.json", document)
    print(json.dumps({
        "causal_passed": document["causal_passed"],
        "n_groups": len(records), "checks": checks,
        "changed_values_by_side": side_totals, "side_ratio": ratio,
        "causal_scores_npz_sha256": npz_sha, **hashes,
    }, indent=2))
    return 0 if document["causal_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
