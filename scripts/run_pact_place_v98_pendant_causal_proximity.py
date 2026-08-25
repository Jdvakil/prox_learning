#!/usr/bin/env python3
"""V9.8 frozen-qpos causal smoke and raw admission runner.

The renderer is imported unchanged from V9.3.  Only the parked worlds and the
role set differ: ``panel`` and ``ceiling_pendant``.  The artifact never
authorizes collection; S2b is an admission record consumed by the human gate.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "scripts", ROOT / "submodules" / "molmospaces"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pact_place_corridor_contract import sha256_file, sha256_payload  # noqa: E402
from pact_place_v98_pendant_contract import (  # noqa: E402
    ADMISSION_FLOOR,
    SAMPLER_CLASS,
)
from run_pact_place_expert_screen import _make_config  # noqa: E402
from run_pact_place_v9_v0c3_causal_proximity import (  # noqa: E402
    ABS_DELTA_FLOOR_M,
    INBOUND_DECISION_PHASES,
    OUTBOUND_DECISION_PHASES,
    PARK_Z_M,
    SCENE_XML,
    _causal_metrics,
    _find_episode_dir,
    _free_joint_qpos_address,
    _render_observation,
)
from run_pact_place_v98_pendant_preview import build_row  # noqa: E402

DEFAULT_SOURCE_ROOT = ROOT / "diagnostics_output/pact_place_v95_raw_smoke"
DEFAULT_CONFIGURATION = ROOT / "configs/pact_place_corridor_v98.json"
DEFAULT_OUTPUT_ROOT = ROOT / "diagnostics_output/pact_place_v98_pendant_causal"
ROLE_WINDOWS = {
    "panel": OUTBOUND_DECISION_PHASES,
    "ceiling_pendant": INBOUND_DECISION_PHASES | OUTBOUND_DECISION_PHASES,
}
V95_CLUTTER_SLOT_COUNT = 8
FREE_JOINT_QPOS_DIM = 7


def _summary_and_rows(
    source_root: Path, configuration: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    summary_path = source_root / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        items = list(summary.get("results") or [])
        source_rows = {
            (str(row.get("layout_family_id")), str(row.get("intrusion_side"))): row
            for row in list(summary.get("manifest_rows") or [])
        }
    else:
        items = []
        source_rows = {}
        for result_path in sorted((source_root / "expert_screen_rows").glob("*/result.json")):
            item = json.loads(result_path.read_text())
            item["result_path"] = str(result_path)
            items.append(item)
    rows = {
        int(row["role_index"]): row
        for row in list(configuration.get("expert_screen_rows") or [])
    }
    return items, rows, source_rows


def _source_jobs(source_root: Path, configuration: dict[str, Any], require_clean: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    items, configured_rows, source_rows = _summary_and_rows(source_root, configuration)
    configured_families = {
        str(row.get("layout_family_id"))
        for row in configured_rows.values()
        if row.get("layout_family_id")
    }
    jobs = []
    skipped = []
    seen: set[tuple[str, str]] = set()
    for item in items:
        family_id = str(item.get("family_id") or item.get("layout_family_id") or "")
        side = str(item.get("intrusion_side") or "")
        if not family_id or side not in {"left", "right"}:
            row = configured_rows.get(int(item.get("role_index", -1)))
            family_id = str((row or {}).get("layout_family_id") or family_id)
        key = (family_id, side)
        if configured_families and family_id not in configured_families:
            continue
        if key in seen or not family_id:
            continue
        source_clean = bool(item.get("clean_success"))
        if require_clean and not source_clean:
            skipped.append({"family_id": family_id, "intrusion_side": side, "reason": "source_physics_not_clean"})
            continue
        row = source_rows.get(key) or configured_rows.get(int(item.get("role_index", -1)))
        if row is None:
            row = next(
                (
                    candidate
                    for candidate in configured_rows.values()
                    if str(candidate.get("layout_family_id")) == family_id
                    and str(candidate.get("intrusion_side")) == side
                ),
                None,
            )
        if row is None:
            raise RuntimeError(f"no V9.8 configured row for {key}")
        result_path = Path(item.get("result_path") or "")
        if not result_path.is_file():
            result_path = _find_episode_dir(source_root, str(item["episode_id"])) / "result.json"
        result = json.loads(result_path.read_text())
        trajectory_path = Path(result.get("trajectory_path") or "")
        if not trajectory_path.is_file():
            trajectory_path = result_path.parent / "trajectory.json"
        if not trajectory_path.is_file():
            raise RuntimeError(f"missing trajectory for source episode {item['episode_id']}")
        panel = (result.get("scene_params") or {}).get("protr_center")
        if panel is None:
            raise RuntimeError(f"source episode {item['episode_id']} has no active panel pose")
        row = dict(row)
        configured_row = next(
            (
                candidate
                for candidate in configured_rows.values()
                if str(candidate.get("layout_family_id")) == family_id
                and str(candidate.get("intrusion_side")) == side
            ),
            None,
        )
        if configured_row is None:
            raise RuntimeError(f"no configured pendant fixture for {key}")
        row["sampler_class"] = SAMPLER_CLASS
        row["pact_mounted_ceiling_fixture"] = dict(
            configured_row["pact_mounted_ceiling_fixture"]
        )
        row["pact_v98_contract_version"] = "pact_place_v9_8_pendant_v1"
        row.pop("row_sha256", None)
        row["row_sha256"] = sha256_payload(row)
        row["scene_template_house_index"] = int(row.get("scene_template_house_index", 1))
        jobs.append(
            {
                "family_id": family_id,
                "intrusion_side": side,
                "source_episode_id": str(item["episode_id"]),
                "row": row,
                "result_path": str(result_path),
                "trajectory_path": str(trajectory_path),
                "expected_panel_center_m": [float(value) for value in panel],
                "source_physics_clean": source_clean,
            }
        )
        seen.add(key)
    jobs.sort(key=lambda job: (job["family_id"], job["intrusion_side"]))
    return jobs, skipped


def _clutter_qpos_start(model) -> int:
    starts = [
        int(model.jnt_qposadr[j])
        for j in range(model.njnt)
        if model.body(int(model.jnt_bodyid[j])).name.startswith("pact_clutter_")
    ]
    others = [
        int(model.jnt_qposadr[j])
        for j in range(model.njnt)
        if not model.body(int(model.jnt_bodyid[j])).name.startswith("pact_clutter_")
    ]
    if not starts or min(starts) <= max(others):
        raise RuntimeError("V9.8 clutter free joints are not a contiguous qpos suffix")
    return min(starts)


def _run_variant(job: dict[str, Any], output_root: Path) -> dict[str, Any]:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("MLSPACES_ASSETS_DIR", str(ROOT / "assets"))
    os.environ.pop("DISPLAY", None)
    import mujoco
    from molmo_spaces.data_generation.pipeline import cleanup_episode_resources

    result = json.loads(Path(job["result_path"]).read_text())
    steps = list(json.loads(Path(job["trajectory_path"]).read_text())["steps"])
    phases = [str(step.get("policy_phase")) for step in steps]
    scratch = Path(tempfile.mkdtemp(prefix="pact_v98_causal_"))
    task = sampler = None
    try:
        config = _make_config(scratch / "dummy.json", scene_xml=SCENE_XML, sampler_class=SAMPLER_CLASS)
        config.proximity_sensor_period_ms = 16.6667
        sampler = config.task_sampler_config.task_sampler_class(config)
        sampler.seed_task_sampling(int(result["selected_seed"]["seed_u32"]))
        sampler.set_pact_manifest_row(job["row"])
        task = sampler.sample_task(house_index=int(job["row"]["scene_template_house_index"]))
        if task is None:
            raise RuntimeError("V9.8 sample_task returned None")
        task.reset()
        model, data = task.env.mj_model, task.env.current_data
        sensor_names = list(task._proximity_camera_names)
        if len(sensor_names) != 40 or len(set(sensor_names)) != 40:
            raise RuntimeError("V9.8 requires the frozen 40-sensor suite")
        clutter_start = _clutter_qpos_start(model)
        source_nq = len(steps[0]["qpos"])
        if clutter_start != source_nq - V95_CLUTTER_SLOT_COUNT * FREE_JOINT_QPOS_DIM:
            raise RuntimeError(f"shared qpos block changed: {clutter_start} vs {source_nq}")
        clutter_reset = np.asarray(data.qpos[clutter_start:], dtype=float).copy()
        panel_id = int(np.asarray(model.body(f"pact_intrusion_{job['intrusion_side']}").mocapid).reshape(-1)[0])
        pendant_id = int(np.asarray(model.body("pact_clutter_mount_ceiling").mocapid).reshape(-1)[0])
        if panel_id < 0 or pendant_id < 0:
            raise RuntimeError("V9.8 panel and pendant must be mocap-controlled")
        panel_position = np.asarray(data.mocap_pos[panel_id], dtype=float).copy()
        pendant_position = np.asarray(data.mocap_pos[pendant_id], dtype=float).copy()
        if not np.allclose(panel_position, job["expected_panel_center_m"], atol=1e-9):
            raise RuntimeError("source panel pose does not match the frozen episode")

        def set_world(step_index: int) -> None:
            qpos = np.asarray(steps[step_index]["qpos"], dtype=float)
            if qpos.shape != (int(model.nq),):
                raise RuntimeError(f"qpos shape mismatch: {qpos.shape} vs {model.nq}")
            data.qpos[:clutter_start] = qpos[:clutter_start]
            data.qpos[clutter_start:] = clutter_reset
            data.mocap_pos[panel_id] = panel_position
            data.mocap_pos[pendant_id] = pendant_position

        worlds = {"present": None, "panel_parked": "panel", "ceiling_pendant_parked": "ceiling_pendant"}
        frames = {name: [] for name in worlds}
        indices = list(range(len(steps)))
        for step_index in indices:
            for world, parked in worlds.items():
                set_world(step_index)
                if parked == "panel":
                    data.mocap_pos[panel_id, 2] = PARK_Z_M
                elif parked == "ceiling_pendant":
                    data.mocap_pos[pendant_id, 2] = PARK_Z_M
                mujoco.mj_forward(model, data)
                frames[world].append(_render_observation(task, sensor_names))
        stacked = {name: np.stack(value).astype(np.float32) for name, value in frames.items()}
        repeat_deltas = []
        for local in sorted({0, len(indices) // 2, len(indices) - 1}):
            set_world(indices[local])
            mujoco.mj_forward(model, data)
            repeat_deltas.append(float(np.max(np.abs(_render_observation(task, sensor_names) - stacked["present"][local]))))
        noise_floor = max(repeat_deltas)
        threshold = max(ABS_DELTA_FLOOR_M, noise_floor * 10.0)
        output_path = output_root / "raw" / f"{job['family_id']}_{job['intrusion_side']}.npz"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            present=stacked["present"],
            panel_parked=stacked["panel_parked"],
            ceiling_pendant_parked=stacked["ceiling_pendant_parked"],
            trajectory_indices=np.asarray(indices, dtype=np.int32),
            policy_phases=np.asarray(phases, dtype="U40"),
            sensor_names=np.asarray(sensor_names, dtype="U40"),
        )
        masks = {
            role: np.asarray([phase in window for phase in phases], dtype=bool)
            for role, window in ROLE_WINDOWS.items()
        }
        effects = {}
        for role, mask in masks.items():
            parked_world = "panel_parked" if role == "panel" else "ceiling_pendant_parked"
            effects[role] = _causal_metrics(
                stacked["present"][mask], stacked[parked_world][mask], sensor_names,
                np.asarray(indices, dtype=np.int32)[mask], [phase for phase, keep in zip(phases, mask) if keep], threshold,
            )
        return {
            "family_id": job["family_id"],
            "intrusion_side": job["intrusion_side"],
            "source_episode_id": job["source_episode_id"],
            "source_physics_clean": bool(job["source_physics_clean"]),
            "row_sha256": job["row"]["row_sha256"],
            "layout_id": job["row"]["layout_id"],
            "raw_tensor_path": str(output_path.relative_to(ROOT)),
            "raw_tensor_sha256": sha256_file(output_path),
            "tensor_shape_per_world": list(stacked["present"].shape),
            "sensor_count": len(sensor_names),
            "substeps": int(stacked["present"].shape[2]),
            "n_replayed_steps": len(indices),
            "baseline_repeat_max_abs_delta_m": noise_floor,
            "causal_threshold_m": threshold,
            "counterfactual_worlds": list(worlds),
            "effects": effects,
        }
    finally:
        cleanup_episode_resources(task=task, policy=None, task_sampler=sampler, preloaded_policy=None, close_task_sampler=sampler is not None)
        shutil.rmtree(scratch, ignore_errors=True)


def _verdict(metrics: dict[str, Any]) -> dict[str, Any]:
    links = {str(item["link"]) for item in metrics["per_sensor"] if int(item["changed_values"]) > 0}
    return {
        "changed_values": int(metrics["changed_values"]),
        "changed_sensors": int(metrics["changed_sensors"]),
        "responding_links": sorted(links),
        "meets_min_sensors": int(metrics["changed_sensors"]) >= ADMISSION_FLOOR["min_distinct_changed_sensors_per_role_side"],
        "meets_min_changed_values": int(metrics["changed_values"]) >= ADMISSION_FLOOR["min_changed_values_per_role_side"],
        "meets_corridor_link": bool(links & set(ADMISSION_FLOOR["required_responding_links_any_of"])),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--configuration", type=Path, default=DEFAULT_CONFIGURATION)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--admission", action="store_true", help="record S2b admission role; still never authorizes collection")
    parser.add_argument("--allow-dirty-source", dest="require_clean", action="store_false")
    args = parser.parse_args()
    configuration = json.loads(args.configuration.resolve().read_text())
    source_root = args.smoke_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    jobs, skipped = _source_jobs(source_root, configuration, args.require_clean)
    if len(jobs) != 6:
        raise SystemExit(f"V9.8 requires six clean variants; selected {len(jobs)}, skipped {skipped}")
    config_document = {
        "schema_version": "pact_place_v9_8_causal_config_v1",
        "role": "raw_causal_admission" if args.admission else "occlusion_aware_smoke_not_admission",
        "authorizes_gate": False,
        "authorizes_collection": False,
        "written_before_run": True,
        "admission_floor": ADMISSION_FLOOR,
        "rendering_path": "run_pact_place_v9_v0c3_causal_proximity._render_observation (unchanged)",
        "production_tensor_contract": [40, 4, 8, 8],
        "counterfactual_worlds": ["present", "panel_parked", "ceiling_pendant_parked"],
        "source_root": str(source_root.relative_to(ROOT)),
        "configuration_sha256": sha256_file(args.configuration.resolve()),
        "variants": [{"family_id": job["family_id"], "intrusion_side": job["intrusion_side"]} for job in jobs],
        "skipped": skipped,
    }
    config_document["config_sha256"] = sha256_payload(config_document)
    (output_root / "config.json").write_text(json.dumps(config_document, indent=2, sort_keys=True) + "\n")
    if args.workers == 1:
        results = [_run_variant(job, output_root) for job in jobs]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            results = list(executor.map(_run_variant, jobs, [output_root] * len(jobs)))
    results.sort(key=lambda item: (item["family_id"], item["intrusion_side"]))
    verdicts = [
        {
            "family_id": result["family_id"],
            "intrusion_side": result["intrusion_side"],
            "roles": {role: _verdict(result["effects"][role]) for role in ROLE_WINDOWS},
        }
        for result in results
    ]
    balance = []
    for family in sorted({result["family_id"] for result in results}):
        pair = {result["intrusion_side"]: result for result in results if result["family_id"] == family}
        for role in ROLE_WINDOWS:
            values = {side: int(item["effects"][role]["changed_values"]) for side, item in pair.items()}
            low, high = min(values.values()), max(values.values())
            balance.append({
                "family_id": family,
                "role": role,
                "changed_values_by_side": values,
                "max_to_min_ratio": float(high / low) if low else None,
                "passed": bool(len(values) == 2 and low > 0 and high / low <= ADMISSION_FLOOR["max_paired_changed_value_ratio"]),
            })
    role_pass = {
        role: all(
            entry["roles"][role]["meets_min_sensors"]
            and entry["roles"][role]["meets_min_changed_values"]
            and entry["roles"][role]["meets_corridor_link"]
            for entry in verdicts
        ) and all(item["passed"] for item in balance if item["role"] == role)
        for role in ROLE_WINDOWS
    }
    passed = bool(all(job["source_physics_clean"] for job in jobs) and all(role_pass.values()))
    document = {
        "schema_version": "pact_place_v9_8_causal_proximity_v1",
        "role": config_document["role"],
        "passed": passed,
        "authorizes_gate": False,
        "authorizes_collection": False,
        "all_sources_physics_clean": all(job["source_physics_clean"] for job in jobs),
        "uses_real_40_sensor_observation": True,
        "uses_geometry_proxy_for_admission": False,
        "rendering_path": config_document["rendering_path"],
        "production_tensor_contract": [40, 4, 8, 8],
        "admission_floor": ADMISSION_FLOOR,
        "config_sha256": config_document["config_sha256"],
        "role_pass": role_pass,
        "paired_side_balance": balance,
        "verdicts": verdicts,
        "variants": results,
    }
    document["document_sha256"] = sha256_payload(document)
    path = output_root / "validation.json"
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": passed, "role_pass": role_pass, "path": str(path)}, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
