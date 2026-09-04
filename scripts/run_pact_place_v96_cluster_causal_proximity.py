#!/usr/bin/env python3
"""W3: raw counterfactual confirmation for the V9.6 clustered hazards.

The rendering path is the V9.5 validator's, imported unchanged: the same
``_render_observation`` on the real ``[40, 4, 8, 8]`` production tensor, the same
present-versus-parked counterfactual at frozen qpos, the same per-value
threshold ``max(ABS_DELTA_FLOOR_M, noise_floor * 10)``.  Two things differ, and
only two:

1. a hazard is now a **cluster**, so parking it parks every member of that leg;
2. the **aggregate pass rule** is replaced.  V9.5's rule was "any nonzero pixel",
   which a placement can satisfy with a signal no policy could use.  The V9.6
   floor is written to ``config.json`` before the run and is not touched after.

The arm is the frozen V9.5 trajectory.  The V9.6 scene installs twelve prop
slots instead of eight, so ``nq`` differs; the replay copies the arm, gripper and
target block verbatim and imposes the V9.6 clutter poses, after asserting that
the clutter free joints are contiguous and last in both models.

Nothing here authorizes a gate, collection, or V1b.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MOLMO = ROOT / "submodules" / "molmospaces"
for _path in (ROOT / "scripts", MOLMO):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import pact_place_v96_cluster_contract as v96  # noqa: E402
import pact_place_v97_hazard_contract as v97  # noqa: E402
import pact_skin_resolvability as psr  # noqa: E402
from pact_place_corridor_contract import sha256_file  # noqa: E402
from pact_place_v9_contract import sha256_payload  # noqa: E402
from run_pact_place_v9_v0c3_causal_proximity import (  # noqa: E402
    ABS_DELTA_FLOOR_M,
    INBOUND_DECISION_PHASES,
    MAX_PAIRED_CHANGED_VALUE_RATIO,
    OUTBOUND_DECISION_PHASES,
    SCENE_XML,
    _causal_metrics,
    _find_episode_dir,
    _free_joint_qpos_address,
    _render_observation,
)
from run_pact_place_v9_w1_resolvability import HazardSource  # noqa: E402
from run_pact_place_v9_w2_cluster_siting import leg_masks  # noqa: E402

DEFAULT_SMOKE_ROOT = ROOT / "diagnostics_output" / "pact_place_v95_raw_smoke"
DEFAULT_SITING = (
    ROOT / "diagnostics_output" / "pact_place_v9_w2_cluster_siting" / "siting.json"
)
DEFAULT_OUTPUT_ROOT = ROOT / "diagnostics_output" / "pact_place_v96_w3_raw_confirmation"
PARK_Z_M = -2.0
V95_CLUTTER_SLOT_COUNT = 8
FREE_JOINT_QPOS_DIM = 7
ROLE_DECISION_WINDOW = {
    "inbound_cluster": INBOUND_DECISION_PHASES,
    "outbound_cluster": OUTBOUND_DECISION_PHASES,
    "panel": OUTBOUND_DECISION_PHASES,
}
CORRIDOR_LINKS = ("link5_front", "link5_back", "link6")

# ---------------------------------------------------------------------------
# The admission floor.  Written to config.json before the run; never lowered.
# ---------------------------------------------------------------------------
ADMISSION_FLOOR = {
    "window": "v9_5_decision_phase_window",
    "min_distinct_changed_sensors_per_role_side": 3,
    "min_changed_values_per_role_side": 448,
    "min_changed_values_provenance": (
        "the outbound vessel's measured left-side result on link3_sensor_2 in "
        "diagnostics_output/pact_place_v95_v0c5_raw_prerequisite/validation.json"
    ),
    "max_paired_changed_value_ratio": MAX_PAIRED_CHANGED_VALUE_RATIO,
    "required_responding_links_any_of": list(CORRIDOR_LINKS),
    "per_value_threshold": "max(ABS_DELTA_FLOOR_M, baseline_repeat_max_abs_delta_m * 10)",
    "abs_delta_floor_m": ABS_DELTA_FLOOR_M,
}


def _rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(Path(path).resolve())


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
    if not starts:
        raise RuntimeError("V9.6 model exposes no clutter free joints")
    if min(starts) <= max(others):
        raise RuntimeError("clutter free joints are not last in the V9.6 qpos vector")
    return min(starts)


def _configuration_sampler_class(configuration: dict[str, Any]) -> str:
    """V9.7 configurations name their leg UIDs; V9.6 ones name a fixed recipe."""
    if configuration.get("inbound_uids") and configuration.get("outbound_uids"):
        return str(configuration.get("sampler_class") or v97.SAMPLER_CLASS)
    return str(configuration.get("sampler_class") or v96.SAMPLER_CLASS)


def _build_row(v95_row: dict[str, Any], configuration: dict[str, Any]) -> dict[str, Any]:
    if configuration.get("inbound_uids") and configuration.get("outbound_uids"):
        palette = v97.build_hazard_palette(
            configuration["inbound_uids"], configuration["outbound_uids"]
        )
        build_layout = v97.build_hazard_layout
    else:
        palette = v96.build_cluster_palette(configuration["recipe_id"])
        build_layout = v96.build_cluster_layout
    layout = build_layout(
        palette,
        family_id=str(v95_row["layout_family_id"]),
        intrusion_side=str(v95_row["intrusion_side"]),
        inbound=configuration["inbound_cluster"],
        outbound=configuration["outbound_cluster"],
    )
    row = {
        **{
            key: value
            for key, value in v95_row.items()
            if key
            not in {
                "pact_clutter_palette",
                "pact_clutter_layout",
                "layout_id",
                "row_sha256",
                "sampler_class",
                "clutter_x_jitter_m",
                "clutter_y_jitter_m",
            }
        },
        "sampler_class": _configuration_sampler_class(configuration),
        "pact_clutter_palette": list(palette["palette"]),
        "pact_clutter_layout": layout,
        "layout_id": layout["layout_id"],
        # V9.6 sites the clusters explicitly; per-slot jitter belongs to the
        # production manifest, not to a frozen-qpos counterfactual.
        "clutter_x_jitter_m": {},
        "clutter_y_jitter_m": {},
    }
    row["row_sha256"] = sha256_payload(row)
    return row


def _run_variant(job: dict[str, Any]) -> dict[str, Any]:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("MLSPACES_ASSETS_DIR", str(ROOT / "assets"))
    os.environ.pop("DISPLAY", None)

    import mujoco
    from molmo_spaces.data_generation.pipeline import cleanup_episode_resources
    from run_pact_place_expert_screen import _make_config

    result = json.loads(Path(job["result_path"]).read_text())
    steps = list(json.loads(Path(job["trajectory_path"]).read_text())["steps"])
    indices = list(range(len(steps)))
    phases = [str(steps[index].get("policy_phase")) for index in indices]
    row = job["row"]

    scratch = Path(tempfile.mkdtemp(prefix="pact_v96_w3_"))
    task = sampler = None
    try:
        config = _make_config(
            scratch / "dummy.json",
            scene_xml=SCENE_XML,
            sampler_class=str(row["sampler_class"]),
        )
        config.proximity_sensor_period_ms = 16.6667
        sampler = config.task_sampler_config.task_sampler_class(config)
        sampler.seed_task_sampling(int(result["selected_seed"]["seed_u32"]))
        sampler.set_pact_manifest_row(row)
        task = sampler.sample_task(house_index=int(row["scene_template_house_index"]))
        if task is None:
            raise RuntimeError("V9.6 sample_task returned None")
        task.reset()

        model = task.env.mj_model
        data = task.env.current_data
        sensor_names = list(task._proximity_camera_names)
        if len(sensor_names) != 40 or len(set(sensor_names)) != 40:
            raise RuntimeError(f"expected 40 unique proximity cameras: {sensor_names}")

        clutter_start = _clutter_qpos_start(model)
        source_nq = len(steps[0]["qpos"])
        if clutter_start != source_nq - V95_CLUTTER_SLOT_COUNT * FREE_JOINT_QPOS_DIM:
            raise RuntimeError(
                "V9.5 and V9.6 disagree on the shared arm/target qpos block: "
                f"clutter starts at {clutter_start}, source nq is {source_nq}"
            )
        clutter_reset = np.asarray(data.qpos[clutter_start:], dtype=float).copy()

        active_panel = f"pact_intrusion_{job['intrusion_side']}"
        panel_mocap_id = int(np.asarray(model.body(active_panel).mocapid).reshape(-1)[0])
        if panel_mocap_id < 0:
            raise RuntimeError(f"active panel is not mocap-controlled: {active_panel}")
        panel_position = np.asarray(data.mocap_pos[panel_mocap_id], dtype=float).copy()
        expected_panel = np.asarray(job["expected_panel_center_m"], dtype=float)
        if not np.allclose(panel_position, expected_panel, atol=1e-9):
            raise RuntimeError(
                "V9.6 panel pose differs from the frozen V9.5 episode: "
                f"{panel_position.tolist()} vs {expected_panel.tolist()}"
            )

        palette_by_slot = {str(item["slot"]): item for item in row["pact_clutter_palette"]}
        cluster_qadrs: dict[str, list[int]] = {}
        cluster_bodies: dict[str, list[str]] = {}
        for role in v96.CLUSTER_ROLES:
            slots = [
                str(item["slot"])
                for item in row["pact_clutter_palette"]
                if str(item["role"]) == role
            ]
            bodies = [f"pact_clutter_{slot}/{palette_by_slot[slot]['uid']}" for slot in slots]
            cluster_bodies[role] = bodies
            cluster_qadrs[role] = [
                _free_joint_qpos_address(model, body) for body in bodies
            ]

        # The plan requires the silhouette to be verified from posed geometry,
        # not from the palette's nominal dimensions.  Measure it from the
        # renderable meshes the sensors actually see, at the reset pose.
        posed_geometry = {}
        for role, bodies in cluster_bodies.items():
            theta = math.radians(
                float(row["pact_clutter_layout"][role]["theta_deg"])
            )
            direction = np.array([math.cos(theta), math.sin(theta)])
            intervals = []
            aabbs = []
            for body in bodies:
                low, high = HazardSource(model, role, body).pose(data).aabb
                aabbs.append(([float(v) for v in low], [float(v) for v in high]))
                corners = np.array(
                    [[low[0], low[1]], [low[0], high[1]], [high[0], low[1]], [high[0], high[1]]]
                )
                projected = corners @ direction
                intervals.append((float(projected.min()), float(projected.max())))
            intervals.sort()
            span = intervals[-1][1] - intervals[0][0]
            gaps = [
                max(0.0, intervals[i + 1][0] - intervals[i][1])
                for i in range(len(intervals) - 1)
            ]
            lows = np.array([a[0] for a in aabbs])
            highs = np.array([a[1] for a in aabbs])
            posed_geometry[role] = {
                "measured_from": "renderable_mesh_world_aabb_at_reset",
                "member_bodies": list(bodies),
                "member_world_aabbs_m": aabbs,
                "realized_span_along_line_m": float(span),
                "realized_gaps_m": [float(value) for value in gaps],
                "realized_max_gap_m": float(max(gaps)) if gaps else 0.0,
                "union_extent_m": [float(v) for v in (highs.max(axis=0) - lows.min(axis=0))],
                "meets_span_floor": bool(span >= v96.MIN_CLUSTER_SPAN_M - 1e-9),
                "meets_gap_ceiling": bool(
                    (max(gaps) if gaps else 0.0) <= v96.MAX_CLUSTER_GAP_M + 1e-9
                ),
            }

        def set_world(step_index: int) -> None:
            qpos = np.asarray(steps[step_index]["qpos"], dtype=float)
            data.qpos[:clutter_start] = qpos[:clutter_start]
            data.qpos[clutter_start:] = clutter_reset
            data.mocap_pos[panel_mocap_id] = panel_position

        worlds = {
            "present": None,
            "panel_parked": "panel",
            "inbound_cluster_parked": "inbound_cluster",
            "outbound_cluster_parked": "outbound_cluster",
        }
        frames: dict[str, list[np.ndarray]] = {name: [] for name in worlds}
        for step_index in indices:
            for world, parked in worlds.items():
                set_world(step_index)
                if parked == "panel":
                    data.mocap_pos[panel_mocap_id, 2] = PARK_Z_M
                elif parked in cluster_qadrs:
                    for qadr in cluster_qadrs[parked]:
                        data.qpos[qadr + 2] = PARK_Z_M
                mujoco.mj_forward(model, data)
                frames[world].append(_render_observation(task, sensor_names))

        stacked = {name: np.stack(values).astype(np.float32) for name, values in frames.items()}
        trajectory_indices = np.asarray(indices, dtype=np.int32)

        repeat_deltas = []
        for local in sorted({0, len(indices) // 2, len(indices) - 1}):
            set_world(indices[local])
            mujoco.mj_forward(model, data)
            repeated = _render_observation(task, sensor_names)
            repeat_deltas.append(float(np.max(np.abs(repeated - stacked["present"][local]))))
        noise_floor_m = max(repeat_deltas)
        threshold_m = max(ABS_DELTA_FLOOR_M, noise_floor_m * 10.0)

        output_path = Path(job["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            **stacked,
            trajectory_indices=trajectory_indices,
            policy_phases=np.asarray(phases, dtype="U40"),
            sensor_names=np.asarray(sensor_names, dtype="U40"),
        )

        legs = leg_masks(phases)
        windows = {
            "decision": {
                role: np.asarray(
                    [phase in ROLE_DECISION_WINDOW[role] for phase in phases], dtype=bool
                )
                for role in ROLE_DECISION_WINDOW
            },
            "leg": {
                "panel": legs["outbound_cluster"],
                "inbound_cluster": legs["inbound_cluster"],
                "outbound_cluster": legs["outbound_cluster"],
            },
        }
        parked_for_role = {
            "panel": "panel_parked",
            "inbound_cluster": "inbound_cluster_parked",
            "outbound_cluster": "outbound_cluster_parked",
        }
        effects: dict[str, dict[str, Any]] = {}
        for window_name, masks in windows.items():
            for role, mask in masks.items():
                if not mask.any():
                    raise ValueError(f"no frames in the {role} {window_name} window")
                effects.setdefault(window_name, {})[role] = _causal_metrics(
                    stacked["present"][mask],
                    stacked[parked_for_role[role]][mask],
                    sensor_names,
                    trajectory_indices[mask],
                    [phase for phase, keep in zip(phases, mask) if keep],
                    threshold_m,
                )

        return {
            "family_id": job["family_id"],
            "intrusion_side": job["intrusion_side"],
            "source_episode_id": job["source_episode_id"],
            "source_physics_clean": bool(job["source_physics_clean"]),
            "row_sha256": row["row_sha256"],
            "layout_id": row["layout_id"],
            "raw_tensor_path": _rel(output_path),
            "raw_tensor_sha256": sha256_file(output_path),
            "tensor_shape_per_world": list(stacked["present"].shape),
            "sensor_count": len(sensor_names),
            "substeps": int(stacked["present"].shape[2]),
            "n_replayed_steps": len(indices),
            "baseline_repeat_max_abs_delta_m": noise_floor_m,
            "causal_threshold_m": threshold_m,
            "active_panel": active_panel,
            "cluster_bodies": cluster_bodies,
            "declared_cluster_geometry": job["declared_cluster_geometry"],
            "posed_cluster_geometry": posed_geometry,
            "effects": effects,
        }
    finally:
        cleanup_episode_resources(
            task=task, policy=None, task_sampler=sampler,
            preloaded_policy=None, close_task_sampler=sampler is not None,
        )
        shutil.rmtree(scratch, ignore_errors=True)


def _role_side_verdict(metrics: dict[str, Any]) -> dict[str, Any]:
    responding = [
        item for item in metrics["per_sensor"] if int(item["changed_values"]) > 0
    ]
    links = {str(item["link"]) for item in responding}
    return {
        "changed_values": int(metrics["changed_values"]),
        "changed_sensors": int(metrics["changed_sensors"]),
        "responding_links": sorted(links),
        "meets_min_sensors": bool(
            int(metrics["changed_sensors"])
            >= ADMISSION_FLOOR["min_distinct_changed_sensors_per_role_side"]
        ),
        "meets_min_changed_values": bool(
            int(metrics["changed_values"])
            >= ADMISSION_FLOOR["min_changed_values_per_role_side"]
        ),
        "meets_corridor_link": bool(links & set(CORRIDOR_LINKS)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-root", type=Path, default=DEFAULT_SMOKE_ROOT)
    parser.add_argument("--configuration", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--family", action="append")
    parser.add_argument("--side", choices=("left", "right"), action="append")
    parser.add_argument(
        "--allow-dirty-source", dest="require_clean_source", action="store_false",
        help="render variants whose source episode is not collision-free "
             "(never valid for admission; V9.5's false pass came from one)",
    )
    parser.set_defaults(require_clean_source=True)
    args = parser.parse_args()
    if not 1 <= args.workers <= 6:
        raise SystemExit("workers must be in [1, 6]")

    configuration = json.loads(args.configuration.resolve().read_text())
    smoke_root = args.smoke_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    smoke_summary_path = smoke_root / "summary.json"
    smoke_summary = json.loads(smoke_summary_path.read_text())
    retained_rows = {
        (str(item["layout_family_id"]), str(item["intrusion_side"])): item
        for item in list(smoke_summary.get("manifest_rows") or [])
    }

    jobs = []
    skipped: list[dict[str, Any]] = []
    for item in smoke_summary["results"]:
        family_id, side = str(item["family_id"]), str(item["intrusion_side"])
        if args.family and family_id not in set(args.family):
            continue
        if args.side and side not in set(args.side):
            continue
        # A raw confirmation on a source episode that was not collision-free is
        # what produced V9.5's false pass. Refuse before rendering, not after.
        source_clean = bool(item.get("clean_success"))
        if args.require_clean_source and not source_clean:
            skipped.append(
                {
                    "family_id": family_id,
                    "intrusion_side": side,
                    "episode_id": str(item["episode_id"]),
                    "reason": "source_physics_not_clean",
                }
            )
            continue
        episode_dir = _find_episode_dir(smoke_root, str(item["episode_id"]))
        source_result = json.loads((episode_dir / "result.json").read_text())
        panel = (source_result.get("scene_params") or {}).get("protr_center")
        if panel is None:
            raise RuntimeError(f"frozen episode {item['episode_id']} has no panel pose")
        row = _build_row(retained_rows[(family_id, side)], configuration)
        jobs.append(
            {
                "family_id": family_id,
                "intrusion_side": side,
                "source_episode_id": str(item["episode_id"]),
                "row": row,
                "result_path": str(episode_dir / "result.json"),
                "trajectory_path": str(episode_dir / "trajectory.json"),
                "output_path": str(output_root / "raw" / f"{family_id}_{side}.npz"),
                "expected_panel_center_m": [float(value) for value in panel],
                "source_physics_clean": source_clean,
                "declared_cluster_geometry": {
                    role: {
                        key: row["pact_clutter_layout"][role][key]
                        for key in ("span_along_line_m", "gap_m", "union_extent_m",
                                    "union_center_m", "member_widths_m")
                    }
                    for role in v96.CLUSTER_ROLES
                },
            }
        )

    if not jobs:
        raise SystemExit(
            "no variant with clean source physics matched the filters; "
            f"skipped: {json.dumps(skipped, sort_keys=True)}"
        )
    config_document = {
        "schema_version": "pact_place_v9_6_w3_config_v1",
        "role": "preregistered_admission_floor",
        "authorizes_gate": False,
        "authorizes_collection": False,
        "written_before_the_run": True,
        "admission_floor": ADMISSION_FLOOR,
        "configuration": configuration,
        "configuration_sha256": sha256_payload(configuration),
        "variant_count": len(jobs),
        "requires_clean_source_physics": bool(args.require_clean_source),
        "all_rendered_variants_have_clean_source": all(
            job["source_physics_clean"] for job in jobs
        ),
        "skipped_dirty_source_variants": skipped,
        "rendered_variants": [
            {"family_id": job["family_id"], "intrusion_side": job["intrusion_side"]}
            for job in jobs
        ],
    }
    config_document["config_sha256"] = sha256_payload(config_document)
    (output_root / "config.json").write_text(
        json.dumps(psr.jsonable(config_document), indent=2, sort_keys=True) + "\n"
    )
    print(f"wrote the admission floor to {output_root / 'config.json'}", flush=True)

    results: list[dict[str, Any]] = []
    if args.workers == 1:
        for job in jobs:
            print(f"Rendering {job['family_id']} / {job['intrusion_side']}", flush=True)
            results.append(_run_variant(job))
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_run_variant, job): job for job in jobs}
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                print(
                    json.dumps(
                        {
                            "family_id": result["family_id"],
                            "side": result["intrusion_side"],
                            **{
                                f"{role}_changed": result["effects"]["decision"][role][
                                    "changed_values"
                                ]
                                for role in ("panel", "inbound_cluster", "outbound_cluster")
                            },
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                results.append(result)
    results.sort(key=lambda item: (item["family_id"], item["intrusion_side"]))

    verdicts = []
    for result in results:
        entry = {
            "family_id": result["family_id"],
            "intrusion_side": result["intrusion_side"],
            "windows": {},
        }
        for window in ("decision", "leg"):
            entry["windows"][window] = {
                role: _role_side_verdict(result["effects"][window][role])
                for role in ("panel", "inbound_cluster", "outbound_cluster")
            }
        verdicts.append(entry)

    balance = []
    for family_id in sorted({item["family_id"] for item in results}):
        pair = {
            item["intrusion_side"]: item
            for item in results
            if item["family_id"] == family_id
        }
        for role in ("inbound_cluster", "outbound_cluster"):
            values = {
                side: int(item["effects"]["decision"][role]["changed_values"])
                for side, item in pair.items()
            }
            low = min(values.values()) if len(values) == 2 else 0
            high = max(values.values()) if values else 0
            ratio = float(high / low) if low > 0 else None
            balance.append(
                {
                    "family_id": family_id,
                    "role": role,
                    "changed_values_by_side": values,
                    "max_to_min_ratio": ratio,
                    "passed": bool(
                        set(values) == {"left", "right"}
                        and low > 0
                        and ratio is not None
                        and ratio <= ADMISSION_FLOOR["max_paired_changed_value_ratio"]
                    ),
                }
            )

    def _role_passes(role: str) -> bool:
        return all(
            entry["windows"]["decision"][role]["meets_min_sensors"]
            and entry["windows"]["decision"][role]["meets_min_changed_values"]
            and entry["windows"]["decision"][role]["meets_corridor_link"]
            for entry in verdicts
        ) and all(
            item["passed"] for item in balance if item["role"] == role
        )

    role_pass = {role: _role_passes(role) for role in v96.CLUSTER_ROLES}
    posed_ok = all(
        result["posed_cluster_geometry"][role]["meets_span_floor"]
        and result["posed_cluster_geometry"][role]["meets_gap_ceiling"]
        for result in results
        for role in v96.CLUSTER_ROLES
    )
    clean_sources = all(bool(item["source_physics_clean"]) for item in results)
    passed = bool(
        len(results) == len(jobs)
        and all(role_pass.values())
        and posed_ok
        and clean_sources
    )

    document = {
        "schema_version": "pact_place_v9_6_w3_raw_confirmation_v1",
        "role": "blocking_clustered_hazard_raw_confirmation",
        "passed": passed,
        "authorizes_gate": False,
        "authorizes_collection": False,
        "authorizes_v1b": False,
        "uses_real_40_sensor_observation": True,
        "uses_geometry_proxy_for_admission": False,
        "requires_physics_regeneration_for_admission": True,
        "rendering_path": "run_pact_place_v9_v0c3_causal_proximity._render_observation (unchanged)",
        "production_tensor_contract": [40, 4, 8, 8],
        "counterfactual_worlds": list(
            ["present", "panel_parked", "inbound_cluster_parked", "outbound_cluster_parked"]
        ),
        "frozen_qpos_no_simulation_between_worlds": True,
        "admission_floor": ADMISSION_FLOOR,
        "config_sha256": config_document["config_sha256"],
        "configuration": configuration,
        "validator_path": _rel(Path(__file__)),
        "validator_sha256": sha256_file(Path(__file__).resolve()),
        "contract_sha256": sha256_file(ROOT / "scripts/pact_place_v96_cluster_contract.py"),
        "smoke_summary_path": _rel(smoke_summary_path),
        "smoke_summary_sha256": sha256_file(smoke_summary_path),
        "variant_count": len(results),
        "role_pass": role_pass,
        "posed_geometry_meets_span_and_gap_contract": posed_ok,
        "all_sources_physics_clean": clean_sources,
        "skipped_dirty_source_variants": skipped,
        "paired_side_balance": balance,
        "verdicts": verdicts,
        "variants": results,
    }
    document["document_sha256"] = sha256_payload(psr.jsonable(document))
    path = output_root / "validation.json"
    path.write_text(json.dumps(psr.jsonable(document), indent=2, sort_keys=True) + "\n")
    print(path, flush=True)
    print(json.dumps({"passed": passed, "role_pass": role_pass, "posed_geometry_ok": posed_ok}, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
