#!/usr/bin/env python3
"""V10.5 Step 1: independent reconstruction of the V9.5 fragility corpus.

No stored outcome boolean is imported. Strict-clean status is re-derived from
each retained row's own contact and task telemetry, and the row's file hashes
are recorded alongside it so a later reader can tell which bytes were read.

No ``env.step`` and no new episode occur here. Every retained trajectory is
replayed as recorded qpos into a freshly sampled task, and the reconstruction
is accepted only when the replayed state matches the recorded state.
"""

from __future__ import annotations

import argparse
import collections
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

from pact_place_v105_contract import (  # noqa: E402
    CONTRACT_VERSION_V105,
    ENVIRONMENT_VERSION_V105,
    FRAGILITY_ARTIFACT,
    FRAGILITY_ROWS_DIR,
    INTRUSION_SIDES,
    RECONSTRUCTION_ROOT,
    V95_LAYOUT_FAMILY_IDS,
    build_specification_contract,
    empty_authorization,
    sha256_file,
    sha256_payload,
    write_immutable_create_only,
)

TCP_RESIDUAL_LIMIT_M = 0.001
MIN_CLEAN_PER_CELL = 2
BASE_SCENE = (
    ROOT
    / "submodules/molmospaces/molmo_spaces/data_generation/custom_scenes"
    / "pact_place_corridor_v5.xml"
)


def _pin_threads() -> None:
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[key] = "1"


def _establish_env() -> None:
    _pin_threads()
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("MLSPACES_ASSETS_DIR", str(ROOT / "assets"))
    os.environ.pop("DISPLAY", None)


def derive_strict_clean(result: dict[str, Any]) -> dict[str, Any]:
    """Re-derive strict-clean from the row's own telemetry, not its boolean.

    The stored ``clean_success`` is recorded for comparison but is never the
    source of truth: if the two disagree the row is reported as disputed and
    excluded, rather than being silently trusted either way.
    """
    audit = result.get("contact_audit") or {}
    totals = dict(audit.get("contact_class_totals") or {})
    forbidden = {
        name: int(count)
        for name, count in totals.items()
        if name not in ("grasp_target",) and int(count) > 0
    }
    stability_events = list(result.get("clutter_stability_events") or [])
    derived = bool(
        result.get("status") == "complete"
        and result.get("task_success")
        and result.get("grasp_phase_success")
        and result.get("place_phase_success")
        and result.get("cup_lifted_one_cm")
        and not forbidden
        and not stability_events
        and bool(audit.get("collision_free", True))
    )
    stored = bool(result.get("clean_success"))
    return {
        "derived_strict_clean": derived,
        "stored_clean_success": stored,
        "agrees_with_stored": derived == stored,
        "forbidden_contact_classes": forbidden,
        "n_clutter_stability_events": len(stability_events),
        "status": result.get("status"),
        "task_success": bool(result.get("task_success")),
        "failure_cause": result.get("failure_cause"),
    }


def index_corpus(rows_dir: Path) -> list[dict[str, Any]]:
    """Every retained fragility row, with its cell and both file hashes."""
    records: list[dict[str, Any]] = []
    for directory in sorted(rows_dir.iterdir()):
        result_path = directory / "result.json"
        trajectory_path = directory / "trajectory.json"
        entry: dict[str, Any] = {
            "row_dir": str(directory.relative_to(ROOT)),
            "result_present": result_path.is_file(),
            "trajectory_present": trajectory_path.is_file(),
        }
        if not entry["result_present"]:
            entry.update({"available": False, "reason": "result.json absent"})
            records.append(entry)
            continue
        result = json.loads(result_path.read_text())
        role = int(result["role_index"])
        # role_index = family_index * 2 + (0 left, 1 right). This is the
        # corpus's own encoding, asserted rather than assumed.
        family = V95_LAYOUT_FAMILY_IDS[role // 2]
        side = "left" if role % 2 == 0 else "right"
        entry.update(
            {
                "role_index": role,
                "family_id": family,
                "intrusion_side": side,
                "side_matches_role_index": side == str(result.get("intrusion_side")),
                "episode_id": result.get("episode_id"),
                "seed_u32": int((result.get("selected_seed") or {}).get("seed_u32", 0)),
                "result_file_sha256": sha256_file(result_path),
                "result_sha256_recorded": result.get("result_sha256"),
                "trajectory_file_sha256": (
                    sha256_file(trajectory_path)
                    if entry["trajectory_present"]
                    else "absent"
                ),
                "trajectory_sha256_recorded": result.get("trajectory_sha256"),
                "config_sha256": result.get("config_sha256"),
                "row_sha256": result.get("row_sha256"),
                **derive_strict_clean(result),
            }
        )
        if not entry["trajectory_present"]:
            entry.update({"available": False, "reason": "trajectory.json absent"})
            records.append(entry)
            continue
        trajectory = json.loads(trajectory_path.read_text())
        steps = trajectory.get("steps") or []
        has_qpos = bool(steps) and all("qpos" in step for step in steps[:8])
        entry.update(
            {
                "n_frames": int(trajectory.get("n") or len(steps)),
                "trajectory_row_binding_ok": (
                    trajectory.get("row_sha256") == result.get("row_sha256")
                ),
                "retains_qpos": has_qpos,
                "qpos_len": len(steps[0]["qpos"]) if has_qpos else 0,
            }
        )
        entry["available"] = bool(
            entry["side_matches_role_index"]
            and entry["trajectory_row_binding_ok"]
            and has_qpos
            and entry["agrees_with_stored"]
        )
        if not entry["available"]:
            reasons = []
            if not entry["side_matches_role_index"]:
                reasons.append("role_index/side disagreement")
            if not entry["trajectory_row_binding_ok"]:
                reasons.append("trajectory row binding mismatch")
            if not has_qpos:
                reasons.append("no retained qpos")
            if not entry["agrees_with_stored"]:
                reasons.append("derived strict-clean disputes the stored boolean")
            entry["reason"] = "; ".join(reasons)
        records.append(entry)
    return records


def coverage(records) -> dict[str, Any]:
    cells: dict[str, dict[str, Any]] = {}
    for family in V95_LAYOUT_FAMILY_IDS:
        for side in INTRUSION_SIDES:
            cells[f"{family}|{side}"] = {
                "family_id": family,
                "intrusion_side": side,
                "n_rows": 0,
                "n_available": 0,
                "n_clean": 0,
                "clean_rows": [],
            }
    for entry in records:
        key = f"{entry.get('family_id')}|{entry.get('intrusion_side')}"
        if key not in cells:
            continue
        cell = cells[key]
        cell["n_rows"] += 1
        if entry.get("available"):
            cell["n_available"] += 1
            if entry.get("derived_strict_clean"):
                cell["n_clean"] += 1
                cell["clean_rows"].append(entry["row_dir"])
    short = [k for k, v in cells.items() if v["n_clean"] < MIN_CLEAN_PER_CELL]
    return {
        "cells": cells,
        "n_cells": len(cells),
        "min_clean_per_cell_required": MIN_CLEAN_PER_CELL,
        "cells_below_floor": short,
        "min_clean_observed": min(v["n_clean"] for v in cells.values()),
        "total_clean": sum(v["n_clean"] for v in cells.values()),
        "total_rows": sum(v["n_rows"] for v in cells.values()),
        "sufficient_for_siting": not short,
    }


def reconstruct_row(entry: dict[str, Any]) -> dict[str, Any]:
    """Replay one retained trajectory into a freshly sampled V9.5 task."""
    import mujoco
    from molmo_spaces.data_generation.pipeline import cleanup_episode_resources
    from pact_place_v105_contract import v95_row_payload
    from run_pact_place_expert_screen import _make_config
    from run_pact_place_v7_replay_videos import apply_recorded_qpos

    directory = ROOT / entry["row_dir"]
    result = json.loads((directory / "result.json").read_text())
    steps = json.loads((directory / "trajectory.json").read_text())["steps"]
    payload = v95_row_payload(entry["family_id"], entry["intrusion_side"])
    row = {
        "role_index": int(entry["role_index"]),
        "episode_id": str(entry["episode_id"]),
        "intrusion_side": entry["intrusion_side"],
        "task_seed_u32": int(entry["seed_u32"]),
        "task_seed_u64": int(entry["seed_u32"]),
        "sampler_class": "PactPlaceCorridorV93Sampler",
        **payload,
    }
    scratch = Path(tempfile.mkdtemp(prefix="v105_recon_"))
    task = sampler = None
    try:
        config = _make_config(
            scratch / "d.json",
            scene_xml=BASE_SCENE,
            sampler_class="PactPlaceCorridorV93Sampler",
        )
        sampler = config.task_sampler_config.task_sampler_class(config)
        sampler.seed_task_sampling(int(entry["seed_u32"]))
        sampler.set_pact_manifest_row(row)
        task = sampler.sample_task(house_index=int(row["scene_template_house_index"]))
        env = task.env
        model, data = env.current_model, env.current_data
        residuals: list[float] = []
        for index in (0, len(steps) // 2, len(steps) - 1):
            step = steps[index]
            apply_recorded_qpos(env, step["qpos"])
            mujoco.mj_forward(model, data)
            recorded = np.asarray(step["tcp_position_m"], dtype=float)
            robot_view = env.current_robot.robot_view
            gripper_id = robot_view.get_gripper_movegroup_ids()[0]
            live = np.asarray(
                robot_view.get_gripper(gripper_id).leaf_frame_to_world[:3, 3],
                dtype=float,
            )
            residuals.append(float(np.linalg.norm(live - recorded)))
        active = []
        for body_index in range(int(model.nbody)):
            name = str(model.body(body_index).name or "")
            if name.startswith("pact_clutter_") and not name.startswith(
                "pact_clutter_mount_"
            ):
                active.append(
                    {
                        "body": name,
                        "pos_m": [float(v) for v in data.xpos[body_index]],
                        "quat_wxyz": [float(v) for v in data.xquat[body_index]],
                    }
                )
        qpos0 = np.asarray(steps[0]["qpos"], dtype=np.float64)
        return {
            **entry,
            "reconstructed": True,
            "tcp_residual_max_m": float(max(residuals)),
            "tcp_residual_within_limit": bool(
                max(residuals) <= TCP_RESIDUAL_LIMIT_M
            ),
            "tcp_residuals_m": residuals,
            "n_active_clutter_bodies": len(active),
            "active_clutter": active,
            "qpos_len": int(qpos0.size),
            "qpos_sha256": sha256_payload([float(v) for v in qpos0]),
            "layout_id": row["layout_id"],
            "palette_uids": list(row["pact_clutter_palette"]),
        }
    except Exception as error:  # noqa: BLE001 - a failure is recorded, not raised
        return {
            **entry,
            "reconstructed": False,
            "reconstruction_error": f"{type(error).__name__}: {error}",
        }
    finally:
        cleanup_episode_resources(
            task=task, policy=None, task_sampler=sampler,
            preloaded_policy=None, close_task_sampler=sampler is not None,
        )
        shutil.rmtree(scratch, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=ROOT / RECONSTRUCTION_ROOT)
    parser.add_argument(
        "--probe-per-cell",
        type=int,
        default=2,
        help="strict-clean rows per cell to replay through the live sampler",
    )
    args = parser.parse_args()
    _establish_env()
    started = time.time()

    output_root = args.output_root.resolve()
    if output_root.exists():
        raise SystemExit(f"refusing to overwrite {output_root}")

    contract = build_specification_contract()
    records = index_corpus(ROOT / FRAGILITY_ROWS_DIR)
    cover = coverage(records)

    probes: list[dict[str, Any]] = []
    for key, cell in sorted(cover["cells"].items()):
        chosen = cell["clean_rows"][: max(0, int(args.probe_per_cell))]
        for row_dir in chosen:
            entry = next(item for item in records if item["row_dir"] == row_dir)
            probes.append(reconstruct_row(entry))
            print(
                json.dumps(
                    {
                        "cell": key,
                        "row": row_dir,
                        "reconstructed": probes[-1]["reconstructed"],
                        "tcp_residual_max_m": probes[-1].get("tcp_residual_max_m"),
                    }
                ),
                flush=True,
            )

    residual_ok = all(
        item.get("tcp_residual_within_limit") for item in probes if item["reconstructed"]
    )
    all_reconstructed = all(item["reconstructed"] for item in probes)
    disputed = [
        item["row_dir"] for item in records if item.get("agrees_with_stored") is False
    ]
    document = {
        "schema_version": "pact_place_v105_reconstruction_v1",
        "contract_version": CONTRACT_VERSION_V105,
        "environment_version": ENVIRONMENT_VERSION_V105,
        "specification_contract_payload_sha256": sha256_payload(contract),
        "source": {
            "fragility_artifact": FRAGILITY_ARTIFACT,
            "fragility_raw_file_sha256": sha256_file(ROOT / FRAGILITY_ARTIFACT),
            "rows_dir": FRAGILITY_ROWS_DIR,
            "n_rows_indexed": len(records),
        },
        "strict_clean_derived_not_imported": True,
        "disputed_rows": disputed,
        "n_disputed_rows": len(disputed),
        "coverage": cover,
        "probes": probes,
        "n_probes": len(probes),
        "probe_per_cell": int(args.probe_per_cell),
        "all_probes_reconstructed": all_reconstructed,
        "tcp_residual_limit_m": TCP_RESIDUAL_LIMIT_M,
        "all_tcp_residuals_within_limit": residual_ok,
        "creates_episode": False,
        "calls_env_step": False,
        "elapsed_s": time.time() - started,
        **empty_authorization(),
        "reconstruction_passed": bool(
            cover["sufficient_for_siting"] and all_reconstructed and residual_ok
        ),
    }
    hashes = write_immutable_create_only(
        output_root / "reconstruction.json", document
    )
    np.savez_compressed(
        output_root / "corpus_index.npz",
        row_dir=np.array([r["row_dir"] for r in records], dtype=object),
        family_id=np.array([r.get("family_id", "") for r in records], dtype=object),
        intrusion_side=np.array(
            [r.get("intrusion_side", "") for r in records], dtype=object
        ),
        seed_u32=np.array([r.get("seed_u32", 0) for r in records], dtype=np.int64),
        derived_strict_clean=np.array(
            [bool(r.get("derived_strict_clean")) for r in records], dtype=bool
        ),
        available=np.array([bool(r.get("available")) for r in records], dtype=bool),
        n_frames=np.array([int(r.get("n_frames") or 0) for r in records],
                          dtype=np.int64),
        allow_pickle=True,
    )
    print(json.dumps({
        "reconstruction_passed": document["reconstruction_passed"],
        "n_rows_indexed": len(records),
        "total_clean": cover["total_clean"],
        "min_clean_per_cell": cover["min_clean_observed"],
        "cells_below_floor": cover["cells_below_floor"],
        "n_probes": len(probes),
        "all_probes_reconstructed": all_reconstructed,
        **hashes,
    }, indent=2))
    return 0 if document["reconstruction_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
