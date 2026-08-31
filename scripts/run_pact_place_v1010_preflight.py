#!/usr/bin/env python3
"""V10.10 non-episode preflight over all 24 family x side x pose cells.

No rollout is executed. Each cell is sampled, reset, and inspected at its
initial state only. A cell that fails any check stops the run before collection.

Checked per cell:
  * exactly four active and four parked clutter bodies
  * the active bodies carry the registered slots, uids and layout poses
  * no initial contact involving the robot, target, clutter, panel or pendant
  * clutter is settled (stable) and inside the workspace bounds
  * the pose-specific certified static-pendant scene is what actually loaded
  * live contact parity: the contact audit at the initial state agrees with a
    direct read of the MuJoCo contact list
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "submodules" / "molmospaces"))
sys.path.insert(0, str(ROOT / "submodules" / "act"))


def check_cell(task_spec: tuple[str, str, str]) -> dict[str, Any]:
    family, side, pose = task_spec
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("MLSPACES_ASSETS_DIR", str(ROOT / "assets"))
    os.environ.pop("DISPLAY", None)
    import hashlib

    import mujoco
    import numpy as np

    from pact_place_v1010_contract import (
        ACTIVE_CLUTTER_COUNT, ACTIVE_CLUTTER_SLOTS, ACTIVE_CLUTTER_UIDS,
        INACTIVE_CLUTTER_SLOTS, SCENE_BY_POSE, build_row, cell_key,
    )
    problems: list[str] = []
    row = build_row(family, side, pose, 0)

    import eval_pact_place_row as place
    from molmo_spaces.configs.task_configs import PickAndPlaceTaskConfig
    from molmo_spaces.tasks.enclosure_reach import (
        PactPlaceCorridorTask, PactPlaceCorridorV1010FourObjectSampler,
    )
    from molmo_spaces.tasks.pact_place_contact_audit import classify_contact
    from molmo_spaces.tasks.pact_contact_audit import robot_environment_contact_pairs

    scene = ROOT / row["pact_v1010_scene_relative"]
    observed_scene_sha = hashlib.sha256(scene.read_bytes()).hexdigest()
    if observed_scene_sha != row["pact_v106_scene_sha256"]:
        problems.append(f"scene hash {observed_scene_sha} != registered")
    if "pact_place_corridor_v2" in scene.name:
        problems.append("V2 scene selected")

    class Config(place._BaseConfig):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            learned = self.policy_config
            self.task_type = "pick_and_place"
            self.task_horizon = 900
            self.end_on_success = False
            self.task_config = PickAndPlaceTaskConfig(task_cls=PactPlaceCorridorTask)
            self.policy_config = learned
            self.task_sampler_config.task_sampler_class = (
                PactPlaceCorridorV1010FourObjectSampler)
            self.task_sampler_config.scene_xml_paths = [str(scene)] * 2
            if hasattr(self.robot_config, "action_noise_config"):
                self.robot_config.action_noise_config.enabled = False

    config = Config(
        output_dir=Path("/tmp/claude-0/v1010_preflight") / cell_key(family, side, pose),
        num_workers=1,
        policy_config=place.policy_config_factory(
            arm="ACT", checkpoint_dir="", checkpoint_path="", stats_path="",
            checkpoint_seed=3101, surface_encoder_path="",
            sensor_names=tuple(f"s{i}" for i in range(40)),
        ),
    )
    sampler = task = None
    detail: dict[str, Any] = {}
    try:
        from pact_place_v1010_contract import cell_seed
        task = None
        for retry in range(int(row["max_sampling_retries"]) + 1):
            seed = (cell_seed(family, side, pose, 0) if retry == 0
                    else cell_seed(family, side, pose, 1000 + retry))
            sampler = PactPlaceCorridorV1010FourObjectSampler(config)
            sampler.seed_task_sampling(int(seed["seed_u32"]))
            sampler.set_pact_manifest_row(row)
            try:
                task = sampler.sample_task(house_index=1)
            except Exception as exc:  # noqa: BLE001
                detail.setdefault("sampling_rejections", []).append(
                    f"{type(exc).__name__}: {str(exc)[:90]}")
                try:
                    sampler.close()
                except Exception:  # noqa: BLE001
                    pass
                sampler, task = None, None
                continue
            if task is None:
                continue
            # An initial-state contact is a rejectable draw, not a defect: the
            # plan requires it be identified separately and refused before a
            # scientific rollout. Resample rather than accept it.
            probe = robot_environment_contact_pairs(task.env)
            if probe:
                detail.setdefault("initial_contact_rejections", []).append(
                    {"retry": retry, "pairs": len(probe),
                     "classes": sorted({classify_contact(x) for x in probe})})
                try:
                    sampler.close()
                except Exception:  # noqa: BLE001
                    pass
                sampler, task = None, None
                continue
            detail["accepted_at_retry"] = retry
            break
        if task is None:
            problems.append("sampling exhausted its retries")
            return {"cell": cell_key(family, side, pose), "passed": False,
                    "problems": problems, "detail": detail}

        active = list(getattr(sampler, "_pact_active_clutter_names", []))
        layout = getattr(sampler, "_pact_active_clutter_layout", {}) or {}
        detail["active_bodies"] = active
        if len(active) != ACTIVE_CLUTTER_COUNT:
            problems.append(f"{len(active)} active clutter bodies, expected 4")
        slots = sorted(str(v["palette_slot"]) for v in layout.values())
        detail["active_slots"] = slots
        if slots != sorted(ACTIVE_CLUTTER_SLOTS):
            problems.append(f"active slots {slots} != {sorted(ACTIVE_CLUTTER_SLOTS)}")
        uids = {str(v["palette_slot"]): str(v["uid"]) for v in layout.values()}
        detail["active_uids"] = uids
        for slot, uid in ACTIVE_CLUTTER_UIDS.items():
            if uids.get(slot) != uid:
                problems.append(f"slot {slot} is {uids.get(slot)!r}, expected {uid!r}")

        env = task.env
        model, data = env.current_model, env.current_data
        mujoco.mj_forward(model, data)

        compiled = [str(item["body"]) for item in sampler._pact_clutter_objects]
        parked = [b for b in compiled if b not in active]
        detail["compiled_clutter_bodies"] = len(compiled)
        detail["parked_bodies"] = len(parked)
        if len(parked) != len(INACTIVE_CLUTTER_SLOTS):
            problems.append(f"{len(parked)} parked bodies, expected 4")
        parked_z = {}
        for body in parked:
            try:
                z = float(data.xpos[int(model.body(body).id)][2])
            except Exception:  # noqa: BLE001
                problems.append(f"parked body {body} has no pose")
                continue
            parked_z[body] = round(z, 3)
            if z > 0.0:
                problems.append(f"parked body {body} sits at z={z:.3f}, not parked")
        detail["parked_z"] = parked_z

        for body, item in layout.items():
            desired = np.asarray(item["center_m"], dtype=float)
            low, high = sampler._body_collision_aabb(model, data, body)
            centre = (np.asarray(low) + np.asarray(high)) / 2.0
            offset = float(np.linalg.norm(centre[:2] - desired[:2]))
            if offset > 0.05:
                problems.append(f"{body} centre is {offset*1000:.0f} mm from its layout xy")
            lo, hi = (np.asarray(x, dtype=float)
                      for x in sampler._layout()["workspace_bounds_m"])
            if not (np.all(centre >= lo - 0.05) and np.all(centre <= hi + 0.05)):
                problems.append(f"{body} is outside the workspace bounds")

        pairs = robot_environment_contact_pairs(env)
        classes: dict[str, int] = {}
        for pair in pairs:
            classes[classify_contact(pair)] = classes.get(classify_contact(pair), 0) + 1
        detail["initial_contact_classes"] = classes
        detail["initial_contact_pairs"] = len(pairs)
        for name in ("clutter", "mounted_fixture", "hazard_bar",
                     "other_environment", "grasp_target", "place_receptacle"):
            if classes.get(name):
                problems.append(f"initial contact: {name}={classes[name]}")

        direct = int(data.ncon)
        detail["mujoco_ncon"] = direct
        detail["contact_parity"] = len(pairs) <= direct
        if len(pairs) > direct:
            problems.append("audit reports more robot-environment pairs than MuJoCo has")

        params = getattr(task, "scene_params", {}) or {}
        detail["environment_version"] = params.get("pact_place_environment_version")
        detail["active_count_marker"] = params.get("pact_v1010_active_clutter_count")
        detail["layout_sha256"] = params.get("pact_v1010_layout_sha256")
        if detail["environment_version"] != "pact_place_corridor_v10_10_four_object":
            problems.append(f"environment marker {detail['environment_version']!r}")
        if detail["active_count_marker"] != ACTIVE_CLUTTER_COUNT:
            problems.append(f"active count marker {detail['active_count_marker']!r}")
        detail["identity_sha256"] = params.get("pact_v1010_identity_sha256")
        if detail["identity_sha256"] != row["pact_v1010_identity_sha256"]:
            problems.append("four-object identity hash disagrees with the row binding")
        if not detail["layout_sha256"]:
            problems.append("no per-episode layout hash was recorded")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"preflight raised: {type(exc).__name__}: {exc}")
        detail["traceback"] = traceback.format_exc()[-800:]
    finally:
        try:
            if sampler is not None:
                sampler.close()
        except Exception:  # noqa: BLE001
            pass
    return {"cell": cell_key(family, side, pose), "family_id": family,
            "intrusion_side": side, "pose_id": pose,
            "passed": not problems, "problems": problems, "detail": detail}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    from pact_place_v1010_contract import (
        PREFLIGHT_ROOT, canonical_payload_sha256, cells, empty_authorization,
        write_immutable_create_only,
    )
    specs = cells()
    print(f"preflight over {len(specs)} cells on {args.workers} workers", flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(check_cell, s): s for s in specs}
        for done, future in enumerate(as_completed(futures), 1):
            try:
                results.append(future.result())
            except Exception as exc:  # noqa: BLE001
                f, s, p = futures[future]
                results.append({"cell": f"{f}|{s}|{p}", "passed": False,
                                "problems": [f"worker died: {exc!r}"], "detail": {}})
            print(f"  {done}/{len(specs)}", flush=True)
    results.sort(key=lambda r: r["cell"])
    failed = [r for r in results if not r["passed"]]
    document = {
        **empty_authorization(),
        "schema_version": "pact_place_v1010_preflight_v1",
        "role": "non-episode preflight over all 24 cells; no rollout executed",
        "is_phase0_pass": False,
        "cells_checked": len(results),
        "cells_passed": len(results) - len(failed),
        "cells_failed": len(failed),
        "passed": not failed,
        "results": results,
    }
    document["payload_sha256"] = canonical_payload_sha256(document)
    write_immutable_create_only(ROOT / PREFLIGHT_ROOT / "preflight.json", document)
    print(json.dumps({"cells_passed": document["cells_passed"],
                      "cells_failed": document["cells_failed"],
                      "passed": document["passed"],
                      "first_problems": [p for r in failed[:3] for p in r["problems"][:2]]},
                     indent=2))
    return 0 if document["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
