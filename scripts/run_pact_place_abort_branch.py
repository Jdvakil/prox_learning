#!/usr/bin/env python3
"""Diagnostic: which check_failure branch aborts the eight suspect place-corridor rows.

Not a gate. Writes diagnostics_output/pact_place_abort_branch/ with role.json and
no decision token. Does not modify frozen Phase-0 records.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MOLMO = ROOT / "submodules" / "molmospaces"
for search_path in (ROOT / "scripts", MOLMO):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from pact_place_corridor_contract import (  # noqa: E402
    FAIL_TOKEN,
    PASS_TOKEN,
    load_contract,
    sha256_file,
    sha256_payload,
)
from run_pact_place_expert_screen import verify_protected_artifacts  # noqa: E402

OUTPUT_ROOT = ROOT / "diagnostics_output/pact_place_abort_branch"
WORKTREE_ROOT = Path("/tmp/pact_place_abort_branch_trees")
PYTHON = "/root/act_retrain_venv/bin/python"

SCREENS: dict[str, dict[str, Any]] = {
    "v1": {
        "config": ROOT / "configs/pact_place_corridor_v1.json",
        "frozen_root": ROOT / "diagnostics_output/pact_place_corridor",
        "parent": "84594895c4dcdff7c2d582ce7bc5c15e4562378b",
        "submodule": "b00dc3523b0930afdae4e95b1aac0ba7211714f3",
        "rows": [13, 16, 21, 22],
        "controls": [],
        "use_current_tree": False,
    },
    "v2": {
        "config": ROOT / "configs/pact_place_corridor_v2.json",
        "frozen_root": ROOT / "diagnostics_output/pact_place_corridor_v2",
        "parent": "9fb040624de03fae250305b1426d0e0767a0611d",
        "submodule": "2828751ee6a1fb5ffcaa30d47fda45859f835510",
        "rows": [3],
        "controls": [],
        "use_current_tree": False,
    },
    "v3": {
        "config": ROOT / "configs/pact_place_corridor_v3.json",
        "frozen_root": ROOT / "diagnostics_output/pact_place_corridor_v3",
        "parent": "aad1cd2",
        "submodule": "1cbb1800db66c871f41f2afc3a360affd1b40f1d",
        "rows": [2, 6, 21],
        "controls": [5, 15, 17],
        "use_current_tree": True,
    },
}

REPRODUCTION_KEYS = (
    "task_success",
    "clean_success",
    "terminal_policy_phase",
    "terminal_action_index",
    "position_error_m",
)

FORBIDDEN_TOKENS = (
    PASS_TOKEN,
    FAIL_TOKEN,
    "proceed_to_collection",
    "proceed_",
)


def _json_dump(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def frozen_result(screen: str, role_index: int) -> tuple[Path, dict[str, Any]]:
    frozen_root = SCREENS[screen]["frozen_root"]
    matches = list(
        (frozen_root / "expert_screen_rows").glob(f"{role_index:02d}_*/result.json")
    )
    if len(matches) != 1:
        raise RuntimeError(f"expected one frozen result for {screen} row {role_index}: {matches}")
    return matches[0], json.loads(matches[0].read_text())


def worktree_path(screen: str) -> Path:
    if SCREENS[screen]["use_current_tree"]:
        return ROOT
    return WORKTREE_ROOT / screen


def ensure_worktree(screen: str) -> Path:
    spec = SCREENS[screen]
    destination = worktree_path(screen)
    if spec["use_current_tree"]:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not (destination / ".git").exists() and not (destination / ".git").is_file():
        subprocess.run(
            ["git", "worktree", "add", "--detach", str(destination), spec["parent"]],
            cwd=ROOT,
            check=True,
        )
    molmo_dest = destination / "submodules" / "molmospaces"
    if not (molmo_dest / ".git").exists() and not (molmo_dest / ".git").is_file():
        molmo_dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "worktree", "add", "--detach", str(molmo_dest), spec["submodule"]],
            cwd=MOLMO,
            check=True,
        )
    return destination


def launch_row_argv(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--molmo", type=Path, required=True)
    args, passthrough = parser.parse_known_args(argv)
    scripts = ROOT / "scripts"
    for path in (scripts, args.molmo):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("MLSPACES_ASSETS_DIR", str(ROOT / "assets"))
    os.environ.pop("DISPLAY", None)
    from pact_place_abort_branch_telemetry import install

    install()
    sys.argv = [str(args.runner), *passthrough]
    source = args.runner.read_text()
    # The producing-tree runner hashes protected files relative to its own
    # worktree, which predates later frozen diagnostics. MAIN already verified
    # those hashes before launch.
    source = source.replace("verify_protected_artifacts(contract)", "None")
    namespace = {
        "__name__": "__main__",
        "__file__": str(args.runner),
    }
    exec(compile(source, str(args.runner), "exec"), namespace)
    return 0


def run_one_row(screen: str, role_index: int) -> dict[str, Any]:
    spec = SCREENS[screen]
    tree = ensure_worktree(screen)
    contract = load_contract(spec["config"])
    row = contract["expert_screen_rows"][role_index]
    output = OUTPUT_ROOT / screen
    sidecar = (
        output
        / "expert_screen_rows"
        / f"{row['role_index']:02d}_{row['episode_id'][:16]}"
        / "abort_branch_telemetry.json"
    )
    env = os.environ.copy()
    env["MUJOCO_GL"] = "egl"
    env["PYOPENGL_PLATFORM"] = "egl"
    env["PYTHONUNBUFFERED"] = "1"
    env["MLSPACES_ASSETS_DIR"] = str(ROOT / "assets")
    env["PACT_PLACE_ABORT_SIDECAR"] = str(sidecar)
    env.pop("DISPLAY", None)
    molmo = tree / "submodules" / "molmospaces"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "scripts"), str(molmo), str(tree / "scripts"), env.get("PYTHONPATH", "")]
    )
    command = [
        PYTHON,
        str(ROOT / "scripts" / "run_pact_place_abort_branch.py"),
        "launch-row",
        "--runner",
        str(tree / "scripts" / "run_pact_place_expert_screen.py"),
        "--molmo",
        str(molmo),
        "--config",
        str(spec["config"]),
        "--output-root",
        str(output),
        "--development-row",
        str(role_index),
        "--workers",
        "1",
    ]
    completed = subprocess.run(command, cwd=tree, env=env, check=False)
    result_path = (
        output
        / "expert_screen_rows"
        / f"{row['role_index']:02d}_{row['episode_id'][:16]}"
        / "result.json"
    )
    if completed.returncode != 0 or not result_path.is_file():
        raise RuntimeError(
            f"{screen} row {role_index} failed rc={completed.returncode} path={result_path}"
        )
    return json.loads(result_path.read_text())


def reproduction_fields(result: dict[str, Any]) -> dict[str, Any]:
    tracking = result.get("terminal_tracking") or {}
    return {
        "task_success": result.get("task_success"),
        "clean_success": result.get("clean_success"),
        "terminal_policy_phase": result.get("terminal_policy_phase"),
        "terminal_action_index": result.get("terminal_action_index"),
        "position_error_m": tracking.get("position_error_m"),
    }


def compare_reproduction(frozen: dict[str, Any], rerun: dict[str, Any]) -> dict[str, Any]:
    expected = reproduction_fields(frozen)
    observed = reproduction_fields(rerun)
    return {
        "reproduced": expected == observed,
        "expected": expected,
        "observed": observed,
    }


def measure_wall_thickness() -> dict[str, Any]:
    from pact_place_abort_branch_telemetry import (
        cup_collision_geom_ids,
        wall_thickness_along_radial_ray,
    )
    from run_pact_place_expert_screen import _make_config
    from molmo_spaces.data_generation.pipeline import cleanup_episode_resources
    from molmo_spaces.data_generation.runtime_compat import assert_supported_runtime

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("MLSPACES_ASSETS_DIR", str(ROOT / "assets"))
    os.environ.pop("DISPLAY", None)
    assert_supported_runtime(strict=True)

    grasp_points = []
    for screen, spec in SCREENS.items():
        for role_index in spec["rows"]:
            _, result = frozen_result(screen, role_index)
            grasp_points.append(
                {
                    "screen": screen,
                    "role_index": role_index,
                    "adjusted_grasp_object_local_position_m": result["grasp_diagnostics"][
                        "adjusted_grasp_object_local_position_m"
                    ],
                }
            )

    _, v3_row2 = frozen_result("v3", 2)
    contract = load_contract(SCREENS["v3"]["config"])
    row = contract["expert_screen_rows"][2]
    scratch = Path(tempfile.mkdtemp(prefix="pact_place_abort_wall_"))
    task = sampler = None
    try:
        config = _make_config(scratch / "dummy.json")
        sampler = config.task_sampler_config.task_sampler_class(config)
        sampler.seed_task_sampling(int(v3_row2["selected_seed"]["seed_u32"]))
        sampler.set_pact_manifest_row(row)
        task = sampler.sample_task(house_index=int(row["scene_template_house_index"]))
        if task is None:
            raise RuntimeError("sample_task returned None for wall-thickness probe")
        task.reset()
        env = task.env
        model, data = env.current_model, env.current_data
        pickup_name = task.config.task_config.pickup_obj_name
        manager = env.object_managers[env.current_batch_index]
        pickup = manager.get_object_by_name(pickup_name)
        cup_geoms = cup_collision_geom_ids(model, env, pickup_name)
        measurements = []
        for item in grasp_points:
            measured = wall_thickness_along_radial_ray(
                model,
                data,
                body_id=int(pickup.object_id),
                local_point=item["adjusted_grasp_object_local_position_m"],
                cup_geom_ids=cup_geoms,
            )
            measured.update(item)
            measurements.append(measured)
        trip = 0.002
        return {
            "schema_version": "pact_place_abort_branch_wall_thickness_v1",
            "object_uid": "Cup_10",
            "pickup_name": pickup_name,
            "n_collision_geoms": len(cup_geoms),
            "empty_gripper_trip_m": trip,
            "inter_finger_dist_range0_m": 0.0,
            "measurements": measurements,
        }
    finally:
        cleanup_episode_resources(
            task=task,
            policy=None,
            task_sampler=sampler,
            preloaded_policy=None,
            close_task_sampler=sampler is not None,
        )
        shutil.rmtree(scratch, ignore_errors=True)


def _row_report(screen: str, role_index: int, is_control: bool) -> dict[str, Any]:
    frozen_path, frozen = frozen_result(screen, role_index)
    rerun_matches = list(
        (OUTPUT_ROOT / screen / "expert_screen_rows").glob(
            f"{role_index:02d}_*/result.json"
        )
    )
    sidecar_matches = list(
        (OUTPUT_ROOT / screen / "expert_screen_rows").glob(
            f"{role_index:02d}_*/abort_branch_telemetry.json"
        )
    )
    if len(rerun_matches) != 1:
        return {
            "screen": screen,
            "role_index": role_index,
            "is_control": is_control,
            "status": "missing_rerun",
        }
    rerun = json.loads(rerun_matches[0].read_text())
    sidecar = (
        json.loads(sidecar_matches[0].read_text()) if sidecar_matches else None
    )
    reproduction = compare_reproduction(frozen, rerun)
    tracking = rerun.get("terminal_tracking") or {}
    terminal = None if sidecar is None else sidecar.get("terminal")
    holding_dips = []
    if sidecar is not None:
        for step in sidecar.get("steps") or []:
            if step.get("empty_gripper_predicate") and step.get("is_holding_object"):
                holding_dips.append(
                    {
                        "policy_phase": step.get("policy_phase"),
                        "action_index": step.get("action_index"),
                        "inter_finger_dist_m": step.get("inter_finger_dist_m"),
                        "gripper_empty_trip_m": step.get("gripper_empty_trip_m"),
                    }
                )
    place = rerun.get("place_metrics") or {}
    contacts = rerun.get("terminal_robot_environment_contacts") or []
    gripper_cup_contact = False
    for pair in contacts:
        names = " ".join(
            str(pair.get(key) or "")
            for key in ("body1", "body2", "geom1", "geom2", "root1", "root2")
        )
        if "pad" in names.lower() and "cup_10" in names.lower():
            gripper_cup_contact = True
            break
    return {
        "screen": screen,
        "role_index": role_index,
        "is_control": is_control,
        "frozen_path": str(frozen_path.relative_to(ROOT)),
        "reproduction": reproduction,
        "interpreted": bool(reproduction["reproduced"]),
        "branch": tracking.get("check_failure_branch")
        if terminal is None
        else terminal.get("branch"),
        "terminal": terminal,
        "terminal_tracking": {
            key: tracking.get(key)
            for key in (
                "check_failure_branch",
                "inter_finger_dist_m",
                "gripper_empty_trip_m",
                "check_pos_err_m",
                "position_error_m",
                "rotation_error_rad",
                "rotation_error_deg",
                "is_holding_object",
                "object_to_tcp_distance_m",
                "object_to_tcp_offset_m",
            )
        },
        "supported_by_receptacle": place.get("supported_by_receptacle"),
        "robot_contact": place.get("robot_contact"),
        "gripper_cup_contact": gripper_cup_contact,
        "n_holding_empty_predicates": len(holding_dips),
        "first_holding_empty_predicate": None if not holding_dips else holding_dips[0],
        "last_holding_empty_predicate": None if not holding_dips else holding_dips[-1],
        "terminal_policy_phase": rerun.get("terminal_policy_phase"),
        "task_success": rerun.get("task_success"),
        "clean_success": rerun.get("clean_success"),
    }


def classify_outcome(rows: list[dict[str, Any]]) -> dict[str, Any]:
    suspects = [row for row in rows if not row["is_control"] and row.get("interpreted")]
    discarded = [row for row in rows if not row.get("interpreted")]
    branches = {row.get("branch") for row in suspects}
    by_row = []
    for row in suspects:
        branch = row.get("branch")
        phase = str(row.get("terminal_policy_phase") or "")
        empty = branch == "empty_gripper"
        held = bool(row.get("gripper_cup_contact")) if empty else None
        if empty and held and phase.startswith("placement"):
            kind = "category_error_during_intended_release"
        elif empty and held and (
            phase.startswith("outbound") or phase in {"lift", "preplace"}
        ):
            kind = "transport_check_fired_while_pads_still_on_cup"
        elif empty and held is False:
            kind = "empty_gripper_without_pad_cup_contact"
        elif branch == "rot_err":
            kind = "real_rot_err_defect"
        elif branch == "pos_err":
            kind = "real_pos_err_defect"
        else:
            kind = "other"
        by_row.append(
            {
                "screen": row["screen"],
                "role_index": row["role_index"],
                "branch": branch,
                "kind": kind,
                "object_demonstrably_in_gripper": held,
            }
        )
    if discarded:
        named = "non_reproduction"
    elif branches == {"empty_gripper"}:
        named = "branch_1_empty_gripper"
    elif "rot_err" in branches:
        named = "branch_3_rot_err"
    else:
        named = "something_else"
    return {
        "named_outcome": named,
        "discarded_non_reproducing_rows": [
            {"screen": row["screen"], "role_index": row["role_index"]}
            for row in discarded
            if not row["is_control"]
        ],
        "per_suspect_kind": by_row,
    }


def write_role_and_forbid_tokens() -> None:
    role = {
        "role": "diagnostic_not_a_gate",
        "authorizes_collection": False,
        "next_action": "none_diagnostic_only",
        "purpose": "identify_check_failure_branch_on_eight_suspect_place_corridor_rows",
        "rows": {
            screen: {"suspects": spec["rows"], "controls": spec["controls"]}
            for screen, spec in SCREENS.items()
        },
    }
    _json_dump(OUTPUT_ROOT / "role.json", role)
    dumped = json.dumps(role)
    if any(token in dumped for token in FORBIDDEN_TOKENS):
        raise RuntimeError("role.json emitted a gate token")
    for path in OUTPUT_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix in {".mp4", ".png", ".jpg"}:
            continue
        text = path.read_text(errors="replace")
        for token in FORBIDDEN_TOKENS:
            if token in text:
                raise RuntimeError(f"{token} in {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        nargs="?",
        default="all",
        choices=("all", "step0", "run", "report", "launch-row"),
    )
    args, rest = parser.parse_known_args(argv)
    if args.command == "launch-row":
        return launch_row_argv(rest)

    contract = load_contract(SCREENS["v3"]["config"])
    verify_protected_artifacts(contract)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    if args.command in {"all", "step0"}:
        thickness = measure_wall_thickness()
        _json_dump(OUTPUT_ROOT / "wall_thickness.json", thickness)
        print(json.dumps({"wall_thickness": thickness}, indent=2, sort_keys=True)[:4000])

    if args.command in {"all", "run"}:
        jobs: list[tuple[str, int]] = []
        for screen, spec in SCREENS.items():
            ensure_worktree(screen)
            for role_index in spec["rows"] + spec["controls"]:
                jobs.append((screen, role_index))
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _run(job: tuple[str, int]) -> tuple[str, int, str]:
            screen, role_index = job
            print(f"running {screen} row {role_index}", flush=True)
            run_one_row(screen, role_index)
            return screen, role_index, "ok"

        failures = []
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(_run, job): job for job in jobs}
            for future in as_completed(futures):
                job = futures[future]
                try:
                    screen, role_index, status = future.result()
                    print(f"finished {screen} row {role_index} {status}", flush=True)
                except Exception as error:  # noqa: BLE001
                    screen, role_index = job
                    failures.append((screen, role_index, error))
                    print(f"FAILED {screen} row {role_index}: {error}", flush=True)
        if failures:
            raise RuntimeError(
                "abort-branch rows failed: "
                + ", ".join(f"{screen}:{role_index}" for screen, role_index, _ in failures)
            )

    if args.command in {"all", "report"}:
        rows = []
        for screen, spec in SCREENS.items():
            for role_index in spec["rows"]:
                rows.append(_row_report(screen, role_index, False))
            for role_index in spec["controls"]:
                rows.append(_row_report(screen, role_index, True))
        analysis = {
            "schema_version": "pact_place_abort_branch_analysis_v1",
            "role": "diagnostic_not_a_gate",
            "authorizes_collection": False,
            "next_action": "none_diagnostic_only",
            "wall_thickness": json.loads((OUTPUT_ROOT / "wall_thickness.json").read_text())
            if (OUTPUT_ROOT / "wall_thickness.json").is_file()
            else None,
            "rows": rows,
            "outcome": classify_outcome(rows),
        }
        analysis["analysis_sha256"] = sha256_payload(
            {key: value for key, value in analysis.items() if key != "analysis_sha256"}
        )
        _json_dump(OUTPUT_ROOT / "analysis.json", analysis)
        write_role_and_forbid_tokens()
        verify_protected_artifacts(contract)
        print(json.dumps(analysis["outcome"], indent=2, sort_keys=True))

    verify_protected_artifacts(contract)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
