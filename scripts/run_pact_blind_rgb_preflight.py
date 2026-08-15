#!/usr/bin/env python3
"""Run the frozen six-row sighted/blind intervention preflight."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import run_pact_confirmatory_schedule as base


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/root/act_retrain_venv/bin/python")
EVALUATOR = ROOT / "submodules/act/eval_pact_blind_rgb_row.py"
BLUR_OUTPUT = Path("/root/pact_blur_sweep_artifacts/evaluation_v1")
ARMS = ("ACT", "PACT", "PACT_PERMUTED")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def scientific_outcome(result: dict[str, Any]) -> dict[str, Any]:
    audit = result["contact_audit"]
    return {
        "task_success": result["task_success"],
        "collision_free_task_success": result["collision_free_task_success"],
        "failure_taxonomy": result["failure_taxonomy"],
        "contact_class_totals": audit["contact_class_totals"],
        "frames_with_contact": audit["frames_with_contact"],
        "maximum_penetration_depth_m": audit["maximum_penetration_depth_m"],
        "first_contact_step": audit["first_contact_step"],
        "non_target_contact_entries": audit["non_target_contact_entries"],
        "collision_free": audit["collision_free"],
    }


def command_for(row: dict[str, Any], manifest: Path, output: Path) -> list[str]:
    command = base.command_for(
        row, manifest_path=manifest, output_dir=output, save_video=False
    )
    command[0] = str(PYTHON)
    command[1] = str(EVALUATOR)
    if row["blind_rgb"]:
        command.append("--blind-rgb")
    if row["arm"] == "PACT_PERMUTED":
        command.extend(
            [
                "--surface-encoder",
                row["surface_encoder_path"],
                "--surface-encoder-sha256",
                row["surface_encoder_sha256"],
                "--token-plan-manifest",
                row["token_plan_manifest_path"],
                "--token-plan-row",
                str(row["token_plan_row"]),
            ]
        )
    command.extend(["--attempt-index", "0"])
    return command


def run_one(row: dict[str, Any], manifest: Path, output_root: Path, env: dict[str, str]) -> dict:
    output = output_root / f"{row['vision_condition']}_{row['arm'].lower()}"
    output.mkdir(parents=True, exist_ok=False)
    completed = subprocess.run(
        command_for(row, manifest, output),
        cwd=ROOT / "submodules/act",
        env=env,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    (output / "stdout.log").write_text(completed.stdout)
    (output / "stderr.log").write_text(completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(
            f"preflight {row['vision_condition']} {row['arm']} failed: {completed.stderr[-1000:]}"
        )
    result = json.loads((output / "result.json").read_text())
    expected = {
        "rollout_id": row["rollout_id"],
        "schedule_row_sha256": row["schedule_row_sha256"],
        "arm": row["arm"],
        "blind_rgb": row["blind_rgb"],
        "blur_sigma": 0.0,
    }
    for key, value in expected.items():
        if result.get(key) != value:
            raise RuntimeError(f"preflight identity mismatch: {key}")
    payloads = []
    for path in sorted(output.glob("*.hdf5")):
        payloads.append({"name": path.name, "sha256": base.sha256_file(path), "bytes": path.stat().st_size})
        path.unlink()
    return {"row": row, "result": result, "output": str(output), "removed_payloads": payloads}


def blur_reference(blur_schedule: dict, row: dict) -> dict:
    matches = [
        item
        for item in blur_schedule["rows"]
        if item["instance_index"] == row["instance_index"]
        and item["checkpoint_seed"] == row["checkpoint_seed"]
        and item["arm"] == row["arm"]
        and item["blur_sigma"] == 0.0
    ]
    if len(matches) != 1:
        raise RuntimeError("sharp no-flag blur reference did not resolve exactly once")
    return json.loads((BLUR_OUTPUT / matches[0]["output_relpath"] / "result.json").read_text())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--blur-schedule", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    schedule = json.loads(args.schedule.read_text())
    payload = dict(schedule)
    schedule_sha = payload.pop("schedule_sha256", None)
    if schedule_sha != canonical_hash(payload):
        raise SystemExit("blind-RGB schedule self-hash mismatch")
    selected = [
        row
        for row in schedule["rows"]
        if row["instance_index"] == 0 and row["checkpoint_seed"] == 3101
    ]
    if len(selected) != 6 or {
        (row["vision_condition"], row["arm"]) for row in selected
    } != {(condition, arm) for condition in ("sighted", "blind") for arm in ARMS}:
        raise SystemExit("six fixed preflight rows did not resolve")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit("preflight output root is not empty")
    env = dict(os.environ)
    env.update(
        {
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
            "PYTHONUNBUFFERED": "1",
            "MLSPACES_ASSETS_DIR": str(ROOT / "assets"),
            "PYTHONPATH": f"{ROOT / 'submodules/molmospaces'}:{ROOT / 'submodules/act'}:{ROOT / 'scripts'}",
            "PACT_CONTACT_AUDIT_SUMMARY_ONLY": "1",
        }
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        runs = list(
            pool.map(
                lambda row: run_one(row, args.manifest.resolve(), args.output_root, env),
                selected,
            )
        )
    by_cell = {(run["row"]["vision_condition"], run["row"]["arm"]): run for run in runs}
    blur_schedule = json.loads(args.blur_schedule.read_text())
    checks = {}
    for arm in ARMS:
        sighted = by_cell[("sighted", arm)]["result"]
        blind = by_cell[("blind", arm)]["result"]
        sighted_info = sighted["policy_info"]
        blind_info = blind["policy_info"]
        sighted_diag = sighted_info["blur_diagnostic"]
        blind_diag = blind_info["blur_diagnostic"]
        reference = blur_reference(blur_schedule, by_cell[("sighted", arm)]["row"])
        reference_info = reference["policy_info"]
        checks[arm] = {
            "blind_input_exact_imagenet_mean": blind_diag[
                "first_policy_visual_input_is_exact_imagenet_mean"
            ] is True,
            "blind_input_constant_shape_1_1_3_240_320": blind_diag[
                "first_policy_visual_input_shape"
            ] == [1, 1, 3, 240, 320],
            "blind_visual_changed": blind_diag["first_visual_input_changed"] is True,
            "same_unmodified_first_camera_frame": (
                sighted_diag["first_sharp_visual_input_sha256"]
                == blind_diag["first_sharp_visual_input_sha256"]
            ),
            "raw_proximity_byte_identical": (
                sighted_info["first_raw_proximity_sha256"]
                == blind_info["first_raw_proximity_sha256"]
            ),
            "action_trace_changed": (
                sighted_info["model_output_trace_sha256"]
                != blind_info["model_output_trace_sha256"]
            ),
            "sighted_visual_bit_identical_to_no_flag": (
                sighted_diag["first_policy_visual_input_sha256"]
                == reference_info["blur_diagnostic"]["first_policy_visual_input_sha256"]
            ),
            "sighted_action_trace_bit_identical_to_no_flag": (
                sighted_info["model_output_trace_sha256"]
                == reference_info["model_output_trace_sha256"]
            ),
            "sighted_scientific_outcome_bit_identical_to_no_flag": (
                scientific_outcome(sighted) == scientific_outcome(reference)
            ),
        }
    passed = all(all(values.values()) for values in checks.values())
    document = {
        "schema_version": "pact_blind_rgb_preflight_v1",
        "schedule_sha256": schedule_sha,
        "fixed_selection": {
            "instance_index": 0,
            "checkpoint_seed": 3101,
            "arms": list(ARMS),
            "vision_conditions": ["sighted", "blind"],
            "selected_before_execution": True,
        },
        "checks": checks,
        "sighted_reference": {
            "experiment": "frozen blur sweep sigma=0",
            "output_root": str(BLUR_OUTPUT),
            "same_instances": True,
            "no_flag": True,
        },
        "policy_endpoint_values_used_for_design_or_selection": False,
        "passed": passed,
        "runs": [
            {
                "arm": run["row"]["arm"],
                "vision_condition": run["row"]["vision_condition"],
                "rollout_id": run["row"]["rollout_id"],
                "output": run["output"],
                "trajectory_payloads_removed_after_hashing": run["removed_payloads"],
            }
            for run in runs
        ],
    }
    document["preflight_sha256"] = canonical_hash(document)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": passed, "preflight_sha256": document["preflight_sha256"]}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
