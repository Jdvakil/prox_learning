#!/usr/bin/env python3
"""Run the frozen six-row sharp/blur intervention check before dispatch."""

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
EVALUATOR = ROOT / "submodules/act/eval_pact_blur_sweep_row.py"
V3_OUTPUT = Path("/root/pact_geometry_generalization_v3_artifacts/evaluation_v1")
ARMS = ("ACT", "PACT", "PACT_PERMUTED")
ENDPOINT_KEYS = (
    "task_success",
    "collision_free_task_success",
    "failure_taxonomy",
    "contact_audit",
)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def command_for(row: dict[str, Any], manifest: Path, output: Path) -> list[str]:
    command = base.command_for(
        row, manifest_path=manifest, output_dir=output, save_video=False
    )
    command[0] = str(PYTHON)
    command[1] = str(EVALUATOR)
    command.extend(["--blur-sigma", str(row["blur_sigma"])])
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


def run_one(
    row: dict[str, Any], manifest: Path, output_root: Path, environment: dict[str, str]
) -> dict[str, Any]:
    output = output_root / (
        f"sigma_{str(row['blur_sigma']).replace('.', 'p')}_{row['arm'].lower()}"
    )
    output.mkdir(parents=True, exist_ok=False)
    command = command_for(row, manifest, output)
    completed = subprocess.run(
        command,
        cwd=ROOT / "submodules/act",
        env=environment,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    (output / "stdout.log").write_text(completed.stdout)
    (output / "stderr.log").write_text(completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(f"preflight {row['arm']} sigma={row['blur_sigma']} failed")
    result_path = output / "result.json"
    if not result_path.is_file():
        raise RuntimeError(f"preflight {row['arm']} sigma={row['blur_sigma']} lacks result")
    result = json.loads(result_path.read_text())
    for key, expected in (
        ("rollout_id", row["rollout_id"]),
        ("schedule_row_sha256", row["schedule_row_sha256"]),
        ("arm", row["arm"]),
        ("blur_sigma", row["blur_sigma"]),
    ):
        if result.get(key) != expected:
            raise RuntimeError(f"preflight identity mismatch: {key}")
    trajectory_hashes = []
    for path in sorted(output.glob("*.hdf5")):
        trajectory_hashes.append(
            {"name": path.name, "sha256": base.sha256_file(path), "bytes": path.stat().st_size}
        )
        path.unlink()
    return {
        "row": row,
        "output": str(output),
        "result": result,
        "trajectory_payloads_removed_after_hashing": trajectory_hashes,
    }


def v3_reference(schedule: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    matches = [
        item
        for item in schedule["rows"]
        if item["instance_episode_id"] == row["instance_episode_id"]
        and item["checkpoint_seed"] == row["checkpoint_seed"]
        and item["arm"] == row["arm"]
    ]
    if len(matches) != 1:
        raise RuntimeError("v3 sharp reference did not resolve exactly once")
    return json.loads((V3_OUTPUT / matches[0]["output_relpath"] / "result.json").read_text())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--v3-schedule", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    schedule = json.loads(args.schedule.read_text())
    schedule_payload = dict(schedule)
    schedule_sha = schedule_payload.pop("schedule_sha256", None)
    if schedule_sha != canonical_hash(schedule_payload):
        raise SystemExit("blur schedule self-hash mismatch")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit("preflight output root is not empty")
    selected = [
        row
        for row in schedule["rows"]
        if row["instance_index"] == 0
        and row["checkpoint_seed"] == 3101
        and row["blur_sigma"] in (0.0, 2.0)
    ]
    if len(selected) != 6 or {row["arm"] for row in selected} != set(ARMS):
        raise SystemExit("six fixed preflight rows did not resolve")
    environment = dict(os.environ)
    environment.update(
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
                lambda row: run_one(row, args.manifest.resolve(), args.output_root, environment),
                selected,
            )
        )
    by_cell = {(run["row"]["arm"], run["row"]["blur_sigma"]): run for run in runs}
    checks = {}
    for arm in ARMS:
        sharp = by_cell[(arm, 0.0)]["result"]
        blurred = by_cell[(arm, 2.0)]["result"]
        sharp_info = sharp["policy_info"]
        blurred_info = blurred["policy_info"]
        checks[arm] = {
            "sigma_zero_visual_exact_identity": sharp_info["blur_diagnostic"]["first_visual_input_changed"] is False,
            "sigma_two_visual_input_changed": blurred_info["blur_diagnostic"]["first_visual_input_changed"] is True,
            "same_unblurred_first_visual_input": (
                sharp_info["blur_diagnostic"]["first_sharp_visual_input_sha256"]
                == blurred_info["blur_diagnostic"]["first_sharp_visual_input_sha256"]
            ),
            "proximity_first_frame_unchanged": (
                sharp_info["first_raw_proximity_sha256"]
                == blurred_info["first_raw_proximity_sha256"]
            ),
            "action_trace_changed": (
                sharp_info["model_output_trace_sha256"]
                != blurred_info["model_output_trace_sha256"]
            ),
        }
    v3_schedule = json.loads(args.v3_schedule.read_text())
    sharp_reproduction = {}
    for arm in ("PACT", "PACT_PERMUTED"):
        sharp = by_cell[(arm, 0.0)]["result"]
        reference = v3_reference(v3_schedule, by_cell[(arm, 0.0)]["row"])
        current_endpoint = {key: sharp[key] for key in ENDPOINT_KEYS}
        reference_endpoint = {key: reference[key] for key in ENDPOINT_KEYS}
        sharp_reproduction[arm] = {
            "exact_endpoint_match": current_endpoint == reference_endpoint,
            "current_endpoint_sha256": canonical_hash(current_endpoint),
            "v3_endpoint_sha256": canonical_hash(reference_endpoint),
        }
    passed = all(all(values.values()) for values in checks.values()) and all(
        item["exact_endpoint_match"] for item in sharp_reproduction.values()
    )
    document = {
        "schema_version": "pact_blur_sweep_preflight_v1",
        "schedule_sha256": schedule_sha,
        "fixed_selection": {
            "instance_index": 0,
            "checkpoint_seed": 3101,
            "arms": list(ARMS),
            "sigmas": [0.0, 2.0],
            "selected_before_execution": True,
        },
        "checks": checks,
        "sharp_v3_reproduction": sharp_reproduction,
        "endpoint_values_used_for_parameter_selection": False,
        "passed": passed,
        "runs": [
            {
                "arm": run["row"]["arm"],
                "blur_sigma": run["row"]["blur_sigma"],
                "rollout_id": run["row"]["rollout_id"],
                "output": run["output"],
                "trajectory_payloads_removed_after_hashing": run[
                    "trajectory_payloads_removed_after_hashing"
                ],
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
