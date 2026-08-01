#!/usr/bin/env python3
"""Wait for seed 3103, freeze remaining artifacts, smoke, and launch full contact run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/root/act_retrain_venv/bin/python")
TRAINING_SUMMARY = ROOT / "diagnostics_output/pact_contact_endpoint/policy_training_seed3103.json"
CONTACT_DIR = ROOT / "diagnostics_output/pact_contact_endpoint"
WORKER_AMENDMENT = CONTACT_DIR / "worker_amendment_v1.json"
TOKEN_DATA = Path("/root/pact_contact_endpoint_artifacts/token_plan_v1")
OUTPUT_ROOT = Path("/root/pact_contact_endpoint_artifacts/evaluation_v1")
STATUS = Path("/root/pact_contact_endpoint_artifacts/preparation_status.json")
GENERATED_PATHS = (
    "diagnostics_output/pact_contact_endpoint/policy_training_seed3103.json",
    "diagnostics_output/pact_contact_endpoint/policy_training.json",
    "diagnostics_output/pact_contact_endpoint/token_plan.json",
    "diagnostics_output/pact_contact_endpoint/schedule.json",
    "diagnostics_output/pact_contact_endpoint/dispatch.json",
    "diagnostics_output/pact_contact_endpoint/storage_amendment.json",
)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_status(stage: str, **details: Any) -> None:
    document = {
        "schema_version": "pact_contact_preparation_status_v1",
        "pid": os.getpid(),
        "stage": stage,
        "updated_utc": utc_now(),
        **details,
    }
    document["status_sha256"] = canonical_hash(document)
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{STATUS.name}.", dir=STATUS.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(document, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, STATUS)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def protected_processes() -> list[int]:
    matches = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if b"eval_act_obstacle_on_policy.py" in command:
            matches.append(int(entry.name))
    return matches


def training_complete() -> dict[str, Any] | None:
    if not TRAINING_SUMMARY.is_file():
        return None
    try:
        document = json.loads(TRAINING_SUMMARY.read_text())
    except json.JSONDecodeError:
        return None
    if (
        document.get("schema_version") == "pact_contact_seed3103_training_v1"
        and set(document.get("arms", {})) == {"ACT", "PACT"}
        and "PACT_ZERO" in document
        and "PACT_PERMUTED" in document
    ):
        return document
    return None


def wait_for_training(poll_seconds: float) -> dict[str, Any]:
    while True:
        document = training_complete()
        if document is not None:
            return document
        write_status("waiting_for_seed3103_training")
        time.sleep(poll_seconds)


def wait_for_protected_clear(poll_seconds: float) -> None:
    while True:
        active = protected_processes()
        if not active:
            return
        write_status("waiting_for_protected_evaluator", protected_pids=active)
        time.sleep(poll_seconds)


def build_remaining_artifacts() -> None:
    token_manifest = CONTACT_DIR / "token_plan.json"
    write_status("building_permutation_token_plan")
    run(
        [
            str(PYTHON),
            str(ROOT / "scripts/build_pact_contact_token_plan.py"),
            "--dataset-dir",
            "/root/pact_frontend_screen_artifacts/dataset_embedding32_v1",
            "--split-manifest",
            str(ROOT / "diagnostics_output/pact_vs_act/full_act_split_v2.json"),
            "--dataset-manifest",
            "/root/pact_frontend_screen_artifacts/manifests/encoded_conversion_v1.json",
            "--data-dir",
            str(TOKEN_DATA),
            "--manifest-output",
            str(token_manifest),
        ]
    )
    write_status("building_policy_registry")
    run(
        [
            str(PYTHON),
            str(ROOT / "scripts/build_pact_contact_policy_registry.py"),
            "--seed3101",
            str(ROOT / "diagnostics_output/pact_frontend_screen/policy_training.json"),
            "--seed3102",
            str(ROOT / "diagnostics_output/pact_seed_replication/policy_training.json"),
            "--seed3103",
            str(TRAINING_SUMMARY),
            "--output",
            str(CONTACT_DIR / "policy_training.json"),
        ]
    )
    write_status("building_schedule")
    run(
        [
            str(PYTHON),
            str(ROOT / "scripts/build_pact_contact_schedule.py"),
            "--manifest",
            str(ROOT / "configs/pact_contact_endpoint_manifest_v1.json"),
            "--policy-registry",
            str(CONTACT_DIR / "policy_training.json"),
            "--token-plan",
            str(token_manifest),
            "--occlusion-subset",
            str(CONTACT_DIR / "occlusion_subset.json"),
            "--power",
            str(CONTACT_DIR / "power.json"),
            "--preregistration",
            str(ROOT / "configs/pact_contact_endpoint_preregistration_v1.json"),
            "--worker-amendment",
            str(WORKER_AMENDMENT),
            "--output",
            str(CONTACT_DIR / "schedule.json"),
        ]
    )
    write_status("freezing_dispatch")
    run(
        [
            str(PYTHON),
            str(ROOT / "scripts/freeze_pact_contact_dispatch.py"),
            "--schedule",
            str(CONTACT_DIR / "schedule.json"),
            "--manifest",
            str(ROOT / "configs/pact_contact_endpoint_manifest_v1.json"),
            "--policy-registry",
            str(CONTACT_DIR / "policy_training.json"),
            "--preregistration",
            str(ROOT / "configs/pact_contact_endpoint_preregistration_v1.json"),
            "--worker-amendment",
            str(WORKER_AMENDMENT),
            "--token-plan",
            str(token_manifest),
            "--occlusion",
            str(CONTACT_DIR / "occlusion_subset.json"),
            "--power",
            str(CONTACT_DIR / "power.json"),
            "--analysis-script",
            str(ROOT / "scripts/analyze_pact_contact_endpoint.py"),
            "--output-root",
            str(OUTPUT_ROOT),
            "--output",
            str(CONTACT_DIR / "dispatch.json"),
        ]
    )
    write_status("freezing_storage")
    run(
        [
            str(PYTHON),
            str(ROOT / "scripts/build_pact_contact_storage_amendment.py"),
            "--schedule",
            str(CONTACT_DIR / "schedule.json"),
            "--dispatch",
            str(CONTACT_DIR / "dispatch.json"),
            "--manifest",
            str(ROOT / "configs/pact_contact_endpoint_manifest_v1.json"),
            "--analyzer",
            str(ROOT / "scripts/analyze_pact_contact_endpoint.py"),
            "--output-root",
            str(OUTPUT_ROOT),
            "--output",
            str(CONTACT_DIR / "storage_amendment.json"),
        ]
    )


def test_and_commit(expected_head: str) -> str:
    write_status("testing_frozen_dispatch")
    run([str(PYTHON), "-m", "pytest", "-q", "tests"])
    if git("rev-parse", "HEAD") != expected_head:
        raise RuntimeError("HEAD changed during contact preparation")
    status = git("status", "--porcelain").splitlines()
    observed = {line[3:] for line in status if len(line) >= 4}
    if observed != set(GENERATED_PATHS):
        raise RuntimeError(f"unexpected worktree paths before dispatch commit: {sorted(observed)}")
    run(["git", "add", "--", *GENERATED_PATHS])
    run(["git", "commit", "-m", "Freeze PACT contact endpoint dispatch"])
    return git("rev-parse", "HEAD")


def smoke_and_launch(expected_head: str) -> None:
    wait_for_protected_clear(30.0)
    write_status("running_detachment_smoke")
    run(
        [
            str(PYTHON),
            str(ROOT / "scripts/prove_pact_contact_detachment.py"),
            "--schedule",
            str(CONTACT_DIR / "schedule.json"),
            "--dispatch-contract",
            str(CONTACT_DIR / "dispatch.json"),
            "--manifest",
            str(ROOT / "configs/pact_contact_endpoint_manifest_v1.json"),
            "--output-root",
            str(OUTPUT_ROOT),
        ]
    )
    wait_for_protected_clear(30.0)
    write_status("launching_full_stack", expected_head=expected_head)
    run(
        [
            str(PYTHON),
            str(ROOT / "scripts/launch_pact_contact_full_stack.py"),
            "--schedule",
            str(CONTACT_DIR / "schedule.json"),
            "--dispatch",
            str(CONTACT_DIR / "dispatch.json"),
            "--manifest",
            str(ROOT / "configs/pact_contact_endpoint_manifest_v1.json"),
            "--storage-amendment",
            str(CONTACT_DIR / "storage_amendment.json"),
            "--output-root",
            str(OUTPUT_ROOT),
            "--expected-head",
            expected_head,
        ]
    )
    write_status("full_stack_launched", expected_head=expected_head)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args()
    if git("rev-parse", "HEAD") != args.expected_head:
        raise SystemExit("contact preparation started from an unexpected HEAD")
    status = git("status", "--porcelain").splitlines()
    observed = {line[3:] for line in status if len(line) >= 4}
    allowed_training_output = {str(TRAINING_SUMMARY.relative_to(ROOT))}
    if observed and observed != allowed_training_output:
        raise SystemExit(f"contact preparation has unexpected worktree paths: {sorted(observed)}")
    if not WORKER_AMENDMENT.is_file():
        raise SystemExit("frozen worker amendment is missing")
    if OUTPUT_ROOT.exists() or TOKEN_DATA.exists() or (CONTACT_DIR / "token_plan.json").exists():
        raise SystemExit("contact preparation targets are not fresh")
    write_status("waiting_for_seed3103_training")
    training = wait_for_training(args.poll_seconds)
    write_status(
        "seed3103_training_complete",
        act_best_validation_loss=training["arms"]["ACT"]["best_validation_loss"],
        pact_best_validation_loss=training["arms"]["PACT"]["best_validation_loss"],
    )
    build_remaining_artifacts()
    frozen_head = test_and_commit(args.expected_head)
    smoke_and_launch(frozen_head)
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except BaseException as error:
        write_status("failed", error=f"{type(error).__name__}: {error}")
        raise
    raise SystemExit(exit_code)
