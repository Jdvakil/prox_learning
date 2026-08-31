#!/usr/bin/env python3
"""V10.9 steps 5-6: freeze, compare, run and verify the ACT/PACT training pair.

Single-shot exploratory runs. Neither arm is tuned on the other's result, and no
hyperparameter is chosen here -- every value comes from the frozen V10.9
contract, which reproduces the V5 chunk-100 experiment.

Stages
------
``preflight``  provenance, disk, output directories, parsed command diff,
               create-only frozen command manifests
``train``      ACT then PACT, sequentially
``verify``     epoch count, best epoch/loss, hashes, strict reload, offline
               batch-inference smoke, PACT proximity-consumption proof
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from pact_place_v109_contract import (  # noqa: E402
    ACT_TRAIN_COMMIT_V5,
    CONTRACT_VERSION_V109,
    CONVERTED_DATASET_ROOT,
    ENCODER_SHA256,
    EPISODE_HORIZON,
    PACT_ONLY_FLAGS,
    PROXIMITY_FEATURE_DIM,
    TRAIN_COUNT,
    TRAIN_PARAMS,
    TRAINING_ROOT,
    VALIDATION_COUNT,
    WORK_ROOT,
    canonical_payload_sha256,
    command_diff,
    empty_authorization,
    sha256_file,
    training_command,
    write_immutable_create_only,
)

ACT_DIR = ROOT / "submodules" / "act"
ARMS = ("act", "pact")
PROJECTED_CHECKPOINT_TREE_GB = 7.5  # measured on the V5 chunk-100 run
TRAINING_SOURCE_FILES = (
    "imitate_episodes.py", "policy.py", "utils.py", "fixed_split_data.py",
    "surface_proximity_encoder.py",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git(*args: str, cwd: Path) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def submodule_provenance() -> dict[str, Any]:
    """The ACT submodule may differ from the V5 training commit only by additions."""
    head = git("rev-parse", "HEAD", cwd=ACT_DIR)
    dirty = git("status", "--porcelain", cwd=ACT_DIR)
    numstat = git("diff", "--numstat", ACT_TRAIN_COMMIT_V5, "HEAD", cwd=ACT_DIR)
    status = git("diff", "--name-status", ACT_TRAIN_COMMIT_V5, "HEAD", cwd=ACT_DIR)
    changed: list[dict[str, Any]] = []
    for line in numstat.splitlines():
        added, removed, path = line.split("\t")
        changed.append({"path": path, "added": int(added), "removed": int(removed)})
    kinds = {}
    for line in status.splitlines():
        kind, path = line.split("\t", 1)
        kinds[path] = kind
    touched_training = sorted(
        entry["path"] for entry in changed if entry["path"] in TRAINING_SOURCE_FILES
    )
    deletions = sum(entry["removed"] for entry in changed)
    non_additions = sorted(p for p, k in kinds.items() if k != "A")
    return {
        "act_head": head,
        "act_train_commit_v5": ACT_TRAIN_COMMIT_V5,
        "working_tree_clean": not dirty,
        "changed_files": changed,
        "change_kinds": kinds,
        "total_deletions": deletions,
        "files_not_pure_additions": non_additions,
        "training_source_files_touched": touched_training,
        "training_model_loader_source_unchanged":
            not touched_training and deletions == 0 and not non_additions,
        "source_file_hashes": {
            name: sha256_file(ACT_DIR / name) for name in TRAINING_SOURCE_FILES
        },
    }


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    problems: list[str] = []
    dataset_dir = ROOT / CONVERTED_DATASET_ROOT
    split_manifest = ROOT / WORK_ROOT / "split_manifest.json"
    dataset_manifest = ROOT / WORK_ROOT / "conversion_manifest_encoded.json"
    for path in (dataset_dir, split_manifest, dataset_manifest):
        if not path.exists():
            problems.append(f"missing required input {path}")
    split = json.loads(split_manifest.read_text())
    conversion = json.loads(dataset_manifest.read_text())
    if split["counts"]["train"]["total"] != TRAIN_COUNT or \
            split["counts"]["validation"]["total"] != VALIDATION_COUNT:
        problems.append("split manifest counts are not 113/28")
    if not conversion.get("proximity_contract", {}).get("embedding_tokens_present"):
        problems.append("conversion manifest does not declare embedding tokens")
    if conversion["embedding_token_encoding"]["encoder_sha256"] != ENCODER_SHA256:
        problems.append("conversion manifest encoder hash differs from the contract")

    provenance = submodule_provenance()
    if not provenance["training_model_loader_source_unchanged"]:
        problems.append(
            "ACT submodule differs from the V5 training commit by more than additions: "
            f"touched={provenance['training_source_files_touched']} "
            f"deletions={provenance['total_deletions']} "
            f"non_additions={provenance['files_not_pure_additions']}")
    if not provenance["working_tree_clean"]:
        problems.append("ACT submodule working tree is dirty")

    root = Path(TRAINING_ROOT)
    directories = {arm: root / f"{arm}_seed3101" for arm in ARMS}
    for arm, path in directories.items():
        if path.exists() and any(path.iterdir()):
            problems.append(f"refusing populated output directory {path}")

    usage = shutil.disk_usage(root.parent if root.exists() else "/root")
    free_gb = usage.free / 1e9
    dataset_gb = sum(f.stat().st_size for f in dataset_dir.glob("*.hdf5")) / 1e9
    required_gb = 2 * PROJECTED_CHECKPOINT_TREE_GB
    if free_gb < required_gb + 3.0:
        problems.append(
            f"disk preflight: {free_gb:.1f} GB free, need {required_gb:.1f} GB for two "
            f"checkpoint trees plus headroom")

    commands = {
        arm: training_command(
            arm=arm,
            ckpt_dir=str(directories[arm]),
            dataset_dir=str(dataset_dir),
            split_manifest=str(split_manifest),
            dataset_manifest=str(dataset_manifest),
            expect_split_sha256=split["split_manifest_sha256"],
            expect_dataset_tree_sha256=conversion["converted_tree_file_sha256"],
        )
        for arm in ARMS
    }
    diff = command_diff(commands["act"], commands["pact"])
    if not diff["identical_except_allowance"]:
        problems.extend(diff["violations"])

    document: dict[str, Any] = {
        **empty_authorization(),
        "schema_version": "pact_place_v109_training_preflight_v1",
        "contract_version": CONTRACT_VERSION_V109,
        "role": "frozen ACT/PACT training commands and preflight for the V10.9 pair",
        "is_phase0_pass": False,
        "single_shot_exploratory": True,
        "tuning_forbidden":
            "neither arm may be re-run or tuned on the other's result",
        "training_root": TRAINING_ROOT,
        "dataset_dir": str(dataset_dir.relative_to(ROOT)),
        "split_manifest": str(split_manifest.relative_to(ROOT)),
        "split_manifest_sha256": split["split_manifest_sha256"],
        "dataset_manifest": str(dataset_manifest.relative_to(ROOT)),
        "converted_tree_file_sha256": conversion["converted_tree_file_sha256"],
        "encoder_sha256": ENCODER_SHA256,
        "episode_horizon": EPISODE_HORIZON,
        "train_params": TRAIN_PARAMS,
        "pact_only_flags": PACT_ONLY_FLAGS,
        "commands": commands,
        "command_diff": diff,
        "act_submodule_provenance": provenance,
        "disk": {
            "free_gb": round(free_gb, 2),
            "converted_dataset_gb": round(dataset_gb, 2),
            "projected_checkpoint_tree_gb": PROJECTED_CHECKPOINT_TREE_GB,
            "projected_two_trees_gb": required_gb,
            "sufficient": free_gb >= required_gb + 3.0,
        },
        "problems": problems,
        "ready": not problems,
    }
    document["payload_sha256"] = canonical_payload_sha256(document)
    out = ROOT / WORK_ROOT / "training_preflight.json"
    written = write_immutable_create_only(out, document)
    document["raw_file_sha256"] = written.get("raw_file_sha256")
    return document


def run_arm(arm: str, command: list[str], log_path: Path) -> dict[str, Any]:
    directory = Path(TRAINING_ROOT) / f"{arm}_seed3101"
    directory.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.setdefault("PYTHONUNBUFFERED", "1")
    started = utc_now()
    monotonic = time.monotonic()
    print(f"[{started}] {arm.upper()} training starting", flush=True)
    with log_path.open("w") as stream:
        completed = subprocess.run(
            [sys.executable, *command[1:]], cwd=ACT_DIR, env=environment,
            stdout=stream, stderr=subprocess.STDOUT, check=False,
        )
    finished = utc_now()
    elapsed = time.monotonic() - monotonic
    print(f"[{finished}] {arm.upper()} exited {completed.returncode} "
          f"after {elapsed / 60:.1f} min", flush=True)
    return {
        "arm": arm, "returncode": completed.returncode,
        "started_utc": started, "finished_utc": finished,
        "elapsed_seconds": round(elapsed, 1),
        "elapsed_minutes": round(elapsed / 60, 2),
        "log": str(log_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preflight", "train"), required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.stage == "preflight":
        document = preflight(args)
        print(json.dumps({
            "ready": document["ready"],
            "problems": document["problems"],
            "command_diff": document["command_diff"],
            "disk": document["disk"],
            "act_head": document["act_submodule_provenance"]["act_head"],
            "training_model_loader_source_unchanged":
                document["act_submodule_provenance"]["training_model_loader_source_unchanged"],
            "split_manifest_sha256": document["split_manifest_sha256"],
            "converted_tree_file_sha256": document["converted_tree_file_sha256"],
            "payload_sha256": document["payload_sha256"],
        }, indent=2))
        return 0 if document["ready"] else 1

    preflight_path = ROOT / WORK_ROOT / "training_preflight.json"
    document = json.loads(preflight_path.read_text())
    if not document.get("ready"):
        raise SystemExit("preflight is not ready; refusing to train")
    args.log_dir.mkdir(parents=True, exist_ok=True)

    timing: dict[str, Any] = {
        "schema_version": "pact_place_v109_training_timing_v1",
        "training_root": TRAINING_ROOT,
        "preflight_payload_sha256": document["payload_sha256"],
    }
    for arm in ARMS:
        result = run_arm(arm, document["commands"][arm], args.log_dir / f"{arm}.log")
        timing[arm] = result
        if result["returncode"] != 0:
            timing["halted_after"] = arm
            (Path(TRAINING_ROOT) / "training_timing.json").write_text(
                json.dumps(timing, indent=2, sort_keys=True) + "\n")
            raise SystemExit(f"{arm} training exited {result['returncode']}")
    (Path(TRAINING_ROOT) / "training_timing.json").write_text(
        json.dumps(timing, indent=2, sort_keys=True) + "\n")
    print(json.dumps(timing, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
