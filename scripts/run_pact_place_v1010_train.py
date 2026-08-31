#!/usr/bin/env python3
"""V10.10: freeze, compare, run and verify the ACT/PACT training pair.

Same V10.9/V5 chunk-100 settings and the same fail-closed flag diff. The only
parameter that is computed rather than fixed is the horizon:
``max(635, T_max + 8)`` over the converted episodes.
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
    ACT_TRAIN_COMMIT_V5, PACT_ONLY_FLAGS, TRAIN_PARAMS, command_diff,
    training_command,
)
from pact_place_v1010_contract import (  # noqa: E402
    BASE_EPISODE_HORIZON, CONTRACT_VERSION_V1010, CONVERTED_DATASET_ROOT,
    ENCODER_SHA256, HORIZON_MARGIN, TRAINING_ROOT, TRAIN_COUNT,
    VALIDATION_COUNT, WORK_ROOT, canonical_payload_sha256, empty_authorization,
    sha256_file, write_immutable_create_only,
)

ACT_DIR = ROOT / "submodules" / "act"
ARMS = ("act", "pact")
TRAINING_SOURCE_FILES = ("imitate_episodes.py", "policy.py", "utils.py",
                         "fixed_split_data.py", "surface_proximity_encoder.py")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ACT_DIR, check=True,
                          capture_output=True, text=True).stdout.strip()


def episode_horizon(conversion: dict[str, Any]) -> tuple[int, int]:
    t_max = int(conversion["timesteps"]["converted_t_max"])
    return max(BASE_EPISODE_HORIZON, t_max + HORIZON_MARGIN), t_max


def preflight() -> dict[str, Any]:
    problems: list[str] = []
    dataset_dir = ROOT / CONVERTED_DATASET_ROOT
    split_manifest = ROOT / WORK_ROOT / "split_manifest.json"
    dataset_manifest = ROOT / WORK_ROOT / "conversion_manifest_encoded.json"
    for path in (dataset_dir, split_manifest, dataset_manifest):
        if not path.exists():
            problems.append(f"missing {path}")
    split = json.loads(split_manifest.read_text())
    conversion = json.loads(dataset_manifest.read_text())
    if split["counts"]["train"]["total"] != TRAIN_COUNT or \
            split["counts"]["validation"]["total"] != VALIDATION_COUNT:
        problems.append("split is not 120/24")
    if conversion["embedding_token_encoding"]["encoder_sha256"] != ENCODER_SHA256:
        problems.append("encoder hash differs from the contract")
    horizon, t_max = episode_horizon(conversion)

    numstat = git("diff", "--numstat", ACT_TRAIN_COMMIT_V5, "HEAD")
    changed = []
    for line in numstat.splitlines():
        added, removed, path = line.split("\t")
        changed.append({"path": path, "added": int(added), "removed": int(removed)})
    touched = sorted(c["path"] for c in changed if c["path"] in TRAINING_SOURCE_FILES)
    deletions = sum(c["removed"] for c in changed)
    if touched or deletions:
        problems.append(f"training source changed: {touched}, {deletions} deletions")
    if git("status", "--porcelain"):
        problems.append("ACT submodule working tree is dirty")

    root = Path(TRAINING_ROOT)
    directories = {arm: root / f"{arm}_seed3101" for arm in ARMS}
    for arm, path in directories.items():
        if path.exists() and any(path.iterdir()):
            problems.append(f"refusing populated {path}")
    # Intermediate checkpoints are pruned when each arm finishes, so the two
    # full trees never coexist. The peak is one training tree (~7.6 GiB) plus
    # the previous arm's retained best/last (~0.9 GiB).
    free_gib = shutil.disk_usage("/root").free / 2**30
    required_gib = 7.6 + 0.9 + 3.0   # peak tree + retained pair + headroom
    if free_gib < required_gib:
        problems.append(
            f"{free_gib:.1f} GiB free; the training peak needs {required_gib:.1f} "
            "(one full tree plus the previous arm's retained checkpoints)")

    params = dict(TRAIN_PARAMS)
    params["episode_horizon"] = horizon
    commands = {}
    for arm in ARMS:
        command = training_command(
            arm=arm, ckpt_dir=str(directories[arm]), dataset_dir=str(dataset_dir),
            split_manifest=str(split_manifest), dataset_manifest=str(dataset_manifest),
            expect_split_sha256=split["split_manifest_sha256"],
            expect_dataset_tree_sha256=conversion["converted_tree_file_sha256"])
        index = command.index("--episode_horizon")
        command[index + 1] = str(horizon)
        commands[arm] = command
    diff = command_diff(commands["act"], commands["pact"])
    if not diff["identical_except_allowance"]:
        problems.extend(diff["violations"])

    document = {
        **empty_authorization(),
        "schema_version": "pact_place_v1010_training_preflight_v1",
        "contract_version": CONTRACT_VERSION_V1010,
        "is_phase0_pass": False,
        "training_root": TRAINING_ROOT,
        "dataset_dir": str(dataset_dir.relative_to(ROOT)),
        "split_manifest_sha256": split["split_manifest_sha256"],
        "converted_tree_file_sha256": conversion["converted_tree_file_sha256"],
        "encoder_sha256": ENCODER_SHA256,
        "converted_t_max": t_max,
        "episode_horizon": horizon,
        "episode_horizon_rule": "max(635, T_max + 8)",
        "train_params": params, "pact_only_flags": PACT_ONLY_FLAGS,
        "commands": commands, "command_diff": diff,
        "act_submodule": {"head": git("rev-parse", "HEAD"),
                          "v5_commit": ACT_TRAIN_COMMIT_V5,
                          "changed_files": changed,
                          "training_source_unchanged": not touched and not deletions},
        "disk_free_gib": round(free_gib, 2),
        "problems": problems, "ready": not problems,
    }
    document["payload_sha256"] = canonical_payload_sha256(document)
    write_immutable_create_only(ROOT / WORK_ROOT / "training_preflight.json", document)
    return document


def run_arm(arm: str, command: list[str], log_path: Path) -> dict[str, Any]:
    directory = Path(TRAINING_ROOT) / f"{arm}_seed3101"
    directory.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.setdefault("PYTHONUNBUFFERED", "1")
    started, monotonic = utc_now(), time.monotonic()
    print(f"[{started}] {arm.upper()} starting", flush=True)
    with log_path.open("w") as stream:
        completed = subprocess.run([sys.executable, *command[1:]], cwd=ACT_DIR,
                                   env=environment, stdout=stream,
                                   stderr=subprocess.STDOUT, check=False)
    elapsed = time.monotonic() - monotonic
    print(f"[{utc_now()}] {arm.upper()} exited {completed.returncode} "
          f"after {elapsed/60:.1f} min", flush=True)
    return {"arm": arm, "returncode": completed.returncode, "started_utc": started,
            "finished_utc": utc_now(), "elapsed_minutes": round(elapsed / 60, 2),
            "log": str(log_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preflight", "train"), required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "preflight":
        document = preflight()
        print(json.dumps({k: document[k] for k in (
            "ready", "problems", "episode_horizon", "converted_t_max",
            "disk_free_gib", "split_manifest_sha256")}, indent=2))
        return 0 if document["ready"] else 1

    document = json.loads((ROOT / WORK_ROOT / "training_preflight.json").read_text())
    if not document.get("ready"):
        raise SystemExit("preflight is not ready")
    args.log_dir.mkdir(parents=True, exist_ok=True)
    timing: dict[str, Any] = {"schema_version": "pact_place_v1010_training_timing_v1",
                              "training_root": TRAINING_ROOT,
                              "preflight_payload_sha256": document["payload_sha256"]}
    for arm in ARMS:
        result = run_arm(arm, document["commands"][arm], args.log_dir / f"{arm}.log")
        timing[arm] = result
        if result["returncode"] == 0:
            # Free the intermediate checkpoints before the next arm trains. The
            # plan permits pruning them after verification, and verification
            # reads only policy_best, policy_last, the statistics, the epoch log
            # and the run manifest -- none of which is touched here.
            directory = Path(TRAINING_ROOT) / f"{arm}_seed3101"
            pruned, freed = [], 0
            for pattern in ("policy_epoch_*.ckpt", "resume_bundle.ckpt"):
                for path in sorted(directory.glob(pattern)):
                    freed += path.stat().st_size
                    pruned.append(path.name)
                    path.unlink()
            timing[f"{arm}_pruned"] = {
                "files": pruned, "count": len(pruned),
                "freed_gib": round(freed / 2**30, 2),
                "retained": ["policy_best.ckpt", "policy_last.ckpt",
                             "dataset_stats.pkl", "run_manifest.json",
                             "epoch_log.jsonl"],
            }
            print(f"    pruned {len(pruned)} intermediate checkpoints, "
                  f"{freed / 2**30:.1f} GiB", flush=True)
        if result["returncode"] != 0:
            timing["halted_after"] = arm
            (Path(TRAINING_ROOT) / "training_timing.json").write_text(
                json.dumps(timing, indent=2, sort_keys=True) + "\n")
            raise SystemExit(f"{arm} exited {result['returncode']}")
    (Path(TRAINING_ROOT) / "training_timing.json").write_text(
        json.dumps(timing, indent=2, sort_keys=True) + "\n")
    print(json.dumps(timing, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
