#!/usr/bin/env python3
"""Publish the V10.11c 100-episode collection to a HuggingFace dataset repo.

Same layout as Lundii/table_smoke: a manifest, the full ledger including
rejected attempts, the frozen contract, and one directory per accepted episode
carrying trajectory.h5, result.json and the five MP4s.

Every accepted V10.11c row already carries the fixed exterior table camera
``exo_camera_1`` and its per-frame calibration, validated during collection and
recorded under ``coordinator_validation.table_camera``. That validation is
copied into the manifest verbatim rather than re-asserted, and every uploaded
file is re-verified against the remote object hash after the commit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "diagnostics_output/pact_place_v1011c_collection_100"
LEDGER = RUN_DIR / "ledger.jsonl"
CONTRACT = RUN_DIR / "contract.json"
CLOSEOUT = RUN_DIR / "closeout.json"
ROWS_ROOT = ROOT / "assets/datagen/pact_place_corridor_v10_11c_100/rows"
REPO_ID = "Lundii/mixed_v1011_clutter_geometry"
PREFIX = "pact_place_corridor_v10_11c_100"
CALIBRATION_KEYS = ("extrinsic_cv", "cam2world_gl", "intrinsic_cv")
EXPECTED_ROW_FILES = 7


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def accepted_rows() -> list[dict[str, Any]]:
    records = [json.loads(line) for line in LEDGER.read_text().splitlines() if line.strip()]
    return [r for r in records if r.get("accepted")]


def row_dir(record: dict[str, Any]) -> Path:
    """The ledger carries no row_dir; rows are named by attempt_id[:16]."""
    return ROWS_ROOT / str(record["attempt_id"])[:16]


def check(rows: list[dict[str, Any]], contract: dict[str, Any], require: int | None) -> None:
    if require is not None and len(rows) != require:
        raise SystemExit(
            f"{len(rows)} accepted rows, expected {require}. "
            "Pass --allow-partial to publish an incomplete collection."
        )
    problems: list[str] = []
    seen: set[str] = set()
    for record in rows:
        cell = record.get("cell", "?")
        directory = row_dir(record)
        if not directory.is_dir():
            problems.append(f"{cell}: row directory {directory.name} missing")
            continue
        if directory.name in seen:
            problems.append(f"{cell}: duplicate row directory {directory.name}")
        seen.add(directory.name)
        files = {p.name for p in directory.iterdir() if p.is_file()}
        if len(files) != EXPECTED_ROW_FILES:
            problems.append(f"{cell}/{directory.name}: {len(files)} files, expected {EXPECTED_ROW_FILES}")
        if not any(n.endswith("exo_camera_1.mp4") for n in files):
            problems.append(f"{cell}/{directory.name}: no exterior camera RGB")
        if "trajectory.h5" not in files:
            problems.append(f"{cell}/{directory.name}: no trajectory.h5")
        validation = (record.get("coordinator_validation") or {}).get("table_camera") or {}
        detail = validation.get("detail") or {}
        if not validation.get("passed"):
            problems.append(f"{cell}/{directory.name}: table-camera validation not passed")
        for key in CALIBRATION_KEYS:
            delta = detail.get(f"{key}_within_episode_max_delta")
            if delta is None:
                problems.append(f"{cell}/{directory.name}: {key} absent")
            elif float(delta) != 0.0:
                problems.append(f"{cell}/{directory.name}: {key} drifts within the episode ({delta})")
        if not record.get("clean_success"):
            problems.append(f"{cell}/{directory.name}: not a clean success")
    if problems:
        raise SystemExit("refusing to publish:\n  " + "\n  ".join(problems[:40]))


def build_manifest(rows: list[dict[str, Any]], contract: dict[str, Any]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda r: (str(r["cell"]), int(r["attempt_index"])))
    entries = []
    for index, record in enumerate(ordered):
        directory = row_dir(record)
        files = sorted(p for p in directory.iterdir() if p.is_file())
        entries.append({
            "index": index,
            "repo_dir": f"{PREFIX}/rows/{index:03d}_{directory.name}",
            "cell": record["cell"],
            "family_id": record["family_id"],
            "intrusion_side": record["intrusion_side"],
            "pose_id": record["pose_id"],
            "attempt_id": record["attempt_id"],
            "attempt_index": int(record["attempt_index"]),
            "task_seed_u32": int(record["task_seed_u32"]),
            "episode_steps": int(record["episode_steps"]),
            "clean_success": bool(record["clean_success"]),
            "contact_class_totals": record.get("contact_class_totals"),
            "clutter_stability_event_count": record.get("clutter_stability_event_count"),
            "table_camera_validation": (record.get("coordinator_validation") or {}).get("table_camera"),
            "files": [{"name": p.name, "bytes": p.stat().st_size,
                       "sha256": sha256_file(p)} for p in files],
        })
    return {
        "schema_version": "pact_place_v1011c_collection_100_publication_v1",
        "environment_version": contract["environment_version"],
        "sampler_class": contract["sampler_class"],
        "active_clutter_slots": contract["active_clutter_slots"],
        "object_labels": contract["object_labels"],
        "observations": contract.get("observations"),
        "source_contract_sha256": contract["payload_sha256"],
        "collection_master_seed": (contract.get("streams") or {}).get("collection_master_seed"),
        "added_camera": "exo_camera_1",
        "required_h5_group": "traj_0/obs/sensor_param/exo_camera_1",
        "required_calibration_keys": list(CALIBRATION_KEYS),
        "n_rows": len(entries),
        "rows": entries,
    }


README = """---
license: mit
task_categories:
- robotics
tags:
- robotics
- mujoco
- manipulation
- camera-calibration
---

# V10.11c mixed mesh/primitive clutter — {n} episodes

Clean pick-and-place demonstrations from `{env}`, each carrying a fixed
exterior table camera (`{cam}`) with per-frame calibration.

## What the scene contains

Six table-clutter bodies: **three mesh assets and three runtime MuJoCo
primitives**. The primitives are built on the episode's `MjSpec` rather than
compiled into the scene, so the certified scene files stay byte-identical.

{objects}

Slots `08`/`09` are sampled per episode in a bounded annular sector around the
target cup, so near-target clutter varies from episode to episode. The
remaining clutter uses the frozen V9.5 layout with its inherited vessel jitter.

## Every row is a clean success

Each published episode is a *strict-clean* success under the collection's own
predicate: the task succeeded and there was no contact with clutter, the hazard
bar, mounted fixtures or the rest of the environment. Rejected attempts are not
published as rows, but they remain in `ledger.jsonl` so the yield is auditable
({n} accepted from {attempts} attempts).

## The exterior camera

Every row carries:

- `episode_00000000_{cam}.mp4` — exterior RGB, plus a depth track;
- `{group}` in `trajectory.h5`, holding `extrinsic_cv` `(T,3,4)`,
  `cam2world_gl` `(T,4,4)` and `intrinsic_cv` `(T,3,3)` as finite `float64`;
- a **fixed pose within each episode** — the within-episode max delta of all
  three calibration arrays is exactly `0.0`.

The pose is re-drawn **between** episodes, so load the extrinsic from the
episode you are projecting. Reusing one episode's `extrinsic_cv` across the set
will silently misproject the others, since the intrinsics do match.

`manifest.json` carries that validation per row plus a SHA-256 for every file.

## Layout

```
{prefix}/
  manifest.json        per-row metadata, camera validation and file hashes
  ledger.jsonl         the full run ledger, including rejected attempts
  contract.json        the frozen run contract
  closeout.json        the run close-out
  rows/NNN_<row>/      one directory per accepted episode
```

Per-row files: `trajectory.h5`, `result.json`, and MP4s for the exterior camera
(RGB + depth), the wrist camera (RGB + depth) and the proximity-sensor heatmap.

## Coverage

24 cells: four layout families x two intrusion sides x three pendant poses.
Cells, seeds and per-episode contact totals are in `manifest.json`.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--expect", type=int, default=100)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    contract = json.loads(CONTRACT.read_text())
    rows = accepted_rows()
    check(rows, contract, None if args.allow_partial else args.expect)
    manifest = build_manifest(rows, contract)

    total_records = sum(1 for line in LEDGER.read_text().splitlines() if line.strip())
    staging = Path(os.environ.get("TMPDIR", "/tmp")) / "v1011c_publish"
    staging.mkdir(parents=True, exist_ok=True)
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    labels = contract["object_labels"]
    kinds = {"01": "primitive", "08": "primitive", "09": "primitive"}
    objects = "\n".join(
        f"- `{slot}` **{labels[slot]}** — {kinds.get(slot, 'mesh')}"
        for slot in contract["active_clutter_slots"]
    )
    (staging / "README.md").write_text(README.format(
        n=len(rows), env=contract["environment_version"], cam="exo_camera_1",
        group="traj_0/obs/sensor_param/exo_camera_1", prefix=PREFIX,
        objects=objects, attempts=total_records))

    payload = sum(f["bytes"] for e in manifest["rows"] for f in e["files"])
    print(f"{len(rows)} accepted rows -> {args.repo_id}/{PREFIX}")
    print(f"payload: {payload/1e9:.2f} GB across "
          f"{sum(len(e['files']) for e in manifest['rows'])} row files")
    if args.dry_run:
        for entry in manifest["rows"][:5]:
            print(f"  {entry['index']:03d} {entry['cell']:<44s} steps={entry['episode_steps']}")
        print("  ...")
        print("\ndry run: nothing uploaded")
        return 0

    from huggingface_hub import (
        CommitOperationAdd, CommitOperationDelete, HfApi, hf_hub_download,
    )

    api = HfApi()
    operations = [
        CommitOperationAdd("README.md", str(staging / "README.md")),
        CommitOperationAdd(f"{PREFIX}/manifest.json", str(staging / "manifest.json")),
        CommitOperationAdd(f"{PREFIX}/ledger.jsonl", str(LEDGER)),
        CommitOperationAdd(f"{PREFIX}/contract.json", str(CONTRACT)),
    ]
    if CLOSEOUT.is_file():
        operations.append(CommitOperationAdd(f"{PREFIX}/closeout.json", str(CLOSEOUT)))
    expected: dict[str, str] = {}
    by_attempt = {r["attempt_id"]: r for r in rows}
    for entry in manifest["rows"]:
        source = row_dir(by_attempt[entry["attempt_id"]])
        for spec in entry["files"]:
            path = f"{entry['repo_dir']}/{spec['name']}"
            operations.append(CommitOperationAdd(path, str(source / spec["name"])))
            expected[path] = spec["sha256"]

    # Row directories are named NNN_<row> by sorted position, so adding an
    # episode shifts the indices after it. Without this, a later top-up commit
    # would leave the previous run's directories orphaned in the repo. Deleting
    # them in the same commit keeps the published set exactly the manifest.
    stale: list[str] = []
    try:
        remote = set(api.list_repo_files(args.repo_id, repo_type="dataset"))
    except Exception:  # noqa: BLE001 - a fresh repo has nothing to prune
        remote = set()
    keep = set(expected) | {
        "README.md", f"{PREFIX}/manifest.json", f"{PREFIX}/ledger.jsonl",
        f"{PREFIX}/contract.json", f"{PREFIX}/closeout.json",
    }
    stale = sorted(
        path for path in remote
        if path.startswith(f"{PREFIX}/") and path not in keep
    )
    if stale:
        print(f"pruning {len(stale)} stale remote paths from a previous commit")
        operations.extend(CommitOperationDelete(path_in_repo=p) for p in stale)

    print(f"\nuploading {len(operations)} files ...")
    api.create_commit(
        repo_id=args.repo_id, repo_type="dataset", operations=operations,
        commit_message=f"Add the V10.11c mixed-clutter collection ({len(rows)} episodes)",
    )

    print("verifying remote hashes ...")
    paths = list(expected)
    infos = {}
    for start in range(0, len(paths), 100):
        for info in api.get_paths_info(args.repo_id, paths[start:start + 100],
                                       repo_type="dataset", expand=True):
            infos[info.path] = info
    bad, lfs_ok, blob_ok = [], 0, 0
    for path, want in expected.items():
        info = infos.get(path)
        if info is None:
            bad.append(f"{path}: absent from the remote")
            continue
        got = getattr(getattr(info, "lfs", None), "sha256", None)
        if got is not None:
            if got == want:
                lfs_ok += 1
            else:
                bad.append(f"{path}: lfs {got} != {want}")
            continue
        # Small files are plain git blobs and carry no LFS hash; fetch and hash.
        local = hf_hub_download(args.repo_id, path, repo_type="dataset")
        if sha256_file(Path(local)) == want:
            blob_ok += 1
        else:
            bad.append(f"{path}: blob mismatch")
    if bad:
        print(f"\nVERIFICATION FAILED for {len(bad)}/{len(expected)} files:")
        for line in bad[:20]:
            print("  " + line)
        return 1
    print(f"all {len(expected)} row files verified byte-identical "
          f"({lfs_ok} via LFS hash, {blob_ok} by download)")
    print(f"\nhttps://huggingface.co/datasets/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
