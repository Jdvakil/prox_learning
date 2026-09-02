#!/usr/bin/env python3
"""Publish the V10.10 table-camera validation set to a HuggingFace dataset repo.

The point of this ten-episode set is that a collaborator can read a fixed
exterior table camera out of our trajectories: an ``exo_camera_1`` RGB MP4 plus
``extrinsic_cv`` / ``cam2world_gl`` / ``intrinsic_cv`` under
``traj_0/obs/sensor_param/exo_camera_1``. So the upload carries the per-row
camera validation verbatim rather than just the media, and every uploaded file
is re-verified against the remote object hash after the commit -- a byte
comparison, not a "the call returned 200" comparison.
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
RUN_DIR = ROOT / "diagnostics_output/pact_place_v1010_tablecam_validation10"
LEDGER = RUN_DIR / "ledger.jsonl"
CONTRACT = RUN_DIR / "contract.json"
REPO_ID = "Lundii/table_smoke"
PREFIX = "pact_place_corridor_v10_10_tablecam_validation10"
REQUIRED_MP4 = "exo_camera_1.mp4"
CALIBRATION_KEYS = ("extrinsic_cv", "cam2world_gl", "intrinsic_cv")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def accepted_rows() -> list[dict[str, Any]]:
    records = [json.loads(line) for line in LEDGER.read_text().splitlines() if line.strip()]
    return [r for r in records if r.get("accepted")]


def check(rows: list[dict[str, Any]], contract: dict[str, Any], allow_partial: bool) -> None:
    """Refuse to publish a set that does not actually carry the camera."""
    targets = list(contract["target_cells"])
    have = [r["cell"] for r in rows]
    missing = [t for t in targets if t not in have]
    if missing and not allow_partial:
        raise SystemExit(
            f"{len(rows)}/{len(targets)} accepted; still pending {missing}. "
            "Pass --allow-partial to publish an incomplete set."
        )
    if len(set(have)) != len(have):
        raise SystemExit(f"duplicate cells among accepted rows: {sorted(have)}")
    problems: list[str] = []
    for row in rows:
        row_dir = Path(row["row_dir"])
        validation = row.get("table_camera_validation") or {}
        detail = validation.get("detail") or {}
        if not validation.get("passed"):
            problems.append(f"{row['cell']}: validation not passed")
        if not any(p.name.endswith(REQUIRED_MP4) for p in row_dir.glob("*.mp4")):
            problems.append(f"{row['cell']}: no {REQUIRED_MP4}")
        for key in CALIBRATION_KEYS:
            delta = detail.get(f"{key}_within_episode_max_delta")
            if delta is None:
                problems.append(f"{row['cell']}: {key} absent")
            elif float(delta) != 0.0:
                problems.append(f"{row['cell']}: {key} moves within the episode ({delta})")
        if not Path(row["trajectory_h5"]).is_file():
            problems.append(f"{row['cell']}: trajectory.h5 missing")
    if problems:
        raise SystemExit("refusing to publish:\n  " + "\n  ".join(problems))


def camera_world_pose(h5_path: str) -> dict[str, Any]:
    """Where the fixed camera actually sits, in world coordinates.

    The validation only asserts the pose is constant *within* an episode. It is
    re-drawn *between* episodes, which a consumer has to know: reading one
    episode's extrinsic and reusing it for the rest would silently misproject
    every other episode.
    """
    import h5py
    import numpy as np

    group = "traj_0/obs/sensor_param/exo_camera_1"
    with h5py.File(h5_path, "r") as handle:
        extrinsic = handle[f"{group}/extrinsic_cv"][0]
        intrinsic = handle[f"{group}/intrinsic_cv"][0]
    rotation, translation = extrinsic[:3, :3], extrinsic[:3, 3]
    position = -rotation.T @ translation
    return {
        "position_world_m": [round(float(v), 6) for v in position],
        "height_m": round(float(position[2]), 6),
        "ground_radius_m": round(float(np.hypot(position[0], position[1])), 6),
        "azimuth_deg": round(float(np.degrees(np.arctan2(position[1], position[0]))), 4),
        "intrinsic_cv_0": [[round(float(v), 4) for v in row] for row in intrinsic],
    }


def build_manifest(rows: list[dict[str, Any]], contract: dict[str, Any]) -> dict[str, Any]:
    entries = []
    order = {cell: i for i, cell in enumerate(contract["target_cells"])}
    for row in sorted(rows, key=lambda r: order[r["cell"]]):
        row_dir = Path(row["row_dir"])
        index = order[row["cell"]]
        files = sorted(p for p in row_dir.iterdir() if p.is_file())
        entries.append({
            "index": index,
            "repo_dir": f"{PREFIX}/rows/{index:03d}_{row_dir.name}",
            "cell": row["cell"],
            "family_id": row["family_id"],
            "intrusion_side": row["intrusion_side"],
            "pose_id": row["pose_id"],
            "attempt_id": row["attempt_id"],
            "attempt_index": int(row["attempt_index"]),
            "task_seed_u32": int(row["task_seed_u32"]),
            "episode_steps": int(row["episode_steps"]),
            "task_success": bool(row["task_success"]),
            "defects": row.get("defects") or [],
            "table_camera_validation": row.get("table_camera_validation"),
            "camera_pose": camera_world_pose(row["trajectory_h5"]),
            "files": [{"name": p.name, "bytes": p.stat().st_size,
                       "sha256": sha256_file(p)} for p in files],
        })
    return {
        "schema_version": "pact_place_v1010_tablecam_validation10_publication_v1",
        "environment_version": contract["environment_version"],
        "sampler_class": contract["sampler_class"],
        "camera_system": contract["camera_system"],
        "added_camera": contract["added_camera"],
        "required_h5_group": contract["required_h5_group"],
        "required_calibration_keys": list(contract["required_calibration_keys"]),
        "source_contract_sha256": contract["payload_sha256"],
        "base_ledger_sha256": contract["base_ledger_sha256"],
        "training_corpus_modified": contract["training_corpus_modified"],
        "n_rows": len(entries),
        "camera_geometry": camera_geometry(entries),
        "rows": entries,
    }


def camera_geometry(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarise how the camera pose varies across the ten episodes."""
    poses = [e["camera_pose"] for e in entries]
    heights = {p["height_m"] for p in poses}
    radii = [p["ground_radius_m"] for p in poses]
    azimuths = sorted(p["azimuth_deg"] for p in poses)
    intrinsics = {json.dumps(p["intrinsic_cv_0"]) for p in poses}
    return {
        "fixed_within_episode": True,
        "resampled_between_episodes": len({tuple(p["position_world_m"]) for p in poses}) > 1,
        "height_m": sorted(heights),
        "ground_radius_m_min_max": [min(radii), max(radii)],
        "ground_radius_m_spread": round(max(radii) - min(radii), 6),
        "azimuth_deg_min_max": [azimuths[0], azimuths[-1]],
        "intrinsics_shared_across_episodes": len(intrinsics) == 1,
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

# V10.10 table-camera validation set

Ten pick-and-place demonstrations from the `{env}` environment, re-run with one
extra **fixed exterior table camera** (`{cam}`) so the camera schema can be
validated end to end before it is used at scale.

This is a *schema* validation set, not a training corpus. The 144-row V10.10
training corpus is unchanged and is not part of this repo.

## What each row guarantees

Every published row was checked before upload and carries:

- `episode_00000000_{cam}.mp4` — decodable exterior RGB, frame count equal to the
  trajectory's frame count;
- `{group}` in `trajectory.h5`, holding `extrinsic_cv` `(T,3,4)`,
  `cam2world_gl` `(T,4,4)` and `intrinsic_cv` `(T,3,3)`, all finite `float64`;
- **a genuinely fixed pose** — the within-episode max delta of all three
  calibration arrays is exactly `0.0`, so the camera does not drift across the
  episode.

## Read this before you reuse an extrinsic

The camera is fixed *within* an episode but **re-drawn between episodes**. All
ten sit at the same height on a near-constant ground radius, at a different
azimuth each time, and they share one intrinsic matrix. Concretely:

{geometry}

So load the extrinsic from the episode you are projecting. Reading one episode's
`extrinsic_cv` and reusing it across the set will silently misproject the other
nine — the intrinsics match, so nothing will look obviously wrong.

`manifest.json` records those checks per row, along with a SHA-256 for every
uploaded file. Each row is also a *clean* success under the collection's own
predicate: the task succeeded and no contact was made with clutter, the hazard
bar, mounted fixtures or the rest of the environment.

## Layout

```
{prefix}/
  manifest.json        per-row metadata, camera validation and file hashes
  ledger.jsonl         the full run ledger, including rejected attempts
  contract.json        the frozen run contract
  rows/NNN_<row>/      one directory per accepted episode
```

Per-row files: `trajectory.h5`, `result.json`, and MP4s for the exterior camera
(RGB + depth), the wrist camera (RGB + depth) and the proximity-sensor heatmap.

## Coverage

Ten cells spanning four layout families, both intrusion sides and all three
pendant poses. Cells and seeds are listed in `manifest.json`.

## One inconsistency worth naming

{oddity}
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    contract = json.loads(CONTRACT.read_text())
    rows = accepted_rows()
    check(rows, contract, args.allow_partial)
    manifest = build_manifest(rows, contract)

    staging = Path(os.environ.get("TMPDIR", "/tmp")) / "tablecam_publish"
    staging.mkdir(parents=True, exist_ok=True)
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    # The smoke row predates per-frame contact-audit summarisation, so its
    # result.json is orders of magnitude larger. Name the row that is actually
    # affected instead of assuming it is the first one.
    fat = [e for e in manifest["rows"]
           for f in e["files"]
           if f["name"] == "result.json" and f["bytes"] > 5_000_000]
    if fat:
        oddity = (
            "`result.json` in "
            + ", ".join(f"row `{e['index']:03d}` ({e['cell']})" for e in fat)
            + " is far larger than the others (~73 MB vs ~11 KB): that row was\n"
            "collected before per-frame contact-audit summarisation was enabled.\n"
            "The trajectories, videos and calibration are unaffected."
        )
    else:
        oddity = "None: every row carries the summarised `result.json`."
    geom = manifest["camera_geometry"]
    geometry = (
        f"- height: **{geom['height_m'][0]:.3f} m** (identical on all ten)\n"
        f"- ground radius: **{geom['ground_radius_m_min_max'][0]:.3f}"
        f"–{geom['ground_radius_m_min_max'][1]:.3f} m** "
        f"(spread {geom['ground_radius_m_spread']*1000:.0f} mm)\n"
        f"- azimuth: **{geom['azimuth_deg_min_max'][0]:.1f}° to "
        f"{geom['azimuth_deg_min_max'][1]:.1f}°**\n"
        f"- intrinsics shared across episodes: "
        f"**{'yes' if geom['intrinsics_shared_across_episodes'] else 'no'}**"
    )
    (staging / "README.md").write_text(README.format(
        env=contract["environment_version"], cam=contract["added_camera"],
        group=contract["required_h5_group"], prefix=PREFIX, oddity=oddity,
        geometry=geometry))

    print(f"{len(rows)} accepted rows -> {args.repo_id}/{PREFIX}")
    total = sum(f["bytes"] for e in manifest["rows"] for f in e["files"])
    print(f"payload: {total/1e6:.0f} MB across "
          f"{sum(len(e['files']) for e in manifest['rows'])} row files")
    for entry in manifest["rows"]:
        print(f"  {entry['index']:03d} {entry['cell']:<44s} "
              f"steps={entry['episode_steps']:<4d} "
              f"{sum(f['bytes'] for f in entry['files'])/1e6:6.1f} MB")
    if args.dry_run:
        print("\ndry run: nothing uploaded")
        return 0

    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi()
    operations = [
        CommitOperationAdd("README.md", str(staging / "README.md")),
        CommitOperationAdd(f"{PREFIX}/manifest.json", str(staging / "manifest.json")),
        CommitOperationAdd(f"{PREFIX}/ledger.jsonl", str(LEDGER)),
        CommitOperationAdd(f"{PREFIX}/contract.json", str(CONTRACT)),
        CommitOperationAdd(f"{PREFIX}/closeout.json", str(RUN_DIR / "closeout.json")),
    ]
    expected: dict[str, str] = {}
    for entry in manifest["rows"]:
        source = Path(next(r for r in rows if r["attempt_id"] == entry["attempt_id"])["row_dir"])
        for spec in entry["files"]:
            path = f"{entry['repo_dir']}/{spec['name']}"
            operations.append(CommitOperationAdd(path, str(source / spec["name"])))
            expected[path] = spec["sha256"]

    print(f"\nuploading {len(operations)} files ...")
    api.create_commit(
        repo_id=args.repo_id, repo_type="dataset", operations=operations,
        commit_message=f"Add V10.10 table-camera validation set ({len(rows)} episodes)",
    )

    # Verify against the remote object hashes rather than trusting the commit.
    print("verifying remote hashes ...")
    paths = list(expected)
    infos = {}
    for start in range(0, len(paths), 100):
        for info in api.get_paths_info(args.repo_id, paths[start:start + 100],
                                       repo_type="dataset", expand=True):
            infos[info.path] = info
    # Small files are committed as plain git blobs and carry no LFS hash, so
    # they have to be fetched and hashed. Treating a missing LFS hash as a
    # failure would flag every one of them; treating it as a pass would leave
    # them unverified.
    from huggingface_hub import hf_hub_download

    bad = []
    lfs_verified = blob_verified = 0
    for path, want in expected.items():
        info = infos.get(path)
        if info is None:
            bad.append(f"{path}: absent from the remote")
            continue
        got = getattr(getattr(info, "lfs", None), "sha256", None)
        if got is not None:
            if got == want:
                lfs_verified += 1
            else:
                bad.append(f"{path}: lfs {got} != {want}")
            continue
        local = hf_hub_download(args.repo_id, path, repo_type="dataset")
        got = sha256_file(Path(local))
        if got == want:
            blob_verified += 1
        else:
            bad.append(f"{path}: blob {got} != {want}")
    if bad:
        print(f"\nVERIFICATION FAILED for {len(bad)}/{len(expected)} files:")
        for line in bad[:20]:
            print("  " + line)
        return 1
    print(f"all {len(expected)} row files verified byte-identical on the remote "
          f"({lfs_verified} via LFS hash, {blob_verified} by download)")
    print(f"\nhttps://huggingface.co/datasets/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
