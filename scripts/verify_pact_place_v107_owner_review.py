#!/usr/bin/env python3
"""Independent verifier for the V10.7 owner visual-review packet.

Recomputes everything the manifest claims rather than reading its booleans:
raw file hashes, the source result and trajectory hashes, the scene and
assembly hashes, decoded frame counts and durations, the selection constraints,
and the authorization fields. Also re-derives the deterministic selection from
``pool.json`` and requires it to equal what was published.

Read-only. It renders nothing and writes nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "scripts", ROOT / "submodules" / "molmospaces"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from pact_place_v107_contract import (  # noqa: E402
    POOL_MIN_CLEAN,
    POOL_ROOT,
    REVIEW_FPS,
    recompute_payload_sha256,
    sha256_file,
)
from run_pact_place_v107_owner_review import (  # noqa: E402
    OWNER_REVIEW_ROOT,
    candidate_rows,
    select_six,
    verify_selection,
)

AUTHORIZATION_MUST_BE_FALSE = (
    "authorizes_phase0", "authorizes_gate", "authorizes_collection",
    "authorizes_conversion", "authorizes_training", "authorizes_evaluation",
    "phase0_passed", "human_approval_present", "eligible_for_human_review",
    "authorizes_downstream_work",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / OWNER_REVIEW_ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest_path = root / "review_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"no packet at {root}")
    manifest = json.loads(manifest_path.read_text())
    problems: list[str] = []

    # 1. the manifest is self-consistent
    if manifest.get("payload_sha256") != recompute_payload_sha256(manifest_path):
        problems.append("review_manifest.json self-hash mismatch")

    # 2. every bound upstream artifact still matches
    for name, entry in manifest["bound_inputs"].items():
        path = ROOT / entry["path"]
        if not path.is_file():
            problems.append(f"bound input missing: {name}")
            continue
        if sha256_file(path) != entry["raw_file_sha256"]:
            problems.append(f"bound input raw hash drifted: {name}")
        if recompute_payload_sha256(path) != entry["payload_sha256"]:
            problems.append(f"bound input payload hash drifted: {name}")

    # 3. scenes and assemblies
    for pose, entry in manifest["scenes"].items():
        path = ROOT / entry["relative"]
        if not path.is_file() or sha256_file(path) != entry["scene_sha256"]:
            problems.append(f"scene drifted: {pose}")

    # 4. the pool really did fail, and is not reinterpreted
    pool = json.loads((ROOT / POOL_ROOT / "pool.json").read_text())
    if pool["eligibility"]["pool_passed"]:
        problems.append("pool.json reports a pass; this packet assumes failure")
    if manifest.get("pool_passed") is not False:
        problems.append("manifest does not carry pool_passed=false")
    if int(manifest["pool_clean_successes"]) != int(
        pool["eligibility"]["clean_successes"]
    ):
        problems.append("manifest pool count disagrees with pool.json")
    if int(manifest["pool_clean_successes"]) >= POOL_MIN_CLEAN:
        problems.append("manifest claims a clean count at or above the floor")

    # 5. selection re-derived from pool.json
    plan = select_six(candidate_rows(pool))
    if not plan["found"]:
        problems.append("selection could not be re-derived")
    else:
        checks = verify_selection(plan)
        if not checks["passed"]:
            problems.append(f"re-derived selection fails: {checks['checks']}")
        derived_success = [r["role_index"] for r in plan["successes"]]
        derived_failure = [r["role_index"] for r in plan["failures"]]
        if derived_success != manifest["selected_success_roles"]:
            problems.append(
                f"success roles differ: {derived_success} vs "
                f"{manifest['selected_success_roles']}")
        if derived_failure != manifest["selected_failure_roles"]:
            problems.append(
                f"failure roles differ: {derived_failure} vs "
                f"{manifest['selected_failure_roles']}")

    # 6. videos: present, nonempty, decodable, reconciled
    import cv2

    video_reports: list[dict[str, Any]] = []
    videos_dir = root / "videos"
    published = sorted(videos_dir.glob("*.mp4")) if videos_dir.is_dir() else []
    if len(published) != int(manifest["n_videos"]):
        problems.append(
            f"{len(published)} MP4 files on disk, manifest says "
            f"{manifest['n_videos']}")
    for record in manifest["videos"]:
        path = videos_dir / record["video_name"]
        report: dict[str, Any] = {"video": record["video_name"]}
        if not path.is_file():
            problems.append(f"missing video: {record['video_name']}")
            video_reports.append({**report, "present": False})
            continue
        size = int(path.stat().st_size)
        digest = sha256_file(path)
        capture = cv2.VideoCapture(str(path))
        try:
            opened = capture.isOpened()
            fps = float(capture.get(cv2.CAP_PROP_FPS)) if opened else 0.0
            counted = 0
            while opened:
                ok, _ = capture.read()
                if not ok:
                    break
                counted += 1
        finally:
            capture.release()
        report.update({
            "present": True, "nonempty": size > 0, "size_bytes": size,
            "decodable": opened, "decoded_frames": counted, "decoded_fps": fps,
            "decoded_duration_s": counted / fps if fps > 0 else None,
            "hash_matches": digest == record["video_raw_file_sha256"],
            "frames_match_manifest": counted == record["n_frames_rendered"],
            "frames_match_retained_trajectory": counted == record[
                "retained_trajectory_n"],
            "fps_matches": abs(fps - float(REVIEW_FPS)) <= 0.5,
            "duration_matches": abs(
                (counted / fps if fps > 0 else -1.0) - record["duration_s"]
            ) <= 0.25,
        })
        for key in ("nonempty", "decodable", "hash_matches",
                    "frames_match_manifest",
                    "frames_match_retained_trajectory", "fps_matches",
                    "duration_matches"):
            if not report[key]:
                problems.append(f"{record['video_name']}: {key} failed")
        video_reports.append(report)

    # 7. source result/trajectory hashes still match
    for record in manifest["videos"]:
        role = int(record["role_index"])
        source = next(
            (r for r in candidate_rows(pool) if r["role_index"] == role), None)
        if source is None:
            problems.append(f"role {role} is no longer a valid candidate")
            continue
        if source["result_raw_file_sha256"] != record["source_result_sha256"]:
            problems.append(f"role {role}: result.json drifted")
        if source["trajectory_raw_file_sha256"] != record[
            "source_trajectory_sha256"
        ]:
            problems.append(f"role {role}: trajectory.json drifted")

    # 8. failures are natural
    for record in manifest["videos"]:
        if record["clean"]:
            continue
        if int(record["pendant_contact_frames_in_replay"]) != 0:
            problems.append(
                f"role {record['role_index']}: failure shows pendant contact")

    # 9. authorization
    for field in AUTHORIZATION_MUST_BE_FALSE:
        if manifest.get(field) is not False:
            problems.append(f"authorization field not false: {field}")
    if manifest.get("eligible_for_owner_visual_review") is not True:
        problems.append("eligible_for_owner_visual_review is not true")
    if (root / "human_approval.json").exists():
        problems.append("human_approval.json exists")

    # 10. review text carries the required statements
    review = (root / "REVIEW.md").read_text() if (root / "REVIEW.md").is_file() else ""
    for phrase in ("FAILED", "solely for owner visual assessment",
                   "does not make the pool pass",
                   "does not\n> authorize any downstream work"):
        if phrase.replace("\n> ", " ") not in review.replace("\n> ", " "):
            problems.append(f"REVIEW.md is missing a required statement: {phrase!r}")

    print(json.dumps({
        "verified": not problems,
        "n_problems": len(problems),
        "problems": problems[:12],
        "n_videos": len(published),
        "pool_clean_successes": manifest["pool_clean_successes"],
        "pool_passed": manifest["pool_passed"],
        "eligible_for_owner_visual_review": manifest[
            "eligible_for_owner_visual_review"],
        "selection_rederived_matches": (
            plan.get("found")
            and [r["role_index"] for r in plan["successes"]]
            == manifest["selected_success_roles"]
            and [r["role_index"] for r in plan["failures"]]
            == manifest["selected_failure_roles"]
        ),
        "videos": video_reports,
    }, indent=2))
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
