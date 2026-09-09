#!/usr/bin/env python3
"""V10.7 Phase-0 gate: the frozen 24-row manifest, executed at most once.

Runs only against an owner-authored ``human_approval.json`` that binds the exact
review manifest and all six reviewed MP4 hashes. The verifier recomputes every
binding from file bytes rather than trusting an embedded value, and refuses a
record marked ``created_by_agent``.

The agent does not create the approval record. That is not a formality: an
approval this program wrote would either be rejected here for being
agent-created, or would have to carry a false ``created_by_agent: false``.

Once approved, the frozen 24 rows execute exactly once, in registered order,
with no cherry-picking, no replacement row, no threshold relaxation, and no
scientific retry. A worker that raises is recorded as an incomplete
infrastructure failure against its official row; the run always closes out with
an immutable ``gate.json``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import os
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "scripts", ROOT / "submodules" / "molmospaces"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from pact_place_v107_contract import (  # noqa: E402
    CAUSAL_ROOT,
    CERT_ROOT,
    CONTRACT_VERSION_V107,
    ENVIRONMENT_VERSION,
    INTRUSION_SIDES,
    N_PHASE0_ROWS,
    PHASE0_MIN_CLEAN,
    PHASE0_MIN_CLEAN_PER_POSE,
    PHASE0_MIN_CLEAN_PER_SIDE,
    PHASE0_MIN_CLEAN_PER_SIDE_POSE,
    PHASE0_ROOT,
    PLAN_RELATIVE,
    POOL_ROOT,
    POSE_IDS,
    SELECTION_ROOT,
    SPEC_ROOT,
    V95_LAYOUT_FAMILY_IDS,
    empty_authorization,
    group_key,
    phase0_rows,
    recompute_payload_sha256,
    row_defects,
    sha256_file,
    sha256_payload,
    verify_against_specification,
    wilson_interval,
    write_immutable_create_only,
)
# The live-risk confirmation is inherited, not invented here: it is the same
# 35 mm bound the V10.5 and V10.6 contracts registered, applied per side and
# per pose. Importing it keeps that lineage explicit.
from pact_place_v106_contract import PHASE0_RISK_CONFIRM_MAX_M  # noqa: E402

OWNER_REVIEW_ROOT = "diagnostics_output/pact_place_v107_owner_review"
GATE_IMPLEMENTATION_PATHS = (
    "scripts/run_pact_place_v107_phase0.py",
    "scripts/pact_place_v107_contract.py",
    "scripts/pact_place_v106_contract.py",
    "scripts/pact_place_v106_geometry.py",
    "scripts/pact_place_v105_clearance.py",
    "submodules/molmospaces/molmo_spaces/tasks/enclosure_reach.py",
    "scripts/run_pact_place_expert_screen.py",
)
INFRASTRUCTURE_STATUS = "infrastructure_failure"


class ApprovalError(PermissionError):
    """The owner record is missing, stale, partial, or agent-created."""


class IntegrityError(RuntimeError):
    """A bound input drifted before or after execution."""


def gate_implementation_digest() -> str:
    return sha256_payload(
        [[p, sha256_file(ROOT / p)] for p in GATE_IMPLEMENTATION_PATHS]
    )


# ---------------------------------------------------------------------------
# Approval
# ---------------------------------------------------------------------------
def expected_bindings(review_root: Path) -> dict[str, str]:
    """Every binding the owner must sign, recomputed from file bytes."""
    manifest_path = review_root / "review_manifest.json"
    bindings = {
        "contract_version": CONTRACT_VERSION_V107,
        "environment_version": ENVIRONMENT_VERSION,
        "review_manifest_payload_sha256": recompute_payload_sha256(manifest_path),
        "review_manifest_raw_file_sha256": sha256_file(manifest_path),
        "selection_payload_sha256": recompute_payload_sha256(
            ROOT / SELECTION_ROOT / "selection.json"),
        "certification_payload_sha256": recompute_payload_sha256(
            ROOT / CERT_ROOT / "certification.json"),
        "causal_payload_sha256": recompute_payload_sha256(
            ROOT / CAUSAL_ROOT / "causal.json"),
        "pool_payload_sha256": recompute_payload_sha256(
            ROOT / POOL_ROOT / "pool.json"),
        "gate_implementation_digest": gate_implementation_digest(),
    }
    for video in sorted((review_root / "videos").glob("*.mp4")):
        bindings[f"video_sha256:{video.name}"] = sha256_file(video)
    return bindings


def assert_approval(approval, expected, video_names) -> None:
    if not approval:
        raise ApprovalError(
            "Phase 0 requires an owner-authored human_approval.json")
    if approval.get("decision") != "approve_phase0":
        raise ApprovalError(f"decision={approval.get('decision')!r}")
    if approval.get("created_by_agent"):
        raise ApprovalError("refusing an agent-created approval record")
    if approval.get("created_by") not in ("owner", "human"):
        raise ApprovalError(f"created_by={approval.get('created_by')!r}")
    missing = sorted(k for k in expected if k not in approval)
    if missing:
        raise ApprovalError(f"approval is missing bindings: {missing}")
    for key, digest in expected.items():
        if approval.get(key) != digest:
            raise ApprovalError(
                f"binding is stale for {key}: {approval.get(key)!r} != {digest!r}")
    reviewed = approval.get("reviewed_videos")
    if not isinstance(reviewed, list):
        raise ApprovalError("approval must list the reviewed videos")
    if sorted(reviewed) != sorted(video_names):
        raise ApprovalError(
            f"video inventory wrong: extra={sorted(set(reviewed) - set(video_names))} "
            f"missing={sorted(set(video_names) - set(reviewed))}")


# ---------------------------------------------------------------------------
# Integrity, run before and after execution
# ---------------------------------------------------------------------------
def integrity_report(review_root: Path, rows=None, approval_path=None,
                     phase: str = "pre_run") -> dict[str, Any]:
    """Recompute the packet, specification, scenes, runner, manifest and bindings."""
    problems: list[str] = []
    manifest_path = review_root / "review_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("payload_sha256") != recompute_payload_sha256(manifest_path):
        problems.append("review_manifest.json self-hash mismatch")
    if manifest.get("pool_passed") is not False:
        problems.append("review manifest does not carry pool_passed=false")

    videos: dict[str, str] = {}
    for name, digest in (manifest.get("video_raw_file_sha256") or {}).items():
        path = review_root / "videos" / name
        if not path.is_file():
            problems.append(f"missing reviewed video: {name}")
            continue
        observed = sha256_file(path)
        videos[name] = observed
        if observed != digest:
            problems.append(f"reviewed video drifted: {name}")
    published = sorted(p.name for p in (review_root / "videos").glob("*.mp4"))
    if len(published) != int(manifest.get("n_videos", -1)):
        problems.append("published video count disagrees with the manifest")

    spec_path = ROOT / SPEC_ROOT / "specification.json"
    spec = json.loads(spec_path.read_text())
    drift = verify_against_specification(spec)
    code_drift = [d for d in drift["drift"] if d["path"] != PLAN_RELATIVE]
    doc_drift = [d for d in drift["drift"] if d["path"] == PLAN_RELATIVE]
    if code_drift:
        problems.append(
            f"code/data drift: {[d['path'] for d in code_drift]}")

    certification = json.loads((ROOT / CERT_ROOT / "certification.json").read_text())
    scenes: dict[str, str] = {}
    for pose in POSE_IDS:
        entry = certification["published_scenes"][pose]
        path = ROOT / entry["relative"]
        if not path.is_file():
            problems.append(f"missing scene: {pose}")
            continue
        observed = sha256_file(path)
        scenes[pose] = observed
        if observed != entry["sha256"]:
            problems.append(f"scene drifted: {pose}")

    manifest_checks: dict[str, Any] = {}
    if rows is not None:
        pool = json.loads((ROOT / POOL_ROOT / "pool.json").read_text())
        pool_ids = {r["episode_id"] for r in pool["rows"]}
        ids = [r["episode_id"] for r in rows]
        manifest_checks = {
            "n_rows": len(rows),
            "n_rows_correct": len(rows) == N_PHASE0_ROWS,
            "episode_ids_unique": len(set(ids)) == len(ids),
            "disjoint_from_pool": not (set(ids) & pool_ids),
            "row_sha256_present": all(r.get("row_sha256") for r in rows),
            "balance": {
                "by_side": {s: sum(1 for r in rows if r["intrusion_side"] == s)
                            for s in INTRUSION_SIDES},
                "by_pose": {p: sum(1 for r in rows if r["pose_id"] == p)
                            for p in POSE_IDS},
            },
        }
        for name, ok in (("n_rows_correct", manifest_checks["n_rows_correct"]),
                         ("episode_ids_unique", manifest_checks["episode_ids_unique"]),
                         ("disjoint_from_pool", manifest_checks["disjoint_from_pool"]),
                         ("row_sha256_present", manifest_checks["row_sha256_present"])):
            if not ok:
                problems.append(f"manifest check failed: {name}")

    approval_state: dict[str, Any] = {"present": False}
    if approval_path is not None and Path(approval_path).is_file():
        approval_state = {
            "present": True,
            "path": str(approval_path),
            "raw_file_sha256": sha256_file(Path(approval_path)),
        }

    return {
        "phase": phase,
        "review_manifest_payload_sha256": recompute_payload_sha256(manifest_path),
        "review_manifest_raw_file_sha256": sha256_file(manifest_path),
        "reviewed_video_sha256": videos,
        "n_reviewed_videos": len(published),
        "specification_payload_sha256": recompute_payload_sha256(spec_path),
        "specification_files_checked": drift["n_checked"],
        "code_and_data_drift": code_drift,
        "documentation_drift": doc_drift,
        "documentation_drift_is_expected": True,
        "scene_sha256": scenes,
        "gate_implementation_digest": gate_implementation_digest(),
        "gate_implementation_files": {
            p: sha256_file(ROOT / p) for p in GATE_IMPLEMENTATION_PATHS},
        "thresholds": {
            "min_clean": PHASE0_MIN_CLEAN,
            "min_clean_per_side": PHASE0_MIN_CLEAN_PER_SIDE,
            "min_clean_per_pose": PHASE0_MIN_CLEAN_PER_POSE,
            "min_clean_per_side_pose": PHASE0_MIN_CLEAN_PER_SIDE_POSE,
            "risk_confirmation_max_m": PHASE0_RISK_CONFIRM_MAX_M,
            "n_rows": N_PHASE0_ROWS,
        },
        "manifest_checks": manifest_checks,
        "approval": approval_state,
        "approval_bindings": expected_bindings(review_root),
        "problems": problems,
        "passed": not problems,
    }


# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------
def _summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "min": float(ordered[0]),
        "p25": float(ordered[max(0, len(ordered) // 4 - 1)]),
        "median": float(statistics.median(ordered)),
        "p75": float(ordered[min(len(ordered) - 1, (3 * len(ordered)) // 4)]),
        "max": float(ordered[-1]),
        "mean": float(statistics.fmean(ordered)),
    }


def distributions(rows, results, output_root: Path) -> dict[str, Any]:
    """Episode length and commanded/realized speed, as the inherited plan asks."""
    from run_pact_place_expert_screen import _result_path

    by_role = {int(r["role_index"]): r for r in rows}
    steps: list[float] = []
    commanded: list[float] = []
    realized: list[float] = []
    per_row: list[dict[str, Any]] = []
    for result in results:
        role = int(result["role_index"])
        telemetry = result.get("pact_v106_frame_telemetry") or {}
        n_steps = result.get("episode_steps")
        if n_steps:
            steps.append(float(n_steps))
        row_commanded: list[float] = []
        row_realized: list[float] = []
        row = by_role.get(role)
        if row is not None:
            trajectory = _result_path(output_root, row).parent / "trajectory.json"
            if trajectory.is_file():
                try:
                    for step in json.loads(trajectory.read_text())["steps"]:
                        value = step.get("commanded_speed_m_s")
                        if value is not None:
                            row_commanded.append(float(value))
                        value = step.get("realized_tcp_speed_m_s")
                        if value is not None:
                            row_realized.append(float(value))
                except Exception:  # noqa: BLE001 - a distribution must not fail a gate
                    pass
        commanded.extend(row_commanded)
        realized.extend(row_realized)
        per_row.append({
            "role_index": role,
            "episode_steps": n_steps,
            "max_realized_tcp_speed_m_s": telemetry.get(
                "max_realized_tcp_speed_m_s"),
            "n_commanded_samples": len(row_commanded),
            "n_realized_samples": len(row_realized),
        })
    return {
        "episode_steps": _summary(steps),
        "commanded_tcp_speed_m_s": _summary(commanded),
        "realized_tcp_speed_m_s": _summary(realized),
        "per_row": per_row,
        "source": "retained per-frame trajectories of the official rows",
    }


# ---------------------------------------------------------------------------
# Gate accounting
# ---------------------------------------------------------------------------
def gate_eligibility(rows, results) -> dict[str, Any]:
    """16/24, every balance floor, zero pendant contact, and live-risk confirmation."""
    by_role = {int(r["role_index"]): r for r in results}
    clean = 0
    by_side = dict.fromkeys(INTRUSION_SIDES, 0)
    by_pose = dict.fromkeys(POSE_IDS, 0)
    by_cell = {group_key(p, s): 0 for p in POSE_IDS for s in INTRUSION_SIDES}
    by_family = dict.fromkeys(V95_LAYOUT_FAMILY_IDS, 0)
    n_side = dict.fromkeys(INTRUSION_SIDES, 0)
    n_pose = dict.fromkeys(POSE_IDS, 0)
    n_cell = dict.fromkeys(by_cell, 0)
    incomplete = 0
    infrastructure = 0
    pendant_contact_rows = 0
    taxonomy: dict[str, int] = {}
    table: list[dict[str, Any]] = []
    minimum = None
    risk_side: dict[str, dict[str, Any]] = {}
    risk_pose: dict[str, dict[str, Any]] = {}

    for row in rows:
        side, pose, family = row["intrusion_side"], row["pose_id"], row["family_id"]
        role = int(row["role_index"])
        n_side[side] += 1
        n_pose[pose] += 1
        n_cell[group_key(pose, side)] += 1
        result = by_role.get(role)
        if result is None:
            incomplete += 1
            taxonomy["row_missing"] = taxonomy.get("row_missing", 0) + 1
            table.append({"role_index": role, "side": side, "pose_id": pose,
                          "family_id": family, "status": "missing",
                          "clean": False, "defects": ["row_missing"]})
            continue
        status = result.get("status")
        if status == INFRASTRUCTURE_STATUS:
            infrastructure += 1
            incomplete += 1
            taxonomy[INFRASTRUCTURE_STATUS] = taxonomy.get(
                INFRASTRUCTURE_STATUS, 0) + 1
            table.append({"role_index": role, "side": side, "pose_id": pose,
                          "family_id": family, "status": status, "clean": False,
                          "defects": result.get("v107_defects") or [],
                          "infrastructure_error": result.get("infrastructure_error")})
            continue
        telemetry = result.get("pact_v106_frame_telemetry") or {}
        contact_frames = int(
            telemetry.get("pendant_robot_or_target_contact_frames") or 0)
        if contact_frames:
            pendant_contact_rows += 1
        value = telemetry.get("min_clearance_m")
        if value is not None:
            minimum = value if minimum is None else min(minimum, float(value))
        # Live-risk confirmation: a completed row that actually came close.
        risk = telemetry.get("min_lobe_stem_clearance_m")
        if risk is not None:
            witness = {"role_index": role, "side": side, "pose_id": pose,
                       "min_lobe_stem_clearance_m": float(risk),
                       "status": status}
            if float(risk) < float(
                risk_side.get(side, {}).get("min_lobe_stem_clearance_m", 1e9)
            ):
                risk_side[side] = witness
            if float(risk) < float(
                risk_pose.get(pose, {}).get("min_lobe_stem_clearance_m", 1e9)
            ):
                risk_pose[pose] = witness
        defects = result.get("v107_defects") or []
        row_clean = bool(result.get("v107_clean_success"))
        if row_clean:
            clean += 1
            by_side[side] += 1
            by_pose[pose] += 1
            by_cell[group_key(pose, side)] += 1
            by_family[family] += 1
        else:
            for defect in defects:
                head = defect.split("=")[0]
                taxonomy[head] = taxonomy.get(head, 0) + 1
        table.append({
            "role_index": role, "side": side, "pose_id": pose,
            "family_id": family, "status": status, "clean": row_clean,
            "defects": defects, "episode_steps": result.get("episode_steps"),
            "min_clearance_m": value,
            "min_lobe_stem_clearance_m": risk,
            "pendant_contact_frames": contact_frames,
            "result_sha256": result.get("result_sha256"),
        })

    side_ok = all(by_side[s] >= PHASE0_MIN_CLEAN_PER_SIDE for s in INTRUSION_SIDES)
    pose_ok = all(by_pose[p] >= PHASE0_MIN_CLEAN_PER_POSE for p in POSE_IDS)
    cell_ok = all(v >= PHASE0_MIN_CLEAN_PER_SIDE_POSE for v in by_cell.values())
    risk_side_ok = {
        s: bool(s in risk_side
                and risk_side[s]["min_lobe_stem_clearance_m"]
                <= PHASE0_RISK_CONFIRM_MAX_M)
        for s in INTRUSION_SIDES
    }
    risk_pose_ok = {
        p: bool(p in risk_pose
                and risk_pose[p]["min_lobe_stem_clearance_m"]
                <= PHASE0_RISK_CONFIRM_MAX_M)
        for p in POSE_IDS
    }
    limiting: list[str] = []
    if clean < PHASE0_MIN_CLEAN:
        limiting.append(f"clean {clean} < {PHASE0_MIN_CLEAN}")
    if not side_ok:
        limiting.append(f"per-side {by_side} < {PHASE0_MIN_CLEAN_PER_SIDE}")
    if not pose_ok:
        limiting.append(f"per-pose {by_pose} < {PHASE0_MIN_CLEAN_PER_POSE}")
    if not cell_ok:
        limiting.append(f"side x pose {by_cell} < {PHASE0_MIN_CLEAN_PER_SIDE_POSE}")
    if pendant_contact_rows:
        limiting.append(f"{pendant_contact_rows} rows with pendant contact")
    missing_side_risk = [s for s, ok in risk_side_ok.items() if not ok]
    if missing_side_risk:
        limiting.append(
            f"no completed row within {PHASE0_RISK_CONFIRM_MAX_M} m lobe/stem "
            f"clearance on side(s) {missing_side_risk}")
    missing_pose_risk = [p for p, ok in risk_pose_ok.items() if not ok]
    if missing_pose_risk:
        limiting.append(
            f"no completed row within {PHASE0_RISK_CONFIRM_MAX_M} m lobe/stem "
            f"clearance for pose(s) {missing_pose_risk}")
    if incomplete:
        limiting.append(f"{incomplete} incomplete rows")
    low, high = wilson_interval(clean, len(rows) or 1)
    return {
        **empty_authorization(),
        "n_rows": len(rows), "n_results": len(results),
        "clean_successes": clean,
        "clean_by_side": by_side, "clean_by_pose": by_pose,
        "clean_by_side_pose": by_cell, "clean_by_family": by_family,
        "n_by_side": n_side, "n_by_pose": n_pose, "n_by_side_pose": n_cell,
        "min_clean_required": PHASE0_MIN_CLEAN,
        "min_clean_per_side": PHASE0_MIN_CLEAN_PER_SIDE,
        "min_clean_per_pose": PHASE0_MIN_CLEAN_PER_POSE,
        "min_clean_per_side_pose": PHASE0_MIN_CLEAN_PER_SIDE_POSE,
        "pendant_contact_rows": pendant_contact_rows,
        "min_pendant_clearance_m": minimum,
        "risk_confirmation_max_m": PHASE0_RISK_CONFIRM_MAX_M,
        "risk_witness_by_side": risk_side,
        "risk_witness_by_pose": risk_pose,
        "risk_confirmed_by_side": risk_side_ok,
        "risk_confirmed_by_pose": risk_pose_ok,
        "failure_taxonomy": dict(sorted(taxonomy.items())),
        "row_table": table,
        "wilson_95_interval": [low, high],
        "incomplete_rows": incomplete,
        "infrastructure_failures": infrastructure,
        "limiting_predicates": limiting,
        "phase0_passed": not limiting,
    }


def run_gate_row(payload: dict[str, Any]) -> dict[str, Any]:
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("MLSPACES_ASSETS_DIR", str(ROOT / "assets"))
    os.environ.pop("DISPLAY", None)
    from run_pact_place_expert_screen import run_row

    return run_row(payload["row"], config_sha256=payload["config_sha256"],
                   output_root=payload["output_root"],
                   scene_xml=payload["scene_xml"])


def infrastructure_result(row: dict[str, Any], error: BaseException) -> dict[str, Any]:
    """An official row whose worker raised. Never replaced, never retried."""
    return {
        "role_index": int(row["role_index"]),
        "episode_id": row["episode_id"],
        "intrusion_side": row["intrusion_side"],
        "status": INFRASTRUCTURE_STATUS,
        "infrastructure_error": f"{type(error).__name__}: {error}",
        "infrastructure_traceback": "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        )[-2000:],
        "v107_defects": [f"status={INFRASTRUCTURE_STATUS}",
                         "incomplete_infrastructure_failure"],
        "v107_clean_success": False,
        "counted_as_incomplete": True,
        "replaced": False,
        "retried": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=ROOT / PHASE0_ROOT)
    parser.add_argument("--review-root", type=Path,
                        default=ROOT / OWNER_REVIEW_ROOT)
    parser.add_argument("--approval", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--print-approval-template", action="store_true",
                        help="print the exact record the owner must write")
    parser.add_argument("--integrity-only", action="store_true",
                        help="run the pre-run integrity check and exit")
    args = parser.parse_args()
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("MLSPACES_ASSETS_DIR", str(ROOT / "assets"))
    os.environ.pop("DISPLAY", None)
    started = time.time()

    review_root = args.review_root.resolve()
    approval_path = (args.approval.resolve() if args.approval
                     else review_root / "human_approval.json")
    expected = expected_bindings(review_root)
    video_names = sorted(p.name for p in (review_root / "videos").glob("*.mp4"))

    if args.print_approval_template:
        print(json.dumps({
            "decision": "approve_phase0",
            "created_by": "owner",
            "created_by_agent": False,
            "reviewed_videos": video_names,
            **expected,
        }, indent=2, sort_keys=True))
        return 0

    if args.integrity_only:
        report = integrity_report(review_root, approval_path=approval_path,
                                  phase="pre_run")
        print(json.dumps(report, indent=2, default=str))
        return 0 if report["passed"] else 1

    # Nothing below creates a directory or a row without a valid owner record.
    if not approval_path.is_file():
        raise ApprovalError(
            f"missing owner approval: {approval_path}. The agent must not "
            "create this file. Run with --print-approval-template to get the "
            "exact record to write.")
    approval = json.loads(approval_path.read_text())
    assert_approval(approval, expected, video_names)

    certification = json.loads(
        (ROOT / CERT_ROOT / "certification.json").read_text())
    scene_by_pose = {
        p: {"relative": certification["published_scenes"][p]["relative"],
            "sha256": certification["published_scenes"][p]["sha256"]}
        for p in POSE_IDS
    }
    assembly_by_pose = {
        p: sha256_payload(next(c for c in certification["compiled_checks"]
                               if c["pose_id"] == p))
        for p in POSE_IDS
    }
    rows = phase0_rows(selected=certification["selected"],
                       scene_by_pose=scene_by_pose,
                       assembly_by_pose=assembly_by_pose)

    pre_run = integrity_report(review_root, rows=rows,
                               approval_path=approval_path, phase="pre_run")
    if not pre_run["passed"]:
        raise IntegrityError(f"pre-run integrity failed: {pre_run['problems']}")

    output_root = args.output_root.resolve()
    if output_root.exists():
        raise RuntimeError(
            f"{output_root} already exists; Phase 0 runs exactly once")
    output_root.mkdir(parents=True)

    config = {
        "schema_version": "pact_place_v107_phase0_config_v1",
        "contract_version": CONTRACT_VERSION_V107,
        "environment_version": ENVIRONMENT_VERSION,
        "selected": certification["selected"],
        "scene_by_pose": scene_by_pose,
        "approval_path": str(approval_path),
        "approval_raw_file_sha256": sha256_file(approval_path),
        "gate_implementation_digest": gate_implementation_digest(),
        "frozen_before_row_0": True,
        "no_cherry_picking": True,
        "no_replacement_rows": True,
        "no_threshold_relaxation": True,
        "no_scientific_retries": True,
        "expert_screen_rows": rows,
        **empty_authorization(),
    }
    config["config_sha256"] = sha256_payload(config)
    write_immutable_create_only(output_root / "gate_manifest.json", config)

    results: list[dict[str, Any]] = []
    worker_exceptions: list[dict[str, Any]] = []
    try:
        print(f"executing frozen Phase-0 gate: {len(rows)} rows", flush=True)
        context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max(1, min(args.workers, len(rows))),
            mp_context=context, max_tasks_per_child=1,
        ) as executor:
            futures = {
                executor.submit(run_gate_row, {
                    "row": row, "config_sha256": config["config_sha256"],
                    "output_root": str(output_root),
                    "scene_xml": str(
                        ROOT / scene_by_pose[row["pose_id"]]["relative"]),
                }): row
                for row in rows
            }
            for done, future in enumerate(
                concurrent.futures.as_completed(futures), start=1
            ):
                row = futures[future]
                try:
                    result = future.result()
                    result["v107_defects"] = row_defects(result)
                    result["v107_clean_success"] = not result["v107_defects"]
                except BaseException as error:  # noqa: BLE001
                    # An official row whose worker died is an incomplete
                    # infrastructure failure. It is recorded, never replaced.
                    result = infrastructure_result(row, error)
                    worker_exceptions.append({
                        "role_index": int(row["role_index"]),
                        "error": result["infrastructure_error"],
                    })
                results.append(result)
                print(json.dumps({
                    "done": done, "of": len(rows),
                    "role": int(result["role_index"]),
                    "clean": result["v107_clean_success"],
                    "status": result.get("status"),
                    "defects": (result.get("v107_defects") or [])[:2],
                }), flush=True)
    except BaseException as error:  # noqa: BLE001
        # Even a catastrophic failure closes out: rows that never reported are
        # recorded as incomplete so gate.json is always written.
        reported = {int(r["role_index"]) for r in results}
        for row in rows:
            if int(row["role_index"]) not in reported:
                results.append(infrastructure_result(row, error))
        worker_exceptions.append({"role_index": None,
                                  "error": f"{type(error).__name__}: {error}"})
    finally:
        results.sort(key=lambda item: int(item["role_index"]))
        seen: dict[int, int] = {}
        for item in results:
            role = int(item["role_index"])
            seen[role] = seen.get(role, 0) + 1
        duplicates = sorted(r for r, n in seen.items() if n > 1)
        eligibility = gate_eligibility(rows, results)
        if duplicates:
            eligibility["limiting_predicates"].append(
                f"duplicate results for roles {duplicates}")
            eligibility["phase0_passed"] = False
        post_run = integrity_report(review_root, rows=rows,
                                    approval_path=approval_path,
                                    phase="post_run")
        document = {
            "schema_version": "pact_place_v107_phase0_gate_v1",
            "contract_version": CONTRACT_VERSION_V107,
            "environment_version": ENVIRONMENT_VERSION,
            "config_sha256": config["config_sha256"],
            "approval_raw_file_sha256": config["approval_raw_file_sha256"],
            "approval_bindings": expected,
            "gate_implementation_digest": config["gate_implementation_digest"],
            "n_rows": N_PHASE0_ROWS,
            "pre_run_integrity": pre_run,
            "post_run_integrity": post_run,
            "post_run_integrity_passed": bool(post_run["passed"]),
            "eligibility": eligibility,
            "distributions": distributions(rows, results, output_root),
            "duplicate_result_roles": duplicates,
            "worker_exceptions": worker_exceptions,
            "n_worker_exceptions": len(worker_exceptions),
            "executed_once": True,
            "no_row_replaced_or_reseeded": True,
            "infrastructure_retries": [],
            **empty_authorization(),
            # empty_authorization() is spread FIRST on purpose: spreading it
            # after would silently reset a passing gate back to false.
            "phase0_passed": bool(
                eligibility["phase0_passed"] and post_run["passed"]),
            "authorizes_collection": False,
            "authorizes_conversion": False,
            "authorizes_training": False,
            "authorizes_evaluation": False,
            "permanent_stop": not bool(eligibility["phase0_passed"]),
            "elapsed_s": time.time() - started,
        }
        hashes = write_immutable_create_only(output_root / "gate.json", document)
        print(json.dumps({
            "phase0_passed": document["phase0_passed"],
            "clean_successes": eligibility["clean_successes"],
            "clean_by_side": eligibility["clean_by_side"],
            "clean_by_pose": eligibility["clean_by_pose"],
            "clean_by_side_pose": eligibility["clean_by_side_pose"],
            "pendant_contact_rows": eligibility["pendant_contact_rows"],
            "min_pendant_clearance_m": eligibility["min_pendant_clearance_m"],
            "risk_confirmed_by_side": eligibility["risk_confirmed_by_side"],
            "risk_confirmed_by_pose": eligibility["risk_confirmed_by_pose"],
            "infrastructure_failures": eligibility["infrastructure_failures"],
            "limiting_predicates": eligibility["limiting_predicates"],
            "post_run_integrity_passed": document["post_run_integrity_passed"],
            "authorizes_collection": False, "authorizes_training": False,
            "authorizes_evaluation": False,
            **hashes,
        }, indent=2))
    return 0 if document["phase0_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
