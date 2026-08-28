#!/usr/bin/env python3
"""Deterministic V10.4 contract, provenance, and manifests.

V10.4 starts unauthorized and stays unauthorized. Only an owner-supplied
``human_approval.json`` bound to every listed byte hash may unlock Phase 0, and
even a passing Phase 0 authorizes nothing downstream of the plan.

All provenance is byte-level: hashes are recomputed from file bytes. An
embedded JSON ``artifact_sha256`` is never trusted as the verification result.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Sequence

from pact_place_v104_geometry import (
    ALL_GEOMS_V104,
    BASE_CONFIG_RELATIVE,
    BASE_SAMPLER_CLASS,
    BASE_SCENE_RELATIVE,
    CONTRACT_VERSION_V104,
    ENVIRONMENT_VERSION_V104,
    PENDANT_BODY_V104,
    SCENE_XML_RELATIVE_V104,
    assembly_expectations,
    production_assembly,
    scene_xml_sha256,
)
from pact_place_v104_runtime import (
    INITIAL_FREE_SPACE_SPEED_CAP_M_S,
    SAMPLER_CLASS_V104,
    TASK_HORIZON_V104,
)

ROOT = Path(__file__).resolve().parents[1]

CONTRACT_VERSION = CONTRACT_VERSION_V104
ENVIRONMENT_VERSION = ENVIRONMENT_VERSION_V104
SAMPLER_CLASS = SAMPLER_CLASS_V104

REVIEW_STREAM = "pact-place-v10.4-first-shot-review-production"
REVIEW_MASTER_SEED = 2026104002
N_REVIEW_ROWS = 6
GATE_STREAM = "pact-place-v10.4-first-shot-phase0"
GATE_MASTER_SEED = 2026104001
N_GATE_ROWS = 24

MIN_REVIEW_CLEAN = 5
MIN_REVIEW_CLEAN_PER_SIDE = 2
N_REVIEW_PER_SIDE = 3
MIN_GATE_CLEAN = 20
MIN_GATE_CLEAN_PER_SIDE = 9
N_GATE_PER_SIDE = 12
REVIEW_MIN_CLEARANCE_M = 0.020
GATE_MIN_CLEARANCE_M = 0.015

POLICY_TIMESTEP_MS = 66.0
REVIEW_FPS = 1000.0 / POLICY_TIMESTEP_MS
REVIEW_FRAME_STRIDE = 1
N_REVIEW_VIDEOS = 6

PREFLIGHT_ROOT = "diagnostics_output/pact_place_v104_preflight"
PRODUCTION_ROOT = "diagnostics_output/pact_place_v104_review_production"
CAUSAL_ROOT = "diagnostics_output/pact_place_v104_causal"
REVIEW_ROOT = "diagnostics_output/pact_place_v104_review"
PHASE0_ROOT = "diagnostics_output/pact_place_v104_phase0"

# Causal panel-preservation floors. Not lowered.
CAUSAL_MIN_CHANGED_SENSORS = 3
CAUSAL_MIN_CHANGED_VALUES = 448
CAUSAL_PANEL_PRESERVATION_FLOOR = 7209
CAUSAL_HISTORICAL_PANEL_MINIMUM = 28836
CAUSAL_MAX_SIDE_RATIO = 4.0

DISALLOWED_CONTACT_CLASSES = (
    "mounted_fixture",
    "clutter",
    "hazard_bar",
    "other_environment",
)


def sha256_bytes_of(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_payload(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


class ImmutableArtifactError(RuntimeError):
    """Refused to replace an artifact that already exists."""


def write_immutable_create_only(path: Path, document: dict[str, Any]) -> str:
    """Atomic create-if-absent. Refuses to replace an existing artifact."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(document)
    payload.pop("artifact_sha256", None)
    digest = sha256_payload(payload)
    payload["artifact_sha256"] = digest
    text = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError as error:
            raise ImmutableArtifactError(
                f"refusing to replace an existing artifact: {target}"
            ) from error
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return digest


def empty_authorization() -> dict[str, bool]:
    return {
        "eligible_for_human_review": False,
        "human_approval_present": False,
        "authorizes_phase0": False,
        "authorizes_gate": False,
        "authorizes_collection": False,
        "authorizes_training": False,
        "authorizes_evaluation": False,
        "phase0_passed": False,
    }


# ---------------------------------------------------------------------------
# Byte-level provenance
# ---------------------------------------------------------------------------
def _v6c_row_paths() -> list[str]:
    document = json.loads((ROOT / BASE_CONFIG_RELATIVE).read_text())
    out: list[str] = []
    for row in document["expert_screen_rows"]:
        directory = (
            f"diagnostics_output/pact_place_corridor_v6c/expert_screen_rows/"
            f"{int(row['role_index']):02d}_{row['episode_id'][:16]}"
        )
        out.append(f"{directory}/result.json")
        out.append(f"{directory}/trajectory.json")
    return out


def protected_artifact_paths() -> list[str]:
    paths = [
        BASE_CONFIG_RELATIVE,
        BASE_SCENE_RELATIVE,
        "diagnostics_output/pact_place_corridor_v6c/expert_screen.json",
        "submodules/molmospaces/molmo_spaces/data_generation/custom_scenes/"
        "pact_place_corridor_v5.xml",
        "diagnostics_output/pact_place_v99_baseline_reconstruction/reconstruction.json",
        "diagnostics_output/pact_place_v99_siting/snapshots/snapshots.json",
        "diagnostics_output/pact_place_v101_empirical_review/review_manifest.json",
        "diagnostics_output/pact_place_v101_empirical_review/summary.json",
        "diagnostics_output/pact_place_v101_empirical_causal/causal.json",
        "diagnostics_output/pact_place_v102_preflight/preflight.json",
        "diagnostics_output/pact_place_v102_preflight/contact_parity_root_cause.json",
        "diagnostics_output/pact_place_v103_ik_search/search.json",
        "diagnostics_output/pact_place_v103_ik_search/endpoint_certificate.json",
    ]
    paths.extend(_v6c_row_paths())
    reconstruction = ROOT / "diagnostics_output/pact_place_v99_baseline_reconstruction"
    if reconstruction.is_dir():
        paths.extend(
            sorted(
                str(item.relative_to(ROOT))
                for item in reconstruction.glob("*.npz")
            )
        )
    return [path for path in paths if (ROOT / path).is_file()]


def protected_artifact_hashes() -> dict[str, str]:
    return {path: sha256_bytes_of(ROOT / path) for path in protected_artifact_paths()}


IMPLEMENTATION_PATHS = (
    "scripts/pact_place_v104_contract.py",
    "scripts/pact_place_v104_geometry.py",
    "scripts/pact_place_v104_clearance.py",
    "scripts/pact_place_v104_runtime.py",
    "scripts/run_pact_place_v104_preflight.py",
    "scripts/run_pact_place_v104_review_production.py",
    "scripts/run_pact_place_v104_causal.py",
    "scripts/run_pact_place_v104_review_video.py",
    "scripts/run_pact_place_v104_phase0.py",
    "scripts/pact_geom_distance.py",
    "scripts/run_pact_place_expert_screen.py",
    "tests/test_pact_place_v104_first_shot.py",
    SCENE_XML_RELATIVE_V104,
    "submodules/molmospaces/molmo_spaces/tasks/enclosure_reach.py",
    "submodules/molmospaces/molmo_spaces/tasks/pact_place_contact_audit.py",
)


def implementation_hashes() -> dict[str, str]:
    out: dict[str, str] = {}
    for path in IMPLEMENTATION_PATHS:
        target = ROOT / path
        out[path] = sha256_bytes_of(target) if target.is_file() else "absent"
    return out


def implementation_sha256() -> str:
    return sha256_payload(implementation_hashes())


def verify_protected_artifacts(expected: dict[str, str] | None = None) -> dict[str, Any]:
    observed = protected_artifact_hashes()
    mismatches = []
    if expected:
        for path, digest in expected.items():
            if observed.get(path) != digest:
                mismatches.append(
                    {"path": path, "expected": digest, "observed": observed.get(path)}
                )
    return {
        "n_artifacts": len(observed),
        "hashes": observed,
        "mismatches": mismatches,
        "verified_from_file_bytes": True,
        "passed": not mismatches,
    }


# ---------------------------------------------------------------------------
# Manifests
# ---------------------------------------------------------------------------
def _row_seed(stream: str, master_seed: int, role_index: int) -> tuple[int, int]:
    digest = hashlib.sha256(
        f"{stream}:{int(master_seed)}:{int(role_index)}".encode()
    ).digest()
    return int.from_bytes(digest[:4], "big"), int.from_bytes(digest[:8], "big")


def _sides(n_rows: int) -> list[str]:
    """Frozen, balanced, alternating side order. Never chosen after outcomes."""
    return ["left" if index % 2 == 0 else "right" for index in range(n_rows)]


def build_rows(
    *, stream: str, master_seed: int, n_rows: int, scene_sha256: str,
    assembly_sha256: str, implementation_digest: str,
) -> list[dict[str, Any]]:
    sides = _sides(n_rows)
    rows: list[dict[str, Any]] = []
    for role_index in range(n_rows):
        rng = random.Random(
            int.from_bytes(
                hashlib.sha256(
                    f"{stream}:jitter:{master_seed}:{role_index}".encode()
                ).digest()[:8],
                "big",
            )
        )
        seed_u32, seed_u64 = _row_seed(stream, master_seed, role_index)
        row = {
            "role_index": role_index,
            "episode_id": hashlib.sha256(
                f"{stream}:{master_seed}:{role_index}".encode()
            ).hexdigest(),
            "intrusion_side": sides[role_index],
            "scene_template_house_index": 1,
            "task_seed_u32": seed_u32,
            "task_seed_u64": seed_u64,
            "max_sampling_retries": 4,
            "clutter_x_jitter_m": {
                slot: round(rng.uniform(-0.020, 0.020), 9)
                for slot in ("l0", "l1", "r0", "r1")
            },
            "clutter_y_jitter_m": {
                slot: round(rng.uniform(-0.020, 0.020), 9)
                for slot in ("l0", "l1", "r0", "r1")
            },
            "panel_x_jitter_m": round(rng.uniform(-0.015, 0.015), 9),
            "panel_face_jitter_m": round(rng.uniform(-0.005, 0.005), 9),
            "sampler_class": SAMPLER_CLASS,
            "seed_stream": stream,
            "pact_v104_contract_version": CONTRACT_VERSION,
            "pact_v104_scene_sha256": scene_sha256,
            "pact_v104_assembly_sha256": assembly_sha256,
            "implementation_sha256": implementation_digest,
        }
        row["row_sha256"] = sha256_payload(row)
        rows.append(row)
    return rows


def build_contract() -> dict[str, Any]:
    assembly = production_assembly()
    scene_path = ROOT / SCENE_XML_RELATIVE_V104
    scene_bytes_sha = sha256_bytes_of(scene_path) if scene_path.is_file() else None
    serialized_sha = scene_xml_sha256(assembly)
    if scene_bytes_sha is not None and scene_bytes_sha != serialized_sha:
        raise ValueError(
            "the on-disk V10.4 scene does not match the serialized production assembly"
        )
    assembly_sha = sha256_payload(assembly)
    digest = implementation_sha256()
    kwargs = dict(
        scene_sha256=serialized_sha,
        assembly_sha256=assembly_sha,
        implementation_digest=digest,
    )
    review_rows = build_rows(
        stream=REVIEW_STREAM, master_seed=REVIEW_MASTER_SEED, n_rows=N_REVIEW_ROWS, **kwargs
    )
    gate_rows = build_rows(
        stream=GATE_STREAM, master_seed=GATE_MASTER_SEED, n_rows=N_GATE_ROWS, **kwargs
    )
    review_ids = {row["episode_id"] for row in review_rows}
    gate_ids = {row["episode_id"] for row in gate_rows}
    review_seeds = {row["task_seed_u32"] for row in review_rows}
    gate_seeds = {row["task_seed_u32"] for row in gate_rows}
    if review_ids & gate_ids:
        raise RuntimeError("review and Phase-0 episode IDs intersect")
    if review_seeds & gate_seeds:
        raise RuntimeError("review and Phase-0 task seeds intersect")
    for rows, per_side in ((review_rows, N_REVIEW_PER_SIDE), (gate_rows, N_GATE_PER_SIDE)):
        for side in ("left", "right"):
            count = sum(1 for row in rows if row["intrusion_side"] == side)
            if count != per_side:
                raise RuntimeError(f"side balance is {count}, expected {per_side}")
    payload = {
        "schema_version": CONTRACT_VERSION,
        "contract_version": CONTRACT_VERSION,
        "environment_version": ENVIRONMENT_VERSION,
        "sampler_class": SAMPLER_CLASS,
        "base_sampler_class": BASE_SAMPLER_CLASS,
        "base_config": BASE_CONFIG_RELATIVE,
        "base_scene": BASE_SCENE_RELATIVE,
        "scene_xml": SCENE_XML_RELATIVE_V104,
        "scene_sha256": serialized_sha,
        "scene_bytes_sha256": scene_bytes_sha,
        "pendant_body": PENDANT_BODY_V104,
        "pendant_geoms": list(ALL_GEOMS_V104),
        "assembly": assembly,
        "assembly_sha256": assembly_sha,
        "assembly_expectations": assembly_expectations(assembly),
        "speed_amendment": {
            "initial_free_space_speed_cap_m_s": float(INITIAL_FREE_SPACE_SPEED_CAP_M_S),
            "all_later_segment_speeds": "inherited byte-for-byte from V6c",
            "task_horizon": int(TASK_HORIZON_V104),
        },
        "review_stream": REVIEW_STREAM,
        "review_master_seed": REVIEW_MASTER_SEED,
        "n_review_rows": N_REVIEW_ROWS,
        "gate_stream": GATE_STREAM,
        "gate_master_seed": GATE_MASTER_SEED,
        "n_gate_rows": N_GATE_ROWS,
        "review_min_clearance_m": REVIEW_MIN_CLEARANCE_M,
        "gate_min_clearance_m": GATE_MIN_CLEARANCE_M,
        "review_thresholds": {
            "min_clean": MIN_REVIEW_CLEAN,
            "min_clean_per_side": MIN_REVIEW_CLEAN_PER_SIDE,
            "rows_per_side": N_REVIEW_PER_SIDE,
        },
        "gate_thresholds": {
            "min_clean": MIN_GATE_CLEAN,
            "min_clean_per_side": MIN_GATE_CLEAN_PER_SIDE,
            "rows_per_side": N_GATE_PER_SIDE,
        },
        "causal_floors": {
            "min_changed_sensors": CAUSAL_MIN_CHANGED_SENSORS,
            "min_changed_values": CAUSAL_MIN_CHANGED_VALUES,
            "panel_preservation_floor": CAUSAL_PANEL_PRESERVATION_FLOOR,
            "historical_panel_minimum": CAUSAL_HISTORICAL_PANEL_MINIMUM,
            "max_side_ratio": CAUSAL_MAX_SIDE_RATIO,
        },
        "review_fps": REVIEW_FPS,
        "review_frame_stride": REVIEW_FRAME_STRIDE,
        "n_review_videos": N_REVIEW_VIDEOS,
        "implementation_sha256": digest,
        "implementation_files": implementation_hashes(),
        "protected_artifacts": protected_artifact_hashes(),
        "review_rows": review_rows,
        "gate_rows": gate_rows,
        "clean_rate_is_not_an_estimate": True,
        **empty_authorization(),
    }
    payload["contract_sha256"] = sha256_payload(payload)
    return payload


# ---------------------------------------------------------------------------
# Row admission
# ---------------------------------------------------------------------------
def row_defects(result: dict[str, Any], *, min_clearance_m: float) -> list[str]:
    status = result.get("status")
    if status == "sampling_failure":
        return ["sampling_failure"]
    if status == "infrastructure_failure":
        return ["infrastructure_failure"]
    if status != "complete":
        return ["nonterminal"]
    defects: list[str] = []
    audit = result.get("contact_audit") or {}
    totals = audit.get("contact_class_totals") or {}
    if not totals:
        defects.append("missing_contact_audit")
    for key in DISALLOWED_CONTACT_CLASSES:
        if int(totals.get(key, 0)) > 0:
            defects.append(f"{key}_contact")
    from run_pact_place_expert_screen import place_receptacle_outside_placement

    if place_receptacle_outside_placement(audit) > 0:
        defects.append("place_receptacle_outside_placement")
    if result.get("clutter_stability_events"):
        defects.append("clutter_stability_event")
    frames = result.get("pact_v104_frame_telemetry") or {}
    if not frames or int(frames.get("n_frames") or 0) <= 0:
        defects.append("missing_frame_telemetry")
    else:
        if int(frames.get("n_frames_measured") or 0) != int(frames.get("n_frames") or 0):
            defects.append("incomplete_frame_clearance")
        if int(frames.get("pendant_contact_frames") or 0) > 0:
            defects.append("pendant_contact")
        minimum = frames.get("min_clearance_m")
        if minimum is None:
            defects.append("missing_frame_clearance")
        elif float(minimum) < float(min_clearance_m) - 1e-12:
            defects.append("clearance_below_floor")
        for name, value in (frames.get("per_component_min_clearance_m") or {}).items():
            if value is None:
                defects.append(f"missing_component_clearance:{name}")
            elif float(value) < float(min_clearance_m) - 1e-12:
                defects.append(f"component_clearance_below_floor:{name}")
    amendment = result.get("pact_v104_speed_amendment") or {}
    if not amendment.get("applied"):
        defects.append("speed_amendment_not_applied")
    elif int(amendment.get("n_segments_changed") or 0) != 1:
        defects.append("speed_amendment_changed_wrong_segment_count")
    if result.get("bow_fallback_taken"):
        defects.append("route_fallback")
    tracking = result.get("terminal_tracking") or {}
    if int(tracking.get("sequential_ik_failures") or 0) > 0:
        defects.append("ik_cascade")
    scene = result.get("scene_params") or {}
    if str(scene.get("pact_place_environment_version") or "") != ENVIRONMENT_VERSION:
        defects.append("environment_marker_mismatch")
    if not result.get("task_success"):
        defects.append("task_failure")
    if not result.get("grasp_phase_success"):
        defects.append("grasp_failure")
    if not result.get("place_phase_success"):
        defects.append("place_failure")
    if not result.get("clean_success"):
        defects.append("not_strict_clean_success")
    return sorted(set(defects))


def is_clean_success(result: dict[str, Any], *, min_clearance_m: float) -> bool:
    return not row_defects(result, min_clearance_m=min_clearance_m)


def stage_eligibility(
    rows: Sequence[dict[str, Any]],
    results: Sequence[dict[str, Any]],
    *,
    expected_rows: int,
    min_clean: int,
    min_clean_per_side: int,
    min_clearance_m: float,
    passed_key: str,
) -> dict[str, Any]:
    by_episode = {str(item["episode_id"]): item for item in results}
    failures: list[dict[str, Any]] = []
    reconciled = len(rows) == expected_rows and len(results) == expected_rows
    if not reconciled:
        failures.append(
            {"code": "row_count", "rows": len(rows), "results": len(results)}
        )
    infrastructure = 0
    pendant_contact_rows = 0
    clean_by_side = {"left": 0, "right": 0}
    n_clean = 0
    minima: list[float] = []
    for row in rows:
        result = by_episode.get(str(row["episode_id"]))
        if result is None:
            reconciled = False
            failures.append({"code": "missing_result", "role_index": row["role_index"]})
            continue
        if result.get("row_sha256") != row.get("row_sha256"):
            reconciled = False
            failures.append({"code": "row_sha_mismatch", "role_index": row["role_index"]})
        if result.get("status") == "infrastructure_failure":
            infrastructure += 1
        frames = result.get("pact_v104_frame_telemetry") or {}
        totals = (result.get("contact_audit") or {}).get("contact_class_totals") or {}
        if int(frames.get("pendant_contact_frames") or 0) or int(
            totals.get("mounted_fixture", 0)
        ):
            pendant_contact_rows += 1
            failures.append({"code": "pendant_contact", "role_index": row["role_index"]})
        if frames.get("min_clearance_m") is not None:
            minima.append(float(frames["min_clearance_m"]))
        for defect in row_defects(result, min_clearance_m=min_clearance_m):
            failures.append(
                {
                    "code": defect,
                    "role_index": row["role_index"],
                    "intrusion_side": row.get("intrusion_side"),
                }
            )
        if is_clean_success(result, min_clearance_m=min_clearance_m):
            n_clean += 1
            clean_by_side[str(row["intrusion_side"])] += 1
    side_short = [
        {"code": "side_clean_shortfall", "side": side, "clean": count,
         "required": min_clean_per_side}
        for side, count in sorted(clean_by_side.items())
        if count < min_clean_per_side
    ]
    failures.extend(side_short)
    passed = bool(
        reconciled
        and infrastructure == 0
        and pendant_contact_rows == 0
        and n_clean >= min_clean
        and not side_short
        and (not minima or min(minima) >= min_clearance_m - 1e-12)
    )
    # empty_authorization() must be spread first: phase0_passed is both an
    # outcome and an authorization field, and spreading it last would silently
    # reset a passing gate to False.
    return {
        **empty_authorization(),
        passed_key: passed,
        "reconciled": reconciled,
        "n_rows": len(results),
        "infrastructure_failures": infrastructure,
        "pendant_contact_rows": pendant_contact_rows,
        "clean_successes": n_clean,
        "min_clean_required": min_clean,
        "clean_by_side": clean_by_side,
        "min_clean_per_side_required": min_clean_per_side,
        "min_observed_clearance_m": float(min(minima)) if minima else None,
        "clearance_floor_m": float(min_clearance_m),
        "failures": failures,
        "clean_rate_is_not_an_estimate": True,
    }


def review_eligibility(rows, results) -> dict[str, Any]:
    return stage_eligibility(
        rows, results, expected_rows=N_REVIEW_ROWS, min_clean=MIN_REVIEW_CLEAN,
        min_clean_per_side=MIN_REVIEW_CLEAN_PER_SIDE,
        min_clearance_m=REVIEW_MIN_CLEARANCE_M, passed_key="production_pack_passed",
    )


def gate_eligibility(rows, results) -> dict[str, Any]:
    return stage_eligibility(
        rows, results, expected_rows=N_GATE_ROWS, min_clean=MIN_GATE_CLEAN,
        min_clean_per_side=MIN_GATE_CLEAN_PER_SIDE,
        min_clearance_m=GATE_MIN_CLEARANCE_M, passed_key="phase0_passed",
    )


def assert_phase0_approval(approval: dict[str, Any] | None, expected: dict[str, str]) -> None:
    """An owner record must exist and bind every listed byte hash."""
    if not approval:
        raise PermissionError("Phase 0 requires an owner-supplied human_approval.json")
    if approval.get("decision") != "approve_phase0":
        raise PermissionError(f"Phase 0 refused: decision={approval.get('decision')!r}")
    if approval.get("created_by_agent"):
        raise PermissionError("Phase 0 refuses an agent-created approval record")
    missing = [key for key in expected if key not in approval]
    if missing:
        raise PermissionError(f"Phase 0 approval is missing bindings: {sorted(missing)}")
    for key, digest in expected.items():
        if approval.get(key) != digest:
            raise PermissionError(
                f"Phase 0 approval binding is stale for {key}: "
                f"{approval.get(key)!r} != {digest!r}"
            )
