#!/usr/bin/env python3
"""V10.7 qualification repair: contract, risk-aligned ranking, registered gates.

V10.6 geometry results are historical inputs here. Nothing in V10.4, V10.5 or
V10.6 is modified; V10.7 re-derives selection from the sealed V10.6 scores,
re-certifies, and runs the production pool.

Two deliberate departures from V10.6, both registered before execution:

* **Ranking is risk-aligned.** Universal >=15 mm clearance is the first key;
  additional clearance is demoted below risk relevance, because a pendant that
  is merely far away is not the environment this programme is trying to build.
* **The cardinal-TCP contact perturbation is retired as a gate.** It measured
  the reach of a straight-line TCP displacement, not physical reachability. The
  registered relevance test is natural exact clearance in the 15-35 mm band for
  all six pose x side groups, plus six-group causal sensing.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

for _p in (ROOT / "scripts", ROOT / "submodules" / "molmospaces"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from pact_place_v105_contract import (  # noqa: E402
    ImmutableArtifactError,
    canonical_payload_sha256,
    empty_authorization,
    recompute_payload_sha256,
    sha256_file,
    sha256_payload,
    v95_row_payload,
    wilson_interval,
    write_immutable_create_only,
    write_immutable_text_create_only,
)
from pact_place_v106_contract import (  # noqa: E402
    INTRUSION_SIDES,
    V95_LAYOUT_FAMILY_IDS,
)
from pact_place_v106_geometry import (  # noqa: E402
    ALL_GEOMS_V106,
    CLEARANCE_FLOOR_M,
    CONTRACT_VERSION_V106,
    ENVIRONMENT_VERSION_V106,
    PENDANT_BODY_V106,
    POSE_IDS,
    POSE_OFFSETS_M,
    RISK_BAND_M,
    SAMPLER_CLASS_V106,
    build_assembly,
    scene_xml_text,
)

CONTRACT_VERSION_V107 = "pact_place_v107_qualification_repair_v1"
# The environment family is unchanged; V10.7 re-selects within it, so the
# runtime marker, sampler and pendant body stay V10.6's on purpose.
ENVIRONMENT_VERSION = ENVIRONMENT_VERSION_V106
SAMPLER_CLASS = SAMPLER_CLASS_V106
PENDANT_BODY = PENDANT_BODY_V106
PLAN_RELATIVE = "docs/PACT_PLACE_V107_QUALIFICATION_REPAIR_PLAN.md"

SPEC_ROOT = "diagnostics_output/pact_place_v107_specification"
SELECTION_ROOT = "diagnostics_output/pact_place_v107_selection"
CERT_ROOT = "diagnostics_output/pact_place_v107_certification"
CAUSAL_ROOT = "diagnostics_output/pact_place_v107_causal"
DIAGNOSTIC_ROOT = "diagnostics_output/pact_place_v107_contact_diagnostic"
POOL_ROOT = "diagnostics_output/pact_place_v107_pool"
REVIEW_ROOT = "diagnostics_output/pact_place_v107_review"
PHASE0_ROOT = "diagnostics_output/pact_place_v107_phase0"

SCENES_DIR_RELATIVE = (
    "submodules/molmospaces/molmo_spaces/data_generation/custom_scenes"
)

# ---------------------------------------------------------------------------
# Sealed historical inputs
# ---------------------------------------------------------------------------
V106_SITING_JSON = "diagnostics_output/pact_place_v106_siting/siting.json"
V106_SITING_NPZ = "diagnostics_output/pact_place_v106_siting/per_row_scores.npz"
V106_CERT_JSON = (
    "diagnostics_output/pact_place_v106_certification/certification.json"
)
V106_CAUSAL_JSON = "diagnostics_output/pact_place_v106_causal/risk_causal.json"
V105_AUDIT_JSON = "diagnostics_output/pact_place_v105_audit/audit.json"
V105_RECON_JSON = (
    "diagnostics_output/pact_place_v105_reconstruction/reconstruction.json"
)
V105_CORPUS_NPZ = (
    "diagnostics_output/pact_place_v105_reconstruction/corpus_index.npz"
)
BASE_SCENE_V5 = f"{SCENES_DIR_RELATIVE}/pact_place_corridor_v5.xml"
BASE_SCENE_V3 = f"{SCENES_DIR_RELATIVE}/pact_place_corridor_v3.xml"

SEALED_INPUTS: tuple[str, ...] = (
    V106_SITING_JSON, V106_SITING_NPZ, V106_CERT_JSON, V106_CAUSAL_JSON,
    V105_AUDIT_JSON, V105_RECON_JSON, V105_CORPUS_NPZ,
    BASE_SCENE_V5, BASE_SCENE_V3,
)

IMPLEMENTATION_PATHS: tuple[str, ...] = (
    PLAN_RELATIVE,
    "scripts/pact_place_v107_contract.py",
    "scripts/run_pact_place_v107_specify.py",
    "scripts/run_pact_place_v107_select.py",
    "scripts/run_pact_place_v107_certify.py",
    "scripts/run_pact_place_v107_causal.py",
    "scripts/run_pact_place_v107_contact_diagnostic.py",
    "scripts/run_pact_place_v107_pool.py",
    "scripts/pact_place_v106_geometry.py",
    "scripts/pact_place_v106_contract.py",
    "scripts/pact_place_v105_clearance.py",
    "scripts/pact_place_v105_siting_core.py",
    "scripts/pact_place_v105_contract.py",
    "scripts/pact_place_v95_contract.py",
    "scripts/pact_geom_distance.py",
    "scripts/audit_pact_place_v105.py",
    "scripts/run_pact_place_expert_screen.py",
    "submodules/molmospaces/molmo_spaces/tasks/enclosure_reach.py",
    "submodules/molmospaces/molmo_spaces/tasks/pact_place_contact_audit.py",
    "tests/test_pact_place_v107.py",
)

# ---------------------------------------------------------------------------
# Streams. Distinct from every V10.5 and V10.6 stream.
# ---------------------------------------------------------------------------
POOL_STREAM = "pact-place-v10.7-qualification-pool"
POOL_MASTER_SEED = 2026107002
PHASE0_STREAM = "pact-place-v10.7-qualification-phase0"
PHASE0_MASTER_SEED = 2026107001

N_POOL_ROWS = 48
N_POOL_REPLICATES = 2
N_PHASE0_ROWS = 24
N_REVIEW_VIDEOS = 6

POOL_MIN_CLEAN = 32
POOL_MIN_CLEAN_PER_SIDE = 14
POOL_MIN_CLEAN_PER_POSE = 8
POOL_MIN_CLEAN_PER_SIDE_POSE = 4

PHASE0_MIN_CLEAN = 16
PHASE0_MIN_CLEAN_PER_SIDE = 7
PHASE0_MIN_CLEAN_PER_POSE = 4
PHASE0_MIN_CLEAN_PER_SIDE_POSE = 2

TASK_HORIZON = 1050
POLICY_TIMESTEP_MS = 66.0
REVIEW_FPS = 1000.0 / POLICY_TIMESTEP_MS
REVIEW_FRAME_STRIDE = 1

# Causal thresholds, unchanged from V10.6 but now required for all six groups.
CAUSAL_MIN_CHANGED_VALUES = 448
CAUSAL_MIN_CHANGED_SENSORS = 3
CAUSAL_LINK_TOKENS = ("link5", "link6")
CAUSAL_MIN_ONSET_FRAMES = 5
CAUSAL_MIN_ONSET_SECONDS = 0.10
CAUSAL_MAX_SIDE_RATIO = 4.0
CAUSAL_WINDOW_FRAMES = 60
PROXIMITY_TENSOR_SHAPE = (40, 4, 8, 8)

THRESHOLD_NEAR_M = 0.020
N_GROUPS = len(POSE_IDS) * len(INTRUSION_SIDES)


class HashDriftError(RuntimeError):
    """A bound input or implementation file no longer matches the contract."""


def file_hashes(paths: Sequence[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for relative in paths:
        target = ROOT / relative
        out[relative] = (
            {"raw_file_sha256": sha256_file(target), "present": True,
             "size_bytes": int(target.stat().st_size)}
            if target.is_file()
            else {"raw_file_sha256": "absent", "present": False, "size_bytes": 0}
        )
    return out


def implementation_digest(paths: Sequence[str]) -> str:
    return sha256_payload(
        [[p, file_hashes([p])[p]["raw_file_sha256"]] for p in paths]
    )


def verify_against_specification(spec: dict[str, Any]) -> dict[str, Any]:
    """Fail closed on any drift in a bound input or implementation file."""
    drift: list[dict[str, Any]] = []
    for section in ("sealed_inputs", "implementation_files"):
        for relative, bound in spec[section].items():
            target = ROOT / relative
            observed = sha256_file(target) if target.is_file() else "absent"
            if observed != bound["raw_file_sha256"]:
                drift.append({
                    "section": section, "path": relative,
                    "bound": bound["raw_file_sha256"], "observed": observed,
                })
    return {
        "n_checked": sum(len(spec[s]) for s in
                         ("sealed_inputs", "implementation_files")),
        "drift": drift,
        "n_drift": len(drift),
        "passed": not drift,
    }


def assert_no_drift(spec: dict[str, Any]) -> dict[str, Any]:
    report = verify_against_specification(spec)
    if not report["passed"]:
        raise HashDriftError(
            f"{report['n_drift']} bound file(s) drifted: "
            f"{[d['path'] for d in report['drift'][:5]]}"
        )
    return report


# ---------------------------------------------------------------------------
# Risk-aligned ranking
# ---------------------------------------------------------------------------
def group_key(pose_id: str, side: str) -> str:
    return f"{pose_id}|{side}"


def candidate_statistics(
    per_row: list[dict[str, Any]], bundle_key: str
) -> dict[str, Any]:
    """Everything the ranking needs, recomputed from the sealed row scores."""
    import numpy as np

    group_min: dict[str, float] = {}
    group_band: dict[str, int] = {}
    group_total: dict[str, int] = {}
    group_ge_floor: dict[str, int] = {}
    clearances: list[float] = []
    contacts = 0
    window_below = 0
    initial_below = 0
    env_intersections = 0
    directions: dict[str, set] = {side: set() for side in INTRUSION_SIDES}
    witnesses: dict[str, dict[str, Any]] = {}
    for row in per_row:
        if not row.get("ok"):
            continue
        side = str(row["intrusion_side"])
        for pose in POSE_IDS:
            score = row["scores"].get(f"{bundle_key}|{pose}")
            if score is None:
                continue
            key = group_key(pose, side)
            group_total[key] = group_total.get(key, 0) + 1
            value = score.get("min_clearance_m")
            if value is not None:
                value = float(value)
                clearances.append(value)
                if value >= CLEARANCE_FLOOR_M:
                    group_ge_floor[key] = group_ge_floor.get(key, 0) + 1
                if value < group_min.get(key, float("inf")):
                    group_min[key] = value
                    limiting = score.get("min_witness") or {}
                    witnesses[key] = {
                        "pose_id": pose, "intrusion_side": side,
                        "row_dir": row["row_dir"], "family_id": row["family_id"],
                        "seed_u32": int(row["seed_u32"]),
                        "min_clearance_m": value,
                        "min_lobe_stem_m": score.get("min_lobe_stem_m"),
                        "frame": int(limiting.get("frame") or 0),
                        "component": limiting.get("box"),
                        "probe_body": limiting.get("probe_body"),
                        "phase": limiting.get("phase"),
                    }
            risk = score.get("min_lobe_stem_m")
            if risk is not None and RISK_BAND_M[0] <= float(risk) <= RISK_BAND_M[1]:
                group_band[key] = group_band.get(key, 0) + 1
            if score.get("robot_or_target_contact"):
                contacts += 1
            if score.get("env_intersects"):
                env_intersections += 1
            for name, v in (score.get("window_min_m") or {}).items():
                if v is not None and float(v) < CLEARANCE_FLOOR_M:
                    window_below += 1
            for name, v in (score.get("initial_min_m") or {}).items():
                if v is not None and float(v) < CLEARANCE_FLOOR_M:
                    initial_below += 1
            for direction, v in (score.get("risk_by_direction_m") or {}).items():
                if v is not None and RISK_BAND_M[0] <= float(v) <= RISK_BAND_M[1]:
                    directions[side].add(direction)
    n_eval = len(clearances)
    below = sum(1 for v in clearances if v < CLEARANCE_FLOOR_M)
    minima = [group_min[k] for k in sorted(group_min)]
    return {
        "bundle_key": bundle_key,
        "n_evaluations": n_eval,
        "absolute_min_clearance_m": min(clearances) if clearances else None,
        "n_below_floor": below,
        "n_contacts": contacts,
        "n_env_intersections": env_intersections,
        "n_window_below_floor": window_below,
        "n_initial_below_floor": initial_below,
        "fraction_ge_floor": (n_eval - below) / n_eval if n_eval else None,
        "group_minimum_m": dict(sorted(group_min.items())),
        "group_band_evaluations": {
            k: group_band.get(k, 0) for k in sorted(group_total)
        },
        "group_total_evaluations": dict(sorted(group_total.items())),
        "group_evaluations_ge_floor": {
            k: group_ge_floor.get(k, 0) for k in sorted(group_total)
        },
        "band_evaluations_total": sum(group_band.values()),
        "n_groups": len(group_total),
        "mean_group_minimum_m": (float(np.mean(minima)) if minima else None),
        "max_group_minimum_m": (max(minima) if minima else None),
        "all_group_minima_in_band": bool(
            minima
            and len(minima) == N_GROUPS
            and all(RISK_BAND_M[0] <= v <= RISK_BAND_M[1] for v in minima)
        ),
        "universal_clearance": bool(
            n_eval > 0 and below == 0 and contacts == 0
            and window_below == 0 and initial_below == 0
            and env_intersections == 0
        ),
        "direction_band_witnesses": {
            k: sorted(v) for k, v in directions.items()
        },
        "loaded_outbound_both_sides": all(
            "loaded_outbound" in directions[side] for side in INTRUSION_SIDES
        ),
        "witnesses": witnesses,
    }


def is_qualified(stats: dict[str, Any]) -> dict[str, Any]:
    """The registered relevance test. No contact perturbation appears here."""
    checks = {
        "universal_clearance_15mm": bool(stats["universal_clearance"]),
        "all_six_group_minima_in_15_35mm": bool(stats["all_group_minima_in_band"]),
        "six_groups_present": stats["n_groups"] == N_GROUPS,
        "loaded_outbound_risk_both_sides": bool(
            stats["loaded_outbound_both_sides"]
        ),
    }
    return {"checks": checks, "qualified": all(checks.values())}


def risk_aligned_rank_key(stats: dict[str, Any]) -> tuple:
    """Universal clearance first; extra clearance demoted below risk.

    Key order, most significant first:

    1. universal >=15 mm clearance (hard preference, not a tiebreak);
    2. every group minimum inside the 15-35 mm relevance band;
    3. MORE risk-band evaluations -- a pendant that is merely far away is not
       the environment this programme is trying to build;
    4. a smaller mean group minimum, for the same reason;
    5. only then, more absolute clearance;
    6. deterministic radii tie-break.

    Keys 3 and 4 are what make this risk-aligned rather than
    clearance-maximising: V10.6 ranked additional clearance above relevance and
    selected the farthest admissible pendant.
    """
    x_m, r_neg, r_pos = (float(v) for v in stats["bundle_key"].split("|"))
    return (
        0 if stats["universal_clearance"] else 1,
        0 if stats["all_group_minima_in_band"] else 1,
        -int(stats["band_evaluations_total"]),
        float(stats["mean_group_minimum_m"] or 1e9),
        -float(stats["absolute_min_clearance_m"] or 0.0),
        r_neg,
        r_pos,
        x_m,
    )


# ---------------------------------------------------------------------------
# Manifests
# ---------------------------------------------------------------------------
def _row_seed(stream: str, master_seed: int, index: int) -> tuple[int, int]:
    digest = hashlib.sha256(f"{stream}:{master_seed}:{index}".encode()).digest()
    seed_u64 = int.from_bytes(digest[:8], "big")
    return seed_u64 % (2**32), seed_u64


def cells() -> list[tuple[str, str, str]]:
    return [
        (family, side, pose)
        for family in V95_LAYOUT_FAMILY_IDS
        for side in INTRUSION_SIDES
        for pose in POSE_IDS
    ]


def build_rows(
    *, stream: str, master_seed: int, replicates: int,
    selected: dict[str, float], scene_by_pose: dict[str, dict[str, str]],
    assembly_by_pose: dict[str, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    index = 0
    cache: dict[tuple[str, str], dict[str, Any]] = {}
    for replicate in range(int(replicates)):
        for family, side, pose in cells():
            seed_u32, seed_u64 = _row_seed(stream, master_seed, index)
            key = (family, side)
            if key not in cache:
                cache[key] = v95_row_payload(family, side)
            row: dict[str, Any] = {
                "role_index": index, "replicate": replicate,
                "family_id": family, "intrusion_side": side, "pose_id": pose,
                "pose_offset_m": POSE_OFFSETS_M[pose],
                "seed_stream": stream,
                "task_seed_u32": int(seed_u32), "task_seed_u64": int(seed_u64),
                "environment_version": ENVIRONMENT_VERSION,
                "contract_version": CONTRACT_VERSION_V107,
                "sampler_class": SAMPLER_CLASS,
                "pact_v106_x_m": float(selected["x_m"]),
                "pact_v106_r_neg_m": float(selected["r_neg_m"]),
                "pact_v106_r_pos_m": float(selected["r_pos_m"]),
                "pact_v106_scene_sha256": scene_by_pose[pose]["sha256"],
                "pact_v106_assembly_sha256": assembly_by_pose[pose],
                "pact_v107_scene_relative": scene_by_pose[pose]["relative"],
                **{k: (dict(v) if isinstance(v, dict)
                       else list(v) if isinstance(v, list) else v)
                   for k, v in cache[key].items()},
            }
            row["episode_id"] = hashlib.sha256(
                f"{stream}:{master_seed}:{family}:{side}:{pose}:{replicate}".encode()
            ).hexdigest()
            row["row_sha256"] = sha256_payload(row)
            rows.append(row)
            index += 1
    return rows


def pool_rows(**kw):
    return build_rows(stream=POOL_STREAM, master_seed=POOL_MASTER_SEED,
                      replicates=N_POOL_REPLICATES, **kw)


def phase0_rows(**kw):
    return build_rows(stream=PHASE0_STREAM, master_seed=PHASE0_MASTER_SEED,
                      replicates=1, **kw)


def row_defects(result: dict[str, Any]) -> list[str]:
    """Strict-clean semantics, inherited from the V9.5 lineage."""
    defects: list[str] = []
    if result.get("status") != "complete":
        defects.append(f"status={result.get('status')}")
    if not result.get("task_success"):
        defects.append("task_not_successful")
    if not result.get("grasp_phase_success"):
        defects.append("grasp_phase_failed")
    if not result.get("place_phase_success"):
        defects.append("place_phase_failed")
    if not result.get("cup_lifted_one_cm"):
        defects.append("cup_not_lifted")
    audit = result.get("contact_audit") or {}
    for name, count in (audit.get("contact_class_totals") or {}).items():
        if name != "grasp_target" and int(count) > 0:
            defects.append(f"contact:{name}={count}")
    if result.get("clutter_stability_events"):
        defects.append(
            f"clutter_stability_events={len(result['clutter_stability_events'])}"
        )
    telemetry = result.get("pact_v106_frame_telemetry") or {}
    if not telemetry:
        defects.append("missing_frame_telemetry")
    else:
        if telemetry.get("pendant_robot_or_target_contact_frames"):
            defects.append("pendant_contact")
        minimum = telemetry.get("min_clearance_m")
        if minimum is None:
            defects.append("missing_clearance_telemetry")
    return defects


def is_clean(result: dict[str, Any]) -> bool:
    return not row_defects(result)


def pool_eligibility(rows, results) -> dict[str, Any]:
    by_role = {int(item["role_index"]): item for item in results}
    clean = 0
    by_side = dict.fromkeys(INTRUSION_SIDES, 0)
    by_pose = dict.fromkeys(POSE_IDS, 0)
    by_cell = {group_key(p, s): 0 for p in POSE_IDS for s in INTRUSION_SIDES}
    by_family = dict.fromkeys(V95_LAYOUT_FAMILY_IDS, 0)
    n_side = dict.fromkeys(INTRUSION_SIDES, 0)
    n_pose = dict.fromkeys(POSE_IDS, 0)
    n_cell = dict.fromkeys(by_cell, 0)
    incomplete = 0
    failures: list[dict[str, Any]] = []
    for row in rows:
        side, pose, family = row["intrusion_side"], row["pose_id"], row["family_id"]
        n_side[side] += 1
        n_pose[pose] += 1
        n_cell[group_key(pose, side)] += 1
        result = by_role.get(int(row["role_index"]))
        if result is None:
            incomplete += 1
            failures.append({"role_index": int(row["role_index"]),
                             "reason": "row missing"})
            continue
        if result.get("v107_clean_success"):
            clean += 1
            by_side[side] += 1
            by_pose[pose] += 1
            by_cell[group_key(pose, side)] += 1
            by_family[family] += 1
        else:
            failures.append({
                "role_index": int(row["role_index"]), "side": side,
                "pose_id": pose, "family_id": family,
                "defects": result.get("v107_defects") or [],
            })
    side_ok = all(by_side[s] >= POOL_MIN_CLEAN_PER_SIDE for s in INTRUSION_SIDES)
    pose_ok = all(by_pose[p] >= POOL_MIN_CLEAN_PER_POSE for p in POSE_IDS)
    cell_ok = all(v >= POOL_MIN_CLEAN_PER_SIDE_POSE for v in by_cell.values())
    limiting: list[str] = []
    if clean < POOL_MIN_CLEAN:
        limiting.append(f"pool clean {clean} < {POOL_MIN_CLEAN}")
    if not side_ok:
        limiting.append(f"per-side {by_side} < {POOL_MIN_CLEAN_PER_SIDE}")
    if not pose_ok:
        limiting.append(f"per-pose {by_pose} < {POOL_MIN_CLEAN_PER_POSE}")
    if not cell_ok:
        limiting.append(f"side x pose {by_cell} < {POOL_MIN_CLEAN_PER_SIDE_POSE}")
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
        "min_clean_required": POOL_MIN_CLEAN,
        "min_clean_per_side": POOL_MIN_CLEAN_PER_SIDE,
        "min_clean_per_pose": POOL_MIN_CLEAN_PER_POSE,
        "min_clean_per_side_pose": POOL_MIN_CLEAN_PER_SIDE_POSE,
        "wilson_95_interval": [low, high],
        "incomplete_rows": incomplete,
        "failures": failures,
        "limiting_predicates": limiting,
        "scaled_from_phase0_bar": f"{PHASE0_MIN_CLEAN}/{N_PHASE0_ROWS}",
        "pool_passed": not limiting,
    }


__all__ = [
    "ALL_GEOMS_V106", "CAUSAL_LINK_TOKENS", "CAUSAL_MAX_SIDE_RATIO",
    "CAUSAL_MIN_CHANGED_SENSORS", "CAUSAL_MIN_CHANGED_VALUES",
    "CAUSAL_MIN_ONSET_FRAMES", "CAUSAL_MIN_ONSET_SECONDS", "CAUSAL_ROOT",
    "CAUSAL_WINDOW_FRAMES", "CERT_ROOT", "CLEARANCE_FLOOR_M",
    "CONTRACT_VERSION_V107", "DIAGNOSTIC_ROOT", "ENVIRONMENT_VERSION",
    "HashDriftError", "IMPLEMENTATION_PATHS", "ImmutableArtifactError",
    "INTRUSION_SIDES", "N_GROUPS", "N_POOL_ROWS", "N_REVIEW_VIDEOS",
    "PENDANT_BODY", "PHASE0_ROOT", "PLAN_RELATIVE", "POOL_MASTER_SEED",
    "POOL_MIN_CLEAN", "POOL_MIN_CLEAN_PER_POSE", "POOL_MIN_CLEAN_PER_SIDE",
    "POOL_MIN_CLEAN_PER_SIDE_POSE", "POOL_ROOT", "POOL_STREAM", "POSE_IDS",
    "POSE_OFFSETS_M", "PROXIMITY_TENSOR_SHAPE", "REVIEW_FPS",
    "REVIEW_FRAME_STRIDE", "REVIEW_ROOT", "RISK_BAND_M", "ROOT",
    "SAMPLER_CLASS", "SCENES_DIR_RELATIVE", "SEALED_INPUTS", "SELECTION_ROOT",
    "SPEC_ROOT", "TASK_HORIZON", "THRESHOLD_NEAR_M", "V105_AUDIT_JSON",
    "V106_CAUSAL_JSON", "V106_CERT_JSON", "V106_SITING_JSON", "V106_SITING_NPZ",
    "V95_LAYOUT_FAMILY_IDS", "assert_no_drift", "build_assembly", "build_rows",
    "candidate_statistics", "canonical_payload_sha256", "cells",
    "empty_authorization", "file_hashes", "group_key", "implementation_digest",
    "is_clean", "is_qualified", "phase0_rows", "pool_eligibility", "pool_rows",
    "recompute_payload_sha256", "risk_aligned_rank_key", "row_defects",
    "scene_xml_text", "sha256_file", "sha256_payload",
    "verify_against_specification", "wilson_interval",
    "write_immutable_create_only", "write_immutable_text_create_only",
]
