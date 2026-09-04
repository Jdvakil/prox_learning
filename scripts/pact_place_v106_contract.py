#!/usr/bin/env python3
"""V10.6 contract: asymmetric global lattice, preregistered fallback, scaled floors.

Every threshold in this file is fixed before any V10.6 scoring runs and is
never edited after results are seen. The fallback admission rule in particular
is written down here in full, ahead of time, precisely because it is the rule
that could otherwise be tuned to whatever the data happened to show.
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
    FRAGILITY_ARTIFACT,
    FRAGILITY_ROWS_DIR,
    INTRUSION_SIDES,
    V95_LAYOUT_FAMILY_IDS,
    V95_VESSEL_JITTER,
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
from pact_place_v105_contract import ImmutableArtifactError  # noqa: E402
from pact_place_v106_geometry import (  # noqa: E402
    BASE_SCENE_RELATIVE_V106,
    CLEARANCE_FLOOR_M,
    CONTRACT_VERSION_V106,
    ENVIRONMENT_VERSION_V106,
    FALLBACK_ABSOLUTE_MIN_CLEARANCE_M,
    FALLBACK_MAX_CONTACTS,
    FALLBACK_MIN_FRACTION_GE_FLOOR,
    FALLBACK_MIN_GROUP_FRACTION_GE_FLOOR,
    LATTICE_R_NEG_M,
    LATTICE_R_POS_M,
    LATTICE_X_M,
    POSE_IDS,
    POSE_OFFSETS_M,
    RISK_BAND_M,
    SAMPLER_CLASS_V106,
    lattice_candidates,
)

PLAN_RELATIVE = "docs/PACT_PLACE_V106_ASYMMETRIC_PENDANT_PLAN.md"

SITING_ROOT = "diagnostics_output/pact_place_v106_siting"
CERT_ROOT = "diagnostics_output/pact_place_v106_certification"
CAUSAL_ROOT = "diagnostics_output/pact_place_v106_causal"
POOL_ROOT = "diagnostics_output/pact_place_v106_review_pool"
REVIEW_ROOT = "diagnostics_output/pact_place_v106_review"
PHASE0_ROOT = "diagnostics_output/pact_place_v106_phase0"

V105_AUDIT_ARTIFACT = "diagnostics_output/pact_place_v105_audit/audit.json"
V105_SITING_ARTIFACT = "diagnostics_output/pact_place_v105_siting/siting.json"
V105_SCORES_ARTIFACT = "diagnostics_output/pact_place_v105_siting/per_row_scores.npz"
V105_RECON_ARTIFACT = (
    "diagnostics_output/pact_place_v105_reconstruction/reconstruction.json"
)

# ---------------------------------------------------------------------------
# Streams. Distinct from every V10.5 stream.
# ---------------------------------------------------------------------------
REVIEW_STREAM = "pact-place-v10.6-asymmetric-review"
REVIEW_MASTER_SEED = 2026106002
PHASE0_STREAM = "pact-place-v10.6-asymmetric-phase0"
PHASE0_MASTER_SEED = 2026106001

N_PHASE0_ROWS = 24
N_REVIEW_ROWS = 48
N_REVIEW_REPLICATES = 2
N_REVIEW_VIDEOS = 6

# Phase-0 gate, inherited from V10.5.
PHASE0_MIN_CLEAN = 16
PHASE0_MIN_CLEAN_PER_SIDE = 7
PHASE0_MIN_CLEAN_PER_POSE = 4
PHASE0_MIN_CLEAN_PER_SIDE_POSE = 2
PHASE0_RISK_CONFIRM_MAX_M = 0.035

# Scaled review-pool yield floors. A 48-row pool must clear the same rate the
# 24-row gate will later demand, so a curated six-video packet cannot be
# published for an environment already unlikely to pass 16/24.
POOL_MIN_CLEAN = 32
POOL_MIN_CLEAN_PER_SIDE = 14
POOL_MIN_CLEAN_PER_POSE = 8
POOL_MIN_CLEAN_PER_SIDE_POSE = 4
POOL_N_PER_SIDE = 24
POOL_N_PER_POSE = 16
POOL_N_PER_SIDE_POSE = 8

INITIAL_FREE_SPACE_SPEED_CAP_M_S = 0.12
TASK_HORIZON_V106 = 1050
POLICY_TIMESTEP_MS = 66.0
REVIEW_FPS = 1000.0 / POLICY_TIMESTEP_MS
REVIEW_FRAME_STRIDE = 1

N_HISTORICAL_CLEAN_TRAJECTORIES = 98
N_EVALUATIONS_PER_BUNDLE = N_HISTORICAL_CLEAN_TRAJECTORIES * len(POSE_IDS)  # 294

SPEC_IMPLEMENTATION_PATHS: tuple[str, ...] = (
    PLAN_RELATIVE,
    "scripts/pact_place_v106_geometry.py",
    "scripts/pact_place_v106_contract.py",
    "scripts/pact_place_v105_clearance.py",
    "scripts/pact_place_v105_siting_core.py",
    "scripts/pact_place_v95_contract.py",
    "scripts/pact_geom_distance.py",
    BASE_SCENE_RELATIVE_V106,
    "submodules/molmospaces/molmo_spaces/data_generation/custom_scenes/"
    "pact_place_corridor_v3.xml",
    "submodules/molmospaces/molmo_spaces/tasks/enclosure_reach.py",
    "submodules/molmospaces/molmo_spaces/tasks/pact_place_contact_audit.py",
    "tests/test_pact_place_v106.py",
)


def file_hashes(paths: Sequence[str]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for relative in paths:
        target = ROOT / relative
        out[relative] = (
            {"raw_file_sha256": sha256_file(target), "present": True}
            if target.is_file()
            else {"raw_file_sha256": "absent", "present": False}
        )
    return out


def implementation_digest(paths: Sequence[str]) -> str:
    return sha256_payload(
        [[p, file_hashes([p])[p]["raw_file_sha256"]] for p in paths]
    )


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
    *, stream: str, master_seed: int, replicates: int = 1,
    scene_by_pose: dict[str, dict[str, str]] | None = None,
    assembly_by_pose: dict[str, str] | None = None,
    selected: dict[str, float] | None = None,
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
                "environment_version": ENVIRONMENT_VERSION_V106,
                "contract_version": CONTRACT_VERSION_V106,
                "sampler_class": SAMPLER_CLASS_V106,
                **{k: (dict(v) if isinstance(v, dict)
                       else list(v) if isinstance(v, list) else v)
                   for k, v in cache[key].items()},
            }
            if selected is not None:
                row["pact_v106_x_m"] = float(selected["x_m"])
                row["pact_v106_r_neg_m"] = float(selected["r_neg_m"])
                row["pact_v106_r_pos_m"] = float(selected["r_pos_m"])
            if scene_by_pose is not None:
                row["pact_v106_scene_relative"] = scene_by_pose[pose]["relative"]
                row["pact_v106_scene_sha256"] = scene_by_pose[pose]["sha256"]
            if assembly_by_pose is not None:
                row["pact_v106_assembly_sha256"] = assembly_by_pose[pose]
            row["episode_id"] = hashlib.sha256(
                f"{stream}:{master_seed}:{family}:{side}:{pose}:{replicate}".encode()
            ).hexdigest()
            row["row_sha256"] = sha256_payload(row)
            rows.append(row)
            index += 1
    return rows


def phase0_rows(**kw):
    return build_rows(stream=PHASE0_STREAM, master_seed=PHASE0_MASTER_SEED,
                      replicates=1, **kw)


def review_rows(**kw):
    return build_rows(stream=REVIEW_STREAM, master_seed=REVIEW_MASTER_SEED,
                      replicates=N_REVIEW_REPLICATES, **kw)


def streams_are_disjoint(review, gate) -> dict[str, Any]:
    r_ids = {r["episode_id"] for r in review}
    g_ids = {r["episode_id"] for r in gate}
    r_seeds = {r["task_seed_u64"] for r in review}
    g_seeds = {r["task_seed_u64"] for r in gate}
    return {
        "episode_id_overlap": sorted(r_ids & g_ids),
        "seed_overlap": sorted(r_seeds & g_seeds),
        "streams_differ": REVIEW_STREAM != PHASE0_STREAM,
        "disjoint": not (r_ids & g_ids) and not (r_seeds & g_seeds),
    }


# ---------------------------------------------------------------------------
# Preregistered offline admission
# ---------------------------------------------------------------------------
def admit_candidate(stats: dict[str, Any]) -> dict[str, Any]:
    """Universal-clearance preferred; otherwise the preregistered fallback.

    ``stats`` is whatever the aggregator produced for one three-pose bundle.
    This function encodes the rule and nothing else: it is written before any
    V10.6 score exists and must not be edited afterwards.
    """
    n_eval = int(stats["n_evaluations"])
    absolute_min = stats["absolute_min_clearance_m"]
    n_contacts = int(stats["n_contacts"])
    n_below = int(stats["n_below_floor"])
    fraction = stats["fraction_ge_floor"]
    groups = stats["evaluations_ge_floor_by_group"]
    band = stats["band_evaluations_by_group"]
    directions = stats["direction_band_witnesses"]

    universal = bool(
        n_contacts == 0
        and absolute_min is not None
        and float(absolute_min) >= CLEARANCE_FLOOR_M
        and n_below == 0
    )
    group_fractions = {
        key: (value["n_ge_floor"] / value["n"] if value["n"] else 0.0)
        for key, value in groups.items()
    }
    every_group_has_band_witness = all(int(v) > 0 for v in band.values())
    loaded_outbound_both_sides = all(
        "loaded_outbound" in directions.get(side, []) for side in INTRUSION_SIDES
    )
    windows_ok = int(stats["n_window_below_floor"]) == 0
    initial_ok = int(stats["n_initial_below_floor"]) == 0

    checks = {
        "zero_exact_contacts": n_contacts == FALLBACK_MAX_CONTACTS,
        "absolute_min_at_least_10mm": (
            absolute_min is not None
            and float(absolute_min) >= FALLBACK_ABSOLUTE_MIN_CLEARANCE_M
        ),
        "at_least_90pct_evaluations_ge_15mm": (
            fraction is not None
            and float(fraction) >= FALLBACK_MIN_FRACTION_GE_FLOOR
        ),
        "every_group_at_least_80pct_ge_15mm": all(
            value >= FALLBACK_MIN_GROUP_FRACTION_GE_FLOOR
            for value in group_fractions.values()
        ),
        "grasp_lift_release_universally_ge_15mm": windows_ok,
        "initial_state_universally_ge_15mm": initial_ok,
        "every_pose_side_group_has_band_witness": every_group_has_band_witness,
        "loaded_outbound_risk_on_both_sides": loaded_outbound_both_sides,
    }
    fallback_ok = all(checks.values())
    return {
        "n_evaluations": n_eval,
        "absolute_min_clearance_m": absolute_min,
        "n_contacts": n_contacts,
        "n_below_floor": n_below,
        "fraction_ge_floor": fraction,
        "group_fractions_ge_floor": group_fractions,
        "band_evaluations_by_group": band,
        "direction_band_witnesses": directions,
        "universal_clearance": universal,
        "fallback_checks": checks,
        "fallback_admissible": bool(fallback_ok),
        "inbound_risk_required": False,
        "admitted": bool(universal or fallback_ok),
        "admission_basis": (
            "universal_clearance" if universal
            else ("preregistered_fallback" if fallback_ok else None)
        ),
    }


def rank_key(stats: dict[str, Any], admission: dict[str, Any]):
    """Deterministic, untruncated ranking. Universal beats fallback always."""
    band_total = sum(int(v) for v in stats["band_evaluations_by_group"].values())
    return (
        0 if admission["universal_clearance"] else 1,
        -(float(stats["fraction_ge_floor"] or 0.0)),
        -(float(stats["absolute_min_clearance_m"] or 0.0)),
        -band_total,
        -float(stats["r_neg_m"]),
        -float(stats["r_pos_m"]),
    )


def pool_eligibility(rows, results) -> dict[str, Any]:
    """Scaled 48-row yield floors, checked before any packet is published."""
    by_role = {int(item["role_index"]): item for item in results}
    clean = 0
    by_side = dict.fromkeys(INTRUSION_SIDES, 0)
    by_pose = dict.fromkeys(POSE_IDS, 0)
    by_cell = {f"{s}|{p}": 0 for s in INTRUSION_SIDES for p in POSE_IDS}
    n_side = dict.fromkeys(INTRUSION_SIDES, 0)
    n_pose = dict.fromkeys(POSE_IDS, 0)
    n_cell = dict.fromkeys(by_cell, 0)
    incomplete = 0
    for row in rows:
        side, pose = row["intrusion_side"], row["pose_id"]
        n_side[side] += 1
        n_pose[pose] += 1
        n_cell[f"{side}|{pose}"] += 1
        result = by_role.get(int(row["role_index"]))
        if result is None:
            incomplete += 1
            continue
        if result.get("v106_clean_success"):
            clean += 1
            by_side[side] += 1
            by_pose[pose] += 1
            by_cell[f"{side}|{pose}"] += 1
    side_ok = all(by_side[s] >= POOL_MIN_CLEAN_PER_SIDE for s in INTRUSION_SIDES)
    pose_ok = all(by_pose[p] >= POOL_MIN_CLEAN_PER_POSE for p in POSE_IDS)
    cell_ok = all(v >= POOL_MIN_CLEAN_PER_SIDE_POSE for v in by_cell.values())
    limiting = []
    if clean < POOL_MIN_CLEAN:
        limiting.append(f"pool clean {clean} < {POOL_MIN_CLEAN}")
    if not side_ok:
        limiting.append(f"per-side floor {by_side} < {POOL_MIN_CLEAN_PER_SIDE}")
    if not pose_ok:
        limiting.append(f"per-pose floor {by_pose} < {POOL_MIN_CLEAN_PER_POSE}")
    if not cell_ok:
        limiting.append(
            f"side x pose floor {by_cell} < {POOL_MIN_CLEAN_PER_SIDE_POSE}"
        )
    if incomplete:
        limiting.append(f"{incomplete} incomplete rows")
    low, high = wilson_interval(clean, len(rows) or 1)
    return {
        **empty_authorization(),
        "n_rows": len(rows), "n_results": len(results),
        "clean_successes": clean,
        "clean_by_side": by_side, "clean_by_pose": by_pose,
        "clean_by_side_pose": by_cell,
        "n_by_side": n_side, "n_by_pose": n_pose, "n_by_side_pose": n_cell,
        "min_clean_required": POOL_MIN_CLEAN,
        "min_clean_per_side": POOL_MIN_CLEAN_PER_SIDE,
        "min_clean_per_pose": POOL_MIN_CLEAN_PER_POSE,
        "min_clean_per_side_pose": POOL_MIN_CLEAN_PER_SIDE_POSE,
        "wilson_95_interval": [low, high],
        "incomplete_rows": incomplete,
        "limiting_predicates": limiting,
        "scaled_from_phase0_bar": f"{PHASE0_MIN_CLEAN}/{N_PHASE0_ROWS}",
        "pool_passed": not limiting,
    }


def build_specification_contract() -> dict[str, Any]:
    return {
        "schema_version": "pact_place_v106_specification_contract_v1",
        "stage": "specification",
        "contract_version": CONTRACT_VERSION_V106,
        "environment_version": ENVIRONMENT_VERSION_V106,
        "sampler_class": SAMPLER_CLASS_V106,
        "supersedes": "pact_place_v105_v95_clutter_static_pendant_v1",
        "v105_narrative_treated_as_untrusted": True,
        "lineage": {
            "base_scene": BASE_SCENE_RELATIVE_V106,
            "sampler_behavior": "PactPlaceCorridorV93Sampler",
            "palette": "load_v95_palette",
            "layouts": "build_v95_layout",
            "uses_v95_low_wall": False,
            "per_family_placement": False,
            "global_asymmetric_placement": True,
        },
        "lattice": {
            "x_m": list(LATTICE_X_M),
            "r_neg_m": list(LATTICE_R_NEG_M),
            "r_pos_m": list(LATTICE_R_POS_M),
            "pose_offsets_m": dict(POSE_OFFSETS_M),
            "n_candidates": len(lattice_candidates()),
            "n_scenes": len(lattice_candidates()) * len(POSE_IDS),
            "may_be_extended_after_results": False,
        },
        "admission": {
            "preferred": "universal >=15 mm clearance",
            "fallback_preregistered_before_any_scoring": True,
            "fallback_max_contacts": FALLBACK_MAX_CONTACTS,
            "fallback_absolute_min_clearance_m": FALLBACK_ABSOLUTE_MIN_CLEARANCE_M,
            "fallback_min_fraction_ge_floor": FALLBACK_MIN_FRACTION_GE_FLOOR,
            "fallback_min_group_fraction_ge_floor":
                FALLBACK_MIN_GROUP_FRACTION_GE_FLOOR,
            "requires_inbound_risk": False,
            "requires_loaded_outbound_risk_both_sides": True,
            "rationale_for_dropping_inbound": (
                "The sealed V10.5 evidence shows the meaningful near-pass is "
                "predominantly loaded outbound."
            ),
        },
        "evaluation_accounting": {
            "n_historical_clean_trajectories": N_HISTORICAL_CLEAN_TRAJECTORIES,
            "n_poses": len(POSE_IDS),
            "n_evaluations_per_bundle": N_EVALUATIONS_PER_BUNDLE,
            "band_evaluations_by_group_definition": (
                "count of (trajectory, pose) evaluations whose lobe/stem "
                "minimum lies inside the 15-35 mm band, per pose_id|side group"
            ),
        },
        "review_pool_floors": {
            "n_rows": N_REVIEW_ROWS,
            "min_clean": POOL_MIN_CLEAN,
            "min_clean_per_side": POOL_MIN_CLEAN_PER_SIDE,
            "min_clean_per_pose": POOL_MIN_CLEAN_PER_POSE,
            "min_clean_per_side_pose": POOL_MIN_CLEAN_PER_SIDE_POSE,
            "checked_before_publishing_any_packet": True,
        },
        "phase0": {
            "n_rows": N_PHASE0_ROWS, "min_clean": PHASE0_MIN_CLEAN,
            "min_clean_per_side": PHASE0_MIN_CLEAN_PER_SIDE,
            "min_clean_per_pose": PHASE0_MIN_CLEAN_PER_POSE,
            "min_clean_per_side_pose": PHASE0_MIN_CLEAN_PER_SIDE_POSE,
        },
        "streams": {
            "review_stream": REVIEW_STREAM,
            "review_master_seed": REVIEW_MASTER_SEED,
            "phase0_stream": PHASE0_STREAM,
            "phase0_master_seed": PHASE0_MASTER_SEED,
        },
        "immutable_inputs": file_hashes(
            (FRAGILITY_ARTIFACT, BASE_SCENE_RELATIVE_V106,
             V105_AUDIT_ARTIFACT, V105_SITING_ARTIFACT, V105_SCORES_ARTIFACT,
             V105_RECON_ARTIFACT)
        ),
        "v105_audit_payload_sha256": (
            recompute_payload_sha256(ROOT / V105_AUDIT_ARTIFACT)
            if (ROOT / V105_AUDIT_ARTIFACT).is_file() else "absent"
        ),
        "implementation_files": file_hashes(SPEC_IMPLEMENTATION_PATHS),
        "implementation_digest": implementation_digest(SPEC_IMPLEMENTATION_PATHS),
        "does_not_modify_v104_or_v105_artifacts": True,
        **empty_authorization(),
    }


__all__ = [
    "CERT_ROOT", "CAUSAL_ROOT", "ImmutableArtifactError",
    "INITIAL_FREE_SPACE_SPEED_CAP_M_S", "INTRUSION_SIDES",
    "N_EVALUATIONS_PER_BUNDLE", "N_PHASE0_ROWS", "N_REVIEW_ROWS",
    "N_REVIEW_VIDEOS", "PHASE0_MIN_CLEAN", "PHASE0_MIN_CLEAN_PER_POSE",
    "PHASE0_MIN_CLEAN_PER_SIDE", "PHASE0_MIN_CLEAN_PER_SIDE_POSE",
    "PHASE0_RISK_CONFIRM_MAX_M", "PHASE0_ROOT", "PHASE0_STREAM", "PLAN_RELATIVE",
    "POOL_MIN_CLEAN", "POOL_MIN_CLEAN_PER_POSE", "POOL_MIN_CLEAN_PER_SIDE",
    "POOL_MIN_CLEAN_PER_SIDE_POSE", "POOL_ROOT", "REVIEW_FPS", "REVIEW_ROOT",
    "REVIEW_STREAM", "ROOT", "SITING_ROOT", "SPEC_IMPLEMENTATION_PATHS",
    "TASK_HORIZON_V106", "V105_AUDIT_ARTIFACT", "V105_RECON_ARTIFACT",
    "V105_SCORES_ARTIFACT", "V105_SITING_ARTIFACT", "V95_LAYOUT_FAMILY_IDS",
    "admit_candidate", "build_rows", "build_specification_contract",
    "canonical_payload_sha256", "cells", "empty_authorization", "file_hashes",
    "implementation_digest", "phase0_rows", "pool_eligibility", "rank_key",
    "recompute_payload_sha256", "review_rows", "sha256_file", "sha256_payload",
    "streams_are_disjoint", "v95_row_payload", "wilson_interval",
    "write_immutable_create_only", "write_immutable_text_create_only",
]
