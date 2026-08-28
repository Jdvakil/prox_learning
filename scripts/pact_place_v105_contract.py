#!/usr/bin/env python3
"""V10.5 contract: two acyclic stages, explicit file lists, no circular hash.

Stage 1 is the *specification* contract: it binds this plan, the immutable
inputs, the lattice, the predicates, the ranking, and the implementation file
list — everything that must be fixed before the offline search can be trusted.

Stage 2 is the *execution* contract: it binds the specification plus the
reconstruction, siting and causal outputs, the selected scenes, the review and
Phase-0 runner code, the manifest derivation, and every threshold. It is frozen
before the first live row.

Neither stage hashes itself into its own inputs. For every JSON artifact the
raw-file SHA-256 and the canonical payload SHA-256 (computed with the hash
field omitted) are stored separately and never conflated — a pre-insertion
self-hash is neither of those and is labelled as such wherever it appears.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

for _p in (ROOT / "scripts", ROOT / "submodules" / "molmospaces"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from pact_place_v105_geometry import (  # noqa: E402
    BASE_SCENE_RELATIVE_V105,
    CLEARANCE_FLOOR_M,
    CONTRACT_VERSION_V105,
    ENVIRONMENT_VERSION_V105,
    LATTICE_R_M,
    LATTICE_X_M,
    POSE_IDS,
    POSE_OFFSETS_M,
    POSE_ORDERING_MIN_SEPARATION_M,
    RISK_BAND_M,
    SAMPLER_CLASS_V105,
    lattice_candidates,
)

PLAN_RELATIVE = "docs/PACT_PLACE_V105_V95_CLUTTER_STATIC_PENDANT_PLAN.md"

# ---------------------------------------------------------------------------
# Output roots
# ---------------------------------------------------------------------------
RECONSTRUCTION_ROOT = "diagnostics_output/pact_place_v105_reconstruction"
SITING_ROOT = "diagnostics_output/pact_place_v105_siting"
CAUSAL_ROOT = "diagnostics_output/pact_place_v105_causal"
REVIEW_ROOT = "diagnostics_output/pact_place_v105_review"
PHASE0_ROOT = "diagnostics_output/pact_place_v105_phase0"
SCENE_OUTPUT_DIR = (
    "submodules/molmospaces/molmo_spaces/data_generation/custom_scenes"
)

# ---------------------------------------------------------------------------
# Immutable inputs
# ---------------------------------------------------------------------------
FRAGILITY_ARTIFACT = (
    "diagnostics_output/pact_place_v95_seed_fragility/fragility.json"
)
FRAGILITY_ROWS_DIR = (
    "diagnostics_output/pact_place_v95_seed_fragility/expert_screen_rows"
)
V95_LAYOUT_FAMILY_IDS: tuple[str, ...] = (
    "F0_target_side_stagger",
    "F1_inner_panel_stagger",
    "F2_outer_panel_stagger",
    "F3_aperture_side_stagger",
)
INTRUSION_SIDES: tuple[str, ...] = ("left", "right")

# ---------------------------------------------------------------------------
# Streams and seeds
# ---------------------------------------------------------------------------
REVIEW_STREAM = "pact-place-v10.5-v95-clutter-review"
REVIEW_MASTER_SEED = 2026105002
PHASE0_STREAM = "pact-place-v10.5-v95-clutter-phase0"
PHASE0_MASTER_SEED = 2026105001

N_PHASE0_ROWS = 24
N_REVIEW_ROWS = 48
N_REVIEW_REPLICATES = 2
N_REVIEW_VIDEOS = 6

# ---------------------------------------------------------------------------
# Gate thresholds. Frozen; never changed in response to an outcome.
# ---------------------------------------------------------------------------
PHASE0_MIN_CLEAN = 16
PHASE0_MIN_CLEAN_PER_SIDE = 7
PHASE0_N_PER_SIDE = 12
PHASE0_MIN_CLEAN_PER_POSE = 4
PHASE0_N_PER_POSE = 8
PHASE0_MIN_CLEAN_PER_SIDE_POSE = 2
PHASE0_N_PER_SIDE_POSE = 4
PHASE0_RISK_CONFIRM_MAX_M = 0.035

# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
INITIAL_FREE_SPACE_SPEED_CAP_M_S = 0.12
POLICY_TIMESTEP_MS = 66.0
REVIEW_FPS = 1000.0 / POLICY_TIMESTEP_MS
REVIEW_FRAME_STRIDE = 1
TASK_HORIZON_V105 = 1050

CAUSAL_MIN_CHANGED_VALUES = 448
CAUSAL_MIN_CHANGED_SENSORS = 3
CAUSAL_REQUIRED_LINK_SENSORS = ("link5", "link6")
CAUSAL_MIN_ONSET_FRAMES = 5
CAUSAL_MIN_ONSET_SECONDS = 0.10
CAUSAL_MAX_SIDE_RATIO = 4.0
PROXIMITY_TENSOR_SHAPE = (40, 4, 8, 8)

CONTACT_CERTIFICATE_MAGNITUDES_M = (0.005, 0.010, 0.015, 0.020, 0.025, 0.030)

DISALLOWED_CONTACT_CLASSES = (
    "mounted_fixture",
    "clutter",
    "hazard_bar",
    "other_environment",
)

# ---------------------------------------------------------------------------
# Explicit ordered implementation file lists. Unrelated repository changes
# must not silently alter a contract, so nothing globs.
# ---------------------------------------------------------------------------
SPEC_IMPLEMENTATION_PATHS: tuple[str, ...] = (
    PLAN_RELATIVE,
    "scripts/pact_place_v105_geometry.py",
    "scripts/pact_place_v105_contract.py",
    "scripts/pact_place_v105_clearance.py",
    "scripts/pact_place_v95_contract.py",
    "scripts/pact_geom_distance.py",
    BASE_SCENE_RELATIVE_V105,
    "submodules/molmospaces/molmo_spaces/data_generation/custom_scenes/"
    "pact_place_corridor_v3.xml",
    "submodules/molmospaces/molmo_spaces/tasks/enclosure_reach.py",
    "submodules/molmospaces/molmo_spaces/tasks/pact_place_contact_audit.py",
    "tests/test_pact_place_v105.py",
)

EXECUTION_IMPLEMENTATION_PATHS: tuple[str, ...] = SPEC_IMPLEMENTATION_PATHS + (
    "scripts/run_pact_place_v105_reconstruct.py",
    "scripts/run_pact_place_v105_siting.py",
    "scripts/run_pact_place_v105_review.py",
    "scripts/run_pact_place_v105_phase0.py",
    "scripts/run_pact_place_expert_screen.py",
)


# ---------------------------------------------------------------------------
# Hashing. Raw-file and payload hashes are distinct and never interchanged.
# ---------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    """SHA-256 of the file's bytes exactly as they sit on disk."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_payload(value: Any) -> str:
    """SHA-256 of a canonical JSON encoding of a value."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def canonical_payload_sha256(document: dict[str, Any]) -> str:
    """A document's payload hash, with its own hash field omitted."""
    return sha256_payload(
        {k: v for k, v in document.items() if k != "payload_sha256"}
    )


def recompute_payload_sha256(path: Path) -> str:
    """Recompute a stored artifact's payload hash from its bytes."""
    return canonical_payload_sha256(json.loads(Path(path).read_text()))


class ImmutableArtifactError(RuntimeError):
    """Refused to replace an artifact that already exists."""


def write_immutable_create_only(path: Path, document: dict[str, Any]) -> dict[str, str]:
    """Atomic create-if-absent. Returns both hashes, labelled distinctly."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: v for k, v in document.items() if k != "payload_sha256"}
    digest = sha256_payload(payload)
    payload["payload_sha256"] = digest
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
    return {"payload_sha256": digest, "raw_file_sha256": sha256_file(target)}


def write_immutable_text_create_only(path: Path, text: str) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
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
    return sha256_file(target)


def empty_authorization() -> dict[str, bool]:
    """Every authorization defaults false.

    Callers must spread this FIRST and set outcome keys afterwards. Spreading
    it last silently resets a passing result — a defect this project has
    already shipped once.
    """
    return {
        "eligible_for_human_review": False,
        "human_approval_present": False,
        "authorizes_phase0": False,
        "authorizes_gate": False,
        "authorizes_collection": False,
        "authorizes_conversion": False,
        "authorizes_training": False,
        "authorizes_evaluation": False,
        "phase0_passed": False,
    }


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
    """Digest of an explicit ordered file list, not of a directory scan."""
    return sha256_payload(
        [[relative, file_hashes([relative])[relative]["raw_file_sha256"]]
         for relative in paths]
    )


# ---------------------------------------------------------------------------
# Manifests
# ---------------------------------------------------------------------------
def _row_seed(stream: str, master_seed: int, index: int) -> tuple[int, int]:
    digest = hashlib.sha256(f"{stream}:{master_seed}:{index}".encode()).digest()
    seed_u64 = int.from_bytes(digest[:8], "big")
    return seed_u64 % (2**32), seed_u64


def phase0_cells() -> list[tuple[str, str, str]]:
    """The exact Cartesian product: 4 families x 2 sides x 3 poses = 24."""
    return [
        (family, side, pose)
        for family in V95_LAYOUT_FAMILY_IDS
        for side in INTRUSION_SIDES
        for pose in POSE_IDS
    ]


# The V9.3 per-family vessel jitter, inherited verbatim from the settled V9.5
# lineage. Reproduced here rather than imported so the contract binds an
# explicit value, and asserted against the source helper by the tests.
V95_VESSEL_JITTER = (
    ({"01": -0.015, "06": -0.004}, {"01": -0.004, "06": 0.009}),
    ({"01": -0.005, "06": 0.003}, {"01": 0.003, "06": -0.006}),
    ({"01": 0.006, "06": -0.002}, {"01": -0.002, "06": 0.0045}),
    ({"01": 0.015, "06": 0.004}, {"01": 0.004, "06": -0.009}),
)


def v95_row_payload(family_id: str, intrusion_side: str) -> dict[str, Any]:
    """The settled V9.5 palette, layout and jitter for one family/side cell."""
    from pact_place_v95_contract import build_v95_layout, load_v95_palette

    palette = load_v95_palette()
    layout = build_v95_layout(
        palette, family_id=family_id, intrusion_side=intrusion_side
    )
    jitter = V95_VESSEL_JITTER[V95_LAYOUT_FAMILY_IDS.index(family_id)]
    return {
        "family": family_id,
        "layout_family_id": family_id,
        "layout_id": layout["layout_id"],
        "family_attempt": 0,
        "scene_template_house_index": 1,
        "max_sampling_retries": 12,
        "clutter_x_jitter_m": dict(jitter[0]),
        "clutter_y_jitter_m": dict(jitter[1]),
        "panel_face_jitter_m": 0.0,
        "panel_x_jitter_m": 0.0,
        "target_x_jitter_m": 0.0,
        "target_y_jitter_m": 0.0,
        "pact_clutter_palette": list(palette["palette"]),
        "pact_clutter_layout": layout,
    }


def build_rows(
    *,
    stream: str,
    master_seed: int,
    replicates: int = 1,
    scene_by_pose: dict[str, dict[str, str]] | None = None,
    assembly_by_pose: dict[str, str] | None = None,
    selected_x_m: float | None = None,
    selected_r_m: float | None = None,
) -> list[dict[str, Any]]:
    """Deterministic manifest rows over the family x side x pose product.

    Each row is a settled V9.5 row plus the pose/scene binding: the clutter
    palette, layout, and vessel jitter are the inherited ones, so nothing about
    the household-object environment is invented here.
    """
    cells = phase0_cells()
    rows: list[dict[str, Any]] = []
    index = 0
    payload_cache: dict[tuple[str, str], dict[str, Any]] = {}
    for replicate in range(int(replicates)):
        for family, side, pose in cells:
            seed_u32, seed_u64 = _row_seed(stream, master_seed, index)
            key = (family, side)
            if key not in payload_cache:
                payload_cache[key] = v95_row_payload(family, side)
            row: dict[str, Any] = {
                "role_index": index,
                "replicate": replicate,
                "family_id": family,
                "intrusion_side": side,
                "pose_id": pose,
                "pose_offset_m": POSE_OFFSETS_M[pose],
                "seed_stream": stream,
                "task_seed_u32": int(seed_u32),
                "task_seed_u64": int(seed_u64),
                "environment_version": ENVIRONMENT_VERSION_V105,
                "contract_version": CONTRACT_VERSION_V105,
                "sampler_class": SAMPLER_CLASS_V105,
                **{k: (dict(v) if isinstance(v, dict) else
                       list(v) if isinstance(v, list) else v)
                   for k, v in payload_cache[key].items()},
            }
            if selected_x_m is not None:
                row["pact_v105_x_m"] = float(selected_x_m)
            if selected_r_m is not None:
                row["pact_v105_r_m"] = float(selected_r_m)
            if scene_by_pose is not None:
                row["pact_v105_scene_relative"] = scene_by_pose[pose]["relative"]
                row["pact_v105_scene_sha256"] = scene_by_pose[pose]["sha256"]
            if assembly_by_pose is not None:
                row["pact_v105_assembly_sha256"] = assembly_by_pose[pose]
            row["episode_id"] = hashlib.sha256(
                f"{stream}:{master_seed}:{family}:{side}:{pose}:{replicate}".encode()
            ).hexdigest()
            row["row_sha256"] = sha256_payload(row)
            rows.append(row)
            index += 1
    return rows


def phase0_rows(**kwargs) -> list[dict[str, Any]]:
    return build_rows(
        stream=PHASE0_STREAM, master_seed=PHASE0_MASTER_SEED, replicates=1, **kwargs
    )


def review_rows(**kwargs) -> list[dict[str, Any]]:
    return build_rows(
        stream=REVIEW_STREAM,
        master_seed=REVIEW_MASTER_SEED,
        replicates=N_REVIEW_REPLICATES,
        **kwargs,
    )


def streams_are_disjoint(review, gate) -> dict[str, Any]:
    review_ids = {row["episode_id"] for row in review}
    gate_ids = {row["episode_id"] for row in gate}
    review_seeds = {row["task_seed_u64"] for row in review}
    gate_seeds = {row["task_seed_u64"] for row in gate}
    return {
        "episode_id_overlap": sorted(review_ids & gate_ids),
        "seed_overlap": sorted(review_seeds & gate_seeds),
        "streams_differ": REVIEW_STREAM != PHASE0_STREAM,
        "disjoint": not (review_ids & gate_ids) and not (review_seeds & gate_seeds),
    }


# ---------------------------------------------------------------------------
# Gate accounting
# ---------------------------------------------------------------------------
def gate_eligibility(rows, results) -> dict[str, Any]:
    """Phase-0 counting at 16/24 with every balance floor."""
    by_role = {int(item["role_index"]): item for item in results}
    clean_by_side: dict[str, int] = {side: 0 for side in INTRUSION_SIDES}
    clean_by_pose: dict[str, int] = {pose: 0 for pose in POSE_IDS}
    clean_by_cell: dict[str, int] = {
        f"{side}|{pose}": 0 for side in INTRUSION_SIDES for pose in POSE_IDS
    }
    clean_by_family: dict[str, int] = {f: 0 for f in V95_LAYOUT_FAMILY_IDS}
    n_by_side = dict.fromkeys(INTRUSION_SIDES, 0)
    n_by_pose = dict.fromkeys(POSE_IDS, 0)
    n_by_cell = dict.fromkeys(clean_by_cell, 0)
    clean = 0
    failures: list[dict[str, Any]] = []
    risk_by_side: dict[str, float] = {}
    risk_by_pose: dict[str, float] = {}
    incomplete = 0
    for row in rows:
        role = int(row["role_index"])
        side, pose, family = row["intrusion_side"], row["pose_id"], row["family_id"]
        n_by_side[side] += 1
        n_by_pose[pose] += 1
        n_by_cell[f"{side}|{pose}"] += 1
        result = by_role.get(role)
        if result is None:
            incomplete += 1
            failures.append({"role_index": role, "reason": "row missing"})
            continue
        defects = result.get("v105_defects") or []
        if result.get("v105_clean_success"):
            clean += 1
            clean_by_side[side] += 1
            clean_by_pose[pose] += 1
            clean_by_cell[f"{side}|{pose}"] += 1
            clean_by_family[family] += 1
        else:
            failures.append(
                {"role_index": role, "side": side, "pose_id": pose,
                 "family_id": family, "defects": defects}
            )
        telemetry = result.get("pact_v105_frame_telemetry") or {}
        closest = telemetry.get("min_lobe_stem_clearance_m")
        if closest is not None:
            risk_by_side[side] = min(risk_by_side.get(side, 1e9), float(closest))
            risk_by_pose[pose] = min(risk_by_pose.get(pose, 1e9), float(closest))
    side_ok = all(
        clean_by_side[s] >= PHASE0_MIN_CLEAN_PER_SIDE for s in INTRUSION_SIDES
    )
    pose_ok = all(clean_by_pose[p] >= PHASE0_MIN_CLEAN_PER_POSE for p in POSE_IDS)
    cell_ok = all(
        value >= PHASE0_MIN_CLEAN_PER_SIDE_POSE for value in clean_by_cell.values()
    )
    risk_side_ok = all(
        risk_by_side.get(s, 1e9) <= PHASE0_RISK_CONFIRM_MAX_M for s in INTRUSION_SIDES
    )
    risk_pose_ok = all(
        risk_by_pose.get(p, 1e9) <= PHASE0_RISK_CONFIRM_MAX_M for p in POSE_IDS
    )
    passed = bool(
        clean >= PHASE0_MIN_CLEAN
        and side_ok
        and pose_ok
        and cell_ok
        and risk_side_ok
        and risk_pose_ok
        and incomplete == 0
        and len(results) == len(rows) == N_PHASE0_ROWS
    )
    limiting = []
    if clean < PHASE0_MIN_CLEAN:
        limiting.append(f"clean {clean} < {PHASE0_MIN_CLEAN}")
    if not side_ok:
        limiting.append(f"per-side floor: {clean_by_side}")
    if not pose_ok:
        limiting.append(f"per-pose floor: {clean_by_pose}")
    if not cell_ok:
        limiting.append(f"side x pose floor: {clean_by_cell}")
    if not risk_side_ok:
        limiting.append(f"per-side risk confirmation: {risk_by_side}")
    if not risk_pose_ok:
        limiting.append(f"per-pose risk confirmation: {risk_by_pose}")
    if incomplete:
        limiting.append(f"{incomplete} incomplete rows")
    return {
        **empty_authorization(),
        "n_rows": len(rows),
        "n_results": len(results),
        "clean_successes": clean,
        "clean_by_side": clean_by_side,
        "clean_by_pose": clean_by_pose,
        "clean_by_side_pose": clean_by_cell,
        "clean_by_family": clean_by_family,
        "n_by_side": n_by_side,
        "n_by_pose": n_by_pose,
        "n_by_side_pose": n_by_cell,
        "min_clean_required": PHASE0_MIN_CLEAN,
        "min_clean_per_side": PHASE0_MIN_CLEAN_PER_SIDE,
        "min_clean_per_pose": PHASE0_MIN_CLEAN_PER_POSE,
        "min_clean_per_side_pose": PHASE0_MIN_CLEAN_PER_SIDE_POSE,
        "closest_lobe_stem_by_side_m": risk_by_side,
        "closest_lobe_stem_by_pose_m": risk_by_pose,
        "risk_confirmation_max_m": PHASE0_RISK_CONFIRM_MAX_M,
        "incomplete_rows": incomplete,
        "failures": failures,
        "limiting_predicates": limiting,
        "phase0_passed": passed,
    }


def wilson_interval(successes: int, total: int, z: float = 1.96):
    """Exact Wilson score interval; no normal approximation to the mean."""
    if total <= 0:
        return (0.0, 0.0)
    phat = successes / total
    denominator = 1.0 + z * z / total
    centre = (phat + z * z / (2 * total)) / denominator
    margin = (
        z
        * ((phat * (1 - phat) / total + z * z / (4 * total * total)) ** 0.5)
        / denominator
    )
    return (max(0.0, centre - margin), min(1.0, centre + margin))


# ---------------------------------------------------------------------------
# Specification contract
# ---------------------------------------------------------------------------
def build_specification_contract() -> dict[str, Any]:
    """Stage 1. Binds only inputs that already exist."""
    document = {
        "schema_version": "pact_place_v105_specification_contract_v1",
        "stage": "specification",
        "contract_version": CONTRACT_VERSION_V105,
        "environment_version": ENVIRONMENT_VERSION_V105,
        "sampler_class": SAMPLER_CLASS_V105,
        "lineage": {
            "base_scene": BASE_SCENE_RELATIVE_V105,
            "sampler_behavior": "PactPlaceCorridorV93Sampler",
            "palette": "load_v95_palette",
            "layouts": "build_v95_layout",
            "uses_v95_low_wall": False,
            "uses_v94_v95_wall_fixture": False,
            "uses_old_ceiling_mount": False,
            "imports_v98_to_v103_route_branch": False,
            "note": (
                "The fixture-free settled V9.5 clutter lineage. The 51% "
                "seed-robustness result came from this lineage, not from the "
                "V9.5 low wall."
            ),
        },
        "lattice": {
            "x_m": list(LATTICE_X_M),
            "r_m": list(LATTICE_R_M),
            "pose_offsets_m": dict(POSE_OFFSETS_M),
            "pose_ids": list(POSE_IDS),
            "n_candidates": len(lattice_candidates()),
            "n_scenes": len(lattice_candidates()) * len(POSE_IDS),
            "may_be_extended_after_results": False,
            "searchable_dimensions": ["x", "r"],
            "frozen_dimensions": [
                "z", "lobe half extents", "stem thickness", "crossbar thickness",
                "route parameters", "clutter positions", "thresholds",
            ],
        },
        "predicates": {
            "clearance_floor_m": CLEARANCE_FLOOR_M,
            "risk_band_m": list(RISK_BAND_M),
            "pose_ordering_min_separation_m": POSE_ORDERING_MIN_SEPARATION_M,
            "contact_certificate_magnitudes_m": list(
                CONTACT_CERTIFICATE_MAGNITUDES_M
            ),
            "min_clean_per_cell_for_siting": 2,
        },
        "causal": {
            "tensor_shape": list(PROXIMITY_TENSOR_SHAPE),
            "min_changed_values": CAUSAL_MIN_CHANGED_VALUES,
            "min_changed_sensors": CAUSAL_MIN_CHANGED_SENSORS,
            "required_link_sensors": list(CAUSAL_REQUIRED_LINK_SENSORS),
            "min_onset_frames": CAUSAL_MIN_ONSET_FRAMES,
            "min_onset_seconds": CAUSAL_MIN_ONSET_SECONDS,
            "max_side_ratio": CAUSAL_MAX_SIDE_RATIO,
            "geometry_proxy_alone_is_insufficient": True,
        },
        "ranking": [
            "maximize risk-band witnesses across side/direction",
            "minimize |median lobe/stem clearance - 25 mm|",
            "minimize left/right median imbalance",
            "maximize raw causal changed values at earliest valid onset",
            "prefer larger r, then larger x",
        ],
        "ranking_truncation": None,
        "thresholds": {
            "phase0_min_clean": PHASE0_MIN_CLEAN,
            "phase0_n_rows": N_PHASE0_ROWS,
            "phase0_min_clean_per_side": PHASE0_MIN_CLEAN_PER_SIDE,
            "phase0_min_clean_per_pose": PHASE0_MIN_CLEAN_PER_POSE,
            "phase0_min_clean_per_side_pose": PHASE0_MIN_CLEAN_PER_SIDE_POSE,
            "phase0_risk_confirmation_max_m": PHASE0_RISK_CONFIRM_MAX_M,
            "review_pool_rows": N_REVIEW_ROWS,
            "review_videos": N_REVIEW_VIDEOS,
        },
        "streams": {
            "review_stream": REVIEW_STREAM,
            "review_master_seed": REVIEW_MASTER_SEED,
            "phase0_stream": PHASE0_STREAM,
            "phase0_master_seed": PHASE0_MASTER_SEED,
        },
        "runtime": {
            "initial_free_space_speed_cap_m_s": INITIAL_FREE_SPACE_SPEED_CAP_M_S,
            "other_segment_speeds_unchanged": True,
            "pendant_in_planner_obstacle_list": False,
            "pendant_specific_lane_or_route_search": False,
            "task_horizon": TASK_HORIZON_V105,
        },
        "immutable_inputs": file_hashes(
            (FRAGILITY_ARTIFACT, BASE_SCENE_RELATIVE_V105)
        ),
        "implementation_files": file_hashes(SPEC_IMPLEMENTATION_PATHS),
        "implementation_digest": implementation_digest(SPEC_IMPLEMENTATION_PATHS),
        "hash_discipline": {
            "raw_file_and_payload_hashes_stored_separately": True,
            "pre_insertion_self_hash_is_labelled_as_such": True,
            "hashes_an_explicit_ordered_file_list": True,
            "circular_aggregate_hash": False,
        },
        "historical_discrepancy_preserved": {
            "note": (
                "The 7/8-versus-6/8 validated-seed discrepancy is recorded, "
                "not resolved. Strict-clean status is read from each retained "
                "source row's own telemetry, with that row's file hash."
            ),
            "turned_into_a_truth_claim": False,
        },
        **empty_authorization(),
    }
    return document


__all__ = [
    "CAUSAL_MAX_SIDE_RATIO",
    "CAUSAL_MIN_CHANGED_SENSORS",
    "CAUSAL_MIN_CHANGED_VALUES",
    "CAUSAL_MIN_ONSET_FRAMES",
    "CAUSAL_MIN_ONSET_SECONDS",
    "CAUSAL_ROOT",
    "CONTACT_CERTIFICATE_MAGNITUDES_M",
    "DISALLOWED_CONTACT_CLASSES",
    "EXECUTION_IMPLEMENTATION_PATHS",
    "FRAGILITY_ARTIFACT",
    "FRAGILITY_ROWS_DIR",
    "ImmutableArtifactError",
    "INITIAL_FREE_SPACE_SPEED_CAP_M_S",
    "INTRUSION_SIDES",
    "N_PHASE0_ROWS",
    "N_REVIEW_ROWS",
    "N_REVIEW_VIDEOS",
    "PHASE0_MIN_CLEAN",
    "PHASE0_MIN_CLEAN_PER_POSE",
    "PHASE0_MIN_CLEAN_PER_SIDE",
    "PHASE0_MIN_CLEAN_PER_SIDE_POSE",
    "PHASE0_RISK_CONFIRM_MAX_M",
    "PHASE0_ROOT",
    "PHASE0_STREAM",
    "PLAN_RELATIVE",
    "PROXIMITY_TENSOR_SHAPE",
    "RECONSTRUCTION_ROOT",
    "REVIEW_FPS",
    "REVIEW_ROOT",
    "REVIEW_STREAM",
    "ROOT",
    "SCENE_OUTPUT_DIR",
    "SITING_ROOT",
    "SPEC_IMPLEMENTATION_PATHS",
    "TASK_HORIZON_V105",
    "V95_LAYOUT_FAMILY_IDS",
    "build_rows",
    "build_specification_contract",
    "canonical_payload_sha256",
    "empty_authorization",
    "file_hashes",
    "gate_eligibility",
    "implementation_digest",
    "phase0_cells",
    "phase0_rows",
    "recompute_payload_sha256",
    "review_rows",
    "sha256_file",
    "v95_row_payload",
    "V95_VESSEL_JITTER",
    "sha256_payload",
    "streams_are_disjoint",
    "wilson_interval",
    "write_immutable_create_only",
    "write_immutable_text_create_only",
]
