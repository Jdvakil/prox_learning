"""Pure helpers for the V9.8 offset-pendant retained-qpos diagnostic.

No MuJoCo, no episode runner, no physics step. Geometry constants and
routing are not modified here.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

MAX_TCP_RESIDUAL_M = 0.001
LAG_REPRODUCE_TOLERANCE_M = 0.002
SCHEMA_VERSION = "pact_place_v9_8_offset_contact_diagnosis_v1"
SCENE_XML_NAME = "pact_place_corridor_v5.xml"
MOUNT_FIXTURE_BODY_PREFIX = "pact_clutter_mount_"
LINK6_BODY_NAME = "robot_0/fr3_link6"
LINK6_BODY_LABEL = "fr3_link6_body_origin"

WORKING_CONCLUSION = (
    "Both pre-registered offset candidates failed the paired preservation "
    "rule, so V9.8 stops before the 24-row gate. The failure mechanism and "
    "the physical validity of the lag-derived window remain unresolved."
)

CEILING_FIXTURE_PHASES = (
    "inbound_ceiling_fixture_approach",
    "inbound_ceiling_fixture_pass",
    "inbound_ceiling_fixture_exit",
)
POST_BOW_PHASES = frozenset({"pregrasp", "grasp"})
EARLY_PHASE_PREFIXES = (
    "inbound_vessel_",
    "inbound_cross_vessel_",
)
ONSET_CATEGORIES = (
    "early_approach_coverage",
    "protected_ceiling_bow_contact",
    "post_bow_pregrasp_coverage",
    "unreconstructed",
)
CAUSAL_CATEGORIES = (
    "route_composition_coverage_failure",
    "verified_envelope_failure",
    "mixed_route_and_envelope_failure",
    "mechanism_unresolved",
)
NAMED_LAG_QUANTITIES = (
    "tcp_to_fr3_link6_body_origin_lateral_m",
    "tcp_to_contacting_robot_geom_body_origin_lateral_m",
    "tcp_to_collision_facing_extent_lateral_m",
    "signed_robot_geom_to_pendant_geom_distance_m",
)
DESIGN_LAG_NEG_M = 0.208
DESIGN_LAG_POS_M = 0.108
PROVENANCE_RANGE_NEG_M = (0.198, 0.208)
PROVENANCE_RANGE_POS_M = (0.107, 0.108)
FACE_WINDOW_M = (0.044, 0.156)

RUN_SPECS = (
    {
        "run_id": "offset_wide",
        "candidate": "wide",
        "kind": "offset",
        "paired": "diagnostics_output/pact_place_v98_paired_offset_wide/paired.json",
        "rows_root": (
            "diagnostics_output/pact_place_v98_paired_offset_wide/"
            "expert_screen_rows"
        ),
    },
    {
        "run_id": "offset_cons",
        "candidate": "cons",
        "kind": "offset",
        "paired": "diagnostics_output/pact_place_v98_paired_offset_cons/paired.json",
        "rows_root": (
            "diagnostics_output/pact_place_v98_paired_offset_cons/"
            "expert_screen_rows"
        ),
    },
    {
        "run_id": "centred_halfy016",
        "candidate": "centred_halfy016",
        "kind": "centred_source",
        "paired": "diagnostics_output/pact_place_v98_paired_halfy016/paired.json",
        "rows_root": (
            "diagnostics_output/pact_place_v98_paired_halfy016/"
            "expert_screen_rows"
        ),
    },
    {
        "run_id": "centred_halfy014",
        "candidate": "centred_halfy014",
        "kind": "centred_source",
        "paired": "diagnostics_output/pact_place_v98_paired_halfy014/paired.json",
        "rows_root": (
            "diagnostics_output/pact_place_v98_paired_halfy014/"
            "expert_screen_rows"
        ),
    },
    {
        "run_id": "centred_halfy012",
        "candidate": "centred_halfy012",
        "kind": "centred_source",
        "paired": "diagnostics_output/pact_place_v98_paired_halfy012/paired.json",
        "rows_root": (
            "diagnostics_output/pact_place_v98_paired_halfy012/"
            "expert_screen_rows"
        ),
    },
)

IMMUTABLE_JSON_RELATIVE = (
    "diagnostics_output/pact_place_v95_raw_smoke/summary.json",
    "diagnostics_output/pact_place_v95_smoke_repro_guard_v5/guard.json",
    "diagnostics_output/pact_place_v98_bow_clearance_predict.json",
    "diagnostics_output/pact_place_v98_paired_offset_wide/paired.json",
    "diagnostics_output/pact_place_v98_paired_offset_wide/wrist_lag.json",
    "diagnostics_output/pact_place_v98_paired_offset_cons/paired.json",
    "diagnostics_output/pact_place_v98_paired_halfy016/paired.json",
    "diagnostics_output/pact_place_v98_paired_halfy014/paired.json",
    "diagnostics_output/pact_place_v98_paired_halfy012/paired.json",
)


def trajectory_phase_sequence(
    steps: Sequence[Mapping[str, Any]],
) -> list[tuple[int, str]]:
    sequence: list[tuple[int, str]] = []
    for item in steps:
        if "step" not in item:
            continue
        sequence.append((int(item["step"]), str(item.get("policy_phase") or "")))
    return sequence


def trajectory_phase_at(
    steps: Sequence[Mapping[str, Any]], step: int | None
) -> str | None:
    if step is None:
        return None
    want = int(step)
    for item in steps:
        if int(item.get("step", -1)) == want:
            phase = item.get("policy_phase")
            return None if phase is None else str(phase)
    return None


def classify_onset_category(
    phase: str | None,
    *,
    step: int | None = None,
    trajectory_phases: Sequence[tuple[int, str]] | None = None,
) -> str:
    """Classify contact onset from the first-contact phase, never terminal."""
    if phase is None or str(phase) == "" or step is None:
        return "unreconstructed"
    phase_name = str(phase)
    if phase_name in CEILING_FIXTURE_PHASES:
        return "protected_ceiling_bow_contact"
    if any(phase_name.startswith(prefix) for prefix in EARLY_PHASE_PREFIXES):
        return "early_approach_coverage"
    ceiling_steps = [
        item_step
        for item_step, item_phase in (trajectory_phases or ())
        if item_phase in CEILING_FIXTURE_PHASES
    ]
    ceiling_finished = bool(ceiling_steps) and int(step) > max(ceiling_steps)
    if phase_name in POST_BOW_PHASES:
        if ceiling_finished:
            return "post_bow_pregrasp_coverage"
        return "early_approach_coverage"
    if ceiling_finished:
        return "post_bow_pregrasp_coverage"
    return "unreconstructed"


def reconstruction_is_valid(max_tcp_residual_m: float | None) -> bool:
    return (
        max_tcp_residual_m is not None
        and float(max_tcp_residual_m) <= MAX_TCP_RESIDUAL_M
    )


def lag_rows_for_aggregate(
    rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Drop any row whose reconstruction residual exceeds 1 mm."""
    kept: list[dict[str, Any]] = []
    for row in rows:
        if not row.get("reconstruction_valid"):
            continue
        residual = row.get("max_tcp_residual_m")
        if not reconstruction_is_valid(
            None if residual is None else float(residual)
        ):
            continue
        kept.append(dict(row))
    return kept


def tcp_x_relation(tcp_x_m: float, center_x_m: float, half_x_m: float) -> str:
    low = float(center_x_m) - float(half_x_m)
    high = float(center_x_m) + float(half_x_m)
    value = float(tcp_x_m)
    if value < low:
        return "before"
    if value > high:
        return "after"
    return "inside"


def pendant_aabb_faces(
    center_m: Sequence[float], half_m: Sequence[float]
) -> dict[str, float]:
    cx, cy, cz = (float(center_m[0]), float(center_m[1]), float(center_m[2]))
    hx, hy, hz = (float(half_m[0]), float(half_m[1]), float(half_m[2]))
    return {
        "x_min_m": cx - hx,
        "x_max_m": cx + hx,
        "y_min_m": cy - hy,
        "y_max_m": cy + hy,
        "z_min_m": cz - hz,
        "z_max_m": cz + hz,
        "center_m": [cx, cy, cz],
        "half_m": [hx, hy, hz],
    }


def split_robot_fixture_sides(pair: Mapping[str, Any]) -> dict[str, Any]:
    """Assign robot vs pendant sides from names; never guess a link."""
    left = " ".join(
        str(pair.get(key) or "") for key in ("geom1", "body1", "root1")
    )
    mount_on_left = MOUNT_FIXTURE_BODY_PREFIX in left
    robot_prefix = 2 if mount_on_left else 1
    mount_prefix = 1 if mount_on_left else 2
    return {
        "robot_geom": pair.get(f"geom{robot_prefix}"),
        "robot_body": pair.get(f"body{robot_prefix}"),
        "robot_root": pair.get(f"root{robot_prefix}"),
        "robot_geom_id": pair.get(f"geom{robot_prefix}_id"),
        "pendant_geom": pair.get(f"geom{mount_prefix}"),
        "pendant_body": pair.get(f"body{mount_prefix}"),
        "pendant_root": pair.get(f"root{mount_prefix}"),
        "pendant_geom_id": pair.get(f"geom{mount_prefix}_id"),
    }


def fixture_from_result(result: Mapping[str, Any]) -> dict[str, Any] | None:
    scene = result.get("scene_params") or {}
    fixture = scene.get("pact_v98_pendant_fixture")
    if isinstance(fixture, dict) and fixture:
        return dict(fixture)
    return None


def lookup_manifest(
    source: Mapping[str, Any], episode_id: str
) -> dict[str, Any]:
    matches = [
        dict(row)
        for row in source.get("manifest_rows") or []
        if row.get("episode_id") == episode_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one manifest row for episode_id={episode_id!r}, "
            f"found {len(matches)}"
        )
    return matches[0]


def patch_manifest_for_row(
    manifest: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    sampler_class: str,
    contract_version: str,
) -> dict[str, Any]:
    """Row-specific patch. Never reuse another row's episode or layout."""
    episode_id = str(result.get("episode_id") or manifest.get("episode_id") or "")
    if str(manifest.get("episode_id")) != episode_id:
        raise ValueError(
            "manifest episode_id does not match the result; refusing first-row "
            f"template reuse ({manifest.get('episode_id')!r} vs {episode_id!r})"
        )
    fixture = fixture_from_result(result)
    if fixture is None:
        raise ValueError(
            f"result {episode_id} has no scene_params.pact_v98_pendant_fixture"
        )
    scene = result.get("scene_params") or {}
    lateral_bow = bool(scene.get("pact_v98_pendant_lateral_bow", True))
    patched = dict(manifest)
    patched["sampler_class"] = sampler_class
    patched["pact_mounted_ceiling_fixture"] = dict(fixture)
    patched["pact_v98_contract_version"] = str(
        scene.get("pact_v98_contract_version")
        or fixture.get("siting_contract")
        or contract_version
    )
    patched["pact_v98_pendant_lateral_bow"] = lateral_bow
    selected = result.get("selected_seed") or {}
    if selected.get("seed_u32") is not None:
        patched["task_seed_u32"] = int(selected["seed_u32"])
    if selected.get("seed_u64") is not None:
        patched["task_seed_u64"] = int(selected["seed_u64"])
    patched["source_episode_id"] = episode_id
    return patched


def lag_toward_centreline_m(tcp_y_m: float, point_y_m: float) -> float:
    """Positive when the named point sits closer to y=0 than the TCP."""
    tcp_y = float(tcp_y_m)
    point_y = float(point_y_m)
    if tcp_y < 0.0:
        return point_y - tcp_y
    return tcp_y - point_y


def range_contains(value: float, bounds: tuple[float, float], tol_m: float) -> bool:
    return bounds[0] - tol_m <= float(value) <= bounds[1] + tol_m


def definition_reproduces_provenance(
    peaks: Mapping[str, Sequence[float]],
    *,
    tolerance_m: float = LAG_REPRODUCE_TOLERANCE_M,
) -> bool:
    neg = [float(value) for value in peaks.get("neg") or []]
    pos = [float(value) for value in peaks.get("pos") or []]
    if not neg or not pos:
        return False
    return all(
        range_contains(value, PROVENANCE_RANGE_NEG_M, tolerance_m) for value in neg
    ) and all(
        range_contains(value, PROVENANCE_RANGE_POS_M, tolerance_m) for value in pos
    )


def select_causal_category(
    *,
    baseline_clean_onset_categories: Sequence[str],
    lag_reproduced: bool,
    reconstruction_ok_for_baseline_clean: bool,
    protected_clearance_violated: bool = False,
) -> str:
    if not baseline_clean_onset_categories:
        return "mechanism_unresolved"
    if any(item == "unreconstructed" for item in baseline_clean_onset_categories):
        return "mechanism_unresolved"
    outside = {
        "early_approach_coverage",
        "post_bow_pregrasp_coverage",
    }
    protected = "protected_ceiling_bow_contact"
    has_outside = any(item in outside for item in baseline_clean_onset_categories)
    has_protected = any(item == protected for item in baseline_clean_onset_categories)
    envelope = bool(
        lag_reproduced
        and reconstruction_ok_for_baseline_clean
        and has_protected
        and protected_clearance_violated
    )
    if envelope and has_outside:
        return "mixed_route_and_envelope_failure"
    if envelope and not has_outside:
        return "verified_envelope_failure"
    if has_outside and not has_protected:
        return "route_composition_coverage_failure"
    if has_outside and has_protected and not envelope:
        return "mixed_route_and_envelope_failure"
    return "mechanism_unresolved"


def empty_authorization() -> dict[str, bool]:
    return {
        "authorizes_new_episodes": False,
        "authorizes_gate": False,
        "authorizes_collection": False,
        "episodes_ran": False,
        "physics_stepped": False,
    }


def count_categories(
    rows: Sequence[Mapping[str, Any]], *, baseline_clean_only: bool
) -> dict[str, int]:
    counts = {key: 0 for key in ONSET_CATEGORIES}
    for row in rows:
        if baseline_clean_only and not row.get("baseline_clean_success"):
            continue
        category = str(row.get("onset_category") or "unreconstructed")
        if category not in counts:
            counts[category] = 0
        counts[category] += 1
    return counts
