#!/usr/bin/env python3
"""V10.4 review-v2: certify all three diagnostic negative controls.

A control is a separately compiled diagnostic scene in which the complete
pendant assembly is rigidly translated inward along y until the target
component reaches 5-30 mm of signed penetration against the retained
production trajectory. Nothing here touches the production scene, its
geometry, its routing, its speeds, its seeds, or its results: the production
XML's bytes are re-verified before and after every control.

Certification is stricter than the search. A shift is only accepted once the
diagnostic scene has been reloaded through the real task sampler and shown to
be compiled-static, correctly bounded, and in agreement across four
independent contact instruments at the certified frame.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "scripts", ROOT / "submodules" / "molmospaces"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from pact_place_v104_geometry import (  # noqa: E402
    ALL_GEOMS_V104,
    PENDANT_BODY_V104,
    SCENE_XML_RELATIVE_V104,
    production_assembly,
    scene_xml_text,
)
from pact_place_v104_review_v2_contract import (  # noqa: E402
    BASE_SCENE_V3_RELATIVE,
    BASE_SCENE_V5_RELATIVE,
    CONTROL_ANCHOR_TOLERANCE_M,
    CONTROL_ANCHORS,
    CONTROL_ORDER_V2,
    CONTROL_PENETRATION_BAND_M,
    CONTROL_SHIFT_GRID_V2_M,
    CONTROL_SPEC,
    CONTROL_WINDOW_LEAD_FRAMES,
    CONTROL_WINDOW_TRAIL_FRAMES,
    PRODUCTION_SCENE_SHA256,
    SCENE_METADATA_RELATIVE_V104,
    sha256_bytes_of,
)

DIAGNOSTIC_SCENE_STEM = "pact_place_v104_diagnostic_control"
ROBOT_PREFIX = "robot_0/"


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def inward_sign(component_name: str, assembly: dict[str, Any] | None = None) -> float:
    """+1 moves the assembly toward y=0 from the negative side, -1 from positive."""
    assembly = assembly or production_assembly()
    component = next(
        item for item in assembly["components"] if item["name"] == component_name
    )
    return 1.0 if float(component["center_m"][1]) < 0 else -1.0


def shifted_assembly(
    assembly: dict[str, Any], sign: float, shift_m: float
) -> dict[str, Any]:
    """Rigidly translate the COMPLETE assembly along y. No component moves alone."""
    moved = dict(assembly)
    moved["components"] = [
        {
            **item,
            "center_m": [
                float(item["center_m"][0]),
                round(float(item["center_m"][1]) + float(sign) * float(shift_m), 9),
                float(item["center_m"][2]),
            ],
        }
        for item in assembly["components"]
    ]
    return moved


def build_scene_bundle(assembly: dict[str, Any], destination: Path) -> Path:
    """A complete, self-sufficient diagnostic scene directory.

    MuJoCo resolves ``<include>`` relative to the including file, and the task
    sampler looks for ``<stem>_metadata.json`` beside the scene, so both the V3
    and V5 shells and a metadata copy renamed to the diagnostic stem must be
    present or the compile fails outside the production tree.
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for relative in (BASE_SCENE_V3_RELATIVE, BASE_SCENE_V5_RELATIVE):
        source = ROOT / relative
        shutil.copyfile(source, destination / source.name)
    metadata_source = ROOT / SCENE_METADATA_RELATIVE_V104
    shutil.copyfile(
        metadata_source, destination / f"{DIAGNOSTIC_SCENE_STEM}_metadata.json"
    )
    scene_path = destination / f"{DIAGNOSTIC_SCENE_STEM}.xml"
    scene_path.write_text(scene_xml_text(assembly))
    return scene_path


def production_scene_unchanged() -> dict[str, Any]:
    observed = sha256_bytes_of(ROOT / SCENE_XML_RELATIVE_V104)
    return {
        "path": SCENE_XML_RELATIVE_V104,
        "expected_sha256": PRODUCTION_SCENE_SHA256,
        "observed_sha256": observed,
        "byte_identical": observed == PRODUCTION_SCENE_SHA256,
    }


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------
def search_control_shift(
    row: dict[str, Any],
    seed_u32: int,
    component_name: str,
    steps: Sequence[dict[str, Any]],
    *,
    grid_m: Sequence[float] = CONTROL_SHIFT_GRID_V2_M,
    band_m: tuple[float, float] = CONTROL_PENETRATION_BAND_M,
) -> dict[str, Any]:
    """First frozen-grid shift whose target component enters the band.

    Every admissible retained frame is evaluated at every tested shift. The
    candidate list is never truncated: a rigid inward shift of ``s`` changes
    any frame's clearance by at most ``s``, so frames whose unshifted clearance
    exceeds ``base_min + s`` provably cannot be the worst frame at that shift
    and only those are skipped. That bound is sound; a fixed cap is not, and
    a fixed cap is what hid the true worst frame in v1.
    """
    import mujoco
    from molmo_spaces.data_generation.pipeline import cleanup_episode_resources
    from pact_geom_distance import true_distance
    from pact_place_v104_clearance import (
        assembly_boxes,
        frame_clearances,
        geom_shape_cache,
        robot_collision_geom_ids,
        target_collision_geom_ids,
    )
    from run_pact_place_expert_screen import _make_config
    from run_pact_place_v7_replay_videos import apply_recorded_qpos

    from pact_place_v104_review_v2_contract import CONTRACT_VERSION_V2  # noqa: F401

    assembly = production_assembly()
    sign = inward_sign(component_name, assembly)
    component = next(
        item for item in assembly["components"] if item["name"] == component_name
    )

    scratch = Path(tempfile.mkdtemp(prefix="v104_ctrl_search_"))
    task = sampler = None
    tested: list[dict[str, Any]] = []
    try:
        config = _make_config(
            scratch / "d.json",
            scene_xml=ROOT / SCENE_XML_RELATIVE_V104,
            sampler_class="PactPlaceCorridorV104Sampler",
        )
        sampler = config.task_sampler_config.task_sampler_class(config)
        sampler.seed_task_sampling(int(seed_u32))
        sampler.set_pact_manifest_row(row)
        task = sampler.sample_task(house_index=int(row["scene_template_house_index"]))
        env = task.env
        model, data = env.current_model, env.current_data
        probe = robot_collision_geom_ids(model) + target_collision_geom_ids(task)
        cache = geom_shape_cache(model, probe)
        pendant_ids = {name: int(model.geom(name).id) for name in ALL_GEOMS_V104}
        saved_pos = {
            name: np.asarray(model.geom_pos[gid], dtype=float).copy()
            for name, gid in pendant_ids.items()
        }
        target_gid = pendant_ids[component["geom"]]

        base_boxes = [
            box for box in assembly_boxes(assembly) if box["name"] == component_name
        ]
        per_frame: list[float] = []
        for step in steps:
            apply_recorded_qpos(env, step["qpos"])
            mujoco.mj_forward(model, data)
            value = frame_clearances(model, data, base_boxes, probe, cache)["min_m"]
            per_frame.append(float(value) if value is not None else float("inf"))
        base_min = float(min(per_frame))
        order = sorted(range(len(steps)), key=lambda index: per_frame[index])

        offset = np.array([0.0, 0.0, 0.0], dtype=float)
        try:
            for shift in grid_m:
                bound = base_min + float(shift) + 1e-6
                candidates = [index for index in order if per_frame[index] <= bound]
                worst = 0.0
                worst_step = None
                if candidates:
                    offset[1] = sign * float(shift)
                    for name, gid in pendant_ids.items():
                        model.geom_pos[gid] = saved_pos[name] + offset
                    for index in candidates:
                        apply_recorded_qpos(env, steps[index]["qpos"])
                        mujoco.mj_forward(model, data)
                        distance = float(true_distance(model, data, probe, [target_gid]))
                        if distance < worst:
                            worst, worst_step = distance, index
                    for name, gid in pendant_ids.items():
                        model.geom_pos[gid] = saved_pos[name]
                penetration = float(-worst)
                tested.append(
                    {
                        "shift_m": float(shift),
                        "max_penetration_m": penetration,
                        "n_frames_measured": len(candidates),
                        "n_frames_total": len(steps),
                        "candidate_list_capped": False,
                    }
                )
                if band_m[0] <= penetration <= band_m[1]:
                    return {
                        "found": True,
                        "component": component_name,
                        "shift_m": float(shift),
                        "inward_sign": float(sign),
                        "penetration_m": penetration,
                        "max_frame": int(worst_step),
                        "unshifted_min_clearance_m": base_min,
                        "n_shifts_tested": len(tested),
                        "n_frames_measured_at_selected_shift": len(candidates),
                        "tested": tested,
                        "assembly": shifted_assembly(assembly, sign, shift),
                        "whole_assembly_translated": True,
                    }
        finally:
            for name, gid in pendant_ids.items():
                model.geom_pos[gid] = saved_pos[name]
            mujoco.mj_forward(model, data)
        return {
            "found": False,
            "component": component_name,
            "inward_sign": float(sign),
            "unshifted_min_clearance_m": base_min,
            "n_shifts_tested": len(tested),
            "max_penetration_reached_m": max(
                (item["max_penetration_m"] for item in tested), default=0.0
            ),
            "tested": tested,
            "whole_assembly_translated": True,
        }
    finally:
        cleanup_episode_resources(
            task=task,
            policy=None,
            task_sampler=sampler,
            preloaded_policy=None,
            close_task_sampler=sampler is not None,
        )
        shutil.rmtree(scratch, ignore_errors=True)


# ---------------------------------------------------------------------------
# Certification on a separately compiled diagnostic scene
# ---------------------------------------------------------------------------
def certify_control(
    control: str,
    row: dict[str, Any],
    seed_u32: int,
    steps: Sequence[dict[str, Any]],
    found: dict[str, Any],
) -> dict[str, Any]:
    """Reload the diagnostic scene through the real sampler and prove it out."""
    import mujoco
    from molmo_spaces.data_generation.pipeline import cleanup_episode_resources
    from molmo_spaces.tasks.pact_place_contact_audit import classify_contact
    from pact_geom_distance import gjk_distance, true_distance
    from pact_place_v104_clearance import (
        _shape,
        assembly_boxes,
        geom_shape_cache,
        robot_collision_geom_ids,
        target_collision_geom_ids,
    )
    from run_pact_place_expert_screen import _make_config
    from run_pact_place_v7_replay_videos import apply_recorded_qpos

    assembly = found["assembly"]
    component_name = found["component"]
    component = next(
        item for item in assembly["components"] if item["name"] == component_name
    )
    anchor = CONTROL_ANCHORS[control]

    scene_before = production_scene_unchanged()
    scratch = Path(tempfile.mkdtemp(prefix="v104_ctrl_cert_"))
    task = sampler = None
    try:
        scene_path = build_scene_bundle(assembly, scratch / "scene")
        config = _make_config(
            scratch / "d.json",
            scene_xml=scene_path,
            sampler_class="PactPlaceCorridorV104Sampler",
        )
        sampler = config.task_sampler_config.task_sampler_class(config)
        sampler.seed_task_sampling(int(seed_u32))
        sampler.set_pact_manifest_row(row)
        task = sampler.sample_task(house_index=int(row["scene_template_house_index"]))
        env = task.env
        model, data = env.current_model, env.current_data
        probe = robot_collision_geom_ids(model) + target_collision_geom_ids(task)
        cache = geom_shape_cache(model, probe)
        pendant_ids = {name: int(model.geom(name).id) for name in ALL_GEOMS_V104}
        target_gid = pendant_ids[component["geom"]]

        # --- compiled-static, on the reloaded diagnostic model -------------
        body_id = int(model.body(PENDANT_BODY_V104).id)
        by_geom = {item["geom"]: item for item in assembly["components"]}
        bounds: list[dict[str, Any]] = []
        for name, gid in pendant_ids.items():
            expected_half = np.asarray(by_geom[name]["half_m"], dtype=float)
            expected_pos = np.asarray(by_geom[name]["center_m"], dtype=float)
            size = np.asarray(model.geom_size[gid], dtype=float)
            aabb = np.asarray(model.geom_aabb[gid], dtype=float)
            bounds.append(
                {
                    "geom": name,
                    "compiled_pos_m": [float(v) for v in model.geom_pos[gid]],
                    "pos_matches_shifted_assembly": bool(
                        np.allclose(model.geom_pos[gid], expected_pos, atol=1e-9)
                    ),
                    "size_matches": bool(np.allclose(size, expected_half, atol=1e-12)),
                    "geom_aabb_half": [float(v) for v in aabb[3:]],
                    "geom_aabb_encloses": bool(
                        np.all(aabb[3:] >= expected_half - 1e-12)
                    ),
                    "geom_rbound_m": float(model.geom_rbound[gid]),
                    "rbound_encloses": bool(
                        float(model.geom_rbound[gid])
                        >= float(np.linalg.norm(expected_half)) - 1e-9
                    ),
                }
            )
        static = {
            "body": PENDANT_BODY_V104,
            "body_dofnum": int(model.body_dofnum[body_id]),
            "body_jntnum": int(model.body_jntnum[body_id]),
            "body_mocapid": int(model.body_mocapid[body_id]),
            "no_joint_freejoint_or_mocap": bool(
                int(model.body_dofnum[body_id]) == 0
                and int(model.body_jntnum[body_id]) == 0
                and int(model.body_mocapid[body_id]) < 0
            ),
            "runtime_bound_repair_applied": False,
            "compiled_static": True,
        }
        bounds_ok = all(
            item["size_matches"]
            and item["geom_aabb_encloses"]
            and item["rbound_encloses"]
            and item["pos_matches_shifted_assembly"]
            for item in bounds
        )

        # --- full-path audit, every pendant component ----------------------
        secondary: list[dict[str, Any]] = []
        first_contact_frame: int | None = None
        target_penetration: list[float] = []
        component_by_geom_name = {
            item["geom"]: item["name"] for item in assembly["components"]
        }
        for index, step in enumerate(steps):
            apply_recorded_qpos(env, step["qpos"])
            mujoco.mj_forward(model, data)
            distance = float(true_distance(model, data, probe, [target_gid]))
            target_penetration.append(-distance)
            hit: dict[str, float] = {}
            for contact_index in range(int(data.ncon)):
                contact = data.contact[contact_index]
                if float(contact.dist) > 0.0:
                    continue
                geom1, geom2 = int(contact.geom1), int(contact.geom2)
                for gid in (geom1, geom2):
                    name = model.geom(gid).name or ""
                    if name in component_by_geom_name:
                        key = component_by_geom_name[name]
                        hit[key] = min(
                            hit.get(key, 0.0), float(contact.dist)
                        )
            if component_name in hit and first_contact_frame is None:
                first_contact_frame = index
            for key, dist in sorted(hit.items()):
                if key == component_name:
                    continue
                secondary.append(
                    {
                        "frame": index,
                        "component": key,
                        "penetration_m": float(-dist),
                    }
                )
        max_frame = int(np.argmax(np.asarray(target_penetration, dtype=float)))
        max_penetration = float(target_penetration[max_frame])

        # --- four-instrument agreement at the certified frame --------------
        apply_recorded_qpos(env, steps[max_frame]["qpos"])
        mujoco.mj_forward(model, data)
        signed = float(true_distance(model, data, probe, [target_gid]))
        # gjk_distance is the independent analytic instrument: it takes two
        # convex shapes and returns 0.0 on intersection. Build the target's
        # shape from the compiled model so it is not read back from the
        # assembly dict that produced the scene.
        target_cache = geom_shape_cache(model, [target_gid])
        target_shape = _shape(model, data, target_gid, target_cache)
        unsigned = float("inf")
        for probe_gid in probe:
            probe_shape = _shape(model, data, int(probe_gid), cache)
            if not probe_shape.supported:
                continue
            unsigned = min(unsigned, float(gjk_distance(probe_shape, target_shape)))
            if unsigned == 0.0:
                break
        live_pairs: list[dict[str, Any]] = []
        classes: set[str] = set()
        limiting_body = None
        deepest = 0.0
        for contact_index in range(int(data.ncon)):
            contact = data.contact[contact_index]
            if float(contact.dist) > 0.0:
                continue
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            if target_gid not in (geom1, geom2):
                continue
            name1 = model.geom(geom1).name or f"geom_{geom1}"
            name2 = model.geom(geom2).name or f"geom_{geom2}"
            body1 = model.body(int(model.geom_bodyid[geom1])).name or ""
            body2 = model.body(int(model.geom_bodyid[geom2])).name or ""
            pair = {
                "geom1": name1,
                "geom2": name2,
                "body1": body1,
                "body2": body2,
                "distance_m": float(contact.dist),
            }
            classes.add(classify_contact(pair))
            live_pairs.append(pair)
            if float(contact.dist) < deepest:
                deepest = float(contact.dist)
                robot_body = body1 if body1.startswith(ROBOT_PREFIX) else body2
                limiting_body = robot_body.removeprefix(ROBOT_PREFIX)
        parity = {
            "frame": max_frame,
            "signed_distance_m": signed,
            "signed_reports_penetration": signed < 0.0,
            "gjk_unsigned_distance_m": unsigned,
            "gjk_reports_intersection": unsigned == 0.0,
            "live_contact_pairs": live_pairs[:8],
            "n_live_contact_pairs": len(live_pairs),
            "live_reports_contact": bool(live_pairs),
            "contact_classes": sorted(classes),
            "classified_mounted_fixture": classes == {"mounted_fixture"},
            "limiting_robot_body": limiting_body,
        }
        parity["all_four_agree"] = bool(
            parity["signed_reports_penetration"]
            and parity["gjk_reports_intersection"]
            and parity["live_reports_contact"]
            and parity["classified_mounted_fixture"]
        )

        # --- audited anchors ------------------------------------------------
        anchor_check = {
            "expected_shift_m": anchor["shift_m"],
            "observed_shift_m": float(found["shift_m"]),
            "shift_matches": abs(float(found["shift_m"]) - anchor["shift_m"]) <= 1e-9,
            "expected_penetration_m": anchor["penetration_m"],
            "observed_penetration_m": max_penetration,
            "penetration_within_tolerance": bool(
                abs(max_penetration - anchor["penetration_m"])
                <= CONTROL_ANCHOR_TOLERANCE_M
            ),
            "expected_max_frame": anchor["max_frame"],
            "observed_max_frame": max_frame,
            "max_frame_matches": max_frame == int(anchor["max_frame"]),
            "expected_limiting_robot_body": anchor["limiting_robot_body"],
            "observed_limiting_robot_body": limiting_body,
            "limiting_body_matches": limiting_body == anchor["limiting_robot_body"],
            "tolerance_m": CONTROL_ANCHOR_TOLERANCE_M,
        }
        anchor_check["passed"] = bool(
            anchor_check["shift_matches"]
            and anchor_check["penetration_within_tolerance"]
            and anchor_check["max_frame_matches"]
            and anchor_check["limiting_body_matches"]
        )

        window = control_window(
            first_contact_frame=first_contact_frame,
            max_frame=max_frame,
            secondary=secondary,
            n_frames=len(steps),
        )
        scene_after = production_scene_unchanged()
        in_band = bool(
            CONTROL_PENETRATION_BAND_M[0]
            <= max_penetration
            <= CONTROL_PENETRATION_BAND_M[1]
        )
        certificate = {
            "control": control,
            "component": component_name,
            "component_geom": component["geom"],
            # The shifted assembly travels with the certificate. Without it the
            # renderer falls back to production geometry and silently draws a
            # control that never touches anything.
            "assembly": assembly,
            "source_role_index": int(row["role_index"]),
            "source_intrusion_side": str(row["intrusion_side"]),
            "source_episode_id": str(row["episode_id"]),
            "shift_m": float(found["shift_m"]),
            "inward_sign": float(found["inward_sign"]),
            "whole_assembly_translated": True,
            "grid_m": {
                "first": float(CONTROL_SHIFT_GRID_V2_M[0]),
                "last": float(CONTROL_SHIFT_GRID_V2_M[-1]),
                "n_points": len(CONTROL_SHIFT_GRID_V2_M),
                "increment_m": 0.001,
            },
            "n_shifts_tested": int(found["n_shifts_tested"]),
            "candidate_list_capped": False,
            "n_retained_frames": len(steps),
            "penetration_m": max_penetration,
            "penetration_in_band": in_band,
            "band_m": list(CONTROL_PENETRATION_BAND_M),
            "max_frame": max_frame,
            "first_contact_frame": first_contact_frame,
            "diagnostic_scene_stem": DIAGNOSTIC_SCENE_STEM,
            "diagnostic_scene_sha256": sha256_bytes_of(scene_path),
            "reloaded_through_task_sampler": True,
            "sampler_class": "PactPlaceCorridorV104Sampler",
            "static": static,
            "compiled_bounds": bounds,
            "compiled_bounds_ok": bounds_ok,
            "parity": parity,
            "anchor": anchor_check,
            "window": window,
            "secondary_contacts": secondary,
            "n_secondary_contacts": len(secondary),
            "secondary_components": sorted({item["component"] for item in secondary}),
            "secondary_summary": _secondary_summary(secondary),
            "production_scene_before": scene_before,
            "production_scene_after": scene_after,
            "production_scene_byte_identical": bool(
                scene_before["byte_identical"] and scene_after["byte_identical"]
            ),
        }
        certificate["certified"] = bool(
            in_band
            and static["no_joint_freejoint_or_mocap"]
            and bounds_ok
            and parity["all_four_agree"]
            and anchor_check["passed"]
            and window["valid"]
            and certificate["production_scene_byte_identical"]
        )
        return certificate
    finally:
        cleanup_episode_resources(
            task=task,
            policy=None,
            task_sampler=sampler,
            preloaded_policy=None,
            close_task_sampler=sampler is not None,
        )
        shutil.rmtree(scratch, ignore_errors=True)


def _secondary_summary(secondary: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """First frame a non-target component touches, and its eventual deepest hit.

    These are two different frames and must not be reported as one. The first
    frame is what trims the clip; the deepest penetration happens later, well
    outside the published window.
    """
    if not secondary:
        return {"any": False}
    first = min(secondary, key=lambda item: int(item["frame"]))
    deepest = max(secondary, key=lambda item: float(item["penetration_m"]))
    return {
        "any": True,
        "first_frame": int(first["frame"]),
        "first_component": str(first["component"]),
        "first_penetration_m": float(first["penetration_m"]),
        "max_penetration_m": float(deepest["penetration_m"]),
        "max_penetration_frame": int(deepest["frame"]),
        "max_penetration_component": str(deepest["component"]),
        "last_frame": int(max(int(item["frame"]) for item in secondary)),
    }


def control_window(
    *,
    first_contact_frame: int | None,
    max_frame: int,
    secondary: Sequence[dict[str, Any]],
    n_frames: int,
    lead: int = CONTROL_WINDOW_LEAD_FRAMES,
    trail: int = CONTROL_WINDOW_TRAIL_FRAMES,
) -> dict[str, Any]:
    """Contact-centered trim: lead frames before first contact, trail after peak.

    The window is cut one frame short of the first non-target component's
    contact so a clip labelled for one component never shows another one
    intruding.
    """
    if first_contact_frame is None:
        return {
            "valid": False,
            "reason": "target component never makes live contact",
        }
    first = max(0, int(first_contact_frame) - int(lead))
    last = min(n_frames - 1, int(max_frame) + int(trail))
    secondary_frames = sorted(int(item["frame"]) for item in secondary)
    truncated_by = None
    cutoff = next(
        (frame for frame in secondary_frames if frame > int(first_contact_frame)), None
    )
    if cutoff is not None and cutoff - 1 < last:
        last = cutoff - 1
        truncated_by = cutoff
    valid = last >= first and first <= int(max_frame) <= last
    return {
        "valid": bool(valid),
        "first_frame": int(first),
        "last_frame": int(last),
        "n_frames": int(last - first + 1) if valid else 0,
        "first_contact_frame": int(first_contact_frame),
        "max_penetration_frame": int(max_frame),
        "includes_max_penetration_frame": bool(first <= int(max_frame) <= last),
        "lead_frames": int(lead),
        "trail_frames": int(trail),
        "truncated_by_secondary_contact_at_frame": truncated_by,
        "excluded_secondary_frames": [
            frame for frame in secondary_frames if frame > last
        ],
    }


def certify_all(
    production_root: Path, manifest: dict[str, Any]
) -> dict[str, Any]:
    """Search and certify every control, in registered order."""
    from run_pact_place_expert_screen import _result_path

    rows = {int(row["role_index"]): row for row in manifest["rows"]}
    certificates: list[dict[str, Any]] = []
    shortfalls: list[dict[str, Any]] = []
    for control in CONTROL_ORDER_V2:
        spec = CONTROL_SPEC[control]
        row = rows[int(spec["source_role_index"])]
        result_path = _result_path(Path(production_root), row)
        result = json.loads(result_path.read_text())
        steps = json.loads(
            (result_path.parent / "trajectory.json").read_text()
        )["steps"]
        seed_u32 = int(result["selected_seed"]["seed_u32"])
        found = search_control_shift(row, seed_u32, spec["component"], steps)
        if not found["found"]:
            shortfalls.append(
                {
                    "control": control,
                    "component": spec["component"],
                    "source_role_index": int(row["role_index"]),
                    "reason": "no grid shift reached the registered band",
                    "max_penetration_reached_m": found["max_penetration_reached_m"],
                    "n_shifts_tested": found["n_shifts_tested"],
                }
            )
            continue
        certificate = certify_control(control, row, seed_u32, steps, found)
        certificates.append(certificate)
        if not certificate["certified"]:
            shortfalls.append(
                {
                    "control": control,
                    "component": spec["component"],
                    "source_role_index": int(row["role_index"]),
                    "reason": "certification failed",
                    "penetration_in_band": certificate["penetration_in_band"],
                    "anchor_passed": certificate["anchor"]["passed"],
                    "parity_all_four_agree": certificate["parity"]["all_four_agree"],
                    "compiled_bounds_ok": certificate["compiled_bounds_ok"],
                    "window_valid": certificate["window"]["valid"],
                }
            )
    return {
        "certificates": certificates,
        "shortfalls": shortfalls,
        "n_certified": sum(1 for item in certificates if item["certified"]),
        "all_certified": len(certificates) == len(CONTROL_ORDER_V2)
        and not shortfalls,
    }


__all__ = [
    "DIAGNOSTIC_SCENE_STEM",
    "build_scene_bundle",
    "certify_all",
    "certify_control",
    "control_window",
    "inward_sign",
    "production_scene_unchanged",
    "search_control_shift",
    "shifted_assembly",
]
