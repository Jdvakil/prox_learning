"""Behavioral tests for the V10.2 raised, collision-legible pendant.

Runs before any V10.2 episode. Every expectation is recomputed from source
constants and live MuJoCo state; nothing asserts a stored result boolean.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "scripts", ROOT / "submodules" / "molmospaces"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from pact_place_v10_compound_pendant_contract import (  # noqa: E402
    ALL_GEOMS,
    CEILING_TOP_Z_M,
    CROSSBAR_GEOM,
    CROSSBAR_HEIGHT_M,
    EMPIRICAL_LIVE_CONTACT_V1,
    ENDPOINT_ONLY_PRIMITIVE,
    ENVIRONMENT_VERSION as V10_ENVIRONMENT_VERSION,
    GROUP_FREEZE_PRIMITIVE,
    HOOD_TOP_BOTTOM_Z_M,
    LOBE_GEOMS,
    STEM_GEOMS,
    STEM_HALF_M as V10_STEM_HALF_M,
    STEM_TOP_Z_M,
    component_aabb,
)
from pact_place_v10_geometry import (  # noqa: E402
    active_components,
    forbidden_static_overlap,
    planning_probe_assembly,
)
from pact_place_v10_route import resolve_v10_runtime_route  # noqa: E402
from pact_place_v10_scene import pose_assembly_geoms  # noqa: E402
from pact_place_v102_geometry import (  # noqa: E402
    ENVIRONMENT_VERSION_V102,
    PROBE_LABEL_V102,
    RAISED_LOWEST_PENDANT_Z_M,
    RAISED_NEGATIVE_LOBE,
    RAISED_POSITIVE_LOBE,
    RAISED_SHELF_GAP_M,
    RAISED_STEM_Y_M,
    SHELF_TOP_Z_M,
    STEM_HALF_V102_M,
    STEM_SQUARE_V102_M,
    planning_probe_v102_raised_assembly,
    raised_assembly_expectations,
)
from pact_place_v102_raised_pendant_contract import (  # noqa: E402
    CONTRACT_VERSION,
    ENVIRONMENT_VERSION,
    MIN_PENDANT_CLEARANCE_M,
    N_GATE_ROWS,
    N_REVIEW_ROWS,
    N_SCREEN_ROWS,
    PHYSICS_CLEAN_FAMILIES,
    SAMPLER_CLASS,
    build_contract,
    distribution_counts,
    frozen_assembly,
    frozen_route_for_side,
    is_v102_clean_success,
    paired_side_clutter_identical,
    registered_assembly_expectations,
    row_defects,
    sha256_payload,
)
from pact_place_v102_route import (  # noqa: E402
    EMPIRICAL_LIVE_CONTACT_V2,
    EMPTY_ARM_APPROACH_SPEED_M_S,
    PENDANT_PASS_SPEED_M_S,
    PREGRASP_APPROACH_SPEED_M_S,
    classify_route_piece,
    resolve_v102_runtime_route,
    route_is_v102,
    route_piece_speed,
    sequential_ik_component_clearance,
    speed_cap_violation,
    speed_schedule,
    speed_schedule_sha256,
)
from run_pact_place_v102_review_video import (  # noqa: E402
    REVIEW_FPS,
    REVIEW_FRAME_STRIDE,
    video_duration_s,
)

# Historical V10 contiguous-group-freeze lane parameters (no V10.2 markers).
V10_HISTORICAL_ROUTE = {
    "inbound_lane_y_m": -0.12,
    "inbound_padding_m": 0.10,
    "outbound_lane_y_m": -0.12,
    "outbound_padding_m": 0.10,
}

V10_SCENE = (
    ROOT
    / "submodules/molmospaces/molmo_spaces/data_generation/custom_scenes"
    / "pact_place_corridor_v10.xml"
)


def _compiled_v10_model():
    import mujoco

    return mujoco.MjModel.from_xml_path(str(V10_SCENE))


def _make_policy(*, environment_version: str, route: dict, assembly: dict):
    from molmo_spaces.tasks.enclosure_reach import PactPlaceCorridorPolicy

    policy = PactPlaceCorridorPolicy.__new__(PactPlaceCorridorPolicy)
    robot = type("Robot", (), {"robot_view": object(), "kinematics": object()})()
    env = type(
        "Env",
        (),
        {"current_robot": robot, "current_model": object(), "current_data": object()},
    )()
    policy.task = type(
        "Task",
        (),
        {
            "env": env,
            "scene_params": {
                "pact_place_environment_version": environment_version,
                "pact_v10_pendant_assembly": assembly,
                "pact_v10_route": route,
                "intrusion_side": "left",
                "ap_w": 0.85,
            },
        },
    )()
    policy._pact_place_v10_route = {}
    return policy


def _lane_segments(policy):
    from molmo_spaces.policy.solvers.object_manipulation.base_object_manipulation_planner_policy import (  # noqa: E501
        TCPMoveSegment,
    )

    start = policy._place_pose(np.array([0.95, 0.02, 0.88]), np.eye(3))
    mid = policy._place_pose(np.array([0.70, 0.02, 0.88]), np.eye(3))
    end = policy._place_pose(np.array([0.40, 0.02, 0.88]), np.eye(3))
    return [
        TCPMoveSegment(name="a", start_pose=start, end_pose=mid, speed=0.20),
        TCPMoveSegment(name="b", start_pose=mid, end_pose=end, speed=0.20),
    ], start, end


class RaisedGeometryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.assembly = planning_probe_v102_raised_assembly()

    def test_expectations_are_recomputed_from_source(self) -> None:
        negative_bottom = float(
            RAISED_NEGATIVE_LOBE["center_m"][2] - RAISED_NEGATIVE_LOBE["half_m"][2]
        )
        positive_bottom = float(
            RAISED_POSITIVE_LOBE["center_m"][2] - RAISED_POSITIVE_LOBE["half_m"][2]
        )
        derived_lowest = min(negative_bottom, positive_bottom)
        observed = raised_assembly_expectations(self.assembly)
        self.assertAlmostEqual(observed["lowest_pendant_z_m"], derived_lowest, places=9)
        self.assertAlmostEqual(observed["lowest_pendant_z_m"], RAISED_LOWEST_PENDANT_Z_M, places=9)
        self.assertAlmostEqual(
            observed["shelf_to_pendant_gap_m"], derived_lowest - SHELF_TOP_Z_M, places=9
        )
        self.assertAlmostEqual(observed["shelf_to_pendant_gap_m"], RAISED_SHELF_GAP_M, places=9)
        derived_stem_y = [
            float(RAISED_NEGATIVE_LOBE["center_m"][1] - RAISED_NEGATIVE_LOBE["half_m"][1]),
            float(RAISED_POSITIVE_LOBE["center_m"][1] + RAISED_POSITIVE_LOBE["half_m"][1]),
        ]
        self.assertEqual(observed["stem_center_y_m"], derived_stem_y)
        self.assertEqual(list(RAISED_STEM_Y_M), derived_stem_y)
        self.assertEqual(observed["stem_top_z_m"], [STEM_TOP_Z_M, STEM_TOP_Z_M])
        self.assertAlmostEqual(observed["crossbar_top_z_m"], HOOD_TOP_BOTTOM_Z_M, places=9)
        self.assertAlmostEqual(
            observed["crossbar_top_z_m"], CEILING_TOP_Z_M, places=9
        )
        self.assertFalse(observed["physical_swing_dynamics"])
        self.assertTrue(observed["kinematic_fixed_assembly"])
        # The registered table must equal the derived table, not a stored copy.
        self.assertEqual(registered_assembly_expectations(), observed)

    def test_lobe_xy_and_asymmetry_unchanged_from_v101(self) -> None:
        v101 = planning_probe_assembly()
        by_side_v101 = {
            item["side"]: item
            for item in active_components(v101)
            if item["role"] == "lobe"
        }
        by_side_v102 = {
            item["side"]: item
            for item in active_components(self.assembly)
            if item["role"] == "lobe"
        }
        for side in ("negative", "positive"):
            self.assertEqual(
                list(by_side_v102[side]["center_m"][:2]),
                list(by_side_v101[side]["center_m"][:2]),
            )
            self.assertEqual(
                list(by_side_v102[side]["half_m"][:2]),
                list(by_side_v101[side]["half_m"][:2]),
            )
        self.assertNotEqual(
            list(by_side_v102["negative"]["half_m"]),
            list(by_side_v102["positive"]["half_m"]),
        )

    def test_stem_and_crossbar_are_twelve_millimetre_square(self) -> None:
        self.assertAlmostEqual(STEM_SQUARE_V102_M, 0.012, places=9)
        self.assertAlmostEqual(STEM_HALF_V102_M, 0.006, places=9)
        self.assertNotAlmostEqual(STEM_HALF_V102_M, V10_STEM_HALF_M, places=9)
        for item in active_components(self.assembly):
            if item["role"] == "stem":
                self.assertAlmostEqual(item["half_m"][0], STEM_HALF_V102_M, places=9)
                self.assertAlmostEqual(item["half_m"][1], STEM_HALF_V102_M, places=9)
            if item["role"] == "crossbar":
                self.assertAlmostEqual(item["half_m"][0], STEM_HALF_V102_M, places=9)
                self.assertAlmostEqual(
                    2.0 * item["half_m"][2], CROSSBAR_HEIGHT_M, places=9
                )

    def test_stems_span_lobe_top_to_stem_top_and_meet_the_crossbar(self) -> None:
        components = {item["name"]: item for item in active_components(self.assembly)}
        crossbar_lo, crossbar_hi = component_aabb(
            components["crossbar"]["center_m"], components["crossbar"]["half_m"]
        )
        for slot, lobe_source in ((0, RAISED_NEGATIVE_LOBE), (1, RAISED_POSITIVE_LOBE)):
            lobe = components[f"lobe_{slot}"]
            stem = components[f"stem_{slot}"]
            _lobe_lo, lobe_hi = component_aabb(lobe["center_m"], lobe["half_m"])
            stem_lo, stem_hi = component_aabb(stem["center_m"], stem["half_m"])
            self.assertAlmostEqual(float(stem_lo[2]), float(lobe_hi[2]), places=9)
            self.assertAlmostEqual(float(stem_hi[2]), STEM_TOP_Z_M, places=9)
            self.assertAlmostEqual(float(stem_hi[2]), float(crossbar_lo[2]), places=9)
        self.assertAlmostEqual(float(crossbar_hi[2]), HOOD_TOP_BOTTOM_Z_M, places=9)

    def test_no_active_component_overlaps_forbidden_static_geometry(self) -> None:
        for item in active_components(self.assembly):
            self.assertFalse(
                forbidden_static_overlap(
                    item, allow_hood_top=item["role"] == "crossbar"
                ),
                item["name"],
            )

    def test_v101_probe_geometry_is_untouched(self) -> None:
        v101 = planning_probe_assembly()
        self.assertEqual(v101.get("probe_label"), "probe_v2")
        for item in active_components(v101):
            if item["role"] == "lobe":
                self.assertAlmostEqual(item["center_m"][2], 0.86, places=9)
            if item["role"] in {"stem", "crossbar"}:
                self.assertAlmostEqual(item["half_m"][0], V10_STEM_HALF_M, places=9)
        self.assertEqual(self.assembly.get("probe_label"), PROBE_LABEL_V102)
        self.assertNotEqual(self.assembly["assembly_id"], v101["assembly_id"])


class SceneGeomTest(unittest.TestCase):
    """Collision and visible geometry are the same geoms, and parking kills them."""

    def setUp(self) -> None:
        self.model = _compiled_v10_model()
        self.assembly = planning_probe_v102_raised_assembly()

    def test_one_geom_per_component_no_visual_only_sleeve(self) -> None:
        names = [
            self.model.geom(index).name or ""
            for index in range(int(self.model.ngeom))
        ]
        for name in ALL_GEOMS:
            self.assertEqual(names.count(name), 1, name)
        pendant_like = [
            name for name in names if name.startswith("pact_clutter_mount_v10")
        ]
        self.assertEqual(sorted(pendant_like), sorted(ALL_GEOMS))

    def test_active_collision_and_visible_sizes_are_identical(self) -> None:
        pose_assembly_geoms(self.model, self.assembly, parked=False)
        by_geom = {
            item["geom"]: item
            for item in self.assembly["components"]
            if item.get("active")
        }
        for name, item in by_geom.items():
            geom_id = int(self.model.geom(name).id)
            size = np.asarray(self.model.geom_size[geom_id], dtype=float)
            self.assertTrue(
                np.allclose(size, np.asarray(item["half_m"], dtype=float), atol=1e-12),
                name,
            )
            self.assertEqual(int(self.model.geom_contype[geom_id]), 8, name)
            self.assertEqual(int(self.model.geom_conaffinity[geom_id]), 15, name)
            self.assertAlmostEqual(float(self.model.geom_rgba[geom_id][3]), 1.0, places=9)
            if item["role"] in {"stem", "crossbar"}:
                self.assertAlmostEqual(float(size[0]), STEM_HALF_V102_M, places=12)
            if item["role"] == "stem":
                self.assertAlmostEqual(float(size[1]), STEM_HALF_V102_M, places=12)

    def test_parked_controls_disable_every_component_collision_geom(self) -> None:
        pose_assembly_geoms(self.model, self.assembly, parked=False)
        pose_assembly_geoms(self.model, self.assembly, parked=True)
        for name in LOBE_GEOMS + STEM_GEOMS + (CROSSBAR_GEOM,):
            geom_id = int(self.model.geom(name).id)
            self.assertEqual(int(self.model.geom_contype[geom_id]), 0, name)
            self.assertEqual(int(self.model.geom_conaffinity[geom_id]), 0, name)
            self.assertAlmostEqual(float(self.model.geom_rgba[geom_id][3]), 0.0, places=9)

    def test_inactive_slot_geoms_are_disabled_when_active(self) -> None:
        pose_assembly_geoms(self.model, self.assembly, parked=False)
        for name in (LOBE_GEOMS[2], STEM_GEOMS[2]):
            geom_id = int(self.model.geom(name).id)
            self.assertEqual(int(self.model.geom_contype[geom_id]), 0, name)
            self.assertEqual(int(self.model.geom_conaffinity[geom_id]), 0, name)


class StemContactParityTest(unittest.TestCase):
    """Deliberate stem overlap must be seen by data.contact and the classifier."""

    def _model_with_overlap(self, overlap: bool):
        import mujoco

        # Concentric box/box is degenerate in MuJoCo; cross the probe geom.
        offset = 0.03 if overlap else 0.5
        xml = f"""
        <mujoco>
          <worldbody>
            <body name="robot_0/link5" pos="0 0 0">
              <!-- A dof is required: MuJoCo skips contacts between two welded
                   static bodies, and the real arm links are articulated. -->
              <joint name="robot_0/link5_j" type="slide" axis="1 0 0"/>
              <geom name="robot_0/link5_g" type="box" size="0.04 0.04 0.04"
                    contype="1" conaffinity="15"/>
            </body>
            <body name="pact_clutter_mount_v10" pos="{offset} 0 0">
              <geom name="pact_clutter_mount_v10_stem_0_g" type="box"
                    size="{STEM_HALF_V102_M} {STEM_HALF_V102_M} 0.1625"
                    contype="8" conaffinity="15"/>
            </body>
          </worldbody>
        </mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        return model, data

    def _pairs(self, model, data):
        records = []
        for index in range(int(data.ncon)):
            contact = data.contact[index]
            if float(contact.dist) > 0.0:
                continue
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            body1 = int(model.geom_bodyid[geom1])
            body2 = int(model.geom_bodyid[geom2])
            records.append(
                {
                    "geom1": model.geom(geom1).name,
                    "geom2": model.geom(geom2).name,
                    "body1": model.body(body1).name,
                    "body2": model.body(body2).name,
                    "root1": model.body(int(model.body_rootid[body1])).name,
                    "root2": model.body(int(model.body_rootid[body2])).name,
                    "distance_m": float(contact.dist),
                }
            )
        return records

    def test_overlap_seen_by_data_contact_and_classifier(self) -> None:
        from molmo_spaces.tasks.pact_place_contact_audit import classify_contact

        model, data = self._model_with_overlap(overlap=True)
        pairs = self._pairs(model, data)
        self.assertTrue(pairs, "deliberate stem overlap produced no data.contact entry")
        self.assertIn("mounted_fixture", {classify_contact(pair) for pair in pairs})

    def test_separated_stem_produces_no_contact(self) -> None:
        model, data = self._model_with_overlap(overlap=False)
        self.assertEqual(self._pairs(model, data), [])

    def test_stem_and_robot_geoms_are_collision_compatible(self) -> None:
        model, _data = self._model_with_overlap(overlap=True)
        stem = int(model.geom("pact_clutter_mount_v10_stem_0_g").id)
        robot = int(model.geom("robot_0/link5_g").id)
        self.assertTrue(
            (int(model.geom_contype[stem]) & int(model.geom_conaffinity[robot]))
            or (int(model.geom_contype[robot]) & int(model.geom_conaffinity[stem]))
        )


class SpeedScheduleTest(unittest.TestCase):
    def test_piece_classification_and_speeds(self) -> None:
        self.assertEqual(
            classify_route_piece("inbound_pendant_approach"), "empty_arm_approach"
        )
        for name in (
            "inbound_pendant_pass",
            "inbound_pendant_exit",
            "outbound_pendant_pass",
            "outbound_pendant_exit",
        ):
            self.assertEqual(classify_route_piece(name), "pendant_pass")
        self.assertEqual(
            classify_route_piece("inbound_pendant_rejoin"), "pregrasp_approach"
        )
        for name in ("outbound_pendant_approach", "outbound_pendant_rejoin"):
            self.assertEqual(classify_route_piece(name), "historical_transport")
        self.assertAlmostEqual(
            route_piece_speed("inbound_pendant_approach", inherited_speed_m_s=0.20),
            EMPTY_ARM_APPROACH_SPEED_M_S,
        )
        self.assertAlmostEqual(
            route_piece_speed("inbound_pendant_pass", inherited_speed_m_s=0.20),
            PENDANT_PASS_SPEED_M_S,
        )
        self.assertAlmostEqual(
            route_piece_speed("outbound_pendant_pass", inherited_speed_m_s=0.20),
            PENDANT_PASS_SPEED_M_S,
        )
        self.assertAlmostEqual(
            route_piece_speed("inbound_pendant_rejoin", inherited_speed_m_s=0.20),
            PREGRASP_APPROACH_SPEED_M_S,
        )
        self.assertAlmostEqual(
            route_piece_speed("outbound_pendant_approach", inherited_speed_m_s=0.20),
            0.20,
        )
        with self.assertRaises(ValueError):
            classify_route_piece("inbound_pendant_teleport")

    def test_speed_caps(self) -> None:
        self.assertIsNone(
            speed_cap_violation("inbound_pendant_pass", PENDANT_PASS_SPEED_M_S)
        )
        self.assertEqual(
            speed_cap_violation("inbound_pendant_pass", 0.20),
            "pendant_pass_speed_above_cap",
        )
        self.assertEqual(
            speed_cap_violation("inbound_pendant_approach", 0.20),
            "initial_approach_speed_above_cap",
        )
        self.assertIsNone(speed_cap_violation("outbound_pendant_approach", 0.20))


class DispatchGuardTest(unittest.TestCase):
    def test_v102_cannot_activate_without_exact_marker_and_hash(self) -> None:
        v101_route = {
            "rewrite_primitive": ENDPOINT_ONLY_PRIMITIVE,
            "qualification_mode": EMPIRICAL_LIVE_CONTACT_V1,
        }
        self.assertIsNone(
            resolve_v102_runtime_route(
                {"pact_place_environment_version": V10_ENVIRONMENT_VERSION}, v101_route
            )
        )
        self.assertIsNone(resolve_v102_runtime_route({}, {}))
        good = frozen_route_for_side("left")
        scene = {"pact_place_environment_version": ENVIRONMENT_VERSION_V102}
        self.assertTrue(route_is_v102(scene, good))
        dispatch = resolve_v102_runtime_route(scene, good)
        self.assertEqual(dispatch["qualification_mode"], EMPIRICAL_LIVE_CONTACT_V2)
        self.assertTrue(dispatch["use_endpoint_only"])
        self.assertTrue(dispatch["skip_offline_strict_environment"])
        # Tampered hash.
        bad = dict(good)
        bad["speed_schedule_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            resolve_v102_runtime_route(scene, bad)
        # Tampered schedule.
        bad = dict(good)
        schedule = dict(speed_schedule())
        schedule["pendant_pass_m_s"] = 0.20
        bad["speed_schedule"] = schedule
        with self.assertRaises(ValueError):
            resolve_v102_runtime_route(scene, bad)
        # V10.2 route markers under the V10 scene marker.
        with self.assertRaises(ValueError):
            resolve_v102_runtime_route(
                {"pact_place_environment_version": V10_ENVIRONMENT_VERSION}, good
            )
        # V10.2 scene marker with a V10.1 route.
        with self.assertRaises(ValueError):
            resolve_v102_runtime_route(scene, v101_route)

    def test_v10_and_v101_dispatch_unchanged(self) -> None:
        historical = resolve_v10_runtime_route({})
        self.assertEqual(historical["rewrite_primitive"], GROUP_FREEZE_PRIMITIVE)
        self.assertFalse(historical["use_endpoint_only"])
        self.assertFalse(historical["skip_offline_strict_environment"])
        empirical = resolve_v10_runtime_route(
            {
                "rewrite_primitive": ENDPOINT_ONLY_PRIMITIVE,
                "qualification_mode": EMPIRICAL_LIVE_CONTACT_V1,
            }
        )
        self.assertTrue(empirical["use_endpoint_only"])
        self.assertTrue(empirical["skip_offline_strict_environment"])
        with self.assertRaises(ValueError):
            resolve_v10_runtime_route(
                {"qualification_mode": EMPIRICAL_LIVE_CONTACT_V2}
            )

    def test_environment_predicates(self) -> None:
        from molmo_spaces.tasks.enclosure_reach import (
            PACT_PLACE_V102_ENVIRONMENT_VERSION,
            PACT_PLACE_V10_ENVIRONMENT_VERSION,
            PactPlaceCorridorV102RaisedPendantSampler,
            PactPlaceCorridorV10CompoundPendantSampler,
        )

        self.assertEqual(
            PactPlaceCorridorV10CompoundPendantSampler.PACT_PLACE_ENVIRONMENT_VERSION,
            PACT_PLACE_V10_ENVIRONMENT_VERSION,
        )
        self.assertEqual(
            PactPlaceCorridorV102RaisedPendantSampler.PACT_PLACE_ENVIRONMENT_VERSION,
            PACT_PLACE_V102_ENVIRONMENT_VERSION,
        )
        self.assertEqual(PACT_PLACE_V102_ENVIRONMENT_VERSION, ENVIRONMENT_VERSION)
        v10 = _make_policy(
            environment_version=V10_ENVIRONMENT_VERSION,
            route=dict(V10_HISTORICAL_ROUTE),
            assembly=planning_probe_assembly(),
        )
        v102 = _make_policy(
            environment_version=ENVIRONMENT_VERSION_V102,
            route=frozen_route_for_side("left"),
            assembly=frozen_assembly(),
        )
        self.assertTrue(v10._v10_enabled())
        self.assertFalse(v10._v102_enabled())
        self.assertTrue(v102._v10_enabled())
        self.assertTrue(v102._v102_enabled())
        self.assertTrue(v10._v9_enabled())
        self.assertTrue(v102._v9_enabled())


class LaneRewriteTest(unittest.TestCase):
    def test_v102_assigns_speed_per_named_piece(self) -> None:
        policy = _make_policy(
            environment_version=ENVIRONMENT_VERSION_V102,
            route=frozen_route_for_side("left"),
            assembly=frozen_assembly(),
        )
        called = {"n": 0}

        def boom(*_args, **_kwargs):
            called["n"] += 1
            raise AssertionError("scalar environment preclearance must not run")

        policy._v10_evaluate_nominal_and_robust = boom  # type: ignore[method-assign]
        policy._v102_route_sequential_ik = lambda planned, include_target: {
            "waypoints_attempted": int(len(planned["planned_positions_m"])),
            "waypoints_solved": int(len(planned["planned_positions_m"])),
            "complete_sequential_ik": True,
            "ik_failure_indices": [],
            "per_component_min_clearance_m": {"lobe_0": 0.05},
            "min_clearance_m": 0.05,
            "qpos_restored": True,
        }
        segments, start, end = _lane_segments(policy)
        rebuilt = policy._v10_apply_lane(
            segments, prefix="inbound_pendant", include_target=False
        )
        self.assertEqual(called["n"], 0)
        self.assertGreater(len(rebuilt), 0)
        self.assertTrue(np.allclose(rebuilt[0].start_pose[:3, 3], start[:3, 3], atol=1e-9))
        self.assertTrue(np.allclose(rebuilt[-1].end_pose[:3, 3], end[:3, 3], atol=1e-9))
        by_name: dict[str, set] = {}
        for segment in rebuilt:
            by_name.setdefault(segment.name, set()).add(round(float(segment.speed), 9))
        self.assertIn("inbound_pendant_pass", by_name)
        for name, speeds in by_name.items():
            self.assertEqual(len(speeds), 1, name)
            expected = route_piece_speed(name, inherited_speed_m_s=0.20)
            self.assertAlmostEqual(next(iter(speeds)), expected, places=9)
            self.assertIsNone(speed_cap_violation(name, next(iter(speeds))))
        record = policy._pact_place_v10_route["inbound_pendant"]
        self.assertEqual(record["qualification_mode"], EMPIRICAL_LIVE_CONTACT_V2)
        self.assertEqual(record["speed_schedule_sha256"], speed_schedule_sha256())
        self.assertTrue(record["complete_sequential_ik"])
        self.assertEqual(
            record["waypoints_attempted"], record["waypoints_solved"]
        )
        self.assertFalse(record["fallback_taken"])
        self.assertTrue(record["frozen_endpoint_preserved"])
        self.assertGreaterEqual(record["min_abs_detour_m"], 0.05)
        names = {item["name"] for item in record["piece_speeds"]}
        self.assertEqual(names, set(by_name))
        # The fast empty-arm approach is not copied onto the pendant pass.
        approach = [
            item
            for item in record["piece_speeds"]
            if item["name"] == "inbound_pendant_approach"
        ]
        if approach:
            self.assertAlmostEqual(
                approach[0]["requested_speed_m_s"], EMPTY_ARM_APPROACH_SPEED_M_S
            )
        pass_pieces = [
            item
            for item in record["piece_speeds"]
            if item["name"].endswith("_pass") or item["name"].endswith("_exit")
        ]
        self.assertTrue(pass_pieces)
        for item in pass_pieces:
            self.assertAlmostEqual(
                item["requested_speed_m_s"], PENDANT_PASS_SPEED_M_S, places=9
            )
            self.assertAlmostEqual(item["inherited_speed_m_s"], 0.20, places=9)

    def test_v10_rows_keep_the_inherited_single_speed(self) -> None:
        policy = _make_policy(
            environment_version=V10_ENVIRONMENT_VERSION,
            route=dict(V10_HISTORICAL_ROUTE),
            assembly=planning_probe_assembly(),
        )
        calls = {"n": 0}

        def fake_evaluate(planned, **kwargs):
            calls["n"] += 1
            self.assertFalse(kwargs.get("endpoint_only"))
            return {
                "nominal": {"pendant_clearance_m": 0.03, "ik_ok": True, "ik_failures": 0},
                "environment": {"environment_clear": True},
                "n_corners_evaluated": 8,
                "robust_ik_ok": True,
                "accepted": True,
                "min_robust_clearance_m": 0.021,
            }

        policy._v10_evaluate_nominal_and_robust = fake_evaluate  # type: ignore[method-assign]
        segments, _start, _end = _lane_segments(policy)
        rebuilt = policy._v10_apply_lane(
            segments, prefix="inbound_pendant", include_target=False
        )
        self.assertEqual(calls["n"], 1)
        self.assertGreater(len(rebuilt), 0)
        self.assertEqual({round(float(item.speed), 9) for item in rebuilt}, {0.20})
        record = policy._pact_place_v10_route["inbound_pendant"]
        self.assertEqual(record["rewrite_primitive"], GROUP_FREEZE_PRIMITIVE)
        self.assertTrue(record["offline_strict_environment_preclearance_used"])
        self.assertNotIn("speed_schedule_sha256", record)
        self.assertTrue(record["ik_ok"])
        for item in record["piece_speeds"]:
            self.assertEqual(item["speed_class"], "inherited")

    def test_v102_row_with_a_tampered_schedule_is_refused(self) -> None:
        route = dict(frozen_route_for_side("left"))
        route["speed_schedule_sha256"] = "0" * 64
        policy = _make_policy(
            environment_version=ENVIRONMENT_VERSION_V102,
            route=route,
            assembly=frozen_assembly(),
        )
        segments, _start, _end = _lane_segments(policy)
        with self.assertRaises(ValueError):
            policy._v10_apply_lane(
                segments, prefix="inbound_pendant", include_target=False
            )


class SequentialIkAccountingTest(unittest.TestCase):
    def _callables(self, *, fail_at=None, raise_at=None):
        state = {"qpos": {"arm": np.zeros(3)}, "restored": 0}

        def set_qpos(value):
            state["qpos"] = {key: np.asarray(item).copy() for key, item in value.items()}

        def get_qpos():
            return {key: np.asarray(item).copy() for key, item in state["qpos"].items()}

        calls = {"n": 0}

        def solve_ik(_pose, _seed):
            index = calls["n"]
            calls["n"] += 1
            if raise_at is not None and index == raise_at:
                raise RuntimeError("ik exploded")
            if fail_at is not None and index in set(fail_at):
                return None
            return {"arm": np.full(3, float(index))}

        def forward():
            state["restored"] += 1

        return state, set_qpos, get_qpos, solve_ik, forward

    def _run(self, **kwargs):
        state, set_qpos, get_qpos, solve_ik, forward = self._callables(**kwargs)
        saved = {"arm": np.array([9.0, 9.0, 9.0])}
        positions = np.zeros((5, 3))
        rotations = np.stack([np.eye(3)] * 5)
        report = sequential_ik_component_clearance(
            positions,
            rotations,
            saved_qpos=saved,
            set_qpos=set_qpos,
            get_qpos=get_qpos,
            solve_ik=solve_ik,
            forward=forward,
            place_pose=lambda p, r: np.eye(4),
            component_names=["lobe_0", "stem_0"],
            measure_components=lambda: {"lobe_0": 0.04, "stem_0": 0.03},
        )
        return report, state, saved

    def test_success_path(self) -> None:
        report, state, saved = self._run()
        self.assertEqual(report["waypoints_attempted"], 5)
        self.assertEqual(report["waypoints_solved"], 5)
        self.assertTrue(report["complete_sequential_ik"])
        self.assertAlmostEqual(report["min_clearance_m"], 0.03)
        self.assertEqual(
            report["per_component_min_clearance_m"], {"lobe_0": 0.04, "stem_0": 0.03}
        )
        self.assertTrue(report["qpos_restored"])
        self.assertTrue(np.allclose(state["qpos"]["arm"], saved["arm"]))

    def test_failure_path_never_reports_a_pass(self) -> None:
        report, state, saved = self._run(fail_at=[1])
        self.assertEqual(report["waypoints_attempted"], 5)
        self.assertEqual(report["waypoints_solved"], 4)
        self.assertFalse(report["complete_sequential_ik"])
        self.assertEqual(report["ik_failure_indices"], [1])
        self.assertTrue(np.allclose(state["qpos"]["arm"], saved["arm"]))

    def test_single_waypoint_abort_is_not_an_ik_pass(self) -> None:
        report, _state, _saved = self._run(fail_at=[1, 2, 3, 4])
        self.assertEqual(report["waypoints_solved"], 1)
        self.assertFalse(report["complete_sequential_ik"])

    def test_exception_path_restores_qpos(self) -> None:
        state, set_qpos, get_qpos, solve_ik, forward = self._callables(raise_at=2)
        saved = {"arm": np.array([9.0, 9.0, 9.0])}
        with self.assertRaises(RuntimeError):
            sequential_ik_component_clearance(
                np.zeros((5, 3)),
                np.stack([np.eye(3)] * 5),
                saved_qpos=saved,
                set_qpos=set_qpos,
                get_qpos=get_qpos,
                solve_ik=solve_ik,
                forward=forward,
                place_pose=lambda p, r: np.eye(4),
                component_names=["lobe_0"],
                measure_components=lambda: {"lobe_0": 0.04},
            )
        self.assertTrue(np.allclose(state["qpos"]["arm"], saved["arm"]))


class ContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = build_contract()

    def test_deterministic_regeneration_and_self_hash(self) -> None:
        again = build_contract()
        self.assertEqual(self.contract["contract_sha256"], again["contract_sha256"])
        payload = dict(self.contract)
        digest = payload.pop("contract_sha256")
        self.assertEqual(digest, sha256_payload(payload))
        self.assertEqual(self.contract["contract_version"], CONTRACT_VERSION)
        self.assertEqual(self.contract["environment_version"], ENVIRONMENT_VERSION)
        self.assertEqual(self.contract["sampler_class"], SAMPLER_CLASS)

    def test_row_distributions_and_seed_disjointness(self) -> None:
        screen = self.contract["screen_rows"]
        review = self.contract["review_rows"]
        gate = self.contract["gate_rows"]
        self.assertEqual((len(screen), len(review), len(gate)), (N_SCREEN_ROWS, N_REVIEW_ROWS, N_GATE_ROWS))
        for rows, per_cell in ((screen, 1), (review, 2), (gate, 4)):
            counts = distribution_counts(rows)
            self.assertEqual(len(counts), 6)
            self.assertTrue(all(value == per_cell for value in counts.values()))
            self.assertTrue(paired_side_clutter_identical(rows))
            self.assertEqual(
                {row["layout_family_id"] for row in rows}, set(PHYSICS_CLEAN_FAMILIES)
            )
        seeds = [
            {row["task_seed_u32"] for row in rows} for rows in (screen, review, gate)
        ]
        self.assertFalse(seeds[0] & seeds[1])
        self.assertFalse(seeds[0] & seeds[2])
        self.assertFalse(seeds[1] & seeds[2])

    def test_f3_is_not_admitted(self) -> None:
        families = {
            row["layout_family_id"]
            for rows in ("screen_rows", "review_rows", "gate_rows")
            for row in self.contract[rows]
        }
        self.assertNotIn("F3_aperture_side_stagger", families)

    def test_every_row_carries_the_registered_route_and_assembly(self) -> None:
        assembly = frozen_assembly()
        for key in ("screen_rows", "review_rows", "gate_rows"):
            for row in self.contract[key]:
                self.assertEqual(row["sampler_class"], SAMPLER_CLASS)
                self.assertEqual(
                    row["pact_v10_pendant_assembly"]["assembly_id"],
                    assembly["assembly_id"],
                )
                self.assertEqual(
                    row["pact_v10_route"], frozen_route_for_side(row["intrusion_side"])
                )
                self.assertEqual(
                    row["pact_v102_assembly_sha256"], sha256_payload(assembly)
                )
                self.assertEqual(
                    row["pact_v102_assembly_sha256"],
                    self.contract["assembly_self_sha256"],
                )
                self.assertTrue(row["pact_v10_pendant_assembly"]["components"])
                self.assertTrue(
                    route_is_v102(
                        {"pact_place_environment_version": ENVIRONMENT_VERSION},
                        row["pact_v10_route"],
                    )
                )

    def test_authorizations_are_false(self) -> None:
        for key in (
            "authorizes_gate",
            "authorizes_collection",
            "authorizes_training",
            "authorizes_evaluation",
            "phase0_passed",
            "eligible_for_separate_collection_authorization",
        ):
            self.assertFalse(self.contract[key], key)


class RowAdmissionTest(unittest.TestCase):
    def _clean_result(self, **overrides):
        route = {
            "rewrite_primitive": ENDPOINT_ONLY_PRIMITIVE,
            "qualification_mode": EMPIRICAL_LIVE_CONTACT_V2,
            "lane_y_m": -0.30,
            "padding_m": 0.08,
            "min_abs_detour_m": 0.11,
            "detour_meets_minimum": True,
            "fallback_taken": False,
            "clipped": False,
            "wrong_way": False,
            "frozen_endpoint_preserved": True,
            "offline_strict_environment_preclearance_used": False,
            "strict_environment_preclearance_intentionally_not_used": True,
            "speed_schedule_sha256": speed_schedule_sha256(),
            "waypoints_attempted": 120,
            "waypoints_solved": 120,
            "complete_sequential_ik": True,
            "piece_speeds": [
                {"name": "inbound_pendant_approach", "requested_speed_m_s": 0.15},
                {"name": "inbound_pendant_pass", "requested_speed_m_s": 0.045},
            ],
        }
        payload = {
            "status": "complete",
            "episode_id": "abc",
            "row_sha256": "row",
            "task_success": True,
            "clean_success": True,
            "pendant_v10": {"inbound": dict(route), "outbound": dict(route)},
            "contact_audit": {
                "contact_class_totals": {
                    "hazard_bar": 0,
                    "other_environment": 0,
                    "clutter": 0,
                    "mounted_fixture": 0,
                    "place_receptacle": 8,
                }
            },
            "clutter_stability_events": [],
            "pendant_frame_telemetry": {
                "n_frames": 151,
                "n_frames_measured": 151,
                "min_clearance_m": 0.031,
                "per_component_min_clearance_m": {
                    "lobe_0": 0.031,
                    "stem_0": 0.052,
                    "crossbar": 0.30,
                },
                "live_pendant_contact_frames": 0,
                "segment_speeds": [
                    {"name": "inbound_pendant_pass", "commanded_speed_m_s": 0.045}
                ],
            },
        }
        payload.update(overrides)
        return payload

    def test_clean_row_passes(self) -> None:
        self.assertEqual(row_defects(self._clean_result()), [])
        self.assertTrue(is_v102_clean_success(self._clean_result()))

    def test_pendant_contact_rejects(self) -> None:
        result = self._clean_result()
        result["contact_audit"]["contact_class_totals"]["mounted_fixture"] = 3
        self.assertIn("mounted_fixture_contact", row_defects(result))
        self.assertFalse(is_v102_clean_success(result))

    def test_live_frame_contact_rejects(self) -> None:
        result = self._clean_result()
        result["pendant_frame_telemetry"]["live_pendant_contact_frames"] = 1
        self.assertIn("live_pendant_contact", row_defects(result))

    def test_clearance_floor_rejects(self) -> None:
        result = self._clean_result()
        result["pendant_frame_telemetry"]["min_clearance_m"] = (
            MIN_PENDANT_CLEARANCE_M - 0.001
        )
        result["pendant_frame_telemetry"]["per_component_min_clearance_m"]["lobe_0"] = (
            MIN_PENDANT_CLEARANCE_M - 0.001
        )
        defects = row_defects(result)
        self.assertIn("frame_clearance_below_floor", defects)
        self.assertIn("component_clearance_below_floor:lobe_0", defects)

    def test_speed_cap_and_ik_rejections(self) -> None:
        result = self._clean_result()
        result["pendant_v10"]["inbound"]["piece_speeds"] = [
            {"name": "inbound_pendant_pass", "requested_speed_m_s": 0.20}
        ]
        self.assertIn("inbound_pendant_pass_speed_above_cap", row_defects(result))
        result = self._clean_result()
        result["pendant_v10"]["outbound"]["complete_sequential_ik"] = False
        self.assertIn("outbound_incomplete_sequential_ik", row_defects(result))

    def test_missing_telemetry_rejects(self) -> None:
        result = self._clean_result()
        result.pop("pendant_frame_telemetry")
        self.assertIn("missing_frame_telemetry", row_defects(result))
        result = self._clean_result()
        result["pendant_v10"]["inbound"].pop("piece_speeds")
        self.assertIn("missing_telemetry", row_defects(result))

    def test_v101_qualification_mode_is_not_accepted(self) -> None:
        result = self._clean_result()
        result["pendant_v10"]["inbound"]["qualification_mode"] = (
            EMPIRICAL_LIVE_CONTACT_V1
        )
        self.assertIn("missing_telemetry", row_defects(result))

    def test_sampling_and_infrastructure_failures_reject(self) -> None:
        self.assertEqual(
            row_defects({"status": "sampling_failure"}), ["sampling_failure"]
        )
        self.assertEqual(
            row_defects({"status": "infrastructure_failure"}),
            ["infrastructure_failure"],
        )


class ReviewRendererTimingTest(unittest.TestCase):
    def test_frame_rate_matches_the_policy_timestep(self) -> None:
        self.assertEqual(REVIEW_FRAME_STRIDE, 1)
        self.assertAlmostEqual(REVIEW_FPS, 1000.0 / 66.0, places=9)

    def test_151_policy_frames_are_about_ten_seconds(self) -> None:
        duration = video_duration_s(151)
        self.assertAlmostEqual(duration, 151 * 0.066, places=6)
        self.assertGreater(duration, 9.9)
        self.assertLess(duration, 10.1)

    def test_v101_playback_was_1_32x_and_v102_is_1x(self) -> None:
        # V10.1 rendered every second 66 ms frame at 10 fps.
        v101_playback = (2 * 0.066) / (1.0 / 10.0)
        self.assertAlmostEqual(v101_playback, 1.32, places=9)
        v102_playback = (REVIEW_FRAME_STRIDE * 0.066) / (1.0 / REVIEW_FPS)
        self.assertAlmostEqual(v102_playback, 1.0, places=9)


if __name__ == "__main__":
    unittest.main()
