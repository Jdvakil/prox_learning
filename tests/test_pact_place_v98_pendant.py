from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "scripts", ROOT / "submodules" / "molmospaces"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pact_place_v98_pendant_contract import (  # noqa: E402
    ADMISSION_FLOOR,
    CEILING_TOP_Z_M,
    DEFAULT_APERTURE_WIDTH_M,
    MIN_DETOUR_SLACK_M,
    OFFSET_CANDIDATES,
    PENDANT_BOTTOM_Z_BOUNDS_M,
    PENDANT_CENTER_Y_BOUNDS_M,
    PENDANT_CENTER_Y_M,
    PENDANT_DEPTH_BOUNDS_M,
    PENDANT_HALF_Y_BOUNDS_M,
    PENDANT_HALF_Y_CONS_M,
    PENDANT_HALF_Y_WIDE_M,
    WRIST_LAG_NEG_M,
    WRIST_LAG_POS_M,
    build_pendant_fixture,
    ceiling_fixture_bow_policy_constants,
    fixture_bow_detour_slack_m,
    fixture_bow_lateral_limit_m,
    fixture_bow_policy_constants,
    fixture_bow_waypoint_abs_y_m,
    pendant_aabb,
    pendant_admissible_faces_m,
    pendant_faces_m,
    validate_pendant_geometry,
)
from run_pact_place_v98_pendant_preview import build_row  # noqa: E402


class PactPlaceV98PendantTest(unittest.TestCase):
    def test_contract_rejects_out_of_range_geometry(self) -> None:
        nominal = build_pendant_fixture()
        cases = [
            ([0.72, 0.0, nominal["center_m"][2]], nominal["half_m"]),
            ([0.72, 0.001, nominal["center_m"][2]], nominal["half_m"]),
            ([0.72, 0.100, 1.30], nominal["half_m"]),
            ([0.72, 0.100, 1.3075], [0.1, 0.11, 0.2075]),
            ([0.72, 0.100, 1.3075], [0.1, 0.19, 0.2075]),
            ([0.40, 0.100, nominal["center_m"][2]], nominal["half_m"]),
        ]
        for center, half in cases:
            with self.subTest(center=center, half=half):
                with self.assertRaises(ValueError):
                    validate_pendant_geometry(center, half)

    def test_builder_honors_all_frozen_bounds(self) -> None:
        for bottom in PENDANT_BOTTOM_Z_BOUNDS_M:
            for name, spec in OFFSET_CANDIDATES.items():
                fixture = build_pendant_fixture(
                    bottom_z_m=bottom,
                    half_y_m=spec["half_y_m"],
                    center_y_m=spec["center_y_m"],
                )
                center, half = fixture["center_m"], fixture["half_m"]
                with self.subTest(name=name, bottom=bottom):
                    self.assertGreaterEqual(center[1], PENDANT_CENTER_Y_BOUNDS_M[0])
                    self.assertLessEqual(center[1], PENDANT_CENTER_Y_BOUNDS_M[1])
                    self.assertAlmostEqual(center[2] + half[2], CEILING_TOP_Z_M)
                    self.assertGreaterEqual(
                        center[0] - half[0], PENDANT_DEPTH_BOUNDS_M[0]
                    )
                    self.assertLessEqual(
                        center[0] + half[0], PENDANT_DEPTH_BOUNDS_M[1]
                    )
                    self.assertEqual(fixture["lateral_lane_cost_m"], 0.0)

    def test_contract_admits_both_offset_candidates(self) -> None:
        self.assertEqual(PENDANT_HALF_Y_BOUNDS_M, (0.040, 0.080))
        self.assertTrue(
            PENDANT_HALF_Y_BOUNDS_M[0]
            <= PENDANT_HALF_Y_CONS_M
            <= PENDANT_HALF_Y_WIDE_M
            <= PENDANT_HALF_Y_BOUNDS_M[1]
        )
        for name, spec in OFFSET_CANDIDATES.items():
            fixture = build_pendant_fixture(
                bottom_z_m=spec["bottom_z_m"],
                half_y_m=spec["half_y_m"],
                center_y_m=spec["center_y_m"],
            )
            with self.subTest(name=name):
                self.assertAlmostEqual(fixture["center_m"][1], spec["center_y_m"])
                self.assertAlmostEqual(fixture["half_m"][1], spec["half_y_m"])

    def test_opposite_panel_sides_carry_byte_identical_fixture_geometry(self) -> None:
        left = build_row(
            cell_index=0,
            candidate=0,
            panel_side="left",
            implementation_sha256="test",
            seed=980024,
        )
        right = build_row(
            cell_index=1,
            candidate=0,
            panel_side="right",
            implementation_sha256="test",
            seed=980024,
        )
        self.assertEqual(
            left["pact_mounted_ceiling_fixture"],
            right["pact_mounted_ceiling_fixture"],
        )
        self.assertEqual(
            left["pact_mounted_ceiling_fixture"]["center_m"][1],
            right["pact_mounted_ceiling_fixture"]["center_m"][1],
        )

    def test_aabb_and_sampler_registration_are_explicit(self) -> None:
        fixture = build_pendant_fixture()
        low, high = pendant_aabb(fixture)
        self.assertAlmostEqual(low[0], 0.62)
        self.assertAlmostEqual(high[0], 0.82)
        self.assertAlmostEqual(low[1], PENDANT_CENTER_Y_M - PENDANT_HALF_Y_WIDE_M)
        self.assertAlmostEqual(high[1], PENDANT_CENTER_Y_M + PENDANT_HALF_Y_WIDE_M)
        self.assertAlmostEqual(low[2], 1.15)
        self.assertAlmostEqual(high[2], CEILING_TOP_Z_M)
        source = inspect.getsource(
            __import__(
                "molmo_spaces.tasks.enclosure_reach",
                fromlist=["PactPlaceCorridorV98PendantSampler"],
            ).PactPlaceCorridorV98PendantSampler
        )
        self.assertIn('"obstacle_aabbs"', source)
        self.assertIn('"pact_v9_hazards"', source)
        self.assertEqual(
            ADMISSION_FLOOR["min_distinct_changed_sensors_per_role_side"], 3
        )
        self.assertEqual(ADMISSION_FLOOR["min_changed_values_per_role_side"], 448)

    def test_mounted_fixture_is_strict_non_target_contact(self) -> None:
        from types import SimpleNamespace
        from unittest.mock import patch

        from molmo_spaces.tasks.pact_place_contact_audit import PactPlaceContactAudit

        pair = {
            "geom1": "robot_0/fr3_link6_collision",
            "geom2": "pact_clutter_mount_ceiling_g",
            "body1": "robot_0/fr3_link6",
            "body2": "pact_clutter_mount_ceiling",
            "root1": "robot_0/",
            "root2": "pact_clutter_mount_ceiling",
            "distance_m": -0.001,
        }
        audit = PactPlaceContactAudit()
        audit.set_phase("inbound", "pregrasp")
        env = SimpleNamespace(current_data=SimpleNamespace(time=0.01))
        with patch(
            "molmo_spaces.tasks.pact_place_contact_audit.place_environment_contact_pairs",
            return_value=[pair],
        ):
            audit.observe(env, step=107)
        summary = audit.summary()
        self.assertEqual(summary["contact_class_totals"]["mounted_fixture"], 1)
        self.assertGreater(summary["non_target_contact_entries"], 0)
        self.assertFalse(summary["collision_free"])

    def test_pinned_seed_is_constant_across_cells_and_candidates(self) -> None:
        from run_pact_place_v98_pendant_preview import DEFAULT_SEED

        self.assertEqual(DEFAULT_SEED, 955339)
        seen = []
        for cell_index in range(6):
            for candidate in range(4):
                row = build_row(
                    cell_index=cell_index,
                    candidate=candidate,
                    panel_side="left" if cell_index % 2 == 0 else "right",
                    implementation_sha256="test",
                )
                self.assertEqual(row["task_seed_u32"], 955339)
                self.assertEqual(row["task_seed_u64"], 955339)
                seen.append(
                    (row["layout_family_id"], row["intrusion_side"], row["task_seed_u64"])
                )
        self.assertEqual(len({item[2] for item in seen}), 1)
        self.assertEqual(len({(item[0], item[1]) for item in seen}), 6)
        row = build_row(
            cell_index=0,
            candidate=0,
            panel_side="left",
            implementation_sha256="test",
        )
        self.assertIs(row["pact_v98_pendant_lateral_bow"], True)

    def test_fixture_bow_waypoint_algebra_and_zero_slack_at_half_y_018(self) -> None:
        import numpy as np
        from types import SimpleNamespace

        from molmo_spaces.policy.solvers.object_manipulation.base_object_manipulation_planner_policy import (  # noqa: E402
            TCPMoveSegment,
        )
        from molmo_spaces.tasks.enclosure_reach import PactPlaceCorridorPolicy

        constants = fixture_bow_policy_constants()
        self.assertEqual(constants["aperture_width_m"], DEFAULT_APERTURE_WIDTH_M)
        self.assertAlmostEqual(constants["envelope_half_y_m"], 0.10)
        self.assertAlmostEqual(constants["safe_gap_m"], 0.025)
        self.assertAlmostEqual(constants["aperture_edge_reserve_m"], 0.02)
        lateral_limit = fixture_bow_lateral_limit_m(constants)
        self.assertAlmostEqual(lateral_limit, 0.305)
        for half_y, slack in ((0.18, 0.0), (0.16, 0.02), (0.14, 0.04), (0.12, 0.06)):
            waypoint = fixture_bow_waypoint_abs_y_m(half_y, constants)
            self.assertAlmostEqual(waypoint, 0.125 + half_y, places=6)
            self.assertAlmostEqual(
                fixture_bow_detour_slack_m(half_y, constants), slack, places=6
            )
            self.assertAlmostEqual(lateral_limit - waypoint, slack, places=6)
        self.assertLess(
            fixture_bow_detour_slack_m(0.18, constants) + 1e-12, MIN_DETOUR_SLACK_M
        )

        policy = PactPlaceCorridorPolicy.__new__(PactPlaceCorridorPolicy)
        policy.task = SimpleNamespace(scene_params={"ap_w": DEFAULT_APERTURE_WIDTH_M})
        policy._pact_place_bow_diagnostics = (
            PactPlaceCorridorPolicy._empty_bow_diagnostics()
        )
        policy.policy_config = SimpleNamespace(speed_fast=0.12, speed_slow=0.045)
        start = np.eye(4)
        start[:3, 3] = [0.40, 0.0, 0.885]
        end = np.eye(4)
        end[:3, 3] = [0.95, 0.0, 0.885]
        segment = TCPMoveSegment(
            name="synthetic_inbound",
            start_pose=start,
            end_pose=end,
            speed=0.12,
        )
        for side in (1.0, -1.0):
            for half_y in (0.18, 0.16, 0.14, 0.12):
                policy._pact_place_bow_diagnostics = (
                    PactPlaceCorridorPolicy._empty_bow_diagnostics()
                )
                pieces, bowed = policy._bow_segment(
                    segment,
                    prefix="inbound_ceiling_fixture",
                    envelope_half_y=PactPlaceCorridorPolicy.MOUNTED_FIXTURE_ENVELOPE_HALF_Y,
                    safe_gap=PactPlaceCorridorPolicy.MOUNTED_FIXTURE_SAFE_GAP,
                    center=np.array([0.72, 0.0, 1.3075]),
                    half=np.array([0.10, half_y, 0.2075]),
                    preferred_waypoint_side=side,
                )
                self.assertTrue(bowed)
                waypoint_y = float(pieces[0].end_pose[1, 3])
                expected = side * (0.125 + half_y)
                if abs(expected) > lateral_limit:
                    expected = side * lateral_limit
                self.assertAlmostEqual(waypoint_y, expected, places=6)
                if abs(half_y - 0.18) < 1e-12:
                    self.assertAlmostEqual(abs(waypoint_y), lateral_limit, places=6)
                diag = policy._pact_place_bow_diagnostics["inbound_ceiling_fixture"]
                if abs(half_y - 0.18) < 1e-12:
                    self.assertAlmostEqual(
                        diag["accepted_bow_m"], diag["planned_bow_m"], places=4
                    )

    def test_v98_lateral_bow_is_off_by_default(self) -> None:
        from molmo_spaces.tasks.enclosure_reach import PactPlaceCorridorPolicy

        self.assertEqual(
            PactPlaceCorridorPolicy._mounted_fixture_roles(
                "pact_place_corridor_v9_8_pendant"
            ),
            (),
        )
        self.assertEqual(
            PactPlaceCorridorPolicy._mounted_fixture_roles(
                "pact_place_corridor_v9_8_pendant",
                pendant_lateral_bow=False,
            ),
            (),
        )
        self.assertEqual(
            PactPlaceCorridorPolicy._mounted_fixture_roles(
                "pact_place_corridor_v9_8_pendant",
                pendant_lateral_bow=True,
            ),
            ("ceiling_fixture",),
        )
        self.assertEqual(
            PactPlaceCorridorPolicy._mounted_fixture_roles(
                "pact_place_corridor_v9_4_mounted_preview"
            ),
            ("wall_fixture", "ceiling_fixture"),
        )
        self.assertEqual(
            PactPlaceCorridorPolicy._mounted_fixture_roles(
                "pact_place_corridor_v9_5_low_wall"
            ),
            ("wall_fixture",),
        )
        self.assertEqual(
            PactPlaceCorridorPolicy._mounted_fixture_roles(
                "pact_place_corridor_v9_3"
            ),
            (),
        )

    def test_ceiling_envelopes_are_lag_plus_four_mm_and_wall_stays(self) -> None:
        from molmo_spaces.tasks.enclosure_reach import PactPlaceCorridorPolicy

        self.assertAlmostEqual(
            PactPlaceCorridorPolicy.MOUNTED_FIXTURE_ENVELOPE_HALF_Y, 0.10
        )
        self.assertAlmostEqual(
            PactPlaceCorridorPolicy.CEILING_FIXTURE_ENVELOPE_HALF_Y_NEG,
            WRIST_LAG_NEG_M + 0.004,
        )
        self.assertAlmostEqual(
            PactPlaceCorridorPolicy.CEILING_FIXTURE_ENVELOPE_HALF_Y_POS,
            WRIST_LAG_POS_M + 0.004,
        )
        wall_policy = PactPlaceCorridorPolicy.__new__(PactPlaceCorridorPolicy)
        wall_policy.task = type("T", (), {"scene_params": {
            "pact_place_environment_version": "pact_place_corridor_v9_5_low_wall",
        }})()
        self.assertAlmostEqual(
            wall_policy._mounted_fixture_bow_envelope_half_y("wall_fixture", -1.0),
            0.10,
        )
        self.assertAlmostEqual(
            wall_policy._mounted_fixture_bow_envelope_half_y("ceiling_fixture", -1.0),
            0.10,
        )
        v98 = PactPlaceCorridorPolicy.__new__(PactPlaceCorridorPolicy)
        v98.task = type("T", (), {"scene_params": {
            "pact_place_environment_version": "pact_place_corridor_v9_8_pendant",
        }})()
        self.assertAlmostEqual(
            v98._mounted_fixture_bow_envelope_half_y("ceiling_fixture", -1.0),
            0.212,
        )
        self.assertAlmostEqual(
            v98._mounted_fixture_bow_envelope_half_y("ceiling_fixture", 1.0),
            0.112,
        )

    def test_offset_waypoint_algebra_cancels_cross_y(self) -> None:
        import numpy as np
        from types import SimpleNamespace

        from molmo_spaces.policy.solvers.object_manipulation.base_object_manipulation_planner_policy import (  # noqa: E402
            TCPMoveSegment,
        )
        from molmo_spaces.tasks.enclosure_reach import PactPlaceCorridorPolicy

        constants = ceiling_fixture_bow_policy_constants()
        a_min, b_max = pendant_admissible_faces_m(constants)
        self.assertAlmostEqual(a_min, 0.044, places=6)
        self.assertAlmostEqual(b_max, 0.156, places=6)
        gap = constants["safe_gap_m"]
        env_neg = constants["envelope_half_y_neg_m"]
        env_pos = constants["envelope_half_y_pos_m"]
        policy = PactPlaceCorridorPolicy.__new__(PactPlaceCorridorPolicy)
        policy.task = SimpleNamespace(
            scene_params={
                "ap_w": DEFAULT_APERTURE_WIDTH_M,
                "pact_place_environment_version": "pact_place_corridor_v9_8_pendant",
            }
        )
        policy.policy_config = SimpleNamespace(speed_fast=0.12, speed_slow=0.045)
        spec = OFFSET_CANDIDATES["wide"]
        a_face, b_face = pendant_faces_m(spec["center_y_m"], spec["half_y_m"])
        expected_neg = a_face - gap - env_neg
        expected_pos = b_face + gap + env_pos
        center = np.array([0.72, spec["center_y_m"], 1.3075])
        half = np.array([0.10, spec["half_y_m"], 0.2075])
        for cross_y in (-0.05, 0.0, 0.05):
            for side, expected, envelope in (
                (-1.0, expected_neg, env_neg),
                (1.0, expected_pos, env_pos),
            ):
                start = np.eye(4)
                start[:3, 3] = [0.40, cross_y, 0.885]
                end = np.eye(4)
                end[:3, 3] = [0.95, cross_y, 0.885]
                segment = TCPMoveSegment(
                    name="synthetic_inbound",
                    start_pose=start,
                    end_pose=end,
                    speed=0.12,
                )
                policy._pact_place_bow_diagnostics = (
                    PactPlaceCorridorPolicy._empty_bow_diagnostics()
                )
                pieces, bowed = policy._bow_segment(
                    segment,
                    prefix="inbound_ceiling_fixture",
                    envelope_half_y=envelope,
                    safe_gap=gap,
                    center=center,
                    half=half,
                    preferred_waypoint_side=side,
                )
                self.assertTrue(bowed)
                waypoint_y = float(pieces[0].end_pose[1, 3])
                self.assertAlmostEqual(waypoint_y, expected, places=6)
                diag = policy._pact_place_bow_diagnostics["inbound_ceiling_fixture"]
                self.assertAlmostEqual(
                    diag["accepted_bow_m"], diag["planned_bow_m"], places=6
                )


if __name__ == "__main__":
    unittest.main()
