from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "scripts", ROOT / "submodules" / "molmospaces"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pact_place_v10_compound_pendant_contract import (
    ENDPOINT_ONLY_PRIMITIVE,
    GROUP_FREEZE_PRIMITIVE,
    MAX_SEGMENT_ROTATION_DEG,
    MAX_SEGMENT_TRANSLATION_M,
    MIN_DETOUR_M,
)
from pact_place_v10_geometry import planning_probe_assembly, union_fixture
from pact_place_v10_route import (
    apply_constant_lane_endpoint_only,
    assign_members_across_routes,
    frozen_endpoint_preserved,
    path_step_limits,
    plan_lane,
    plan_lane_endpoint_only,
)
from pact_place_v99_route import apply_constant_lane, densify_path


def _straight_path(xs, y=0.02, z=0.88):
    positions = np.column_stack(
        [np.asarray(xs, dtype=float), np.full(len(xs), y), np.full(len(xs), z)]
    )
    rotations = np.stack([np.eye(3)] * len(xs))
    return positions, rotations


class PactPlaceV10EndpointOnlyRouteTest(unittest.TestCase):
    def test_group_with_final_endpoint_rewrites_other_samples(self) -> None:
        positions, rotations = _straight_path([0.90, 0.80, 0.70, 0.60, 0.50])
        dense_p, dense_r = densify_path(positions, rotations)
        planned, _, rewritten = apply_constant_lane_endpoint_only(
            dense_p,
            dense_r,
            lane_y=-0.20,
            entry_x=0.92,
            exit_x=0.48,
            freeze_start=False,
            freeze_final=True,
        )
        self.assertGreater(int(np.sum(rewritten)), 0)
        self.assertFalse(bool(rewritten[-1]))
        self.assertTrue(np.any(rewritten[:-1]))
        self.assertTrue(np.allclose(planned[rewritten, 1], -0.20))
        self.assertTrue(np.allclose(planned[-1], dense_p[-1]))

    def test_group_with_start_endpoint_rewrites_other_samples(self) -> None:
        positions, rotations = _straight_path([0.50, 0.60, 0.70, 0.80, 0.90])
        dense_p, dense_r = densify_path(positions, rotations)
        planned, _, rewritten = apply_constant_lane_endpoint_only(
            dense_p,
            dense_r,
            lane_y=0.20,
            entry_x=0.48,
            exit_x=0.92,
            freeze_start=True,
            freeze_final=False,
        )
        self.assertGreater(int(np.sum(rewritten)), 0)
        self.assertFalse(bool(rewritten[0]))
        self.assertTrue(np.any(rewritten[1:]))
        self.assertTrue(np.allclose(planned[rewritten, 1], 0.20))
        self.assertTrue(np.allclose(planned[0], dense_p[0]))

    def test_only_requested_endpoint_unchanged(self) -> None:
        positions, rotations = _straight_path([0.90, 0.75, 0.60, 0.45])
        dense_p, dense_r = densify_path(positions, rotations)
        planned, planned_r, rewritten = apply_constant_lane_endpoint_only(
            dense_p,
            dense_r,
            lane_y=-0.18,
            entry_x=0.92,
            exit_x=0.40,
            freeze_start=False,
            freeze_final=True,
        )
        self.assertTrue(np.allclose(planned[-1], dense_p[-1]))
        self.assertTrue(np.allclose(planned_r[-1], dense_r[-1]))
        changed = np.flatnonzero(rewritten)
        self.assertTrue(len(changed) > 0)
        self.assertFalse(np.allclose(planned[changed[0], 1], dense_p[changed[0], 1]))

    def test_frozen_position_and_rotation_preserved_after_plan(self) -> None:
        assembly = planning_probe_assembly()
        positions, rotations = _straight_path([0.95, 0.70, 0.40])
        planned = plan_lane_endpoint_only(
            positions,
            rotations,
            assembly=assembly,
            panel_side="left",
            lane_y_m=-0.12,
            padding_m=0.10,
            freeze_start=False,
            freeze_final=True,
        )
        stock_p = planned["stock_positions_m"]
        stock_r = planned["stock_rotations"]
        preserved = frozen_endpoint_preserved(
            planned["planned_positions_m"],
            planned["planned_rotations"],
            stock_p,
            stock_r,
            freeze_start=False,
            freeze_final=True,
        )
        self.assertTrue(preserved["preserved"])
        self.assertEqual(planned["rewrite_primitive"], ENDPOINT_ONLY_PRIMITIVE)

    def test_densified_steps_respect_translation_and_rotation_caps(self) -> None:
        assembly = planning_probe_assembly()
        positions, rotations = _straight_path([0.95, 0.70, 0.40])
        planned = plan_lane_endpoint_only(
            positions,
            rotations,
            assembly=assembly,
            panel_side="left",
            lane_y_m=-0.20,
            padding_m=0.10,
            freeze_start=False,
            freeze_final=True,
        )
        steps = path_step_limits(
            planned["planned_positions_m"], planned["planned_rotations"]
        )
        self.assertLessEqual(steps["max_translation_m"], MAX_SEGMENT_TRANSLATION_M + 1e-12)
        self.assertLessEqual(steps["max_rotation_deg"], MAX_SEGMENT_ROTATION_DEG + 1e-12)
        self.assertTrue(steps["within_limits"])
        self.assertTrue(planned["continuous_after_densify"])

    def test_detour_measured_throughout_physical_slab(self) -> None:
        assembly = planning_probe_assembly()
        fixture = union_fixture(assembly)
        positions, rotations = _straight_path(np.linspace(0.95, 0.40, 12))
        planned = plan_lane_endpoint_only(
            positions,
            rotations,
            assembly=assembly,
            panel_side="left",
            lane_y_m=-0.18,
            padding_m=0.10,
            freeze_start=False,
            freeze_final=True,
        )
        lo = float(fixture["center_m"][0] - fixture["half_m"][0])
        hi = float(fixture["center_m"][0] + fixture["half_m"][0])
        xs = planned["planned_positions_m"][:, 0]
        in_slab = (xs >= lo - 1e-12) & (xs <= hi + 1e-12)
        self.assertGreater(int(np.sum(in_slab)), 1)
        self.assertGreaterEqual(int(planned["detour"]["n_samples"]), int(np.sum(in_slab)))
        self.assertIn("min_abs_detour_m", planned["detour"])

    def test_frozen_endpoint_inside_physical_slab_can_fail_detour(self) -> None:
        fixture = {"center_m": [0.70, 0.0, 0.90], "half_m": [0.20, 0.10, 0.10]}
        positions, rotations = _straight_path([0.95, 0.85, 0.75, 0.70])
        planned = plan_lane_endpoint_only(
            positions,
            rotations,
            fixture=fixture,
            panel_side="left",
            lane_y_m=-0.20,
            padding_m=0.08,
            freeze_start=False,
            freeze_final=True,
        )
        self.assertGreater(planned["rewritten_samples"], 0)
        self.assertTrue(
            planned["physical_x_lo_m"] - 1e-12
            <= planned["planned_positions_m"][-1, 0]
            <= planned["physical_x_hi_m"] + 1e-12
        )
        self.assertLess(planned["detour"]["min_abs_detour_m"], MIN_DETOUR_M)
        self.assertFalse(planned["detour"]["meets_minimum"])
        self.assertFalse(planned["accepted_geometry"])

    def test_historical_v99_and_route_v1_still_suppress_whole_group(self) -> None:
        positions, rotations = _straight_path([0.90, 0.80, 0.70, 0.60, 0.50])
        dense_p, dense_r = densify_path(positions, rotations)
        v99_p, _, v99_mask = apply_constant_lane(
            dense_p,
            dense_r,
            lane_y=-0.20,
            entry_x=0.92,
            exit_x=0.48,
            freeze_start=False,
            freeze_final=True,
        )
        v2_p, _, v2_mask = apply_constant_lane_endpoint_only(
            dense_p,
            dense_r,
            lane_y=-0.20,
            entry_x=0.92,
            exit_x=0.48,
            freeze_start=False,
            freeze_final=True,
        )
        self.assertEqual(int(np.sum(v99_mask)), 0)
        self.assertTrue(np.allclose(v99_p, dense_p))
        self.assertGreater(int(np.sum(v2_mask)), 0)
        assembly = planning_probe_assembly()
        v1 = plan_lane(
            positions,
            rotations,
            assembly=assembly,
            panel_side="left",
            lane_y_m=-0.12,
            padding_m=0.10,
            freeze_start=False,
            freeze_final=True,
        )
        self.assertEqual(v1["rewrite_primitive"], GROUP_FREEZE_PRIMITIVE)

    def test_exhausted_first_route_recovers_on_second(self) -> None:
        members = [10, 11]
        identities = [
            {"padding_m": 0.08, "lane_y_m": -0.10},
            {"padding_m": 0.10, "lane_y_m": -0.20},
        ]

        def evaluate_route(identity):
            return {
                "ik_ok": True,
                "environment_clear": True,
                "n_corners_evaluated": 8,
                "robust_ok": True,
                "route": identity,
            }

        def score_members(remaining, report):
            lane = float(report["route"]["lane_y_m"])
            if abs(lane + 0.10) < 1e-9:
                return {int(member): False for member in remaining}
            return {int(member): True for member in remaining}

        result = assign_members_across_routes(
            members,
            identities,
            evaluate_route=evaluate_route,
            score_members=score_members,
        )
        self.assertEqual(result["exhausted"], [])
        self.assertEqual(result["alternative_route_recoveries"], 2)
        self.assertEqual(result["selected"][10]["lane_y_m"], -0.20)
        self.assertEqual(result["morphology_clearance"]["failed"], 2)
        self.assertEqual(result["morphology_clearance"]["passed"], 2)
        self.assertEqual(result["nominal_ik"]["attempted"], 2)
        self.assertEqual(result["all_eight_corners_evaluated"], True)

    def test_nominal_clearance_empty_skips_corners_and_keeps_searching(self) -> None:
        members = [1]
        identities = [
            {"padding_m": 0.08, "lane_y_m": -0.10},
            {"padding_m": 0.10, "lane_y_m": -0.20},
        ]

        def evaluate_route(identity):
            if abs(float(identity["lane_y_m"]) + 0.10) < 1e-9:
                return {
                    "ik_ok": True,
                    "environment_clear": True,
                    "nominal_clearance_empty": True,
                    "n_corners_evaluated": 0,
                    "robust_ok": False,
                }
            return {
                "ik_ok": True,
                "environment_clear": True,
                "n_corners_evaluated": 8,
                "robust_ok": True,
                "route": identity,
            }

        def score_members(remaining, report):
            return {int(member): True for member in remaining}

        result = assign_members_across_routes(
            members,
            identities,
            evaluate_route=evaluate_route,
            score_members=score_members,
        )
        self.assertEqual(result["selected"][1]["lane_y_m"], -0.20)
        self.assertEqual(result["robust_routes"]["attempted"], 1)
        self.assertEqual(result["robust_routes"]["not_evaluated"], 1)
        self.assertEqual(result["morphology_clearance"]["failed"], 1)
        self.assertEqual(result["morphology_clearance"]["passed"], 1)
        self.assertEqual(result["all_eight_corners_evaluated"], True)


if __name__ == "__main__":
    unittest.main()
