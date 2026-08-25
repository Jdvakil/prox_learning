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
    PENDANT_BOTTOM_Z_BOUNDS_M,
    PENDANT_DEPTH_BOUNDS_M,
    PENDANT_HALF_Y_BOUNDS_M,
    build_pendant_fixture,
    pendant_aabb,
    validate_pendant_geometry,
)
from run_pact_place_v98_pendant_preview import build_row  # noqa: E402


class PactPlaceV98PendantTest(unittest.TestCase):
    def test_contract_rejects_out_of_range_geometry(self) -> None:
        nominal = build_pendant_fixture()
        cases = [
            ([0.72, 0.001, nominal["center_m"][2]], nominal["half_m"]),
            ([0.72, 0.0, 1.30], nominal["half_m"]),
            ([0.72, 0.0, 1.3325], [0.1, 0.11, 0.1825]),
            ([0.72, 0.0, 1.3325], [0.1, 0.19, 0.1825]),
            ([0.40, 0.0, 1.3325], nominal["half_m"]),
        ]
        for center, half in cases:
            with self.subTest(center=center, half=half):
                with self.assertRaises(ValueError):
                    validate_pendant_geometry(center, half)

    def test_builder_honors_all_frozen_bounds(self) -> None:
        for bottom in PENDANT_BOTTOM_Z_BOUNDS_M:
            for half_y in PENDANT_HALF_Y_BOUNDS_M:
                fixture = build_pendant_fixture(bottom_z_m=bottom, half_y_m=half_y)
                center, half = fixture["center_m"], fixture["half_m"]
                self.assertEqual(center[1], 0.0)
                self.assertAlmostEqual(center[2] + half[2], CEILING_TOP_Z_M)
                self.assertGreaterEqual(center[0] - half[0], PENDANT_DEPTH_BOUNDS_M[0])
                self.assertLessEqual(center[0] + half[0], PENDANT_DEPTH_BOUNDS_M[1])
                self.assertEqual(fixture["lateral_lane_cost_m"], 0.0)

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
        self.assertEqual(low, [0.62, -0.18, 1.15])
        self.assertEqual(high, [0.82, 0.18, 1.5150000000000001])
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


if __name__ == "__main__":
    unittest.main()
