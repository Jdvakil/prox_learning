from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from pact_place_v95_contract import load_v95_palette
from run_pact_place_v95_low_wall_preview import build_row, build_wall_fixture


class PactPlaceV95LowWallTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.palette = load_v95_palette()

    def test_fixture_is_attached_low_and_inside_depth(self) -> None:
        for support in ("wall_left", "wall_right"):
            for seed in range(32):
                fixture = build_wall_fixture(support=support, seed=seed)
                center, half = fixture["center_m"], fixture["half_m"]
                side = 1.0 if support == "wall_left" else -1.0
                self.assertAlmostEqual(
                    side * (center[1] + side * half[1]), 0.45, places=8
                )
                self.assertGreaterEqual(center[0] - half[0], 0.58)
                self.assertLessEqual(center[0] + half[0], 0.86)
                self.assertGreaterEqual(center[2] - half[2], 0.87)
                self.assertLessEqual(center[2] - half[2], 0.98)
                self.assertGreaterEqual(center[2] + half[2], 1.06)
                self.assertLessEqual(center[2] + half[2], 1.15)
                self.assertGreaterEqual(1.03 - (center[2] - half[2]), 0.05)

    def test_panel_side_does_not_change_wall_geometry(self) -> None:
        rows = []
        for index, panel in enumerate(("left", "right")):
            rows.append(
                build_row(
                    cell_index=index,
                    candidate=0,
                    panel_side=panel,
                    wall_support="wall_left",
                    palette_document=self.palette,
                    implementation_sha256="test",
                    seed=950024,
                )
            )
        self.assertEqual(
            rows[0]["pact_mounted_wall_fixture"],
            rows[1]["pact_mounted_wall_fixture"],
        )
        self.assertEqual(
            {row["sampler_class"] for row in rows},
            {"PactPlaceCorridorV95LowWallSampler"},
        )
        self.assertTrue(all("pact_mounted_fixtures" not in row for row in rows))

    def test_fixture_randomizes_without_ceiling(self) -> None:
        fixtures = [build_wall_fixture(support="wall_right", seed=i) for i in range(8)]
        self.assertEqual(len({str(item) for item in fixtures}), 8)
        self.assertTrue(all(item["support"] == "wall_right" for item in fixtures))

    def test_v95_uses_distinct_wider_inbound_vessel_without_mutating_v93(self) -> None:
        from pact_place_v9_contract import load_palette

        v93 = load_palette()
        v93_inbound = next(
            item for item in v93["palette"] if item.get("role") == "inbound_vessel"
        )
        v95_inbound = next(
            item for item in self.palette["palette"] if item.get("role") == "inbound_vessel"
        )
        self.assertEqual(v93_inbound["uid"], "Soap_Bottle_1")
        self.assertEqual(v95_inbound["uid"], "Soap_Bottle_11")
        self.assertGreater(v95_inbound["dimensions_m"][1], v93_inbound["dimensions_m"][1])


if __name__ == "__main__":
    unittest.main()
