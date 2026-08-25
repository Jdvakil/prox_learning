from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from pact_place_v9_contract import load_palette
from run_pact_place_v94_mounted_preview import build_fixtures, build_row


class PactPlaceV94MountedPreviewTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.palette = load_palette()

    def test_fixtures_are_attached_and_in_bounds(self) -> None:
        seen = set()
        for support in ("wall_left", "wall_right"):
            for ceiling_side in (-1, 1):
                for seed in range(12):
                    fixtures = build_fixtures(
                        wall_support=support,
                        ceiling_side=ceiling_side,
                        seed=seed,
                    )
                    self.assertEqual(len(fixtures), 2)
                    wall, ceiling = fixtures
                    wall_sign = 1.0 if support == "wall_left" else -1.0
                    self.assertAlmostEqual(
                        wall_sign
                        * (wall["center_m"][1] + wall_sign * wall["half_m"][1]),
                        0.45,
                        places=7,
                    )
                    self.assertAlmostEqual(
                        ceiling["center_m"][2] + ceiling["half_m"][2],
                        1.515,
                        places=7,
                    )
                    self.assertEqual(
                        1 if ceiling["center_m"][1] > 0.0 else -1,
                        ceiling_side,
                    )
                    for item in fixtures:
                        lo = item["center_m"][0] - item["half_m"][0]
                        hi = item["center_m"][0] + item["half_m"][0]
                        self.assertGreaterEqual(lo, 0.58)
                        self.assertLessEqual(hi, 1.36)
                    seen.add(
                        tuple(
                            round(value, 5)
                            for item in fixtures
                            for value in item["center_m"] + item["half_m"]
                        )
                    )
        self.assertGreater(len(seen), 12)

    def test_four_cells_balance_panel_and_wall_without_label_leak(self) -> None:
        cells = (
            ("left", "wall_left", 1),
            ("left", "wall_right", -1),
            ("right", "wall_left", -1),
            ("right", "wall_right", 1),
        )
        rows = [
            build_row(
                cell_index=index,
                candidate=0,
                panel_side=panel,
                wall_support=wall,
                ceiling_side=ceiling,
                palette_document=self.palette,
                implementation_sha256="test-implementation",
                seed=940024,
            )
            for index, (panel, wall, ceiling) in enumerate(cells)
        ]
        self.assertEqual(
            {(row["intrusion_side"], row["mounted_wall_support"]) for row in rows},
            {
                ("left", "wall_left"),
                ("left", "wall_right"),
                ("right", "wall_left"),
                ("right", "wall_right"),
            },
        )
        self.assertEqual(
            {row["sampler_class"] for row in rows},
            {"PactPlaceCorridorV94MountedPreviewSampler"},
        )
        self.assertTrue(all(len(row["pact_mounted_fixtures"]) == 2 for row in rows))
        self.assertEqual(len({row["episode_id"] for row in rows}), 4)
        self.assertEqual(len({row["row_sha256"] for row in rows}), 4)

    def test_fixture_randomization_changes_geometry(self) -> None:
        first = build_fixtures(wall_support="wall_left", ceiling_side=1, seed=1)
        second = build_fixtures(wall_support="wall_left", ceiling_side=1, seed=2)
        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
