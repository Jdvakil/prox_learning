from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from pact_place_v9_contract import (
    LAYOUT_FAMILIES,
    REVIEW_MASTER_SEED,
    V9_REVIEW_STREAM,
    _rows,
    build_layout,
    load_palette,
    panel_corridor_metrics,
    route_blocker_metrics,
    validate_layout,
)


class PactPlaceV9RedesignTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.palette = load_palette()

    def test_every_family_blocks_and_admits_nominal_loaded_route(self) -> None:
        observed_directions = []
        for family_id in LAYOUT_FAMILIES:
            for side in ("left", "right"):
                layout = build_layout(
                    self.palette, family_id=family_id, intrusion_side=side
                )
                metrics = route_blocker_metrics(layout)
                corridor = panel_corridor_metrics(layout)
                self.assertTrue(metrics["direct_route_blocked"])
                self.assertTrue(metrics["detour_admitted"])
                self.assertGreaterEqual(metrics["required_bow_m"], 0.04)
                self.assertLessEqual(
                    abs(metrics["planned_waypoint_y_m"]), metrics["lateral_limit_m"]
                )
                self.assertTrue(corridor["panel_active"])
                self.assertTrue(corridor["detour_admitted"])
                self.assertGreaterEqual(corridor["corridor_margin_m"], 0.019)
                observed_directions.append(metrics["bow_direction"])
        self.assertEqual(observed_directions.count("+y"), 4)
        self.assertEqual(observed_directions.count("-y"), 4)

    def test_visible_clutter_does_not_leak_panel_side(self) -> None:
        for family_id in LAYOUT_FAMILIES:
            left = build_layout(self.palette, family_id=family_id, intrusion_side="left")
            right = build_layout(self.palette, family_id=family_id, intrusion_side="right")
            self.assertEqual(left["objects"], right["objects"])
            self.assertEqual(left["expected_bow_direction"], "-y")
            self.assertEqual(right["expected_bow_direction"], "+y")

    def test_both_vessels_are_substantial_and_layout_is_two_dimensional(self) -> None:
        observed_pairs = set()
        for family_id in LAYOUT_FAMILIES:
            layout = build_layout(self.palette, family_id=family_id, intrusion_side="left")
            vessels = [item for item in layout["objects"] if item["role"].endswith("vessel")]
            self.assertEqual({item["role"] for item in vessels}, {"inbound_vessel", "outbound_vessel"})
            self.assertTrue(all(2.0 * item["half_m"][1] >= 0.06 for item in vessels))
            pair = tuple((round(item["center_m"][0], 3), round(item["center_m"][1], 3)) for item in vessels)
            observed_pairs.add(pair)
            self.assertEqual(len({round(item["center_m"][1], 3) for item in vessels}), 2)
        self.assertEqual(len(observed_pairs), len(LAYOUT_FAMILIES))

    def test_review_rows_are_paired_with_nonzero_xy_vessel_jitter(self) -> None:
        layouts = {
            family: {
                side: build_layout(self.palette, family_id=family, intrusion_side=side)
                for side in ("left", "right")
            }
            for family in LAYOUT_FAMILIES
        }
        rows = _rows(
            stream=V9_REVIEW_STREAM,
            master_seed=REVIEW_MASTER_SEED,
            palette=list(self.palette["palette"]),
            layouts=layouts,
        )
        for index in range(0, len(rows), 2):
            left, right = rows[index : index + 2]
            self.assertEqual((left["intrusion_side"], right["intrusion_side"]), ("left", "right"))
            self.assertEqual(left["layout_family_id"], right["layout_family_id"])
            self.assertEqual(left["clutter_x_jitter_m"], right["clutter_x_jitter_m"])
            self.assertEqual(left["clutter_y_jitter_m"], right["clutter_y_jitter_m"])
        self.assertTrue(any(any(value != 0.0 for value in row["clutter_x_jitter_m"].values()) for row in rows))
        self.assertTrue(any(any(value != 0.0 for value in row["clutter_y_jitter_m"].values()) for row in rows))

    def test_old_transverse_line_is_rejected(self) -> None:
        layout = copy.deepcopy(build_layout(self.palette, intrusion_side="left"))
        for item in layout["objects"]:
            item["center_m"][0] = 0.715
        with self.assertRaisesRegex(ValueError, "transverse line"):
            validate_layout(layout)

    def test_nonblocking_perimeter_vessel_cannot_be_route_blocker(self) -> None:
        layout = copy.deepcopy(build_layout(self.palette, intrusion_side="left"))
        blocker = next(
            item
            for item in layout["objects"]
            if item["palette_slot"] == layout["route_blocker_slot"]
        )
        blocker["center_m"][1] = 0.34
        with self.assertRaisesRegex(ValueError, "does not obstruct"):
            validate_layout(layout)


if __name__ == "__main__":
    unittest.main()
