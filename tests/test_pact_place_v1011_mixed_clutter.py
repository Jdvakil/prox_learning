from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "scripts", ROOT / "submodules" / "molmospaces"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from pact_place_v1011_contract import (  # noqa: E402
    ACTIVE_CLUTTER_SLOTS,
    INACTIVE_CLUTTER_SLOTS,
    MESH_SLOTS,
    PRIMITIVE_SLOTS,
    build_contract,
    build_row,
    cells,
    preflight_rows,
)
from pact_place_v1011b_contract import (  # noqa: E402
    PRIMITIVE_HEIGHTS_M as V1011B_PRIMITIVE_HEIGHTS_M,
    build_contract as build_v1011b_contract,
    build_row as build_v1011b_row,
    preflight_rows as v1011b_preflight_rows,
)
from pact_place_v1011c_contract import (  # noqa: E402
    HEIGHT_MULTIPLIER_FROM_PARENT as V1011C_HEIGHT_MULTIPLIER,
    PRIMITIVE_HEIGHTS_M as V1011C_PRIMITIVE_HEIGHTS_M,
    build_contract as build_v1011c_contract,
    build_row as build_v1011c_row,
    preflight_rows as v1011c_preflight_rows,
)
from molmo_spaces.tasks.enclosure_reach import (  # noqa: E402
    PACT_PLACE_V106_LANE_ENVIRONMENT_VERSIONS,
    PactPlaceCorridorV1011MixedClutterSampler,
    PactPlaceCorridorV1011BTallPrimitiveSampler,
    PactPlaceCorridorV1011C33PctTallerPrimitiveSampler,
)
from molmo_spaces.tasks.pact_place_contact_audit import classify_contact  # noqa: E402


class V1011ContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.row = build_row(*cells()[0], 0)

    def test_composition_and_palette_contract(self) -> None:
        palette = self.row["pact_clutter_palette"]
        self.assertEqual(len(palette), 10)
        by_slot = {str(item["slot"]): item for item in palette}
        self.assertEqual(set(ACTIVE_CLUTTER_SLOTS), set(PRIMITIVE_SLOTS) | set(MESH_SLOTS))
        self.assertEqual(set(INACTIVE_CLUTTER_SLOTS), {"00", "02", "05", "07"})
        self.assertEqual(sum(bool(item.get("primitive")) for item in palette), 3)
        self.assertEqual(by_slot["01"]["category"], "vase")
        self.assertEqual(by_slot["01"]["role"], "outbound_vessel")
        self.assertEqual(by_slot["06"]["role"], "inbound_vessel")
        self.assertEqual(by_slot["01"]["dimensions_m"], [0.09, 0.09, 0.22])
        self.assertEqual(by_slot["08"]["dimensions_m"], [0.07, 0.07, 0.1])
        self.assertEqual(by_slot["09"]["dimensions_m"], [0.07, 0.07, 0.1])
        counts = {}
        for item in palette:
            self.assertEqual(item["slot_class"], "prop")
            self.assertEqual(item["support"], "shelf_standing")
            counts[item["category"]] = counts.get(item["category"], 0) + 1
        self.assertLessEqual(max(counts.values()), 2)

    def test_route_predicates_are_rederived_and_admitted(self) -> None:
        layout = self.row["pact_clutter_layout"]
        self.assertTrue(layout["nominal_route_metrics"]["direct_route_blocked"])
        self.assertTrue(layout["nominal_route_metrics"]["detour_admitted"])
        self.assertTrue(layout["panel_corridor_metrics"]["detour_admitted"])

    def test_streams_and_preflight_population(self) -> None:
        contract = build_contract()
        self.assertTrue(contract["streams"]["disjoint"])
        self.assertEqual(len(preflight_rows()), 96)
        self.assertFalse(contract["authorizes_collection"])
        self.assertFalse(contract["authorizes_training"])
        self.assertFalse(contract["authorizes_evaluation"])

    def test_identity_excludes_episode_positions(self) -> None:
        sampler = PactPlaceCorridorV1011MixedClutterSampler
        palette = copy.deepcopy(self.row["pact_clutter_palette"])
        observed = sampler.mixed_identity_sha256(palette)
        self.assertEqual(observed, self.row["pact_v1011_identity_sha256"])
        # Layout placement is deliberately not part of the identity.
        self.row["pact_clutter_layout"]["objects"][0]["center_m"][0] += 0.01
        self.assertEqual(sampler.mixed_identity_sha256(palette), observed)

    def test_environment_is_registered_as_v106_lane(self) -> None:
        self.assertIn(
            "pact_place_corridor_v10_11_mixed_clutter",
            PACT_PLACE_V106_LANE_ENVIRONMENT_VERSIONS,
        )


class V1011PlacementTests(unittest.TestCase):
    def _sampler(self):
        sampler = PactPlaceCorridorV1011MixedClutterSampler.__new__(
            PactPlaceCorridorV1011MixedClutterSampler
        )
        sampler._pact_manifest_row = build_row(*cells()[0], 0)
        return sampler

    def test_single_target_draw_and_zero_second_jitter(self) -> None:
        np.random.seed(2026)
        first = self._sampler()._draw_theta()
        np.random.seed(2026)
        second = self._sampler()._draw_theta()
        self.assertEqual(first["pact_v1011_target_draw_count"], 1)
        self.assertEqual(PactPlaceCorridorV1011MixedClutterSampler.OBJ_JIT_XY, (0.0, 0.0))
        self.assertEqual(first["pact_v1011_target_rest_m"], second["pact_v1011_target_rest_m"])
        self.assertEqual(
            first["pact_v1011_near_target_placements"],
            second["pact_v1011_near_target_placements"],
        )

    def test_annulus_bounds_and_area_uniform_parameter(self) -> None:
        sampler = self._sampler()
        sampler._v1011_target_rest = (0.75, 0.0, 0.72)
        half = np.asarray([0.035, 0.035, 0.05])
        np.random.seed(44)
        details = [
            sampler._sample_near_target_center(
                half=half,
                object_planar_radius_m=0.035,
                occupied=[],
            )[1]
            for _ in range(2000)
        ]
        radii = np.asarray([item["radius_m"] for item in details])
        units = np.asarray([item["area_uniform_u"] for item in details])
        rmin = details[0]["radius_min_m"]
        self.assertTrue(np.all(radii >= rmin))
        self.assertTrue(np.all(radii <= sampler.NEAR_RADIUS_MAX_M))
        self.assertAlmostEqual(float(units.mean()), 0.5, delta=0.03)

    def test_box_annulus_uses_rotation_invariant_half_diagonal(self) -> None:
        sampler = self._sampler()
        sampler._v1011_target_rest = (0.75, 0.0, 0.72)
        half = np.asarray([0.035, 0.035, 0.05])
        target_radius, _ = sampler.target_planar_bounding_radius_m()
        np.random.seed(91)
        _, detail = sampler._sample_near_target_center(
            half=half,
            object_planar_radius_m=float(np.linalg.norm(half[:2])),
            occupied=[],
        )
        expected = target_radius + float(np.linalg.norm(half[:2])) + 0.020
        self.assertAlmostEqual(detail["radius_min_m"], expected, places=12)
        self.assertAlmostEqual(
            detail["object_planar_bounding_radius_m"],
            float(np.linalg.norm(half[:2])),
            places=12,
        )

    def test_near_objects_are_in_sector_and_do_not_overlap(self) -> None:
        np.random.seed(17)
        theta = self._sampler()._draw_theta()
        layout = theta["pact_clutter_layout"]
        target = np.asarray(theta["pact_v1011_target_rest_m"][:2], dtype=float)
        by_slot = {item["palette_slot"]: item for item in layout["objects"]}
        for slot in ("08", "09"):
            delta = np.asarray(by_slot[slot]["center_m"][:2]) - target
            angle = float(np.arctan2(delta[1], delta[0]))
            self.assertGreaterEqual(angle, np.deg2rad(-65.0) - 1e-9)
            self.assertLessEqual(angle, np.deg2rad(65.0) + 1e-9)
            self.assertGreaterEqual(float(np.linalg.norm(delta)), 0.09)
            self.assertLessEqual(float(np.linalg.norm(delta)), 0.220 + 1e-9)
        c8, c9 = (np.asarray(by_slot[s]["center_m"][:2]) for s in ("08", "09"))
        self.assertTrue(np.any(np.abs(c8 - c9) >= np.asarray([0.08, 0.08]) - 1e-9))

    def test_primitive_namespace_classifies_as_clutter_only(self) -> None:
        pair = {
            "geom1": "robot_0/fr3_link7_collision",
            "geom2": "pact_clutter_08/pact_primitive_cylinder_08_collision",
            "body1": "robot_0/fr3_link7",
            "body2": "pact_clutter_08/pact_primitive_cylinder_08",
            "root1": "robot_0/link0",
            "root2": "pact_clutter_08/pact_primitive_cylinder_08",
        }
        self.assertEqual(classify_contact(pair), "clutter")
        self.assertNotIn("pact_clutter_mount_", " ".join(pair.values()))


class V1011BTallPrimitiveTests(unittest.TestCase):
    def setUp(self) -> None:
        family, side, pose = cells()[0]
        self.parent = build_row(family, side, pose, 0)
        self.row = build_v1011b_row(family, side, pose, 0)

    def test_only_primitive_z_dimensions_change(self) -> None:
        parent = {
            str(item["slot"]): item for item in self.parent["pact_clutter_palette"]
        }
        tall = {
            str(item["slot"]): item for item in self.row["pact_clutter_palette"]
        }
        for slot, expected_height in V1011B_PRIMITIVE_HEIGHTS_M.items():
            self.assertEqual(tall[slot]["dimensions_m"][:2], parent[slot]["dimensions_m"][:2])
            self.assertEqual(tall[slot]["dimensions_m"][2], expected_height)
        for slot in MESH_SLOTS:
            self.assertEqual(tall[slot], parent[slot])

    def test_target_and_route_geometry_are_not_resized(self) -> None:
        self.assertEqual(
            self.row["pact_v106_scene_sha256"],
            self.parent["pact_v106_scene_sha256"],
        )
        self.assertEqual(self.row["target_x_jitter_m"], self.parent["target_x_jitter_m"])
        self.assertEqual(self.row["target_y_jitter_m"], self.parent["target_y_jitter_m"])
        parent_objects = {
            str(item["palette_slot"]): item
            for item in self.parent["pact_clutter_layout"]["objects"]
        }
        tall_objects = {
            str(item["palette_slot"]): item
            for item in self.row["pact_clutter_layout"]["objects"]
        }
        for slot in V1011B_PRIMITIVE_HEIGHTS_M:
            self.assertEqual(
                tall_objects[slot]["center_m"][:2], parent_objects[slot]["center_m"][:2]
            )
            self.assertEqual(
                tall_objects[slot]["half_m"][:2], parent_objects[slot]["half_m"][:2]
            )

    def test_contract_is_fresh_and_not_authorized(self) -> None:
        contract = build_v1011b_contract()
        self.assertTrue(contract["streams"]["disjoint"])
        self.assertEqual(len(v1011b_preflight_rows()), 96)
        self.assertTrue(contract["height_only_amendment"]["xy_footprints_unchanged"])
        self.assertTrue(contract["height_only_amendment"]["cup_and_target_dimensions_unchanged"])
        self.assertFalse(contract["authorizes_collection"])

    def test_sampler_marker_and_height_validation(self) -> None:
        sampler = PactPlaceCorridorV1011BTallPrimitiveSampler.__new__(
            PactPlaceCorridorV1011BTallPrimitiveSampler
        )
        sampler._pact_manifest_row = self.row
        np.random.seed(2026)
        theta = sampler._draw_theta()
        self.assertEqual(
            theta["pact_place_environment_version"],
            "pact_place_corridor_v10_11b_tall_primitives",
        )
        self.assertEqual(
            theta["pact_v1011b_tall_primitive_heights_m"],
            V1011B_PRIMITIVE_HEIGHTS_M,
        )
        self.assertTrue(theta["pact_v1011b_footprints_unchanged"])

    def test_environment_is_registered_as_v106_lane(self) -> None:
        self.assertIn(
            "pact_place_corridor_v10_11b_tall_primitives",
            PACT_PLACE_V106_LANE_ENVIRONMENT_VERSIONS,
        )


class V1011C33PctTallerPrimitiveTests(unittest.TestCase):
    def setUp(self) -> None:
        family, side, pose = cells()[0]
        self.parent = build_v1011b_row(family, side, pose, 0)
        self.row = build_v1011c_row(family, side, pose, 0)

    def test_all_three_heights_are_exactly_parent_times_1_33(self) -> None:
        parent_palette = {
            str(item["slot"]): item for item in self.parent["pact_clutter_palette"]
        }
        palette = {
            str(item["slot"]): item for item in self.row["pact_clutter_palette"]
        }
        self.assertEqual(V1011C_HEIGHT_MULTIPLIER, 1.33)
        for slot, height in V1011C_PRIMITIVE_HEIGHTS_M.items():
            self.assertAlmostEqual(
                height,
                parent_palette[slot]["dimensions_m"][2] * 1.33,
                places=12,
            )
            self.assertEqual(palette[slot]["dimensions_m"][2], height)
            self.assertEqual(
                palette[slot]["dimensions_m"][:2],
                parent_palette[slot]["dimensions_m"][:2],
            )

    def test_mesh_target_scene_and_xy_layout_are_unchanged(self) -> None:
        parent_palette = {
            str(item["slot"]): item for item in self.parent["pact_clutter_palette"]
        }
        palette = {
            str(item["slot"]): item for item in self.row["pact_clutter_palette"]
        }
        for slot in MESH_SLOTS:
            self.assertEqual(palette[slot], parent_palette[slot])
        self.assertEqual(
            self.row["pact_v106_scene_sha256"], self.parent["pact_v106_scene_sha256"]
        )
        parent_objects = {
            str(item["palette_slot"]): item
            for item in self.parent["pact_clutter_layout"]["objects"]
        }
        objects = {
            str(item["palette_slot"]): item
            for item in self.row["pact_clutter_layout"]["objects"]
        }
        for slot in V1011C_PRIMITIVE_HEIGHTS_M:
            self.assertEqual(objects[slot]["center_m"][:2], parent_objects[slot]["center_m"][:2])
            self.assertEqual(objects[slot]["half_m"][:2], parent_objects[slot]["half_m"][:2])

    def test_contract_population_and_authorization_boundary(self) -> None:
        contract = build_v1011c_contract()
        self.assertTrue(contract["streams"]["disjoint"])
        self.assertEqual(len(v1011c_preflight_rows()), 96)
        self.assertEqual(contract["height_only_amendment"]["multiplier"], 1.33)
        self.assertTrue(contract["height_only_amendment"]["cup_and_target_dimensions_unchanged"])
        self.assertFalse(contract["authorizes_collection"])

    def test_sampler_accepts_exact_c_contract(self) -> None:
        sampler = PactPlaceCorridorV1011C33PctTallerPrimitiveSampler.__new__(
            PactPlaceCorridorV1011C33PctTallerPrimitiveSampler
        )
        sampler._pact_manifest_row = self.row
        np.random.seed(2026)
        theta = sampler._draw_theta()
        self.assertEqual(
            theta["pact_place_environment_version"],
            "pact_place_corridor_v10_11c_33pct_taller_primitives",
        )
        self.assertEqual(
            theta["pact_v1011c_primitive_heights_m"],
            V1011C_PRIMITIVE_HEIGHTS_M,
        )

    def test_environment_is_registered_as_v106_lane(self) -> None:
        self.assertIn(
            "pact_place_corridor_v10_11c_33pct_taller_primitives",
            PACT_PLACE_V106_LANE_ENVIRONMENT_VERSIONS,
        )


if __name__ == "__main__":
    unittest.main()
