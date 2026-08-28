from __future__ import annotations

import ast
import inspect
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "scripts", ROOT / "submodules" / "molmospaces"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pact_place_corridor_contract import PLACE_V5_SCENE_SHA256, sha256_file
from pact_place_v99_geometry import (
    aabb_overlap,
    aabb_separation_m,
    enumerate_lattice,
    evaluate_stock_necessity,
    filter_lattice_dual_transit,
    filter_lattice_for_cells,
    geoms_intersect_box,
    lattice_raw_count,
)
from pact_place_v99_pendant_contract import (
    ADMISSION_FLOOR,
    CEILING_TOP_Z_M,
    CONTRACT_VERSION,
    CORRIDOR_LINKS,
    DEFAULT_SEED,
    ENVIRONMENT_VERSION,
    HALF_X_CHOICES_M,
    MAX_GRASP_JOINT_ERROR_RAD,
    MIN_DETOUR_M,
    MIN_GATE_CLEAN_SUCCESSES,
    N_GATE_ROWS,
    PHYSICS_CLEAN_FAMILIES,
    SAMPLER_CLASS,
    SCENE_XML_RELATIVE,
    V9_8_FORBIDDEN_KEYS,
    build_pendant_fixture,
    empty_authorization,
    grasp_posture_preserved,
    six_cell_gate_summary,
    validate_pendant_geometry,
)
from pact_place_v99_route import (
    PERTURBATION_CORNERS,
    apply_constant_lane,
    densify_path,
    interpolate_y_at_x,
    lane_y_grid,
    min_abs_detour_in_slab_m,
    panel_lane_sign,
    perturbation_corners,
    plan_lane,
    select_at_most_two_candidates,
    travel_sign_through_slab,
)


class PactPlaceV99PendantTest(unittest.TestCase):
    def test_contract_excludes_v98_lag_and_window_inputs(self) -> None:
        source = Path(
            ROOT / "scripts" / "pact_place_v99_pendant_contract.py"
        ).read_text()
        tree = ast.parse(source)
        imported = {
            alias.name
            for node in tree.body
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        from_modules = [
            node.module
            for node in tree.body
            if isinstance(node, ast.ImportFrom)
        ]
        self.assertNotIn("pact_place_v98_pendant_contract", from_modules)
        self.assertNotIn("WRIST_LAG_NEG_M", imported)
        self.assertNotIn("WRIST_LAG_POS_M", source)
        self.assertNotIn("FACE_WINDOW", source)
        self.assertNotIn("pendant_admissible_faces_m", source)
        fixture = build_pendant_fixture(
            center_x_m=0.72,
            center_y_m=0.0,
            half_x_m=0.05,
            half_y_m=0.10,
            bottom_z_m=1.10,
        )
        with self.assertRaises(ValueError) as raised:
            validate_pendant_geometry(
                fixture["center_m"],
                fixture["half_m"],
                wrist_lag_neg_m=0.208,
                face_window_m=(0.044, 0.156),
            )
        message = str(raised.exception)
        self.assertIn("V9.8 lag/window", message)
        for key in ("wrist_lag_neg_m", "face_window_m"):
            self.assertIn(key, V9_8_FORBIDDEN_KEYS)

    def test_geometry_validation_and_lattice_are_deterministic(self) -> None:
        raw = lattice_raw_count()
        self.assertEqual(raw, 17 * 4 * 13 * 8 * 7)
        first = enumerate_lattice()
        second = enumerate_lattice()
        self.assertEqual(len(first), len(second))
        self.assertGreater(len(first), 0)
        self.assertLess(len(first), raw)
        self.assertEqual(
            [item["center_m"] + item["half_m"] for item in first[:5]],
            [item["center_m"] + item["half_m"] for item in second[:5]],
        )
        rejected_low_x = build_pendant_fixture
        with self.assertRaises(ValueError):
            rejected_low_x(
                center_x_m=0.58,
                center_y_m=0.0,
                half_x_m=0.03,
                half_y_m=0.04,
                bottom_z_m=1.10,
            )
        ok = build_pendant_fixture(
            center_x_m=0.70,
            center_y_m=0.0,
            half_x_m=0.09,
            half_y_m=0.18,
            bottom_z_m=1.25,
        )
        self.assertAlmostEqual(ok["center_m"][2] + ok["half_m"][2], CEILING_TOP_Z_M)
        self.assertEqual(ok["center_m"][1], second[0]["center_m"][1] * 0.0)
        self.assertIn(ok["half_m"][0], HALF_X_CHOICES_M)

    def test_actual_v5_scene_hash_matches_frozen_contract(self) -> None:
        scene = ROOT / SCENE_XML_RELATIVE
        self.assertTrue(scene.is_file())
        self.assertEqual(sha256_file(scene), PLACE_V5_SCENE_SHA256)

    def test_aabb_necessity_requires_both_traversals_and_grasp_clearance(self) -> None:
        fixture = build_pendant_fixture(
            center_x_m=0.72,
            center_y_m=0.0,
            half_x_m=0.05,
            half_y_m=0.08,
            bottom_z_m=1.10,
        )
        n, g = 6, 2
        robot_lo = np.zeros((n, g, 3))
        robot_hi = np.ones((n, g, 3)) * 0.01
        robot_lo[1, 0] = [0.70, -0.01, 1.12]
        robot_hi[1, 0] = [0.74, 0.01, 1.20]
        robot_lo[4, 0] = [0.70, -0.01, 1.12]
        robot_hi[4, 0] = [0.74, 0.01, 1.20]
        target_lo = np.full((n, 1, 3), 10.0)
        target_hi = np.full((n, 1, 3), 10.1)
        inbound = np.array([False, True, False, False, False, False])
        outbound = np.array([False, False, False, False, True, False])
        grasp = np.array([False, False, True, True, False, False])
        initial = np.array([True, False, False, False, False, False])
        report = evaluate_stock_necessity(
            fixture=fixture,
            robot_lo=robot_lo,
            robot_hi=robot_hi,
            target_lo=target_lo,
            target_hi=target_hi,
            inbound_mask=inbound,
            outbound_mask=outbound,
            grasp_mask=grasp,
            initial_mask=initial,
            min_grasp_clearance_m=0.025,
        )
        self.assertTrue(report["accepted"])
        robot_lo[0, 0] = [0.70, -0.01, 1.12]
        robot_hi[0, 0] = [0.74, 0.01, 1.20]
        collided = evaluate_stock_necessity(
            fixture=fixture,
            robot_lo=robot_lo,
            robot_hi=robot_hi,
            target_lo=target_lo,
            target_hi=target_hi,
            inbound_mask=inbound,
            outbound_mask=outbound,
            grasp_mask=grasp,
            initial_mask=initial,
            min_grasp_clearance_m=0.025,
        )
        self.assertTrue(collided["initial_state_collision"])
        self.assertFalse(collided["accepted"])

    def test_aabb_overlap_is_not_exact_contact_or_clearance(self) -> None:
        import mujoco
        from pact_geom_distance import GeomShape, gjk_distance
        from pact_place_v99_exact import evaluate_fixture_exact

        cube = GeomShape.posed_axis_aligned_box(
            np.array([0.5, 0.5, 0.5]), np.array([0.5, 0.5, 0.5])
        )
        sphere = GeomShape(
            int(mujoco.mjtGeom.mjGEOM_SPHERE),
            np.array([1.20, 1.20, 0.5]),
            np.eye(3),
            np.array([0.25, 0.0, 0.0]),
        )
        sphere_lo, sphere_hi = sphere.world_aabb()
        cube_lo, cube_hi = cube.world_aabb()
        self.assertTrue(aabb_overlap(sphere_lo, sphere_hi, cube_lo, cube_hi))
        exact = gjk_distance(cube, sphere)
        self.assertGreater(exact, 0.025)
        nested = GeomShape.posed_axis_aligned_box(
            np.array([0.5, 0.5, 0.5]), np.array([0.1, 0.1, 0.1])
        )
        self.assertEqual(gjk_distance(cube, nested), 0.0)

        fixture = build_pendant_fixture(
            center_x_m=0.72,
            center_y_m=0.0,
            half_x_m=0.05,
            half_y_m=0.08,
            bottom_z_m=1.10,
        )
        n, g = 6, 1
        robot_lo = np.full((n, g, 3), 10.0)
        robot_hi = np.full((n, g, 3), 10.1)
        robot_lo[1, 0] = [0.70, -0.01, 1.12]
        robot_hi[1, 0] = [0.74, 0.01, 1.20]
        robot_lo[2, 0] = [0.70, -0.01, 1.12]
        robot_hi[2, 0] = [0.74, 0.01, 1.20]
        robot_lo[4, 0] = [0.70, -0.01, 1.12]
        robot_hi[4, 0] = [0.74, 0.01, 1.20]
        target_lo = np.full((n, 1, 3), 10.0)
        target_hi = np.full((n, 1, 3), 10.1)
        inbound = np.array([False, True, False, False, False, False])
        outbound = np.array([False, False, False, False, True, False])
        grasp = np.array([False, False, True, True, False, False])
        initial = np.array([True, False, False, False, False, False])
        aabb_report = evaluate_stock_necessity(
            fixture=fixture,
            robot_lo=robot_lo,
            robot_hi=robot_hi,
            target_lo=target_lo,
            target_hi=target_hi,
            inbound_mask=inbound,
            outbound_mask=outbound,
            grasp_mask=grasp,
            initial_mask=initial,
            min_grasp_clearance_m=0.025,
        )
        self.assertFalse(aabb_report["accepted"])
        self.assertTrue(aabb_report["necessary_both_traversals"])
        self.assertEqual(aabb_report["min_grasp_window_aabb_clearance_m"], 0.0)
        dual = filter_lattice_dual_transit(
            [fixture],
            [
                {
                    "robot_lo": robot_lo,
                    "robot_hi": robot_hi,
                    "target_lo": target_lo,
                    "target_hi": target_hi,
                    "inbound_mask": inbound,
                    "outbound_mask": outbound,
                    "grasp_mask": grasp,
                    "initial_mask": initial,
                }
            ],
        )
        certified = filter_lattice_for_cells(
            [fixture],
            [
                {
                    "robot_lo": robot_lo,
                    "robot_hi": robot_hi,
                    "target_lo": target_lo,
                    "target_hi": target_hi,
                    "inbound_mask": inbound,
                    "outbound_mask": outbound,
                    "grasp_mask": grasp,
                    "initial_mask": initial,
                }
            ],
            min_grasp_clearance_m=0.025,
        )
        self.assertEqual(len(dual), 1)
        self.assertEqual(len(certified), 0)

        box_type = int(mujoco.mjtGeom.mjGEOM_BOX)
        cell = {
            "robot_gtype": np.array([box_type], dtype=np.int32),
            "robot_size": np.array([[0.02, 0.02, 0.02]], dtype=np.float64),
            "robot_pos": np.zeros((n, 1, 3), dtype=np.float32),
            "robot_mat": np.tile(np.eye(3, dtype=np.float32).reshape(9), (n, 1, 1)),
            "robot_verts": [None],
            "robot_lo": robot_lo,
            "robot_hi": robot_hi,
            "target_gtype": np.array([box_type], dtype=np.int32),
            "target_size": np.array([[0.01, 0.01, 0.01]], dtype=np.float64),
            "target_pos": np.full((n, 1, 3), 10.0, dtype=np.float32),
            "target_mat": np.tile(np.eye(3, dtype=np.float32).reshape(9), (n, 1, 1)),
            "target_verts": [None],
            "target_lo": target_lo,
            "target_hi": target_hi,
            "inbound_mask": inbound,
            "outbound_mask": outbound,
            "grasp_mask": grasp,
            "initial_mask": initial,
        }
        cell["robot_pos"][1, 0] = [0.72, 0.0, 1.22]
        cell["robot_pos"][2, 0] = [0.72, 0.0, 0.70]
        cell["robot_pos"][4, 0] = [0.72, 0.0, 1.22]
        cell["robot_pos"][0, 0] = [10.0, 10.0, 10.0]
        exact_hit = evaluate_fixture_exact(fixture, [cell], min_clearance_m=0.025)
        self.assertTrue(exact_hit["inbound_stock_contact"])
        self.assertTrue(exact_hit["outbound_stock_contact"])
        self.assertTrue(exact_hit["grasp_window_clear"])
        self.assertTrue(exact_hit["accepted"])
        occupying = dict(cell)
        occupying["robot_pos"] = cell["robot_pos"].copy()
        occupying["robot_pos"][2, 0] = [0.72, 0.0, 1.22]
        occupying["robot_lo"] = robot_lo.copy()
        occupying["robot_hi"] = robot_hi.copy()
        blocked = evaluate_fixture_exact(fixture, [occupying], min_clearance_m=0.025)
        self.assertFalse(blocked["grasp_window_clear"])
        self.assertFalse(blocked["accepted"])
        self.assertEqual(len(blocked["per_cell"]), 1)
        self.assertIsNotNone(blocked["per_cell"][0]["grasp_witness"]["frame"])
        self.assertIsNotNone(blocked["per_cell"][0]["grasp_clearance_margin_m"])
        self.assertLess(blocked["per_cell"][0]["grasp_clearance_margin_m"], 0.0)

    def test_exact_scores_all_predicates_on_all_cells(self) -> None:
        import mujoco
        from pact_place_v99_exact import evaluate_fixture_exact

        fixture = build_pendant_fixture(
            center_x_m=0.72,
            center_y_m=0.0,
            half_x_m=0.05,
            half_y_m=0.08,
            bottom_z_m=1.10,
        )
        n, g = 6, 1
        box_type = int(mujoco.mjtGeom.mjGEOM_BOX)
        inbound = np.array([False, True, False, False, False, False])
        outbound = np.array([False, False, False, False, True, False])
        grasp = np.array([False, False, True, True, False, False])
        initial = np.array([True, False, False, False, False, False])

        def _cell(*, inbound_hit: bool, grasp_hit: bool, role: int) -> dict:
            robot_lo = np.full((n, g, 3), 10.0)
            robot_hi = np.full((n, g, 3), 10.1)
            robot_pos = np.zeros((n, 1, 3), dtype=np.float64)
            robot_pos[:, 0] = [10.0, 10.0, 10.0]
            robot_lo[4, 0] = [0.70, -0.01, 1.12]
            robot_hi[4, 0] = [0.74, 0.01, 1.20]
            robot_pos[4, 0] = [0.72, 0.0, 1.22]
            if inbound_hit:
                robot_lo[1, 0] = [0.70, -0.01, 1.12]
                robot_hi[1, 0] = [0.74, 0.01, 1.20]
                robot_pos[1, 0] = [0.72, 0.0, 1.22]
            if grasp_hit:
                robot_lo[2, 0] = [0.70, -0.01, 1.12]
                robot_hi[2, 0] = [0.74, 0.01, 1.20]
                robot_pos[2, 0] = [0.72, 0.0, 1.22]
            else:
                robot_pos[2, 0] = [0.72, 0.0, 0.70]
            return {
                "role_index": role,
                "robot_gtype": np.array([box_type], dtype=np.int32),
                "robot_size": np.array([[0.02, 0.02, 0.02]], dtype=np.float64),
                "robot_pos": robot_pos,
                "robot_mat": np.tile(np.eye(3, dtype=np.float64).reshape(9), (n, 1, 1)),
                "robot_verts": [None],
                "robot_lo": robot_lo,
                "robot_hi": robot_hi,
                "target_gtype": np.array([box_type], dtype=np.int32),
                "target_size": np.array([[0.01, 0.01, 0.01]], dtype=np.float64),
                "target_pos": np.full((n, 1, 3), 10.0, dtype=np.float64),
                "target_mat": np.tile(np.eye(3, dtype=np.float64).reshape(9), (n, 1, 1)),
                "target_verts": [None],
                "target_lo": np.full((n, 1, 3), 10.0),
                "target_hi": np.full((n, 1, 3), 10.1),
                "inbound_mask": inbound,
                "outbound_mask": outbound,
                "grasp_mask": grasp,
                "initial_mask": initial,
            }

        missed_inbound = _cell(inbound_hit=False, grasp_hit=False, role=600)
        occupied_grasp = _cell(inbound_hit=True, grasp_hit=True, role=601)
        report = evaluate_fixture_exact(
            fixture, [missed_inbound, occupied_grasp], min_clearance_m=0.025
        )
        self.assertEqual(len(report["per_cell"]), 2)
        self.assertFalse(report["per_cell"][0]["inbound_stock_contact"])
        self.assertTrue(report["per_cell"][0]["outbound_stock_contact"])
        self.assertTrue(report["per_cell"][0]["grasp_window_clear"])
        self.assertTrue(report["per_cell"][1]["inbound_stock_contact"])
        self.assertTrue(report["per_cell"][1]["outbound_stock_contact"])
        self.assertFalse(report["per_cell"][1]["grasp_window_clear"])
        self.assertFalse(report["inbound_stock_contact"])
        self.assertFalse(report["grasp_window_clear"])
        self.assertFalse(report["accepted"])

    def test_live_mj_forward_contact_parity_and_near_threshold(self) -> None:
        from pact_place_v99_exact import live_mj_forward_parity_cases

        report = live_mj_forward_parity_cases()
        self.assertTrue(report["nested_is_contact"])
        self.assertTrue(report["sphere_clearance_gt_25mm"])
        self.assertAlmostEqual(report["near_gap_m"], 0.025, places=5)
        self.assertLess(report["below_threshold"]["true_distance_m"], 0.025)
        self.assertTrue(report["parity_ok"])

    def test_convex_hull_gjk_matches_full_vertex_cloud(self) -> None:
        import mujoco
        from pact_geom_distance import GeomShape, convex_hull_vertices, gjk_distance

        full = np.vstack(
            [
                np.array([[ix, iy, iz] for ix in (-0.04, 0.04) for iy in (-0.03, 0.03) for iz in (-0.02, 0.02)], dtype=np.float64),
                np.array([[0.0, 0.0, 0.0], [0.01, -0.01, 0.005]], dtype=np.float64),
            ]
        )
        hull = convex_hull_vertices(full)
        self.assertLess(len(hull), len(full))
        box = GeomShape.posed_axis_aligned_box(
            np.array([0.20, 0.0, 0.0], dtype=np.float64),
            np.array([0.05, 0.05, 0.05], dtype=np.float64),
        )
        mesh_type = int(mujoco.mjtGeom.mjGEOM_MESH)
        eye = np.eye(3, dtype=np.float64)
        origin = np.zeros(3, dtype=np.float64)
        size = np.zeros(3, dtype=np.float64)
        full_d = gjk_distance(GeomShape(mesh_type, origin, eye, size, full), box)
        hull_d = gjk_distance(GeomShape(mesh_type, origin, eye, size, hull), box)
        self.assertAlmostEqual(full_d, hull_d, places=9)

    def test_lane_directions_detour_corners_and_densify(self) -> None:
        self.assertEqual(panel_lane_sign("left"), -1.0)
        self.assertEqual(panel_lane_sign("right"), 1.0)
        left = lane_y_grid("left")
        right = lane_y_grid("right")
        self.assertTrue(np.all(left < 0.0))
        self.assertTrue(np.all(right > 0.0))
        self.assertAlmostEqual(float(np.min(np.diff(left))), -0.01, places=6)
        positions = np.array(
            [[0.90, 0.02, 0.88], [0.50, 0.02, 0.88]], dtype=float
        )
        rotations = np.stack([np.eye(3), np.eye(3)])
        dense_p, dense_r = densify_path(positions, rotations)
        steps = np.linalg.norm(np.diff(dense_p, axis=0), axis=1)
        self.assertTrue(np.all(steps <= 0.005 + 1e-9))
        self.assertEqual(len(dense_p), len(dense_r))
        self.assertEqual(len(PERTURBATION_CORNERS), 8)
        fixture = build_pendant_fixture(
            center_x_m=0.72,
            center_y_m=0.0,
            half_x_m=0.05,
            half_y_m=0.08,
            bottom_z_m=1.10,
        )
        left_plan = plan_lane(
            positions,
            rotations,
            fixture=fixture,
            panel_side="left",
            lane_y_m=-0.12,
            padding_m=0.10,
        )
        inbound_plus_x = np.array(
            [[0.50, 0.02, 0.88], [0.90, 0.02, 0.88]], dtype=float
        )
        right_plan = plan_lane(
            inbound_plus_x,
            rotations,
            fixture=fixture,
            panel_side="right",
            lane_y_m=0.12,
            padding_m=0.10,
        )
        self.assertLess(left_plan["travel_sign"], 0.0)
        self.assertGreater(left_plan["entry_x_m"], left_plan["exit_x_m"])
        self.assertGreater(right_plan["travel_sign"], 0.0)
        self.assertLess(right_plan["entry_x_m"], right_plan["exit_x_m"])
        self.assertTrue(left_plan["detour"]["meets_minimum"])
        self.assertGreaterEqual(left_plan["detour"]["min_abs_detour_m"], MIN_DETOUR_M)
        self.assertFalse(left_plan["wrong_way"])
        self.assertFalse(left_plan["clipped"])
        self.assertEqual(len(left_plan["perturbation_corners"]), 8)
        too_small = plan_lane(
            positions,
            rotations,
            fixture=fixture,
            panel_side="left",
            lane_y_m=-0.02,
            padding_m=0.10,
        )
        self.assertFalse(too_small["detour"]["meets_minimum"])
        corners = perturbation_corners(-0.12, 0.67, 0.57)
        self.assertEqual(len(corners), 8)
        keys = {
            (round(item["delta_lane_y_m"], 6), round(item["delta_entry_x_m"], 6), round(item["delta_exit_x_m"], 6))
            for item in corners
        }
        self.assertEqual(len(keys), 8)

    def test_carried_target_and_all_robot_geoms_enter_intersection_mask(self) -> None:
        box_lo, box_hi = np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])
        robot_lo = np.array([[[-2.0, -2.0, -2.0], [0.2, 0.2, 0.2]]])
        robot_hi = np.array([[[-1.9, -1.9, -1.9], [0.3, 0.3, 0.3]]])
        target_lo = np.array([[[0.4, 0.4, 0.4]]])
        target_hi = np.array([[[0.5, 0.5, 0.5]]])
        self.assertTrue(bool(geoms_intersect_box(robot_lo, robot_hi, box_lo, box_hi)[0]))
        self.assertTrue(bool(geoms_intersect_box(target_lo, target_hi, box_lo, box_hi)[0]))
        self.assertTrue(aabb_overlap(robot_lo[0, 1], robot_hi[0, 1], box_lo, box_hi))
        self.assertGreater(
            aabb_separation_m(robot_lo[0, 0], robot_hi[0, 0], box_lo, box_hi), 0.0
        )

    def test_candidate_ranking_and_six_cell_gate(self) -> None:
        shared = {
            "worst_cell_changed_sensors": 5,
            "fixture": {"center_m": [0.7, 0.0, 1.3], "half_m": [0.05, 0.1, 0.2]},
        }
        a = {
            **shared,
            "key": (0.70, 0.0, 1.3, 0.05, 0.10, 0.20),
            "worst_cell_changed_value_fraction": 0.40,
            "min_robust_clearance_m": 0.021,
            "volume_m3": 0.004,
        }
        b = {
            **shared,
            "key": (0.80, 0.0, 1.3, 0.05, 0.08, 0.20),
            "worst_cell_changed_value_fraction": 0.30,
            "min_robust_clearance_m": 0.040,
            "volume_m3": 0.003,
        }
        selected = select_at_most_two_candidates([a, b])
        self.assertEqual([item["rank_role"] for item in selected], ["signal", "clearance"])
        self.assertEqual(selected[0]["key"], a["key"])
        self.assertEqual(selected[1]["key"], b["key"])
        same = select_at_most_two_candidates([a])
        self.assertEqual(len(same), 1)
        rows = []
        for family in PHYSICS_CLEAN_FAMILIES:
            for side in ("left", "right"):
                for repeat in range(4):
                    rows.append(
                        {
                            "family": family,
                            "intrusion_side": side,
                            "clean_success": not (
                                family == "F2_outer_panel_stagger"
                                and side == "right"
                                and repeat > 0
                            ),
                            "status": "complete",
                        }
                    )
        summary = six_cell_gate_summary(rows)
        self.assertEqual(summary["n_rows"], N_GATE_ROWS)
        self.assertGreaterEqual(summary["clean_successes"], MIN_GATE_CLEAN_SUCCESSES)
        self.assertTrue(summary["passed"])
        self.assertFalse(summary["authorizes_collection"])
        rows[0]["clean_success"] = False
        rows[1]["clean_success"] = False
        rows[2]["clean_success"] = False
        rows[3]["clean_success"] = False
        failed = six_cell_gate_summary(rows)
        self.assertIn("F0_target_side_stagger:left", failed["uncovered_cells"])
        self.assertFalse(failed["passed"])

    def test_canonical_grasp_posture_and_authorization_defaults(self) -> None:
        baseline = [0.1, -0.2, 0.3, -1.0, 0.0, 1.2, -0.4]
        close = [value + 0.0005 for value in baseline]
        far = list(baseline)
        far[3] += 0.002
        self.assertTrue(grasp_posture_preserved(baseline, close)["preserved"])
        self.assertFalse(grasp_posture_preserved(baseline, far)["preserved"])
        self.assertEqual(MAX_GRASP_JOINT_ERROR_RAD, 0.001)
        auth = empty_authorization()
        self.assertFalse(auth["authorizes_collection"])
        self.assertFalse(auth["authorizes_paired_screen"])
        self.assertFalse(auth["authorizes_24_row_gate"])
        self.assertEqual(auth["authorizes_s2b"], False)
        self.assertEqual(DEFAULT_SEED, 955339)
        self.assertEqual(ADMISSION_FLOOR["min_changed_values_per_role_side"], 448)
        self.assertEqual(tuple(CORRIDOR_LINKS), ("link5_front", "link5_back", "link6"))

    def test_sampler_policy_runner_and_contract_dispatch(self) -> None:
        from molmo_spaces.tasks.enclosure_reach import (
            PactPlaceCorridorPolicy,
            PactPlaceCorridorV98PendantSampler,
            PactPlaceCorridorV99PendantSampler,
        )
        from run_pact_place_expert_screen import _make_config

        self.assertEqual(
            PactPlaceCorridorV99PendantSampler.PACT_PLACE_ENVIRONMENT_VERSION,
            ENVIRONMENT_VERSION,
        )
        self.assertEqual(
            PactPlaceCorridorV99PendantSampler.__name__, SAMPLER_CLASS
        )
        self.assertNotEqual(
            PactPlaceCorridorV98PendantSampler.PACT_PLACE_ENVIRONMENT_VERSION,
            ENVIRONMENT_VERSION,
        )
        self.assertEqual(
            PactPlaceCorridorPolicy._mounted_fixture_roles(
                "pact_place_corridor_v9_9_pendant"
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
        policy = PactPlaceCorridorPolicy.__new__(PactPlaceCorridorPolicy)
        policy.task = type(
            "T",
            (),
            {"scene_params": {"pact_place_environment_version": ENVIRONMENT_VERSION}},
        )()
        self.assertTrue(policy._v9_enabled())
        self.assertTrue(policy._v99_enabled())
        v98 = PactPlaceCorridorPolicy.__new__(PactPlaceCorridorPolicy)
        v98.task = type(
            "T",
            (),
            {
                "scene_params": {
                    "pact_place_environment_version": "pact_place_corridor_v9_8_pendant"
                }
            },
        )()
        self.assertTrue(v98._v9_enabled())
        self.assertFalse(v98._v99_enabled())
        config = _make_config(
            ROOT / "diagnostics_output" / "pact_place_v99_dispatch_probe.json",
            scene_xml=ROOT / SCENE_XML_RELATIVE,
            sampler_class=SAMPLER_CLASS,
        )
        self.assertIs(
            config.task_sampler_config.task_sampler_class,
            PactPlaceCorridorV99PendantSampler,
        )
        v93 = _make_config(
            ROOT / "diagnostics_output" / "pact_place_v99_dispatch_probe.json",
            scene_xml=ROOT / SCENE_XML_RELATIVE,
            sampler_class="PactPlaceCorridorV93Sampler",
        )
        from molmo_spaces.tasks.enclosure_reach import PactPlaceCorridorV93Sampler

        self.assertIs(v93.task_sampler_config.task_sampler_class, PactPlaceCorridorV93Sampler)
        source = inspect.getsource(PactPlaceCorridorPolicy._compute_trajectory)
        self.assertIn("_v99_apply_lane", source)
        ik_source = inspect.getsource(PactPlaceCorridorPolicy._v99_sequential_ik_clearance)
        self.assertIn("kinematics.ik", ik_source)
        self.assertIn("seed", ik_source)
        sampler_source = inspect.getsource(PactPlaceCorridorV99PendantSampler)
        self.assertIn("pact_place_v99_pendant_contract", sampler_source)
        self.assertNotIn("pact_place_v98_pendant_contract", sampler_source)
        self.assertIn("inbound_pendant_approach", inspect.getsource(PactPlaceCorridorPolicy.get_all_phases))
        self.assertEqual(CONTRACT_VERSION, "pact_place_v9_9_pendant_v1")

    def test_aabb_siting_note_is_broadphase_not_occupation(self) -> None:
        from search_pact_place_v99_pendant import AABB_BROADPHASE_NOTE

        lowered = AABB_BROADPHASE_NOTE.lower()
        self.assertIn("broad-phase", lowered)
        self.assertNotIn("occupies the grasp", lowered)
        self.assertNotIn("one fixed pendant cannot", lowered)

    def test_closeout_scoped_conclusion_and_no_routing_stage(self) -> None:
        from pact_place_v99_exact import SCOPED_CONCLUSION
        from search_pact_place_v99_pendant import main as search_main

        self.assertEqual(
            SCOPED_CONCLUSION,
            "no survivor in the registered fixed rectangular-box lattice",
        )
        search_source = inspect.getsource(search_main)
        self.assertNotIn("--stage route", search_source)
        self.assertNotIn("run_pact_place_v99_v95_pendant_paired", search_source)
        self.assertIn("routing_run", search_source)
        self.assertIn("v99_closed", search_source)

    def test_interpolate_stock_y_and_travel_sign(self) -> None:
        path = np.array([[0.8, 0.02], [0.6, 0.02], [0.4, -0.01]], dtype=float)
        self.assertAlmostEqual(interpolate_y_at_x(path, 0.7)[0], 0.02)
        self.assertLess(travel_sign_through_slab(np.column_stack([path[:, 0], path[:, 1], np.zeros(3)]), 0.55, 0.75), 0.0)
        rewritten, rotations, _mask = apply_constant_lane(
            np.column_stack([path[:, 0], path[:, 1], np.zeros(3)]),
            np.stack([np.eye(3)] * 3),
            lane_y=-0.20,
            entry_x=0.75,
            exit_x=0.55,
        )
        self.assertTrue(np.allclose(rewritten[1, 1], -0.20))
        self.assertTrue(np.allclose(rotations[0], np.eye(3)))


if __name__ == "__main__":
    unittest.main()
