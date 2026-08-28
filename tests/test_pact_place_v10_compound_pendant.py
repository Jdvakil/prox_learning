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
from pact_place_v10_compound_pendant_contract import (
    ALL_GEOMS,
    ASSEMBLY_PARK_XYZ_M,
    CEILING_TOP_Z_M,
    CONTRACT_VERSION,
    ENVIRONMENT_VERSION,
    HOOD_TOP_BOTTOM_Z_M,
    MIN_DETOUR_M,
    PENDANT_BODY,
    PROBE_NEGATIVE_LOBE,
    PROBE_POSITIVE_LOBE,
    PROBE_STEM_Y_M,
    SAMPLER_CLASS,
    SCENE_XML_RELATIVE,
    STEM_TOP_Z_M,
    V5_SCENE_XML_RELATIVE,
    apply_human_environment_qualification,
    empty_authorization,
    reject_v98_kwargs,
)
from pact_place_v10_geometry import (
    NECESSITY_ALL_BITS,
    active_components,
    build_assembly,
    build_lobe,
    covers_all_necessity,
    enumerate_lobes,
    forbidden_static_overlap,
    lattice_raw_count,
    next_search_family,
    planning_probe_assembly,
    stream_covering_three_lobe_sets,
    stream_covering_two_lobe_pairs,
    stream_two_lobe_pairs,
    union_fixture,
    validate_lobe_geometry,
)
from pact_place_v10_route import (
    PERTURBATION_CORNERS,
    densify_path,
    named_lane_segments,
    panel_lane_sign,
    plan_lane,
    select_at_most_two_candidates,
)
from pact_place_v99_pendant_contract import ENVIRONMENT_VERSION as V99_ENVIRONMENT


def _lobe(*, cy: float, hy: float = 0.04, cx: float = 0.70, cz: float = 0.90, hx: float = 0.02, hz: float = 0.04):
    return build_lobe(
        center_x_m=cx,
        center_y_m=cy,
        center_z_m=cz,
        half_x_m=hx,
        half_y_m=hy,
        half_z_m=hz,
    )


class PactPlaceV10CompoundPendantTest(unittest.TestCase):
    def test_contract_excludes_v98_and_does_not_reopen_v99(self) -> None:
        source = Path(ROOT / "scripts" / "pact_place_v10_compound_pendant_contract.py").read_text()
        from_modules = [
            node.module
            for node in ast.parse(source).body
            if isinstance(node, ast.ImportFrom)
        ]
        self.assertNotIn("pact_place_v98_pendant_contract", from_modules)
        with self.assertRaises(ValueError) as raised:
            reject_v98_kwargs({"wrist_lag_neg_m": 0.208})
        self.assertIn("V9.8 lag/window", str(raised.exception))
        self.assertEqual(CONTRACT_VERSION, "pact_place_v10_compound_pendant_v1")
        self.assertEqual(ENVIRONMENT_VERSION, "pact_place_corridor_v10_compound_pendant")
        self.assertNotEqual(ENVIRONMENT_VERSION, V99_ENVIRONMENT)
        self.assertEqual(SAMPLER_CLASS, "PactPlaceCorridorV10CompoundPendantSampler")

    def test_lattice_counts_bounds_and_probe_fixture(self) -> None:
        self.assertEqual(lattice_raw_count(), 16 * 3 * 20 * 4 * 7 * 4)
        with self.assertRaises(ValueError):
            validate_lobe_geometry((0.60, -0.26, 0.90), (0.03, 0.02, 0.04))
        probe = planning_probe_assembly()
        self.assertEqual(probe["topology"], "two_lobe")
        self.assertTrue(probe["identical_on_both_panel_sides"])
        self.assertEqual(probe["active_on"], ["inbound_empty", "outbound_loaded"])
        lobes = [item for item in probe["components"] if item["role"] == "lobe" and item["active"]]
        stems = [item for item in probe["components"] if item["role"] == "stem" and item["active"]]
        unused = [item for item in probe["components"] if not item["active"]]
        self.assertEqual(len(lobes), 2)
        self.assertEqual(len(stems), 2)
        self.assertEqual(len(unused), 2)
        self.assertEqual(
            [tuple(item["center_m"]) for item in lobes],
            [PROBE_NEGATIVE_LOBE["center_m"], PROBE_POSITIVE_LOBE["center_m"]],
        )
        stem_ys = tuple(sorted(float(item["center_m"][1]) for item in stems))
        self.assertEqual(stem_ys, PROBE_STEM_Y_M)
        bar = next(item for item in probe["components"] if item["role"] == "crossbar")
        self.assertAlmostEqual(bar["center_m"][2] + bar["half_m"][2], HOOD_TOP_BOTTOM_Z_M)
        self.assertAlmostEqual(STEM_TOP_Z_M, 1.505)
        self.assertAlmostEqual(CEILING_TOP_Z_M, 1.515)
        enumerated = enumerate_lobes()
        self.assertLess(len(enumerated), lattice_raw_count())
        probe_keys = {tuple(item["key"]) for item in lobes}
        have = {tuple(item["key"]) for item in enumerated}
        self.assertTrue(probe_keys <= have)

    def test_connectivity_and_forbidden_overlap(self) -> None:
        assembly = planning_probe_assembly()
        stems = [item for item in active_components(assembly) if item["role"] == "stem"]
        bar = next(item for item in assembly["components"] if item["role"] == "crossbar")
        for stem in stems:
            self.assertAlmostEqual(stem["center_m"][2] + stem["half_m"][2], bar["center_m"][2] - bar["half_m"][2])
        for item in active_components(assembly):
            allow = item["role"] == "crossbar"
            self.assertFalse(forbidden_static_overlap(item, allow_hood_top=allow))
        hood_intruder = {
            "center_m": [0.95, 0.0, 1.53],
            "half_m": [0.05, 0.05, 0.05],
        }
        self.assertTrue(forbidden_static_overlap(hood_intruder, allow_hood_top=True))

    def test_deterministic_set_cover_and_two_to_three_escalation(self) -> None:
        negative = _lobe(cy=-0.20)
        positive = _lobe(cy=0.20)
        extra = _lobe(cy=0.28, hy=0.02)
        pairs = list(stream_two_lobe_pairs([negative, positive, extra]))
        self.assertEqual(len(pairs), 2)
        bits = {
            tuple(negative["key"]): 0x0F00,
            tuple(positive["key"]): 0x00FF,
            tuple(extra["key"]): 0x000F,
        }
        covering = list(stream_covering_two_lobe_pairs([negative, positive, extra], bits))
        self.assertEqual(len(covering), 1)
        self.assertEqual(covering[0][0]["key"], negative["key"])
        self.assertEqual(covering[0][1]["key"], positive["key"])
        self.assertTrue(covers_all_necessity([0x0F00, 0x00FF]))
        self.assertFalse(covers_all_necessity([0x0F00, 0x000F]))
        self.assertIsNone(
            next_search_family(two_lobe_exact_survivors=["kept"], two_lobe_failed_later=False)
        )
        self.assertIsNone(
            next_search_family(two_lobe_exact_survivors=[], two_lobe_failed_later=True)
        )
        self.assertEqual(
            next_search_family(two_lobe_exact_survivors=[], two_lobe_failed_later=False),
            "three_lobe",
        )
        three_bits = {
            tuple(negative["key"]): 0x0F00,
            tuple(positive["key"]): 0x00F0,
            tuple(extra["key"]): 0x000F,
        }
        triples = list(
            stream_covering_three_lobe_sets([negative, positive, extra], three_bits)
        )
        self.assertEqual(len(triples), 1)

    def test_lobe_only_necessity_and_union_clearance(self) -> None:
        import mujoco
        from pact_place_v10_exact import evaluate_assembly_exact, necessity_bit

        assembly = planning_probe_assembly()
        n, g = 6, 1
        box_type = int(mujoco.mjtGeom.mjGEOM_BOX)
        robot_pos = np.full((n, g, 3), 10.0, dtype=np.float64)
        robot_lo = np.full((n, g, 3), 9.9)
        robot_hi = np.full((n, g, 3), 10.1)
        inbound = np.array([False, True, False, False, False, False])
        outbound = np.array([False, False, False, False, True, False])
        grasp = np.array([False, False, True, True, False, False])
        initial = np.array([True, False, False, False, False, False])
        lobe = next(item for item in assembly["components"] if item["role"] == "lobe" and item["active"])
        stem = next(item for item in assembly["components"] if item["role"] == "stem" and item["active"])
        robot_pos[1, 0] = lobe["center_m"]
        robot_lo[1, 0] = np.asarray(lobe["center_m"]) - 0.001
        robot_hi[1, 0] = np.asarray(lobe["center_m"]) + 0.001
        robot_pos[4, 0] = stem["center_m"]
        robot_lo[4, 0] = np.asarray(stem["center_m"]) - 0.001
        robot_hi[4, 0] = np.asarray(stem["center_m"]) + 0.001
        robot_pos[2, 0] = [0.70, 0.0, 0.50]
        cell = {
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
        report = evaluate_assembly_exact(assembly, [cell])
        self.assertFalse(report["stem_or_crossbar_counted_as_necessity"])
        stem_row = next(item for item in report["per_component"] if item["role"] == "stem")
        lobe_row = next(item for item in report["per_component"] if item["role"] == "lobe")
        self.assertTrue(stem_row["per_cell"][0]["outbound_contact"])
        self.assertFalse(bool(report["lobe_necessity_bits"] & necessity_bit(0, True)))
        self.assertTrue(lobe_row["per_cell"][0]["inbound_contact"])
        self.assertTrue(bool(report["lobe_necessity_bits"] & necessity_bit(0, False)))
        from pact_place_v10_exact import (
            component_clearance_summary,
            evaluate_assembly_from_component_caches,
            lobe_necessity_bits,
        )

        lobe_scores = {}
        stem_scores = {}
        lobes = [item for item in active_components(assembly) if item["role"] == "lobe"]
        stems = [item for item in active_components(assembly) if item["role"] == "stem"]
        for lobe in lobes:
            scored = lobe_necessity_bits(lobe, [cell])
            lobe_scores[tuple(lobe["key"])] = {
                "bits": scored["bits"],
                "grasp_clear_all": scored["grasp_clear_all"],
                "initial_clear_all": scored["initial_clear_all"],
                "min_grasp_clearance_margin_m": scored["min_grasp_clearance_margin_m"],
            }
        for stem in stems:
            parent = next(item for item in lobes if item.get("slot") == stem.get("slot"))
            stem_scores[tuple(parent["key"])] = component_clearance_summary(stem, [cell])
        bar = next(item for item in assembly["components"] if item["role"] == "crossbar")
        cached = evaluate_assembly_from_component_caches(
            assembly,
            lobe_scores=lobe_scores,
            stem_scores=stem_scores,
            crossbar_score=component_clearance_summary(bar, [cell]),
        )
        self.assertEqual(cached["accepted"], report["accepted"])
        self.assertEqual(cached["lobe_necessity_bits"], report["lobe_necessity_bits"])
        self.assertEqual(cached["grasp_window_clear"], report["grasp_window_clear"])
        union = union_fixture(assembly)
        for item in active_components(assembly):
            for axis in range(3):
                self.assertLessEqual(
                    union["center_m"][axis] - union["half_m"][axis],
                    item["center_m"][axis] - item["half_m"][axis] + 1e-9,
                )
                self.assertGreaterEqual(
                    union["center_m"][axis] + union["half_m"][axis],
                    item["center_m"][axis] + item["half_m"][axis] - 1e-9,
                )

    def test_candidate_ranking_collapses_identical_assemblies(self) -> None:
        signal = {
            "assembly_id": "a",
            "worst_cell_changed_value_fraction": 0.90,
            "worst_cell_changed_sensors": 12,
            "min_robust_clearance_m": 0.030,
            "volume_m3": 0.010,
        }
        clearance = {
            "assembly_id": "b",
            "worst_cell_changed_value_fraction": 0.40,
            "worst_cell_changed_sensors": 6,
            "min_robust_clearance_m": 0.080,
            "volume_m3": 0.002,
        }
        selected = select_at_most_two_candidates([signal, clearance])
        self.assertEqual([item["assembly_id"] for item in selected], ["a", "b"])
        self.assertEqual(selected[0]["rank_role"], "signal")
        self.assertEqual(selected[1]["rank_role"], "clearance")
        same = select_at_most_two_candidates([signal, dict(signal, min_robust_clearance_m=0.09)])
        self.assertEqual(len(same), 1)
        self.assertEqual(same[0]["assembly_id"], "a")

    def test_authorization_transitions_never_open_collection(self) -> None:
        auth = empty_authorization()
        for key in (
            "authorizes_collection",
            "authorizes_training",
            "authorizes_eval",
            "environment_qualified",
            "authorizes_paired_screen",
        ):
            self.assertFalse(auth[key])
        qualified = apply_human_environment_qualification({"keep": 1}, approved=True)
        self.assertTrue(qualified["environment_qualified"])
        self.assertFalse(qualified["authorizes_collection"])
        self.assertFalse(qualified["authorizes_training"])
        self.assertFalse(qualified["authorizes_eval"])
        self.assertEqual(qualified["keep"], 1)

    def test_union_route_preserves_lane_contract(self) -> None:
        self.assertEqual(panel_lane_sign("left"), -1.0)
        self.assertEqual(panel_lane_sign("right"), 1.0)
        assembly = planning_probe_assembly()
        positions = np.array([[0.90, 0.02, 0.88], [0.40, 0.02, 0.88]], dtype=float)
        rotations = np.stack([np.eye(3), np.eye(3)])
        dense_p, dense_r = densify_path(positions, rotations)
        self.assertTrue(np.all(np.linalg.norm(np.diff(dense_p, axis=0), axis=1) <= 0.005 + 1e-9))
        left_plan = plan_lane(
            positions,
            rotations,
            assembly=assembly,
            panel_side="left",
            lane_y_m=-0.12,
            padding_m=0.10,
        )
        self.assertLess(left_plan["travel_sign"], 0.0)
        self.assertTrue(left_plan["detour"]["meets_minimum"])
        self.assertGreaterEqual(left_plan["detour"]["min_abs_detour_m"], MIN_DETOUR_M)
        self.assertFalse(left_plan["clipped"])
        self.assertFalse(left_plan["wrong_way"])
        self.assertEqual(len(left_plan["perturbation_corners"]), 8)
        self.assertEqual(len(PERTURBATION_CORNERS), 8)
        rebuilt = named_lane_segments(
            left_plan["planned_positions_m"],
            left_plan["planned_rotations"],
            prefix="inbound_pendant",
            entry_x=float(left_plan["entry_x_m"]),
            exit_x=float(left_plan["exit_x_m"]),
            stock_end=positions[-1],
        )
        self.assertTrue(any("inbound_pendant" in str(item.get("name", "")) for item in rebuilt))
        self.assertTrue(np.allclose(left_plan["planned_positions_m"][-1], positions[-1]))

    def test_scene_compile_active_inactive_and_parked(self) -> None:
        import mujoco
        from pact_place_v10_scene import pose_assembly_on_data

        scene = ROOT / SCENE_XML_RELATIVE
        self.assertTrue(scene.is_file())
        v5 = ROOT / V5_SCENE_XML_RELATIVE
        self.assertEqual(sha256_file(v5), PLACE_V5_SCENE_SHA256)
        model = mujoco.MjModel.from_xml_path(str(scene))
        data = mujoco.MjData(model)
        assembly = planning_probe_assembly()
        pose_assembly_on_data(model, data, assembly, parked=False)
        mujoco.mj_forward(model, data)
        body = int(model.body(PENDANT_BODY).id)
        self.assertTrue(np.allclose(data.xpos[body], 0.0, atol=1e-8))
        active = {item["geom"]: item for item in active_components(assembly)}
        for name in ALL_GEOMS:
            gid = int(model.geom(name).id)
            if name in active:
                self.assertEqual(int(model.geom_contype[gid]), 8)
                self.assertGreater(float(model.geom_rgba[gid][3]), 0.5)
                self.assertTrue(np.allclose(model.geom_pos[gid], active[name]["center_m"]))
                self.assertTrue(np.allclose(data.geom_xpos[gid], active[name]["center_m"], atol=1e-7))
            else:
                self.assertEqual(int(model.geom_contype[gid]), 0)
                self.assertEqual(int(model.geom_conaffinity[gid]), 0)
                self.assertEqual(float(model.geom_rgba[gid][3]), 0.0)
        pose_assembly_on_data(model, data, assembly, parked=True)
        mujoco.mj_forward(model, data)
        self.assertTrue(np.allclose(data.xpos[body], ASSEMBLY_PARK_XYZ_M, atol=1e-7))
        for name in ALL_GEOMS:
            gid = int(model.geom(name).id)
            self.assertEqual(int(model.geom_contype[gid]), 0)
            self.assertEqual(float(model.geom_rgba[gid][3]), 0.0)
        self.assertEqual(len(ALL_GEOMS), 7)

    def test_gjk_true_distance_parity_per_component_role(self) -> None:
        from pact_place_v10_exact import live_component_role_parity_cases
        from pact_place_v99_exact import NEAR_THRESHOLD_M, live_mj_forward_parity_cases

        shared = live_mj_forward_parity_cases()
        self.assertTrue(shared["parity_ok"])
        report = live_component_role_parity_cases()
        self.assertTrue(report["parity_ok"])
        for role in ("lobe", "stem", "crossbar"):
            self.assertTrue(report["roles"][role]["parity_ok"])
            self.assertAlmostEqual(
                report["roles"][role]["near"]["true_distance_m"],
                NEAR_THRESHOLD_M,
                places=4,
            )

    def test_aabb_overlap_is_not_exact_contact(self) -> None:
        import mujoco
        from pact_geom_distance import GeomShape, gjk_distance
        from pact_place_v99_geometry import aabb_overlap

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
        self.assertGreater(gjk_distance(cube, sphere), 0.025)

    def test_sampler_policy_dispatch_and_parked_control(self) -> None:
        from molmo_spaces.tasks.enclosure_reach import (
            PactPlaceCorridorPolicy,
            PactPlaceCorridorV10CompoundPendantSampler,
            PactPlaceCorridorV99PendantSampler,
        )
        from run_pact_place_expert_screen import _make_config

        self.assertEqual(
            PactPlaceCorridorV10CompoundPendantSampler.PACT_PLACE_ENVIRONMENT_VERSION,
            ENVIRONMENT_VERSION,
        )
        self.assertEqual(
            PactPlaceCorridorV10CompoundPendantSampler.__name__, SAMPLER_CLASS
        )
        self.assertNotEqual(
            PactPlaceCorridorV99PendantSampler.PACT_PLACE_ENVIRONMENT_VERSION,
            ENVIRONMENT_VERSION,
        )
        self.assertEqual(
            PactPlaceCorridorPolicy._mounted_fixture_roles(ENVIRONMENT_VERSION),
            (),
        )
        policy = PactPlaceCorridorPolicy.__new__(PactPlaceCorridorPolicy)
        policy.task = type(
            "T",
            (),
            {"scene_params": {"pact_place_environment_version": ENVIRONMENT_VERSION}},
        )()
        self.assertTrue(policy._v9_enabled())
        self.assertTrue(policy._v10_enabled())
        self.assertFalse(policy._v99_enabled())
        config = _make_config(
            ROOT / "diagnostics_output" / "pact_place_v10_dispatch_probe.json",
            scene_xml=ROOT / SCENE_XML_RELATIVE,
            sampler_class=SAMPLER_CLASS,
        )
        self.assertIs(
            config.task_sampler_config.task_sampler_class,
            PactPlaceCorridorV10CompoundPendantSampler,
        )
        source = inspect.getsource(PactPlaceCorridorPolicy._compute_trajectory)
        self.assertIn("_v10_apply_lane", source)
        self.assertIn("_v99_apply_lane", source)
        ik_source = inspect.getsource(PactPlaceCorridorPolicy._v10_sequential_ik_clearance)
        self.assertIn("kinematics.ik", ik_source)
        self.assertIn("seed", ik_source)
        self.assertIn("_v10_active_pendant_geom_ids", ik_source)
        self.assertIn("_v10_strict_environment_geom_ids", ik_source)
        self.assertIn("sequential_ik_split_clearance", ik_source)
        self.assertNotIn(
            "obstacle_ids = self._v10_active_pendant_geom_ids() + self._v10_strict_environment_geom_ids()",
            ik_source,
        )
        apply_source = inspect.getsource(PactPlaceCorridorPolicy._v10_apply_lane)
        self.assertIn("_v10_evaluate_nominal_and_robust", apply_source)
        sampler_source = inspect.getsource(PactPlaceCorridorV10CompoundPendantSampler)
        self.assertIn("pact_place_v10", sampler_source)
        self.assertNotIn("pact_place_v98_pendant_contract", sampler_source)
        self.assertIn("pact_v10_pendant_parked", sampler_source)
        info_source = inspect.getsource(PactPlaceCorridorPolicy.get_info)
        self.assertIn("pendant_v10", info_source)

    def test_search_runner_is_hash_stable_and_stops_before_route(self) -> None:
        from pact_place_v10_runtime import canonicalize

        text = Path(ROOT / "scripts/search_pact_place_v10_compound_pendant.py").read_text()
        self.assertIn("--stage", text)
        self.assertIn("--workers", text)
        self.assertIn("planning_probe", text)
        self.assertIn("no_exact_compound_survivor", text)
        self.assertIn("establish_v10_runtime_env", text)
        self.assertIn("score_lattice_parallel", text)
        self.assertIn("exact_survivors.npz", text)
        self.assertNotIn("env.step", text)
        self.assertIn("refusing to overwrite the superseded V10 siting v1", text)
        v2 = Path(ROOT / "scripts/search_pact_place_v10_siting_v2.py").read_text()
        self.assertIn("SITING_SCHEMA_V2", v2)
        self.assertIn("V2_SITING_RELATIVE", v2)
        self.assertNotIn("--stage route", v2)
        first = canonicalize({"survivors": [{"assembly_id": "b"}, {"assembly_id": "a"}]})
        second = canonicalize({"survivors": [{"assembly_id": "a"}, {"assembly_id": "b"}]})
        self.assertEqual(first, second)

    def test_planning_probe_reproduces_retained_qpos_result(self) -> None:
        snapshot = ROOT / "diagnostics_output/pact_place_v99_siting/snapshots/snapshots.json"
        if not snapshot.is_file():
            self.skipTest("V9.9 snapshots are not present")
        from pact_place_v10_exact import evaluate_planning_probe, verify_v99_inputs

        _reconstruction, _document, cells = verify_v99_inputs()
        report = evaluate_planning_probe(cells)
        self.assertEqual(report["lobe_necessity_bits"], NECESSITY_ALL_BITS)
        self.assertTrue(report["lobe_necessity_ok"])
        self.assertTrue(report["grasp_window_clear"])
        self.assertTrue(report["reproduced_probe"])
        self.assertTrue(report["initial_state_clear"])
        self.assertEqual(report["assembly"]["probe_label"], "probe_v2")
        self.assertFalse(report["stem_or_crossbar_counted_as_necessity"])


if __name__ == "__main__":
    unittest.main()
