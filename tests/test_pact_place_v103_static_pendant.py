"""Step-0A behavioral tests for the V10.3 static-pendant joint-route search.

Everything here is recomputed from source, live MuJoCo state, or the immutable
V10.2 payload. No test asserts a stored result boolean.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "scripts", ROOT / "submodules" / "molmospaces"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from pact_place_v10_compound_pendant_contract import (  # noqa: E402
    HOOD_TOP_BOTTOM_Z_M,
    STEM_TOP_Z_M,
)
from pact_place_v10_geometry import (  # noqa: E402
    forbidden_static_overlap,
    planning_probe_assembly,
)
from pact_place_v10_route import resolve_v10_runtime_route  # noqa: E402
from pact_place_v102_geometry import (  # noqa: E402
    RAISED_STEM_Y_M,
    SHELF_TOP_Z_M,
    STEM_HALF_V102_M,
    planning_probe_v102_raised_assembly,
)
from pact_place_v102_route import resolve_v102_runtime_route  # noqa: E402
from pact_place_v103_contract import (  # noqa: E402
    CONTRACT_VERSION,
    MIN_ROW_CLEARANCE_M,
    V102_ITEM5_COMPLETE_CASES,
    V102_ITEM6_CASES_MEETING_FLOOR,
    empty_authorization,
    implementation_sha256,
    search_lattice,
    v102_item5_case_table,
    v102_item5_complete_count,
    v102_item6_cases_meeting_floor,
    verify_protected_artifacts,
)
from pact_place_v103_geometry import (  # noqa: E402
    ALL_GEOMS_V103,
    ENVIRONMENT_VERSION_V103,
    FORBIDDEN_HEIGHTS_M,
    HEIGHT_LATTICE_M,
    MIN_SHELF_SEPARATION_M,
    PENDANT_BODY_V103,
    assembly_expectations,
    build_v103_assembly,
    enumerate_v103_assemblies,
    scene_xml_text,
    validate_height,
)
from pact_place_v103_joint_route import (  # noqa: E402
    CONTROL_POSES_INBOUND,
    CONTROL_POSES_OUTBOUND,
    CORNER_MIN_CLEARANCE_M,
    DEDUP_LINF_RAD,
    EDGE_MIN_CLEARANCE_M,
    LANE_MAGNITUDES_M,
    MAX_JOINT_STEP_RAD,
    NODE_MIN_CLEARANCE_M,
    N_ARM_JOINTS,
    N_HALTON_SEEDS,
    PASS_Z_OFFSETS_M,
    STAGING_BUFFERS_M,
    build_control_poses,
    dedup_joint_solutions,
    enumerate_templates,
    halton_joint_seeds,
    qpos_sequence_sha256,
    segment_duration_s,
    segment_speed_class,
    select_path,
    stock_pose_at_x,
    template_corners,
)

V103_XML = ROOT / "submodules/molmospaces/molmo_spaces/data_generation/custom_scenes/pact_place_corridor_v10_3.xml"


class V102ErrataTest(unittest.TestCase):
    def test_item5_counts_five_complete_cases(self) -> None:
        cases = v102_item5_case_table()
        self.assertEqual(len(cases), 12)
        self.assertEqual(v102_item5_complete_count(), 5)
        self.assertEqual(V102_ITEM5_COMPLETE_CASES, 5)
        inbound = [c for c in cases if c["direction"] == "inbound"]
        outbound = [c for c in cases if c["direction"] == "outbound"]
        self.assertEqual(sum(1 for c in inbound if c["complete_sequential_ik"]), 5)
        self.assertEqual(sum(1 for c in outbound if c["complete_sequential_ik"]), 0)
        f1_left = next(
            c
            for c in inbound
            if c["family"].startswith("F1") and c["intrusion_side"] == "left"
        )
        self.assertEqual(
            (f1_left["waypoints_solved"], f1_left["waypoints_attempted"]), (163, 165)
        )

    def test_item6_met_the_floor_nowhere(self) -> None:
        self.assertEqual(v102_item6_cases_meeting_floor(), 0)
        self.assertEqual(V102_ITEM6_CASES_MEETING_FLOOR, 0)

    def test_protected_history_is_unchanged(self) -> None:
        observed = verify_protected_artifacts()
        self.assertGreaterEqual(len(observed), 19)

    def test_historical_route_dispatch_unchanged(self) -> None:
        historical = resolve_v10_runtime_route({})
        self.assertEqual(historical["rewrite_primitive"], "contiguous_group_freeze")
        self.assertFalse(historical["use_endpoint_only"])
        self.assertIsNone(resolve_v102_runtime_route({}, {}))
        v101 = resolve_v10_runtime_route(
            {
                "rewrite_primitive": "endpoint_only",
                "qualification_mode": "empirical_live_contact_v1",
            }
        )
        self.assertTrue(v101["skip_offline_strict_environment"])

    def test_v101_and_v102_geometry_unchanged(self) -> None:
        v101 = planning_probe_assembly()
        self.assertEqual(v101.get("probe_label"), "probe_v2")
        v102 = planning_probe_v102_raised_assembly()
        self.assertEqual(v102.get("probe_label"), "probe_v102_raised")
        lobes = [i for i in v102["components"] if i["role"] == "lobe" and i["active"]]
        for lobe in lobes:
            self.assertAlmostEqual(float(lobe["center_m"][2]), 1.14, places=9)


class V103GeometryTest(unittest.TestCase):
    def test_height_lattice_is_the_registered_one(self) -> None:
        self.assertEqual(HEIGHT_LATTICE_M, (0.92, 0.96, 1.00, 1.04))
        for value in HEIGHT_LATTICE_M:
            self.assertEqual(validate_height(value), value)
            self.assertGreaterEqual(value - SHELF_TOP_Z_M, MIN_SHELF_SEPARATION_M - 1e-9)
        for bad in FORBIDDEN_HEIGHTS_M:
            with self.assertRaises(ValueError):
                validate_height(bad)
        for bad in (0.90, 0.94, 1.02, 1.06):
            with self.assertRaises(ValueError):
                validate_height(bad)

    def test_shape_is_inherited_from_v102_and_only_z_moves(self) -> None:
        v102 = planning_probe_v102_raised_assembly()
        v102_by_name = {i["name"]: i for i in v102["components"] if i["active"]}
        for assembly in enumerate_v103_assemblies():
            by_name = {i["name"]: i for i in assembly["components"]}
            for name in ("lobe_0", "lobe_1"):
                self.assertEqual(
                    list(by_name[name]["center_m"][:2]),
                    list(v102_by_name[name]["center_m"][:2]),
                )
                self.assertEqual(
                    list(by_name[name]["half_m"]), list(v102_by_name[name]["half_m"])
                )
            for name in ("stem_0", "stem_1", "crossbar"):
                self.assertAlmostEqual(
                    float(by_name[name]["half_m"][0]), STEM_HALF_V102_M, places=12
                )
            stems = sorted(
                (i for i in assembly["components"] if i["role"] == "stem"),
                key=lambda i: float(i["center_m"][1]),
            )
            self.assertEqual(
                [round(float(i["center_m"][1]), 9) for i in stems],
                list(RAISED_STEM_Y_M),
            )

    def test_positive_lobe_bottom_is_twenty_millimetres_higher(self) -> None:
        for assembly in enumerate_v103_assemblies():
            report = assembly_expectations(assembly)
            self.assertAlmostEqual(
                report["positive_lobe_bottom_z_m"]
                - report["negative_lobe_bottom_z_m"],
                0.02,
                places=9,
            )
            self.assertAlmostEqual(
                report["lowest_pendant_z_m"],
                float(assembly["lowest_lobe_bottom_z_m"]),
                places=9,
            )

    def test_stems_reach_the_crossbar_and_the_crossbar_reaches_the_hood(self) -> None:
        for assembly in enumerate_v103_assemblies():
            report = assembly_expectations(assembly)
            self.assertEqual(report["stem_square_m"], [0.012, 0.012])
            self.assertAlmostEqual(report["crossbar_top_z_m"], HOOD_TOP_BOTTOM_Z_M)
            by_name = {i["name"]: i for i in assembly["components"]}
            for slot in (0, 1):
                lobe = by_name[f"lobe_{slot}"]
                stem = by_name[f"stem_{slot}"]
                lobe_top = float(lobe["center_m"][2]) + float(lobe["half_m"][2])
                stem_bottom = float(stem["center_m"][2]) - float(stem["half_m"][2])
                stem_top = float(stem["center_m"][2]) + float(stem["half_m"][2])
                self.assertAlmostEqual(stem_bottom, lobe_top, places=9)
                self.assertAlmostEqual(stem_top, STEM_TOP_Z_M, places=9)

    def test_no_component_overlaps_forbidden_static_geometry(self) -> None:
        for assembly in enumerate_v103_assemblies():
            for item in assembly["components"]:
                self.assertFalse(
                    forbidden_static_overlap(
                        item, allow_hood_top=item["role"] == "crossbar"
                    ),
                    f"{assembly['assembly_id']}:{item['name']}",
                )

    def test_assembly_declares_itself_static(self) -> None:
        for assembly in enumerate_v103_assemblies():
            self.assertTrue(assembly["static"])
            self.assertFalse(assembly["has_joint"])
            self.assertFalse(assembly["is_mocap"])
            self.assertFalse(assembly["runtime_repositioned"])
            self.assertFalse(assembly["visual_only_sleeve"])
            self.assertTrue(assembly["collision_and_visible_stem_identical"])


class V103SceneTest(unittest.TestCase):
    """The scene text is the compiled artifact; nothing repositions it."""

    def setUp(self) -> None:
        self.assembly = build_v103_assembly(1.00)
        self.text = scene_xml_text(self.assembly)

    def test_body_carries_no_joint_freejoint_or_mocap(self) -> None:
        self.assertIn(f'<body name="{PENDANT_BODY_V103}" pos="0 0 0">', self.text)
        body = self.text.split(f'<body name="{PENDANT_BODY_V103}"')[1].split("</body>")[0]
        for token in ("<joint", "<freejoint", 'mocap="true"'):
            self.assertNotIn(token, body)

    def test_every_component_is_compiled_at_its_final_pose_and_size(self) -> None:
        for item in self.assembly["components"]:
            self.assertIn(f'name="{item["geom"]}"', self.text)
            pos = " ".join(f"{float(v):.9g}" for v in item["center_m"])
            size = " ".join(f"{float(v):.9g}" for v in item["half_m"])
            self.assertIn(f'pos="{pos}"', self.text)
            self.assertIn(f'size="{size}"', self.text)
        self.assertEqual(len(ALL_GEOMS_V103), 5)

    def test_compiles_and_visible_equals_collision(self) -> None:
        import mujoco

        try:
            model = mujoco.MjModel.from_xml_string(
                self.text,
                {
                    "pact_place_corridor_v5.xml": (
                        ROOT
                        / "submodules/molmospaces/molmo_spaces/data_generation/"
                        "custom_scenes/pact_place_corridor_v5.xml"
                    ).read_bytes(),
                    "pact_place_corridor_v3.xml": (
                        ROOT
                        / "submodules/molmospaces/molmo_spaces/data_generation/"
                        "custom_scenes/pact_place_corridor_v3.xml"
                    ).read_bytes(),
                },
            )
        except Exception as error:  # pragma: no cover - asset-dependent
            self.skipTest(f"standalone compile unavailable: {error}")
        by_name = {i["geom"]: i for i in self.assembly["components"]}
        names = [model.geom(i).name or "" for i in range(int(model.ngeom))]
        for geom, item in by_name.items():
            self.assertEqual(names.count(geom), 1, geom)
            gid = int(model.geom(geom).id)
            self.assertTrue(
                np.allclose(model.geom_size[gid], item["half_m"], atol=1e-12)
            )
            self.assertEqual(int(model.geom_contype[gid]), 8)
            self.assertEqual(int(model.geom_conaffinity[gid]), 15)
            self.assertAlmostEqual(float(model.geom_rgba[gid][3]), 1.0)
            body = str(model.body(int(model.geom_bodyid[gid])).name or "")
            self.assertEqual(body, PENDANT_BODY_V103)
            self.assertEqual(int(model.body_dofnum[int(model.body(body).id)]), 0)
            self.assertLess(int(model.body_mocapid[int(model.body(body).id)]), 0)

    def test_pendant_cannot_move_under_repeated_forward(self) -> None:
        import mujoco

        try:
            model = mujoco.MjModel.from_xml_string(
                self.text,
                {
                    "pact_place_corridor_v5.xml": (
                        ROOT
                        / "submodules/molmospaces/molmo_spaces/data_generation/"
                        "custom_scenes/pact_place_corridor_v5.xml"
                    ).read_bytes(),
                    "pact_place_corridor_v3.xml": (
                        ROOT
                        / "submodules/molmospaces/molmo_spaces/data_generation/"
                        "custom_scenes/pact_place_corridor_v3.xml"
                    ).read_bytes(),
                },
            )
        except Exception as error:  # pragma: no cover - asset-dependent
            self.skipTest(f"standalone compile unavailable: {error}")
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        first = {
            geom: np.asarray(data.geom_xpos[int(model.geom(geom).id)]).copy()
            for geom in ALL_GEOMS_V103
        }
        for _ in range(25):
            mujoco.mj_forward(model, data)
        for geom, value in first.items():
            self.assertTrue(
                np.allclose(data.geom_xpos[int(model.geom(geom).id)], value, atol=0.0)
            )

    def test_pendant_geoms_classify_as_mounted_fixture(self) -> None:
        from molmo_spaces.tasks.pact_place_contact_audit import classify_contact

        for geom in ALL_GEOMS_V103:
            pair = {
                "geom1": geom,
                "geom2": "robot_0/fr3_link5_collision",
                "body1": PENDANT_BODY_V103,
                "body2": "robot_0/fr3_link5",
                "root1": PENDANT_BODY_V103,
                "root2": "robot_0/",
            }
            self.assertEqual(classify_contact(pair), "mounted_fixture")


class StaticContactParityTest(unittest.TestCase):
    """Clear, exactly touching, and penetrating fixtures must agree."""

    def _model(self, gap: float):
        import mujoco

        half_z = 0.2525
        # robot box half 0.04; stem half 0.006 -> face separation == gap
        x = 0.04 + 0.006 + gap
        xml = f"""
        <mujoco>
          <worldbody>
            <body name="robot_0/fr3_link5" pos="0 0 0">
              <joint type="slide" axis="1 0 0"/>
              <geom name="robot_0/fr3_link5_collision" type="box"
                    size="0.04 0.04 0.04" contype="1" conaffinity="1"/>
            </body>
            <body name="pact_clutter_mount_v103" pos="{x} 0 0">
              <geom name="pact_clutter_mount_v103_stem_0_g" type="box"
                    size="0.006 0.006 {half_z}" contype="8" conaffinity="15"/>
            </body>
          </worldbody>
        </mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        return model, data

    def _distance(self, model, data) -> float:
        from pact_geom_distance import true_distance

        return float(
            true_distance(
                model,
                data,
                [int(model.geom("robot_0/fr3_link5_collision").id)],
                [int(model.geom("pact_clutter_mount_v103_stem_0_g").id)],
            )
        )

    def _contacts(self, model, data):
        from molmo_spaces.tasks.pact_place_contact_audit import classify_contact

        out = []
        for index in range(int(data.ncon)):
            contact = data.contact[index]
            if float(contact.dist) > 0.0:
                continue
            g1, g2 = int(contact.geom1), int(contact.geom2)
            record = {
                "geom1": model.geom(g1).name,
                "geom2": model.geom(g2).name,
                "body1": model.body(int(model.geom_bodyid[g1])).name,
                "body2": model.body(int(model.geom_bodyid[g2])).name,
                "root1": model.body(int(model.geom_bodyid[g1])).name,
                "root2": model.body(int(model.geom_bodyid[g2])).name,
                "distance_m": float(contact.dist),
            }
            out.append((record, classify_contact(record)))
        return out

    def test_clear_fixture_has_positive_distance_and_no_contact(self) -> None:
        model, data = self._model(0.05)
        self.assertGreater(self._distance(model, data), 0.04)
        self.assertEqual(self._contacts(model, data), [])

    def test_touching_fixture_agrees(self) -> None:
        model, data = self._model(0.0)
        self.assertLessEqual(abs(self._distance(model, data)), 1e-6)

    def test_penetrating_fixture_is_seen_by_distance_and_data_contact(self) -> None:
        model, data = self._model(-0.008)
        distance = self._distance(model, data)
        self.assertLess(distance, 0.0)
        contacts = self._contacts(model, data)
        self.assertTrue(contacts, "penetration produced no data.contact entry")
        self.assertIn("mounted_fixture", {label for _pair, label in contacts})
        self.assertAlmostEqual(
            min(float(pair["distance_m"]) for pair, _ in contacts), distance, places=4
        )


class TemplateLatticeTest(unittest.TestCase):
    def test_lattice_size_and_values(self) -> None:
        for side in ("left", "right"):
            templates = enumerate_templates(side)
            self.assertEqual(len(templates), 120)
            self.assertEqual(
                sorted({t["lane_magnitude_m"] for t in templates}),
                sorted(LANE_MAGNITUDES_M),
            )
            self.assertEqual(
                sorted({t["staging_buffer_m"] for t in templates}),
                sorted(STAGING_BUFFERS_M),
            )
            self.assertEqual(
                sorted({t["pass_z_offset_m"] for t in templates}),
                sorted(PASS_Z_OFFSETS_M),
            )
            self.assertEqual(len({t["pass_rotation_key"] for t in templates}), 5)
            self.assertEqual(len({t["template_key"] for t in templates}), 120)
        signs = {t["lane_y_m"] < 0 for t in enumerate_templates("left")}
        self.assertEqual(signs, {True})
        signs = {t["lane_y_m"] > 0 for t in enumerate_templates("right")}
        self.assertEqual(signs, {True})

    def test_lateral_staging_is_outside_the_physical_slab(self) -> None:
        for side in ("left", "right"):
            for template in enumerate_templates(side):
                self.assertLess(template["near_staging_x_m"], 0.69)
                self.assertGreater(template["far_staging_x_m"], 0.71)
                self.assertAlmostEqual(
                    0.69 - template["near_staging_x_m"],
                    template["staging_buffer_m"],
                    places=9,
                )
                self.assertAlmostEqual(
                    template["far_staging_x_m"] - 0.71,
                    template["staging_buffer_m"],
                    places=9,
                )

    def test_eight_corners_perturb_lane_buffer_and_pass_z(self) -> None:
        template = enumerate_templates("left")[0]
        corners = template_corners(template)
        self.assertEqual(len(corners), 8)
        self.assertEqual(len({c["corner_key"] for c in corners}), 8)
        for corner in corners:
            self.assertAlmostEqual(
                abs(corner["lane_magnitude_m"] - template["lane_magnitude_m"]),
                0.005,
                places=9,
            )
            self.assertAlmostEqual(
                abs(corner["staging_buffer_m"] - template["staging_buffer_m"]),
                0.005,
                places=9,
            )
            self.assertAlmostEqual(
                abs(corner["pass_z_offset_m"] - template["pass_z_offset_m"]),
                0.005,
                places=9,
            )

    def test_control_pose_topology_and_lateral_blend(self) -> None:
        stock = {
            "positions_m": np.array(
                [[0.50, -0.08, 0.85], [0.70, -0.12, 0.88], [0.75, -0.12, 0.89]]
            ),
            "rotations": np.stack([np.eye(3)] * 3),
            "start_position_m": np.array([0.50, -0.08, 0.85]),
            "start_rotation": np.eye(3),
            "end_position_m": np.array([0.75, -0.12, 0.89]),
            "end_rotation": np.eye(3),
        }
        template = enumerate_templates("left")[0]
        inbound = build_control_poses(stock, template, direction="inbound")
        self.assertEqual(
            [p["name"] for p in inbound], list(CONTROL_POSES_INBOUND[1:-1])
        )
        outbound = build_control_poses(stock, template, direction="outbound")
        self.assertEqual(
            [p["name"] for p in outbound], list(CONTROL_POSES_OUTBOUND[1:-1])
        )
        for poses in (inbound, outbound):
            for pose in poses:
                x = float(pose["position_m"][0])
                self.assertTrue(x < 0.69 or x > 0.71, pose["name"])
            lane = [p for p in poses if p["name"].endswith("lane_staging")]
            self.assertEqual(len(lane), 2)
            for pose in lane:
                self.assertAlmostEqual(
                    float(pose["position_m"][1]), template["lane_y_m"], places=9
                )

    def test_stock_pose_interpolation_clamps_beyond_the_path(self) -> None:
        positions = np.array([[0.50, -0.08, 0.85], [0.75, -0.12, 0.89]])
        rotations = np.stack([np.eye(3)] * 2)
        pos, _rot = stock_pose_at_x(positions, rotations, 0.83)
        self.assertAlmostEqual(float(pos[1]), -0.12, places=9)
        pos, _rot = stock_pose_at_x(positions, rotations, 0.40)
        self.assertAlmostEqual(float(pos[1]), -0.08, places=9)
        pos, _rot = stock_pose_at_x(positions, rotations, 0.625)
        self.assertAlmostEqual(float(pos[0]), 0.625, places=9)
        self.assertAlmostEqual(float(pos[1]), -0.10, places=9)


class PlannerMechanicsTest(unittest.TestCase):
    def test_halton_seeds_are_fixed_and_inside_the_inner_range(self) -> None:
        low = np.array([-2.7, -1.7, -2.9, -3.0, -2.8, 0.5, -3.0])
        high = np.array([2.7, 1.7, 2.9, -0.15, 2.8, 4.5, 3.0])
        seeds = halton_joint_seeds(low, high)
        again = halton_joint_seeds(low, high)
        self.assertEqual(seeds.shape, (N_HALTON_SEEDS, N_ARM_JOINTS))
        self.assertTrue(np.array_equal(seeds, again))
        center = 0.5 * (low + high)
        half = 0.5 * (high - low) * 0.80
        self.assertTrue(np.all(seeds >= center - half - 1e-12))
        self.assertTrue(np.all(seeds <= center + half + 1e-12))

    def test_dedup_is_l_infinity_at_the_registered_tolerance(self) -> None:
        base = np.zeros(N_ARM_JOINTS)
        near = base.copy()
        near[3] = DEDUP_LINF_RAD * 0.5
        far = base.copy()
        far[3] = DEDUP_LINF_RAD * 2.0
        kept = dedup_joint_solutions([base, near, far])
        self.assertEqual(kept, [0, 2])

    def test_ranking_prefers_clearance_then_margin_then_travel(self) -> None:
        def node(key, margin=1.0, orientation=0.0):
            return {
                "node_key": key,
                "joint_limit_margin_rad": margin,
                "orientation_deviation_deg": orientation,
            }

        layers = [[node("s")], [node("a"), node("b")], [node("e")]]
        edges = {
            (0, 0, 0): {"passed": True, "min_clearance_m": 0.05, "joint_travel_rad": 1.0,
                        "min_joint_limit_margin_rad": 1.0},
            (0, 0, 1): {"passed": True, "min_clearance_m": 0.09, "joint_travel_rad": 3.0,
                        "min_joint_limit_margin_rad": 1.0},
            (1, 0, 0): {"passed": True, "min_clearance_m": 0.08, "joint_travel_rad": 1.0,
                        "min_joint_limit_margin_rad": 1.0},
            (1, 1, 0): {"passed": True, "min_clearance_m": 0.07, "joint_travel_rad": 1.0,
                        "min_joint_limit_margin_rad": 1.0},
        }
        chosen = select_path(layers, edges)
        self.assertIsNotNone(chosen)
        self.assertEqual(chosen["node_indices"], [0, 1, 0])
        self.assertAlmostEqual(chosen["min_clearance_m"], 0.07)

    def test_ranking_returns_none_when_an_edge_is_missing(self) -> None:
        layers = [[{"node_key": "s"}], [{"node_key": "a"}], [{"node_key": "e"}]]
        edges = {
            (0, 0, 0): {"passed": True, "min_clearance_m": 0.05, "joint_travel_rad": 1.0,
                        "min_joint_limit_margin_rad": 1.0},
        }
        self.assertIsNone(select_path(layers, edges))
        self.assertIsNone(select_path([[], [{"node_key": "a"}]], {}))

    def test_registered_clearance_floors(self) -> None:
        self.assertAlmostEqual(NODE_MIN_CLEARANCE_M, 0.020)
        self.assertAlmostEqual(EDGE_MIN_CLEARANCE_M, 0.020)
        self.assertAlmostEqual(CORNER_MIN_CLEARANCE_M, 0.015)
        self.assertAlmostEqual(MAX_JOINT_STEP_RAD, 0.01)
        self.assertAlmostEqual(MIN_ROW_CLEARANCE_M, 0.015)

    def test_segment_speed_classes_and_durations(self) -> None:
        self.assertEqual(
            segment_speed_class("actual_initial->near_stock_staging", direction="inbound"),
            "empty_arm_approach",
        )
        self.assertEqual(
            segment_speed_class(
                "near_lane_staging->far_lane_staging", direction="inbound"
            ),
            "pendant_pass",
        )
        self.assertEqual(
            segment_speed_class(
                "far_lane_staging->near_lane_staging", direction="outbound"
            ),
            "pendant_pass",
        )
        self.assertEqual(
            segment_speed_class(
                "far_pregrasp_staging->actual_pregrasp_endpoint", direction="inbound"
            ),
            "pregrasp_approach",
        )
        slow = segment_duration_s(
            tcp_arc_length_m=0.45,
            joint_displacements_rad=np.zeros(N_ARM_JOINTS),
            commanded_speed_m_s=0.045,
            velocity_limits_rad_s=np.full(N_ARM_JOINTS, 2.62),
        )
        self.assertAlmostEqual(slow["duration_s"], 10.0, places=6)
        self.assertEqual(slow["binding"], "tcp")
        joint_bound = segment_duration_s(
            tcp_arc_length_m=0.01,
            joint_displacements_rad=np.array([2.0, 0, 0, 0, 0, 0, 0]),
            commanded_speed_m_s=0.15,
            velocity_limits_rad_s=np.full(N_ARM_JOINTS, 2.62),
        )
        self.assertEqual(joint_bound["binding"], "joint")
        self.assertAlmostEqual(joint_bound["duration_s"], 2.0 / (2.62 * 0.5), places=9)

    def test_qpos_hash_is_order_sensitive(self) -> None:
        a = np.zeros(N_ARM_JOINTS)
        b = np.ones(N_ARM_JOINTS)
        self.assertNotEqual(
            qpos_sequence_sha256([a, b]), qpos_sequence_sha256([b, a])
        )
        self.assertEqual(qpos_sequence_sha256([a, b]), qpos_sequence_sha256([a, b]))


class ContractTest(unittest.TestCase):
    def test_authorizations_start_false(self) -> None:
        for key, value in empty_authorization().items():
            self.assertFalse(value, key)

    def test_lattice_is_reported_intact(self) -> None:
        lattice = search_lattice()
        self.assertEqual(lattice["height_lattice_m"], list(HEIGHT_LATTICE_M))
        self.assertEqual(lattice["n_templates_per_side"], 120)
        self.assertEqual(lattice["n_halton_seeds"], 24)
        self.assertEqual(lattice["node_min_clearance_m"], 0.020)
        self.assertEqual(lattice["corner_min_clearance_m"], 0.015)

    def test_environment_marker_is_unique_to_v103(self) -> None:
        self.assertEqual(
            ENVIRONMENT_VERSION_V103,
            "pact_place_corridor_v10_3_static_pendant_joint_route",
        )
        self.assertNotIn("v10_2", ENVIRONMENT_VERSION_V103)
        self.assertTrue(implementation_sha256())
        self.assertTrue(CONTRACT_VERSION.startswith("pact_place_v103_"))


if __name__ == "__main__":
    unittest.main()
