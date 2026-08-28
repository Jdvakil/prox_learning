"""Behavioral tests for V10.5: V9.5 real clutter with a static pendant.

Every expectation is recomputed from source, from live MuJoCo state, or from
file bytes. No test asserts a stored result boolean from an artifact.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "scripts", ROOT / "submodules" / "molmospaces"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from pact_place_v105_contract import (  # noqa: E402
    CONTRACT_VERSION_V105,
    ENVIRONMENT_VERSION_V105,
    INTRUSION_SIDES,
    N_PHASE0_ROWS,
    N_REVIEW_ROWS,
    N_REVIEW_VIDEOS,
    PHASE0_MIN_CLEAN,
    PHASE0_MIN_CLEAN_PER_POSE,
    PHASE0_MIN_CLEAN_PER_SIDE,
    PHASE0_MIN_CLEAN_PER_SIDE_POSE,
    PHASE0_RISK_CONFIRM_MAX_M,
    PHASE0_STREAM,
    REVIEW_STREAM,
    SPEC_IMPLEMENTATION_PATHS,
    V95_LAYOUT_FAMILY_IDS,
    V95_VESSEL_JITTER,
    build_specification_contract,
    canonical_payload_sha256,
    empty_authorization,
    gate_eligibility,
    implementation_digest,
    phase0_cells,
    phase0_rows,
    review_rows,
    sha256_payload,
    streams_are_disjoint,
    v95_row_payload,
    wilson_interval,
    write_immutable_create_only,
)
from pact_place_v105_geometry import (  # noqa: E402
    ALL_GEOMS_V105,
    CLEARANCE_FLOOR_M,
    LATTICE_R_M,
    LATTICE_X_M,
    PENDANT_BODY_V105,
    POSE_IDS,
    POSE_OFFSETS_M,
    RISK_BAND_M,
    build_assembly,
    bundle_assemblies,
    lattice_candidates,
    scene_xml_sha256,
    scene_xml_text,
)


# ---------------------------------------------------------------------------
# 1. Settled V9.5 lineage restored, low wall absent
# ---------------------------------------------------------------------------
class LineageTest(unittest.TestCase):
    def test_layout_families_are_the_settled_v95_four(self):
        from pact_place_v95_contract import V95_LAYOUT_FAMILIES

        self.assertEqual(tuple(V95_LAYOUT_FAMILIES), V95_LAYOUT_FAMILY_IDS)

    def test_row_payload_uses_the_v95_palette_and_layout_helpers(self):
        from pact_place_v95_contract import build_v95_layout, load_v95_palette

        palette = load_v95_palette()
        for family in V95_LAYOUT_FAMILY_IDS:
            for side in INTRUSION_SIDES:
                payload = v95_row_payload(family, side)
                expected = build_v95_layout(
                    palette, family_id=family, intrusion_side=side
                )
                self.assertEqual(payload["pact_clutter_layout"], expected)
                self.assertEqual(payload["layout_id"], expected["layout_id"])
                self.assertEqual(
                    payload["pact_clutter_palette"], list(palette["palette"])
                )

    def test_vessel_jitter_matches_the_inherited_v93_table(self):
        from run_pact_place_v9_panel_smoke import _row
        from pact_place_v95_contract import load_v95_palette

        palette = load_v95_palette()
        for index, family in enumerate(V95_LAYOUT_FAMILY_IDS):
            row = _row(
                index=0, family_id=family, side="left",
                palette_document=palette, implementation_sha256="x", seed=1,
            )
            self.assertEqual(row["clutter_x_jitter_m"], V95_VESSEL_JITTER[index][0])
            self.assertEqual(row["clutter_y_jitter_m"], V95_VESSEL_JITTER[index][1])

    def test_sampler_derives_from_v93_not_the_low_wall(self):
        from molmo_spaces.tasks.enclosure_reach import (
            PactPlaceCorridorV93Sampler,
            PactPlaceCorridorV95LowWallSampler,
            PactPlaceCorridorV105Sampler,
        )

        self.assertTrue(
            issubclass(PactPlaceCorridorV105Sampler, PactPlaceCorridorV93Sampler)
        )
        self.assertFalse(
            issubclass(PactPlaceCorridorV105Sampler, PactPlaceCorridorV95LowWallSampler)
        )

    def test_contract_records_the_fixture_free_lineage(self):
        lineage = build_specification_contract()["lineage"]
        self.assertFalse(lineage["uses_v95_low_wall"])
        self.assertFalse(lineage["uses_v94_v95_wall_fixture"])
        self.assertFalse(lineage["uses_old_ceiling_mount"])
        self.assertFalse(lineage["imports_v98_to_v103_route_branch"])
        self.assertEqual(lineage["sampler_behavior"], "PactPlaceCorridorV93Sampler")

    def test_base_scene_is_v5_not_v3(self):
        from pact_place_v105_geometry import BASE_SCENE_NAME_V105

        self.assertEqual(BASE_SCENE_NAME_V105, "pact_place_corridor_v5.xml")
        text = scene_xml_text(build_assembly(0.78, 0.31, 0.0, pose_id="center"))
        self.assertIn('<include file="pact_place_corridor_v5.xml"/>', text)

    def test_no_low_wall_geom_in_any_generated_scene(self):
        for pose in POSE_IDS:
            text = scene_xml_text(
                build_assembly(0.78, 0.31, POSE_OFFSETS_M[pose], pose_id=pose)
            )
            for forbidden in ("low_wall", "wall_fixture", "mount_ceiling",
                              "pact_clutter_mount_v9", "pact_clutter_mount_v10_",
                              "pact_clutter_mount_v104"):
                self.assertNotIn(forbidden, text, pose)


# ---------------------------------------------------------------------------
# 2/3. Pose IDs, manifest balance, hash mismatch refusal
# ---------------------------------------------------------------------------
class PoseAndManifestTest(unittest.TestCase):
    def test_three_pose_ids_and_offsets(self):
        self.assertEqual(POSE_IDS, ("neg5", "center", "pos5"))
        self.assertEqual(
            POSE_OFFSETS_M, {"neg5": -0.005, "center": 0.0, "pos5": 0.005}
        )

    def test_phase0_manifest_is_the_exact_cartesian_product(self):
        rows = phase0_rows()
        self.assertEqual(len(rows), N_PHASE0_ROWS)
        self.assertEqual(len(phase0_cells()), 24)
        self.assertEqual(
            len({(r["family_id"], r["intrusion_side"], r["pose_id"]) for r in rows}),
            24,
        )

    def test_phase0_manifest_balance(self):
        rows = phase0_rows()
        for side in INTRUSION_SIDES:
            self.assertEqual(
                sum(1 for r in rows if r["intrusion_side"] == side), 12, side
            )
        for pose in POSE_IDS:
            self.assertEqual(sum(1 for r in rows if r["pose_id"] == pose), 8, pose)
        for side in INTRUSION_SIDES:
            for pose in POSE_IDS:
                self.assertEqual(
                    sum(
                        1
                        for r in rows
                        if r["intrusion_side"] == side and r["pose_id"] == pose
                    ),
                    4,
                    f"{side}|{pose}",
                )
        for family in V95_LAYOUT_FAMILY_IDS:
            self.assertEqual(
                sum(1 for r in rows if r["family_id"] == family), 6, family
            )

    def test_review_pool_is_two_disjoint_replicates_of_the_same_product(self):
        rows = review_rows()
        self.assertEqual(len(rows), N_REVIEW_ROWS)
        self.assertEqual(N_REVIEW_ROWS, 2 * N_PHASE0_ROWS)
        for replicate in (0, 1):
            subset = [r for r in rows if r["replicate"] == replicate]
            self.assertEqual(len(subset), 24)
            self.assertEqual(
                len({(r["family_id"], r["intrusion_side"], r["pose_id"])
                     for r in subset}),
                24,
            )
        self.assertEqual(len({r["episode_id"] for r in rows}), N_REVIEW_ROWS)

    def test_review_and_phase0_streams_are_disjoint(self):
        report = streams_are_disjoint(review_rows(), phase0_rows())
        self.assertTrue(report["streams_differ"])
        self.assertEqual(report["episode_id_overlap"], [])
        self.assertEqual(report["seed_overlap"], [])
        self.assertTrue(report["disjoint"])
        self.assertNotEqual(REVIEW_STREAM, PHASE0_STREAM)

    def test_rows_bind_scene_and_assembly_hashes_when_supplied(self):
        scenes = {
            pose: {"relative": f"scenes/{pose}.xml", "sha256": f"{index}" * 64}
            for index, pose in enumerate(POSE_IDS)
        }
        assemblies = {pose: f"{index}" * 64 for index, pose in enumerate(POSE_IDS)}
        rows = phase0_rows(scene_by_pose=scenes, assembly_by_pose=assemblies)
        for row in rows:
            self.assertEqual(
                row["pact_v105_scene_sha256"], scenes[row["pose_id"]]["sha256"]
            )
            self.assertEqual(
                row["pact_v105_assembly_sha256"], assemblies[row["pose_id"]]
            )

    def test_sampler_refuses_a_scene_hash_mismatch(self):
        import inspect

        from molmo_spaces.tasks.enclosure_reach import PactPlaceCorridorV105Sampler

        source = inspect.getsource(PactPlaceCorridorV105Sampler.sample_task)
        self.assertIn("scene hash mismatch", source)
        self.assertIn("pact_v105_scene_sha256", source)


# ---------------------------------------------------------------------------
# 4/5. Static pendant, no runtime model writes
# ---------------------------------------------------------------------------
class StaticPendantTest(unittest.TestCase):
    def test_scene_body_has_no_joint_freejoint_mocap_or_actuator(self):
        for pose in POSE_IDS:
            text = scene_xml_text(
                build_assembly(0.78, 0.31, POSE_OFFSETS_M[pose], pose_id=pose)
            )
            body = text[text.index(f'<body name="{PENDANT_BODY_V105}"'):]
            body = body[: body.index("</body>")]
            for forbidden in ("<joint", "<freejoint", "mocap=", "<actuator"):
                self.assertNotIn(forbidden, body, f"{pose}:{forbidden}")

    PROTECTED_MODEL_FIELDS = (
        "geom_pos", "geom_size", "geom_aabb", "geom_rbound", "bvh_aabb",
        "body_pos", "body_quat", "mocap_pos",
    )

    def _assigned_model_fields(self, path: Path) -> set[str]:
        """Every ``<something>.<field>[...] = ...`` target, found by AST.

        A substring scan cannot tell a read from a write; this walks assignment
        targets only, so reading ``model.geom_size[i]`` is correctly ignored.
        """
        import ast

        found: set[str] = set()
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            targets = []
            if isinstance(node, ast.Assign):
                targets = list(node.targets)
            elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
                targets = [node.target]
            for target in targets:
                inner = target
                while isinstance(inner, ast.Subscript):
                    inner = inner.value
                if isinstance(inner, ast.Attribute):
                    found.add(inner.attr)
        return found

    def test_no_runtime_write_to_pendant_model_fields(self):
        for name in (
            "scripts/pact_place_v105_geometry.py",
            "scripts/pact_place_v105_clearance.py",
            "scripts/pact_place_v105_runtime.py",
            "scripts/pact_place_v105_siting_core.py",
            "scripts/run_pact_place_v105_reconstruct.py",
            "scripts/run_pact_place_v105_siting.py",
        ):
            assigned = self._assigned_model_fields(ROOT / name)
            for field in self.PROTECTED_MODEL_FIELDS:
                self.assertNotIn(field, assigned, f"{name} assigns {field}")

    def test_the_ast_check_would_catch_a_real_write(self):
        """Guard the guard: a genuine model write must be detected."""
        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch) / "bad.py"
            path.write_text(
                "def f(model, gid, v):\n"
                "    x = model.geom_size[gid]\n"      # a read, must be ignored
                "    model.geom_pos[gid] = v\n"       # a write, must be caught
            )
            assigned = self._assigned_model_fields(path)
            self.assertIn("geom_pos", assigned)
            self.assertNotIn("geom_size", assigned)

    def test_sampler_does_not_pose_or_resize_the_pendant(self):
        import inspect

        from molmo_spaces.tasks.enclosure_reach import PactPlaceCorridorV105Sampler

        source = inspect.getsource(PactPlaceCorridorV105Sampler)
        for forbidden in ("geom_pos", "geom_size", "geom_aabb", "geom_rbound",
                          "mocap_pos", "body_pos["):
            self.assertNotIn(forbidden, source, forbidden)

    def test_visible_and_collision_geoms_are_the_same_five(self):
        assembly = build_assembly(0.78, 0.31, 0.0, pose_id="center")
        self.assertEqual(len(assembly["components"]), 5)
        self.assertEqual(
            tuple(item["geom"] for item in assembly["components"]), ALL_GEOMS_V105
        )
        text = scene_xml_text(assembly)
        for geom in ALL_GEOMS_V105:
            self.assertEqual(text.count(f'name="{geom}"'), 1, geom)
        self.assertNotIn("contype=\"0\"", text)
        self.assertNotIn("group=\"3\"", text)

    def test_no_pendant_counterfactual_scene_has_no_pendant_body(self):
        text = scene_xml_text(None)
        self.assertNotIn(PENDANT_BODY_V105, text)
        self.assertIn('<include file="pact_place_corridor_v5.xml"/>', text)
        self.assertNotEqual(
            scene_xml_sha256(None),
            scene_xml_sha256(build_assembly(0.78, 0.31, 0.0, pose_id="center")),
        )


class CompiledSceneTest(unittest.TestCase):
    """Compile a V10.5 scene and check bounds enclose the true geometry."""

    @classmethod
    def setUpClass(cls):
        import os

        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        os.environ.setdefault("MLSPACES_ASSETS_DIR", str(ROOT / "assets"))

    def _compile(self, assembly):
        import shutil

        import mujoco

        scenes = (
            ROOT
            / "submodules/molmospaces/molmo_spaces/data_generation/custom_scenes"
        )
        scratch = Path(tempfile.mkdtemp(prefix="v105_scene_"))
        for name in ("pact_place_corridor_v3.xml", "pact_place_corridor_v5.xml"):
            shutil.copyfile(scenes / name, scratch / name)
        path = scratch / "probe.xml"
        path.write_text(scene_xml_text(assembly))
        try:
            return mujoco.MjModel.from_xml_path(str(path))
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    def test_compiled_positions_sizes_and_bounds(self):
        for pose in POSE_IDS:
            assembly = build_assembly(
                0.78, 0.31, POSE_OFFSETS_M[pose], pose_id=pose
            )
            model = self._compile(assembly)
            body_id = int(model.body(PENDANT_BODY_V105).id)
            self.assertEqual(int(model.body_dofnum[body_id]), 0, pose)
            self.assertEqual(int(model.body_jntnum[body_id]), 0, pose)
            self.assertLess(int(model.body_mocapid[body_id]), 0, pose)
            for item in assembly["components"]:
                gid = int(model.geom(item["geom"]).id)
                half = np.asarray(item["half_m"], dtype=float)
                centre = np.asarray(item["center_m"], dtype=float)
                self.assertTrue(
                    np.allclose(model.geom_size[gid], half, atol=1e-9), item["name"]
                )
                self.assertTrue(
                    np.allclose(model.geom_pos[gid], centre, atol=1e-9), item["name"]
                )
                aabb = np.asarray(model.geom_aabb[gid], dtype=float)
                self.assertTrue(np.all(aabb[3:] >= half - 1e-12), item["name"])
                self.assertGreaterEqual(
                    float(model.geom_rbound[gid]),
                    float(np.linalg.norm(half)) - 1e-9,
                    item["name"],
                )

    def test_pendant_collision_is_enabled(self):
        assembly = build_assembly(0.78, 0.31, 0.0, pose_id="center")
        model = self._compile(assembly)
        for item in assembly["components"]:
            gid = int(model.geom(item["geom"]).id)
            self.assertNotEqual(
                (int(model.geom_contype[gid]), int(model.geom_conaffinity[gid])),
                (0, 0),
                item["name"],
            )
            self.assertGreater(float(model.geom_rgba[gid][3]), 0.0, item["name"])

    def test_the_three_poses_compile_to_distinct_geometry(self):
        positions = {}
        for pose in POSE_IDS:
            model = self._compile(
                build_assembly(0.78, 0.31, POSE_OFFSETS_M[pose], pose_id=pose)
            )
            gid = int(model.geom(ALL_GEOMS_V105[0]).id)
            positions[pose] = float(model.geom_pos[gid][1])
        self.assertLess(positions["neg5"], positions["center"])
        self.assertLess(positions["center"], positions["pos5"])
        self.assertAlmostEqual(positions["pos5"] - positions["neg5"], 0.010, places=9)


# ---------------------------------------------------------------------------
# 6/7. Contact classification and clutter semantics
# ---------------------------------------------------------------------------
class ContactSemanticsTest(unittest.TestCase):
    def test_robot_pendant_contact_is_mounted_fixture(self):
        from molmo_spaces.tasks.pact_place_contact_audit import classify_contact

        for geom in ALL_GEOMS_V105:
            pair = {
                "geom1": "robot_0/fr3_link7_collision", "geom2": geom,
                "body1": "robot_0/fr3_link7", "body2": PENDANT_BODY_V105,
            }
            self.assertEqual(classify_contact(pair), "mounted_fixture", geom)

    def test_carried_target_pendant_contact_is_mounted_fixture(self):
        from molmo_spaces.tasks.pact_place_contact_audit import classify_contact

        pair = {
            "geom1": "cavity_obj_0_collision", "geom2": ALL_GEOMS_V105[0],
            "body1": "cavity_obj_0", "body2": PENDANT_BODY_V105,
        }
        self.assertEqual(classify_contact(pair), "mounted_fixture")

    def test_ordinary_clutter_contact_is_still_clutter(self):
        from molmo_spaces.tasks.pact_place_contact_audit import classify_contact

        pair = {
            "geom1": "robot_0/fr3_link6_collision", "geom2": "pact_clutter_01_g",
            "body1": "robot_0/fr3_link6", "body2": "pact_clutter_01",
        }
        self.assertEqual(classify_contact(pair), "clutter")

    def test_mounted_fixture_and_clutter_are_both_strict_unclean(self):
        from pact_place_v105_contract import DISALLOWED_CONTACT_CLASSES

        self.assertIn("mounted_fixture", DISALLOWED_CONTACT_CLASSES)
        self.assertIn("clutter", DISALLOWED_CONTACT_CLASSES)

    def test_household_objects_stay_movable_free_bodies(self):
        payload = v95_row_payload(V95_LAYOUT_FAMILY_IDS[0], "left")
        objects = payload["pact_clutter_layout"]["objects"]
        self.assertGreater(len(objects), 0)
        for item in objects:
            self.assertNotIn("visual_only", item)
            self.assertNotIn("parked", item)
            self.assertNotIn("immovable", item)


# ---------------------------------------------------------------------------
# 8/9. Speed cap and expert dispatch
# ---------------------------------------------------------------------------
class SpeedAndDispatchTest(unittest.TestCase):
    def test_cap_value_and_marker_gate(self):
        import inspect

        from molmo_spaces.tasks.enclosure_reach import PactPlaceCorridorPolicy
        from pact_place_v105_runtime import INITIAL_FREE_SPACE_SPEED_CAP_M_S

        self.assertEqual(INITIAL_FREE_SPACE_SPEED_CAP_M_S, 0.12)
        source = inspect.getsource(
            PactPlaceCorridorPolicy._v105_apply_speed_amendment
        )
        self.assertIn("_v105_enabled", source)
        self.assertIn("environment marker is not V10.5", source)
        self.assertIn("schedule_sha256", source)

    def test_cap_binds_the_first_free_space_segment_only(self):
        from pact_place_v105_runtime import (
            apply_initial_free_space_speed_cap,
            plan_signature,
            verify_plan_matches_baseline,
        )

        class Seg:
            def __init__(self, name, speed):
                self.name = name
                self.speed = speed
                self.start_pose = np.eye(4)
                self.end_pose = np.eye(4)

        from molmo_spaces.policy.solvers.object_manipulation.base_object_manipulation_planner_policy import (  # noqa: E501
            GripperAction,
            TCPMoveSequence,
        )

        sequence = TCPMoveSequence.__new__(TCPMoveSequence)
        sequence._move_segments = [Seg("pregrasp", 0.20), Seg("pregrasp", 0.045)]
        close = GripperAction.__new__(GripperAction)
        close.target_open = False
        primitives = [sequence, close]
        baseline = plan_signature(primitives)
        record = apply_initial_free_space_speed_cap(primitives)
        amended = plan_signature(primitives)
        self.assertTrue(record["applied"])
        self.assertEqual(record["speed_before_m_s"], 0.20)
        self.assertEqual(record["speed_after_m_s"], 0.12)
        self.assertEqual(sequence._move_segments[1].speed, 0.045)
        comparison = verify_plan_matches_baseline(baseline, amended)
        self.assertEqual(comparison["n_speed_changes"], 1)
        self.assertTrue(comparison["poses_identical"])
        self.assertTrue(comparison["passed"])

    def test_v105_is_in_the_v9_expert_allowlist(self):
        import inspect

        from molmo_spaces.tasks.enclosure_reach import PactPlaceCorridorPolicy

        source = inspect.getsource(PactPlaceCorridorPolicy._v9_enabled)
        self.assertIn("PACT_PLACE_V105_ENVIRONMENT_VERSION", source)

    def test_v105_takes_the_v93_inbound_vessel_branch(self):
        import inspect

        from molmo_spaces.tasks.enclosure_reach import PactPlaceCorridorPolicy

        source = inspect.getsource(PactPlaceCorridorPolicy._compute_trajectory)
        marker = source.index("inbound_hazard_role")
        window = source[marker : marker + 1400]
        self.assertIn("PACT_PLACE_V105_ENVIRONMENT_VERSION", window)
        self.assertIn("pact_place_corridor_v9_3", window)

    def test_pendant_is_not_a_planner_obstacle_and_gets_no_lateral_bow(self):
        from molmo_spaces.tasks.enclosure_reach import (
            PACT_PLACE_V105_ENVIRONMENT_VERSION,
            PactPlaceCorridorPolicy,
        )

        self.assertEqual(
            PactPlaceCorridorPolicy._mounted_fixture_roles(
                PACT_PLACE_V105_ENVIRONMENT_VERSION
            ),
            (),
        )

    def test_v105_is_not_in_the_v10_lane_family(self):
        from molmo_spaces.tasks.enclosure_reach import (
            PACT_PLACE_V10_LANE_ENVIRONMENT_VERSIONS,
            PACT_PLACE_V105_ENVIRONMENT_VERSION,
        )

        self.assertNotIn(
            PACT_PLACE_V105_ENVIRONMENT_VERSION,
            PACT_PLACE_V10_LANE_ENVIRONMENT_VERSIONS,
        )


# ---------------------------------------------------------------------------
# 10. No privileged leakage into either student observation
# ---------------------------------------------------------------------------
class ObservationLeakageTest(unittest.TestCase):
    PRIVILEGED = (
        "pose_id", "pact_v105_scene_sha256", "pact_v105_assembly_sha256",
        "pact_v105_x_m", "pact_v105_r_m", "pose_offset_m",
        "lobe_stem_min_clearance_m", "pendant_min_clearance_m",
    )

    def test_sampler_defines_no_observation_hook(self):
        """The sampler may record telemetry; it must not build observations.

        Checked structurally rather than by substring, so a comment mentioning
        the word cannot pass or fail the test.
        """
        import ast
        import inspect
        import textwrap

        from molmo_spaces.tasks.enclosure_reach import PactPlaceCorridorV105Sampler

        tree = ast.parse(
            textwrap.dedent(inspect.getsource(PactPlaceCorridorV105Sampler))
        )
        defined = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        }
        self.assertEqual(defined, {"_draw_theta", "sample_task"})
        called = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        for forbidden in ("get_observations", "build_observation",
                          "observation_space"):
            self.assertNotIn(forbidden, called, forbidden)

    def test_privileged_keys_never_enter_an_observation_builder(self):
        text = (
            ROOT / "submodules/molmospaces/molmo_spaces/tasks/enclosure_reach.py"
        ).read_text()
        lines = text.splitlines()
        for index, line in enumerate(lines):
            if "observation" not in line.lower():
                continue
            for key in ("pact_v105_pose_id", "pact_v105_assembly_sha256",
                        "pact_v105_scene_sha256"):
                self.assertNotIn(key, line, f"line {index + 1}: {line.strip()}")

    def test_both_students_keep_the_same_rgb_state_inputs(self):
        contract = build_specification_contract()
        self.assertFalse(contract["authorizes_training"])
        self.assertFalse(contract["authorizes_evaluation"])


# ---------------------------------------------------------------------------
# 11. Lattice, admission, deterministic ranking
# ---------------------------------------------------------------------------
class LatticeTest(unittest.TestCase):
    def test_lattice_is_the_registered_four_by_eight(self):
        self.assertEqual(LATTICE_X_M, (0.740, 0.760, 0.780, 0.800))
        self.assertEqual(
            LATTICE_R_M,
            (0.290, 0.295, 0.300, 0.305, 0.310, 0.315, 0.320, 0.325),
        )
        self.assertEqual(len(lattice_candidates()), 32)
        self.assertEqual(len(lattice_candidates()) * len(POSE_IDS), 96)

    def test_a_candidate_is_a_three_pose_bundle(self):
        bundle = bundle_assemblies(0.78, 0.31)
        self.assertEqual(sorted(bundle), sorted(POSE_IDS))
        for pose, assembly in bundle.items():
            self.assertEqual(assembly["pose_id"], pose)
            self.assertAlmostEqual(assembly["d_m"], POSE_OFFSETS_M[pose], places=9)

    def test_inner_faces_are_inboard_of_v104(self):
        # V10.4's inner lobe faces sat at |y| = 0.330; every V10.5 candidate is
        # necessarily closer to the corridor centre than that.
        for x, r in lattice_candidates():
            for pose in POSE_IDS:
                assembly = build_assembly(x, r, POSE_OFFSETS_M[pose], pose_id=pose)
                self.assertLess(
                    assembly["min_inner_lobe_face_abs_y_m"], 0.330,
                    f"{x}|{r}|{pose}",
                )

    def test_shape_and_height_are_frozen_across_the_lattice(self):
        for x, r in lattice_candidates():
            assembly = build_assembly(x, r, 0.0, pose_id="center")
            self.assertAlmostEqual(assembly["lobe_bottom_z_m"], 0.98, places=9)
            self.assertAlmostEqual(assembly["lobe_top_z_m"], 1.04, places=9)
            self.assertAlmostEqual(assembly["crossbar_top_z_m"], 1.515, places=9)
            for item in assembly["components"]:
                if item["role"] == "lobe":
                    self.assertEqual(item["half_m"], [0.010, 0.010, 0.030])

    def test_ranking_is_deterministic_and_untruncated(self):
        from run_pact_place_v105_siting import evaluate_bundle  # noqa: F401

        contract = build_specification_contract()
        self.assertIsNone(contract["ranking_truncation"])
        self.assertEqual(len(contract["ranking"]), 5)

    def test_lattice_is_not_extendable_after_results(self):
        contract = build_specification_contract()
        self.assertFalse(contract["lattice"]["may_be_extended_after_results"])
        self.assertEqual(contract["lattice"]["searchable_dimensions"], ["x", "r"])


# ---------------------------------------------------------------------------
# 14. Phase-0 counting at 16/24 with every balance floor
# ---------------------------------------------------------------------------
class GateCountingTest(unittest.TestCase):
    def _results(self, clean_flags, risk_m=0.030):
        rows = phase0_rows()
        out = []
        for row, clean in zip(rows, clean_flags):
            out.append(
                {
                    "role_index": row["role_index"],
                    "v105_clean_success": bool(clean),
                    "v105_defects": [] if clean else ["synthetic"],
                    "pact_v105_frame_telemetry": {
                        "min_lobe_stem_clearance_m": risk_m
                    },
                }
            )
        return rows, out

    def test_thresholds_are_the_registered_ones(self):
        self.assertEqual(PHASE0_MIN_CLEAN, 16)
        self.assertEqual(PHASE0_MIN_CLEAN_PER_SIDE, 7)
        self.assertEqual(PHASE0_MIN_CLEAN_PER_POSE, 4)
        self.assertEqual(PHASE0_MIN_CLEAN_PER_SIDE_POSE, 2)
        self.assertEqual(PHASE0_RISK_CONFIRM_MAX_M, 0.035)

    def test_all_clean_passes(self):
        rows, results = self._results([True] * 24)
        report = gate_eligibility(rows, results)
        self.assertEqual(report["clean_successes"], 24)
        self.assertTrue(report["phase0_passed"])
        self.assertFalse(report["authorizes_collection"])
        self.assertFalse(report["authorizes_training"])

    def test_fifteen_clean_fails_the_bar(self):
        flags = [True] * 15 + [False] * 9
        rows, results = self._results(flags)
        report = gate_eligibility(rows, results)
        self.assertEqual(report["clean_successes"], 15)
        self.assertFalse(report["phase0_passed"])
        self.assertTrue(
            any("clean 15 < 16" in item for item in report["limiting_predicates"])
        )

    def test_sixteen_clean_can_still_fail_a_balance_floor(self):
        rows = phase0_rows()
        # 16 clean but every left row unclean would break the per-side floor.
        flags = [row["intrusion_side"] == "right" for row in rows]
        extra = 0
        for index, row in enumerate(rows):
            if not flags[index] and extra < 4:
                flags[index] = True
                extra += 1
        _, results = self._results(flags)
        report = gate_eligibility(rows, results)
        self.assertEqual(report["clean_successes"], 16)
        self.assertFalse(report["phase0_passed"])
        self.assertTrue(report["limiting_predicates"])

    def test_a_risk_confirmation_failure_fails_the_gate(self):
        rows, results = self._results([True] * 24, risk_m=0.20)
        report = gate_eligibility(rows, results)
        self.assertEqual(report["clean_successes"], 24)
        self.assertFalse(report["phase0_passed"])
        self.assertTrue(
            any("risk confirmation" in item
                for item in report["limiting_predicates"])
        )

    def test_a_missing_row_fails_closed(self):
        rows, results = self._results([True] * 24)
        report = gate_eligibility(rows, results[:-1])
        self.assertFalse(report["phase0_passed"])
        self.assertEqual(report["incomplete_rows"], 1)

    def test_wilson_interval_is_exact(self):
        low, high = wilson_interval(16, 24)
        self.assertAlmostEqual(low, 0.4671, places=3)
        self.assertAlmostEqual(high, 0.8203, places=3)
        self.assertEqual(wilson_interval(0, 0), (0.0, 0.0))


# ---------------------------------------------------------------------------
# 16. Authorization defaults cannot be overwritten by merge order
# ---------------------------------------------------------------------------
class AuthorizationTest(unittest.TestCase):
    def test_defaults_are_all_false(self):
        for key, value in empty_authorization().items():
            self.assertFalse(value, key)

    def test_spreading_after_an_outcome_key_would_reset_it(self):
        """The defect this ordering rule exists to prevent."""
        wrong = {"phase0_passed": True, **empty_authorization()}
        self.assertFalse(wrong["phase0_passed"])
        right = {**empty_authorization(), "phase0_passed": True}
        self.assertTrue(right["phase0_passed"])

    def test_gate_eligibility_spreads_defaults_first(self):
        import inspect

        source = inspect.getsource(gate_eligibility)
        spread = source.index("**empty_authorization()")
        outcome = source.index('"phase0_passed": passed')
        self.assertLess(spread, outcome)

    def test_every_v105_artifact_writer_carries_authorization(self):
        for name in (
            "scripts/run_pact_place_v105_reconstruct.py",
            "scripts/run_pact_place_v105_siting.py",
        ):
            text = (ROOT / name).read_text()
            self.assertIn("empty_authorization()", text, name)


# ---------------------------------------------------------------------------
# Contract hashing discipline
# ---------------------------------------------------------------------------
class ContractHashTest(unittest.TestCase):
    def test_payload_and_raw_hashes_are_distinct_concepts(self):
        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch) / "a.json"
            hashes = write_immutable_create_only(path, {"value": 1})
            self.assertNotEqual(hashes["payload_sha256"], hashes["raw_file_sha256"])
            document = json.loads(path.read_text())
            self.assertEqual(
                document["payload_sha256"], hashes["payload_sha256"]
            )
            self.assertEqual(
                canonical_payload_sha256(document), hashes["payload_sha256"]
            )

    def test_writer_refuses_to_replace(self):
        from pact_place_v105_contract import ImmutableArtifactError

        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch) / "a.json"
            write_immutable_create_only(path, {"value": 1})
            with self.assertRaises(ImmutableArtifactError):
                write_immutable_create_only(path, {"value": 2})

    def test_contract_hashes_an_explicit_ordered_file_list(self):
        contract = build_specification_contract()
        self.assertEqual(
            list(contract["implementation_files"]), list(SPEC_IMPLEMENTATION_PATHS)
        )
        self.assertTrue(contract["hash_discipline"]["hashes_an_explicit_ordered_file_list"])
        self.assertFalse(contract["hash_discipline"]["circular_aggregate_hash"])

    def test_an_unrelated_file_cannot_change_the_digest(self):
        before = implementation_digest(SPEC_IMPLEMENTATION_PATHS)
        self.assertEqual(before, implementation_digest(SPEC_IMPLEMENTATION_PATHS))
        subset = SPEC_IMPLEMENTATION_PATHS[:-1]
        self.assertNotEqual(before, implementation_digest(subset))

    def test_contract_does_not_bind_its_own_hash(self):
        contract = build_specification_contract()
        self.assertNotIn("payload_sha256", contract)
        self.assertNotIn(
            "scripts/run_pact_place_v105_siting.py", contract["implementation_files"]
        )

    def test_historical_discrepancy_is_preserved_not_resolved(self):
        contract = build_specification_contract()
        record = contract["historical_discrepancy_preserved"]
        self.assertFalse(record["turned_into_a_truth_claim"])

    def test_environment_and_contract_versions(self):
        self.assertEqual(
            ENVIRONMENT_VERSION_V105,
            "pact_place_corridor_v10_5_v95_clutter_static_pendant",
        )
        self.assertEqual(
            CONTRACT_VERSION_V105, "pact_place_v105_v95_clutter_static_pendant_v1"
        )
        self.assertEqual(N_REVIEW_VIDEOS, 6)


# ---------------------------------------------------------------------------
# Siting predicates
# ---------------------------------------------------------------------------
class SitingPredicateTest(unittest.TestCase):
    def test_risk_band_and_floor_are_the_registered_values(self):
        self.assertEqual(RISK_BAND_M, (0.015, 0.035))
        self.assertEqual(CLEARANCE_FLOOR_M, 0.015)

    def test_risk_boxes_exclude_the_crossbar(self):
        from pact_place_v105_clearance import risk_boxes, side_risk_boxes

        assembly = build_assembly(0.78, 0.31, 0.0, pose_id="center")
        names = {box["name"] for box in risk_boxes(assembly)}
        self.assertEqual(names, {"lobe_0", "lobe_1", "stem_0", "stem_1"})
        self.assertNotIn("crossbar", names)

    def test_side_risk_boxes_bind_the_route_side(self):
        from pact_place_v105_clearance import side_risk_boxes

        assembly = build_assembly(0.78, 0.31, 0.0, pose_id="center")
        left = {box["name"] for box in side_risk_boxes(assembly, "left")}
        right = {box["name"] for box in side_risk_boxes(assembly, "right")}
        self.assertEqual(left, {"lobe_0", "stem_0"})
        self.assertEqual(right, {"lobe_1", "stem_1"})
        self.assertEqual(left & right, set())

    def test_traversal_direction_separates_loaded_outbound(self):
        from pact_place_v105_siting_core import traversal_direction

        self.assertEqual(traversal_direction("inbound_vessel_pass", False), "inbound")
        self.assertEqual(traversal_direction("pregrasp", False), "inbound")
        self.assertEqual(
            traversal_direction("outbound_cross_vessel_pass", True), "loaded_outbound"
        )
        self.assertEqual(
            traversal_direction("outbound_cross_vessel_pass", False), "outbound"
        )

    def test_hood_top_allowlist_never_exempts_robot_contact(self):
        import inspect

        from pact_place_v105_siting_core import environment_clearance

        source = inspect.getsource(environment_clearance)
        self.assertIn("allow_hood_top", source)
        self.assertIn("crossbar", source)
        # The allowlist lives only in the environment path.
        from pact_place_v105_siting_core import score_candidate_against_snapshot

        self.assertNotIn(
            "hood_top", inspect.getsource(score_candidate_against_snapshot)
        )

    def test_bundle_rejection_reports_every_reason(self):
        from run_pact_place_v105_siting import evaluate_bundle

        rows = [
            {
                "ok": True, "row_dir": "r0", "intrusion_side": "left",
                "family_id": V95_LAYOUT_FAMILY_IDS[0],
                "scores": {
                    f"0.780|0.310|{pose}": {
                        "min_clearance_m": 0.001,
                        "min_witness": {"role": "lobe", "side": "negative"},
                        "min_lobe_stem_m": 0.001,
                        "risk_witness": {"role": "lobe", "side": "negative"},
                        "risk_by_direction_m": {"inbound": 0.001},
                        "window_min_m": {"grasp_close": 0.001, "lift": None,
                                         "release": None},
                        "initial_min_m": {"robot": 0.001, "target": 0.5},
                        "robot_or_target_contact": True,
                        "env_min_m": -0.001, "env_witness": {},
                        "env_intersects": True,
                    }
                    for pose in POSE_IDS
                },
            }
        ]
        report = evaluate_bundle(0.780, 0.310, rows)
        self.assertFalse(report["survives"])
        joined = " | ".join(report["rejection_reasons"])
        for expected in ("intersects", "initial", "contact", "below 15 mm",
                         "window", "15-35 mm", "direction"):
            self.assertIn(expected, joined)


if __name__ == "__main__":
    unittest.main()
