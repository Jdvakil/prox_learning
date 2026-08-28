"""Behavioral tests for the V10.4 first-shot static pendant.

Every expectation is recomputed from source, from live MuJoCo state, or from
file bytes. No test asserts a stored result boolean from an artifact.
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

from pact_place_v104_contract import (  # noqa: E402
    CONTRACT_VERSION,
    ENVIRONMENT_VERSION,
    GATE_MIN_CLEARANCE_M,
    GATE_STREAM,
    ImmutableArtifactError,
    MIN_GATE_CLEAN,
    MIN_GATE_CLEAN_PER_SIDE,
    MIN_REVIEW_CLEAN,
    MIN_REVIEW_CLEAN_PER_SIDE,
    N_GATE_PER_SIDE,
    N_GATE_ROWS,
    N_REVIEW_PER_SIDE,
    N_REVIEW_ROWS,
    REVIEW_MIN_CLEARANCE_M,
    REVIEW_STREAM,
    SAMPLER_CLASS,
    assert_phase0_approval,
    build_contract,
    empty_authorization,
    gate_eligibility,
    is_clean_success,
    review_eligibility,
    row_defects,
    sha256_bytes_of,
    verify_protected_artifacts,
    write_immutable_create_only,
)
from pact_place_v104_geometry import (  # noqa: E402
    ALL_GEOMS_V104,
    CROSSBAR_GEOM_V104,
    PENDANT_BODY_V104,
    SCENE_XML_RELATIVE_V104,
    assembly_expectations,
    corner_assemblies,
    production_assembly,
    scene_xml_sha256,
    scene_xml_text,
)
from pact_place_v104_runtime import (  # noqa: E402
    INITIAL_FREE_SPACE_SPEED_CAP_M_S,
    SpeedAmendmentError,
    TASK_HORIZON_V104,
    apply_initial_free_space_speed_cap,
    locate_initial_free_space_segment,
    plan_signature,
    verify_plan_matches_baseline,
)

V3_SCENE = ROOT / "submodules/molmospaces/molmo_spaces/data_generation/custom_scenes/pact_place_corridor_v3.xml"


def _compiled_model():
    import mujoco

    return mujoco.MjModel.from_xml_path(str(ROOT / SCENE_XML_RELATIVE_V104))


class _Seg:
    def __init__(self, name, speed, start=None, end=None):
        self.name = name
        self.speed = speed
        self.start_pose = np.eye(4) if start is None else start
        self.end_pose = np.eye(4) if end is None else end


class _Seq:
    def __init__(self, segments, holding=False):
        self._move_segments = list(segments)
        self.is_holding_object = holding


class _Grip:
    def __init__(self, target_open):
        self.target_open = target_open


def _fake_primitives():
    from molmo_spaces.policy.solvers.object_manipulation.base_object_manipulation_planner_policy import (  # noqa: E501
        GripperAction,
        TCPMoveSequence,
    )

    _Seq.__bases__  # keep linters quiet
    seq = TCPMoveSequence.__new__(TCPMoveSequence)
    seq._move_segments = [
        _Seg("pregrasp", 0.20),
        _Seg("pregrasp", 0.045),
        _Seg("grasp", 0.08),
    ]
    seq.is_holding_object = False
    open_action = GripperAction.__new__(GripperAction)
    open_action.target_open = True
    close_action = GripperAction.__new__(GripperAction)
    close_action.target_open = False
    later = TCPMoveSequence.__new__(TCPMoveSequence)
    later._move_segments = [_Seg("lift", 0.08)]
    later.is_holding_object = True
    return [open_action, seq, close_action, later]


# 1. geometry, symmetry, attachment, serialization
class GeometryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.assembly = production_assembly()
        self.by_name = {i["name"]: i for i in self.assembly["components"]}

    def test_exact_registered_coordinates(self) -> None:
        expected = {
            "lobe_0": ((0.78, -0.34, 1.01), (0.010, 0.010, 0.030)),
            "lobe_1": ((0.78, 0.34, 1.01), (0.010, 0.010, 0.030)),
            "stem_0": ((0.78, -0.35, 1.2725), (0.006, 0.006, 0.2325)),
            "stem_1": ((0.78, 0.35, 1.2725), (0.006, 0.006, 0.2325)),
            "crossbar": ((0.78, 0.0, 1.510), (0.006, 0.356, 0.005)),
        }
        self.assertEqual(set(self.by_name), set(expected))
        for name, (center, half) in expected.items():
            self.assertEqual(list(self.by_name[name]["center_m"]), list(center), name)
            self.assertEqual(list(self.by_name[name]["half_m"]), list(half), name)

    def test_derived_consequences(self) -> None:
        report = assembly_expectations(self.assembly)
        self.assertEqual(report["lobe_bottom_z_m"], [0.98, 0.98])
        self.assertEqual(report["lobe_top_z_m"], [1.04, 1.04])
        self.assertAlmostEqual(report["shelf_to_lobe_gap_m"], 0.26, places=9)
        self.assertEqual(report["stem_square_m"], [0.012, 0.012])
        self.assertEqual(report["stem_square_y_m"], [0.012, 0.012])
        self.assertEqual(report["stem_top_z_m"], [1.505, 1.505])
        self.assertAlmostEqual(report["crossbar_top_z_m"], 1.515, places=9)
        self.assertAlmostEqual(report["crossbar_height_m"], 0.010, places=9)
        self.assertAlmostEqual(report["crossbar_square_x_m"], 0.012, places=9)

    def test_symmetry_in_y_and_side_independence(self) -> None:
        for a, b in (("lobe_0", "lobe_1"), ("stem_0", "stem_1")):
            self.assertEqual(
                self.by_name[a]["center_m"][1], -self.by_name[b]["center_m"][1]
            )
            self.assertEqual(self.by_name[a]["half_m"], self.by_name[b]["half_m"])
        self.assertEqual(self.by_name["crossbar"]["center_m"][1], 0.0)
        self.assertTrue(self.assembly["symmetric_in_y"])
        self.assertTrue(self.assembly["identical_on_both_panel_sides"])

    def test_stems_join_outward_lobe_faces_and_crossbar_spans_them(self) -> None:
        for slot, sign in ((0, -1.0), (1, 1.0)):
            lobe = self.by_name[f"lobe_{slot}"]
            stem = self.by_name[f"stem_{slot}"]
            face = lobe["center_m"][1] + sign * lobe["half_m"][1]
            self.assertAlmostEqual(stem["center_m"][1], face, places=9)
            self.assertAlmostEqual(
                stem["aabb_lo_m"][2], lobe["aabb_hi_m"][2], places=9
            )
        bar = self.by_name["crossbar"]
        for slot in (0, 1):
            self.assertLessEqual(
                abs(self.by_name[f"stem_{slot}"]["center_m"][1]), bar["half_m"][1] + 1e-9
            )

    def test_serialization_is_deterministic_and_matches_disk(self) -> None:
        text = scene_xml_text(self.assembly)
        self.assertEqual(scene_xml_sha256(self.assembly), scene_xml_sha256(self.assembly))
        self.assertEqual(
            sha256_bytes_of(ROOT / SCENE_XML_RELATIVE_V104), scene_xml_sha256(self.assembly)
        )
        for item in self.assembly["components"]:
            self.assertIn(f'name="{item["geom"]}"', text)
        self.assertIn('<include file="pact_place_corridor_v3.xml"/>', text)

    def test_corners_are_rigid_five_millimetre_translations(self) -> None:
        corners = corner_assemblies(self.assembly)
        self.assertEqual(len(corners), 8)
        for corner in corners:
            shift = np.asarray(corner["corner_translation_m"], dtype=float)
            self.assertTrue(np.allclose(np.abs(shift), 0.005))
            for base, moved in zip(self.assembly["components"], corner["components"]):
                self.assertTrue(
                    np.allclose(
                        np.asarray(moved["center_m"]) - np.asarray(base["center_m"]),
                        shift,
                        atol=1e-12,
                    )
                )
                self.assertEqual(moved["half_m"], base["half_m"])


# 2, 3, 4. compiled-static body, identical visible/collision, hood attachment
class CompiledSceneTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = _compiled_model()
        self.assembly = production_assembly()

    def test_body_is_static_with_no_joint_freejoint_or_mocap(self) -> None:
        body_id = int(self.model.body(PENDANT_BODY_V104).id)
        self.assertEqual(int(self.model.body_dofnum[body_id]), 0)
        self.assertEqual(int(self.model.body_jntnum[body_id]), 0)
        self.assertLess(int(self.model.body_mocapid[body_id]), 0)
        text = (ROOT / SCENE_XML_RELATIVE_V104).read_text()
        body = text.split(f'<body name="{PENDANT_BODY_V104}"')[1].split("</body>")[0]
        for token in ("<joint", "<freejoint", 'mocap="true"'):
            self.assertNotIn(token, body)

    def test_visible_and_collision_geometry_are_identical(self) -> None:
        names = [self.model.geom(i).name or "" for i in range(int(self.model.ngeom))]
        by_geom = {i["geom"]: i for i in self.assembly["components"]}
        pendant_like = [n for n in names if n.startswith("pact_clutter_mount_v104")]
        self.assertEqual(sorted(pendant_like), sorted(ALL_GEOMS_V104))
        for geom, item in by_geom.items():
            self.assertEqual(names.count(geom), 1, geom)
            gid = int(self.model.geom(geom).id)
            self.assertTrue(
                np.allclose(self.model.geom_size[gid], item["half_m"], atol=1e-12)
            )
            self.assertTrue(
                np.allclose(self.model.geom_pos[gid], item["center_m"], atol=1e-12)
            )
            self.assertEqual(int(self.model.geom_contype[gid]), 8)
            self.assertEqual(int(self.model.geom_conaffinity[gid]), 15)
            self.assertAlmostEqual(float(self.model.geom_rgba[gid][3]), 1.0)

    def test_compiled_bounds_need_no_runtime_repair(self) -> None:
        for item in self.assembly["components"]:
            gid = int(self.model.geom(item["geom"]).id)
            half = np.asarray(item["half_m"], dtype=float)
            aabb = np.asarray(self.model.geom_aabb[gid], dtype=float)
            self.assertTrue(np.all(aabb[3:] >= half - 1e-12), item["geom"])
            self.assertGreaterEqual(
                float(self.model.geom_rbound[gid]), float(np.linalg.norm(half)) - 1e-9
            )

    def test_pendant_cannot_move_under_repeated_forward(self) -> None:
        import mujoco

        data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, data)
        first = {
            g: np.asarray(data.geom_xpos[int(self.model.geom(g).id)]).copy()
            for g in ALL_GEOMS_V104
        }
        for _ in range(30):
            mujoco.mj_forward(self.model, data)
        for geom, value in first.items():
            self.assertTrue(
                np.allclose(data.geom_xpos[int(self.model.geom(geom).id)], value, atol=0.0)
            )

    def test_only_the_designed_hood_face_touches(self) -> None:
        import mujoco
        from pact_geom_distance import true_distance

        data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, data)
        pendant = {int(self.model.geom(g).id) for g in ALL_GEOMS_V104}
        touches = []
        for geom in ALL_GEOMS_V104:
            gid = int(self.model.geom(geom).id)
            for other in range(int(self.model.ngeom)):
                if other in pendant:
                    continue
                if int(self.model.geom_contype[other]) == 0 and int(
                    self.model.geom_conaffinity[other]
                ) == 0:
                    continue
                if float(true_distance(self.model, data, [gid], [other])) <= 0.0:
                    touches.append((geom, self.model.geom(other).name or ""))
        self.assertEqual(touches, [(CROSSBAR_GEOM_V104, "hood_top")])

    def test_pendant_contacts_classify_mounted_fixture(self) -> None:
        from molmo_spaces.tasks.pact_place_contact_audit import classify_contact

        for geom in ALL_GEOMS_V104:
            for other_body, other_geom in (
                ("robot_0/fr3_link7", "robot_0/fr3_link7_collision"),
                ("cavity_obj_Cup_10", "cavity_obj_Cup_10_g"),
            ):
                pair = {
                    "geom1": geom, "geom2": other_geom,
                    "body1": PENDANT_BODY_V104, "body2": other_body,
                    "root1": PENDANT_BODY_V104, "root2": other_body,
                }
                self.assertEqual(classify_contact(pair), "mounted_fixture")


# 8. speed amendment
class SpeedAmendmentTest(unittest.TestCase):
    def test_caps_exactly_one_segment_by_primitive_order(self) -> None:
        primitives = _fake_primitives()
        located = locate_initial_free_space_segment(primitives)
        self.assertEqual(located["primitive_index"], 1)
        self.assertEqual(located["segment_index"], 0)
        self.assertEqual(located["gripper_close_primitive_index"], 2)
        before = plan_signature(primitives)
        record = apply_initial_free_space_speed_cap(primitives)
        after = plan_signature(primitives)
        self.assertTrue(record["applied"])
        self.assertAlmostEqual(record["original_speed_m_s"], 0.20)
        self.assertAlmostEqual(record["cap_m_s"], INITIAL_FREE_SPACE_SPEED_CAP_M_S)
        self.assertEqual(record["n_segments_changed"], 1)
        comparison = verify_plan_matches_baseline(before, after)
        self.assertTrue(comparison["poses_identical"])
        self.assertTrue(comparison["exactly_one_speed_change"])
        self.assertAlmostEqual(primitives[1]._move_segments[1].speed, 0.045)
        self.assertAlmostEqual(primitives[3]._move_segments[0].speed, 0.08)

    def test_refuses_when_the_segment_cannot_be_bound(self) -> None:
        from molmo_spaces.policy.solvers.object_manipulation.base_object_manipulation_planner_policy import (  # noqa: E501
            GripperAction,
        )

        only_open = GripperAction.__new__(GripperAction)
        only_open.target_open = True
        with self.assertRaises(SpeedAmendmentError):
            locate_initial_free_space_segment([only_open])
        close = GripperAction.__new__(GripperAction)
        close.target_open = False
        with self.assertRaises(SpeedAmendmentError):
            locate_initial_free_space_segment([close])

    def test_baseline_comparison_detects_a_moved_pose(self) -> None:
        primitives = _fake_primitives()
        before = plan_signature(primitives)
        primitives[1]._move_segments[1].end_pose = np.eye(4) * 2.0
        after = plan_signature(primitives)
        comparison = verify_plan_matches_baseline(before, after)
        self.assertFalse(comparison["poses_identical"])

    def test_registered_horizon(self) -> None:
        self.assertEqual(TASK_HORIZON_V104, 1050)
        self.assertAlmostEqual(INITIAL_FREE_SPACE_SPEED_CAP_M_S, 0.12)


# 7, 8 (live). V6c route unchanged and gated on the marker.
class LiveRoutePreservationTest(unittest.TestCase):
    def test_v6c_plan_unchanged_and_only_v104_is_capped(self) -> None:
        import os
        import shutil
        import tempfile

        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("MLSPACES_ASSETS_DIR", str(ROOT / "assets"))
        os.environ.pop("DISPLAY", None)
        from molmo_spaces.data_generation.pipeline import (
            cleanup_episode_resources,
            setup_policy,
        )
        from molmo_spaces.env.abstract_sensors import SensorSuite
        from run_pact_place_expert_screen import _make_config

        config_doc = json.loads(
            (ROOT / "configs/pact_place_corridor_v6c.json").read_text()
        )
        row = config_doc["expert_screen_rows"][0]
        row_dir = (
            ROOT / "diagnostics_output/pact_place_corridor_v6c/expert_screen_rows"
            / f"00_{row['episode_id'][:16]}"
        )
        seed = int(json.loads((row_dir / "result.json").read_text())["selected_seed"]["seed_u32"])

        def build(scene, sampler_class):
            scratch = Path(tempfile.mkdtemp())
            task = policy = sampler = None
            try:
                config = _make_config(
                    scratch / "d.json", scene_xml=scene, sampler_class=sampler_class
                )
                sampler = config.task_sampler_config.task_sampler_class(config)
                sampler.seed_task_sampling(seed)
                sampler.set_pact_manifest_row(row)
                task = sampler.sample_task(house_index=1)
                task._sensor_suite = SensorSuite(
                    [task._sensor_suite.sensors[u] for u in ("qpos", "tcp_pose")]
                )
                policy = setup_policy(config, task, None, None)
                task.reset()
                primitives = policy._compute_trajectory()
                return (
                    plan_signature(primitives),
                    dict(getattr(policy, "_pact_place_v104_speed_amendment", {}) or {}),
                    int(config.task_horizon),
                )
            finally:
                cleanup_episode_resources(
                    task=task, policy=policy, task_sampler=sampler,
                    preloaded_policy=None, close_task_sampler=sampler is not None,
                )
                shutil.rmtree(scratch, ignore_errors=True)

        base_plan, base_amend, base_horizon = build(V3_SCENE, "PactPlaceCorridorV3Sampler")
        v104_plan, v104_amend, v104_horizon = build(
            ROOT / SCENE_XML_RELATIVE_V104, SAMPLER_CLASS
        )
        self.assertFalse(base_amend.get("applied"))
        self.assertEqual(base_horizon, 900)
        self.assertTrue(v104_amend.get("applied"))
        self.assertEqual(v104_horizon, TASK_HORIZON_V104)
        comparison = verify_plan_matches_baseline(base_plan, v104_plan)
        self.assertTrue(comparison["poses_identical"], comparison["failures"][:3])
        self.assertTrue(comparison["exactly_one_speed_change"])
        change = comparison["speed_changes"][0]
        self.assertEqual(change["primitive_index"], 1)
        self.assertEqual(change["segment_index"], 0)
        self.assertAlmostEqual(change["from_m_s"], 0.20)
        self.assertAlmostEqual(change["to_m_s"], INITIAL_FREE_SPACE_SPEED_CAP_M_S)


# 10, 11, 15. manifests, eligibility, gate counting
class ContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = build_contract()

    def test_deterministic_and_self_hashed(self) -> None:
        again = build_contract()
        self.assertEqual(self.contract["contract_sha256"], again["contract_sha256"])
        self.assertEqual(self.contract["contract_version"], CONTRACT_VERSION)
        self.assertEqual(self.contract["environment_version"], ENVIRONMENT_VERSION)

    def test_streams_are_disjoint_and_side_balanced(self) -> None:
        review, gate = self.contract["review_rows"], self.contract["gate_rows"]
        self.assertEqual((len(review), len(gate)), (N_REVIEW_ROWS, N_GATE_ROWS))
        self.assertFalse(
            {r["episode_id"] for r in review} & {r["episode_id"] for r in gate}
        )
        self.assertFalse(
            {r["task_seed_u32"] for r in review} & {r["task_seed_u32"] for r in gate}
        )
        for rows, per_side in ((review, N_REVIEW_PER_SIDE), (gate, N_GATE_PER_SIDE)):
            for side in ("left", "right"):
                self.assertEqual(
                    sum(1 for r in rows if r["intrusion_side"] == side), per_side
                )
        self.assertEqual(self.contract["review_stream"], REVIEW_STREAM)
        self.assertEqual(self.contract["gate_stream"], GATE_STREAM)

    def test_provenance_is_byte_level(self) -> None:
        report = verify_protected_artifacts(self.contract["protected_artifacts"])
        self.assertTrue(report["verified_from_file_bytes"])
        self.assertTrue(report["passed"])
        self.assertGreaterEqual(report["n_artifacts"], 50)

    def test_provenance_detects_a_changed_byte(self) -> None:
        tampered = dict(self.contract["protected_artifacts"])
        key = sorted(tampered)[0]
        tampered[key] = "0" * 64
        report = verify_protected_artifacts(tampered)
        self.assertFalse(report["passed"])
        self.assertEqual(report["mismatches"][0]["path"], key)

    def test_authorizations_start_false(self) -> None:
        for key, value in empty_authorization().items():
            self.assertFalse(value, key)
            self.assertFalse(self.contract[key], key)


# 16. immutable writer
class ImmutableWriterTest(unittest.TestCase):
    def test_refuses_to_replace_an_existing_artifact(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "a.json"
            first = write_immutable_create_only(target, {"value": 1})
            self.assertTrue(target.is_file())
            self.assertEqual(
                json.loads(target.read_text())["artifact_sha256"], first
            )
            with self.assertRaises(ImmutableArtifactError):
                write_immutable_create_only(target, {"value": 2})
            self.assertEqual(json.loads(target.read_text())["value"], 1)


# 6, 11, 15. row admission and eligibility
class AdmissionTest(unittest.TestCase):
    def _result(self, **overrides):
        payload = {
            "status": "complete",
            "episode_id": "e",
            "row_sha256": "r",
            "task_success": True,
            "grasp_phase_success": True,
            "place_phase_success": True,
            "clean_success": True,
            "clutter_stability_events": [],
            "bow_fallback_taken": False,
            "terminal_tracking": {"sequential_ik_failures": 0},
            "scene_params": {"pact_place_environment_version": ENVIRONMENT_VERSION},
            "contact_audit": {
                "contact_class_totals": {
                    "mounted_fixture": 0, "clutter": 0, "hazard_bar": 0,
                    "other_environment": 0, "place_receptacle": 4,
                },
                "place_receptacle_outside_placement_entries": 0,
            },
            "pact_v104_speed_amendment": {"applied": True, "n_segments_changed": 1},
            "pact_v104_frame_telemetry": {
                "n_frames": 500, "n_frames_measured": 500,
                "min_clearance_m": 0.055, "pendant_contact_frames": 0,
                "per_component_min_clearance_m": {
                    "lobe_0": 0.055, "lobe_1": 0.20, "stem_0": 0.12,
                    "stem_1": 0.25, "crossbar": 0.30,
                },
            },
        }
        payload.update(overrides)
        return payload

    def test_clean_row_passes(self) -> None:
        self.assertEqual(row_defects(self._result(), min_clearance_m=REVIEW_MIN_CLEARANCE_M), [])
        self.assertTrue(is_clean_success(self._result(), min_clearance_m=REVIEW_MIN_CLEARANCE_M))

    def test_pendant_contact_makes_the_row_unclean(self) -> None:
        r = self._result()
        r["contact_audit"]["contact_class_totals"]["mounted_fixture"] = 2
        self.assertIn("mounted_fixture_contact", row_defects(r, min_clearance_m=REVIEW_MIN_CLEARANCE_M))
        r = self._result()
        r["pact_v104_frame_telemetry"]["pendant_contact_frames"] = 1
        self.assertIn("pendant_contact", row_defects(r, min_clearance_m=REVIEW_MIN_CLEARANCE_M))

    def test_clearance_floor_and_speed_amendment(self) -> None:
        r = self._result()
        r["pact_v104_frame_telemetry"]["min_clearance_m"] = 0.019
        r["pact_v104_frame_telemetry"]["per_component_min_clearance_m"]["lobe_0"] = 0.019
        defects = row_defects(r, min_clearance_m=REVIEW_MIN_CLEARANCE_M)
        self.assertIn("clearance_below_floor", defects)
        self.assertIn("component_clearance_below_floor:lobe_0", defects)
        # the same row passes the looser gate floor
        self.assertNotIn(
            "clearance_below_floor", row_defects(r, min_clearance_m=GATE_MIN_CLEARANCE_M)
        )
        r = self._result()
        r["pact_v104_speed_amendment"] = {"applied": False}
        self.assertIn(
            "speed_amendment_not_applied", row_defects(r, min_clearance_m=REVIEW_MIN_CLEARANCE_M)
        )

    def test_fail_closed_statuses(self) -> None:
        for status, code in (
            ("sampling_failure", "sampling_failure"),
            ("infrastructure_failure", "infrastructure_failure"),
            ("running", "nonterminal"),
        ):
            self.assertEqual(
                row_defects({"status": status}, min_clearance_m=REVIEW_MIN_CLEARANCE_M), [code]
            )

    def test_review_eligibility_thresholds(self) -> None:
        rows = [
            {"role_index": i, "episode_id": f"e{i}", "row_sha256": f"r{i}",
             "intrusion_side": "left" if i % 2 == 0 else "right"}
            for i in range(N_REVIEW_ROWS)
        ]
        results = []
        for i, row in enumerate(rows):
            r = self._result(episode_id=row["episode_id"], row_sha256=row["row_sha256"])
            if i == 5:
                r["task_success"] = False
            results.append(r)
        report = review_eligibility(rows, results)
        self.assertEqual(report["clean_successes"], MIN_REVIEW_CLEAN)
        self.assertTrue(report["production_pack_passed"])
        # a second failure on the same side breaks the per-side rule
        results[3]["task_success"] = False
        report = review_eligibility(rows, results)
        self.assertFalse(report["production_pack_passed"])
        self.assertLess(report["clean_by_side"]["right"], MIN_REVIEW_CLEAN_PER_SIDE)

    def test_gate_counting_and_no_replacement(self) -> None:
        rows = [
            {"role_index": i, "episode_id": f"g{i}", "row_sha256": f"s{i}",
             "intrusion_side": "left" if i % 2 == 0 else "right"}
            for i in range(N_GATE_ROWS)
        ]
        results = [
            self._result(episode_id=row["episode_id"], row_sha256=row["row_sha256"])
            for row in rows
        ]
        for index in (0, 1, 2, 3):
            results[index]["task_success"] = False
        report = gate_eligibility(rows, results)
        self.assertEqual(report["clean_successes"], MIN_GATE_CLEAN)
        self.assertTrue(report["phase0_passed"])
        self.assertGreaterEqual(report["clean_by_side"]["left"], MIN_GATE_CLEAN_PER_SIDE)
        results[4]["task_success"] = False
        self.assertFalse(gate_eligibility(rows, results)["phase0_passed"])
        # a missing row cannot be replaced: it fails reconciliation
        self.assertFalse(gate_eligibility(rows, results[:-1])["phase0_passed"])
        for key in ("authorizes_collection", "authorizes_training", "authorizes_evaluation"):
            self.assertFalse(gate_eligibility(rows, results)[key])


# 14. approval contract
class ApprovalTest(unittest.TestCase):
    def _expected(self):
        return {"contract_sha256": "a" * 64, "scene_sha256": "b" * 64}

    def test_missing_or_rejecting_record(self) -> None:
        with self.assertRaises(PermissionError):
            assert_phase0_approval(None, self._expected())
        with self.assertRaises(PermissionError):
            assert_phase0_approval({"decision": "reject"}, self._expected())

    def test_agent_created_record_is_refused(self) -> None:
        approval = {"decision": "approve_phase0", "created_by_agent": True, **self._expected()}
        with self.assertRaises(PermissionError):
            assert_phase0_approval(approval, self._expected())

    def test_missing_and_stale_bindings(self) -> None:
        with self.assertRaises(PermissionError):
            assert_phase0_approval(
                {"decision": "approve_phase0", "contract_sha256": "a" * 64},
                self._expected(),
            )
        with self.assertRaises(PermissionError):
            assert_phase0_approval(
                {"decision": "approve_phase0", "contract_sha256": "a" * 64,
                 "scene_sha256": "c" * 64},
                self._expected(),
            )
        assert_phase0_approval(
            {"decision": "approve_phase0", **self._expected()}, self._expected()
        )


if __name__ == "__main__":
    unittest.main()
