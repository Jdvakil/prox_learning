"""Behavioral tests for the V10.4 review-v2 diagnostic-control repair.

Every expectation is recomputed from source, from live MuJoCo state, or from
file bytes. No test asserts a stored result boolean from an artifact, and the
tests that matter most are the ones that make a tampered input fail.
"""

from __future__ import annotations

import copy
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "scripts", ROOT / "submodules" / "molmospaces"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from pact_place_v104_contract import sha256_payload  # noqa: E402
from pact_place_v104_geometry import (  # noqa: E402
    ALL_GEOMS_V104,
    SCENE_XML_RELATIVE_V104,
    production_assembly,
)
from pact_place_v104_review_v2_contract import (  # noqa: E402
    BRIDGE_ALLOWLIST_NEW_SHA256,
    BRIDGE_ALLOWLIST_OLD_SHA256,
    BRIDGE_ALLOWLIST_PATH,
    CONTRACT_VERSION_V2,
    CONTROL_ANCHOR_TOLERANCE_M,
    CONTROL_ANCHORS,
    CONTROL_ORDER_V2,
    CONTROL_PENETRATION_BAND_M,
    CONTROL_SHIFT_GRID_V2_M,
    CONTROL_SPEC,
    CONTROL_WINDOW_ANCHORS,
    EXECUTED_V1_CONTRACT_SHA256,
    LEFT_LOBE_SECONDARY_STEM_FIRST_FRAME,
    LEFT_LOBE_SECONDARY_STEM_MAX_FRAME,
    LEFT_LOBE_SECONDARY_STEM_MAX_PENETRATION_M,
    EXECUTED_V1_IMPLEMENTATION_SHA256,
    N_REVIEW_V2_VIDEOS,
    PHASE0_V2_ROOT,
    PRODUCTION_SCENE_SHA256,
    REVIEW_V2_ROOT,
    SCENE_METADATA_SHA256,
    SUCCESS_ROLES,
    V1_INPUTS,
    build_provenance_bridge,
    sha256_bytes_of,
    verify_production_rows,
    verify_scene_and_metadata,
    verify_scoped_implementation,
    verify_v1_inputs,
)

PRODUCTION_ROOT = ROOT / "diagnostics_output/pact_place_v104_review_production"


def _mirror_root_with_preflight(destination: Path, preflight: dict) -> Path:
    """A root that is the real tree except for one rewritten preflight.

    The implementation paths are symlinked, not copied, so the only thing the
    verifier sees differently is the binding table under test.
    """
    for name in ("scripts", "tests", "submodules", "configs"):
        source = ROOT / name
        if source.exists():
            (destination / name).symlink_to(source, target_is_directory=True)
    relative = V1_INPUTS["preflight"]["path"]
    target = destination / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(preflight))
    return destination


def _copy_tree_for_tamper(destination: Path) -> Path:
    """A writable mirror of just the paths the bridge reads."""
    for relative in (
        "diagnostics_output/pact_place_v104_preflight/preflight.json",
        "diagnostics_output/pact_place_v104_review_production/production_manifest.json",
        "diagnostics_output/pact_place_v104_review_production/config.json",
        "diagnostics_output/pact_place_v104_causal/causal.json",
        "diagnostics_output/pact_place_v104_causal/raw/left.npz",
        "diagnostics_output/pact_place_v104_causal/raw/right.npz",
        SCENE_XML_RELATIVE_V104,
        "submodules/molmospaces/molmo_spaces/data_generation/custom_scenes/"
        "pact_place_corridor_v10_4_metadata.json",
    ):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    rows_source = PRODUCTION_ROOT / "expert_screen_rows"
    shutil.copytree(
        rows_source,
        destination
        / "diagnostics_output/pact_place_v104_review_production/expert_screen_rows",
    )
    return destination


class ProvenanceBridgeTest(unittest.TestCase):
    def test_bridge_passes_on_the_real_tree(self):
        bridge = build_provenance_bridge()
        self.assertEqual(bridge["failed_sections"], [])
        self.assertTrue(bridge["bridge_passed"])
        self.assertTrue(bridge["verified_from_file_bytes"])
        self.assertFalse(bridge["trusted_embedded_hashes"])

    def test_executed_v1_aggregate_is_distinguished_from_the_live_one(self):
        bridge = build_provenance_bridge()
        self.assertEqual(
            bridge["executed_v1_contract_sha256"], EXECUTED_V1_CONTRACT_SHA256
        )
        self.assertEqual(
            bridge["executed_v1_implementation_sha256"],
            EXECUTED_V1_IMPLEMENTATION_SHA256,
        )
        self.assertTrue(bridge["live_aggregate_is_not_the_executed_aggregate"])
        # The scoped production aggregate is a different, narrower value and
        # must not be confused with either v1 number.
        self.assertNotEqual(
            bridge["scoped_production_sha256"], EXECUTED_V1_IMPLEMENTATION_SHA256
        )

    def test_payload_hash_is_recomputed_not_trusted(self):
        for name, spec in V1_INPUTS.items():
            document = json.loads((ROOT / spec["path"]).read_text())
            recomputed = sha256_payload(
                {k: v for k, v in document.items() if k != "artifact_sha256"}
            )
            self.assertEqual(
                recomputed, spec["payload_sha256"], f"{name} payload hash drifted"
            )

    def test_tampered_payload_with_matching_self_hash_is_rejected(self):
        """The attack the verifier exists for: edit content, re-stamp the hash."""
        with tempfile.TemporaryDirectory() as scratch:
            base = _copy_tree_for_tamper(Path(scratch))
            path = base / V1_INPUTS["production"]["path"]
            document = json.loads(path.read_text())
            document["eligibility"]["min_observed_clearance_m"] = 0.999
            document["artifact_sha256"] = sha256_payload(
                {k: v for k, v in document.items() if k != "artifact_sha256"}
            )
            path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
            report = verify_v1_inputs(base)
            entry = next(
                item for item in report["checks"] if item["input"] == "production"
            )
            self.assertTrue(entry["payload_self_consistent"])
            self.assertFalse(entry["payload_matches_expected"])
            self.assertFalse(entry["passed"])
            self.assertFalse(report["passed"])

    def test_tampered_raw_bytes_alone_are_rejected(self):
        with tempfile.TemporaryDirectory() as scratch:
            base = _copy_tree_for_tamper(Path(scratch))
            path = base / V1_INPUTS["causal"]["path"]
            path.write_text(path.read_text() + "\n")
            report = verify_v1_inputs(base)
            entry = next(
                item for item in report["checks"] if item["input"] == "causal"
            )
            self.assertFalse(entry["file_matches_expected"])
            self.assertFalse(report["passed"])

    def test_allowlist_holds_exactly_one_path(self):
        report = verify_scoped_implementation()
        self.assertEqual(report["allowlist"], [BRIDGE_ALLOWLIST_PATH])
        self.assertEqual(report["allowlist_size"], 1)
        self.assertEqual(len(report["bridged"]), 1)
        bridged = report["bridged"][0]
        self.assertEqual(bridged["path"], BRIDGE_ALLOWLIST_PATH)
        self.assertEqual(bridged["old_sha256"], BRIDGE_ALLOWLIST_OLD_SHA256)
        self.assertEqual(bridged["current_sha256"], BRIDGE_ALLOWLIST_NEW_SHA256)
        self.assertEqual(report["mismatched"], [])
        self.assertTrue(report["passed"])

    def _report_with_binding(self, path: str, digest: str):
        preflight = json.loads((ROOT / V1_INPUTS["preflight"]["path"]).read_text())
        bound = dict(preflight["implementation_files"])
        bound[path] = digest
        preflight["implementation_files"] = bound
        with tempfile.TemporaryDirectory() as scratch:
            base = _mirror_root_with_preflight(Path(scratch), preflight)
            return verify_scoped_implementation(base)

    def test_mirror_fixture_reproduces_the_real_result(self):
        """Guard the fixture itself: an untouched mirror must still pass."""
        preflight = json.loads((ROOT / V1_INPUTS["preflight"]["path"]).read_text())
        with tempfile.TemporaryDirectory() as scratch:
            base = _mirror_root_with_preflight(Path(scratch), preflight)
            report = verify_scoped_implementation(base)
        self.assertTrue(report["passed"])
        self.assertEqual(report["mismatched"], [])
        self.assertEqual(len(report["bridged"]), 1)

    def test_allowlisted_path_still_fails_on_an_unexpected_new_hash(self):
        """The bridge forgives one exact transition, not one filename."""
        report = self._report_with_binding(BRIDGE_ALLOWLIST_PATH, "0" * 64)
        self.assertFalse(report["passed"])
        self.assertEqual(
            [item["path"] for item in report["mismatched"]], [BRIDGE_ALLOWLIST_PATH]
        )
        self.assertEqual(report["bridged"], [])

    def test_a_second_drifted_file_fails_closed(self):
        report = self._report_with_binding(
            "scripts/pact_place_v104_geometry.py", "1" * 64
        )
        self.assertFalse(report["passed"])
        self.assertEqual(
            [item["path"] for item in report["mismatched"]],
            ["scripts/pact_place_v104_geometry.py"],
        )
        # The one legitimate bridge is still recognised alongside the failure.
        self.assertEqual(len(report["bridged"]), 1)

    def test_a_drifted_scene_binding_fails_closed(self):
        report = self._report_with_binding(SCENE_XML_RELATIVE_V104, "3" * 64)
        self.assertFalse(report["passed"])
        self.assertEqual(
            [item["path"] for item in report["mismatched"]], [SCENE_XML_RELATIVE_V104]
        )

    def test_scene_and_metadata_bind_to_the_audited_bytes(self):
        report = verify_scene_and_metadata()
        self.assertEqual(report["observed_scene_sha256"], PRODUCTION_SCENE_SHA256)
        self.assertEqual(report["observed_metadata_sha256"], SCENE_METADATA_SHA256)
        self.assertTrue(report["passed"])

    def test_all_six_rows_reconcile_strict_clean_without_replacement(self):
        report = verify_production_rows()
        self.assertEqual(report["n_rows"], 6)
        self.assertEqual(report["n_clean"], 6)
        self.assertEqual(report["clean_by_side"], {"left": 3, "right": 3})
        self.assertTrue(report["reconciled_all_strict_clean"])
        self.assertFalse(report["replacement_episodes_generated"])
        self.assertTrue(report["config_binding_ok"])
        self.assertTrue(report["config_self_consistent"])
        self.assertEqual(report["pendant_contact_rows"], 0)
        self.assertTrue(report["passed"])

    def test_a_changed_trajectory_byte_fails_the_row_check(self):
        with tempfile.TemporaryDirectory() as scratch:
            base = _copy_tree_for_tamper(Path(scratch))
            rows = (
                base
                / "diagnostics_output/pact_place_v104_review_production"
                / "expert_screen_rows"
            )
            trajectory = sorted(rows.glob("*/trajectory.json"))[0]
            document = json.loads(trajectory.read_text())
            document["row_sha256"] = "2" * 64
            trajectory.write_text(json.dumps(document))
            report = verify_production_rows(base)
            self.assertFalse(report["passed"])
            self.assertTrue(report["failures"])


class SelectionTest(unittest.TestCase):
    def test_registered_success_roles_and_sides(self):
        self.assertEqual(SUCCESS_ROLES, ((0, "left"), (3, "right"), (4, "left")))

    def test_registered_control_source_roles_and_components(self):
        self.assertEqual(
            CONTROL_ORDER_V2,
            ("left_lobe_contact", "right_lobe_contact", "stem_contact"),
        )
        self.assertEqual(
            CONTROL_SPEC["left_lobe_contact"],
            {"source_role_index": 0, "component": "lobe_0"},
        )
        self.assertEqual(
            CONTROL_SPEC["right_lobe_contact"],
            {"source_role_index": 3, "component": "lobe_1"},
        )
        self.assertEqual(
            CONTROL_SPEC["stem_contact"],
            {"source_role_index": 0, "component": "stem_0"},
        )

    def test_control_source_roles_match_the_manifest_sides(self):
        manifest = json.loads(
            (PRODUCTION_ROOT / "production_manifest.json").read_text()
        )
        rows = {int(row["role_index"]): row for row in manifest["rows"]}
        self.assertEqual(rows[0]["intrusion_side"], "left")
        self.assertEqual(rows[3]["intrusion_side"], "right")
        self.assertEqual(rows[4]["intrusion_side"], "left")

    def test_success_roles_are_all_strict_clean_in_the_retained_manifest(self):
        manifest = json.loads(
            (PRODUCTION_ROOT / "production_manifest.json").read_text()
        )
        results = {int(item["role_index"]): item for item in manifest["results"]}
        for role, side in SUCCESS_ROLES:
            self.assertTrue(results[role]["clean_success"], f"role {role}")
            self.assertEqual(results[role]["intrusion_side"], side)


class GridTest(unittest.TestCase):
    def test_grid_is_201_points_from_zero_to_two_hundred_millimetres(self):
        self.assertEqual(len(CONTROL_SHIFT_GRID_V2_M), 201)
        self.assertEqual(CONTROL_SHIFT_GRID_V2_M[0], 0.0)
        self.assertEqual(CONTROL_SHIFT_GRID_V2_M[-1], 0.200)
        deltas = {
            round(b - a, 6)
            for a, b in zip(CONTROL_SHIFT_GRID_V2_M, CONTROL_SHIFT_GRID_V2_M[1:])
        }
        self.assertEqual(deltas, {0.001})

    def test_grid_reaches_every_audited_anchor_shift(self):
        for control, anchor in CONTROL_ANCHORS.items():
            self.assertIn(anchor["shift_m"], CONTROL_SHIFT_GRID_V2_M, control)

    def test_v1_grid_could_not_reach_the_left_lobe_anchor(self):
        """Why v2 exists: the v1 grid stopped 15 mm short of first contact."""
        v1_grid = tuple(round(0.001 * i, 3) for i in range(0, 161))
        self.assertNotIn(CONTROL_ANCHORS["left_lobe_contact"]["shift_m"], v1_grid)
        self.assertIn(
            CONTROL_ANCHORS["left_lobe_contact"]["shift_m"], CONTROL_SHIFT_GRID_V2_M
        )

    def test_every_anchor_penetration_lies_inside_the_registered_band(self):
        low, high = CONTROL_PENETRATION_BAND_M
        for control, anchor in CONTROL_ANCHORS.items():
            self.assertGreaterEqual(anchor["penetration_m"], low, control)
            self.assertLessEqual(anchor["penetration_m"], high, control)


class RigidTranslationTest(unittest.TestCase):
    def test_whole_assembly_translates_not_one_component(self):
        from pact_place_v104_control_certify import inward_sign, shifted_assembly

        assembly = production_assembly()
        sign = inward_sign("lobe_0", assembly)
        moved = shifted_assembly(assembly, sign, 0.175)
        self.assertEqual(len(moved["components"]), len(assembly["components"]))
        for before, after in zip(assembly["components"], moved["components"]):
            self.assertEqual(after["name"], before["name"])
            self.assertAlmostEqual(
                float(after["center_m"][1]) - float(before["center_m"][1]),
                sign * 0.175,
                places=9,
            )
            # x, z and every half-extent are untouched: this is rigid.
            self.assertEqual(after["center_m"][0], before["center_m"][0])
            self.assertEqual(after["center_m"][2], before["center_m"][2])
            self.assertEqual(list(after["half_m"]), list(before["half_m"]))

    def test_inward_sign_points_toward_the_arm_from_each_side(self):
        from pact_place_v104_control_certify import inward_sign

        self.assertEqual(inward_sign("lobe_0"), 1.0)
        self.assertEqual(inward_sign("lobe_1"), -1.0)
        self.assertEqual(inward_sign("stem_0"), 1.0)

    def test_shifting_moves_the_assembly_toward_y_zero(self):
        from pact_place_v104_control_certify import inward_sign, shifted_assembly

        assembly = production_assembly()
        for component in ("lobe_0", "lobe_1"):
            sign = inward_sign(component, assembly)
            moved = shifted_assembly(assembly, sign, 0.100)
            before = next(
                i for i in assembly["components"] if i["name"] == component
            )
            after = next(i for i in moved["components"] if i["name"] == component)
            self.assertLess(
                abs(float(after["center_m"][1])), abs(float(before["center_m"][1]))
            )


class CandidateCapTest(unittest.TestCase):
    """The v1 defect that hid the true worst frame."""

    def test_sound_bound_keeps_frames_beyond_candidate_sixty_four(self):
        # A rigid inward shift of s changes any frame's clearance by at most s,
        # so the admissible set is {frames within base_min + s}, which grows
        # with s and is not a fixed-size prefix.
        per_frame = [0.06 + 0.0001 * index for index in range(600)]
        base_min = min(per_frame)
        order = sorted(range(len(per_frame)), key=lambda i: per_frame[i])
        candidates = [i for i in order if per_frame[i] <= base_min + 0.175 + 1e-6]
        self.assertGreater(len(candidates), 64)

    def test_a_worst_frame_past_index_sixty_four_would_survive_the_bound(self):
        per_frame = [0.10] * 300
        per_frame[212] = 0.061          # the true worst frame, far down the list
        base_min = min(per_frame)
        order = sorted(range(len(per_frame)), key=lambda i: per_frame[i])
        candidates = [i for i in order if per_frame[i] <= base_min + 0.083 + 1e-6]
        self.assertIn(212, candidates)
        self.assertGreater(len(candidates), 64)
        # A fixed 64-item cap would have kept 212 only by luck of ordering;
        # the bound keeps it by construction.
        self.assertEqual(len(candidates), len(per_frame))

    def test_search_records_that_the_candidate_list_was_not_capped(self):
        import inspect

        from pact_place_v104_control_certify import search_control_shift

        source = inspect.getsource(search_control_shift)
        self.assertIn("candidate_list_capped", source)
        self.assertNotIn("[:64]", source)


class WindowTest(unittest.TestCase):
    def test_window_anchors_are_lead_and_trail_consistent(self):
        from pact_place_v104_control_certify import control_window

        window = control_window(
            first_contact_frame=85, max_frame=88, secondary=[], n_frames=543
        )
        self.assertEqual(window["first_frame"], 40)
        self.assertEqual(window["last_frame"], 103)
        self.assertTrue(window["includes_max_penetration_frame"])

    def test_secondary_contact_truncates_the_window_one_frame_early(self):
        from pact_place_v104_control_certify import control_window

        window = control_window(
            first_contact_frame=85,
            max_frame=88,
            secondary=[{"frame": 90, "component": "stem_0", "penetration_m": 0.03815}],
            n_frames=543,
        )
        self.assertEqual(window["first_frame"], 40)
        self.assertEqual(window["last_frame"], 89)
        self.assertEqual(window["n_frames"], 50)
        self.assertEqual(window["truncated_by_secondary_contact_at_frame"], 90)
        self.assertTrue(window["includes_max_penetration_frame"])
        self.assertEqual(window, {**window, "valid": True})

    def test_left_lobe_window_matches_the_registered_anchor(self):
        from pact_place_v104_control_certify import control_window

        window = control_window(
            first_contact_frame=85,
            max_frame=88,
            secondary=[{"frame": 90, "component": "stem_0", "penetration_m": 0.03815}],
            n_frames=543,
        )
        anchor = CONTROL_WINDOW_ANCHORS["left_lobe_contact"]
        self.assertEqual(window["first_frame"], anchor["first_frame"])
        self.assertEqual(window["last_frame"], anchor["last_frame"])
        self.assertEqual(window["n_frames"], anchor["n_frames"])

    def test_excluded_secondary_frames_are_reported_not_silently_dropped(self):
        from pact_place_v104_control_certify import control_window

        window = control_window(
            first_contact_frame=85,
            max_frame=88,
            secondary=[
                {"frame": 90, "component": "stem_0", "penetration_m": 0.03815},
                {"frame": 120, "component": "stem_0", "penetration_m": 0.05},
            ],
            n_frames=543,
        )
        self.assertEqual(window["excluded_secondary_frames"], [90, 120])

    def test_window_is_invalid_when_the_target_never_touches(self):
        from pact_place_v104_control_certify import control_window

        window = control_window(
            first_contact_frame=None, max_frame=0, secondary=[], n_frames=100
        )
        self.assertFalse(window["valid"])

    def test_registered_window_frame_counts_are_internally_consistent(self):
        for control, anchor in CONTROL_WINDOW_ANCHORS.items():
            self.assertEqual(
                anchor["last_frame"] - anchor["first_frame"] + 1,
                anchor["n_frames"],
                control,
            )


class SceneBundleTest(unittest.TestCase):
    def test_bundle_carries_the_v3_and_v5_includes_and_renamed_metadata(self):
        from pact_place_v104_control_certify import (
            DIAGNOSTIC_SCENE_STEM,
            build_scene_bundle,
            inward_sign,
            shifted_assembly,
        )

        assembly = production_assembly()
        moved = shifted_assembly(assembly, inward_sign("lobe_0", assembly), 0.175)
        with tempfile.TemporaryDirectory() as scratch:
            scene = build_scene_bundle(moved, Path(scratch) / "scene")
            directory = scene.parent
            self.assertTrue((directory / "pact_place_corridor_v3.xml").is_file())
            self.assertTrue((directory / "pact_place_corridor_v5.xml").is_file())
            metadata = directory / f"{DIAGNOSTIC_SCENE_STEM}_metadata.json"
            self.assertTrue(metadata.is_file())
            self.assertEqual(
                sha256_bytes_of(metadata),
                SCENE_METADATA_SHA256,
                "the diagnostic metadata must be a byte copy of the production one",
            )
            text = scene.read_text()
            for geom in ALL_GEOMS_V104:
                self.assertIn(geom, text)
            self.assertIn('<include file="pact_place_corridor_v3.xml"/>', text)

    def test_building_a_bundle_never_touches_the_production_scene(self):
        from pact_place_v104_control_certify import (
            build_scene_bundle,
            inward_sign,
            production_scene_unchanged,
            shifted_assembly,
        )

        before = sha256_bytes_of(ROOT / SCENE_XML_RELATIVE_V104)
        assembly = production_assembly()
        moved = shifted_assembly(assembly, inward_sign("lobe_1", assembly), 0.132)
        with tempfile.TemporaryDirectory() as scratch:
            build_scene_bundle(moved, Path(scratch) / "scene")
        after = production_scene_unchanged()
        self.assertEqual(before, after["observed_sha256"])
        self.assertTrue(after["byte_identical"])
        self.assertEqual(after["observed_sha256"], PRODUCTION_SCENE_SHA256)

    def test_diagnostic_scene_differs_from_production_for_a_nonzero_shift(self):
        from pact_place_v104_control_certify import inward_sign, shifted_assembly
        from pact_place_v104_geometry import scene_xml_text

        assembly = production_assembly()
        moved = shifted_assembly(assembly, inward_sign("lobe_0", assembly), 0.175)
        self.assertNotEqual(scene_xml_text(assembly), scene_xml_text(moved))


class LiveDiagnosticSceneTest(unittest.TestCase):
    """Compile a shifted assembly through MuJoCo and check it stays static."""

    @classmethod
    def setUpClass(cls):
        import os

        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        os.environ.setdefault("MLSPACES_ASSETS_DIR", str(ROOT / "assets"))

    def test_shifted_assembly_compiles_static_with_enclosing_bounds(self):
        import mujoco
        import numpy as np

        from pact_place_v104_control_certify import (
            build_scene_bundle,
            inward_sign,
            shifted_assembly,
        )
        from pact_place_v104_geometry import PENDANT_BODY_V104

        assembly = production_assembly()
        moved = shifted_assembly(assembly, inward_sign("lobe_0", assembly), 0.175)
        with tempfile.TemporaryDirectory() as scratch:
            scene = build_scene_bundle(moved, Path(scratch) / "scene")
            model = mujoco.MjModel.from_xml_path(str(scene))
        body_id = int(model.body(PENDANT_BODY_V104).id)
        self.assertEqual(int(model.body_dofnum[body_id]), 0)
        self.assertEqual(int(model.body_jntnum[body_id]), 0)
        self.assertLess(int(model.body_mocapid[body_id]), 0)
        by_geom = {item["geom"]: item for item in moved["components"]}
        for name in ALL_GEOMS_V104:
            gid = int(model.geom(name).id)
            expected_half = np.asarray(by_geom[name]["half_m"], dtype=float)
            expected_pos = np.asarray(by_geom[name]["center_m"], dtype=float)
            self.assertTrue(
                np.allclose(model.geom_size[gid], expected_half, atol=1e-12), name
            )
            self.assertTrue(
                np.allclose(model.geom_pos[gid], expected_pos, atol=1e-9), name
            )
            aabb = np.asarray(model.geom_aabb[gid], dtype=float)
            self.assertTrue(np.all(aabb[3:] >= expected_half - 1e-12), name)
            self.assertGreaterEqual(
                float(model.geom_rbound[gid]),
                float(np.linalg.norm(expected_half)) - 1e-9,
                name,
            )

    def test_compiled_bounds_track_the_shift_rather_than_going_stale(self):
        """The V10.2 defect must not reappear: bounds move with the geometry."""
        import mujoco
        import numpy as np

        from pact_place_v104_control_certify import (
            build_scene_bundle,
            inward_sign,
            shifted_assembly,
        )

        assembly = production_assembly()
        sign = inward_sign("lobe_0", assembly)
        positions = {}
        for shift in (0.0, 0.175):
            moved = shifted_assembly(assembly, sign, shift)
            with tempfile.TemporaryDirectory() as scratch:
                scene = build_scene_bundle(moved, Path(scratch) / "scene")
                model = mujoco.MjModel.from_xml_path(str(scene))
            gid = int(model.geom("pact_clutter_mount_v104_lobe_0_g").id)
            positions[shift] = float(model.geom_pos[gid][1])
            self.assertGreater(float(model.geom_rbound[gid]), 0.001)
        self.assertAlmostEqual(
            positions[0.175] - positions[0.0], sign * 0.175, places=9
        )


class CertificateCarriesGeometryTest(unittest.TestCase):
    """A control must render the geometry it was certified against."""

    def test_renderer_has_no_silent_production_fallback_for_controls(self):
        import inspect

        import run_pact_place_v104_review_v2 as review

        source = inspect.getsource(review.render_clip)
        self.assertNotIn('job.get("assembly") or production_assembly()', source)
        self.assertIn('assembly = job["assembly"]', source)

    def test_certify_control_stores_the_shifted_assembly(self):
        import inspect

        from pact_place_v104_control_certify import certify_control

        source = inspect.getsource(certify_control)
        self.assertIn('"assembly": assembly,', source)

    def test_secondary_summary_separates_first_frame_from_deepest_frame(self):
        from pact_place_v104_control_certify import _secondary_summary

        secondary = [
            {"frame": 90, "component": "stem_0", "penetration_m": 0.000103},
            {"frame": 193, "component": "stem_0", "penetration_m": 0.03815},
            {"frame": 212, "component": "stem_0", "penetration_m": 0.01},
        ]
        summary = _secondary_summary(secondary)
        self.assertEqual(summary["first_frame"], LEFT_LOBE_SECONDARY_STEM_FIRST_FRAME)
        self.assertAlmostEqual(
            summary["max_penetration_m"],
            LEFT_LOBE_SECONDARY_STEM_MAX_PENETRATION_M,
            places=5,
        )
        self.assertEqual(
            summary["max_penetration_frame"], LEFT_LOBE_SECONDARY_STEM_MAX_FRAME
        )
        # The two are different frames; conflating them is the error this guards.
        self.assertNotEqual(summary["first_frame"], summary["max_penetration_frame"])

    def test_no_secondary_contact_reports_cleanly(self):
        from pact_place_v104_control_certify import _secondary_summary

        self.assertEqual(_secondary_summary([]), {"any": False})


class ContactClassificationTest(unittest.TestCase):
    def test_pendant_geoms_classify_as_mounted_fixture(self):
        from molmo_spaces.tasks.pact_place_contact_audit import classify_contact

        for geom in ALL_GEOMS_V104:
            pair = {
                "geom1": "robot_0/fr3_link7_collision",
                "geom2": geom,
                "body1": "robot_0/fr3_link7",
                "body2": "pact_clutter_mount_v104",
            }
            self.assertEqual(classify_contact(pair), "mounted_fixture", geom)

    def test_bench_clutter_still_classifies_as_clutter(self):
        from molmo_spaces.tasks.pact_place_contact_audit import classify_contact

        pair = {
            "geom1": "robot_0/fr3_link7_collision",
            "geom2": "pact_clutter_l0_g",
            "body1": "robot_0/fr3_link7",
            "body2": "pact_clutter_l0",
        }
        self.assertEqual(classify_contact(pair), "clutter")


class Phase0V2ApprovalTest(unittest.TestCase):
    def _bindings(self):
        return {
            "contract_version_v2": CONTRACT_VERSION_V2,
            "scoped_production_sha256": "a" * 64,
            "review_manifest_sha256": "b" * 64,
        }

    def _videos(self):
        return [f"v{index}.mp4" for index in range(N_REVIEW_V2_VIDEOS)]

    def _approval(self, **overrides):
        approval = {
            "decision": "approve_phase0",
            "created_by_agent": False,
            "reviewed_videos": self._videos(),
            **self._bindings(),
        }
        approval.update(overrides)
        return approval

    def _assert(self, approval):
        from run_pact_place_v104_phase0_v2 import assert_phase0_v2_approval

        assert_phase0_v2_approval(
            approval, self._bindings(), video_names=self._videos()
        )

    def test_a_complete_record_passes(self):
        self._assert(self._approval())

    def test_missing_record_is_refused(self):
        with self.assertRaises(PermissionError):
            self._assert(None)

    def test_wrong_decision_is_refused(self):
        with self.assertRaises(PermissionError):
            self._assert(self._approval(decision="looks_fine"))

    def test_agent_created_record_is_refused(self):
        with self.assertRaises(PermissionError):
            self._assert(self._approval(created_by_agent=True))

    def test_partial_bindings_are_refused(self):
        approval = self._approval()
        del approval["review_manifest_sha256"]
        with self.assertRaises(PermissionError):
            self._assert(approval)

    def test_every_stale_binding_is_refused_individually(self):
        for key in self._bindings():
            with self.subTest(binding=key):
                with self.assertRaises(PermissionError):
                    self._assert(self._approval(**{key: "0" * 64}))

    def test_extra_video_is_refused(self):
        with self.assertRaises(PermissionError):
            self._assert(
                self._approval(reviewed_videos=self._videos() + ["extra.mp4"])
            )

    def test_missing_video_is_refused(self):
        with self.assertRaises(PermissionError):
            self._assert(self._approval(reviewed_videos=self._videos()[:-1]))

    def test_video_list_must_be_a_list(self):
        with self.assertRaises(PermissionError):
            self._assert(self._approval(reviewed_videos="all of them"))

    def test_verifier_recomputes_rather_than_trusting_embedded_hashes(self):
        from run_pact_place_v104_phase0_v2 import _recomputed_payload_sha256

        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch) / "artifact.json"
            document = {"value": 1, "artifact_sha256": "9" * 64}
            path.write_text(json.dumps(document))
            recomputed = _recomputed_payload_sha256(path)
            self.assertNotEqual(recomputed, "9" * 64)
            self.assertEqual(recomputed, sha256_payload({"value": 1}))

    def test_gate_thresholds_are_unchanged_from_v1(self):
        from pact_place_v104_contract import (
            GATE_MIN_CLEARANCE_M,
            MIN_GATE_CLEAN,
            MIN_GATE_CLEAN_PER_SIDE,
            N_GATE_ROWS,
        )

        self.assertEqual(N_GATE_ROWS, 24)
        self.assertEqual(MIN_GATE_CLEAN, 20)
        self.assertEqual(MIN_GATE_CLEAN_PER_SIDE, 9)
        self.assertEqual(GATE_MIN_CLEARANCE_M, 0.015)

    def test_gate_runner_creates_no_directory_before_approval(self):
        import inspect

        import run_pact_place_v104_phase0_v2 as gate

        source = inspect.getsource(gate.main)
        approval_at = source.index("assert_phase0_v2_approval")
        mkdir_at = source.index("output_root.mkdir")
        self.assertLess(
            approval_at, mkdir_at, "the gate must validate before it creates anything"
        )


class DocumentedSchemaMatchesVerifierTest(unittest.TestCase):
    """REVIEW.md must document exactly the bindings the gate demands.

    An owner who follows the documented schema verbatim must not then be
    rejected for a missing binding.
    """

    REVIEW_ROOT = ROOT / REVIEW_V2_ROOT

    def _documented_keys(self):
        text = (self.REVIEW_ROOT / "REVIEW.md").read_text()
        start = text.index("```json")
        end = text.index("```", start + 7)
        block = json.loads(text[start + 7 : end])
        return block, set(block) - {"decision", "created_by_agent", "reviewed_videos"}

    def test_documented_bindings_match_the_verifier_exactly(self):
        if not (self.REVIEW_ROOT / "REVIEW.md").is_file():
            self.skipTest("review packet not published")
        from run_pact_place_v104_phase0_v2 import expected_bindings_v2

        _, documented = self._documented_keys()
        expected = set(expected_bindings_v2(self.REVIEW_ROOT))
        self.assertEqual(
            documented,
            expected,
            f"documented-only={sorted(documented - expected)} "
            f"verifier-only={sorted(expected - documented)}",
        )

    def test_documented_binding_values_are_current(self):
        if not (self.REVIEW_ROOT / "REVIEW.md").is_file():
            self.skipTest("review packet not published")
        from run_pact_place_v104_phase0_v2 import expected_bindings_v2

        block, _ = self._documented_keys()
        for key, value in expected_bindings_v2(self.REVIEW_ROOT).items():
            self.assertEqual(block[key], value, key)

    def test_documented_video_list_is_the_published_inventory(self):
        if not (self.REVIEW_ROOT / "REVIEW.md").is_file():
            self.skipTest("review packet not published")
        block, _ = self._documented_keys()
        published = sorted(
            item.name for item in (self.REVIEW_ROOT / "videos").glob("*.mp4")
        )
        self.assertEqual(sorted(block["reviewed_videos"]), published)
        self.assertEqual(len(published), N_REVIEW_V2_VIDEOS)

    def test_the_documented_record_actually_passes_the_verifier(self):
        """End to end: paste the schema, fill nothing in, and it validates."""
        if not (self.REVIEW_ROOT / "REVIEW.md").is_file():
            self.skipTest("review packet not published")
        from run_pact_place_v104_phase0_v2 import (
            assert_phase0_v2_approval,
            expected_bindings_v2,
            review_video_hashes,
        )

        block, _ = self._documented_keys()
        assert_phase0_v2_approval(
            block,
            expected_bindings_v2(self.REVIEW_ROOT),
            video_names=sorted(review_video_hashes(self.REVIEW_ROOT)),
        )

    def test_the_documented_record_is_marked_owner_authored(self):
        if not (self.REVIEW_ROOT / "REVIEW.md").is_file():
            self.skipTest("review packet not published")
        block, _ = self._documented_keys()
        self.assertFalse(block["created_by_agent"])
        self.assertEqual(block["decision"], "approve_phase0")


class PublishedPacketTest(unittest.TestCase):
    ROOT_DIR = ROOT / REVIEW_V2_ROOT

    def setUp(self):
        if not self.ROOT_DIR.is_dir():
            self.skipTest("review packet not published")

    def test_exactly_six_videos_three_and_three(self):
        manifest = json.loads((self.ROOT_DIR / "review_manifest.json").read_text())
        self.assertEqual(manifest["n_videos"], N_REVIEW_V2_VIDEOS)
        self.assertEqual(manifest["n_production_successes"], 3)
        self.assertEqual(manifest["n_diagnostic_controls"], 3)
        self.assertEqual(
            len(list((self.ROOT_DIR / "videos").glob("*.mp4"))), N_REVIEW_V2_VIDEOS
        )

    def test_published_video_bytes_match_the_manifest(self):
        manifest = json.loads((self.ROOT_DIR / "review_manifest.json").read_text())
        for name, digest in manifest["video_sha256"].items():
            self.assertEqual(
                sha256_bytes_of(self.ROOT_DIR / "videos" / name), digest, name
            )

    def test_control_windows_match_the_registered_anchors(self):
        manifest = json.loads((self.ROOT_DIR / "review_manifest.json").read_text())
        by_control = {
            item["control"]: item for item in manifest["control_certificates"]
        }
        for control, anchor in CONTROL_WINDOW_ANCHORS.items():
            window = by_control[control]["window"]
            self.assertEqual(window["first_frame"], anchor["first_frame"], control)
            self.assertEqual(window["last_frame"], anchor["last_frame"], control)
            self.assertEqual(window["n_frames"], anchor["n_frames"], control)

    def test_certified_controls_match_the_audited_anchors(self):
        manifest = json.loads((self.ROOT_DIR / "review_manifest.json").read_text())
        by_control = {
            item["control"]: item for item in manifest["control_certificates"]
        }
        for control, anchor in CONTROL_ANCHORS.items():
            certificate = by_control[control]
            self.assertTrue(certificate["certified"], control)
            self.assertEqual(certificate["shift_m"], anchor["shift_m"], control)
            self.assertLessEqual(
                abs(certificate["penetration_m"] - anchor["penetration_m"]),
                CONTROL_ANCHOR_TOLERANCE_M,
                control,
            )
            self.assertEqual(certificate["max_frame"], anchor["max_frame"], control)
            self.assertEqual(
                certificate["parity"]["limiting_robot_body"],
                anchor["limiting_robot_body"],
                control,
            )

    def test_every_authorization_field_is_false(self):
        for name in ("review_manifest.json", "review_preflight.json",
                     "provenance_bridge.json", "control_certificates.json"):
            document = json.loads((self.ROOT_DIR / name).read_text())
            for field in (
                "authorizes_phase0",
                "authorizes_gate",
                "authorizes_collection",
                "authorizes_training",
                "authorizes_evaluation",
                "phase0_passed",
                "human_approval_present",
            ):
                self.assertFalse(document[field], f"{name}:{field}")

    def test_only_the_manifest_is_eligible_for_human_review(self):
        manifest = json.loads((self.ROOT_DIR / "review_manifest.json").read_text())
        self.assertTrue(manifest["eligible_for_human_review"])

    def test_artifacts_are_create_only(self):
        from pact_place_v104_review_v2_contract import (
            ImmutableArtifactError,
            write_immutable_create_only,
            write_immutable_text_create_only,
        )

        with self.assertRaises(ImmutableArtifactError):
            write_immutable_create_only(
                self.ROOT_DIR / "review_manifest.json", {"clobbered": True}
            )
        with self.assertRaises(ImmutableArtifactError):
            write_immutable_text_create_only(self.ROOT_DIR / "REVIEW.md", "clobbered")

    def test_republishing_over_the_final_directory_is_refused(self):
        import subprocess

        completed = subprocess.run(
            [sys.executable, str(ROOT / "scripts/run_pact_place_v104_review_v2.py")],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("refusing to publish over an existing", completed.stderr)


class NoApprovalArtifactTest(unittest.TestCase):
    def test_no_human_approval_file_exists_anywhere(self):
        for root in (ROOT / REVIEW_V2_ROOT, ROOT / PHASE0_V2_ROOT):
            self.assertFalse(
                (root / "human_approval.json").exists(),
                f"the agent must never create {root}/human_approval.json",
            )

    def test_no_v2_source_file_writes_a_human_approval(self):
        for name in (
            "scripts/pact_place_v104_review_v2_contract.py",
            "scripts/pact_place_v104_control_certify.py",
            "scripts/run_pact_place_v104_review_v2.py",
            "scripts/run_pact_place_v104_phase0_v2.py",
        ):
            text = (ROOT / name).read_text()
            for verb in ("write_text", "write_immutable_create_only", "json.dump"):
                for line in text.splitlines():
                    if verb in line and "human_approval" in line:
                        self.fail(f"{name} appears to write an approval: {line}")


if __name__ == "__main__":
    unittest.main()
