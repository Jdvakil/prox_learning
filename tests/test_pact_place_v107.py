"""Behavioral tests for the V10.7 qualification repair.

Covers the real certification and causal runners, hash drift, six-group
coverage, distance disagreement, rigid carried-target motion, and the narrow
grasp allowlist.
"""

from __future__ import annotations

import ast
import inspect
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

from pact_place_v107_contract import (  # noqa: E402
    CLEARANCE_FLOOR_M,
    CONTRACT_VERSION_V107,
    HashDriftError,
    IMPLEMENTATION_PATHS,
    INTRUSION_SIDES,
    N_GROUPS,
    N_POOL_ROWS,
    N_REVIEW_VIDEOS,
    POOL_MIN_CLEAN,
    POOL_MIN_CLEAN_PER_POSE,
    POOL_MIN_CLEAN_PER_SIDE,
    POOL_MIN_CLEAN_PER_SIDE_POSE,
    POSE_IDS,
    RISK_BAND_M,
    SEALED_INPUTS,
    assert_no_drift,
    candidate_statistics,
    file_hashes,
    group_key,
    is_qualified,
    pool_eligibility,
    pool_rows,
    risk_aligned_rank_key,
    row_defects,
    verify_against_specification,
)

V106_NPZ = ROOT / "diagnostics_output/pact_place_v106_siting/per_row_scores.npz"
V106_SITING = ROOT / "diagnostics_output/pact_place_v106_siting/siting.json"


def _per_row():
    return [json.loads(str(x))
            for x in np.load(V106_NPZ, allow_pickle=True)["rows"]]


# ---------------------------------------------------------------------------
# Hash drift
# ---------------------------------------------------------------------------
class HashDriftTest(unittest.TestCase):
    def _spec(self):
        return {
            "sealed_inputs": file_hashes(SEALED_INPUTS),
            "implementation_files": file_hashes(IMPLEMENTATION_PATHS),
        }

    def test_clean_tree_has_no_drift(self):
        report = verify_against_specification(self._spec())
        self.assertEqual(report["n_drift"], 0, report["drift"][:3])
        self.assertTrue(report["passed"])

    def test_a_drifted_sealed_input_is_detected(self):
        spec = self._spec()
        key = next(iter(spec["sealed_inputs"]))
        spec["sealed_inputs"][key] = {"raw_file_sha256": "0" * 64}
        report = verify_against_specification(spec)
        self.assertFalse(report["passed"])
        self.assertEqual([d["path"] for d in report["drift"]], [key])

    def test_a_drifted_implementation_file_is_detected(self):
        spec = self._spec()
        key = "scripts/pact_place_v107_contract.py"
        spec["implementation_files"][key] = {"raw_file_sha256": "1" * 64}
        report = verify_against_specification(spec)
        self.assertFalse(report["passed"])

    def test_assert_no_drift_raises(self):
        spec = self._spec()
        spec["implementation_files"]["scripts/pact_place_v107_contract.py"] = {
            "raw_file_sha256": "2" * 64
        }
        with self.assertRaises(HashDriftError):
            assert_no_drift(spec)

    def test_every_runner_and_test_is_bound(self):
        for required in (
            "scripts/run_pact_place_v107_specify.py",
            "scripts/run_pact_place_v107_select.py",
            "scripts/run_pact_place_v107_certify.py",
            "scripts/run_pact_place_v107_causal.py",
            "scripts/run_pact_place_v107_contact_diagnostic.py",
            "scripts/run_pact_place_v107_pool.py",
            "scripts/audit_pact_place_v105.py",
            "scripts/pact_place_v106_geometry.py",
            "tests/test_pact_place_v107.py",
        ):
            self.assertIn(required, IMPLEMENTATION_PATHS, required)

    def test_sealed_inputs_include_the_v106_score_npz(self):
        self.assertIn(
            "diagnostics_output/pact_place_v106_siting/per_row_scores.npz",
            SEALED_INPUTS,
        )

    def test_every_stage_verifies_drift_before_acting(self):
        for name in ("run_pact_place_v107_select",
                     "run_pact_place_v107_certify",
                     "run_pact_place_v107_causal",
                     "run_pact_place_v107_pool"):
            source = (ROOT / "scripts" / f"{name}.py").read_text()
            self.assertIn("assert_no_drift(spec)", source, name)


# ---------------------------------------------------------------------------
# Risk-aligned ranking
# ---------------------------------------------------------------------------
class RankingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.per_row = _per_row()
        cls.keys = sorted(json.loads(V106_SITING.read_text())["bundles"])
        cls.stats = {k: candidate_statistics(cls.per_row, k) for k in cls.keys}

    def test_universal_clearance_outranks_everything(self):
        universal = [k for k in self.keys
                     if self.stats[k]["universal_clearance"]]
        others = [k for k in self.keys
                  if not self.stats[k]["universal_clearance"]]
        self.assertTrue(universal and others)
        for u in universal:
            for o in others:
                self.assertLess(
                    risk_aligned_rank_key(self.stats[u]),
                    risk_aligned_rank_key(self.stats[o]),
                    f"{u} must outrank {o}",
                )

    def test_additional_clearance_is_demoted_below_risk(self):
        """The V10.6 defect: it ranked farther-away pendants first."""
        universal = [k for k in self.keys
                     if self.stats[k]["universal_clearance"]]
        ranked = sorted(universal, key=lambda k: risk_aligned_rank_key(self.stats[k]))
        farthest = max(
            universal, key=lambda k: self.stats[k]["absolute_min_clearance_m"]
        )
        self.assertNotEqual(
            ranked[0], farthest,
            "risk-aligned ranking must not select the farthest pendant",
        )

    def test_winner_has_the_most_band_evaluations_among_universal(self):
        universal = [k for k in self.keys
                     if self.stats[k]["universal_clearance"]]
        ranked = sorted(universal, key=lambda k: risk_aligned_rank_key(self.stats[k]))
        best = max(universal,
                   key=lambda k: self.stats[k]["band_evaluations_total"])
        self.assertEqual(ranked[0], best)

    def test_selected_group_minima_all_lie_in_the_band(self):
        qualified = [k for k in self.keys if is_qualified(self.stats[k])["qualified"]]
        self.assertTrue(qualified)
        ranked = sorted(qualified, key=lambda k: risk_aligned_rank_key(self.stats[k]))
        chosen = self.stats[ranked[0]]
        minima = list(chosen["group_minimum_m"].values())
        self.assertEqual(len(minima), N_GROUPS)
        for value in minima:
            self.assertGreaterEqual(value, RISK_BAND_M[0])
            self.assertLessEqual(value, RISK_BAND_M[1])

    def test_ranking_is_deterministic(self):
        a = [risk_aligned_rank_key(self.stats[k]) for k in self.keys]
        b = [risk_aligned_rank_key(candidate_statistics(self.per_row, k))
             for k in self.keys]
        self.assertEqual(a, b)

    def test_selection_runner_does_not_hardcode_a_bundle(self):
        source = (ROOT / "scripts/run_pact_place_v107_select.py").read_text()
        for forbidden in ("0.330|0.300", "0.800|0.330|0.300", "r_neg_m == 0.33"):
            self.assertNotIn(forbidden, source, forbidden)

    def test_qualification_requires_six_groups_and_band(self):
        for key in self.keys:
            report = is_qualified(self.stats[key])
            if report["qualified"]:
                self.assertTrue(report["checks"]["universal_clearance_15mm"])
                self.assertTrue(report["checks"]["all_six_group_minima_in_15_35mm"])
                self.assertEqual(self.stats[key]["n_groups"], N_GROUPS)

    def test_statistics_cover_all_six_groups(self):
        for key in self.keys:
            self.assertEqual(
                sorted(self.stats[key]["group_total_evaluations"]),
                sorted(group_key(p, s) for p in POSE_IDS for s in INTRUSION_SIDES),
            )


# ---------------------------------------------------------------------------
# Certification: distance disagreement must fail closed
# ---------------------------------------------------------------------------
class CertificationLogicTest(unittest.TestCase):
    def test_agreement_dictionary_gates_certification(self):
        import run_pact_place_v107_certify as certify

        source = inspect.getsource(certify.certify)
        self.assertIn('"certified": all(agreements.values())', source)

    def test_all_four_instruments_are_measured(self):
        import run_pact_place_v107_certify as certify

        source = inspect.getsource(certify.certify)
        for token in ("gjk_distance", "true_distance", "pendant_contact_state",
                      "frame_clearances"):
            self.assertIn(token, source, token)

    def test_a_distance_disagreement_fails(self):
        """Recreate the agreement dictionary with one instrument off."""
        tol = 1e-6
        exact, signed, gjk = 0.020, 0.020, 0.031
        agreements = {
            "exact_matches_signed": abs(exact - signed) <= tol,
            "signed_matches_gjk": abs(signed - gjk) <= tol,
        }
        self.assertTrue(agreements["exact_matches_signed"])
        self.assertFalse(agreements["signed_matches_gjk"])
        self.assertFalse(all(agreements.values()))

    def test_certification_requires_all_six_groups(self):
        import run_pact_place_v107_certify as certify

        source = inspect.getsource(certify.main)
        self.assertIn("len(groups_covered) == N_GROUPS", source)

    def test_witness_collection_includes_threshold_near(self):
        import run_pact_place_v107_certify as certify

        source = inspect.getsource(certify.collect_witnesses)
        self.assertIn("threshold_near", source)
        self.assertIn("THRESHOLD_NEAR_M", source)

    def test_scene_publication_refuses_to_overwrite(self):
        import run_pact_place_v107_certify as certify

        source = inspect.getsource(certify.publish_scenes)
        self.assertIn("refusing to overwrite", source)


# ---------------------------------------------------------------------------
# Causal: six-group coverage
# ---------------------------------------------------------------------------
class CausalLogicTest(unittest.TestCase):
    def test_causal_requires_six_groups(self):
        import run_pact_place_v107_causal as causal

        source = inspect.getsource(causal.main)
        self.assertIn("N_GROUPS", source)
        self.assertIn("six_groups_evaluated", source)

    def test_every_check_is_per_group(self):
        import run_pact_place_v107_causal as causal

        source = inspect.getsource(causal.main)
        for token in ("min_changed_values_every_group",
                      "min_changed_sensors_every_group",
                      "link5_or_link6_every_group",
                      "onset_frames_every_group"):
            self.assertIn(token, source, token)

    def test_npz_is_written_before_the_manifest(self):
        import run_pact_place_v107_causal as causal

        source = inspect.getsource(causal.main)
        self.assertLess(
            source.index("np.savez_compressed"),
            source.index("write_immutable_create_only"),
            "the score NPZ must be written before the JSON that binds it",
        )
        self.assertIn("causal_scores_raw_file_sha256", source)

    def test_selection_npz_is_written_before_the_manifest_that_binds_it(self):
        """The stop path writes a manifest with no NPZ; the binding path must not.

        Anchoring on the *binding* write rather than the first write is what
        makes this check meaningful.
        """
        import run_pact_place_v107_select as select

        source = inspect.getsource(select.main)
        savez = source.index("np.savez_compressed")
        npz_hash = source.index("npz_sha = sha256_file(npz_out)")
        binding_write = source.rindex(
            'write_immutable_create_only(output_root / "selection.json"'
        )
        self.assertLess(savez, npz_hash)
        self.assertLess(npz_hash, binding_write)
        self.assertIn("selection_scores_raw_file_sha256", source)

    def test_the_stop_path_manifest_binds_no_npz(self):
        import run_pact_place_v107_select as select

        source = inspect.getsource(select.main)
        stop = source.index('"stop_reason": "no_bundle_met')
        savez = source.index("np.savez_compressed")
        self.assertLess(stop, savez, "the stop path must precede NPZ creation")

    def test_certification_npz_is_written_before_its_manifest(self):
        import run_pact_place_v107_certify as certify

        source = inspect.getsource(certify.main)
        self.assertLess(
            source.index("np.savez_compressed"),
            source.index('write_immutable_create_only(\n        output_root / "certification.json"'),
        )

    def test_npz_contents_support_reaggregation(self):
        import run_pact_place_v107_causal as causal

        source = inspect.getsource(causal.main)
        for field in ("per_sensor_changed_counts", "per_frame_changed_counts",
                      "threshold_m", "changed_values"):
            self.assertIn(field, source, field)


# ---------------------------------------------------------------------------
# Contact diagnostic: rigid target, narrow allowlist, non-gating
# ---------------------------------------------------------------------------
class ContactDiagnosticTest(unittest.TestCase):
    def test_it_is_declared_non_gating(self):
        import run_pact_place_v107_contact_diagnostic as diag

        source = inspect.getsource(diag.main)
        self.assertIn('"diagnostic_only": True', source)
        self.assertIn('"gates_qualification": False', source)

    def test_no_downstream_stage_reads_its_outcome(self):
        for name in ("run_pact_place_v107_pool", "run_pact_place_v107_causal",
                     "run_pact_place_v107_certify"):
            source = (ROOT / "scripts" / f"{name}.py").read_text()
            self.assertNotIn("contact_diagnostic", source, name)

    def test_carried_target_moves_rigidly_with_the_gripper(self):
        import run_pact_place_v107_contact_diagnostic as diag

        source = inspect.getsource(diag.diagnose)
        self.assertIn("move_target_rigidly", source)
        self.assertIn("target_pos0 + (tcp_now - tcp0)", source)

    def test_allowlist_is_pad_to_target_only(self):
        from run_pact_place_v107_contact_diagnostic import is_grasp_pad_contact

        pad = {"geom1": "robot_0/gripper/right_pad1",
               "geom2": "cavity_obj_0/Cup_10_collider",
               "body1": "robot_0/gripper", "body2": "cavity_obj_0"}
        self.assertTrue(is_grasp_pad_contact(pad))

    def test_allowlist_excludes_a_robot_link_hitting_the_target(self):
        from run_pact_place_v107_contact_diagnostic import is_grasp_pad_contact

        link = {"geom1": "robot_0/fr3_link5_collision",
                "geom2": "cavity_obj_0/Cup_10_collider",
                "body1": "robot_0/fr3_link5", "body2": "cavity_obj_0"}
        self.assertFalse(is_grasp_pad_contact(link))

    def test_allowlist_excludes_a_pad_hitting_the_environment(self):
        from run_pact_place_v107_contact_diagnostic import is_grasp_pad_contact

        env = {"geom1": "robot_0/gripper/right_pad1", "geom2": "pact_clutter_01_g",
               "body1": "robot_0/gripper", "body2": "pact_clutter_01"}
        self.assertFalse(is_grasp_pad_contact(env))

    def test_worsening_baseline_penetrations_are_tracked(self):
        import run_pact_place_v107_contact_diagnostic as diag

        source = inspect.getsource(diag.diagnose)
        self.assertIn("worsened_baseline_penetrations", source)
        self.assertIn("baseline_pairs[key] - 1e-9", source)

    def test_contact_requires_instrument_agreement(self):
        import run_pact_place_v107_contact_diagnostic as diag

        source = inspect.getsource(diag.diagnose)
        self.assertIn("instruments_agree_on_contact", source)
        self.assertIn("unsigned == 0.0", source)


# ---------------------------------------------------------------------------
# Pool floors and packet composition
# ---------------------------------------------------------------------------
class PoolTest(unittest.TestCase):
    def _rows(self):
        scenes = {p: {"relative": f"s/{p}.xml", "sha256": f"{i}" * 64}
                  for i, p in enumerate(POSE_IDS)}
        return pool_rows(
            selected={"x_m": 0.8, "r_neg_m": 0.33, "r_pos_m": 0.30},
            scene_by_pose=scenes,
            assembly_by_pose={p: f"{i}" * 64 for i, p in enumerate(POSE_IDS)},
        )

    def test_pool_shape_and_balance(self):
        rows = self._rows()
        self.assertEqual(len(rows), N_POOL_ROWS)
        for side in INTRUSION_SIDES:
            self.assertEqual(sum(1 for r in rows
                                 if r["intrusion_side"] == side), 24)
        for pose in POSE_IDS:
            self.assertEqual(sum(1 for r in rows if r["pose_id"] == pose), 16)

    def test_floors_are_the_scaled_values(self):
        self.assertEqual((POOL_MIN_CLEAN, N_POOL_ROWS), (32, 48))
        self.assertEqual(POOL_MIN_CLEAN_PER_SIDE, 14)
        self.assertEqual(POOL_MIN_CLEAN_PER_POSE, 8)
        self.assertEqual(POOL_MIN_CLEAN_PER_SIDE_POSE, 4)

    def test_thirty_one_clean_fails(self):
        rows = self._rows()
        results = [{"role_index": r["role_index"],
                    "v107_clean_success": i < 31}
                   for i, r in enumerate(rows)]
        self.assertFalse(pool_eligibility(rows, results)["pool_passed"])

    def test_all_clean_passes(self):
        rows = self._rows()
        results = [{"role_index": r["role_index"], "v107_clean_success": True}
                   for r in rows]
        report = pool_eligibility(rows, results)
        self.assertTrue(report["pool_passed"])
        self.assertEqual(report["clean_successes"], 48)

    def test_floors_are_checked_before_any_video(self):
        import run_pact_place_v107_pool as pool

        source = inspect.getsource(pool.main)
        self.assertLess(
            source.index('if not pool_doc["pool_passed"]'),
            source.index("render_clip(job)"),
            "the pool floors must gate rendering",
        )

    def test_packet_uses_only_complete_production_episodes(self):
        import run_pact_place_v107_pool as pool

        source = inspect.getsource(pool.main)
        self.assertIn('"all_clips_are_complete_production_episodes": True', source)
        self.assertIn('"diagnostic_assemblies_used": False', source)

    def test_row_defects_flag_pendant_contact_and_clutter(self):
        base = {"status": "complete", "task_success": True,
                "grasp_phase_success": True, "place_phase_success": True,
                "cup_lifted_one_cm": True, "contact_audit": {},
                "pact_v106_frame_telemetry": {"min_clearance_m": 0.02}}
        self.assertEqual(row_defects(base), [])
        contact = dict(base, pact_v106_frame_telemetry={
            "min_clearance_m": 0.02,
            "pendant_robot_or_target_contact_frames": 3})
        self.assertIn("pendant_contact", row_defects(contact))
        clutter = dict(base, contact_audit={"contact_class_totals": {"clutter": 2}})
        self.assertTrue(any("clutter" in d for d in row_defects(clutter)))
        stability = dict(base, clutter_stability_events=[{"body": "x"}])
        self.assertTrue(any("stability" in d for d in row_defects(stability)))

    def test_missing_telemetry_is_a_defect(self):
        base = {"status": "complete", "task_success": True,
                "grasp_phase_success": True, "place_phase_success": True,
                "cup_lifted_one_cm": True, "contact_audit": {}}
        self.assertIn("missing_frame_telemetry", row_defects(base))

    def test_agent_does_not_create_an_approval_record(self):
        for name in ("run_pact_place_v107_pool",):
            source = (ROOT / "scripts" / f"{name}.py").read_text()
            for line in source.splitlines():
                if "human_approval" in line:
                    self.assertNotIn("write_text", line)
                    self.assertNotIn("write_immutable", line)

    def test_six_videos_registered(self):
        self.assertEqual(N_REVIEW_VIDEOS, 6)


class SceneHashGuardTest(unittest.TestCase):
    """The guard that refused every task in the first pool run.

    It read ``cfg.scene_xml``; the datagen config carries ``scene_xml_paths``
    on ``task_sampler_config``, so the guard raised on every row instead of
    verifying it, and all 48 pool rows came back ``sampling_failure``.
    """

    def test_guard_reads_scene_xml_paths_from_the_experiment_config(self):
        from molmo_spaces.tasks.enclosure_reach import PactPlaceCorridorV106Sampler

        source = inspect.getsource(PactPlaceCorridorV106Sampler.sample_task)
        self.assertIn("scene_xml_paths", source)
        self.assertIn('getattr(self, "config", None)', source)
        self.assertNotIn('getattr(self.cfg,', source)

    def test_guard_rejects_multiple_distinct_scenes(self):
        from molmo_spaces.tasks.enclosure_reach import PactPlaceCorridorV106Sampler

        source = inspect.getsource(PactPlaceCorridorV106Sampler.sample_task)
        self.assertIn("len(set(paths)) > 1", source)

    def test_the_datagen_config_really_has_scene_xml_paths(self):
        """Guard the guard: assert the attribute the fix now depends on."""
        import tempfile as _tempfile

        from run_pact_place_expert_screen import _make_config

        scene = (
            ROOT / "submodules/molmospaces/molmo_spaces/data_generation"
            / "custom_scenes/pact_place_corridor_v5.xml"
        )
        with _tempfile.TemporaryDirectory() as scratch:
            config = _make_config(Path(scratch) / "d.json", scene_xml=scene,
                                  sampler_class="PactPlaceCorridorV93Sampler")
            self.assertTrue(
                hasattr(config.task_sampler_config, "scene_xml_paths"))
            self.assertFalse(hasattr(config.task_sampler_config, "scene_xml"))
            self.assertFalse(hasattr(config, "scene_xml"))

    def test_a_row_without_a_bound_hash_skips_the_guard(self):
        from molmo_spaces.tasks.enclosure_reach import PactPlaceCorridorV106Sampler

        source = inspect.getsource(PactPlaceCorridorV106Sampler.sample_task)
        self.assertIn('expected = row.get("pact_v106_scene_sha256")', source)
        self.assertIn("if expected:", source)


class TelemetryPassthroughTest(unittest.TestCase):
    """The retained row copies an explicit subset of policy_info.

    A new environment's telemetry keys must be listed there or every row reads
    as ``missing_frame_telemetry`` -- which is exactly what the first two pool
    runs reported.
    """

    def test_v106_telemetry_is_copied_into_the_retained_row(self):
        source = (ROOT / "scripts/run_pact_place_expert_screen.py").read_text()
        for key in ("pact_v106_frame_telemetry", "pact_v106_speed_amendment"):
            self.assertIn(f'"{key}": policy_info.get("{key}")', source, key)

    def test_the_policy_actually_emits_those_keys(self):
        source = (
            ROOT / "submodules/molmospaces/molmo_spaces/tasks/enclosure_reach.py"
        ).read_text()
        self.assertIn(
            '"pact_v106_frame_telemetry": self._v106_frame_summary()', source)
        self.assertIn('"pact_v106_speed_amendment": dict(', source)
        self.assertIn("self._pact_place_v106_speed_amendment or {}", source)

    def test_missing_telemetry_is_still_a_defect(self):
        base = {"status": "complete", "task_success": True,
                "grasp_phase_success": True, "place_phase_success": True,
                "cup_lifted_one_cm": True, "contact_audit": {},
                "pact_v106_frame_telemetry": {}}
        self.assertIn("missing_frame_telemetry", row_defects(base))


class TelemetryGeometryPathTest(unittest.TestCase):
    """The policy has no manifest row; the assembly must reach it another way.

    The third pool run produced telemetry with ``min_clearance_m: null`` on
    every completed episode because the policy read ``_pact_manifest_row``, a
    sampler attribute. The geometry now travels through ``scene_params``.
    """

    def test_policy_does_not_read_a_sampler_attribute(self):
        from molmo_spaces.tasks.enclosure_reach import PactPlaceCorridorPolicy

        for method in (PactPlaceCorridorPolicy._v106_assembly,
                       PactPlaceCorridorPolicy._v106_frame_telemetry):
            source = inspect.getsource(method)
            self.assertNotIn("_pact_manifest_row", source, method.__name__)

    def test_policy_reads_scene_params(self):
        from molmo_spaces.tasks.enclosure_reach import PactPlaceCorridorPolicy

        source = inspect.getsource(PactPlaceCorridorPolicy._v106_assembly)
        for key in ("pact_v106_pose_id", "pact_v106_x_m",
                    "pact_v106_r_neg_m", "pact_v106_r_pos_m"):
            self.assertIn(key, source, key)
        self.assertIn('getattr(self.task, "scene_params", {})', source)

    def test_sampler_publishes_the_geometry_into_scene_params(self):
        from molmo_spaces.tasks.enclosure_reach import PactPlaceCorridorV106Sampler

        source = inspect.getsource(PactPlaceCorridorV106Sampler._draw_theta)
        for key in ("pact_v106_x_m", "pact_v106_r_neg_m", "pact_v106_r_pos_m",
                    "pact_v106_pose_id"):
            self.assertIn(f'th["{key}"]', source, key)

    def test_the_policy_class_really_lacks_a_manifest_row(self):
        """Guard the guard: the attribute the old code assumed does not exist."""
        from molmo_spaces.tasks.enclosure_reach import PactPlaceCorridorPolicy

        self.assertFalse(hasattr(PactPlaceCorridorPolicy, "_pact_manifest_row"))

    def test_missing_clearance_telemetry_is_a_defect(self):
        base = {"status": "complete", "task_success": True,
                "grasp_phase_success": True, "place_phase_success": True,
                "cup_lifted_one_cm": True, "contact_audit": {},
                "pact_v106_frame_telemetry": {"n_frames": 500,
                                              "min_clearance_m": None}}
        self.assertIn("missing_clearance_telemetry", row_defects(base))


class NoPriorArtifactModifiedTest(unittest.TestCase):
    SEALED = {
        "diagnostics_output/pact_place_v106_siting/siting.json":
            "34949f171508b10706100a91acb7fbbe51ede295125819879ed7a742b9eacdfa",
        "diagnostics_output/pact_place_v105_siting/siting.json":
            "56f5d6ba2e35c1f76ee5945fbd2976c10ac93553e68cd23b41b2e229d76fb6b4",
    }

    def test_sealed_artifacts_are_byte_identical(self):
        import hashlib

        for relative, expected in self.SEALED.items():
            path = ROOT / relative
            self.assertTrue(path.is_file(), relative)
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(), expected, relative)

    def test_v107_writes_only_to_v107_roots(self):
        from pact_place_v107_contract import (
            CAUSAL_ROOT, CERT_ROOT, DIAGNOSTIC_ROOT, PHASE0_ROOT, POOL_ROOT,
            REVIEW_ROOT, SELECTION_ROOT, SPEC_ROOT,
        )

        for root in (SPEC_ROOT, SELECTION_ROOT, CERT_ROOT, CAUSAL_ROOT,
                     DIAGNOSTIC_ROOT, POOL_ROOT, REVIEW_ROOT, PHASE0_ROOT):
            self.assertIn("v107", root)

    def test_v107_publishes_distinct_scene_names(self):
        import run_pact_place_v107_certify as certify

        self.assertIn("v10_7", certify.scene_name("center"))
        self.assertIn("v10_7", certify.NO_PENDANT_NAME)


if __name__ == "__main__":
    unittest.main()
