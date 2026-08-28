from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "scripts", ROOT / "submodules" / "molmospaces"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pact_place_v98_offset_contact_lib import (  # noqa: E402
    MAX_TCP_RESIDUAL_M,
    classify_onset_category,
    empty_authorization,
    lag_rows_for_aggregate,
    lookup_manifest,
    patch_manifest_for_row,
    reconstruction_is_valid,
    select_causal_category,
    trajectory_phase_sequence,
)
from pact_place_v98_pendant_contract import (  # noqa: E402
    CONTRACT_VERSION,
    SAMPLER_CLASS,
)

SOURCE_SUMMARY = ROOT / "diagnostics_output/pact_place_v95_raw_smoke/summary.json"
WIDE_600 = (
    ROOT
    / "diagnostics_output/pact_place_v98_paired_offset_wide/expert_screen_rows"
    / "600_99f1791b5695bf8a/result.json"
)
WIDE_601 = next(
    (
        ROOT
        / "diagnostics_output/pact_place_v98_paired_offset_wide/expert_screen_rows"
    ).glob("601_*/result.json")
)


class PactPlaceV98OffsetContactDiagnosisTest(unittest.TestCase):
    def test_patch_uses_each_rows_own_manifest_not_first_template(self) -> None:
        source = json.loads(SOURCE_SUMMARY.read_text())
        first = source["manifest_rows"][0]
        second = source["manifest_rows"][1]
        result_600 = json.loads(WIDE_600.read_text())
        result_601 = json.loads(WIDE_601.read_text())
        patched_600 = patch_manifest_for_row(
            lookup_manifest(source, result_600["episode_id"]),
            result_600,
            sampler_class=SAMPLER_CLASS,
            contract_version=CONTRACT_VERSION,
        )
        patched_601 = patch_manifest_for_row(
            lookup_manifest(source, result_601["episode_id"]),
            result_601,
            sampler_class=SAMPLER_CLASS,
            contract_version=CONTRACT_VERSION,
        )
        self.assertEqual(patched_600["episode_id"], first["episode_id"])
        self.assertEqual(patched_601["episode_id"], second["episode_id"])
        self.assertNotEqual(patched_600["episode_id"], patched_601["episode_id"])
        self.assertNotEqual(patched_600["layout_id"], patched_601["layout_id"])
        self.assertEqual(patched_600["layout_id"], first["layout_id"])
        self.assertEqual(patched_601["layout_id"], second["layout_id"])
        with self.assertRaises(ValueError):
            patch_manifest_for_row(
                first,
                result_601,
                sampler_class=SAMPLER_CLASS,
                contract_version=CONTRACT_VERSION,
            )

    def test_rejects_tcp_residual_above_one_millimetre(self) -> None:
        self.assertTrue(reconstruction_is_valid(0.0005))
        self.assertTrue(reconstruction_is_valid(MAX_TCP_RESIDUAL_M))
        self.assertFalse(reconstruction_is_valid(0.0011))
        self.assertFalse(reconstruction_is_valid(0.106))
        self.assertFalse(reconstruction_is_valid(None))

    def test_onset_uses_first_contact_phase_not_terminal(self) -> None:
        phases = [
            (0, "pregrasp"),
            (10, "inbound_vessel_pass"),
            (50, "inbound_cross_vessel_pass"),
            (70, "inbound_ceiling_fixture_approach"),
            (80, "inbound_ceiling_fixture_pass"),
            (100, "inbound_ceiling_fixture_exit"),
            (110, "pregrasp"),
        ]
        left = classify_onset_category(
            "pregrasp", step=107, trajectory_phases=phases
        )
        right = classify_onset_category(
            "inbound_cross_vessel_pass", step=53, trajectory_phases=phases
        )
        right_vessel = classify_onset_category(
            "inbound_vessel_pass", step=51, trajectory_phases=phases
        )
        self.assertEqual(left, "post_bow_pregrasp_coverage")
        self.assertEqual(right, "early_approach_coverage")
        self.assertEqual(right_vessel, "early_approach_coverage")
        misread_terminal = classify_onset_category(
            "inbound_ceiling_fixture_exit",
            step=117,
            trajectory_phases=phases,
        )
        self.assertEqual(misread_terminal, "protected_ceiling_bow_contact")
        self.assertNotEqual(right, misread_terminal)
        self.assertNotEqual(left, misread_terminal)

    def test_invalid_rows_are_excluded_from_lag_aggregates(self) -> None:
        rows = [
            {
                "reconstruction_valid": False,
                "max_tcp_residual_m": 0.106,
                "lag_samples": [{"tcp_to_fr3_link6_body_origin_lateral_m": 0.208}],
            },
            {
                "reconstruction_valid": True,
                "max_tcp_residual_m": 0.0004,
                "lag_samples": [{"tcp_to_fr3_link6_body_origin_lateral_m": 0.05}],
            },
        ]
        kept = lag_rows_for_aggregate(rows)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["max_tcp_residual_m"], 0.0004)

    def test_split_assigns_robot_not_the_mount(self) -> None:
        from pact_place_v98_offset_contact_lib import split_robot_fixture_sides

        sides = split_robot_fixture_sides(
            {
                "geom1": "pact_clutter_mount_ceiling_g",
                "geom2": "robot_0/fr3_link5_collision",
                "body1": "pact_clutter_mount_ceiling",
                "body2": "robot_0/fr3_link5",
                "root1": "pact_clutter_mount_ceiling",
                "root2": "robot_0/base",
                "geom1_id": 24,
                "geom2_id": 90,
            }
        )
        self.assertEqual(sides["robot_geom"], "robot_0/fr3_link5_collision")
        self.assertEqual(sides["robot_body"], "robot_0/fr3_link5")
        self.assertEqual(sides["pendant_geom"], "pact_clutter_mount_ceiling_g")
        self.assertEqual(sides["robot_geom_id"], 90)

    def test_split_when_robot_is_geom1(self) -> None:
        from pact_place_v98_offset_contact_lib import split_robot_fixture_sides

        sides = split_robot_fixture_sides(
            {
                "geom1": "robot_0/fr3_link6_collision",
                "geom2": "pact_clutter_mount_ceiling_g",
                "body1": "robot_0/fr3_link6",
                "body2": "pact_clutter_mount_ceiling",
                "root1": "robot_0/",
                "root2": "pact_clutter_mount_ceiling",
            }
        )
        self.assertEqual(sides["robot_body"], "robot_0/fr3_link6")
        self.assertEqual(sides["pendant_body"], "pact_clutter_mount_ceiling")

    def test_mounted_fixture_observe_feeds_non_target(self) -> None:
        from molmo_spaces.tasks.pact_place_contact_audit import PactPlaceContactAudit

        pair = {
            "geom1": "robot_0/fr3_link6_collision",
            "geom2": "pact_clutter_mount_ceiling_g",
            "body1": "robot_0/fr3_link6",
            "body2": "pact_clutter_mount_ceiling",
            "root1": "robot_0/",
            "root2": "pact_clutter_mount_ceiling",
            "distance_m": -0.002,
        }
        audit = PactPlaceContactAudit()
        audit.set_phase("inbound", "inbound_vessel_pass")
        env = SimpleNamespace(current_data=SimpleNamespace(time=1.5))
        with patch(
            "molmo_spaces.tasks.pact_place_contact_audit.place_environment_contact_pairs",
            return_value=[pair],
        ):
            audit.observe(env, step=53)
        summary = audit.summary()
        self.assertEqual(summary["contact_class_totals"]["mounted_fixture"], 1)
        self.assertGreater(summary["non_target_contact_entries"], 0)
        self.assertFalse(summary["collision_free"])

    def test_authorization_fields_remain_false(self) -> None:
        fields = empty_authorization()
        self.assertFalse(fields["authorizes_new_episodes"])
        self.assertFalse(fields["authorizes_gate"])
        self.assertFalse(fields["authorizes_collection"])
        self.assertFalse(fields["episodes_ran"])
        self.assertFalse(fields["physics_stepped"])
        category = select_causal_category(
            baseline_clean_onset_categories=[
                "post_bow_pregrasp_coverage",
                "early_approach_coverage",
            ]
            * 3,
            lag_reproduced=False,
            reconstruction_ok_for_baseline_clean=True,
        )
        self.assertEqual(category, "route_composition_coverage_failure")
        unresolved = select_causal_category(
            baseline_clean_onset_categories=["unreconstructed"],
            lag_reproduced=False,
            reconstruction_ok_for_baseline_clean=False,
        )
        self.assertEqual(unresolved, "mechanism_unresolved")
        envelope = select_causal_category(
            baseline_clean_onset_categories=["protected_ceiling_bow_contact"] * 6,
            lag_reproduced=True,
            reconstruction_ok_for_baseline_clean=True,
            protected_clearance_violated=True,
        )
        self.assertEqual(envelope, "verified_envelope_failure")

    def test_written_diagnosis_keeps_authorization_false(self) -> None:
        path = (
            ROOT
            / "diagnostics_output/pact_place_v98_offset_contact_diagnosis"
            / "diagnosis.json"
        )
        self.assertTrue(path.is_file())
        document = json.loads(path.read_text())
        self.assertFalse(document["authorizes_new_episodes"])
        self.assertFalse(document["authorizes_gate"])
        self.assertFalse(document["authorizes_collection"])
        self.assertFalse(document["episodes_ran"])
        self.assertFalse(document["physics_stepped"])
        self.assertEqual(
            document["causal_category"], "route_composition_coverage_failure"
        )
        self.assertEqual(
            document["lag_provenance"]["status"], "unverified_provenance"
        )
        self.assertEqual(
            document["lag_provenance"]["face_window_status"],
            "physical_input_invalid",
        )

    def test_trajectory_phase_sequence_roundtrip(self) -> None:
        sequence = trajectory_phase_sequence(
            [
                {"step": 51, "policy_phase": "inbound_vessel_pass"},
                {"step": 107, "policy_phase": "pregrasp"},
            ]
        )
        self.assertEqual(
            sequence,
            [(51, "inbound_vessel_pass"), (107, "pregrasp")],
        )


if __name__ == "__main__":
    unittest.main()
