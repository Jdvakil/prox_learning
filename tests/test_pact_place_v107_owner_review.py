"""Targeted tests for the V10.7 owner visual-review packet.

The packet is review-only over a FAILED pool. These tests check that it says
so, that its selection is derived rather than hardcoded, that it reuses
retained trajectories without executing anything, and that its videos
reconcile against their sources.
"""

from __future__ import annotations

import inspect
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "scripts", ROOT / "submodules" / "molmospaces"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from pact_place_v107_contract import POOL_MIN_CLEAN, POOL_ROOT, sha256_file  # noqa: E402
import run_pact_place_v107_owner_review as owner  # noqa: E402

PACKET = ROOT / owner.OWNER_REVIEW_ROOT
POOL = json.loads((ROOT / POOL_ROOT / "pool.json").read_text())


def _manifest():
    return json.loads((PACKET / "review_manifest.json").read_text())


class ReviewOnlyTest(unittest.TestCase):
    def test_runner_never_executes_an_episode(self):
        source = inspect.getsource(owner)
        for forbidden in ("env.step(", ".step(", "run_row("):
            self.assertNotIn(forbidden, source, forbidden)

    def test_runner_declares_review_only(self):
        source = inspect.getsource(owner.main)
        for flag in ('"review_only": True',
                     '"creates_episode": False',
                     '"calls_env_step": False',
                     '"resamples_tasks": False',
                     '"changes_geometry": False',
                     '"changes_thresholds": False',
                     '"reruns_pool": False',
                     '"reinterprets_pool_result": False'):
            self.assertIn(flag, source, flag)

    def test_runner_refuses_a_passing_pool(self):
        source = inspect.getsource(owner.main)
        self.assertIn('if eligibility["pool_passed"]', source)
        self.assertIn("this packet is for a FAILED pool", source)

    def test_runner_creates_no_approval_record(self):
        source = inspect.getsource(owner)
        for line in source.splitlines():
            if "human_approval" in line:
                self.assertNotIn("write_text", line)
                self.assertNotIn("write_immutable", line)


class SelectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.candidates = owner.candidate_rows(POOL)
        cls.plan = owner.select_six(cls.candidates)

    def test_selection_is_derived_not_hardcoded(self):
        source = inspect.getsource(owner)
        for forbidden in ("[6, 28, 8]", "[45, 40, 20]", "role_index == 6"):
            self.assertNotIn(forbidden, source, forbidden)

    def test_selection_is_found_and_valid(self):
        self.assertTrue(self.plan["found"])
        self.assertTrue(owner.verify_selection(self.plan)["passed"])

    def test_objective_order_is_registered(self):
        self.assertEqual(
            self.plan["objective"]["order"],
            ["minimise max pendant clearance", "then total clearance",
             "then sorted role-index tuple"],
        )

    def test_selection_is_deterministic(self):
        again = owner.select_six(owner.candidate_rows(POOL))
        self.assertEqual(
            [r["role_index"] for r in again["successes"]],
            [r["role_index"] for r in self.plan["successes"]])
        self.assertEqual(
            [r["role_index"] for r in again["failures"]],
            [r["role_index"] for r in self.plan["failures"]])

    def test_selection_minimises_the_maximum_clearance(self):
        """No valid subset may have a strictly smaller maximum."""
        import itertools

        chosen = self.plan["successes"] + self.plan["failures"]
        best_max = max(r["min_clearance_m"] for r in chosen)
        clean = [r for r in self.candidates if r["clean"]]
        failed = [r for r in self.candidates if not r["clean"]]
        for successes in owner._class_options(clean):
            for failures in owner._class_options(failed):
                six = successes + failures
                sides = [r["intrusion_side"] for r in six]
                if sides.count("left") != 3 or sides.count("right") != 3:
                    continue
                self.assertGreaterEqual(
                    max(r["min_clearance_m"] for r in six), best_max - 1e-12)

    def test_every_constraint_holds_on_the_chosen_six(self):
        checks = owner.verify_selection(self.plan)["checks"]
        for name, value in checks.items():
            self.assertTrue(value, name)

    def test_candidates_require_complete_telemetry(self):
        for record in self.candidates:
            self.assertIsNotNone(record["min_clearance_m"])
            self.assertGreater(record["n_frames"], 0)
            self.assertEqual(record["trajectory_n"], record["n_frames"])

    def test_rebuilt_rows_match_the_executed_rows(self):
        rows = owner.rebuild_pool_rows(POOL)
        recorded = {int(r["role_index"]): r for r in POOL["rows"]}
        self.assertEqual(len(rows), len(recorded))
        for role, row in rows.items():
            self.assertEqual(row["row_sha256"], recorded[role]["row_sha256"])


class PublishedPacketTest(unittest.TestCase):
    def setUp(self):
        if not PACKET.is_dir():
            self.skipTest("owner-review packet not published")

    def test_pool_failure_is_carried_not_reinterpreted(self):
        m = _manifest()
        self.assertFalse(m["pool_passed"])
        self.assertTrue(m["eligible_for_owner_visual_review"])
        self.assertEqual(m["pool_clean_successes"],
                         POOL["eligibility"]["clean_successes"])
        self.assertLess(m["pool_clean_successes"], POOL_MIN_CLEAN)
        self.assertTrue(m["publishing_these_videos_does_not_make_the_pool_pass"])
        self.assertFalse(m["authorizes_downstream_work"])

    def test_every_authorization_field_is_false(self):
        m = _manifest()
        for field in ("authorizes_phase0", "authorizes_gate",
                      "authorizes_collection", "authorizes_conversion",
                      "authorizes_training", "authorizes_evaluation",
                      "phase0_passed", "human_approval_present",
                      "eligible_for_human_review"):
            self.assertFalse(m[field], field)

    def test_no_approval_file_exists(self):
        self.assertFalse((PACKET / "human_approval.json").exists())

    def test_six_videos_nonempty_and_hash_matched(self):
        m = _manifest()
        self.assertEqual(m["n_videos"], 6)
        published = sorted((PACKET / "videos").glob("*.mp4"))
        self.assertEqual(len(published), 6)
        for record in m["videos"]:
            path = PACKET / "videos" / record["video_name"]
            self.assertTrue(path.is_file(), record["video_name"])
            self.assertGreater(path.stat().st_size, 0)
            self.assertEqual(sha256_file(path), record["video_raw_file_sha256"])

    def test_clips_are_complete_untrimmed_trajectories(self):
        m = _manifest()
        self.assertTrue(m["clips_are_complete_retained_trajectories"])
        self.assertFalse(m["clips_trimmed"])
        for record in m["videos"]:
            self.assertTrue(record["complete_retained_trajectory"])
            self.assertFalse(record["trimmed"])
            self.assertEqual(record["n_frames_rendered"],
                             record["retained_trajectory_n"])

    def test_all_failures_are_natural(self):
        m = _manifest()
        self.assertTrue(m["all_failures_are_natural"])
        self.assertTrue(m["no_induced_pendant_collision"])
        for record in m["videos"]:
            if not record["clean"]:
                self.assertEqual(record["pendant_contact_frames_in_replay"], 0)

    def test_source_hashes_are_bound_and_current(self):
        m = _manifest()
        candidates = {r["role_index"]: r for r in owner.candidate_rows(POOL)}
        for record in m["videos"]:
            source = candidates[record["role_index"]]
            self.assertEqual(source["result_raw_file_sha256"],
                             record["source_result_sha256"])
            self.assertEqual(source["trajectory_raw_file_sha256"],
                             record["source_trajectory_sha256"])

    def test_scene_and_assembly_hashes_are_bound(self):
        m = _manifest()
        for record in m["videos"]:
            path = ROOT / record["scene_relative"]
            self.assertTrue(path.is_file())
            self.assertEqual(sha256_file(path), record["scene_sha256"])
            self.assertTrue(record["assembly_sha256"])

    def test_documentation_drift_is_partitioned_not_ignored(self):
        drift = _manifest()["drift_check"]
        self.assertTrue(drift["code_and_data_clean"])
        self.assertEqual(drift["n_code_and_data_drift"], 0)
        self.assertTrue(drift["documentation_drift_is_expected"])

    def test_review_md_carries_the_required_statements(self):
        text = (PACKET / "REVIEW.md").read_text()
        for phrase in ("FAILED", "solely for owner visual assessment",
                       "does not make the pool pass",
                       "authorize any downstream work",
                       "offline certification", "six-group causality"):
            self.assertIn(phrase, text, phrase)

    def test_frame_and_duration_reconciliation_is_recorded(self):
        m = _manifest()
        for name, report in m["video_verification"].items():
            self.assertTrue(report["passed"], name)
            self.assertTrue(report["frames_match_retained_trajectory"], name)
            self.assertTrue(report["duration_matches"], name)


if __name__ == "__main__":
    unittest.main()
