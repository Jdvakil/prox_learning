"""Dedicated behavioral tests for the repaired V10.7 Phase-0 gate.

Covers approval rejection, every gate predicate, live-risk confirmation,
pendant contact, incomplete/missing/duplicate results, worker exceptions,
one-shot output behaviour, authorization ordering, and immutable artifact
creation. The gate is never executed here.
"""

from __future__ import annotations

import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "scripts", ROOT / "submodules" / "molmospaces"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import run_pact_place_v107_phase0 as gate  # noqa: E402
from pact_place_v106_contract import PHASE0_RISK_CONFIRM_MAX_M  # noqa: E402
from pact_place_v107_contract import (  # noqa: E402
    CERT_ROOT, INTRUSION_SIDES, N_PHASE0_ROWS, PHASE0_MIN_CLEAN,
    PHASE0_MIN_CLEAN_PER_POSE, PHASE0_MIN_CLEAN_PER_SIDE,
    PHASE0_MIN_CLEAN_PER_SIDE_POSE, PHASE0_ROOT, POSE_IDS, phase0_rows,
    sha256_payload,
)

REVIEW_ROOT = ROOT / gate.OWNER_REVIEW_ROOT


def _rows():
    certification = json.loads(
        (ROOT / CERT_ROOT / "certification.json").read_text())
    scene_by_pose = {
        p: {"relative": certification["published_scenes"][p]["relative"],
            "sha256": certification["published_scenes"][p]["sha256"]}
        for p in POSE_IDS
    }
    assembly_by_pose = {
        p: sha256_payload(next(c for c in certification["compiled_checks"]
                               if c["pose_id"] == p))
        for p in POSE_IDS
    }
    return phase0_rows(selected=certification["selected"],
                       scene_by_pose=scene_by_pose,
                       assembly_by_pose=assembly_by_pose)


def _result(row, *, clean=True, risk=0.020, contact=0, status="complete",
            steps=520):
    defects = [] if clean else ["contact:clutter=7"]
    return {
        "role_index": int(row["role_index"]), "status": status,
        "episode_steps": steps,
        "v107_clean_success": bool(clean), "v107_defects": defects,
        "pact_v106_frame_telemetry": {
            "min_clearance_m": 0.018,
            "min_lobe_stem_clearance_m": risk,
            "pendant_robot_or_target_contact_frames": contact,
            "max_realized_tcp_speed_m_s": 0.21,
        },
    }


def _all_pass(rows):
    """A result set that satisfies every predicate."""
    return [_result(r, clean=True, risk=0.020) for r in rows]


class ApprovalRejectionTest(unittest.TestCase):
    EXPECTED = {"contract_version": "v", "review_manifest_payload_sha256": "a" * 64}
    VIDEOS = [f"v{i}.mp4" for i in range(6)]

    def _approval(self, **over):
        record = {"decision": "approve_phase0", "created_by": "owner",
                  "created_by_agent": False,
                  "reviewed_videos": list(self.VIDEOS), **self.EXPECTED}
        record.update(over)
        return record

    def _assert(self, approval):
        gate.assert_approval(approval, self.EXPECTED, self.VIDEOS)

    def test_complete_record_passes(self):
        self._assert(self._approval())

    def test_missing_record_refused(self):
        with self.assertRaises(gate.ApprovalError):
            self._assert(None)

    def test_wrong_decision_refused(self):
        with self.assertRaises(gate.ApprovalError):
            self._assert(self._approval(decision="looks_ok"))

    def test_agent_created_refused(self):
        with self.assertRaises(gate.ApprovalError):
            self._assert(self._approval(created_by_agent=True))

    def test_wrong_author_refused(self):
        with self.assertRaises(gate.ApprovalError):
            self._assert(self._approval(created_by="assistant"))

    def test_partial_bindings_refused(self):
        record = self._approval()
        del record["review_manifest_payload_sha256"]
        with self.assertRaises(gate.ApprovalError):
            self._assert(record)

    def test_every_stale_binding_refused(self):
        for key in self.EXPECTED:
            with self.subTest(binding=key):
                with self.assertRaises(gate.ApprovalError):
                    self._assert(self._approval(**{key: "0" * 64}))

    def test_extra_and_missing_videos_refused(self):
        with self.assertRaises(gate.ApprovalError):
            self._assert(self._approval(reviewed_videos=self.VIDEOS + ["x.mp4"]))
        with self.assertRaises(gate.ApprovalError):
            self._assert(self._approval(reviewed_videos=self.VIDEOS[:-1]))

    def test_video_list_must_be_a_list(self):
        with self.assertRaises(gate.ApprovalError):
            self._assert(self._approval(reviewed_videos="all"))

    def test_bindings_include_runner_digest_and_six_videos(self):
        bindings = gate.expected_bindings(REVIEW_ROOT)
        self.assertIn("gate_implementation_digest", bindings)
        self.assertEqual(
            sum(1 for k in bindings if k.startswith("video_sha256:")), 6)
        self.assertEqual(bindings["gate_implementation_digest"],
                         gate.gate_implementation_digest())


class GatePredicateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = _rows()

    def test_all_pass_is_a_pass(self):
        report = gate.gate_eligibility(self.rows, _all_pass(self.rows))
        self.assertEqual(report["clean_successes"], N_PHASE0_ROWS)
        self.assertEqual(report["limiting_predicates"], [])
        self.assertTrue(report["phase0_passed"])

    def test_fifteen_clean_fails_the_bar(self):
        results = [_result(r, clean=(i < 15), risk=0.020)
                   for i, r in enumerate(self.rows)]
        report = gate.gate_eligibility(self.rows, results)
        self.assertEqual(report["clean_successes"], 15)
        self.assertFalse(report["phase0_passed"])
        self.assertTrue(any(f"clean 15 < {PHASE0_MIN_CLEAN}" in p
                            for p in report["limiting_predicates"]))

    def test_per_side_floor(self):
        results = [_result(r, clean=(r["intrusion_side"] == "right"))
                   for r in self.rows]
        report = gate.gate_eligibility(self.rows, results)
        self.assertFalse(report["phase0_passed"])
        self.assertTrue(any("per-side" in p for p in report["limiting_predicates"]))

    def test_per_pose_and_cell_floors_exist(self):
        self.assertEqual(PHASE0_MIN_CLEAN_PER_SIDE, 7)
        self.assertEqual(PHASE0_MIN_CLEAN_PER_POSE, 4)
        self.assertEqual(PHASE0_MIN_CLEAN_PER_SIDE_POSE, 2)
        results = [_result(r, clean=(r["pose_id"] != "center"))
                   for r in self.rows]
        report = gate.gate_eligibility(self.rows, results)
        self.assertFalse(report["phase0_passed"])
        joined = " ".join(report["limiting_predicates"])
        self.assertIn("per-pose", joined)

    def test_pendant_contact_fails_even_when_counts_pass(self):
        results = _all_pass(self.rows)
        results[0]["pact_v106_frame_telemetry"][
            "pendant_robot_or_target_contact_frames"] = 4
        report = gate.gate_eligibility(self.rows, results)
        self.assertEqual(report["pendant_contact_rows"], 1)
        self.assertFalse(report["phase0_passed"])
        self.assertTrue(any("pendant contact" in p
                            for p in report["limiting_predicates"]))

    def test_minimum_pendant_clearance_is_reported(self):
        results = _all_pass(self.rows)
        results[3]["pact_v106_frame_telemetry"]["min_clearance_m"] = 0.0161
        report = gate.gate_eligibility(self.rows, results)
        self.assertAlmostEqual(report["min_pendant_clearance_m"], 0.0161)

    def test_failure_taxonomy_is_recorded(self):
        results = [_result(r, clean=(i % 2 == 0)) for i, r in enumerate(self.rows)]
        report = gate.gate_eligibility(self.rows, results)
        self.assertIn("contact:clutter", report["failure_taxonomy"])
        self.assertEqual(len(report["row_table"]), N_PHASE0_ROWS)


class RiskConfirmationTest(unittest.TestCase):
    """The inherited live-risk gate the first runner dropped."""

    @classmethod
    def setUpClass(cls):
        cls.rows = _rows()

    def test_threshold_is_the_inherited_value(self):
        self.assertEqual(PHASE0_RISK_CONFIRM_MAX_M, 0.035)
        report = gate.gate_eligibility(self.rows, _all_pass(self.rows))
        self.assertEqual(report["risk_confirmation_max_m"], 0.035)

    def test_witnesses_are_reported_per_side_and_pose(self):
        report = gate.gate_eligibility(self.rows, _all_pass(self.rows))
        self.assertEqual(sorted(report["risk_witness_by_side"]),
                         sorted(INTRUSION_SIDES))
        self.assertEqual(sorted(report["risk_witness_by_pose"]),
                         sorted(POSE_IDS))
        for witness in report["risk_witness_by_side"].values():
            self.assertIn("role_index", witness)
            self.assertLessEqual(witness["min_lobe_stem_clearance_m"], 0.035)

    def test_a_side_with_no_close_row_fails(self):
        results = [
            _result(r, clean=True,
                    risk=0.020 if r["intrusion_side"] == "left" else 0.090)
            for r in self.rows
        ]
        report = gate.gate_eligibility(self.rows, results)
        self.assertFalse(report["risk_confirmed_by_side"]["right"])
        self.assertFalse(report["phase0_passed"])
        self.assertTrue(any("side(s) ['right']" in p
                            for p in report["limiting_predicates"]))

    def test_a_pose_with_no_close_row_fails(self):
        results = [
            _result(r, clean=True,
                    risk=0.020 if r["pose_id"] != "pos5" else 0.080)
            for r in self.rows
        ]
        report = gate.gate_eligibility(self.rows, results)
        self.assertFalse(report["risk_confirmed_by_pose"]["pos5"])
        self.assertFalse(report["phase0_passed"])
        self.assertTrue(any("pose(s) ['pos5']" in p
                            for p in report["limiting_predicates"]))

    def test_an_infrastructure_row_cannot_supply_a_witness(self):
        results = _all_pass(self.rows)
        for index, row in enumerate(self.rows):
            if row["pose_id"] == "neg5":
                results[index] = gate.infrastructure_result(
                    row, RuntimeError("boom"))
        report = gate.gate_eligibility(self.rows, results)
        self.assertFalse(report["risk_confirmed_by_pose"]["neg5"])
        self.assertFalse(report["phase0_passed"])

    def test_exactly_at_the_threshold_counts(self):
        results = [_result(r, clean=True, risk=PHASE0_RISK_CONFIRM_MAX_M)
                   for r in self.rows]
        report = gate.gate_eligibility(self.rows, results)
        self.assertTrue(all(report["risk_confirmed_by_side"].values()))
        self.assertTrue(all(report["risk_confirmed_by_pose"].values()))


class IncompleteAndDuplicateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = _rows()

    def test_a_missing_row_fails_and_is_counted(self):
        report = gate.gate_eligibility(self.rows, _all_pass(self.rows)[:-1])
        self.assertEqual(report["incomplete_rows"], 1)
        self.assertFalse(report["phase0_passed"])
        self.assertIn("row_missing", report["failure_taxonomy"])

    def test_an_infrastructure_row_is_incomplete_not_clean(self):
        results = _all_pass(self.rows)
        results[0] = gate.infrastructure_result(self.rows[0], OSError("disk"))
        report = gate.gate_eligibility(self.rows, results)
        self.assertEqual(report["infrastructure_failures"], 1)
        self.assertEqual(report["incomplete_rows"], 1)
        self.assertEqual(report["clean_successes"], N_PHASE0_ROWS - 1)
        self.assertFalse(report["phase0_passed"])

    def test_infrastructure_result_records_no_replacement_or_retry(self):
        record = gate.infrastructure_result(self.rows[0], ValueError("x"))
        self.assertEqual(record["status"], gate.INFRASTRUCTURE_STATUS)
        self.assertFalse(record["v107_clean_success"])
        self.assertTrue(record["counted_as_incomplete"])
        self.assertFalse(record["replaced"])
        self.assertFalse(record["retried"])
        self.assertIn("infrastructure_traceback", record)

    def test_duplicate_results_are_detected(self):
        results = _all_pass(self.rows)
        results.append(dict(results[0]))
        seen: dict[int, int] = {}
        for item in results:
            role = int(item["role_index"])
            seen[role] = seen.get(role, 0) + 1
        self.assertEqual(sorted(r for r, n in seen.items() if n > 1), [0])
        source = inspect.getsource(gate.main)
        self.assertIn("duplicate results for roles", source)
        self.assertIn('eligibility["phase0_passed"] = False', source)


class WorkerExceptionTest(unittest.TestCase):
    def test_future_result_is_wrapped(self):
        source = inspect.getsource(gate.main)
        self.assertIn("result = future.result()", source)
        self.assertIn("except BaseException as error", source)
        self.assertIn("infrastructure_result(row, error)", source)

    def test_close_out_is_in_a_finally_block(self):
        source = inspect.getsource(gate.main)
        self.assertIn("finally:", source)
        self.assertLess(source.index("finally:"),
                        source.index('write_immutable_create_only(output_root / "gate.json"'))

    def test_unreported_rows_are_backfilled_on_catastrophe(self):
        source = inspect.getsource(gate.main)
        self.assertIn("reported = {int(r[\"role_index\"]) for r in results}", source)
        self.assertIn("if int(row[\"role_index\"]) not in reported", source)

    def test_worker_exceptions_are_recorded_in_the_artifact(self):
        source = inspect.getsource(gate.main)
        self.assertIn('"worker_exceptions": worker_exceptions', source)
        self.assertIn('"n_worker_exceptions"', source)


class IntegrityTest(unittest.TestCase):
    def test_pre_run_integrity_passes_on_the_real_tree(self):
        report = gate.integrity_report(REVIEW_ROOT, rows=_rows(), phase="pre_run")
        self.assertEqual(report["problems"], [])
        self.assertTrue(report["passed"])

    def test_it_checks_packet_spec_scenes_runner_manifest_and_bindings(self):
        report = gate.integrity_report(REVIEW_ROOT, rows=_rows(), phase="pre_run")
        for key in ("review_manifest_payload_sha256", "reviewed_video_sha256",
                    "specification_payload_sha256", "scene_sha256",
                    "gate_implementation_digest", "gate_implementation_files",
                    "manifest_checks", "thresholds", "approval_bindings"):
            self.assertIn(key, report, key)
        self.assertEqual(len(report["reviewed_video_sha256"]), 6)
        self.assertEqual(len(report["scene_sha256"]), 3)

    def test_manifest_checks_cover_disjointness_and_balance(self):
        checks = gate.integrity_report(
            REVIEW_ROOT, rows=_rows(), phase="pre_run")["manifest_checks"]
        self.assertTrue(checks["n_rows_correct"])
        self.assertTrue(checks["episode_ids_unique"])
        self.assertTrue(checks["disjoint_from_pool"])
        self.assertEqual(checks["balance"]["by_side"], {"left": 12, "right": 12})
        self.assertEqual(checks["balance"]["by_pose"],
                         {"neg5": 8, "center": 8, "pos5": 8})

    def test_both_phases_are_run_and_recorded(self):
        source = inspect.getsource(gate.main)
        self.assertIn('phase="pre_run"', source)
        self.assertIn('phase="post_run"', source)
        self.assertIn('"pre_run_integrity": pre_run', source)
        self.assertIn('"post_run_integrity": post_run', source)

    def test_post_run_integrity_failure_blocks_a_pass(self):
        source = inspect.getsource(gate.main)
        self.assertIn(
            'eligibility["phase0_passed"] and post_run["passed"]', source)

    def test_pre_run_failure_stops_before_creating_the_output_root(self):
        source = inspect.getsource(gate.main)
        self.assertLess(source.index("pre-run integrity failed"),
                        source.index("output_root.mkdir"))

    def test_a_tampered_video_is_detected(self):
        import shutil

        with tempfile.TemporaryDirectory() as scratch:
            mirror = Path(scratch) / "packet"
            shutil.copytree(REVIEW_ROOT, mirror)
            victim = sorted((mirror / "videos").glob("*.mp4"))[0]
            victim.write_bytes(victim.read_bytes() + b"\x00")
            report = gate.integrity_report(mirror, phase="pre_run")
            self.assertFalse(report["passed"])
            self.assertTrue(any("drifted" in p for p in report["problems"]))


class OneShotAndAuthorizationTest(unittest.TestCase):
    def test_output_root_must_not_already_exist(self):
        source = inspect.getsource(gate.main)
        self.assertIn("Phase 0 runs exactly once", source)
        self.assertIn("if output_root.exists()", source)

    def test_approval_is_checked_before_anything_is_created(self):
        source = inspect.getsource(gate.main)
        self.assertLess(source.index("assert_approval("),
                        source.index("output_root.mkdir"))

    def test_authorization_defaults_are_spread_first(self):
        source = inspect.getsource(gate.main)
        spread = source.rindex("**empty_authorization()")
        outcome = source.index('"phase0_passed": bool(')
        self.assertLess(spread, outcome)

    def test_spreading_defaults_last_would_reset_a_pass(self):
        from pact_place_v107_contract import empty_authorization

        wrong = {"phase0_passed": True, **empty_authorization()}
        self.assertFalse(wrong["phase0_passed"])
        right = {**empty_authorization(), "phase0_passed": True}
        self.assertTrue(right["phase0_passed"])

    def test_a_pass_authorizes_nothing_downstream(self):
        source = inspect.getsource(gate.main)
        for field in ("authorizes_collection", "authorizes_conversion",
                      "authorizes_training", "authorizes_evaluation"):
            self.assertIn(f'"{field}": False', source)

    def test_artifacts_are_immutable_create_only(self):
        from pact_place_v107_contract import (
            ImmutableArtifactError, write_immutable_create_only,
        )

        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch) / "gate.json"
            write_immutable_create_only(path, {"value": 1})
            with self.assertRaises(ImmutableArtifactError):
                write_immutable_create_only(path, {"value": 2})
        source = inspect.getsource(gate.main)
        self.assertIn(
            'write_immutable_create_only(output_root / "gate_manifest.json"', source)
        self.assertIn(
            'write_immutable_create_only(output_root / "gate.json"', source)

    def test_agent_never_writes_the_approval(self):
        source = inspect.getsource(gate)
        for line in source.splitlines():
            if "human_approval" in line:
                self.assertNotIn("write_text", line)
                self.assertNotIn("write_immutable", line)

    def test_gate_has_not_been_run(self):
        self.assertFalse((ROOT / PHASE0_ROOT).exists())
        self.assertFalse((REVIEW_ROOT / "human_approval.json").exists())


class DistributionTest(unittest.TestCase):
    def test_summary_statistics(self):
        summary = gate._summary([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(summary["n"], 4)
        self.assertEqual(summary["min"], 1.0)
        self.assertEqual(summary["max"], 4.0)
        self.assertAlmostEqual(summary["median"], 2.5)
        self.assertAlmostEqual(summary["mean"], 2.5)

    def test_empty_summary_is_safe(self):
        self.assertEqual(gate._summary([]), {"n": 0})

    def test_distributions_cover_length_and_both_speeds(self):
        rows = _rows()
        results = _all_pass(rows)
        report = gate.distributions(rows, results, ROOT / PHASE0_ROOT)
        for key in ("episode_steps", "commanded_tcp_speed_m_s",
                    "realized_tcp_speed_m_s", "per_row"):
            self.assertIn(key, report, key)
        self.assertEqual(report["episode_steps"]["n"], len(rows))
        self.assertEqual(len(report["per_row"]), len(rows))

    def test_distributions_survive_missing_trajectories(self):
        rows = _rows()
        report = gate.distributions(rows, _all_pass(rows),
                                    ROOT / "diagnostics_output/does_not_exist")
        self.assertEqual(report["commanded_tcp_speed_m_s"], {"n": 0})


if __name__ == "__main__":
    unittest.main()
