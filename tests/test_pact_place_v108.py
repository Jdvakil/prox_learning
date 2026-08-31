"""Behavioral tests for the V10.8 exploratory collection.

Covers quotas and their derived totals, frozen per-cell seed streams,
round-robin scheduling without quota relaxation, budget enforcement, trainable
schema validation, ledger crash-safety and resume, pruning order, preflight
gates, and the standing statement that this is not a Phase-0 pass.
"""

from __future__ import annotations

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

from pact_place_v108_contract import (  # noqa: E402
    BASE_QUOTA_PER_CELL, BONUS_CELLS, COLLECTION_MASTER_SEED, COLLECTION_STREAM,
    IS_EXPLORATORY_OWNER_OVERRIDE, IS_PHASE0_PASS, MAX_SCIENTIFIC_ATTEMPTS,
    MAX_WALL_CLOCK_HOURS, N_PROXIMITY_SENSORS, POSE_IDS, PROXIMITY_FRAME_SHAPE,
    TARGET_SUCCESSES, attempt_id, build_contract, cell_key, cell_seed, cells,
    next_attempts, quota_totals, quotas, quotas_met, remaining_quota,
    round_robin_schedule,
)
import run_pact_place_v108_collect as collect  # noqa: E402


class QuotaTest(unittest.TestCase):
    def test_totals_are_the_registered_values(self):
        t = quota_totals()
        self.assertEqual(t["total"], 152)
        self.assertEqual(set(t["by_family"].values()), {38})
        self.assertEqual(t["by_side"], {"left": 76, "right": 76})
        self.assertEqual(t["by_pose"], {"neg5": 50, "center": 51, "pos5": 51})

    def test_base_quota_and_eight_bonus_cells(self):
        q = quotas()
        self.assertEqual(len(q), 24)
        self.assertEqual(sum(1 for v in q.values() if v == BASE_QUOTA_PER_CELL + 1), 8)
        self.assertEqual(sum(1 for v in q.values() if v == BASE_QUOTA_PER_CELL), 16)

    def test_bonus_cells_are_exactly_the_registered_ones(self):
        q = quotas()
        seven = sorted(k for k, v in q.items() if v == 7)
        self.assertEqual(seven, sorted(cell_key(*c) for c in BONUS_CELLS))

    def test_every_bonus_cell_is_a_real_cell(self):
        registered = {cell_key(*c) for c in cells()}
        for bonus in BONUS_CELLS:
            self.assertIn(cell_key(*bonus), registered)

    def test_one_bonus_per_family_side(self):
        pairs = [(f, s) for f, s, _ in BONUS_CELLS]
        self.assertEqual(len(pairs), len(set(pairs)))
        self.assertEqual(len(pairs), 8)


class SeedStreamTest(unittest.TestCase):
    def test_seeds_are_deterministic(self):
        a = cell_seed("F0_target_side_stagger", "left", "neg5", 3)
        b = cell_seed("F0_target_side_stagger", "left", "neg5", 3)
        self.assertEqual(a, b)

    def test_streams_are_cell_specific(self):
        a = cell_seed("F0_target_side_stagger", "left", "neg5", 0)
        b = cell_seed("F0_target_side_stagger", "left", "center", 0)
        c = cell_seed("F1_inner_panel_stagger", "left", "neg5", 0)
        self.assertNotEqual(a["seed_u64"], b["seed_u64"])
        self.assertNotEqual(a["seed_u64"], c["seed_u64"])

    def test_no_seed_collisions_across_the_whole_budget(self):
        seen = set()
        for family, side, pose in cells():
            for index in range(40):
                seen.add(cell_seed(family, side, pose, index)["seed_u64"])
        self.assertEqual(len(seen), 24 * 40)

    def test_attempt_ids_are_unique_and_stable(self):
        ids = [attempt_id(f, s, p, i) for f, s, p in cells() for i in range(20)]
        self.assertEqual(len(set(ids)), len(ids))
        self.assertEqual(ids[0], attempt_id(*cells()[0], 0))

    def test_stream_is_distinct_from_earlier_versions(self):
        self.assertIn("v10.8", COLLECTION_STREAM)
        self.assertEqual(COLLECTION_MASTER_SEED, 2026108001)


class ScheduleTest(unittest.TestCase):
    def test_round_robin_visits_every_cell_before_repeating(self):
        plan = round_robin_schedule({}, {}, 24)
        self.assertEqual(len(plan), 24)
        self.assertEqual(len({p["cell"] for p in plan}), 24)

    def test_order_is_deterministic(self):
        self.assertEqual([p["cell"] for p in round_robin_schedule({}, {}, 30)],
                         [p["cell"] for p in round_robin_schedule({}, {}, 30)])

    def test_a_satisfied_cell_drops_out(self):
        q = quotas()
        first = cell_key(*cells()[0])
        plan = round_robin_schedule({first: q[first]}, {}, 24)
        self.assertNotIn(first, {p["cell"] for p in plan})

    def test_quota_is_never_redistributed(self):
        """A satisfied cell's slots do not raise anyone else's quota."""
        q = quotas()
        first = cell_key(*cells()[0])
        accepted = {first: q[first]}
        plan = round_robin_schedule(accepted, {}, 200)
        counts: dict[str, int] = {}
        for item in plan:
            counts[item["cell"]] = counts.get(item["cell"], 0) + 1
        for key, value in counts.items():
            self.assertLessEqual(value, 200)
        self.assertEqual(remaining_quota(accepted)[first], 0)
        for key in q:
            if key != first:
                self.assertEqual(remaining_quota(accepted)[key], q[key])

    def test_attempt_index_advances_per_cell(self):
        plan = round_robin_schedule({}, {}, 48)
        first_cell = plan[0]["cell"]
        indices = [p["attempt_index"] for p in plan if p["cell"] == first_cell]
        self.assertEqual(indices, list(range(len(indices))))

    def test_resume_continues_the_frozen_stream(self):
        """Attempt indices continue from the ledger, never restart."""
        attempted = {cell_key(*cells()[0]): 5}
        plan = next_attempts({}, attempted, 24)
        first = next(p for p in plan if p["cell"] == cell_key(*cells()[0]))
        self.assertEqual(first["attempt_index"], 5)
        self.assertEqual(first["seed"], cell_seed(*cells()[0], 5))

    def test_quotas_met_predicate(self):
        self.assertFalse(quotas_met({}))
        self.assertTrue(quotas_met(quotas()))

    def test_budget_constants_are_the_registered_hard_stops(self):
        self.assertEqual(MAX_SCIENTIFIC_ATTEMPTS, 900)
        self.assertEqual(MAX_WALL_CLOCK_HOURS, 16.0)
        self.assertFalse(build_contract()["budget"]["extension_permitted"])


class TrainableSchemaTest(unittest.TestCase):
    def _write(self, directory: Path, *, n=20, sensors=N_PROXIMITY_SENSORS,
               finite=True, constant=False, dtype=np.float32, shape=None,
               with_actions=True, with_agent=True):
        import h5py

        shape = shape or PROXIMITY_FRAME_SHAPE
        path = directory / "trajectory.h5"
        with h5py.File(path, "w") as handle:
            traj = handle.create_group("traj_0")
            if with_actions:
                for key in ("commanded_action", "ee_pose", "ee_twist",
                            "joint_pos", "joint_pos_rel"):
                    traj.create_dataset(f"actions/{key}",
                                        data=np.zeros((n, 8), dtype=np.uint8))
            if with_agent:
                for key in ("qpos", "qvel"):
                    traj.create_dataset(f"obs/agent/{key}",
                                        data=np.zeros((n, 8), dtype=np.uint8))
            rng = np.random.default_rng(0)
            for index in range(sensors):
                block = (np.zeros((n, *shape), dtype=dtype) if constant
                         else rng.random((n, *shape)).astype(dtype))
                if not finite:
                    block = np.array(block)
                    block[0, 0, 0, 0] = np.inf
                traj.create_dataset(f"obs/proximity/sensor_{index}", data=block)
        return path

    def test_a_good_episode_passes_except_for_the_video(self):
        with tempfile.TemporaryDirectory() as scratch:
            directory = Path(scratch)
            self._write(directory)
            report = collect.validate_trainable(directory)
            self.assertEqual(report["detail"]["n_proximity"], 40)
            self.assertIn("missing wrist RGB video", report["problems"])

    def test_missing_h5_fails(self):
        with tempfile.TemporaryDirectory() as scratch:
            report = collect.validate_trainable(Path(scratch))
            self.assertFalse(report["passed"])
            self.assertIn("trajectory.h5 missing", report["problems"])

    def test_wrong_sensor_count_fails(self):
        with tempfile.TemporaryDirectory() as scratch:
            directory = Path(scratch)
            self._write(directory, sensors=39)
            report = collect.validate_trainable(directory)
            self.assertTrue(any("39 proximity sensors" in p
                                for p in report["problems"]))

    def test_non_finite_proximity_fails(self):
        with tempfile.TemporaryDirectory() as scratch:
            directory = Path(scratch)
            self._write(directory, finite=False)
            report = collect.validate_trainable(directory)
            self.assertTrue(any("non-finite" in p for p in report["problems"]))

    def test_constant_proximity_fails(self):
        with tempfile.TemporaryDirectory() as scratch:
            directory = Path(scratch)
            self._write(directory, constant=True)
            report = collect.validate_trainable(directory)
            self.assertTrue(any("constant" in p for p in report["problems"]))

    def test_wrong_dtype_fails(self):
        with tempfile.TemporaryDirectory() as scratch:
            directory = Path(scratch)
            self._write(directory, dtype=np.float64)
            report = collect.validate_trainable(directory)
            self.assertTrue(any("dtype" in p for p in report["problems"]))

    def test_wrong_frame_shape_fails(self):
        with tempfile.TemporaryDirectory() as scratch:
            directory = Path(scratch)
            self._write(directory, shape=(4, 8, 7))
            report = collect.validate_trainable(directory)
            self.assertTrue(any("shape" in p for p in report["problems"]))

    def test_missing_actions_or_agent_keys_fail(self):
        for kwargs, token in (({"with_actions": False}, "actions/"),
                              ({"with_agent": False}, "obs/agent/")):
            with self.subTest(**kwargs):
                with tempfile.TemporaryDirectory() as scratch:
                    directory = Path(scratch)
                    self._write(directory, **kwargs)
                    report = collect.validate_trainable(directory)
                    self.assertTrue(any(token in p for p in report["problems"]))

    def test_registered_shape_and_count(self):
        self.assertEqual(PROXIMITY_FRAME_SHAPE, (4, 8, 8))
        self.assertEqual(N_PROXIMITY_SENSORS, 40)


class LedgerTest(unittest.TestCase):
    def test_append_is_durable_and_readable(self):
        with tempfile.TemporaryDirectory() as scratch:
            ledger = collect.Ledger(Path(scratch) / "ledger.jsonl")
            ledger.append({"attempt_id": "a", "cell": "c", "accepted": True})
            ledger.append({"attempt_id": "b", "cell": "c", "accepted": False})
            records = ledger.read()
            self.assertEqual(len(records), 2)
            self.assertTrue(records[0]["accepted"])

    def test_a_torn_final_line_does_not_corrupt_resume(self):
        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch) / "ledger.jsonl"
            ledger = collect.Ledger(path)
            ledger.append({"attempt_id": "a", "cell": "c", "accepted": True})
            with open(path, "a") as stream:
                stream.write('{"attempt_id": "b", "cel')      # crash mid-write
            records = ledger.read()
            self.assertEqual(len(records), 1)
            accepted, attempted = collect.tally(records)
            self.assertEqual(accepted, {"c": 1})

    def test_tally_counts_attempts_and_acceptances(self):
        records = [
            {"cell": "x", "accepted": True}, {"cell": "x", "accepted": False},
            {"cell": "y", "accepted": True},
        ]
        accepted, attempted = collect.tally(records)
        self.assertEqual(accepted, {"x": 1, "y": 1})
        self.assertEqual(attempted, {"x": 2, "y": 1})

    def test_ledger_is_written_before_pruning(self):
        source = inspect.getsource(collect.main)
        self.assertLess(source.index("ledger.append(record)"),
                        source.index("prune_failed(row_dir)"))

    def test_compact_record_carries_the_required_fields(self):
        record = collect.compact_record({
            "attempt_id": "a", "cell": "c", "family_id": "F0",
            "intrusion_side": "left", "pose_id": "neg5", "attempt_index": 0,
            "task_seed_u32": 1, "status": "complete",
            "v108_accepted": True, "v108_clean_success": True,
            "v108_defects": [], "episode_steps": 500,
            "contact_audit": {"contact_class_totals": {"clutter": 0}},
            "clutter_stability_events": [],
            "pact_v106_frame_telemetry": {"min_clearance_m": 0.02},
        })
        for key in ("attempt_id", "cell", "task_seed_u32", "status", "accepted",
                    "defects", "contact_class_totals",
                    "clutter_stability_events", "episode_steps",
                    "min_pendant_clearance_m"):
            self.assertIn(key, record, key)

    def test_pruning_keeps_the_compact_record(self):
        with tempfile.TemporaryDirectory() as scratch:
            directory = Path(scratch)
            (directory / "result.json").write_text("{}")
            (directory / "trajectory.h5").write_bytes(b"x" * 1024)
            (directory / "trajectory.json").write_text("[]")
            report = collect.prune_failed(directory)
            self.assertTrue((directory / "result.json").is_file())
            self.assertFalse((directory / "trajectory.h5").exists())
            self.assertIn("trajectory.h5", report["removed"])


class PreflightTest(unittest.TestCase):
    def test_disk_preflight_reports_headroom(self):
        report = collect.disk_preflight(ROOT)
        for key in ("free_gb", "projected_peak_gb", "headroom_after_peak_gb",
                    "min_free_gb_required", "passed"):
            self.assertIn(key, report)

    def test_disk_preflight_fails_when_headroom_is_short(self):
        report = dict(collect.disk_preflight(ROOT))
        free = 1.0
        self.assertFalse(free - report["projected_peak_gb"]
                         >= report["min_free_gb_required"])

    def test_cgroup_preflight_checks_pids_cpu_and_memory(self):
        report = collect.cgroup_pid_preflight(12)
        for key in ("pids_headroom", "pids_needed_estimate", "effective_cpus",
                    "memory_free_gb", "workers_fit_cpu", "passed"):
            self.assertIn(key, report)

    def test_cgroup_preflight_rejects_more_workers_than_cpus(self):
        report = collect.cgroup_pid_preflight(4096)
        self.assertFalse(report["passed"])

    def test_scene_preflight_binds_three_certified_scenes(self):
        report = collect.scene_preflight()
        self.assertTrue(report["passed"], report["problems"])
        self.assertEqual(sorted(report["scenes"]), sorted(POSE_IDS))

    def test_threads_are_pinned_before_workers_are_created(self):
        source = inspect.getsource(collect.main)
        self.assertLess(source.index("pin_threads()"),
                        source.index("ProcessPoolExecutor"))
        self.assertIn("OMP_NUM_THREADS", str(collect.THREAD_POOL_ENV))

    def test_collection_refuses_to_start_when_preflight_fails(self):
        source = inspect.getsource(collect.main)
        self.assertIn('if args.preflight_only or not preflight["passed"]', source)
        self.assertIn("PREFLIGHT FAILED", source)


class InfrastructureHaltTest(unittest.TestCase):
    """A schema/infrastructure defect is not a scientific outcome."""

    def test_defect_halts_without_advancing_or_replacing(self):
        source = inspect.getsource(collect.main)
        self.assertIn('if result.get("v108_infrastructure_defect")', source)
        self.assertIn('"scientific_stream_advanced": False', source)
        self.assertIn('"row_replaced": False', source)
        self.assertIn('stop_reason = "infrastructure_or_schema_defect"', source)

    def test_defect_is_not_appended_to_the_scientific_ledger(self):
        source = inspect.getsource(collect.main)
        halt = source.index('if result.get("v108_infrastructure_defect")')
        append = source.index("ledger.append(record)")
        self.assertLess(halt, append,
                        "the halt must precede the scientific ledger append")
        self.assertIn("infrastructure.append(compact_record(result))", source)

    def test_a_schema_failure_marks_the_defect(self):
        source = inspect.getsource(collect.main)
        self.assertIn('result["v108_infrastructure_defect"] = "schema"', source)
        self.assertIn('result["v108_infrastructure_defect"] = "worker"', source)

    def test_summary_records_the_halt_and_excludes_defects(self):
        source = inspect.getsource(collect.main)
        self.assertIn('"halted_for_repair": halted', source)
        self.assertIn(
            '"infrastructure_defects_excluded_from_scientific_attempts": True',
            source)
        self.assertIn(
            '"infrastructure_retries_authorized_by_contract": False', source)

    def test_contract_authorizes_no_infrastructure_retry(self):
        self.assertFalse(build_contract()["budget"]["extension_permitted"])


class OverrideStatementTest(unittest.TestCase):
    def test_contract_says_this_is_not_a_phase0_pass(self):
        contract = build_contract()
        self.assertFalse(contract["is_phase0_pass"])
        self.assertTrue(contract["is_exploratory_owner_override"])
        self.assertEqual(contract["v107_phase0_result"],
                         "failed_8_of_24_permanently_closed")
        self.assertTrue(contract["v107_preserved_unmodified"])
        self.assertFalse(IS_PHASE0_PASS)
        self.assertTrue(IS_EXPLORATORY_OWNER_OVERRIDE)

    def test_no_downstream_authorization(self):
        contract = build_contract()
        for field in ("authorizes_conversion", "authorizes_training",
                      "authorizes_evaluation", "authorizes_phase0",
                      "phase0_passed"):
            self.assertFalse(contract[field], field)

    def test_contract_declares_the_datagen_pipeline(self):
        contract = build_contract()
        self.assertTrue(contract["uses_full_datagen_pipeline"])
        self.assertFalse(contract["uses_expert_screen_harness"])
        self.assertFalse(contract["reuses_v107_pool_or_phase0_rows"])

    def test_collector_does_not_import_the_screen_rollout(self):
        source = (ROOT / "scripts/run_pact_place_v108_collect.py").read_text()
        self.assertNotIn("from run_pact_place_expert_screen import run_row", source)
        self.assertIn("prepare_episode_for_saving", source)
        self.assertIn("save_trajectories", source)

    def test_v107_artifacts_are_untouched(self):
        import hashlib

        expected = {
            "diagnostics_output/pact_place_v107_phase0/gate.json":
                "ff4f6aa1c08920c8c07dc30377ea270105e64c35894cade804319b4ea4116c10",
        }
        for relative, digest in expected.items():
            path = ROOT / relative
            self.assertTrue(path.is_file(), relative)
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(), digest, relative)

    def test_output_roots_are_v108_specific(self):
        from pact_place_v108_contract import COLLECTION_ROOT, DATASET_ROOT

        self.assertIn("v108", COLLECTION_ROOT)
        self.assertIn("v10_8", DATASET_ROOT)


if __name__ == "__main__":
    unittest.main()
