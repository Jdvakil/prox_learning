from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "scripts", ROOT / "submodules" / "molmospaces"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import pact_place_v1011c_collection_contract as contract
import run_pact_place_v1011c_collect as collector


class V1011CCollectionContractTest(unittest.TestCase):
    def test_target_is_exactly_one_hundred(self) -> None:
        totals = contract.quota_totals()
        self.assertEqual(totals["total"], 100)
        self.assertEqual(totals["by_family"], {
            "F0_target_side_stagger": 25,
            "F1_inner_panel_stagger": 25,
            "F2_outer_panel_stagger": 25,
            "F3_aperture_side_stagger": 25,
        })
        self.assertEqual(totals["by_side"], {"left": 50, "right": 50})
        self.assertEqual(totals["by_pose"], {
            "neg5": 33, "center": 34, "pos5": 33,
        })
        self.assertEqual(sorted(contract.quotas().values()).count(5), 4)
        self.assertEqual(sorted(contract.quotas().values()).count(4), 20)

    def test_streams_are_new_and_disjoint(self) -> None:
        detail = contract.streams_are_disjoint()
        self.assertTrue(detail["disjoint"])
        self.assertEqual(detail["overlap"], [])
        self.assertNotEqual(contract.SMOKE_MASTER_SEED, contract.COLLECTION_MASTER_SEED)

    def test_rows_are_v1011c_and_unique(self) -> None:
        rows = [contract.build_row(*cell, 0) for cell in contract.cells()]
        self.assertEqual(len(rows), 24)
        self.assertEqual(len({row["attempt_id"] for row in rows}), 24)
        for row in rows:
            self.assertEqual(row["environment_version"], contract.ENVIRONMENT_VERSION)
            self.assertEqual(row["sampler_class"], contract.SAMPLER_CLASS)
            self.assertEqual(row["seed_stream"], contract.COLLECTION_STREAM)
            self.assertTrue(row["pact_v1011c_table_camera_required"])
            self.assertEqual(
                row["pact_v1011c_primitive_heights_m"],
                {"01": 0.32585000000000003, "08": 0.2394, "09": 0.2394},
            )

    def test_upstream_gate_is_explicit_and_green(self) -> None:
        bindings = contract.upstream_bindings()
        self.assertEqual(set(bindings), set(contract.UPSTREAM_ARTIFACTS))
        self.assertTrue(all(item["passed"] for item in bindings.values()))
        environment = json.loads(
            (ROOT / contract.UPSTREAM_ARTIFACTS["environment_contract"]).read_text()
        )
        self.assertNotIn("passed", environment)
        self.assertEqual(environment["sampler_class"], contract.SAMPLER_CLASS)

    def test_contract_requires_table_calibration_and_raw_skin(self) -> None:
        payload = contract.build_contract()
        self.assertEqual(payload["collection"]["target_successes"], 100)
        self.assertEqual(payload["observations"]["table_camera_rgb"], "exo_camera_1")
        self.assertEqual(payload["observations"]["raw_proximity_sensors"], 40)
        self.assertEqual(
            payload["observations"]["table_camera_calibration_keys"],
            ["extrinsic_cv", "cam2world_gl", "intrinsic_cv"],
        )
        self.assertEqual(payload["observations"]["contact_audit_storage"], "summary_only")
        self.assertFalse(payload["authorizes_training"])
        self.assertFalse(payload["authorizes_evaluation"])

    def test_one_in_flight_scheduler_accounting(self) -> None:
        records = []
        key = next(iter(contract.quotas()))
        family, side, pose = key.split("|")
        for index in range(2):
            row = contract.build_row(family, side, pose, index)
            records.append({
                "attempt_id": row["attempt_id"], "cell": key,
                "attempt_index": index, "accepted": index == 1,
            })
        accepted, attempted, next_index = collector.tally(records)
        self.assertEqual(accepted[key], 1)
        self.assertEqual(attempted[key], 2)
        self.assertEqual(next_index[key], 2)

    def test_worker_rebinds_sampler_seed_and_summary_storage(self) -> None:
        import run_pact_place_v1010_tablecam_validation as tablecam

        original_run = tablecam.run_attempt
        original_sampler = tablecam.SAMPLER_CLASS
        original_seed = tablecam.cell_seed
        try:
            def fake_run(payload):
                return {
                    "sampler": tablecam.SAMPLER_CLASS,
                    "seed": tablecam.cell_seed("F0_target_side_stagger", "left", "neg5", 0),
                    "summary": os.environ.get("PACT_CONTACT_AUDIT_SUMMARY_ONLY"),
                }

            tablecam.run_attempt = fake_run
            result = collector.worker({"row": {"smoke_only": False}})
            self.assertEqual(result["sampler"], contract.SAMPLER_CLASS)
            self.assertEqual(
                result["seed"],
                contract.cell_seed("F0_target_side_stagger", "left", "neg5", 0),
            )
            self.assertEqual(result["summary"], "1")
        finally:
            tablecam.run_attempt = original_run
            tablecam.SAMPLER_CLASS = original_sampler
            tablecam.cell_seed = original_seed


if __name__ == "__main__":
    unittest.main()
