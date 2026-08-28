from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "scripts", ROOT / "submodules" / "molmospaces"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pact_place_v10_compound_pendant_contract import (
    EMPIRICAL_LIVE_CONTACT_V1,
    ENDPOINT_ONLY_PRIMITIVE,
    ENVIRONMENT_VERSION,
    GROUP_FREEZE_PRIMITIVE,
    PROBE_NEGATIVE_LOBE,
    PROBE_POSITIVE_LOBE,
)
from pact_place_v10_geometry import planning_probe_assembly
from pact_place_v10_route import (
    plan_lane,
    plan_lane_endpoint_only,
    resolve_v10_runtime_route,
)
from pact_place_v101_empirical_qualification_contract import (
    CONTRACT_VERSION,
    GATE_MASTER_SEED,
    GATE_STREAM,
    LEFT_LANE_Y_M,
    N_GATE_ROWS,
    N_REVIEW_ROWS,
    PHYSICS_CLEAN_FAMILIES,
    REVIEW_MASTER_SEED,
    REVIEW_STREAM,
    RIGHT_LANE_Y_M,
    SLAB_PADDING_M,
    V99_SNAPSHOT_RELATIVE,
    admit_fixed_route_on_stock,
    admit_six_cell_fixed_route,
    assert_phase0_approval,
    build_contract,
    distribution_counts,
    frozen_assembly,
    frozen_route_for_side,
    is_v101_clean_success,
    lowest_clean_row_per_cell,
    paired_side_clutter_identical,
    review_eligibility,
    route_telemetry_complete,
    sha256_payload,
    verify_protected_artifacts,
)
from pact_place_v99_exact import load_clean_snapshots
from run_pact_place_v101_empirical_causal import _side_balance, _verdict
from run_pact_place_v9_v0c3_causal_proximity import ABS_DELTA_FLOOR_M


def _make_policy(route=None):
    from molmo_spaces.tasks.enclosure_reach import PactPlaceCorridorPolicy

    policy = PactPlaceCorridorPolicy.__new__(PactPlaceCorridorPolicy)
    robot = type("Robot", (), {"robot_view": object(), "kinematics": object()})()
    env = type("Env", (), {"current_robot": robot, "current_model": object(), "current_data": object()})()
    policy.task = type(
        "Task",
        (),
        {
            "env": env,
            "scene_params": {
                "pact_place_environment_version": ENVIRONMENT_VERSION,
                "pact_v10_pendant_assembly": planning_probe_assembly(),
                "pact_v10_route": route
                or {
                    "inbound_lane_y_m": -0.12,
                    "inbound_padding_m": 0.10,
                    "outbound_lane_y_m": -0.12,
                    "outbound_padding_m": 0.10,
                },
                "intrusion_side": "left",
                "ap_w": 0.85,
            },
        },
    )()
    policy._pact_place_v10_route = {}
    return policy


def _complete_result(**overrides):
    inbound = {
        "rewrite_primitive": ENDPOINT_ONLY_PRIMITIVE,
        "qualification_mode": EMPIRICAL_LIVE_CONTACT_V1,
        "lane_y_m": LEFT_LANE_Y_M,
        "padding_m": SLAB_PADDING_M,
        "min_abs_detour_m": 0.28,
        "detour_meets_minimum": True,
        "fallback_taken": False,
        "clipped": False,
        "wrong_way": False,
        "frozen_endpoint_preserved": True,
        "offline_strict_environment_preclearance_used": False,
        "strict_environment_preclearance_intentionally_not_used": True,
    }
    outbound = dict(inbound)
    outbound["lane_y_m"] = LEFT_LANE_Y_M
    payload = {
        "status": "complete",
        "episode_id": "abc",
        "row_sha256": "row",
        "task_success": True,
        "clean_success": True,
        "pendant_v10": {"inbound": inbound, "outbound": outbound},
        "contact_audit": {
            "contact_class_totals": {
                "hazard_bar": 0,
                "other_environment": 0,
                "clutter": 0,
                "mounted_fixture": 0,
                "place_receptacle": 12,
            }
        },
    }
    payload.update(overrides)
    return payload


class PactPlaceV101ContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = build_contract()

    def test_deterministic_regeneration_and_self_hash(self) -> None:
        again = build_contract()
        self.assertEqual(self.contract["contract_sha256"], again["contract_sha256"])
        payload = dict(self.contract)
        digest = payload.pop("contract_sha256")
        self.assertEqual(digest, sha256_payload(payload))
        self.assertEqual(self.contract["contract_version"], CONTRACT_VERSION)
        self.assertEqual(self.contract["review_stream"], REVIEW_STREAM)
        self.assertEqual(self.contract["gate_stream"], GATE_STREAM)
        self.assertEqual(self.contract["review_master_seed"], REVIEW_MASTER_SEED)
        self.assertEqual(self.contract["gate_master_seed"], GATE_MASTER_SEED)

    def test_exact_12_24_distributions(self) -> None:
        review = self.contract["review_rows"]
        gate = self.contract["gate_rows"]
        self.assertEqual(len(review), N_REVIEW_ROWS)
        self.assertEqual(len(gate), N_GATE_ROWS)
        review_counts = distribution_counts(review)
        gate_counts = distribution_counts(gate)
        self.assertEqual(len(review_counts), 6)
        self.assertEqual(len(gate_counts), 6)
        self.assertTrue(all(count == 2 for count in review_counts.values()))
        self.assertTrue(all(count == 4 for count in gate_counts.values()))
        self.assertEqual(
            {row["layout_family_id"] for row in review + gate},
            set(PHYSICS_CLEAN_FAMILIES),
        )
        self.assertEqual(sum(1 for row in review if row["intrusion_side"] == "left"), 6)
        self.assertEqual(sum(1 for row in review if row["intrusion_side"] == "right"), 6)

    def test_paired_side_clutter_identity_and_seed_disjointness(self) -> None:
        self.assertTrue(paired_side_clutter_identical(self.contract["review_rows"]))
        self.assertTrue(paired_side_clutter_identical(self.contract["gate_rows"]))
        review_seeds = {row["task_seed_u32"] for row in self.contract["review_rows"]}
        gate_seeds = {row["task_seed_u32"] for row in self.contract["gate_rows"]}
        self.assertFalse(review_seeds & gate_seeds)
        self.assertEqual(len(review_seeds), N_REVIEW_ROWS)
        self.assertEqual(len(gate_seeds), N_GATE_ROWS)

    def test_exact_assembly_and_route_values(self) -> None:
        assembly = frozen_assembly()
        lobes = [item for item in assembly["components"] if item.get("role") == "lobe"]
        negative = next(item for item in lobes if item.get("side") == "negative")
        positive = next(item for item in lobes if item.get("side") == "positive")
        self.assertEqual(list(negative["center_m"]), list(PROBE_NEGATIVE_LOBE["center_m"]))
        self.assertEqual(list(negative["half_m"]), list(PROBE_NEGATIVE_LOBE["half_m"]))
        self.assertEqual(list(positive["center_m"]), list(PROBE_POSITIVE_LOBE["center_m"]))
        self.assertEqual(list(positive["half_m"]), list(PROBE_POSITIVE_LOBE["half_m"]))
        self.assertEqual(assembly["assembly_id"], planning_probe_assembly()["assembly_id"])
        for row in self.contract["review_rows"] + self.contract["gate_rows"]:
            route = row["pact_v10_route"]
            expected = frozen_route_for_side(row["intrusion_side"])
            self.assertEqual(route, expected)
            self.assertEqual(route["rewrite_primitive"], ENDPOINT_ONLY_PRIMITIVE)
            self.assertEqual(route["qualification_mode"], EMPIRICAL_LIVE_CONTACT_V1)
            self.assertEqual(route["inbound_padding_m"], SLAB_PADDING_M)
            self.assertEqual(route["outbound_padding_m"], SLAB_PADDING_M)
            self.assertEqual(row["sampler_class"], "PactPlaceCorridorV10CompoundPendantSampler")
            self.assertEqual(
                row["pact_v10_pendant_assembly"]["assembly_id"], assembly["assembly_id"]
            )

    def test_protected_artifact_hashes(self) -> None:
        observed = verify_protected_artifacts()
        self.assertGreaterEqual(len(observed), 10)

    def test_authorizations_remain_false(self) -> None:
        for key in (
            "authorizes_gate",
            "authorizes_collection",
            "authorizes_training",
            "authorizes_evaluation",
            "phase0_passed",
        ):
            self.assertFalse(self.contract[key])


class PactPlaceV101RuntimeTest(unittest.TestCase):
    def test_resolve_historical_and_empirical_markers(self) -> None:
        historical = resolve_v10_runtime_route({})
        self.assertEqual(historical["rewrite_primitive"], GROUP_FREEZE_PRIMITIVE)
        self.assertFalse(historical["use_endpoint_only"])
        self.assertFalse(historical["skip_offline_strict_environment"])
        empirical = resolve_v10_runtime_route(
            {
                "rewrite_primitive": ENDPOINT_ONLY_PRIMITIVE,
                "qualification_mode": EMPIRICAL_LIVE_CONTACT_V1,
            }
        )
        self.assertTrue(empirical["use_endpoint_only"])
        self.assertTrue(empirical["skip_offline_strict_environment"])
        with self.assertRaises(ValueError):
            resolve_v10_runtime_route({"rewrite_primitive": "skip_all"})
        with self.assertRaises(ValueError):
            resolve_v10_runtime_route({"qualification_mode": "bypass_env"})
        with self.assertRaises(ValueError):
            resolve_v10_runtime_route(
                {
                    "rewrite_primitive": GROUP_FREEZE_PRIMITIVE,
                    "qualification_mode": EMPIRICAL_LIVE_CONTACT_V1,
                }
            )

    def test_endpoint_only_dispatch_skips_scalar_env_and_preserves_endpoints(self) -> None:
        from molmo_spaces.policy.solvers.object_manipulation.base_object_manipulation_planner_policy import (
            TCPMoveSegment,
        )

        policy = _make_policy(frozen_route_for_side("left"))
        evaluated = {"n": 0}

        def boom(*_args, **_kwargs):
            evaluated["n"] += 1
            raise AssertionError("scalar environment preclearance must not run")

        policy._v10_evaluate_nominal_and_robust = boom  # type: ignore[method-assign]
        start = policy._place_pose(np.array([0.90, 0.02, 0.88]), np.eye(3))
        mid = policy._place_pose(np.array([0.70, 0.02, 0.88]), np.eye(3))
        end = policy._place_pose(np.array([0.40, 0.02, 0.88]), np.eye(3))
        segments = [
            TCPMoveSegment(name="a", start_pose=start, end_pose=mid, speed=0.2),
            TCPMoveSegment(name="b", start_pose=mid, end_pose=end, speed=0.2),
        ]
        rebuilt = policy._v10_apply_lane(
            segments, prefix="inbound_pendant", include_target=False
        )
        self.assertGreater(len(rebuilt), 0)
        self.assertEqual(evaluated["n"], 0)
        self.assertTrue(
            np.allclose(rebuilt[0].start_pose[:3, 3], start[:3, 3], atol=1e-9)
        )
        self.assertTrue(
            np.allclose(rebuilt[-1].end_pose[:3, 3], end[:3, 3], atol=1e-9)
        )
        record = policy._pact_place_v10_route["inbound_pendant"]
        self.assertEqual(record["rewrite_primitive"], ENDPOINT_ONLY_PRIMITIVE)
        self.assertEqual(record["qualification_mode"], EMPIRICAL_LIVE_CONTACT_V1)
        self.assertFalse(record["offline_strict_environment_preclearance_used"])
        self.assertTrue(record["strict_environment_preclearance_intentionally_not_used"])
        self.assertFalse(record["fallback_taken"])
        self.assertFalse(record["clipped"])
        self.assertFalse(record["wrong_way"])
        self.assertTrue(record["frozen_endpoint_preserved"])
        self.assertGreaterEqual(record["min_abs_detour_m"], 0.05)

    def test_historical_rows_still_call_group_freeze_and_scalar_env(self) -> None:
        from molmo_spaces.policy.solvers.object_manipulation.base_object_manipulation_planner_policy import (
            TCPMoveSegment,
        )

        policy = _make_policy()
        called = {"evaluate": 0}

        def fake_evaluate(planned, **kwargs):
            called["evaluate"] += 1
            self.assertFalse(kwargs.get("endpoint_only"))
            return {
                "nominal": {"pendant_clearance_m": 0.03, "ik_ok": True, "ik_failures": 0},
                "environment": {"environment_clear": True},
                "n_corners_evaluated": 8,
                "robust_ik_ok": True,
                "accepted": True,
                "min_robust_clearance_m": 0.021,
            }

        policy._v10_evaluate_nominal_and_robust = fake_evaluate  # type: ignore[method-assign]
        start = policy._place_pose(np.array([0.90, 0.02, 0.88]), np.eye(3))
        end = policy._place_pose(np.array([0.40, 0.02, 0.88]), np.eye(3))
        rebuilt = policy._v10_apply_lane(
            [TCPMoveSegment(name="a", start_pose=start, end_pose=end, speed=0.2)],
            prefix="inbound_pendant",
            include_target=False,
        )
        self.assertGreater(len(rebuilt), 0)
        self.assertEqual(called["evaluate"], 1)
        record = policy._pact_place_v10_route["inbound_pendant"]
        self.assertEqual(record["rewrite_primitive"], GROUP_FREEZE_PRIMITIVE)
        self.assertTrue(record["offline_strict_environment_preclearance_used"])
        self.assertFalse(record["strict_environment_preclearance_intentionally_not_used"])


class PactPlaceV101ReviewCausalGateTest(unittest.TestCase):
    def test_review_eligibility_and_no_gate_authorization(self) -> None:
        rows = [
            {
                "role_index": index,
                "episode_id": f"e{index:02d}",
                "row_sha256": f"r{index:02d}",
                "layout_family_id": PHYSICS_CLEAN_FAMILIES[index // 4],
                "intrusion_side": "left" if index % 2 == 0 else "right",
            }
            for index in range(12)
        ]
        results = []
        for row in rows:
            inbound_lane = LEFT_LANE_Y_M if row["intrusion_side"] == "left" else RIGHT_LANE_Y_M
            result = _complete_result(
                episode_id=row["episode_id"],
                row_sha256=row["row_sha256"],
            )
            result["pendant_v10"]["inbound"]["lane_y_m"] = inbound_lane
            result["pendant_v10"]["outbound"]["lane_y_m"] = inbound_lane
            results.append(result)
        verdict = review_eligibility(rows, results)
        self.assertTrue(verdict["eligible_for_human_review"])
        self.assertFalse(verdict["authorizes_gate"])
        self.assertFalse(verdict["authorizes_collection"])
        self.assertEqual(verdict["clean_successes"], 12)
        dirty = [dict(item) for item in results]
        dirty[0] = dict(dirty[0], clean_success=False, task_success=False)
        dirty[1] = dict(dirty[1], clean_success=False, task_success=False)
        dirty[2] = dict(dirty[2], clean_success=False, task_success=False)
        self.assertFalse(review_eligibility(rows, dirty)["eligible_for_human_review"])
        missing = [dict(item) for item in results]
        missing[0]["pendant_v10"]["inbound"].pop("rewrite_primitive")
        self.assertFalse(route_telemetry_complete(missing[0]))
        self.assertFalse(review_eligibility(rows, missing)["eligible_for_human_review"])

    def test_lowest_clean_row_is_lowest_role_index(self) -> None:
        rows = [
            {
                "role_index": 3,
                "episode_id": "late",
                "layout_family_id": "F0_target_side_stagger",
                "intrusion_side": "left",
            },
            {
                "role_index": 1,
                "episode_id": "early",
                "layout_family_id": "F0_target_side_stagger",
                "intrusion_side": "left",
            },
        ]
        results = [
            _complete_result(episode_id="late"),
            _complete_result(episode_id="early"),
        ]
        selected = lowest_clean_row_per_cell(rows, results)
        self.assertEqual(
            selected[("F0_target_side_stagger", "left")]["row"]["episode_id"],
            "early",
        )

    def test_causal_floors_side_balance_and_missing_fail_closed(self) -> None:
        metrics = {
            "changed_values": 448,
            "changed_sensors": 3,
            "per_sensor": [
                {"link": "link5_front", "changed_values": 200},
                {"link": "link3", "changed_values": 248},
            ],
        }
        self.assertTrue(_verdict(metrics)["passed"])
        silent = {
            "changed_values": 10,
            "changed_sensors": 1,
            "per_sensor": [{"link": "link3", "changed_values": 10}],
        }
        self.assertFalse(_verdict(silent)["passed"])
        cells = {
            ("F0_target_side_stagger", "left"): {
                "windows": {
                    "inbound": {"changed_values": 500},
                    "outbound": {"changed_values": 500},
                }
            },
            ("F0_target_side_stagger", "right"): {
                "windows": {
                    "inbound": {"changed_values": 2000},
                    "outbound": {"changed_values": 500},
                }
            },
        }
        reports, failures = _side_balance(cells)
        self.assertTrue(any(item["code"] == "missing_side_window" for item in failures))
        balanced = {
            (family, side): {
                "windows": {
                    "inbound": {"changed_values": 500},
                    "outbound": {"changed_values": 600},
                }
            }
            for family in PHYSICS_CLEAN_FAMILIES
            for side in ("left", "right")
        }
        reports, failures = _side_balance(balanced)
        self.assertEqual(failures, [])
        self.assertTrue(all(item["passed"] for item in reports))
        unbalanced = dict(balanced)
        unbalanced[("F1_inner_panel_stagger", "right")] = {
            "windows": {
                "inbound": {"changed_values": 5000},
                "outbound": {"changed_values": 600},
            }
        }
        _reports, failures = _side_balance(unbalanced)
        self.assertTrue(any(item["code"] == "side_imbalance" for item in failures))
        self.assertEqual(ABS_DELTA_FLOOR_M, 1.0e-5)

    def test_gate_requires_bound_approval_and_keeps_collection_false(self) -> None:
        with self.assertRaises(PermissionError):
            assert_phase0_approval(
                None,
                review_manifest_sha256="a",
                causal_artifact_sha256="b",
                contract_sha256="c",
            )
        with self.assertRaises(PermissionError):
            assert_phase0_approval(
                {
                    "decision": "approve_phase0",
                    "review_manifest_sha256": "nope",
                    "causal_artifact_sha256": "b",
                    "contract_sha256": "c",
                },
                review_manifest_sha256="a",
                causal_artifact_sha256="b",
                contract_sha256="c",
            )
        assert_phase0_approval(
            {
                "decision": "approve_phase0",
                "review_manifest_sha256": "a",
                "causal_artifact_sha256": "b",
                "contract_sha256": "c",
            },
            review_manifest_sha256="a",
            causal_artifact_sha256="b",
            contract_sha256="c",
        )
        from pact_place_v101_empirical_qualification_contract import empty_authorization

        auth = empty_authorization()
        self.assertFalse(auth["authorizes_collection"])
        self.assertFalse(auth["phase0_passed"])


class PactPlaceV101GeometryAdmissionTest(unittest.TestCase):
    def test_synthetic_stock_admits_frozen_lanes(self) -> None:
        inbound = np.array([[0.90, 0.02, 0.88], [0.40, 0.02, 0.88]], dtype=float)
        outbound = np.array([[0.40, 0.02, 0.88], [0.90, 0.02, 0.88]], dtype=float)
        rotations = np.stack([np.eye(3), np.eye(3)])
        left_in = admit_fixed_route_on_stock(
            inbound, rotations, panel_side="left", direction="inbound"
        )
        left_out = admit_fixed_route_on_stock(
            outbound, rotations, panel_side="left", direction="outbound"
        )
        right_in = admit_fixed_route_on_stock(
            inbound, rotations, panel_side="right", direction="inbound"
        )
        self.assertTrue(left_in["admitted"])
        self.assertTrue(left_out["admitted"])
        self.assertTrue(right_in["admitted"])
        self.assertEqual(left_in["lane_y_m"], LEFT_LANE_Y_M)
        self.assertEqual(right_in["lane_y_m"], RIGHT_LANE_Y_M)
        planned = plan_lane_endpoint_only(
            inbound,
            rotations,
            assembly=planning_probe_assembly(),
            panel_side="left",
            lane_y_m=LEFT_LANE_Y_M,
            padding_m=SLAB_PADDING_M,
            freeze_start=False,
            freeze_final=True,
        )
        self.assertTrue(planned["frozen_endpoints"]["preserved"])
        self.assertGreaterEqual(planned["detour"]["min_abs_detour_m"], 0.05)

    def test_snapshot_stock_admits_all_twelve_evaluations(self) -> None:
        snapshot_root = ROOT / Path(V99_SNAPSHOT_RELATIVE).parent
        _meta, cells = load_clean_snapshots(snapshot_root)
        reports = admit_six_cell_fixed_route(cells)
        self.assertEqual(len(reports), 12)
        self.assertTrue(all(item["admitted"] for item in reports))


if __name__ == "__main__":
    unittest.main()
