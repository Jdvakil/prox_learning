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

from pact_geom_distance import CONTACT_DISTANCE_M
from pact_place_v10_compound_pendant_contract import (
    ENVIRONMENT_VERSION,
    MIN_NOMINAL_CLEARANCE_M,
    MIN_ROBUST_CLEARANCE_M,
    NOMINAL_PERTURBATION_INDEX,
    REGISTERED_COMPLETE_SIGNAL_SCREEN_LIMIT,
    SIGNAL_SCREEN_LIMIT_UNREGISTERED,
)
from pact_place_v10_geometry import (
    build_assembly,
    lobe_from_key,
    planning_probe_assembly,
    union_fixture,
)
from pact_place_v10_route import (
    evaluate_environment_no_intersection,
    min_assembly_pendant_clearance,
    plan_lane,
    plan_lane_at_parameters,
    qpos_dicts_equal,
    route_ik_cache_key,
    sequential_ik_split_clearance,
    signal_screen_admission,
)


PENDANT_IDS = [10, 11]
ENV_IDS = [20, 21]
ROBOT_IDS = [1, 2]
TARGET_IDS = [99]


class FakeKinematics:
    def __init__(self, mode: str = "success") -> None:
        self.mode = mode
        self.calls: list[dict] = []

    def ik(self, gid, pose, groups, seed, base_pose=None):
        self.calls.append(
            {
                "pose": np.asarray(pose, dtype=float).copy(),
                "seed": {key: np.asarray(value).copy() for key, value in seed.items()},
            }
        )
        if self.mode == "fail":
            return None
        if self.mode == "raise" and len(self.calls) >= 2:
            raise RuntimeError("injected ik exception")
        updated = {key: np.asarray(value, dtype=float).copy() for key, value in seed.items()}
        updated["arm"] = np.asarray(updated["arm"], dtype=float) + 1.0
        return updated


class FakeRobotView:
    def __init__(self, initial: dict | None = None) -> None:
        self.qpos = {
            key: np.asarray(value, dtype=float).copy()
            for key, value in (initial or {"arm": np.array([0.0, 1.0])}).items()
        }
        self.sets: list[dict] = []

        class _Base:
            pose = np.eye(4)

        self.base = _Base()

    def get_qpos_dict(self):
        return {key: np.asarray(value).copy() for key, value in self.qpos.items()}

    def set_qpos_dict(self, qpos):
        copied = {key: np.asarray(value).copy() for key, value in qpos.items()}
        self.sets.append(copied)
        self.qpos = copied

    def get_gripper_movegroup_ids(self):
        return [0]

    def move_group_ids(self):
        return [0]


def _make_policy(*, kinematics=None, view=None):
    from molmo_spaces.tasks.enclosure_reach import PactPlaceCorridorPolicy

    view = view or FakeRobotView()
    kinematics = kinematics or FakeKinematics("success")
    policy = PactPlaceCorridorPolicy.__new__(PactPlaceCorridorPolicy)
    robot = type("Robot", (), {"robot_view": view, "kinematics": kinematics})()
    env = type("Env", (), {"current_robot": robot, "current_model": object(), "current_data": object()})()
    policy.task = type(
        "Task",
        (),
        {
            "env": env,
            "scene_params": {
                "pact_place_environment_version": ENVIRONMENT_VERSION,
                "pact_v10_pendant_assembly": planning_probe_assembly(),
                "pact_v10_route": {
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
    policy._v99_collision_robot_geom_ids = lambda: list(ROBOT_IDS)
    policy._v99_target_geom_ids = lambda: list(TARGET_IDS)
    policy._v10_active_pendant_geom_ids = lambda: list(PENDANT_IDS)
    policy._v10_strict_environment_geom_ids = lambda: list(ENV_IDS)
    return policy, view, kinematics


def _distance_fn(pendant, environment):
    def true_distance(model, data, probe, obstacles):
        ids = [int(item) for item in obstacles]
        if set(ids) == set(PENDANT_IDS):
            return pendant() if callable(pendant) else pendant
        if set(ids) == set(ENV_IDS):
            return environment() if callable(environment) else environment
        raise AssertionError(f"unexpected obstacle ids {ids}")

    return true_distance


class PactPlaceV10RouteRuntimeTest(unittest.TestCase):
    def test_pendant_26mm_env_1mm_passes(self) -> None:
        policy, view, kin = _make_policy()
        saved = view.get_qpos_dict()
        with mock.patch("pact_geom_distance.true_distance", _distance_fn(0.026, 0.001)):
            with mock.patch("mujoco.mj_forward"):
                report = policy._v10_sequential_ik_clearance(
                    np.array([[0.8, 0.0, 0.9], [0.7, -0.12, 0.9]]),
                    np.stack([np.eye(3), np.eye(3)]),
                    include_target=False,
                )
        self.assertTrue(report["ik_ok"])
        self.assertTrue(report["meets_pendant"])
        self.assertTrue(report["environment_clear"])
        self.assertGreaterEqual(report["pendant_clearance_m"], MIN_NOMINAL_CLEARANCE_M)
        self.assertLess(report["environment_clearance_m"], MIN_NOMINAL_CLEARANCE_M)
        self.assertTrue(report["environment_clear"])
        self.assertTrue(qpos_dicts_equal(view.get_qpos_dict(), saved))
        self.assertEqual(report["probe_ids"], ROBOT_IDS)
        self.assertFalse(any(item in report["pendant_geom_ids"] for item in ENV_IDS))

    def test_strict_environment_contact_at_threshold_fails(self) -> None:
        policy, _view, _kin = _make_policy()
        with mock.patch(
            "pact_geom_distance.true_distance",
            _distance_fn(0.026, CONTACT_DISTANCE_M),
        ):
            with mock.patch("mujoco.mj_forward"):
                report = policy._v10_sequential_ik_clearance(
                    np.array([[0.8, 0.0, 0.9]]),
                    np.stack([np.eye(3)]),
                    include_target=False,
                )
        self.assertTrue(report["meets_pendant"])
        self.assertFalse(report["environment_clear"])
        self.assertFalse(report["accepted"])
        self.assertFalse(
            evaluate_environment_no_intersection([CONTACT_DISTANCE_M])["environment_clear"]
        )

    def test_pendant_nominal_24mm_fails(self) -> None:
        policy, _view, _kin = _make_policy()
        with mock.patch("pact_geom_distance.true_distance", _distance_fn(0.024, 0.001)):
            with mock.patch("mujoco.mj_forward"):
                report = policy._v10_sequential_ik_clearance(
                    np.array([[0.8, 0.0, 0.9]]),
                    np.stack([np.eye(3)]),
                    include_target=False,
                )
        self.assertFalse(report["meets_pendant"])
        self.assertTrue(report["environment_clear"])
        self.assertFalse(report["accepted"])

    def test_one_robust_corner_19mm_fails(self) -> None:
        policy, _view, _kin = _make_policy()
        assembly = planning_probe_assembly()
        positions = np.array([[0.90, 0.02, 0.88], [0.40, 0.02, 0.88]], dtype=float)
        rotations = np.stack([np.eye(3), np.eye(3)])
        planned = plan_lane(
            positions,
            rotations,
            assembly=assembly,
            panel_side="left",
            lane_y_m=-0.12,
            padding_m=0.10,
        )
        pendant_by_call = [0.026] + [0.021] * 7 + [0.019]
        calls = {"n": 0}

        def wrapped(positions_m, rotations_m, **kwargs):
            index = calls["n"]
            calls["n"] += 1
            pendant = pendant_by_call[index]
            min_p = float(kwargs.get("min_pendant_m") or MIN_NOMINAL_CLEARANCE_M)
            return {
                "ik_ok": True,
                "ik_failures": 0,
                "pendant_clearance_m": pendant,
                "environment_clearance_m": 0.001,
                "meets_pendant": pendant + 1e-12 >= min_p,
                "environment_clear": True,
                "accepted": pendant + 1e-12 >= min_p,
            }

        policy._v10_sequential_ik_clearance = wrapped  # type: ignore[method-assign]
        report = policy._v10_evaluate_nominal_and_robust(
            planned,
            include_target=False,
            fixture=planned["union_fixture"],
            freeze_start=False,
            freeze_final=True,
            aperture_width_m=0.85,
            panel_side="left",
            padding_m=0.10,
        )
        self.assertEqual(calls["n"], 9)
        self.assertEqual(report["n_corners_evaluated"], 8)
        self.assertTrue(report["pendant"]["meets_nominal"])
        self.assertFalse(report["pendant"]["meets_robust"])
        self.assertEqual(
            sum(1 for item in report["pendant"]["robust_corners"] if not item["meets_robust"]),
            1,
        )
        self.assertFalse(report["accepted"])

    def test_none_distance_fails_closed(self) -> None:
        policy, _view, _kin = _make_policy()
        with mock.patch("pact_geom_distance.true_distance", _distance_fn(None, 0.001)):
            with mock.patch("mujoco.mj_forward"):
                missing_pendant = policy._v10_sequential_ik_clearance(
                    np.array([[0.8, 0.0, 0.9]]),
                    np.stack([np.eye(3)]),
                    include_target=False,
                )
        with mock.patch("pact_geom_distance.true_distance", _distance_fn(0.026, None)):
            with mock.patch("mujoco.mj_forward"):
                missing_env = policy._v10_sequential_ik_clearance(
                    np.array([[0.8, 0.0, 0.9]]),
                    np.stack([np.eye(3)]),
                    include_target=False,
                )
        self.assertFalse(missing_pendant["meets_pendant"])
        self.assertTrue(missing_pendant["missing_pendant_distance"])
        self.assertFalse(missing_env["environment_clear"])
        self.assertTrue(missing_env["missing_environment_distance"])
        self.assertFalse(evaluate_environment_no_intersection([None])["environment_clear"])

    def test_eight_distinct_perturbation_evaluations(self) -> None:
        policy, _view, kin = _make_policy()
        assembly = planning_probe_assembly()
        positions = np.array([[0.90, 0.02, 0.88], [0.40, 0.02, 0.88]], dtype=float)
        rotations = np.stack([np.eye(3), np.eye(3)])
        planned = plan_lane(
            positions,
            rotations,
            assembly=assembly,
            panel_side="left",
            lane_y_m=-0.12,
            padding_m=0.10,
        )
        captured = []

        def wrapped(positions_m, rotations_m, **kwargs):
            captured.append(
                {
                    "first_y": float(np.asarray(positions_m)[len(positions_m) // 2, 1]),
                    "min_pendant_m": kwargs.get("min_pendant_m"),
                    "path": np.asarray(positions_m, dtype=float).copy(),
                }
            )
            return {
                "ik_ok": True,
                "ik_failures": 0,
                "pendant_clearance_m": 0.03,
                "environment_clearance_m": 0.001,
                "meets_pendant": True,
                "environment_clear": True,
                "accepted": True,
            }

        policy._v10_sequential_ik_clearance = wrapped  # type: ignore[method-assign]
        report = policy._v10_evaluate_nominal_and_robust(
            planned,
            include_target=False,
            fixture=planned["union_fixture"],
            freeze_start=False,
            freeze_final=True,
            aperture_width_m=0.85,
            panel_side="left",
            padding_m=0.10,
        )
        self.assertEqual(len(captured), 9)
        self.assertEqual(report["n_corners_evaluated"], 8)
        corner_paths = captured[1:]
        self.assertEqual(len(corner_paths), 8)
        unique = {tuple(np.round(item["path"][:, :2], 6).ravel()) for item in corner_paths}
        self.assertEqual(len(unique), 8)
        self.assertTrue(
            any(
                abs(item["first_y"] - captured[0]["first_y"]) > 1e-6
                for item in corner_paths
            )
        )
        self.assertEqual(captured[0]["min_pendant_m"], MIN_NOMINAL_CLEARANCE_M)
        self.assertTrue(all(item["min_pendant_m"] == MIN_ROBUST_CLEARANCE_M for item in corner_paths))

    def test_cache_keys_differ_across_route_identity(self) -> None:
        union_a = (0.69, -0.24, 0.82, 0.71, 0.24, 1.515)
        union_b = (0.70, -0.24, 0.82, 0.72, 0.24, 1.515)
        base = dict(
            cell_role_index=600,
            direction="inbound",
            union_key=union_a,
            padding_m=0.10,
            lane_y_m=-0.12,
            perturbation_index=0,
        )
        keys = [
            route_ik_cache_key(**base),
            route_ik_cache_key(**{**base, "perturbation_index": 1}),
            route_ik_cache_key(**{**base, "perturbation_index": NOMINAL_PERTURBATION_INDEX}),
            route_ik_cache_key(**{**base, "cell_role_index": 601}),
            route_ik_cache_key(**{**base, "direction": "outbound"}),
            route_ik_cache_key(**{**base, "lane_y_m": -0.13}),
            route_ik_cache_key(**{**base, "padding_m": 0.12}),
            route_ik_cache_key(**{**base, "union_key": union_b}),
        ]
        self.assertEqual(len(keys), len(set(keys)))

    def test_union_equivalent_morphologies_keep_distinct_clearance(self) -> None:
        probes = [
            {
                "lo": np.array([0.68, -0.17, 0.88]),
                "hi": np.array([0.72, -0.13, 0.92]),
                "gtype": 6,
                "pos": np.array([0.70, -0.15, 0.90]),
                "mat": np.eye(3).ravel(),
                "size": np.array([0.02, 0.02, 0.02]),
                "verts": None,
            }
        ]
        negative_a = lobe_from_key((0.70, -0.20, 0.90, 0.02, 0.04, 0.04))
        negative_b = lobe_from_key((0.70, -0.22, 0.90, 0.02, 0.02, 0.04))
        positive = lobe_from_key((0.70, 0.20, 0.90, 0.02, 0.04, 0.04))
        a = build_assembly([negative_a, positive])
        b = build_assembly([negative_b, positive])
        from pact_place_v10_catalog import union_key_from_two_lobe_key as catalog_union

        self.assertEqual(
            catalog_union([negative_a["key"], positive["key"]]),
            catalog_union([negative_b["key"], positive["key"]]),
        )
        clear_a = min_assembly_pendant_clearance(a["components"], probes)
        clear_b = min_assembly_pendant_clearance(b["components"], probes)
        self.assertIsNotNone(clear_a)
        self.assertIsNotNone(clear_b)
        self.assertNotEqual(clear_a, clear_b)
        self.assertNotEqual(a["volume_m3"], b["volume_m3"])

    def test_qpos_restored_after_success_failure_and_exception(self) -> None:
        initial = {"arm": np.array([3.0, 4.0])}
        positions = np.array([[0.8, 0.0, 0.9], [0.7, -0.12, 0.9]])
        rotations = np.stack([np.eye(3), np.eye(3)])
        for mode in ("success", "fail", "raise"):
            view = FakeRobotView(initial)
            kin = FakeKinematics(mode)
            policy, _, _ = _make_policy(kinematics=kin, view=view)
            policy.task.env.current_robot.robot_view = view
            policy.task.env.current_robot.kinematics = kin
            with mock.patch("pact_geom_distance.true_distance", _distance_fn(0.026, 0.001)):
                with mock.patch("mujoco.mj_forward"):
                    if mode == "raise":
                        with self.assertRaises(RuntimeError):
                            policy._v10_sequential_ik_clearance(
                                positions, rotations, include_target=False
                            )
                    else:
                        policy._v10_sequential_ik_clearance(
                            positions, rotations, include_target=False
                        )
            self.assertTrue(qpos_dicts_equal(view.get_qpos_dict(), initial), mode)
            self.assertTrue(qpos_dicts_equal(view.sets[-1], initial), mode)

    def test_environment_contact_can_abort_remaining_waypoints(self) -> None:
        initial = {"arm": np.array([0.0, 0.0])}
        view = FakeRobotView(initial)
        kin = FakeKinematics("success")
        env_calls = {"n": 0}

        def measure_environment():
            env_calls["n"] += 1
            return CONTACT_DISTANCE_M

        scored = sequential_ik_split_clearance(
            np.array([[0.8, 0.0, 0.9], [0.7, -0.12, 0.9], [0.6, -0.12, 0.9]]),
            np.stack([np.eye(3)] * 3),
            saved_qpos=initial,
            set_qpos=view.set_qpos_dict,
            get_qpos=view.get_qpos_dict,
            solve_ik=lambda pose, seed: kin.ik(0, pose, [], seed),
            forward=lambda: None,
            place_pose=lambda p, r: np.eye(4),
            measure_pendant=lambda: 1.0,
            measure_environment=measure_environment,
            min_pendant_m=0.025,
            abort_on_ik_failure=True,
            abort_on_environment_failure=True,
        )
        self.assertTrue(scored["ik_ok"])
        self.assertFalse(scored["environment_clear"])
        self.assertEqual(env_calls["n"], 1)
        self.assertEqual(len(kin.calls), 1)
        self.assertTrue(qpos_dicts_equal(view.get_qpos_dict(), initial))

    def test_inbound_probes_robot_outbound_adds_target(self) -> None:
        policy, _view, _kin = _make_policy()
        with mock.patch("pact_geom_distance.true_distance", _distance_fn(0.026, 0.001)):
            with mock.patch("mujoco.mj_forward"):
                inbound = policy._v10_sequential_ik_clearance(
                    np.array([[0.8, 0.0, 0.9]]),
                    np.stack([np.eye(3)]),
                    include_target=False,
                )
                outbound = policy._v10_sequential_ik_clearance(
                    np.array([[0.8, 0.0, 0.9]]),
                    np.stack([np.eye(3)]),
                    include_target=True,
                )
        self.assertEqual(inbound["probe_ids"], ROBOT_IDS)
        self.assertEqual(outbound["probe_ids"], ROBOT_IDS + TARGET_IDS)

    def test_perturbed_parameters_change_generated_route(self) -> None:
        assembly = planning_probe_assembly()
        fixture = union_fixture(assembly)
        positions = np.array([[0.90, 0.02, 0.88], [0.40, 0.02, 0.88]], dtype=float)
        rotations = np.stack([np.eye(3), np.eye(3)])
        planned = plan_lane(
            positions,
            rotations,
            assembly=assembly,
            panel_side="left",
            lane_y_m=-0.12,
            padding_m=0.10,
        )
        corner = planned["perturbation_corners"][0]
        perturbed = plan_lane_at_parameters(
            planned["stock_positions_m"],
            planned["stock_rotations"],
            fixture=fixture,
            panel_side="left",
            lane_y_m=float(corner["lane_y_m"]),
            padding_m=0.10,
            entry_x_m=float(corner["entry_x_m"]),
            exit_x_m=float(corner["exit_x_m"]),
            freeze_start=False,
            freeze_final=True,
        )
        self.assertFalse(
            planned["planned_positions_m"].shape == perturbed["planned_positions_m"].shape
            and np.allclose(planned["planned_positions_m"], perturbed["planned_positions_m"])
        )
        self.assertNotEqual(planned["lane_y_m"], perturbed["lane_y_m"])

    def test_signal_screen_limit_is_unregistered(self) -> None:
        self.assertIsNone(REGISTERED_COMPLETE_SIGNAL_SCREEN_LIMIT)
        decision = signal_screen_admission(1)
        self.assertFalse(decision["signal_screen_run"])
        self.assertEqual(decision["stop_reason"], SIGNAL_SCREEN_LIMIT_UNREGISTERED)
        self.assertFalse(decision["post_hoc_shortlist"])


if __name__ == "__main__":
    unittest.main()
