"""Infrastructure guards for the V10.10 paired evaluator."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_pact_place_v1010_eval as runner  # noqa: E402
import finalize_pact_place_v1010_eval as finalizer  # noqa: E402


class V1010EvalInfrastructureTests(unittest.TestCase):
    def test_native_thread_limits_override_inherited_values(self) -> None:
        inherited = {key: "128" for key in runner.NATIVE_THREAD_ENV}
        with mock.patch.dict(os.environ, inherited, clear=False):
            environment = runner.subprocess_environment(h5_only=True)
        self.assertEqual(environment["OMP_NUM_THREADS"], "1")
        self.assertEqual(environment["OMP_THREAD_LIMIT"], "1")
        self.assertEqual(environment["OPENBLAS_NUM_THREADS"], "1")
        self.assertEqual(environment["MKL_NUM_THREADS"], "1")
        self.assertEqual(environment["PACT_V109_TRAJECTORY_H5_ONLY"], "1")

    def test_h5_flag_is_removed_when_not_requested(self) -> None:
        with mock.patch.dict(
            os.environ, {"PACT_V109_TRAJECTORY_H5_ONLY": "1"}, clear=False
        ):
            environment = runner.subprocess_environment(h5_only=False)
        self.assertNotIn("PACT_V109_TRAJECTORY_H5_ONLY", environment)

    def test_worker_pool_above_registered_limit_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "PID/thread budget"):
            runner.run_stage(
                [], Path("manifest.json"), Path("output"), {}, "test",
                runner.MAX_EVAL_WORKERS + 1,
            )

    def test_registered_limit_is_four(self) -> None:
        self.assertEqual(runner.MAX_EVAL_WORKERS, 4)
        self.assertEqual(runner.DEFAULT_ROLLOUT_TIMEOUT_MINUTES, 45.0)

    def test_finalizer_uses_the_registered_primary_and_balanced_corpus(self) -> None:
        self.assertEqual(finalizer.PRIMARY_ENDPOINT,
                         "collision_free_task_success")
        self.assertEqual(finalizer.TRAIN_COUNT, 120)
        self.assertEqual(finalizer.VALIDATION_COUNT, 24)

    def test_finalizer_rejects_partial_or_mismatched_run(self) -> None:
        manifest = {
            "role": "role", "manifest_sha256": "manifest",
            "rows": [{"candidate_index": 0}],
        }
        verification = {
            "arms": {
                "act": {"hashes": {"policy_best.ckpt": "act"}},
                "pact": {"hashes": {"policy_best.ckpt": "pact"}},
            }
        }
        run = {
            "schema_version": "pact_place_v1010_full_run_v1",
            "role": "role", "manifest_sha256": "manifest",
            "eval_root": "root", "instances": 1,
            "rollouts_attempted": 1, "rollouts_complete": 1,
            "failures": [{"status": "no_result"}], "results": [],
            "arms": {
                "ACT": {"checkpoint_sha256": "act"},
                "PACT": {"checkpoint_sha256": "pact"},
            },
        }
        problems = finalizer.validate_completed_run(
            run, manifest, verification, eval_root="root")
        self.assertTrue(any("did not attempt exactly 2" in p for p in problems))
        self.assertTrue(any("infrastructure failures" in p for p in problems))


if __name__ == "__main__":
    unittest.main()
