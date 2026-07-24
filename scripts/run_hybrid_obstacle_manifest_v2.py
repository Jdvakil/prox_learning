#!/usr/bin/env python3
"""Dedicated launcher for the manifest-driven hybrid-obstacle collection.

This is the only entry point that drives ``ManifestRolloutRunner``. It does not
change the behavior of ``ParallelRolloutRunner`` for any other config.

The parent validates the committed manifest and therefore knows the complete
candidate-ID set before a single worker launches. Rows -- not houses -- go into
the work queue, each is claimed atomically exactly once, and the run exits
nonzero if any row is left unreconciled.

Usage:
    python scripts/run_hybrid_obstacle_manifest_v2.py \
        --output-dir /path/to/run_a --workers 1 --smoke

Environment (see prox_learning/README.md):
    MUJOCO_GL=egl
    MLSPACES_ASSETS_DIR=<root>/assets
    PYTHONPATH=<root>/submodules/molmospaces
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submodules" / "molmospaces"))
os.environ.setdefault("MLSPACES_ASSETS_DIR", str(ROOT / "assets"))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

DEFAULT_MANIFEST = ROOT / "configs" / "hybrid_obstacle_candidate_manifest_v2.json"
DEFAULT_SMOKE = ROOT / "configs" / "hybrid_obstacle_manifest_v2_smoke8.json"
CONTRACT = ROOT / "configs" / "hybrid_obstacle_independent_v2.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run only the committed 8-row smoke subset",
    )
    parser.add_argument("--smoke-subset", type=Path, default=DEFAULT_SMOKE)
    parser.add_argument(
        "--run-id",
        default=None,
        help="explicit run identifier; a fresh one per invocation is the default, "
        "which is what makes an abandoned claim distinguishable from a live one",
    )
    args = parser.parse_args()

    # Imported after sys.path/env setup, and after argument parsing so that
    # --help stays fast.
    from molmo_spaces.data_generation.config_registry import get_config_class
    from molmo_spaces.data_generation.main import auto_import_configs
    from molmo_spaces.data_generation.manifest_runner import ManifestRolloutRunner
    from molmo_spaces.data_generation.runtime_compat import assert_supported_runtime

    # Fail fast on an unsupported MuJoCo/Warp, rather than letting it surface as
    # an empty dataset. Preserved from the worker-completeness work.
    assert_supported_runtime(strict=True)

    auto_import_configs()
    config_class = get_config_class("FrankaSkinHybridObstacleManifestV2Config")

    manifest = json.loads(args.manifest.read_text())
    config = config_class(
        output_dir=args.output_dir,
        num_workers=args.workers,
        manifest_path=str(args.manifest),
        smoke_subset_path=str(args.smoke_subset) if args.smoke else None,
        run_id=args.run_id or uuid.uuid4().hex,
        expected_sensor_order_sha256=manifest["sensor_order_sha256"],
        expected_env_config_sha256=manifest["env_config_sha256"],
        expected_runtime_contract_sha256=manifest["runtime_contract_sha256"],
    )

    print(f"manifest        {args.manifest}")
    print(f"manifest_sha256 {manifest['manifest_sha256']}")
    print(f"subset          {'smoke8' if args.smoke else 'full 160-row'}")
    print(f"output_dir      {args.output_dir}")
    print(f"workers         {args.workers}")
    print(f"run_id          {config.run_id}")

    runner = ManifestRolloutRunner(config)
    succeeded, total = runner.run()
    print(f"rows succeeded  {succeeded}/{total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
