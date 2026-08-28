#!/usr/bin/env python3
"""V10.4 Step-1: the six fixed production qualification rows.

Only a passing immutable Step 0 authorizes these episodes. Runs exactly the
prewritten six-row manifest through the normal live rollout path. No row is
resampled, substituted, or replaced. The six-row outcome is a qualification
check, not a clean-rate estimate.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "scripts", ROOT / "submodules" / "molmospaces"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from pact_place_v104_contract import (  # noqa: E402
    CONTRACT_VERSION,
    ENVIRONMENT_VERSION,
    N_REVIEW_ROWS,
    PREFLIGHT_ROOT,
    PRODUCTION_ROOT,
    REVIEW_MASTER_SEED,
    REVIEW_MIN_CLEARANCE_M,
    REVIEW_STREAM,
    SAMPLER_CLASS,
    build_contract,
    empty_authorization,
    implementation_sha256,
    is_clean_success,
    review_eligibility,
    row_defects,
    sha256_payload,
    verify_protected_artifacts,
    write_immutable_create_only,
)
from pact_place_v104_geometry import SCENE_XML_RELATIVE_V104  # noqa: E402
from run_pact_place_expert_screen import _result_path, run_row  # noqa: E402

SCENE_XML = ROOT / SCENE_XML_RELATIVE_V104


def load_preflight(path: Path, contract: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"missing Step-0 preflight: {path}")
    document = json.loads(path.read_text())
    if not document.get("preflight_passed"):
        raise SystemExit("Step 0 did not pass; no V10.4 episode is authorized")
    if document.get("implementation_sha256") != contract["implementation_sha256"]:
        raise SystemExit(
            "the implementation changed after Step 0; rerun the preflight before "
            "generating episodes"
        )
    if document.get("scene_sha256") != contract["scene_sha256"]:
        raise SystemExit("the production scene changed after Step 0")
    return document


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=ROOT / PRODUCTION_ROOT)
    parser.add_argument("--preflight", type=Path, default=ROOT / PREFLIGHT_ROOT / "preflight.json")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("MLSPACES_ASSETS_DIR", str(ROOT / "assets"))
    os.environ.pop("DISPLAY", None)

    contract = build_contract()
    preflight = load_preflight(args.preflight.resolve(), contract)
    provenance = verify_protected_artifacts(contract["protected_artifacts"])
    if not provenance["passed"]:
        raise SystemExit(f"protected artifact drift: {provenance['mismatches'][:3]}")

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rows = list(contract["review_rows"])
    if len(rows) != N_REVIEW_ROWS:
        raise RuntimeError("the production manifest must contain exactly six rows")

    config = {
        "schema_version": "pact_place_v104_production_config_v1",
        "contract_version": CONTRACT_VERSION,
        "environment_version": ENVIRONMENT_VERSION,
        "contract_sha256": contract["contract_sha256"],
        "implementation_sha256": implementation_sha256(),
        "preflight_sha256": preflight.get("artifact_sha256"),
        "scene_xml": SCENE_XML_RELATIVE_V104,
        "scene_sha256": contract["scene_sha256"],
        "sampler_class": SAMPLER_CLASS,
        "review_stream": REVIEW_STREAM,
        "review_master_seed": REVIEW_MASTER_SEED,
        "n_rows": N_REVIEW_ROWS,
        "expert_screen_rows": rows,
        **empty_authorization(),
    }
    config["config_sha256"] = sha256_payload(config)
    config_path = output_root / "config.json"
    if not config_path.is_file():
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")

    results: list[dict[str, Any]] = []
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max(1, min(args.workers, len(rows))),
        mp_context=context,
        max_tasks_per_child=1,
    ) as executor:
        futures = {
            executor.submit(
                run_row,
                row,
                config_sha256=config["config_sha256"],
                output_root=str(output_root),
                scene_xml=str(SCENE_XML),
            ): row
            for row in rows
        }
        for future in concurrent.futures.as_completed(futures):
            row = futures[future]
            result = future.result()
            results.append(result)
            frames = result.get("pact_v104_frame_telemetry") or {}
            print(
                f"row {row['role_index']} {row['intrusion_side']:<5} "
                f"status={result['status']} clean={is_clean_success(result, min_clearance_m=REVIEW_MIN_CLEARANCE_M)} "
                f"min_clearance={frames.get('min_clearance_m')} "
                f"contact_frames={frames.get('pendant_contact_frames')} "
                f"steps={result.get('episode_steps')}",
                flush=True,
            )
    results.sort(key=lambda item: int(item["role_index"]))
    eligibility = review_eligibility(rows, results)
    manifest = {
        "schema_version": "pact_place_v104_production_manifest_v1",
        "contract_version": CONTRACT_VERSION,
        "environment_version": ENVIRONMENT_VERSION,
        "contract_sha256": contract["contract_sha256"],
        "config_sha256": config["config_sha256"],
        "preflight_sha256": preflight.get("artifact_sha256"),
        "implementation_sha256": implementation_sha256(),
        "scene_sha256": contract["scene_sha256"],
        "review_stream": REVIEW_STREAM,
        "n_rows": N_REVIEW_ROWS,
        "rows": rows,
        "results": [
            {
                "role_index": item["role_index"],
                "episode_id": item["episode_id"],
                "intrusion_side": item.get("intrusion_side"),
                "status": item["status"],
                "task_success": item.get("task_success"),
                "grasp_phase_success": item.get("grasp_phase_success"),
                "place_phase_success": item.get("place_phase_success"),
                "clean_success": item.get("clean_success"),
                "v104_clean_success": is_clean_success(
                    item, min_clearance_m=REVIEW_MIN_CLEARANCE_M
                ),
                "v104_defects": row_defects(item, min_clearance_m=REVIEW_MIN_CLEARANCE_M),
                "failure_cause": item.get("failure_cause"),
                "episode_steps": item.get("episode_steps"),
                "contact_audit": item.get("contact_audit"),
                "pact_v104_frame_telemetry": item.get("pact_v104_frame_telemetry"),
                "pact_v104_speed_amendment": item.get("pact_v104_speed_amendment"),
                "result_sha256": item.get("result_sha256"),
                "trajectory_sha256": item.get("trajectory_sha256"),
            }
            for item in results
        ],
        "eligibility": eligibility,
        "clean_rate_is_not_an_estimate": True,
        **empty_authorization(),
    }
    digest = write_immutable_create_only(
        output_root / "production_manifest.json", manifest
    )
    print(json.dumps({
        "production_pack_passed": eligibility["production_pack_passed"],
        "clean_successes": eligibility["clean_successes"],
        "clean_by_side": eligibility["clean_by_side"],
        "min_observed_clearance_m": eligibility["min_observed_clearance_m"],
        "artifact_sha256": digest,
    }, indent=2))
    return 0 if eligibility["production_pack_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
