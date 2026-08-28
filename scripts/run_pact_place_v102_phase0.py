#!/usr/bin/env python3
"""V10.2 Step-4 twenty-four-row Phase-0 gate.

Refuses to run without an owner ``approve_phase0`` bound to the V10.2 contract,
preflight, six-row screen, review, and causal hashes. On pass it writes
``eligible_for_separate_collection_authorization: true`` and leaves collection,
training, and evaluation unauthorized. On fail it writes a permanent stop.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "scripts", ROOT / "submodules" / "molmospaces"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from pact_place_v10_compound_pendant_contract import (  # noqa: E402
    PLACE_V10_SCENE_SHA256,
    SCENE_XML_RELATIVE,
)
from pact_place_v10_runtime import establish_v10_runtime_env, write_immutable  # noqa: E402
from pact_place_v102_raised_pendant_contract import (  # noqa: E402
    CONTRACT_VERSION,
    ENVIRONMENT_VERSION,
    GATE_MASTER_SEED,
    GATE_STREAM,
    N_GATE_ROWS,
    SAMPLER_CLASS,
    assert_phase0_approval,
    build_contract,
    empty_authorization,
    gate_eligibility,
    implementation_sha256,
    is_v102_clean_success,
    row_defects,
    sha256_payload,
)
from run_pact_place_expert_screen import (  # noqa: E402
    TERMINAL_STATUSES,
    _result_path,
    _validate_existing,
    run_row,
)

DEFAULT_OUTPUT_ROOT = ROOT / "diagnostics_output" / "pact_place_v102_phase0"
SCENE_XML = ROOT / SCENE_XML_RELATIVE


def load_approval(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise PermissionError(f"missing owner approval file: {path}")
    document = json.loads(path.read_text())
    stored = document.get("artifact_sha256")
    payload = dict(document)
    payload.pop("artifact_sha256", None)
    if stored and stored != sha256_payload(payload):
        raise PermissionError("human_approval.json self-hash mismatch")
    return document


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--preflight-sha256", required=True)
    parser.add_argument("--screen-manifest-sha256", required=True)
    parser.add_argument("--review-manifest-sha256", required=True)
    parser.add_argument("--causal-artifact-sha256", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 12:
        raise SystemExit("workers must be in [1, 12]")
    establish_v10_runtime_env()
    contract = build_contract()
    approval = load_approval(args.approval.resolve())
    assert_phase0_approval(
        approval,
        contract_sha256=contract["contract_sha256"],
        preflight_sha256=args.preflight_sha256,
        screen_manifest_sha256=args.screen_manifest_sha256,
        review_manifest_sha256=args.review_manifest_sha256,
        causal_artifact_sha256=args.causal_artifact_sha256,
    )
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rows = list(contract["gate_rows"])
    if len(rows) != N_GATE_ROWS:
        raise RuntimeError("gate contract must contain exactly 24 rows")
    manifest = {
        "schema_version": "pact_place_v102_phase0_manifest_v1",
        "frozen_before_row_0": True,
        "contract_version": CONTRACT_VERSION,
        "environment_version": ENVIRONMENT_VERSION,
        "contract_sha256": contract["contract_sha256"],
        "preflight_sha256": args.preflight_sha256,
        "screen_manifest_sha256": args.screen_manifest_sha256,
        "review_manifest_sha256": args.review_manifest_sha256,
        "causal_artifact_sha256": args.causal_artifact_sha256,
        "approval_sha256": approval.get("artifact_sha256"),
        "gate_stream": GATE_STREAM,
        "gate_master_seed": GATE_MASTER_SEED,
        "n_rows": N_GATE_ROWS,
        "rows": rows,
        **empty_authorization(),
    }
    write_immutable(output_root / "phase0_manifest.json", manifest)
    config = {
        "schema_version": "pact_place_v102_phase0_config_v1",
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": contract["contract_sha256"],
        "implementation_sha256": implementation_sha256(),
        "scene_xml": SCENE_XML_RELATIVE,
        "scene_sha256": PLACE_V10_SCENE_SHA256,
        "sampler_class": SAMPLER_CLASS,
        "expert_screen_rows": rows,
        **empty_authorization(),
    }
    config["config_sha256"] = sha256_payload(config)
    (output_root / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n"
    )
    results: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for row in rows:
        existing_path = _result_path(output_root, row)
        if existing_path.is_file():
            existing = json.loads(existing_path.read_text())
            if existing.get("status") in TERMINAL_STATUSES:
                kept = _validate_existing(existing_path, row, config["config_sha256"])
                if kept is None:
                    raise RuntimeError(
                        f"refusing to replace terminal gate row {row['role_index']}"
                    )
                results.append(kept)
                continue
        pending.append(row)
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers, mp_context=context, max_tasks_per_child=1
    ) as executor:
        futures = {
            executor.submit(
                run_row,
                row,
                config_sha256=config["config_sha256"],
                output_root=str(output_root),
                scene_xml=str(SCENE_XML),
            ): row
            for row in pending
        }
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda item: int(item["role_index"]))
    eligibility = gate_eligibility(rows, results)
    document = {
        "schema_version": "pact_place_v102_phase0_result_v1",
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": contract["contract_sha256"],
        "eligibility": eligibility,
        "rows": [
            {
                "role_index": item["role_index"],
                "status": item["status"],
                "v102_clean_success": is_v102_clean_success(item),
                "v102_defects": row_defects(item),
            }
            for item in results
        ],
        **empty_authorization(),
        "phase0_passed": bool(eligibility["phase0_passed"]),
        "eligible_for_separate_collection_authorization": bool(
            eligibility["phase0_passed"]
        ),
        "authorizes_collection": False,
        "authorizes_training": False,
        "authorizes_evaluation": False,
        "permanent_stop": not bool(eligibility["phase0_passed"]),
    }
    write_immutable(output_root / "phase0_result.json", document)
    if not eligibility["phase0_passed"]:
        write_immutable(output_root / "phase0_stop.json", document)
    print(json.dumps(eligibility, indent=2))
    return 0 if eligibility["phase0_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
