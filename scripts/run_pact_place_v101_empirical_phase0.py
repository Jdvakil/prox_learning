#!/usr/bin/env python3
"""V10.1 24-row Phase-0 gate. Refuses to run without bound owner approval."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "scripts", ROOT / "submodules" / "molmospaces"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pact_place_v10_runtime import establish_v10_runtime_env, write_immutable  # noqa: E402
from pact_place_v101_empirical_qualification_contract import (  # noqa: E402
    CONTRACT_VERSION,
    GATE_MASTER_SEED,
    GATE_STREAM,
    N_GATE_ROWS,
    PHYSICS_CLEAN_FAMILIES,
    PLACE_V10_SCENE_SHA256,
    SAMPLER_CLASS,
    SCENE_XML_RELATIVE,
    assert_phase0_approval,
    build_contract,
    cell_key,
    empty_authorization,
    implementation_sha256,
    is_v101_clean_success,
    sha256_payload,
)
from run_pact_place_expert_screen import (  # noqa: E402
    TERMINAL_STATUSES,
    _result_path,
    _validate_existing,
    run_row,
)

DEFAULT_OUTPUT_ROOT = ROOT / "diagnostics_output" / "pact_place_v101_empirical_phase0"
SCENE_XML = ROOT / SCENE_XML_RELATIVE
MIN_GATE_CLEAN = 20
MIN_CELL_CLEAN = 1


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


def gate_eligibility(rows: list[dict[str, Any]], results: list[dict[str, Any]]) -> dict[str, Any]:
    by_episode = {str(item["episode_id"]): item for item in results}
    failures: list[dict[str, Any]] = []
    reconciled = len(rows) == N_GATE_ROWS and len(results) == N_GATE_ROWS
    if not reconciled:
        failures.append(
            {"code": "row_count", "rows": len(rows), "results": len(results)}
        )
    infrastructure = 0
    n_clean = 0
    clean_by_cell: dict[tuple[str, str], int] = {
        (family, side): 0
        for family in PHYSICS_CLEAN_FAMILIES
        for side in ("left", "right")
    }
    for row in rows:
        result = by_episode.get(str(row["episode_id"]))
        if result is None:
            reconciled = False
            failures.append({"code": "missing_result", "role_index": row["role_index"]})
            continue
        if result.get("row_sha256") != row.get("row_sha256"):
            reconciled = False
            failures.append({"code": "row_sha_mismatch", "role_index": row["role_index"]})
        if result.get("status") not in TERMINAL_STATUSES:
            failures.append({"code": "nonterminal", "role_index": row["role_index"]})
        if result.get("status") == "infrastructure_failure":
            infrastructure += 1
            failures.append({"code": "infrastructure_failure", "role_index": row["role_index"]})
        if is_v101_clean_success(result):
            n_clean += 1
            key = cell_key(str(row["layout_family_id"]), str(row["intrusion_side"]))
            clean_by_cell[key] += 1
    cell_failures = [
        {"code": "cell_clean_shortfall", "family": family, "side": side, "clean": count}
        for (family, side), count in sorted(clean_by_cell.items())
        if count < MIN_CELL_CLEAN
    ]
    failures.extend(cell_failures)
    passed = bool(
        reconciled
        and infrastructure == 0
        and n_clean >= MIN_GATE_CLEAN
        and not cell_failures
    )
    return {
        "reconciled": reconciled,
        "infrastructure_failures": infrastructure,
        "clean_successes": n_clean,
        "min_clean_successes": MIN_GATE_CLEAN,
        "clean_by_cell": {
            f"{family}:{side}": count for (family, side), count in sorted(clean_by_cell.items())
        },
        "phase0_passed": passed,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--approval", type=Path, required=True)
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
        review_manifest_sha256=args.review_manifest_sha256,
        causal_artifact_sha256=args.causal_artifact_sha256,
        contract_sha256=contract["contract_sha256"],
    )
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rows = list(contract["gate_rows"])
    if len(rows) != N_GATE_ROWS:
        raise RuntimeError("gate contract must contain exactly 24 rows")
    manifest = {
        "schema_version": "pact_place_v101_empirical_phase0_manifest_v1",
        "frozen_before_row_0": True,
        "contract_sha256": contract["contract_sha256"],
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
        "schema_version": "pact_place_v101_empirical_phase0_config_v1",
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
    pending = []
    for row in rows:
        existing_path = _result_path(output_root, row)
        if existing_path.is_file():
            existing = json.loads(existing_path.read_text())
            if existing.get("status") in TERMINAL_STATUSES:
                kept = _validate_existing(existing_path, row, config["config_sha256"])
                if kept is None:
                    raise RuntimeError(f"refusing to replace terminal gate row {row['role_index']}")
                results.append(kept)
                continue
        pending.append(row)
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers,
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
            for row in pending
        }
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda item: int(item["role_index"]))
    eligibility = gate_eligibility(rows, results)
    document = {
        "schema_version": "pact_place_v101_empirical_phase0_result_v1",
        "contract_sha256": contract["contract_sha256"],
        "eligibility": eligibility,
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
