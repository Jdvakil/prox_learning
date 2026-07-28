#!/usr/bin/env python3
"""Adjudicate resumed Gate B/C while carrying settled expert/surface evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_pact_environment_gate import _pilot_act_results, _status_ledger
from pact_collision_contract import load_manifest
from pact_gate_statistics import (
    environment_decision,
    gate_b_core,
    gate_c_core,
    one_outcome_robust_classification,
    wilson_interval,
)

EXPECTED_SCHEDULE_SHA256 = (
    "e0515adf10a12cca22412d349d37b56ec5400446894b450b0e84edbe139b564e"
)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_self_hash(document: dict[str, Any], key: str) -> str:
    payload = dict(document)
    observed = payload.pop(key)
    if canonical_hash(payload) != observed:
        raise ValueError(f"{key} mismatch")
    return observed


def recovery_record(
    *,
    contract: dict[str, Any],
    execution: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    if (
        execution.get("schedule_sha256") != EXPECTED_SCHEDULE_SHA256
        or execution.get("dispatch_contract_sha256")
        != contract["dispatch_contract_sha256"]
        or execution.get("expected") != 64
        or not execution.get("terminal_ledger_reconciled")
    ):
        raise ValueError("repaired dispatch did not reconcile its frozen ledger")
    smoke_path = output_root / contract["launch_smoke"]["required_artifact"]
    smoke = json.loads(smoke_path.read_text())
    validate_self_hash(smoke, "launch_smoke_sha256")
    if (
        not smoke.get("passed")
        or smoke.get("rollout_id") != contract["launch_smoke"]["rollout_id"]
        or smoke.get("smoke_invocations") != 1
    ):
        raise ValueError("launch smoke did not pass exactly once")

    drivers = [
        json.loads(path.read_text())
        for path in sorted(output_root.glob("rows/*/driver_result.json"))
    ]
    if len(drivers) != 64:
        raise ValueError("repaired dispatch does not contain 64 terminal drivers")
    return {
        "dispatch_contract_sha256": contract["dispatch_contract_sha256"],
        "prior_failed_dispatch": contract["prior_failed_dispatch"],
        "retry_justification": contract["retry_justification"],
        "launch_smoke": {
            "rollout_id": smoke["rollout_id"],
            "schedule_index": smoke["schedule_index"],
            "launch_smoke_sha256": smoke["launch_smoke_sha256"],
            "scientific_result_sha256": smoke["scientific_result_sha256"],
            "passed": True,
            "smoke_invocations": 1,
        },
        "repaired_dispatch": {
            "output_root": str(output_root),
            "execution_summary_sha256": file_hash(
                output_root / "execution_summary.json"
            ),
            "terminal_rows": len(drivers),
            "scientific_results": sum(
                driver.get("status") == "complete" for driver in drivers
            ),
            "post_boundary_failures": sum(
                driver.get("status") == "post_boundary_failure"
                for driver in drivers
            ),
            "pre_observation_infrastructure_failures": sum(
                int(driver.get("pre_observation_infrastructure_failures", 0))
                for driver in drivers
            ),
            "rows_with_multiple_attempts": sum(
                int(driver.get("attempt_count", 0)) > 1 for driver in drivers
            ),
            "scientific_rows_rerun": 0,
        },
    }


def analyze(
    *,
    manifest: dict[str, Any],
    prior_gate: dict[str, Any],
    act_results: list[dict[str, Any]],
    schedule: dict[str, Any],
    recovery: dict[str, Any],
    prior_gate_sha256: str,
) -> dict[str, Any]:
    if (
        prior_gate.get("schema_version") != "pact_environment_gate_v2"
        or prior_gate.get("manifest_sha256") != manifest["manifest_sha256"]
        or prior_gate["surface_observability"]["robust_classification"]
        != "adequate"
        or prior_gate["expert"]["usable_clean_demonstrations"] != 58
        or prior_gate["expert"]["no_scientific_outcome"] != 2
    ):
        raise ValueError("settled Phase 1 evidence does not match handoff 3")
    if schedule["schedule_sha256"] != EXPECTED_SCHEDULE_SHA256:
        raise ValueError("resumed analysis changed the scientific schedule")

    act_ledger = _status_ledger(act_results)
    scientific = [
        result for result in act_results if result["status"] == "scientific_outcome"
    ]
    n = len(scientific)
    primary = sum(
        bool(result["collision_free_task_success"]) for result in scientific
    )
    task_success = sum(bool(result["task_success"]) for result in scientific)
    hazard = sum(
        int(result["contact_audit"]["contact_class_totals"].get("hazard_bar", 0))
        > 0
        for result in scientific
    )
    other = sum(
        int(
            result["contact_audit"]["contact_class_totals"].get(
                "other_environment", 0
            )
        )
        > 0
        for result in scientific
    )
    gate_b = one_outcome_robust_classification(primary, n, gate_b_core)
    gate_c = one_outcome_robust_classification(hazard, n, gate_c_core)
    minimum_rows = n >= 61
    infrastructure_progression = bool(
        prior_gate["expert"]["progression_target_strictly_below_5_percent"]
        and act_ledger["progression_target_strictly_below_5_percent"]
    )
    decision = environment_decision(
        surface_classification="adequate",
        gate_b_classification=gate_b["robust_classification"],
        gate_c_classification=gate_c["robust_classification"],
        usable_demo_floor_met=prior_gate["expert"][
            "usable_clean_demo_floor_met"
        ],
        infrastructure_progression_met=infrastructure_progression,
        minimum_scientific_rows_met=minimum_rows,
    )
    return {
        "schema_version": "pact_environment_gate_v2",
        "route": "collision",
        "gate_a_applicable": False,
        "manifest_sha256": manifest["manifest_sha256"],
        "pilot_schedule_sha256": schedule["schedule_sha256"],
        "old_v1_rows_used": False,
        "settled_phase1_carried_forward_without_remeasurement": {
            "prior_gate_sha256": prior_gate_sha256,
            "expert": True,
            "surface_observability": True,
        },
        "expert": prior_gate["expert"],
        "surface_observability": prior_gate["surface_observability"],
        "act": {
            **act_ledger,
            "scientific_outcomes": n,
            "minimum_scientific_rows": 61,
            "minimum_scientific_rows_met": minimum_rows,
            "ordinary_task_success": task_success,
            "ordinary_task_success_rate": task_success / n if n else None,
            "ordinary_task_success_wilson_95": (
                list(wilson_interval(task_success, n)) if n else None
            ),
            "collision_free_task_success": primary,
            "episodes_with_hazard_bar_contact": hazard,
            "episodes_with_other_environment_contact": other,
            "failure_taxonomy": dict(
                Counter(result["failure_taxonomy"] for result in scientific)
            ),
            "contact_entry_totals": {
                contact_class: sum(
                    int(
                        result["contact_audit"]["contact_class_totals"].get(
                            contact_class, 0
                        )
                    )
                    for result in scientific
                )
                for contact_class in (
                    "grasp_target",
                    "hazard_bar",
                    "other_environment",
                )
            },
        },
        "gate_b": gate_b,
        "gate_c": gate_c,
        "infrastructure_progression_met": infrastructure_progression,
        "infrastructure_recovery": recovery,
        "science_gate_classifications": {
            "surface_observability": "adequate",
            "gate_b": gate_b["robust_classification"],
            "gate_c": gate_c["robust_classification"],
        },
        "all_applicable_gates_pass": decision == "PACT_ENVIRONMENT_ADEQUATE",
        "decision": decision,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--prior-gate", required=True, type=Path)
    parser.add_argument("--pilot-schedule", required=True, type=Path)
    parser.add_argument("--dispatch-contract", required=True, type=Path)
    parser.add_argument("--pilot-output-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    prior_gate = json.loads(args.prior_gate.read_text())
    act_results, schedule = _pilot_act_results(
        args.pilot_schedule, args.pilot_output_root
    )
    contract = json.loads(args.dispatch_contract.read_text())
    validate_self_hash(contract, "dispatch_contract_sha256")
    execution = json.loads(
        (args.pilot_output_root / "execution_summary.json").read_text()
    )
    recovery = recovery_record(
        contract=contract,
        execution=execution,
        output_root=args.pilot_output_root,
    )
    result = analyze(
        manifest=manifest,
        prior_gate=prior_gate,
        act_results=act_results,
        schedule=schedule,
        recovery=recovery,
        prior_gate_sha256=file_hash(args.prior_gate),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(result["decision"])
    return 0 if result["all_applicable_gates_pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
