#!/usr/bin/env python3
"""Audit the fresh remediation-v2 expert development attempts."""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from pact_collision_contract import load_manifest, rows_for_role
from pact_gate_statistics import wilson_interval

TERMINAL = {"success", "task_failure", "sampling_failure", "infrastructure_failure"}


def _clean(result: dict[str, Any]) -> bool:
    totals = result.get("contact_audit", {}).get("contact_class_totals", {})
    return bool(
        result.get("task_success")
        and int(totals.get("hazard_bar", 0)) == 0
        and int(totals.get("other_environment", 0)) == 0
    )


def analyze(manifest: dict[str, Any], collection: Path) -> dict[str, Any]:
    rows = rows_for_role(manifest, "development")
    results = []
    for row in rows:
        path = collection / "rows" / row["episode_id"] / "result.json"
        if not path.exists():
            raise RuntimeError(f"unreconciled development row: {path}")
        result = json.loads(path.read_text())
        if (
            result.get("row_sha256") != row["row_sha256"]
            or result.get("episode_id") != row["episode_id"]
        ):
            raise RuntimeError(f"development row identity mismatch: {path}")
        if result.get("status") not in TERMINAL:
            raise RuntimeError(f"nonterminal development result: {path}")
        results.append(result)

    attempts = len(results)
    status_counts = dict(collections.Counter(result["status"] for result in results))
    no_outcome = sum(
        result["status"] in ("sampling_failure", "infrastructure_failure")
        for result in results
    )
    clean = sum(_clean(result) for result in results)
    ordinary = sum(bool(result.get("task_success")) for result in results)
    clean_floor = 24
    no_outcome_rate = no_outcome / attempts
    clean_rate = clean / attempts
    return {
        "schema_version": "pact_expert_development_audit_v2",
        "manifest_sha256": manifest["manifest_sha256"],
        "old_v1_rows_used": False,
        "attempts": attempts,
        "status_counts": status_counts,
        "ordinary_task_success": ordinary,
        "ordinary_task_success_rate": ordinary / attempts,
        "ordinary_task_success_wilson_95": list(
            wilson_interval(ordinary, attempts)
        ),
        "usable_clean_demonstrations": clean,
        "usable_clean_demo_rate": clean_rate,
        "usable_clean_demo_wilson_95": list(wilson_interval(clean, attempts)),
        "development_readiness_clean_demo_floor": clean_floor,
        "development_readiness_clean_demo_floor_met": clean >= clean_floor,
        "no_scientific_outcome": no_outcome,
        "no_scientific_outcome_rate": no_outcome_rate,
        "no_scientific_outcome_wilson_95": list(
            wilson_interval(no_outcome, attempts)
        ),
        "infrastructure_scale_target_strictly_below_5_percent": (
            no_outcome_rate < 0.05
        ),
        "ready_to_freeze_and_collect_pilot": (
            clean >= clean_floor and no_outcome_rate < 0.05
        ),
        "contacting_success_rows": [
            {
                "role_index": int(result["role_index"]),
                "episode_id": result["episode_id"],
                "contact_class_totals": result.get("contact_audit", {}).get(
                    "contact_class_totals", {}
                ),
            }
            for result in results
            if result.get("task_success") and not _clean(result)
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--collection", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = analyze(load_manifest(args.manifest), args.collection)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "clean": report["usable_clean_demonstrations"],
        "attempts": report["attempts"],
        "no_outcome": report["no_scientific_outcome"],
        "ready": report["ready_to_freeze_and_collect_pilot"],
    }, sort_keys=True))
    return 0 if report["ready_to_freeze_and_collect_pilot"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
