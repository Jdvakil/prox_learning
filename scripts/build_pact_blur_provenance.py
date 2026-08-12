#!/usr/bin/env python3
"""Record immutable provenance for the calibrated inference-time blur sweep."""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
from pathlib import Path
from typing import Any

from pact_blur_sweep_contract import sha256_file, sha256_payload


ROOT = Path(__file__).resolve().parents[1]
ACT = ROOT / "submodules/act"
UPSTREAM = Path("/root/prox_learning/submodules/act")
UPSTREAM_COMMIT = "ec447930e1d025fed549ef2f58354aa87001c28c"
EXECUTION_AMENDMENT = (
    ROOT
    / "diagnostics_output/pact_blur_sweep/execution_recovery_amendment.json"
)


def git(path: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def function_source(path: Path, name: str) -> str:
    source = path.read_text()
    tree = ast.parse(source)
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == name
    )
    return "".join(source.splitlines(keepends=True)[node.lineno - 1 : node.end_lineno])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--schedule", required=True, type=Path)
    parser.add_argument("--dispatch", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    schedule = json.loads(args.schedule.read_text())
    dispatch = json.loads(args.dispatch.read_text())
    calibration = json.loads(args.calibration.read_text())
    copied = function_source(ACT / "pact_blur.py", "blur_images")
    upstream = subprocess.check_output(
        [
            "git",
            "-C",
            str(UPSTREAM),
            "show",
            f"{UPSTREAM_COMMIT}:imitate_episodes.py",
        ],
        text=True,
    )
    # Parse directly from the in-memory upstream source; no temporary is written.
    tree = ast.parse(upstream)
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == "blur_images"
    )
    upstream_function = "".join(
        upstream.splitlines(keepends=True)[node.lineno - 1 : node.end_lineno]
    )
    if copied != upstream_function:
        raise SystemExit("copied blur primitive differs from frozen upstream")
    document: dict[str, Any] = {
        "schema_version": "pact_blur_sweep_provenance_v1",
        "repository": {
            "path": str(ROOT),
            "branch": git(ROOT, "branch", "--show-current"),
            "head": git(ROOT, "rev-parse", "HEAD"),
            "act_submodule_head": git(ACT, "rev-parse", "HEAD"),
            "no_push_performed": True,
        },
        "upstream_blur_primitive": {
            "repository": str(UPSTREAM),
            "commit": UPSTREAM_COMMIT,
            "source": "imitate_episodes.py:blur_images",
            "function_source_sha256": sha256_payload(upstream_function),
            "copied_source_exact_match": True,
            "upstream_branch_merged": False,
        },
        "frozen_documents": {
            "manifest": {
                "path": str(args.manifest.resolve()),
                "file_sha256": sha256_file(args.manifest),
                "self_sha256": manifest["manifest_sha256"],
            },
            "schedule": {
                "path": str(args.schedule.resolve()),
                "file_sha256": sha256_file(args.schedule),
                "self_sha256": schedule["schedule_sha256"],
            },
            "dispatch": {
                "path": str(args.dispatch.resolve()),
                "file_sha256": sha256_file(args.dispatch),
                "self_sha256": dispatch["dispatch_contract_sha256"],
            },
            "calibration": {
                "path": str(args.calibration.resolve()),
                "file_sha256": sha256_file(args.calibration),
                "self_sha256": calibration["calibration_sha256"],
            },
        },
        "scientific_design": schedule["analysis_contract"],
        "decision_rule": schedule["decision_rule"],
        "calibrated_sigmas": schedule["blur_sigmas"],
        "rollouts": schedule["rollouts"],
        "workers": schedule["workers"],
        "no_retraining": True,
        "no_new_encoder": True,
        "no_new_demonstrations": True,
        "protected_scientific_artifacts": manifest[
            "protected_scientific_artifacts"
        ],
        "no_upstream_merge_model_construction_hashes": manifest[
            "no_upstream_merge"
        ],
        "checkpoint_records": manifest["frozen_artifacts"]["checkpoints"],
        "surface_encoder": manifest["frozen_artifacts"]["surface_encoder"],
        "token_plan": manifest["frozen_artifacts"]["token_plan"],
    }
    if EXECUTION_AMENDMENT.exists():
        amendment = json.loads(EXECUTION_AMENDMENT.read_text())
        if amendment["schedule_sha256"] != schedule["schedule_sha256"]:
            raise SystemExit("execution amendment belongs to a different schedule")
        if amendment["dispatch_contract_sha256"] != dispatch["dispatch_contract_sha256"]:
            raise SystemExit("execution amendment belongs to a different dispatch")
        document["execution_recovery_amendment"] = {
            "path": str(EXECUTION_AMENDMENT),
            "file_sha256": sha256_file(EXECUTION_AMENDMENT),
            "self_sha256": amendment["amendment_sha256"],
            "schedule_changed": amendment["schedule_changed"],
            "endpoint_fields_changed": amendment["endpoint_fields_changed"],
            "worker_count_changed": amendment["worker_count_changed"],
            "all_inflight_rows_rerun": amendment["recovery"][
                "all_inflight_rows_rerun"
            ],
            "completed_rows_preserved": amendment["recovery"][
                "completed_rows_preserved"
            ],
        }
    document["provenance_sha256"] = sha256_payload(document)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(document["provenance_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
